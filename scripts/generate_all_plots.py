#!/usr/bin/env python3
"""Generate all presentation figures from saved trajectory data.

Usage
-----
# Headless (default) — save everything, don't open windows:
    uv run python scripts/generate_all_plots.py --no-show

# Interactive spot-check — open each figure as it is produced:
    uv run python scripts/generate_all_plots.py --show

Missing-data combos are skipped gracefully (printed as [SKIP]), so the
script can be re-run incrementally as more trajectory files arrive.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from forage_rl.types import Trajectory
from forage_rl.agents.registry import Agent, PolicySpec, agent_display_label
from forage_rl.analysis import (
    oracle_patch_optimal_prt_by_state,
    patch_exit_action_indices,
    recovery_auc,
    within_episode_boundary_window_recovery_curve_for_trajectory,
    within_episode_recovery_curve_for_trajectory,
    within_episode_signed_recovery_curve_for_trajectory,
)
from forage_rl.analysis.patch_timing import infer_hidden_states_for_trajectory
from forage_rl.config import FIGURES_DIR
from forage_rl.environments import load_builtin_maze_spec
from forage_rl.environments.maze import Maze, maze_from_builtin_maze_spec
from forage_rl.utils import list_run_dataset_run_ids, load_run_dataset
from forage_rl.visualization import (
    plot_aggregate_comparison,
    plot_aggregate_trajectory_stats,
    plot_boundary_window_recovery_comparison,
    plot_episode_return_comparison,
    plot_patch_timing_summary,
    plot_recovery_curve_comparison,
    plot_recovery_heatmap,
    plot_recovery_heatmap_delta,
    plot_signed_recovery_curve_comparison,
    plot_single_run_stats,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TABULAR_AGENTS: list[Agent] = [
    Agent.QLearning,
    Agent.SRTD,
    Agent.SRMB,
    Agent.SRDyna,
    Agent.MBRL,
]

NEURAL_POLICIES: list[PolicySpec] = [
    PolicySpec(agent=Agent.DQN, context_mode="prev_reward"),
    PolicySpec(agent=Agent.ELMAN, context_mode="prev_reward"),
    PolicySpec(agent=Agent.GRU, context_mode="prev_reward"),
    PolicySpec(agent=Agent.LSTM, context_mode="prev_reward"),
]

PERTURBATION_SINGLE_POLICY = PolicySpec(
    agent=Agent.GRU, context_mode="prev_reward"
)

REVALUATION_SINGLE_POLICIES: list[Agent | PolicySpec] = [
    Agent.QLearning,
    PolicySpec(agent=Agent.DQN, context_mode="prev_reward"),
    PolicySpec(agent=Agent.LSTM, context_mode="prev_reward"),
    Agent.MBRL,
    Agent.SRDyna,
]

PLOTTED_POLICIES: list[Agent | PolicySpec] = [*TABULAR_AGENTS, *NEURAL_POLICIES]

# Heatmaps follow the MF-to-MB taxonomy order used in the manuscript.
HEATMAP_POLICIES: list[Agent | PolicySpec] = [
    Agent.QLearning,
    PolicySpec(agent=Agent.DQN, context_mode="prev_reward"),
    PolicySpec(agent=Agent.LSTM, context_mode="prev_reward"),
    Agent.SRTD,
    Agent.SRDyna,
    Agent.SRMB,
    Agent.MBRL,
]
HEATMAP_POLICY_LABELS: dict[Agent | PolicySpec, str] = {
    Agent.QLearning: "Q-Learning",
    PolicySpec(agent=Agent.DQN, context_mode="prev_reward"): "DQN",
    PolicySpec(agent=Agent.LSTM, context_mode="prev_reward"): "DRQN",
    Agent.SRTD: "SR-MF (SR-TD)",
    Agent.SRDyna: "Dyna-Q",
    Agent.SRMB: "SR-MB",
    Agent.MBRL: "Model-Based",
}
HEATMAP_GROUP_BOUNDARIES: tuple[int, ...] = (3, 4, 6)

PERTURBATION_MAZES: list[str] = [
    "full_one_way_perturbed_latent_learning",
    "full_one_way_perturbed_detour",
    "full_one_way_perturbed_revaluation",
]

PERTURBATION_LABELS: dict[str, str] = {
    "full_one_way_perturbed_latent_learning": "Latent\nLearning",
    "full_one_way_perturbed_detour": "Detour",
    "full_one_way_perturbed_revaluation": "Revaluation",
}

OBSERVABILITY_CONDITIONS: list[bool] = [True, False]
BASELINE_MAZE: str = "full_one_way"
HORIZON: int = 1000
PERTURBATION_T: int = 500
AUC_WINDOW: int = 50

# ---------------------------------------------------------------------------
# Output subdirectories
# ---------------------------------------------------------------------------


def _make_dirs() -> dict[str, Path]:
    dirs = {
        "baseline": FIGURES_DIR / "01_baseline",
        "recovery_fo": FIGURES_DIR / "02_recovery_curves" / "FO",
        "recovery_po": FIGURES_DIR / "02_recovery_curves" / "PO",
        "heatmap": FIGURES_DIR / "03_heatmap",
        "comparisons_fo": FIGURES_DIR / "04_comparisons" / "FO",
        "comparisons_po": FIGURES_DIR / "04_comparisons" / "PO",
        "patch_timing": FIGURES_DIR / "05_patch_timing",
        "single": FIGURES_DIR / "06_single",
        "new": FIGURES_DIR / "07_new",
        "average": FIGURES_DIR / "08_folder",
        "perbsingle": FIGURES_DIR / "09_perbsingle",
        "new_single": FIGURES_DIR / "11_new",
        "newer_single": FIGURES_DIR / "12_newer",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ---------------------------------------------------------------------------
# Recovery-curve helper
# ---------------------------------------------------------------------------


def _build_patch_labels(maze_name: str, observable: bool) -> dict[int, str]:
    """Build patch_labels dict appropriate for the observability mode.

    - FO: maps true state id → state label (one per state)
    - PO: maps observation group id → observation label (from maze.observation_labels)
    """
    spec = load_builtin_maze_spec(maze_name)
    if observable:
        return {int(s.id): s.label for s in spec.states}
    else:
        return {i: label for i, label in enumerate(spec.maze.observation_labels)}


def _policy_display_label(policy: Agent | PolicySpec) -> str:
    if isinstance(policy, PolicySpec):
        return policy.display_label
    return agent_display_label(policy)


def _run_ids_for(
    policy: Agent | PolicySpec,
    maze_name: str,
    observable: bool,
    horizon: int = HORIZON,
) -> list[int]:
    if isinstance(policy, PolicySpec):
        return list_run_dataset_run_ids(
            policy.agent,
            maze_name,
            observable,
            context_mode=policy.context_mode,
            horizon=horizon,
        )
    return list_run_dataset_run_ids(policy, maze_name, observable, horizon=horizon)


def _load_run_dataset_for(
    policy: Agent | PolicySpec,
    run_id: int,
    maze_name: str,
    observable: bool,
    horizon: int = HORIZON,
):
    if isinstance(policy, PolicySpec):
        return load_run_dataset(
            policy.agent,
            run_id,
            maze_name,
            observable,
            context_mode=policy.context_mode,
            horizon=horizon,
        )
    return load_run_dataset(policy, run_id, maze_name, observable, horizon=horizon)


def _post_perturbation_benchmark_prt(
    maze_name: str,
    horizon: int = HORIZON,
) -> dict[int, int]:
    """Return the FO oracle PRT benchmark for the post-perturbation maze."""
    spec = load_builtin_maze_spec(maze_name)
    if spec.perturbation is not None:
        spec = spec.perturbed()
    maze = Maze(spec, seed=0, horizon=horizon, observable=True)
    return oracle_patch_optimal_prt_by_state(maze)


def _flatten_run_dataset(run_dataset) -> Trajectory:
    return Trajectory(transitions=list(run_dataset.iter_transitions()))


def _infer_piecewise_hidden_states(
    trajectory: Trajectory,
    maze_name: str,
    horizon: int,
    perturbation_t: int,
) -> list[int]:
    """Infer PO true states with pre/post maze dynamics split at perturbation_t."""
    transitions = trajectory.transitions
    pre_transitions = transitions[:perturbation_t]
    post_transitions = transitions[perturbation_t:]
    resolved_states: list[int] = []

    if pre_transitions:
        pre_maze = maze_from_builtin_maze_spec(
            maze_name, observable=True, horizon=horizon
        )
        resolved_states.extend(
            infer_hidden_states_for_trajectory(
                Trajectory(transitions=pre_transitions),
                maze=pre_maze,
            )
        )

    if post_transitions:
        post_spec = load_builtin_maze_spec(maze_name)
        if post_spec.perturbation is not None:
            post_spec = post_spec.perturbed()
        post_maze = Maze(post_spec, seed=0, horizon=horizon, observable=True)
        resolved_states.extend(
            infer_hidden_states_for_trajectory(
                Trajectory(transitions=post_transitions),
                maze=post_maze,
            )
        )

    return resolved_states


def _resolved_states_for_recovery(
    trajectory: Trajectory,
    maze_name: str,
    observable: bool,
    horizon: int,
    perturbation_t: int,
) -> list[int] | None:
    if observable:
        return None
    return _infer_piecewise_hidden_states(
        trajectory,
        maze_name=maze_name,
        horizon=horizon,
        perturbation_t=perturbation_t,
    )


def _compute_recovery_curves(
    agents: list[Agent | PolicySpec],
    maze_name: str,
    observable: bool,
    horizon: int = HORIZON,
    perturbation_t: int = PERTURBATION_T,
) -> dict[Agent | PolicySpec, list[np.ndarray]]:
    """Load saved runs and compute a within-episode recovery curve per run.

    Benchmark PRTs come from the post-perturbation FO oracle so the recovery
    curve measures adaptation to the changed environment.
    """
    maze = maze_from_builtin_maze_spec(maze_name, observable=True, horizon=horizon)
    patch_labels = _build_patch_labels(maze_name, observable)
    exit_actions = patch_exit_action_indices(maze)
    benchmark_prt = _post_perturbation_benchmark_prt(maze_name, horizon=horizon)

    curves_by_agent: dict[Agent | PolicySpec, list[np.ndarray]] = {}
    for agent in agents:
        run_ids = _run_ids_for(agent, maze_name, observable, horizon=horizon)
        if not run_ids:
            print(
                f"  [SKIP] No data: {_policy_display_label(agent)} / {maze_name} / "
                f"{'FO' if observable else 'PO'}"
            )
            continue
        agent_curves: list[np.ndarray] = []
        for run_id in run_ids:
            run_dataset = _load_run_dataset_for(
                agent, run_id, maze_name, observable, horizon=horizon
            )
            combined = _flatten_run_dataset(run_dataset)
            resolved_states = _resolved_states_for_recovery(
                combined,
                maze_name=maze_name,
                observable=observable,
                horizon=horizon,
                perturbation_t=perturbation_t,
            )
            curve = within_episode_recovery_curve_for_trajectory(
                combined,
                patch_labels=patch_labels,
                exit_actions=exit_actions,
                benchmark_prt_by_state=benchmark_prt,
                perturbation_timestep=perturbation_t,
                resolved_states=resolved_states,
            )
            agent_curves.append(curve)
        if agent_curves:
            curves_by_agent[agent] = agent_curves
    return curves_by_agent


def _compute_signed_recovery_curves(
    agents: list[Agent | PolicySpec],
    maze_name: str,
    observable: bool,
    horizon: int = HORIZON,
    perturbation_t: int = PERTURBATION_T,
) -> dict[Agent | PolicySpec, list[np.ndarray]]:
    """Same as _compute_recovery_curves but returns signed deviations."""
    maze = maze_from_builtin_maze_spec(maze_name, observable=True, horizon=horizon)
    patch_labels = _build_patch_labels(maze_name, observable)
    exit_actions = patch_exit_action_indices(maze)
    benchmark_prt = _post_perturbation_benchmark_prt(maze_name, horizon=horizon)

    curves_by_agent: dict[Agent | PolicySpec, list[np.ndarray]] = {}
    for agent in agents:
        run_ids = _run_ids_for(agent, maze_name, observable, horizon=horizon)
        if not run_ids:
            continue
        agent_curves: list[np.ndarray] = []
        for run_id in run_ids:
            run_dataset = _load_run_dataset_for(
                agent, run_id, maze_name, observable, horizon=horizon
            )
            combined = _flatten_run_dataset(run_dataset)
            resolved_states = _resolved_states_for_recovery(
                combined,
                maze_name=maze_name,
                observable=observable,
                horizon=horizon,
                perturbation_t=perturbation_t,
            )
            curve = within_episode_signed_recovery_curve_for_trajectory(
                combined,
                patch_labels=patch_labels,
                exit_actions=exit_actions,
                benchmark_prt_by_state=benchmark_prt,
                perturbation_timestep=perturbation_t,
                resolved_states=resolved_states,
            )
            agent_curves.append(curve)
        if agent_curves:
            curves_by_agent[agent] = agent_curves
    return curves_by_agent


def _compute_boundary_window_curves(
    agents: list[Agent | PolicySpec],
    maze_name: str,
    observable: bool,
    *,
    horizon: int = HORIZON,
    perturbation_t: int = PERTURBATION_T,
    boundary_window: int = 100,
) -> dict[Agent | PolicySpec, list[np.ndarray]]:
    """Load saved runs and compute before/after boundary-window curves per run."""
    maze = maze_from_builtin_maze_spec(maze_name, observable=True, horizon=horizon)
    patch_labels = _build_patch_labels(maze_name, observable)
    exit_actions = patch_exit_action_indices(maze)
    benchmark_prt = _post_perturbation_benchmark_prt(maze_name, horizon=horizon)

    curves_by_agent: dict[Agent | PolicySpec, list[np.ndarray]] = {}
    for agent in agents:
        run_ids = _run_ids_for(agent, maze_name, observable, horizon=horizon)
        if not run_ids:
            print(
                f"  [SKIP] No boundary-window data: {_policy_display_label(agent)} / "
                f"{maze_name} / {'FO' if observable else 'PO'}"
            )
            continue
        agent_curves: list[np.ndarray] = []
        for run_id in run_ids:
            run_dataset = _load_run_dataset_for(
                agent, run_id, maze_name, observable, horizon=horizon
            )
            combined = _flatten_run_dataset(run_dataset)
            resolved_states = _resolved_states_for_recovery(
                combined,
                maze_name=maze_name,
                observable=observable,
                horizon=horizon,
                perturbation_t=perturbation_t,
            )
            curve = within_episode_boundary_window_recovery_curve_for_trajectory(
                combined,
                patch_labels=patch_labels,
                exit_actions=exit_actions,
                benchmark_prt_by_state=benchmark_prt,
                perturbation_timestep=perturbation_t,
                window=boundary_window,
                resolved_states=resolved_states,
            )
            agent_curves.append(curve)
        if agent_curves:
            curves_by_agent[agent] = agent_curves
    return curves_by_agent


def _mean_auc(
    curves_by_agent: dict[Agent | PolicySpec, list[np.ndarray]],
    window: int = AUC_WINDOW,
) -> dict[Agent | PolicySpec, float]:
    """Compute mean recovery AUC across runs for each agent."""
    result: dict[Agent | PolicySpec, float] = {}
    for agent, curves in curves_by_agent.items():
        aucs = [recovery_auc(c, window) for c in curves]
        finite = [a for a in aucs if not np.isnan(a)]
        result[agent] = float(np.mean(finite)) if finite else float("nan")
    return result


# ---------------------------------------------------------------------------
# Section 1 — Baseline
# ---------------------------------------------------------------------------


def section_1_baseline(dirs: dict[str, Path], show: bool) -> None:
    print("\n=== Section 1: Baseline learning ===")
    for observable in OBSERVABILITY_CONDITIONS:
        obs_tag = "FO" if observable else "PO"
        run_ids = _run_ids_for(PLOTTED_POLICIES[0], BASELINE_MAZE, observable, horizon=HORIZON)
        if not run_ids:
            print(f"  [SKIP] No baseline data for {obs_tag}")
            continue

        fp = dirs["baseline"] / f"episode_return_comparison_{BASELINE_MAZE}_{obs_tag}.png"
        print(f"  episode return comparison {obs_tag}")
        plot_episode_return_comparison(
            BASELINE_MAZE,
            observable=observable,
            agents=PLOTTED_POLICIES,
            horizon=HORIZON,
            save=True,
            show=show,
            filepath=fp,
        )

        for agent in PLOTTED_POLICIES:
            run_ids_agent = _run_ids_for(agent, BASELINE_MAZE, observable, horizon=HORIZON)
            if not run_ids_agent:
                print(f"  [SKIP] No data: {_policy_display_label(agent)} baseline {obs_tag}")
                continue
            agent_slug = agent.artifact_label if isinstance(agent, PolicySpec) else agent.value
            fp = dirs["baseline"] / f"trajectory_stats_{agent_slug}_{BASELINE_MAZE}_{obs_tag}.png"
            print(f"  trajectory stats: {_policy_display_label(agent)} {obs_tag}")
            plot_aggregate_trajectory_stats(
                agent,
                BASELINE_MAZE,
                observable=observable,
                horizon=HORIZON,
                save=True,
                show=show,
                filepath=fp,
            )


# ---------------------------------------------------------------------------
# Section 2 — Per-perturbation recovery curves
# ---------------------------------------------------------------------------


def section_2_recovery_curves(dirs: dict[str, Path], show: bool) -> None:
    print("\n=== Section 2: Recovery curves ===")
    for observable in OBSERVABILITY_CONDITIONS:
        obs_tag = "FO" if observable else "PO"
        out_dir = dirs[f"recovery_{obs_tag.lower()}"]

        for maze_name in PERTURBATION_MAZES:
            short = maze_name.replace("full_one_way_perturbed_", "")
            print(f"  {obs_tag} / {short}")

            curves = _compute_recovery_curves(
                PLOTTED_POLICIES, maze_name, observable
            )
            signed = _compute_signed_recovery_curves(
                PLOTTED_POLICIES, maze_name, observable
            )

            if not curves:
                print("    [SKIP] No recovery data")
                continue

            fp_abs = out_dir / f"recovery_{short}_{obs_tag}.png"
            plot_recovery_curve_comparison(
                curves,
                maze_name=maze_name,
                observable=observable,
                perturbation_label=short,
                save=True,
                show=show,
                filepath=fp_abs,
            )

            if signed:
                fp_signed = out_dir / f"recovery_signed_{short}_{obs_tag}.png"
                plot_signed_recovery_curve_comparison(
                    signed,
                    maze_name=maze_name,
                    observable=observable,
                    perturbation_label=short,
                    save=True,
                    show=show,
                    filepath=fp_signed,
                )


# ---------------------------------------------------------------------------
# Section 3 — Recovery AUC heatmap
# ---------------------------------------------------------------------------


def section_3_heatmap(dirs: dict[str, Path], show: bool) -> None:
    print("\n=== Section 3: Recovery AUC heatmaps ===")

    # Collect AUC matrices for FO and PO
    auc_fo: dict[Agent | PolicySpec, dict[str, float]] = {a: {} for a in HEATMAP_POLICIES}
    auc_po: dict[Agent | PolicySpec, dict[str, float]] = {a: {} for a in HEATMAP_POLICIES}

    for observable in OBSERVABILITY_CONDITIONS:
        store = auc_fo if observable else auc_po
        for maze_name in PERTURBATION_MAZES:
            curves = _compute_recovery_curves(HEATMAP_POLICIES, maze_name, observable)
            mean_aucs = _mean_auc(curves)
            for agent in HEATMAP_POLICIES:
                store[agent][maze_name] = mean_aucs.get(agent, float("nan"))

    fp_fo = dirs["heatmap"] / "recovery_heatmap_FO.png"
    print("  heatmap FO")
    plot_recovery_heatmap(
        auc_fo,
        PERTURBATION_LABELS,
        HEATMAP_POLICIES,
        observable=True,
        agent_labels=HEATMAP_POLICY_LABELS,
        column_group_boundaries=HEATMAP_GROUP_BOUNDARIES,
        save=True,
        show=show,
        filepath=fp_fo,
    )

    fp_po = dirs["heatmap"] / "recovery_heatmap_PO.png"
    print("  heatmap PO")
    plot_recovery_heatmap(
        auc_po,
        PERTURBATION_LABELS,
        HEATMAP_POLICIES,
        observable=False,
        agent_labels=HEATMAP_POLICY_LABELS,
        column_group_boundaries=HEATMAP_GROUP_BOUNDARIES,
        save=True,
        show=show,
        filepath=fp_po,
    )

    fp_delta = dirs["heatmap"] / "recovery_heatmap_fo_po_delta.png"
    print("  heatmap FO-PO delta")
    plot_recovery_heatmap_delta(
        auc_fo,
        auc_po,
        PERTURBATION_LABELS,
        HEATMAP_POLICIES,
        agent_labels=HEATMAP_POLICY_LABELS,
        column_group_boundaries=HEATMAP_GROUP_BOUNDARIES,
        save=True,
        show=show,
        filepath=fp_delta,
    )


# ---------------------------------------------------------------------------
# Section 4 — Algorithmic comparison plots
# ---------------------------------------------------------------------------


def section_4_comparisons(dirs: dict[str, Path], show: bool) -> None:
    print("\n=== Section 4: Algorithmic comparisons ===")
    pairs = [
        (Agent.QLearning, Agent.SRTD),
        (Agent.SRTD, Agent.SRMB),
        (Agent.SRMB, Agent.SRDyna),
    ]
    for observable in OBSERVABILITY_CONDITIONS:
        obs_tag = "FO" if observable else "PO"
        out_dir = dirs[f"comparisons_{obs_tag.lower()}"]
        for source, compare in pairs:
            src_slug = source.value
            cmp_slug = compare.value
            run_ids = list_run_dataset_run_ids(
                source, BASELINE_MAZE, observable, horizon=HORIZON
            )
            if not run_ids:
                print(
                    f"  [SKIP] No data: {agent_display_label(source)} {obs_tag}"
                )
                continue
            fp = out_dir / f"compare_{src_slug}_vs_{cmp_slug}_{obs_tag}.png"
            print(f"  {agent_display_label(source)} vs {agent_display_label(compare)} {obs_tag}")
            try:
                plot_aggregate_comparison(
                    source,
                    compare_to=[compare],
                    maze_name=BASELINE_MAZE,
                    observable=observable,
                    horizon=HORIZON,
                    save=True,
                    show=show,
                    filepath=fp,
                )
            except FileNotFoundError as e:
                print(f"    [SKIP] Missing logprobs data: {e.filename}")


# ---------------------------------------------------------------------------
# Section 5 — Patch timing summary
# ---------------------------------------------------------------------------


def section_5_patch_timing(dirs: dict[str, Path], show: bool) -> None:
    print("\n=== Section 5: Patch timing summaries ===")
    for observable in OBSERVABILITY_CONDITIONS:
        obs_tag = "FO" if observable else "PO"
        for agent in PLOTTED_POLICIES:
            run_ids = _run_ids_for(agent, BASELINE_MAZE, observable, horizon=HORIZON)
            if not run_ids:
                print(
                    f"  [SKIP] No data: {_policy_display_label(agent)} {obs_tag}"
                )
                continue
            agent_slug = agent.artifact_label if isinstance(agent, PolicySpec) else agent.value
            fp = dirs["patch_timing"] / f"patch_timing_{agent_slug}_{BASELINE_MAZE}_{obs_tag}.png"
            print(f"  {_policy_display_label(agent)} {obs_tag}")
            plot_patch_timing_summary(
                agent,
                BASELINE_MAZE,
                observable=observable,
                horizon=HORIZON,
                save=True,
                show=show,
                filepath=fp,
            )


# ---------------------------------------------------------------------------
# Section 6 — Single-run trajectory stats
# ---------------------------------------------------------------------------


def section_6_single_run_stats(dirs: dict[str, Path], show: bool) -> None:
    print("\n=== Section 6: Single-run trajectory stats ===")
    preferred_observability = [False, True]
    for agent in PLOTTED_POLICIES:
        selected_observable: bool | None = None
        for observable in preferred_observability:
            run_ids = _run_ids_for(agent, BASELINE_MAZE, observable, horizon=HORIZON)
            if run_ids:
                selected_observable = observable
                break
        if selected_observable is None:
            print(f"  [SKIP] No data: {_policy_display_label(agent)}")
            continue

        obs_tag = "FO" if selected_observable else "PO"
        agent_slug = (
            agent.artifact_label if isinstance(agent, PolicySpec) else agent.value
        )
        fp = dirs["single"] / f"single_run_{agent_slug}_{BASELINE_MAZE}_{obs_tag}.png"
        print(f"  {_policy_display_label(agent)} {obs_tag}")
        plot_single_run_stats(
            agent,
            BASELINE_MAZE,
            observable=selected_observable,
            horizon=HORIZON,
            save=True,
            show=show,
            filepath=fp,
        )


# ---------------------------------------------------------------------------
# Section 7 — Boundary-window bars
# ---------------------------------------------------------------------------


def section_7_boundary_window_bars(dirs: dict[str, Path], show: bool) -> None:
    print("\n=== Section 7: Boundary-window before/after bars ===")
    boundary_window = 100
    out_dir = dirs["new"]
    for observable in OBSERVABILITY_CONDITIONS:
        obs_tag = "FO" if observable else "PO"
        for maze_name in PERTURBATION_MAZES:
            short = maze_name.replace("full_one_way_perturbed_", "")
            print(f"  {obs_tag} / {short}")
            curves = _compute_boundary_window_curves(
                PLOTTED_POLICIES,
                maze_name,
                observable,
                boundary_window=boundary_window,
            )
            if not curves:
                print("    [SKIP] No boundary-window data")
                continue

            fp = out_dir / f"boundary_window_{short}_{obs_tag}.png"
            plot_boundary_window_recovery_comparison(
                curves,
                boundary_window=boundary_window,
                maze_name=maze_name,
                observable=observable,
                perturbation_label=short,
                save=True,
                show=show,
                filepath=fp,
            )


# ---------------------------------------------------------------------------
# Section 8 — Boundary-window averages
# ---------------------------------------------------------------------------


def section_8_boundary_window_averages(dirs: dict[str, Path], show: bool) -> None:
    print("\n=== Section 8: Boundary-window average bars ===")
    boundary_window = 100
    out_dir = dirs["average"]
    for observable in OBSERVABILITY_CONDITIONS:
        obs_tag = "FO" if observable else "PO"
        for maze_name in PERTURBATION_MAZES:
            short = maze_name.replace("full_one_way_perturbed_", "")
            print(f"  {obs_tag} / {short}")
            curves = _compute_boundary_window_curves(
                PLOTTED_POLICIES,
                maze_name,
                observable,
                boundary_window=boundary_window,
            )
            if not curves:
                print("    [SKIP] No boundary-window data")
                continue

            fp = out_dir / f"boundary_window_average_{short}_{obs_tag}.png"
            plot_boundary_window_recovery_comparison(
                curves,
                boundary_window=boundary_window,
                maze_name=maze_name,
                observable=observable,
                perturbation_label=short,
                save=True,
                show=show,
                filepath=fp,
                metric="average",
            )


# ---------------------------------------------------------------------------
# Section 9 — Perturbation single-run GRU PO
# ---------------------------------------------------------------------------


def section_9_perturbation_single_run_gru_po(
    dirs: dict[str, Path], show: bool
) -> None:
    print("\n=== Section 9: Perturbation single-run GRU PO ===")
    out_dir = dirs["perbsingle"]
    observable = False
    obs_tag = "PO"

    baseline_run_ids = _run_ids_for(
        PERTURBATION_SINGLE_POLICY,
        BASELINE_MAZE,
        observable,
        horizon=HORIZON,
    )
    if not baseline_run_ids:
        print(f"  [SKIP] No data: GRU {obs_tag} / one_way")
    else:
        baseline_fp = out_dir / f"perbsingle_gru_one_way_{obs_tag}.png"
        print(f"  GRU {obs_tag} / one_way")
        plot_single_run_stats(
            PERTURBATION_SINGLE_POLICY,
            BASELINE_MAZE,
            observable=observable,
            run_id=baseline_run_ids[0],
            horizon=HORIZON,
            save=True,
            show=show,
            filepath=baseline_fp,
        )

    for maze_name in PERTURBATION_MAZES:
        short = maze_name.replace("full_one_way_perturbed_", "")
        run_ids = _run_ids_for(
            PERTURBATION_SINGLE_POLICY,
            maze_name,
            observable,
            horizon=HORIZON,
        )
        if not run_ids:
            print(f"  [SKIP] No data: GRU {obs_tag} / {short}")
            continue

        fp = out_dir / f"perbsingle_gru_{short}_{obs_tag}.png"
        print(f"  GRU {obs_tag} / {short}")
        plot_single_run_stats(
            PERTURBATION_SINGLE_POLICY,
            maze_name,
            observable=observable,
            run_id=run_ids[0],
            horizon=HORIZON,
            save=True,
            show=show,
            filepath=fp,
        )


# ---------------------------------------------------------------------------
# Section 11 — Revaluation single-run trajectory stats
# ---------------------------------------------------------------------------


def section_11_revaluation_single_run_stats(
    dirs: dict[str, Path], show: bool
) -> None:
    print("\n=== Section 11: Revaluation single-run trajectory stats ===")
    out_dir = dirs["new_single"]
    maze_name = "full_one_way_perturbed_revaluation"
    short = maze_name.replace("full_one_way_perturbed_", "")

    for observable in OBSERVABILITY_CONDITIONS:
        obs_tag = "FO" if observable else "PO"
        for policy in REVALUATION_SINGLE_POLICIES:
            run_ids = _run_ids_for(policy, maze_name, observable, horizon=HORIZON)
            if not run_ids:
                print(
                    f"  [SKIP] No data: {_policy_display_label(policy)} "
                    f"{obs_tag} / {short}"
                )
                continue

            agent_slug = (
                policy.artifact_label if isinstance(policy, PolicySpec) else policy.value
            )
            fp = out_dir / f"single_run_{short}_{agent_slug}_{obs_tag}.png"
            print(f"  {_policy_display_label(policy)} {obs_tag} / {short}")
            plot_single_run_stats(
                policy,
                maze_name,
                observable=observable,
                run_id=run_ids[0],
                horizon=HORIZON,
                save=True,
                show=show,
                filepath=fp,
            )


# ---------------------------------------------------------------------------
# Section 12 — Detour single-run trajectory stats
# ---------------------------------------------------------------------------


def section_12_detour_single_run_stats(dirs: dict[str, Path], show: bool) -> None:
    print("\n=== Section 12: Detour single-run trajectory stats ===")
    out_dir = dirs["newer_single"]
    maze_name = "full_one_way_perturbed_detour"
    short = maze_name.replace("full_one_way_perturbed_", "")

    for observable in OBSERVABILITY_CONDITIONS:
        obs_tag = "FO" if observable else "PO"
        for policy in REVALUATION_SINGLE_POLICIES:
            run_ids = _run_ids_for(policy, maze_name, observable, horizon=HORIZON)
            if not run_ids:
                print(
                    f"  [SKIP] No data: {_policy_display_label(policy)} "
                    f"{obs_tag} / {short}"
                )
                continue

            agent_slug = (
                policy.artifact_label if isinstance(policy, PolicySpec) else policy.value
            )
            fp = out_dir / f"single_run_{short}_{agent_slug}_{obs_tag}.png"
            print(f"  {_policy_display_label(policy)} {obs_tag} / {short}")
            plot_single_run_stats(
                policy,
                maze_name,
                observable=observable,
                run_id=run_ids[0],
                horizon=HORIZON,
                save=True,
                show=show,
                filepath=fp,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all presentation figures.")
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display each figure interactively (default: save only)",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Save figures without displaying (default)",
    )
    args = parser.parse_args()

    dirs = _make_dirs()
    print(f"Output root: {FIGURES_DIR}")

    section_1_baseline(dirs, show=args.show)
    section_2_recovery_curves(dirs, show=args.show)
    section_3_heatmap(dirs, show=args.show)
    section_4_comparisons(dirs, show=args.show)
    section_5_patch_timing(dirs, show=args.show)
    section_6_single_run_stats(dirs, show=args.show)
    section_7_boundary_window_bars(dirs, show=args.show)
    section_8_boundary_window_averages(dirs, show=args.show)
    section_9_perturbation_single_run_gru_po(dirs, show=args.show)
    section_11_revaluation_single_run_stats(dirs, show=args.show)
    section_12_detour_single_run_stats(dirs, show=args.show)

    print("\nDone. Figures saved under:", FIGURES_DIR)


if __name__ == "__main__":
    main()
