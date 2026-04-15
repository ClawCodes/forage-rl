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

from forage_rl.agents.registry import Agent, agent_display_label
from forage_rl.analysis import (
    patch_exit_action_indices,
    recovery_auc,
    resolve_patch_benchmark_prt,
    within_episode_recovery_curve_for_trajectory,
    within_episode_signed_recovery_curve_for_trajectory,
)
from forage_rl.config import FIGURES_DIR
from forage_rl.environments import load_builtin_maze_spec
from forage_rl.environments.maze import maze_from_builtin_maze_spec
from forage_rl.utils import list_run_dataset_run_ids, load_run_dataset
from forage_rl.visualization import (
    plot_aggregate_comparison,
    plot_aggregate_trajectory_stats,
    plot_episode_return_comparison,
    plot_patch_timing_summary,
    plot_recovery_curve_comparison,
    plot_recovery_heatmap,
    plot_recovery_heatmap_delta,
    plot_signed_recovery_curve_comparison,
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


def _compute_recovery_curves(
    agents: list[Agent],
    maze_name: str,
    observable: bool,
    horizon: int = HORIZON,
    perturbation_t: int = PERTURBATION_T,
) -> dict[Agent, list[np.ndarray]]:
    """Load saved runs and compute a within-episode recovery curve per run.

    Benchmark PRTs come from the base (full_one_way) oracle so the recovery
    curve measures deviation from the pre-perturbation optimum.
    """
    maze = maze_from_builtin_maze_spec(maze_name, observable=True, horizon=horizon)
    patch_labels = _build_patch_labels(maze_name, observable)
    exit_actions = patch_exit_action_indices(maze)
    benchmark_prt = resolve_patch_benchmark_prt(
        maze_name=BASELINE_MAZE, observable=True, horizon=horizon
    )

    curves_by_agent: dict[Agent, list[np.ndarray]] = {}
    for agent in agents:
        run_ids = list_run_dataset_run_ids(
            agent, maze_name, observable, horizon=horizon
        )
        if not run_ids:
            print(
                f"  [SKIP] No data: {agent_display_label(agent)} / {maze_name} / "
                f"{'FO' if observable else 'PO'}"
            )
            continue
        agent_curves: list[np.ndarray] = []
        for run_id in run_ids:
            run_dataset = load_run_dataset(
                agent, run_id, maze_name, observable, horizon=horizon
            )
            # Flatten all episodes in the run into a single trajectory
            from forage_rl import Trajectory

            combined = Trajectory(transitions=list(run_dataset.iter_transitions()))
            curve = within_episode_recovery_curve_for_trajectory(
                combined,
                patch_labels=patch_labels,
                exit_actions=exit_actions,
                benchmark_prt_by_state=benchmark_prt,
                perturbation_timestep=perturbation_t,
            )
            agent_curves.append(curve)
        if agent_curves:
            curves_by_agent[agent] = agent_curves
    return curves_by_agent


def _compute_signed_recovery_curves(
    agents: list[Agent],
    maze_name: str,
    observable: bool,
    horizon: int = HORIZON,
    perturbation_t: int = PERTURBATION_T,
) -> dict[Agent, list[np.ndarray]]:
    """Same as _compute_recovery_curves but returns signed deviations."""
    maze = maze_from_builtin_maze_spec(maze_name, observable=True, horizon=horizon)
    patch_labels = _build_patch_labels(maze_name, observable)
    exit_actions = patch_exit_action_indices(maze)
    benchmark_prt = resolve_patch_benchmark_prt(
        maze_name=BASELINE_MAZE, observable=True, horizon=horizon
    )

    curves_by_agent: dict[Agent, list[np.ndarray]] = {}
    for agent in agents:
        run_ids = list_run_dataset_run_ids(
            agent, maze_name, observable, horizon=horizon
        )
        if not run_ids:
            continue
        agent_curves: list[np.ndarray] = []
        for run_id in run_ids:
            run_dataset = load_run_dataset(
                agent, run_id, maze_name, observable, horizon=horizon
            )
            from forage_rl import Trajectory

            combined = Trajectory(transitions=list(run_dataset.iter_transitions()))
            curve = within_episode_signed_recovery_curve_for_trajectory(
                combined,
                patch_labels=patch_labels,
                exit_actions=exit_actions,
                benchmark_prt_by_state=benchmark_prt,
                perturbation_timestep=perturbation_t,
            )
            agent_curves.append(curve)
        if agent_curves:
            curves_by_agent[agent] = agent_curves
    return curves_by_agent


def _mean_auc(
    curves_by_agent: dict[Agent, list[np.ndarray]],
    window: int = AUC_WINDOW,
) -> dict[Agent, float]:
    """Compute mean recovery AUC across runs for each agent."""
    result: dict[Agent, float] = {}
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
        run_ids = list_run_dataset_run_ids(
            TABULAR_AGENTS[0], BASELINE_MAZE, observable, horizon=HORIZON
        )
        if not run_ids:
            print(f"  [SKIP] No baseline data for {obs_tag}")
            continue

        fp = dirs["baseline"] / f"episode_return_comparison_{BASELINE_MAZE}_{obs_tag}.png"
        print(f"  episode return comparison {obs_tag}")
        plot_episode_return_comparison(
            BASELINE_MAZE,
            observable=observable,
            agents=TABULAR_AGENTS,
            horizon=HORIZON,
            save=True,
            show=show,
            filepath=fp,
        )

        for agent in TABULAR_AGENTS:
            run_ids_agent = list_run_dataset_run_ids(
                agent, BASELINE_MAZE, observable, horizon=HORIZON
            )
            if not run_ids_agent:
                print(f"  [SKIP] No data: {agent_display_label(agent)} baseline {obs_tag}")
                continue
            agent_slug = agent.value
            fp = dirs["baseline"] / f"trajectory_stats_{agent_slug}_{BASELINE_MAZE}_{obs_tag}.png"
            print(f"  trajectory stats: {agent_display_label(agent)} {obs_tag}")
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
                TABULAR_AGENTS, maze_name, observable
            )
            signed = _compute_signed_recovery_curves(
                TABULAR_AGENTS, maze_name, observable
            )

            if not curves:
                print(f"    [SKIP] No recovery data")
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
    auc_fo: dict[Agent, dict[str, float]] = {a: {} for a in TABULAR_AGENTS}
    auc_po: dict[Agent, dict[str, float]] = {a: {} for a in TABULAR_AGENTS}

    for observable in OBSERVABILITY_CONDITIONS:
        store = auc_fo if observable else auc_po
        for maze_name in PERTURBATION_MAZES:
            curves = _compute_recovery_curves(TABULAR_AGENTS, maze_name, observable)
            mean_aucs = _mean_auc(curves)
            for agent in TABULAR_AGENTS:
                store[agent][maze_name] = mean_aucs.get(agent, float("nan"))

    fp_fo = dirs["heatmap"] / "recovery_heatmap_FO.png"
    print("  heatmap FO")
    plot_recovery_heatmap(
        auc_fo,
        PERTURBATION_LABELS,
        TABULAR_AGENTS,
        observable=True,
        save=True,
        show=show,
        filepath=fp_fo,
    )

    fp_po = dirs["heatmap"] / "recovery_heatmap_PO.png"
    print("  heatmap PO")
    plot_recovery_heatmap(
        auc_po,
        PERTURBATION_LABELS,
        TABULAR_AGENTS,
        observable=False,
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
        TABULAR_AGENTS,
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
        for agent in TABULAR_AGENTS:
            run_ids = list_run_dataset_run_ids(
                agent, BASELINE_MAZE, observable, horizon=HORIZON
            )
            if not run_ids:
                print(
                    f"  [SKIP] No data: {agent_display_label(agent)} {obs_tag}"
                )
                continue
            agent_slug = agent.value
            fp = dirs["patch_timing"] / f"patch_timing_{agent_slug}_{BASELINE_MAZE}_{obs_tag}.png"
            print(f"  {agent_display_label(agent)} {obs_tag}")
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

    print("\nDone. Figures saved under:", FIGURES_DIR)


if __name__ == "__main__":
    main()
