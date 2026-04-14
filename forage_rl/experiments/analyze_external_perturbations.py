"""Analyze externally generated perturbation trajectories against patch benchmarks."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from forage_rl.agents.registry import Agent, PolicySpec, is_neural_agent, validate_context_mode
from forage_rl.analysis import (
    CombinedPerturbationRun,
    build_patch_benchmark_maze,
    load_combined_perturbation_runs,
    recovery_auc,
    recovery_curve_for_episode_sequence,
    resolve_patch_benchmark_kind,
    resolve_patch_benchmark_prt,
    signed_recovery_curve_for_episode_sequence,
)
from forage_rl.analysis.patch_timing import infer_hidden_states_for_trajectory
from forage_rl.visualization import (
    plot_recovery_auc_comparison,
    plot_recovery_curve_comparison,
    plot_signed_recovery_curve_comparison,
)


@dataclass(frozen=True)
class ExternalRecoveryRunResult:
    run: CombinedPerturbationRun
    benchmark_kind: str
    post_episode_indices: list[int]
    absolute_recovery_curve: np.ndarray
    signed_recovery_curve: np.ndarray
    recovery_auc: float
    no_leave_episode_count: int


def _manifest_policy(run: CombinedPerturbationRun) -> PolicySpec | str:
    try:
        agent = Agent(run.agent)
    except ValueError:
        return run.agent

    if is_neural_agent(agent):
        if run.context_mode is None:
            return PolicySpec(agent=agent, context_mode="legacy_context")
        return PolicySpec(agent=agent, context_mode=validate_context_mode(run.context_mode))
    return PolicySpec(agent=agent)


def _analysis_patch_labels(run: CombinedPerturbationRun, benchmark_maze) -> dict[int, str]:
    if run.observable:
        return {
            state_spec.id: state_spec.label
            for state_spec in benchmark_maze.maze_spec.states
        }
    return {
        observation_group: label
        for observation_group, label in enumerate(benchmark_maze.maze_spec.observation_labels)
    }


def _benchmark_label(benchmark_kind: str, observable: bool) -> str:
    if benchmark_kind == "true_mvt":
        return (
            "Compared Against FO Ideal Benchmark (True MVT)"
            if not observable
            else "Patch-Leaving Benchmark (True MVT)"
        )
    return (
        "Compared Against FO Ideal Benchmark"
        if not observable
        else "Patch-Leaving Benchmark (FO Oracle)"
    )


def _analyze_run(
    run: CombinedPerturbationRun,
    *,
    recovery_window: int,
    benchmark_mode: str,
) -> ExternalRecoveryRunResult:
    benchmark_kind = (
        run.benchmark_kind
        if run.benchmark_kind is not None and benchmark_mode == "auto"
        else (
            resolve_patch_benchmark_kind(run.maze_name, run.observable)
            if benchmark_mode == "auto"
            else benchmark_mode
        )
    )
    benchmark_maze = build_patch_benchmark_maze(
        maze_name=run.maze_name,
        horizon=run.horizon,
        benchmark_params=run.benchmark_params,
    )
    benchmark_prt_by_state = resolve_patch_benchmark_prt(
        maze_name=run.maze_name,
        observable=run.observable,
        horizon=run.horizon,
        benchmark_mode=benchmark_kind,
        benchmark_params=run.benchmark_params,
    )
    post_episode_indices, post_episode_trajectories = run.split_post_perturbation_episodes()
    resolved_states_by_episode = None
    if not run.observable:
        resolved_states_by_episode = [
            infer_hidden_states_for_trajectory(trajectory, maze=benchmark_maze)
            for trajectory in post_episode_trajectories
        ]

    absolute_curve = recovery_curve_for_episode_sequence(
        post_episode_trajectories,
        patch_labels=_analysis_patch_labels(run, benchmark_maze),
        leave_action=benchmark_maze.action_labels.index("leave"),
        benchmark_prt_by_state=benchmark_prt_by_state,
        resolved_states_by_episode=resolved_states_by_episode,
    )
    signed_curve = signed_recovery_curve_for_episode_sequence(
        post_episode_trajectories,
        patch_labels=_analysis_patch_labels(run, benchmark_maze),
        leave_action=benchmark_maze.action_labels.index("leave"),
        benchmark_prt_by_state=benchmark_prt_by_state,
        resolved_states_by_episode=resolved_states_by_episode,
    )
    return ExternalRecoveryRunResult(
        run=run,
        benchmark_kind=benchmark_kind,
        post_episode_indices=post_episode_indices,
        absolute_recovery_curve=absolute_curve,
        signed_recovery_curve=signed_curve,
        recovery_auc=recovery_auc(absolute_curve, recovery_window),
        no_leave_episode_count=int(np.count_nonzero(~np.isfinite(absolute_curve))),
    )


def _matches_filter(value: str, selected: set[str] | None) -> bool:
    return selected is None or value in selected


def _print_group_summary(
    condition_key: tuple[str, bool, str, str, str],
    results: list[ExternalRecoveryRunResult],
) -> None:
    maze_name, observable, perturbation_kind, perturbation_id, benchmark_kind = condition_key
    print(
        f"[{maze_name} {'FO' if observable else 'PO'} {perturbation_kind}/{perturbation_id}] "
        f"runs={len(results)} benchmark={benchmark_kind}"
    )
    no_leave_by_agent: dict[str, int] = defaultdict(int)
    for result in results:
        no_leave_by_agent[str(result.run.agent)] += result.no_leave_episode_count
    for agent, no_leave_count in sorted(no_leave_by_agent.items()):
        print(f"  agent={agent} no_leave_episodes={no_leave_count}")


def run_external_perturbation_analysis(
    *,
    input_manifest: str,
    maze_filter: set[str] | None = None,
    observability_filter: set[bool] | None = None,
    perturbation_kind_filter: set[str] | None = None,
    agent_filter: set[str] | None = None,
    recovery_window: int = 100,
    benchmark_mode: str = "auto",
    save: bool = False,
    show: bool = False,
) -> dict[tuple[str, bool, str, str, str], list[ExternalRecoveryRunResult]]:
    """Analyze external perturbation trajectories and optionally plot grouped figures."""
    if recovery_window <= 0:
        raise ValueError(f"recovery_window must be > 0, got {recovery_window}")

    manifest_runs = load_combined_perturbation_runs(input_manifest)
    selected_runs = [
        run
        for run in manifest_runs
        if _matches_filter(run.maze_name, maze_filter)
        and (observability_filter is None or run.observable in observability_filter)
        and _matches_filter(run.perturbation_kind, perturbation_kind_filter)
        and _matches_filter(run.agent, agent_filter)
    ]

    grouped_results: dict[
        tuple[str, bool, str, str, str],
        list[ExternalRecoveryRunResult],
    ] = defaultdict(list)
    for run in selected_runs:
        result = _analyze_run(
            run,
            recovery_window=recovery_window,
            benchmark_mode=benchmark_mode,
        )
        grouped_results[
            (
                run.maze_name,
                run.observable,
                run.perturbation_kind,
                run.perturbation_id,
                result.benchmark_kind,
            )
        ].append(result)

    for condition_key, results in sorted(grouped_results.items()):
        maze_name, observable, perturbation_kind, perturbation_id, benchmark_kind = condition_key
        curves_by_policy: dict[PolicySpec | str, list[np.ndarray]] = defaultdict(list)
        signed_curves_by_policy: dict[PolicySpec | str, list[np.ndarray]] = defaultdict(list)
        aucs_by_policy: dict[PolicySpec | str, list[float]] = defaultdict(list)
        for result in results:
            policy = _manifest_policy(result.run)
            curves_by_policy[policy].append(result.absolute_recovery_curve)
            signed_curves_by_policy[policy].append(result.signed_recovery_curve)
            aucs_by_policy[policy].append(result.recovery_auc)

        perturbation_label = (
            f"{perturbation_kind}_{perturbation_id}"
            if perturbation_id
            else perturbation_kind
        )
        benchmark_label = _benchmark_label(benchmark_kind, observable)
        condition_label = (
            f"{perturbation_kind}, {perturbation_id}"
            if perturbation_id
            else perturbation_kind
        )
        filename_suffix = benchmark_kind

        plot_recovery_curve_comparison(
            curves_by_policy,
            maze_name=maze_name,
            observable=observable,
            perturbation_label=perturbation_label,
            condition_label=condition_label,
            filename_suffix=filename_suffix,
            benchmark_label=benchmark_label,
            save=save,
            show=show,
        )
        plot_signed_recovery_curve_comparison(
            signed_curves_by_policy,
            maze_name=maze_name,
            observable=observable,
            perturbation_label=perturbation_label,
            condition_label=condition_label,
            filename_suffix=filename_suffix,
            benchmark_label=benchmark_label,
            save=save,
            show=show,
        )
        plot_recovery_auc_comparison(
            aucs_by_policy,
            maze_name=maze_name,
            observable=observable,
            perturbation_label=perturbation_label,
            condition_label=condition_label,
            filename_suffix=filename_suffix,
            benchmark_label=benchmark_label,
            save=save,
            show=show,
        )
        _print_group_summary(condition_key, results)

    return dict(grouped_results)


def _parse_optional_set(values: list[str] | None) -> set[str] | None:
    if values is None or values == ["all"]:
        return None
    return set(values)


def _parse_observability_filter(value: str) -> set[bool] | None:
    if value == "all":
        return None
    if value == "fo":
        return {True}
    return {False}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze externally generated perturbation trajectories."
    )
    parser.add_argument("--input-manifest", required=True, help="Path to a JSON manifest.")
    parser.add_argument(
        "--maze",
        nargs="+",
        default=["all"],
        help="Maze name(s) to analyze, or 'all'.",
    )
    parser.add_argument(
        "--observability",
        choices=["fo", "po", "all"],
        default="all",
        help="Observability filter.",
    )
    parser.add_argument(
        "--perturbation-kind",
        nargs="+",
        default=["all"],
        help="Perturbation kind(s) to analyze, or 'all'.",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["all"],
        help="Agent label(s) to analyze, or 'all'.",
    )
    parser.add_argument(
        "--recovery-window",
        type=int,
        default=100,
        help="Number of post-perturbation episodes to include in AUC.",
    )
    parser.add_argument(
        "--benchmark-mode",
        choices=["auto", "true_mvt", "fo_oracle"],
        default="auto",
        help="Benchmark mode override. Default resolves by maze family.",
    )
    parser.add_argument("--save", action="store_true", help="Save figures to disk.")
    parser.add_argument("--show", action="store_true", help="Display figures interactively.")

    args = parser.parse_args()
    run_external_perturbation_analysis(
        input_manifest=args.input_manifest,
        maze_filter=_parse_optional_set(args.maze),
        observability_filter=_parse_observability_filter(args.observability),
        perturbation_kind_filter=_parse_optional_set(args.perturbation_kind),
        agent_filter=_parse_optional_set(args.agents),
        recovery_window=args.recovery_window,
        benchmark_mode=args.benchmark_mode,
        save=args.save,
        show=args.show,
    )


if __name__ == "__main__":
    main()
