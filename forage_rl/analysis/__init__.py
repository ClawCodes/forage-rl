"""Analysis helpers for trajectory and benchmark summaries."""

from .benchmark_resolver import (
    build_patch_benchmark_maze,
    resolve_patch_benchmark_kind,
    resolve_patch_benchmark_prt,
)
from .mvt import (
    reward_schedule_for_state,
    simple_one_way_true_mvt_optimal_prt,
    simple_true_mvt_optimal_exploit_steps,
    simple_true_mvt_optimal_prt,
)
from .oracle_patch_benchmark import oracle_patch_optimal_prt_by_state
from .perturbation_inputs import (
    CombinedPerturbationRun,
    episode_trajectories_from_combined_stream,
    load_combined_perturbation_runs,
    load_combined_trajectory,
    split_post_perturbation_episodes,
)
from .recovery import (
    episode_prt_deviation_from_benchmark,
    recovery_auc,
    recovery_curve_for_episode_sequence,
    recovery_curve_for_run,
    signed_recovery_curve_for_episode_sequence,
    signed_recovery_curve_for_run,
)

__all__ = [
    "build_patch_benchmark_maze",
    "CombinedPerturbationRun",
    "episode_prt_deviation_from_benchmark",
    "episode_trajectories_from_combined_stream",
    "load_combined_perturbation_runs",
    "load_combined_trajectory",
    "oracle_patch_optimal_prt_by_state",
    "recovery_auc",
    "recovery_curve_for_episode_sequence",
    "recovery_curve_for_run",
    "resolve_patch_benchmark_kind",
    "resolve_patch_benchmark_prt",
    "reward_schedule_for_state",
    "signed_recovery_curve_for_episode_sequence",
    "signed_recovery_curve_for_run",
    "simple_one_way_true_mvt_optimal_prt",
    "simple_true_mvt_optimal_exploit_steps",
    "simple_true_mvt_optimal_prt",
    "split_post_perturbation_episodes",
]
