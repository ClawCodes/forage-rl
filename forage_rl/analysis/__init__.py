"""Analysis helpers for trajectory and benchmark summaries."""

from .action_semantics import patch_exit_action_indices
from .benchmark_resolver import (
    build_patch_benchmark_maze,
    resolve_patch_benchmark_kind,
    resolve_patch_benchmark_prt,
)
from .manifest_builder import (
    ManifestArtifact,
    build_external_perturbation_manifest,
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
    boundary_window_segment_means,
    episode_prt_deviation_from_benchmark,
    recovery_auc,
    recovery_curve_for_episode_sequence,
    recovery_curve_for_run,
    signed_recovery_curve_for_episode_sequence,
    signed_recovery_curve_for_run,
    within_episode_boundary_window_recovery_curve_for_trajectory,
    within_episode_recovery_curve_for_trajectory,
    within_episode_signed_boundary_window_recovery_curve_for_trajectory,
    within_episode_signed_recovery_curve_for_trajectory,
)

__all__ = [
    "build_patch_benchmark_maze",
    "build_external_perturbation_manifest",
    "boundary_window_segment_means",
    "CombinedPerturbationRun",
    "episode_prt_deviation_from_benchmark",
    "episode_trajectories_from_combined_stream",
    "load_combined_perturbation_runs",
    "load_combined_trajectory",
    "ManifestArtifact",
    "oracle_patch_optimal_prt_by_state",
    "patch_exit_action_indices",
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
    "within_episode_boundary_window_recovery_curve_for_trajectory",
    "within_episode_recovery_curve_for_trajectory",
    "within_episode_signed_boundary_window_recovery_curve_for_trajectory",
    "within_episode_signed_recovery_curve_for_trajectory",
]
