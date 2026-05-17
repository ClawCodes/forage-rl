import json
from pathlib import Path

import numpy as np

from forage_rl.analysis import (
    patch_exit_action_indices,
    within_episode_boundary_window_recovery_curve_for_trajectory,
    within_episode_recovery_curve_for_trajectory,
    within_episode_signed_boundary_window_recovery_curve_for_trajectory,
    within_episode_signed_recovery_curve_for_trajectory,
)
from forage_rl.analysis.oracle_patch_benchmark import oracle_patch_optimal_prt_by_state
from forage_rl.analysis.patch_timing import aggregate_curves
from forage_rl.environments import Maze, load_maze_spec
from forage_rl.experiments.analyze_external_perturbations import (
    run_external_perturbation_analysis,
)
from forage_rl.types import TimedTransition, Trajectory


def _visit(
    *,
    state: int,
    next_state: int,
    dwell: int,
    stay_action: int = 0,
    exit_action: int = 1,
) -> list[TimedTransition]:
    transitions: list[TimedTransition] = []
    for time_spent in range(dwell - 1):
        transitions.append(
            TimedTransition(
                state=state,
                action=stay_action,
                reward=0.0,
                next_state=state,
                time_spent=time_spent,
            )
        )
    transitions.append(
        TimedTransition(
            state=state,
            action=exit_action,
            reward=0.0,
            next_state=next_state,
            time_spent=dwell - 1,
        )
    )
    return transitions


def _build_single_episode_trajectory() -> Trajectory[TimedTransition]:
    transitions: list[TimedTransition] = []
    transitions.extend(_visit(state=0, next_state=1, dwell=500))
    transitions.extend(_visit(state=1, next_state=0, dwell=3))
    transitions.extend(_visit(state=0, next_state=1, dwell=3))

    tail_length = 1000 - len(transitions)
    for time_spent in range(tail_length):
        transitions.append(
            TimedTransition(
                state=1,
                action=0,
                reward=0.0,
                next_state=1,
                time_spent=time_spent,
            )
        )

    assert len(transitions) == 1000
    return Trajectory(transitions=transitions)


def test_within_episode_recovery_uses_only_complete_post_boundary_visits():
    trajectory = _build_single_episode_trajectory()

    curve = within_episode_recovery_curve_for_trajectory(
        trajectory,
        patch_labels={0: "Upper Patch", 1: "Lower Patch"},
        exit_actions={1},
        benchmark_prt_by_state={0: 2, 1: 3},
        perturbation_timestep=500,
    )
    signed_curve = within_episode_signed_recovery_curve_for_trajectory(
        trajectory,
        patch_labels={0: "Upper Patch", 1: "Lower Patch"},
        exit_actions={1},
        benchmark_prt_by_state={0: 2, 1: 3},
        perturbation_timestep=500,
    )

    np.testing.assert_allclose(curve, np.array([0.0, 1.0]))
    np.testing.assert_allclose(signed_curve, np.array([0.0, 1.0]))


def test_boundary_window_curve_captures_pre_and_post_boundary_visits():
    trajectory = _build_single_episode_trajectory()

    curve = within_episode_boundary_window_recovery_curve_for_trajectory(
        trajectory,
        patch_labels={0: "Upper Patch", 1: "Lower Patch"},
        exit_actions={1},
        benchmark_prt_by_state={0: 500, 1: 3},
        perturbation_timestep=500,
        window=100,
    )
    signed_curve = within_episode_signed_boundary_window_recovery_curve_for_trajectory(
        trajectory,
        patch_labels={0: "Upper Patch", 1: "Lower Patch"},
        exit_actions={1},
        benchmark_prt_by_state={0: 500, 1: 3},
        perturbation_timestep=500,
        window=100,
    )

    assert curve.shape == (201,)
    assert signed_curve.shape == (201,)
    assert curve[0] == 0.0
    assert signed_curve[0] == 0.0
    assert curve[99] == 0.0
    assert signed_curve[99] == 0.0
    assert curve[100] == 0.0
    assert signed_curve[100] == 0.0
    assert curve[101] == 0.0
    assert signed_curve[101] == 0.0
    assert curve[102] == 0.0
    assert signed_curve[102] == 0.0
    assert curve[103] == 497.0
    assert signed_curve[103] == -497.0
    assert np.isnan(curve[106])
    assert np.isnan(signed_curve[106])


def test_boundary_window_curve_supports_asymmetric_windows():
    trajectory = _build_single_episode_trajectory()

    curve = within_episode_boundary_window_recovery_curve_for_trajectory(
        trajectory,
        patch_labels={0: "Upper Patch", 1: "Lower Patch"},
        exit_actions={1},
        benchmark_prt_by_state={0: 500, 1: 3},
        perturbation_timestep=500,
        window_before=50,
        window_after=100,
    )

    assert curve.shape == (151,)
    assert curve[0] == 0.0
    assert curve[50] == 0.0
    assert curve[53] == 497.0
    assert np.isnan(curve[56])


def test_aggregate_curves_handles_variable_length_inputs():
    summary = aggregate_curves([np.array([1.0, 2.0]), np.array([3.0])])

    np.testing.assert_allclose(summary.x, np.array([0, 1]))
    np.testing.assert_allclose(summary.mean, np.array([2.0, 2.0]))
    np.testing.assert_allclose(summary.std, np.array([1.0, 0.0]))


def test_external_analysis_auto_falls_back_to_within_episode_for_single_episode(tmp_path):
    trajectory = _build_single_episode_trajectory()
    trajectory_path = tmp_path / "single_episode.npy"
    np.save(trajectory_path, trajectory.to_numpy())

    manifest_path = tmp_path / "manifest.json"
    manifest_payload = {
        "runs": [
            {
                "run_id": "single-episode-mid-perturb",
                "agent": "q_learning",
                "maze_name": "simple",
                "observable": True,
                "perturbation_kind": "detour",
                "perturbation_id": "mid_episode_500",
                "trajectory_path": str(trajectory_path),
                "perturbation_timestep": 500,
                "episode_lengths": [1000],
                "transition_type": "TimedTransition",
                "horizon": 1000,
            }
        ]
    }
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    results = run_external_perturbation_analysis(
        input_manifest=str(manifest_path),
        recovery_window=2,
        recovery_granularity="auto",
        save=False,
        show=False,
    )

    assert len(results) == 1
    condition_key, grouped_results = next(iter(results.items()))
    assert condition_key[-1] == "within_episode"
    assert len(grouped_results) == 1
    result = grouped_results[0]
    assert result.recovery_granularity == "within_episode"
    assert result.absolute_recovery_curve.size > 0
    assert np.isfinite(result.recovery_auc)


def test_detour_benchmark_supports_multiple_patch_exit_actions():
    repo_root = Path(__file__).resolve().parents[2]
    spec_path = (
        repo_root
        / "forage_rl"
        / "environments"
        / "maze_specs"
        / "full_one_way_perturbed_detour.toml"
    )
    maze = Maze(load_maze_spec(spec_path).perturbed(), seed=0, horizon=1000)

    exit_actions = patch_exit_action_indices(maze)
    optimal_prt_by_state = oracle_patch_optimal_prt_by_state(maze)

    assert len(exit_actions) == 2
    assert set(optimal_prt_by_state) == {0, 1, 2, 3, 4, 5}
