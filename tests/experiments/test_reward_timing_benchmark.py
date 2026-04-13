import json
import sys
from pathlib import Path

import numpy as np
import pytest

import forage_rl.config as config_module
import forage_rl.experiments.generate_trajectories as generation_module
import forage_rl.experiments.inspect_neural_context as inspect_context_module
import forage_rl.experiments.model_inference as inference_module
import forage_rl.experiments.regenerate_artifacts as artifacts_module
import forage_rl.experiments.reward_timing_benchmark as reward_timing_module
import forage_rl.experiments.train_pretrained_agents as pretrain_module
import forage_rl.utils.io as io_module
import forage_rl.visualization.plots as plots_module
from forage_rl import RunDataset, TimedTransition, Trajectory
from forage_rl.agents.registry import Agent, EvaluatorSpec, PolicySpec


@pytest.fixture
def isolated_reward_timing_dirs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    data_dir = tmp_path / "data"
    trajectories_dir = data_dir / "trajectories"
    logprobs_dir = data_dir / "logprobs"
    checkpoints_dir = data_dir / "checkpoints"
    figures_dir = tmp_path / "figures"

    monkeypatch.setattr(config_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(config_module, "TRAJECTORIES_DIR", trajectories_dir)
    monkeypatch.setattr(config_module, "LOGPROBS_DIR", logprobs_dir)
    monkeypatch.setattr(config_module, "CHECKPOINTS_DIR", checkpoints_dir)
    monkeypatch.setattr(config_module, "FIGURES_DIR", figures_dir)
    monkeypatch.setattr(io_module, "TRAJECTORIES_DIR", trajectories_dir)
    monkeypatch.setattr(io_module, "LOGPROBS_DIR", logprobs_dir)
    monkeypatch.setattr(io_module, "CHECKPOINTS_DIR", checkpoints_dir)
    monkeypatch.setattr(plots_module, "FIGURES_DIR", figures_dir)

    return tmp_path


def _timed_trajectory(*steps: tuple[int, int, float, int, int]) -> Trajectory:
    transitions = [
        TimedTransition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            time_spent=time_spent,
        )
        for state, action, reward, next_state, time_spent in steps
    ]
    return Trajectory(transitions=transitions)


def _benchmark_run_dataset() -> RunDataset:
    return RunDataset(
        trajectories=[
            _timed_trajectory(
                (0, 0, 1.0, 0, 0),
                (0, 0, 0.0, 0, 1),
                (0, 1, 0.0, 1, 2),
                (1, 0, 0.0, 1, 0),
                (1, 1, 0.0, 0, 1),
            ),
            _timed_trajectory(
                (1, 0, 1.0, 1, 0),
                (1, 0, 0.0, 1, 1),
                (1, 1, 0.0, 0, 2),
                (0, 0, 0.0, 0, 0),
                (0, 1, 0.0, 1, 1),
            ),
        ]
    )


def _mvt_deviation_run_dataset() -> RunDataset:
    return RunDataset(
        trajectories=[
            _timed_trajectory(
                (0, 0, 1.0, 0, 0),
                (0, 0, 0.0, 0, 1),
                (0, 0, 0.0, 0, 2),
                (0, 1, 0.0, 1, 3),
                (1, 0, 1.0, 1, 0),
                (1, 0, 1.0, 1, 1),
                (1, 0, 1.0, 1, 2),
                (1, 0, 1.0, 1, 3),
                (1, 1, 0.0, 0, 4),
            )
        ]
    )


def test_extract_decision_rows_tracks_zero_streak_and_observed_patch_labels():
    patch_labels = reward_timing_module._observation_group_patch_labels("full")
    trajectory = _timed_trajectory(
        (0, 0, 0.0, 0, 0),
        (0, 0, 1.0, 0, 1),
        (0, 1, 0.0, 1, 2),
        (1, 0, 0.0, 1, 0),
        (1, 1, 0.0, 0, 1),
    )

    rows = reward_timing_module.extract_decision_rows(
        trajectory,
        patch_labels=patch_labels,
    )

    assert [row.patch_label for row in rows] == [
        "Upper Patch",
        "Upper Patch",
        "Upper Patch",
        "Lower Patch",
        "Lower Patch",
    ]
    assert [row.state for row in rows] == [0, 0, 0, 1, 1]
    assert [row.prev_reward for row in rows] == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert [row.zero_streak for row in rows] == [0, 1, 0, 0, 1]


def test_fit_patch_threshold_rule_recovers_expected_thresholds():
    rows = [
        reward_timing_module.DecisionRow(0, "Upper Patch", 0, 0.0, 0, 0),
        reward_timing_module.DecisionRow(0, "Upper Patch", 1, 1.0, 0, 0),
        reward_timing_module.DecisionRow(0, "Upper Patch", 2, 0.0, 1, 1),
        reward_timing_module.DecisionRow(3, "Lower Patch", 0, 0.0, 0, 0),
        reward_timing_module.DecisionRow(3, "Lower Patch", 1, 0.0, 1, 1),
    ]

    thresholds, score = reward_timing_module.fit_patch_threshold_rule(
        rows,
        horizon=6,
        leave_action=1,
    )

    assert thresholds == (2, 1)
    assert score == pytest.approx(1.0)


def test_fit_zero_streak_rule_recovers_expected_threshold():
    rows = [
        reward_timing_module.DecisionRow(0, "Upper Patch", 0, 0.0, 0, 0),
        reward_timing_module.DecisionRow(0, "Upper Patch", 1, 0.0, 1, 0),
        reward_timing_module.DecisionRow(0, "Upper Patch", 2, 0.0, 2, 1),
        reward_timing_module.DecisionRow(3, "Lower Patch", 0, 1.0, 0, 0),
        reward_timing_module.DecisionRow(3, "Lower Patch", 1, 0.0, 1, 0),
        reward_timing_module.DecisionRow(3, "Lower Patch", 2, 0.0, 2, 1),
    ]

    threshold, score = reward_timing_module.fit_zero_streak_rule(
        rows,
        horizon=6,
        leave_action=1,
    )

    assert threshold == 2
    assert score == pytest.approx(1.0)


def test_mvt_optimal_dwell_by_state_matches_full_maze_policy():
    assert reward_timing_module._mvt_optimal_dwell_by_state(
        maze_name="full",
        horizon=100,
    ) == {
        0: 2,
        1: 2,
        2: 7,
        3: 7,
        4: 2,
        5: 2,
    }


def test_normalized_curve_auc_handles_empty_singleton_and_multi_point_curves():
    assert np.isnan(
        reward_timing_module._normalized_curve_auc(np.array([], dtype=float))
    )
    assert np.isnan(
        reward_timing_module._normalized_curve_auc(
            np.array([np.nan, np.nan], dtype=float)
        )
    )
    assert reward_timing_module._normalized_curve_auc(
        np.array([np.nan, 0.4, np.nan], dtype=float)
    ) == pytest.approx(0.4)

    auc = reward_timing_module._normalized_curve_auc(
        np.array([np.nan, 0.0, 0.5, 1.0, np.nan], dtype=float)
    )

    assert auc == pytest.approx(0.5)
    assert 0.0 <= auc <= 1.0


def test_summarize_policy_run_reports_signed_mvt_deviation_by_patch():
    summary = reward_timing_module.summarize_policy_run(
        _mvt_deviation_run_dataset(),
        maze_name="full",
        patch_labels=reward_timing_module._observation_group_patch_labels("full"),
        leave_action=1,
        horizon=10,
        optimal_dwell_by_state={0: 2, 3: 7},
    )

    assert summary.upper_mvt_deviation == pytest.approx(2.0)
    assert summary.lower_mvt_deviation == pytest.approx(-2.0)
    assert 0.0 <= summary.upper_leave_prob_auc <= 1.0
    assert 0.0 <= summary.lower_leave_prob_auc <= 1.0


def test_reward_timing_artifact_paths_are_distinct_and_labeled(
    isolated_reward_timing_dirs: Path,
):
    run_dataset = _benchmark_run_dataset()

    prev_reward_path = io_module.save_run_dataset(
        run_dataset,
        Agent.DQN,
        0,
        "full",
        False,
        context_mode="prev_reward",
    )
    prev_reward_time_path = io_module.save_run_dataset(
        run_dataset,
        Agent.DQN,
        0,
        "full",
        False,
        context_mode="prev_reward_time",
    )
    logprob_path = io_module.save_logprobs(
        np.array([1.0, 2.0], dtype=float),
        Agent.DQN,
        EvaluatorSpec(agent=Agent.DQN, mode="fresh", context_mode="prev_reward_time"),
        0,
        "full",
        False,
        source_context_mode="prev_reward",
    )

    assert prev_reward_path.name == "full_PO_dqn_prev_reward_run_dataset_0.npz"
    assert prev_reward_time_path.name == "full_PO_dqn_prev_reward_time_run_dataset_0.npz"
    assert logprob_path.name == (
        "full_PO_source_dqn_prev_reward_eval_dqn_prev_reward_time_fresh_log_likelihoods_0.npy"
    )
    assert (
        io_module.checkpoint_path(
            Agent.DQN,
            "full",
            False,
            context_mode="prev_reward",
        ).name
        == "full_PO_dqn_prev_reward_final.pt"
    )
    assert (
        io_module.checkpoint_path(
            Agent.DQN,
            "full",
            False,
            context_mode="prev_reward_time",
        ).name
        == "full_PO_dqn_prev_reward_time_final.pt"
    )
    assert json.loads(prev_reward_path.with_suffix(".json").read_text(encoding="utf-8"))[
        "context_mode"
    ] == "prev_reward"
    assert json.loads(
        prev_reward_time_path.with_suffix(".json").read_text(encoding="utf-8")
    )["context_mode"] == "prev_reward_time"


def test_plot_labels_and_sorting_support_reward_timing_context_modes():
    policy = PolicySpec(agent=Agent.GRU, context_mode="prev_reward_time")
    compare_to = [
        EvaluatorSpec(agent=Agent.DQN, mode="fresh", context_mode="legacy_context"),
        EvaluatorSpec(agent=Agent.DQN, mode="fresh", context_mode="prev_reward_time"),
        EvaluatorSpec(agent=Agent.DQN, mode="fresh", context_mode="observation_only"),
        EvaluatorSpec(agent=Agent.DQN, mode="fresh", context_mode="prev_reward"),
    ]

    assert plots_module._policy_display_label(policy, include_context=True) == (
        "GRU (obs+prev_reward+time)"
    )
    specs = plots_module._comparison_specs(Agent.MBRL, compare_to)
    assert [spec.context_mode for spec in specs] == [
        "observation_only",
        "prev_reward",
        "prev_reward_time",
        "legacy_context",
    ]
    assert len(
        {
            plots_module._line_style(EvaluatorSpec(agent=Agent.DQN, mode="fresh", context_mode=context_mode))["linestyle"]
            for context_mode in (
                "observation_only",
                "prev_reward",
                "prev_reward_time",
                "legacy_context",
            )
        }
    ) == 4


def test_reward_timing_benchmark_report_and_figures_are_generated(
    isolated_reward_timing_dirs: Path,
):
    run_dataset = _benchmark_run_dataset()
    for policy in reward_timing_module.reward_timing_policies():
        io_module.save_run_dataset(
            run_dataset,
            policy.agent,
            0,
            "full",
            False,
            context_mode=policy.context_mode,
        )

    artifacts = reward_timing_module.analyze_reward_timing_benchmark(
        maze_name="full",
        observable=False,
        num_datasets=1,
        verbose=False,
    )

    report_text = artifacts.report_path.read_text(encoding="utf-8")
    assert artifacts.matched_run_ids == (0,)
    assert artifacts.report_path.exists()
    assert len(artifacts.figure_paths) == 4
    assert all(path.exists() for path in artifacts.figure_paths)
    assert any(
        path.name == "mvt_residency_deviation_auc_full_PO_reward_timing_benchmark.png"
        for path in artifacts.figure_paths
    )
    assert "Reward Timing Benchmark" in report_text
    assert "Tail return uses the last 20% of episodes per run" in report_text
    assert "best (k_upper, k_lower)" in report_text
    assert "## MVT Timing Summary" in report_text
    assert "MVT deviation uses the true-state full-MDP value-iteration policy" in report_text
    assert "Leave-time AUC is normalized" in report_text
    assert "DQN (obs+prev_reward)" in report_text
    assert "LSTM (obs+prev_reward+time)" in report_text


def test_tail_window_episodes_scales_with_run_length():
    assert reward_timing_module._tail_window_episodes(1) == 1
    assert reward_timing_module._tail_window_episodes(2) == 1
    assert reward_timing_module._tail_window_episodes(10) == 2
    assert reward_timing_module._tail_window_episodes(50) == 10


def test_generation_main_accepts_reward_timing_context_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        generation_module,
        "run_generation_experiment",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_trajectories",
            "--agents",
            "dqn",
            "--context-mode",
            "prev_reward_time",
            "--quiet",
        ],
    )

    generation_module.main()

    assert captured["context_mode"] == "prev_reward_time"


def test_model_inference_main_accepts_reward_timing_context_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        inference_module,
        "run_inference_experiment",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_inference",
            "--source-agents",
            "dqn",
            "--compare-to",
            "dqn:fresh",
            "--context-mode",
            "prev_reward",
            "--quiet",
        ],
    )

    inference_module.main()

    assert captured["source_context_mode"] == "prev_reward"
    assert captured["compare_to"] == [
        EvaluatorSpec(agent=Agent.DQN, mode="fresh", context_mode="prev_reward")
    ]


def test_train_pretrained_main_accepts_reward_timing_context_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        pretrain_module,
        "train_pretrained_agents",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_pretrained_agents",
            "--agents",
            "dqn",
            "--context-mode",
            "prev_reward_time",
            "--quiet",
        ],
    )

    pretrain_module.main()

    assert captured["context_mode"] == "prev_reward_time"


def test_inspect_neural_context_main_accepts_reward_timing_context_mode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    captured: dict[str, object] = {}

    def fake_inspect_neural_context(**kwargs):
        captured.update(kwargs)
        return [
            {
                "step_index": 0,
                "state": 0,
                "time_spent": 0,
                "prev_action": None,
                "prev_reward": 0.0,
                "action": 0,
                "reward": 1.0,
                "next_state": 0,
                "encoded_feature": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            }
        ]

    monkeypatch.setattr(
        inspect_context_module,
        "inspect_neural_context",
        fake_inspect_neural_context,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inspect_neural_context",
            "--agent",
            "dqn",
            "--maze",
            "full",
            "--run-id",
            "0",
            "--episode-index",
            "0",
            "--context-mode",
            "prev_reward",
        ],
    )

    inspect_context_module.main()

    assert captured["context_mode"] == "prev_reward"
    assert "feature" in capsys.readouterr().out


def test_regenerate_artifacts_main_accepts_reward_timing_benchmark_flag(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        artifacts_module,
        "regenerate_artifacts",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "regenerate_artifacts",
            "--reward-timing-benchmark",
            "--quiet",
        ],
    )

    artifacts_module.main()

    assert captured["reward_timing_benchmark"] is True


def test_reward_timing_benchmark_preset_dispatches_expected_context_modes(
    monkeypatch: pytest.MonkeyPatch,
):
    generation_calls: list[dict[str, object]] = []
    inference_calls: list[dict[str, object]] = []
    analyze_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        artifacts_module,
        "run_generation_experiment",
        lambda **kwargs: generation_calls.append(kwargs),
    )
    monkeypatch.setattr(
        artifacts_module,
        "run_inference_experiment",
        lambda **kwargs: inference_calls.append(kwargs),
    )
    monkeypatch.setattr(
        artifacts_module,
        "plot_episode_return_comparison",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        artifacts_module,
        "plot_aggregate_trajectory_stats",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        artifacts_module,
        "plot_aggregate_comparison",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        artifacts_module,
        "analyze_reward_timing_benchmark",
        lambda **kwargs: analyze_calls.append(kwargs),
    )

    artifacts_module.regenerate_artifacts(
        reward_timing_benchmark=True,
        num_runs=1,
        num_episodes=1,
        num_datasets=1,
        workers=1,
        verbose=False,
    )

    assert [(call["maze_name"], call["observable"], call["context_mode"]) for call in generation_calls] == [
        ("full", False, "prev_reward"),
        ("full", False, "prev_reward_time"),
    ]
    assert [
        (call["maze_name"], call["observable"], call["source_context_mode"])
        for call in inference_calls
    ] == [
        ("full", False, "prev_reward"),
        ("full", False, "prev_reward_time"),
    ]
    assert analyze_calls == [
        {
            "maze_name": "full",
            "observable": False,
            "policies": reward_timing_module.reward_timing_policies(),
            "num_datasets": 1,
            "filename_suffix": "reward_timing_benchmark",
            "horizon": None,
            "verbose": False,
        }
    ]
