import json
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import forage_rl.config as config_module
import forage_rl.experiments.generate_trajectories as generation_module
import forage_rl.experiments.inspect_neural_context as inspect_context_module
import forage_rl.experiments.model_inference as inference_module
import forage_rl.experiments.parallel as parallel_module
import forage_rl.experiments.regenerate_artifacts as artifacts_module
import forage_rl.experiments.train_pretrained_agents as pretrain_module
import forage_rl.utils.io as io_module
import forage_rl.visualization.plots as plots_module
from forage_rl import RunDataset, TimedTransition, Trajectory, Transition
from forage_rl.agents import get_agent
from forage_rl.agents.registry import (
    Agent,
    EvaluatorSpec,
    PolicySpec,
    canonical_agent,
    neural_agents,
    recurrent_agents,
    registered_agents,
)
from forage_rl.environments import Maze, MazePOMDP, load_builtin_maze_spec
from forage_rl.utils.torch_support import resolve_device, torch_available


torch_required = pytest.mark.skipif(not torch_available(), reason="torch not installed")
RECURRENT_AGENT_TYPES = recurrent_agents()


@pytest.fixture
def isolated_io_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
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


def _simple_run_dataset() -> RunDataset:
    return RunDataset(
        trajectories=[
            _timed_trajectory((0, 0, 1.0, 0, 0), (0, 1, 0.0, 1, 1)),
            _timed_trajectory((1, 0, 1.0, 1, 0)),
        ]
    )


def _model_param_l1_delta(agent, before_state_dict: dict[str, object]) -> float:
    total = 0.0
    for name, value in agent.q_network.state_dict().items():
        total += float((value - before_state_dict[name]).abs().sum().item())
    return total


def _fixed_horizon_run_dataset(
    *,
    num_episodes: int,
    horizon: int,
    offset: int = 0,
) -> RunDataset:
    trajectories: list[Trajectory] = []
    for episode_index in range(num_episodes):
        transitions = [
            (
                (offset + episode_index + step) % 2,
                0,
                float((offset + episode_index + step) % 2),
                (offset + episode_index + step + 1) % 2,
                step,
            )
            for step in range(horizon)
        ]
        trajectories.append(_timed_trajectory(*transitions))
    return RunDataset(trajectories=trajectories)


def _recurrent_replay_steps(length: int) -> list[dict[str, object]]:
    return [
        {
            "state": step % 2,
            "time_spent": step,
            "prev_action": None if step == 0 else (step - 1) % 2,
            "prev_reward": 0.0 if step == 0 else float((step - 1) % 2),
            "action": step % 2,
            "reward": float(step % 2),
            "next_state": (step + 1) % 2,
            "next_time_spent": step + 1,
            "done": step == length - 1,
        }
        for step in range(length)
    ]


def _save_logprob_series(
    source: Agent,
    evaluator: Agent | EvaluatorSpec,
    run_id: int,
    values: list[float],
    *,
    maze_name: str = "simple",
    observable: bool = True,
    source_context_mode: str = "legacy_context",
) -> None:
    io_module.save_logprobs(
        np.array(values, dtype=float),
        source,
        evaluator,
        run_id,
        maze_name,
        observable,
        source_context_mode=source_context_mode,
    )


def test_trajectory_rejects_empty_transitions():
    with pytest.raises(ValidationError):
        Trajectory(transitions=[])


def test_run_dataset_rejects_empty_trajectories():
    with pytest.raises(ValidationError):
        RunDataset(trajectories=[])


def test_save_and_load_run_dataset_round_trip(isolated_io_dirs: Path):
    run_dataset = _simple_run_dataset()

    filepath = io_module.save_run_dataset(run_dataset, Agent.MBRL, 0, "simple", True)
    metadata_path = filepath.with_suffix(".json")

    assert filepath.name == "simple_FO_mbrl_run_dataset_0.npz"
    assert metadata_path.exists()

    loaded = io_module.load_run_dataset(Agent.MBRL, 0, "simple", True)
    assert loaded.num_episodes() == 2
    assert loaded.num_transitions() == 3
    assert loaded.transition_cls() is TimedTransition
    assert [transition.state for transition in loaded.iter_transitions()] == [0, 0, 1]


def test_neural_context_mode_artifacts_use_distinct_paths(isolated_io_dirs: Path):
    run_dataset = _simple_run_dataset()

    legacy_path = io_module.save_run_dataset(
        run_dataset,
        Agent.DQN,
        0,
        "simple",
        True,
        context_mode="legacy_context",
    )
    obs_only_path = io_module.save_run_dataset(
        run_dataset,
        Agent.DQN,
        0,
        "simple",
        True,
        context_mode="observation_only",
    )

    assert legacy_path.name == "simple_FO_dqn_run_dataset_0.npz"
    assert obs_only_path.name == "simple_FO_dqn_obs_only_run_dataset_0.npz"
    assert io_module.list_run_dataset_run_ids(Agent.DQN, "simple", True) == [0]
    assert io_module.list_run_dataset_run_ids(
        Agent.DQN,
        "simple",
        True,
        context_mode="observation_only",
    ) == [0]
    assert (
        io_module.checkpoint_path(Agent.DQN, "simple", True).name
        == "simple_FO_dqn_final.pt"
    )
    assert (
        io_module.checkpoint_path(
            Agent.DQN,
            "simple",
            True,
            context_mode="observation_only",
        ).name
        == "simple_FO_dqn_obs_only_final.pt"
    )
    metadata = json.loads(obs_only_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["context_mode"] == "observation_only"
    assert metadata["horizon"] == 100


def test_horizon_specific_artifacts_use_distinct_paths_and_exact_listing(
    isolated_io_dirs: Path,
):
    run_dataset = _fixed_horizon_run_dataset(num_episodes=1, horizon=300)

    default_path = io_module.save_run_dataset(
        run_dataset,
        Agent.MBRL,
        0,
        "simple",
        True,
    )
    custom_path = io_module.save_run_dataset(
        run_dataset,
        Agent.MBRL,
        0,
        "simple",
        True,
        horizon=300,
    )
    custom_logprob_path = io_module.save_logprobs(
        np.array([1.0, 2.0], dtype=float),
        Agent.MBRL,
        Agent.MBRL,
        0,
        "simple",
        True,
        horizon=300,
    )

    assert default_path.name == "simple_FO_mbrl_run_dataset_0.npz"
    assert custom_path.name == "simple_FO_h300_mbrl_run_dataset_0.npz"
    assert custom_logprob_path.name == "simple_FO_h300_source_mbrl_eval_mbrl_fresh_log_likelihoods_0.npy"
    assert io_module.checkpoint_path(Agent.DQN, "simple", True, horizon=300).name == (
        "simple_FO_h300_dqn_final.pt"
    )
    assert io_module.checkpoint_metadata_path(
        Agent.DQN,
        "simple",
        True,
        horizon=300,
    ).name == "simple_FO_h300_dqn_final.json"
    assert io_module.list_run_dataset_run_ids(Agent.MBRL, "simple", True) == [0]
    assert io_module.list_run_dataset_run_ids(
        Agent.MBRL,
        "simple",
        True,
        horizon=300,
    ) == [0]
    assert json.loads(custom_path.with_suffix(".json").read_text(encoding="utf-8"))["horizon"] == 300


def test_list_run_dataset_run_ids_and_inference_use_sparse_saved_ids(
    isolated_io_dirs: Path,
):
    run_dataset = RunDataset(
        trajectories=[_timed_trajectory((0, 0, 1.0, 0, 0))]
    )

    for run_id in [0, 2, 5]:
        io_module.save_run_dataset(run_dataset, Agent.MBRL, run_id, "simple", True)

    assert io_module.list_run_dataset_run_ids(Agent.MBRL, "simple", True) == [0, 2, 5]

    inference_module.run_inference_experiment(
        source_agents=[Agent.MBRL],
        compare_to=[Agent.MBRL],
        maze_name="simple",
        num_datasets=2,
        observable=True,
        verbose=False,
        workers=1,
    )

    assert {path.name for path in config_module.LOGPROBS_DIR.glob("*.npy")} == {
        "simple_FO_source_mbrl_eval_mbrl_fresh_log_likelihoods_0.npy",
        "simple_FO_source_mbrl_eval_mbrl_fresh_log_likelihoods_2.npy",
    }


def test_evaluate_run_dataset_reuses_one_agent_instance(monkeypatch: pytest.MonkeyPatch):
    run_dataset = _simple_run_dataset()
    calls: list[Agent] = []
    original_get_agent = inference_module.get_agent

    def counting_get_agent(name, maze, **kwargs):
        calls.append(name)
        return original_get_agent(name, maze, **kwargs)

    monkeypatch.setattr(inference_module, "get_agent", counting_get_agent)

    results = inference_module.evaluate_run_dataset(
        run_dataset,
        maze_name="simple",
        evaluators=[Agent.MBRL],
        observable=True,
    )

    assert calls == [Agent.MBRL]
    assert len(next(iter(results.values()))) == run_dataset.num_transitions()


def test_resolve_worker_count(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(parallel_module.os, "cpu_count", lambda: 12)

    assert parallel_module.resolve_worker_count(20, None) == 8
    assert parallel_module.resolve_worker_count(3, None) == 3
    assert parallel_module.resolve_worker_count(5, 1) == 1
    assert parallel_module.resolve_worker_count(5, 10) == 5
    assert parallel_module.resolve_worker_count(0, None) == 0

    with pytest.raises(ValueError):
        parallel_module.resolve_worker_count(5, 0)


def test_resolve_execution_strategy_clamps_neural_gpu_workers(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(parallel_module.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(parallel_module, "resolve_device", lambda _: "mps")

    strategy = parallel_module.resolve_execution_strategy(
        20,
        None,
        uses_torch=True,
        device="auto",
    )
    assert strategy.worker_count == 1
    assert strategy.mp_context is not None
    assert strategy.worker_note is not None


def test_selected_settings_defaults_and_filters(capsys: pytest.CaptureFixture[str]):
    assert artifacts_module._selected_settings() == [
        ("simple", True),
        ("full", True),
        ("full", False),
    ]
    assert artifacts_module._selected_settings(["simple"], "all") == [("simple", True)]
    assert artifacts_module._selected_settings(["full"], "po") == [("full", False)]

    settings = artifacts_module._selected_settings(["simple"], "po", verbose=True)
    assert settings == []
    assert "Skipping simple/PO" in capsys.readouterr().out


def test_run_generation_experiment_splits_cpu_and_neural_batches(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[dict[str, object]] = []

    def fake_build_generation_tasks(
        agent_types,
        maze_name,
        num_runs,
        num_episodes,
        observable,
        base_seed,
        device,
        context_mode,
        horizon,
    ):
        return [
            (
                agent_type,
                0,
                maze_name,
                num_episodes,
                observable,
                base_seed,
                device,
                context_mode,
                horizon,
            )
            for agent_type in agent_types
        ]

    def fake_execute_generation_tasks(
        tasks,
        *,
        agent_types,
        workers,
        device,
        verbose,
        uses_torch,
        batch_label,
    ):
        calls.append(
            {
                "tasks": tasks,
                "agent_types": agent_types,
                "uses_torch": uses_torch,
                "batch_label": batch_label,
                "device": device,
            }
        )

    monkeypatch.setattr(generation_module, "_build_generation_tasks", fake_build_generation_tasks)
    monkeypatch.setattr(generation_module, "_execute_generation_tasks", fake_execute_generation_tasks)

    generation_module.run_generation_experiment(
        agent_types=[Agent.MBRL, Agent.DQN],
        maze_name="simple",
        num_runs=1,
        num_episodes=1,
        observable=True,
        verbose=False,
        workers=10,
        device="cuda",
    )

    assert calls == [
        {
            "tasks": [
                (
                    Agent.MBRL,
                    0,
                    "simple",
                    1,
                    True,
                    None,
                    "cuda",
                    "legacy_context",
                    None,
                )
            ],
            "agent_types": [Agent.MBRL],
            "uses_torch": False,
            "batch_label": "CPU-only",
            "device": "cuda",
        },
        {
            "tasks": [
                (
                    Agent.DQN,
                    0,
                    "simple",
                    1,
                    True,
                    None,
                    "cuda",
                    "legacy_context",
                    None,
                )
            ],
            "agent_types": [Agent.DQN],
            "uses_torch": True,
            "batch_label": "neural",
            "device": "cuda",
        },
    ]


def test_run_inference_experiment_splits_cpu_and_neural_batches_and_dedupes_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    calls: list[dict[str, object]] = []

    def fake_build_inference_tasks(
        source_agents,
        compare_to,
        maze_name,
        num_datasets,
        observable,
        device,
        base_seed,
        source_context_mode,
        horizon,
    ):
        return (
            [
                {
                    "compare_to": tuple(compare_to),
                    "maze_name": maze_name,
                    "source_context_mode": source_context_mode,
                    "horizon": horizon,
                }
            ],
            [Agent.QLearning],
        )

    def fake_execute_inference_tasks(
        tasks,
        *,
        source_agents,
        workers,
        device,
        verbose,
        uses_torch,
        batch_label,
    ):
        calls.append(
            {
                "tasks": tasks,
                "uses_torch": uses_torch,
                "batch_label": batch_label,
                "device": device,
            }
        )

    monkeypatch.setattr(inference_module, "_build_inference_tasks", fake_build_inference_tasks)
    monkeypatch.setattr(inference_module, "_execute_inference_tasks", fake_execute_inference_tasks)
    monkeypatch.setattr(
        inference_module,
        "_select_run_ids",
        lambda source, *_args, **_kwargs: [0] if source == Agent.MBRL else [],
    )

    inference_module.run_inference_experiment(
        source_agents=[Agent.MBRL, Agent.QLearning],
        compare_to=[
            Agent.MBRL,
            EvaluatorSpec(agent=Agent.DQN, mode="pretrained"),
        ],
        maze_name="simple",
        num_datasets=1,
        observable=True,
        verbose=False,
        workers=10,
        device="cuda",
    )

    output = capsys.readouterr().out
    assert output.count("No trajectory files for q_learning") == 1
    assert calls == [
        {
            "tasks": [
                {
                    "compare_to": (EvaluatorSpec(agent=Agent.MBRL, mode="fresh"),),
                    "maze_name": "simple",
                    "source_context_mode": "legacy_context",
                    "horizon": None,
                }
            ],
            "uses_torch": False,
            "batch_label": "CPU-only",
            "device": "cuda",
        },
        {
            "tasks": [
                {
                    "compare_to": (
                        EvaluatorSpec(agent=Agent.DQN, mode="pretrained"),
                    ),
                    "maze_name": "simple",
                    "source_context_mode": "legacy_context",
                    "horizon": None,
                }
            ],
            "uses_torch": True,
            "batch_label": "neural",
            "device": "cuda",
        },
    ]


def test_registered_agents_include_neural_agents():
    assert Agent.DQN in registered_agents()
    assert Agent.ELMAN in registered_agents()
    assert Agent.GRU in registered_agents()
    assert Agent.LSTM in registered_agents()
    assert Agent.DRQN not in registered_agents()


@torch_required
def test_drqn_alias_maps_to_canonical_lstm_implementation():
    maze = Maze(load_builtin_maze_spec("simple"), seed=5)

    agent = get_agent(Agent.DRQN, maze, num_episodes=1, device="cpu", seed=11)

    assert canonical_agent(Agent.DRQN) == Agent.LSTM
    assert agent.agent_name == Agent.LSTM


def test_generate_trajectories_timing_summary_verbose_only(
    isolated_io_dirs: Path,
    capsys: pytest.CaptureFixture[str],
):
    generation_module.generate_trajectories(
        agent_type=Agent.QLearning,
        maze_name="simple",
        num_runs=1,
        num_episodes=1,
        observable=True,
        verbose=True,
        workers=1,
        base_seed=31,
    )
    verbose_output = capsys.readouterr().out
    assert "Timing Summary:" in verbose_output

    generation_module.generate_trajectories(
        agent_type=Agent.QLearning,
        maze_name="simple",
        num_runs=1,
        num_episodes=1,
        observable=True,
        verbose=False,
        workers=1,
        base_seed=31,
    )
    quiet_output = capsys.readouterr().out
    assert "Timing Summary:" not in quiet_output


def test_agent_seed_creates_reproducible_rng():
    maze = Maze(load_builtin_maze_spec("simple"), seed=7)
    agent_a = get_agent(Agent.MBRL, maze, num_episodes=1, seed=10_007)
    agent_b = get_agent(Agent.MBRL, maze, num_episodes=1, seed=10_007)

    assert agent_a.rng is not maze.rng
    assert agent_b.rng is not maze.rng
    assert agent_a.rng.random() == pytest.approx(agent_b.rng.random())


def test_resolve_device_auto_without_torch_defaults_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("forage_rl.utils.torch_support.torch_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_plot_aggregate_trajectory_stats_uses_run_datasets(
    isolated_io_dirs: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run_dataset = _fixed_horizon_run_dataset(num_episodes=2, horizon=100)
    io_module.save_run_dataset(run_dataset, Agent.MBRL, 0, "simple", True)
    io_module.save_run_dataset(run_dataset, Agent.MBRL, 1, "simple", True)

    monkeypatch.setattr(plots_module.plt, "show", lambda: None)

    plots_module.plot_aggregate_trajectory_stats(
        Agent.MBRL,
        maze_name="simple",
        observable=True,
        save=False,
        show=False,
    )


def test_resolve_aggregate_trajectory_cohort_prefers_longest_homogeneous_match(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        plots_module,
        "_list_run_ids_for_policy",
        lambda *args, **kwargs: list(range(100)),
    )
    monkeypatch.setattr(
        plots_module,
        "_load_run_dataset_metadata_for_policy",
        lambda policy, run_id, maze_name, observable: {
            "num_episodes": 100 if run_id < 8 else 6,
            "num_transitions": 10_000 if run_id < 8 else 600,
            "horizon": 100,
        },
    )

    cohort = plots_module._resolve_aggregate_trajectory_cohort(
        Agent.MBRL,
        "full",
        True,
    )

    assert cohort.run_ids == tuple(range(8))
    assert cohort.horizon == 100
    assert cohort.episodes_per_run == 100
    assert cohort.transitions_per_run == 10_000
    assert cohort.excluded_cohorts == ((92, 6, 600, 100),)
    assert cohort.auto_selected is True


def test_plot_aggregate_trajectory_stats_explicit_run_ids_require_one_homogeneous_cohort(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        plots_module,
        "_list_run_ids_for_policy",
        lambda *args, **kwargs: list(range(10)),
    )
    monkeypatch.setattr(
        plots_module,
        "_load_run_dataset_metadata_for_policy",
        lambda policy, run_id, maze_name, observable: {
            "num_episodes": 100 if run_id < 8 else 6,
            "num_transitions": 10_000 if run_id < 8 else 600,
            "horizon": 100,
        },
    )
    monkeypatch.setattr(
        plots_module,
        "_load_run_dataset_for_policy",
        lambda source, run_id, maze_name, observable, horizon=None: _fixed_horizon_run_dataset(
            num_episodes=100 if run_id < 8 else 6,
            horizon=100,
            offset=run_id,
        ),
    )

    captured: dict[str, object] = {}

    def fake_plot_mean_trajectory_stats(*args, **kwargs):
        captured["matched_run_ids"] = kwargs["matched_run_ids"]
        captured["episodes_per_run"] = kwargs["episodes_per_run"]
        captured["plotted_episodes"] = kwargs["plotted_episodes"]
        return None

    monkeypatch.setattr(plots_module, "plot_mean_trajectory_stats", fake_plot_mean_trajectory_stats)

    plots_module.plot_aggregate_trajectory_stats(
        Agent.MBRL,
        maze_name="simple",
        observable=True,
        run_ids=[1, 2],
        save=False,
        show=False,
    )

    assert captured["matched_run_ids"] == [1, 2]
    assert captured["episodes_per_run"] == 100
    assert captured["plotted_episodes"] == 200

    with pytest.raises(ValueError, match="homogeneous matched cohort"):
        plots_module.plot_aggregate_trajectory_stats(
            Agent.MBRL,
            maze_name="simple",
            observable=True,
            run_ids=[0, 8],
            save=False,
            show=False,
        )


def test_plot_aggregate_trajectory_stats_pools_episodes_across_runs(
    isolated_io_dirs: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run_dataset_a = _fixed_horizon_run_dataset(num_episodes=2, horizon=100, offset=0)
    run_dataset_b = _fixed_horizon_run_dataset(num_episodes=2, horizon=100, offset=1)
    io_module.save_run_dataset(run_dataset_a, Agent.MBRL, 0, "simple", True)
    io_module.save_run_dataset(run_dataset_b, Agent.MBRL, 1, "simple", True)

    captured: dict[str, object] = {}

    def fake_plot_mean_trajectory_stats(
        reward_sequences,
        state_sequences,
        maze,
        source,
        save=False,
        show=True,
        run_count=None,
        episodes_per_run=None,
        transitions_per_run=None,
        plotted_episodes=None,
        matched_run_ids=None,
        excluded_cohorts=None,
        setting_note=None,
        filename_suffix=None,
        benchmark_label=None,
        benchmark_note=None,
        timing_trajectories=None,
    ):
        captured["reward_lengths"] = [len(seq) for seq in reward_sequences]
        captured["state_lengths"] = [len(seq) for seq in state_sequences]
        captured["run_count"] = run_count
        captured["episodes_per_run"] = episodes_per_run
        captured["transitions_per_run"] = transitions_per_run
        captured["plotted_episodes"] = plotted_episodes
        captured["matched_run_ids"] = matched_run_ids
        captured["excluded_cohorts"] = excluded_cohorts
        captured["setting_note"] = setting_note
        captured["filename_suffix"] = filename_suffix
        captured["benchmark_label"] = benchmark_label
        captured["benchmark_note"] = benchmark_note
        captured["source"] = source
        captured["maze_states"] = maze.num_states
        captured["timing_trajectory_count"] = (
            None if timing_trajectories is None else len(timing_trajectories)
        )
        return None

    monkeypatch.setattr(plots_module, "plot_mean_trajectory_stats", fake_plot_mean_trajectory_stats)

    plots_module.plot_aggregate_trajectory_stats(
        Agent.MBRL,
        maze_name="simple",
        observable=True,
        save=False,
        show=False,
    )

    assert captured["reward_lengths"] == [100, 100, 100, 100]
    assert captured["state_lengths"] == [100, 100, 100, 100]
    assert captured["run_count"] == 2
    assert captured["episodes_per_run"] == 2
    assert captured["transitions_per_run"] == 200
    assert captured["plotted_episodes"] == 4
    assert captured["matched_run_ids"] == [0, 1]
    assert captured["excluded_cohorts"] == ()
    assert captured["setting_note"] is None
    assert captured["filename_suffix"] is None
    assert captured["benchmark_label"] is None
    assert captured["benchmark_note"] is None
    assert captured["source"] == Agent.MBRL
    assert captured["maze_states"] == 2
    assert captured["timing_trajectory_count"] == 4


def test_plot_aggregate_trajectory_stats_respects_custom_horizon(
    isolated_io_dirs: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run_dataset = _fixed_horizon_run_dataset(num_episodes=1, horizon=300)
    io_module.save_run_dataset(run_dataset, Agent.MBRL, 0, "simple", True, horizon=300)

    captured: dict[str, object] = {}

    def fake_plot_mean_trajectory_stats(
        reward_sequences,
        state_sequences,
        maze,
        source,
        **kwargs,
    ):
        captured["reward_lengths"] = [len(seq) for seq in reward_sequences]
        captured["maze_horizon"] = maze.horizon
        return None

    monkeypatch.setattr(plots_module, "plot_mean_trajectory_stats", fake_plot_mean_trajectory_stats)

    plots_module.plot_aggregate_trajectory_stats(
        Agent.MBRL,
        maze_name="simple",
        observable=True,
        save=False,
        show=False,
        horizon=300,
    )

    assert captured["reward_lengths"] == [300]
    assert captured["maze_horizon"] == 300


def test_plot_aggregate_trajectory_stats_uses_benchmark_suffix_and_notes(
    isolated_io_dirs: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run_dataset = _fixed_horizon_run_dataset(num_episodes=1, horizon=100)
    io_module.save_run_dataset(run_dataset, Agent.MBRL, 0, "full", True)
    saved_paths: list[Path] = []

    monkeypatch.setattr(
        plots_module.plt,
        "savefig",
        lambda path, *args, **kwargs: saved_paths.append(Path(path)),
    )

    plots_module.plot_aggregate_trajectory_stats(
        Agent.MBRL,
        maze_name="full",
        observable=True,
        save=True,
        show=False,
        filename_suffix="full_baseline",
        benchmark_label="Full Baseline Benchmark",
        benchmark_note="Suite role: baseline benchmark on full with all four agents.",
    )

    assert saved_paths == [
        isolated_io_dirs / "figures" / "mean_trajectory_stats_mbrl_full_FO_full_baseline.png"
    ]


def test_patch_state_indices_for_simple_and_full_mazes():
    simple_maze = Maze(load_builtin_maze_spec("simple"), seed=5)
    full_maze = Maze(load_builtin_maze_spec("full"), seed=5)

    simple_upper, simple_lower = plots_module._patch_state_indices(simple_maze)
    full_upper, full_lower = plots_module._patch_state_indices(full_maze)

    np.testing.assert_array_equal(simple_upper, np.array([0]))
    np.testing.assert_array_equal(simple_lower, np.array([1]))
    np.testing.assert_array_equal(full_upper, np.array([0, 1, 2]))
    np.testing.assert_array_equal(full_lower, np.array([3, 4, 5]))


def test_state_occupancy_fractions_match_expected_values():
    maze = Maze(load_builtin_maze_spec("simple"), seed=5)
    state_sequences = [
        np.array([0, 1, 1, 0]),
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 0, 1]),
        np.array([0, 0, 0, 0]),
    ]

    occupancy = plots_module._state_occupancy_fractions(state_sequences, maze)

    np.testing.assert_allclose(
        occupancy,
        np.array(
            [
                [1.0, 0.5, 0.5, 0.5],
                [0.0, 0.5, 0.5, 0.5],
            ]
        ),
    )


def test_cumulative_group_occupancy_shares_match_expected_values():
    state_sequences = [
        np.array([0, 3, 0, 3]),
        np.array([1, 1, 5, 5]),
    ]

    cumulative = plots_module._cumulative_group_occupancy_shares(
        state_sequences,
        [np.array([0, 1, 2]), np.array([3, 4, 5])],
    )

    np.testing.assert_allclose(
        cumulative,
        np.array(
            [
                [1.0, 0.75, 2.0 / 3.0, 0.5],
                [0.0, 0.25, 1.0 / 3.0, 0.5],
            ]
        ),
    )


def test_plot_mean_trajectory_stats_single_episode_uses_episode_aligned_axes():
    maze = Maze(load_builtin_maze_spec("simple"), seed=5)
    reward_sequences = [np.array([1.0, 0.0, 1.0])]
    state_sequences = [np.array([0, 0, 1])]

    fig = plots_module.plot_mean_trajectory_stats(
        reward_sequences,
        state_sequences,
        maze,
        Agent.MBRL,
        save=False,
        show=False,
        run_count=1,
        episodes_per_run=1,
        transitions_per_run=3,
        plotted_episodes=1,
        matched_run_ids=[0],
    )

    assert len(fig.axes[0].collections) == 0
    assert "Trajectory Overview" in fig._suptitle.get_text()
    assert "Source policy: MBRL" in fig._suptitle.get_text()
    assert any("Low sample size" in text.get_text() for text in fig.texts)
    assert any("fresh/pretrained" in text.get_text() for text in fig.texts)
    assert any("horizon=100" in text.get_text() for text in fig.texts)
    assert any(
        "runs=1, episodes_per_run=1, plotted_episodes=1" in text.get_text()
        for text in fig.texts
    )
    assert any("matched_run_ids=0" in text.get_text() for text in fig.texts)
    assert any(
        "Right panel shows cumulative within-episode residency share" in text.get_text()
        for text in fig.texts
    )
    assert any(
        "With few plotted episodes, early-step cumulative shares can still move in coarse increments."
        in text.get_text()
        for text in fig.texts
    )
    assert not any(
        "Occupancy percentages are quantized" in text.get_text() for text in fig.texts
    )
    assert fig.axes[1].get_title() == "Patch Cumulative Residency Share (single episode)"
    assert fig.axes[1].get_ylabel() == "Cumulative Share of Episode (%)"
    assert fig.axes[0].get_xlabel() == "Transition Within Episode"
    assert fig.axes[1].get_xlabel() == "Transition Within Episode"
    assert [line.get_label() for line in fig.axes[1].lines] == [
        "Upper Patch",
        "Lower Patch",
    ]
    np.testing.assert_allclose(
        fig.axes[1].lines[0].get_ydata(),
        np.array([100.0, 100.0, 200.0 / 3.0]),
    )
    np.testing.assert_allclose(
        fig.axes[1].lines[1].get_ydata(),
        np.array([0.0, 0.0, 100.0 / 3.0]),
    )


def test_plot_mean_trajectory_stats_uses_patch_cumulative_residency_for_full_maze():
    maze = Maze(load_builtin_maze_spec("full"), seed=5)
    reward_sequences = [
        np.array([1.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    state_sequences = [
        np.array([0, 3, 4]),
        np.array([1, 1, 5]),
    ]

    fig = plots_module.plot_mean_trajectory_stats(
        reward_sequences,
        state_sequences,
        maze,
        Agent.MBRL,
        save=False,
        show=False,
        run_count=2,
        episodes_per_run=1,
        transitions_per_run=3,
    )

    assert fig.axes[1].get_title() == "Patch Cumulative Residency Share (episodes=2)"
    assert fig.axes[1].get_ylabel() == "Cumulative Share of Episode (%)"
    assert fig.axes[0].get_xlabel() == "Transition Within Episode"
    assert fig.axes[1].get_xlabel() == "Transition Within Episode"
    assert [line.get_label() for line in fig.axes[1].lines] == [
        "Upper Patch",
        "Lower Patch",
    ]
    np.testing.assert_allclose(
        fig.axes[1].lines[0].get_ydata(),
        np.array([100.0, 75.0, 50.0]),
    )
    np.testing.assert_allclose(
        fig.axes[1].lines[1].get_ydata(),
        np.array([0.0, 25.0, 50.0]),
    )
    assert any(
        "Right panel shows cumulative within-episode residency share" in text.get_text()
        for text in fig.texts
    )
    assert any(
        "With few plotted episodes, early-step cumulative shares can still move in coarse increments."
        in text.get_text()
        for text in fig.texts
    )


def test_plot_mean_trajectory_stats_uses_observed_patch_cumulative_residency_for_full_pomdp():
    maze = MazePOMDP(load_builtin_maze_spec("full"), seed=5)
    reward_sequences = [
        np.array([1.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    state_sequences = [
        np.array([0, 1, 1]),
        np.array([1, 0, 1]),
    ]

    fig = plots_module.plot_mean_trajectory_stats(
        reward_sequences,
        state_sequences,
        maze,
        Agent.MBRL,
        save=False,
        show=False,
        run_count=2,
        episodes_per_run=1,
        transitions_per_run=3,
    )

    assert fig.axes[1].get_title() == "Observed Patch Cumulative Residency Share (episodes=2)"
    assert fig.axes[1].get_ylabel() == "Cumulative Share of Episode (%)"
    assert [line.get_label() for line in fig.axes[1].lines] == [
        "Observed Upper Patch",
        "Observed Lower Patch",
    ]
    np.testing.assert_allclose(
        fig.axes[1].lines[0].get_ydata(),
        np.array([50.0, 50.0, 100.0 / 3.0]),
    )
    np.testing.assert_allclose(
        fig.axes[1].lines[1].get_ydata(),
        np.array([50.0, 50.0, 200.0 / 3.0]),
    )


def test_plot_mean_trajectory_stats_adds_patch_timing_panels_when_trajectories_are_available():
    maze = Maze(load_builtin_maze_spec("simple"), seed=5)
    trajectory = _timed_trajectory(
        (0, 0, 1.0, 0, 0),
        (0, 1, 0.0, 1, 1),
        (1, 0, 0.0, 1, 0),
        (1, 1, 0.0, 0, 1),
    )

    fig = plots_module.plot_mean_trajectory_stats(
        [np.array([1.0, 0.0, 0.0, 0.0])],
        [np.array([0, 0, 1, 1])],
        maze,
        Agent.MBRL,
        save=False,
        show=False,
        run_count=1,
        episodes_per_run=1,
        transitions_per_run=4,
        timing_trajectories=[trajectory],
    )

    assert len(fig.axes) == 5
    assert fig.axes[2].get_title() == "Leave Probability by Time Spent in Patch"
    assert fig.axes[2].get_ylabel() == "Leave Probability"
    assert [line.get_label() for line in fig.axes[2].lines] == [
        "Upper Patch",
        "Lower Patch",
    ]
    assert fig.axes[3].get_title() == "Patch Residency vs MVT (AUC Overlay)"
    assert fig.axes[3].get_ylabel() == "Signed Dwell Deviation"
    assert fig.axes[4].get_ylabel() == "Normalized AUC"
    assert any(
        "Bottom-left panel shows leave probability conditioned on time spent in patch."
        in text.get_text()
        for text in fig.texts
    )
    assert any(
        "Bottom-right panel shows signed deviation from the MVT-optimal dwell and normalized leave-time AUC."
        in text.get_text()
        for text in fig.texts
    )


def test_plot_mean_trajectory_stats_annotates_simple_pomdp_as_equivalent_to_fo():
    maze = MazePOMDP(load_builtin_maze_spec("simple"), seed=5)
    fig = plots_module.plot_mean_trajectory_stats(
        [np.array([1.0, 0.0, 1.0])],
        [np.array([0, 1, 1])],
        maze,
        Agent.MBRL,
        save=False,
        show=False,
        run_count=1,
        episodes_per_run=1,
        transitions_per_run=3,
    )

    assert any("simple/PO is equivalent to simple/FO" in text.get_text() for text in fig.texts)


def test_plot_mean_trajectory_stats_uses_within_episode_axes_for_po_maze():
    maze = MazePOMDP(load_builtin_maze_spec("full"), seed=5)
    reward_sequences = [
        np.zeros(100, dtype=float),
        np.ones(100, dtype=float),
    ]
    state_sequences = [
        np.concatenate([np.zeros(50, dtype=int), np.ones(50, dtype=int)]),
        np.concatenate([np.ones(50, dtype=int), np.zeros(50, dtype=int)]),
    ]

    fig = plots_module.plot_mean_trajectory_stats(
        reward_sequences,
        state_sequences,
        maze,
        PolicySpec(agent=Agent.DQN, context_mode="observation_only"),
        save=False,
        show=False,
        run_count=2,
        episodes_per_run=1,
        transitions_per_run=100,
        plotted_episodes=2,
        matched_run_ids=[0, 1],
    )

    assert fig.axes[0].get_xlabel() == "Transition Within Episode"
    assert fig.axes[1].get_xlabel() == "Transition Within Episode"
    assert fig.axes[1].get_title() == "Observed Patch Cumulative Residency Share (episodes=2)"
    assert fig.axes[1].get_ylabel() == "Cumulative Share of Episode (%)"
    assert [line.get_label() for line in fig.axes[1].lines] == [
        "Observed Upper Patch",
        "Observed Lower Patch",
    ]
    assert len(fig.axes[1].collections) == 0
    assert any(
        "Right panel shows cumulative within-episode residency share" in text.get_text()
        for text in fig.texts
    )


def test_plot_episode_return_comparison_draws_agent_learning_curves(
    isolated_io_dirs: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(plots_module.plt, "show", lambda: None)

    io_module.save_run_dataset(
        RunDataset(
            trajectories=[
                _timed_trajectory((0, 0, 1.0, 0, 0)),
                _timed_trajectory((0, 0, 2.0, 0, 0)),
            ]
        ),
        Agent.MBRL,
        0,
        "simple",
        True,
    )
    io_module.save_run_dataset(
        RunDataset(
            trajectories=[
                _timed_trajectory((0, 0, 3.0, 0, 0)),
                _timed_trajectory((0, 0, 4.0, 0, 0)),
            ]
        ),
        Agent.MBRL,
        1,
        "simple",
        True,
    )
    io_module.save_run_dataset(
        RunDataset(
            trajectories=[
                _timed_trajectory((0, 0, 0.0, 0, 0)),
                _timed_trajectory((0, 0, 1.0, 0, 0)),
            ]
        ),
        Agent.DQN,
        0,
        "simple",
        True,
    )
    io_module.save_run_dataset(
        RunDataset(
            trajectories=[
                _timed_trajectory((0, 0, 2.0, 0, 0)),
                _timed_trajectory((0, 0, 3.0, 0, 0)),
            ]
        ),
        Agent.DQN,
        1,
        "simple",
        True,
    )

    fig = plots_module.plot_episode_return_comparison(
        maze_name="simple",
        observable=True,
        agents=[Agent.MBRL, Agent.DQN],
        save=False,
        show=False,
    )

    assert fig.axes[0].get_xlabel() == "Episode Within Run"
    assert fig.axes[0].get_ylabel() == "Episode Return"
    assert fig.axes[0].get_title() == (
        "Episode Return by Training Episode (simple, FO, runs=2, episodes_per_run=2)"
    )
    assert any("horizon=100" in text.get_text() for text in fig.texts)
    assert [line.get_label() for line in fig.axes[0].lines] == ["MBRL", "DQN"]
    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), np.array([2.0, 3.0]))
    np.testing.assert_allclose(fig.axes[0].lines[1].get_ydata(), np.array([1.0, 2.0]))


def test_plot_episode_return_comparison_uses_single_episode_diagnostic_mode(
    isolated_io_dirs: Path,
):
    for run_id, rewards in enumerate(([1.0], [3.0])):
        io_module.save_run_dataset(
            RunDataset(
                trajectories=[_timed_trajectory((0, 0, rewards[0], 0, 0))]
            ),
            Agent.MBRL,
            run_id,
            "simple",
            True,
            horizon=300,
        )
    for run_id, rewards in enumerate(([2.0], [4.0])):
        io_module.save_run_dataset(
            RunDataset(
                trajectories=[_timed_trajectory((0, 0, rewards[0], 0, 0))]
            ),
            Agent.DQN,
            run_id,
            "simple",
            True,
            horizon=300,
        )

    fig = plots_module.plot_episode_return_comparison(
        maze_name="simple",
        observable=True,
        agents=[Agent.MBRL, Agent.DQN],
        save=False,
        show=False,
        horizon=300,
    )

    assert fig.axes[0].get_title() == (
        "Episode-1 Return by Agent (simple, FO, runs=2, episodes_per_run=1)"
    )
    assert fig.axes[0].get_xticks().tolist() == [1]
    x_min, x_max = fig.axes[0].get_xlim()
    assert x_min < 1 < x_max
    assert any(
        "Single-episode diagnostic only" in text.get_text()
        for text in fig.texts
    )
    legend_text = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert legend_text == ["MBRL", "DQN"]


def test_plot_episode_return_comparison_distinguishes_context_mode_labels(
    isolated_io_dirs: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(plots_module.plt, "show", lambda: None)

    legacy_policy = PolicySpec(agent=Agent.DQN, context_mode="legacy_context")
    obs_only_policy = PolicySpec(agent=Agent.DQN, context_mode="observation_only")
    legacy_run_dataset = RunDataset(
        trajectories=[
            _timed_trajectory((0, 0, 1.0, 0, 0)),
            _timed_trajectory((0, 0, 2.0, 0, 0)),
        ]
    )
    obs_only_run_dataset = RunDataset(
        trajectories=[
            _timed_trajectory((0, 0, 3.0, 0, 0)),
            _timed_trajectory((0, 0, 4.0, 0, 0)),
        ]
    )

    io_module.save_run_dataset(
        legacy_run_dataset,
        Agent.DQN,
        0,
        "full",
        False,
        context_mode="legacy_context",
    )
    io_module.save_run_dataset(
        obs_only_run_dataset,
        Agent.DQN,
        0,
        "full",
        False,
        context_mode="observation_only",
    )

    fig = plots_module.plot_episode_return_comparison(
        maze_name="full",
        observable=False,
        agents=[obs_only_policy, legacy_policy],
        save=False,
        show=False,
    )

    assert [line.get_label() for line in fig.axes[0].lines] == [
        "DQN (obs-only)",
        "DQN (legacy-context)",
    ]
    assert fig.axes[0].lines[0].get_linestyle() == "-"
    assert fig.axes[0].lines[1].get_linestyle() == "--"


def test_plot_episode_return_comparison_full_context_benchmark_uses_neural_variants_only(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(plots_module, "list_run_dataset_run_ids", lambda *args, **kwargs: [0])
    monkeypatch.setattr(
        plots_module,
        "_load_run_dataset_metadata_for_policy",
        lambda *args, **kwargs: {
            "num_episodes": 2,
            "num_transitions": 2,
            "horizon": 100,
        },
    )
    monkeypatch.setattr(
        plots_module,
        "load_run_dataset",
        lambda *args, **kwargs: RunDataset(
            trajectories=[
                _timed_trajectory((0, 0, 1.0, 0, 0)),
                _timed_trajectory((0, 0, 2.0, 0, 0)),
            ]
        ),
    )

    fig = plots_module.plot_episode_return_comparison(
        maze_name="full",
        observable=False,
        agents=artifacts_module._neural_context_policies(),
        save=False,
        show=False,
        benchmark_label="Full Context Benchmark",
        benchmark_note="Suite role: neural context benchmark on full.",
    )

    assert [line.get_label() for line in fig.axes[0].lines] == [
        policy.display_label for policy in artifacts_module._neural_context_policies()
    ]
    assert "Full Context Benchmark" in fig.axes[0].get_title()
    assert any("neural context benchmark on full" in text.get_text() for text in fig.texts)


def test_plot_episode_return_comparison_uses_horizon_specific_filename_for_custom_horizon(
    isolated_io_dirs: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    saved_paths: list[Path] = []

    io_module.save_run_dataset(
        RunDataset(
            trajectories=[_timed_trajectory((0, 0, 1.0, 0, 0))]
        ),
        Agent.MBRL,
        0,
        "simple",
        True,
        horizon=300,
    )
    monkeypatch.setattr(
        plots_module.plt,
        "savefig",
        lambda path, *args, **kwargs: saved_paths.append(Path(path)),
    )

    plots_module.plot_episode_return_comparison(
        maze_name="simple",
        observable=True,
        agents=[Agent.MBRL],
        save=True,
        show=False,
        horizon=300,
    )

    assert saved_paths == [
        isolated_io_dirs / "figures" / "episode_return_comparison_simple_FO_h300.png"
    ]


def test_plot_episode_return_comparison_rejects_mixed_horizons(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(plots_module, "_list_run_ids_for_policy", lambda *args, **kwargs: [0, 1])
    monkeypatch.setattr(
        plots_module,
        "_load_run_dataset_metadata_for_policy",
        lambda _policy, run_id, *_args, **_kwargs: {
            "num_episodes": 2,
            "num_transitions": 2 if run_id == 0 else 6,
            "horizon": 100 if run_id == 0 else 300,
        },
    )
    monkeypatch.setattr(
        plots_module,
        "_load_run_dataset_for_policy",
        lambda *_args, **_kwargs: RunDataset(
            trajectories=[
                _timed_trajectory((0, 0, 1.0, 0, 0)),
                _timed_trajectory((0, 0, 2.0, 0, 0)),
            ]
        ),
    )

    with pytest.raises(ValueError, match="exact horizon"):
        plots_module.plot_episode_return_comparison(
            maze_name="simple",
            observable=True,
            agents=[Agent.MBRL],
            save=False,
            show=False,
        )


def test_running_win_rate_uses_cumulative_average(monkeypatch: pytest.MonkeyPatch):
    figure, axis = plots_module.plt.subplots()
    calls: list[tuple[Agent, Agent | EvaluatorSpec, int]] = []

    def fake_load_logprobs(source, evaluator, run_id, maze_name, observable):
        calls.append((source, evaluator, run_id))
        if evaluator == EvaluatorSpec(agent=Agent.MBRL, mode="fresh"):
            return np.array([1.0, 0.0, 3.0])
        return np.array([0.0, 1.0, 2.0])

    monkeypatch.setattr(plots_module, "load_logprobs", fake_load_logprobs)
    monkeypatch.setattr(plots_module, "list_run_dataset_run_ids", lambda *args, **kwargs: [0])

    plots_module._draw_running_win_rate(
        axis,
        Agent.MBRL,
        [EvaluatorSpec(agent=Agent.DQN, mode="fresh")],
        maze_name="simple",
        observable=True,
    )

    np.testing.assert_allclose(axis.lines[0].get_ydata(), np.array([1.0, 0.5, 2.0 / 3.0]))
    assert axis.get_ylabel() == "Source Lead Rate"
    assert axis.get_title() == "Source Running Lead Rate vs Evaluator (runs=1)"
    assert len(axis.texts) == 0
    legend_text = [text.get_text() for text in axis.get_legend().get_texts()]
    assert "Parity" in legend_text
    assert calls == [
        (Agent.MBRL, EvaluatorSpec(agent=Agent.MBRL, mode="fresh"), 0),
        (Agent.MBRL, EvaluatorSpec(agent=Agent.DQN, mode="fresh"), 0),
    ]


def test_mode_aware_evaluators_are_loaded_and_labeled(
    isolated_io_dirs: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run_dataset = RunDataset(
        trajectories=[_timed_trajectory((0, 0, 1.0, 0, 0), (0, 1, 0.0, 1, 1))]
    )
    io_module.save_run_dataset(run_dataset, Agent.MBRL, 0, "simple", True)

    _save_logprob_series(Agent.MBRL, Agent.MBRL, 0, [1.0, 2.0])
    _save_logprob_series(
        Agent.MBRL,
        EvaluatorSpec(agent=Agent.DQN, mode="fresh"),
        0,
        [0.5, 1.5],
    )
    _save_logprob_series(
        Agent.MBRL,
        EvaluatorSpec(agent=Agent.DQN, mode="pretrained"),
        0,
        [0.2, 1.8],
    )

    monkeypatch.setattr(plots_module.plt, "show", lambda: None)

    fig = plots_module.plot_aggregate_comparison(
        Agent.MBRL,
        [
            EvaluatorSpec(agent=Agent.DQN, mode="fresh"),
            EvaluatorSpec(agent=Agent.DQN, mode="pretrained"),
        ],
        maze_name="simple",
        observable=True,
        save=False,
        show=False,
    )

    legend_text = [text.get_text() for text in fig.axes[1].get_legend().get_texts()]
    x_tick_labels = [tick.get_text() for tick in fig.axes[0].get_xticklabels()]
    assert "DQN (fresh evaluator)" in legend_text
    assert "DQN (pretrained evaluator)" in legend_text
    assert "Parity" in legend_text
    assert any("DQN (fresh evaluator)" in label for label in x_tick_labels)
    assert any("DQN (pretrained evaluator)" in label for label in x_tick_labels)
    assert fig.axes[0].get_ylabel() == "Final Source Lead Rate"
    assert fig.axes[0].get_title() == "Final Source Lead Rate on 'MBRL' Trajectories (runs=1)"
    assert fig.axes[1].get_ylabel() == "Source Lead Rate"
    assert fig.axes[1].get_title() == "Source Running Lead Rate vs Evaluator (runs=1)"
    assert not any(
        "cumulative log-likelihood" in text.get_text() for text in fig.axes[1].texts
    )
    assert any(
        "cumulative log-likelihood" in text.get_text() for text in fig.texts
    )
    assert "Source-Centric Model Comparison on Saved Source Trajectories" in fig._suptitle.get_text()
    assert "Source: MBRL saved trajectory-generating policy" in fig._suptitle.get_text()
    assert any("Low sample size (runs=1)" in text.get_text() for text in fig.texts)


def test_plot_aggregate_comparison_labels_context_mode_variants(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        plots_module,
        "list_run_dataset_run_ids",
        lambda *args, **kwargs: [0],
    )

    def fake_load_logprobs(source, evaluator, run_id, maze_name, observable, **kwargs):
        spec = plots_module._normalize_evaluator(evaluator)
        if spec.context_mode == "observation_only":
            return np.array([1.0, 2.0, 3.0])
        return np.array([0.5, 1.5, 2.5])

    monkeypatch.setattr(plots_module, "load_logprobs", fake_load_logprobs)

    fig = plots_module.plot_aggregate_comparison(
        PolicySpec(agent=Agent.DQN, context_mode="observation_only"),
        [
            EvaluatorSpec(
                agent=Agent.LSTM,
                mode="fresh",
                context_mode="observation_only",
            ),
            EvaluatorSpec(
                agent=Agent.DQN,
                mode="fresh",
                context_mode="legacy_context",
            ),
        ],
        maze_name="full",
        observable=False,
        save=False,
        show=False,
    )

    legend_text = [text.get_text() for text in fig.axes[1].get_legend().get_texts()]
    assert "LSTM (obs-only, fresh evaluator)" in legend_text
    assert "DQN (legacy-context, fresh evaluator)" in legend_text
    assert "Source: DQN (obs-only) online-trained run policy" in fig._suptitle.get_text()


def test_plot_aggregate_comparison_suite_variants_use_short_filenames_and_labels(
    isolated_io_dirs: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    saved_paths: list[Path] = []

    monkeypatch.setattr(plots_module, "list_run_dataset_run_ids", lambda *args, **kwargs: [0])
    monkeypatch.setattr(
        plots_module,
        "load_logprobs",
        lambda *args, **kwargs: np.array([0.5, 1.0, 1.5]),
    )
    monkeypatch.setattr(
        plots_module.plt,
        "savefig",
        lambda path, *args, **kwargs: saved_paths.append(Path(path)),
    )

    full_fig = plots_module.plot_aggregate_comparison(
        PolicySpec(agent=Agent.DQN, context_mode="observation_only"),
        artifacts_module._neural_context_evaluators(),
        maze_name="full",
        observable=False,
        save=True,
        show=False,
        filename_suffix="full_context",
        benchmark_label="Full Context Benchmark",
        benchmark_note="Suite role: neural context benchmark on full.",
    )
    assert saved_paths == [
        isolated_io_dirs / "figures" / "agg_compare_dqn_obs_only_full_PO_full_context.png",
    ]
    assert "Full Context Benchmark" in full_fig._suptitle.get_text()
    assert "Source-Likelihood Diagnostic on Saved Source Trajectories" in full_fig._suptitle.get_text()
    assert any("neural context benchmark on full" in text.get_text() for text in full_fig.texts)


def test_plot_mean_trajectory_stats_uses_horizon_specific_filename_for_custom_horizon(
    isolated_io_dirs: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    saved_paths: list[Path] = []
    maze = Maze(load_builtin_maze_spec("simple"), seed=5, horizon=300)

    monkeypatch.setattr(
        plots_module.plt,
        "savefig",
        lambda path, *args, **kwargs: saved_paths.append(Path(path)),
    )

    plots_module.plot_mean_trajectory_stats(
        [np.array([1.0, 0.0, 1.0])],
        [np.array([0, 0, 1])],
        maze,
        Agent.MBRL,
        save=True,
        show=False,
        run_count=1,
        episodes_per_run=1,
        transitions_per_run=3,
        plotted_episodes=1,
    )

    assert saved_paths == [
        isolated_io_dirs / "figures" / "mean_trajectory_stats_mbrl_simple_FO_h300.png"
    ]


def test_plot_aggregate_comparison_uses_horizon_specific_filename_for_custom_horizon(
    isolated_io_dirs: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    saved_paths: list[Path] = []

    monkeypatch.setattr(plots_module, "list_run_dataset_run_ids", lambda *args, **kwargs: [0])
    monkeypatch.setattr(
        plots_module,
        "load_logprobs",
        lambda *args, **kwargs: np.array([0.5, 1.0, 1.5]),
    )
    monkeypatch.setattr(
        plots_module.plt,
        "savefig",
        lambda path, *args, **kwargs: saved_paths.append(Path(path)),
    )

    plots_module.plot_aggregate_comparison(
        Agent.MBRL,
        [EvaluatorSpec(agent=Agent.DQN, mode="fresh")],
        maze_name="simple",
        observable=True,
        save=True,
        show=False,
        horizon=300,
    )

    assert saved_paths == [
        isolated_io_dirs
        / "figures"
        / "agg_compare_mbrl_to_dqn_fresh_simple_FO_h300.png"
    ]


def test_plot_aggregate_comparison_describes_neural_source_policy(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(plots_module, "list_run_dataset_run_ids", lambda *args, **kwargs: [0])

    def fake_load_logprobs(source, evaluator, run_id, maze_name, observable):
        spec = plots_module._normalize_evaluator(evaluator)
        if spec == EvaluatorSpec(agent=Agent.DQN, mode="fresh"):
            return np.array([1.0, 2.0, 3.0])
        return np.array([0.5, 1.5, 2.5])

    monkeypatch.setattr(plots_module, "load_logprobs", fake_load_logprobs)

    fig = plots_module.plot_aggregate_comparison(
        Agent.DQN,
        [Agent.MBRL],
        maze_name="simple",
        observable=True,
        save=False,
        show=False,
    )

    assert "Source-Centric Model Comparison on Saved Source Trajectories" in fig._suptitle.get_text()
    assert "Source: DQN online-trained run policy" in fig._suptitle.get_text()


def test_bar_chart_treats_final_ties_as_half_wins(monkeypatch: pytest.MonkeyPatch):
    figure, axis = plots_module.plt.subplots()

    def fake_load_logprobs(source, evaluator, run_id, maze_name, observable):
        spec = plots_module._normalize_evaluator(evaluator)
        if spec == EvaluatorSpec(agent=Agent.MBRL, mode="fresh"):
            return (
                np.array([1.0, 2.0])
                if run_id == 0
                else np.array([1.0, 1.0])
            )
        return (
            np.array([0.0, 2.0])
            if run_id == 0
            else np.array([1.0, 3.0])
        )

    monkeypatch.setattr(plots_module, "load_logprobs", fake_load_logprobs)
    monkeypatch.setattr(plots_module, "list_run_dataset_run_ids", lambda *args, **kwargs: [0, 1])

    plots_module._draw_model_accuracies(
        axis,
        Agent.MBRL,
        [EvaluatorSpec(agent=Agent.DQN, mode="fresh")],
        maze_name="simple",
        observable=True,
    )

    assert axis.patches[0].get_height() == pytest.approx(0.25)
    assert axis.patches[1].get_height() == pytest.approx(0.75)
    assert axis.get_ylabel() == "Final Source Lead Rate"
    assert axis.get_title() == "Final Source Lead Rate on 'MBRL' Trajectories (runs=2)"
    legend_text = [text.get_text() for text in axis.get_legend().get_texts()]
    assert "Parity (0.50)" in legend_text


def test_plot_cumulative_sum_accuracy_uses_source_lead_rate_titles(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(plots_module, "list_run_dataset_run_ids", lambda *args, **kwargs: [0])

    def fake_load_logprobs(source, evaluator, run_id, maze_name, observable):
        spec = plots_module._normalize_evaluator(evaluator)
        if spec == EvaluatorSpec(agent=Agent.MBRL, mode="fresh"):
            return np.array([1.0, 2.0, 3.0])
        return np.array([0.5, 1.5, 2.5])

    monkeypatch.setattr(plots_module, "load_logprobs", fake_load_logprobs)

    fig = plots_module.plot_cumulative_sum_accuracy(
        Agent.MBRL,
        [EvaluatorSpec(agent=Agent.DQN, mode="fresh")],
        maze_name="simple",
        observable=True,
        save=False,
        show=False,
    )

    assert fig.axes[0].get_ylabel() == "Source Lead Rate"
    assert fig.axes[0].get_title() == (
        "Source Running Lead Rate over Observed Transitions "
        "(Source: MBRL saved trajectory-generating policy, runs=1)"
    )


def test_regenerate_artifacts_uses_headless_plotting(
    monkeypatch: pytest.MonkeyPatch,
):
    trajectory_calls: list[tuple[Agent, str, bool, bool, list[object] | None]] = []
    comparison_calls: list[tuple[Agent, str, bool, bool]] = []
    episode_return_calls: list[tuple[str, bool, bool]] = []

    monkeypatch.setattr(artifacts_module, "train_pretrained_agents", lambda *args, **kwargs: None)
    monkeypatch.setattr(artifacts_module, "run_generation_experiment", lambda *args, **kwargs: None)
    monkeypatch.setattr(artifacts_module, "run_inference_experiment", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        artifacts_module,
        "plot_aggregate_trajectory_stats",
        lambda source, maze_name, observable, save, show, cohort_policies=None, run_ids=None, filename_suffix=None, benchmark_label=None, benchmark_note=None, horizon=None: trajectory_calls.append(
            (source, maze_name, observable, show, cohort_policies)
        ),
    )
    monkeypatch.setattr(
        artifacts_module,
        "plot_aggregate_comparison",
        lambda source, evaluators, maze_name, observable, save, show, filename_suffix=None, benchmark_label=None, benchmark_note=None, horizon=None: comparison_calls.append(
            (source, maze_name, observable, show)
        ),
    )
    monkeypatch.setattr(
        artifacts_module,
        "plot_episode_return_comparison",
        lambda maze_name, observable, save, show, agents=None, filename_suffix=None, benchmark_label=None, benchmark_note=None, horizon=None: episode_return_calls.append(
            (maze_name, observable, show)
        ),
    )

    artifacts_module.regenerate_artifacts(
        mazes=["simple", "full"],
        observability="all",
        skip_generation=True,
        skip_inference=True,
        verbose=False,
    )

    assert {(maze_name, observable) for _, maze_name, observable, _, _ in trajectory_calls} == {
        ("simple", True),
        ("full", True),
        ("full", False),
    }
    assert {(maze_name, observable) for maze_name, observable, _ in episode_return_calls} == {
        ("simple", True),
        ("full", True),
        ("full", False),
    }
    assert all(show is False for *_, show, _ in trajectory_calls)
    assert all(show is False for *_, show in comparison_calls)
    assert all(show is False for *_, show in episode_return_calls)


def test_regenerate_artifacts_benchmark_suite_uses_expected_settings(
    monkeypatch: pytest.MonkeyPatch,
):
    generation_calls: list[dict[str, object]] = []
    inference_calls: list[dict[str, object]] = []
    episode_return_calls: list[dict[str, object]] = []
    trajectory_calls: list[dict[str, object]] = []
    comparison_calls: list[dict[str, object]] = []

    monkeypatch.setattr(artifacts_module, "train_pretrained_agents", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        artifacts_module,
        "run_generation_experiment",
        lambda *args, **kwargs: generation_calls.append(kwargs),
    )
    monkeypatch.setattr(
        artifacts_module,
        "run_inference_experiment",
        lambda *args, **kwargs: inference_calls.append(kwargs),
    )
    monkeypatch.setattr(
        artifacts_module,
        "plot_episode_return_comparison",
        lambda maze_name, observable, save, show, agents=None, filename_suffix=None, benchmark_label=None, benchmark_note=None, horizon=None: episode_return_calls.append(
            {
                "maze_name": maze_name,
                "observable": observable,
                "agents": agents,
                "filename_suffix": filename_suffix,
                "benchmark_label": benchmark_label,
                "benchmark_note": benchmark_note,
            }
        ),
    )
    monkeypatch.setattr(
        artifacts_module,
        "plot_aggregate_trajectory_stats",
        lambda source, maze_name, observable, save, show, cohort_policies=None, run_ids=None, filename_suffix=None, benchmark_label=None, benchmark_note=None, horizon=None: trajectory_calls.append(
            {
                "source": source,
                "maze_name": maze_name,
                "observable": observable,
                "cohort_policies": cohort_policies,
                "filename_suffix": filename_suffix,
                "benchmark_label": benchmark_label,
                "benchmark_note": benchmark_note,
            }
        ),
    )
    monkeypatch.setattr(
        artifacts_module,
        "plot_aggregate_comparison",
        lambda source, evaluators, maze_name, observable, save, show, filename_suffix=None, benchmark_label=None, benchmark_note=None, horizon=None: comparison_calls.append(
            {
                "source": source,
                "evaluators": evaluators,
                "maze_name": maze_name,
                "observable": observable,
                "filename_suffix": filename_suffix,
                "benchmark_label": benchmark_label,
                "benchmark_note": benchmark_note,
            }
        ),
    )

    artifacts_module.regenerate_artifacts(
        skip_generation=False,
        skip_inference=False,
        skip_figures=False,
        benchmark_suite=True,
        verbose=False,
    )

    assert [(call["maze_name"], call["observable"], call["context_mode"]) for call in generation_calls] == [
        ("full", True, "legacy_context"),
        ("full", False, "legacy_context"),
        ("full", True, "observation_only"),
        ("full", False, "observation_only"),
    ]
    assert generation_calls[0]["agent_types"] == artifacts_module._default_sources()
    assert generation_calls[2]["agent_types"] == neural_agents()

    assert [(call["maze_name"], call["observable"], call["source_context_mode"]) for call in inference_calls] == [
        ("full", True, "legacy_context"),
        ("full", False, "legacy_context"),
        ("full", True, "observation_only"),
        ("full", True, "legacy_context"),
        ("full", False, "observation_only"),
        ("full", False, "legacy_context"),
    ]

    assert episode_return_calls == [
        {
            "maze_name": "full",
            "observable": True,
            "agents": artifacts_module._default_sources(),
            "filename_suffix": "full_baseline",
            "benchmark_label": "Full Baseline Benchmark",
            "benchmark_note": "Suite role: baseline benchmark on full with all four agents.",
        },
        {
            "maze_name": "full",
            "observable": False,
            "agents": artifacts_module._default_sources(),
            "filename_suffix": "full_baseline",
            "benchmark_label": "Full Baseline Benchmark",
            "benchmark_note": "Suite role: baseline benchmark on full with all four agents.",
        },
        {
            "maze_name": "full",
            "observable": True,
            "agents": artifacts_module._neural_context_policies(),
            "filename_suffix": "full_context",
            "benchmark_label": "Full Context Benchmark",
            "benchmark_note": (
                "Suite role: neural context benchmark on full with obs-only and "
                "legacy-context DQN/ELMAN/GRU/LSTM variants."
            ),
        },
        {
            "maze_name": "full",
            "observable": False,
            "agents": artifacts_module._neural_context_policies(),
            "filename_suffix": "full_context",
            "benchmark_label": "Full Context Benchmark",
            "benchmark_note": (
                "Suite role: neural context benchmark on full with obs-only and "
                "legacy-context DQN/ELMAN/GRU/LSTM variants."
            ),
        },
    ]

    assert {(call["maze_name"], call["observable"], call["filename_suffix"]) for call in trajectory_calls} == {
        ("full", True, "full_baseline"),
        ("full", False, "full_baseline"),
        ("full", True, "full_context"),
        ("full", False, "full_context"),
    }
    assert any(
        call["cohort_policies"] == list(artifacts_module._default_sources())
        for call in trajectory_calls
        if (call["maze_name"], call["observable"], call["filename_suffix"])
        == ("full", True, "full_baseline")
    )
    assert any(
        call["cohort_policies"] == artifacts_module._neural_context_policies()
        for call in trajectory_calls
        if (call["maze_name"], call["observable"], call["filename_suffix"])
        == ("full", False, "full_context")
    )
    assert {(call["maze_name"], call["observable"], call["filename_suffix"]) for call in comparison_calls} == {
        ("full", True, "full_baseline"),
        ("full", False, "full_baseline"),
        ("full", True, "full_context"),
        ("full", False, "full_context"),
    }


def test_regenerate_artifacts_rejects_conflicting_presets():
    with pytest.raises(ValueError, match="mutually exclusive presets"):
        artifacts_module.regenerate_artifacts(
            reward_timing_benchmark=True,
            benchmark_suite=True,
            verbose=False,
        )


def test_regenerate_artifacts_uses_default_training_episode_budget(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    monkeypatch.setattr(artifacts_module, "train_pretrained_agents", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        artifacts_module,
        "run_generation_experiment",
        lambda *args, **kwargs: captured.update(num_episodes=kwargs["num_episodes"]),
    )
    monkeypatch.setattr(artifacts_module, "run_inference_experiment", lambda *args, **kwargs: None)

    artifacts_module.regenerate_artifacts(
        mazes=["simple"],
        observability="fo",
        skip_inference=True,
        skip_figures=True,
        verbose=False,
    )

    assert config_module.DefaultParams.NUM_TRAINING_EPISODES == 100
    assert captured["num_episodes"] == 100


def test_aggregate_plot_helpers_close_saved_figures(monkeypatch: pytest.MonkeyPatch):
    closed: list[object] = []

    monkeypatch.setattr(plots_module.plt, "savefig", lambda *args, **kwargs: None)
    monkeypatch.setattr(plots_module.plt, "close", lambda fig: closed.append(fig))
    monkeypatch.setattr(plots_module, "ensure_directories", lambda: None)
    monkeypatch.setattr(plots_module, "list_run_dataset_run_ids", lambda *args, **kwargs: [0])
    monkeypatch.setattr(
        plots_module,
        "load_run_dataset_metadata",
        lambda *args, **kwargs: {
            "num_episodes": 2,
            "num_transitions": 3,
            "horizon": 100,
        },
    )

    def fake_load_logprobs(source, evaluator, run_id, maze_name, observable):
        spec = plots_module._normalize_evaluator(evaluator)
        if spec == EvaluatorSpec(agent=Agent.MBRL, mode="fresh"):
            return np.array([1.0, 2.0])
        return np.array([0.5, 1.5])

    monkeypatch.setattr(plots_module, "load_logprobs", fake_load_logprobs)
    monkeypatch.setattr(plots_module, "load_run_dataset", lambda *args, **kwargs: _simple_run_dataset())

    plots_module.plot_mean_trajectory_stats(
        [np.array([1.0, 0.0, 1.0])],
        [np.array([0, 1, 1])],
        Maze(load_builtin_maze_spec("simple"), seed=5),
        Agent.MBRL,
        save=True,
        show=False,
        run_count=1,
        episodes_per_run=1,
        transitions_per_run=3,
    )
    plots_module.plot_aggregate_comparison(
        Agent.MBRL,
        [EvaluatorSpec(agent=Agent.DQN, mode="fresh")],
        maze_name="simple",
        observable=True,
        save=True,
        show=False,
    )
    plots_module.plot_episode_return_comparison(
        maze_name="simple",
        observable=True,
        agents=[Agent.MBRL],
        save=True,
        show=False,
    )

    assert len(closed) == 3


@torch_required
def test_neural_feature_encoding_uses_prev_action_and_reward_for_fo_and_po():
    fo_maze = Maze(load_builtin_maze_spec("simple"), seed=5)
    fo_agent = get_agent(Agent.DQN, fo_maze, num_episodes=1, device="cpu", seed=11)

    assert fo_agent.feature_dim == fo_agent.obs_dim + 1 + 1 + fo_maze.num_actions
    np.testing.assert_allclose(
        fo_agent.encode_feature_array(0, 0),
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        fo_agent.encode_feature_array(1, 2, prev_action=1, prev_reward=1.5),
        np.array([0.0, 1.0, 2.0 / 99.0, 1.5, 0.0, 1.0], dtype=np.float32),
    )

    po_maze = MazePOMDP(load_builtin_maze_spec("full"), seed=5)
    po_agent = get_agent(Agent.DQN, po_maze, num_episodes=1, device="cpu", seed=11)

    np.testing.assert_allclose(
        po_agent.encode_feature_array(1, 3, prev_action=0, prev_reward=-0.5),
        np.array([0.0, 1.0, 3.0 / 99.0, -0.5, 1.0, 0.0], dtype=np.float32),
    )


@torch_required
def test_neural_feature_encoding_supports_observation_only_context_mode():
    maze = MazePOMDP(load_builtin_maze_spec("full"), seed=5)
    agent = get_agent(
        Agent.DQN,
        maze,
        num_episodes=1,
        device="cpu",
        seed=11,
        context_mode="observation_only",
    )

    assert agent.context_mode == "observation_only"
    assert agent.feature_dim == agent.obs_dim
    assert agent.feature_schema_components == ("observation_one_hot",)
    np.testing.assert_allclose(
        agent.encode_feature_array(1, 3, prev_action=0, prev_reward=-0.5),
        np.array([0.0, 1.0], dtype=np.float32),
    )


@torch_required
def test_neural_agents_normalize_numpy_observation_dimensions_for_model_construction():
    maze = Maze(load_builtin_maze_spec("simple"), seed=5)
    maze.observation_space.n = np.int64(maze.observation_space.n)

    dqn_agent = get_agent(Agent.DQN, maze, num_episodes=1, device="cpu", seed=11)
    assert isinstance(dqn_agent.obs_dim, int)
    assert isinstance(dqn_agent.feature_dim, int)
    for agent_type in [*RECURRENT_AGENT_TYPES, Agent.DRQN]:
        recurrent_agent = get_agent(
            agent_type,
            maze,
            num_episodes=1,
            device="cpu",
            seed=11,
        )
        assert isinstance(recurrent_agent.obs_dim, int)
        assert isinstance(recurrent_agent.feature_dim, int)
        assert recurrent_agent.q_network.recurrent.input_size == recurrent_agent.feature_dim


@torch_required
def test_context_trace_tracks_episode_context_history():
    maze = Maze(load_builtin_maze_spec("simple"), seed=5)
    agent = get_agent(Agent.DQN, maze, num_episodes=1, device="cpu", seed=11)
    trajectory = _timed_trajectory(
        (0, 1, 1.0, 0, 0),
        (0, 0, 0.5, 1, 1),
    )

    rows = agent.context_trace(trajectory)

    assert [row["step_index"] for row in rows] == [0, 1]
    assert rows[0]["prev_action"] is None
    assert rows[0]["prev_reward"] == pytest.approx(0.0)
    np.testing.assert_allclose(
        np.asarray(rows[0]["encoded_feature"]),
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert rows[1]["prev_action"] == 1
    assert rows[1]["prev_reward"] == pytest.approx(1.0)
    np.testing.assert_allclose(
        np.asarray(rows[1]["encoded_feature"]),
        np.array([1.0, 0.0, 1.0 / 99.0, 1.0, 0.0, 1.0], dtype=np.float32),
    )


@torch_required
def test_dqn_train_returns_run_dataset_and_episode_simulate():
    maze = Maze(load_builtin_maze_spec("simple"), seed=5)
    agent = get_agent(Agent.DQN, maze, num_episodes=1, device="cpu", seed=11)

    run_dataset = agent.train(verbose=False)
    assert isinstance(run_dataset, RunDataset)
    assert run_dataset.num_episodes() == 1
    assert run_dataset.num_transitions() > 0

    episode = run_dataset.trajectories[0]
    assert len(agent.simulate(episode)) == len(episode)


@torch_required
def test_dqn_simulate_stores_prev_context_and_uses_action_reward_for_next_features():
    maze = Maze(load_builtin_maze_spec("simple"), seed=5)
    agent = get_agent(
        Agent.DQN,
        maze,
        num_episodes=1,
        device="cpu",
        seed=11,
        batch_size=2,
        warmup_steps=0,
    )
    first_episode = _timed_trajectory((0, 1, 1.0, 0, 0), (0, 0, 0.0, 1, 1))
    second_episode = _timed_trajectory((1, 0, 0.5, 1, 0))

    first_logs = agent.simulate(first_episode)
    second_logs = agent.simulate(second_episode)
    replay_entries = list(agent.replay_buffer)

    assert len(first_logs) == len(first_episode)
    assert len(second_logs) == len(second_episode)
    assert replay_entries[0]["prev_action"] is None
    assert replay_entries[0]["prev_reward"] == pytest.approx(0.0)
    assert replay_entries[1]["prev_action"] == 1
    assert replay_entries[1]["prev_reward"] == pytest.approx(1.0)
    assert replay_entries[2]["prev_action"] is None
    assert replay_entries[2]["prev_reward"] == pytest.approx(0.0)

    states, next_states, actions, rewards, dones = agent._build_batch_tensors(
        replay_entries[:2]
    )
    np.testing.assert_allclose(
        states[1].detach().cpu().numpy(),
        agent.encode_feature_array(0, 1, prev_action=1, prev_reward=1.0),
    )
    np.testing.assert_allclose(
        next_states[0].detach().cpu().numpy(),
        agent.encode_feature_array(0, 1, prev_action=1, prev_reward=1.0),
    )
    np.testing.assert_array_equal(actions.detach().cpu().numpy(), np.array([1, 0]))
    np.testing.assert_allclose(rewards.detach().cpu().numpy(), np.array([1.0, 0.0]))
    np.testing.assert_allclose(dones.detach().cpu().numpy(), np.array([0.0, 1.0]))


@torch_required
@pytest.mark.parametrize("agent_type", RECURRENT_AGENT_TYPES)
def test_recurrent_agents_reset_state_per_episode_and_persist_learning(
    agent_type: Agent,
):
    maze = Maze(load_builtin_maze_spec("simple"), seed=5)
    agent = get_agent(agent_type, maze, num_episodes=2, device="cpu", seed=11)

    run_dataset = agent.train(verbose=False)
    assert isinstance(run_dataset, RunDataset)
    assert run_dataset.num_episodes() == 2

    first_episode = run_dataset.trajectories[0]
    second_episode = run_dataset.trajectories[1]

    first_logs = agent.simulate(first_episode)
    replay_after_first = len(agent.replay_buffer)
    training_steps_after_first = agent.training_steps

    assert len(first_logs) == len(first_episode)
    assert agent.hidden is None

    second_logs = agent.simulate(second_episode)
    assert len(second_logs) == len(second_episode)
    assert agent.hidden is None
    assert len(agent.replay_buffer) >= replay_after_first
    assert agent.training_steps > training_steps_after_first


@torch_required
@pytest.mark.parametrize("agent_type", RECURRENT_AGENT_TYPES)
def test_recurrent_agents_batch_tensors_use_prev_context_and_reset_each_episode(
    agent_type: Agent,
):
    maze = Maze(load_builtin_maze_spec("simple"), seed=5)
    agent = get_agent(
        agent_type,
        maze,
        num_episodes=1,
        device="cpu",
        seed=11,
        batch_size=1,
    )
    first_episode = _timed_trajectory((0, 1, 1.0, 0, 0), (0, 0, 0.0, 1, 1))
    second_episode = _timed_trajectory((1, 0, 0.5, 1, 0))

    first_logs = agent.simulate(first_episode)
    first_replayed_episode = agent.replay_buffer[0]

    assert len(first_logs) == len(first_episode)
    assert first_replayed_episode[0]["prev_action"] is None
    assert first_replayed_episode[0]["prev_reward"] == pytest.approx(0.0)
    assert first_replayed_episode[1]["prev_action"] == 1
    assert first_replayed_episode[1]["prev_reward"] == pytest.approx(1.0)

    window = agent._sample_window_from_episode(first_replayed_episode)
    (
        features,
        next_features,
        actions,
        rewards,
        dones,
        valid_mask,
        loss_mask,
    ) = agent._build_batch_tensors(
        [window],
        include_masks=True,
    )
    np.testing.assert_allclose(
        features[0, 1].detach().cpu().numpy(),
        agent.encode_feature_array(0, 1, prev_action=1, prev_reward=1.0),
    )
    np.testing.assert_allclose(
        next_features[0, 0].detach().cpu().numpy(),
        agent.encode_feature_array(0, 1, prev_action=1, prev_reward=1.0),
    )
    np.testing.assert_array_equal(actions.detach().cpu().numpy(), np.array([[1, 0]]))
    np.testing.assert_allclose(rewards.detach().cpu().numpy(), np.array([[1.0, 0.0]]))
    np.testing.assert_allclose(dones.detach().cpu().numpy(), np.array([[0.0, 1.0]]))
    np.testing.assert_allclose(valid_mask.detach().cpu().numpy(), np.array([[1.0, 1.0]]))
    np.testing.assert_allclose(loss_mask.detach().cpu().numpy(), np.array([[1.0, 1.0]]))

    second_logs = agent.simulate(second_episode)
    assert len(second_logs) == len(second_episode)
    assert agent.replay_buffer[1][0]["prev_action"] is None
    assert agent.replay_buffer[1][0]["prev_reward"] == pytest.approx(0.0)
    assert agent.hidden is None


@torch_required
@pytest.mark.parametrize("agent_type", RECURRENT_AGENT_TYPES)
def test_recurrent_agents_single_episode_training_updates_weights_with_live_prefix(
    agent_type: Agent,
):
    maze = MazePOMDP(load_builtin_maze_spec("full"), seed=5, horizon=80)
    agent = get_agent(agent_type, maze, num_episodes=1, device="cpu", seed=11)
    before = {name: value.clone() for name, value in agent.q_network.state_dict().items()}

    run_dataset = agent.train(verbose=False)

    assert run_dataset.num_episodes() == 1
    assert run_dataset.num_transitions() == 80
    assert len(agent.replay_buffer) == 1
    assert _model_param_l1_delta(agent, before) > 0.0


@torch_required
@pytest.mark.parametrize("agent_type", RECURRENT_AGENT_TYPES)
def test_recurrent_agents_train_from_current_episode_before_finalization(
    agent_type: Agent,
):
    maze = Maze(load_builtin_maze_spec("simple"), seed=5)
    agent = get_agent(
        agent_type,
        maze,
        num_episodes=1,
        device="cpu",
        seed=11,
        batch_size=2,
    )
    agent._store_step(0, 0, None, 0.0, 1, 1.0, 0, 1, False)
    agent._store_step(0, 1, 1, 1.0, 0, 0.0, 1, 0, True)
    agent.training_steps = agent.batch_size
    before = {name: value.clone() for name, value in agent.q_network.state_dict().items()}

    windows = agent._sample_windows()
    agent._train_from_replay()

    assert len(agent.replay_buffer) == 0
    assert len(agent.current_episode) == 2
    assert len(windows) == agent.batch_size
    assert all(len(window["steps"]) == 2 for window in windows)
    assert all(window["loss_start_index"] == 0 for window in windows)
    assert _model_param_l1_delta(agent, before) > 0.0


@torch_required
@pytest.mark.parametrize("agent_type", RECURRENT_AGENT_TYPES)
def test_recurrent_agents_simulate_updates_from_live_episode_prefix(agent_type: Agent):
    maze = Maze(load_builtin_maze_spec("simple"), seed=5)
    agent = get_agent(
        agent_type,
        maze,
        num_episodes=1,
        device="cpu",
        seed=11,
        batch_size=2,
    )
    episode = _timed_trajectory((0, 1, 1.0, 0, 0), (0, 0, 0.0, 1, 1))
    before = {name: value.clone() for name, value in agent.q_network.state_dict().items()}

    logs = agent.simulate(episode)

    assert len(logs) == len(episode)
    assert len(agent.replay_buffer) == 1
    assert _model_param_l1_delta(agent, before) > 0.0


@torch_required
@pytest.mark.parametrize("agent_type", RECURRENT_AGENT_TYPES)
def test_recurrent_sample_window_uses_burn_in_prefix_when_available(agent_type: Agent):
    maze = Maze(load_builtin_maze_spec("simple"), seed=5)
    agent = get_agent(
        agent_type,
        maze,
        num_episodes=1,
        device="cpu",
        seed=11,
        batch_size=1,
        sequence_length=4,
        burn_in=2,
    )
    episode = _recurrent_replay_steps(8)

    window = agent._sample_window_from_episode(episode, learn_start=2)

    assert len(window["steps"]) == 6
    assert window["loss_start_index"] == 2
    assert [step["time_spent"] for step in window["steps"]] == [0, 1, 2, 3, 4, 5]
    assert window["steps"][0]["time_spent"] < window["steps"][2]["time_spent"]
    assert window["steps"][1]["time_spent"] < window["steps"][2]["time_spent"]


@torch_required
@pytest.mark.parametrize("agent_type", RECURRENT_AGENT_TYPES)
def test_recurrent_burn_in_truncates_cleanly_at_episode_start(agent_type: Agent):
    maze = Maze(load_builtin_maze_spec("simple"), seed=5)
    agent = get_agent(
        agent_type,
        maze,
        num_episodes=1,
        device="cpu",
        seed=11,
        batch_size=1,
        sequence_length=4,
        burn_in=2,
    )
    episode = _recurrent_replay_steps(5)

    window = agent._sample_window_from_episode(episode, learn_start=1)
    (
        _features,
        _next_features,
        _actions,
        _rewards,
        _dones,
        valid_mask,
        loss_mask,
    ) = agent._build_batch_tensors([window], include_masks=True)

    assert len(window["steps"]) == 5
    assert window["loss_start_index"] == 1
    assert float(loss_mask.sum().item()) > 0.0
    np.testing.assert_allclose(valid_mask.detach().cpu().numpy(), np.array([[1, 1, 1, 1, 1]]))
    np.testing.assert_allclose(loss_mask.detach().cpu().numpy(), np.array([[0, 1, 1, 1, 1]]))


@torch_required
@pytest.mark.parametrize("agent_type", RECURRENT_AGENT_TYPES)
def test_recurrent_burn_in_zero_preserves_full_loss_mask(agent_type: Agent):
    maze = Maze(load_builtin_maze_spec("simple"), seed=5)
    agent = get_agent(
        agent_type,
        maze,
        num_episodes=1,
        device="cpu",
        seed=11,
        batch_size=1,
        sequence_length=4,
        burn_in=0,
    )
    episode = _recurrent_replay_steps(8)

    window = agent._sample_window_from_episode(episode, learn_start=2)
    (
        _features,
        _next_features,
        _actions,
        _rewards,
        _dones,
        valid_mask,
        loss_mask,
    ) = agent._build_batch_tensors([window], include_masks=True)

    assert len(window["steps"]) == 4
    assert window["loss_start_index"] == 0
    np.testing.assert_allclose(valid_mask.detach().cpu().numpy(), np.array([[1, 1, 1, 1]]))
    np.testing.assert_allclose(loss_mask.detach().cpu().numpy(), valid_mask.detach().cpu().numpy())


@torch_required
def test_checkpoint_schema_round_trip_and_legacy_checkpoints_fail(
    isolated_io_dirs: Path,
):
    maze = Maze(load_builtin_maze_spec("simple"), seed=5)
    agent = get_agent(Agent.DQN, maze, num_episodes=1, device="cpu", seed=11)
    agent.training_steps = 7
    current_checkpoint = isolated_io_dirs / "current_dqn.pt"
    agent.save_checkpoint(current_checkpoint)

    checkpoint_payload = agent.torch.load(
        current_checkpoint,
        map_location=agent.torch_device,
        weights_only=False,
    )
    assert (
        checkpoint_payload["feature_schema_version"]
        == config_module.DefaultParams.NEURAL_FEATURE_SCHEMA_VERSION
    )
    assert checkpoint_payload["feature_dim"] == agent.feature_dim
    assert checkpoint_payload["feature_components"] == list(
        agent.feature_schema_components
    )
    assert checkpoint_payload["context_mode"] == "legacy_context"

    loaded_agent = get_agent(
        Agent.DQN,
        Maze(load_builtin_maze_spec("simple"), seed=5),
        num_episodes=1,
        device="cpu",
        seed=11,
        init_mode="pretrained",
        checkpoint_path=current_checkpoint,
    )
    assert loaded_agent.training_steps == 7

    legacy_checkpoint = isolated_io_dirs / "legacy_dqn.pt"
    agent.torch.save(
        {
            "model_state_dict": agent.q_network.state_dict(),
            "target_model_state_dict": agent.target_network.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "training_steps": 3,
        },
        legacy_checkpoint,
    )
    with pytest.raises(ValueError, match="Legacy checkpoint"):
        get_agent(
            Agent.DQN,
            Maze(load_builtin_maze_spec("simple"), seed=5),
            num_episodes=1,
            device="cpu",
            seed=11,
            init_mode="pretrained",
            checkpoint_path=legacy_checkpoint,
        )


@torch_required
def test_checkpoint_context_mode_must_match_loaded_agent(
    isolated_io_dirs: Path,
):
    maze = MazePOMDP(load_builtin_maze_spec("full"), seed=5)
    agent = get_agent(
        Agent.DQN,
        maze,
        num_episodes=1,
        device="cpu",
        seed=11,
        context_mode="observation_only",
    )
    checkpoint_path = isolated_io_dirs / "obs_only_dqn.pt"
    agent.save_checkpoint(checkpoint_path)

    checkpoint_payload = agent.torch.load(
        checkpoint_path,
        map_location=agent.torch_device,
        weights_only=False,
    )
    assert checkpoint_payload["context_mode"] == "observation_only"

    with pytest.raises(ValueError, match="context_mode=observation_only"):
        get_agent(
            Agent.DQN,
            MazePOMDP(load_builtin_maze_spec("full"), seed=5),
            num_episodes=1,
            device="cpu",
            seed=11,
            init_mode="pretrained",
            checkpoint_path=checkpoint_path,
            context_mode="legacy_context",
        )


@torch_required
def test_checkpoint_metadata_horizon_must_match_loaded_agent(
    isolated_io_dirs: Path,
):
    pretrain_module.train_pretrained_agents(
        agent_types=[Agent.DQN],
        maze_name="simple",
        num_episodes=1,
        observable=True,
        device="cpu",
        seed=0,
        verbose=False,
        horizon=300,
    )

    custom_checkpoint_path = io_module.checkpoint_path(
        Agent.DQN,
        "simple",
        True,
        horizon=300,
    )

    with pytest.raises(ValueError, match="horizon=300"):
        get_agent(
            Agent.DQN,
            Maze(load_builtin_maze_spec("simple"), seed=5),
            num_episodes=1,
            device="cpu",
            seed=11,
            init_mode="pretrained",
            checkpoint_path=custom_checkpoint_path,
        )


@torch_required
def test_inspect_neural_context_cli_prints_expected_fields(
    isolated_io_dirs: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    run_dataset = RunDataset(
        trajectories=[
            _timed_trajectory((0, 1, 1.0, 0, 0), (0, 0, 0.0, 1, 1)),
            _timed_trajectory((1, 0, 0.5, 1, 0)),
        ]
    )
    io_module.save_run_dataset(run_dataset, Agent.DQN, 0, "simple", True)

    rows = inspect_context_module.inspect_neural_context(
        agent_type=Agent.DQN,
        maze_name="simple",
        run_id=0,
        episode_index=0,
        observable=True,
        device="cpu",
    )
    assert rows[0]["prev_action"] is None
    assert rows[1]["prev_action"] == 1

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inspect_neural_context",
            "--agent",
            "dqn",
            "--maze",
            "simple",
            "--run-id",
            "0",
            "--episode-index",
            "0",
            "--steps",
            "2",
            "--device",
            "cpu",
        ],
    )
    inspect_context_module.main()
    output = capsys.readouterr().out

    assert "prev_action" in output
    assert "prev_reward" in output
    assert "feature" in output
    assert "start" in output
    assert "[1.000, 0.000, 0.000, 0.000, 0.000, 0.000]" in output


@torch_required
def test_inspect_neural_context_supports_observation_only_runs(
    isolated_io_dirs: Path,
):
    run_dataset = RunDataset(
        trajectories=[_timed_trajectory((0, 1, 1.0, 0, 0), (0, 0, 0.0, 1, 1))]
    )
    io_module.save_run_dataset(
        run_dataset,
        Agent.DQN,
        0,
        "full",
        False,
        context_mode="observation_only",
    )

    rows = inspect_context_module.inspect_neural_context(
        agent_type=Agent.DQN,
        maze_name="full",
        run_id=0,
        episode_index=0,
        observable=False,
        device="cpu",
        context_mode="observation_only",
    )

    np.testing.assert_allclose(
        np.asarray(rows[0]["encoded_feature"]),
        np.array([1.0, 0.0], dtype=np.float32),
    )


@torch_required
def test_pretrained_checkpoint_and_inference_paths_are_distinct(
    isolated_io_dirs: Path,
):
    pretrain_module.train_pretrained_agents(
        agent_types=[Agent.DQN],
        maze_name="simple",
        num_episodes=1,
        observable=True,
        device="cpu",
        seed=0,
        verbose=False,
    )

    run_dataset = RunDataset(
        trajectories=[Trajectory(transitions=[TimedTransition(
            state=0,
            action=0,
            reward=1.0,
            next_state=0,
            time_spent=0,
        )])]
    )
    io_module.save_run_dataset(run_dataset, Agent.MBRL, 0, "simple", True)

    inference_module.run_inference_experiment(
        source_agents=[Agent.MBRL],
        compare_to=[
            EvaluatorSpec(agent=Agent.DQN, mode="fresh"),
            EvaluatorSpec(agent=Agent.DQN, mode="pretrained"),
        ],
        maze_name="simple",
        num_datasets=1,
        observable=True,
        verbose=False,
        workers=1,
        device="cpu",
        base_seed=0,
    )

    assert {path.name for path in config_module.LOGPROBS_DIR.glob("*.npy")} == {
        "simple_FO_source_mbrl_eval_dqn_fresh_log_likelihoods_0.npy",
        "simple_FO_source_mbrl_eval_dqn_pretrained_log_likelihoods_0.npy",
    }
    assert io_module.checkpoint_path(Agent.DQN, "simple", True).exists()
    assert io_module.checkpoint_metadata_path(Agent.DQN, "simple", True).exists()
    metadata = json.loads(
        io_module.checkpoint_metadata_path(Agent.DQN, "simple", True).read_text(
            encoding="utf-8"
        )
    )
    assert (
        metadata["feature_schema_version"]
        == config_module.DefaultParams.NEURAL_FEATURE_SCHEMA_VERSION
    )
    assert metadata["horizon"] == 100
    assert metadata["feature_dim"] > 0
    assert metadata["feature_components"] == [
        "observation_one_hot",
        "normalized_time_spent",
        "prev_reward",
        "prev_action_one_hot",
    ]


@torch_required
def test_obs_only_pretrained_checkpoint_and_inference_paths_are_distinct(
    isolated_io_dirs: Path,
):
    pretrain_module.train_pretrained_agents(
        agent_types=[Agent.DQN],
        maze_name="full",
        num_episodes=1,
        observable=False,
        device="cpu",
        seed=0,
        verbose=False,
        context_mode="observation_only",
    )

    run_dataset = RunDataset(
        trajectories=[
            Trajectory(
                transitions=[
                    TimedTransition(
                        state=0,
                        action=0,
                        reward=1.0,
                        next_state=0,
                        time_spent=0,
                    )
                ]
            )
        ]
    )
    io_module.save_run_dataset(
        run_dataset,
        Agent.DQN,
        0,
        "full",
        False,
        context_mode="observation_only",
    )

    inference_module.run_inference_experiment(
        source_agents=[Agent.DQN],
        compare_to=[
            EvaluatorSpec(
                agent=Agent.DQN,
                mode="fresh",
                context_mode="observation_only",
            ),
            EvaluatorSpec(
                agent=Agent.DQN,
                mode="pretrained",
                context_mode="observation_only",
            ),
        ],
        maze_name="full",
        num_datasets=1,
        observable=False,
        verbose=False,
        workers=1,
        device="cpu",
        base_seed=0,
        source_context_mode="observation_only",
    )

    assert {path.name for path in config_module.LOGPROBS_DIR.glob("*.npy")} == {
        "full_PO_source_dqn_obs_only_eval_dqn_obs_only_fresh_log_likelihoods_0.npy",
        "full_PO_source_dqn_obs_only_eval_dqn_obs_only_pretrained_log_likelihoods_0.npy",
    }
    assert (
        io_module.checkpoint_path(
            Agent.DQN,
            "full",
            False,
            context_mode="observation_only",
        ).exists()
    )
    metadata = json.loads(
        io_module.checkpoint_metadata_path(
            Agent.DQN,
            "full",
            False,
            context_mode="observation_only",
        ).read_text(encoding="utf-8")
    )
    assert metadata["context_mode"] == "observation_only"
    assert metadata["horizon"] == 100


def test_inference_supports_custom_horizon_trajectories(
    isolated_io_dirs: Path,
):
    run_dataset = RunDataset(
        trajectories=[
            Trajectory(
                transitions=[
                    TimedTransition(
                        state=0,
                        action=0,
                        reward=1.0,
                        next_state=0,
                        time_spent=step,
                    )
                    for step in range(300)
                ]
            )
        ]
    )
    io_module.save_run_dataset(
        run_dataset,
        Agent.MBRL,
        0,
        "simple",
        True,
        horizon=300,
    )

    inference_module.run_inference_experiment(
        source_agents=[Agent.MBRL],
        compare_to=[Agent.MBRL],
        maze_name="simple",
        num_datasets=1,
        observable=True,
        verbose=False,
        workers=1,
        device="cpu",
        base_seed=0,
        horizon=300,
    )

    assert {path.name for path in config_module.LOGPROBS_DIR.glob("*.npy")} == {
        "simple_FO_h300_source_mbrl_eval_mbrl_fresh_log_likelihoods_0.npy",
    }


def test_inference_rejects_horizon_metadata_mismatch(isolated_io_dirs: Path):
    run_dataset = RunDataset(
        trajectories=[_timed_trajectory((0, 0, 1.0, 0, 0))]
    )
    dataset_path = io_module.save_run_dataset(run_dataset, Agent.MBRL, 0, "simple", True)
    metadata_path = dataset_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["horizon"] = 300
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="requested setting"):
        inference_module._evaluate_dataset_task(
            (
                Agent.MBRL,
                0,
                "simple",
                (EvaluatorSpec(agent=Agent.MBRL, mode="fresh"),),
                True,
                "cpu",
                0,
                "legacy_context",
                None,
            )
        )


def test_agent_parsers_accept_recurrent_names_and_drqn_alias():
    assert generation_module._parse_agents(["all"]) == registered_agents()
    assert generation_module._parse_agents(["elman", "gru", "lstm", "drqn"]) == [
        Agent.ELMAN,
        Agent.GRU,
        Agent.LSTM,
        Agent.DRQN,
    ]

    assert pretrain_module._parse_agents(["all"]) == neural_agents()
    assert pretrain_module._parse_agents(["dqn", "elman", "gru", "lstm", "drqn"]) == [
        Agent.DQN,
        Agent.ELMAN,
        Agent.GRU,
        Agent.LSTM,
        Agent.DRQN,
    ]

    assert inference_module._parse_agents(["all"]) == registered_agents()
    assert inference_module._parse_evaluators(
        ["elman:fresh", "gru:pretrained", "lstm:pretrained", "drqn:fresh"],
        "observation_only",
    ) == [
        EvaluatorSpec(
            agent=Agent.ELMAN,
            mode="fresh",
            context_mode="observation_only",
        ),
        EvaluatorSpec(
            agent=Agent.GRU,
            mode="pretrained",
            context_mode="observation_only",
        ),
        EvaluatorSpec(
            agent=Agent.LSTM,
            mode="pretrained",
            context_mode="observation_only",
        ),
        EvaluatorSpec(
            agent=Agent.DRQN,
            mode="fresh",
            context_mode="observation_only",
        ),
    ]


def test_generate_trajectories_main_accepts_horizon_flag(
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
            "mbrl",
            "--maze",
            "simple",
            "--horizon",
            "300",
            "--quiet",
        ],
    )

    generation_module.main()

    assert captured["horizon"] == 300


def test_model_inference_main_accepts_horizon_flag(
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
            "mbrl",
            "--compare-to",
            "mbrl",
            "--maze",
            "simple",
            "--horizon",
            "300",
            "--quiet",
        ],
    )

    inference_module.main()

    assert captured["horizon"] == 300


def test_train_pretrained_main_accepts_horizon_flag(
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
            "--maze",
            "simple",
            "--horizon",
            "300",
            "--quiet",
        ],
    )

    pretrain_module.main()

    assert captured["horizon"] == 300


def test_regenerate_artifacts_main_accepts_horizon_flag(
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
            "--mazes",
            "simple",
            "--observability",
            "fo",
            "--horizon",
            "300",
            "--quiet",
        ],
    )

    artifacts_module.main()

    assert captured["horizon"] == 300


@pytest.mark.parametrize(
    ("agent_arg", "expected_agent"),
    [
        ("elman", Agent.ELMAN),
        ("gru", Agent.GRU),
        ("lstm", Agent.LSTM),
        ("drqn", Agent.DRQN),
    ],
)
def test_inspect_neural_context_cli_accepts_recurrent_agent_tokens(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    agent_arg: str,
    expected_agent: Agent,
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
                "encoded_feature": np.array([1.0], dtype=np.float32),
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
            agent_arg,
            "--maze",
            "simple",
            "--run-id",
            "0",
            "--episode-index",
            "0",
        ],
    )

    inspect_context_module.main()

    assert captured["agent_type"] == expected_agent
    assert "feature" in capsys.readouterr().out


def test_lstm_artifacts_use_canonical_names_when_saved_from_drqn_alias(
    isolated_io_dirs: Path,
):
    run_dataset = RunDataset(trajectories=[_timed_trajectory((0, 0, 1.0, 0, 0))])

    dataset_path = io_module.save_run_dataset(run_dataset, Agent.DRQN, 0, "simple", True)
    logprob_path = io_module.save_logprobs(
        np.array([1.0, 2.0], dtype=float),
        Agent.DRQN,
        EvaluatorSpec(agent=Agent.DRQN, mode="fresh"),
        0,
        "simple",
        True,
    )

    assert dataset_path.name == "simple_FO_lstm_run_dataset_0.npz"
    assert json.loads(dataset_path.with_suffix(".json").read_text(encoding="utf-8"))["agent"] == "lstm"
    assert logprob_path.name == "simple_FO_source_lstm_eval_lstm_fresh_log_likelihoods_0.npy"
    assert io_module.checkpoint_path(Agent.DRQN, "simple", True).name == "simple_FO_lstm_final.pt"
    assert io_module.checkpoint_metadata_path(Agent.DRQN, "simple", True).name == "simple_FO_lstm_final.json"


def test_lstm_loads_fall_back_to_legacy_drqn_artifacts(
    isolated_io_dirs: Path,
):
    config_module.ensure_directories()
    legacy_dataset_path = (
        config_module.TRAJECTORIES_DIR / "simple_FO_drqn_run_dataset_0.npz"
    )
    np.savez(
        legacy_dataset_path,
        __transition_type__=np.array("TimedTransition", dtype=np.str_),
        episode_00000=_timed_trajectory((0, 0, 1.0, 0, 0)).to_numpy(),
    )
    legacy_logprob_path = (
        config_module.LOGPROBS_DIR
        / "simple_FO_source_drqn_eval_drqn_fresh_log_likelihoods_0.npy"
    )
    np.save(legacy_logprob_path, np.array([0.5, 1.5], dtype=float))
    legacy_checkpoint_path = config_module.CHECKPOINTS_DIR / "simple_FO_drqn_final.pt"
    legacy_checkpoint_path.write_bytes(b"legacy")

    loaded_run_dataset = io_module.load_run_dataset(Agent.LSTM, 0, "simple", True)
    loaded_logprobs = io_module.load_logprobs(
        Agent.LSTM,
        EvaluatorSpec(agent=Agent.LSTM, mode="fresh"),
        0,
        "simple",
        True,
    )

    assert loaded_run_dataset.num_episodes() == 1
    np.testing.assert_allclose(loaded_logprobs, np.array([0.5, 1.5], dtype=float))
    assert (
        io_module.resolve_checkpoint_load_path(Agent.LSTM, "simple", True)
        == legacy_checkpoint_path
    )
