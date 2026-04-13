import sys

import pytest

import forage_rl.experiments.regenerate_artifacts as artifacts_module
from forage_rl.agents.registry import Agent, EvaluatorSpec


def _expected_sources() -> tuple[Agent, ...]:
    return (
        Agent.MBRL,
        Agent.QLearning,
        Agent.DQN,
        Agent.ELMAN,
        Agent.GRU,
        Agent.LSTM,
    )


def _expected_fresh_evaluators() -> tuple[Agent | EvaluatorSpec, ...]:
    return (
        Agent.MBRL,
        Agent.QLearning,
        EvaluatorSpec(agent=Agent.DQN, mode="fresh"),
        EvaluatorSpec(agent=Agent.ELMAN, mode="fresh"),
        EvaluatorSpec(agent=Agent.GRU, mode="fresh"),
        EvaluatorSpec(agent=Agent.LSTM, mode="fresh"),
    )


def _expected_pretrained_evaluators() -> tuple[Agent | EvaluatorSpec, ...]:
    return (
        Agent.MBRL,
        Agent.QLearning,
        EvaluatorSpec(agent=Agent.DQN, mode="pretrained"),
        EvaluatorSpec(agent=Agent.ELMAN, mode="pretrained"),
        EvaluatorSpec(agent=Agent.GRU, mode="pretrained"),
        EvaluatorSpec(agent=Agent.LSTM, mode="pretrained"),
    )


def test_default_full_scenarios_keep_six_canonical_source_agents():
    scenarios = artifacts_module._default_scenarios(
        mazes=["full"],
        observability="all",
    )

    assert len(scenarios) == 2
    assert {scenario.observable for scenario in scenarios} == {True, False}
    for scenario in scenarios:
        assert scenario.maze_name == "full"
        assert scenario.source_agents == _expected_sources()
        assert scenario.figure_policies == _expected_sources()


def test_filter_evaluators_preserves_default_set_in_all_mode():
    evaluators = tuple(artifacts_module._default_evaluators())

    assert artifacts_module._filter_evaluators(evaluators, "all") == evaluators


def test_filter_evaluators_keeps_tabular_and_fresh_neural_only():
    evaluators = tuple(artifacts_module._default_evaluators())

    assert artifacts_module._filter_evaluators(evaluators, "fresh") == (
        _expected_fresh_evaluators()
    )


def test_filter_evaluators_keeps_tabular_and_pretrained_neural_only():
    evaluators = tuple(artifacts_module._default_evaluators())

    assert artifacts_module._filter_evaluators(evaluators, "pretrained") == (
        _expected_pretrained_evaluators()
    )


def test_regenerate_artifacts_rejects_fresh_mode_with_pretraining():
    with pytest.raises(ValueError, match="Fresh-only artifact regeneration"):
        artifacts_module.regenerate_artifacts(
            evaluator_mode="fresh",
            train_pretrained=True,
            skip_generation=True,
            skip_inference=True,
            skip_figures=True,
            verbose=False,
        )


def test_regenerate_artifacts_fresh_mode_filters_inference_and_plot_comparisons(
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
        lambda **kwargs: episode_return_calls.append(kwargs),
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
                "horizon": horizon,
            }
        ),
    )
    monkeypatch.setattr(
        artifacts_module,
        "plot_aggregate_comparison",
        lambda source, compare_to, maze_name, observable, save, show, filename_suffix=None, benchmark_label=None, benchmark_note=None, horizon=None: comparison_calls.append(
            {
                "source": source,
                "compare_to": compare_to,
                "maze_name": maze_name,
                "observable": observable,
                "horizon": horizon,
            }
        ),
    )

    artifacts_module.regenerate_artifacts(
        mazes=["full"],
        observability="all",
        num_runs=2,
        num_episodes=1,
        num_datasets=2,
        horizon=600,
        evaluator_mode="fresh",
        device="cpu",
        verbose=False,
    )

    expected_sources = list(_expected_sources())
    expected_evaluators = list(_expected_fresh_evaluators())

    assert len(generation_calls) == 2
    assert all(call["agent_types"] == expected_sources for call in generation_calls)
    assert all(call["horizon"] == 600 for call in generation_calls)

    assert len(inference_calls) == 2
    assert all(call["source_agents"] == expected_sources for call in inference_calls)
    assert all(call["compare_to"] == expected_evaluators for call in inference_calls)
    assert all(call["num_datasets"] == 2 for call in inference_calls)
    assert all(call["horizon"] == 600 for call in inference_calls)

    assert len(episode_return_calls) == 2
    assert all(call["agents"] == expected_sources for call in episode_return_calls)
    assert all(call["horizon"] == 600 for call in episode_return_calls)

    assert len(trajectory_calls) == len(expected_sources) * 2
    assert all(call["cohort_policies"] == expected_sources for call in trajectory_calls)
    assert all(call["horizon"] == 600 for call in trajectory_calls)

    assert len(comparison_calls) == len(expected_sources) * 2
    assert all(call["compare_to"] == expected_evaluators for call in comparison_calls)
    assert all(call["horizon"] == 600 for call in comparison_calls)


@pytest.mark.parametrize("mode", ["all", "fresh", "pretrained"])
def test_regenerate_artifacts_main_accepts_evaluator_mode_flag(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
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
            "full",
            "--observability",
            "all",
            "--evaluator-mode",
            mode,
            "--quiet",
        ],
    )

    artifacts_module.main()

    assert captured["evaluator_mode"] == mode


def test_regenerate_artifacts_main_rejects_train_pretrained_in_fresh_mode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "regenerate_artifacts",
            "--mazes",
            "full",
            "--observability",
            "all",
            "--evaluator-mode",
            "fresh",
            "--train-pretrained",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        artifacts_module.main()

    assert (
        "--train-pretrained cannot be used with --evaluator-mode fresh"
        in capsys.readouterr().err
    )
