"""Scenario definitions for artifact regeneration workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from forage_rl.agents.registry import (
    Agent,
    EvaluatorSpec,
    NeuralContextMode,
    PolicySpec,
    neural_agents,
    recurrent_agents,
)
from forage_rl.experiments.reward_timing_benchmark import reward_timing_policies

EvaluatorMode = Literal["all", "fresh", "pretrained"]


@dataclass(frozen=True)
class ArtifactScenario:
    maze_name: str
    observable: bool
    source_agents: tuple[Agent, ...]
    evaluators: tuple[Agent | EvaluatorSpec, ...]
    figure_policies: tuple[Agent | PolicySpec, ...]
    train_context_modes: tuple[NeuralContextMode, ...]
    generation_context_modes: tuple[NeuralContextMode, ...]
    inference_context_modes: tuple[NeuralContextMode, ...]
    filename_suffix: str | None = None
    benchmark_label: str | None = None
    benchmark_note: str | None = None


def default_sources() -> list[Agent]:
    return [Agent.MBRL, Agent.QLearning, Agent.DQN, *recurrent_agents()]


def default_evaluators() -> list[Agent | EvaluatorSpec]:
    evaluators: list[Agent | EvaluatorSpec] = [
        Agent.MBRL,
        Agent.QLearning,
    ]
    for agent in neural_agents():
        evaluators.append(EvaluatorSpec(agent=agent, mode="fresh"))
        evaluators.append(EvaluatorSpec(agent=agent, mode="pretrained"))
    return evaluators


def neural_context_policies() -> list[PolicySpec]:
    policies: list[PolicySpec] = []
    for context_mode in ("observation_only", "legacy_context"):
        for agent in neural_agents():
            policies.append(PolicySpec(agent=agent, context_mode=context_mode))
    return policies


def neural_context_evaluators() -> list[EvaluatorSpec]:
    evaluators: list[EvaluatorSpec] = []
    for mode in ("fresh", "pretrained"):
        for context_mode in ("observation_only", "legacy_context"):
            for agent in neural_agents():
                evaluators.append(
                    EvaluatorSpec(
                        agent=agent,
                        mode=mode,
                        context_mode=context_mode,
                    )
                )
    return evaluators


def reward_timing_evaluators() -> list[EvaluatorSpec]:
    evaluators: list[EvaluatorSpec] = []
    for mode in ("fresh", "pretrained"):
        for context_mode in ("prev_reward", "prev_reward_time"):
            for agent in neural_agents():
                evaluators.append(
                    EvaluatorSpec(
                        agent=agent,
                        mode=mode,
                        context_mode=context_mode,
                    )
                )
    return evaluators


def filter_evaluators(
    evaluators: tuple[Agent | EvaluatorSpec, ...],
    evaluator_mode: EvaluatorMode,
) -> tuple[Agent | EvaluatorSpec, ...]:
    """Filter neural evaluator modes while preserving tabular baselines."""
    if evaluator_mode == "all":
        return evaluators

    filtered: list[Agent | EvaluatorSpec] = []
    for evaluator in evaluators:
        if isinstance(evaluator, Agent):
            filtered.append(evaluator)
            continue
        if evaluator.mode == evaluator_mode:
            filtered.append(evaluator)
    return tuple(filtered)


_SUPPORTED_SETTINGS: tuple[tuple[str, bool], ...] = (
    ("simple", True),
    ("full", True),
    ("full", False),
)


def selected_settings(
    mazes: list[str] | None = None,
    observability: str = "all",
    *,
    verbose: bool = False,
) -> list[tuple[str, bool]]:
    requested_mazes = ["simple", "full"] if mazes is None else mazes
    selected = [
        (maze_name, observable)
        for maze_name, observable in _SUPPORTED_SETTINGS
        if maze_name in requested_mazes
        and (
            observability == "all"
            or (observability == "fo" and observable)
            or (observability == "po" and not observable)
        )
    ]
    if "simple" in requested_mazes and observability in {"all", "po"} and verbose:
        print(
            "Skipping simple/PO in regenerate_artifacts because it is redundant "
            "with simple/FO for the current artifact set."
        )
    return selected


def default_scenarios(
    mazes: list[str] | None = None,
    observability: str = "all",
    *,
    verbose: bool = False,
) -> list[ArtifactScenario]:
    settings = selected_settings(mazes, observability, verbose=verbose)
    source_agents = tuple(default_sources())
    evaluators = tuple(default_evaluators())
    return [
        ArtifactScenario(
            maze_name=maze_name,
            observable=observable,
            source_agents=source_agents,
            evaluators=evaluators,
            figure_policies=source_agents,
            train_context_modes=("legacy_context",),
            generation_context_modes=("legacy_context",),
            inference_context_modes=("legacy_context",),
        )
        for maze_name, observable in settings
    ]


def reward_timing_benchmark_scenarios() -> list[ArtifactScenario]:
    policies = tuple(reward_timing_policies())
    return [
        ArtifactScenario(
            maze_name="full",
            observable=False,
            source_agents=tuple(neural_agents()),
            evaluators=tuple(reward_timing_evaluators()),
            figure_policies=policies,
            train_context_modes=("prev_reward", "prev_reward_time"),
            generation_context_modes=("prev_reward", "prev_reward_time"),
            inference_context_modes=("prev_reward", "prev_reward_time"),
            filename_suffix="reward_timing_benchmark",
            benchmark_label="Reward Timing Benchmark",
            benchmark_note=(
                "Suite role: full/PO clean comparison of obs+prev_reward vs "
                "obs+prev_reward+time."
            ),
        )
    ]


def benchmark_suite_scenarios() -> list[ArtifactScenario]:
    legacy_sources = tuple(default_sources())
    neural_sources = tuple(neural_agents())
    neural_policies = tuple(neural_context_policies())
    neural_evaluators = tuple(neural_context_evaluators())
    baseline_evaluators = tuple(default_evaluators())

    return [
        ArtifactScenario(
            maze_name="full",
            observable=True,
            source_agents=legacy_sources,
            evaluators=baseline_evaluators,
            figure_policies=legacy_sources,
            train_context_modes=("legacy_context",),
            generation_context_modes=("legacy_context",),
            inference_context_modes=("legacy_context",),
            filename_suffix="full_baseline",
            benchmark_label="Full Baseline Benchmark",
            benchmark_note="Suite role: baseline benchmark on full with all four agents.",
        ),
        ArtifactScenario(
            maze_name="full",
            observable=False,
            source_agents=legacy_sources,
            evaluators=baseline_evaluators,
            figure_policies=legacy_sources,
            train_context_modes=("legacy_context",),
            generation_context_modes=("legacy_context",),
            inference_context_modes=("legacy_context",),
            filename_suffix="full_baseline",
            benchmark_label="Full Baseline Benchmark",
            benchmark_note="Suite role: baseline benchmark on full with all four agents.",
        ),
        ArtifactScenario(
            maze_name="full",
            observable=True,
            source_agents=neural_sources,
            evaluators=neural_evaluators,
            figure_policies=neural_policies,
            train_context_modes=("observation_only",),
            generation_context_modes=("observation_only",),
            inference_context_modes=("observation_only", "legacy_context"),
            filename_suffix="full_context",
            benchmark_label="Full Context Benchmark",
            benchmark_note=(
                "Suite role: neural context benchmark on full with obs-only and "
                "legacy-context DQN/ELMAN/GRU/LSTM variants."
            ),
        ),
        ArtifactScenario(
            maze_name="full",
            observable=False,
            source_agents=neural_sources,
            evaluators=neural_evaluators,
            figure_policies=neural_policies,
            train_context_modes=("observation_only",),
            generation_context_modes=("observation_only",),
            inference_context_modes=("observation_only", "legacy_context"),
            filename_suffix="full_context",
            benchmark_label="Full Context Benchmark",
            benchmark_note=(
                "Suite role: neural context benchmark on full with obs-only and "
                "legacy-context DQN/ELMAN/GRU/LSTM variants."
            ),
        ),
    ]
