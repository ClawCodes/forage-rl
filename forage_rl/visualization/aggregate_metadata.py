"""Aggregate-plot metadata, labeling, and styling helpers."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from forage_rl.agents.registry import (
    Agent,
    EvaluatorSpec,
    PolicySpec,
    agent_display_label,
    canonical_agent,
    context_mode_display_label,
    context_mode_sort_key,
    is_neural_agent,
)
from forage_rl.environments import resolve_effective_horizon
from forage_rl.environments.maze import MazeMDP

PolicyInput = Agent | PolicySpec
EvaluatorInput = Agent | EvaluatorSpec

BASE_COLORS: dict[Agent, str] = {
    Agent.MBRL: "#e74c3c",
    Agent.QLearning: "#8e44ad",
    Agent.DQN: "#27ae60",
    Agent.ELMAN: "#2980b9",
    Agent.GRU: "#16a085",
    Agent.LSTM: "#f39c12",
}
COMPARISON_ORDER: dict[tuple[Agent, str], int] = {
    (Agent.MBRL, "fresh"): 0,
    (Agent.QLearning, "fresh"): 1,
    (Agent.DQN, "fresh"): 2,
    (Agent.DQN, "pretrained"): 3,
    (Agent.ELMAN, "fresh"): 4,
    (Agent.ELMAN, "pretrained"): 5,
    (Agent.GRU, "fresh"): 6,
    (Agent.GRU, "pretrained"): 7,
    (Agent.LSTM, "fresh"): 8,
    (Agent.LSTM, "pretrained"): 9,
}
SOURCE_LEAD_NOTE = (
    "Metric: source lead on cumulative log-likelihood\n"
    "1.0 = source always ahead\n"
    "0.5 = tied\n"
    "0.0 = evaluator always ahead"
)
CONTEXT_LINESTYLES: dict[str, str] = {
    "observation_only": "-",
    "prev_reward": ":",
    "prev_reward_time": "-.",
    "legacy_context": "--",
}
CONTEXT_HATCHES: dict[str, str] = {
    "observation_only": "xx",
    "prev_reward": "..",
    "prev_reward_time": "oo",
    "legacy_context": "//",
}
def normalize_policy(policy: PolicyInput) -> PolicySpec:
    if isinstance(policy, PolicySpec):
        return policy
    return PolicySpec(agent=policy)


def normalize_evaluator(evaluator: EvaluatorInput) -> EvaluatorSpec:
    if isinstance(evaluator, EvaluatorSpec):
        return evaluator
    return EvaluatorSpec(agent=evaluator, mode="fresh")


def base_agent_display_label(agent: Agent) -> str:
    return agent_display_label(agent)


def context_display_label(context_mode: str) -> str:
    return context_mode_display_label(context_mode)


def policy_uses_explicit_context(policy: PolicyInput) -> bool:
    return isinstance(policy, PolicySpec) and is_neural_agent(policy.agent)


def include_context_labels(
    *,
    policies: list[PolicyInput] | None = None,
    evaluators: list[EvaluatorInput] | None = None,
) -> bool:
    if policies is not None and any(policy_uses_explicit_context(policy) for policy in policies):
        return True
    if evaluators is not None:
        return any(
            is_neural_agent(normalize_evaluator(evaluator).agent)
            and normalize_evaluator(evaluator).context_mode != "legacy_context"
            for evaluator in evaluators
        )
    return False


def policy_display_label(
    policy: PolicyInput,
    *,
    include_context: bool = False,
) -> str:
    spec = normalize_policy(policy)
    base = base_agent_display_label(spec.agent)
    if is_neural_agent(spec.agent) and include_context:
        return f"{base} ({context_display_label(spec.context_mode)})"
    return base


def policy_artifact_label(policy: PolicyInput) -> str:
    spec = normalize_policy(policy)
    if isinstance(policy, PolicySpec):
        return spec.artifact_label
    return canonical_agent(spec.agent).value


def comparison_specs(
    source: PolicyInput,
    compare_to: list[EvaluatorInput],
) -> list[EvaluatorSpec]:
    source_spec = normalize_policy(source)
    specs = [normalize_evaluator(evaluator) for evaluator in compare_to]
    filtered = [
        spec
        for spec in specs
        if not (
            canonical_agent(spec.agent) == canonical_agent(source_spec.agent)
            and spec.mode == "fresh"
            and spec.context_mode == source_spec.context_mode
        )
    ]
    return sorted(
        filtered,
        key=lambda spec: (
            COMPARISON_ORDER.get((canonical_agent(spec.agent), spec.mode), 100),
            context_mode_sort_key(spec.context_mode),
            canonical_agent(spec.agent).value,
            spec.mode,
        ),
    )


def self_evaluator(source: PolicyInput) -> EvaluatorSpec:
    source_spec = normalize_policy(source)
    return EvaluatorSpec(
        agent=source_spec.agent,
        mode="fresh",
        context_mode=source_spec.context_mode,
    )


def trajectory_source_policy_description(source: PolicyInput) -> str:
    source_spec = normalize_policy(source)
    label = policy_display_label(
        source,
        include_context=policy_uses_explicit_context(source),
    )
    if is_neural_agent(source_spec.agent):
        return f"Source policy: {label} online-trained run policy"
    return f"Source policy: {label} saved trajectory-generating policy"


def source_policy_description(source: PolicyInput) -> str:
    source_spec = normalize_policy(source)
    label = policy_display_label(
        source,
        include_context=policy_uses_explicit_context(source),
    )
    if is_neural_agent(source_spec.agent):
        return f"Source: {label} online-trained run policy"
    return f"Source: {label} saved trajectory-generating policy"


def compact_source_policy_line(source: PolicyInput) -> str:
    return (
        "Source policy: "
        f"{policy_display_label(source, include_context=policy_uses_explicit_context(source))}"
    )


def display_label(
    evaluator: EvaluatorInput,
    *,
    include_context: bool = False,
) -> str:
    spec = normalize_evaluator(evaluator)
    if is_neural_agent(spec.agent) and include_context:
        return (
            f"{base_agent_display_label(spec.agent)} "
            f"({context_display_label(spec.context_mode)}, {spec.mode} evaluator)"
        )
    if is_neural_agent(spec.agent):
        return f"{base_agent_display_label(spec.agent)} ({spec.mode} evaluator)"
    return base_agent_display_label(spec.agent)


def filename_label(evaluator: EvaluatorInput) -> str:
    spec = normalize_evaluator(evaluator)
    if is_neural_agent(spec.agent):
        return spec.label
    return canonical_agent(spec.agent).value


def line_style(evaluator: EvaluatorInput) -> dict[str, object]:
    spec = normalize_evaluator(evaluator)
    color = BASE_COLORS.get(canonical_agent(spec.agent), "#7f8c8d")
    style: dict[str, object] = {
        "color": color,
        "linewidth": 2.5,
        "linestyle": CONTEXT_LINESTYLES[spec.context_mode],
    }
    style["alpha"] = 0.6 if spec.mode == "fresh" else 0.95
    return style


def bar_style(evaluator: EvaluatorInput) -> dict[str, object]:
    spec = normalize_evaluator(evaluator)
    base_color = BASE_COLORS.get(canonical_agent(spec.agent), "#7f8c8d")
    style: dict[str, object] = {
        "color": base_color,
        "edgecolor": "#2c3e50",
        "linewidth": 1.2,
        "hatch": CONTEXT_HATCHES[spec.context_mode],
    }
    style["alpha"] = 0.45 if spec.mode == "fresh" else 0.9
    return style


def policy_line_style(policy: PolicyInput) -> dict[str, object]:
    spec = normalize_policy(policy)
    style: dict[str, object] = {
        "color": BASE_COLORS.get(canonical_agent(spec.agent), "#7f8c8d"),
        "linewidth": 2.5,
        "alpha": 0.95,
        "linestyle": CONTEXT_LINESTYLES[spec.context_mode],
    }
    if policy_uses_explicit_context(policy) and spec.context_mode == "legacy_context":
        style["alpha"] = 0.8
    return style


def running_win_rate(source_cumsum: np.ndarray, eval_cumsum: np.ndarray) -> np.ndarray:
    wins = np.where(
        np.isclose(source_cumsum, eval_cumsum),
        0.5,
        (source_cumsum > eval_cumsum).astype(float),
    )
    return np.cumsum(wins) / np.arange(1, len(wins) + 1)


def low_sample_note(run_count: int, sample_count: int) -> Optional[str]:
    notes: list[str] = []
    if run_count < 2:
        notes.append(f"runs={run_count}")
    if sample_count < 2:
        notes.append(f"episodes={sample_count}")
    if not notes:
        return None
    return f"Low sample size ({', '.join(notes)})"


def trajectory_sample_metadata_note(
    run_count: int,
    episodes_per_run: str,
    plotted_episodes: int,
) -> str:
    return (
        f"runs={run_count}, episodes_per_run={episodes_per_run}, "
        f"plotted_episodes={plotted_episodes}"
    )


def cumulative_residency_interpretation_note() -> str:
    return (
        "Right panel shows cumulative within-episode residency share up to each "
        "transition, not current occupancy at that transition."
    )


def cumulative_residency_coarse_sample_note(plotted_episodes: int) -> Optional[str]:
    if plotted_episodes >= 5 or plotted_episodes <= 0:
        return None
    return (
        "With few plotted episodes, early-step cumulative shares can still move "
        "in coarse increments."
    )


def count_label(values: list[int]) -> str:
    if not values:
        return "0"
    unique_values = sorted(set(values))
    if len(unique_values) == 1:
        return str(unique_values[0])
    return f"{unique_values[0]}-{unique_values[-1]}"


def figure_horizon_suffix(maze_name: str, horizon: int | None) -> str:
    default_horizon = resolve_effective_horizon(maze_name, None)
    resolved_horizon = resolve_effective_horizon(maze_name, horizon)
    if resolved_horizon == default_horizon:
        return ""
    return f"_h{resolved_horizon}"


def setting_note(maze_name: str, observable: bool) -> Optional[str]:
    if maze_name == "simple" and not observable:
        return (
            "Note: simple/PO is equivalent to simple/FO for this spec because each "
            "observation group maps to one true state."
        )
    return None


def benchmark_title(label: str | None, title: str) -> str:
    if label is None:
        return title
    return f"{label}\n{title}"


def add_figure_notes(fig: plt.Figure, notes: list[str]) -> None:
    for index, note in enumerate(note for note in notes if note):
        fig.text(
            0.5,
            0.01 + (0.03 * index),
            note,
            ha="center",
            fontsize=10,
            color="#7f8c8d",
        )


def aggregate_trajectory_axis_label() -> str:
    return "Transition Within Episode"


def aggregate_comparison_filename(
    source: PolicyInput,
    compare_to: list[EvaluatorInput],
    maze_name: str,
    observable: bool,
    horizon: int | None = None,
    filename_suffix: str | None = None,
) -> str:
    obs_tag = "FO" if observable else "PO"
    horizon_suffix = figure_horizon_suffix(maze_name, horizon)
    base = f"agg_compare_{policy_artifact_label(source)}_{maze_name}_{obs_tag}{horizon_suffix}"
    if filename_suffix is not None:
        return f"{base}_{filename_suffix}.png"
    comparisons = "_".join(filename_label(evaluator) for evaluator in compare_to)
    return (
        "agg_compare_"
        f"{policy_artifact_label(source)}_to_{comparisons}_{maze_name}_{obs_tag}"
        f"{horizon_suffix}.png"
    )
