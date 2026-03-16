"""Metric helpers for all-agent model-comparison plots."""

from __future__ import annotations

from typing import Optional

import numpy as np

from forage_rl.agents import registered_agents
from forage_rl.utils import get_logprob_run_ids, load_logprobs


MISSING_LOGPROBS_MESSAGE = (
    "No log probability files found. Run trajectory generation and model "
    "inference for the selected agents first."
)


def normalize_agents(agents: Optional[list[str]]) -> list[str]:
    """Return a stable ordered agent list."""
    values = registered_agents() if agents is None else agents
    return list(dict.fromkeys(values))


def _validate_num_datasets(num_datasets: Optional[int]) -> Optional[int]:
    """Return a validated optional dataset cap."""
    if num_datasets is None:
        return None
    if num_datasets <= 0:
        raise ValueError("num_datasets must be > 0")
    return num_datasets


def _shared_run_ids(
    source: str,
    evaluator: str,
    num_datasets: Optional[int] = None,
    env_key: str | None = None,
) -> list[int]:
    """Return sorted shared run ids for a source/evaluator log-prob pair."""
    num_datasets = _validate_num_datasets(num_datasets)
    source_self_ids = set(
        get_logprob_run_ids(f"source_{source}_eval_{source}", env_key=env_key)
    )
    source_eval_ids = set(
        get_logprob_run_ids(f"source_{source}_eval_{evaluator}", env_key=env_key)
    )
    shared_run_ids = sorted(source_self_ids & source_eval_ids)
    if num_datasets is not None:
        return shared_run_ids[:num_datasets]
    return shared_run_ids


def load_self_vs_eval_pairs(
    source: str,
    evaluator: str,
    num_datasets: Optional[int] = None,
    env_key: str | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load aligned self-eval and evaluator arrays for a source agent."""
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    run_ids = _shared_run_ids(source, evaluator, num_datasets, env_key=env_key)
    for run_id in run_ids:
        source_self = load_logprobs(
            f"source_{source}_eval_{source}",
            run_id,
            env_key=env_key,
        )
        source_eval = load_logprobs(
            f"source_{source}_eval_{evaluator}",
            run_id,
            env_key=env_key,
        )
        pairs.append((source_self, source_eval))
    return pairs


def step_accuracy(source_self: np.ndarray, source_eval: np.ndarray) -> np.ndarray:
    """Compute transition-wise self-vs-evaluator accuracy."""
    min_len = min(len(source_self), len(source_eval))
    accuracy = np.zeros(min_len)
    for idx in range(min_len):
        if np.isclose(source_self[idx], source_eval[idx]):
            accuracy[idx] = 0.5
        elif source_self[idx] > source_eval[idx]:
            accuracy[idx] = 1.0
        else:
            accuracy[idx] = 0.0
    return accuracy


def _final_win_score(source_self: np.ndarray, source_eval: np.ndarray) -> float:
    """Score the final cumulative log-prob comparison with tie-aware semantics."""
    source_final = source_self[-1]
    eval_final = source_eval[-1]
    if np.isclose(source_final, eval_final):
        return 0.5
    if source_final > eval_final:
        return 1.0
    return 0.0


def compute_model_comparison(
    agents: Optional[list[str]] = None,
    num_datasets: Optional[int] = None,
    env_key: str | None = None,
) -> tuple[list[str], list[float]] | None:
    """Return average self-vs-other win rate for each selected source agent."""
    num_datasets = _validate_num_datasets(num_datasets)
    selected_agents = normalize_agents(agents)
    if len(selected_agents) < 2:
        return None

    agent_scores: list[float] = []
    for source in selected_agents:
        comparison_scores = []
        for evaluator in selected_agents:
            if evaluator == source:
                continue
            pairs = load_self_vs_eval_pairs(
                source, evaluator, num_datasets, env_key=env_key
            )
            if not pairs:
                continue
            wins = [
                _final_win_score(source_self, source_eval)
                for source_self, source_eval in pairs
            ]
            comparison_scores.append(float(np.mean(wins)))
        agent_scores.append(
            float(np.mean(comparison_scores)) if comparison_scores else np.nan
        )

    if all(np.isnan(score) for score in agent_scores):
        return None
    return selected_agents, agent_scores


def compute_mean_cumulative_accuracy(
    agents: Optional[list[str]] = None,
    num_datasets: Optional[int] = None,
    env_key: str | None = None,
) -> np.ndarray | None:
    """Return average transition-wise classification accuracy across all pairs."""
    num_datasets = _validate_num_datasets(num_datasets)
    selected_agents = normalize_agents(agents)
    if len(selected_agents) < 2:
        return None

    accuracies: list[np.ndarray] = []
    for source in selected_agents:
        for evaluator in selected_agents:
            if evaluator == source:
                continue
            pairs = load_self_vs_eval_pairs(
                source, evaluator, num_datasets, env_key=env_key
            )
            accuracies.extend(
                step_accuracy(source_self, source_eval)
                for source_self, source_eval in pairs
            )

    if not accuracies:
        return None

    min_len = min(len(accuracy) for accuracy in accuracies)
    return np.mean([accuracy[:min_len] for accuracy in accuracies], axis=0)


def compute_source_win_rates(
    source: str,
    compare_to: list[str],
    num_datasets: Optional[int] = None,
    env_key: str | None = None,
) -> tuple[list[str], list[float], list[float]] | None:
    """Return per-evaluator win rates, preserving missing-data comparisons as NaN."""
    num_datasets = _validate_num_datasets(num_datasets)
    comparisons = [agent for agent in compare_to if agent != source]
    if not comparisons:
        return None

    source_rates: list[float] = []
    eval_rates: list[float] = []
    for evaluator in comparisons:
        pairs = load_self_vs_eval_pairs(
            source, evaluator, num_datasets, env_key=env_key
        )
        if not pairs:
            source_rates.append(np.nan)
            eval_rates.append(np.nan)
            continue

        rate = float(
            np.mean(
                [
                    _final_win_score(source_self, source_eval)
                    for source_self, source_eval in pairs
                ]
            )
        )
        source_rates.append(rate)
        eval_rates.append(1.0 - rate)

    return comparisons, source_rates, eval_rates


def compute_pairwise_accuracy_matrix(
    agents: Optional[list[str]] = None,
    num_datasets: Optional[int] = None,
    env_key: str | None = None,
) -> tuple[list[str], np.ndarray] | None:
    """Return a heatmap-ready matrix of self-vs-other final accuracy rates."""
    num_datasets = _validate_num_datasets(num_datasets)
    selected_agents = normalize_agents(agents)
    if len(selected_agents) < 2:
        return None

    matrix = np.full((len(selected_agents), len(selected_agents)), np.nan)
    for row, source in enumerate(selected_agents):
        for col, evaluator in enumerate(selected_agents):
            if source == evaluator:
                continue
            pairs = load_self_vs_eval_pairs(
                source, evaluator, num_datasets, env_key=env_key
            )
            if not pairs:
                continue
            matrix[row, col] = np.mean(
                [
                    _final_win_score(source_self, source_eval)
                    for source_self, source_eval in pairs
                ]
            )

    if np.isnan(matrix).all():
        return None
    return selected_agents, matrix


def compute_pairwise_logprob_gap_matrix(
    agents: Optional[list[str]] = None,
    num_datasets: Optional[int] = None,
    env_key: str | None = None,
) -> tuple[list[str], np.ndarray] | None:
    """Return a heatmap-ready matrix of mean final cumulative log-prob gaps."""
    num_datasets = _validate_num_datasets(num_datasets)
    selected_agents = normalize_agents(agents)
    if len(selected_agents) < 2:
        return None

    matrix = np.full((len(selected_agents), len(selected_agents)), np.nan)
    for row, source in enumerate(selected_agents):
        for col, evaluator in enumerate(selected_agents):
            if source == evaluator:
                continue
            pairs = load_self_vs_eval_pairs(
                source, evaluator, num_datasets, env_key=env_key
            )
            if not pairs:
                continue
            matrix[row, col] = np.mean(
                [
                    source_self[-1] - source_eval[-1]
                    for source_self, source_eval in pairs
                ]
            )

    if np.isnan(matrix).all():
        return None
    return selected_agents, matrix


def compute_pairwise_cumulative_accuracy(
    source: str,
    evaluator: str,
    num_datasets: Optional[int] = None,
    env_key: str | None = None,
) -> np.ndarray | None:
    """Return mean transition-wise accuracy for a specific source/evaluator pair."""
    num_datasets = _validate_num_datasets(num_datasets)
    pairs = load_self_vs_eval_pairs(source, evaluator, num_datasets, env_key=env_key)
    if not pairs:
        return None

    accuracies = [
        step_accuracy(source_self, source_eval) for source_self, source_eval in pairs
    ]
    min_len = min(len(accuracy) for accuracy in accuracies)
    return np.mean([accuracy[:min_len] for accuracy in accuracies], axis=0)


__all__ = [
    "MISSING_LOGPROBS_MESSAGE",
    "compute_mean_cumulative_accuracy",
    "compute_model_comparison",
    "compute_pairwise_accuracy_matrix",
    "compute_pairwise_cumulative_accuracy",
    "compute_pairwise_logprob_gap_matrix",
    "compute_source_win_rates",
    "_final_win_score",
    "load_self_vs_eval_pairs",
    "normalize_agents",
    "step_accuracy",
]
