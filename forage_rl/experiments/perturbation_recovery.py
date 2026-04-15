"""Run simple-maze perturbation experiments against a true analytic MVT benchmark."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from forage_rl import RunDataset
from forage_rl.agents import get_agent, registered_agents
from forage_rl.agents.registry import Agent, PolicySpec
from forage_rl.config import TRAJECTORIES_DIR, ensure_directories
from forage_rl.environments import SimpleMaze, load_builtin_maze_spec, resolve_effective_horizon
from forage_rl.analysis.mvt import simple_true_mvt_optimal_prt
from forage_rl.analysis.recovery import (
    recovery_auc,
    recovery_curve_for_run,
    signed_recovery_curve_for_run,
)
from forage_rl.utils.artifact_names import artifact_prefix, neural_context_suffix
from forage_rl.visualization import (
    plot_recovery_auc_comparison,
    plot_recovery_curve_comparison,
    plot_signed_recovery_curve_comparison,
)


DEFAULT_RECOVERY_AGENTS: tuple[Agent, ...] = (
    Agent.MBRL,
    Agent.QLearning,
    Agent.DQN,
    Agent.ELMAN,
    Agent.GRU,
    Agent.LSTM,
)
DEFAULT_PERTURBATION_ID = "decay_swap"


@dataclass(frozen=True)
class RecoveryRunResult:
    policy: PolicySpec
    run_id: int
    curve: np.ndarray
    signed_curve: np.ndarray
    auc: float
    filepath: Path


def _parse_agents(values: list[str]) -> list[Agent]:
    if values == ["all"]:
        return registered_agents()
    return [Agent(value) for value in values]


def _policy_spec(agent: Agent, context_mode: str) -> PolicySpec:
    return PolicySpec(
        agent=agent,
        context_mode=context_mode if agent in {Agent.DQN, Agent.ELMAN, Agent.GRU, Agent.LSTM} else "legacy_context",
    )


def _simple_decays() -> list[float]:
    spec = load_builtin_maze_spec("simple")
    return [
        float(state_spec.decay)
        for state_spec in sorted(spec.states, key=lambda state: state.id)
    ]


def _build_simple_maze(*, decays: list[float], horizon: int, seed: int | None) -> SimpleMaze:
    return SimpleMaze(decays=decays, horizon=horizon, seed=seed)


def _swap_agent_maze(agent, maze) -> None:
    """Swap the environment while preserving the agent's learned state."""
    agent.maze = maze


def _save_perturbation_run_dataset(
    run_dataset: RunDataset,
    *,
    policy: PolicySpec,
    run_id: int,
    maze_name: str,
    observable: bool,
    perturbation_id: str,
    horizon: int,
    num_pre_episodes: int,
    num_post_episodes: int,
) -> Path:
    ensure_directories()
    artifact_name = (
        f"{artifact_prefix(maze_name, observable, horizon)}_{policy.agent.value}"
        f"{neural_context_suffix(policy.agent, policy.context_mode)}"
        f"_perturb_{perturbation_id}_run_dataset_{run_id}.npz"
    )
    filepath = TRAJECTORIES_DIR / artifact_name

    payload: dict[str, object] = {
        "__transition_type__": np.array(
            run_dataset.transition_cls().__name__,
            dtype=np.str_,
        ),
    }
    for episode_index, trajectory in enumerate(run_dataset.trajectories):
        payload[f"episode_{episode_index:05d}"] = trajectory.to_numpy()
    np.savez(filepath, **payload)

    metadata = {
        "container_type": "perturbation_run_dataset",
        "agent": policy.agent.value,
        "maze_name": maze_name,
        "observable": observable,
        "benchmark_kind": "true_mvt",
        "benchmark_label": "True MVT Benchmark",
        "perturbation_id": perturbation_id,
        "horizon": horizon,
        "num_pre_episodes": num_pre_episodes,
        "num_post_episodes": num_post_episodes,
        "perturbation_episode": num_pre_episodes,
        "num_episodes": run_dataset.num_episodes(),
        "num_transitions": run_dataset.num_transitions(),
    }
    if policy.agent in {Agent.DQN, Agent.ELMAN, Agent.GRU, Agent.LSTM}:
        metadata["context_mode"] = policy.context_mode
    filepath.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return filepath


def _combine_run_datasets(pre_run: RunDataset, post_run: RunDataset) -> RunDataset:
    return RunDataset(trajectories=[*pre_run.trajectories, *post_run.trajectories])


def _run_single_recovery_experiment(
    *,
    policy: PolicySpec,
    run_id: int,
    maze_name: str,
    observable: bool,
    pre_episodes: int,
    post_episodes: int,
    recovery_window: int,
    device: str,
    seed: int,
    horizon: int | None,
    perturbation_id: str,
) -> RecoveryRunResult:
    if maze_name != "simple":
        raise ValueError("True MVT recovery is currently implemented only for the simple maze.")
    if not observable:
        raise ValueError("True MVT recovery is currently implemented only for simple/FO.")
    if perturbation_id != DEFAULT_PERTURBATION_ID:
        raise ValueError(
            f"Unsupported perturbation_id {perturbation_id!r}. Expected {DEFAULT_PERTURBATION_ID!r}."
        )

    resolved_horizon = resolve_effective_horizon(maze_name, horizon)
    base_decays = _simple_decays()
    perturbed_decays = list(reversed(base_decays))

    run_seed = seed + run_id
    agent_seed = run_seed + 1
    baseline_maze = _build_simple_maze(
        decays=base_decays,
        horizon=resolved_horizon,
        seed=run_seed,
    )
    agent = get_agent(
        policy.agent,
        baseline_maze,
        num_episodes=pre_episodes,
        seed=agent_seed,
        device=device,
        context_mode=policy.context_mode,
    )
    pre_run = agent.train(verbose=False)

    perturbed_maze = _build_simple_maze(
        decays=perturbed_decays,
        horizon=resolved_horizon,
        seed=run_seed,
    )
    _swap_agent_maze(agent, perturbed_maze)
    agent.num_episodes = post_episodes
    post_run = agent.train(verbose=False)

    benchmark_prt_by_state = simple_true_mvt_optimal_prt(perturbed_maze, resolved_horizon)
    patch_labels = {
        state_spec.id: state_spec.label
        for state_spec in perturbed_maze.maze_spec.states
    }
    leave_action = perturbed_maze.action_labels.index("leave")
    curve = recovery_curve_for_run(
        post_run,
        patch_labels=patch_labels,
        exit_actions=leave_action,
        benchmark_prt_by_state=benchmark_prt_by_state,
    )
    signed_curve = signed_recovery_curve_for_run(
        post_run,
        patch_labels=patch_labels,
        exit_actions=leave_action,
        benchmark_prt_by_state=benchmark_prt_by_state,
    )
    auc = recovery_auc(curve, recovery_window)

    combined_dataset = _combine_run_datasets(pre_run, post_run)
    filepath = _save_perturbation_run_dataset(
        combined_dataset,
        policy=policy,
        run_id=run_id,
        maze_name=maze_name,
        observable=observable,
        perturbation_id=perturbation_id,
        horizon=resolved_horizon,
        num_pre_episodes=pre_episodes,
        num_post_episodes=post_episodes,
    )
    return RecoveryRunResult(
        policy=policy,
        run_id=run_id,
        curve=curve,
        signed_curve=signed_curve,
        auc=auc,
        filepath=filepath,
    )


def run_perturbation_recovery_experiment(
    *,
    agents: list[Agent] | None = None,
    maze_name: str = "simple",
    observability: str = "fo",
    num_runs: int = 100,
    pre_episodes: int = 100,
    post_episodes: int = 100,
    recovery_window: int = 100,
    device: str = "auto",
    context_mode: str = "legacy_context",
    seed: int = 0,
    horizon: int | None = None,
    perturbation_id: str = DEFAULT_PERTURBATION_ID,
    verbose: bool = True,
) -> dict[PolicySpec, list[RecoveryRunResult]]:
    """Generate perturbation recovery datasets and figures for simple/FO."""
    if maze_name != "simple":
        raise ValueError("True MVT perturbation recovery is currently implemented only for simple.")
    if observability != "fo":
        raise ValueError("True MVT perturbation recovery is currently implemented only for FO.")

    selected_agents = list(DEFAULT_RECOVERY_AGENTS if agents is None else agents)
    if recovery_window <= 0:
        raise ValueError(f"recovery_window must be > 0, got {recovery_window}")

    results_by_policy: dict[PolicySpec, list[RecoveryRunResult]] = {}
    for agent in selected_agents:
        policy = _policy_spec(agent, context_mode)
        policy_results: list[RecoveryRunResult] = []
        for run_id in range(num_runs):
            result = _run_single_recovery_experiment(
                policy=policy,
                run_id=run_id,
                maze_name=maze_name,
                observable=True,
                pre_episodes=pre_episodes,
                post_episodes=post_episodes,
                recovery_window=recovery_window,
                device=device,
                seed=seed,
                horizon=horizon,
                perturbation_id=perturbation_id,
            )
            policy_results.append(result)
            if verbose:
                print(
                    f"[{agent.value} run {run_id + 1}/{num_runs}] "
                    f"saved {result.filepath.name} (AUC={result.auc:.2f})"
                )
        results_by_policy[policy] = policy_results

    absolute_curves = {
        policy: [result.curve for result in results]
        for policy, results in results_by_policy.items()
    }
    signed_curves = {
        policy: [result.signed_curve for result in results]
        for policy, results in results_by_policy.items()
    }
    auc_values = {
        policy: [result.auc for result in results]
        for policy, results in results_by_policy.items()
    }
    benchmark_label = "True MVT Benchmark"
    plot_recovery_curve_comparison(
        absolute_curves,
        maze_name=maze_name,
        observable=True,
        perturbation_label=perturbation_id,
        save=True,
        show=False,
        benchmark_label=benchmark_label,
    )
    plot_signed_recovery_curve_comparison(
        signed_curves,
        maze_name=maze_name,
        observable=True,
        perturbation_label=perturbation_id,
        save=True,
        show=False,
        benchmark_label=benchmark_label,
    )
    plot_recovery_auc_comparison(
        auc_values,
        maze_name=maze_name,
        observable=True,
        perturbation_label=perturbation_id,
        save=True,
        show=False,
        benchmark_label=benchmark_label,
    )
    return results_by_policy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run simple-maze perturbation recovery analysis against a true MVT benchmark."
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=[agent.value for agent in DEFAULT_RECOVERY_AGENTS],
        help="Agent names to evaluate, or 'all' to include every registered agent.",
    )
    parser.add_argument(
        "--maze",
        default="simple",
        help="Maze name. Only 'simple' is currently supported for the true MVT benchmark.",
    )
    parser.add_argument(
        "--observability",
        choices=["fo", "po"],
        default="fo",
        help="Observability setting. Only 'fo' is currently supported.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="Number of independent perturbation runs to generate.",
    )
    parser.add_argument(
        "--pre-episodes",
        type=int,
        default=100,
        help="Number of baseline episodes before the perturbation boundary.",
    )
    parser.add_argument(
        "--post-episodes",
        type=int,
        default=100,
        help="Number of perturbed episodes after the perturbation boundary.",
    )
    parser.add_argument(
        "--recovery-window",
        type=int,
        default=100,
        help="Number of post-perturbation episodes to include in AUC.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for neural agents: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--context-mode",
        default="legacy_context",
        help="Neural input context mode for neural agents.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic base seed for generation.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Optional horizon override for the simple maze.",
    )
    parser.add_argument(
        "--perturbation-id",
        default=DEFAULT_PERTURBATION_ID,
        help="Perturbation identifier. Only the default decay swap is supported.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run output.")

    args = parser.parse_args()
    run_perturbation_recovery_experiment(
        agents=_parse_agents(args.agents),
        maze_name=args.maze,
        observability=args.observability,
        num_runs=args.num_runs,
        pre_episodes=args.pre_episodes,
        post_episodes=args.post_episodes,
        recovery_window=args.recovery_window,
        device=args.device,
        context_mode=args.context_mode,
        seed=args.seed,
        horizon=args.horizon,
        perturbation_id=args.perturbation_id,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
