"""Generate training trajectories for registered agents."""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TypedDict

from forage_rl.agents import get_agent, registered_agents
from forage_rl.agents.context import (
    DEFAULT_NEURAL_CONTEXT_MODE,
    NEURAL_CONTEXT_MODES,
    NeuralContextMode,
    validate_context_mode,
)
from forage_rl.agents.registry import Agent
from forage_rl.config import DefaultParams, ensure_output_directories
from forage_rl.environments import (
    Maze,
    load_builtin_maze_spec,
    resolve_effective_horizon,
)
from forage_rl.experiments.parallel import (
    build_torch_batches,
    is_neural_agent,
    resolve_execution_strategy,
)
from forage_rl.utils.torch_support import configure_torch_worker
from forage_rl.utils import save_run_dataset


GenerationTask = tuple[
    Agent,
    int,
    str,
    int,
    bool,
    int | None,
    str,
    NeuralContextMode,
    int | None,
]


class GenerationResult(TypedDict):
    agent_type: Agent
    run_id: int
    num_transitions: int
    filepath: str
    elapsed: float


def _derive_agent_seed(run_seed: int | None) -> int | None:
    """Derive a deterministic agent seed distinct from the maze seed."""
    if run_seed is None:
        return None
    return run_seed + 1


def _parse_agents(values: list[str]) -> list[Agent]:
    if values == ["all"]:
        return registered_agents()
    return [Agent(value) for value in values]


def _build_generation_tasks(
    agent_types: list[Agent],
    maze_name: str,
    num_runs: int,
    num_episodes: int,
    observable: bool,
    base_seed: int | None,
    device: str,
    context_mode: NeuralContextMode,
    horizon: int | None,
) -> list[GenerationTask]:
    return [
        (
            agent_type,
            run_id,
            maze_name,
            num_episodes,
            observable,
            base_seed,
            device,
            context_mode,
            horizon,
        )
        for agent_type in agent_types
        for run_id in range(num_runs)
    ]


def _generate_single_run(task: GenerationTask) -> GenerationResult:
    (
        agent_type,
        run_id,
        maze_name,
        num_episodes,
        observable,
        base_seed,
        device,
        context_mode,
        horizon,
    ) = task

    resolved_horizon = resolve_effective_horizon(maze_name, horizon)
    maze_spec = load_builtin_maze_spec(maze_name)
    run_seed = None if base_seed is None else base_seed + run_id
    agent_seed = _derive_agent_seed(run_seed)

    if is_neural_agent(agent_type):
        configure_torch_worker(device)

    start = time.perf_counter()
    maze = Maze(
        maze_spec, seed=run_seed, horizon=resolved_horizon, observable=observable
    )
    agent = get_agent(
        agent_type,
        maze,
        num_episodes=num_episodes,
        seed=agent_seed,
        device=device,
        context_mode=context_mode,
    )
    run_dataset = agent.train(verbose=False)
    filepath = save_run_dataset(
        run_dataset,
        agent_type,
        run_id,
        maze_name,
        observable,
        context_mode=context_mode,
        horizon=resolved_horizon,
    )

    return {
        "agent_type": agent_type,
        "run_id": run_id,
        "num_transitions": run_dataset.num_transitions(),
        "filepath": str(filepath),
        "elapsed": time.perf_counter() - start,
    }


def _print_generation_result(
    result: GenerationResult,
    completed: int,
    task_count: int,
) -> None:
    print(
        f"[{completed}/{task_count}] {result['agent_type']} run "
        f"{result['run_id'] + 1} saved {result['num_transitions']} transitions "
        f"to {result['filepath']}"
    )


def _print_timing_summary(task_count: int, worker_count: int, elapsed: float) -> None:
    throughput = task_count / elapsed if elapsed > 0 else float("inf")
    print(
        "\nTiming Summary: "
        f"workers={worker_count}, tasks={task_count}, "
        f"wall_time={elapsed:.2f}s, throughput={throughput:.2f} tasks/s"
    )


def _execute_generation_tasks(
    tasks: list[GenerationTask],
    *,
    agent_types: list[Agent],
    workers: int | None,
    device: str,
    verbose: bool,
    uses_torch: bool,
    batch_label: str,
) -> None:
    """Execute one homogeneous generation batch."""
    task_count = len(tasks)
    if task_count == 0:
        return

    strategy = resolve_execution_strategy(
        task_count,
        workers,
        uses_torch=uses_torch,
        device=device,
    )
    worker_count = strategy.worker_count

    if verbose:
        print(
            f"Generating {task_count} trajectories across {len(agent_types)} "
            f"{batch_label} agent(s) with {worker_count} worker(s)..."
        )
        if strategy.worker_note is not None:
            print(strategy.worker_note)

    start = time.perf_counter()

    if worker_count == 1:
        for completed, task in enumerate(tasks, start=1):
            result = _generate_single_run(task)
            if verbose:
                _print_generation_result(result, completed, task_count)
    else:
        executor_kwargs = {"max_workers": worker_count}
        if strategy.mp_context is not None:
            executor_kwargs["mp_context"] = strategy.mp_context
        with ProcessPoolExecutor(**executor_kwargs) as executor:
            futures = [executor.submit(_generate_single_run, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                if verbose:
                    _print_generation_result(result, completed, task_count)

    elapsed = time.perf_counter() - start
    if verbose:
        _print_timing_summary(task_count, worker_count, elapsed)


def run_generation_experiment(
    agent_types: list[Agent] | None = None,
    maze_name: str = "simple",
    num_runs: int = DefaultParams.NUM_RUN_DATASETS,
    num_episodes: int = DefaultParams.TRAINING_EPISODES,
    observable: bool = True,
    verbose: bool = True,
    workers: int | None = None,
    base_seed: int | None = None,
    device: str = "auto",
    context_mode: NeuralContextMode = DEFAULT_NEURAL_CONTEXT_MODE,
    horizon: int | None = None,
) -> None:
    """Generate trajectories for one or more registered agents."""
    ensure_output_directories()

    agent_types = registered_agents() if agent_types is None else agent_types
    batches = build_torch_batches(agent_types, device=device)

    total_task_count = sum(len(agent_batch) * num_runs for agent_batch, _, _ in batches)
    if total_task_count == 0:
        print("\nTrajectory generation complete!")
        return

    for batch_agents, batch_uses_torch, batch_label in batches:
        if not batch_agents:
            continue
        tasks = _build_generation_tasks(
            agent_types=batch_agents,
            maze_name=maze_name,
            num_runs=num_runs,
            num_episodes=num_episodes,
            observable=observable,
            base_seed=base_seed,
            device=device,
            context_mode=context_mode,
            horizon=horizon,
        )
        _execute_generation_tasks(
            tasks,
            agent_types=batch_agents,
            workers=workers,
            device=device,
            verbose=verbose,
            uses_torch=batch_uses_torch,
            batch_label=batch_label,
        )

    print("\nTrajectory generation complete!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training trajectories")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["all"],
        help="Agent name(s) to run, or 'all' for every canonical registered agent.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=DefaultParams.NUM_RUN_DATASETS,
        help="Number of independent runs",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=DefaultParams.TRAINING_EPISODES,
        help="Episodes per run",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: auto-capped to available CPUs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed; each run uses base seed + run id",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for neural agents: auto, cpu, cuda, or mps",
    )
    parser.add_argument(
        "--context-mode",
        type=validate_context_mode,
        default=DEFAULT_NEURAL_CONTEXT_MODE,
        help=(
            "Neural input context mode for DQN/Elman/GRU/LSTM; ignored by "
            f"non-neural agents. Valid modes: {', '.join(NEURAL_CONTEXT_MODES)}."
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Optional episode-length override; default uses the built-in maze horizon.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument(
        "--maze",
        default="simple",
        help="Built-in maze spec name (e.g. simple, full)",
    )
    parser.add_argument(
        "--pomdp",
        action="store_true",
        help="Use partially observable maze (PO); default is fully observable (FO)",
    )

    args = parser.parse_args()

    run_generation_experiment(
        agent_types=_parse_agents(args.agents),
        maze_name=args.maze,
        num_runs=args.num_runs,
        num_episodes=args.num_episodes,
        observable=not args.pomdp,
        verbose=not args.quiet,
        workers=args.workers,
        base_seed=args.seed,
        device=args.device,
        context_mode=args.context_mode,
        horizon=args.horizon,
    )


if __name__ == "__main__":
    main()
