"""Generate training trajectories for registered agents."""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TypedDict

from forage_rl.agents import get_agent, registered_agents
from forage_rl.agents.registry import Agent
from forage_rl.config import DefaultParams, ensure_directories
from forage_rl.environments import Maze, MazePOMDP, load_builtin_maze_spec
from forage_rl.experiments.parallel import resolve_worker_count
from forage_rl.utils import save_trajectories


GenerationTask = tuple[Agent, int, str, int, bool, int | None]


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
) -> list[GenerationTask]:
    return [
        (agent_type, run_id, maze_name, num_episodes, observable, base_seed)
        for agent_type in agent_types
        for run_id in range(num_runs)
    ]


def _generate_single_run(task: GenerationTask) -> GenerationResult:
    agent_type, run_id, maze_name, num_episodes, observable, base_seed = task

    maze_spec = load_builtin_maze_spec(maze_name)
    maze_cls = Maze if observable else MazePOMDP
    run_seed = None if base_seed is None else base_seed + run_id
    agent_seed = _derive_agent_seed(run_seed)

    start = time.perf_counter()
    maze = maze_cls(maze_spec, seed=run_seed)
    agent = get_agent(agent_type, maze, num_episodes=num_episodes, seed=agent_seed)
    transitions = agent.train(verbose=False)
    filepath = save_trajectories(transitions, agent_type, run_id, maze_name, observable)

    return {
        "agent_type": agent_type,
        "run_id": run_id,
        "num_transitions": len(transitions),
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


def run_generation_experiment(
    agent_types: list[Agent] | None = None,
    maze_name: str = "simple",
    num_runs: int = DefaultParams.NUM_TRAINING_RUNS,
    num_episodes: int = DefaultParams.NUM_TRAINING_EPISODES,
    observable: bool = True,
    verbose: bool = True,
    workers: int | None = None,
    base_seed: int | None = None,
) -> None:
    """Generate trajectories for one or more registered agents."""
    ensure_directories()

    agent_types = registered_agents() if agent_types is None else agent_types
    tasks = _build_generation_tasks(
        agent_types=agent_types,
        maze_name=maze_name,
        num_runs=num_runs,
        num_episodes=num_episodes,
        observable=observable,
        base_seed=base_seed,
    )

    task_count = len(tasks)
    worker_count = resolve_worker_count(task_count, workers)

    if task_count == 0:
        print("\nTrajectory generation complete!")
        return

    if verbose:
        print(
            f"Generating {task_count} trajectories across {len(agent_types)} agent(s) "
            f"with {worker_count} worker(s)..."
        )

    start = time.perf_counter()

    if worker_count == 1:
        for completed, task in enumerate(tasks, start=1):
            result = _generate_single_run(task)
            if verbose:
                _print_generation_result(result, completed, task_count)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_generate_single_run, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                if verbose:
                    _print_generation_result(result, completed, task_count)

    elapsed = time.perf_counter() - start
    if verbose:
        _print_timing_summary(task_count, worker_count, elapsed)

    print("\nTrajectory generation complete!")


def generate_trajectories(
    agent_type: Agent,
    maze_name: str = "simple",
    num_runs: int = DefaultParams.NUM_TRAINING_RUNS,
    num_episodes: int = DefaultParams.NUM_TRAINING_EPISODES,
    observable: bool = True,
    verbose: bool = True,
    workers: int | None = None,
    base_seed: int | None = None,
) -> None:
    """Generate trajectories from a single registered agent."""
    run_generation_experiment(
        agent_types=[agent_type],
        maze_name=maze_name,
        num_runs=num_runs,
        num_episodes=num_episodes,
        observable=observable,
        verbose=verbose,
        workers=workers,
        base_seed=base_seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training trajectories")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["all"],
        help="Agent name(s) to run, or 'all' for every registered agent",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=DefaultParams.NUM_TRAINING_RUNS,
        help="Number of independent runs",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=DefaultParams.NUM_TRAINING_EPISODES,
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
    )


if __name__ == "__main__":
    main()
