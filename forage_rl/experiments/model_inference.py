"""Run model inference experiment comparing registered agents."""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np

from forage_rl import Trajectory
from forage_rl.agents import get_agent, registered_agents
from forage_rl.agents.registry import Agent
from forage_rl.config import ensure_directories
from forage_rl.environments import Maze, MazePOMDP, load_builtin_maze_spec
from forage_rl.experiments.parallel import resolve_worker_count
from forage_rl.utils import (
    list_trajectory_run_ids,
    load_trajectories,
    save_logprobs,
)


InferenceTask = tuple[Agent, int, str, tuple[Agent, ...], bool]


def _parse_agents(values: list[str]) -> list[Agent]:
    if values == ["all"]:
        return registered_agents()
    return [Agent(value) for value in values]


def evaluate_trajectory(
    trajectory: Trajectory,
    maze_name: str = "simple",
    agents: list[Agent] | None = None,
    observable: bool = True,
) -> dict[Agent, np.ndarray]:
    """Evaluate a trajectory under each specified agent model."""
    if agents is None:
        agents = registered_agents()

    maze_spec = load_builtin_maze_spec(maze_name)
    maze_cls = Maze if observable else MazePOMDP
    results = {}
    for agent_name in agents:
        agent = get_agent(agent_name, maze_cls(maze_spec))
        results[agent_name] = np.array(agent.simulate(trajectory))
    return results


def _select_run_ids(
    source: Agent,
    maze_name: str,
    observable: bool,
    num_datasets: Optional[int],
) -> list[int]:
    run_ids = list_trajectory_run_ids(source, maze_name, observable)
    if num_datasets is None:
        return run_ids
    return run_ids[:num_datasets]


def _build_inference_tasks(
    source_agents: list[Agent],
    compare_to: list[Agent],
    maze_name: str,
    num_datasets: Optional[int],
    observable: bool,
) -> tuple[list[InferenceTask], list[Agent]]:
    tasks: list[InferenceTask] = []
    missing_sources: list[Agent] = []

    for source in source_agents:
        run_ids = _select_run_ids(source, maze_name, observable, num_datasets)
        if not run_ids:
            missing_sources.append(source)
            continue

        tasks.extend(
            (source, run_id, maze_name, tuple(compare_to), observable)
            for run_id in run_ids
        )

    return tasks, missing_sources


def _evaluate_dataset_task(task: InferenceTask) -> dict[str, object]:
    source, run_id, maze_name, compare_to, observable = task

    start = time.perf_counter()
    trajectory = load_trajectories(source, run_id, maze_name, observable)
    results = evaluate_trajectory(
        trajectory=trajectory,
        maze_name=maze_name,
        agents=list(compare_to),
        observable=observable,
    )

    totals: dict[str, float] = {}
    for evaluator, log_liks in results.items():
        save_logprobs(
            np.cumsum(log_liks),
            source,
            evaluator,
            run_id,
            maze_name,
            observable,
        )
        totals[evaluator.value] = float(np.sum(log_liks))

    return {
        "source": source,
        "run_id": run_id,
        "trajectory_length": len(trajectory),
        "totals": totals,
        "elapsed": time.perf_counter() - start,
    }


def _print_inference_result(
    result: dict[str, object],
    completed: int,
    task_count: int,
) -> None:
    print(
        f"[{completed}/{task_count}] source={result['source']} "
        f"run={result['run_id']} transitions={result['trajectory_length']}"
    )
    totals = result["totals"]
    assert isinstance(totals, dict)
    for evaluator, total in sorted(totals.items()):
        print(f"  [{evaluator}] total log-likelihood: {total:.2f}")


def _print_timing_summary(task_count: int, worker_count: int, elapsed: float) -> None:
    throughput = task_count / elapsed if elapsed > 0 else float("inf")
    print(
        "\nTiming Summary: "
        f"workers={worker_count}, tasks={task_count}, "
        f"wall_time={elapsed:.2f}s, throughput={throughput:.2f} tasks/s"
    )


def run_inference_experiment(
    source_agents: list[Agent] | None = None,
    compare_to: list[Agent] | None = None,
    maze_name: str = "simple",
    num_datasets: Optional[int] = None,
    observable: bool = True,
    verbose: bool = True,
    workers: int | None = None,
) -> None:
    """Run the full model inference experiment."""
    if source_agents is None:
        source_agents = registered_agents()
    if compare_to is None:
        compare_to = registered_agents()

    ensure_directories()

    tasks, missing_sources = _build_inference_tasks(
        source_agents=source_agents,
        compare_to=compare_to,
        maze_name=maze_name,
        num_datasets=num_datasets,
        observable=observable,
    )

    for source in missing_sources:
        print(
            f"No trajectory files for {source.value}. Run generate_trajectories.py first."
        )

    task_count = len(tasks)
    if task_count == 0:
        print("\nInference experiment complete!")
        return

    worker_count = resolve_worker_count(task_count, workers)

    if verbose:
        print(
            f"Evaluating {task_count} trajectory dataset(s) across "
            f"{len(source_agents)} source agent(s) with {worker_count} worker(s)..."
        )

    start = time.perf_counter()

    if worker_count == 1:
        for completed, task in enumerate(tasks, start=1):
            result = _evaluate_dataset_task(task)
            if verbose:
                _print_inference_result(result, completed, task_count)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_evaluate_dataset_task, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                if verbose:
                    _print_inference_result(result, completed, task_count)

    elapsed = time.perf_counter() - start
    if verbose:
        _print_timing_summary(task_count, worker_count, elapsed)

    print("\nInference experiment complete!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model inference experiment")
    parser.add_argument(
        "--source-agents",
        nargs="+",
        default=["all"],
        help="Source agent name(s) whose trajectories to evaluate, or 'all'",
    )
    parser.add_argument(
        "--compare-to",
        nargs="+",
        default=["all"],
        help="Evaluator agent name(s) to compare against, or 'all'",
    )
    parser.add_argument(
        "--num-datasets",
        type=int,
        default=None,
        help="Number of datasets to process per source agent (default: all available)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: auto-capped to available CPUs)",
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
        help="Use partially observable trajectories (PO); default is fully observable (FO)",
    )

    args = parser.parse_args()

    run_inference_experiment(
        source_agents=_parse_agents(args.source_agents),
        compare_to=_parse_agents(args.compare_to),
        maze_name=args.maze,
        num_datasets=args.num_datasets,
        observable=not args.pomdp,
        verbose=not args.quiet,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
