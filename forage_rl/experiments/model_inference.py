"""Run model inference experiment comparing registered agents."""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np

from forage_rl import RunDataset
from forage_rl.agents import get_agent, registered_agents
from forage_rl.agents.registry import (
    Agent,
    EvaluatorSpec,
    NEURAL_CONTEXT_MODES,
    NeuralContextMode,
)
from forage_rl.config import DefaultParams
from forage_rl.config import ensure_output_directories
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
from forage_rl.utils import (
    list_run_dataset_run_ids,
    load_run_dataset,
    load_run_dataset_metadata,
    save_logprobs,
)
from forage_rl.utils.torch_support import configure_torch_worker


EvaluatorInput = Agent | EvaluatorSpec
InferenceTask = tuple[
    Agent,
    int,
    str,
    tuple[EvaluatorSpec, ...],
    bool,
    str,
    int,
    NeuralContextMode,
    int | None,
]


def _parse_agents(values: list[str]) -> list[Agent]:
    if values == ["all"]:
        return registered_agents()
    return [Agent(value) for value in values]


def _normalize_evaluator(evaluator: EvaluatorInput) -> EvaluatorSpec:
    if isinstance(evaluator, EvaluatorSpec):
        return evaluator
    return EvaluatorSpec(agent=evaluator, mode="fresh")


def _parse_evaluators(
    values: list[str],
    neural_context_mode: NeuralContextMode = "legacy_context",
) -> list[EvaluatorInput]:
    if values == ["all"]:
        return registered_agents()

    evaluators: list[EvaluatorInput] = []
    for value in values:
        if ":" not in value:
            evaluators.append(Agent(value))
            continue

        agent_name, mode = value.split(":", maxsplit=1)
        agent = Agent(agent_name)
        if mode not in {"fresh", "pretrained"}:
            raise ValueError(
                f"Unsupported evaluator mode {mode!r}. Expected fresh or pretrained."
            )
        if mode == "pretrained" and not is_neural_agent(agent):
            raise ValueError(
                f"Pretrained evaluators are only supported for neural agents, got {agent.value}."
            )
        context_mode = (
            neural_context_mode if is_neural_agent(agent) else "legacy_context"
        )
        evaluators.append(
            EvaluatorSpec(agent=agent, mode=mode, context_mode=context_mode)
        )

    return evaluators


def _build_evaluator_agents(
    maze_name: str,
    evaluators: list[EvaluatorInput],
    observable: bool,
    device: str,
    seed: int,
    horizon: int | None = None,
) -> dict[EvaluatorSpec, object]:
    """Instantiate one evaluator agent per spec for reuse across a run."""
    resolved_horizon = resolve_effective_horizon(maze_name, horizon)
    maze_spec = load_builtin_maze_spec(maze_name)
    evaluator_agents: dict[EvaluatorSpec, object] = {}
    for evaluator in (_normalize_evaluator(item) for item in evaluators):
        if is_neural_agent(evaluator.agent):
            configure_torch_worker(device)
        evaluator_agents[evaluator] = get_agent(
            evaluator.agent,
            Maze(maze_spec, horizon=resolved_horizon, observable=observable),
            device=device,
            init_mode=evaluator.mode,
            checkpoint_path=evaluator.checkpoint_path,
            context_mode=evaluator.context_mode,
            seed=seed,
        )
    return evaluator_agents


def evaluate_run_dataset(
    run_dataset: RunDataset,
    maze_name: str = "simple",
    evaluators: list[EvaluatorInput] | None = None,
    observable: bool = True,
    device: str = "auto",
    seed: int = DefaultParams.FRESH_EVALUATOR_SEED,
    source_context_mode: NeuralContextMode = "legacy_context",
    horizon: int | None = None,
) -> dict[EvaluatorSpec, np.ndarray]:
    """Evaluate one run dataset with persistent per-run evaluator state."""
    if evaluators is None:
        evaluators = registered_agents()

    resolved_horizon = resolve_effective_horizon(maze_name, horizon)
    evaluator_agents = _build_evaluator_agents(
        maze_name=maze_name,
        evaluators=evaluators,
        observable=observable,
        device=device,
        seed=seed,
        horizon=resolved_horizon,
    )
    results: dict[EvaluatorSpec, list[np.ndarray]] = {
        evaluator: [] for evaluator in evaluator_agents
    }

    for trajectory in run_dataset:
        for evaluator, agent in evaluator_agents.items():
            results[evaluator].append(np.array(agent.simulate(trajectory)))

    return {evaluator: np.concatenate(chunks) for evaluator, chunks in results.items()}


def _select_run_ids(
    source: Agent,
    maze_name: str,
    observable: bool,
    num_datasets: Optional[int],
    source_context_mode: NeuralContextMode = "legacy_context",
    horizon: int | None = None,
) -> list[int]:
    if is_neural_agent(source) and source_context_mode != "legacy_context":
        run_ids = list_run_dataset_run_ids(
            source,
            maze_name,
            observable,
            context_mode=source_context_mode,
            horizon=horizon,
        )
    else:
        run_ids = list_run_dataset_run_ids(
            source,
            maze_name,
            observable,
            horizon=horizon,
        )
    if num_datasets is None:
        return run_ids
    return run_ids[:num_datasets]


def _build_inference_tasks(
    source_agents: list[Agent],
    compare_to: list[EvaluatorInput],
    maze_name: str,
    num_datasets: Optional[int],
    observable: bool,
    device: str,
    base_seed: int,
    source_context_mode: NeuralContextMode,
    horizon: int | None,
) -> tuple[list[InferenceTask], list[Agent]]:
    tasks: list[InferenceTask] = []
    missing_sources: list[Agent] = []
    normalized_compare_to = tuple(_normalize_evaluator(item) for item in compare_to)

    for source in source_agents:
        run_ids = _select_run_ids(
            source,
            maze_name,
            observable,
            num_datasets,
            source_context_mode,
            horizon,
        )
        if not run_ids:
            missing_sources.append(source)
            continue

        tasks.extend(
            (
                source,
                run_id,
                maze_name,
                normalized_compare_to,
                observable,
                device,
                base_seed,
                source_context_mode,
                horizon,
            )
            for run_id in run_ids
        )

    return tasks, missing_sources


def _evaluate_dataset_task(task: InferenceTask) -> dict[str, object]:
    (
        source,
        run_id,
        maze_name,
        compare_to,
        observable,
        device,
        seed,
        source_context_mode,
        requested_horizon,
    ) = task

    start = time.perf_counter()
    metadata = load_run_dataset_metadata(
        source,
        run_id,
        maze_name,
        observable,
        context_mode=source_context_mode,
        horizon=requested_horizon,
    )
    expected_horizon = resolve_effective_horizon(maze_name, requested_horizon)
    source_horizon = int(metadata["horizon"])
    if source_horizon != expected_horizon:
        raise ValueError(
            "Saved run dataset horizon does not match the requested setting; "
            f"run_id={run_id} has horizon={source_horizon}, expected {expected_horizon}."
        )
    run_dataset = load_run_dataset(
        source,
        run_id,
        maze_name,
        observable,
        context_mode=source_context_mode,
        horizon=requested_horizon,
    )
    results = evaluate_run_dataset(
        run_dataset=run_dataset,
        maze_name=maze_name,
        evaluators=list(compare_to),
        observable=observable,
        device=device,
        seed=seed,
        source_context_mode=source_context_mode,
        horizon=source_horizon,
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
            source_context_mode=source_context_mode,
            horizon=source_horizon,
        )
        totals[evaluator.label] = float(np.sum(log_liks))

    return {
        "source": source,
        "run_id": run_id,
        "trajectory_length": run_dataset.num_transitions(),
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


def _execute_inference_tasks(
    tasks: list[InferenceTask],
    *,
    source_agents: list[Agent],
    workers: int | None,
    device: str,
    verbose: bool,
    uses_torch: bool,
    batch_label: str,
) -> None:
    """Execute one homogeneous inference batch."""
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
            f"Evaluating {task_count} trajectory dataset(s) across "
            f"{len(source_agents)} source agent(s) with {worker_count} worker(s) "
            f"for {batch_label} evaluators..."
        )
        if strategy.worker_note is not None:
            print(strategy.worker_note)

    start = time.perf_counter()

    if worker_count == 1:
        for completed, task in enumerate(tasks, start=1):
            result = _evaluate_dataset_task(task)
            if verbose:
                _print_inference_result(result, completed, task_count)
    else:
        executor_kwargs = {"max_workers": worker_count}
        if strategy.mp_context is not None:
            executor_kwargs["mp_context"] = strategy.mp_context
        with ProcessPoolExecutor(**executor_kwargs) as executor:
            futures = [executor.submit(_evaluate_dataset_task, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                if verbose:
                    _print_inference_result(result, completed, task_count)

    elapsed = time.perf_counter() - start
    if verbose:
        _print_timing_summary(task_count, worker_count, elapsed)


def run_inference_experiment(
    source_agents: list[Agent] | None = None,
    compare_to: list[EvaluatorInput] | None = None,
    maze_name: str = "simple",
    num_datasets: Optional[int] = None,
    observable: bool = True,
    verbose: bool = True,
    workers: int | None = None,
    device: str = "auto",
    base_seed: int = DefaultParams.FRESH_EVALUATOR_SEED,
    source_context_mode: NeuralContextMode = "legacy_context",
    horizon: int | None = None,
) -> None:
    """Run the full model inference experiment."""
    if source_agents is None:
        source_agents = registered_agents()
    if compare_to is None:
        compare_to = registered_agents()

    ensure_output_directories()

    normalized_compare_to = [_normalize_evaluator(item) for item in compare_to]
    evaluator_batches = build_torch_batches(normalized_compare_to, device=device)

    total_task_count = 0
    reported_missing_sources: set[Agent] = set()
    for evaluators, _, _ in evaluator_batches:
        if not evaluators:
            continue
        _, missing_sources = _build_inference_tasks(
            source_agents=source_agents,
            compare_to=evaluators,
            maze_name=maze_name,
            num_datasets=num_datasets,
            observable=observable,
            device=device,
            base_seed=base_seed,
            source_context_mode=source_context_mode,
            horizon=horizon,
        )
        for source in missing_sources:
            if source in reported_missing_sources:
                continue
            reported_missing_sources.add(source)
            print(
                f"No trajectory files for {source.value}. Run generate_trajectories.py first."
            )
        total_task_count += sum(
            len(
                _select_run_ids(
                    source,
                    maze_name,
                    observable,
                    num_datasets,
                    source_context_mode,
                    horizon,
                )
            )
            for source in source_agents
            if source not in reported_missing_sources
        )

    if total_task_count == 0:
        print("\nInference experiment complete!")
        return

    for evaluators, batch_uses_torch, batch_label in evaluator_batches:
        if not evaluators:
            continue
        tasks, _ = _build_inference_tasks(
            source_agents=source_agents,
            compare_to=evaluators,
            maze_name=maze_name,
            num_datasets=num_datasets,
            observable=observable,
            device=device,
            base_seed=base_seed,
            source_context_mode=source_context_mode,
            horizon=horizon,
        )
        _execute_inference_tasks(
            tasks,
            source_agents=source_agents,
            workers=workers,
            device=device,
            verbose=verbose,
            uses_torch=batch_uses_torch,
            batch_label=batch_label,
        )

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
        help="Evaluator agent name(s), e.g. mbrl, dqn:fresh, lstm:pretrained, or 'all'.",
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
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for neural evaluators: auto, cpu, cuda, or mps",
    )
    parser.add_argument(
        "--context-mode",
        choices=list(NEURAL_CONTEXT_MODES),
        default="legacy_context",
        help="Neural input context mode for source/evaluator neural policies.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DefaultParams.FRESH_EVALUATOR_SEED,
        help="Deterministic seed for fresh neural evaluators",
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
        help="Use partially observable trajectories (PO); default is fully observable (FO)",
    )

    args = parser.parse_args()

    run_inference_experiment(
        source_agents=_parse_agents(args.source_agents),
        compare_to=_parse_evaluators(args.compare_to, args.context_mode),
        maze_name=args.maze,
        num_datasets=args.num_datasets,
        observable=not args.pomdp,
        verbose=not args.quiet,
        workers=args.workers,
        device=args.device,
        base_seed=args.seed,
        source_context_mode=args.context_mode,
        horizon=args.horizon,
    )


if __name__ == "__main__":
    main()
