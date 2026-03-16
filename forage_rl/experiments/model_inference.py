"""Run cross-agent model inference on saved trajectories."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from forage_rl.agents import get_agent, registered_agents
from forage_rl.config import DefaultParams, ensure_directories
from forage_rl.environments import (
    EnvironmentTarget,
    build_environment,
    normalize_environment_targets,
)
from forage_rl.experiments import scheduling
from forage_rl.utils import get_trajectory_run_ids, load_trajectories, save_logprobs


@dataclass(frozen=True)
class _InferenceJob:
    env_target: EnvironmentTarget
    source_agent: str
    evaluator_name: str
    run_id: int
    num_episodes: int
    base_seed: int
    device: str


@dataclass(frozen=True)
class _InferenceResult:
    env_target: EnvironmentTarget
    source_agent: str
    evaluator_name: str
    run_id: int
    filepath: Path | None
    resolved_device: str | None
    missing_trajectory: bool = False


def _run_inference_job(job: _InferenceJob) -> _InferenceResult:
    run_seed = job.base_seed + job.run_id
    try:
        trajectory = load_trajectories(
            job.source_agent,
            job.run_id,
            env_key=job.env_target.key,
        )
    except FileNotFoundError:
        return _InferenceResult(
            env_target=job.env_target,
            source_agent=job.source_agent,
            evaluator_name=job.evaluator_name,
            run_id=job.run_id,
            filepath=None,
            resolved_device=None,
            missing_trajectory=True,
        )

    maze = build_environment(job.env_target, seed=run_seed)
    evaluator = get_agent(
        job.evaluator_name,
        maze,
        num_episodes=job.num_episodes,
        seed=run_seed,
        device=job.device,
    )
    log_likelihoods = evaluator.simulate(trajectory)
    cumulative = np.cumsum(log_likelihoods, dtype=float)
    label = f"source_{job.source_agent}_eval_{job.evaluator_name}"
    filepath = save_logprobs(
        cumulative,
        label,
        job.run_id,
        env_key=job.env_target.key,
    )
    resolved_device = str(evaluator.device) if hasattr(evaluator, "device") else None
    return _InferenceResult(
        env_target=job.env_target,
        source_agent=job.source_agent,
        evaluator_name=job.evaluator_name,
        run_id=job.run_id,
        filepath=filepath,
        resolved_device=resolved_device,
    )


def _log_inference_result(
    result: _InferenceResult,
    reported_devices: set[tuple[str, str]],
    verbose: bool,
) -> None:
    if not verbose:
        return
    if result.missing_trajectory:
        print(
            f"Skipping missing trajectory file for {result.source_agent} "
            f"run {result.run_id}"
        )
        return
    if result.resolved_device is not None:
        device_key = (result.evaluator_name, result.resolved_device)
        if device_key not in reported_devices:
            print(
                f"Using device {result.resolved_device} "
                f"for evaluator {result.evaluator_name} on {result.env_target.label}"
            )
            reported_devices.add(device_key)
    label = f"source_{result.source_agent}_eval_{result.evaluator_name}"
    print(f"Saved {label} to {result.filepath}")


def _run_inference_jobs(
    jobs: list[_InferenceJob],
    jobs_per_process: int,
    verbose: bool,
) -> None:
    reported_devices: set[tuple[str, str]] = set()

    def handle_result(result: _InferenceResult) -> None:
        _log_inference_result(
            result=result,
            reported_devices=reported_devices,
            verbose=verbose,
        )

    scheduling.run_jobs(
        jobs=jobs,
        jobs_per_process=jobs_per_process,
        should_pool=lambda job: scheduling.should_pool_job(
            job.evaluator_name, job.device
        ),
        run_job=_run_inference_job,
        handle_result=handle_result,
        format_error=lambda job: (
            "Model inference failed for "
            f"source={job.source_agent} eval={job.evaluator_name} "
            f"run_id={job.run_id}"
        ),
    )


def run_model_inference(
    env_target: EnvironmentTarget,
    source_agents: list[str],
    eval_agents: list[str],
    num_runs: int = DefaultParams.NUM_TRAINING_RUNS,
    num_episodes: int = DefaultParams.NUM_EPISODES,
    base_seed: int = DefaultParams.BASE_SEED,
    device: str = "auto",
    jobs: int = 1,
    verbose: bool = True,
) -> None:
    """Score saved trajectories under every selected evaluator agent."""
    scheduling.validate_positive_int(num_runs, "num_runs")
    scheduling.validate_positive_int(num_episodes, "num_episodes")
    scheduling.validate_positive_int(jobs, "jobs")
    ensure_directories()
    source_agents = scheduling.validate_registered_agent_names(
        source_agents,
        "source_agents",
    )
    eval_agents = scheduling.validate_registered_agent_names(
        eval_agents,
        "eval_agents",
    )
    source_agents, unsupported_source_agents = scheduling.split_supported_agent_names(
        source_agents,
        env_target,
    )
    eval_agents, unsupported_eval_agents = scheduling.split_supported_agent_names(
        eval_agents,
        env_target,
    )
    unsupported_agents = scheduling.normalize_agent_names(
        unsupported_source_agents + unsupported_eval_agents
    )
    if unsupported_agents and verbose:
        print(
            scheduling.format_unsupported_agents_message(
                unsupported_agents,
                env_target,
            )
        )
    if not source_agents or not eval_agents:
        return
    inference_jobs: list[_InferenceJob] = []

    for source_agent in source_agents:
        source_run_ids = get_trajectory_run_ids(
            source_agent,
            env_key=env_target.key,
        )[:num_runs]

        if not source_run_ids:
            if verbose:
                print(
                    f"No trajectory files found for {source_agent}. "
                    "Run trajectory generation first."
                )
            continue

        requested_runs = len(source_run_ids)
        for run_index, run_id in enumerate(source_run_ids, start=1):
            if verbose:
                print(
                    f"\n{'=' * 60}\n"
                    f"Source {source_agent} run {run_index}/{requested_runs} (id={run_id})\n"
                    f"{'=' * 60}"
                )

            for evaluator_name in eval_agents:
                inference_jobs.append(
                    _InferenceJob(
                        env_target=env_target,
                        source_agent=source_agent,
                        evaluator_name=evaluator_name,
                        run_id=run_id,
                        num_episodes=num_episodes,
                        base_seed=base_seed,
                        device=device,
                    )
                )

    _run_inference_jobs(
        jobs=inference_jobs,
        jobs_per_process=jobs,
        verbose=verbose,
    )


def main() -> None:
    """Parse CLI args and run cross-agent model inference."""
    parser = argparse.ArgumentParser(
        description="Evaluate saved trajectories with selected agents"
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["all"],
        help="Trajectory source agent(s), or 'all' for every registered agent",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="Environment target(s): simple:full, full:full, full:pomdp",
    )
    parser.add_argument(
        "--eval-agents",
        nargs="+",
        default=None,
        help="Evaluator agent(s), or 'all' for every registered agent",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=DefaultParams.NUM_TRAINING_RUNS,
        help="Number of saved trajectory runs to evaluate per source agent",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=DefaultParams.NUM_EPISODES,
        help="Training episodes used when replaying evaluator learning dynamics",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=DefaultParams.BASE_SEED,
        help="Base seed used to derive deterministic per-run evaluator seeds",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for deep agents: auto, cpu, cuda, or cuda:N",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Worker processes for CPU-safe jobs; CUDA deep evaluators stay sequential",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()
    verbose = not args.quiet

    source_agents = scheduling.normalize_agent_names(
        registered_agents() if args.agents == ["all"] else args.agents
    )
    env_targets = normalize_environment_targets(args.envs)
    if args.eval_agents is None or args.eval_agents == ["all"]:
        eval_agents = scheduling.normalize_agent_names(registered_agents())
    else:
        eval_agents = scheduling.normalize_agent_names(args.eval_agents)

    for env_target in env_targets:
        run_model_inference(
            env_target=env_target,
            source_agents=source_agents,
            eval_agents=eval_agents,
            num_runs=args.num_runs,
            num_episodes=args.num_episodes,
            base_seed=args.base_seed,
            device=args.device,
            jobs=args.jobs,
            verbose=verbose,
        )

    if verbose:
        print("\nModel inference complete!")


if __name__ == "__main__":
    main()
