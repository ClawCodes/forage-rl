"""Generate training trajectories for registered agents."""

import argparse
from dataclasses import dataclass
from pathlib import Path

from forage_rl.agents import get_agent, registered_agents
from forage_rl.config import DefaultParams, ensure_directories
from forage_rl.environments import (
    EnvironmentTarget,
    build_environment,
    normalize_environment_targets,
)
from forage_rl.experiments import scheduling
from forage_rl.utils import save_trajectories


@dataclass(frozen=True)
class _GenerationJob:
    env_target: EnvironmentTarget
    agent_type: str
    run_id: int
    num_episodes: int
    base_seed: int
    device: str


@dataclass(frozen=True)
class _GenerationResult:
    env_target: EnvironmentTarget
    agent_type: str
    run_id: int
    filepath: Path
    num_transitions: int
    resolved_device: str | None


def _run_generation_job(job: _GenerationJob) -> _GenerationResult:
    run_seed = job.base_seed + job.run_id
    maze = build_environment(job.env_target, seed=run_seed)
    agent = get_agent(
        job.agent_type,
        maze,
        num_episodes=job.num_episodes,
        seed=run_seed,
        device=job.device,
    )
    transitions = agent.train(verbose=False)
    filepath = save_trajectories(
        transitions,
        job.agent_type,
        job.run_id,
        env_key=job.env_target.key,
    )
    resolved_device = str(agent.device) if hasattr(agent, "device") else None
    return _GenerationResult(
        env_target=job.env_target,
        agent_type=job.agent_type,
        run_id=job.run_id,
        filepath=filepath,
        num_transitions=len(transitions),
        resolved_device=resolved_device,
    )


def _log_generation_result(
    result: _GenerationResult,
    num_runs_by_agent: dict[str, int],
    reported_devices: set[tuple[str, str]],
    verbose: bool,
) -> None:
    if not verbose:
        return
    if result.resolved_device is not None:
        device_key = (result.agent_type, result.resolved_device)
        if device_key not in reported_devices:
            print(
                f"Using device {result.resolved_device} for "
                f"{result.agent_type} on {result.env_target.label}"
            )
            reported_devices.add(device_key)
    print(
        f"Saved {result.num_transitions} transitions for "
        f"{result.env_target.label} {result.agent_type} "
        f"run {result.run_id + 1}/{num_runs_by_agent[result.agent_type]} "
        f"(id={result.run_id}) to {result.filepath}"
    )


def _run_generation_jobs(
    jobs: list[_GenerationJob],
    jobs_per_process: int,
    verbose: bool,
) -> None:
    reported_devices: set[tuple[str, str]] = set()
    num_runs_by_agent: dict[str, int] = {}
    for job in jobs:
        num_runs_by_agent[job.agent_type] = num_runs_by_agent.get(job.agent_type, 0) + 1

    def handle_result(result: _GenerationResult) -> None:
        _log_generation_result(
            result=result,
            num_runs_by_agent=num_runs_by_agent,
            reported_devices=reported_devices,
            verbose=verbose,
        )

    scheduling.run_jobs(
        jobs=jobs,
        jobs_per_process=jobs_per_process,
        should_pool=lambda job: scheduling.should_pool_job(job.agent_type, job.device),
        run_job=_run_generation_job,
        handle_result=handle_result,
        format_error=lambda job: (
            f"Trajectory generation failed for {job.agent_type} run_id={job.run_id}"
        ),
    )


def generate_trajectories(
    env_target: EnvironmentTarget,
    agent_type: str,
    num_runs: int = DefaultParams.NUM_TRAINING_RUNS,
    num_episodes: int = DefaultParams.NUM_TRAINING_EPISODES,
    base_seed: int = DefaultParams.BASE_SEED,
    device: str = "auto",
    jobs: int = 1,
    verbose: bool = True,
) -> None:
    """Generate trajectories from a registered agent.

    Args:
        env_target: Environment target describing the maze and observability.
        agent_type: Name of the agent in the registry.
        num_runs: Number of independent training runs.
        num_episodes: Episodes per run.
        base_seed: Base seed; run i uses seed (base_seed + i).
        device: Requested torch device for deep agents; ignored by tabular agents.
        jobs: Number of worker processes for CPU-safe jobs.
        verbose: Whether to print progress.
    """
    scheduling.validate_positive_int(num_runs, "num_runs")
    scheduling.validate_positive_int(num_episodes, "num_episodes")
    scheduling.validate_positive_int(jobs, "jobs")
    agent_type = scheduling.validate_registered_agent_names([agent_type], "agent_type")[
        0
    ]
    _, unsupported_agents = scheduling.split_supported_agent_names(
        [agent_type],
        env_target,
    )
    if unsupported_agents:
        if verbose:
            print(
                scheduling.format_unsupported_agents_message(
                    unsupported_agents,
                    env_target,
                )
            )
        return

    ensure_directories()
    generation_jobs = [
        _GenerationJob(
            env_target=env_target,
            agent_type=agent_type,
            run_id=run_id,
            num_episodes=num_episodes,
            base_seed=base_seed,
            device=device,
        )
        for run_id in range(num_runs)
    ]
    _run_generation_jobs(
        jobs=generation_jobs,
        jobs_per_process=jobs,
        verbose=verbose,
    )


def main() -> None:
    """Parse CLI args and generate trajectories for selected agent(s)."""
    parser = argparse.ArgumentParser(description="Generate training trajectories")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["all"],
        help="Agent name(s) to run, or 'all' for every registered agent",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="Environment target(s): simple:full, full:full, full:pomdp",
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
        "--base-seed",
        type=int,
        default=DefaultParams.BASE_SEED,
        help="Base seed used to derive deterministic per-run seeds",
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
        help="Worker processes for CPU-safe jobs; CUDA deep-agent jobs stay sequential",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()
    verbose = not args.quiet

    agents = scheduling.normalize_agent_names(
        registered_agents() if args.agents == ["all"] else args.agents
    )
    env_targets = normalize_environment_targets(args.envs)

    for env_target in env_targets:
        env_agents, unsupported_agents = scheduling.split_supported_agent_names(
            agents,
            env_target,
        )
        if unsupported_agents and verbose:
            print(
                scheduling.format_unsupported_agents_message(
                    unsupported_agents,
                    env_target,
                )
            )
        for agent_type in env_agents:
            if verbose:
                print(
                    f"\nGenerating {agent_type} trajectories for {env_target.label}..."
                )
            generate_trajectories(
                env_target=env_target,
                agent_type=agent_type,
                num_runs=args.num_runs,
                num_episodes=args.num_episodes,
                base_seed=args.base_seed,
                device=args.device,
                jobs=args.jobs,
                verbose=verbose,
            )

    if verbose:
        print("\nTrajectory generation complete!")


if __name__ == "__main__":
    main()
