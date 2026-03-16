"""Shared scheduling helpers for long-running experiment scripts."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, TypeVar

from forage_rl.agents.registry import AGENT_REGISTRY
from forage_rl.agents.torch_q import TorchQAgentBase, resolve_torch_device


TJob = TypeVar("TJob")
TResult = TypeVar("TResult")


def normalize_agent_names(agent_names: list[str]) -> list[str]:
    """Deduplicate agent names while preserving the first-seen order."""
    return list(dict.fromkeys(agent_names))


def validate_registered_agent_names(
    agent_names: list[str],
    label: str,
) -> list[str]:
    """Return deduplicated agent names after validating they are registered."""
    normalized_names = normalize_agent_names(agent_names)
    if not normalized_names:
        raise ValueError(f"{label} must not be empty")
    unknown_names = [name for name in normalized_names if name not in AGENT_REGISTRY]
    if unknown_names:
        unknown_display = ", ".join(repr(name) for name in unknown_names)
        raise ValueError(
            f"Unknown {label}: {unknown_display}. "
            f"Available: {list(AGENT_REGISTRY.keys())}"
        )
    return normalized_names


def agent_supported_for_env(agent_name: str, env_target) -> bool:
    """Return whether an agent supports the requested environment target."""
    return not (
        agent_name == "value_iteration"
        and getattr(env_target, "observability", None) == "pomdp"
    )


def split_supported_agent_names(
    agent_names: list[str],
    env_target,
) -> tuple[list[str], list[str]]:
    """Split validated agent names into supported and unsupported subsets."""
    supported: list[str] = []
    unsupported: list[str] = []
    for agent_name in agent_names:
        if agent_supported_for_env(agent_name, env_target):
            supported.append(agent_name)
        else:
            unsupported.append(agent_name)
    return supported, unsupported


def format_unsupported_agents_message(
    agent_names: list[str],
    env_target,
) -> str:
    """Return a concise message for skipped unsupported agent/environment pairs."""
    return (
        f"Skipping unsupported agents for {env_target.label}: {', '.join(agent_names)}"
    )


def validate_positive_int(value: int, name: str) -> int:
    """Return a validated positive integer argument."""
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def validate_optional_positive_int(value: int | None, name: str) -> int | None:
    """Return a validated optional positive integer argument."""
    if value is None:
        return None
    return validate_positive_int(value, name)


def is_deep_agent(agent_name: str) -> bool:
    """Return whether a registered agent is Torch-based."""
    factory = AGENT_REGISTRY.get(agent_name)
    return factory is not None and issubclass(factory, TorchQAgentBase)


def resolved_requested_device(agent_name: str, device: str) -> str | None:
    """Resolve the requested device for a deep agent, or None for tabular agents."""
    if not is_deep_agent(agent_name):
        return None
    return str(resolve_torch_device(device))


def should_pool_job(agent_name: str, device: str) -> bool:
    """Return whether a job is safe to execute in the CPU worker pool."""
    resolved_device = resolved_requested_device(agent_name, device)
    return resolved_device is None or not resolved_device.startswith("cuda")


def run_jobs(
    *,
    jobs: list[TJob],
    jobs_per_process: int,
    should_pool: Callable[[TJob], bool],
    run_job: Callable[[TJob], TResult],
    handle_result: Callable[[TResult], None],
    format_error: Callable[[TJob], str],
) -> None:
    """Run jobs via a CPU process pool when safe, else sequentially."""
    pooled_jobs = [job for job in jobs if should_pool(job)]
    sequential_jobs = [job for job in jobs if not should_pool(job)]

    if jobs_per_process > 1 and pooled_jobs:
        max_workers = min(jobs_per_process, len(pooled_jobs))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {executor.submit(run_job, job): job for job in pooled_jobs}
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - exercised via tests
                    raise RuntimeError(format_error(job)) from exc
                handle_result(result)
    else:
        sequential_jobs = pooled_jobs + sequential_jobs

    for job in sequential_jobs:
        handle_result(run_job(job))


__all__ = [
    "agent_supported_for_env",
    "format_unsupported_agents_message",
    "is_deep_agent",
    "normalize_agent_names",
    "resolved_requested_device",
    "run_jobs",
    "should_pool_job",
    "split_supported_agent_names",
    "validate_optional_positive_int",
    "validate_positive_int",
    "validate_registered_agent_names",
]
