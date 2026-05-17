"""Utility modules for file I/O and data handling."""

from typing import Any

__all__ = [
    "checkpoint_metadata_path",
    "checkpoint_path",
    "get_run_count",
    "list_run_dataset_run_ids",
    "load_checkpoint_metadata",
    "load_logprobs",
    "load_run_dataset",
    "load_run_dataset_metadata",
    "save_logprobs",
    "save_run_dataset",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import io

        value = getattr(io, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
