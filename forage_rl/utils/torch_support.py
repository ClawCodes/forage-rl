"""Torch device helpers that keep non-neural installs importable."""

from __future__ import annotations

import importlib
from types import ModuleType


def _load_torch() -> ModuleType:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for neural-agent support. Install the optional "
            "'neural' dependency group or a compatible torch build first."
        ) from exc


def torch_available() -> bool:
    """Return whether torch can be imported in the current environment."""
    try:
        importlib.import_module("torch")
    except ModuleNotFoundError:
        return False
    return True


def require_torch() -> ModuleType:
    """Import torch or raise a clear error for neural-agent callers."""
    return _load_torch()


def resolve_device(device: str = "auto") -> str:
    """Resolve a requested device string to cuda, mps, or cpu."""
    normalized = device.lower()
    if normalized not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(
            f"Unsupported device {device!r}. Expected one of auto, cpu, cuda, mps."
        )

    if normalized == "cpu":
        return "cpu"

    if not torch_available():
        if normalized == "auto":
            return "cpu"
        raise ValueError(
            f"Requested device {normalized!r}, but PyTorch is not installed."
        )

    torch = _load_torch()

    if normalized == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(torch.backends, "mps", None)
        if (
            mps_backend is not None
            and mps_backend.is_built()
            and mps_backend.is_available()
        ):
            return "mps"
        return "cpu"

    if normalized == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested device 'cuda', but CUDA is unavailable.")

    if normalized == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if (
            mps_backend is None
            or not mps_backend.is_built()
            or not mps_backend.is_available()
        ):
            raise ValueError(
                "Requested device 'mps', but Apple Metal (MPS) is unavailable."
            )

    return normalized


def configure_torch_worker(device: str) -> None:
    """Clamp Torch CPU thread usage inside worker processes."""
    resolved_device = resolve_device(device)
    if resolved_device != "cpu" or not torch_available():
        return
    torch = _load_torch()
    torch.set_num_threads(1)
