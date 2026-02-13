"""Helpers for loading maze specifications from TOML files."""

from pathlib import Path
from typing import Any
import tomllib

from .specs import MazeSpec
from forage_rl.config import MAZE_SPECS_DIR


def _load_spec_data(data: dict[str, Any], source: str) -> MazeSpec:
    try:
        return MazeSpec.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid maze spec at {source}: {exc}") from exc


def load_maze_spec(path: Path | str) -> MazeSpec:
    """Load and validate a maze spec from a TOML file path."""
    spec_path = Path(path).expanduser().resolve()
    if not spec_path.exists():
        raise ValueError(f"Maze spec file does not exist: {spec_path}")

    with spec_path.open("rb") as f:
        data = tomllib.load(f)
    return _load_spec_data(data, str(spec_path))


def load_builtin_maze_spec(name: str = "simple") -> MazeSpec:
    """Load a bundled maze spec by name (without file extension)."""
    spec = MAZE_SPECS_DIR.joinpath(name).with_suffix(".toml").resolve()
    if not spec.is_file():
        raise ValueError(f"Bundled maze spec '{name}' was not found")

    with spec.open("rb") as f:
        data = tomllib.load(f)
    return _load_spec_data(data, f"builtin:{name}")
