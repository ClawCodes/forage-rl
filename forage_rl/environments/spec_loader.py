"""Helpers for loading maze specifications from TOML files."""

from pathlib import Path
import tomllib

from .specs import MazeSpec
from forage_rl.config import MAZE_SPECS_DIR


def _load_spec_data(path: Path) -> MazeSpec:
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
        return MazeSpec.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid maze spec at {path}") from exc


def load_maze_spec(path: Path | str) -> MazeSpec:
    """Load and validate a maze spec from a TOML file path."""
    spec_path = Path(path).expanduser().resolve()
    if not spec_path.exists():
        raise ValueError(f"Maze spec file does not exist: {spec_path}")

    return _load_spec_data(spec_path)


def load_builtin_maze_spec(name: str = "simple") -> MazeSpec:
    """Load an existing maze spec from a spec name."""
    spec_path = MAZE_SPECS_DIR.joinpath(name).with_suffix(".toml").resolve()
    if not spec_path.is_file():
        raise ValueError(f"Maze spec '{name}' was not found")

    return _load_spec_data(spec_path)
