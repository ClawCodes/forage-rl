from forage_rl.environments.specs import MazeSpec
from forage_rl.config import MAZE_SPECS_DIR
import tomllib


class TestSpecs:
    def test_expand_compact_format_returns_flattened_simple_maze(self):
        # Arrange
        with open(MAZE_SPECS_DIR / "simple.toml", "rb") as f:
            raw_data = tomllib.load(f)

        # Verify input is compact format (states is a dict, not a list)
        assert isinstance(raw_data["states"], dict)

        # Act
        result = MazeSpec._expand_compact_format(raw_data)  # ty: ignore

        # Assert
        # States expanded from dict to list of dicts
        assert len(result["states"]) == 2
        assert result["states"][0] == {
            "id": 0,
            "label": "Upper Patch",
            "decay": 0.2,
            "observation_group": 0,
        }
        assert result["states"][1] == {
            "id": 1,
            "label": "Lower Patch",
            "decay": 3.0,
            "observation_group": 1,
        }

        # Transitions expanded from action-name keyed dicts to flat list of dicts
        assert len(result["transitions"]) == 4
        # State 0 stay -> action index 0
        assert result["transitions"][0] == {
            "state": 0,
            "action": 0,
            "next_state": 0,
            "prob": 1.0,
        }
        # State 0 leave -> action index 1
        assert result["transitions"][1] == {
            "state": 0,
            "action": 1,
            "next_state": 1,
            "prob": 1.0,
        }
        # State 1 stay -> action index 0
        assert result["transitions"][2] == {
            "state": 1,
            "action": 0,
            "next_state": 1,
            "prob": 1.0,
        }
        # State 1 leave -> action index 1
        assert result["transitions"][3] == {
            "state": 1,
            "action": 1,
            "next_state": 0,
            "prob": 1.0,
        }
