import tomllib

import pytest
from pydantic import ValidationError

from forage_rl.config import MAZE_SPECS_DIR
from forage_rl.environments.specs import MazeSpec


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

        assert len(result["transitions"]) == 4
        assert result["transitions"][0] == {
            "state": 0,
            "action": 0,
            "next_state": 0,
            "prob": 1.0,
        }
        assert result["transitions"][1] == {
            "state": 0,
            "action": 1,
            "next_state": 1,
            "prob": 1.0,
        }
        assert result["transitions"][2] == {
            "state": 1,
            "action": 0,
            "next_state": 1,
            "prob": 1.0,
        }
        assert result["transitions"][3] == {
            "state": 1,
            "action": 1,
            "next_state": 0,
            "prob": 1.0,
        }

    def test_expand_compact_format_preserves_observation_labels(self):
        with open(MAZE_SPECS_DIR / "simple.toml", "rb") as f:
            raw_data = tomllib.load(f)

        result = MazeSpec._expand_compact_format(raw_data)  # ty: ignore

        assert result["maze"]["observation_labels"] == ["Upper Patch", "Lower Patch"]

    def test_reward_probs_validation_rejects_out_of_range_values(self):
        with pytest.raises(ValidationError, match="reward_probs values must lie in"):
            MazeSpec.model_validate(
                {
                    "maze": {
                        "name": "invalid",
                        "horizon": 10,
                        "initial_state": 0,
                        "action_labels": ["stay", "leave"],
                    },
                    "states": [
                        {
                            "id": 0,
                            "label": "Upper Patch",
                            "reward_probs": [1.2],
                            "observation_group": 0,
                        },
                        {
                            "id": 1,
                            "label": "Lower Patch",
                            "decay": 0.5,
                            "observation_group": 1,
                        },
                    ],
                    "transitions": [
                        {"state": 0, "action": 0, "next_state": 0, "prob": 1.0},
                        {"state": 0, "action": 1, "next_state": 1, "prob": 1.0},
                        {"state": 1, "action": 0, "next_state": 1, "prob": 1.0},
                        {"state": 1, "action": 1, "next_state": 0, "prob": 1.0},
                    ],
                }
            )

    def test_initial_reward_prob_validation_rejects_out_of_range_values(self):
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            MazeSpec.model_validate(
                {
                    "maze": {
                        "name": "invalid",
                        "horizon": 10,
                        "initial_state": 0,
                        "action_labels": ["stay", "leave"],
                    },
                    "states": [
                        {
                            "id": 0,
                            "label": "Upper Patch",
                            "decay": 0.5,
                            "initial_reward_prob": 1.2,
                            "observation_group": 0,
                        },
                        {
                            "id": 1,
                            "label": "Lower Patch",
                            "decay": 0.5,
                            "observation_group": 1,
                        },
                    ],
                    "transitions": [
                        {"state": 0, "action": 0, "next_state": 0, "prob": 1.0},
                        {"state": 0, "action": 1, "next_state": 1, "prob": 1.0},
                        {"state": 1, "action": 0, "next_state": 1, "prob": 1.0},
                        {"state": 1, "action": 1, "next_state": 0, "prob": 1.0},
                    ],
                }
            )

    def test_reward_probs_reject_non_default_initial_reward_prob(self):
        with pytest.raises(
            ValidationError,
            match="initial_reward_prob cannot be used with reward_probs",
        ):
            MazeSpec.model_validate(
                {
                    "maze": {
                        "name": "invalid",
                        "horizon": 10,
                        "initial_state": 0,
                        "action_labels": ["stay", "leave"],
                    },
                    "states": [
                        {
                            "id": 0,
                            "label": "Upper Patch",
                            "reward_probs": [0.7, 0.5],
                            "initial_reward_prob": 0.7,
                            "observation_group": 0,
                        },
                        {
                            "id": 1,
                            "label": "Lower Patch",
                            "decay": 0.5,
                            "observation_group": 1,
                        },
                    ],
                    "transitions": [
                        {"state": 0, "action": 0, "next_state": 0, "prob": 1.0},
                        {"state": 0, "action": 1, "next_state": 1, "prob": 1.0},
                        {"state": 1, "action": 0, "next_state": 1, "prob": 1.0},
                        {"state": 1, "action": 1, "next_state": 0, "prob": 1.0},
                    ],
                }
            )
