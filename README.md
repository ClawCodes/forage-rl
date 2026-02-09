# forage-rl

Config-driven reinforcement learning sandbox for foraging mazes.

## Overview

The environment is defined by a TOML maze specification:
- states and decay rates
- action labels
- transition probabilities per `(state, action)`

`Maze()` defaults to the built-in 2-state simple maze.

## Quick Start

```bash
uv run python -m forage_rl.experiments.generate_trajectories --algo both --num-runs 5 --num-episodes 6
uv run python -m forage_rl.experiments.model_inference --num-datasets 5
```

Use a custom maze file:

```bash
uv run python -m forage_rl.experiments.generate_trajectories --maze-config forage_rl/maze_specs/full.toml
uv run python -m forage_rl.experiments.model_inference --maze-config forage_rl/maze_specs/full.toml
```

## Maze Specs

Built-in specs:
- `forage_rl/maze_specs/simple.toml` (default)
- `forage_rl/maze_specs/full.toml` (example 6-state stochastic)

Canonical shape:

```toml
[maze]
name = "simple"
horizon = 100
initial_state = 0
action_labels = ["stay", "leave"]

[[states]]
id = 0
label = "Upper Patch"
decay = 0.2
observation_group = 0

[[states]]
id = 1
label = "Lower Patch"
decay = 3.0
observation_group = 1

[[transitions]]
state = 0
action = 0
next_state = 0
prob = 1.0
```

Validation rules:
- state IDs must be contiguous `0..N-1`
- each state has non-negative `decay`
- each state must define `observation_group`
- observation groups must be contiguous `0..K-1`
- each `(state, action)` has at least one transition row
- probabilities for each `(state, action)` sum to `1.0`
- `next_state` and `initial_state` must be in bounds
- actions must be in `0..num_actions-1`

Transition timing modes are inferred from transition rows:
- Step mode (default): no transition rows define `duration`; each step advances time by `1`.
- Duration mode: all transition rows define integer `duration >= 1`; mixed rows are invalid.

## API Notes

- `Maze()` -> default simple spec
- `Maze.from_file(path)` -> load custom TOML
- `Maze.from_spec(spec)` -> load validated `MazeSpec`
- `Maze.transition_distribution(state, action)` -> planner-facing probabilities
- `Maze.transition_duration(state, action, next_state)` -> time cost of that transition (`1` in step mode)
- `Maze.step(action)` -> `(Transition, done)`

Inference methods require timed trajectories:
- `QLearningTime.simulate_q_learning(...)`
- `MBRL.simulate_model_based_rl(...)`

## Reproducibility

Environment stochasticity is controlled via `seed`:

```python
from forage_rl.environments import Maze
maze = Maze(seed=123)
```

Experiment CLIs also accept `--seed`.

`model_inference.py` clamps `--num-datasets` to available files and prints a warning
instead of crashing if the requested count is too large.
