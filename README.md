# forage-rl

Foraging reinforcement learning experiments comparing tabular and neural agents
across trajectory generation, model inference, and artifact-regeneration
workflows.

## Install

Base install:

```bash
uv sync
```

Neural agents use the optional `neural` extra:

```bash
uv sync --extra neural
```

For CUDA or Apple Silicon acceleration, install a PyTorch build appropriate for
your machine.

## Agents

- `mbrl`
- `q_learning`
- `dqn`
- `elman`
- `gru`
- `lstm`

`dqn`, `elman`, `gru`, and `lstm` support `cpu`, `cuda`, and Apple Silicon
`mps` through PyTorch device resolution. Neural workloads clamp to one worker
on `cuda` and `mps`; CPU neural workloads still use multiprocessing with
`spawn`.

`drqn` remains accepted as a legacy alias for `lstm`, but canonical saved
artifacts and benchmark labels now use `lstm`.

## Legacy Artifact Pipeline

Run the older artifact pipeline with:

```bash
uv run python -m forage_rl.experiments.regenerate_artifacts --train-pretrained --device auto
```

By default it:

- refreshes canonical neural checkpoints in `data/checkpoints/`
- generates run datasets for `mbrl`, `q_learning`, `dqn`, `elman`, `gru`, and
  `lstm` in `data/trajectories/`
- runs inference with tabular fresh evaluators plus all neural agents in both
  fresh and pretrained modes, writing log-likelihood outputs to
  `data/logprobs/`
- renders figures to `outputs/figures/`

The default settings cover `simple/FO`, `full/FO`, and `full/PO`.

Aggregate trajectory figures are source-policy plots from generated run
datasets. `fresh` and `pretrained` evaluator modes apply only to the model
inference comparison outputs.

## Fast Long-Horizon Smoke Graphs

For a quick smoke run that renders `full/FO` and `full/PO` graphs with
600-step trajectories and no pretrained-checkpoint retraining, run:

```bash
uv run python -m forage_rl.experiments.regenerate_artifacts \
  --mazes full \
  --observability all \
  --num-runs 2 \
  --num-episodes 1 \
  --num-datasets 2 \
  --horizon 600 \
  --evaluator-mode fresh \
  --device cpu
```

Interpretation:

- `--num-episodes 1` means one 600-step episode per run dataset
- aggregate trajectory figures are the main full-trajectory graphs for this
  smoke test
- episode-return figures will show one point per agent, which is expected here
- aggregate trajectory plots remain episode-aligned and should span
  `Transition Within Episode = 1..600`

Figure x-axis taxonomy:

- episode return plots: `Episode Within Run`
- aggregate trajectory plots: `Transition Within Episode`
- likelihood diagnostics: `Observed Transitions`

Aggregate trajectory plots now pool all episodes from one homogeneous matched
cohort of saved runs. When mixed saved cohorts exist, they automatically select
the longest homogeneous matched cohort before plotting.

Use `--horizon` on generation, inference, pretrained-checkpoint training, and
artifact regeneration when you want longer or shorter episodes than the built-in
maze default. `--num-episodes` changes how many episodes are generated per run;
`--horizon` changes how many transitions each episode can contain.

## Trajectory Generation

```bash
python -m forage_rl.experiments.generate_trajectories --agents mbrl q_learning dqn elman gru lstm --maze full --device auto
```

`Trajectory` now means exactly one episode. One training run is stored as a
`RunDataset`, which is the persisted unit under `data/trajectories/`.

The canonical artifact pipeline now generates `100` episodes per run dataset by
default so learning curves and aggregate trajectory plots reflect later-stage
behavior instead of the first few episodes only.

Run dataset paths:

- `data/trajectories/{maze}_{FO|PO}_{agent}_run_dataset_{run_id}.npz`
- `data/trajectories/{maze}_{FO|PO}_{agent}_run_dataset_{run_id}.json`

Custom-horizon run datasets add `_h{horizon}` immediately after `{FO|PO}` so
they remain distinct from the built-in default-horizon artifacts.

Available neural context modes:

- `observation_only`: `observation_one_hot`
- `prev_reward`: `observation_one_hot + prev_reward`
- `prev_reward_time`: `observation_one_hot + normalized_time_spent + prev_reward`
- `legacy_context`:
  `observation_one_hot + normalized_time_spent + prev_reward + prev_action_one_hot`

Non-legacy neural artifacts use distinct suffixes so they do not collide:

- `observation_only -> _obs_only`
- `prev_reward -> _prev_reward`
- `prev_reward_time -> _prev_reward_time`

## Pretrained Neural Evaluators

Train canonical legacy final checkpoints for all neural agents with:

```bash
python -m forage_rl.experiments.train_pretrained_agents --agents all --maze simple --device auto
```

Canonical checkpoint paths:

- `data/checkpoints/{maze}_{FO|PO}_{agent}_final.pt`
- `data/checkpoints/{maze}_{FO|PO}_{agent}_final.json`

Custom-horizon checkpoints also use an `_h{horizon}` suffix after `{FO|PO}`.

## Inference

Evaluator tokens support fresh and pretrained neural evaluators:

```bash
python -m forage_rl.experiments.model_inference \
  --source-agents mbrl q_learning \
  --compare-to mbrl q_learning dqn:fresh dqn:pretrained elman:fresh elman:pretrained gru:fresh gru:pretrained lstm:fresh lstm:pretrained \
  --maze simple \
  --device auto
```

`all` expands to every registered agent in fresh mode only. Fresh neural
evaluators are deterministic by default and seeded with `0` unless overridden
via `--seed`.

Run-level inference preserves evaluator learning state across episodes within a
single `RunDataset`, while recurrent hidden state resets at each episode
boundary.

Inference, checkpoint loading, and plotting are horizon-strict: a requested
custom horizon only uses artifacts generated for that exact horizon, and stale
artifacts without horizon metadata must be regenerated.

## Maze Specs

Built-in mazes live under `forage_rl/environments/maze_specs/`. The filename
without extension becomes the maze name accepted by the experiment entrypoints.

To load a custom maze spec in Python:

```python
from forage_rl.environments import Maze
from forage_rl.environments.spec_loader import load_maze_spec

spec = load_maze_spec("/path/to/my_maze.toml")
maze = Maze(spec)
```

Maze specs are TOML files validated by `MazeSpec`. State ids must be contiguous,
transition probabilities for each `(state, action)` pair must sum to `1.0`, and
transition formats with and without duration fields cannot be mixed in the same
file.
