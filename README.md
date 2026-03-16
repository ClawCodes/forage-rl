# forage-rl

`forage-rl` is a compact research codebase for comparing model-based,
tabular model-free, and deep replay-based RL agents on a foraging-style maze
task. The core workflow is:

1. train agents and save trajectories
2. replay those trajectories under candidate evaluator models
3. compare cumulative log-likelihoods and rendered plots

The supported agent set is intentionally small:

- `mbrl`: tabular model-based baseline
- `q_learning`: tabular model-free baseline
- `dqn`: feed-forward DQN with replay and a target network
- `rdqn`: recurrent DQN with replay and a target network

## Quickstart

Install dependencies and run the test suite:

```bash
uv sync
uv run pytest -q
```

`uv run pytest -q` is the supported local test command. Plain `pytest -q`
is not guaranteed to work unless `forage_rl` is already installed into the
active Python environment.

Deep-agent scripts accept `--device auto|cpu|cuda|cuda:N`. The default is
`auto`, which uses CUDA when available and safely falls back to CPU otherwise.
The long-running generation and inference scripts also accept `--jobs N` for
CPU-safe parallelism. `--jobs` and `--num-runs` must be positive integers.
CUDA deep-agent jobs stay sequential by default to avoid GPU contention.
Experiment APIs reject empty agent lists instead of silently no-oping.

The main experiment scripts also accept `--envs` so the same pipeline can run
over multiple environment targets in one command. The supported targets are:

- `simple:full`
- `full:full`
- `full:pomdp`

## Repository layout

- `forage_rl/agents`: agent implementations plus the shared registry used by the main pipeline.
- `forage_rl/environments`: maze environments and built-in maze specs. `Maze` and `MazePOMDP` are the primary entry points; `SimpleMaze` remains as a compatibility shim for the default two-patch setup.
- `forage_rl/experiments`: runnable scripts for trajectory generation, replay inference, plotting, and scheduling helpers.
- `forage_rl/visualization`: metric helpers and plotting functions for saved inference outputs.
- `forage_rl/utils`: trajectory/log-prob I/O and saved run discovery helpers.
- `forage_rl/types.py`: transition and trajectory schemas used in memory and on disk.
- `tests`: regression coverage for agents, environments, experiments, I/O, and plotting.
- Built-in maze specs live in `forage_rl/environments/maze_specs`. Custom specs can be loaded from file paths with `Maze.from_file(...)` / `load_maze_spec(...)`.
- `data/trajectories`: saved rollout datasets keyed by environment, source agent, and run id.
- `data/logprobs`: saved cumulative replay log-likelihood arrays keyed by environment, source/evaluator/run.
- `outputs/figures`: generated PNG summaries produced by the comparison pipeline.

For POMDP specs, observation-level timing is inferred from the transition structure itself rather than action labels. Custom aliased-observation specs are supported as long as each `(observation, action, next_observation)` edge implies one consistent elapsed time and reset behavior.

## Maze spec format

Built-in examples live in `forage_rl/environments/maze_specs/simple.toml` and
`forage_rl/environments/maze_specs/full.toml`.

Specs use a compact TOML format with one `[states.<id>]` table per concrete
state and a nested `[states.<id>.transitions]` table keyed by action label:

```toml
[maze]
name = "simple"
horizon = 100
initial_state = 0
action_labels = ["stay", "leave"]

[states.0]
label = "Upper Patch"
decay = 0.2
observation_group = 0
[states.0.transitions]
stay = [[0, 1.0]]
leave = [[1, 1.0]]
```

To model variable-time edges, add a third value to each transition outcome:
`[next_state, prob, duration]`.

`observation_group` controls partial observability. States that share the same
group collapse to the same observation in `MazePOMDP`; meaningful POMDP specs
therefore need at least one aliased group, and each aliased
`(observation, action, next_observation)` transition must imply one consistent
elapsed time and reset rule.

## Saved artifacts

- Trajectories: `.npy` arrays containing transition sequences from training runs. POMDP trajectories store both the agent-visible states used for replay and the underlying true-state metadata used for analysis.
- Log-prob files: `.npy` arrays containing cumulative replay log-likelihood traces.
- Figures: PNG summaries for overall comparison, pairwise accuracy, log-prob gaps, and per-source behavior. Figure filenames include the environment key so runs from different mazes/observability settings do not collide.

The pipeline is staged: trajectory generation writes `data/trajectories`,
inference reads those trajectories and writes `data/logprobs`, and plotting
reads both to produce `outputs/figures`. Those figures are generated outputs,
not source assets. Generation and inference can parallelize CPU-safe jobs with
`--jobs`, but plotting remains a lightweight serial step. `plot_all_agents
--quiet` suppresses normal stdout output.

## Main pipeline

### 1. Generate trajectories

```bash
uv run python -m forage_rl.experiments.generate_trajectories --envs simple:full full:full full:pomdp --agents all --num-runs 30 --num-episodes 200 --device auto --jobs 8
```

Why this script exists: it trains each registered agent independently and
saves the resulting rollout dataset to `data/trajectories/`. Every later stage
depends on these saved runs.

CPU-heavy example:

```bash
uv run python -m forage_rl.experiments.generate_trajectories --envs full:full full:pomdp --agents mbrl q_learning --num-runs 30 --num-episodes 200 --jobs 8
```

### 2. Run model inference

```bash
uv run python -m forage_rl.experiments.model_inference --envs simple:full full:full full:pomdp --agents all --num-runs 30 --num-episodes 200 --device auto --jobs 8
```

Why this script exists: it replays each saved source trajectory under every
selected evaluator agent and saves cumulative log-likelihood traces to
`data/logprobs/`, making the models directly comparable on the same data.

### 3. Render all-agent plots

```bash
MPLBACKEND=Agg uv run python -m forage_rl.experiments.plot_all_agents --envs simple:full full:full full:pomdp --agents all --num-runs 30
```

Why this script exists: it turns the saved inference outputs into summary
figures in `outputs/figures/`, including overall comparison plots, pairwise
matrices, cumulative accuracy curves, and per-source trajectory statistics.
When only one agent is requested, the pairwise summary bundle is skipped and
only the per-source plots are rendered.

## CLI Reference

### `generate_trajectories`

Train selected registered agents and save rollout datasets to `data/trajectories`.

Invocation prefix:

```bash
uv run python -m forage_rl.experiments.generate_trajectories
```

| Flag | Default | Meaning | Notes |
| --- | --- | --- | --- |
| `--agents` | `all` | Agent name(s) to train | Accepts one or more registered agent names, or `all` for every registered agent. |
| `--envs` | built-in target set | Environment target(s) to run | Accepted values are `simple:full`, `full:full`, and `full:pomdp`. When omitted, the script uses the default built-in target set. |
| `--num-runs` | `100` | Number of independent training runs per agent/environment | Must be a positive integer. |
| `--num-episodes` | `6` | Episodes per training run | Must be a positive integer. |
| `--base-seed` | `0` | Base seed used to derive deterministic per-run seeds | Run `i` uses seed `base_seed + i`. |
| `--device` | `auto` | Torch device for deep agents | Accepted values are `auto`, `cpu`, `cuda`, or `cuda:N`. Ignored by tabular agents. |
| `--jobs` | `1` | Worker processes for CPU-safe jobs | Must be a positive integer. CPU-safe jobs can parallelize; CUDA deep-agent jobs stay sequential. |
| `--quiet` | disabled | Suppress normal stdout output | Useful for batch runs and tests. |

### `model_inference`

Replay saved source trajectories under selected evaluator agents and save cumulative log-likelihood traces to `data/logprobs`.

Invocation prefix:

```bash
uv run python -m forage_rl.experiments.model_inference
```

| Flag | Default | Meaning | Notes |
| --- | --- | --- | --- |
| `--agents` | `all` | Source agent(s) whose saved trajectories will be evaluated | Accepts one or more registered agent names, or `all` for every registered agent. |
| `--envs` | built-in target set | Environment target(s) to run | Accepted values are `simple:full`, `full:full`, and `full:pomdp`. When omitted, the script uses the default built-in target set. |
| `--eval-agents` | `all` | Evaluator agent(s) used to score each source trajectory | Accepts one or more registered agent names, or `all` when omitted. |
| `--num-runs` | `100` | Number of saved runs to evaluate per source agent/environment | Must be a positive integer. The script uses up to that many discovered saved runs. |
| `--num-episodes` | `200` | Training episodes used when replaying evaluator learning dynamics | Must be a positive integer. |
| `--base-seed` | `0` | Base seed used to derive deterministic per-run evaluator seeds | Run `i` uses seed `base_seed + i`. |
| `--device` | `auto` | Torch device for deep evaluator agents | Accepted values are `auto`, `cpu`, `cuda`, or `cuda:N`. Ignored by tabular agents. |
| `--jobs` | `1` | Worker processes for CPU-safe inference jobs | Must be a positive integer. CPU-safe jobs can parallelize; CUDA deep evaluators stay sequential. |
| `--quiet` | disabled | Suppress normal stdout output | Missing source trajectory files are skipped silently when this is enabled. |

### `plot_all_agents`

Render the standard saved-figure bundle from generated trajectories and replay outputs.

Invocation prefix:

```bash
uv run python -m forage_rl.experiments.plot_all_agents
```

| Flag | Default | Meaning | Notes |
| --- | --- | --- | --- |
| `--agents` | `all` | Agent name(s) to include in the plot bundle | Accepts one or more registered agent names, or `all` for every registered agent. |
| `--envs` | built-in target set | Environment target(s) to plot | Accepted values are `simple:full`, `full:full`, and `full:pomdp`. When omitted, the script uses the default built-in target set. |
| `--num-runs` | all available runs | Optional cap on saved runs included per source agent/environment | When omitted, plotting uses all discovered saved runs. If provided, it must be a positive integer. |
| `--show` | disabled | Display figures interactively | By default figures are saved without opening GUI windows. |
| `--quiet` | disabled | Suppress normal stdout output | Useful for non-interactive figure generation. |

## Trajectory schema and regeneration

Replay and inference require terminal-aware timed trajectories. Older 4-column
and 5-column trajectory files are not valid replay input and should be
regenerated with the current pipeline. POMDP rollouts also save true-state
metadata so plots can show hidden-state residency while replay remains
observation-correct.

This repository no longer supports the older deep-agent stream variants or the
removed observed-state flicker schema. If you have existing deep-agent
trajectories, log-probs, or figures from those older variants, regenerate them
with the current `dqn` / `rdqn` pipeline before comparing results.

The environment key is now part of generated artifact names:

- trajectories: `env_<maze>__<observability>__<agent>_trajectories_<run>.npy`
- log-probs: `env_<maze>__<observability>__source_<source>_eval_<eval>_log_likelihoods_<run>.npy`
- figures: `...__<maze>__<observability>.png`
