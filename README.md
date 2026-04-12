# forage-rl

A mouse foraging simulator for studying reinforcement learning agents in patch-foraging environments.

---

## Installation

```bash
pip install -e .
```

This installs the `forage` CLI command alongside the package.

---

## The `forage` CLI

The `forage` command runs the full pipeline for a given source agent: generate training trajectories, evaluate them under one or more evaluator agents, and save comparison plots.

```
forage --source <agent> [--compare-to <agent> ...] [options]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--source` | required | Agent whose trajectories are generated and evaluated |
| `--compare-to` | `all` | One or more evaluator agents, or `all` for every registered agent |
| `--num-runs` | 100 | Number of independent trajectory files to generate |
| `--num-episodes` | 6 | Training episodes per run |
| `--maze` | `simple` | Built-in maze name (see `forage_rl/environments/maze_specs/`) |
| `--pomdp` | false | Use partially observable maze instead of fully observable |
| `--quiet` | false | Suppress per-step output |

Valid agent names are `q_learning` and `mbrl`.

### Examples

Run the full pipeline with `q_learning` as the source agent, evaluated against all registered agents:

```bash
forage --source q_learning
```

Compare `q_learning` against `mbrl` only, on a custom maze with 10 runs:

```bash
forage --source q_learning --compare-to mbrl --num-runs 10 --maze simple
```

Use the partially observable variant:

```bash
forage --source mbrl --compare-to q_learning --pomdp
```

### Pipeline steps

1. **Validate** — checks all agent names; prints valid options and exits on unknown names
2. **Confirm overwrites** — prompts interactively if trajectory files already exist for an agent
3. **Generate trajectories** — trains each agent across `--num-runs` independent runs in parallel; results saved to `data/trajectories/`
4. **Run inference** — evaluates each trajectory under every evaluator agent; log-likelihoods saved to `data/logprobs/`
5. **Plot trajectory stats** — saves per-agent reward and residency figures to `outputs/figures/`
6. **Plot comparison** — saves a win-rate comparison figure to `outputs/figures/`

---

## Defining a maze

Mazes are defined as TOML files and validated against a schema on load.

### Adding a built-in maze

Place a `.toml` file in `forage_rl/environments/maze_specs/`. The filename (without extension) becomes the maze name passed to `--maze`.

```
forage_rl/environments/maze_specs/my_maze.toml  →  --maze my_maze
```

### Loading from an arbitrary file path

To load a maze spec from any path in Python, use `load_maze_spec`:

```python
from forage_rl.environments import load_maze_spec, Maze

spec = load_maze_spec("/path/to/my_maze.toml")
maze = Maze(spec)
```

### TOML format

A maze spec has three sections: `[maze]`, one `[states.<id>]` block per state, and transition probabilities nested under each state.

```toml
[maze]
name        = "my_maze"   # string identifier
horizon     = 100         # max timesteps per episode
initial_state = 0         # state id the agent starts in
action_labels = ["stay", "leave"]  # defines action indices (0, 1, ...)

[states.0]
label             = "Rich Patch"
decay             = 0.2          # reward depletion rate (must be > 0)
observation_group = 0            # groups states that look identical in POMDP mode

[states.0.transitions]
# Each defined action maps to a list of [next_state, probability] outcomes.
# Probabilities for each defined (state, action) pair must sum to 1.0.
stay  = [[0, 1.0]]          # staying keeps the agent in state 0
leave = [[1, 1.0]]          # leaving moves to state 1

[states.1]
label             = "Poor Patch"
decay             = 3.0
observation_group = 1
[states.1.transitions]
stay  = [[1, 1.0]]
leave = [[0, 1.0]]
```

#### Stochastic transitions

List multiple `[next_state, probability]` outcomes for a single action:

```toml
[states.0.transitions]
stay  = [[0, 1.0]]
leave = [[1, 0.4], [2, 0.6]]   # 40% → state 1, 60% → state 2
```

#### Timed transitions (optional)

Add a duration field `[next_state, probability, duration]` to encode travel time. All rows must use this format if any row does:

```toml
[states.0.transitions]
stay  = [[0, 1.0, 1]]
leave = [[1, 1.0, 3]]   # takes 3 timesteps to leave
```

#### Validation rules

- State ids must be contiguous from `0` to `N-1`
- `initial_state` must be a valid state id
- `observation_group` values must be contiguous from `0` to `K-1`
- Every state must define at least one action
- Every defined `(state, action)` pair must have transitions that sum to exactly `1.0`
- No duplicate `(state, action, next_state)` rows
- Transition modes (with/without duration) cannot be mixed within the same file

---

## Creating and registering an agent

### 1. Implement the agent

All agents extend `BaseAgent` from `forage_rl/agents/base.py` and must implement two methods:

```python
from forage_rl.agents.base import BaseAgent
from forage_rl.environments import Maze
from forage_rl import Trajectory


class MyAgent(BaseAgent):
    def __init__(self, maze: Maze, num_episodes: int = 200):
        super().__init__(maze)
        self.num_episodes = num_episodes

    def train(self, verbose: bool = True) -> Trajectory:
        """Train on the maze and return the collected trajectory."""
        transitions = []
        for episode in range(self.num_episodes):
            state, _ = self.maze.reset()
            done = False
            while not done:
                action = ...  # your action selection logic
                transition, done = self.maze.step_transition(action)
                transitions.append(transition)
        return Trajectory(transitions=transitions)

    def simulate(self, trajectory: Trajectory) -> list[float]:
        """Return per-transition log-likelihoods for inference."""
        log_likelihoods = []
        for t in trajectory.transitions:
            log_prob = ...  # log P(action | state) under your model
            log_likelihoods.append(log_prob)
            # optionally update internal state here
        return log_likelihoods
```

`train` is called during trajectory generation. `simulate` is called during inference — it receives a trajectory and should return one log-likelihood per transition, updating the agent's internal state as it steps through the trajectory.

`BaseAgent` provides Boltzmann action selection helpers if needed:

```python
probs  = self.boltzmann_action_probs(q_values)   # softmax over Q-values
action = self.choose_action_boltzmann(q_values)  # sample from softmax
```

### 2. Register the agent

Open `forage_rl/agents/registry.py` and add your agent to the `Agent` enum and `AGENT_REGISTRY`:

```python
from forage_rl.agents.my_agent import MyAgent

class Agent(StrEnum):
    MBRL      = "mbrl"
    QLearning = "q_learning"
    MyAgent   = "my_agent"   # add your entry here

AGENT_REGISTRY: dict[Agent, AgentFactory] = {
    Agent.MBRL:      lambda maze, num_episodes=DefaultParams.NUM_EPISODES: MBRL(...),
    Agent.QLearning: lambda maze, num_episodes=DefaultParams.NUM_EPISODES: QLearningTime(...),
    Agent.MyAgent:   lambda maze, num_episodes=DefaultParams.NUM_EPISODES: MyAgent(
        maze, num_episodes=num_episodes
    ),
}
```

Once registered, `my_agent` is immediately available as a `--source` or `--compare-to` argument in the CLI.

```bash
forage --source my_agent --compare-to q_learning mbrl
```
