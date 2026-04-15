# Trajectory Reward Tables

This summary is computed from the saved POMDP trajectory artifacts in `data/trajectories` using `run_dataset_0` for each agent and maze.

Metric:

- `total_reward`: sum of transition rewards over the saved 1000-step episode
- `reward_per_step`: `total_reward / 1000`

Setup notes:

- trajectories were generated with `--num-runs 1 --num-episodes 1 --pomdp`
- neural agents use `--context-mode prev_reward`
- for the perturbed mazes, `1000` is the built-in horizon, so filenames omit the `_h1000` suffix

## Leaders By Maze

| Maze | Best Agent | Total Reward | Reward Per Step |
| --- | --- | ---: | ---: |
| `full` | `sr_dyna` | 264 | 0.264 |
| `full_one_way` | `sr_dyna` | 217 | 0.217 |
| `full_one_way_perturbed_detour` | `elman` | 202 | 0.202 |
| `full_one_way_perturbed_latent_learning` | `gru` | 41 | 0.041 |
| `full_one_way_perturbed_observability` | `sr_dyna` | 230 | 0.230 |
| `full_one_way_perturbed_revaluation` | `sr_dyna` | 311 | 0.311 |

## Cross-Maze Reward Matrix

| Agent | `full` | `full_one_way` | `detour` | `latent_learning` | `observability` | `revaluation` | Avg Reward | Avg Rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mbrl` | 195 | 134 | 170 | 16 | 134 | 188 | 139.50 | 7.83 |
| `q_learning` | 188 | 184 | 154 | 17 | 184 | 164 | 148.50 | 6.50 |
| `sr_td` | 237 | 174 | 197 | 16 | 174 | 285 | 180.50 | 4.67 |
| `sr_mb` | 232 | 167 | 188 | 16 | 167 | 300 | 178.33 | 6.17 |
| `sr_dyna` | 264 | 217 | 193 | 29 | 230 | 311 | 207.33 | 2.00 |
| `dqn` | 227 | 193 | 185 | 37 | 185 | 220 | 174.50 | 4.50 |
| `elman` | 169 | 174 | 202 | 25 | 174 | 253 | 166.17 | 5.50 |
| `gru` | 173 | 182 | 198 | 41 | 182 | 247 | 170.50 | 4.33 |
| `lstm` | 234 | 201 | 193 | 39 | 201 | 204 | 178.67 | 3.50 |

Column names:

- `detour` = `full_one_way_perturbed_detour`
- `latent_learning` = `full_one_way_perturbed_latent_learning`
- `observability` = `full_one_way_perturbed_observability`
- `revaluation` = `full_one_way_perturbed_revaluation`

## Per-Maze Rankings

### `full`

| Rank | Agent | Total Reward | Reward Per Step |
| ---: | --- | ---: | ---: |
| 1 | `sr_dyna` | 264 | 0.264 |
| 2 | `sr_td` | 237 | 0.237 |
| 3 | `lstm` | 234 | 0.234 |
| 4 | `sr_mb` | 232 | 0.232 |
| 5 | `dqn` | 227 | 0.227 |
| 6 | `mbrl` | 195 | 0.195 |
| 7 | `q_learning` | 188 | 0.188 |
| 8 | `gru` | 173 | 0.173 |
| 9 | `elman` | 169 | 0.169 |

### `full_one_way`

| Rank | Agent | Total Reward | Reward Per Step |
| ---: | --- | ---: | ---: |
| 1 | `sr_dyna` | 217 | 0.217 |
| 2 | `lstm` | 201 | 0.201 |
| 3 | `dqn` | 193 | 0.193 |
| 4 | `q_learning` | 184 | 0.184 |
| 5 | `gru` | 182 | 0.182 |
| 6 | `sr_td` | 174 | 0.174 |
| 7 | `elman` | 174 | 0.174 |
| 8 | `sr_mb` | 167 | 0.167 |
| 9 | `mbrl` | 134 | 0.134 |

### `full_one_way_perturbed_detour`

| Rank | Agent | Total Reward | Reward Per Step |
| ---: | --- | ---: | ---: |
| 1 | `elman` | 202 | 0.202 |
| 2 | `gru` | 198 | 0.198 |
| 3 | `sr_td` | 197 | 0.197 |
| 4 | `sr_dyna` | 193 | 0.193 |
| 5 | `lstm` | 193 | 0.193 |
| 6 | `sr_mb` | 188 | 0.188 |
| 7 | `dqn` | 185 | 0.185 |
| 8 | `mbrl` | 170 | 0.170 |
| 9 | `q_learning` | 154 | 0.154 |

### `full_one_way_perturbed_latent_learning`

| Rank | Agent | Total Reward | Reward Per Step |
| ---: | --- | ---: | ---: |
| 1 | `gru` | 41 | 0.041 |
| 2 | `lstm` | 39 | 0.039 |
| 3 | `dqn` | 37 | 0.037 |
| 4 | `sr_dyna` | 29 | 0.029 |
| 5 | `elman` | 25 | 0.025 |
| 6 | `q_learning` | 17 | 0.017 |
| 7 | `mbrl` | 16 | 0.016 |
| 8 | `sr_td` | 16 | 0.016 |
| 9 | `sr_mb` | 16 | 0.016 |

### `full_one_way_perturbed_observability`

| Rank | Agent | Total Reward | Reward Per Step |
| ---: | --- | ---: | ---: |
| 1 | `sr_dyna` | 230 | 0.230 |
| 2 | `lstm` | 201 | 0.201 |
| 3 | `dqn` | 185 | 0.185 |
| 4 | `q_learning` | 184 | 0.184 |
| 5 | `gru` | 182 | 0.182 |
| 6 | `sr_td` | 174 | 0.174 |
| 7 | `elman` | 174 | 0.174 |
| 8 | `sr_mb` | 167 | 0.167 |
| 9 | `mbrl` | 134 | 0.134 |

### `full_one_way_perturbed_revaluation`

| Rank | Agent | Total Reward | Reward Per Step |
| ---: | --- | ---: | ---: |
| 1 | `sr_dyna` | 311 | 0.311 |
| 2 | `sr_mb` | 300 | 0.300 |
| 3 | `sr_td` | 285 | 0.285 |
| 4 | `elman` | 253 | 0.253 |
| 5 | `gru` | 247 | 0.247 |
| 6 | `dqn` | 220 | 0.220 |
| 7 | `lstm` | 204 | 0.204 |
| 8 | `mbrl` | 188 | 0.188 |
| 9 | `q_learning` | 164 | 0.164 |
