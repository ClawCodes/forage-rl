# Analysis README

This package contains the patch-timing and perturbation-recovery analysis code for `forage_rl`.

The package now has two related but distinct jobs:

1. Stationary patch-timing diagnostics for already-saved trajectories.
2. Post-perturbation recovery analysis against a patch-leaving benchmark.

The core quantity throughout the package is patch residence time, abbreviated `PRT`.

## Core Definitions

### Transition Time Variables

For any saved decision inside a patch:

- `time_spent` is zero-based.
- `time_spent = 0` means the agent is making its first decision after entering the patch.
- `dwell` or `PRT` is one-based.

The conversion is:

```text
PRT = time_spent + 1
```

So:

- leave at `time_spent = 0` means `PRT = 1`
- leave at `time_spent = 4` means `PRT = 5`

### Episode Recovery Metrics

For each post-perturbation episode `t`, the recovery analysis extracts all realized leave events and compares each one with a benchmark leave threshold `PRT*(s)` for the relevant state `s`.

For a realized leave event:

```text
actual_dwell = time_spent + 1
signed_deviation = actual_dwell - PRT*(s)
absolute_deviation = |actual_dwell - PRT*(s)|
```

Episode-level summaries are:

```text
delta_abs(t) = mean over leave events in episode t of absolute_deviation
delta_signed(t) = mean over leave events in episode t of signed_deviation
```

If an episode has no usable leave events, the analysis records `NaN`.

The recovery AUC over a fixed post-perturbation window `T` is the raw discrete sum:

```text
AUC = sum_{t=1}^T delta_abs(t)
```

Only finite episode values are included in the sum.

## Benchmark Families

The package intentionally supports two benchmark families.

### 1. True MVT

This is used only where the benchmark really is an analytic marginal value theorem calculation.

Supported mazes:

- `simple`
- `simple_one_way`

For a patch with expected stay rewards `r_s(0), r_s(1), ...`, the total expected gain from exploiting `k` times is:

```text
G_s(k) = sum_{i=0}^{k-1} r_s(i)
```

For `simple`, one cycle is:

- exploit upper patch `k_upper` times, then leave
- exploit lower patch `k_lower` times, then leave

The benchmark maximizes:

```text
(G_upper(k_upper) + G_lower(k_lower)) / (k_upper + 1 + k_lower + 1)
```

The `+1` terms are the leave actions.

For `simple_one_way`, the same idea is used but corridor travel is added explicitly:

```text
(G_upper(k_upper) + G_lower(k_lower)) /
(k_upper + k_lower + tau_upper_to_lower + tau_lower_to_upper)
```

where `tau_*` are the deterministic leave-path lengths through the corridor states.

After choosing optimal exploit counts `k*`, the reported benchmark PRT is:

```text
PRT* = k* + 1
```

### 2. FO Oracle

This is used for:

- `full`
- `full_one_way`
- PO comparisons against those mazes

This is not a closed-form MVT benchmark. It is the earliest leave time implied by the optimal fully observed MDP policy.

Let `pi*(s, t)` be the optimal policy for true state `s` at local patch time `t = time_spent`.

Then:

```text
PRT*(s) = min { t + 1 : pi*(s, t) = leave }
```

If the optimal policy never leaves before the horizon, the code uses the horizon as the fallback dwell threshold.

For PO trajectories, the comparison is still against the FO ideal benchmark. The code first infers the hidden true state and then compares the realized dwell against the FO benchmark for that inferred state.

## File-by-File Guide

### `mvt.py`

Purpose:

- exact true-MVT benchmarks for simple-style mazes

Main functions:

- `reward_schedule_for_state(...)`
- `simple_true_mvt_optimal_exploit_steps(...)`
- `simple_true_mvt_optimal_prt(...)`
- `simple_one_way_true_mvt_optimal_prt(...)`

Key math:

- `reward_schedule_for_state` builds the expected reward sequence for repeated stay actions in one patch.
- `simple_true_mvt_optimal_exploit_steps` performs an integer scan over all feasible exploit counts.
- `simple_one_way_true_mvt_optimal_prt` performs the same integer scan but includes deterministic corridor travel in the denominator.

Important limitation:

- These helpers are intentionally narrow. They are exact only for the simple-family mazes that match the required structure.

### `oracle_patch_benchmark.py`

Purpose:

- compute FO oracle patch-leaving thresholds directly from the optimal MDP policy

Main function:

- `oracle_patch_optimal_prt_by_state(...)`

Key math:

- Solve the fully observed MDP with value iteration.
- For each patch state, find the first local time index where the optimal action is `leave`.
- Convert that zero-based time to one-based `PRT`.

Important detail:

- Corridor-only transit states are skipped. The returned map is only for patch states that support both `stay` and `leave`.

### `benchmark_resolver.py`

Purpose:

- choose the honest benchmark type for a maze
- rebuild a perturbed benchmark maze from manifest metadata
- return benchmark `PRT*` values in a uniform interface

Main functions:

- `resolve_patch_benchmark_kind(...)`
- `build_patch_benchmark_maze(...)`
- `resolve_patch_benchmark_prt(...)`

Behavior:

- `simple` and `simple_one_way` resolve to `true_mvt`
- `full` and `full_one_way` resolve to `fo_oracle`
- PO still resolves to the corresponding FO benchmark family

Benchmark reconstruction:

- by default, use the built-in maze spec
- optionally override state parameters or transitions from `benchmark_params`
- this is how external perturbation metadata can describe a post-perturbation maze without generating perturbations in this repo

### `perturbation_inputs.py`

Purpose:

- load externally produced combined trajectories
- reconstruct episodes from one transition stream
- determine which episodes count as post-perturbation

Main pieces:

- `CombinedPerturbationRun`
- `load_combined_trajectory(...)`
- `episode_trajectories_from_combined_stream(...)`
- `split_post_perturbation_episodes(...)`
- `load_combined_perturbation_runs(...)`

Boundary rule:

Let `p` be the externally supplied `perturbation_timestep`, measured in global transition coordinates.

- transitions with index `< p` are pre-perturbation
- transitions with index `>= p` are post-perturbation

Episode handling:

- reconstruct episode boundaries from `episode_lengths`
- the first episode with `start_index >= p` is treated as fully post-perturbation
- if `p` falls inside an episode, that mixed episode is dropped by default

This is a deliberate design choice to avoid contaminating one recovery point with both pre- and post-perturbation dynamics.

### `recovery.py`

Purpose:

- compute episode-wise absolute and signed recovery curves
- compute raw discrete recovery AUC

Main functions:

- `episode_prt_deviation_from_benchmark(...)`
- `recovery_curve_for_episode_sequence(...)`
- `signed_recovery_curve_for_episode_sequence(...)`
- `recovery_curve_for_run(...)`
- `signed_recovery_curve_for_run(...)`
- `recovery_auc(...)`

Implementation detail:

- the module extracts decision rows from trajectories using `patch_timing.extract_decision_rows(...)`
- it compares only leave events whose resolved state appears in the benchmark map

In one-way mazes, that means corridor transit leaves are ignored because they are not patch-leaving decisions.

### `patch_timing.py`

Purpose:

- stationary oracle patch-timing diagnostics
- hidden-state inference utilities
- generic leave-event extraction helpers

It is not the external perturbation analysis entrypoint.

Important components:

- `DecisionRow`
- `extract_decision_rows(...)`
- `infer_hidden_states_for_trajectory(...)`
- `oracle_optimal_dwell_by_state(...)`
- `oracle_residency_deviation_by_patch(...)`
- `aggregate_curves(...)`

Hidden-state inference:

- For PO trajectories, the code tracks a posterior over true states inside each observation group.
- Stay rewards contribute Bernoulli likelihood terms.
- Leave transitions propagate probability mass to the next observation group.
- The inferred state for a visit is the maximum-posterior state after updating on the observed rewards.

This produces the true-state labels needed for PO-vs-FO-ideal recovery comparisons.

### `__init__.py`

Purpose:

- re-export the public analysis API

This is the user-facing surface for the package-level imports.

## End-to-End External Perturbation Flow

The external perturbation workflow is:

1. Load a manifest entry as `CombinedPerturbationRun`.
2. Load one combined transition stream from disk.
3. Reconstruct episode boundaries from `episode_lengths`.
4. Drop the mixed boundary episode if the perturbation happens mid-episode.
5. Rebuild the perturbed benchmark maze from manifest metadata.
6. Resolve the benchmark type:
   - true MVT for `simple` or `simple_one_way`
   - FO oracle for `full` or `full_one_way`
7. If the trajectories are PO, infer hidden true states.
8. Compute episode-wise absolute and signed deviation curves.
9. Compute raw discrete AUC over the requested recovery window.

## Interpretation Rules

The code is designed to preserve the interpretation distinction:

- `simple` and `simple_one_way`: true MVT benchmark
- `full` and `full_one_way`: FO oracle benchmark
- PO: compared against FO ideal benchmark

This matters because the same PRT/AUC machinery can be applied across all perturbation families, but the benchmark meaning is not identical across all mazes.

For detour, latent learning, and policy revaluation, this should be treated as exploratory patch-timing analysis rather than the only normative task metric.

## Current Caveats

- Recovery currently ignores leave events whose resolved state is not present in the benchmark map. That is correct for corridor transit states, but it also means incomplete benchmark coverage can be masked if the caller constructs a bad benchmark map.
- The FO oracle benchmark depends on the built-in MDP solver and therefore inherits its modeling assumptions.
- The external manifest path focuses on `episode_lengths`; it does not yet expose alternative boundary encodings such as `episode_start_indices`.
