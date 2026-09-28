# Project Log

Decisions, their rationale, and every use of the evaluation window. The experiments themselves are documented in [EXPERIMENTS.md](EXPERIMENTS.md).

## Result

| | Validation (2026-06-17 to 08-17) | Eval month (2026-08-17 to 09-17) |
|---|---|---|
| Baseline DQN (3 seeds) | -5.83% ± 4.33% | +22.61% (official seed 0) |
| Final agent (EXP-12, ensemble of 20 DQN networks) | +11.28% | -2.20% (market +21.12%) |

The validation figure is the best configuration found on a block that was used for every selection decision, so it is optimistic. Selection on that single sideways block chose a regime-specific mean-reversion strategy; in the trending eval month the multi-hour signal was not detectable, the mostly flat policy trailed the market structurally, and the tails (a 2x position before a sharp drop, shorts against the rally) pushed it below zero. Main improvements for future work: walk-forward validation across several market regimes, and no leverage while the edge is about the size of the fee.

## Timeline

| Date | Phase | Milestone |
|---|---|---|
| 2026-09-27 | 0 | Course material and project brief reviewed; Python 3.12 environment with pinned versions; repository structure |
| 2026-09-27 | 1 | Data pipeline (`rl_trading/data.py`), environment mechanics verified experimentally (`rl_trading/envs.py`), benchmarks and validation protocol (`rl_trading/evaluate.py`), notebook 01 |
| 2026-09-27 | 2 | Algorithm choice (DQN with target network and experience replay); network, replay buffer, agent and training loop; unit checks |
| 2026-09-28 | 2 | Baseline trained with 3 seeds; one-time baseline evaluation on the eval window |
| 2026-09-28 | 3 | Baseline diagnostics: overestimation, memorisation, action gap, trading behaviour; ranked hypotheses; notebook 02 |
| 2026-09-28 | 4 | EXP-01 to EXP-14 (see EXPERIMENTS.md), including two pre-registered replication tests; notebook 03 |
| 2026-09-28 | 5 | Final ensemble frozen (`models/final/`), one evaluation on the eval window, post hoc analysis; notebook 04 |

## Decisions and rationale

| Decision | Rationale |
|---|---|
| Python 3.12, pinned package versions (including torch 2.14.0) | Required by the course; reproducibility |
| Validation block 2026-06-17 to 2026-08-17, carved out of the training year | The eval month is a test set: all selection happens on validation, the eval window is used once for the baseline and once for the final model |
| Algorithm: DQN with target network and experience replay (course lab TD3) | Small discrete action space, continuous state, scarce and noisy data: off-policy replay reuses every transition; REINFORCE, SARSA, A2C and PPO are on-policy |
| Replay buffer as a numpy ring buffer that copies on insert | The environment returns observations as views into its memory and rewrites them when an hour is revisited; copying makes corrupted transitions impossible |
| Agent randomness from its own seeded generators; `validate()` restores the global random state | The environment draws initial positions and start hours from the global `np.random`; this keeps environment randomness identical across experiments (paired comparisons) |
| Baseline: course hyperparameters, ε decay 0.912 per episode | Same schedule shape as the course lab (minimum reached after half of training), adapted to episodes of 7,295 steps |
| Baseline reported with its last model, not the best validation checkpoint | Validation returns fluctuated by about 10 points between episodes; the best of 100 noisy evaluations would be selection on noise |
| Baseline eval run with all 3 seeds, seed 0 fixed in advance as the official baseline CSV; market return not inspected | A distribution instead of a single draw as reference, without selecting a seed on the test set |
| Success bars on validation: positive after fees, above buy and hold, above always short (or equal with a policy that reacts to the market) | Always short wins both the training year and validation by betting on the regime, so beating buy and hold alone is not evidence of skill |
| Experiment order driven by diagnostics and revised with evidence | After EXP-01, overfitting was identified as the main bottleneck (data before reward scaling); after the feature analysis, the horizon γ was tested before new features |
| Training range 2023-04-01 to 2026-06-17 (gap-free, including the 7-day warm-up) | The project brief allows changing the training range as long as it ends before the evaluation window. This range removed memorisation in EXP-03 with recent market regimes only; missing exchange hours are never filled with invented candles |
| Reward changes only in the training environment | Validation and evaluation keep the default log-return reward, so `total_reward` in the CSV stays comparable |
| Methods beyond the course slides (here: ensembles of Q-networks) are kept only if they measurably improve validation results | Keeps the solution close to the course and every component justified by evidence |
| γ = 0.9 | Overestimation fell from 0.035 to 0.003 (the max bias is amplified by about 1 / (1 - γ)); γ = 0.5 was too short-sighted to pay back fees |
| Ensemble of Q-networks | Single networks with informative features memorise seed-specific noise; averaging their Q-values shrinks uncertain exposed actions towards flat and improved every configuration, confirmed on fresh seeds with criteria committed before training (EXP-13, EXP-14). Related methods: Averaged-DQN (Anschel et al., 2017), Bootstrapped DQN (Osband et al., 2016) |
| Final configuration EXP-12 instead of EXP-09 | The EXP-14 decision rule favoured EXP-09 on one 6-network comparison inside the noise (-2.17% vs -2.81%). The broader ensemble size analysis (20 subsets per size) favoured EXP-12 at every size of 3 or more, with equal results in both validation months and fewer trades. Not driven by the eval window: the EXP-12 ensemble is about 94% flat |
| 20 networks, seeds 0 to 19, fixed before seeds 12 to 19 were trained | The validation return still improved at 12 networks; fixing the composition in advance avoids selecting seeds |
| Submit the validated models trained up to 2026-06-17 (no retraining on the validation months) | The submitted ensemble is exactly the one that was validated |
| Submit the frozen final model, not a model chosen after the eval run | Choosing the submission by its test result would be data snooping; the result is reported as it is |

## Runs on the evaluation window (2026-08-17 to 2026-09-17)

| Date | Model | Purpose | Result |
|---|---|---|---|
| 2026-09-28 | Baseline EXP-00, last model, seeds 0, 1, 2 (seed 0 fixed in advance as official) | Baseline reference, not used for any selection | `eval/baseline_evaluation_results.csv` +22.61%; seeds 1 and 2: +23.52%, +13.20% |
| 2026-09-28 | Final model: EXP-12, ensemble of 20 networks, frozen in `models/final/` | Final submission | `eval/evaluation_results.csv` -2.20% (market +21.12%) |
| 2026-09-28 | Benchmarks and main intermediate models | Post hoc analysis after the submission was fixed, not used for selection | `eval/posthoc_eval_comparison.csv` |

**Known exposure:** the market return was not inspected after the baseline run, but the baseline's long-leaning models making +13% to +24% implied that the eval month had risen. Decisions on the action set, leverage and long/short tendency were therefore justified with validation evidence only; the final model is mostly flat.

## Engineering notes (issues found and fixed)

| Issue | Effect | Fix |
|---|---|---|
| Observations are numpy views into the environment's memory | Stored transitions silently change their position features when an hour is revisited | Replay buffer copies on insert (verified with a test against the real environment) |
| Importing the environment turns every warning into an exception | Harmless library warnings crash the program | `make_env` restores the default warning filter |
| `env.reset(seed=...)` does not control initial positions | Unseeded runs | Seeding through `np.random`; agent uses its own generators |
| `validate()` reseeded the global random state during training | Every training episode after a validation would start from the same initial position | Random state saved and restored |
| Parallel runs rewrote the shared data cache | Two runs read a half-written file and crashed (rerun, results unaffected) | Cache written only after a download, atomically |
