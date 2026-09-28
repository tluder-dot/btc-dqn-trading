# EXPERIMENTS

Protocol: one change at a time against the current best config, at least 3 seeds, same training budget, selection on the validation split (2026-06-17 to 2026-08-17) only. Metrics are reported as mean ± std over seeds. A difference smaller than the seed spread is treated as noise.

## Summary

| ID | Change vs parent | Parent | Seeds | Val portfolio return | Val excess vs B&H | Turnover | Verdict |
|---|---|---|---|---|---|---|---|
| EXP-00 baseline | TD3 DQN, TD3 hyperparameters, ε decay 0.912/episode, 100 × full dev-train | none | 0, 1, 2 | -5.83% ± 4.33% | -1.56% | 202 | reference |
| EXP-01 normalize | z-score input normalisation (fitted on dev-train) | EXP-00 | 0, 1, 2 | -23.58% ± 11.03% | -19.31% | 987 | rejected (for now) |
| EXP-02 random starts | 720-step random-start episodes, same number of updates | EXP-00 | 0, 1, 2 | -3.71% ± 4.54% | +0.56% | 105 | inconclusive, kept as structure |
| EXP-03 3 years | dev-train from 2023-04-01 (28,152 h) | EXP-02 | 0, 1, 2 | -7.64% ± 8.44% | -3.38% | 108 | inconclusive (less memorisation, no val gain) |
| EXP-04 all history | dev-train = 17 gap-free segments since 2017 (70,202 h) | EXP-02 | 0, 1, 2 | -9.15% ± 13.21% | -4.88% | 908 | inconclusive (no memorisation, no val gain) |
| EXP-05 γ 0.9 | discount 0.99 → 0.9 | EXP-03 | 0, 1, 2 | -3.97% ± 3.95% | +0.30% | 83 | kept (overestimation /12, seeds agree) |
| EXP-06 γ 0.5 | discount 0.99 → 0.5 | EXP-03 | 0, 1, 2 | -10.46% ± 7.14% | -6.19% | 453 | rejected |
| EXP-07 Double DQN | Double DQN targets | EXP-05 | 0, 1, 2 | -6.25% ± 11.47% | -1.98% | 65 | rejected (bias already small) |
| EXP-08 normalize 3y | z-score normalisation | EXP-05 | 0, 1, 2 | -13.28% ± 9.11% | -9.01% | 493 | rejected (memorises) |
| EXP-09 MR features | mean-reversion features + normalisation | EXP-05 | 0, 1, 2 | -8.32% ± 19.84% | -4.05% | 926 | rejected (sees signal, memorises) |
| EXP-10 MR + all history | EXP-09 on all history since 2017 | EXP-09 | 0, 1, 2 | -14.25% ± 7.63% | -9.98% | 262 | rejected |
| EXP-11 MR + small net | EXP-09 with hidden size 32 | EXP-09 | 0, 1, 2 | -18.75% ± 7.90% | -14.48% | 682 | rejected |
| EXP-12 MR + turnover penalty | EXP-09 + training penalty 0.0005 per unit change | EXP-09 | 0, 1, 2 | -10.57% ± 17.25% | -6.30% | 739 | rejected as single model |
| EXP-13 ensemble replication | ensemble of fresh seeds 3-5 vs single seeds 3-5 (EXP-05, 09, 12) | - | 3, 4, 5 | see entry | | | replicated (2 of 3, pre-registered) |
| EXP-14 6-seed ensemble | ensemble of fresh seeds 6-11 (EXP-09, EXP-12) | - | 6-11 | EXP-09 -2.17%, EXP-12 -2.81% | +2.1% / +1.5% | 414 / 288 | criterion 1 met, criterion 2 not met |
| Ensemble size (exploratory) | k = 1 to 12 networks averaged, seeds 0-11 | - | 0-11 | k=12: EXP-09 +5.00%, EXP-12 +6.10% | +9.3% / +10.4% | 205 / 126 | monotone improvement with k |

### Benchmarks (Phase 1.3)

Default positions `[-1, 0, 0.5, 1, 2]`, course fees, 10 episodes per strategy with random initial positions (`np.random.seed(0)`), mean ± std. Figure: `figures/phase1_benchmarks.png`.

| Strategy | Dev-train return (7,295 h) | Val return (1,463 h) | Val Sharpe (ann.) | Val max DD | Val position changes | Val fees |
|---|---|---|---|---|---|---|
| Random | -66.09% ± 15.05% | -18.21% ± 9.06% | -2.99 | -23.7% | 1,174 | 16.6% |
| Always flat (0) | -0.01% | -0.01% | n/a (no exposure) | 0.0% | ≤ 1 | 0.01% |
| Always 0.5 | -22.00% | -2.14% | -0.74 | -5.9% | ≤ 1 | 0.01% |
| Buy and hold (1) | -43.99% | -4.27% | -0.73 | -11.8% | ≤ 1 | 0.01% |
| Always 2x (2) | -87.98% | -8.54% | -0.70 | -23.5% | ≤ 1 | 0.01% |
| Always short (-1) | **+43.97%** | **+4.24%** | +0.76 | -11.7% | ≤ 1 | 0.02% |

Success bars for the agent on validation (proposed in 1.3):
1. Beat flat and random: positive return after fees.
2. Beat buy and hold: positive excess return (this is what the CSV reports as `excess_return`).
3. Beat always short (+4.24%), the "lazy" policy a bear-market training set invites, **or** reach a similar return with a policy whose position shares show that it actually reacts to the market.

---

## Entry template

### EXP-XX: <short name>

- **Hypothesis:** I observe X, I believe the cause is Y, I test Z, I expect W.
- **Change:** exact change.
- **Config:** `configs/<file>.json` (diff vs parent: ...)
- **Seeds:** 0, 1, 2
- **Results (validation, mean ± std):** portfolio return, excess over B&H, Sharpe, max drawdown, position changes, time share per position.
- **Verdict:** kept / rejected / inconclusive
- **Why:** explanation.

---

### EXP-00: Baseline (TD3 DQN)

- **Hypothesis:** the TD3 DQN (target network + replay) with TD3 hyperparameters learns something from the default features; reference point for all experiments.
- **Change:** none (reference). Only adaptation vs TD3: ε decay 0.912 per episode so ε reaches 0.01 after 50 of 100 episodes (TD3 shape).
- **Config:** `configs/baseline.json` (lr 1e-4, γ 0.99, batch 64, buffer 50k, target sync 250, hidden 128, 5 default features, positions [-1, 0, 0.5, 1, 2], default reward, 100 episodes of 7,295 steps).
- **Seeds:** 0, 1, 2 (7 min per seed, run in parallel).
- **Results (last model, validation, 10 episodes per env):**

| Env | Seed 0 | Seed 1 | Seed 2 | Mean ± std | Buy and hold | Always short |
|---|---|---|---|---|---|---|
| val (1,463 h) | -9.90% | -1.27% | -6.33% | **-5.83% ± 4.33%** | -4.27% | +4.24% |
| val_w1 (743 h) | -11.28% | +1.55% | -10.13% | -6.62% ± 7.10% | -2.70% | +2.67% |
| val_w2 (743 h) | -1.17% | -3.83% | +4.30% | -0.23% ± 4.14% | -1.55% | +1.53% |

  Other val metrics (mean over seeds): Sharpe -1.14, max drawdown -11.7%, 202 position changes, fees about 2.3%. Position shares differ completely between seeds: seed 0 is 70% flat and 25% 2x, seed 1 is 84% at 0.5, seed 2 is 90% long. Short is used only about 2% of the time.
- **Training behaviour:** training return at ε = 0.01 (last 10 episodes) +25%, +80%, +151% per seed with about 1,300 to 1,500 position changes per episode, while validation stays around buy and hold. Validation return of consecutive episodes (last 50, ε = 0.01) fluctuates with a std of 8 to 11 percentage points. Mean max-Q on the probe states falls from about 0.2 to about 0.035 and stays there. Loss about 3e-5, no trend.
- **Verdict:** reference. Fails success bar 1 (not positive after fees) and bar 2 (mean below buy and hold, although within the seed spread). Clearly better than random (-18%).
- **Why (first reading, to be tested in Phase 3):** large train/validation gap (memorising the training hours), unstable greedy policy (8 to 11 pp swings between episodes), different seeds end in unrelated near-constant policies. Details in Phase 3.
- **Eval window (one-time baseline reference, 2026-09-28, not used for selection):** seed 0 (official CSV) +22.61%, seed 1 +23.52%, seed 2 +13.20%; mean +19.78% ± 5.71%. Much higher than on validation because the long-leaning policies met a (by inference) rising eval month; this is market direction, not skill.
- **Note on best checkpoints:** the best in-training validation checkpoints reach +17%, +36%, +35%, but they were selected on the same validation data they are reported on (5 noisy evaluations per episode), so these numbers are biased upwards and are not used.

---

### EXP-01: Input normalisation

- **Hypothesis (Phase 3, #1):** I observe Q-values almost constant across states and an action gap of 2.4e-4; I believe the cause is signal scale (inputs about 1.000 ± 0.004); I test z-score normalisation of the market features (statistics fitted on dev-train only); I expect larger action gaps, a more stable policy and better validation results.
- **Change:** `normalize_features: true` (5 market features; position features unchanged).
- **Config:** `configs/exp01_normalize.json` (diff vs baseline: normalize_features).
- **Seeds:** 0, 1, 2.
- **Results (last model):**

| Metric | EXP-00 baseline | EXP-01 |
|---|---|---|
| Val return (1,463 h) | -5.83% ± 4.33% | **-23.58% ± 11.03%** (seeds -12.6%, -23.4%, -34.7%) |
| val_w1 / val_w2 | -6.62% / -0.23% | -13.34% ± 10.39% / -11.44% ± 3.35% |
| Greedy return on training hours | +32% | x42,829 (about +4.3 million %) |
| Corr(position, next-hour return) train / val | +0.050 / +0.002 | +0.291 / -0.013 |
| Median action gap (val) | 2.4e-4 | 2.8e-3 |
| Best vs last checkpoint agreement (val hours) | 2% | 24% |
| Position changes on val / fees | 201 / 2.3% | 987 / 16.4% |
| Overestimation Q - G (val) | 0.043 | 0.256 |

  Position shares on val are now spread over all actions (short 28 to 40%, 2x 16 to 30%).
- **Verdict:** rejected as a standalone change (validation 18 points worse, far beyond the seed spread).
- **Why:** the mechanism worked as hypothesised: the network now distinguishes states (action gap x12, checkpoints agree 24% instead of 2%). But the extra resolution was used to memorise the training hours almost perfectly (x42,829 in-sample, correlation +0.29 in-sample, -0.01 on validation) and to trade about 1,000 times on validation, with 16% fees. **Signal scale was not the main bottleneck; overfitting is.** Normalisation is kept in mind for later: once there is enough data (Phase 3 hypothesis #2) and more features with different scales, the extra resolution may become useful. To be retested in the combination check.

---

### EXP-02, EXP-03, EXP-04: training distribution and amount of data

- **Hypothesis (Phase 3, #2):** I observe train +32% vs validation -6% and a position/next-hour correlation of +0.05 on training hours vs 0 on validation; I believe the cause is too little data seen too often (100 passes over one bear-market year); I test (EXP-02) random-start episodes of 720 steps as a control, (EXP-03) 3.2 gap-free years, (EXP-04) all Binance history since 2017 in 17 gap-free segments, always with the same number of updates (about 729k); I expect a smaller train/validation gap and a positive correlation on validation.
- **Changes:** EXP-02 `max_episode_duration` 721, 1,013 episodes, ε decay 0.9909 per episode (same shape as the baseline), validation every 10 episodes. EXP-03 additionally `train_start` 2023-04-01 (dev-train +131% buy and hold). EXP-04 instead `train_start` 2017-08-17 with `train_segments` (features computed per gap-free segment, segment chosen per episode in proportion to its start hours).
- **Configs:** `configs/exp02_random_starts.json`, `configs/exp03_3years.json`, `configs/exp04_all_history.json`.
- **Seeds:** 0, 1, 2 each.
- **Results (last model; train metrics on the longest training frame):**

| Metric | EXP-00 | EXP-02 | EXP-03 | EXP-04 |
|---|---|---|---|---|
| Training hours | 7,296 | 7,296 | 28,152 | 70,202 |
| Passes over the training hours | 100 | 100 | 26 | 10 |
| Val return | -5.83% ± 4.33% | -3.71% ± 4.54% | -7.64% ± 8.44% | -9.15% ± 13.21% |
| Val per seed | -9.9 / -1.3 / -6.3 | -3.8 / +0.9 / -8.2 | -15.1 / -9.3 / +1.5 | -20.1 / -12.9 / +5.5 |
| val_w1 / val_w2 (mean) | -6.6 / -0.2 | -3.8 / -0.2 | -5.7 / -2.2 | -2.9 / -6.7 |
| Corr(position, next hour) train | +0.050 | +0.046 | +0.015 | +0.010 |
| Corr(position, next hour) val | +0.002 | -0.004 | -0.018 | -0.008 |
| Position changes on val / fees | 201 / 2.3% | 105 / 1.7% | 108 / 1.6% | 908 / 7.5% |
| Best vs last checkpoint agreement | 2% | 5% | 47% | 28% |
| Overestimation Q - G (val) | 0.043 | 0.047 | 0.035 | 0.042 |
| Favourite position on val per seed | flat / 0.5 / long | 0.5 / short / long | 2x / 0.5 / short | long-2x flip / flat-0.5 flip / short |

- **Verdicts:** EXP-02 inconclusive (+2.1 points vs baseline, below the seed spread), kept as the episode structure because longer histories need it and it halves the trades. EXP-03 and EXP-04 inconclusive on validation (means below the baseline, spreads larger than the differences).
- **Why:** more data removes memorisation, as hypothesised: the in-sample correlation falls from 0.050 to 0.015 and 0.010, and in-sample returns collapse (EXP-04 loses 56% on its longest training segment). But no out-of-sample signal appears: without memorisation the agent falls back to a near-constant position whose direction differs by seed (in EXP-03 seed 2 is 94% short although its training data rose +131%). The validation return then reflects which constant bet a seed happened to pick. Q-values of all actions still differ by only about 4e-4, far below the noise of hourly returns, so even the average drift of the training data is not learned. **Conclusion: data amount was a real cause of memorisation, but the default features give the agent nothing to learn beyond a bet on the direction.**

---

### EXP-05 and EXP-06: discount factor γ

- **Hypothesis (Phase 3, #4, moved forward after the feature analysis):** I observe an overestimation of about 0.035 to 0.047 in every run so far and a one-hour mean-reversion edge (about 2e-4 in Q) that no agent learned; I believe the cause is the long horizon 1 / (1 - γ) = 100 hours, over which the max bias accumulates and only noise remains, while the signal lasts 1 to 4 hours; I test γ = 0.9 (horizon about 10 h) and γ = 0.5 (about 2 h); I expect much less overestimation and, if the edge becomes visible, more consistent policies.
- **Change:** `gamma` 0.9 (EXP-05) and 0.5 (EXP-06) instead of 0.99.
- **Configs:** `configs/exp05_gamma09.json`, `configs/exp06_gamma05.json` (parent EXP-03, 3 years).
- **Seeds:** 0, 1, 2 each.
- **Results (last model):**

| Metric | EXP-03 (γ 0.99) | EXP-05 (γ 0.9) | EXP-06 (γ 0.5) |
|---|---|---|---|
| Val return | -7.64% ± 8.44% | **-3.97% ± 3.95%** | -10.46% ± 7.14% |
| Val per seed | -15.1 / -9.3 / +1.5 | -8.3 / -0.5 / -3.1 | -16.6 / -12.1 / -2.6 |
| val_w1 / val_w2 (mean) | -5.7 / -2.2 | -2.5 / -1.5 | -5.1 / -5.5 |
| Overestimation Q - G (val) | 0.035 | **0.003** | 0.0007 |
| Corr(position, next hour) train / val | +0.015 / -0.018 | +0.010 / +0.014 | +0.014 / -0.021 |
| Position changes on val / fees | 108 / 1.6% | 83 / 0.8% | 453 / 3.4% |
| Favourite position per seed | 2x / 0.5 / short | **long / long / long** | 0.5-long flip / 2x / short |

- **Verdicts:** EXP-05 kept. EXP-06 rejected. (Correction after EXP-13: the seed agreement below came from seeds 0 to 2 only and does not replicate on seeds 3 to 5; see EXP-13.)
- **Why:** γ = 0.9 cuts the overestimation by a factor of about 12 (0.035 → 0.003), as predicted by the 1 / (1 - γ) amplification of the max bias. For the first time all three seeds agree: all hold long about 96% of the time, which is the correct constant position for their training data (+131% over the 3 years). The validation mean is the best so far and the seed spread the smallest, but the result equals buy and hold (-4.27%), because the policy is essentially "long". The mean-reversion edge is still not exploited (83 changes, correlation on validation +0.014, within noise). γ = 0.5 is too short-sighted: a switch must pay its fee within about 2 hours, the policy churns (453 changes, one seed 1,228) and seeds disagree again.
- **Caution (eval exposure):** EXP-05's policy is long. From the baseline eval run I suspect the eval month rose, so a long policy would look good there. EXP-05 is kept for the measured reduction in overestimation and the seed agreement on validation, not because it is long. Success bar 2 (beat buy and hold on validation) is not met.

---

### EXP-07, EXP-08, EXP-09: Double DQN, normalisation on 3 years, mean-reversion features

- **Hypotheses:** EXP-07: the remaining overestimation (0.003) is the max bias; Double DQN (Session 2, slides 14 to 16) removes it and may stabilise the policy. EXP-08: normalisation failed on 1 year because of memorisation; with 4x more data its finer resolution may help instead. EXP-09: the consistent mean-reversion signals from the feature analysis (4h return, distance to 24h average, RSI 14h) let the agent exploit the short-term edge; normalisation is needed because the scales differ, EXP-08 isolates its effect.
- **Changes vs EXP-05:** EXP-07 `agent: double_dqn`. EXP-08 `normalize_features: true`. EXP-09 `features: mean_reversion` and `normalize_features: true` (7 market features).
- **Configs:** `configs/exp07_double_dqn.json`, `configs/exp08_normalize_3y.json`, `configs/exp09_mr_features.json`.
- **Seeds:** 0, 1, 2 each. (EXP-08 seeds 0 and 2 crashed on the first launch because of a cache race between 9 parallel processes, fixed with atomic cache writes, and were rerun; the runs are deterministic, so this does not change the results.)
- **Results (last model):**

| Metric | EXP-05 (parent) | EXP-07 Double DQN | EXP-08 normalise | EXP-09 MR features |
|---|---|---|---|---|
| Val return | **-3.97% ± 3.95%** | -6.25% ± 11.47% | -13.28% ± 9.11% | -8.32% ± 19.84% |
| Val per seed | -8.3 / -0.5 / -3.1 | -19.4 / -1.3 / +1.9 | -21.8 / -14.4 / -3.7 | -13.2 / +13.5 / -25.3 |
| Overestimation Q - G (val) | 0.003 | 0.0013 | 0.012 | 0.017 |
| Corr(position, next hour) train / val | 0.010 / +0.014 | 0.013 / +0.013 | 0.148 / -0.014 | 0.231 / +0.008 |
| Greedy return on the 3 training years | +153% | +179% | x26,000 | x10^12 |
| Position changes on val / fees | 83 / 0.8% | 65 / 0.9% | 493 / 7.0% | 926 / 12.6% |
| Policy per seed | long / long / long | 2x / long / 0.5 | mixed / flat / flat | mixed, about 45% flat |

- **Verdicts:** all three rejected.
- **Why:** EXP-07 halves the overestimation once more (0.003 → 0.0013), but with γ = 0.9 there was little bias left to remove; it brings no measurable gain and the seeds disagree again. EXP-08 and EXP-09 show the same pattern as EXP-01, now on 4x more data: with normalised inputs the network resolves states (in-sample correlation 0.15 to 0.23 instead of 0.01) and uses that to memorise the training years and to trade 500 to 900 times on validation. The mean-reversion features do carry the signal the network picks up in-sample, but nothing in the learning process stops it from fitting noise. **Lesson: information is not the bottleneck any more; the learning process lacks a brake against overfitting and churn.**

---

### EXP-10, EXP-11, EXP-12: regularising the informative setup (EXP-09)

- **Hypothesis:** I observe that with normalised mean-reversion features the network resolves the signal in-sample (correlation 0.23) but memorises it and churns (926 trades on validation); I believe the learning process has no brake against overfitting; I test more data per unit of resolution (EXP-10, all history), less capacity (EXP-11, hidden 32) and a training-only switching cost (EXP-12, 0.0005 per unit of position change, 5x the fee); I expect a smaller in-sample correlation, fewer trades and better validation. (Documented branch: EXP-09 was rejected, but it is the only setup in which the agent sees the signal.)
- **Configs:** `configs/exp10_mr_all_history.json`, `configs/exp11_mr_small_net.json`, `configs/exp12_mr_turnover_penalty.json`.
- **Seeds:** 0, 1, 2 each.
- **Results (last model):**

| Metric | EXP-09 | EXP-10 all history | EXP-11 hidden 32 | EXP-12 turnover penalty |
|---|---|---|---|---|
| Val return | -8.32% ± 19.84% | -14.25% ± 7.63% | -18.75% ± 7.90% | -10.57% ± 17.25% |
| Val per seed | -13.2 / +13.5 / -25.3 | -17.6 / -19.6 / -5.5 | -27.8 / -13.6 / -14.9 | -16.2 / -24.3 / +8.8 |
| Corr(position, next hour) train / val | 0.231 / +0.008 | 0.054 / -0.030 | 0.118 / -0.023 | 0.255 / -0.002 |
| Greedy return on the training frame | x10^12 | x57 | x86,000 | x10^11 |
| Position changes on val / fees | 926 / 12.6% | 262 / 3.9% | 682 / 8.3% | 739 / 8.6% |

- **Verdicts:** all three rejected.
- **Why:** more data reduces memorisation the most (in-sample correlation 0.23 → 0.05, trades 926 → 262), but validation is worse. The small network still memorises (x86,000). The turnover penalty is applied (training reward 0.24 per episode below the log return, about 400 switches x 0.0005) but the agent still switches in 55% of training hours: its memorised edge exceeds even a 6x higher switching cost, so a price on trading cannot stop memorisation.

### Two checks on existing models (no training)

**3-seed ensembles (average of the Q-values, `EnsembleAgent`)**, 10 validation episodes:

| Config | Mean of single seeds | Ensemble of seeds 0 to 2 | val_w1 / val_w2 | Trades | Main position |
|---|---|---|---|---|---|
| EXP-05 | -3.97% | -2.8% | -1.3% / -1.5% | 95 | long 96% |
| EXP-07 | -6.25% | -7.9% | -4.9% / -3.1% | 82 | long 96% |
| EXP-08 | -13.28% | -6.1% | -2.8% / -3.4% | 457 | flat 79% |
| EXP-09 | -8.32% | -4.8% | -6.5% / +2.7% | 523 | flat 76% |
| EXP-10 | -14.25% | -13.0% | -9.0% / -5.5% | 261 | flat 89% |
| EXP-11 | -18.75% | -11.7% | -7.2% / -5.7% | 487 | 0.5 76% |
| EXP-12 | -10.57% | -1.3% | -6.9% / +5.7% | 322 | flat 85% |

Averaging helps every normalised configuration (by 1 to 9 points) and roughly halves trading; where the seeds disagree, the average stays flat. Caveat: one number per ensemble and seven ensembles looked at on validation, so picking the best one would be selection bias; the effect has to be replicated on new seeds.

**Early checkpoints**: mean validation return per fifth of training (3 seeds, validation during training) improves over training for most configs (EXP-09: -21.6% → -5.3%, EXP-05: -9.0% → -1.2%). Early stopping would not help; longer training might.

---

### EXP-13 (pre-registered before training): ensemble replication on fresh seeds

- **Hypothesis:** averaging the Q-values of several seeds cancels seed-specific memorised noise and keeps shared signal (observed on seeds 0 to 2 after looking at 7 ensembles, so it must be replicated).
- **Test:** train fresh seeds 3, 4, 5 for EXP-05, EXP-09 and EXP-12 (configs unchanged); compare the ensemble of seeds 3 to 5 with the mean of the single seeds 3 to 5 on validation (10 episodes, full validation period).
- **Success criterion (fixed before training):** the ensemble beats the mean single seed in at least 2 of the 3 configs and trades less. Then the ensemble is kept.
- **Results (validation, 10 episodes):**

| Config | Single seeds 3 / 4 / 5 | Mean single | Trades single | Ensemble 3-5 | Trades ensemble | Better and fewer trades? |
|---|---|---|---|---|---|---|
| EXP-05 | -1.1% / -2.5% / -0.6% | -1.4% | 118 | +27.7% | 239 | better, but more trades |
| EXP-09 | -25.1% / +4.5% / -8.1% | -9.6% | 880 | -1.5% | 722 | yes |
| EXP-12 | -13.0% / -12.8% / -15.8% | -13.9% | 753 | -3.2% | 602 | yes |

- **Verdict:** criterion met in 2 of 3 configs: **ensemble effect replicated, ensemble kept** (it measurably helps).
- **The EXP-05 +27.7% is luck:** over all 20 three-seed ensembles from seeds 0 to 5, EXP-05 has median -2.1%, range -6.1% to +27.5%. Its averaged Q-values are practically tied (short 0.00273, flat 0.00273, 0.5 0.00278, long 0.00279), so the argmax flips on tiny differences.
- **Correction to EXP-05:** "all seeds agree on long" does not replicate. Seeds 3, 4, 5 end 92% short, 100% long and 93% flat. Over 6 seeds EXP-05 is -2.68% ± 2.93% on validation. The overestimation reduction by γ = 0.9 stands; the seed agreement was a lucky draw of seeds 0 to 2.

### Exploratory (not pre-registered): ensemble size, seeds 0 to 5

| Config | Single seed (6) | 3-seed ensembles, median [min, max] of 20 | 6-seed ensemble val | val_w1 / val_w2 | Trades | Main position |
|---|---|---|---|---|---|---|
| EXP-05 | -2.68% ± 2.93% | -2.1% [-6.1%, +27.5%] | +0.07% | -0.81% / +0.95% | 58 | long 97% |
| EXP-09 | -8.95% ± 15.70% | -4.7% [-14.2%, +22.6%] | +4.04% | +1.40% / +2.39% | 339 | flat 86% |
| EXP-12 | -12.22% ± 11.11% | +0.7% [-12.3%, +11.9%] | +14.33% | +11.56% / +2.44% | 244 | flat 89% |

The more seeds are averaged, the better the normalised mean-reversion configs become (EXP-12: -12.2% single → +0.7% median of 3 → +14.3% for 6). Consistent with variance reduction, but the 6-seed numbers were found after looking and are single draws; they must be replicated on fresh seeds before any decision.


---

### EXP-14 (pre-registered before training): 6-seed ensemble on fresh seeds 6 to 11

- **Hypothesis:** averaging more seeds reduces the variance from memorised noise further; the exploratory 6-seed ensembles (seeds 0 to 5) reached +14.33% (EXP-12) and +4.04% (EXP-09) on validation.
- **Test:** train fresh seeds 6 to 11 for EXP-12 and EXP-09 (configs unchanged); evaluate the ensemble of seeds 6 to 11 and the single seeds 6 to 11 on validation (10 episodes; full period plus both windows).
- **Success criteria (fixed before training), per config:**
  1. The 6-seed ensemble beats the mean single seed and buy and hold (-4.27%) on the full validation period.
  2. Stronger: it is positive and above always short (+4.24%) (success bar 3).
- **Decision rule:** a config that passes criterion 1 becomes the final candidate; if both pass, the one with the higher ensemble return, reported together with the 0 to 5 ensemble.

- **Results (pre-registered evaluation, validation, 10 episodes):**

| | EXP-12 (MR + penalty) | EXP-09 (MR features) |
|---|---|---|
| Single seeds 6 to 11 | -23.9 / -19.1 / -25.4 / -15.3 / -4.7 / -19.5% | +23.5 / -21.5 / -13.4 / -5.7 / -8.3 / +3.3% |
| Single mean ± std | -17.98% ± 7.43% | -3.70% ± 15.65% |
| 6-seed ensemble (seeds 6 to 11) | -2.81% | -2.17% |
| val_w1 / val_w2 | -0.57% / -2.56% | -0.12% / -2.09% |
| Trades / main position | 288 / flat 86% | 414 / flat 83% |
| Criterion 1 (> single mean and > buy and hold) | met | met |
| Criterion 2 (> 0 and > always short) | not met | not met |

- **Verdict:** the ensemble reliably lifts the configs above the mean single seed and above buy and hold (replicated), but the exploratory +14.33% of seeds 0 to 5 does not replicate: a 6-seed ensemble lands between about -3% and +14% depending on the seeds. Pre-registered decision rule: both pass criterion 1, the higher ensemble return wins: **EXP-09** (-2.17% vs -2.81%, a difference within noise; EXP-09 is also the simpler config without reward shaping).

### Exploratory: validation return vs ensemble size (seeds 0 to 11)

For each ensemble size k, 20 random seed subsets (k = 12: the one full set), validation with 3 episodes each. Figure: `figures/phase4_ensemble_size.png`.

| k | EXP-09 median [10%, 90%] | EXP-09 trades | EXP-12 median [10%, 90%] | EXP-12 trades |
|---|---|---|---|---|
| 1 | -13.1% [-25.2%, +23.5%] | 880 | -15.8% [-24.2%, +8.8%] | 779 |
| 2 | -13.1% [-23.0%, +3.4%] | 752 | -8.4% [-14.0%, -1.2%] | 634 |
| 3 | -5.3% [-15.7%, +9.4%] | 662 | -2.4% [-8.9%, +9.1%] | 536 |
| 4 | -2.1% [-14.5%, +9.0%] | 490 | +1.3% [-9.7%, +4.2%] | 439 |
| 6 | -1.6% [-6.1%, +6.8%] | 414 | +1.4% [-3.8%, +11.2%] | 268 |
| 8 | +1.7% [-2.2%, +5.6%] | 306 | +5.5% [+1.7%, +11.0%] | 218 |
| 12 | +5.00% (10 episodes; w1 -0.72%, w2 +5.64%; 92% flat) | 205 | +6.10% (10 episodes; w1 +3.00%, w2 +2.96%; 94% flat) | 126 |

- **Reading:** the median rises and the spread shrinks steadily with k in both configs, and trading falls by a factor of 4 to 6. With 12 networks both ensembles are positive and above always short (+4.24%) on the full validation period. EXP-12 is ahead at every k ≥ 3 and trades less. Caveats: exploratory (seeds 0 to 5 had been looked at before), one number at k = 12, and the 2 validation months are the only out-of-sample evidence.

---

### Final configuration choice (Phase 5, 2026-09-28)

- **Chosen:** EXP-12 as a 20-network ensemble (seeds 0 to 19, composition fixed before training seeds 12 to 19).
- **Deviation from the EXP-14 decision rule, stated openly:** the rule picked EXP-09 on one 6-network comparison (-2.17% vs -2.81%, inside the noise). The later ensemble size analysis (seeds 0 to 11, 20 random subsets per size) favours EXP-12 at every size ≥ 3, is even across both validation months with 12 networks (+3.0% / +3.0%) and trades less. The final choice weighs this broader evidence and the higher expected return over the single pre-registered comparison.
- **Not driven by the eval exposure:** the EXP-12 ensemble is about 94% flat on validation; the suspected rise of the eval month would favour long policies, not this one.

---

### Final model: validation and the one eval run (2026-09-28)

| | Validation (2 months) | val_w1 | val_w2 | Eval month (10 episodes) |
|---|---|---|---|---|
| Final model (EXP-12, 20 networks) | **+11.28%** (Sharpe 3.55, max DD -3.9%, 290 trades, 78% flat) | +8.17% | +2.67% | **-2.20%** |
| 20 single networks of EXP-12 | -10.86% ± 10.84% (3 of 20 positive) | | | not evaluated |
| Baseline EXP-00 seed 0 | -9.90% | -11.28% | -1.17% | +22.61% |
| Buy and hold | -4.26% | -2.69% | -1.54% | +21.12% |
| Always short | +4.24% | +2.67% | +1.53% | (not computed) |

- **Reading:** +11.3% is the best configuration found on validation (baseline mean -5.8%); it is optimistic, because every selection decision used the same two months, and the 20 member networks average -10.9% on that block. The eval month was a strong uptrend (+21%, most of it in the first 110 hours). Trailing buy and hold is structural for a mostly flat policy; ending below zero comes from the tails: the worst hour was a 2x position before a sharp drop (about -4.5%), and contrarian shorts against the rally cost about 2.6% (log). The baseline's +22.6% comes from its long exposure in a rising month, not from skill (-9.9% on validation).
- **Limitation this reveals:** two months of sideways validation data selected a regime-specific strategy. Validation did not contain a trending market, so it could not warn me.

### Post hoc (after freezing the final model, not used for selection): all main models on the eval month

Computed with `validate` on the eval env (10 episodes, seeded initial positions); table in `eval/posthoc_eval_comparison.csv`, figure `figures/phase5_val_vs_eval.png`.

| Model | Validation | Eval month | Main position on eval |
|---|---|---|---|
| Buy and hold | -4.27% | +21.12% | long |
| Always short | +4.24% | -21.12% | short |
| Flat | 0.00% | 0.00% | flat |
| Random | -18.21% | +8.72% | all (595 trades) |
| Baseline EXP-00, 3 single networks | -5.83% | +19.80% ± 5.72% (3/3 positive) | long / 0.5 |
| EXP-05 γ 0.9, 6 single networks | -2.68% | +14.04% ± 12.93% (5/6 positive) | long 66% |
| EXP-05 γ 0.9, ensemble of 6 | +0.07% | +13.72% | long 97% |
| EXP-09 MR features, 20 single networks | -9.30% | -5.24% ± 8.02% (4/20 positive) | mixed |
| EXP-09 MR features, ensemble of 20 | +2.15% | -7.53% | flat 92% |
| EXP-12 MR + penalty, 20 single networks | -10.86% | -2.86% ± 8.24% (5/20 positive) | mixed |
| **EXP-12 ensemble of 20 (final)** | **+11.28%** | **-2.20%** | flat 78% |

- **Reading:** the eval ranking is almost the reverse of the validation ranking. Long-leaning models win the +21% trend month; the cautious mean-reversion ensembles that won the sideways validation lose slightly. Even the random agent (0.5 long on average) beats the final model on this month.
- **Root cause:** all selection used one two-month sideways validation block, which could not distinguish skill from fit to the sideways regime. No feature carries trend information (the 168-hour return has no correlation with the next hour in any period), so the models that won the eval month were near-constant long bets, not trend-followers.
- **What I would do differently:** walk-forward validation over several blocks from different regimes (bull, bear, sideways), keeping only configurations that hold up in all of them; no leverage while the measured edge is about the size of the fee.

### Post hoc analysis of the final model (after the submission was fixed, not used for selection)

**Why the ensemble is mostly flat (validation states along the greedy path).** Across-network standard deviation of Q per action, and the ensemble's mean Q:

| Position | -1 | 0 (flat) | 0.5 | 1 | 2 |
|---|---|---|---|---|---|
| Std of Q across the 20 networks | 0.0070 | **0.0019** | 0.0021 | 0.0042 | **0.0105** |
| Ensemble mean Q | 0.0026 | **0.0094** | 0.0088 | 0.0058 | -0.0010 |

The networks disagree least about flat (no market exposure) and most about 2x. Averaging shrinks the exposed actions towards a common mean below flat, so flat wins by default: the ensemble is flat in 78% of states, a single network in 41%. This shrinkage is also why the ensemble cannot follow a trend. Related methods: Averaged-DQN (Anschel, Baram and Shimkin, ICML 2017) and Bootstrapped DQN (Osband, Blundell, Pritzel and Van Roy, NeurIPS 2016).

**Where the eval result comes from (one greedy episode per period).**

| | Validation | Eval month |
|---|---|---|
| Return | +11.33% | -2.09% |
| Worst single hour | -2.38% (at 2x) | -4.48% (at 2x) |
| Log return while short / flat / 0.5 / long / 2x | +0.008 / -0.009 / +0.057 / +0.002 / +0.048 | -0.026 / -0.005 / +0.017 / -0.002 / -0.005 |
| Same policy with 2x capped at 1x | +8.07% | +0.01% |

Leverage added about 3 points on validation and cost about 2 on the eval month: with an edge about the size of the fee, it mainly amplifies variance.

**Did the signal survive?** Rank correlation with the next-hour return (noise ±0.012 / ±0.052 / ±0.073):

| Feature | Dev-train (3 years) | Validation | Eval month |
|---|---|---|---|
| 4-hour return | -0.051 | -0.060 | -0.020 |
| Distance to the 24-hour average | -0.041 | -0.064 | -0.003 |
| RSI 14h | -0.030 | -0.066 | -0.002 |
| Last-hour return | -0.062 | -0.031 | -0.072 |
| 168-hour return | -0.006 | -0.023 | -0.021 |

The multi-hour mean reversion was stronger on validation than on average and is not detectable on the eval month; only the one-hour reversal remains. One month cannot prove that the signal vanished, but the result is consistent with it and explains why validation overstated the edge.
