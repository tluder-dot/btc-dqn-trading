"""Evaluation: benchmark agents, episode metrics and multi-episode evaluation.

Every agent only needs `choose_action_eval(state) -> action_index`, the same
interface that the course's `evaluate_agent` uses.
"""

import numpy as np
import pandas as pd

from rl_trading.envs import TRADING_FEES

HOURS_PER_YEAR = 24 * 365


# ---------------------------------------------------------------------------
# Benchmark agents
# ---------------------------------------------------------------------------

class ConstantAgent:
    """Always chooses the same position (e.g. flat, buy and hold, 2x, short)."""

    def __init__(self, positions, position):
        self.action_index = positions.index(position)

    def choose_action_eval(self, state):
        return self.action_index


class RandomAgent:
    """Chooses a uniformly random position every step (own seeded generator)."""

    def __init__(self, n_actions, seed=0):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def choose_action_eval(self, state):
        return int(self.rng.integers(self.n_actions))


# ---------------------------------------------------------------------------
# One episode and its metrics
# ---------------------------------------------------------------------------

def run_episode(agent, env):
    """Run one greedy episode and return its metrics plus the equity curve.

    Same loop as the course's `evaluate_agent`; metrics are computed from the
    env's history (unrounded) instead of the 2-decimal strings of get_metrics().
    """
    obs, info = env.reset()
    done = truncated = False
    steps, reward_total = 0, 0.0
    while not (done or truncated):
        action = agent.choose_action_eval(obs)
        obs, reward, done, truncated, info = env.step(action)
        reward_total += reward
        steps += 1

    history = env.unwrapped.historical_info
    values = np.asarray(history["portfolio_valuation"], dtype=float)  # V_0 .. V_T
    closes = np.asarray(history["data_close"], dtype=float)
    positions = np.asarray(history["position"], dtype=float)          # initial position, then one per step
    metrics = episode_metrics(values, closes, positions, env.unwrapped.positions)
    metrics.update(steps=steps, total_reward=reward_total, bankrupt=bool(done))
    return metrics, values / values[0]


def episode_metrics(values, closes, positions, position_list):
    """Metrics of one episode from its valuation, price and position paths."""
    log_ret = np.diff(np.log(values))
    running_max = np.maximum.accumulate(values)
    changes = np.abs(np.diff(positions))  # position changes, including the first switch from the initial position
    metrics = {
        "portfolio_return": values[-1] / values[0] - 1,
        "market_return": closes[-1] / closes[0] - 1,
        "sharpe": (log_ret.mean() / log_ret.std() * np.sqrt(HOURS_PER_YEAR)) if log_ret.std() > 0 else 0.0,
        "max_drawdown": (values / running_max - 1).min(),
        "n_changes": int((changes > 0).sum()),
        "fees_approx": TRADING_FEES * changes.sum(),  # in fractions of the portfolio
    }
    metrics["excess_return"] = metrics["portfolio_return"] - metrics["market_return"]
    for p in position_list:  # share of steps spent in each position (after acting)
        metrics[f"share_pos_{p:g}"] = float(np.mean(positions[1:] == p))
    return metrics


# ---------------------------------------------------------------------------
# Several episodes (mirrors evaluate_agent: num_episodes resets of the same env)
# ---------------------------------------------------------------------------

def evaluate(agent, env, num_episodes=10):
    """Run `num_episodes` episodes; return a DataFrame of metrics and the equity curves."""
    rows, curves = [], []
    for ep in range(num_episodes):
        metrics, equity = run_episode(agent, env)
        rows.append({"episode": ep + 1, **metrics})
        curves.append(equity)
    return pd.DataFrame(rows), curves


def rollout_with_q(agent, env):
    """One greedy episode with per-step details for diagnostics.

    Returns a DataFrame with one row per step t: the Q-values of all actions in
    s_t, the greedy action, the position held after acting, the reward
    r_t = log(V_(t+1) / V_t) and the market log return over the same hour.
    """
    import torch  # only needed here

    obs, info = env.reset()
    prev_close = info["data_close"]
    rows, done, truncated = [], False, False
    while not (done or truncated):
        state = torch.as_tensor(np.array(obs, dtype=np.float32), device=agent.device).unsqueeze(0)
        with torch.no_grad():
            q = agent.q_network(state).squeeze(0).cpu().numpy()
        action = int(q.argmax())
        obs, reward, done, truncated, info = env.step(action)
        rows.append({"action": action, "position": info["position"], "reward": reward,
                     "market_log_return": np.log(info["data_close"] / prev_close),
                     **{f"q_{i}": q[i] for i in range(len(q))}})
        prev_close = info["data_close"]
    return pd.DataFrame(rows)


def discounted_returns(rewards, gamma):
    """G_t = sum_k gamma^k r_(t+k) for every t (computed backwards, as in TD4)."""
    returns = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def diagnose_agent(agent, envs, gamma, horizon_cut=460):
    """Phase 3 diagnostics of a greedy agent, one row per env (see notebook 02).

    return            portfolio return of one greedy episode
    mean_pred_Q       mean Q(s_t, a*_t) of the chosen actions
    mean_realised_G   mean discounted return realised afterwards (same units)
    overestimation    mean_pred_Q - mean_realised_G
    corr_pos_next_1h  correlation of the held position with the next-hour market return
    action_gap        median of Q(best) - Q(second best)
    changes, mean_hold_h, fees_approx, share_pos_*   trading behaviour
    The last `horizon_cut` steps are excluded from the Q vs G comparison because
    their realised return is truncated (gamma^460 < 0.01 for gamma = 0.99).
    """
    rows = []
    rng_state = np.random.get_state()
    for name, env in envs.items():
        np.random.seed(0)
        df = rollout_with_q(agent, env)
        q = df[[c for c in df.columns if c.startswith("q_")]].to_numpy()
        pred = q.max(axis=1)
        realised = discounted_returns(df["reward"].to_numpy(), gamma)
        keep = slice(0, max(1, len(df) - horizon_cut))
        q_sorted = np.sort(q, axis=1)
        pos = df["position"].to_numpy(float)
        changes = np.flatnonzero(np.diff(pos) != 0)
        rows.append({
            "env": name,
            "return": np.expm1(df["reward"].sum()),
            "mean_pred_Q": pred[keep].mean(),
            "mean_realised_G": realised[keep].mean(),
            "overestimation": (pred[keep] - realised[keep]).mean(),
            "corr_pos_next_1h": np.corrcoef(pos, df["market_log_return"])[0, 1] if pos.std() > 0 else 0.0,
            "action_gap": np.median(q_sorted[:, -1] - q_sorted[:, -2]),
            "changes": len(changes),
            "mean_hold_h": len(pos) / (len(changes) + 1),
            "fees_approx": TRADING_FEES * np.abs(np.diff(pos)).sum(),
            **{f"share_pos_{p:g}": float(np.mean(pos == p)) for p in env.unwrapped.positions},
        })
    np.random.set_state(rng_state)
    return pd.DataFrame(rows).set_index("env")


def policy_agreement(agent_a, agent_b, states):
    """Share of states on which two agents choose the same greedy action."""
    import torch
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(states, dtype=np.float32))
        a = agent_a.q_network(x).argmax(dim=1)
        b = agent_b.q_network(x).argmax(dim=1)
    return float((a == b).float().mean())


def summarize_experiment(run_root, model="last", seeds=(0, 1, 2)):
    """Mean ± std over seeds of the final validation results of one experiment.

    run_root: runs/<exp_id>; reads seed_<k>/final_val.csv (10 episodes per env)
    for the given seeds (default 0, 1, 2: the seeds of every experiment's main run).
    """
    from pathlib import Path
    frames = [pd.read_csv(Path(run_root) / f"seed_{s}" / "final_val.csv").assign(seed=f"seed_{s}") for s in seeds]
    df = pd.concat(frames)
    df = df[df["model"] == model]
    return df.groupby("env")[SUMMARY_COLUMNS[:1] + ["excess_return", "n_changes"]].agg(["mean", "std"]), df


SUMMARY_COLUMNS = ["portfolio_return", "market_return", "excess_return", "sharpe",
                   "max_drawdown", "n_changes", "fees_approx"]


def validate(agent, envs, num_episodes=10, seed=0):
    """Evaluate `agent` on every env in `envs` (see envs.make_validation_envs).

    Before each env, np.random is seeded so every agent faces the same random
    initial positions: comparisons between agents are then paired, not luck.
    The global np.random state is restored afterwards, so validating during
    training does not change the training env's random initial positions.
    Returns (summary, details): summary has one row per env with the mean of
    each metric plus the std of the portfolio return across episodes;
    details maps env name to the per-episode DataFrame.
    """
    rng_state = np.random.get_state()
    rows, details = [], {}
    for name, env in envs.items():
        np.random.seed(seed)
        df, _ = evaluate(agent, env, num_episodes)
        details[name] = df
        share_cols = [c for c in df.columns if c.startswith("share_pos_")]
        row = {"env": name, "steps": int(df["steps"].iloc[0])}
        row.update(df[SUMMARY_COLUMNS + share_cols].mean())
        row["portfolio_return_std"] = df["portfolio_return"].std()
        row["bankruptcies"] = int(df["bankrupt"].sum())
        rows.append(row)
    np.random.set_state(rng_state)
    return pd.DataFrame(rows).set_index("env"), details


# ---------------------------------------------------------------------------
# Official evaluation function (copied verbatim from
# course_material/Project_Instruction.ipynb, do not modify)
# ---------------------------------------------------------------------------

import time  # noqa: E402  (used only by evaluate_agent, kept next to it)


def evaluate_agent(agent, env, num_episodes=10, max_steps=None, render=False, csv_path="evaluation_results.csv", renderer_logs_dir="render_logs"):
    """
    Evaluate the agent on the environment for a number of episodes.
    """

    results = []
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        step = 0
        reward_total = 0.0
        while not done and not truncated:
            action = agent.choose_action_eval(obs)
            obs, reward, done, truncated, info = env.step(action)
            reward_total += reward
            step += 1
            if (max_steps is not None) and (step >= max_steps):
                break

        # Get metrics from the environment
        metrics = env.get_metrics()
        # Transform metrics
        # Assume metrics contain keys "Portfolio Return" and "Market Return" as strings like "45.24%"
        port_ret = float(metrics["Portfolio Return"].strip('%')) / 100.0
        market_ret = float(metrics["Market Return"].strip('%')) / 100.0

        results.append({
            "episode": ep+1,
            "portfolio_return": port_ret,
            "market_return": market_ret,
            "excess_return": port_ret - market_ret,
            "steps": step,
            "total_reward": reward_total,
        })
        if render:
            print(f"Eval Episode {ep+1}: Total Reward: {reward_total:.2f}, Portfolio Return: {port_ret:.2%}, Market Return: {market_ret:.2%}, Excess Return: {(port_ret - market_ret):.2%}, Steps: {step}")
            time.sleep(1)  # Pause between episodes in case the execution is too fast and files are not saved properly
            env.save_for_render(dir = renderer_logs_dir)

    df_results = pd.DataFrame(results)

    df_results.to_csv(csv_path, index=False)
    print(f"Saved submission to {csv_path}")

    return df_results


# ---------------------------------------------------------------------------
# Training plots (TD style: raw values plus moving average, one line per seed)
# ---------------------------------------------------------------------------

def plot_training_runs(run_dirs, benchmarks=None, title="", save_path=None, ma_window=10):
    """2 x 3 grid of training diagnostics for one experiment (several seeds).

    run_dirs:   list of runs/<exp>/seed_<k> directories
    benchmarks: optional {label: validation return} drawn as reference lines
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from pathlib import Path

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    seed_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    bench_colors = {"buy and hold": "#8a8984", "always short": "#8a8984", "flat": "#c3c2b7"}
    pct = FuncFormatter(lambda v, _: f"{v:+.0%}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=surface)
    panels = [
        (axes[0, 0], "Training episode portfolio return", "train_portfolio_return", "train", pct),
        (axes[0, 1], "Mean TD loss per episode (log scale)", "mean_loss", "train", None),
        (axes[0, 2], "Mean max Q on fixed probe states", "mean_max_q", "train", None),
        (axes[1, 0], "Validation return, greedy policy", "val_portfolio_return", "val", pct),
        (axes[1, 1], "Validation position changes (1,463 h)", "val_n_changes", "val", None),
        (axes[1, 2], "Epsilon", "epsilon", "train", None),
    ]
    for ax, panel_title, col, source, fmt in panels:
        ax.set_facecolor(surface)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=ink2, labelsize=9, length=0)
        ax.set_title(panel_title, loc="left", color=ink, fontsize=10.5)
        ax.set_xlabel("training episode", color=ink2, fontsize=9)
        for k, run_dir in enumerate(run_dirs):
            log = pd.read_csv(Path(run_dir) / ("train_log.csv" if source == "train" else "val_log.csv"))
            color = seed_colors[k % len(seed_colors)]
            y = log[col]
            if col in ("epsilon",) and k > 0:
                continue  # identical for all seeds
            raw_alpha = 0.25 if len(y) >= ma_window and col not in ("epsilon",) else 1.0
            ax.plot(log["episode"], y, color=color, lw=1, alpha=raw_alpha)
            if raw_alpha < 1:
                ma = y.rolling(ma_window).mean()
                ax.plot(log["episode"], ma, color=color, lw=1.8, label=f"seed {Path(run_dir).name.split('_')[-1]}")
        if col == "mean_loss":
            ax.set_yscale("log")
        if fmt is not None:
            ax.yaxis.set_major_formatter(fmt)
        if col == "val_portfolio_return" and benchmarks:
            for label, value in benchmarks.items():
                ax.axhline(value, color=bench_colors.get(label, "#8a8984"), lw=1.1, ls="--")
                # flat on the left, others on the right; negative values below their line
                x, ha = (0.0, "left") if value == 0 else (1.0, "right")
                dy, va = (-3, "top") if value < 0 else (3, "bottom")
                ax.annotate(f"{label} {value:+.1%}", (x, value), xycoords=("axes fraction", "data"),
                            xytext=(2 if ha == "left" else -2, dy), textcoords="offset points",
                            ha=ha, va=va, fontsize=8.5, color=ink2)
    axes[0, 0].legend(frameon=False, fontsize=9, labelcolor=ink2, title=f"moving avg ({ma_window})",
                      title_fontsize=8.5, loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", color=ink, fontsize=12.5)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, facecolor=surface)
    return fig


def plot_experiment_summary(experiments, run_root, benchmarks=None, title="", save_path=None, env="val", seeds=(0, 1, 2)):
    """Validation return per experiment: seeds as dots, mean ± std as a bar with error bar.

    experiments: list of (exp_id, label, verdict) with verdict in {"reference", "kept", "rejected", "inconclusive"}
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from pathlib import Path

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    colors = {"reference": "#8a8984", "kept": "#2a78d6", "rejected": "#c3c2b7", "inconclusive": "#9ec5f0"}
    fig, ax = plt.subplots(figsize=(max(8, 1.05 * len(experiments) + 2), 4.8), facecolor=surface)
    ax.set_facecolor(surface)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=ink2, labelsize=9, length=0)
    for i, (exp_id, label, verdict) in enumerate(experiments):
        _, per_seed = summarize_experiment(Path(run_root) / exp_id, seeds=seeds)
        values = per_seed[per_seed["env"] == env]["portfolio_return"].to_numpy()
        ax.bar(i, values.mean(), 0.62, color=colors[verdict], yerr=values.std(ddof=1),
               error_kw=dict(ecolor=ink2, lw=1, capsize=3))
        ax.scatter(np.full(len(values), i) + np.linspace(-0.12, 0.12, len(values)), values,
                   s=14, color=ink, zorder=3)
        ax.annotate(f"{values.mean():+.1%}", (i, max(values.max(), values.mean() + values.std(ddof=1))),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5, color=ink)
    for label, value in (benchmarks or {}).items():
        ax.axhline(value, color="#8a8984", lw=1, ls="--")
        ax.annotate(f"{label} {value:+.1%}", (1.0, value), xycoords=("axes fraction", "data"),
                    xytext=(-2, 3 if value >= 0 else -3), textcoords="offset points", ha="right",
                    va="bottom" if value >= 0 else "top", fontsize=8.5, color=ink2)
    ax.axhline(0, color=grid, lw=1)
    ax.set_xticks(range(len(experiments)), [label for _, label, _ in experiments], fontsize=8.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0%}"))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, list(colors), frameon=False, fontsize=8.5, labelcolor=ink2, loc="lower left", ncol=4)
    ax.set_title(title, loc="left", color=ink, fontsize=12)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, facecolor=surface)
    return fig
