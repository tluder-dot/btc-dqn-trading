"""Training loop (TD3 style) with logging, validation and checkpointing.

Usage (from the project root):
    python -m rl_trading.train --config configs/baseline.json --seed 0

Writes to runs/<exp_id>/seed_<seed>/:
    config.json        the exact config used (plus seed)
    train_log.csv      one row per training episode
    val_log.csv        one row per validation during training
    probe_states.npy   fixed states used to track the mean max-Q over training
    last.pt, best.pt   weights at the end / at the best validation return
    final_val.csv      validation summary of the last and the best model (10 episodes)
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rl_trading.agents import DoubleDQNAgent, DQNAgent
from rl_trading.data import (FEATURE_SETS, VAL_START, apply_normalizer, fit_normalizer, gap_free_segments,
                              load_history, load_raw_data, split_data)
from rl_trading.envs import make_env, make_turnover_penalty_reward, make_validation_envs
from rl_trading.evaluate import validate

ROOT = Path(__file__).resolve().parents[1]
AGENT_CLASSES = {"dqn": DQNAgent, "double_dqn": DoubleDQNAgent}


def set_seeds(seed):
    """Seed every source of randomness (the env draws from the global np.random)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_splits(config):
    """dev_train / val / eval dataframes exactly as a run with this config sees them.

    Used by training and by every later evaluation, so both share one pipeline.
    Optional z-score normalisation is fitted on dev_train only.
    Returns (splits, normalizer statistics or None).
    """
    add_features = FEATURE_SETS[config["features"]]
    train_start = pd.Timestamp(config.get("train_start", "2025-08-17"), tz="UTC")
    if config.get("train_segments", False):
        # Long history with exchange gaps: features per gap-free segment (no window
        # crosses a gap), keep segments long enough for one training episode.
        splits = split_data(add_features(load_raw_data(ROOT / "data")))  # val and eval as always
        segments = [add_features(s) for s in gap_free_segments(load_history(ROOT / "data", train_start), VAL_START)]
        min_len = config["env"]["max_episode_duration"] + 1
        splits["dev_train_segments"] = [s for s in segments if len(s) >= min_len]
        splits["dev_train"] = pd.concat(splits["dev_train_segments"])
    else:
        raw = load_raw_data(ROOT / "data", train_start=train_start)
        splits = split_data(add_features(raw), train_start=train_start)
    stats = None
    if config.get("normalize_features", False):
        stats = fit_normalizer(splits["dev_train"])
        splits = {name: [apply_normalizer(s, stats) for s in df] if isinstance(df, list) else apply_normalizer(df, stats)
                  for name, df in splits.items()}
    return splits, stats


def make_probe_states(train_df, positions, n=1024, seed=0):
    """Fixed batch of states to track Q-value estimates over training.

    Market features from n evenly spaced training hours, combined with a
    random position (last position = real position). Built once, never trained on.
    """
    features = train_df[[c for c in train_df.columns if "feature" in c]].to_numpy(np.float32)
    idx = np.linspace(0, len(features) - 1, n).astype(int)
    pos = np.random.default_rng(seed).choice(positions, size=n).astype(np.float32)
    return np.column_stack([features[idx], pos, pos])


def mean_max_q(agent, probe_states):
    with torch.no_grad():
        q = agent.q_network(torch.as_tensor(probe_states, device=agent.device))
    return q.max(dim=1).values.mean().item()


def train(config, seed, out_root=ROOT / "runs"):
    out_dir = Path(out_root) / config["exp_id"] / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump({**config, "seed": seed}, f, indent=2)

    torch.set_num_threads(1)  # small MLP: one thread per run, several seeds in parallel
    set_seeds(seed)

    # Data and environments (validation envs keep the default reward)
    splits, stats = prepare_splits(config)
    if stats is not None:
        with open(out_dir / "normalizer.json", "w") as f:
            json.dump(stats, f, indent=2)
    positions = config["positions"]
    duration = config["env"]["max_episode_duration"]
    train_frames = splits.get("dev_train_segments", [splits["dev_train"]])
    penalty = config.get("reward", {}).get("turnover_penalty", 0.0)
    reward_function = make_turnover_penalty_reward(penalty) if penalty > 0 else None  # training envs only
    train_envs = [make_env(f, positions=positions, max_episode_duration=duration, reward_function=reward_function)
                  for f in train_frames]
    # Several segments: pick one per episode in proportion to its number of start hours,
    # so every start hour is equally likely (own generator, env randomness untouched)
    start_counts = np.array([len(f) - (duration if isinstance(duration, int) else 0) for f in train_frames], float)
    segment_probs = start_counts / start_counts.sum()
    segment_rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
    train_env = train_envs[0]
    all_val_envs = make_validation_envs(splits["val"], positions=positions)
    val_envs = {name: all_val_envs[name] for name in config["train"]["val_envs"]}

    # Agent
    state_size = train_env.observation_space.shape[0]
    agent = AGENT_CLASSES[config["agent"]](state_size=state_size, n_actions=len(positions),
                                           seed=seed, **config["agent_params"])
    probe_states = make_probe_states(splits["dev_train"], positions)
    np.save(out_dir / "probe_states.npy", probe_states)

    episodes = config["train"]["episodes"]
    eval_every = config["train"]["eval_every"]
    train_rows, val_rows = [], []
    best_val = -np.inf
    start = time.time()

    for episode in range(episodes):

        # Reset environment (choose a training segment first if there are several)
        if len(train_envs) > 1:
            train_env = train_envs[segment_rng.choice(len(train_envs), p=segment_probs)]
        state, _ = train_env.reset()
        terminated = truncated = False
        episode_reward, episode_losses, actions = 0.0, [], []

        while not (terminated or truncated):
            # Choose action (epsilon-greedy)
            action = agent.choose_action(state)

            # Take action in environment
            next_state, reward, terminated, truncated, info = train_env.step(action)

            # Store experience (done = terminated only; truncation is not terminal)
            agent.store_experience(state, action, reward, next_state, terminated)

            # Train the agent (one update per environment step, as in TD3)
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

            # Move to next state
            state = next_state
            episode_reward += reward
            actions.append(action)

        # Decay exploration rate (per episode, as in TD3)
        epsilon_used = agent.epsilon
        agent.decay_epsilon()

        # Log the training episode
        history = train_env.unwrapped.historical_info
        values = np.asarray(history["portfolio_valuation"], dtype=float)
        held = np.asarray(history["position"], dtype=float)
        action_counts = np.bincount(actions, minlength=len(positions)) / len(actions)
        row = {
            "episode": episode + 1,
            "steps": len(actions),
            "epsilon": epsilon_used,
            "train_reward": episode_reward,
            "train_portfolio_return": values[-1] / values[0] - 1,
            "train_position_changes": int((np.diff(held) != 0).sum()),
            "mean_loss": float(np.mean(episode_losses)) if episode_losses else np.nan,
            "mean_max_q": mean_max_q(agent, probe_states),
            "updates": agent.step_count,
            "minutes": (time.time() - start) / 60,
            **{f"action_share_{p:g}": s for p, s in zip(positions, action_counts)},
        }
        train_rows.append(row)
        pd.DataFrame(train_rows).to_csv(out_dir / "train_log.csv", index=False)

        # Validate the greedy policy every `eval_every` episodes and at the end
        msg = ""
        if (episode + 1) % eval_every == 0 or episode + 1 == episodes:
            summary, _ = validate(agent, val_envs, num_episodes=config["train"]["val_episodes"])
            val_row = {"episode": episode + 1}
            for env_name, r in summary.iterrows():
                for col in ["portfolio_return", "excess_return", "sharpe", "max_drawdown", "n_changes"]:
                    val_row[f"{env_name}_{col}"] = r[col]
                for p in positions:
                    val_row[f"{env_name}_share_pos_{p:g}"] = r[f"share_pos_{p:g}"]
            val_rows.append(val_row)
            pd.DataFrame(val_rows).to_csv(out_dir / "val_log.csv", index=False)

            val_return = summary.loc["val", "portfolio_return"]
            if val_return > best_val:
                best_val = val_return
                agent.save_model(out_dir / "best.pt")
            msg = f" | val return {val_return:+.2%} (best {best_val:+.2%})"

        print(f"episode {episode + 1:3d}/{episodes}  eps {epsilon_used:.3f}  "
              f"train return {row['train_portfolio_return']:+8.2%}  loss {row['mean_loss']:.2e}  "
              f"maxQ {row['mean_max_q']:+.4f}  changes {row['train_position_changes']:5d}  "
              f"{row['minutes']:.1f} min{msg}", flush=True)

    agent.save_model(out_dir / "last.pt")

    # Final validation of the last and the best model on all validation envs (10 episodes)
    final = []
    for tag in ["last", "best"]:
        agent.load_model(out_dir / f"{tag}.pt")
        summary, _ = validate(agent, all_val_envs, num_episodes=10)
        final.append(summary.reset_index().assign(model=tag))
    final = pd.concat(final)
    final.to_csv(out_dir / "final_val.csv", index=False)
    print(final[["model", "env", "portfolio_return", "excess_return", "n_changes"]].to_string(index=False))
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Train a DQN agent on BTC/USDT hourly data")
    parser.add_argument("--config", required=True, help="path to an experiment config (JSON)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "runs"), help="output root directory")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    train(config, args.seed, args.out)


if __name__ == "__main__":
    main()
