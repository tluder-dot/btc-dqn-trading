"""Environment factories for training, validation and evaluation.

All envs share the same positions, fees and borrow rate as the instruction
notebook. Custom reward functions are for training envs only: validation and
eval envs keep the default log return reward so their rewards stay comparable.
"""

import warnings

import gymnasium as gym
import numpy as np
import gym_trading_env  # noqa: F401  (registers "TradingEnv" with gymnasium)
import pandas as pd

# Default action set of the instruction notebook: -1 short, 0 flat, 0.5 half,
# 1 fully long, 2 long with 2x leverage. The action is the index into this list.
POSITIONS = [-1, 0, 0.5, 1, 2]

TRADING_FEES = 0.01 / 100             # 0.01% of the traded amount (fixed by the course)
BORROW_INTEREST_RATE = 0.0003 / 100   # 0.0003% per step (fixed by the course)


def prepare_env_data(frame):
    """Drop the timezone from the index (copied from the instruction notebook).

    TradingEnv stores history dates without timezone information; keep UTC
    clock times but match that format for save_for_render().
    """
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index, utc=True).tz_localize(None)
    if "date_close" in frame.columns:
        frame["date_close"] = pd.to_datetime(frame["date_close"], utc=True).dt.tz_localize(None)
    return frame


def make_env(df, positions=POSITIONS, reward_function=None, max_episode_duration="max",
             initial_position="random", verbose=0, name="BTCUSD"):
    """Create a TradingEnv on `df` with the course's fees and borrow rate.

    reward_function: None keeps the default log return reward (required for
        validation and eval envs).
    max_episode_duration: "max" = one episode over the whole df (as in eval);
        an int N = episodes of N - 1 steps starting at a random index.
    """
    kwargs = dict(
        name=name,
        df=prepare_env_data(df),
        positions=positions,
        trading_fees=TRADING_FEES,
        borrow_interest_rate=BORROW_INTEREST_RATE,
        max_episode_duration=max_episode_duration,
        initial_position=initial_position,
        verbose=verbose,
    )
    if reward_function is not None:
        kwargs["reward_function"] = reward_function
    env = gym.make("TradingEnv", **kwargs)

    # The first gym.make("TradingEnv") imports gym_trading_env.environments, which
    # calls warnings.filterwarnings("error") process wide: from then on every
    # harmless pandas / numpy / torch warning would raise an exception.
    # Put the normal "show the warning" behaviour back in front of that filter.
    warnings.simplefilter("default")
    return env


# One eval episode covers 744 hourly candles (2026-08-17 to 2026-09-17), i.e. 743 steps.
EVAL_EPISODE_CANDLES = 744


def make_validation_envs(val_df, positions=POSITIONS):
    """Validation envs with the default reward, keyed by name.

    "val":    the full validation period (1,464 candles, 1,463 steps)
    "val_w1": its first 744 candles  (same length as an eval episode)
    "val_w2": its last 744 candles   (overlaps val_w1 by 24 candles, since 1,464 < 2 x 744)
    The two windows put returns on the same time scale as the eval CSV and show
    whether a result holds in both months or comes from one of them.
    """
    return {
        "val": make_env(val_df, positions=positions),
        "val_w1": make_env(val_df.iloc[:EVAL_EPISODE_CANDLES], positions=positions),
        "val_w2": make_env(val_df.iloc[-EVAL_EPISODE_CANDLES:], positions=positions),
    }


def make_turnover_penalty_reward(penalty):
    """Training reward: log return minus a cost per unit of position change.

    r'_t = log(V_(t+1) / V_t) - penalty * |position_t - position_(t-1)|
    The log return already contains the real 0.01% fee; the extra penalty makes
    switching more expensive during training only, so the agent learns to switch
    only when the expected gain covers it. Validation and eval keep the default reward.
    """
    def reward_function(history):
        log_return = np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])
        change = abs(history["position", -1] - history["position", -2])
        return log_return - penalty * change
    return reward_function
