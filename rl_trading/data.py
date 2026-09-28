"""Data pipeline: download, validation, features and splits.

Mirrors the data cell of course_material/Project_Instruction.ipynb (same Binance
source, same validate_hourly, same default features, same cache file name) and
adds my validation split carved out of the training year.

All timestamps are UTC. Ranges are start-inclusive and end-exclusive.
"""

import json
import os
import ssl
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import certifi
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Fixed dates
# ---------------------------------------------------------------------------

# Fixed evaluation window shared by all students (never change).
EVAL_START = pd.Timestamp("2026-08-17", tz="UTC")
EVAL_END = pd.Timestamp("2026-09-17", tz="UTC")
TRAIN_START = EVAL_START - pd.DateOffset(years=1)  # 2025-08-17

# My validation split: the last two months before the eval window.
# Everything before VAL_START is dev-train; all model selection uses validation.
VAL_START = pd.Timestamp("2026-06-17", tz="UTC")

# Extra history before TRAIN_START so rolling features are defined from the first
# training candle on (the notebook uses 7 days = 168 candles).
WARMUP_DAYS = 7

OHLCV = ["open", "high", "low", "close", "volume"]


def validate_hourly(frame, start, end, label):
    """Check that `frame` holds exactly one valid candle per hour in [start, end).

    Copied from Project_Instruction.ipynb. Raises instead of silently returning
    a shorter dataset.
    """
    expected = pd.date_range(start, end, freq="h", inclusive="left")
    missing = expected.difference(frame.index)
    unexpected = frame.index.difference(expected)
    if (frame.empty or frame.index.has_duplicates or not frame.index.is_monotonic_increasing
            or len(missing) or len(unexpected)):
        raise ValueError(f"{label}: incomplete/invalid hourly data; "
                         f"missing={len(missing)}, unexpected={len(unexpected)}, "
                         f"first missing={missing[:5].tolist()}")
    values = frame[OHLCV]
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{label}: non-finite OHLCV values")
    if ((values[["open", "high", "low", "close"]] <= 0).any().any()
            or (values["volume"] < 0).any()
            or (values["high"] < values[["open", "low", "close"]].max(axis=1)).any()
            or (values["low"] > values[["open", "high", "close"]].min(axis=1)).any()):
        raise ValueError(f"{label}: invalid OHLCV values")
    print(f"{label}: {len(frame):,} candles, {frame.index.min()} to {frame.index.max()}; no missing hours")


# ---------------------------------------------------------------------------
# Download and cache
# ---------------------------------------------------------------------------

def download_klines(start, end, symbol="BTCUSDT", interval="1h"):
    """Download hourly klines from the public Binance market-data API.

    Same pagination loop as the instruction notebook: 1000 candles per request,
    the cursor advances to the hour after the last candle received.
    """
    rows = []
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cursor < end_ms:
        query = urlencode(dict(symbol=symbol, interval=interval, startTime=cursor,
                               endTime=end_ms - 1, limit=1000))
        with urlopen("https://data-api.binance.vision/api/v3/klines?" + query, timeout=30,
                     context=ssl.create_default_context(cafile=certifi.where())) as response:
            batch = json.load(response)
        if not isinstance(batch, list) or not batch:
            raise RuntimeError(f"Binance returned no usable data at {cursor}: {batch}")
        next_cursor = int(batch[-1][0]) + 3_600_000
        if next_cursor <= cursor:
            raise RuntimeError("Binance pagination did not advance")
        rows.extend(batch)
        cursor = next_cursor
        time.sleep(0.1)

    # Kline row: [open_time, open, high, low, close, volume, close_time, ...]
    df = pd.DataFrame([row[1:6] for row in rows], columns=OHLCV, dtype=float,
                      index=pd.to_datetime([row[0] for row in rows], unit="ms", utc=True))
    df.index.name = "date_open"
    df["date_close"] = pd.to_datetime([row[6] for row in rows], unit="ms", utc=True)
    return df


def load_raw_data(data_dir="data", train_start=TRAIN_START, warmup_days=WARMUP_DAYS):
    """Return validated raw OHLCV candles from (train_start - warm-up) to EVAL_END.

    train_start defaults to the course's training year; an earlier start gives
    more history (the project brief allows changing the training range as long
    as training ends before the eval window). Uses a dated pickle cache (same
    file name pattern as the notebook), so each range is downloaded only once.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    download_start = train_start - pd.Timedelta(days=warmup_days)
    data_path = data_dir / f"binance-BTCUSDT-1h-{download_start:%Y%m%d}-{EVAL_END:%Y%m%d}.pkl"

    cached = data_path.exists()
    df = pd.read_pickle(data_path) if cached else download_klines(download_start, EVAL_END)

    # Validate before caching, so a broken download is never saved.
    validate_hourly(df, download_start, EVAL_END, "Downloaded data")
    if not cached:
        _save_atomic(df, data_path)
    return df


def _save_atomic(df, path):
    """Write to a temporary file, then rename: parallel runs never read a half-written cache."""
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    df.to_pickle(tmp)
    os.replace(tmp, path)


# First hourly candle of BTCUSDT on Binance
HISTORY_START = pd.Timestamp("2017-08-17", tz="UTC")


def load_history(data_dir="data", start=HISTORY_START):
    """All BTCUSDT hourly candles from `start` to EVAL_END, including the few
    hours missing because of exchange outages (128 hours before 2023-03-25).

    Same checks as validate_hourly except completeness; missing hours are never
    filled in, the history is split into gap-free segments instead.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / f"binance-BTCUSDT-1h-{start:%Y%m%d}-{EVAL_END:%Y%m%d}-with-gaps.pkl"
    cached = data_path.exists()
    df = pd.read_pickle(data_path) if cached else download_klines(start, EVAL_END)
    values = df[OHLCV]
    if (df.index.has_duplicates or not df.index.is_monotonic_increasing
            or not np.isfinite(values.to_numpy()).all()
            or (values[["open", "high", "low", "close"]] <= 0).any().any()
            or (values["high"] < values[["open", "low", "close"]].max(axis=1)).any()
            or (values["low"] > values[["open", "high", "close"]].min(axis=1)).any()):
        raise ValueError("history: invalid candles")
    if not cached:
        _save_atomic(df, data_path)
    return df


def gap_free_segments(df, end):
    """Split candles before `end` into maximal runs of consecutive hours."""
    df = df[df.index < end]
    breaks = np.flatnonzero(np.diff(df.index.asi8) != 3_600 * 10**9)
    bounds = zip(np.concatenate([[0], breaks + 1]), np.concatenate([breaks + 1, [len(df)]]))
    return [df.iloc[a:b].copy() for a, b in bounds]


# ---------------------------------------------------------------------------
# Features and splits
# ---------------------------------------------------------------------------

def add_default_features(df):
    """Add the 5 default features of the instruction notebook.

    Every column whose name contains "feature" becomes part of the observation.
    All features are causal: the value at candle t uses only candles <= t.
    """
    df = df.copy()
    df["feature_close"] = df["close"].pct_change()           # 1h return: close_t / close_(t-1) - 1
    df["feature_open"] = df["open"] / df["close"]            # candle shape relative to close
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]
    df["feature_volume"] = df["volume"] / df["volume"].rolling(7 * 24).max()  # volume vs 7-day max
    # The first rows have undefined features (no previous close, incomplete 168h window).
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def fit_normalizer(train_df):
    """z-score statistics (mean, std) of every feature column, fitted on training rows only."""
    cols = [c for c in train_df.columns if "feature" in c]
    return {c: [float(train_df[c].mean()), float(train_df[c].std())] for c in cols}  # lists: same as in JSON


def apply_normalizer(df, stats):
    """x' = (x - mean) / std with the training statistics (no look-ahead into val/eval)."""
    df = df.copy()
    for c, (mean, std) in stats.items():
        df[c] = (df[c] - mean) / std
    return df


def add_mean_reversion_features(df):
    """Default features without the duplicate feature_open, plus three short-term
    mean-reversion signals (chosen by their rank correlation with the next-hour
    return on the 3-year dev-train, see notebook 03, section 3).

    All causal: every window ends at the current candle.
    """
    df = df.copy()
    log_close = np.log(df["close"])
    df["feature_ret_4h"] = log_close.diff(4)                                        # 4-hour log return
    df["feature_dist_sma_24h"] = df["close"] / df["close"].rolling(24).mean() - 1   # distance to 24h average
    step = log_close.diff()
    gain = step.clip(lower=0).rolling(14).mean()
    loss = (-step.clip(upper=0)).rolling(14).mean()
    df["feature_rsi_14h"] = gain / (gain + loss)                                    # RSI in [0, 1]
    # default features last: their dropna removes the warm-up rows of all features at once
    return add_default_features(df).drop(columns=["feature_open"])  # open_t = close_(t-1): duplicate


# Feature pipelines selectable by name in the experiment configs
FEATURE_SETS = {
    "default": add_default_features,
    "mean_reversion": add_mean_reversion_features,
}


def split_data(df, train_start=TRAIN_START):
    """Split a featured dataframe into dev-train, validation and eval.

    dev_train: train_start to VAL_START  (training during development)
    val:       VAL_START to EVAL_START   (model selection)
    eval:      EVAL_START to EVAL_END    (fixed test window, touched twice in total)
    Each split is validated to contain every hour of its range.
    """
    ranges = {
        "dev_train": (train_start, VAL_START),
        "val": (VAL_START, EVAL_START),
        "eval": (EVAL_START, EVAL_END),
    }
    splits = {}
    for name, (start, end) in ranges.items():
        part = df.loc[(df.index >= start) & (df.index < end)].copy()
        validate_hourly(part, start, end, name)
        splits[name] = part
    assert splits["dev_train"].index.max() < splits["val"].index.min(), "dev_train/val overlap"
    assert splits["val"].index.max() < splits["eval"].index.min(), "val/eval overlap"
    return splits
