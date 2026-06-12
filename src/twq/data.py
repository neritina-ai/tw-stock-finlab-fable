"""Data loading utilities: wide price matrices, adjusted prices, universe."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL = ROOT / "data" / "local"
FINLAB = ROOT / "data" / "finlab"
ACTIONS = ROOT / "data" / "actions"
DERIVED = ROOT / "data" / "derived"

# Taiwan market round-trip transaction costs (user-specified)
BUY_COST = 0.001425
SELL_COST = 0.001425 + 0.003


@lru_cache(maxsize=None)
def load(name: str) -> pd.DataFrame:
    """Load a wide matrix by short name from local/derived/finlab dirs."""
    for d in (LOCAL, DERIVED, FINLAB):
        p = d / f"{name}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if df.index.dtype == object and str(df.index[0])[:2] == "20":
                df.index = pd.to_datetime(df.index)
            return df
    raise FileNotFoundError(name)


def trading_days() -> pd.DatetimeIndex:
    return load("close").index


def align(df: pd.DataFrame, like: pd.DataFrame | None = None) -> pd.DataFrame:
    """Reindex a matrix to the master close grid (or `like`), ffill-free."""
    ref = like if like is not None else load("close")
    return df.reindex(index=ref.index, columns=ref.columns)


def quarterly_to_daily(qdf: pd.DataFrame, lag_days: dict | None = None) -> pd.DataFrame:
    """Map a quarter-indexed matrix ('2013-Q1' style) onto the daily grid,
    becoming visible only after statutory publication deadlines.

    Q1 -> May 15, Q2 -> Aug 14, Q3 -> Nov 14, Q4 -> Mar 31 of next year.
    """
    pub = {1: (0, 5, 15), 2: (0, 8, 14), 3: (0, 11, 14), 4: (1, 3, 31)}
    rows = []
    for idx, row in qdf.iterrows():
        y, q = str(idx).split("-Q")
        dy, m, d = pub[int(q)]
        rows.append((pd.Timestamp(int(y) + dy, m, d), row))
    out = pd.DataFrame({ts: r for ts, r in rows}).T.sort_index()
    days = trading_days()
    out = out.reindex(days.union(out.index)).ffill().reindex(days)
    return out.reindex(columns=load("close").columns)


def monthly_to_daily(mdf: pd.DataFrame, pub_day: int = 10) -> pd.DataFrame:
    """Map a month-indexed (or revenue-dated) matrix onto the daily grid.

    Finlab monthly_revenue index is the revenue month's 10th of next month
    (already publication-shifted). We additionally ffill onto trading days.
    """
    mdf = mdf.copy()
    mdf.index = pd.to_datetime(mdf.index)
    days = trading_days()
    out = mdf.reindex(days.union(mdf.index)).ffill().reindex(days)
    return out.reindex(columns=load("close").columns)


def adj_ohlc() -> dict[str, pd.DataFrame]:
    """Adjusted open/high/low/close built by scripts/build_adjustment.py."""
    factor = load("adj_factor")  # cumulative, anchored 1.0 at the last day
    out = {}
    for f in ("open", "high", "low", "close"):
        out[f] = load(f) * factor
    return out


def prev_close_raw() -> pd.DataFrame:
    return load("close").shift(1)


def limit_pct(dates: pd.DatetimeIndex) -> np.ndarray:
    """Daily price limit: 7% before 2015-06-01, 10% after."""
    return np.where(dates < pd.Timestamp("2015-06-01"), 0.07, 0.10)


def default_universe(min_turnover_ntd: float = 10e6, win: int = 20) -> pd.DataFrame:
    """Liquidity universe: rolling-median daily turnover above threshold.

    Shifted by one day so membership at t uses info through t-1.
    """
    to = load("turnover")
    med = to.rolling(win, min_periods=win // 2).median()
    return (med > min_turnover_ntd).shift(1).fillna(False).astype(bool)
