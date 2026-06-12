"""Cross-sectional factor utilities. All inputs are (date x stock) matrices."""

import numpy as np
import pandas as pd


def zscore_xs(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Cross-sectional z-score per row, clipped."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1)
    z = df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
    return z.clip(-clip, clip)


def rank_xs(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank in [0, 1] per row (NaN preserved)."""
    return df.rank(axis=1, pct=True)


def momentum(adj_close: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    return adj_close.shift(skip) / adj_close.shift(lookback) - 1


def near_high(adj_close: pd.DataFrame, win: int = 252) -> pd.DataFrame:
    return adj_close / adj_close.rolling(win, min_periods=win // 2).max()


def volatility(adj_close: pd.DataFrame, win: int = 60) -> pd.DataFrame:
    return adj_close.pct_change(fill_method=None).rolling(win, min_periods=win // 2).std()


def downside_vol(adj_close: pd.DataFrame, win: int = 60) -> pd.DataFrame:
    r = adj_close.pct_change(fill_method=None)
    return r.where(r < 0).rolling(win, min_periods=win // 4).std()


def month_starts(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First trading day of each month."""
    s = pd.Series(days, index=days)
    return pd.DatetimeIndex(s.groupby([days.year, days.month]).first())


def week_starts(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(days, index=days)
    iso = days.isocalendar()
    return pd.DatetimeIndex(s.groupby([iso.year.to_numpy(), iso.week.to_numpy()]).first())


def top_n_weights(score: pd.DataFrame, n: int, eligible: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight the top-n by score among eligible names, per row."""
    sc = score.where(eligible)
    ranks = sc.rank(axis=1, ascending=False, method="first")
    sel = ranks <= n
    cnt = sel.sum(axis=1).replace(0, np.nan)
    return sel.div(cnt, axis=0).fillna(0.0)
