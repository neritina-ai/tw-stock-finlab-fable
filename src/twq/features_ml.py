"""Price/volume feature matrix + weekly sample table for ML ranking.

Finlab free-tier fundamentals truncate at 2018-12-28, so every feature here is
derived from the local OHLCV/turnover matrices (2007-04 onward) to keep the
full WFV period (2008-2026) validatable.

Alignment convention: one sample per (weekly trade day d, stock). Features are
computed at the previous trading day t = d-1 (the last close known when
trading at d's open) but stored indexed by d — i.e. the table and the score
panels built from it are ALREADY shifted; consumers must not shift again.

Label: y_raw = adj_open(d+21) / adj_open(d) - 1, exactly the window a trade
entered at d's open can capture (no overnight-gap leak from using closes).
y_rank is its cross-sectional percentile among eligible names at d — every
date has mean 0.5, so the per-day-constant market features can only matter
through interactions with stock features, never as level shifts.
y_end is the actual date of open(d+21), used for exact embargo when training.

Stock features are percentile-ranked within the eligible set per day (matches
deployment: we only ever order eligible candidates). Market features are raw.
"""

import numpy as np
import pandas as pd

from . import data as D
from . import factors as F
from . import universe as U

HORIZON = 21  # forward trading days for the label
CACHE = D.DERIVED / "ml_weekly_h21.parquet"

MARKET_FEATS = ["mkt_mom63", "mkt_vol21", "mkt_dd"]


def _stock_panels() -> dict[str, pd.DataFrame]:
    """Raw (unshifted, info through t) daily feature matrices."""
    adj = D.adj_ohlc()
    ac, ao, ah, al = adj["close"], adj["open"], adj["high"], adj["low"]
    raw_to = D.load("turnover")
    vol = D.load("volume")
    r = ac.pct_change(fill_method=None)

    to = raw_to.where(raw_to > 0)
    rng = ((ah - al) / ac).where(ah >= al)
    hl = (ah - al).where((ah - al) > 0)
    clv = ((ac - al) / hl).clip(0, 1)

    f = {
        # momentum / trend
        "ret5": F.momentum(ac, 5),
        "ret21": F.momentum(ac, 21),
        "ret63": F.momentum(ac, 63),
        "ret126": F.momentum(ac, 126),
        "mom_12_1": F.momentum(ac, 252, skip=21),
        "nh52": F.near_high(ac, 252),
        "nl52": ac / ac.rolling(252, min_periods=126).min(),
        "magap20": ac / ac.rolling(20, min_periods=10).mean() - 1,
        "magap60": ac / ac.rolling(60, min_periods=30).mean() - 1,
        "magap120": ac / ac.rolling(120, min_periods=60).mean() - 1,
        "tstat63": r.rolling(63, min_periods=40).mean()
        / r.rolling(63, min_periods=40).std(),
        # risk / distribution
        "vol21": r.rolling(21, min_periods=15).std(),
        "vol63": r.rolling(63, min_periods=40).std(),
        "dvol63": F.downside_vol(ac, 63),
        "skew63": r.rolling(63, min_periods=40).skew(),
        "max1_21": r.rolling(21, min_periods=15).max(),
        "min1_21": r.rolling(21, min_periods=15).min(),
        "rng21": rng.rolling(21, min_periods=15).mean(),
        # liquidity / volume
        "amihud21": (r.abs() / to).rolling(21, min_periods=15).mean(),
        "ldv63": np.log(to.rolling(63, min_periods=30).median()),
        "tratio": to.rolling(21, min_periods=10).median()
        / to.rolling(126, min_periods=60).median(),
        "volcv21": vol.rolling(21, min_periods=15).std()
        / vol.rolling(21, min_periods=15).mean(),
        # microstructure-ish
        "clv21": clv.rolling(21, min_periods=15).mean(),
        "ogap21": (ao / ac.shift(1) - 1).rolling(21, min_periods=15).mean(),
        "ups63": (r > 0).rolling(63, min_periods=40).mean(),
    }

    # rolling beta to TAIEX (cov/var via rolling moments, vectorized)
    mkt = D.load("twse_index")["close"].reindex(ac.index).ffill()
    rm = mkt.pct_change(fill_method=None)
    rm_mat = pd.DataFrame(
        np.broadcast_to(rm.to_numpy()[:, None], ac.shape), index=ac.index, columns=ac.columns
    )
    exy = (r * rm_mat).rolling(63, min_periods=40).mean()
    ex = r.rolling(63, min_periods=40).mean()
    ey = rm.rolling(63, min_periods=40).mean()
    varm = rm.rolling(63, min_periods=40).var(ddof=0)
    f["beta63"] = (exy - ex.mul(ey, axis=0)).div(varm.replace(0, np.nan), axis=0)
    return f


def _market_feats(days: pd.DatetimeIndex) -> pd.DataFrame:
    mkt = D.load("twse_index")["close"].reindex(days).ffill()
    rm = mkt.pct_change(fill_method=None)
    return pd.DataFrame(
        {
            "mkt_mom63": mkt / mkt.shift(63) - 1,
            "mkt_vol21": rm.rolling(21, min_periods=15).std() * np.sqrt(252),
            "mkt_dd": mkt / mkt.rolling(252, min_periods=126).max() - 1,
        },
        index=days,
    )


def build_weekly_table(refresh: bool = False) -> pd.DataFrame:
    """Long table: one row per (weekly trade day, eligible stock).

    Columns: date, stock, <ranked stock feats>, <raw market feats>, y_rank,
    y_end. Cached to data/derived/ml_weekly_h21.parquet.
    """
    if CACHE.exists() and not refresh:
        return pd.read_parquet(CACHE)

    days = D.trading_days()
    ws = F.week_starts(days).sort_values()
    pos = days.get_indexer(ws)
    ws, pos = ws[pos > 0], pos[pos > 0]  # need a prior day for features
    tpos = pos - 1

    elig = U.eligible()  # row d = tradable at d's open (already shifted)
    E = elig.iloc[pos]
    E.index = ws

    panels = _stock_panels()
    cols = E.columns

    # label at d: open(d+21)/open(d)-1, ranked among eligible at d
    ao = D.adj_ohlc()["open"]
    y_raw = (ao.shift(-HORIZON) / ao - 1).iloc[pos]
    y_raw.index = ws
    y_rank = y_raw.where(E).rank(axis=1, pct=True)

    end_pos = pos + HORIZON
    y_end = pd.Series(
        [days[p] if p < len(days) else pd.NaT for p in end_pos], index=ws
    )

    ev = E.to_numpy()
    rix, cix = np.where(ev)
    out = {
        "date": ws.to_numpy()[rix],
        "stock": cols.to_numpy()[cix],
    }
    for name, m in panels.items():
        feat_d = m.iloc[tpos]  # info through d-1
        feat_d.index = ws
        ranked = feat_d.where(E).rank(axis=1, pct=True)
        out[name] = ranked.to_numpy(dtype="float32")[rix, cix]
    mf = _market_feats(days).iloc[tpos]
    for name in MARKET_FEATS:
        out[name] = mf[name].to_numpy(dtype="float32")[rix]
    out["y_rank"] = y_rank.to_numpy(dtype="float32")[rix, cix]
    out["y_end"] = y_end.to_numpy()[rix]

    table = pd.DataFrame(out)
    table.to_parquet(CACHE, index=False)
    return table


def feature_names(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c not in ("date", "stock", "y_rank", "y_end")]
