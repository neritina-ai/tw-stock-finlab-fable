"""px-composite-v1: monthly cross-sectional price/volume composite, long-only.

Signals (computed strictly from data through t-1, emitted on monthly rebalance
days at whose open we trade):

- mom:    126d momentum skipping the last 10d
- nh52:   proximity to 52-week high
- rev5:   5-day short-term reversal (negative recent runup preferred)
- lvol:   low 60d volatility preference

Composite presets blend cross-sectional ranks. Universe: liquid common stocks
(median 20d turnover above threshold, price >= 10, listed >= 130d, TDRs
excluded). Optional market-regime scaling by TAIEX vs its 120d EMA.
"""

import numpy as np
import pandas as pd

from .. import data as D
from .. import factors as F

PRESETS = {
    "mom_tilt": {"mom": 0.5, "nh52": 0.5},
    "defensive": {"nh52": 0.4, "rev5": 0.3, "lvol": 0.3},
    "all4": {"mom": 0.25, "nh52": 0.25, "rev5": 0.25, "lvol": 0.25},
}


def _base():
    adj_close = D.load("close") * D.load("adj_factor")
    days = adj_close.index
    raw_close = D.load("close")
    turnover = D.load("turnover")

    feats = {
        "mom": F.momentum(adj_close, 126, 10),
        "nh52": F.near_high(adj_close, 252),
        "rev5": -F.momentum(adj_close, 5, 0),
        "lvol": -F.volatility(adj_close, 60),
    }
    listed_days = raw_close.notna().rolling(130).sum()
    tdr = raw_close.columns.str.startswith("91")
    uni = (
        (turnover.rolling(20, min_periods=10).median() > 20e6)
        & (raw_close > 10)
        & (listed_days >= 100)
    )
    uni.loc[:, tdr] = False

    taiex = D.load("twse_index")["close"].reindex(days).ffill()
    risk_on = taiex > taiex.ewm(span=120).mean()
    return adj_close, days, feats, uni, risk_on


_CACHE: dict = {}


def make_weights(params: dict, start: str, end: str) -> pd.DataFrame:
    if "base" not in _CACHE:
        _CACHE["base"] = _base()
    adj_close, days, feats, uni, risk_on = _CACHE["base"]

    blend = PRESETS[params["preset"]]
    score = None
    for k, w in blend.items():
        r = F.rank_xs(feats[k]) * w
        score = r if score is None else score.add(r, fill_value=0)

    # info through t-1 for trade at t's open
    score = score.shift(1)
    eligible = uni.shift(1).fillna(False).astype(bool)

    rebal_days = F.month_starts(days)
    rebal_days = rebal_days[(rebal_days >= pd.Timestamp(start)) & (rebal_days <= pd.Timestamp(end))]

    W = F.top_n_weights(score.loc[rebal_days], params["top_n"], eligible.loc[rebal_days])

    scale = pd.Series(1.0, index=rebal_days)
    if params.get("risk_off_scale", 1.0) < 1.0:
        on = risk_on.shift(1).reindex(rebal_days).fillna(True)
        scale[~on] = params["risk_off_scale"]
    W = W.mul(scale, axis=0)

    return W.reindex(days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))])
