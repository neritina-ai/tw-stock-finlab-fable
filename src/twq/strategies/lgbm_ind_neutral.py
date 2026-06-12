"""lgbm-ind-neutral: industry-relative ranking over the all4 panel ensemble.

Attempt 9 (branch ``ind-neutral``, parent ``param-ensemble``). Attempt 8 closed
the selection layer: equal-weight all4 + b4/s4/top30 is the honest ceiling of
the family, and stages 1/3 are alpha-limited (ceilings 1.62 / 1.24-1.30).
This module raises the ceiling itself by removing industry-beta bets from the
score: each panel's score is ranked *within its TWSE industry* per week, so
the ensemble rewards best-in-industry stock picking instead of crowding into
whichever industry the models currently like (the realized books show 金融牆 /
electronics walls doing exactly that).

- Industry map: data/finlab/security_categories.parquet ``category`` (today's
  snapshot; membership is sticky so the PIT error is small — noted, standard
  practice). ETF/DR rows carry no panel scores, so no exclusion is needed.
- Per week, an industry with fewer than MIN_IND scored names — and any name
  missing from the snapshot (mostly pre-2026 delistings) — falls back to its
  global cross-sectional rank: degrade to the baseline, never to a 2-name
  derby.
- mode "indsz" additionally residualizes the combined score against
  ln(20-day mean dollar turnover) cross-sectionally per week, then re-ranks.
  Pre-registration said ln(market_value), but the free tier serves market_value
  only through 2018-12-28; trailing dollar turnover is the standard liquid-size
  proxy available over the whole window (shifted one day, no lookahead).

Era splice (2018-12-28), banding, sparse emission and taiex_half exposure are
inherited unchanged from lgbm_h63_ensemble via its base cache (mix "indn_*").
"""

import numpy as np
import pandas as pd

from .. import data as D
from . import lgbm_h63_ensemble as E

MIN_IND = 5  # scored names an industry needs that week to rank within itself

_LOCAL: dict = {}


def _bucket_map(cols: pd.Index) -> pd.Series:
    if "buckets" not in _LOCAL:
        sc = pd.read_parquet("data/finlab/security_categories.parquet")
        _LOCAL["buckets"] = sc.set_index("stock_id")["category"].reindex(cols)
    return _LOCAL["buckets"]


def _ind_rank(name: str, cols: pd.Index) -> pd.DataFrame:
    """Within-industry pct rank per week, global-rank fallback for thin weeks."""
    key = f"indrank_{name}"
    if key not in _LOCAL:
        ST = D.load(name).reindex(columns=cols).T  # names x weeks
        g = ST.groupby(_bucket_map(cols))
        r_ind = g.rank(pct=True)
        cnt = g.transform("count")  # scored names in the bucket that week
        r_glob = ST.rank(pct=True)
        # unmapped names have NaN bucket -> NaN cnt -> condition False -> global
        _LOCAL[key] = r_ind.where(cnt >= MIN_IND, r_glob).T
    return _LOCAL[key]


def _size_z(cols: pd.Index) -> pd.DataFrame:
    if "size_z" not in _LOCAL:
        amt = D.load("turnover").reindex(columns=cols)
        adv = amt.rolling(20, min_periods=5).mean().shift(1)
        _LOCAL["size_z"] = np.log(adv.where(adv > 0))
    return _LOCAL["size_z"]


def _size_residual(score: pd.DataFrame, cols: pd.Index) -> pd.DataFrame:
    """Weekly cross-sectional OLS residual of score vs ln(size), re-ranked."""
    M = score.notna()
    Z = _size_z(cols).reindex(index=score.index).where(M)
    # scored names missing a size estimate count as average-size that week
    Z = Z.T.fillna(Z.mean(axis=1)).T.where(M)
    zc = Z.sub(Z.mean(axis=1), axis=0)
    sc = score.sub(score.mean(axis=1), axis=0).where(M)
    var = zc.pow(2).sum(axis=1)
    beta = ((sc * zc).sum(axis=1) / var.where(var > 0)).fillna(0.0)
    resid = sc - zc.mul(beta, axis=0)
    return resid.rank(axis=1, pct=True)


def _base(mode: str):
    cols = D.load("lgbm_scores_h21_nomkt").columns
    pre_names, post_names = E.MIXES["all4"]
    pre = E._mean([_ind_rank(n, cols) for n in pre_names])
    post = E._mean([_ind_rank(n, cols) for n in post_names])
    score = pd.concat([pre.loc[: E.SPLICE], post.loc[post.index > E.SPLICE]])
    if mode == "indsz":
        score = _size_residual(score, cols)

    days = D.trading_days()
    taiex = D.load("twse_index")["close"].reindex(days).ffill()
    risk_taiex = (taiex > taiex.ewm(span=120).mean()).shift(1).fillna(True).astype(bool)
    regimes = {
        "taiex": (risk_taiex, 0.0),
        "taiex_half": (risk_taiex, 0.5),
        "none": (None, 1.0),
    }
    return days, score, regimes


def make_weights(params: dict, start: str, end: str) -> pd.DataFrame:
    mode = params.get("neutral", "ind")
    mix = f"indn_{mode}"  # cache-key only; never resolved through E.MIXES
    if f"base_{mix}" not in E._CACHE:
        E._CACHE[f"base_{mix}"] = _base(mode)
    p = {k: v for k, v in params.items() if k != "neutral"}
    return E.make_weights(dict(p, mix=mix), start, end)
