"""lgbm-reversal-blend: blend industry-neutral short-horizon reversal into all4.

Attempt 10 (branch ``reversal-blend``, parent ``ind-neutral``). Attempt 9
established that the chop-window weakness (stages 1/3) is an *absolute*
trend-alpha shortage, not a concentration artefact — reorganizing the existing
LGBM signals (selection, weights, industry neutralization) is triple-falsified,
so a complementary alpha SOURCE is needed. Short-horizon reversal is the
canonical chop-market factor and is naturally industry-neutral, which lets it
reuse the ``lgbm_ind_neutral`` within-industry-ranking machinery.

Pre-registered gate (research/ind-neutral.md "下一步"): the standalone weekly
IC of the industry-neutral reversal vs the 21d forward open->open return must
be positive in the 2014-15 AND 2018-19 chop subwindows before any WFV is run
(scripts/rev_ic_check.py). Outcome: 2018-19 is strongly positive
(revC_ind +0.027, t=3.2) but **2014-15 is flat-to-negative** (-0.002, t=-0.2)
— the gate fails in the very window it was meant to rescue, and because the
reversal IC (~0.01) is an order of magnitude below all4 (~0.11), every blend
column dilutes all4's IC in every window. This module is kept to quantify the
*realized* cost of blending anyway; the WFV confirms the dilution, so the
attempt is a negative result (champion remains all4 plain, 2/6, avg 1.749).

Construction
------------
revC = mean of the within-TWSE-industry pct ranks of -ret5d and -ret10d
(adjusted close through d-1, ``.shift(1)`` no lookahead), masked to the
all4-scored universe per week; industries with < MIN_IND scored names fall
back to global rank exactly as in lgbm_ind_neutral. The blended score is

    (1 - w) * all4_rankmean + w * revC          (revC missing -> neutral 0.5)

re-masked to the all4 universe. Era splice (2018-12-28), banding, sparse
emission and taiex_half exposure are inherited unchanged from
lgbm_h63_ensemble via its base cache (mix ``revb_<w>``). ``rev_w`` in
{0.25, 0.5} is the only new hyperparameter (registered before any WFV run).
"""

import pandas as pd

from .. import data as D
from . import lgbm_h63_ensemble as E
from . import lgbm_ind_neutral as IN


def _rev_ind(cols: pd.Index, mask: pd.DataFrame) -> pd.DataFrame:
    """Within-industry pct rank of combined -ret5d/-ret10d, all4 universe only."""
    ac = D.adj_ohlc()["close"].reindex(columns=cols)
    parts = []
    for lag in (5, 10):
        rev = (-(ac / ac.shift(lag) - 1).shift(1)).reindex(index=mask.index).where(mask)
        st = rev.T  # names x weeks
        g = st.groupby(IN._bucket_map(cols))
        r_ind = g.rank(pct=True)
        cnt = g.transform("count")
        parts.append(r_ind.where(cnt >= IN.MIN_IND, st.rank(pct=True)).T)
    return (parts[0] + parts[1]) / 2


def _base(w: float):
    cols = D.load("lgbm_scores_h21_nomkt").columns
    pre_names, post_names = E.MIXES["all4"]
    pre = E._mean([E._rank(n, cols) for n in pre_names])
    post = E._mean([E._rank(n, cols) for n in post_names])
    all4 = pd.concat([pre.loc[: E.SPLICE], post.loc[post.index > E.SPLICE]])
    mask = all4.notna()
    rev = _rev_ind(cols, mask)
    score = ((1 - w) * all4 + w * rev.fillna(0.5)).where(mask)

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
    w = float(params.get("rev_w", 0.25))
    mix = f"revb_{w}"  # cache-key only; never resolved through E.MIXES
    if f"base_{mix}" not in E._CACHE:
        E._CACHE[f"base_{mix}"] = _base(w)
    p = {k: v for k, v in params.items() if k != "rev_w"}
    return E.make_weights(dict(p, mix=mix), start, end)
