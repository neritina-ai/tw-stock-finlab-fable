"""lgbm-fund-splice: fundamental/chip-enhanced ranking through 2018, price-only after.

Attempt 5 (branch ``fund-splice``, parent ``lgbm-stockonly``). Finlab's free
tier serves HISTORY ONLY: monthly revenue from 2005, margin balances from
2009, valuation ratios from 2010, institutional flows from 2012-05 — all
ending 2018-12-28. That coverage sits exactly on the weak WFV stages
(price-only selection ceilings: 2014-15 → 0.99, 2018-19 → 1.09), so the score
panel is spliced from two strictly-OOS panels:

- dates <= 2018-12-28: ``lgbm_scores_h21_fund`` (26 stock + 13 fund features)
- dates after:          ``lgbm_scores_h21_nomkt`` (attempt 4, price-only)

Each panel is converted to per-day cross-sectional rank-pct before splicing so
the EWM hold score sees one continuous scale across the boundary. Exposure is
FIXED at taiex_half (attempt 4: avg -0.14 from per-stage ceiling, never the
worst regime), removing the regime axis from WFV selection; the turnover
penalty is dropped too (it over-blunted stage 1 last time). Banding + sparse
emission unchanged from attempts 2-4.

Panel rows are weekly trade days, already aligned (features through d-1
scoring trades at d's open) — no further shift here.
"""

import pandas as pd

from .. import data as D

SPLICE = pd.Timestamp("2018-12-28")  # last date served by the finlab free tier

_CACHE: dict = {}


def _base():
    nomkt = D.load("lgbm_scores_h21_nomkt")
    fund = D.load("lgbm_scores_h21_fund").reindex(columns=nomkt.columns)
    nr = nomkt.rank(axis=1, pct=True)
    fr = fund.rank(axis=1, pct=True)
    score = pd.concat([fr.loc[:SPLICE], nr.loc[nr.index > SPLICE]])

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
    if "base" not in _CACHE:
        _CACHE["base"] = _base()
    days, score, regimes = _CACHE["base"]

    hs = int(params.get("hold_smooth", 1))
    hkey = f"hold{hs}"
    if hkey not in _CACHE:
        # ewm carries values across NaN rows; re-mask so a name that left the
        # universe (NaN score) cannot stay in the band on a stale hold score
        _CACHE[hkey] = (
            score.ewm(span=hs, min_periods=1).mean().where(score.notna())
            if hs > 1
            else score
        )
    hold_score = _CACHE[hkey]

    top_n = params.get("top_n", 30)
    band = int(top_n * params["band_mult"])
    risk_on, off_scale = regimes[params.get("regime", "taiex_half")]

    rebal = score.index
    rebal = rebal[(rebal >= pd.Timestamp(start)) & (rebal <= pd.Timestamp(end))]

    held: list[str] = []
    last_scale = -1.0
    weeks_since = 0
    rows: dict[pd.Timestamp, dict[str, float]] = {}

    for d in rebal:
        s = score.loc[d].dropna().sort_values(ascending=False)
        if s.empty:
            continue
        ks = hold_score.loc[d].dropna().sort_values(ascending=False)
        band_set = set(ks.index[:band])
        keep = [x for x in held if x in band_set]
        adds = [x for x in s.index if x not in keep][: top_n - len(keep)]
        new_held = keep + adds

        scale = 1.0
        if risk_on is not None and not risk_on.loc[d]:
            scale = off_scale

        changed = set(new_held) != set(held)
        weeks_since += 1
        if changed or abs(scale - last_scale) > 0.05 or weeks_since >= 8:
            if scale > 0 and new_held:
                w = scale / len(new_held)
                rows[d] = {x: w for x in new_held}
            elif held:  # liquidate (any finite row rebalances; NaN cells = 0)
                rows[d] = {held[0]: 0.0}
            last_scale = scale
            weeks_since = 0
        held = new_held

    W = pd.DataFrame.from_dict(rows, orient="index")
    out_days = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
    return W.reindex(index=out_days, columns=D.load("close").columns)
