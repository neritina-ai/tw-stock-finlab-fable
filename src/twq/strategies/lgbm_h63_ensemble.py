"""lgbm-h63-ensemble: add a slow (63-day-label) ranker, ensembled by rank mean.

Attempt 6 (branch ``h63-ensemble``, parent ``fund-splice``). Same dual-era
splice as fund-splice (fund-feature panels meaningful through 2018-12-28,
price-only after), but the per-era score is an equal-weight mean of per-day
cross-sectional rank-pcts over several strictly-OOS panels:

- mix "h63":     pre  mean(rank(fund_h21),  rank(fund_h63))
                 post mean(rank(nomkt_h21), rank(nomkt_h63))
  The h63 label (open-to-open 63-trading-day forward return) retrains the same
  features toward slower alpha; a rank mean is the cheapest robust ensemble
  and should also calm turnover.
- mix "blend21": pre  mean(rank(fund_h21), rank(nomkt_h21));  post rank(nomkt_h21)
  Zero-retrain A/B aimed at stage 2 (2016-17), where REPLACING nomkt with the
  fund model cost -0.24 — blending keeps both opinions.
- mix "all4":    pre  mean of all four ranks; post as "h63"

Exposure fixed at taiex_half, plain-Sharpe selection, banding + sparse
emission unchanged from attempts 2-5. Panel rows are weekly trade days already
aligned (features through d-1 score trades at d's open) — no further shift.
"""

import pandas as pd

from .. import data as D

SPLICE = pd.Timestamp("2018-12-28")  # last date served by the finlab free tier

MIXES = {
    "h63": (
        ["lgbm_scores_h21_fund", "lgbm_scores_h63_fund"],
        ["lgbm_scores_h21_nomkt", "lgbm_scores_h63_nomkt"],
    ),
    "blend21": (
        ["lgbm_scores_h21_fund", "lgbm_scores_h21_nomkt"],
        ["lgbm_scores_h21_nomkt"],
    ),
    "all4": (
        [
            "lgbm_scores_h21_fund",
            "lgbm_scores_h63_fund",
            "lgbm_scores_h21_nomkt",
            "lgbm_scores_h63_nomkt",
        ],
        ["lgbm_scores_h21_nomkt", "lgbm_scores_h63_nomkt"],
    ),
}

_CACHE: dict = {}


def _rank(name: str, cols: pd.Index) -> pd.DataFrame:
    key = f"rank_{name}"
    if key not in _CACHE:
        _CACHE[key] = D.load(name).reindex(columns=cols).rank(axis=1, pct=True)
    return _CACHE[key]


def _mean(panels: list[pd.DataFrame]) -> pd.DataFrame:
    # NaN in any panel keeps the name out that week (panels share row coverage,
    # so this only bites names one model genuinely didn't score)
    out = panels[0]
    for p in panels[1:]:
        out = out + p
    return out / len(panels)


def _base(mix: str):
    cols = D.load("lgbm_scores_h21_nomkt").columns
    pre_names, post_names = MIXES[mix]
    pre = _mean([_rank(n, cols) for n in pre_names])
    post = _mean([_rank(n, cols) for n in post_names])
    score = pd.concat([pre.loc[:SPLICE], post.loc[post.index > SPLICE]])

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
    mix = params.get("mix", "h63")
    bkey = f"base_{mix}"
    if bkey not in _CACHE:
        _CACHE[bkey] = _base(mix)
    days, score, regimes = _CACHE[bkey]

    hs = int(params.get("hold_smooth", 1))
    hkey = f"hold{hs}_{mix}"
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
