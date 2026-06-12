"""lgbm-rank-v1: rolling LightGBM cross-sectional scores + banded rotation.

Attempt 3 (branch ``lgbm-rank-v1``, parent ``band-voltarget``). Replaces the
hand-crafted defensive blend (hindsight ceiling ~1.8 in attempt 2) with
point-in-time LightGBM scores built by scripts/train_lgbm_scores.py, and
reuses the turnover mechanism attempt 2 validated:

- Weekly decisions; held names kept while inside the top band_mult*N of the
  hold score; vacancies filled by the highest-scored non-held names.
- Hold score = EMA of the score panel over `hold_smooth` weekly rows (a
  smoothed score churns less, replacing attempt 2's slow_keep factor split).
- Regime gate: "taiex" (TAIEX > EMA120, the OOS-robust gate) or "none".
  breadth (train-overfit) and vol targeting (never selected) are dropped.
- Sparse emission: trade only when membership changes, the gate flips, or 8
  weeks have passed (re-equalize drift).

The score panel rows are weekly trade days and are ALREADY aligned (features
through d-1 scoring trades at d's open) — no further shift here. Universe
fixes from attempt 2: 00-prefix ETFs excluded (see twq.universe).
"""

import pandas as pd

from .. import data as D

_CACHE: dict = {}


def _base():
    score = D.load("lgbm_scores_h21")
    days = D.trading_days()
    taiex = D.load("twse_index")["close"].reindex(days).ffill()
    risk_taiex = (taiex > taiex.ewm(span=120).mean()).shift(1).fillna(True).astype(bool)
    return days, score, {"taiex": risk_taiex, "none": None}


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
    risk_on = regimes.get(params.get("regime"))

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
            scale = 0.0

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
