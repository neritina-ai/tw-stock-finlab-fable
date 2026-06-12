"""lgbm-stockonly: pure cross-sectional LightGBM ranking + exposure overlay.

Attempt 4 (branch ``lgbm-stockonly``, parent ``lgbm-rank-v1``). Attempt 3's
model mixed 3 market-state features into X; whenever one crossed a tree split
threshold the whole ranking re-shuffled, flipping the book wholesale (11-23
names in a week) and pushing turnover to 10-19x/yr. Here the two concerns are
split:

- **Ranking**: scores from scripts/train_lgbm_scores.py --no-market-feats
  (panel lgbm_scores_h21_nomkt) — the model sees only the 26 per-stock
  cross-sectional features, so day-to-day ranking drift is incremental.
- **Exposure**: a portfolio-level overlay. "taiex" = flat when TAIEX <=
  EMA120 (the OOS-robust gate), "taiex_half" = half exposure instead of flat
  (hedges gate whipsaw eras like 2018-19 while keeping crash protection),
  "none" = always fully invested.

Turnover mechanism unchanged from attempts 2-3 (banding + sparse emission),
but the search space moves to wider bands (6-8x) and longer hold smoothing,
and WFV selection uses a turnover-penalized metric (see scripts/run_wfv.py).

Panel rows are weekly trade days, already aligned (features through d-1
scoring trades at d's open) — no further shift here.
"""

import pandas as pd

from .. import data as D

_CACHE: dict = {}


def _base():
    score = D.load("lgbm_scores_h21_nomkt")
    days = D.trading_days()
    taiex = D.load("twse_index")["close"].reindex(days).ffill()
    risk_taiex = (taiex > taiex.ewm(span=120).mean()).shift(1).fillna(True).astype(bool)
    # regime name -> (risk_on series or None, scale when risk-off)
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
    risk_on, off_scale = regimes[params.get("regime", "none")]

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
