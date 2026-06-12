"""band-voltarget: weekly banded top-N rotation, vol-targeted, regime-gated.

Attempt 2 (branch ``band-voltarget``, parent ``px-composite-v1``). Keeps the
defensive blend that won every WFV stage in v1 (52w-high proximity .4, 5d
reversal .3, low 60d vol .3) and attacks v1's two weaknesses — cost drag and
unmanaged volatility:

- Weekly decision days instead of monthly full rebalances.
- Banding: a held name is kept while it stays inside the top band_mult*N by
  score; vacancies are filled with the best-ranked non-held names. band_mult
  1.0 degenerates to plain weekly top-N (control).
- slow_keep: entries are ranked by the full blend, but the keep/band test
  uses only the slow components (nh52+lvol) — the 5d reversal sleeve churns
  the composite rank wholesale every week (58x/yr turnover in the unbanded
  weekly control), so it must not drive exits.
- Vol targeting: exposure = min(1, vol_target / 60d vol of an equal-weighted
  return proxy of the current holdings) when vol_target is set.
- Regime gate: "taiex" (TAIEX > EMA120) or "breadth" (EMA10 of the fraction
  of universe names above their 120d MA > 0.40); off -> all cash.
- A row is emitted (= trade at open) only when membership changes, exposure
  moves > 0.05, or 8 weeks have passed (re-equalize drift); other weeks the
  backtest drifts the held weights at zero cost.
"""

import numpy as np
import pandas as pd

from .. import data as D
from .. import factors as F

BLEND = {"nh52": 0.4, "rev5": 0.3, "lvol": 0.3}
BLEND_HOLD = {"nh52": 0.55, "lvol": 0.45}


def _blend(feats: dict, blend: dict) -> pd.DataFrame:
    score = None
    for k, w in blend.items():
        r = F.rank_xs(feats[k]) * w
        score = r if score is None else score.add(r, fill_value=0)
    return score.shift(1)  # info through t-1 for trades at t's open


def _base():
    adj_close = D.load("close") * D.load("adj_factor")
    days = adj_close.index
    raw_close = D.load("close")
    turnover = D.load("turnover")

    feats = {
        "nh52": F.near_high(adj_close, 252),
        "rev5": -F.momentum(adj_close, 5, 0),
        "lvol": -F.volatility(adj_close, 60),
    }
    score = _blend(feats, BLEND)
    score_hold = _blend(feats, BLEND_HOLD)

    listed_days = raw_close.notna().rolling(130).sum()
    tdr = raw_close.columns.str.startswith("91")
    uni = (
        (turnover.rolling(20, min_periods=10).median() > 20e6)
        & (raw_close > 10)
        & (listed_days >= 100)
    )
    uni.loc[:, tdr] = False
    eligible = uni.shift(1).fillna(False).astype(bool)

    rets = adj_close.pct_change(fill_method=None)

    taiex = D.load("twse_index")["close"].reindex(days).ffill()
    risk_taiex = (taiex > taiex.ewm(span=120).mean()).shift(1).fillna(True).astype(bool)

    above = adj_close > adj_close.rolling(120, min_periods=60).mean()
    breadth = (above & uni).sum(axis=1) / uni.sum(axis=1).replace(0, np.nan)
    risk_breadth = (breadth.ewm(span=10).mean() > 0.40).shift(1).fillna(True).astype(bool)

    return days, score, score_hold, eligible, rets, {"taiex": risk_taiex, "breadth": risk_breadth}


_CACHE: dict = {}


def make_weights(params: dict, start: str, end: str) -> pd.DataFrame:
    if "base" not in _CACHE:
        _CACHE["base"] = _base()
    days, score, score_hold, eligible, rets, regimes = _CACHE["base"]
    keep_score = score_hold if params.get("slow_keep", True) else score

    top_n = params.get("top_n", 30)
    band = int(top_n * params["band_mult"])
    vt = params.get("vol_target") or 0.0
    risk_on = regimes.get(params.get("regime"))

    rebal = F.week_starts(days)
    rebal = rebal[(rebal >= pd.Timestamp(start)) & (rebal <= pd.Timestamp(end))].sort_values()

    ret_np = rets.to_numpy()
    day_pos = {d: i for i, d in enumerate(days)}
    col_pos = {c: i for i, c in enumerate(rets.columns)}

    held: list[str] = []
    last_scale = -1.0
    weeks_since = 0
    rows: dict[pd.Timestamp, dict[str, float]] = {}

    for d in rebal:
        s = score.loc[d].where(eligible.loc[d]).dropna().sort_values(ascending=False)
        if s.empty:
            continue
        ks = keep_score.loc[d].where(eligible.loc[d]).dropna().sort_values(ascending=False)
        band_set = set(ks.index[:band])
        keep = [x for x in held if x in band_set]
        adds = [x for x in s.index if x not in keep][: top_n - len(keep)]
        new_held = keep + adds

        scale = 1.0
        if risk_on is not None and not risk_on.loc[d]:
            scale = 0.0
        if scale > 0 and vt > 0 and new_held:
            p = day_pos[d]
            ci = [col_pos[x] for x in new_held]
            win = ret_np[max(0, p - 60) : p][:, ci]
            if win.shape[0] >= 30:
                pr = np.nanmean(win, axis=1)
                v = np.nanstd(pr, ddof=1) * np.sqrt(252)
                if np.isfinite(v) and v > 1e-8:
                    scale = min(1.0, vt / v)

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
    return W.reindex(index=out_days, columns=rets.columns)
