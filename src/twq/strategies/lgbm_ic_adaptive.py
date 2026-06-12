"""lgbm-ic-adaptive: IC-adaptive panel weights over the all4 rank ensemble.

Attempt 7 (branch ``ic-adaptive``, parent ``h63-ensemble``). Attempt 6 showed
static equal-weight rank-mean is a partial free lunch: it breaks single-model
ceilings where panels disagree (stage 2/3) but dilutes stages where one model
dominates (stage 1: fund-h21 1.70 alone -> 1.30 in all4). Here the per-era
panel weights adapt to realized skill instead:

- For each panel, weekly realized rank-IC = Spearman corr (per day, joint
  eligible set) between the panel's score and the realized 21-trading-day
  open->open forward return. A COMMON 21d evaluation horizon is used for all
  panels (including h63) so weights compare like with like - each panel's own
  label horizon would structurally favour slow labels (their rank ICs run
  higher) and triple the information lag.
- Point-in-time: the IC of week d is usable at decision week t only when the
  label window has fully closed before t's open, i.e. pos(d) + 21 <= pos(t) - 1
  on the trading-day calendar.
- Weight(t) per panel = trailing mean of the last 52 usable weekly ICs,
  clipped at 0, raised to ic_gamma, renormalized. Fewer than 26 IC
  observations (or all clipped to 0) falls back to equal weights, so early
  history reproduces all4. ic_gamma sharpens the tilt: all four panels carry
  persistently positive IC in the 0.08-0.14 range, so the linear (gamma=1)
  weights hover near 0.55/0.45 splits; gamma=3 lets a dominant panel take
  ~2/3. gamma is selected on the train window like any other hyperparameter
  (registered in the search space before any WFV run).

Era structure, splice (2018-12-28), taiex_half exposure, banding and sparse
emission are inherited unchanged from lgbm_h63_ensemble: this module only
replaces the equal-weight rank mean with the IC-weighted rank sum, then
delegates to that module's make_weights via its base cache (mix "ic4").
"""

import numpy as np
import pandas as pd

from .. import data as D
from . import lgbm_h63_ensemble as E

H_EVAL = 21  # common evaluation horizon, trading days
IC_WINDOW = 52  # trailing weekly ICs in the mean
IC_MIN = 26  # fewer usable ICs than this -> equal-weight fallback
MIN_NAMES = 100  # joint names needed for a week's IC to count

_DIAG: dict = {}  # panel-weight frames kept for inspection/research notes


def _ic_series(name: str, cols: pd.Index, fwd: pd.DataFrame) -> pd.Series:
    """Weekly realized rank-IC of one panel vs the common forward return."""
    r = E._rank(name, cols)
    f = fwd.reindex(index=r.index, columns=cols)
    m = r.notna() & f.notna()
    ra = r.where(m).rank(axis=1, pct=True)
    fa = f.where(m).rank(axis=1, pct=True)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    fa = fa.sub(fa.mean(axis=1), axis=0)
    num = (ra * fa).sum(axis=1)
    den = np.sqrt(ra.pow(2).sum(axis=1) * fa.pow(2).sum(axis=1))
    ic = num / den
    ic[m.sum(axis=1) < MIN_NAMES] = np.nan
    return ic


def _ic_means(names: list[str], cols: pd.Index) -> pd.DataFrame:
    """PIT trailing-mean realized IC per panel (NaN where history is short)."""
    key = "ic_means_" + "|".join(names)
    if key in _DIAG:
        return _DIAG[key]
    ao = D.adj_ohlc()["open"]
    fwd = ao.shift(-H_EVAL) / ao - 1
    widx = E._rank(names[0], cols).index
    dpos = D.trading_days().searchsorted(widx)
    # weeks [0, k_i) have pos(d) + H_EVAL <= pos(t_i) - 1: label closed pre-open
    k = np.searchsorted(dpos, dpos - H_EVAL, side="left")
    lo = np.maximum(k - IC_WINDOW, 0)

    means = {}
    for n in names:
        ic = _ic_series(n, cols, fwd).to_numpy()
        fin = np.isfinite(ic)
        cs = np.concatenate([[0.0], np.cumsum(np.where(fin, ic, 0.0))])
        cc = np.concatenate([[0], np.cumsum(fin)])
        s, c = cs[k] - cs[lo], cc[k] - cc[lo]
        means[n] = np.where(c >= IC_MIN, s / np.maximum(c, 1), np.nan)
    _DIAG[key] = pd.DataFrame(means, index=widx)
    return _DIAG[key]


def _panel_weights(names: list[str], cols: pd.Index, gamma: int) -> pd.DataFrame:
    """Per-week panel weights from PIT trailing realized IC (rows sum to 1)."""
    M = _ic_means(names, cols)
    raw = M.clip(lower=0.0) ** gamma
    tot = raw.sum(axis=1)
    fallback = M.isna().any(axis=1) | (tot <= 0)
    W = raw.div(tot.where(tot > 0), axis=0)
    W.loc[fallback] = 1.0 / len(names)
    return W


def _weighted_score(names: list[str], cols: pd.Index, w: pd.DataFrame) -> pd.DataFrame:
    out = None
    for n in names:
        term = E._rank(n, cols).mul(w[n], axis=0)
        out = term if out is None else out + term
    return out


def _base(gamma: int):
    cols = D.load("lgbm_scores_h21_nomkt").columns
    pre_names, post_names = E.MIXES["all4"]
    pre_w = _panel_weights(pre_names, cols, gamma)
    post_w = _panel_weights(post_names, cols, gamma)
    _DIAG[f"pre_w_g{gamma}"], _DIAG[f"post_w_g{gamma}"] = pre_w, post_w
    pre = _weighted_score(pre_names, cols, pre_w)
    post = _weighted_score(post_names, cols, post_w)
    score = pd.concat([pre.loc[: E.SPLICE], post.loc[post.index > E.SPLICE]])

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
    gamma = int(params.get("ic_gamma", 1))
    mix = f"ic4g{gamma}"  # cache-key only; never resolved through E.MIXES
    if f"base_{mix}" not in E._CACHE:
        E._CACHE[f"base_{mix}"] = _base(gamma)
    p = {k: v for k, v in params.items() if k != "ic_gamma"}
    return E.make_weights(dict(p, mix=mix), start, end)
