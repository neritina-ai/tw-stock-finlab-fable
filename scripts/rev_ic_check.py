"""Attempt-10 gate-check: standalone weekly IC of industry-neutral
short-horizon reversal vs the 21-trading-day forward open->open return.

Pre-registered gate (research/ind-neutral.md "下一步"): the reversal factor's
mean weekly IC must be positive over the full panel period AND in the
2014-15 / 2018-19 chop subwindows before any WFV is run. Variants:

- rev5 / rev10: -(past 5/10-day return) on adjusted closes through d-1
  (daily shift(1) before reindexing to weekly panel rows - no lookahead),
  masked to the all4-scored universe.
- *_raw: global weekly pct rank; *_ind: within-TWSE-industry pct rank via
  the lgbm_ind_neutral machinery (MIN_IND fallback to global).
- blends: (1-w)*all4 + w*rev_ind with neutral 0.5 fill where rev is missing,
  w in {0.25, 0.5} - the exact scores a WFV module would emit, so blend IC
  vs all4 IC per subwindow is the direct go/no-go evidence.

Forward return matches lgbm_ic_adaptive's evaluation convention
(open(d) -> open(d+21), the return the trade actually captures).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from twq import data as D
from twq.strategies import lgbm_h63_ensemble as E
from twq.strategies import lgbm_ind_neutral as IN

H_EVAL = 21
MIN_NAMES = 100

WINDOWS = [
    ("full", None, None),
    ("2014-15", "2014-01-01", "2015-12-31"),
    ("2016-17", "2016-01-01", "2017-12-31"),
    ("2018-19", "2018-01-01", "2019-12-31"),
    ("2020-21", "2020-01-01", "2021-12-31"),
    ("2022-23", "2022-01-01", "2023-12-31"),
    ("2024-26", "2024-01-01", "2026-04-19"),
]


def _ind_rank_df(F: pd.DataFrame, cols: pd.Index) -> pd.DataFrame:
    """lgbm_ind_neutral._ind_rank for an in-memory frame instead of a panel name."""
    ST = F.T
    g = ST.groupby(IN._bucket_map(cols))
    r_ind = g.rank(pct=True)
    cnt = g.transform("count")
    r_glob = ST.rank(pct=True)
    return r_ind.where(cnt >= IN.MIN_IND, r_glob).T


def _ic(F: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    m = F.notna() & fwd.notna()
    ra = F.where(m).rank(axis=1, pct=True)
    fa = fwd.where(m).rank(axis=1, pct=True)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    fa = fa.sub(fa.mean(axis=1), axis=0)
    num = (ra * fa).sum(axis=1)
    den = np.sqrt(ra.pow(2).sum(axis=1) * fa.pow(2).sum(axis=1))
    ic = num / den
    ic[m.sum(axis=1) < MIN_NAMES] = np.nan
    return ic


def main() -> None:
    panel = D.load("lgbm_scores_h21_nomkt")
    cols, widx = panel.columns, panel.index

    pre_names, post_names = E.MIXES["all4"]
    pre = E._mean([E._rank(n, cols) for n in pre_names])
    post = E._mean([E._rank(n, cols) for n in post_names])
    all4 = pd.concat([pre.loc[: E.SPLICE], post.loc[post.index > E.SPLICE]])
    M = all4.notna()

    ohlc = D.adj_ohlc()
    ac = ohlc["close"].reindex(columns=cols)
    ao = ohlc["open"].reindex(columns=cols)
    fwd = (ao.shift(-H_EVAL) / ao - 1).reindex(index=widx)

    factors: dict[str, pd.DataFrame] = {}
    for L in (5, 10):
        rev = (-(ac / ac.shift(L) - 1).shift(1)).reindex(index=widx).where(M)
        factors[f"rev{L}_raw"] = rev.rank(axis=1, pct=True)
        factors[f"rev{L}_ind"] = _ind_rank_df(rev, cols)
    factors["revC_ind"] = (factors["rev5_ind"] + factors["rev10_ind"]) / 2
    factors["all4"] = all4
    for tag in ("rev5_ind", "rev10_ind", "revC_ind"):
        for w in (0.25, 0.5):
            factors[f"{tag[:-4]}+w{w}"] = (
                (1 - w) * all4 + w * factors[tag].fillna(0.5)
            ).where(M)

    ics = {name: _ic(F, fwd) for name, F in factors.items()}

    print(f"{'factor':<14}", end="")
    for wname, _, _ in WINDOWS:
        print(f"{wname:>16}", end="")
    print()
    for name, ic in ics.items():
        print(f"{name:<14}", end="")
        for _, lo, hi in WINDOWS:
            s = ic if lo is None else ic.loc[lo:hi]
            s = s.dropna()
            t = s.mean() / s.std() * np.sqrt(len(s)) if len(s) > 2 else np.nan
            print(f"{s.mean():>8.4f}({t:>5.1f})", end="")
        print()

    # signal persistence: weekly lag-1 rank autocorrelation (banding feasibility)
    print("\nweekly lag-1 rank autocorr:")
    for name in ("rev5_ind", "rev10_ind", "revC_ind", "all4"):
        F = factors[name]
        ac1 = F.corrwith(F.shift(1), axis=1, method="spearman").mean()
        print(f"  {name:<12} {ac1:.3f}")


if __name__ == "__main__":
    main()
