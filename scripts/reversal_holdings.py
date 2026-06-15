"""Reproduce the holdings-change table in research/reversal-blend.md.

Top-30 (raw score, pre-banding) overlap of the reversal blend vs all4 at the
stage-1/3/6 representative dates, plus mean weekly top-30 membership change.
Quantifies how far blending the industry-neutral short-horizon reversal pushes
the book away from all4, and how much extra it churns — the offsetting cost
behind attempt 10's negative WFV.

Usage: uv run python scripts/reversal_holdings.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from twq import data as D
from twq.strategies import lgbm_h63_ensemble as E
from twq.strategies import lgbm_ind_neutral as IN

panel = D.load("lgbm_scores_h21_nomkt")
cols, widx = panel.columns, panel.index
pre_names, post_names = E.MIXES["all4"]
pre = E._mean([E._rank(n, cols) for n in pre_names])
post = E._mean([E._rank(n, cols) for n in post_names])
all4 = pd.concat([pre.loc[: E.SPLICE], post.loc[post.index > E.SPLICE]])
M = all4.notna()

ac = D.adj_ohlc()["close"].reindex(columns=cols)
revs = {}
for L in (5, 10):
    rev = (-(ac / ac.shift(L) - 1).shift(1)).reindex(index=widx).where(M)
    ST = rev.T
    g = ST.groupby(IN._bucket_map(cols))
    r_ind = g.rank(pct=True)
    cnt = g.transform("count")
    revs[L] = r_ind.where(cnt >= IN.MIN_IND, ST.rank(pct=True)).T
revC = (revs[5] + revs[10]) / 2

books = {
    "all4": all4,
    "w0.25": (0.75 * all4 + 0.25 * revC.fillna(0.5)).where(M),
    "w0.5": (0.5 * all4 + 0.5 * revC.fillna(0.5)).where(M),
}


def top30(S, d):
    return set(S.loc[d].dropna().nlargest(30).index)


print("top-30 overlap of blend vs all4 (raw score, pre-banding):")
for dstr in ("2015-06-01", "2019-06-01", "2026-04-01"):
    d = widx[widx.searchsorted(pd.Timestamp(dstr))]
    a = top30(books["all4"], d)
    line = f"  {d.date()}  "
    for k in ("w0.25", "w0.5"):
        line += f"overlap {k}: {len(a & top30(books[k], d))}/30   "
    print(line)

print("\nmean weekly top-30 membership change (names/week):")
for k, S in books.items():
    prev, changes = None, []
    for d in widx[::1]:
        cur = top30(S, d) if S.loc[d].notna().any() else None
        if cur is None:
            continue
        if prev is not None:
            changes.append(len(cur - prev))
        prev = cur
    print(f"  {k:<7} {pd.Series(changes).mean():.2f}")
