"""Build the cumulative price-adjustment factor matrix (data/derived/adj_factor).

Event sources
-------------
1. 2007..2018-12-28: implied from finlab etl:adj_close. On ex-event day e the
   ratio (adj return)/(raw return) equals 1/f_e where f_e = ref/preclose, so
   f_e is recovered from the two series; non-event days give exactly 1.
2. 2019..now: official exchange tables (TWSE TWT49U / TWTAUU / TWTB8U, TPEX
   exDailyQ): f_e = reference price / pre-event close.
3. Long-halt resumes (>=6 missing trading days) with |open/last_close - 1| >
   20% not explained by 1-2 (mostly TPEX capital reductions, which have no
   public history endpoint): synthetic value-flat factor f = open/last_close.

adj_factor.loc[t, s] = product of f_e for all events e after t (anchored 1.0
at the last grid date), so adj_price = raw_price * adj_factor.

Validation: 2015-2018 official-table factors are compared against the
finlab-implied factors; mismatches > 0.5% are reported.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "derived"


def parse_roc_date(s: str) -> pd.Timestamp | None:
    s = str(s).strip().replace("年", "/").replace("月", "/").replace("日", "")
    parts = s.split("/")
    if len(parts) != 3:
        return None
    try:
        return pd.Timestamp(int(parts[0]) + 1911, int(parts[1]), int(parts[2]))
    except ValueError:
        return None


def fnum(x) -> float:
    try:
        return float(str(x).replace(",", "").strip())
    except ValueError:
        return np.nan


def implied_events(close: pd.DataFrame) -> pd.DataFrame:
    """Event factors implied by finlab adj_close over the local raw close."""
    adj = pd.read_parquet(ROOT / "data" / "finlab" / "adj_close.parquet")
    adj.index = pd.to_datetime(adj.index)
    cols = close.columns.intersection(adj.columns)
    adj = adj.reindex(index=close.index, columns=cols)
    raw = close[cols]

    adj_ret = adj / adj.shift(1)
    raw_ret = raw / raw.shift(1)
    f = (raw_ret / adj_ret).where(np.isfinite(adj_ret) & np.isfinite(raw_ret))
    ev = f.stack()
    ev = ev[(ev - 1).abs() > 0.0025]
    ev = ev[(ev > 0.2) & (ev < 3.0)]  # sanity bounds
    out = ev.rename("factor").reset_index()
    out.columns = ["date", "stock", "factor"]
    return out


def official_events() -> pd.DataFrame:
    A = ROOT / "data" / "actions"
    frames = []

    tw = pd.read_parquet(A / "twse_exdiv.parquet")
    tw_f = pd.DataFrame(
        {
            "date": tw["資料日期"].map(parse_roc_date),
            "stock": tw["股票代號"].astype(str).str.strip(),
            "pre": tw["除權息前收盤價"].map(fnum),
            "ref": tw["除權息參考價"].map(fnum),
        }
    )
    frames.append(tw_f)

    tp = pd.read_parquet(A / "tpex_exdiv.parquet")
    tp_f = pd.DataFrame(
        {
            "date": tp["除權息日期"].map(parse_roc_date),
            "stock": tp["代號"].astype(str).str.strip(),
            "pre": tp["除權息前收盤價"].map(fnum),
            "ref": tp["除權息參考價"].map(fnum),
        }
    )
    frames.append(tp_f)

    cr = pd.read_parquet(A / "twse_capred.parquet")
    cr_f = pd.DataFrame(
        {
            "date": cr["恢復買賣日期"].map(parse_roc_date),
            "stock": cr["股票代號"].astype(str).str.strip(),
            "pre": cr["停止買賣前收盤價格"].map(fnum),
            "ref": cr["恢復買賣參考價"].map(fnum),
        }
    )
    frames.append(cr_f)

    pc = pd.read_parquet(A / "twse_parchg.parquet")
    if len(pc):
        pc_f = pd.DataFrame(
            {
                "date": pc["恢復買賣日期"].map(parse_roc_date),
                "stock": pc["股票代號"].astype(str).str.strip(),
                "pre": pc["停止買賣前收盤價格"].map(fnum),
                "ref": pc["恢復買賣參考價"].map(fnum),
            }
        )
        frames.append(pc_f)

    ev = pd.concat(frames, ignore_index=True).dropna(subset=["date", "pre", "ref"])
    ev = ev[(ev.pre > 0) & (ev.ref > 0)]
    ev["factor"] = ev.ref / ev.pre
    ev = ev[(ev.factor > 0.2) & (ev.factor < 3.0)]
    return ev[["date", "stock", "factor"]]


def synthetic_halt_events(close: pd.DataFrame, open_: pd.DataFrame, known: pd.DataFrame) -> pd.DataFrame:
    """Value-flat factors for big jumps across long halts without a known event."""
    known_set = set(zip(known.date, known.stock))
    rows = []
    dates = close.index
    pos = {d: i for i, d in enumerate(dates)}
    for s in close.columns:
        c = close[s].dropna()
        if len(c) < 2:
            continue
        idxs = np.array([pos[d] for d in c.index])
        gaps = np.where(np.diff(idxs) > 6)[0]
        for g in gaps:
            resume_date = c.index[g + 1]
            last_close = c.iloc[g]
            o = open_.at[resume_date, s]
            base = o if np.isfinite(o) else c.iloc[g + 1]
            jump = base / last_close - 1
            if abs(jump) > 0.20 and (resume_date, s) not in known_set:
                rows.append((resume_date, s, base / last_close))
    return pd.DataFrame(rows, columns=["date", "stock", "factor"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    close = pd.read_parquet(ROOT / "data" / "local" / "close.parquet")
    open_ = pd.read_parquet(ROOT / "data" / "local" / "open.parquet")

    cutoff = pd.Timestamp("2019-01-01")
    imp = implied_events(close)
    off = official_events()
    off = off[off.stock.isin(close.columns)]

    # --- validation on the 2015-2018 overlap ---
    val_imp = imp[(imp.date >= "2015-01-01") & (imp.date < "2019-01-01")]
    val_off = off[(off.date >= "2015-01-01") & (off.date < "2019-01-01")]
    m = val_imp.merge(val_off, on=["date", "stock"], suffixes=("_imp", "_off"))
    if len(m):
        diff = (m.factor_imp / m.factor_off - 1).abs()
        print(f"validation: {len(m)} matched events 2015-2018")
        print(f"  matched within 0.5%: {(diff < 0.005).mean():.3%}, median diff {diff.median():.5f}")
        only_imp = len(val_imp) - len(m)
        only_off = len(val_off) - len(m)
        print(f"  implied-only events: {only_imp}, official-only: {only_off}")

    events = pd.concat(
        [imp[imp.date < cutoff], off[off.date >= cutoff]], ignore_index=True
    )
    syn = synthetic_halt_events(close, open_, events)
    syn_recent = syn[syn.date >= cutoff]  # pre-2019 jumps already in finlab factors
    print(f"events: implied {len(imp[imp.date < cutoff])}, official {len(off[off.date >= cutoff])}, synthetic {len(syn_recent)}")
    events = pd.concat([events, syn_recent], ignore_index=True)
    events = events.groupby(["date", "stock"], as_index=False).factor.prod()

    fmat = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    ev_in_grid = events[events.date.isin(close.index) & events.stock.isin(close.columns)]
    skipped = len(events) - len(ev_in_grid)
    if skipped:
        print(f"  ({skipped} events off-grid skipped)")
    fmat.values[
        close.index.get_indexer(ev_in_grid.date), close.columns.get_indexer(ev_in_grid.stock)
    ] = ev_in_grid.factor.to_numpy()

    # adj_factor[t] = prod of event factors AFTER t  (event on day e applies to days < e)
    adj_factor = fmat.iloc[::-1].cumprod().iloc[::-1].shift(-1, fill_value=1.0)
    # 64-bit parquet
    adj_factor.to_parquet(OUT / "adj_factor.parquet")
    print(f"adj_factor saved: {adj_factor.shape}")

    # quick sanity: TSMC 2330 total adjustment over the span
    s = "2330"
    if s in adj_factor.columns:
        print(f"2330 factor at 2007 start: {adj_factor[s].iloc[0]:.4f} (close {close[s].iloc[0]}) -> adj {close[s].iloc[0]*adj_factor[s].iloc[0]:.2f}")


if __name__ == "__main__":
    main()
