"""Download TWSE/TPEX corporate-action reference price tables.

Finlab free data (etl:adj_close) stops at 2018-12-28, so post-2019 price
adjustment factors must come from the exchanges' official tables:

- TWSE ex-div:            /exchangeReport/TWT49U   (quarterly chunks)
- TWSE capital reduction: /exchangeReport/TWTAUU   (yearly chunks)
- TWSE par value change:  /exchangeReport/TWTB8U   (yearly chunks)
- TPEX ex-div:            /web/stock/exright/dailyquo/exDailyQ_result.php (monthly)

2015-2018 is also fetched so the derived factors can be validated against the
finlab adj_close overlap. Requests are throttled to stay polite.

Output: data/actions/{twse_exdiv,twse_capred,twse_parchg,tpex_exdiv}.parquet
"""

import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent.parent / "data" / "actions"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
START = date(2015, 1, 1)
END = date(2026, 6, 12)
SLEEP = 2.0

session = requests.Session()
session.headers.update(HEADERS)


def quarters(start: date, end: date):
    y, q = start.year, (start.month - 1) // 3
    while True:
        s = date(y, q * 3 + 1, 1)
        if s > end:
            return
        e_month = q * 3 + 3
        e = date(y + (e_month == 12), 1 if e_month == 12 else e_month + 1, 1) - pd.Timedelta(days=1)
        yield s, min(e.date() if hasattr(e, "date") else e, end)
        q += 1
        if q == 4:
            q, y = 0, y + 1


def months(start: date, end: date):
    y, m = start.year, start.month
    while True:
        s = date(y, m, 1)
        if s > end:
            return
        e = (pd.Timestamp(s) + pd.offsets.MonthEnd(0)).date()
        yield s, min(e, end)
        m += 1
        if m == 13:
            m, y = 1, y + 1


def roc(d: date) -> str:
    return f"{d.year - 1911}/{d.month:02d}/{d.day:02d}"


def parse_roc_date(s: str) -> str:
    s = s.strip().replace("年", "/").replace("月", "/").replace("日", "")
    parts = s.split("/")
    return f"{int(parts[0]) + 1911:04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"


def fetch_twse(report: str, chunks) -> pd.DataFrame:
    rows, fields = [], None
    for s, e in chunks:
        url = f"https://www.twse.com.tw/exchangeReport/{report}"
        params = {"response": "json", "startDate": s.strftime("%Y%m%d"), "endDate": e.strftime("%Y%m%d")}
        for attempt in range(3):
            try:
                r = session.get(url, params=params, timeout=30)
                d = r.json()
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
        else:
            print(f"  [{report}] {s}..{e} FAILED after retries")
            continue
        data = d.get("data") or []
        if d.get("fields"):
            fields = d["fields"]
        rows.extend(data)
        print(f"  [{report}] {s}..{e}: {len(data)} rows")
        time.sleep(SLEEP)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=fields)


def fetch_tpex_exdiv() -> pd.DataFrame:
    frames = []
    for s, e in months(START, END):
        url = "https://www.tpex.org.tw/web/stock/exright/dailyquo/exDailyQ_result.php"
        params = {"l": "zh-tw", "d": roc(s), "ed": roc(e)}
        for attempt in range(3):
            try:
                r = session.get(url, params=params, timeout=30)
                d = r.json()
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
        else:
            print(f"  [tpex] {s}..{e} FAILED after retries")
            continue
        tables = d.get("tables") or []
        if tables:
            fields = tables[0].get("fields") or []
            data = tables[0].get("data") or []
            if data:
                width = max(len(row) for row in data)
                cols = list(fields) + [f"extra_{i}" for i in range(len(fields), width)]
                data = [list(row) + [None] * (width - len(row)) for row in data]
                frames.append(pd.DataFrame(data, columns=cols[:width]))
            print(f"  [tpex] {s}..{e}: {len(data)} rows")
        time.sleep(SLEEP)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("twse_exdiv", lambda: fetch_twse("TWT49U", quarters(START, END))),
        ("twse_capred", lambda: fetch_twse("TWTAUU", [(date(y, 1, 1), min(date(y, 12, 31), END)) for y in range(START.year, END.year + 1)])),
        ("twse_parchg", lambda: fetch_twse("TWTB8U", [(date(y, 1, 1), min(date(y, 12, 31), END)) for y in range(START.year, END.year + 1)])),
        ("tpex_exdiv", fetch_tpex_exdiv),
    ]
    for name, fn in jobs:
        out = OUT / f"{name}.parquet"
        if out.exists():
            print(f"[skip] {name} exists")
            continue
        print(f"[fetch] {name}")
        df = fn()
        df.to_parquet(out)
        print(f"[done] {name}: {df.shape}\n")


if __name__ == "__main__":
    main()
