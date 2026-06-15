"""Download Finlab free datasets to local parquet files.

Free quota is 500 MB/day, so we are selective: raw OHLCV already lives in
S:\\tw_stock.db; from Finlab we only need adjusted close (to derive the
adjustment factor), fundamentals, monthly revenue, valuation ratios and a few
market-level series. Datasets that turn out to be VIP-only are skipped with a
log line instead of failing the whole run.

Usage:
    uv run python scripts/download_finlab.py            # download DATASETS
    uv run python scripts/download_finlab.py key1 key2  # download specific keys
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "finlab"

# (dataset_key, short_file_name)
DATASETS = [
    # --- adjustment / market value ---
    ("etl:adj_close", "adj_close"),
    ("etl:market_value", "market_value"),
    # --- monthly revenue (published ~10th of next month) ---
    ("monthly_revenue:當月營收", "rev"),
    ("monthly_revenue:去年同月增減(%)", "rev_yoy"),
    ("monthly_revenue:上月比較增減(%)", "rev_mom"),
    # --- quarterly financial statements ---
    ("financial_statement:營業收入淨額", "fs_revenue"),
    ("financial_statement:營業利益", "fs_op_income"),
    ("financial_statement:稅後淨利", "fs_net_income"),
    ("financial_statement:股東權益總額", "fs_equity"),
    ("financial_statement:股本", "fs_capital"),
    ("financial_statement:營業活動之淨現金流入_流出", "fs_op_cashflow"),
    ("financial_statement:資產總額", "fs_assets"),
    # --- derived fundamental features ---
    ("fundamental_features:ROE稅後", "ff_roe"),
    ("fundamental_features:營業利益率", "ff_op_margin"),
    ("fundamental_features:稅後淨利成長率", "ff_ni_growth"),
    ("fundamental_features:營收成長率", "ff_rev_growth"),
    # --- daily valuation ratios ---
    ("price_earning_ratio:本益比", "per"),
    ("price_earning_ratio:股價淨值比", "pbr"),
    ("price_earning_ratio:殖利率(%)", "dividend_yield"),
    # --- institutional investors (may be VIP) ---
    ("institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)", "ii_foreign"),
    ("institutional_investors_trading_summary:投信買賣超股數", "ii_trust"),
    # --- margin transactions (may be VIP) ---
    ("margin_transactions:融資今日餘額", "margin_balance"),
    # --- market level ---
    ("benchmark_return:發行量加權股價報酬指數", "taiex_tr"),
    # --- classification ---
    ("security_categories", "security_categories"),
    # --- broker / others left out: likely VIP or too big ---
]


def main() -> None:
    import finlab
    from finlab import data

    token = os.getenv("FINLAB_API_TOKEN")
    if not token:
        sys.exit("FINLAB_API_TOKEN not set")
    finlab.login(token)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    wanted = sys.argv[1:]
    items = [(k, n) for k, n in DATASETS if not wanted or k in wanted or n in wanted]

    total_mb = 0.0
    for key, name in items:
        out = DATA_DIR / f"{name}.parquet"
        if out.exists():
            print(f"[skip] {key} -> {out.name} (exists)")
            continue
        t0 = time.time()
        try:
            df = data.get(key)
        except Exception as e:  # noqa: BLE001 - VIP/quota errors must not kill the run
            print(f"[FAIL] {key}: {type(e).__name__}: {e}")
            continue
        if isinstance(df, pd.Series):
            df = df.to_frame()
        df.columns = [str(c) for c in df.columns]
        if isinstance(df.index, pd.PeriodIndex):
            df.index = df.index.astype(str)
        df.to_parquet(out)
        mb = out.stat().st_size / 1e6
        total_mb += mb
        print(f"[ok]   {key} -> {out.name}  shape={df.shape}  {mb:.1f}MB  {time.time()-t0:.1f}s")

    print(f"\nTotal new downloads on disk: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
