"""Extract S:\\tw_stock.db into wide (date x stock) parquet matrices.

Produces data/local/{open,high,low,close,volume,turnover}.parquet for all
common stocks (4-digit numeric IDs, TWSE+OTC), excluding the ETL marker rows
ID in ('TWSE','OTC'). Also saves the stock master table and the TWSE index.
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd

DB = r"S:\tw_stock.db"
OUT = Path(__file__).resolve().parent.parent / "data" / "local"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)

    stock = pd.read_sql("SELECT * FROM stock WHERE ID NOT IN ('TWSE','OTC')", con)
    stock.to_parquet(OUT / "stock_master.parquet")
    common = sorted(s for s in stock.ID if re.fullmatch(r"\d{4}", s))
    print(f"stocks: {len(stock)} total, {len(common)} common (4-digit)")

    q = """
        SELECT date, ID, open, high, low, close, volume, turnover
        FROM quote WHERE ID NOT IN ('TWSE','OTC')
    """
    df = pd.read_sql(q, con, parse_dates=["date"])
    df = df[df.ID.isin(common)]
    print(f"quote rows: {len(df):,}")

    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        wide = df.pivot(index="date", columns="ID", values=col).astype("float64")
        wide.to_parquet(OUT / f"{col}.parquet")
        print(f"  {col}: {wide.shape}")

    twse = pd.read_sql("SELECT * FROM twse", con, parse_dates=["date"]).set_index("date")
    twse.to_parquet(OUT / "twse_index.parquet")
    print(f"twse index: {twse.shape}")


if __name__ == "__main__":
    main()
