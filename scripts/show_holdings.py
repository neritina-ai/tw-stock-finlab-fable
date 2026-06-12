"""Print emitted portfolio rows (holdings changes) for a strategy/params.

Usage:
  uv run python scripts/show_holdings.py band_voltarget "{'band_mult': 4.0}" 2025-06-01 2026-04-19
"""

import ast
import importlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    name, params_s, start, end = sys.argv[1:5]
    params = ast.literal_eval(params_s)
    mod = importlib.import_module(f"twq.strategies.{name}")

    master = pd.read_parquet(ROOT / "data" / "local" / "stock_master.parquet")
    id2name = master.drop_duplicates("ID", keep="last").set_index("ID")["name"]

    W = mod.make_weights(params, start, end)
    rows = W.dropna(how="all")
    prev: set = set()
    for d, row in rows.iterrows():
        held = row[row > 1e-6]
        names = set(held.index)
        adds = sorted(names - prev)
        drops = sorted(prev - names)
        expo = held.sum()
        line = f"{d.date()}  expo={expo:.2f} n={len(held)}"
        if adds:
            line += "  +[" + " ".join(f"{s} {id2name.get(s, '?')}" for s in adds) + "]"
        if drops:
            line += "  -[" + " ".join(f"{s} {id2name.get(s, '?')}" for s in drops) + "]"
        print(line)
        prev = names
    if len(rows):
        d, row = list(rows.iterrows())[-1]
        held = row[row > 1e-6].sort_values(ascending=False)
        print(f"\nfinal book {d.date()} (n={len(held)}):")
        for s, w in held.items():
            print(f"  {s} {id2name.get(s, '?')}  {w:.3f}")


if __name__ == "__main__":
    main()
