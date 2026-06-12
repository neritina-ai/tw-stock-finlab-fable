"""Run walk-forward validation for a strategy module.

Usage: uv run python scripts/run_wfv.py px_composite
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from twq import wfv  # noqa: E402

SPACES = {
    "px_composite": {
        "preset": ["mom_tilt", "defensive", "all4"],
        "top_n": [30, 50],
        "risk_off_scale": [1.0, 0.5, 0.0],
    },
    "band_voltarget": {
        "band_mult": [2.0, 4.0],
        "slow_keep": [True, False],
        "vol_target": [None, 0.12, 0.18],
        "regime": ["taiex", "breadth"],
    },
}


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "px_composite"
    mod = importlib.import_module(f"twq.strategies.{name}")
    stages = wfv.make_stages(first_test="2014-01-01", test_years=2)
    rep = wfv.run_wfv(mod.make_weights, SPACES[name], stages)
    out = Path(__file__).resolve().parent.parent / "research" / f"wfv_{name}.csv"
    rep.to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
