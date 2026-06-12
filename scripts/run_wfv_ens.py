"""Walk-forward validation with selection-layer variants (attempt 8).

Pre-registered variants (k=3 and the 5-year metric window are fixed
constants, chosen before any run, never tuned):

- top3:   top-3 combos by full-window train Sharpe, OOS = equal-weight mean
          of the 3 sleeves' net return series (portfolio-of-portfolios)
- r5:     top-1 combo by trailing-5y train Sharpe
- top3r5: top-3 combos by trailing-5y train Sharpe, OOS = mean of sleeves

Usage: uv run python scripts/run_wfv_ens.py lgbm_mix_all4 [lgbm_ic_adaptive ...]
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from run_wfv import SPACES  # noqa: E402
from twq import wfv  # noqa: E402

VARIANTS = [
    {"name": "top3", "k": 3, "recent_years": None},
    {"name": "r5", "k": 1, "recent_years": 5},
    {"name": "top3r5", "k": 3, "recent_years": 5},
]


def main() -> None:
    names = sys.argv[1:] or ["lgbm_mix_all4"]
    stages = wfv.make_stages(first_test="2014-01-01", test_years=2)
    research = Path(__file__).resolve().parent.parent / "research"
    for name in names:
        mod = importlib.import_module(f"twq.strategies.{name}")
        print(f"===== {name} =====")
        reps = wfv.run_wfv_variants(mod.make_weights, SPACES[name], stages, VARIANTS)
        for vname, rep in reps.items():
            out = research / f"wfv_{name}_{vname}.csv"
            rep.to_csv(out, index=False)
            print(f"saved {out}")


if __name__ == "__main__":
    main()
