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
    "lgbm_rank": {
        "band_mult": [2.0, 4.0],
        "hold_smooth": [1, 4],
        "top_n": [30, 50],
        "regime": ["taiex", "none"],
    },
    "lgbm_stockonly": {
        "band_mult": [4.0, 8.0],
        "hold_smooth": [4, 8],
        "top_n": [30, 50],
        "regime": ["taiex", "taiex_half", "none"],
    },
}


def _sharpe(res):
    return res.sharpe()


def _sharpe_to_penalized(res):
    # turnover-penalized train selection: ~12x/yr churn costs 0.6 Sharpe,
    # breaking near-ties in favour of the cheaper config (OOS still raw)
    return res.sharpe() - 0.05 * (res.summary().get("ann_turnover") or 0.0)


METRICS = {"lgbm_stockonly": _sharpe_to_penalized}


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "px_composite"
    mod = importlib.import_module(f"twq.strategies.{name}")
    stages = wfv.make_stages(first_test="2014-01-01", test_years=2)
    rep = wfv.run_wfv(
        mod.make_weights, SPACES[name], stages, select_metric=METRICS.get(name, _sharpe)
    )
    out = Path(__file__).resolve().parent.parent / "research" / f"wfv_{name}.csv"
    rep.to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
