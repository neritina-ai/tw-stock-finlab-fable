"""Hindsight-best OOS per WFV stage (the selection ceiling).

Evaluates every combo in run_wfv.SPACES[strategy] directly on each stage's
TEST window. This is deliberately look-ahead: it answers "if train-window
selection were perfect, what could this engine score?" — separating
selection error from alpha-ceiling problems when reading WFV results.

Usage: uv run python scripts/hindsight_best.py lgbm_stockonly
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_wfv import SPACES  # noqa: E402
from twq import backtest, wfv  # noqa: E402


def main() -> None:
    name = sys.argv[1]
    mod = importlib.import_module(f"twq.strategies.{name}")
    stages = wfv.make_stages(first_test="2014-01-01", test_years=2)
    combos = wfv.grid(SPACES[name])

    print(f"=== hindsight-best OOS per stage ({name}, selection ceiling) ===")
    for st in stages:
        best, by_regime = None, {}
        for p in combos:
            W = mod.make_weights(p, st.train_start, st.test_end)
            res = backtest.run(W, start=st.test_start, end=st.test_end)
            s = res.summary()
            s["params"] = p
            r = p.get("regime", "-")
            if r not in by_regime or s["sharpe"] > by_regime[r]:
                by_regime[r] = round(s["sharpe"], 2)
            if best is None or s["sharpe"] > best["sharpe"]:
                best = s
        print(
            f"{st.test_start}..{st.test_end}  best={best['sharpe']:.2f} "
            f"{best['params']}  cagr={best['cagr']:.3f} maxdd={best['maxdd']:.3f} "
            f"to={best['ann_turnover']:.1f}  best-by-regime={by_regime}"
        )


if __name__ == "__main__":
    main()
