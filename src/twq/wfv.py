"""Walk-forward validation framework.

A strategy is a callable `make_weights(params, start, end) -> DataFrame` that
must only use information available at each emitted date (caller-shifted).
For each stage, hyperparameters are chosen on the training window (expanding,
from `data_start`) and evaluated frozen on the unseen test window. The stop
condition for the research program is: every stage's out-of-sample net
Sharpe > 2.
"""

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import backtest


@dataclass
class Stage:
    train_start: str
    train_end: str  # exclusive
    test_start: str
    test_end: str


def make_stages(
    data_start: str = "2008-06-01",
    first_test: str = "2014-01-01",
    last_test: str = "2026-04-19",
    test_years: int = 2,
    min_test_days: int = 365,
) -> list[Stage]:
    starts = []
    t = pd.Timestamp(first_test)
    last = pd.Timestamp(last_test)
    while t < last:
        starts.append(t)
        t = t + pd.DateOffset(years=test_years)
    # a final stub shorter than min_test_days is merged into the previous stage
    if len(starts) > 1 and (last - starts[-1]).days < min_test_days:
        starts.pop()
    stages = []
    for i, s in enumerate(starts):
        te = starts[i + 1] - pd.Timedelta(days=1) if i + 1 < len(starts) else last
        stages.append(
            Stage(
                train_start=data_start,
                train_end=str(s.date()),
                test_start=str(s.date()),
                test_end=str(te.date()),
            )
        )
    return stages


def grid(params_space: dict) -> list[dict]:
    keys = list(params_space)
    return [dict(zip(keys, vals)) for vals in itertools.product(*params_space.values())]


def run_wfv(
    make_weights,
    params_space: dict,
    stages: list[Stage],
    select_metric=lambda res: res.sharpe(),
    tiebreak_eps: float | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Expanding-window walk-forward: per stage, grid-search on train, freeze
    best params, evaluate on test. Returns a stage-level report.

    tiebreak_eps: if set, combos within eps of the best train metric are tied
    and the lowest-train-turnover one wins (must be pre-registered, not tuned).
    """
    rows = []
    candidates = grid(params_space)
    for i, st in enumerate(stages):
        evals = []  # (params, train metric, train annualized turnover)
        for p in candidates:
            W = make_weights(p, st.train_start, st.train_end)
            res = backtest.run(W, start=st.train_start, end=st.train_end)
            score = select_metric(res)
            if np.isfinite(score):
                evals.append((p, score, res.summary().get("ann_turnover") or 0.0))
        top = max(e[1] for e in evals)
        if tiebreak_eps:
            best, best_score, best_to = min(
                (e for e in evals if e[1] >= top - tiebreak_eps), key=lambda e: e[2]
            )
        else:
            best, best_score, best_to = max(evals, key=lambda e: e[1])
        W = make_weights(best, st.train_start, st.test_end)
        res = backtest.run(W, start=st.test_start, end=st.test_end)
        s = res.summary()
        row = {
            "stage": i + 1,
            "test_start": st.test_start,
            "test_end": st.test_end,
            "params": str(best),
            "train_sharpe": round(best_score, 3),
            "train_to": round(best_to, 1),
            "oos_sharpe": s.get("sharpe"),
            "oos_cagr": s.get("cagr"),
            "oos_maxdd": s.get("maxdd"),
            "ann_turnover": s.get("ann_turnover"),
            "avg_pos": s.get("avg_positions"),
        }
        rows.append(row)
        if verbose:
            print(row)
    rep = pd.DataFrame(rows)
    if verbose and len(rep):
        ok = (rep.oos_sharpe > 2).all()
        print(f"\nWFV stages passed (Sharpe>2): {(rep.oos_sharpe > 2).sum()}/{len(rep)}  -> {'STOP CONDITION MET' if ok else 'not met'}")
    return rep
