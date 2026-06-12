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


def _sharpe_of(ret: pd.Series, ann: int = 252) -> float:
    r = ret.dropna()
    if len(r) < 20 or r.std() == 0:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(ann))


def run_wfv_variants(
    make_weights,
    params_space: dict,
    stages: list[Stage],
    variants: list[dict],
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Selection-layer variants sharing one train pass per stage.

    Each variant is {"name": str, "k": int, "recent_years": int | None}:
    combos are ranked by train Sharpe computed on the full train window
    (recent_years=None) or only its trailing `recent_years` years (the
    strategy itself still trains on the full window - only the selection
    metric window changes), and the top-k are kept. For k>1 the OOS result is the
    equal-weight mean of the k sleeves' net daily return series
    (portfolio-of-portfolios; each sleeve's costs are netted inside its own
    backtest, and cross-sleeve netting of overlapping trades could only
    lower real costs, so the combined series is conservatively implementable).
    k and recent_years must be pre-registered, not tuned on OOS.
    """
    candidates = grid(params_space)
    metric_windows = {v.get("recent_years") for v in variants}
    reports: dict[str, list] = {v["name"]: [] for v in variants}
    for i, st in enumerate(stages):
        evals = []  # (params, {recent_years_or_None: train sharpe})
        for p in candidates:
            W = make_weights(p, st.train_start, st.train_end)
            res = backtest.run(W, start=st.train_start, end=st.train_end)
            scores = {}
            for ry in metric_windows:
                r = res.ret
                if ry is not None:
                    r = r.loc[r.index >= pd.Timestamp(st.train_end) - pd.DateOffset(years=ry)]
                scores[ry] = _sharpe_of(r)
            evals.append((p, scores))

        oos_cache: dict[str, backtest.Result] = {}

        def _oos(p: dict) -> backtest.Result:
            key = str(sorted(p.items()))
            if key not in oos_cache:
                W = make_weights(p, st.train_start, st.test_end)
                oos_cache[key] = backtest.run(W, start=st.test_start, end=st.test_end)
            return oos_cache[key]

        for v in variants:
            ry = v.get("recent_years")
            ranked = sorted(
                (e for e in evals if np.isfinite(e[1][ry])),
                key=lambda e: e[1][ry],
                reverse=True,
            )
            picks = ranked[: v["k"]]
            sleeves = [_oos(p) for p, _ in picks]
            comb = pd.concat([s.ret for s in sleeves], axis=1).mean(axis=1)
            to = pd.concat([s.turnover for s in sleeves], axis=1).mean(axis=1)
            wsum = sum(s.weights for s in sleeves) / len(sleeves)
            pos = (wsum > 1e-6).sum(axis=1)
            nav = (1 + comb).cumprod()
            years = len(comb) / 252
            row = {
                "stage": i + 1,
                "test_start": st.test_start,
                "test_end": st.test_end,
                "params": " | ".join(str(p) for p, _ in picks),
                "train_sharpe": round(picks[0][1][ry], 3),
                "oos_sharpe": round(_sharpe_of(comb), 3),
                "oos_cagr": round(float(nav.iloc[-1] ** (1 / years) - 1), 4),
                "oos_maxdd": round(float((nav / nav.cummax() - 1).min()), 4),
                "ann_turnover": round(float(to.mean() * 252), 1),
                "avg_pos": round(float(pos[pos > 0].mean()), 1) if (pos > 0).any() else 0.0,
            }
            reports[v["name"]].append(row)
            if verbose:
                print(v["name"], row)
    out = {}
    for v in variants:
        rep = pd.DataFrame(reports[v["name"]])
        out[v["name"]] = rep
        if verbose and len(rep):
            n_ok = (rep.oos_sharpe > 2).sum()
            print(f"\n[{v['name']}] stages passed (Sharpe>2): {n_ok}/{len(rep)}  -> {'STOP CONDITION MET' if n_ok == len(rep) else 'not met'}")
    return out
