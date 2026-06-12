"""Rolling point-in-time LightGBM scorer for lgbm-rank-v1.

Expanding-window refits every --refit-weeks weekly dates from --first-score
onward. At each cutoff c_i the model trains only on samples whose label
window fully closed before c_i (y_end < c_i — exact embargo, no calendar
approximation), then scores all weekly dates in [c_i, c_{i+1}). Every score
in the output panel is therefore strictly out-of-sample, so downstream WFV
train-window selection on the panel stays leak-free.

Output: data/derived/lgbm_scores_h21.parquet (weekly trade days x stocks),
research/lgbm_feature_importance.csv, and per-segment Spearman rank-IC
diagnostics on stdout.

--no-market-feats (attempt 4, lgbm-stockonly) drops the 3 market-state
columns from X so the model learns a pure cross-sectional stock ranking;
regime/exposure is then handled by a portfolio overlay instead of tree
interactions (which caused wholesale book flips when a market feature
crossed a split threshold). Writes lgbm_scores_h21_nomkt.parquet.

Usage: uv run python scripts/train_lgbm_scores.py [--refresh-table] [--no-market-feats]
"""

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from twq import data as D  # noqa: E402
from twq import features_ml as FM  # noqa: E402

LGB_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 63,
    "seed": 42,
    "verbosity": -1,
}


def fit(X, y, feats, trees, device):
    p = dict(LGB_PARAMS, device=device)
    ds = lgb.Dataset(X, y, feature_name=feats)
    return lgb.train(p, ds, num_boost_round=trees)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-table", action="store_true")
    ap.add_argument("--refit-weeks", type=int, default=26)
    ap.add_argument("--first-score", default="2011-01-01")
    ap.add_argument("--trees", type=int, default=400)
    ap.add_argument("--no-market-feats", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    table = FM.build_weekly_table(refresh=args.refresh_table)
    feats = FM.feature_names(table)
    suffix = ""
    if args.no_market_feats:
        feats = [f for f in feats if f not in FM.MARKET_FEATS]
        suffix = "_nomkt"
    dates = np.sort(table["date"].unique())
    print(
        f"table: {len(table):,} rows, {len(feats)} feats, "
        f"{len(dates)} weeks {pd.Timestamp(dates[0]).date()}..{pd.Timestamp(dates[-1]).date()} "
        f"({time.time()-t0:.0f}s)"
    )

    first = np.searchsorted(dates, np.datetime64(args.first_score))
    cut_idx = list(range(first, len(dates), args.refit_weeks))
    cutoffs = [dates[i] for i in cut_idx]
    bounds = cutoffs + [np.datetime64("2100-01-01")]

    Xall = table[feats].to_numpy(dtype="float32")
    device = "gpu"
    scores = []
    ics = []
    imps = np.zeros(len(feats))
    for k, c in enumerate(cutoffs):
        tr = (table["y_end"].to_numpy() < c) & table["y_rank"].notna().to_numpy()
        seg = (table["date"].to_numpy() >= c) & (table["date"].to_numpy() < bounds[k + 1])
        if tr.sum() < 20000 or seg.sum() == 0:
            continue
        t1 = time.time()
        try:
            model = fit(Xall[tr], table["y_rank"].to_numpy()[tr], feats, args.trees, device)
        except lgb.basic.LightGBMError as e:
            print(f"GPU failed ({str(e)[:80]}), falling back to CPU")
            device = "cpu"
            model = fit(Xall[tr], table["y_rank"].to_numpy()[tr], feats, args.trees, device)
        imps += model.feature_importance("gain")

        sub = table.loc[seg, ["date", "stock", "y_rank"]].copy()
        sub["score"] = model.predict(Xall[seg])
        scores.append(sub)

        ic = np.nan
        val = sub.dropna(subset=["y_rank"])
        if len(val):
            g = val.groupby("date")
            sr = g["score"].rank(pct=True)
            ic = val.assign(sr=sr).groupby("date").apply(
                lambda x: x["sr"].corr(x["y_rank"])
            ).mean()
            ics.append(ic)
        print(
            f"refit {k+1}/{len(cutoffs)} cutoff={pd.Timestamp(c).date()} "
            f"n_train={tr.sum():,} seg_weeks={len(sub['date'].unique())} "
            f"seg_IC={ic:.4f} [{device}] ({time.time()-t1:.0f}s)"
        )

    allsc = pd.concat(scores)
    panel = allsc.pivot(index="date", columns="stock", values="score")
    panel = panel.reindex(columns=D.load("close").columns)
    out = D.DERIVED / f"lgbm_scores_h21{suffix}.parquet"
    panel.to_parquet(out)

    icv = pd.Series(ics)
    print(f"\nsegments={len(icv)} mean_IC={icv.mean():.4f} IC_IR(ann)={icv.mean()/icv.std()*np.sqrt(len(icv)/15):.2f}")
    fi = pd.Series(imps, index=feats).sort_values(ascending=False)
    fi.to_csv(Path(__file__).resolve().parent.parent / "research" / f"lgbm_feature_importance{suffix}.csv")
    print("top features by gain:")
    print((fi / fi.sum()).head(12).round(4).to_string())
    print(f"saved {out}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
