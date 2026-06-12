"""Vectorized daily portfolio backtester for Taiwan stocks.

Convention
----------
- A strategy produces a target weight matrix W (date x stock). A row with at
  least one finite value is a REBALANCE day: trade at that day's OPEN to the
  given weights (NaN cells = 0). An all-NaN row means hold and let weights
  drift with prices.
- W.loc[t] must be decided from information available up to the close of t-1
  (the caller is responsible for that shift).
- PnL day t = overnight return on yesterday's closing weights compounded with
  open->close return on today's (post-trade) weights.
- Costs: buys pay BUY_COST, sells pay SELL_COST on traded notional.
- Non-tradeable guards: an increase is skipped when the stock opens at/above
  its limit-up price or has no quote; a decrease is skipped at limit-down
  opens. The drifted previous weight is carried instead.
- Halted/missing days: position value carried flat (return 0), consistent with
  the value-preserving treatment of capital reductions.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data as D


@dataclass
class Result:
    nav: pd.Series
    ret: pd.Series  # daily net returns
    cost: pd.Series  # daily cost drag
    weights: pd.DataFrame  # post-trade weights for each day
    turnover: pd.Series  # one-sided traded notional / NAV

    def sharpe(self, ann: int = 252) -> float:
        r = self.ret.dropna()
        if len(r) < 20 or r.std() == 0:
            return np.nan
        return float(r.mean() / r.std() * np.sqrt(ann))

    def summary(self) -> dict:
        r = self.ret.dropna()
        if not len(r):
            return {}
        nav = (1 + r).cumprod()
        dd = (nav / nav.cummax() - 1).min()
        years = len(r) / 252
        cagr = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
        pos = (self.weights > 1e-6).sum(axis=1)
        return {
            "sharpe": round(self.sharpe(), 3),
            "cagr": round(float(cagr), 4),
            "maxdd": round(float(dd), 4),
            "ann_turnover": round(float(self.turnover.mean() * 252), 1),
            "avg_positions": round(float(pos[pos > 0].mean()), 1) if (pos > 0).any() else 0.0,
            "days": len(r),
        }


def run(
    target_w: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    buy_cost: float = D.BUY_COST,
    sell_cost: float = D.SELL_COST,
    limit_guard: bool = True,
) -> Result:
    """Backtest a target-weight matrix. Rows should sum to <= 1 (rest cash)."""
    adj = D.adj_ohlc()
    aopen, aclose = adj["open"], adj["close"]
    raw_open, raw_prev_close = D.load("open"), D.load("close").shift(1)

    W = target_w.reindex(index=aclose.index, columns=aclose.columns)
    if start:
        W = W.loc[W.index >= pd.Timestamp(start)]
    if end:
        W = W.loc[W.index <= pd.Timestamp(end)]
    idx, cols = W.index, W.columns

    ao = aopen.loc[idx].to_numpy()
    ac = aclose.loc[idx].to_numpy()
    ac_prev = aclose.shift(1).loc[idx].to_numpy()
    ro = raw_open.loc[idx].to_numpy()
    rpc = raw_prev_close.loc[idx].to_numpy()
    lim = D.limit_pct(idx)

    tradeable = np.isfinite(ro) & np.isfinite(ac)
    with np.errstate(invalid="ignore", divide="ignore"):
        ogap = ro / rpc - 1
        r_on = np.where(np.isfinite(ao) & np.isfinite(ac_prev), ao / ac_prev - 1, 0.0)
        r_id = np.where(np.isfinite(ao) & np.isfinite(ac), ac / ao - 1, 0.0)
    no_buy = ~tradeable | (limit_guard & np.isfinite(ogap) & (ogap >= lim[:, None] - 0.002))
    no_sell = ~tradeable | (limit_guard & np.isfinite(ogap) & (ogap <= -lim[:, None] + 0.002))

    Wt = W.to_numpy()
    rebal = np.isfinite(Wt).any(axis=1)
    Wt = np.nan_to_num(Wt, nan=0.0)

    T, N = Wt.shape
    held = np.zeros(N)  # weights at previous close
    rets = np.empty(T)
    costs = np.zeros(T)
    tos = np.zeros(T)
    realized = np.empty((T, N))

    for t in range(T):
        pv_on = 1.0 + held @ r_on[t]
        w_open = held * (1.0 + r_on[t]) / pv_on

        if rebal[t]:
            new_w = Wt[t].copy()
            inc = new_w > w_open + 1e-12
            dec = new_w < w_open - 1e-12
            new_w[inc & no_buy[t]] = w_open[inc & no_buy[t]]
            new_w[dec & no_sell[t]] = w_open[dec & no_sell[t]]
            dw = new_w - w_open
            buy_n = dw[dw > 0].sum()
            sell_n = -dw[dw < 0].sum()
            costs[t] = buy_n * buy_cost + sell_n * sell_cost
            tos[t] = buy_n + sell_n
        else:
            new_w = w_open

        pv_id = 1.0 + new_w @ r_id[t]
        rets[t] = pv_on * pv_id - 1.0 - costs[t]
        held = new_w * (1.0 + r_id[t]) / pv_id
        realized[t] = held

    ret = pd.Series(rets, index=idx)
    return Result(
        nav=(1 + ret).cumprod(),
        ret=ret,
        cost=pd.Series(costs, index=idx),
        weights=pd.DataFrame(realized, index=idx, columns=cols),
        turnover=pd.Series(tos, index=idx),
    )
