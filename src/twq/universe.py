"""Shared tradable-universe definition.

Mirrors the filter used by band_voltarget (20d median turnover, price floor,
listing age, no 91xx TDR) and adds the 00-prefix ETF exclusion that attempt 2
flagged (0056 slipped into the book and diluted stock-selection alpha).

`eligible()` is shifted by one day: row d answers "tradable at d's open using
info through d-1", matching how strategies and the ML sampler consume it.
"""

from functools import lru_cache

import pandas as pd

from . import data as D


@lru_cache(maxsize=None)
def eligible(
    min_turnover_ntd: float = 20e6,
    min_close: float = 10.0,
    min_listed_days: int = 100,
    exclude_etf: bool = True,
) -> pd.DataFrame:
    raw_close = D.load("close")
    turnover = D.load("turnover")

    listed_days = raw_close.notna().rolling(130).sum()
    uni = (
        (turnover.rolling(20, min_periods=10).median() > min_turnover_ntd)
        & (raw_close > min_close)
        & (listed_days >= min_listed_days)
    )
    uni.loc[:, raw_close.columns.str.startswith("91")] = False  # TDR
    if exclude_etf:
        uni.loc[:, raw_close.columns.str.startswith("00")] = False
    return uni.shift(1).fillna(False).astype(bool)
