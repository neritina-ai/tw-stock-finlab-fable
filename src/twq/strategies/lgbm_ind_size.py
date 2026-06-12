"""Attempt-9 A/B variant: industry-relative all4 ranking plus weekly
cross-sectional size residualization (ln 20d mean dollar turnover).
See lgbm_ind_neutral."""

from . import lgbm_ind_neutral as _ind


def make_weights(params: dict, start: str, end: str):
    return _ind.make_weights(dict(params, neutral="indsz"), start, end)
