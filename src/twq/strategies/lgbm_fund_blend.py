"""Zero-retrain A/B (attempt 6): pre-2019 score = mean rank of fund + nomkt h21
panels instead of replacing nomkt with fund. See lgbm_h63_ensemble mix table."""

from . import lgbm_h63_ensemble as _ens


def make_weights(params: dict, start: str, end: str):
    return _ens.make_weights(dict(params, mix="blend21"), start, end)
