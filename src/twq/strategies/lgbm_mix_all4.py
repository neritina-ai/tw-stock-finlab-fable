"""Attempt-6 variant: pre-2019 score = mean rank of all four panels
(fund/nomkt x h21/h63), post-2019 = mean rank of nomkt h21/h63.
See lgbm_h63_ensemble mix table."""

from . import lgbm_h63_ensemble as _ens


def make_weights(params: dict, start: str, end: str):
    return _ens.make_weights(dict(params, mix="all4"), start, end)
