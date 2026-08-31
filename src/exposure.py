"""Residual-exposure experiment, separated from the core simulator."""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from model import BASE_PARAMS, CV_GRID, Params, exposure_bound, s_of_cv


def exposure_experiment(cvs=CV_GRID, samples: int = 20000, seed: int = 1, p: Params = BASE_PARAMS) -> List[Dict[str, float | str]]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float | str]] = []
    for cv in cvs:
        s = float(s_of_cv(float(cv), p))
        vmin = rng.uniform(0.2, 0.8, samples)
        vmax = vmin * s
        p_geo = np.sqrt(vmin * vmax)
        p_mean = 0.5 * (vmin + vmax)
        geo = np.maximum(p_geo / vmin, vmax / p_geo)
        mean_rule = np.maximum(p_mean / vmin, vmax / p_mean)
        rows.append({"mechanism": "geometric", "x_key": "cv", "x": float(cv), "seed_mean": float(geo.mean()), "ci95": 0.0, "exposure": float(geo.mean()), "analytic": float(exposure_bound(float(cv), p))})
        rows.append({"mechanism": "naive_mean", "x_key": "cv", "x": float(cv), "seed_mean": float(mean_rule.mean()), "ci95": 0.0, "exposure": float(mean_rule.mean()), "analytic": float(exposure_bound(float(cv), p))})
    return rows


def exposure_spread_experiment(spreads=(1.5, 2.0, 3.0, 4.0, 6.0, 8.0)) -> List[Dict[str, float | str]]:
    """Evaluate the two pricing rules exactly along the fiber-spread axis."""
    rows: List[Dict[str, float | str]] = []
    for spread in spreads:
        spread = float(spread)
        analytic = math.sqrt(spread)
        rows.append({"spread": spread, "rule": "geometric", "exposure": analytic, "analytic": analytic})
        rows.append({"spread": spread, "rule": "naive_mean", "exposure": (1.0 + spread) / 2.0, "analytic": analytic})
    return rows


def exposure_error_experiment(epsilons=tuple(i * 0.05 for i in range(13)), cv: float = 1.0, p: Params = BASE_PARAMS) -> List[Dict[str, float | str]]:
    """Exact evaluation of the degraded bound under verifier false negatives.

    Manuscript (Assumption, blue text for referee M3): with false-negative rate
    epsilon, the effective spread is s_eps = s(cv) + eps * (s0 - s(cv)) and the
    exposure bound degrades to sqrt(s_eps); geometric-mean pricing on the
    effective fiber attains it, the arithmetic rule exceeds it.
    """
    rows: List[Dict[str, float | str]] = []
    s_cv = float(s_of_cv(cv, p))
    for eps in epsilons:
        eps = float(eps)
        s_eff = s_cv + eps * (p.S0 - s_cv)
        analytic = math.sqrt(s_eff)
        rows.append({"epsilon": eps, "cv": cv, "rule": "geometric", "exposure": analytic, "analytic": analytic})
        rows.append({"epsilon": eps, "cv": cv, "rule": "naive_mean", "exposure": (1.0 + s_eff) / 2.0, "analytic": analytic})
    return rows
