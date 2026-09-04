"""Core simulation model for the Verifiable Agent Economy experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Dict, Iterable, List

import numpy as np


@dataclass(frozen=True)
class Params:
    V_HIGH: float = 1.0
    V_LOW: float = 0.25
    KAP_HIGH: float = 0.40
    KAP_LOW: float = 0.10
    SIGMA: float = 1.0
    STAKE: float = 0.60
    GAMMA: float = 0.50
    DELTA: float = 0.95
    K_DET: float = 1.0
    ALPHA_CV: float = 0.05
    COORD: float = 0.92
    MATCH_FACTOR: float = 0.15
    # Shirking-gain population: heavier tail (G_SD, G_HI raised 2026-07-15) so that
    # deterrence saturates gradually across agents rather than all at once; this keeps
    # the honesty and welfare curves smooth instead of piecewise-linear.
    G_MEAN: float = 0.30
    G_SD: float = 0.20
    G_LO: float = 0.05
    G_HI: float = 1.20
    OVERHEAD_DISPUTE: float = 0.05
    DRT_C_MIN: float = 0.10

    @property
    def S0(self) -> float:
        return self.V_HIGH / self.V_LOW

    @property
    def P_PAY(self) -> float:
        return math.sqrt(self.V_HIGH * self.V_LOW)

    @property
    def BASE_PROFIT(self) -> float:
        return self.P_PAY - self.KAP_HIGH

    @property
    def C_CONT(self) -> float:
        return (self.DELTA / (1.0 - self.DELTA)) * self.BASE_PROFIT * self.MATCH_FACTOR

    def as_dict(self) -> Dict[str, float]:
        out = self.__dict__.copy()
        out.update(
            {
                "S0": self.S0,
                "P_PAY": self.P_PAY,
                "BASE_PROFIT": self.BASE_PROFIT,
                "C_CONT": self.C_CONT,
            }
        )
        return out


BASE_PARAMS = Params()
CV_GRID = np.linspace(0.0, 3.0, 13)
LEARNING_CV_GRID = np.linspace(0.0, 3.0, 13)
SYBIL_GRID = np.linspace(0.0, 0.8, 9)
STAKE_GRID = np.array([i * 0.05 for i in range(13)] + [0.8, 1.0, 1.2])
MECHANISMS = ["Mstar", "no_reputation", "centralized", "receipt_only"]
THEORY_MECHANISMS = ["Mstar", "receipt_only", "no_reputation", "centralized"]
LEARNING_MECHANISMS = ["Mstar", "no_reputation", "receipt_only"]


def mech_config(mech: str, p: Params = BASE_PARAMS) -> Dict[str, float | bool]:
    configs = {
        "Mstar": dict(verify=True, rho_rep=1.0, stake=p.STAKE, gamma=p.GAMMA, coord=1.0),
        "receipt_only": dict(verify=False, rho_rep=1.0, stake=p.STAKE, gamma=p.GAMMA, coord=1.0),
        "no_reputation": dict(verify=True, rho_rep=0.0, stake=0.0, gamma=0.0, coord=1.0),
        "centralized": dict(verify=True, rho_rep=0.0, stake=0.0, gamma=0.0, coord=p.COORD),
        "no_stake": dict(verify=True, rho_rep=1.0, stake=0.0, gamma=0.0, coord=1.0),
        "stake_only": dict(verify=True, rho_rep=0.0, stake=p.STAKE, gamma=p.GAMMA, coord=1.0),
    }
    return configs[mech]


def s_of_cv(cv: float | np.ndarray, p: Params = BASE_PARAMS) -> float | np.ndarray:
    return 1.0 + (p.S0 - 1.0) * np.exp(-p.K_DET * cv)


def p_detect(cv: float, verify: bool, p: Params = BASE_PARAMS) -> float:
    if not verify:
        return 0.0
    return 1.0 - math.exp(-p.K_DET * cv)


def exposure_bound(cv: float | np.ndarray, p: Params = BASE_PARAMS) -> float | np.ndarray:
    return np.sqrt(s_of_cv(cv, p))


def shirk_decision(g: np.ndarray, pd: float, cfg: Dict[str, float | bool], whitewasher: np.ndarray, p: Params) -> np.ndarray:
    normal_deter = pd * (p.P_PAY + p.SIGMA * float(cfg["stake"])) + pd * float(cfg["rho_rep"]) * p.C_CONT
    ww_deter = pd * (p.P_PAY + p.SIGMA * float(cfg["stake"]) + float(cfg["gamma"]))
    deter = np.where(whitewasher, ww_deter, normal_deter)
    return g > deter


def simulate_run(mech: str, cv: float, sybil_frac: float = 0.0, n: int = 50, T: int = 4000, seed: int = 0, p: Params = BASE_PARAMS) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    cfg = mech_config(mech, p)
    pd = p_detect(cv, bool(cfg["verify"]), p)
    vcost = p.ALPHA_CV * cv if bool(cfg["verify"]) else 0.0

    g = np.clip(rng.normal(p.G_MEAN, p.G_SD, n), p.G_LO, p.G_HI)
    is_ww = rng.random(n) < sybil_frac
    shirks_by_agent = shirk_decision(g, pd, cfg, is_ww, p)

    choices = rng.integers(n, size=T)
    shirk = shirks_by_agent[choices]
    detected = shirk & (rng.random(T) < pd)
    value = np.where(shirk, p.V_LOW, p.V_HIGH)
    cost = np.where(shirk, p.KAP_LOW, p.KAP_HIGH)
    welfare = float(cfg["coord"]) * value - cost - vcost
    return {
        "welfare": float(welfare.mean()),
        "honesty": float((~shirk).mean()),
        "honesty_ci_source": float((~shirk).mean()),
        "dispute": float(detected.mean()),
        "shirk": float(shirk.mean()),
        "undetected_shortfall": float((shirk & ~detected).mean()),
    }


def aggregate_records(records: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted(records[0].keys())
    out: Dict[str, float] = {}
    for key in keys:
        vals = np.array([r[key] for r in records], dtype=float)
        out[key] = float(vals.mean())
        out[f"{key}_ci95"] = float(1.96 * vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return out


def sweep_cv(mechanisms: Iterable[str], cvs: Iterable[float] = CV_GRID, seeds: int = 100, n: int = 50, T: int = 4000, p: Params = BASE_PARAMS) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for mech in mechanisms:
        for cv in cvs:
            records = [simulate_run(mech, float(cv), n=n, T=T, seed=s, p=p) for s in range(seeds)]
            agg = aggregate_records(records)
            rows.append({"mechanism": mech, "x_key": "cv", "x": float(cv), "seed_mean": agg["welfare"], "ci95": agg["welfare_ci95"], **agg})
    return rows


def sweep_sybil(fracs: Iterable[float] = SYBIL_GRID, seeds: int = 100, n: int = 50, T: int = 4000, p: Params = BASE_PARAMS, cv: float = 1.0) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for mech in ["Mstar", "no_stake", "no_reputation"]:
        for frac in fracs:
            records = [simulate_run(mech, cv, sybil_frac=float(frac), n=n, T=T, seed=s, p=p) for s in range(seeds)]
            agg = aggregate_records(records)
            rows.append({"mechanism": mech, "x_key": "sybil", "x": float(frac), "seed_mean": agg["welfare"], "ci95": agg["welfare_ci95"], **agg})
    return rows


def gain_cdf(value: float, p: Params = BASE_PARAMS) -> float:
    """CDF of the clipped-normal shirking gain used by the simulator."""
    if value < p.G_LO:
        return 0.0
    if value >= p.G_HI:
        return 1.0
    z = (value - p.G_MEAN) / (p.G_SD * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def sweep_stake(stakes: Iterable[float] = STAKE_GRID, seeds: int = 100, cv: float = 0.5, n: int = 50, T: int = 4000, p: Params = BASE_PARAMS) -> List[Dict[str, float | str]]:
    """Honesty vs admission stake for the full mechanism and two ablation baselines.

    stake_only removes the reputation continuation (stake is the only deterrent);
    receipt_only shows that a stake deters nothing when there is no verification.
    """
    rows: List[Dict[str, float | str]] = []
    for mech in ["Mstar", "stake_only", "receipt_only"]:
        for stake in stakes:
            stake = float(stake)
            variant = replace(p, STAKE=stake)
            cfg = mech_config(mech, variant)
            pdm = p_detect(cv, bool(cfg["verify"]), variant)
            records = [simulate_run(mech, cv, n=n, T=T, seed=s, p=variant) for s in range(seeds)]
            agg = aggregate_records(records)
            deterrence = pdm * (variant.P_PAY + variant.SIGMA * float(cfg["stake"])) + pdm * float(cfg["rho_rep"]) * variant.C_CONT
            rows.append(
                {
                    "mechanism": mech,
                    "x_key": "stake",
                    "x": stake,
                    "seed_mean": agg["honesty"],
                    "ci95": agg["honesty_ci95"],
                    "analytic": gain_cdf(deterrence, variant),
                    **agg,
                }
            )
    return rows


DELTA_GRID = np.array([0.80, 0.84, 0.87, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99])


def sweep_delta(deltas: Iterable[float] = DELTA_GRID, seeds: int = 100, cv: float = 0.5, n: int = 50, T: int = 4000, p: Params = BASE_PARAMS) -> List[Dict[str, float | str]]:
    """Honesty vs discount factor (agent patience) — the continuation-value lever of Prop. IC.

    stake_only has no delta dependence (reputation carries no continuation value
    there), so it plots flat: patience matters only through reputation.
    """
    rows: List[Dict[str, float | str]] = []
    for mech in ["Mstar", "stake_only"]:
        for delta in deltas:
            delta = float(delta)
            variant = replace(p, DELTA=delta)
            cfg = mech_config(mech, variant)
            pdm = p_detect(cv, bool(cfg["verify"]), variant)
            records = [simulate_run(mech, cv, n=n, T=T, seed=s, p=variant) for s in range(seeds)]
            agg = aggregate_records(records)
            deterrence = pdm * (variant.P_PAY + variant.SIGMA * float(cfg["stake"])) + pdm * float(cfg["rho_rep"]) * variant.C_CONT
            rows.append(
                {
                    "mechanism": mech,
                    "x_key": "delta",
                    "x": delta,
                    "seed_mean": agg["honesty"],
                    "ci95": agg["honesty_ci95"],
                    "analytic": gain_cdf(deterrence, variant),
                    **agg,
                }
            )
    return rows


def simulate_learning_run(mech: str, cv: float, n: int = 40, T: int = 8000, seed: int = 0, warmup: float = 0.6, p: Params = BASE_PARAMS, trace: bool = False) -> Dict[str, float | List[float]]:
    rng = np.random.default_rng(1000 + seed)
    cfg = mech_config(mech, p)
    pd = p_detect(cv, bool(cfg["verify"]), p)
    vcost = p.ALPHA_CV * cv if bool(cfg["verify"]) else 0.0
    use_rep = float(cfg["rho_rep"]) > 0.0
    Q = np.zeros((n, 2))
    R = np.zeros(n)
    alpha, eta, beta = 0.1, 0.05, 3.0
    hon_acc = 0.0
    w_acc = 0.0
    dispute_acc = 0.0
    cnt = 0
    start = int(warmup * T)
    honesty_trace: List[float] = []
    for t in range(T):
        eps = max(0.02, 0.30 * math.exp(-3.0 * t / T))
        if use_rep:
            weights = np.exp(beta * R)
            i = int(rng.choice(n, p=weights / weights.sum()))
        else:
            i = int(rng.integers(n))
        action = int(rng.integers(2)) if rng.random() < eps else int(Q[i, 1] > Q[i, 0])
        if action == 0:
            reward = p.P_PAY - p.KAP_HIGH
            value = p.V_HIGH
            good = True
            detected = False
        else:
            detected = bool(cfg["verify"]) and (rng.random() < pd)
            if detected:
                reward = -p.KAP_LOW - p.SIGMA * float(cfg["stake"])
                good = False
            else:
                reward = p.P_PAY - p.KAP_LOW
                good = True
            value = p.V_LOW
        Q[i, action] += alpha * (reward - Q[i, action])
        R[i] = min(1.0, R[i] + eta) if good else max(0.0, R[i] - eta)
        honest_now = 1.0 if action == 0 else 0.0
        if trace:
            honesty_trace.append(honest_now)
        if t >= start:
            hon_acc += honest_now
            w_acc += float(cfg["coord"]) * value - p.KAP_HIGH * (action == 0) - p.KAP_LOW * (action == 1) - vcost
            dispute_acc += 1.0 if detected else 0.0
            cnt += 1
    out: Dict[str, float | List[float]] = {
        "welfare": w_acc / cnt,
        "honesty": hon_acc / cnt,
        "dispute": dispute_acc / cnt,
    }
    if trace:
        out["honesty_trace"] = honesty_trace
    return out


def sweep_learning(mechanisms: Iterable[str] = LEARNING_MECHANISMS, cvs: Iterable[float] = LEARNING_CV_GRID, seeds: int = 50, n: int = 40, T: int = 8000, p: Params = BASE_PARAMS) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for mech in mechanisms:
        for cv in cvs:
            records = [simulate_learning_run(mech, float(cv), n=n, T=T, seed=s, p=p) for s in range(seeds)]
            agg = aggregate_records(records)  # type: ignore[arg-type]
            rows.append({"mechanism": mech, "x_key": "cv", "x": float(cv), "seed_mean": agg["welfare"], "ci95": agg["welfare_ci95"], **agg})
    return rows


def param_variant(name: str, value: float, base: Params = BASE_PARAMS) -> Params:
    if name == "s0":
        return replace(base, V_LOW=1.0 / value)
    return replace(base, **{name: value})


# --- Market thickness ---------------------------------------------------------
# Continuation value scales with how much a reputation loss costs an agent in
# future matching. MATCH_FACTOR is that scale: 0.0 is a monopolist service agent
# whose reputation loss forfeits no future business, larger values are thicker
# markets with substitutes ready to displace a downgraded agent.
THICKNESS_GRID = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30])
THICKNESS_STAKES = [0.0, 0.30, 0.60, 1.00]


def deterrence_at(stake: float, cv: float, match_factor: float, use_rep: bool = True, p: Params = BASE_PARAMS) -> float:
    """Total per-deviation deterrence: forfeited payment + slashed stake + continuation loss."""
    variant = replace(p, STAKE=stake, MATCH_FACTOR=match_factor)
    pdm = p_detect(cv, True, variant)
    cont = variant.C_CONT if use_rep else 0.0
    return pdm * (variant.P_PAY + variant.SIGMA * stake) + pdm * cont


def sweep_thickness(thicknesses: Iterable[float] = THICKNESS_GRID, stakes: Iterable[float] = THICKNESS_STAKES,
                    seeds: int = 100, cv: float = 0.5, n: int = 50, T: int = 4000,
                    p: Params = BASE_PARAMS) -> List[Dict[str, float | str]]:
    """Honesty vs market thickness at several admission stakes (thin markets)."""
    rows: List[Dict[str, float | str]] = []
    for stake in stakes:
        for mf in thicknesses:
            stake = float(stake); mf = float(mf)
            variant = replace(p, STAKE=stake, MATCH_FACTOR=mf)
            records = [simulate_run("Mstar", cv, n=n, T=T, seed=s, p=variant) for s in range(seeds)]
            agg = aggregate_records(records)
            rows.append({
                "mechanism": f"stake={stake:.2f}",
                "x_key": "match_factor",
                "x": mf,
                "stake": stake,
                "seed_mean": agg["honesty"],
                "ci95": agg["honesty_ci95"],
                "analytic": gain_cdf(deterrence_at(stake, cv, mf, True, p), variant),
                "deterrence": deterrence_at(stake, cv, mf, True, p),
                **agg,
            })
    return rows


def required_stake(cv: float, match_factor: float, target: float = 0.95, p: Params = BASE_PARAMS,
                   hi: float = 20.0, tol: float = 1e-4) -> float:
    """Smallest admission stake whose deterrence holds the analytic honest fraction at `target`."""
    lo = 0.0
    if gain_cdf(deterrence_at(hi, cv, match_factor, True, p), replace(p, STAKE=hi, MATCH_FACTOR=match_factor)) < target:
        return float("nan")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        v = gain_cdf(deterrence_at(mid, cv, match_factor, True, p), replace(p, STAKE=mid, MATCH_FACTOR=match_factor))
        if v >= target:
            hi = mid
        else:
            lo = mid
    return hi


# --- Verification families and false positives --------------------------------
# The base model uses d(cv)=1-exp(-K cv) and s(cv)=1+(s0-1)exp(-K cv).
# These are one family: residual spread tracks the undetected mass, so that
# s(cv) = 1 + (s0-1)*(1-d(cv)) holds identically. Alternative detection laws are
# therefore obtained by replacing d and deriving s through the same coupling,
# rather than by positing an unrelated second curve.
VERIFIER_FAMILIES = ["exponential", "linear", "logarithmic", "threshold", "power"]
PHI_GRID = np.linspace(0.0, 0.30, 13)


def d_family(cv: float | np.ndarray, family: str = "exponential", p: Params = BASE_PARAMS,
             c0: float = 0.75, d_hi: float = 0.85) -> float | np.ndarray:
    """Detection probability under alternative verification technologies."""
    k = p.K_DET
    cv = np.asarray(cv, dtype=float)
    if family == "exponential":
        out = 1.0 - np.exp(-k * cv)
    elif family == "linear":
        out = np.minimum(1.0, k * cv / 3.0)
    elif family == "logarithmic":
        out = np.minimum(1.0, k * np.log1p(cv) / np.log1p(3.0))
    elif family == "threshold":
        out = np.where(cv < c0, 0.0, d_hi)
    elif family == "power":
        out = 1.0 - np.power(1.0 + cv, -k)
    else:
        raise ValueError(family)
    return float(out) if out.ndim == 0 else out


def s_from_d(d: float | np.ndarray, p: Params = BASE_PARAMS) -> float | np.ndarray:
    """Residual spread implied by detection strength under the family coupling."""
    return 1.0 + (p.S0 - 1.0) * (1.0 - np.asarray(d, dtype=float))


def ic_slack_fp(stake: float, cv: float, phi: float, match_factor: float | None = None,
                family: str = "exponential", p: Params = BASE_PARAMS) -> float:
    """Deterrence available under false-positive rate phi (the d-phi differential)."""
    mf = p.MATCH_FACTOR if match_factor is None else match_factor
    variant = replace(p, STAKE=stake, MATCH_FACTOR=mf)
    d = d_family(cv, family, variant)
    return max(0.0, d - phi) * (variant.P_PAY + variant.SIGMA * stake + variant.C_CONT)


def honest_payoff_fp(stake: float, phi: float, match_factor: float | None = None,
                     p: Params = BASE_PARAMS) -> float:
    """Expected payoff to a compliant agent when honest work is wrongly flagged at rate phi."""
    mf = p.MATCH_FACTOR if match_factor is None else match_factor
    variant = replace(p, STAKE=stake, MATCH_FACTOR=mf)
    return variant.P_PAY - variant.KAP_HIGH - phi * (variant.P_PAY + variant.SIGMA * stake + variant.C_CONT)


def max_phi_participation(stake: float, match_factor: float | None = None, p: Params = BASE_PARAMS) -> float:
    """Largest false-positive rate a compliant agent tolerates before exiting."""
    mf = p.MATCH_FACTOR if match_factor is None else match_factor
    variant = replace(p, STAKE=stake, MATCH_FACTOR=mf)
    return (variant.P_PAY - variant.KAP_HIGH) / (variant.P_PAY + variant.SIGMA * stake + variant.C_CONT)


def simulate_run_fp(mech: str, cv: float, phi: float = 0.0, family: str = "exponential",
                    n: int = 50, T: int = 4000, seed: int = 0, p: Params = BASE_PARAMS) -> Dict[str, float]:
    """simulate_run with a false-positive verifier and a selectable detection family.

    Honest deliveries are wrongly flagged with probability phi, which refunds the escrow
    and slashes the stake exactly as a true detection does. Compliant agents whose expected
    payoff turns negative exit the market rather than continue to participate.
    """
    rng = np.random.default_rng(seed)
    cfg = mech_config(mech, p)
    verify = bool(cfg["verify"])
    d = d_family(cv, family, p) if verify else 0.0
    phi_eff = phi if verify else 0.0
    vcost = p.ALPHA_CV * cv if verify else 0.0

    g = np.clip(rng.normal(p.G_MEAN, p.G_SD, n), p.G_LO, p.G_HI)
    deter = max(0.0, d - phi_eff) * (p.P_PAY + p.SIGMA * float(cfg["stake"]) + float(cfg["rho_rep"]) * p.C_CONT)
    shirks_by_agent = g > deter

    participates = honest_payoff_fp(float(cfg["stake"]), phi_eff, p=p) >= 0.0
    if not participates:
        # compliant agents withdraw; only agents that intended to shirk remain
        active = shirks_by_agent
        if not active.any():
            return {"welfare": 0.0, "honesty": 0.0, "dispute": 0.0, "shirk": 0.0,
                    "undetected_shortfall": 0.0, "participation": 0.0, "wrongful_slash": 0.0}
    else:
        active = np.ones(n, dtype=bool)

    idx = np.flatnonzero(active)
    choices = idx[rng.integers(len(idx), size=T)]
    shirk = shirks_by_agent[choices]
    detected = shirk & (rng.random(T) < d)
    wrongful = (~shirk) & (rng.random(T) < phi_eff)
    value = np.where(shirk, p.V_LOW, p.V_HIGH)
    cost = np.where(shirk, p.KAP_LOW, p.KAP_HIGH)
    welfare = float(cfg["coord"]) * value - cost - vcost
    return {
        "welfare": float(welfare.mean()),
        "honesty": float((~shirk).mean()),
        "dispute": float(detected.mean()),
        "shirk": float(shirk.mean()),
        "undetected_shortfall": float((shirk & ~detected).mean()),
        "participation": float(active.mean()),
        "wrongful_slash": float(wrongful.mean()),
    }


def sweep_family(families: Iterable[str] = VERIFIER_FAMILIES, cvs: Iterable[float] = CV_GRID,
                 seeds: int = 100, n: int = 50, T: int = 4000, p: Params = BASE_PARAMS):
    """Welfare vs verification budget under each detection family."""
    rows = []
    for fam in families:
        for cv in cvs:
            recs = [simulate_run_fp("Mstar", float(cv), 0.0, fam, n=n, T=T, seed=s, p=p) for s in range(seeds)]
            agg = aggregate_records(recs)
            rows.append({"mechanism": fam, "x_key": "cv", "x": float(cv),
                         "seed_mean": agg["welfare"], "ci95": agg["welfare_ci95"],
                         "d": d_family(float(cv), fam, p), "s": float(s_from_d(d_family(float(cv), fam, p), p)), **agg})
    return rows


def sweep_false_positive(phis: Iterable[float] = PHI_GRID, cv: float = 1.0, seeds: int = 100,
                         n: int = 50, T: int = 4000, p: Params = BASE_PARAMS):
    """Welfare, honesty, participation, and wrongful slashing vs false-positive rate."""
    rows = []
    for phi in phis:
        recs = [simulate_run_fp("Mstar", cv, float(phi), "exponential", n=n, T=T, seed=s, p=p) for s in range(seeds)]
        agg = aggregate_records(recs)
        rows.append({"mechanism": "Mstar", "x_key": "phi", "x": float(phi),
                     "seed_mean": agg["welfare"], "ci95": agg["welfare_ci95"],
                     "honest_payoff": honest_payoff_fp(p.STAKE, float(phi), p=p),
                     "phi_max": max_phi_participation(p.STAKE, p=p), **agg})
    return rows


def simulate_run_measured(cv: float, d: float, phi: float, n: int = 50, T: int = 4000,
                          seed: int = 0, p: Params = BASE_PARAMS) -> Dict[str, float]:
    """Welfare/honesty under externally measured detection and false-positive rates."""
    rng = np.random.default_rng(seed)
    vcost = p.ALPHA_CV * cv
    g = np.clip(rng.normal(p.G_MEAN, p.G_SD, n), p.G_LO, p.G_HI)
    deter = max(0.0, d - phi) * (p.P_PAY + p.SIGMA * p.STAKE + p.C_CONT)
    shirks = g > deter
    choices = rng.integers(n, size=T)
    shirk = shirks[choices]
    value = np.where(shirk, p.V_LOW, p.V_HIGH)
    cost = np.where(shirk, p.KAP_LOW, p.KAP_HIGH)
    welfare = value - cost - vcost
    return {"welfare": float(welfare.mean()), "honesty": float((~shirk).mean()),
            "shirk": float(shirk.mean())}


def sweep_measured_verifier(measured_rows, seeds: int = 100, n: int = 50, T: int = 4000,
                            p: Params = BASE_PARAMS):
    """Welfare vs verification budget using the measured (d, phi) of the audit verifier."""
    rows = []
    for m in measured_rows:
        cv, d, phi = float(m["cv"]), float(m["d_measured"]), float(m["phi_measured"])
        recs = [simulate_run_measured(cv, d, phi, n=n, T=T, seed=s, p=p) for s in range(seeds)]
        agg = aggregate_records(recs)
        rows.append({"mechanism": "measured_audit", "x_key": "cv", "x": cv,
                     "seed_mean": agg["welfare"], "ci95": agg["welfare_ci95"],
                     "d_measured": d, "phi_measured": phi,
                     "s_measured": float(m["s_measured"]), "k_tests": float(m["k_tests"]), **agg})
    return rows


# --- Matching-rule ablation ---------------------------------------------------
# The analytical runs select the transacting agent uniformly, so the matching rule
# mu* is not exercised there. This path makes selection endogenous: agents post
# heterogeneous prices, the mechanism selects under a stated rule, and reputation
# feeds back into future selection. Continuation value is therefore produced by the
# matching rule rather than supplied as a constant.
MATCHING_RULES = ["score", "reputation_only", "price_only", "random"]


def simulate_matching_run(rule: str = "score", cv: float = 1.0, n: int = 40, T: int = 8000,
                          seed: int = 0, warmup: float = 0.6, lam: float = 1.0, n_bid: int = 5,
                          stake: float | None = None, p: Params = BASE_PARAMS) -> Dict[str, float]:
    """Q-learning agents under an explicit matching rule.

    Each period a random subset of `n_bid` agents is admissible for the task, reflecting
    task-specific capability, and the rule selects within that subset. Without a per-period
    candidate set a deterministic argmax would award every task to one agent and the rules
    would be indistinguishable.
    """
    rng = np.random.default_rng(5000 + seed)
    S = p.STAKE if stake is None else stake
    pd = p_detect(cv, True, p)
    vcost = p.ALPHA_CV * cv
    prices = np.clip(rng.normal(p.P_PAY, 0.08, n), 0.20, 0.95)
    Q = np.zeros((n, 2))
    R = np.zeros(n)
    eta, alpha = 0.05, 0.1
    start = int(warmup * T)
    hon = w = cnt = 0.0
    sel_counts = np.zeros(n)

    for t in range(T):
        eps = max(0.02, 0.30 * math.exp(-3.0 * t / T))
        cand = rng.choice(n, size=min(n_bid, n), replace=False)
        if rule == "score":
            score = R[cand] - lam * prices[cand]
        elif rule == "reputation_only":
            score = R[cand]
        elif rule == "price_only":
            score = -prices[cand]
        elif rule == "random":
            score = rng.random(len(cand))
        else:
            raise ValueError(rule)
        i = int(cand[int(np.argmax(score + rng.normal(0.0, 1e-9, len(cand))))])
        sel_counts[i] += 1

        action = int(rng.integers(2)) if rng.random() < eps else int(Q[i, 1] > Q[i, 0])
        if action == 0:
            reward = prices[i] - p.KAP_HIGH
            value, good = p.V_HIGH, True
        else:
            detected = rng.random() < pd
            if detected:
                reward = -p.KAP_LOW - p.SIGMA * S
                value, good = p.V_LOW, False
            else:
                reward = prices[i] - p.KAP_LOW
                value, good = p.V_LOW, True
        Q[i, action] += alpha * (reward - Q[i, action])
        R[i] = min(1.0, R[i] + eta) if good else max(0.0, R[i] - eta)

        if t >= start:
            hon += 1.0 if action == 0 else 0.0
            w += value - (p.KAP_HIGH if action == 0 else p.KAP_LOW) - vcost
            cnt += 1

    shares = sel_counts / sel_counts.sum()
    return {"welfare": w / cnt, "honesty": hon / cnt,
            "selection_concentration": float((shares ** 2).sum()),
            "mean_price_paid": float((prices * shares).sum()),
            "mean_selected_reputation": float((R * shares).sum())}


def sweep_matching(rules: Iterable[str] = MATCHING_RULES, cv: float = 1.0, seeds: int = 40,
                   n: int = 40, T: int = 8000, p: Params = BASE_PARAMS):
    rows = []
    for rule in rules:
        recs = [simulate_matching_run(rule, cv, n=n, T=T, seed=s, p=p) for s in range(seeds)]
        agg = aggregate_records(recs)
        rows.append({"mechanism": rule, "x_key": "rule", "x": 0.0,
                     "seed_mean": agg["welfare"], "ci95": agg["welfare_ci95"], **agg})
    return rows


def ablation_table(cv: float = 1.0, seeds: int = 100, n: int = 50, T: int = 4000,
                   p: Params = BASE_PARAMS):
    """One consolidated component ablation at a fixed verification budget."""
    variants = [
        ("Mstar", "Full mechanism"),
        ("no_stake", "Verification + reputation, no staking"),
        ("stake_only", "Verification + staking, no reputation"),
        ("no_reputation", "Verification only, open matching"),
        ("centralized", "Verification only, centralized routing"),
        ("receipt_only", "Receipt only"),
    ]
    rows = []
    for mech, label in variants:
        recs = [simulate_run(mech, cv, n=n, T=T, seed=s, p=p) for s in range(seeds)]
        agg = aggregate_records(recs)
        cfg = mech_config(mech, p)
        rows.append({"mechanism": mech, "label": label,
                     "verification": bool(cfg["verify"]), "stake": float(cfg["stake"]),
                     "reputation": float(cfg["rho_rep"]) > 0.0, "routing": float(cfg["coord"]),
                     "welfare": agg["welfare"], "welfare_ci95": agg["welfare_ci95"],
                     "honesty": agg["honesty"], "undetected_shortfall": agg["undetected_shortfall"]})
    return rows


def measure_selection_sensitivity(rule: str = "score", n: int = 40, T: int = 40000, lam: float = 1.0,
                                  n_bid: int = 5, eta: float = 0.05, seed: int = 0,
                                  p: Params = BASE_PARAMS) -> Dict[str, float]:
    """Measure how much selection an agent forfeits when its reputation is downgraded.

    This is the empirical counterpart of Delta_i in Proposition 1. One focal agent is held
    at reputation R_focal while all others sit at 1.0; the market runs under `rule` and we
    record how often the focal agent is selected. Comparing an undamaged focal agent with a
    slashed one gives the per-period selection loss that a reputation downgrade causes, and
    hence the continuation value the matching rule actually creates.
    """
    def share_at(r_focal: float) -> float:
        rng = np.random.default_rng(7000 + seed)
        prices = np.clip(rng.normal(p.P_PAY, 0.08, n), 0.20, 0.95)
        R = np.ones(n)
        R[0] = r_focal
        hits = 0
        for _ in range(T):
            cand = rng.choice(n, size=min(n_bid, n), replace=False)
            if rule == "score":
                score = R[cand] - lam * prices[cand]
            elif rule == "reputation_only":
                score = R[cand]
            elif rule == "price_only":
                score = -prices[cand]
            elif rule == "random":
                score = rng.random(len(cand))
            else:
                raise ValueError(rule)
            i = int(cand[int(np.argmax(score + rng.normal(0.0, 1e-9, len(cand))))])
            hits += int(i == 0)
        return hits / T

    intact = share_at(1.0)
    slashed = share_at(max(0.0, 1.0 - eta))
    wiped = share_at(0.0)
    profit = p.P_PAY - p.KAP_HIGH
    delta_per_slash = max(0.0, intact - slashed) * profit
    return {"mechanism": rule, "share_intact": intact, "share_after_slash": slashed,
            "share_wiped": wiped, "delta_i": delta_per_slash,
            "delta_i_wiped": max(0.0, intact - wiped) * profit}


def matching_ablation(rules: Iterable[str] = MATCHING_RULES, cv: float = 1.0,
                      p: Params = BASE_PARAMS):
    """Matching rules ranked by the continuation value they create and the honesty it buys."""
    rows = []
    for rule in rules:
        m = measure_selection_sensitivity(rule, p=p)
        pd = p_detect(cv, True, p)
        cont = (p.DELTA / (1.0 - p.DELTA)) * m["delta_i"]
        deter = pd * (p.P_PAY + p.SIGMA * p.STAKE + cont)
        rows.append({**m, "continuation_value": cont, "deterrence": deter,
                     "honesty_predicted": gain_cdf(deter, p)})
    return rows
