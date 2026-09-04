"""Sampling-audit verifier, with its primitives measured rather than assumed.

This module instantiates one concrete verifier class in place of the assumed
residual-spread curve s(cv) of Assumption 2: a sampling audit over an executable
task. Service outputs are real Python implementations of a stated specification. A test bank of independent cases is generated from the reference
implementation. The verifier draws k test cases without replacement, where k is what the
verification budget buys, and reports:

  - the residual value interval implied by the audit, and its spread s,
  - the probability d that a genuine shortfall is flagged,
  - the probability phi that a compliant delivery is wrongly flagged.

No value here comes from a fitted curve. Every number is produced by executing the
implementations against the tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, List, Tuple

import numpy as np


# --- Specification and reference implementation -------------------------------
# Task: return the top-k records by score, descending; ties broken by ascending id.
Record = Tuple[int, float]


def reference(records: List[Record], k: int) -> List[Record]:
    return sorted(records, key=lambda r: (-r[1], r[0]))[:k]


# --- Defective implementations (each is a real, distinct program) -------------
def defect_tie_order(records: List[Record], k: int) -> List[Record]:
    """Breaks ties by descending id instead of ascending."""
    return sorted(records, key=lambda r: (-r[1], -r[0]))[:k]


def defect_off_by_one(records: List[Record], k: int) -> List[Record]:
    """Returns k-1 records."""
    return sorted(records, key=lambda r: (-r[1], r[0]))[: max(0, k - 1)]


def defect_unsorted_tail(records: List[Record], k: int) -> List[Record]:
    """Selects the correct set but does not order the tail."""
    top = sorted(records, key=lambda r: (-r[1], r[0]))[:k]
    return top[:1] + top[1:][::-1] if len(top) > 2 else top


def defect_truncate_scores(records: List[Record], k: int) -> List[Record]:
    """Rounds scores to integers before ranking, collapsing near ties."""
    return sorted(records, key=lambda r: (-round(r[1]), r[0]))[:k]


def near_compliant(records: List[Record], k: int) -> List[Record]:
    """Compliant, but resolves an underspecified input differently.

    The specification does not state what to return when more records are requested
    than exist. The reference returns all records in ranked order; this implementation
    returns all records in the order received. Both are defensible readings, so a
    disagreement on such an input is a specification ambiguity rather than a shortfall.
    """
    if k > len(records):
        return list(records)
    return sorted(records, key=lambda r: (-r[1], r[0]))[:k]


def defect_ignores_ties_partially(records: List[Record], k: int) -> List[Record]:
    """Correct unless the k-th and (k+1)-th scores tie, where it picks the wrong one."""
    ranked = sorted(records, key=lambda r: (-r[1], r[0]))
    if k < len(ranked) and ranked[k - 1][1] == ranked[k][1]:
        ranked[k - 1], ranked[k] = ranked[k], ranked[k - 1]
    return ranked[:k]


def defect_drop_negative(records: List[Record], k: int) -> List[Record]:
    """Silently discards records with negative scores."""
    kept = [r for r in records if r[1] >= 0]
    return sorted(kept, key=lambda r: (-r[1], r[0]))[:k]


# Compliant deliveries are correct or differ only on behaviour the specification
# leaves open. Shortfall deliveries violate the specification on a graded fraction
# of inputs. Neither class is perfect or wholly broken, which is the regime in which
# a sampling audit is actually informative.
HONEST_IMPLS: List[Callable] = [reference, near_compliant]
SHORTFALL_IMPLS: List[Callable] = [
    defect_tie_order,
    defect_truncate_scores,
    defect_drop_negative,
    defect_ignores_ties_partially,
]


# --- Test bank ----------------------------------------------------------------
def make_test_bank(n_tests: int, rng: np.random.Generator) -> List[Tuple[List[Record], int]]:
    """Independent test cases, with score ties and negatives deliberately represented."""
    bank = []
    for _ in range(n_tests):
        m = int(rng.integers(6, 16))
        # score grid fine enough that ties are common but not universal
        scores = rng.choice(np.arange(-2.0, 3.01, 0.25), size=m, replace=True)
        records = [(i, float(scores[i])) for i in range(m)]
        rng.shuffle(records)
        if rng.random() < 0.05:
            # underspecified request: more records asked for than exist
            k = m + int(rng.integers(1, 3))
        else:
            k = int(rng.integers(2, max(3, m // 2 + 1)))
        bank.append((records, k))
    return bank


def pass_vector(impl: Callable, bank: List[Tuple[List[Record], int]]) -> np.ndarray:
    """1 where the implementation matches the reference on that test, else 0."""
    out = np.zeros(len(bank), dtype=np.int8)
    for i, (records, k) in enumerate(bank):
        try:
            out[i] = 1 if impl(list(records), k) == reference(list(records), k) else 0
        except Exception:
            out[i] = 0
    return out


# --- Audit verifier -----------------------------------------------------------
def wilson_interval(x: int, k: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for the true pass rate given x passes out of k draws."""
    if k == 0:
        return (0.0, 1.0)
    phat = x / k
    denom = 1.0 + z * z / k
    centre = (phat + z * z / (2 * k)) / denom
    half = z * math.sqrt(phat * (1 - phat) / k + z * z / (4 * k * k)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def tests_for_budget(cv: float, tests_per_unit: float = 8.0) -> int:
    """Verification budget converted into audit sample size."""
    return int(round(tests_per_unit * cv))


@dataclass(frozen=True)
class AuditResult:
    cv: float
    k: int
    s_measured: float
    d_measured: float
    phi_measured: float
    v_min: float
    v_max: float


def measure(cv: float, bank_size: int = 240, trials: int = 400, seed: int = 0,
            v_floor: float = 0.25, accept_threshold: float = 0.90) -> AuditResult:
    """Measure (s, d, phi) for the sampling-audit verifier at verification budget cv.

    Value is mapped from the true pass rate q by v(q) = v_floor + (1 - v_floor) * q, so that
    a fully correct delivery is worth 1 and a wholly incorrect one is worth v_floor. This is
    the only modelling choice; the pass rates themselves are executed, not assumed.
    """
    rng = np.random.default_rng(1234 + seed)
    bank = make_test_bank(bank_size, rng)
    honest = [pass_vector(f, bank) for f in HONEST_IMPLS]
    short = [pass_vector(f, bank) for f in SHORTFALL_IMPLS]
    k = tests_for_budget(cv)

    def value_of(q: float) -> float:
        return v_floor + (1.0 - v_floor) * q

    if k == 0:
        # Bare receipt. The audit draws no test, so the Wilson interval is the whole
        # unit interval and the residual spread is value_of(1)/value_of(0), which is
        # exactly the raw receipt spread s0 of the analytical model.
        return AuditResult(cv, 0, value_of(1.0) / value_of(0.0), 0.0, 0.0,
                           value_of(0.0), value_of(1.0))

    spreads: List[float] = []
    flags_short = 0
    flags_honest = 0
    n_short = 0
    n_honest = 0
    for t in range(trials):
        for pv, is_short in [(pv, False) for pv in honest] + [(pv, True) for pv in short]:
            idx = rng.choice(len(bank), size=min(k, len(bank)), replace=False)
            x = int(pv[idx].sum())
            lo, hi = wilson_interval(x, min(k, len(bank)))
            spreads.append(value_of(hi) / value_of(lo))
            flagged = (x / min(k, len(bank))) < accept_threshold
            if is_short:
                n_short += 1
                flags_short += int(flagged)
            else:
                n_honest += 1
                flags_honest += int(flagged)
    q_all = [pv.mean() for pv in honest + short]
    return AuditResult(
        cv=cv, k=k,
        s_measured=float(np.mean(spreads)),
        d_measured=flags_short / max(1, n_short),
        phi_measured=flags_honest / max(1, n_honest),
        v_min=value_of(min(q_all)), v_max=value_of(max(q_all)),
    )


def measure_curve(cvs, **kw) -> List[Dict[str, float]]:
    rows = []
    for cv in cvs:
        r = measure(float(cv), **kw)
        rows.append({"cv": r.cv, "k_tests": r.k, "s_measured": r.s_measured,
                     "d_measured": r.d_measured, "phi_measured": r.phi_measured,
                     "v_min": r.v_min, "v_max": r.v_max})
    return rows


def measure_calibrated(cv: float, bank_size: int = 240, trials: int = 800, seed: int = 0,
                       v_floor: float = 0.25, phi_target: float = 0.05) -> Dict[str, float]:
    """Measure (s, d, phi) with the acceptance rule calibrated to hold phi at phi_target.

    A fixed pass-rate threshold interacts with the discreteness of the sample count, so the
    realized false-positive rate oscillates in k. Audit practice instead fixes the tolerable
    false-positive rate and derives the acceptance rule from the compliant class's own
    sampling distribution. We follow that convention: for each k the allowed number of
    failing tests is the largest value that keeps the compliant flag rate at or below
    phi_target, estimated from the compliant implementations, and the same rule is then
    applied to the shortfall class to obtain d.
    """
    rng = np.random.default_rng(1234 + seed)
    bank = make_test_bank(bank_size, rng)
    honest = [pass_vector(f, bank) for f in HONEST_IMPLS]
    short = [pass_vector(f, bank) for f in SHORTFALL_IMPLS]
    k = tests_for_budget(cv)

    def value_of(q: float) -> float:
        return v_floor + (1.0 - v_floor) * q

    if k == 0:
        return {"cv": cv, "k_tests": 0, "s_measured": value_of(1.0) / value_of(0.0),
                "d_measured": 0.0, "phi_measured": 0.0, "max_failures": -1.0}

    kk = min(k, len(bank))

    def draw_failures(pv):
        idx = rng.choice(len(bank), size=kk, replace=False)
        return kk - int(pv[idx].sum()), idx

    honest_fails, spreads = [], []
    for _ in range(trials):
        for pv in honest:
            f, idx = draw_failures(pv)
            honest_fails.append(f)
            lo, hi = wilson_interval(kk - f, kk)
            spreads.append(value_of(hi) / value_of(lo))
    # largest failure allowance whose compliant flag rate stays within phi_target
    hf = np.asarray(honest_fails)
    max_fail = 0
    for cand in range(0, kk + 1):
        if float((hf > cand).mean()) <= phi_target:
            max_fail = cand
            break
    else:
        max_fail = kk
    phi_hat = float((hf > max_fail).mean())

    short_flags = []
    for _ in range(trials):
        for pv in short:
            f, idx = draw_failures(pv)
            short_flags.append(f > max_fail)
            lo, hi = wilson_interval(kk - f, kk)
            spreads.append(value_of(hi) / value_of(lo))
    return {"cv": cv, "k_tests": kk, "s_measured": float(np.mean(spreads)),
            "d_measured": float(np.mean(short_flags)), "phi_measured": phi_hat,
            "max_failures": float(max_fail)}


def measure_calibrated_curve(cvs, **kw) -> List[Dict[str, float]]:
    return [measure_calibrated(float(cv), **kw) for cv in cvs]
