"""Theory-relevant metrics added for Prof. Qu's Suggestion 5."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import math
import os
from typing import Dict, List

import numpy as np

from model import BASE_PARAMS, CV_GRID, THEORY_MECHANISMS, Params, aggregate_records, p_detect, simulate_learning_run, simulate_run


def dispute_resolution_rounds(cv: float, p: Params = BASE_PARAMS) -> float:
    budget = max(cv, p.DRT_C_MIN)
    return math.ceil(math.log((p.S0 - 1.0) / 0.1) / (p.K_DET * budget))


def convergence_latency(trace: List[float], window: int = 250, band: float = 0.05, hold: int = 500) -> tuple[float, str, float]:
    arr = np.asarray(trace, dtype=float)
    if arr.size < window + hold:
        return float(arr.size), "not_converged", float(arr.mean()) if arr.size else 0.0
    kernel = np.ones(window) / window
    ma = np.convolve(arr, kernel, mode="valid")
    final = float(arr[int(0.75 * arr.size):].mean())
    status = "converged_to_dishonesty" if final < 0.10 else "converged_to_honesty"
    ok = np.abs(ma - final) <= band
    need = max(1, hold - window + 1)
    run = 0
    for idx, is_ok in enumerate(ok):
        run = run + 1 if is_ok else 0
        if run >= need:
            return float(idx + window), status, final
    return float(arr.size), "not_converged", final


def metric_row(metric: str, theory: str, mechanism: str, x: float, seed_mean: float, ci95: float, aux: float | str = "", **extra: float | str) -> Dict[str, float | str]:
    row: Dict[str, float | str] = {
        "metric": metric,
        "theory": theory,
        "mechanism": mechanism,
        "x_key": "cv",
        "x": float(x),
        "seed_mean": seed_mean,
        "ci95": ci95,
        "q1": "",
        "q3": "",
        "rco_verification": "",
        "rco_dispute": "",
        "status": "",
        "aux": aux,
    }
    row.update(extra)
    return row


def trajectory_points(trace: List[float], window: int = 250, step: int = 50) -> tuple[List[int], List[float]]:
    arr = np.asarray(trace, dtype=float)
    if arr.size < window:
        return [], []
    moving = np.convolve(arr, np.ones(window) / window, mode="valid")
    periods = list(range(window, arr.size + 1, step))
    values = [float(moving[period - window]) for period in periods]
    return periods, values


def _learning_diagnostic_task(args: tuple[str, float, int, int, bool, Params]) -> tuple[str, int, float, str, float, List[int], List[float]]:
    mech, cv, seed, learning_T, collect_trajectory, p = args
    out = simulate_learning_run(mech, cv, n=40, T=learning_T, seed=seed, p=p, trace=True)
    trace = out["honesty_trace"]  # type: ignore[assignment]
    latency, status, final = convergence_latency(trace)  # type: ignore[arg-type]
    periods, values = trajectory_points(trace) if collect_trajectory else ([], [])  # type: ignore[arg-type]
    return mech, seed, latency, status, final, periods, values


def learning_diagnostics(trajectory_seeds: int = 50, latency_seeds: int = 200, cv: float = 1.0, learning_T: int = 8000, p: Params = BASE_PARAMS) -> tuple[List[Dict[str, float | str]], List[Dict[str, float | str]]]:
    trajectory_mechanisms = ["Mstar", "no_reputation", "centralized", "receipt_only"]
    latency_mechanisms = ["Mstar", "no_reputation", "receipt_only"]
    tasks: List[tuple[str, float, int, int, bool, Params]] = []
    for mech in trajectory_mechanisms:
        count = latency_seeds if mech in latency_mechanisms else trajectory_seeds
        tasks.extend((mech, cv, seed, learning_T, seed < trajectory_seeds, p) for seed in range(count))

    workers = min(8, max(1, os.cpu_count() or 1))
    if workers == 1:
        results = [_learning_diagnostic_task(task) for task in tasks]
    else:
        chunk = max(1, len(tasks) // (workers * 8))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_learning_diagnostic_task, tasks, chunksize=chunk))

    trace_by_mech: Dict[str, List[tuple[int, List[int], List[float]]]] = {mech: [] for mech in trajectory_mechanisms}
    latency_by_mech: Dict[str, List[tuple[float, str, float]]] = {mech: [] for mech in latency_mechanisms}
    for mech, seed, latency, status, final, periods, values in results:
        if periods:
            trace_by_mech[mech].append((seed, periods, values))
        if mech in latency_by_mech:
            latency_by_mech[mech].append((latency, status, final))

    open_traces = sorted(trace_by_mech["no_reputation"])
    centralized_traces = sorted(trace_by_mech["centralized"])
    if len(open_traces) != len(centralized_traces):
        raise AssertionError("Verification-Only trajectory seed counts differ")
    for left, right in zip(open_traces, centralized_traces):
        if left[0] != right[0] or left[1] != right[1] or not np.array_equal(left[2], right[2]):
            raise AssertionError(f"Verification-Only trajectories diverge at seed {left[0]}")

    trajectory_rows: List[Dict[str, float | str]] = []
    for mech in trajectory_mechanisms:
        traces = sorted(trace_by_mech[mech])
        periods = traces[0][1]
        matrix = np.asarray([item[2] for item in traces], dtype=float)
        means = matrix.mean(axis=0)
        cis = 1.96 * matrix.std(axis=0, ddof=1) / math.sqrt(matrix.shape[0])
        for idx, period in enumerate(periods):
            trajectory_rows.append(
                {
                    "mechanism": mech,
                    "x_key": "period",
                    "x": float(period),
                    "seed_mean": float(means[idx]),
                    "ci95": float(cis[idx]),
                    "honesty": float(means[idx]),
                    "honesty_ci95": float(cis[idx]),
                    "seeds": trajectory_seeds,
                }
            )

    latency_rows: List[Dict[str, float | str]] = []
    for mech in latency_mechanisms:
        values = latency_by_mech[mech]
        counts = Counter(status for _, status, _ in values)
        dominant = counts.most_common(1)[0][0]
        selected = np.asarray([latency for latency, status, _ in values if status == dominant], dtype=float)
        q1, median, q3 = np.percentile(selected, [25, 50, 75])
        latency_rows.append(
            {
                "mechanism": mech,
                "x_key": "cv",
                "x": float(cv),
                "seeds": latency_seeds,
                "status": dominant,
                "median": float(median),
                "q1": float(q1),
                "q3": float(q3),
                "honest_converged": counts.get("converged_to_honesty", 0),
                "dishonest_converged": counts.get("converged_to_dishonesty", 0),
                "not_converged": counts.get("not_converged", 0),
                "final_honesty_mean": float(np.mean([final for _, _, final in values])),
            }
        )
    return trajectory_rows, latency_rows


def theory_metrics(seeds: int = 100, n: int = 50, T: int = 4000, p: Params = BASE_PARAMS) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for mech in THEORY_MECHANISMS:
        for cv in CV_GRID:
            base = [simulate_run(mech, float(cv), n=n, T=T, seed=s, p=p) for s in range(seeds)]
            agg = aggregate_records(base)
            cfg_pd = p_detect(float(cv), mech != "receipt_only", p)
            if mech == "receipt_only":
                drt_mean = float("inf")
                drt_ci = 0.0
            else:
                # DRT is a per-dispute institutional latency: if a dispute is opened
                # at this verification budget, every opened dispute has this many
                # escalation rounds. Dispute incidence is plotted separately.
                drt_mean = float(dispute_resolution_rounds(float(cv), p))
                drt_ci = 0.0
            rco_verification = p.ALPHA_CV * float(cv) / p.P_PAY if mech != "receipt_only" else 0.0
            rco_dispute = agg["dispute"] * p.OVERHEAD_DISPUTE / p.P_PAY if mech != "receipt_only" else 0.0
            rco = rco_verification + rco_dispute
            rows.append(metric_row("moral_hazard_residual", "agency", mech, float(cv), agg["undetected_shortfall"], agg["undetected_shortfall_ci95"], cfg_pd))
            rows.append(
                metric_row(
                    "residual_coordination_overhead",
                    "transaction_cost",
                    mech,
                    float(cv),
                    rco,
                    agg["dispute_ci95"] * p.OVERHEAD_DISPUTE / p.P_PAY,
                    p.OVERHEAD_DISPUTE,
                    rco_verification=rco_verification,
                    rco_dispute=rco_dispute,
                )
            )
            rows.append(metric_row("dispute_resolution_time", "institutional", mech, float(cv), drt_mean, drt_ci, dispute_resolution_rounds(float(cv), p) if mech != "receipt_only" else float("inf")))

    return rows
