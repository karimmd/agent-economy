"""Run every experiment and write CSV/JSON results only."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Dict, Iterable, List

import numpy as np

from audit_verifier import measure_calibrated_curve
from exposure import exposure_error_experiment, exposure_experiment, exposure_spread_experiment
from metrics import learning_diagnostics, theory_metrics
from model import BASE_PARAMS, CV_GRID, DELTA_GRID, LEARNING_MECHANISMS, MECHANISMS, STAKE_GRID, SYBIL_GRID, THICKNESS_GRID, THICKNESS_STAKES, MATCHING_RULES, PHI_GRID, VERIFIER_FAMILIES, Params, ablation_table, matching_ablation, param_variant, required_stake, sweep_cv, sweep_false_positive, sweep_family, sweep_measured_verifier, sweep_delta, sweep_learning, sweep_stake, sweep_sybil, sweep_thickness


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def nearest(rows: List[Dict[str, object]], mechanism: str, x: float, key: str = "welfare") -> float:
    candidates = [r for r in rows if r.get("mechanism") == mechanism]
    row = min(candidates, key=lambda r: abs(float(r["x"]) - x))
    return float(row[key])


def peak(rows: List[Dict[str, object]], mechanism: str) -> Dict[str, float]:
    candidates = [r for r in rows if r.get("mechanism") == mechanism]
    row = max(candidates, key=lambda r: float(r["welfare"]))
    return {"cv": float(row["x"]), "welfare": float(row["welfare"]), "honesty": float(row["honesty"])}


def theory_value(rows: List[Dict[str, object]], metric: str, mechanism: str, x: float) -> float:
    candidates = [r for r in rows if r.get("metric") == metric and r.get("mechanism") == mechanism]
    row = min(candidates, key=lambda r: abs(float(r["x"]) - x))
    return float(row["seed_mean"])


def headline_numbers(welfare_rows, exposure_rows, exposure_spread_rows, stake_rows, sybil_rows, learning_rows, latency_rows, theory_rows, thickness_rows, family_rows, measured_rows, audit_rows, fp_rows, abl_rows, match_rows, coord_rows) -> Dict[str, object]:
    mstar_peak = peak(welfare_rows, "Mstar")
    learning_peak = peak(learning_rows, "Mstar")
    exposure_geo = {f"{x:.1f}": nearest(exposure_rows, "geometric", x, "exposure") for x in [0.0, 1.0, 2.0, 3.0]}
    exposure_naive = {f"{x:.1f}": nearest(exposure_rows, "naive_mean", x, "exposure") for x in [0.0, 1.0, 2.0, 3.0]}
    return {
        "setup": {"n": 50, "T": 4000, "seeds": 100, "learning_T": 8000, "learning_seeds": 50, "bcl_seeds": 200, "cv_grid_points": len(CV_GRID), "stake_grid_points": len(STAKE_GRID), "sybil_grid_points": len(SYBIL_GRID)},
        "welfare": {
            "Mstar_peak": mstar_peak,
            "receipt_only_plateau": nearest(welfare_rows, "receipt_only", 1.5),
            "at_cv_1_5": {
                "Mstar": {"welfare": nearest(welfare_rows, "Mstar", 1.5), "honesty": nearest(welfare_rows, "Mstar", 1.5, "honesty")},
                "no_reputation": {"welfare": nearest(welfare_rows, "no_reputation", 1.5), "honesty": nearest(welfare_rows, "no_reputation", 1.5, "honesty")},
                "centralized": {"welfare": nearest(welfare_rows, "centralized", 1.5), "honesty": nearest(welfare_rows, "centralized", 1.5, "honesty")},
            },
        },
        "exposure": {
            "geometric": exposure_geo,
            "naive_mean": exposure_naive,
            "max_abs_geo_minus_bound": max(abs(float(r["exposure"]) - float(r["analytic"])) for r in exposure_rows if r["mechanism"] == "geometric"),
            "spread": {
                rule: {str(r["spread"]): float(r["exposure"]) for r in exposure_spread_rows if r["rule"] == rule}
                for rule in ("geometric", "naive_mean")
            },
        },
        "stake": {
            "honesty_at_0": nearest(stake_rows, "Mstar", 0.0, "honesty"),
            "honesty_at_0_2": nearest(stake_rows, "Mstar", 0.2, "honesty"),
            "honesty_at_0_4": nearest(stake_rows, "Mstar", 0.4, "honesty"),
            "analytic_95_threshold": next(float(r["x"]) for r in stake_rows if r["mechanism"] == "Mstar" and float(r["analytic"]) >= 0.95),
        },
        "sybil": {
            "Mstar_frac_0": nearest(sybil_rows, "Mstar", 0.0),
            "Mstar_frac_0_8": nearest(sybil_rows, "Mstar", 0.8),
            "no_stake_frac_0": nearest(sybil_rows, "no_stake", 0.0),
            "no_stake_frac_0_8": nearest(sybil_rows, "no_stake", 0.8),
        },
        "learning": {
            "Mstar_peak": learning_peak,
            "receipt_only_at_peak_cv": nearest(learning_rows, "receipt_only", learning_peak["cv"]),
            "latency_cv_1": {str(r["mechanism"]): {k: r[k] for k in r if k not in ("mechanism", "x_key", "x")} for r in latency_rows},
        },
        "ablation": {
            r["mechanism"]: {"welfare": r["welfare"], "honesty": r["honesty"]} for r in abl_rows
        },
        "matching": {
            f'{r["mechanism"]}@{r["stake"]}': {"delta_i": r["delta_i"], "honesty": r["honesty_predicted"]}
            for r in match_rows if r["stake"] in (0.2, 0.6)
        },
        "routing_penalty": {str(r["coord"]): r["peak_welfare"] for r in coord_rows},
        "verification": {
            "family_peaks": {fam: peak(family_rows, fam) for fam in VERIFIER_FAMILIES},
            "measured_peak": peak(measured_rows, "measured_audit"),
            "audit_s_at_0": next(float(r["s_measured"]) for r in audit_rows if r["cv"] == 0.0),
            "audit_s_at_3": next(float(r["s_measured"]) for r in audit_rows if r["cv"] == 3.0),
            "audit_d_at_3": next(float(r["d_measured"]) for r in audit_rows if r["cv"] == 3.0),
            "phi_max_participation": next(float(r["phi_max"]) for r in fp_rows),
            "welfare_at_phi_0": next(float(r["welfare"]) for r in fp_rows if abs(float(r["x"])) < 1e-9),
        },
        "thickness": {
            "required_stake_monopoly": required_stake(0.5, 0.0),
            "required_stake_baseline": required_stake(0.5, 0.15),
            "honesty_monopoly_stake_0_6": next(float(r["honesty"]) for r in thickness_rows if r["mechanism"] == "stake=0.60" and abs(float(r["x"])) < 1e-9),
            "honesty_baseline_stake_0_6": next(float(r["honesty"]) for r in thickness_rows if r["mechanism"] == "stake=0.60" and abs(float(r["x"]) - 0.15) < 1e-9),
            "honesty_monopoly_stake_1_0": next(float(r["honesty"]) for r in thickness_rows if r["mechanism"] == "stake=1.00" and abs(float(r["x"])) < 1e-9),
        },
        "theory": {
            "MHR_Mstar_cv_0": theory_value(theory_rows, "moral_hazard_residual", "Mstar", 0.0),
            "MHR_Mstar_cv_3": theory_value(theory_rows, "moral_hazard_residual", "Mstar", 3.0),
            "RCO_Mstar_cv_1_5": theory_value(theory_rows, "residual_coordination_overhead", "Mstar", 1.5),
            "RCO_Mstar_cv_1_5_verification": next(float(r["rco_verification"]) for r in theory_rows if r["metric"] == "residual_coordination_overhead" and r["mechanism"] == "Mstar" and abs(float(r["x"]) - 1.5) < 1e-9),
            "RCO_Mstar_cv_1_5_dispute": next(float(r["rco_dispute"]) for r in theory_rows if r["metric"] == "residual_coordination_overhead" and r["mechanism"] == "Mstar" and abs(float(r["x"]) - 1.5) < 1e-9),
            "RCO_verification_only_cv_0_5_dispute": next(float(r["rco_dispute"]) for r in theory_rows if r["metric"] == "residual_coordination_overhead" and r["mechanism"] == "no_reputation" and abs(float(r["x"]) - 0.5) < 1e-9),
            "RCO_receipt_only_cv_1_5": theory_value(theory_rows, "residual_coordination_overhead", "receipt_only", 1.5),
            "DRT_Mstar_cv_1": theory_value(theory_rows, "dispute_resolution_time", "Mstar", 1.0),
            "DRT_rounds": {f"{x:g}": theory_value(theory_rows, "dispute_resolution_time", "Mstar", x) for x in [0.25, 0.5, 1.0, 2.0, 3.0]},
            "DRT_receipt_only": "censored_infinite",
        },
    }


def sensitivity() -> List[Dict[str, object]]:
    sweeps = [
        ("n", [20, 50, 100]),
        ("T", [4000, 20000]),
        ("s0", [2, 4, 8]),
        ("SIGMA", [0.5, 1.0]),
        ("STAKE", [0.3, 0.6, 1.0]),
        ("DELTA", [0.90, 0.95, 0.99]),
        ("ALPHA_CV", [0.025, 0.05, 0.1]),
        ("OVERHEAD_DISPUTE", [0.025, 0.05, 0.1]),
    ]
    rows: List[Dict[str, object]] = []
    for param, values in sweeps:
        for value in values:
            p = param_variant(param, float(value)) if param not in ("n", "T") else BASE_PARAMS
            n = int(value) if param == "n" else 50
            T = int(value) if param == "T" else 4000
            cv_rows = sweep_cv(["Mstar"], CV_GRID, seeds=100, n=n, T=T, p=p)
            pk = peak(cv_rows, "Mstar")
            rows.append({"parameter": param, "value": value, "peak_cv": pk["cv"], "peak_welfare": pk["welfare"], "honesty_at_peak": pk["honesty"]})
    return rows


def write_sensitivity_preview(rows: List[Dict[str, object]]) -> None:
    lines = ["| Parameter | Values | Welfare peak (value @ c_v) | Honesty at peak |", "|---|---|---|---|"]
    params = []
    for r in rows:
        if r["parameter"] not in params:
            params.append(r["parameter"])
    for param in params:
        group = [r for r in rows if r["parameter"] == param]
        lines.append(
            "| {} | {} | {} | {} |".format(
                param,
                ", ".join(str(r["value"]) for r in group),
                "; ".join(f"{float(r['peak_welfare']):.2f} @ {float(r['peak_cv']):.2f}" for r in group),
                "; ".join(f"{float(r['honesty_at_peak']):.2f}" for r in group),
            )
        )
    (RESULTS / "sensitivity_table.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    timings: Dict[str, float] = {}
    start_all = time.perf_counter()

    t = time.perf_counter()
    welfare_rows = sweep_cv(MECHANISMS, CV_GRID, seeds=100)
    write_csv(RESULTS / "welfare_cv.csv", welfare_rows)
    timings["welfare_cv"] = time.perf_counter() - t

    t = time.perf_counter()
    exposure_rows = exposure_experiment(CV_GRID)
    write_csv(RESULTS / "exposure_cv.csv", exposure_rows)
    timings["exposure_cv"] = time.perf_counter() - t

    exposure_spread_rows = exposure_spread_experiment()
    write_csv(RESULTS / "exposure_spread.csv", exposure_spread_rows)

    exposure_error_rows = exposure_error_experiment()
    write_csv(RESULTS / "exposure_error.csv", exposure_error_rows)

    dispute_rows = [{k: v for k, v in r.items()} for r in welfare_rows if r["mechanism"] in ["Mstar", "no_reputation", "centralized"]]
    write_csv(RESULTS / "dispute_cv.csv", dispute_rows)

    t = time.perf_counter()
    sybil_rows = sweep_sybil(SYBIL_GRID, seeds=100, cv=1.0)
    write_csv(RESULTS / "sybil.csv", sybil_rows)
    timings["sybil"] = time.perf_counter() - t

    t = time.perf_counter()
    stake_rows = sweep_stake(STAKE_GRID, seeds=100, cv=0.5)
    write_csv(RESULTS / "stake_sweep.csv", stake_rows)
    timings["stake_sweep"] = time.perf_counter() - t

    t = time.perf_counter()
    delta_rows = sweep_delta(DELTA_GRID, seeds=100, cv=0.5)
    write_csv(RESULTS / "delta_sweep.csv", delta_rows)
    timings["delta_sweep"] = time.perf_counter() - t

    t = time.perf_counter()
    learning_rows = sweep_learning(LEARNING_MECHANISMS, CV_GRID, seeds=50)
    write_csv(RESULTS / "learning_cv.csv", learning_rows)
    timings["learning_cv"] = time.perf_counter() - t

    t = time.perf_counter()
    trajectory_rows, latency_rows = learning_diagnostics(trajectory_seeds=50, latency_seeds=200, cv=1.0)
    write_csv(RESULTS / "learning_trajectory.csv", trajectory_rows)
    write_csv(RESULTS / "learning_latency.csv", latency_rows)
    timings["learning_diagnostics"] = time.perf_counter() - t

    t = time.perf_counter()
    thickness_rows = sweep_thickness(seeds=100, cv=0.5)
    write_csv(RESULTS / "thickness.csv", thickness_rows)
    req_rows = [
        {"match_factor": float(mf), "cv": 0.5, "target": 0.95, "required_stake": required_stake(0.5, float(mf))}
        for mf in THICKNESS_GRID
    ]
    write_csv(RESULTS / "required_stake.csv", req_rows)
    timings["thickness"] = time.perf_counter() - t

    t = time.perf_counter()
    abl_rows = ablation_table(cv=1.0, seeds=100)
    write_csv(RESULTS / "ablation.csv", abl_rows)
    match_rows = []
    for stake in [0.0, 0.2, 0.4, 0.6]:
        for r in matching_ablation(cv=1.0, p=param_variant("STAKE", stake)):
            match_rows.append({"stake": stake, **r})
    write_csv(RESULTS / "matching_ablation.csv", match_rows)
    coord_rows = []
    for coord in [0.88, 0.92, 0.96, 1.0]:
        cv_rows = sweep_cv(["centralized"], CV_GRID, seeds=100, p=param_variant("COORD", coord))
        pk = peak(cv_rows, "centralized")
        coord_rows.append({"coord": coord, "peak_cv": pk["cv"], "peak_welfare": pk["welfare"], "honesty_at_peak": pk["honesty"]})
    write_csv(RESULTS / "routing_penalty.csv", coord_rows)
    timings["ablation_wp5"] = time.perf_counter() - t

    t = time.perf_counter()
    family_rows = sweep_family(VERIFIER_FAMILIES, CV_GRID, seeds=100)
    write_csv(RESULTS / "verifier_families.csv", family_rows)
    fp_rows = sweep_false_positive(PHI_GRID, cv=1.0, seeds=100)
    write_csv(RESULTS / "false_positive.csv", fp_rows)
    audit_grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    audit_rows = measure_calibrated_curve(audit_grid, trials=800)
    write_csv(RESULTS / "audit_verifier.csv", audit_rows)
    measured_rows = sweep_measured_verifier(audit_rows, seeds=100)
    write_csv(RESULTS / "measured_welfare.csv", measured_rows)
    timings["verification_wp4"] = time.perf_counter() - t

    t = time.perf_counter()
    theory_rows = theory_metrics(seeds=100)
    write_csv(RESULTS / "theory_metrics.csv", theory_rows)
    timings["theory_metrics"] = time.perf_counter() - t

    t = time.perf_counter()
    sens_rows = sensitivity()
    write_csv(RESULTS / "sensitivity.csv", sens_rows)
    write_sensitivity_preview(sens_rows)
    timings["sensitivity"] = time.perf_counter() - t

    (RESULTS / "params.json").write_text(json.dumps({"base": BASE_PARAMS.as_dict(), "cv_grid": list(map(float, CV_GRID)), "stake_grid": list(map(float, STAKE_GRID)), "sybil_grid": list(map(float, SYBIL_GRID)), "delta_grid": list(map(float, DELTA_GRID)), "exposure_error": {"cv": 1.0, "epsilon_max": 0.6}, "learning_trajectory": {"cv": 1.0, "window": 250, "step": 50, "seeds": 50}, "learning_latency_seeds": 200, "timings_seconds": timings}, indent=2, sort_keys=True) + "\n")
    (RESULTS / "headline_numbers.json").write_text(json.dumps(headline_numbers(welfare_rows, exposure_rows, exposure_spread_rows, stake_rows, sybil_rows, learning_rows, latency_rows, theory_rows, thickness_rows, family_rows, measured_rows, audit_rows, fp_rows, abl_rows, match_rows, coord_rows), indent=2, sort_keys=True) + "\n")
    timings["total"] = time.perf_counter() - start_all
    print(json.dumps(timings, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
