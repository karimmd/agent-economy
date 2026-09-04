# Verifiable Agent Economy — simulation code and artifacts

Simulation code, parameters, seeds, and generated artifacts for the paper *Can We Trust Autonomous
Agents? A Formal Framework for Verifiable Agent Economies*.

Every value in the paper's numerical sections is produced by the scripts in `src/`. No figure or
table value is entered by hand.

## Quick start

```bash
python src/run_all.py       # runs every experiment, writes results/ only
python src/make_figures.py  # reads results/, writes figures/; never re-simulates
```

`run_all.py` takes about 3.5 minutes on a single core. The two stages are separated so that
plotting cannot silently change a reported number.

## Environment

Tested with Python 3.14.7, numpy 2.5.2, matplotlib 3.11.0.

One caveat matters for exact reproduction. numpy built against MKL and numpy built against
OpenBLAS differ in the last bits of `std` and percentile reductions. That difference is invisible
almost everywhere, but it moves one printed value: the lower quartile of Receipt-Only convergence
latency in `results/learning_latency.csv` (3309.0 under OpenBLAS, 3308.5 under MKL). The reported
value is the OpenBLAS one. To reproduce it:

```bash
conda install -c conda-forge "libblas=*=*openblas"
python -c "import numpy; print(numpy.show_config('dicts')['Build Dependencies']['blas']['name'])"
# expect: openblas
```

Every other artifact is identical under both builds.

## Seeds

Deterministic and fixed, never drawn from the clock.

| Experiment family | Seed policy |
|---|---|
| Analytical best-response runs | `seed = run index`, `0 .. seeds-1` |
| Learning runs | `1000 + seed` |
| Matching-rule market | `5000 + seed` |
| Selection-sensitivity measurement | `7000 + seed` |
| Audit verifier | `1234 + seed` |

Seed counts: 100 for analytical sweeps, 50 for learning welfare and trajectories, 200 for
convergence latency, 40 for the matching ablation, 800 audit trials per budget.

## Parameters

The complete parameter set is the `Params` dataclass in `src/model.py`, dumped verbatim to
`results/params.json` on every run together with every sweep grid. Read that file rather than this
one for the authoritative values.

## Mechanism variants

Defined in `mech_config()` in `src/model.py`. Each toggles exactly one component.

| Variant | Verification | Stake | Reputation | Routing factor |
|---|---|---|---|---|
| `Mstar` | yes | 0.60 | yes | 1.00 |
| `no_stake` | yes | 0 | yes | 1.00 |
| `stake_only` | yes | 0.60 | no | 1.00 |
| `no_reputation` | yes | 0 | no | 1.00 |
| `centralized` | yes | 0 | no | 0.92 |
| `receipt_only` | no | 0.60 | yes | 1.00 |

The routing factor multiplies delivered value in the welfare accounting,
`W = coord * v(o) - kappa(o) - ALPHA_CV * c_v`. Only `centralized` uses a value below 1.

## Q-learning configuration

`simulate_learning_run()` and `simulate_matching_run()` in `src/model.py`. Tabular Q-learning over
a binary action (deliver honestly, deliver a shortfall), learning rate `alpha = 0.1`,
epsilon-greedy exploration decaying as `eps = max(0.02, 0.30 * exp(-3t/T))`, reputation step
`eta = 0.05`, horizon `T = 8000`, `n = 40` agents, statistics accumulated after a 60% warm-up.
Selection is reputation-weighted via `softmax(3.0 * R)` in the learning runs, and by the stated
matching rule over a per-period candidate set of 5 in the matching ablation.

## Verifier instantiation

`src/audit_verifier.py` measures the verification primitives rather than assuming them: it
executes real implementations of a ranking specification against a generated test bank, buys a
sample of `k` tests with the verification budget, derives the residual interval from the Wilson
interval on the audited pass rate, and calibrates the acceptance rule to hold false positives at
0.05.

## Where each reported artifact comes from

Figures 1, 2, 3, 4 and 13 of the paper are drawn by hand and are not produced here.

| Paper item | Produced by | Source data |
|---|---|---|
| Figure 5 (welfare, honesty) | `plot_welfare`, `plot_delta` | `welfare_cv.csv`, `delta_sweep.csv` |
| Figure 6 (residual exposure) | `plot_exposure_spread`, `plot_exposure_error` | `exposure_spread.csv`, `exposure_error.csv` |
| Figure 7 (stake, Sybil) | `plot_stake`, `plot_sybil` | `stake_sweep.csv`, `sybil.csv` |
| Figure 8 (detection laws) | `plot_families` | `verifier_families.csv` |
| Figure 9 (measured verifier, false positives) | `plot_measured_verifier`, `plot_false_positive` | `audit_verifier.csv`, `false_positive.csv` |
| Figure 10 (market thickness) | `plot_thickness`, `plot_required_stake` | `thickness.csv`, `required_stake.csv` |
| Figure 11 (learning agents) | `plot_learning_trajectory`, `plot_learning_welfare` | `learning_trajectory.csv`, `learning_cv.csv` |
| Figure 12 (theory diagnostics) | `plot_moral_hazard`, `plot_rco_grouped` | `theory_metrics.csv` |
| Table 5 (detection laws) | `sweep_family`, `sweep_measured_verifier` | `verifier_families.csv`, `measured_welfare.csv` |
| Table 6 (audit verifier) | `measure_calibrated_curve` | `audit_verifier.csv` |
| Table 7 (component ablation) | `ablation_table` | `ablation.csv` |
| Table 8 (matching ablation) | `matching_ablation` | `matching_ablation.csv` |
| Table 9 (sensitivity) | `sensitivity` | `sensitivity.csv` |
| Routing-penalty sensitivity | `run_all.sensitivity` | `routing_penalty.csv` |

Headline values quoted in the text are collected in `results/headline_numbers.json`.

## Layout

```
src/          simulation and plotting code
results/      generated CSV and JSON artifacts (regenerated by run_all.py)
figures/      generated PDF and PNG figures (regenerated by make_figures.py)
```
