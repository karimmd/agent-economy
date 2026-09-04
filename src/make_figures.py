"""Build publication figures from generated CSV files only."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from plot_style import COLORS, LABELS, MARKERS, bar_kwargs, line_with_ci, panel, save_figure


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PANEL_NAMES = [
    "fig5a_welfare_cv",
    "fig5b_honesty_delta",
    "fig6a_exposure_spread",
    "fig6b_exposure_error",
    "fig7a_honesty_stake",
    "fig7b_welfare_sybil",
    "fig8a_learning_trajectory",
    "fig8b_learning_cv",
    "fig9a_moral_hazard",
    "fig9b_coordination_overhead",
    "fig11a_thickness_honesty",
    "fig11b_required_stake",
    "fig12a_measured_verifier",
    "fig12b_false_positive",
    "fig13_families",
]

INFEASIBLE_GRAY = "#E9E9E9"
BOUND_GRAY = "#3A3A3A"


def read_csv(name: str):
    with (RESULTS / name).open() as f:
        return list(csv.DictReader(f))


def rows_for(rows, mechanism: str):
    return sorted((r for r in rows if r["mechanism"] == mechanism), key=lambda r: float(r["x"]))


def at_x(rows, mechanism: str, x: float):
    candidates = rows_for(rows, mechanism)
    return min(candidates, key=lambda r: abs(float(r["x"]) - x))


def assert_equal_series(rows, left: str, right: str, field: str, tol: float = 1e-12) -> None:
    lrows = rows_for(rows, left)
    rrows = rows_for(rows, right)
    if len(lrows) != len(rrows):
        raise AssertionError(f"{left}/{right} row count mismatch for {field}")
    for lrow, rrow in zip(lrows, rrows):
        if abs(float(lrow["x"]) - float(rrow["x"])) > tol or abs(float(lrow[field]) - float(rrow[field])) > tol:
            raise AssertionError(f"{left}/{right} diverge for {field} at x={lrow['x']}: {lrow[field]} vs {rrow[field]}")


def plot_welfare(rows):
    fig, ax = panel()
    for mech in ["receipt_only", "centralized", "no_reputation", "Mstar"]:
        line_with_ci(ax, rows, mech, "welfare", "welfare_ci95")
    ax.set_xlabel("Verification budget")
    ax.set_ylabel("Social welfare per period")
    ax.set_ylim(0.05, 0.78)
    handles, labels = ax.get_legend_handles_labels()
    labels = [l.replace("Verification-Only (open matching)", "Verification-Only (open)") for l in labels]
    ax.legend(handles[::-1], labels[::-1], loc="upper right", ncol=1, fontsize=8.8, handlelength=1.4, borderaxespad=0.35)
    save_figure(fig, FIGURES / "fig5a_welfare_cv")


def plot_delta(rows):
    """Honesty vs discount factor: the continuation-value (patience) lever of Prop. IC.
    Stake-without-reputation is delta-independent by construction and plots flat."""
    fig, ax = panel()
    for mech, style_key in [("stake_only", "stake_only"), ("Mstar", "Mstar")]:
        sub = rows_for(rows, mech)
        xs = [float(r["x"]) for r in sub]
        ax.plot(xs, [float(r["honesty"]) for r in sub], color=COLORS[style_key], marker=MARKERS[style_key], linewidth=2.6, markersize=7.5, markeredgecolor="white", markeredgewidth=1.1, label=LABELS[style_key], zorder=5 if mech == "Mstar" else 3)
        ax.plot(xs, [float(r["analytic"]) for r in sub], "--", color="black", linewidth=1.8, zorder=4, label="Analytical prediction" if mech == "Mstar" else None)
    ax.set_xlabel("Discount factor")
    ax.set_ylabel("Honest-delivery rate")
    ax.set_ylim(0.55, 1.06)
    ax.legend(loc="upper left", fontsize=9.5)
    save_figure(fig, FIGURES / "fig5b_honesty_delta")


def _exposure_axes(ax, xs_dense, bound_dense):
    """Shared frontier styling: infeasible region + thick theoretical-minimum line."""
    ax.fill_between(xs_dense, 1.0, bound_dense, color=INFEASIBLE_GRAY, linewidth=0, zorder=1, label="Infeasible region")
    ax.plot(xs_dense, bound_dense, color=BOUND_GRAY, linewidth=4.5, alpha=0.85, solid_capstyle="round", zorder=2, label=LABELS["analytic"])


def plot_exposure_spread(rows):
    fig, ax = panel()
    geometric = sorted((r for r in rows if r["rule"] == "geometric"), key=lambda r: float(r["spread"]))
    arithmetic = sorted((r for r in rows if r["rule"] == "naive_mean"), key=lambda r: float(r["spread"]))
    xs = [float(r["spread"]) for r in geometric]
    xs_dense = np.linspace(min(xs), max(xs), 200)
    _exposure_axes(ax, xs_dense, np.sqrt(xs_dense))
    ax.plot(xs, [float(r["exposure"]) for r in arithmetic], color=COLORS["naive_mean"], marker=MARKERS["naive_mean"], linestyle="--", linewidth=2.4, markersize=8, markeredgecolor="white", markeredgewidth=1.2, label=LABELS["naive_mean"], zorder=4)
    ax.plot(xs, [float(r["exposure"]) for r in geometric], color=COLORS["geometric"], linewidth=2.0, marker="o", markersize=8, markerfacecolor="white", markeredgecolor=COLORS["geometric"], markeredgewidth=2.0, label=LABELS["geometric"], zorder=5)
    ax.set_xlabel("Receipt quality spread")
    ax.set_ylabel("Residual exposure")
    ax.set_ylim(1.0, 4.75)
    ax.set_xlim(min(xs) - 0.15, max(xs) + 0.15)
    ax.legend(loc="upper left", fontsize=9.5)
    save_figure(fig, FIGURES / "fig6a_exposure_spread")


def plot_exposure_error(rows):
    """Degraded frontier under verifier false negatives: the bound becomes
    sqrt(s_eps); geometric pricing still rides it."""
    fig, ax = panel()
    geometric = sorted((r for r in rows if r["rule"] == "geometric"), key=lambda r: float(r["epsilon"]))
    arithmetic = sorted((r for r in rows if r["rule"] == "naive_mean"), key=lambda r: float(r["epsilon"]))
    xs = np.array([float(r["epsilon"]) for r in geometric])
    bound = np.array([float(r["analytic"]) for r in geometric])
    _exposure_axes(ax, xs, bound)
    ax.plot(xs, [float(r["exposure"]) for r in arithmetic], color=COLORS["naive_mean"], marker=MARKERS["naive_mean"], linestyle="--", linewidth=2.4, markersize=8, markeredgecolor="white", markeredgewidth=1.2, label=LABELS["naive_mean"], zorder=4)
    ax.plot(xs, [float(r["exposure"]) for r in geometric], color=COLORS["geometric"], linewidth=2.0, marker="o", markersize=8, markerfacecolor="white", markeredgecolor=COLORS["geometric"], markeredgewidth=2.0, label=LABELS["geometric"], zorder=5)
    ax.set_xlabel("Verifier false-negative rate")
    ax.set_ylabel("Residual exposure")
    ax.set_ylim(1.0, 2.35)
    ax.legend(loc="upper left", fontsize=9.5)
    save_figure(fig, FIGURES / "fig6b_exposure_error")


def plot_stake(rows):
    fig, ax = panel()
    for mech, style_key in [("receipt_only", "receipt_only"), ("stake_only", "stake_only"), ("Mstar", "Mstar")]:
        sub = rows_for(rows, mech)
        xs = [float(r["x"]) for r in sub]
        ax.plot(xs, [float(r["honesty"]) for r in sub], color=COLORS[style_key], marker=MARKERS[style_key], linewidth=2.6, markersize=7.5, markeredgecolor="white", markeredgewidth=1.1, label=LABELS[style_key], zorder=5 if mech == "Mstar" else 3)
        if mech != "receipt_only":
            ax.plot(xs, [float(r["analytic"]) for r in sub], "--", color="black", linewidth=1.8, zorder=4, label="Analytical prediction" if mech == "Mstar" else None)
    ax.set_xlabel("Admission stake")
    ax.set_ylabel("Honest-delivery rate")
    ax.set_ylim(-0.04, 1.06)
    ax.legend(loc="center right", fontsize=9.5)
    save_figure(fig, FIGURES / "fig7a_honesty_stake")


def plot_sybil(rows):
    fig, ax = panel()
    line_with_ci(ax, rows, "Mstar", "welfare", "welfare_ci95", label=LABELS["Mstar_staked"])
    line_with_ci(ax, rows, "no_stake", "welfare", "welfare_ci95")
    line_with_ci(ax, rows, "no_reputation", "welfare", "welfare_ci95", label=LABELS["verification_only"])
    ax.set_xlabel("Sybil re-registration fraction")
    ax.set_ylabel("Social welfare per period")
    ax.set_ylim(0.30, 0.60)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=9.5, handlelength=1.3, columnspacing=0.9)
    save_figure(fig, FIGURES / "fig7b_welfare_sybil")


def plot_thickness(rows):
    """Honesty vs market thickness at several admission stakes."""
    fig, ax = panel()
    shades = ["#B8CBE0", "#7FA6C9", "#3F6F9F", "#1F3F63"]
    stakes = sorted({float(r["stake"]) for r in rows})
    for stake, col in zip(stakes, shades):
        sub = sorted((r for r in rows if abs(float(r["stake"]) - stake) < 1e-9), key=lambda r: float(r["x"]))
        xs = [float(r["x"]) for r in sub]
        ax.plot(xs, [float(r["honesty"]) for r in sub], color=col, marker="o", linewidth=2.4,
                markersize=6.5, markeredgecolor="white", markeredgewidth=1.0,
                label=f"$S={stake:.2f}$", zorder=5)
        ax.plot(xs, [float(r["analytic"]) for r in sub], "--", color="black", linewidth=1.3, zorder=4,
                label="Analytical" if stake == stakes[-1] else None)
    ax.axvline(0.0, color="#B03030", linewidth=1.2, linestyle=":", zorder=2)
    ax.annotate("monopolist", xy=(0.0, 0.12), xytext=(0.012, 0.06), fontsize=8.5, color="#B03030")
    ax.set_xlabel(r"Market thickness $\kappa$")
    ax.set_ylabel("Honest-delivery rate")
    ax.set_ylim(-0.04, 1.12)
    ax.legend(loc="lower right", fontsize=7.6, ncol=2, columnspacing=0.8,
              handlelength=1.4, borderpad=0.35, labelspacing=0.3, framealpha=0.92)
    save_figure(fig, FIGURES / "fig11a_thickness_honesty")


def plot_required_stake(rows):
    """Admission stake needed to hold the 95% honesty target as the market thins."""
    fig, ax = panel()
    sub = sorted(rows, key=lambda r: float(r["match_factor"]))
    xs = [float(r["match_factor"]) for r in sub]
    ys = [float(r["required_stake"]) for r in sub]
    ax.plot(xs, ys, color="#1F3F63", marker="s", linewidth=2.6, markersize=7.0,
            markeredgecolor="white", markeredgewidth=1.1, zorder=5)
    ax.fill_between(xs, min(ys) * 0.94, ys, color=INFEASIBLE_GRAY, zorder=1)
    ax.annotate("stake below requirement", xy=(0.11, 0.60), fontsize=8.4, color="#5A5A5A")
    ax.annotate(f"{ys[0]:.2f}", xy=(xs[0], ys[0]), xytext=(xs[0] + 0.012, ys[0] + 0.012), fontsize=9.5)
    ax.annotate(f"{ys[-1]:.2f}", xy=(xs[-1], ys[-1]), xytext=(xs[-1] - 0.055, ys[-1] + 0.012), fontsize=9.5)
    ax.set_xlabel(r"Market thickness $\kappa$")
    ax.set_ylabel(r"Stake for 95\% honesty")
    ax.set_ylim(min(ys) * 0.94, max(ys) * 1.12)
    save_figure(fig, FIGURES / "fig11b_required_stake")


def plot_measured_verifier(audit, families):
    """Measured audit (s, d) against the assumed curves of Assumption 2."""
    import math
    fig, ax = panel()
    cvs = [float(r["cv"]) for r in audit]
    s_meas = [float(r["s_measured"]) for r in audit]
    d_meas = [float(r["d_measured"]) for r in audit]
    s0 = 4.0
    s_assum = [1.0 + (s0 - 1.0) * math.exp(-c) for c in cvs]
    d_assum = [1.0 - math.exp(-c) for c in cvs]
    ax.plot(cvs, s_meas, color="#1F3F63", marker="o", linewidth=2.5, markersize=6.5,
            markeredgecolor="white", markeredgewidth=1.0, label="$s$ measured", zorder=5)
    ax.plot(cvs, s_assum, "--", color="#1F3F63", linewidth=1.8, label="$s$ assumed", zorder=4)
    ax.set_xlabel("Verification budget $c_v$")
    ax.set_ylabel("Residual spread $s$")
    ax.set_ylim(0.9, 4.25)
    ax2 = ax.twinx()
    ax2.plot(cvs, d_meas, color="#B03030", marker="s", linewidth=2.5, markersize=6.0,
             markeredgecolor="white", markeredgewidth=1.0, label="$d$ measured", zorder=5)
    ax2.plot(cvs, d_assum, "--", color="#B03030", linewidth=1.8, label="$d$ assumed", zorder=4)
    ax2.set_ylabel("Detection probability $d$", color="#B03030")
    ax2.tick_params(axis="y", labelcolor="#B03030")
    ax2.set_ylim(-0.03, 1.03)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.6, ncol=2,
              columnspacing=0.8, handlelength=1.5, borderpad=0.35, labelspacing=0.3)
    save_figure(fig, FIGURES / "fig12a_measured_verifier")


def plot_false_positive(rows):
    """Welfare and compliant-agent payoff against the false-positive rate."""
    fig, ax = panel()
    xs = [float(r["x"]) for r in rows]
    ax.plot(xs, [float(r["welfare"]) for r in rows], color="#1F3F63", marker="o", linewidth=2.6,
            markersize=6.5, markeredgecolor="white", markeredgewidth=1.0,
            label="Social welfare", zorder=5)
    ax.plot(xs, [float(r["honesty"]) for r in rows], color="#7FA6C9", marker="^", linewidth=2.2,
            markersize=6.0, markeredgecolor="white", markeredgewidth=1.0,
            label="Honest delivery", zorder=4)
    ax.plot(xs, [float(r["honest_payoff"]) for r in rows], color="#B03030", marker="v", linewidth=2.2,
            markersize=6.0, markeredgecolor="white", markeredgewidth=1.0,
            label="Compliant payoff", zorder=4)
    phi_max = float(rows[0]["phi_max"])
    ax.axvline(phi_max, color="#B03030", linestyle=":", linewidth=1.6, zorder=3)
    ax.axhline(0.0, color="#888888", linewidth=0.9, zorder=2)
    ax.annotate(r"$\varphi_{\max}$", xy=(phi_max, 0.86), xytext=(phi_max + 0.012, 0.86),
                fontsize=9.5, color="#B03030")
    ax.set_xlabel(r"False-positive rate $\varphi$")
    ax.set_ylabel("Level")
    ax.set_ylim(-0.32, 1.06)
    ax.legend(loc="lower left", fontsize=8.2, handlelength=1.5, labelspacing=0.3)
    save_figure(fig, FIGURES / "fig12b_false_positive")


def plot_families(rows):
    """Welfare vs verification budget under alternative detection laws."""
    fig, ax = panel(figsize=(5.4, 3.5))
    styles = {"exponential": ("#1F3F63", "o"), "linear": ("#7FA6C9", "s"),
              "logarithmic": ("#B03030", "^"), "power": ("#4C8C57", "D"),
              "threshold": ("#9A6FB0", "v")}
    for fam, (col, mk) in styles.items():
        sub = rows_for(rows, fam)
        xs = [float(r["x"]) for r in sub]
        ys = [float(r["welfare"]) for r in sub]
        ax.plot(xs, ys, color=col, marker=mk, linewidth=2.2, markersize=5.8,
                markeredgecolor="white", markeredgewidth=0.9, label=fam, zorder=4)
        bi = max(range(len(ys)), key=lambda i: ys[i])
        ax.plot([xs[bi]], [ys[bi]], marker="*", markersize=13, color=col,
                markeredgecolor="black", markeredgewidth=0.7, zorder=6)
    ax.set_xlabel("Verification budget $c_v$")
    ax.set_ylabel("Social welfare per period")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False,
              fontsize=8.2, handlelength=1.2, columnspacing=0.7)
    save_figure(fig, FIGURES / "fig13_families")


def smooth(values, k: int = 9):
    """Centered moving average for display; window stated in the caption."""
    kernel = np.ones(k) / k
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="valid")


def plot_learning_trajectory(rows):
    assert_equal_series(rows, "no_reputation", "centralized", "honesty")
    fig, ax = panel()
    explore_end = 2.93  # epsilon falls below 0.1 at t = T*ln(3)/3 for T=8000
    ax.axvspan(0, explore_end, color="0.94", zorder=0)
    ax.text(explore_end / 2, 0.03, "exploration phase", ha="center", va="bottom", fontsize=9, color="#666666", style="italic")
    k = 9
    for mech, label in [("Mstar", LABELS["Mstar"]), ("no_reputation", LABELS["verification_only"]), ("receipt_only", LABELS["receipt_only"])]:
        sub = rows_for(rows, mech)
        xs = smooth([float(r["x"]) / 1000.0 for r in sub], k)
        ys = smooth([float(r["honesty"]) for r in sub], k)
        cis = smooth([float(r["honesty_ci95"]) for r in sub], k)
        color = COLORS[mech]
        ax.plot(xs, ys, color=color, linewidth=2.8, marker=MARKERS[mech], markersize=6.5, markevery=18, markeredgecolor="white", markeredgewidth=1.0, label=label, zorder=5 if mech == "Mstar" else 3)
        ax.fill_between(xs, ys - cis, ys + cis, color=color, alpha=0.18, linewidth=0, zorder=2)
    ax.set_xlabel("Period (thousands)")
    ax.set_ylabel("Honest-delivery rate")
    ax.set_ylim(0, 1.04)
    ax.legend(loc="center right")
    save_figure(fig, FIGURES / "fig8a_learning_trajectory")


def plot_learning_welfare(rows):
    fig, ax = panel()
    line_with_ci(ax, rows, "receipt_only", "welfare", "welfare_ci95")
    line_with_ci(ax, rows, "no_reputation", "welfare", "welfare_ci95", label=LABELS["verification_only"])
    line_with_ci(ax, rows, "Mstar", "welfare", "welfare_ci95")
    ax.set_xlabel("Verification budget")
    ax.set_ylabel("Social welfare per period")
    ax.set_ylim(0.10, 0.74)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper right", fontsize=9.5)
    save_figure(fig, FIGURES / "fig8b_learning_cv")


def metric_rows(rows, metric):
    return [dict(r, y=r["seed_mean"], y_ci95=r["ci95"]) for r in rows if r["metric"] == metric]


def plot_moral_hazard(rows):
    mrows = metric_rows(rows, "moral_hazard_residual")
    assert_equal_series(mrows, "no_reputation", "centralized", "seed_mean")
    fig, ax = panel()
    line_with_ci(ax, mrows, "Mstar", "y", "y_ci95")
    line_with_ci(ax, mrows, "no_reputation", "y", "y_ci95", label=LABELS["verification_only"])
    line_with_ci(ax, mrows, "receipt_only", "y", "y_ci95")
    ax.set_xlabel("Verification budget")
    ax.set_ylabel("Moral-hazard residual")
    ax.set_ylim(-0.04, 1.06)
    ax.legend(loc="center right")
    save_figure(fig, FIGURES / "fig9a_moral_hazard")


def plot_rco_grouped(rows):
    """Grouped TOTAL-overhead bars (distinct color + hatch per mechanism) with the
    verification-cost floor, common to both mechanisms, drawn as a dashed marker line
    per group. Bar height above the dashed line = dispute-handling overhead."""
    mrows = metric_rows(rows, "residual_coordination_overhead")
    xs = [0.5, 1.0, 1.5, 2.0, 3.0]
    pos = np.arange(len(xs))
    width = 0.32
    fig, ax = panel()
    for mech, style_key, offset in [("Mstar", "Mstar", -0.18), ("no_reputation", "verification_only", 0.18)]:
        totals = [float(at_x(mrows, mech, x)["rco_verification"]) + float(at_x(mrows, mech, x)["rco_dispute"]) for x in xs]
        ax.bar(pos + offset, totals, width, label=LABELS[style_key], **bar_kwargs(style_key), zorder=3)
    ax.set_xticks(pos, [f"{x:g}" for x in xs])
    ax.set_xlabel("Verification budget")
    ax.set_ylabel("Overhead share of price")
    ax.set_ylim(0, 0.40)
    ax.legend(loc="upper left", fontsize=9.5, handlelength=1.6)
    save_figure(fig, FIGURES / "fig9b_coordination_overhead")


def make_contact_sheet() -> None:
    files = [FIGURES / f"{name}.png" for name in PANEL_NAMES]
    thumbs = []
    for path in files:
        image = Image.open(path).convert("RGB")
        image.thumbnail((520, 400))
        canvas = Image.new("RGB", (560, 450), "white")
        canvas.paste(image, ((560 - image.width) // 2, 20))
        ImageDraw.Draw(canvas).text((12, 420), path.name, fill=(0, 0, 0))
        thumbs.append(canvas)
    sheet = Image.new("RGB", (1120, ((len(thumbs) + 1) // 2) * 450), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % 2) * 560, (index // 2) * 450))
    sheet.save(FIGURES / "contact_sheet.png")


def main():
    welfare = read_csv("welfare_cv.csv")
    theory = read_csv("theory_metrics.csv")
    plot_welfare(welfare)
    plot_delta(read_csv("delta_sweep.csv"))
    plot_exposure_spread(read_csv("exposure_spread.csv"))
    plot_exposure_error(read_csv("exposure_error.csv"))
    plot_stake(read_csv("stake_sweep.csv"))
    plot_sybil(read_csv("sybil.csv"))
    plot_learning_trajectory(read_csv("learning_trajectory.csv"))
    plot_learning_welfare(read_csv("learning_cv.csv"))
    plot_moral_hazard(theory)
    plot_rco_grouped(theory)
    plot_thickness(read_csv("thickness.csv"))
    plot_required_stake(read_csv("required_stake.csv"))
    plot_measured_verifier(read_csv("audit_verifier.csv"), read_csv("verifier_families.csv"))
    plot_false_positive(read_csv("false_positive.csv"))
    plot_families(read_csv("verifier_families.csv"))
    make_contact_sheet()


if __name__ == "__main__":
    main()
