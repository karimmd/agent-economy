"""House style for the two-panel experiment figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "Mstar": "#D62728",
    "no_reputation": "#2E86AB",
    "centralized": "#2E9E68",
    "verification_only": "#2E86AB",
    "receipt_only": "#7F7F7F",
    "no_stake": "#9467BD",
    "stake_only": "#E07B39",
    "geometric": "#D62728",
    "naive_mean": "#9467BD",
    "analytic": "#000000",
    "rco_verification": "#D62728",
    "rco_dispute": "#F4A261",
}

MARKERS = {
    "Mstar": "o",
    "no_reputation": "s",
    "centralized": "^",
    "verification_only": "s",
    "receipt_only": "D",
    "no_stake": "v",
    "stake_only": "^",
    "geometric": "o",
    "naive_mean": "s",
}

LABELS = {
    "Mstar": "Full mechanism",
    "Mstar_staked": "With staking",
    "no_reputation": "Verification-Only (open matching)",
    "centralized": "Verification-Only (centralized)",
    "verification_only": "Verification-Only",
    "receipt_only": "Receipt-Only",
    "no_stake": "Without staking",
    "stake_only": "Stake without reputation",
    "geometric": "Geometric-mean pricing",
    "naive_mean": "Arithmetic-mean pricing",
    "analytic": "Theoretical minimum",
    "rco_verification": "Verification cost (common)",
    "rco_dispute": "Dispute handling",
}

HATCH_BARS = {
    "Mstar": "",
    "verification_only": "///",
    "receipt_only": "...",
}

HATCHES = {
    "Mstar": "",
    "verification_only": "///",
    "receipt_only": "...",
    "no_stake": "///",
    "rco_verification": "",
    "rco_dispute": "///",
}


def setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.labelweight": "normal",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "black",
            "legend.fancybox": False,
            "axes.linewidth": 1.5,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel(figsize=(4.6, 3.5)):
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def line_with_ci(ax, rows, mechanism: str, y: str, ci: str, label: str | None = None, linestyle: str = "-"):
    xs = [float(r["x"]) for r in rows if r["mechanism"] == mechanism]
    ys = [float(r[y]) for r in rows if r["mechanism"] == mechanism]
    cis = [float(r[ci]) if r.get(ci, "") not in ("", "inf") else 0.0 for r in rows if r["mechanism"] == mechanism]
    color = COLORS[mechanism]
    zorder = 5 if mechanism in ("Mstar", "geometric") else 3
    ax.plot(
        xs,
        ys,
        linestyle,
        color=color,
        marker=MARKERS[mechanism],
        linewidth=2.8,
        markersize=8.5,
        markeredgecolor="white",
        markeredgewidth=1.2,
        label=label or LABELS[mechanism],
        zorder=zorder,
    )
    ax.fill_between(xs, [v - c for v, c in zip(ys, cis)], [v + c for v, c in zip(ys, cis)], color=color, alpha=0.15, linewidth=0, zorder=zorder - 1)


def bar_kwargs(key: str) -> dict:
    return {
        "color": COLORS[key],
        "edgecolor": "black",
        "linewidth": 1.5,
        "hatch": HATCHES.get(key, ""),
        "alpha": 0.92,
    }


def error_kwargs() -> dict:
    return {"ecolor": "black", "elinewidth": 1.7, "capsize": 4, "capthick": 1.7}


def save_figure(fig, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05, facecolor="white")
    fig.savefig(outpath.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
