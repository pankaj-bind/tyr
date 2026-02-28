#!/usr/bin/env python3
"""
Tyr — Publication-Ready Graph Generator
==========================================
Reads benchmark results from ``Research Paper/data/paper_results_150.csv``
and generates IEEE/ACM-style figures for the ICSE / PLDI research paper.

Output
------
    Research Paper/figures/verdict_distribution.pdf
    Research Paper/figures/verdict_distribution.png
    Research Paper/figures/latency_distribution.pdf
    Research Paper/figures/latency_distribution.png

Requirements
------------
    pip install matplotlib seaborn pandas

Usage
-----
    python generate_paper_graphs.py
    python generate_paper_graphs.py --csv path/to/results.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("ERROR: `pandas` is required.  pip install pandas")

try:
    import matplotlib
    matplotlib.use("Agg")  # headless — no GUI needed
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("ERROR: `matplotlib` is required.  pip install matplotlib")

try:
    import seaborn as sns
except ImportError:
    sys.exit("ERROR: `seaborn` is required.  pip install seaborn")


# ═══════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV   = _PROJECT_ROOT / "Research Paper" / "data" / "paper_results_150.csv"
FIG_DIR       = _PROJECT_ROOT / "Research Paper" / "figures"


# ═══════════════════════════════════════════════════════════════════════
# IEEE / ACM Style Configuration
# ═══════════════════════════════════════════════════════════════════════

def _apply_ieee_style() -> None:
    """Configure matplotlib for IEEE / ACM paper aesthetics."""
    sns.set_style("whitegrid", {
        "axes.edgecolor": ".3",
        "grid.color": ".85",
        "grid.linestyle": "-",
    })
    plt.rcParams.update({
        # Fonts — serif family, matching LaTeX body text
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif",
                              "Liberation Serif", "serif"],
        "font.size":         10,
        "axes.labelsize":    11,
        "axes.titlesize":    13,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "legend.framealpha": 0.9,
        # Figure quality
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.05,
        # Grid
        "axes.grid":         True,
        "grid.alpha":        0.3,
        # Lines
        "lines.linewidth":   1.5,
        "lines.markersize":  5,
    })


# ═══════════════════════════════════════════════════════════════════════
# COLOR PALETTE
# ═══════════════════════════════════════════════════════════════════════

VERDICT_COLORS = {
    "UNSAT":   "#27ae60",   # green  — formally verified
    "SAT":     "#e74c3c",   # red    — hallucination caught
    "WARNING": "#f39c12",   # orange — concrete fallback only
    "TIMEOUT": "#95a5a6",   # gray   — timed out
    "ERROR":   "#8e44ad",   # purple — internal error
}

VERDICT_LABELS = {
    "UNSAT":   "Verified Equivalent (UNSAT)",
    "SAT":     "Hallucination Caught (SAT)",
    "WARNING": "Concrete Fallback (WARNING)",
    "TIMEOUT": "Timeout",
    "ERROR":   "Error",
}

VERDICT_ORDER = ["UNSAT", "SAT", "WARNING", "TIMEOUT", "ERROR"]


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1 — Verification Verdict Distribution  (Pie Chart)
# ═══════════════════════════════════════════════════════════════════════

def generate_verdict_pie(df: pd.DataFrame, fig_dir: Path) -> None:
    """Create a clean pie chart of verification verdicts."""
    counts = df["verdict"].value_counts()
    n_total = len(df)

    # Filter to only verdicts that appear; preserve canonical order
    labels, sizes, colors = [], [], []
    for v in VERDICT_ORDER:
        if v in counts.index and counts[v] > 0:
            labels.append(VERDICT_LABELS.get(v, v))
            sizes.append(counts[v])
            colors.append(VERDICT_COLORS.get(v, "#bdc3c7"))

    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%\n({int(round(p * n_total / 100))})",
        startangle=90,
        pctdistance=0.72,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(fontsize=9),
    )

    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight("bold")
        t.set_color("white")

    # Make percentage text readable on lighter slices
    for i, v in enumerate([v for v in VERDICT_ORDER if v in counts.index
                           and counts[v] > 0]):
        if v in ("WARNING", "TIMEOUT"):
            autotexts[i].set_color("#2c3e50")

    ax.set_title(
        f"Verification Verdict Distribution  (n = {n_total})",
        fontsize=13, fontweight="bold", pad=18,
    )

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"verdict_distribution.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1/2] verdict_distribution.{{pdf,png}} saved")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2 — Latency Distribution  (Histogram + Boxplot)
# ═══════════════════════════════════════════════════════════════════════

def generate_latency_distribution(df: pd.DataFrame, fig_dir: Path) -> None:
    """Create a combined histogram + box-plot of verification latencies."""
    latencies = pd.to_numeric(df["latency_ms"], errors="coerce").dropna()

    if latencies.empty:
        print("  [2/2] SKIPPED — no valid latency data")
        return

    mean_val   = latencies.mean()
    median_val = latencies.median()
    p95_val    = latencies.quantile(0.95)

    fig, (ax_hist, ax_box) = plt.subplots(
        2, 1, figsize=(7, 5),
        gridspec_kw={"height_ratios": [3.5, 1]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.08)

    # ── Histogram ─────────────────────────────────────────────────────
    # Choose a reasonable bin count
    n_bins = min(40, max(15, int(len(latencies) ** 0.5)))

    sns.histplot(
        latencies, bins=n_bins, color="#3498db", edgecolor="white",
        alpha=0.75, ax=ax_hist, stat="count", linewidth=0.6,
    )

    # Reference lines
    ax_hist.axvline(mean_val, color="#e74c3c", linestyle="--",
                    linewidth=1.5, label=f"Mean: {mean_val:,.0f} ms")
    ax_hist.axvline(median_val, color="#2ecc71", linestyle="-.",
                    linewidth=1.5, label=f"Median: {median_val:,.0f} ms")
    ax_hist.axvline(p95_val, color="#f39c12", linestyle=":",
                    linewidth=1.8, label=f"P95: {p95_val:,.0f} ms")

    ax_hist.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax_hist.set_ylabel("Frequency", fontsize=11)
    ax_hist.set_title(
        "Verification Latency Distribution",
        fontsize=13, fontweight="bold",
    )
    ax_hist.grid(axis="y", alpha=0.3)
    ax_hist.grid(axis="x", alpha=0.0)

    # ── Box-plot ──────────────────────────────────────────────────────
    sns.boxplot(
        x=latencies, color="#3498db", ax=ax_box,
        fliersize=3, linewidth=1.2, width=0.5,
        flierprops=dict(marker="o", markerfacecolor="#e74c3c",
                        markeredgecolor="#e74c3c", alpha=0.5),
    )
    ax_box.set_xlabel("Latency (ms)", fontsize=11)
    ax_box.grid(axis="x", alpha=0.3)
    ax_box.set_yticks([])

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"latency_distribution.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [2/2] latency_distribution.{{pdf,png}} saved")


# ═══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tyr — Generate publication-ready graphs from benchmark CSV",
    )
    ap.add_argument("--csv", default=str(DEFAULT_CSV),
                    help=f"Path to benchmark results CSV (default: {DEFAULT_CSV})")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"ERROR: CSV not found: {csv_path}\n"
                 f"       Run evaluate_dataset.py first.")

    _apply_ieee_style()

    df = pd.read_csv(csv_path)
    print(f"\n{'═' * 60}")
    print(f"  Tyr — Paper Graph Generator")
    print(f"  Input : {csv_path}  ({len(df)} rows)")
    print(f"  Output: {FIG_DIR}/")
    print(f"{'═' * 60}\n")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    generate_verdict_pie(df, FIG_DIR)
    generate_latency_distribution(df, FIG_DIR)

    print(f"\n{'═' * 60}")
    print(f"  All figures saved to {FIG_DIR}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
