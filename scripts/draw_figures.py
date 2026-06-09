#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

FIG_WIDTH_IN = 6.30

FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

def draw_f1_k_l2_heatmap() -> Path:
    K_values = [3, 5, 7, 10, 15]
    L2_thresholds = [0.8, 1.0, 1.2, 1.4, 1.6, 2.0]
    grid = np.array(
        [
            [0.0, 70.0, 86.7, 86.7, 86.7, 86.7],     # K=3
            [0.0, 73.3, 96.7, 96.7, 96.7, 96.7],     # K=5  (chosen row)
            [0.0, 76.7, 100.0, 100.0, 100.0, 100.0], # K=7
            [0.0, 76.7, 100.0, 100.0, 100.0, 100.0], # K=10
            [0.0, 76.7, 100.0, 100.0, 100.0, 100.0], # K=15
        ]
    )

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.2))
    im = ax.imshow(grid, cmap="viridis", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(L2_thresholds)))
    ax.set_xticklabels([f"≤ {t}" for t in L2_thresholds])
    ax.set_yticks(range(len(K_values)))
    ax.set_yticklabels([f"K = {k}" for k in K_values])
    ax.set_xlabel("L2 distance threshold")
    ax.set_ylabel("Top-K")

    for i, row in enumerate(grid):
        for j, val in enumerate(row):
            text_colour = "white" if val < 50 else "black"
            ax.text(
                j, i, f"{val:.1f}%",
                ha="center", va="center",
                color=text_colour, fontsize=8,
            )

    chosen_row, chosen_col = 1, 2
    rect = plt.Rectangle(
        (chosen_col - 0.5, chosen_row - 0.5), 1, 1,
        fill=False, edgecolor="red", linewidth=2.0,
    )
    ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Hit Rate (%)")

    out = FIGURES_DIR / "F1_k_l2_heatmap.png"
    fig.savefig(out, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out

def draw_f2_refusal_sweep() -> Path:
    thresholds = [0.900, 0.920, 1.020, 1.100, 1.200]
    t3_recall = [96.6, 89.7, 20.7, 3.4, 0.0]
    t12_fp    = [78.4, 71.2, 17.1, 2.7, 0.0]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.5))

    ax.plot(thresholds, t3_recall, marker="o", linewidth=2.0,
            color="#440154", label="Tier 3 refusal recall (%)")
    ax.plot(thresholds, t12_fp, marker="s", linewidth=2.0,
            color="#21918C", label="Tier 1/2 false-positive rate (%)")

    ax.axvline(1.020, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.annotate(
        "chosen threshold\nL2_REJECT_MIN = 1.020",
        xy=(1.020, 50), xytext=(1.060, 60),
        fontsize=8, color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
    )

    ax.axhline(80, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(1.16, 81.5, "≥80% T3 recall target",
            fontsize=7, color="grey", ha="right")
    ax.axhline(5, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(1.16, 6.5, "≤5% T1/T2 FP target",
            fontsize=7, color="grey", ha="right")

    ax.set_xlabel("L2_REJECT_MIN (refusal-gate threshold)")
    ax.set_ylabel("Rate (%)")
    ax.set_xlim(0.85, 1.22)
    ax.set_ylim(-5, 105)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, linestyle=":", alpha=0.4)

    out = FIGURES_DIR / "F2_refusal_sweep.png"
    fig.savefig(out, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out


def draw_f3_retriever_bar() -> Path:
    rows = [
        ("Cardio T1", 59.0, 25.6, 2.6, 100.0),
        ("Cardio T2", 54.1, 43.2, 13.5, 100.0),
        ("Endo T1",   60.6, 15.2, 0.0, 100.0),
        ("Endo T2",   52.3, 34.1, 0.0, 100.0),
        ("Gastro T1", 52.8, 41.7, 5.6, 100.0),
        ("Gastro T2", 60.0, 37.1, 2.9, 100.0),
        ("Infect T1", 50.0, 31.2, 0.0, 100.0),
        ("Infect T2", 62.5, 43.8, 0.0, 100.0),
        ("Overall",   56.2, 34.0, 3.8, 100.0),
    ]

    labels = [r[0] for r in rows]
    faiss  = [r[1] for r in rows]
    bm25   = [r[2] for r in rows]
    random = [r[3] for r in rows]
    oracle = [r[4] for r in rows]

    x = np.arange(len(labels))
    width = 0.20

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.8))

    palette = plt.get_cmap("viridis")
    c_faiss  = palette(0.20)
    c_bm25   = palette(0.55)
    c_random = palette(0.85)
    c_oracle = "#cccccc"

    ax.bar(x - 1.5 * width, faiss,  width, label="FAISS",  color=c_faiss)
    ax.bar(x - 0.5 * width, bm25,   width, label="BM25",   color=c_bm25)
    ax.bar(x + 0.5 * width, random, width, label="Random", color=c_random)
    ax.bar(x + 1.5 * width, oracle, width, label="Oracle", color=c_oracle, alpha=0.7)

    ax.axvline(x[-1] - 0.5, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Recall@5 (%)")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper left", framealpha=0.95, ncol=4, fontsize=7)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    out = FIGURES_DIR / "F3_retriever_bar.png"
    fig.savefig(out, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out


def main() -> None:
    out1 = draw_f1_k_l2_heatmap()
    out2 = draw_f2_refusal_sweep()
    out3 = draw_f3_retriever_bar()
    for p in (out1, out2, out3):
        size_kb = p.stat().st_size / 1024
        print(f"  wrote {p.relative_to(p.parents[2])}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
