#!/usr/bin/env python3
"""Generates reviewed scientific plots displaying exact point means and seed standard deviations.

Saves:
  - <out-dir>/plot_test_log_loss_reviewed.png
  - <out-dir>/plot_paired_differences_reviewed.png
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.guard import guard_output_dir

def main():
    parser = argparse.ArgumentParser(description="Generate reviewed scientific comparison plots.")
    parser.add_argument("--out-dir", required=True, help="Directory where generated plots will be saved (must be new/unprotected)")
    parser.add_argument("--results-json", default=None, help="Path to test_results_summary.json (default: results/test_results_summary.json)")
    args = parser.parse_args()

    out_dir = guard_output_dir(args.out_dir, repo_root=base_dir, allow_existing=False)
    os.makedirs(out_dir, exist_ok=True)

    input_json = os.path.abspath(args.results_json) if args.results_json else os.path.join(base_dir, "results", "test_results_summary.json")
    if not os.path.isfile(input_json):
        raise FileNotFoundError(f"Results JSON file not found: {input_json}")

    with open(input_json, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # 1. Bar plot of exact point mean log loss across models and encodings
    encodings = ["primary", "alt1", "alt2"]
    enc_labels = ["Primary", "Alt1 (Seed 1729)", "Alt2 (Seed 2718)"]
    models = ["IQP", "Bitflip", "Autoregressive", "Table"]
    model_colors = {
        "IQP": "#2b5c8f",
        "Bitflip": "#d95f02",
        "Autoregressive": "#7570b3",
        "Table": "#1b9e77"
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    x = np.arange(len(encodings))
    bar_width = 0.2

    for i, m in enumerate(models):
        chunk_means = []
        chunk_sds = []
        char_means = []
        char_sds = []
        for enc in encodings:
            sub = df[(df["encoding"] == enc) & (df["model"] == m)]
            chunk_means.append(sub["test_log_loss_chunk"].mean())
            chunk_sds.append(sub["test_log_loss_chunk"].std())
            char_means.append(sub["test_log_loss_char"].mean())
            char_sds.append(sub["test_log_loss_char"].std())
            
        offset = (i - 1.5) * bar_width
        ax1.bar(
            x + offset, chunk_means, bar_width,
            yerr=chunk_sds, capsize=4,
            label=m, color=model_colors[m], alpha=0.9
        )
        ax2.bar(
            x + offset, char_means, bar_width,
            yerr=char_sds, capsize=4,
            label=m, color=model_colors[m], alpha=0.9
        )

    ax1.set_ylabel("Held-Out Test NLL (nats/chunk)", fontweight="bold")
    ax1.set_title("Chunk Negative Log-Likelihood (Primary Metric)\n(Exact Point Means ± Across-Seed SD across 5 Fits)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(enc_labels, fontweight="bold")
    ax1.set_ylim(0, 12.0)
    ax1.grid(True, axis="y", ls="--", alpha=0.3)
    ax1.legend(loc="upper right", framealpha=0.9)

    ax2.set_ylabel("Normalized Test NLL (nats/char, chunk/3)", fontweight="bold")
    ax2.set_title("Character-Normalized Negative Log-Likelihood (/3)\n(Exact Point Means ± Across-Seed SD across 5 Fits)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(enc_labels, fontweight="bold")
    ax2.set_ylim(0, 4.0)
    ax2.grid(True, axis="y", ls="--", alpha=0.3)
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    out_bar = os.path.join(out_dir, "plot_test_log_loss_reviewed.png")
    plt.savefig(out_bar, dpi=300)
    plt.close()
    print(f"Saved: {out_bar}")

    # 2. Paired differences plot
    fig, ax = plt.subplots(figsize=(10, 5))

    baselines = ["Bitflip", "Autoregressive", "Table"]
    base_colors = {
        "Bitflip": "#d95f02",
        "Autoregressive": "#7570b3",
        "Table": "#1b9e77"
    }

    diff_data = []
    for enc in encodings:
        for b in baselines:
            diffs_char = []
            for s in [11, 23, 37, 53, 71]:
                iqp_row = df[(df["encoding"] == enc) & (df["model"] == "IQP") & (df["seed"] == s)].iloc[0]
                b_row = df[(df["encoding"] == enc) & (df["model"] == b) & (df["seed"] == s)].iloc[0]
                diffs_char.append(iqp_row["test_log_loss_char"] - b_row["test_log_loss_char"])
            diff_data.append({
                "encoding": enc,
                "baseline": b,
                "mean_diff": np.mean(diffs_char),
                "sd_diff": np.std(diffs_char, ddof=1)
            })

    df_diff = pd.DataFrame(diff_data)

    x_pos = np.arange(len(encodings))
    bar_width = 0.25

    for j, b in enumerate(baselines):
        b_sub = df_diff[df_diff["baseline"] == b]
        means = b_sub["mean_diff"].values
        sds = b_sub["sd_diff"].values
        offset = (j - 1.0) * bar_width
        ax.bar(
            x_pos + offset, means, bar_width,
            yerr=sds, capsize=4,
            label=f"IQP − {b}", color=base_colors[b], alpha=0.9
        )

    ax.axhline(0, color="black", lw=1.2, ls="--")
    ax.set_ylabel("Paired Difference in NLL (nats/char)", fontweight="bold")
    ax.set_title("Paired Performance Difference: IQP vs. Baselines (nats/char)\n(Exact Point Means ± Seed Pair SD; > 0 indicates IQP underperformance)", fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(enc_labels, fontweight="bold")
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    out_paired = os.path.join(out_dir, "plot_paired_differences_reviewed.png")
    plt.savefig(out_paired, dpi=300)
    plt.close()
    print(f"Saved: {out_paired}")

if __name__ == "__main__":
    main()
