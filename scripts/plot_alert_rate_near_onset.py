"""Figure: alert rate before and after the labelled attack start, for each capture.

Usage:
    python scripts/plot_alert_rate_near_onset.py <scores_dir> [output.png]

<scores_dir> has the score files of the Kaggle runs:
    baseline/<attack>.parquet                      from notebook 02
    supervised/gbt_<attack>.parquet                from notebook 03
    confound/gbt_time_matched_<attack>.parquet     from notebook 04, VARIANT = "time_matched"
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BIN = 2_000            # packets in one point of the line
BEFORE, AFTER = 200_000, 100_000
TARGET_FPR = 1e-3

SERIES = {  # model -> (label, colour, file pattern)
    "pca_baseline": ("PCA baseline (benign traffic only)", "#2a78d6", "baseline/{a}.parquet"),
    "gbt": ("Trees, standard training (other eight captures)", "#eb6834", "supervised/gbt_{a}.parquet"),
    "gbt_time_matched": ("Trees, time-matched training", "#1baf7a", "confound/gbt_time_matched_{a}.parquet"),
}
TITLES = {
    "arp_mitm": "ARP MitM", "active_wiretap": "Active Wiretap", "fuzzing": "Fuzzing",
    "mirai_botnet": "Mirai", "os_scan": "OS Scan", "ssdp_flood": "SSDP Flood",
    "ssl_renegotiation": "SSL Renegotiation", "syn_dos": "SYN DoS", "video_injection": "Video Injection",
}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def binned_alert_rate(scores_dir: Path, attack: str, model: str, results: pd.DataFrame):
    row = results.loc[(attack, model)]
    onset = int(row["benign_packets_before_attack"])
    d = pd.read_parquet(scores_dir / SERIES[model][2].format(a=attack)).sort_values("idx")
    alerts = d["score"].to_numpy() > row["threshold"]
    rel = np.arange(len(alerts)) - onset
    keep = (rel >= -BEFORE) & (rel < AFTER)
    bins = np.floor_divide(rel[keep], BIN)
    rate = pd.Series(alerts[keep]).groupby(bins).mean() * 100
    return (rate.index.to_numpy() + 0.5) * BIN / 1000, rate.to_numpy()


def main(scores_dir: Path, out: Path):
    results = pd.concat([
        pd.read_csv(REPO / "results" / "baseline_pca.csv"),
        pd.read_csv(REPO / "results" / "supervised_leave_one_out.csv"),
        pd.read_csv(REPO / "results" / "supervised_time_matched.csv"),
    ])
    results = results[(results.target_fpr == TARGET_FPR) & (results.k == 1) & (results.n == 1)]
    results = results.set_index(["attack", "model"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.2), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, attack in zip(axes.flat, TITLES):
        ax.set_facecolor(SURFACE)
        ax.axvline(0, color=INK2, linewidth=1, zorder=1)
        for model, (label, colour, _) in SERIES.items():
            x, y = binned_alert_rate(scores_dir, attack, model, results)
            ax.plot(x, y, color=colour, linewidth=1.6, label=label, zorder=3, solid_joinstyle="round")
        ax.set_title(TITLES[attack], loc="left", fontsize=10, color=INK, fontweight="bold", pad=6)
        ax.set_ylim(-3, 103)
        ax.set_xlim(-BEFORE / 1000, AFTER / 1000)
        ax.set_yticks([0, 50, 100])
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(colors=MUTED, length=0, labelsize=8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
    axes[0, 2].annotate("labelled\nattack start", xy=(0, 80), xytext=(6, 80), color=INK2, fontsize=8, va="center")
    for ax in axes[-1]:
        ax.set_xlabel("Packets from the labelled attack start (thousands)", color=INK2, fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Packets with an alert (%)", color=INK2, fontsize=8.5)

    fig.suptitle("Alert rate before and after the labelled attack start", x=0.045, y=0.99, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.text(0.045, 0.953,
             f"Each point is the mean of {BIN:,} packets. Threshold: set on benign packets of the capture for a target "
             f"false positive rate of {TARGET_FPR:g}.", ha="left", fontsize=9, color=INK2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.04, 0.945), ncol=3, frameon=False,
               fontsize=9, labelcolor=INK2, handlelength=1.6, columnspacing=2.5)
    fig.tight_layout(rect=(0.03, 0.02, 0.99, 0.925))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print("Saved", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "reports" / "figures" / "alert_rate_near_attack_start.png")
