"""Compare the three-timescale runs with and without the limits (6a)-(6c): one panel per measure.

Reads Results/09_ramp_rate_control/reference_without_limits/ and every ramp_weight_<w>/ folder
(runs with the limits, ramp penalty weight w), and writes into Results/09_ramp_rate_control/:
    90_ramp_limits_comparison.csv   the measures of every run
    90_ramp_limits_comparison.png   one bar panel per measure
    92_cost_vs_ramp_weight.png      electricity cost against the ramp penalty weight
    93_p2p_vs_ramp_weight.png       P2P traded energy against the ramp penalty weight

Example:
    python plot_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import project_paths  # noqa: E402

project_paths.register(__file__)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from plot_realtime_comparison import COLORS, INK, PLAN_COLOR, _save, _style  # noqa: E402
from three_timescale import drop_pauses  # noqa: E402

RESULTS = project_paths.RESULTS_DIR / "09_ramp_rate_control"
SUFFIX = "_with_error_correlated_parallel"
REFERENCE = "Without the limits"
WEIGHT_COLOURS = ["#9cc3e6", COLORS["rolling_admm"], "#1d3f6e"]  # light to dark: low to high weight


def find_runs():
    """The reference run and every ramp_weight_<w>/ run that has finished, by increasing weight."""
    runs = {REFERENCE: RESULTS / "reference_without_limits"}
    weights = []
    for folder in RESULTS.glob("ramp_weight_*"):
        if (folder / f"89_traded_energy_limits{SUFFIX}.csv").exists():
            weights.append((float(folder.name.split("_")[-1]), folder))
    for weight, folder in sorted(weights):
        runs[f"With the limits, ramp weight {weight:g}"] = folder
    colours = {REFERENCE: "#b5b4ae"}
    for index, run in enumerate(list(runs)[1:]):
        colours[run] = WEIGHT_COLOURS[min(index, len(WEIGHT_COLOURS) - 1)]
    return runs, colours
OUTPUT_CSV = RESULTS / "90_ramp_limits_comparison.csv"
OUTPUT_FIGURE = RESULTS / "90_ramp_limits_comparison.png"
COST_FIGURE = RESULTS / "92_cost_vs_ramp_weight.png"
SHARING_FIGURE = RESULTS / "93_p2p_vs_ramp_weight.png"


def measures(folder):
    timing, _ = drop_pauses(pd.read_csv(folder / f"85_three_timescale_timing{SUFFIX}.csv"))
    dispatch = pd.read_csv(folder / f"86_three_timescale_dispatch{SUFFIX}.csv")
    traded = pd.read_csv(folder / f"89_traded_energy_limits{SUFFIX}.csv")
    producers = traded[traded["role"] == "producer"]
    consumers = traded[traded["role"] == "consumer"]
    ratio = producers["change_kwh"].abs() / producers["ramp_limit_kwh"]
    excess = (consumers["cumulative_kwh"] - consumers["cumulative_cap_kwh"]).clip(lower=0)
    hour_end = consumers.groupby(["hour", "building"]).last()
    minute = timing[timing["level"] == "minute"]
    dt = 1 / 60
    return {
        "ramp_over_limit_pct": 100 * float((ratio > 1 + 1e-6).mean()),
        "largest_ramp_kw_per_min": float(60 * producers["change_kwh"].abs().max()),
        "cumulative_over_allocation_pct": 100 * float((excess > 1e-6).mean()),
        "hour_end_excess_kwh": float((hour_end["cumulative_kwh"] - hour_end["cumulative_cap_kwh"]).clip(lower=0).sum()),
        "electricity_cost_gbp": float((dispatch["grid_import_kw"] * dispatch["price_gbp_per_kwh"]).sum() * dt),
        "grid_import_kwh": float(dispatch["grid_import_kw"].sum() * dt),
        "p2p_traded_kwh": float(dispatch["p2p_traded_kw"].sum() * dt),
        "minute_solve_median_s": float(minute["parallel_s"].median()),
        "minute_solve_max_s": float(minute["parallel_s"].max()),
    }


PANELS = [  # (measure, title, unit, value format, lower is better)
    ("ramp_over_limit_pct", "(6a)/(6b) producer-minutes\nover the ramp limit", "% of producer-minutes", "{:.2f}%"),
    ("largest_ramp_kw_per_min", "(6a)/(6b) largest ramp", "kW/min", "{:.2f}"),
    ("cumulative_over_allocation_pct", "(6c) consumer-minutes\nover the allocation", "% of consumer-minutes", "{:.0f}%"),
    ("hour_end_excess_kwh", "(6c) excess at the end of each\nhour, summed over the day", "kWh", "{:.2f}"),
    ("electricity_cost_gbp", "Electricity cost", "GBP per day", "£{:.2f}"),
    ("minute_solve_median_s", "Minute solve, measured:\nmedian", "seconds", "{:.2f} s"),
    ("minute_solve_max_s", "Minute solve, measured:\nslowest", "seconds", "{:.2f} s"),
]


WEIGHT_PLOTS = [  # (measure, output, title, axis label, value format, change format, y-axis span)
    ("electricity_cost_gbp", COST_FIGURE, "Daily electricity cost", "Electricity cost (GBP per day)",
     "£{:.2f}", "{:+.2f}", 1.0),
    ("p2p_traded_kwh", SHARING_FIGURE, "Energy shared peer to peer", "P2P traded energy (kWh per day)",
     "{:.1f} kWh", "{:+.2f}", 4.0),
]


def plot_against_weight(table, run_colours, measure, output, title, label, fmt, change_fmt, span):
    """One measure against the ramp penalty weight (0 = the run without the limits)."""
    runs = list(table.columns)
    weights = [0.0 if run == REFERENCE else float(run.rsplit(" ", 1)[-1]) for run in runs]
    values = [table.loc[measure, run] for run in runs]
    x = list(range(len(runs)))
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    _style(axis)
    axis.plot(x, values, color=PLAN_COLOR, lw=1.5, zorder=2)
    axis.scatter(x, values, s=70, color=[run_colours[run] for run in runs], edgecolor="white", lw=2, zorder=3)
    for position, value in zip(x, values):
        change = value - values[0]
        text = fmt.format(value)
        if position:
            text += f"\n({change_fmt.format(change)}, {100 * change / values[0]:+.1f}%)"
        axis.annotate(text, (position, value), xytext=(0, 9), textcoords="offset points", ha="center",
                      va="bottom", fontsize=8, color=INK)
    axis.set_xticks(x)
    axis.set_xticklabels(["0\n(no limits)" if w == 0 else f"{w:g}" for w in weights])
    axis.set_xlim(-0.4, len(x) - 0.6)
    middle = float(np.mean(values))
    axis.set_ylim(middle - span / 2, middle + span / 2)
    axis.set_xlabel("Ramp penalty weight (GBP per kW² per hour)", fontsize=9, color=INK)
    axis.set_ylabel(label, fontsize=9, color=INK)
    unit = fmt.replace("{:.2f}", "1").replace("{:.1f}", f"{span:g}")
    axis.set_title(f"{title} against the ramp penalty weight\n"
                   f"full day, N = 10, ramp rate 10%/min, correlated forecast error (y axis spans {unit})",
                   fontsize=10, color=INK)
    fig.tight_layout()
    _save(fig, output)
    plt.close(fig)


def main():
    run_folders, run_colours = find_runs()
    table = pd.DataFrame({run: measures(folder) for run, folder in run_folders.items()})
    table.to_csv(OUTPUT_CSV, index_label="measure")
    runs = list(run_folders)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(2.2 * len(PANELS) + 1, 4.8))
    for axis, (measure, title, unit, fmt) in zip(axes, PANELS):
        _style(axis)
        values = [table.loc[measure, run] for run in runs]
        x = np.arange(len(runs))
        axis.bar(x, values, width=0.62, color=[run_colours[run] for run in runs], edgecolor="white", linewidth=0)
        for position, value in zip(x, values):
            axis.annotate(fmt.format(value), (position, value), xytext=(0, 3), textcoords="offset points",
                          ha="center", va="bottom", fontsize=8, color=INK)
        axis.set_xticks([])
        axis.margins(y=0.18)
        axis.set_ylim(bottom=0)
        axis.set_title(title, fontsize=9, color=INK)
        axis.set_ylabel(unit, fontsize=8, color=PLAN_COLOR)
        axis.grid(False, axis="x")
    handles = [plt.Rectangle((0, 0), 1, 1, color=run_colours[run]) for run in runs]
    fig.legend(handles, runs, loc="lower center", ncol=len(runs), frameon=False, fontsize=9)
    fig.suptitle("Three-timescale ADMM with and without the limits (6a)-(6c) in the minute-level objective\n"
                 "full day, N = 10, ramp rate 10%/min, correlated forecast error, 10 parallel workers",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    _save(fig, OUTPUT_FIGURE)
    plt.close(fig)
    for spec in WEIGHT_PLOTS:
        plot_against_weight(table, run_colours, *spec)
    print(table.round(3).to_string())
    print(f"\n{OUTPUT_FIGURE}\n{OUTPUT_CSV}")


if __name__ == "__main__":
    main()
