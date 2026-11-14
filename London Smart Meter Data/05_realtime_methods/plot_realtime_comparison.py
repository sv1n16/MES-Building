"""Plot the three-method real-time comparison written by compare_realtime_methods.py.

Figures (Results/05_realtime_methods):
    37_three_method_grid_import_run_<n>.png  community grid import vs plan
    38_three_method_dispatch_run_<n>.png     battery, P2P, curtailment, temperature
    39_three_method_costs.png                realised cost per method across runs
    40_three_method_solve_times.png          per-interval solve time vs interval length
    46_forecast_vs_actual_inputs_run_<n>.png forecast vs actual load, PV and outdoor temperature
    47_imbalance_charge_sensitivity.png      cost vs plan as deviation from the plan is charged
    48_case_overview.png                     all sensitivity cases side by side (--overview)
    49_per_building_grid_import.png          daily grid import per building and its change vs plan
    50_per_building_cost_heatmap.png         per-building cost vs plan, local vs rolling ADMM, all cases (--overview)
    51_per_building_import_heatmap.png       per-building grid import vs plan, same layout (--overview)

With --case <name> the figures are read from and written to cases/<name>/.

Usage:
    python plot_realtime_comparison.py
    python plot_realtime_comparison.py --run 2
    python plot_realtime_comparison.py --case biased_evening
    python plot_realtime_comparison.py --overview
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import project_paths  # noqa: E402  (shared modules on sys.path; results -> Results/<experiment>/)

project_paths.register(__file__)

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import pandas as pd

BASE_DIR = project_paths.experiment_dir("05_realtime_methods")
OUTPUT_DIR = BASE_DIR
RESULTS_FILE = OUTPUT_DIR / "33_three_method_results.csv"
TIMING_FILE = OUTPUT_DIR / "34_three_method_solve_times.csv"
SUMMARY_FILE = OUTPUT_DIR / "35_three_method_summary.csv"

METHODS = ("open_loop", "fixed_schedule", "local_reoptimisation", "rolling_admm", "rolling_central")
LABELS = {
    "open_loop": "Open loop (day-ahead plan, no feedback)",
    "fixed_schedule": "Fixed schedule (plan followed, error to grid)",
    "local_reoptimisation": "Local re-optimisation (no sharing)",
    "rolling_admm": "Rolling ADMM re-optimisation",
    "rolling_central": "Rolling centralised re-optimisation",
}
COLORS = {"fixed_schedule": "#2a78d6", "local_reoptimisation": "#eb6834", "rolling_admm": "#1baf7a",
          "open_loop": "#eda100", "rolling_central": "#8b78b5"}
SHORT_LABELS = {"open_loop": "Open\nloop", "fixed_schedule": "Fixed", "local_reoptimisation": "Local",
                "rolling_admm": "Rolling\nADMM", "rolling_central": "Rolling\ncentral"}
PLAN_COLOR = "#52514e"
CASE_LABEL = ""  # set by --case; appended to figure titles
IMBALANCE_PRICES = np.linspace(0.0, 0.5, 51)  # GBP per kWh of community import away from the plan
INK = "#0b0b0b"
GRID = "#e4e3df"
CASE_NAMES = {"base": "Base", "biased_evening": "Biased evening\nload (+15%)",
              "correlated": "Correlated\nerror (rho = 0.8)",
              "biased_peaks_sunny_correlated": "Biased peaks, PV +30%\n& correlated"}
# diverging: blue = cheaper, neutral gray = no change, red = dearer
DIVERGING = LinearSegmentedColormap.from_list("cost_change", ["#2a78d6", "#f0efec", "#e34948"])


def _save(fig, output, attempts=6):
    """Save a figure, retrying while OneDrive or an open viewer briefly locks the file."""
    for attempt in range(attempts):
        try:
            fig.savefig(output, dpi=160)
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(2.0 * (attempt + 1))


def _case_suffix():
    return f" — {CASE_LABEL}" if CASE_LABEL else ""


def _style(axis):
    axis.grid(True, axis="y", color=GRID, lw=0.7)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#b5b4ae")
    axis.tick_params(colors=PLAN_COLOR, labelsize=8)


def _community(data, method, column, how="sum"):
    subset = data[data["method"] == method]
    grouped = subset.groupby("interval")[column]
    return (grouped.sum() if how == "sum" else grouped.mean()).sort_index()


def plot_grid_import(data, run):
    run_data = data[data["simulation_run"] == run]
    fig, axis = plt.subplots(figsize=(12, 5))
    plan = _community(run_data, "fixed_schedule", "planned_p_el")
    axis.plot(plan.index, plan.values, color=PLAN_COLOR, lw=2, ls=":", label="Day-ahead plan")
    for method in METHODS:
        series = _community(run_data, method, "final_p_el")
        axis.plot(series.index, series.values, color=COLORS[method], lw=2, label=LABELS[method])
    axis.set_xlabel("Interval (h)")
    axis.set_ylabel("Community grid import (kW)")
    axis.set_title(f"Community grid import, same noise scenario — run {run}{_case_suffix()}", loc="left", fontweight="bold", color=INK)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    _style(axis)
    fig.tight_layout()
    output = OUTPUT_DIR / f"37_three_method_grid_import_run_{run}.png"
    _save(fig, output)
    plt.close(fig)
    print(f"Saved {output}")


def plot_dispatch(data, run):
    run_data = data[data["simulation_run"] == run].copy()
    run_data["planned_battery_net"] = run_data["planned_discharge"] - run_data["planned_charge"]
    run_data["final_battery_net"] = run_data["final_discharge"] - run_data["final_charge"]
    panels = (
        ("Battery net discharge (kW)", "battery_net", "sum"),
        ("P2P energy delivered (kW)", "p2p_export", "sum"),
        ("Curtailed PV (kW)", "curtail", "sum"),
        ("Heat pump electrical power (kW)", "p_hp", "sum"),
        ("Mean indoor temperature (°C)", "T_in", "mean"),
    )
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 14), sharex=True)
    for axis, (title, name, how) in zip(axes, panels):
        plan = _community(run_data, "fixed_schedule", f"planned_{name}", how)
        axis.plot(plan.index, plan.values, color=PLAN_COLOR, lw=2, ls=":", label="Day-ahead plan")
        for method in METHODS:
            series = _community(run_data, method, f"final_{name}", how)
            axis.plot(series.index, series.values, color=COLORS[method], lw=2, label=LABELS[method])
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold", color=INK)
        _style(axis)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[-1].set_xlabel("Interval (h)")
    fig.suptitle(f"Dispatch by method — run {run}{_case_suffix()}", x=0.06, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output = OUTPUT_DIR / f"38_three_method_dispatch_run_{run}.png"
    _save(fig, output)
    plt.close(fig)
    print(f"Saved {output}")


def _peak_hours(data):
    """Intervals on the peak tariff, from the plan's community cost per kWh imported."""
    plan = data[data["method"] == "fixed_schedule"].groupby("interval")[["planned_electricity_cost", "planned_p_el"]].sum()
    price = plan["planned_electricity_cost"] / plan["planned_p_el"].where(plan["planned_p_el"] > 1e-9)
    return price.index[price > 0.5 * price.max()]


def _with_peak_import(summary, data):
    """Add community grid import during peak-tariff hours (kWh) to the per-run summary."""
    peak = data[data["interval"].isin(_peak_hours(data))]
    totals = peak.groupby(["simulation_run", "method"])[["final_p_el", "planned_p_el"]].sum().reset_index()
    totals = totals.rename(columns={"final_p_el": "final_peak_tariff_import_kwh",
                                    "planned_p_el": "planned_peak_tariff_import_kwh"})
    return summary.merge(totals, on=["simulation_run", "method"], how="left")


def plot_costs(summary, data):
    summary = _with_peak_import(summary, data)
    hours = _peak_hours(data)
    metrics = (
        ("final_total_cost_gbp", "Realised community cost (GBP)"),
        ("final_peak_tariff_import_kwh", f"Peak-tariff grid import, {hours.min():02d}–{hours.max() + 1:02d} h (kWh)"),
        ("final_p2p_kwh", "P2P energy delivered (kWh)"),
        ("grid_import_abs_deviation_kwh", "|Grid import − plan| (kWh)"),
        ("final_peak_grid_kw", "Peak community grid import (kW)"),
    )
    fig, axes = plt.subplots(1, len(metrics), figsize=(18.5, 4.8))
    x = np.arange(len(METHODS))
    for axis, (column, title) in zip(axes, metrics):
        for i, method in enumerate(METHODS):
            values = summary.loc[summary["method"] == method, column].to_numpy(float)
            axis.bar(i, values.mean(), width=0.6, color=COLORS[method])
            jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.zeros(1)
            axis.scatter(i + jitter, values, s=14, color=INK, zorder=3, linewidths=0)
            axis.annotate(f"{values.mean():.1f}", (i, values.mean()), textcoords="offset points",
                          xytext=(0, 4), ha="center", fontsize=8, color=INK)
        planned_column = {"final_total_cost_gbp": "planned_total_cost_gbp",
                          "final_peak_tariff_import_kwh": "planned_peak_tariff_import_kwh",
                          "final_p2p_kwh": "planned_p2p_kwh",
                          "final_peak_grid_kw": "planned_peak_grid_kw"}.get(column)
        if planned_column:
            axis.axhline(summary[planned_column].mean(), color=PLAN_COLOR, ls=":", lw=1.5, label="Day-ahead plan")
            axis.legend(frameon=False, fontsize=7, loc="lower right")
        axis.set_xticks(x)
        axis.set_xticklabels([SHORT_LABELS[method] for method in METHODS])
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold", color=INK)
        _style(axis)
    runs = summary["simulation_run"].nunique()
    fig.suptitle(f"Outcomes per method (bars = mean, dots = {runs} paired run{'s' if runs != 1 else ''}){_case_suffix()}",
                 x=0.04, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output = OUTPUT_DIR / "39_three_method_costs.png"
    _save(fig, output)
    plt.close(fig)
    print(f"Saved {output}")


def plot_building_import(data):
    """Daily grid import per building: sharing moves import between buildings, not out of the community."""
    runs = data["simulation_run"].nunique()
    daily = data.groupby(["method", "building"])[["final_p_el", "planned_p_el"]].sum() / runs
    buildings = list(dict.fromkeys(data["building"]))
    plan = daily.loc["fixed_schedule", "planned_p_el"].reindex(buildings)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True)
    x = np.arange(len(buildings))
    width = 0.8 / len(METHODS)
    for j, method in enumerate(METHODS):
        values = daily.loc[method, "final_p_el"].reindex(buildings)
        offset = x + (j - (len(METHODS) - 1) / 2) * width
        axes[0].bar(offset, values.values, width=width * 0.92, color=COLORS[method], label=LABELS[method])
        axes[1].bar(offset, (values - plan).values, width=width * 0.92, color=COLORS[method])
    axes[0].scatter(x, plan.values, marker="_", s=900, color=PLAN_COLOR, linewidths=2, zorder=3,
                    label="Day-ahead plan")
    axes[1].axhline(0, color=PLAN_COLOR, lw=1)
    axes[0].set_ylabel("Daily grid import (kWh)")
    axes[1].set_ylabel("Change vs plan (kWh)")
    axes[0].set_title("Daily grid import per building", loc="left", fontsize=10, fontweight="bold", color=INK)
    axes[1].set_title("Change from the day-ahead plan per building", loc="left", fontsize=10,
                      fontweight="bold", color=INK)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(buildings, rotation=30, ha="right")
    for axis in axes:
        _style(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=5, loc="upper left", bbox_to_anchor=(0.04, 0.95))
    fig.suptitle(f"Where each building's grid import goes (mean of {runs} run{'s' if runs != 1 else ''}; "
                 f"P2P sharing moves import between buildings){_case_suffix()}",
                 x=0.04, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output = OUTPUT_DIR / "49_per_building_grid_import.png"
    _save(fig, output)
    plt.close(fig)
    print(f"Saved {output}")


def plot_solve_times(timings):
    steps = timings[(timings["scope"] == "community") & (timings["method"] != "fixed_schedule")]
    interval_seconds = float(steps["interval_seconds"].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, column, title in (
        (axes[0], "wall_seconds", "Sequential wall time per interval (one machine)"),
        (axes[1], "parallel_latency_seconds", "Estimated latency with parallel building controllers"),
    ):
        for method in ("local_reoptimisation", "rolling_admm", "rolling_central"):
            subset = steps[steps["method"] == method].groupby("interval")[column]
            if subset.ngroups == 0:
                continue
            mean = subset.mean()
            # interval 0 reproduces the day-ahead plan: shown as an unjoined hollow marker
            realtime = mean[mean.index > 0]
            axis.plot(realtime.index, realtime.values, color=COLORS[method], lw=2, marker="o", ms=4,
                      label=LABELS[method])
            if 0 in mean.index:
                axis.plot([0], [mean[0]], ls="none", marker="o", ms=6, mfc="white", mec=COLORS[method], mew=1.5)
        axis.axhline(interval_seconds, color="#e34948", lw=1.2, ls="--")
        axis.annotate(f"Interval length ({interval_seconds:.0f} s)", (0, interval_seconds), textcoords="offset points",
                      xytext=(2, 4), fontsize=8, color=INK)
        axis.set_yscale("log")
        axis.set_xlabel("Interval (h) — remaining horizon shrinks to the right")
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold", color=INK)
        _style(axis)
    axes[0].set_ylabel("Seconds (log scale)")
    axes[0].legend(frameon=False, fontsize=8, loc="center right")
    fig.suptitle(f"Solve time per real-time step (mean over runs; hollow = interval 0, re-solves the plan)"
                 f"{_case_suffix()}",
                 x=0.04, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output = OUTPUT_DIR / "40_three_method_solve_times.png"
    _save(fig, output)
    plt.close(fig)
    print(f"Saved {output}")


def plot_inputs(data, run):
    """Forecast vs actual for every noisy input (identical across methods)."""
    inputs = data[data["method"] == "fixed_schedule"]
    rows = (
        ("Community electrical load", "load_kw", "kW", "sum"),
        ("Community PV generation", "pv_kw", "kW", "sum"),
        ("Outdoor temperature", "t_out_c", "°C", "first"),
    )
    actual_color = COLORS["fixed_schedule"]
    fig, axes = plt.subplots(len(rows), 2, figsize=(15, 12), gridspec_kw={"width_ratios": (2.6, 1)})
    for (left, right), (title, name, units, how) in zip(axes, rows):
        predicted, actual = f"predicted_{name}", f"actual_{name}"
        if predicted not in inputs or actual not in inputs:
            left.set_visible(False)
            right.set_visible(False)
            continue
        per_run = inputs.groupby(["simulation_run", "interval"])[[predicted, actual]].agg(how).reset_index()
        forecast = per_run[per_run["simulation_run"] == run].set_index("interval")[predicted]
        selected = per_run[per_run["simulation_run"] == run].set_index("interval")[actual]
        left.plot(forecast.index, forecast.values, color=INK, lw=2, label="Day-ahead forecast")
        left.plot(selected.index, selected.values, color=actual_color, lw=1.8, marker="o", ms=3.5,
                  label=f"Actual, run {run}")
        left.set_ylabel(units)
        left.set_title(f"{title}: forecast vs actual", loc="left", fontsize=10, fontweight="bold", color=INK)
        _style(left)
        left.legend(frameon=False, fontsize=8)

        # Per building (load/PV) or per interval (temperature) over all runs.
        points = inputs[[predicted, actual]] if how == "sum" else per_run[[predicted, actual]]
        error = points[actual] - points[predicted]
        low = float(min(points[predicted].min(), points[actual].min()))
        high = float(max(points[predicted].max(), points[actual].max()))
        right.plot([low, high], [low, high], color=PLAN_COLOR, lw=1, ls=":")
        right.scatter(points[predicted], points[actual], s=6, color=actual_color, alpha=0.35, linewidths=0)
        scope = "per building-interval" if how == "sum" else "per interval"
        right.set_title(
            f"{scope}, all runs\nbias {error.mean():+.2f}, MAE {error.abs().mean():.2f}, "
            f"RMSE {np.sqrt((error ** 2).mean()):.2f} {units}",
            loc="left", fontsize=9, color=INK,
        )
        right.set_xlabel(f"Forecast ({units})")
        right.set_ylabel(f"Actual ({units})")
        _style(right)
    axes[-1][0].set_xlabel("Interval (h)")
    fig.suptitle(f"Noisy inputs used by every method — run {run} shown, all runs in range and scatter{_case_suffix()}",
                 x=0.04, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output = OUTPUT_DIR / f"46_forecast_vs_actual_inputs_run_{run}.png"
    _save(fig, output)
    plt.close(fig)
    print(f"Saved {output}")


def plot_imbalance(summary):
    """Cost overrun vs plan when each kWh of community import away from the plan is charged."""
    fig, axis = plt.subplots(figsize=(11, 5.5))
    for method in METHODS:
        rows = summary[summary["method"] == method]
        overrun = rows["cost_change_vs_plan_gbp"].to_numpy(float)[:, None]
        deviation = rows["grid_import_abs_deviation_kwh"].to_numpy(float)[:, None]
        cost = overrun + deviation * IMBALANCE_PRICES[None, :]
        axis.plot(IMBALANCE_PRICES, cost.mean(axis=0), color=COLORS[method], lw=2, label=LABELS[method])
        axis.annotate(f"{cost.mean(axis=0)[-1]:.1f}", (IMBALANCE_PRICES[-1], cost.mean(axis=0)[-1]),
                      textcoords="offset points", xytext=(4, 0), va="center", fontsize=8, color=INK)
    axis.axhline(0, color=PLAN_COLOR, lw=1, ls=":")
    axis.set_xlim(0, IMBALANCE_PRICES[-1] * 1.08)
    axis.set_xlabel("Charge per kWh of community grid import away from the day-ahead plan (GBP/kWh)")
    axis.set_ylabel("Cost above day-ahead plan (GBP)")
    axis.set_title(f"Cost of forecast error when deviation from the schedule is charged{_case_suffix()}\n"
                   "mean over runs; at 0 GBP/kWh this is the tariff-only result",
                   loc="left", fontsize=10, fontweight="bold", color=INK)
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    _style(axis)
    fig.tight_layout()
    output = OUTPUT_DIR / "47_imbalance_charge_sensitivity.png"
    _save(fig, output)
    plt.close(fig)
    print(f"Saved {output}")


def plot_overview(imbalance_price=0.10):
    """Every sensitivity case side by side (reads each case's summary)."""
    from compare_realtime_methods import CASES

    base_dir = BASE_DIR
    tables = {}
    for case in CASES:
        path = (base_dir if case == "base" else base_dir / "cases" / case) / SUMMARY_FILE.name
        if path.exists() and path.stat().st_size:
            tables[case] = pd.read_csv(path)
    if not tables:
        raise FileNotFoundError("No case summaries found.")
    names = CASE_NAMES
    panels = (
        ("Cost above plan (GBP)", lambda t: t["cost_change_vs_plan_gbp"]),
        (f"Cost above plan incl. {imbalance_price:.2f} GBP/kWh deviation charge (GBP)",
         lambda t: t["cost_change_vs_plan_gbp"] + imbalance_price * t["grid_import_abs_deviation_kwh"]),
        ("|Grid import - plan| (kWh)", lambda t: t["grid_import_abs_deviation_kwh"]),
    )
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 13))
    width = 0.8 / len(METHODS)
    cases = list(tables)
    for axis, (title, metric) in zip(axes, panels):
        for j, method in enumerate(METHODS):
            for i, case in enumerate(cases):
                rows = tables[case][tables[case]["method"] == method]
                if rows.empty:
                    continue
                values = metric(rows).to_numpy(float)
                x = i + (j - (len(METHODS) - 1) / 2) * width
                axis.bar(x, values.mean(), width=width * 0.92, color=COLORS[method],
                         label=LABELS[method] if i == 0 else None)
                axis.annotate(f"{values.mean():.1f}", (x, max(values.mean(), 0)), textcoords="offset points",
                              xytext=(0, 3), ha="center", fontsize=7, color=INK)
        axis.axhline(0, color=PLAN_COLOR, lw=1)
        axis.set_xticks(range(len(cases)))
        axis.set_xticklabels([names.get(case, case) for case in cases])
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold", color=INK)
        _style(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=3, loc="upper left", bbox_to_anchor=(0.04, 0.975))
    fig.suptitle("Real-time methods across forecast-error cases (bars = mean over runs)",
                 x=0.04, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    output = base_dir / "48_case_overview.png"
    _save(fig, output)
    plt.close(fig)
    print(f"Saved {output}")


BUILDING_HEATMAPS = {
    "cost": {
        "column": "total_cost", "quantity": "cost", "units": "GBP/day", "more": "dearer",
        "title": "Per-building cost relative to the day-ahead plan, local re-optimisation vs rolling ADMM "
                 "vs rolling central (mean over runs; includes P2P settlement)",
        "file": "50_per_building_cost_heatmap.png", "format": "{:+.2f}", "zero": 0.005,
    },
    "import": {
        "column": "p_el", "quantity": "grid import", "units": "kWh/day", "more": "imports more",
        "title": "Per-building grid import relative to the day-ahead plan, local re-optimisation vs rolling ADMM "
                 "vs rolling central (mean over runs)",
        "file": "51_per_building_import_heatmap.png", "format": "{:+.1f}", "zero": 0.05,
    },
}


def plot_building_cost_heatmap():
    plot_building_heatmap("cost")


def plot_building_heatmap(kind):
    """Per-building change from plan for each re-optimiser, and the local - ADMM gap, by case.

    Rolling central gets its own panel when every plotted case has it.
    """
    from compare_realtime_methods import CASES

    spec = BUILDING_HEATMAPS[kind]
    final, planned = f"final_{spec['column']}", f"planned_{spec['column']}"
    quantity, units = spec["quantity"], spec["units"]

    base_dir = BASE_DIR
    methods = ("local_reoptimisation", "rolling_admm")
    columns = {}
    has_central = True
    for case in CASES:
        path = (base_dir if case == "base" else base_dir / "cases" / case) / RESULTS_FILE.name
        if not (path.exists() and path.stat().st_size):
            continue
        data = pd.read_csv(path, usecols=["simulation_run", "method", "building", final, planned])
        data = data[data["method"].isin(methods + ("rolling_central",))]
        runs = data.groupby("method")["simulation_run"].nunique()
        if not set(methods) <= set(runs.index):
            continue
        has_central &= "rolling_central" in runs.index
        totals = data.groupby(["method", "building"])[[final, planned]].sum()
        change = (totals[final] - totals[planned]).div(runs, level="method")
        columns[case] = change
    if not columns:
        raise FileNotFoundError("No case results with both local re-optimisation and rolling ADMM found.")

    table = pd.DataFrame(columns)  # index (method, building), one column per case
    buildings = list(dict.fromkeys(table.index.get_level_values("building")))
    local = table.loc["local_reoptimisation"].reindex(buildings)
    admm = table.loc["rolling_admm"].reindex(buildings)
    central = table.loc["rolling_central"].reindex(buildings) if has_central else None
    gap = local - admm
    cases = list(table.columns)

    shown = [local, admm] + ([central] if has_central else [])
    method_limit = max(max(float(np.nanmax(np.abs(values.values))) for values in shown), 1e-6)
    gap_limit = max(float(np.nanmax(np.abs(gap.values))), 1e-6)
    panels = [
        (f"Local re-optimisation:\n{quantity} above plan ({units})", local, method_limit),
        (f"Rolling ADMM:\n{quantity} above plan ({units})", admm, method_limit),
    ]
    if has_central:
        panels.append((f"Rolling central:\n{quantity} above plan ({units})", central, method_limit))
    panels.append(
        (f"Local − rolling ADMM ({units})\nred = local {spec['more']}, blue = ADMM {spec['more']}", gap, gap_limit)
    )
    zero_label = spec["format"].format(0.0).lstrip("+")
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(cases) * len(panels) / 3 + 5, 0.5 * len(buildings) + 3),
                             sharey=True)
    for axis, (title, values, limit) in zip(axes, panels):
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit)
        image = axis.imshow(values.to_numpy(float), cmap=DIVERGING, norm=norm, aspect="auto")
        for (row, col), value in np.ndenumerate(values.to_numpy(float)):
            if np.isfinite(value):
                label = zero_label if abs(value) < spec["zero"] else spec["format"].format(value)
                axis.text(col, row, label, ha="center", va="center", fontsize=8, color=INK)
        axis.set_xticks(range(len(cases)))
        axis.set_xticklabels([CASE_NAMES.get(case, case) for case in cases], fontsize=8)
        axis.set_yticks(range(len(buildings)))
        axis.set_yticklabels(buildings, fontsize=8)
        # 2px surface gap between cells
        axis.set_xticks(np.arange(-0.5, len(cases)), minor=True)
        axis.set_yticks(np.arange(-0.5, len(buildings)), minor=True)
        axis.grid(which="minor", color="white", linewidth=2)
        axis.tick_params(which="both", length=0, colors=PLAN_COLOR)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold", color=INK)
        colorbar = fig.colorbar(image, ax=axis, shrink=0.8, pad=0.02)
        colorbar.ax.tick_params(labelsize=8, colors=PLAN_COLOR)
        colorbar.outline.set_visible(False)
    fig.suptitle(spec["title"], x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output = base_dir / spec["file"]
    _save(fig, output)
    plt.close(fig)
    print(f"Saved {output}")


def _use_case(case):
    """Read from and write to cases/<case>/ and label the figures."""
    global OUTPUT_DIR, RESULTS_FILE, TIMING_FILE, SUMMARY_FILE, CASE_LABEL
    from compare_realtime_methods import CASES

    CASE_LABEL = CASES[case]["description"]
    if case == "base":
        return
    OUTPUT_DIR = OUTPUT_DIR / "cases" / case
    RESULTS_FILE = OUTPUT_DIR / RESULTS_FILE.name
    TIMING_FILE = OUTPUT_DIR / TIMING_FILE.name
    SUMMARY_FILE = OUTPUT_DIR / SUMMARY_FILE.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=int, default=1, help="Scenario to show in the time-series figures.")
    parser.add_argument("--case", default=None, help="Sensitivity case folder to plot (see compare_realtime_methods.CASES).")
    parser.add_argument("--overview", action="store_true", help="Only draw the all-cases overview figure.")
    args = parser.parse_args()
    if args.overview:
        plot_overview()
        for kind in BUILDING_HEATMAPS:
            plot_building_heatmap(kind)
        return
    if args.case:
        _use_case(args.case)
    data = pd.read_csv(RESULTS_FILE)
    from compare_realtime_methods import clean_suspended_timings  # pauses from sleep/lock removed

    timings = clean_suspended_timings(pd.read_csv(TIMING_FILE))
    summary = pd.read_csv(SUMMARY_FILE)
    if args.run not in set(data["simulation_run"]):
        parser.error(f"run {args.run} not found in {RESULTS_FILE}")
    plot_grid_import(data, args.run)
    plot_dispatch(data, args.run)
    plot_costs(summary, data)
    plot_building_import(data)
    plot_solve_times(timings)
    plot_inputs(data, args.run)
    plot_imbalance(summary)


if __name__ == "__main__":
    main()
