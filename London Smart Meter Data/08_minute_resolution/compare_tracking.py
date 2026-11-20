"""What should the minute level penalise? Trade tracking against grid tracking with a terminal value.

PURPOSE
    Compare the two minute-level objectives of three_timescale.py on the same day and the same
    correlated forecast error (both with --parallel):

    trades  (previous)  + W Δt Σ_t Σ_j (q_j,t − q_j^plan)²  + W_soc (s_59 − s^plan)² + W_T (T_59 − T^plan)²
                        follow the hourly plan's pairwise trades; fixed end-of-hour state targets
    grid    (new)       + W_g Δt Σ_t (g_t − g^plan)²  − μ_soc s_59 − μ_T T_59  + small quadratics
                        follow the plan's grid import, trades free to re-route; end-of-hour state
                        valued at the hourly solve's shadow prices (the rest of the day)

Reads   Results/08_minute_resolution/3_tracking_comparison/runs/<mode>/85_*.csv, 86_*.csv
        (made by:  three_timescale.py --correlated --parallel --tracking <mode> --output-dir runs/<mode>)
Writes  Results/08_minute_resolution/3_tracking_comparison/
            95_tracking_comparison.csv                 one row per mode, every measure
            95_tracking_comparison_<measure>.png       one bar figure per measure
            96_tracking_timeseries_<measure>.png       one time-series figure per measure

Examples:
    python compare_tracking.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import project_paths  # noqa: E402  (shared modules on sys.path; results -> Results/<experiment>/)

project_paths.register(__file__)

import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from plot_realtime_comparison import INK, PLAN_COLOR, _save, _style  # noqa: E402

FOLDER = project_paths.RESULTS_DIR / "08_minute_resolution" / "3_tracking_comparison"
MODES = {
    "trades": {"label": "Previous: track the plan's P2P trades", "colour": "#52514e"},
    "grid": {"label": "New: track grid import, end-of-hour state at its value", "colour": "#1baf7a"},
}
DT = 1 / 60.0  # hours per minute

# (column, title, unit, format)
MEASURES = [
    ("energy_cost_gbp", "Electricity + gas cost", "GBP per day", "£{:.2f}"),
    ("electricity_cost_gbp", "Electricity cost", "GBP per day", "£{:.2f}"),
    ("comfort_cost_gbp", "Comfort penalty", "GBP per day", "£{:.3f}"),
    ("grid_import_kwh", "Grid import", "kWh per day", "{:.1f}"),
    ("peak_grid_kw", "Peak community grid import", "kW", "{:.1f}"),
    ("p2p_traded_kwh", "Energy shared P2P", "kWh per day", "{:.1f}"),
    ("p2p_rerouted_kwh", "P2P re-routed away from the hourly plan", "kWh per day", "{:.1f}"),
    ("grid_deviation_kwh", "Grid import away from the hourly plan", "kWh per day (sum over buildings)", "{:.1f}"),
    ("battery_deviation_kwh", "Battery power away from the hourly plan", "kWh per day (sum over buildings)", "{:.1f}"),
    ("end_of_day_battery_kwh", "Battery energy at the end of the day", "kWh", "{:.2f}"),
    ("minute_solve_median_s", "Minute solve time, median", "s (measured, 10 workers)", "{:.2f}"),
    ("minute_solve_max_s", "Minute solve time, max", "s (measured, 10 workers)", "{:.2f}"),
    ("minute_iterations_median", "Minute solve ADMM iterations, median", "iterations", "{:.0f}"),
]


def load(mode):
    folder = FOLDER / "runs" / mode
    dispatch = pd.read_csv(next(folder.glob("86_three_timescale_dispatch*.csv")))
    timing = pd.read_csv(next(folder.glob("85_three_timescale_timing*.csv")))
    return dispatch, timing


def summarise(dispatch, timing):
    minute = timing[timing["level"] == "minute"]
    return {
        "minutes": len(dispatch),
        "energy_cost_gbp": float((dispatch["electricity_cost_gbp"] + dispatch["gas_cost_gbp"]).sum()),
        "electricity_cost_gbp": float(dispatch["electricity_cost_gbp"].sum()),
        "gas_cost_gbp": float(dispatch["gas_cost_gbp"].sum()),
        "comfort_cost_gbp": float(dispatch["comfort_cost_gbp"].sum()),
        "grid_import_kwh": float(dispatch["grid_import_kw"].sum() * DT),
        "peak_grid_kw": float(dispatch["grid_import_kw"].max()),
        "p2p_traded_kwh": float(dispatch["p2p_traded_kw"].sum() * DT),
        "p2p_rerouted_kwh": float(dispatch["trade_deviation_from_plan_kw"].sum() * DT),
        "grid_deviation_kwh": float(dispatch["grid_deviation_from_plan_kw"].sum() * DT),
        "battery_deviation_kwh": float(dispatch["battery_deviation_from_plan_kw"].sum() * DT),
        "end_of_day_battery_kwh": float(dispatch["battery_energy_kwh"].iloc[-1]),
        "minute_solve_median_s": float(minute["parallel_s"].median()),
        "minute_solve_p95_s": float(minute["parallel_s"].quantile(0.95)),
        "minute_solve_max_s": float(minute["parallel_s"].max()),
        "minute_iterations_median": float(minute["iterations"].median()),
        "minute_iterations_max": float(minute["iterations"].max()),
        "hourly_solve_median_s": float(timing.loc[timing["level"] == "hourly", "parallel_s"].median()),
        "minute_solves_not_converged": int((~minute["converged"].astype(bool)).sum()),
    }


def bar_figure(table, column, title, unit, fmt):
    fig, axis = plt.subplots(figsize=(6.2, 4.6))
    _style(axis)
    modes = list(table.index)
    values = table[column].to_numpy()
    bars = axis.bar(range(len(modes)), values, width=0.55, color=[MODES[m]["colour"] for m in modes])
    for bar, value in zip(bars, values):
        axis.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 3),
                      textcoords="offset points", ha="center", va="bottom", fontsize=9, color=INK)
    if len(values) == 2 and abs(values[0]) > 1e-12:
        change = (values[1] - values[0]) / abs(values[0]) * 100
        title = f"{title}\nnew vs previous: {change:+.1f}%"
    axis.set_xticks(range(len(modes)))
    axis.set_xticklabels(["Previous:\ntrack P2P trades", "New: track grid,\nvalue end-of-hour state"],
                         fontsize=8)
    axis.set_ylabel(unit, fontsize=9, color=INK)
    axis.set_title(title, fontsize=10, color=INK)
    axis.margins(y=0.15)
    axis.grid(False, axis="x")
    fig.tight_layout()
    _save(fig, FOLDER / f"95_tracking_comparison_{column}.png")
    plt.close(fig)


def _clock(axis):
    axis.set_xlabel("Time of day", fontsize=9, color=INK)
    axis.set_xticks(range(0, 25, 3))
    axis.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x) % 24:02d}:00"))


def timeseries_figure(runs, name, title, ylabel, series, plan=None, hourly_mean=False):
    """One measure through the day for both modes (minute values, or hourly means for noisy series)."""
    fig, axis = plt.subplots(figsize=(11, 4.4))
    _style(axis)
    for mode, (dispatch, _timing) in runs.items():
        values = series(dispatch)
        if hourly_mean:
            values = values.groupby(dispatch["hour"]).mean()
            axis.step(np.append(values.index, values.index[-1] + 1), np.append(values, values.iloc[-1]),
                      where="post", color=MODES[mode]["colour"], lw=1.8, label=MODES[mode]["label"])
        else:
            axis.plot(dispatch["minute_of_day"] / 60.0, values, color=MODES[mode]["colour"], lw=1.2,
                      label=MODES[mode]["label"])
    if plan is not None:
        dispatch = runs["trades"][0]
        axis.plot(dispatch["minute_of_day"] / 60.0, plan(dispatch), color=PLAN_COLOR, lw=1, ls=":",
                  label="Hourly plan (previous run)")
    axis.axhline(0, color=PLAN_COLOR, lw=0.8)
    axis.set_ylabel(ylabel, fontsize=9, color=INK)
    axis.set_title(title, fontsize=10, color=INK)
    _clock(axis)
    axis.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    _save(fig, FOLDER / f"96_tracking_timeseries_{name}.png")
    plt.close(fig)


def main():
    runs = {mode: load(mode) for mode in MODES if (FOLDER / "runs" / mode).exists()}
    if set(runs) != set(MODES):
        raise SystemExit(f"Need both runs in {FOLDER / 'runs'} (found: {sorted(runs)}).")
    table = pd.DataFrame({mode: summarise(*data) for mode, data in runs.items()}).T
    table.index.name = "tracking"
    table.to_csv(FOLDER / "95_tracking_comparison.csv")
    for column, title, unit, fmt in MEASURES:
        bar_figure(table, column, title, unit, fmt)

    timeseries_figure(runs, "grid_deviation", "Grid import away from the hourly plan (hourly mean)",
                      "Sum over buildings of |g − g^plan| (kW)",
                      lambda d: d["grid_deviation_from_plan_kw"], hourly_mean=True)
    timeseries_figure(runs, "p2p_rerouted", "P2P trades away from the hourly plan (hourly mean)",
                      "Sum over pairs of |q − q^plan| (kW)",
                      lambda d: d["trade_deviation_from_plan_kw"], hourly_mean=True)
    timeseries_figure(runs, "p2p_traded", "Energy shared P2P (hourly mean)", "P2P traded (kW)",
                      lambda d: d["p2p_traded_kw"], hourly_mean=True)
    timeseries_figure(runs, "battery_energy", "Community battery energy", "Battery energy (kWh)",
                      lambda d: d["battery_energy_kwh"])
    timeseries_figure(runs, "grid_import", "Community grid import (hourly mean)", "Grid import (kW)",
                      lambda d: d["grid_import_kw"], hourly_mean=True)
    grid, trades = runs["grid"][0], runs["trades"][0]
    cumulative = {mode: (d["electricity_cost_gbp"] + d["gas_cost_gbp"]).cumsum() for mode, (d, _) in runs.items()}
    fig, axis = plt.subplots(figsize=(11, 4.4))
    _style(axis)
    axis.plot(grid["minute_of_day"] / 60.0, cumulative["grid"] - cumulative["trades"], color=MODES["grid"]["colour"],
              lw=1.6)
    axis.axhline(0, color=PLAN_COLOR, lw=0.8)
    axis.set_ylabel("New − previous (GBP since midnight)", fontsize=9, color=INK)
    axis.set_title("Cumulative electricity + gas cost, new tracking minus previous (below 0 = new is cheaper)",
                   fontsize=10, color=INK)
    _clock(axis)
    fig.tight_layout()
    _save(fig, FOLDER / "96_tracking_timeseries_cumulative_cost_difference.png")
    plt.close(fig)
    print(table.T.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"Figures and table in {FOLDER}")


if __name__ == "__main__":
    main()
