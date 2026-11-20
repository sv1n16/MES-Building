"""Time series of every building's minute grid-import ramp (kW/min) over the day, one figure per run.

PURPOSE
    Figures only: each building's minute-to-minute grid ramp against its limit.
      ramp_b,t  = g_b,t − g_b,t−1                (kW per minute)
      limit_b   = ramp rate (%/min) × (peak load + heat pump + battery charge power)

The ramp is the minute-to-minute change in the building's grid import, in kW per minute. Each
building's limit is the ramp rate (% per minute) times its rated import (peak load + heat pump +
battery charging), drawn as dashed lines in the building's colour. Minutes over the limit are circled.
All figures of a set share the y-axis range, so they can be compared side by side.

Sets of runs (Results/09_ramp_rate_control/runs/), plan-deviation weight 1, squared penalty 1000:
    ramp_rate  no limit, 10, 2 and 1 %/min    -> 1_ramp_rate_sweep/91_grid_ramp_timeseries_<run>.png
    clouds     the same with passing clouds    -> 2_cloud_transients/91_grid_ramp_timeseries_clouds_<run>.png
Runs that have not finished are left out.

Example:
    python plot_ramp_timeseries.py                         # both sets
    python plot_ramp_timeseries.py --set clouds --hours 16 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import project_paths  # noqa: E402

project_paths.register(__file__)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from plot_realtime_comparison import INK, PLAN_COLOR, _save, _style  # noqa: E402
from three_timescale_ramp_limits import run_folder  # noqa: E402

RESULTS = project_paths.RESULTS_DIR / "09_ramp_rate_control"
BUILDING_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cc7", "#d4a017", "#c2417a", "#3fa7b5",
                    "#7a5230", "#5b6b2f", "#1d3f6e"]
OVER_COLOUR = "#d03b3b"
SETS = {"ramp_rate": (False, RESULTS / "1_ramp_rate_sweep", "91_grid_ramp_timeseries"),
        "clouds": (True, RESULTS / "2_cloud_transients", "91_grid_ramp_timeseries_clouds")}


def _runs(clouds):
    """(file tag, title, folder, rate the folder's limit was written at, limit shown)."""
    runs = [("no_limit", "No limit (10%/min limits shown)", run_folder(True, clouds, 0.10, 0), 0.10, 0.10)]
    runs += [(f"{100 * r:g}pct", f"{100 * r:g}%/min limit, squared penalty 1000",
              run_folder(False, clouds, r, 1000), r, r) for r in (0.10, 0.02, 0.01)]
    return runs


def ramp_rates(folder, file_rate, shown_rate):
    """Every limited building's signed ramp and its limit, kW per minute."""
    files = sorted(folder.glob("89_traded_energy_limits_with_error*.csv"))
    if len(files) != 1:
        return None
    traded = pd.read_csv(files[0])
    ramps = traded[traded["role"] != "consumer"].copy()  # every building (grid), or the producers (p2p)
    ramps["ramp_kw_per_min"] = 60 * ramps["change_kwh"]
    ramps["limit_kw_per_min"] = 60 * ramps["ramp_limit_kwh"] / file_rate * shown_rate
    ramps["hours"] = ramps["minute_of_day"] / 60.0
    return ramps


def plot_set(name, start, end):
    clouds, folder, stem = SETS[name]
    data = [(tag, title, ramp_rates(path, file_rate, shown)) for tag, title, path, file_rate, shown in _runs(clouds)]
    data = [(tag, title, ramps) for tag, title, ramps in data if ramps is not None]
    if not data:
        print(f"{name}: no finished runs")
        return
    buildings = sorted(data[0][2]["building"].unique())
    colours = {b: BUILDING_COLOURS[i % len(BUILDING_COLOURS)] for i, b in enumerate(buildings)}
    reach = max(ramps["ramp_kw_per_min"].abs().max() for _, _, ramps in data) * 1.08
    suffix = "" if (start, end) == (0, 24) else f"_{start:g}h_to_{end:g}h"
    folder.mkdir(parents=True, exist_ok=True)
    for tag, title, ramps in data:  # one figure per run, same y-axis
        fig, axis = plt.subplots(figsize=(12, 4.6))
        _style(axis)
        window = ramps[(ramps["hours"] >= start) & (ramps["hours"] <= end)]
        for building, rows in window.groupby("building"):
            axis.plot(rows["hours"], rows["ramp_kw_per_min"], lw=0.8, color=colours[building], alpha=0.9,
                      label=building)
            limit = rows["limit_kw_per_min"].iloc[0]
            for sign in (1, -1):
                axis.axhline(sign * limit, color=colours[building], lw=0.8, ls="--", alpha=0.6)
        over = window[window["ramp_kw_per_min"].abs() > window["limit_kw_per_min"] * (1 + 1e-6)]
        axis.scatter(over["hours"], over["ramp_kw_per_min"], s=18, facecolor="none", edgecolor=OVER_COLOUR,
                     lw=1.2, zorder=4, label="over the limit")
        axis.axhline(0, color=PLAN_COLOR, lw=0.5)
        share = 100 * (ramps["ramp_kw_per_min"].abs() > ramps["limit_kw_per_min"] * (1 + 1e-6)).mean()
        largest = ramps.loc[ramps["ramp_kw_per_min"].abs().idxmax()]
        axis.set_title(f"{title}{' — with passing clouds' if clouds else ''}\n{share:.2f}% of building-minutes over "
                       f"their limit; largest ramp {abs(largest['ramp_kw_per_min']):.2f} kW/min (its limit "
                       f"{largest['limit_kw_per_min']:.2f} kW/min); dashed: each building's ± limit",
                       fontsize=10, color=INK)
        axis.set_ylim(-reach, reach)
        axis.set_xlim(start, end)
        axis.set_ylabel("Change in grid import (kW/min)", fontsize=9, color=INK)
        axis.set_xlabel("Time of day", fontsize=9, color=INK)
        axis.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x) % 24:02d}:{int(round((x % 1) * 60)) % 60:02d}"))
        axis.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()
        output = folder / f"{stem}_{tag}{suffix}.png"
        _save(fig, output)
        plt.close(fig)
        print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=(*SETS, "all"), default="all")
    parser.add_argument("--hours", nargs=2, type=float, default=(0, 24), help="Time window shown (start end).")
    args = parser.parse_args()
    for name in (SETS if args.set == "all" else [args.set]):
        plot_set(name, *args.hours)


if __name__ == "__main__":
    main()
