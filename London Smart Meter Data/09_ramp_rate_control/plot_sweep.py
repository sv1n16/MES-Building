"""Ramp limit on grid imports at 10, 2 and 1 %/min: with and without the limit, one figure per measure.

PURPOSE
    Figures only: the same day with and without a grid-import ramp limit (10, 2, 1 %/min),
    to show what the limit costs and how many violations it removes.

For each ramp limit (squared penalty, weight 1000; plan-deviation weight 1), grouped bars compare the day
without the limit with the day where that limit is in the minute-level objective. The run without the
limit is measured against each limit in turn (the same day, read at 10, 2 and 1 %/min).

Two sets, runs from Results/09_ramp_rate_control/runs/:
    ramp_rate   the correlated-error day                     -> 1_ramp_rate_sweep/96_grid_ramp_sweep_<measure>.png
    clouds      the same day with passing clouds              -> 2_cloud_transients/97_grid_ramp_sweep_clouds_<measure>.png
                (three_timescale.cloud_shading)
Each with a .csv of every measure. Limits whose run has not finished are left out.

Example:
    python plot_sweep.py                  # both sets
    python plot_sweep.py --set clouds
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from plot_realtime_comparison import COLORS, INK, PLAN_COLOR, _save, _style  # noqa: E402
from three_timescale_ramp_limits import run_folder  # noqa: E402

RESULTS = project_paths.RESULTS_DIR / "09_ramp_rate_control"
WEIGHT = 1000.0
RATES = (0.10, 0.02, 0.01)
REFERENCE_RATE = 0.10  # the rate the reference run's limit was written with
CLOUDS = "passing clouds (PV -50 to -80% for 3-15 min, ~2 per hour)"
SETS = {  # name: (clouds, output stem, conditions)
    "ramp_rate": (False, RESULTS / "1_ramp_rate_sweep" / "96_grid_ramp_sweep", "correlated forecast error"),
    "clouds": (True, RESULTS / "2_cloud_transients" / "97_grid_ramp_sweep_clouds",
               f"correlated forecast error + {CLOUDS}"),
}
WITHOUT, WITH = "Without the limit", "With the limit (squared penalty, weight 1000)"
COLOURS = {WITHOUT: "#b5b4ae", WITH: COLORS["rolling_admm"]}


def _file(folder, stem):
    files = sorted(folder.glob(f"{stem}_with_error*.csv"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected one {stem} file in {folder}, found {len(files)}")
    return files[0]


def measures(folder, file_rate, rate):
    """The run in folder (its limit written at file_rate), read against the limit `rate`."""
    traded = pd.read_csv(_file(folder, "89_traded_energy_limits"))
    dispatch = pd.read_csv(_file(folder, "86_three_timescale_dispatch"))
    if len(dispatch) < 24 * 60:
        raise FileNotFoundError(f"{folder} is still running")
    limited = traded[traded["role"] != "consumer"]  # every building (grid), or the producers (p2p)
    limit = limited["ramp_limit_kwh"] / file_rate * rate
    ratio = limited["change_kwh"].abs() / limit
    dt = 1 / 60
    return {
        "over_limit_pct": 100 * float((ratio > 1 + 1e-6).mean()),
        "largest_ramp_kw_per_min": float(60 * limited["change_kwh"].abs().max()),
        "community_grid_ramp_max": float(dispatch["grid_import_kw"].diff().abs().max()),
        "community_grid_ramp_p99": float(dispatch["grid_import_kw"].diff().abs().quantile(0.99)),
        "electricity_cost_gbp": float((dispatch["grid_import_kw"] * dispatch["price_gbp_per_kwh"]).sum() * dt),
        "p2p_traded_kwh": float(dispatch["p2p_traded_kw"].sum() * dt),
        "grid_import_kwh": float(dispatch["grid_import_kw"].sum() * dt),
    }


FIGURES = [  # (measure, title, unit, value format)
    ("over_limit_pct", "Building-minutes over the grid-import ramp limit", "% of building-minutes", "{:.2f}%"),
    ("largest_ramp_kw_per_min", "Largest building grid-import ramp", "kW/min", "{:.2f}"),
    ("community_grid_ramp_max", "Community grid import: largest ramp", "kW/min", "{:.1f}"),
    ("community_grid_ramp_p99", "Community grid import: 99th-percentile ramp", "kW/min", "{:.2f}"),
    ("electricity_cost_gbp", "Electricity cost", "GBP per day", "£{:.2f}"),
    ("p2p_traded_kwh", "Energy shared peer to peer", "kWh per day", "{:.1f}"),
    ("grid_import_kwh", "Grid import", "kWh per day", "{:.0f}"),
]


def plot_set(name):
    clouds, stem, conditions = SETS[name]
    reference = run_folder(True, clouds, REFERENCE_RATE, WEIGHT)
    rows = []
    for rate in RATES:
        try:
            limited = measures(run_folder(False, clouds, rate, WEIGHT), rate, rate)
            without = measures(reference, REFERENCE_RATE, rate)
        except FileNotFoundError as missing:
            print(f"{name}: {missing}")
            continue
        rows.append({"ramp_limit_pct_per_min": 100 * rate, "run": WITHOUT, **without})
        rows.append({"ramp_limit_pct_per_min": 100 * rate, "run": WITH, **limited})
    if not rows:
        return
    table = pd.DataFrame(rows)
    stem.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(stem.with_suffix(".csv"), index=False)

    limits = list(dict.fromkeys(table["ramp_limit_pct_per_min"]))
    width = 0.38
    for measure, title, unit, fmt in FIGURES:  # one figure per measure
        fig, axis = plt.subplots(figsize=(6.8, 4.4))
        _style(axis)
        for offset, run in ((-width / 2, WITHOUT), (width / 2, WITH)):
            values = [table[(table["ramp_limit_pct_per_min"] == lim) & (table["run"] == run)][measure].iloc[0]
                      for lim in limits]
            x = np.arange(len(limits)) + offset
            axis.bar(x, values, width=width - 0.03, color=COLOURS[run], linewidth=0, label=run)
            for position, value in zip(x, values):
                axis.annotate(fmt.format(value), (position, value), xytext=(0, 2), textcoords="offset points",
                              ha="center", va="bottom", fontsize=8, color=INK)
        axis.set_xticks(np.arange(len(limits)))
        axis.set_xticklabels([f"{lim:g}%/min" for lim in limits], fontsize=9)
        axis.set_xlabel("Ramp limit on each building's grid import", fontsize=9, color=PLAN_COLOR)
        axis.margins(y=0.2)
        axis.set_ylim(bottom=0)
        axis.set_ylabel(unit, fontsize=9, color=PLAN_COLOR)
        axis.grid(False, axis="x")
        axis.legend(frameon=False, fontsize=8, loc="upper left")
        axis.set_title(f"{title}\nfull day, N = 10, plan-deviation weight 1, {conditions}", fontsize=9.5, color=INK)
        fig.tight_layout()
        _save(fig, stem.parent / f"{stem.name}_{measure}.png")
        plt.close(fig)
    print(table.round(3).to_string(index=False))
    print(stem.parent)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=(*SETS, "all"), default="all")
    args = parser.parse_args()
    for name in (SETS if args.set == "all" else [args.set]):
        plot_set(name)


if __name__ == "__main__":
    main()
