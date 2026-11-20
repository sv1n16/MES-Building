"""Proportional (linear) ramp penalty against the squared one and both together.

PURPOSE
    Figures only: squared vs proportional penalty on ramp excess r± ≥ 0.
      squared       W_r/Δt Σ_t (r⁺_t² + r⁻_t²)   (soft: small excess almost free)
      proportional  W_lin Σ_t (r⁺_t + r⁻_t)      (exact penalty: excess cut to 0 once W_lin is large enough)

Grid-import ramp limit at 10 %/min, plan-deviation weight 1, correlated forecast error, no clouds:
    No limit                 runs/plan1_no_limit/
    Squared only             runs/plan1_10pct_squared1000/            1000 GBP per kW^2 per hour
    Proportional only        runs/plan1_10pct_linear1000/             1000 GBP per kWh of excess
    Both                     runs/plan1_10pct_squared1000_linear1000/
One figure per measure (one bar per penalty), in Results/09_ramp_rate_control/4_penalty_type/:
    100_penalty_type_<measure>.png, and 100_penalty_type.csv with every measure.
Runs that have not finished are left out.

Example:
    python plot_penalty_type.py
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
import pandas as pd  # noqa: E402

from plot_realtime_comparison import INK, PLAN_COLOR, _save, _style  # noqa: E402
from plot_sweep import _file  # noqa: E402
from three_timescale_ramp_limits import run_folder  # noqa: E402

OUTPUT = project_paths.RESULTS_DIR / "09_ramp_rate_control" / "4_penalty_type"
RUNS = {  # (run folder, bar colour)
    "No limit": (run_folder(True, False, 0.10, 0), "#b5b4ae"),
    "Squared only": (run_folder(False, False, 0.10, 1000, penalty="quadratic"), "#9cc3e6"),
    "Proportional only": (run_folder(False, False, 0.10, 1000, penalty="linear"), "#2a78d6"),
    "Both": (run_folder(False, False, 0.10, 1000, penalty="both"), "#0f2e5c"),
}
COUNT_THRESHOLD = 1e-6  # kWh per minute over the limit counted as a violation (as in plot_penalty_sweep)


def measures(folder):
    traded = pd.read_csv(_file(folder, "89_traded_energy_limits"))
    dispatch = pd.read_csv(_file(folder, "86_three_timescale_dispatch"))
    if len(dispatch) < 24 * 60:
        raise FileNotFoundError(f"{folder} is still running")
    timing = pd.read_csv(_file(folder, "85_three_timescale_timing"))
    minute = timing[timing["level"] == "minute"]["parallel_s"]
    excess = (traded["change_kwh"].abs() - traded["ramp_limit_kwh"]).clip(lower=0) * 60  # kW/min over the limit
    dt = 1 / 60
    return {
        "minutes_over_limit": int((excess > COUNT_THRESHOLD * 60).sum()),
        "total_excess_kw_per_min": float(excess.sum()),
        "largest_excess_kw_per_min": float(excess.max()),
        "community_largest_ramp_kw_per_min": float(dispatch["grid_import_kw"].diff().abs().max()),
        "electricity_cost_gbp": float((dispatch["grid_import_kw"] * dispatch["price_gbp_per_kwh"]).sum() * dt),
        "p2p_traded_kwh": float(dispatch["p2p_traded_kw"].sum() * dt),
        "minute_solve_median_s": float(minute.median()),
        "minute_solve_max_s": float(minute.max()),
    }


FIGURES = [  # (measure, title, axis label, value format)
    ("minutes_over_limit", "Building-minutes over the 10%/min grid-import ramp limit", "building-minutes", "{:.0f}"),
    ("total_excess_kw_per_min", "Total ramp excess over the day (sum over building-minutes)", "kW/min", "{:.3f}"),
    ("largest_excess_kw_per_min", "Largest ramp excess in any building-minute", "kW/min", "{:.4f}"),
    ("community_largest_ramp_kw_per_min", "Community grid import: largest ramp", "kW/min", "{:.1f}"),
    ("electricity_cost_gbp", "Electricity cost", "GBP per day", "£{:.2f}"),
    ("p2p_traded_kwh", "Energy shared peer to peer", "kWh per day", "{:.1f}"),
    ("minute_solve_median_s", "Minute solve, measured: median", "seconds", "{:.2f} s"),
    ("minute_solve_max_s", "Minute solve, measured: slowest", "seconds", "{:.1f} s"),
]


def main():
    rows = {}
    for run, (folder, _) in RUNS.items():
        try:
            rows[run] = measures(folder)
        except FileNotFoundError:
            print(f"not finished: {run}")
    table = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "100_penalty_type.csv", index_label="measure")
    runs = list(table.columns)
    for measure, title, label, fmt in FIGURES:
        fig, axis = plt.subplots(figsize=(6.4, 4.2))
        _style(axis)
        values = [table.loc[measure, run] for run in runs]
        bars = axis.bar(range(len(runs)), values, width=0.6, color=[RUNS[r][1] for r in runs], linewidth=0)
        for bar, value in zip(bars, values):
            axis.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 3),
                          textcoords="offset points", ha="center", va="bottom", fontsize=9, color=INK)
        axis.set_xticks(range(len(runs)))
        axis.set_xticklabels(runs, fontsize=9)
        axis.margins(y=0.15)
        axis.set_ylim(bottom=0)
        axis.set_ylabel(label, fontsize=9, color=PLAN_COLOR)
        axis.grid(False, axis="x")
        axis.set_title(f"{title}\nramp limit on grid imports, 10%/min; plan-deviation weight 1; full day, N = 10",
                       fontsize=10, color=INK)
        fig.tight_layout()
        _save(fig, OUTPUT / f"100_penalty_type_{measure}.png")
        plt.close(fig)
    print(table.round(4).to_string())
    print(OUTPUT)


if __name__ == "__main__":
    main()
