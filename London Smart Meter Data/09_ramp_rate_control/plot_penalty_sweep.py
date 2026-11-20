"""Sweep of both penalties: plan deviation and grid-import ramp, against cost, sharing and violations.

PURPOSE
    Figures only: effect of the two minute-level penalty weights.
      W   (plan deviation)  W Δt Σ_t Σ_j (q_j,t − q_j^plan)²   (older runs also tracked grid and battery)
      W_r (ramp)            W_r/Δt Σ_t (r⁺_t² + r⁻_t²),  r± = excess of |ΔE_t| over the limit

Runs with the ramp limit on each building's grid import (--grid) at 10 %/min, correlated
forecast error, no clouds, for every combination of
    plan-deviation weight  0, 1, 3, 10, 30, 100 GBP per kW^2 per hour (the end-of-hour battery and
                           temperature targets stay in every run)
    ramp weight            0 (no limit: the reference run), 100, 1000, 10000 GBP per kW^2 per hour
Grouped bars (one group per ramp weight, one bar per plan-deviation weight) for:
    electricity cost, energy shared peer to peer, building-minutes over the ramp limit and the
    community's largest grid ramp.
Writes Results/09_ramp_rate_control/3_penalty_sweep/98_penalty_sweep_<measure>.png (one figure per measure) and .csv. Unfinished runs are left out.

Example:
    python plot_penalty_sweep.py
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

from plot_realtime_comparison import INK, PLAN_COLOR, _save, _style  # noqa: E402
from plot_sweep import _file  # noqa: E402
from three_timescale_ramp_limits import run_folder  # noqa: E402

RESULTS = project_paths.RESULTS_DIR / "09_ramp_rate_control" / "3_penalty_sweep"  # figures; runs in runs/
OUTPUT = RESULTS / "98_penalty_sweep"
RATE = 0.10
PLAN_WEIGHTS = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0)
RAMP_WEIGHTS = (0.0, 100.0, 1000.0, 10000.0)
PLAN_COLOURS = dict(zip(PLAN_WEIGHTS, ["#c6dbef", "#9cc3e6", "#5fa0dc", "#2a78d6", "#1d58a6", "#0f2e5c"]))
# light to dark: weaker to stronger plan tracking


def folder(plan_weight, ramp_weight):
    return run_folder(ramp_weight == 0, False, RATE, ramp_weight, True, plan_weight)


def measures(path):
    traded = pd.read_csv(_file(path, "89_traded_energy_limits"))
    dispatch = pd.read_csv(_file(path, "86_three_timescale_dispatch"))
    if len(dispatch) < 24 * 60:  # a run still in progress
        raise FileNotFoundError(f"{path} has {len(dispatch)} of 1440 minutes")
    ratio = traded["change_kwh"].abs() / traded["ramp_limit_kwh"]  # every building, against 10 %/min
    dt = 1 / 60
    return {
        "electricity_cost_gbp": float((dispatch["grid_import_kw"] * dispatch["price_gbp_per_kwh"]).sum() * dt),
        "p2p_traded_kwh": float(dispatch["p2p_traded_kw"].sum() * dt),
        "over_limit_pct": 100 * float((ratio > 1 + 1e-6).mean()),
        "largest_building_ramp_kw_per_min": float(60 * traded["change_kwh"].abs().max()),
        "community_grid_ramp_max": float(dispatch["grid_import_kw"].diff().abs().max()),
    }


PANELS = [  # (measure, title, unit, value format)
    ("electricity_cost_gbp", "Electricity cost", "GBP per day", "£{:.2f}"),
    ("p2p_traded_kwh", "Energy shared peer to peer", "kWh per day", "{:.1f}"),
    ("over_limit_pct", "Ramp-limit violations:\nbuilding-minutes over 10%/min", "% of building-minutes", "{:.2f}%"),
    ("community_grid_ramp_max", "Community grid import:\nlargest ramp", "kW/min", "{:.1f}"),
]


def main():
    rows = []
    for plan in PLAN_WEIGHTS:
        for ramp in RAMP_WEIGHTS:
            try:
                rows.append({"plan_weight": plan, "ramp_weight": ramp, **measures(folder(plan, ramp))})
            except FileNotFoundError:
                print(f"missing: plan weight {plan:g}, ramp weight {ramp:g}")
    table = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT.with_suffix(".csv"), index=False)

    plans = [p for p in PLAN_WEIGHTS if p in set(table["plan_weight"])]
    width = 0.8 / len(plans)
    for measure, title, unit, fmt in PANELS:  # one figure per measure
        fig, axis = plt.subplots(figsize=(8, 4.8))
        _style(axis)
        for index, plan in enumerate(plans):
            offset = (index - (len(plans) - 1) / 2) * width
            for position, ramp in enumerate(RAMP_WEIGHTS):
                row = table[(table["plan_weight"] == plan) & (table["ramp_weight"] == ramp)]
                if row.empty:
                    continue
                value = float(row[measure].iloc[0])
                axis.bar(position + offset, value, width=width - 0.02, color=PLAN_COLOURS[plan], linewidth=0)
                axis.annotate(fmt.format(value), (position + offset, value), xytext=(0, 2), textcoords="offset points",
                              ha="center", va="bottom", fontsize=6.5, color=INK, rotation=90)
        axis.set_xticks(np.arange(len(RAMP_WEIGHTS)))
        axis.set_xticklabels(["0\n(no limit)" if w == 0 else f"{w:g}" for w in RAMP_WEIGHTS], fontsize=9)
        axis.set_xlabel("Ramp penalty weight (GBP per kW² per hour)", fontsize=9, color=PLAN_COLOR)
        axis.margins(y=0.2)
        axis.set_ylim(bottom=0)
        axis.set_ylabel(unit, fontsize=9, color=PLAN_COLOR)
        axis.grid(False, axis="x")
        heading = title.replace(chr(10), " ")
        axis.set_title(heading + "\n10%/min ramp limit on grid imports; full day, N = 10, correlated forecast error",
                       fontsize=10, color=INK)
        handles = [plt.Rectangle((0, 0), 1, 1, color=PLAN_COLOURS[p]) for p in plans]
        axis.legend(handles, [f"{p:g}" for p in plans], title="Plan-deviation weight", fontsize=8, title_fontsize=8,
                    frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()
        _save(fig, OUTPUT.parent / f"{OUTPUT.name}_{measure}.png")
        plt.close(fig)
    print(table.round(3).to_string(index=False))
    print(OUTPUT.parent)


if __name__ == "__main__":
    main()
