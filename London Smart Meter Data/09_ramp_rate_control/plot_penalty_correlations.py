"""Correlations in the penalty sweep: how the two penalties relate to the outcomes, and the trade-offs.

PURPOSE
    Figures only: how strongly the plan-deviation weight W and ramp weight W_r drive
    cost, sharing and ramp violations (Spearman rank correlation ρ_s), and the
    cost-vs-smoothness trade-off.

Uses every finished run of the plan-deviation x ramp penalty sweep (plot_penalty_sweep.py: grid-import
ramp limit at 10 %/min). Writes into Results/09_ramp_rate_control/3_penalty_sweep/, one figure each:
99_penalty_correlations_matrix.png, 99_penalty_correlations_cost_vs_<measure>.png, and the .csv:
    a) Spearman rank correlation between the penalty weights and the outcomes (cost, P2P traded energy,
       building-minutes over the ramp limit, the community's largest grid ramp)
    b-d) electricity cost against each outcome, one point per run: colour = plan-deviation weight,
       size = ramp weight. Read together, they show the price of smoothing the grid.

Example:
    python plot_penalty_correlations.py
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

from plot_penalty_sweep import PLAN_COLOURS, PLAN_WEIGHTS, RAMP_WEIGHTS, RESULTS, folder, measures  # noqa: E402
from plot_realtime_comparison import INK, PLAN_COLOR, _save, _style  # noqa: E402

OUTPUT = RESULTS / "99_penalty_correlations"
LABELS = {
    "plan_weight": "Plan-deviation\nweight",
    "ramp_weight": "Ramp\nweight",
    "electricity_cost_gbp": "Electricity\ncost",
    "p2p_traded_kwh": "P2P\nshared",
    "over_limit_pct": "Minutes over\nramp limit",
    "community_grid_ramp_max": "Community\nlargest ramp",
}
SIZES = {0.0: 25, 100.0: 60, 1000.0: 120, 10000.0: 220}
SCATTERS = [  # (x measure, x label)
    ("community_grid_ramp_max", "Community grid import: largest ramp (kW/min)"),
    ("over_limit_pct", "Building-minutes over the 10%/min limit (%)"),
    ("p2p_traded_kwh", "Energy shared peer to peer (kWh per day)"),
]


def collect():
    rows = []
    for plan in PLAN_WEIGHTS:
        for ramp in RAMP_WEIGHTS:
            path = folder(plan, ramp)
            try:
                row = {"plan_weight": plan, "ramp_weight": ramp, **measures(path)}
            except FileNotFoundError:
                continue
            rows.append(row)
    return pd.DataFrame(rows)


def diverging(value):
    """Blue (negative) - grey (0) - red (positive)."""
    low, mid, high = np.array([0.16, 0.47, 0.84]), np.array([0.93, 0.93, 0.92]), np.array([0.82, 0.23, 0.23])
    return tuple(mid + (high - mid) * value if value >= 0 else mid + (low - mid) * -value)


def main():
    table = collect()
    if len(table) < 4:
        print("fewer than four finished runs")
        return
    table.to_csv(OUTPUT.with_suffix(".csv"), index=False)
    columns = list(LABELS)
    corr = table[columns].corr(method="spearman")

    # a) the correlation matrix, on its own
    fig, heat = plt.subplots(figsize=(7.2, 6.2))
    for i, row in enumerate(columns):
        for j, col in enumerate(columns):
            value = corr.loc[row, col]
            heat.add_patch(plt.Rectangle((j, i), 0.96, 0.96, color=diverging(value)))
            heat.text(j + 0.48, i + 0.5, f"{value:+.2f}", ha="center", va="center", fontsize=9,
                      color="white" if abs(value) > 0.6 else INK)
    heat.set_xlim(0, len(columns))
    heat.set_ylim(len(columns), 0)
    heat.set_xticks(np.arange(len(columns)) + 0.48)
    heat.set_yticks(np.arange(len(columns)) + 0.5)
    heat.set_xticklabels([LABELS[c] for c in columns], fontsize=8.5)
    heat.set_yticklabels([LABELS[c].replace(chr(10), " ") for c in columns], fontsize=9)
    heat.tick_params(length=0)
    for side in heat.spines.values():
        side.set_visible(False)
    heat.set_title(f"Spearman rank correlation, penalty sweep ({len(table)} runs)\nblue: move in opposite "
                   "directions, red: together", fontsize=10, color=INK)
    fig.tight_layout()
    _save(fig, OUTPUT.parent / f"{OUTPUT.name}_matrix.png")
    plt.close(fig)

    # b-d) cost against each outcome, one figure each
    plans = [p for p in PLAN_WEIGHTS if p in set(table["plan_weight"])]
    ramps = [r for r in RAMP_WEIGHTS if r in set(table["ramp_weight"])]
    handles = [plt.Line2D([], [], ls="none", marker="o", markersize=8, color=PLAN_COLOURS[p]) for p in plans]
    handles += [plt.Line2D([], [], ls="none", marker="o", markersize=np.sqrt(SIZES[r]), color=PLAN_COLOR,
                           markerfacecolor="none") for r in ramps]
    labels = [f"plan {p:g}" for p in plans] + [f"ramp {r:g}" for r in ramps]
    for measure, label in SCATTERS:
        fig, axis = plt.subplots(figsize=(8, 5))
        _style(axis)
        for _, run in table.iterrows():
            axis.scatter(run[measure], run["electricity_cost_gbp"], s=SIZES[run["ramp_weight"]],
                         color=PLAN_COLOURS[run["plan_weight"]], edgecolor="white", linewidth=1.5, zorder=3)
        rho = corr.loc[measure, "electricity_cost_gbp"]
        axis.set_xlabel(label, fontsize=9, color=INK)
        axis.set_ylabel("Electricity cost (GBP per day)", fontsize=9, color=INK)
        name = LABELS[measure].replace(chr(10), " ").lower()
        axis.set_title(f"Electricity cost against {name} (Spearman ρ = {rho:+.2f})\npenalty sweep, 10%/min "
                       "grid-import ramp limit, full day, N = 10", fontsize=10, color=INK)
        axis.margins(0.12)
        axis.legend(handles, labels, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0),
                    title="colour: plan weight\nsize: ramp weight", title_fontsize=8, labelspacing=1.1)
        fig.tight_layout()
        _save(fig, OUTPUT.parent / f"{OUTPUT.name}_cost_vs_{measure}.png")
        plt.close(fig)
    print(corr.round(2).to_string())
    print(OUTPUT.parent)


if __name__ == "__main__":
    main()
