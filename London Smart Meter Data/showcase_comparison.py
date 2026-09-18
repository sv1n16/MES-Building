"""Showcase-day figures: central optimisation vs the decentralised ADMM methods.

Imports the model/solver from central_optimisation_showcase and the two ADMM
implementations, runs them on the showcase day and builds every showcase figure:

  plots/06_showcase_optimisation.{png,html}          3-panel central summary
  plots/07_building_schedules_{shared,own_battery}.{png,html}
  plots/08_energy_share.{png,html}                   building-to-building sharing
  plots/11_building_benefit.{png,html,csv}           per-building £ split of the benefit
  plots/13_sharing_savings.{png,html,csv}            per-building operating-cost saving:
                                                     central market vs exchange-ADMM (pool)
                                                     vs bilateral ADMM (pairwise P2P)
  plots/15_comfort_peak_scenarios.{png,html,csv}     per building, three panels: RMS temperature
                                                     deviation from setpoint, peak grid import (kW),
                                                     and the hour that peak falls on
                                                     — no_battery / own_battery / shared
  plots/16_comfort_peak_methods.{png,html,csv}       same three panels, central market vs the ADMM methods
  plots/17_comfort_peak_total.{png,html,csv}         community RMS temp deviation + community peak
                                                     grid import, every scenario + method side by side
  plots/18_consumption_timeseries.{png,html,csv}     community-average grid import (kW) over the day,
                                                     one line per scenario + algorithm
  plots/19_cost_comparison.{png,html,csv}            plots/05 style: demand + cumulative community
                                                     grid-electricity cost, non-sharing vs central
                                                     vs exchange-ADMM vs bilateral ADMM
  plots/computational_analysis.csv                   per-component wall time, ADMM iterations /
                                                     residuals / convergence, op cost, peak, status

All three sharing outcomes in figure 13 are settled at the SAME price
(central_optimisation_showcase.SHARE_TRADE_FEE_FRAC × grid tariff, i.e.
FEE_MODE='market'), so the per-building costs are directly comparable and any
remaining spread is the algorithm, not the tariff.

Scaled communities (N=100, N=1000) are driven by run_scaling.py, which builds the
dataset and points SHOWCASE_DATASET / SHOWCASE_PLOTS_SUBDIR (=> plots/N<NNNN>/) /
SHOWCASE_SOLVE_TIMELIMIT at this script. At large N the central MIQCP (N²·T sharing
variables) and the bilateral ADMM (N²/2 trade pairs) typically run out of memory
or hit the time limit — those are caught, recorded in computational_analysis.csv,
and the figures are built from whatever converged (exchange-ADMM scales).

Usage:
  python showcase_comparison.py                central + both ADMM solves, all figures
  python showcase_comparison.py --no-admm      central-only (skip both ADMM solves)
  python showcase_comparison.py --admm-plots   also (re)generate plots/12 and plots/14
  python run_scaling.py                        N = 100, 1000 into plots/N0100/, plots/N1000/

The P2P trade-fee sweep (plots/09) stays in central_optimisation_showcase.py:
  python central_optimisation_showcase.py --sweep
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo

import central_optimisation_showcase as C
import admm_energy_sharing as EXC
import admm_bilateral_p2p as BIL
from plot_timeseries import CONSUMPTION_COLOR, INK, MUTED
from central_optimisation_showcase import (
    PLOTS_DIR,
    n_buildings,
    time_horizon,
    dt,
    BUILDING_IDS,
    LOAD,
    PV_B,
    PRICE,
    T_OUT,
    T_SET,
    IS_HIGH,
    IS_LOW,
    CAP,
    HAS_BATTERY,
    ASSETS,
    SHOWCASE_DAY,
    SHARE_TRADE_FEE_FRAC,
    FEE_MODE,
    p_th_nom,
    gas_price,
    alpha,
    build_model,
    solve_and_extract,
    _require_solver,
)

# ---- colours (were in central_optimisation_showcase) ----
SHARED_COLOR = "#2e8b6e"
BATTERY_COLOR = "#7b4bc9"
GRID_COLOR = "#8a8f98"
PRICE_HIGH_COLOR = "#d98a29"
PRICE_LOW_COLOR = "#5b8fc9"
_ASSET_COLOR = {"battery+pv": "#2e8b6e", "battery": "#7b4bc9", "pv": "#e0a800", "none": "#8a8f98"}
EXCH_COLOR = "#3b6bb0"   # exchange-ADMM (common-pool price)
BIL_COLOR = "#c0392b"    # bilateral consensus-ADMM (pairwise P2P)


# ============================================================================
# SHARED PLOT HELPERS
# ============================================================================
def _shade_tariff(ax):
    for t in range(time_horizon):
        if IS_HIGH[t]:
            ax.axvspan(t, t + 1, color=PRICE_HIGH_COLOR, alpha=0.13, lw=0)
        elif IS_LOW[t]:
            ax.axvspan(t, t + 1, color=PRICE_LOW_COLOR, alpha=0.13, lw=0)


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, axis="y", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, time_horizon)
    ax.set_xticks(range(0, time_horizon + 1, 3))


def _hi_window():
    idx = np.where(IS_HIGH)[0]
    return (int(idx[0]), int(idx[-1]) + 1) if len(idx) else (None, None)


def _add_tariff_bands(fig, rows, cols):
    lo0, lo1 = _hi_window()
    if lo0 is None:
        return
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.add_vrect(x0=lo0, x1=lo1, fillcolor="#d98a29", opacity=0.10, line_width=0, row=r, col=c)


def _sx(y):
    y = np.asarray(y, float)
    return list(range(time_horizon)) + [time_horizon], list(y) + [y[-1]]


# ============================================================================
# 06 — central showcase summary
# ============================================================================
def plot_showcase(no_batt: dict, own: dict, shared: dict, day: str) -> None:
    edges = np.arange(time_horizon + 1)

    def step(y):  # value per hour -> step arrays over hour edges
        return edges, np.concatenate([y, y[-1:]])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 11.5), sharex=True)

    # --- Panel 1: base electrical demand, per building + community total ---
    _shade_tariff(ax1)
    for b in range(n_buildings):
        ax1.step(*step(LOAD[b]), where="post", color=CONSUMPTION_COLOR, lw=1.0, alpha=0.45)
    ax1.step(*step(LOAD.sum(axis=0)), where="post", color=INK, lw=1.9)
    ax1.plot([], [], color=CONSUMPTION_COLOR, alpha=0.6, label=f"Individual buildings (n={n_buildings})")
    ax1.plot([], [], color=INK, lw=1.9, label="Community total (base load, excl. heating)")
    ax1.fill_between([], [], color=PRICE_HIGH_COLOR, alpha=0.2, label="High tariff (67 p/kWh)")
    ax1.fill_between([], [], color=PRICE_LOW_COLOR, alpha=0.2, label="Low tariff (4 p/kWh)")
    ax1.set_ylabel("Electrical demand (kW)", color=INK, fontsize=10)
    ax1.set_title(
        f"{pd.Timestamp(day):%A %d %b %Y} — {n_buildings} diverse buildings, central optimisation",
        color=INK,
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)

    # --- Panel 2: cumulative operating cost (elec + gas + trade fee), 3 scenarios ---
    _shade_tariff(ax2)
    cc = {
        k: np.cumsum(v["elec_t"] + v["gas_t"] + v["fee_t"])
        for k, v in (("nb", no_batt), ("own", own), ("shr", shared))
    }
    ax2.step(
        *step(cc["nb"]), where="post", color=MUTED, ls="--", lw=1.6, label=f"No battery  (£{no_batt['op_cost']:.2f})"
    )
    ax2.step(
        *step(cc["own"]),
        where="post",
        color=PRICE_HIGH_COLOR,
        lw=2.4,
        label=f"Own battery each  (£{own['op_cost']:.2f})",
    )
    ax2.step(
        *step(cc["shr"]),
        where="post",
        color=SHARED_COLOR,
        lw=2.4,
        label=f"Shared battery pool  (£{shared['op_cost']:.2f})",
    )
    batt_benefit = no_batt["op_cost"] - own["op_cost"]
    share_benefit = own["op_cost"] - shared["op_cost"]
    _fm = shared.get("fee_mode")
    fee_note = (f"{shared['energy_lost_kWh']:.1f} kWh lost" if _fm == "loss"
                else f"£{shared['trade_fee']:.2f} forfeited" if _fm == "forfeit"
                else f"£{shared['market_transfer']:.2f} buyer→seller")
    ax2.annotate(
        f"sharing saves £{share_benefit:.2f}/day\n" f"({shared['shared_energy_kWh']:.1f} kWh traded, {fee_note})",
        xy=(time_horizon - 0.3, cc["shr"][-1]),
        xytext=(-160, 26),
        textcoords="offset points",
        fontsize=9,
        color=SHARED_COLOR,
        va="center",
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=SHARED_COLOR, lw=1),
    )
    ax2.set_ylabel("Cumulative community operating cost (£)\nelec + gas + community fee", color=INK, fontsize=10)
    ax2.set_title(
        f"Batteries save £{batt_benefit:.2f}/day; pooling the same "
        f"{CAP.sum():.0f} kWh of storage saves £{share_benefit:.2f} more",
        color=INK,
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    ax2.legend(frameon=False, fontsize=9, loc="upper left")

    # --- Panel 3: community grid import, own-battery vs shared, + what fills the gap ---
    _shade_tariff(ax3)
    grid_own = own["grid"].sum(axis=0)
    grid_shr = shared["grid"].sum(axis=0)
    disch_shr = shared["discharge"].sum(axis=0)
    traded = shared["share"].sum(axis=(0, 1))
    hp_shr = shared["p_hp"].sum(axis=0)
    e_own, y_own = step(grid_own)
    _, y_shr = step(grid_shr)
    ax3.fill_between(e_own, y_shr, y_own, step="post", color=SHARED_COLOR, alpha=0.18)
    ax3.step(e_own, y_own, where="post", color=PRICE_HIGH_COLOR, lw=2.2, label="Grid import — own battery each")
    ax3.step(e_own, y_shr, where="post", color=SHARED_COLOR, lw=2.2, label="Grid import — shared pool")
    ax3.step(*step(hp_shr), where="post", color=MUTED, lw=1.3, ls="--", label="of which heat-pump electricity")
    ax3.step(*step(disch_shr), where="post", color=BATTERY_COLOR, lw=1.6, label="Battery discharge (shared)")
    ax3.step(*step(traded), where="post", color=INK, lw=1.4, ls=":", label="Energy traded building↔building")
    ax3.set_ylabel("Community power (kW)", color=INK, fontsize=10)
    ax3.set_xlabel("Hour of day", color=INK, fontsize=10)
    ax3.set_title(
        "High-tariff window: shared pool (green) imports less than own batteries (orange) " "— shaded gap",
        color=INK,
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    ax3.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)

    ax1.set_ylim(0, LOAD.sum(axis=0).max() * 1.30)
    ax2.set_ylim(0, cc["nb"][-1] * 1.15)
    ax3.set_ylim(0, grid_own.max() * 1.32)
    for ax in (ax1, ax2, ax3):
        _style(ax)
    fig.tight_layout()
    out = PLOTS_DIR / "06_showcase_optimisation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_building_schedules(res: dict, day: str, scenario: str = "shared") -> None:
    """One row per building: left = electricity schedule, right = heat + temperature."""
    edges = np.arange(time_horizon + 1)

    def step(y):
        return edges, np.concatenate([y, y[-1:]])

    fig, axes = plt.subplots(n_buildings, 2, figsize=(13, 2.15 * n_buildings), sharex=True, squeeze=False)

    for b, lclid in enumerate(BUILDING_IDS):
        axE, axH = axes[b]
        for ax in (axE, axH):
            _shade_tariff(ax)

        # ---- left: electricity (kW), battery SOC on a twin axis ----
        net_batt = res["charge"][b] - res["discharge"][b]  # + charge, - discharge
        axE.step(*step(LOAD[b]), where="post", color=INK, lw=1.4, label="Base load")
        axE.step(*step(res["p_hp"][b]), where="post", color="#2e8b6e", lw=1.4, label="Heat-pump elec.")
        axE.step(*step(PV_B[b]), where="post", color="#e0a800", lw=1.4, label="PV")
        axE.step(*step(res["grid"][b]), where="post", color=GRID_COLOR, lw=1.8, label="Grid import")
        axE.step(*step(net_batt), where="post", color=BATTERY_COLOR, lw=1.4, label="Battery net (+chg/−dis)")
        axE.step(
            *step(res["recv"][b] - res["sent"][b]),
            where="post",
            color=PRICE_HIGH_COLOR,
            lw=1.4,
            ls="--",
            label="P2P net (+recv/−sent)",
        )
        axE.axhline(0, color=MUTED, lw=0.6)
        if HAS_BATTERY[b]:
            axS = axE.twinx()
            se, sy = step(res["soc"][b])
            axS.fill_between(se, sy, step="post", color="#9aa7b8", alpha=0.14, lw=0)
            axS.step(se, sy, where="post", color="#7d8da6", lw=1.0)
            axS.set_ylim(0, CAP[b] * 1.05)
            axS.set_ylabel("SOC (kWh)", color="#6f7f96", fontsize=8)
            axS.tick_params(colors="#6f7f96", labelsize=7)
        axE.set_ylabel(f"{lclid}\n{ASSETS[b]}\nElectricity (kW)", color=INK, fontsize=8)
        if b == 0:
            axE.legend(frameon=False, fontsize=7.2, loc="upper left", ncol=2)
            axE.set_title(
                f"{pd.Timestamp(day):%d %b %Y} — {scenario} scenario: per-building electricity",
                color=INK,
                fontsize=10.5,
                fontweight="bold",
                loc="left",
                pad=8,
            )

        # ---- right: heat (kW) + temperature on a twin axis ----
        axH.step(*step(res["q_hp_th"][b]), where="post", color="#3b6bb0", lw=1.4, label="HP heat")
        axH.step(*step(res["q_boiler"][b]), where="post", color="#c0392b", lw=1.4, label="Boiler heat")
        axH.step(*step(res["q_total"][b]), where="post", color=INK, lw=1.0, ls=":", label="Total heat")
        axH.set_ylabel("Heat (kW)", color=INK, fontsize=9)
        axH.set_ylim(0, max(res["q_total"][b].max(), p_th_nom) * 1.15)
        axT = axH.twinx()
        axT.step(*step(res["T_in"][b]), where="post", color="#c0392b", lw=1.3, label="T in")
        axT.step(*step(T_SET), where="post", color="#2e8b6e", lw=1.0, ls="--", label="T set")
        axT.step(*step(T_OUT), where="post", color="#5b8fc9", lw=1.0, label="T out")
        axT.set_ylabel("Temp (°C)", color=INK, fontsize=8)
        axT.tick_params(labelsize=7)
        if b == 0:
            h1, l1 = axH.get_legend_handles_labels()
            h2, l2 = axT.get_legend_handles_labels()
            axH.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.2, loc="upper left", ncol=3)
            axH.set_title(
                f"{pd.Timestamp(day):%d %b %Y} — {scenario} scenario: per-building heat & temperature",
                color=INK,
                fontsize=10.5,
                fontweight="bold",
                loc="left",
                pad=8,
            )

        for ax in (axE, axH):
            for s in ("top",):
                ax.spines[s].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.grid(True, axis="y", color="#ededed", lw=0.7)
            ax.set_axisbelow(True)
            ax.set_xlim(0, time_horizon)
            ax.set_xticks(range(0, time_horizon + 1, 6))

    axes[-1, 0].set_xlabel("Hour of day", color=INK, fontsize=9)
    axes[-1, 1].set_xlabel("Hour of day", color=INK, fontsize=9)
    fig.tight_layout()
    out = PLOTS_DIR / f"07_building_schedules_{scenario}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")


def plot_energy_share(res: dict, day: str) -> None:
    """Static (matplotlib) view of the building-to-building energy sharing."""
    share = res["share"]  # (from, to, t) kW == kWh/h
    M = share.sum(axis=2)  # cumulative kWh, sender x receiver
    net = res["sent"] - res["recv"]  # (b, t)  + = net exporter
    hourly = share.sum(axis=(0, 1))
    edges = np.arange(time_horizon + 1)

    fig, (axH, axN, axT) = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.1, 1.3, 1.0]})

    im = axH.imshow(M, cmap="Greens", aspect="auto")
    axH.set_xticks(range(n_buildings))
    axH.set_yticks(range(n_buildings))
    axH.set_xticklabels(BUILDING_IDS, rotation=90, fontsize=7)
    axH.set_yticklabels(BUILDING_IDS, fontsize=7)
    axH.set_xlabel("receiver", fontsize=9)
    axH.set_ylabel("sender", fontsize=9)
    axH.set_title(
        f"Cumulative energy shared (kWh) — {M.sum():.1f} kWh total",
        color=INK,
        fontsize=10,
        fontweight="bold",
        loc="left",
    )
    for i in range(n_buildings):
        for j in range(n_buildings):
            if M[i, j] > 0.05:
                axH.text(
                    j,
                    i,
                    f"{M[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if M[i, j] > M.max() * 0.5 else INK,
                )
    fig.colorbar(im, ax=axH, fraction=0.046, pad=0.04)

    for b in range(n_buildings):
        if np.abs(net[b]).max() < 0.05:
            continue
        axN.step(edges, np.concatenate([net[b], net[b][-1:]]), where="post", lw=1.5, label=BUILDING_IDS[b])
    axN.axhline(0, color=MUTED, lw=0.8)
    _shade_tariff(axN)
    axN.set_xlim(0, time_horizon)
    axN.legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
    axN.set_xlabel("Hour of day", fontsize=9)
    axN.set_ylabel("Net P2P position (kW)  + export / − import", fontsize=9)
    axN.set_title("Who exports, who imports, and when", color=INK, fontsize=10, fontweight="bold", loc="left")

    _shade_tariff(axT)
    axT.bar(np.arange(time_horizon) + 0.5, hourly, width=0.9, color=SHARED_COLOR, alpha=0.85)
    axT.set_xlim(0, time_horizon)
    axT.set_xlabel("Hour of day", fontsize=9)
    axT.set_ylabel("Energy shared (kWh per hour)", fontsize=9)
    axTc = axT.twinx()
    axTc.step(edges, np.concatenate([np.cumsum(hourly), [np.cumsum(hourly)[-1]]]), where="post", color=INK, lw=1.5)
    axTc.set_ylabel("cumulative kWh", fontsize=9)
    axTc.set_ylim(bottom=0)
    axT.set_title("Community energy shared per hour", color=INK, fontsize=10, fontweight="bold", loc="left")

    for ax in (axN, axT):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(True, axis="y", color="#ededed", lw=0.7)
        ax.set_axisbelow(True)
    fig.suptitle(
        f"{pd.Timestamp(day):%A %d %b %Y} — building-to-building energy sharing (shared scenario)",
        color=INK,
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = PLOTS_DIR / "08_energy_share.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Interactive (Plotly) versions -> self-contained .html files
# ---------------------------------------------------------------------------
def plotly_showcase(no_batt: dict, own: dict, shared: dict, day: str) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Base electrical demand — 10 diverse buildings",
            "Cumulative operating cost (elec + gas + trade fee)",
            "Community grid import: own battery vs shared pool",
        ),
    )
    for b in range(n_buildings):
        x, y = _sx(LOAD[b])
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                line=dict(color="rgba(59,107,176,.35)", width=1, shape="hv"),
                name="individual buildings",
                legendgroup="ind",
                showlegend=(b == 0),
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
    x, y = _sx(LOAD.sum(axis=0))
    fig.add_trace(
        go.Scatter(x=x, y=y, line=dict(color="black", width=2, shape="hv"), name="community total"), row=1, col=1
    )

    for res, nm, col in (
        (no_batt, "no battery", "#8a8f98"),
        (own, "own battery each", "#d98a29"),
        (shared, "shared battery pool", "#2e8b6e"),
    ):
        cc = np.cumsum(res["elec_t"] + res["gas_t"] + res["fee_t"])
        x, y = _sx(cc)
        dash = "dash" if nm == "no battery" else "solid"
        fig.add_trace(
            go.Scatter(
                x=x, y=y, line=dict(color=col, width=2.5, shape="hv", dash=dash), name=f"{nm} (£{res['op_cost']:.2f})"
            ),
            row=2,
            col=1,
        )

    go_ = own["grid"].sum(axis=0)
    gs_ = shared["grid"].sum(axis=0)
    for arr, nm, col in (
        (go_, "grid import — own battery", "#d98a29"),
        (gs_, "grid import — shared pool", "#2e8b6e"),
        (shared["p_hp"].sum(axis=0), "of which heat-pump elec.", "#8a8f98"),
        (shared["discharge"].sum(axis=0), "battery discharge (shared)", "#7b4bc9"),
        (shared["share"].sum(axis=(0, 1)), "energy traded B↔B", "#1a1a1a"),
    ):
        x, y = _sx(arr)
        d = "dot" if "traded" in nm else ("dash" if "heat-pump" in nm else "solid")
        fig.add_trace(go.Scatter(x=x, y=y, line=dict(color=col, width=2, shape="hv", dash=d), name=nm), row=3, col=1)

    _add_tariff_bands(fig, 3, 1)
    fig.update_yaxes(title_text="kW", row=1, col=1)
    fig.update_yaxes(title_text="£", row=2, col=1)
    fig.update_yaxes(title_text="kW", row=3, col=1)
    fig.update_xaxes(title_text="Hour of day", row=3, col=1, dtick=3)
    b_ben = no_batt["op_cost"] - own["op_cost"]
    s_ben = own["op_cost"] - shared["op_cost"]
    fig.update_layout(
        template="plotly_white",
        height=950,
        hovermode="x unified",
        title=f"{pd.Timestamp(day):%A %d %b %Y} — energy-sharing showcase  "
        f"(batteries save £{b_ben:.2f}/day, sharing £{s_ben:.2f}/day more)",
        legend=dict(orientation="h", y=-0.08),
    )
    out = PLOTS_DIR / "06_showcase_optimisation.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"saved {out}")


def plotly_building_schedules(res: dict, day: str, scenario: str) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    specs = [[{"secondary_y": True}, {"secondary_y": True}] for _ in range(n_buildings)]
    titles = []
    for bid in BUILDING_IDS:
        titles += [f"{bid} — electricity", f"{bid} — heat & temperature"]
    fig = make_subplots(
        rows=n_buildings,
        cols=2,
        shared_xaxes=True,
        specs=specs,
        vertical_spacing=0.012,
        horizontal_spacing=0.09,
        subplot_titles=titles,
    )

    def add(col, r, y, name, color, grp, sl, dash="solid", width=1.3, sec=False):
        x, yy = _sx(y)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=yy,
                name=name,
                legendgroup=grp,
                showlegend=sl,
                line=dict(color=color, width=width, shape="hv", dash=dash),
            ),
            row=r,
            col=col,
            secondary_y=sec,
        )

    for b, bid in enumerate(BUILDING_IDS):
        r, sl = b + 1, (b == 0)
        add(1, r, LOAD[b], "base load", "black", "l", sl)
        add(1, r, res["p_hp"][b], "heat-pump elec.", "#2e8b6e", "hpe", sl)
        add(1, r, PV_B[b], "PV", "#e0a800", "pv", sl)
        add(1, r, res["grid"][b], "grid import", "#8a8f98", "g", sl, width=1.8)
        add(1, r, res["charge"][b] - res["discharge"][b], "battery net (+chg/−dis)", "#7b4bc9", "bn", sl)
        add(1, r, res["recv"][b] - res["sent"][b], "P2P net (+recv/−sent)", "#d98a29", "p2p", sl, dash="dash")
        if HAS_BATTERY[b]:
            add(1, r, res["soc"][b], "SOC (kWh)", "#7d8da6", "soc", sl, width=1, sec=True)
        add(2, r, res["q_hp_th"][b], "HP heat", "#3b6bb0", "hph", sl)
        add(2, r, res["q_boiler"][b], "boiler heat", "#c0392b", "blr", sl)
        add(2, r, res["q_total"][b], "total heat", "black", "th", sl, dash="dot", width=1)
        add(2, r, res["T_in"][b], "T in", "#c0392b", "tin", sl, width=1.2, sec=True)
        add(2, r, T_SET, "T set", "#2e8b6e", "tset", sl, dash="dash", width=1, sec=True)
        add(2, r, T_OUT, "T out", "#5b8fc9", "tout", sl, width=1, sec=True)
        fig.update_yaxes(title_text="kW", row=r, col=1, secondary_y=False)
        if HAS_BATTERY[b]:
            fig.update_yaxes(title_text="SOC", row=r, col=1, secondary_y=True, range=[0, CAP[b] * 1.05])
        fig.update_yaxes(title_text="kW", row=r, col=2, secondary_y=False)
        fig.update_yaxes(title_text="°C", row=r, col=2, secondary_y=True)

    _add_tariff_bands(fig, n_buildings, 2)
    fig.update_xaxes(title_text="Hour of day", row=n_buildings, col=1, dtick=6)
    fig.update_xaxes(title_text="Hour of day", row=n_buildings, col=2, dtick=6)
    fig.update_layout(
        template="plotly_white",
        height=260 * n_buildings,
        hovermode="closest",
        title=f"{pd.Timestamp(day):%A %d %b %Y} — {scenario}: per-building schedules",
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
    )
    for a in fig.layout.annotations:
        a.font.size = 10
    out = PLOTS_DIR / f"07_building_schedules_{scenario}.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"saved {out}")


def plotly_energy_share(res: dict, day: str) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    share = res["share"]
    M = share.sum(axis=2)  # sender x receiver, kWh
    net = res["sent"] - res["recv"]
    hourly = share.sum(axis=(0, 1))

    # --- Sankey of cumulative flows (separate seller / buyer node columns) ---
    labels = [f"{b} ▶" for b in BUILDING_IDS] + [f"▶ {b}" for b in BUILDING_IDS]
    s_idx, t_idx, vals = [], [], []
    for i in range(n_buildings):
        for j in range(n_buildings):
            if M[i, j] > 1e-6:
                s_idx.append(i)
                t_idx.append(n_buildings + j)
                vals.append(float(M[i, j]))
    sankey = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(label=labels, pad=12, thickness=14, color=["#2e8b6e"] * n_buildings + ["#d98a29"] * n_buildings),
            link=dict(source=s_idx, target=t_idx, value=vals, color="rgba(46,139,110,.35)"),
        )
    )
    sankey.update_layout(
        template="plotly_white",
        height=460,
        title=f"{pd.Timestamp(day):%A %d %b %Y} — cumulative energy shared " f"({M.sum():.1f} kWh):  seller ▶ buyer",
    )

    # --- heatmap + per-hour bar + net-position lines ---
    grid = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.45, 0.55],
        subplot_titles=("Cumulative kWh: sender → receiver", "Net P2P position over the day (+export / −import)"),
    )
    grid.add_trace(
        go.Heatmap(
            z=M,
            x=BUILDING_IDS,
            y=BUILDING_IDS,
            colorscale="Greens",
            colorbar=dict(title="kWh", len=0.9),
            hovertemplate="from %{y}<br>to %{x}<br>%{z:.2f} kWh<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for b in range(n_buildings):
        if np.abs(net[b]).max() < 0.05:
            continue
        x, y = _sx(net[b])
        grid.add_trace(go.Scatter(x=x, y=y, name=BUILDING_IDS[b], line=dict(width=1.6, shape="hv")), row=1, col=2)
    grid.add_hline(y=0, line=dict(color="#8a8f98", width=1), row=1, col=2)
    grid.update_xaxes(title_text="receiver", row=1, col=1)
    grid.update_yaxes(title_text="sender", autorange="reversed", row=1, col=1)
    grid.update_xaxes(title_text="Hour of day", dtick=3, row=1, col=2)
    grid.update_yaxes(title_text="kW", row=1, col=2)
    lo0, lo1 = _hi_window()
    if lo0 is not None:
        grid.add_vrect(x0=lo0, x1=lo1, fillcolor="#d98a29", opacity=0.10, line_width=0, row=1, col=2)
    grid.update_layout(
        template="plotly_white", height=460, hovermode="closest", title="Who trades with whom, and when"
    )

    bar = go.Figure()
    bar.add_trace(go.Bar(x=list(range(time_horizon)), y=hourly, name="per hour", marker_color="#2e8b6e"))
    bar.add_trace(
        go.Scatter(
            x=list(range(time_horizon)),
            y=np.cumsum(hourly),
            name="cumulative",
            line=dict(color="#1a1a1a", width=2),
            yaxis="y2",
        )
    )
    if lo0 is not None:
        bar.add_vrect(x0=lo0 - 0.5, x1=lo1 - 0.5, fillcolor="#d98a29", opacity=0.10, line_width=0)
    bar.update_layout(
        template="plotly_white",
        height=340,
        title="Community energy shared per hour",
        xaxis_title="Hour of day",
        yaxis_title="kWh / hour",
        yaxis2=dict(title="cumulative kWh", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )

    out = PLOTS_DIR / "08_energy_share.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'>" "<title>Energy sharing between buildings</title></head><body>")
        f.write(sankey.to_html(full_html=False, include_plotlyjs=True))
        f.write(grid.to_html(full_html=False, include_plotlyjs=False))
        f.write(bar.to_html(full_html=False, include_plotlyjs=False))
        f.write("</body></html>")
    print(f"saved {out}")


# ============================================================================
# 11 — per-building benefit split
# ============================================================================
_BENEFIT_SEGS = (
    ("grid_saving", SHARED_COLOR, "grid-cost saving"),
    ("gas_saving", "#c0392b", "gas-cost saving"),
    ("trade", "#3b6bb0", "trade earnings / −forfeit"),
)


def _building_benefit_table(own: dict, shared: dict, bil: dict | None = None) -> pd.DataFrame:
    """Per-building £/day change from sharing vs the own-battery scenario.

    Central (`shared`) always. If `bil` (the admm_bilateral_p2p summary dict) is
    given, the same grid / gas / trade / net split is added for the bilateral
    consensus-ADMM as `bil_*` columns — both settled at SHARE_TRADE_FEE_FRAC ×
    tariff, so the two sets of numbers are directly comparable.
    """
    d_elec = own["elec_b"] - shared["elec_b"]          # grid-cost saving (+ = saved)
    d_gas = own["gas_b"] - shared["gas_b"]
    trade = -shared["trade_cost_b"]                    # + = building is paid / forfeits nothing
    cols = {
        "LCLid": BUILDING_IDS,
        "assets": ASSETS,
        "grid_saving": d_elec,
        "gas_saving": d_gas,
        "trade": trade,
        "net_benefit": d_elec + d_gas + trade,
        "sent_kWh": shared["sent"].sum(axis=1),
        "recv_kWh": shared["recv"].sum(axis=1),
    }
    if bil is not None:
        bflow = bil["flow"]
        b_elec = own["elec_b"] - bil["elec_b"]
        b_gas = own["gas_b"] - bil["gas_b"]
        b_trade = -bil["settle_b"]                     # settle_b: + pays / − receives
        cols.update(
            {
                "bil_grid_saving": b_elec,
                "bil_gas_saving": b_gas,
                "bil_trade": b_trade,
                "bil_net_benefit": b_elec + b_gas + b_trade,
                "bil_sent_kWh": np.array([np.maximum(bflow[b], 0).sum() for b in range(n_buildings)]),
                "bil_recv_kWh": np.array([np.maximum(-bflow[b], 0).sum() for b in range(n_buildings)]),
            }
        )
    return pd.DataFrame(cols).sort_values("net_benefit", ascending=False).reset_index(drop=True)


def plot_building_benefit(own: dict, shared: dict, day: str, bil: dict | None = None) -> None:
    df = _building_benefit_table(own, shared, bil)
    has_bil = "bil_net_benefit" in df
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(13.5 if has_bil else 12, 5))

    def stacked_bar(prefix, xpos, width, hatch, label_segs):
        bpos = np.zeros(len(df))
        bneg = np.zeros(len(df))
        for col, color, lab in _BENEFIT_SEGS:
            v = df[f"{prefix}_{col}" if prefix else col].to_numpy()
            pos = np.where(v >= 0, v, 0.0)
            neg = np.where(v < 0, v, 0.0)
            ax.bar(xpos, pos, bottom=bpos, color=color, width=width, hatch=hatch,
                   edgecolor="white" if hatch else "none", linewidth=0,
                   label=lab if label_segs else None)
            ax.bar(xpos, neg, bottom=bneg, color=color, width=width, hatch=hatch,
                   edgecolor="white" if hatch else "none", linewidth=0)
            bpos += pos
            bneg += neg
        return bpos, bneg

    if has_bil:
        w = 0.38
        p_c, n_c = stacked_bar("", x - w / 2, w, None, True)
        p_b, n_b = stacked_bar("bil", x + w / 2, w, "///", False)
        ax.plot(x - w / 2, df["net_benefit"], "D", color=INK, ms=5.5, label="net benefit — central")
        ax.plot(x + w / 2, df["bil_net_benefit"], "D", color="#e0a800", ms=5.5,
                markeredgecolor="#8a6d00", label="net benefit — bilateral ADMM")
        top = np.maximum(p_c, p_b)
        bot = np.minimum(n_c, n_b)
    else:
        w = 0.0
        p_c, n_c = stacked_bar("", x, 0.62, None, True)
        ax.plot(x, df["net_benefit"], "D", color=INK, ms=6, label="net benefit")
        top, bot = p_c, n_c

    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylim(bot.min() * 1.25 - 0.35, top.max() * 1.30 + 0.35)

    for i in range(len(df)):
        s, r = df["sent_kWh"].iloc[i], df["recv_kWh"].iloc[i]
        role = "seller" if s > r + 0.1 else ("buyer" if r > s + 0.1 else "·")
        nc = df["net_benefit"].iloc[i]
        xc = x[i] - w / 2
        ax.annotate(f"{role}\n£{nc:.2f}", (xc, p_c[i] if nc >= 0 else n_c[i]),
                    textcoords="offset points", xytext=(0, 6 if nc >= 0 else -16),
                    ha="center", fontsize=6.8, color=INK)
        if has_bil:
            nb = df["bil_net_benefit"].iloc[i]
            ax.annotate(f"£{nb:.2f}", (x[i] + w / 2, p_b[i] if nb >= 0 else n_b[i]),
                        textcoords="offset points", xytext=(0, 6 if nb >= 0 else -14),
                        ha="center", fontsize=6.8, color="#8a6d00")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}\n{a}" for i, a in zip(df["LCLid"], df["assets"])],
                       rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("£ / day  (+ = better off from sharing)", color=INK, fontsize=10)
    _comm = f"community £{df['net_benefit'].sum():.2f}/day"
    if has_bil:
        _comm += f"; bilateral ADMM £{df['bil_net_benefit'].sum():.2f}/day"
    ax.set_title(
        f"{pd.Timestamp(day):%A %d %b %Y} — per-building benefit from P2P sharing "
        f"(vs own battery; FEE_MODE='{shared.get('fee_mode')}', frac {SHARE_TRADE_FEE_FRAC:g}; {_comm})"
        + ("   ·   hatched = bilateral ADMM" if has_bil else ""),
        color=INK, fontsize=10.5, fontweight="bold", loc="left", pad=10,
    )
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, axis="y", color="#ededed", lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=5 if has_bil else 4, loc="upper right")
    fig.tight_layout()
    out = PLOTS_DIR / "11_building_benefit.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")
    df.to_csv(PLOTS_DIR / "11_building_benefit.csv", index=False)

    _plotly_building_benefit(df, shared, day)


def _plotly_building_benefit(df: pd.DataFrame, shared: dict, day: str) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return
    has_bil = "bil_net_benefit" in df
    lbl = [df["LCLid"], df["assets"]]
    segs = (
        ("grid_saving", "#2e8b6e", "grid-cost saving"),
        ("gas_saving", "#c0392b", "gas-cost saving"),
        ("trade", "#3b6bb0", "trade earnings / −forfeit"),
    )
    fig = go.Figure()
    for col, color, lab in segs:
        fig.add_bar(x=lbl, y=df[col], name=lab, marker_color=color,
                    offsetgroup="central", legendgroup=lab)
    fig.add_trace(go.Scatter(x=lbl, y=df["net_benefit"], mode="markers",
                             name="net benefit — central" if has_bil else "net benefit",
                             marker=dict(color="#1a1a1a", size=9, symbol="diamond")))
    if has_bil:
        for col, color, lab in segs:
            fig.add_bar(x=lbl, y=df[f"bil_{col}"], name=lab, legendgroup=lab, showlegend=False,
                        offsetgroup="bilateral",
                        marker=dict(color=color, pattern=dict(shape="/", fgcolor="white")))
        fig.add_trace(go.Scatter(x=lbl, y=df["bil_net_benefit"], mode="markers",
                                 name="net benefit — bilateral ADMM",
                                 marker=dict(color="#e0a800", size=9, symbol="diamond",
                                             line=dict(color="#8a6d00", width=1.5))))
    _comm = f"community £{df['net_benefit'].sum():.2f}/day"
    if has_bil:
        _comm += f"; bilateral ADMM £{df['bil_net_benefit'].sum():.2f}/day"
    fig.update_layout(
        barmode="relative", template="plotly_white", height=480, hovermode="x unified",
        title=f"{pd.Timestamp(day):%A %d %b %Y} — per-building benefit from P2P sharing "
              f"(vs own battery; FEE_MODE='{shared.get('fee_mode')}'; {_comm})"
              + ("   ·   striped bars = bilateral ADMM" if has_bil else ""),
        yaxis_title="£ / day (+ = better off)", legend=dict(orientation="h", y=-0.25),
    )
    out = PLOTS_DIR / "11_building_benefit.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"saved {out}")


# ============================================================================
# 13 — per-building sharing savings: central vs the ADMM methods
# ============================================================================
def plot_sharing_savings(
    own: dict,
    shared: dict,
    day: str,
    exch_cost_b: np.ndarray | None = None,
    bil_cost_b: np.ndarray | None = None,
    exch_rel_err: float | None = None,
    bil_rel_err: float | None = None,
) -> None:
    """Per-building operating-cost saving from P2P sharing vs own-battery (no sharing).

    `shared` is the centralised market optimum. `exch_cost_b` / `bil_cost_b` are the
    per-building £/day operating costs from the exchange-ADMM (single common-pool
    price) and the bilateral consensus-ADMM (pairwise P2P), both settled at the same
    SHARE_TRADE_FEE_FRAC × grid tariff as `shared`, so the three are comparable.
    """
    df = pd.DataFrame(
        {
            "LCLid": BUILDING_IDS,
            "assets": ASSETS,
            "own_cost": own["cost_b"],
            "shared_cost": shared["cost_b"],
            "sent_kWh": shared["sent"].sum(axis=1),
            "recv_kWh": shared["recv"].sum(axis=1),
        }
    )
    df["saving"] = df["own_cost"] - df["shared_cost"]
    if exch_cost_b is not None:
        df["exch_cost"] = np.asarray(exch_cost_b, float)
        df["exch_saving"] = df["own_cost"] - df["exch_cost"]
    if bil_cost_b is not None:
        df["bil_cost"] = np.asarray(bil_cost_b, float)
        df["bil_saving"] = df["own_cost"] - df["bil_cost"]
    df["role"] = np.where(df["sent_kWh"] > df["recv_kWh"] + 0.1, "net seller",
                          np.where(df["recv_kWh"] > df["sent_kWh"] + 0.1, "net buyer", "—"))
    df = df.sort_values("saving", ascending=False).reset_index(drop=True)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [2.6, 1]})
    x = np.arange(len(df))

    axL.bar(x - 0.2, df["own_cost"], 0.4, color="#c9ced6", label="own battery (no sharing)")
    axL.bar(x + 0.2, df["shared_cost"], 0.4,
            color=[_ASSET_COLOR[a] for a in df["assets"]], label="shared — central market")
    if "exch_cost" in df:
        axL.scatter(x + 0.2, df["exch_cost"], s=44, marker="o", facecolors="none",
                    edgecolors=EXCH_COLOR, linewidths=1.6, zorder=5, label="exchange-ADMM (common pool)")
    if "bil_cost" in df:
        axL.scatter(x + 0.2, df["bil_cost"], s=46, marker="D", color=BIL_COLOR,
                    zorder=6, label="bilateral ADMM (pairwise P2P)")
    for i, (s, r) in enumerate(zip(df["saving"], df["role"])):
        top = max(df["own_cost"].iloc[i], df["shared_cost"].iloc[i])
        axL.annotate(f"{r}\n{'+' if s >= 0 else '−'}£{abs(s):.2f}", (i, top),
                     textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7.5, color=INK)
    axL.set_ylim(0, df["own_cost"].max() * 1.24)
    axL.set_xticks(x)
    axL.set_xticklabels([f"{i}\n{a}" for i, a in zip(df["LCLid"], df["assets"])],
                        rotation=45, ha="right", fontsize=7.5)
    axL.set_ylabel("operating cost (£/day)", color=INK, fontsize=10)
    notes = []
    if exch_rel_err is not None:
        notes.append(f"exchange-ADMM {exch_rel_err:.1%}")
    if bil_rel_err is not None:
        notes.append(f"bilateral {bil_rel_err:.1%}")
    rms_note = f"; RMS vs central: {', '.join(notes)}" if notes else ""
    axL.set_title(f"{pd.Timestamp(day):%A %d %b %Y} — per-building operating cost: sharing vs no sharing "
                  f"(FEE_MODE='{shared.get('fee_mode')}', frac {SHARE_TRADE_FEE_FRAC:g}{rms_note})",
                  color=INK, fontsize=11, fontweight="bold", loc="left", pad=22)
    axL.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)

    # right: saving aggregated by asset class, one bar per method
    order = [a for a in ("battery+pv", "battery", "pv", "none") if a in df["assets"].values]
    method_cols = ["saving"] + [c for c in ("exch_saving", "bil_saving") if c in df]
    gg = df.groupby("assets")[method_cols].sum().reindex(order)
    y = np.arange(len(order))
    nm = len(method_cols)
    bh = 0.8 / nm
    meth_lab = {"saving": "central market", "exch_saving": "exchange-ADMM", "bil_saving": "bilateral ADMM"}
    meth_col = {"saving": None, "exch_saving": EXCH_COLOR, "bil_saving": BIL_COLOR}
    for j, c in enumerate(method_cols):
        off = (nm - 1) / 2 - j
        color = [_ASSET_COLOR[a] for a in order] if c == "saving" else meth_col[c]
        axR.barh(y + off * bh, gg[c].to_numpy(), bh, color=color,
                 hatch=None if c == "saving" else "///", edgecolor="white", label=meth_lab[c])
    for i, tot in enumerate(gg["saving"].to_numpy()):
        axR.annotate(f"£{tot:+.2f}", (tot, y[i] + ((nm - 1) / 2) * bh),
                     textcoords="offset points", xytext=(6 if tot >= 0 else -6, 0),
                     ha="left" if tot >= 0 else "right", va="center", fontsize=7.5, color=INK)
    axR.set_yticks(y)
    axR.set_yticklabels(order, fontsize=9)
    axR.axvline(0, color=MUTED, lw=0.8)
    allv = gg.to_numpy().ravel()
    _lo, _hi = min(0.0, float(allv.min())), max(0.01, float(allv.max()))
    axR.set_xlim(_lo - 0.1 * _hi, _hi * 2.15)
    axR.set_xlabel("total saving from sharing (£/day)", fontsize=9)
    axR.set_title(f"By asset class  (community £{df['saving'].sum():+.2f}/day)",
                  color=INK, fontsize=10.5, fontweight="bold", loc="left", pad=10)
    if nm > 1:
        axR.legend(frameon=False, fontsize=7.5, loc="lower right")

    for ax in (axL, axR):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(True, axis="y" if ax is axL else "x", color="#ededed", lw=0.7)
        ax.set_axisbelow(True)
    fig.tight_layout()
    out = PLOTS_DIR / "13_sharing_savings.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")
    df.to_csv(PLOTS_DIR / "13_sharing_savings.csv", index=False)
    _plotly_sharing_savings(df, gg, order, shared, day)


def _plotly_sharing_savings(df: pd.DataFrame, gg: pd.DataFrame, order: list, shared: dict, day: str) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return
    lbl = [df["LCLid"], df["assets"]]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.68, 0.32], horizontal_spacing=0.09,
                        subplot_titles=("Operating cost per building: no sharing vs sharing",
                                        "Saving by asset class (£/day)"))
    fig.add_bar(x=lbl, y=df["own_cost"], name="own battery (no sharing)", marker_color="#c9ced6",
                row=1, col=1)
    fig.add_bar(x=lbl, y=df["shared_cost"], name="shared — central market",
                marker_color=[_ASSET_COLOR[a] for a in df["assets"]], row=1, col=1)
    if "exch_cost" in df:
        fig.add_trace(go.Scatter(x=lbl, y=df["exch_cost"], mode="markers", name="exchange-ADMM (common pool)",
                                 marker=dict(color=EXCH_COLOR, size=10, symbol="circle-open",
                                             line=dict(width=2))), row=1, col=1)
    if "bil_cost" in df:
        fig.add_trace(go.Scatter(x=lbl, y=df["bil_cost"], mode="markers", name="bilateral ADMM (pairwise P2P)",
                                 marker=dict(color=BIL_COLOR, size=9, symbol="diamond")), row=1, col=1)
    for c, lab, col in (("saving", "central market", None),
                        ("exch_saving", "exchange-ADMM", EXCH_COLOR),
                        ("bil_saving", "bilateral ADMM", BIL_COLOR)):
        if c not in gg:
            continue
        fig.add_bar(x=order, y=gg[c].to_numpy(), name=lab, showlegend=(c != "saving"),
                    marker_color=([_ASSET_COLOR[a] for a in order] if col is None else col),
                    row=1, col=2)
    fig.update_layout(template="plotly_white", height=470, barmode="group",
                      title=f"{pd.Timestamp(day):%A %d %b %Y} — per-building saving from P2P sharing "
                            f"(FEE_MODE='{shared.get('fee_mode')}'; community £{df['saving'].sum():+.2f}/day)",
                      legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(title_text="£/day", row=1, col=1)
    out = PLOTS_DIR / "13_sharing_savings.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"saved {out}")


# ============================================================================
# 15-17 — thermal comfort (temperature deviation) + peak grid import
# ============================================================================
# each `series` here is  {label: {"T_in": (building, t), "grid": (building, t)}}
_DEV_COLOR = {
    "no_battery": "#8a8f98",
    "own_battery": "#d98a29",
    "shared": "#2e8b6e",
    "central market": "#2e8b6e",
    "exchange-ADMM": EXCH_COLOR,
    "bilateral ADMM": BIL_COLOR,
}


def _tin_from_subs(subs) -> np.ndarray:
    """(building, t) indoor temperature from a list of solved ADMM subproblems."""
    return np.array([[pyo.value(m.T_in[t]) for t in range(time_horizon)] for m in subs])


def _grid_from_subs(subs) -> np.ndarray:
    """(building, t) grid import from a list of solved ADMM subproblems."""
    return np.array([[pyo.value(m.p_el[t]) for t in range(time_horizon)] for m in subs])


def _dev_stats(tin: np.ndarray):
    """tin: (building, t). -> (rms_b, mean_signed_b, community_rms, comfort_penalty).

    rms_b / mean_signed_b are per building over the day (°C; mean_signed < 0 = the
    building is left below setpoint). comfort_penalty is the objective term
    alpha * Σ (T_in − T_set)²  (same quantity solve_and_extract calls 'comfort')."""
    d = np.asarray(tin, float) - T_SET[None, :]
    return (
        np.sqrt((d ** 2).mean(axis=1)),
        d.mean(axis=1),
        float(np.sqrt((d ** 2).mean())),
        float(alpha * (d ** 2).sum()),
    )


def _peak_stats(grid: np.ndarray):
    """grid: (building, t). -> (peak_b, community_peak_kW, peak_hour, community_kWh)."""
    g = np.asarray(grid, float)
    tot = g.sum(axis=0)
    return g.max(axis=1), float(tot.max()), int(tot.argmax()), float(tot.sum() * dt)


def _peak_hour_b(grid: np.ndarray) -> np.ndarray:
    """(building,) hour at which each building's grid import peaks (first if tied)."""
    return np.asarray(grid, float).argmax(axis=1)


def _comfort_peak_csv(series: dict, out_stem: str) -> None:
    rows = []
    for lab, d in series.items():
        rms_b, mean_b, comm_rms, pen = _dev_stats(d["T_in"])
        peak_b, comm_peak, peak_hr, comm_kwh = _peak_stats(d["grid"])
        hr_b = _peak_hour_b(d["grid"])
        for i, bid in enumerate(BUILDING_IDS):
            rows.append({"series": lab, "LCLid": bid, "assets": ASSETS[i],
                         "rms_dev_C": rms_b[i], "mean_signed_dev_C": mean_b[i],
                         "peak_grid_import_kW": peak_b[i], "peak_grid_hour": int(hr_b[i]),
                         "community_rms_C": comm_rms, "comfort_penalty": pen,
                         "community_peak_kW": comm_peak, "community_peak_hour": peak_hr,
                         "community_import_kWh": comm_kwh})
    pd.DataFrame(rows).to_csv(PLOTS_DIR / f"{out_stem}.csv", index=False)


def _plot_comfort_peak_bars(series: dict, day: str, out_stem: str, title: str) -> None:
    """Three panels, grouped per building by scenario/method:
    left   — RMS temperature deviation from setpoint (◆ = mean signed dev),
    middle — peak grid import (kW),
    right  — hour of day at which each building's grid import peaks."""
    labels = list(series)
    n = len(labels)
    x = np.arange(n_buildings)
    bw = 0.8 / n
    fig, (axT, axP, axH) = plt.subplots(1, 3, figsize=(19.5, 5))

    hr_mat = np.array([_peak_hour_b(series[lab]["grid"]) for lab in labels])  # (series, building)

    for j, lab in enumerate(labels):
        rms_b, mean_b, comm_rms, _pen = _dev_stats(series[lab]["T_in"])
        peak_b, comm_peak, peak_hr, _kwh = _peak_stats(series[lab]["grid"])
        off = (j - (n - 1) / 2) * bw
        col = _DEV_COLOR.get(lab, "#777")
        axT.bar(x + off, rms_b, bw, color=col, label=f"{lab}  (RMS {comm_rms:.2f}°C)")
        axT.scatter(x + off, mean_b, s=15, marker="D", zorder=5, linewidths=0.4,
                    edgecolors="white",
                    c=["#3b6bb0" if v < 0 else "#c0392b" for v in mean_b])
        axP.bar(x + off, peak_b, bw, color=col,
                label=f"{lab}  (community peak {comm_peak:.1f} kW @ {peak_hr:02d}h)")
        axH.scatter(x + off, hr_mat[j], s=44, color=col, edgecolors="white", linewidths=0.5,
                    zorder=5, label=lab)

    # connect each building's peak hour across the series so a shift is visible
    if n > 1:
        offs = (np.arange(n) - (n - 1) / 2) * bw
        for b in range(n_buildings):
            axH.plot(b + offs, hr_mat[:, b], color=MUTED, lw=0.6, zorder=1)

    axT.axhline(0, color=MUTED, lw=0.8)
    axT.set_ylabel("RMS deviation from setpoint (°C)   ·   ◆ mean signed dev "
                   "(blue = under-heated)", color=INK, fontsize=9)
    axT.set_title("Thermal comfort", color=INK, fontsize=10.5, fontweight="bold", loc="left", pad=8)
    axP.set_ylabel("peak grid import over the day (kW)", color=INK, fontsize=9.5)
    axP.set_title("Peak grid import", color=INK, fontsize=10.5, fontweight="bold", loc="left", pad=8)

    axH.axhspan(17, 22, color=PRICE_HIGH_COLOR, alpha=0.13, lw=0)
    axH.set_ylim(-0.6, time_horizon)
    axH.set_yticks(range(0, time_horizon + 1, 3))
    axH.set_ylabel("hour of the building's peak grid import\n(shaded band = 17–22h high tariff)",
                   color=INK, fontsize=9)
    axH.set_title("Peak-import time", color=INK, fontsize=10.5, fontweight="bold", loc="left", pad=8)

    for ax in (axT, axP, axH):
        ax.set_xticks(x)
        ax.set_xticklabels([f"{i}\n{a}" for i, a in zip(BUILDING_IDS, ASSETS)],
                           rotation=45, ha="right", fontsize=7.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(True, axis="y", color="#ededed", lw=0.7)
        ax.set_axisbelow(True)
    axT.legend(frameon=False, fontsize=7.5, ncol=1, loc="upper right")
    axP.legend(frameon=False, fontsize=7.5, ncol=1, loc="upper right")
    axH.legend(frameon=False, fontsize=7.5, ncol=1, loc="upper right")

    fig.suptitle(f"{pd.Timestamp(day):%A %d %b %Y} — {title}",
                 color=INK, fontsize=11.5, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = PLOTS_DIR / f"{out_stem}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")
    _comfort_peak_csv(series, out_stem)
    _plotly_comfort_peak_bars(series, day, out_stem, title)


def plot_comfort_peak_total(series: dict, day: str) -> None:
    """Two panels, one bar per approach (all scenarios + methods side by side):
    left — community RMS temperature deviation; right — community peak grid import."""
    labels = list(series)
    x = np.arange(len(labels))
    dev = {lab: _dev_stats(series[lab]["T_in"]) for lab in labels}
    pk = {lab: _peak_stats(series[lab]["grid"]) for lab in labels}
    rms = [dev[lab][2] for lab in labels]
    pens = [dev[lab][3] for lab in labels]
    peaks = [pk[lab][1] for lab in labels]
    hrs = [pk[lab][2] for lab in labels]
    cols = [_DEV_COLOR.get(lab, "#777") for lab in labels]

    fig, (axT, axP) = plt.subplots(1, 2, figsize=(4.0 + 1.5 * len(labels), 4.7))
    axT.bar(x, rms, 0.6, color=cols)
    for i, (v, p) in enumerate(zip(rms, pens)):
        axT.annotate(f"{v:.2f}°C\npenalty {p:.0f}", (i, v), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=8, color=INK)
    axT.set_ylabel("community RMS deviation from setpoint (°C)", color=INK, fontsize=9.5)
    axT.set_ylim(0, max(rms) * 1.30)
    axT.set_title("Thermal comfort", color=INK, fontsize=10.5, fontweight="bold", loc="left", pad=8)

    axP.bar(x, peaks, 0.6, color=cols)
    for i, (v, h) in enumerate(zip(peaks, hrs)):
        axP.annotate(f"{v:.1f} kW\n@ {h:02d}h", (i, v), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=8, color=INK)
    axP.set_ylabel("community peak grid import (kW)", color=INK, fontsize=9.5)
    axP.set_ylim(0, max(peaks) * 1.22)
    axP.set_title("Peak grid import", color=INK, fontsize=10.5, fontweight="bold", loc="left", pad=8)

    for ax in (axT, axP):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(True, axis="y", color="#ededed", lw=0.7)
        ax.set_axisbelow(True)

    fig.suptitle(f"{pd.Timestamp(day):%A %d %b %Y} — community thermal comfort & peak grid "
                 f"import, all approaches", color=INK, fontsize=11.5, fontweight="bold",
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = PLOTS_DIR / "17_comfort_peak_total.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")
    pd.DataFrame({"approach": labels, "community_rms_dev_C": rms, "comfort_penalty": pens,
                  "community_peak_kW": peaks, "community_peak_hour": hrs}).to_csv(
        PLOTS_DIR / "17_comfort_peak_total.csv", index=False)
    _plotly_comfort_peak_total(series, day)


def _plotly_comfort_peak_bars(series: dict, day: str, out_stem: str, title: str) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return
    lbl = [BUILDING_IDS, ASSETS]
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.055,
                        subplot_titles=("RMS temperature deviation from setpoint (°C)",
                                        "Peak grid import (kW)",
                                        "Hour of the building's peak grid import"))
    for lab in series:
        rms_b, _m, comm_rms, _p = _dev_stats(series[lab]["T_in"])
        peak_b, comm_peak, hr, _k = _peak_stats(series[lab]["grid"])
        hr_b = _peak_hour_b(series[lab]["grid"])
        col = _DEV_COLOR.get(lab, "#777")
        fig.add_bar(x=lbl, y=rms_b, name=f"{lab} (RMS {comm_rms:.2f}°C)", marker_color=col,
                    legendgroup=lab, row=1, col=1)
        fig.add_bar(x=lbl, y=peak_b, name=f"{lab} (peak {comm_peak:.1f} kW @ {hr:02d}h)",
                    marker_color=col, legendgroup=lab, showlegend=False, row=1, col=2)
        fig.add_trace(go.Scatter(x=lbl, y=hr_b, mode="markers", name=lab, legendgroup=lab,
                                 showlegend=False,
                                 marker=dict(color=col, size=11, line=dict(color="white", width=1))),
                      row=1, col=3)
    fig.add_hrect(y0=17, y1=22, fillcolor=PRICE_HIGH_COLOR, opacity=0.12, line_width=0, row=1, col=3)
    fig.update_yaxes(title_text="hour of day", range=[-0.6, time_horizon], dtick=3, row=1, col=3)
    fig.update_layout(template="plotly_white", height=470, barmode="group",
                      title=f"{pd.Timestamp(day):%A %d %b %Y} — {title}",
                      legend=dict(orientation="h", y=-0.25))
    out = PLOTS_DIR / f"{out_stem}.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"saved {out}")


def _plotly_comfort_peak_total(series: dict, day: str) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return
    labels = list(series)
    rms = [_dev_stats(series[lab]["T_in"])[2] for lab in labels]
    peaks = [_peak_stats(series[lab]["grid"])[1] for lab in labels]
    cols = [_DEV_COLOR.get(lab, "#777") for lab in labels]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Community RMS temperature deviation (°C)",
                                        "Community peak grid import (kW)"))
    fig.add_bar(x=labels, y=rms, marker_color=cols, text=[f"{v:.2f}" for v in rms],
                textposition="outside", showlegend=False, row=1, col=1)
    fig.add_bar(x=labels, y=peaks, marker_color=cols, text=[f"{v:.1f}" for v in peaks],
                textposition="outside", showlegend=False, row=1, col=2)
    fig.update_layout(template="plotly_white", height=440,
                      title=f"{pd.Timestamp(day):%A %d %b %Y} — community comfort & peak import, "
                            f"all approaches")
    out = PLOTS_DIR / "17_comfort_peak_total.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"saved {out}")


# ============================================================================
# 18 — community-average grid import over the day, per scenario / algorithm
# ============================================================================
def plot_consumption_timeseries(grids: dict, day: str) -> None:
    """grids: {label: grid (building, t)}. One step line per scenario / algorithm of
    the community-average grid import (kW, mean over the buildings) across the day."""
    edges = np.arange(time_horizon + 1)

    def step(y):
        return edges, np.concatenate([y, y[-1:]])

    fig, ax = plt.subplots(figsize=(12, 5))
    lo0, lo1 = _hi_window()
    if lo0 is not None:
        ax.axvspan(lo0, lo1, color=PRICE_HIGH_COLOR, alpha=0.13, lw=0)

    yb = LOAD.mean(axis=0)
    ax.step(*step(yb), where="post", color=MUTED, lw=1.3, ls="--",
            label=f"base load only, no heating  (mean {yb.mean():.2f} kW)")

    for lab, g in grids.items():
        y = np.asarray(g, float).mean(axis=0)
        ax.step(*step(y), where="post", color=_DEV_COLOR.get(lab, "#777"), lw=2.1,
                label=f"{lab}  (mean {y.mean():.2f} kW, peak {y.max():.2f} kW @ {int(y.argmax()):02d}h)")

    ax.set_xlim(0, time_horizon)
    ax.set_xticks(range(0, time_horizon + 1, 3))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("hour of day", color=INK, fontsize=10)
    ax.set_ylabel("average grid import per building (kW)", color=INK, fontsize=10)
    ax.set_title(f"{pd.Timestamp(day):%A %d %b %Y} — community-average electricity drawn from the "
                 f"grid  (shaded band = 17–22h high tariff)",
                 color=INK, fontsize=11, fontweight="bold", loc="left", pad=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, axis="y", color="#ededed", lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    out = PLOTS_DIR / "18_consumption_timeseries.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")
    pd.DataFrame({"hour": list(range(time_horizon)), "base_load_kW": yb,
                  **{lab: np.asarray(g, float).mean(axis=0) for lab, g in grids.items()}}).to_csv(
        PLOTS_DIR / "18_consumption_timeseries.csv", index=False)
    _plotly_consumption_timeseries(grids, day)


def _plotly_consumption_timeseries(grids: dict, day: str) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return
    hrs = list(range(time_horizon)) + [time_horizon]

    def sx(y):
        return list(y) + [y[-1]]

    fig = go.Figure()
    yb = LOAD.mean(axis=0)
    fig.add_trace(go.Scatter(x=hrs, y=sx(yb), name="base load only, no heating", line_shape="hv",
                             line=dict(color=MUTED, width=1.4, dash="dash")))
    for lab, g in grids.items():
        y = np.asarray(g, float).mean(axis=0)
        fig.add_trace(go.Scatter(x=hrs, y=sx(y), name=f"{lab} (peak {y.max():.2f} kW)",
                                 line_shape="hv",
                                 line=dict(color=_DEV_COLOR.get(lab, "#777"), width=2.3)))
    lo0, lo1 = _hi_window()
    if lo0 is not None:
        fig.add_vrect(x0=lo0, x1=lo1, fillcolor=PRICE_HIGH_COLOR, opacity=0.12, line_width=0)
    fig.update_layout(template="plotly_white", height=460, hovermode="x unified",
                      title=f"{pd.Timestamp(day):%A %d %b %Y} — community-average grid import",
                      xaxis_title="hour of day",
                      yaxis_title="average grid import per building (kW)",
                      legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(rangemode="tozero")
    out = PLOTS_DIR / "18_consumption_timeseries.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"saved {out}")


# ============================================================================
# 19 — cumulative community electricity cost (plots/05 style) for every approach
# ============================================================================
def plot_cost_comparison(grids: dict, day: str) -> None:
    """Recreation of plots/05_sharing_showcase in the plots/05 two-panel style, but
    from the actual optimisation / ADMM dispatch instead of proxy water-filling.
    grids: ordered {label: grid (building, t)} — expected keys include no_battery,
    own_battery (the non-sharing references), shared (central market), exchange-ADMM,
    bilateral ADMM. Panel 1: demand shapes. Panel 2: cumulative community grid-
    electricity cost, one line per approach."""
    edges = np.arange(time_horizon + 1)

    def step(y):
        return edges, np.concatenate([y, y[-1:]])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.8), sharex=True)

    # --- Panel 1: base electrical demand ---
    _shade_tariff(ax1)
    for b in range(n_buildings):
        ax1.step(*step(LOAD[b]), where="post", color=CONSUMPTION_COLOR, lw=1.0, alpha=0.4)
    ax1.step(*step(LOAD.sum(axis=0)), where="post", color=INK, lw=1.9)
    ax1.plot([], [], color=CONSUMPTION_COLOR, alpha=0.6, label=f"Individual buildings (n={n_buildings})")
    ax1.plot([], [], color=INK, lw=1.9, label="Community total (base load)")
    ax1.fill_between([], [], color=PRICE_HIGH_COLOR, alpha=0.2, label="High tariff (67 p/kWh)")
    ax1.fill_between([], [], color=PRICE_LOW_COLOR, alpha=0.2, label="Low tariff (4 p/kWh)")
    ax1.set_ylabel("Base electrical demand (kW)", color=INK, fontsize=10)
    ax1.set_ylim(0, LOAD.sum(axis=0).max() * 1.28)
    ax1.set_title(f"{pd.Timestamp(day):%A %d %b %Y} — {n_buildings} buildings: demand and the "
                  f"cost of every sharing scheme", color=INK, fontsize=12, fontweight="bold",
                  loc="left", pad=10)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)

    # --- Panel 2: cumulative community grid-electricity cost ---
    _shade_tariff(ax2)
    cum = {lab: np.cumsum(PRICE * np.asarray(g, float).sum(axis=0) * dt) for lab, g in grids.items()}
    for lab, cc in cum.items():
        ls = "--" if lab == "no_battery" else "-"
        ax2.step(*step(cc), where="post", color=_DEV_COLOR.get(lab, "#777"),
                 lw=2.4 if lab in ("shared", "own_battery") else 2.0, ls=ls,
                 label=f"{lab}  (£{cc[-1]:.2f})")
    if "own_battery" in cum and "shared" in cum:
        _, yo = step(cum["own_battery"])
        _, ys = step(cum["shared"])
        ax2.fill_between(edges, ys, yo, step="post", color=SHARED_COLOR, alpha=0.15)
        save = cum["own_battery"][-1] - cum["shared"][-1]
        parts = [f"central sharing saves £{save:.2f}/day vs own-battery"]
        for k in ("exchange-ADMM", "bilateral ADMM"):
            if k in cum:
                parts.append(f"{k} £{cum['own_battery'][-1] - cum[k][-1]:.2f}")
        ax2.annotate("  ·  ".join(parts),
                     xy=(time_horizon, (cum["own_battery"][-1] + cum["shared"][-1]) / 2),
                     xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                     fontsize=8.5, color=SHARED_COLOR, fontweight="bold")
    ax2.set_ylabel("Cumulative community grid-electricity cost (£)", color=INK, fontsize=10)
    ax2.set_xlabel("Hour of day", color=INK, fontsize=10)
    ax2.set_ylim(bottom=0)
    if "no_battery" in cum:
        ax2.set_ylim(0, cum["no_battery"][-1] * 1.12)
    ax2.set_title("Same storage, different coupling: own-battery vs pooled (central) vs the "
                  "decentralised ADMM schemes", color=INK, fontsize=10.5, fontweight="bold",
                  loc="left", pad=10)
    ax2.legend(frameon=False, fontsize=9, loc="upper left")

    for ax in (ax1, ax2):
        _style(ax)
    fig.tight_layout()
    out = PLOTS_DIR / "19_cost_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")
    pd.DataFrame({"hour": list(range(time_horizon)),
                  **{f"cum_cost_{lab}": cc for lab, cc in cum.items()}}).to_csv(
        PLOTS_DIR / "19_cost_comparison.csv", index=False)
    _plotly_cost_comparison(grids, cum, day)


def _plotly_cost_comparison(grids: dict, cum: dict, day: str) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return
    hrs = list(range(time_horizon)) + [time_horizon]

    def sx(y):
        return list(y) + [y[-1]]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=("Base electrical demand (kW)",
                                        "Cumulative community grid-electricity cost (£)"))
    for b in range(n_buildings):
        fig.add_trace(go.Scatter(x=hrs, y=sx(LOAD[b]), line=dict(color="rgba(59,107,176,.3)", width=1,
                                 shape="hv"), showlegend=(b == 0), name="individual buildings",
                                 legendgroup="ind", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hrs, y=sx(LOAD.sum(axis=0)), line=dict(color="black", width=2, shape="hv"),
                             name="community total"), row=1, col=1)
    for lab, cc in cum.items():
        dash = "dash" if lab == "no_battery" else "solid"
        fig.add_trace(go.Scatter(x=hrs, y=sx(cc), name=f"{lab} (£{cc[-1]:.2f})", line_shape="hv",
                                 line=dict(color=_DEV_COLOR.get(lab, "#777"), width=2.4, dash=dash)),
                      row=2, col=1)
    lo0, lo1 = _hi_window()
    if lo0 is not None:
        for r in (1, 2):
            fig.add_vrect(x0=lo0, x1=lo1, fillcolor=PRICE_HIGH_COLOR, opacity=0.1, line_width=0, row=r, col=1)
    fig.update_xaxes(title_text="hour of day", dtick=3, row=2, col=1)
    fig.update_layout(template="plotly_white", height=780, hovermode="x unified",
                      title=f"{pd.Timestamp(day):%A %d %b %Y} — cost of every sharing scheme",
                      legend=dict(orientation="h", y=-0.08))
    out = PLOTS_DIR / "19_cost_comparison.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"saved {out}")


# ============================================================================
# ADMM RUNNERS  (each re-settled at SHARE_TRADE_FEE_FRAC × tariff for comparability)
# ============================================================================
_N_PAIRS = n_buildings * (n_buildings - 1) // 2
_N_SHARE_VARS_CENTRAL = n_buildings * n_buildings * time_horizon
# optional escape hatches for the scaling runs (default: attempt everything)
_SKIP_CENTRAL_ABOVE_N = int(os.environ.get("SHOWCASE_SKIP_CENTRAL_ABOVE_N", "0") or 0)
_SKIP_BILATERAL_ABOVE_N = int(os.environ.get("SHOWCASE_SKIP_BILATERAL_ABOVE_N", "0") or 0)


def _comp_row(component, status, wall, **kw):
    row = {
        "n_buildings": n_buildings, "time_horizon": time_horizon,
        "component": component, "status": status,
        "wall_seconds": round(wall, 2) if wall is not None else "",
        "iterations": "", "max_iterations": "", "converged": "",
        "final_primal_resid": "", "final_dual_resid": "",
        "community_op_cost_gbp": "", "community_peak_kW": "", "rms_cost_vs_central": "",
        "n_trade_pairs": _N_PAIRS, "n_share_vars_central": _N_SHARE_VARS_CENTRAL,
        "note": "",
    }
    row.update({k: v for k, v in kw.items() if v is not None})
    return row


def _try_central(label: str, **kw):
    """(result_dict | None, comp_row). Wraps build_model + solve_and_extract so an
    OOM / time-limit / solver failure at large N is caught and recorded."""
    t0 = time.perf_counter()
    try:
        r = solve_and_extract(build_model(**kw), label)
        wall = time.perf_counter() - t0
        peak = float(np.asarray(r["grid"], float).sum(axis=0).max())
        return r, _comp_row(f"central:{label}", "ok", wall,
                            community_op_cost_gbp=round(r["op_cost"], 2),
                            community_peak_kW=round(peak, 2))
    except Exception as e:  # noqa: BLE001 - MemoryError / RuntimeError / solver error
        wall = time.perf_counter() - t0
        print(f"  ✗ central '{label}' unavailable — {type(e).__name__}: {e}")
        return None, _comp_row(f"central:{label}", "failed", wall,
                               note=f"{type(e).__name__}: {e}")


def run_exchange_admm(sh):
    """Run admm_energy_sharing (exchange-ADMM). Returns (cost_b, rel, bundle, comp_row).
    On failure returns (None, None, None, comp_row). `sh` may be None (large N) — the
    RMS-vs-central column is then blank."""
    t0 = time.perf_counter()
    try:
        pex, price, hist, subs = EXC.run_admm()
        S = EXC.summarise(pex, price, subs)
    except Exception as e:  # noqa: BLE001
        wall = time.perf_counter() - t0
        print(f"exchange-ADMM failed ({type(e).__name__}: {e})")
        return None, None, None, _comp_row("exchange-ADMM", "failed", wall,
                                           note=f"{type(e).__name__}: {e}")
    wall = time.perf_counter() - t0
    frac = SHARE_TRADE_FEE_FRAC
    settle_b = -np.array([(frac * PRICE * pex[b] * dt).sum() for b in range(n_buildings)])
    cost_b = S["elec_b"] + S["gas_b"] + settle_b
    rel = None
    if sh is not None:
        rel = float(np.sqrt(np.mean(
            ((cost_b - sh["cost_b"]) / np.maximum(np.abs(sh["cost_b"]), 1e-3)) ** 2)))
    n_it = len(hist)
    fp, fd = float(hist[-1, 1]), float(hist[-1, 2])
    conv = bool(fp < EXC.EPS_PRIMAL and fd < EXC.EPS_DUAL)
    peak = float(_grid_from_subs(subs).sum(axis=0).max())
    print(f"exchange-ADMM: {n_it} iters, community £{S['admm_op']:.2f}/day, {S['traded']:.1f} kWh traded"
          + (f", RMS vs central {rel:.2%}" if rel is not None else ""))
    return cost_b, rel, (pex, price, hist, subs, S), _comp_row(
        "exchange-ADMM", "ok", wall, iterations=n_it, max_iterations=EXC.MAX_ITERS,
        converged=conv, final_primal_resid=round(fp, 5), final_dual_resid=round(fd, 5),
        community_op_cost_gbp=round(S["admm_op"], 2), community_peak_kW=round(peak, 2),
        rms_cost_vs_central=(round(rel, 4) if rel is not None else None))


def run_bilateral_admm():
    """Run admm_bilateral_p2p. Returns (cost_b, rel, bundle, comp_row); (None, None,
    None, comp_row) on failure (OOM at large N is expected)."""
    t0 = time.perf_counter()
    try:
        subs, q_all, lam_all, Z, hist = BIL.run_admm()
        S = BIL.summarise(subs, lam_all, Z, SHOWCASE_DAY)
    except Exception as e:  # noqa: BLE001
        wall = time.perf_counter() - t0
        print(f"bilateral ADMM failed ({type(e).__name__}: {e})")
        return None, None, None, _comp_row("bilateral ADMM", "failed", wall,
                                           note=f"{type(e).__name__}: {e}")
    wall = time.perf_counter() - t0
    n_it = len(hist)
    fp, fd = float(hist[-1, 3]), float(hist[-1, 4])
    conv = bool(fp < BIL.EPS_PRIMAL and fd < BIL.EPS_DUAL)
    peak = float(_grid_from_subs(subs).sum(axis=0).max())
    rel = S["rel_err"] if np.isfinite(S["rel_err"]) else None
    print(f"bilateral ADMM: {n_it} iters, community £{S['admm_op']:.2f}/day, {S['traded']:.1f} kWh traded"
          + (f", RMS vs central {rel:.2%}" if rel is not None else ""))
    return S["admm_cost_b"], rel, (subs, q_all, lam_all, Z, hist, S), _comp_row(
        "bilateral ADMM", "ok", wall, iterations=n_it, max_iterations=BIL.MAX_ITERS,
        converged=conv, final_primal_resid=round(fp, 5), final_dual_resid=round(fd, 5),
        community_op_cost_gbp=round(S["admm_op"], 2), community_peak_kW=round(peak, 2),
        rms_cost_vs_central=(round(rel, 4) if rel is not None else None))


# ============================================================================
def main(with_admm: bool = True, admm_plots: bool = False) -> None:
    run_t0 = time.perf_counter()
    print(
        f"Showcase day: {SHOWCASE_DAY} | {n_buildings} buildings | {time_horizon} h | "
        f"total storage {CAP.sum():.1f} kWh\n"
    )
    if n_buildings > 40:
        print(f"! large community (N={n_buildings}): the central MIQCP has N²·T = "
              f"{_N_SHARE_VARS_CENTRAL:,} sharing variables and bilateral ADMM has "
              f"{_N_PAIRS:,} trade pairs — both may run out of memory or hit the solve "
              f"time limit. Failures are caught; exchange-ADMM scales.\n")
    _require_solver()

    comp = []
    if _SKIP_CENTRAL_ABOVE_N and n_buildings > _SKIP_CENTRAL_ABOVE_N:
        print(f"central solves skipped (N={n_buildings} > SHOWCASE_SKIP_CENTRAL_ABOVE_N="
              f"{_SKIP_CENTRAL_ABOVE_N})")
        nb = ob = sh = None
        for _l in ("no_battery", "own_battery", "shared"):
            comp.append(_comp_row(f"central:{_l}", "skipped", None,
                                  note=f"N > SHOWCASE_SKIP_CENTRAL_ABOVE_N={_SKIP_CENTRAL_ABOVE_N}"))
    else:
        nb, r = _try_central("no_battery", use_battery=False, sharing=False); comp.append(r)
        ob, r = _try_central("own_battery", use_battery=True, sharing=False); comp.append(r)
        sh, r = _try_central("shared", use_battery=True, sharing=True); comp.append(r)
    central_ok = nb is not None and ob is not None and sh is not None

    if central_ok:
        b_ben = nb["op_cost"] - ob["op_cost"]
        s_ben = ob["op_cost"] - sh["op_cost"]
        print(f"\nno_battery £{nb['op_cost']:.2f}  own_battery £{ob['op_cost']:.2f}  "
              f"shared £{sh['op_cost']:.2f}   (battery benefit £{b_ben:.2f}/day, "
              f"sharing benefit £{s_ben:.2f}/day)\n")
    else:
        print("\n! central solves incomplete — the central sharing-cost figures "
              "(06/07/08/11/13) will be skipped for this N\n")

    exch_cost_b = bil_cost_b = exch_rel = bil_rel = None
    exch_bundle = bil_bundle = bil_S = None
    if with_admm:
        print("--- exchange-ADMM (pooled sharing, admm_energy_sharing) ---")
        exch_cost_b, exch_rel, exch_bundle, r = run_exchange_admm(sh); comp.append(r)
        if _SKIP_BILATERAL_ABOVE_N and n_buildings > _SKIP_BILATERAL_ABOVE_N:
            print(f"\nbilateral ADMM skipped (N={n_buildings} > SHOWCASE_SKIP_BILATERAL_ABOVE_N="
                  f"{_SKIP_BILATERAL_ABOVE_N})")
            comp.append(_comp_row("bilateral ADMM", "skipped", None,
                                  note=f"N > SHOWCASE_SKIP_BILATERAL_ABOVE_N={_SKIP_BILATERAL_ABOVE_N}"))
        else:
            print("\n--- bilateral consensus-ADMM (pairwise P2P, admm_bilateral_p2p) ---")
            bil_cost_b, bil_rel, bil_bundle, r = run_bilateral_admm(); comp.append(r)
            bil_S = bil_bundle[-1] if bil_bundle is not None else None

    # ---- central-only figures (need all three central solves) ----
    if central_ok:
        plot_showcase(nb, ob, sh, SHOWCASE_DAY)
        plot_building_schedules(sh, SHOWCASE_DAY, "shared")
        plot_building_schedules(ob, SHOWCASE_DAY, "own_battery")
        plot_energy_share(sh, SHOWCASE_DAY)
        plot_building_benefit(ob, sh, SHOWCASE_DAY, bil=bil_S)
        try:
            plotly_showcase(nb, ob, sh, SHOWCASE_DAY)
            plotly_building_schedules(sh, SHOWCASE_DAY, "shared")
            plotly_building_schedules(ob, SHOWCASE_DAY, "own_battery")
            plotly_energy_share(sh, SHOWCASE_DAY)
        except ImportError:
            print("plotly not installed — skipped .html plots (pip install plotly)")
        plot_sharing_savings(
            ob, sh, SHOWCASE_DAY,
            exch_cost_b=exch_cost_b, bil_cost_b=bil_cost_b,
            exch_rel_err=exch_rel, bil_rel_err=bil_rel,
        )

    # ---- comfort / peak / consumption — from whatever solved ----
    scen = {}
    for lab, res in (("no_battery", nb), ("own_battery", ob), ("shared", sh)):
        if res is not None:
            scen[lab] = {"T_in": res["T_in"], "grid": res["grid"]}
    if scen:
        _plot_comfort_peak_bars(
            scen, SHOWCASE_DAY, "15_comfort_peak_scenarios",
            "by scenario: per-building temperature deviation from setpoint & peak grid import",
        )

    method = {}
    if sh is not None:
        method["central market"] = {"T_in": sh["T_in"], "grid": sh["grid"]}
    if exch_bundle is not None:
        method["exchange-ADMM"] = {"T_in": _tin_from_subs(exch_bundle[3]),
                                   "grid": _grid_from_subs(exch_bundle[3])}
    if bil_bundle is not None:
        method["bilateral ADMM"] = {"T_in": _tin_from_subs(bil_bundle[0]),
                                    "grid": _grid_from_subs(bil_bundle[0])}
    if len(method) > 1:
        _plot_comfort_peak_bars(
            method, SHOWCASE_DAY, "16_comfort_peak_methods",
            "by method: per-building temperature deviation from setpoint & peak grid import",
        )

    total = dict(scen)
    for k in ("exchange-ADMM", "bilateral ADMM"):
        if k in method:
            total[k] = method[k]
    if total:
        plot_comfort_peak_total(total, SHOWCASE_DAY)
        plot_consumption_timeseries({k: v["grid"] for k, v in total.items()}, SHOWCASE_DAY)
        plot_cost_comparison({k: v["grid"] for k, v in total.items()}, SHOWCASE_DAY)

    # ---- optionally regenerate the ADMM-specific figures from the same solves ----
    if admm_plots and with_admm:
        if exch_bundle is not None:
            pex, price, hist, subs, S = exch_bundle
            try:
                EXC.plot_admm(pex, price, hist, subs, S, SHOWCASE_DAY)
            except Exception as e:  # noqa: BLE001
                print(f"skipped 12_admm_convergence ({e})")
        if bil_bundle is not None:
            _subs, _q, _lam, Z, hist, S = bil_bundle
            try:
                BIL.plot_admm(hist, Z, S, SHOWCASE_DAY)
            except Exception as e:  # noqa: BLE001
                print(f"skipped 14_admm_bilateral ({e})")

    # ---- computational-analysis CSV ----
    comp.append(_comp_row("(whole run)", "ok", time.perf_counter() - run_t0))
    comp_path = PLOTS_DIR / "computational_analysis.csv"
    pd.DataFrame(comp).to_csv(comp_path, index=False)
    print(f"\nsaved {comp_path}")
    print(pd.DataFrame(comp)[["component", "status", "wall_seconds", "iterations",
                              "converged", "community_op_cost_gbp", "community_peak_kW",
                              "rms_cost_vs_central"]].to_string(index=False))


if __name__ == "__main__":
    main(
        with_admm="--no-admm" not in sys.argv,
        admm_plots="--admm-plots" in sys.argv,
    )
