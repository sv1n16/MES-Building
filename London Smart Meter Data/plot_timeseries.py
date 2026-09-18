"""Plot household consumption vs time and PV generation vs time.

Figures produced:
  plots/01_consumption_over_time.png    - mean half-hourly consumption per ToU household
  plots/02_pv_generation_over_time.png  - mean PV generation across monitored endpoints
  plots/03_pv_endpoints_gen_vs_import.png - per PV endpoint: generation and grid import together
  plots/04_consumption_per_building.png - daily consumption for a sample of individual LCLids
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from data_exploration import load_pv_data, load_tou_data


@lru_cache(maxsize=1)
def _tou_daily() -> pd.DataFrame:
    """ToU meter data, cleaned: LCLid, DateTime, kWh (half-hourly). Cached per process."""
    data = load_tou_data()
    data["DateTime"] = pd.to_datetime(data["DateTime"], errors="coerce")
    data["kWh"] = pd.to_numeric(data["KWH/hh (per half hour)"], errors="coerce")
    return data.dropna(subset=["DateTime", "kWh"])[["LCLid", "DateTime", "kWh"]]

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

INK = "#1a1a1a"
MUTED = "#6b6b6b"
CONSUMPTION_COLOR = "#3b6bb0"  # sequential blue
PV_COLOR = "#d98a29"  # amber


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))


def plot_consumption() -> None:
    data = _tou_daily()

    # Mean half-hourly consumption per household, then resample to daily.
    per_slot = data.groupby("DateTime")["kWh"].mean()
    daily = per_slot.resample("D").mean()
    daily = daily[(daily.index >= "2013-01-01")]  # trim sparse early tail
    rolling = daily.rolling(7, center=True, min_periods=3).mean()

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(daily.index, daily.values, color=CONSUMPTION_COLOR, linewidth=0.8, alpha=0.35,
            label="Daily mean")
    ax.plot(rolling.index, rolling.values, color=CONSUMPTION_COLOR, linewidth=2,
            label="7-day rolling mean")
    _style_axes(ax)
    ax.set_ylabel("Consumption per household\n(kWh / half-hour)", color=INK, fontsize=10)
    ax.set_title(
        f"Household electricity consumption over time  "
        f"({data['LCLid'].nunique():,} ToU households)",
        color=INK, fontsize=12, fontweight="bold", loc="left", pad=12,
    )
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out = PLOTS_DIR / "01_consumption_over_time.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_pv_generation() -> None:
    pv = load_pv_data()
    pv = pv[["SerialNo", "datetime", "P_GEN_MAX"]].copy()
    pv["datetime"] = pd.to_datetime(pv["datetime"], errors="coerce")
    pv["P_GEN_MAX"] = pd.to_numeric(pv["P_GEN_MAX"], errors="coerce")
    pv = pv.dropna()

    per_hour = pv.groupby("datetime")["P_GEN_MAX"].mean()
    daily = per_hour.resample("D").mean()
    rolling = daily.rolling(7, center=True, min_periods=3).mean()

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(daily.index, daily.values, color=PV_COLOR, linewidth=0.8, alpha=0.35,
            label="Daily mean")
    ax.plot(rolling.index, rolling.values, color=PV_COLOR, linewidth=2,
            label="7-day rolling mean")
    _style_axes(ax)
    ax.set_ylabel("PV generation per endpoint\n(kW, hourly mean)", color=INK, fontsize=10)
    ax.set_title(
        f"PV generation over time  ({pv['SerialNo'].nunique()} monitored endpoints)",
        color=INK, fontsize=12, fontweight="bold", loc="left", pad=12,
    )
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out = PLOTS_DIR / "02_pv_generation_over_time.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_pv_endpoints_gen_vs_import() -> None:
    pv = load_pv_data()
    cols = ["SerialNo", "Substation", "datetime", "P_GEN_MAX", "P_IMPORT_MAX"]
    pv = pv[cols].copy()
    pv["datetime"] = pd.to_datetime(pv["datetime"], errors="coerce")
    for c in ("P_GEN_MAX", "P_IMPORT_MAX"):
        pv[c] = pd.to_numeric(pv[c], errors="coerce")
    pv = pv.dropna(subset=["datetime"])

    endpoints = (
        pv.groupby(["SerialNo", "Substation"])["datetime"].count()
        .sort_values(ascending=False).index.tolist()
    )
    ncols, nrows = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 9), sharex=True)
    axes = axes.ravel()

    for ax, (serial, substation) in zip(axes, endpoints):
        g = pv[pv["SerialNo"].eq(serial)].set_index("datetime").sort_index()
        imp = g["P_IMPORT_MAX"].resample("D").mean().rolling(7, center=True, min_periods=3).mean()
        gen = g["P_GEN_MAX"].resample("D").mean().rolling(7, center=True, min_periods=3).mean()
        ax.plot(imp.index, imp.values, color=CONSUMPTION_COLOR, linewidth=1.8, label="Grid import")
        ax.plot(gen.index, gen.values, color=PV_COLOR, linewidth=1.8, label="PV generation")
        _style_axes(ax)
        ax.set_ylim(bottom=0)
        ax.set_title(substation, color=INK, fontsize=10, fontweight="bold", loc="left", pad=6)

    for ax in axes[len(endpoints):]:
        ax.set_visible(False)

    for ax in axes[:len(endpoints)]:
        if not ax.get_subplotspec().is_first_col():
            continue
        ax.set_ylabel("Power (kW)\n7-day mean of hourly max", color=INK, fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=10, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("PV Customer Endpoints: generation vs grid import over time",
                 color=INK, fontsize=13, fontweight="bold", x=0.09, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = PLOTS_DIR / "03_pv_endpoints_gen_vs_import.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_consumption_per_building(building_ids: list[str] | None = None, n: int = 12) -> None:
    """Daily consumption over time, one panel per LCLid.

    building_ids: explicit list of LCLids to plot. If None, a diverse sample of `n`
    buildings is chosen with data_exploration.select_diverse_buildings.
    """
    data = _tou_daily()
    data = data[data["DateTime"] >= "2013-01-01"]

    if building_ids is None:
        # Buildings covering most of the window, spread across the consumption range.
        span = data.groupby("LCLid")["DateTime"].agg(lambda s: (s.max() - s.min()).days)
        total = data.groupby("LCLid")["kWh"].sum()
        eligible = total[(span >= 300) & (total > 0)].sort_values()
        picks = (eligible.index[(len(eligible) - 1) * i // (n - 1)] for i in range(n))
        building_ids = list(dict.fromkeys(picks))

    ncols = 3
    nrows = -(-len(building_ids) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.3 * nrows),
                             sharex=True, squeeze=False)
    axes = axes.ravel()

    for ax, lclid in zip(axes, building_ids):
        g = data[data["LCLid"].eq(lclid)].set_index("DateTime").sort_index()
        daily = g["kWh"].resample("D").sum().replace(0, pd.NA).dropna()
        daily = daily[daily.index >= "2013-01-01"]
        rolling = daily.rolling(7, center=True, min_periods=3).mean()
        ax.plot(daily.index, daily.values, color=CONSUMPTION_COLOR, linewidth=0.7, alpha=0.3)
        ax.plot(rolling.index, rolling.values, color=CONSUMPTION_COLOR, linewidth=1.6)
        _style_axes(ax)
        ax.set_ylim(bottom=0)
        ax.set_title(lclid, color=INK, fontsize=10, fontweight="bold", loc="left", pad=6)

    for ax in axes[len(building_ids):]:
        ax.set_visible(False)
    for ax in axes[:len(building_ids)]:
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("kWh / day", color=INK, fontsize=9)

    fig.suptitle(
        "Daily electricity consumption per building (7-day rolling mean; raw daily faint)",
        color=INK, fontsize=13, fontweight="bold", x=0.06, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = PLOTS_DIR / "04_consumption_per_building.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    plot_consumption()
    plot_consumption_per_building()
    plot_pv_generation()
    plot_pv_endpoints_gen_vs_import()
