"""Collate the input dataset for the best energy-sharing showcase day.

Picks the day with `sharing_showcase` (diverse 10 ToU buildings, cost-optimal
day ranking) and writes an hourly, 24-step optimisation input to
`data/showcase_<day>.csv`, plus per-building battery sizes to
`data/showcase_<day>_batteries.csv`.

Columns in the main file (one row per hour):
    datetime, hour, tariff_label, price_gbp_per_kwh, outdoor_temp_c, setpoint_c,
    load_<LCLid>_kw ...            (mean grid demand in that hour, kW)
    pv_<LCLid>_kw ...              (PV generation, kW; 0 for buildings without PV)

Everything is built from this project's own data:
  * loads / tariff  -> the London (LCL) smart-meter data + Tariffs.xlsx
  * PV generation   -> the project's "PV Data" folder (UK Power Networks LV
                       monitoring): real February per-kWp profile, mean of the
                       3-4 kWp domestic systems. Each PV building is sized PV_KWP.
The LCL data has no heating or weather, so `setpoint_c` is a synthetic domestic
schedule (18 C base, 21 C comfort windows) and `outdoor_temp_c` a synthetic cold-
February diurnal.

The community is HETEROGENEOUS (BUILDING_ASSETS below): some buildings get a
battery, some PV, some both, some neither. The batteries CSV lists all buildings
with capacity 0 where a battery is absent.

Also writes plots/10_showcase_inputs.{png,html}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_exploration import TARIFF_PRICE_MAP, load_tariff_schedule
from plot_timeseries import CONSUMPTION_COLOR, INK, MUTED, PLOTS_DIR
from sharing_showcase import TARGET_HOURS, _prep, evaluate_days

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

PRICE_HIGH_COLOR = "#d98a29"
PRICE_LOW_COLOR = "#5b8fc9"
PRICE_MEDIUM_COLOR = "#8b78b5"

# synthetic heating setpoint (the LCL data carries no thermal information)
SETPOINT_BASE_C = 18.0
SETPOINT_COMFORT_C = 21.0
SETPOINT_COMFORT_HOURS = ((6, 8), (16, 22))  # 21 C morning + evening; 18 C otherwise
OUTDOOR_TEMP_MEAN_C = 4.5  # synthetic: daily mean outdoor temperature
OUTDOOR_TEMP_SWING_C = 3.0  # synthetic: half of peak-to-trough; min ~03:00, max ~15:00

# --- heterogeneous community: asset each building gets, in select_diverse order ---
BUILDING_ASSETS = [
    "battery+pv",
    "pv",
    "battery",
    "none",
    "battery+pv",
    "pv",
    "battery",
    "none",
    "battery+pv",
    "none",
]
PV_KWP = 3.5  # installed PV capacity (kWp) for buildings that have PV


def building_assets(n: int) -> list[str]:
    """Asset label for each of n buildings. n==10 -> BUILDING_ASSETS verbatim;
    larger communities tile the same 10-pattern (keeps the 3:2:2:3 mix)."""
    if n == len(BUILDING_ASSETS):
        return list(BUILDING_ASSETS)
    return (BUILDING_ASSETS * (n // len(BUILDING_ASSETS) + 1))[:n]


UKPN_PV_CSV = (
    Path(__file__).parent
    / "PV Data"
    / "2014-11-28 Cleansed and Processed"
    / "EXPORT HourlyData"
    / "EXPORT HourlyData - Customer Endpoints.csv"
)
# fallback = the profile computed from UKPN_PV_CSV (Feb-mean, per kWp, kW)
_PV_SHAPE_FEB = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.018,
        0.113,
        0.246,
        0.363,
        0.406,
        0.401,
        0.383,
        0.282,
        0.175,
        0.068,
        0.002,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
)


def london_pv_shape(n_hours: int = 24) -> np.ndarray:
    """Per-kWp February PV generation (kW/kWp) from the project's UK Power Networks PV data."""
    if not UKPN_PV_CSV.exists():
        print(f"  ! {UKPN_PV_CSV.name} not found — using baked Feb per-kWp profile")
        return _PV_SHAPE_FEB[:n_hours].copy()
    kwp = {"Bancroft Close": 3.50, "Alverston Close": 3.00, "Maple Drive East": 4.00, "Forest Road": 3.00}
    df = pd.read_csv(UKPN_PV_CSV, usecols=["Substation", "datetime", "P_GEN_MAX"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["g"] = pd.to_numeric(df["P_GEN_MAX"], errors="coerce").clip(lower=0)
    feb = df[(df["datetime"].dt.month == 2) & df["Substation"].isin(kwp)].copy()
    feb["perkwp"] = feb["g"] / feb["Substation"].map(kwp)
    s = feb.groupby(feb["datetime"].dt.hour)["perkwp"].mean().reindex(range(n_hours)).fillna(0.0)
    return s.to_numpy(float)


def synthetic_outdoor_temp(hours: np.ndarray) -> np.ndarray:
    # warmest ~15:00, coldest ~03:00
    return OUTDOOR_TEMP_MEAN_C + OUTDOOR_TEMP_SWING_C * np.cos(2 * np.pi * (hours - 15) / 24)


def synthetic_setpoint(hours: np.ndarray) -> np.ndarray:
    sp = np.full(len(hours), SETPOINT_BASE_C, float)
    for lo, hi in SETPOINT_COMFORT_HOURS:
        sp[(hours >= lo) & (hours < hi)] = SETPOINT_COMFORT_C
    return sp


def _step(y):
    y = np.asarray(y, float)
    return np.arange(len(y) + 1), np.concatenate([y, y[-1:]])


def plot_dataset(out: pd.DataFrame, batt: pd.DataFrame, day) -> None:
    """Visualise the collated optimisation inputs for the showcase day."""
    load_cols = [c for c in out.columns if c.startswith("load_")]
    ids = [c[len("load_") : -len("_kw")] for c in load_cols]
    L = out[load_cols].to_numpy(float).T
    PVb = out[[f"pv_{i}_kw" for i in ids]].to_numpy(float).T
    is_high = (out["tariff_label"] == "High").to_numpy()
    is_low = (out["tariff_label"] == "Low").to_numpy()
    is_medium = ~(is_high | is_low)

    def shade(ax):
        for t in range(len(out)):
            if is_high[t]:
                ax.axvspan(t, t + 1, color=PRICE_HIGH_COLOR, alpha=0.13, lw=0)
            elif is_low[t]:
                ax.axvspan(t, t + 1, color=PRICE_LOW_COLOR, alpha=0.13, lw=0)
            elif is_medium[t]:
                ax.axvspan(t, t + 1, color=PRICE_MEDIUM_COLOR, alpha=0.10, lw=0)

    fig, (a1, a2, a3, a4) = plt.subplots(4, 1, figsize=(11, 11.5))

    shade(a1)
    for b in range(len(ids)):
        a1.step(*_step(L[b]), where="post", color=CONSUMPTION_COLOR, lw=1.0, alpha=0.45)
    a1.step(*_step(L.sum(axis=0)), where="post", color=INK, lw=1.9)
    a1.step(*_step(PVb.sum(axis=0)), where="post", color="#e0a800", lw=1.9)
    a1.plot([], [], color=CONSUMPTION_COLOR, alpha=0.6, label=f"individual loads (n={len(ids)})")
    a1.plot([], [], color=INK, lw=1.9, label="community load")
    a1.plot([], [], color="#e0a800", lw=1.9, label=f"community PV ({int((PVb.max(axis=1) > 0).sum())} bldgs)")
    a1.set_ylabel("Power (kW)", color=INK, fontsize=10)
    a1.set_title(
        f"{pd.Timestamp(day):%A %d %b %Y} — collated optimisation inputs",
        color=INK,
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    a1.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper left")

    shade(a2)
    a2.step(*_step(out["price_gbp_per_kwh"].to_numpy()), where="post", color=PRICE_HIGH_COLOR, lw=2)
    a2.set_ylabel("Electricity price (£/kWh)", color=INK, fontsize=10)

    shade(a3)
    a3.step(
        *_step(out["outdoor_temp_c"].to_numpy()),
        where="post",
        color=PRICE_LOW_COLOR,
        lw=2,
        label="outdoor (synthetic)",
    )
    a3.step(
        *_step(out["setpoint_c"].to_numpy()), where="post", color=INK, lw=1.5, ls="--", label="setpoint (synthetic)"
    )
    a3.set_ylabel("Temperature (°C)", color=INK, fontsize=10)
    a3.set_xlabel("Hour of day", color=INK, fontsize=10)
    a3.set_ylim(0, 26)
    a3.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)

    order = np.argsort(-batt["capacity_kwh"].to_numpy())
    xb = np.arange(len(ids))
    a4.bar(xb - 0.19, batt["capacity_kwh"].to_numpy()[order], 0.36, color=CONSUMPTION_COLOR, label="battery (kWh)")
    a4.bar(xb + 0.19, batt["pv_kwp"].to_numpy()[order], 0.36, color="#e0a800", label="PV (kWp)")
    a4.set_xticks(xb)
    a4.set_xticklabels(
        [f"{i}\n{a}" for i, a in zip(batt["LCLid"].to_numpy()[order], batt["assets"].to_numpy()[order])],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    a4.set_ylabel("capacity", color=INK, fontsize=10)
    nb = int((batt["capacity_kwh"] > 0).sum())
    npv = int((batt["pv_kwp"] > 0).sum())
    a4.set_title(
        f"Heterogeneous assets: {nb} batteries ({batt['capacity_kwh'].sum():.0f} kWh), "
        f"{npv} PV ({batt['pv_kwp'].sum():.0f} kWp)",
        color=INK,
        fontsize=10,
        fontweight="bold",
        loc="left",
        pad=8,
    )
    a4.legend(frameon=False, fontsize=8)

    for ax in (a1, a2, a3):
        ax.set_xlim(0, len(out))
        ax.set_xticks(range(0, len(out) + 1, 3))
    for ax in (a1, a2, a3, a4):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(MUTED)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(True, axis="y", color="#ededed", lw=0.7)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    out_png = PLOTS_DIR / "10_showcase_inputs.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"saved {out_png}")

    _plotly_dataset(out, batt, day, ids, L, is_high)


def _plotly_dataset(out, batt, day, ids, L, is_high) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return
    is_low = (out["tariff_label"] == "Low").to_numpy()
    is_medium = ~(is_high | is_low)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=("Electrical load (kW)", "Electricity price (£/kWh)", "Temperature (°C)"),
    )
    for b in range(len(ids)):
        x, y = _step(L[b])
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                line=dict(color="rgba(59,107,176,.35)", width=1, shape="hv"),
                name=ids[b],
                legendgroup="b",
                showlegend=(b == 0),
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
    x, y = _step(L.sum(axis=0))
    fig.add_trace(
        go.Scatter(x=x, y=y, line=dict(color="black", width=2, shape="hv"), name="community load"), row=1, col=1
    )
    pv_cols = [c for c in out.columns if c.startswith("pv_")]
    if pv_cols:
        x, y = _step(out[pv_cols].to_numpy(float).sum(axis=1))
        fig.add_trace(
            go.Scatter(x=x, y=y, line=dict(color="#e0a800", width=2, shape="hv"), name="community PV"), row=1, col=1
        )
    x, y = _step(out["price_gbp_per_kwh"].to_numpy())
    fig.add_trace(go.Scatter(x=x, y=y, line=dict(color="#d98a29", width=2, shape="hv"), name="price"), row=2, col=1)
    x, y = _step(out["outdoor_temp_c"].to_numpy())
    fig.add_trace(
        go.Scatter(x=x, y=y, line=dict(color="#5b8fc9", width=2, shape="hv"), name="outdoor (synthetic)"), row=3, col=1
    )
    x, y = _step(out["setpoint_c"].to_numpy())
    fig.add_trace(
        go.Scatter(x=x, y=y, line=dict(color="black", width=1.5, shape="hv", dash="dash"), name="setpoint"),
        row=3,
        col=1,
    )
    for mask, colour, opacity in ((is_low, "#5b8fc9", 0.08), (is_medium, "#8b78b5", 0.07), (is_high, "#d98a29", 0.10)):
        for start in np.flatnonzero(mask & ~np.r_[False, mask[:-1]]):
            stop = start
            while stop + 1 < len(mask) and mask[stop + 1]:
                stop += 1
            for r in (1, 2, 3):
                fig.add_vrect(x0=start, x1=stop + 1, fillcolor=colour, opacity=opacity, line_width=0, row=r, col=1)
    fig.update_xaxes(title_text="Hour of day", dtick=3, row=3, col=1)
    fig.update_layout(
        template="plotly_white",
        height=800,
        hovermode="x unified",
        title=f"{pd.Timestamp(day):%A %d %b %Y} — collated optimisation inputs",
    )
    out_html = PLOTS_DIR / "10_showcase_inputs.html"
    fig.write_html(out_html, include_plotlyjs=True)
    print(f"saved {out_html}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10, help="community size (default 10)")
    ap.add_argument(
        "--day",
        default=None,
        help="showcase day YYYY-MM-DD; for --n != 10 defaults to showcase_latest.txt "
        "(the day is NOT re-ranked for scaled communities)",
    )
    args = ap.parse_args()
    n = args.n
    scaled = n != 10

    if scaled:
        day_str = args.day or (
            (DATA_DIR / "showcase_latest.txt").read_text().strip()
            if (DATA_DIR / "showcase_latest.txt").exists()
            else "2013-02-21"
        )
        day = pd.Timestamp(day_str).normalize()
        print(f"Scaled community: N={n}, fixed day {day:%Y-%m-%d} (no day re-ranking)\n")
        wide, caps, price = _prep(n_buildings=n, day=day)
    else:
        wide, caps, price = _prep()
        ranking = evaluate_days(wide, caps, price)
        day = ranking.index[0]
        stats = ranking.loc[day]
        print(f"Best showcase day: {day:%Y-%m-%d} ({day:%A})")
        print(
            f"  isolated £{stats['cost_isolated']:.2f} | shared £{stats['cost_shared']:.2f} "
            f"| sharing saves £{stats['share_benefit']:.2f}/day (proxy dispatch)\n"
        )

    # --- half-hourly load for the day -> hourly mean kW (kWh per half hour, 2 per hour) ---
    block = wide.loc[wide.index.normalize() == day].sort_index()
    assert len(block) == 48, f"expected 48 half-hours, got {len(block)}"
    load_kw_hh = block * 2.0  # kWh/hh -> average kW over the half hour
    load_hourly = load_kw_hh.resample("1h").mean()  # 24 rows, mean kW per hour

    labels = load_tariff_schedule().set_index("DateTime")["TariffLabel"].reindex(block.index)
    label_hourly = labels.resample("1h").first()
    price_hourly_p = label_hourly.map(TARIFF_PRICE_MAP).astype(float)

    hours = np.arange(24)
    out = pd.DataFrame({"datetime": load_hourly.index})
    out["hour"] = hours
    out["tariff_label"] = label_hourly.to_numpy()
    out["price_gbp_per_kwh"] = (price_hourly_p / 100.0).to_numpy()
    out["outdoor_temp_c"] = synthetic_outdoor_temp(hours)
    out["setpoint_c"] = synthetic_setpoint(hours)

    ids = list(load_hourly.columns)
    assets = building_assets(len(ids))
    pv_shape = london_pv_shape(len(hours))
    for i, lclid in enumerate(ids):
        out[f"load_{lclid}_kw"] = load_hourly[lclid].to_numpy()
        out[f"pv_{lclid}_kw"] = PV_KWP * pv_shape if "pv" in assets[i] else np.zeros(len(hours))

    suffix = f"_N{n:04d}" if scaled else ""
    main_path = DATA_DIR / f"showcase_{day:%Y-%m-%d}{suffix}.csv"
    out.to_csv(main_path, index=False)
    print(f"wrote {main_path}  ({out.shape[0]} hours x {len(ids)} buildings)")

    has_batt = np.array(["battery" in a for a in assets])
    cap_v = caps.reindex(ids).to_numpy(float)
    batt = pd.DataFrame(
        {
            "LCLid": ids,
            "assets": assets,
            "capacity_kwh": np.where(has_batt, cap_v, 0.0),
            "max_power_kw": np.where(has_batt, cap_v / TARGET_HOURS, 0.0),
            "initial_soc_kwh": np.where(has_batt, cap_v, 0.0),  # charged full overnight
            "pv_kwp": np.where([("pv" in a) for a in assets], PV_KWP, 0.0),
        }
    )
    batt_path = DATA_DIR / f"showcase_{day:%Y-%m-%d}{suffix}_batteries.csv"
    batt.to_csv(batt_path, index=False)
    print(f"wrote {batt_path}")
    if not scaled:
        print(batt.round(2).to_string(index=False))
    else:
        nb = int((batt["capacity_kwh"] > 0).sum())
        npv = int((batt["pv_kwp"] > 0).sum())
        print(f"  {nb} batteries ({batt['capacity_kwh'].sum():.0f} kWh), {npv} PV ({batt['pv_kwp'].sum():.0f} kWp)")

    if not scaled:
        # a pointer file so the optimisation script always finds the latest showcase day
        (DATA_DIR / "showcase_latest.txt").write_text(f"{day:%Y-%m-%d}\n")
        plot_dataset(out, batt, day)
    else:
        print("  (scaled dataset: showcase_latest.txt and 10_showcase_inputs plot left untouched)")


if __name__ == "__main__":
    main()
