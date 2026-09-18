"""Find the 24h window that best showcases battery sharing between buildings.

Scenario (from the user's optimisation objective):
  * Community = 10 diverse ToU buildings, each with its own battery.
  * Objective is COST: sum_t price[t] * grid_import[b,t] * dt  (+ gas + comfort +
    a seller-side P2P trade fee). Gas/comfort don't depend on electricity sharing,
    so the day-ranking here uses the electricity-cost term only.
  * "Isolated" = each battery serves only its own building.
  * "Shared"   = the batteries act as one pool == perfect P2P routing. Same total
    kWh of storage; only the coupling changes. A trade fee `SHARE_FEE_FRAC` is
    charged on the energy that actually moves between buildings.

Dispatch model (per day): every battery is full at 00:00 (charged overnight on the
Low tariff), then discharges to cut grid import, no intraday recharge, no export.
For a fixed daily kWh budget the cost-optimal policy is price water-filling:
spend battery energy on the highest-price half-hours first across Low, Medium,
and High prices. Both scenarios use
their own optimal policy, so the comparison is apples-to-apples.

  share_benefit(day) [£] = cost_isolated - cost_shared - trade_fee

The best showcase day maximises that. Sharing pays off when a High-tariff period
(67.2 p/kWh) coincides with lopsided demand: one building's battery runs dry while
a neighbour's still has charge.

Outputs a ranking and plots/05_sharing_showcase.png.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_exploration import (
    TARIFF_PRICE_MAP,
    estimate_battery_size_per_building,
    load_tariff_schedule,
    load_tou_data,
    select_diverse_buildings,
)
from plot_timeseries import CONSUMPTION_COLOR, INK, MUTED, PLOTS_DIR

TARGET_HOURS = 2.0  # battery capacity = peak 30-min demand * TARGET_HOURS
N_BUILDINGS = 10
MIN_COMPLETE_DAYS = 300  # community drawn only from buildings this well covered
MIN_DAILY_KWH = 4.0  # drop near-zero / vacant meters
MIN_SHAPE_CV = 0.45  # drop flat "constant use" meters (CV of the mean day profile)
SHARE_FEE_FRAC = 0.10  # seller-side P2P trade fee, as a fraction of the tariff price
DT = 0.5  # hours per half-hour slot

SHARED_COLOR = "#2e8b6e"  # green - the "shared" scenario
PRICE_HIGH_COLOR = "#d98a29"
PRICE_LOW_COLOR = "#5b8fc9"
PRICE_MEDIUM_COLOR = "#8b78b5"


def _prep(n_buildings: int = N_BUILDINGS, day=None) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (wide half-hourly kWh, battery kWh, half-hourly price p/kWh).

    `n_buildings` picks the community size (default 10). The diversity filters
    (coverage / zero-load / flat-profile) are relaxed automatically when they leave
    fewer than `n_buildings` candidates — needed for the N=100 / N=1000 scaling
    runs, where only ~1,100 ToU households exist in total. `day` (a date) restricts
    the returned `wide` to that day only, so the huge all-days unstack is skipped
    for large communities (battery sizes are still estimated from full history)."""
    raw = load_tou_data()
    raw["DateTime"] = pd.to_datetime(raw["DateTime"], errors="coerce")
    raw["KWH/hh (per half hour)"] = pd.to_numeric(raw["KWH/hh (per half hour)"], errors="coerce")
    raw = raw.dropna(subset=["DateTime", "KWH/hh (per half hour)"])
    raw["DateTime"] = raw["DateTime"].dt.floor("30min")

    per_day = raw.groupby(["LCLid", raw["DateTime"].dt.normalize()]).size()
    complete = (per_day == 48).groupby(level="LCLid").sum()
    slot = raw["DateTime"].dt.hour * 2 + raw["DateTime"].dt.minute // 30
    prof = raw.groupby(["LCLid", slot])["KWH/hh (per half hour)"].mean().unstack()
    daily_mean = prof.sum(axis=1) * 1.0  # 48 half-hours -> kWh/day (values already per hh)
    cv = prof.std(axis=1) / prof.mean(axis=1)

    def _candidates(min_days, min_kwh, min_cv):
        p = set(complete[complete >= min_days].index)
        k = prof.index[(daily_mean >= min_kwh) & (cv >= min_cv)]
        return p & set(k)

    pool = _candidates(MIN_COMPLETE_DAYS, MIN_DAILY_KWH, MIN_SHAPE_CV)
    for md, mk, mc in [(200, 2.0, 0.30), (100, 1.0, 0.15), (60, 0.5, 0.0), (14, 0.0, 0.0)]:
        if len(pool) >= n_buildings:
            break
        pool = _candidates(md, mk, mc)
        print(f"  relaxed filters (days>={md}, kWh/day>={mk}, cv>={mc}): {len(pool)} candidates")
    if len(pool) < n_buildings:
        pool = set(prof.dropna().index)
        print(f"  using all {len(pool)} buildings with a complete 48-slot profile")
    print(f"{len(pool)} candidate buildings (need {n_buildings}).")

    raw = raw[raw["LCLid"].isin(pool)]
    ids, _ = select_diverse_buildings(raw.copy(), n_buildings)
    sub_all = raw[raw["LCLid"].isin(ids)].copy()

    caps = (
        estimate_battery_size_per_building(sub_all, target_hours=TARGET_HOURS)
        .set_index("LCLid")["recommended_battery_kWh"]
        .reindex(ids)
    )
    sub = sub_all
    if day is not None:
        sub = sub_all[sub_all["DateTime"].dt.normalize() == pd.Timestamp(day).normalize()]
    wide = (
        sub.groupby(["DateTime", "LCLid"])["KWH/hh (per half hour)"]
        .sum()
        .unstack("LCLid")
        .reindex(columns=ids)
        .sort_index()
    )

    tariff = load_tariff_schedule().set_index("DateTime")["TariffLabel"]
    price = tariff.map(TARIFF_PRICE_MAP).astype(float)
    return wide, caps, price


def _price_waterfill_grid(load: np.ndarray, price: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """Cost-optimal discharge-only dispatch. load/(T,B), price/(T,), capacity/(B,).

    Battery full at t=0; spend it on the highest-price slots first. Returns grid (T,B).
    """
    grid = load.copy()
    order = np.argsort(-price)  # highest price first
    for b in range(load.shape[1]):
        budget = capacity[b]
        for t in order:
            if budget <= 0:
                break
            take = min(budget, grid[t, b])
            grid[t, b] -= take
            budget -= take
    return grid


def evaluate_days(wide: pd.DataFrame, caps: pd.Series, price: pd.Series) -> pd.DataFrame:
    cap_v = caps.to_numpy(float)
    cap_pool = np.array([cap_v.sum()])
    rows = []
    for day, block in wide.groupby(wide.index.normalize()):
        if len(block) != 48 or block.isna().any().any():
            continue
        p = price.reindex(block.index)
        if p.isna().any():
            continue
        p = p.to_numpy(float)
        load = block.to_numpy(float)  # (48, B)
        agg = load.sum(axis=1, keepdims=True)  # (48, 1)

        grid_iso = _price_waterfill_grid(load, p, cap_v)
        grid_shr = _price_waterfill_grid(agg, p, cap_pool)[:, 0]

        cost_nobatt = float(p @ agg[:, 0]) * DT / 100
        cost_iso = float(p @ grid_iso.sum(axis=1)) * DT / 100
        cost_shr = float(p @ grid_shr) * DT / 100

        shared_energy = float(np.maximum(grid_iso.sum(axis=1) - grid_shr, 0).sum())  # kWh moved
        fee = float(p @ np.maximum(grid_iso.sum(axis=1) - grid_shr, 0)) * DT / 100 * SHARE_FEE_FRAC
        rows.append(
            {
                "day": day.normalize(),
                "total_load_kWh": float(agg.sum()),
                "low_hh": int((p == TARIFF_PRICE_MAP["Low"]).sum()),
                "medium_hh": int((p == TARIFF_PRICE_MAP["Medium"]).sum()),
                "high_hh": int((p == TARIFF_PRICE_MAP["High"]).sum()),
                "cost_nobatt": cost_nobatt,
                "cost_isolated": cost_iso,
                "cost_shared": cost_shr + fee,
                "battery_benefit": cost_nobatt - cost_iso,
                "share_benefit": cost_iso - cost_shr - fee,
                "shared_energy_kWh": shared_energy,
            }
        )
    return pd.DataFrame(rows).set_index("day").sort_values("share_benefit", ascending=False)


def plot_showcase(wide, caps, price, day, stats) -> None:
    block = wide.loc[wide.index.normalize() == day]
    hours = block.index.hour + block.index.minute / 60
    p = price.reindex(block.index).to_numpy(float)
    load = block.to_numpy(float)
    cap_v = caps.to_numpy(float)
    agg = load.sum(axis=1)

    grid_iso = _price_waterfill_grid(load, p, cap_v).sum(axis=1)
    grid_shr = _price_waterfill_grid(load.sum(axis=1, keepdims=True), p, np.array([cap_v.sum()]))[:, 0]

    cum_cost_nb = np.cumsum(p * agg) * DT / 100
    cum_cost_iso = np.cumsum(p * grid_iso) * DT / 100
    cum_cost_shr = np.cumsum(p * grid_shr) * DT / 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.6), sharex=True)

    def shade_tariff(ax):
        for i, h in enumerate(hours):
            if p[i] == TARIFF_PRICE_MAP["High"]:
                ax.axvspan(h, h + 0.5, color=PRICE_HIGH_COLOR, alpha=0.13, lw=0)
            elif p[i] == TARIFF_PRICE_MAP["Low"]:
                ax.axvspan(h, h + 0.5, color=PRICE_LOW_COLOR, alpha=0.13, lw=0)
            elif p[i] == TARIFF_PRICE_MAP["Medium"]:
                ax.axvspan(h, h + 0.5, color=PRICE_MEDIUM_COLOR, alpha=0.10, lw=0)

    # Panel 1: demand shapes + community total, tariff periods shaded
    shade_tariff(ax1)
    for col in block.columns:
        ax1.plot(hours, block[col].to_numpy(float) * 2, color=CONSUMPTION_COLOR, lw=1.0, alpha=0.5)
    ax1.plot(hours, agg * 2, color=INK, lw=1.8)
    ax1.plot([], [], color=CONSUMPTION_COLOR, alpha=0.6, label="Individual buildings (n=10)")
    ax1.plot([], [], color=INK, lw=1.8, label="Community total")
    ax1.fill_between(
        [], [], color=PRICE_HIGH_COLOR, alpha=0.2, label=f"High tariff ({TARIFF_PRICE_MAP['High']:.0f} p/kWh)"
    )
    ax1.fill_between(
        [], [], color=PRICE_LOW_COLOR, alpha=0.2, label=f"Low tariff ({TARIFF_PRICE_MAP['Low']:.0f} p/kWh)"
    )
    ax1.fill_between(
        [], [], color=PRICE_MEDIUM_COLOR, alpha=0.2, label=f"Medium tariff ({TARIFF_PRICE_MAP['Medium']:.0f} p/kWh)"
    )
    ax1.set_ylabel("Demand (kW)", color=INK, fontsize=10)
    ax1.set_title(
        f"{day:%A %d %b %Y} — diverse demand across {len(block.columns)} buildings, "
        f"{stats['high_hh'] / 2:.1f} h at the High tariff",
        color=INK,
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)

    # Panel 2: cumulative electricity cost over the day
    shade_tariff(ax2)
    ax2.plot(hours, cum_cost_nb, color=MUTED, lw=1.6, ls="--", label=f"No battery  (£{stats['cost_nobatt']:.2f})")
    ax2.plot(
        hours, cum_cost_iso, color=PRICE_HIGH_COLOR, lw=2.4, label=f"Own battery each  (£{stats['cost_isolated']:.2f})"
    )
    ax2.plot(
        hours, cum_cost_shr, color=SHARED_COLOR, lw=2.4, label=f"Shared battery pool  (£{stats['cost_shared']:.2f})"
    )
    ax2.fill_between(hours, cum_cost_shr, cum_cost_iso, color=SHARED_COLOR, alpha=0.15)
    ax2.annotate(
        f"sharing saves £{stats['share_benefit']:.2f}/day\n"
        f"({stats['shared_energy_kWh']:.1f} kWh routed between buildings)",
        xy=(hours[-1], (cum_cost_iso[-1] + cum_cost_shr[-1]) / 2),
        xytext=(-165, 0),
        textcoords="offset points",
        fontsize=9,
        color=SHARED_COLOR,
        va="center",
        fontweight="bold",
    )
    ax2.set_ylabel("Cumulative electricity cost (£)", color=INK, fontsize=10)
    ax2.set_xlabel("Hour of day", color=INK, fontsize=10)
    ax2.set_title(
        f"Same {cap_v.sum():.0f} kWh of storage: pooling it beats per-building batteries by "
        f"£{stats['share_benefit']:.2f} on top of the £{stats['battery_benefit']:.2f} the batteries already save",
        color=INK,
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    ax2.legend(frameon=False, fontsize=9, loc="upper left")

    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(True, axis="y", color="#e6e6e6", lw=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))

    fig.tight_layout()
    out = PLOTS_DIR / "05_sharing_showcase.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    wide, caps, price = _prep()
    print("\nCommunity buildings and battery sizes (kWh):")
    print(caps.round(2).to_string())
    print(f"Total community storage: {caps.sum():.1f} kWh\n")

    ranking = evaluate_days(wide, caps, price)
    print(f"Evaluated {len(ranking)} complete tariff-covered days. Top 10 by daily £ saved from sharing:\n")
    cols = [
        "total_load_kWh",
        "high_hh",
        "cost_nobatt",
        "cost_isolated",
        "cost_shared",
        "battery_benefit",
        "share_benefit",
        "shared_energy_kWh",
    ]
    print(ranking[cols].head(10).round(2).to_string())

    best = ranking.index[0]
    print(f"\nBest showcase day: {best:%Y-%m-%d} ({best:%A})")
    plot_showcase(wide, caps, price, best, ranking.loc[best])
