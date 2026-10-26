"""Rule-based peer-to-peer energy sharing for the ToU building dataset.

This is a transparent baseline for comparison with the optimisation scripts.
It uses no solver. Each day starts with the same seeded interior battery state
as the bilateral ADMM model and follows fixed dispatch rules, then reports the
grid cost with and without sharing.

Rules, in order, for each half-hour:
  1. During Medium or High-tariff slots, each building serves its own demand
      from its battery, subject to its reserve and maximum discharge rate.
  2. If demand remains, buildings request energy from the community.
  3. Medium or High-tariff donors export spare battery energy, keeping the
      reserve and respecting their maximum discharge rate. Requests are served
      in descending order of unmet demand; ties use building order.
  4. Any demand still unmet is imported from the grid.
    5. Low-tariff slots do not trigger battery discharge or sharing.

The rule intentionally favours simple, auditable behaviour over optimality. It
is useful as a deterministic benchmark for ``sharing_showcase.py`` and the
ADMM implementations.

Run from this directory with:

    python rule_based_energy_sharing.py

Results are written to ``data/rule_based_sharing_results.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_exploration import TARIFF_PRICE_MAP
from sharing_showcase import N_BUILDINGS, SHARE_FEE_FRAC, _prep

RESERVE_FRAC = 0.20
HIGH_PRICE = TARIFF_PRICE_MAP["High"]
LOW_PRICE = TARIFF_PRICE_MAP["Low"]
OUTPUT_FILE = Path(__file__).parent / "data" / "rule_based_sharing_results.csv"
INIT_SOC_SEED = 42
INIT_SOC_FRAC_LO = 0.15
INIT_SOC_FRAC_HI = 0.85


def dispatch_rule_based(
    load: np.ndarray,
    price: np.ndarray,
    capacity: np.ndarray,
    reserve_frac: float = RESERVE_FRAC,
    initial_soc: np.ndarray | None = None,
    max_discharge: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Dispatch one complete day using the fixed sharing rules.

    Args:
        load: Demand in kWh, shaped ``(time, buildings)``.
        price: Tariff price in p/kWh, shaped ``(time,)``.
        capacity: Battery capacity in kWh, shaped ``(buildings,)``.
        reserve_frac: Fraction of each battery that cannot be discharged.
        initial_soc: Optional initial state of charge in kWh. If omitted, use
            the bilateral model's seeded 15%-85% interior initialization.
        max_discharge: Optional maximum battery output per time slot. If omitted,
            each building's maximum is its largest demand in ``load``.

    Returns:
        ``grid_isolated`` and ``grid_shared`` shaped ``(time, buildings)``,
        plus ``shared`` shaped ``(sender, receiver, time)``.
    """
    load = np.asarray(load, dtype=float)
    price = np.asarray(price, dtype=float)
    capacity = np.asarray(capacity, dtype=float)
    if load.ndim != 2 or price.shape != (load.shape[0],) or capacity.shape != (load.shape[1],):
        raise ValueError("load, price, and capacity have incompatible shapes")
    if np.any(load < 0) or np.any(capacity < 0):
        raise ValueError("load and capacity must be non-negative")
    if not 0 <= reserve_frac < 1:
        raise ValueError("reserve_frac must be in [0, 1)")

    n_slots, n_buildings = load.shape
    if initial_soc is None:
        initial_fractions = np.random.default_rng(INIT_SOC_SEED).uniform(
            INIT_SOC_FRAC_LO, INIT_SOC_FRAC_HI, size=n_buildings
        )
        battery = np.where(capacity > 1e-9, initial_fractions * capacity, 0.0)
    else:
        battery = np.asarray(initial_soc, dtype=float).copy()
        if battery.shape != (n_buildings,) or np.any(battery < 0) or np.any(battery > capacity):
            raise ValueError("initial_soc must be within [0, capacity] for each building")
    reserve = capacity * reserve_frac
    if max_discharge is None:
        max_discharge = load.max(axis=0)
    else:
        max_discharge = np.asarray(max_discharge, dtype=float)
        if max_discharge.shape != (n_buildings,) or np.any(max_discharge < 0):
            raise ValueError("max_discharge must be non-negative for each building")
    grid_isolated = np.zeros_like(load)
    grid_shared = np.zeros_like(load)
    battery_discharge = np.zeros_like(load)
    shared = np.zeros((n_buildings, n_buildings, n_slots))

    for t in range(n_slots):
        own_discharge = np.zeros(n_buildings)
        if price[t] > LOW_PRICE:
            available = np.maximum(battery - reserve, 0.0)
            own_discharge = np.minimum(load[t], np.minimum(available, max_discharge))
            battery -= own_discharge
            battery_discharge[t] += own_discharge

        unmet = load[t] - own_discharge
        grid_isolated[t] = unmet

        if price[t] <= LOW_PRICE:
            grid_shared[t] = unmet
            continue

        donor_energy = np.minimum(
            np.maximum(battery - reserve, 0.0),
            np.maximum(max_discharge - own_discharge, 0.0),
        )
        remaining = unmet.copy()
        # Largest unmet loads get priority, which makes the rule deterministic.
        for receiver in np.argsort(-remaining, kind="stable"):
            if remaining[receiver] <= 0:
                continue
            for donor in np.argsort(-donor_energy, kind="stable"):
                if donor == receiver or donor_energy[donor] <= 0:
                    continue
                amount = min(remaining[receiver], donor_energy[donor])
                shared[donor, receiver, t] += amount
                donor_energy[donor] -= amount
                battery[donor] -= amount
                battery_discharge[t, donor] += amount
                remaining[receiver] -= amount
                if remaining[receiver] <= 1e-12:
                    break
        grid_shared[t] = remaining

    return {
        "grid_isolated": grid_isolated,
        "grid_shared": grid_shared,
        "battery_discharge": battery_discharge,
        "shared": shared,
    }


def evaluate_days(wide: pd.DataFrame, caps: pd.Series, price: pd.Series) -> pd.DataFrame:
    """Evaluate every complete day in ``wide`` and rank sharing savings."""
    capacity = caps.to_numpy(float)
    rows = []
    for day, block in wide.groupby(wide.index.normalize()):
        if len(block) != 48 or block.isna().any().any():
            continue
        day_price = price.reindex(block.index)
        if day_price.isna().any():
            continue

        result = dispatch_rule_based(block.to_numpy(float), day_price.to_numpy(float), capacity)
        p = day_price.to_numpy(float)
        isolated = result["grid_isolated"]
        shared = result["grid_shared"]
        traded = result["shared"].sum()
        cost_isolated = float((isolated.sum(axis=1) * p).sum() * 0.5 / 100)
        cost_shared = float((shared.sum(axis=1) * p).sum() * 0.5 / 100)
        fee = float((result["shared"].sum(axis=(0, 1)) * p).sum() * 0.5 / 100 * SHARE_FEE_FRAC)
        rows.append(
            {
                "day": day,
                "total_load_kWh": float(block.to_numpy(float).sum()),
                "low_hh": int((p <= LOW_PRICE).sum()),
                "medium_hh": int(((p > LOW_PRICE) & (p < HIGH_PRICE)).sum()),
                "high_hh": int((p == HIGH_PRICE).sum()),
                "cost_isolated": cost_isolated,
                "cost_shared": cost_shared + fee,
                "share_benefit": cost_isolated - cost_shared - fee,
                "shared_energy_kWh": float(traded),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("day").sort_values("share_benefit", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-buildings", type=int, default=N_BUILDINGS)
    parser.add_argument("--top", type=int, default=10, help="Number of ranked days to print")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()

    wide, caps, price = _prep(n_buildings=args.n_buildings)
    ranking = evaluate_days(wide, caps, price)
    if ranking.empty:
        raise RuntimeError("No complete tariff-covered days were available")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.output, index=True, date_format="%Y-%m-%d")

    print("Rule-based energy sharing")
    print(
        f"Buildings: {args.n_buildings} | initial SOC: seeded {INIT_SOC_FRAC_LO:.0%}-{INIT_SOC_FRAC_HI:.0%} | "
        f"reserve: {RESERVE_FRAC:.0%} | sharing fee: {SHARE_FEE_FRAC:.0%}"
    )
    print(f"Saved {len(ranking)} evaluated days to {args.output}")
    columns = ["low_hh", "medium_hh", "high_hh", "cost_isolated", "cost_shared", "share_benefit", "shared_energy_kWh"]
    print("\nTop days by sharing benefit (£/day):")
    print(ranking[columns].head(args.top).round(3).to_string())


if __name__ == "__main__":
    main()
