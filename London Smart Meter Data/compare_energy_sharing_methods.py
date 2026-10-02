"""Compare rule-based, central, exchange-ADMM, and bilateral-ADMM sharing.

The comparison uses the current hourly showcase dataset and writes one row per
method to ``plots/22_energy_sharing_method_comparison.csv``. Metrics are:

* operating cost: electricity + gas + community-level fees, GBP/day;
* peak grid import: maximum community grid import, kW;
* peak grid import hour: hour at which the community peak occurs;
* thermal comfort: RMS indoor-temperature error from the setpoint, deg C;
* net energy received: positive per-building net P2P imports, counted once, kWh/day.

Building-level KPIs versus the battery-enabled no-sharing baseline are written
to ``plots/28_building_method_comparison.csv`` with a signed cost-savings
heatmap in ``plots/29_building_cost_savings_heatmap.{png,pdf}``. Building energy
costs include grid electricity and gas, and exclude P2P cash settlements.

The rule-based method uses the same loads, PV, battery capacities, initial SOC,
and battery power limits as the optimisation methods. Its thermal schedule is a
simple thermostat: meet the required heat with a heat pump first, then a boiler.
Its sharing dispatch is deterministic and solver-free.

Run from this directory:

    py -3 compare_energy_sharing_methods.py

Use ``--no-admm`` for a quick central-versus-rule comparison.
"""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import admm_bilateral_p2p_valley as BIL
import admm_energy_sharing as EXC
import central_optimisation_showcase as C
from rule_based_energy_sharing import dispatch_rule_based

OUTPUT_DIR = Path(__file__).parent / "plots" / "MES ADMM Optimisation Community Size 10"
OUTPUT_FILE = OUTPUT_DIR / "22_energy_sharing_method_comparison.csv"
PLOT_FILE = OUTPUT_DIR / "22_energy_sharing_method_comparison.png"
SURPLUS_PLOT_FILE = OUTPUT_DIR / "22_sharing_surplus_comparison.png"
DESTINATION_PLOT_FILE = OUTPUT_DIR / "25_sharing_destinations_comparison.png"
GRID_SAVING_PLOT_FILE = OUTPUT_DIR / "26_shared_energy_vs_grid_saving.png"
HEATMAP_PLOT_FILE = OUTPUT_DIR / "23_building_grid_import_heatmap.png"
SHARED_HEATMAP_PLOT_FILE = OUTPUT_DIR / "24_building_shared_energy_heatmap.png"
BUILDING_COMPARISON_FILE = OUTPUT_DIR / "28_building_method_comparison.csv"
BUILDING_SAVINGS_PLOT_FILE = OUTPUT_DIR / "29_building_cost_savings_heatmap.png"
BUILDING_COST_BARS_FILE = OUTPUT_DIR / "30_building_cost_comparison.png"
METHOD_COLORS = {
    "centralised no-sharing": "#555555",
    "rule-based": "#8a8f98",
    "central": "#2e8b6e",
    "exchange-ADMM": "#3b6bb0",
    "bilateral ADMM": "#c0392b",
    "bilateral ADMM (no valley fill)": "#b08a5a",
    "bilateral ADMM + valley filling": "#c0392b",
}
METHOD_ORDER = (
    "centralised no-sharing",
    "rule-based",
    "central",
    "exchange-ADMM",
    "bilateral ADMM (no valley fill)",
    "bilateral ADMM + valley filling",
)


def _rms_comfort(t_in: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(t_in) - C.T_SET[None, :]) ** 2)))


def _max_temperature_deviation(t_in: np.ndarray) -> float:
    """Return the largest absolute indoor-temperature error in deg C."""
    return float(np.max(np.abs(np.asarray(t_in) - C.T_SET[None, :])))


def _grid_peak(grid: np.ndarray) -> float:
    return float(np.asarray(grid, dtype=float).sum(axis=0).max())


def _grid_peak_hour(grid: np.ndarray) -> int:
    """Return the first hour containing the maximum community grid import."""
    community_grid = np.asarray(grid, dtype=float).sum(axis=0)
    return int(np.argmax(community_grid))


def _grid_import_total(grid: np.ndarray) -> float:
    """Return total community electricity imported from the grid in kWh."""
    return float(np.asarray(grid, dtype=float).sum() * C.dt)


def _building_energy_cost(grid: np.ndarray, gas: np.ndarray) -> np.ndarray:
    """Return each building's grid-electricity and gas cost, excluding P2P settlements."""
    electricity_cost = (C.PRICE[None, :] * np.asarray(grid, dtype=float) * C.dt).sum(axis=1)
    gas_cost = (C.gas_price / 100.0 * np.asarray(gas, dtype=float) * C.dt).sum(axis=1)
    return electricity_cost + gas_cost


def _available_surplus(discharge: np.ndarray, p_hp: np.ndarray, charge: np.ndarray) -> np.ndarray:
    """Community energy available for export under each model's constraint."""
    return np.maximum(C.PV_B + discharge, 0.0).sum(axis=0)


def _sub_arrays(subs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = np.array([EXC._val(m, "p_el") for m in subs])
    gas = np.array([EXC._val(m, "gas") for m in subs])
    t_in = np.array([EXC._val(m, "T_in") for m in subs])
    return grid, gas, t_in, np.asarray(C.PRICE, dtype=float)


def _net_from_pairwise_matrix(mat: np.ndarray) -> np.ndarray:
    """Net per-building P2P energy from a bilateral (sender, receiver, t) matrix.

    Assumes ``mat[i, j, t]`` is the (non-negative) energy sent from building
    ``i`` to building ``j`` at hour ``t`` -- axis 0 = sender, axis 1 =
    receiver, axis 2 = time. Net for building b = received - sent, so
    positive = net importer via P2P that hour, negative = net exporter
    (same convention as the exchange-ADMM ``pex`` net-trade array).

    VERIFY this axis/sign convention against ``result["shared"]`` /
    ``result["share"]`` in rule_based_energy_sharing.py and
    central_optimisation_showcase.py before trusting the plot -- swap axis
    0/1 below if the convention is reversed there.
    """
    mat = np.asarray(mat, dtype=float)
    received = mat.sum(axis=0)  # sum over senders -> (n_buildings, T)
    sent = mat.sum(axis=1)  # sum over receivers -> (n_buildings, T)
    return received - sent


def _net_from_trade_dict(trades: dict, n_buildings: int, time_horizon: int) -> np.ndarray:
    """Net per-building P2P trade from the bilateral-ADMM ``trades`` dict.

    Assumes each key is a building-index pair ``(i, j)`` and the value is a
    length-T array of energy sent from i to j (positive = i exports to j).
    Net for building b = received - sent (same convention as
    ``_net_from_pairwise_matrix``).

    VERIFY this against admm_bilateral_p2p_valley.py -- if ``trades`` stores
    both (i, j) and (j, i) as independent non-negative flows, or the value
    is already net-signed per unordered pair, adjust the accumulation below.
    """
    net = np.zeros((n_buildings, time_horizon))
    for (i, j), flow in trades.items():
        flow = np.asarray(flow, dtype=float)
        net[i] -= flow  # i sends -> reduces i's net
        net[j] += flow  # j receives -> increases j's net
    return net


def _net_shared_per_hour(net_imports: np.ndarray) -> np.ndarray:
    """Return community net P2P energy received per hour, counting imports once."""
    return np.maximum(np.asarray(net_imports, dtype=float), 0.0).sum(axis=0)


def _net_shared_quantity(net_imports: np.ndarray) -> float:
    """Return net P2P energy received across the day in kWh."""
    return float(_net_shared_per_hour(net_imports).sum() * C.dt)


def _community_trade_balance(net_imports: np.ndarray) -> tuple[float, float, float]:
    """Return (bought, sold, net) community P2P energy totals in kWh/day.

    The community balance is identically zero because every kWh sold by one
    building is bought by another. Positive values represent net imports;
    negative values mean net exports.
    """
    net_imports = np.asarray(net_imports, dtype=float)
    bought = float(np.maximum(net_imports, 0.0).sum() * C.dt)
    sold = float(np.maximum(-net_imports, 0.0).sum() * C.dt)
    return bought, sold, bought - sold


def _battery_soc(charge: np.ndarray, discharge: np.ndarray, capacity, initial_soc) -> np.ndarray:
    """Approximate per-building battery state of charge over time (kWh).

    Integrates soc[t] = soc[t-1] + charge[t]*dt - discharge[t]*dt assuming
    unity round-trip efficiency, since none of the run_* functions currently
    expose an already-solved SOC trajectory. If central_optimisation_showcase
    / admm_energy_sharing / admm_bilateral_p2p_valley already return a solved
    ``soc`` array (e.g. via ``EXC._val(m, "soc")`` or a "soc" key in the
    extraction dict), swap this out for that directly -- this integration is
    a best-effort approximation and ignores charge/discharge efficiency
    losses and self-discharge that your actual model may include.
    """
    charge = np.asarray(charge, dtype=float)
    discharge = np.asarray(discharge, dtype=float)
    n_buildings, time_horizon = charge.shape
    cap = np.broadcast_to(np.asarray(capacity, dtype=float), (n_buildings,))
    prev = np.broadcast_to(np.asarray(initial_soc, dtype=float), (n_buildings,)).copy()
    soc = np.zeros((n_buildings, time_horizon))
    for t in range(time_horizon):
        prev = np.clip(prev + charge[:, t] * C.dt - discharge[:, t] * C.dt, 0.0, cap)
        soc[:, t] = prev
    return soc


def _thermal_rule() -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic heat-pump electricity and gas for the showcase day."""
    t_in = np.broadcast_to(C.INIT_T_IN[:, None], (C.n_buildings, C.time_horizon)).copy()
    p_hp = np.zeros_like(t_in)
    gas = np.zeros_like(t_in)
    cop = C.cop_base + 0.01 * (C.T_OUT - C.T_ref)

    for b in range(C.n_buildings):
        for t in range(1, C.time_horizon):
            required_heat = (C.T_SET[t] - t_in[b, t - 1]) / (C.dt / 10.0)
            required_heat += 0.5 * (t_in[b, t - 1] - C.T_OUT[t])
            heat = float(np.clip(required_heat, 0.0, C.p_th_nom + C.max_thermal_power))
            hp_heat = min(heat, C.p_th_nom)
            p_hp[b, t] = hp_heat / cop[t]
            gas[b, t] = max(heat - C.p_th_nom, 0.0) / C.efficiency
            t_in[b, t] = t_in[b, t - 1] + C.dt / 10.0 * (heat - 0.5 * (t_in[b, t - 1] - C.T_OUT[t]))
    return p_hp, gas, t_in


def run_no_sharing() -> dict:
    """Solve the common no-P2P baseline used by every sharing comparison."""
    result = C.solve_and_extract(C.build_model(True, False), "comparison_no_sharing")
    bought, sold, net = _community_trade_balance(np.zeros((C.n_buildings, C.time_horizon)))
    return {
        "method": "centralised no-sharing",
        "operating_cost_gbp_per_day": float(result["op_cost"]),
        "peak_grid_import_kw": _grid_peak(result["grid"]),
        "peak_grid_import_hour": _grid_peak_hour(result["grid"]),
        "grid_import_kwh_per_day": _grid_import_total(result["grid"]),
        "grid_saving_kwh_per_day": 0.0,
        "thermal_comfort_rms_degC": _rms_comfort(result["T_in"]),
        "thermal_comfort_max_deviation_degC": _max_temperature_deviation(result["T_in"]),
        "net_energy_received_kwh_per_day": 0.0,
        "energy_bought_kwh_per_day": bought,
        "energy_sold_kwh_per_day": sold,
        "net_energy_shared_kwh_per_day": net,
        "community_energy_balance_error_kwh_per_day": abs(net),
        "_grid_series": result["grid"].sum(axis=0),
        "_grid_per_building": np.asarray(result["grid"], dtype=float),
        "_energy_cost_per_building": np.asarray(result["elec_b"] + result["gas_b"], dtype=float),
        "_temperature_per_building": np.asarray(result["T_in"], dtype=float),
        "_grid_baseline_total": _grid_import_total(result["grid"]),
        "_grid_saving_total": 0.0,
        "_available_surplus": _available_surplus(result["discharge"], result["p_hp"], result["charge"]),
        "_shared_per_building": np.zeros((C.n_buildings, C.time_horizon)),
        "_p_hp_per_building": np.asarray(result["p_hp"], dtype=float),
        "_charge_per_building": np.asarray(result["charge"], dtype=float),
        "_discharge_per_building": np.asarray(result["discharge"], dtype=float),
        "note": "centralised no-sharing baseline",
    }


def run_rule_based() -> dict:
    p_hp, gas, t_in = _thermal_rule()
    net_load = np.maximum(C.LOAD + p_hp - C.PV_B, 0.0).T
    result = dispatch_rule_based(
        net_load,
        C.PRICE * 100.0,
        C.CAP,
        initial_soc=C.INIT_SOC,
        max_discharge=C.MAXPOW * C.dt,
    )
    grid = result["grid_shared"].T
    baseline_grid = result["grid_isolated"].T
    shared_net = _net_from_pairwise_matrix(result["shared"]) / C.dt
    cost = float((C.PRICE[None, :] * grid * C.dt).sum())
    cost += float((C.gas_price / 100.0 * gas * C.dt).sum())
    discharge_rb = result["battery_discharge"].T
    if "battery_charge" in result:
        charge_rb = result["battery_charge"].T
    else:
        # rule_based_energy_sharing.py may only track discharge explicitly;
        # verify -- if charging is real but untracked, SOC below will be wrong.
        charge_rb = np.zeros_like(discharge_rb)
    bought, sold, net = _community_trade_balance(shared_net)
    return {
        "method": "rule-based",
        "operating_cost_gbp_per_day": cost,
        "peak_grid_import_kw": _grid_peak(grid),
        "peak_grid_import_hour": _grid_peak_hour(grid),
        "grid_import_kwh_per_day": _grid_import_total(grid),
        "grid_saving_kwh_per_day": _grid_import_total(baseline_grid) - _grid_import_total(grid),
        "thermal_comfort_rms_degC": _rms_comfort(t_in),
        "thermal_comfort_max_deviation_degC": _max_temperature_deviation(t_in),
        "net_energy_received_kwh_per_day": _net_shared_quantity(shared_net),
        "energy_bought_kwh_per_day": bought,
        "energy_sold_kwh_per_day": sold,
        "net_energy_shared_kwh_per_day": net,
        "community_energy_balance_error_kwh_per_day": abs(net),
        "_grid_series": grid.sum(axis=0),
        "_grid_per_building": np.asarray(grid, dtype=float),
        "_energy_cost_per_building": _building_energy_cost(grid, gas),
        "_temperature_per_building": np.asarray(t_in, dtype=float),
        "_grid_baseline_total": _grid_import_total(baseline_grid),
        "_grid_saving_total": _grid_import_total(baseline_grid) - _grid_import_total(grid),
        "_available_surplus": _available_surplus(discharge_rb, p_hp, charge_rb),
        "_shared_per_building": shared_net,
        "_p_hp_per_building": np.asarray(p_hp, dtype=float),
        "_charge_per_building": charge_rb,
        "_discharge_per_building": discharge_rb,
        "note": "deterministic thermostat and rule dispatch",
    }


def run_central() -> dict:
    baseline = C.solve_and_extract(C.build_model(True, False), "comparison_central_baseline")
    result = C.solve_and_extract(
        C.build_model(True, True, fee_frac=C.SHARE_TRADE_FEE_FRAC, fee_mode="market"),
        "comparison_central",
        fee_frac=C.SHARE_TRADE_FEE_FRAC,
        fee_mode="market",
    )
    shared_net = _net_from_pairwise_matrix(result["share"])
    bought, sold, net = _community_trade_balance(shared_net)
    return {
        "method": "central",
        "operating_cost_gbp_per_day": float(result["op_cost"]),
        "peak_grid_import_kw": _grid_peak(result["grid"]),
        "peak_grid_import_hour": _grid_peak_hour(result["grid"]),
        "grid_import_kwh_per_day": _grid_import_total(result["grid"]),
        "grid_saving_kwh_per_day": _grid_import_total(baseline["grid"]) - _grid_import_total(result["grid"]),
        "thermal_comfort_rms_degC": _rms_comfort(result["T_in"]),
        "thermal_comfort_max_deviation_degC": _max_temperature_deviation(result["T_in"]),
        "net_energy_received_kwh_per_day": _net_shared_quantity(shared_net),
        "energy_bought_kwh_per_day": bought,
        "energy_sold_kwh_per_day": sold,
        "net_energy_shared_kwh_per_day": net,
        "community_energy_balance_error_kwh_per_day": abs(net),
        "_grid_series": result["grid"].sum(axis=0),
        "_grid_per_building": np.asarray(result["grid"], dtype=float),
        "_energy_cost_per_building": np.asarray(result["elec_b"] + result["gas_b"], dtype=float),
        "_temperature_per_building": np.asarray(result["T_in"], dtype=float),
        "_grid_baseline_total": _grid_import_total(baseline["grid"]),
        "_grid_saving_total": _grid_import_total(baseline["grid"]) - _grid_import_total(result["grid"]),
        "_available_surplus": _available_surplus(result["discharge"], result["p_hp"], result["charge"]),
        "_shared_per_building": _net_from_pairwise_matrix(result["share"]),
        "_p_hp_per_building": np.asarray(result["p_hp"], dtype=float),
        "_charge_per_building": np.asarray(result["charge"], dtype=float),
        "_discharge_per_building": np.asarray(result["discharge"], dtype=float),
        "note": "central market model",
    }


def run_exchange_admm() -> dict:
    pex, _price, _hist, subs = EXC.run_admm()
    grid, gas, t_in, price = _sub_arrays(subs)
    cost = float((price[None, :] * grid * C.dt).sum())
    cost += float((C.gas_price / 100.0 * gas * C.dt).sum())
    discharge = np.array([EXC._val(m, "discharge") for m in subs])
    charge = np.array([EXC._val(m, "charge") for m in subs])
    p_hp = np.array([EXC._val(m, "p_hp") for m in subs])
    baseline = C.solve_and_extract(C.build_model(True, False), "comparison_exchange_baseline")
    shared_net = -np.asarray(pex, dtype=float)
    bought, sold, net = _community_trade_balance(shared_net)
    return {
        "method": "exchange-ADMM",
        "operating_cost_gbp_per_day": cost,
        "peak_grid_import_kw": _grid_peak(grid),
        "peak_grid_import_hour": _grid_peak_hour(grid),
        "grid_import_kwh_per_day": _grid_import_total(grid),
        "grid_saving_kwh_per_day": _grid_import_total(baseline["grid"]) - _grid_import_total(grid),
        "thermal_comfort_rms_degC": _rms_comfort(t_in),
        "thermal_comfort_max_deviation_degC": _max_temperature_deviation(t_in),
        "net_energy_received_kwh_per_day": _net_shared_quantity(shared_net),
        "energy_bought_kwh_per_day": bought,
        "energy_sold_kwh_per_day": sold,
        "net_energy_shared_kwh_per_day": net,
        "community_energy_balance_error_kwh_per_day": abs(net),
        "_grid_series": grid.sum(axis=0),
        "_grid_per_building": np.asarray(grid, dtype=float),
        "_energy_cost_per_building": _building_energy_cost(grid, gas),
        "_temperature_per_building": np.asarray(t_in, dtype=float),
        "_grid_baseline_total": _grid_import_total(baseline["grid"]),
        "_grid_saving_total": _grid_import_total(baseline["grid"]) - _grid_import_total(grid),
        "_available_surplus": _available_surplus(discharge, p_hp, charge),
        # Exchange ADMM defines positive pex as exporting; negate it so positive
        # values consistently mean net energy received, as in the other methods.
        "_shared_per_building": -np.asarray(pex, dtype=float),
        "_p_hp_per_building": np.asarray(p_hp, dtype=float),
        "_charge_per_building": np.asarray(charge, dtype=float),
        "_discharge_per_building": np.asarray(discharge, dtype=float),
        "note": "exchange ADMM pooled sharing",
    }


def _run_bilateral_admm_variant(use_valley: bool) -> dict:
    """Run the bilateral ADMM with or without the valley-filling import-shaping term.

    The model's module-level ``use_congestion`` flag is the valley-filling knob in
    ``admm_bilateral_p2p_valley.py``: setting it to ``True`` adds a convex import
    cost term that shifts flexible demand into lower-price hours. This helper
    exposes that behaviour in the same comparison workflow as the other methods.
    """
    previous = bool(BIL.use_congestion)
    BIL.use_congestion = use_valley
    try:
        subs, _q, _lam, trades, _hist = BIL.run_admm()
    finally:
        BIL.use_congestion = previous

    grid, gas, t_in, price = _sub_arrays(subs)
    cost = float((price[None, :] * grid * C.dt).sum())
    cost += float((C.gas_price / 100.0 * gas * C.dt).sum())
    discharge = np.array([EXC._val(m, "discharge") for m in subs])
    charge = np.array([EXC._val(m, "charge") for m in subs])
    p_hp = np.array([EXC._val(m, "p_hp") for m in subs])
    baseline = C.solve_and_extract(C.build_model(True, False), "comparison_bilateral_baseline")
    shared_net = _net_from_trade_dict(trades, C.n_buildings, C.time_horizon)
    bought, sold, net = _community_trade_balance(shared_net)
    method_name = "bilateral ADMM + valley filling" if use_valley else "bilateral ADMM (no valley fill)"
    return {
        "method": method_name,
        "operating_cost_gbp_per_day": cost,
        "peak_grid_import_kw": _grid_peak(grid),
        "peak_grid_import_hour": _grid_peak_hour(grid),
        "grid_import_kwh_per_day": _grid_import_total(grid),
        "grid_saving_kwh_per_day": _grid_import_total(baseline["grid"]) - _grid_import_total(grid),
        "thermal_comfort_rms_degC": _rms_comfort(t_in),
        "thermal_comfort_max_deviation_degC": _max_temperature_deviation(t_in),
        "net_energy_received_kwh_per_day": _net_shared_quantity(shared_net),
        "energy_bought_kwh_per_day": bought,
        "energy_sold_kwh_per_day": sold,
        "net_energy_shared_kwh_per_day": net,
        "community_energy_balance_error_kwh_per_day": abs(net),
        "_grid_series": grid.sum(axis=0),
        "_grid_per_building": np.asarray(grid, dtype=float),
        "_energy_cost_per_building": _building_energy_cost(grid, gas),
        "_temperature_per_building": np.asarray(t_in, dtype=float),
        "_grid_baseline_total": _grid_import_total(baseline["grid"]),
        "_grid_saving_total": _grid_import_total(baseline["grid"]) - _grid_import_total(grid),
        "_available_surplus": _available_surplus(discharge, p_hp, charge),
        "_shared_per_building": _net_from_trade_dict(trades, C.n_buildings, C.time_horizon),
        "_p_hp_per_building": np.asarray(p_hp, dtype=float),
        "_charge_per_building": np.asarray(charge, dtype=float),
        "_discharge_per_building": np.asarray(discharge, dtype=float),
        "note": (
            "bilateral pairwise P2P sharing" if use_valley else "bilateral pairwise P2P sharing without valley filling"
        ),
    }


def run_bilateral_admm() -> dict:
    """Default bilateral ADMM run used in the comparison plots and summary table."""
    return _run_bilateral_admm_variant(use_valley=True)


def run_bilateral_admm_variant(use_valley: bool) -> dict:
    """Public helper for the valley-filling comparison in the main comparison script."""
    return _run_bilateral_admm_variant(use_valley=use_valley)


def _building_configuration_summary() -> str:
    pv_buildings = np.any(np.asarray(C.PV_B) > 1e-6, axis=1)
    battery_buildings = np.asarray(C.HAS_BATTERY, dtype=bool)
    both = int(np.count_nonzero(pv_buildings & battery_buildings))
    pv_only = int(np.count_nonzero(pv_buildings & ~battery_buildings))
    battery_only = int(np.count_nonzero(~pv_buildings & battery_buildings))
    neither = int(np.count_nonzero(~pv_buildings & ~battery_buildings))
    return (
        f"{C.n_buildings} buildings | PV {int(pv_buildings.sum())} | "
        f"battery {int(battery_buildings.sum())} ({C.CAP.sum():.1f} kWh) | "
        f"both {both}, PV only {pv_only}, battery only {battery_only}, neither {neither}"
    )


def build_building_comparison_table(results: list[dict]) -> pd.DataFrame:
    """Compare each method with the battery-enabled no-sharing baseline per building."""
    baseline = next(result for result in results if result["method"] == "centralised no-sharing")
    baseline_cost = np.asarray(baseline["_energy_cost_per_building"], dtype=float)
    baseline_grid = np.asarray(baseline["_grid_per_building"], dtype=float).sum(axis=1) * C.dt
    pv_present = np.any(np.asarray(C.PV_B, dtype=float) > 1e-6, axis=1)
    battery_capacity = np.asarray(C.CAP, dtype=float)
    has_battery = np.asarray(C.HAS_BATTERY, dtype=bool)
    rows = []

    for result in results:
        method_cost = np.asarray(result["_energy_cost_per_building"], dtype=float)
        method_grid = np.asarray(result["_grid_per_building"], dtype=float).sum(axis=1) * C.dt
        cost_saving = baseline_cost - method_cost
        saving_pct = np.divide(
            100.0 * cost_saving,
            baseline_cost,
            out=np.full_like(cost_saving, np.nan),
            where=np.abs(baseline_cost) > 1e-9,
        )
        signed_net = np.asarray(result["_shared_per_building"], dtype=float)
        temperatures = np.asarray(result["_temperature_per_building"], dtype=float)
        comfort_error = temperatures - C.T_SET[None, :]

        for building in range(C.n_buildings):
            rows.append(
                {
                    "building_id": C.BUILDING_IDS[building],
                    "assets": C.ASSETS[building],
                    "has_pv": bool(pv_present[building]),
                    "has_battery": bool(has_battery[building]),
                    "battery_capacity_kwh": battery_capacity[building],
                    "initial_soc_kwh": float(C.INIT_SOC[building]),
                    "initial_temperature_degC": float(C.INIT_T_IN[building]),
                    "method": result["method"],
                    "baseline_method": baseline["method"],
                    "energy_cost_gbp_per_day": method_cost[building],
                    "baseline_energy_cost_gbp_per_day": baseline_cost[building],
                    "energy_cost_saving_gbp_per_day": cost_saving[building],
                    "energy_cost_saving_pct": saving_pct[building],
                    "grid_import_kwh_per_day": method_grid[building],
                    "baseline_grid_import_kwh_per_day": baseline_grid[building],
                    "grid_import_saving_kwh_per_day": baseline_grid[building] - method_grid[building],
                    "thermal_comfort_rms_degC": float(np.sqrt(np.mean(comfort_error[building] ** 2))),
                    "thermal_comfort_max_deviation_degC": float(np.max(np.abs(comfort_error[building]))),
                    "net_p2p_received_kwh_per_day": float(np.maximum(signed_net[building], 0.0).sum() * C.dt),
                    "net_p2p_exported_kwh_per_day": float(np.maximum(-signed_net[building], 0.0).sum() * C.dt),
                }
            )

    return pd.DataFrame(rows)


def plot_building_cost_savings(building_table: pd.DataFrame, output: Path = BUILDING_SAVINGS_PLOT_FILE) -> None:
    """Plot per-building cost savings versus the battery-enabled no-sharing baseline."""
    method_order = [method for method in METHOD_ORDER if method in building_table["method"].unique()]
    savings = building_table.pivot(
        index="building_id", columns="method", values="energy_cost_saving_gbp_per_day"
    ).reindex(columns=method_order)
    asset_by_building = building_table.drop_duplicates("building_id").set_index("building_id")["assets"]
    labels = [f"{building} ({asset_by_building[building]})" for building in savings.index]
    values = savings.to_numpy(dtype=float)
    limit = max(float(np.nanmax(np.abs(values))), 1e-6)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig, ax = plt.subplots(figsize=(max(9, 1.7 * len(method_order) + 4), max(5, 0.48 * len(labels) + 2)))
    image = ax.imshow(values, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(method_order)))
    ax.set_xticklabels(method_order, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Method")
    ax.set_ylabel("Building and assets")
    ax.set_title(
        "Per-building energy-cost saving vs battery-enabled no-sharing baseline\n"
        "Grid electricity + gas cost; P2P cash settlements excluded",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    for building_index in range(values.shape[0]):
        for method_index in range(values.shape[1]):
            ax.text(
                method_index,
                building_index,
                f"{values[building_index, method_index]:+.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, label="£/day saved (+) or extra cost (-)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")
    print(f"Saved {output.with_suffix('.pdf')}")


def plot_building_cost_comparison(building_table: pd.DataFrame, output: Path = BUILDING_COST_BARS_FILE) -> None:
    """Compare each method's per-building energy cost with the no-sharing baseline."""
    baseline_name = "centralised no-sharing"
    methods = [
        method for method in METHOD_ORDER if method != baseline_name and method in building_table["method"].unique()
    ]
    buildings = building_table.loc[building_table["method"] == baseline_name, "building_id"].tolist()
    baseline_rows = building_table.loc[building_table["method"] == baseline_name].set_index("building_id")
    asset_by_building = baseline_rows["assets"]

    fig, axes = plt.subplots(
        len(methods), 1, figsize=(max(12, 1.1 * len(buildings) + 4), 3.4 * len(methods)), squeeze=False
    )
    x = np.arange(len(buildings))
    width = 0.38

    for axis, method in zip(axes.ravel(), methods):
        method_rows = building_table.loc[building_table["method"] == method].set_index("building_id")
        baseline_cost = baseline_rows.loc[buildings, "baseline_energy_cost_gbp_per_day"].to_numpy(float)
        sharing_cost = method_rows.loc[buildings, "energy_cost_gbp_per_day"].to_numpy(float)
        method_colour = METHOD_COLORS.get(method, "#2e8b6e")
        baseline_bars = axis.bar(x - width / 2, baseline_cost, width, color="#b8bec5", label=baseline_name)
        method_bars = axis.bar(x + width / 2, sharing_cost, width, color=method_colour, label=method)
        axis.set_title(method, loc="left", fontsize=11, fontweight="bold")
        axis.set_ylabel("Energy cost (£/day)")
        axis.set_xticks(x)
        axis.set_xticklabels(
            [f"{building}\n{asset_by_building[building]}" for building in buildings],
            rotation=35,
            ha="right",
        )
        axis.grid(True, axis="y", color="#e6e6e6", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_ylim(bottom=0)
        axis.legend(frameon=False, ncol=2, loc="upper right")
        for bars in (baseline_bars, method_bars):
            for bar in bars:
                axis.annotate(
                    f"{bar.get_height():.2f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    fig.suptitle(
        "Per-building energy cost: sharing method vs centralized no-sharing baseline\n"
        "Grid electricity + gas cost; P2P cash settlements excluded",
        x=0.06,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")
    print(f"Saved {output.with_suffix('.pdf')}")


def plot_comparison(table: pd.DataFrame, output: Path = PLOT_FILE) -> None:
    """Plot the requested community-level comparison metrics."""
    order = {method: index for index, method in enumerate(METHOD_ORDER)}
    table = table.assign(_method_order=table["method"].map(order).fillna(len(order)))
    table = table.sort_values("_method_order").drop(columns="_method_order")
    panels = [
        ("operating_cost_gbp_per_day", "Operating cost (GBP/day)"),
        ("peak_grid_import_kw", "Peak grid import (kW)"),
        ("peak_grid_import_hour", "Time of peak grid import (hour)"),
        ("grid_import_kwh_per_day", "Total grid import (kWh/day)"),
        ("grid_saving_kwh_per_day", "Grid saving vs no sharing (kWh/day)"),
        ("thermal_comfort_rms_degC", "Thermal comfort RMS error (deg C)"),
        ("thermal_comfort_max_deviation_degC", "Maximum temperature deviation (deg C)"),
        ("net_energy_received_kwh_per_day", "Net energy received (kWh/day)"),
        ("net_energy_shared_kwh_per_day", "Community net P2P balance (kWh/day)"),
    ]
    methods = table["method"].tolist()
    colours = [METHOD_COLORS.get(method, "#777777") for method in methods]
    fig, axes = plt.subplots(5, 2, figsize=(12, 18), squeeze=False)

    for axis, (column, title) in zip(axes.ravel(), panels):
        values = table[column].to_numpy(float)
        bars = axis.bar(methods, values, color=colours, width=0.68)
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.set_ylabel("value")
        axis.grid(True, axis="y", color="#e6e6e6", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="x", rotation=25, labelsize=9)
        axis.set_ylim(bottom=0)
        for bar, value in zip(bars, values):
            axis.annotate(
                f"{value:.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    title = f"{pd.Timestamp(C.SHOWCASE_DAY):%A %d %b %Y} - energy-sharing method comparison"
    configuration = _building_configuration_summary()
    fig.suptitle(
        f"{title}\n{configuration}",
        x=0.08,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=160)
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(str(pdf_output), format="pdf")
    print(f"Saved {output}")
    print(f"Saved {pdf_output}")

    html_output = output.with_suffix(".html")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        svg = StringIO()
        fig.savefig(svg, format="svg")
        html_output.write_text(
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{title}</title></head><body>{svg.getvalue()}</body></html>",
            encoding="utf-8",
        )
    else:
        interactive = make_subplots(rows=5, cols=2, subplot_titles=[label for _, label in panels])
        for panel_index, (column, _label) in enumerate(panels):
            row, col = divmod(panel_index, 2)
            for method, colour in zip(methods, colours):
                value = float(table.loc[table["method"] == method, column].iloc[0])
                interactive.add_trace(
                    go.Bar(
                        x=[method],
                        y=[value],
                        name=method,
                        legendgroup=method,
                        showlegend=(panel_index == 0),
                        marker_color=colour,
                        text=[f"{value:.2f}"],
                        textposition="outside",
                        hovertemplate=f"{method}: %{{y:.2f}}<extra></extra>",
                    ),
                    row=row + 1,
                    col=col + 1,
                )
            interactive.update_yaxes(rangemode="tozero", row=row + 1, col=col + 1)
        interactive.update_layout(
            template="plotly_white",
            height=1100,
            barmode="group",
            title=f"{title}<br><sup>{configuration}</sup>",
            legend=dict(orientation="h", y=-0.08),
        )
        interactive.write_html(html_output, include_plotlyjs=True)
    plt.close(fig)
    print(f"Saved {html_output}")


def plot_sharing_surplus_comparison(results: list[dict], output: Path = SURPLUS_PLOT_FILE) -> None:
    """Compare available surplus and net energy received for every method."""
    hours = np.arange(C.time_horizon)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    for result in results:
        colour = METHOD_COLORS.get(result["method"], "#777777")
        axes[0].step(hours, result["_available_surplus"], where="post", color=colour, lw=2, label=result["method"])
        axes[1].step(
            hours,
            _net_shared_per_hour(result["_shared_per_building"]),
            where="post",
            color=colour,
            lw=2,
            label=result["method"],
        )
    axes[0].set_title("Export-capable energy (PV + battery)", loc="left", fontsize=11, fontweight="bold")
    axes[1].set_title("Net energy received", loc="left", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Hour of day")
    for axis in axes:
        axis.set_ylabel("Energy per hour (kWh)")
        axis.set_ylim(bottom=0)
        axis.set_xlim(0, C.time_horizon)
        axis.set_xticks(range(0, C.time_horizon + 1, 3))
        axis.grid(True, axis="y", color="#e6e6e6", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False, ncol=4, loc="upper left")

    fig.suptitle(
        f"{pd.Timestamp(C.SHOWCASE_DAY):%A %d %b %Y} - available surplus versus net energy received",
        x=0.08,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=160)
    plt.close(fig)
    print(f"Saved {output}")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return
    interactive = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Export-capable energy (PV + battery)", "Net energy received"),
    )
    html_hours = list(hours) + [C.time_horizon]
    for result in results:
        colour = METHOD_COLORS.get(result["method"], "#777777")
        for row, key, name in (
            (1, "_available_surplus", "available surplus"),
            (2, "_shared_per_building", "net energy received"),
        ):
            values = (
                list(_net_shared_per_hour(result[key])) if row == 2 else list(np.asarray(result[key], dtype=float))
            )
            interactive.add_trace(
                go.Scatter(
                    x=html_hours,
                    y=values + [values[-1]],
                    name=result["method"],
                    legendgroup=result["method"],
                    showlegend=(row == 1),
                    line=dict(color=colour, width=2.2, shape="hv"),
                    hovertemplate=f"{result['method']} {name}: %{{y:.2f}} kWh<extra></extra>",
                ),
                row=row,
                col=1,
            )
    interactive.update_layout(
        template="plotly_white",
        height=720,
        hovermode="x unified",
        title=f"{C.SHOWCASE_DAY} - sharing surplus versus actual sharing",
        legend=dict(orientation="h", y=-0.12),
    )
    interactive.update_xaxes(title_text="Hour of day", dtick=3, row=2, col=1)
    interactive.update_yaxes(title_text="kWh per hour", rangemode="tozero")
    html_output = output.with_suffix(".html")
    interactive.write_html(html_output, include_plotlyjs=True)
    print(f"Saved {html_output}")


def plot_actual_shared_comparison(results: list[dict], output: Path = DESTINATION_PLOT_FILE) -> None:
    """Compare net P2P energy received and grid import across methods."""
    selected = {
        "centralised no-sharing",
        "rule-based",
        "central",
        "exchange-ADMM",
        "bilateral ADMM (no valley fill)",
        "bilateral ADMM + valley filling",
    }
    results = [result for result in results if result["method"] in selected]
    hours = np.arange(C.time_horizon)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    grid_ax = ax.twinx()

    tariff_bands = (
        (C.IS_LOW, "#5b8fc9", 0.08),
        (C.IS_MEDIUM, "#8b78b5", 0.07),
        (C.IS_HIGH, "#d98a29", 0.10),
    )
    for mask, colour, alpha in tariff_bands:
        for hour in np.flatnonzero(mask):
            ax.axvspan(hour, hour + 1, color=colour, alpha=alpha, lw=0, zorder=0)

    for result in results:
        colour = METHOD_COLORS.get(result["method"], "#777777")
        ax.step(
            hours,
            _net_shared_per_hour(result["_shared_per_building"]),
            where="post",
            color=colour,
            lw=2.3,
            label=result["method"],
        )
        grid_ax.step(
            hours,
            result["_grid_series"],
            where="post",
            color=colour,
            lw=1.5,
            ls="--",
            alpha=0.75,
            label=f"{result['method']} grid import",
        )

    ax.set_title("Net energy received and grid import by method", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Net energy received per hour (kWh)")
    grid_ax.set_ylabel("Community grid import (kW)")
    ax.set_xlim(0, C.time_horizon)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(0, C.time_horizon + 1, 3))
    ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    grid_handles, grid_labels = grid_ax.get_legend_handles_labels()
    ax.legend(handles + grid_handles, labels + grid_labels, frameon=False, ncol=4, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    grid_ax.spines["top"].set_visible(False)
    fig.suptitle(
        f"{pd.Timestamp(C.SHOWCASE_DAY):%A %d %b %Y} - net energy received comparison",
        x=0.08,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=160)
    plt.close(fig)
    print(f"Saved {output}")

    try:
        import plotly.graph_objects as go
    except ImportError:
        return
    html_hours = list(hours) + [C.time_horizon]
    interactive = go.Figure()
    for result in results:
        values = list(_net_shared_per_hour(result["_shared_per_building"]))
        colour = METHOD_COLORS.get(result["method"], "#777777")
        interactive.add_trace(
            go.Scatter(
                x=html_hours,
                y=values + [values[-1]],
                name=result["method"],
                line=dict(color=colour, width=2.4, shape="hv"),
                hovertemplate=f"{result['method']} net received: %{{y:.2f}} kWh<extra></extra>",
            )
        )
        grid_values = list(np.asarray(result["_grid_series"], dtype=float))
        interactive.add_trace(
            go.Scatter(
                x=html_hours,
                y=grid_values + [grid_values[-1]],
                name=f"{result['method']} grid import",
                legendgroup=f"{result['method']} grid",
                line=dict(color=colour, width=1.6, dash="dash", shape="hv"),
                yaxis="y2",
                hovertemplate=f"{result['method']} grid import: %{{y:.2f}} kW<extra></extra>",
            )
        )

    for mask, colour, opacity in (
        (C.IS_LOW, "#5b8fc9", 0.08),
        (C.IS_MEDIUM, "#8b78b5", 0.07),
        (C.IS_HIGH, "#d98a29", 0.10),
    ):
        for hour in np.flatnonzero(mask):
            interactive.add_vrect(
                x0=int(hour), x1=int(hour) + 1, fillcolor=colour, opacity=opacity, line_width=0, layer="below"
            )
    interactive.update_layout(
        template="plotly_white",
        height=500,
        hovermode="x unified",
        title=f"{C.SHOWCASE_DAY} - net energy received and grid import comparison",
        xaxis_title="Hour of day",
        yaxis_title="Net energy received per hour (kWh)",
        yaxis2=dict(title="Community grid import (kW)", overlaying="y", side="right", rangemode="tozero"),
        legend=dict(orientation="h", y=-0.18),
    )
    interactive.update_xaxes(dtick=3)
    interactive.update_yaxes(rangemode="tozero")
    html_output = output.with_suffix(".html")
    interactive.write_html(html_output, include_plotlyjs=True)
    print(f"Saved {html_output}")


def plot_shared_energy_vs_grid_saving(results: list[dict], output: Path = GRID_SAVING_PLOT_FILE) -> None:
    """Plot net energy received against grid saving versus no-sharing baselines."""
    selected = {
        "centralised no-sharing",
        "rule-based",
        "central",
        "exchange-ADMM",
        "bilateral ADMM (no valley fill)",
        "bilateral ADMM + valley filling",
    }
    results = [result for result in results if result["method"] in selected]
    fig, ax = plt.subplots(figsize=(8, 6))
    for result in results:
        colour = METHOD_COLORS.get(result["method"], "#777777")
        ax.scatter(
            result["net_energy_received_kwh_per_day"],
            result["_grid_saving_total"],
            s=130,
            color=colour,
            label=result["method"],
            zorder=3,
        )
        ax.annotate(
            result["method"],
            (result["net_energy_received_kwh_per_day"], result["_grid_saving_total"]),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xlabel("Net energy received (kWh/day)")
    ax.set_ylabel("Grid-import saving vs no-sharing baseline (kWh/day)")
    ax.set_title("Shared energy versus actual grid saving", loc="left", fontsize=12, fontweight="bold")
    ax.grid(True, color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=160)
    plt.close(fig)
    print(f"Saved {output}")


def _plot_building_heatmap(
    results: list[dict],
    key: str,
    output: Path,
    cbar_label: str,
    suptitle_suffix: str,
    cmap: str = "RdYlGn_r",
) -> None:
    """Shared machinery for the per-building x hour heatmaps.

    Rows are buildings, columns within each panel are hours of day, one
    panel per method, color encodes ``result[key]`` (n_buildings x T). All
    panels share a single, zero-centred colour scale so methods are
    directly comparable.
    """
    order = {method: index for index, method in enumerate(METHOD_ORDER)}
    ordered = sorted(results, key=lambda r: order.get(r["method"], len(order)))
    ordered = [r for r in ordered if key in r]
    if not ordered:
        return

    n_methods = len(ordered)
    n_buildings = C.n_buildings
    hours = np.arange(C.time_horizon)

    vmax = max(float(np.max(r[key])) for r in ordered)
    vmin = min(float(np.min(r[key])) for r in ordered)
    norm = TwoSlopeNorm(vmin=min(vmin, -1e-6), vcenter=0.0, vmax=max(vmax, 1e-6))

    fig, axes = plt.subplots(1, n_methods, figsize=(3.1 * n_methods + 1.2, 0.35 * n_buildings + 1.8), sharey=True)
    axes = np.atleast_1d(axes)

    im = None
    for axis, result in zip(axes, ordered):
        mat = result[key]
        im = axis.pcolormesh(
            hours,
            np.arange(n_buildings),
            mat,
            cmap=cmap,
            norm=norm,
            shading="nearest",
            edgecolors="white",
            linewidth=0.4,
        )
        axis.set_title(result["method"], fontsize=10, fontweight="bold")
        axis.set_xlabel("Hour [h]")
        axis.set_xticks(range(0, C.time_horizon + 1, 3))
        axis.set_yticks(np.arange(n_buildings))
        axis.invert_yaxis()

    axes[0].set_yticklabels([f"B{b + 1}" for b in range(n_buildings)])
    axes[0].set_ylabel("Building")

    cbar = fig.colorbar(im, ax=list(axes), pad=0.02)
    cbar.set_label(cbar_label)

    fig.suptitle(
        f"{pd.Timestamp(C.SHOWCASE_DAY):%A %d %b %Y} - {suptitle_suffix}",
        x=0.08,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def plot_building_grid_heatmap(results: list[dict], output: Path = HEATMAP_PLOT_FILE) -> None:
    """Per-building grid import heatmap, one column per method."""
    _plot_building_heatmap(
        results,
        "_grid_per_building",
        output,
        cbar_label="Hourly Grid Import [kW]",
        suptitle_suffix="grid import by building and method",
    )


def plot_building_shared_heatmap(results: list[dict], output: Path = SHARED_HEATMAP_PLOT_FILE) -> None:
    """Per-building net P2P shared-energy heatmap, one column per method.

    Positive = net importer via peer-to-peer sharing that hour, negative =
    net exporter. See ``_net_from_pairwise_matrix`` / ``_net_from_trade_dict``
    for the sign-convention assumptions used to derive this per method --
    verify against your actual modules before trusting the sign.
    """
    _plot_building_heatmap(
        results,
        "_shared_per_building",
        output,
        cbar_label="Hourly Net Energy Shared [kW]",
        suptitle_suffix="net P2P energy shared by building and method",
    )


def plot_building_energy_timeseries(result: dict, output: Path) -> None:
    """One subplot per building: PV, heat pump electricity, grid import, and
    net P2P energy shared on the left axis (kW), battery SOC on a twin
    right axis (kWh, approximated -- see ``_battery_soc``).
    """
    required = ("_p_hp_per_building", "_charge_per_building", "_discharge_per_building", "_grid_per_building")
    if not all(k in result for k in required):
        return  # this method didn't expose the arrays this plot needs

    n_buildings = C.n_buildings
    hours = np.arange(C.time_horizon)
    pv = np.asarray(C.PV_B, dtype=float)
    p_hp = result["_p_hp_per_building"]
    grid = result["_grid_per_building"]
    shared = result.get("_shared_per_building", np.zeros((n_buildings, C.time_horizon)))
    soc = _battery_soc(result["_charge_per_building"], result["_discharge_per_building"], C.CAP, C.INIT_SOC)

    fig, axes = plt.subplots(n_buildings, 1, figsize=(11, 2.3 * n_buildings), sharex=True)
    axes = np.atleast_1d(axes)

    series_style = [
        (pv, "PV generation", "#e8b923", "-"),
        (p_hp, "Heat pump elec.", "#8b3fa0", "-"),
        (grid, "Grid import", "#3b6bb0", "-"),
        (shared, "Net P2P shared", "#c0392b", "--"),
    ]

    for b, axis in enumerate(axes):
        for values, label, colour, style in series_style:
            axis.plot(hours, values[b], color=colour, linestyle=style, lw=1.6, label=label)
        soc_ax = axis.twinx()
        soc_ax.plot(hours, soc[b], color="#2e8b6e", lw=1.4, ls=":", label="Battery SOC")
        soc_ax.set_ylabel("SOC [kWh]", fontsize=8)
        soc_ax.set_ylim(bottom=0)

        axis.set_title(f"Building {b + 1}", loc="left", fontsize=10, fontweight="bold")
        axis.set_ylabel("Power [kW]", fontsize=8)
        axis.grid(True, axis="y", color="#e6e6e6", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)

        if b == 0:
            handles, labels = axis.get_legend_handles_labels()
            soc_handles, soc_labels = soc_ax.get_legend_handles_labels()
            axis.legend(
                handles + soc_handles,
                labels + soc_labels,
                frameon=False,
                ncol=5,
                loc="upper left",
                bbox_to_anchor=(0, 1.35),
                fontsize=8,
            )

    axes[-1].set_xlabel("Hour of day")
    axes[-1].set_xticks(range(0, C.time_horizon + 1, 3))
    axes[-1].set_xlim(0, C.time_horizon)

    fig.suptitle(
        f"{pd.Timestamp(C.SHOWCASE_DAY):%A %d %b %Y} - {result['method']}: per-building electricity balance",
        x=0.06,
        ha="left",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-admm", action="store_true", help="skip both ADMM runs")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()

    C._require_solver()
    rows = [run_no_sharing(), run_rule_based(), run_central()]
    if not args.no_admm:
        rows.extend([run_exchange_admm(), run_bilateral_admm_variant(False), run_bilateral_admm()])

    table = pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} for row in rows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    building_table = build_building_comparison_table(rows)
    building_comparison_file = args.output.with_name(BUILDING_COMPARISON_FILE.name)
    building_savings_plot = args.output.with_name(BUILDING_SAVINGS_PLOT_FILE.name)
    building_table.to_csv(building_comparison_file, index=False)
    print(f"Saved {building_comparison_file}")
    plot_building_cost_savings(building_table, building_savings_plot)
    plot_building_cost_comparison(building_table, args.output.with_name(BUILDING_COST_BARS_FILE.name))
    plot_comparison(table, args.output.with_suffix(".png"))
    plot_sharing_surplus_comparison(rows, args.output.with_name("22_sharing_surplus_comparison.png"))
    plot_actual_shared_comparison(rows, args.output.with_name("25_sharing_destinations_comparison.png"))
    plot_shared_energy_vs_grid_saving(rows, args.output.with_name("26_shared_energy_vs_grid_saving.png"))
    plot_building_grid_heatmap(rows, args.output.with_name("23_building_grid_import_heatmap.png"))
    plot_building_shared_heatmap(rows, args.output.with_name("24_building_shared_energy_heatmap.png"))
    for row in rows:
        slug = row["method"].lower().replace(" ", "_")
        plot_building_energy_timeseries(row, args.output.with_name(f"27_building_timeseries_{slug}.png"))
    print(f"Showcase day: {C.SHOWCASE_DAY} | buildings: {C.n_buildings}")
    print(table.round(3).to_string(index=False))
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
