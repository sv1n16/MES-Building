"""Benchmark bilateral ADMM tolerances against the centralized sharing solution.

Run from this directory with:

    py -3.13 admm_bilateral_tolerance_sweep.py

The script writes a per-tolerance summary CSV and a per-building deviation CSV
under ``plots/MES ADMM Optimisation Community Size 10`` by default.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

import admm_bilateral_p2p as BIL
import central_optimisation_showcase as C

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "plots" / "MES ADMM Optimisation Community Size 10"
DEFAULT_TOLERANCES = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)


def _parse_tolerances(value: str) -> list[float]:
    try:
        tolerances = [float(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("tolerances must be comma-separated positive numbers") from error
    if not tolerances or any(not np.isfinite(item) or item <= 0 for item in tolerances):
        raise argparse.ArgumentTypeError("tolerances must be comma-separated positive finite numbers")
    return tolerances


def _admm_building_costs(subs, trades: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ADMM costs, grid energy, comfort RMS, and temperature trajectories."""

    def series(model, name: str) -> np.ndarray:
        variable = getattr(model, name)
        return np.array([pyo.value(variable[t]) for t in range(C.time_horizon)], dtype=float)

    electricity_cost = np.zeros(C.n_buildings)
    gas_cost = np.zeros(C.n_buildings)
    grid_energy = np.zeros(C.n_buildings)
    temperatures = np.zeros((C.n_buildings, C.time_horizon))

    for building, model in enumerate(subs):
        grid = series(model, "p_el")
        gas = series(model, "gas")
        temperatures[building] = series(model, "T_in")
        electricity_cost[building] = float((C.PRICE * grid * C.dt).sum())
        gas_cost[building] = float((C.gas_price / 100.0 * gas * C.dt).sum())
        grid_energy[building] = float(grid.sum() * C.dt)

    settlement = np.zeros(C.n_buildings)
    settlement_price = BIL.FEE_FRAC * C.PRICE
    for lower, upper in BIL.PAIRS:
        flow = np.asarray(trades[(lower, upper)], dtype=float)
        payment = settlement_price * flow * C.dt
        settlement[lower] -= float(payment.sum())
        settlement[upper] += float(payment.sum())

    comfort_error = temperatures - C.T_SET[None, :]
    comfort_rms = np.sqrt(np.mean(comfort_error**2, axis=1))
    return electricity_cost + gas_cost + settlement, grid_energy, comfort_rms, temperatures


def _admm_shared_energy(trades: dict) -> tuple[float, np.ndarray, np.ndarray, float, float, float]:
    """Return gross energy transferred plus community sold/bought/net totals."""
    sent = np.zeros(C.n_buildings)
    received = np.zeros(C.n_buildings)
    sold = 0.0
    bought = 0.0
    for (lower, upper), flow in trades.items():
        flow = np.asarray(flow, dtype=float)
        lower_to_upper = np.maximum(flow, 0.0).sum() * C.dt
        upper_to_lower = np.maximum(-flow, 0.0).sum() * C.dt
        sent[lower] += lower_to_upper
        received[upper] += lower_to_upper
        sent[upper] += upper_to_lower
        received[lower] += upper_to_lower
        sold += lower_to_upper + upper_to_lower
        bought += lower_to_upper + upper_to_lower
    # For a paired settlement, the community supply and demand are the same
    # physical quantity; the signed net balance is therefore zero.
    community_net = bought - sold
    return float(sent.sum()), sent, received, float(sold), float(bought), community_net


def run_tolerance_sweep(tolerances: list[float], max_iterations: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run each tolerance from a fresh ADMM state and compare with one central solve."""
    C._require_solver()
    central = C.solve_and_extract(
        C.build_model(True, True, fee_frac=BIL.FEE_FRAC, fee_mode="market"),
        "tolerance_sweep_central_reference",
        fee_frac=BIL.FEE_FRAC,
        fee_mode="market",
    )

    original_settings = (BIL.EPS_PRIMAL, BIL.EPS_DUAL, BIL.MAX_ITERS)
    summary_rows = []
    building_rows = []
    BIL.MAX_ITERS = min(max_iterations, 1000)
    central_share = np.asarray(central["share"], dtype=float)
    central_sent = central_share.sum(axis=(1, 2)) * C.dt
    central_received = central_share.sum(axis=(0, 2)) * C.dt
    central_shared_energy = float(central_sent.sum())

    try:
        for tolerance in tolerances:
            BIL.EPS_PRIMAL = tolerance
            BIL.EPS_DUAL = tolerance
            print(f"\n=== Bilateral ADMM tolerance {tolerance:g}; max iterations {BIL.MAX_ITERS} ===")
            start_time = time.perf_counter()
            subs, _q, _lam, trades, history = BIL.run_admm()
            elapsed_seconds = time.perf_counter() - start_time

            if history.size == 0:
                raise RuntimeError("ADMM returned no iteration history")
            final = history[-1]
            iterations = int(final[0]) + 1
            primal_rms = float(final[3])
            dual_rms = float(final[4])
            converged = primal_rms <= tolerance and dual_rms <= tolerance

            admm_cost, admm_grid_energy, admm_comfort_rms, _temperatures = _admm_building_costs(subs, trades)
            admm_shared_energy, admm_sent, admm_received, sold_energy, bought_energy, net_balance = (
                _admm_shared_energy(trades)
            )
            central_cost = np.asarray(central["cost_b"], dtype=float)
            central_grid_energy = np.asarray(central["grid"], dtype=float).sum(axis=1) * C.dt
            central_comfort_rms = np.sqrt(
                np.mean((np.asarray(central["T_in"], dtype=float) - C.T_SET[None, :]) ** 2, axis=1)
            )
            cost_deviation = admm_cost - central_cost
            abs_cost_deviation = np.abs(cost_deviation)
            relative_cost_deviation = cost_deviation / np.maximum(np.abs(central_cost), 1e-3)
            grid_deviation = admm_grid_energy - central_grid_energy
            comfort_deviation = admm_comfort_rms - central_comfort_rms

            summary_rows.append(
                {
                    "tolerance": tolerance,
                    "max_iterations": BIL.MAX_ITERS,
                    "iterations": iterations,
                    "converged_to_tolerance": converged,
                    "elapsed_seconds": elapsed_seconds,
                    "final_primal_residual_rms_kw": primal_rms,
                    "final_dual_residual_rms_kw": dual_rms,
                    "central_energy_shared_kwh_per_day": central_shared_energy,
                    "admm_energy_shared_kwh_per_day": admm_shared_energy,
                    "energy_shared_deviation_kwh_per_day": admm_shared_energy - central_shared_energy,
                    "admm_energy_sold_kwh_per_day": sold_energy,
                    "admm_energy_bought_kwh_per_day": bought_energy,
                    "admm_net_energy_shared_kwh_per_day": net_balance,
                    "admm_community_balance_error_kwh_per_day": abs(net_balance),
                    "central_community_energy_cost_gbp_per_day": float(central_cost.sum()),
                    "admm_community_energy_cost_gbp_per_day": float(admm_cost.sum()),
                    "community_energy_cost_deviation_gbp_per_day": float(admm_cost.sum() - central_cost.sum()),
                    "mean_abs_building_cost_deviation_gbp_per_day": float(abs_cost_deviation.mean()),
                    "max_abs_building_cost_deviation_gbp_per_day": float(abs_cost_deviation.max()),
                    "building_cost_deviation_rms_pct": float(100.0 * np.sqrt(np.mean(relative_cost_deviation**2))),
                    "max_abs_grid_import_deviation_kwh_per_day": float(np.abs(grid_deviation).max()),
                    "mean_abs_comfort_rms_deviation_degC": float(np.abs(comfort_deviation).mean()),
                }
            )

            for building, building_id in enumerate(C.BUILDING_IDS):
                building_rows.append(
                    {
                        "tolerance": tolerance,
                        "iterations": iterations,
                        "elapsed_seconds": elapsed_seconds,
                        "final_primal_residual_rms_kw": primal_rms,
                        "final_dual_residual_rms_kw": dual_rms,
                        "central_energy_shared_kwh_per_day": central_shared_energy,
                        "admm_energy_shared_kwh_per_day": admm_shared_energy,
                        "building_id": building_id,
                        "assets": C.ASSETS[building],
                        "central_energy_cost_gbp_per_day": central_cost[building],
                        "admm_energy_cost_gbp_per_day": admm_cost[building],
                        "cost_deviation_gbp_per_day": cost_deviation[building],
                        "absolute_cost_deviation_gbp_per_day": abs_cost_deviation[building],
                        "cost_deviation_pct": 100.0 * relative_cost_deviation[building],
                        "central_grid_import_kwh_per_day": central_grid_energy[building],
                        "admm_grid_import_kwh_per_day": admm_grid_energy[building],
                        "grid_import_deviation_kwh_per_day": grid_deviation[building],
                        "central_comfort_rms_degC": central_comfort_rms[building],
                        "admm_comfort_rms_degC": admm_comfort_rms[building],
                        "comfort_rms_deviation_degC": comfort_deviation[building],
                        "central_energy_sent_kwh_per_day": central_sent[building],
                        "admm_energy_sent_kwh_per_day": admm_sent[building],
                        "energy_sent_deviation_kwh_per_day": admm_sent[building] - central_sent[building],
                        "central_energy_received_kwh_per_day": central_received[building],
                        "admm_energy_received_kwh_per_day": admm_received[building],
                        "energy_received_deviation_kwh_per_day": admm_received[building] - central_received[building],
                    }
                )

            print(
                f"Completed {iterations} iterations in {elapsed_seconds:.2f}s; "
                f"final primal={primal_rms:.3g} kW, dual={dual_rms:.3g} kW"
            )
    finally:
        BIL.EPS_PRIMAL, BIL.EPS_DUAL, BIL.MAX_ITERS = original_settings

    return pd.DataFrame(summary_rows), pd.DataFrame(building_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tolerances",
        type=_parse_tolerances,
        default=list(DEFAULT_TOLERANCES),
        help="comma-separated shared primal/dual RMS tolerances (default: 1e-2,1e-3,1e-4,1e-5,1e-6)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="ADMM iteration cap (maximum allowed: 1000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "31_admm_tolerance_sweep_summary.csv",
        help="summary CSV path; per-building detail is written beside it",
    )
    args = parser.parse_args()
    if args.max_iterations < 1:
        parser.error("--max-iterations must be at least 1")

    summary, building_details = run_tolerance_sweep(args.tolerances, min(args.max_iterations, 1000))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    details_path = args.output.with_name("32_admm_tolerance_sweep_per_building.csv")
    summary.to_csv(args.output, index=False)
    building_details.to_csv(details_path, index=False)
    print(f"\nSaved summary: {args.output}")
    print(f"Saved per-building deviations: {details_path}")
    print(summary.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
