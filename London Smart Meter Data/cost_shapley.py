"""Allocate community operating-cost savings to buildings with Shapley values.

The coalition value is the coalition's no-sharing electricity and gas cost
minus its optimized shared-coalition electricity and gas cost. The default
estimates Shapley values by random permutations; ``--exact`` evaluates all
coalitions and computes exact values. Coalition solves are checkpointed so a
later run can reuse them.

Run from this directory:

    python cost_shapley.py --permutations 100
    python cost_shapley.py --exact
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import central_optimisation_showcase as C

OUTPUT_DIR = Path(__file__).parent / "plots" / "MES ADMM Optimisation Community Size 10"
DEFAULT_CSV = OUTPUT_DIR / "34_cost_shapley_values.csv"
DEFAULT_CACHE = OUTPUT_DIR / "34_shapley_coalition_costs.json"
DEFAULT_PLOT = OUTPUT_DIR / "34_cost_shapley_values.png"

_BUILDING_CONSTRAINTS = (
    "charge_max",
    "discharge_max",
    "soc_balance",
    "no_charge_at_start",
    "no_discharge_at_start",
    "heat_pump_output",
    "q_heat_constraint",
    "cop_calculation",
    "boiler_max",
    "gas_consumption_calc",
    "electricity_balance",
    "heat_balance",
    "thermal_dynamics",
    "energy_export_limit",
    "energy_import_limit",
)
_BUILDING_VARIABLES = (
    "charge",
    "discharge",
    "soc",
    "charging_state",
    "p_el_vars",
    "curtail",
    "p_hp",
    "q_heat_vars",
    "cop",
    "f",
    "q_heat",
    "gas_consumption",
    "q_boiler_vars",
    "T_in",
)


def _signature() -> str:
    digest = hashlib.sha256()
    for array in (C.LOAD, C.PV_B, C.PRICE, C.T_OUT, C.T_SET, C.CAP, C.MAXPOW, C.INIT_SOC, C.INIT_T_IN):
        values = np.ascontiguousarray(array)
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    settings = {
        "showcase_day": str(C.SHOWCASE_DAY),
        "building_ids": [str(value) for value in C.BUILDING_IDS],
        "fee_frac": float(C.SHARE_TRADE_FEE_FRAC),
        "fee_mode": "market",
        "use_battery": True,
        "central_relax_binaries": bool(C.RELAX_CENTRAL_BINARIES),
        "central_model_hash": hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest(),
    }
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _exclude_nonmembers(model, coalition: set[int]) -> None:
    """Disconnect and disable buildings outside one coalition in the central model."""
    outside = set(range(C.n_buildings)) - coalition
    for building in outside:
        for name in _BUILDING_CONSTRAINTS:
            constraint = getattr(model, name)
            for hour in range(C.time_horizon):
                index = (building, hour)
                if index in constraint:
                    constraint[index].deactivate()

        for name in _BUILDING_VARIABLES:
            variable = getattr(model, name)
            for hour in range(C.time_horizon):
                if name == "T_in":
                    variable[building, hour].fix(float(C.INIT_T_IN[building]))
                elif name == "cop":
                    variable[building, hour].fix(2.2)
                else:
                    variable[building, hour].fix(0.0)

    for sender in range(C.n_buildings):
        for receiver in range(C.n_buildings):
            if sender not in coalition or receiver not in coalition:
                for hour in range(C.time_horizon):
                    model.energy_share[sender, receiver, hour].fix(0.0)


def _load_checkpoint(path: Path, signature: str) -> tuple[np.ndarray | None, dict[str, float]]:
    if not path.is_file():
        return None, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != 1 or payload.get("signature") != signature:
        raise ValueError(
            f"Coalition cache {path} does not match the current dataset/model. "
            "Choose another --cache path or remove the stale cache before calculating."
        )
    baseline = payload.get("baseline_cost_per_building")
    baseline_array = None if baseline is None else np.asarray(baseline, dtype=float)
    coalition_costs = {str(mask): float(cost) for mask, cost in payload.get("coalition_costs", {}).items()}
    return baseline_array, coalition_costs


def _save_checkpoint(path: Path, signature: str, baseline: np.ndarray, coalition_costs: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "signature": signature,
        "baseline_cost_per_building": baseline.tolist(),
        "coalition_costs": coalition_costs,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def _prepare_cache(path: Path) -> tuple[str, np.ndarray, dict[str, float]]:
    signature = _signature()
    baseline, coalition_costs = _load_checkpoint(path, signature)
    if baseline is None:
        baseline_result = C.solve_and_extract(C.build_model(True, False), "shapley_no_sharing_baseline")
        baseline = np.asarray(baseline_result["elec_b"] + baseline_result["gas_b"], dtype=float)
        _save_checkpoint(path, signature, baseline, coalition_costs)
    elif baseline.shape != (C.n_buildings,):
        raise ValueError(f"Coalition cache has an invalid baseline vector shape: {baseline.shape}")
    coalition_costs.setdefault("0", 0.0)
    return signature, baseline, coalition_costs


def _coalition_operating_cost(
    mask: int,
    cache_path: Path,
    signature: str,
    baseline: np.ndarray,
    coalition_costs: dict[str, float],
) -> float:
    key = str(mask)
    if key in coalition_costs:
        return coalition_costs[key]
    coalition = {building for building in range(C.n_buildings) if mask & (1 << building)}
    model = C.build_model(True, True, fee_frac=C.SHARE_TRADE_FEE_FRAC, fee_mode="market")
    _exclude_nonmembers(model, coalition)
    result = C.solve_and_extract(
        model,
        f"shapley_coalition_{mask:0{C.n_buildings}b}",
        fee_frac=C.SHARE_TRADE_FEE_FRAC,
        fee_mode="market",
    )
    # Market settlements are transfers within the coalition, not community cost.
    cost = float(np.asarray(result["elec_b"] + result["gas_b"], dtype=float)[list(coalition)].sum())
    coalition_costs[key] = cost
    _save_checkpoint(cache_path, signature, baseline, coalition_costs)
    return cost


def _coalition_saving(mask: int, baseline: np.ndarray, coalition_costs: dict[str, float]) -> float:
    members = [building for building in range(C.n_buildings) if mask & (1 << building)]
    return float(baseline[members].sum() - coalition_costs[str(mask)]) if members else 0.0


def _estimate_by_permutations(
    permutations: int,
    seed: int,
    baseline: np.ndarray,
    coalition_costs: dict[str, float],
    cache_path: Path,
    signature: str,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    marginal_values = np.zeros((permutations, C.n_buildings), dtype=float)
    for sample in range(permutations):
        order = rng.permutation(C.n_buildings)
        mask = 0
        previous_saving = 0.0
        for building in order:
            next_mask = mask | (1 << int(building))
            _coalition_operating_cost(next_mask, cache_path, signature, baseline, coalition_costs)
            saving = _coalition_saving(next_mask, baseline, coalition_costs)
            marginal_values[sample, building] = saving - previous_saving
            mask = next_mask
            previous_saving = saving
        print(f"Completed permutation {sample + 1}/{permutations}")
    values = marginal_values.mean(axis=0)
    errors = (
        marginal_values.std(axis=0, ddof=1) / math.sqrt(permutations)
        if permutations > 1
        else np.full(C.n_buildings, np.nan)
    )
    return values, errors


def _estimate_exact(
    baseline: np.ndarray,
    coalition_costs: dict[str, float],
    cache_path: Path,
    signature: str,
) -> tuple[np.ndarray, np.ndarray]:
    n = C.n_buildings
    full_mask = (1 << n) - 1
    for mask in range(full_mask + 1):
        _coalition_operating_cost(mask, cache_path, signature, baseline, coalition_costs)
        if mask % 32 == 0 or mask == full_mask:
            print(f"Evaluated coalition mask {mask}/{full_mask}")

    values = np.zeros(n, dtype=float)
    denominator = math.factorial(n)
    for building in range(n):
        bit = 1 << building
        for mask in range(full_mask + 1):
            if mask & bit:
                continue
            size = bin(mask).count("1")
            weight = math.factorial(size) * math.factorial(n - size - 1) / denominator
            values[building] += weight * (
                _coalition_saving(mask | bit, baseline, coalition_costs)
                - _coalition_saving(mask, baseline, coalition_costs)
            )
    return values, np.zeros(n, dtype=float)


def _write_outputs(
    csv_path: Path,
    plot_path: Path,
    baseline: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    coalition_costs: dict[str, float],
    method: str,
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    n = C.n_buildings
    full_mask = (1 << n) - 1
    total_saving = _coalition_saving(full_mask, baseline, coalition_costs)
    if not np.isclose(values.sum(), total_saving, rtol=1e-7, atol=1e-6):
        raise RuntimeError("Shapley cost-savings allocations do not sum to the full-community saving.")
    rows = []
    for building in range(n):
        rows.append(
            {
                "building_id": C.BUILDING_IDS[building],
                "assets": C.ASSETS[building],
                "baseline_cost_gbp_per_day": baseline[building],
                "shapley_operating_cost_gbp_per_day": baseline[building] - values[building],
                "shapley_cost_saving_gbp_per_day": values[building],
                "shapley_standard_error_gbp_per_day": errors[building],
                "share_of_total_saving_pct": (
                    100.0 * values[building] / total_saving if abs(total_saving) > 1e-9 else np.nan
                ),
                "total_community_saving_gbp_per_day": total_saving,
                "estimation": method,
                "permutations": permutations if method == "permutation sampling" else np.nan,
                "random_seed": seed if method == "permutation sampling" else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    coalition_csv = csv_path.with_name(f"{csv_path.stem}_coalition_costs.csv")
    pd.DataFrame(
        [
            {
                "coalition_mask": int(mask),
                "coalition_size": bin(int(mask)).count("1"),
                "members": ";".join(C.BUILDING_IDS[b] for b in range(n) if int(mask) & (1 << b)),
                "operating_cost_gbp_per_day": cost,
                "no_sharing_cost_gbp_per_day": baseline[[b for b in range(n) if int(mask) & (1 << b)]].sum(),
                "cost_saving_gbp_per_day": _coalition_saving(int(mask), baseline, coalition_costs),
            }
            for mask, cost in coalition_costs.items()
        ]
    ).sort_values("coalition_mask").to_csv(coalition_csv, index=False)

    ordered = table.sort_values("shapley_cost_saving_gbp_per_day")
    pv_peak = np.asarray(C.PV_B, dtype=float).max(axis=1)
    building_labels = []
    for building_id in ordered["building_id"]:
        building = C.BUILDING_IDS.index(building_id)
        pv_label = f"PV {pv_peak[building]:.1f} kW peak" if pv_peak[building] > 1e-6 else "no PV"
        battery_label = f"battery {C.CAP[building]:.1f} kWh" if C.HAS_BATTERY[building] else "no battery"
        building_labels.append(f"{building_id} | {pv_label} | {battery_label}")

    fig, axis = plt.subplots(figsize=(14, max(5, 0.55 * n + 2)))
    colors = ["#c0392b" if value < 0 else "#2e8b6e" for value in ordered["shapley_cost_saving_gbp_per_day"]]
    axis.barh(building_labels, ordered["shapley_cost_saving_gbp_per_day"], color=colors)
    axis.axvline(0, color="#555555", lw=0.8)
    axis.set_xlabel("Allocated operating-cost saving (GBP/day)")
    axis.set_ylabel("Building")
    axis.set_title(f"Cost-savings Shapley values ({method})", loc="left", fontsize=12, fontweight="bold")
    axis.grid(True, axis="x", color="#e6e6e6", linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {csv_path}")
    print(f"Saved {coalition_csv}")
    print(f"Saved {plot_path}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--exact", action="store_true", help="evaluate all coalitions for exact Shapley values")
    mode.add_argument("--permutations", type=int, default=100, help="sampled random orderings (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for permutation sampling")
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV, help="Shapley results CSV")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="checkpoint file for coalition costs")
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT, help="Shapley bar chart output")
    args = parser.parse_args()
    if not args.exact and args.permutations < 1:
        parser.error("--permutations must be at least 1")

    C._require_solver()
    signature, baseline, coalition_costs = _prepare_cache(args.cache)
    if args.exact:
        values, errors = _estimate_exact(baseline, coalition_costs, args.cache, signature)
        method = "exact"
    else:
        values, errors = _estimate_by_permutations(
            args.permutations, args.seed, baseline, coalition_costs, args.cache, signature
        )
        method = "permutation sampling"

    table = _write_outputs(
        args.output,
        args.plot,
        baseline,
        values,
        errors,
        coalition_costs,
        method,
        args.permutations,
        args.seed,
    )
    total_saving = _coalition_saving((1 << C.n_buildings) - 1, baseline, coalition_costs)
    print(f"Total community operating-cost saving: £{total_saving:.4f}/day")
    print(f"Sum of building Shapley values: £{table['shapley_cost_saving_gbp_per_day'].sum():.4f}/day")
    print(
        table[
            [
                "building_id",
                "shapley_operating_cost_gbp_per_day",
                "shapley_cost_saving_gbp_per_day",
                "shapley_standard_error_gbp_per_day",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
