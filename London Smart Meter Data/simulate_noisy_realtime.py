"""Noisy real-time simulation using the bilateral ADMM schedule.

Examples:
    python simulate_noisy_realtime.py
    python simulate_noisy_realtime.py --load-noise 0.08 --pv-noise 0.30
    python simulate_noisy_realtime.py --pace-seconds 1
"""

import argparse
import csv
import hashlib
import time
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pyomo.environ as pyo

import admm_bilateral_p2p_valley as optimiser

_SCHEDULE_VARIABLES = (
    "charge", "discharge", "soc", "cs", "p_el", "curtail", "p_hp",
    "f", "q_boiler", "gas", "T_in",
)
_TRADE_VARIABLES = ("q", "qpos", "qneg")
SCENARIO_FILE_NAME = "shared_noise_scenarios.npz"
DEFAULT_LOAD_NOISE = 0.20  # std of multiplicative load error per building and hour


def simulate(
    schedule,
    *,
    load_noise,
    pv_noise,
    pace_seconds,
    actual_load=None,
    actual_pv=None,
    actual_t_out=None,
    step_timings=None,
):
    """Simulate noisy load/PV/outdoor temperature against the fixed schedule.

    Heat-pump electrical power and boiler output stay at their planned values;
    the heat-pump heat output follows the actual COP and the indoor temperature
    follows the optimiser's thermal model driven by the actual outdoor temperature.
    """
    if load_noise < 0 or pv_noise < 0 or pace_seconds < 0:
        raise ValueError("Noise levels and pace must be non-negative.")

    C = optimiser.C
    rng = np.random.default_rng()
    planned = schedule["variables"]
    planned_trades = schedule["trades"]
    planned_trade_variables = schedule["trade_variables"]

    if actual_load is None:
        actual_load = np.maximum(
            0.0,
            C.LOAD * (1.0 + rng.normal(0.0, load_noise, size=C.LOAD.shape)),
        )
    if actual_pv is None:
        actual_pv = np.maximum(
            0.0,
            C.PV_B * (1.0 + rng.normal(0.0, pv_noise, size=C.PV_B.shape)),
        )
    actual_load = np.asarray(actual_load, dtype=float)
    actual_pv = np.asarray(actual_pv, dtype=float)
    actual_t_out = np.asarray(C.T_OUT if actual_t_out is None else actual_t_out, dtype=float)
    actual_cop = C.cop_base + 0.01 * (actual_t_out - C.T_ref)
    expected_shape = (optimiser.N, optimiser.T)
    if actual_load.shape != expected_shape or actual_pv.shape != expected_shape:
        raise ValueError(f"Actual load and PV profiles must have shape {expected_shape}.")

    soc = np.asarray(C.INIT_SOC, dtype=float).copy()
    indoor_temperature = np.asarray(C.INIT_T_IN, dtype=float).copy()
    records = []

    if step_timings is None:
        step_timings = []
    for t in range(optimiser.T):
        step_started = time.perf_counter()
        N = optimiser.N
        load_t = actual_load[:, t]
        pv_t = actual_pv[:, t]
        hp_t = np.array([float(planned["p_hp"][b, t]) for b in range(N)])
        demand = load_t + hp_t
        pv_to_load = np.minimum(pv_t, demand)

        # 1. Battery follows its planned setpoints (grid charging allowed, as in
        #    the plan), clipped only by power and the actual state of charge.
        #    The plan can charge and discharge in the same interval (relaxed
        #    binary), so the SOC limits are checked on the net change and only
        #    the excess is trimmed. Matches the optimiser's no-charge/no-discharge
        #    constraint at t=0.
        charge = np.zeros(N)
        discharge = np.zeros(N)
        for b in range(N):
            if C.HAS_BATTERY[b] and t > 0:
                charge[b] = min(max(float(planned["charge"][b, t]), 0.0), float(C.MAXPOW[b]))
                discharge[b] = min(max(float(planned["discharge"][b, t]), 0.0), float(C.MAXPOW[b]))
                soc_next = soc[b] + (C.eta_charge * charge[b] - discharge[b] / C.eta_discharge) * optimiser.DT
                if soc_next > float(C.CAP[b]):
                    charge[b] -= min(charge[b], (soc_next - float(C.CAP[b])) / (C.eta_charge * optimiser.DT))
                elif soc_next < 0.0:
                    discharge[b] -= min(discharge[b], -soc_next * C.eta_discharge / optimiser.DT)

        # 2. Planned P2P trades are honoured up to what is physically possible
        #    with the actual values, using the optimiser's own gross limits:
        #    seller exports <= its PV + battery discharge,
        #    buyer imports  <= its load + heat pump + battery charging.
        export_room = pv_t + discharge
        import_room = demand + charge
        p2p_import = np.zeros(N)
        p2p_export = np.zeros(N)
        actual_trades = np.zeros((N, N), dtype=float)
        for lo, hi in optimiser.PAIRS:
            planned_trade = float(planned_trades[lo, hi, t])
            seller, buyer = (lo, hi) if planned_trade >= 0 else (hi, lo)
            realised = max(min(abs(planned_trade), export_room[seller], import_room[buyer]), 0.0)
            if realised > 0:
                export_room[seller] -= realised
                import_room[buyer] -= realised
                p2p_export[seller] += realised
                p2p_import[buyer] += realised
                actual_trades[seller, buyer] = realised
                actual_trades[buyer, seller] = -realised

        # 3. The grid absorbs the remaining forecast error. Surplus that would
        #    otherwise be curtailed first tops up the battery if it has room.
        charge_correction = np.zeros(N)
        discharge_correction = np.zeros(N)
        grid_import = np.zeros(N)
        pv_surplus = np.zeros(N)
        for b in range(N):
            net = demand[b] + charge[b] + p2p_export[b] - pv_t[b] - discharge[b] - p2p_import[b]
            if net >= 0:
                grid_import[b] = net
            else:
                surplus = -net
                if C.HAS_BATTERY[b] and t > 0:
                    soc_after_plan = soc[b] + (C.eta_charge * charge[b] - discharge[b] / C.eta_discharge) * optimiser.DT
                    headroom = min(
                        float(C.MAXPOW[b]) - charge[b],
                        max((float(C.CAP[b]) - soc_after_plan) / (C.eta_charge * optimiser.DT), 0.0),
                    )
                    charge_correction[b] = min(surplus, max(headroom, 0.0))
                    charge[b] += charge_correction[b]
                    surplus -= charge_correction[b]
                pv_surplus[b] = surplus
            soc[b] += (C.eta_charge * charge[b] - discharge[b] / C.eta_discharge) * optimiser.DT

        # Thermal state with the planned heating and the actual outdoor temperature
        # (same recursion as the optimiser; T_in at t=0 is the initial state).
        hp_heat = actual_cop[t] * hp_t
        if t > 0:
            for b in range(N):
                indoor_temperature[b] += optimiser.DT / 10 * (
                    hp_heat[b] + float(planned["q_boiler"][b, t])
                    - 0.5 * (indoor_temperature[b] - actual_t_out[t])
                )

        for b in range(N):
            planned_grid = float(planned["p_el"][b, t])
            final_values = {name: float(values[b, t]) for name, values in planned.items()}
            final_values.update({
                "charge": float(charge[b]),
                "discharge": float(discharge[b]),
                "soc": float(soc[b]),
                "cs": float(charge[b] > 1e-9),
                "p_el": float(grid_import[b]),
                "curtail": float(pv_surplus[b]),
                "f": float(hp_heat[b] / C.p_th_nom),
                "T_in": float(indoor_temperature[b]),
            })
            record = {
                "building": C.BUILDING_IDS[b],
                "interval": t,
                "predicted_load_kw": float(C.LOAD[b, t]),
                "actual_load_kw": float(actual_load[b, t]),
                "predicted_pv_kw": float(C.PV_B[b, t]),
                "actual_pv_kw": float(actual_pv[b, t]),
                "predicted_t_out_c": float(C.T_OUT[t]),
                "actual_t_out_c": float(actual_t_out[t]),
                "indoor_temperature_c": float(indoor_temperature[b]),
                "heat_pump_kw": float(planned["p_hp"][b, t]),
                "pv_to_load_kw": float(pv_to_load[b]),
                "battery_charge_kw": float(charge[b]),
                "battery_discharge_kw": float(discharge[b]),
                "battery_soc_kwh": float(soc[b]),
                "p2p_import_kw": float(p2p_import[b]),
                "p2p_export_kw": float(p2p_export[b]),
                "planned_grid_import_kw": planned_grid,
                "actual_grid_import_kw": float(grid_import[b]),
                "grid_import_error_kw": float(grid_import[b] - planned_grid),
                "grid_import_error_kwh": float(
                    (grid_import[b] - planned_grid) * optimiser.DT
                ),
                "unused_pv_kw": float(pv_surplus[b]),
                "planned_p2p_net_trade_kw": float(planned_trades[b, :, t].sum()),
                "final_p2p_net_trade_kw": float(actual_trades[b].sum()),
                "planned_p2p_import_kw": float(np.maximum(-planned_trades[b, :, t], 0.0).sum()),
                "planned_p2p_export_kw": float(np.maximum(planned_trades[b, :, t], 0.0).sum()),
            }
            for name, values in planned.items():
                record[f"planned_{name}"] = float(values[b, t])
                record[f"final_{name}"] = final_values[name]
                record[f"deviation_{name}"] = final_values[name] - float(values[b, t])
            record["planned_soc_target_kwh"] = float(planned["soc"][b, t])
            record["actual_soc_deviation_kwh"] = float(soc[b] - planned["soc"][b, t])
            record["battery_charge_correction_kw"] = float(charge_correction[b])
            record["battery_discharge_correction_kw"] = float(discharge_correction[b])
            for neighbour in range(optimiser.N):
                if neighbour == b:
                    continue
                neighbour_id = str(C.BUILDING_IDS[neighbour])
                record[f"planned_trade_to_{neighbour_id}_kw"] = float(planned_trades[b, neighbour, t])
                record[f"final_trade_to_{neighbour_id}_kw"] = float(actual_trades[b, neighbour])
                record[f"planned_q_to_{neighbour_id}_kw"] = float(planned_trade_variables["q"][b, neighbour, t])
                record[f"final_q_to_{neighbour_id}_kw"] = float(actual_trades[b, neighbour])
                record[f"deviation_q_to_{neighbour_id}_kw"] = (
                    record[f"final_q_to_{neighbour_id}_kw"]
                    - record[f"planned_q_to_{neighbour_id}_kw"]
                )
                record[f"planned_qpos_to_{neighbour_id}_kw"] = float(planned_trade_variables["qpos"][b, neighbour, t])
                record[f"final_qpos_to_{neighbour_id}_kw"] = float(max(actual_trades[b, neighbour], 0.0))
                record[f"deviation_qpos_to_{neighbour_id}_kw"] = (
                    record[f"final_qpos_to_{neighbour_id}_kw"]
                    - record[f"planned_qpos_to_{neighbour_id}_kw"]
                )
                record[f"planned_qneg_to_{neighbour_id}_kw"] = float(planned_trade_variables["qneg"][b, neighbour, t])
                record[f"final_qneg_to_{neighbour_id}_kw"] = float(max(-actual_trades[b, neighbour], 0.0))
                record[f"deviation_qneg_to_{neighbour_id}_kw"] = (
                    record[f"final_qneg_to_{neighbour_id}_kw"]
                    - record[f"planned_qneg_to_{neighbour_id}_kw"]
                )
            records.append(record)

        community_import = float(grid_import.sum())
        step_timings.append(time.perf_counter() - step_started)
        print(
            f"Interval {t + 1:02d}/{optimiser.T}: "
            f"community grid import = {community_import:.2f} kW"
        )

        if pace_seconds:
            time.sleep(pace_seconds)

    return records


def _correlated_normal(rng, shape, correlation):
    """Standard-normal draws, shape (runs, N, T), with `correlation` between buildings.

    Each building's error is sqrt(rho) x a community-wide factor plus
    sqrt(1 - rho) x its own factor. rho = 0 gives the independent draws used so
    far (identical random sequence, so existing seeds reproduce).
    """
    if correlation <= 0.0:
        return rng.normal(0.0, 1.0, size=shape)
    common = rng.normal(0.0, 1.0, size=(shape[0], 1, shape[2]))
    own = rng.normal(0.0, 1.0, size=shape)
    return np.sqrt(correlation) * common + np.sqrt(1.0 - correlation) * own


def load_or_create_scenarios(scenario_file, *, runs, load_noise, pv_noise, temp_noise=1.0,
                             seed=None, force_refresh=False, load_bias=None, correlation=0.0):
    """Shared noisy trajectories reused by every method.

    Returns (actual_load, actual_pv, actual_t_out) with shapes (runs, N, T),
    (runs, N, T) and (runs, T). Load/PV noise is multiplicative per building;
    outdoor-temperature noise is additive (deg C) and common to the community.
    A saved file is reused when its noise levels and forecasts match and it
    holds at least `runs` scenarios, so separately launched simulations see
    identical deviations. A file saved before temperature noise existed keeps
    its load/PV draws and only gains temperature draws.

    load_bias (length T, fraction) shifts the actual load away from the
    forecast systematically, e.g. +0.15 in hours the forecast underestimates.
    correlation makes load and PV errors partly common to all buildings.
    """
    scenario_file = Path(scenario_file)
    C = optimiser.C
    rng = np.random.default_rng(seed)
    load_bias = np.zeros(C.LOAD.shape[1]) if load_bias is None else np.asarray(load_bias, dtype=float)
    if scenario_file.exists() and not force_refresh:
        with np.load(scenario_file, allow_pickle=False) as cache:
            stored = {name: cache[name].copy() for name in cache.files}
        same_load_pv = (
            np.isclose(float(stored["load_noise"]), load_noise)
            and np.isclose(float(stored["pv_noise"]), pv_noise)
            and stored["actual_load"].shape[1:] == C.LOAD.shape
            and stored["actual_load"].shape[0] >= runs
            and np.allclose(stored["forecast_load"], C.LOAD)
            and np.allclose(stored["forecast_pv"], C.PV_B)
            and np.allclose(stored.get("load_bias", np.zeros_like(load_bias)), load_bias)
            and np.isclose(float(stored.get("correlation", 0.0)), correlation)
        )
        if same_load_pv:
            count = stored["actual_load"].shape[0]
            same_temperature = (
                "actual_t_out" in stored
                and np.isclose(float(stored["temp_noise"]), temp_noise)
                and np.allclose(stored["forecast_t_out"], C.T_OUT)
            )
            if not same_temperature:
                stored["actual_t_out"] = C.T_OUT[None] + rng.normal(0.0, temp_noise, size=(count,) + C.T_OUT.shape)
                stored["forecast_t_out"] = C.T_OUT
                stored["temp_noise"] = np.array(temp_noise)
                np.savez_compressed(scenario_file, **stored)
                print(f"Added outdoor-temperature noise to shared scenarios: {scenario_file}")
            else:
                print(f"Loaded shared noise scenarios: {scenario_file}")
            return (stored["actual_load"][:runs].copy(), stored["actual_pv"][:runs].copy(),
                    stored["actual_t_out"][:runs].copy())
        print("Shared noise scenarios do not match the request; drawing new ones.")

    shape = (runs,) + C.LOAD.shape
    load_error = load_bias[None, None, :] + load_noise * _correlated_normal(rng, shape, correlation)
    actual_load = np.maximum(0.0, C.LOAD[None] * (1.0 + load_error))
    actual_pv = np.maximum(0.0, C.PV_B[None] * (1.0 + pv_noise * _correlated_normal(rng, shape, correlation)))
    actual_t_out = C.T_OUT[None] + rng.normal(0.0, temp_noise, size=(runs,) + C.T_OUT.shape)
    scenario_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        scenario_file,
        actual_load=actual_load,
        actual_pv=actual_pv,
        actual_t_out=actual_t_out,
        forecast_load=C.LOAD,
        forecast_pv=C.PV_B,
        forecast_t_out=C.T_OUT,
        load_noise=np.array(load_noise),
        pv_noise=np.array(pv_noise),
        temp_noise=np.array(temp_noise),
        load_bias=load_bias,
        correlation=np.array(correlation),
        seed=np.array(-1 if seed is None else seed),
    )
    print(f"Saved shared noise scenarios: {scenario_file}")
    return actual_load, actual_pv, actual_t_out


def _schedule_signature():
    digest = hashlib.sha256()
    for source_path in (Path(optimiser.__file__), Path(optimiser.C.__file__)):
        digest.update(source_path.read_bytes())

    for name in (
        "LOAD", "PV_B", "PRICE", "T_OUT", "T_SET", "CAP", "MAXPOW",
        "INIT_SOC", "INIT_T_IN", "HAS_BATTERY",
    ):
        values = np.ascontiguousarray(getattr(optimiser.C, name))
        digest.update(name.encode("ascii"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())

    settings = (
        optimiser.C.SHOWCASE_DAY,
        optimiser.C.SOLVER,
        optimiser.C.dt,
        optimiser.C.eta_charge,
        optimiser.C.eta_discharge,
        optimiser.C.alpha,
        optimiser.C.p_th_nom,
        optimiser.C.cop_base,
        optimiser.C.T_ref,
        optimiser.C.hp_max_power,
        optimiser.C.max_thermal_power,
        optimiser.C.gas_price,
        optimiser.C.efficiency,
        optimiser.C.SOLVE_TIME_LIMIT,
        optimiser.RHO,
        optimiser.MAX_ITERS,
        optimiser.EPS_PRIMAL,
        optimiser.EPS_DUAL,
        optimiser.ADAPT_RHO,
        optimiser.STOP_STREAK,
        optimiser.RELAX_BINARIES,
        optimiser.FEE_FRAC,
        optimiser.use_congestion,
    )
    digest.update(repr(settings).encode("utf-8"))
    return digest.hexdigest()


def load_or_solve_schedule(cache_file, *, force_refresh=False):
    """Load a compatible solved schedule, or run ADMM and cache its outputs."""
    signature = _schedule_signature()
    expected_shape = (optimiser.N, optimiser.T)

    if cache_file.exists() and not force_refresh:
        try:
            with np.load(cache_file, allow_pickle=False) as cache:
                if str(cache["signature"].item()) == signature:
                    variables = {
                        name: cache[f"variable_{name}"].copy()
                        for name in _SCHEDULE_VARIABLES
                    }
                    trade_variables = {
                        name: cache[f"trade_variable_{name}"].copy()
                        for name in _TRADE_VARIABLES
                    }
                    trades = cache["trades"]
                    if (
                        all(values.shape == expected_shape for values in variables.values())
                        and all(
                            values.shape == (optimiser.N, optimiser.N, optimiser.T)
                            for values in trade_variables.values()
                        )
                        and trades.shape == (optimiser.N, optimiser.N, optimiser.T)
                    ):
                        print(f"Loaded cached optimisation schedule: {cache_file}")
                        return {
                            "variables": variables,
                            "trade_variables": trade_variables,
                            "trades": trades.copy(),
                        }
            print("Cached schedule is stale or has an unexpected shape; resolving ADMM.")
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            print("Cached schedule could not be read; resolving ADMM.")
    elif force_refresh:
        print("Cache refresh requested; resolving ADMM.")

    subs, _q_all, _lam_all, Z, _hist = optimiser.run_admm()
    variables = {
        name: np.array(
            [[pyo.value(getattr(subs[b], name)[t]) for t in range(optimiser.T)] for b in range(optimiser.N)],
            dtype=float,
        )
        for name in _SCHEDULE_VARIABLES
    }
    trade_variables = {
        name: np.zeros((optimiser.N, optimiser.N, optimiser.T), dtype=float)
        for name in _TRADE_VARIABLES
    }
    for b, model in enumerate(subs):
        for k, neighbour in enumerate(optimiser.NB[b]):
            for name in _TRADE_VARIABLES:
                trade_variables[name][b, neighbour] = np.array(
                    [pyo.value(getattr(model, name)[k, t]) for t in range(optimiser.T)],
                    dtype=float,
                )
    trades = np.zeros((optimiser.N, optimiser.N, optimiser.T), dtype=float)
    for lo, hi in optimiser.PAIRS:
        trades[lo, hi] = Z[(lo, hi)]
        trades[hi, lo] = -Z[(lo, hi)]

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        signature=np.array(signature),
        **{f"variable_{name}": values for name, values in variables.items()},
        **{f"trade_variable_{name}": values for name, values in trade_variables.items()},
        trades=trades,
    )
    print(f"Cached optimisation schedule: {cache_file}")
    return {"variables": variables, "trade_variables": trade_variables, "trades": trades}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--load-noise",
        type=float,
        default=DEFAULT_LOAD_NOISE,
        help="Load noise standard deviation as a fraction (default: %(default)s).",
    )
    parser.add_argument(
        "--pv-noise",
        type=float,
        default=0.30,
        help="PV noise standard deviation as a fraction (default: 0.30).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=50,
        help="Number of independent noisy simulations (default: 50).",
    )
    parser.add_argument(
        "--pace-seconds",
        type=float,
        default=0.0,
        help="Delay between intervals; 0 runs without a delay.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-solve ADMM and replace the cached schedule.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed used when drawing new scenarios.")
    parser.add_argument(
        "--temp-noise",
        type=float,
        default=1.0,
        help="Outdoor-temperature noise standard deviation in deg C (default: 1.0).",
    )
    parser.add_argument(
        "--refresh-scenarios",
        action="store_true",
        help="Draw new shared noise scenarios instead of reusing the saved ones.",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1.")

    output_dir = optimiser.C.PLOTS_DIR / "ADMM Optimisation Real-time"
    cache_file = output_dir / "bilateral_admm_schedule.npz"
    schedule = load_or_solve_schedule(
        cache_file,
        force_refresh=args.refresh_cache,
    )
    scenario_load, scenario_pv, scenario_t_out = load_or_create_scenarios(
        output_dir / SCENARIO_FILE_NAME,
        runs=args.runs,
        load_noise=args.load_noise,
        pv_noise=args.pv_noise,
        temp_noise=args.temp_noise,
        seed=args.seed,
        force_refresh=args.refresh_scenarios,
    )
    all_records = []

    for run_number in range(1, args.runs + 1):
        print(f"\nSimulation {run_number}/{args.runs}")
        records = simulate(
            schedule,
            load_noise=args.load_noise,
            pv_noise=args.pv_noise,
            pace_seconds=args.pace_seconds,
            actual_load=scenario_load[run_number - 1],
            actual_pv=scenario_pv[run_number - 1],
            actual_t_out=scenario_t_out[run_number - 1],
        )
        for record in records:
            record["simulation_run"] = run_number
        all_records.extend(records)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "28_noisy_realtime_simulation.csv"

    with output_file.open("w", newline="", encoding="utf-8") as file:
        fieldnames = list(dict.fromkeys(
            field
            for record in all_records
            for field in record
        ))
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(all_records)

    run_errors = []
    for run_number in range(1, args.runs + 1):
        run_records = [row for row in all_records if row["simulation_run"] == run_number]
        run_errors.append(
            sum(row["grid_import_error_kw"] for row in run_records) * optimiser.DT
        )

    print(f"\nCompleted {args.runs} simulations")
    print(f"  Mean total grid-import error: {np.mean(run_errors):+.2f} kWh")
    print(f"  Std. dev. of total error:     {np.std(run_errors):.2f} kWh")
    print(f"  Results saved to:    {output_file}")


if __name__ == "__main__":
    main()