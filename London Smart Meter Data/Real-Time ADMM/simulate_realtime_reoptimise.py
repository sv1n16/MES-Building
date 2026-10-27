"""Receding-horizon re-optimisation of the bilateral ADMM day-ahead plan.

Both methods run on the shared noisy scenarios from
simulate_noisy_realtime.load_or_create_scenarios, so they (and the fixed-schedule
baseline) see exactly the same load / PV / outdoor-temperature deviations. At
every interval the model inputs for the whole remaining horizon are taken from
that noisy trajectory (future_inputs="noisy", the default) or, alternatively,
only the current interval is measured and the rest stays at the day-ahead
forecast (future_inputs="forecast"). Only the first-interval decisions are
applied, and the realised battery SOC / indoor temperature become the next
solve's initial state.

local   Each building re-optimises its own remaining horizon on its own. No new
        P2P trades are negotiated: the planned trade with each neighbour is an
        upper bound that the building may honour or curtail (same direction,
        never more). Objective per building:
            operating cost + P2P settlement
            + cost_penalty  * |cost - original planned cost| / |original planned cost|
            + share_penalty * (planned P2P energy - delivered P2P energy)   [GBP/kWh]
        After all buildings solve, each pair trades the smaller of the seller's
        offer and the buyer's acceptance; a buyer's shortfall is imported from
        the grid and a seller's unsold surplus is curtailed.
admm    Bilateral ADMM is re-run over the remaining horizon at every interval,
        warm-started from the previous interval's duals/trades (or cold).

Every solve is timed (model build + solver call) and written to a timing CSV
so the real-time feasibility of each method can be analysed.

Examples:
    python simulate_realtime_reoptimise.py --method local --runs 5
    python simulate_realtime_reoptimise.py --method admm --runs 1 --admm-start cold
"""

from __future__ import annotations

import argparse
import csv
import time
from contextlib import contextmanager

import numpy as np
import pyomo.environ as pyo

import simulate_noisy_realtime as baseline

OPT = baseline.optimiser
C = OPT.C
N = OPT.N
FULL_T = OPT.T
DT = C.dt
INTERVAL_SECONDS = DT * 3600.0

FORECAST_LOAD = C.LOAD.copy()
FORECAST_PV = C.PV_B.copy()
FULL_PRICE = C.PRICE.copy()
FULL_T_OUT = C.T_OUT.copy()
FULL_T_SET = C.T_SET.copy()
FULL_CONGESTION = np.asarray(OPT.congestion_weight, dtype=float).copy()
DAY_START_SOC = np.asarray(C.INIT_SOC, dtype=float).copy()
DAY_START_T_IN = np.asarray(C.INIT_T_IN, dtype=float).copy()

RESULTS_DIR = C.PLOTS_DIR / "ADMM Optimisation Real-time"
SCHEDULE_CACHE = RESULTS_DIR / "bilateral_admm_schedule.npz"
SCENARIO_FILE = RESULTS_DIR / baseline.SCENARIO_FILE_NAME

_FIRST_STEP_NAMES = ("charge", "discharge", "soc", "p_el", "curtail", "p_hp", "q_boiler", "gas", "T_in")
_RECORD_NAMES = _FIRST_STEP_NAMES + ("p2p_import", "p2p_export")
_ACCEPTABLE = {
    pyo.TerminationCondition.optimal,
    pyo.TerminationCondition.locallyOptimal,
    pyo.TerminationCondition.feasible,
    pyo.TerminationCondition.maxTimeLimit,
}


# ============================================================================
# REMAINING-HORIZON MODEL DATA
# ============================================================================
@contextmanager
def remaining_horizon(start, load, pv, t_out, soc, indoor_temperature):
    """Point the optimiser at intervals start..T-1 with the given input windows.

    load/pv are (N, T-start) and t_out is (T-start,). Interval 0 keeps the
    day-ahead semantics (SOC/T_in fixed, no battery use), so it matches the plan
    and the fixed-schedule baseline; later intervals start from the realised
    state at the end of the previous interval.
    """
    load = np.asarray(load, dtype=float).copy()
    pv = np.asarray(pv, dtype=float).copy()

    central_names = ("LOAD", "PV_B", "PRICE", "T_OUT", "T_SET", "time_horizon", "INIT_SOC", "INIT_T_IN")
    optimiser_names = ("T", "DT", "COP", "PV_B", "PRICE", "congestion_weight", "ROLLING_FIRST_INTERVAL")
    old_central = {name: getattr(C, name) for name in central_names}
    old_optimiser = {name: getattr(OPT, name) for name in optimiser_names}

    C.LOAD = load
    C.PV_B = pv
    C.PRICE = FULL_PRICE[start:].copy()
    C.T_OUT = np.asarray(t_out, dtype=float).copy()
    C.T_SET = FULL_T_SET[start:].copy()
    C.time_horizon = load.shape[1]
    C.INIT_SOC = np.asarray(soc, dtype=float).copy()
    C.INIT_T_IN = np.asarray(indoor_temperature, dtype=float).copy()

    OPT.T = C.time_horizon
    OPT.DT = DT
    OPT.COP = C.cop_base + 0.01 * (C.T_OUT - C.T_ref)
    OPT.PV_B = C.PV_B
    OPT.PRICE = C.PRICE
    # slice the day's weights rather than re-normalising over the remaining prices
    OPT.congestion_weight = FULL_CONGESTION[start:].copy()
    OPT.ROLLING_FIRST_INTERVAL = start > 0
    try:
        yield
    finally:
        for name, value in old_central.items():
            setattr(C, name, value)
        for name, value in old_optimiser.items():
            setattr(OPT, name, value)


def horizon_inputs(start, actual_load, actual_pv, actual_t_out, future_inputs="noisy"):
    """Model inputs for intervals start..T-1.

    "noisy":    the noisy trajectory for the whole remaining horizon.
    "forecast": measured values for the current interval, day-ahead forecast after.
    """
    if future_inputs == "noisy":
        return actual_load[:, start:], actual_pv[:, start:], actual_t_out[start:]
    if future_inputs != "forecast":
        raise ValueError(f"future_inputs must be 'noisy' or 'forecast', not {future_inputs!r}")
    load = FORECAST_LOAD[:, start:].copy()
    pv = FORECAST_PV[:, start:].copy()
    t_out = FULL_T_OUT[start:].copy()
    load[:, 0] = actual_load[:, start]
    pv[:, 0] = actual_pv[:, start]
    t_out[0] = actual_t_out[start]
    return load, pv, t_out


def _make_solver(nonconvex=False):
    C._require_solver()
    solver = pyo.SolverFactory(C.SOLVER)
    options = [("OutputFlag", 0), ("LogFile", "")]
    if nonconvex:
        options.append(("NonConvex", 2))
    if C.SOLVE_TIME_LIMIT > 0:
        options.append(("TimeLimit", C.SOLVE_TIME_LIMIT))
    for option, value in options:
        try:
            solver.options[option] = value
        except Exception:
            pass
    return solver


# ============================================================================
# COSTS AND RECORDS (shared by all methods, including the fixed schedule)
# ============================================================================
def planned_first_step(schedule, building, interval):
    planned = schedule["variables"]
    trade = schedule["trades"][building, :, interval]
    values = {name: float(planned[name][building, interval]) for name in _FIRST_STEP_NAMES}
    values["p2p_import"] = float(np.maximum(-trade, 0.0).sum())
    values["p2p_export"] = float(np.maximum(trade, 0.0).sum())
    return values


def interval_costs(interval, values):
    """Realised operating cost of one building-interval (GBP).

    The P2P settlement (FEE_FRAC x grid tariff) cancels across the community.
    """
    price = FULL_PRICE[interval]
    costs = {
        "electricity_cost": price * values["p_el"] * DT,
        "gas_cost": C.gas_price / 100.0 * values["gas"] * DT,
        "comfort_cost": C.alpha * (values["T_in"] - FULL_T_SET[interval]) ** 2,
        "p2p_settlement": OPT.FEE_FRAC * price * (values["p2p_import"] - values["p2p_export"]) * DT,
    }
    costs["total_cost"] = sum(costs.values())
    return costs


def make_record(run, method, interval, building, actual_load, actual_pv, actual_t_out, final, schedule):
    planned = planned_first_step(schedule, building, interval)
    record = {
        "simulation_run": run,
        "method": method,
        "interval": interval,
        "building": C.BUILDING_IDS[building],
        "predicted_load_kw": float(FORECAST_LOAD[building, interval]),
        "actual_load_kw": float(actual_load),
        "predicted_pv_kw": float(FORECAST_PV[building, interval]),
        "actual_pv_kw": float(actual_pv),
        "predicted_t_out_c": float(FULL_T_OUT[interval]),
        "actual_t_out_c": float(actual_t_out),
    }
    for name in _RECORD_NAMES:
        record[f"planned_{name}"] = planned[name]
        record[f"final_{name}"] = float(final[name])
        record[f"deviation_{name}"] = float(final[name]) - planned[name]
    for name, value in interval_costs(interval, planned).items():
        record[f"planned_{name}"] = float(value)
    for name, value in interval_costs(interval, final).items():
        record[f"final_{name}"] = float(value)
    record["foregone_p2p_kwh"] = (
        planned["p2p_import"] + planned["p2p_export"] - float(final["p2p_import"]) - float(final["p2p_export"])
    ) * DT
    return record


def timing_row(run, method, interval, scope, *, building="community", iteration="",
               wall_seconds=0.0, build_seconds=0.0, solver_seconds=0.0,
               parallel_latency_seconds=None, solver_calls=1, admm_iterations="",
               horizon_length=None, termination=""):
    latency = wall_seconds if parallel_latency_seconds is None else parallel_latency_seconds
    return {
        "simulation_run": run,
        "method": method,
        "interval": interval,
        "scope": scope,
        "building": building,
        "iteration": iteration,
        "horizon_length": FULL_T - interval if horizon_length is None else horizon_length,
        "wall_seconds": float(wall_seconds),
        "build_seconds": float(build_seconds),
        "solver_seconds": float(solver_seconds),
        "parallel_latency_seconds": float(latency),
        "solver_calls": solver_calls,
        "admm_iterations": admm_iterations,
        "interval_seconds": INTERVAL_SECONDS,
        "latency_fraction_of_interval": float(latency) / INTERVAL_SECONDS,
        "realtime_feasible": bool(latency < INTERVAL_SECONDS),
        "termination": termination,
    }


def append_rows(path, rows):
    """Append rows to a CSV, writing the header when the file is empty."""
    if not rows:
        return
    fields = list(rows[0])
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="raise")
        if file.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)
        file.flush()


# ============================================================================
# METHOD 2: LOCAL NON-SHARING RE-OPTIMISATION
# ============================================================================
def _planned_local_cost(schedule, building, start):
    """Original plan's local cost over start..T-1, same terms as the local objective."""
    planned = schedule["variables"]
    trade = schedule["trades"][building, :, start:]
    price = FULL_PRICE[start:]
    p_el = planned["p_el"][building, start:]
    cost = np.sum(price * p_el * DT)
    cost += np.sum(C.gas_price / 100.0 * planned["gas"][building, start:] * DT)
    cost += np.sum(C.alpha * (planned["T_in"][building, start:] - FULL_T_SET[start:]) ** 2)
    if OPT.use_congestion:
        cost += np.sum(FULL_CONGESTION[start:] * p_el**2)
    p2p_net_import = np.maximum(-trade, 0.0).sum(axis=0) - np.maximum(trade, 0.0).sum(axis=0)
    cost += np.sum(OPT.FEE_FRAC * price * p2p_net_import * DT)
    return float(cost)


def _solve_local_building(building, start, schedule, solver, cost_penalty, share_penalty):
    """Must be called inside remaining_horizon(start, ...)."""
    build_started = time.perf_counter()
    m = OPT.build_subproblem(building)
    m.obj.deactivate()
    neighbours = OPT.NB[building]
    planned_trade = schedule["trades"][building][neighbours][:, start:]  # (N-1, L), + = building delivers
    for k in m.k:
        for t in m.t:
            m.qpos[k, t].setub(max(float(planned_trade[k, t]), 0.0))
            m.qneg[k, t].setub(max(float(-planned_trade[k, t]), 0.0))

    cost = sum(
        OPT.PRICE[t] * m.p_el[t] * DT
        + C.gas_price / 100.0 * m.gas[t] * DT
        + C.alpha * (m.T_in[t] - C.T_SET[t]) ** 2
        + OPT.FEE_FRAC * OPT.PRICE[t] * sum(m.qneg[k, t] - m.qpos[k, t] for k in m.k) * DT
        for t in m.t
    )
    if OPT.use_congestion:
        cost += sum(OPT.congestion_weight[t] * m.p_el[t] ** 2 for t in m.t)

    original_cost = _planned_local_cost(schedule, building, start)
    scale = max(abs(original_cost), 1.0)
    m.cost_deviation = pyo.Var(domain=pyo.NonNegativeReals)
    m.c_cost_dev_pos = pyo.Constraint(expr=(cost - original_cost) / scale <= m.cost_deviation)
    m.c_cost_dev_neg = pyo.Constraint(expr=(original_cost - cost) / scale <= m.cost_deviation)
    share_deviation = sum(
        (abs(float(planned_trade[k, t])) - m.qpos[k, t] - m.qneg[k, t]) * DT for k in m.k for t in m.t
    )
    # m.local_obj = pyo.Objective(
    #     expr=cost + cost_penalty * m.cost_deviation + share_penalty * share_deviation,
    #     sense=pyo.minimize,
    # )
    m.local_obj = pyo.Objective(
            expr=cost +  share_penalty * share_deviation,
            sense=pyo.minimize,
        )
    build_seconds = time.perf_counter() - build_started

    solve_started = time.perf_counter()
    result = solver.solve(m, tee=False)
    solver_seconds = time.perf_counter() - solve_started
    termination = result.solver.termination_condition
    if termination not in _ACCEPTABLE or pyo.value(m.p_el[0], exception=False) is None:
        raise RuntimeError(
            f"Local solve failed for building {C.BUILDING_IDS[building]} at interval {start}: "
            f"status={result.solver.status}, termination={termination}"
        )

    values = {name: float(pyo.value(getattr(m, name)[0])) for name in _FIRST_STEP_NAMES}
    offer = np.zeros(N)
    accept = np.zeros(N)
    for k, neighbour in enumerate(neighbours):
        offer[neighbour] = max(float(pyo.value(m.qpos[k, 0])), 0.0)
        accept[neighbour] = max(float(pyo.value(m.qneg[k, 0])), 0.0)
    values["offer"] = offer
    values["accept"] = accept
    values["original_cost"] = original_cost
    values["cost_deviation"] = float(pyo.value(m.cost_deviation))
    values["build_seconds"] = build_seconds
    values["solver_seconds"] = solver_seconds
    values["termination"] = str(termination)
    return values


def _reconcile_trades(outputs):
    """Realise each pair at min(seller offer, buyer acceptance); settle the rest locally."""
    offer = np.array([out["offer"] for out in outputs])  # offer[b, j]: b -> j
    accept = np.array([out["accept"] for out in outputs])  # accept[j, b]: j takes from b
    realised = np.minimum(offer, accept.T)  # realised[b, j]: b -> j
    finals = []
    for b, out in enumerate(outputs):
        p2p_export = float(realised[b].sum())
        p2p_import = float(realised[:, b].sum())
        final = {name: out[name] for name in _FIRST_STEP_NAMES}
        final["p_el"] = out["p_el"] + float(accept[b].sum()) - p2p_import
        final["curtail"] = out["curtail"] + float(offer[b].sum()) - p2p_export
        final["p2p_import"] = p2p_import
        final["p2p_export"] = p2p_export
        finals.append(final)
    return finals


def simulate_local(schedule, actual_load, actual_pv, actual_t_out, *, run, cost_penalty, share_penalty,
                   future_inputs="noisy", results_file=None, timing_file=None):
    solver = _make_solver(nonconvex=True)  # |cost - reference| makes one constraint non-convex
    soc = DAY_START_SOC.copy()
    temperature = DAY_START_T_IN.copy()
    records, timings = [], []
    for interval in range(FULL_T):
        outputs = []
        step_timings = []
        step_started = time.perf_counter()
        inputs = horizon_inputs(interval, actual_load, actual_pv, actual_t_out, future_inputs)
        with remaining_horizon(interval, *inputs, soc, temperature):
            for building in range(N):
                started = time.perf_counter()
                out = _solve_local_building(building, interval, schedule, solver, cost_penalty, share_penalty)
                outputs.append(out)
                step_timings.append(timing_row(
                    run, "local_reoptimisation", interval, "building",
                    building=C.BUILDING_IDS[building],
                    wall_seconds=time.perf_counter() - started,
                    build_seconds=out["build_seconds"],
                    solver_seconds=out["solver_seconds"],
                    termination=out["termination"],
                ))
        step_wall = time.perf_counter() - step_started
        for building, out in enumerate(outputs):
            soc[building] = out["soc"]
            temperature[building] = out["T_in"]

        finals = _reconcile_trades(outputs)
        step_records = [
            make_record(run, "local_reoptimisation", interval, b, actual_load[b, interval],
                        actual_pv[b, interval], actual_t_out[interval], finals[b], schedule)
            for b in range(N)
        ]
        step_timings.append(timing_row(
            run, "local_reoptimisation", interval, "community",
            wall_seconds=step_wall,
            build_seconds=sum(row["build_seconds"] for row in step_timings),
            solver_seconds=sum(row["solver_seconds"] for row in step_timings),
            # buildings solve in parallel on their own controllers
            parallel_latency_seconds=max(row["wall_seconds"] for row in step_timings),
            solver_calls=N,
            termination="all_buildings_solved",
        ))
        if results_file is not None:
            append_rows(results_file, step_records)
        if timing_file is not None:
            append_rows(timing_file, step_timings)
        records.extend(step_records)
        timings.extend(step_timings)
        print(
            f"  run {run} local    interval {interval + 1:02d}/{FULL_T}: "
            f"sequential {step_wall:6.2f}s, parallel {step_timings[-1]['parallel_latency_seconds']:5.2f}s, "
            f"grid {sum(f['p_el'] for f in finals):6.2f} kW",
            flush=True,
        )
    return records, timings


# ============================================================================
# METHOD 3: ROLLING BILATERAL ADMM OVER THE REMAINING HORIZON
# ============================================================================
def simulate_rolling_admm(schedule, actual_load, actual_pv, actual_t_out, *, run, warm_start=True,
                          future_inputs="noisy", results_file=None, timing_file=None,
                          method="rolling_admm"):
    soc = DAY_START_SOC.copy()
    temperature = DAY_START_T_IN.copy()
    previous_lam = None
    previous_Z = {(lo, hi): schedule["trades"][lo, hi].copy() for lo, hi in OPT.PAIRS} if warm_start else None
    records, timings = [], []
    for interval in range(FULL_T):
        solve_log = []
        init_lam = init_Z = None
        if warm_start:
            # drop the interval that has just been dispatched
            shift = 0 if interval == 0 else 1
            init_Z = {pair: values[shift:] for pair, values in previous_Z.items()}
            if previous_lam is not None:
                init_lam = [lam[:, shift:] for lam in previous_lam]
        started = time.perf_counter()
        inputs = horizon_inputs(interval, actual_load, actual_pv, actual_t_out, future_inputs)
        with remaining_horizon(interval, *inputs, soc, temperature):
            subs, _q_all, lam_all, Z, hist = OPT.run_admm(
                init_lam=init_lam, init_Z=init_Z, solve_log=solve_log, verbose=False
            )
            outputs = [{name: float(pyo.value(getattr(m, name)[0])) for name in _FIRST_STEP_NAMES} for m in subs]
        wall = time.perf_counter() - started
        previous_lam, previous_Z = lam_all, Z

        flows = np.zeros((N, N))  # flows[b, j]: b -> j at the current interval
        for lo, hi in OPT.PAIRS:
            z = float(Z[(lo, hi)][0])
            if z >= 0:
                flows[lo, hi] = z
            else:
                flows[hi, lo] = -z
        for b, out in enumerate(outputs):
            out["p2p_export"] = float(flows[b].sum())
            out["p2p_import"] = float(flows[:, b].sum())
            soc[b] = out["soc"]
            temperature[b] = out["T_in"]

        log = np.array(solve_log, dtype=float).reshape(-1, 3)
        iterations = int(len(hist))
        per_iteration_max = [float(log[log[:, 0] == i, 2].max()) for i in range(iterations)]
        per_iteration_sum = [float(log[log[:, 0] == i, 2].sum()) for i in range(iterations)]
        step_timings = [
            timing_row(run, method, interval, "admm_iteration", iteration=i,
                       wall_seconds=per_iteration_sum[i], solver_seconds=per_iteration_sum[i],
                       parallel_latency_seconds=per_iteration_max[i], solver_calls=N)
            for i in range(iterations)
        ]
        solver_total = float(log[:, 2].sum())
        step_timings.append(timing_row(
            run, method, interval, "community",
            wall_seconds=wall,
            build_seconds=max(wall - solver_total, 0.0),
            solver_seconds=solver_total,
            # subproblems of one iteration run in parallel; iterations are sequential
            parallel_latency_seconds=sum(per_iteration_max),
            solver_calls=len(log),
            admm_iterations=iterations,
            termination=f"{'warm' if warm_start else 'cold'}_start",
        ))
        step_records = [
            make_record(run, method, interval, b, actual_load[b, interval],
                        actual_pv[b, interval], actual_t_out[interval], outputs[b], schedule)
            for b in range(N)
        ]
        if results_file is not None:
            append_rows(results_file, step_records)
        if timing_file is not None:
            append_rows(timing_file, step_timings)
        records.extend(step_records)
        timings.extend(step_timings)
        print(
            f"  run {run} ADMM     interval {interval + 1:02d}/{FULL_T}: "
            f"{iterations:4d} iters, sequential {wall:7.2f}s, parallel {sum(per_iteration_max):6.2f}s, "
            f"grid {sum(o['p_el'] for o in outputs):6.2f} kW",
            flush=True,
        )
    return records, timings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=("local", "admm", "both"), default="local")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--load-noise", type=float, default=baseline.DEFAULT_LOAD_NOISE)
    parser.add_argument("--pv-noise", type=float, default=0.30)
    parser.add_argument("--temp-noise", type=float, default=1.0, help="Outdoor-temperature noise std (deg C).")
    parser.add_argument("--future-inputs", choices=("noisy", "forecast"), default="noisy",
                        help="Remaining-horizon inputs: the noisy trajectory, or forecast after the current interval.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cost-penalty", type=float, default=10.0)
    parser.add_argument("--share-penalty", type=float, default=0.1, help="GBP per kWh of planned P2P not delivered.")
    parser.add_argument("--admm-start", choices=("warm", "cold"), default="warm")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--refresh-scenarios", action="store_true")
    args = parser.parse_args()
    if args.runs < 1 or min(args.load_noise, args.pv_noise, args.cost_penalty, args.share_penalty) < 0:
        parser.error("--runs must be positive and noise/penalty values non-negative.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    schedule = baseline.load_or_solve_schedule(SCHEDULE_CACHE, force_refresh=args.refresh_cache)
    loads, pvs, t_outs = baseline.load_or_create_scenarios(
        SCENARIO_FILE, runs=args.runs, load_noise=args.load_noise, pv_noise=args.pv_noise,
        temp_noise=args.temp_noise, seed=args.seed, force_refresh=args.refresh_scenarios,
    )
    results_file = RESULTS_DIR / "reoptimized_realtime_simulation.csv"
    timing_file = RESULTS_DIR / "reoptimized_realtime_solve_times.csv"
    for path in (results_file, timing_file):
        path.write_text("", encoding="utf-8")

    for run in range(1, args.runs + 1):
        if args.method in ("local", "both"):
            simulate_local(schedule, loads[run - 1], pvs[run - 1], t_outs[run - 1], run=run,
                           cost_penalty=args.cost_penalty, share_penalty=args.share_penalty,
                           future_inputs=args.future_inputs, results_file=results_file, timing_file=timing_file)
        if args.method in ("admm", "both"):
            simulate_rolling_admm(schedule, loads[run - 1], pvs[run - 1], t_outs[run - 1], run=run,
                                  warm_start=args.admm_start == "warm", future_inputs=args.future_inputs,
                                  results_file=results_file, timing_file=timing_file)
    print(f"Results saved to: {results_file}")
    print(f"Solve times saved to: {timing_file}")


if __name__ == "__main__":
    main()
