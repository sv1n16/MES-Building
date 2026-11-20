"""Receding-horizon centralised re-optimisation of the community.

PROBLEM   (solved at every hour h over the remaining hours t = h..T−1)

    min  Σ_b Σ_t [ π_t g_b,t Δt + c_gas G_b,t Δt + α (T_b,t − T^set_t)² + w_t g_b,t²
                 + 1e-4 Σ_j q⁺_bj,t
                 (+ Σ_j (γ_bj/2)(q⁺_bj,t + q⁻_bj,t) Δt  network charge, if OPT.NETWORK_CHARGE)
                 (+ Σ_j (κ_bj/2) π_t q_bj,t² Δt          network losses, if OPT.NETWORK_LOSSES) ]
    s.t. building model B_b for every b (as in the ADMM subproblems)
         q_bj,t + q_jb,t = 0   for every pair {b,j}      (trades agree exactly)

    i.e. the problem bilateral ADMM converges to, solved in one call.

At every interval a single central operator re-solves the whole community over
the remaining horizon and applies only the first-interval decisions; the
realised battery SOC / indoor temperature become the next solve's initial
state. It is the centralised counterpart of rolling_admm in
simulate_realtime_reoptimise.py: the model is the bilateral ADMM subproblems
stacked into one problem with the pairwise trade consensus imposed as hard
constraints, so it solves exactly the problem ADMM converges to (same
constraints, congestion term, routing regulariser and relaxed charge indicator)
in a single solver call. The P2P settlement (FEE_FRAC x grid tariff) cancels
across the community and is left out of the objective, as in ADMM.

The scenarios, remaining-horizon inputs, records and timing rows are shared with
simulate_realtime_reoptimise, so the results slot straight into
compare_realtime_methods.py.

Examples:
    python simulate_realtime_central.py --runs 1
    python simulate_realtime_central.py --runs 3 --future-inputs forecast
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import project_paths  # noqa: E402  (shared modules on sys.path; results -> Results/<experiment>/)

project_paths.register(__file__)

import argparse
import time

import numpy as np
import pyomo.environ as pyo

import simulate_noisy_realtime as baseline
import simulate_realtime_reoptimise as reopt

OPT = reopt.OPT
C = reopt.C
N = reopt.N
FULL_T = reopt.FULL_T
DT = reopt.DT

METHOD = "rolling_central"


def build_central_model():
    """All building subproblems in one model, coupled by hard trade consensus.

    Must be called inside reopt.remaining_horizon(...).
    """
    m = pyo.ConcreteModel()
    subs = []
    for b in range(N):
        sub = OPT.build_subproblem(b)
        sub.obj.deactivate()
        setattr(m, f"building_{b}", sub)
        subs.append(sub)

    # hard consensus instead of ADMM's penalty:  q_lo[hi,t] + q_hi[lo,t] = 0
    # b's copy of the trade with j (+ = b delivers) must mirror j's copy with b
    m.c_consensus = pyo.ConstraintList()
    for lo, hi in OPT.PAIRS:
        k_lo, k_hi = OPT.NB[lo].index(hi), OPT.NB[hi].index(lo)
        for t in range(OPT.T):
            m.c_consensus.add(subs[lo].q[k_lo, t] + subs[hi].q[k_hi, t] == 0)

    # Σ_t π g Δt + c_gas G Δt + α (T − T_set)² [+ w_t g²] + 1e-4 Σ q⁺   (no λ / ρ terms)
    gamma = OPT.network_charge_matrix()[0] if OPT.NETWORK_CHARGE else None  # γ_ij, GBP per kWh traded
    kappa = OPT.network_loss_matrix() if OPT.NETWORK_LOSSES else None  # κ_ij, fraction lost per kW traded

    def building_cost(b, sub):
        cost = sum(
            OPT.PRICE[t] * sub.p_el[t] * DT
            + C.gas_price / 100.0 * sub.gas[t] * DT
            + C.alpha * (sub.T_in[t] - C.T_SET[t]) ** 2
            for t in sub.t
        )
        if OPT.use_congestion:
            cost += sum(OPT.congestion_weight[t] * sub.p_el[t] ** 2 for t in sub.t)
        # same routing regulariser as the ADMM subproblems
        cost += 1e-4 * sum(sub.qpos[k, t] for k in sub.k for t in sub.t)
        if gamma is not None:  # same exogenous network charge as the ADMM subproblems (half per side)
            cost += sum(gamma[b, OPT.NB[b][k]] / 2.0 * (sub.qpos[k, t] + sub.qneg[k, t]) * DT
                        for k in sub.k for t in sub.t)
        if kappa is not None:  # same quadratic per-pair losses as the ADMM subproblems (half per side)
            cost += sum(kappa[b, OPT.NB[b][k]] / 2.0 * OPT.PRICE[t] * sub.q[k, t] ** 2 * DT
                        for k in sub.k for t in sub.t)
        return cost

    m.obj = pyo.Objective(expr=sum(building_cost(b, sub) for b, sub in enumerate(subs)), sense=pyo.minimize)
    return m, subs


def simulate_rolling_central(schedule, actual_load, actual_pv, actual_t_out, *, run,
                             future_inputs="noisy", results_file=None, timing_file=None,
                             method=METHOD, deadline=None):
    solver = reopt._make_solver(nonconvex=not OPT.RELAX_BINARIES)
    soc = reopt.DAY_START_SOC.copy()
    temperature = reopt.DAY_START_T_IN.copy()
    records, timings = [], []
    for interval in range(FULL_T):
        reopt.check_deadline(deadline, f"rolling central before interval {interval}")
        started = time.perf_counter()
        inputs = reopt.horizon_inputs(interval, actual_load, actual_pv, actual_t_out, future_inputs)
        with reopt.remaining_horizon(interval, *inputs, soc, temperature):
            m, subs = build_central_model()
            build_seconds = time.perf_counter() - started
            solve_started = time.perf_counter()
            result = solver.solve(m, tee=False)
            solver_seconds = time.perf_counter() - solve_started
            termination = result.solver.termination_condition
            if termination not in reopt._ACCEPTABLE or pyo.value(subs[0].p_el[0], exception=False) is None:
                raise RuntimeError(
                    f"Central solve failed at interval {interval}: "
                    f"status={result.solver.status}, termination={termination}"
                )
            outputs = [{name: float(pyo.value(getattr(sub, name)[0])) for name in reopt._FIRST_STEP_NAMES}
                       for sub in subs]
            flows = np.zeros((N, N))  # flows[b, j]: b -> j at the current interval
            for b, sub in enumerate(subs):
                for k, neighbour in enumerate(OPT.NB[b]):
                    flows[b, neighbour] = max(float(pyo.value(sub.qpos[k, 0])), 0.0)
        wall = time.perf_counter() - started

        for b, out in enumerate(outputs):
            out["p2p_export"] = float(flows[b].sum())
            out["p2p_import"] = float(flows[:, b].sum())
            soc[b] = out["soc"]
            temperature[b] = out["T_in"]

        step_timings = [reopt.timing_row(
            run, method, interval, "community",
            wall_seconds=wall,
            build_seconds=build_seconds,
            solver_seconds=solver_seconds,
            # one central solver: no parallelism to exploit
            parallel_latency_seconds=wall,
            solver_calls=1,
            termination=str(termination),
        )]
        step_records = [
            reopt.make_record(run, method, interval, b, actual_load[b, interval],
                              actual_pv[b, interval], actual_t_out[interval], outputs[b], schedule)
            for b in range(N)
        ]
        if results_file is not None:
            reopt.append_rows(results_file, step_records)
        if timing_file is not None:
            reopt.append_rows(timing_file, step_timings)
        records.extend(step_records)
        timings.extend(step_timings)
        print(
            f"  run {run} central  interval {interval + 1:02d}/{FULL_T}: "
            f"build {build_seconds:5.2f}s, solve {solver_seconds:5.2f}s, "
            f"grid {sum(o['p_el'] for o in outputs):6.2f} kW",
            flush=True,
        )
    return records, timings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--load-noise", type=float, default=baseline.DEFAULT_LOAD_NOISE)
    parser.add_argument("--pv-noise", type=float, default=0.30)
    parser.add_argument("--temp-noise", type=float, default=1.0, help="Outdoor-temperature noise std (deg C).")
    parser.add_argument("--future-inputs", choices=("noisy", "forecast"), default="noisy",
                        help="Remaining-horizon inputs: the noisy trajectory, or forecast after the current interval.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--refresh-scenarios", action="store_true")
    args = parser.parse_args()
    if args.runs < 1 or min(args.load_noise, args.pv_noise, args.temp_noise) < 0:
        parser.error("--runs must be positive and noise values non-negative.")

    reopt.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    schedule = baseline.load_or_solve_schedule(reopt.SCHEDULE_CACHE, force_refresh=args.refresh_cache)
    loads, pvs, t_outs = baseline.load_or_create_scenarios(
        reopt.SCENARIO_FILE, runs=args.runs, load_noise=args.load_noise, pv_noise=args.pv_noise,
        temp_noise=args.temp_noise, seed=args.seed, force_refresh=args.refresh_scenarios,
    )
    results_file = reopt.RESULTS_DIR / "central_realtime_simulation.csv"
    timing_file = reopt.RESULTS_DIR / "central_realtime_solve_times.csv"
    for path in (results_file, timing_file):
        path.write_text("", encoding="utf-8")

    for run in range(1, args.runs + 1):
        simulate_rolling_central(schedule, loads[run - 1], pvs[run - 1], t_outs[run - 1], run=run,
                                 future_inputs=args.future_inputs,
                                 results_file=results_file, timing_file=timing_file)
    print(f"Results saved to: {results_file}")
    print(f"Solve times saved to: {timing_file}")


if __name__ == "__main__":
    main()
