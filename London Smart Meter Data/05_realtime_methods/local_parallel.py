"""Parallel local re-optimisation: the counterpart of the parallel ADMM for a fair comparison.

Same method as simulate_realtime_reoptimise.simulate_local (each building re-solves
its remaining horizon on its own, planned trades as upper bounds, P2P settlement and
share penalty, trades reconciled at min(seller offer, buyer acceptance)), with the
same two speed-ups as admm_parallel:

  * the buildings are spread over --workers persistent processes, one Gurobi
    thread each;
  * the models are built directly in gurobipy instead of through Pyomo.

The Pyomo version also carries the two |cost - planned cost| constraints of the
(commented-out) cost penalty. They do not change the solution (their deviation
variable is free and not in the objective) but make Gurobi treat the model as
non-convex; they are left out here.

Timing columns as for local_reoptimisation: wall_seconds is the measured time
with the worker pool on this machine; parallel_latency_seconds is the slowest
building (one controller per building).

Examples:
    python local_parallel.py --runs 1 --workers 8
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

from admm_parallel import DEFAULT_WORKERS, ParallelADMM

METHOD = "local_reoptimisation_parallel"


def simulate_local_parallel(schedule, actual_load, actual_pv, actual_t_out, *, run, share_penalty=0.1,
                            workers=DEFAULT_WORKERS, future_inputs="noisy", results_file=None, timing_file=None,
                            method=METHOD, deadline=None):
    import simulate_realtime_reoptimise as reopt

    OPT, N, FULL_T = reopt.OPT, reopt.N, reopt.FULL_T
    soc = reopt.DAY_START_SOC.copy()
    temperature = reopt.DAY_START_T_IN.copy()
    records, timings = [], []
    with ParallelADMM(N, workers) as pool:
        for interval in range(FULL_T):
            reopt.check_deadline(deadline, f"parallel local re-optimisation before interval {interval}")
            step_started = time.perf_counter()
            planned = [schedule["trades"][b][OPT.NB[b]][:, interval:] for b in range(N)]
            inputs = reopt.horizon_inputs(interval, actual_load, actual_pv, actual_t_out, future_inputs)
            with reopt.remaining_horizon(interval, *inputs, soc, temperature):
                solved = pool.local_step(planned, share_penalty, reopt._FIRST_STEP_NAMES)
            step_wall = time.perf_counter() - step_started

            outputs, step_timings = [], []
            for b, values in enumerate(solved):
                offer, accept = np.zeros(N), np.zeros(N)
                for k, neighbour in enumerate(OPT.NB[b]):
                    offer[neighbour] = max(float(values["offer"][k]), 0.0)
                    accept[neighbour] = max(float(values["accept"][k]), 0.0)
                out = {name: values[name] for name in reopt._FIRST_STEP_NAMES}
                out.update(offer=offer, accept=accept)
                outputs.append(out)
                soc[b] = out["soc"]
                temperature[b] = out["T_in"]
                step_timings.append(reopt.timing_row(
                    run, method, interval, "building", building=reopt.C.BUILDING_IDS[b],
                    wall_seconds=values["wall_seconds"], build_seconds=values["build_seconds"],
                    solver_seconds=values["solver_seconds"], termination="optimal",
                ))

            finals = reopt._reconcile_trades(outputs)
            step_records = [
                reopt.make_record(run, method, interval, b, actual_load[b, interval], actual_pv[b, interval],
                                  actual_t_out[interval], finals[b], schedule)
                for b in range(N)
            ]
            step_timings.append(reopt.timing_row(
                run, method, interval, "community",
                wall_seconds=step_wall,
                build_seconds=sum(row["build_seconds"] for row in step_timings),
                solver_seconds=sum(row["solver_seconds"] for row in step_timings),
                # buildings solve in parallel on their own controllers
                parallel_latency_seconds=max(row["wall_seconds"] for row in step_timings),
                solver_calls=N,
                termination=f"all_buildings_solved_{len(pool.assignment)}_workers",
            ))
            if results_file is not None:
                reopt.append_rows(results_file, step_records)
            if timing_file is not None:
                reopt.append_rows(timing_file, step_timings)
            records.extend(step_records)
            timings.extend(step_timings)
            print(
                f"  run {run} local-par interval {interval + 1:02d}/{FULL_T}: wall {step_wall:6.2f}s "
                f"({len(pool.assignment)} workers), per-building parallel {step_timings[-1]['parallel_latency_seconds']:5.2f}s, "
                f"grid {sum(f['p_el'] for f in finals):6.2f} kW",
                flush=True,
            )
    return records, timings


def main():
    import simulate_noisy_realtime as baseline
    import simulate_realtime_reoptimise as reopt

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--load-noise", type=float, default=baseline.DEFAULT_LOAD_NOISE)
    parser.add_argument("--pv-noise", type=float, default=0.30)
    parser.add_argument("--temp-noise", type=float, default=1.0)
    parser.add_argument("--future-inputs", choices=("noisy", "forecast"), default="noisy")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--share-penalty", type=float, default=0.1, help="GBP per kWh of planned P2P not delivered.")
    args = parser.parse_args()

    reopt.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    schedule = baseline.load_or_solve_schedule(reopt.SCHEDULE_CACHE)
    loads, pvs, t_outs = baseline.load_or_create_scenarios(
        reopt.SCENARIO_FILE, runs=args.runs, load_noise=args.load_noise, pv_noise=args.pv_noise,
        temp_noise=args.temp_noise, seed=args.seed,
    )
    results_file = reopt.RESULTS_DIR / "parallel_local_realtime_simulation.csv"
    timing_file = reopt.RESULTS_DIR / "parallel_local_realtime_solve_times.csv"
    for path in (results_file, timing_file):
        path.write_text("", encoding="utf-8")
    for run in range(1, args.runs + 1):
        simulate_local_parallel(schedule, loads[run - 1], pvs[run - 1], t_outs[run - 1], run=run,
                                share_penalty=args.share_penalty, workers=args.workers,
                                future_inputs=args.future_inputs, results_file=results_file, timing_file=timing_file)
    print(f"Results saved to: {results_file}\nSolve times saved to: {timing_file}")


if __name__ == "__main__":
    main()
