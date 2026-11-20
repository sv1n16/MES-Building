"""Run the real-time comparison for several community sizes, with a compute budget.

PURPOSE
    Do the real-time methods still fit their time step as the community grows
    (N = 5 … 1000)? Driver only: runs compare_realtime_methods per N under a wall-time
    budget and records which steps finish.

For every N it:
  1. builds data/showcase_<day>_N<NNNN>.{csv,_batteries.csv} if missing (N=10 uses
     the default showcase dataset), with 01_data_preparation/build_showcase_dataset.py;
  2. runs each step in its own subprocess with SHOWCASE_DATASET / SHOWCASE_PLOTS_SUBDIR
     pointing at that size, so every output lands in
         Results/06_realtime_scaling/N<NNNN>/
     steps: day-ahead ADMM schedule, then each method of compare_realtime_methods;
  3. gives each step a wall-time budget. Inside the step the budget is checked
     before every interval and after every ADMM subproblem solve, and the central
     solve gets a Gurobi time limit; a step that runs out stops there and keeps
     the intervals it already wrote. As a backstop the driver kills a step that
     overruns the budget by --grace seconds (e.g. stuck building a huge model) or
     exhausts memory. Each step is flagged in
         Results/06_realtime_scaling/scaling_status.csv
     with status completed / budget_exceeded (did not converge or finish in time) /
     killed_timeout / killed_memory / failed / skipped_no_plan, the elapsed time
     and the number of intervals completed;
  4. summarises whatever was completed (compare_realtime_methods.summarise_run) into
         Results/06_realtime_scaling/scaling_summary.csv

Without a converged day-ahead schedule there is no plan to follow or deviate from,
so the methods for that N are skipped.

--parallel solves the day-ahead plan with admm_parallel (step schedule_parallel;
when a plan already exists the parallel solve is still timed, into
bilateral_admm_schedule_parallel.npz) and adds the rolling_admm_parallel method.
--keep-results keeps finished steps and only runs steps that are new, skipped or
failed, so the parallel ADMM can be added to an earlier sequential run.

Examples:
    python scale_realtime_methods.py
    python scale_realtime_methods.py --sizes 5 10 20 --budget 600
    python scale_realtime_methods.py --parallel --keep-results --workers 8
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import project_paths  # noqa: E402  (shared modules on sys.path; results -> Results/<experiment>/)

project_paths.register(__file__)

import argparse
import json
import os
import subprocess
import time

import pandas as pd
import psutil

ROOT = project_paths.ROOT
DATA_DIR = project_paths.DATA_DIR
BUILDER = ROOT / "01_data_preparation" / "build_showcase_dataset.py"
SCALING_DIR = project_paths.experiment_dir("06_realtime_scaling")
STATUS_FILE = SCALING_DIR / "scaling_status.csv"
SUMMARY_FILE = SCALING_DIR / "scaling_summary.csv"
DAY = (DATA_DIR / "showcase_latest.txt").read_text().strip() if (DATA_DIR / "showcase_latest.txt").exists() else "2013-02-21"
METHODS = ("open_loop", "fixed_schedule", "local_reoptimisation", "rolling_admm", "rolling_central")
PARALLEL_METHODS = ("local_reoptimisation_parallel", "rolling_admm_parallel")
RETRY_STATUSES = {None, "skipped_no_plan", "failed"}  # --keep-results re-runs these (None = never run)
PARALLEL_SCHEDULE_FILE = "bilateral_admm_schedule_parallel.npz"
# Kill a step only when it is itself the memory hog: it holds more than MEMORY_SHARE of
# system memory, or more than MEMORY_MIN_GB while less than MEMORY_FLOOR is free.
# (Other applications alone can leave little memory free; that is not the step's fault.)
MEMORY_SHARE = 0.6
MEMORY_MIN_GB = 4.0  # below this Windows can page the step out; it gets slower, not stuck
MEMORY_FLOOR = 0.03


def _stem(n):
    return f"showcase_{DAY}_N{n:04d}"


def _results_dir(n):
    return SCALING_DIR / f"N{n:04d}"


# ============================================================================
# WORKER (one step, inside a subprocess pointed at one community size)
# ============================================================================
def _scenario_args():
    import simulate_noisy_realtime as baseline

    return argparse.Namespace(
        load_noise=baseline.DEFAULT_LOAD_NOISE, pv_noise=0.30, temp_noise=1.0, seed=2026,
        load_bias=0.0, bias_hours="16-22", pv_bias=0.0, correlation=0.0,
    )


def _run_step(step, runs, deadline, budget, workers=None):
    import admm_parallel as par
    import compare_realtime_methods as compare
    import simulate_noisy_realtime as baseline
    import simulate_open_loop as open_loop
    import simulate_realtime_central as central
    import simulate_realtime_reoptimise as reopt

    info = {}
    args = _scenario_args()
    if step == "schedule":
        original = reopt.OPT.run_admm
        log = reopt.DeadlineLog(deadline, "day-ahead ADMM")
        reopt.OPT.run_admm = lambda **kwargs: original(solve_log=log, verbose=False, **kwargs)
        try:
            baseline.load_or_solve_schedule(reopt.SCHEDULE_CACHE)
        finally:
            info["admm_iterations"] = int(max(i for i, _b, _s in log)) + 1 if log else 0
        compare._scenarios(args, runs)
        return info

    if step == "schedule_parallel":
        # always times a parallel solve: it becomes the plan when there is none yet,
        # otherwise it goes to a side file and the existing plan is kept
        workers = workers or par.DEFAULT_WORKERS
        plan_exists = par.load_cached_schedule(reopt.SCHEDULE_CACHE) is not None
        target = reopt.SCHEDULE_CACHE.with_name(PARALLEL_SCHEDULE_FILE) if plan_exists else reopt.SCHEDULE_CACHE
        started = time.perf_counter()
        _schedule, iterations = par.solve_schedule_parallel(target, workers=workers, deadline=deadline,
                                                            force_refresh=True)
        notes = [f"timing copy in {target.name}"] if plan_exists else []
        if par.needs_low_memory(reopt.N):
            notes.append("low-memory mode (models rebuilt every iteration)")
        info.update(admm_iterations=iterations, solve_seconds=time.perf_counter() - started, workers=workers,
                    note="; ".join(notes))
        compare._scenarios(args, runs)
        return info

    if step == "summarise":
        data = pd.read_csv(compare.RESULTS_FILE)
        timings = pd.read_csv(compare.TIMING_FILE)
        rows = []
        for (run, method), records in data.groupby(["simulation_run", "method"]):
            steps = timings[(timings.simulation_run == run) & (timings.method == method)]
            row = compare.summarise_run(records.to_dict("records"), steps.to_dict("records"), run, method)
            row["intervals_completed"] = int(records["interval"].nunique())
            rows.append(row)
        pd.DataFrame(rows).to_csv(compare.SUMMARY_FILE, index=False)
        return info

    if not reopt.SCHEDULE_CACHE.exists():
        raise RuntimeError("no day-ahead schedule")
    schedule = baseline.load_or_solve_schedule(reopt.SCHEDULE_CACHE)
    loads, pvs, t_outs = compare._scenarios(args, runs)
    if step == "rolling_central":
        # set after loading the schedule: the time limit is part of the schedule signature
        compare.C.SOLVE_TIME_LIMIT = budget
    files = {"results_file": compare.RESULTS_FILE, "timing_file": compare.TIMING_FILE}
    for run in range(1, runs + 1):
        inputs = (schedule, loads[run - 1], pvs[run - 1], t_outs[run - 1])
        if step == "open_loop":
            open_loop.simulate_open_loop(*inputs, run=run, **files)
        elif step == "fixed_schedule":
            compare.simulate_fixed(*inputs, run=run)
        elif step == "local_reoptimisation":
            reopt.simulate_local(*inputs, run=run, cost_penalty=10.0, share_penalty=0.1, deadline=deadline, **files)
        elif step == "rolling_admm":
            reopt.simulate_rolling_admm(*inputs, run=run, warm_start=True, deadline=deadline, **files)
        elif step == "local_reoptimisation_parallel":
            import local_parallel

            local_parallel.simulate_local_parallel(*inputs, run=run, share_penalty=0.1,
                                                   workers=workers or par.DEFAULT_WORKERS, deadline=deadline, **files)
        elif step == "rolling_admm_parallel":
            par.simulate_rolling_admm_parallel(*inputs, run=run, workers=workers or par.DEFAULT_WORKERS,
                                               warm_start=True, deadline=deadline, **files)
        elif step == "rolling_central":
            _records, steps = central.simulate_rolling_central(*inputs, run=run, deadline=deadline, **files)
            if any(row["termination"] == "maxTimeLimit" for row in steps):
                raise reopt.BudgetExceeded("rolling central hit the Gurobi time limit (not proven optimal)")
        else:
            raise ValueError(f"unknown step {step!r}")
    return info


def worker(step, runs, budget, status_path, workers=None):
    import simulate_realtime_reoptimise as reopt

    started = time.perf_counter()
    status = {"status": "completed", "note": ""}
    try:
        status.update(_run_step(step, runs, started + budget, budget, workers))
    except reopt.BudgetExceeded as error:
        status.update(status="budget_exceeded", note=str(error))
    except MemoryError:
        status.update(status="killed_memory", note="MemoryError")
    except Exception as error:  # noqa: BLE001 - recorded in the status file
        status.update(status="failed", note=repr(error)[:500])
    status["elapsed_seconds"] = time.perf_counter() - started
    Path(status_path).write_text(json.dumps(status), encoding="utf-8")
    print(f"[{step}] {status['status']} after {status['elapsed_seconds']:.0f}s {status['note']}", flush=True)


# ============================================================================
# DRIVER
# ============================================================================
def ensure_dataset(n):
    if n == 10:
        return
    target = DATA_DIR / f"{_stem(n)}.csv"
    if target.exists() and (DATA_DIR / f"{_stem(n)}_batteries.csv").exists():
        return
    print(f"  building dataset for N={n} ...", flush=True)
    # the dataset goes to data/; its input figure (10) to Results/06_realtime_scaling/N<NNNN>/
    env = dict(os.environ, MES_EXPERIMENT="06_realtime_scaling", SHOWCASE_PLOTS_SUBDIR=f"N{n:04d}")
    subprocess.run([sys.executable, str(BUILDER), "--n", str(n), "--day", DAY], env=env, check=True)


def _kill(process):
    try:
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass
    process.wait()


def run_step(n, step, runs, budget, grace, log, workers=None):
    env = dict(os.environ, SHOWCASE_PLOTS_SUBDIR=f"N{n:04d}", PYTHONUNBUFFERED="1")
    if n == 10:
        env.pop("SHOWCASE_DATASET", None)
    else:
        env["SHOWCASE_DATASET"] = _stem(n)
    status_path = _results_dir(n) / f".status_{step}.json"
    status_path.unlink(missing_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", step, "--runs", str(runs),
               "--budget", str(budget), "--status-file", str(status_path)]
    if workers:
        command += ["--workers", str(workers)]
    started = time.perf_counter()
    process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    killed = None
    peak_rss = 0.0
    while process.poll() is None:
        time.sleep(2.0)
        try:
            proc = psutil.Process(process.pid)
            rss = proc.memory_info().rss + sum(c.memory_info().rss for c in proc.children(recursive=True))
            peak_rss = max(peak_rss, rss)
        except psutil.NoSuchProcess:
            break
        memory = psutil.virtual_memory()
        if rss > MEMORY_SHARE * memory.total or (
            rss > MEMORY_MIN_GB * 2**30 and memory.available < MEMORY_FLOOR * memory.total
        ):
            killed = ("killed_memory", f"killed using {rss / 2**30:.1f} GB with "
                                       f"{memory.available / 2**30:.1f} GB of memory free")
        elif time.perf_counter() - started > budget + grace:
            killed = ("killed_timeout", f"killed after budget {budget:.0f}s + grace {grace:.0f}s")
        if killed:
            _kill(process)
            break
    elapsed = time.perf_counter() - started
    if killed:
        status = {"status": killed[0], "note": killed[1], "elapsed_seconds": elapsed}
    elif status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
    else:
        status = {"status": "failed", "note": f"exit code {process.returncode}, no status written",
                  "elapsed_seconds": elapsed}
    status["peak_memory_gb"] = peak_rss / 2**30
    return status


def _intervals_completed(n, method):
    path = _results_dir(n) / "34_three_method_solve_times.csv"
    if not (path.exists() and path.stat().st_size):
        return 0
    timings = pd.read_csv(path)
    steps = timings[(timings["method"] == method) & (timings["scope"] == "community")]
    return int(len(steps))


def _drop_method_rows(n, methods):
    """Remove earlier rows of the methods about to be re-run from the results and timings."""
    for name in ("33_three_method_results.csv", "34_three_method_solve_times.csv"):
        path = _results_dir(n) / name
        if path.exists() and path.stat().st_size:
            table = pd.read_csv(path)
            table[~table["method"].isin(methods)].to_csv(path, index=False)


def _write_status(rows):
    SCALING_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(STATUS_FILE, index=False)


def _collect_summaries(sizes, status_rows):
    status = pd.DataFrame(status_rows)
    tables = []
    for n in sizes:
        path = _results_dir(n) / "35_three_method_summary.csv"
        if path.exists() and path.stat().st_size:
            tables.append(pd.read_csv(path).assign(n_buildings=n))
    if not tables:
        return
    summary = pd.concat(tables, ignore_index=True)
    flags = status.rename(columns={"step": "method"})[["n_buildings", "method", "status", "elapsed_seconds", "note"]]
    summary = summary.merge(flags, on=["n_buildings", "method"], how="left")
    first = ["n_buildings", "method", "status", "intervals_completed", "elapsed_seconds"]
    summary[first + [c for c in summary.columns if c not in first]].to_csv(SUMMARY_FILE, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 10, 20, 50, 100, 1000])
    parser.add_argument("--runs", type=int, default=1, help="Noise scenarios per size.")
    parser.add_argument("--budget", type=float, default=3600.0, help="Wall-time budget per method and size (s).")
    parser.add_argument("--schedule-budget", type=float, default=7200.0, help="Budget for the day-ahead ADMM (s).")
    parser.add_argument("--grace", type=float, default=600.0, help="Overrun allowed before the driver kills a step (s).")
    parser.add_argument("--parallel", action="store_true",
                        help="Solve the day-ahead plan with admm_parallel and add the rolling_admm_parallel method.")
    parser.add_argument("--workers", type=int, default=None, help="Worker processes for the parallel ADMM.")
    parser.add_argument("--keep-results", action="store_true",
                        help="Keep finished steps; only run steps that are new, skipped or failed.")
    parser.add_argument("--retry", nargs="*", default=[],
                        help="With --keep-results, also re-run steps with these statuses (e.g. killed_memory).")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--status-file", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.runs, args.budget, args.status_file, args.workers)
        return

    status_rows = pd.read_csv(STATUS_FILE).to_dict("records") if STATUS_FILE.exists() else []
    previous = {(row["n_buildings"], row["step"]): row["status"] for row in status_rows}
    if not args.keep_results:
        status_rows = [row for row in status_rows if row["n_buildings"] not in args.sizes]
    schedule_step = "schedule_parallel" if args.parallel else "schedule"
    methods = METHODS + (PARALLEL_METHODS if args.parallel else ())
    for n in args.sizes:
        print(f"\n{'=' * 70}\nN = {n}  ->  {_results_dir(n)}\n{'=' * 70}", flush=True)
        try:
            ensure_dataset(n)
        except Exception as error:  # noqa: BLE001
            status_rows.append({"n_buildings": n, "step": "dataset", "status": "failed", "note": repr(error)[:500]})
            _write_status(status_rows)
            continue
        out_dir = _results_dir(n)
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.keep_results:
            retry = RETRY_STATUSES | set(args.retry)
            steps = [step for step in methods if previous.get((n, step)) in retry]
        else:
            steps = list(methods)
            for name in ("33_three_method_results.csv", "34_three_method_solve_times.csv", "35_three_method_summary.csv"):
                (out_dir / name).write_text("", encoding="utf-8")
        _drop_method_rows(n, steps)
        # with --keep-results a day-ahead solve that already completed is reused, not re-timed
        reuse_plan = args.keep_results and previous.get((n, schedule_step)) == "completed"
        if not steps and reuse_plan:
            continue
        with (out_dir / "run_log.txt").open("a" if args.keep_results else "w", encoding="utf-8") as log:
            have_plan = reuse_plan
            for step in ([] if reuse_plan else [schedule_step]) + steps:
                if step != schedule_step and not have_plan:
                    status = {"status": "skipped_no_plan", "note": "day-ahead ADMM did not finish", "elapsed_seconds": 0.0}
                else:
                    budget = args.schedule_budget if step == schedule_step else args.budget
                    status = run_step(n, step, args.runs, budget, args.grace, log, args.workers)
                if step == schedule_step:
                    have_plan = status["status"] == "completed"
                    status["intervals_completed"] = ""
                else:
                    status["intervals_completed"] = _intervals_completed(n, step)
                status_rows = [row for row in status_rows if (row["n_buildings"], row["step"]) != (n, step)]
                status_rows.append({"n_buildings": n, "step": step, **status})
                _write_status(status_rows)
                print(f"  N={n:<5} {step:<29} {status['status']:<16} {status['elapsed_seconds']:8.0f}s "
                      f"intervals {status['intervals_completed']!s:>3}  {status.get('note', '')}", flush=True)
            if have_plan:
                run_step(n, "summarise", args.runs, args.budget, args.grace, log)
        _collect_summaries(args.sizes, status_rows)
    print(f"\nStatus: {STATUS_FILE}\nSummary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
