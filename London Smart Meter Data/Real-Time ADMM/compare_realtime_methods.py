"""Paired comparison of three real-time strategies on identical noisy scenarios.

0. open_loop             Day-ahead plan applied exactly as scheduled with no
                         real-time feedback; all forecast error goes to the
                         grid (simulate_open_loop.py).
1. fixed_schedule        Day-ahead ADMM plan followed as the reference, with only the
                         forecast error absorbed (simulate_noisy_realtime rules).
2. local_reoptimisation  Each building re-optimises its remaining horizon on its
                         own (no new sharing), penalised for deviating from its
                         original planned cost and from its planned P2P trades.
3. rolling_admm          Bilateral ADMM re-run over the remaining horizon every
                         interval.

All three read the same scenario (load, PV and outdoor temperature) from
shared_noise_scenarios.npz and step through it interval by interval. The
re-optimising methods use the noisy trajectory as their input data for the whole
remaining horizon (--future-inputs forecast keeps the day-ahead forecast beyond
the current interval instead). Outputs in
Results/ADMM Optimisation Real-time:
    33_three_method_results.csv      one row per run/method/interval/building
    34_three_method_solve_times.csv  per-solve timings (building / ADMM iteration / community)
    35_three_method_summary.csv      per run and method totals
    36_three_method_summary_mean.csv mean and std over runs
Run plot_realtime_comparison.py afterwards for the figures.

Examples:
    python compare_realtime_methods.py --runs 3
    python compare_realtime_methods.py --runs 1 --admm-start cold
"""

from __future__ import annotations

import argparse
import atexit
import os
import time

import numpy as np
import pandas as pd
import psutil

import simulate_noisy_realtime as baseline
import simulate_open_loop as open_loop
import simulate_realtime_reoptimise as reopt

C = reopt.C
N, T = reopt.N, reopt.FULL_T
RESULTS_DIR = reopt.RESULTS_DIR
RESULTS_FILE = RESULTS_DIR / "33_three_method_results.csv"
TIMING_FILE = RESULTS_DIR / "34_three_method_solve_times.csv"
SUMMARY_FILE = RESULTS_DIR / "35_three_method_summary.csv"
MEAN_SUMMARY_FILE = RESULTS_DIR / "36_three_method_summary_mean.csv"
LOCK_FILE = RESULTS_DIR / ".compare_realtime_methods.lock"
SCENARIO_FILE = reopt.SCENARIO_FILE

# Sensitivity cases. Each writes to Results/ADMM Optimisation Real-time/cases/<name>/.
CASES = {
    "base": {"description": "Independent, unbiased noise; re-optimisers see the noisy future", "args": {}},
    "biased_evening": {
        "description": "Forecast underestimates load by 15% from 16:00 to 22:00",
        "args": {"load_bias": 0.15, "bias_hours": "16-22"},
    },
    "correlated": {
        "description": "80% of load and PV error shared by all buildings",
        "args": {"correlation": 0.8},
    },
    "forecast_inputs": {
        "description": "Re-optimisers see the measured current hour and the forecast after it",
        "args": {"future_inputs": "forecast"},
        "base_scenarios": True,  # same noise as the base case, only the information differs
    },
}


def _apply_case(args):
    """Apply a case's settings and point every output (and its scenarios) at its folder."""
    global RESULTS_DIR, RESULTS_FILE, TIMING_FILE, SUMMARY_FILE, MEAN_SUMMARY_FILE, LOCK_FILE, SCENARIO_FILE
    case = CASES[args.case]
    for name, value in case["args"].items():
        setattr(args, name, value)
    if args.case == "base":
        return
    RESULTS_DIR = reopt.RESULTS_DIR / "cases" / args.case
    RESULTS_FILE = RESULTS_DIR / RESULTS_FILE.name
    TIMING_FILE = RESULTS_DIR / TIMING_FILE.name
    SUMMARY_FILE = RESULTS_DIR / SUMMARY_FILE.name
    MEAN_SUMMARY_FILE = RESULTS_DIR / MEAN_SUMMARY_FILE.name
    LOCK_FILE = RESULTS_DIR / LOCK_FILE.name
    if not case.get("base_scenarios"):
        SCENARIO_FILE = RESULTS_DIR / baseline.SCENARIO_FILE_NAME
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_bias(args):
    """Per-hour load bias (fraction) from --load-bias over --bias-hours 'start-end'."""
    bias = np.zeros(reopt.FULL_T)
    if args.load_bias:
        start, end = (int(hour) for hour in args.bias_hours.split("-"))
        bias[start:end] = args.load_bias
    return bias


def _scenarios(args, runs, force_refresh=False):
    return baseline.load_or_create_scenarios(
        SCENARIO_FILE, runs=runs, load_noise=args.load_noise, pv_noise=args.pv_noise,
        temp_noise=args.temp_noise, seed=args.seed, force_refresh=force_refresh,
        load_bias=_load_bias(args), correlation=args.correlation,
    )


def _acquire_results_lock():
    """Refuse to start while another comparison is writing the same result files.

    A lock left behind by a killed run is ignored once its process has exited.
    """
    if LOCK_FILE.exists():
        try:
            owner = int(LOCK_FILE.read_text(encoding="utf-8").strip())
        except ValueError:
            owner = None
        if owner is not None and owner != os.getpid() and psutil.pid_exists(owner):
            raise SystemExit(
                f"Another compare_realtime_methods.py run (PID {owner}) is writing {RESULTS_DIR}. "
                f"Wait for it to finish or stop it; delete {LOCK_FILE.name} only if that process is gone."
            )
    LOCK_FILE.write_text(str(os.getpid()), encoding="utf-8")
    atexit.register(lambda: LOCK_FILE.unlink(missing_ok=True))
METHODS = ("open_loop", "fixed_schedule", "local_reoptimisation", "rolling_admm")


def simulate_fixed(schedule, actual_load, actual_pv, actual_t_out, *, run, write=True):
    step_seconds = []
    rows = baseline.simulate(
        schedule, load_noise=0.0, pv_noise=0.0, pace_seconds=0.0,
        actual_load=actual_load, actual_pv=actual_pv, actual_t_out=actual_t_out,
        step_timings=step_seconds,
    )
    planned = schedule["variables"]
    records, timings = [], []
    for interval in range(T):
        for building in range(N):
            row = rows[interval * N + building]
            final = {
                "charge": row["battery_charge_kw"],
                "discharge": row["battery_discharge_kw"],
                "soc": row["battery_soc_kwh"],
                "p_el": row["actual_grid_import_kw"],
                "curtail": row["unused_pv_kw"],
                "p_hp": row["heat_pump_kw"],
                "q_boiler": float(planned["q_boiler"][building, interval]),
                "gas": float(planned["gas"][building, interval]),
                "T_in": row["final_T_in"],
                "p2p_import": row["p2p_import_kw"],
                "p2p_export": row["p2p_export_kw"],
            }
            records.append(reopt.make_record(
                run, "fixed_schedule", interval, building, actual_load[building, interval],
                actual_pv[building, interval], actual_t_out[interval], final, schedule,
            ))
        timings.append(reopt.timing_row(
            run, "fixed_schedule", interval, "community",
            wall_seconds=step_seconds[interval], solver_calls=0, termination="rule_based_dispatch",
        ))
    if write:
        reopt.append_rows(RESULTS_FILE, records)
        reopt.append_rows(TIMING_FILE, timings)
    return records, timings


SUSPEND_FACTOR = 20.0


def clean_suspended_timings(timings):
    """Remove machine sleep/lock pauses from ADMM timings.

    An ADMM iteration more than SUSPEND_FACTOR times slower than the median
    iteration of the same run/method/interval is taken to include a pause
    (Windows keeps the clock running while a locked or sleeping machine
    suspends the process). Its time is replaced by that median, the excess is
    removed from the interval's community row, and both rows are flagged in a
    `suspended` column. Local building solves are left as measured: a slow
    non-convex solve there can be genuine.
    """
    t = pd.DataFrame(timings).copy()
    t["suspended"] = False
    iterations = t["scope"] == "admm_iteration"
    if not iterations.any():
        return t
    keys = ["simulation_run", "method", "interval"]
    grouped = t[iterations].groupby(keys)
    median_wall = grouped["wall_seconds"].transform("median")
    median_latency = grouped["parallel_latency_seconds"].transform("median")
    flagged = t[iterations]["wall_seconds"] > SUSPEND_FACTOR * median_wall
    for index in flagged[flagged].index:
        excess_wall = t.at[index, "wall_seconds"] - median_wall[index]
        excess_latency = t.at[index, "parallel_latency_seconds"] - median_latency[index]
        t.loc[index, ["wall_seconds", "solver_seconds"]] = median_wall[index]
        t.at[index, "parallel_latency_seconds"] = median_latency[index]
        t.at[index, "suspended"] = True
        community = (t["scope"] == "community") & (t[keys] == t.loc[index, keys]).all(axis=1)
        t.loc[community, "wall_seconds"] -= excess_wall
        t.loc[community, "solver_seconds"] -= excess_wall
        t.loc[community, "parallel_latency_seconds"] -= excess_latency
        t.loc[community, "suspended"] = True
    t["latency_fraction_of_interval"] = t["parallel_latency_seconds"] / t["interval_seconds"]
    t["realtime_feasible"] = t["parallel_latency_seconds"] < t["interval_seconds"]
    return t


def summarise_run(records, timings, run, method):
    data = pd.DataFrame(records)
    community = data.groupby("interval").sum(numeric_only=True)
    all_steps = clean_suspended_timings(timings)
    steps = all_steps[all_steps["scope"] == "community"]
    row = {"simulation_run": run, "method": method}
    for name in ("electricity_cost", "gas_cost", "comfort_cost", "p2p_settlement", "total_cost"):
        row[f"planned_{name}_gbp"] = float(data[f"planned_{name}"].sum())
        row[f"final_{name}_gbp"] = float(data[f"final_{name}"].sum())
    row["cost_change_vs_plan_gbp"] = row["final_total_cost_gbp"] - row["planned_total_cost_gbp"]
    row["planned_grid_import_kwh"] = float(community["planned_p_el"].sum() * reopt.DT)
    row["final_grid_import_kwh"] = float(community["final_p_el"].sum() * reopt.DT)
    row["grid_import_abs_deviation_kwh"] = float(np.abs(community["deviation_p_el"]).sum() * reopt.DT)
    row["planned_peak_grid_kw"] = float(community["planned_p_el"].max())
    row["final_peak_grid_kw"] = float(community["final_p_el"].max())
    row["planned_p2p_kwh"] = float(community["planned_p2p_export"].sum() * reopt.DT)
    row["final_p2p_kwh"] = float(community["final_p2p_export"].sum() * reopt.DT)
    row["curtailed_pv_kwh"] = float(community["final_curtail"].sum() * reopt.DT)
    row["mean_abs_temperature_deviation_c"] = float(np.abs(data["deviation_T_in"]).mean())
    row["total_wall_seconds"] = float(steps["wall_seconds"].sum())
    row["mean_step_wall_seconds"] = float(steps["wall_seconds"].mean())
    row["max_step_wall_seconds"] = float(steps["wall_seconds"].max())
    row["mean_step_parallel_latency_seconds"] = float(steps["parallel_latency_seconds"].mean())
    row["max_step_parallel_latency_seconds"] = float(steps["parallel_latency_seconds"].max())
    row["all_steps_realtime_feasible"] = bool(steps["realtime_feasible"].all())
    iterations = pd.to_numeric(steps["admm_iterations"], errors="coerce")
    row["mean_admm_iterations"] = float(iterations.mean()) if iterations.notna().any() else np.nan
    row["max_admm_iterations"] = float(iterations.max()) if iterations.notna().any() else np.nan
    row["suspended_iterations_removed"] = int(
        (all_steps["suspended"] & (all_steps["scope"] == "admm_iteration")).sum()
    )
    return row


def _write_mean_summary(table):
    numeric = table.drop(columns=["simulation_run"]).select_dtypes("number").columns
    mean = table.groupby("method")[list(numeric)].agg(["mean", "std"])
    mean.columns = [f"{name}_{stat}" for name, stat in mean.columns]
    mean.reindex(METHODS).to_csv(MEAN_SUMMARY_FILE)


DISPATCH_ONLY = ("open_loop", "fixed_schedule")


def refresh_dispatch_only(args):
    """Re-simulate the methods that need no solver and resummarise.

    Open loop and the fixed schedule only dispatch the cached plan, so after a
    rule change they can be recomputed on the saved scenarios in seconds while
    the re-optimisation results are kept.
    """
    data = pd.read_csv(RESULTS_FILE)
    timings = pd.read_csv(TIMING_FILE)
    runs = sorted(data["simulation_run"].unique())
    schedule = baseline.load_or_solve_schedule(reopt.SCHEDULE_CACHE)
    loads, pvs, t_outs = _scenarios(args, max(runs))
    new_records, new_timings = [], []
    for run in runs:
        inputs = (schedule, loads[run - 1], pvs[run - 1], t_outs[run - 1])
        for simulate in (open_loop.simulate_open_loop, simulate_fixed):
            records, steps = simulate(*inputs, run=run, **({"write": False} if simulate is simulate_fixed else {}))
            new_records.extend(records)
            new_timings.extend(steps)
    keep = ~data.method.isin(DISPATCH_ONLY)
    data = pd.concat([data[keep], pd.DataFrame(new_records)], ignore_index=True)
    timings = pd.concat([timings[~timings.method.isin(DISPATCH_ONLY)], pd.DataFrame(new_timings)], ignore_index=True)
    data.to_csv(RESULTS_FILE, index=False)
    timings.to_csv(TIMING_FILE, index=False)
    settings = pd.read_csv(SUMMARY_FILE)
    template = settings.drop_duplicates("simulation_run")
    templates = [template.assign(method=method) for method in DISPATCH_ONLY]
    pd.concat([settings[~settings.method.isin(DISPATCH_ONLY)]] + templates, ignore_index=True).to_csv(
        SUMMARY_FILE, index=False)
    print(f"Re-simulated {', '.join(DISPATCH_ONLY)} for runs {runs}.")
    resummarise()


def resummarise():
    """Rebuild the summaries from the saved results and timings (no re-solving)."""
    data = pd.read_csv(RESULTS_FILE)
    timings = pd.read_csv(TIMING_FILE)
    settings = pd.read_csv(SUMMARY_FILE)
    setting_columns = ["case", "load_noise", "pv_noise", "temp_noise_c", "load_bias", "bias_hours",
                       "correlation", "future_inputs", "cost_penalty", "share_penalty_gbp_per_kwh", "admm_start"]
    settings = settings.set_index(["simulation_run", "method"])[[c for c in setting_columns if c in settings]]
    rows = []
    for (run, method), records in data.groupby(["simulation_run", "method"]):
        steps = timings[(timings.simulation_run == run) & (timings.method == method)]
        row = summarise_run(records.to_dict("records"), steps.to_dict("records"), run, method)
        if (run, method) in settings.index:
            row.update(settings.loc[(run, method)].to_dict())
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(SUMMARY_FILE, index=False)
    _write_mean_summary(table)
    print(f"Rebuilt {SUMMARY_FILE.name} and {MEAN_SUMMARY_FILE.name} from saved results "
          f"({int(table['suspended_iterations_removed'].sum())} suspended ADMM iterations removed).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--load-noise", type=float, default=baseline.DEFAULT_LOAD_NOISE)
    parser.add_argument("--pv-noise", type=float, default=0.30)
    parser.add_argument("--temp-noise", type=float, default=1.0, help="Outdoor-temperature noise std (deg C).")
    parser.add_argument("--future-inputs", choices=("noisy", "forecast"), default="noisy",
                        help="Remaining-horizon inputs for re-optimisation.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cost-penalty", type=float, default=10.0)
    parser.add_argument("--share-penalty", type=float, default=0.1, help="GBP per kWh of planned P2P not delivered.")
    parser.add_argument("--admm-start", choices=("warm", "cold"), default="warm")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--refresh-scenarios", action="store_true")
    parser.add_argument("--refresh-dispatch-only", action="store_true",
                        help="Re-simulate open loop and fixed schedule on the saved scenarios (no re-solving).")
    parser.add_argument("--resummarise", action="store_true",
                        help="Rebuild 35/36 summaries from the saved results without re-solving.")
    parser.add_argument("--case", choices=tuple(CASES), default="base",
                        help="Sensitivity case; non-base cases write to cases/<case>/ (overrides the options below).")
    parser.add_argument("--load-bias", type=float, default=0.0, help="Systematic load error (fraction) in --bias-hours.")
    parser.add_argument("--bias-hours", default="16-22", help="Hours 'start-end' (end exclusive) for --load-bias.")
    parser.add_argument("--correlation", type=float, default=0.0,
                        help="Share of load/PV error common to all buildings (0-1).")
    args = parser.parse_args()
    _apply_case(args)
    if args.refresh_dispatch_only:
        _acquire_results_lock()
        refresh_dispatch_only(args)
        return
    if args.resummarise:
        _acquire_results_lock()
        resummarise()
        return
    if args.runs < 1 or min(args.load_noise, args.pv_noise, args.temp_noise, args.cost_penalty, args.share_penalty) < 0:
        parser.error("--runs must be positive and noise/penalty values non-negative.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Case '{args.case}': {CASES[args.case]['description']} -> {RESULTS_DIR}", flush=True)
    schedule = baseline.load_or_solve_schedule(reopt.SCHEDULE_CACHE, force_refresh=args.refresh_cache)
    loads, pvs, t_outs = _scenarios(args, args.runs, force_refresh=args.refresh_scenarios)
    _acquire_results_lock()
    for path in (RESULTS_FILE, TIMING_FILE, SUMMARY_FILE, MEAN_SUMMARY_FILE):
        path.write_text("", encoding="utf-8")

    summary = []
    for run in range(1, args.runs + 1):
        print(f"\n========== Paired scenario {run}/{args.runs} ==========", flush=True)
        actual_load, actual_pv, actual_t_out = loads[run - 1], pvs[run - 1], t_outs[run - 1]
        started = time.perf_counter()
        outcomes = {
            "open_loop": open_loop.simulate_open_loop(
                schedule, actual_load, actual_pv, actual_t_out, run=run,
                results_file=RESULTS_FILE, timing_file=TIMING_FILE,
            ),
            "fixed_schedule": simulate_fixed(schedule, actual_load, actual_pv, actual_t_out, run=run),
            "local_reoptimisation": reopt.simulate_local(
                schedule, actual_load, actual_pv, actual_t_out, run=run,
                cost_penalty=args.cost_penalty, share_penalty=args.share_penalty,
                future_inputs=args.future_inputs,
                results_file=RESULTS_FILE, timing_file=TIMING_FILE,
            ),
            "rolling_admm": reopt.simulate_rolling_admm(
                schedule, actual_load, actual_pv, actual_t_out, run=run, warm_start=args.admm_start == "warm",
                future_inputs=args.future_inputs,
                results_file=RESULTS_FILE, timing_file=TIMING_FILE,
            ),
        }
        run_rows = []
        for method in METHODS:
            row = summarise_run(*outcomes[method], run, method)
            row.update({
                "case": args.case, "load_bias": args.load_bias, "bias_hours": args.bias_hours,
                "correlation": args.correlation,
                "load_noise": args.load_noise, "pv_noise": args.pv_noise, "temp_noise_c": args.temp_noise,
                "future_inputs": args.future_inputs,
                "cost_penalty": args.cost_penalty, "share_penalty_gbp_per_kwh": args.share_penalty,
                "admm_start": args.admm_start,
            })
            run_rows.append(row)
        reopt.append_rows(SUMMARY_FILE, run_rows)
        summary.extend(run_rows)
        print(f"Scenario {run} finished in {time.perf_counter() - started:.0f}s")
        for row in run_rows:
            print(
                f"  {row['method']:<22} cost {row['final_total_cost_gbp']:8.2f} GBP "
                f"(plan {row['planned_total_cost_gbp']:.2f}), grid {row['final_grid_import_kwh']:7.2f} kWh, "
                f"P2P {row['final_p2p_kwh']:6.2f}/{row['planned_p2p_kwh']:.2f} kWh, "
                f"max parallel step {row['max_step_parallel_latency_seconds']:.2f}s",
                flush=True,
            )

    _write_mean_summary(pd.DataFrame(summary))
    print(f"\nResults: {RESULTS_FILE}\nSolve times: {TIMING_FILE}\nSummary: {SUMMARY_FILE}\nMean summary: {MEAN_SUMMARY_FILE}")


if __name__ == "__main__":
    main()
