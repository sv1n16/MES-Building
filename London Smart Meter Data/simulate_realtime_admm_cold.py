"""Rolling ADMM restarted from scratch at every interval, compared with warm start.

At every interval bilateral ADMM is re-solved over the remaining horizon starting
from zero duals and zero agreed trades (cold start), instead of the previous
interval's duals and trades (warm start, as in compare_realtime_methods.py).
It runs on the same shared noisy scenarios and with the same future inputs, so
the results pair one-to-one with the fixed schedule, the local non-sharing
re-optimisation and the warm-started ADMM in 33-35_three_method_*.csv.

Scenarios run as parallel processes. Each Gurobi solve is limited to one thread
(one core per building controller) so that processes do not compete for cores;
the summary reports per-subproblem solve times for both ADMM variants so the
timing comparison can be checked.

Outputs in Results/ADMM Optimisation Real-time:
    41_admm_cold_start_results_run<n>.csv      one row per interval/building
    42_admm_cold_start_solve_times_run<n>.csv  per-iteration and per-interval timings
    43_admm_warm_vs_cold_summary.csv           per run, all four methods
    44_admm_warm_vs_cold_summary_mean.csv      mean and std over runs
    45_admm_warm_vs_cold.png                   outcomes, iterations and solve times

Examples:
    python simulate_realtime_admm_cold.py --runs 5 --jobs 5
    python simulate_realtime_admm_cold.py --summary-only
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import compare_realtime_methods as compare
import plot_realtime_comparison as plots
import simulate_noisy_realtime as baseline
import simulate_realtime_reoptimise as reopt

RESULTS_DIR = reopt.RESULTS_DIR
COLD = "rolling_admm_cold"
WARM = "rolling_admm"
METHODS = ("fixed_schedule", "local_reoptimisation", WARM, COLD)
LABELS = {
    "fixed_schedule": "Fixed schedule",
    "local_reoptimisation": "Local re-optimisation (no sharing)",
    WARM: "Rolling ADMM, warm start",
    COLD: "Rolling ADMM, cold start",
}
COLORS = {"fixed_schedule": "#2a78d6", "local_reoptimisation": "#eb6834", WARM: "#1baf7a", COLD: "#6250d6"}
SUMMARY_FILE = RESULTS_DIR / "43_admm_warm_vs_cold_summary.csv"
MEAN_SUMMARY_FILE = RESULTS_DIR / "44_admm_warm_vs_cold_summary_mean.csv"
FIGURE_FILE = RESULTS_DIR / "45_admm_warm_vs_cold.png"


def cold_results_file(run):
    return RESULTS_DIR / f"41_admm_cold_start_results_run{run}.csv"


def cold_timing_file(run):
    return RESULTS_DIR / f"42_admm_cold_start_solve_times_run{run}.csv"


def _single_thread_solvers(threads):
    """Limit every Gurobi solve in this process to `threads` threads."""
    pyo = reopt.pyo
    factory = pyo.SolverFactory

    def limited(*args, **kwargs):
        solver = factory(*args, **kwargs)
        solver.options["Threads"] = threads
        return solver

    pyo.SolverFactory = limited


def run_cold_scenario(run, runs, load_noise, pv_noise, temp_noise, future_inputs, threads):
    """Worker: one scenario of cold-start rolling ADMM (runs in its own process)."""
    _single_thread_solvers(threads)
    schedule = baseline.load_or_solve_schedule(reopt.SCHEDULE_CACHE)
    loads, pvs, t_outs = baseline.load_or_create_scenarios(
        reopt.SCENARIO_FILE, runs=runs, load_noise=load_noise, pv_noise=pv_noise, temp_noise=temp_noise,
    )
    results_file, timing_file = cold_results_file(run), cold_timing_file(run)
    for path in (results_file, timing_file):
        path.write_text("", encoding="utf-8")
    started = time.perf_counter()
    reopt.simulate_rolling_admm(
        schedule, loads[run - 1], pvs[run - 1], t_outs[run - 1], run=run, warm_start=False,
        future_inputs=future_inputs, results_file=results_file, timing_file=timing_file, method=COLD,
    )
    return run, time.perf_counter() - started


def _load_all(runs):
    data = pd.read_csv(compare.RESULTS_FILE)
    timings = pd.read_csv(compare.TIMING_FILE)
    cold_data, cold_timings = [], []
    for run in runs:
        if cold_results_file(run).exists() and cold_results_file(run).stat().st_size:
            cold_data.append(pd.read_csv(cold_results_file(run)))
            cold_timings.append(pd.read_csv(cold_timing_file(run)))
    if not cold_data:
        raise FileNotFoundError("No cold-start results found; run without --summary-only first.")
    data = pd.concat([data] + cold_data, ignore_index=True)
    timings = pd.concat([timings] + cold_timings, ignore_index=True)
    complete = sorted(set(pd.concat(cold_data)["simulation_run"]) & set(data.loc[data.method == WARM, "simulation_run"]))
    return data[data.simulation_run.isin(complete)], timings[timings.simulation_run.isin(complete)], complete


def summarise(runs):
    data, timings, complete = _load_all(runs)
    rows = []
    for run in complete:
        for method in METHODS:
            records = data[(data.method == method) & (data.simulation_run == run)].to_dict("records")
            steps = timings[(timings.method == method) & (timings.simulation_run == run)]
            row = compare.summarise_run(records, steps.to_dict("records"), run, method)
            calls = steps[steps.scope == "community"]
            row["total_solver_calls"] = float(pd.to_numeric(calls["solver_calls"]).sum())
            iters = steps[steps.scope == "admm_iteration"]
            row["mean_subproblem_seconds"] = (
                float((iters["wall_seconds"] / iters["solver_calls"]).mean()) if len(iters) else np.nan
            )
            rows.append(row)
    table = pd.DataFrame(rows)

    # Same problem, same tolerance: both ADMM variants should reach the same dispatch.
    warm = data[(data.method == WARM)].set_index(["simulation_run", "interval", "building"])
    cold = data[(data.method == COLD)].set_index(["simulation_run", "interval", "building"])
    gap = (warm["final_p_el"] - cold["final_p_el"].reindex(warm.index)).abs()
    table["max_grid_import_gap_warm_vs_cold_kw"] = float(gap.max())

    table.to_csv(SUMMARY_FILE, index=False)
    numeric = table.drop(columns=["simulation_run"]).select_dtypes("number").columns
    mean = table.groupby("method")[list(numeric)].agg(["mean", "std"])
    mean.columns = [f"{name}_{stat}" for name, stat in mean.columns]
    mean = mean.reindex(METHODS)
    mean.to_csv(MEAN_SUMMARY_FILE)
    return table, mean, timings, complete


def plot(table, timings, complete):
    steps = timings[timings.scope == "community"].copy()
    steps["admm_iterations"] = pd.to_numeric(steps["admm_iterations"], errors="coerce")
    fig = plt.figure(figsize=(17, 9.5))
    grid = fig.add_gridspec(2, 4)
    bars = (
        ("cost_change_vs_plan_gbp", "Cost vs day-ahead plan (GBP)"),
        ("final_p2p_kwh", "P2P energy delivered (kWh)"),
        ("final_peak_grid_kw", "Peak grid import (kW)"),
        ("total_wall_seconds", "Total solve time per day (s, one machine)"),
    )
    x = np.arange(len(METHODS))
    for col, (column, title) in enumerate(bars):
        axis = fig.add_subplot(grid[0, col])
        for i, method in enumerate(METHODS):
            values = table.loc[table.method == method, column].to_numpy(float)
            axis.bar(i, values.mean(), width=0.62, color=COLORS[method])
            jitter = np.linspace(-0.13, 0.13, len(values)) if len(values) > 1 else np.zeros(1)
            axis.scatter(i + jitter, values, s=12, color=plots.INK, zorder=3, linewidths=0)
            axis.annotate(f"{values.mean():.1f}", (i, max(values.mean(), 0)), textcoords="offset points",
                          xytext=(0, 9), ha="center", fontsize=8, color=plots.INK)
        if column == "total_wall_seconds":
            axis.set_yscale("log")
        if column == "final_peak_grid_kw":
            axis.axhline(table["planned_peak_grid_kw"].mean(), color=plots.PLAN_COLOR, ls=":", lw=1.5)
        if column == "final_p2p_kwh":
            axis.axhline(table["planned_p2p_kwh"].mean(), color=plots.PLAN_COLOR, ls=":", lw=1.5)
        if column == "cost_change_vs_plan_gbp":
            axis.axhline(0, color=plots.PLAN_COLOR, lw=1)
        axis.set_xticks(x)
        axis.set_xticklabels(["Fixed", "Local", "ADMM\nwarm", "ADMM\ncold"])
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold", color=plots.INK)
        plots._style(axis)

    axis = fig.add_subplot(grid[1, :2])
    for method in (WARM, COLD):
        series = steps[steps.method == method].groupby("interval")["admm_iterations"]
        axis.plot(series.mean().index, series.mean().values, color=COLORS[method], lw=2, marker="o", ms=4,
                  label=LABELS[method])
        axis.fill_between(series.mean().index, series.min().values, series.max().values,
                          color=COLORS[method], alpha=0.15, lw=0)
    axis.set_yscale("log")
    axis.set_xlabel("Interval (h)")
    axis.set_ylabel("ADMM iterations (log scale)")
    axis.set_title("Iterations to converge per interval", loc="left", fontsize=10, fontweight="bold", color=plots.INK)
    axis.legend(frameon=False, fontsize=8)
    plots._style(axis)

    axis = fig.add_subplot(grid[1, 2:])
    interval_seconds = float(steps["interval_seconds"].iloc[0])
    for method in ("local_reoptimisation", WARM, COLD):
        series = steps[steps.method == method].groupby("interval")["parallel_latency_seconds"]
        axis.plot(series.mean().index, series.mean().values, color=COLORS[method], lw=2, marker="o", ms=4,
                  label=LABELS[method])
        axis.fill_between(series.mean().index, series.min().values, series.max().values,
                          color=COLORS[method], alpha=0.15, lw=0)
    axis.axhline(interval_seconds, color="#e34948", lw=1.2, ls="--")
    axis.annotate(f"Interval length ({interval_seconds:.0f} s)", (0, interval_seconds), textcoords="offset points",
                  xytext=(2, 4), fontsize=8, color=plots.INK)
    axis.set_yscale("log")
    axis.set_xlabel("Interval (h)")
    axis.set_ylabel("Seconds (log scale)")
    axis.set_title("Latency per step with parallel building controllers", loc="left", fontsize=10,
                   fontweight="bold", color=plots.INK)
    axis.legend(frameon=False, fontsize=8, loc="lower left")
    plots._style(axis)

    fig.suptitle(f"Warm- vs cold-started rolling ADMM, with fixed and local baselines "
                 f"(bars = mean, dots = {len(complete)} paired runs, bands = min–max)",
                 x=0.03, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURE_FILE, dpi=160)
    plt.close(fig)
    print(f"Saved {FIGURE_FILE}")


PARTIAL_FILE = RESULTS_DIR / "43_admm_warm_vs_cold_partial.csv"
SUSPEND_FACTOR = 20.0  # an iteration this many times slower than the median was suspended (sleep/lock)


def _clean_iterations(iterations):
    """Replace iterations that include a machine sleep/suspend with the run's median."""
    iterations = iterations.copy()
    median = iterations.groupby(["simulation_run", "interval"])["wall_seconds"].transform("median")
    suspended = iterations["wall_seconds"] > SUSPEND_FACTOR * median
    for column in ("wall_seconds", "parallel_latency_seconds"):
        typical = iterations.groupby(["simulation_run", "interval"])[column].transform("median")
        iterations.loc[suspended, column] = typical[suspended]
    return iterations, int(suspended.sum())


def partial_comparison(runs):
    """Compare warm and cold start on the intervals the cold-start runs completed.

    Iteration counts are compared directly. Times are reported two ways: as
    measured (sleep-corrected) and projected onto the warm run's hardware basis,
    i.e. cold iterations x the warm run's median time per iteration at the same
    interval (same remaining horizon), because the cold runs shared the machine
    five ways on one thread each.
    """
    timings = pd.read_csv(compare.TIMING_FILE)
    warm_iter = timings[(timings.method == WARM) & (timings.scope == "admm_iteration")]
    warm_steps = timings[(timings.method == WARM) & (timings.scope == "community")].copy()
    local_steps = timings[(timings.method == "local_reoptimisation") & (timings.scope == "community")]
    cold_iter = pd.concat(
        [pd.read_csv(cold_timing_file(r)) for r in runs if cold_timing_file(r).stat().st_size], ignore_index=True
    )
    cold_iter = cold_iter[cold_iter.scope == "admm_iteration"]
    cold_iter, n_suspended = _clean_iterations(cold_iter)
    # an interval counts as complete once its community row was written
    cold_steps = pd.concat(
        [pd.read_csv(cold_timing_file(r)) for r in runs if cold_timing_file(r).stat().st_size], ignore_index=True
    )
    cold_steps = cold_steps[cold_steps.scope == "community"]
    complete = cold_steps[["simulation_run", "interval"]].drop_duplicates()
    cold_iter = cold_iter.merge(complete, on=["simulation_run", "interval"])

    warm_per_iteration = warm_iter.groupby("interval").agg(
        warm_seconds_per_iteration=("wall_seconds", "median"),
        warm_latency_per_iteration=("parallel_latency_seconds", "median"),
    )
    rows = []
    for (run, interval), group in cold_iter.groupby(["simulation_run", "interval"]):
        warm = warm_steps[(warm_steps.simulation_run == run) & (warm_steps.interval == interval)].iloc[0]
        local = local_steps[(local_steps.simulation_run == run) & (local_steps.interval == interval)].iloc[0]
        basis = warm_per_iteration.loc[interval]
        rows.append({
            "simulation_run": run,
            "interval": interval,
            "horizon_length": int(group["horizon_length"].iloc[0]),
            "warm_iterations": int(float(warm["admm_iterations"])),
            "cold_iterations": int(len(group)),
            "cold_hit_max_iterations": bool(len(group) >= reopt.OPT.MAX_ITERS),
            "local_parallel_latency_seconds": float(local["parallel_latency_seconds"]),
            "warm_wall_seconds": float(warm["wall_seconds"]),
            "warm_parallel_latency_seconds": float(warm["parallel_latency_seconds"]),
            "cold_wall_seconds_measured": float(group["wall_seconds"].sum()),
            "cold_parallel_latency_seconds_measured": float(group["parallel_latency_seconds"].sum()),
            "cold_wall_seconds_projected": float(len(group) * basis["warm_seconds_per_iteration"]),
            "cold_parallel_latency_seconds_projected": float(len(group) * basis["warm_latency_per_iteration"]),
        })
    table = pd.DataFrame(rows)
    table.to_csv(PARTIAL_FILE, index=False)

    # Full-day projection: cold start needs ~the same iterations every interval
    # (it never reuses information), so scale by the warm per-iteration times.
    cold_mean_iterations = table["cold_iterations"].mean()
    projected_day = float((cold_mean_iterations * warm_per_iteration["warm_seconds_per_iteration"]).sum())
    projected_day_latency = float((cold_mean_iterations * warm_per_iteration["warm_latency_per_iteration"]).sum())
    warm_day = warm_steps.groupby("simulation_run")["wall_seconds"].sum().mean()
    warm_day_latency = warm_steps.groupby("simulation_run")["parallel_latency_seconds"].sum().mean()
    local_day_latency = local_steps.groupby("simulation_run")["parallel_latency_seconds"].sum().mean()
    projection = {
        "cold_mean_iterations_per_interval": cold_mean_iterations,
        "warm_mean_iterations_per_interval": float(pd.to_numeric(warm_steps["admm_iterations"]).mean()),
        "warm_day_wall_seconds": warm_day,
        "cold_day_wall_seconds_projected": projected_day,
        "warm_day_parallel_latency_seconds": warm_day_latency,
        "cold_day_parallel_latency_seconds_projected": projected_day_latency,
        "local_day_parallel_latency_seconds": local_day_latency,
        "suspended_iterations_removed": n_suspended,
    }
    _plot_partial(table, warm_steps, local_steps, warm_per_iteration, cold_mean_iterations)
    return table, projection


def _plot_partial(table, warm_steps, local_steps, warm_per_iteration, cold_mean_iterations):
    warm_steps = warm_steps.copy()
    warm_steps["admm_iterations"] = pd.to_numeric(warm_steps["admm_iterations"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))

    axis = axes[0]
    warm = warm_steps.groupby("interval")["admm_iterations"]
    axis.plot(warm.mean().index, warm.mean().values, color=COLORS[WARM], lw=2, marker="o", ms=4, label=LABELS[WARM])
    axis.fill_between(warm.mean().index, warm.min().values, warm.max().values, color=COLORS[WARM], alpha=0.15, lw=0)
    cold = table.groupby("interval")["cold_iterations"]
    axis.plot(cold.mean().index, cold.mean().values, color=COLORS[COLD], lw=2, marker="o", ms=5,
              label=f"{LABELS[COLD]} (measured, intervals {table.interval.min()}–{table.interval.max()})")
    axis.scatter(table["interval"], table["cold_iterations"], color=COLORS[COLD], s=12, zorder=3)
    later = warm_per_iteration.index[warm_per_iteration.index > table.interval.max()]
    axis.plot(later, np.full(len(later), cold_mean_iterations), color=COLORS[COLD], lw=1.5, ls="--",
              label="Cold start, projected (no reuse between intervals)")
    axis.axhline(reopt.OPT.MAX_ITERS, color=plots.PLAN_COLOR, lw=1, ls=":")
    axis.annotate("Iteration cap", (23, reopt.OPT.MAX_ITERS), textcoords="offset points", xytext=(-4, 4),
                  ha="right", fontsize=8, color=plots.INK)
    axis.set_yscale("log")
    axis.set_xlabel("Interval (h)")
    axis.set_ylabel("ADMM iterations (log scale)")
    axis.set_title("Iterations per interval", loc="left", fontsize=10, fontweight="bold", color=plots.INK)
    axis.legend(frameon=False, fontsize=8, loc="center right")
    plots._style(axis)

    axis = axes[1]
    interval_seconds = reopt.INTERVAL_SECONDS
    for method, steps in (("local_reoptimisation", local_steps), (WARM, warm_steps)):
        series = steps.groupby("interval")["parallel_latency_seconds"]
        axis.plot(series.mean().index, series.mean().values, color=COLORS[method], lw=2, marker="o", ms=4,
                  label=LABELS[method])
    projected = cold_mean_iterations * warm_per_iteration["warm_latency_per_iteration"]
    axis.plot(projected.index, projected.values, color=COLORS[COLD], lw=1.5, ls="--",
              label="Cold start, projected")
    measured = table.groupby("interval")["cold_parallel_latency_seconds_projected"].mean()
    axis.plot(measured.index, measured.values, color=COLORS[COLD], lw=2, marker="o", ms=5,
              label="Cold start, measured iterations")
    axis.axhline(interval_seconds, color="#e34948", lw=1.2, ls="--")
    axis.annotate(f"Interval length ({interval_seconds:.0f} s)", (0, interval_seconds), textcoords="offset points",
                  xytext=(2, 4), fontsize=8, color=plots.INK)
    axis.set_yscale("log")
    axis.set_xlabel("Interval (h)")
    axis.set_ylabel("Seconds (log scale)")
    axis.set_title("Latency per step with parallel building controllers", loc="left", fontsize=10,
                   fontweight="bold", color=plots.INK)
    axis.legend(frameon=False, fontsize=8, loc="lower left")
    plots._style(axis)

    fig.suptitle("Rolling ADMM: warm start vs restart from scratch "
                 f"({table.simulation_run.nunique()} paired scenarios; cold-start run stopped after "
                 f"interval {table.interval.max()})", x=0.03, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_FILE, dpi=160)
    plt.close(fig)
    print(f"Saved {FIGURE_FILE}")


def _cold_runs_complete(runs):
    for run in runs:
        path = cold_timing_file(run)
        if not path.exists() or not path.stat().st_size:
            return False
        steps = pd.read_csv(path)
        if steps.loc[steps.scope == "community", "interval"].nunique() < reopt.FULL_T:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=5, help="Scenarios 1..runs (must already exist in the comparison).")
    parser.add_argument("--jobs", type=int, default=5, help="Parallel worker processes.")
    parser.add_argument("--threads", type=int, default=1, help="Gurobi threads per solve.")
    parser.add_argument("--load-noise", type=float, default=baseline.DEFAULT_LOAD_NOISE)
    parser.add_argument("--pv-noise", type=float, default=0.30)
    parser.add_argument("--temp-noise", type=float, default=1.0)
    parser.add_argument("--future-inputs", choices=("noisy", "forecast"), default="noisy")
    parser.add_argument("--summary-only", action="store_true", help="Skip solving; rebuild summary and figure.")
    args = parser.parse_args()
    runs = list(range(1, args.runs + 1))

    if not args.summary_only:
        print(f"Cold-start rolling ADMM: scenarios {runs}, {args.jobs} processes, {args.threads} thread(s) per solve",
              flush=True)
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [
                pool.submit(run_cold_scenario, run, args.runs, args.load_noise, args.pv_noise,
                            args.temp_noise, args.future_inputs, args.threads)
                for run in runs
            ]
            for future in as_completed(futures):
                run, seconds = future.result()
                print(f"Scenario {run} cold-start ADMM finished in {seconds / 60:.1f} min", flush=True)

    if not _cold_runs_complete(runs):
        table, projection = partial_comparison(runs)
        pd.set_option("display.width", 200)
        print(table.round(2).to_string(index=False))
        for name, value in projection.items():
            print(f"  {name}: {value:,.1f}")
        print(f"Partial comparison: {PARTIAL_FILE}")
        return

    table, mean, timings, complete = summarise(runs)
    columns = [
        "cost_change_vs_plan_gbp", "final_p2p_kwh", "final_peak_grid_kw", "grid_import_abs_deviation_kwh",
        "total_wall_seconds", "max_step_wall_seconds", "max_step_parallel_latency_seconds",
        "mean_admm_iterations", "max_admm_iterations", "mean_subproblem_seconds",
    ]
    print(mean[[f"{c}_mean" for c in columns]].round(3).T.to_string())
    print(f"Max |grid import| gap, warm vs cold ADMM: {table['max_grid_import_gap_warm_vs_cold_kw'].iloc[0]:.4f} kW")
    plot(table, timings, complete)
    print(f"Summary: {SUMMARY_FILE}\nMean summary: {MEAN_SUMMARY_FILE}")


if __name__ == "__main__":
    main()
