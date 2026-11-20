"""Plot the community-size scaling run written by scale_realtime_methods.py.

PURPOSE
    Figures only: how solve time, ADMM iterations and per-building savings change with
    community size N.
    saving_b(method) = cost_b(open loop) − cost_b(method)

Figures (Results/06_realtime_scaling):
    60_scaling_solve_time.png         day-ahead and real-time solve time vs community size
    61_scaling_admm_iterations.png    ADMM iterations vs community size
    62_size_vs_building_savings.png   per-building saving of each re-optimiser vs open loop, vs size
    scaling_building_savings.csv      the per-building savings behind figure 62

Steps that did not finish are drawn as a cross at the time they were stopped and
labelled with the reason (not converged in budget / out of memory).

Usage:
    python plot_scaling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import project_paths  # noqa: E402  (shared modules on sys.path; results -> Results/<experiment>/)

project_paths.register(__file__)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_realtime_comparison import COLORS, GRID, INK, LABELS, PLAN_COLOR, _save, _style

SCALING_DIR = project_paths.experiment_dir("06_realtime_scaling")
STATUS_FILE = SCALING_DIR / "scaling_status.csv"
INTERVAL_SECONDS = 3600.0
MINUTE_SECONDS = 60.0

COLORS = {**COLORS, "rolling_admm_parallel": COLORS["rolling_admm"],
          "local_reoptimisation_parallel": COLORS["local_reoptimisation"]}
LABELS = {**LABELS, "rolling_admm": "Rolling ADMM (sequential, Pyomo)",
          "rolling_admm_parallel": "Rolling ADMM (parallel, 8 workers)",
          "local_reoptimisation": "Local re-optimisation (sequential, Pyomo)",
          "local_reoptimisation_parallel": "Local re-optimisation (parallel, 8 workers)"}
LINESTYLES = {"rolling_admm_parallel": "--", "local_reoptimisation_parallel": "--"}
MARKERS = {"local_reoptimisation": "s", "local_reoptimisation_parallel": "P", "rolling_admm": "o",
           "rolling_admm_parallel": "D", "rolling_central": "^"}
REALTIME_METHODS = ("local_reoptimisation", "local_reoptimisation_parallel", "rolling_admm", "rolling_admm_parallel",
                    "rolling_central")
SCHEDULE_COLORS = {"schedule": COLORS["rolling_admm"], "schedule_parallel": COLORS["rolling_admm"]}
SCHEDULE_LABELS = {"schedule": "Day-ahead ADMM (sequential, Pyomo)", "schedule_parallel": "Day-ahead ADMM (parallel, 8 workers)"}
FAILED = {"budget_exceeded": "not converged\nin budget", "killed_timeout": "not converged\nin budget",
          "killed_memory": "out of\nmemory", "stopped_not_converged": "not converged\n(stopped)"}

# The sequential day-ahead solves for N=5 and N=20 were done in earlier passes of the
# scaling run and later reloaded from cache (status file shows ~0 s). Their solve
# times and iteration counts are taken from those passes' run logs.
SEQUENTIAL_SCHEDULE_OVERRIDE = {5: 91.0, 20: 168.0}


def _status():
    status = pd.read_csv(STATUS_FILE)
    status["elapsed_seconds"] = pd.to_numeric(status["elapsed_seconds"], errors="coerce")
    return status


def _timings(n):
    path = SCALING_DIR / f"N{n:04d}" / "34_three_method_solve_times.csv"
    if not (path.exists() and path.stat().st_size):
        return pd.DataFrame()
    timings = pd.read_csv(path)
    return timings[timings["scope"] == "community"]


def _results(n):
    path = SCALING_DIR / f"N{n:04d}" / "33_three_method_results.csv"
    if not (path.exists() and path.stat().st_size):
        return pd.DataFrame()
    return pd.read_csv(path)


def _mark_failure(axis, n, seconds, status, color):
    axis.plot([n], [seconds], ls="none", marker="X", ms=10, mfc="white", mec=color, mew=2, zorder=4)
    axis.annotate(FAILED.get(status, status), (n, seconds), textcoords="offset points", xytext=(8, -4),
                  fontsize=7, color=INK, va="top")


def _reference_line(axis, sizes, seconds, label, below=False):
    """Red dashed time threshold, labelled at the left end (above the line, or below it)."""
    axis.axhline(seconds, color="#e34948", lw=1.2, ls="--", zorder=1)
    axis.annotate(label, (sizes[0], seconds), textcoords="offset points", xytext=(2, -4 if below else 4),
                  va="top" if below else "bottom", fontsize=8, color=INK)


def _log_axes(axis, sizes):
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(sizes)
    axis.set_xticklabels([str(n) for n in sizes])
    axis.minorticks_off()
    axis.set_xlabel("Community size N (buildings)")
    axis.grid(True, axis="x", color=GRID, lw=0.7)


def plot_solve_time(status):
    sizes = sorted(status["n_buildings"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharey=True)

    # (a) day-ahead plan
    axis = axes[0]
    for step in ("schedule", "schedule_parallel"):
        rows = status[status["step"] == step].set_index("n_buildings")
        done, failed = [], []
        for n in sizes:
            if n not in rows.index:
                continue
            row = rows.loc[n]
            seconds = row["elapsed_seconds"]
            if step == "schedule" and n in SEQUENTIAL_SCHEDULE_OVERRIDE:
                seconds = SEQUENTIAL_SCHEDULE_OVERRIDE[n]
            if step == "schedule_parallel" and pd.notna(row.get("solve_seconds")):
                seconds = row["solve_seconds"]
            (done if row["status"] == "completed" else failed).append((n, seconds, row["status"]))
        if done:
            axis.plot([d[0] for d in done], [d[1] for d in done], color=SCHEDULE_COLORS[step], lw=2,
                      ls="--" if step == "schedule_parallel" else "-",
                      marker="D" if step == "schedule_parallel" else "o", ms=6, label=SCHEDULE_LABELS[step])
        for n, seconds, state in failed:
            _mark_failure(axis, n, max(seconds, 1.0), state, SCHEDULE_COLORS[step])
    axis.set_title("Day-ahead plan (one ADMM solve, 24 h horizon)", loc="left", fontsize=10,
                   fontweight="bold", color=INK)
    axis.set_ylabel("Seconds (log scale)")

    # (b) and (c) real-time steps
    for axis, column, title in (
        (axes[1], "wall_seconds", "Real-time step, measured on this laptop\n(mean over the 23 re-solves after hour 0)"),
        (axes[2], "parallel_latency_seconds",
         "Real-time step, one controller per building (estimated)\n(slowest building per ADMM iteration, summed)"),
    ):
        for method in REALTIME_METHODS:
            xs, ys = [], []
            for n in sizes:
                steps = _timings(n)
                steps = steps[(steps["method"] == method) & (steps["interval"] > 0)] if len(steps) else steps
                row = status[(status["n_buildings"] == n) & (status["step"] == method)]
                state = row["status"].iloc[0] if len(row) else None
                if state == "completed" and len(steps):
                    xs.append(n)
                    ys.append(float(steps[column].mean()))
                elif state in FAILED and column == "wall_seconds":
                    # stopped before finishing: the time spent per interval completed (or in total if none)
                    completed = int(pd.to_numeric(row["intervals_completed"], errors="coerce").fillna(0).iloc[0])
                    spent = float(row["elapsed_seconds"].iloc[0])
                    _mark_failure(axis, n, spent / max(completed, 1), state, COLORS[method])
            if xs:
                axis.plot(xs, ys, color=COLORS[method], lw=2, ls=LINESTYLES.get(method, "-"),
                          marker=MARKERS[method], ms=6, label=LABELS[method])
        _reference_line(axis, sizes, INTERVAL_SECONDS, "Interval length (3600 s)")
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold", color=INK)

    # day-ahead plan: 1 hour reference; real-time steps: 1 minute (their interval line is drawn above)
    _reference_line(axes[0], sizes, INTERVAL_SECONDS, "1 hour (3600 s)")
    for axis in axes[1:]:
        _reference_line(axis, sizes, MINUTE_SECONDS, "1 minute (60 s)", below=True)
    for axis in axes:
        _style(axis)
        _log_axes(axis, sizes)
    handles = [Line2D([], [], color=SCHEDULE_COLORS["schedule"], lw=2, marker="o", label=SCHEDULE_LABELS["schedule"]),
               Line2D([], [], color=SCHEDULE_COLORS["schedule"], lw=2, ls="--", marker="D",
                      label=SCHEDULE_LABELS["schedule_parallel"])]
    handles += [Line2D([], [], color=COLORS[m], lw=2, ls=LINESTYLES.get(m, "-"), marker=MARKERS[m], label=LABELS[m])
                for m in REALTIME_METHODS]
    handles.append(Line2D([], [], ls="none", marker="X", ms=9, mfc="white", mec=PLAN_COLOR, mew=2,
                          label="Stopped, not converged: budget, memory or manual (time spent)"))
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0.04, 0.95))
    fig.suptitle("Computation time vs community size (Intel Core Ultra 5 135U, 16 GB; 1 h budget per method, "
                 "2 h for the day-ahead plan)", x=0.04, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output = SCALING_DIR / "60_scaling_solve_time.png"
    _save(fig, output)
    plt.close(fig)
    print(f"Saved {output}")


def plot_iterations(status):
    sizes = sorted(status["n_buildings"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axis = axes[0]
    rows = status[(status["step"] == "schedule_parallel") & (status["status"] == "completed")]
    rows = rows[pd.to_numeric(rows["admm_iterations"], errors="coerce") > 0].sort_values("n_buildings")
    axis.plot(rows["n_buildings"], rows["admm_iterations"].astype(float), color=COLORS["rolling_admm"], lw=2,
              marker="D", ms=6)
    for n, value in zip(rows["n_buildings"], rows["admm_iterations"].astype(float)):
        axis.annotate(f"{value:.0f}", (n, value), textcoords="offset points", xytext=(0, 6), ha="center",
                      fontsize=8, color=INK)
    axis.set_title("Day-ahead plan: iterations to converge", loc="left", fontsize=10, fontweight="bold", color=INK)
    axis.set_ylabel("ADMM iterations")

    axis = axes[1]
    for method in ("rolling_admm", "rolling_admm_parallel"):
        xs, first, later = [], [], []
        for n in sizes:
            steps = _timings(n)
            if not len(steps):
                continue
            steps = steps[steps["method"] == method]
            iterations = pd.to_numeric(steps["admm_iterations"], errors="coerce")
            if len(steps) != 24 or iterations.isna().all():
                continue
            xs.append(n)
            first.append(float(iterations[steps["interval"] == 0].iloc[0]))
            later.append(float(iterations[steps["interval"] > 0].mean()))
        if xs:
            axis.plot(xs, later, color=COLORS[method], lw=2, ls=LINESTYLES.get(method, "-"), marker=MARKERS[method],
                      ms=6, label=f"{LABELS[method]}: mean, hours 1-23")
            axis.plot(xs, first, color=COLORS[method], lw=1, ls=":", marker=MARKERS[method], ms=5, mfc="white",
                      label=f"{LABELS[method]}: hour 0")
    axis.set_title("Real-time steps: iterations per re-solve (warm-started)", loc="left", fontsize=10,
                   fontweight="bold", color=INK)
    axis.set_ylabel("ADMM iterations")
    axis.legend(frameon=False, fontsize=7)
    for axis in axes:
        _style(axis)
        axis.set_xscale("log")
        axis.set_xticks(sizes)
        axis.set_xticklabels([str(n) for n in sizes])
        axis.minorticks_off()
        axis.set_xlabel("Community size N (buildings)")
    fig.suptitle("ADMM iterations vs community size", x=0.04, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output = SCALING_DIR / "61_scaling_admm_iterations.png"
    _save(fig, output)
    plt.close(fig)
    print(f"Saved {output}")


def building_savings(status):
    """Per-building daily saving of each completed re-optimiser against the open loop (GBP/day)."""
    rows = []
    for n in sorted(status["n_buildings"].unique()):
        data = _results(n)
        if not len(data):
            continue
        done = set(status[(status["n_buildings"] == n) & (status["status"] == "completed")]["step"])
        if "open_loop" not in done:
            continue
        cost = data.groupby(["method", "building"])["final_total_cost"].sum()
        reference = cost.loc["open_loop"]
        for method in ("local_reoptimisation", "rolling_admm_parallel", "rolling_admm", "rolling_central"):
            if method not in done or method not in cost.index.get_level_values("method"):
                continue
            if data.loc[data["method"] == method, "interval"].nunique() != 24:
                continue
            saving = reference - cost.loc[method].reindex(reference.index)
            rows.extend({"n_buildings": n, "method": method, "building": b, "saving_gbp_per_day": float(s),
                         "open_loop_cost_gbp_per_day": float(reference[b])} for b, s in saving.items())
    table = pd.DataFrame(rows)
    table.to_csv(SCALING_DIR / "scaling_building_savings.csv", index=False)
    return table


def _spearman(x, y, permutations=20000, seed=0):
    """Spearman rho and a two-sided permutation p-value (numpy only)."""
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(permutations)])
    return rho, float((np.sum(np.abs(null) >= abs(rho)) + 1) / (permutations + 1))


def plot_size_vs_savings(status):
    table = building_savings(status)
    # the parallel ADMM is the same algorithm (results agree to 1e-5 GBP); it is the one that finished at every N
    panels = [m for m in ("local_reoptimisation", "rolling_admm_parallel", "rolling_central") if m in set(table["method"])]
    sizes = sorted(table["n_buildings"].unique())
    # own y-axis per panel: local re-optimisation spreads ~10x wider than the sharing methods
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 5.2))
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(1)
    for axis, method in zip(axes, panels):
        subset = table[table["method"] == method]
        color = COLORS[method]
        for n in sizes:
            values = subset.loc[subset["n_buildings"] == n, "saving_gbp_per_day"].to_numpy()
            if not len(values):
                continue
            jitter = np.exp(rng.uniform(-0.09, 0.09, len(values)))  # multiplicative: log x-axis
            axis.scatter(n * jitter, values, s=12, color=color, alpha=0.45, linewidths=0, zorder=2)
        means = subset.groupby("n_buildings")["saving_gbp_per_day"].mean()
        axis.plot(means.index, means.values, color=INK, lw=2, marker="o", ms=7, mfc=color, mec="white", mew=1.5,
                  zorder=3, label="Mean per building")
        for n, value in means.items():
            axis.annotate(f"{value:+.3f}", (n, value), textcoords="offset points", xytext=(0, 9), ha="center",
                          fontsize=8, color=INK)
        rho, p_value = _spearman(np.log(subset["n_buildings"]), subset["saving_gbp_per_day"])
        log_n = np.log10(means.index.to_numpy(float))
        slope, intercept = np.polyfit(log_n, means.to_numpy(), 1)
        fit_x = np.logspace(log_n.min(), log_n.max(), 50)
        axis.plot(fit_x, intercept + slope * np.log10(fit_x), color=PLAN_COLOR, lw=1.2, ls="--", zorder=1,
                  label=f"Fit to means: {slope:+.3f} GBP/day per 10x N")
        axis.axhline(0, color=PLAN_COLOR, lw=1)
        axis.set_title(f"{LABELS[method]}\nSpearman rho = {rho:+.2f} (p = {p_value:.3f}), "
                       f"{len(subset)} building-days", loc="left", fontsize=9, fontweight="bold", color=INK)
        axis.set_xscale("log")
        axis.set_xticks(sizes)
        axis.set_xticklabels([str(n) for n in sizes])
        axis.minorticks_off()
        axis.set_xlabel("Community size N (buildings)")
        axis.legend(frameon=False, fontsize=7, loc="lower left")
        _style(axis)
    for axis in axes:
        axis.set_ylabel("Saving vs open loop per building (GBP/day)")
    fig.suptitle("Does a bigger community save more per building? Real-time re-optimisation vs open loop "
                 "(dots = buildings, one noise scenario per size)", x=0.04, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output = SCALING_DIR / "62_size_vs_building_savings.png"
    _save(fig, output)
    plt.close(fig)
    print(f"Saved {output}")
    return table


def main():
    status = _status()
    plot_solve_time(status)
    plot_iterations(status)
    table = plot_size_vs_savings(status)
    summary = table.groupby(["method", "n_buildings"])["saving_gbp_per_day"].agg(["mean", "median", "std", "count"])
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
