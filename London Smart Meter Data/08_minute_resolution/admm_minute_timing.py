"""How long does bilateral ADMM take to optimise one hour of minute-level data? (N = 10)

PURPOSE
    Is bilateral ADMM fast enough for 1-minute control? Measures the parallel solve time
    for one hour (60 one-minute steps) and the cost gap to the central solution.

PROBLEM   (one hour h, Δt = 1/60 h, minutes t = 0..59; same for ADMM and central)

    min  Σ_b Σ_t [ π_t g Δt + c_gas G Δt + α Δt (T − T^set)² + w_t Δt g² ]
    s.t. building model B_b, trades agree (ADMM: in the limit; central: exactly),
         initial SOC / T_in = the hourly day-ahead plan's state at the start of hour h

    ADMM time (parallel estimate) = Σ_iterations max_b (subproblem solve time) + build / N

The hourly showcase inputs are turned into 1-minute data, and the community problem is
solved over one hour (60 one-minute steps) with bilateral ADMM, for several start hours.

Minute data (--interpolation)
    linear  load, PV and outdoor temperature are interpolated linearly between the
            hourly mid-points (PV clipped at zero); tariff and heating setpoint are held
            constant within each hour (they are step schedules). Default.
    step    every input held constant within its hour (keeps the hourly averages exactly).

Model at minute resolution
    The step length is 1/60 h. Energy costs, battery and thermal dynamics already scale
    with it. The two terms charged per step, the comfort penalty alpha (T_in - setpoint)^2
    and the congestion term, are multiplied by the step length so that one hour of minutes
    weighs the same as one hourly step. The initial battery charge and indoor temperature
    of each hour are those the day-ahead hourly plan predicts (cached in
    Results/05_realtime_methods). The first minute may use the battery (rolling mode).

Timing (parallel estimate only)
    Every subproblem solve is timed. ADMM time = sum over iterations of the slowest
    building's solve (one solver per building, run in parallel), plus the model build
    divided by N. ADMM is run from a cold start (zero prices and trades) and from a warm
    start (the hourly plan's trades for that hour). The same hour is also solved centrally
    as a cost and time reference.

Outputs (Results/08_minute_resolution/):
    80_minute_inputs.png          hourly against minute inputs
    81_admm_minute_timing.csv     per start hour and start mode: iterations, ADMM parallel time,
                                  central time, cost gap ADMM vs central
    82_admm_minute_timing.png     ADMM parallel time and iterations per start hour, against
                                  central and the 1-minute / 1-hour limits

Run it when the machine is otherwise idle: other jobs inflate the timings.

Examples:
    python admm_minute_timing.py
    python admm_minute_timing.py --hours all
    python admm_minute_timing.py --hours 17 --starts cold
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import project_paths  # noqa: E402  (shared modules on sys.path; results -> Results/<experiment>/)

project_paths.register(__file__)

import argparse  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyomo.environ as pyo  # noqa: E402

import admm_bilateral_p2p_valley as OPT  # noqa: E402
import simulate_noisy_realtime as baseline  # noqa: E402
import simulate_realtime_central as central  # noqa: E402
from plot_realtime_comparison import COLORS, INK, PLAN_COLOR, _save, _style  # noqa: E402

C = OPT.C
N, HOURS = C.n_buildings, C.time_horizon
RESULTS_DIR = C.PLOTS_DIR  # Results/08_minute_resolution/
PLAN_FILE = project_paths.RESULTS_DIR / "05_realtime_methods" / "bilateral_admm_schedule.npz"
INPUTS_FIGURE = RESULTS_DIR / "80_minute_inputs.png"
TIMING_FILE = RESULTS_DIR / "81_admm_minute_timing.csv"
TIMING_FIGURE = RESULTS_DIR / "82_admm_minute_timing.png"
CONVERGENCE_FILE = RESULTS_DIR / "83_admm_minute_convergence.csv"
CONVERGENCE_FIGURE = RESULTS_DIR / "83_admm_minute_convergence.png"
DISPATCH_FILE = RESULTS_DIR / "84_minute_dispatch.csv"
DISPATCH_FIGURE = RESULTS_DIR / "84_minute_dispatch.png"
DEFAULT_HOURS = (3, 8, 13, 17, 20)  # night, morning setpoint rise, midday PV, evening peak start, evening


# ============================================================================
# MINUTE DATA
# ============================================================================
def to_minutes(hourly, steps_per_hour, how):
    """(..., H) hourly averages -> (..., H * steps_per_hour) minute values."""
    hourly = np.asarray(hourly, dtype=float)
    if how == "step":
        return np.repeat(hourly, steps_per_hour, axis=-1)
    hours = hourly.shape[-1]
    centres = np.arange(hours) + 0.5
    minutes = (np.arange(hours * steps_per_hour) + 0.5) / steps_per_hour
    flat = hourly.reshape(-1, hours)
    out = np.array([np.interp(minutes, centres, row) for row in flat])
    return out.reshape(hourly.shape[:-1] + (hours * steps_per_hour,))


def minute_inputs(how, steps_per_hour=60):
    """Whole-day minute arrays for the model inputs."""
    return {
        "LOAD": np.maximum(to_minutes(C.LOAD, steps_per_hour, how), 0.0),
        "PV_B": np.maximum(to_minutes(C.PV_B, steps_per_hour, how), 0.0),
        "T_OUT": to_minutes(C.T_OUT, steps_per_hour, how),
        "PRICE": np.repeat(C.PRICE, steps_per_hour),  # tariff: constant within each hour
        "T_SET": np.repeat(C.T_SET, steps_per_hour),  # setpoint schedule: constant within each hour
        "congestion": np.repeat(np.asarray(OPT.congestion_weight, dtype=float), steps_per_hour),
    }


def minute_window(data, hour, minutes, soc, indoor):
    """Point the optimiser at `minutes` one-minute steps starting at `hour`."""
    return minute_window_at(data, hour * 60, minutes, soc, indoor)


@contextmanager
def minute_window_at(data, start, minutes, soc, indoor):
    """Point the optimiser at `minutes` one-minute steps starting at minute `start` of the day."""
    dt = 1.0 / 60.0
    stop = start + minutes
    central_names = ("LOAD", "PV_B", "PRICE", "T_OUT", "T_SET", "time_horizon", "dt", "alpha",
                     "INIT_SOC", "INIT_T_IN")
    optimiser_names = ("T", "DT", "COP", "PV_B", "PRICE", "congestion_weight", "ROLLING_FIRST_INTERVAL")
    old_central = {name: getattr(C, name) for name in central_names}
    old_optimiser = {name: getattr(OPT, name) for name in optimiser_names}
    old_central_model_dt = central.DT  # simulate_realtime_central keeps its own copy of the step
    central.DT = dt
    C.LOAD = data["LOAD"][:, start:stop].copy()
    C.PV_B = data["PV_B"][:, start:stop].copy()
    C.PRICE = data["PRICE"][start:stop].copy()
    C.T_OUT = data["T_OUT"][start:stop].copy()
    C.T_SET = data["T_SET"][start:stop].copy()
    C.time_horizon, C.dt = minutes, dt
    C.alpha = old_central["alpha"] * dt  # comfort penalty is charged per step
    C.INIT_SOC = np.asarray(soc, dtype=float).copy()
    C.INIT_T_IN = np.asarray(indoor, dtype=float).copy()
    OPT.T, OPT.DT = minutes, dt
    OPT.COP = C.cop_base + 0.01 * (C.T_OUT - C.T_ref)
    OPT.PV_B, OPT.PRICE = C.PV_B, C.PRICE
    OPT.congestion_weight = data["congestion"][start:stop] * dt  # also charged per step
    OPT.ROLLING_FIRST_INTERVAL = True  # the battery may act from the first minute
    try:
        yield
    finally:
        central.DT = old_central_model_dt
        for name, value in old_central.items():
            setattr(C, name, value)
        for name, value in old_optimiser.items():
            setattr(OPT, name, value)


def plan_state(schedule, hour):
    """Battery charge and indoor temperature at the start of `hour`, from the hourly plan."""
    if hour == 0:
        return np.asarray(C.INIT_SOC, dtype=float), np.asarray(C.INIT_T_IN, dtype=float)
    planned = schedule["variables"]
    return planned["soc"][:, hour - 1].copy(), planned["T_in"][:, hour - 1].copy()


# ============================================================================
# SOLVES
# ============================================================================
def _community_cost(subs):
    """Operating cost of the hour (energy, gas, comfort, congestion), as both methods minimise it."""
    total = 0.0
    for m in subs:
        for t in m.t:
            total += (OPT.PRICE[t] * pyo.value(m.p_el[t]) * OPT.DT
                      + C.gas_price / 100.0 * pyo.value(m.gas[t]) * OPT.DT
                      + C.alpha * (pyo.value(m.T_in[t]) - C.T_SET[t]) ** 2
                      + OPT.congestion_weight[t] * pyo.value(m.p_el[t]) ** 2)
    return total


def _dispatch(subs, hour, method):
    """Community values per minute of the window (inside minute_window)."""
    rows = []
    for t in range(OPT.T):
        rows.append({
            "start_hour": hour, "method": method, "minute": t,
            "grid_import_kw": sum(pyo.value(m.p_el[t]) for m in subs),
            "battery_discharge_minus_charge_kw": sum(pyo.value(m.discharge[t]) - pyo.value(m.charge[t]) for m in subs),
            "p2p_traded_kw": sum(pyo.value(m.qpos[k, t]) for m in subs for k in m.k),
            "mean_indoor_temperature_c": float(np.mean([pyo.value(m.T_in[t]) for m in subs])),
            "setpoint_c": float(C.T_SET[t]),
            "load_kw": float(C.LOAD[:, t].sum()), "pv_kw": float(C.PV_B[:, t].sum()),
            "heat_pump_kw": sum(pyo.value(m.p_hp[t]) for m in subs),
        })
    return rows


def solve_admm(schedule, hour, start):
    """ADMM over the current window; returns timing and cost, the convergence history and the dispatch."""
    init_Z = None
    if start == "warm":
        trades = schedule["trades"][:, :, hour]
        init_Z = {(lo, hi): np.full(OPT.T, float(trades[lo, hi])) for lo, hi in OPT.PAIRS}
    log = []
    started = time.perf_counter()
    subs, _q, _lam, _Z, hist = OPT.run_admm(init_Z=init_Z, solve_log=log, verbose=False)
    wall = time.perf_counter() - started
    log = np.array(log, dtype=float).reshape(-1, 3)
    iterations = int(len(hist))
    per_iteration_max = [float(log[log[:, 0] == i, 2].max()) for i in range(iterations)]
    solve_total = float(log[:, 2].sum())
    setup = max(wall - solve_total, 0.0)  # building the N models (done once, one per building controller)
    converged = iterations < OPT.MAX_ITERS
    elapsed = setup / N + np.cumsum(per_iteration_max)  # parallel estimate at the end of each iteration
    convergence = [{"start_hour": hour, "admm_start": start, "iteration": int(h[0]),
                    "primal_rms_kw": float(h[3]), "dual_rms_kw": float(h[4]), "parallel_elapsed_s": float(e)}
                   for h, e in zip(hist, elapsed)]
    summary = {"iterations": iterations, "converged": converged,
               "admm_parallel_s": sum(per_iteration_max) + setup / N,
               "admm_parallel_solve_s": sum(per_iteration_max), "admm_setup_per_building_s": setup / N,
               "admm_cost_gbp": _community_cost(subs)}
    return summary, convergence, _dispatch(subs, hour, f"ADMM ({start} start)")


def solve_central(hour):
    solver = pyo.SolverFactory(C.SOLVER)
    for option, value in (("OutputFlag", 0), ("LogFile", "")):
        try:
            solver.options[option] = value
        except Exception:
            pass
    started = time.perf_counter()
    m, subs = central.build_central_model()
    result = solver.solve(m, tee=False)
    wall = time.perf_counter() - started
    termination = result.solver.termination_condition
    if termination not in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.locallyOptimal):
        raise RuntimeError(f"Central solve failed: {termination}")
    return {"central_s": wall, "central_cost_gbp": _community_cost(subs)}, _dispatch(subs, hour, "Central")


# ============================================================================
# FIGURES
# ============================================================================
def plot_inputs(data, how):
    minutes = np.arange(HOURS * 60) / 60.0
    building = int(np.argmax(C.PV_B.sum(axis=1)))
    panels = [
        ("Load, building " + str(C.BUILDING_IDS[building]) + " (kW)", C.LOAD[building], data["LOAD"][building]),
        ("PV, building " + str(C.BUILDING_IDS[building]) + " (kW)", C.PV_B[building], data["PV_B"][building]),
        ("Outdoor temperature (°C)", C.T_OUT, data["T_OUT"]),
        ("Tariff (p/kWh)", C.PRICE * 100, data["PRICE"] * 100),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 8), sharex=True)
    for axis, (title, hourly, minute) in zip(axes, panels):
        _style(axis)
        axis.step(np.arange(HOURS + 1), np.append(hourly, hourly[-1]), where="post", color=PLAN_COLOR, lw=1.5,
                  label="Hourly data (hour average)")
        axis.plot(minutes, minute, color=COLORS["rolling_admm"], lw=1.2, label="Minute data")
        axis.set_title(title, loc="left", fontsize=9, color=INK)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[-1].set_xlabel("Hour of day", fontsize=9, color=INK)
    axes[-1].set_xticks(range(0, HOURS + 1, 2))
    fig.suptitle(f"Hourly inputs turned into 1-minute data ({how} interpolation; tariff and setpoint are steps)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, INPUTS_FIGURE)
    plt.close(fig)


def plot_timing(table, minutes):
    hours = sorted(table["start_hour"].unique())
    starts = list(dict.fromkeys(table["admm_start"]))
    x = np.arange(len(hours))
    width = 0.8 / (len(starts) + 1)
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    _style(left)
    left.set_yscale("log")
    shades = {"cold": COLORS["rolling_admm"], "warm": "#0f6b4a"}
    for i, start in enumerate(starts):
        part = table[table["admm_start"] == start].set_index("start_hour").reindex(hours)
        left.bar(x + (i - len(starts) / 2) * width, part["admm_parallel_s"], width=width * 0.92,
                 color=shades.get(start, COLORS["rolling_admm"]), hatch="//" if start == "warm" else None,
                 edgecolor="white", linewidth=0, label=f"ADMM, {start} start (parallel estimate)")
    reference = table.groupby("start_hour")["central_s"].median().reindex(hours)
    left.bar(x + (len(starts) / 2) * width, reference, width=width * 0.92, color=COLORS["rolling_central"],
             edgecolor="white", linewidth=0, label="Central (one solve)")
    for seconds, label in ((60.0, "1 min (re-solving every minute)"), (3600.0, "1 h")):
        left.axhline(seconds, color=PLAN_COLOR, lw=1, ls=":")
        left.annotate(label, (1.0, seconds), xycoords=("axes fraction", "data"), xytext=(-2, 2),
                      textcoords="offset points", ha="right", va="bottom", fontsize=7, color=PLAN_COLOR)
    left.set_ylim(top=3600 * 30)
    left.set_xticks(x)
    left.set_xticklabels([f"{h:02d}:00" for h in hours], fontsize=8)
    left.set_xlabel("Hour optimised (start time)", fontsize=9, color=INK)
    left.set_ylabel(f"Time to optimise {minutes} minutes (s, log scale)", fontsize=9, color=INK)
    left.set_title("a) Computation time", loc="left", fontsize=10, color=INK)
    left.legend(frameon=False, fontsize=7, loc="upper left")
    left.grid(False, axis="x")

    _style(right)
    width = 0.8 / max(len(starts), 1)
    for i, start in enumerate(starts):
        part = table[table["admm_start"] == start].set_index("start_hour").reindex(hours)
        right.bar(x + (i - (len(starts) - 1) / 2) * width, part["iterations"], width=width * 0.92,
                  color=shades.get(start, COLORS["rolling_admm"]), hatch="//" if start == "warm" else None,
                  edgecolor="white", linewidth=0, label=f"{start} start")
    right.axhline(OPT.MAX_ITERS, color=PLAN_COLOR, lw=1, ls=":")
    right.set_xticks(x)
    right.set_xticklabels([f"{h:02d}:00" for h in hours], fontsize=8)
    right.set_xlabel("Hour optimised (start time)", fontsize=9, color=INK)
    right.set_ylabel("ADMM iterations to converge", fontsize=9, color=INK)
    right.set_title(f"b) Iterations (cap {OPT.MAX_ITERS}, dotted)", loc="left", fontsize=10, color=INK)
    right.legend(frameon=False, fontsize=8)
    right.grid(False, axis="x")
    fig.suptitle(f"Bilateral ADMM on one hour of 1-minute data ({minutes} steps, N = {N} buildings)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, TIMING_FIGURE)
    plt.close(fig)


START_COLOURS = {"cold": COLORS["rolling_admm"], "warm": "#0f6b4a"}


def plot_convergence(history, table):
    """ADMM residuals against elapsed time (parallel estimate), cold and warm, for each start hour."""
    hours = sorted(history["start_hour"].unique())
    fig, axes = plt.subplots(1, len(hours), figsize=(6.2 * len(hours), 4.8), squeeze=False, sharey=True)
    for axis, hour in zip(axes[0], hours):
        _style(axis)
        axis.set_yscale("log")
        for start, part in history[history["start_hour"] == hour].groupby("admm_start"):
            colour = START_COLOURS.get(start, COLORS["rolling_admm"])
            axis.plot(part["parallel_elapsed_s"], part["primal_rms_kw"], color=colour, lw=2,
                      label=f"{start} start: consensus gap (primal)")
            axis.plot(part["parallel_elapsed_s"], part["dual_rms_kw"], color=colour, lw=1.5, ls="--",
                      label=f"{start} start: change in agreed trades (dual)")
            end = part.iloc[-1]
            axis.annotate(f"{start}: {int(end['iteration']) + 1} it., {end['parallel_elapsed_s']:.1f} s",
                          (end["parallel_elapsed_s"], end["primal_rms_kw"]), xytext=(4, -10),
                          textcoords="offset points", fontsize=7, color=INK)
        central_s = float(table.loc[table["start_hour"] == hour, "central_s"].iloc[0])
        axis.axvline(central_s, color=COLORS["rolling_central"], lw=1.5, ls=":")
        axis.annotate(f"central solve {central_s:.1f} s", (central_s, 1.0), xycoords=("data", "axes fraction"),
                      xytext=(3, -12), textcoords="offset points", fontsize=7, color=INK)
        axis.axhline(OPT.EPS_PRIMAL, color=PLAN_COLOR, lw=1, ls=":")
        axis.annotate(f"stopping tolerance {OPT.EPS_PRIMAL:g} kW", (1.0, OPT.EPS_PRIMAL),
                      xycoords=("axes fraction", "data"), xytext=(-2, 3), textcoords="offset points",
                      ha="right", fontsize=7, color=PLAN_COLOR)
        axis.set_xlabel("Elapsed time, parallel estimate (s)", fontsize=9, color=INK)
        axis.set_title(f"Hour starting {hour:02d}:00", loc="left", fontsize=10, color=INK)
    axes[0][0].set_ylabel("RMS residual (kW, log scale)", fontsize=9, color=INK)
    axes[0][0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle(f"How ADMM converges on one hour of 1-minute data (N = {N})", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, CONVERGENCE_FIGURE)
    plt.close(fig)


def plot_dispatch(dispatch, table):
    """The optimised hour minute by minute: ADMM against central."""
    hours = sorted(dispatch["start_hour"].unique())
    panels = [("grid_import_kw", "Community grid import (kW)"),
              ("battery_discharge_minus_charge_kw", "Batteries: discharge − charge (kW)"),
              ("p2p_traded_kw", "P2P energy traded (kW)\n(routing is not unique:\nsame total cost)"),
              ("mean_indoor_temperature_c", "Mean indoor temperature (°C)")]
    styles = {"Central": {"color": COLORS["rolling_central"], "lw": 3, "alpha": 0.6},
              "ADMM (cold start)": {"color": START_COLOURS["cold"], "lw": 1.5},
              "ADMM (warm start)": {"color": START_COLOURS["warm"], "lw": 1.2, "ls": "--"}}
    fig, axes = plt.subplots(len(panels), len(hours), figsize=(6.5 * len(hours), 9), sharex=True, squeeze=False)
    for c, hour in enumerate(hours):
        part = dispatch[dispatch["start_hour"] == hour]
        for r, (column, label) in enumerate(panels):
            axis = axes[r][c]
            _style(axis)
            for method, rows in part.groupby("method", sort=False):
                axis.plot(rows["minute"], rows[column], label=method, **styles.get(method, {}))
            if column == "mean_indoor_temperature_c":
                setpoint = part[part["method"] == "Central"]
                axis.step(setpoint["minute"], setpoint["setpoint_c"], where="post", color=PLAN_COLOR, lw=1,
                          ls=":", label="Setpoint")
            if column == "grid_import_kw":
                inputs = part[part["method"] == "Central"]
                axis.plot(inputs["minute"], inputs["load_kw"], color=PLAN_COLOR, lw=1, ls="-.",
                          label="Community load (excl. heat pumps)")
            axis.ticklabel_format(axis="y", useOffset=False, style="plain")
            low, high = axis.get_ylim()
            if high - low < 1.0:  # tiny numerical wiggles: show them on a sensible scale
                middle = (high + low) / 2
                axis.set_ylim(middle - 1.0, middle + 1.0)
            if c == 0:
                axis.set_ylabel(label, fontsize=8, color=INK)
        price = 100 * float(C.PRICE[hour])
        axes[0][c].set_title(f"Hour starting {hour:02d}:00 (tariff {price:.0f} p/kWh)", loc="left",
                             fontsize=10, color=INK)
        axes[-1][c].set_xlabel("Minute of the hour", fontsize=9, color=INK)
    handles, labels = [], []
    for axis in axes[:, 0]:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"The optimised hour at 1-minute resolution: ADMM against central (N = {N})",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    _save(fig, DISPATCH_FIGURE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", nargs="+", default=[str(h) for h in DEFAULT_HOURS],
                        help="Start hours to optimise (0-23), or 'all'.")
    parser.add_argument("--minutes", type=int, default=60, help="Horizon in one-minute steps.")
    parser.add_argument("--starts", nargs="+", choices=("cold", "warm"), default=["cold", "warm"])
    parser.add_argument("--interpolation", choices=("linear", "step"), default="linear")
    parser.add_argument("--plot-only", action="store_true", help="Redraw the figures from the saved CSVs.")
    args = parser.parse_args()
    if args.plot_only:
        table = pd.read_csv(TIMING_FILE)
        plot_inputs(minute_inputs(table["interpolation"].iloc[0]), table["interpolation"].iloc[0])
        plot_timing(table, int(table["minutes"].iloc[0]))
        plot_convergence(pd.read_csv(CONVERGENCE_FILE), table)
        plot_dispatch(pd.read_csv(DISPATCH_FILE), table)
        print(f"Redrew the figures in {RESULTS_DIR}")
        return
    hours = list(range(HOURS)) if args.hours == ["all"] else sorted({int(h) for h in args.hours})
    if not all(0 <= h < HOURS and h * 60 + args.minutes <= HOURS * 60 for h in hours):
        parser.error("start hours must leave a full horizon inside the day")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    schedule = baseline.load_or_solve_schedule(PLAN_FILE)
    data = minute_inputs(args.interpolation)
    plot_inputs(data, args.interpolation)
    rows, convergence, dispatch = [], [], []
    for hour in hours:
        soc, indoor = plan_state(schedule, hour)
        with minute_window(data, hour, args.minutes, soc, indoor):
            reference, central_dispatch = solve_central(hour)
            dispatch.extend(central_dispatch)
            for start in args.starts:
                result, history, admm_dispatch = solve_admm(schedule, hour, start)
                convergence.extend(history)
                dispatch.extend(admm_dispatch)
                row = {"start_hour": hour, "minutes": args.minutes, "n_buildings": N, "admm_start": start,
                       "interpolation": args.interpolation, **result, **reference}
                row["cost_gap_admm_minus_central_gbp"] = row["admm_cost_gbp"] - row["central_cost_gbp"]
                rows.append(row)
                print(f"{hour:02d}:00 {start:>4}: {row['iterations']:4d} iterations, ADMM parallel "
                      f"{row['admm_parallel_s']:7.2f} s, central {row['central_s']:5.2f} s, "
                      f"cost gap {row['cost_gap_admm_minus_central_gbp']:+.4f} GBP", flush=True)
            pd.DataFrame(rows).to_csv(TIMING_FILE, index=False)
            pd.DataFrame(convergence).to_csv(CONVERGENCE_FILE, index=False)
            pd.DataFrame(dispatch).to_csv(DISPATCH_FILE, index=False)
    table = pd.DataFrame(rows)
    plot_timing(table, args.minutes)
    plot_convergence(pd.DataFrame(convergence), table)
    plot_dispatch(pd.DataFrame(dispatch), table)
    print(f"\n{TIMING_FILE}\n{TIMING_FIGURE}\n{CONVERGENCE_FIGURE}\n{DISPATCH_FIGURE}\n{INPUTS_FIGURE}")
    print(table.groupby("admm_start")[["iterations", "admm_parallel_s", "central_s"]].describe().round(2).T)


if __name__ == "__main__":
    main()
