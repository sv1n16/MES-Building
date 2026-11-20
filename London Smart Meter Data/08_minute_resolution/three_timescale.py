"""Three-timescale rolling optimisation with bilateral ADMM: day-ahead, hourly, minute (N = 10).

PURPOSE
    Can bilateral ADMM run a community at three time resolutions at once (plan the day,
    re-plan every hour, control every minute), and is each solve fast enough for its step?

OBJECTIVE AT EACH LEVEL  (per building b; constraints = building model B_b, see
central_optimisation_showcase; ADMM adds λ q + ρ/2 (q − z)² to every level)

    day-ahead / hourly   (Δt = 1 h, hours t = h..23)
        min  Σ_t [ π_t g_t Δt + c_gas G_t Δt + α (T_t − T^set_t)² + w_t g_t² ]

    minute               (Δt = 1/60 h, minutes t = m..59 of hour h; track_hourly_plan)
        min  Σ_t [ π_t g_t Δt + c_gas G_t Δt + α Δt (T_t − T^set_t)² + w_t Δt g_t² ]    economic (tie-breaker)
         --tracking trades (default):
           + W Δt Σ_t Σ_j (q_j,t − q_j^plan)²                                       follow sharing plan
           + W_soc (s_59 − s^plan)² + W_T (T_59 − T^plan)²                              end-of-hour state
         --tracking grid:
           + W_g Δt Σ_t (g_t − g^plan)²                     follow the plan's grid import; trades re-route freely
           − μ_soc s_59 − μ_T T_59                           end-of-hour state valued for the rest of the day
           + W_soc (s_59 − s^plan)² + W_T (T_59 − T^plan)²   (small, for stability)
        ^plan = the hourly solution for hour h; μ = its shadow prices of battery charge (£/kWh) and
        indoor temperature (£/°C) at the end of hour h (duals of the dynamics into hour h+1, needs
        --parallel). W = --tracking-weight (10), W_g = --grid-tracking-weight (10); W_soc = --soc-weight
        (100 trades / 1 grid), W_T = --temperature-weight (10 trades / 1 grid), all £ per unit².
        No ramp-rate control here: 09_ramp_rate_control/three_timescale_ramp_limits.py adds it
        through the EXTENSION hook (see below).

1. Day-ahead   one ADMM solve of the whole day at hourly resolution (24 steps).
2. Hourly      at the start of every hour h, ADMM re-solves the remaining hours h..23 at
               hourly resolution, from the state actually reached. Warm start: the
               day-ahead plan for the first hour, then the previous hourly solution
               (trades and dual prices, shifted by one hour).
3. Minute      within hour h, at every minute m, ADMM re-solves the remaining minutes
               m..59 of the hour at 1-minute resolution and applies the first minute. Its
               goal is to stay within the hourly plan: each building tracks the hourly
               solution for hour h, with quadratic penalties on deviating from each of
               its P2P trades (grid import and battery power are not tracked),
               and on missing the plan's battery charge and indoor temperature at the end
               of the hour. The economic cost stays in the objective as a tie-breaker, so
               the minute level only departs from the plan where the minute data forces it
               (load and PV vary within the hour). Warm start: the hourly solution for hour
               h at minute 0 (its trades, and its dual prices scaled to the 1-minute step),
               then the previous minute's solution shifted by one minute. The state reached
               at the end of the hour is the starting state of the next hourly solve.

The inputs are the day's data; the minute inputs are made from them as in
admm_minute_timing.py (linear interpolation of load, PV and outdoor temperature, tariff and
setpoint held within each hour), so there is no forecast error: the levels differ only in
time resolution. Every solve is timed with the parallel estimate (slowest building per
iteration, summed over iterations, plus model build / N), or, with --parallel, measured with
the building subproblems solved at the same time by worker processes (one per building).

With --error, what actually happens differs from the forecast:
  hourly level  knows the actual average of the current hour (an experiment-05 noise scenario,
                --hourly-case base or correlated); later hours stay at the forecast;
  minute level  measures the actual current minute: the hourly actuals plus minute-level
                fluctuation of load and PV (--minute-load-noise, --minute-pv-noise), which can be
                correlated across buildings (--minute-correlation) and in time
                (--minute-time-constant, AR(1)); after it, it assumes the hourly level's values.
--correlated sets the correlated case: hourly case "correlated" (80% of the error common to all
buildings), minute correlation 0.8 and a 10-minute time constant. Outputs get the suffixes
_with_error[_correlated][_parallel].

Outputs (Results/08_minute_resolution/1_three_timescale/):
    85_three_timescale_timing.csv     one row per solve: level, hour, minute, horizon, iterations,
                                      parallel time
    86_three_timescale_dispatch.csv   per minute: community grid import, batteries, P2P, indoor
                                      temperature, and the hourly plan's values for the same hour
    87_three_timescale_timing.png     time per solve at each level through the day, against the
                                      1-minute and 1-hour lines, and the iterations
    88_three_timescale_dispatch.png   the day at minute resolution against the hourly and
                                      day-ahead plans

Run it while the machine is otherwise idle and kept awake.

Examples:
    python three_timescale.py --hours 17            # one hour of minutes (60 minute solves)
    python three_timescale.py --hours 16 17 18 19
    python three_timescale.py                       # the whole day (24 hourly, 1440 minute solves)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import project_paths  # noqa: E402  (shared modules on sys.path; results -> Results/<experiment>/)

project_paths.register(__file__)

import argparse  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager, nullcontext  # noqa: E402

import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyomo.environ as pyo  # noqa: E402

import simulate_realtime_reoptimise as reopt  # noqa: E402
from admm_minute_timing import OPT, C, minute_inputs, minute_window_at, to_minutes  # noqa: E402
from plot_realtime_comparison import COLORS, INK, PLAN_COLOR, _save, _style  # noqa: E402

N, HOURS = C.n_buildings, C.time_horizon
RESULTS_DIR = C.PLOTS_DIR
SCENARIO_FILE = project_paths.RESULTS_DIR / "05_realtime_methods" / "shared_noise_scenarios.npz"
LEVEL_COLOURS = {"day-ahead": PLAN_COLOR, "hourly": "#0f6b4a", "minute": COLORS["rolling_admm"]}


# ============================================================================
# EXTENSION HOOK
# ============================================================================
# None here. Another experiment can set three_timescale.EXTENSION to an object before calling
# main() to add options, minute-level objective terms, per-minute records and figures without
# changing this file (09_ramp_rate_control/three_timescale_ramp_limits.RampLimits does). Methods:
#   add_arguments(parser)                     extra command-line options
#   set_outputs(folder, suffix)               its own output files
#   start_day(args)                           after the day-ahead solve
#   start_hour(hourly, args)                  before the minutes of an hour (hourly = plan for it)
#   minute_plan(hourly, minute, args) -> plan the plan passed to the minute solve; may add
#                                             plan["worker_tracking"] (merged into the parallel
#                                             workers' tracking data, see admm_parallel._build)
#   pyomo_terms(m, b, plan, dt) -> expr       extra objective terms for building b (sequential solver)
#   end_minute(step, hour, minute, args)      after a minute is applied
#   save(), plot(args)                        write its records / draw its figures
EXTENSION = None


def set_outputs(suffix="", directory=None):
    """Output files; runs with forecast error get their own (suffix '_with_error')."""
    global TIMING_FILE, DISPATCH_FILE, TIMING_FIGURE, DISPATCH_FIGURE
    folder = Path(directory) if directory else RESULTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    TIMING_FILE = folder / f"85_three_timescale_timing{suffix}.csv"
    DISPATCH_FILE = folder / f"86_three_timescale_dispatch{suffix}.csv"
    TIMING_FIGURE = folder / f"87_three_timescale_timing{suffix}.png"
    DISPATCH_FIGURE = folder / f"88_three_timescale_dispatch{suffix}.png"
    if EXTENSION is not None:
        EXTENSION.set_outputs(folder, suffix)


set_outputs()


# ============================================================================
# FORECAST ERROR
# ============================================================================
def error_inputs(args, forecast_minutes):
    """What actually happens, and what each level knows about it.

    Returns (hourly_actual, hour_info, actual):
    hourly_actual  actual hourly load, PV and outdoor temperature (experiment 05's shared noise
                   scenario --scenario): the hourly level knows the current hour's values
    hour_info      the same at minute resolution: what the minute level assumes for the
                   rest of the hour
    actual         hour_info plus independent minute-level fluctuation of load and PV
                   (--minute-load-noise, --minute-pv-noise): measured minute by minute
    Without --error all three are the forecast.
    """
    if not args.error:
        hourly = {"LOAD": reopt.FORECAST_LOAD, "PV_B": reopt.FORECAST_PV, "T_OUT": reopt.FULL_T_OUT}
        return hourly, forecast_minutes, forecast_minutes
    scenario_file = SCENARIO_FILE if args.hourly_case == "base" else (
        SCENARIO_FILE.parent / "cases" / args.hourly_case / SCENARIO_FILE.name)
    with np.load(scenario_file, allow_pickle=False) as stored:
        run = args.scenario - 1
        hourly = {"LOAD": stored["actual_load"][run].copy(), "PV_B": stored["actual_pv"][run].copy(),
                  "T_OUT": stored["actual_t_out"][run].copy()}
    hour_info = dict(forecast_minutes)
    hour_info["LOAD"] = np.maximum(to_minutes(hourly["LOAD"], 60, args.interpolation), 0.0)
    hour_info["PV_B"] = np.maximum(to_minutes(hourly["PV_B"], 60, args.interpolation), 0.0)
    hour_info["T_OUT"] = to_minutes(hourly["T_OUT"], 60, args.interpolation)
    rng = np.random.default_rng(args.error_seed)
    actual = dict(hour_info)
    for name, noise in (("LOAD", args.minute_load_noise), ("PV_B", args.minute_pv_noise)):
        shocks = minute_noise(rng, hour_info[name].shape, args.minute_correlation, args.minute_time_constant)
        actual[name] = np.maximum(hour_info[name] * (1 + noise * shocks), 0.0)
    if args.cloud_transients:
        actual["PV_B"] = actual["PV_B"] * cloud_shading(np.random.default_rng(args.cloud_seed),
                                                        actual["PV_B"].shape, args)
    return hourly, hour_info, actual


def cloud_shading(rng, shape, args):
    """PV multiplier (N, minutes) for passing clouds that no forecast sees.

    Clouds arrive at random (Poisson, --cloud-rate per hour, 07:00-19:00). Each cuts PV by a
    depth drawn from --cloud-depth (fraction) for a duration drawn from --cloud-duration
    (minutes), with the drop and the recovery each within one minute. All buildings are
    shaded by the same cloud, each 0-2 minutes after the first as it passes over.
    """
    buildings, minutes = shape
    shading = np.ones(shape)
    start, end = 7 * 60, 19 * 60
    events = rng.poisson(args.cloud_rate * (end - start) / 60.0)
    for _ in range(events):
        onset = int(rng.integers(start, end))
        depth = rng.uniform(*args.cloud_depth)
        duration = int(rng.integers(args.cloud_duration[0], args.cloud_duration[1] + 1))
        delays = rng.integers(0, 3, size=buildings)
        for building, delay in enumerate(delays):
            first = min(onset + int(delay), minutes)
            last = min(first + duration, minutes)
            shading[building, first:last] = np.minimum(shading[building, first:last], 1.0 - depth)
    return shading


def minute_noise(rng, shape, correlation, time_constant):
    """Unit-variance minute shocks (N, minutes), correlated across buildings and in time.

    correlation: share of the variance common to all buildings (as in experiment 05).
    time_constant: minutes; > 0 makes each series an AR(1) process with coefficient
    exp(-1 / time_constant), so fluctuations persist instead of jumping every minute.
    """
    buildings, minutes = shape
    shocks = (np.sqrt(correlation) * rng.standard_normal((1, minutes))
              + np.sqrt(1.0 - correlation) * rng.standard_normal((buildings, minutes)))
    if time_constant > 0:
        phi = np.exp(-1.0 / time_constant)
        persistent = np.empty_like(shocks)
        persistent[:, 0] = shocks[:, 0]
        for t in range(1, minutes):
            persistent[:, t] = phi * persistent[:, t - 1] + np.sqrt(1.0 - phi**2) * shocks[:, t]
        shocks = persistent
    return shocks


def minute_view(hour_info, actual, minute_of_day):
    """Inputs of a minute solve: the measured current minute, the hourly level's view after it."""
    view = dict(hour_info)
    for name in ("LOAD", "PV_B"):
        view[name] = hour_info[name].copy()
        view[name][:, minute_of_day] = actual[name][:, minute_of_day]
    view["T_OUT"] = hour_info["T_OUT"].copy()
    view["T_OUT"][minute_of_day] = actual["T_OUT"][minute_of_day]
    return view


# ============================================================================
# ADMM WITH TIMING
# ============================================================================
def timed_admm(init_lam=None, init_Z=None):
    """OPT.run_admm on the current window, with the parallel-estimate time."""
    log = []
    started = time.perf_counter()
    subs, _q, lam, Z, hist = OPT.run_admm(init_lam=init_lam, init_Z=init_Z, solve_log=log, verbose=False)
    wall = time.perf_counter() - started
    log = np.array(log, dtype=float).reshape(-1, 3)
    iterations = int(len(hist))
    per_iteration_max = [float(log[log[:, 0] == i, 2].max()) for i in range(iterations)]
    setup = max(wall - float(log[:, 2].sum()), 0.0)
    return subs, lam, Z, {"iterations": iterations, "parallel_s": sum(per_iteration_max) + setup / N,
                          "converged": iterations < OPT.MAX_ITERS}


STEP_NAMES = ("soc", "T_in", "p_el", "charge", "discharge", "gas")


def values_from_subs(subs):
    """Per-building solution arrays: name -> (N, T); q and qpos -> (N, N-1, T), + = delivers."""
    values = {name: np.array([[float(pyo.value(getattr(m, name)[t])) for t in m.t] for m in subs])
              for name in STEP_NAMES}
    for name in ("q", "qpos", "qneg"):
        values[name] = np.array([[[float(pyo.value(getattr(m, name)[k, t])) for t in m.t] for k in m.k]
                                 for m in subs])
    return values


def first_step(values):
    step = {name: values[name][:, 0] for name in STEP_NAMES}
    step["q"] = values["q"][:, :, 0]
    step["p2p"] = float(values["qpos"][:, :, 0].sum())
    step["sold"] = values["qpos"][:, :, 0].sum(axis=1)  # per building, kW
    step["bought"] = values["qneg"][:, :, 0].sum(axis=1)
    return step


class SequentialSolver:
    """Pyomo ADMM (admm_bilateral_p2p_valley.run_admm), timed with the parallel estimate."""

    kind = "parallel estimate"

    def solve(self, init_lam, init_Z, plan, args):
        with (track_hourly_plan(plan, args) if plan is not None else nullcontext()):
            subs, lam, Z, info = timed_admm(init_lam=init_lam, init_Z=init_Z)
        return values_from_subs(subs), lam, Z, info

    def close(self):
        pass


class ParallelSolver:
    """The building subproblems solved at the same time by worker processes (05 admm_parallel.ParallelADMM).

    One persistent gurobipy worker per building by default (--workers); the reported time is
    the measured wall time of the ADMM run and of collecting the solution.
    """

    def __init__(self, workers):
        import admm_parallel

        self.parallel = admm_parallel
        self.pool = admm_parallel.ParallelADMM(N, workers)
        self.kind = f"measured, {len(self.pool.assignment)} parallel workers"

    def solve(self, init_lam, init_Z, plan, args):
        extra = self.parallel.EXTRA_DATA
        extra.clear()
        if plan is not None:
            extra["tracking"] = {
                "mode": args.tracking, "weight": args.tracking_weight, "q": plan["q"], "soc": plan["soc"],
                "T_in": plan["T_in"], "soc_weight": args.soc_weight, "temperature_weight": args.temperature_weight,
            }
            if args.tracking == "grid":
                extra["tracking"].update(grid_weight=args.grid_tracking_weight, p_el=plan["p_el"],
                                         soc_value=plan["soc_value"], temperature_value=plan["temperature_value"])
            extra["tracking"].update(plan.get("worker_tracking", {}))  # extension data, if any
        log = []
        started = time.perf_counter()
        try:
            _q, lam, Z, hist = self.pool.run(init_lam=init_lam, init_Z=init_Z, solve_log=log, verbose=False)
            solved = self.pool.extract(STEP_NAMES + ("q", "qpos", "qneg"))
            # the hourly plan's shadow prices of the end-of-hour state, for --tracking grid
            state_values = self.pool.state_values() if plan is None and args.tracking == "grid" else None
        finally:
            extra.clear()
        wall = time.perf_counter() - started
        log = np.array(log, dtype=float).reshape(-1, 3)
        iterations = int(len(hist))
        estimate = sum(float(log[log[:, 0] == i, 2].max()) for i in range(iterations))
        values = {name: np.array([solved[b][name] for b in range(N)]) for name in STEP_NAMES + ("q", "qpos", "qneg")}
        if state_values is not None:
            values["soc_value"], values["temperature_value"] = state_values[:, 0], state_values[:, 1]
        return values, lam, Z, {"iterations": iterations, "parallel_s": wall, "estimate_s": estimate,
                                "converged": iterations < OPT.MAX_ITERS}

    def close(self):
        self.pool.close()


@contextmanager
def track_hourly_plan(plan, args):
    """Minute subproblems that stay within the hourly sharing plan for this hour (see the module docstring).

    plan: the hourly solution's first hour (per building q, soc, T_in; extension data optional).
    The tracking penalty is per minute and scaled by the step, in GBP per kW^2 per hour.
    """
    original = OPT.build_subproblem

    def build_subproblem(b):
        m = original(b)
        last, dt, w = OPT.T - 1, OPT.DT, args.tracking_weight
        if args.tracking == "grid":
            # grid import only:  W_g Δt Σ_t (g_t − g^plan)², and the end-of-hour state at the plan's shadow prices
            tracking = sum(args.grid_tracking_weight * dt * (m.p_el[t] - float(plan["p_el"][b])) ** 2 for t in m.t)
            tracking += -float(plan["soc_value"][b]) * m.soc[last] - float(plan["temperature_value"][b]) * m.T_in[last]
        else:
            # sharing plan only:  W Δt Σ_t Σ_j (q_j,t − q_j^plan)²   (grid import and battery are free to move)
            tracking = sum(w * dt * sum((m.q[k, t] - float(plan["q"][b, k])) ** 2 for k in m.k) for t in m.t)
        # end-of-hour state handed to the next hourly solve:  W_soc (s_end − s^plan)² + W_T (T_end − T^plan)²
        extra = (args.soc_weight * (m.soc[last] - float(plan["soc"][b])) ** 2
                 + args.temperature_weight * (m.T_in[last] - float(plan["T_in"][b])) ** 2)
        if EXTENSION is not None:  # e.g. ramp-limit penalties (09_ramp_rate_control)
            extra += EXTENSION.pyomo_terms(m, b, plan, dt)
        m.obj.set_value(m.obj.expr + tracking + extra)
        return m

    OPT.build_subproblem = build_subproblem
    try:
        yield
    finally:
        OPT.build_subproblem = original


@contextmanager
def admm_tolerance(eps):
    """ADMM stopping tolerances (primal and dual RMS, kW) set to eps for the solves inside."""
    old = OPT.EPS_PRIMAL, OPT.EPS_DUAL
    OPT.EPS_PRIMAL = OPT.EPS_DUAL = eps
    try:
        yield
    finally:
        OPT.EPS_PRIMAL, OPT.EPS_DUAL = old


# ============================================================================
# THE THREE LEVELS
# ============================================================================
def run(hours, args):
    solver = ParallelSolver(args.workers) if args.parallel else SequentialSolver()
    try:
        return _run(hours, args, solver)
    finally:
        solver.close()


def _run(hours, args, solver):
    data = minute_inputs(args.interpolation)  # the forecast at minute resolution
    hourly_actual, hour_info, actual = error_inputs(args, data)
    timing, dispatch = [], []

    # 1. day-ahead (whole day, from the forecast): hourly resolution (default) or minute resolution
    #    (--day-ahead-resolution minute), stopped at --day-ahead-tolerance
    minute_day_ahead = args.day_ahead_resolution == "minute"
    steps = HOURS * 60 if minute_day_ahead else HOURS
    print(f"Day-ahead solve ({steps} steps, tolerance {args.day_ahead_tolerance:g}, {solver.kind}) ...", flush=True)
    window = (minute_window_at(data, 0, steps, np.asarray(C.INIT_SOC), np.asarray(C.INIT_T_IN))
              if minute_day_ahead else nullcontext())
    with window, admm_tolerance(args.day_ahead_tolerance):
        day_values, day_lam, day_Z, info = solver.solve(None, None, None, args)
    timing.append({"level": "day-ahead", "hour": 0, "minute": 0, "horizon_steps": steps, "timing": solver.kind,
                   **info})
    print(f"  day-ahead: {info['iterations']} iterations, {info['parallel_s']:.2f} s", flush=True)
    if minute_day_ahead:
        # hourly view of the minute plan: mean power per hour, state at the end of each hour; as the
        # hourly warm start: trades averaged per hour, prices (per 1-minute step) summed per hour
        day_ahead = {"p_el": day_values["p_el"].reshape(N, HOURS, 60).mean(axis=2),
                     "soc": day_values["soc"][:, 59::60], "T_in": day_values["T_in"][:, 59::60]}
        day_Z = {pair: np.asarray(z).reshape(HOURS, 60).mean(axis=1) for pair, z in day_Z.items()}
        day_lam = [np.asarray(lam).reshape(N - 1, HOURS, 60).sum(axis=2) for lam in day_lam]
    else:
        day_ahead = {name: day_values[name] for name in ("p_el", "soc", "T_in")}

    soc = np.asarray(C.INIT_SOC, dtype=float).copy()
    indoor = np.asarray(C.INIT_T_IN, dtype=float).copy()
    previous_Z = {pair: day_Z[pair].copy() for pair in OPT.PAIRS}
    previous_lam = [lam.copy() for lam in day_lam]
    previous_hour = 0
    if EXTENSION is not None:
        EXTENSION.start_day(args)
    for hour in range(0, max(hours) + 1):  # hours before the first minute-level one roll at hourly level only
        # 2. hourly solve of the remaining hours, from the state actually reached
        shift = hour - previous_hour
        init_Z = {pair: values[shift:] for pair, values in previous_Z.items()}
        init_lam = [lam[:, shift:] for lam in previous_lam]
        # the current hour's actual values, the forecast for the later hours
        inputs = reopt.horizon_inputs(hour, hourly_actual["LOAD"], hourly_actual["PV_B"], hourly_actual["T_OUT"],
                                      "forecast")
        with reopt.remaining_horizon(hour, *inputs, soc, indoor):
            values, lam, Z, info = solver.solve(init_lam, init_Z, None, args)
            hourly = first_step(values)
            if "soc_value" in values:  # --tracking grid: what the end-of-hour state is worth later in the day
                hourly["soc_value"], hourly["temperature_value"] = values["soc_value"], values["temperature_value"]
        previous_Z, previous_lam, previous_hour = Z, lam, hour
        timing.append({"level": "hourly", "hour": hour, "minute": 0, "horizon_steps": HOURS - hour,
                       "timing": solver.kind, **info})
        print(f"hour {hour:02d}: hourly {info['iterations']:3d} it. {info['parallel_s']:6.2f} s", flush=True)
        if hour not in hours:
            # an hour that is not simulated at minute level follows the hourly solution
            soc, indoor = hourly["soc"], hourly["T_in"]
            continue

        # 3. minute solves within the hour, warm-started from the hourly solution
        minute_Z = {pair: np.full(60, float(Z[pair][0])) for pair in OPT.PAIRS}
        minute_lam = [np.repeat(lam_b[:, :1] / 60.0, 60, axis=1) for lam_b in lam]  # price per 1-minute step
        minute_times = []
        if EXTENSION is not None:
            EXTENSION.start_hour(hourly, args)
        for minute in range(60):
            length = 60 - minute
            now = hour * 60 + minute
            view = minute_view(hour_info, actual, now) if args.error else data
            with minute_window_at(view, now, length, soc, indoor):
                plan = EXTENSION.minute_plan(hourly, minute, args) if EXTENSION is not None else hourly
                values, lam_m, Z_m, info = solver.solve(
                    [lam_b[:, -length:] for lam_b in minute_lam],
                    {pair: z[-length:] for pair, z in minute_Z.items()}, plan, args)
                step = first_step(values)
            if EXTENSION is not None:
                EXTENSION.end_minute(step, hour, minute, args)
            minute_Z, minute_lam = Z_m, lam_m
            soc, indoor = step["soc"], step["T_in"]
            minute_times.append(info["parallel_s"])
            timing.append({"level": "minute", "hour": hour, "minute": minute, "horizon_steps": length,
                           "timing": solver.kind, **info})
            dispatch.append({
                "hour": hour, "minute": minute, "minute_of_day": hour * 60 + minute,
                "grid_import_kw": float(step["p_el"].sum()),
                "battery_discharge_minus_charge_kw": float((step["discharge"] - step["charge"]).sum()),
                "p2p_traded_kw": step["p2p"], "battery_energy_kwh": float(step["soc"].sum()),
                "mean_indoor_temperature_c": float(step["T_in"].mean()),
                "setpoint_c": float(data["T_SET"][hour * 60 + minute]),
                "actual_load_kw": float(actual["LOAD"][:, now].sum()),
                "forecast_load_kw": float(data["LOAD"][:, now].sum()),
                "actual_pv_kw": float(actual["PV_B"][:, now].sum()),
                "forecast_pv_kw": float(data["PV_B"][:, now].sum()),
                "price_gbp_per_kwh": float(data["PRICE"][hour * 60 + minute]),
                # realised costs of this minute (GBP): electricity, gas, comfort α Δt Σ_b (T − T_set)²
                "electricity_cost_gbp": float(data["PRICE"][now] * step["p_el"].sum() / 60.0),
                "gas_cost_gbp": float(C.gas_price / 100.0 * step["gas"].sum() / 60.0),
                "comfort_cost_gbp": float(C.alpha / 60.0 * ((step["T_in"] - data["T_SET"][now]) ** 2).sum()),
                "hourly_plan_soc_value_gbp_per_kwh": (float(np.mean(hourly["soc_value"]))
                                                      if "soc_value" in hourly else np.nan),
                "hourly_plan_grid_import_kw": float(hourly["p_el"].sum()),
                "grid_deviation_from_plan_kw": float(np.abs(step["p_el"] - hourly["p_el"]).sum()),
                "battery_deviation_from_plan_kw": float(np.abs((step["discharge"] - step["charge"])
                                                                - (hourly["discharge"] - hourly["charge"])).sum()),
                "trade_deviation_from_plan_kw": float(np.abs(step["q"] - hourly["q"]).sum() / 2),
                "hourly_plan_battery_kw": float((hourly["discharge"] - hourly["charge"]).sum()),
                "hourly_plan_p2p_kw": float(np.maximum(hourly["q"], 0).sum()),
                "hourly_plan_battery_energy_kwh": float(hourly["soc"].sum()),
                "day_ahead_grid_import_kw": float(day_ahead["p_el"][:, hour].sum()),
                "day_ahead_battery_energy_kwh": float(day_ahead["soc"][:, hour].sum()),
            })
        print(f"          minutes: median {np.median(minute_times):.2f} s, max {max(minute_times):.2f} s; "
              f"end of hour battery {soc.sum():.2f} kWh (hourly target {hourly['soc'].sum():.2f})", flush=True)
        pd.DataFrame(timing).to_csv(TIMING_FILE, index=False)
        pd.DataFrame(dispatch).to_csv(DISPATCH_FILE, index=False)
        if EXTENSION is not None:
            EXTENSION.save()
    return pd.DataFrame(timing), pd.DataFrame(dispatch)


# ============================================================================
# FIGURES
# ============================================================================
PAUSE_FACTOR, PAUSE_FLOOR_S = 20.0, 60.0


def drop_pauses(timing):
    """Remove solves that include a machine sleep (Windows keeps the clock running).

    A solve counts as paused when it took over PAUSE_FACTOR times its level's median and
    over PAUSE_FLOOR_S seconds. Returns the cleaned table and the dropped rows.
    """
    median = timing.groupby("level")["parallel_s"].transform("median")
    paused = (timing["parallel_s"] > PAUSE_FACTOR * median) & (timing["parallel_s"] > PAUSE_FLOOR_S)
    return timing[~paused], timing[paused]


def plot_timing(timing):
    kind = timing["timing"].iloc[0] if "timing" in timing else "parallel estimate"
    timing, paused = drop_pauses(timing)
    if len(paused):
        print(f"Dropped {len(paused)} solve(s) that include a machine sleep: "
              + ", ".join(f"{r.level} {int(r.hour):02d}:{int(r.minute):02d} ({r.parallel_s:.0f} s)"
                          for r in paused.itertuples()))
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [2.2, 1]})
    _style(left)
    left.set_yscale("log")
    minute = timing[timing["level"] == "minute"]
    hourly = timing[timing["level"] == "hourly"]
    day = timing[timing["level"] == "day-ahead"]
    left.scatter(minute["hour"] + minute["minute"] / 60.0, minute["parallel_s"], s=6,
                 color=LEVEL_COLOURS["minute"], label="Minute solves (remaining minutes of the hour)")
    left.scatter(hourly["hour"], hourly["parallel_s"], s=40, marker="D", color=LEVEL_COLOURS["hourly"],
                 edgecolor="white", linewidth=0.8, zorder=3, label="Hourly solves (remaining hours)")
    if not day.empty:
        left.scatter([-0.5], day["parallel_s"], s=70, marker="s", color=LEVEL_COLOURS["day-ahead"],
                     edgecolor="white", linewidth=0.8, zorder=3, label="Day-ahead solve (whole day)")
    for seconds, label in ((60.0, "1 min"), (3600.0, "1 h")):
        left.axhline(seconds, color=PLAN_COLOR, lw=1, ls=":")
        left.annotate(label, (1.0, seconds), xycoords=("axes fraction", "data"), xytext=(-2, 2),
                      textcoords="offset points", ha="right", va="bottom", fontsize=7, color=PLAN_COLOR)
    left.set_ylim(top=3600 * 30)
    left.set_xlabel("Hour of day", fontsize=9, color=INK)
    left.set_ylabel(f"Time per solve, {kind} (s, log scale)", fontsize=9, color=INK)
    left.set_title("a) Every solve of the three levels through the day", loc="left", fontsize=10, color=INK)
    left.legend(frameon=False, fontsize=7, loc="upper left")

    _style(right)
    levels = [lv for lv in ("day-ahead", "hourly", "minute") if lv in set(timing["level"])]
    data = [timing.loc[timing["level"] == lv, "parallel_s"] for lv in levels]
    plot = right.boxplot(data, patch_artist=True, widths=0.5, medianprops={"color": INK},
                         flierprops={"marker": "o", "markersize": 2.5})
    for patch, level in zip(plot["boxes"], levels):
        patch.set_facecolor(LEVEL_COLOURS[level])
        patch.set_edgecolor("white")
    right.set_yscale("log")
    for seconds in (60.0, 3600.0):
        right.axhline(seconds, color=PLAN_COLOR, lw=1, ls=":")
    right.set_ylim(left.get_ylim())
    labels = []
    for level in levels:
        part = timing[timing["level"] == level]
        labels.append(f"{level}\n{len(part)} solve{'s' if len(part) != 1 else ''}\n"
                      f"median {part['parallel_s'].median():.2f} s\n{part['iterations'].median():.0f} it.")
    right.set_xticklabels(labels, fontsize=7)
    right.set_title("b) Per level", loc="left", fontsize=10, color=INK)
    right.grid(False, axis="x")
    fig.suptitle(f"Three-timescale bilateral ADMM: computation time per solve (N = {N}, {kind})",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, TIMING_FIGURE)
    plt.close(fig)


def plot_dispatch(dispatch):
    """The day at minute level against the hourly plan it tracks (and the day-ahead plan)."""
    t = dispatch["minute_of_day"] / 60.0
    hours = dispatch.groupby("hour").first()
    hour_edges = np.append(hours.index.to_numpy(), hours.index.max() + 1)

    def plan_steps(axis, column, colour, style, label):
        axis.step(hour_edges, np.append(hours[column].to_numpy(), hours[column].iloc[-1]), where="post",
                  color=colour, lw=1.5, ls=style, label=label)

    with_error = "actual_load_kw" in dispatch and not np.allclose(dispatch["actual_load_kw"],
                                                                  dispatch["forecast_load_kw"])
    fig, axes = plt.subplots(6 if with_error else 5, 1, figsize=(12, 14.5 if with_error else 12), sharex=True)
    if with_error:
        inputs, grid, battery, p2p, deviation, temperature = axes
    else:
        grid, battery, p2p, deviation, temperature = axes
    for axis in axes:
        _style(axis)
    if with_error:
        inputs.plot(t, dispatch["actual_load_kw"], color="#2a78d6", lw=0.8, label="Actual load (measured each minute)")
        inputs.plot(t, dispatch["forecast_load_kw"], color="#2a78d6", lw=1.5, ls=":", label="Forecast load")
        inputs.plot(t, dispatch["actual_pv_kw"], color="#eda100", lw=0.8, label="Actual PV")
        inputs.plot(t, dispatch["forecast_pv_kw"], color="#eda100", lw=1.5, ls=":", label="Forecast PV")
        inputs.set_ylabel("Community load and PV (kW)", fontsize=8, color=INK)
    minute_style = {"color": LEVEL_COLOURS["minute"], "lw": 1.3, "label": "Minute level (applied)"}
    grid.plot(t, dispatch["grid_import_kw"], **minute_style)
    plan_steps(grid, "hourly_plan_grid_import_kw", LEVEL_COLOURS["hourly"], "--", "Hourly plan for that hour")
    plan_steps(grid, "day_ahead_grid_import_kw", LEVEL_COLOURS["day-ahead"], ":", "Day-ahead plan")
    grid.set_ylabel("Community grid\nimport (kW)", fontsize=8, color=INK)
    battery.plot(t, dispatch["battery_discharge_minus_charge_kw"], **minute_style)
    plan_steps(battery, "hourly_plan_battery_kw", LEVEL_COLOURS["hourly"], "--", "Hourly plan for that hour")
    battery.set_ylabel("Batteries:\ndischarge − charge (kW)", fontsize=8, color=INK)
    p2p.plot(t, dispatch["p2p_traded_kw"], **minute_style)
    plan_steps(p2p, "hourly_plan_p2p_kw", LEVEL_COLOURS["hourly"], "--", "Hourly plan for that hour")
    p2p.set_ylabel("P2P traded (kW)", fontsize=8, color=INK)
    for column, colour, label in (("grid_deviation_from_plan_kw", "#2a78d6", "Grid import"),
                                  ("battery_deviation_from_plan_kw", "#eda100", "Battery power"),
                                  ("trade_deviation_from_plan_kw", "#e34948", "P2P trades")):
        deviation.plot(t, dispatch[column], color=colour, lw=1.2, label=f"Deviation: {label.lower()}")
    deviation.set_ylabel("Departure from the\nhourly plan, summed\nover buildings (kW)", fontsize=8, color=INK)
    temperature.plot(t, dispatch["mean_indoor_temperature_c"], **minute_style)
    temperature.plot(t, dispatch["setpoint_c"], color=PLAN_COLOR, lw=1, ls=":", label="Setpoint")
    temperature.set_ylabel("Mean indoor\ntemperature (°C)", fontsize=8, color=INK)
    for axis in axes:
        axis.ticklabel_format(axis="y", useOffset=False, style="plain")
    temperature.set_xlabel("Time of day", fontsize=9, color=INK)
    temperature.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x) % 24:02d}:{int(round((x % 1) * 60)) % 60:02d}"))
    handles, labels = [], []
    for axis in axes:
        for handle, text in zip(*axis.get_legend_handles_labels()):
            if text not in labels:
                handles.append(handle)
                labels.append(text)
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle(f"Three-timescale bilateral ADMM: the minute level tracking the hourly plan (N = {N})",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    _save(fig, DISPATCH_FIGURE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", nargs="+", type=int, default=list(range(HOURS)),
                        help="Hours simulated at minute level (default: the whole day). Other hours up to "
                             "the last one are rolled at hourly level only.")
    parser.add_argument("--interpolation", choices=("linear", "step"), default="linear")
    parser.add_argument("--day-ahead-resolution", choices=("hourly", "minute"), default="hourly",
                        help="Resolution of the day-ahead plan: 24 hourly steps, or 1440 one-minute steps (outputs "
                             "get the suffix _minute_day_ahead). The hourly and minute levels are unchanged.")
    parser.add_argument("--day-ahead-tolerance", type=float, default=1e-4,
                        help="ADMM stopping tolerance of the day-ahead solve only (RMS kW, primal and dual); "
                             "the hourly and minute solves keep 1e-4.")
    parser.add_argument("--tracking", choices=("trades", "grid"), default="trades",
                        help="What the minute level follows: the hourly plan's P2P trades (default), or its grid "
                             "import with the end-of-hour state at the plan's shadow prices (needs --parallel; "
                             "outputs get the suffix _grid_tracking). See the module docstring.")
    parser.add_argument("--tracking-weight", type=float, default=10.0,
                        help="GBP per kW^2 per hour of departing from the hourly plan's P2P trades (--tracking trades).")
    parser.add_argument("--grid-tracking-weight", type=float, default=10.0,
                        help="GBP per kW^2 per hour of departing from the hourly plan's grid import (--tracking grid).")
    parser.add_argument("--soc-weight", type=float, default=None,
                        help="GBP per kWh^2 of missing the hourly plan's end-of-hour battery charge "
                             "(default 100 with --tracking trades, 1 with --tracking grid).")
    parser.add_argument("--temperature-weight", type=float, default=None,
                        help="GBP per degC^2 of missing the hourly plan's end-of-hour indoor temperature "
                             "(default 10 with --tracking trades, 1 with --tracking grid).")
    parser.add_argument("--error", action="store_true",
                        help="Add forecast error (outputs get the suffix _with_error; see error_inputs).")
    parser.add_argument("--scenario", type=int, default=1,
                        help="Noise scenario of experiment 05 used as the actual hourly load, PV and temperature.")
    parser.add_argument("--minute-load-noise", type=float, default=0.10,
                        help="Std of the independent minute-level load fluctuation (fraction).")
    parser.add_argument("--minute-pv-noise", type=float, default=0.10,
                        help="Std of the independent minute-level PV fluctuation (fraction).")
    parser.add_argument("--error-seed", type=int, default=2026)
    parser.add_argument("--hourly-case", default="base",
                        help="Experiment-05 noise case for the actual hourly values (base, correlated, ...).")
    parser.add_argument("--minute-correlation", type=float, default=0.0,
                        help="Share of the minute fluctuation common to all buildings (0-1).")
    parser.add_argument("--minute-time-constant", type=float, default=0.0,
                        help="Minutes; > 0 makes the minute fluctuation persist (AR(1)) instead of i.i.d.")
    parser.add_argument("--correlated", action="store_true",
                        help="Shortcut: --error --hourly-case correlated --minute-correlation 0.8 "
                             "--minute-time-constant 10.")
    parser.add_argument("--cloud-transients", action="store_true",
                        help="Add passing clouds to the actual minute PV (see cloud_shading); outputs get "
                             "the suffix _clouds.")
    parser.add_argument("--cloud-rate", type=float, default=2.0, help="Clouds per hour, 07:00-19:00.")
    parser.add_argument("--cloud-depth", type=float, nargs=2, default=(0.5, 0.8),
                        help="Range of the PV drop, fraction of the clear-sky value.")
    parser.add_argument("--cloud-duration", type=int, nargs=2, default=(3, 15), help="Range of minutes shaded.")
    parser.add_argument("--cloud-seed", type=int, default=7)
    parser.add_argument("--parallel", action="store_true",
                        help="Solve the building subproblems in parallel worker processes and report "
                             "measured times (05 admm_parallel.ParallelADMM).")
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers (one per building for N = 10).")
    if EXTENSION is not None:
        EXTENSION.add_arguments(parser)
    parser.add_argument("--output-dir", default=None, help="Folder for the outputs (default: the experiment's).")
    parser.add_argument("--plot-only", action="store_true", help="Redraw the figures from the saved CSVs.")
    parser.add_argument("--network-charge", action="store_true",
                        help="Exogenous distance-based network charge on every P2P kWh (Baroche et al. 2019; "
                             "admm_bilateral_p2p_valley.NETWORK_CHARGE). Outputs get _network_charge.")
    parser.add_argument("--network-losses", action="store_true",
                        help="Quadratic per-pair network losses valued at the tariff (Talebi et al. 2025; "
                             "admm_bilateral_p2p_valley.NETWORK_LOSSES). Outputs get _network_losses.")
    parser.add_argument("--network-loss-per-kw", type=float, default=0.02,
                        help="Average fraction lost per kW traded (default 0.02: a 1 kW trade loses 2%%).")
    parser.add_argument("--network-charge-rate", type=float, default=0.01,
                        help="Average network charge, GBP per kWh traded (default 0.01 = 1 p/kWh).")
    args = parser.parse_args()
    OPT.NETWORK_CHARGE, OPT.NETWORK_CHARGE_GBP_PER_KWH = args.network_charge, args.network_charge_rate
    OPT.NETWORK_LOSSES, OPT.NETWORK_LOSS_PER_KW = args.network_losses, args.network_loss_per_kw
    grid_tracking = args.tracking == "grid"
    if grid_tracking and not args.parallel:
        parser.error("--tracking grid needs --parallel (the shadow prices come from the gurobipy workers' duals)")
    if args.soc_weight is None:
        args.soc_weight = 1.0 if grid_tracking else 100.0
    if args.temperature_weight is None:
        args.temperature_weight = 1.0 if grid_tracking else 10.0
    if args.correlated:
        args.error, args.hourly_case = True, "correlated"
        args.minute_correlation, args.minute_time_constant = 0.8, 10.0
    suffix = ""
    if args.error:
        suffix = "_with_error" + ("_correlated" if args.hourly_case != "base" or args.minute_correlation > 0
                                  or args.minute_time_constant > 0 else "")
    if args.cloud_transients:
        if not args.error:
            parser.error("--cloud-transients needs --error (or --correlated)")
        suffix += "_clouds"
    if args.parallel:
        suffix += "_parallel"
    if args.day_ahead_resolution == "minute":
        suffix += "_minute_day_ahead"
    if args.network_charge:
        suffix += "_network_charge"
    if args.network_losses:
        suffix += "_network_losses"
    if grid_tracking:
        suffix += "_grid_tracking"
    default_dir = RESULTS_DIR / "1_three_timescale" if EXTENSION is None else None  # 09 keeps its own folder
    set_outputs(suffix, args.output_dir or default_dir)
    if args.plot_only:
        plot_timing(pd.read_csv(TIMING_FILE))
        plot_dispatch(pd.read_csv(DISPATCH_FILE))
        if EXTENSION is not None:
            EXTENSION.plot(args)
        print(f"Redrew {TIMING_FIGURE.name} and {DISPATCH_FIGURE.name}")
        return
    hours = sorted(set(args.hours))
    if not all(0 <= h < HOURS for h in hours):
        parser.error(f"hours must be in 0..{HOURS - 1}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timing, dispatch = run(hours, args)
    plot_timing(timing)
    plot_dispatch(dispatch)
    if EXTENSION is not None:
        EXTENSION.plot(args)
    print(timing.groupby("level")[["iterations", "parallel_s"]].describe().round(2).T)
    print(f"\n{TIMING_FILE}\n{DISPATCH_FILE}\n{TIMING_FIGURE}\n{DISPATCH_FIGURE}")


if __name__ == "__main__":
    main()
