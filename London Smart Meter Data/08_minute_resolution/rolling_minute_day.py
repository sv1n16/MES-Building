"""Day-ahead plan at minute resolution, then the rest of the day re-planned every minute (N = 10).

PURPOSE
    Can bilateral ADMM plan the whole day at 1-minute resolution, and then re-plan the whole
    remaining day every minute, fast enough to run in real time?

LEVELS  (no hourly level and no plan tracking: every solve is the economic problem itself)
    1. Day-ahead   one ADMM solve of the whole day at minute resolution (1440 steps), from the
                   forecast, stopped at --day-ahead-tolerance (default 1e-3). With --tighten-day-ahead
                   it is then tightened to --tolerance, still before the day starts; without it, the
                   first minute solve has to do that work (~330 iterations, ~460 s at N = 10).
    2. Every minute m = 0..1439: ADMM re-solves minutes m..1439 from the state actually reached,
                   stopped at --tolerance (default 1e-4), and applies minute m. Warm start: the
                   previous solve's trades and dual prices shifted by one minute (minute 0: the
                   day-ahead plan).

PROBLEM PER BUILDING AND SOLVE  (admm_bilateral_p2p_valley subproblem at Δt = 1/60 h)
    min  Σ_t [ π_t g_t Δt + c_gas G_t Δt + α Δt (T_t − T^set_t)² + w_t Δt g_t² ]
         + Σ λ q + ρ/2 Σ (q − z)² + 1e-4 Σ q⁺
    s.t. building model B_b (see common/central_optimisation_showcase.py)

WHAT EACH SOLVE KNOWS  (with --correlated or --error; see three_timescale.error_inputs)
    current minute        measured (actual minute load, PV, outdoor temperature)
    rest of the hour      the hour's actual average (as the hourly level of three_timescale.py knows it)
    later hours           the forecast
    The day-ahead plan uses the forecast throughout.

TIMING
    Measured wall time of every solve with --workers parallel processes (one per building).

Outputs (Results/08_minute_resolution/2_minute_rolling/, suffix _with_error[_correlated] and _first<M>min if --minutes < 1440):
    90_rolling_minute_day_timing<suffix>.csv     one row per solve: level, minute, horizon, iterations, time
    91_rolling_minute_day_dispatch<suffix>.csv   per minute: grid import, battery, P2P, temperature, the day-ahead plan
    92_rolling_minute_day_timing<suffix>.png     time per solve through the day against the 1-minute line, iterations

Examples:
    python rolling_minute_day.py --correlated --minutes 60     # day-ahead + the first hour, to gauge the time
    python rolling_minute_day.py --correlated                  # the whole day: 1 day-ahead + 1440 minute solves
    python rolling_minute_day.py --correlated --plot-only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import project_paths  # noqa: E402

project_paths.register(__file__)  # results -> Results/08_minute_resolution/

import argparse  # noqa: E402
import time  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import three_timescale as TS  # noqa: E402
from three_timescale import C, N, OPT, minute_inputs, minute_window_at  # noqa: E402

DAY = 24 * 60


def rolling_view(forecast, hour_info, actual, now):
    """Inputs of the solve at minute `now` over the rest of the day (see the module docstring)."""
    view = dict(forecast)
    end_of_hour = (now // 60 + 1) * 60
    for name in ("LOAD", "PV_B"):
        view[name] = forecast[name].copy()
        view[name][:, now:end_of_hour] = hour_info[name][:, now:end_of_hour]
        view[name][:, now] = actual[name][:, now]
    view["T_OUT"] = forecast["T_OUT"].copy()
    view["T_OUT"][now:end_of_hour] = hour_info["T_OUT"][now:end_of_hour]
    view["T_OUT"][now] = actual["T_OUT"][now]
    return view


def outputs(args):
    suffix = ""
    if args.error:
        suffix = "_with_error" + ("_correlated" if args.hourly_case != "base" else "")
    if args.tighten_day_ahead:
        suffix += "_tightened"
    if args.network_charge:
        suffix += "_network_charge"
    if args.network_losses:
        suffix += "_network_losses"
    if args.minutes < DAY:
        suffix += f"_first{args.minutes}min"
    folder = Path(args.output_dir) if args.output_dir else TS.RESULTS_DIR / "2_minute_rolling"
    folder.mkdir(parents=True, exist_ok=True)
    return (folder / f"90_rolling_minute_day_timing{suffix}.csv",
            folder / f"91_rolling_minute_day_dispatch{suffix}.csv",
            folder / f"92_rolling_minute_day_timing{suffix}.png")


def run(args, timing_file, dispatch_file):
    data = minute_inputs("linear")  # the forecast at minute resolution
    _, hour_info, actual = TS.error_inputs(args, data)
    solver = TS.ParallelSolver(args.workers)
    timing, dispatch = [], []
    try:
        # 1. day-ahead at minute resolution, from the forecast
        print(f"Day-ahead: {DAY} one-minute steps, tolerance {args.day_ahead_tolerance:g} ({solver.kind}) ...",
              flush=True)
        with minute_window_at(data, 0, DAY, np.asarray(C.INIT_SOC), np.asarray(C.INIT_T_IN)), \
                TS.admm_tolerance(args.day_ahead_tolerance):
            plan, lam, Z, info = solver.solve(None, None, None, args)
        timing.append({"level": "day-ahead", "minute": 0, "horizon_steps": DAY, **info})
        print(f"  {info['iterations']} iterations, {info['parallel_s']:.1f} s", flush=True)
        if args.tighten_day_ahead:
            # still before the day starts: tighten the loose plan to --tolerance on the forecast, so the
            # first minute solve starts from a converged plan (otherwise minute 0 does this work, ~330 it.)
            print(f"Day-ahead tightened to {args.tolerance:g} ...", flush=True)
            with minute_window_at(data, 0, DAY, np.asarray(C.INIT_SOC), np.asarray(C.INIT_T_IN)), \
                    TS.admm_tolerance(args.tolerance):
                plan, lam, Z, info = solver.solve(lam, Z, None, args)
            timing.append({"level": "day-ahead (tightened)", "minute": 0, "horizon_steps": DAY, **info})
            print(f"  {info['iterations']} iterations, {info['parallel_s']:.1f} s", flush=True)

        # 2. the rest of the day re-planned every minute
        soc = np.asarray(C.INIT_SOC, dtype=float).copy()
        indoor = np.asarray(C.INIT_T_IN, dtype=float).copy()
        for now in range(args.minutes):
            length = DAY - now
            if now > 0:  # warm start: the previous solve shifted by one minute
                lam = [l[:, 1:] for l in lam]
                Z = {pair: z[1:] for pair, z in Z.items()}
            view = rolling_view(data, hour_info, actual, now) if args.error else data
            with minute_window_at(view, now, length, soc, indoor), TS.admm_tolerance(args.tolerance):
                values, lam, Z, info = solver.solve(lam, Z, None, args)
            step = TS.first_step(values)
            soc, indoor = step["soc"], step["T_in"]
            timing.append({"level": "minute", "minute": now, "horizon_steps": length, **info})
            dispatch.append({
                "minute_of_day": now, "grid_import_kw": float(step["p_el"].sum()),
                "battery_discharge_minus_charge_kw": float((step["discharge"] - step["charge"]).sum()),
                "p2p_traded_kw": step["p2p"], "battery_energy_kwh": float(step["soc"].sum()),
                "mean_indoor_temperature_c": float(step["T_in"].mean()),
                "price_gbp_per_kwh": float(data["PRICE"][now]),
                "actual_load_kw": float(actual["LOAD"][:, now].sum()),
                "forecast_load_kw": float(data["LOAD"][:, now].sum()),
                "actual_pv_kw": float(actual["PV_B"][:, now].sum()),
                "forecast_pv_kw": float(data["PV_B"][:, now].sum()),
                "day_ahead_grid_import_kw": float(plan["p_el"][:, now].sum()),
                "day_ahead_battery_energy_kwh": float(plan["soc"][:, now].sum()),
            })
            if now % 10 == 0 or now == args.minutes - 1:
                done = pd.DataFrame(timing[1:])
                print(f"minute {now:4d} ({now // 60:02d}:{now % 60:02d}): {info['iterations']:3d} it. "
                      f"{info['parallel_s']:6.2f} s; so far median {done['parallel_s'].median():.2f} s, "
                      f"max {done['parallel_s'].max():.2f} s, total {done['parallel_s'].sum():.0f} s", flush=True)
            if now % 60 == 59 or now == args.minutes - 1:
                pd.DataFrame(timing).to_csv(timing_file, index=False)
                pd.DataFrame(dispatch).to_csv(dispatch_file, index=False)
    finally:
        solver.close()
    return pd.DataFrame(timing)


def plot_timing(timing, figure, kind):
    minute = timing[timing["level"] == "minute"]
    day = timing[timing["level"] == "day-ahead"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2.2, 1]})
    for axis in (left, right):
        TS._style(axis)
    hours = minute["minute"] / 60.0
    left.scatter(hours, minute["parallel_s"], s=5, color=TS.LEVEL_COLOURS["minute"],
                 label="Every-minute re-plan of the rest of the day")
    tightened = timing[timing["level"] == "day-ahead (tightened)"]
    if not day.empty:
        left.scatter([-0.8], day["parallel_s"], s=70, marker="s", color=TS.LEVEL_COLOURS["day-ahead"],
                     label=f"Day-ahead plan, 1440 steps ({int(day['iterations'].iloc[0])} it.)")
    if not tightened.empty:
        left.scatter([-0.4], tightened["parallel_s"], s=70, marker="D", color=TS.LEVEL_COLOURS["hourly"],
                     label=f"Day-ahead tightened before the day ({int(tightened['iterations'].iloc[0])} it.)")
    left.axhline(60.0, color=TS.PLAN_COLOR, lw=1, ls=":")
    left.annotate("1 min", (1.0, 60.0), xycoords=("axes fraction", "data"), xytext=(-2, 2),
                  textcoords="offset points", ha="right", va="bottom", fontsize=7, color=TS.PLAN_COLOR)
    left.set_yscale("log")
    left.set_xlabel("Hour of day", fontsize=9, color=TS.INK)
    left.set_ylabel(f"Time per solve, {kind} (s, log scale)", fontsize=9, color=TS.INK)
    left.set_title("a) Every solve; the horizon shrinks from 1440 to 1 minute through the day",
                   loc="left", fontsize=10, color=TS.INK)
    left.legend(frameon=False, fontsize=7, loc="upper right")
    right.scatter(hours, minute["iterations"], s=5, color=TS.LEVEL_COLOURS["minute"])
    right.set_xlabel("Hour of day", fontsize=9, color=TS.INK)
    right.set_ylabel("ADMM iterations", fontsize=9, color=TS.INK)
    right.set_title(f"b) Iterations per minute solve (median {minute['iterations'].median():.0f}, "
                    f"max {minute['iterations'].max():.0f})", loc="left", fontsize=10, color=TS.INK)
    fig.suptitle(f"Day-ahead at minute resolution, then the rest of the day re-planned every minute "
                 f"(N = {N}, {kind}; median {minute['parallel_s'].median():.2f} s, "
                 f"max {minute['parallel_s'].max():.2f} s per minute)", fontsize=11, color=TS.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    TS._save(fig, figure)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=int, default=DAY, help="Minutes of the day to re-plan (default 1440).")
    parser.add_argument("--day-ahead-tolerance", type=float, default=1e-3)
    parser.add_argument("--tolerance", type=float, default=1e-4, help="ADMM tolerance of the every-minute solves.")
    parser.add_argument("--tighten-day-ahead", action="store_true",
                        help="Before the day starts, tighten the day-ahead plan to --tolerance (warm-started from "
                             "the loose plan), so minute 0 starts from a converged plan. Outputs get _tightened.")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--error", action="store_true", help="Forecast error, as three_timescale.py --error.")
    parser.add_argument("--correlated", action="store_true",
                        help="Correlated forecast error, as three_timescale.py --correlated.")
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--error-seed", type=int, default=2026)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--plot-only", action="store_true")
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
    # forecast-error settings, as three_timescale.py
    args.interpolation, args.cloud_transients = "linear", False
    args.hourly_case, args.minute_load_noise, args.minute_pv_noise = "base", 0.10, 0.10
    args.minute_correlation, args.minute_time_constant = 0.0, 0.0
    if args.correlated:
        args.error, args.hourly_case = True, "correlated"
        args.minute_correlation, args.minute_time_constant = 0.8, 10.0
    if not 1 <= args.minutes <= DAY:
        parser.error(f"--minutes must be in 1..{DAY}")
    timing_file, dispatch_file, figure = outputs(args)
    kind = f"measured, {args.workers} parallel workers"
    if not args.plot_only:
        run(args, timing_file, dispatch_file)
    timing = pd.read_csv(timing_file)
    plot_timing(timing, figure, kind)
    minute = timing[timing["level"] == "minute"]
    print(timing.groupby("level")[["iterations", "parallel_s"]].describe().round(2).T)
    print(f"not converged: {int((~minute['converged'].astype(bool)).sum())} of {len(minute)} minute solves")
    print(f"\n{timing_file}\n{dispatch_file}\n{figure}")


if __name__ == "__main__":
    main()
