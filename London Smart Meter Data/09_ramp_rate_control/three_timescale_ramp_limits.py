"""Three-timescale ADMM with ramp-rate and cumulative limits (6a)-(6c) in the minute-level objective (N = 10).

PURPOSE
    Can the minute-level controller keep each building's power ramps within an operator's
    ramp-rate limit, and what does that cost? The three-timescale model itself is
    08_minute_resolution/three_timescale.py, which has no ramp control; this file adds it as
    a three_timescale.EXTENSION (class RampLimits: options, penalty terms, per-minute records,
    figure 89), sets the options and an output folder, then calls three_timescale.main().

MINUTE-LEVEL OBJECTIVE PER BUILDING  (minute m of hour h, over minutes t = m..59)

    min  own cost  +  ADMM terms (λ q + ρ/2 (q − z)²)                  as in three_timescale.py
       + sharing-plan tracking  W·Δt·Σ_t Σ_j (q_j,t − q_j^plan)²
       + end-of-hour SOC / temperature targets
       + ramp penalty on the excess over the limit ΔE:
           r⁺_t ≥ E_t − E_{t−1} − ΔE,   r⁻_t ≥ E_{t−1} − E_t − ΔE,   r± ≥ 0
           quadratic  (default)        W_r/Δt · Σ_t (r⁺_t² + r⁻_t²)          --ramp-weight
           linear     (--ramp-penalty)  W_lin · Σ_t (r⁺_t + r⁻_t)              --ramp-linear-weight
       + (P2P mode only, consumers) W_cum · Σ_t max(0, Σ_{l≤t} E_l − D_t)²      --cumulative-weight

    E_t = energy per minute: grid import g_t·Δt with --grid; otherwise P2P sales
    (producers) or grid + P2P purchases (consumers), as in (6a)-(6c) below.
    The excess r± is penalised, not forbidden, so the limit is soft.

Runs 08_minute_resolution/three_timescale.py with, by default:
    --ramp-limits   (6a)-(6c) as quadratic penalties in every minute subproblem:
                    (6a) E(n,t) - E(n,t-1) <= dE(n)   producers n (buildings with PV or a battery)
                    (6b) E(n,t-1) - E(n,t) <= dE(n)   producers
                    (6c) sum_{l<=t} E(n,l) <= D(n,t)  consumers n (buildings with neither)
                    E(n,t): energy a producer sells peer to peer in the minute, or a consumer buys
                    (grid + P2P). dE(n) = ramp rate x rated power (PV peak + battery power) x 1 min,
                    ramp rate 10 %/min (the operators' benchmark; --ramp-rate 0.05-0.20).
                    D(n,t): the consumer's purchases in the hourly plan, cumulated over the minutes
                    of the hour (x (1 + --demand-slack)).
    --correlated    forecast error: experiment 05's correlated hourly case, minute fluctuations
                    80% common to all buildings with a 10-minute time constant
    --parallel      building subproblems solved at the same time by 10 worker processes
                    (measured times)
Every run goes to its own folder Results/09_ramp_rate_control/runs/<name>/ (see run_folder), e.g.
plan1_no_limit, plan1_10pct_squared1000, plan1_10pct_linear1000, plan1_2pct_squared1000_clouds,
p2p_plan10_10pct_squared1000. --reference: the same run without the limit. --clouds: passing clouds
in the actual minute PV (three_timescale.cloud_shading). Any option of three_timescale.py can be
added or overridden on the command line.

With --grid the ramp limit applies to every building's grid import only (--ramp-on grid; the P2P
limits (6a)-(6c) are off), rated at its largest possible import, and the hourly-plan deviation
weight is 1 instead of 10 GBP per kW^2 per hour (also for --grid --reference). --ramp-penalty
linear | quadratic | both chooses the proportional or squared penalty on the excess.

Examples:
    python three_timescale_ramp_limits.py                    # the whole day, with the limits
    python three_timescale_ramp_limits.py --reference        # the whole day, without them
    python three_timescale_ramp_limits.py --hours 17
    python three_timescale_ramp_limits.py --ramp-rate 0.02     # tighter limit, 2 %/min
    python three_timescale_ramp_limits.py --clouds             # with passing clouds
    python three_timescale_ramp_limits.py --clouds --reference
    python three_timescale_ramp_limits.py --grid               # limit on grid imports only, lower tracking
    python three_timescale_ramp_limits.py --grid --reference
    python three_timescale_ramp_limits.py --grid --ramp-penalty linear   # exact (proportional) penalty
    python three_timescale_ramp_limits.py --plot-only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import project_paths  # noqa: E402

project_paths.register(__file__)  # results -> Results/09_ramp_rate_control/

import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyomo.environ as pyo  # noqa: E402

import three_timescale  # noqa: E402  (08_minute_resolution is on the path via project_paths)
from three_timescale import C, INK, N, OPT, PLAN_COLOR, _save, _style, plt  # noqa: E402


# ============================================================================
# RAMP LIMITS AS A three_timescale EXTENSION
# ============================================================================
class RampLimits:
    """Ramp-rate and cumulative limits added to the minute level of three_timescale.py.

    Installed as three_timescale.EXTENSION by main(). Writes, next to the 85-88 outputs,
    89_traded_energy_limits<suffix>.csv/.png: per building and minute the energy E, its change
    against the limit ΔE, and (consumers, P2P mode) the cumulative purchases against D.
    Without --ramp-limits (the reference run) nothing is added to the objective but the same
    records are written, so the reference can be measured against the limit.
    """

    def add_arguments(self, parser):
        parser.add_argument("--ramp-limits", action="store_true",
                            help="Add (6a)-(6c) as penalties: producers' P2P sales ramp within --ramp-rate per "
                                 "minute, consumers' cumulative purchases within the hourly plan's allocation.")
        parser.add_argument("--ramp-on", choices=("p2p", "grid"), default="p2p",
                            help="p2p: (6a)-(6c) on producers' P2P sales and consumers' cumulative purchases; "
                                 "grid: a ramp limit on every building's grid import only, rated at its largest "
                                 "possible import (peak load + heat pump + battery charging).")
        parser.add_argument("--ramp-penalty", choices=("quadratic", "linear", "both"), default="quadratic",
                            help="--ramp-on grid: penalty on the ramp excess: quadratic (--ramp-weight x excess^2), "
                                 "linear (--ramp-linear-weight x excess, exact: no excess once the weight exceeds "
                                 "what the excess is worth) or both.")
        parser.add_argument("--ramp-linear-weight", type=float, default=1000.0,
                            help="GBP per kWh of ramp excess (the proportional penalty).")
        parser.add_argument("--ramp-rate", type=float, default=0.10,
                            help="Ramp-rate limit, fraction of rated power (PV peak + battery power) per minute "
                                 "(operators use 0.05-0.20; 0.10 is the benchmark).")
        parser.add_argument("--ramp-weight", type=float, default=100.0,
                            help="GBP per kW^2 per hour of ramping beyond the limit (10x the tracking weight).")
        parser.add_argument("--cumulative-weight", type=float, default=100.0,
                            help="GBP per kWh^2 of cumulative purchases beyond the allocation D.")
        parser.add_argument("--demand-slack", type=float, default=0.0,
                            help="D = hourly plan's purchases x (1 + slack), cumulated over the minutes.")

    def set_outputs(self, folder, suffix):
        self.file = folder / f"89_traded_energy_limits{suffix}.csv"
        self.figure = folder / f"89_traded_energy_limits{suffix}.png"

    # ---- per day / hour / minute bookkeeping --------------------------------------------------
    def start_day(self, args):
        # (6a)-(6c): producers' P2P sales may change by at most delta per minute; consumers' cumulative
        # purchases (grid + P2P) within the hour may not exceed the hourly plan's allocation (D)
        self.grid_mode = args.ramp_on == "grid"
        self.producer = (np.asarray(C.PV_B).max(axis=1) > 1e-9) | np.asarray(C.HAS_BATTERY, bool)
        rated_kw = np.asarray(C.PV_B).max(axis=1) + np.asarray(C.MAXPOW, float)
        if self.grid_mode:
            # --ramp-on grid: every building's grid import, rated at its largest possible import
            # (peak load + heat-pump electrical rating at the lowest COP + battery charging power)
            rated_kw = (np.asarray(C.LOAD, float).max(axis=1) + C.p_th_nom / float(np.min(OPT.COP))
                        + np.asarray(C.MAXPOW, float))
        self.delta = args.ramp_rate * rated_kw / 60.0  # ΔE, kWh per one-minute step
        self.previous_energy = None
        self.rows = []

    def start_hour(self, hourly, args):
        planned = np.where(self.producer, hourly["sold"], hourly["p_el"] + hourly["bought"]) / 60.0  # kWh/min
        if self.grid_mode:
            planned = hourly["p_el"] / 60.0
        self.cap_per_minute = planned * (1.0 + args.demand_slack)  # D per minute
        self.bought_so_far = np.zeros(N)
        if self.previous_energy is None:
            self.previous_energy = planned.copy()

    def minute_plan(self, hourly, minute, args):
        if not args.ramp_limits:
            return hourly
        if self.grid_mode:
            ramp = {"mode": "grid", "delta": self.delta, "previous": self.previous_energy,
                    "ramp_weight": args.ramp_weight,
                    "quadratic": args.ramp_penalty in ("quadratic", "both"),
                    "linear_weight": args.ramp_linear_weight if args.ramp_penalty in ("linear", "both") else 0.0}
        else:
            ramp = {"producer": self.producer, "delta": self.delta, "previous": self.previous_energy,
                    "cap": [np.arange(minute + 1, 61) * self.cap_per_minute[b] - self.bought_so_far[b]
                            for b in range(N)],
                    "ramp_weight": args.ramp_weight, "cumulative_weight": args.cumulative_weight}
        # "ramp" is read by pyomo_terms (sequential) and, via worker_tracking, by admm_parallel._build
        return {**hourly, "ramp": ramp, "worker_tracking": {"ramp": ramp}}

    def end_minute(self, step, hour, minute, args):
        energy = np.where(self.producer, step["sold"], step["p_el"] + step["bought"]) / 60.0  # applied, kWh
        self.bought_so_far += np.where(self.producer, 0.0, energy)
        if self.grid_mode:
            energy = step["p_el"] / 60.0
        now = hour * 60 + minute
        for b in range(N):
            role = "grid" if self.grid_mode else ("producer" if self.producer[b] else "consumer")
            self.rows.append({
                "hour": hour, "minute": minute, "minute_of_day": now, "building": str(C.BUILDING_IDS[b]),
                "role": role, "energy_kwh": energy[b],
                "change_kwh": energy[b] - self.previous_energy[b], "ramp_limit_kwh": self.delta[b],
                "cumulative_kwh": self.bought_so_far[b] if role == "consumer" else np.nan,
                "cumulative_cap_kwh": (minute + 1) * self.cap_per_minute[b] if role == "consumer" else np.nan,
            })
        self.previous_energy = energy

    def save(self):
        pd.DataFrame(self.rows).to_csv(self.file, index=False)

    # ---- objective terms (sequential Pyomo solver; admm_parallel._build has the gurobipy twin) ----
    @staticmethod
    def pyomo_terms(m, b, plan, dt):
        """Soft limits: r⁺_t ≥ E_t − E_{t−1} − ΔE, r⁻_t ≥ E_{t−1} − E_t − ΔE, penalised in the objective."""
        ramp = plan.get("ramp")
        if ramp is None:
            return 0.0
        extra = 0.0
        if ramp.get("mode") == "grid":  # every building's grid import: E_t = g_t Δt
            energy = {t: dt * m.p_el[t] for t in m.t}
            m.ramp_up = pyo.Var(m.t, domain=pyo.NonNegativeReals)
            m.ramp_down = pyo.Var(m.t, domain=pyo.NonNegativeReals)
            previous, delta = float(ramp["previous"][b]), float(ramp["delta"][b])
            m.c_ramp_up = pyo.Constraint(m.t, rule=lambda mm, t: mm.ramp_up[t] >= energy[t] - (
                energy[t - 1] if t > 0 else previous) - delta)
            m.c_ramp_down = pyo.Constraint(m.t, rule=lambda mm, t: mm.ramp_down[t] >= (
                energy[t - 1] if t > 0 else previous) - energy[t] - delta)
            if ramp.get("quadratic", True):
                extra += ramp["ramp_weight"] / dt * sum(m.ramp_up[t] ** 2 + m.ramp_down[t] ** 2 for t in m.t)
            if ramp.get("linear_weight", 0.0) > 0:  # proportional (exact) penalty, GBP per kWh of excess
                extra += ramp["linear_weight"] * sum(m.ramp_up[t] + m.ramp_down[t] for t in m.t)
        elif ramp["producer"][b]:  # (6a)-(6b): E_t = P2P sales Δt
            energy = {t: dt * sum(m.qpos[k, t] for k in m.k) for t in m.t}
            m.ramp_up = pyo.Var(m.t, domain=pyo.NonNegativeReals)
            m.ramp_down = pyo.Var(m.t, domain=pyo.NonNegativeReals)
            previous, delta = float(ramp["previous"][b]), float(ramp["delta"][b])
            m.c_ramp_up = pyo.Constraint(m.t, rule=lambda mm, t: mm.ramp_up[t] >= energy[t] - (
                energy[t - 1] if t > 0 else previous) - delta)
            m.c_ramp_down = pyo.Constraint(m.t, rule=lambda mm, t: mm.ramp_down[t] >= (
                energy[t - 1] if t > 0 else previous) - energy[t] - delta)
            extra += ramp["ramp_weight"] / dt * sum(m.ramp_up[t] ** 2 + m.ramp_down[t] ** 2 for t in m.t)
        else:  # (6c) consumers: E_t = (grid + P2P purchases) Δt, cumulative within D
            energy = {t: dt * (m.p_el[t] + sum(m.qneg[k, t] for k in m.k)) for t in m.t}
            m.cumulative_excess = pyo.Var(m.t, domain=pyo.NonNegativeReals)
            m.c_cumulative = pyo.Constraint(m.t, rule=lambda mm, t: mm.cumulative_excess[t] >= sum(
                energy[l] for l in range(t + 1)) - float(ramp["cap"][b][t]))
            extra += ramp["cumulative_weight"] * sum(m.cumulative_excess[t] ** 2 for t in m.t)
        return extra

    # ---- figure ------------------------------------------------------------------------------
    def plot(self, args):
        if self.file.exists():
            plot_ramp(pd.read_csv(self.file), args, self.figure)


def plot_ramp(ramp, args, figure):
    """Ramp limits: a) each limited building's minute ramp (kW/min) against its own limit;
    b) consumers' cumulative purchases against D (6c), or with --ramp-on grid the community's grid import."""
    grid_mode = bool((ramp["role"] == "grid").any())
    limited = ramp[ramp["role"] != "consumer"].copy()
    consumers = ramp[ramp["role"] == "consumer"].copy()
    limited["ratio"] = limited["change_kwh"].abs() / limited["ramp_limit_kwh"]
    limited["ramp_kw_per_min"] = 60 * limited["change_kwh"]  # change in power, kW per minute
    share = float((limited["ratio"] > 1.0 + 1e-6).mean())
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    for axis in (top, bottom):
        _style(axis)
    for index, (building, rows) in enumerate(limited.groupby("building")):
        colour = f"C{index}"
        top.plot(rows["minute_of_day"] / 60.0, rows["ramp_kw_per_min"], color=colour, lw=0.8, label=building)
        limit = 60 * rows["ramp_limit_kwh"].iloc[0]
        for sign in (1, -1):
            top.axhline(sign * limit, color=colour, lw=0.7, ls="--", alpha=0.7)
    top.axhline(0, color=PLAN_COLOR, lw=0.5)
    what = "grid import" if grid_mode else "P2P sales"
    who = "buildings' grid-import" if grid_mode else "(6a)-(6b) producers'"
    top.set_ylabel(f"Change in {what}\npower (kW/min)", fontsize=8, color=INK)
    top.set_title(f"a) {who} ramp rate against each one's limit (dashed, {100 * args.ramp_rate:g}% of rated "
                  f"power per minute): {100 * share:.1f}% of building-minutes over", loc="left", fontsize=10, color=INK)
    top.legend(frameon=False, fontsize=7, ncol=5, loc="upper left")
    if grid_mode:
        community = limited.groupby("minute_of_day")["energy_kwh"].sum() * 60
        bottom.plot(community.index / 60.0, community.values, lw=1, color=PLAN_COLOR)
        steps = community.diff().abs()
        bottom.set_ylabel("Community grid\nimport (kW)", fontsize=8, color=INK)
        bottom.set_title(f"b) Community grid import; largest change {steps.max():.2f} kW/min", loc="left",
                         fontsize=10, color=INK)
    else:
        for building, rows in consumers.groupby("building"):
            bottom.plot(rows["minute_of_day"] / 60.0, rows["cumulative_kwh"], lw=1.2, label=f"{building}: purchases")
            bottom.plot(rows["minute_of_day"] / 60.0, rows["cumulative_cap_kwh"], lw=1, ls=":", color=PLAN_COLOR)
        excess = (consumers["cumulative_kwh"] - consumers["cumulative_cap_kwh"]).clip(lower=0)
        bottom.set_ylabel("Cumulative purchases\nwithin the hour (kWh)", fontsize=8, color=INK)
        bottom.set_title(f"b) (6c) consumers' cumulative purchases (grid + P2P) against the allocation D (dotted); "
                         f"largest excess {excess.max():.3f} kWh", loc="left", fontsize=10, color=INK)
        bottom.legend(frameon=False, fontsize=8, loc="upper left")
    bottom.set_xlabel("Time of day", fontsize=9, color=INK)
    bottom.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x) % 24:02d}:{int(round((x % 1) * 60)) % 60:02d}"))
    title = ("Grid-import ramp limit" if grid_mode else "Traded energy against the ramp and cumulative limits (6a)-(6c)")
    fig.suptitle(f"{title}, N = {N}" + ("" if args.ramp_limits else " — limits NOT in the objective (reference)"),
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, figure)
    plt.close(fig)


OUTPUT_DIR = project_paths.RESULTS_DIR / "09_ramp_rate_control"
DEFAULT_RAMP_WEIGHT = 1000.0  # GBP per kW^2 per hour: 100x the plan-tracking weight
GRID_TRACKING_WEIGHT = 1.0  # --grid: plan-deviation weight, GBP per kW^2 per hour (10 in three_timescale)


def _value(given, flag, default):
    return float(given[given.index(flag) + 1]) if flag in given else default


RUNS_DIR = OUTPUT_DIR / "runs"


def run_folder(reference, clouds, rate, weight, grid=True, plan_weight=None, penalty="quadratic",
               linear_weight=1000.0):
    """Results/09_ramp_rate_control/runs/<name>/ for one full-day run.

    name = [p2p_]plan<plan weight>_<limit>[_clouds], with <limit> = no_limit, or
    <rate>pct_squared<w>, <rate>pct_linear<w> or <rate>pct_squared<w>_linear<w> for the penalty used.
    p2p_ marks the limits (6a)-(6c) on P2P trades (no --grid); grid-import runs have no prefix.
    """
    plan = (GRID_TRACKING_WEIGHT if grid else 10.0) if plan_weight is None else plan_weight
    if reference:
        limit = "no_limit"
    elif penalty == "linear":
        limit = f"{100 * rate:g}pct_linear{linear_weight:g}"
    elif penalty == "both":
        limit = f"{100 * rate:g}pct_squared{weight:g}_linear{linear_weight:g}"
    else:
        limit = f"{100 * rate:g}pct_squared{weight:g}"
    name = f"{'' if grid else 'p2p_'}plan{plan:g}_{limit}{'_clouds' if clouds else ''}"
    return RUNS_DIR / name


def main():
    """Translate this script's shorthand flags into three_timescale.py options, then run it.

    Flags set here only when the user has not given them:
      always             --correlated --parallel
      --clouds           --cloud-transients
      --grid             --ramp-on grid, --tracking-weight GRID_TRACKING_WEIGHT
      not --reference    --ramp-limits, --ramp-weight DEFAULT_RAMP_WEIGHT
      always             --output-dir runs/<run_folder(...)>
    """
    given = sys.argv[1:]
    reference = "--reference" in given
    clouds = "--clouds" in given
    grid = "--grid" in given
    given = [arg for arg in given if arg not in ("--reference", "--clouds", "--grid")]
    defaults = [flag for flag in ("--correlated", "--parallel") if flag not in given]
    if clouds and "--cloud-transients" not in given:
        defaults.append("--cloud-transients")
    if grid:
        defaults += ["--ramp-on", "grid"]
        if "--tracking-weight" not in given:  # the reference gets the same, lower plan-deviation weight
            defaults += ["--tracking-weight", f"{GRID_TRACKING_WEIGHT:g}"]
    weight = _value(given, "--ramp-weight", DEFAULT_RAMP_WEIGHT)
    rate = _value(given, "--ramp-rate", 0.10)
    if "--ramp-weight" not in given and not reference:
        defaults += ["--ramp-weight", f"{weight:g}"]
    if not reference and "--ramp-limits" not in given:
        defaults.append("--ramp-limits")
    if "--output-dir" not in given:
        plan_weight = _value(given, "--tracking-weight", GRID_TRACKING_WEIGHT if grid else 10.0)
        penalty = given[given.index("--ramp-penalty") + 1] if "--ramp-penalty" in given else "quadratic"
        linear_weight = _value(given, "--ramp-linear-weight", 1000.0)
        defaults += ["--output-dir", str(run_folder(reference, clouds, rate, weight, grid, plan_weight,
                                                    penalty, linear_weight))]
    sys.argv = [sys.argv[0], *defaults, *given]
    three_timescale.EXTENSION = RampLimits()  # adds the ramp options, penalties, records and figure 89
    three_timescale.main()


if __name__ == "__main__":
    main()
