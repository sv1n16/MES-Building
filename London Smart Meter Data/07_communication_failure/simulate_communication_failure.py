"""Robustness of the real-time methods to communication failures.

PURPOSE
    How much does each real-time method lose when messages stop arriving, and is
    peer-to-peer ADMM more robust than a central coordinator?
    The objectives are unchanged from 05_realtime_methods (rolling ADMM, local
    re-optimisation, rolling central). A lost connection {b,j} in hour h only adds
        q_bj,h = 0                     (no trade agreed over it this hour)
    and freezes z_bj, λ_bj for later hours at their last agreed values.
SCORED
    cost increase = cost(level) − cost(level 0) on the same noise scenario.

The real-time comparison of 05_realtime_methods is re-run while communication fails.
Every method sees the same failure pattern and the same noisy load / PV /
outdoor-temperature scenarios as experiment 05, so the methods are compared pairwise.
Failure level 0 is the no-failure reference and reproduces the 05 results exactly.

Tests (--test, or --test all; each writes to its own subfolder)
---------------------------------------------------------------
building                  k random buildings offline every hour (1-hour outages)
building_long_outage      as building, but each outage lasts 4 hours
building_late_detection   as building, but the failure is only noticed mid-negotiation
building_correlated       as building, on the 05 case "correlated" (80% of the error common)
link                      every communication link fails independently with probability p
                          per hour: the N(N-1)/2 peer-to-peer links of ADMM and local
                          re-optimisation, the N building-to-coordinator links of central
coordinator               the central coordinator is down for h consecutive hours

Communication topology
----------------------
Local re-optimisation and ADMM exchange messages peer to peer, so a failure removes
pairs of buildings: a building failure removes all its pairs, a link failure one pair.
Rolling central is a star: a building talks only to the coordinator, so a building or
link failure cuts that building off, and a coordinator failure cuts off every building.
Open loop and the fixed schedule use no real-time messages (the plan and its trades were
exchanged day-ahead and the recourse uses local measurements only): they are unaffected.

What each method does with a lost connection ("expect reconnection")
--------------------------------------------------------------------
No P2P trade can be agreed over a lost connection in that hour, so its trade is zero in
the current hour. Nobody knows how long an outage lasts, so for the later hours of the
horizon everyone keeps planning with the last agreed trades and prices over that
connection, as if it will be back next hour. (Assuming instead that the connection is
lost for the rest of the day made every cut-off building stop charging its battery for
its neighbours, which cost about GBP 1/day with one building offline per hour; see
Results/07_communication_failure/_superseded_permanent_failure_model.)

rolling_admm          The connected buildings run bilateral ADMM over the remaining pairs
                      and re-route the current hour's sharing. Over a lost pair, both ends
                      keep the last agreed trade z and their last dual price lambda for the
                      later hours (no z or lambda update, as in asynchronous ADMM). A cut-off
                      building therefore re-optimises with its own controller on stale
                      prices and agreements.
local_reoptimisation  Each building re-optimises on its own; the current hour's trades over
                      lost connections are cancelled and nothing is re-routed. Later hours
                      keep their planned trade limits, as without failures.
rolling_central       The coordinator plans every building over the remaining horizon. A
                      cut-off building has no optimiser and receives no set-points: it
                      executes the last central plan it received (the day-ahead plan if
                      never reached) with the fixed-schedule recourse (battery clipped to
                      its actual SOC, grid takes the balance, surplus tops up the battery,
                      then curtailed). The coordinator knows those set-points, so it fixes
                      the building's current hour to them, starts it from the state its last
                      plan predicted (it has no measurement), and plans its later hours.
rolling_central_local_fallback
                      As rolling_central, but a cut-off building re-optimises its remaining
                      horizon itself, the way local re-optimisation does, with the trades of
                      its last central plan as the limits for the later hours. Separates the
                      value of local intelligence from that of decentralised coordination.
rolling_admm_fallback As rolling_admm, but with an iteration budget (--admm-fallback-iters,
                      default 20: about the time of one central solve at N = 10). If the
                      buildings have not agreed within it, every building falls back to local
                      re-optimisation with the trades of the last agreement everyone reached as
                      limits (lost connections: zero this hour), and trades are settled at
                      min(offer, acceptance). The next hour warm-starts from where the capped
                      ADMM stopped, so its progress is not lost.

Detection (--detection, set by the test)
    start  Failures are known at the start of the hour (missed heartbeat): every solve
           of that hour fixes the current-hour trades over lost connections to zero.
    late   Failures are noticed only after the plan is under way. ADMM: the lost pairs
           take part in the first --drop-iteration iterations, then their messages stop:
           their current-hour trade is fixed to zero, their later hours keep the z and
           lambda of that moment, and the others carry on iterating. Local: every
           building solves assuming its planned trades; lost trades are not delivered,
           so the buyer imports the shortfall and the seller curtails. Central: the
           coordinator plans with every building; the cut-off building does not receive
           its set-points (falls back as above) and its partners' trades with it are not
           delivered, as for local.

Outputs, per test in Results/07_communication_failure/<test>/:
    70_comm_failure_masks.npz          the failure patterns of every level, plus settings
    71_comm_failure_results.csv        one row per level/run/method/interval/building
    72_comm_failure_solve_times.csv    timing rows as in 05
    73_comm_failure_summary.csv        per level, run and method, with the change against no failure
    74_comm_failure_summary_mean.csv   mean and std over runs
    75_comm_failure_robustness.png     cost, cost increase, P2P delivered and peak import against the level
and, across tests, in Results/07_communication_failure/:
    76_comm_failure_overview.png       cost increase against the level for every test
    77_comm_failure_overview.csv       its data

The day-ahead plan and noise scenarios are read from Results/05_realtime_methods
(--plan-dir to change; cases from its cases/<case>/ folder). Nothing is written there.

Run time: rolling ADMM needs many more iterations while connections are lost (tens to
a couple of hundred per hour instead of 2), about 15 minutes per run and level at N = 10.

Examples:
    python simulate_communication_failure.py --test building --runs 3
    python simulate_communication_failure.py --test all --runs 3
    python simulate_communication_failure.py --test link coordinator --runs 1
    python simulate_communication_failure.py --test building --levels 0 1 --runs 1
    python simulate_communication_failure.py --summarise-only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import project_paths  # noqa: E402  (shared modules on sys.path; results -> Results/<experiment>/)

project_paths.register(__file__)

import argparse
import itertools
import os
import time
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyomo.environ as pyo  # noqa: E402

import compare_realtime_methods as compare  # noqa: E402
import simulate_noisy_realtime as baseline  # noqa: E402
import simulate_open_loop as open_loop  # noqa: E402
import simulate_realtime_central as central  # noqa: E402
import simulate_realtime_reoptimise as reopt  # noqa: E402
from plot_realtime_comparison import COLORS, INK, LABELS, PLAN_COLOR, _save, _style  # noqa: E402

OPT, C = reopt.OPT, reopt.C
N, FULL_T, DT = reopt.N, reopt.FULL_T, reopt.DT

RESULTS_ROOT = C.PLOTS_DIR  # Results/07_communication_failure/[<SHOWCASE_PLOTS_SUBDIR>]
OVERVIEW_FIGURE = RESULTS_ROOT / "76_comm_failure_overview.png"
OVERVIEW_FILE = RESULTS_ROOT / "77_comm_failure_overview.csv"

CENTRAL_LOCAL = "rolling_central_local_fallback"
CENTRAL_NONE = "rolling_central_no_fallback"
ADMM_FALLBACK = "rolling_admm_fallback"
ALL_METHODS = ("open_loop", "fixed_schedule", "local_reoptimisation", "rolling_admm", ADMM_FALLBACK, CENTRAL_NONE,
               "rolling_central", CENTRAL_LOCAL)
P2P_METHODS = ("local_reoptimisation", "rolling_admm", ADMM_FALLBACK)
CENTRAL_METHODS = (CENTRAL_NONE, "rolling_central", CENTRAL_LOCAL)
COLORS = {**COLORS, CENTRAL_LOCAL: COLORS["rolling_central"], CENTRAL_NONE: "#4b3b80", ADMM_FALLBACK: "#0d7a52"}
LABELS = {
    **LABELS,
    ADMM_FALLBACK: "Rolling ADMM, fallback: no agreement within the iteration budget -> local re-optimisation "
                   "on the last agreed trades",
    CENTRAL_NONE: "Rolling central, no fallback (cut-off building runs uncontrolled)",
    "rolling_central": "Rolling central, fallback: cut-off building follows its last plan",
    CENTRAL_LOCAL: "Rolling central, fallback: cut-off building re-optimises locally",
}
MARKERS = {"open_loop": "s", "fixed_schedule": "D", "local_reoptimisation": "^", "rolling_admm": "o",
           ADMM_FALLBACK: "h", CENTRAL_NONE: "X", "rolling_central": "v", CENTRAL_LOCAL: "P"}
LINESTYLES = {CENTRAL_LOCAL: "--", "open_loop": ":", CENTRAL_NONE: "-.", ADMM_FALLBACK: "--"}

DEFAULTS = {"outage_hours": 1, "target": "random", "detection": "start", "case": "base"}
TESTS = {  # run in this order
    "coordinator": {"mode": "coordinator", "levels": (0, 1, 3, 6, 12, 24),
                    "description": "central coordinator down for h consecutive hours"},
    "building": {"mode": "building", "levels": (0, 1, 2, 3),
                 "description": "k random buildings offline every hour"},
    "building_long_outage": {"mode": "building", "levels": (0, 1, 2), "outage_hours": 4,
                             "description": "k buildings offline every hour, 4-hour outages"},
    "building_late_detection": {"mode": "building", "levels": (0, 1, 2, 3), "detection": "late",
                                "description": "k buildings offline every hour, noticed mid-negotiation"},
    "building_correlated": {"mode": "building", "levels": (0, 1, 2, 3), "case": "correlated",
                            "description": "k buildings offline every hour, forecast error 80% common to all buildings"},
    "link": {"mode": "link", "levels": (0, 0.05, 0.1, 0.2),
             "description": "every communication link fails with probability p per hour"},
}
LEVEL_SYMBOLS = {"building": "k", "link": "p", "coordinator": "h"}
LEVEL_LABELS = {"building": "Buildings offline every hour", "link": "Link failure probability per hour",
                "coordinator": "Hours the coordinator is down"}

_PLAN_NAMES = ("charge", "discharge", "p_hp", "q_boiler", "gas")


class Outputs:
    def __init__(self, folder: Path):
        self.dir = folder
        self.masks = folder / "70_comm_failure_masks.npz"
        self.results = folder / "71_comm_failure_results.csv"
        self.timings = folder / "72_comm_failure_solve_times.csv"
        self.summary = folder / "73_comm_failure_summary.csv"
        self.mean = folder / "74_comm_failure_summary_mean.csv"
        self.figure = folder / "75_comm_failure_robustness.png"


def default_plan_dir() -> Path:
    path = project_paths.RESULTS_DIR / "05_realtime_methods"
    sub = os.environ.get("SHOWCASE_PLOTS_SUBDIR", "").strip()
    return path / sub if sub else path


# ============================================================================
# FAILURE PATTERNS
# ============================================================================
def pairs_from_offline(offline):
    """Lost peer-to-peer connections (..., N, N, T) of buildings offline (..., N, T)."""
    lost = offline[..., :, None, :] | offline[..., None, :, :]
    idx = np.arange(N)
    lost[..., idx, idx, :] = False
    return lost


def _lasting(starts, hours):
    """A failure starting at hour t lasts `hours` hours (along the last axis)."""
    out = starts.copy()
    for lag in range(1, hours):
        out[..., lag:] |= starts[..., :-lag]
    return out


def draw_patterns(test, levels, runs, seed):
    """level -> (lost_p2p (runs, N, N, T), central_offline (runs, N, T)).

    building     Each hour, buildings are taken in a random order until at least k are
                 offline; an outage
                 lasts outage_hours. The orders are shared by all levels, so with 1-hour
                 outages the offline set at level k is contained in the set at level k + 1.
    link         Every link starts failing with probability p per hour, from the same
                 uniform draws for every p (so the failures are nested in p).
    coordinator  The coordinator is down for h consecutive hours; one random position per run
                 places the outage of every length (h = 24: the whole day).
    """
    rng = np.random.default_rng(seed)
    hours = test["outage_hours"]
    patterns = {}
    if test["mode"] == "building":
        orders = np.argsort(-rng.random((runs, FULL_T, N)), axis=-1)  # same draws as before
        for level in levels:
            offline = np.zeros((runs, N, FULL_T), dtype=bool)
            for run in range(runs):
                hours_left = np.zeros(N, dtype=int)
                for t in range(FULL_T):
                    for b in orders[run, t]:
                        if (hours_left > 0).sum() >= level:
                            break
                        if hours_left[b] == 0:
                            hours_left[b] = hours
                    offline[run, :, t] = hours_left > 0
                    hours_left = np.maximum(hours_left - 1, 0)
            patterns[level] = (pairs_from_offline(offline), offline)
    elif test["mode"] == "link":
        u_pairs = rng.random((runs, N, N, FULL_T))
        u_pairs = np.triu(np.moveaxis(u_pairs, -1, 1), k=1)  # (runs, T, N, N), upper triangle
        u_pairs = np.moveaxis(u_pairs + np.swapaxes(u_pairs, -1, -2), 1, -1)
        u_star = rng.random((runs, N, FULL_T))
        idx = np.arange(N)
        for p in levels:
            lost = _lasting(u_pairs < p, hours)
            lost[:, idx, idx, :] = False
            patterns[p] = (lost, _lasting(u_star < p, hours))
    elif test["mode"] == "coordinator":
        position = rng.random(runs)  # where in the day the outage falls, shared by every length h
        for h in levels:
            down = np.zeros((runs, N, FULL_T), dtype=bool)
            for run in range(runs):
                start = int(position[run] * (FULL_T - int(h) + 1))
                down[run, :, start:start + int(h)] = True
            patterns[h] = (np.zeros((runs, N, N, FULL_T), dtype=bool), down)
    else:
        raise ValueError(f"Unknown failure mode {test['mode']!r}")
    return patterns


# ============================================================================
# HOOKS INTO THE 05 SIMULATIONS
# ============================================================================
def _fix_pairs(m, b, lost_now, times=(0,)):
    """Fix b's trades over lost connections to zero at the given horizon steps (0 = current hour)."""
    for k, neighbour in enumerate(OPT.NB[b]):
        if lost_now[b, neighbour]:
            for t in times:
                m.q[k, t].fix(0.0)
                m.qpos[k, t].fix(0.0)
                m.qneg[k, t].fix(0.0)


@contextmanager
def failure_hooks(lost):
    """Wrap build_subproblem: no trade over a lost connection in the current hour.

    lost is (N, N, T). The interval being solved is read from the optimiser's horizon
    (reopt.remaining_horizon sets OPT.T = T - interval), so the 05 local re-optimisation
    runs unchanged on top of this. Later hours keep their planned trade limits.
    """
    original = OPT.build_subproblem

    def build_subproblem(b):
        m = original(b)
        _fix_pairs(m, b, lost[:, :, FULL_T - OPT.T])
        return m

    OPT.build_subproblem = build_subproblem
    try:
        yield
    finally:
        OPT.build_subproblem = original


def run_admm_with_failures(lost_now, drop_iteration, init_lam=None, init_Z=None, solve_log=None):
    """OPT.run_admm with some peer-to-peer connections lost. Inside remaining_horizon.

    Up to `drop_iteration` iterations every pair exchanges messages as usual (0 for
    failures known at the start of the hour). From then on, over a lost pair: the
    current-hour trade is fixed to zero at both ends; for the later hours both ends keep
    the agreed trade z of that moment and their own dual lambda (no z or lambda update,
    as in asynchronous ADMM), so each end still plans to trade what was last agreed.
    Residuals and the stopping test are over the connected pairs only. Same settings,
    subproblems and update order as OPT.run_admm, which is used when nothing is lost.
    """
    T, NB, PAIRS = OPT.T, OPT.NB, OPT.PAIRS
    lost_pairs = [pair for pair in PAIRS if lost_now[pair]]
    if not lost_pairs:
        return OPT.run_admm(init_lam=init_lam, init_Z=init_Z, solve_log=solve_log, verbose=False)
    C._require_solver()
    solver = pyo.SolverFactory(C.SOLVER)
    options = [("LogFile", ""), ("OutputFlag", 0)]
    if C.SOLVE_TIME_LIMIT > 0:
        options.append(("TimeLimit", C.SOLVE_TIME_LIMIT))
    for option, value in options:
        try:
            solver.options[option] = value
        except Exception:
            pass

    subs = [OPT.build_subproblem(b) for b in range(N)]
    q_all = [np.zeros((N - 1, T)) for _ in range(N)]
    lam_all = ([np.array(lam, dtype=float).copy() for lam in init_lam] if init_lam is not None
               else [np.zeros((N - 1, T)) for _ in range(N)])
    Z = ({p: np.array(init_Z[p], dtype=float).copy() for p in PAIRS} if init_Z is not None
         else {p: np.zeros(T) for p in PAIRS})
    zloc_all = OPT._zloc_from_Z(Z)
    rho = float(OPT.RHO)
    frozen = {}
    copies = []  # (b, k) of every lost pair, both ends
    for lo, hi in lost_pairs:
        copies += [(lo, NB[lo].index(hi)), (hi, NB[hi].index(lo))]
    active = [np.ones(N - 1, dtype=bool) for _ in range(N)]
    streak, hist = 0, []
    for i in range(OPT.MAX_ITERS):
        if i == drop_iteration:  # messages over the lost pairs stop from this iteration
            for pair in lost_pairs:
                frozen[pair] = Z[pair].copy()
                frozen[pair][0] = 0.0
                Z[pair] = frozen[pair].copy()
            for b, k in copies:
                active[b][k] = False
                for var in (subs[b].q, subs[b].qpos, subs[b].qneg):
                    var[k, 0].fix(0.0)
            zloc_all = OPT._zloc_from_Z(Z)

        for b, m in enumerate(subs):  # (32) x-update
            m.rho = rho
            for k in range(N - 1):
                for t in range(T):
                    m.lam[k, t] = float(lam_all[b][k, t])
                    m.zloc[k, t] = float(zloc_all[b][k, t])
            started = time.perf_counter()
            result = solver.solve(m, tee=False)
            if solve_log is not None:
                solve_log.append((i, b, time.perf_counter() - started))
            termination = result.solver.termination_condition
            if termination not in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.locallyOptimal):
                raise RuntimeError(f"building {b} subproblem did not solve to optimality at ADMM iter {i}: "
                                   f"status={result.solver.status}, termination={termination}")
            q_all[b] = OPT._q(m)

        Z_new = OPT.z_update(q_all)  # (33), then the lost pairs keep their frozen agreement
        Z_new.update({pair: values.copy() for pair, values in frozen.items()})
        zloc_new = OPT._zloc_from_Z(Z_new)
        lam_new = [lam_all[b] + rho * (q_all[b] - zloc_new[b]) for b in range(N)]  # (34)
        for b, k in (copies if frozen else []):
            lam_new[b][k] = lam_all[b][k]

        connected = [p for p in PAIRS if p not in frozen]
        n_copies = sum(int(a.sum()) for a in active) * T
        n_pairs = len(connected) * T
        prim = sum(float(np.sum((rho * (q_all[b] - zloc_new[b]))[active[b]] ** 2)) for b in range(N))
        dual = rho * sum(float(np.sum((Z_new[p] - Z[p]) ** 2)) for p in PAIRS)
        r_rms = float(np.sqrt(prim / rho**2 / n_copies)) if n_copies else 0.0
        s_rms = float(np.sqrt(dual / rho / n_pairs)) if n_pairs else 0.0
        hist.append((i, prim, dual, r_rms, s_rms))
        eps1 = (rho * OPT.EPS_PRIMAL) ** 2 * n_copies
        eps2 = rho * OPT.EPS_DUAL**2 * n_pairs
        streak = streak + 1 if (prim <= eps1 and dual <= eps2) else 0
        Z, zloc_all, lam_all = Z_new, zloc_new, lam_new
        if streak >= OPT.STOP_STREAK and i >= drop_iteration:
            break
    return subs, q_all, lam_all, Z, np.array(hist)


@contextmanager
def admm_failures(lost, drop_iteration):
    """Make reopt.simulate_rolling_admm run run_admm_with_failures for each interval."""
    original = OPT.run_admm

    def run_admm(init_lam=None, init_Z=None, solve_log=None, verbose=True):
        lost_now = lost[:, :, FULL_T - OPT.T]
        if not lost_now.any():
            return original(init_lam=init_lam, init_Z=init_Z, solve_log=solve_log, verbose=verbose)
        OPT.run_admm = original  # run_admm_with_failures falls back to the original
        try:
            return run_admm_with_failures(lost_now, drop_iteration, init_lam, init_Z, solve_log)
        finally:
            OPT.run_admm = run_admm

    OPT.run_admm = run_admm
    try:
        yield
    finally:
        OPT.run_admm = original


@contextmanager
def late_local_reconcile(lost):
    """Local re-optimisation with lost trades noticed only at the exchange of offers.

    Every building solved assuming its planned trades; trades over lost connections are
    not delivered, so the buyer imports the shortfall and the seller curtails (the same
    settlement as reopt._reconcile_trades for any undelivered offer).
    """
    original = reopt._reconcile_trades
    counter = itertools.count()  # simulate_local reconciles once per interval, in order

    def reconcile(outputs):
        lost_now = lost[:, :, next(counter)]
        offer = np.array([out["offer"] for out in outputs])
        accept = np.array([out["accept"] for out in outputs])
        realised = np.minimum(offer, accept.T)
        realised[lost_now] = 0.0
        finals = []
        for b, out in enumerate(outputs):
            p2p_export = float(realised[b].sum())
            p2p_import = float(realised[:, b].sum())
            final = {name: out[name] for name in reopt._FIRST_STEP_NAMES}
            final["p_el"] = out["p_el"] + float(accept[b].sum()) - p2p_import
            final["curtail"] = out["curtail"] + float(offer[b].sum()) - p2p_export
            final["p2p_import"] = p2p_import
            final["p2p_export"] = p2p_export
            finals.append(final)
        return finals

    reopt._reconcile_trades = reconcile
    try:
        yield
    finally:
        reopt._reconcile_trades = original


# ============================================================================
# ROLLING CENTRAL WITH CUT-OFF BUILDINGS
# ============================================================================
def last_plan_dispatch(b, interval, plan, soc, indoor, load, pv, t_out):
    """One building's interval under its last received set-points, no trades.

    Same recourse as the fixed schedule (simulate_noisy_realtime.simulate): battery
    set-points clipped by power and actual SOC, heat pump and boiler as planned, the grid
    takes the balance, and surplus tops up the battery before it is curtailed.
    """
    start, values = plan
    setpoint = {name: float(values[name][interval - start]) for name in _PLAN_NAMES}
    capacity, max_power = float(C.CAP[b]), float(C.MAXPOW[b])
    charge = discharge = 0.0
    battery = bool(C.HAS_BATTERY[b]) and interval > 0
    if battery:
        charge = min(max(setpoint["charge"], 0.0), max_power)
        discharge = min(max(setpoint["discharge"], 0.0), max_power)
        soc_next = soc + (C.eta_charge * charge - discharge / C.eta_discharge) * DT
        if soc_next > capacity:
            charge -= min(charge, (soc_next - capacity) / (C.eta_charge * DT))
        elif soc_next < 0.0:
            discharge -= min(discharge, -soc_next * C.eta_discharge / DT)
    net = load + setpoint["p_hp"] + charge - pv - discharge
    grid, surplus = max(net, 0.0), max(-net, 0.0)
    if battery and surplus > 0:
        soc_after = soc + (C.eta_charge * charge - discharge / C.eta_discharge) * DT
        headroom = min(max_power - charge, max((capacity - soc_after) / (C.eta_charge * DT), 0.0))
        extra = min(surplus, max(headroom, 0.0))
        charge += extra
        surplus -= extra
    if interval > 0:
        cop = C.cop_base + 0.01 * (t_out - C.T_ref)
        indoor = indoor + DT / 10 * (cop * setpoint["p_hp"] + setpoint["q_boiler"] - 0.5 * (indoor - t_out))
    return {
        "charge": charge,
        "discharge": discharge,
        "soc": soc + (C.eta_charge * charge - discharge / C.eta_discharge) * DT,
        "p_el": grid,
        "curtail": surplus,
        "p_hp": setpoint["p_hp"],
        "q_boiler": setpoint["q_boiler"],
        "gas": setpoint["gas"],
        "T_in": float(indoor),
    }


def uncontrolled_dispatch(b, interval, soc, indoor, load, pv, t_out):
    """One building's interval with no set-points and no optimiser (central without fallback).

    Battery idle, no trades. Its basic thermostat supplies the heat that brings the
    indoor temperature back to the setpoint by the end of the hour (same thermal model as
    the optimiser), from the heat pump first and the boiler for the rest. The grid takes
    the electrical balance; surplus PV is curtailed.
    """
    cop = C.cop_base + 0.01 * (t_out - C.T_ref)
    setpoint = float(reopt.FULL_T_SET[interval])
    heat = max(10.0 * (setpoint - indoor) / DT + 0.5 * (indoor - t_out), 0.0)
    heat_pump_heat = min(heat, C.p_th_nom, cop * C.hp_max_power)
    q_boiler = min(heat - heat_pump_heat, C.max_thermal_power)
    p_hp = heat_pump_heat / cop
    if interval > 0:
        indoor = indoor + DT / 10 * (heat_pump_heat + q_boiler - 0.5 * (indoor - t_out))
    net = load + p_hp - pv
    return {
        "charge": 0.0,
        "discharge": 0.0,
        "soc": float(soc),
        "p_el": max(net, 0.0),
        "curtail": max(-net, 0.0),
        "p_hp": p_hp,
        "q_boiler": q_boiler,
        "gas": q_boiler / C.efficiency,
        "T_in": float(indoor),
    }


_STATE_NAMES = ("soc", "T_in")


def _day_ahead_plan(schedule, b):
    """The day-ahead plan as the set-points a building holds before the coordinator reaches it."""
    values = {name: schedule["variables"][name][b].copy() for name in _PLAN_NAMES + _STATE_NAMES}
    values["q"] = schedule["trades"][b][OPT.NB[b]].copy()  # (N-1, T), + = b delivers
    return 0, values


def _plan_from_model(interval, sub):
    """The remaining-horizon set-points the coordinator sends to one building."""
    def series(var):
        return np.array([float(pyo.value(var[t], exception=False) or 0.0) for t in sub.t])
    values = {name: series(getattr(sub, name)) for name in _PLAN_NAMES + _STATE_NAMES}
    values["q"] = np.array([[float(pyo.value(sub.q[k, t], exception=False) or 0.0) for t in sub.t]
                            for k in sub.k])
    return interval, values


def _estimated_state(plan, interval, b):
    """State at the start of `interval` as predicted by the building's last plan."""
    start, values = plan
    if interval == 0 or interval - 1 < start:
        return float(reopt.DAY_START_SOC[b]), float(reopt.DAY_START_T_IN[b])
    return float(values["soc"][interval - 1 - start]), float(values["T_in"][interval - 1 - start])


@contextmanager
def central_cut_off(offline_now, plans, interval, fix_setpoints=True):
    """The coordinator's model when it cannot reach some buildings (failure known at the start).

    No trade with a cut-off building in the current hour. With fix_setpoints, its current
    hour is fixed to the set-points of its last plan, which it executes (last-plan
    fallback); otherwise the coordinator does not know what it will do and leaves it free.
    Its later hours are planned as usual (the coordinator expects to reach it again).
    """
    original = OPT.build_subproblem
    lost_now = pairs_from_offline(offline_now[:, None])[:, :, 0]

    def build_subproblem(b):
        m = original(b)
        _fix_pairs(m, b, lost_now)
        if offline_now[b] and fix_setpoints:
            start, values = plans[b]
            for name in ("charge", "discharge", "p_hp", "q_boiler"):
                var = getattr(m, name)[0]
                if not var.fixed:
                    value = max(float(values[name][interval - start]), var.lb or 0.0)
                    if name == "p_hp":
                        # the plan used the forecast COP; with the actual COP the same electrical
                        # power may exceed the heat pump's thermal rating (f <= 1) in the model
                        value = min(value, (1.0 - 1e-9) * C.p_th_nom / float(OPT.COP[0]))
                    var.fix(min(value, var.ub) if var.ub is not None else value)
        return m

    OPT.build_subproblem = build_subproblem
    try:
        yield
    finally:
        OPT.build_subproblem = original


def local_fallback(b, interval, plan, schedule, solver, args):
    """A cut-off building re-optimising itself as local re-optimisation does. Inside remaining_horizon.

    The trades of its last central plan are the limits for the later hours; there is no
    trade in the current hour.
    """
    start, values = plan
    trades = np.zeros((N, N, FULL_T))
    length = values["q"].shape[1]
    trades[b, OPT.NB[b], start:start + length] = values["q"]
    trades[b, :, interval] = 0.0
    reference = {"variables": schedule["variables"], "trades": trades}
    out = reopt._solve_local_building(b, interval, reference, solver, args.cost_penalty, args.share_penalty)
    return {name: out[name] for name in reopt._FIRST_STEP_NAMES}


def simulate_central(schedule, actual_load, actual_pv, actual_t_out, offline, *, run, args,
                     fallback, detection, method):
    """Rolling central (simulate_realtime_central) with buildings cut off from the coordinator.

    offline (N, T): buildings the coordinator cannot reach. fallback is what a cut-off
    building does: "none" (runs uncontrolled), "last_plan" or "local"; detection as in
    the module docstring.
    """
    solver = reopt._make_solver(nonconvex=not OPT.RELAX_BINARIES)
    local_solver = reopt._make_solver(nonconvex=True) if fallback == "local" else None
    soc = reopt.DAY_START_SOC.copy()
    temperature = reopt.DAY_START_T_IN.copy()
    plans = [_day_ahead_plan(schedule, b) for b in range(N)]
    records, timings = [], []
    for interval in range(FULL_T):
        down = offline[:, interval]
        started = time.perf_counter()
        inputs = reopt.horizon_inputs(interval, actual_load, actual_pv, actual_t_out, args.future_inputs)
        known = detection == "start" and down.any()
        # the coordinator has no measurement from a building it cannot reach (known failure)
        model_soc, model_temperature = soc.copy(), temperature.copy()
        if known:
            for b in np.flatnonzero(down):
                model_soc[b], model_temperature[b] = _estimated_state(plans[b], interval, b)
        fixing = [fallback != "none"]
        if fixing[0] and known:
            fixing.append(False)  # safeguard: retry with the cut-off buildings left free
        for fix_setpoints in fixing:
            hooks = central_cut_off(down, plans, interval, fix_setpoints=fix_setpoints) if known else nullcontext()
            with reopt.remaining_horizon(interval, *inputs, model_soc, model_temperature), hooks:
                m, subs = central.build_central_model()
                build_seconds = time.perf_counter() - started
                solve_started = time.perf_counter()
                result = solver.solve(m, tee=False)
                solver_seconds = time.perf_counter() - solve_started
                termination = result.solver.termination_condition
                solved = termination in reopt._ACCEPTABLE and pyo.value(subs[0].p_el[0], exception=False) is not None
                if solved:
                    break
                if fix_setpoints is not fixing[-1]:
                    print(f"  warning: central solve with fixed set-points failed at interval {interval} "
                          f"({termination}); retrying with the cut-off buildings free", flush=True)
                    continue
                raise RuntimeError(f"Central solve failed at interval {interval}: "
                                   f"status={result.solver.status}, termination={termination}")
        with reopt.remaining_horizon(interval, *inputs, model_soc, model_temperature):
            outputs = [{name: float(pyo.value(getattr(sub, name)[0])) for name in reopt._FIRST_STEP_NAMES}
                       for sub in subs]
            flows = np.zeros((N, N))  # flows[b, j]: b -> j as planned by the coordinator
            for b, sub in enumerate(subs):
                for k, neighbour in enumerate(OPT.NB[b]):
                    flows[b, neighbour] = max(float(pyo.value(sub.qpos[k, 0])), 0.0)
                if not down[b]:  # the set-points sent: kept in case b is cut off later
                    plans[b] = _plan_from_model(interval, sub)
        if detection == "late":
            # the coordinator planned trades with buildings that never got their set-points
            for b in np.flatnonzero(down):
                for j in range(N):
                    if not down[j]:
                        outputs[j]["p_el"] += flows[b, j]  # undelivered import -> grid
                        outputs[j]["curtail"] += flows[j, b]  # unsold export -> curtailed
                flows[b, :] = 0.0
                flows[:, b] = 0.0
        if down.any() and fallback == "local":
            inputs_now = reopt.horizon_inputs(interval, actual_load, actual_pv, actual_t_out, args.future_inputs)
            with reopt.remaining_horizon(interval, *inputs_now, soc, temperature):  # its own measured state
                for b in np.flatnonzero(down):
                    outputs[b] = local_fallback(b, interval, plans[b], schedule, local_solver, args)
        elif down.any() and fallback == "none":
            for b in np.flatnonzero(down):
                outputs[b] = uncontrolled_dispatch(b, interval, soc[b], temperature[b],
                                                   float(actual_load[b, interval]), float(actual_pv[b, interval]),
                                                   float(actual_t_out[interval]))
        elif down.any():
            for b in np.flatnonzero(down):
                outputs[b] = last_plan_dispatch(b, interval, plans[b], soc[b], temperature[b],
                                                float(actual_load[b, interval]), float(actual_pv[b, interval]),
                                                float(actual_t_out[interval]))
        wall = time.perf_counter() - started

        for b, out in enumerate(outputs):
            out["p2p_export"] = float(flows[b].sum())
            out["p2p_import"] = float(flows[:, b].sum())
            soc[b] = out["soc"]
            temperature[b] = out["T_in"]

        timings.append(reopt.timing_row(
            run, method, interval, "community", wall_seconds=wall, build_seconds=build_seconds,
            solver_seconds=solver_seconds, parallel_latency_seconds=wall, solver_calls=1,
            termination=str(termination),
        ))
        records.extend(
            reopt.make_record(run, method, interval, b, actual_load[b, interval], actual_pv[b, interval],
                              actual_t_out[interval], outputs[b], schedule)
            for b in range(N)
        )
        print(f"  run {run} central  interval {interval + 1:02d}/{FULL_T}: {int(down.sum())} cut off, "
              f"solve {solver_seconds:5.2f}s, grid {sum(o['p_el'] for o in outputs):6.2f} kW", flush=True)
    return records, timings


# ============================================================================
# ROLLING ADMM WITH AN ITERATION BUDGET AND A LOCAL FALLBACK
# ============================================================================
ADMM_FALLBACK_ITERS = 20


def _agreed_trades(agreed):
    """(N, N, FULL_T) trades of the last agreement (+ = b delivers to j) from (start, Z)."""
    start, Z = agreed
    trades = np.zeros((N, N, FULL_T))
    for (lo, hi), z in Z.items():
        z = np.asarray(z, dtype=float)
        trades[lo, hi, start:start + len(z)] = z
        trades[hi, lo, start:start + len(z)] = -z
    return trades


def simulate_admm_fallback(schedule, actual_load, actual_pv, actual_t_out, lost, *, run, args, late):
    """Rolling ADMM with at most args.admm_fallback_iters iterations per hour.

    Converged within the budget: identical to rolling_admm. Otherwise every building
    re-optimises locally (reopt._solve_local_building) with the last agreement everyone
    reached as its trade limits; the current hour's trades over lost connections are zero
    (failure_hooks). The capped iterate still warm-starts the next hour.
    Parallel time = Σ_iterations max_b(subproblem time) + max_b(local fallback time).
    """
    cap = args.admm_fallback_iters
    local_solver = reopt._make_solver(nonconvex=True)
    soc, temperature = reopt.DAY_START_SOC.copy(), reopt.DAY_START_T_IN.copy()
    previous_lam = None
    previous_Z = {(lo, hi): schedule["trades"][lo, hi].copy() for lo, hi in OPT.PAIRS}
    agreed = (0, previous_Z)  # (interval it starts at, Z): the last agreement everyone reached
    records, timings = [], []
    for interval in range(FULL_T):
        shift = 0 if interval == 0 else 1
        init_Z = {pair: values[shift:] for pair, values in previous_Z.items()}
        init_lam = None if previous_lam is None else [lam[:, shift:] for lam in previous_lam]
        solve_log = []
        started = time.perf_counter()
        inputs = reopt.horizon_inputs(interval, actual_load, actual_pv, actual_t_out, args.future_inputs)
        original_cap = OPT.MAX_ITERS
        OPT.MAX_ITERS = cap
        try:
            with admm_failures(lost, args.drop_iteration if late else 0), \
                    reopt.remaining_horizon(interval, *inputs, soc, temperature):
                subs, _q, lam_all, Z, hist = OPT.run_admm(init_lam=init_lam, init_Z=init_Z,
                                                          solve_log=solve_log, verbose=False)
                outputs = [{name: float(pyo.value(getattr(m, name)[0])) for name in reopt._FIRST_STEP_NAMES}
                           for m in subs]
        finally:
            OPT.MAX_ITERS = original_cap
        previous_lam, previous_Z = lam_all, Z
        iterations = int(len(hist))
        converged = iterations < cap or (hist[-1, 3] <= OPT.EPS_PRIMAL and hist[-1, 4] <= OPT.EPS_DUAL)
        log = np.array(solve_log, dtype=float).reshape(-1, 3)
        admm_parallel = sum(float(log[log[:, 0] == i, 2].max()) for i in range(iterations))

        fallback_parallel = 0.0
        if converged:
            agreed = (interval, Z)
            flows = np.zeros((N, N))  # flows[b, j]: b -> j at the current interval
            for lo, hi in OPT.PAIRS:
                z = float(Z[(lo, hi)][0])
                flows[lo, hi], flows[hi, lo] = max(z, 0.0), max(-z, 0.0)
            for b, out in enumerate(outputs):
                out["p2p_export"] = float(flows[b].sum())
                out["p2p_import"] = float(flows[:, b].sum())
        else:  # no agreement in time: every building re-optimises on the last agreed trades
            reference = {"variables": schedule["variables"], "trades": _agreed_trades(agreed)}
            local = []
            with failure_hooks(lost), reopt.remaining_horizon(interval, *inputs, soc, temperature):
                for b in range(N):
                    solve_started = time.perf_counter()
                    local.append(reopt._solve_local_building(b, interval, reference, local_solver,
                                                             args.cost_penalty, args.share_penalty))
                    fallback_parallel = max(fallback_parallel, time.perf_counter() - solve_started)
            outputs = reopt._reconcile_trades(local)
        wall = time.perf_counter() - started
        for b, out in enumerate(outputs):
            soc[b], temperature[b] = out["soc"], out["T_in"]

        timings.append(reopt.timing_row(
            run, ADMM_FALLBACK, interval, "community", wall_seconds=wall, solver_seconds=float(log[:, 2].sum()),
            parallel_latency_seconds=admm_parallel + fallback_parallel, solver_calls=len(log) + (0 if converged else N),
            admm_iterations=iterations, termination="converged" if converged else "fallback",
        ))
        records.extend(
            reopt.make_record(run, ADMM_FALLBACK, interval, b, actual_load[b, interval], actual_pv[b, interval],
                              actual_t_out[interval], outputs[b], schedule)
            for b in range(N)
        )
        print(f"  run {run} ADMM+fb  interval {interval + 1:02d}/{FULL_T}: {iterations:3d} iters, "
              f"{'agreed  ' if converged else 'FALLBACK'} parallel {admm_parallel + fallback_parallel:5.2f}s, "
              f"grid {sum(o['p_el'] for o in outputs):6.2f} kW", flush=True)
    return records, timings


# ============================================================================
# RUNNING THE METHODS
# ============================================================================
def run_method(method, inputs, lost, offline, *, run, test, args):
    late = test["detection"] == "late"
    if method == "open_loop":
        return open_loop.simulate_open_loop(*inputs, run=run)
    if method == "fixed_schedule":
        return compare.simulate_fixed(*inputs, run=run, write=False)
    if method == "local_reoptimisation":
        with late_local_reconcile(lost) if late else failure_hooks(lost):
            return reopt.simulate_local(*inputs, run=run, cost_penalty=args.cost_penalty,
                                        share_penalty=args.share_penalty, future_inputs=args.future_inputs)
    if method == "rolling_admm":
        with admm_failures(lost, args.drop_iteration if late else 0):
            return reopt.simulate_rolling_admm(*inputs, run=run, warm_start=args.admm_start == "warm",
                                               future_inputs=args.future_inputs)
    if method == ADMM_FALLBACK:
        return simulate_admm_fallback(*inputs, lost, run=run, args=args, late=late)
    if method in CENTRAL_METHODS:
        return simulate_central(*inputs, offline, run=run, args=args,
                                fallback={CENTRAL_LOCAL: "local", CENTRAL_NONE: "none"}.get(method, "last_plan"),
                                detection=test["detection"], method=method)
    raise ValueError(f"Unknown method {method!r}")


def annotate(records, timings, level, method, lost, offline):
    """Mark each building-interval with the connections this method's topology lost."""
    index = {str(building): b for b, building in enumerate(C.BUILDING_IDS)}
    if method in CENTRAL_METHODS:
        lost_links = offline.astype(int)  # one link per building, to the coordinator
        cut_off = offline
    else:  # peer-to-peer (also shown for open loop / fixed schedule, which ignore it)
        lost_links = lost.sum(axis=1)
        cut_off = lost_links >= max(N - 1, 1)
    for record in records:
        b, t = index[str(record["building"])], record["interval"]
        record["failure_level"] = level
        record["lost_links"] = int(lost_links[b, t])
        record["building_offline"] = bool(cut_off[b, t])
    for row in timings:
        row["failure_level"] = level
        row["offline_buildings"] = int(cut_off[:, row["interval"]].sum())


def load_scenarios(plan_dir, case, runs, seed, out_dir):
    """The 05 noise scenarios of a case, read without ever rewriting 05's files."""
    folder = plan_dir if case == "base" else plan_dir / "cases" / case
    path = folder / baseline.SCENARIO_FILE_NAME
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if (stored["actual_load"].shape[0] >= runs and stored["actual_load"].shape[1:] == C.LOAD.shape
                    and np.allclose(stored["forecast_load"], C.LOAD) and np.allclose(stored["forecast_pv"], C.PV_B)):
                print(f"Using the shared noise scenarios of {path}")
                noise = {name: float(stored[name]) for name in ("load_noise", "pv_noise", "temp_noise")}
                return (stored["actual_load"][:runs].copy(), stored["actual_pv"][:runs].copy(),
                        stored["actual_t_out"][:runs].copy(), noise)
    print(f"{path} is missing or has fewer than {runs} matching scenarios; drawing '{case}' scenarios here.")
    settings = SimpleNamespace(load_bias=0.0, bias_hours="16-22", pv_bias=0.0, correlation=0.0)
    for name, value in compare.CASES[case]["args"].items():
        setattr(settings, name, value)
    noise = {"load_noise": baseline.DEFAULT_LOAD_NOISE, "pv_noise": 0.30, "temp_noise": 1.0}
    loads, pvs, t_outs = baseline.load_or_create_scenarios(
        out_dir / baseline.SCENARIO_FILE_NAME, runs=runs, seed=seed, load_bias=compare._load_bias(settings),
        correlation=settings.correlation, pv_bias=settings.pv_bias, **noise)
    return loads, pvs, t_outs, noise


# ============================================================================
# SUMMARY AND FIGURES
# ============================================================================
def summarise(records, timings, run, method, level):
    row = compare.summarise_run(records, timings, run, method)
    data = pd.DataFrame(records)
    offline = data["building_offline"].astype(bool)
    row["failure_level"] = level
    row["offline_building_hours"] = int(offline.sum())
    row["lost_link_hours"] = int(data["lost_links"].sum())
    row["offline_building_hours_cost_gbp"] = float(data.loc[offline, "final_total_cost"].sum())
    row["online_building_hours_cost_gbp"] = float(data.loc[~offline, "final_total_cost"].sum())
    row["foregone_p2p_kwh"] = float(data["foregone_p2p_kwh"].sum())
    return row


def _method_order(series):
    order = {method: i for i, method in enumerate(ALL_METHODS)}
    return series.map(order) if series.name == "method" else series


def add_changes(table):
    """Columns comparing each row with the same run and method without failures (level 0)."""
    table = table.copy()
    reference = table[table["failure_level"] == 0].set_index(["simulation_run", "method"])
    keys = list(zip(table["simulation_run"], table["method"]))
    for column, name in (("final_total_cost_gbp", "cost_increase_vs_no_failure_gbp"),
                         ("final_grid_import_kwh", "grid_import_increase_vs_no_failure_kwh"),
                         ("final_p2p_kwh", "p2p_change_vs_no_failure_kwh"),
                         ("final_peak_grid_kw", "peak_grid_increase_vs_no_failure_kw")):
        base = reference[column] if len(reference) else pd.Series(dtype=float)
        table[name] = table[column].to_numpy() - np.array([base.get(key, np.nan) for key in keys])
    has_central = len(reference) and "rolling_central" in reference.index.get_level_values("method")
    central_base = reference.xs("rolling_central", level="method")["final_total_cost_gbp"] \
        if has_central else pd.Series(dtype=float)
    table["cost_vs_central_no_failure_gbp"] = table["final_total_cost_gbp"].to_numpy() - np.array(
        [central_base.get(run, np.nan) for run in table["simulation_run"]])
    return table.sort_values(["failure_level", "simulation_run", "method"], key=_method_order)


def finish_summary(table, out):
    """Add the change against no failure and write the per-run and mean summaries."""
    table = add_changes(table)
    table.to_csv(out.summary, index=False)
    numeric = list(table.drop(columns=["simulation_run", "failure_level"]).select_dtypes("number").columns)
    mean = table.groupby(["failure_level", "method"])[numeric].agg(["mean", "std"])
    mean.columns = [f"{name}_{stat}" for name, stat in mean.columns]
    mean.reset_index().sort_values(["failure_level", "method"], key=_method_order).to_csv(out.mean, index=False)
    return table


HATCHES = {CENTRAL_LOCAL: "//", "open_loop": "..", ADMM_FALLBACK: "\\\\"}  # second cue where colours are close
METHOD_TITLES = {"open_loop": "Open loop", "fixed_schedule": "Fixed schedule",
                 "local_reoptimisation": "Local re-optimisation", "rolling_admm": "Rolling ADMM",
                 ADMM_FALLBACK: "ADMM + iteration-budget fallback",
                 CENTRAL_NONE: "Central, no fallback", "rolling_central": "Central + last-plan fallback",
                 CENTRAL_LOCAL: "Central + local fallback"}


def method_bars(axis, table, column, groups, method, *, group_column="failure_level", labels=None, title=True):
    """One method's panel: mean ± std over runs of `column`, one bar per group (e.g. failure level)."""
    _style(axis)
    stats = (table[table["method"] == method].groupby(group_column)[column]
             .agg(["mean", "std"]).reindex(groups))
    x = np.arange(len(groups))
    axis.bar(x, stats["mean"], width=0.6, yerr=stats["std"].fillna(0.0), capsize=3, color=COLORS[method],
             hatch=HATCHES.get(method), edgecolor="white", linewidth=0,
             error_kw={"elinewidth": 1, "ecolor": PLAN_COLOR})
    for position, value, spread in zip(x, stats["mean"], stats["std"].fillna(0.0)):
        if np.isfinite(value):  # label beyond the end of the error bar
            end = value + spread if value >= 0 else value - spread
            axis.annotate(f"{value:+.2f}", (position, end), xytext=(0, 3 if value >= 0 else -3),
                          textcoords="offset points", ha="center", va="bottom" if value >= 0 else "top",
                          fontsize=7, color=INK)
    axis.axhline(0, color=PLAN_COLOR, lw=1)
    axis.set_xticks(x)
    axis.set_xticklabels(labels or [f"{g:g}" if isinstance(g, (int, float, np.number)) else str(g) for g in groups],
                         fontsize=8)
    axis.grid(False, axis="x")
    axis.margins(y=0.15)
    if title:
        axis.set_title(METHOD_TITLES[method], fontsize=9, color=INK)


def plot_robustness(table, test, out):
    """One column per method; rows: cost vs benchmark, cost of failures, P2P lost, peak change."""
    levels = sorted(table["failure_level"].unique())
    failures = [level for level in levels if level != 0]
    if not failures:
        print("Only the no-failure level: skipping the robustness figure.")
        return
    methods = [m for m in ALL_METHODS if m in set(table["method"])]
    rows = [
        ("cost_vs_central_no_failure_gbp", levels, "a) Cost minus rolling\ncentral without\nfailures (GBP/day)"),
        ("cost_increase_vs_no_failure_gbp", failures, "b) Cost increase vs\nno failure (GBP/day)"),
        ("p2p_change_vs_no_failure_kwh", failures, "c) Change in P2P\ndelivered (kWh/day)"),
        ("peak_grid_increase_vs_no_failure_kw", failures, "d) Change in peak\nimport (kW)"),
    ]
    rows = [row for row in rows if row[0] in table]
    fig, axes = plt.subplots(len(rows), len(methods), figsize=(2.3 * len(methods) + 1.2, 2.3 * len(rows) + 1.2),
                             sharey="row", squeeze=False)
    for r, (column, row_levels, ylabel) in enumerate(rows):
        for c, method in enumerate(methods):
            method_bars(axes[r, c], table, column, row_levels, method, title=r == 0)
        axes[r, 0].set_ylabel(ylabel, fontsize=8, color=INK)
    for axis in axes[-1]:
        axis.set_xlabel(LEVEL_LABELS[test["mode"]], fontsize=8, color=INK)
    runs = table["simulation_run"].nunique()
    fig.suptitle(f"Communication failures by method: {test['description']}\n"
                 f"N = {N}, mean ± std over {runs} run{'s' if runs != 1 else ''}", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out.figure)
    plt.close(fig)
    print(f"Figure: {out.figure}")


def load_summary(name):
    """A test's per-run summary with the change columns (also while it is still being written)."""
    path = Outputs(RESULTS_ROOT / name).summary
    if not (path.exists() and path.stat().st_size):
        return None
    table = pd.read_csv(path)
    # compare methods on the same scenarios: keep a (level, run) only once every method has finished it
    finished = table.groupby(["failure_level", "simulation_run"])["method"].transform("nunique") \
        == table["method"].nunique()
    return add_changes(table[finished])


def plot_overview():
    """One panel per method: cost increase at each test's highest failure level, one bar per test."""
    tables = {}
    for name in TESTS:
        table = load_summary(name)
        if table is not None and (table["failure_level"] != 0).any() and (table["failure_level"] == 0).any():
            tables[name] = table
    if not tables:
        print("No test results to overview yet.")
        return
    rows, top = [], []
    for name, table in tables.items():
        mean = table.groupby(["failure_level", "method"])["cost_increase_vs_no_failure_gbp"].agg(["mean", "std"])
        rows.append(mean.reset_index().assign(test=name, description=TESTS[name]["description"]))
        level = table["failure_level"].max()
        top.append(table[table["failure_level"] == level].assign(
            test=f"{name.replace('_', ' ')}\n({LEVEL_SYMBOLS[TESTS[name]['mode']]} = {level:g})"))
    pd.concat(rows, ignore_index=True)[["test", "description", "failure_level", "method", "mean", "std"]].rename(
        columns={"mean": "cost_increase_vs_no_failure_gbp_mean", "std": "cost_increase_vs_no_failure_gbp_std"}
    ).to_csv(OVERVIEW_FILE, index=False)

    top = pd.concat(top, ignore_index=True)
    tests = list(dict.fromkeys(top["test"]))
    methods = [m for m in ALL_METHODS if m in set(top["method"])]
    columns = 3
    rows_n = int(np.ceil(len(methods) / columns))
    panel_width = max(0.75 * len(tests), 3.4)
    fig, axes = plt.subplots(rows_n, columns, figsize=(panel_width * columns + 1.5, 3.4 * rows_n + 1.4),
                             sharey=True, squeeze=False)
    for axis, method in zip(axes.flat, methods):
        method_bars(axis, top, "cost_increase_vs_no_failure_gbp", tests, method, group_column="test")
        plt.setp(axis.get_xticklabels(), rotation=60, ha="right", fontsize=7)
    for axis in axes.flat[len(methods):]:
        axis.set_visible(False)
    for axis in axes[:, 0]:
        axis.set_ylabel("Cost increase vs no failure (GBP/day)", fontsize=8, color=INK)
    fig.suptitle(f"Cost of communication failures by method, at each test's highest level "
                 f"(N = {N}, mean ± std over runs)", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, OVERVIEW_FIGURE)
    plt.close(fig)
    print(f"Overview: {OVERVIEW_FIGURE}")


def summarise_saved(test_names):
    for name in test_names:
        out = Outputs(RESULTS_ROOT / name)
        if not (out.results.exists() and out.results.stat().st_size):
            continue
        data = pd.read_csv(out.results)
        timings = pd.read_csv(out.timings)
        settings = {}
        if out.summary.exists() and out.summary.stat().st_size:
            saved = pd.read_csv(out.summary)
            columns = [c for c in SETTING_COLUMNS if c in saved]
            settings = saved.drop_duplicates("method").set_index("method")[columns].to_dict("index")
        rows = []
        for (level, run, method), records in data.groupby(["failure_level", "simulation_run", "method"]):
            steps = timings[(timings.failure_level == level) & (timings.simulation_run == run)
                            & (timings.method == method)]
            row = summarise(records.to_dict("records"), steps.to_dict("records"), run, method, level)
            row.update(settings.get(method, {}))
            rows.append(row)
        table = finish_summary(pd.DataFrame(rows), out)
        plot_robustness(table, {**DEFAULTS, **TESTS[name]}, out)
        print(f"Rebuilt the summaries and figure of '{name}' from saved results.")
    plot_overview()


def saved_summary_rows(out, settings):
    """Summary rows of every finished (level, run, method) in a test's saved results."""
    if not (out.results.exists() and out.results.stat().st_size):
        return []
    data = pd.read_csv(out.results)
    timings = pd.read_csv(out.timings)
    rows = []
    for (level, run, method), records in data.groupby(["failure_level", "simulation_run", "method"]):
        if len(records) != FULL_T * N:
            continue
        steps = timings[(timings.failure_level == level) & (timings.simulation_run == run)
                        & (timings.method == method)]
        row = summarise(records.to_dict("records"), steps.to_dict("records"), run, method, level)
        row.update(settings)
        rows.append(row)
    return rows


SETTING_COLUMNS = ("test", "failure_mode", "target", "detection", "case", "outage_hours", "drop_iteration",
                   "failure_seed", "future_inputs", "share_penalty_gbp_per_kwh", "admm_start",
                   "load_noise", "pv_noise", "temp_noise")


# ============================================================================
# MAIN
# ============================================================================
def run_test(name, args, plan_dir, schedule, methods):
    test = {**DEFAULTS, **TESTS[name]}
    levels = sorted(set(args.levels if args.levels is not None else test["levels"]))
    mode = test["mode"]
    if mode == "building" and not all(float(k).is_integer() and 0 <= k <= N for k in levels):
        raise SystemExit(f"{name}: levels are buildings offline per hour, integers 0..{N}.")
    if mode == "link" and not all(0 <= p <= 1 for p in levels):
        raise SystemExit(f"{name}: levels are link failure probabilities in [0, 1].")
    if mode == "coordinator" and not all(float(h).is_integer() and 0 <= h <= FULL_T for h in levels):
        raise SystemExit(f"{name}: levels are coordinator down hours, integers 0..{FULL_T}.")
    if mode != "link":
        levels = [int(level) for level in levels]

    out = Outputs(RESULTS_ROOT / name)
    out.dir.mkdir(parents=True, exist_ok=True)
    print(f"\n#################### Test '{name}': {test['description']} -> {out.dir}", flush=True)
    loads, pvs, t_outs, noise = load_scenarios(plan_dir, test["case"], args.runs, args.seed, out.dir)
    patterns = draw_patterns(test, levels, args.runs, args.failure_seed)
    np.savez_compressed(
        out.masks,
        **{f"lost_p2p_level_{level:g}": lost for level, (lost, _) in patterns.items()},
        **{f"central_offline_level_{level:g}": offline for level, (_, offline) in patterns.items()},
        building_ids=np.array([str(b) for b in C.BUILDING_IDS]), levels=np.array(levels, dtype=float),
        mode=np.array(mode), outage_hours=np.array(test["outage_hours"]), target=np.array(test["target"]),
        detection=np.array(test["detection"]), case=np.array(test["case"]), failure_seed=np.array(args.failure_seed),
    )
    settings = {"test": name, "failure_mode": mode, "target": test["target"], "detection": test["detection"],
                "case": test["case"], "outage_hours": test["outage_hours"], "drop_iteration": args.drop_iteration,
                "failure_seed": args.failure_seed, "future_inputs": args.future_inputs,
                "share_penalty_gbp_per_kwh": args.share_penalty, "admm_start": args.admm_start,
                "admm_fallback_iters": args.admm_fallback_iters, **noise}
    done = set()
    if args.resume and out.results.exists() and out.results.stat().st_size:
        # keep every finished (level, run, method); the summary restarts from them, without change columns
        saved = pd.read_csv(out.results)
        sizes = saved.groupby(["failure_level", "simulation_run", "method"]).size()
        done = {(float(level), int(run), method) for (level, run, method), size in sizes.items() if size == FULL_T * N}
        rows = saved_summary_rows(out, settings)
        out.summary.write_text("", encoding="utf-8")
        reopt.append_rows(out.summary, rows)
        print(f"Resuming '{name}': {len(done)} finished level/scenario/method runs kept.", flush=True)
    else:
        for path in (out.results, out.timings, out.summary, out.mean):
            path.write_text("", encoding="utf-8")

    for level in levels:
        lost_all, offline_all = patterns[level]
        for run in range(1, args.runs + 1):
            todo = [method for method in methods if (float(level), run, method) not in done]
            if not todo:
                continue
            lost, offline = lost_all[run - 1], offline_all[run - 1]
            print(f"\n===== {name}: level {level:g}, scenario {run}/{args.runs}: "
                  f"{int(lost.sum()) // 2} lost peer links, {int(offline.sum())} building-hours cut off "
                  f"from the coordinator =====", flush=True)
            inputs = (schedule, loads[run - 1], pvs[run - 1], t_outs[run - 1])
            for method in todo:
                started = time.perf_counter()
                records, timings = run_method(method, inputs, lost, offline, run=run, test=test, args=args)
                annotate(records, timings, level, method, lost, offline)
                reopt.append_rows(out.results, records)
                reopt.append_rows(out.timings, timings)
                row = summarise(records, timings, run, method, level)
                row.update(settings)
                reopt.append_rows(out.summary, [row])  # partial progress, rewritten below
                print(f"  {method:<32} cost {row['final_total_cost_gbp']:8.2f} GBP, "
                      f"P2P {row['final_p2p_kwh']:6.2f} kWh, peak {row['final_peak_grid_kw']:6.2f} kW "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)

    table = finish_summary(pd.DataFrame(saved_summary_rows(out, settings)), out)
    plot_robustness(table, test, out)
    mean = pd.read_csv(out.mean)
    print(f"\n{name}: mean cost increase against no failure (GBP/day):")
    print(mean.pivot(index="failure_level", columns="method",
                     values="cost_increase_vs_no_failure_gbp_mean").reindex(columns=methods).round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", nargs="+", choices=(*TESTS, "all"), default=["building"])
    parser.add_argument("--runs", type=int, default=3, help="Noise scenarios (the first runs of experiment 05).")
    parser.add_argument("--levels", type=float, nargs="+", default=None,
                        help="Override the failure levels of the test(s); include 0 for the reference.")
    parser.add_argument("--failure-seed", type=int, default=7, help="Seed of the failure patterns.")
    parser.add_argument("--drop-iteration", type=int, default=1,
                        help="Late detection: ADMM iterations completed before lost connections drop out.")
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(ALL_METHODS))
    parser.add_argument("--future-inputs", choices=("noisy", "forecast"), default="noisy")
    parser.add_argument("--cost-penalty", type=float, default=10.0)
    parser.add_argument("--share-penalty", type=float, default=0.1, help="GBP per kWh of planned P2P not delivered.")
    parser.add_argument("--admm-start", choices=("warm", "cold"), default="warm")
    parser.add_argument("--admm-fallback-iters", type=int, default=ADMM_FALLBACK_ITERS,
                        help=f"{ADMM_FALLBACK}: ADMM iterations allowed per hour before every building "
                             f"falls back to local re-optimisation on the last agreed trades.")
    parser.add_argument("--seed", type=int, default=2026, help="Noise seed, only if scenarios must be drawn.")
    parser.add_argument("--plan-dir", type=Path, default=None,
                        help="Folder with the day-ahead plan cache and noise scenarios (default: Results/05_realtime_methods).")
    parser.add_argument("--summarise-only", action="store_true",
                        help="Rebuild the summaries and figures from the saved results (no solving).")
    parser.add_argument("--resume", action="store_true",
                        help="Keep the finished level/scenario/method runs of each test and run only the missing ones.")
    args = parser.parse_args()
    names = list(TESTS) if "all" in args.test else list(dict.fromkeys(args.test))
    if args.summarise_only:
        summarise_saved(list(TESTS) if args.test == ["building"] else names)
        return
    if args.runs < 1 or args.drop_iteration < 1:
        parser.error("--runs and --drop-iteration must be at least 1.")
    methods = [m for m in ALL_METHODS if m in args.methods]

    plan_dir = args.plan_dir or default_plan_dir()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    schedule = baseline.load_or_solve_schedule(plan_dir / "bilateral_admm_schedule.npz")
    started = time.perf_counter()
    for name in names:
        run_test(name, args, plan_dir, schedule, methods)
    plot_overview()
    print(f"\nAll tests finished in {(time.perf_counter() - started) / 60:.0f} min.")


if __name__ == "__main__":
    main()
