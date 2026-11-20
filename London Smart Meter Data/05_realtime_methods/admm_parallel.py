"""Parallel bilateral ADMM: the building subproblems of each iteration solved at once.

PURPOSE
    Same answer as the sequential bilateral ADMM, but measured the way it would
    run in the field: one controller per building, all solving at the same time.

OBJECTIVES BUILT BY _build()   (constraints = building model B_b in every case)

  1. ADMM subproblem (default; = admm_bilateral_p2p_valley.build_subproblem)
         Σ_t π g Δt + c_gas G Δt + α (T − T_set)² + w_t g²
       + Σ λ q + ρ/2 Σ (q − z)² + 1e-4 Σ q⁺
       [+ Σ (γ_bj/2)(q⁺ + q⁻) Δt   exogenous network charge, if admm_bilateral_p2p_valley.NETWORK_CHARGE]
       [+ Σ (κ_bj/2) π_t q² Δt      quadratic network losses, if admm_bilateral_p2p_valley.NETWORK_LOSSES]
     (λ q − ρ z q is linear and rewritten in the Obj attribute every iteration.)

  2. + sharing-plan tracking  (EXTRA_DATA["tracking"], minute level of three_timescale.py)
       mode "trades":  + W Δt Σ_t Σ_j (q_j − q_j^plan)²       (sharing plan)
       mode "grid":    + W_g Δt Σ_t (g_t − g^plan)² − μ_soc s_end − μ_T T_end
                       (trades free to re-route; μ = the hourly solve's shadow prices, ParallelADMM.state_values)
       both:           + W_soc (s_end − s^plan)² + W_T (T_end − T^plan)²

  3. + ramp limits    (tracking["ramp"], 09_ramp_rate_control)
       grid mode:      E_t = g_t Δt,  r⁺_t ≥ E_t − E_{t−1} − ΔE,  r⁻_t ≥ E_{t−1} − E_t − ΔE
                       + W_r/Δt Σ (r⁺² + r⁻²)   [and/or  + W_lin Σ (r⁺ + r⁻)]
       P2P mode (6a)-(6c):
         producers     E_t = Δt Σ_j q⁺_j, same up/down excess, + W_r/Δt Σ (r⁺² + r⁻²)
         consumers     E_t = Δt (g + Σ_j q⁻_j),  x_t ≥ Σ_{l≤t} E_l − D_t,  + W_cum Σ x_t²

  4. local re-optimisation  (local={...}, = simulate_realtime_reoptimise)
         own cost + FEE_FRAC π Δt Σ_j (q⁻_j − q⁺_j)
       + share_penalty Δt Σ_j (|q_j^plan| − q⁺_j − q⁻_j)
       with 0 ≤ q⁺_j ≤ max(q_j^plan, 0), 0 ≤ q⁻_j ≤ max(−q_j^plan, 0)

Same algorithm as admm_bilateral_p2p_valley.run_admm (same subproblem, z- and
lambda-updates, residuals and stopping test), with two speed-ups:

  * the N building subproblems of an iteration are spread over --workers
    persistent processes, one Gurobi thread each;
  * every worker builds its buildings' models once per ADMM run directly in
    gurobipy and, between iterations, only rewrites the linear objective
    coefficients of the trades (lambda - rho z) instead of rebuilding the
    model through Pyomo.

Provides
    ParallelADMM                    pool of workers; .run(...) mirrors run_admm
    solve_schedule_parallel(...)    day-ahead schedule, written to the same cache
                                    (and signature) as simulate_noisy_realtime
    simulate_rolling_admm_parallel  rolling ADMM, same records/timings as
                                    simulate_realtime_reoptimise.simulate_rolling_admm

Timing columns: wall_seconds is the measured time with the worker pool on this
machine; parallel_latency_seconds keeps the one-controller-per-building estimate
(sum over iterations of the slowest subproblem), as for rolling_admm.

Examples:
    python admm_parallel.py --schedule-only
    python admm_parallel.py --runs 1 --workers 8
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import project_paths  # noqa: E402  (shared modules on sys.path; results -> Results/<experiment>/)

project_paths.register(__file__)

import argparse
import multiprocessing as mp
import os
import time
import traceback

import gurobipy as gp
import numpy as np
from gurobipy import GRB

METHOD = "rolling_admm_parallel"
DEFAULT_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 2))


# ============================================================================
# WORKER SIDE (only numpy + gurobipy, so spawned workers start quickly)
# ============================================================================
def _build(data, b, env, local=None):
    """gurobipy copy of admm_bilateral_p2p_valley.build_subproblem(b).

    With local={"planned": (N-1, T) trades, "share_penalty": GBP/kWh} it is instead the
    local re-optimisation model of simulate_realtime_reoptimise._solve_local_building.
    """
    T, K, DT = data["T"], data["N"] - 1, data["DT"]
    load, pv = data["LOAD"][b], data["PV_B"][b]
    maxpow, cap = float(data["MAXPOW"][b]), float(data["CAP"][b])
    rolling = data["ROLLING_FIRST_INTERVAL"]
    eta_c, eta_d = data["eta_charge"], data["eta_discharge"]
    m = gp.Model(env=env)
    ts, ks = range(T), range(K)

    def series(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS):
        return [m.addVar(lb=lb, ub=ub, vtype=vtype) for _ in ts]

    def matrix(lb=0.0):
        return [[m.addVar(lb=lb) for _ in ts] for _ in ks]

    battery_ub = maxpow if data["HAS_BATTERY"][b] else 0.0
    charge = series(ub=battery_ub)
    discharge = series(ub=battery_ub)
    soc = series(ub=max(cap, 1e-6))
    cs = series(ub=1.0 if data["HAS_BATTERY"][b] else 0.0,
                vtype=GRB.CONTINUOUS if data["RELAX_BINARIES"] else GRB.BINARY)
    p_el, curtail = series(), series()
    p_hp = series(ub=data["hp_max_power"])
    f = series(ub=1.0)
    q_boiler = series(ub=data["max_thermal_power"])
    gas, T_in = series(), series()
    q = matrix(lb=-GRB.INFINITY)
    qpos, qneg = matrix(), matrix()

    init_soc, init_t = float(data["INIT_SOC"][b]), float(data["INIT_T_IN"][b])
    p_th, t_out, cop = data["p_th_nom"], data["T_OUT"], data["COP"]
    links = {"soc": [None] * T, "T_in": [None] * T}  # state dynamics t-1 -> t (their duals price the state)
    for t in ts:
        m.addConstr(charge[t] <= maxpow * cs[t])
        m.addConstr(discharge[t] <= maxpow * (1 - cs[t]))
        step = (eta_c * charge[t] - (1.0 / eta_d) * discharge[t]) * DT
        heat = p_th * f[t] + q_boiler[t]
        if t > 0:
            links["soc"][t] = m.addConstr(soc[t] == soc[t - 1] + step)
            links["T_in"][t] = m.addConstr(T_in[t] == T_in[t - 1] + DT / 10 * (heat - 0.5 * (T_in[t - 1] - t_out[t])))
        elif rolling:
            m.addConstr(soc[0] == init_soc + step)
            m.addConstr(T_in[0] == init_t + DT / 10 * (heat - 0.5 * (init_t - t_out[0])))
        else:
            m.addConstr(soc[0] == init_soc)
            m.addConstr(T_in[0] == init_t)
            m.addConstr(charge[0] == 0)
            m.addConstr(discharge[0] == 0)
        m.addConstr(cop[t] * p_hp[t] == p_th * f[t])
        m.addConstr(gas[t] == q_boiler[t] / data["efficiency"])
        for k in ks:
            m.addConstr(q[k][t] == qpos[k][t] - qneg[k][t])
        m.addConstr(p_el[t] - gp.quicksum(q[k][t] for k in ks) - curtail[t]
                    == load[t] + charge[t] + p_hp[t] - discharge[t] - pv[t])
        m.addConstr(gp.quicksum(qpos[k][t] for k in ks) <= pv[t] + discharge[t])
        m.addConstr(gp.quicksum(qneg[k][t] for k in ks) <= load[t] + p_hp[t] + charge[t])

    price, t_set, rho = data["PRICE"], data["T_SET"], data["RHO"]
    obj = gp.QuadExpr()
    for t in ts:
        obj += price[t] * DT * p_el[t] + data["gas_price"] / 100.0 * DT * gas[t]
        obj += data["alpha"] * (T_in[t] - t_set[t]) * (T_in[t] - t_set[t])
        if data["use_congestion"]:
            obj += data["congestion_weight"][t] * p_el[t] * p_el[t]
        if local is None:
            for k in ks:
                # lambda q - rho z q is linear in q: written to the Obj attribute before every solve
                obj += (rho / 2.0) * q[k][t] * q[k][t] + 1e-4 * qpos[k][t]
                if data.get("NETWORK_CHARGE") is not None:  # half of γ_bj per kWh traded (Baroche et al. 2019)
                    neighbour = k if k < b else k + 1  # = admm_bilateral_p2p_valley.NB[b][k]
                    obj += data["NETWORK_CHARGE"][b][neighbour] / 2.0 * DT * (qpos[k][t] + qneg[k][t])
                if data.get("NETWORK_LOSSES") is not None:  # quadratic per-pair losses at the tariff, half per side
                    neighbour = k if k < b else k + 1
                    obj += data["NETWORK_LOSSES"][b][neighbour] / 2.0 * price[t] * DT * q[k][t] * q[k][t]
    q_offset = np.zeros((K, T))
    tracking = data.get("tracking")
    if tracking is not None and local is None:
        last = T - 1
        if tracking.get("mode", "trades") == "grid":
            # grid tracking (three_timescale.py --tracking grid): W_g dt (g - g_plan)^2, trades free to
            # re-route, and the end-of-horizon state valued at the hourly solve's shadow prices:
            # - mu_soc s_end - mu_T T_end  (plus the small quadratics below)
            w_grid, grid_plan = tracking["grid_weight"], float(tracking["p_el"][b])
            for t in ts:
                obj += w_grid * DT * (p_el[t] - grid_plan) * (p_el[t] - grid_plan)
            obj += -float(tracking["soc_value"][b]) * soc[last] - float(tracking["temperature_value"][b]) * T_in[last]
        else:
            # sharing-plan tracking (08_minute_resolution/three_timescale.py): quadratic penalty on
            # departing from the plan's P2P trades, W dt (q - q_plan)^2, and on missing its end-of-horizon state.
            # The trades' linear part (-2 w dt q_plan) goes with lambda - rho z into the Obj attribute (see _solve).
            w = tracking["weight"]
            for t in ts:
                for k in ks:
                    obj += w * DT * q[k][t] * q[k][t]
                    q_offset[k, t] = -2.0 * w * DT * float(tracking["q"][b][k])
        obj += tracking["soc_weight"] * (soc[last] - float(tracking["soc"][b])) * (soc[last] - float(tracking["soc"][b]))
        obj += (tracking["temperature_weight"] * (T_in[last] - float(tracking["T_in"][b]))
                * (T_in[last] - float(tracking["T_in"][b])))
        ramp = tracking.get("ramp")
        if ramp is not None and ramp.get("mode") == "grid":
            # ramp limit on the building's grid import: E = grid energy per step, up/down within delta
            energy = [DT * p_el[t] for t in ts]
            previous, delta = float(ramp["previous"][b]), float(ramp["delta"][b])
            for t in ts:
                before = energy[t - 1] if t > 0 else previous
                up, down = m.addVar(lb=0.0), m.addVar(lb=0.0)
                m.addConstr(up >= energy[t] - before - delta)
                m.addConstr(down >= before - energy[t] - delta)
                if ramp.get("quadratic", True):
                    obj += ramp["ramp_weight"] / DT * (up * up + down * down)
                if ramp.get("linear_weight", 0.0) > 0:  # proportional (exact) penalty, GBP per kWh of excess
                    obj += ramp["linear_weight"] * (up + down)
        elif ramp is not None:
            # (6a)-(6c) as quadratic penalties on their excess (08_minute_resolution/ramp_limits)
            if ramp["producer"][b]:
                # producers: E = energy sold P2P per step; ramp up/down within delta per step
                energy = [DT * gp.quicksum(qpos[k][t] for k in ks) for t in ts]
                previous, delta = float(ramp["previous"][b]), float(ramp["delta"][b])
                for t in ts:
                    before = energy[t - 1] if t > 0 else previous
                    up, down = m.addVar(lb=0.0), m.addVar(lb=0.0)
                    m.addConstr(up >= energy[t] - before - delta)
                    m.addConstr(down >= before - energy[t] - delta)
                    obj += ramp["ramp_weight"] / DT * (up * up + down * down)  # = w DT (excess / DT)^2
            else:
                # consumers: E = energy bought (grid + P2P) per step; cumulative within the cap
                energy = [DT * (p_el[t] + gp.quicksum(qneg[k][t] for k in ks)) for t in ts]
                cumulative = gp.LinExpr()
                for t in ts:
                    cumulative += energy[t]
                    excess = m.addVar(lb=0.0)
                    m.addConstr(excess >= cumulative - float(ramp["cap"][b][t]))
                    obj += ramp["cumulative_weight"] * excess * excess
    if local is not None:
        # simulate_realtime_reoptimise._solve_local_building: planned trades are upper bounds
        # (same direction, never more), P2P settlement, and a charge on planned P2P not delivered
        planned, share_penalty = local["planned"], local["share_penalty"]
        for k in ks:
            for t in ts:
                qpos[k][t].ub = max(float(planned[k, t]), 0.0)
                qneg[k][t].ub = max(float(-planned[k, t]), 0.0)
                obj += data["FEE_FRAC"] * price[t] * DT * (qneg[k][t] - qpos[k][t])
                obj += share_penalty * DT * (abs(float(planned[k, t])) - qpos[k][t] - qneg[k][t])
    m.setObjective(obj, GRB.MINIMIZE)
    m.update()
    names = ("charge", "discharge", "soc", "cs", "p_el", "curtail", "p_hp", "f", "q_boiler", "gas", "T_in")
    variables = dict(zip(names, (charge, discharge, soc, cs, p_el, curtail, p_hp, f, q_boiler, gas, T_in)))
    variables.update(q=q, qpos=qpos, qneg=qneg)
    return {"model": m, "vars": variables, "q_flat": [var for row in q for var in row], "shape": (K, T),
            "q_offset": q_offset, "links": links}


def _state_values(entry):
    """Value (GBP per unit, + = worth having) of the battery charge and indoor temperature at the end
    of the first step, from the duals of the dynamics linking step 0 to step 1 (convex subproblem).

    Raising the state at the end of step 0 by d shifts row 1 by -a d, where a is the state's coefficient
    in that row; the objective changes by Pi (-a d), so the value is Pi * a. 0 for a one-step horizon.
    """
    values = []
    for name in ("soc", "T_in"):
        row = entry["links"][name][1] if len(entry["links"][name]) > 1 else None
        if row is None:
            values.append(0.0)
            continue
        coefficient = entry["model"].getCoeff(row, entry["vars"][name][0])
        values.append(float(row.Pi) * float(coefficient))
    return tuple(values)


def _values(entry, name):
    variables = entry["vars"][name]
    if variables and isinstance(variables[0], list):
        flat = [var for row in variables for var in row]
        return np.array(entry["model"].getAttr("X", flat)).reshape(len(variables), -1)
    return np.array(entry["model"].getAttr("X", variables))


def _operating_cost(data, entry):
    DT = data["DT"]
    return float(
        np.sum(data["PRICE"] * _values(entry, "p_el") * DT)
        + np.sum(data["gas_price"] / 100.0 * _values(entry, "gas") * DT)
        + np.sum(data["alpha"] * (_values(entry, "T_in") - data["T_SET"]) ** 2)
    )


class _WorkerError:
    """Traceback from a worker, re-raised by the driver."""

    def __init__(self, text):
        self.text = text


VAR_NAMES = ("charge", "discharge", "soc", "cs", "p_el", "curtail", "p_hp", "f", "q_boiler", "gas", "T_in",
             "q", "qpos", "qneg")


def _solve(models, data, buildings, rho, lam, zloc, env=None, cache=None):
    """Solve the buildings' subproblems for the given duals and agreed trades.

    With cache (low-memory mode) each model is built, solved and disposed of, and
    only its solution values are kept in cache; the timing then includes the build.
    """
    out = {}
    for b in buildings:
        started = time.perf_counter()
        entry = models[b] if cache is None else _build(data, b, env)
        entry["model"].setAttr("Obj", entry["q_flat"],
                               (lam[b] - rho * zloc[b] + entry.get("q_offset", 0.0)).ravel().tolist())
        entry["model"].optimize()
        seconds = time.perf_counter() - started
        status = entry["model"].Status
        if status != GRB.OPTIMAL:
            out[b] = ("error", status)
            continue
        q = np.array(entry["model"].getAttr("X", entry["q_flat"])).reshape(entry["shape"])
        out[b] = (q, _operating_cost(data, entry), seconds)
        if cache is not None:
            cache[b] = {name: _values(entry, name) for name in VAR_NAMES}
            entry["model"].dispose()
    return out


def _solve_local(data, buildings, planned, share_penalty, first_step, env):
    """Build and solve each building's local re-optimisation; timed like the Pyomo version (build + solve)."""
    out = {}
    for b in buildings:
        started = time.perf_counter()
        entry = _build(data, b, env, local={"planned": planned[b], "share_penalty": share_penalty})
        build_seconds = time.perf_counter() - started
        entry["model"].optimize()
        seconds = time.perf_counter() - started
        if entry["model"].Status != GRB.OPTIMAL:
            raise RuntimeError(f"local solve failed for building {b}: Gurobi status {entry['model'].Status}")
        values = {name: float(_values(entry, name)[0]) for name in first_step}
        values["offer"] = _values(entry, "qpos")[:, 0]  # per neighbour index k
        values["accept"] = _values(entry, "qneg")[:, 0]
        values.update(build_seconds=build_seconds, solver_seconds=seconds - build_seconds, wall_seconds=seconds)
        out[b] = values
    return out


def _worker(conn, buildings):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1)
    env.start()
    models, data, cache = {}, None, None
    while True:
        command, payload = conn.recv()
        if command == "stop":
            break
        try:
            if command == "build":
                data, lean = payload
                for entry in models.values():
                    entry["model"].dispose()
                models, cache = ({}, {}) if lean else ({b: _build(data, b, env) for b in buildings}, None)
                reply = None
            elif command == "solve":
                reply = _solve(models, data, buildings, *payload, env=env, cache=cache)
            elif command == "extract":
                if cache is not None:
                    reply = {b: {name: cache[b][name] for name in payload} for b in buildings}
                else:
                    reply = {b: {name: _values(models[b], name) for name in payload} for b in buildings}
            elif command == "state_values":
                reply = {b: _state_values(models[b]) for b in buildings}
            elif command == "local":
                data, planned, share_penalty, first_step = payload
                reply = _solve_local(data, buildings, planned, share_penalty, first_step, env)
            else:
                raise ValueError(f"unknown command {command!r}")
        except Exception:  # noqa: BLE001 - re-raised in the driver
            reply = _WorkerError(traceback.format_exc())
        conn.send(reply)


# ============================================================================
# DRIVER SIDE
# ============================================================================
EXTRA_DATA = {}  # merged into _problem_data(); e.g. {"tracking": ...} from 08_minute_resolution/three_timescale.py


def _problem_data():
    """Snapshot of the optimiser's module state (honours reopt.remaining_horizon), plus EXTRA_DATA."""
    import admm_bilateral_p2p_valley as OPT

    C = OPT.C
    return {
        "N": OPT.N, "T": OPT.T, "DT": OPT.DT, "RHO": float(OPT.RHO),
        "LOAD": np.asarray(C.LOAD, float), "PV_B": np.asarray(OPT.PV_B, float),
        "PRICE": np.asarray(OPT.PRICE, float), "COP": np.asarray(OPT.COP, float),
        "T_OUT": np.asarray(C.T_OUT, float), "T_SET": np.asarray(C.T_SET, float),
        "congestion_weight": np.asarray(OPT.congestion_weight, float), "use_congestion": bool(OPT.use_congestion),
        "INIT_SOC": np.asarray(C.INIT_SOC, float), "INIT_T_IN": np.asarray(C.INIT_T_IN, float),
        "CAP": np.asarray(C.CAP, float), "MAXPOW": np.asarray(C.MAXPOW, float),
        "HAS_BATTERY": np.asarray(OPT.HAS_BATTERY, bool), "ROLLING_FIRST_INTERVAL": bool(OPT.ROLLING_FIRST_INTERVAL),
        "RELAX_BINARIES": bool(OPT.RELAX_BINARIES),
        "eta_charge": C.eta_charge, "eta_discharge": C.eta_discharge, "p_th_nom": C.p_th_nom,
        "efficiency": C.efficiency, "hp_max_power": C.hp_max_power, "max_thermal_power": C.max_thermal_power,
        "gas_price": C.gas_price, "alpha": C.alpha, "FEE_FRAC": float(OPT.FEE_FRAC),
        "NETWORK_CHARGE": OPT.network_charge_matrix()[0] if OPT.NETWORK_CHARGE else None,
        "NETWORK_LOSSES": OPT.network_loss_matrix() if OPT.NETWORK_LOSSES else None,
        **EXTRA_DATA,
    }


MODEL_MB_PER_NEIGHBOUR = 0.05  # gurobipy subproblem memory per trading partner (measured: ~49 MB at N=1000)
LOW_MEMORY_SHARE = 0.4  # keep persistent models only if they need less than this share of RAM


def needs_low_memory(n_buildings):
    """True when N persistent subproblem models would not fit comfortably in memory."""
    import psutil

    needed = MODEL_MB_PER_NEIGHBOUR * (n_buildings - 1) * n_buildings * 2**20
    return needed > LOW_MEMORY_SHARE * psutil.virtual_memory().total


class ParallelADMM:
    """Persistent worker pool; buildings are dealt round-robin to the workers."""

    def __init__(self, n_buildings, workers=DEFAULT_WORKERS):
        workers = max(1, min(workers, n_buildings))
        context = mp.get_context("spawn")
        self.assignment = [list(range(w, n_buildings, workers)) for w in range(workers)]
        self.pipes, self.processes = [], []
        for buildings in self.assignment:
            parent, child = context.Pipe()
            process = context.Process(target=_worker, args=(child, buildings), daemon=True)
            process.start()
            self.pipes.append(parent)
            self.processes.append(process)

    def _broadcast(self, command, payload):
        return self._scatter(command, lambda _buildings: payload)

    def _scatter(self, command, payload_for):
        """Send each worker payload_for(its buildings), so per-building arrays go only where they are used."""
        for pipe, buildings in zip(self.pipes, self.assignment):
            pipe.send((command, payload_for(buildings)))
        replies = [pipe.recv() for pipe in self.pipes]
        for reply in replies:
            if isinstance(reply, _WorkerError):
                raise RuntimeError(f"ADMM worker failed on {command!r}:\n{reply.text}")
        return replies

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(("stop", None))
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def run(self, init_lam=None, init_Z=None, solve_log=None, iteration_log=None, verbose=True, deadline=None):
        """Same contract as admm_bilateral_p2p_valley.run_admm, but returns trades only.

        Returns (q_all, lam_all, Z, hist); call .extract() for the schedules.
        iteration_log, if given, receives the measured wall seconds of each iteration.
        """
        import admm_bilateral_p2p_valley as OPT
        import simulate_realtime_reoptimise as reopt

        data = _problem_data()
        N, T = data["N"], data["T"]
        self.lean = needs_low_memory(N)
        self._broadcast("build", (data, self.lean))

        lam_all = [np.zeros((N - 1, T)) for _ in range(N)]
        Z = {p: np.zeros(T) for p in OPT.PAIRS}
        zloc_all = [np.zeros((N - 1, T)) for _ in range(N)]
        if init_lam is not None:
            lam_all = [np.array(lam, dtype=float).copy() for lam in init_lam]
        if init_Z is not None:
            Z = {p: np.array(init_Z[p], dtype=float).copy() for p in OPT.PAIRS}
            zloc_all = OPT._zloc_from_Z(Z)
        q_all = [np.zeros((N - 1, T)) for _ in range(N)]
        rho = float(OPT.RHO)
        streak, hist = 0, []
        npair_t, ncoup_t = len(OPT.PAIRS) * T, N * (N - 1) * T
        for i in range(OPT.MAX_ITERS):
            started = time.perf_counter()
            results = {}
            replies = self._scatter("solve", lambda bs: (rho, {b: lam_all[b] for b in bs}, {b: zloc_all[b] for b in bs}))
            for chunk in replies:
                results.update(chunk)
            if iteration_log is not None:
                iteration_log.append(time.perf_counter() - started)
            operating_cost = 0.0
            for b in range(N):
                if isinstance(results[b][0], str):
                    raise RuntimeError(f"building {b} subproblem failed at ADMM iter {i}: Gurobi status {results[b][1]}")
                q_all[b], cost, seconds = results[b]
                operating_cost += cost
                if solve_log is not None:
                    solve_log.append((i, b, seconds))

            Z_new = OPT.z_update(q_all)
            zloc_new = OPT._zloc_from_Z(Z_new)
            lam_new = [lam_all[b] + rho * (q_all[b] - zloc_new[b]) for b in range(N)]
            prim = sum(float(np.sum((rho * (q_all[b] - zloc_new[b])) ** 2)) for b in range(N))
            dual = rho * sum(float(np.sum((Z_new[p] - Z[p]) ** 2)) for p in OPT.PAIRS)
            r_rms = float(np.sqrt(sum(float(np.sum((q_all[b] - zloc_new[b]) ** 2)) for b in range(N)) / ncoup_t))
            s_rms = float(np.sqrt(sum(float(np.sum((Z_new[p] - Z[p]) ** 2)) for p in OPT.PAIRS) / npair_t))
            traded = sum(float(np.sum(np.abs(Z_new[p]))) for p in OPT.PAIRS)
            z_norm = float(np.sqrt(sum(float(np.sum(Z_new[p] ** 2)) for p in OPT.PAIRS)))
            lambda_norm = float(np.sqrt(sum(float(np.sum(lam_new[b] ** 2)) for b in range(N))))
            hist.append((i, prim, dual, r_rms, s_rms, traded, operating_cost, z_norm, lambda_norm))
            if verbose and i % 10 == 0:
                print(f"  iter {i:3d}  primal(RMS) {r_rms:.7f} kW   dual(RMS) {s_rms:.7f}   "
                      f"traded {traded:6.1f} kWh   {time.perf_counter() - started:5.2f}s", flush=True)
            eps1 = (rho * OPT.EPS_PRIMAL) ** 2 * ncoup_t
            eps2 = rho * OPT.EPS_DUAL**2 * npair_t
            streak = streak + 1 if (prim <= eps1 and dual <= eps2) else 0
            Z, zloc_all, lam_all = Z_new, zloc_new, lam_new
            if streak >= OPT.STOP_STREAK:
                if verbose:
                    print(f"  converged at iter {i}", flush=True)
                break
            reopt.check_deadline(deadline, f"parallel ADMM iteration {i} did not converge")
        else:
            if verbose:
                print(f"  stopped at MAX_ITERS ({OPT.MAX_ITERS})", flush=True)
        return q_all, lam_all, Z, np.array(hist)

    def extract(self, names):
        out = {}
        for chunk in self._broadcast("extract", list(names)):
            out.update(chunk)
        return [out[b] for b in sorted(out)]

    def state_values(self):
        """(N, 2): each building's value of battery charge (GBP/kWh) and indoor temperature (GBP/degC)
        at the end of the first step, from the last ADMM iteration's subproblem duals (see _state_values)."""
        out = {}
        for chunk in self._broadcast("state_values", None):
            out.update(chunk)
        return np.array([out[b] for b in sorted(out)], dtype=float)

    def local_step(self, planned, share_penalty, first_step):
        """One local re-optimisation per building over the current horizon (no ADMM).

        planned[b] is building b's (N-1, T) planned trades; returns a list of
        per-building dicts with the first-step values, offer/accept per neighbour
        index and build/solve timings.
        """
        out = {}
        data = _problem_data()
        replies = self._scatter("local", lambda bs: (data, {b: planned[b] for b in bs}, share_penalty, list(first_step)))
        for chunk in replies:
            out.update(chunk)
        return [out[b] for b in sorted(out)]


# ============================================================================
# DAY-AHEAD SCHEDULE AND ROLLING SIMULATION
# ============================================================================
def load_cached_schedule(cache_file):
    """The cached day-ahead schedule if it matches the current problem, else None."""
    import simulate_noisy_realtime as baseline

    if not cache_file.exists():
        return None
    OPT = baseline.optimiser

    # load_or_solve_schedule falls back to the sequential run_admm when the cache
    # is stale; stub it out so that case returns None instead
    def _no_cache(**_):
        raise LookupError("no compatible cache")

    original, OPT.run_admm = OPT.run_admm, _no_cache
    try:
        return baseline.load_or_solve_schedule(cache_file)
    except LookupError:
        return None
    finally:
        OPT.run_admm = original


def solve_schedule_parallel(cache_file, *, workers=DEFAULT_WORKERS, deadline=None, force_refresh=False, verbose=True):
    """Day-ahead ADMM in parallel, cached exactly like simulate_noisy_realtime.load_or_solve_schedule.

    A compatible cache is loaded as is (0 iterations returned); the signature is
    the same, so either script can reuse the other's schedule.
    """
    import simulate_noisy_realtime as baseline

    OPT = baseline.optimiser
    if not force_refresh:
        cached = load_cached_schedule(cache_file)
        if cached is not None:
            return cached, 0

    with ParallelADMM(OPT.N, workers) as pool:
        _q, _lam, Z, hist = pool.run(verbose=verbose, deadline=deadline)
        solved = pool.extract(baseline._SCHEDULE_VARIABLES + baseline._TRADE_VARIABLES)
    N, T = OPT.N, OPT.T
    variables = {name: np.array([solved[b][name] for b in range(N)], dtype=float) for name in baseline._SCHEDULE_VARIABLES}
    trade_variables = {name: np.zeros((N, N, T)) for name in baseline._TRADE_VARIABLES}
    for b in range(N):
        for k, neighbour in enumerate(OPT.NB[b]):
            for name in baseline._TRADE_VARIABLES:
                trade_variables[name][b, neighbour] = solved[b][name][k]
    trades = np.zeros((N, N, T))
    for lo, hi in OPT.PAIRS:
        trades[lo, hi] = Z[(lo, hi)]
        trades[hi, lo] = -Z[(lo, hi)]
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        signature=np.array(baseline._schedule_signature()),
        **{f"variable_{name}": values for name, values in variables.items()},
        **{f"trade_variable_{name}": values for name, values in trade_variables.items()},
        trades=trades,
    )
    print(f"Cached optimisation schedule: {cache_file} ({len(hist)} parallel ADMM iterations)", flush=True)
    return {"variables": variables, "trade_variables": trade_variables, "trades": trades}, len(hist)


def simulate_rolling_admm_parallel(schedule, actual_load, actual_pv, actual_t_out, *, run, workers=DEFAULT_WORKERS,
                                   warm_start=True, future_inputs="noisy", results_file=None, timing_file=None,
                                   method=METHOD, deadline=None):
    import simulate_realtime_reoptimise as reopt

    OPT, N, FULL_T = reopt.OPT, reopt.N, reopt.FULL_T
    first_step = reopt._FIRST_STEP_NAMES
    soc = reopt.DAY_START_SOC.copy()
    temperature = reopt.DAY_START_T_IN.copy()
    previous_lam = None
    previous_Z = {(lo, hi): schedule["trades"][lo, hi].copy() for lo, hi in OPT.PAIRS} if warm_start else None
    records, timings = [], []
    with ParallelADMM(N, workers) as pool:
        for interval in range(FULL_T):
            reopt.check_deadline(deadline, f"parallel rolling ADMM before interval {interval}")
            solve_log, iteration_wall = [], []
            init_lam = init_Z = None
            if warm_start:
                shift = 0 if interval == 0 else 1
                init_Z = {pair: values[shift:] for pair, values in previous_Z.items()}
                if previous_lam is not None:
                    init_lam = [lam[:, shift:] for lam in previous_lam]
            started = time.perf_counter()
            inputs = reopt.horizon_inputs(interval, actual_load, actual_pv, actual_t_out, future_inputs)
            with reopt.remaining_horizon(interval, *inputs, soc, temperature):
                _q, lam_all, Z, hist = pool.run(init_lam=init_lam, init_Z=init_Z, solve_log=solve_log,
                                                iteration_log=iteration_wall, verbose=False, deadline=deadline)
                solved = pool.extract(first_step)
            wall = time.perf_counter() - started
            outputs = [{name: float(solved[b][name][0]) for name in first_step} for b in range(N)]
            previous_lam, previous_Z = lam_all, Z

            flows = np.zeros((N, N))
            for lo, hi in OPT.PAIRS:
                z = float(Z[(lo, hi)][0])
                if z >= 0:
                    flows[lo, hi] = z
                else:
                    flows[hi, lo] = -z
            for b, out in enumerate(outputs):
                out["p2p_export"] = float(flows[b].sum())
                out["p2p_import"] = float(flows[:, b].sum())
                soc[b] = out["soc"]
                temperature[b] = out["T_in"]

            log = np.array(solve_log, dtype=float).reshape(-1, 3)
            iterations = int(len(hist))
            per_iteration_max = [float(log[log[:, 0] == i, 2].max()) for i in range(iterations)]
            step_timings = [
                reopt.timing_row(run, method, interval, "admm_iteration", iteration=i,
                                 wall_seconds=iteration_wall[i], solver_seconds=float(log[log[:, 0] == i, 2].sum()),
                                 parallel_latency_seconds=per_iteration_max[i], solver_calls=N)
                for i in range(iterations)
            ]
            solver_total = float(log[:, 2].sum())
            step_timings.append(reopt.timing_row(
                run, method, interval, "community",
                wall_seconds=wall,
                build_seconds=max(wall - sum(iteration_wall), 0.0),
                solver_seconds=solver_total,
                parallel_latency_seconds=sum(per_iteration_max),
                solver_calls=len(log),
                admm_iterations=iterations,
                termination=f"{'warm' if warm_start else 'cold'}_start_{len(pool.assignment)}_workers",
            ))
            step_records = [
                reopt.make_record(run, method, interval, b, actual_load[b, interval], actual_pv[b, interval],
                                  actual_t_out[interval], outputs[b], schedule)
                for b in range(N)
            ]
            if results_file is not None:
                reopt.append_rows(results_file, step_records)
            if timing_file is not None:
                reopt.append_rows(timing_file, step_timings)
            records.extend(step_records)
            timings.extend(step_timings)
            print(
                f"  run {run} ADMM-par interval {interval + 1:02d}/{FULL_T}: {iterations:4d} iters, "
                f"wall {wall:7.2f}s ({len(pool.assignment)} workers), per-building parallel {sum(per_iteration_max):6.2f}s, "
                f"grid {sum(o['p_el'] for o in outputs):6.2f} kW",
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
    parser.add_argument("--admm-start", choices=("warm", "cold"), default="warm")
    parser.add_argument("--schedule-only", action="store_true", help="Only solve (or load) the day-ahead schedule.")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    reopt.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    schedule, iterations = solve_schedule_parallel(reopt.SCHEDULE_CACHE, workers=args.workers,
                                                   force_refresh=args.refresh_cache)
    print(f"Day-ahead schedule ready in {time.perf_counter() - started:.1f}s ({iterations} iterations; 0 = cached)")
    if args.schedule_only:
        return
    loads, pvs, t_outs = baseline.load_or_create_scenarios(
        reopt.SCENARIO_FILE, runs=args.runs, load_noise=args.load_noise, pv_noise=args.pv_noise,
        temp_noise=args.temp_noise, seed=args.seed,
    )
    results_file = reopt.RESULTS_DIR / "parallel_admm_realtime_simulation.csv"
    timing_file = reopt.RESULTS_DIR / "parallel_admm_realtime_solve_times.csv"
    for path in (results_file, timing_file):
        path.write_text("", encoding="utf-8")
    for run in range(1, args.runs + 1):
        simulate_rolling_admm_parallel(schedule, loads[run - 1], pvs[run - 1], t_outs[run - 1], run=run,
                                       workers=args.workers, warm_start=args.admm_start == "warm",
                                       future_inputs=args.future_inputs,
                                       results_file=results_file, timing_file=timing_file)
    print(f"Results saved to: {results_file}\nSolve times saved to: {timing_file}")


if __name__ == "__main__":
    main()
