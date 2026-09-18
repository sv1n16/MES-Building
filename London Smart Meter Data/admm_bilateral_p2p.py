"""Fully-decentralised consensus-ADMM for peer-to-peer energy sharing.

Reimplementation of the algorithm in

    Y. Wang, L. Wu, S. Wang, "A Fully-Decentralized Consensus-Based ADMM
    Approach for DC-OPF With Demand Response," IEEE Trans. Smart Grid 8(6),
    2017, pp. 2637-2647.

mapped onto this project's building-community use case. The paper decomposes a
network by *tie lines*: each pair of adjacent subsystems shares a boundary
variable (a bus angle), duplicated in both subsystems and reconciled through a
global consensus variable owned by a "leading" subsystem. Here the analogue of a
tie line is a **bilateral P2P trade**: for every pair of buildings {b,j} and hour
t there is a trade  Z[{b,j},t]  (kW, + = b->j). Building b keeps its own copy
q_b[j,t]; building j keeps q_j[b,t] (its convention: + = j->b). Consensus:

    q_b[j,t] = Z[{b,j},t]        q_j[b,t] = -Z[{b,j},t]                      (30)

Each building solves only its own subproblem (its DER schedule + its trade
copies); global variables are the pairwise agreed trades. This implements the
paper's **Algorithm 2** (fully decentralised): for every pair {b,j} the leading
building min(b,j) receives the single copy q_j[b,t] from its neighbour, computes
Z locally and returns it -- only neighbour-to-neighbour messages, no coordinator.
Paper eqs (31)-(36):

    x-update (32)   min_x  C_b(x_b) + Σ λ_b[j,t]·q_b[j,t]
                                    + (ρ/2) Σ (q_b[j,t] - z_b[j,t])²   s.t. χ_b
    z-update (33)   Z[{b,j},t] = ( q_b[j,t] - q_j[b,t] ) / 2      (avg of copies)
    λ-update (34)   λ_b[j,t] += ρ ( q_b[j,t] - z_b[j,t] )
    stop     (36)   ‖λ^{i+1}-λ^i‖² ≤ ε₁ ,  ρ‖z^{i+1}-z^i‖² ≤ ε₂

The paper's Algorithm 1 (identical arithmetic, but a coordinator does every
average) and Algorithm 3 (Algorithm 2 + Nesterov extrapolation, eqs 37-39) are
not implemented here.

The community DER model (battery, fixed-COP heat pump, gas boiler, thermal,
per-building PV, per-building battery presence) is imported from
central_optimisation_showcase. With `RELAX_BINARIES = False` (default) the
charge/discharge indicator stays Binary, so subproblems are MIQPs matching the
central model and the ADMM is non-convex (no convergence guarantee). Set
`RELAX_BINARIES = True` for convex-QP subproblems and the paper's guaranteed
global convergence.

Outputs: plots/14_admm_bilateral.{png,html} and a console comparison with the
centralised market solution.
"""

from __future__ import annotations

import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt

import central_optimisation_showcase as C
from plot_timeseries import INK, MUTED

# ---------------------------------------------------------------- ADMM settings
RHO = 1  # tuned at N=10 (ρ sweep 0.1–32, scratchpad/bilateral_rho_sweep*.py):
#           iterations-to-converge bottoms at ρ≈8–16 (8 iters vs 19 at ρ=1), and ρ≥16
#           also best matches central's minimal-routing solution (per-building cost RMS
#           5.0% vs 7.5% mid-range; traded volume 8.6 vs 15.6 kWh). Identical result with
#           RELAX_BINARIES True or False (the charge/discharge binary is slack on the
#           showcase day). Re-sweep if N changes.
MAX_ITERS = 400
EPS_PRIMAL = 1e-3  # eq (36) ε₁ as an RMS consensus violation (kW); ‖Δλ‖² ≤ (ρ·EPS_PRIMAL)²·N(N-1)T
EPS_DUAL = 1e-3  # eq (36) ε₂ as an RMS change in the agreed trades (kW); ρ‖Δz‖² ≤ ρ·EPS_DUAL²·|PAIRS|·T
ADAPT_RHO = False  # residual balancing (Boyd §3.4.1)
STOP_STREAK = 2  # need this many consecutive iters below tol
RELAX_BINARIES = True  # False -> exact Binary charge/discharge indicator (MIQP subproblems,
#                         matching central_optimisation_showcase); ADMM is then non-convex and
#                         convergence is no longer guaranteed (may oscillate / need rho tuning).
#                         True -> relax cs to [0,1]: convex QP subproblems, provable convergence.
FEE_FRAC = 0.5  # P2P settlement price = FEE_FRAC × grid tariff, i.e. central's fee_mode="market"

N, T, DT = C.n_buildings, C.time_horizon, C.dt
COP = C.cop_base + 0.01 * (C.T_OUT - C.T_ref)
PV_B, HAS_BATTERY, PRICE = C.PV_B, C.HAS_BATTERY, C.PRICE

NB = [[j for j in range(N) if j != b] for b in range(N)]  # neighbours of b
PAIRS = [(a, b) for a in range(N) for b in range(a + 1, N)]  # unordered pairs
PRICE_HIGH_COLOR, PRICE_LOW_COLOR, ADMM_COLOR = "#d98a29", "#5b8fc9", "#2e8b6e"
PRICE_MEDIUM_COLOR = "#8b78b5"


# ============================================================================
# BUILDING SUBPROBLEM  (local DER schedule + local copies of its trades)
# ============================================================================
def build_subproblem(b: int) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel()
    m.t = pyo.RangeSet(0, T - 1)
    m.k = pyo.RangeSet(0, N - 2)  # neighbour index (into NB[b])
    dom = pyo.NonNegativeReals if RELAX_BINARIES else pyo.Binary
    pv_b = PV_B[b]

    m.charge = pyo.Var(m.t, bounds=(0, C.MAXPOW[b]), initialize=0)
    m.discharge = pyo.Var(m.t, bounds=(0, C.MAXPOW[b]), initialize=0)
    m.soc = pyo.Var(m.t, bounds=(0, max(C.CAP[b], 1e-6)), initialize=float(C.INIT_SOC[b]))
    m.cs = pyo.Var(m.t, domain=dom, bounds=(0, 1))
    m.p_el = pyo.Var(m.t, bounds=(0, None), initialize=0)
    m.curtail = pyo.Var(m.t, bounds=(0, None), initialize=0)
    m.p_hp = pyo.Var(m.t, bounds=(0, C.hp_max_power), initialize=0)
    m.f = pyo.Var(m.t, bounds=(0, 1), initialize=0)
    m.q_boiler = pyo.Var(m.t, bounds=(0, C.max_thermal_power), initialize=0)
    m.gas = pyo.Var(m.t, domain=pyo.NonNegativeReals, initialize=0)
    m.T_in = pyo.Var(m.t, bounds=(0, None), initialize=C.T_init)
    m.q = pyo.Var(m.k, m.t, initialize=0)  # b's copy of trade with NB[b][k] (+ = b delivers)
    m.qpos = pyo.Var(m.k, m.t, bounds=(0, None), initialize=0)  # gross delivered to NB[b][k]
    m.qneg = pyo.Var(m.k, m.t, bounds=(0, None), initialize=0)  # gross received from NB[b][k]

    m.rho = pyo.Param(initialize=RHO, mutable=True)
    m.lam = pyo.Param(m.k, m.t, initialize=0.0, mutable=True)  # dual on the consensus
    m.zloc = pyo.Param(m.k, m.t, initialize=0.0, mutable=True)  # agreed trade, b's sign

    # --- battery ---
    m.c_cmax = pyo.Constraint(m.t, rule=lambda mm, t: mm.charge[t] <= mm.cs[t] * C.MAXPOW[b])
    m.c_dmax = pyo.Constraint(m.t, rule=lambda mm, t: mm.discharge[t] <= (1 - mm.cs[t]) * C.MAXPOW[b])

    def soc_bal(mm, t):
        if t == 0:
            return mm.soc[t] == float(C.INIT_SOC[b])
        return (
            mm.soc[t] == mm.soc[t - 1] + (C.eta_charge * mm.charge[t] - (1.0 / C.eta_discharge) * mm.discharge[t]) * DT
        )

    m.c_soc = pyo.Constraint(m.t, rule=soc_bal)
    m.c_nc0 = pyo.Constraint(rule=lambda mm: mm.charge[0] == 0)
    m.c_nd0 = pyo.Constraint(rule=lambda mm: mm.discharge[0] == 0)
    if not HAS_BATTERY[b]:
        for t in range(T):
            m.charge[t].fix(0.0)
            m.discharge[t].fix(0.0)
            m.cs[t].fix(0)

    # --- heat pump / boiler (fixed COP -> linear) ---
    m.c_hp = pyo.Constraint(m.t, rule=lambda mm, t: COP[t] * mm.p_hp[t] == C.p_th_nom * mm.f[t])
    m.c_gas = pyo.Constraint(m.t, rule=lambda mm, t: mm.gas[t] == mm.q_boiler[t] / C.efficiency)

    def q_heat(mm, t):
        return C.p_th_nom * mm.f[t] + mm.q_boiler[t]

    def thermal(mm, t):
        if t == 0:
            return mm.T_in[t] == C.T_init
        return mm.T_in[t] == mm.T_in[t - 1] + DT / 10 * (q_heat(mm, t) - 0.5 * (mm.T_in[t - 1] - C.T_OUT[t]))

    m.c_therm = pyo.Constraint(m.t, rule=thermal)

    # --- electricity balance & per-hour trade limits ---
    def qnet(mm, t):
        return sum(mm.q[k, t] for k in mm.k)

    # split each signed copy into gross delivered / gross received so the export
    # and import ceilings bind on GROSS flow (matching central_optimisation_showcase),
    # not on the net -- otherwise b could relay: import from one neighbour and
    # re-export to another beyond its own generation / consumption.
    m.c_qsplit = pyo.Constraint(m.k, m.t, rule=lambda mm, k, t: mm.q[k, t] == mm.qpos[k, t] - mm.qneg[k, t])

    m.c_bal = pyo.Constraint(
        m.t,
        rule=lambda mm, t: mm.p_el[t] - qnet(mm, t) - mm.curtail[t]
        == C.LOAD[b, t] + mm.charge[t] + mm.p_hp[t] - mm.discharge[t] - pv_b[t],
    )
    m.c_exp = pyo.Constraint(m.t, rule=lambda mm, t: sum(mm.qpos[k, t] for k in mm.k) <= pv_b[t] + mm.discharge[t])
    m.c_imp = pyo.Constraint(
        m.t, rule=lambda mm, t: sum(mm.qneg[k, t] for k in mm.k) <= C.LOAD[b, t] + mm.p_hp[t] + mm.charge[t]
    )

    def obj(mm):
        cost = sum(
            PRICE[t] * mm.p_el[t] * DT
            + C.gas_price / 100.0 * mm.gas[t] * DT
            + C.alpha * (mm.T_in[t] - C.T_SET[t]) ** 2
            for t in mm.t
        )
        lin = sum(mm.lam[k, t] * mm.q[k, t] for k in mm.k for t in mm.t)  # (32) λᵀx
        prox = (mm.rho / 2.0) * sum((mm.q[k, t] - mm.zloc[k, t]) ** 2 for k in mm.k for t in mm.t)
        # routing regulariser: identical to central's `routing_reg = 1e-4 * sent`.
        # It also pins the qpos/qneg split to the true parts of q.  The prox term
        # already makes the objective strictly convex in q, so no separate 1e-4*q²
        # tie-breaker is needed (and that term is a persistent bias absent from central).
        route = 1e-4 * sum(mm.qpos[k, t] for k in mm.k for t in mm.t)
        return cost + lin + prox + route

    m.obj = pyo.Objective(rule=obj, sense=pyo.minimize)
    return m


def _q(m) -> np.ndarray:
    return np.array([[pyo.value(m.q[k, t]) for t in range(T)] for k in range(N - 1)])


# ============================================================================
# GLOBAL-VARIABLE (z) UPDATE  -- paper eq (33)
# ============================================================================
def z_update(q_all: list[np.ndarray]) -> dict:
    """Consensus variable for every pair, exactly the paper's eq (33):

        Z_{lo,hi}[t] = 1/2 ( q_lo[hi,t]  +  E * q_hi[lo,t] )                 (33)

    the *plain average of the two local copies* (no multiplier term -- eq (33)
    carries none; the dual only enters via the eq (34) update).  Here the two
    buildings store the trade with opposite sign conventions (each: + = I
    deliver), so the copy-mapping E of eq (8) is the scalar -1:

        q_hi[lo,t]  ==  -q_lo[hi,t]   at consensus   ->   Z = 1/2 (q_lo - q_hi)

    Algorithm 2: the leading building min(lo,hi) evaluates this locally from its
    own copy q_lo[hi] and the single message q_hi[lo] sent by its neighbour.
    """
    Z = {}
    for lo, hi in PAIRS:
        q_lo_to_hi = q_all[lo][NB[lo].index(hi)]  # lo's copy   theta_{lo,hi}
        q_hi_to_lo = q_all[hi][NB[hi].index(lo)]  # hi's copy   theta_{hi,lo}  (E = -1)
        Z[(lo, hi)] = 0.5 * (q_lo_to_hi - q_hi_to_lo)
    return Z


def _zloc_from_Z(Z: dict) -> list[np.ndarray]:
    """Each building's signed copy of the agreed trades (b's convention: + = b delivers)."""
    zloc_all = [np.zeros((N - 1, T)) for _ in range(N)]
    for (lo, hi), z in Z.items():
        zloc_all[lo][NB[lo].index(hi)] = z
        zloc_all[hi][NB[hi].index(lo)] = -z
    return zloc_all


# ============================================================================
# ADMM LOOP
# ============================================================================
def run_admm():
    C._require_solver()
    solver = pyo.SolverFactory(C.SOLVER)
    _opts = [("LogFile", ""), ("OutputFlag", 0)]
    if C.SOLVE_TIME_LIMIT > 0:  # per-subproblem cap (env SHOWCASE_SOLVE_TIMELIMIT); matters for MIQP
        _opts.append(("TimeLimit", C.SOLVE_TIME_LIMIT))
    for opt, val in _opts:
        try:
            solver.options[opt] = val
        except Exception:
            pass

    subs = [build_subproblem(b) for b in range(N)]
    q_all = [np.zeros((N - 1, T)) for _ in range(N)]
    lam_all = [np.zeros((N - 1, T)) for _ in range(N)]
    Z = {p: np.zeros(T) for p in PAIRS}
    zloc_all = [np.zeros((N - 1, T)) for _ in range(N)]

    rho = float(RHO)
    streak = 0
    hist = []
    npair_t = len(PAIRS) * T
    ncoup_t = N * (N - 1) * T

    print(
        f"bilateral ADMM (Algorithm 2)  N={N}  pairs={len(PAIRS)}  T={T}  rho={RHO}  "
        f"relax_binaries={RELAX_BINARIES}"
    )
    for i in range(MAX_ITERS):
        # ---- (32) x-update: each building solves its own subproblem ----
        for b, m in enumerate(subs):
            m.rho = float(rho)
            for k in range(N - 1):
                for t in range(T):
                    m.lam[k, t] = float(lam_all[b][k, t])
                    m.zloc[k, t] = float(zloc_all[b][k, t])
            res = solver.solve(m, tee=False)
            tc = res.solver.termination_condition
            if tc not in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.locallyOptimal):
                raise RuntimeError(
                    f"building {b} subproblem did not solve to optimality at ADMM iter {i}: "
                    f"status={res.solver.status}, termination={tc}"
                )
            q_all[b] = _q(m)

        # ---- (33) z-update ----
        Z_new = z_update(q_all)
        zloc_new = _zloc_from_Z(Z_new)

        # ---- (34) λ-update ----
        lam_new = [lam_all[b] + rho * (q_all[b] - zloc_new[b]) for b in range(N)]

        # ---- residuals (eq 20/21/36) ----
        prim = sum(float(np.sum((rho * (q_all[b] - zloc_new[b])) ** 2)) for b in range(N))  # ‖Δλ‖²
        dual = rho * sum(float(np.sum((Z_new[p] - Z[p]) ** 2)) for p in PAIRS)  # ρ‖Δz‖²
        r_rms = float(np.sqrt(sum(float(np.sum((q_all[b] - zloc_new[b]) ** 2)) for b in range(N)) / ncoup_t))
        s_rms = float(np.sqrt(sum(float(np.sum((Z_new[p] - Z[p]) ** 2)) for p in PAIRS) / npair_t))
        traded = sum(float(np.sum(np.abs(Z_new[p]))) for p in PAIRS)
        hist.append((i, prim, dual, r_rms, s_rms, traded))
        if i % 10 == 0:
            print(
                f"  iter {i:3d}  primal(RMS) {r_rms:.4f} kW   dual(RMS) {s_rms:.4f}   "
                f"traded {traded:6.1f} kWh   rho {rho:.1f}"
            )

        # ---- stopping test: paper eq (36), ‖Δλ‖² ≤ ε₁ and ρ‖Δz‖² ≤ ε₂ ----
        # ε₁, ε₂ are derived from the interpretable RMS tolerances so they scale
        # with problem size and ρ.  These reduce exactly to  RMS(consensus
        # violation) ≤ EPS_PRIMAL  and  RMS(Δ agreed-trade) ≤ EPS_DUAL  (both kW),
        # since Δλ = ρ·r; keeping them in the eq-(36) form makes prim/dual the
        # quantities that actually gate termination.
        eps1 = (rho * EPS_PRIMAL) ** 2 * ncoup_t
        eps2 = rho * EPS_DUAL**2 * npair_t
        streak = streak + 1 if (prim <= eps1 and dual <= eps2) else 0

        # ---- commit this iterate as the new state (z/dual residuals used the old Z) ----
        Z, zloc_all, lam_all = Z_new, zloc_new, lam_new

        if streak >= STOP_STREAK:
            print(f"  converged at iter {i}  (primal {r_rms:.4f}, dual {s_rms:.4f})")
            break

        # ---- residual balancing (Boyd §3.4.1) ----
        if ADAPT_RHO:
            if prim > 100.0 * dual:
                rho *= 2.0
            elif dual > 100.0 * prim:
                rho *= 0.5
    else:
        print(f"  stopped at MAX_ITERS ({MAX_ITERS}); r_rms={r_rms:.4f} s_rms={s_rms:.4f}")

    return subs, q_all, lam_all, Z, np.array(hist)


# ============================================================================
# COMPARE + PLOT
# ============================================================================
def _pair_price(lam_all: list[np.ndarray]) -> dict:
    """DIAGNOSTIC only: the emergent shadow price of each pair's consensus
    (mean |dual| of the two sides, £/kWh). Settlement does NOT use this -- it
    uses the fixed FEE_FRAC × grid tariff (see `summarise`)."""
    pp = {}
    for lo, hi in PAIRS:
        a = np.abs(lam_all[lo][NB[lo].index(hi)])
        b = np.abs(lam_all[hi][NB[hi].index(lo)])
        pp[(lo, hi)] = 0.5 * (a + b)
    return pp


def summarise(subs, lam_all, Z, day):
    elec_b = np.array([(PRICE * np.array([pyo.value(m.p_el[t]) for t in range(T)]) * DT).sum() for m in subs])
    gas_b = np.array(
        [(C.gas_price / 100.0 * np.array([pyo.value(m.gas[t]) for t in range(T)]) * DT).sum() for m in subs]
    )
    pp = _pair_price(lam_all)  # diagnostic only (see _pair_price)

    # flow b->j and each building's settlement.  The P2P price is FIXED at
    # FEE_FRAC × grid tariff -- exactly central's fee_mode="market" -- NOT the
    # emergent shadow price, so per-building costs are comparable to central.
    settle_price = FEE_FRAC * PRICE  # £/kWh, per hour
    flow = {b: np.zeros((N - 1, T)) for b in range(N)}
    settle_b = np.zeros(N)
    for lo, hi in PAIRS:
        z = Z[(lo, hi)]
        flow[lo][NB[lo].index(hi)] = z
        flow[hi][NB[hi].index(lo)] = -z
        pay = settle_price * z * DT  # hi pays lo at FEE_FRAC × tariff when z>0
        settle_b[lo] -= pay.sum()  # seller receives -> cost down
        settle_b[hi] += pay.sum()  # buyer pays      -> cost up
    admm_cost_b = elec_b + gas_b + settle_b
    admm_op = float((elec_b + gas_b).sum())
    net_export = np.array([flow[b].sum(axis=0) for b in range(N)])  # (b, t)
    # headline "energy traded" = gross pairwise volume  Σ_pairs Σ_t |Z| -- the same
    # definition as central's shared_energy_kWh (Σ of all directional energy_share),
    # so the two columns are comparable.  net_served is the smaller "P2P net import
    # served" figure (Σ of positive net export); shown as a sub-line, not compared.
    traded = 0.5 * sum(float(np.sum(np.abs(flow[b]))) for b in range(N))
    net_served = float(np.maximum(net_export, 0).sum())

    # central reference — may be intractable at large N (N² sharing variables); the
    # decentralised result is still valid on its own, so degrade to ob=sh=None.
    ob = sh = None
    rel_err = float("nan")
    try:
        ob = C.solve_and_extract(C.build_model(True, False), "own_battery")
        sh = C.solve_and_extract(
            C.build_model(True, True, fee_frac=FEE_FRAC, fee_mode="market"),
            "shared",
            fee_frac=FEE_FRAC,
            fee_mode="market",
        )
        rel_err = float(np.sqrt(np.mean(((admm_cost_b - sh["cost_b"]) / np.maximum(np.abs(sh["cost_b"]), 1e-3)) ** 2)))
    except Exception as e:  # noqa: BLE001
        print(f"  (bilateral ADMM: central reference unavailable — {type(e).__name__}: {e})")

    print("\n" + "=" * 74)
    print(f"{'':24}{'bilateral ADMM':>16}{'central @ price':>18}")
    print("-" * 74)
    print(f"{'community op cost £/day':24}{admm_op:>16.2f}" + (f"{sh['op_cost']:>18.2f}" if sh else f"{'n/a':>18}"))
    print(
        f"{'energy traded kWh/day':24}{traded:>16.1f}"
        + (f"{sh['shared_energy_kWh']:>18.1f}" if sh else f"{'n/a':>18}")
    )
    print(f"{'  (P2P net import served)':24}{net_served:>16.1f}{'':>18}")
    print(f"{'RMS rel. error vs central':24}" + (f"{rel_err:>16.2%}" if sh else f"{'n/a':>16}") + f"{'':>18}")
    print(f"{'own-battery baseline £':24}{'':>16}" + (f"{ob['op_cost']:>18.2f}" if ob else f"{'n/a':>18}"))
    print("=" * 74)
    if ob is not None and sh is not None:
        print(f"{'building':12}{'asset':12}{'ADMM £':>9}{'central £':>11}{'own-batt £':>12}{'saving £':>10}")
        for b in range(N):
            print(
                f"{C.BUILDING_IDS[b]:12}{C.ASSETS[b]:12}{admm_cost_b[b]:>9.2f}{sh['cost_b'][b]:>11.2f}"
                f"{ob['cost_b'][b]:>12.2f}{ob['cost_b'][b] - admm_cost_b[b]:>10.2f}"
            )

    return dict(
        elec_b=elec_b,
        gas_b=gas_b,
        settle_b=settle_b,
        admm_cost_b=admm_cost_b,
        admm_op=admm_op,
        traded=traded,
        net_served=net_served,
        ob=ob,
        sh=sh,
        pp=pp,
        settle_price=settle_price,
        flow=flow,
        rel_err=rel_err,
    )


def _shade(ax):
    for t in range(T):
        if C.IS_HIGH[t]:
            ax.axvspan(t, t + 1, color=PRICE_HIGH_COLOR, alpha=0.13, lw=0)
        elif C.IS_LOW[t]:
            ax.axvspan(t, t + 1, color=PRICE_LOW_COLOR, alpha=0.13, lw=0)
        elif C.IS_MEDIUM[t]:
            ax.axvspan(t, t + 1, color=PRICE_MEDIUM_COLOR, alpha=0.10, lw=0)


def plot_admm(hist, Z, S, day):
    edges = np.arange(T + 1)

    def step(y):
        return edges, np.concatenate([y, y[-1:]])

    fig, ax = plt.subplots(2, 2, figsize=(13, 8.6))
    a_conv, a_price, a_flow, a_cost = ax.ravel()

    a_conv.semilogy(hist[:, 0], hist[:, 3], "-o", ms=3, color=ADMM_COLOR, label="primal (RMS consensus, kW)")
    a_conv.semilogy(hist[:, 0], hist[:, 4], "-s", ms=3, color=PRICE_HIGH_COLOR, label="dual (RMS Δ trade, kW)")
    a_conv.axhline(EPS_PRIMAL, color=MUTED, ls=":", lw=1)
    a_conv.set_xlabel("ADMM iteration", fontsize=9)
    a_conv.set_ylabel("residual", fontsize=9)
    a_conv.set_title(
        "Convergence — Algorithm 2 (fully decentralised)", color=INK, fontsize=10, fontweight="bold", loc="left"
    )
    a_conv.legend(frameon=False, fontsize=8)

    _shade(a_price)
    active = [p for p in PAIRS if np.abs(Z[p]).max() > 0.02]
    if active:
        pmat = np.array([S["pp"][p] for p in active])
        a_price.fill_between(
            edges[:-1],
            pmat.min(axis=0),
            pmat.max(axis=0),
            step="post",
            color=MUTED,
            alpha=0.18,
            label="emergent |dual| range (diagnostic)",
        )
    a_price.step(
        *step(S["settle_price"]),
        where="post",
        color=ADMM_COLOR,
        lw=2.4,
        label=f"P2P settlement price ({FEE_FRAC:g}× tariff, fixed)",
    )
    a_price.step(*step(PRICE), where="post", color=INK, lw=1.4, ls="--", label="grid tariff")
    a_price.set_xlim(0, T)
    a_price.set_xlabel("Hour of day", fontsize=9)
    a_price.set_ylabel("£ / kWh", fontsize=9)
    a_price.set_title(
        "P2P settlement price — fixed at FEE_FRAC × tariff (as central), shadow price for reference",
        color=INK,
        fontsize=10,
        fontweight="bold",
        loc="left",
    )
    a_price.legend(frameon=False, fontsize=8)

    _shade(a_flow)
    for b in range(N):
        a_flow.step(*step(S["flow"][b].sum(axis=0)), where="post", lw=1.0, alpha=0.55)
    tot = np.zeros(T)
    for p in PAIRS:
        tot = tot + np.abs(Z[p])
    a_flow.step(*step(tot), where="post", color=INK, lw=2, label="Σ |trades| (kWh/h)")
    a_flow.axhline(0, color=MUTED, lw=0.8)
    a_flow.set_xlim(0, T)
    a_flow.set_xlabel("Hour of day", fontsize=9)
    a_flow.set_ylabel("net traded per building (kW)\n+ export / − import", fontsize=9)
    a_flow.set_title(
        "Per-building net trade (thin) and community trade volume",
        color=INK,
        fontsize=10,
        fontweight="bold",
        loc="left",
    )
    a_flow.legend(frameon=False, fontsize=8)

    x = np.arange(N)
    w = 0.27
    _have_central = S.get("ob") is not None and S.get("sh") is not None
    if _have_central:
        a_cost.bar(x - w, S["ob"]["cost_b"], w, color=MUTED, label="own battery (no sharing)")
        a_cost.bar(x, S["sh"]["cost_b"], w, color=PRICE_HIGH_COLOR, label=f"central market @ {FEE_FRAC:g}× tariff")
        a_cost.bar(
            x + w, S["admm_cost_b"], w, color=ADMM_COLOR, label=f"bilateral ADMM + settlement @ {FEE_FRAC:g}× tariff"
        )
    else:
        a_cost.bar(
            x, S["admm_cost_b"], 0.6, color=ADMM_COLOR, label=f"bilateral ADMM + settlement @ {FEE_FRAC:g}× tariff"
        )
        a_cost.text(
            0.5,
            0.9,
            "central reference unavailable at this N",
            ha="center",
            va="top",
            transform=a_cost.transAxes,
            color=MUTED,
            fontsize=9,
        )
    a_cost.set_xticks(x)
    a_cost.set_xticklabels(
        [f"{i}\n{a}" for i, a in zip(C.BUILDING_IDS, C.ASSETS)], rotation=45, ha="right", fontsize=7
    )
    a_cost.set_ylabel("operating cost (£/day)", fontsize=9)
    a_cost.set_title(
        (
            f"Per-building cost — matches central to {S['rel_err']:.1%} RMS"
            if _have_central
            else "Per-building cost (bilateral ADMM + settlement)"
        ),
        color=INK,
        fontsize=10,
        fontweight="bold",
        loc="left",
    )
    a_cost.legend(frameon=False, fontsize=8)

    for a in (a_conv, a_price, a_flow, a_cost):
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=8)
        a.grid(True, axis="y", color="#ededed", lw=0.7)
        a.set_axisbelow(True)

    fig.suptitle(
        f"{day} — fully-decentralised bilateral P2P sharing (consensus-ADMM, Wang et al. 2017)  "
        f"community £{S['admm_op']:.2f}/day"
        + (f" vs central £{S['sh']['op_cost']:.2f}" if S.get("sh") is not None else "")
        + f"; {S['traded']:.1f} kWh traded",
        color=INK,
        fontsize=11.5,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = C.PLOTS_DIR / "14_admm_bilateral.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")
    _plotly(hist, Z, S, day)


def plot_energy_sources_per_building(subs, S, day):
    """Plot each building's electricity supply mix over the showcase day."""
    edges = np.arange(T + 1)

    def values(model, name):
        variable = getattr(model, name)
        return np.array([pyo.value(variable[t]) for t in range(T)], dtype=float)

    def step(values):
        values = np.asarray(values, dtype=float)
        return edges, np.concatenate([values, values[-1:]])

    source_names = ("PV", "Battery discharge", "P2P import", "Grid import")
    source_colors = ("#e0a800", "#7b4bc9", "#2e8b6e", "#8a8f98")
    fig, axes = plt.subplots(N, 1, figsize=(13, 1.9 * N), sharex=True, squeeze=False)

    for b, building_id in enumerate(C.BUILDING_IDS):
        ax = axes[b, 0]
        _shade(ax)
        grid = values(subs[b], "p_el")
        discharge = values(subs[b], "discharge")
        p_hp = values(subs[b], "p_hp")
        p2p_import = np.maximum(-S["flow"][b], 0.0).sum(axis=0)
        p2p_export = np.maximum(S["flow"][b], 0.0).sum(axis=0)
        sources = np.vstack((C.PV_B[b], discharge, p2p_import, grid))
        ax.stackplot(edges[:-1], sources, labels=source_names, colors=source_colors, alpha=0.82, step="post")
        demand = C.LOAD[b] + p_hp
        ax.step(*step(demand), where="post", color=INK, lw=1.4, label="Demand + heat pump")
        if p2p_export.max() > 1e-6:
            ax.step(
                *step(-p2p_export),
                where="post",
                color="#c0392b",
                lw=1.5,
                ls="--",
                label=f"P2P export / seller ({p2p_export.sum():.1f} kWh)",
            )
        seller_note = f"\nseller: {p2p_export.sum():.1f} kWh" if p2p_export.sum() > 1e-6 else "\nno sales"
        ax.set_ylabel(f"{building_id}\n{C.ASSETS[b]}{seller_note}\nkW", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", color="#ededed", lw=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=7)
        if b == 0:
            ax.set_title(
                f"{day} - where each building's electricity comes from (bilateral ADMM)",
                loc="left",
                fontsize=11,
                fontweight="bold",
                pad=8,
            )
            ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=5)

    axes[-1, 0].set_xlabel("Hour of day", fontsize=9)
    axes[-1, 0].set_xlim(0, T)
    axes[-1, 0].set_xticks(range(0, T + 1, 3))
    fig.tight_layout()
    out = C.PLOTS_DIR / "23_energy_sources_per_building_bilateral.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")
    _plotly_energy_sources_per_building(subs, S, day)


def _plotly_energy_sources_per_building(subs, S, day):
    """Interactive Plotly version of the bilateral source-mix plot."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not installed - skipped bilateral source-mix HTML plot")
        return

    hours = list(range(T)) + [T]
    source_names = ("PV", "Battery discharge", "P2P import", "Grid import")
    source_colors = ("#e0a800", "#7b4bc9", "#2e8b6e", "#8a8f98")
    titles = []

    for b, building_id in enumerate(C.BUILDING_IDS):
        p2p_export = np.maximum(S["flow"][b], 0.0).sum(axis=0)
        seller_note = f"seller: {p2p_export.sum():.1f} kWh" if p2p_export.sum() > 1e-6 else "no sales"
        titles.append(f"{building_id} ({C.ASSETS[b]}; {seller_note})")

    fig = make_subplots(rows=N, cols=1, shared_xaxes=True, vertical_spacing=0.012, subplot_titles=titles)
    for b in range(N):
        grid = np.array([pyo.value(subs[b].p_el[t]) for t in range(T)], dtype=float)
        discharge = np.array([pyo.value(subs[b].discharge[t]) for t in range(T)], dtype=float)
        p_hp = np.array([pyo.value(subs[b].p_hp[t]) for t in range(T)], dtype=float)
        p2p_import = np.maximum(-S["flow"][b], 0.0).sum(axis=0)
        p2p_export = np.maximum(S["flow"][b], 0.0).sum(axis=0)
        sources = (C.PV_B[b], discharge, p2p_import, grid)

        for name, colour, values in zip(source_names, source_colors, sources):
            y = list(np.asarray(values, dtype=float)) + [float(values[-1])]
            fig.add_trace(
                go.Scatter(
                    x=hours,
                    y=y,
                    name=name,
                    legendgroup=name,
                    showlegend=(b == 0),
                    stackgroup=f"sources_{b}",
                    line=dict(color=colour, width=0.8, shape="hv"),
                    hovertemplate=f"{name}: %{{y:.2f}} kW<extra></extra>",
                ),
                row=b + 1,
                col=1,
            )

        demand = C.LOAD[b] + p_hp
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=list(demand) + [float(demand[-1])],
                name="Demand + heat pump",
                legendgroup="Demand + heat pump",
                showlegend=(b == 0),
                line=dict(color=INK, width=1.5, shape="hv"),
                hovertemplate="Demand + heat pump: %{y:.2f} kW<extra></extra>",
            ),
            row=b + 1,
            col=1,
        )
        if p2p_export.max() > 1e-6:
            fig.add_trace(
                go.Scatter(
                    x=hours,
                    y=list(-p2p_export) + [float(-p2p_export[-1])],
                    name="P2P export / seller",
                    legendgroup="P2P export / seller",
                    showlegend=(b == 0),
                    line=dict(color="#c0392b", width=1.5, dash="dash", shape="hv"),
                    hovertemplate="P2P export: %{y:.2f} kW<extra></extra>",
                ),
                row=b + 1,
                col=1,
            )

    fig.update_xaxes(title_text="Hour of day", dtick=3, row=N, col=1)
    fig.update_yaxes(title_text="kW")
    fig.update_layout(
        template="plotly_white",
        height=max(500, 210 * N),
        hovermode="x unified",
        title=f"{day} - where each building's electricity comes from (bilateral ADMM)",
        legend=dict(orientation="h", y=1.01, yanchor="bottom"),
    )
    out = C.PLOTS_DIR / "23_energy_sources_per_building_bilateral.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"saved {out}")


def _sharing_surplus_series(subs, S):
    """Return export-capable energy and actual gross bilateral sharing.

    Bilateral ADMM constrains gross exports by ``PV + battery discharge``.
    This is deliberately not netted against local demand because the model can
    serve local demand from grid import while exporting that available DER.
    """
    available = np.zeros(T)
    for b in range(N):
        discharge = np.array([pyo.value(subs[b].discharge[t]) for t in range(T)], dtype=float)
        available += np.maximum(C.PV_B[b] + discharge, 0.0)

    actual = np.zeros(T)
    for z in S["flow"].values():
        actual += np.maximum(z, 0.0).sum(axis=0)
    if np.any(actual > available + 1e-6):
        print("warning: actual bilateral sharing exceeds export-capable energy")
    return available, actual


def _sharing_destinations(subs, S):
    """Break export-capable energy into shared, curtailed, and retained residual."""
    available, actual = _sharing_surplus_series(subs, S)
    curtailed = np.zeros(T)
    for b in range(N):
        curtailed += np.array([pyo.value(subs[b].curtail[t]) for t in range(T)], dtype=float)
    retained = np.maximum(available - actual - curtailed, 0.0)
    return actual, retained, curtailed


def plot_sharing_surplus(subs, S, day):
    """Plot surplus available for sharing against actual bilateral sharing."""
    available, actual = _sharing_surplus_series(subs, S)
    hours = np.arange(T)
    edges = np.arange(T + 1)

    def step(values):
        values = np.asarray(values, dtype=float)
        return edges, np.concatenate([values, values[-1:]])

    fig, ax = plt.subplots(figsize=(12, 5.5))
    _shade(ax)
    ax.step(*step(available), where="post", color="#e0a800", lw=2.4, label="Export-capable energy (PV + battery)")
    ax.step(*step(actual), where="post", color=ADMM_COLOR, lw=2.4, label="Actually shared")
    ax.fill_between(
        edges,
        step(available)[1],
        step(actual)[1],
        step="post",
        color="#e0a800",
        alpha=0.16,
        label="Surplus not shared",
    )
    ax.set_xlim(0, T)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(0, T + 1, 3))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Energy per hour (kWh)")
    ax.set_title(
        f"{day} - available surplus versus actual bilateral sharing",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, axis="y", color="#ededed", lw=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = C.PLOTS_DIR / "24_sharing_surplus_timeseries.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")
    _plotly_sharing_surplus(available, actual, day)


def plot_sharing_destinations(subs, S, day):
    """Show where export-capable energy went when it was not shared."""
    actual, retained, curtailed = _sharing_destinations(subs, S)
    edges = np.arange(T + 1)

    def step(values):
        values = np.asarray(values, dtype=float)
        return edges, np.concatenate([values, values[-1:]])

    fig, ax = plt.subplots(figsize=(12, 5.5))
    _shade(ax)
    ax.stackplot(
        edges[:-1],
        actual,
        retained,
        curtailed,
        labels=("Actually shared", "Retained locally / displaced grid import", "Curtailed / unused"),
        colors=(ADMM_COLOR, "#5b8fc9", "#c0392b"),
        alpha=0.82,
        step="post",
    )
    ax.step(
        *step(actual + retained + curtailed),
        where="post",
        color="#e0a800",
        lw=2.2,
        label="Export-capable energy (PV + battery)",
    )
    ax.set_xlim(0, T)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(0, T + 1, 3))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Energy per hour (kWh)")
    ax.set_title(
        f"{day} - destinations of export-capable energy (bilateral ADMM)", loc="left", fontsize=12, fontweight="bold"
    )
    ax.legend(frameon=False, loc="upper left", ncol=4)
    ax.grid(True, axis="y", color="#ededed", lw=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = C.PLOTS_DIR / "25_sharing_destinations_bilateral.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")
    _plotly_sharing_destinations(actual, retained, curtailed, day)


def _plotly_sharing_destinations(actual, retained, curtailed, day):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed - skipped destinations HTML plot")
        return

    hours = list(range(T)) + [T]
    series = (
        ("Actually shared", actual, ADMM_COLOR),
        ("Retained locally / displaced grid import", retained, "#5b8fc9"),
        ("Curtailed / unused", curtailed, "#c0392b"),
    )
    fig = go.Figure()
    for name, values, colour in series:
        values = list(np.asarray(values, dtype=float))
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=values + [values[-1]],
                name=name,
                stackgroup="destinations",
                line=dict(color=colour, width=0.8, shape="hv"),
                hovertemplate=f"{name}: %{{y:.2f}} kWh<extra></extra>",
            )
        )
    total = np.asarray(actual) + np.asarray(retained) + np.asarray(curtailed)
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=list(total) + [float(total[-1])],
            name="Export-capable energy (PV + battery)",
            line=dict(color="#e0a800", width=2.2, shape="hv"),
            fill=None,
            hovertemplate="Export-capable energy: %{y:.2f} kWh<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=500,
        hovermode="x unified",
        title=f"{day} - destinations of export-capable energy (bilateral ADMM)",
        xaxis_title="Hour of day",
        yaxis_title="Energy per hour (kWh)",
        legend=dict(orientation="h", y=-0.18),
    )
    fig.update_xaxes(dtick=3)
    fig.update_yaxes(rangemode="tozero")
    out = C.PLOTS_DIR / "25_sharing_destinations_bilateral.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"saved {out}")


def _plotly_sharing_surplus(available, actual, day):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed - skipped surplus HTML plot")
        return

    hours = list(range(T)) + [T]
    available = list(np.asarray(available, dtype=float))
    actual = list(np.asarray(actual, dtype=float))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=available + [available[-1]],
            name="Export-capable energy (PV + battery)",
            line=dict(color="#e0a800", width=2.5, shape="hv"),
            hovertemplate="Export-capable energy: %{y:.2f} kWh<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=actual + [actual[-1]],
            name="Actually shared",
            line=dict(color=ADMM_COLOR, width=2.5, shape="hv"),
            hovertemplate="Actually shared: %{y:.2f} kWh<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=available + [available[-1]],
            name="Surplus not shared",
            line=dict(color="#e0a800", width=0),
            fill="tonexty",
            fillcolor="rgba(224,168,0,0.16)",
            hoverinfo="skip",
            showlegend=True,
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=500,
        hovermode="x unified",
        title=f"{day} - available surplus versus actual bilateral sharing",
        xaxis_title="Hour of day",
        yaxis_title="Energy per hour (kWh)",
        legend=dict(orientation="h", y=-0.18),
    )
    fig.update_xaxes(dtick=3)
    fig.update_yaxes(rangemode="tozero")
    out = C.PLOTS_DIR / "24_sharing_surplus_timeseries.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"saved {out}")


def _plotly(hist, Z, S, day):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return
    hrs = list(range(T))
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Convergence (Algorithm 2)",
            "P2P settlement price (fixed) vs grid tariff",
            "Per-building net trade",
            "Per-building cost",
        ),
    )
    fig.add_trace(go.Scatter(x=hist[:, 0], y=hist[:, 3], name="primal RMS", line=dict(color=ADMM_COLOR)), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=hist[:, 0], y=hist[:, 4], name="dual RMS", line=dict(color=PRICE_HIGH_COLOR)), row=1, col=1
    )
    fig.update_yaxes(type="log", row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=hrs,
            y=S["settle_price"],
            name=f"settlement price ({FEE_FRAC:g}× tariff)",
            line_shape="hv",
            line=dict(color=ADMM_COLOR, width=2.5),
        ),
        row=1,
        col=2,
    )
    active = [p for p in PAIRS if np.abs(Z[p]).max() > 0.02]
    if active:
        pm = np.array([S["pp"][p] for p in active]).mean(axis=0)
        fig.add_trace(
            go.Scatter(
                x=hrs,
                y=pm,
                name="emergent |dual| (diagnostic)",
                line_shape="hv",
                line=dict(color=MUTED, width=1.5, dash="dot"),
            ),
            row=1,
            col=2,
        )
    fig.add_trace(
        go.Scatter(x=hrs, y=PRICE, name="grid tariff", line_shape="hv", line=dict(color="#1a1a1a", dash="dash")),
        row=1,
        col=2,
    )
    for b in range(N):
        fig.add_trace(
            go.Scatter(
                x=hrs,
                y=S["flow"][b].sum(axis=0),
                name=C.BUILDING_IDS[b],
                line_shape="hv",
                line=dict(width=1),
                opacity=0.5,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    _cost_series = [("bilateral ADMM", S["admm_cost_b"], "#2e8b6e")]
    if S.get("ob") is not None and S.get("sh") is not None:
        _cost_series = [
            ("own battery", S["ob"]["cost_b"], "#8a8f98"),
            ("central market", S["sh"]["cost_b"], "#d98a29"),
        ] + _cost_series
    for nm, y, col in _cost_series:
        fig.add_trace(go.Bar(x=C.BUILDING_IDS, y=y, name=nm, marker_color=col), row=2, col=2)
    fig.update_layout(
        template="plotly_white",
        height=820,
        hovermode="x unified",
        barmode="group",
        title=f"{day} — fully-decentralised bilateral P2P consensus-ADMM (Algorithm 2)",
        legend=dict(orientation="h", y=-0.12),
    )
    out = C.PLOTS_DIR / "14_admm_bilateral.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"saved {out}")


if __name__ == "__main__":
    subs, q_all, lam_all, Z, hist = run_admm()
    S = summarise(subs, lam_all, Z, C.SHOWCASE_DAY)
    plot_admm(hist, Z, S, C.SHOWCASE_DAY)
    plot_energy_sources_per_building(subs, S, C.SHOWCASE_DAY)
    plot_sharing_surplus(subs, S, C.SHOWCASE_DAY)
    plot_sharing_destinations(subs, S, C.SHOWCASE_DAY)
