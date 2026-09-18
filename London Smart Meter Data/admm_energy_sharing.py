"""Decentralised (ADMM) energy sharing — exchange-ADMM decomposition of the
central showcase optimisation in central_optimisation_showcase.py.

Each building solves ONLY its own subproblem: minimise its private operating cost
(grid electricity + gas + thermal-comfort penalty) by choosing a net
export-to-pool profile  pex_b[t]  (kW, + = exporting to the community, - = importing
from it), subject to its own battery / heat-pump / boiler / thermal constraints.
A single common dual variable `u` is updated by consensus so that the community
imbalance  Σ_b pex_b[t] -> 0  (lossless sharing).

At convergence the dual  y[t] = RHO * u[t]  is the *emergent* P2P price: it is the
shadow price of the sharing balance, i.e. the marginal value of a kWh in the
community at time t. This is the decentralised counterpart of FEE_MODE="market"
in the central model — the price is endogenous, not an exogenous fraction of the
grid tariff — and for the convex (binary-relaxed) model the ADMM allocation
matches the central "market" solution.

Binaries (`charging_state`, preventing simultaneous charge+discharge): with
`RELAX_BINARIES = False` (default) the subproblems are exact MIQPs, matching the
central model, but the ADMM is non-convex and the iteration may oscillate rather
than converge. Set `RELAX_BINARIES = True` to relax `cs` to [0,1]: each subproblem
is then a convex QP and ADMM provably converges (the relaxation is usually tight
anyway because charging losses make simultaneous charge+discharge wasteful).

Outputs: plots/12_admm_convergence.{png,html} and a console comparison with the
central solve.  Run with the Gurobi conda env.
"""

from __future__ import annotations

import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt

import central_optimisation_showcase as C
from plot_timeseries import INK, MUTED

# ---------------------------------------------------------------- ADMM settings
RHO = 1  # penalty parameter (£ / kW^2)
MAX_ITERS = 200
EPS_PRIMAL = 0.0001  # kW   – max_t | Σ_b pex_b[t] |
EPS_DUAL = 0.0003  # kW   – scaled change in the consensus average
RELAX_BINARIES = True  # False -> exact Binary charge/discharge indicator (MIQP subproblems,
#                         matching central_optimisation_showcase); non-convex ADMM, convergence
#                         no longer guaranteed. True -> relax cs to [0,1]: convex QP, provable.
ADAPT_RHO = False  # residual-balancing (Boyd §3.4.1)

PRICE_HIGH_COLOR = "#d98a29"
PRICE_LOW_COLOR = "#5b8fc9"
PRICE_MEDIUM_COLOR = "#8b78b5"
ADMM_COLOR = "#2e8b6e"

N, T, DT = C.n_buildings, C.time_horizon, C.dt
COP = C.cop_base + 0.01 * (C.T_OUT - C.T_ref)  # fixed per timestep (T_out is data)
PV_B = C.PV_B  # (building, t) kW — heterogeneous
HAS_BATTERY = C.HAS_BATTERY


# ============================================================================
# BUILDING SUBPROBLEM
# ============================================================================
def build_subproblem(b: int) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel()
    m.t = pyo.RangeSet(0, T - 1)
    dom = pyo.NonNegativeReals if RELAX_BINARIES else pyo.Binary
    pv_b = PV_B[b]  # this building's PV (kW)

    m.charge = pyo.Var(m.t, bounds=(0, C.MAXPOW[b]), initialize=0)
    m.discharge = pyo.Var(m.t, bounds=(0, C.MAXPOW[b]), initialize=0)
    m.soc = pyo.Var(m.t, bounds=(0, max(C.CAP[b], 1e-6)), initialize=float(C.INIT_SOC[b]))
    m.cs = pyo.Var(m.t, domain=dom, bounds=(0, 1))
    m.p_el = pyo.Var(m.t, bounds=(0, None), initialize=0)  # grid import
    m.curtail = pyo.Var(m.t, bounds=(0, None), initialize=0)
    m.p_hp = pyo.Var(m.t, bounds=(0, C.hp_max_power), initialize=0)
    m.f = pyo.Var(m.t, bounds=(0, 1), initialize=0)
    m.q_boiler = pyo.Var(m.t, bounds=(0, C.max_thermal_power), initialize=0)
    m.gas = pyo.Var(m.t, domain=pyo.NonNegativeReals, initialize=0)
    m.T_in = pyo.Var(m.t, bounds=(0, None), initialize=C.T_init)
    m.pex = pyo.Var(m.t, initialize=0)  # net export to the pool

    m.rho = pyo.Param(initialize=RHO, mutable=True)
    m.center = pyo.Param(m.t, initialize=0.0, mutable=True)  # pex_b - xbar - u

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
    if not HAS_BATTERY[b]:  # building has no battery
        for t in range(T):
            m.charge[t].fix(0.0)
            m.discharge[t].fix(0.0)
            m.cs[t].fix(0)

    # --- heat pump / boiler (COP fixed -> linear) ---
    m.c_hp = pyo.Constraint(m.t, rule=lambda mm, t: COP[t] * mm.p_hp[t] == C.p_th_nom * mm.f[t])
    m.c_gas = pyo.Constraint(m.t, rule=lambda mm, t: mm.gas[t] == mm.q_boiler[t] / C.efficiency)

    def q_heat(mm, t):
        return C.p_th_nom * mm.f[t] + mm.q_boiler[t]

    def thermal(mm, t):
        if t == 0:
            return mm.T_in[t] == C.T_init
        return mm.T_in[t] == mm.T_in[t - 1] + DT / 10 * (q_heat(mm, t) - 0.5 * (mm.T_in[t - 1] - C.T_OUT[t]))

    m.c_therm = pyo.Constraint(m.t, rule=thermal)

    # --- electricity balance & pool interaction limits ---
    m.c_bal = pyo.Constraint(
        m.t,
        rule=lambda mm, t: mm.p_el[t] - mm.pex[t] - mm.curtail[t]
        == C.LOAD[b, t] + mm.charge[t] + mm.p_hp[t] - mm.discharge[t] - pv_b[t],
    )
    m.c_exp = pyo.Constraint(m.t, rule=lambda mm, t: mm.pex[t] <= pv_b[t] + mm.discharge[t])
    m.c_imp = pyo.Constraint(m.t, rule=lambda mm, t: mm.pex[t] >= -(C.LOAD[b, t] + mm.p_hp[t] + mm.charge[t]))

    def obj(mm):
        cost = sum(
            C.PRICE[t] * mm.p_el[t] * DT
            + C.gas_price / 100.0 * mm.gas[t] * DT
            + C.alpha * (mm.T_in[t] - C.T_SET[t]) ** 2
            for t in mm.t
        )
        pen = (mm.rho / 2.0) * sum((mm.pex[t] - mm.center[t]) ** 2 for t in mm.t)
        return cost + pen

    m.obj = pyo.Objective(rule=obj, sense=pyo.minimize)
    return m


def _val(m, name):
    v = getattr(m, name)
    return np.array([pyo.value(v[t]) for t in range(T)])


# ============================================================================
# ADMM LOOP
# ============================================================================
def run_admm():
    C._require_solver()
    solver = pyo.SolverFactory(C.SOLVER)
    _opts = [("LogFile", ""), ("OutputFlag", 0)]
    if C.SOLVE_TIME_LIMIT > 0:  # per-subproblem cap (env SHOWCASE_SOLVE_TIMELIMIT)
        _opts.append(("TimeLimit", C.SOLVE_TIME_LIMIT))
    for opt, val in _opts:
        try:
            solver.options[opt] = val
        except Exception:
            pass

    subs = [build_subproblem(b) for b in range(N)]
    pex = np.zeros((N, T))
    u = np.zeros(T)
    rho = RHO
    hist = []

    print(f"ADMM  N={N}  T={T}  rho0={RHO}  relax_binaries={RELAX_BINARIES}")
    for k in range(MAX_ITERS):
        xbar = pex.mean(axis=0)
        for b, m in enumerate(subs):
            center = pex[b] - xbar - u
            for t in range(T):
                m.center[t] = float(center[t])
            m.rho = float(rho)
            solver.solve(m, tee=False)
            pex[b] = _val(m, "pex")

        xbar_new = pex.mean(axis=0)
        u = u + xbar_new
        primal = float(np.abs(pex.sum(axis=0)).max())  # |Σ_b pex_b[t]|
        dual = float(rho * N * np.abs(xbar_new - xbar).max())
        hist.append((k, primal, dual, rho))
        if k % 5 == 0 or (primal < EPS_PRIMAL and dual < EPS_DUAL):
            print(f"  iter {k:3d}   primal {primal:8.4f} kW   dual {dual:8.4f}   rho {rho:.2f}")
        if primal < EPS_PRIMAL and dual < EPS_DUAL:
            print(f"  converged at iter {k}")
            break
        if ADAPT_RHO:
            if primal > 10 * dual:
                rho *= 2.0
                u /= 2.0
            elif dual > 10 * primal:
                rho *= 0.5
                u *= 2.0

    # -rho*u is the price a net exporter is paid per kWh (shadow price of Σ pex = 0,
    # signed so that excess demand for pool energy -> higher price).
    price = -rho * u
    return pex, price, np.array(hist), subs


# ============================================================================
# COMPARE + PLOT
# ============================================================================
def summarise(pex, price, subs):
    elec_b = np.array([(C.PRICE * _val(m, "p_el") * DT).sum() for m in subs])
    gas_b = np.array([(C.gas_price / 100.0 * _val(m, "gas") * DT).sum() for m in subs])
    settle_b = -np.array([(price * pex[b] * DT).sum() for b in range(N)])  # +pays / -receives
    admm_cost_b = elec_b + gas_b + settle_b
    admm_op = float((elec_b + gas_b).sum())
    traded = float(np.maximum(pex, 0).sum())

    # central reference (own-battery + "market" at frac=1, the clearing price the
    # ADMM dual converges to). May be intractable at large N -> None, and the
    # decentralised results still stand on their own.
    ob = sh = None
    try:
        C._require_solver()
        ob = C.solve_and_extract(C.build_model(True, False), "own_battery")
        sh = C.solve_and_extract(
            C.build_model(True, True, fee_frac=1.0, fee_mode="market"), "shared", fee_frac=1.0, fee_mode="market"
        )
    except Exception as e:  # noqa: BLE001
        print(f"  (exchange-ADMM: central reference unavailable — {type(e).__name__}: {e})")

    print("\n" + "=" * 72)
    print(f"{'':24}{'ADMM (decentralised)':>22}{'central @ price':>16}")
    print("-" * 72)
    _shv = f"{sh['op_cost']:>16.2f}" if sh else f"{'n/a':>16}"
    _sht = f"{sh['shared_energy_kWh']:>16.1f}" if sh else f"{'n/a':>16}"
    _obv = f"{ob['op_cost']:>16.2f}" if ob else f"{'n/a':>16}"
    print(f"{'community op cost £/day':24}{admm_op:>22.2f}{_shv}")
    print(f"{'energy traded kWh/day':24}{traded:>22.1f}{_sht}")
    print(f"{'own-battery baseline £':24}{'':>22}{_obv}")
    print("=" * 72)
    if ob is not None and sh is not None:
        print(f"{'building':12}{'ADMM £':>10}{'central £':>11}{'own-batt £':>12}{'ADMM benefit £':>16}")
        for b, bid in enumerate(C.BUILDING_IDS):
            own_b = ob["cost_b"][b]
            print(
                f"{bid:12}{admm_cost_b[b]:>10.2f}{sh['cost_b'][b]:>11.2f}{own_b:>12.2f}"
                f"{own_b - admm_cost_b[b]:>16.2f}"
            )

    return dict(
        elec_b=elec_b,
        gas_b=gas_b,
        settle_b=settle_b,
        admm_cost_b=admm_cost_b,
        admm_op=admm_op,
        traded=traded,
        ob=ob,
        sh=sh,
    )


def _shade(ax):
    for t in range(T):
        if C.IS_HIGH[t]:
            ax.axvspan(t, t + 1, color=PRICE_HIGH_COLOR, alpha=0.13, lw=0)
        elif C.IS_LOW[t]:
            ax.axvspan(t, t + 1, color=PRICE_LOW_COLOR, alpha=0.13, lw=0)
        elif C.IS_MEDIUM[t]:
            ax.axvspan(t, t + 1, color=PRICE_MEDIUM_COLOR, alpha=0.10, lw=0)


def plot_admm(pex, price, hist, subs, S, day):
    hours = np.arange(T)
    edges = np.arange(T + 1)

    def step(y):
        return edges, np.concatenate([y, y[-1:]])

    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    a_conv, a_price, a_pex, a_cost = ax.ravel()

    # --- 1: convergence ---
    a_conv.semilogy(hist[:, 0], hist[:, 1], "-o", ms=3, color=ADMM_COLOR, label="primal  |Σ pex|")
    a_conv.semilogy(hist[:, 0], hist[:, 2], "-s", ms=3, color=PRICE_HIGH_COLOR, label="dual")
    a_conv.axhline(EPS_PRIMAL, color=MUTED, ls=":", lw=1)
    a_conv.set_xlabel("ADMM iteration", fontsize=9)
    a_conv.set_ylabel("residual (kW)", fontsize=9)
    a_conv.set_title("Convergence", color=INK, fontsize=10, fontweight="bold", loc="left")
    a_conv.legend(frameon=False, fontsize=8)

    # --- 2: emergent P2P price vs grid tariff ---
    _shade(a_price)
    a_price.step(*step(price), where="post", color=ADMM_COLOR, lw=2.2, label="ADMM P2P price  y[t]")
    a_price.step(*step(C.PRICE), where="post", color=INK, lw=1.4, ls="--", label="grid tariff")
    a_price.set_xlim(0, T)
    a_price.set_xlabel("Hour of day", fontsize=9)
    a_price.set_ylabel("£ / kWh", fontsize=9)
    a_price.set_title(
        "Emergent P2P price (shadow price of the sharing balance)",
        color=INK,
        fontsize=10,
        fontweight="bold",
        loc="left",
    )
    a_price.legend(frameon=False, fontsize=8)

    # --- 3: per-building net export + community residual ---
    _shade(a_pex)
    for b in range(N):
        a_pex.step(*step(pex[b]), where="post", lw=1.0, alpha=0.55)
    a_pex.step(*step(pex.sum(axis=0)), where="post", color=INK, lw=2, label="Σ pex  (→ 0)")
    a_pex.axhline(0, color=MUTED, lw=0.8)
    a_pex.set_xlim(0, T)
    a_pex.set_xlabel("Hour of day", fontsize=9)
    a_pex.set_ylabel("net export to pool (kW)\n+ export / − import", fontsize=9)
    a_pex.set_title(
        "Per-building pool interaction (thin) and community imbalance",
        color=INK,
        fontsize=10,
        fontweight="bold",
        loc="left",
    )
    a_pex.legend(frameon=False, fontsize=8)

    # --- 4: per-building cost, ADMM vs central vs own-battery ---
    x = np.arange(N)
    w = 0.27
    if S.get("ob") is not None and S.get("sh") is not None:
        a_cost.bar(x - w, S["ob"]["cost_b"], w, color=MUTED, label="own battery (no sharing)")
        a_cost.bar(x, S["sh"]["cost_b"], w, color=PRICE_HIGH_COLOR, label="central market @ clearing price")
        a_cost.bar(x + w, S["admm_cost_b"], w, color=ADMM_COLOR, label="ADMM + settlement")
    else:
        a_cost.bar(x, S["admm_cost_b"], 0.6, color=ADMM_COLOR, label="ADMM + settlement")
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
    a_cost.set_xticklabels(C.BUILDING_IDS, rotation=45, ha="right", fontsize=7)
    a_cost.set_ylabel("operating cost (£/day)", fontsize=9)
    a_cost.set_title(
        "ADMM = central market at the clearing price; sellers capture the surplus",
        color=INK,
        fontsize=10,
        fontweight="bold",
        loc="left",
    )
    a_cost.legend(frameon=False, fontsize=8)

    for a in (a_conv, a_price, a_pex, a_cost):
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=8)
        a.grid(True, axis="y", color="#ededed", lw=0.7)
        a.set_axisbelow(True)

    fig.suptitle(
        f"{day} — decentralised (ADMM) energy sharing  "
        f"(community £{S['admm_op']:.2f}/day"
        + (f" vs central £{S['sh']['op_cost']:.2f}" if S.get("sh") is not None else "")
        + f"; {S['traded']:.1f} kWh traded)",
        color=INK,
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = C.PLOTS_DIR / "12_admm_convergence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")
    _plotly_admm(pex, price, hist, S, day)


def _plotly_admm(pex, price, hist, S, day):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return
    hours = list(range(T))
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Convergence (residuals)",
            "Emergent P2P price vs grid tariff",
            "Per-building net export & Σ pex",
            "Per-building cost",
        ),
    )
    fig.add_trace(
        go.Scatter(x=hist[:, 0], y=hist[:, 1], name="primal |Σ pex|", line=dict(color=ADMM_COLOR)), row=1, col=1
    )
    fig.add_trace(go.Scatter(x=hist[:, 0], y=hist[:, 2], name="dual", line=dict(color=PRICE_HIGH_COLOR)), row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)

    fig.add_trace(
        go.Scatter(x=hours, y=price, name="ADMM P2P price", line_shape="hv", line=dict(color=ADMM_COLOR, width=2.5)),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=hours, y=C.PRICE, name="grid tariff", line_shape="hv", line=dict(color="#1a1a1a", width=1.5, dash="dash")
        ),
        row=1,
        col=2,
    )

    for b in range(N):
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=pex[b],
                name=C.BUILDING_IDS[b],
                line_shape="hv",
                line=dict(width=1),
                opacity=0.5,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    fig.add_trace(
        go.Scatter(x=hours, y=pex.sum(axis=0), name="Σ pex", line_shape="hv", line=dict(color="#1a1a1a", width=2)),
        row=2,
        col=1,
    )

    if S.get("ob") is not None and S.get("sh") is not None:
        fig.add_trace(
            go.Bar(x=C.BUILDING_IDS, y=S["ob"]["cost_b"], name="own battery", marker_color="#8a8f98"), row=2, col=2
        )
        fig.add_trace(
            go.Bar(x=C.BUILDING_IDS, y=S["sh"]["cost_b"], name="central (market)", marker_color="#d98a29"),
            row=2,
            col=2,
        )
    fig.add_trace(
        go.Bar(x=C.BUILDING_IDS, y=S["admm_cost_b"], name="ADMM + settlement", marker_color="#2e8b6e"), row=2, col=2
    )

    fig.update_layout(
        template="plotly_white",
        height=820,
        hovermode="x unified",
        title=f"{day} — decentralised (ADMM) energy sharing",
        legend=dict(orientation="h", y=-0.12),
    )
    out = C.PLOTS_DIR / "12_admm_convergence.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"saved {out}")


if __name__ == "__main__":
    pex, price, hist, subs = run_admm()
    S = summarise(pex, price, subs)
    plot_admm(pex, price, hist, subs, S, C.SHOWCASE_DAY)
