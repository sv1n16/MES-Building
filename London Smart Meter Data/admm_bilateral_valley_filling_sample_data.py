"""
Illustrative ADMM implementation showing two ways to use dual variables to
reduce grid import and incentivize valley filling in a peer-to-peer
energy-sharing community.

Approach 1: Shape the LOCAL grid-import cost (ToU price + convex congestion
            term). The trading-balance dual lambda(t) then naturally
            reflects this shape, and flexible buildings (with a battery)
            shift consumption into the cheap valley hours -- no extra ADMM
            block needed.

Approach 2: Add an explicit AGGREGATE valley-filling term. A second
            consensus quantity (total community import per timestep) is
            driven toward a flattened target profile via its own dual
            variable mu(t), on top of the trading-balance dual lambda(t).

This is a self-contained, synthetic-data example meant to be adapted to your
actual bilateral energy_share[i,j,t] formulation. The ADMM mechanics
(local solve -> global update -> dual update) are the same; only the local
subproblem's variables/objective would need to be swapped for your real
per-building model. Requires pyomo + Gurobi (gurobipy). The congestion and
consensus penalty terms are quadratic but convex, so this is a QP each
building solves each ADMM iteration -- well within Gurobi's native QP
support, no NonConvex flag needed.
"""

import numpy as np
import pyomo.environ as pyo

# ---------------------------------------------------------------------------
# 1. Synthetic problem data
# ---------------------------------------------------------------------------
np.random.seed(0)

N_BUILDINGS = 4
T = 24  # hourly resolution, one day
DT = 1.0  # hours

# time-of-use grid price: cheap at night (valley), expensive 17:00-21:00 (peak)
hours = np.arange(T)
grid_price = 0.10 + 0.15 * np.exp(-0.5 * ((hours - 19) / 2.5) ** 2)  # $/kWh
grid_price += 0.02 * np.sin(hours / T * 2 * np.pi)  # small ripple

# convex congestion / capacity cost, weighted higher at peak hours
# (ingredient for approach 1 -- shapes the MARGINAL cost of importing)
congestion_weight = 0.01 + 0.04 * (grid_price - grid_price.min()) / (grid_price.max() - grid_price.min())

BETA = 0.8  # existing p2p trading fraction of grid price

# per-building fixed net demand (load - PV), with some diversity
base_load = 1.5 + 0.8 * np.sin((hours - 8) / T * 2 * np.pi) ** 2
pv_shape = np.clip(np.sin((hours - 6) / 12 * np.pi), 0, None)

net_demand = np.zeros((N_BUILDINGS, T))
for b in range(N_BUILDINGS):
    load_scale = 0.8 + 0.4 * np.random.rand()
    pv_scale = np.random.uniform(0.0, 2.0)  # some buildings have more PV
    net_demand[b] = load_scale * base_load - pv_scale * pv_shape

# battery specs (the flexibility that actually enables valley filling)
BATTERY_CAP = 4.0  # kWh
BATTERY_POWER = 1.5  # kW
BATTERY_EFF = 0.95

RHO = 5.0  # ADMM penalty, trading-balance consensus
RHO2 = 5.0  # ADMM penalty, aggregate-import consensus (approach 2)
N_ITERS = 60


# ---------------------------------------------------------------------------
# 2. Local (per-building) subproblem
# ---------------------------------------------------------------------------
def solve_building(b, lam, xbar, mu, target_share, use_congestion, use_valley):
    """
    lam            : trading-balance dual, array over t
    xbar           : current average net_trade, array over t
    mu             : valley-filling dual, array over t          (approach 2)
    target_share   : this building's share of the valley target, array over t
    use_congestion : include convex ToU-shaped congestion cost   (approach 1)
    use_valley     : include the mu-driven aggregate-tracking term (approach 2)
    """
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, T - 1)

    m.imp = pyo.Var(m.T, domain=pyo.NonNegativeReals)  # grid import
    m.exp = pyo.Var(m.T, domain=pyo.NonNegativeReals)  # grid export/curtailment
    m.trade = pyo.Var(m.T, domain=pyo.Reals)  # net p2p trade (+ = buying)
    m.soc = pyo.Var(m.T, bounds=(0, BATTERY_CAP))
    m.ch = pyo.Var(m.T, bounds=(0, BATTERY_POWER))
    m.dis = pyo.Var(m.T, bounds=(0, BATTERY_POWER))

    def balance_rule(mm, t):
        return net_demand[b, t] + mm.ch[t] - mm.dis[t] == mm.imp[t] - mm.exp[t] + mm.trade[t]

    m.balance = pyo.Constraint(m.T, rule=balance_rule)

    def soc_rule(mm, t):
        prev = mm.soc[t - 1] if t > 0 else BATTERY_CAP / 2
        return mm.soc[t] == prev + BATTERY_EFF * mm.ch[t] * DT - mm.dis[t] * DT / BATTERY_EFF

    m.soc_dyn = pyo.Constraint(m.T, rule=soc_rule)

    def obj_rule(mm):
        cost = 0
        for t in mm.T:
            grid_cost = grid_price[t] * mm.imp[t]
            if use_congestion:
                # approach 1: shapes the marginal cost of importing so that
                # lam[t] and the battery schedule react by pushing import
                # into the valley hours
                grid_cost += congestion_weight[t] * mm.imp[t] ** 2

            trading_cost = BETA * grid_price[t] * mm.trade[t]

            # ADMM augmented term for the trading-balance consensus
            admm_trade = lam[t] * mm.trade[t] + (RHO / 2) * (mm.trade[t] - xbar[t]) ** 2

            cost += grid_cost - grid_price[t] * mm.exp[t] + trading_cost + admm_trade

            if use_valley:
                # approach 2: pulls this building's import toward its share
                # of the flattened community target profile
                cost += mu[t] * mm.imp[t] + (RHO2 / 2) * (mm.imp[t] - target_share[t]) ** 2

        return cost

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    solver = pyo.SolverFactory("gurobi", solver_io="python")
    solver.solve(m, tee=False)

    imp = np.array([pyo.value(m.imp[t]) for t in m.T])
    trade = np.array([pyo.value(m.trade[t]) for t in m.T])
    return imp, trade


# ---------------------------------------------------------------------------
# 3. ADMM outer loop
# ---------------------------------------------------------------------------
def run_admm(use_congestion, use_valley, n_iters=N_ITERS):
    lam = np.zeros(T)
    xbar = np.zeros(T)
    mu = np.zeros(T)
    target_profile = None  # set after first pass if use_valley

    imports = np.zeros((N_BUILDINGS, T))
    trades = np.zeros((N_BUILDINGS, T))

    for _ in range(n_iters):
        target_share = (target_profile / N_BUILDINGS) if target_profile is not None else np.zeros(T)

        for b in range(N_BUILDINGS):
            imports[b], trades[b] = solve_building(b, lam, xbar, mu, target_share, use_congestion, use_valley)

        xbar = trades.mean(axis=0)  # trading-balance consensus
        lam = lam + RHO * xbar  # dual ascent, target sum(trade) = 0

        if use_valley:
            total_import_local = imports.sum(axis=0)
            if target_profile is None:
                # first pass: set the target as the flattened aggregate
                # import (replace with any target shape you actually want)
                target_profile = np.full(T, total_import_local.mean())
            mu = mu + RHO2 * (total_import_local - target_profile)

    return imports, trades


# ---------------------------------------------------------------------------
# 4. Compare: no shaping / approach 1 only / approach 1+2
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    imp_base, _ = run_admm(use_congestion=False, use_valley=False)
    imp_c1, _ = run_admm(use_congestion=True, use_valley=False)
    imp_c2, _ = run_admm(use_congestion=True, use_valley=True)

    print("Aggregate community grid import per hour:")
    print(f"{'hour':>4} {'base':>8} {'approach1':>10} {'approach1+2':>12}")
    for t in range(T):
        print(f"{t:>4} {imp_base.sum(axis=0)[t]:8.2f} " f"{imp_c1.sum(axis=0)[t]:10.2f} {imp_c2.sum(axis=0)[t]:12.2f}")

    print("\nPeak-to-valley ratio (max/min aggregate import):")
    for name, imp in [("base", imp_base), ("approach 1", imp_c1), ("approach 1+2", imp_c2)]:
        agg = imp.sum(axis=0)
        print(f"  {name:<12}: {agg.max() / max(agg.min(), 1e-6):.2f}")
