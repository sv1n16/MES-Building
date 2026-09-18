"""Central multi-building optimisation (electric HP + gas boiler + ideal P2P sharing)
for the collated *showcase day* — the shared model library, the solver wrapper and
the P2P trade-fee sweep.

  * Input data  -> data/showcase_<day>.csv  +  data/showcase_<day>_batteries.csv
                   (produced by build_showcase_dataset.py). Hourly, 24 steps.
  * Community   -> HETEROGENEOUS: each building has a battery, PV, both, or neither
                   (batteries CSV 'assets' column; PV in per-building pv_<id>_kw
                   columns). A building's battery only acts if it HAS one.
  * build_model(use_battery, sharing, fee_frac, fee_mode) -> a Pyomo model.
    Scenarios: "no_battery"  (batteries off, PV on)
               "own_battery" (batteries on for buildings that have one, no P2P)
               "shared"      (batteries on, P2P sharing on)
  * solve_and_extract(model, label, ...) -> dict of per-building / community results.

Run directly:
  python central_optimisation_showcase.py           solve the 3 scenarios, print the
                                                    community operating-cost table
  python central_optimisation_showcase.py --sweep   sweep the P2P fee across all three
                                                    FEE_MODE interpretations
                                                    -> plots/09_trade_fee_sweep.{png,html,csv}

The showcase figures (06 optimisation summary, 07 building schedules, 08 energy
share, 11 building benefit, 13 sharing savings incl. the decentralised ADMM
methods) are built by showcase_comparison.py, which imports this module.

The physical model (heat pump, boiler, COP, thermal dynamics, electricity/heat
balances, export/import limits, quadratic comfort penalty) and the symmetric
energy_share constraints (bilateral variable, no self-sharing, matched
export/import limits) are unchanged from
central_optimisation_multi_building_electric_hp_boiler_ideal_energy_sharing.py.

The P2P trading fee `frac` has NO external operator. FEE_MODE picks what it means:
  "loss":    a fraction frac of every shared kWh is lost in transfer.
  "forfeit": buyer & seller each forfeit frac*price per kWh (received by no one).
  "market":  buyer pays seller frac*price per kWh (pure transfer, profit flat).
`--sweep` runs all three and compares -> plots/09_trade_fee_sweep.{png,html,csv}.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt

from plot_timeseries import INK, MUTED, PLOTS_DIR

# ---------------------------------------------------------------- config / data
DATA_DIR = Path(__file__).parent / "data"
_day_ptr = DATA_DIR / "showcase_latest.txt"
SHOWCASE_DAY = _day_ptr.read_text().strip() if _day_ptr.exists() else "2013-02-21"

# --- optional overrides for scaled-community runs (set by run_scaling.py) ---
#   SHOWCASE_DATASET               dataset stem under data/ (default showcase_<day>)
#   SHOWCASE_PLOTS_SUBDIR          route every figure into plots/<subdir>/
#   SHOWCASE_SOLVE_TIMELIMIT       per-solve Gurobi time limit in seconds (0 = none)
#   SHOWCASE_RELAX_CENTRAL_BINARIES  relax charging_state to [0,1] (helps large N)
_DATASET_STEM = os.environ.get("SHOWCASE_DATASET", "").strip() or f"showcase_{SHOWCASE_DAY}"
SOLVE_TIME_LIMIT = float(os.environ.get("SHOWCASE_SOLVE_TIMELIMIT", "0") or 0)
RELAX_CENTRAL_BINARIES = (
    os.environ.get("SHOWCASE_RELAX_CENTRAL_BINARIES", "").strip().lower() not in ("", "0", "false", "no")
)

data_hr = pd.read_csv(DATA_DIR / f"{_DATASET_STEM}.csv")
batt = pd.read_csv(DATA_DIR / f"{_DATASET_STEM}_batteries.csv")

_plots_sub = os.environ.get("SHOWCASE_PLOTS_SUBDIR", "").strip()
if _plots_sub:
    PLOTS_DIR = PLOTS_DIR / _plots_sub
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LOAD_COLS = [c for c in data_hr.columns if c.startswith("load_")]
BUILDING_IDS = [c[len("load_") : -len("_kw")] for c in LOAD_COLS]
n_buildings = len(LOAD_COLS)
time_horizon = len(data_hr)
dt = 1.0

PRICE = data_hr["price_gbp_per_kwh"].to_numpy(float)  # £/kWh
T_OUT = data_hr["outdoor_temp_c"].to_numpy(float)
T_SET = data_hr["setpoint_c"].to_numpy(float)
LOAD = data_hr[LOAD_COLS].to_numpy(float).T  # (building, t) kW
# per-building PV generation (kW). Older datasets had a single "pv_kw" column.
if all(f"pv_{b}_kw" in data_hr.columns for b in BUILDING_IDS):
    PV_B = data_hr[[f"pv_{b}_kw" for b in BUILDING_IDS]].to_numpy(float).T  # (building, t)
else:
    _pv1 = data_hr["pv_kw"].to_numpy(float) if "pv_kw" in data_hr.columns else np.zeros(time_horizon)
    PV_B = np.tile(_pv1, (n_buildings, 1))
IS_HIGH = (data_hr["tariff_label"] == "High").to_numpy()
IS_LOW = (data_hr["tariff_label"] == "Low").to_numpy()

CAP = batt.set_index("LCLid")["capacity_kwh"].reindex(BUILDING_IDS).to_numpy(float)
MAXPOW = batt.set_index("LCLid")["max_power_kw"].reindex(BUILDING_IDS).to_numpy(float)
HAS_BATTERY = CAP > 1e-9

# Starting SOC. The batteries CSV has every battery "charged full overnight"; that
# biases the results (free stored energy to dump). Instead give each battery a
# random interior starting SOC in [INIT_SOC_FRAC_LO, INIT_SOC_FRAC_HI]*capacity,
# drawn once with a fixed seed so the central model and the ADMM variants (which
# import INIT_SOC from here) all see the same values. Non-battery buildings: 0.
INIT_SOC_SEED = 42
INIT_SOC_FRAC_LO, INIT_SOC_FRAC_HI = 0.15, 0.85
_init_soc_frac = np.random.default_rng(INIT_SOC_SEED).uniform(
    INIT_SOC_FRAC_LO, INIT_SOC_FRAC_HI, size=n_buildings
)
INIT_SOC = np.where(HAS_BATTERY, _init_soc_frac * CAP, 0.0)
if "assets" in batt.columns:
    ASSETS = batt.set_index("LCLid")["assets"].reindex(BUILDING_IDS).to_numpy(str)
else:
    ASSETS = np.array(["battery" if h else "none" for h in HAS_BATTERY])

# ---- physical parameters (unchanged from the original model) ----
eta_charge = 0.9
eta_discharge = 0.9
p_th_nom = 12.0
T_ref = 7.0
cop_base = 2.18
T_init = 20.0
max_thermal_power = 20.0
efficiency = 0.9
gas_price = 5.0  # p/kWh
hp_max_power = 12.0
alpha = 0.5  # comfort-penalty weight
SHARE_TRADE_FEE_FRAC = 0.5
# How the P2P trading fee `frac` acts (no external operator in any of these):
# "loss":    a fraction `frac` of every shared kWh is physically lost in transfer;
#            the buyer receives only (1-frac). Community profit falls with frac.
# "forfeit": buyer and seller each give up frac*price per kWh as a transaction cost
#            received by no one. Community profit falls with frac (~2x "loss" in £).
# "market":  buyer pays seller frac*price per kWh for the energy. Pure buyer<->seller
#            transfer -> community profit is FLAT vs frac; only the split changes.
FEE_MODE = "market"  # "loss" | "forfeit" | "market"
SOLVER = "gurobi_direct"


# ============================================================================
# MODEL
# ============================================================================
def build_model(
    use_battery: bool, sharing: bool, fee_frac: float = SHARE_TRADE_FEE_FRAC, fee_mode: str = FEE_MODE
) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel()
    m.buildings = pyo.RangeSet(0, n_buildings - 1)
    m.t = pyo.RangeSet(0, time_horizon - 1)
    m.dt = pyo.Param(initialize=dt)

    # --- parameters ---
    m.pv_supply = pyo.Param(
        m.buildings, m.t, initialize={(b, t): float(PV_B[b, t]) for b in m.buildings for t in m.t}
    )
    m.electric_load = pyo.Param(
        m.buildings, m.t, initialize={(b, t): float(LOAD[b, t]) for b in m.buildings for t in m.t}
    )
    m.T_out = pyo.Param(m.buildings, m.t, initialize={(b, t): float(T_OUT[t]) for b in m.buildings for t in m.t})
    m.T_set = pyo.Param(m.buildings, m.t, initialize={(b, t): float(T_SET[t]) for b in m.buildings for t in m.t})
    m.C = pyo.Param(m.buildings, initialize={b: 10.0 for b in m.buildings})
    m.U = pyo.Param(m.buildings, initialize={b: 0.5 for b in m.buildings})

    # --- battery variables (per-building bounds) ---
    m.charge = pyo.Var(m.buildings, m.t, bounds=lambda mm, b, t: (0, MAXPOW[b]), initialize=0)
    m.discharge = pyo.Var(m.buildings, m.t, bounds=lambda mm, b, t: (0, MAXPOW[b]), initialize=0)
    m.soc = pyo.Var(
        m.buildings,
        m.t,
        bounds=lambda mm, b, t: (0, CAP[b]),
        initialize={(b, t): float(INIT_SOC[b]) for b in m.buildings for t in m.t},
    )
    m.charging_state = pyo.Var(
        m.buildings, m.t,
        domain=pyo.NonNegativeReals if RELAX_CENTRAL_BINARIES else pyo.Binary,
        bounds=(0, 1) if RELAX_CENTRAL_BINARIES else None,
    )

    # --- electrical variables ---
    m.p_el_vars = pyo.Var(m.buildings, m.t, bounds=(0, None))  # grid import
    m.curtail = pyo.Var(m.buildings, m.t, bounds=(0, None), initialize=0)
    m.p_hp = pyo.Var(m.buildings, m.t, bounds=(0, hp_max_power))

    # --- heat pump / boiler variables ---
    m.q_heat_vars = pyo.Var(m.buildings, m.t, bounds=(0, p_th_nom), initialize=0)
    m.cop = pyo.Var(m.buildings, m.t, bounds=(1, None), initialize=2.2)
    m.f = pyo.Var(m.buildings, m.t, bounds=(0, 1), initialize=0.5)
    m.q_heat = pyo.Var(m.buildings, m.t, bounds=(0, None), initialize=0)
    m.gas_consumption = pyo.Var(m.buildings, m.t, domain=pyo.NonNegativeReals, initialize=0)
    m.q_boiler_vars = pyo.Var(m.buildings, m.t, bounds=(0, max_thermal_power), initialize=0)

    # --- thermal ---
    m.T_in = pyo.Var(m.buildings, m.t, bounds=(0, None), initialize=T_init)

    # --- energy sharing ---
    m.energy_share = pyo.Var(m.buildings, m.buildings, m.t, domain=pyo.NonNegativeReals, initialize=0)

    # ---------------- battery constraints ----------------
    m.charge_max = pyo.Constraint(
        m.buildings, m.t, rule=lambda mm, b, t: mm.charge[b, t] <= mm.charging_state[b, t] * MAXPOW[b]
    )
    m.discharge_max = pyo.Constraint(
        m.buildings, m.t, rule=lambda mm, b, t: mm.discharge[b, t] <= (1 - mm.charging_state[b, t]) * MAXPOW[b]
    )

    def soc_balance_rule(mm, b, t):
        if t == 0:
            return mm.soc[b, t] == float(INIT_SOC[b])
        return (
            mm.soc[b, t]
            == mm.soc[b, t - 1] + (eta_charge * mm.charge[b, t] - (1.0 / eta_discharge) * mm.discharge[b, t]) * mm.dt
        )

    m.soc_balance = pyo.Constraint(m.buildings, m.t, rule=soc_balance_rule)

    m.no_charge_at_start = pyo.Constraint(
        m.buildings, m.t, rule=lambda mm, b, t: mm.charge[b, t] == 0 if t == 0 else pyo.Constraint.Skip
    )
    m.no_discharge_at_start = pyo.Constraint(
        m.buildings, m.t, rule=lambda mm, b, t: mm.discharge[b, t] == 0 if t == 0 else pyo.Constraint.Skip
    )

    # ---------------- heat pump & boiler ----------------
    m.heat_pump_output = pyo.Constraint(
        m.buildings, m.t, rule=lambda mm, b, t: mm.q_heat_vars[b, t] == mm.cop[b, t] * mm.p_hp[b, t]
    )
    m.q_heat_constraint = pyo.Constraint(
        m.buildings, m.t, rule=lambda mm, b, t: mm.q_heat_vars[b, t] == p_th_nom * mm.f[b, t]
    )
    m.cop_calculation = pyo.Constraint(
        m.buildings, m.t, rule=lambda mm, b, t: mm.cop[b, t] == cop_base + 0.01 * (mm.T_out[b, t] - T_ref)
    )
    m.boiler_max = pyo.Constraint(m.buildings, m.t, rule=lambda mm, b, t: mm.q_boiler_vars[b, t] <= max_thermal_power)
    m.gas_consumption_calc = pyo.Constraint(
        m.buildings, m.t, rule=lambda mm, b, t: mm.gas_consumption[b, t] == mm.q_boiler_vars[b, t] / efficiency
    )

    # delivered fraction: in "loss" mode a fraction fee_frac of every shared kWh is
    # lost in transfer, so the buyer only receives (1 - fee_frac) of what was sent.
    delivered = (1.0 - fee_frac) if fee_mode == "loss" else 1.0

    # ---------------- balances ----------------
    def electricity_balance_rule(mm, b, t):
        energy_received = delivered * sum(mm.energy_share[i, b, t] for i in mm.buildings if i != b)
        energy_sent = sum(mm.energy_share[b, j, t] for j in mm.buildings if j != b)
        return (mm.p_el_vars[b, t] + energy_received - energy_sent - mm.curtail[b, t]) == (
            mm.electric_load[b, t] + mm.charge[b, t] + mm.p_hp[b, t] - mm.discharge[b, t] - mm.pv_supply[b, t]
        )

    m.electricity_balance = pyo.Constraint(m.buildings, m.t, rule=electricity_balance_rule)

    m.heat_balance = pyo.Constraint(
        m.buildings, m.t, rule=lambda mm, b, t: mm.q_heat_vars[b, t] + mm.q_boiler_vars[b, t] == mm.q_heat[b, t]
    )

    def thermal_dynamics_rule(mm, b, t):
        if t == 0:
            return mm.T_in[b, t] == T_init
        return mm.T_in[b, t] == mm.T_in[b, t - 1] + mm.dt / 10 * (
            mm.q_heat[b, t] - 0.5 * (mm.T_in[b, t - 1] - mm.T_out[b, t])
        )

    m.thermal_dynamics = pyo.Constraint(m.buildings, m.t, rule=thermal_dynamics_rule)

    # ---------------- sharing limits ----------------
    def energy_export_limit_rule(mm, b, t):
        energy_sent = sum(mm.energy_share[b, j, t] for j in mm.buildings if j != b)
        return energy_sent <= mm.pv_supply[b, t] + mm.discharge[b, t]

    m.energy_export_limit = pyo.Constraint(m.buildings, m.t, rule=energy_export_limit_rule)

    m.no_self_sharing = pyo.Constraint(m.buildings, m.t, rule=lambda mm, b, t: mm.energy_share[b, b, t] == 0)

    def energy_import_limit_rule(mm, b, t):
        energy_received = delivered * sum(mm.energy_share[i, b, t] for i in mm.buildings if i != b)
        return energy_received <= mm.electric_load[b, t] + mm.p_hp[b, t] + mm.charge[b, t]

    m.energy_import_limit = pyo.Constraint(m.buildings, m.t, rule=energy_import_limit_rule)

    # ---------------- scenario switches ----------------
    # a building's battery is active only if it HAS one and the scenario allows it
    for b in m.buildings:
        if use_battery and HAS_BATTERY[b]:
            continue
        for t in m.t:
            m.charge[b, t].fix(0.0)
            m.discharge[b, t].fix(0.0)
            m.charging_state[b, t].fix(0)
    if not sharing:
        for idx in m.energy_share:
            m.energy_share[idx].fix(0.0)

    # ---------------- objective ----------------
    def objective_rule(mm):
        total = 0.0
        for b in mm.buildings:
            for t in mm.t:
                electricity_cost = PRICE[t] * mm.p_el_vars[b, t] * dt
                gas_cost = gas_price / 100.0 * mm.gas_consumption[b, t] * dt
                comfort_penalty = alpha * (mm.T_in[b, t] - mm.T_set[b, t]) ** 2

                sent = sum(mm.energy_share[b, j, t] for j in mm.buildings if j != b)
                recv = sum(mm.energy_share[i, b, t] for i in mm.buildings if i != b)
                if fee_mode == "loss":
                    # a fraction of every shared kWh is physically lost (handled in the
                    # electricity balance) -> no cash term here.
                    trade_term = 0.0
                elif fee_mode == "forfeit":
                    # both parties give up frac*price per kWh as a transaction cost,
                    # received by no one -> a real loss on the community's books.
                    trade_term = PRICE[t] * fee_frac * dt * (sent + recv)
                else:  # "market": buyer pays seller frac*price per kWh (nets to zero)
                    trade_term = PRICE[t] * fee_frac * dt * (recv - sent)

                # tiny regulariser: prefer the minimal-routing solution among cost-ties
                routing_reg = 1e-4 * sent

                total += electricity_cost + gas_cost + comfort_penalty + trade_term + routing_reg
        return total

    m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    return m


def solve_and_extract(
    m: pyo.ConcreteModel, label: str, fee_frac: float = SHARE_TRADE_FEE_FRAC, fee_mode: str = FEE_MODE
) -> dict:
    solver = pyo.SolverFactory(SOLVER)
    try:
        solver.options["DualReductions"] = 0
        solver.options["NonConvex"] = 2
        solver.options["LogFile"] = ""  # avoid a Windows tempfile-lock error in gurobi_direct
        if SOLVE_TIME_LIMIT > 0:
            solver.options["TimeLimit"] = SOLVE_TIME_LIMIT
    except Exception:
        pass
    res = solver.solve(m, tee=False)
    tc = res.solver.termination_condition
    print(f"[{label}] status={res.solver.status} termination={tc}")

    # accept an incumbent from a time-limited / iteration-limited run, but bail out
    # cleanly if there is no usable solution at all (large-N scaling runs).
    try:
        _probe = pyo.value(m.p_el_vars[0, 0])
    except Exception:
        _probe = None
    _acceptable = (
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.locallyOptimal,
        pyo.TerminationCondition.feasible,
        pyo.TerminationCondition.maxTimeLimit,
        pyo.TerminationCondition.maxIterations,
    )
    if _probe is None or tc not in _acceptable:
        raise RuntimeError(f"[{label}] no usable solution (termination={tc}, status={res.solver.status})")

    def arr(v):
        return np.array([[pyo.value(v[b, t]) for t in m.t] for b in m.buildings])

    grid = arr(m.p_el_vars)
    disch = arr(m.discharge)
    chg = arr(m.charge)
    soc = arr(m.soc)
    p_hp = arr(m.p_hp)
    T_in = arr(m.T_in)
    q_hp_th = arr(m.q_heat_vars)
    q_boiler = arr(m.q_boiler_vars)
    q_total = arr(m.q_heat)
    gas_cons = arr(m.gas_consumption)
    share = np.array([[[pyo.value(m.energy_share[i, j, t]) for t in m.t] for j in m.buildings] for i in m.buildings])
    recv = share.sum(axis=0)  # (building, t) energy received
    sent = share.sum(axis=1)  # (building, t) energy sent

    elec_t = PRICE * grid.sum(axis=0) * dt  # £ per hour
    gas_t = gas_price / 100.0 * gas_cons.sum(axis=0) * dt  # £ per hour
    comfort = float(alpha * ((T_in - T_SET[None, :]) ** 2).sum())

    traded_kWh = float(sent.sum())                 # gross energy sent (== gross received)
    seller_fee_b = (fee_frac * PRICE[None, :] * sent * dt).sum(axis=1)
    buyer_fee_b = (fee_frac * PRICE[None, :] * recv * dt).sum(axis=1)
    if fee_mode == "forfeit":
        # both sides forfeit frac*price -> real loss on the community's books
        fee_b = seller_fee_b + buyer_fee_b
        community_fee = float(fee_b.sum())
        fee_t = fee_frac * PRICE * (sent + recv).sum(axis=0) * dt
        seller_earn_b = np.zeros(n_buildings)
        energy_lost_kWh = 0.0
    elif fee_mode == "market":
        # buyer pays seller frac*price -> pure transfer, nets to zero community-wide
        fee_b = buyer_fee_b                        # what each building pays out
        seller_earn_b = seller_fee_b              # what each building receives
        community_fee = 0.0
        fee_t = np.zeros(time_horizon)
        energy_lost_kWh = 0.0
    else:  # "loss"
        fee_b = np.zeros(n_buildings)
        seller_earn_b = np.zeros(n_buildings)
        community_fee = 0.0                        # no cash; the cost is in extra grid import
        fee_t = np.zeros(time_horizon)
        energy_lost_kWh = fee_frac * traded_kWh
    transfer_b = fee_b                             # back-compat alias

    # ---- per-building operating cost (£/day) ----
    elec_b = (PRICE[None, :] * grid * dt).sum(axis=1)
    gas_b = gas_price / 100.0 * (gas_cons * dt).sum(axis=1)
    if fee_mode == "forfeit":
        trade_cost_b = seller_fee_b + buyer_fee_b          # forfeited by each building
    elif fee_mode == "market":
        trade_cost_b = buyer_fee_b - seller_fee_b          # net cash out (< 0 for net sellers)
    else:  # loss - cost already inside grid import
        trade_cost_b = np.zeros(n_buildings)
    cost_b = elec_b + gas_b + trade_cost_b

    return {
        "label": label,
        "grid": grid,
        "discharge": disch,
        "charge": chg,
        "soc": soc,
        "p_hp": p_hp,
        "T_in": T_in,
        "share": share,
        "recv": recv,
        "sent": sent,
        "q_hp_th": q_hp_th,
        "q_boiler": q_boiler,
        "q_total": q_total,
        "elec_t": elec_t,
        "gas_t": gas_t,
        "fee_t": fee_t,
        "transfer_b": transfer_b,
        "fee_b": fee_b,
        "seller_earn_b": seller_earn_b,
        "seller_fee": float(seller_fee_b.sum()),
        "buyer_fee": float(buyer_fee_b.sum()),
        "market_transfer": float(seller_fee_b.sum()) if fee_mode == "market" else 0.0,
        "energy_lost_kWh": energy_lost_kWh,
        "elec_b": elec_b,
        "gas_b": gas_b,
        "trade_cost_b": trade_cost_b,
        "cost_b": cost_b,
        "fee_mode": fee_mode,
        "elec_cost": float(elec_t.sum()),
        "gas_cost": float(gas_t.sum()),
        "trade_fee": community_fee,
        "comfort": comfort,
        "op_cost": float(elec_t.sum() + gas_t.sum() + community_fee),
        "shared_energy_kWh": float(share.sum()),
    }


# ============================================================================
# TRADE-FEE SWEEP  -> plots/09_trade_fee_sweep.{png,html,csv}
# ============================================================================
FEE_MODES = ("loss", "forfeit", "market")
_MODE_COLOR = {"loss": "#c0392b", "forfeit": "#2e8b6e", "market": "#3b6bb0"}
_MODE_LABEL = {
    "loss": "loss (energy lost in transfer)",
    "forfeit": "forfeit (both give up frac·price)",
    "market": "market (buyer pays seller frac·price)",
}


def sweep_trade_fee(fracs, fee_mode: str) -> tuple[pd.DataFrame, float, float]:
    """Re-solve the shared model across a range of trade-fee fractions for one fee mode."""
    nb = solve_and_extract(build_model(use_battery=False, sharing=False), "no_battery")
    ob = solve_and_extract(build_model(use_battery=True, sharing=False), "own_battery")
    rows = []
    for frac in fracs:
        r = solve_and_extract(
            build_model(use_battery=True, sharing=True, fee_frac=frac, fee_mode=fee_mode),
            f"{fee_mode} f={frac:.2f}",
            fee_frac=frac,
            fee_mode=fee_mode,
        )
        rows.append(
            {
                "fee_mode": fee_mode,
                "fee_frac": frac,
                "traded_kWh": r["shared_energy_kWh"],
                "op_cost": r["op_cost"],
                "sharing_benefit": ob["op_cost"] - r["op_cost"],   # community, vs own-battery
                "community_fee": r["trade_fee"],                   # forfeit only
                "market_transfer": r["market_transfer"],           # market only (buyer->seller £)
                "energy_lost_kWh": r["energy_lost_kWh"],           # loss only
                "seller_fee": r["seller_fee"],
                "buyer_fee": r["buyer_fee"],
            }
        )
    return pd.DataFrame(rows), ob["op_cost"], nb["op_cost"]


def plot_fee_sweep(dfs: dict, ob_op: float, nb_op: float, day: str) -> None:
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.7))
    base = float(dfs["market"]["sharing_benefit"].iloc[0])  # benefit at frac=0 (same for all)

    for mode, d in dfs.items():
        x = d["fee_frac"].to_numpy()
        c = _MODE_COLOR[mode]
        a1.plot(x, d["traded_kWh"], "-o", color=c, lw=2, label=_MODE_LABEL[mode])
        a2.plot(x, d["sharing_benefit"], "-o", color=c, lw=2, label=mode)
        y3 = d["market_transfer"].to_numpy() if mode == "market" else base - d["sharing_benefit"].to_numpy()
        a3.plot(x, y3, "-o", color=c, lw=2,
                label=("market: £ moved buyer→seller" if mode == "market" else f"{mode}: community £ lost"))

    a2.axhline(base, color=MUTED, ls=":", lw=1)
    a2.text(0.02, base, f"  benefit at zero fee = £{base:.2f}", fontsize=7.5, va="bottom", color=MUTED)

    a1.set_ylabel("energy traded (kWh/day)", fontsize=9)
    a1.set_title("Trading volume", color=INK, fontsize=10, fontweight="bold", loc="left")
    a2.set_ylabel("community profit kept (£/day)", fontsize=9)
    a2.set_title("Profit vs fee: loss & forfeit decline, market stays flat",
                 color=INK, fontsize=10, fontweight="bold", loc="left")
    a3.set_ylabel("£/day", fontsize=9)
    a3.set_title("Value destroyed (loss, forfeit) vs merely moved (market)",
                 color=INK, fontsize=10, fontweight="bold", loc="left")
    for ax in (a1, a2, a3):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.axvline(SHARE_TRADE_FEE_FRAC, color=MUTED, ls=":", lw=1)
        ax.set_xlabel("trade-fee fraction of grid price", fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(True, axis="y", color="#ededed", lw=0.7)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle(
        f"{pd.Timestamp(day):%A %d %b %Y} — P2P trading fee: three interpretations (no operator; "
        f"own-battery baseline £{ob_op:.2f}/day; dotted x = current fee {SHARE_TRADE_FEE_FRAC:g})",
        color=INK, fontsize=12, fontweight="bold", x=0.02, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = PLOTS_DIR / "09_trade_fee_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plotly_fee_sweep(dfs: dict, ob_op: float, nb_op: float, day: str) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    base = float(dfs["market"]["sharing_benefit"].iloc[0])
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.07,
                        subplot_titles=("Energy traded (kWh/day)",
                                        "Community profit kept (£/day)",
                                        "Value destroyed / moved (£/day)"))
    for mode, d in dfs.items():
        x = d["fee_frac"].to_numpy()
        c = _MODE_COLOR[mode]
        fig.add_trace(go.Scatter(x=x, y=d["traded_kWh"], mode="lines+markers", name=mode,
                                 legendgroup=mode, line=dict(color=c, width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=d["sharing_benefit"], mode="lines+markers", name=mode,
                                 legendgroup=mode, showlegend=False, line=dict(color=c, width=2.5)),
                      row=1, col=2)
        y3 = d["market_transfer"] if mode == "market" else base - d["sharing_benefit"]
        fig.add_trace(go.Scatter(x=x, y=y3, mode="lines+markers", name=mode, legendgroup=mode,
                                 showlegend=False, line=dict(color=c, width=2.5)), row=1, col=3)
    fig.add_hline(y=base, line=dict(color="#8a8f98", dash="dot"), row=1, col=2)
    for cc in (1, 2, 3):
        fig.add_vline(x=SHARE_TRADE_FEE_FRAC, line=dict(color="#8a8f98", dash="dot"), row=1, col=cc)
        fig.update_xaxes(title_text="trade-fee fraction of grid price", row=1, col=cc)
        fig.update_yaxes(rangemode="tozero", row=1, col=cc)
    fig.update_layout(template="plotly_white", height=460, hovermode="x unified",
                      title=f"{pd.Timestamp(day):%A %d %b %Y} — P2P trading fee: loss vs forfeit vs "
                            f"market (no operator; own-battery baseline £{ob_op:.2f}/day)",
                      legend=dict(orientation="h", y=-0.22))
    out = PLOTS_DIR / "09_trade_fee_sweep.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"saved {out}")


def _require_solver():
    if not pyo.SolverFactory(SOLVER).available(exception_flag=False):
        raise SystemExit(
            f"Solver '{SOLVER}' not available. This MIQCP model needs Gurobi "
            f"(bilinear heat-pump constraint + binary charging state + quadratic comfort term). "
            f"Set SOLVER at the top of this file to your licensed Gurobi interface."
        )


def run_sweep() -> None:
    _require_solver()
    fracs = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    print(f"Trade-fee sweep over {fracs}  (modes: {', '.join(FEE_MODES)})\n")
    dfs, ob_op, nb_op = {}, None, None
    for mode in FEE_MODES:
        dfs[mode], ob_op, nb_op = sweep_trade_fee(fracs, mode)

    for mode in FEE_MODES:
        d = dfs[mode]
        print(f"\n=== {_MODE_LABEL[mode]} ===")
        print(f"{'fee_frac':>9}{'traded kWh':>12}{'op cost £':>11}{'profit kept £':>15}{'detail':>14}")
        print("-" * 61)
        for _, r in d.iterrows():
            if mode == "loss":
                detail = f"{r['energy_lost_kWh']:.1f} kWh lost"
            elif mode == "forfeit":
                detail = f"£{r['community_fee']:.2f} forfeit"
            else:
                detail = f"£{r['market_transfer']:.2f} moved"
            print(
                f"{r['fee_frac']:>9.2f}{r['traded_kWh']:>12.1f}{r['op_cost']:>11.2f}"
                f"{r['sharing_benefit']:>15.2f}{detail:>18}"
            )
    print(f"\nbenefit at zero fee ~ GBP {dfs['market']['sharing_benefit'].iloc[0]:.2f}/day")
    print(f"own-battery baseline op cost GBP {ob_op:.2f}/day  (no-battery GBP {nb_op:.2f}/day)")

    pd.concat(dfs.values(), ignore_index=True).to_csv(PLOTS_DIR / "09_trade_fee_sweep.csv", index=False)
    print(f"saved {PLOTS_DIR / '09_trade_fee_sweep.csv'}")
    plot_fee_sweep(dfs, ob_op, nb_op, SHOWCASE_DAY)
    try:
        plotly_fee_sweep(dfs, ob_op, nb_op, SHOWCASE_DAY)
    except ImportError:
        print("plotly not installed — skipped .html")


# ============================================================================
def main() -> None:
    print(
        f"Showcase day: {SHOWCASE_DAY} | {n_buildings} buildings | {time_horizon} h | "
        f"total storage {CAP.sum():.1f} kWh\n"
    )

    _require_solver()

    scenarios = {
        "no_battery": build_model(use_battery=False, sharing=False),
        "own_battery": build_model(use_battery=True, sharing=False),
        "shared": build_model(use_battery=True, sharing=True),
    }
    results = {k: solve_and_extract(m, k) for k, m in scenarios.items()}

    print("\n" + "=" * 76)
    print(
        f"{'scenario':<13}{'elec £':>9}{'gas £':>9}{'fee £':>8}{'OP COST £':>11}" f"{'comfort':>10}{'traded kWh':>12}"
    )
    print("-" * 76)
    for k, r in results.items():
        print(
            f"{k:<13}{r['elec_cost']:>9.2f}{r['gas_cost']:>9.2f}{r['trade_fee']:>8.2f}"
            f"{r['op_cost']:>11.2f}{r['comfort']:>10.1f}{r['shared_energy_kWh']:>12.1f}"
        )
    print("=" * 76)
    b_ben = results["no_battery"]["op_cost"] - results["own_battery"]["op_cost"]
    s_ben = results["own_battery"]["op_cost"] - results["shared"]["op_cost"]
    print(
        f"battery benefit: £{b_ben:.2f}/day   |   extra benefit from sharing: £{s_ben:.2f}/day"
        f"   (community operating cost = elec + gas + community fee)"
    )
    sh_r = results["shared"]
    if FEE_MODE == "loss":
        detail = f"{sh_r['energy_lost_kWh']:.1f} kWh lost in transfer"
    elif FEE_MODE == "forfeit":
        detail = f"£{sh_r['trade_fee']:.2f}/day forfeited (sellers £{sh_r['seller_fee']:.2f} + buyers £{sh_r['buyer_fee']:.2f})"
    else:
        detail = f"£{sh_r['market_transfer']:.2f}/day moved buyer->seller (community-neutral)"
    print(f"P2P fee mode '{FEE_MODE}' at frac {SHARE_TRADE_FEE_FRAC:g}: {detail}")
    print("\nShowcase figures: run  python showcase_comparison.py")


if __name__ == "__main__":
    if "--sweep" in sys.argv or "sweep" in sys.argv:
        run_sweep()
    else:
        main()
