from gurobipy import Model, GRB
import pyomo.environ as pyo
import json
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from pyomo.util.infeasible import log_infeasible_constraints
import pyomo.environ as pyo

model = pyo.ConcreteModel()

data = pd.read_csv("data\\processed_data_2018_02_21.csv")
data.columns = data.columns.str.strip().str.lower()
cols = data.columns[data.columns.str.contains("consumption", case=False)]

consumption_data = data[cols]

# Get top 3 column names by total consumption
top3_consumers = consumption_data.sum().sort_values(ascending=False).head(4).index.tolist()
cols = top3_consumers  # all building consumption columns

data["price"] = data["price (p/kwh)"] / 100.0

# --- create hour index ---
data["hour"] = data.index // 12  # 12 rows per hour for 5-minute data

# --- aggregate ---
data_hr = pd.DataFrame()
# price is already converted from p/kWh to £/kWh above; do NOT divide by 100 again
data_hr["price"] = data.groupby("hour")["price"].mean()
data_hr["pv"] = data.groupby("hour")["pv"].mean()
data_hr["outdoor temperature"] = data.groupby("hour")["outdoor temperature"].mean()
data_hr["temperature setpoint"] = data.groupby("hour")["temperature setpoint"].mean()

for c in cols:
    data_hr[c] = data.groupby("hour")[c].max()

# Parameters
time_horizon = len(data_hr)  # Number of time steps
delta_t = 1  # Time step in hours
battery_capacity = 12.0  # kWh
max_power = 4.6  # kW
initial_soc = 0.8 * battery_capacity  # kWh
eta_charge = 0.9  # Charging efficiency
eta_discharge = 0.9
n_buildings = len(cols)
p_th_nom = 12
T_ref = 7
cop = np.ones(time_horizon) * 2.18
T_init = 20.0
max_thermal_power = 20.0  # kW
efficiency = 0.9  # Boiler efficiency (fraction)
gas_price = 10  # Gas price (p/kWh)
hp_max_power = 10.0  # Maximum heat pump electrical power (kW)

# ---- Sets ----
model.buildings = pyo.RangeSet(0, n_buildings - 1)  # buildings
model.t = pyo.RangeSet(0, time_horizon - 1)
dt = 1.0
model.dt = pyo.Param(initialize=dt)
alpha = 1

# ---- Parameters ----

irradiance = data_hr["pv"].values.flatten() / 1000
pv_data = irradiance
# Example: PV supply for each building and time (assuming same PV for all buildings)
pv_supplies = {(b, t): pv_data[t] for b in range(n_buildings) for t in range(time_horizon)}
model.pv_supply = pyo.Param(model.buildings, model.t, initialize=pv_supplies)

# ---- Decision Variables ----

# Battery variables
model.charge = pyo.Var(model.buildings, model.t, bounds=(0, max_power), initialize=0)
model.discharge = pyo.Var(model.buildings, model.t, bounds=(0, max_power), initialize=0)

## Initial SOC: random values between 0 and 1 (fractions) for each building,
## converted to kWh by multiplying by battery_capacity
initial_soc_frac = np.random.rand(n_buildings)
initial_soc = [float(f * battery_capacity) for f in initial_soc_frac]

model.soc = pyo.Var(
    model.buildings, model.t, bounds=(0, battery_capacity), initialize={(b): initial_soc[b] for b in model.buildings}
)
model.charging_state = pyo.Var(model.buildings, model.t, domain=pyo.Binary)

# Electrical variables
model.p_el_vars = pyo.Var(model.buildings, model.t, bounds=(0, None))  # imported electricity from grid
model.p_hp = pyo.Var(model.buildings, model.t, bounds=(0, hp_max_power))  # heat pump electrical power
model.P_peak = pyo.Var(bounds=(0, None))
# Energy sharing variables (exported to / imported from the common pool)
model.share_export = pyo.Var(model.buildings, model.t, bounds=(0, None))
model.share_import = pyo.Var(model.buildings, model.t, bounds=(0, None))

# Fixed sharing cost (per kWh) — set lower than grid price (user can adjust)
share_cost = 0.8 * float(data_hr["price"].mean())
model.share_cost = pyo.Param(initialize=share_cost)
# Heat pump and boiler variables
model.q_heat_vars = pyo.Var(
    model.buildings,
    model.t,
    bounds=(0, p_th_nom),
    initialize=0,
)
model.cop = pyo.Var(model.buildings, model.t, bounds=(1, None), initialize=2.2)
model.f = pyo.Var(model.buildings, model.t, bounds=(0, 1), initialize=0.5)
model.q_heat = pyo.Var(model.buildings, model.t, bounds=(0, None), initialize=0)
model.gas_consumption = pyo.Var(model.buildings, model.t, domain=pyo.Reals, bounds=(0, None), initialize=0)
model.q_boiler_vars = pyo.Var(model.buildings, model.t, bounds=(0, max_thermal_power), initialize=0)

# Peak gas consumption variable
model.P_gas_peak = pyo.Var(bounds=(0, None))

# Thermal variables
model.T_in = pyo.Var(model.buildings, model.t, bounds=(0, None), initialize=T_init)

# ---- Parameters (Data) ----

model.electric_load = pyo.Param(
    model.buildings,
    model.t,
    initialize={(b, t): float(data_hr[cols[b]].values[t]) / 1000 for b in model.buildings for t in model.t},
)

model.T_out = pyo.Param(
    model.buildings,
    model.t,
    initialize={(b, t): data_hr["outdoor temperature"].values[t] for b in model.buildings for t in model.t},
)

model.T_set = pyo.Param(
    model.buildings,
    model.t,
    initialize={(b, t): data_hr["temperature setpoint"].values[t] for b in model.buildings for t in model.t},
)

model.C = pyo.Param(model.buildings, initialize={b: 10 for b in model.buildings})
model.U = pyo.Param(model.buildings, initialize={b: 0.5 for b in model.buildings})


# ============================================================================
# CONSTRAINT DEFINITIONS
# ============================================================================

# ---- Battery Constraints ----


def charge_max_rule(model, b, t):
    """Maximum charge power limited by charging state"""
    return model.charge[b, t] <= model.charging_state[b, t] * max_power


model.charge_max = pyo.Constraint(model.buildings, model.t, rule=charge_max_rule)


def discharge_max_rule(model, b, t):
    """Maximum discharge power limited by charging state (cannot charge and discharge simultaneously)"""
    return model.discharge[b, t] <= (1 - model.charging_state[b, t]) * max_power


model.discharge_max = pyo.Constraint(model.buildings, model.t, rule=discharge_max_rule)


def soc_balance_rule(model, b, t):
    """State of charge dynamics"""
    if t == 0:
        return model.soc[b, t] == initial_soc[b]
    else:
        return (
            model.soc[b, t]
            == model.soc[b, t - 1]
            + (eta_charge * model.charge[b, t] - (1.0 / eta_discharge) * model.discharge[b, t]) * model.dt
        )


model.soc_balance = pyo.Constraint(model.buildings, model.t, rule=soc_balance_rule)


def no_charge_at_start_rule(model, b, t):
    """No charging in first time step"""
    if t == 0:
        return model.charge[b, t] == 0
    return pyo.Constraint.Skip


model.no_charge_at_start = pyo.Constraint(model.buildings, model.t, rule=no_charge_at_start_rule)


def no_discharge_at_start_rule(model, b, t):
    """No discharging in first time step"""
    if t == 0:
        return model.discharge[b, t] == 0
    return pyo.Constraint.Skip


model.no_discharge_at_start = pyo.Constraint(model.buildings, model.t, rule=no_discharge_at_start_rule)

# ---- Heat Pump & Boiler Constraints ----


def heat_pump_output_rule(model, b, t):
    """Heat pump output as fraction of nominal power"""
    return model.q_heat_vars[b, t] == model.cop[b, t] * model.p_hp[b, t]


model.heat_pump_output = pyo.Constraint(model.buildings, model.t, rule=heat_pump_output_rule)


def q_heat_rule(model, b, t):
    return model.q_heat_vars[b, t] == p_th_nom * model.f[b, t]


model.q_heat_constraint = pyo.Constraint(model.buildings, model.t, rule=q_heat_rule)


def cop_calculation_rule(model, b, t):
    """COP varies with outdoor temperature"""
    return model.cop[b, t] == cop[0] + 0.01 * (model.T_out[b, t] - T_ref)


model.cop_calculation = pyo.Constraint(model.buildings, model.t, rule=cop_calculation_rule)


def boiler_max_rule(model, b, t):
    """Maximum boiler output"""
    return model.q_boiler_vars[b, t] <= max_thermal_power


model.boiler_max = pyo.Constraint(model.buildings, model.t, rule=boiler_max_rule)


def gas_consumption_rule(model, b, t):
    """Gas consumption from boiler output"""
    return model.gas_consumption[b, t] == model.q_boiler_vars[b, t] / efficiency if efficiency > 0 else 0.0


model.gas_consumption_calc = pyo.Constraint(model.buildings, model.t, rule=gas_consumption_rule)

# ---- General System Constraints ----


def electricity_balance_rule(model, b, t):
    """Electricity balance: grid import = load + charging - PV - discharging + heat pump - exports + imports"""
    return model.p_el_vars[b, t] == (
        model.electric_load[b, t]
        + model.charge[b, t]
        - model.pv_supply[b, t]
        - model.discharge[b, t]
        + model.p_hp[b, t]
        + model.share_import[b, t]
        - model.share_export[b, t]
    )


model.electricity_balance = pyo.Constraint(model.buildings, model.t, rule=electricity_balance_rule)


def heat_balance_rule(model, b, t):
    """Heat balance: total heat = heat pump + boiler"""
    return model.q_heat_vars[b, t] + model.q_boiler_vars[b, t] == model.q_heat[b, t]


model.heat_balance = pyo.Constraint(model.buildings, model.t, rule=heat_balance_rule)

# ---- Thermal Dynamics Constraints ----


def thermal_dynamics_rule(model, b, t):
    """Building thermal dynamics with heat loss"""
    if t == 0:
        return model.T_in[b, t] == T_init
    else:
        return model.T_in[b, t] == model.T_in[b, t - 1] + model.dt / 10 * (
            model.q_heat[b, t] - 0.5 * (model.T_in[b, t - 1] - model.T_out[b, t])
        )


model.thermal_dynamics = pyo.Constraint(model.buildings, model.t, rule=thermal_dynamics_rule)


def peak_rule(model, t):
    return sum(model.p_el_vars[b, t] for b in model.buildings) <= model.P_peak


model.peak_constraint = pyo.Constraint(model.t, rule=peak_rule)


def sharing_balance_rule(model, t):
    """Conservation of shared energy across buildings at each time step"""
    return sum(model.share_import[b, t] for b in model.buildings) == sum(
        model.share_export[b, t] for b in model.buildings
    )


model.sharing_balance = pyo.Constraint(model.t, rule=sharing_balance_rule)


# def peak_gas_rule(model, t):
#     """Peak gas consumption limit"""
#     return sum(model.gas_consumption[b, t] for b in model.buildings) <= model.P_gas_peak


# model.peak_gas_constraint = pyo.Constraint(model.t, rule=peak_gas_rule)
# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

# =====================================================================
# ADMM PARAMETERS
# =====================================================================

rho = 10000.0  # Increased from 100 to force stronger consensus
max_iter = 100
tol = 1e-2

# Define lam (per-building duals) and z as Pyomo Params so they can be updated in the loop
# `lam` is now indexed by building and time: one dual for each consensus constraint `p_b,t - z_t = 0`.
model.lam = pyo.Param(
    model.buildings,
    model.t,
    initialize={(b, t): 0.0 for b in model.buildings for t in model.t},
    mutable=True,
)
model.z = pyo.Param(model.t, initialize=0.0, mutable=True)

# Keep Python arrays for residual calculations
z_py = np.zeros(time_horizon)
# lam_py stores per-building duals: shape (n_buildings, time_horizon)
lam_py = np.zeros((n_buildings, time_horizon))

# =====================================================================
# MODIFY OBJECTIVE FOR ADMM
# =====================================================================


def objective_rule(model):
    total_cost = 0
    for t in model.t:
        # total_cost += model.P_peak  # explicit peak electricity minimization
        # total_cost += model.P_gas_peak  # explicit peak gas minimization
        for b in model.buildings:
            # energy costs: grid imports
            total_cost += data_hr["price"][t] * model.p_el_vars[b, t] * model.dt  # electricity cost from grid
            # cost for energy imported through sharing pool (fixed, lower than grid price)
            total_cost += model.share_cost * model.share_import[b, t] * model.dt
            total_cost += gas_price / 100 * model.gas_consumption[b, t] * model.dt  # gas cost
            # thermal comfort penalty
            total_cost += alpha * (model.T_in[b, t] - model.T_set[b, t]) ** 2
            # ADMM consensus penalty (on grid imports) using per-building duals `lam[b,t]`
            total_cost += model.lam[b, t] * model.p_el_vars[b, t] + (rho / 2) * (model.p_el_vars[b, t] - model.z[t]) ** 2

    return total_cost


model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

solver = pyo.SolverFactory("gurobi_direct")
solver.options["LogFile"] = ""
# =====================================================================
# ADMM LOOP
# =====================================================================

# Track residuals for convergence plot
primal_residuals = []
dual_residuals = []
iterations = []

for k in range(max_iter):

    print(f"\n========== ADMM Iteration {k} ==========")

    # ---- Solve local problems with current dual variables and consensus ----
    solver.solve(model, tee=False)

    # ---- Collect building imports ----
    p_vals = np.array([[pyo.value(model.p_el_vars[b, t]) for t in model.t] for b in model.buildings])

    total_import = p_vals.sum(axis=0)

    # ---- z update (consensus) ----
    # With per-building duals the optimal z minimizes sum_b (rho/2)||p_b - z + lam_b/rho||^2
    # Closed form: z = (1/n) * sum_b (p_b + lam_b / rho)
    z_new = np.sum(p_vals + lam_py / rho, axis=0) / n_buildings

    # ---- convergence check (before updating) ----
    # Primal residual: how much do individual imports deviate from consensus?
    primal_res = np.linalg.norm(p_vals - z_new)
    # Dual residual: is the consensus variable changing?
    dual_res = np.linalg.norm(z_new - z_py) * rho

    print("Primal residual:", primal_res)
    print("Dual residual:", dual_res)
    print("Consensus z:", z_new[:5], "...")  # Print first 5 values for debugging

    # Store residuals for plotting
    primal_residuals.append(primal_res)
    dual_residuals.append(dual_res)
    iterations.append(k)

    # ---- dual and consensus update ----
    # Update per-building duals: y_b := y_b + rho * (p_b - z)
    lam_py = lam_py + rho * (p_vals - z_new)
    z_py[:] = z_new

    # Update Pyomo Params for both lam[b,t] and z[t]
    for b in model.buildings:
        for t in model.t:
            model.lam[b, t].set_value(float(lam_py[b, t]))
    for t in model.t:
        model.z[t].set_value(float(z_py[t]))

    if primal_res < tol and dual_res < tol:
        print("ADMM converged")
        break

# ============================================================================
# PLOT CONVERGENCE
# ============================================================================

import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot primal residual
ax1.semilogy(iterations, primal_residuals, "b-o", linewidth=2, markersize=6, label="Primal Residual")
ax1.axhline(y=tol, color="r", linestyle="--", linewidth=2, label=f"Tolerance = {tol}")
ax1.set_xlabel("Iteration", fontsize=12)
ax1.set_ylabel("Primal Residual ||p - z||", fontsize=12)
ax1.set_title("Primal Residual Convergence", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# Plot dual residual
ax2.semilogy(iterations, dual_residuals, "g-s", linewidth=2, markersize=6, label="Dual Residual")
ax2.axhline(y=tol, color="r", linestyle="--", linewidth=2, label=f"Tolerance = {tol}")
ax2.set_xlabel("Iteration", fontsize=12)
ax2.set_ylabel("Dual Residual ρ||z - z_prev||", fontsize=12)
ax2.set_title("Dual Residual Convergence", fontsize=14, fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

# Combined plot to see trade-off
ax3.semilogy(iterations, primal_residuals, "b-o", linewidth=2, markersize=6, label="Primal ||p - z||")
ax3.semilogy(iterations, dual_residuals, "g-s", linewidth=2, markersize=6, label="Dual ρ||z - z_prev||")
ax3.axhline(y=tol, color="r", linestyle="--", linewidth=2, label=f"Tolerance = {tol}")
ax3.set_xlabel("Iteration", fontsize=12)
ax3.set_ylabel("Residual Value (log scale)", fontsize=12)
ax3.set_title("Primal vs Dual Residuals (Trade-off Check)", fontsize=14, fontweight="bold")
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=11)

plt.tight_layout()
plt.savefig("Results/ADMM_convergence.png", dpi=300, bbox_inches="tight")
print("\nConvergence plot saved to Results/ADMM_convergence.png")
plt.show()

# Print convergence summary
print(f"\n{'='*60}")
print("ADMM CONVERGENCE SUMMARY")
print(f"{'='*60}")
print(f"Iterations completed: {len(iterations)}")
print(f"Initial primal residual: {primal_residuals[0]:.6e}")
print(f"Final primal residual: {primal_residuals[-1]:.6e}")
print(f"Initial dual residual: {dual_residuals[0]:.6e}")
print(f"Final dual residual: {dual_residuals[-1]:.6e}")
print(f"Tolerance: {tol}")
print(f"Converged: {primal_residuals[-1] < tol and dual_residuals[-1] < tol}")

# Trade-off analysis
print(f"\n{'--- TRADE-OFF ANALYSIS ---'}")
primal_reduction = (
    (primal_residuals[0] - primal_residuals[-1]) / primal_residuals[0] * 100 if primal_residuals[0] > 0 else 0
)
dual_reduction = (dual_residuals[0] - dual_residuals[-1]) / dual_residuals[0] * 100 if dual_residuals[0] > 0 else 0
print(f"Primal reduction: {primal_reduction:.1f}%")
print(f"Dual reduction: {dual_reduction:.1f}%")
print(f"Final primal/dual ratio: {primal_residuals[-1] / max(dual_residuals[-1], 1e-10):.2f} (>1 = primal worse)")

# Check for trade-off pattern
late_iterations = min(10, len(iterations) // 2)
primal_change = primal_residuals[-1] - primal_residuals[-late_iterations]
dual_change = dual_residuals[-1] - dual_residuals[-late_iterations]
if primal_change > 0 and dual_change < 0:
    print(f"⚠️  TRADE-OFF DETECTED: Primal increasing while dual decreasing (rho too high)")
elif primal_change > 0.1 * primal_residuals[-late_iterations]:
    print(f"✓ Primal plateaued (no consensus reached)")
elif dual_change < 0 and primal_change < 0.01:
    print(f"✓ Algorithm stable but consensus imperfect (normal for high rho)")
else:
    print(f"✓ Both residuals improving or stable")

print(f"{'='*60}\n")

# ============================================================================
# EXTRACT RESULTS
# ============================================================================

charge_schedule = np.array([[pyo.value(model.charge[b, t]) for t in model.t] for b in model.buildings])
discharge_schedule = np.array([[pyo.value(model.discharge[b, t]) for t in model.t] for b in model.buildings])
soc_schedule = np.array([[pyo.value(model.soc[b, t]) for t in model.t] for b in model.buildings])
net_charge_schedule = charge_schedule - discharge_schedule
grid_import_schedule = np.array([[pyo.value(model.p_el_vars[b, t]) for t in model.t] for b in model.buildings])
heatpump_schedule = np.array([[pyo.value(model.p_hp[b, t]) for t in model.t] for b in model.buildings])
pv_schedule = np.array([[model.pv_supply[b, t] for t in model.t] for b in model.buildings])
load_schedule = np.array([[model.electric_load[b, t] for t in model.t] for b in model.buildings])
share_import_schedule = np.array([[pyo.value(model.share_import[b, t]) for t in model.t] for b in model.buildings])
share_export_schedule = np.array([[pyo.value(model.share_export[b, t]) for t in model.t] for b in model.buildings])
charging_state_schedule = np.array([[pyo.value(model.charging_state[b, t]) for t in model.t] for b in model.buildings])
T_in = [[pyo.value(model.T_in[b, t]) for t in model.t] for b in model.buildings]
T_out = [[pyo.value(model.T_out[b, t]) for t in model.t] for b in model.buildings]
T_set = [[model.T_set[b, t] for t in model.t] for b in model.buildings]
q_boiler_schedule = np.array([[pyo.value(model.q_boiler_vars[b, t]) for t in model.t] for b in model.buildings])
q_heatpump_schedule = np.array([[pyo.value(model.q_heat_vars[b, t]) for t in model.t] for b in model.buildings])
q_total_schedule = np.array([[pyo.value(model.q_heat[b, t]) for t in model.t] for b in model.buildings])


# --- Calculate total costs per building ---
electricity_costs = []
gas_costs = []
total_costs = []
fig = make_subplots(
    rows=n_buildings,
    cols=3,
    subplot_titles=(f"Battery and Heat Pump Operation", "Building Temperatures", "Costs"),
    shared_xaxes=True,
    shared_yaxes=False,
)

for b in model.buildings:
    row = b + 1

    # Electricity & Gas costs
    # Electricity costs: grid imports priced at time-varying tariff + fixed sharing cost for pooled imports
    elec_cost_b_schedule = [
        data_hr["price"].values[t] * pyo.value(model.p_el_vars[b, t])
        + pyo.value(model.share_cost) * pyo.value(model.share_import[b, t])
        for t in model.t
    ]

    elec_cost_b = sum(elec_cost_b_schedule)
    # gas_price is in p/kWh so convert to £/kWh by dividing by 100
    gas_cost_b_schedule = [gas_price / 100.0 * pyo.value(model.gas_consumption[b, t]) * model.dt for t in model.t]
    gas_cost_b = sum(gas_cost_b_schedule)

    total_b = elec_cost_b + gas_cost_b
    electricity_costs.append(elec_cost_b)
    gas_costs.append(gas_cost_b)
    total_costs.append(total_b)

    # Energy variables (Column 1)
    fig.add_trace(
        go.Scatter(
            y=load_schedule[b], mode="lines", name=f"Load (kW) {b}", line=dict(color="black"), showlegend=(b == 0)
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            y=pv_schedule[b], mode="lines", name="PV Supply (kW)", line=dict(color="orange"), showlegend=(b == 0)
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            y=net_charge_schedule[b],
            mode="lines",
            name="Battery Net Charge (kW)",
            line=dict(color="blue"),
            showlegend=(b == 0),
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            y=soc_schedule[b], mode="lines", name="Battery SOC (kWh)", line=dict(color="purple"), showlegend=(b == 0)
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            y=heatpump_schedule[b], mode="lines", name="Heat Pump (kW)", line=dict(color="green"), showlegend=(b == 0)
        ),
        row=row,
        col=1,
    )

    # Temperatures (Column 2)
    T_in = [pyo.value(model.T_in[b, t]) for t in model.t]
    T_out = [pyo.value(model.T_out[b, t]) for t in model.t]
    T_set = [model.T_set[b, t] for t in model.t]

    fig.add_trace(
        go.Scatter(y=T_in, mode="lines", name="Indoor Temp (°C)", line=dict(color="firebrick"), showlegend=(b == 0)),
        row=row,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            y=T_out, mode="lines", name="Outdoor Temp (°C)", line=dict(color="deepskyblue"), showlegend=(b == 0)
        ),
        row=row,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            y=T_set, mode="lines", name="Setpoint (°C)", line=dict(color="darkgreen", dash="dash"), showlegend=(b == 0)
        ),
        row=row,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            y=q_boiler_schedule[b],
            mode="lines",
            name="Boiler Output (kW)",
            line=dict(color="red", width=2),
            showlegend=(b == 0),
        ),
        row=row,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            y=q_heatpump_schedule[b],
            mode="lines",
            name="Heat Pump Output (kW)",
            line=dict(color="blue", width=2),
            showlegend=(b == 0),
        ),
        row=row,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            y=q_total_schedule[b],
            mode="lines",
            name="Total Heat Demand (kW)",
            line=dict(color="black", dash="dash"),
            showlegend=(b == 0),
        ),
        row=row,
        col=2,
    )

    # Electricity price (Column 3)
    fig.add_trace(
        go.Scatter(
            y=data_hr["price"].values,
            mode="lines",
            name="Electricity Price (£/kWh)",
            line=dict(color="pink"),
            showlegend=(b == 0),
        ),
        row=row,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            y=elec_cost_b_schedule,
            mode="lines",
            name="Electricity Consumption Cost",
            line=dict(color="gold"),
            showlegend=(b == 0),
        ),
        row=row,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            y=gas_cost_b_schedule,
            mode="lines",
            name="Gas Consumption Cost",
            line=dict(color="brown"),
            showlegend=(b == 0),
        ),
        row=row,
        col=3,
    )
    # Axis labels
    fig.update_xaxes(title_text="Time", row=row, col=1)
    fig.update_yaxes(title_text="Power/Energy (kW/kWh)", title_font=dict(size=10), row=row, col=1)

    fig.update_xaxes(title_text="Time", row=row, col=2)
    fig.update_yaxes(title_text="Temperature (°C)", title_font=dict(size=10), row=row, col=2)

    fig.update_xaxes(title_text="Time", row=row, col=3)
    fig.update_yaxes(title_text="Cost (£)", title_font=dict(size=10), row=row, col=3)

fig.show()
