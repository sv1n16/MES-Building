from gurobipy import Model, GRB
import pyomo.environ as pyo
import json
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from src.Classes.boiler import GasBoiler
from src.Classes.battery import Battery
from src.Classes.photovoltaic import PVModule
from src.Classes.building import Building
from src.Classes.heatpump import HeatPump

from pyomo.util.infeasible import log_infeasible_constraints
import pyomo.environ as pyo

model = pyo.ConcreteModel()


# Load data
radiation = pd.read_csv("data\\radiation.csv").values.flatten()
electric_load = pd.read_csv("data\\load.csv").values.flatten()
price = pd.read_csv("data\\price.csv").values.flatten()
heatload = pd.read_csv("data\\heat_load.csv").values.flatten()
temperature_setpoint = [17, 17, 17.5, 17.5, 17.5, 17.5, 20, 20, 20, 20, 18, 18]
outdoor_temperature = [6.0, 6.17, 6.67, 7.46, 8.5, 9.71, 11.0, 12.29, 13.5, 14.54, 15.33, 15.83]


# Parameters
time_horizon = len(price)  # Number of time steps
delta_t = 1  # Time step in hours
battery_capacity = 12  # kWh
max_power = 4.6  # kW
initial_soc = 0.8 * battery_capacity  # kWh
eta_charge = 0.9  # Charging efficiency
eta_discharge = 0.9
n_buildings = 1
p_th_nom = 12
T_ref = 7
cop = np.ones(time_horizon) * 2.18
T_init = 15
max_thermal_power = 20.0  # kW
efficiency = 0.9  # Boiler efficiency (fraction)
gas_price = 0.5  # Gas price (currency/kWh)
# ---- Sets ----
model.buildings = pyo.RangeSet(0, n_buildings - 1)  # buildings
model.t = pyo.RangeSet(0, time_horizon - 1)
dt = 1.0
model.dt = pyo.Param(initialize=dt)

# ---- Parameters ----

pv = PVModule(time_horizon=time_horizon, start_point=2, radiation=radiation, area=25.0, beta=30.0, eta_noct=0.15)
pv_data = pv.p_el_supply
# Example: PV supply for each building and time (assuming same PV for all buildings)
pv_supplies = {(b, t): pv.p_el_supply[t] for b in range(n_buildings) for t in range(time_horizon)}
model.pv_supply = pyo.Param(model.buildings, model.t, initialize=pv_supplies)

# Now, define variables indexed by building and time

# Battery variables

model.charge = pyo.Var(model.buildings, model.t, bounds=(0, max_power), initialize=0)
model.discharge = pyo.Var(model.buildings, model.t, bounds=(0, max_power), initialize=0)
model.soc = pyo.Var(model.buildings, model.t, bounds=(0, battery_capacity), initialize=initial_soc)
model.charging_state = pyo.Var(model.buildings, model.t, domain=pyo.Binary)
model.electric_load = pyo.Param(
    model.buildings, model.t, initialize={(b, t): float(electric_load[t]) for b in model.buildings for t in model.t}
)

model.p_hp = pyo.Var(model.buildings, model.t, bounds=(0, None))  # imported electricity from grid
model.p_el_vars = pyo.Var(model.buildings, model.t, bounds=(0, None))  # imported electricity from grid

# Add missing variables and parameters for heat pump and building thermal model
model.q_heat_vars = pyo.Var(model.buildings, model.t, bounds=(0, None), initialize=0)
model.cop = pyo.Var(model.buildings, model.t, bounds=(1, None), initialize=2.2)
model.f = pyo.Var(model.buildings, model.t, bounds=(0, 1), initialize=0.5)
model.T_out = pyo.Param(
    model.buildings, model.t, initialize={(b, t): outdoor_temperature[t] for b in model.buildings for t in model.t}
)
model.q_heat = pyo.Var(model.buildings, model.t, bounds=(0, None), initialize=0)
model.C = pyo.Param(model.buildings, initialize={b: 10 for b in model.buildings})  # Example value, replace with actual
model.U = pyo.Param(model.buildings, initialize={b: 0.5 for b in model.buildings})
model.T_in = pyo.Var(model.buildings, model.t, bounds=(0, None), initialize=T_init)
model.T_set = pyo.Param(
    model.buildings, model.t, initialize={(b, t): temperature_setpoint[t] for b in model.buildings for t in model.t}
)
# Boiler
model.gas_consumption = pyo.Var(model.buildings, model.t, domain=pyo.Reals, bounds=(0, None), initialize=0)
model.q_boiler_vars = pyo.Var(model.buildings, model.t, bounds=(0, max_thermal_power), initialize=0)


for b in model.buildings:
    constr_name_charge = f"max_charge_constr_{b}"
    constr_name_discharge = f"max_discharge_constr_{b}"
    constr_name_soc = f"soc_constr_{b}"
    constr_no_charge_at_start_rule = f"no_charge_at_start_{b}"
    constr_no_discharge_at_start_rule = f"no_discharge_at_start_{b}"
    constr_name_p_coupl = f"p_coupl_constr_{b}"
    constr_name_q_heat = f"q_heat_constr_{b}"
    constr_name_cop_constr = f"cop_constr_{b}"
    constr_name_gas_consumption = f"gas_consumption_constr_{b}"
    constr_name_boiler_max = f"boiler_max_constr_{b}"

    def c_rule(model, b, t):
        return model.charge[b, t] <= model.charging_state[b, t] * max_power

    def d_rule(model, b, t):
        return model.discharge[b, t] <= (1 - model.charging_state[b, t]) * max_power

    def p_coupl_rule(model, b, t):
        return model.q_heat_vars[b, t] == model.cop[b, t] * model.p_hp[b, t]

    # def q_heat_rule(model, b, t):
    #     return model.q_heat_vars[b, t] == p_th_nom * model.f[b, t]

    def cop_rule(model, b, t):
        return model.cop[b, t] == cop[0] + 0.01 * (model.T_out[b, t] - T_ref)

    #
    # Ensure charge and discharge start at 0
    def no_charge_at_start_rule(model):
        return model.charge[b, 0] == 0

    def no_discharge_at_start_rule(model):
        return model.discharge[b, 0] == 0

    def soc_rule(model, b, t):
        if t == 0:
            return model.soc[b, t] == initial_soc
        else:
            return (
                model.soc[b, t]
                == model.soc[b, t - 1]
                + (eta_charge * model.charge[b, t] - (1.0 / eta_discharge) * model.discharge[b, t]) * 1
            )

    def gas_consumption_rule(model, b, t):
        return model.gas_consumption[b, t] == model.q_boiler_vars[b, t] / efficiency if efficiency > 0 else 0.0

    # Boiler output is zero if off, and between -max_thermal_power and 0 if on
    def boiler_max(model, b, t):
        return model.q_boiler_vars[b, t] <= max_thermal_power

    if constr_name_charge in model.component_map(pyo.Constraint):
        model.del_component(constr_name_charge)
    if constr_name_discharge in model.component_map(pyo.Constraint):
        model.del_component(constr_name_discharge)
    if constr_name_soc in model.component_map(pyo.Constraint):
        model.del_component(constr_name_soc)

    setattr(model, constr_name_charge, pyo.Constraint(model.buildings, model.t, rule=c_rule))
    setattr(model, constr_name_discharge, pyo.Constraint(model.buildings, model.t, rule=d_rule))
    setattr(model, constr_name_soc, pyo.Constraint(model.buildings, model.t, rule=soc_rule))
    setattr(
        model,
        constr_no_discharge_at_start_rule,
        pyo.Constraint(model.buildings, model.t, rule=no_discharge_at_start_rule),
    )
    setattr(
        model, constr_no_charge_at_start_rule, pyo.Constraint(model.buildings, model.t, rule=no_charge_at_start_rule)
    )
    setattr(model, constr_name_p_coupl, pyo.Constraint(model.buildings, model.t, rule=p_coupl_rule))
    # setattr(model, constr_name_q_heat, pyo.Constraint(model.buildings, model.t, rule=q_heat_rule))
    setattr(model, constr_name_cop_constr, pyo.Constraint(model.buildings, model.t, rule=cop_rule))
    setattr(model, constr_name_gas_consumption, pyo.Constraint(model.buildings, model.t, rule=gas_consumption_rule))
    setattr(model, constr_name_boiler_max, pyo.Constraint(model.buildings, model.t, rule=boiler_max))


## General constraints
def electricity_balance_rule(model, b, t):
    return model.electric_load[b, t] + model.p_hp[b, t] + model.charge[b, t] == (
        model.pv_supply[b, t] + model.discharge[b, t] + model.p_el_vars[b, t]
    )


model.electric_balance = pyo.Constraint(model.buildings, model.t, rule=electricity_balance_rule)


def heat_balance_rule(model, b, t):
    return model.q_heat_vars[b, t] + model.q_boiler_vars[b, t] == model.q_heat[b, t]


model.heat_demand_match = pyo.Constraint(model.buildings, model.t, rule=heat_balance_rule)


def thermal_dynamics(
    model,
    b,
    t,
    T_init,
):
    if t == 0:
        return model.T_in[b, t] == T_init
    return model.T_in[b, t] == model.T_in[b, t - 1] + model.dt / model.C[b] * (
        model.q_heat[b, t] - model.U[b] * (model.T_in[b, t - 1] - model.T_out[b, t])
    )


model.thermal_inertia = pyo.Constraint(
    model.buildings,
    model.t,
    rule=lambda model, b, t: thermal_dynamics(model, b, t, T_init),
)


def objective(model):
    return sum(
        price[t]
        * (
            -model.pv_supply[b, t]
            + model.discharge[b, t]
            - model.charge[b, t]
            + model.p_hp[b, t]
            + model.electric_load[b, t]
        )
        * model.dt
        + gas_price * model.gas_consumption[b, t] * model.dt
        + 100 * (model.T_in[b, t] - model.T_set[b, t]) ** 2
        for b in model.buildings
        for t in model.t
    )


model.obj = pyo.Objective(rule=objective, sense=pyo.minimize)


# In Pyomo we’ll just use explicit bounds instead of this constraint,
# since the domain/bounds were already set.
solver = pyo.SolverFactory("gurobi_direct")
result = solver.solve(model, tee=True, logfile="Results/Temp_setpoint/solver_log.txt")

log_infeasible_constraints(model)

# Extract results
charge_schedule = np.array([[pyo.value(model.charge[b, t]) for t in model.t] for b in model.buildings])
discharge_schedule = np.array([[pyo.value(model.discharge[b, t]) for t in model.t] for b in model.buildings])
soc_schedule = np.array([[pyo.value(model.soc[b, t]) for t in model.t] for b in model.buildings])
heatpump_schedule = np.array([[pyo.value(model.p_hp[b, t]) for t in model.t] for b in model.buildings])
pv_schedule = np.array([[model.pv_supply[b, t] for t in model.t] for b in model.buildings])
load_schedule = np.array([[model.electric_load[b, t] for t in model.t] for b in model.buildings])
charging_state_schedule = np.array([[pyo.value(model.charging_state[b, t]) for t in model.t] for b in model.buildings])
T_in = [pyo.value(model.T_in[b, t]) for t in model.t]
T_out = [pyo.value(model.T_out[b, t]) for t in model.t]
T_set = [model.T_set[b, t] for t in model.t]
q_boiler_schedule = np.array([[pyo.value(model.q_boiler_vars[b, t]) for t in model.t] for b in model.buildings])
q_heatpump_schedule = np.array([[pyo.value(model.q_heat_vars[b, t]) for t in model.t] for b in model.buildings])
q_total_schedule = np.array([[pyo.value(model.q_heat[b, t]) for t in model.t] for b in model.buildings])

## Compute costs

# --- Calculate total costs per building ---
electricity_costs = []
gas_costs = []
total_costs = []

for b in model.buildings:
    # Electricity cost per building
    elec_cost_b = sum(
        price[t]
        * (
            pyo.value(model.p_hp[b, t])
            + pyo.value(model.charge[b, t])
            - pyo.value(model.discharge[b, t])
            - pyo.value(model.pv_supply[b, t])
            + model.electric_load[b, t]
        )
        * delta_t
        for t in model.t
    )
    print("Gas consumption:", [pyo.value(model.gas_consumption[b, t]) for t in model.t])
    # Gas cost per building
    gas_cost_b = sum(gas_price * pyo.value(model.gas_consumption[b, t]) * delta_t for t in model.t)

    total_b = elec_cost_b + gas_cost_b

    electricity_costs.append(elec_cost_b)
    gas_costs.append(gas_cost_b)
    total_costs.append(total_b)
    print(
        f"Building {b}: Electricity Cost = {elec_cost_b:.2f}, Gas Cost = {gas_cost_b:.2f}, Total Cost = {total_b:.2f}"
    )
    fig = go.Figure()

    # Energy-related variables
    fig.add_trace(go.Scatter(y=load_schedule[b], mode="lines", name="Load (kW)", line=dict(color="black")))
    fig.add_trace(go.Scatter(y=pv_schedule[b], mode="lines", name="PV Supply (kW)", line=dict(color="orange")))
    fig.add_trace(go.Scatter(y=charge_schedule[b], mode="lines", name="Battery Charge (kW)", line=dict(color="blue")))
    fig.add_trace(
        go.Scatter(y=discharge_schedule[b], mode="lines", name="Battery Discharge (kW)", line=dict(color="red"))
    )
    fig.add_trace(go.Scatter(y=soc_schedule[b], mode="lines", name="Battery SOC (kWh)", line=dict(color="purple")))
    fig.add_trace(go.Scatter(y=heatpump_schedule[b], mode="lines", name="Heat Pump (kW)", line=dict(color="green")))

    # Temperature plot (Indoor vs Outdoor vs Setpoint)
    T_in = [pyo.value(model.T_in[b, t]) for t in model.t]
    T_out = [pyo.value(model.T_out[b, t]) for t in model.t]
    T_set = [model.T_set[b, t] for t in model.t]

    fig.add_trace(go.Scatter(y=T_in, mode="lines", name="Indoor Temp (°C)", line=dict(color="firebrick")))
    fig.add_trace(go.Scatter(y=T_out, mode="lines", name="Outdoor Temp (°C)", line=dict(color="deepskyblue")))
    fig.add_trace(go.Scatter(y=T_set, mode="lines", name="Setpoint (°C)", line=dict(color="darkgreen", dash="dash")))

    # Add secondary y-axis for electricity price
    fig.add_trace(go.Scatter(y=price, mode="lines", name="Electricity Price", line=dict(color="gold"), yaxis="y2"))

    # Layout
    fig.update_layout(
        title=f"Building {b}: Energy, Temperature, and Price",
        xaxis_title="Time Step",
        yaxis_title="Power / Energy / Temperature",
        yaxis2=dict(title="Electricity Price", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        height=600,
        width=1000,
    )

    fig.show()


# --- Plot Boiler & Heat Pump Outputs ---

for b in range(n_buildings):
    fig = make_subplots(rows=1, cols=1, subplot_titles=[f"Building {b} Thermal Outputs"])

    fig.add_trace(
        go.Scatter(
            y=q_boiler_schedule[b],
            mode="lines",
            name="Boiler Output (kW)",
            line=dict(color="red", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            y=q_heatpump_schedule[b],
            mode="lines",
            name="Heat Pump Output (kW)",
            line=dict(color="blue", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            y=q_total_schedule[b],
            mode="lines",
            name="Total Heat Demand (kW)",
            line=dict(color="black", dash="dash"),
        )
    )

    fig.update_layout(
        title=f"Building {b} — Heat Supply from Boiler and Heat Pump",
        xaxis_title="Time Step",
        yaxis_title="Thermal Power (kW)",
        legend=dict(traceorder="normal"),
        width=900,
        height=400,
    )
    fig.add_trace(go.Scatter(y=T_in, mode="lines", name="Indoor Temperature (°C)", line=dict(color="blue")))
    fig.add_trace(go.Scatter(y=T_set, mode="lines", name="Setpoint (°C)", line=dict(color="red", dash="dot")))

    fig.update_layout(
        title=f"Building {b} — Indoor Temperature vs Setpoint",
        xaxis_title="Time Step",
        yaxis_title="Temperature (°C)",
        width=900,
        height=400,
    )

    fig.show()
