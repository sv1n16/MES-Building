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
import sys

model = pyo.ConcreteModel()

### Units information ####
#
#
# # Load: W
# Price: p/kWh
# Gas price: p/kWh


###


data = pd.read_csv("data\\processed_data_2018_02_21.csv")
data.columns = data.columns.str.strip().str.lower()
cols = data.columns[data.columns.str.contains("consumption", case=False)]
cols = cols[1:5]  # all building consumption columns
if "price" in data.columns:
    data["price"] = data["price"] / 100.0

# --- create hour index ---
data["hour"] = data.index // 60

# --- aggregate ---
data_hr = pd.DataFrame()
# price is already converted from p/kWh to £/kWh above; do NOT divide by 100 again
data_hr["price"] = data.groupby("hour")["price"].mean()
data_hr["pv"] = data.groupby("hour")["pv"].mean()
data_hr["outdoor temperature"] = data.groupby("hour")["outdoor temperature"].mean()
data_hr["temperature setpoint"] = data.groupby("hour")["temperature setpoint"].mean()

for c in cols:
    data_hr[c] = data.groupby("hour")[c].mean()

# Convert electricity price from p/kWh to £/kWh

print("[sanity] price range (hourly) £/kWh:", data_hr["price"].min(), data_hr["price"].max())
print("[sanity] pv min/max (kW):", data_hr["pv"].min(), data_hr["pv"].max())
for c in cols:
    print(f"[sanity] load '{c}' min/max (kW):", data_hr[c].min(), data_hr[c].max())
print(
    "[sanity] temperature setpoint min/max (°C):",
    data_hr["temperature setpoint"].min(),
    data_hr["temperature setpoint"].max(),
)
data = data_hr

irradiance = data["pv"].values.flatten()
# Parameters
time_horizon = int(len(data))  # Number of hours in the time horizon
# Time step (explicit units): hours. Set dt_hours=1.0 for hourly timesteps,
# or dt_hours = 1.0/60.0 for 1-minute timesteps (in hours).

battery_capacity = 12  # kWh
max_power = 4.6  # kW
eta_charge = 0.9  # Charging efficiency
eta_discharge = 0.9
n_buildings = len(cols)  # Number of buildings
p_th_nom = 12
T_ref = 7
cop = np.ones(time_horizon) * 2.18
T_init = 20
max_thermal_power = 20.0  # kW
efficiency = 0.9  # Boiler efficiency (fraction)
gas_price = 1  # Gas price (p/kWh)
# ---- Sets ----
model.buildings = pyo.RangeSet(0, n_buildings - 1)  # buildings
model.t = pyo.RangeSet(0, time_horizon - 1)
dt = 1  # 60  # minutes
model.dt = pyo.Param(initialize=dt)
alpha = 1000
# ---- Parameters ----

# pv = PVModule(time_horizon=time_horizon, start_point=1, radiation=radiation, area=25.0, beta=30.0, eta_noct=0.15)
pv_data = irradiance
# Example: PV supply for each building and time (assuming same PV for all buildings)
pv_supplies = {(b, t): irradiance[t] for b in range(n_buildings) for t in range(time_horizon)}
model.pv_supply = pyo.Param(model.buildings, model.t, initialize=pv_supplies)

# Now, define variables indexed by building and time

# Battery variables

model.charge = pyo.Var(model.buildings, model.t, bounds=(0, max_power), initialize=0)
model.discharge = pyo.Var(model.buildings, model.t, bounds=(0, max_power), initialize=0)
## Initial SOC: random values between 0 and 1 (fractions) for each building,
## converted to kWh by multiplying by battery_capacity
initial_soc_frac = np.random.rand(n_buildings)
initial_soc = [float(f * battery_capacity) for f in initial_soc_frac]
print("[sanity] initial SOC fractions:", initial_soc_frac)
print("[sanity] initial SOC (kWh):", initial_soc)

model.soc = pyo.Var(
    model.buildings, model.t, bounds=(0, battery_capacity), initialize={(b): initial_soc[b] for b in model.buildings}
)
model.charging_state = pyo.Var(model.buildings, model.t, domain=pyo.Binary)
model.electric_load = pyo.Param(
    model.buildings,
    model.t,
    initialize={(b, t): float(data[cols[b]].values[t]) for b in model.buildings for t in model.t},
)

model.p_hp = pyo.Var(model.buildings, model.t, bounds=(0, None))  # imported electricity from grid
model.p_el_vars = pyo.Var(model.buildings, model.t, bounds=(0, None))  # imported electricity from grid

# Add missing variables and parameters for heat pump and building thermal model
model.q_heat_vars = pyo.Var(model.buildings, model.t, bounds=(0, None), initialize=0)
model.cop = pyo.Var(model.buildings, model.t, bounds=(1, None), initialize=2.2)
model.f = pyo.Var(model.buildings, model.t, bounds=(0, 1), initialize=0.5)
model.T_out = pyo.Param(
    model.buildings,
    model.t,
    initialize={(b, t): data["outdoor temperature"].values[t] for b in model.buildings for t in model.t},
)
model.q_heat = pyo.Var(model.buildings, model.t, bounds=(0, None), initialize=0)
model.C = pyo.Param(model.buildings, initialize={b: 10 for b in model.buildings})  # Example value, replace with actual
model.U = pyo.Param(model.buildings, initialize={b: 0.5 for b in model.buildings})
model.T_in = pyo.Var(model.buildings, model.t, bounds=(0, None), initialize=T_init)
model.T_set = pyo.Param(
    model.buildings,
    model.t,
    initialize={(b, t): data["temperature setpoint"].values[t] for b in model.buildings for t in model.t},
)
# Boiler
model.gas_consumption = pyo.Var(model.buildings, model.t, domain=pyo.Reals, bounds=(0, None), initialize=0)
model.q_boiler_vars = pyo.Var(model.buildings, model.t, bounds=(0, max_thermal_power), initialize=0)


### Plot Demand ###

fig = go.Figure()
for col in cols:
    fig.add_trace(go.Scatter(x=data.index, y=data[col], name="Consumption (kW)"))
    fig.add_trace(go.Scatter(x=data.index, y=irradiance, name="PV Supply (kW)"))


fig.update_xaxes(title_text="Time")
fig.update_yaxes(title_text="Power (kW)")
fig.show()


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
            return model.soc[b, t] == initial_soc[b]
        else:
            return (
                model.soc[b, t]
                == model.soc[b, t - 1]
                + (eta_charge * model.charge[b, t] - (1.0 / eta_discharge) * model.discharge[b, t]) * model.dt
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
    return model.p_el_vars[b, t] == (
        model.electric_load[b, t]
        + model.charge[b, t]
        - model.pv_supply[b, t]
        - model.discharge[b, t]
        + model.p_hp[b, t]
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
        (model.q_heat[b, t]) - model.U[b] * (model.T_in[b, t - 1] - model.T_out[b, t])
    )


model.thermal_inertia = pyo.Constraint(
    model.buildings,
    model.t,
    rule=lambda model, b, t: thermal_dynamics(model, b, t, T_init),
)


def temp_dev_lower_rule(model, b, t):
    return model.T_in[b, t] >= model.T_set[b, t] - 0.5


def temp_dev_high_rule(model, b, t):
    return model.T_in[b, t] <= model.T_set[b, t] + 2.0


model.temp_dev_lower = pyo.Constraint(model.buildings, model.t, rule=temp_dev_lower_rule)


def objective(model):
    return sum(
        data["price"][t] * model.p_el_vars[b, t] * model.dt
        + gas_price / 100 * model.gas_consumption[b, t] * model.dt
        + 1000 * (model.T_in[b, t] - model.T_set[b, t]) ** 2
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
net_charge_schedule = charge_schedule - discharge_schedule

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
    elec_cost_b_schedule = [
        data["price"].values[t]
        * (
            pyo.value(model.p_hp[b, t])
            + pyo.value(model.charge[b, t])
            - pyo.value(model.discharge[b, t])
            - pyo.value(model.pv_supply[b, t])
            + model.electric_load[b, t]
        )
        * 1
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
        go.Scatter(y=load_schedule[b], mode="lines", name="Load (kW)", line=dict(color="black"), showlegend=(b == 0)),
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
            y=data["price"].values,
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
