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
battery_capacity = 12.0  # kWh
max_power = 4.6  # kW
initial_soc = 0.6 * battery_capacity  # kWh
eta_charge = 0.9  # Charging efficiency
eta_discharge = 0.9
n_buildings = 5

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

model.p_el_vars = pyo.Var(model.buildings, model.t, bounds=(0, None))  # or appropriate bounds


# Load & PV per building (toy example)


for b in model.buildings:
    constr_name_charge = f"max_charge_constr_{b}"
    constr_name_discharge = f"max_discharge_constr_{b}"
    constr_name_soc = f"soc_constr_{b}"
    constr_no_charge_at_start_rule = f"no_charge_at_start_{b}"
    constr_no_discharge_at_start_rule = f"no_discharge_at_start_{b}"

    def c_rule(model, b, t):
        return model.charge[b, t] <= model.charging_state[b, t] * max_power

    def d_rule(model, b, t):
        return model.discharge[b, t] <= (1 - model.charging_state[b, t]) * max_power

    # Remove if exists
    if constr_name_charge in model.component_map(pyo.Constraint):
        model.del_component(constr_name_charge)
    if constr_name_discharge in model.component_map(pyo.Constraint):
        model.del_component(constr_name_discharge)
    if constr_name_soc in model.component_map(pyo.Constraint):
        model.del_component(constr_name_soc)

    # Ensure charge and discharge start at 0
    def no_charge_at_start_rule(model):
        return model.charge[b, 0] == 0

    def no_discharge_at_start_rule(model):
        return model.discharge[b, 0] == 0

    def soc_rule(model, t):
        if t == 0:
            return model.soc[b, t] == initial_soc
        else:
            return (
                model.soc[b, t]
                == model.soc[b, t - 1]
                + (eta_charge * model.charge[b, t] - (1.0 / eta_discharge) * model.discharge[b, t]) * 1
            )

    setattr(model, constr_name_charge, pyo.Constraint(model.buildings, model.t, rule=c_rule))
    setattr(model, constr_name_discharge, pyo.Constraint(model.buildings, model.t, rule=d_rule))
    setattr(model, constr_name_soc, pyo.Constraint(model.t, rule=soc_rule))
    setattr(
        model,
        constr_no_discharge_at_start_rule,
        pyo.Constraint(model.buildings, model.t, rule=no_discharge_at_start_rule),
    )
    setattr(
        model, constr_no_charge_at_start_rule, pyo.Constraint(model.buildings, model.t, rule=no_charge_at_start_rule)
    )


def electricity_balance_rule(model, b, t):
    return model.electric_load[b, t] == (
        model.pv_supply[b, t] + model.discharge[b, t] - model.charge[b, t] + model.p_el_vars[b, t]
    )


model.electric_balance = pyo.Constraint(model.buildings, model.t, rule=electricity_balance_rule)


def objective(model):
    return sum(
        price[t]
        * (
            model.pv_supply[b, t]
            + model.discharge[b, t]
            - model.charge[b, t]
            + model.p_el_vars[b, t]
            + model.electric_load[b, t]
        )
        * model.dt
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
grid_import_schedule = np.array([[pyo.value(model.p_el_vars[b, t]) for t in model.t] for b in model.buildings])
pv_schedule = np.array([[model.pv_supply[b, t] for t in model.t] for b in model.buildings])
load_schedule = np.array([[model.electric_load[b, t] for t in model.t] for b in model.buildings])

# Plot results with Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=n_buildings, cols=1, shared_xaxes=True, subplot_titles=[f"Building {b}" for b in range(n_buildings)]
)

for b in range(n_buildings):
    fig.add_trace(
        go.Scatter(y=load_schedule[b], mode="lines", name="Load", line=dict(color="black")), row=b + 1, col=1
    )
    fig.add_trace(
        go.Scatter(y=pv_schedule[b], mode="lines", name="PV Supply", line=dict(color="orange")), row=b + 1, col=1
    )
    fig.add_trace(
        go.Scatter(y=charge_schedule[b], mode="lines", name="Battery Charge", line=dict(color="blue")),
        row=b + 1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(y=discharge_schedule[b], mode="lines", name="Battery Discharge", line=dict(color="red")),
        row=b + 1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(y=soc_schedule[b], mode="lines", name="Battery SOC", line=dict(color="purple")), row=b + 1, col=1
    )
    fig.add_trace(
        go.Scatter(y=grid_import_schedule[b], mode="lines", name="Grid Import", line=dict(color="green")),
        row=b + 1,
        col=1,
    )
    fig.update_yaxes(title_text="Power/Energy (kW/kWh)", row=b + 1, col=1)

fig.update_xaxes(title_text="Time step", row=n_buildings, col=1)
fig.update_layout(
    height=400 * n_buildings,
    width=1000,
    title_text="Multi-Building Energy Schedules",
    legend=dict(traceorder="normal"),
)
fig.show()
