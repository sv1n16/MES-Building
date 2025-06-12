from gurobipy import Model, GRB
import pyomo.environ as pyo

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from src.Classes.boiler import GasBoiler
from src.Classes.battery import Battery
from src.Classes.photovoltaic import PVModule
from src.Classes.building import Building
from src.Classes.heatpump import HeatPump

# Load data
radiation = pd.read_csv("data\\radiation.csv").values.flatten()
load = pd.read_csv("data\\load.csv").values.flatten()
price = pd.read_csv("data\\price.csv").values.flatten()
heatload = pd.read_csv("data\\heat_load.csv").values.flatten()
print(pyo.SolverFactory("gurobi_direct").available())
# Parameters
time_horizon = len(price)  # Number of time steps
delta_t = 1  # Time step in hours
battery_capacity = 12.0  # kWh
max_power = 4.6  # kW
initial_soc = 5  # kWh
eta_charge = 0.9  # Charging efficiency
eta_discharge = 0.9
p_th_nom = 8  # Nominal thermal power of the heat pump in kW
# Update battery and building schedules
bat = Battery()
pv = PVModule(time_horizon=time_horizon, start_point=2, radiation=radiation, area=25.0, beta=30.0, eta_noct=0.15)
hp = HeatPump(time_horizon=time_horizon)
boiler = GasBoiler(time_horizon=time_horizon)
bd = Building([bat, pv])
# Gurobi model
model = pyo.ConcreteModel()
model.t = pyo.RangeSet(0, time_horizon - 1)

hp.p_th_heat = heatload
boiler.p_th_heat = heatload
hp.set_parameters(model)
hp.set_constraints(model)

bat.set_parameters(model)
bat.set_constraints(model)

boiler.set_parameters(model)
boiler.set_constraints(model)
pv.p_el_schedule = -1 * pv.p_el_supply


# Exclusivity: can't be on at the same time
def exclusive_on_rule(model, t):
    return model.hp_on[t] + model.boiler_on[t] <= 1


model.exclusive_on_constr = pyo.Constraint(model.t, rule=exclusive_on_rule)


def heat_demand_match_rule(model, t):
    return -model.p_th_heat_vars[t] - model.p_th_boiler_vars[t] == heatload[t]


model.heat_demand_match = pyo.Constraint(model.t, rule=heat_demand_match_rule)


# Objective: Minimize total cost
def objective(model):
    return sum(
        price[t] * (load[t] - pv.p_el_supply[t] + model.charge[t] - model.discharge[t] + model.p_el_vars[t]) * delta_t
        + boiler.gas_price * model.gas_consumption[t] * delta_t
        for t in model.t
    )


model.obj = pyo.Objective(rule=objective, sense=pyo.minimize)


solver = pyo.SolverFactory("gurobi_direct")
result = solver.solve(model, tee=True)

# Extract results
charge_schedule = [pyo.value(model.charge[t]) for t in model.t]
discharge_schedule = [pyo.value(model.discharge[t]) for t in model.t]
soc_schedule = [pyo.value(model.soc[t]) for t in model.t]

# Debug: Print results
print("Optimized charging schedule:", charge_schedule)
print("Optimized discharging schedule:", discharge_schedule)
print("Optimized SOC schedule:", soc_schedule)

bat.p_el_charge_schedule = charge_schedule
bat.p_el_discharge_schedule = discharge_schedule
bat.energy_el_schedule = soc_schedule
bat.power_el_schedule = np.array(charge_schedule) - np.array(discharge_schedule)
hp_thermal_output = -np.array([pyo.value(model.p_th_heat_vars[t]) for t in model.t])
hp_electric_consumption = [pyo.value(model.p_el_vars[t]) for t in model.t]
bd.p_el_schedule = load + charge_schedule - discharge_schedule - pv.p_el_supply + hp_electric_consumption

hp_on_schedule = [pyo.value(model.hp_on[t]) for t in model.t]
boiler_on_schedule = [pyo.value(model.boiler_on[t]) for t in model.t]
boiler_thermal_output = -np.array([pyo.value(model.p_th_boiler_vars[t]) for t in model.t])
boiler_gas_consumption = [pyo.value(model.gas_consumption[t]) for t in model.t]

# # Calculate costs
costs = np.array([price[t] * bd.p_el_schedule[t] for t in range(time_horizon)])

# Calculate the cost at each time step
costs = np.array([price[t] * bd.p_el_schedule[t] for t in range(time_horizon)])
gas_costs = np.array(boiler_gas_consumption) * boiler.gas_price
print("Electricity Costs:", sum(costs))
print("Gas Costs:", sum(gas_costs))
total_costs = costs + gas_costs
print("Total costs (electricity + gas):", sum(total_costs))


plot_time = list(range(time_horizon))
fig = make_subplots(
    rows=9,
    cols=2,
    shared_xaxes=True,
    subplot_titles=(
        "Battery SOC",
        "HP/Boiler On/Off",
        "Battery Charge/Discharge",
        "HP Thermal Output",
        "Building Import/Export",
        "Boiler Thermal Output",
        "PV Power Export",
        "Thermal Load",
        "Forecasted Load",
        "HP Electrical Consumption",
        "Energy Market Price",
        "Boiler Gas Consumption",
        "PV Supply",
        "Cost Over Time",
        "",
    ),
)

# Column 1: Electrical
fig.add_trace(go.Scatter(x=plot_time, y=bat.energy_el_schedule, name="Battery SOC"), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_time, y=bat.power_el_schedule, name="Battery Charge/Discharge"), row=2, col=1)
fig.add_trace(go.Scatter(x=plot_time, y=bd.p_el_schedule, name="Building Import/Export"), row=3, col=1)
fig.add_trace(go.Scatter(x=plot_time, y=pv.p_el_schedule, name="PV Power Export"), row=4, col=1)
fig.add_trace(go.Scatter(x=plot_time, y=load, name="Forecasted Load"), row=5, col=1)
fig.add_trace(go.Scatter(x=plot_time, y=price, name="Energy Market Price (ct/kWh)"), row=6, col=1)
fig.add_trace(go.Scatter(x=plot_time, y=pv.p_el_supply, name="PV Supply (kWh)"), row=7, col=1)
fig.add_trace(go.Scatter(x=plot_time, y=costs, name="Cost Over Time"), row=8, col=1)


fig.add_trace(go.Scatter(x=plot_time, y=hp_on_schedule, name="HP On (binary)"), row=1, col=2)
fig.add_trace(go.Scatter(x=plot_time, y=boiler_on_schedule, name="Boiler On (binary)"), row=1, col=2)
fig.add_trace(go.Scatter(x=plot_time, y=hp_thermal_output, name="HP Thermal Output (kWh)"), row=2, col=2)
fig.add_trace(go.Scatter(x=plot_time, y=boiler_thermal_output, name="Boiler Thermal Output (kWh)"), row=3, col=2)
fig.add_trace(go.Scatter(x=plot_time, y=heatload, name="Thermal Load (kWh)"), row=4, col=2)
fig.add_trace(go.Scatter(x=plot_time, y=hp_electric_consumption, name="HP Electrical Consumption (kWh)"), row=5, col=2)
fig.add_trace(go.Scatter(x=plot_time, y=boiler_gas_consumption, name="Boiler Gas Consumption (kWh)"), row=6, col=2)

# # Update xaxis properties
# for i in range(1, 9):
#     fig.update_xaxes(title_text="Time", row=i, col=1, title_font=dict(size=8))
#     fig.update_xaxes(title_text="Time", row=i, col=2, title_font=dict(size=8))
# Update yaxis properties
for row in range(1, 10):  # rows 1 to 9
    for col in range(1, 3):  # columns 1 and 2
        fig.update_yaxes(title_font=dict(size=8), row=row, col=col)

fig.update_yaxes(title_text="Battery", row=1, col=1)
fig.update_yaxes(title_text="Building (kWh)", row=2, col=1)
fig.update_yaxes(title_text="PV (kWh)", row=3, col=1)
fig.update_yaxes(title_text="Load (kWh)", row=4, col=1)
fig.update_yaxes(title_text="Energy Price", row=5, col=1)
fig.update_yaxes(title_text="PV Generation", row=6, col=1)
fig.update_yaxes(title_text="Cost (ct)", row=7, col=1)
fig.update_yaxes(title_text="Thermal (kWh)", row=8, col=1)
fig.update_yaxes(title_text="HP Elec (kWh)", row=9, col=1)

fig.update_layout(title_text="Scheduling Results Local Central Optimisation")
fig.update_layout(uniformtext_minsize=8)

fig.update_layout(
    title_text="Scheduling Results Local Central Optimisation",
    title_font=dict(size=12),
    uniformtext_minsize=8,
    height=900,  # Increase the figure height (default is ~450-600)
    width=1200,  # Optionally increase the width as well
)

for annotation in fig["layout"]["annotations"]:
    annotation["font"] = dict(size=8)  # or any size you prefer


fig.show()
