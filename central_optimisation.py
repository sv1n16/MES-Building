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


# Load data
radiation = pd.read_csv("data\\radiation.csv").values.flatten()
load = pd.read_csv("data\\load.csv").values.flatten()
price = pd.read_csv("data\\price.csv").values.flatten()
heatload = pd.read_csv("data\\heat_load.csv").values.flatten()
temperature_setpoint = [17, 17, 17.5, 17.5, 17.5, 17.5, 20, 20, 20, 20, 18, 18]
outdoor_temperature = [6.0, 6.17, 6.67, 7.46, 8.5, 9.71, 11.0, 12.29, 13.5, 14.54, 15.33, 15.83]
print(pyo.SolverFactory("gurobi_direct").available())


# Parameters
time_horizon = len(price)  # Number of time steps
delta_t = 1  # Time step in hours
p_th_nom = 8  # Nominal thermal power of the heat pump in kW
initial_temp = 15  # Initial temperature in degrees Celsius
comfort_weight = 10
model = pyo.ConcreteModel()
model.t = pyo.RangeSet(0, time_horizon - 1)


building_parameters = {
    "battery": {
        "capacity": 12,
        "max_power": 4.6,
        "initial_soc": 0.8,
        "eta_charge": 0.9,
        "eta_discharge": 0.9,
    },
    "heatpump": {
        "p_th_nom": 8,
    },
    "thermal": {
        "size": 100.0,  # Example value, set as needed m2
        "construction_type": "medium",  # construction_type: str – 'light', 'medium', or 'heavy'
        "insulation_level": "average",  # insulation_level: str – 'good', 'average', or 'poor'
    },
    "pv": {
        "time_horizon": 12,
        "start_point": 2,  # Example value, set as needed
        "radiation": radiation,
        "area": 25.0,  # Example value, set as needed
        "beta": 30.0,  # Example value, set as needed
        "eta_noct": 0.15,  # Example value, set as needed
    },
    "global": {
        "horizon": 12,
        "delta_t": 1,
    },
}
components = []

# Update battery and building schedules
bat = Battery(**building_parameters["battery"])
pv = PVModule(time_horizon=time_horizon, start_point=2, radiation=radiation, area=25.0, beta=30.0, eta_noct=0.15)
hp = HeatPump(time_horizon=time_horizon)
boiler = GasBoiler(time_horizon=time_horizon)
# components.append(pv)
components.append(hp)
components.append(boiler)
components.append(bat)

bd = Building(
    building_components=components,
    model=model,
    T_out=outdoor_temperature,
    T_init=initial_temp,
    T_set=temperature_setpoint,
    **building_parameters
)

model = bd.get_model()
# Gurobi model


pv.p_el_schedule = -1 * pv.p_el_supply


# model.exclusive_on_constr = pyo.Constraint(model.t, rule=exclusive_on_rule)


# Objective: Minimize total cost
def objective(model):
    return sum(
        price[t] * (load[t] - pv.p_el_supply[t] + model.charge[t] - model.discharge[t] + model.p_el_vars[t]) * delta_t
        + boiler.gas_price * model.gas_consumption[t] * delta_t
        + comfort_weight * (model.T_in[t] - model.T_set[t]) ** 2
        for t in model.t
    )


model.obj = pyo.Objective(rule=objective, sense=pyo.minimize)


solver = pyo.SolverFactory("gurobi_direct")
result = solver.solve(model, tee=True, logfile="Results/Temp_setpoint/solver_log.txt")

log_infeasible_constraints(model)
# Extract results
charge_schedule = [pyo.value(model.charge[t]) for t in model.t]
discharge_schedule = [pyo.value(model.discharge[t]) for t in model.t]
soc_schedule = [pyo.value(model.soc[t]) for t in model.t]

# Debug: Print results
print("Optimized charging schedule:", charge_schedule)
print("Optimized discharging schedule:", discharge_schedule)
print("Optimized SOC schedule:", soc_schedule)

components[0].p_el_charge_schedule = charge_schedule
components[0].p_el_discharge_schedule = discharge_schedule
components[0].energy_el_schedule = soc_schedule
components[0].power_el_schedule = np.array(charge_schedule) - np.array(discharge_schedule)
hp_thermal_output = -np.array([pyo.value(model.q_heat_vars[t]) for t in model.t])
hp_electric_consumption = [pyo.value(model.p_el_vars[t]) for t in model.t]
indoor_temperature = [pyo.value(model.T_in[t]) for t in model.t]

bd.p_el_schedule = load + charge_schedule - discharge_schedule - pv.p_el_supply + hp_electric_consumption

# hp_on_schedule = [pyo.value(model.hp_on[t]) for t in model.t]
# boiler_on_schedule = [pyo.value(model.boiler_on[t]) for t in model.t]
boiler_thermal_output = -np.array([pyo.value(model.q_boiler_vars[t]) for t in model.t])
boiler_gas_consumption = [pyo.value(model.gas_consumption[t]) for t in model.t]
thermal_load_setpoint = [pyo.value(model.q_heat[t]) for t in model.t]
# # Calculate costs
costs = np.array([price[t] * bd.p_el_schedule[t] for t in range(time_horizon)])

# Calculate the cost at each time step
costs = np.array([price[t] * bd.p_el_schedule[t] for t in range(time_horizon)])
gas_costs = np.array(boiler_gas_consumption) * boiler.gas_price
print("Electricity Costs:", sum(costs))
print("Gas Costs:", sum(gas_costs))
total_costs = costs + gas_costs
print("Total costs (electricity + gas):", sum(total_costs))


output_data = {
    "battery_charge_schedule": (
        components[0].p_el_charge.tolist()
        if hasattr(components[0], "p_el_charge")
        else components[0].p_el_charge_schedule
    ),
    "battery_discharge_schedule": (
        components[0].p_el_discharge.tolist()
        if hasattr(components[0], "p_el_discharge")
        else components[0].p_el_discharge_schedule
    ),
    "battery_soc_schedule": components[0].energy_el_schedule,
    "heatpump_thermal_output": hp_thermal_output.tolist(),
    "boiler_thermal_output": boiler_thermal_output.tolist(),
    "indoor_temperature": indoor_temperature,
    "temperature_setpoint": temperature_setpoint,
    "outdoor_temperature": outdoor_temperature,
    "electricity_price": price.tolist(),
    "electricity_costs": costs.tolist(),
    "gas_costs": gas_costs.tolist(),
    "total_costs": total_costs.tolist(),
}

with open("Results/schedules/central_optimisation_schedules_and_costs.json", "w") as f:
    json.dump(output_data, f, indent=2)


plot_time = list(range(time_horizon))
fig = make_subplots(
    rows=8,
    cols=2,
    shared_xaxes=True,
    subplot_titles=(
        "Battery SOC",
        "Temperature (Indoor/Outdoor/Setpoint)",
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
        "Electricity Cost (£/h)",
        "Gas Cost (£/h)",
    ),
)

# Column 1: Electrical
fig.add_trace(go.Scatter(x=plot_time, y=components[0].energy_el_schedule, name="Battery SOC"), row=1, col=1)
fig.add_trace(
    go.Scatter(x=plot_time, y=components[0].power_el_schedule.tolist(), name="Battery Charge/Discharge"), row=2, col=1
)
fig.add_trace(go.Scatter(x=plot_time, y=bd.p_el_schedule.tolist(), name="Building Import/Export"), row=3, col=1)
fig.add_trace(go.Scatter(x=plot_time, y=pv.p_el_schedule.tolist(), name="PV Power Export"), row=4, col=1)
fig.add_trace(
    go.Scatter(x=plot_time, y=load.tolist() if hasattr(load, "tolist") else load, name="Forecasted Load"), row=5, col=1
)
fig.add_trace(
    go.Scatter(
        x=plot_time, y=price.tolist() if hasattr(price, "tolist") else price, name="Energy Market Price (ct/kWh)"
    ),
    row=6,
    col=1,
)
fig.add_trace(go.Scatter(x=plot_time, y=pv.p_el_supply.tolist(), name="PV Supply (kWh)"), row=7, col=1)
fig.add_trace(go.Scatter(x=plot_time, y=costs.tolist(), name="Cost Over Time"), row=8, col=1)

# Column 2: Thermal and costs
fig.add_trace(go.Scatter(x=plot_time, y=indoor_temperature, name="Indoor Temp (°C)"), row=1, col=2)
fig.add_trace(go.Scatter(x=plot_time, y=outdoor_temperature, name="Outdoor Temp (°C)"), row=1, col=2)
fig.add_trace(go.Scatter(x=plot_time, y=temperature_setpoint, name="Setpoint Temp (°C)"), row=1, col=2)

fig.add_trace(go.Scatter(x=plot_time, y=hp_thermal_output.tolist(), name="HP Thermal Output (kWh)"), row=2, col=2)
fig.add_trace(
    go.Scatter(x=plot_time, y=boiler_thermal_output.tolist(), name="Boiler Thermal Output (kWh)"), row=3, col=2
)
fig.add_trace(go.Scatter(x=plot_time, y=thermal_load_setpoint, name="Thermal Load (kWh)"), row=4, col=2)
fig.add_trace(go.Scatter(x=plot_time, y=hp_electric_consumption, name="HP Electrical Consumption (kWh)"), row=5, col=2)
fig.add_trace(go.Scatter(x=plot_time, y=boiler_gas_consumption, name="Boiler Gas Consumption (kWh)"), row=6, col=2)
fig.add_trace(go.Scatter(x=plot_time, y=costs.tolist(), name="Electricity Cost (£/h)"), row=7, col=2)
fig.add_trace(go.Scatter(x=plot_time, y=gas_costs.tolist(), name="Gas Cost (£/h)"), row=8, col=2)

fig.update_yaxes(title_text="Battery SOC (kWh)", row=1, col=1)
fig.update_yaxes(title_text="Charge/Discharge (kW)", row=2, col=1)
fig.update_yaxes(title_text="Import/Export (kW)", row=3, col=1)
fig.update_yaxes(title_text="PV Export (kW)", row=4, col=1)
fig.update_yaxes(title_text="Load (kW)", row=5, col=1)
fig.update_yaxes(title_text="Price (p/kWh)", row=6, col=1)
fig.update_yaxes(title_text="PV Supply (kWh)", row=7, col=1)
fig.update_yaxes(title_text="Total Cost (£)", row=8, col=1)

fig.update_yaxes(title_text="Temperature (°C)", row=1, col=2)
fig.update_yaxes(title_text="HP Thermal Output (kWh)", row=2, col=2)
fig.update_yaxes(title_text="Boiler Thermal Output (kWh)", row=3, col=2)
fig.update_yaxes(title_text="Thermal Load (kWh)", row=4, col=2)
fig.update_yaxes(title_text="HP Electrical Consumption (kWh)", row=5, col=2)
fig.update_yaxes(title_text="Boiler Gas Consumption (kWh)", row=6, col=2)
fig.update_yaxes(title_text="Electricity Cost (£/h)", row=7, col=2)
fig.update_yaxes(title_text="Gas Cost (£/h)", row=8, col=2)

fig.update_layout(
    title_text="Scheduling Results: Local Central Optimisation",
    title_font=dict(size=14),
    height=1200,
    width=1200,
)

for annotation in fig["layout"]["annotations"]:
    annotation["font"] = dict(size=10)

fig.show()
