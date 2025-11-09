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
initial_soc = 5  # kWh
eta_charge = 0.9  # Charging efficiency
eta_discharge = 0.9
p_th_nom = 12  # Nominal thermal power of the heat pump in kW
p_th_nom = 8  # Nominal thermal power of the heat pump in kW
initial_temp = 15  # Initial temperature in degrees Celsius
max_thermal_power = 20.0  # kW, maximum thermal output of boiler
building_parameters = {
    "battery": {
        "capacity": battery_capacity,
        "max_power": max_power,
        "initial_soc": initial_soc,
        "eta_charge": eta_charge,
        "eta_discharge": eta_discharge,
    },
    "heatpump": {
        "p_th_nom": p_th_nom,
    },
    "thermal": {
        "size": 100.0,  # Example value, set as needed m2
        "construction_type": "medium",  # construction_type: str – 'light', 'medium', or 'heavy'
        "insulation_level": "average",  # insulation_level: str – 'good', 'average', or 'poor'
    },
    "pv": {
        "time_horizon": time_horizon,
        "start_point": 2,  # Example value, set as needed
        "radiation": radiation,
        "area": 25.0,  # Example value, set as needed
        "beta": 30.0,  # Example value, set as needed
        "eta_noct": 0.15,  # Example value, set as needed
    },
}

# model

model = pyo.ConcreteModel()
n_buildings = 1  # Set this to the number of buildings you want to model
model.buildings = pyo.RangeSet(0, n_buildings - 1)
model.t = pyo.RangeSet(0, time_horizon - 1)

# Example: Collect initial temperature for each building
initial_temps = {b: initial_temp for b in range(n_buildings)}
model.T_init = pyo.Param(model.buildings, initialize=initial_temps)

# Example: Battery capacity for each building
battery_capacities = {b: battery_capacity for b in range(n_buildings)}
model.battery_capacity = pyo.Param(model.buildings, initialize=battery_capacities)

# Example: Setpoint temperature for each building and time
setpoints = {(b, t): temperature_setpoint[t] for b in range(n_buildings) for t in range(time_horizon)}
model.T_set = pyo.Param(model.buildings, model.t, initialize=setpoints)

# Example: Load for each building and time (assuming same load for all buildings)
loads = {(b, t): electric_load[t] for b in range(n_buildings) for t in range(time_horizon)}
model.electric_load = pyo.Param(model.buildings, model.t, initialize=loads)


# Update battery and building schedules
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

# Example: PV supply for each building and time (assuming same PV for all buildings)
pv_supplies = {(b, t): pv.p_el_supply[t] for b in range(n_buildings) for t in range(time_horizon)}
model.pv_supply = pyo.Param(model.buildings, model.t, initialize=pv_supplies)

# Now, define variables indexed by building and time

# Battery variables


model.charge = pyo.Var(model.buildings, model.t, bounds=(0, max_power))
model.discharge = pyo.Var(model.buildings, model.t, bounds=(0, max_power))
model.soc = pyo.Var(model.buildings, model.t, bounds=(0, battery_capacity))


# Building variables
model.T_in = pyo.Var(model.buildings, model.t)
model.p_el_vars = pyo.Var(model.buildings, model.t)
model.T_set = pyo.Param(model.buildings, model.t, initialize={t: temperature_setpoint[t] for t in range(time_horizon)})
model.T_out = pyo.Param(model.buildings, model.t, initialize={t: outdoor_temperature[t] for t in range(time_horizon)})
model.T_in = pyo.Var(model.buildings, model.t, within=pyo.NonNegativeReals, initialize=initial_temp)
model.q_heat = pyo.Var(model.buildings, model.t, within=pyo.NonNegativeReals)

# Boiler Variables

model.gas_consumption = pyo.Var(model.buildings, model.t)
model.q_boiler_vars = pyo.Var(model.buildings, model.t, bounds=(0, max_thermal_power), initialize=0)

# Heatpump Variables
model.q_heat_vars = pyo.Var(
    model.buildings,
    model.t,
    bounds=(
        0,
        p_th_nom,
    ),
    initialize=0,
)
model.cop = pyo.Var(model.buildings, model.t, initialize=2.2)
model.p_heat_vars = pyo.Var(model.buildings, model.t, bounds=(0, None), initialize=0)
model.f = pyo.Var(model.buildings, model.t, bounds=(0.1, 1), initialize=0.1)


# Define buildings

for building_index in model.buildings:
    #  bat.set_parameters(model, building_index)
    #  bat.set_constraints(model, building_index)
    bd = Building(
        building_components=components,
        model=model,
        T_out=outdoor_temperature,
        T_init=initial_temp,
        T_set=temperature_setpoint,
        building_index=building_index,  # Pass as a named argument if needed
        **building_parameters,
    )


model = bd.get_model()
# Gurobi model


pv.p_el_schedule = -1 * pv.p_el_supply


penalty_weight = 10


# Objective: Minimize total cost for n buildings
print("Buildings:", model.charge[0, 0])
print("Time steps:", list(model.t))


def objective(model):
    return sum(
        price[t]
        * (
            model.electric_load[b, t]
            - model.pv_supply[b, t]
            + model.charge[b, t]
            - model.discharge[b, t]
            + model.p_el_vars[b, t]
        )
        * delta_t
        + boiler.gas_price * model.gas_consumption[b, t] * delta_t
        + penalty_weight * (model.T_in[b, t] - model.T_set[b, t]) ** 2
        for b in model.buildings
        for t in model.t
    )


model.obj = pyo.Objective(rule=objective, sense=pyo.minimize)


solver = pyo.SolverFactory("gurobi_direct")
result = solver.solve(model, tee=True, logfile="Results/Temp_setpoint/solver_log.txt")

log_infeasible_constraints(model)
# Extract results
charge_schedule = [[pyo.value(model.charge[b, t]) for t in model.t] for b in model.buildings]
discharge_schedule = [[pyo.value(model.discharge[b, t]) for t in model.t] for b in model.buildings]
soc_schedule = [[pyo.value(model.soc[b, t]) for t in model.t] for b in model.buildings]

# Debug: Print results
print("Optimized charging schedule:", charge_schedule)
print("Optimized discharging schedule:", discharge_schedule)
print("Optimized SOC schedule:", soc_schedule)

bat.p_el_charge_schedule = charge_schedule
bat.p_el_discharge_schedule = discharge_schedule
bat.energy_el_schedule = soc_schedule
bat.power_el_schedule = np.array(charge_schedule) - np.array(discharge_schedule)
hp_thermal_output = -np.array([pyo.value(model.q_heat_vars[t]) for t in model.t])
hp_electric_consumption = [pyo.value(model.p_el_vars[t]) for t in model.t]
indoor_temperature = [pyo.value(model.T_in[t]) for t in model.t]

bd.p_el_schedule = electric_load + charge_schedule - discharge_schedule - pv.p_el_supply + hp_electric_consumption

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
    "battery_charge_schedule": (bat.p_el_charge.tolist() if hasattr(bat, "p_el_charge") else bat.p_el_charge_schedule),
    "battery_discharge_schedule": (
        bat.p_el_discharge.tolist() if hasattr(bat, "p_el_discharge") else bat.p_el_discharge_schedule
    ),
    "battery_soc_schedule": bat.energy_el_schedule,
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

# with open("Results/schedules/central_optimisation_schedules_and_costs.json", "w") as f:
#     json.dump(output_data, f, indent=2)
