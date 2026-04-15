import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from src.Classes.boiler import GasBoiler
from src.Classes.battery import Battery
from src.Classes.photovoltaic import PVModule
from src.Classes.building import Building
from src.Classes.heatpump import HeatPump

# This is a simple power scheduling example to demonstrate the integration and interaction of PV and battery storage
# systems using the central optimization algorithm.


# Generate a schedule that charges the battery at night and discharges during the day
p_el_charge = np.array([4.6, 3.2, 0.0, 4.6, 4.6, 4.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # charging schedule
p_el_discharge = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.6, 4.6, 4.6, 4.6, 4.6, 4.6])  # discharge schedule

# Limit discharge to maximum power allowed
max_discharge_power = 4.6  # kW - adjust this to limit discharge
p_el_discharge = np.minimum(p_el_discharge, max_discharge_power)
p_el_charge = np.minimum(p_el_charge, max_discharge_power)

time_horizon = 12  # 12 hours
delta_t = 1  # 1 hour time step
radiation = pd.read_csv("data\\radiation.csv").values.flatten()
load = pd.read_csv("data\\load.csv").values.flatten()
price = pd.read_csv("data\\price.csv").values.flatten() / 100.0  # convert p/kWh -> £/kWh
heat_load = pd.read_csv("data\\heat_load.csv").values.flatten()

# Initialize battery, PV, heat pump, and boiler components
bat = Battery(p_el_demand=p_el_charge, p_el_supply=p_el_discharge)
pv = PVModule(
    time_horizon=time_horizon,
    start_point=2,
    radiation=radiation,
    area=25.0,
    beta=30.0,
    eta_noct=0.15,
)
eh = HeatPump(time_horizon=time_horizon)
boiler = GasBoiler(time_horizon=time_horizon)
bd = Building(building_components=[bat, pv, eh, boiler])

# Update the battery schedule
bat.energy_el_schedule = bat.battery_energy_schedule(time_horizon, delta_t)
bat.power_el_schedule = p_el_charge - p_el_discharge
pv.p_el_schedule = -1 * pv.p_el_supply

# Generate Heatpump schedule
p_th_heat_hp = np.zeros(time_horizon)
p_th_heat_boiler = np.zeros(time_horizon)

p_th_heat_hp[:5] = heat_load[:5]
p_th_heat_boiler[5:] = heat_load[5:]

# update the heatpump schedule
eh.p_th_heat = p_th_heat_hp  # Set the heat pump's thermal output schedule
eh.p_el_heat = [abs(p) / eh.cop[t] if eh.cop[t] > 0 else 0 for t, p in enumerate(p_th_heat_hp)]
eh.p_el_schedule = eh.p_el_heat  # Set the heat pump's electrical power schedule


boiler.set_thermal_output(p_th_heat_boiler)

# Update the building's power schedule to include the battery schedule and PV schedule
bd.p_el_schedule = load + bat.power_el_schedule + pv.p_el_schedule + eh.p_el_schedule

# Debug: Print the battery's power schedule
print("Battery's power schedule (p_el_schedule):")
print(bat.power_el_schedule)


# Temperature setpoint (example, adjust as needed)
temperature_setpoint = np.array([17, 17, 17.5, 17.5, 17.5, 17.5, 20, 20, 20, 20, 18, 18])
outdoor_temperature = np.array([6.0, 6.17, 6.67, 7.46, 8.5, 9.71, 11.0, 12.29, 13.5, 14.54, 15.33, 15.83])
initial_temp = 15.0
C = 10.0  # Building thermal capacity (kWh/°C), adjust as needed
U = 0.5  # Building thermal conductance (kW/°C), adjust as needed
hp_max = 8.0  # Heat pump max thermal power (kW)
boiler_max = 20.0  # Boiler max thermal power (kW)

indoor_temperature = np.zeros(time_horizon)
indoor_temperature[0] = initial_temp
required_heat = np.zeros(time_horizon)
p_th_heat_hp = np.zeros(time_horizon)
p_th_heat_boiler = np.zeros(time_horizon)

for t in range(time_horizon):
    T_prev = indoor_temperature[t - 1] if t > 0 else initial_temp
    T_set = temperature_setpoint[t]
    T_out = outdoor_temperature[t]
    # Calculate required heat to reach setpoint
    required_heat[t] = C * (T_set - T_prev) / delta_t + U * (T_set - T_out)
    required_heat[t] = max(required_heat[t], 0)  # Only heat, no cooling
    # Assign heat pump and boiler outputs
    if required_heat[t] <= hp_max:
        p_th_heat_hp[t] = required_heat[t]
        p_th_heat_boiler[t] = 0
    else:
        p_th_heat_hp[t] = hp_max
        p_th_heat_boiler[t] = required_heat[t] - hp_max
        p_th_heat_boiler[t] = min(p_th_heat_boiler[t], boiler_max)
    # Simulate indoor temperature for next step
    if t < time_horizon - 1:
        indoor_temperature[t + 1] = T_prev + delta_t / C * (
            p_th_heat_hp[t] + p_th_heat_boiler[t] - U * (T_prev - T_out)
        )

# update the heatpump and boiler schedules
eh.p_th_heat = p_th_heat_hp
boiler.set_thermal_output(p_th_heat_boiler)


# Calculate the cost at each time step (match central optimisation)
# Only charge for positive grid imports, not exports
total_electricity = load + bat.power_el_schedule - pv.p_el_supply + eh.p_el_schedule
costs = np.array([price[t] * max(0, total_electricity[t]) for t in range(time_horizon)])
gas_costs = np.array(boiler.gas_consumption_schedule) * boiler.gas_price
total_costs = costs + gas_costs
print("Electricity Costs:", sum(costs))
print("Gas Costs:", sum(gas_costs))
print("Total costs (electricity + gas):", sum(total_costs))


# save results for analysis
import json

output_data = {
    "battery_charge_schedule": p_el_charge.tolist(),
    "battery_discharge_schedule": p_el_discharge.tolist(),
    "battery_soc_schedule": bat.energy_el_schedule.tolist(),
    "heatpump_thermal_output": p_th_heat_hp.tolist(),
    "boiler_thermal_output": p_th_heat_boiler.tolist(),
    "indoor_temperature": indoor_temperature.tolist(),
    "temperature_setpoint": temperature_setpoint.tolist(),
    "outdoor_temperature": outdoor_temperature.tolist(),
    "electricity_price": price.tolist(),
    "electricity_costs": costs.tolist(),
    "gas_costs": gas_costs.tolist(),
    "total_costs": total_costs.tolist(),
    "load": load.tolist(),
    "pv_supply": pv.p_el_supply.tolist(),
    "battery_net_charge": bat.power_el_schedule.tolist(),
    "heatpump_electrical": eh.p_el_schedule,
    "required_heat": required_heat.tolist(),
    "grid_import": bd.p_el_schedule.tolist(),
}

with open("Results/schedules/open_loop_schedules_and_costs.json", "w") as f:
    json.dump(output_data, f, indent=2)


# Plot the cost over the simulation horizon
plot_time = list(range(time_horizon))
fig = make_subplots(
    rows=9,
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

# Column 2: Thermal, cost, and gas
fig.add_trace(go.Scatter(x=plot_time, y=indoor_temperature, name="Indoor Temp (°C)"), row=1, col=2)
fig.add_trace(go.Scatter(x=plot_time, y=outdoor_temperature, name="Outdoor Temp (°C)"), row=1, col=2)
fig.add_trace(go.Scatter(x=plot_time, y=temperature_setpoint, name="Setpoint Temp (°C)"), row=1, col=2)
fig.add_trace(go.Scatter(x=plot_time, y=p_th_heat_hp, name="HP Thermal Output (kWh)"), row=2, col=2)
fig.add_trace(go.Scatter(x=plot_time, y=p_th_heat_boiler, name="Boiler Thermal Output (kWh)"), row=2, col=2)
fig.add_trace(go.Scatter(x=plot_time, y=required_heat, name="Thermal Load (kWh)"), row=2, col=2)
fig.add_trace(go.Scatter(x=plot_time, y=eh.p_el_schedule, name="HP Electrical Consumption (kWh)"), row=3, col=2)
fig.add_trace(
    go.Scatter(x=plot_time, y=boiler.gas_consumption_schedule, name="Boiler Gas Consumption (kWh)"), row=4, col=2
)
fig.add_trace(go.Scatter(x=plot_time, y=costs, name="Electricity Cost (£/h)"), row=5, col=2)
fig.add_trace(go.Scatter(x=plot_time, y=gas_costs, name="Gas Cost (£/h)"), row=6, col=2)
fig.add_trace(go.Scatter(x=plot_time, y=total_costs, name="Total Cost (£/h)"), row=7, col=2)
fig.add_trace(go.Scatter(x=plot_time, y=bd.p_el_schedule, name="Electricity Consumption (kWh)"), row=8, col=2)
fig.add_trace(go.Scatter(x=plot_time, y=boiler.gas_consumption_schedule, name="Gas Consumption (kWh)"), row=9, col=2)

# Update yaxis properties for clarity
for row in range(1, 10):
    for col in range(1, 3):
        fig.update_yaxes(title_font=dict(size=8), row=row, col=col)

fig.update_yaxes(title_text="Battery", row=1, col=1)
fig.update_yaxes(title_text="Building (kWh)", row=2, col=1)
fig.update_yaxes(title_text="PV (kWh)", row=3, col=1)
fig.update_yaxes(title_text="Load (kWh)", row=4, col=1)
fig.update_yaxes(title_text="Energy Price", row=5, col=1)
fig.update_yaxes(title_text="PV Generation", row=6, col=1)
fig.update_yaxes(title_text="Cost (ct)", row=7, col=1)
fig.update_yaxes(title_text="Thermal (kWh)", row=2, col=1)
fig.update_yaxes(title_text="HP Elec (kWh)", row=9, col=1)
fig.update_yaxes(title_text="Temperature (°C)", row=1, col=2)
fig.update_yaxes(title_text="Electricity Cost (£/h)", row=5, col=2)
fig.update_yaxes(title_text="Gas Cost (£/h)", row=6, col=2)
fig.update_yaxes(title_text="Total Cost (£/h)", row=7, col=2)
fig.update_yaxes(title_text="Electricity (kWh)", row=8, col=2)
fig.update_yaxes(title_text="Gas (kWh)", row=9, col=2)

fig.update_layout(
    title_text="Scheduling Results Local Open Loop",
    title_font=dict(size=12),
    uniformtext_minsize=8,
    height=900,
    width=1200,
)
for annotation in fig["layout"]["annotations"]:
    annotation["font"] = dict(size=8)
fig.show()
# ============================================================================
# PLOT IN CENTRAL OPTIMIZATION FORMAT (3 COLUMNS)
# ============================================================================

# Prepare electricity consumption cost schedule (matching central optimization calculation)
grid_import = bd.p_el_schedule
elec_cost_schedule = [price[t] * grid_import[t] * delta_t for t in range(time_horizon)]
total_cost_schedule = [elec_cost_schedule[t] + gas_costs[t] for t in range(time_horizon)]

# Create figure with 3 columns format (matching central_optimisation_multi_building_electric_hp_boiler_ideal.py)
fig_central_format = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=("Battery and Heat Pump Operation", "Building Temperatures", "Costs"),
    shared_xaxes=True,
    shared_yaxes=False,
)

plot_time_list = list(range(time_horizon))

# Column 1: Battery and Heat Pump Operation
fig_central_format.add_trace(
    go.Scatter(
        y=load, mode="lines", name="Load (kW)", line=dict(color="black"), showlegend=True
    ),
    row=1,
    col=1,
)
fig_central_format.add_trace(
    go.Scatter(
        y=pv.p_el_supply, mode="lines", name="PV Supply (kW)", line=dict(color="orange"), showlegend=True
    ),
    row=1,
    col=1,
)
fig_central_format.add_trace(
    go.Scatter(
        y=bat.power_el_schedule,
        mode="lines",
        name="Battery Net Charge (kW)",
        line=dict(color="blue"),
        showlegend=True,
    ),
    row=1,
    col=1,
)
fig_central_format.add_trace(
    go.Scatter(
        y=bat.energy_el_schedule, mode="lines", name="Battery SOC (kWh)", line=dict(color="purple"), showlegend=True
    ),
    row=1,
    col=1,
)
fig_central_format.add_trace(
    go.Scatter(
        y=eh.p_el_schedule, mode="lines", name="Heat Pump (kW)", line=dict(color="green"), showlegend=True
    ),
    row=1,
    col=1,
)

# Column 2: Building Temperatures and Thermal Outputs
fig_central_format.add_trace(
    go.Scatter(y=indoor_temperature, mode="lines", name="Indoor Temp (°C)", line=dict(color="firebrick"), showlegend=True),
    row=1,
    col=2,
)
fig_central_format.add_trace(
    go.Scatter(
        y=outdoor_temperature, mode="lines", name="Outdoor Temp (°C)", line=dict(color="deepskyblue"), showlegend=True
    ),
    row=1,
    col=2,
)
fig_central_format.add_trace(
    go.Scatter(
        y=temperature_setpoint, mode="lines", name="Setpoint (°C)", line=dict(color="darkgreen", dash="dash"), showlegend=True
    ),
    row=1,
    col=2,
)
fig_central_format.add_trace(
    go.Scatter(
        y=p_th_heat_boiler,
        mode="lines",
        name="Boiler Output (kW)",
        line=dict(color="red", width=2),
        showlegend=True,
    ),
    row=1,
    col=2,
)
fig_central_format.add_trace(
    go.Scatter(
        y=p_th_heat_hp,
        mode="lines",
        name="Heat Pump Output (kW)",
        line=dict(color="cyan", width=2),
        showlegend=True,
    ),
    row=1,
    col=2,
)
fig_central_format.add_trace(
    go.Scatter(
        y=p_th_heat_hp + p_th_heat_boiler,
        mode="lines",
        name="Total Heat Demand (kW)",
        line=dict(color="black", dash="dash"),
        showlegend=True,
    ),
    row=1,
    col=2,
)

# Column 3: Costs and Electricity Price
fig_central_format.add_trace(
    go.Scatter(
        y=price,
        mode="lines",
        name="Electricity Price (£/kWh)",
        line=dict(color="pink"),
        showlegend=True,
    ),
    row=1,
    col=3,
)
fig_central_format.add_trace(
    go.Scatter(
        y=elec_cost_schedule,
        mode="lines",
        name="Electricity Consumption Cost",
        line=dict(color="gold"),
        showlegend=True,
    ),
    row=1,
    col=3,
)
fig_central_format.add_trace(
    go.Scatter(
        y=gas_costs,
        mode="lines",
        name="Gas Consumption Cost",
        line=dict(color="brown"),
        showlegend=True,
    ),
    row=1,
    col=3,
)

# Axis labels
fig_central_format.update_xaxes(title_text="Time", row=1, col=1)
fig_central_format.update_yaxes(title_text="Power/Energy (kW/kWh)", title_font=dict(size=10), row=1, col=1)

fig_central_format.update_xaxes(title_text="Time", row=1, col=2)
fig_central_format.update_yaxes(title_text="Temperature (°C)", title_font=dict(size=10), row=1, col=2)

fig_central_format.update_xaxes(title_text="Time", row=1, col=3)
fig_central_format.update_yaxes(title_text="Cost (£)", title_font=dict(size=10), row=1, col=3)

fig_central_format.update_layout(
    title_text="Open Loop Optimization Results - Central Format",
    width=1400,
    height=500,
    hovermode="x unified",
)

fig_central_format.show()