import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import pandas as pd
import json
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from src.Classes.boiler import GasBoiler
from src.Classes.battery import Battery
from src.Classes.photovoltaic import PVModule
from src.Classes.building import Building
from src.Classes.heatpump import HeatPump

# Single building open loop scheduling using central optimization dataset

# ============================================================================
# LOAD DATA - Same as central optimization file
# ============================================================================

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

# --- aggregate to hourly ---
data_hr = pd.DataFrame()
data_hr["price"] = data.groupby("hour")["price"].mean()
data_hr["pv"] = data.groupby("hour")["pv"].mean()
data_hr["outdoor temperature"] = data.groupby("hour")["outdoor temperature"].mean()
data_hr["temperature setpoint"] = data.groupby("hour")["temperature setpoint"].mean()

for c in cols:
    data_hr[c] = data.groupby("hour")[c].max()

# Parameters
time_horizon = len(data_hr)
delta_t = 1  # Time step in hours
battery_capacity = 12.0  # kWh
max_power = 4.6  # kW
eta_charge = 0.9  # Charging efficiency
eta_discharge = 0.9
p_th_nom = 12
T_ref = 7
cop = np.ones(time_horizon) * 2.18
T_init = 20.0
max_thermal_power = 20.0  # kW
efficiency = 0.9  # Boiler efficiency (fraction)
gas_price = 5  # Gas price (p/kWh)
hp_max_power = 10.0  # Maximum heat pump electrical power (kW)

# Thermal parameters
C = 10.0  # Building thermal capacity (kWh/°C)
U = 0.5  # Building thermal conductance (kW/°C)
price = data_hr["price"].values
radiation = data_hr["pv"].values / 1000  # Convert to kW
outdoor_temperature = data_hr["outdoor temperature"].values
temperature_setpoint = data_hr["temperature setpoint"].values

# Generate charge/discharge schedules based on time of day (same for all buildings)
p_el_charge = np.zeros(time_horizon)
p_el_discharge = np.zeros(time_horizon)

# Charge during night (0-6) and early morning (22-23)
p_el_charge[(data_hr.index < 6) | (data_hr.index >= 22)] = max_power

# Discharge during day peak hours (16-20)
p_el_discharge[(data_hr.index >= 16) & (data_hr.index < 20)] = max_power

p_el_charge = np.minimum(p_el_charge, max_power)
p_el_discharge = np.minimum(p_el_discharge, max_power)

# Calculate heat demand (same for all buildings)
heat_load = np.maximum(0, 2.0 * (temperature_setpoint - outdoor_temperature) / 10)

# ============================================================================
# SINGLE BUILDING SETUP (First building for visualization)
# ============================================================================

building_col = cols[0]
load = data_hr[building_col].values / 1000  # Convert to kW

bat = Battery(p_el_demand=p_el_charge, p_el_supply=p_el_discharge)
pv = PVModule(
    time_horizon=time_horizon,
    start_point=0,
    radiation=radiation,
    area=25.0,
    beta=30.0,
    eta_noct=0.15,
)
eh = HeatPump(time_horizon=time_horizon)
boiler = GasBoiler(time_horizon=time_horizon)
bd = Building(building_components=[bat, pv, eh, boiler])

# Update battery schedule
bat.energy_el_schedule = bat.battery_energy_schedule(time_horizon, delta_t)
bat.power_el_schedule = p_el_charge - p_el_discharge
pv.p_el_schedule = -1 * pv.p_el_supply

# Generate Heatpump and Boiler schedule
p_th_heat_hp = np.zeros(time_horizon)
p_th_heat_boiler = np.zeros(time_horizon)

# Simple strategy: use heat pump when available, boiler as backup
for t in range(time_horizon):
    if heat_load[t] > 0:
        if heat_load[t] <= 5.0:  # Heat pump capacity
            p_th_heat_hp[t] = heat_load[t]
            p_th_heat_boiler[t] = 0
        else:
            p_th_heat_hp[t] = 5.0
            p_th_heat_boiler[t] = min(heat_load[t] - 5.0, 15.0)

eh.p_th_heat = p_th_heat_hp
eh.p_el_heat = [abs(p) / eh.cop[t] if eh.cop[t] > 0 else 0 for t, p in enumerate(p_th_heat_hp)]
eh.p_el_schedule = eh.p_el_heat

boiler.set_thermal_output(p_th_heat_boiler)

# Update building's power schedule
bd.p_el_schedule = load + bat.power_el_schedule + pv.p_el_schedule + eh.p_el_schedule

# ============================================================================
# THERMAL DYNAMICS
# ============================================================================

initial_temp = T_init
# Building thermal dynamics
indoor_temperature = np.zeros(time_horizon)
indoor_temperature[0] = initial_temp

required_heat = np.zeros(time_horizon)

for t in range(time_horizon):
    T_prev = indoor_temperature[t] if t == 0 else indoor_temperature[t]
    T_set = temperature_setpoint[t]
    T_out = outdoor_temperature[t]

    # Calculate heat required to reach setpoint
    required_heat[t] = C * (T_set - T_prev) / delta_t + U * (T_set - T_out)
    required_heat[t] = max(required_heat[t], 0)

    # Allocate between heat pump and boiler
    if required_heat[t] <= hp_max_power:
        p_th_heat_hp[t] = required_heat[t]
        p_th_heat_boiler[t] = 0
    else:
        p_th_heat_hp[t] = hp_max_power
        p_th_heat_boiler[t] = min(required_heat[t] - hp_max_power, max_thermal_power)

    # Update temperature for next step
    if t < time_horizon - 1:
        indoor_temperature[t + 1] = T_prev + delta_t / C * (
            p_th_heat_hp[t] + p_th_heat_boiler[t] - U * (T_prev - T_out)
        )

eh.p_th_heat = p_th_heat_hp
boiler.set_thermal_output(p_th_heat_boiler)

# ============================================================================
# COSTS CALCULATION
# ============================================================================

# Building costs (only charge for positive imports, not exports)
total_electricity = load + bat.power_el_schedule - pv.p_el_supply + eh.p_el_schedule
costs = np.array([price[t] * max(0, total_electricity[t]) for t in range(time_horizon)])
gas_costs = np.array(boiler.gas_consumption_schedule) * boiler.gas_price
total_costs = costs + gas_costs

print("\n=== SINGLE BUILDING (From Central Optimization Dataset) ===")
print(f"Time horizon: {time_horizon} hours")
print(f"Building: {building_col}")
print(f"Electricity Costs: £{np.sum(costs):.2f}")
print(f"Gas Costs: £{np.sum(gas_costs):.2f}")
print(f"Total costs (electricity + gas): £{np.sum(total_costs):.2f}")
print(f"Average indoor temperature: {np.mean(indoor_temperature):.2f}°C")
print(f"Outdoor temperature range: {np.min(outdoor_temperature):.2f}°C - {np.max(outdoor_temperature):.2f}°C")

# ============================================================================
# PLOT IN SINGLE BUILDING FORMAT (2 ROWS x 2 COLUMNS)
# ============================================================================

# Create figure with 2 rows x 2 columns format
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "Battery and Heat Pump Operation",
        "Building Temperatures",
        "Temperature Deviation",
        "Costs",
    ),
    shared_xaxes=True,
    shared_yaxes=False,
)

# ============================================================================
# COLUMN 1: Battery and Heat Pump Operation
# ============================================================================
fig.add_trace(
    go.Scatter(y=load, mode="lines", name="Load (kW)", line=dict(color="black"), showlegend=True),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(y=pv.p_el_supply, mode="lines", name="PV Supply (kW)", line=dict(color="orange"), showlegend=True),
    row=1,
    col=1,
)
fig.add_trace(
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
fig.add_trace(
    go.Scatter(
        y=bat.energy_el_schedule, mode="lines", name="Battery SOC (kWh)", line=dict(color="purple"), showlegend=True
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(y=eh.p_el_schedule, mode="lines", name="Heat Pump (kW)", line=dict(color="green"), showlegend=True),
    row=1,
    col=1,
)

# ============================================================================
# COLUMN 2: Building Temperatures and Thermal Outputs
# ============================================================================
fig.add_trace(
    go.Scatter(
        y=indoor_temperature, mode="lines", name="Indoor Temp (°C)", line=dict(color="firebrick"), showlegend=True
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter(
        y=outdoor_temperature, mode="lines", name="Outdoor Temp (°C)", line=dict(color="deepskyblue"), showlegend=True
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter(
        y=temperature_setpoint,
        mode="lines",
        name="Setpoint (°C)",
        line=dict(color="darkgreen", dash="dash"),
        showlegend=True,
    ),
    row=1,
    col=2,
)
fig.add_trace(
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
fig.add_trace(
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
fig.add_trace(
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

# ============================================================================
# COLUMN 3 (Row 2, Col 1): Temperature Deviation
# ============================================================================
temperature_deviation = indoor_temperature - temperature_setpoint

fig.add_trace(
    go.Scatter(
        y=temperature_deviation,
        mode="lines",
        name="Temp Deviation (°C)",
        line=dict(color="red", width=2),
        showlegend=True,
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        y=np.zeros(len(temperature_deviation)),
        mode="lines",
        name="Zero Deviation",
        line=dict(color="black", dash="dash"),
        showlegend=True,
    ),
    row=2,
    col=1,
)

# ============================================================================
# COLUMN 4 (Row 2, Col 2): Costs and Electricity Price
# ============================================================================
fig.add_trace(
    go.Scatter(
        y=price,
        mode="lines",
        name="Electricity Price (£/kWh)",
        line=dict(color="pink"),
        showlegend=True,
    ),
    row=2,
    col=2,
)
fig.add_trace(
    go.Scatter(
        y=costs,
        mode="lines",
        name="Electricity Consumption Cost",
        line=dict(color="gold"),
        showlegend=True,
    ),
    row=2,
    col=2,
)
fig.add_trace(
    go.Scatter(
        y=gas_costs,
        mode="lines",
        name="Gas Consumption Cost",
        line=dict(color="brown"),
        showlegend=True,
    ),
    row=2,
    col=2,
)

# Update layout
fig.update_xaxes(title_text="Time (hours)", row=1, col=1)
fig.update_yaxes(title_text="Power/Energy (kW/kWh)", row=1, col=1)

fig.update_xaxes(title_text="Time (hours)", row=1, col=2)
fig.update_yaxes(title_text="Temperature (°C)", row=1, col=2)

fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
fig.update_yaxes(title_text="Deviation (°C)", row=2, col=1)

fig.update_xaxes(title_text="Time (hours)", row=2, col=2)
fig.update_yaxes(title_text="Cost (£)", row=2, col=2)

fig.update_layout(
    title_text="Open Loop Single Building Operation (Central Optimization Dataset)",
    height=800,
    showlegend=True,
)

fig.show()

# ============================================================================
# SAVE SINGLE BUILDING RESULTS TO JSON WITH PARAMETERS
# ============================================================================

# Calculate peak demand metrics for single building
grid_import = load + bat.power_el_schedule - pv.p_el_supply + eh.p_el_schedule

peak_electricity_demand = np.max(grid_import)
time_of_peak_electricity = int(np.argmax(grid_import))
average_electricity_demand = np.mean(grid_import)
peak_heat_pump_output = np.max(p_th_heat_hp)

single_building_data = {
    "parameters": {
        "time_horizon_hours": int(time_horizon),
        "delta_t_hours": float(delta_t),
        "battery_capacity_kwh": float(battery_capacity),
        "max_power_kw": float(max_power),
        "eta_charge": float(eta_charge),
        "eta_discharge": float(eta_discharge),
        "p_th_nom": float(p_th_nom),
        "T_ref_celsius": float(T_ref),
        "cop": float(cop[0]),
        "T_init_celsius": float(T_init),
        "max_thermal_power_kw": float(max_thermal_power),
        "boiler_efficiency": float(efficiency),
        "gas_price_p_per_kwh": float(gas_price),
        "hp_max_power_kw": float(hp_max_power),
        "thermal_capacity_kwh_per_celsius": float(C),
        "thermal_conductance_kw_per_celsius": float(U),
    },
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
    "load": load.tolist(),
    "pv_supply": pv.p_el_supply.tolist(),
    "battery_net_charge": bat.power_el_schedule.tolist(),
    "heatpump_electrical": eh.p_el_schedule if isinstance(eh.p_el_schedule, list) else eh.p_el_schedule.tolist(),
    "grid_import": grid_import.tolist(),
    "peak_demand": {
        "peak_electricity_demand_kw": float(peak_electricity_demand),
        "time_of_peak_electricity_demand_hour": time_of_peak_electricity,
        "average_electricity_demand_kw": float(average_electricity_demand),
    },
    "costs": {
        "total_electricity_cost_gbp": float(np.sum(costs)),
        "total_gas_cost_gbp": float(np.sum(gas_costs)),
        "total_cost_gbp": float(np.sum(total_costs)),
    },
}

with open("Results/schedules/open_loop_single_building_schedules_and_costs.json", "w") as f:
    json.dump(single_building_data, f, indent=2)

print("\n" + "=" * 70)
print(f"Single building results saved to: Results/schedules/open_loop_single_building_schedules_and_costs.json")
print("=" * 70)

# ============================================================================
# SAVE TOTAL DEMAND VS TIME DATA
# ============================================================================

demand_vs_time = {
    "time_hours": list(range(time_horizon)),
    "building": building_col,
    "total_electricity_demand_kw": grid_import.tolist(),
    "total_thermal_demand_kw": total_heat_demand.tolist(),
    "load_kw": load.tolist(),
    "pv_supply_kw": pv.p_el_supply.tolist(),
    "battery_net_charge_kw": bat.power_el_schedule.tolist(),
    "heatpump_electrical_kw": eh.p_el_schedule if isinstance(eh.p_el_schedule, list) else eh.p_el_schedule.tolist(),
    "boiler_thermal_kw": p_th_heat_boiler.tolist(),
    "heatpump_thermal_kw": p_th_heat_hp.tolist(),
}

with open("Results/schedules/open_loop_demand_vs_time.json", "w") as f:
    json.dump(demand_vs_time, f, indent=2)

print(f"Demand vs time data saved to: Results/schedules/open_loop_demand_vs_time.json")

# ============================================================================
# MULTI-BUILDING PEAK DEMAND ANALYSIS
# ============================================================================
# Process all buildings and save their peak demand information

print("\n" + "=" * 70)
print("MULTI-BUILDING PEAK DEMAND ANALYSIS")
print("=" * 70)

all_buildings_peak_demand = {}

for building_idx, building_col in enumerate(cols):
    load_b = data_hr[building_col].values / 1000  # Convert to kW

    # Create components for this building
    bat_b = Battery(p_el_demand=p_el_charge, p_el_supply=p_el_discharge)
    pv_b = PVModule(
        time_horizon=time_horizon,
        start_point=0,
        radiation=radiation,
        area=25.0,
        beta=30.0,
        eta_noct=0.15,
    )
    eh_b = HeatPump(time_horizon=time_horizon)
    boiler_b = GasBoiler(time_horizon=time_horizon)

    # Update battery and PV schedules
    bat_b.energy_el_schedule = bat_b.battery_energy_schedule(time_horizon, delta_t)
    bat_b.power_el_schedule = p_el_charge - p_el_discharge
    pv_b.p_el_schedule = -1 * pv_b.p_el_supply

    # Generate heat schedules
    p_th_heat_hp_b = np.zeros(time_horizon)
    p_th_heat_boiler_b = np.zeros(time_horizon)

    for t in range(time_horizon):
        if heat_load[t] > 0:
            if heat_load[t] <= 5.0:
                p_th_heat_hp_b[t] = heat_load[t]
                p_th_heat_boiler_b[t] = 0
            else:
                p_th_heat_hp_b[t] = 5.0
                p_th_heat_boiler_b[t] = min(heat_load[t] - 5.0, 15.0)

    eh_b.p_th_heat = p_th_heat_hp_b
    eh_b.p_el_heat = [abs(p) / eh_b.cop[t] if eh_b.cop[t] > 0 else 0 for t, p in enumerate(p_th_heat_hp_b)]
    eh_b.p_el_schedule = eh_b.p_el_heat
    boiler_b.set_thermal_output(p_th_heat_boiler_b)

    # Calculate grid import and demands
    grid_import_b = load_b + bat_b.power_el_schedule - pv_b.p_el_supply + eh_b.p_el_schedule
    total_heat_demand_b = p_th_heat_hp_b + p_th_heat_boiler_b

    # Calculate peak metrics
    peak_electricity_demand = np.max(grid_import_b)
    time_of_peak_electricity = int(np.argmax(grid_import_b))
    average_electricity_demand = np.mean(grid_import_b)

    peak_thermal_demand = np.max(total_heat_demand_b)
    time_of_peak_thermal = int(np.argmax(total_heat_demand_b))
    average_thermal_demand = np.mean(total_heat_demand_b)

    peak_battery_charge = np.max(p_el_charge)
    peak_battery_discharge = np.max(p_el_discharge)
    peak_heat_pump_output = np.max(p_th_heat_hp_b)
    peak_boiler_output = np.max(p_th_heat_boiler_b)

    # Store building data
    all_buildings_peak_demand[building_col] = {
        "peak_electricity_demand_kw": float(peak_electricity_demand),
        "time_of_peak_electricity_demand_hour": time_of_peak_electricity,
        "average_electricity_demand_kw": float(average_electricity_demand),
        "peak_thermal_demand_kw": float(peak_thermal_demand),
        "time_of_peak_thermal_demand_hour": time_of_peak_thermal,
        "average_thermal_demand_kw": float(average_thermal_demand),
        "peak_battery_charge_kw": float(peak_battery_charge),
        "peak_battery_discharge_kw": float(peak_battery_discharge),
        "peak_heat_pump_output_kw": float(peak_heat_pump_output),
        "peak_boiler_output_kw": float(peak_boiler_output),
    }

    print(f"\nBuilding {building_idx}: {building_col}")
    print(f"  Peak Electricity: {peak_electricity_demand:.2f} kW (hour {time_of_peak_electricity})")
    print(f"  Avg Electricity: {average_electricity_demand:.2f} kW")
    print(f"  Peak Thermal: {peak_thermal_demand:.2f} kW (hour {time_of_peak_thermal})")
    print(f"  Avg Thermal: {average_thermal_demand:.2f} kW")

# Save all buildings peak demand data to JSON
with open("Results/schedules/open_loop_all_buildings_peak_demand.json", "w") as f:
    json.dump(all_buildings_peak_demand, f, indent=2)

print("\n" + "=" * 70)
print(f"All buildings peak demand data saved to: Results/schedules/all_buildings_peak_demand.json")
print("=" * 70)
