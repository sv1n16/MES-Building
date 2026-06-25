import numpy as np
import pandas as pd
import json

from src.Classes.building import Building
from src.Classes.boiler import GasBoiler
from src.Classes.battery import Battery
from src.Classes.photovoltaic import PVModule
from src.Classes.heatpump import HeatPump

# ============================================================================
# LOAD DATA
# ============================================================================

data = pd.read_csv("data\\processed_data_2018_02_21.csv")
data.columns = data.columns.str.strip().str.lower()
cols = data.columns[data.columns.str.contains("consumption", case=False)]

consumption_data = data[cols]
top3_consumers = consumption_data.sum().sort_values(ascending=False).head(4).index.tolist()
cols = top3_consumers

data["price"] = data["price (p/kwh)"] / 100.0
data["hour"] = data.index // 12

data_hr = pd.DataFrame()
data_hr["price"] = data.groupby("hour")["price"].mean()
data_hr["pv"] = data.groupby("hour")["pv"].mean()
data_hr["outdoor temperature"] = data.groupby("hour")["outdoor temperature"].mean()
data_hr["temperature setpoint"] = data.groupby("hour")["temperature setpoint"].mean()

for c in cols:
    data_hr[c] = data.groupby("hour")[c].max()

# Parameters (matching central_optimisation_all_buildings_peak_demand.json)
time_horizon = len(data_hr)
delta_t = 1.0
battery_capacity = 12.0
max_power = 4.6
eta_charge = 0.9
eta_discharge = 0.9
p_th_nom = 12.0
T_ref = 7.0
cop = np.ones(time_horizon) * 2.18
T_init = 20.0
max_thermal_power = 20.0
efficiency = 0.9
gas_price = 5.0
hp_max_power = 10.0

C = 10.0
U = 0.5
price = data_hr["price"].values
radiation = data_hr["pv"].values / 1000
outdoor_temperature = data_hr["outdoor temperature"].values
temperature_setpoint = data_hr["temperature setpoint"].values

p_el_charge = np.zeros(time_horizon)
p_el_discharge = np.zeros(time_horizon)
p_el_charge[(data_hr.index < 6) | (data_hr.index >= 20)] = max_power
p_el_discharge[(data_hr.index >= 16) & (data_hr.index < 20)] = max_power
p_el_charge = np.minimum(p_el_charge, max_power)
p_el_discharge = np.minimum(p_el_discharge, max_power)

heat_load = np.maximum(0, 2.0 * (temperature_setpoint - outdoor_temperature) / 10)

# ============================================================================
# LOAD INITIAL CONDITIONS FROM CENTRAL OPTIMISATION
# ============================================================================

# Load initial conditions from central optimisation JSON
with open("Results/schedules/central_optimisation_all_buildings_peak_demand.json", "r") as f:
    central_opt_data = json.load(f)

# Extract initial conditions for each building
initial_T_in = {}
initial_soc = {}
for b, building_col in enumerate(cols):
    building_data = central_opt_data["all_buildings_peak_demand"].get(building_col, {})
    initial_conditions = building_data.get("initial_conditions", {})
    initial_T_in[b] = float(initial_conditions.get("initial_indoor_temperature_celsius", 20.0))
    initial_soc[b] = float(initial_conditions.get("initial_battery_soc_kwh", 0.8 * battery_capacity))

print(f"\nInitial conditions loaded from central_optimisation_all_buildings_peak_demand.json:")
for b in range(len(cols)):
    print(f"  Building {b}: Temperature {initial_T_in[b]:.2f}°C, Battery SOC {initial_soc[b]:.2f} kWh")

# ==============================================================
# MULTI-BUILDING PEAK DEMAND ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("MULTI-BUILDING PEAK DEMAND ANALYSIS")
print("=" * 70)

all_buildings_peak_demand = {}

for b, building_col in enumerate(cols):
    load_b = data_hr[building_col].values / 1000

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
    bd = Building(building_components=[bat_b, pv_b, eh_b, boiler_b])
    bat_b.energy_el_schedule = bat_b.battery_energy_schedule(time_horizon, delta_t)
    bat_b.power_el_schedule = p_el_charge - p_el_discharge
    pv_b.p_el_schedule = -1 * pv_b.p_el_supply

    p_th_heat_hp_b = np.zeros(time_horizon)
    p_th_heat_boiler_b = np.zeros(time_horizon)

    indoor_temperature = np.zeros(time_horizon)
    indoor_temperature[0] = initial_T_in[b]
    required_heat = np.zeros(time_horizon)

    for t in range(time_horizon):
        T_prev = indoor_temperature[t]
        T_set = temperature_setpoint[t]
        T_out = outdoor_temperature[t]

        required_heat[t] = C * (T_set - T_prev) / delta_t + U * (T_set - T_out)
        required_heat[t] = max(required_heat[t], 0)

        if required_heat[t] <= hp_max_power:
            p_th_heat_hp_b[t] = required_heat[t]
            p_th_heat_boiler_b[t] = 0
        else:
            p_th_heat_hp_b[t] = hp_max_power
            p_th_heat_boiler_b[t] = min(required_heat[t] - hp_max_power, max_thermal_power)

        if t < time_horizon - 1:
            indoor_temperature[t + 1] = T_prev + delta_t / C * (
                p_th_heat_hp_b[t] + p_th_heat_boiler_b[t] - U * (T_prev - T_out)
            )

    eh_b.p_th_heat = p_th_heat_hp_b
    boiler_b.set_thermal_output(p_th_heat_boiler_b)

    eh_b.p_th_heat = p_th_heat_hp_b
    eh_b.p_el_heat = [abs(p) / eh_b.cop[t] if eh_b.cop[t] > 0 else 0 for t, p in enumerate(p_th_heat_hp_b)]
    eh_b.p_el_schedule = eh_b.p_el_heat
    boiler_b.set_thermal_output(p_th_heat_boiler_b)

    # Curtail surplus — no export to grid
    raw_grid_import_b = load_b + bat_b.power_el_schedule - pv_b.p_el_supply + eh_b.p_el_schedule
    grid_import_b = np.maximum(raw_grid_import_b, 0)

    total_heat_demand_b = p_th_heat_hp_b + p_th_heat_boiler_b

    # electricity_cost_b = np.sum(price * grid_import_b)
    # gas_consumption_b = np.sum(p_th_heat_boiler_b) * delta_t * gas_price / 100.0
    # gas_costs = np.array(boiler_b.gas_consumption_schedule) * boiler_b.gas_price
    # total_cost_b = electricity_cost_b + gas_consumption_b
    electricity_cost_b = np.sum(np.array([price[t] * grid_import_b[t] for t in range(time_horizon)]))
    gas_costs = np.sum(np.array(boiler_b.gas_consumption_schedule) * boiler_b.gas_price)
    total_costs = electricity_cost_b + gas_costs
    print("\nBuilding: {}".format(building_col))
    print(f"Electricity Costs: £{electricity_cost_b:.2f}")
    print(f"Gas Costs: £{gas_costs:.2f}")
    print(f"Total costs (electricity + gas): £{total_costs:.2f}")
    T_in_b = np.zeros(time_horizon)
    T_in_b[0] = initial_T_in[b]
    for t in range(time_horizon - 1):
        T_in_b[t + 1] = (
            T_in_b[t]
            + delta_t
            * (U * (outdoor_temperature[t] - T_in_b[t]) + p_th_heat_hp_b[t] + p_th_heat_boiler_b[t] - load_b[t])
            / C
        )

    temp_deviation = np.abs(T_in_b - temperature_setpoint)
    avg_temp_deviation = np.mean(temp_deviation)

    peak_electricity_demand = np.max(grid_import_b)
    time_of_peak_electricity = int(np.argmax(grid_import_b))
    average_electricity_demand = np.mean(grid_import_b)

    all_buildings_peak_demand[cols[b]] = {
        "initial_conditions": {
            "initial_indoor_temperature_celsius": float(initial_T_in[b]),
            "initial_battery_soc_kwh": float(initial_soc[b]),
        },
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
            "thermal_capacity_kwh_per_celsius": 10.0,
            "thermal_conductance_kw_per_celsius": 0.5,
            "temperature_deviation_penalty_alpha": 0.5,
        },
        "peak_loads": {
            "peak_electricity_demand_kw": float(peak_electricity_demand),
            "time_of_peak_electricity_demand_hour": time_of_peak_electricity,
            "average_electricity_demand_kw": float(average_electricity_demand),
            "total_cost_gbp": float(total_costs),
            "average_thermal_setpoint_deviation_celsius": float(avg_temp_deviation),
        },
        "schedules": {
            "grid_import_schedule": grid_import_b.tolist(),
            "load_schedule": load_b[b].tolist(),
            "pv_schedule": pv_b.p_el_schedule.tolist(),
            "battery_charge_schedule": bat_b.energy_el_schedule.tolist(),
            "battery_discharge_schedule": p_el_discharge.tolist(),
            "battery_soc_schedule": bat_b.power_el_schedule.tolist(),
            "heatpump_thermal_output": p_th_heat_hp_b.tolist(),
            "boiler_thermal_output": p_th_heat_boiler_b.tolist(),
            "indoor_temperature": T_in_b.tolist(),
            "temperature_setpoint": temperature_setpoint.tolist(),
            "electricity_costs": [float(price[t] * grid_import_b[t]) for t in range(time_horizon)],
        },
    }

consolidated_data = {"all_buildings_peak_demand": all_buildings_peak_demand}

with open("Results/schedules/open_loop_all_buildings_peak_demand.json", "w") as f:
    json.dump(consolidated_data, f, indent=2)

print("Saved to: Results/schedules/open_loop_all_buildings_peak_demand.json")
print("=" * 70)
