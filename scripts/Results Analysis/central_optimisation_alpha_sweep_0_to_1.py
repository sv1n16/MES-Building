"""
Alpha sweep optimization: 15 values from 0 to 1
Saves building 0 results as JSON for each alpha value
"""

import pyomo.environ as pyo
import json
import numpy as np
import pandas as pd
import os
from pathlib import Path

# Create output directory
output_dir = Path("Results/schedules/temp_analysis/temp_analysis_hp_boiler_27_02")
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
data = pd.read_csv("data\\processed_data_2018_02_21.csv")
data.columns = data.columns.str.strip().str.lower()
cols = data.columns[data.columns.str.contains("consumption", case=False)]
consumption_data = data[cols]
top3_consumers = consumption_data.sum().sort_values(ascending=False).head(2).index.tolist()
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

# Parameters
time_horizon = len(data_hr)
delta_t = 1
battery_capacity = 12.0
max_power = 4.6
initial_soc = 0.8 * battery_capacity
eta_charge = 0.9
eta_discharge = 0.9
n_buildings = len(cols)
p_th_nom = 12
T_ref = 7
cop_base = np.ones(time_horizon) * 2.18
T_init = 20.0
max_thermal_power = 50.0
efficiency = 0.9
gas_price = 5
hp_max_power = 10.0

# Generate 15 alpha values from 0 to 1
alpha_values = np.logspace(-2, 1, 20)

print(f"\nRunning optimization for {len(alpha_values)} alpha values: {alpha_values}")
print("=" * 70)

for alpha_idx, alpha in enumerate(alpha_values):
    print(f"\n[{alpha_idx + 1}/{len(alpha_values)}] Running optimization with alpha = {alpha:.4f}")

    # ========================================================================
    # BUILD PYOMO MODEL
    # ========================================================================
    model = pyo.ConcreteModel()

    # Sets
    model.buildings = pyo.RangeSet(0, n_buildings - 1)
    model.t = pyo.RangeSet(0, time_horizon - 1)
    dt = 1.0
    model.dt = pyo.Param(initialize=dt)

    # Variables
    model.charge = pyo.Var(model.buildings, model.t, bounds=(0, max_power), initialize=0)
    model.discharge = pyo.Var(model.buildings, model.t, bounds=(0, max_power), initialize=0)

    initial_soc_frac = np.random.rand(n_buildings)
    initial_soc_vals = [float(f * battery_capacity) for f in initial_soc_frac]

    model.soc = pyo.Var(
        model.buildings,
        model.t,
        bounds=(0, battery_capacity),
        initialize={(b): initial_soc_vals[b] for b in model.buildings},
    )
    model.charging_state = pyo.Var(model.buildings, model.t, domain=pyo.Binary)

    model.p_el_vars = pyo.Var(model.buildings, model.t, bounds=(0, None))
    model.p_hp = pyo.Var(model.buildings, model.t, bounds=(0, hp_max_power))
    model.P_peak = pyo.Var(bounds=(0, None))
    model.P_gas_peak = pyo.Var(bounds=(0, None))

    model.q_heat_vars = pyo.Var(model.buildings, model.t, bounds=(0, p_th_nom), initialize=0)
    model.cop = pyo.Var(model.buildings, model.t, bounds=(1, None), initialize=2.2)
    model.f = pyo.Var(model.buildings, model.t, bounds=(0, 1), initialize=0.5)
    model.q_heat = pyo.Var(model.buildings, model.t, bounds=(0, None), initialize=0)
    model.gas_consumption = pyo.Var(model.buildings, model.t, domain=pyo.Reals, bounds=(0, None), initialize=0)
    model.q_boiler_vars = pyo.Var(model.buildings, model.t, bounds=(0, max_thermal_power), initialize=0)
    model.T_in = pyo.Var(model.buildings, model.t, bounds=(0, None), initialize=T_init)

    # Parameters
    model.electric_load = pyo.Param(
        model.buildings,
        model.t,
        initialize={(b, t): float(data_hr[cols[b]].values[t]) / 1000 for b in model.buildings for t in model.t},
    )

    irradiance = data_hr["pv"].values.flatten() / 1000
    pv_supplies = {(b, t): irradiance[t] for b in range(n_buildings) for t in range(time_horizon)}
    model.pv_supply = pyo.Param(model.buildings, model.t, initialize=pv_supplies)

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

    # Constraints
    def charge_max_rule(model, b, t):
        return model.charge[b, t] <= model.charging_state[b, t] * max_power

    model.charge_max = pyo.Constraint(model.buildings, model.t, rule=charge_max_rule)

    def discharge_max_rule(model, b, t):
        return model.discharge[b, t] <= (1 - model.charging_state[b, t]) * max_power

    model.discharge_max = pyo.Constraint(model.buildings, model.t, rule=discharge_max_rule)

    def soc_balance_rule(model, b, t):
        if t == 0:
            return model.soc[b, t] == initial_soc_vals[b]
        else:
            return (
                model.soc[b, t]
                == model.soc[b, t - 1]
                + (eta_charge * model.charge[b, t] - (1.0 / eta_discharge) * model.discharge[b, t]) * model.dt
            )

    model.soc_balance = pyo.Constraint(model.buildings, model.t, rule=soc_balance_rule)

    def no_charge_at_start_rule(model, b, t):
        if t == 0:
            return model.charge[b, t] == 0
        return pyo.Constraint.Skip

    model.no_charge_at_start = pyo.Constraint(model.buildings, model.t, rule=no_charge_at_start_rule)

    def no_discharge_at_start_rule(model, b, t):
        if t == 0:
            return model.discharge[b, t] == 0
        return pyo.Constraint.Skip

    model.no_discharge_at_start = pyo.Constraint(model.buildings, model.t, rule=no_discharge_at_start_rule)

    def heat_pump_output_rule(model, b, t):
        return model.q_heat_vars[b, t] == model.cop[b, t] * model.p_hp[b, t]

    model.heat_pump_output = pyo.Constraint(model.buildings, model.t, rule=heat_pump_output_rule)

    def q_heat_rule(model, b, t):
        return model.q_heat_vars[b, t] == p_th_nom * model.f[b, t]

    model.q_heat_constraint = pyo.Constraint(model.buildings, model.t, rule=q_heat_rule)

    def cop_calculation_rule(model, b, t):
        return model.cop[b, t] == cop_base[0] + 0.01 * (model.T_out[b, t] - T_ref)

    model.cop_calculation = pyo.Constraint(model.buildings, model.t, rule=cop_calculation_rule)

    def boiler_max_rule(model, b, t):
        return model.q_boiler_vars[b, t] <= max_thermal_power

    model.boiler_max = pyo.Constraint(model.buildings, model.t, rule=boiler_max_rule)

    def gas_consumption_rule(model, b, t):
        return model.gas_consumption[b, t] == model.q_boiler_vars[b, t] / efficiency if efficiency > 0 else 0.0

    model.gas_consumption_calc = pyo.Constraint(model.buildings, model.t, rule=gas_consumption_rule)

    def electricity_balance_rule(model, b, t):
        return model.p_el_vars[b, t] == (
            model.electric_load[b, t]
            + model.charge[b, t]
            - model.pv_supply[b, t]
            - model.discharge[b, t]
            + model.p_hp[b, t]
        )

    model.electricity_balance = pyo.Constraint(model.buildings, model.t, rule=electricity_balance_rule)

    def heat_balance_rule(model, b, t):
        return model.q_heat_vars[b, t] + model.q_boiler_vars[b, t] == model.q_heat[b, t]

    model.heat_balance = pyo.Constraint(model.buildings, model.t, rule=heat_balance_rule)

    def thermal_dynamics_rule(model, b, t):
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

    def peak_gas_rule(model, t):
        return sum(model.gas_consumption[b, t] for b in model.buildings) <= model.P_gas_peak

    model.peak_gas_constraint = pyo.Constraint(model.t, rule=peak_gas_rule)

    # Objective
    def objective_rule(model):
        total_cost = 0
        for t in model.t:
            total_cost += model.P_peak
            total_cost += model.P_gas_peak
            for b in model.buildings:
                total_cost += data_hr["price"][t] * model.p_el_vars[b, t] * model.dt
                total_cost += gas_price / 100 * model.gas_consumption[b, t] * model.dt
                total_cost += alpha * (model.T_in[b, t] - model.T_set[b, t]) ** 2
        return total_cost

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Solve
    solver = pyo.SolverFactory("gurobi_direct")
    solver.options["LogFile"] = ""
    solver.solve(model, tee=False)

    # ========================================================================
    # EXTRACT BUILDING 0 RESULTS
    # ========================================================================
    b = 0
    charge_schedule = [pyo.value(model.charge[b, t]) for t in model.t]
    discharge_schedule = [pyo.value(model.discharge[b, t]) for t in model.t]
    soc_schedule = [pyo.value(model.soc[b, t]) for t in model.t]
    grid_import = [pyo.value(model.p_el_vars[b, t]) for t in model.t]
    heatpump_power = [pyo.value(model.p_hp[b, t]) for t in model.t]
    pv = [pyo.value(model.pv_supply[b, t]) for t in model.t]
    electric_load = [model.electric_load[b, t] for t in model.t]
    charging_state = [pyo.value(model.charging_state[b, t]) for t in model.t]
    q_boiler = [pyo.value(model.q_boiler_vars[b, t]) for t in model.t]
    q_heatpump = [pyo.value(model.q_heat_vars[b, t]) for t in model.t]
    q_total = [pyo.value(model.q_heat[b, t]) for t in model.t]
    T_in = [pyo.value(model.T_in[b, t]) for t in model.t]
    T_out = [pyo.value(model.T_out[b, t]) for t in model.t]
    T_set = [model.T_set[b, t] for t in model.t]

    # Create JSON output
    results = {
        "alpha": float(alpha),
        "charge": charge_schedule,
        "discharge": discharge_schedule,
        "soc": soc_schedule,
        "grid_import": grid_import,
        "heatpump_power": heatpump_power,
        "pv": pv,
        "electric_load": electric_load,
        "charging_state": charging_state,
        "q_boiler": q_boiler,
        "q_heatpump": q_heatpump,
        "q_total": q_total,
        "T_in": T_in,
        "T_out": T_out,
        "T_set": T_set,
    }

    # Save JSON
    json_filename = output_dir / f"building_0_alpha_{alpha:.4f}.json"
    with open(json_filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"   ✓ Saved: {json_filename}")

print("\n" + "=" * 70)
print(f"✓ Completed! Generated {len(alpha_values)} JSON files")
print(f"   Location: {output_dir}")
