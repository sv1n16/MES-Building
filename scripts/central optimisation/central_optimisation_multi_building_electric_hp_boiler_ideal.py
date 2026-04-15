from gurobipy import Model, GRB
import pyomo.environ as pyo
import json
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from pyomo.util.infeasible import log_infeasible_constraints
import pyomo.environ as pyo

model = pyo.ConcreteModel()

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

# --- aggregate ---
data_hr = pd.DataFrame()
# price is already converted from p/kWh to £/kWh above; do NOT divide by 100 again
data_hr["price"] = data.groupby("hour")["price"].mean()
data_hr["pv"] = data.groupby("hour")["pv"].mean()
data_hr["outdoor temperature"] = data.groupby("hour")["outdoor temperature"].mean()
data_hr["temperature setpoint"] = data.groupby("hour")["temperature setpoint"].mean()

for c in cols:
    data_hr[c] = data.groupby("hour")[c].max()

# Parameters
time_horizon = len(data_hr)  # Number of time steps
delta_t = 1  # Time step in hours
battery_capacity = 12.0  # kWh
max_power = 4.6  # kW
initial_soc = 0.8 * battery_capacity  # kWh
eta_charge = 0.9  # Charging efficiency
eta_discharge = 0.9
n_buildings = len(cols)
p_th_nom = 12
T_ref = 7
cop = np.ones(time_horizon) * 2.18
T_init = 20.0
max_thermal_power = 20.0  # kW
efficiency = 0.9  # Boiler efficiency (fraction)
gas_price = 5  # Gas price (p/kWh)
hp_max_power = 10.0  # Maximum heat pump electrical power (kW)

# ---- Sets ----
model.buildings = pyo.RangeSet(0, n_buildings - 1)  # buildings
model.t = pyo.RangeSet(0, time_horizon - 1)
dt = 1.0
model.dt = pyo.Param(initialize=dt)
alpha = 0.5

# ---- Parameters ----

irradiance = data_hr["pv"].values.flatten() / 1000
pv_data = irradiance
# Example: PV supply for each building and time (assuming same PV for all buildings)
pv_supplies = {(b, t): pv_data[t] for b in range(n_buildings) for t in range(time_horizon)}
model.pv_supply = pyo.Param(model.buildings, model.t, initialize=pv_supplies)

# ---- Decision Variables ----

# Battery variables
model.charge = pyo.Var(model.buildings, model.t, bounds=(0, max_power), initialize=0)
model.discharge = pyo.Var(model.buildings, model.t, bounds=(0, max_power), initialize=0)

## Initial SOC: random values between 0 and 1 (fractions) for each building,
## converted to kWh by multiplying by battery_capacity
initial_soc_frac = np.random.rand(n_buildings)
initial_soc = [float(f * battery_capacity) for f in initial_soc_frac]

model.soc = pyo.Var(
    model.buildings, model.t, bounds=(0, battery_capacity), initialize={(b): initial_soc[b] for b in model.buildings}
)
model.charging_state = pyo.Var(model.buildings, model.t, domain=pyo.Binary)

# Electrical variables
model.p_el_vars = pyo.Var(model.buildings, model.t, bounds=(0, None))  # imported electricity from grid
model.p_hp = pyo.Var(model.buildings, model.t, bounds=(0, hp_max_power))  # heat pump electrical power

# Heat pump and boiler variables
model.q_heat_vars = pyo.Var(
    model.buildings,
    model.t,
    bounds=(0, p_th_nom),
    initialize=0,
)
model.cop = pyo.Var(model.buildings, model.t, bounds=(1, None), initialize=2.2)
model.f = pyo.Var(model.buildings, model.t, bounds=(0, 1), initialize=0.5)
model.q_heat = pyo.Var(model.buildings, model.t, bounds=(0, None), initialize=0)
model.gas_consumption = pyo.Var(model.buildings, model.t, domain=pyo.Reals, bounds=(0, None), initialize=0)
model.q_boiler_vars = pyo.Var(model.buildings, model.t, bounds=(0, max_thermal_power), initialize=0)

# Thermal variables
model.T_in = pyo.Var(model.buildings, model.t, bounds=(0, None), initialize=T_init)

# ---- Parameters (Data) ----

model.electric_load = pyo.Param(
    model.buildings,
    model.t,
    initialize={(b, t): float(data_hr[cols[b]].values[t]) / 1000 for b in model.buildings for t in model.t},
)

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

# ============================================================================
# CONSTRAINT DEFINITIONS
# ============================================================================

# ---- Battery Constraints ----


def charge_max_rule(model, b, t):
    """Maximum charge power limited by charging state"""
    return model.charge[b, t] <= model.charging_state[b, t] * max_power


model.charge_max = pyo.Constraint(model.buildings, model.t, rule=charge_max_rule)


def discharge_max_rule(model, b, t):
    """Maximum discharge power limited by charging state (cannot charge and discharge simultaneously)"""
    return model.discharge[b, t] <= (1 - model.charging_state[b, t]) * max_power


model.discharge_max = pyo.Constraint(model.buildings, model.t, rule=discharge_max_rule)


def soc_balance_rule(model, b, t):
    """State of charge dynamics"""
    if t == 0:
        return model.soc[b, t] == initial_soc[b]
    else:
        return (
            model.soc[b, t]
            == model.soc[b, t - 1]
            + (eta_charge * model.charge[b, t] - (1.0 / eta_discharge) * model.discharge[b, t]) * model.dt
        )


model.soc_balance = pyo.Constraint(model.buildings, model.t, rule=soc_balance_rule)


def no_charge_at_start_rule(model, b, t):
    """No charging in first time step"""
    if t == 0:
        return model.charge[b, t] == 0
    return pyo.Constraint.Skip


model.no_charge_at_start = pyo.Constraint(model.buildings, model.t, rule=no_charge_at_start_rule)


def no_discharge_at_start_rule(model, b, t):
    """No discharging in first time step"""
    if t == 0:
        return model.discharge[b, t] == 0
    return pyo.Constraint.Skip


model.no_discharge_at_start = pyo.Constraint(model.buildings, model.t, rule=no_discharge_at_start_rule)

# ---- Heat Pump & Boiler Constraints ----


def heat_pump_output_rule(model, b, t):
    """Heat pump output as fraction of nominal power"""
    return model.q_heat_vars[b, t] == model.cop[b, t] * model.p_hp[b, t]


model.heat_pump_output = pyo.Constraint(model.buildings, model.t, rule=heat_pump_output_rule)


def q_heat_rule(model, b, t):
    return model.q_heat_vars[b, t] == p_th_nom * model.f[b, t]


model.q_heat_constraint = pyo.Constraint(model.buildings, model.t, rule=q_heat_rule)


def cop_calculation_rule(model, b, t):
    """COP varies with outdoor temperature"""
    return model.cop[b, t] == cop[0] + 0.01 * (model.T_out[b, t] - T_ref)


model.cop_calculation = pyo.Constraint(model.buildings, model.t, rule=cop_calculation_rule)


def boiler_max_rule(model, b, t):
    """Maximum boiler output"""
    return model.q_boiler_vars[b, t] <= max_thermal_power


model.boiler_max = pyo.Constraint(model.buildings, model.t, rule=boiler_max_rule)


def gas_consumption_rule(model, b, t):
    """Gas consumption from boiler output"""
    return model.gas_consumption[b, t] == model.q_boiler_vars[b, t] / efficiency if efficiency > 0 else 0.0


model.gas_consumption_calc = pyo.Constraint(model.buildings, model.t, rule=gas_consumption_rule)

# ---- General System Constraints ----


def electricity_balance_rule(model, b, t):
    """Electricity balance: grid import = load + charging - PV - discharging + heat pump"""
    return model.p_el_vars[b, t] == (
        model.electric_load[b, t]
        + model.charge[b, t]
        - model.pv_supply[b, t]
        - model.discharge[b, t]
        + model.p_hp[b, t]
    )


model.electricity_balance = pyo.Constraint(model.buildings, model.t, rule=electricity_balance_rule)


def heat_balance_rule(model, b, t):
    """Heat balance: total heat = heat pump + boiler"""
    return model.q_heat_vars[b, t] + model.q_boiler_vars[b, t] == model.q_heat[b, t]


model.heat_balance = pyo.Constraint(model.buildings, model.t, rule=heat_balance_rule)

# ---- Thermal Dynamics Constraints ----


def thermal_dynamics_rule(model, b, t):
    """Building thermal dynamics with heat loss"""
    if t == 0:
        return model.T_in[b, t] == T_init
    else:
        return model.T_in[b, t] == model.T_in[b, t - 1] + model.dt / 10 * (
            model.q_heat[b, t] - 0.5 * (model.T_in[b, t - 1] - model.T_out[b, t])
        )


model.thermal_dynamics = pyo.Constraint(model.buildings, model.t, rule=thermal_dynamics_rule)


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================


def objective_rule(model):
    """Minimize total cost: electricity + gas + temperature deviation penalty"""
    return sum(
        data_hr["price"][t] * model.p_el_vars[b, t] * model.dt
        + gas_price / 100 * model.gas_consumption[b, t] * model.dt
        + alpha * (model.T_in[b, t] - model.T_set[b, t]) ** 2
        for b in model.buildings
        for t in model.t
    )


model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# ============================================================================
# SOLVE
# ============================================================================

solver = pyo.SolverFactory("gurobi_direct")
result = solver.solve(model, tee=True, logfile="Results/Temp_setpoint/solver_log.txt")

log_infeasible_constraints(model)

# ============================================================================
# EXTRACT RESULTS
# ============================================================================

charge_schedule = np.array([[pyo.value(model.charge[b, t]) for t in model.t] for b in model.buildings])
discharge_schedule = np.array([[pyo.value(model.discharge[b, t]) for t in model.t] for b in model.buildings])
soc_schedule = np.array([[pyo.value(model.soc[b, t]) for t in model.t] for b in model.buildings])
net_charge_schedule = charge_schedule - discharge_schedule
grid_import_schedule = np.array([[pyo.value(model.p_el_vars[b, t]) for t in model.t] for b in model.buildings])
heatpump_schedule = np.array([[pyo.value(model.p_hp[b, t]) for t in model.t] for b in model.buildings])
pv_schedule = np.array([[model.pv_supply[b, t] for t in model.t] for b in model.buildings])
load_schedule = np.array([[model.electric_load[b, t] for t in model.t] for b in model.buildings])
charging_state_schedule = np.array([[pyo.value(model.charging_state[b, t]) for t in model.t] for b in model.buildings])
T_in = [[pyo.value(model.T_in[b, t]) for t in model.t] for b in model.buildings]
T_out = [[pyo.value(model.T_out[b, t]) for t in model.t] for b in model.buildings]
T_set = [[model.T_set[b, t] for t in model.t] for b in model.buildings]
q_boiler_schedule = np.array([[pyo.value(model.q_boiler_vars[b, t]) for t in model.t] for b in model.buildings])
q_heatpump_schedule = np.array([[pyo.value(model.q_heat_vars[b, t]) for t in model.t] for b in model.buildings])
q_total_schedule = np.array([[pyo.value(model.q_heat[b, t]) for t in model.t] for b in model.buildings])


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

    # Electricity & Gas costs (only charge for positive imports, not exports)
    elec_cost_b_schedule = [
        data["price"].values[t]
        * max(
            0,
            pyo.value(model.p_hp[b, t])
            + pyo.value(model.charge[b, t])
            - pyo.value(model.discharge[b, t])
            - pyo.value(model.pv_supply[b, t])
            + model.electric_load[b, t],
        )
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
        go.Scatter(
            y=load_schedule[b], mode="lines", name=f"Load (kW) {b}", line=dict(color="black"), showlegend=(b == 0)
        ),
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
            y=data_hr["price"].values,
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

# ============================================================================
# SAVE RESULTS TO JSON FOR BUILDING 0
# ============================================================================

b = 0  # Save results for building 0

# Calculate peak demand metrics for building 0
grid_import_b = grid_import_schedule[b]
total_heat_demand_b = q_total_schedule[b]
charge_b = charge_schedule[b]
discharge_b = discharge_schedule[b]
heatpump_output_b = q_heatpump_schedule[b]
boiler_output_b = q_boiler_schedule[b]

peak_electricity_demand = np.max(grid_import_b)
time_of_peak_electricity_demand = int(np.argmax(grid_import_b))
average_electricity_demand = np.mean(grid_import_b)
peak_heat_pump_output = np.max(heatpump_output_b)


output_data = {
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
    "battery_charge_schedule": charge_schedule[b].tolist(),
    "battery_discharge_schedule": discharge_schedule[b].tolist(),
    "battery_soc_schedule": soc_schedule[b].tolist(),
    "heatpump_thermal_output": q_heatpump_schedule[b].tolist(),
    "boiler_thermal_output": q_boiler_schedule[b].tolist(),
    "indoor_temperature": T_in[b].tolist() if isinstance(T_in[b], np.ndarray) else T_in[b],
    "temperature_setpoint": T_set[b].tolist() if isinstance(T_set[b], np.ndarray) else T_set[b],
    "outdoor_temperature": T_out[b].tolist() if isinstance(T_out[b], np.ndarray) else T_out[b],
    "electricity_price": data_hr["price"].values.tolist(),
    "electricity_costs": [
        float(c)
        for c in [data_hr["price"].values[t] * max(0, grid_import_schedule[b][t]) for t in range(time_horizon)]
    ],
    "gas_costs": [gas_price / 100.0 * pyo.value(model.gas_consumption[b, t]) for t in model.t],
    "total_costs": [electricity_costs[b] / time_horizon, gas_costs[b] / time_horizon],  # Per time step
    "load": load_schedule[b].tolist(),
    "pv_supply": pv_schedule[b].tolist(),
    "battery_net_charge": net_charge_schedule[b].tolist(),
    "heatpump_electrical": heatpump_schedule[b].tolist(),
    "grid_import": grid_import_schedule[b].tolist(),
    "total_heat_demand": q_total_schedule[b].tolist(),
    "peak_demand": {
        "peak_electricity_demand_kw": float(peak_electricity_demand),
        "time_of_peak_electricity_demand_hour": time_of_peak_electricity_demand,
        "average_electricity_demand_kw": float(average_electricity_demand),
        "peak_heat_pump_output_kw": float(peak_heat_pump_output),
    },
    "costs": {
        "total_electricity_cost_gbp": float(electricity_costs[b]),
        "total_gas_cost_gbp": float(gas_costs[b]),
        "total_cost_gbp": float(total_costs[b]),
    },
}

with open("Results/schedules/central_optimisation_schedules_and_costs.json", "w") as f:
    json.dump(output_data, f, indent=2)

print("\nCentral optimization results saved to: Results/schedules/central_optimisation_schedules_and_costs.json")
print(f"Building 0 Summary:")
print(f"Total Electricity Cost: £{electricity_costs[b]:.2f}")
print(f"Total Gas Cost: £{gas_costs[b]:.2f}")
print(f"Total Cost: £{total_costs[b]:.2f}")

# ============================================================================
# SAVE TOTAL DEMAND VS TIME DATA
# ============================================================================

demand_vs_time = {
    "time_hours": list(range(time_horizon)),
    "building": cols[0],
    "total_electricity_demand_kw": grid_import_schedule[0].tolist(),
    "total_thermal_demand_kw": q_total_schedule[0].tolist(),
    "load_kw": load_schedule[0].tolist(),
    "pv_supply_kw": pv_schedule[0].tolist(),
    "battery_net_charge_kw": net_charge_schedule[0].tolist(),
    "heatpump_electrical_kw": heatpump_schedule[0].tolist(),
    "boiler_thermal_kw": q_boiler_schedule[0].tolist(),
    "heatpump_thermal_kw": q_heatpump_schedule[0].tolist(),
}

with open("Results/schedules/central_optimisation_demand_vs_time.json", "w") as f:
    json.dump(demand_vs_time, f, indent=2)

print("Demand vs time data saved to: Results/schedules/central_optimisation_demand_vs_time.json")

# ============================================================================
# MULTI-BUILDING PEAK DEMAND ANALYSIS
# ============================================================================
# Process all buildings from optimization results and save their peak demand information

print("\n" + "=" * 70)
print("MULTI-BUILDING PEAK DEMAND ANALYSIS (OPTIMIZED)")
print("=" * 70)

all_buildings_peak_demand_opt = {}

for b in model.buildings:
    grid_import_b = grid_import_schedule[b]
    total_heat_demand_b = q_total_schedule[b]
    charge_b = charge_schedule[b]
    discharge_b = discharge_schedule[b]
    heatpump_output_b = q_heatpump_schedule[b]
    boiler_output_b = q_boiler_schedule[b]

    # Calculate peak metrics
    peak_electricity_demand = np.max(grid_import_b)
    time_of_peak_electricity = int(np.argmax(grid_import_b))
    peak_heat_pump_output = np.max(heatpump_output_b)

    # Store building data
    all_buildings_peak_demand_opt[cols[b]] = {
        "peak_electricity_demand_kw": float(peak_electricity_demand),
        "time_of_peak_electricity_demand_hour": time_of_peak_electricity,
        "average_electricity_demand_kw": float(average_electricity_demand),
        "peak_heat_pump_output_kw": float(peak_heat_pump_output),
    }

    print(f"\nBuilding {b}: {cols[b]}")
    print(f"  Peak Electricity: {peak_electricity_demand:.2f} kW (hour {time_of_peak_electricity})")
    print(f"  Avg Electricity: {average_electricity_demand:.2f} kW")


# Save all buildings peak demand data to JSON
with open("Results/schedules/central_optimisation_all_buildings_peak_demand.json", "w") as f:
    json.dump(all_buildings_peak_demand_opt, f, indent=2)

print("\n" + "=" * 70)
print(
    f"All buildings peak demand data saved to: Results/schedules/central_optimisation_all_buildings_peak_demand.json"
)
print("=" * 70)
