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
print(data["price"].mean())
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
alpha = 0.03
beta = 300.0  # Deadband penalty coefficient for lower bound violation (higher penalty)
theta = 10.0  # Upper deadband penalty coefficient for upper bound violation

# ---- Occupancy Model Parameters ----
occupancy_profile = {
    0: 1,
    1: 1,
    2: 1,  # 02:00-03:00 (night - occupied)
    3: 1,  # 03:00-04:00 (night - occupied)F
    4: 1,  # 04:00-05:00 (night - occupied)
    5: 1,  # 05:00-06:00 (night - occupied)
    6: 1,  # 06:00-07:00 (morning - occupied)
    7: 1,  # 07:00-08:00 (morning - occupied)
    8: 1,  # 08:00-09:00 (morning - occupied)
    9: 0,  # 09:00-10:00 (unoccupied)
    10: 0,  # 10:00-11:00 (unoccupied)
    11: 0,  # 11:00-12:00 (unoccupied)
    12: 0,  # 12:00-13:00 (unoccupied)
    13: 0,  # 13:00-14:00 (unoccupied)
    14: 0,  # 14:00-15:00 (unoccupied)
    15: 0,  # 15:00-16:00 (unoccupied)
    16: 1,  # 16:00-17:00 (evening - occupied)
    17: 1,  # 17:00-18:00 (evening - occupied)
    18: 1,  # 18:00-19:00 (evening - occupied)
    19: 1,  # 19:00-20:00 (evening - occupied)
    20: 1,  # 20:00-21:00 (evening - occupied)
    21: 1,  # 21:00-22:00 (evening - occupied)
    22: 1,  # 22:00-23:00 (night - occupied)
    23: 1,  # 23:00-00:00 (night - occupied)
}


T_lower_nominal = 18.0
T_upper_nominal = 24.0
T_lower_unoccupied = 10.0
T_upper_unoccupied = 20.0

# Night hours deadband (larger deadband during night: 15-20)
night_hours = list(range(0, 6)) + list(range(22, 24))  # 0-5 and 22-23
T_lower_night = 15.0
T_upper_night = 20.0

# ---- Randomize Initial Building Temperatures ----
np.random.seed(42)
T_init_lower = 18.0  # Lower bound for random initial temperature
T_init_upper = 22.0  # Upper bound for random initial temperature
initial_T_frac = np.random.rand(n_buildings)
initial_T_in = [float(T_init_lower + f * (T_init_upper - T_init_lower)) for f in initial_T_frac]
print(f"Initial building temperatures (randomized within {T_init_lower}-{T_init_upper}C range):")
for b in range(n_buildings):
    print(f"  Building {b}: {initial_T_in[b]:.2f}C")

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
model.T_in = pyo.Var(
    model.buildings,
    model.t,
    bounds=(0, None),
    initialize=lambda m, b, t: initial_T_in[b] if t == 0 else T_init,
)

# Deadband slack variables for penalty calculation
model.T_below_lower = pyo.Var(
    model.buildings, model.t, bounds=(0, None), initialize=0
)  # Temperature below lower bound
model.T_above_upper = pyo.Var(
    model.buildings, model.t, bounds=(0, None), initialize=0
)  # Temperature above upper bound

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

# Occupancy parameter
model.occupancy_profile = pyo.Param(
    model.t, initialize={t: occupancy_profile.get(t % 24, 0.5) for t in range(time_horizon)}
)

model.T_lower_bound = pyo.Param(
    model.buildings,
    model.t,
    initialize={
        (b, t): (
            T_lower_night
            if (t % 24) in night_hours
            else (model.T_set[b, t] - 1.0) * model.occupancy_profile[t]
            + T_lower_unoccupied * (1 - model.occupancy_profile[t])
        )
        for b in model.buildings
        for t in model.t
    },
)
model.T_upper_bound = pyo.Param(
    model.buildings,
    model.t,
    initialize={
        (b, t): (
            T_upper_night
            if (t % 24) in night_hours
            else (model.T_set[b, t] + 1.0) * model.occupancy_profile[t]
            + T_upper_unoccupied * (1 - model.occupancy_profile[t])
        )
        for b in model.buildings
        for t in model.t
    },
)

# ============================================================================
# CONSTRAINT DEFINITIONS
# ============================================================================


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
        return model.T_in[b, t] == initial_T_in[b]
    else:
        return model.T_in[b, t] == model.T_in[b, t - 1] + model.dt / 10 * (
            model.q_heat[b, t] - 0.5 * (model.T_in[b, t - 1] - model.T_out[b, t])
        )


model.thermal_dynamics = pyo.Constraint(model.buildings, model.t, rule=thermal_dynamics_rule)


# ---- Deadband Slack Constraints ----
def temperature_below_lower_rule_1(model, b, t):
    """Slack variable for temperature below lower deadband - lower bound"""
    return model.T_below_lower[b, t] >= 0


def temperature_below_lower_rule_2(model, b, t):
    """Slack variable for temperature below lower deadband - upper bound"""
    return model.T_below_lower[b, t] >= model.T_lower_bound[b, t] - model.T_in[b, t]


model.temperature_below_lower_1 = pyo.Constraint(model.buildings, model.t, rule=temperature_below_lower_rule_1)
model.temperature_below_lower_2 = pyo.Constraint(model.buildings, model.t, rule=temperature_below_lower_rule_2)


def temperature_above_upper_rule_1(model, b, t):
    """Slack variable for temperature above upper deadband - lower bound"""
    return model.T_above_upper[b, t] >= 0


def temperature_above_upper_rule_2(model, b, t):
    """Slack variable for temperature above upper deadband - upper bound"""
    return model.T_above_upper[b, t] >= model.T_in[b, t] - model.T_upper_bound[b, t]


model.temperature_above_upper_1 = pyo.Constraint(model.buildings, model.t, rule=temperature_above_upper_rule_1)
model.temperature_above_upper_2 = pyo.Constraint(model.buildings, model.t, rule=temperature_above_upper_rule_2)


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================


def objective_rule(model):

    return sum(
        # Energy costs
        data_hr["price"][t] * model.p_el_vars[b, t] * model.dt
        + gas_price / 100 * model.gas_consumption[b, t] * model.dt
        # Deadband penalty (weighted by occupancy): occupancy⋅[β⋅max(0, Tlow−T)² + θ⋅max(0, T−Thigh)²]
        + model.occupancy_profile[t] * (beta * model.T_below_lower[b, t] ** 2 + theta * model.T_above_upper[b, t] ** 2)
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
    T_lower = [pyo.value(model.T_lower_bound[b, t]) for t in model.t]
    T_upper = [pyo.value(model.T_upper_bound[b, t]) for t in model.t]
    T_below = [pyo.value(model.T_below_lower[b, t]) for t in model.t]
    T_above = [pyo.value(model.T_above_upper[b, t]) for t in model.t]
    occupancy = [pyo.value(model.occupancy_profile[t]) for t in model.t]

    # Add deadband as filled area (upper bound)
    fig.add_trace(
        go.Scatter(
            y=T_upper,
            mode="lines",
            name="Temp Upper Deadband (°C)",
            line=dict(color="lightgray", width=1),
            showlegend=(b == 0),
        ),
        row=row,
        col=2,
    )

    # Add deadband as filled area (lower bound)
    fig.add_trace(
        go.Scatter(
            y=T_lower,
            mode="lines",
            name="Temp Lower Deadband (°C)",
            line=dict(color="lightgray", width=1),
            fill="tonexty",
            fillcolor="rgba(211, 211, 211, 0.3)",
            showlegend=(b == 0),
        ),
        row=row,
        col=2,
    )

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

# Add occupancy shading for unoccupied hours (only once for column 2, applied to all rows)
for t in range(time_horizon):
    occupancy_val = pyo.value(model.occupancy_profile[t])
    if occupancy_val < 0.5:  # Unoccupied (0)
        fig.add_vrect(x0=t - 0.5, x1=t + 0.5, fillcolor="gray", opacity=0.1, layer="below", line_width=0, col=2)

fig.show()

# ============================================================================
# SAVE TOTAL DEMAND VS TIME DATA
# ============================================================================
# ===================================================================
# MULTI-BUILDING ANALYSIS
# ============================================================================
# Process all buildings from optimization results and save their peak demand information


print("\n" + "=" * 70)
print("MULTI-BUILDING ANALYSIS (OPTIMIZED)")
print("=" * 70)

all_buildings_peak_demand_opt = {}

for b in model.buildings:
    T_in = [pyo.value(model.T_in[b, t]) for t in model.t]
    T_out = [pyo.value(model.T_out[b, t]) for t in model.t]
    T_set = [model.T_set[b, t] for t in model.t]
    grid_import_b = grid_import_schedule[b]
    charge_b = charge_schedule[b]
    discharge_b = discharge_schedule[b]
    heatpump_output_b = q_heatpump_schedule[b]
    boiler_output_b = q_boiler_schedule[b]

    # Calculate peak metrics
    peak_electricity_demand = np.max(grid_import_b)
    time_of_peak_electricity = int(np.argmax(grid_import_b))
    average_electricity_demand_b = np.mean(grid_import_b)

    # Extract temperature data for this building

    # Calculate thermal setpoint deviation
    temp_deviation = np.abs(np.array(T_in) - np.array(T_set))
    avg_temp_deviation = np.mean(temp_deviation)

    # Calculate total cost for this building
    electricity_cost_b = float(
        np.sum([data_hr["price"].values[t] * max(0, grid_import_b[t]) for t in range(time_horizon)])
    )
    gas_cost_b = float(gas_price / 100.0 * np.sum([pyo.value(model.gas_consumption[b, t]) for t in model.t]))
    total_cost_b = electricity_cost_b + gas_cost_b

    # Store building data
    all_buildings_peak_demand_opt[cols[b]] = {
        "initial_conditions": {
            "initial_indoor_temperature_celsius": float(initial_T_in[b]),
            "initial_battery_soc_kwh": float(initial_soc[b]),
        },
        "peak_loads": {
            "peak_electricity_demand_kw": float(peak_electricity_demand),
            "time_of_peak_electricity_demand_hour": time_of_peak_electricity,
            "average_electricity_demand_kw": float(average_electricity_demand_b),
            "total_cost_gbp": float(total_cost_b),
            "average_thermal_setpoint_deviation_celsius": float(avg_temp_deviation),
        },
        "schedules": {
            "grid_import_schedule": grid_import_schedule[b].tolist(),
            "load_schedule": load_schedule[b].tolist(),
            "pv_schedule": pv_schedule[b].tolist(),
            "battery_charge_schedule": charge_schedule[b].tolist(),
            "battery_discharge_schedule": discharge_schedule[b].tolist(),
            "battery_soc_schedule": soc_schedule[b].tolist(),
            "heatpump_thermal_output": q_heatpump_schedule[b].tolist(),
            "boiler_thermal_output": q_boiler_schedule[b].tolist(),
            "indoor_temperature": T_in,
            "temperature_setpoint": T_set,
            "outdoor_temperature": T_out,
            "electricity_price": data_hr["price"].values.tolist(),
            "electricity_costs": [
                float(c)
                for c in [data_hr["price"].values[t] * max(0, grid_import_schedule[b][t]) for t in range(time_horizon)]
            ],
            "occupancy_profile": [float(pyo.value(model.occupancy_profile[t])) for t in model.t],
            "temperature_lower_deadband_celsius": [float(pyo.value(model.T_lower_bound[b, t])) for t in model.t],
            "temperature_upper_deadband_celsius": [float(pyo.value(model.T_upper_bound[b, t])) for t in model.t],
        },
    }

print(all_buildings_peak_demand_opt[cols[b]]["schedules"])
# Create consolidated JSON with single building info on top and all buildings peak demand
consolidated_data_opt = {
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
    "all_buildings_peak_demand": all_buildings_peak_demand_opt,
}

# Save consolidated data to JSON
with open("Results/schedules/central_optimisation_all_buildings_peak_demand.json", "w") as f:
    json.dump(consolidated_data_opt, f, indent=2)

print("\n" + "=" * 70)
print(
    f"Consolidated data (single building + all buildings peak demand) saved to: Results/schedules/central_optimisation_all_buildings_peak_demand.json"
)
print("=" * 70)
