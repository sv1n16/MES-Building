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
theta = 50.0  # Upper deadband penalty coefficient for upper bound violation

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
T_lower = [[model.T_lower_bound[b, t] for t in model.t] for b in model.buildings]
T_upper = [[model.T_upper_bound[b, t] for t in model.t] for b in model.buildings]
q_boiler_schedule = np.array([[pyo.value(model.q_boiler_vars[b, t]) for t in model.t] for b in model.buildings])
q_heatpump_schedule = np.array([[pyo.value(model.q_heat_vars[b, t]) for t in model.t] for b in model.buildings])
q_total_schedule = np.array([[pyo.value(model.q_heat[b, t]) for t in model.t] for b in model.buildings])
occupancy_schedule = np.array([[model.occupancy_profile[t] for t in model.t] for b in model.buildings])
T_below_schedule = np.array([[pyo.value(model.T_below_lower[b, t]) for t in model.t] for b in model.buildings])
T_above_schedule = np.array([[pyo.value(model.T_above_upper[b, t]) for t in model.t] for b in model.buildings])


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

    # Individual cost breakdowns
    load_cost_schedule = [data["price"].values[t] * max(0, model.electric_load[b, t]) for t in model.t]
    charge_cost_schedule = [data["price"].values[t] * max(0, pyo.value(model.charge[b, t])) for t in model.t]
    hp_cost_schedule = [data["price"].values[t] * max(0, pyo.value(model.p_hp[b, t])) for t in model.t]

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
    T_in_b = [pyo.value(model.T_in[b, t]) for t in model.t]
    T_out_b = [pyo.value(model.T_out[b, t]) for t in model.t]
    T_set_b = [model.T_set[b, t] for t in model.t]
    T_lower_b = [model.T_lower_bound[b, t] for t in model.t]
    T_upper_b = [model.T_upper_bound[b, t] for t in model.t]

    fig.add_trace(
        go.Scatter(y=T_in_b, mode="lines", name="Indoor Temp (°C)", line=dict(color="firebrick"), showlegend=(b == 0)),
        row=row,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            y=T_out_b, mode="lines", name="Outdoor Temp (°C)", line=dict(color="deepskyblue"), showlegend=(b == 0)
        ),
        row=row,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            y=T_set_b,
            mode="lines",
            name="Setpoint (°C)",
            line=dict(color="darkgreen", dash="dash"),
            showlegend=(b == 0),
        ),
        row=row,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            y=T_lower_b,
            mode="lines",
            name="Lower Deadband (°C)",
            line=dict(color="orange", dash="dot"),
            showlegend=(b == 0),
        ),
        row=row,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            y=T_upper_b,
            mode="lines",
            name="Upper Deadband (°C)",
            line=dict(color="purple", dash="dot"),
            showlegend=(b == 0),
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
            y=load_cost_schedule,
            mode="lines",
            name="Load Cost",
            line=dict(color="black"),
            showlegend=(b == 0),
        ),
        row=row,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            y=charge_cost_schedule,
            mode="lines",
            name="Battery Charge Cost",
            line=dict(color="blue"),
            showlegend=(b == 0),
        ),
        row=row,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            y=hp_cost_schedule,
            mode="lines",
            name="Heat Pump Cost",
            line=dict(color="green"),
            showlegend=(b == 0),
        ),
        row=row,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            y=elec_cost_b_schedule,
            mode="lines",
            name="Total Electricity Cost",
            line=dict(color="gold", width=2),
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
# PLOT PENALTY VALUES AND OCCUPANCY (Building 0)
# ============================================================================

b = 0  # Building 0

# Calculate occupancy and penalty values
occupancy_values = [model.occupancy_profile[t] for t in model.t]
penalty_values = [
    model.occupancy_profile[t]
    * (beta * pyo.value(model.T_below_lower[b, t]) ** 2 + theta * pyo.value(model.T_above_upper[b, t]) ** 2)
    for t in model.t
]

# Create figure with secondary y-axis
fig_penalty = go.Figure()

# Add occupancy trace (left y-axis)
fig_penalty.add_trace(
    go.Scatter(
        x=list(range(time_horizon)),
        y=occupancy_values,
        mode="lines+markers",
        name="Occupancy",
        line=dict(color="blue", width=2),
        yaxis="y1",
    )
)

# Add penalty trace (right y-axis)
fig_penalty.add_trace(
    go.Scatter(
        x=list(range(time_horizon)),
        y=penalty_values,
        mode="lines",
        name="Thermal Comfort Penalty",
        line=dict(color="red", width=3),
        yaxis="y2",
    )
)

fig_penalty.update_layout(
    title="Building 0: Occupancy Profile and Thermal Comfort Penalty",
    xaxis=dict(title="Time (hours)"),
    yaxis=dict(title="Occupancy", side="left", tickfont=dict(color="blue"), range=[0, 1.1]),
    yaxis2=dict(title="Penalty Value", side="right", overlaying="y", tickfont=dict(color="red")),
    hovermode="x unified",
    width=1000,
    height=600,
    legend=dict(x=0.02, y=0.98),
)

fig_penalty.show()

# ============================================================================
# SAVE RESULTS TO JSON FOR BUILDING 0
# ============================================================================

b = 0  # Save results for building 0

output_data = {
    "battery_charge_schedule": charge_schedule[b].tolist(),
    "battery_discharge_schedule": discharge_schedule[b].tolist(),
    "battery_soc_schedule": soc_schedule[b].tolist(),
    "heatpump_thermal_output": q_heatpump_schedule[b].tolist(),
    "boiler_thermal_output": q_boiler_schedule[b].tolist(),
    "indoor_temperature": T_in[b].tolist() if isinstance(T_in[b], np.ndarray) else T_in[b],
    "temperature_setpoint": T_set[b].tolist() if isinstance(T_set[b], np.ndarray) else T_set[b],
    "temperature_lower_deadband": T_lower[b].tolist() if isinstance(T_lower[b], np.ndarray) else T_lower[b],
    "temperature_upper_deadband": T_upper[b].tolist() if isinstance(T_upper[b], np.ndarray) else T_upper[b],
    "outdoor_temperature": T_out[b].tolist() if isinstance(T_out[b], np.ndarray) else T_out[b],
    "occupancy_profile": occupancy_schedule[b].tolist(),
    "temperature_below_lower_deadband": T_below_schedule[b].tolist(),
    "temperature_above_upper_deadband": T_above_schedule[b].tolist(),
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
}

with open("Results/schedules/central_optimisation_schedules_and_costs.json", "w") as f:
    json.dump(output_data, f, indent=2)

print("\nCentral optimization results saved to: Results/schedules/central_optimisation_schedules_and_costs.json")
print(f"Building 0 Summary:")
print(f"Total Electricity Cost: £{electricity_costs[b]:.2f}")
print(f"Total Gas Cost: £{gas_costs[b]:.2f}")
print(f"Total Cost: £{total_costs[b]:.2f}")
