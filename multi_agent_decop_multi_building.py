from gurobipy import Model, GRB
import pyomo.environ as pyo
import json
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import sys

model = pyo.ConcreteModel()

### Units information ####
#
#
# # Load: W
# Price: p/kWh
# Gas price: p/kWh


###


data = pd.read_csv("data\\processed_data_2018_02_21.csv")
data.columns = data.columns.str.strip().str.lower()
cols = data.columns[data.columns.str.contains("consumption", case=False)]
cols = cols[1:5]  # all building consumption columns
if "price" in data.columns:
    data["price"] = data["price"] / 100.0

# --- create hour index ---
data["hour"] = data.index // 60

# --- aggregate ---
data_hr = pd.DataFrame()
# price is already converted from p/kWh to £/kWh above; do NOT divide by 100 again
data_hr["price"] = data.groupby("hour")["price"].mean()
data_hr["pv"] = data.groupby("hour")["pv"].mean()
data_hr["outdoor temperature"] = data.groupby("hour")["outdoor temperature"].mean()
data_hr["temperature setpoint"] = data.groupby("hour")["temperature setpoint"].mean()

for c in cols:
    data_hr[c] = data.groupby("hour")[c].mean()

# Convert electricity price from p/kWh to £/kWh

print("[sanity] price range (hourly) £/kWh:", data_hr["price"].min(), data_hr["price"].max())
print("[sanity] pv min/max (kW):", data_hr["pv"].min(), data_hr["pv"].max())
for c in cols:
    print(f"[sanity] load '{c}' min/max (kW):", data_hr[c].min(), data_hr[c].max())
print(
    "[sanity] temperature setpoint min/max (°C):",
    data_hr["temperature setpoint"].min(),
    data_hr["temperature setpoint"].max(),
)
data = data_hr

irradiance = data["pv"].values.flatten()
# Parameters
time_horizon = int(len(data))  # Number of hours in the time horizon
# Time step (explicit units): hours. Set dt_hours=1.0 for hourly timesteps,
# or dt_hours = 1.0/60.0 for 1-minute timesteps (in hours).

battery_capacity = 12  # kWh
max_power = 4.6  # kW
eta_charge = 0.9  # Charging efficiency
eta_discharge = 0.9
n_buildings = len(cols)  # Number of buildings
p_th_nom = 12
T_ref = 7
cop = np.ones(time_horizon) * 2.18
T_init = 20
max_thermal_power = 20.0  # kW
efficiency = 0.9  # Boiler efficiency (fraction)
gas_price = 1  # Gas price (p/kWh)
# ---- Sets ----
model.buildings = pyo.RangeSet(0, n_buildings - 1)  # buildings
model.t = pyo.RangeSet(0, time_horizon - 1)
dt = 1  # 60  # minutes
model.dt = pyo.Param(initialize=dt)
alpha = 1000
# ---- Parameters ----

# pv = PVModule(time_horizon=time_horizon, start_point=1, radiation=radiation, area=25.0, beta=30.0, eta_noct=0.15)
pv_data = irradiance
# Example: PV supply for each building and time (assuming same PV for all buildings)
pv_supplies = {(b, t): irradiance[t] for b in range(n_buildings) for t in range(time_horizon)}
model.pv_supply = pyo.Param(model.buildings, model.t, initialize=pv_supplies)

# Now, define variables indexed by building and time

# Battery variables

model.charge = pyo.Var(model.buildings, model.t, bounds=(0, max_power), initialize=0)
model.discharge = pyo.Var(model.buildings, model.t, bounds=(0, max_power), initialize=0)
## Initial SOC: random values between 0 and 1 (fractions) for each building,
## converted to kWh by multiplying by battery_capacity
initial_soc_frac = np.random.rand(n_buildings)
initial_soc = [float(f * battery_capacity) for f in initial_soc_frac]
print("[sanity] initial SOC fractions:", initial_soc_frac)
print("[sanity] initial SOC (kWh):", initial_soc)


### Plot Demand ###

fig = go.Figure()
for col in cols:
    fig.add_trace(go.Scatter(x=data.index, y=data[col], name="Consumption (kW)"))
    fig.add_trace(go.Scatter(x=data.index, y=irradiance, name="PV Supply (kW)"))


fig.update_xaxes(title_text="Time")
fig.update_yaxes(title_text="Power (kW)")
fig.show()
