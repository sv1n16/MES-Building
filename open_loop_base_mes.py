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


time_horizon = 12  # 24 hours
delta_t = 1  # 1 hour time step
radiation = pd.read_csv("data\\radiation.csv").values.flatten()
load = pd.read_csv("data\\load.csv").values.flatten()
price = pd.read_csv("data\\price.csv").values.flatten()
heat_load = pd.read_csv("data\\heat_load.csv").values.flatten()

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
bd = Building([bat, pv])

# Update the battery schedule
# bp_el_schedule = bat.p_el_demand - bat.p_el_supply
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


boiler = GasBoiler(time_horizon=time_horizon)
boiler.set_thermal_output(p_th_heat_boiler)

# Update the building's power schedule to include the battery schedule and PV schedule
bd.p_el_schedule = load + bat.power_el_schedule + pv.p_el_schedule + eh.p_el_schedule

# Debug: Print the battery's power schedule
print("Battery's power schedule (p_el_schedule):")
print(bat.power_el_schedule)

# Calculate the cost at each time step
costs = np.array([price[t] * bd.p_el_schedule[t] for t in range(time_horizon)])
gas_costs = np.array(boiler.gas_consumption_schedule) * boiler.gas_price
print("Electricity Costs:", sum(costs))
total_costs = costs + gas_costs
print("Total costs (electricity + gas):", sum(total_costs))


# Plot the cost over the simulation horizon
plot_time = list(range(time_horizon))
fig = make_subplots(rows=9, cols=1, shared_xaxes=True)

fig.add_trace(
    go.Scatter(x=plot_time, y=bat.energy_el_schedule, name="battery state of charge (kWh)"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(x=plot_time, y=bat.power_el_schedule, name="battery charge/discharge (kWh)"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(x=plot_time, y=bd.p_el_schedule, name="building import/export"),
    row=2,
    col=1,
)
fig.add_trace(go.Scatter(x=plot_time, y=pv.p_el_schedule, name="pv power export"), row=3, col=1)
fig.add_trace(go.Scatter(x=plot_time, y=load, name="forecasted load (kW)"), row=4, col=1)
fig.add_trace(go.Scatter(x=plot_time, y=price, name="energy market price (ct/kWh)"), row=5, col=1)
fig.add_trace(
    go.Scatter(x=plot_time, y=pv.p_el_supply, name="photovoltaic supply (kWh)"),
    row=6,
    col=1,
)
fig.add_trace(go.Scatter(x=plot_time, y=costs, name="cost over time"), row=7, col=1)
fig.add_trace(go.Scatter(x=plot_time, y=eh.p_el_schedule, name="heat pump power consumption (kW)"), row=8, col=1)
fig.add_trace(
    go.Scatter(x=plot_time, y=p_th_heat_boiler, name="boiler thermal output (kWh)"),
    row=9,
    col=1,
)
fig.add_trace(
    go.Scatter(x=plot_time, y=boiler.gas_consumption_schedule, name="boiler gas input (kWh)"),
    row=9,
    col=1,
)
fig.add_trace(
    go.Scatter(x=plot_time, y=heat_load, name="Heat Load(kWh)"),
    row=9,
    col=1,
)
# Update xaxis properties
fig.update_xaxes(title_text="Time", row=1, col=1)
fig.update_xaxes(title_text="Time", row=2, col=1)
fig.update_xaxes(title_text="Time", row=3, col=1)
fig.update_xaxes(title_text="Time", row=4, col=1)
fig.update_xaxes(title_text="Time", row=5, col=1)
fig.update_xaxes(title_text="Time", row=6, col=1)
fig.update_xaxes(title_text="Time", row=7, col=1)
fig.update_xaxes(title_text="Time", row=8, col=1)
fig.update_xaxes(title_text="Time", row=9, col=1)

# Update yaxis properties
fig.update_yaxes(title_text="Battery", title_font=dict(size=8), row=1, col=1)
fig.update_yaxes(title_text="Building (kWh)", title_font=dict(size=8), row=2, col=1)
fig.update_yaxes(title_text="PV (kWh)", title_font=dict(size=8), row=3, col=1)
fig.update_yaxes(title_text="Load (kWh)", title_font=dict(size=8), row=4, col=1)
fig.update_yaxes(title_text="Energy Price", title_font=dict(size=8), row=5, col=1)
fig.update_yaxes(title_text="PV Generation", title_font=dict(size=8), row=6, col=1)
fig.update_yaxes(title_text="Cost (ct)", title_font=dict(size=8), row=7, col=1)
fig.update_yaxes(title_text="Heat/Gas (kWh)", title_font=dict(size=8), row=9, col=1)

fig.update_layout(
    title_text="Scheduling Results Local Open Loop",
    title_x=0.5,
    title_y=0.95,
    yaxis=dict(title_font=dict(size=10), tickfont=dict(size=12)),
)
fig.update_layout(uniformtext_minsize=8)
fig.show()
