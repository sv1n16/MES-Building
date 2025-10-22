import plotly.graph_objects as go
import json
import numpy as np

# Load cost arrays and price from JSON files
with open("Results/schedules/open_loop_schedules_and_costs.json", "r") as f:
    open_loop_data = json.load(f)
with open("Results/schedules/central_optimisation_schedules_and_costs.json", "r") as f:
    optimised_data = json.load(f)

open_loop_costs = np.array(open_loop_data["total_costs"])
optimised_costs = np.array(optimised_data["total_costs"])
electricity_price = np.array(open_loop_data["electricity_price"])
temperature_setpoint = np.array([17, 17, 17.5, 17.5, 17.5, 17.5, 20, 20, 20, 20, 18, 18])

plot_time = np.arange(len(open_loop_costs))

fig = go.Figure()
fig.add_trace(go.Scatter(x=plot_time, y=open_loop_costs, mode="lines+markers", name="Open Loop Cost (£/h)"))
fig.add_trace(go.Scatter(x=plot_time, y=optimised_costs, mode="lines+markers", name="Optimised Cost (£/h)"))
fig.add_trace(go.Scatter(x=plot_time, y=electricity_price, mode="lines", name="Electricity Price (p/kWh)", yaxis="y2"))

fig.update_layout(
    title="Cost and Electricity Price Over Time",
    xaxis_title="Time (h)",
    yaxis=dict(title="Cost (£/h)"),
    yaxis2=dict(title="Electricity Price (p/kWh)", overlaying="y", side="right"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.3,
        xanchor="center",
        x=0.5,
        font=dict(size=14),
    ),
    template="plotly_white",
    margin=dict(l=40, r=40, t=40, b=40),
)
fig.show()

# Battery charging schedule comparison
open_loop_charge = np.array(open_loop_data["battery_soc_schedule"])
optimised_charge = np.array(optimised_data["battery_soc_schedule"])

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=plot_time, y=open_loop_charge, mode="lines+markers", name="Open Loop Battery Charge (kW)"))
fig2.add_trace(go.Scatter(x=plot_time, y=optimised_charge, mode="lines+markers", name="Optimised Battery Charge (kW)"))
fig2.update_layout(
    title="Battery Charging Schedule",
    xaxis_title="Time (h)",
    yaxis_title="Battery State of Charge (kWh)",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.3,
        xanchor="center",
        x=0.5,
        font=dict(size=14),
    ),
    template="plotly_white",
    margin=dict(l=40, r=40, t=40, b=40),
)
fig2.show()

# Define arrays for heat output and temperature plots
open_loop_hp = np.array(open_loop_data["heatpump_thermal_output"])
optimised_hp = np.array(optimised_data["heatpump_thermal_output"])
open_loop_boiler = np.array(open_loop_data["boiler_thermal_output"])
optimised_boiler = np.array(optimised_data["boiler_thermal_output"])
open_loop_temp = np.array(open_loop_data["indoor_temperature"])
optimised_temp = np.array(optimised_data["indoor_temperature"])
open_loop_temp_setpoint = np.array(open_loop_data["temperature_setpoint"])
optimised_temp_setpoint = np.array(optimised_data["temperature_setpoint"])

# Heat pump and boiler output comparison (figure 3)
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=plot_time, y=open_loop_hp, mode="lines+markers", name="Open Loop Heat Pump Output (kW)"))
fig3.add_trace(go.Scatter(x=plot_time, y=optimised_hp, mode="lines+markers", name="Optimised Heat Pump Output (kW)"))
fig3.add_trace(go.Scatter(x=plot_time, y=open_loop_boiler, mode="lines+markers", name="Open Loop Boiler Output (kW)"))
fig3.add_trace(go.Scatter(x=plot_time, y=optimised_boiler, mode="lines+markers", name="Optimised Boiler Output (kW)"))
fig3.update_layout(
    title="Heat Pump and Boiler Output Comparison",
    xaxis_title="Time (h)",
    yaxis_title="Thermal Output (kW)",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.3,
        xanchor="center",
        x=0.5,
        font=dict(size=14),
    ),
    template="plotly_white",
    margin=dict(l=40, r=40, t=40, b=40),
)
fig3.show()

# Indoor temperature and setpoint comparison (figure 4)
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=plot_time, y=open_loop_temp, mode="lines", name="Open Loop Indoor Temp (°C)"))
fig4.add_trace(go.Scatter(x=plot_time, y=optimised_temp, mode="lines", name="Optimised Indoor Temp (°C)"))
fig4.add_trace(
    go.Scatter(
        x=plot_time,
        y=open_loop_temp_setpoint,
        mode="lines",
        name="Open Loop Temp Setpoint (°C)",
        line=dict(dash="dash"),
    )
)
fig4.add_trace(
    go.Scatter(
        x=plot_time,
        y=optimised_temp_setpoint,
        mode="lines",
        name="Optimised Temp Setpoint (°C)",
        line=dict(dash="dash"),
    )
)
fig4.update_layout(
    title="Indoor Temperature and Setpoint Comparison",
    xaxis_title="Time (h)",
    yaxis_title="Temperature (°C)",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.3,
        xanchor="center",
        x=0.5,
        font=dict(size=14),
    ),
    template="plotly_white",
    margin=dict(l=40, r=40, t=40, b=40),
)
fig4.show()
