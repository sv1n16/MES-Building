import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
import os

# Load complete data from single building JSON files
with open("Results/schedules/open_loop_single_building_schedules_and_costs.json", "r") as f:
    open_loop_data = json.load(f)
with open("Results/schedules/central_optimisation_schedules_and_costs.json", "r") as f:
    optimised_data = json.load(f)

# Extract costs - handle both old and new JSON formats
if "costs" in open_loop_data:
    open_loop_costs = open_loop_data["costs"]
    optimised_costs = optimised_data["costs"]
else:
    # Calculate from cost arrays if costs dictionary doesn't exist
    open_loop_elec_cost = np.sum(np.array(open_loop_data.get("electricity_costs", [])))
    open_loop_gas_cost = np.sum(np.array(open_loop_data.get("gas_costs", [])))
    open_loop_costs = {
        "total_electricity_cost_gbp": float(open_loop_elec_cost),
        "total_gas_cost_gbp": float(open_loop_gas_cost),
        "total_cost_gbp": float(open_loop_elec_cost + open_loop_gas_cost),
    }

    optimised_elec_cost = np.sum(np.array(optimised_data.get("electricity_costs", [])))
    optimised_gas_cost = np.sum(np.array(optimised_data.get("gas_costs", [])))
    optimised_costs = {
        "total_electricity_cost_gbp": float(optimised_elec_cost),
        "total_gas_cost_gbp": float(optimised_gas_cost),
        "total_cost_gbp": float(optimised_elec_cost + optimised_gas_cost),
    }

print("Open Loop Total Costs: £{:.2f}".format(open_loop_costs["total_cost_gbp"]))
print("Optimised Total Costs: £{:.2f}".format(optimised_costs["total_cost_gbp"]))

# Extract peak demand information
open_loop_peak = open_loop_data.get("peak_demand", {})
optimised_peak = optimised_data.get("peak_demand", {})

print("\n" + "=" * 70)
print("PEAK DEMAND ANALYSIS")
print("=" * 70)
print("\nELECTRICITY DEMAND:")
print(
    f"  Open Loop Peak: {open_loop_peak.get('peak_electricity_demand_kw', 'N/A'):.2f} kW at hour {open_loop_peak.get('time_of_peak_electricity_demand_hour', 'N/A')}"
)
print(
    f"  Optimised Peak: {optimised_peak.get('peak_electricity_demand_kw', 'N/A'):.2f} kW at hour {optimised_peak.get('time_of_peak_electricity_demand_hour', 'N/A')}"
)
print(f"  Open Loop Average: {open_loop_peak.get('average_electricity_demand_kw', 'N/A'):.2f} kW")
print(f"  Optimised Average: {optimised_peak.get('average_electricity_demand_kw', 'N/A'):.2f} kW")


electricity_price = np.array(open_loop_data["electricity_price"])
open_loop_costs_array = np.array(open_loop_data["electricity_costs"])
optimised_costs_array = np.array(optimised_data["electricity_costs"])

plot_time = np.arange(len(open_loop_costs_array))

fig = go.Figure()
fig.add_trace(go.Scatter(x=plot_time, y=open_loop_costs_array, mode="lines+markers", name="Open Loop Cost (£/h)"))
fig.add_trace(go.Scatter(x=plot_time, y=optimised_costs_array, mode="lines+markers", name="Optimised Cost (£/h)"))
fig.add_trace(go.Scatter(x=plot_time, y=electricity_price, mode="lines", name="Electricity Price (£/kWh)", yaxis="y2"))

fig.update_layout(
    title="Cost and Electricity Price Over Time",
    xaxis_title="Time (h)",
    yaxis=dict(title="Cost (£/h)"),
    yaxis2=dict(title="Electricity Price (£/kWh)", overlaying="y", side="right"),
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
        name="Setpoint (°C)",
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

# ============================================================================
# PEAK DEMAND COMPARISON FIGURE
# ============================================================================
fig_peak = go.Figure()

# Extract peak demands
peak_metrics = ["Electricity", "Thermal", "Battery\nCharge", "Battery\nDischarge", "Heat Pump", "Boiler"]
open_loop_peaks = open_loop_peak.get("peak_electricity_demand_kw", 0)

optimised_peaks = optimised_peak.get("peak_electricity_demand_kw", 0)


fig_peak.add_trace(go.Bar(x=peak_metrics, y=open_loop_peaks, name="Open Loop", marker_color="steelblue"))
fig_peak.add_trace(go.Bar(x=peak_metrics, y=optimised_peaks, name="Optimised", marker_color="darkorange"))

fig_peak.update_layout(
    title="Peak Demand Comparison",
    xaxis_title="Component",
    yaxis_title="Peak Power (kW)",
    barmode="group",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="center",
        x=0.5,
        font=dict(size=12),
    ),
    template="plotly_white",
    margin=dict(l=40, r=40, t=40, b=100),
)
fig_peak.show()

# ============================================================================
# AVERAGE DEMAND COMPARISON FIGURE
# ============================================================================
fig_avg = go.Figure()

# Extract average demands
avg_metrics = ["Electricity", "Thermal"]
open_loop_avgs = [
    open_loop_peak.get("average_electricity_demand_kw", 0),
    open_loop_peak.get("average_thermal_demand_kw", 0),
]
optimised_avgs = [
    optimised_peak.get("average_electricity_demand_kw", 0),
    optimised_peak.get("average_thermal_demand_kw", 0),
]

fig_avg.add_trace(go.Bar(x=avg_metrics, y=open_loop_avgs, name="Open Loop", marker_color="steelblue"))
fig_avg.add_trace(go.Bar(x=avg_metrics, y=optimised_avgs, name="Optimised", marker_color="darkorange"))

fig_avg.update_layout(
    title="Average Demand Comparison",
    xaxis_title="Component",
    yaxis_title="Average Power (kW)",
    barmode="group",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="center",
        x=0.5,
        font=dict(size=12),
    ),
    template="plotly_white",
    margin=dict(l=40, r=40, t=40, b=100),
)
fig_avg.show()

# ============================================================================
# TOTAL DEMAND VS TIME COMPARISON
# ============================================================================

# Check if demand vs time files exist, then load and plot
if os.path.exists("Results/schedules/open_loop_demand_vs_time.json") and os.path.exists(
    "Results/schedules/central_optimisation_demand_vs_time.json"
):
    # Load demand vs time data
    with open("Results/schedules/open_loop_demand_vs_time.json", "r") as f:
        open_loop_demand_vs_time = json.load(f)
    with open("Results/schedules/central_optimisation_demand_vs_time.json", "r") as f:
        optimised_demand_vs_time = json.load(f)

    time_hrs = open_loop_demand_vs_time["time_hours"]

    # ============================================================================
    # FIGURE: Total Electricity Demand vs Time
    # ============================================================================
    fig_elec_demand = go.Figure()

    fig_elec_demand.add_trace(
        go.Scatter(
            x=time_hrs,
            y=open_loop_demand_vs_time["total_electricity_demand_kw"],
            mode="lines",
            name="Open Loop",
            line=dict(color="steelblue", width=2),
        )
    )
    fig_elec_demand.add_trace(
        go.Scatter(
            x=time_hrs,
            y=optimised_demand_vs_time["total_electricity_demand_kw"],
            mode="lines",
            name="Optimised",
            line=dict(color="darkorange", width=2),
        )
    )

    fig_elec_demand.update_layout(
        title="Total Electricity Demand vs Time",
        xaxis_title="Hour of Day",
        yaxis_title="Total Electricity Demand (kW)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=100),
    )
    fig_elec_demand.show()


# ============================================================================
# PLOT OPEN LOOP RESULTS IN CENTRAL OPTIMIZATION FORMAT
# ============================================================================

time_horizon = len(open_loop_data["load"])
plot_time_list = list(range(time_horizon))

# Extract all data from open_loop_data
load_schedule = np.array(open_loop_data["load"])
pv_schedule = np.array(open_loop_data["pv_supply"])
net_charge_schedule = np.array(open_loop_data["battery_net_charge"])
soc_schedule = np.array(open_loop_data["battery_soc_schedule"])
heatpump_schedule = np.array(open_loop_data["heatpump_electrical"])
T_in = np.array(open_loop_data["indoor_temperature"])
T_out = np.array(open_loop_data["outdoor_temperature"])
T_set = np.array(open_loop_data["temperature_setpoint"])
q_boiler_schedule = np.array(open_loop_data["boiler_thermal_output"])
q_heatpump_schedule = np.array(open_loop_data["heatpump_thermal_output"])
q_total_schedule = q_boiler_schedule + q_heatpump_schedule
electricity_price = np.array(open_loop_data["electricity_price"])
elec_cost_schedule = np.array(open_loop_data["electricity_costs"])
gas_cost_schedule = np.array(open_loop_data["gas_costs"])
grid_import = np.array(open_loop_data["grid_import"])

# Extract all data from optimised_data
opt_load_schedule = np.array(optimised_data["load"])
opt_pv_schedule = np.array(optimised_data["pv_supply"])
opt_net_charge_schedule = np.array(optimised_data["battery_net_charge"])
opt_soc_schedule = np.array(optimised_data["battery_soc_schedule"])
opt_heatpump_schedule = np.array(optimised_data["heatpump_electrical"])
opt_T_in = np.array(optimised_data["indoor_temperature"])
opt_T_out = np.array(optimised_data["outdoor_temperature"])
opt_T_set = np.array(optimised_data["temperature_setpoint"])
opt_q_boiler_schedule = np.array(optimised_data["boiler_thermal_output"])
opt_q_heatpump_schedule = np.array(optimised_data["heatpump_thermal_output"])
opt_q_total_schedule = np.array(optimised_data["total_heat_demand"])
opt_electricity_price = np.array(optimised_data["electricity_price"])
opt_elec_cost_schedule = np.array(optimised_data["electricity_costs"])
opt_gas_cost_schedule = np.array(optimised_data["gas_costs"])
opt_grid_import = np.array(optimised_data["grid_import"])

# ============================================================================
# FIGURE 1: Battery and Heat Pump Operation
# ============================================================================
fig_col1 = go.Figure()

# External inputs (non-decision variables)
fig_col1.add_trace(go.Scatter(y=load_schedule, mode="lines", name="Load", line=dict(color="black"), showlegend=True))
fig_col1.add_trace(
    go.Scatter(y=pv_schedule, mode="lines", name="PV Supply", line=dict(color="orange"), showlegend=True)
)

# Open Loop - Decision variables
fig_col1.add_trace(
    go.Scatter(
        y=net_charge_schedule,
        mode="lines",
        name="Battery Net Charge (Open Loop)",
        line=dict(color="blue"),
        showlegend=True,
    )
)
fig_col1.add_trace(
    go.Scatter(
        y=soc_schedule, mode="lines", name="Battery SOC (Open Loop)", line=dict(color="purple"), showlegend=True
    )
)
fig_col1.add_trace(
    go.Scatter(
        y=heatpump_schedule, mode="lines", name="Heat Pump (Open Loop)", line=dict(color="green"), showlegend=True
    )
)

# Optimised - Decision variables
fig_col1.add_trace(
    go.Scatter(
        y=opt_net_charge_schedule,
        mode="lines",
        name="Battery Net Charge (Optimised)",
        line=dict(color="blue", dash="dash"),
        showlegend=True,
    )
)
fig_col1.add_trace(
    go.Scatter(
        y=opt_soc_schedule,
        mode="lines",
        name="Battery SOC (Optimised)",
        line=dict(color="purple", dash="dash"),
        showlegend=True,
    )
)
fig_col1.add_trace(
    go.Scatter(
        y=opt_heatpump_schedule,
        mode="lines",
        name="Heat Pump (Optimised)",
        line=dict(color="green", dash="dash"),
        showlegend=True,
    )
)


fig_col1.update_layout(
    title_text="Battery and Heat Pump Operation",
    xaxis_title="Time (h)",
    yaxis_title="Power/Energy (kW/kWh)",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(
        orientation="v",
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
    ),
)

fig_col1.show()

# ============================================================================
# FIGURE 2: Building Temperatures and Thermal Outputs
# ============================================================================
fig_col2 = go.Figure()

# External inputs (non-decision variables)
fig_col2.add_trace(
    go.Scatter(y=T_out, mode="lines", name="Outdoor Temp", line=dict(color="deepskyblue"), showlegend=True)
)
fig_col2.add_trace(
    go.Scatter(y=T_set, mode="lines", name="Setpoint", line=dict(color="darkgreen", dash="dash"), showlegend=True)
)

# Open Loop - Decision variables
fig_col2.add_trace(
    go.Scatter(y=T_in, mode="lines", name="Indoor Temp (Open Loop)", line=dict(color="firebrick"), showlegend=True)
)
fig_col2.add_trace(
    go.Scatter(
        y=q_boiler_schedule,
        mode="lines",
        name="Boiler Output (Open Loop)",
        line=dict(color="red", width=2),
        showlegend=True,
    )
)
fig_col2.add_trace(
    go.Scatter(
        y=q_heatpump_schedule,
        mode="lines",
        name="Heat Pump Output (Open Loop)",
        line=dict(color="cyan", width=2),
        showlegend=True,
    )
)
fig_col2.add_trace(
    go.Scatter(
        y=q_total_schedule,
        mode="lines",
        name="Total Heat Demand (Open Loop)",
        line=dict(color="black", dash="dash"),
        showlegend=True,
    )
)

# Optimised - Decision variables
fig_col2.add_trace(
    go.Scatter(
        y=opt_T_in,
        mode="lines",
        name="Indoor Temp (Optimised)",
        line=dict(color="crimson", dash="dash"),
        showlegend=True,
    )
)
fig_col2.add_trace(
    go.Scatter(
        y=opt_q_boiler_schedule,
        mode="lines",
        name="Boiler Output (Optimised)",
        line=dict(color="red", width=2, dash="dash"),
        showlegend=True,
    )
)
fig_col2.add_trace(
    go.Scatter(
        y=opt_q_heatpump_schedule,
        mode="lines",
        name="Heat Pump Output (Optimised)",
        line=dict(color="cyan", width=2, dash="dash"),
        showlegend=True,
    )
)
fig_col2.add_trace(
    go.Scatter(
        y=opt_q_total_schedule,
        mode="lines",
        name="Total Heat Demand (Optimised)",
        line=dict(color="black", dash="dot"),
        showlegend=True,
    )
)

fig_col2.update_layout(
    title_text="Building Temperatures and Thermal Outputs",
    xaxis_title="Time (h)",
    yaxis_title="Temperature (°C) / Thermal Power (kW)",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(
        orientation="v",
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
    ),
)

fig_col2.show()

# ============================================================================
# FIGURE 3: Costs and Electricity Price
# ============================================================================
fig_col3 = go.Figure()

# External input (non-decision variable)
fig_col3.add_trace(
    go.Scatter(
        y=electricity_price,
        mode="lines",
        name="Electricity Price",
        line=dict(color="pink"),
        showlegend=True,
    )
)

# Open Loop
fig_col3.add_trace(
    go.Scatter(
        y=elec_cost_schedule,
        mode="lines",
        name="Electricity Cost (Open Loop)",
        line=dict(color="gold"),
        showlegend=True,
    )
)
fig_col3.add_trace(
    go.Scatter(
        y=gas_cost_schedule,
        mode="lines",
        name="Gas Cost (Open Loop)",
        line=dict(color="brown"),
        showlegend=True,
    )
)

# Optimised
fig_col3.add_trace(
    go.Scatter(
        y=opt_elec_cost_schedule,
        mode="lines",
        name="Electricity Cost (Optimised)",
        line=dict(color="gold", dash="dash"),
        showlegend=True,
    )
)
fig_col3.add_trace(
    go.Scatter(
        y=opt_gas_cost_schedule,
        mode="lines",
        name="Gas Cost (Optimised)",
        line=dict(color="brown", dash="dash"),
        showlegend=True,
    )
)

fig_col3.update_layout(
    title_text="Costs and Electricity Price",
    xaxis_title="Time (h)",
    yaxis_title="Cost (£)",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(
        orientation="v",
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
    ),
)

fig_col3.show()

# ============================================================================
# PEAK DEMAND TIMES BY BUILDING
# ============================================================================

# Load multi-building peak demand data
with open("Results/schedules/open_loop_all_buildings_peak_demand.json", "r") as f:
    open_loop_all_buildings = json.load(f)
with open("Results/schedules/central_optimisation_all_buildings_peak_demand.json", "r") as f:
    optimised_all_buildings = json.load(f)

# Extract building names and peak times
building_names = list(open_loop_all_buildings.keys())
open_loop_elec_peaks = [open_loop_all_buildings[b]["time_of_peak_electricity_demand_hour"] for b in building_names]
optimised_elec_peaks = [optimised_all_buildings[b]["time_of_peak_electricity_demand_hour"] for b in building_names]
open_loop_thermal_peaks = [open_loop_all_buildings[b]["time_of_peak_thermal_demand_hour"] for b in building_names]
optimised_thermal_peaks = [optimised_all_buildings[b]["time_of_peak_thermal_demand_hour"] for b in building_names]

# ============================================================================
# FIGURE: Peak Electricity Demand Times by Building
# ============================================================================
fig_elec_times = go.Figure()

fig_elec_times.add_trace(
    go.Bar(
        x=building_names,
        y=open_loop_elec_peaks,
        name="Open Loop",
        marker_color="steelblue",
    )
)
fig_elec_times.add_trace(
    go.Bar(
        x=building_names,
        y=optimised_elec_peaks,
        name="Optimised",
        marker_color="darkorange",
    )
)

fig_elec_times.update_layout(
    title="Peak Electricity Demand Times by Building",
    xaxis_title="Building",
    yaxis_title="Hour of Day (0-23)",
    barmode="group",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="center",
        x=0.5,
        font=dict(size=12),
    ),
    template="plotly_white",
    margin=dict(l=40, r=40, t=40, b=100),
    yaxis=dict(range=[0, 24]),
)
fig_elec_times.show()

# ============================================================================
# FIGURE: Peak Thermal Demand Times by Building
# ============================================================================
fig_thermal_times = go.Figure()

fig_thermal_times.add_trace(
    go.Bar(
        x=building_names,
        y=open_loop_thermal_peaks,
        name="Open Loop",
        marker_color="steelblue",
    )
)
fig_thermal_times.add_trace(
    go.Bar(
        x=building_names,
        y=optimised_thermal_peaks,
        name="Optimised",
        marker_color="darkorange",
    )
)

fig_thermal_times.update_layout(
    title="Peak Thermal Demand Times by Building",
    xaxis_title="Building",
    yaxis_title="Hour of Day (0-23)",
    barmode="group",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="center",
        x=0.5,
        font=dict(size=12),
    ),
    template="plotly_white",
    margin=dict(l=40, r=40, t=40, b=100),
    yaxis=dict(range=[0, 24]),
)
fig_thermal_times.show()
