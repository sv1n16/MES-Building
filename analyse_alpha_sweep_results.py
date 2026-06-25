"""
Analyze alpha sweep results: temperature and cost analysis
Plots temperature profiles and cost vs alpha
Uses Plotly for interactive visualizations
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

# Load data for electricity price
data = pd.read_csv("data\\processed_data_2018_02_21.csv")
data.columns = data.columns.str.strip().str.lower()
data["price"] = data["price (p/kwh)"] / 100.0
data["hour"] = data.index // 12
data_hr = pd.DataFrame()
data_hr["price"] = data.groupby("hour")["price"].mean()

# Results directory
results_dir = Path("Results/schedules/temp_analysis/temp_analysis_hp_boiler_27_02")

# Load all JSON files
json_files = sorted(results_dir.glob("building_0_alpha_*.json"))
print(f"\nFound {len(json_files)} result files")

if len(json_files) == 0:
    print("ERROR: No JSON files found in Results/schedules/temp_analysis/")
    exit(1)

# Parse results
alphas = []
temps = []
costs = []
temp_deviations = []
grid_imports = []

for json_file in json_files:
    with open(json_file, "r") as f:
        data_json = json.load(f)

    alpha = data_json["alpha"]
    T_in = np.array(data_json["T_in"])
    T_set = np.array(data_json["T_set"])
    q_boiler = np.array(data_json["q_boiler"])
    q_heatpump = np.array(data_json["q_heatpump"])
    grid_import = np.array(data_json["grid_import"])
    gas_consumption = q_boiler  # Using q_boiler as proxy for gas


    T_deviation = [T_in[t] - T_set[t] for t in range(len(T_in))]
    mean_deviation = np.mean(T_deviation)
    max_deviation = np.max(np.abs(T_deviation))
    print(f"   Temperature Deviation from Setpoint (Building 0):")
    print(f"      Mean deviation: {mean_deviation:.2f}°C")
    print(f"      Max absolute deviation: {max_deviation:.2f}°C")
    # Calculate metrics
    temp_deviation = np.mean((T_in - T_set) ** 2)
    temp_deviation_mean = np.mean(np.abs(T_in - T_set))
    # Calculate electricity cost
    electricity_price = data_hr["price"].values[: len(grid_import)]
    elec_cost = np.sum(electricity_price * grid_import)

    # Calculate gas cost (p/kWh = 5)
    gas_cost = np.sum(gas_consumption * 5 / 100)

    # Total cost
    total_cost = elec_cost + gas_cost

    alphas.append(alpha)
    temps.append(T_in)
    costs.append(total_cost)
    temp_deviations.append(temp_deviation_mean)
    grid_imports.append(grid_import)

    print(f"Alpha = {alpha:.4f}: Cost = £{total_cost:.2f}, Temp Dev = {temp_deviation_mean:.2f} °C")


alphas = np.array(alphas)
costs = np.array(costs)
temp_deviations = np.array(temp_deviations)

# Load setpoint once
with open(json_files[0], "r") as f:
    T_set_profile = np.array(json.load(f)["T_set"])

time_range = np.arange(len(temps[0]))

# ============================================================================
# PLOT 1: Temperature profiles for different alpha values
# ============================================================================
fig1 = go.Figure()

colorscale = px.colors.sample_colorscale("viridis", len(alphas))

for idx, (alpha_val, temp_profile) in enumerate(zip(alphas, temps)):
    fig1.add_trace(
        go.Scatter(
            x=time_range,
            y=temp_profile,
            mode="markers",
            name=f"α = {alpha_val:.4f}",
            line=dict(color=colorscale[idx], width=2),
        )
    )

fig1.add_trace(
    go.Scatter(
        x=time_range,
        y=T_set_profile,
        mode="markers",
        name="Setpoint",
        line=dict(color="red", width=2.5, dash="dash"),
        opacity=0.8,
    )
)

fig1.update_layout(
    title=dict(text="Building 0: Indoor Temperature Profiles for Different Alpha Values", font=dict(size=16)),
    xaxis_title="Time (hours)",
    yaxis_title="Temperature (°C)",
    legend=dict(font=dict(size=10)),
    width=1200,
    height=500,
    template="plotly_white",
)
fig1.show()
# fig1.write_html("Results/alpha_sweep_temperature_profiles.html")
# fig1.write_image("Results/alpha_sweep_temperature_profiles.png", scale=2)
print("\n✓ Saved: Results/alpha_sweep_temperature_profiles.html / .png")

# ============================================================================
# PLOT 2: Cost vs Alpha
# ============================================================================
fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=alphas,
        y=costs,
        mode="markers",
        name="Total Cost",
        marker=dict(color="blue", size=10, opacity=0.7),
        hovertemplate="α = %{x:.4f}<br>Cost = £%{y:.2f}<extra></extra>",
    )
)

fig2.update_layout(
    title=dict(text="Total Cost vs Penalty Weight", font=dict(size=16)),
    xaxis_title="Temperature Penalty Weight",
    yaxis_title="Total Cost (£)",
    legend=dict(font=dict(size=11)),
    width=900,
    height=550,
    template="plotly_white",
)
fig2.show()
# fig2.write_html("Results/alpha_sweep_cost_vs_alpha.html")
# fig2.write_image("Results/alpha_sweep_cost_vs_alpha.png", scale=2)
print("✓ Saved: Results/alpha_sweep_cost_vs_alpha.html / .png")

# ============================================================================
# PLOT 3: Temperature Deviation vs Alpha + Pareto frontier
# ============================================================================
fig3 = make_subplots(rows=1, cols=2, subplot_titles=("Temperature Comfort", "Cost-Comfort Trade-off"))

# Left: Temperature deviation vs alpha
fig3.add_trace(
    go.Scatter(
        x=alphas,
        y=temp_deviations,
        mode="markers",
        name="Temp Deviation²",
        line=dict(color="green"),
        marker=dict(size=6),
        hovertemplate="α = %{x:.4f}<br>Dev² = %{y:.2f} °C²<extra></extra>",
    ),
    row=1,
    col=1,
)

# Right: Pareto plot - Cost vs Comfort
fig3.add_trace(
    go.Scatter(
        x=temp_deviations,
        y=costs,
        mode="markers+text",
        name="Cost vs Comfort",
        marker=dict(color="red", size=10, opacity=0.7),
        text=[f"α={a:.2f}" for a in alphas],
        textposition="top right",
        textfont=dict(size=8),
        hovertemplate="Dev² = %{x:.2f} °C²<br>Cost = £%{y:.2f}<extra></extra>",
    ),
    row=1,
    col=2,
)

fig3.update_xaxes(title_text="Alpha (Temperature Penalty Weight)", row=1, col=1)
fig3.update_yaxes(title_text="Mean Temperature Deviation (°C)", row=1, col=1)
fig3.update_xaxes(title_text="Temperature Deviation (°C)", row=1, col=2)
fig3.update_yaxes(title_text="Total Cost (£)", row=1, col=2)

fig3.update_layout(
    title=dict(text="Building 0: Trade-off Analysis", font=dict(size=16)),
    width=1200,
    height=550,
    template="plotly_white",
    showlegend=False,
)
fig3.show()
# fig3.write_html("Results/alpha_sweep_tradeoff_analysis.html")
# fig3.write_image("Results/alpha_sweep_tradeoff_analysis.png", scale=2)
print("✓ Saved: Results/alpha_sweep_tradeoff_analysis.html / .png")

# ============================================================================
# PLOT 4: Overlay all temperature profiles with alpha color gradient
# ============================================================================
fig4 = go.Figure()

norm_alphas = (alphas - alphas.min()) / (alphas.max() - alphas.min() + 1e-12)
colorscale_plasma = px.colors.sample_colorscale("plasma", len(alphas))

for i, (alpha_val, norm_val) in enumerate(zip(alphas, norm_alphas)):
    fig4.add_trace(
        go.Scatter(
            x=time_range,
            y=temps[i],
            mode="markers",
            name=f"α = {alpha_val:.4f}",
            line=dict(color=colorscale_plasma[i], width=1.5),
            opacity=0.7,
            showlegend=False,
            hovertemplate=f"α={alpha_val:.4f}<br>Time: %{{x}}<br>Temp: %{{y:.2f}}°C<extra></extra>",
        )
    )

# Setpoint
fig4.add_trace(
    go.Scatter(
        x=time_range,
        y=T_set_profile,
        mode="markers",
        name="Setpoint",
        line=dict(color="black", width=3, dash="dash"),
        opacity=1.0,
    )
)

# Add a colorbar via a dummy scatter with colorscale
fig4.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            colorscale="plasma",
            cmin=alphas.min(),
            cmax=alphas.max(),
            color=[alphas.min()],
            colorbar=dict(title="Alpha (Temperature Penalty)", titlefont=dict(size=11)),
            showscale=True,
        ),
        showlegend=False,
    )
)

fig4.update_layout(
    title=dict(text="Building 0: Temperature Profiles - All Alpha Values (Color = Alpha)", font=dict(size=16)),
    xaxis_title="Time (hours)",
    yaxis_title="Temperature (°C)",
    legend=dict(font=dict(size=11)),
    width=1200,
    height=550,
    template="plotly_white",
)
fig4.show()
# fig4.write_html("Results/alpha_sweep_temperature_heatmap.html")
# fig4.write_image("Results/alpha_sweep_temperature_heatmap.png", scale=2)
print("✓ Saved: Results/alpha_sweep_temperature_heatmap.html / .png")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 70)
print("ALPHA SWEEP SUMMARY STATISTICS")
print("=" * 70)
print(f"Alpha range: {alphas.min():.4f} to {alphas.max():.4f}")
print(f"Number of alpha values: {len(alphas)}")
print(f"\nCost:")
print(f"  Min: £{costs.min():.2f} (at α = {alphas[np.argmin(costs)]:.4f})")
print(f"  Max: £{costs.max():.2f} (at α = {alphas[np.argmax(costs)]:.4f})")
print(f"  Range: £{costs.max() - costs.min():.2f}")

print(f"\nTemperature Comfort (Deviation²):")
print(f"  Best: {temp_deviations.min():.2f} °C² (at α = {alphas[np.argmin(temp_deviations)]:.4f})")
print(f"  Worst: {temp_deviations.max():.2f} °C² (at α = {alphas[np.argmax(temp_deviations)]:.4f})")
print(f"  Improvement: {(1 - temp_deviations.min() / temp_deviations.max()) * 100:.1f}%")

print(f"\n" + "=" * 70)
print("✓ Analysis complete! Check Results/ for plots (.html and .png)")
print("=" * 70)
