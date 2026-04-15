"""
Deadband Penalty Parameter Sweep Analysis

This script performs a sensitivity analysis on the deadband penalty coefficients
(beta for lower bound, theta for upper bound) to find the optimal balance between
energy cost and thermal comfort.

It tests a range of beta and theta values, records the results, and generates
plots to visualize the cost-comfort trade-off (Pareto frontier).
"""

import subprocess
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import tempfile
import shutil

print("=" * 80)
print("DEADBAND PENALTY PARAMETER SWEEP ANALYSIS")
print("=" * 80)

# Define parameter ranges to test
# beta: penalty for lower bound violations (higher = more aggressive heating, more cost)
# theta: penalty for upper bound violations (higher = more aggressive cooling, more cost)

beta_values = [10, 50, 100, 200, 300, 500, 750, 1000]
theta_values = [5, 10, 20, 50, 100, 200, 300, 500]

results = []

print("\nParameter Combinations to Test:")
print(f"  Beta values (lower bound penalty): {beta_values}")
print(f"  Theta values (upper bound penalty): {theta_values}")
print(f"  Total combinations: {len(beta_values) * len(theta_values)}")
print("\nStarting sweep... (this may take several minutes)\n")

# Read the original model file
model_file = "central_optimisation_multi_building_electric_hp_boiler_ideal_thermal_deadband.py"
with open(model_file, "r") as f:
    original_model = f.read()

combination_count = 0
total_combinations = len(beta_values) * len(theta_values)

# Test each combination
for beta in beta_values:
    for theta in theta_values:
        combination_count += 1
        print(f"[{combination_count}/{total_combinations}] Testing beta={beta}, theta={theta}...", end=" ")

        # Create a temporary copy of the model with modified parameters
        temp_model = original_model.replace(
            f"beta = 300.0  # Deadband penalty coefficient for lower bound violation (higher penalty)",
            f"beta = {float(beta)}  # Deadband penalty coefficient for lower bound violation (higher penalty)",
        )
        temp_model = temp_model.replace(
            f"theta = 50.0  # Upper deadband penalty coefficient for upper bound violation",
            f"theta = {float(theta)}  # Upper deadband penalty coefficient for upper bound violation",
        )

        # Create a temporary file
        temp_file = f"_temp_model_{beta}_{theta}.py"
        with open(temp_file, "w") as f:
            f.write(temp_model)

        # Run the temporary model
        try:
            result = subprocess.run(
                ["python", temp_file],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                # Extract key metrics from output
                output = result.stdout
                lines = output.split("\n")

                total_elec_cost = None
                total_gas_cost = None
                total_cost = None
                avg_temp = None
                min_temp = None
                max_temp = None

                for line in lines:
                    if "Total Electricity Cost: £" in line:
                        try:
                            total_elec_cost = float(line.split("£")[1].strip())
                        except:
                            pass
                    elif "Total Gas Cost: £" in line:
                        try:
                            total_gas_cost = float(line.split("£")[1].strip())
                        except:
                            pass
                    elif "Total Cost: £" in line and "Electricity" not in line and "Gas" not in line:
                        try:
                            total_cost = float(line.split("£")[1].strip())
                        except:
                            pass
                    elif "Average Indoor Temperature:" in line:
                        try:
                            val = line.split(":")[1].replace(" C", "").strip()
                            avg_temp = float(val)
                        except:
                            pass
                    elif "Minimum Indoor Temperature:" in line:
                        try:
                            val = line.split(":")[1].replace(" C", "").strip()
                            min_temp = float(val)
                        except:
                            pass
                    elif "Maximum Indoor Temperature:" in line:
                        try:
                            val = line.split(":")[1].replace(" C", "").strip()
                            max_temp = float(val)
                        except:
                            pass

                # Calculate comfort metrics
                setpoint = 20.0  # Default setpoint
                avg_deviation = abs(avg_temp - setpoint) if avg_temp else None
                temp_range = (max_temp - min_temp) if (max_temp and min_temp) else None

                results.append(
                    {
                        "beta": beta,
                        "theta": theta,
                        "total_cost": total_cost,
                        "electricity_cost": total_elec_cost,
                        "gas_cost": total_gas_cost,
                        "avg_temp": avg_temp,
                        "min_temp": min_temp,
                        "max_temp": max_temp,
                        "avg_deviation": avg_deviation,
                        "temp_range": temp_range,
                        "status": "success",
                    }
                )
                print("✓ Success")
            else:
                print("✗ Failed (solver issue)")
                results.append(
                    {
                        "beta": beta,
                        "theta": theta,
                        "status": "failed",
                        "error": result.stderr[:100] if result.stderr else "Unknown error",
                    }
                )
        except subprocess.TimeoutExpired:
            print("✗ Timeout")
            results.append(
                {
                    "beta": beta,
                    "theta": theta,
                    "status": "timeout",
                }
            )
        except Exception as e:
            print(f"✗ Exception: {str(e)[:50]}")
            results.append(
                {
                    "beta": beta,
                    "theta": theta,
                    "status": "exception",
                    "error": str(e)[:100],
                }
            )
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

# Convert results to DataFrame
df_results = pd.DataFrame(results)
successful_results = df_results[df_results["status"] == "success"].copy()

print("\n" + "=" * 80)
print("SWEEP COMPLETE")
print("=" * 80)
print(f"\nSuccessful runs: {len(successful_results)} / {len(results)}")

if len(successful_results) > 0:
    # Save results to CSV
    successful_results.to_csv("Results/deadband_penalty_sweep_results.csv", index=False)
    print("Results saved to: Results/deadband_penalty_sweep_results.csv")

    # Print summary statistics
    print("\n" + "-" * 80)
    print("SUMMARY STATISTICS")
    print("-" * 80)

    print("\nCost Range:")
    print(f"  Minimum: £{successful_results['total_cost'].min():.2f}")
    print(f"  Maximum: £{successful_results['total_cost'].max():.2f}")
    print(f"  Range: £{successful_results['total_cost'].max() - successful_results['total_cost'].min():.2f}")

    print("\nTemperature Deviation Range:")
    print(f"  Best (closest to setpoint): {successful_results['avg_deviation'].min():.2f}°C")
    print(f"  Worst: {successful_results['avg_deviation'].max():.2f}°C")

    print("\nTemperature Spread (max - min):")
    print(f"  Minimum: {successful_results['temp_range'].min():.2f}°C")
    print(f"  Maximum: {successful_results['temp_range'].max():.2f}°C")

    # Find Pareto optimal points
    print("\n" + "-" * 80)
    print("PARETO OPTIMAL POINTS (Best Cost-Comfort Trade-offs)")
    print("-" * 80)

    # Normalize metrics for Pareto analysis
    successful_results["cost_norm"] = (successful_results["total_cost"] - successful_results["total_cost"].min()) / (
        successful_results["total_cost"].max() - successful_results["total_cost"].min()
    )
    successful_results["deviation_norm"] = (
        successful_results["avg_deviation"] - successful_results["avg_deviation"].min()
    ) / (successful_results["avg_deviation"].max() - successful_results["avg_deviation"].min())

    # Find Pareto frontier (points not dominated on both objectives)
    pareto_points = []
    for idx, row in successful_results.iterrows():
        is_dominated = False
        for idx2, row2 in successful_results.iterrows():
            if idx == idx2:
                continue
            # row2 dominates row if it's better on both cost AND deviation
            if (row2["cost_norm"] < row["cost_norm"]) and (row2["deviation_norm"] < row["deviation_norm"]):
                is_dominated = True
                break
        if not is_dominated:
            pareto_points.append(idx)

    pareto_df = successful_results.iloc[pareto_points].sort_values("total_cost")
    print("\nPareto Points (sorted by cost):\n")
    for idx, row in pareto_df.iterrows():
        print(
            f"  beta={int(row['beta']):4.0f}, theta={int(row['theta']):3.0f} | "
            f"Cost: £{row['total_cost']:7.2f} | "
            f"Avg Deviation: {row['avg_deviation']:5.2f}°C | "
            f"Temp Range: {row['temp_range']:5.2f}°C"
        )

    # Create visualization plots
    print("\n" + "-" * 80)
    print("GENERATING VISUALIZATIONS")
    print("-" * 80)

    # Plot 1: 3D Surface - Cost vs Beta vs Theta
    pivot_cost = successful_results.pivot_table(values="total_cost", index="beta", columns="theta", aggfunc="first")

    fig1 = go.Figure(
        data=[
            go.Surface(
                x=pivot_cost.columns,
                y=pivot_cost.index,
                z=pivot_cost.values,
                colorscale="Viridis",
                name="Total Cost",
            )
        ]
    )
    fig1.update_layout(
        title="Total Energy Cost vs Beta and Theta",
        scene=dict(
            xaxis_title="Theta (Upper Bound Penalty)",
            yaxis_title="Beta (Lower Bound Penalty)",
            zaxis_title="Total Cost (£)",
        ),
        width=1000,
        height=700,
    )
    fig1.write_html("Results/deadband_sweep_cost_surface.html")
    print("✓ Saved: Results/deadband_sweep_cost_surface.html")

    # Plot 2: Pareto Frontier
    fig2 = go.Figure()

    # Add all points
    fig2.add_trace(
        go.Scatter(
            x=successful_results["avg_deviation"],
            y=successful_results["total_cost"],
            mode="markers",
            name="All Combinations",
            marker=dict(size=8, color="lightblue", line=dict(color="blue", width=1)),
            text=[f"β={int(r['beta'])}, θ={int(r['theta'])}" for _, r in successful_results.iterrows()],
            hovertemplate="<b>%{text}</b><br>Deviation: %{x:.2f}°C<br>Cost: £%{y:.2f}<extra></extra>",
        )
    )

    # Highlight Pareto points
    fig2.add_trace(
        go.Scatter(
            x=pareto_df["avg_deviation"],
            y=pareto_df["total_cost"],
            mode="markers+lines",
            name="Pareto Optimal",
            marker=dict(size=12, color="red", symbol="star", line=dict(color="darkred", width=2)),
            line=dict(color="red", width=2),
            text=[f"β={int(r['beta'])}, θ={int(r['theta'])}" for _, r in pareto_df.iterrows()],
            hovertemplate="<b>%{text}</b><br>Deviation: %{x:.2f}°C<br>Cost: £%{y:.2f}<extra></extra>",
        )
    )

    fig2.update_layout(
        title="Pareto Frontier: Cost vs Thermal Comfort",
        xaxis_title="Average Temperature Deviation from Setpoint (°C)",
        yaxis_title="Total Energy Cost (£)",
        hovermode="closest",
        height=700,
        width=1000,
        template="plotly_white",
    )
    fig2.write_html("Results/deadband_sweep_pareto_frontier.html")
    print("✓ Saved: Results/deadband_sweep_pareto_frontier.html")

    # Plot 3: Heatmap - Cost
    fig3 = go.Figure(
        data=[
            go.Heatmap(
                x=pivot_cost.columns,
                y=pivot_cost.index,
                z=pivot_cost.values,
                colorscale="Viridis",
                colorbar=dict(title="Cost (£)"),
            )
        ]
    )
    fig3.update_layout(
        title="Total Cost Heatmap",
        xaxis_title="Theta (Upper Bound Penalty)",
        yaxis_title="Beta (Lower Bound Penalty)",
        width=900,
        height=700,
    )
    fig3.write_html("Results/deadband_sweep_cost_heatmap.html")
    print("✓ Saved: Results/deadband_sweep_cost_heatmap.html")

    # Plot 4: Heatmap - Average Deviation
    pivot_deviation = successful_results.pivot_table(
        values="avg_deviation", index="beta", columns="theta", aggfunc="first"
    )

    fig4 = go.Figure(
        data=[
            go.Heatmap(
                x=pivot_deviation.columns,
                y=pivot_deviation.index,
                z=pivot_deviation.values,
                colorscale="RdYlGn_r",
                colorbar=dict(title="Deviation (°C)"),
            )
        ]
    )
    fig4.update_layout(
        title="Average Temperature Deviation Heatmap",
        xaxis_title="Theta (Upper Bound Penalty)",
        yaxis_title="Beta (Lower Bound Penalty)",
        width=900,
        height=700,
    )
    fig4.write_html("Results/deadband_sweep_deviation_heatmap.html")
    print("✓ Saved: Results/deadband_sweep_deviation_heatmap.html")

    # Plot 5: Cost vs Beta (average across theta values)
    fig5 = go.Figure()
    avg_by_beta = successful_results.groupby("beta").agg(
        {
            "total_cost": ["mean", "min", "max"],
            "avg_deviation": ["mean", "min", "max"],
        }
    )

    fig5.add_trace(
        go.Scatter(
            x=avg_by_beta.index,
            y=avg_by_beta[("total_cost", "mean")],
            mode="lines+markers",
            name="Average Cost",
            line=dict(color="blue", width=2),
            marker=dict(size=8),
            fill="tozeroy",
        )
    )

    fig5.update_layout(
        title="Impact of Beta on Total Cost",
        xaxis_title="Beta (Lower Bound Penalty)",
        yaxis_title="Average Total Cost (£)",
        height=600,
        width=1000,
        template="plotly_white",
    )
    fig5.write_html("Results/deadband_sweep_beta_sensitivity.html")
    print("✓ Saved: Results/deadband_sweep_beta_sensitivity.html")

    # Plot 6: Cost vs Theta (average across beta values)
    fig6 = go.Figure()
    avg_by_theta = successful_results.groupby("theta").agg(
        {
            "total_cost": ["mean", "min", "max"],
            "avg_deviation": ["mean", "min", "max"],
        }
    )

    fig6.add_trace(
        go.Scatter(
            x=avg_by_theta.index,
            y=avg_by_theta[("total_cost", "mean")],
            mode="lines+markers",
            name="Average Cost",
            line=dict(color="green", width=2),
            marker=dict(size=8),
            fill="tozeroy",
        )
    )

    fig6.update_layout(
        title="Impact of Theta on Total Cost",
        xaxis_title="Theta (Upper Bound Penalty)",
        yaxis_title="Average Total Cost (£)",
        height=600,
        width=1000,
        template="plotly_white",
    )
    fig6.write_html("Results/deadband_sweep_theta_sensitivity.html")
    print("✓ Saved: Results/deadband_sweep_theta_sensitivity.html")

    print("\n" + "-" * 80)
    print("RECOMMENDATIONS")
    print("-" * 80)

    # Find best on different criteria
    best_cost_idx = successful_results["total_cost"].idxmin()
    best_comfort_idx = successful_results["avg_deviation"].idxmin()
    best_pareto_idx = pareto_df.iloc[len(pareto_df) // 2].name  # Middle of Pareto

    print("\n1. LOWEST COST (highest energy efficiency):")
    row = successful_results.loc[best_cost_idx]
    print(f"   beta={int(row['beta'])}, theta={int(row['theta'])}")
    print(f"   Cost: £{row['total_cost']:.2f}")
    print(f"   Avg Temp Deviation: {row['avg_deviation']:.2f}°C")

    print("\n2. BEST THERMAL COMFORT (lowest temperature deviation):")
    row = successful_results.loc[best_comfort_idx]
    print(f"   beta={int(row['beta'])}, theta={int(row['theta'])}")
    print(f"   Cost: £{row['total_cost']:.2f}")
    print(f"   Avg Temp Deviation: {row['avg_deviation']:.2f}°C")

    print("\n3. BALANCED (Pareto optimal middle ground):")
    row = successful_results.loc[best_pareto_idx]
    print(f"   beta={int(row['beta'])}, theta={int(row['theta'])}")
    print(f"   Cost: £{row['total_cost']:.2f}")
    print(f"   Avg Temp Deviation: {row['avg_deviation']:.2f}°C")

    print("\n" + "=" * 80)
    print("Analysis complete! Check Results/ folder for visualizations.")
    print("=" * 80)

else:
    print("\n⚠ No successful runs. Check model for errors.")
