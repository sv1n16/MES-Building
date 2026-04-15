"""
Comparison Script: Deadband vs Quadratic Penalty Optimization Models

This script runs both optimization models with identical initial conditions
and compares their results across multiple metrics.
"""

import subprocess
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def format_value(val):
    """Format value as float if numeric, otherwise return as string"""
    if isinstance(val, str):
        return val
    try:
        return f"{float(val):.2f}"
    except (ValueError, TypeError):
        return "N/A"


print("=" * 80)
print("RUNNING OPTIMIZATION MODELS FOR COMPARISON")
print("=" * 80)

# Run deadband model
print("\n[1/2] Running DEADBAND model...")
result_deadband = subprocess.run(
    ["python", "central_optimisation_multi_building_electric_hp_boiler_ideal_thermal_deadband.py"],
    capture_output=True,
    text=True,
)

if result_deadband.returncode != 0:
    print("ERROR: Deadband model failed!")
    print(result_deadband.stderr)
    exit(1)
print("✓ Deadband model completed successfully")

# Run quadratic penalty model
print("\n[2/2] Running QUADRATIC PENALTY model...")
result_quadratic = subprocess.run(
    ["python", "central_optimisation_multi_building_electric_hp_boiler_ideal_occupancy.py"],
    capture_output=True,
    text=True,
)

if result_quadratic.returncode != 0:
    print("ERROR: Quadratic penalty model failed!")
    print(result_quadratic.stderr)
    exit(1)
print("✓ Quadratic penalty model completed successfully")

print("\n" + "=" * 80)
print("EXTRACTING RESULTS FOR COMPARISON")
print("=" * 80)

# Load saved results from deadband model
# Note: Both models save to the same file, so we need to run them sequentially
# and capture results during execution


# Parse console output for summary statistics
def extract_summary_stats(output_text):
    """Extract summary statistics from model output"""
    stats = {}
    lines = output_text.split("\n")

    for i, line in enumerate(lines):
        if "Total Electricity Cost: £" in line:
            val = line.split("£")[1].strip()
            stats["electricity_cost"] = float(val)
        elif "Total Gas Cost: £" in line:
            val = line.split("£")[1].strip()
            stats["gas_cost"] = float(val)
        elif "Total Cost: £" in line:
            val = line.split("£")[1].strip()
            stats["total_cost"] = float(val)
        elif "Average Indoor Temperature:" in line:
            val = line.split(":")[1].replace(" C", "").strip()
            stats["avg_temp"] = float(val)
        elif "Minimum Indoor Temperature:" in line:
            val = line.split(":")[1].replace(" C", "").strip()
            stats["min_temp"] = float(val)
        elif "Maximum Indoor Temperature:" in line:
            val = line.split(":")[1].replace(" C", "").strip()
            stats["max_temp"] = float(val)

    return stats


print("\nExtracting statistics from deadband model...")
deadband_stats = extract_summary_stats(result_deadband.stdout)
print(f"  Deadband extracted metrics: {len(deadband_stats)} fields found")

print("Extracting statistics from quadratic penalty model...")
quadratic_stats = extract_summary_stats(result_quadratic.stdout)
print(f"  Quadratic penalty extracted metrics: {len(quadratic_stats)} fields found")

print("\n" + "=" * 80)
print("COMPARISON RESULTS")
print("=" * 80)

# Create comparison table
print("\n" + "-" * 80)
print("ENERGY COST COMPARISON (Building 0)")
print("-" * 80)

comparison_data = {
    "Metric": [
        "Total Electricity Cost (£)",
        "Total Gas Cost (£)",
        "Total Energy Cost (£)",
    ],
    "Deadband Model": [
        format_value(deadband_stats.get("electricity_cost", "N/A")),
        format_value(deadband_stats.get("gas_cost", "N/A")),
        format_value(deadband_stats.get("total_cost", "N/A")),
    ],
    "Quadratic Penalty Model": [
        format_value(quadratic_stats.get("electricity_cost", "N/A")),
        format_value(quadratic_stats.get("gas_cost", "N/A")),
        format_value(quadratic_stats.get("total_cost", "N/A")),
    ],
}

# Calculate differences
if all(k in deadband_stats and k in quadratic_stats for k in ["total_cost"]):
    cost_diff = quadratic_stats["total_cost"] - deadband_stats["total_cost"]
    cost_pct = (cost_diff / deadband_stats["total_cost"] * 100) if deadband_stats["total_cost"] != 0 else 0
    comparison_data["Difference (£)"] = [
        format_value(quadratic_stats.get("electricity_cost", 0) - deadband_stats.get("electricity_cost", 0)),
        format_value(quadratic_stats.get("gas_cost", 0) - deadband_stats.get("gas_cost", 0)),
        f"{cost_diff:.2f} ({cost_pct:+.1f}%)" if isinstance(cost_diff, (int, float)) else "N/A",
    ]

# Print table
for metric in comparison_data["Metric"]:
    idx = comparison_data["Metric"].index(metric)
    print(f"\n{metric}")
    print(f"  Deadband:              {comparison_data['Deadband Model'][idx]}")
    print(f"  Quadratic Penalty:     {comparison_data['Quadratic Penalty Model'][idx]}")
    if "Difference (£)" in comparison_data:
        print(f"  Difference:            {comparison_data['Difference (£)'][idx]}")

print("\n" + "-" * 80)
print("TEMPERATURE STATISTICS COMPARISON (Building 0)")
print("-" * 80)


temp_data = {
    "Metric": [
        "Average Temperature (C)",
        "Minimum Temperature (C)",
        "Maximum Temperature (C)",
    ],
    "Deadband Model": [
        format_value(deadband_stats.get("avg_temp", "N/A")),
        format_value(deadband_stats.get("min_temp", "N/A")),
        format_value(deadband_stats.get("max_temp", "N/A")),
    ],
    "Quadratic Penalty Model": [
        format_value(quadratic_stats.get("avg_temp", "N/A")),
        format_value(quadratic_stats.get("min_temp", "N/A")),
        format_value(quadratic_stats.get("max_temp", "N/A")),
    ],
}

for metric in temp_data["Metric"]:
    idx = temp_data["Metric"].index(metric)
    print(f"\n{metric}")
    print(f"  Deadband:              {temp_data['Deadband Model'][idx]}")
    print(f"  Quadratic Penalty:     {temp_data['Quadratic Penalty Model'][idx]}")

print("\n" + "=" * 80)
print("CONSOLE OUTPUT COMPARISON")
print("=" * 80)

print("\n--- DEADBAND MODEL SUMMARY ---")
print(result_deadband.stdout[-1000:] if len(result_deadband.stdout) > 1000 else result_deadband.stdout)

print("\n--- QUADRATIC PENALTY MODEL SUMMARY ---")
print(result_quadratic.stdout[-1000:] if len(result_quadratic.stdout) > 1000 else result_quadratic.stdout)

# Save comparison results to file
# Format values safely
elec_db = format_value(deadband_stats.get("electricity_cost"))
gas_db = format_value(deadband_stats.get("gas_cost"))
total_db = format_value(deadband_stats.get("total_cost"))
elec_qp = format_value(quadratic_stats.get("electricity_cost"))
gas_qp = format_value(quadratic_stats.get("gas_cost"))
total_qp = format_value(quadratic_stats.get("total_cost"))

avg_temp_db = format_value(deadband_stats.get("avg_temp"))
min_temp_db = format_value(deadband_stats.get("min_temp"))
max_temp_db = format_value(deadband_stats.get("max_temp"))
avg_temp_qp = format_value(quadratic_stats.get("avg_temp"))
min_temp_qp = format_value(quadratic_stats.get("min_temp"))
max_temp_qp = format_value(quadratic_stats.get("max_temp"))

# Calculate safe differences
try:
    elec_diff = deadband_stats.get("electricity_cost", 0) - quadratic_stats.get("electricity_cost", 0)
    elec_diff_str = f"{elec_diff:+.2f}"
except:
    elec_diff_str = "N/A"

try:
    gas_diff = deadband_stats.get("gas_cost", 0) - quadratic_stats.get("gas_cost", 0)
    gas_diff_str = f"{gas_diff:+.2f}"
except:
    gas_diff_str = "N/A"

try:
    total_diff = deadband_stats.get("total_cost", 0) - quadratic_stats.get("total_cost", 0)
    total_diff_str = f"{total_diff:+.2f}"
except:
    total_diff_str = "N/A"

comparison_report = f"""
OPTIMIZATION MODEL COMPARISON REPORT
{'=' * 80}
Generated: 2026-03-25

MODELS COMPARED:
1. Deadband Model: central_optimisation_multi_building_electric_hp_boiler_ideal_thermal_deadband.py
2. Quadratic Penalty Model: central_optimisation_multi_building_electric_hp_boiler_ideal_occupancy.py

CONFIGURATION:
- Alpha (Temperature Penalty Coefficient): 0.5
- Random Seed: 42 (identical initial conditions)
- Initial Temperature Range: 18-24C
- Occupancy Profile: 1 during 0-8h and 16-23h, 0 during 9-15h

ENERGY COST RESULTS (Building 0):
{'-' * 80}
Deadband Model:
  Electricity Cost: {elec_db} £
  Gas Cost: {gas_db} £
  Total Cost: {total_db} £

Quadratic Penalty Model:
  Electricity Cost: {elec_qp} £
  Gas Cost: {gas_qp} £
  Total Cost: {total_qp} £

Difference:
  Electricity: {elec_diff_str} £
  Gas: {gas_diff_str} £
  Total: {total_diff_str} £

TEMPERATURE STATISTICS (Building 0):
{'-' * 80}
Deadband Model:
  Average Temperature: {avg_temp_db} C
  Minimum Temperature: {min_temp_db} C
  Maximum Temperature: {max_temp_db} C

Quadratic Penalty Model:
  Average Temperature: {avg_temp_qp} C
  Minimum Temperature: {min_temp_qp} C
  Maximum Temperature: {max_temp_qp} C

KEY DIFFERENCES:
{'-' * 80}
The deadband model uses hard constraints with slack variables and heavy penalties
for deadband violations during occupancy. The quadratic penalty model directly
penalizes deviations from setpoint, weighted by occupancy.

PENALTY FORMULATIONS:
- Deadband: occupancy * [beta*max(0, T_lower - T)^2 + theta*max(0, T - T_upper)^2]
           where beta=300, theta=50
           
- Quadratic: occupancy · α · (T_in - T_set)²
            where α=0.5

INITIAL CONDITIONS:
Both models initialized with identical random seed (42) for reproducibility.
"""

with open("Results/schedules/model_comparison_report.txt", "w") as f:
    f.write(comparison_report)

print("\n" + "=" * 80)
print("✓ Comparison report saved to: Results/schedules/model_comparison_report.txt")
print("=" * 80)
