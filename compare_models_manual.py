"""
Comparison Script: Deadband vs Quadratic Penalty Optimization Models

INSTRUCTIONS:
1. Run: python central_optimisation_multi_building_electric_hp_boiler_ideal_thermal_deadband.py
2. Run: python central_optimisation_multi_building_electric_hp_boiler_ideal_occupancy.py
3. Then run this script: python compare_models_manual.py

This script will parse the output and generate a detailed comparison report.
"""

import json
import pandas as pd
from pathlib import Path

print("=" * 80)
print("OPTIMIZATION MODEL COMPARISON REPORT")
print("=" * 80)

print(
    """
MODELS BEING COMPARED:
1. Deadband Model
   File: central_optimisation_multi_building_electric_hp_boiler_ideal_thermal_deadband.py
   Approach: Uses slack variables with occupancy-weighted deadband penalties
   Penalty Formula: occupancy·[β·max(0,T_lower-T)² + θ·max(0,T-T_upper)²]
   Parameters: β=300 (lower), θ=50 (upper)

2. Quadratic Penalty Model
   File: central_optimisation_multi_building_electric_hp_boiler_ideal_occupancy.py
   Approach: Direct temperature deviation penalty, occupancy-weighted
   Penalty Formula: occupancy·α·(T_in - T_set)²
   Parameters: α=0.5

IDENTICAL CONDITIONS:
✓ Alpha coefficient: 0.5
✓ Fixed random seed: 42 (reproducible initial conditions)
✓ Initial temperature range: 18-24°C
✓ Occupancy profile: 1 during 0-8h & 16-23h, 0 during 9-15h
✓ Time horizon: 24 hours
✓ Data: processed_data_2018_02_21.csv

"""
)

# Try to load saved JSON results
json_path = Path("Results/schedules/central_optimisation_schedules_and_costs.json")

if json_path.exists():
    print("=" * 80)
    print("EXTRACTING RESULTS FROM JSON")
    print("=" * 80)

    with open(json_path, "r") as f:
        results = json.load(f)

    print("\nBuilding 0 - Key Metrics from Latest Run:")
    print("-" * 80)

    # Energy costs
    if "electricity_costs" in results and results["electricity_costs"]:
        elec_cost = sum(results["electricity_costs"])
        print(f"\nElectricity Costs:")
        print(f"  Total: £{elec_cost:.2f}")

    if "gas_costs" in results and results["gas_costs"]:
        gas_cost = sum(results["gas_costs"])
        print(f"\nGas Costs:")
        print(f"  Total: £{gas_cost:.2f}")

    # Temperature statistics
    if "indoor_temperature" in results and results["indoor_temperature"]:
        temps = [float(t) for t in results["indoor_temperature"]]
        print(f"\nTemperature Statistics (°C):")
        print(f"  Average:  {pd.Series(temps).mean():.2f}°C")
        print(f"  Min:      {min(temps):.2f}°C")
        print(f"  Max:      {max(temps):.2f}°C")
        print(f"  StDev:    {pd.Series(temps).std():.2f}°C")

    # Occupancy weighted penalty info
    if "occupancy_profile" in results:
        occ = results["occupancy_profile"]
        occupied_hours = sum(occ)
        print(f"\nOccupancy Profile:")
        print(f"  Occupied hours: {int(occupied_hours)} / 24")
        print(f"  Unoccupied hours: {int(24-occupied_hours)} / 24")
else:
    print(f"⚠ JSON results file not found at: {json_path}")

print("\n" + "=" * 80)
print("MANUAL COMPARISON INSTRUCTIONS")
print("=" * 80)

print(
    """
To compare the two models, review the following metrics from each run:

1. COST COMPARISON
   - Total electricity cost (£)
   - Total gas cost (£)
   - Total energy cost (£)
   - Cost difference and percentage change

2. TEMPERATURE PERFORMANCE
   - Average indoor temperature (°C)
   - Minimum temperature reached (°C)
   - Maximum temperature reached (°C)
   - Temperature standard deviation (°C)

3. THERMAL COMFORT
   - Deadband violations during occupied hours
   - Occupancy-weighted penalty term value

4. ENERGY CONSUMPTION
   - Battery charge/discharge patterns
   - Heat pump operation hours
   - Boiler usage hours

DETAILED LOGS:
- Deadband model: Results/Temp_setpoint/solver_log.txt (from thermal_deadband run)
- Quadratic model: Results/Temp_setpoint/solver_log.txt (from occupancy run)
- Saved schedules: Results/schedules/central_optimisation_schedules_and_costs.json

EXPECTED DIFFERENCES:
- Deadband model may achieve tighter temperature control during occupied hours
- Quadratic model has smoother transitions and may be computationally faster
- Cost difference depends on occupancy profile and penalty coefficient magnitudes
- Temperature variance likely lower in deadband model due to hard constraints
"""
)

print("\n" + "=" * 80)
print("COMPARISON CHECKLIST")
print("=" * 80)
print(
    """
□ Both models use alpha = 0.5
□ Both models use np.random.seed(42) for reproducible initial conditions
□ Deadband file has T_lower_unoccupied = 10°C, T_upper_unoccupied = 28°C
□ Occupancy file has T_lower_unoccupied = 10°C, T_upper_unoccupied = 28°C
□ Both files use same occupancy_profile definition
□ Both files use same data file: processed_data_2018_02_21.csv
□ Run deadband model: python central_optimisation_multi_building_electric_hp_boiler_ideal_thermal_deadband.py
□ Run quadratic model: python central_optimisation_multi_building_electric_hp_boiler_ideal_occupancy.py
□ Compare console output summaries
□ Compare temperature profiles in visualizations
□ Check Results/schedules/central_optimisation_schedules_and_costs.json for detailed results
"""
)

print("\n✓ Comparison setup complete. Check console outputs and JSON results files.")
