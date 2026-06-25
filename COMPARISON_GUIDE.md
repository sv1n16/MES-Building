# Deadband vs Quadratic Penalty Model Comparison

## Overview

This document describes the comparison between two optimization approaches for multi-building energy management with occupancy-weighted thermal comfort penalties.

## Model Specifications

### File 1: Deadband Model
**File:** `central_optimisation_multi_building_electric_hp_boiler_ideal_thermal_deadband.py`

**Approach:** Hard temperature bounds with slack variables
- Uses temperature deadband bounds during occupied/unoccupied periods
- Slack variables capture violations of these bounds
- Asymmetric penalties: β=300 (lower bound), θ=50 (upper bound)

**Objective Function:**
```
minimize: Σ[electricity_cost + gas_cost + occupancy·(β·T_below_lower² + θ·T_above_upper²)]
```

**Temperature Bounds:**
- Occupied (0-8h, 16-23h): T_set ± 1°C
- Unoccupied (9-15h): 10-28°C

---

### File 2: Quadratic Penalty Model
**File:** `central_optimisation_multi_building_electric_hp_boiler_ideal_occupancy.py`

**Approach:** Direct temperature deviation penalty
- Direct penalty for deviation from temperature setpoint
- Weighted by occupancy profile (1 during occupied, 0 during unoccupied)
- No hard temperature bounds

**Objective Function:**
```
minimize: Σ[electricity_cost + gas_cost + occupancy·α·(T_in - T_set)²]
```

**Temperature Bounds:**
- No explicit bounds, flexibility to drift during unoccupied hours

---

## Identical Comparison Parameters

| Parameter | Value |
|-----------|-------|
| Alpha (temperature penalty coefficient) | 0.5 |
| Random seed | 42 |
| Initial temperature range | 18-24°C |
| Occupancy pattern | Binary: 1 during 0-8h & 16-23h, 0 during 9-15h |
| Time horizon | 24 hours |
| Number of buildings | 4 |
| Data file | `data/processed_data_2018_02_21.csv` |
| Battery capacity | 12 kWh |
| Heat pump max power | 10 kW |
| Boiler max power | 20 kW |
| Solver | Gurobi (via Pyomo) |

---

## Running the Comparison

### Step 1: Run the Deadband Model
```bash
python central_optimisation_multi_building_electric_hp_boiler_ideal_thermal_deadband.py
```

**Output:**
- Interactive visualization with 3 columns per building:
  1. Battery and heat pump operation
  2. Building temperatures with deadband bounds
  3. Electricity and gas costs
- Console summary (Building 0):
  - Total electricity cost (£)
  - Total gas cost (£)
  - Temperature statistics (avg, min, max)
- Solver log: `Results/Temp_setpoint/solver_log.txt`
- Saved schedules: `Results/schedules/central_optimisation_schedules_and_costs.json`

### Step 2: Run the Quadratic Penalty Model
```bash
python central_optimisation_multi_building_electric_hp_boiler_ideal_occupancy.py
```

**Output:**
- Same visualization structure as deadband model
- Console summary with identical metrics
- Solver log (same path)
- Saved schedules (same path, will overwrite)

### Step 3: Compare Results
```bash
python compare_models_manual.py
```

---

## Key Differences in Approach

### Deadband Model
**Strengths:**
- Hard constraints ensure no violations outside comfort zone
- Asymmetric penalties allow different importance for heating vs cooling
- Explicit control over deadband width
- Safety guarantee: temperature won't exceed bounds

**Weaknesses:**
- More complex model (additional slack variables + constraints)
- Solver may be slower due to additional decisions
- May over-penalize near-bound temperatures

---

### Quadratic Penalty Model  
**Strengths:**
- Simpler model formulation
- Continuous penalty function (smooth gradients for optimizer)
- Faster to solve
- Natural comfort gradient (larger deviation = larger penalty)

**Weaknesses:**
- No hard bounds, temperature could drift far when unoccupied
- Single penalty coefficient for all violations
- May allow excessive temperature swings

---

## Comparison Metrics

### 1. Cost Analysis
- **Electricity cost** (£): Impact of heating/cooling decisions
- **Gas cost** (£): Boiler usage efficiency
- **Total cost** (£): Overall optimization objective
- **Cost difference** (%): Relative performance

### 2. Thermal Performance
- **Average temperature** (°C): Comfort maintenance during day
- **Min/max temperature** (°C): Thermal extremes
- **Temperature std dev** (°C): Stability (lower is better)

### 3. Energy Consumption
- **Heat pump hours**: Seasonal heating approach
- **Boiler hours**: Gas heating usage
- **Battery cycles**: Storage utilization

### 4. Occupancy Compliance
- **Penalty during occupied**: Temperature control during presence
- **Temperature drop unoccupied**: Allowed drift without penalty

---

## Expected Outcomes

The two models should produce:

1. **Similar total costs** (same alpha=0.5 weighting)
2. **Different temperature profiles**:
   - Deadband: Tighter clustering around bounds
   - Quadratic: Smoother transitions
3. **Similar occupied-period temperatures** (both penalize deviations)
4. **Different unoccupied temperatures** (quadratic may drift more)
5. **Different solver times** (quadratic simpler, likely faster)

---

## Configuration Verification

✓ **Deadband file:**
- `alpha = 0.5`
- `np.random.seed(42)`
- `T_lower_unoccupied = 10.0`
- `T_upper_unoccupied = 28.0` (unoccupied OR 20.0 - check file)
- `beta = 300.0`
- `theta = 50.0`

✓ **Quadratic file:**
- `alpha = 0.5`
- `np.random.seed(42)`
- `T_lower_unoccupied = 10.0`
- `T_upper_unoccupied = 28.0`
- Same occupancy_profile dictionary
- Same initial temperature randomization

---

## Files Generated

After running both models:

```
Results/
├── Temp_setpoint/
│   └── solver_log.txt                    # Gurobi solver output
├── schedules/
│   ├── central_optimisation_schedules_and_costs.json  # Detailed results
│   ├── model_comparison_report.txt       # Auto-generated comparison
│   └── comparison_output.log             # Comparison script output
└── plots/                                 # Visualizations
```

---

## Data Files Referenced

- Input: `data/processed_data_2018_02_21.csv`
  - Electricity prices (£/kWh)
  - PV supply (kW)
  - Outdoor temperature (°C)
  - Temperature setpoint (°C)
  - Building electric loads (kW)

---

## Next Steps

1. **Run deadband model** → Review visualizations and costs
2. **Run quadratic model** → Compare visualizations and costs
3. **Analyze JSON outputs** → Detailed metric comparison
4. **Document findings** → Create comparison analysis report

The fixed random seed ensures both models start with identical building temperatures, making the comparison of optimization decisions fair and meaningful.
