# 03 — Day-ahead sensitivity and fairness (N = 10)

Sensitivity of the day-ahead solution to its settlement and algorithm parameters, and how
the community saving is shared between buildings. Uses the model and showcase community
of [`02_day_ahead_sharing_methods`](../02_day_ahead_sharing_methods/README.md).

## Scripts

| Script | Question | Outputs |
|---|---|---|
| `../common/central_optimisation_showcase.py --sweep` | How do the three fee definitions (loss, forfeit, market) and the fee level change cost and sharing? | `09_trade_fee_sweep.{png,html,csv}` |
| `price_fraction_sweep.py` | Re-settles the same dispatch at P2P prices of 0.1–1.0 × the tariff. Does every building still benefit? | `21_price_fraction_sweep{,_summary}.csv`, `.png` |
| `admm_rho_sweep.py` | Iterations to converge and cost accuracy against the ADMM penalty ρ, for the exchange and bilateral ADMM, with exact or relaxed binaries | `20_admm_rho_sweep.{csv,png}` |
| `admm_bilateral_tolerance_sweep.py` | Accuracy against central, and run time, of bilateral ADMM as the stopping tolerance changes | `31_admm_tolerance_sweep_summary.csv`, `32_admm_tolerance_sweep_per_building.csv` |
| `cost_shapley.py` | Allocates the community saving to buildings by Shapley value (permutation sampling, or `--exact`) | `34_cost_shapley_values.{csv,png}`, coalition cache `34_shapley_coalition_costs.json` |
| `admm_tolerance_shapley.py` | Shapley allocation with bilateral ADMM at several tolerances. Does the allocation depend on ADMM accuracy? | `35_admm_tolerance_shapley_values.csv` (not run yet) |

Results go to `Results/03_day_ahead_sensitivity/`. The ρ and price sweeps use
`02_day_ahead_sharing_methods/admm_bilateral_p2p.py`. The tolerance sweep uses
`common/admm_bilateral_p2p_valley.py`.

## Running

```powershell
C:\conda_envs\myenv\python.exe common\central_optimisation_showcase.py --sweep
C:\conda_envs\myenv\python.exe 03_day_ahead_sensitivity\price_fraction_sweep.py --step 0.1
C:\conda_envs\myenv\python.exe 03_day_ahead_sensitivity\admm_rho_sweep.py --method both
C:\conda_envs\myenv\python.exe 03_day_ahead_sensitivity\admm_bilateral_tolerance_sweep.py
C:\conda_envs\myenv\python.exe 03_day_ahead_sensitivity\cost_shapley.py --permutations 300
```

## Findings (earlier runs)

- **Tolerance sets accuracy and run time** (`31_*`). At 1e-3, bilateral ADMM converges
  in 19 iterations (9 s) and lands within £0.05/day of central. At 1e-4 it takes
  24 iterations (11 s) and lands within £0.01/day. At 1e-5 it did not reach the
  tolerance within 1000 iterations (605 s), although its cost is then essentially exact.
  The studies now use 1e-4.
- **The saving is concentrated** (`34_*`, 300 permutations). Three buildings receive 84%
  of the community saving: MAC000635 30% and MAC004245 28% (both battery + PV), and
  MAC002861 26% (battery only). The other seven get 1–3% each.
