# Noisy Real-Time Simulation

`simulate_noisy_realtime.py` runs the bilateral ADMM optimizer once, then tests its cached day-ahead schedule against independently perturbed load and PV profiles. It runs 50 simulations by default. The optimizer schedule is cached in `Results/ADMM Optimisation Real-time/bilateral_admm_schedule.npz`; a matching cache is reused on later runs.

## Dispatch and Deviation Rules

The day-ahead schedule is the reference; each interval only absorbs the difference between forecast and actual values.

1. **Apply forecast error:** step through the shared noisy scenario (`shared_noise_scenarios.npz`) one interval at a time. Load and PV noise are multiplicative per building, clipped at zero (defaults: 20% load, 30% PV standard deviation). Outdoor-temperature noise is additive (default standard deviation 1 °C, `--temp-noise`) and the same for the whole community.
2. **Follow the battery plan:** charge and discharge at the planned setpoints, including grid charging where the plan uses it, clipped only by power and the actual state of charge. Battery use stays disabled in the first interval, matching the optimiser.
3. **Honour planned P2P trades:** each bilateral trade is delivered up to the planned amount, limited by the same gross limits the optimiser uses. A seller exports at most its actual PV plus battery discharge, and a buyer imports at most its actual load plus heat pump plus battery charging.
4. **Grid absorbs the error:** any remaining shortfall is imported from the grid. Any remaining surplus first tops up the battery if it has spare power and capacity, and only then is curtailed.
5. **Track schedule deviation:** record planned, final, and final-minus-planned values for each per-building decision variable, planned and realised bilateral trades (`q`, `qpos`, `qneg`), battery corrections, and the gap between actual and planned state of charge.

## Scope and Assumptions

- Heat-pump electrical power and boiler output are held at their planned values. Heat-pump heat output follows the actual COP, and indoor temperature follows the optimiser's thermal model driven by the actual outdoor temperature.
- No extra battery discharge is used to cover a shortfall, so the energy the plan stores for later high-price intervals is preserved; the grid covers the error instead.
- The simulation uses the optimizer's interval duration and energy-efficiency parameters. Power values are in kW and state of charge is in kWh.
- `--pace-seconds` adds a delay after each interval. Its default is zero, so batch simulations run without real-time waiting.

## Run

From this folder:

```powershell
python .\simulate_noisy_realtime.py
```

Options:

```powershell
python .\simulate_noisy_realtime.py --runs 50 --load-noise 0.20 --pv-noise 0.30
python .\simulate_noisy_realtime.py --refresh-cache
python .\simulate_noisy_realtime.py --pace-seconds 3600
```

`--refresh-cache` forces a new ADMM solve and replaces the stored schedule. The cache is validated against optimizer source, input profiles, and relevant optimizer settings; a stale or invalid cache is recomputed automatically.

## Outputs

The simulation writes `Results/ADMM Optimisation Real-time/28_noisy_realtime_simulation.csv`. Each row represents one building and interval and includes the simulation run number, noisy inputs, planned/final schedules, deviations, battery operation, realized P2P flows, actual and planned grid imports, and import error. The console summary reports the mean and standard deviation of total grid-import error across runs.

# Real-Time Method Comparison

`compare_realtime_methods.py` runs three real-time strategies on **identical** noisy load/PV scenarios, stored in `Results/ADMM Optimisation Real-time/shared_noise_scenarios.npz` and reused by `simulate_noisy_realtime.py` and `simulate_realtime_reoptimise.py` (use `--refresh-scenarios` to redraw them):

0. **open_loop:** the day-ahead plan applied exactly as scheduled, with no real-time feedback (`simulate_open_loop.py`). Battery, heat pump, boiler and P2P trades follow the plan unchanged; indoor temperature drifts with the actual outdoor temperature; all forecast error goes to grid import, or curtailment for surplus. `python .\compare_realtime_methods.py --refresh-dispatch-only` re-simulates open loop and fixed schedule on the saved scenarios without re-solving the optimisers.
1. **fixed_schedule:** the day-ahead ADMM plan with the rule-based recourse described above.
2. **local_reoptimisation:** every interval, each building re-optimises its own remaining horizon with no new sharing. Each planned trade is an upper bound that the building may honour or reduce, but never exceed. The objective adds `cost_penalty × |cost − planned cost| / |planned cost|` and `share_penalty × (planned − delivered P2P kWh)`. Each pair then trades `min(seller offer, buyer acceptance)`: the buyer imports any shortfall from the grid, and the seller curtails any unsold surplus.
3. **rolling_admm:** bilateral ADMM is re-run over the remaining horizon every interval, warm-started by default from the previous interval's duals and trades (`--admm-start cold` to disable).

The re-optimising methods use the noisy load, PV and outdoor temperature as input data for the whole remaining horizon. `--future-inputs forecast` instead uses measured values for the current interval and the day-ahead forecast after that. Only first-interval decisions are applied; the realised SOC and indoor temperature seed the next solve. Every solve is timed. `34_three_method_solve_times.csv` records the time of each building solve, each ADMM iteration and each community step. It also gives an estimated parallel latency (local: slowest building; ADMM: sum over iterations of the slowest subproblem) and flags whether each step finishes within the interval length.

```powershell
python .\compare_realtime_methods.py --runs 5
python .\plot_realtime_comparison.py --run 1
```

## Warm vs Cold-Start Rolling ADMM

`simulate_realtime_admm_cold.py` re-runs rolling ADMM on the same scenarios, but restarts from zero duals and zero trades at every interval (cold start) instead of reusing the previous interval's solution (warm start). Scenarios run as parallel processes with one Gurobi thread per solve. If the cold-start runs are stopped early, `--summary-only` compares the completed intervals. Iterations inflated by machine sleep are removed. Full-day time is projected from cold-start iteration counts × the warm run's time per iteration at the same horizon (`43_admm_warm_vs_cold_partial.csv`, `45_admm_warm_vs_cold.png`).

```powershell
python .\simulate_realtime_admm_cold.py --runs 5 --jobs 5
python .\simulate_realtime_admm_cold.py --summary-only
```
