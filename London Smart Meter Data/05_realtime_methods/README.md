# 05 — Real-time execution under forecast error (N = 10)

The day-ahead bilateral ADMM plan is executed hour by hour, against load, PV and outdoor
temperature that differ from the forecast. This experiment compares five ways of doing
it, all on **identical** noise scenarios.

## Methods

| Method | Script | What happens each interval |
|---|---|---|
| `open_loop` | `simulate_open_loop.py` | The plan is applied exactly; all forecast error goes to grid import, or to curtailment when there is surplus |
| `fixed_schedule` | `simulate_noisy_realtime.py` (rules) | The plan is followed with rule-based recourse: battery and trades are clipped to what is physically possible, the grid absorbs the rest, and surplus tops up the battery before curtailment |
| `local_reoptimisation` | `simulate_realtime_reoptimise.py` | Each building re-optimises its own remaining horizon. There is no new sharing: planned trades are upper bounds, with a penalty of £0.1/kWh on undelivered planned trades. Each pair then trades min(offer, acceptance) |
| `rolling_admm` | `simulate_realtime_reoptimise.py` | Bilateral ADMM is re-run over the remaining horizon, warm-started from the previous interval's duals and trades (`--admm-start cold` to disable) |
| `rolling_central` | `simulate_realtime_central.py` | The same community problem is solved centrally (one Gurobi call) over the remaining horizon |
| `rolling_admm_parallel` | `admm_parallel.py` | `rolling_admm` with the subproblems solved by persistent gurobipy models in parallel worker processes |
| `local_reoptimisation_parallel` | `local_parallel.py` | `local_reoptimisation` solved in parallel in the same way |

The re-optimising methods use the noisy values as their input for the whole remaining
horizon. With `--future-inputs forecast`, they use measured values for the current
interval and the day-ahead forecast after it. Only first-interval decisions are applied,
and the realised state of charge and indoor temperature seed the next solve.

Every solve is timed. `34_three_method_solve_times.csv` records each building solve, each
ADMM iteration and each community step. It also gives the estimated parallel latency
(slowest building per step for local; per-iteration slowest subproblem, summed over
iterations, for ADMM) and whether the step fits in the interval.

## Noise scenarios

`simulate_noisy_realtime.py` draws the scenarios once into `shared_noise_scenarios.npz`,
and every method reads them:

- multiplicative per-building load (20%) and PV (30%) noise, clipped at zero;
- additive outdoor-temperature noise (1 °C), common to the community;
- optional bias and cross-building correlation.

The sensitivity cases (`compare_realtime_methods.py --case`) are described, with their
results, in
[`Results/05_realtime_methods/cases/README.md`](../Results/05_realtime_methods/cases/README.md):

| Case | Settings |
|---|---|
| `base` | Independent, unbiased noise |
| `biased_evening` | Load 15% above forecast 16:00–22:00 |
| `correlated` | 80% of the load and PV error shared by all buildings |
| `biased_peaks_sunny_correlated` | Load +15% 06:00–09:00 and 16:00–22:00, PV +30%, correlation 0.8 |

## Day-ahead plan cache

`bilateral_admm_schedule.npz` holds the solved day-ahead plan. It is reused while its
signature matches the hash of `common/central_optimisation_showcase.py`,
`common/admm_bilateral_p2p_valley.py`, the input profiles and the solver settings.
`--refresh-cache` forces a new solve.

## Scripts

| Script | Purpose |
|---|---|
| `compare_realtime_methods.py` | Runs all methods on the shared scenarios and writes results, timings and summaries (`33`–`36`). Options include `--runs`, `--case`, `--future-inputs`, `--admm-start`, `--add-central`, `--refresh-dispatch-only` and `--resummarise` |
| `plot_realtime_comparison.py` | Figures 37–40 and 46–51 (`--run N`, `--case NAME`, `--overview`) |
| `simulate_noisy_realtime.py` | Plan cache, noise scenarios, and the fixed-schedule rules (`28_noisy_realtime_simulation.csv` when run alone) |
| `simulate_open_loop.py`, `simulate_realtime_reoptimise.py`, `simulate_realtime_central.py` | The individual methods. Each runs stand-alone too |
| `admm_parallel.py`, `local_parallel.py` | Parallel gurobipy implementations, with `--workers` (default min(8, CPUs − 2)). `admm_parallel` also solves the day-ahead plan in parallel. When the models would need more than 40% of RAM (N = 1000 on this laptop), it switches to a low-memory mode that rebuilds them every iteration |
| `simulate_realtime_admm_cold.py` | Warm- vs cold-start rolling ADMM (`41`–`45`, `--summary-only` after an interrupted run) |

## Running

```powershell
$py = "C:\conda_envs\myenv\python.exe"
& $py 05_realtime_methods\compare_realtime_methods.py --runs 5
& $py 05_realtime_methods\compare_realtime_methods.py --case biased_evening --runs 5
& $py 05_realtime_methods\plot_realtime_comparison.py --run 1
& $py 05_realtime_methods\plot_realtime_comparison.py --overview
& $py 05_realtime_methods\admm_parallel.py --runs 1 --workers 8
```

Outputs go to `Results/05_realtime_methods/` (base case) and
`Results/05_realtime_methods/cases/<case>/`.

## Findings

- Rolling central reaches the same cost as rolling ADMM in every case. The decentralised
  method loses nothing in optimality.
- Parallel ADMM reproduces the sequential iterations and costs. At N = 10 the day-ahead
  plan takes 247 iterations: 12.9 s parallel against 246 s sequential. Parallel local
  re-optimisation gives the same community totals as sequential: 1.3 s against 36.6 s
  per run. The scaling of these times is in
  [`06_realtime_scaling`](../06_realtime_scaling/README.md).
