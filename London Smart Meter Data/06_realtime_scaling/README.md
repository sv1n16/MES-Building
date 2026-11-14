# 06 — Real-time methods against community size

Runs the real-time comparison of [`05_realtime_methods`](../05_realtime_methods/README.md)
for N = 5, 10, 20, 50, 100 and 1000 buildings under a wall-time budget. When a step runs
out of budget or memory, it stops; the intervals already finished are kept, and the step
is flagged as not converged.

## How it runs

`scale_realtime_methods.py` handles each N as follows.

1. It builds `data/showcase_<day>_N<NNNN>` if missing. N = 10 uses the default dataset.
   Households without a complete day of readings are excluded.
2. It runs each step in its own subprocess, with `SHOWCASE_DATASET` and
   `SHOWCASE_PLOTS_SUBDIR=N<NNNN>`, so outputs go to `Results/06_realtime_scaling/N<NNNN>/`.
   The steps are:
   - the day-ahead plan (`schedule` sequential, or `schedule_parallel`);
   - `open_loop`, `fixed_schedule`, `local_reoptimisation`, `rolling_admm` and
     `rolling_central`;
   - with `--parallel`, also `local_reoptimisation_parallel` and `rolling_admm_parallel`.
3. It enforces the budgets:
   - 2 h for the day-ahead plan and 1 h per method (`--schedule-budget`, `--budget`);
   - the budget is checked before every interval and after every ADMM subproblem;
   - rolling central gets a Gurobi time limit;
   - the driver kills a step that overruns by `--grace` (600 s), or whose own memory
     exceeds 60% of RAM (or 4 GB while under 3% of RAM is free).
4. It records each step in `scaling_status.csv` with a status:

   | Status | Meaning |
   |---|---|
   | `completed` | Finished |
   | `budget_exceeded` | Did not converge or finish within the budget |
   | `killed_timeout` | Killed by the driver after the grace period |
   | `killed_memory` | Killed for memory use |
   | `stopped_not_converged` | Stopped manually before converging |
   | `skipped_no_plan` | No day-ahead plan, so nothing to execute |
   | `failed` | Error |

5. It summarises the completed work into `scaling_summary.csv`. The console output of
   each size goes to `N<NNNN>/run_log.txt`.

`--keep-results` re-runs only steps that are new, skipped or failed (add
`--retry killed_memory …` for others) and reuses a completed plan.

`plot_scaling.py` draws:

- `60_scaling_solve_time.png`: the day-ahead solve against the 1-hour line, and the
  real-time step time against the 1-hour and 1-minute lines;
- `61_scaling_admm_iterations.png`;
- `62_size_vs_building_savings.png`: per-building saving against open loop, with Spearman
  ρ and a permutation p-value;
- `scaling_building_savings.csv`.

## Running

```powershell
$py = "C:\conda_envs\myenv\python.exe"
& $py 06_realtime_scaling\scale_realtime_methods.py --parallel --workers 8
& $py 06_realtime_scaling\scale_realtime_methods.py --sizes 5 10 20 --budget 600
& $py 06_realtime_scaling\scale_realtime_methods.py --parallel --keep-results --retry killed_memory
& $py 06_realtime_scaling\plot_scaling.py
```

## Results (1 noise run per size)

**Day-ahead plan** (bilateral ADMM, ε = 1e-4):

| N | Sequential (Pyomo) | Parallel (gurobipy, 8 workers) |
|---|---|---|
| 5 | 91 s | 3 s |
| 10 | 246 s | 7 s |
| 20 | 168 s | 7 s |
| 50 | 822 s | 42 s |
| 100 | not converged in 2 h (77 iterations) | 190 s, 149 iterations (273 s when re-timed) |
| 1000 | out of memory | stopped manually after 2609 s (~4 iterations, ~574 s per iteration in low-memory mode) |

**Mean real-time step** (wall time, s; the interval is 3600 s):

| Method | N = 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|
| Local, sequential | 0.35 | 1.47 | 3.28 | 6.49 | 24.3 |
| Local, parallel | 0.03 | 0.06 | 0.28 | 1.14 | 3.07 |
| Rolling ADMM, sequential | 0.49 | 1.23 | 6.28 | 10.2 | killed (timeout) |
| Rolling ADMM, parallel | 0.02 | 0.05 | 0.13 | 0.73 | 1.94 |
| Rolling central | 0.24 | 0.70 | 2.47 | 8.71 | 37.2 |

N = 1000 has no plan, so its methods are `skipped_no_plan`. At N = 100, interval 19 of
the parallel-local timing rows is missing (probably lost to OneDrive sync); its mean is
over the remaining intervals.

**Savings against community size** (Spearman over buildings):

| Method | ρ | p |
|---|---|---|
| Local | +0.08 | 0.25 |
| Rolling ADMM | −0.18 | 0.017 |
| Rolling central | −0.19 | 0.009 |

The mean saving is £0.06–0.13 per building per day. Parallel and sequential local
re-optimisation give the same community totals, but split them differently between
buildings, because ties between neighbours are broken differently. Figure 62 therefore
uses the sequential local results.

## Computing environment

Python 3.9.21, Pyomo 5.7.3 and Gurobi 11.0.3, on an Intel Core Ultra 5 135U (1.6 GHz base,
up to 4.4 GHz, 12 cores / 14 threads) laptop with 16 GB of memory, running Windows 11. The
parallel solvers use 8 worker processes with one Gurobi thread each.
