# 04 — Day-ahead scaling

Runs the day-ahead showcase comparison
(`02_day_ahead_sharing_methods/showcase_comparison.py`) for larger communities, and
records what each method costs to compute.

## Scripts

| File | Purpose |
|---|---|
| `run_scaling.py` | For each N, builds `data/showcase_<day>_N<NNNN>` if it is missing (`01_data_preparation/build_showcase_dataset.py`), then runs `showcase_comparison.py` with `SHOWCASE_DATASET`, `SHOWCASE_SOLVE_TIMELIMIT` (default 300 s per solve) and `MES_EXPERIMENT=04_day_ahead_scaling`. Each size's figures and `computational_analysis.csv` go to `Results/04_day_ahead_scaling/N<NNNN>/`; all sizes are combined in `scaling_computational_analysis.csv` |
| `algorithm_complexity.md` | Time and memory complexity of the rule-based, central, exchange-ADMM and bilateral-ADMM methods |

## Running

```powershell
C:\conda_envs\myenv\python.exe 04_day_ahead_scaling\run_scaling.py                 # N = 100, 1000
C:\conda_envs\myenv\python.exe 04_day_ahead_scaling\run_scaling.py 100 --timelimit 900 --no-admm
```

## Results so far

`Results/04_day_ahead_scaling/N0100/` holds figures 06–08 and 11 for N = 100, from
2026-09-04. The central MIQCP (N²·T sharing variables) and the bilateral ADMM
(N(N−1)/2 trade pairs) get slow at N = 100, and at N = 1000 they run out of memory while
the model is built. Exchange ADMM keeps O(N·T) state and scales. The real-time scaling
study ([`06_realtime_scaling`](../06_realtime_scaling/README.md)) measures the bilateral
ADMM day-ahead solve, sequential and parallel, up to N = 1000.
