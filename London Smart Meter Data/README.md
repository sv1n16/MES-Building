# Multi-energy building community: P2P energy sharing with ADMM

Day-ahead scheduling and real-time execution of a community of multi-energy buildings
(battery, PV, heat pump, gas boiler, thermal comfort) that share electricity peer to
peer. The communities are built from the London smart-meter (LCL) data on a
time-of-use tariff, with a real PV profile.

## Layout

Code is grouped by experiment. Each experiment folder has a README describing the method,
how to run it and what it writes. Its results go to the folder with the same name under
`Results/`.

| Folder | Experiment | Results |
|---|---|---|
| [`common/`](common/README.md) | Shared model and library code: data loading, the central MIQCP / building model, the bilateral P2P ADMM, plot style, project paths and figure metadata | — |
| [`01_data_preparation/`](01_data_preparation/README.md) | LCL data exploration, showcase-day selection, building the N-building datasets | `Results/01_data_preparation/` |
| [`02_day_ahead_sharing_methods/`](02_day_ahead_sharing_methods/README.md) | Day-ahead comparison: no sharing, rule-based, central, exchange ADMM, bilateral ADMM (± peak reduction), N = 10 | `Results/02_day_ahead_sharing_methods/` |
| [`03_day_ahead_sensitivity/`](03_day_ahead_sensitivity/README.md) | Trade-fee, P2P-price, ADMM ρ and tolerance sweeps; Shapley allocation of the saving | `Results/03_day_ahead_sensitivity/` |
| [`04_day_ahead_scaling/`](04_day_ahead_scaling/README.md) | The day-ahead showcase comparison for N = 100 and 1000; complexity notes | `Results/04_day_ahead_scaling/` |
| [`05_realtime_methods/`](05_realtime_methods/README.md) | Executing the day-ahead plan under forecast error: open loop, fixed schedule, local re-optimisation, rolling ADMM, rolling central, with sequential and parallel solvers | `Results/05_realtime_methods/` |
| [`06_realtime_scaling/`](06_realtime_scaling/README.md) | Real-time methods against community size, N = 5 to 1000, under a compute budget | `Results/06_realtime_scaling/` |
| [`07_communication_failure/`](07_communication_failure/README.md) | Real-time methods when communication fails: building, link and coordinator failures, long outages, late detection, correlated forecast error; ADMM against local, central and plan-following methods | `Results/07_communication_failure/` |
| [`08_minute_resolution/`](08_minute_resolution/README.md) | Hourly inputs turned into 1-minute data; how long bilateral ADMM takes to optimise one hour (60 steps) against central, N = 10 | `Results/08_minute_resolution/` |
| [`09_ramp_rate_control/`](09_ramp_rate_control/README.md) | Three-timescale ADMM with ramp-rate (6a)-(6b) and cumulative (6c) limits as penalties at minute level, correlated forecast error, parallel solving, N = 10 | `Results/09_ramp_rate_control/` |
| [`archive/`](archive/README.md) | Superseded copies of scripts and READMEs, kept for reference (not on the import path) | `Results/archive/` |
| `data/` | Generated model inputs: `showcase_<day>[_N<NNNN>].csv` and `_batteries.csv` | — |
| `Small LCL Data/`, `PV Data/` | Raw input data (not in git) | — |
| `Literature Review/` | Literature notes | — |

## What each script solves

Every script starts with a `PURPOSE` block (the question it answers) and, if it
optimises anything, a `PROBLEM` block with the objective and constraints. The building
model shared by all optimisers (battery, heat pump, boiler, 1R1C thermal, power balance,
P2P limits) is written out once, in full, in
[`common/central_optimisation_showcase.py`](common/central_optimisation_showcase.py);
other scripts call it "building model B_b".

Own cost of building b:  `C_b = Σ_t π_t g_t Δt + c_gas G_t Δt + α (T_t − T_set,t)²`

| Script | Objective (what is minimised) |
|---|---|
| `common/central_optimisation_showcase.py` | `Σ_b C_b + P2P fee + 1e-4·routing`, all buildings in one MIQCP |
| `02_.../admm_energy_sharing.py` | exchange ADMM: `Σ_b C_b` s.t. `Σ_b pex_b,t = 0` (pool) |
| `02_.../admm_bilateral_p2p.py` | bilateral ADMM: per building `C_b + λ·q + ρ/2‖q − z‖²` |
| `common/admm_bilateral_p2p_valley.py` | as above `+ Σ_t w_t g_t²` (peak reduction); used by all real-time scripts |
| `05_.../simulate_realtime_reoptimise.py` (local) | `C_b + w g² + P2P settlement + share_penalty·(undelivered P2P)`, planned trades as caps |
| `05_.../simulate_realtime_reoptimise.py` (admm) | valley ADMM over the remaining hours, every hour |
| `05_.../simulate_realtime_central.py` | `Σ_b (C_b + w g²)` with hard trade consensus, over the remaining hours |
| `05_.../simulate_noisy_realtime.py`, `simulate_open_loop.py` | no optimisation: day-ahead plan plus fixed recourse rules |
| `08_.../three_timescale.py` (minute level) | valley ADMM `+ W·Δt·Σ‖q − q_plan‖²` (sharing plan only) `+` end-of-hour SOC / T targets |
| `09_.../three_timescale_ramp_limits.py` | minute level `+` ramp-excess penalty `W_r/Δt·Σ(r⁺² + r⁻²)` and/or `W_lin·Σ(r⁺ + r⁻)` |
| `03_.../cost_shapley.py`, `admm_tolerance_shapley.py` | Shapley split of `v(S) = Σ_{i∈S} cost_i^alone − cost(S)` |
| `03_.../price_fraction_sweep.py`, `admm_rho_sweep.py`, `admm_bilateral_tolerance_sweep.py` | sweeps of one setting (P2P price, ρ, ε) of the models above |

Scripts not listed are drivers or plotting scripts; their `PURPOSE` block says what they show.

## Running

Use the conda environment with Pyomo and Gurobi:

```powershell
C:\conda_envs\myenv\python.exe 05_realtime_methods\compare_realtime_methods.py --runs 5
```

Scripts can be run from any working directory. Each one starts with a short header
that puts `common/` and every experiment folder on `sys.path`, so modules import one
another by name (see [`common/project_paths.py`](common/project_paths.py)). The
experiment of the script you run decides where its outputs go:
`Results/<experiment>/`, plus `<SHOWCASE_PLOTS_SUBDIR>/` when that variable is set.

## Figure metadata

Every figure saved with matplotlib (PNG, PDF, SVG) or plotly (HTML) is tagged
automatically with:

- title and description, from [`common/figure_catalogue.py`](common/figure_catalogue.py);
- experiment, producing script and full command line;
- dataset (N, T, day), git commit and creation time.

Figures made before this was added were tagged retrospectively, with what the folder and
catalogue tell us. Each results folder has a generated `README.md` index of its files.

```powershell
C:\conda_envs\myenv\python.exe common\tag_figures.py --show Results\06_realtime_scaling\60_scaling_solve_time.png
C:\conda_envs\myenv\python.exe common\tag_figures.py            # tag new untagged files and rebuild the indexes
```

When a script starts writing a new kind of output, add an entry for it to
`common/figure_catalogue.py`.

## Computing environment

Simulations are run with Python 3.9.21, Pyomo 5.7.3 and Gurobi 11.0.3 on a laptop with an
Intel Core Ultra 5 135U CPU (1.6 GHz base, up to 4.4 GHz, 12 cores / 14 threads) and
16 GB of memory, running Windows 11.
