# 08 — Minute-level data: how long does ADMM take to optimise an hour? (N = 10)

The hourly showcase inputs are turned into 1-minute data. The community problem is then
solved over one hour (60 one-minute steps) with bilateral ADMM, for several hours of the
day. The time it takes is compared with the 1-minute and 1-hour limits, and with a
central solve of the same hour.

## Method

**Minute data** (`--interpolation`):
- `linear` (default): load, PV and outdoor temperature are interpolated linearly between
  the hourly mid-points, with PV clipped at zero. The tariff and the heating setpoint stay
  constant within each hour, because they are step schedules.
- `step`: every input is held constant within its hour, which keeps the hourly averages
  exactly.

**Model at minute resolution.**
- The step is 1/60 h. Energy costs, battery dynamics and the thermal model already scale
  with it.
- Two terms are charged per step: the comfort penalty α(T_in − setpoint)² and the
  congestion term. Both are multiplied by the step length, so one hour of minutes weighs
  the same as one hourly step.
- Each hour starts from the battery charge and indoor temperature that the day-ahead
  hourly plan predicts (`Results/05_realtime_methods/bilateral_admm_schedule.npz`).
- The battery may act from the first minute.

**Timing (parallel estimate only).** Every subproblem solve is timed. The ADMM time is the
slowest building's solve in each iteration, summed over iterations (one solver per
building, in parallel), plus the model build divided by N. ADMM runs from two starts:
- **cold**: zero prices and trades;
- **warm**: the hourly plan's trades for that hour.

The same hour is also solved centrally (one solve), as a cost check and timing reference.

## Running

```powershell
$py = "C:\conda_envs\myenv\python.exe"
& $py 08_minute_resolution\admm_minute_timing.py                  # 03:00, 08:00, 13:00, 17:00, 20:00
& $py 08_minute_resolution\admm_minute_timing.py --hours all      # every hour of the day
& $py 08_minute_resolution\admm_minute_timing.py --hours 17 --starts cold
```

Run it while the machine is otherwise idle and kept awake: other jobs and sleep inflate
the timings.

## Outputs (`Results/08_minute_resolution/`)

| File | Content |
|---|---|
| `80_minute_inputs.png` | Hourly against minute inputs (load and PV of one building, outdoor temperature, tariff) |
| `81_admm_minute_timing.csv` | Per start hour and ADMM start: iterations, ADMM parallel time (solve and build), central time, cost of ADMM and central, and their gap |
| `82_admm_minute_timing.png` | ADMM parallel time per start hour (cold and warm) against central, the 1-minute and 1-hour lines, and the iterations |
| `83_admm_minute_convergence.png` (and `.csv`) | ADMM residuals (consensus gap and change in agreed trades) against elapsed time, cold and warm, with the stopping tolerance and the central solve time |
| `84_minute_dispatch.png` (and `.csv`) | The optimised hour minute by minute, ADMM against central: grid import, battery discharge − charge, P2P traded, mean indoor temperature |

`--plot-only` redraws every figure from the saved CSVs without solving.

The files above (single hours, `admm_minute_timing.py`) stay in the root. Subfolders:
- `1_three_timescale/`: `three_timescale.py` (day-ahead hourly, hourly re-solve, minute
  re-solve of the rest of the hour). `85`–`88` and the log.
- `2_minute_rolling/`: the runs that use 1-minute steps for the whole day
  (`rolling_minute_day.py`: a 1440-step day-ahead plan, then the rest of the day re-planned
  every minute). `90_*.csv` timing, `91_*.csv` dispatch, `92_*.png` timing figure, and the logs.
- `3_tracking_comparison/`: what the minute level of `three_timescale.py` penalises
  (`compare_tracking.py`; see the section near the end).

## Result: 17:00, 10 buildings, idle machine

| | Iterations | Time to optimise the hour (two runs, other jobs paused) |
|---|---|---|
| ADMM, cold start | 97 | 13.3 s / 37.5 s (parallel estimate) |
| ADMM, warm start | 56 | 7.7 s / 19.5 s (parallel estimate) |
| Central | — | 1.5 s / 3.1 s |

ADMM reaches the central optimum: £3.5754 (cold) and £3.5745 (warm), against £3.5741 for
the hour. The iterations and costs are identical between runs. The timings differ by
about 2.5 times because of the laptop's clock boost and thermal state (Balanced power
plan), so quote a range or repeat the test. Either way:
- re-planning once an hour uses about 1% of the interval or less;
- re-planning every minute fits within 60 s, but with little margin from a cold start.

A first run made while another job shared the CPU took 49.6 s, 29.3 s and 5.6 s.

Over this peak-tariff hour the community imports nothing: the batteries cover the whole
load at a steady 12.8 kW. ADMM and central give the same battery dispatch and indoor
temperatures. ADMM routes more energy peer to peer (5–6.7 kW against 4.2–5.4 kW) at the
same total cost, because who trades with whom is not unique.

## Three-timescale rolling optimisation (`three_timescale.py`)

All three levels use bilateral ADMM.

1. **Day-ahead:** one solve of the whole day at hourly resolution.
2. **Hourly:** at the start of every hour, a re-solve of the remaining hours from the state
   actually reached. It is warm-started from the day-ahead plan first, then from the
   previous hourly solution (trades and dual prices, shifted by one hour).
3. **Minute:** at every minute of the hour, a re-solve of the remaining minutes of that
   hour. The first minute is applied.
   - **Goal: keep to the hourly sharing plan.** Each building is penalised only for
     departing from the hourly solution's P2P trades:
     `W·Δt·Σ_t Σ_j (q_j,t − q_j^plan)²` (`--tracking-weight` W, default 10 GBP per kW² per
     hour). Grid import and battery power are free to move within the hour.
   - **End-of-hour state:** penalties for missing the plan's battery charge
     (`--soc-weight`, 100) and indoor temperature (`--temperature-weight`, 10), so the next
     hourly solve starts where the plan expected.
   - The economic cost (energy, gas, comfort, congestion) stays in the objective.
   - **Warm start:** the hourly solution's trades, and its dual prices scaled to the
     1-minute step; then the previous minute's solution.
   - **Up:** the state at the end of the hour starts the next hourly solve.

There is no ramp-rate control in this experiment. `09_ramp_rate_control` adds it to the
minute level through the `three_timescale.EXTENSION` hook.

```powershell
& $py 08_minute_resolution\three_timescale.py --correlated --parallel            # the whole day: 24 hourly and 1440 minute solves
& $py 08_minute_resolution\three_timescale.py --correlated --parallel --hours 17 # one hour at minute level (hourly rolling up to it)
& $py 08_minute_resolution\three_timescale.py --correlated --parallel --plot-only
```

| File | Content |
|---|---|
| `85_three_timescale_timing<suffix>.csv` | One row per solve: level, hour, minute, horizon, iterations, time |
| `86_three_timescale_dispatch<suffix>.csv` | Per minute: grid import, battery power and energy, P2P traded, indoor temperature, the hourly and day-ahead plans for that hour, and the departure from the hourly plan |
| `87_three_timescale_timing<suffix>.png` | Every solve of the three levels through the day against the 1-minute and 1-hour lines, and per-level boxes |
| `88_three_timescale_dispatch<suffix>.png` | The minute level against the hourly plan: grid import, battery power, P2P, departure from the plan, indoor temperature |

`<suffix>` is `_with_error`, `_with_error_correlated` and/or `_parallel` for the options used.

## With forecast error, solved in parallel (`--error`, `--correlated`, `--parallel`)

**Error model.**
- The day-ahead plan uses the forecast.
- The hourly level knows the actual average of the current hour, from experiment 05's
  noise scenario 1: 20% load and 30% PV error per building and hour, 1 °C outdoor
  temperature.
- The minute level measures the actual current minute: the hourly actuals plus 10%
  minute-level fluctuation of load and PV.
- `--correlated` takes the hourly actuals from 05's `correlated` case (80% of the error
  common to all buildings). Its minute fluctuations are also 80% common and persist with a
  10-minute time constant.

**Parallel solving.** `--parallel` solves the building subproblems at the same time in 10
worker processes, one per building (05 `admm_parallel.ParallelADMM`, extended with the
sharing-plan tracking terms). Times are measured.

**Current result: correlated error, full day, 10 buildings, sharing-plan tracking only**
(`--correlated --parallel`):

| Level | Solves | Iterations (median / max) | Time per solve (median / 95th percentile / max) |
|---|---|---|---|
| Day-ahead | 1 | 247 | 10.5 s |
| Hourly | 24 | 23.5 / 71 | 0.33 / 0.68 / 0.87 s |
| Minute | 1440 | 2 / 12 | 0.12 / 0.27 / 0.53 s |

Every solve converged and every minute solve fits well inside its 60 seconds.

| Departure from the hourly plan, summed over buildings (median / 95th percentile / max) | |
|---|---|
| P2P trades | 0.01 / 0.29 / 1.04 kW |
| Battery power | 0.23 / 2.45 / 4.2 kW |
| Grid import | 1.53 / 9.9 / 21.6 kW (largest in hour 00) |

- The minute level keeps to the sharing plan; grid import and batteries absorb the
  within-hour variation instead.
- Every hour ends within 0.02 kWh of the plan's battery target.
- Over the day: grid import 420.3 kWh, against 417.1 kWh in the hourly plans and 422.6 kWh
  in the day-ahead plan; electricity cost £32.23 against £33.51 for the hourly plans
  (−3.8%). With grid import no longer tied to the plan, the minute level can move import
  into cheaper minutes.

Earlier runs (no error, independent error) used the previous objective, which also tracked
grid import and battery power. They were removed and have not been re-run.

## What should the minute level penalise? (`--tracking`, `compare_tracking.py`)

Two minute-level objectives, on the same day and the same correlated forecast error
(`three_timescale.py --correlated --parallel --tracking <mode>`):

| `--tracking` | Penalised at the minute level |
|---|---|
| `trades` (default, previous) | W Δt Σ (q_j − q_j^plan)² on every pairwise trade, plus W_soc (s_59 − s^plan)² + W_T (T_59 − T^plan)² (100, 10) |
| `grid` (new) | W_g Δt Σ (g − g^plan)² on grid import only (trades re-route freely), the end-of-hour state at the hourly solve's shadow prices − μ_soc s_59 − μ_T T_59, plus small quadratics (1, 1) |

μ comes from the duals of the battery and temperature dynamics linking hour h to hour h+1 in
the hourly ADMM subproblems (`admm_parallel.ParallelADMM.state_values`), so `--tracking grid`
needs `--parallel`. The battery's value was about £0.05/kWh off-peak and £0.33/kWh for 16:00–21:00.

Results in `Results/08_minute_resolution/3_tracking_comparison/`:

| Path | Content |
|---|---|
| `runs/trades/`, `runs/grid/` | Each run's 85–88 files and log |
| `95_tracking_comparison.csv` | One row per mode, every measure |
| `95_tracking_comparison_<measure>.png` | One bar figure per measure |
| `96_tracking_timeseries_<measure>.png` | One time-series figure per measure, and the cumulative cost difference |

Whole day, N = 10, correlated error, 10 parallel workers:

| Measure | Trades (previous) | Grid (new) |
|---|---|---|
| Electricity + gas cost | £85.98 | £86.09 (+0.1%) |
| Comfort penalty | £94.48 | £73.27 (−22%) |
| Peak community grid import | 34.3 kW | 31.7 kW (−7.8%) |
| Energy shared P2P | 60.9 kWh | 62.5 kWh |
| P2P re-routed away from the hourly plan | 1.3 kWh | 16.7 kWh |
| Grid import away from the hourly plan | 58.2 kWh | 1.5 kWh |
| Minute solves: median / max time | 0.11 / 0.59 s | 0.14 / 10.9 s |
| Minute solves: median / max ADMM iterations | 2 / 12 | 5 / 351 |

With grid tracking the buildings absorb the deviations among themselves (13× more re-routing)
instead of passing them to the grid, so the community follows its hourly grid plan almost
exactly and the evening peak falls. The slow minute solves are all at minute 0 of an hour:
the warm start (the hourly plan's trades) no longer matches how the minute level routes them.
