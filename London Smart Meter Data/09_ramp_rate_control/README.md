# 09 — Ramp-rate control: three-timescale ADMM with a ramp limit on grid imports (N = 10)

This is the three-timescale rolling optimisation of
[`08_minute_resolution/three_timescale.py`](../08_minute_resolution/three_timescale.py) (day-ahead,
hourly, minute), with a ramp limit added to the objective of every minute-level ADMM subproblem. It
runs with correlated forecast error and with the building subproblems solved in parallel (10 workers,
measured times).

## The limit

Each building's grid import may change by at most a set amount per minute:

|g(n,t) − g(n,t−1)| ≤ ramp rate × rated import(n)

- **Rated import:** the building's largest possible import, i.e. peak load + heat pump at its lowest COP + battery charging. At 10%/min this gives 0.36–1.19 kW/min.
- **Ramp rate:** 10% per minute, the operators' benchmark (operators use 5–20%); 2% and 1% are also tested (`--ramp-rate`).
- **The previous minute** is the grid import actually applied.
- **The penalty on the excess** (`--ramp-penalty`) is one of:
  - `linear`, *proportional*: 1000 GBP per kWh of excess (`--ramp-linear-weight`). It is exact: no excess at all.
  - `quadratic`, *squared*: 1000 GBP per kW² per hour (`--ramp-weight`). It always leaves tiny excesses.
  - `both`.
- **Plan deviation:** the weight for departing from the hourly plan is 1 GBP per kW² per hour (`--tracking-weight`; 10 in experiment 08). The end-of-hour battery and temperature targets are always kept; without them the minute level, which only sees to the end of the hour, would empty the batteries.

The limit is implemented in both the Pyomo model (`three_timescale.track_hourly_plan`) and the
parallel gurobipy workers (`05_realtime_methods/admm_parallel._build`).

## Running

```powershell
$py = "C:\conda_envs\myenv\python.exe"
& $py 09_ramp_rate_control\three_timescale_ramp_limits.py --grid --reference                 # no limit
& $py 09_ramp_rate_control\three_timescale_ramp_limits.py --grid --ramp-penalty linear       # recommended
& $py 09_ramp_rate_control\three_timescale_ramp_limits.py --grid --ramp-rate 0.02            # squared, 2 %/min
& $py 09_ramp_rate_control\three_timescale_ramp_limits.py --grid --clouds                    # with passing clouds
& $py 09_ramp_rate_control\three_timescale_ramp_limits.py --grid --tracking-weight 10 --ramp-weight 100
```

One full day takes about 4–8 minutes. Exceptions are plan-deviation weight 0, and ramp weight
10,000 at any plan weight: a few building subproblems become very hard to solve and single minute
solves take many minutes. Those settings are not usable in real time.

`--clouds` adds passing clouds to the actual minute PV (`three_timescale.cloud_shading`): about 2
per hour between 07:00 and 19:00, PV −50 to −80% for 3–15 min, every building within 0–2 min, not forecast.

## Results layout (`Results/09_ramp_rate_control/`)

| Folder | Content | Script |
|---|---|---|
| `runs/<name>/` | One full-day run each, named `plan<p>_<limit>[_clouds]` with `<limit>` = `no_limit`, `<r>pct_squared<w>`, `<r>pct_linear<w>` or `<r>pct_squared<w>_linear<w>`. Each holds `85_` timing, `86_` dispatch, `89_` per-building ramp CSVs and their figures (`87`, `88`, `89`) | `three_timescale_ramp_limits.py` |
| `1_ramp_rate_sweep/` | 10, 2, 1 %/min against no limit: `96_grid_ramp_sweep_<measure>.png` (+ `.csv`), and `91_grid_ramp_timeseries_<run>.png` (each building's ramp against its limit) | `plot_sweep.py`, `plot_ramp_timeseries.py` |
| `2_cloud_transients/` | The same with passing clouds: `97_…`, `91_…_clouds_…` | `plot_sweep.py`, `plot_ramp_timeseries.py` |
| `3_penalty_sweep/` | Plan-deviation weight 0–100 × ramp weight 0–10,000: `98_penalty_sweep_<measure>.png`, `99_penalty_correlations_matrix.png`, `99_penalty_correlations_cost_vs_<measure>.png` (+ `.csv`) | `plot_penalty_sweep.py`, `plot_penalty_correlations.py` |
| `4_penalty_type/` | Squared, proportional or both: `100_penalty_type_<measure>.png` (+ `.csv`) | `plot_penalty_type.py` |
| `logs/` | Console logs of the batch runs | — |

Every figure shows one measure.

## 1. Tighter ramp limits (squared penalty 1000, plan weight 1)

| Grid ramp limit | Community's largest grid ramp | Building-minutes over the limit | Cost | P2P shared |
|---|---|---|---|---|
| None | 28.5 kW/min | — | £32.19 | 62.7 kWh |
| 10%/min | 7.1 kW/min | 0.7% | £32.69 (+1.5%) | 63.1 kWh |
| 2%/min | 1.7 kW/min | 5.0% | £34.73 (+7.9%) | 63.5 kWh |
| 1%/min | 1.4 kW/min | 10.0% | £36.34 (+12.9%) | 63.8 kWh |

- **The grid is smoothed.** The largest ramps are at hour boundaries, when the new hourly plan arrives.
- **The trade-off is real.** Below 2%/min the grid is barely smoother, but cost keeps rising.
- **P2P sharing rises slightly** as the limit tightens: buildings use each other to smooth their own imports.

## 2. Passing clouds

| Grid ramp limit | Community's largest grid ramp | Cost | P2P shared |
|---|---|---|---|
| None | 28.6 kW/min | £32.34 | 60.2 kWh |
| 10%/min | 7.1 kW/min | £32.84 (+1.5%) | 60.5 kWh |
| 2%/min | 1.7 kW/min | £34.85 (+7.8%) | 60.9 kWh |
| 1%/min | 1.4 kW/min | £36.48 (+12.8%) | 61.2 kWh |

Clouds add about £0.15 a day and cut sharing by 2.5 kWh, but do not change the picture. The largest
ramps are at hour boundaries, not under the clouds.

## 3. Plan-deviation and ramp penalties (10%/min, squared penalty)

The sweep covers plan-deviation weights 0, 1, 3, 10, 30 and 100, against ramp weights 0, 100, 1000
and 10,000. That is 22 runs; plan 0 at ramp 1000 and 10,000 were stopped as too slow.

- **Cost follows the plan-deviation weight** (Spearman ρ = +0.96). It rises from £32.0 (weight 0)
  to £34.8 (weight 100) without a limit. The minute level saves money when it may act on what it
  measures instead of following the hourly plan.
- **Grid smoothing follows the ramp weight** (ρ = −0.91 with the community's largest ramp). Ramp
  weight ≥ 100 brings it from about 26–30 to about 7 kW/min. At plan weights 30–100, ramp 100 is not
  enough: the pull back to the hourly plan works against the ramp penalty.
- **Smoothing is almost free:** at any plan weight the limit adds only £0.3–0.5 a day.
- **The share of minutes over the limit is misleading** with the squared penalty. It rises with the
  ramp weight (ρ = +0.95), because the squared penalty spreads one large jump over several minutes
  that are each slightly over. The total or largest excess falls 2,000-fold. See section 4.
- **Best setting:** plan weight 1, ramp weight 1000. It costs £32.69 with a largest community ramp
  of 7.1 kW/min, and every minute solves in seconds.

## 4. Proportional against squared penalty (10%/min, plan weight 1)

| | No limit | Squared only | Proportional only | Both |
|---|---|---|---|---|
| Building-minutes over the limit | 47 | 101 | 0 | 0 |
| Total excess over the day | 44.6 kW/min | 0.17 kW/min | 0 | 0 |
| Community largest grid ramp | 28.5 kW/min | 7.07 kW/min | 7.02 kW/min | 7.02 kW/min |
| Electricity cost | £32.19 | £32.69 | £32.69 | £32.69 |
| P2P shared | 62.7 kWh | 63.1 kWh | 62.7 kWh | 63.1 kWh |
| Minute solve, measured (median / max) | 0.16 / 2.6 s | 0.20 / 3.6 s | 0.11 / 2.0 s | 0.17 / 3.1 s |

The proportional penalty enforces the limit exactly, at the same cost and grid smoothing as the
squared one, and solves fastest. Adding the squared term on top changes nothing but solve time.
**Use the proportional penalty alone.**

## Earlier: limits (6a)–(6c) on P2P trades (results deleted)

The first version limited peer-to-peer trades instead (run without `--grid`):
- **(6a)/(6b):** producers' P2P sales may change by at most ramp rate × (PV peak + battery power) per minute.
- **(6c):** consumers' cumulative purchases within the hour may not exceed the hourly plan's.

The results were deleted. In summary:
- **Cost and sharing never changed** (within £0.10 a day), even at 1%/min or with clouds.
- **Why:** P2P trades are transfers inside the community, which the batteries and heat pumps absorb within the hour.
- **The grid was not smoothed:** the community's largest grid ramp stayed at 26 kW/min in every run.

That is why the limit was moved to grid imports. The plotting script for those runs is in
`archive/09_ramp_rate_control/plot_comparison_p2p_limits.py`.
