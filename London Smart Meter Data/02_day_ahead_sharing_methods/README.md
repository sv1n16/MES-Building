# 02 — Day-ahead energy sharing methods (N = 10)

Day-ahead scheduling of a 10-building community that can share electricity peer to
peer. The community is solved centrally and with two decentralised ADMM algorithms, and
compared with no sharing and a rule-based baseline. The bilateral ADMM schedule is the
day-ahead plan that the real-time study ([`05_realtime_methods`](../05_realtime_methods/README.md))
executes under forecast error.

## Community and model

- **Day:** the showcase day 2013-02-21 (24 hourly steps), chosen in
  [`01_data_preparation`](../01_data_preparation/README.md).
- **Buildings:** 10 LCL ToU households with mixed assets:

  | Assets | Buildings |
  |---|---|
  | Battery + PV | MAC005290, MAC000635, MAC004245 |
  | Battery only | MAC002861, MAC003427 |
  | PV only | MAC000991, MAC001369 |
  | None | MAC001707, MAC003472, MAC005341 |

  Batteries are 3.9–8.0 kWh, with power equal to half the capacity. PV systems are
  3.5 kWp.
- **Per-building model** (`../common/central_optimisation_showcase.py`):
  - battery with 90% charge and discharge efficiency and a charge/discharge indicator;
  - heat pump (5 kW, COP 2.18 at 7 °C, plus 0.01 per °C);
  - gas boiler (90% efficient, 5 p/kWh);
  - first-order thermal model;
  - quadratic comfort penalty, α = 0.5 × (T_in − T_set)².
- **Objective:** grid electricity + gas + comfort penalty, plus P2P settlement. With
  `FEE_MODE = "market"`, the buyer pays the seller 0.5 × the grid tariff per kWh traded.
  This is a pure transfer between buildings, so it does not change the community optimum.

## Methods

| Method | Script | How sharing is decided |
|---|---|---|
| No sharing | `../common/central_optimisation_showcase.py` (`own_battery`) | Each building optimises its own assets |
| Central | `../common/central_optimisation_showcase.py` (`shared`) | One MIQCP for the whole community, with N² sharing variables per hour (Gurobi) |
| Exchange ADMM | `admm_energy_sharing.py` | Each building chooses its net export to a common pool. One dual per hour clears the pool, and at convergence it is the emergent P2P price |
| Bilateral ADMM | `admm_bilateral_p2p.py` | Fully decentralised consensus ADMM (Wang, Wu & Wang, IEEE TSG 2017, Algorithm 2). One agreed trade per building pair per hour, with no coordinator |
| Bilateral ADMM + peak reduction | `../common/admm_bilateral_p2p_valley.py` | As above, plus a quadratic grid-import penalty whose weight rises with price (0.01–0.05 £/kW²). It pushes import into the cheap valley hours |
| Rule-based | `compare_energy_sharing_methods.py` | Solver-free: thermostat heating (heat pump first, then boiler) and deterministic donor/receiver matching |

The charge/discharge indicator is relaxed to [0, 1] (`RELAX_BINARIES = True`), so each
ADMM subproblem is a convex QP and convergence is guaranteed. The relaxation is tight in
practice, because charging losses make simultaneous charge and discharge wasteful.

## Scripts

| Script | Purpose | Main outputs |
|---|---|---|
| `compare_energy_sharing_methods.py` | **Main comparison** of the six methods above | figures 22–30, 33; `28_building_method_comparison.csv` |
| `showcase_comparison.py` | Central vs both ADMMs: schedules, sharing flows, per-building benefit, comfort, peak, cost | figures 06–08, 11, 13, 15–19; `computational_analysis.csv` |
| `admm_energy_sharing.py` | Exchange ADMM | figure 12 |
| `admm_bilateral_p2p.py` | Bilateral ADMM without the peak-reduction term (original version) | figure 14 |
| `rule_based_energy_sharing.py` | Stand-alone half-hourly rule-based sharing over many days | `data/rule_based_sharing_results.csv` |

Results go to `Results/02_day_ahead_sharing_methods/`.

## Running

```powershell
C:\conda_envs\myenv\python.exe 02_day_ahead_sharing_methods\compare_energy_sharing_methods.py
C:\conda_envs\myenv\python.exe 02_day_ahead_sharing_methods\compare_energy_sharing_methods.py --no-admm   # quick central vs rule-based
C:\conda_envs\myenv\python.exe 02_day_ahead_sharing_methods\showcase_comparison.py [--no-admm] [--admm-plots]
```

## Results (showcase day, N = 10)

From `Results/02_day_ahead_sharing_methods/22_energy_sharing_method_comparison.csv`, run
on 2026-10-17. The operating cost is grid electricity + gas + comfort penalty per day.

| Method | Operating cost (£/day) | Saving vs no sharing (£/day) | Peak grid import (kW) | Grid import (kWh) | Energy traded (kWh) | Comfort RMS (°C) |
|---|---|---|---|---|---|---|
| No sharing | 87.50 | — | 33.9 | 435.7 | 0 | 0.73 |
| Rule-based | 174.05 | −86.55 | 35.5 | 495.9 | 2.4 | 0.95 |
| Central | 83.77 | 3.72 | 35.4 | 428.7 | 7.1 | 0.73 |
| Exchange ADMM | 83.77 | 3.72 | 34.4 | 428.7 | 10.4 | 0.73 |
| Bilateral ADMM | 83.77 | 3.72 | 34.4 | 428.7 | 10.4 | 0.73 |
| Bilateral ADMM + peak reduction | 84.63 | 2.86 | **30.4** | 422.7 | 57.6 | 0.73 |

1. **Both ADMM algorithms reach the central optimum**, to within £0.0001/day. The traded
   volume differs (10.4 vs 7.1 kWh) because several routings give the same optimum.
2. **Sharing saves about 4.3%** (£3.72/day). Most of the cost is gas and comfort, which
   sharing does not affect.
3. **Peak reduction trades a little cost for a lower peak.** It cuts the community peak
   by 5 kW (−14% against central) for £0.86/day, and trades about 5.5× more energy.
4. **The rule-based baseline costs twice as much.** Its thermostat ignores the tariff,
   and its sharing rules move very little energy.

**Re-running changes the bilateral settings.** That run used the older copy of the
bilateral optimiser, now `archive/admm_bilateral_p2p_valley_dayahead_2026-10.py`, with
tolerance 1e-5 and at most 1000 iterations. Every study now uses
`common/admm_bilateral_p2p_valley.py`: tolerance 1e-4, at most 500 iterations, plus warm
starts. Re-running `compare_energy_sharing_methods.py` therefore re-solves with these
settings rather than reusing the cached `.results.json`. The tolerance sweep in
[`03_day_ahead_sensitivity`](../03_day_ahead_sensitivity/README.md) puts the 1e-4 solution
within £0.01/day of central.
