# Day-Ahead ADMM Energy Sharing

Day-ahead scheduling of a 10-building community that can share electricity peer
to peer (P2P). The same community is solved centrally and with two decentralised
ADMM algorithms, and compared against no sharing and a rule-based baseline. The
bilateral ADMM schedule produced here is the day-ahead plan that the real-time
study in [`../Real-Time ADMM`](../Real-Time%20ADMM/README.md) executes under
forecast error.

## Community and model

- **Day:** the showcase day `2013-02-21` (24 hourly steps), chosen by
  `sharing_showcase.py` as the day where sharing batteries pays off most.
- **Buildings:** 10 London smart-meter (LCL) households on the time-of-use tariff,
  with mixed assets:

  | Assets | Buildings |
  |---|---|
  | Battery + PV | MAC005290, MAC000635, MAC004245 |
  | Battery only | MAC002861, MAC003427 |
  | PV only | MAC000991, MAC001369 |
  | None | MAC001707, MAC003472, MAC005341 |

  Batteries are 3.9–8.0 kWh with power equal to half the capacity; PV systems are
  3.5 kWp with a real February profile from the project's `PV Data` folder.
  Outdoor temperature and the heating setpoint (18 °C base, 21 °C comfort windows)
  are synthetic, because the LCL data has no heating or weather.
- **Per-building model** (in `../central_optimisation_showcase.py`): battery with
  90% charge and discharge efficiency and a charge/discharge indicator; heat pump
  (5 kW, COP 2.18 at 7 °C plus 0.01 per °C); gas boiler (90% efficient, 5 p/kWh);
  first-order thermal model; quadratic comfort penalty
  (α = 0.5 × (T_in − T_set)²).
- **Tariff:** time of use, from the LCL tariff file (£0.672/kWh in the High
  periods).
- **Objective:** grid electricity + gas + comfort penalty, plus P2P settlement.
  With `FEE_MODE = "market"` the buyer pays the seller a fraction of the grid
  tariff per kWh traded. This is a pure transfer between buildings, so it moves
  benefit between them but does not change the community optimum.

## Methods

| Method | Script | How sharing is decided |
|---|---|---|
| No sharing | `../central_optimisation_showcase.py` (`own_battery`) | Each building optimises its own assets; no P2P |
| Central | `../central_optimisation_showcase.py` (`shared`) | One MIQCP over the whole community with N² sharing variables per hour (Gurobi) |
| Exchange ADMM | `admm_energy_sharing.py` | Each building chooses its net export to a common pool; one shared dual per hour drives the pool imbalance to zero. At convergence the dual is the emergent P2P price |
| Bilateral ADMM | `admm_bilateral_p2p.py` | Fully decentralised consensus ADMM (Wang, Wu & Wang, IEEE Trans. Smart Grid 2017, Algorithm 2). One agreed trade per building pair per hour; each pair reconciles its two copies, with no coordinator |
| Bilateral ADMM + peak reduction | `admm_bilateral_p2p_valley.py` | As above, plus a quadratic grid-import penalty whose weight rises with price (0.01–0.05 £/kW²). This pushes import into the cheap "valley" hours and cuts the peak |
| Rule-based | `compare_energy_sharing_methods.py` | Solver-free: thermostat heating (heat pump first, then boiler) and deterministic donor/receiver matching |

Bilateral ADMM settings: ρ = 1.1, at most 1000 iterations, primal and dual
tolerance 1e-5 kW (RMS), trades settled at 0.5 × the grid tariff. The
charge/discharge indicator is relaxed to [0, 1] (`RELAX_BINARIES = True`), so each
subproblem is a convex QP and ADMM convergence is guaranteed. The relaxation is
tight in practice, because charging losses make simultaneous charge and discharge
wasteful.

## Scripts

| Script | Purpose | Main outputs |
|---|---|---|
| `build_showcase_dataset.py` | Builds the hourly 24-step input (loads, PV, tariff, temperature, setpoint) and the battery/PV assets table | `../data/showcase_<day>.csv`, `../data/showcase_<day>_batteries.csv`, figure 10 |
| `sharing_showcase.py` | Ranks days by the benefit of pooling batteries (price water-filling, no solver) | figure 05 |
| `admm_energy_sharing.py` | Exchange ADMM | figure 12 |
| `admm_bilateral_p2p.py` | Bilateral ADMM | figure 14 |
| `admm_bilateral_p2p_valley.py` | Bilateral ADMM with the optional peak-reduction term (`use_congestion`) | used by the comparison |
| `showcase_comparison.py` | Central vs both ADMMs on the showcase day: schedules, sharing flows, per-building benefit, comfort, peak, cost | figures 06–08, 11, 13, 15–19, `computational_analysis.csv` |
| `compare_energy_sharing_methods.py` | **Main comparison**: no sharing, rule-based, central, exchange ADMM, bilateral ADMM with and without peak reduction | figures 22–30 |
| `cost_shapley.py` | Allocates the community saving to buildings with Shapley values (permutation sampling, or `--exact`) | figure 34 |
| `admm_rho_sweep.py` | Iterations to converge and cost accuracy against ρ | figure 20 |
| `admm_bilateral_tolerance_sweep.py` | Accuracy and run time of bilateral ADMM against its stopping tolerance | `31_*.csv`, `32_*.csv` |
| `price_fraction_sweep.py` | Re-settles the same dispatch at P2P prices from 0.1 to 1.0 × the tariff and checks that every building still benefits | figure 21 |
| `run_scaling.py` | Re-runs `showcase_comparison.py` for larger communities (N = 100, 1000) | `N<NNNN>/` folders |
| `rule_based_energy_sharing.py` | Stand-alone half-hourly rule-based sharing over many days | `../data/rule_based_sharing_results.csv` |
| `inspect_data.py`, `test_collation.py` | Quick checks of the PV, weather and device data | console only |
| `algorithm_complexity.md` | Time and memory complexity of each method | — |

## Running

The scripts import `central_optimisation_showcase.py` and `plot_timeseries.py`
from the parent folder, so put that folder on the Python path. Use the conda
environment that has Pyomo and Gurobi:

```powershell
$env:PYTHONPATH = ".."
python .\compare_energy_sharing_methods.py          # main comparison
python .\compare_energy_sharing_methods.py --no-admm  # quick central vs rule-based
python .\showcase_comparison.py
python .\cost_shapley.py --permutations 300
python .\admm_bilateral_tolerance_sweep.py
python .\admm_rho_sweep.py --method both
```

## Results (showcase day, N = 10)

From `Results/MES ADMM Optimisation Community Size 10/22_energy_sharing_method_comparison.csv`.
Operating cost is grid electricity + gas + comfort penalty per day; the saving is
in building energy cost against no sharing.

| Method | Operating cost (£/day) | Saving vs no sharing (£/day) | Peak grid import (kW) | Grid import (kWh) | Energy traded (kWh) | Comfort RMS (°C) |
|---|---|---|---|---|---|---|
| No sharing | 87.50 | — | 33.9 | 435.7 | 0 | 0.73 |
| Rule-based | 174.05 | −86.55 | 35.5 | 495.9 | 2.4 | 0.95 |
| Central | 83.77 | 3.72 | 35.4 | 428.7 | 7.1 | 0.73 |
| Exchange ADMM | 83.77 | 3.72 | 34.4 | 428.7 | 10.4 | 0.73 |
| Bilateral ADMM | 83.77 | 3.72 | 34.4 | 428.7 | 10.4 | 0.73 |
| Bilateral ADMM + peak reduction | 84.63 | 2.86 | **30.4** | 422.7 | 57.6 | 0.73 |

### Findings

1. **Both ADMM algorithms reach the central optimum.** Exchange and bilateral
   ADMM match the central cost to within £0.0001/day. The traded volume differs
   (10.4 vs 7.1 kWh) because several routings give the same optimum and ADMM
   finds a different one.
2. **Sharing saves about 4.3%** (£3.72/day) over each building optimising alone.
   The saving is small because most cost is gas and comfort, which sharing does
   not affect.
3. **Peak reduction trades a little cost for a lower peak.** The peak-reduction
   variant cuts the community peak by 5 kW (to 30.4 kW, −14% against central) for
   £0.86/day, and trades about 5.5 times more energy (57.6 kWh): buildings with
   spare battery or PV fill each other's high-price hours.
4. **The rule-based baseline is far worse** (twice the operating cost). Its
   thermostat heating ignores the tariff, and its sharing rules move very little
   energy.
5. **The saving is concentrated.** By Shapley value
   (`34_cost_shapley_values.csv`, 300 permutations), three buildings receive 84%
   of the community saving: MAC000635 (30%), MAC004245 (28%), both battery + PV,
   and MAC002861 (26%, battery only). The other seven get 1–3% each.
6. **Tolerance sets the accuracy and run time** (`31_admm_tolerance_sweep_summary.csv`).
   At a tolerance of 1e-3 bilateral ADMM converges in 19 iterations (9 s) and is
   within £0.05/day of the central cost. At 1e-4 it takes 24 iterations (11 s) and
   is within £0.01. At 1e-5 it does not reach the tolerance within 1000 iterations
   in that sweep (605 s), although its cost is then essentially exact.
7. **Scaling:** central and bilateral ADMM need O(N²·T) sharing variables and run
   out of memory or time at N = 1000. Exchange ADMM keeps O(N·T) state and scales
   (see `algorithm_complexity.md`).

## Output folders

- `Results/MES ADMM Optimisation Community Size 10/` holds the current results
  for N = 10 (figures 01–34 and their CSVs). Figures 22–30 are the method
  comparison, 31–32 the tolerance sweep, and 34 the Shapley allocation.
- `Results/` (top level) holds the earlier figures 01–27 and the ρ and price
  sweeps (20, 21). Some of these predate the current dataset, so check dates
  before quoting them.
- `Results/N0100/` holds the scaled-community run (N = 100).

## Notes after the folder move

- `compare_energy_sharing_methods.py`, `cost_shapley.py` and
  `admm_bilateral_tolerance_sweep.py` write to
  `Day Ahead ADMM/plots/MES ADMM Optimisation Community Size 10/`, relative to
  the script. The existing results are in
  `Results/MES ADMM Optimisation Community Size 10/`, so either update
  `OUTPUT_DIR` or move the new files across after a run.
- There are two copies of `admm_bilateral_p2p_valley.py`. The one in the parent
  folder is used by the real-time study, and adds warm starts, per-solve timing
  and `ROLLING_FIRST_INTERVAL`. The copy here is the older version without them.
  Keep only one of them, so the day-ahead and real-time results come from the
  same algorithm.
- `price_fraction_sweep.py` and `price_fraction_sweep admm.py` differ only in
  formatting.
