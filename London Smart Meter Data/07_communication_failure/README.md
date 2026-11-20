# 07 — Robustness to communication failures (N = 10)

The real-time comparison of [`05_realtime_methods`](../05_realtime_methods/README.md) is
re-run while communication fails. Every method sees the same failure pattern and the
same noise scenarios as experiment 05, so the comparison is paired. Failure level 0 is
the no-failure reference and reproduces the 05 results exactly.

## Tests

Each test writes to `Results/07_communication_failure/<test>/`.

| Test | Failure | Levels | What it probes |
|---|---|---|---|
| `building` | k random buildings offline every hour, 1-hour outages | k = 0, 1, 2, 3 | Base robustness |
| `building_long_outage` | As `building`, outages last 4 hours | k = 0, 1, 2 | Central's fallback plan goes stale |
| `building_late_detection` | As `building`, failure noticed mid-negotiation | k = 0, 1, 2, 3 | No warning before the plan is under way |
| `building_correlated` | As `building`, on the 05 case `correlated` (80% of the error common to all buildings, no bias) | k = 0, 1, 2, 3 | Errors that do not cancel out across buildings |
| `link` | Every communication link fails with probability p per hour | p = 0, 0.05, 0.1, 0.2 | Peer-to-peer mesh against star |
| `coordinator` | The central coordinator is down for h consecutive hours (run first) | h = 0, 1, 3, 6, 12, 24 | Single point of failure |

## Failure model

**Topology.** Local re-optimisation and ADMM exchange messages peer to peer, so a
failure removes pairs of buildings: a building failure removes all its pairs, a link
failure removes one pair. Rolling central is a star: a building talks only to the
coordinator. A building or link failure cuts that building off, and a coordinator
failure cuts off every building. Open loop and the fixed schedule use no real-time
messages: the plan and its trades were exchanged the day before, and the recourse uses
local measurements only. They are unaffected by every test.

**Response ("expect reconnection").** No P2P trade can be agreed over a lost connection
in that hour, so its trade is zero in the current hour. Nobody knows how long an outage
will last. For the later hours of the horizon, everyone keeps planning with the last
agreed trades and prices over that connection, as if it will be back next hour.

| Method | Buildings still connected | Cut-off building |
|---|---|---|
| `local_reoptimisation` | Re-optimise alone; the current hour's trades over lost connections are cancelled, nothing is re-routed; later hours keep their planned trade limits | Re-optimises alone, as above |
| `rolling_admm` | Bilateral ADMM over the connected pairs re-routes the current hour's sharing. Over a lost pair, both ends keep the last agreed trade z and their last dual λ for the later hours (asynchronous ADMM: no z or λ update) | Re-optimises with its own controller on those stale prices and agreements |
| `rolling_admm_fallback` | As `rolling_admm`, with an iteration budget (`--admm-fallback-iters`, default 20, about one central solve). Without agreement in time, every building re-optimises locally on the last agreement everyone reached (lost connections: zero this hour); trades settle at min(offer, acceptance). The next hour warm-starts from where the capped ADMM stopped | As the connected buildings |
| `rolling_central` | Solved centrally | No optimiser and no new set-points: executes its last received central plan (day-ahead if never reached) with the fixed-schedule recourse. The coordinator fixes the building's current hour to those set-points, starts it from the state its last plan predicted, and plans its later hours |
| `rolling_central_local_fallback` | Solved centrally | Re-optimises itself the way local re-optimisation does, with the trades of its last central plan as limits for the later hours. Separates local intelligence from decentralised coordination |

The first version assumed a lost connection was lost for the rest of the day. Every
cut-off building then stopped charging its battery for its neighbours, which bought the
evening peak from the grid: about £1/day with one building offline per hour, for ADMM
and for central with the local fallback. Those partial results are kept in
`Results/07_communication_failure/_superseded_permanent_failure_model/`.

**Detection.** By default, failures are known at the start of the hour (a missed
heartbeat), and every solve fixes the current-hour trades over lost connections to zero. In
`building_late_detection` they are noticed only once the plan is under way:

- **ADMM:** the lost pairs take part in the first `--drop-iteration` iterations (default 1).
  Their messages then stop: the current-hour trade is fixed to zero, the later hours keep
  the z and λ of that moment, and the others carry on iterating.
- **Local:** every building solves assuming its planned trades. Lost trades are not
  delivered, so the buyer imports the shortfall and the seller curtails.
- **Central:** the coordinator plans with every building, but the cut-off building never
  gets its set-points. Its partners' trades with it are not delivered.

**Implementation.** The 05 simulations run unchanged on top of hooks:
- **Local re-optimisation:** a wrapper around `build_subproblem` fixes the current-hour trades over lost connections.
- **ADMM:** `run_admm_with_failures` replaces `run_admm` for intervals with a lost connection. It mirrors `run_admm`, holds z and λ of the lost pairs, and stops on the residuals of the connected pairs only.
- **Rolling central:** `simulate_central` re-implements the rolling central loop with the cut-off handling.
- **Late detection, local:** a hook into the trade reconciliation leaves the lost trades undelivered.

With no lost connection, every method calls the 05 code path, so level 0 reproduces 05 exactly.

**Notes.**
- Rolling ADMM needs many more iterations while connections are lost: 24–232 per hour
  instead of 2 when warm-started. This adds about 15 minutes per run and level at N = 10.
  The whole suite with 3 runs takes about a day sequentially. Run it in the background, or
  test by test.

## Running

```powershell
$py = "C:\conda_envs\myenv\python.exe"
& $py 07_communication_failure\simulate_communication_failure.py --test building --runs 3
& $py 07_communication_failure\simulate_communication_failure.py --test all --runs 3
& $py 07_communication_failure\simulate_communication_failure.py --test link coordinator --runs 1
& $py 07_communication_failure\simulate_communication_failure.py --summarise-only   # all tests + overview
```

The day-ahead plan and the noise scenarios come from `Results/05_realtime_methods/`
(and `cases/<case>/`); `--plan-dir` changes the folder. Nothing is written there. If a
case has no saved scenarios, they are drawn into the test folder with the 05 case settings.

## Outputs

Per test, in `Results/07_communication_failure/<test>/`:

| File | Content |
|---|---|
| `70_comm_failure_masks.npz` | `lost_p2p_level_<x>[run, b, j, t]`, `central_offline_level_<x>[run, b, t]`, and the test settings |
| `71_comm_failure_results.csv` | One row per level, run, method, interval and building, with `lost_links` and `building_offline` |
| `72_comm_failure_solve_times.csv` | Timing rows as in 05, with `offline_buildings` |
| `73_comm_failure_summary.csv` | Per level, run and method: the 05 summary columns, costs on offline/online building-hours, and the change against the same method and scenario without failures |
| `74_comm_failure_summary_mean.csv` | Mean and std over runs |
| `75_comm_failure_robustness.png` | One column of panels per method, bars per failure level (mean ± std over runs): cost against rolling central without failures, cost increase against no failure, change in P2P delivered and in peak import |

Across tests, in `Results/07_communication_failure/`: `76_comm_failure_overview.png` (one panel per
method: cost increase at each test's highest level, one bar per test) and `77_comm_failure_overview.csv`.

## Analysis: what the failures cost ADMM and central

`plot_communication_failure.py` shows only the impact of the communication failures: a
method's cost with failures minus the same method's cost on the same noise scenario
without failures. It answers three questions, one folder each, in
`Results/07_communication_failure/`:

| Folder | Question | Methods | Tests |
|---|---|---|---|
| `1_no_fallback/` | Is ADMM more robust than central when central has no fallback? | Rolling ADMM; rolling central, no fallback (cut-off building runs uncontrolled) | Building and link failures |
| `2_with_fallback/` | Does a fallback close the gap, and what does an iteration budget cost ADMM? | Rolling ADMM; ADMM + iteration-budget fallback; rolling central + last-plan fallback; rolling central + local fallback | Building and link failures |
| `3_central_entity_failure/` | What if the central coordinator itself fails? | Rolling ADMM (no coordinator); rolling central without and with each fallback | `coordinator` |

In each folder:

| File | Content |
|---|---|
| `<test>/90_impact_summary.png` | Daily impact (electricity + gas, with failures − without) per failure level, grouped bars, one per method; mean ± std over scenarios |
| `<test>/90_impact_summary_minus_rolling_admm.png`, `_minus_admm_fallback.png` | Each other method's impact minus rolling ADMM's, and (fallback folder) minus ADMM + fallback's: + = the reference is more robust |
| `<test>/91_impact_timeseries_level_<x>_run_<n>.png` | One per failure level and scenario: when each building is offline (building × hour), the hourly impact and the cumulative impact over the day |
| `92_impact_overview.png`, `92_impact_overview_minus_*.png` (and `.csv`) | The same at every test's highest level |

**Computation time (`4_computation_time/`).** Computation time per real-time hour, hours
1–23 of every scenario, for rolling ADMM and ADMM + fallback (parallel estimate, one
solver per building) and each central variant (wall time; the local fallback includes
the cut-off buildings' own solves):

| File | Content |
|---|---|
| `<test>/93_computation_time.png` | Per failure level, one box per method. Log scale, with the 1-minute and 1-hour lines |
| `<test>/93_admm_iterations.png` | ADMM iterations per hour, with and without the iteration budget |
| `95_computation_time_table.csv` | Per test and level: ADMM iterations, every method's median and max time per hour |
| `94_computation_time_overview.png` (and `.csv`) | Every method at each test's highest failure level |

Timings measured while other jobs shared the CPU are inflated; compare them within a test.

**Central without a fallback.** A building the coordinator cannot reach gets no set-points
and runs uncontrolled for that hour. Its battery is idle and it makes no P2P trades. Its
basic thermostat brings the indoor temperature back to the setpoint, from the heat pump
first and the boiler for the rest, and the grid supplies the remaining electricity. The
coordinator still optimises every building it can reach.

Only (level, scenario) pairs that every method in a folder has finished are used, so the
methods are always compared on the same scenarios. The script can run while the suite
is still going.

```powershell
& $py 07_communication_failure\plot_communication_failure.py
& $py 07_communication_failure\plot_communication_failure.py --test coordinator --folder 3_central_entity_failure
```
