"""What every output file in Results/ is: title, description and the script that writes it.

project_paths embeds these in each figure when it is saved, and tag_figures.py uses
them to tag existing figures and to write the index README of each results folder.
Add an entry here when a script starts writing a new kind of output.

Each entry is (filename regex, title, description, producing script).
"""

from __future__ import annotations

import re

_D = "02_day_ahead_sharing_methods"
_S = "03_day_ahead_sensitivity"
_R = "05_realtime_methods"

ENTRIES = [
    # ---------------------------------------------------------------- 01 data preparation
    (r"01_consumption_over_time\.", "Consumption over time",
     "Electricity consumption of the London smart-meter (LCL) households over the data period.",
     "common/plot_timeseries.py"),
    (r"02_pv_generation_over_time\.", "PV generation over time",
     "PV generation over time from the project's PV dataset (hourly export).",
     "common/plot_timeseries.py"),
    (r"03_pv_endpoints_gen_vs_import\.", "PV endpoints: generation vs import",
     "Generation against grid import for the PV customer endpoints.",
     "common/plot_timeseries.py"),
    (r"04_consumption_per_building\.", "Consumption per building",
     "Consumption profiles of a sample of individual LCL households.",
     "common/plot_timeseries.py"),
    (r"05_sharing_showcase\.", "Showcase-day selection",
     "Ranks candidate days by the benefit of pooling batteries (price water-filling, no solver); "
     "the best day becomes the showcase day.",
     "01_data_preparation/sharing_showcase.py"),
    (r"10_showcase_inputs\.", "Showcase-day inputs",
     "Hourly inputs of the showcase dataset: building loads, PV, time-of-use tariff, outdoor temperature "
     "and heating setpoint, plus the battery/PV assets.",
     "01_data_preparation/build_showcase_dataset.py"),
    # ---------------------------------------------------------------- 02 day-ahead sharing methods
    (r"06_showcase_optimisation\.", "Central optimisation summary",
     "Three-panel summary of the central day-ahead optimisation: no battery, own battery and shared (P2P).",
     f"{_D}/showcase_comparison.py"),
    (r"07_building_schedules_(?P<case>\w+)\.", "Building schedules ({case})",
     "Per-building day-ahead schedules (grid, battery, heat pump, boiler, temperature) for the {case} case.",
     f"{_D}/showcase_comparison.py"),
    (r"08_energy_share\.", "Building-to-building sharing",
     "Energy shared between each pair of buildings in the central shared optimisation.",
     f"{_D}/showcase_comparison.py"),
    (r"11_building_benefit\.", "Per-building benefit",
     "Per-building split (GBP) of the benefit of sharing over owning a battery.",
     f"{_D}/showcase_comparison.py"),
    (r"12_admm_convergence\.", "Exchange-ADMM convergence",
     "Convergence of exchange ADMM: residuals and the emergent P2P price per hour.",
     f"{_D}/admm_energy_sharing.py"),
    (r"13_sharing_savings\.", "Per-building sharing savings",
     "Per-building operating-cost saving: central market vs exchange ADMM (pool) vs bilateral ADMM, "
     "all settled at the same P2P price.",
     f"{_D}/showcase_comparison.py"),
    (r"14_admm_bilateral\.", "Bilateral-ADMM convergence",
     "Bilateral P2P ADMM: primal/dual residuals by iteration and the agreed pairwise trades.",
     "common/admm_bilateral_p2p_valley.py or 02_day_ahead_sharing_methods/admm_bilateral_p2p.py"),
    (r"15_(comfort_peak|temp_deviation)_scenarios\.", "Comfort and peak by scenario",
     "Per building: RMS temperature deviation from setpoint, peak grid import and its hour, for "
     "no battery / own battery / shared.",
     f"{_D}/showcase_comparison.py"),
    (r"16_(comfort_peak|temp_deviation)_methods\.", "Comfort and peak by method",
     "Per building: RMS temperature deviation, peak grid import and its hour, central vs the ADMM methods.",
     f"{_D}/showcase_comparison.py"),
    (r"17_(comfort_peak|temp_deviation)_total\.", "Community comfort and peak",
     "Community RMS temperature deviation and community peak grid import for every scenario and method.",
     f"{_D}/showcase_comparison.py"),
    (r"18_consumption_timeseries\.", "Community grid import over the day",
     "Community-average grid import (kW) over the day, one line per scenario and algorithm.",
     f"{_D}/showcase_comparison.py"),
    (r"19_cost_comparison\.", "Cumulative cost comparison",
     "Demand and cumulative community grid-electricity cost: no sharing vs central vs exchange ADMM vs "
     "bilateral ADMM.",
     f"{_D}/showcase_comparison.py"),
    (r"22_energy_sharing_method_comparison\.results\.json", "Method-comparison results cache",
     "Solved outputs of the method comparison, keyed by a signature of inputs and model sources; "
     "reused when nothing has changed.",
     f"{_D}/compare_energy_sharing_methods.py"),
    (r"22_energy_sharing_method_comparison\.", "Energy-sharing method comparison",
     "Main day-ahead comparison: no sharing, rule-based, central, exchange ADMM and bilateral ADMM with "
     "and without peak reduction (cost, peak import, comfort, energy traded).",
     f"{_D}/compare_energy_sharing_methods.py"),
    (r"22_bilateral_admm_valley_comparison\.", "Bilateral ADMM with and without peak reduction",
     "Bilateral ADMM against its peak-reduction (valley-filling) variant: grid import, cost and sharing.",
     f"{_D}/compare_energy_sharing_methods.py"),
    (r"22_sharing_surplus_comparison\.", "Available vs shared energy",
     "Surplus energy available for sharing against the energy actually shared, by method.",
     f"{_D}/compare_energy_sharing_methods.py"),
    (r"23_building_grid_import_heatmap\.", "Grid import heatmap",
     "Grid import per building and hour, by method.",
     f"{_D}/compare_energy_sharing_methods.py"),
    (r"23_energy_sources_per_building_(?P<case>\w+)\.", "Energy sources per building ({case})",
     "Where each building's electricity comes from (grid, PV, battery, P2P) in the {case} solution.",
     "common/admm_bilateral_p2p_valley.py / 02_day_ahead_sharing_methods/showcase_comparison.py"),
    (r"24_building_shared_energy_heatmap\.", "Shared energy heatmap",
     "Energy shared per building and hour, by method.",
     f"{_D}/compare_energy_sharing_methods.py"),
    (r"24_sharing_surplus_timeseries\.", "Sharing surplus over the day",
     "Surplus available for sharing and the energy shared, hour by hour, in the bilateral ADMM solution.",
     "common/admm_bilateral_p2p_valley.py"),
    (r"25_sharing_destinations_(?P<case>\w+)\.", "Sharing destinations ({case})",
     "Which buildings receive the energy each building shares ({case}).",
     "common/admm_bilateral_p2p_valley.py / 02_day_ahead_sharing_methods/compare_energy_sharing_methods.py"),
    (r"26_shared_energy_vs_(?P<what>grid|cost)_saving\.", "Shared energy vs {what} saving",
     "Per building: energy received through sharing against its {what} saving.",
     f"{_D}/compare_energy_sharing_methods.py"),
    (r"27_admm_bilateral_convergence_diagnostics\.", "Bilateral-ADMM convergence diagnostics",
     "Residual and objective diagnostics of the bilateral ADMM iterations.",
     "common/admm_bilateral_p2p_valley.py"),
    (r"27_building_timeseries_(?P<case>[\w+\-]+)\.", "Building time series ({case})",
     "Per-building electricity balance over the day for the {case} method.",
     f"{_D}/compare_energy_sharing_methods.py"),
    (r"28_building_method_comparison\.", "Per-building method comparison",
     "Building-level KPIs of every method against the battery-enabled no-sharing baseline.",
     f"{_D}/compare_energy_sharing_methods.py"),
    (r"29_building_cost_savings_heatmap\.", "Building cost-savings heatmap",
     "Signed per-building energy-cost saving of each method against no sharing.",
     f"{_D}/compare_energy_sharing_methods.py"),
    (r"30_building_cost_comparison\.", "Building cost comparison",
     "Per-building energy cost by method.",
     f"{_D}/compare_energy_sharing_methods.py"),
    (r"33_building_gas_import_heatmap\.", "Gas import heatmap",
     "Gas use per building and hour, by method.",
     f"{_D}/compare_energy_sharing_methods.py"),
    (r"computational_analysis\.csv", "Computational analysis",
     "Wall time, ADMM iterations, residuals, convergence, operating cost, peak and status per component.",
     f"{_D}/showcase_comparison.py"),
    # ---------------------------------------------------------------- 03 day-ahead sensitivity
    (r"09_trade_fee_sweep\.", "P2P trade-fee sweep",
     "Community and per-building cost of the central shared optimisation as the P2P trade fee varies.",
     "common/central_optimisation_showcase.py --sweep"),
    (r"20_admm_rho_sweep\.", "ADMM rho sweep",
     "Iterations to converge and cost accuracy of the ADMM methods against the penalty rho.",
     f"{_S}/admm_rho_sweep.py"),
    (r"21_price_fraction_sweep", "P2P price-fraction sweep",
     "The same dispatch re-settled at P2P prices of 0.1-1.0 x the grid tariff; checks every building "
     "still benefits.",
     f"{_S}/price_fraction_sweep.py"),
    (r"31_admm_tolerance_sweep_summary\.", "ADMM tolerance sweep",
     "Accuracy (vs central) and run time of bilateral ADMM against its stopping tolerance.",
     f"{_S}/admm_bilateral_tolerance_sweep.py"),
    (r"32_admm_tolerance_sweep_per_building\.", "ADMM tolerance sweep per building",
     "Per-building cost error of bilateral ADMM against central at each stopping tolerance.",
     f"{_S}/admm_bilateral_tolerance_sweep.py"),
    (r"34_shapley_coalition_costs\.json", "Shapley coalition-cost cache",
     "Cached central-optimisation cost of every coalition evaluated, keyed by an input/model signature.",
     f"{_S}/cost_shapley.py"),
    (r"34_cost_shapley_values", "Shapley allocation of the sharing saving",
     "Community saving allocated to buildings with Shapley values (permutation sampling or exact).",
     f"{_S}/cost_shapley.py"),
    (r"35_admm_tolerance_shapley", "Shapley values vs ADMM tolerance",
     "Shapley allocation computed with bilateral ADMM at several stopping tolerances.",
     f"{_S}/admm_tolerance_shapley.py"),
    # ---------------------------------------------------------------- 05 real-time methods
    (r"28_noisy_realtime_simulation\.", "Noisy real-time simulation (fixed schedule)",
     "The day-ahead ADMM plan executed against noisy load/PV with rule-based recourse; one row per run, "
     "building and interval.",
     f"{_R}/simulate_noisy_realtime.py"),
    (r"33_three_method_results\.", "Real-time results",
     "Per run, method, building and interval: realised dispatch, P2P trades, grid import and cost.",
     f"{_R}/compare_realtime_methods.py"),
    (r"34_three_method_solve_times\.", "Real-time solve times",
     "Solve time of every building solve, ADMM iteration and community step, with the estimated "
     "parallel latency and whether it fits the interval.",
     f"{_R}/compare_realtime_methods.py"),
    (r"35_three_method_summary\.", "Real-time summary per run",
     "Cost, grid import, deviation from the plan and solve time per method and run.",
     f"{_R}/compare_realtime_methods.py"),
    (r"36_three_method_summary_mean\.", "Real-time summary (mean over runs)",
     "Mean and spread over runs of the per-method summary.",
     f"{_R}/compare_realtime_methods.py"),
    (r"37_three_method_grid_import_run_(?P<run>\d+)\.", "Community grid import vs plan (run {run})",
     "Community grid import over the day for each real-time method against the day-ahead plan, run {run}.",
     f"{_R}/plot_realtime_comparison.py"),
    (r"38_three_method_dispatch_run_(?P<run>\d+)\.", "Real-time dispatch (run {run})",
     "Battery, P2P, curtailment and indoor temperature for each real-time method, run {run}.",
     f"{_R}/plot_realtime_comparison.py"),
    (r"39_three_method_costs\.", "Realised cost per method",
     "Realised community cost of each real-time method across the noise runs.",
     f"{_R}/plot_realtime_comparison.py"),
    (r"40_three_method_solve_times\.", "Real-time solve times",
     "Per-interval solve time of each method against the interval length.",
     f"{_R}/plot_realtime_comparison.py"),
    (r"41_admm_cold_start_results_run(?P<run>\d+)\.", "Cold-start rolling ADMM results (run {run})",
     "Rolling ADMM restarted from zero duals and trades every interval, run {run}.",
     f"{_R}/simulate_realtime_admm_cold.py"),
    (r"42_admm_cold_start_solve_times_run(?P<run>\d+)\.", "Cold-start rolling ADMM solve times (run {run})",
     "Solve times of cold-start rolling ADMM, run {run}.",
     f"{_R}/simulate_realtime_admm_cold.py"),
    (r"4[34]_admm_warm_vs_cold", "Warm vs cold-start rolling ADMM",
     "Iterations and time of warm- vs cold-start rolling ADMM (completed intervals; full day projected).",
     f"{_R}/simulate_realtime_admm_cold.py"),
    (r"45_admm_warm_vs_cold\.", "Warm vs cold-start rolling ADMM",
     "Iterations and solve time per interval, warm- vs cold-start rolling ADMM.",
     f"{_R}/simulate_realtime_admm_cold.py"),
    (r"46_forecast_vs_actual_inputs_run_(?P<run>\d+)\.", "Forecast vs actual inputs (run {run})",
     "Day-ahead forecast against the noisy actual load, PV and outdoor temperature, run {run}.",
     f"{_R}/plot_realtime_comparison.py"),
    (r"47_imbalance_charge_sensitivity\.", "Imbalance-charge sensitivity",
     "Cost against the plan as deviation from the plan is charged at increasing imbalance prices.",
     f"{_R}/plot_realtime_comparison.py"),
    (r"48_case_overview\.", "Noise-case overview",
     "All forecast-error cases side by side: cost and grid import of each method against the plan.",
     f"{_R}/plot_realtime_comparison.py --overview"),
    (r"49_per_building_grid_import\.", "Per-building grid import",
     "Daily grid import per building and its change against the plan, by method.",
     f"{_R}/plot_realtime_comparison.py"),
    (r"50_per_building_cost_heatmap\.", "Per-building cost heatmap",
     "Per-building cost against the plan for each method and forecast-error case.",
     f"{_R}/plot_realtime_comparison.py --overview"),
    (r"51_per_building_import_heatmap\.", "Per-building import heatmap",
     "Per-building grid import against the plan for each method and forecast-error case.",
     f"{_R}/plot_realtime_comparison.py --overview"),
    (r"bilateral_admm_schedule_parallel\.npz", "Day-ahead plan (parallel solve)",
     "Day-ahead bilateral ADMM plan from the parallel solver, kept to time the parallel solve when a "
     "sequential plan already exists.",
     f"{_R}/admm_parallel.py"),
    (r"bilateral_admm_schedule\.npz", "Day-ahead plan cache",
     "Cached day-ahead bilateral ADMM schedule, validated by a signature of the model sources, inputs "
     "and solver settings.",
     f"{_R}/simulate_noisy_realtime.py"),
    (r"shared_noise_scenarios\.npz", "Shared noise scenarios",
     "The noisy load, PV and outdoor-temperature scenarios every real-time method is run on.",
     f"{_R}/simulate_noisy_realtime.py"),
    (r"central_realtime_(simulation|solve_times)\.", "Rolling central (stand-alone run)",
     "Rolling central re-optimisation run on its own.",
     f"{_R}/simulate_realtime_central.py"),
    (r"reoptimized_realtime_(simulation|solve_times)\.", "Local / rolling ADMM (stand-alone run)",
     "Local re-optimisation or rolling ADMM run on its own.",
     f"{_R}/simulate_realtime_reoptimise.py"),
    (r"parallel_admm_realtime_(simulation|solve_times)\.", "Parallel rolling ADMM (stand-alone run)",
     "Rolling ADMM with the parallel gurobipy solver, run on its own.",
     f"{_R}/admm_parallel.py"),
    (r"parallel_local_realtime_(simulation|solve_times)\.", "Parallel local re-optimisation (stand-alone run)",
     "Local re-optimisation with the parallel gurobipy solver, run on its own.",
     f"{_R}/local_parallel.py"),
    (r"noise_scenarios_slide\.pptx", "Noise-scenario slide", "Slide describing the noise scenarios.", "manual"),
    # ---------------------------------------------------------------- 06 real-time scaling
    (r"60_scaling_solve_time\.", "Solve time vs community size",
     "Day-ahead solve time and mean real-time step time (wall and parallel) of each method against N, "
     "with the 1-hour and 1-minute limits; runs that stopped are marked.",
     "06_realtime_scaling/plot_scaling.py"),
    (r"61_scaling_admm_iterations\.", "ADMM iterations vs community size",
     "Iterations of the day-ahead and rolling ADMM against the community size.",
     "06_realtime_scaling/plot_scaling.py"),
    (r"62_size_vs_building_savings\.", "Community size vs per-building savings",
     "Per-building saving against open loop as the community grows, with Spearman correlation and "
     "permutation p-value per method.",
     "06_realtime_scaling/plot_scaling.py"),
    (r"scaling_building_savings\.csv", "Per-building savings by size",
     "Per-building saving against open loop for every community size and method.",
     "06_realtime_scaling/plot_scaling.py"),
    (r"scaling_status\.csv", "Scaling run status",
     "Status of every step and size (completed / budget_exceeded / killed_* / stopped_not_converged / "
     "skipped_no_plan), elapsed time, peak memory and intervals completed.",
     "06_realtime_scaling/scale_realtime_methods.py"),
    (r"scaling_summary\.csv", "Scaling summary",
     "Per-method summary for every community size, joined with the step status.",
     "06_realtime_scaling/scale_realtime_methods.py"),
    (r"run_log\.txt", "Run log", "Console output of every step run for this size.",
     "06_realtime_scaling/scale_realtime_methods.py"),
    # ---------------------------------------------------------------- 07 communication failure
    (r"70_comm_failure_masks\.npz", "Communication-failure patterns",
     "Which buildings are offline in every hour, for each failure level and noise run, with the outage "
     "length and seed.",
     "07_communication_failure/simulate_communication_failure.py"),
    (r"71_comm_failure_results\.csv", "Communication-failure results",
     "Real-time dispatch per failure level, run, method, interval and building, with an offline flag.",
     "07_communication_failure/simulate_communication_failure.py"),
    (r"72_comm_failure_solve_times\.csv", "Communication-failure solve times",
     "Per-solve timings of the real-time methods under communication failures.",
     "07_communication_failure/simulate_communication_failure.py"),
    (r"73_comm_failure_summary\.csv", "Communication-failure summary",
     "Totals per failure level, run and method, with the change against the same method without failures.",
     "07_communication_failure/simulate_communication_failure.py"),
    (r"74_comm_failure_summary_mean\.csv", "Communication-failure summary (mean)",
     "Mean and std over runs of the communication-failure summary, per failure level and method.",
     "07_communication_failure/simulate_communication_failure.py"),
    (r"75_comm_failure_robustness\.", "Robustness to communication failures",
     "Grouped bars per failure level: cost against rolling central without failures, cost increase against "
     "no failure, and the change in P2P energy delivered and peak grid import, of each real-time method, "
     "for one test (buildings offline, link failure probability or "
     "coordinator down hours).",
     "07_communication_failure/simulate_communication_failure.py"),
    (r"76_comm_failure_overview\.", "Communication failures: all tests",
     "Cost increase against no failure of each real-time method against the failure level, one panel per "
     "communication-failure test.",
     "07_communication_failure/simulate_communication_failure.py"),
    (r"78_comm_failure_hourly\.", "Communication failures hour by hour",
     "Tariff, lost connections per hour and the community cost increase against no failure, hour by hour, "
     "for each real-time method at one failure level of a test.",
     "07_communication_failure/plot_communication_failure.py"),
    (r"79_comm_failure_who_pays\.", "Who pays for communication failures",
     "Cost increase against no failure on the buildings while cut off (or with a lost link) and on the "
     "others, per method and failure level; P2P payments included.",
     "07_communication_failure/plot_communication_failure.py"),
    (r"80_comm_failure_building_heatmap\.", "Per-building cost of communication failures",
     "Daily cost increase against no failure per building and re-optimising method at one failure level, "
     "with each building's assets and hours hit.",
     "07_communication_failure/plot_communication_failure.py"),
    (r"81_comm_failure_solve_effort\.", "Solve effort under communication failures",
     "Rolling ADMM iterations against the peer-to-peer links lost in that hour, and the real-time step "
     "latency of each method per failure level against the 1-minute and 1-hour lines.",
     "07_communication_failure/plot_communication_failure.py"),
    (r"82_comm_failure_test_heatmap\.", "Communication failures: method x test",
     "Daily cost increase against no failure for every test and method at the test's highest failure "
     "level; the cheapest re-optimising method per test in bold.",
     "07_communication_failure/plot_communication_failure.py"),
    (r"83_comm_failure_scenario_run_(?P<run>\d+)\.", "Communication failures, scenario {run}",
     "Noise scenario {run} of one test: which building lost which connections hour by hour, the hourly "
     "community cost increase and the change in stored battery energy against no failure per method, and "
     "the daily cost increase of the scenario at every failure level.",
     "07_communication_failure/plot_communication_failure.py"),
    (r"84_comm_failure_vs_admm\.", "Every method against rolling ADMM",
     "Daily cost of each real-time method minus rolling ADMM's on the same scenario and failure level, "
     "against the level (mean and every scenario) and hour by hour at the highest level.",
     "07_communication_failure/plot_communication_failure.py"),
    (r"85_comm_failure_vs_admm_heatmap\.", "Against rolling ADMM: method x test",
     "Each method's daily cost minus rolling ADMM's at every test's highest finished level, with the "
     "number of scenarios where ADMM is cheaper, tied (within GBP 0.01) or dearer.",
     "07_communication_failure/plot_communication_failure.py"),
    (r"90_impact_summary\.", "Impact of communication failures",
     "Daily impact of the communication failures (cost with failures minus without, same method and "
     "scenario) per method and failure level, and central's impact minus rolling ADMM's.",
     "07_communication_failure/plot_communication_failure.py"),
    (r"91_impact_timeseries_level_(?P<level>[\d.]+)_run_(?P<run>\d+)\.",
     "Communication failures over the day (level {level}, scenario {run})",
     "When each building is offline (building x hour), and the hourly and cumulative impact of the "
     "failures for rolling ADMM and rolling central, failure level {level}, noise scenario {run}.",
     "07_communication_failure/plot_communication_failure.py"),
    (r"92_impact_overview\.", "Impact of communication failures: all tests",
     "Daily impact of the communication failures at every test's highest level, per method, with "
     "central minus rolling ADMM.",
     "07_communication_failure/plot_communication_failure.py"),
    (r"93_computation_time\.", "Computation time under communication failures",
     "Computation time per real-time hour of rolling ADMM (sequential and parallel estimate) and rolling "
     "central for every failure level of a test, and the ADMM iterations per hour.",
     "07_communication_failure/plot_communication_failure.py"),
    (r"94_computation_time_overview\.", "Computation time: all tests",
     "Computation time per real-time hour of rolling ADMM (parallel estimate) and rolling central, no failure "
     "against each test's highest failure level.",
     "07_communication_failure/plot_communication_failure.py"),
    # ---------------------------------------------------------------- 08 minute resolution
    (r"80_minute_inputs\.", "Hourly inputs as minute data",
     "Hourly load, PV, outdoor temperature and tariff against the 1-minute data made from them.",
     "08_minute_resolution/admm_minute_timing.py"),
    (r"81_admm_minute_timing\.csv", "ADMM on minute data: timings",
     "Per start hour and ADMM start (cold/warm): iterations, ADMM parallel time, central time and the cost "
     "gap, for one hour of 1-minute data.",
     "08_minute_resolution/admm_minute_timing.py"),
    (r"82_admm_minute_timing\.", "ADMM on minute data: time to optimise an hour",
     "Parallel-estimate time and iterations of bilateral ADMM (cold and warm start) to optimise one hour of "
     "1-minute data, against a central solve and the 1-minute and 1-hour limits.",
     "08_minute_resolution/admm_minute_timing.py"),
    (r"83_admm_minute_convergence\.", "ADMM on minute data: convergence",
     "ADMM primal and dual RMS residuals against the parallel-estimate elapsed time, cold and warm start, "
     "for one hour of 1-minute data, with the stopping tolerance and the central solve time.",
     "08_minute_resolution/admm_minute_timing.py"),
    (r"84_minute_dispatch\.", "The optimised hour at 1-minute resolution",
     "Community grid import, battery discharge minus charge, P2P traded and mean indoor temperature per "
     "minute of the optimised hour, ADMM (cold and warm) against central.",
     "08_minute_resolution/admm_minute_timing.py"),
    (r"85_three_timescale_timing\.csv", "Three-timescale ADMM: solve times",
     "One row per day-ahead, hourly and minute ADMM solve: hour, minute, horizon, iterations, parallel time.",
     "08_minute_resolution/three_timescale.py"),
    (r"86_three_timescale_dispatch\.csv", "Three-timescale ADMM: minute dispatch",
     "Per minute: grid import, battery, P2P and indoor temperature applied by the minute level, the hourly "
     "and day-ahead plans for that hour, and the departure from the hourly plan.",
     "08_minute_resolution/three_timescale.py"),
    (r"87_three_timescale_timing\.", "Three-timescale ADMM: computation time",
     "Parallel-estimate time of every day-ahead, hourly and minute ADMM solve through the day, against the "
     "1-minute and 1-hour lines, with per-level distributions.",
     "08_minute_resolution/three_timescale.py"),
    (r"88_three_timescale_dispatch\.", "Three-timescale ADMM: minute level tracking the hourly plan",
     "Grid import, battery power, P2P traded, departure from the hourly plan and indoor temperature applied "
     "by the minute level, against the hourly and day-ahead plans.",
     "08_minute_resolution/three_timescale.py"),
    (r"89_traded_energy_limits.*\.csv", "Traded energy against the limits (6a)-(6c)",
     "Per minute and building of the three-timescale run: role (producer/consumer), traded energy, its change, "
     "the ramp limit, cumulative purchases and their allocation.",
     "09_ramp_rate_control/three_timescale_ramp_limits.py"),
    (r"89_traded_energy_limits.*\.png", "Traded energy against the ramp and cumulative limits",
     "Producers' largest change in P2P sales relative to the ramp limit (6a)-(6b) per minute, and consumers' "
     "cumulative purchases against their allocation (6c).",
     "09_ramp_rate_control/three_timescale_ramp_limits.py"),
    (r"90_ramp_limits_comparison\.", "With and without the limits (6a)-(6c)",
     "Three-timescale ADMM with and without the ramp and cumulative limits in the minute-level objective: "
     "minutes over the limits, largest ramp, end-of-hour excess, electricity cost and minute solve time.",
     "archive/09_ramp_rate_control/plot_comparison_p2p_limits.py"),
    (r"92_cost_vs_ramp_weight\.", "Electricity cost against the ramp penalty weight",
     "Daily electricity cost of the three-timescale ADMM run against the ramp penalty weight "
     "(0 = without the limits), with the change against no limits.",
     "archive/09_ramp_rate_control/plot_comparison_p2p_limits.py"),
    (r"93_p2p_vs_ramp_weight\.", "Energy sharing against the ramp penalty weight",
     "Daily P2P traded energy of the three-timescale ADMM run against the ramp penalty weight "
     "(0 = without the limits), with the change against no limits.",
     "archive/09_ramp_rate_control/plot_comparison_p2p_limits.py"),
    (r"94_ramp_rate_sweep\.", "Tighter ramp limits: with and without the limits",
     "For ramp limits of 10, 2 and 1 %/min: producer-minutes over the limit, largest ramp (kW/min), "
     "end-of-hour (6c) excess, electricity cost, P2P traded energy and grid import, without and with the limit.",
     "09_ramp_rate_control/plot_sweep.py"),
    (r"95_ramp_rate_sweep_clouds\.", "Tighter ramp limits under passing clouds",
     "As 94_ramp_rate_sweep, on the day with passing clouds in the minute PV (10 and 2 %/min).",
     "09_ramp_rate_control/plot_sweep.py"),
    (r"96_grid_ramp_sweep_", "Ramp limit on grid imports: with and without",
     "For ramp limits of 10, 2 and 1 %/min on each building's grid import (plan-deviation weight 1): "
     "building-minutes over the limit, largest building ramp, community grid ramp (largest and 99th "
     "percentile), electricity cost, P2P traded energy and grid import, without and with the limit.",
     "09_ramp_rate_control/plot_sweep.py"),
    (r"97_grid_ramp_sweep_clouds", "Ramp limit on grid imports under passing clouds",
     "As 96_grid_ramp_sweep, on the day with passing clouds in the minute PV.",
     "09_ramp_rate_control/plot_sweep.py"),
    (r"91_grid_ramp_timeseries", "Buildings' grid-import ramp rate over the day",
     "Each building's minute-to-minute change in grid import (kW/min) against its own limit, one panel per "
     "run (without the limit, 10, 2 and 1 %/min); _clouds with passing clouds.",
     "09_ramp_rate_control/plot_ramp_timeseries.py"),
    (r"98_penalty_sweep", "Plan-deviation and ramp penalty sweep",
     "Grid-import ramp limit at 10 %/min: electricity cost, P2P traded energy, building-minutes over the "
     "limit and the community's largest grid ramp, for plan-deviation weights 0-100 and ramp weights "
     "0, 100, 1000, 10000.",
     "09_ramp_rate_control/plot_penalty_sweep.py"),
    (r"99_penalty_correlations", "Penalty sweep: correlations and trade-offs",
     "Spearman rank correlation between the plan-deviation and ramp weights and the outcomes (cost, P2P "
     "shared, minutes over the ramp limit, community largest grid ramp), and cost against each outcome, "
     "one point per run (colour: plan weight, size: ramp weight).",
     "09_ramp_rate_control/plot_penalty_correlations.py"),
    (r"100_penalty_type", "Proportional, squared or both ramp penalties",
     "Grid-import ramp limit at 10 %/min, plan-deviation weight 1: no limit, squared penalty only, "
     "proportional (linear) penalty only and both; one figure per measure (minutes over the limit, total "
     "and largest excess, community largest ramp, cost, P2P shared, minute solve time).",
     "09_ramp_rate_control/plot_penalty_type.py"),
    (r"91_ramp_rate_timeseries", "Producers' minute ramp rate over the day",
     "Each producer's minute-to-minute change in P2P sales power (kW/min) against its own limit (6a)-(6b) "
     "(ramp rate x rated power), one panel per run: ramp weights (no suffix), ramp rates (_rates) or "
     "with passing clouds (_clouds).",
     "09_ramp_rate_control/plot_ramp_timeseries.py"),
    (r"77_comm_failure_overview\.csv", "Communication failures: all tests (data)",
     "Mean and std of the cost increase against no failure per test, level and method.",
     "07_communication_failure/simulate_communication_failure.py"),
]

_COMPILED = [(re.compile(pattern), title, description, script) for pattern, title, description, script in ENTRIES]


def lookup(filename: str) -> dict:
    """Title, description and script for an output file name ({} if unknown)."""
    for pattern, title, description, script in _COMPILED:
        match = pattern.match(filename)
        if match:
            fields = {k: v.replace("_", " ") for k, v in match.groupdict().items() if v}
            return {"title": title.format(**fields), "description": description.format(**fields), "script": script}
    return {}
