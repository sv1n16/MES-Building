# Shared code

Modules used by more than one experiment. Experiment scripts import them by name, for
example `import central_optimisation_showcase as C`.

| Module | What it holds | Run directly |
|---|---|---|
| `central_optimisation_showcase.py` | Loads the showcase dataset (`data/`, or `SHOWCASE_DATASET`), the building parameters and the central community MIQCP (`build_model`, `solve_and_extract`) | `--sweep`: P2P trade-fee sweep, figure 09, written to `Results/03_day_ahead_sensitivity/` |
| `admm_bilateral_p2p_valley.py` | **The** bilateral P2P ADMM used by both the day-ahead and the real-time studies. It has per-building QP subproblems (`build_subproblem`), consensus trades, RMS residual stopping (ε = 1e-4, at most 500 iterations, ρ = 1.1), the optional peak-reduction (congestion) term, warm starts and `ROLLING_FIRST_INTERVAL` for receding-horizon use | figures 14, 23–25, 27 into `Results/02_day_ahead_sharing_methods/` |
| `data_exploration.py` | Loaders for LCL meter data, the tariff schedule, PV and weather data | — |
| `plot_timeseries.py` | Shared plot colours (`INK`, `MUTED`, …), `PLOTS_DIR`, and the raw-data figures 01–04 | figures 01–04 into `Results/01_data_preparation/` |
| `project_paths.py` | Project folders (`ROOT`, `DATA_DIR`, `RESULTS_DIR`, …), `register()`, `results_dir()`, and the hooks that tag every saved figure with metadata | — |
| `figure_catalogue.py` | Title, description and producing script for every kind of output file | — |
| `tag_figures.py` | Tags figures retrospectively, shows a figure's metadata, writes the index README in each results folder | `--show <figure>`, `--force`, `--index-only` |

## Output routing

`project_paths.register(__file__)` at the top of each script records the experiment of
the script that was run. The first call wins, so importing a module from another
experiment does not change it. Outputs then go to:

```
Results/<experiment>/[<SHOWCASE_PLOTS_SUBDIR>/]
```

`MES_EXPERIMENT` overrides the experiment. `04_day_ahead_scaling/run_scaling.py` uses it
to send the day-ahead scripts it launches into `Results/04_day_ahead_scaling/N<NNNN>/`.
`06_realtime_scaling/scale_realtime_methods.py` gets the same effect by running its
workers from its own folder with `SHOWCASE_PLOTS_SUBDIR=N<NNNN>`.

Environment variables read by `central_optimisation_showcase.py`:

| Variable | Effect |
|---|---|
| `SHOWCASE_DATASET` | Dataset stem under `data/` (default `showcase_<day>`, N = 10) |
| `SHOWCASE_PLOTS_SUBDIR` | Subfolder of the experiment's results folder |
| `SHOWCASE_SOLVE_TIMELIMIT` | Gurobi time limit per solve, in seconds (0 = none) |
| `SHOWCASE_RELAX_CENTRAL_BINARIES` | Relax the central charging binary to [0, 1] |

## Caches depend on these files

The day-ahead plan cache (`bilateral_admm_schedule.npz`), the Shapley coalition cache and
the day-ahead method-comparison cache are validated against a hash of the source of
`central_optimisation_showcase.py` and `admm_bilateral_p2p_valley.py`. Any edit to these
two files, comments included, makes the cached plans be re-solved. The N = 100 plan takes
minutes in parallel and hours sequentially, so batch edits to these files.
