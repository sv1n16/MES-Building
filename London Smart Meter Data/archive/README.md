# Archive

Superseded files, kept for reference. This folder is **not** on the import path, so
nothing here is used by the current scripts.

| File | What it was | Replaced by |
|---|---|---|
| `admm_bilateral_p2p_valley_dayahead_2026-10.py` | The `Day Ahead ADMM/` copy of the bilateral ADMM, with tolerance 1e-5, at most 1000 iterations, no warm start. The October day-ahead method comparison was run with it | `common/admm_bilateral_p2p_valley.py`, now the only copy |
| `simulate_noisy_realtime_2026-10.py` | The `Real-Time ADMM/` copy of the noisy simulation, without `pv_bias` | `05_realtime_methods/simulate_noisy_realtime.py` |
| `price_fraction_sweep_admm_duplicate.py` | `price_fraction_sweep admm.py`, a reformatted duplicate of the price sweep | `03_day_ahead_sensitivity/price_fraction_sweep.py` |
| `README_day_ahead_admm_2026-10.md`, `README_realtime_admm_2026-10.md` | READMEs of the old `Day Ahead ADMM/` and `Real-Time ADMM/` folders | The README in each experiment folder |

Superseded results are in `Results/archive/`:

- `older_day_ahead_*`: the older copy of each day-ahead figure, from where the top-level
  `Results/` folder and `MES ADMM Optimisation Community Size 10/` held both. The newer
  copy is in the experiment folder.
- `stray_test_run_N0010_2026-11-14`: a test run that wrote to an old path.
