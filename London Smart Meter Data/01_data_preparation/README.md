# 01 — Data preparation

Turns the raw London smart-meter (LCL) and PV data into the hourly model inputs that
every other experiment uses.

## Data

- **Loads:** LCL households on the time-of-use (ToU) tariff (`Small LCL Data/`), with
  half-hourly readings aggregated to hourly mean kW. Only ~1,100 ToU meters exist. On a
  given day a household is used only if all 48 half-hours are present; many households
  have a duplicated 49th reading, so distinct half-hours are counted.
- **Tariff:** the LCL ToU schedule (`Tariffs.xlsx`). Low 3.99, Normal/Medium 11.76 and
  High 67.20 p/kWh.
- **PV:** real February per-kWp profile from the UK Power Networks LV monitoring data in
  `PV Data/`. Each PV building is sized at 3.5 kWp.
- **Heating and weather:** the LCL data has neither, so the outdoor temperature (cold
  February diurnal) and setpoint (18 °C base, 21 °C comfort windows) are synthetic.

## Scripts

| Script | Purpose | Outputs |
|---|---|---|
| `../common/plot_timeseries.py` | Raw-data figures | `01_consumption_over_time`, `02_pv_generation_over_time`, `03_pv_endpoints_gen_vs_import`, `04_consumption_per_building` |
| `sharing_showcase.py` | Ranks days by the benefit of pooling batteries (price water-filling, no solver) and picks the showcase day (2013-02-21) | `05_sharing_showcase.png` |
| `build_showcase_dataset.py` | Builds the hourly 24-step input and the battery/PV asset table for an N-building community. The diversity filters relax automatically to reach N | `data/showcase_<day>[_N<NNNN>].csv`, `..._batteries.csv`, `10_showcase_inputs.{png,html}` |
| `inspect_data.py`, `test_collation.py` | Quick checks of the PV, weather and device data | console only |

Figures go to `Results/01_data_preparation/`. When a scaling runner builds a dataset, its
figure 10 goes to that runner's `N<NNNN>/` folder instead.

## Datasets in `data/`

| File | Community |
|---|---|
| `showcase_2013-02-21.csv` (+ `_batteries.csv`) | Default, N = 10. Assets: 3 battery + PV, 2 battery only, 2 PV only, 3 none |
| `showcase_2013-02-21_N0005` … `_N1000` | N = 5, 20, 50, 100, 1000, for the scaling studies |
| `showcase_latest.txt` | The showcase day used by default |

## Running

```powershell
C:\conda_envs\myenv\python.exe common\plot_timeseries.py
C:\conda_envs\myenv\python.exe 01_data_preparation\sharing_showcase.py
C:\conda_envs\myenv\python.exe 01_data_preparation\build_showcase_dataset.py --n 50 --day 2013-02-21
```
