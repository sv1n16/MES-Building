"""Quick look at the raw PV and weather spreadsheets (first rows and columns).

PURPOSE
    Sanity-check the input files before building datasets. No optimisation, no outputs;
    prints the columns, shape and first row of
      PV Data/Weather Data 2014-11-30.xlsx
      PV Data/deviceListTable ... customer addresses removed.xlsx
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import project_paths  # noqa: E402  (shared modules on sys.path; results -> Results/<experiment>/)

project_paths.register(__file__)

import pandas as pd

# Check weather data
weather_df = pd.read_excel("PV Data/Weather Data 2014-11-30.xlsx", sheet_name=0, nrows=3)
print("Weather Data:")
print("Columns:", weather_df.columns.tolist())
print("Shape:", weather_df.shape)
print("First row:")
print(weather_df.iloc[0])

# Check device list for building IDs
device_df = pd.read_excel(
    "PV Data/deviceListTable with explanatory notes v2 - customer addresses removed.xlsx", nrows=3
)
print("\n\nDevice List:")
print("Columns:", device_df.columns.tolist())
print("Shape:", device_df.shape)
print("First row:")
print(device_df.iloc[0])
