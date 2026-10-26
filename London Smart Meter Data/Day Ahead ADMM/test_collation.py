"""Test script to demonstrate building, PV, and weather data collation."""

import pandas as pd
from pathlib import Path
import data_exploration as d

print("=" * 80)
print("DATA COLLATION TEST")
print("=" * 80)

# Load individual data sources with explicit handling
print("\n1. Loading PV Data...")
try:
    pv_data = d.load_pv_data()
    print(f"   [OK] Loaded {len(pv_data):,} hourly PV records")
    print(f"   [OK] Columns: {pv_data.columns.tolist()[:10]}...")
    print(f"   [OK] Date range: {pv_data['datetime'].min()} to {pv_data['datetime'].max()}")
except Exception as e:
    print(f"   [ERROR] {e}")
    pv_data = None

print("\n2. Loading Weather Data...")
try:
    weather_data = d.load_weather_data()
    print(f"   [OK] Loaded {len(weather_data):,} hourly weather records")
    print(f"   [OK] Columns: {weather_data.columns.tolist()[:10]}...")
    print(f"   [OK] Date range: {weather_data['DateTime'].min()} to {weather_data['DateTime'].max()}")
except Exception as e:
    print(f"   [ERROR] {e}")
    weather_data = None

print("\n3. Loading Device List...")
try:
    device_list = d.load_device_list()
    print(f"   [OK] Loaded {len(device_list):,} devices")
    print(f"   [OK] Columns: {device_list.columns.tolist()[:8]}...")
except Exception as e:
    print(f"   [ERROR] {e}")
    device_list = None

print("\n4. Loading Building Meter Data (ToU only)...")
try:
    building_data = d.load_tou_data()
    building_data = d.assign_tariff_prices(building_data)
    print(f"   [OK] Loaded {len(building_data):,} half-hourly meter records")
    print(f"   [OK] Buildings: {building_data['LCLid'].nunique():,}")
    print(f"   [OK] Date range: {building_data['DateTime'].min()} to {building_data['DateTime'].max()}")
except Exception as e:
    print(f"   [ERROR] {e}")
    building_data = None

# Demonstrate collation
if building_data is not None and pv_data is not None and weather_data is not None and device_list is not None:
    print("\n5. Collating Data...")
    try:
        collated = d.collate_building_pv_weather(
            building_data,
            pv_data=pv_data,
            weather_data=weather_data,
            device_list=device_list,
            aggregate_to_hourly=True,
        )
        print(f"   [OK] Collated shape: {collated.shape}")
        print(f"   [OK] Total columns: {len(collated.columns)}")
        print(f"   [OK] Columns: {collated.columns.tolist()[:15]}...")
        print("\n   Sample of collated data:")
        print(collated.head(3).to_string())
    except Exception as e:
        print(f"   [ERROR] during collation: {e}")
elif building_data is not None and pv_data is not None:
    print("\n5. Collating Available Data (Building + PV)...")
    try:
        # Collate building and PV without weather and device list
        collated = d.collate_building_pv_weather(
            building_data,
            pv_data=pv_data,
            weather_data=None,
            device_list=None,
            aggregate_to_hourly=True,
        )
        print(f"   [OK] Collated shape: {collated.shape}")
        print(f"   [OK] Columns: {collated.columns.tolist()[:15]}...")
    except Exception as e:
        print(f"   [ERROR] {e}")
else:
    print("\n5. Cannot perform collation - missing required data")

print("\n" + "=" * 80)
print("COLLATION FUNCTIONS AVAILABLE:")
print("=" * 80)
print("""
Functions added to data_exploration.py:

1. load_pv_data()
   - Loads hourly PV generation data
   - Source: PV Data/2014-11-28 Cleansed and Processed/EXPORT HourlyData/
   
2. load_weather_data()
   - Loads hourly weather data  
   - Source: PV Data/Weather Data 2014-11-30.xlsx
   
3. load_device_list()
   - Loads device metadata and PV sizes
   - Source: PV Data/deviceListTable...xlsx
   
4. collate_building_pv_weather(building_data, pv_data, weather_data, device_list)
   - Joins all three data sources by DateTime
   - Aggregates half-hourly building data to hourly resolution
   - Merges on Substation/location where available
   
Usage Example:
   collated = d.collate_building_pv_weather(
       building_data,
       aggregate_to_hourly=True
   )
""")
