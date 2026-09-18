from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_FOLDER = Path(__file__).parent / "Small LCL Data"
TARIFF_FILE = DATA_FOLDER / "Tariffs.xlsx"
TARIFF_PRICE_MAP = {"Low": 3.99, "Medium": 11.76, "High": 67.20, "Normal": 11.76}


def load_tariff_schedule(tariff_file: Path = TARIFF_FILE) -> pd.DataFrame:
    if not tariff_file.exists():
        raise FileNotFoundError(f"Tariff file not found: {tariff_file}")

    tariff_df = pd.read_excel(tariff_file, sheet_name="Sheet1")
    tariff_df.columns = tariff_df.columns.str.strip()
    tariff_df = tariff_df.rename(columns={"TariffDateTime": "DateTime", "Tariff": "TariffLabel"})
    tariff_df["DateTime"] = pd.to_datetime(tariff_df["DateTime"]).dt.floor("30min")
    tariff_df["TariffLabel"] = tariff_df["TariffLabel"].astype(str).str.strip().str.title()
    tariff_df["TariffLabel"] = tariff_df["TariffLabel"].replace("Normal", "Medium")
    return tariff_df[["DateTime", "TariffLabel"]]


def assign_tariff_prices(data: pd.DataFrame, tariff_schedule: pd.DataFrame | None = None) -> pd.DataFrame:
    if tariff_schedule is None:
        tariff_schedule = load_tariff_schedule()

    tariff_data = data.copy()
    tariff_data["DateTime"] = pd.to_datetime(tariff_data["DateTime"]).dt.floor("30min")

    tariff_lookup = tariff_schedule.set_index("DateTime")["TariffLabel"].to_dict()
    tariff_data["TariffLabel"] = tariff_data["DateTime"].map(tariff_lookup).fillna("Medium")
    tariff_data["TariffPrice_p_per_kWh"] = (
        tariff_data["TariffLabel"].map(TARIFF_PRICE_MAP).fillna(TARIFF_PRICE_MAP["Medium"])
    )
    return tariff_data


def load_tou_data(data_folder: Path = DATA_FOLDER) -> pd.DataFrame:
    csv_files = sorted(data_folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_folder}")

    filtered_chunks = []
    for csv_file in csv_files:
        for chunk in pd.read_csv(csv_file, chunksize=250_000):
            chunk.columns = chunk.columns.str.strip()
            filtered_chunks.append(chunk.loc[chunk["stdorToU"].eq("ToU")])

    data = pd.concat(filtered_chunks, ignore_index=True)
    data.columns = data.columns.str.strip()
    return data


def select_diverse_buildings(data: pd.DataFrame, number_of_buildings: int = 10) -> tuple[list[str], pd.DataFrame]:
    if number_of_buildings < 1:
        raise ValueError("number_of_buildings must be at least 1")

    date_time = pd.to_datetime(data["DateTime"])
    data["slot"] = date_time.dt.hour * 2 + date_time.dt.minute // 30
    data["KWH/hh (per half hour)"] = pd.to_numeric(data["KWH/hh (per half hour)"], errors="coerce")
    profiles = data.pivot_table(
        index="LCLid",
        columns="slot",
        values="KWH/hh (per half hour)",
        aggfunc="mean",
    ).reindex(columns=range(48))
    profiles = profiles.dropna()
    if profiles.empty:
        raise ValueError("No buildings have a complete 48-slot consumption profile")
    if number_of_buildings > len(profiles):
        raise ValueError(f"Only {len(profiles)} complete building profiles are available")

    profile_values = profiles.to_numpy(dtype=float)
    profile_totals = profile_values.sum(axis=1)
    normalized_profiles = profile_values / profile_totals[:, None]

    selected_indices = [int(np.argmax(profile_totals))]
    distances = np.linalg.norm(normalized_profiles - normalized_profiles[selected_indices[0]], axis=1)
    for _ in range(1, number_of_buildings):
        next_index = int(np.argmax(distances))
        selected_indices.append(next_index)
        distances = np.minimum(
            distances,
            np.linalg.norm(normalized_profiles - normalized_profiles[next_index], axis=1),
        )

    selected_ids = profiles.index[selected_indices].tolist()
    return selected_ids, profiles.loc[selected_ids]


def find_complementary_pairs(data: pd.DataFrame, top_n: int = 5) -> list[tuple[str, str, float]]:
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    temp = data.copy()
    temp["DateTime"] = pd.to_datetime(temp["DateTime"]).dt.floor("30min")
    temp["slot"] = temp["DateTime"].dt.hour * 2 + temp["DateTime"].dt.minute // 30
    temp["KWH/hh (per half hour)"] = pd.to_numeric(temp["KWH/hh (per half hour)"], errors="coerce")

    profiles = (
        temp.groupby(["LCLid", "slot"], as_index=False)["KWH/hh (per half hour)"]
        .mean()
        .pivot(index="LCLid", columns="slot", values="KWH/hh (per half hour)")
        .reindex(columns=range(48))
        .dropna()
    )
    if profiles.empty:
        raise ValueError("No building load-shape profiles available")

    normalized = profiles.div(profiles.sum(axis=1), axis=0).fillna(0.0)
    building_ids = list(normalized.index)
    pairs: list[tuple[str, str, float]] = []

    for i in range(len(building_ids)):
        for j in range(i + 1, len(building_ids)):
            a = normalized.iloc[i].to_numpy(float)
            b = normalized.iloc[j].to_numpy(float)
            score = float(np.linalg.norm(a - b))
            pairs.append((building_ids[i], building_ids[j], score))

    return sorted(pairs, key=lambda x: x[2], reverse=True)[:top_n]


def estimate_battery_size_per_building(data: pd.DataFrame, target_hours: float = 1.0) -> pd.DataFrame:
    """Estimate a practical per-building battery size from each building's peak 30-min demand.

    The battery capacity is sized as the building's peak half-hour demand multiplied by the
    target discharge duration. A target_hours of 1.0 corresponds to a one-hour battery.
    """
    temp = data.copy()
    temp["DateTime"] = pd.to_datetime(temp["DateTime"]).dt.floor("30min")
    temp["kWh"] = pd.to_numeric(temp["KWH/hh (per half hour)"], errors="coerce")

    peak_30min = (
        temp.groupby(["LCLid", "DateTime"], as_index=False)["kWh"]
        .sum()
        .groupby("LCLid")["kWh"]
        .max()
        .reset_index()
        .rename(columns={"kWh": "peak_30min_kWh"})
    )

    peak_30min["recommended_battery_kWh"] = peak_30min["peak_30min_kWh"] * target_hours
    peak_30min["recommended_battery_kWh"] = peak_30min["recommended_battery_kWh"].round(2)
    return peak_30min.sort_values("peak_30min_kWh", ascending=False).reset_index(drop=True)


def load_pv_data(pv_folder: Path | None = None) -> pd.DataFrame:
    """Load hourly PV generation data from EXPORT HourlyData - Customer Endpoints.

    Returns:
        DataFrame with columns: SerialNo, Substation, datetime, P_GEN_MIN, P_GEN_MAX, etc.
    """
    if pv_folder is None:
        pv_folder = Path(__file__).parent / "PV Data" / "2014-11-28 Cleansed and Processed" / "EXPORT HourlyData"

    pv_file = pv_folder / "EXPORT HourlyData - Customer Endpoints.csv"
    if not pv_file.exists():
        raise FileNotFoundError(f"PV data file not found: {pv_file}")

    pv_df = pd.read_csv(pv_file)
    pv_df.columns = pv_df.columns.str.strip()
    pv_df["datetime"] = pd.to_datetime(pv_df["datetime"])
    return pv_df


def load_weather_data(weather_file: Path | None = None) -> pd.DataFrame:
    """Load hourly weather data from Weather Data Excel file.

    Returns:
        DataFrame with columns: Site, Date, Time, TempOut, WindSpeed, SolarRad, etc.
    """
    if weather_file is None:
        weather_file = Path(__file__).parent / "PV Data" / "Weather Data 2014-11-30.xlsx"

    if not weather_file.exists():
        raise FileNotFoundError(f"Weather file not found: {weather_file}")

    try:
        import openpyxl  # noqa: F401
    except ImportError:
        raise ImportError("openpyxl is required to load weather data. Install it with: pip install openpyxl")

    weather_df = pd.read_excel(weather_file, sheet_name=0)
    weather_df.columns = weather_df.columns.str.strip()

    # Combine Date and Time into DateTime
    weather_df["DateTime"] = pd.to_datetime(weather_df["Date"].astype(str) + " " + weather_df["Time"].astype(str))
    weather_df = weather_df.drop(columns=["Date", "Time"], errors="ignore")
    return weather_df


def load_device_list(device_file: Path | None = None) -> pd.DataFrame:
    """Load device list to map SerialNo to building metadata and PV size.

    Returns:
        DataFrame with columns: Serial Number, Substation, Name, Apparent PV Size, etc.
    """
    if device_file is None:
        device_file = (
            Path(__file__).parent
            / "PV Data"
            / "deviceListTable with explanatory notes v2 - customer addresses removed.xlsx"
        )

    if not device_file.exists():
        raise FileNotFoundError(f"Device list file not found: {device_file}")

    try:
        import openpyxl  # noqa: F401
    except ImportError:
        raise ImportError("openpyxl is required to load device list. Install it with: pip install openpyxl")

    device_df = pd.read_excel(device_file, sheet_name=0)
    device_df.columns = device_df.columns.str.strip()
    return device_df


def collate_building_pv_weather(
    building_data: pd.DataFrame,
    pv_data: pd.DataFrame | None = None,
    weather_data: pd.DataFrame | None = None,
    device_list: pd.DataFrame | None = None,
    aggregate_to_hourly: bool = True,
) -> pd.DataFrame:
    """Collate building meter data with PV and weather data.

    Building data is half-hourly; PV and weather are hourly. This function aligns them by:
    1. Converting building DateTime to hourly (averaging half-hourly values)
    2. Merging on datetime and substation/location
    3. Joining device metadata (PV size, etc.)

    Args:
        building_data: ToU building meter data with LCLid, DateTime, KWH/hh columns
        pv_data: Hourly PV data. If None, loaded automatically.
        weather_data: Hourly weather data. If None, loaded automatically.
        device_list: Device mapping. If None, loaded automatically.
        aggregate_to_hourly: If True, average building data to hourly resolution.

    Returns:
        Collated DataFrame with all available columns from all three sources.
    """
    if pv_data is None:
        pv_data = load_pv_data()
    if weather_data is None:
        weather_data = load_weather_data()
    if device_list is None:
        device_list = load_device_list()

    # Prepare building data
    building_prep = building_data.copy()
    building_prep["DateTime"] = pd.to_datetime(building_prep["DateTime"])
    building_prep["KWH/hh (per half hour)"] = pd.to_numeric(building_prep["KWH/hh (per half hour)"], errors="coerce")

    if aggregate_to_hourly:
        # Aggregate half-hourly building data to hourly by averaging
        building_prep["hour"] = building_prep["DateTime"].dt.floor("h")
        building_hourly = (
            building_prep.groupby(["LCLid", "hour"])
            .agg(
                {
                    "KWH/hh (per half hour)": "sum",
                    "stdorToU": "first",
                    "TariffLabel": "first",
                    "TariffPrice_p_per_kWh": "first",
                }
            )
            .reset_index()
            .rename(columns={"hour": "DateTime", "KWH/hh (per half hour)": "Demand_kWh_hourly"})
        )
    else:
        building_hourly = building_prep.rename(columns={"KWH/hh (per half hour)": "Demand_kWh"})

    # Prepare PV data
    pv_prep = pv_data[
        ["SerialNo", "Substation", "datetime", "P_GEN_MIN", "P_GEN_MAX", "P_IMPORT_MIN", "P_IMPORT_MAX"]
    ].copy()
    pv_prep = pv_prep.rename(columns={"datetime": "DateTime"})

    # Prepare weather data
    weather_prep = weather_data.copy()

    # Create a merged building-weather table (both hourly, indexed by time)
    building_weather = building_hourly.merge(weather_prep, on="DateTime", how="outer")

    # Merge with PV data on DateTime and Substation
    # Note: building data doesn't have Substation info; we'll merge broadly on DateTime first
    collated = building_weather.merge(pv_prep, on="DateTime", how="outer")

    # Join device list metadata (if device info is available in building data or PV data)
    # Optionally merge on Substation or SerialNo if cross-reference is available
    collated = collated.merge(
        device_list, left_on="Substation", right_on="Substation", how="left", suffixes=("", "_device")
    )

    return collated


if __name__ == "__main__":
    tou_data = load_tou_data()
    tou_data = assign_tariff_prices(tou_data)

    print(f"Loaded {len(tou_data):,} ToU rows from {tou_data['LCLid'].nunique():,} buildings.")
    print("Tariff labels:", tou_data["TariffLabel"].value_counts().to_dict())
    print(
        "Example prices:",
        tou_data[["DateTime", "TariffLabel", "TariffPrice_p_per_kWh"]].head().to_dict(orient="records"),
    )
    selected_buildings, selected_profiles = select_diverse_buildings(tou_data)
    print("Selected buildings:", ", ".join(selected_buildings))

    top_pairs = find_complementary_pairs(tou_data, top_n=5)
    print("Most complementary pairs:")
    for a, b, score in top_pairs:
        print(f"  {a} vs {b} | distance={score:.6f}")

    battery_sizes = estimate_battery_size_per_building(tou_data, target_hours=1.0)
    print("\nPer-building battery sizes (1-hour design):")
    print(battery_sizes.head(10).to_string(index=False))

    # Collate building, PV, and weather data
    print("\n\nCollating building, PV, and weather data...")
    try:
        collated = collate_building_pv_weather(tou_data)
        print(f"Collated data shape: {collated.shape}")
        print(f"Collated columns: {collated.columns.tolist()[:15]}...")  # Show first 15 columns
        print("\nCollated data sample:")
        print(collated.head(3).to_string())
    except Exception as e:
        print(f"Error during collation: {e}")
