# from networkx import display
import pandas as pd

irradiance = pd.read_excel("data\\Irradiance_2018_02_21.xlsx")
irradiance = irradiance.iloc[1:, :]

# ensure numeric minutes
irradiance["Irradiances"] = pd.to_numeric(irradiance["Irradiances"])

base_date = pd.Timestamp("2018-02-18")

irradiance["time"] = base_date + pd.to_timedelta(irradiance["Irradiances"], unit="m")

# format only at the end (optional)
irradiance["time"] = irradiance["time"].dt.strftime("%d-%m-%Y %H:%M")
irradiance["PV"] = irradiance["PV"] / 1000  # assuming PV column exists
irradiance.to_csv("data\\Irradiance_2018_02_21.csv", index=False)


# price = pd.read_csv("data\\electricity_price_uk_dec_2025.csv")

# Resample price to 1-minute if a datetime-like column exists


# price["date"] = pd.to_datetime(price["date"])
# print(price["date"])


# price.set_index("date", inplace=True)
# price_1min = price.resample("1T").interpolate(method="time")

# print(price_1min)

# price_1min_day = price_1min.loc["2018-02-21"]
# price_1min["Price (p/kWh)"].to_csv("data\\electricity_price_uk_dec_2025_1min.csv", index=True)
# price_1min_day["Price (p/kWh)"].to_csv("data\\electricity_price_uk_2018_02_21_1min.csv", index=True)


# # Load data
irradiance = pd.read_csv("data\\Irradiance_2018_02_21.csv")
electric_load = pd.read_csv("data\\load_data_2018_02_21.csv")
price = pd.read_csv("data\\electricity_price_uk_2018_02_21_1min.csv")  # change to 5 min aggregates of 24 hours.

setpoint = {"00:00 - 06:00": 10, "06:00 - 08:00": 21, "08:00 - 15:30": 9, "15:30 - 21:00": 21, "21:00 - 00:00": 10}

idx = pd.date_range(start="2018-02-21", periods=1440, freq="min")

df = pd.DataFrame(index=idx)

df["setpoint"] = None

for period, value in setpoint.items():
    start, end = period.split(" - ")

    start_time = pd.to_datetime(f"2018-02-21 {start}")
    end_time = pd.to_datetime(f"2018-02-21 {end}")

    if end_time <= start_time:
        # wraps over midnight
        mask = (df.index >= start_time) | (df.index < end_time)
    else:
        mask = (df.index >= start_time) & (df.index < end_time)

    df.loc[mask, "setpoint"] = value


day_data = pd.concat([irradiance, electric_load, price], axis=1)
day_data = day_data.iloc[:, :]
day_data["Temperature Setpoint"] = df["setpoint"].values

print(day_data.columns)

day_data.to_csv("data\\processed_data_2018_02_21.csv", index=False)
