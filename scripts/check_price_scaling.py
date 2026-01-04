import pandas as pd

p = pd.read_csv("data/processed_data_2018_02_21.csv")
# normalize column names
p.columns = p.columns.str.strip().str.lower()
print("columns:", p.columns.tolist())
if 'price' in p.columns:
    print("minute price range (p/kWh):", p['price'].min(), p['price'].max(), "mean", p['price'].mean())
else:
    print("price column not found")
# create hour index
p['hour'] = p.index // 60
p_hr = pd.DataFrame()
# in main script price is divided by 100 earlier; but we simulate both behaviours
p_hr['price_mean'] = p.groupby('hour')['price'].mean()
print('hourly mean price (p/kWh) min/max/mean:', p_hr['price_mean'].min(), p_hr['price_mean'].max(), p_hr['price_mean'].mean())
# convert to £/kWh
print('hourly mean price (£/kWh):', p_hr['price_mean'].min()/100.0, p_hr['price_mean'].max()/100.0, p_hr['price_mean'].mean()/100.0)
