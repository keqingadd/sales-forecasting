import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

start_date = datetime(1992, 1, 1)
months = 325
dates = [start_date + timedelta(days=30*i) for i in range(months)]

base_sales = 1000
trend = np.linspace(0, 500, months)
seasonal = 200 * np.sin(2 * np.pi * np.arange(months) / 12)
noise = np.random.normal(0, 50, months)
sales = base_sales + trend + seasonal + noise
sales = np.maximum(sales, 100)

df = pd.DataFrame({
    'Date': [d.strftime('%Y-%m-%d') for d in dates],
    'Alcohol_Sales': sales.astype(int)
})

df.to_csv('D:/AiWork/SalesForecast/data/Alcohol_Sales.csv', index=False)
print(f"Dataset generated: {len(df)} rows")
print(df.head())