import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

# Load monthly sales data
df = pd.read_csv("data/fashion_monthly_sales.csv")

df["Month"] = pd.to_datetime(df["Month"])

# Exclude incomplete March 2025
df = df[df["Month"] < "2025-03-01"]

monthly = (
    df.groupby("Month")["Units_Sold"]
    .sum()
    .sort_index()
    .asfreq("MS")
)

if monthly.isna().any():
    raise ValueError("Missing months detected.")

results = []

# Evaluate predictions across 14 months
for i in range(12, len(monthly)):

    month = monthly.index[i]
    actual = monthly.iloc[i]

    previous_month = monthly.iloc[i - 1]
    previous_year = monthly.iloc[i - 12]

    results.append({
        "Month": month,
        "Actual": actual,
        "Previous_Month_Forecast": previous_month,
        "Seasonal_Forecast": previous_year
    })

results = pd.DataFrame(results)

print("\nFASHIONIQ - HISTORICAL FORECAST EVALUATION")
print("-----------------------------------------")

for column in ["Previous_Month_Forecast", "Seasonal_Forecast"]:

    mae = mean_absolute_error(results["Actual"], results[column])

    rmse = np.sqrt(
        mean_squared_error(results["Actual"], results[column])
    )

    print(f"\n{column}")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))

print("\nMONTHLY FORECAST COMPARISON")
print(results.to_string(index=False))

# Save results
output_dir = Path("models")
output_dir.mkdir(exist_ok=True)

results.to_csv(
    output_dir / "historical_forecast_evaluation.csv",
    index=False
)

print("\nHistorical evaluation completed successfully!")