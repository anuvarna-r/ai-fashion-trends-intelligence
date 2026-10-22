import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/fashion_ml_features.csv")
df["Month"] = pd.to_datetime(df["Month"])

# ==========================================
# 2. SELECT TEST DATA
# ==========================================

test = df[df["Month"] >= "2025-01-01"].copy()

# ==========================================
# 3. SEASONAL BASELINE
# ==========================================

test["Seasonal_Forecast"] = test["Lag_12"]

# ==========================================
# 4. GROWTH-ADJUSTED FORECAST
# ==========================================

# Compare the previous month's sales with the
# corresponding month from the year before.

original = pd.read_csv("data/fashion_monthly_sales.csv")
original["Month"] = pd.to_datetime(original["Month"])

original = original[
    original["Month"] < "2025-03-01"
].copy()

original = original.sort_values(
    ["Category", "Sub Category", "Month"]
)

group = original.groupby(
    ["Category", "Sub Category"]
)["Units_Sold"]

original["Lag_13"] = group.shift(13)

growth_lookup = original[
    ["Month", "Category", "Sub Category", "Lag_13"]
]

test = test.merge(
    growth_lookup,
    on=["Month", "Category", "Sub Category"],
    how="left",
    validate="one_to_one"
)

# Previous month's year-over-year growth
growth_rate = (
    (test["Lag_1"] - test["Lag_13"])
    / test["Lag_13"].replace(0, np.nan)
)

# Limit extreme adjustments
growth_rate = growth_rate.clip(-0.20, 0.20).fillna(0)

test["Growth_Forecast"] = (
    test["Lag_12"] * (1 + growth_rate)
).clip(lower=0)

# ==========================================
# 5. EVALUATE MODELS
# ==========================================

print("\nFASHIONIQ - GROWTH FORECAST EVALUATION")
print("-------------------------------------")

for column in ["Seasonal_Forecast", "Growth_Forecast"]:

    mae = mean_absolute_error(
        test["Units_Sold"],
        test[column]
    )

    rmse = np.sqrt(
        mean_squared_error(
            test["Units_Sold"],
            test[column]
        )
    )

    print(f"\n{column}")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))

print("\nGrowth forecasting evaluation completed!")