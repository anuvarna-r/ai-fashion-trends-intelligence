import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/fashion_monthly_sales.csv")

df["Month"] = pd.to_datetime(df["Month"])

# Exclude incomplete March 2025
df = df[df["Month"] < "2025-03-01"]

# Aggregate total monthly units sold
monthly = (
    df.groupby("Month")["Units_Sold"]
    .sum()
    .sort_index()
)

# Ensure monthly frequency
monthly = monthly.asfreq("MS")

print("\nFASHIONIQ - FORECASTING MODEL")
print("-----------------------------------")

print("Total Complete Months:", len(monthly))

# ==========================================
# 2. TRAIN-TEST SPLIT
# ==========================================

# First 24 months for training
train = monthly.loc[:"2024-12-01"]

# January and February 2025 for testing
test = monthly.loc["2025-01-01":"2025-02-01"]

print("\nTraining Months:", len(train))
print("Testing Months:", len(test))

# ==========================================
# 3. SEASONAL NAIVE FORECAST
# ==========================================

# Predict using the same month last year
predictions = []

for date in test.index:

    previous_year = date - pd.DateOffset(years=1)

    predicted_value = train.loc[previous_year]

    predictions.append(predicted_value)

predictions = np.array(predictions)

# ==========================================
# 4. MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(test, predictions)

rmse = np.sqrt(
    mean_squared_error(test, predictions)
)

print("\nMODEL PERFORMANCE")
print("-----------------------------------")

print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))

# ==========================================
# 5. ACTUAL VS PREDICTED
# ==========================================

results = pd.DataFrame({
    "Month": test.index,
    "Actual_Sales": test.values,
    "Predicted_Sales": predictions
})

print("\nACTUAL VS PREDICTED SALES")
print(results.to_string(index=False))

# ==========================================
# 6. SAVE RESULTS
# ==========================================

output_dir = Path("models")
output_dir.mkdir(exist_ok=True)

results.to_csv(
    output_dir / "forecast_results.csv",
    index=False
)

print("\nForecasting baseline completed successfully!")