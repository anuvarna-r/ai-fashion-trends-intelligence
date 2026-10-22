import pandas as pd
from pathlib import Path

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/fashion_monthly_sales.csv")

df["Month"] = pd.to_datetime(df["Month"])

# Exclude incomplete March 2025
df = df[df["Month"] < "2025-03-01"]

# ==========================================
# 2. SORT DATA
# ==========================================

df = df.sort_values(
    ["Category", "Sub Category", "Month"]
).reset_index(drop=True)

# ==========================================
# 3. CREATE TIME FEATURES
# ==========================================

df["Month_Number"] = df["Month"].dt.month

df["Year"] = df["Month"].dt.year

# ==========================================
# 4. CREATE HISTORICAL SALES FEATURES
# ==========================================

group = df.groupby(
    ["Category", "Sub Category"]
)["Units_Sold"]

df["Lag_1"] = group.shift(1)

df["Lag_2"] = group.shift(2)

df["Lag_3"] = group.shift(3)

df["Lag_12"] = group.shift(12)

# ==========================================
# 5. CREATE ROLLING FEATURES
# ==========================================

# Shift first to prevent target leakage
df["Rolling_Mean_3"] = (
    df.groupby(["Category", "Sub Category"])["Units_Sold"]
    .transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
)

# ==========================================
# 6. REMOVE ROWS WITHOUT HISTORY
# ==========================================

df = df.dropna(
    subset=[
        "Lag_1",
        "Lag_2",
        "Lag_3",
        "Lag_12",
        "Rolling_Mean_3"
    ]
)

# ==========================================
# 7. SAVE ML DATASET
# ==========================================

output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "fashion_ml_features.csv"

df.to_csv(output_path, index=False)

# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

print("\nFASHIONIQ - ML FEATURE ENGINEERING")
print("----------------------------------")

print("\nTotal ML Records:", len(df))

print("\nAvailable Features:")
print(df.columns.tolist())

print("\nSample Records:")
print(df.head().to_string(index=False))

print("\nML dataset saved successfully!")