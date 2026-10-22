import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/fashion_ml_features.csv")
df["Month"] = pd.to_datetime(df["Month"])

# ==========================================
# 2. CREATE RESIDUAL TARGET
# ==========================================

df["Residual"] = df["Units_Sold"] - df["Lag_12"]

# ==========================================
# 3. DEFINE FEATURES
# ==========================================

features = [
    "Category",
    "Sub Category",
    "Month_Number",
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Lag_12",
    "Rolling_Mean_3"
]

# ==========================================
# 4. CHRONOLOGICAL SPLIT
# ==========================================

train = df[df["Month"] < "2025-01-01"].copy()
test = df[df["Month"] >= "2025-01-01"].copy()

X_train = train[features]
y_train = train["Residual"]

X_test = test[features]
y_test = test["Units_Sold"]

# ==========================================
# 5. BUILD MODEL
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["Category", "Sub Category"]
        )
    ],
    remainder="passthrough"
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

# ==========================================
# 6. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)

# ==========================================
# 7. GENERATE FINAL PREDICTIONS
# ==========================================

predicted_residuals = model.predict(X_test)

predictions = (
    test["Lag_12"].values + predicted_residuals
)

predictions = np.maximum(predictions, 0)

# ==========================================
# 8. EVALUATE
# ==========================================

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

baseline = test["Lag_12"].values

baseline_mae = mean_absolute_error(y_test, baseline)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline)
)

print("\nFASHIONIQ - RESIDUAL FORECASTING MODEL")
print("-------------------------------------")

print("\nTraining Records:", len(train))
print("Testing Records:", len(test))

print("\nRESIDUAL RANDOM FOREST")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

print("\nSEASONAL BASELINE")
print("MAE:", round(baseline_mae, 2))
print("RMSE:", round(baseline_rmse, 2))

print("\nResidual forecasting evaluation completed!")