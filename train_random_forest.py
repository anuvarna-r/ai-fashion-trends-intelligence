import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ==========================================
# 1. LOAD ML DATASET
# ==========================================

df = pd.read_csv("data/fashion_ml_features.csv")

df["Month"] = pd.to_datetime(df["Month"])

# ==========================================
# 2. DEFINE FEATURES AND TARGET
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

target = "Units_Sold"

# ==========================================
# 3. CHRONOLOGICAL TRAIN-TEST SPLIT
# ==========================================

train = df[df["Month"] < "2025-01-01"].copy()
test = df[df["Month"] >= "2025-01-01"].copy()

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

# ==========================================
# 4. PREPROCESS CATEGORICAL FEATURES
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

# ==========================================
# 5. BUILD RANDOM FOREST PIPELINE
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
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
# 7. GENERATE PREDICTIONS
# ==========================================

predictions = model.predict(X_test)

# Sales quantities cannot be negative
predictions = np.maximum(predictions, 0)

# ==========================================
# 8. EVALUATE RANDOM FOREST
# ==========================================

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# Seasonal baseline: same month last year
baseline_predictions = test["Lag_12"].values

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_predictions)
)

print("\nFASHIONIQ - RANDOM FOREST FORECASTING")
print("------------------------------------")

print("\nTraining Records:", len(train))
print("Testing Records:", len(test))

print("\nRANDOM FOREST PERFORMANCE")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R² Score:", round(r2, 4))

print("\nSEASONAL BASELINE PERFORMANCE")
print("MAE:", round(baseline_mae, 2))
print("RMSE:", round(baseline_rmse, 2))

# ==========================================
# 9. SAVE PREDICTION RESULTS
# ==========================================

results = test[
    ["Month", "Category", "Sub Category"]
].copy()

results["Actual_Sales"] = y_test.values
results["Random_Forest_Prediction"] = predictions.round(0)
results["Seasonal_Baseline"] = baseline_predictions

output_dir = Path("models")
output_dir.mkdir(exist_ok=True)

results.to_csv(
    output_dir / "random_forest_predictions.csv",
    index=False
)

# ==========================================
# 10. SAVE TRAINED MODEL
# ==========================================

joblib.dump(
    model,
    output_dir / "fashion_random_forest.pkl"
)

print("\nSAMPLE PREDICTIONS")
print(results.head(10).to_string(index=False))

print("\nRandom Forest model trained successfully!")
print("Model saved in models/fashion_random_forest.pkl")