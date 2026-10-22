import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. MODEL PERFORMANCE RESULTS
# ==========================================

results = pd.DataFrame({
    "Model": [
        "Seasonal Baseline",
        "Random Forest",
        "Growth-Adjusted Forecast",
        "Residual Random Forest"
    ],
    "MAE": [
        211.42,
        282.06,
        505.20,
        584.05
    ],
    "RMSE": [
        253.74,
        344.37,
        645.82,
        625.49
    ]
})

# ==========================================
# 2. SORT BY MAE
# ==========================================

results = results.sort_values("MAE")

print("\nFASHIONIQ - MODEL COMPARISON")
print("----------------------------")

print(results.to_string(index=False))

# ==========================================
# 3. CREATE OUTPUT FOLDER
# ==========================================

output_dir = Path("models")
output_dir.mkdir(exist_ok=True)

# ==========================================
# 4. SAVE RESULTS
# ==========================================

results.to_csv(
    output_dir / "model_comparison.csv",
    index=False
)

# ==========================================
# 5. VISUALIZE MODEL PERFORMANCE
# ==========================================

plt.figure(figsize=(11, 6))

bars = plt.bar(
    results["Model"],
    results["MAE"],
    color=["#2E7D32", "#66BB6A", "#FFA726", "#EF5350"]
)

plt.title(
    "FashionIQ - Forecasting Model Comparison",
    fontsize=15,
    fontweight="bold"
)

plt.ylabel("Mean Absolute Error (Units)")

plt.xticks(rotation=15, ha="right")

for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 8,
        f"{height:.2f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    output_dir / "model_comparison.png",
    dpi=300
)

plt.show()

print("\nModel comparison report generated successfully!")