import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load prepared dataset
df = pd.read_csv("data/fashion_monthly_sales.csv")

# Create output directory
output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

# Convert month to datetime
df["Month"] = pd.to_datetime(df["Month"])

print("\nFASHIONIQ - FASHION TREND ANALYSIS")
print("-----------------------------------")

print("\nDataset Records:", len(df))

# ==========================================
# 1. TOP FASHION SUBCATEGORIES
# ==========================================

top_categories = (
    df.groupby("Sub Category")["Units_Sold"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTOP 10 FASHION SUBCATEGORIES")
print(top_categories)

plt.figure(figsize=(12, 6))

top_categories.sort_values().plot(
    kind="barh",
    color="hotpink"
)

plt.title("Top 10 Best-Selling Fashion Subcategories")
plt.xlabel("Total Units Sold")
plt.ylabel("Fashion Subcategory")
plt.tight_layout()

plt.savefig(output_dir / "top_fashion_categories.png")
plt.close()

# ==========================================
# 2. MONTHLY FASHION SALES TREND
# ==========================================

monthly_sales = (
    df.groupby("Month")["Units_Sold"]
    .sum()
    .sort_index()
)

print("\nMONTHLY FASHION SALES")
print(monthly_sales)

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o",
    color="purple"
)

plt.title("Monthly Fashion Sales Trend")
plt.xlabel("Month")
plt.ylabel("Units Sold")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(output_dir / "monthly_fashion_sales.png")
plt.close()

# ==========================================
# 3. FEMININE FASHION ANALYSIS
# ==========================================

feminine = df[df["Category"] == "Feminine"]

feminine_categories = (
    feminine.groupby("Sub Category")["Units_Sold"]
    .sum()
    .sort_values(ascending=False)
)

print("\nFEMININE FASHION CATEGORY PERFORMANCE")
print(feminine_categories)

plt.figure(figsize=(12, 6))

feminine_categories.sort_values().plot(
    kind="barh",
    color="orchid"
)

plt.title("Women's Fashion Category Performance")
plt.xlabel("Units Sold")
plt.ylabel("Fashion Category")
plt.tight_layout()

plt.savefig(output_dir / "feminine_fashion_trends.png")
plt.close()

print("\nFASHION TREND ANALYSIS COMPLETED")
print("-----------------------------------")

print("Charts saved in the data folder.")