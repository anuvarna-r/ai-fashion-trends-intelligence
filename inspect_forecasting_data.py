import pandas as pd

# Load prepared fashion sales data
df = pd.read_csv("data/fashion_monthly_sales.csv")

df["Month"] = pd.to_datetime(df["Month"])

# Exclude incomplete March 2025
df = df[df["Month"] < "2025-03-01"]

print("\nFASHIONIQ - CATEGORY FORECASTING DATA")
print("------------------------------------")

print("\nTotal Records:", len(df))

print("\nDate Range:")
print("Start:", df["Month"].min())
print("End:", df["Month"].max())

# Count monthly observations by category
category_summary = (
    df.groupby(["Category", "Sub Category"])
    .agg(
        Months_Available=("Month", "nunique"),
        Total_Units_Sold=("Units_Sold", "sum")
    )
    .reset_index()
    .sort_values("Total_Units_Sold", ascending=False)
)

print("\nCATEGORY FORECASTING SUMMARY")
print(category_summary.to_string(index=False))

print("\nCategories With Complete 26-Month History:")

complete_categories = category_summary[
    category_summary["Months_Available"] == 26
]

print("Total:", len(complete_categories))

print("\nDATA INSPECTION COMPLETED")