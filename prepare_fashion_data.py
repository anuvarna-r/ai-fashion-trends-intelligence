import pandas as pd
from pathlib import Path

# Project paths
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Load product information
products = pd.read_csv(
    "products.csv",
    usecols=[
        "Product ID",
        "Category",
        "Sub Category",
        "Description EN"
    ]
)

# Store monthly sales summaries
monthly_results = []

# Read large transaction dataset in chunks
chunks = pd.read_csv(
    "transactions.csv",
    usecols=[
        "Product ID",
        "Date",
        "Quantity",
        "Line Total",
        "Transaction Type"
    ],
    chunksize=100000
)

for chunk_number, chunk in enumerate(chunks, start=1):

    # Keep sales only
    chunk = chunk[
        chunk["Transaction Type"] == "Sale"
    ].copy()

    # Convert dates
    chunk["Date"] = pd.to_datetime(
        chunk["Date"],
        errors="coerce"
    )

    # Remove invalid records
    chunk = chunk.dropna(
        subset=["Date", "Product ID", "Quantity"]
    )

    # Connect products and transactions
    chunk = chunk.merge(
        products,
        on="Product ID",
        how="left",
        validate="many_to_one"
    )

    # Extract month
    chunk["Month"] = chunk["Date"].dt.to_period("M").astype(str)

    # Aggregate monthly product-category sales
    summary = chunk.groupby(
        ["Month", "Category", "Sub Category"],
        dropna=False,
        as_index=False
    ).agg(
        Units_Sold=("Quantity", "sum"),
        Revenue=("Line Total", "sum"),
        Transaction_Lines=("Product ID", "size")
    )

    monthly_results.append(summary)

    print(f"Processed chunk {chunk_number}")

# Combine chunk summaries
fashion_data = pd.concat(
    monthly_results,
    ignore_index=True
)

# Aggregate again across chunks
fashion_data = fashion_data.groupby(
    ["Month", "Category", "Sub Category"],
    dropna=False,
    as_index=False
).agg(
    Units_Sold=("Units_Sold", "sum"),
    Revenue=("Revenue", "sum"),
    Transaction_Lines=("Transaction_Lines", "sum")
)

# Sort by month
fashion_data = fashion_data.sort_values(
    ["Month", "Category", "Sub Category"]
)

# Save prepared dataset
output_path = DATA_DIR / "fashion_monthly_sales.csv"

fashion_data.to_csv(output_path, index=False)

print("\nFASHIONIQ DATA PREPARATION COMPLETED")
print("-----------------------------------")
print("Total Monthly Records:", len(fashion_data))
print("Output File:", output_path)

print("\nFirst 10 Records:")
print(fashion_data.head(10).to_string(index=False))