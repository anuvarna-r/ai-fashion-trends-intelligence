import pandas as pd

file_path = "transactions.csv"

total_rows = 0
earliest_date = None
latest_date = None
transaction_types = {}

for chunk in pd.read_csv(
    file_path,
    usecols=["Date", "Transaction Type"],
    chunksize=100000
):
    total_rows += len(chunk)

    dates = pd.to_datetime(chunk["Date"], errors="coerce")

    chunk_min = dates.min()
    chunk_max = dates.max()

    if pd.notna(chunk_min):
        if earliest_date is None or chunk_min < earliest_date:
            earliest_date = chunk_min

    if pd.notna(chunk_max):
        if latest_date is None or chunk_max > latest_date:
            latest_date = chunk_max

    counts = chunk["Transaction Type"].value_counts()

    for transaction_type, count in counts.items():
        transaction_types[transaction_type] = (
            transaction_types.get(transaction_type, 0) + count
        )

print("\nFASHIONIQ - SALES DATASET ANALYSIS")
print("-----------------------------------")

print("Total Transaction Rows:", total_rows)
print("Earliest Transaction:", earliest_date)
print("Latest Transaction:", latest_date)

if earliest_date is not None and latest_date is not None:
    months = (
        (latest_date.year - earliest_date.year) * 12
        + latest_date.month - earliest_date.month + 1
    )
    print("Calendar Months Covered:", months)

print("\nTransaction Types:")
for transaction_type, count in transaction_types.items():
    print(f"{transaction_type}: {count}")