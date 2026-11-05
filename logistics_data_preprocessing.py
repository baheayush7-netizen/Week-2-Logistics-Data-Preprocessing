"""
Week 2 Task: Data Collection, Cleaning and Preprocessing for Logistics Analysis

Dataset:
Brazilian E-Commerce Public Dataset by Olist (Kaggle)
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Before running:
1. Download the dataset ZIP from Kaggle.
2. Extract the CSV files into a folder named "data/raw" beside this script.
3. Install dependencies:
   pip install pandas numpy scikit-learn matplotlib

Run:
   python logistics_data_preprocessing.py

Outputs are saved in data/processed/.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORDERS_FILE = RAW_DIR / "olist_orders_dataset.csv"
ITEMS_FILE = RAW_DIR / "olist_order_items_dataset.csv"


def load_data():
    """Load the two tables needed for order and delivery analysis."""
    if not ORDERS_FILE.exists() or not ITEMS_FILE.exists():
        raise FileNotFoundError(
            "CSV files not found. Put the Olist CSV files in data/raw/. "
            f"Expected: {ORDERS_FILE.name} and {ITEMS_FILE.name}"
        )

    orders = pd.read_csv(ORDERS_FILE)
    items = pd.read_csv(ITEMS_FILE)

    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col], errors="coerce")

    if "shipping_limit_date" in items.columns:
        items["shipping_limit_date"] = pd.to_datetime(
            items["shipping_limit_date"], errors="coerce"
        )

    return orders, items


def inspect_data(orders, items):
    """Print dimensions, missing-value counts and duplicate counts."""
    print("\n--- INITIAL DATA PROFILE ---")
    print("Orders shape:", orders.shape)
    print("Order items shape:", items.shape)
    print("\nMissing values in orders:\n", orders.isna().sum())
    print("\nMissing values in items:\n", items.isna().sum())
    print("\nDuplicate order rows:", orders.duplicated().sum())
    print("Duplicate item rows:", items.duplicated().sum())


def clean_and_merge(orders, items):
    """Remove exact duplicates, validate numeric values and join tables."""
    orders = orders.drop_duplicates().copy()
    items = items.drop_duplicates().copy()

    # Convert expected numeric columns safely.
    for col in ["price", "freight_value"]:
        if col in items.columns:
            items[col] = pd.to_numeric(items[col], errors="coerce")

    # Negative prices/freight are invalid for this analysis; mark as missing.
    for col in ["price", "freight_value"]:
        if col in items.columns:
            items.loc[items[col] < 0, col] = np.nan

    # Keep orders with missing delivery dates: they may be in transit/cancelled.
    # A left join preserves order items even when an order has no delivery date.
    data = items.merge(orders, on="order_id", how="left", validate="many_to_one")

    # Derive delivery duration only where both dates are available.
    data["delivery_days"] = (
        data["order_delivered_customer_date"] -
        data["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    # Positive delay means delivered after the estimated date.
    data["delay_days"] = (
        data["order_delivered_customer_date"] -
        data["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400

    # Remove impossible negative delivery durations from this derived metric.
    data.loc[data["delivery_days"] < 0, "delivery_days"] = np.nan

    return data


def detect_outliers_iqr(series):
    """Return IQR lower/upper limits and an outlier boolean mask."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper, (series < lower) | (series > upper)


def handle_numeric_missing_values(data):
    """Median-impute price/freight only; retain missing delivery timestamps."""
    for col in ["price", "freight_value"]:
        if col in data.columns:
            median_value = data[col].median()
            if pd.notna(median_value):
                data[col] = data[col].fillna(median_value)
    return data


def normalize_features(data):
    """Create Min-Max scaled versions without overwriting original columns."""
    cols = ["price", "freight_value"]
    available = [c for c in cols if c in data.columns]
    if available:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data[available])
        for index, col in enumerate(available):
            data[f"{col}_normalized"] = scaled[:, index]
    return data


def main():
    orders, items = load_data()
    inspect_data(orders, items)

    data = clean_and_merge(orders, items)

    print("\n--- OUTLIER CHECK (IQR METHOD) ---")
    for col in ["price", "freight_value", "delivery_days"]:
        if col in data.columns:
            valid = data[col].dropna()
            if not valid.empty:
                low, high, flags = detect_outliers_iqr(valid)
                print(
                    f"{col}: lower={low:.2f}, upper={high:.2f}, "
                    f"outliers={int(flags.sum())}"
                )
                data[f"{col}_is_outlier"] = False
                data.loc[valid.index, f"{col}_is_outlier"] = flags

    # Do not automatically delete outliers: high freight or long delivery
    # times can be genuine operational events that logistics teams should study.
    data = handle_numeric_missing_values(data)
    data = normalize_features(data)

    # Keep a clear record of the cleaned data and a small quality summary.
    output_file = OUT_DIR / "logistics_preprocessed.csv"
    data.to_csv(output_file, index=False)

    summary = pd.DataFrame({
        "column": data.columns,
        "missing_values": [int(data[c].isna().sum()) for c in data.columns],
        "data_type": [str(data[c].dtype) for c in data.columns],
    })
    summary.to_csv(OUT_DIR / "data_quality_summary.csv", index=False)

    print("\n--- COMPLETED ---")
    print("Processed data:", output_file)
    print("Quality summary:", OUT_DIR / "data_quality_summary.csv")
    print("Final shape:", data.shape)
    print("Note: Missing delivery dates are intentionally retained.")


if __name__ == "__main__":
    main()
