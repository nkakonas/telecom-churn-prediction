import pandas as pd
import numpy as np
from pathlib import Path

def preprocess_all():
    raw_dir = Path("data/raw_data")

    # Load datasets
    files = {
        "customer": "CustomerChurn.xlsx",
        "telco": "Telco_customer_churn.xlsx",
        "demo": "Telco_customer_churn_demographics.xlsx",
        "loc": "Telco_customer_churn_location.xlsx",
        "pop": "Telco_customer_churn_population.xlsx",
        "serv": "Telco_customer_churn_services.xlsx",
        "stat": "Telco_customer_churn_status.xlsx",
    }
    data = {k: pd.read_excel(raw_dir / v) for k, v in files.items()}

    # Basic column cleanup
    def clean_cols(df):
        df.columns = (
            df.columns.str.strip()
            .str.replace(" ", "_")
            .str.replace("-", "_")
            .str.replace("__", "_")
            .str.lower()
        )
        if "customerid" in df.columns:
            df.rename(columns={"customerid": "customer_id"}, inplace=True)
        if "customer_id" in df.columns:
            df["customer_id"] = df["customer_id"].astype(str).str.strip().str.upper()
        if "zip_code" in df.columns:
            df["zip_code"] = df["zip_code"].astype(str).str.strip()
        return df

    for k in data:
        data[k] = clean_cols(data[k])

    # Drop redundant metadata
    for k in ["stat", "serv", "demo", "loc"]:
        df = data[k]
        drop_cols = []
        if "count" in df.columns:
            drop_cols.append("count")
        if "quarter" in df.columns and df["quarter"].nunique() <= 1:
            drop_cols.append("quarter")
        df.drop(columns=drop_cols, inplace=True, errors="ignore")

    telco = data["telco"].copy()

    # Simple numeric / categorical cleaning
    for c in telco.select_dtypes(include="number").columns:
        telco[c] = telco[c].fillna(telco[c].median())
    for c in telco.select_dtypes(exclude="number").columns:
        telco[c] = telco[c].astype(str).str.strip().str.lower().replace("nan", np.nan)

    # Safe merge helper
    def safe_merge(left, right, key, suf):
        overlap = [c for c in right.columns if c in left.columns and c != key]
        right = right.rename(columns={c: f"{c}_{suf}" for c in overlap})
        return left.merge(right, on=key, how="left")

    merged = telco
    merged = safe_merge(merged, data["stat"], "customer_id", "stat")
    merged = safe_merge(merged, data["serv"], "customer_id", "svc")
    merged = safe_merge(merged, data["demo"], "customer_id", "demo")
    merged = safe_merge(merged, data["loc"], "customer_id", "loc")

    if "zip_code" in merged.columns and "zip_code" in data["pop"].columns:
        merged = safe_merge(merged, data["pop"], "zip_code", "pop")

    # Final cleaning
    merged = merged.dropna(axis=1, how="all")  # drop empty columns
    for c in merged.select_dtypes(include="number").columns:
        merged[c] = merged[c].fillna(merged[c].median())
    for c in merged.select_dtypes(exclude="number").columns:
        merged[c] = merged[c].fillna("missing")

    # Convert to binary
    def detect_object_cardinality(df):
        """Return dict {col: count_non_null_unique} for object dtype columns."""
        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
        return {c: df[c].dropna().astype(str).str.strip().nunique() for c in obj_cols}

    def apply_object_conversions(df, max_cardinality=4, save_mapping=True):
        """
        Convert object columns:
          - count == 2 -> create <col>_bin (nullable Int64) mapping sorted unique -> {0,1}
          - 2 < count <= max_cardinality -> track as low-cardinality categorical (keep original)
        Drops original binary columns after conversion.
        Returns (df, mappings) where mappings is a dict of applied mappings.
        """
        obj_card = detect_object_cardinality(df)
        mappings = {"binary": {}, "low_cardinality_cols": []}
        cols_to_drop = []

        for col, count in obj_card.items():
            if count == 0 or count == 1 or count > max_cardinality:
                continue
            # prepare normalized non-null values
            nonnull = df[col].dropna().astype(str).str.strip()
            if count == 2:
                uniq = sorted(nonnull.unique())
                mapping = {uniq[0]: 0, uniq[1]: 1}
                df[f"{col}_bin"] = df[col].astype(str).str.strip().map(mapping).astype("Int64")
                mappings["binary"][col] = mapping
                cols_to_drop.append(col)
            else:
                # Track as low-cardinality categorical but keep original column
                mappings["low_cardinality_cols"].append({
                    "column": col,
                    "cardinality": count,
                    "values": sorted(nonnull.unique())
                })

        # Drop original columns that were converted to binary
        df.drop(columns=cols_to_drop, inplace=True)

        if save_mapping:
            out_dir = Path("data") / "exploration"
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                import json
                with open(out_dir / "object_conversions.json", "w") as f:
                    json.dump(mappings, f, indent=2)
            except Exception:
                # non-fatal: continue if saving mapping fails
                pass

        # print("Applied object conversions -> binary:", list(mappings["binary"].keys()))
        # print(f"Dropped {len(cols_to_drop)} original binary columns")
        # print(f"Tracked {len(mappings['low_cardinality_cols'])} low-cardinality columns:", 
              #[c['column'] for c in mappings['low_cardinality_cols']]
        return df, mappings

    merged, mappings = apply_object_conversions(merged, max_cardinality=4, save_mapping=True)

    out_path = Path("data/merged_telco_preprocessed.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    # Summary
    print("\n--- Merge Summary ---")
    print(f"Rows: {merged.shape[0]}, Columns: {merged.shape[1]}")
    miss = merged.isna().mean().sort_values(ascending=False)
    print("\nTop Missing Columns (%):")
    print((miss.head(10) * 100).round(2))
    if "churn_value" in merged.columns:
        print("\nChurn Value Counts:")
        print(merged["churn_value"].value_counts(dropna=False))
    print("\nSaved:", out_path.resolve())
    print("----------------------\n")
    return merged

# add preprocssing for unsupervised learning
def preprocess_for_unsupervised():
    data = preprocess_all()
    
    # Drop columns not needed for unsupervised learning
    unsupervised_data = data.drop(columns=['country', 'state', 'count', 'churn_value', 'customer_id', 'lat_long', 'id'], errors='ignore')
    
    # Fill missing total_charges
    if 'total_charges' in unsupervised_data.columns:
        unsupervised_data['total_charges'].fillna(
            unsupervised_data['monthly_charges'] * unsupervised_data['tenure_months'], 
            inplace=True
        )
    
    # Categorical column: phone only / internet only / both
    if 'phone_service_bin' in unsupervised_data.columns and 'internet_service_bin' in unsupervised_data.columns:
        unsupervised_data['service_type'] = np.where(
            (unsupervised_data['phone_service_bin'] == 1) & (unsupervised_data['internet_service_bin'].isna()), 'phone_only',
            np.where(
                (unsupervised_data['phone_service_bin'].isna()) & (unsupervised_data['internet_service_bin'].notna()), 'internet_only',
                'both'
            )
        )
    
    out_path = Path("data/unsupervised_telco_preprocessed.csv")
    unsupervised_data.to_csv(out_path, index=False)
    print("Unsupervised data saved:", out_path.resolve())

if __name__ == "__main__":
    print("Starting preprocessing...")
    preprocess_all()
    print("Preprocessing complete.")
    # preprocess_for_unsupervised()
