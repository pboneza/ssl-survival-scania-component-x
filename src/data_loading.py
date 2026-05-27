"""
This is where all the functions related to the checking on the datasets are created.
Basic inspection functions such as checking the shape, names of columns, missing values, etc.
"""

from pathlib import Path
import pandas as pd

# Loading one dataset as a csv file
def load_csv(file_path: Path, name: str) -> pd.DataFrame:
    """
    Load a CSV file and print its shape.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"{name} file not found: {file_path}")

    df = pd.read_csv(file_path)
    print(f"{name} loaded successfully: {df.shape}")
    return df

# loading multiple datasets (3 raw training datasets)
def load_raw_training_data(
    operational_file: Path,
    specifications_file: Path,
    tte_file: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three raw training data files.
    """
    operational_df = load_csv(operational_file, "Operational")
    spec_df = load_csv(specifications_file, "Specifications")
    tte_df = load_csv(tte_file, "TTE")

    return operational_df, spec_df, tte_df

# loading processed dataset
def load_processed_training_data(
    X_train_file: Path,
    X_val_file: Path,
    y_train_file: Path,
    y_val_file: Path,
) -> tuple[pd.DataFrame,pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load four processed datasets, from training and validation
    """
    X_train = load_csv(X_train_file, "X_train")
    X_val = load_csv(X_val_file, "X_val")
    y_train = load_csv(y_train_file, "y_train_trunc")
    y_val = load_csv(y_val_file, "y_val_trunc")

    return X_train, X_val, y_train, y_val

def load_processed_test_data(
    X_test_file: Path,
    y_test_file: Path,
) -> tuple[pd.DataFrame,pd.DataFrame]:
    """
    Load two processed test datasets for final perfomance evaluation
    """
    X_test = load_csv(X_test_file, "X_test")
    y_test = load_csv(y_test_file, "y_test")

    return X_test, y_test

    
# Dataset inspection, check rows, columns, and missing values
def inspect_dataframe(df: pd.DataFrame, name: str, n_rows: int = 5) -> None:
    """
    Print a compact inspection of a dataframe.
    """
    print(f"\n{name} shape: {df.shape}")
    print(f"\n{name} columns:")
    print(df.columns.tolist())

    print(f"\n{name} preview:")
    print(df.head(n_rows))

    print(f"\n{name} top 10 missing values:")
    print(df.isna().sum().sort_values(ascending=False).head(10))

# Before merging the datasets and grouping readouts according to vehicle_id, check for consistency in vehicle naming across the three datasets
def check_vehicle_id_coverage(
    operational_df: pd.DataFrame,
    spec_df: pd.DataFrame,
    tte_df: pd.DataFrame
) -> None:
    """
    Check whether vehicle IDs are aligned across the three raw sources.
    set() converts unique IDs into a Python set, so it's easier to remove
    duplicates, check membership, compare groups and compute differences
    """
    op_ids = set(operational_df["vehicle_id"].unique())
    spec_ids = set(spec_df["vehicle_id"].unique())
    tte_ids = set(tte_df["vehicle_id"].unique())

    print("Unique vehicles in operational:", len(op_ids))
    print("Unique vehicles in specifications:", len(spec_ids))
    print("Unique vehicles in TTE:", len(tte_ids))
    """
    If the output number is the same, good sign but does not tell the whole 
    story. Check if the IDs are exactly the same, through set differences. If 
    the result is empty, then the output count is zero.
    """

    print("\nVehicles in operational but not in specifications:", len(op_ids - spec_ids))
    print("Vehicles in operational but not in TTE:", len(op_ids - tte_ids))
    print("Vehicles in specifications but not in operational:", len(spec_ids - op_ids))
    print("Vehicles in TTE but not in operational:", len(tte_ids - op_ids))
    print("Vehicles in specifications but not in TTE:", len(spec_ids - tte_ids))
    print("Vehicles in TTE but not in specifications:", len(tte_ids - spec_ids))

# merging datasets that only have 1 row per vehicle i.e., specifications and time to event datasets
def merge_vehicle_level_data(
    spec_df: pd.DataFrame,
    tte_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge specifications and TTE at vehicle level.
    One row per vehicle is expected in both inputs.
    """
    if spec_df["vehicle_id"].duplicated().any():
        raise ValueError("Duplicate vehicle_id values found in specifications data.")

    if tte_df["vehicle_id"].duplicated().any():
        raise ValueError("Duplicate vehicle_id values found in TTE data.")

    vehicle_df = spec_df.merge(
        tte_df,
        on="vehicle_id",
        how="inner", # only those in both dataframes
        validate="one_to_one" # 1 row in left and right dataframes
    )

    print("Merged vehicle-level dataframe shape:", vehicle_df.shape)
    return vehicle_df

