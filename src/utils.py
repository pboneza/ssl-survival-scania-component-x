from pathlib import Path
import json
import pandas as pd


def save_dataframe_csv(df: pd.DataFrame, file_path: Path) -> None:
    """
    Save a dataframe as CSV.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Saved dataframe to: {file_path}")


def save_list_json(items: list, file_path: Path) -> None:
    """
    Save a Python list to JSON.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
    print(f"Saved list to: {file_path}")


def save_dict_json(data: dict, file_path: Path) -> None:
    """
    Save a Python dictionary to JSON.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved dictionary to: {file_path}")