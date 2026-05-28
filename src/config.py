# config.py
"""
Central configuration file for project paths.

This file stores the main directory and file paths used across the project.
It avoids hardcoded paths in notebooks and scripts, making the experimental
pipeline easier to reproduce on another machine.
"""

from pathlib import Path


# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

# Project root = parent directory of the src folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# ---------------------------------------------------------------------
# Results directories
# ---------------------------------------------------------------------

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
LOGS_DIR = RESULTS_DIR / "logs"


# ---------------------------------------------------------------------
# Processed data subdirectories
# ---------------------------------------------------------------------

SEQUENCE_DIR = PROCESSED_DATA_DIR / "sequences"
TRUNCATED_DIR = PROCESSED_DATA_DIR / "tabular_truncated"

# ---------------------------------------------------------------------
# Raw input files
# ---------------------------------------------------------------------

OPERATIONAL_FILE = RAW_DATA_DIR / "train_operational_readouts.csv"
SPECIFICATIONS_FILE = RAW_DATA_DIR / "train_specifications.csv"
TTE_FILE = RAW_DATA_DIR / "train_tte.csv"

# ---------------------------------------------------------------------
# Truncated-history tabular files
# ---------------------------------------------------------------------

X_TRAIN_TRUNC_NS_FILE = TRUNCATED_DIR / "X_train_trunc_without_static.csv"
X_VAL_TRUNC_NS_FILE = TRUNCATED_DIR / "X_val_trunc_without_static.csv"
X_TEST_TRUNC_NS_FILE = TRUNCATED_DIR / "X_test_trunc_without_static.csv"

X_TRAIN_TRUNC_WS_FILE = TRUNCATED_DIR / "X_train_trunc_with_static.csv"
X_VAL_TRUNC_WS_FILE = TRUNCATED_DIR / "X_val_trunc_with_static.csv"
X_TEST_TRUNC_WS_FILE = TRUNCATED_DIR / "X_test_trunc_with_static.csv"

Y_TRAIN_TRUNC_FILE = TRUNCATED_DIR / "y_train_trunc.csv"
Y_VAL_TRUNC_FILE = TRUNCATED_DIR / "y_val_trunc.csv"
Y_TEST_TRUNC_FILE = TRUNCATED_DIR / "y_test_trunc.csv"

# ---------------------------------------------------------------------
# Full-history sequence files
# ---------------------------------------------------------------------
X_TRAIN_FULL_RAW_SEQUENCE_FILE = SEQUENCE_DIR / "train_seq_full.pkl"
X_VAL_FULL_RAW_SEQUENCE_FILE = SEQUENCE_DIR / "val_seq_full.pkl"
X_TEST_FULL_RAW_SEQUENCE_FILE = SEQUENCE_DIR / "test_seq_full.pkl"

# ---------------------------------------------------------------------
# Truncated-history sequence files
# ---------------------------------------------------------------------
X_TRAIN_TRUNC_RAW_SEQUENCE_FILE = SEQUENCE_DIR / "train_seq_truncated.pkl"
X_VAL_TRUNC_RAW_SEQUENCE_FILE = SEQUENCE_DIR / "val_seq_truncated.pkl"
X_TEST_TRUNC_RAW_SEQUENCE_FILE = SEQUENCE_DIR / "test_seq_truncated.pkl"
# ---------------------------------------------------------------------
# Utility function
# ---------------------------------------------------------------------

def create_project_directories() -> None:
    """
    Create the main project directories if they do not already exist.
    This is useful when running the pipeline on a new machine.
    """
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        LOGS_DIR,
        SEQUENCE_DIR,
        TRUNCATED_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)