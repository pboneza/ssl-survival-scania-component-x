"""
Performs consistency and integrity checks on the operational readouts dataset (duplicates and monotonicity).
Then computes the time gap feature as it will be needed later as input to modelling, to provide irregular sampling information to the models.
Information on time gaps is inspected to guide modelling decisions further.
Since the dataset article proposes additional processing on numerical counters that might exhibit counter reset due to lost ECU connections,
counter checks are performed to observe the effect on variables.
"""

import pandas as pd
import numpy as np

"""
Group readouts by vehicle_id and sort them per time_step 
(combined in a sort_values function)
"""
def sort_operational_data(operational_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort operational readouts by vehicle_id and time_step.
    Returns a sorted copy of the dataframe.
    """
    sorted_df = operational_df.sort_values(
        by=["vehicle_id", "time_step"],
        ascending=[True, True]
    ).reset_index(drop=True)

    print("Operational data sorted by vehicle_id and time_step.")
    print("Sorted shape:", sorted_df.shape)
    return sorted_df

def check_duplicate_vehicle_time_pairs(operational_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify duplicated (vehicle_id, time_step) pairs in the operational data.
    Returns all duplicated rows for inspection.
    """
    duplicate_mask = operational_df.duplicated(
        subset=["vehicle_id", "time_step"],
        keep=False
    )

    duplicates_df = operational_df.loc[duplicate_mask].copy()

    n_duplicate_rows = duplicates_df.shape[0]
    n_duplicate_pairs = duplicates_df[["vehicle_id", "time_step"]].drop_duplicates().shape[0]

    print("Duplicate row count:", n_duplicate_rows)
    print("Duplicate (vehicle_id, time_step) pair count:", n_duplicate_pairs)

    return duplicates_df

def check_time_monotonicity(operational_df: pd.DataFrame) -> pd.DataFrame:
    """
    Check whether time_step is non-decreasing within each vehicle sequence.
    Returns rows where monotonicity is violated.
    """
    df = operational_df.copy()

    df["prev_time_step"] = df.groupby("vehicle_id")["time_step"].shift(1)
    df["time_diff"] = df["time_step"] - df["prev_time_step"]

    violations_df = df.loc[df["time_diff"] < 0].copy()

    print("Monotonicity violation row count:", violations_df.shape[0])
    print("Affected vehicle count:", violations_df["vehicle_id"].nunique())

    return violations_df

def compute_time_gap_summary(operational_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute time-gap statistics between consecutive readouts within each vehicle.
    Returns the full dataframe of per-row time gaps.
    """
    df = operational_df.copy()

    df["prev_time_step"] = df.groupby("vehicle_id")["time_step"].shift(1)
    df["time_gap"] = df["time_step"] - df["prev_time_step"]

    print("Time-gap column created.")
    print("Rows with observed gap:", df["time_gap"].notna().sum())

    return df


def summarize_time_gaps(gap_df: pd.DataFrame) -> pd.Series:
    """
    Print and return summary statistics for time gaps.
    """
    gap_summary = gap_df["time_gap"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

    print("Time-gap summary statistics:")
    print(gap_summary)

    return gap_summary


def find_large_time_gaps(gap_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Return rows where the gap between consecutive readouts exceeds a chosen threshold.
    """
    large_gaps_df = gap_df.loc[gap_df["time_gap"] > threshold].copy()

    print(f"Rows with time_gap > {threshold}:", large_gaps_df.shape[0])
    print("Affected vehicle count:", large_gaps_df["vehicle_id"].nunique())

    return large_gaps_df

# check if numerical counters have abrupt decreases caused by lost ECU conncection resulting in counter resets
def check_counter_resets(
    operational_df: pd.DataFrame,
    numerical_feature_cols: list[str]
) -> pd.DataFrame:
    """
    Check for decreases in numerical scalar features within each vehicle sequence.
    A negative difference suggests a possible counter reset or inconsistency.

    Returns a long-format dataframe containing all detected decreases.
    """
    reset_records = []

    for feature in numerical_feature_cols:
        prev_col = f"{feature}_prev"
        diff_col = f"{feature}_diff"
        # temporary dataframe for inspections
        temp_df = operational_df[["vehicle_id", "time_step", feature]].copy()
        temp_df[prev_col] = temp_df.groupby("vehicle_id")[feature].shift(1)
        temp_df[diff_col] = temp_df[feature] - temp_df[prev_col]

        feature_resets = temp_df.loc[temp_df[diff_col] < 0, [
            "vehicle_id", "time_step", feature, prev_col, diff_col
        ]].copy()

        if not feature_resets.empty:
            feature_resets["feature"] = feature
            reset_records.append(feature_resets)

    if reset_records:
        resets_df = pd.concat(reset_records, axis=0, ignore_index=True)
    else:
        resets_df = pd.DataFrame(
            columns=["vehicle_id", "time_step", "feature"]
        )

    print("Total reset rows detected:", resets_df.shape[0])

    if not resets_df.empty:
        print("Affected vehicle count:", resets_df["vehicle_id"].nunique())
        print("Affected feature count:", resets_df["feature"].nunique())

    return resets_df


def detect_likely_cumulative_features_from_sequences(
    sequence_dicts,
    sequence_key: str,
    feature_names: list[str],
    tolerance: float = 1e-8,
    min_valid_diffs: int = 50,
    negative_ratio_threshold: float = 0.01,
    strong_positive_ratio_threshold: float = 0.80
) -> pd.DataFrame:
    """
    Detect likely cumulative or non-decreasing features from the existing sequence dictionaries.

    sequence_dicts can be either:
    - list of sequence dictionaries
    - dict mapping vehicle_id -> sequence dictionary
    """

    # Accept both list-of-dicts and dict-of-dicts
    if isinstance(sequence_dicts, dict):
        sequence_iterable = list(sequence_dicts.values())
    else:
        sequence_iterable = sequence_dicts

    n_features = len(feature_names)

    total_diffs = np.zeros(n_features, dtype=np.int64)
    negative_diffs = np.zeros(n_features, dtype=np.int64)
    positive_diffs = np.zeros(n_features, dtype=np.int64)
    zero_diffs = np.zeros(n_features, dtype=np.int64)

    min_diff = np.full(n_features, np.inf, dtype=np.float64)
    max_diff = np.full(n_features, -np.inf, dtype=np.float64)
    mean_abs_diff_sum = np.zeros(n_features, dtype=np.float64)

    for seq_idx, seq in enumerate(sequence_iterable):
        if not isinstance(seq, dict):
            raise TypeError(
                f"Expected each sequence to be a dict, but got {type(seq)} at index {seq_idx}."
            )

        if sequence_key not in seq:
            raise KeyError(f"Sequence {seq_idx} does not contain key '{sequence_key}'.")

        x = seq[sequence_key]

        if not isinstance(x, np.ndarray):
            raise TypeError(
                f"Sequence {seq_idx}, key '{sequence_key}' must contain a numpy array."
            )

        if x.ndim != 2:
            raise ValueError(
                f"Sequence {seq_idx}, key '{sequence_key}' must have shape [T, F]."
            )

        if x.shape[1] != n_features:
            raise ValueError(
                f"Sequence {seq_idx}, key '{sequence_key}' has {x.shape[1]} features, "
                f"but feature_names has length {n_features}."
            )

        if x.shape[0] < 2:
            continue

        diffs = x[1:] - x[:-1]

        total_diffs += diffs.shape[0]

        neg_mask = diffs < -tolerance
        pos_mask = diffs > tolerance
        zero_mask = (~neg_mask) & (~pos_mask)

        negative_diffs += neg_mask.sum(axis=0)
        positive_diffs += pos_mask.sum(axis=0)
        zero_diffs += zero_mask.sum(axis=0)

        min_diff = np.minimum(min_diff, diffs.min(axis=0))
        max_diff = np.maximum(max_diff, diffs.max(axis=0))
        mean_abs_diff_sum += np.abs(diffs).sum(axis=0)

    rows = []

    for j, feature_name in enumerate(feature_names):
        n_total = int(total_diffs[j])

        if n_total == 0:
            negative_ratio = np.nan
            positive_ratio = np.nan
            zero_ratio = np.nan
            mean_abs_diff = np.nan
            local_min_diff = np.nan
            local_max_diff = np.nan
            verdict = "insufficient_data"
        else:
            negative_ratio = negative_diffs[j] / n_total
            positive_ratio = positive_diffs[j] / n_total
            zero_ratio = zero_diffs[j] / n_total
            mean_abs_diff = mean_abs_diff_sum[j] / n_total
            local_min_diff = min_diff[j]
            local_max_diff = max_diff[j]

            if n_total < min_valid_diffs:
                verdict = "insufficient_data"
            elif negative_ratio <= negative_ratio_threshold:
                if positive_ratio >= strong_positive_ratio_threshold:
                    verdict = "likely_cumulative"
                else:
                    verdict = "possibly_cumulative_or_non_decreasing"
            else:
                verdict = "likely_non_cumulative"

        rows.append({
            "feature": feature_name,
            "sequence_key": sequence_key,
            "n_diffs": n_total,
            "negative_ratio": negative_ratio,
            "positive_ratio": positive_ratio,
            "zero_ratio": zero_ratio,
            "min_diff": local_min_diff,
            "max_diff": local_max_diff,
            "mean_abs_diff": mean_abs_diff,
            "verdict": verdict
        })

    summary_df = pd.DataFrame(rows)

    summary_df = summary_df.sort_values(
        by=["verdict", "negative_ratio", "feature"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    return summary_df

def split_features_by_verdict(summary_df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    likely_cumulative = summary_df.loc[
        summary_df["verdict"] == "likely_cumulative", "feature"
    ].tolist()

    possible_cumulative = summary_df.loc[
        summary_df["verdict"] == "possibly_cumulative_or_non_decreasing", "feature"
    ].tolist()

    likely_non_cumulative = summary_df.loc[
        summary_df["verdict"] == "likely_non_cumulative", "feature"
    ].tolist()

    return likely_cumulative, possible_cumulative, likely_non_cumulative