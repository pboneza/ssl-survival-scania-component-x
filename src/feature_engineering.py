"""
Create feature representations for non-sequential models.
    Aggregation scheme:
    - numerical features: first, last, mean, std, min, max, delta, slope
      using first/last non-missing endpoints for endpoint-based summaries
    - histogram features: mean, last non-missing)
    - time gaps: mean, std, min, max, median
    - sequence metadata: sequence_length, final_time_step, observation_density
    - static features: raw Spec_* values
    - targets: duration, event
"""

import numpy as np
import pandas as pd


def _first_valid_value(values: np.ndarray) -> float:
    """
    Return the first non-missing value in a 1D array.
    Returns np.nan if all values are missing.
    """
    valid_idx = np.where(~np.isnan(values))[0] # gives a false condition to np.where() since it returns values where the condition is true.
    if len(valid_idx) == 0: # if there are no valid values at all
        return np.nan 
    return float(values[valid_idx[0]]) # but if there valid values, return the first one.


def _last_valid_value(values: np.ndarray) -> float:
    """
    Return the last non-missing value in a 1D array.
    Returns np.nan if all values are missing.
    """
    valid_idx = np.where(~np.isnan(values))[0]
    if len(valid_idx) == 0:
        return np.nan
    return float(values[valid_idx[-1]]) # same as above, but returns the last one.


def _valid_delta(values: np.ndarray) -> float:
    """
    Compute last valid minus first valid.
    Returns np.nan if fewer than one valid value exists.
    """
    valid_idx = np.where(~np.isnan(values))[0]
    if len(valid_idx) == 0:
        return np.nan

    first_val = values[valid_idx[0]]
    last_val = values[valid_idx[-1]]
    return float(last_val - first_val)


def _valid_slope(values: np.ndarray, time_steps: np.ndarray) -> float:
    """
    Compute slope using first and last non-missing values and their corresponding times.
    Returns np.nan if fewer than two valid values exist or if time span is zero.
    """
    valid_idx = np.where(~np.isnan(values))[0]
    if len(valid_idx) < 2:
        return np.nan

    first_idx = valid_idx[0]
    last_idx = valid_idx[-1]

    first_val = values[first_idx]
    last_val = values[last_idx]

    first_time = time_steps[first_idx]
    last_time = time_steps[last_idx]

    time_span = last_time - first_time
    if time_span == 0:
        return np.nan

    return float((last_val - first_val) / time_span)


def aggregate_sequences_for_tabular_baselines(
    enriched_sequences: dict,
    numerical_feature_cols: list[str],
    histogram_feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate enriched vehicle sequences into one row per vehicle
    for CPH and RSF models.

    Returns
    -------
    predictors_df : pd.DataFrame
        Aggregated predictor matrix with one row per vehicle.
    targets_df : pd.DataFrame
        Target dataframe containing duration and event columns.
        
    Aggregation scheme:
    - numerical features: first, last, mean, std, min, max, delta, slope
      using first/last non-missing endpoints for endpoint-based summaries
    - histogram features: mean, last (last non-missing)
    - time gaps: mean, std, min, max, median
    - sequence metadata: sequence_length, final_time_step, observation_density
    - static features: raw Spec_* values
    - targets: duration, event
    """
    rows = [] # list that will store one dictionary of summary statistics per vehicle

    for vehicle_id, seq in enriched_sequences.items():
        row = {"vehicle_id": vehicle_id} # starts as a dictionary with only vehicle ID
        # pull out the main sequence information and convert them to numpy arrays
        time_steps = np.asarray(seq["time_steps"], dtype=float)
        time_gaps = np.asarray(seq["time_gaps"], dtype=float)
        numerical_sequence = np.asarray(seq["numerical_sequence"], dtype=float)
        histogram_sequence = np.asarray(seq["histogram_sequence"], dtype=float)

        # Numerical feature summaries
        for i, feature_name in enumerate(numerical_feature_cols): # i: column index, plus its name 
            values = numerical_sequence[:, i] 
            # use the above functions to retrieve valid values
            row[f"{feature_name}_first"] = _first_valid_value(values)
            row[f"{feature_name}_last"] = _last_valid_value(values)
            row[f"{feature_name}_delta"] = _valid_delta(values)
            row[f"{feature_name}_slope"] = _valid_slope(values, time_steps)
            # compute summary statistics ignoring the missing values
            row[f"{feature_name}_mean"] = float(np.nanmean(values)) if np.any(~np.isnan(values)) else np.nan
            row[f"{feature_name}_std"] = float(np.nanstd(values)) if np.any(~np.isnan(values)) else np.nan
            row[f"{feature_name}_min"] = float(np.nanmin(values)) if np.any(~np.isnan(values)) else np.nan
            row[f"{feature_name}_max"] = float(np.nanmax(values)) if np.any(~np.isnan(values)) else np.nan
            
        # Histogram feature summaries
        for j, feature_name in enumerate(histogram_feature_cols):
            values = histogram_sequence[:, j]

            row[f"{feature_name}_mean"] = float(np.nanmean(values)) if np.any(~np.isnan(values)) else np.nan
            row[f"{feature_name}_last"] = _last_valid_value(values)

        # Time-gap summaries
        valid_gaps = time_gaps[1:] if len(time_gaps) > 1 else time_gaps

        row["gap_mean"] = float(np.nanmean(valid_gaps)) if np.any(~np.isnan(valid_gaps)) else np.nan
        row["gap_std"] = float(np.nanstd(valid_gaps)) if np.any(~np.isnan(valid_gaps)) else np.nan
        row["gap_min"] = float(np.nanmin(valid_gaps)) if np.any(~np.isnan(valid_gaps)) else np.nan
        row["gap_max"] = float(np.nanmax(valid_gaps)) if np.any(~np.isnan(valid_gaps)) else np.nan
        row["gap_median"] = float(np.nanmedian(valid_gaps)) if np.any(~np.isnan(valid_gaps)) else np.nan

        # Sequence metadata, how may observations per unit of time?
        row["sequence_length"] = int(seq["sequence_length"])
        row["final_time_step"] = float(time_steps[-1])
        row["observation_density"] = (
            float(seq["sequence_length"] / time_steps[-1]) if time_steps[-1] != 0 else 0.0
        )

        # Static features
        for spec_name, spec_value in seq["static_features"].items():
            row[spec_name] = spec_value

        # Targets
        row["duration"] = float(seq["duration"])
        row["event"] = int(seq["event"])

        rows.append(row)

    aggregated_df = pd.DataFrame(rows)
    targets_df = aggregated_df[["duration", "event"]].copy()
    predictors_df = aggregated_df.drop(columns=["duration", "event"]).copy()

    print("Aggregated predictors dataframe shape:", predictors_df.shape)
    print("Aggregated targets dataframe shape:", targets_df.shape)
    print("Vehicle count:", aggregated_df["vehicle_id"].nunique())

    return predictors_df, targets_df


def summarize_missingness(df: pd.DataFrame, top_n: int = 30) -> pd.Series:
    """
    Summarize missing values per column and print the top columns with most missingness.
    """
    missing_counts = df.isna().sum().sort_values(ascending=False)
    nonzero_missing = missing_counts[missing_counts > 0]

    print("Columns with missing values:", nonzero_missing.shape[0])
    print(f"Top {top_n} columns with most missing values:")
    print(nonzero_missing.head(top_n))

    return nonzero_missing