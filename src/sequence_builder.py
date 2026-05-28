"""
Helper functions for constructing vehicle-level sequence representations of operational readouts.

This module builds full multivariate time series trajectories for each vehicle by grouping
and ordering operational readouts according to their observation time. Since vehicles are
associated with varying numbers of readouts, each trajectory is represented as a sequence
of variable length.

As part of sequence construction, both numerical and histogram features are preprocessed
to address missing values inherent in irregularly sampled data. For both feature types,
missing values are first forward-filled within each vehicle trajectory to preserve temporal
continuity. Any remaining missing values at the beginning of a sequence, for which no prior
observations exist, are replaced with zeros.

Following imputation, numerical features are normalized using feature-wise statistics
estimated from the training split only. The same normalization parameters are then applied
to validation and test sequences to ensure consistency and avoid data leakage. In contrast,
histogram features are not standardized due to their non-negative and highly skewed nature.
Instead, a logarithmic transformation (log1p) is applied to compress their scale while
preserving their distributional characteristics. This transformation is deterministic and
is therefore applied identically across all data splits.

In addition to sequence construction, the module computes relevant sequence-level metadata,
such as sequence length and temporal characteristics, which may inform downstream modelling
decisions. The final output consists of structured sequence datasets and associated
vehicle-level information, suitable for both deep sequence models (e.g., RDSM) and
self-supervised learning frameworks.
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from copy import deepcopy

def build_operational_sequences(
    operational_df: pd.DataFrame,
    numerical_feature_cols: list[str],
    histogram_feature_cols: list[str]
) -> dict:
    """
    Build per-vehicle operational sequences from the sorted operational dataframe.

    Returns a dictionary:
        sequences[vehicle_id] = {
            "vehicle_id": int,
            "time_steps": np.ndarray,
            "time_gaps": np.ndarray,
            "numerical_sequence": np.ndarray,
            "histogram_sequence": np.ndarray,
            "sequence_length": int,
        }
    """
    sequences = {}

    for vehicle_id, group in operational_df.groupby("vehicle_id", sort=False):
        time_steps = group["time_step"].to_numpy(dtype=float)
        time_gaps = np.diff(time_steps, prepend=time_steps[0])
        time_gaps[0] = 0.0

        numerical_sequence = group[numerical_feature_cols].to_numpy(dtype=float)
        histogram_sequence = group[histogram_feature_cols].to_numpy(dtype=float)

        sequences[vehicle_id] = {
            "vehicle_id": vehicle_id,
            "time_steps": time_steps,
            "time_gaps": time_gaps,
            "numerical_sequence": numerical_sequence,
            "histogram_sequence": histogram_sequence,
            "sequence_length": len(group),
        }

    print("Built operational sequences.")
    print("Vehicle sequence count:", len(sequences))

    return sequences
    
# Function for a truncation step

def truncate_sequence_at_random_readout(
    seq: dict,
    min_history_points: int = 5,
    rng: np.random.Generator | None = None,
) -> dict | None:
    """
    Truncate one sequence at a random readout and redefine duration
    as remaining time-to-event from that readout.
    """
    if rng is None:
        rng = np.random.default_rng()

    seq_new = deepcopy(seq)

    time_steps = np.asarray(seq_new["time_steps"], dtype=float)
    n_obs = len(time_steps)

    if n_obs < min_history_points:
        return None

    possible_indices = np.arange(min_history_points - 1, n_obs)
    if len(possible_indices) == 0:
        return None

    readout_idx = int(rng.choice(possible_indices))
    readout_time = float(time_steps[readout_idx])

    remaining_duration = float(seq_new["duration"]) - readout_time
    if remaining_duration <= 0:
        return None

    keep = np.arange(n_obs) <= readout_idx

    seq_new["time_steps"] = np.asarray(seq_new["time_steps"])[keep]
    
    truncated_time_steps = seq_new["time_steps"]
    truncated_time_gaps = np.diff(truncated_time_steps, prepend=truncated_time_steps[0])
    truncated_time_gaps[0] = 0.0
    seq_new["time_gaps"] = truncated_time_gaps
    
    seq_new["numerical_sequence"] = np.asarray(seq_new["numerical_sequence"])[keep]
    seq_new["histogram_sequence"] = np.asarray(seq_new["histogram_sequence"])[keep]

    seq_new["sequence_length"] = int(keep.sum())
    seq_new["readout_time"] = readout_time
    seq_new["original_duration"] = float(seq["duration"])
    seq_new["duration"] = remaining_duration
    seq_new["event"] = int(seq["event"])

    return seq_new

# Apply truncation to all sequences

def apply_random_readout_to_sequences(
    enriched_sequences: dict,
    min_history_points: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Apply random-readout truncation to a dictionary of vehicle sequences.
    """
    rng = np.random.default_rng(random_state)

    truncated_sequences = {}
    skipped = 0

    for vehicle_id, seq in enriched_sequences.items():
        truncated_seq = truncate_sequence_at_random_readout(
            seq=seq,
            min_history_points=min_history_points,
            rng=rng,
        )
        if truncated_seq is None:
            skipped += 1
        else:
            truncated_sequences[vehicle_id] = truncated_seq

    print(f"Kept {len(truncated_sequences)} truncated sequences.")
    print(f"Skipped {skipped} sequences.")

    return truncated_sequences

# Checking the sequence length

def summarize_sequence_lengths(operational_sequences: dict) -> pd.Series:
    """
    Summarize sequence lengths across all vehicles.
    """
    lengths = [seq["sequence_length"] for seq in operational_sequences.values()]
    length_series = pd.Series(lengths, name="sequence_length")

    summary = length_series.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

    print("Sequence-length summary:")
    print(summary)

    return summary


def sequence_length_table(operational_sequences: dict) -> pd.DataFrame:
    """
    Return a dataframe with vehicle_id and sequence_length.
    """
    rows = [
        {"vehicle_id": vehicle_id, "sequence_length": seq["sequence_length"]}
        for vehicle_id, seq in operational_sequences.items()
    ]

    length_df = pd.DataFrame(rows)
    print("Sequence-length table shape:", length_df.shape)

    return length_df

def attach_vehicle_level_data_to_sequences(
    operational_sequences: dict,
    vehicle_df: pd.DataFrame
) -> dict:
    """
    Attach static features and survival labels to each vehicle sequence.

    Expected columns in vehicle_df:
    - vehicle_id
    - Spec_0 ... Spec_7
    - length_of_study_time_step
    - in_study_repair
    """
    enriched_sequences = {}

    vehicle_records = vehicle_df.set_index("vehicle_id").to_dict(orient="index")
    """
    creates a dictionnary lookup for faster and simpler checks.
    instead of repeatedly filtering the dataframe, one can just look for values of
    each vehicle ID by only calling the dictionnary and the key (vehicle_id)
    """

    missing_vehicle_ids = [] # to store different vehicle IDs from both dataframes if any
    attached_count = 0 # counter to be incremented after each successful enrichment

    for vehicle_id, seq in operational_sequences.items():
        if vehicle_id not in vehicle_records:
            missing_vehicle_ids.append(vehicle_id)
            continue

        vehicle_info = vehicle_records[vehicle_id]

        static_feature_cols = [col for col in vehicle_df.columns if col.startswith("Spec_")]

        enriched_sequences[vehicle_id] = {
            **seq, # dictionary unpacking: take all the existing key-value pairs in this dict and put them here
            "static_features": {col: vehicle_info[col] for col in static_feature_cols}, # nested dict with only static spec vars
            "duration": vehicle_info["length_of_study_time_step"],
            "event": vehicle_info["in_study_repair"],
        }

        attached_count += 1

    print("Sequences enriched with vehicle-level data:", attached_count)
    print("Vehicle IDs missing from vehicle_df:", len(missing_vehicle_ids))

    return enriched_sequences

def prepare_sequence_dataset(
    sequence_data,
    max_seq_len,
    numerical_scaler,
    histogram_scaler,
    time_gap_scaler
):
    """
    Convert variable-length vehicle sequences into equal-length padded arrays.

    Returns
    -------
    dict with:
        numerical_array      : (N, L, D_num)
        histogram_array      : (N, L, D_hist)
        time_gap_array       : (N, L, 1)
        mask_array           : (N, L)
        duration_array       : (N,)
        event_array          : (N,)
        vehicle_id_array     : (N,)
    """
    n_samples = len(sequence_data)

    n_num_features = sequence_data[0]["numerical_sequence"].shape[1]
    n_hist_features = sequence_data[0]["histogram_sequence"].shape[1]

    numerical_array = np.zeros((n_samples, max_seq_len, n_num_features), dtype=np.float32)
    histogram_array = np.zeros((n_samples, max_seq_len, n_hist_features), dtype=np.float32)
    time_gap_array = np.zeros((n_samples, max_seq_len, 1), dtype=np.float32)
    mask_array = np.zeros((n_samples, max_seq_len), dtype=np.float32)

    duration_array = np.zeros(n_samples, dtype=np.float32)
    event_array = np.zeros(n_samples, dtype=np.int64)
    vehicle_id_array = np.zeros(n_samples, dtype=np.int64)

    for i, seq in enumerate(sequence_data):
        numerical_seq = seq["numerical_sequence"]
        histogram_seq = seq["histogram_sequence"]
        time_gaps = seq["time_gaps"].reshape(-1, 1)

        # scale
        numerical_seq = numerical_scaler.transform(numerical_seq)
        histogram_seq = histogram_scaler.transform(histogram_seq)
        time_gaps = time_gap_scaler.transform(time_gaps)

        seq_len = len(time_gaps)

        # truncate: keep most recent time steps
        if seq_len > max_seq_len:
            numerical_seq = numerical_seq[-max_seq_len:]
            histogram_seq = histogram_seq[-max_seq_len:]
            time_gaps = time_gaps[-max_seq_len:]
            seq_len = max_seq_len

        # right-align valid observations
        numerical_array[i, -seq_len:, :] = numerical_seq
        histogram_array[i, -seq_len:, :] = histogram_seq
        time_gap_array[i, -seq_len:, :] = time_gaps
        mask_array[i, -seq_len:] = 1.0

        duration_array[i] = seq["duration"]
        event_array[i] = seq["event"]
        vehicle_id_array[i] = seq["vehicle_id"]

    return {
        "numerical_array": numerical_array,
        "histogram_array": histogram_array,
        "time_gap_array": time_gap_array,
        "mask_array": mask_array,
        "duration_array": duration_array,
        "event_array": event_array,
        "vehicle_id_array": vehicle_id_array,
    }

# Sanity check helper function

def inspect_truncated_sequence(original_seq: dict, truncated_seq: dict) -> None:
    print("Vehicle ID:", original_seq["vehicle_id"])
    print("Original length:", len(original_seq["time_steps"]))
    print("Truncated length:", len(truncated_seq["time_steps"]))
    print("Original final time:", original_seq["time_steps"][-1])
    print("Readout time:", truncated_seq["readout_time"])
    print("Truncated final time:", truncated_seq["time_steps"][-1])
    print("Original duration:", original_seq["duration"])
    print("Remaining duration:", truncated_seq["duration"])
    print("Event preserved:", original_seq["event"] == truncated_seq["event"])
    print(
        "Duration check:",
        np.isclose(
            truncated_seq["duration"],
            original_seq["duration"] - truncated_seq["readout_time"]
        )
    )

import numpy as np


def forward_fill_sequence_numerical(
    numerical: np.ndarray,
) -> np.ndarray:
    """
    Forward-fill missing values in a 2D numerical sequence array.
    """
    x = np.asarray(numerical, dtype=np.float64).copy()

    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")

    T, D = x.shape

    for d in range(D):
        last_val = np.nan
        for t in range(T):
            if np.isnan(x[t, d]):
                if not np.isnan(last_val):
                    x[t, d] = last_val
            else:
                last_val = x[t, d]

    return x


def zero_fill_remaining_sequence_numerical(
    numerical: np.ndarray,
) -> np.ndarray:
    """
    Replace any remaining NaNs with zero.
    """
    x = np.asarray(numerical, dtype=np.float64).copy()
    x = np.nan_to_num(x, nan=0.0)
    return x


def impute_sequence_numerical(
    numerical: np.ndarray,
) -> np.ndarray:
    """
    Apply forward fill followed by zero fill.
    """
    x = forward_fill_sequence_numerical(numerical)
    x = zero_fill_remaining_sequence_numerical(x)

    if np.isnan(x).any():
        raise ValueError("NaNs remain after imputation.")

    return x

def impute_sequence_dict(
    sequence_dict: dict[str, dict],
    numerical_key: str = "numerical_sequence",
) -> dict[str, dict]:
    """
    Apply numerical imputation to all sequences in a dictionary.
    """
    out = {}

    for vid, seq in sequence_dict.items():
        seq_copy = seq.copy()

        if numerical_key not in seq_copy:
            raise KeyError(f"Missing key '{numerical_key}' in sequence {vid}")

        numerical = seq_copy[numerical_key]
        numerical = impute_sequence_numerical(numerical)
        seq_copy[numerical_key] = numerical

        out[vid] = seq_copy

    return out

def fit_sequence_normalizer(
    sequence_dict: dict[str, dict],
    numerical_key: str = "numerical_sequence",
) -> dict[str, np.ndarray]:
    """
    Fit feature-wise mean and std from training sequences.
    """
    all_values = []

    for vid, seq in sequence_dict.items():
        x = np.asarray(seq[numerical_key], dtype=np.float64)

        if np.isnan(x).any():
            raise ValueError(f"NaNs found in sequence {vid}. Impute before normalization.")

        all_values.append(x)

    all_values = np.vstack(all_values)

    means = all_values.mean(axis=0)
    stds = all_values.std(axis=0)
    stds = np.where(stds == 0.0, 1.0, stds)

    return {"means": means, "stds": stds}

def normalize_sequence_dict(
    sequence_dict: dict[str, dict],
    normalizer: dict[str, np.ndarray]
) -> dict[str, dict]:
    """
    Normalize numerical features in all sequences.

    Parameters
    ----------
    sequence_dict : dict
    normalizer : dict with "means" and "stds"

    Returns
    -------
    dict
    """
    means = normalizer["means"]
    stds = normalizer["stds"]

    out = {}

    for vid, seq in sequence_dict.items():
        seq_copy = seq.copy()

        numerical = seq_copy["numerical_sequence"]

        if np.isnan(numerical).any():
            raise ValueError("NaNs found before normalization.")

        numerical = (numerical - means) / stds

        seq_copy["numerical_sequence"] = numerical
        out[vid] = seq_copy

    return out

# For histogram features, log transformation to preserve accumulation and extreme values

def forward_fill_sequence_histogram(
    histogram: np.ndarray,
) -> np.ndarray:
    """
    Forward-fill missing values in a 2D histogram sequence array.

    Parameters
    ----------
    histogram : np.ndarray of shape (T, D_hist)
        Sequence of histogram features.

    Returns
    -------
    np.ndarray
        Forward-filled histogram array.
    """
    x = np.asarray(histogram, dtype=np.float64).copy()

    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")

    T, D = x.shape

    for d in range(D):
        last_val = np.nan
        for t in range(T):
            if np.isnan(x[t, d]):
                if not np.isnan(last_val):
                    x[t, d] = last_val
            else:
                last_val = x[t, d]

    return x


def zero_fill_remaining_sequence_histogram(
    histogram: np.ndarray,
) -> np.ndarray:
    """
    Replace any remaining NaNs in a histogram sequence with zero.

    This is intended to handle leading missing values that remain after
    forward filling.

    Parameters
    ----------
    histogram : np.ndarray of shape (T, D_hist)

    Returns
    -------
    np.ndarray
        Histogram sequence with remaining NaNs replaced by 0.0.
    """
    x = np.asarray(histogram, dtype=np.float64).copy()
    x = np.nan_to_num(x, nan=0.0)
    return x


def impute_sequence_histogram(
    histogram: np.ndarray,
) -> np.ndarray:
    """
    Apply forward fill followed by zero fill to a histogram sequence.

    Parameters
    ----------
    histogram : np.ndarray of shape (T, D_hist)

    Returns
    -------
    np.ndarray
        Imputed histogram sequence.
    """
    x = forward_fill_sequence_histogram(histogram)
    x = zero_fill_remaining_sequence_histogram(x)

    if np.isnan(x).any():
        raise ValueError("NaNs remain after histogram imputation.")

    return x


def log_transform_sequence_histogram(
    histogram: np.ndarray,
) -> np.ndarray:
    """
    Apply log1p transform to a histogram sequence.

    Parameters
    ----------
    histogram : np.ndarray of shape (T, D_hist)
        Histogram sequence. Values are expected to be nonnegative.

    Returns
    -------
    np.ndarray
        Log-transformed histogram sequence.
    """
    x = np.asarray(histogram, dtype=np.float64).copy()

    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")

    if np.isnan(x).any():
        raise ValueError(
            "Histogram sequence contains NaNs. Impute before log transform."
        )

    if (x < 0).any():
        raise ValueError(
            "Histogram sequence contains negative values. "
            "log1p transform requires nonnegative inputs."
        )

    x = np.log1p(x)
    return x

def impute_histogram_sequence_dict(
    sequence_dict: dict[str, dict],
    histogram_key: str = "histogram_sequence",
) -> dict[str, dict]:
    """
    Apply histogram imputation to all sequences in a dictionary.

    Parameters
    ----------
    sequence_dict : dict[str, dict]
        Mapping vehicle_id -> sequence dict.
    histogram_key : str, default="histogram_sequence"
        Key containing the histogram sequence array.

    Returns
    -------
    dict[str, dict]
        New dictionary with imputed histogram sequences.
    """
    out = {}

    for vid, seq in sequence_dict.items():
        seq_copy = seq.copy()

        if histogram_key not in seq_copy:
            raise KeyError(f"Missing key '{histogram_key}' in sequence {vid}")

        histogram = seq_copy[histogram_key]
        histogram = impute_sequence_histogram(histogram)
        seq_copy[histogram_key] = histogram

        out[vid] = seq_copy

    return out


def log_transform_histogram_sequence_dict(
    sequence_dict: dict[str, dict],
    histogram_key: str = "histogram_sequence",
) -> dict[str, dict]:
    """
    Apply log1p transform to histogram sequences in all sequences in a dictionary.

    Parameters
    ----------
    sequence_dict : dict[str, dict]
        Mapping vehicle_id -> sequence dict.
    histogram_key : str, default="histogram_sequence"
        Key containing the histogram sequence array.

    Returns
    -------
    dict[str, dict]
        New dictionary with log-transformed histogram sequences.
    """
    out = {}

    for vid, seq in sequence_dict.items():
        seq_copy = seq.copy()

        if histogram_key not in seq_copy:
            raise KeyError(f"Missing key '{histogram_key}' in sequence {vid}")

        histogram = seq_copy[histogram_key]
        histogram = log_transform_sequence_histogram(histogram)
        seq_copy[histogram_key] = histogram

        out[vid] = seq_copy

    return out

# full preprocessing pipeline
def preprocess_sequence_splits(
    train_seq: dict[str, dict],
    val_seq: dict[str, dict],
    test_seq: dict[str, dict],
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict], dict[str, np.ndarray]]:
    """
    Full preprocessing pipeline for sequence dictionaries.

    Steps
    -----
    1. Impute numerical sequences:
       - forward fill
       - zero fill remaining leading NaNs

    2. Impute histogram sequences:
       - forward fill
       - zero fill remaining leading NaNs

    3. Fit numerical normalizer on training sequences only.

    4. Apply numerical normalization to all splits.

    5. Apply log1p transform to histogram sequences in all splits.

    Parameters
    ----------
    train_seq : dict[str, dict]
        Training sequence dictionary.
    val_seq : dict[str, dict]
        Validation sequence dictionary.
    test_seq : dict[str, dict]
        Test sequence dictionary.

    Returns
    -------
    tuple[dict[str, dict], dict[str, dict], dict[str, dict], dict[str, np.ndarray]]
        Preprocessed training, validation, and test sequence dictionaries,
        together with the fitted numerical normalizer.
    """
    # Numerical imputation
    train_imp = impute_sequence_dict(train_seq)
    val_imp = impute_sequence_dict(val_seq)
    test_imp = impute_sequence_dict(test_seq)

    # Histogram imputation
    train_imp = impute_histogram_sequence_dict(train_imp)
    val_imp = impute_histogram_sequence_dict(val_imp)
    test_imp = impute_histogram_sequence_dict(test_imp)

    # Fit numerical normalizer on train only
    normalizer = fit_sequence_normalizer(train_imp)

    # Numerical normalization
    train_norm = normalize_sequence_dict(train_imp, normalizer)
    val_norm = normalize_sequence_dict(val_imp, normalizer)
    test_norm = normalize_sequence_dict(test_imp, normalizer)

    # Histogram log transform
    train_norm = log_transform_histogram_sequence_dict(train_norm)
    val_norm = log_transform_histogram_sequence_dict(val_norm)
    test_norm = log_transform_histogram_sequence_dict(test_norm)

    return train_norm, val_norm, test_norm, normalizer
from sklearn.preprocessing import OneHotEncoder

def save_sequence_artifacts(
    train_seq: dict,
    val_seq: dict,
    test_seq: dict,
    normalizer: dict,
    encoder: OneHotEncoder,
    output_dir: str | Path,
):
    """
    Save processed sequence datasets, normalizer and encoder.

    Parameters
    ----------
    train_seq : dict
    val_seq : dict
    test_seq : dict
    normalizer : dict
    encoder: dict
    output_dir : str or Path
        Directory where artifacts will be saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save sequences
    with open(output_dir / "train_sequences_full.pkl", "wb") as f:
        pickle.dump(train_seq, f)

    with open(output_dir / "val_sequences_full.pkl", "wb") as f:
        pickle.dump(val_seq, f)

    with open(output_dir / "test_sequences_full.pkl", "wb") as f:
        pickle.dump(test_seq, f)
    # Save normalizer
    with open(output_dir / "full_sequence_normalizer.pkl", "wb") as f:
        pickle.dump(normalizer, f)
    # Save encoder
    with open(output_dir / "full_sequence_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    print(f"SSL sequences, normalizer and encoder saved to: {output_dir}")


def save_truncated_sequence_artifacts(
    train_trunc_seq: dict,
    val_trunc_seq: dict,
    test_trunc_seq: dict,
    normalizer_trunc: dict,
    encoder_trunc: OneHotEncoder,
    output_dir: str | Path,
):
    """
    Save processed sequence datasets and normalizer.

    Parameters
    ----------
    train_seq : dict
    val_seq : dict
    test_seq : dict
    normalizer : dict
    encoder: dict
    output_dir : str or Path
        Directory where artifacts will be saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save sequences
    with open(output_dir / "train_sequences_truncated.pkl", "wb") as f:
        pickle.dump(train_trunc_seq, f)

    with open(output_dir / "val_sequences_truncated.pkl", "wb") as f:
        pickle.dump(val_trunc_seq, f)

    with open(output_dir / "test_sequences_truncated.pkl", "wb") as f:
        pickle.dump(test_trunc_seq, f)

    # Save normalizer
    with open(output_dir / "truncated_sequence_normalizer.pkl", "wb") as f:
        pickle.dump(normalizer_trunc, f)
    # Save encoder
    with open(output_dir / "truncated_sequence_encoder.pkl", "wb") as f:
        pickle.dump(encoder_trunc, f)

    print(f"Truncated sequences, normalizer and encoder saved to: {output_dir}")


def load_sequence_artifacts(output_dir: str | Path):
    """
    Load processed sequence datasets and normalizer.
    """
    output_dir = Path(output_dir)

    with open(output_dir / "train_sequences.pkl", "rb") as f:
        train_seq = pickle.load(f)

    with open(output_dir / "val_sequences.pkl", "rb") as f:
        val_seq = pickle.load(f)

    with open(output_dir / "test_sequences.pkl", "rb") as f:
        test_seq = pickle.load(f)

    with open(output_dir / "sequence_normalizer.pkl", "rb") as f:
        normalizer = pickle.load(f)

    return train_seq, val_seq, test_seq, normalizer


def build_time_and_frequency_sequences(
    sequence_dicts
):
    """
    Build time-branch and frequency-branch dynamic sequences
    from the existing per-vehicle sequence dictionaries.

    Parameters
    ----------
    sequence_dicts : dict or list
        Either:
        - dict[vehicle_id -> sequence_dict]
        - list[sequence_dict]

    Returns
    -------
    same type as input
        Sequence dictionaries with added keys:
        - dynamic_sequence_time
        - dynamic_sequence_freq
        - numerical_sequence_freq
        - histogram_sequence_freq
    """

    input_is_dict = isinstance(sequence_dicts, dict)

    if input_is_dict:
        items = sequence_dicts.items()
        output = {}
    else:
        items = enumerate(sequence_dicts)
        output = []

    for key, seq in items:
        seq_out = seq.copy()

        x_num = seq["numerical_sequence"]
        x_hist = seq["histogram_sequence"]

        if not isinstance(x_num, np.ndarray) or x_num.ndim != 2:
            raise ValueError("numerical_sequence must be a 2D numpy array.")
        if not isinstance(x_hist, np.ndarray) or x_hist.ndim != 2:
            raise ValueError("histogram_sequence must be a 2D numpy array.")
        if x_num.shape[0] != x_hist.shape[0]:
            raise ValueError("numerical_sequence and histogram_sequence must have the same number of time steps.")

        # -------------------------
        # Time branch
        # -------------------------
        dynamic_sequence_time = np.concatenate([x_num, x_hist], axis=1)

        # -------------------------
        # Frequency branch
        # -------------------------
        x_num_diff = np.zeros_like(x_num)
        x_hist_diff = np.zeros_like(x_hist)

        x_num_diff[1:] = x_num[1:] - x_num[:-1]
        x_hist_diff[1:] = x_hist[1:] - x_hist[:-1]

        dynamic_sequence_freq = np.concatenate([x_num_diff, x_hist_diff], axis=1)

        # Store
        seq_out["numerical_sequence_freq"] = x_num_diff
        seq_out["histogram_sequence_freq"] = x_hist_diff
        seq_out["dynamic_sequence_time"] = dynamic_sequence_time
        seq_out["dynamic_sequence_freq"] = dynamic_sequence_freq

        if input_is_dict:
            output[key] = seq_out
        else:
            output.append(seq_out)

    return output


def fit_time_gap_scaler(sequence_dicts):
    """
    Fit mean/std on log1p(time_gaps) using training sequences only.
    """
    if isinstance(sequence_dicts, dict):
        seqs = sequence_dicts.values()
    else:
        seqs = sequence_dicts

    all_gap_values = []

    for seq in seqs:
        gaps = np.asarray(seq["time_gaps"], dtype=np.float32)
        all_gap_values.append(np.log1p(gaps))

    all_gap_values = np.concatenate(all_gap_values, axis=0)

    mean_ = all_gap_values.mean()
    std_ = all_gap_values.std()

    if std_ < 1e-8:
        std_ = 1.0

    return {"mean": float(mean_), "std": float(std_)}


def transform_time_gaps(sequence_dicts, scaler):
    """
    Add scaled_time_gaps to each sequence dictionary.
    """
    input_is_dict = isinstance(sequence_dicts, dict)

    if input_is_dict:
        items = sequence_dicts.items()
        output = {}
    else:
        items = enumerate(sequence_dicts)
        output = []

    mean_ = scaler["mean"]
    std_ = scaler["std"]

    for key, seq in items:
        seq_out = seq.copy()

        gaps = np.asarray(seq["time_gaps"], dtype=np.float32)
        gaps_log = np.log1p(gaps)
        gaps_scaled = (gaps_log - mean_) / std_

        seq_out["time_gaps_scaled"] = gaps_scaled.astype(np.float32)

        if input_is_dict:
            output[key] = seq_out
        else:
            output.append(seq_out)

    return output

def prepare_ssl_sequences(train_seq, val_seq=None, test_seq=None):
    """
    Prepare time/frequency branches and scaled time gaps.
    """

    train_seq_out = build_time_and_frequency_sequences(train_seq)
    gap_scaler = fit_time_gap_scaler(train_seq_out)
    train_seq_out = transform_time_gaps(train_seq_out, gap_scaler)

    val_seq_out = None
    test_seq_out = None

    if val_seq is not None:
        val_seq_out = build_time_and_frequency_sequences(val_seq)
        val_seq_out = transform_time_gaps(val_seq_out, gap_scaler)

    if test_seq is not None:
        test_seq_out = build_time_and_frequency_sequences(test_seq)
        test_seq_out = transform_time_gaps(test_seq_out, gap_scaler)

    return train_seq_out, val_seq_out, test_seq_out, gap_scaler