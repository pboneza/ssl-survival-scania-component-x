"""
PyTorch dataset utilities for self-supervised pretraining on sequence prefixes.

This module defines the dataset and batching logic used to feed prefix-based
operational sequences into the SSL pretraining pipeline. It assumes that prefix
samples have already been constructed from full vehicle trajectories during the
sequence preprocessing stage.

Each dataset item corresponds to one prefix sample identified by `prefix_id`.
For every prefix, the ordered numerical and histogram feature sequences are
retrieved, together with their associated time information. Two stochastic
augmented views of the same prefix are then generated on the fly for contrastive
self-supervised learning.

The module contains:

- `SSLPrefixDataset`: 
    A custom PyTorch dataset that groups rows belonging to the same prefix,
    extracts the sequence features, and returns two independently augmented
    views of each prefix sample.

- `ssl_prefix_collate_fn`:
    A custom collate function for batching variable-length prefix sequences.
    It pads the sequences within each batch to the maximum batch length and
    creates a padding mask so that padded positions can be ignored during model
    training.

This design allows prefix construction to remain deterministic, while the
augmentation process remains stochastic across training iterations. As a result,
the same prefix can produce different augmented views across epochs, which is
desirable for contrastive representation learning.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.sequence_preprocessing import create_two_augmented_views
from torch.nn.utils.rnn import pad_sequence


class SSLPrefixDataset(Dataset):
    """
    PyTorch dataset for self-supervised learning on prefix samples.

    Each dataset item corresponds to one prefix sequence stored as a dictionary.
    Two stochastic augmented views are generated on the fly for contrastive
    learning.

    Parameters
    ----------
    prefix_dict : dict[str, dict]
        Dictionary of prefix samples keyed by prefix_id. Each prefix sample is
        expected to contain:
        - "prefix_id"
        - "vehicle_id"
        - "prefix_length"
        - "time_steps"
        - "time_gaps"
        - "numerical_sequence"
        - "histogram_sequence"
    numerical_mask_probability : float, default=0.15
        Masking probability for numerical features.
    numerical_noise_std : float, default=0.01
        Noise standard deviation for numerical features.
    numerical_dropout_probability : float, default=0.10
        Dropout probability for numerical features.
    histogram_mask_probability : float, default=0.15
        Masking probability for histogram features.
    random_seed : int | None, default=None
        Optional seed for reproducibility.
    """

    def __init__(
        self,
        prefix_dict: dict[str, dict],
        numerical_mask_probability: float = 0.15,
        numerical_noise_std: float = 0.01,
        numerical_dropout_probability: float = 0.10,
        histogram_mask_probability: float = 0.15,
        random_seed: int | None = None,
    ) -> None:
        if len(prefix_dict) == 0:
            raise ValueError("prefix_dict must not be empty.")

        required_keys = [
            "prefix_id",
            "vehicle_id",
            "prefix_length",
            "time_steps",
            "time_gaps",
            "numerical_sequence",
            "histogram_sequence",
        ]

        for prefix_id, prefix_sample in prefix_dict.items():
            missing_keys = [key for key in required_keys if key not in prefix_sample]
            if missing_keys:
                raise KeyError(
                    f"Prefix '{prefix_id}' is missing required keys: {missing_keys}"
                )

        self.prefix_dict = prefix_dict
        self.prefix_ids = sorted(prefix_dict.keys())

        self.numerical_mask_probability = numerical_mask_probability
        self.numerical_noise_std = numerical_noise_std
        self.numerical_dropout_probability = numerical_dropout_probability
        self.histogram_mask_probability = histogram_mask_probability

        self.rng = np.random.default_rng(random_seed)

    def __len__(self) -> int:
        return len(self.prefix_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        prefix_id = self.prefix_ids[idx]
        prefix_sample = self.prefix_dict[prefix_id]

        vehicle_id = prefix_sample["vehicle_id"]
        sequence_length = int(prefix_sample["prefix_length"])

        numerical_sequence = np.asarray(
            prefix_sample["numerical_sequence"],
            dtype=np.float32,
        )
        histogram_sequence = np.asarray(
            prefix_sample["histogram_sequence"],
            dtype=np.float32,
        )
        time_steps = np.asarray(
            prefix_sample["time_steps"],
            dtype=np.float32,
        )
        time_gaps = np.asarray(
            prefix_sample["time_gaps"],
            dtype=np.float32,
        )

        view_1, view_2 = create_two_augmented_views(
            numerical_sequence=numerical_sequence,
            histogram_sequence=histogram_sequence,
            numerical_mask_probability=self.numerical_mask_probability,
            numerical_noise_std=self.numerical_noise_std,
            numerical_dropout_probability=self.numerical_dropout_probability,
            histogram_mask_probability=self.histogram_mask_probability,
            rng=self.rng,
        )

        sample = {
            "prefix_id": prefix_id,
            "vehicle_id": vehicle_id,
            "sequence_length": sequence_length,
            "time_steps": torch.tensor(time_steps, dtype=torch.float32),
            "time_gaps": torch.tensor(time_gaps, dtype=torch.float32),
            "view_1_numerical": torch.tensor(view_1["numerical"], dtype=torch.float32),
            "view_1_histogram": torch.tensor(view_1["histogram"], dtype=torch.float32),
            "view_2_numerical": torch.tensor(view_2["numerical"], dtype=torch.float32),
            "view_2_histogram": torch.tensor(view_2["histogram"], dtype=torch.float32),
        }

        return sample

from torch.nn.utils.rnn import pad_sequence


def ssl_prefix_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function for batching variable-length SSL prefix samples.

    Parameters
    ----------
    batch : list[dict[str, Any]]
        List of dataset samples returned by SSLPrefixDataset.

    Returns
    -------
    dict[str, Any]
        Batch dictionary with padded tensors and padding mask.
    """
    prefix_ids = [item["prefix_id"] for item in batch]
    vehicle_ids = [item["vehicle_id"] for item in batch]
    sequence_lengths = torch.tensor(
        [item["sequence_length"] for item in batch],
        dtype=torch.long,
    )

    time_steps = pad_sequence(
        [item["time_steps"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )
    time_gaps = pad_sequence(
        [item["time_gaps"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )

    view_1_numerical = pad_sequence(
        [item["view_1_numerical"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )
    view_1_histogram = pad_sequence(
        [item["view_1_histogram"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )
    view_2_numerical = pad_sequence(
        [item["view_2_numerical"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )
    view_2_histogram = pad_sequence(
        [item["view_2_histogram"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )

    max_len = time_steps.size(1)
    padding_mask = (
        torch.arange(max_len).unsqueeze(0) < sequence_lengths.unsqueeze(1)
    )

    return {
        "prefix_ids": prefix_ids,
        "vehicle_ids": vehicle_ids,
        "sequence_lengths": sequence_lengths,
        "padding_mask": padding_mask,  # True = valid position
        "time_steps": time_steps,
        "time_gaps": time_gaps,
        "view_1_numerical": view_1_numerical,
        "view_1_histogram": view_1_histogram,
        "view_2_numerical": view_2_numerical,
        "view_2_histogram": view_2_histogram,
    }

# prefix dataset for ssl_tfc

class TFCPrefixDataset(Dataset):
    """
    PyTorch dataset for TFC-style self-supervised learning on prefix samples.

    Each dataset item corresponds to one stored prefix sample.
    The dataset returns the base time-domain and frequency-domain branches.
    Augmentations are created later inside the training step.

    Expected keys in each prefix sample
    -----------------------------------
    - "prefix_id"
    - "vehicle_id"
    - "prefix_length"
    - "time_steps"
    - "time_gaps" or "time_gaps_scaled"
    - "dynamic_sequence_time"
    - "dynamic_sequence_freq"
    """

    def __init__(
        self,
        prefix_dict: dict[str, dict],
        use_scaled_time_gaps: bool = True,
    ) -> None:
        if len(prefix_dict) == 0:
            raise ValueError("prefix_dict must not be empty.")

        required_keys = [
            "prefix_id",
            "vehicle_id",
            "prefix_length",
            "time_steps",
            "dynamic_sequence_time",
            "dynamic_sequence_freq",
        ]

        time_gap_key = "time_gaps_scaled" if use_scaled_time_gaps else "time_gaps"

        for prefix_id, prefix_sample in prefix_dict.items():
            missing_keys = [key for key in required_keys if key not in prefix_sample]
            if missing_keys:
                raise KeyError(
                    f"Prefix '{prefix_id}' is missing required keys: {missing_keys}"
                )

            if time_gap_key not in prefix_sample:
                raise KeyError(
                    f"Prefix '{prefix_id}' must contain '{time_gap_key}'."
                )

        self.prefix_dict = prefix_dict
        self.prefix_ids = sorted(prefix_dict.keys())
        self.time_gap_key = time_gap_key

    def __len__(self) -> int:
        return len(self.prefix_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        prefix_id = self.prefix_ids[idx]
        prefix_sample = self.prefix_dict[prefix_id]

        vehicle_id = prefix_sample["vehicle_id"]
        sequence_length = int(prefix_sample["prefix_length"])

        x_time = np.asarray(
            prefix_sample["dynamic_sequence_time"],
            dtype=np.float32,
        )
        x_freq = np.asarray(
            prefix_sample["dynamic_sequence_freq"],
            dtype=np.float32,
        )
        time_steps = np.asarray(
            prefix_sample["time_steps"],
            dtype=np.float32,
        )
        time_gaps = np.asarray(
            prefix_sample[self.time_gap_key],
            dtype=np.float32,
        )

        mask = np.ones(sequence_length, dtype=np.float32)

        sample = {
            "prefix_id": prefix_id,
            "vehicle_id": vehicle_id,
            "sequence_length": sequence_length,
            "time_steps": torch.tensor(time_steps, dtype=torch.float32),
            "time_gaps": torch.tensor(time_gaps, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "x_time": torch.tensor(x_time, dtype=torch.float32),
            "x_freq": torch.tensor(x_freq, dtype=torch.float32),
        }

        return sample

# collate function for tfc




def tfc_prefix_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function for batching variable-length TFC prefix samples.
    """
    prefix_ids = [item["prefix_id"] for item in batch]
    vehicle_ids = [item["vehicle_id"] for item in batch]

    sequence_lengths = torch.tensor(
        [item["sequence_length"] for item in batch],
        dtype=torch.long,
    )

    time_steps = pad_sequence(
        [item["time_steps"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )

    time_gaps = pad_sequence(
        [item["time_gaps"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )

    x_time = pad_sequence(
        [item["x_time"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )

    x_freq = pad_sequence(
        [item["x_freq"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )

    max_len = time_steps.size(1)
    padding_mask = (
        torch.arange(max_len).unsqueeze(0) < sequence_lengths.unsqueeze(1)
    )

    return {
        "prefix_ids": prefix_ids,
        "vehicle_ids": vehicle_ids,
        "sequence_lengths": sequence_lengths,
        "padding_mask": padding_mask,   # True = valid position
        "time_steps": time_steps,
        "time_gaps": time_gaps,
        "x_time": x_time,
        "x_freq": x_freq,
    }

class TFCTruncatedSequenceDataset(Dataset):
    """
    PyTorch dataset for unsupervised TFC pretraining on truncated sequences.

    This dataset is intended for the experiment where TFC is pretrained on the
    same truncated operational histories used in downstream survival fine-tuning,
    but without using survival supervision.

    Each item returns:
    - x_time: concatenated numerical and histogram sequence
    - x_freq: differenced version of x_time
    - time_steps
    - time_gaps
    - sequence_length

    Survival labels such as duration and event may exist in the input samples,
    but they are intentionally ignored here.

    Expected keys in each sequence sample
    -------------------------------------
    - "vehicle_id"
    - "time_steps"
    - "time_gaps" or "time_gaps_scaled"
    - "numerical_sequence"
    - "histogram_sequence"

    Optional keys
    -------------
    - "sequence_length"
    - "readout_time"
    - "duration"
    - "event"
    """

    def __init__(
        self,
        sequence_dict: dict[str, dict],
        use_scaled_time_gaps: bool = False,
    ) -> None:
        if len(sequence_dict) == 0:
            raise ValueError("sequence_dict must not be empty.")

        required_keys = [
            "vehicle_id",
            "time_steps",
            "numerical_sequence",
            "histogram_sequence",
        ]

        time_gap_key = "time_gaps_scaled" if use_scaled_time_gaps else "time_gaps"

        for sequence_id, sequence_sample in sequence_dict.items():
            missing_keys = [
                key for key in required_keys
                if key not in sequence_sample
            ]

            if missing_keys:
                raise KeyError(
                    f"Sequence '{sequence_id}' is missing required keys: "
                    f"{missing_keys}"
                )

            if time_gap_key not in sequence_sample:
                raise KeyError(
                    f"Sequence '{sequence_id}' must contain '{time_gap_key}'."
                )

        self.sequence_dict = sequence_dict
        self.sequence_ids = sorted(sequence_dict.keys())
        self.time_gap_key = time_gap_key

    def __len__(self) -> int:
        return len(self.sequence_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sequence_id = self.sequence_ids[idx]
        sequence_sample = self.sequence_dict[sequence_id]

        vehicle_id = sequence_sample["vehicle_id"]

        numerical_sequence = np.asarray(
            sequence_sample["numerical_sequence"],
            dtype=np.float32,
        )

        histogram_sequence = np.asarray(
            sequence_sample["histogram_sequence"],
            dtype=np.float32,
        )

        time_steps = np.asarray(
            sequence_sample["time_steps"],
            dtype=np.float32,
        )

        time_gaps = np.asarray(
            sequence_sample[self.time_gap_key],
            dtype=np.float32,
        )

        x_time = np.concatenate(
            [numerical_sequence, histogram_sequence],
            axis=-1,
        ).astype(np.float32)

        x_freq = np.zeros_like(x_time, dtype=np.float32)

        if x_time.shape[0] > 1:
            x_freq[1:, :] = x_time[1:, :] - x_time[:-1, :]

        if "sequence_length" in sequence_sample:
            sequence_length = int(sequence_sample["sequence_length"])
        else:
            sequence_length = int(x_time.shape[0])

        sample = {
            "sequence_id": sequence_id,
            "vehicle_id": vehicle_id,
            "sequence_length": sequence_length,
            "time_steps": torch.tensor(time_steps, dtype=torch.float32),
            "time_gaps": torch.tensor(time_gaps, dtype=torch.float32),
            "x_time": torch.tensor(x_time, dtype=torch.float32),
            "x_freq": torch.tensor(x_freq, dtype=torch.float32),
        }

        return sample

def tfc_truncated_sequence_collate_fn(
    batch: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Collate function for batching variable-length truncated sequences
    for unsupervised TFC pretraining.

    Padding mask convention:
    - True = valid timestep
    - False = padded timestep
    """

    sequence_ids = [item["sequence_id"] for item in batch]
    vehicle_ids = [item["vehicle_id"] for item in batch]

    sequence_lengths = torch.tensor(
        [item["sequence_length"] for item in batch],
        dtype=torch.long,
    )

    time_steps = pad_sequence(
        [item["time_steps"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )

    time_gaps = pad_sequence(
        [item["time_gaps"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )

    x_time = pad_sequence(
        [item["x_time"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )

    x_freq = pad_sequence(
        [item["x_freq"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )

    max_len = x_time.size(1)

    padding_mask = (
        torch.arange(max_len).unsqueeze(0) < sequence_lengths.unsqueeze(1)
    )

    return {
        "sequence_ids": sequence_ids,
        "vehicle_ids": vehicle_ids,
        "sequence_lengths": sequence_lengths,
        "padding_mask": padding_mask,
        "time_steps": time_steps,
        "time_gaps": time_gaps,
        "x_time": x_time,
        "x_freq": x_freq,
    }

class TFCFixedHorizonTestDataset(Dataset):
    """
    PyTorch dataset for evaluating a trained TFC survival model at a fixed
    prediction horizon.

    Each item is created from a full vehicle sequence. The sequence is truncated
    at the latest observed readout before:

        original_duration - prediction_horizon

    The model therefore receives only the operational history available at that
    readout time.

    This dataset is intended for test-time horizon analysis only.
    """

    def __init__(
        self,
        sequence_dict: dict,
        prediction_horizon: float,
        min_history_points: int = 5,
        use_scaled_time_gaps: bool = True,
        include_static: bool = False,
        static_key: str = "static_features_encoded",
    ) -> None:
        if len(sequence_dict) == 0:
            raise ValueError("sequence_dict must not be empty.")

        if prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive.")

        self.sequence_dict = sequence_dict
        self.sequence_ids = sorted(sequence_dict.keys())
        self.prediction_horizon = float(prediction_horizon)
        self.min_history_points = int(min_history_points)
        self.include_static = include_static
        self.static_key = static_key

        self.time_gap_key = (
            "time_gaps_scaled" if use_scaled_time_gaps else "time_gaps"
        )

        self.valid_sequence_ids = []

        for sequence_id in self.sequence_ids:
            seq = self.sequence_dict[sequence_id]

            required_keys = [
                "vehicle_id",
                "time_steps",
                "numerical_sequence",
                "histogram_sequence",
                "duration",
                "event",
            ]

            missing_keys = [key for key in required_keys if key not in seq]
            if missing_keys:
                raise KeyError(
                    f"Sequence '{sequence_id}' is missing keys: {missing_keys}"
                )

            if self.time_gap_key not in seq:
                raise KeyError(
                    f"Sequence '{sequence_id}' is missing '{self.time_gap_key}'."
                )

            if self.include_static and self.static_key not in seq:
                raise KeyError(
                    f"Sequence '{sequence_id}' is missing static key "
                    f"'{self.static_key}'."
                )

            if self._is_valid_for_horizon(seq):
                self.valid_sequence_ids.append(sequence_id)

        if len(self.valid_sequence_ids) == 0:
            raise ValueError(
                "No sequences are valid for this prediction horizon. "
                "Try a smaller horizon or lower min_history_points."
            )

        skipped = len(self.sequence_ids) - len(self.valid_sequence_ids)

        print(f"Fixed horizon dataset created.")
        print(f"Prediction horizon: {self.prediction_horizon}")
        print(f"Valid sequences: {len(self.valid_sequence_ids)}")
        print(f"Skipped sequences: {skipped}")

    def _get_readout_index(self, seq: dict) -> int | None:
        time_steps = np.asarray(seq["time_steps"], dtype=float)
        original_duration = float(seq["duration"])

        target_readout_time = original_duration - self.prediction_horizon

        if target_readout_time <= 0:
            return None

        possible_indices = np.where(time_steps <= target_readout_time)[0]

        if len(possible_indices) == 0:
            return None

        readout_idx = int(possible_indices[-1])

        if readout_idx < self.min_history_points - 1:
            return None

        return readout_idx

    def _is_valid_for_horizon(self, seq: dict) -> bool:
        readout_idx = self._get_readout_index(seq)
        return readout_idx is not None

    def __len__(self) -> int:
        return len(self.valid_sequence_ids)

    def __getitem__(self, idx: int) -> dict:
        sequence_id = self.valid_sequence_ids[idx]
        seq = self.sequence_dict[sequence_id]

        readout_idx = self._get_readout_index(seq)

        if readout_idx is None:
            raise RuntimeError(
                f"Sequence '{sequence_id}' became invalid unexpectedly."
            )

        keep_slice = slice(0, readout_idx + 1)

        vehicle_id = seq["vehicle_id"]

        time_steps = np.asarray(
            seq["time_steps"],
            dtype=np.float32,
        )[keep_slice]

        time_gaps = np.asarray(
            seq[self.time_gap_key],
            dtype=np.float32,
        )[keep_slice]

        numerical_sequence = np.asarray(
            seq["numerical_sequence"],
            dtype=np.float32,
        )[keep_slice]

        histogram_sequence = np.asarray(
            seq["histogram_sequence"],
            dtype=np.float32,
        )[keep_slice]

        x_time = np.concatenate(
            [numerical_sequence, histogram_sequence],
            axis=-1,
        ).astype(np.float32)

        x_freq = np.zeros_like(x_time, dtype=np.float32)

        if x_time.shape[0] > 1:
            x_freq[1:, :] = x_time[1:, :] - x_time[:-1, :]

        readout_time = float(time_steps[-1])
        original_duration = float(seq["duration"])
        remaining_duration = original_duration - readout_time

        sample = {
            "sequence_id": sequence_id,
            "vehicle_id": vehicle_id,
            "sequence_length": int(x_time.shape[0]),
            "time_steps": torch.tensor(time_steps, dtype=torch.float32),
            "time_gaps": torch.tensor(time_gaps, dtype=torch.float32),
            "x_time": torch.tensor(x_time, dtype=torch.float32),
            "x_freq": torch.tensor(x_freq, dtype=torch.float32),
            "duration": torch.tensor(remaining_duration, dtype=torch.float32),
            "event": torch.tensor(int(seq["event"]), dtype=torch.long),
            "readout_time": torch.tensor(readout_time, dtype=torch.float32),
            "original_duration": torch.tensor(original_duration, dtype=torch.float32),
            "requested_prediction_horizon": torch.tensor(
                self.prediction_horizon,
                dtype=torch.float32,
            ),
            "actual_prediction_horizon": torch.tensor(
                remaining_duration,
                dtype=torch.float32,
            ),
        }

        if self.include_static:
            static_features = np.asarray(
                seq[self.static_key],
                dtype=np.float32,
            )

            sample["static_features"] = torch.tensor(
                static_features,
                dtype=torch.float32,
            )

        return sample

def tfc_fixed_horizon_test_collate_fn(
    batch: list[dict],
) -> dict:
    """
    Collate function for fixed-horizon TFC test datasets.

    Padding mask convention:
    - True = valid timestep
    - False = padded timestep
    """

    sequence_ids = [item["sequence_id"] for item in batch]
    vehicle_ids = [item["vehicle_id"] for item in batch]

    sequence_lengths = torch.tensor(
        [item["sequence_length"] for item in batch],
        dtype=torch.long,
    )

    time_steps = pad_sequence(
        [item["time_steps"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )

    time_gaps = pad_sequence(
        [item["time_gaps"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )

    x_time = pad_sequence(
        [item["x_time"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )

    x_freq = pad_sequence(
        [item["x_freq"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )

    max_len = x_time.size(1)

    padding_mask = (
        torch.arange(max_len).unsqueeze(0) < sequence_lengths.unsqueeze(1)
    )

    batch_out = {
        "sequence_ids": sequence_ids,
        "vehicle_ids": vehicle_ids,
        "sequence_lengths": sequence_lengths,
        "padding_mask": padding_mask,
        "time_steps": time_steps,
        "time_gaps": time_gaps,
        "x_time": x_time,
        "x_freq": x_freq,
        "duration": torch.stack([item["duration"] for item in batch]),
        "event": torch.stack([item["event"] for item in batch]),
        "readout_time": torch.stack([item["readout_time"] for item in batch]),
        "original_duration": torch.stack(
            [item["original_duration"] for item in batch]
        ),
        "requested_prediction_horizon": torch.stack(
            [item["requested_prediction_horizon"] for item in batch]
        ),
        "actual_prediction_horizon": torch.stack(
            [item["actual_prediction_horizon"] for item in batch]
        ),
    }

    if "static_features" in batch[0]:
        batch_out["static_features"] = torch.stack(
            [item["static_features"] for item in batch]
        )

    return batch_out