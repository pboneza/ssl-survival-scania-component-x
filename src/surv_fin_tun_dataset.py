from __future__ import annotations

from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SurvivalSequenceDataset(Dataset):
    """
    PyTorch dataset for supervised survival fine-tuning on truncated sequences.

    Each dataset item corresponds to one truncated labelled sequence.

    Parameters
    ----------
    sequence_dict : dict[int | str, dict]
        Dictionary of truncated vehicle sequences keyed by vehicle_id.
        Each sequence is expected to contain:
        - "vehicle_id"
        - "time_steps"
        - "time_gaps"
        - "numerical_sequence"
        - "histogram_sequence"
        - "sequence_length"
        - "duration"
        - "event"

    use_static_features : bool, default=False
        Whether to return static features if they are present.
        For now, this is optional and assumes static features are already encoded
        numerically. If they are still categorical dictionaries, keep this False.
    """

    def __init__(
        self,
        sequence_dict: dict[Any, dict],
        use_static_features: bool = False,
    ) -> None:
        if len(sequence_dict) == 0:
            raise ValueError("sequence_dict must not be empty.")

        required_keys = [
            "vehicle_id",
            "time_steps",
            "time_gaps",
            "numerical_sequence",
            "histogram_sequence",
            "sequence_length",
            "duration",
            "event",
        ]

        for vehicle_id, sequence_sample in sequence_dict.items():
            missing_keys = [key for key in required_keys if key not in sequence_sample]
            if missing_keys:
                raise KeyError(
                    f"Sequence '{vehicle_id}' is missing required keys: {missing_keys}"
                )

        self.sequence_dict = sequence_dict
        self.vehicle_ids = sorted(sequence_dict.keys())
        self.use_static_features = use_static_features

    def __len__(self) -> int:
        return len(self.vehicle_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        vehicle_id = self.vehicle_ids[idx]
        sample = self.sequence_dict[vehicle_id]

        sequence_length = int(sample["sequence_length"])

        numerical_sequence = np.asarray(
            sample["numerical_sequence"],
            dtype=np.float32,
        )
        histogram_sequence = np.asarray(
            sample["histogram_sequence"],
            dtype=np.float32,
        )
        time_steps = np.asarray(
            sample["time_steps"],
            dtype=np.float32,
        )
        time_gaps = np.asarray(
            sample["time_gaps"],
            dtype=np.float32,
        )

        duration = np.float32(sample["duration"])
        event = np.float32(sample["event"])
        sample_weight = np.float32(sample.get("sample_weight", 1.0))
        is_pseudo = bool(sample.get("is_pseudo", False))

        output = {
            "vehicle_id": sample["vehicle_id"],
            "sequence_length": sequence_length,
            "time_steps": torch.tensor(time_steps, dtype=torch.float32),
            "time_gaps": torch.tensor(time_gaps, dtype=torch.float32),
            "numerical": torch.tensor(numerical_sequence, dtype=torch.float32),
            "histogram": torch.tensor(histogram_sequence, dtype=torch.float32),
            "duration": torch.tensor(duration, dtype=torch.float32),
            "event": torch.tensor(event, dtype=torch.float32),
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
            "is_pseudo": is_pseudo,
        }

        # Optional metadata for inspection/debugging
        if "readout_time" in sample:
            output["readout_time"] = torch.tensor(
                np.float32(sample["readout_time"]),
                dtype=torch.float32,
            )

        if "original_duration" in sample:
            output["original_duration"] = torch.tensor(
                np.float32(sample["original_duration"]),
                dtype=torch.float32,
            )

        if self.use_static_features and "static_features_encoded" in sample:
            static_features = sample["static_features_encoded"]

            if isinstance(static_features, dict):
                raise ValueError(
                    "static_features are still stored as a dictionary of categories. "
                    "Encode them numerically before setting use_static_features=True."
                )

            static_features = np.asarray(static_features, dtype=np.float32)
            output["static_features"] = torch.tensor(
                static_features,
                dtype=torch.float32,
            )

        return output

# Collate function for variable-length supervised batches

def survival_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function for batching variable-length truncated survival sequences.

    Parameters
    ----------
    batch : list[dict[str, Any]]
        List of dataset samples returned by SurvivalSequenceDataset.

    Returns
    -------
    dict[str, Any]
        Batch dictionary with padded tensors and padding mask.
    """
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

    numerical = pad_sequence(
        [item["numerical"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )

    histogram = pad_sequence(
        [item["histogram"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )

    durations = torch.stack([item["duration"] for item in batch], dim=0)
    events = torch.stack([item["event"] for item in batch], dim=0)
    
    sample_weights = torch.stack(
    [item["sample_weight"] for item in batch],
    dim=0,
    )

    is_pseudo = torch.tensor(
        [item["is_pseudo"] for item in batch],
        dtype=torch.bool,
    )
    max_len = time_steps.size(1)
    padding_mask = (
        torch.arange(max_len).unsqueeze(0) < sequence_lengths.unsqueeze(1)
    )
    
    output = {
        "vehicle_ids": vehicle_ids,
        "sequence_lengths": sequence_lengths,
        "padding_mask": padding_mask,   # True = valid position
        "time_steps": time_steps,
        "time_gaps": time_gaps,
        "numerical": numerical,
        "histogram": histogram,
        "duration": durations,
        "event": events,
        "sample_weight": sample_weights,
        "is_pseudo": is_pseudo,
    }

    if "readout_time" in batch[0]:
        output["readout_time"] = torch.stack(
            [item["readout_time"] for item in batch],
            dim=0,
        )

    if "original_duration" in batch[0]:
        output["original_duration"] = torch.stack(
            [item["original_duration"] for item in batch],
            dim=0,
        )

    if "static_features" in batch[0]:
        output["static_features"] = torch.stack(
            [item["static_features"] for item in batch],
            dim=0,
        )

    return output