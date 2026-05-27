"""
Helper functions for stochastic prefix generation for TFC-style SSL.

This module samples one random strict prefix per vehicle sequence.
It preserves the same prefix-sample structure as tfc_preprocessing.py.
"""

from __future__ import annotations

from typing import Any
import random
import numpy as np


def sample_prefix_length(
    sequence_length: int,
    min_prefix_len: int = 5,
    min_fraction: float = 0.3,
    max_fraction: float = 0.9,
    strict_prefix: bool = True,
    beta_a: float = 2.0,
    beta_b: float = 1.0,
) -> int | None:
    """
    Sample one valid degradation-biased random prefix length.

    Beta(2, 1) biases sampling toward later prefixes.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")

    if min_prefix_len <= 0:
        raise ValueError("min_prefix_len must be positive.")

    if not (0 < min_fraction <= max_fraction <= 1):
        raise ValueError("Expected 0 < min_fraction <= max_fraction <= 1.")

    max_prefix_len = sequence_length - 1 if strict_prefix else sequence_length

    if max_prefix_len < min_prefix_len:
        return None

    raw = np.random.beta(beta_a, beta_b)
    prefix_fraction = min_fraction + (max_fraction - min_fraction) * raw

    prefix_len = int(round(sequence_length * prefix_fraction))

    prefix_len = max(prefix_len, min_prefix_len)
    prefix_len = min(prefix_len, max_prefix_len)

    return prefix_len


def build_stochastic_tfc_prefix_for_one_sequence(
    vehicle_sequence: dict[str, Any],
    min_prefix_len: int = 5,
    min_fraction: float = 0.3,
    max_fraction: float = 0.9,
    strict_prefix: bool = True,
    epoch: int | None = None,
) -> dict[str, Any] | None:
    """
    Create one randomly sampled TFC prefix for one vehicle sequence.
    """
    required_keys = [
        "vehicle_id",
        "time_steps",
        "time_gaps",
        "sequence_length",
    ]

    missing_keys = [key for key in required_keys if key not in vehicle_sequence]
    if missing_keys:
        raise KeyError(
            f"vehicle_sequence is missing required keys: {missing_keys}"
        )

    vehicle_id = vehicle_sequence["vehicle_id"]
    sequence_length = int(vehicle_sequence["sequence_length"])

    prefix_len = sample_prefix_length(
        sequence_length=sequence_length,
        min_prefix_len=min_prefix_len,
        min_fraction=min_fraction,
        max_fraction=max_fraction,
        strict_prefix=strict_prefix,
        beta_a=2.0,
        beta_b=1.0,
    )

    if prefix_len is None:
        return None

    if epoch is None:
        prefix_id = f"{vehicle_id}_stoch_{prefix_len}"
    else:
        prefix_id = f"{vehicle_id}_e{epoch}_stoch_{prefix_len}"

    prefix_sample: dict[str, Any] = {
        "prefix_id": prefix_id,
        "vehicle_id": vehicle_id,
        "prefix_number": 1,
        "prefix_length": prefix_len,
        "full_sequence_length": sequence_length,
        "prefix_fraction": prefix_len / sequence_length,
        "time_steps": vehicle_sequence["time_steps"][:prefix_len].copy(),
        "time_gaps": vehicle_sequence["time_gaps"][:prefix_len].copy(),
    }

    if "time_gaps_scaled" in vehicle_sequence:
        prefix_sample["time_gaps_scaled"] = (
            vehicle_sequence["time_gaps_scaled"][:prefix_len].copy()
        )

    if "dynamic_sequence_time" in vehicle_sequence:
        prefix_sample["dynamic_sequence_time"] = (
            vehicle_sequence["dynamic_sequence_time"][:prefix_len].copy()
        )

    if "dynamic_sequence_freq" in vehicle_sequence:
        prefix_sample["dynamic_sequence_freq"] = (
            vehicle_sequence["dynamic_sequence_freq"][:prefix_len].copy()
        )

    if "numerical_sequence" in vehicle_sequence:
        prefix_sample["numerical_sequence"] = (
            vehicle_sequence["numerical_sequence"][:prefix_len].copy()
        )

    if "histogram_sequence" in vehicle_sequence:
        prefix_sample["histogram_sequence"] = (
            vehicle_sequence["histogram_sequence"][:prefix_len].copy()
        )

    if "numerical_sequence_freq" in vehicle_sequence:
        prefix_sample["numerical_sequence_freq"] = (
            vehicle_sequence["numerical_sequence_freq"][:prefix_len].copy()
        )

    if "histogram_sequence_freq" in vehicle_sequence:
        prefix_sample["histogram_sequence_freq"] = (
            vehicle_sequence["histogram_sequence_freq"][:prefix_len].copy()
        )

    if "static_features" in vehicle_sequence:
        prefix_sample["static_features"] = vehicle_sequence["static_features"].copy()

    if "duration" in vehicle_sequence:
        prefix_sample["duration"] = vehicle_sequence["duration"]

    if "event" in vehicle_sequence:
        prefix_sample["event"] = vehicle_sequence["event"]

    return prefix_sample


def build_stochastic_tfc_prefixes_for_all_sequences(
    sequence_dict: dict[int, dict[str, Any]],
    min_prefix_len: int = 5,
    min_fraction: float = 0.3,
    max_fraction: float = 0.9,
    strict_prefix: bool = True,
    seed: int | None = None,
    epoch: int | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Create one randomly sampled prefix per vehicle.

    This should be called again at each SSL epoch if you want stochastic
    prefix generation.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if len(sequence_dict) == 0:
        return {}

    stochastic_prefixes: dict[str, dict[str, Any]] = {}

    for _, vehicle_sequence in sequence_dict.items():
        prefix_sample = build_stochastic_tfc_prefix_for_one_sequence(
            vehicle_sequence=vehicle_sequence,
            min_prefix_len=min_prefix_len,
            min_fraction=min_fraction,
            max_fraction=max_fraction,
            strict_prefix=strict_prefix,
            epoch=epoch,
        )

        if prefix_sample is not None:
            stochastic_prefixes[prefix_sample["prefix_id"]] = prefix_sample

    return stochastic_prefixes