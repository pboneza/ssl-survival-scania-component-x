"""
Helper functions for preparing prefix samples for TFC-style self-supervised learning.

This module assumes that full vehicle sequences have already been prepared for SSL and
stored as dictionaries containing cumulative time-branch inputs and differenced
frequency-branch inputs.

Expected keys in each full vehicle sequence dictionary
------------------------------------------------------
Required:
- "vehicle_id"
- "time_steps"
- "time_gaps"
- "sequence_length"

Preferred for TFC:
- "time_gaps_scaled"
- "dynamic_sequence_time"
- "dynamic_sequence_freq"

Optional backward-compatible modality keys:
- "numerical_sequence"
- "histogram_sequence"
- "numerical_sequence_freq"
- "histogram_sequence_freq"

Optional metadata:
- "static_features"
- "duration"
- "event"

Main functionality
------------------
- Compute valid strict prefix lengths.
- Build prefixes for one vehicle sequence.
- Build prefixes for all vehicle sequences.
- Validate prefix dictionaries for TFC dataset creation.

This module does NOT create stochastic augmented views.
Augmentations must be generated on the fly during model training.
"""

from __future__ import annotations

from typing import Any


def get_prefix_lengths(
    sequence_length: int,
    proportions: tuple[float, ...] = (0.3, 0.6, 0.9),
    min_prefix_len: int = 5,
    strict_prefix: bool = True,
) -> list[int]:
    """
    Compute valid prefix lengths for one sequence.

    Prefixes are sampled at fixed proportions of the full sequence length.
    If strict_prefix is True, the full sequence itself is not allowed.

    Parameters
    ----------
    sequence_length : int
        Number of observations in the full sequence.
    proportions : tuple of float, default=(0.3, 0.6, 0.9)
        Fractions of the full sequence length used to define prefix endpoints.
    min_prefix_len : int, default=5
        Minimum allowed prefix length.
    strict_prefix : bool, default=True
        If True, clip the maximum prefix length to L - 1 so that the
        full sequence itself is not used as a prefix.

    Returns
    -------
    list[int]
        Sorted list of unique valid prefix lengths.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")

    if min_prefix_len <= 0:
        raise ValueError("min_prefix_len must be positive.")

    if not proportions:
        raise ValueError("proportions must contain at least one value.")

    max_prefix_len = sequence_length - 1 if strict_prefix else sequence_length

    if max_prefix_len < min_prefix_len:
        return []

    prefix_lengths: list[int] = []

    for proportion in proportions:
        if proportion <= 0 or proportion > 1:
            raise ValueError(
                f"Each proportion must be in (0, 1], got {proportion}."
            )

        prefix_len = int(round(sequence_length * proportion))

        if prefix_len < min_prefix_len:
            prefix_len = min_prefix_len

        if prefix_len > max_prefix_len:
            prefix_len = max_prefix_len

        prefix_lengths.append(prefix_len)

    return sorted(set(prefix_lengths))


def build_tfc_prefixes_for_one_sequence(
    vehicle_sequence: dict[str, Any],
    proportions: tuple[float, ...] = (0.3, 0.6, 0.9),
    min_prefix_len: int = 5,
    strict_prefix: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Create TFC-ready prefix samples for one vehicle sequence.

    Parameters
    ----------
    vehicle_sequence : dict[str, Any]
        Dictionary containing one vehicle's full sequence.
    proportions : tuple of float, default=(0.3, 0.6, 0.9)
        Fractions of the full sequence length used to define prefix endpoints.
    min_prefix_len : int, default=5
        Minimum allowed prefix length.
    strict_prefix : bool, default=True
        If True, the full sequence itself is not used as a prefix.

    Returns
    -------
    dict[str, dict[str, Any]]
        Dictionary of prefix samples keyed by prefix_id.
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

    prefix_lengths = get_prefix_lengths(
        sequence_length=sequence_length,
        proportions=proportions,
        min_prefix_len=min_prefix_len,
        strict_prefix=strict_prefix,
    )

    if len(prefix_lengths) == 0:
        return {}

    prefixes: dict[str, dict[str, Any]] = {}

    for prefix_number, prefix_len in enumerate(prefix_lengths, start=1):
        prefix_id = f"{vehicle_id}_p{prefix_number}"

        prefix_sample: dict[str, Any] = {
            "prefix_id": prefix_id,
            "vehicle_id": vehicle_id,
            "prefix_number": prefix_number,
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

        # Preferred joint TFC branches
        if "dynamic_sequence_time" in vehicle_sequence:
            prefix_sample["dynamic_sequence_time"] = (
                vehicle_sequence["dynamic_sequence_time"][:prefix_len].copy()
            )

        if "dynamic_sequence_freq" in vehicle_sequence:
            prefix_sample["dynamic_sequence_freq"] = (
                vehicle_sequence["dynamic_sequence_freq"][:prefix_len].copy()
            )

        # Optional modality-specific keys for backward compatibility
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

        # Optional metadata
        if "static_features" in vehicle_sequence:
            prefix_sample["static_features"] = vehicle_sequence["static_features"].copy()

        if "duration" in vehicle_sequence:
            prefix_sample["duration"] = vehicle_sequence["duration"]

        if "event" in vehicle_sequence:
            prefix_sample["event"] = vehicle_sequence["event"]

        prefixes[prefix_id] = prefix_sample

    return prefixes


def build_tfc_prefixes_for_all_sequences(
    sequence_dict: dict[int, dict[str, Any]],
    proportions: tuple[float, ...] = (0.3, 0.6, 0.9),
    min_prefix_len: int = 5,
    strict_prefix: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Create TFC-ready prefix samples for all vehicles.

    Parameters
    ----------
    sequence_dict : dict[int, dict[str, Any]]
        Dictionary of full vehicle sequences keyed by vehicle_id.
    proportions : tuple of float, default=(0.3, 0.6, 0.9)
        Fractions of the full sequence length used to define prefix endpoints.
    min_prefix_len : int, default=5
        Minimum allowed prefix length.
    strict_prefix : bool, default=True
        If True, the full sequence itself is not used as a prefix.

    Returns
    -------
    dict[str, dict[str, Any]]
        Dictionary containing all prefix samples keyed by prefix_id.
    """
    if len(sequence_dict) == 0:
        return {}

    all_prefixes: dict[str, dict[str, Any]] = {}

    for _, vehicle_sequence in sequence_dict.items():
        vehicle_prefixes = build_tfc_prefixes_for_one_sequence(
            vehicle_sequence=vehicle_sequence,
            proportions=proportions,
            min_prefix_len=min_prefix_len,
            strict_prefix=strict_prefix,
        )
        all_prefixes.update(vehicle_prefixes)

    return all_prefixes


def validate_tfc_prefix_dict(
    prefix_dict: dict[str, dict[str, Any]],
    require_joint_sequences: bool = True,
    require_scaled_time_gaps: bool = True,
) -> None:
    """
    Validate a TFC prefix dictionary before dataset creation.

    Parameters
    ----------
    prefix_dict : dict[str, dict[str, Any]]
        Prefix dictionary keyed by prefix_id.
    require_joint_sequences : bool, default=True
        If True, require:
        - dynamic_sequence_time
        - dynamic_sequence_freq
        Otherwise allow modality-specific sequences only.
    require_scaled_time_gaps : bool, default=True
        If True, require time_gaps_scaled.

    Raises
    ------
    ValueError, KeyError
        If validation fails.
    """
    if len(prefix_dict) == 0:
        raise ValueError("prefix_dict must not be empty.")

    required_common = [
        "prefix_id",
        "vehicle_id",
        "prefix_length",
        "time_steps",
    ]

    if require_scaled_time_gaps:
        required_common.append("time_gaps_scaled")
    else:
        required_common.append("time_gaps")

    for prefix_id, prefix_sample in prefix_dict.items():
        missing_common = [k for k in required_common if k not in prefix_sample]
        if missing_common:
            raise KeyError(
                f"Prefix '{prefix_id}' is missing required keys: {missing_common}"
            )

        if require_joint_sequences:
            required_joint = ["dynamic_sequence_time", "dynamic_sequence_freq"]
            missing_joint = [k for k in required_joint if k not in prefix_sample]
            if missing_joint:
                raise KeyError(
                    f"Prefix '{prefix_id}' is missing required joint sequence keys: {missing_joint}"
                )
        else:
            required_modalities = ["numerical_sequence", "histogram_sequence"]
            missing_modalities = [
                k for k in required_modalities if k not in prefix_sample
            ]
            if missing_modalities:
                raise KeyError(
                    f"Prefix '{prefix_id}' is missing required modality sequence keys: {missing_modalities}"
                )


def summarize_tfc_prefix_dict(
    prefix_dict: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """
    Summarize a TFC prefix dictionary.

    Returns
    -------
    dict[str, Any]
        Basic statistics for quick sanity checks.
    """
    if len(prefix_dict) == 0:
        return {
            "n_prefixes": 0,
            "n_unique_vehicles": 0,
            "min_prefix_length": None,
            "max_prefix_length": None,
            "mean_prefix_length": None,
        }

    prefix_lengths = [int(sample["prefix_length"]) for sample in prefix_dict.values()]
    vehicle_ids = [sample["vehicle_id"] for sample in prefix_dict.values()]

    return {
        "n_prefixes": len(prefix_dict),
        "n_unique_vehicles": len(set(vehicle_ids)),
        "min_prefix_length": min(prefix_lengths),
        "max_prefix_length": max(prefix_lengths),
        "mean_prefix_length": sum(prefix_lengths) / len(prefix_lengths),
    }

