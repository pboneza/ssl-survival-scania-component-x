"""
Helper functions for sequence preprocessing prior to self-supervised pretraining.

This module prepares operational sequence data for the SSL stage. Its main purpose is
to generate prefix samples from full vehicle trajectories and to create augmented views
of those prefixes for contrastive learning.

For each full sequence, up to three strict prefixes are sampled at approximately 30%,
60%, and 90% of the observed sequence length. To ensure that the full sequence itself
is not used as a prefix, the 90% endpoint is clipped to at most L - 1 when rounding
would otherwise select the full length L. Prefixes are retained only if they satisfy
the minimum required length.

For each retained prefix sample, two stochastic augmented views are generated. These
paired views are intended for use in contrastive self-supervised learning and provide
inputs that are ready to be wrapped into a PyTorch dataset for SSL pretraining.
"""

from __future__ import annotations

from typing import Iterable
import pandas as pd
import numpy as np


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

    max_prefix_len = sequence_length
    if strict_prefix:
        max_prefix_len = sequence_length - 1

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

    # Remove duplicates caused by rounding and sort
    prefix_lengths = sorted(set(prefix_lengths))

    return prefix_lengths

def build_prefixes_for_one_sequence(
    vehicle_sequence: dict,
    proportions: tuple[float, ...] = (0.3, 0.6, 0.9),
    min_prefix_len: int = 5,
    strict_prefix: bool = True,
) -> dict[str, dict]:
    """
    Create prefix samples for one vehicle sequence stored as a dictionary.

    Parameters
    ----------
    vehicle_sequence : dict
        Dictionary containing one vehicle's full sequence. Expected keys are:
        - "vehicle_id"
        - "time_steps"
        - "time_gaps"
        - "numerical_sequence"
        - "histogram_sequence"
        - "sequence_length"
        Optional keys such as "static_features", "duration", and "event"
        may also be present.
    proportions : tuple of float, default=(0.3, 0.6, 0.9)
        Fractions of the full sequence length used to define prefix endpoints.
    min_prefix_len : int, default=5
        Minimum allowed prefix length.
    strict_prefix : bool, default=True
        If True, the full sequence itself is not used as a prefix.

    Returns
    -------
    dict[str, dict]
        Dictionary of prefix samples keyed by prefix_id.
    """
    required_keys = [
        "vehicle_id",
        "time_steps",
        "time_gaps",
        "numerical_sequence",
        "histogram_sequence",
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

    prefixes: dict[str, dict] = {}

    for prefix_number, prefix_len in enumerate(prefix_lengths, start=1):
        prefix_id = f"{vehicle_id}_p{prefix_number}"

        prefix_sample = {
            "prefix_id": prefix_id,
            "vehicle_id": vehicle_id,
            "prefix_number": prefix_number,
            "prefix_length": prefix_len,
            "full_sequence_length": sequence_length,
            "prefix_fraction": prefix_len / sequence_length,
            "time_steps": vehicle_sequence["time_steps"][:prefix_len].copy(),
            "time_gaps": vehicle_sequence["time_gaps"][:prefix_len].copy(),
            "numerical_sequence": vehicle_sequence["numerical_sequence"][:prefix_len].copy(),
            "histogram_sequence": vehicle_sequence["histogram_sequence"][:prefix_len].copy(),
        }

        # Optional fields: keep if present
        if "static_features" in vehicle_sequence:
            prefix_sample["static_features"] = vehicle_sequence["static_features"].copy()

        if "duration" in vehicle_sequence:
            prefix_sample["duration"] = vehicle_sequence["duration"]

        if "event" in vehicle_sequence:
            prefix_sample["event"] = vehicle_sequence["event"]

        prefixes[prefix_id] = prefix_sample

    return prefixes


def build_prefixes_for_all_sequences(
    sequence_dict: dict[int, dict],
    proportions: tuple[float, ...] = (0.3, 0.6, 0.9),
    min_prefix_len: int = 5,
    strict_prefix: bool = True,
) -> dict[str, dict]:
    """
    Create prefix samples for all vehicles in a dictionary of full sequences.

    Parameters
    ----------
    sequence_dict : dict[int, dict]
        Dictionary of full vehicle sequences keyed by vehicle_id.
    proportions : tuple of float, default=(0.3, 0.6, 0.9)
        Fractions of the full sequence length used to define prefix endpoints.
    min_prefix_len : int, default=5
        Minimum allowed prefix length.
    strict_prefix : bool, default=True
        If True, the full sequence itself is not used as a prefix.

    Returns
    -------
    dict[str, dict]
        Dictionary containing all prefix samples keyed by prefix_id.
    """
    if len(sequence_dict) == 0:
        return {}

    all_prefixes: dict[str, dict] = {}

    for _, vehicle_sequence in sequence_dict.items():
        vehicle_prefixes = build_prefixes_for_one_sequence(
            vehicle_sequence=vehicle_sequence,
            proportions=proportions,
            min_prefix_len=min_prefix_len,
            strict_prefix=strict_prefix,
        )

        all_prefixes.update(vehicle_prefixes)

    return all_prefixes

def mask_numerical_sequence(
    sequence: np.ndarray,
    mask_probability: float = 0.15,
    mask_value: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Randomly mask numerical feature values in a sequence.

    Parameters
    ----------
    sequence : np.ndarray
        Array of shape (T, D_num).
    mask_probability : float, default=0.15
        Probability of masking each value.
    mask_value : float, default=0.0
        Value used for masking.
    rng : np.random.Generator or None, default=None
        Random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Masked numerical sequence.
    """
    if rng is None:
        rng = np.random.default_rng()

    augmented = sequence.copy()
    mask = rng.random(size=augmented.shape) < mask_probability
    augmented[mask] = mask_value

    return augmented


def add_noise_to_numerical_sequence(
    sequence: np.ndarray,
    noise_std: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Add Gaussian noise to a numerical sequence.

    Parameters
    ----------
    sequence : np.ndarray
        Array of shape (T, D_num).
    noise_std : float, default=0.01
        Standard deviation of Gaussian noise.
    rng : np.random.Generator or None, default=None
        Random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy numerical sequence.
    """
    if rng is None:
        rng = np.random.default_rng()

    augmented = sequence.copy()
    noise = rng.normal(loc=0.0, scale=noise_std, size=augmented.shape)
    augmented = augmented + noise

    return augmented


def dropout_numerical_sequence(
    sequence: np.ndarray,
    dropout_probability: float = 0.10,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Randomly drop numerical feature values by setting them to zero.

    Parameters
    ----------
    sequence : np.ndarray
        Array of shape (T, D_num).
    dropout_probability : float, default=0.10
        Probability of dropping each value.
    rng : np.random.Generator or None, default=None
        Random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Numerical sequence after dropout.
    """
    if rng is None:
        rng = np.random.default_rng()

    augmented = sequence.copy()
    dropout_mask = rng.random(size=augmented.shape) < dropout_probability
    augmented[dropout_mask] = 0.0

    return augmented


def augment_numerical_sequence(
    sequence: np.ndarray,
    mask_probability: float = 0.15,
    noise_std: float = 0.01,
    dropout_probability: float = 0.10,
    apply_mask: bool = True,
    apply_noise: bool = True,
    apply_dropout: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Apply stochastic augmentations to a numerical sequence.

    Parameters
    ----------
    sequence : np.ndarray
        Array of shape (T, D_num).
    mask_probability : float, default=0.15
        Probability of masking each value.
    noise_std : float, default=0.01
        Standard deviation of Gaussian noise.
    dropout_probability : float, default=0.10
        Probability of dropping each value.
    apply_mask : bool, default=True
        Whether to apply random masking.
    apply_noise : bool, default=True
        Whether to apply Gaussian noise.
    apply_dropout : bool, default=True
        Whether to apply dropout.
    rng : np.random.Generator or None, default=None
        Random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Augmented numerical sequence.
    """
    if rng is None:
        rng = np.random.default_rng()

    augmented = sequence.copy()

    if apply_mask:
        augmented = mask_numerical_sequence(
            sequence=augmented,
            mask_probability=mask_probability,
            rng=rng,
        )

    if apply_noise:
        augmented = add_noise_to_numerical_sequence(
            sequence=augmented,
            noise_std=noise_std,
            rng=rng,
        )

    if apply_dropout:
        augmented = dropout_numerical_sequence(
            sequence=augmented,
            dropout_probability=dropout_probability,
            rng=rng,
        )

    return augmented


def mask_histogram_sequence(
    sequence: np.ndarray,
    mask_probability: float = 0.15,
    mask_value: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Randomly mask histogram feature values in a sequence.

    Parameters
    ----------
    sequence : np.ndarray
        Array of shape (T, D_hist).
    mask_probability : float, default=0.15
        Probability of masking each value.
    mask_value : float, default=0.0
        Value used for masking.
    rng : np.random.Generator or None, default=None
        Random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Masked histogram sequence.
    """
    if rng is None:
        rng = np.random.default_rng()

    augmented = sequence.copy()
    mask = rng.random(size=augmented.shape) < mask_probability
    augmented[mask] = mask_value

    return augmented


def augment_histogram_sequence(
    sequence: np.ndarray,
    mask_probability: float = 0.15,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Apply masking-based augmentation to a histogram sequence.

    Parameters
    ----------
    sequence : np.ndarray
        Array of shape (T, D_hist).
    mask_probability : float, default=0.15
        Probability of masking each value.
    rng : np.random.Generator or None, default=None
        Random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Augmented histogram sequence.
    """
    if rng is None:
        rng = np.random.default_rng()

    augmented = mask_histogram_sequence(
        sequence=sequence,
        mask_probability=mask_probability,
        rng=rng,
    )

    return augmented


def create_two_augmented_views(
    numerical_sequence: np.ndarray,
    histogram_sequence: np.ndarray,
    numerical_mask_probability: float = 0.15,
    numerical_noise_std: float = 0.01,
    numerical_dropout_probability: float = 0.10,
    histogram_mask_probability: float = 0.15,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Create two independent augmented views of the same prefix sample.

    Parameters
    ----------
    numerical_sequence : np.ndarray
        Numerical feature sequence of shape (T, D_num).
    histogram_sequence : np.ndarray
        Histogram feature sequence of shape (T, D_hist).
    numerical_mask_probability : float, default=0.15
        Masking probability for numerical features.
    numerical_noise_std : float, default=0.01
        Noise standard deviation for numerical features.
    numerical_dropout_probability : float, default=0.10
        Dropout probability for numerical features.
    histogram_mask_probability : float, default=0.15
        Masking probability for histogram features.
    rng : np.random.Generator or None, default=None
        Random generator for reproducibility.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, np.ndarray]]
        Two augmented views. Each view is a dictionary with:
        - "numerical"
        - "histogram"
    """
    if rng is None:
        rng = np.random.default_rng()

    view_1 = {
        "numerical": augment_numerical_sequence(
            sequence=numerical_sequence,
            mask_probability=numerical_mask_probability,
            noise_std=numerical_noise_std,
            dropout_probability=numerical_dropout_probability,
            rng=rng,
        ),
        "histogram": augment_histogram_sequence(
            sequence=histogram_sequence,
            mask_probability=histogram_mask_probability,
            rng=rng,
        ),
    }

    view_2 = {
        "numerical": augment_numerical_sequence(
            sequence=numerical_sequence,
            mask_probability=numerical_mask_probability,
            noise_std=numerical_noise_std,
            dropout_probability=numerical_dropout_probability,
            rng=rng,
        ),
        "histogram": augment_histogram_sequence(
            sequence=histogram_sequence,
            mask_probability=histogram_mask_probability,
            rng=rng,
        ),
    }

    return view_1, view_2