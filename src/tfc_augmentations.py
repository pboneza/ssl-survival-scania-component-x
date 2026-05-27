import torch


def weak_time_augmentation(
    x: torch.Tensor,
    noise_std: float = 0.01,
    dropout_prob: float = 0.1,
) -> torch.Tensor:
    """
    Weak augmentation for time-domain sequences.

    Parameters
    ----------
    x : Tensor [B, T, D]
    """
    # Gaussian noise (small)
    noise = torch.randn_like(x) * noise_std
    x_aug = x + noise

    # Feature dropout (not time dropout)
    dropout_mask = torch.rand_like(x) > dropout_prob
    x_aug = x_aug * dropout_mask

    return x_aug

def compute_frequency_representation(x: torch.Tensor) -> torch.Tensor:
    """
    Compute FFT along time dimension.

    Parameters
    ----------
    x : Tensor [B, T, D]

    Returns
    -------
    Tensor [B, T, D] (magnitude spectrum)
    """
    x_fft = torch.fft.rfft(x, dim=1)
    x_mag = torch.abs(x_fft)

    return x_mag