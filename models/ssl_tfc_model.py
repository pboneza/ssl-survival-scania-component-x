from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Neural network modules for TFC-style self-supervised pretraining and downstream
survival fine-tuning on irregular multivariate vehicle sequences.

This module implements a two-branch architecture:

1. Time branch
   - consumes cumulative sequence levels
   - uses a time-aware recurrent encoder to account explicitly for irregular
     observation gaps

2. Frequency branch
   - consumes differenced sequences transformed through FFT
   - uses a standard recurrent encoder over frequency representations

The module provides:
- a time-aware GRU-style encoder for the time branch
- a recurrent encoder for the frequency branch
- projection heads for contrastive/self-supervised objectives
- a full TFC pretraining model
- a downstream survival fine-tuning model that reuses the pretrained time encoder
"""


class TimeAwareGRUCell(nn.Module):
    """
    GRUCell with explicit hidden-state decay driven by elapsed time gaps.

    Parameters
    ----------
    input_dim : int
        Input feature dimension at each time step.
    hidden_dim : int
        Hidden-state dimension.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.decay_layer = nn.Linear(1, hidden_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_t : torch.Tensor
            Input at time t, shape (B, D).
        h_prev : torch.Tensor
            Previous hidden state, shape (B, H).
        delta_t : torch.Tensor
            Elapsed time gap, shape (B, 1).

        Returns
        -------
        torch.Tensor
            Updated hidden state, shape (B, H).
        """
        gamma_t = torch.exp(-torch.relu(self.decay_layer(delta_t)))
        h_prev_decayed = gamma_t * h_prev
        h_t = self.gru_cell(x_t, h_prev_decayed)
        return h_t


class TimeAwareSequenceEncoder(nn.Module):
    """
    Time-aware recurrent encoder for irregular time-domain sequences.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step.
    hidden_dim : int, default=128
        Hidden dimension of the recurrent encoder.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = TimeAwareGRUCell(input_dim=input_dim, hidden_dim=hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        time_gaps: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, T, D).
        time_gaps : torch.Tensor
            Time-gap tensor of shape (B, T).
        padding_mask : torch.Tensor
            Boolean mask of shape (B, T), where True indicates valid positions.

        Returns
        -------
        torch.Tensor
            Hidden-state sequence of shape (B, T, H).
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        hidden_states = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            dt_t = time_gaps[:, t].unsqueeze(-1)
            valid_t = padding_mask[:, t].unsqueeze(-1).float()

            h_candidate = self.cell(x_t, h, dt_t)
            h = valid_t * h_candidate + (1.0 - valid_t) * h
            hidden_states.append(h.unsqueeze(1))

        hidden_seq = torch.cat(hidden_states, dim=1)
        hidden_seq = hidden_seq * padding_mask.unsqueeze(-1).float()
        return hidden_seq


class FrequencySequenceEncoder(nn.Module):
    """
    Recurrent encoder for frequency-domain sequence representations.

    Parameters
    ----------
    input_dim : int
        Number of frequency-branch input features per step.
    hidden_dim : int, default=128
        Hidden dimension of the recurrent encoder.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, T_f, D_f).
        padding_mask : torch.Tensor
            Boolean mask of shape (B, T_f), where True indicates valid positions.

        Returns
        -------
        torch.Tensor
            Hidden-state sequence of shape (B, T_f, H).
        """
        hidden_seq, _ = self.gru(x)
        hidden_seq = hidden_seq * padding_mask.unsqueeze(-1).float()
        return hidden_seq


class ProjectionHead(nn.Module):
    """
    Projection head for self-supervised contrastive objectives.

    Parameters
    ----------
    input_dim : int
        Input representation dimension.
    projection_dim : int, default=64
        Output projection dimension.
    hidden_dim : int, default=128
        Hidden dimension in the projection MLP.
    """
    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TFCSequenceModel(nn.Module):
    """
    Two-branch TFC self-supervised pretraining model.

    Time branch
    -----------
    - input: cumulative time-domain sequence
    - encoder: time-aware recurrent encoder

    Frequency branch
    ----------------
    - input: FFT magnitude of differenced sequence
    - encoder: recurrent frequency encoder

    Parameters
    ----------
    time_input_dim : int, default=105
        Number of features in the time-domain sequence.
    freq_input_dim : int, default=105
        Number of features in the differenced sequence before FFT.
    time_hidden_dim : int, default=128
        Hidden dimension of the time-aware encoder.
    freq_hidden_dim : int, default=128
        Hidden dimension of the frequency encoder.
    projection_dim : int, default=64
        Output projection dimension for both branches.
    """

    def __init__(
        self,
        time_input_dim: int = 105,
        freq_input_dim: int = 105,
        time_hidden_dim: int = 128,
        freq_hidden_dim: int = 128,
        projection_dim: int = 64,
    ):
        super().__init__()

        self.time_encoder = TimeAwareSequenceEncoder(
            input_dim=time_input_dim,
            hidden_dim=time_hidden_dim,
        )

        self.freq_encoder = FrequencySequenceEncoder(
            input_dim=freq_input_dim,
            hidden_dim=freq_hidden_dim,
        )

        self.time_projection_head = ProjectionHead(
            input_dim=time_hidden_dim,
            projection_dim=projection_dim,
            hidden_dim=128,
        )

        self.freq_projection_head = ProjectionHead(
            input_dim=freq_hidden_dim,
            projection_dim=projection_dim,
            hidden_dim=128,
        )

    @staticmethod
    def masked_mean_pool(
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = padding_mask.unsqueeze(-1).float()
        x_masked = x * mask
        lengths = mask.sum(dim=1).clamp(min=1.0)
        pooled = x_masked.sum(dim=1) / lengths
        return pooled

    @staticmethod
    def compute_frequency_representation(
        x_freq: torch.Tensor,
        padding_mask: torch.Tensor,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute FFT magnitude from the differenced frequency-branch sequence.

        Parameters
        ----------
        x_freq : torch.Tensor
            Differenced sequence tensor of shape (B, T, D).
        padding_mask : torch.Tensor
            Boolean mask of shape (B, T).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - fft_mag: tensor of shape (B, T_f, D)
            - fft_mask: boolean mask of shape (B, T_f)
        """
        x_masked = x_freq * padding_mask.unsqueeze(-1).float()

        # FFT along time dimension
        fft_complex = torch.fft.rfft(x_masked, dim=1)
        fft_mag = torch.log1p(torch.abs(fft_complex) + eps)

        # rfft reduces time dimension from T to floor(T/2) + 1
        freq_len = fft_mag.size(1)
        fft_mask = torch.ones(
            fft_mag.size(0),
            freq_len,
            dtype=padding_mask.dtype,
            device=padding_mask.device,
        )

        return fft_mag, fft_mask.bool()

    def forward_time_branch(
        self,
        x_time: torch.Tensor,
        time_gaps: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for time branch.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - pooled time representation
            - projected time representation
        """
        time_hidden_seq = self.time_encoder(
            x=x_time,
            time_gaps=time_gaps,
            padding_mask=padding_mask,
        )
        time_pooled = self.masked_mean_pool(time_hidden_seq, padding_mask)
        time_proj = self.time_projection_head(time_pooled)
        return time_pooled, time_proj

    def forward_frequency_branch(
        self,
        x_freq: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for frequency branch.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - pooled frequency representation
            - projected frequency representation
        """
        fft_mag, fft_mask = self.compute_frequency_representation(
            x_freq=x_freq,
            padding_mask=padding_mask,
        )
        freq_hidden_seq = self.freq_encoder(
            x=fft_mag,
            padding_mask=fft_mask,
        )
        freq_pooled = self.masked_mean_pool(freq_hidden_seq, fft_mask)
        freq_proj = self.freq_projection_head(freq_pooled)
        return freq_pooled, freq_proj

    def forward_precomputed_frequency_branch(
        self,
        fft_mag: torch.Tensor,
        fft_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a precomputed FFT magnitude representation.
    
        Parameters
        ----------
        fft_mag : torch.Tensor
            FFT magnitude tensor of shape (B, T_f, D).
        fft_mask : torch.Tensor
            Boolean mask of shape (B, T_f), where True indicates valid positions.
    
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - pooled frequency representation
            - projected frequency representation
        """
        freq_hidden_seq = self.freq_encoder(
            x=fft_mag,
            padding_mask=fft_mask,
        )
        freq_pooled = self.masked_mean_pool(freq_hidden_seq, fft_mask)
        freq_proj = self.freq_projection_head(freq_pooled)
        return freq_pooled, freq_proj

    def forward(
        self,
        x_time: torch.Tensor,
        x_freq: torch.Tensor,
        time_gaps: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass through both branches.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing pooled and projected representations.
        """
        time_pooled, time_proj = self.forward_time_branch(
            x_time=x_time,
            time_gaps=time_gaps,
            padding_mask=padding_mask,
        )

        freq_pooled, freq_proj = self.forward_frequency_branch(
            x_freq=x_freq,
            padding_mask=padding_mask,
        )

        return {
            "time_pooled": time_pooled,
            "time_proj": time_proj,
            "freq_pooled": freq_pooled,
            "freq_proj": freq_proj,
        }


class TFCSurvivalFineTuningModel(nn.Module):
    """
    Supervised survival model initialized from a pretrained TFC time encoder.

    This model reuses the pretrained time-aware time encoder learned during
    TFC pretraining and replaces the self-supervised projection head with a
    supervised survival head that predicts a scalar risk score.

    Parameters
    ----------
    pretrained_tfc_model : nn.Module
        Pretrained TFCSequenceModel.
    survival_hidden_dim : int, default=64
        Hidden dimension of the survival head.
    freeze_encoder : bool, default=False
        If True, encoder parameters are frozen.
    """

    def __init__(
        self,
        pretrained_tfc_model: nn.Module,
        survival_hidden_dim: int = 64,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        self.time_encoder = pretrained_tfc_model.time_encoder
        self.masked_mean_pool = pretrained_tfc_model.masked_mean_pool
        encoder_hidden_dim = pretrained_tfc_model.time_encoder.hidden_dim

        self.survival_head = nn.Sequential(
            nn.Linear(encoder_hidden_dim, survival_hidden_dim),
            nn.ReLU(),
            nn.Linear(survival_hidden_dim, 1),
        )

        if freeze_encoder:
            for param in self.time_encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        x_time: torch.Tensor,
        time_gaps: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x_time : torch.Tensor
            Time-domain input tensor of shape (B, T, D).
        time_gaps : torch.Tensor
            Time-gap tensor of shape (B, T).
        padding_mask : torch.Tensor
            Boolean validity mask of shape (B, T).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - pooled sequence representation
            - scalar risk scores of shape (B,)
        """
        time_hidden_seq = self.time_encoder(
            x=x_time,
            time_gaps=time_gaps,
            padding_mask=padding_mask,
        )

        pooled = self.masked_mean_pool(time_hidden_seq, padding_mask)
        risk = self.survival_head(pooled).squeeze(-1)

        return pooled, risk

class TFCSurvivalFineTuningWithStaticModel(nn.Module):
    """
    TFC survival fine-tuning model with optional static-feature encoder.

    static_fusion:
    - "direct": concatenate pooled sequence representation with OHE static features
    - "mlp": encode static features first, then concatenate
    """

    def __init__(
        self,
        pretrained_tfc_model: nn.Module,
        static_dim: int,
        static_fusion: str = "direct",
        static_hidden_dim: int = 32,
        static_repr_dim: int = 16,
        survival_hidden_dim: int = 64,
        dropout: float = 0.2,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        if static_fusion not in {"direct", "mlp"}:
            raise ValueError("static_fusion must be either 'direct' or 'mlp'.")

        self.static_fusion = static_fusion
        self.time_encoder = pretrained_tfc_model.time_encoder
        self.masked_mean_pool = pretrained_tfc_model.masked_mean_pool

        encoder_hidden_dim = pretrained_tfc_model.time_encoder.hidden_dim

        if static_fusion == "mlp":
            self.static_encoder = nn.Sequential(
                nn.Linear(static_dim, static_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(static_hidden_dim, static_repr_dim),
                nn.ReLU(),
            )
            fusion_dim = encoder_hidden_dim + static_repr_dim
        else:
            self.static_encoder = None
            fusion_dim = encoder_hidden_dim + static_dim

        self.survival_head = nn.Sequential(
            nn.Linear(fusion_dim, survival_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(survival_hidden_dim, survival_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(survival_hidden_dim // 2, 1),
        )

        if freeze_encoder:
            for param in self.time_encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        x_time: torch.Tensor,
        time_gaps: torch.Tensor,
        padding_mask: torch.Tensor,
        static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        time_hidden_seq = self.time_encoder(
            x=x_time,
            time_gaps=time_gaps,
            padding_mask=padding_mask,
        )

        pooled = self.masked_mean_pool(time_hidden_seq, padding_mask)

        static = static.float()

        if self.static_fusion == "mlp":
            static_repr = self.static_encoder(static)
        else:
            static_repr = static

        fused = torch.cat([pooled, static_repr], dim=1)
        risk = self.survival_head(fused).squeeze(-1)

        return fused, risk