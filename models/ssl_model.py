import torch
import torch.nn as nn

"""
Neural network modules for self-supervised pretraining and downstream survival fine-tuning
on truncated multivariate vehicle sequences.

This module defines the shared model components used across the two-stage learning pipeline
of the study. The first stage performs self-supervised contrastive pretraining on prefix
samples derived from full vehicle trajectories in order to learn sequence representations
that are robust to stochastic augmentations. The second stage reuses the pretrained sequence
encoders for supervised survival modelling on the truncated sequences prepared for downstream
comparison with baseline methods.

The module contains modality-specific encoders for numerical and histogram sequence inputs,
a projection head used only during contrastive pretraining, the full self-supervised sequence
model, and a survival fine-tuning model that replaces the projection head with a supervised
risk prediction head.

This design ensures architectural consistency between pretraining and fine-tuning, allows
pretrained weights to be transferred reliably across notebooks, and supports a controlled
experimental comparison in which all supervised models are trained on the same truncated
sequence inputs.
"""
class HistogramEncoder(nn.Module):
    """
    Per-time-step feedforward encoder for histogram-bin sequence inputs.

    This encoder transforms each histogram feature vector independently at each
    time step using a small multilayer perceptron. It does not perform recurrent
    temporal modelling by itself; instead, it produces time-indexed hidden
    representations that remain aligned with the numerical sequence branch and
    are later fused and aggregated at the prefix level.

    Parameters
    ----------
    input_dim : int, default=97
        Number of histogram-bin features at each time step.
    hidden_dim : int, default=64
        Dimension of the hidden representation produced for each time step.

    Inputs
    ------
    x : torch.Tensor
        Histogram sequence tensor of shape (B, T, D_hist).

    Returns
    -------
    torch.Tensor
        Encoded histogram sequence of shape (B, T, H_hist).
    """
    def __init__(self, input_dim: int = 97, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DecayNumericalEncoder(nn.Module):
    """
    Recurrent numerical sequence encoder with explicit hidden-state decay.

    This encoder processes numerical time-series inputs using a GRUCell-based
    recurrent architecture. Before each recurrent update, the previous hidden
    state is multiplicatively decayed using a learned function of the elapsed
    time gap since the previous observation. This allows the encoder to account
    explicitly for irregular sampling intervals rather than only treating time
    gaps as ordinary input features.

    Padded positions are handled through a validity mask. At padded time steps,
    the hidden state is left unchanged so that artificial padding does not alter
    the recurrent dynamics.

    Parameters
    ----------
    numerical_dim : int, default=8
        Number of numerical features at each time step.
    hidden_dim : int, default=64
        Dimension of the recurrent hidden state.

    Inputs
    ------
    numerical : torch.Tensor
        Numerical sequence tensor of shape (B, T, D_num).
    time_gaps : torch.Tensor
        Time-gap tensor of shape (B, T), containing elapsed time since the
        previous readout.
    padding_mask : torch.Tensor
        Boolean mask of shape (B, T), where True indicates a valid time step
        and False indicates padding.

    Returns
    -------
    torch.Tensor
        Encoded numerical hidden-state sequence of shape (B, T, H_num).
    """
    def __init__(
        self,
        numerical_dim: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.gru_cell = nn.GRUCell(
            input_size=numerical_dim,
            hidden_size=hidden_dim,
        )

        self.decay_layer = nn.Linear(1, hidden_dim)

    def forward(
        self,
        numerical: torch.Tensor,
        time_gaps: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = numerical.shape
        device = numerical.device

        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        hidden_states = []

        for t in range(seq_len):
            x_t = numerical[:, t, :]
            dt_t = time_gaps[:, t].unsqueeze(-1)
            valid_t = padding_mask[:, t].unsqueeze(-1).float()

            gamma_t = torch.exp(-torch.relu(self.decay_layer(dt_t)))
            h_decayed = gamma_t * h
            h_candidate = self.gru_cell(x_t, h_decayed)

            h = valid_t * h_candidate + (1.0 - valid_t) * h
            hidden_states.append(h.unsqueeze(1))

        hidden_seq = torch.cat(hidden_states, dim=1)
        hidden_seq = hidden_seq * padding_mask.unsqueeze(-1).float()
        return hidden_seq


class ProjectionHead(nn.Module):
    """
    Projection head used for contrastive self-supervised pretraining.

    This module maps pooled sequence representations into a lower-dimensional
    projection space in which the contrastive objective is optimized. It is used
    only during the self-supervised stage and is discarded during downstream
    survival fine-tuning, where the pooled encoder representation is passed to a
    supervised risk prediction head instead.

    Parameters
    ----------
    input_dim : int
        Dimension of the pooled fused representation.
    projection_dim : int, default=64
        Dimension of the output projection used in contrastive learning.
    hidden_dim : int, default=128
        Dimension of the intermediate hidden layer.

    Inputs
    ------
    x : torch.Tensor
        Pooled sequence representation of shape (B, D_in).

    Returns
    -------
    torch.Tensor
        Projected representation of shape (B, D_proj).
    """
    def __init__(self, input_dim: int, projection_dim: int = 64, hidden_dim: int = 128):
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


class SSLSequenceModel(nn.Module):
    """
    Self-supervised sequence model for contrastive pretraining on prefix samples.

    This model combines modality-specific encoders for numerical and histogram
    inputs, fuses their time-indexed hidden representations, aggregates them
    into a fixed-length prefix representation through masked mean pooling, and
    maps the pooled representation into a contrastive projection space.

    It is intended for the self-supervised pretraining stage, where two
    augmented views of the same prefix are encoded and optimized with a
    contrastive loss. The pooled representation learned by this model serves as
    the transferable sequence embedding used later during supervised survival
    fine-tuning.

    Parameters
    ----------
    numerical_dim : int, default=8
        Number of numerical input features per time step.
    histogram_dim : int, default=97
        Number of histogram-bin input features per time step.
    numerical_hidden_dim : int, default=64
        Hidden dimension of the numerical encoder.
    histogram_hidden_dim : int, default=64
        Hidden dimension of the histogram encoder.
    projection_dim : int, default=64
        Dimension of the projection head output.

    Inputs
    ------
    numerical : torch.Tensor
        Numerical sequence tensor of shape (B, T, D_num).
    histogram : torch.Tensor
        Histogram sequence tensor of shape (B, T, D_hist).
    time_gaps : torch.Tensor
        Time-gap tensor of shape (B, T).
    padding_mask : torch.Tensor
        Boolean mask of shape (B, T), where True indicates a valid position.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - pooled representation of shape (B, D_fused)
        - projection of shape (B, D_proj)
    """
    def __init__(
        self,
        numerical_dim: int = 8,
        histogram_dim: int = 97,
        numerical_hidden_dim: int = 64,
        histogram_hidden_dim: int = 64,
        projection_dim: int = 64,
    ):
        super().__init__()

        self.numerical_encoder = DecayNumericalEncoder(
            numerical_dim=numerical_dim,
            hidden_dim=numerical_hidden_dim,
        )

        self.histogram_encoder = HistogramEncoder(
            input_dim=histogram_dim,
            hidden_dim=histogram_hidden_dim,
        )

        fused_dim = numerical_hidden_dim + histogram_hidden_dim
        self.projection_head = ProjectionHead(
            input_dim=fused_dim,
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

    def forward(
        self,
        numerical: torch.Tensor,
        histogram: torch.Tensor,
        time_gaps: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_hidden = self.numerical_encoder(
            numerical, time_gaps, padding_mask
        )

        hist_hidden = self.histogram_encoder(histogram)

        fused = torch.cat([num_hidden, hist_hidden], dim=-1)
        pooled = self.masked_mean_pool(fused, padding_mask)
        proj = self.projection_head(pooled)

        return pooled, proj


class SurvivalFineTuningModel(nn.Module):
    """
    Supervised survival model initialized from pretrained SSL encoders.

    This model reuses the pretrained numerical and histogram encoders learned
    during the self-supervised stage, keeps the same fusion and masked pooling
    mechanism, and replaces the contrastive projection head with a supervised
    survival head that outputs a scalar risk score for each truncated sequence.

    It is designed for downstream survival fine-tuning on the labelled truncated
    sequences used in the controlled comparison with baseline survival models.
    The encoder can optionally be frozen to support lightweight proxy
    evaluation of representation quality before full end-to-end fine-tuning.

    Parameters
    ----------
    pretrained_ssl_model : nn.Module
        Previously initialized or pretrained SSL sequence model from which the
        encoders and pooling logic are reused.
    survival_hidden_dim : int, default=64
        Hidden dimension of the supervised survival head.
    freeze_encoder : bool, default=False
        If True, encoder parameters are frozen and only the survival head is
        trained.

    Inputs
    ------
    numerical : torch.Tensor
        Numerical sequence tensor of shape (B, T, D_num).
    histogram : torch.Tensor
        Histogram sequence tensor of shape (B, T, D_hist).
    time_gaps : torch.Tensor
        Time-gap tensor of shape (B, T).
    padding_mask : torch.Tensor
        Boolean mask of shape (B, T), where True indicates a valid position.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - pooled sequence representation of shape (B, D_fused)
        - scalar risk scores of shape (B,)
    """
    def __init__(
        self,
        pretrained_ssl_model: nn.Module,
        survival_hidden_dim: int = 64,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        self.numerical_encoder = pretrained_ssl_model.numerical_encoder
        self.histogram_encoder = pretrained_ssl_model.histogram_encoder
        self.masked_mean_pool = pretrained_ssl_model.masked_mean_pool

        fused_dim = (
            pretrained_ssl_model.numerical_encoder.hidden_dim +
            pretrained_ssl_model.histogram_encoder.mlp[0].out_features
        )

        self.survival_head = nn.Sequential(
            nn.Linear(fused_dim, survival_hidden_dim),
            nn.ReLU(),
            nn.Linear(survival_hidden_dim, 1)
        )

        if freeze_encoder:
            for param in self.numerical_encoder.parameters():
                param.requires_grad = False
            for param in self.histogram_encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        numerical: torch.Tensor,
        histogram: torch.Tensor,
        time_gaps: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        num_hidden = self.numerical_encoder(
            numerical, time_gaps, padding_mask
        )

        hist_hidden = self.histogram_encoder(histogram)

        fused = torch.cat([num_hidden, hist_hidden], dim=-1)
        pooled = self.masked_mean_pool(fused, padding_mask)
        risk = self.survival_head(pooled).squeeze(-1)

        return pooled, risk