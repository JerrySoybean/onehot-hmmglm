import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from torch import nn

from hmmglm.models.base import Base
from hmmglm.models.parametrizefunc import Softmax


class HMMGLM(Base):
    def __init__(
        self,
        n_states: int,
        n_neurons: int,
        kernel_size: int,
        activation: str = "softplus",
    ) -> None:
        super().__init__(activation=activation)
        self.n_states = n_states
        self.n_neurons = n_neurons
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(
            in_channels=n_neurons,
            out_channels=n_states * n_neurons,
            kernel_size=kernel_size,
        )

        self.transition_matrix = nn.Parameter(torch.ones(n_states, n_states))
        parametrize.register_parametrization(self, "transition_matrix", Softmax())
        if n_states > 1:
            self.transition_matrix = torch.eye(n_states) * 0.98 + 0.02 / (
                n_states - 1
            ) * (1 - torch.eye(n_states))
        else:
            self.transition_matrix = torch.tensor([[1.0]])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., n_time_bins, n_neurons)

        n_time_bins = x.shape[-2]
        x = F.pad(
            x, (0, 0, self.kernel_size, -1), "constant", 0
        )  # (..., n_time_bins + kernel_size - 1, n_neurons)
        x = self.conv(x.transpose(-2, -1))  # (..., n_states * n_neurons, n_time_bins)
        if x.dim() == 2:
            x = x.reshape(self.n_states, self.n_neurons, n_time_bins).permute((0, 2, 1))
        elif x.dim() == 3:
            batch_size = x.shape[0]
            x = x.reshape(
                batch_size, self.n_states, self.n_neurons, n_time_bins
            ).permute((0, 1, 3, 2))
        else:
            raise ValueError("Input tensor must have 2 or 3 dimensions")
        x_pred_mean = self.activation(x)
        return x_pred_mean  # (..., n_states, n_time_bins, n_neurons)

    def update_transition_matrix(self, xi: torch.Tensor) -> None:
        self.transition_matrix = F.softmax(
            (xi.sum(dim=0) / xi.sum(dim=(0, 2))[:, None]).log().clamp(min=-8), dim=-1
        )

