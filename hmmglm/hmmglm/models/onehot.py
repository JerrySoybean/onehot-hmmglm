import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from torch import nn

from hmmglm.models.naive import HMMGLM
from hmmglm.models.parametrizefunc import Sigmoid, Softmax, Softplus


class OneHotHMMGLM(HMMGLM):
    def __init__(
        self,
        n_states: int,
        n_neurons: int,
        kernel_size: int,
        activation: str = "softplus",
        share_bias=False,
        share_kernel=True,
        diag_setting="free",
        adjacency_distribution: str = "categorical",
        adjacency_exists: bool = True,
    ) -> None:
        super().__init__(n_states, n_neurons, kernel_size, activation)
        self.share_bias = share_bias
        self.share_kernel = share_kernel
        self.diag_setting = diag_setting
        self.adjacency_distribution = adjacency_distribution
        self.adjacency_exists = adjacency_exists

        if share_bias is True:
            self.bias = nn.Parameter(torch.zeros(n_neurons))
        else:
            self.bias = nn.Parameter(torch.zeros(n_states, n_neurons))

        if share_kernel is True:
            self.conv_weight = nn.Parameter(
                torch.ones(1, n_neurons, n_neurons, kernel_size) / kernel_size
            )
        else:
            self.conv_weight = nn.Parameter(
                torch.ones(n_states, n_neurons, n_neurons, kernel_size) / kernel_size
            )
        parametrize.register_parametrization(self, "conv_weight", Softmax())

        diag_mask = ~torch.eye(self.n_neurons, dtype=torch.bool)
        diag_mask = diag_mask.unsqueeze(0).expand(self.n_states, -1, -1).contiguous()
        if diag_setting == "none":
            diag_mask = torch.ones_like(diag_mask)
        elif diag_setting == "zero":
            pass
        elif diag_setting == "shared_free":
            self.diag = nn.Conv1d(
                in_channels=n_neurons,
                out_channels=n_neurons,
                kernel_size=kernel_size,
                groups=n_neurons,
                bias=False,
            )
        elif diag_setting == "free":
            self.diag = nn.Conv1d(
                in_channels=n_neurons,
                out_channels=n_states * n_neurons,
                kernel_size=kernel_size,
                groups=n_neurons,
                bias=False,
            )
        else:
            raise ValueError(f"Invalid diag_setting: {diag_setting}")
        self.register_buffer("diag_mask", diag_mask)

        self.strength = nn.Parameter(torch.ones(n_states, n_neurons, n_neurons))
        parametrize.register_parametrization(self, "strength", Softplus())

        if adjacency_exists is True:
            self.adjacency = nn.Parameter(
                torch.ones(n_states, n_neurons, n_neurons, 3) / 3
            )
            self.logit_adjacency_prior = nn.Parameter(
                torch.zeros(n_neurons, n_neurons, 3)
            )
        else:
            self.adjacency = nn.Parameter(
                torch.ones(n_states, n_neurons, n_neurons, 2) / 2
            )
            self.logit_adjacency_prior = nn.Parameter(
                torch.zeros(n_neurons, n_neurons, 2)
            )
        parametrize.register_parametrization(self, "adjacency", Softmax())

    @property
    def weight(self):
        if self.training is True:
            adjacency = F.gumbel_softmax(
                self.parametrizations["adjacency"].original, tau=0.5
            )
        else:
            adjacency = self.adjacency
            adjacency = F.one_hot(
                adjacency.argmax(dim=-1), num_classes=adjacency.shape[-1]
            )
        adjaecncy_exc = (
            adjacency[..., 2] if self.adjacency_exists is True else adjacency[..., 1]
        )
        adjacency_inh = adjacency[..., 0]
        return (adjaecncy_exc - adjacency_inh) * self.strength

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., n_time_bins, n_neurons)

        n_time_bins = x.shape[-2]
        padded_x = F.pad(x, (0, 0, self.kernel_size, -1), "constant", 0).transpose(
            -1, -2
        )  # (..., n_neurons, n_time_bins + kernel_size - 1)
        masked_weight = self.weight * self.diag_mask  # (n_states, n_neurons, n_neurons)
        conv_weight = (masked_weight.unsqueeze(-1) * self.conv_weight).reshape(
            self.n_states * self.n_neurons, self.n_neurons, self.kernel_size
        )
        x = F.conv1d(
            padded_x,
            conv_weight,
        )  # (..., n_states * n_neurons, n_time_bins)
        x = x.reshape(
            *x.shape[:-2], self.n_states, self.n_neurons, n_time_bins
        ).transpose(
            -1, -2
        )  # (..., n_states, n_time_bins, n_neurons)

        if self.diag_setting == "shared_free":
            x = x + self.diag(padded_x).transpose(-1, -2).unsqueeze(1)
        elif self.diag_setting == "free":
            x = x + self.diag(padded_x).reshape(
                *x.shape[:-3], self.n_neurons, self.n_states, n_time_bins
            ).transpose(-3, -2).transpose(
                -2, -1
            )  # (..., n_states, n_time_bins, n_neurons)
        else:
            pass

        if self.share_bias is True:
            x = x + self.bias.unsqueeze(0).unsqueeze(0)
        else:
            x = x + self.bias.unsqueeze(1)
        x = self.activation(x)  # (..., n_states, n_time_bins, n_neurons)
        return x

