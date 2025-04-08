import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from torch import nn

from hmmglm import distributions, inference
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


class LitOneHotHMMGLM(L.LightningModule):
    def __init__(
        self,
        decoder: nn.Module,
        learning_rate: float,
        dataset_size: int,
        forward_backward_every_n_epochs: int = 1,
    ):
        super().__init__()
        self.decoder = decoder
        self.learning_rate = learning_rate
        self.dataset_size = dataset_size
        self.forward_backward_every_n_epochs = forward_backward_every_n_epochs

        self.gamma_list = [None for _ in range(self.dataset_size)]
        self.xi_list = [None for _ in range(self.dataset_size)]

        self.save_hyperparameters(ignore=["decoder"])

    def training_step(self, batch, batch_idx):
        idx = batch[0][0]
        x = batch[1][0]  # (n_time_bins, n_neurons)
        x_pred_mean = self.decoder(x)  # (n_states, n_time_bins, n_neurons)
        emission_log_prob = (
            distributions.poisson_log_likelihood(x[None, :, :], x_pred_mean)
            .sum(dim=-1)
            .T
        )  # (n_time_bins, n_states)

        if self.current_epoch % self.forward_backward_every_n_epochs == 0:
            gamma, xi = inference.forward_backward(
                emission_log_prob, self.decoder.transition_matrix, algorithm="logsumexp"
            )
            self.gamma_list[idx] = gamma
            self.xi_list[idx] = xi

        expected_log_prob = (
            inference.baum_welch(
                emission_log_prob,
                self.decoder.transition_matrix,
                self.gamma_list[idx],
                self.xi_list[idx],
            )
            / x.shape[0]
        )

        adjacency = self.decoder.adjacency
        log_adjacency = F.log_softmax(
            self.decoder.parametrizations["adjacency"].original, dim=-1
        )

        if self.decoder.adjacency_distribution == "gumbel-softmax":
            logit_adjacency_prior = self.decoder.logit_adjacency_prior.expand_as(
                log_adjacency
            )
            adjacency_log_prob = distributions.gumbel_softmax_log_likelihood_log_input(
                log_adjacency,
                logit_adjacency_prior,
                tau=0.5,
            ).sum()
        elif self.decoder.adjacency_distribution == "categorical":
            log_adjacency_prior = F.log_softmax(
                self.decoder.logit_adjacency_prior, dim=-1
            ).expand_as(adjacency)
            adjacency_log_prob = (adjacency * log_adjacency_prior).sum()
        else:
            raise ValueError(
                f"Invalid adjacency_distribution: {self.decoder.adjacency_distribution}"
            )

        adjacency_entropy = -(adjacency * log_adjacency).sum()

        loss = -expected_log_prob - adjacency_log_prob
        if self.decoder.adjacency_distribution == "categorical":
            loss += adjacency_entropy

        self.log_dict(
            {
                "loss": loss.item(),
                "expected_log_prob": expected_log_prob.item(),
                "adjacency_log_prob": adjacency_log_prob.item(),
                "adjacency_entropy": adjacency_entropy.item(),
            },
        )
        return loss

    def on_train_epoch_end(self):
        self.decoder.parametrizations["adjacency"].original.data = F.log_softmax(
            self.decoder.parametrizations["adjacency"].original.data, dim=-1
        ).clamp(-20, 0)

    def on_test_epoch_start(self):
        self.df_result = pd.DataFrame(
            columns=["conditional_log_prob", "firing_rates", "latents"],
        )
        self.df_result = self.df_result.astype(
            {
                "conditional_log_prob": float,
            }
        )

    def test_step(self, batch, batch_idx):
        idx = batch[0][0].item()
        x = batch[1][0]  # (n_time_bins, n_neurons)
        x_pred_mean = self.decoder(x)  # (n_states, n_time_bins, n_neurons)
        emission_log_prob = (
            distributions.poisson_log_likelihood(x[None, :, :], x_pred_mean)
            .sum(dim=-1)
            .T
        )  # (n_time_bins, n_states)
        inferred_states = inference.viterbi(
            emission_log_prob, self.decoder.transition_matrix
        )  # (n_time_bins,)
        x_pred_mean = x_pred_mean[
            inferred_states, torch.arange(len(x))
        ]  # (n_time_bins, n_neurons)

        # conditional_log_prob = distributions.poisson_log_likelihood(x, x_pred_mean).sum(
        #     dim=-1
        # )  # (n_time_bins,)
        conditional_log_prob = emission_log_prob[torch.arange(len(x)), inferred_states]
        self.df_result.at[idx, "conditional_log_prob"] = (
            conditional_log_prob.mean().item()
        )
        self.df_result.at[idx, "firing_rates"] = x_pred_mean.to("cpu")
        self.df_result.at[idx, "latents"] = inferred_states.to("cpu")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


