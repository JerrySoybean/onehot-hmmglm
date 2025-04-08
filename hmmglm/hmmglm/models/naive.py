import lightning as L
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from torch import nn

from hmmglm import distributions, inference
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


class LitHMMGLM(L.LightningModule):
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

        self.save_hyperparameters(
            ignore=[
                "decoder",
            ]
        )

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
        loss = -expected_log_prob

        self.log_dict(
            {
                "loss": loss.item(),
                "expected_log_prob": expected_log_prob.item(),
            },
        )
        return loss

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

        conditional_log_prob = distributions.poisson_log_likelihood(x, x_pred_mean).sum(
            dim=-1
        )  # (n_time_bins,)
        self.df_result.at[idx, "conditional_log_prob"] = (
            conditional_log_prob.mean().item()
        )
        self.df_result.at[idx, "firing_rates"] = x_pred_mean.to("cpu")
        self.df_result.at[idx, "latents"] = inferred_states.to("cpu")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
