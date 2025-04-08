import torch
import torch.nn.functional as F
from torch import nn


def poisson_log_likelihood(
    x: torch.Tensor, mean: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    return x * (mean + eps).log() - mean - torch.lgamma(x + 1)


def gumbel_softmax_log_likelihood(
    x: torch.Tensor, mean: torch.Tensor, tau: float = 1.0
) -> torch.Tensor:
    k = torch.tensor(mean.shape[-1])
    tau = torch.tensor(tau)
    return (
        torch.lgamma(k)
        + (k - 1) * torch.log(tau)
        - k * torch.logsumexp(mean.log() - tau * x.log(), dim=-1)
        + (mean.log() - (tau + 1) * x.log()).sum(dim=-1)
    )


def gumbel_softmax_log_likelihood_log_input(
    log_x: torch.Tensor, log_mean: torch.Tensor, tau: float = 1.0
) -> torch.Tensor:
    """
    Compute the log likelihood of a Gumbel-Softmax distribution with log input.

    Parameters
    ----------
    log_x : torch.Tensor of shape (..., k)
        Log input to the Gumbel-Softmax. Must be normalized.
    log_mean : torch.Tensor of shape (..., k)
        Log mean of the Gumbel-Softmax. Can be normalized or unnormalized.
    tau : float
        Temperature parameter of the Gumbel-Softmax

    Returns
    -------
    torch.Tensor
        Log likelihood of the Gumbel-Softmax
    """
    k = torch.tensor(log_mean.shape[-1])
    tau = torch.tensor(tau)
    return (
        torch.lgamma(k)
        + (k - 1) * torch.log(tau)
        - k * torch.logsumexp(log_mean - tau * log_x, dim=-1)
        + (log_mean - (tau + 1) * log_x).sum(dim=-1)
    )
