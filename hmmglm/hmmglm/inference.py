import torch
import torch.nn.functional as F
from torch import nn, vmap


@torch.no_grad()
def forward_backward(
    emission_log_prob: torch.Tensor,
    transition_matrix: torch.Tensor,
    init_p: torch.Tensor = None,
    algorithm: str = "logsumexp",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward-backward algorithm for HMMs.

    Parameters
    ----------
    emission_log_prob : torch.Tensor of shape (n_time_bins, n_states)
        Log probabilities of the emissions.
    transition_matrix : torch.Tensor of shape (n_states, n_states)
        Transition matrix.
    init_p : torch.Tensor of shape (n_states,), optional
        Initial state probabilities, by default None.
    algorithm : str, optional
        The algorithm to use ["logsumexp" | "scaling"], by default "logsumexp", since it is more numerically stable.

    Returns
    -------
    gamma : torch.Tensor of shape (n_time_bins, n_states)
        Posterior probabilities of the states.
    xi : torch.Tensor of shape (n_time_bins - 1, n_states, n_states)
        Joint probabilities of the states.
    """

    n_time_bins, n_states = emission_log_prob.shape
    device = emission_log_prob.device
    if init_p is None:
        init_p = torch.ones(n_states, device=device) / n_states

    if algorithm == "scaling":
        emission = emission_log_prob.exp().clamp(min=1e-16)
        alpha = torch.zeros((n_time_bins, n_states), device=device)
        c = torch.zeros((n_time_bins,), device=device)
        alpha[0] = init_p * emission[0]
        c[0] = alpha[0].sum()
        alpha[0] = alpha[0] / c[0]

        for t in range(1, n_time_bins):
            alpha[t] = emission[t] * (transition_matrix.T @ alpha[t - 1])
            c[t] = alpha[t].sum()
            alpha[t] = alpha[t] / c[t]

        beta = torch.zeros((n_time_bins, n_states), device=device)
        beta[-1] = 1
        for t in range(n_time_bins - 2, -1, -1):
            beta[t] = transition_matrix @ (beta[t + 1] * emission[t + 1]) / c[t + 1]

        gamma = alpha * beta  # posterior probability of hidden
        if gamma.isnan().sum() > 0:
            raise ValueError()
        xi = (
            1
            / c[1:, None, None]
            * alpha[:-1, :, None]
            * emission[1:, None, :]
            * transition_matrix[None, :, :]
            * beta[1:, None, :]
        )  # (n_time_bins-1) x n_states x n_states
    elif algorithm == "logsumexp":
        log_transition_matrix = torch.log(transition_matrix)
        log_alpha = torch.zeros((n_time_bins, n_states), device=device)
        log_alpha[0] = torch.log(init_p) + emission_log_prob[0]

        for t in range(1, n_time_bins):
            log_alpha[t] = emission_log_prob[t] + torch.logsumexp(
                log_transition_matrix.T + log_alpha[t - 1], dim=1
            )

        log_beta = torch.zeros((n_time_bins, n_states), device=device)
        log_beta[-1] = 0
        for t in range(n_time_bins - 2, -1, -1):
            log_beta[t] = torch.logsumexp(
                log_transition_matrix + log_beta[t + 1] + emission_log_prob[t + 1],
                dim=1,
            )
        log_marginal = torch.logsumexp(log_alpha[-1], dim=0)
        log_gamma = log_alpha + log_beta - log_marginal
        gamma = F.softmax(log_gamma, dim=1)  # because not sum to one

        log_xi = (
            log_alpha[:-1, :, None]
            + log_transition_matrix[None, :, :]
            + log_beta[1:, None, :]
            + emission_log_prob[1:, None, :]
            - log_marginal
        )
        xi = F.softmax(log_xi.reshape((n_time_bins - 1, n_states**2)), dim=1).reshape(
            (n_time_bins - 1, n_states, n_states)
        )
    else:
        raise ValueError("algorithm must be either 'scaling' or 'logsumexp'")

    return gamma, xi


batch_forward_backward = vmap(
    forward_backward, in_dims=(0, None, None), out_dims=(0, 0)
)


def baum_welch(
    emission_log_prob: torch.Tensor,
    transition_matrix: torch.Tensor,
    gamma: torch.Tensor,
    xi: torch.Tensor,
    init_p: torch.Tensor = None,
) -> torch.Tensor:
    """
    Baum-Welch algorithm to estimate the model parameters.

    Parameters
    ----------
    emission_log_prob : torch.Tensor of shape (n_time_bins, n_states)
        Log probabilities of the emissions.
    transition_matrix : torch.Tensor of shape (n_states, n_states)
        Transition matrix.
    gamma : torch.Tensor of shape (n_time_bins, n_states)
        Posterior probabilities of the states.
    xi : torch.Tensor of shape (n_time_bins - 1, n_states, n_states)
        Joint probabilities of the states.
    init_p : torch.Tensor of shape (n_states,), optional
        Initial state probabilities, by default None.

    Returns
    -------
    torch.Tensor
        The HMM target function to be maximized.
    """
    n_time_bins, n_states = emission_log_prob.shape
    if init_p is None:
        init_p = torch.ones(n_states, device=emission_log_prob.device) / n_states

    term_1 = torch.sum(gamma[0] + init_p.log())
    term_2 = torch.sum(torch.sum(xi, dim=0) * transition_matrix.log())
    term_3 = torch.sum(gamma * emission_log_prob)
    return term_1 + term_2 + term_3


batch_baum_welch = vmap(baum_welch, in_dims=(0, None, 0, 0), out_dims=0)


@torch.no_grad()
def viterbi(
    emission_log_prob: torch.Tensor,
    transition_matrix: torch.Tensor,
    init_p: torch.Tensor = None,
) -> torch.Tensor:
    """Viterbi algorithm to inference the most probable latent state sequence."""

    n_time_bins, n_states = emission_log_prob.shape
    omega = torch.zeros((n_time_bins, n_states), device=emission_log_prob.device)
    psi = torch.zeros(
        (n_time_bins, n_states), dtype=torch.int64, device=emission_log_prob.device
    )
    log_transition_matrix = torch.log(transition_matrix)

    if init_p is None:
        init_p = torch.ones(n_states, device=emission_log_prob.device) / n_states
    omega[0] = init_p.log() + emission_log_prob[0]

    for t in range(1, n_time_bins):
        temp_matrix = log_transition_matrix + omega[t - 1][:, None]
        values, psi[t] = torch.max(temp_matrix, dim=0)
        omega[t] = emission_log_prob[t] + values

    state_pred = torch.zeros(n_time_bins, dtype=torch.int64)
    state_pred[-1] = omega[-1].argmax()
    for t in range(n_time_bins - 1, 0, -1):
        state_pred[t - 1] = psi[t, state_pred[t]]
    return state_pred


batch_viterbi = vmap(viterbi, in_dims=(0, None), out_dims=0)
