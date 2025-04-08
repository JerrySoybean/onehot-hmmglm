# %%
import numpy as np
import torch
import torch.distributions as D
import torch.nn.functional as F
from scipy.io import loadmat

from hmmglm import inference

# %%
x = (loadmat("sp500.mat")["price_move"][:, 0] + 1) / 2
q = 0.7
emission_p = np.array([1 - q, q])
transition_matrix = np.array([[0.8, 0.2], [0.2, 0.8]])


def tiny_forward_backward(x, transition_matrix, emission_p):
    n_states = len(emission_p)
    init_p = np.array([0.8, 0.2])
    n_time_bins = len(x)

    alpha = np.zeros((n_time_bins, n_states))
    alpha[0] = init_p * (emission_p ** x[0] * (1 - emission_p) ** (1 - x[0]))

    for t in range(1, n_time_bins):
        alpha[t] = (emission_p ** x[t] * (1 - emission_p) ** (1 - x[t])) * (
            transition_matrix.T @ alpha[t - 1]
        )

    beta = np.zeros((n_time_bins, n_states))
    beta[-1] = 1
    for t in range(n_time_bins - 2, -1, -1):
        beta[t] = transition_matrix @ (
            beta[t + 1] * (emission_p ** x[t + 1] * (1 - emission_p) ** (1 - x[t + 1]))
        )

    complete_data_likelihood = alpha[-1].sum()
    gamma = alpha * beta / complete_data_likelihood  # posterior probability of hidden

    return gamma


gamma_ref = tiny_forward_backward(x, transition_matrix, emission_p)

# %%
transition_matrix = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
emission_p = torch.tensor(x[:, None]).to(torch.float32) * torch.log(
    torch.tensor([1 - q, q])[None, :]
) + (1 - torch.tensor(x[:, None]).to(torch.float32)) * torch.log(
    torch.tensor([q, 1 - q])[None, :]
)

gamma, _ = inference.forward_backward(
    emission_p, transition_matrix, torch.tensor([0.8, 0.2]), "logsumexp"
)

print(np.mean(gamma_ref - gamma.numpy()))
print(np.max(gamma_ref - gamma.numpy()))

# %%
gamma, _ = inference.forward_backward(
    emission_p, transition_matrix, torch.tensor([0.8, 0.2]), "scaling"
)

print(np.mean(gamma_ref - gamma.numpy()))
print(np.max(gamma_ref - gamma.numpy()))
