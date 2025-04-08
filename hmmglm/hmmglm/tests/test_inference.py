import torch
import torch.distributions as D
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st

from hmmglm import inference


def test_forward_backward():
    n_time_bins = 100
    n_states = 3

    torch.manual_seed(0)

    emission_log_prob = torch.randn(n_time_bins, n_states)
    log_transition_matrix = torch.randn(n_states, n_states)
    transition_matrix = F.softmax(log_transition_matrix, dim=1)

    gamma, xi = inference.forward_backward(
        emission_log_prob, transition_matrix, "logsumexp"
    )

    assert gamma.shape == (n_time_bins, n_states)
    assert xi.shape == (n_time_bins - 1, n_states, n_states)

    assert torch.allclose(gamma.sum(1), torch.ones(n_time_bins))
    assert torch.allclose(xi.sum((1, 2)), torch.ones(n_time_bins - 1))

    gamma_1, xi_1 = inference.forward_backward(
        emission_log_prob, transition_matrix, "scaling"
    )

    assert torch.allclose(gamma, gamma_1)
    assert torch.allclose(xi, xi_1)
