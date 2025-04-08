import torch

from hmmglm.utils import pad_for_convolve, pre_convolve


def test_pad_for_convolve():
    torch.manual_seed(0)

    n_trials = 5
    n_time_bins = 100
    n_neurons = 10

    kernel_size = 5

    spikes = torch.randn(n_trials, n_time_bins, n_neurons)
    padded_spikes = pad_for_convolve(spikes, kernel_size)

    expected_padded_spikes = torch.concat(
        [torch.zeros(n_trials, kernel_size, n_neurons), spikes[:, :-1, :]], dim=-2
    )
    assert torch.equal(
        padded_spikes, expected_padded_spikes
    ), f"Expected {expected_padded_spikes}, but got {padded_spikes}"


def test_pre_convolve():
    torch.manual_seed(0)

    n_trials = 5
    n_time_bins = 100
    n_neurons = 10

    kernel_size = 5
    kernel = torch.randn(kernel_size)

    spikes = torch.randn(n_trials, n_time_bins, n_neurons)
    convolved_spikes = pre_convolve(spikes, kernel)

    padded_spikes = pad_for_convolve(spikes, kernel_size)
    expected_convolved_spikes = torch.zeros(n_trials, n_time_bins, n_neurons)
    for k in range(kernel_size):
        expected_convolved_spikes += (
            padded_spikes[:, k : n_time_bins + k, :] * kernel[k]
        )
    assert torch.allclose(
        convolved_spikes, expected_convolved_spikes, atol=1e-6
    ), f"Expected {expected_convolved_spikes}, but got {convolved_spikes}"
