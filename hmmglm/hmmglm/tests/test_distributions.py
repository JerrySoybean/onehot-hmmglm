import torch
import torch.distributions as D
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import nn

from hmmglm import distributions


def test_gumbel_softmax_log_likelihood():
    k = 5
    n_samples = 10
    logit_x = torch.randn(n_samples, k)
    log_x = F.log_softmax(logit_x, dim=-1)
    x = F.softmax(logit_x, dim=-1)

    logit_mean = torch.randn(k)
    log_mean = F.log_softmax(logit_mean, dim=-1)
    mean = F.softmax(logit_mean, dim=-1)

    tau = 0.5
    log_likelihood = distributions.gumbel_softmax_log_likelihood(x, mean, tau)
    log_likelihood_log_input = distributions.gumbel_softmax_log_likelihood_log_input(
        log_x, log_mean, tau
    )

    assert torch.allclose(log_likelihood, log_likelihood_log_input)


def test_gumbel_softmax_log_likelihood_unnormalized():
    k = 5
    n_samples = 10
    logit_x = torch.randn(n_samples, k)
    log_x = F.log_softmax(logit_x, dim=-1)

    logit_mean = torch.randn(k)
    log_mean = F.log_softmax(logit_mean, dim=-1)

    tau = 0.5
    log_likelihood_1 = distributions.gumbel_softmax_log_likelihood_log_input(
        log_x, logit_mean, tau
    )
    log_likelihood_2 = distributions.gumbel_softmax_log_likelihood_log_input(
        log_x, log_mean, tau
    )

    assert torch.allclose(log_likelihood_1, log_likelihood_2)


def test_train_gumbel_softmax():
    tau = 0.5
    k = 5
    n_samples = 1000
    logit_mean_true = torch.randn(k)
    mean_true = F.softmax(logit_mean_true, dim=-1)
    x = F.gumbel_softmax(logit_mean_true.expand(n_samples, k), tau=tau, dim=-1)
    log_x = x.log()

    logit_mean = nn.Parameter(torch.randn(k))
    optimizer = torch.optim.AdamW([logit_mean], lr=1e-2)

    n_epochs = 500
    n_print = 20
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        log_likelihood = distributions.gumbel_softmax_log_likelihood_log_input(
            log_x, logit_mean, tau
        )
        loss = -log_likelihood.mean()
        loss.backward()
        optimizer.step()
        if epoch % (n_epochs // n_print) == 0:
            print(f"epoch: {epoch}, loss: {loss.item()}", flush=True)
    with torch.no_grad():
        mean = F.softmax(logit_mean, dim=-1)

    assert torch.allclose(mean_true, mean, atol=1e-1)
