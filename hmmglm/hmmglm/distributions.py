import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from abc import abstractmethod

from hmmglm import utils


def bernoulli_log_likelihood(x: torch.FloatTensor, mean: torch.FloatTensor, eps: float = 1e-8) -> torch.FloatTensor:
    return x * (mean+eps).log() + (1-x) * (1-mean+eps).log()


def poisson_log_likelihood(x: torch.FloatTensor, mean: torch.FloatTensor, eps: float = 1e-8) -> torch.FloatTensor:
    return x * (mean+eps).log() - mean - torch.lgamma(x+1)


def gumbel_softmax_log_likelihood(x: torch.FloatTensor, ln_p: torch.FloatTensor, tau: float, eps=0) -> torch.FloatTensor:
    n_categories = ln_p.shape[-1]
    ln_x = (x + eps).log()
    ll = torch.lgamma(torch.tensor(n_categories)) + (n_categories - 1) * torch.tensor(tau).log() - n_categories * torch.logsumexp(ln_p - tau * ln_x, dim=-1) + (ln_p - (tau+1) * ln_x).sum(dim=-1)
    return ll


class Distribution(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def log_likelihood(self, x: torch.FloatTensor, mean: torch.FloatTensor, eps: float = 1e-8) -> torch.FloatTensor:
        pass

    @abstractmethod
    def sample(self, mean: torch.FloatTensor, rng: torch.Generator = None) -> torch.FloatTensor:
        pass


class Bernoulli(Distribution):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'Bernoulli'
    
    def log_likelihood(self, x: torch.FloatTensor, mean: torch.FloatTensor, eps: float = 1e-8) -> torch.FloatTensor:
        return bernoulli_log_likelihood(x, mean, eps=eps)
    
    def sample(self, mean: torch.FloatTensor, rng: torch.Generator = None) -> torch.FloatTensor:
        return torch.bernoulli(mean, generator=rng)


class Poisson(Distribution):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'Poisson'

    def log_likelihood(self, x: torch.FloatTensor, mean: torch.FloatTensor, eps: float = 1e-8) -> torch.FloatTensor:
        return poisson_log_likelihood(x, mean, eps=eps)

    def sample(self, mean: torch.FloatTensor, rng: torch.Generator = None, eps: float = 1e-8) -> torch.FloatTensor:
        return torch.poisson(mean + eps, generator=rng)


class Gaussian(Distribution):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'Gaussian'
        self.std = nn.Parameter(torch.tensor(1.))
    
    def log_likelihood(self, x: torch.FloatTensor, mean: torch.FloatTensor, eps: float = 1e-8) -> torch.FloatTensor:
        # return -F.gaussian_nll_loss(mean, x, self.std**2, full=True, eps=eps, reduction='none')
        return -1/2 * (x - mean)**2 / self.std**2 - self.std.log() - 1/2*torch.log(2 * torch.tensor(torch.pi))
    
    def sample(self, mean: torch.FloatTensor, rng: torch.Generator = None) -> torch.FloatTensor:
        return torch.normal(mean, self.std, generator=rng)


class GumbelSoftmax(Distribution):
    def __init__(self, n_categories: int, tau: float) -> None:
        super().__init__(n_categories)
        self.name = 'Gumbel-Softmax'
        self.tau = tau # tau >= 0.5 is important, tau is better to be between 0.5 and 1.
        
    def log_likelihood(self, x: torch.FloatTensor, ln_p: torch.FloatTensor, eps=0) -> torch.FloatTensor: # eps = 0 is important, tau >= 0.5 is enough to promise x is elligible.
        return gumbel_softmax_log_likelihood(x, ln_p, self.tau, eps=eps)
    
    def sample(self, ln_p: torch.FloatTensor, rng=None) -> torch.FloatTensor:
        return F.gumbel_softmax(ln_p, tau=self.tau)