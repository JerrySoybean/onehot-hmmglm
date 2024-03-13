import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from abc import abstractmethod

from hmmglm import utils


class Nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        pass


class Linear(Nonlinearity):
    def __init__(self):
        super().__init__()
        self.name = 'linear'
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x


class Softplus(Nonlinearity):
    def __init__(self, beta: float = 1, lowerbound: float = 0):
        super().__init__()
        self.name = 'softplus'
        self.beta = beta
        self.lowerbound = lowerbound
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return F.softplus(x, beta=self.beta) + self.lowerbound


class Sigmoid(Nonlinearity):
    def __init__(self, lowerbound: float = 1e-8, upperbound: float = 1):
        super().__init__()
        self.name = 'sigmoid'
        self.lowerbound = lowerbound
        self.upperbound = upperbound
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return (self.upperbound - self.lowerbound) * torch.sigmoid(x) + self.lowerbound


class Exp(Nonlinearity):
    def __init__(self, lowerbound: float = 0):
        super().__init__()
        self.name = 'exp'
        self.lowerbound = lowerbound
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return torch.exp(x) + self.lowerbound