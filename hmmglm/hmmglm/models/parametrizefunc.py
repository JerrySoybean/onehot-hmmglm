import torch
import torch.nn.functional as F
from torch import nn


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=-1)

    def right_inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x.log()


class WeightedSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, weight, log_kernel):
        return weight.unsqueeze(-1) * F.softmax(log_kernel, dim=-1)

    def right_inverse(self, z):
        weight = z.sum(dim=-1)
        log_kernel = (z / weight.unsqueeze(-1)).log()
        return weight, log_kernel


class OffDiagWeightedSoftmax(nn.Module):
    def __init__(self, n_neurons: int):
        super().__init__()
        self.n_neurons = n_neurons
        self.register_buffer("diag_mask", ~torch.eye(n_neurons, dtype=bool))

    def forward(self, weight, log_kernel):
        weight = weight * self.diag_mask
        return weight.unsqueeze(-1) * F.softmax(log_kernel, dim=-1)

    def right_inverse(self, z):
        weight = z.sum(dim=-1)
        log_kernel = (z / weight.unsqueeze(-1)).log()
        weight = weight * self.diag_mask
        return weight, log_kernel


class Softplus(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    def right_inverse(self, x: torch.Tensor) -> torch.Tensor:
        return (x.exp() - 1).log()


class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def right_inverse(self, x: torch.Tensor) -> torch.Tensor:
        return (x / (1 - x)).log()


class Mask(nn.Module):
    def __init__(
        self,
        mask: torch.Tensor,
    ):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mask

    def right_inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x
