import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from hmmglm import distributions, inference
from hmmglm.models.parametrizefunc import Softmax


class Base(nn.Module):
    def __init__(
        self,
        activation: str = "softplus",
    ):
        super().__init__()
        if activation == "softplus":
            self.activation = F.softplus
        elif activation == "log1psoftplus":
            self.activation = lambda x: torch.log1p(F.softplus(x))
        elif activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "sqrtsoftplus":
            self.activation = lambda x: torch.sqrt(F.softplus(x))
        else:
            raise ValueError(
                "Activation must be 'softplus', 'log1psoftplus', 'sigmoid', or 'sqrtsoftplus'"
            )
