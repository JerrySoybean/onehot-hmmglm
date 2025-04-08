# %%
from importlib import reload

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from hmmglm.models import model, onehot

reload(onehot)
reload(model)

# %%
a = nn.Conv1d(5, 10, 3, bias=False)
parametrize.register_parametrization(a, "weight", nn.Softmax())
print(a.weight)

# %%
