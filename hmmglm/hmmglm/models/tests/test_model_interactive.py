# %%
from hmmglm.models import model

# %%
decoder = model.HMMGLM(3, 5, 5)

# %%
decoder.transition_matrix

# %%
print(dict(decoder.named_parameters()))
# %%
dict(decoder.named_parameters())["parametrizations.transition_matrix.original"]
# %%
