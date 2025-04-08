import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors

import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import permutations


def exp_basis(decay: float, window_size: int, time_span: float):
    """Exponential decay basis.
    
    \phi(t) = \beta exp(-\beta t)

    Parameters
    ----------
    decay : float
        Decay parameter.
    window_size : int
        Number of time bins descretized.
    time_span : float
        Max influence time span.

    Returns
    -------
    basis : ndarray of shape (window_size,)
        Descretized basis.
    """

    basis = torch.zeros(window_size)
    dt = time_span / window_size
    t = torch.linspace(dt, time_span, window_size)
    basis = torch.exp(-decay * t)
    basis /= (dt * basis.sum(axis=0)) # normalization
    return basis


def convolve_spikes_with_basis(spikes_list: torch.FloatTensor, basis: torch.FloatTensor, direction: str = 'forward') -> torch.FloatTensor:
    """Convolve soft spike train soft_spikes_list[:, :, j] with a single basis.

    Parameters
    ----------
    spikes_list : torch.FloatTensor of shape (n_seq, n_time_bins, n_neurons) or (n_seq, n_time_bins, n_neurons, max_n_spikes)
        Spike train. The values can be continuous that are from soft spike train.
    basis : torch.FloatTensor of shape (window_size,)
        Descretized basis.
    direction : str in ['forward' | 'backward']

    Returns
    -------
    convolved_spikes_list : torch.FloatTensor of shape (n_time_bins, n_neurons)
        Convolved spike train.
    """
    
    window_size = len(basis)
    if len(spikes_list.shape) == 4:
        spikes_list = spikes_list @ torch.arange(spikes_list.shape[-1], dtype=torch.float32)
    n_seq, n_time_bins, n_neurons = spikes_list.shape

    if direction == 'forward':
        convolved_spikes_list = torch.zeros_like(spikes_list)
        padded_spikes_list = torch.cat((torch.zeros((n_seq, window_size, n_neurons)), spikes_list), dim=-2)
        for i in range(window_size):
            convolved_spikes_list = convolved_spikes_list + basis[-(i+1)] * padded_spikes_list[:, i:n_time_bins+i, :]
        return convolved_spikes_list
    elif direction == 'backward':
        rev_convolved_spikes_list = torch.zeros_like(spikes_list)
        padded_spikes_list = torch.cat((spikes_list, torch.zeros((n_seq, window_size, n_neurons))), dim=-2)
        for i in range(window_size):
            rev_convolved_spikes_list = rev_convolved_spikes_list + basis[i] * padded_spikes_list[:, i+1:n_time_bins+i+1, :]
        return rev_convolved_spikes_list


def match_states(one_hot_true_states: torch.LongTensor, gamma: torch.FloatTensor, force=True):
    """Match the 
    Parameters
    ----------
    one_hot_true_states : torch.LongTensor of shape (n_seq, n_time_bins, n_states) or (n_time_bins, n_states)
        One-hot true state sequence(s).
    gamma : torch.FloatTensor of shape (n_seq, n_time_bins, n_states) or (n_time_bins, n_states)
        One-hot posteior probability or one-hot predicted state sequence(s).
    Returns
    -------
    true_to_learned : torch.LongTensor of shape (n_states,)
        `true_to_learned[s]` represents the state in the learned model that corresponds to the state `s` in the original model.
    """

    if len(gamma.shape) == 2:
        one_hot_true_states = one_hot_true_states[None, :]
        gamma = gamma[None, :, :]
    n_states = gamma.shape[2]
    true_to_learned = torch.zeros(n_states, dtype=torch.int64)
    if force is True:
        all_possible_permutations = torch.tensor(list(permutations(range(n_states))))
        n_possible_permutations = len(all_possible_permutations)
        mse_list = torch.zeros(n_possible_permutations)
        for permutation in range(n_possible_permutations):
            mse_list[permutation] = (one_hot_true_states - gamma[:, :, all_possible_permutations[permutation]]).square().mean()
        true_to_learned = all_possible_permutations[mse_list.argmin()]
    else:
        for state in range(n_states):
            true_to_learned[state] = (one_hot_true_states - gamma[:, :, [state]]).square().mean(dim=0).argmin()
    return true_to_learned


def match_weights(weights_true: torch.FloatTensor, weights_pred: torch.FloatTensor) -> torch.LongTensor:
    n_states = weights_true.shape[0]
    true_to_learned = torch.zeros(n_states, dtype=torch.int64)
    all_possible_permutations = torch.tensor(list(permutations(range(n_states))))
    n_possible_permutations = len(all_possible_permutations)
    mse_list = torch.zeros(n_possible_permutations)
    for permutation in range(n_possible_permutations):
        mse_list[permutation] = (weights_pred - weights_true[all_possible_permutations[permutation]]).square().mean()
    true_to_learned = all_possible_permutations[mse_list.argmin()]
    return true_to_learned


def visualize_matrix(matrix, ax, v=None):
    im = ax.matshow(matrix, cmap='seismic', vmin=-v, vmax=v)
    ax.tick_params(left=False, top=False, bottom=False, labelleft=False, labeltop=False)
    return im


def visualize_states(states: torch.LongTensor, ax, height=10):
    mat = states.expand((height, -1))
    ax.matshow(mat, vmin=0, vmax=9, cmap='tab10')
    ax.set(xticks=[], yticks=[])
    return ax


def visualize_states_energy(states: torch.LongTensor, energy: torch.FloatTensor, ax, height=10):
    n_time_bins = states.shape[0]
    color_list = torch.zeros((10, 3))
    tableau = list(mcolors.TABLEAU_COLORS.values())
    for i in range(10):
        h = tableau[i][1:]
        color_list[i] = torch.tensor(tuple(int(h[j:j+2], 16) for j in (0, 2, 4))) / 255
    mat = torch.zeros((n_time_bins, 4))
    mat[:, :3] = color_list[states]
    mat[:, 3] = energy
    mat = mat.expand((height, -1, -1))
    im = ax.imshow(mat)
    ax.set(xticks=[], yticks=[])
    return im