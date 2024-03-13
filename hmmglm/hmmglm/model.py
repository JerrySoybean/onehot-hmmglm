import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from abc import abstractmethod, abstractproperty

from hmmglm import utils, distributions, nonlinearities


class HMMGLM(nn.Module):
    def __init__(
            self,
            n_states: int,
            n_neurons: int,
            basis: torch.FloatTensor,
            distribution: distributions.Distribution = distributions.Poisson(),
            activation: nonlinearities.Nonlinearity = nonlinearities.Sigmoid()
    ) -> None:
        super().__init__()

        self.n_states = n_states
        self.n_neurons = n_neurons
        self.basis = basis
        self.flipped_basis = torch.flip(self.basis, (0,))
        self.window_size = len(self.basis)
        self.distribution = distribution
        self.activation = activation

        ## initialize parameters
        self._bias = nn.Parameter(torch.zeros((n_neurons))) # default is shared bias across all states
        if self.n_states > 1:
            self.transition_matrix = nn.Parameter(torch.eye(n_states) * 0.98 + 0.02 / (n_states - 1) * (1 - torch.eye(n_states)), requires_grad=False)
    
    @property
    def bias(self):
        if len(self._bias.shape) == 1:
            return self._bias.expand(self.n_states, -1)
        else:
            return self._bias
    
    @abstractproperty
    def weight(self):
        pass

    @abstractmethod
    def empty_diagonal(self) -> None:
        pass
    
    @abstractmethod
    def permute_states(self, true_to_learned: torch.LongTensor) -> None:
        pass
    
    def firing_rates(self, convolved_spikes: torch.FloatTensor, states: torch.LongTensor = None) -> torch.FloatTensor:
        if states is None:
            return self.activation(self.bias[:, None, :] + convolved_spikes @ self.weight.permute((0, 2, 1))) # (n_states, n_time_bins, n_neurons)
        else:
            return self.activation(self.bias[states] + (convolved_spikes[:, None, :] @ self.weight[states].permute((0, 2, 1)))[:, 0, :])
    
    def sample(self, n_time_bins: int, n_samples: int = 1) -> torch.FloatTensor:
        with torch.no_grad():
            spikes_list = torch.zeros((n_samples, n_time_bins + self.window_size, self.n_neurons))
            convolved_spikes_list = torch.zeros((n_samples, n_time_bins, self.n_neurons))
            firing_rates_list = torch.zeros((n_samples, n_time_bins, self.n_neurons))
            states_list = torch.zeros((n_samples, n_time_bins), dtype=torch.int64)

            # simulate states
            states_list[:, 0] = torch.multinomial(1/self.n_states * torch.ones(self.n_states), num_samples=n_samples, replacement=True)
            for t in range(n_time_bins):
                states_list[:, t] = torch.multinomial(self.transition_matrix[states_list[:, t-1]], num_samples=1)[:, 0]

            for t in range(n_time_bins):
                state_list = states_list[:, t]
                convolved_spikes_list[:, t, :] = self.flipped_basis @ spikes_list[:, t:t+self.window_size, :]
                firing_rates_list[:, t, :] = self.activation(self.bias[state_list] + (convolved_spikes_list[:, [t], :] @ self.weight[state_list].permute((0, 2, 1)))[:, 0, :])
                spikes_list[:, t+self.window_size, :] = self.distribution.sample(firing_rates_list[:, t, :])
            spikes_list = spikes_list[:, self.window_size:, :]
            return spikes_list, convolved_spikes_list, firing_rates_list, states_list
    
    def forward_backward(self, spikes: torch.FloatTensor, convolved_spikes: torch.FloatTensor, algorithm: str = 'scaling') -> tuple:
        """Forward backward algorithm, corrsponding to the E-step of the EM algorithm.

        """

        with torch.no_grad():
            init_p = torch.ones(self.n_states) / self.n_states
            n_time_bins = len(spikes)
            firing_rates_in_all_states = self.firing_rates(convolved_spikes) # n_states x n_time_bins x n_neurons
            log_emission = self.distribution.log_likelihood(spikes[None, :, :], firing_rates_in_all_states).sum(dim=(-1,)).permute((1, 0)) # n_time_bins x n_states
            
            if algorithm == 'scaling':
                emission = log_emission.exp().clamp(min=1e-16)
                alpha = torch.zeros((n_time_bins, self.n_states))
                c = torch.zeros((n_time_bins,))
                alpha[0] = init_p * emission[0]
                c[0] = alpha[0].sum()
                alpha[0] = alpha[0] / c[0]
                
                for t in range(1, n_time_bins):
                    alpha[t] = emission[t] * (self.transition_matrix.T @ alpha[t-1])
                    c[t] = alpha[t].sum()
                    alpha[t] = alpha[t] / c[t]
                
                beta = torch.zeros((n_time_bins, self.n_states))
                beta[-1] = 1
                for t in range(n_time_bins - 2, -1, -1):
                    beta[t] = self.transition_matrix @ (beta[t + 1] * emission[t+1]) / c[t+1]
                
                gamma = alpha * beta # posterior probability of hidden
                if gamma.isnan().sum() > 0:
                    raise ValueError()
                xi = 1/c[1:, None, None] * alpha[:-1, :, None] * emission[1:, None, :] * self.transition_matrix[None, :, :] * beta[1:, None, :] # (n_time_bins-1) x n_states x n_states
                
            elif algorithm == 'logsumexp':
                log_transition_matrix = (self.transition_matrix).log()
                log_alpha = torch.zeros((n_time_bins, self.n_states))
                log_alpha[0] = torch.log(init_p) + log_emission[0]
                
                for t in range(1, n_time_bins):
                    log_alpha[t] = log_emission[t] + torch.logsumexp(log_transition_matrix.T + log_alpha[t - 1], dim=1)
                
                log_beta = torch.zeros((n_time_bins, self.n_states))
                log_beta[-1] = 0
                for t in range(n_time_bins - 2, -1, -1):
                    log_beta[t] = torch.logsumexp(log_transition_matrix + log_beta[t + 1] + log_emission[t+1], dim=1)
                
                log_complete_data_likelihood = torch.logsumexp(log_alpha[-1], dim=0)
                log_gamma = log_alpha + log_beta - log_complete_data_likelihood # posterior probability of hidden
                gamma = F.softmax(log_gamma, dim=1) # because not sum to one

                log_xi = log_alpha[:-1, :, None] + log_emission[1:, None, :] + log_transition_matrix[None, :, :] + log_beta[1:, None, :] - log_complete_data_likelihood
                xi = F.softmax(log_xi.reshape((n_time_bins-1, self.n_states**2)), dim=1).reshape((n_time_bins-1, self.n_states, self.n_states))
            return gamma, xi
        
    def m_step(self, spikes: torch.FloatTensor, convolved_spikes: torch.FloatTensor, gamma: torch.FloatTensor, xi: torch.FloatTensor, update_transition_matrix: torch.FloatTensor = True) -> torch.FloatTensor:
        """Forward backward algorithm, corrsponding to the E-step of the EM algorithm.

        """

        init_p = torch.ones(self.n_states) / self.n_states
        firing_rates_in_all_states = self.firing_rates(convolved_spikes) # n_states x n_time_bins x n_neurons
        log_emission = self.distribution.log_likelihood(spikes, firing_rates_in_all_states).sum(dim=(-1,)).permute((1, 0)) # n_time_bins x n_states
        term_1 = torch.sum(gamma[0] + init_p.log())
        term_2 = torch.sum(torch.sum(xi, dim=0) * self.transition_matrix.log())
        term_3 = torch.sum(gamma * log_emission)
        if update_transition_matrix is True:
            # xi = xi.clamp(1e-16)
            # self.transition_matrix.data = xi.sum(dim=0) / xi.sum(dim=(0, 2))[:, None]
            self.transition_matrix.data = F.softmax((xi.sum(dim=0) / xi.sum(dim=(0, 2))[:, None]).log().clamp(min=-8), dim=-1)
        return term_1 + term_2 + term_3
    
    def viterbi(self, spikes: torch.FloatTensor, convolved_spikes: torch.FloatTensor) -> torch.FloatTensor:
        """Viterbi algorith
        m to inference the most probable latent state sequence.

        """

        n_time_bins = len(spikes)
        omega = torch.zeros((n_time_bins, self.n_states))
        psi = torch.zeros((n_time_bins, self.n_states), dtype=torch.int64)

        with torch.no_grad():
            firing_rates_in_all_states = self.firing_rates(convolved_spikes) # n_states x n_time_bins x n_neurons
            log_emission = self.distribution.log_likelihood(spikes, firing_rates_in_all_states).sum(dim=(-1,)).permute((1, 0)) # n_time_bins x n_states

            init_p = torch.ones(self.n_states) / self.n_states
            omega[0] = init_p.log() + log_emission[0]

            for t in range(1, n_time_bins):
                temp_matrix = self.transition_matrix.log() + omega[t - 1][:, None]
                values, psi[t] = torch.max(temp_matrix, dim=0)
                omega[t] = log_emission[t] + values
            
            states_pred = torch.zeros(n_time_bins, dtype=torch.int64)
            states_pred[-1] = omega[-1].argmax()
            for t in range(n_time_bins - 1, 0, -1):
                states_pred[t-1] = psi[t, states_pred[t]]
            return states_pred


class NaiveHMMGLM(HMMGLM):
    def __init__(
            self,
            n_states: int,
            n_neurons: int,
            basis: torch.FloatTensor,
            distribution: distributions.Distribution = distributions.Poisson(),
            activation: nonlinearities.Nonlinearity = nonlinearities.Sigmoid()
    ) -> None:
        super().__init__(n_states, n_neurons, basis, distribution, activation)

        self._weight = nn.Parameter(torch.zeros((self.n_states, self.n_neurons, self.n_neurons)))
    
    @property
    def weight(self):
        return self._weight
    
    def empty_diagonal(self) -> None:
        with torch.no_grad():
            self._weight.data[:, torch.arange(self.n_neurons), torch.arange(self.n_neurons)] = 0

    def permute_states(self, true_to_learned: torch.LongTensor) -> None:
        with torch.no_grad():
            self._weight.data[:] = self._weight.data[true_to_learned]
            if len(self._bias.shape) > 1:
                self._bias.data[:] = self._bias.data[true_to_learned]
            self.transition_matrix.data = self.transition_matrix.data[true_to_learned, :][:, true_to_learned]


class GaussianHMMGLM(NaiveHMMGLM):
    def __init__(
            self,
            n_states: int,
            n_neurons: int,
            basis: torch.FloatTensor,
            distribution: distributions.Distribution = distributions.Poisson(),
            activation: nonlinearities.Nonlinearity = nonlinearities.Sigmoid()
    ) -> None:
        super().__init__(n_states, n_neurons, basis, distribution, activation)
        self.weight_prior_mean = nn.Parameter(torch.zeros((self.n_neurons, self.n_neurons)), requires_grad=False)
        self.weight_prior_std = nn.Parameter(torch.tensor(1/3), requires_grad=False)
    
    def empty_diagonal(self) -> None:
        with torch.no_grad():
            self._weight.data[:, torch.arange(self.n_neurons), torch.arange(self.n_neurons)] = 0
            self.weight_prior_mean.data[torch.arange(self.n_neurons), torch.arange(self.n_neurons)] = 0

    def sample_weight(self, rng: torch.Generator = None) -> None:
        with torch.no_grad():
            for state in range(self.n_states):
                self._weight.data[state] = torch.normal(self.weight_prior_mean, self.weight_prior_std, generator=rng)
    
    def update_weight_prior_mean(self) -> None:
        with torch.no_grad():
            self.weight_prior_mean.data = self.weight.mean(dim=0)
    
    def update_weight_prior_std(self) -> None:
        with torch.no_grad():
            empirical_std = self.weight.std(dim=0)
            if len(self.weight_prior_std.shape) == 0:
                self.weight_prior_std.data = empirical_std.mean()
            else:
                self.weight_prior_std.data = empirical_std
    
    def weight_prior_log_likelihood(self) -> torch.FloatTensor:
        return -F.gaussian_nll_loss(self.weight_prior_mean, self.weight, (self.weight_prior_std**2).expand(self.n_neurons, self.n_neurons), full=True, eps=1e-8, reduction='sum')


class OnehotHMMGLM(HMMGLM):
    def __init__(
            self,
            n_states: int,
            n_neurons: int,
            basis: torch.FloatTensor,
            distribution: distributions.Distribution = distributions.Poisson(),
            activation: nonlinearities.Nonlinearity = nonlinearities.Sigmoid(),
            strength_nonlinearity: nonlinearities.Nonlinearity = nonlinearities.Softplus(),
            tau: float = 0.2,
            weight_tau: float = 0.2,
    ) -> None:
        super().__init__(n_states, n_neurons, basis, distribution, activation)

        self.strength_nonlinearity = strength_nonlinearity
        self.tau = tau
        self.weight_tau = weight_tau

        self._adjacency = nn.Parameter(torch.zeros((self.n_states, self.n_neurons, self.n_neurons, 3))) # adjacency in log(it) space
        self._strength = nn.Parameter(torch.zeros((self.n_states, self.n_neurons, self.n_neurons))) # strength in log(it) space, default is a single strength for both excitatory and inhibitory adjacency
        self._adjacency_prior = nn.Parameter(torch.zeros((self.n_neurons, self.n_neurons, 3)))
        self.adjacency_type = 'softmax'

    def set_adjacency_type(self, adjacency_type: str) -> None:
        if adjacency_type not in ['softmax', 'gumbel_softmax', 'hard']:
            raise ValueError()
        self.adjacency_type = adjacency_type
    
    @property
    def adjacency(self):
        if self.adjacency_type == 'softmax':
            return F.softmax(self._adjacency, dim=-1)
        elif self.adjacency_type == 'gumbel_softmax':
            return F.gumbel_softmax(self._adjacency, tau=self.weight_tau, dim=-1)
        elif self.adjacency_type == 'hard':
            return F.one_hot(self.adjacency_index + 1, num_classes=3)
        else:
            raise ValueError()
    
    @property
    def adjacency_index(self):
        with torch.no_grad():
            return self._adjacency.argmax(dim=-1) - 1
    
    @property
    def strength(self):
        return self.strength_nonlinearity(self._strength)[:, :, :, None].expand((-1, -1, -1, 2))

    @property
    def weight(self):
        adjacency = self.adjacency
        strength = self.strength
        
        excitatory_matrix = 1 * adjacency[:, :, :, 2] * strength[:, :, :, 1]
        inhibitory_matrix = (-1) * adjacency[:, :, :, 0] * strength[:, :, :, 0]
        return excitatory_matrix + inhibitory_matrix

    @property
    def adjacency_prior(self):
        return F.softmax(self._adjacency_prior, dim=-1)
    
    def empty_diagonal(self) -> None:
        with torch.no_grad():
            self._strength.data[:, torch.arange(self.n_neurons), torch.arange(self.n_neurons)] = -10
            self._adjacency.data[:, torch.arange(self.n_neurons), torch.arange(self.n_neurons), :] = torch.tensor([0, 5., 0])
            self._adjacency_prior.data[torch.arange(self.n_neurons), torch.arange(self.n_neurons), :] = torch.tensor([0, 5., 0])

    def permute_states(self, true_to_learned: torch.LongTensor) -> None:
        with torch.no_grad():
            self._adjacency.data[:] = self._adjacency.data[true_to_learned]
            self._strength.data[:] = self._strength.data[true_to_learned]
            if len(self._bias.shape) > 1:
                self._bias.data[:] = self._bias.data[true_to_learned]
            self.transition_matrix.data = self.transition_matrix.data[true_to_learned, :][:, true_to_learned]
    
    def correct_extreme_onehot(self) -> None:
        with torch.no_grad():
            loc = (F.softmax(self._adjacency.data, dim=-1).log() < -20).sum(dim=-1) > 0
            self._adjacency.data[loc] /= 2

            loc = (F.softmax(self._adjacency_prior.data, dim=-1).log() < -20).sum(dim=-1) > 0
            self._adjacency_prior.data[loc] /= 2
    
    def sample_adjacency(self, rng: torch.Generator = None) -> None:
        with torch.no_grad():
            for state in range(self.n_states):
                self._adjacency.data[state] = F.gumbel_softmax(self._adjacency_prior, self.tau, dim=-1).log()
    
    def adjacency_prior_log_likelihood(self) -> torch.FloatTensor:
        adjacency = F.softmax(self._adjacency, dim=-1)
        adjacency_prior = self.adjacency_prior

        return distributions.gumbel_softmax_log_likelihood(adjacency, adjacency_prior.expand(self.n_states, self.n_neurons, self.n_neurons, 3), self.tau).sum()
    
    def sampled_adjacency_entropy(self):
        adjacency = F.softmax(self._adjacency, dim=-1)
        return -(adjacency * adjacency.log()).sum()
