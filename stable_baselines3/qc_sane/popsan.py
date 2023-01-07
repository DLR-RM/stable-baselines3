import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

sys.path.append("../../")
from stable_baselines3.qc_sane import core_cuda as core

"""
Parameters for SNN
"""


ENCODER_REGULAR_VTH = 0.999
NEURON_VTH = 0.5
NEURON_CDECAY = 1 / 2
NEURON_VDECAY = 3 / 4
SPIKE_PSEUDO_GRAD_WINDOW = 0.5


class PseudoEncoderSpikeRegular(torch.autograd.Function):
    """Pseudo-gradient function for spike - Regular Spike for encoder"""

    @staticmethod
    def forward(ctx, input):
        return input.gt(ENCODER_REGULAR_VTH).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class PopSpikeEncoderRegularSpike(nn.Module):
    """Learnable Population Coding Spike Encoder with Regular Spike Trains"""

    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        """
        :param obs_dim: observation dimension
        :param pop_dim: population dimension
        :param spike_ts: spike timesteps
        :param mean_range: mean range
        :param std: std
        :param device: device
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.pop_dim = pop_dim
        self.encoder_neuron_num = obs_dim * pop_dim
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike = PseudoEncoderSpikeRegular.apply
        # Compute evenly distributed mean and variance
        tmp_mean = torch.zeros(1, obs_dim, pop_dim)
        delta_mean = (mean_range[1] - mean_range[0]) / (pop_dim - 1)
        for num in range(pop_dim):
            tmp_mean[0, :, num] = mean_range[0] + delta_mean * num
        tmp_std = torch.zeros(1, obs_dim, pop_dim) + std
        self.mean = nn.Parameter(tmp_mean)
        self.std = nn.Parameter(tmp_std)

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: pop_spikes
        """
        obs = obs.view(-1, self.obs_dim, 1)
        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(
            -(1.0 / 2.0) * (obs - self.mean).pow(2) / self.std.pow(2)
        ).view(-1, self.encoder_neuron_num)
        pop_volt = torch.zeros(batch_size, self.encoder_neuron_num, device=self.device)
        pop_spikes = torch.zeros(
            batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device
        )
        # Generate Regular Spike Trains
        for step in range(self.spike_ts):
            pop_volt = pop_volt + pop_act
            pop_spikes[:, :, step] = self.pseudo_spike(pop_volt)
            pop_volt = pop_volt - pop_spikes[:, :, step] * ENCODER_REGULAR_VTH
        return pop_spikes


class PopSpikeDecoder(nn.Module):
    """Population Coding Spike Decoder"""

    def __init__(self, act_dim, pop_dim, output_activation=nn.Tanh):
        """
        :param act_dim: action dimension
        :param pop_dim:  population dimension
        :param output_activation: activation function added on output
        """
        super().__init__()
        self.act_dim = act_dim
        self.pop_dim = pop_dim
        self.decoder = nn.Conv1d(act_dim, act_dim, pop_dim, groups=act_dim)
        self.output_activation = output_activation()

    def forward(self, pop_act):
        """
        :param pop_act: output population activity
        :return: raw_act
        """
        pop_act = pop_act.view(-1, self.act_dim, self.pop_dim)
        raw_act = self.output_activation(self.decoder(pop_act).view(-1, self.act_dim))
        return raw_act


class PseudoSpikeRect(torch.autograd.Function):
    """Pseudo-gradient function for spike - Derivative of Rect Function"""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = abs(input - NEURON_VTH) < SPIKE_PSEUDO_GRAD_WINDOW
        return grad_input * spike_pseudo_grad.float()


class SpikeMLP(nn.Module):
    """Spike MLP with Input and Output population neurons"""

    def __init__(self, in_pop_dim, out_pop_dim, hidden_sizes, spike_ts, device):
        """
        :param in_pop_dim: input population dimension
        :param out_pop_dim: output population dimension
        :param hidden_sizes: list of hidden layer sizes
        :param spike_ts: spike timesteps
        :param device: device
        """
        super().__init__()
        self.in_pop_dim = in_pop_dim
        self.out_pop_dim = out_pop_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_num = len(hidden_sizes)
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike = PseudoSpikeRect.apply
        # Define Layers (Hidden Layers + Output Population)
        self.hidden_layers = nn.ModuleList([nn.Linear(in_pop_dim, hidden_sizes[0])])
        if self.hidden_num > 1:
            for layer in range(1, self.hidden_num):
                self.hidden_layers.extend(
                    [nn.Linear(hidden_sizes[layer - 1], hidden_sizes[layer])]
                )
        self.out_pop_layer = nn.Linear(hidden_sizes[-1], out_pop_dim)

    def neuron_model(self, syn_func, pre_layer_output, current, volt, spike):
        """
        LIF Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param spike: spike of last step
        :return: current, volt, spike
        """
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        volt = volt * NEURON_VDECAY * (1.0 - spike) + current
        spike = self.pseudo_spike(volt)
        return current, volt, spike

    def forward(self, in_pop_spikes, batch_size):
        """
        :param in_pop_spikes: input population spikes
        :param batch_size: batch size
        :return: out_pop_act
        """

        # Define LIF Neuron states: Current, Voltage, and Spike
        hidden_states = []
        for layer in range(self.hidden_num):
            hidden_states.append(
                [
                    torch.zeros(
                        batch_size, self.hidden_sizes[layer], device=self.device
                    )
                    for _ in range(3)
                ]
            )
        out_pop_states = [
            torch.zeros(batch_size, self.out_pop_dim, device=self.device)
            for _ in range(3)
        ]
        out_pop_act = torch.zeros(batch_size, self.out_pop_dim, device=self.device)
        # Start Spike Timestep Iteration
        for step in range(self.spike_ts):
            in_pop_spike_t = in_pop_spikes[:, :, step]
            (
                hidden_states[0][0],
                hidden_states[0][1],
                hidden_states[0][2],
            ) = self.neuron_model(
                self.hidden_layers[0],
                in_pop_spike_t,
                hidden_states[0][0],
                hidden_states[0][1],
                hidden_states[0][2],
            )
            if self.hidden_num > 1:
                for layer in range(1, self.hidden_num):
                    (
                        hidden_states[layer][0],
                        hidden_states[layer][1],
                        hidden_states[layer][2],
                    ) = self.neuron_model(
                        self.hidden_layers[layer],
                        hidden_states[layer - 1][2],
                        hidden_states[layer][0],
                        hidden_states[layer][1],
                        hidden_states[layer][2],
                    )
            out_pop_states[0], out_pop_states[1], out_pop_states[2] = self.neuron_model(
                self.out_pop_layer,
                hidden_states[-1][2],
                out_pop_states[0],
                out_pop_states[1],
                out_pop_states[2],
            )
            out_pop_act += out_pop_states[2]
        out_pop_act = out_pop_act / self.spike_ts
        return out_pop_act


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianPopSpikeActor(nn.Module):
    """Squashed Gaussian Stochastic Population Coding Spike Actor with Fix Encoder"""

    def __init__(
        self,
        obs_dim,
        act_dim,
        en_pop_dim,
        de_pop_dim,
        hidden_sizes,
        mean_range,
        std,
        spike_ts,
        act_limit,
        device,
    ):
        """
        :param obs_dim: observation dimension
        :param act_dim: action dimension
        :param en_pop_dim: encoder population dimension
        :param de_pop_dim: decoder population dimension
        :param hidden_sizes: list of hidden layer sizes
        :param mean_range: mean range for encoder
        :param std: std for encoder
        :param spike_ts: spike timesteps
        :param act_limit: action limit
        :param device: device
        """
        super().__init__()
        self.act_limit = act_limit
        self.encoder = PopSpikeEncoderRegularSpike(
            obs_dim, en_pop_dim, spike_ts, mean_range, std, device
        )
        self.snn = SpikeMLP(
            obs_dim * en_pop_dim, act_dim * de_pop_dim, hidden_sizes, spike_ts, device
        )

        self.decoder = PopSpikeDecoder(
            act_dim, de_pop_dim, output_activation=nn.Identity
        )
        # Use a complete separate deep MLP to predict log std
        self.log_std_network = core.mlp(
            [obs_dim] + list(hidden_sizes) + [act_dim], nn.SELU
        )
        print("#" * 5, "Actor SNN\n\nPoPSpike:\n", self.encoder)
        print("#" * 5, "SNN_o/p_spikeactivity\n", self.snn)
        print("#" * 5, "Actor_mean (mu)\n", self.decoder)
        print("#" * 5, "Actor_std (log_sigma)\n", self.log_std_network)

    def forward(self, obs, batch_size, deterministic=False, with_logprob=True):
        """
        :param obs: observation
        :param batch_size: batch size
        :param deterministic: If true use deterministic action
        :param with_logprob: if true return log prob
        :return: action scale with action limit
        """
        in_pop_spikes = self.encoder(obs, batch_size)
        # print("########",in_pop_spikes.shape)
        out_pop_activity = self.snn(in_pop_spikes, batch_size)
        mu = self.decoder(out_pop_activity)
        log_std = self.log_std_network(obs)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()
        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                axis=1
            )
        else:
            logp_pi = None
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi
