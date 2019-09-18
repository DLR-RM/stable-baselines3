import numpy as np
import torch as th

from torchy_baselines.common.utils import  discount_cumsum


class ReplayBuffer(object):
    """
    Taken from https://github.com/apourchot/CEM-RL
    """
    def __init__(self, buffer_size, state_dim, action_dim, device='cpu'):
        super(ReplayBuffer, self).__init__()
        # params
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pos = 0
        self.full = False
        self.device = device

        self.states = th.zeros(self.buffer_size, self.state_dim)
        self.actions = th.zeros(self.buffer_size, self.action_dim)
        self.next_states = th.zeros(self.buffer_size, self.state_dim)
        self.rewards = th.zeros(self.buffer_size, 1)
        self.dones = th.zeros(self.buffer_size, 1)

    def size(self):
        if self.full:
            return self.buffer_size
        return self.pos

    def get_pos(self):
        return self.pos

    def add(self, state, next_state, action, reward, done):

        self.states[self.pos] = th.FloatTensor(state)
        self.next_states[self.pos] = th.FloatTensor(next_state)
        self.actions[self.pos] = th.FloatTensor(action)
        self.rewards[self.pos] = th.FloatTensor([reward])
        self.dones[self.pos] = th.FloatTensor([done])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def reset(self):
        self.pos = 0
        self.full = False

    def sample(self, batch_size):

        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = th.LongTensor(
            np.random.randint(0, upper_bound, size=batch_size))
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds):
        return (self.states[batch_inds].to(self.device),
                self.actions[batch_inds].to(self.device),
                self.next_states[batch_inds].to(self.device),
                self.dones[batch_inds].to(self.device),
                self.rewards[batch_inds].to(self.device))


class RolloutBuffer(ReplayBuffer):
    def __init__(self, buffer_size, state_dim, action_dim, device='cpu',
                lambda_=1, gamma=0.99):
        super(RolloutBuffer, self).__init__(buffer_size, state_dim, action_dim, device)

        self.lambda_ = lambda_
        self.gamma = gamma
        # TODO: add n_envs
        self.returns = th.zeros(self.buffer_size, 1)
        self.values = th.zeros(self.buffer_size, 1)
        self.log_probs = th.zeros(self.buffer_size, 1)
        self.advantages = th.zeros(self.buffer_size, 1)
        self.path_start_idx = 0

    def finish_path(self, last_value=0):
        """
        From https://github.com/openai/spinningup/blob/master/spinup/algos/ppo/ppo.py
        """
        if self.full:
            self.pos = self.buffer_size
        path_slice = slice(self.path_start_idx, self.pos)
        rewards = np.append(self.rewards[path_slice].detach().cpu().numpy(), last_value)
        values = np.append(self.values[path_slice].detach().cpu().numpy(), last_value)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantages[path_slice, 0] = th.FloatTensor(discount_cumsum(deltas, self.gamma * self.lambda_).copy())
        # the next line computes rewards-to-go, to be targets for the value function
        self.returns[path_slice, 0] = th.FloatTensor(discount_cumsum(rewards, self.gamma)[:-1].copy())

        self.path_start_idx = self.pos

    def add(self, state, next_state, action, reward, done, value, log_prob):
        self.values[self.pos] = th.FloatTensor([value])
        self.log_probs[self.pos] = th.FloatTensor([log_prob])
        super(RolloutBuffer, self).add(state, next_state, action, reward, done)

    def reset(self):
        self.path_start_idx = 0
        super(RolloutBuffer, self).reset()

    def _get_samples(self, batch_inds):
        return (self.states[batch_inds].to(self.device),
                self.actions[batch_inds].to(self.device),
                self.next_states[batch_inds].to(self.device),
                self.dones[batch_inds].to(self.device),
                self.rewards[batch_inds].to(self.device),
                self.values[batch_inds].to(self.device),
                self.log_probs[batch_inds].to(self.device),
                self.advantages[batch_inds].to(self.device),
                self.returns[batch_inds].to(self.device))
