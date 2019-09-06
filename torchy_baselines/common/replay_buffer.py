import numpy as np
import torch as th


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

    def sample(self, batch_size):

        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = th.LongTensor(
            np.random.randint(0, upper_bound, size=batch_size))

        return (self.states[batch_inds].to(self.device),
                self.actions[batch_inds].to(self.device),
                self.next_states[batch_inds].to(self.device),
                self.dones[batch_inds].to(self.device),
                self.rewards[batch_inds].to(self.device))
