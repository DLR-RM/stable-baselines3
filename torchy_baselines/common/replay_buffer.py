import numpy as np
import torch as th

# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Expects tuples of (state, next_state, action, reward, done)
# class ReplayBuffer(object):
# 	def __init__(self, max_size=1e6):
# 		self.storage = []
# 		self.max_size = max_size
# 		self.ptr = 0
#
# 	def add(self, data):
# 		if len(self.storage) == self.max_size:
# 			self.storage[int(self.ptr)] = data
# 			self.ptr = (self.ptr + 1) % self.max_size
# 		else:
# 			self.storage.append(data)
#
# 	def sample(self, batch_size):
# 		ind = np.random.randint(0, len(self.storage), size=batch_size)
# 		x, y, u, r, d = [], [], [], [], []
#
# 		for i in ind:
# 			X, Y, U, R, D = self.storage[i]
# 			x.append(np.array(X, copy=False))
# 			y.append(np.array(Y, copy=False))
# 			u.append(np.array(U, copy=False))
# 			r.append(np.array(R, copy=False))
# 			d.append(np.array(D, copy=False))
#
# 		return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class ReplayBuffer(object):

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
