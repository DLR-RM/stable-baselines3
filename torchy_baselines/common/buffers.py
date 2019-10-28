import numpy as np
import torch as th


class BaseBuffer(object):
    def __init__(self, buffer_size, obs_dim, action_dim, device='cpu', n_envs=1):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(tensor):
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param tensor: (th.Tensor)
        :return: (th.Tensor)
        """
        shape = tensor.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return tensor.transpose(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self):
        if self.full:
            return self.buffer_size
        return self.pos

    def get_pos(self):
        return self.pos

    def add(self, *args, **kwargs):
        raise NotImplementedError()

    def reset(self):
        self.pos = 0
        self.full = False

    def sample(self, batch_size):
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = th.LongTensor(
            np.random.randint(0, upper_bound, size=batch_size))
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds):
        raise NotImplementedError()


class ReplayBuffer(BaseBuffer):
    """
    Taken from https://github.com/apourchot/CEM-RL
    """

    def __init__(self, buffer_size, obs_dim, action_dim, device='cpu', n_envs=1):
        super(ReplayBuffer, self).__init__(buffer_size, obs_dim, action_dim, device, n_envs=n_envs)

        assert n_envs == 1
        self.observations = th.zeros(self.buffer_size, self.n_envs, self.obs_dim)
        self.actions = th.zeros(self.buffer_size, self.n_envs, self.action_dim)
        self.next_observations = th.zeros(self.buffer_size, self.n_envs, self.obs_dim)
        self.rewards = th.zeros(self.buffer_size, self.n_envs)
        self.dones = th.zeros(self.buffer_size, self.n_envs)

    def add(self, obs, next_obs, action, reward, done):
        # Copy to avoid modification by reference
        self.observations[self.pos] = th.FloatTensor(np.array(obs).copy())
        self.next_observations[self.pos] = th.FloatTensor(np.array(next_obs).copy())
        self.actions[self.pos] = th.FloatTensor(np.array(action).copy())
        self.rewards[self.pos] = th.FloatTensor(np.array(reward).copy())
        self.dones[self.pos] = th.FloatTensor(np.array(done).copy())

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(self, batch_inds):
        return (self.observations[batch_inds, 0, :].to(self.device),
                self.actions[batch_inds, 0, :].to(self.device),
                self.next_observations[batch_inds, 0, :].to(self.device),
                self.dones[batch_inds].to(self.device),
                self.rewards[batch_inds].to(self.device))


class RolloutBuffer(BaseBuffer):
    def __init__(self, buffer_size, obs_dim, action_dim, device='cpu',
                 gae_lambda=1, gamma=0.99, n_envs=1):
        super(RolloutBuffer, self).__init__(buffer_size, obs_dim, action_dim, device, n_envs=n_envs)
        # TODO: try the buffer on the gpu?
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.dones, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self):
        self.observations = th.zeros(self.buffer_size, self.n_envs, self.obs_dim)
        self.actions = th.zeros(self.buffer_size, self.n_envs, self.action_dim)
        self.rewards = th.zeros(self.buffer_size, self.n_envs)
        self.returns = th.zeros(self.buffer_size, self.n_envs)
        self.dones = th.zeros(self.buffer_size, self.n_envs)
        self.values = th.zeros(self.buffer_size, self.n_envs)
        self.log_probs = th.zeros(self.buffer_size, self.n_envs)
        self.advantages = th.zeros(self.buffer_size, self.n_envs)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def compute_returns_and_advantage(self, last_value, dones=False):
        """
        From PPO2
        """
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = th.FloatTensor(1.0 - dones)
                next_value = last_value.clone().cpu().flatten()
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def add(self, obs, action, reward, done, value, log_prob):
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        self.observations[self.pos] = th.FloatTensor(np.array(obs).copy())
        self.actions[self.pos] = th.FloatTensor(np.array(action).copy())
        self.rewards[self.pos] = th.FloatTensor(np.array(reward).copy())
        self.dones[self.pos] = th.FloatTensor(np.array(done).copy())
        self.values[self.pos] = th.FloatTensor(value.clone().cpu().flatten())
        self.log_probs[self.pos] = th.FloatTensor(log_prob.cpu().clone())
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size=None):
        assert self.full
        indices = th.randperm(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in ['observations', 'actions', 'values',
                           'log_probs', 'advantages', 'returns']:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds):
        return (self.observations[batch_inds].to(self.device),
                self.actions[batch_inds].to(self.device),
                self.values[batch_inds].flatten().to(self.device),
                self.log_probs[batch_inds].flatten().to(self.device),
                self.advantages[batch_inds].flatten().to(self.device),
                self.returns[batch_inds].flatten().to(self.device))
