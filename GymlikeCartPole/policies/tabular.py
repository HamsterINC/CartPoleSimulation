from .base import BasePolicy
import numpy as np

class TabularPolicy(BasePolicy):
    def __init__(
        self,
        action_dim,
        state_dim=5,
        n_bins=(6, 6, 12, 12, 12),
        lr=0.1,
        gamma=0.99,
        eps=1.0,
        eps_min=0.05,
        eps_decay=0.996,
    ):
        self.n_actions = action_dim
        self.n_bins = n_bins
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        self.q = np.zeros((*n_bins, action_dim))

        self.low = np.array([-2.4, -5.0, -1.0, -1.0, -5.0])
        self.high = np.array([2.4, 5.0, 1.0, 1.0, 5.0])

    def _discretize(self, obs):
        ratios = (obs - self.low) / (self.high - self.low)
        ratios = np.clip(ratios, 0, 0.999)
        return tuple((ratios * self.n_bins).astype(int))

    def act(self, state):
        s = self._discretize(state)
        if np.random.rand() < self.eps:
            return np.random.randint(self.n_actions), {}
        return np.argmax(self.q[s]), {}

    def end_episode(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def save(self, path):
        np.save(path, self.q)

    def load(self, path):
        self.q = np.load(f"{path}.npy")


class QLearningPolicy(TabularPolicy):
    def observe(self, state, action, reward, next_state, done, info=None):
        s = self._discretize(state)
        s_next = self._discretize(next_state)

        target = reward
        if not done:
            target += self.gamma * np.max(self.q[s_next])

        self.q[s + (action,)] += self.lr * (
            target - self.q[s + (action,)]
        )


class SARSAPolicy(TabularPolicy):
    def observe(self, state, action, reward, next_state, done, info=None):
        next_action, _ = self.act(next_state)  # get next action for SARSA
        s = self._discretize(state)
        s_next = self._discretize(next_state)

        target = reward
        if not done:
            target += self.gamma * self.q[s_next + (next_action,)]

        self.q[s + (action,)] += self.lr * (
            target - self.q[s + (action,)]
        )