import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from .base import BasePolicy


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNPolicy(BasePolicy):
    def __init__(
        self,
        action_dim,
        state_dim=5,
        lr=1e-3,
        gamma=0.99,
        eps_start=1.0,
        eps_min=0.05,
        eps_decay=0.996,
        buffer_size=10000,
        batch_size=64,
        target_update=10,
        device=None
    ):
        self.n_actions = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device or ("cuda:5" if torch.cuda.is_available() else "cpu")

        # Q networks
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.SGD(self.q_net.parameters(), lr=float(lr))

        # ε-greedy
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        # Replay buffer
        self.buffer = deque(maxlen=buffer_size)
        self.steps = 0

    def act(self, state):
        if np.random.rand() < self.eps:
            return np.random.randint(self.n_actions), {}
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return int(q_values.argmax().item()), {}

    def observe(self, state, action, reward, next_state, done, info=None):
        # store in replay buffer
        self.buffer.append((state, action, reward, next_state, done))
        self.steps += 1

        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q(s,a)
        q_values = self.q_net(states).gather(1, actions)

        # target Q
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target = rewards + self.gamma * q_next * (1 - dones)

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def end_episode(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def save(self, path):
        torch.save(self.q_net.state_dict(), f"{path}.pt")

    def load(self, path):
        self.q_net.load_state_dict(torch.load(f"{path}.pt"))
        self.target_net.load_state_dict(self.q_net.state_dict())