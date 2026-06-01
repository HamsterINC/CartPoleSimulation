# ------------------------
# PPO Agent
# ------------------------

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .base import BasePolicy

# ------------------------
# Actor-Critic Network
# ------------------------
class ActorCriticContinuous(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()
        )
        # Log std (learned)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        mu = self.actor(x)
        log_std = torch.clamp(self.log_std, -20, 2) # Prevent extreme values
        sigma = torch.exp(log_std).expand_as(mu)  # match batch shape
        value = self.critic(x)
        return mu, sigma, value
    
class ActorCriticShared(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        
        # 1. Shared Backbone (The "Physics" Processor)
        # This is where 90% of your FPGA DSP/Logic resources will go.
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            # We use one wide hidden layer for better FPGA backprop stability

        )
        
        # 2. Actor Head
        # Takes 256 features -> Outputs the mean action
        self.actor_head = nn.Sequential(
            nn.Linear(hidden, hidden), # 256 
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()
        )
        
        # Log std is still independent (not part of the backbone)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # 3. Critic Head
        # Takes the SAME 256 features -> Outputs the state value (V)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden, hidden), # 256 
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        # Pass through backbone once
        features = self.backbone(x)
        
        # Branch out to heads
        action_mean = self.actor_head(features)
        log_std = torch.clamp(self.log_std, -20, 2)
        sigma = torch.exp(log_std).expand_as(action_mean)
        value = self.critic_head(features)
        
        return action_mean, sigma, value

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, lam=0.95,
                 clip_eps=0.2, epochs=10, batch_size=64, entropy_coef=0.01, horizon=1000, hidden_layer=256):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.horizon = horizon
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCriticShared(state_dim, action_dim, hidden_layer).to(self.device)
        self.optimizer = optim.Adam(
                    self.model.parameters(), 
                    lr=float(lr), 
                    # eps=1e-8,  # Increased from 1e-8 to prevent FP16 division-by-zero / NaN issues
                    # weight_decay=1e-4
                )

        self.memory = []
        self.steps_count = 0

    def act(self, state, train=False):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mu, sigma, value = self.model(state_t)
        
        dist = torch.distributions.Normal(mu, sigma)
        
        t = float(train)
        action =  t * dist.sample() + (1 - t) * mu # Tensorized if statement to speed up training loop

        log_prob = dist.log_prob(action)

        return action.cpu().numpy().flatten(), {
            "log_prob": log_prob.cpu().numpy().flatten(), 
            "value": value.item()
        }

    def observe(self, state, action, reward, next_state, done, info):
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'done': done,
            'log_prob': info['log_prob'],
            'value': info['value'],
            'next_state': next_state # Kept for bootstrapping
        })
        self.steps_count += 1

        # Trigger update based on total steps across all episodes
        if self.steps_count >= self.horizon:
            self.update()
            self.memory = []
            self.steps_count = 0

    def compute_gae(self, rewards, values, dones, last_value):
        advantages = []
        gae = 0
        # Bootstrap with the last value if the episode didn't end
        next_value = last_value 
        
        for t in reversed(range(len(rewards))):
            # delta = r + gamma * V(s') * (1-done) - V(s)
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages.insert(0, gae)
            next_value = values[t]
        return advantages

    def update(self):
        # Extract data from memory
        states = torch.tensor([m['state'] for m in self.memory], dtype=torch.float32, device=self.device)
        actions = torch.tensor([m['action'] for m in self.memory], dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor([m['log_prob'] for m in self.memory], dtype=torch.float32, device=self.device)
        rewards = [m['reward'] for m in self.memory]
        values = [m['value'] for m in self.memory]
        dones = [m['done'] for m in self.memory]

        # Bootstrap: Get value of the very last state to "finish" the GAE calculation
        last_state = torch.tensor(self.memory[-1]['next_state'], dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, _, last_value = self.model(last_state)
            last_value = last_value.item() if not dones[-1] else 0

        # Compute Advantages and Returns
        advs = self.compute_gae(rewards, values, dones, last_value)
        advantages = torch.tensor(advs, dtype=torch.float32, device=self.device).unsqueeze(-1)
        returns = advantages + torch.tensor(values, dtype=torch.float32, device=self.device).unsqueeze(-1)
        
        # Standardize advantages to stabilize training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Epochs
        dataset_size = len(self.memory)
        indices = np.arange(dataset_size)
        
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for i in range(0, dataset_size, self.batch_size):
                idx = indices[i:i + self.batch_size]
                
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_returns = returns[idx]
                batch_advs = advantages[idx]

                mu, sigma, values_new = self.model(batch_states)
                dist = torch.distributions.Normal(mu, sigma)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO Clip Loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advs
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advs
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(values_new, batch_returns)
                
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
    
    def save(self, path):
        torch.save(self.model.state_dict(), f"{path}.pt")
    def load(self, path):
        self.model.load_state_dict(torch.load(f"{path}.pt", map_location=self.device))

