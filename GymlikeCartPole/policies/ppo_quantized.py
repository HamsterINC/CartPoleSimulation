import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ----------------------------------------------------
# Actor-Critic Network (Explicitly clamping for 16-bit)
# ----------------------------------------------------
class ActorCriticContinuous(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
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
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        mu = self.actor(x)
        # CRITICAL FOR 16-BIT: 
        # FP16 overflows at exp(11). We clamp tightly between -10 and 2.
        log_std = torch.clamp(self.log_std, -10.0, 2.0) 
        sigma = torch.exp(log_std).expand_as(mu)
        value = self.critic(x)
        return mu, sigma, value


class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, lam=0.95,
                 clip_eps=0.2, epochs=10, batch_size=64, entropy_coef=0.01, 
                 horizon=1000, hidden_layer=256, use_bf16=False):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.horizon = horizon
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine 16-bit dtype (BF16 is highly recommended if hardware supports it)
        self.dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_bf16_supported()) else torch.float16
        
        self.model = ActorCriticContinuous(state_dim, action_dim, hidden_layer).to(self.device)
        self.optimizer = optim.Adam(
                    self.model.parameters(), 
                    lr=float(lr), 
                    eps=1e-4,  # Increased from 1e-8 to prevent FP16 division-by-zero / NaN issues
                    weight_decay=1e-4
                )
        
        # Gradient Scaler is mandatory for FP16 to prevent underflowing gradients.
        # It is NOT needed (and does nothing) for BF16.
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.dtype == torch.float16))

        self.memory = []
        self.steps_count = 0

    def act(self, state, train=False):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Inference runs in 16-bit precision
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                mu, sigma, value = self.model(state_t)
        
        # Distributions must be kept in FP32 for numerical stability!
        dist = torch.distributions.Normal(mu.float(), sigma.float())
        
        t = float(train)
        action = t * dist.sample() + (1 - t) * mu.float()
        log_prob = dist.log_prob(action)

        return action.cpu().numpy().flatten(), {
            "log_prob": log_prob.cpu().numpy().flatten(), 
            "value": value.float().item()
        }

    def observe(self, state, action, reward, next_state, done, info):
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'done': done,
            'log_prob': info['log_prob'],
            'value': info['value'],
            'next_state': next_state
        })
        self.steps_count += 1

        if self.steps_count >= self.horizon:
            self.update()
            self.memory = []
            self.steps_count = 0

    def compute_gae(self, rewards, values, dones, last_value):
        advantages = []
        gae = 0
        next_value = last_value 
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages.insert(0, gae)
            next_value = values[t]
        return advantages

    def update(self):
        # We keep the buffer tensors in FP32 to calculate GAE and log probs reliably
        states = torch.tensor([m['state'] for m in self.memory], dtype=torch.float32, device=self.device)
        actions = torch.tensor([m['action'] for m in self.memory], dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor([m['log_prob'] for m in self.memory], dtype=torch.float32, device=self.device)
        rewards = [m['reward'] for m in self.memory]
        values = [m['value'] for m in self.memory]
        dones = [m['done'] for m in self.memory]

        last_state = torch.tensor(self.memory[-1]['next_state'], dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                _, _, last_value = self.model(last_state)
            last_value = last_value.float().item() if not dones[-1] else 0

        advs = self.compute_gae(rewards, values, dones, last_value)
        advantages = torch.tensor(advs, dtype=torch.float32, device=self.device).unsqueeze(-1)
        returns = advantages + torch.tensor(values, dtype=torch.float32, device=self.device).unsqueeze(-1)
        
        # Standardize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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

                # ----------------------------------------------------
                # 16-Bit Mixed Precision Forward Pass
                # ----------------------------------------------------
                with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                    mu, sigma, values_new = self.model(batch_states)
                    
                    # Convert to float32 before feeding into probability distribution!
                    dist = torch.distributions.Normal(mu.float(), sigma.float())
                    
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()

                    # Compute ratio & policy losses in FP32 to prevent division/exponent underflow
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advs
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advs
                    
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = nn.functional.mse_loss(values_new.float(), batch_returns)
                    
                    loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

                # ----------------------------------------------------
                # 16-Bit Backward Pass & Scaling
                # ----------------------------------------------------
                self.optimizer.zero_grad()
                
                # Scales loss, calls backward() on scaled loss to create scaled gradients
                self.scaler.scale(loss).backward()
                
                # Unscales gradients before clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                
                # Step optimizer and update scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()