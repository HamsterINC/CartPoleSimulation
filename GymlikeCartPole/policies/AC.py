import torch
import torch.nn as nn
import torch.optim as optim
from .base import BasePolicy

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return torch.softmax(self.actor(x), dim=-1), self.critic(x)

class OnlineActorCritic(BasePolicy):
    def __init__(self, state_dim, action_dim, lr=5e-5, gamma=0.99):
        self.gamma = gamma
        self.eps = gamma
        self.device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(lr))

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        
        # Forward pass through your model
        # Assuming self.model returns (action_probs, state_value)
        probs, value = self.model(state)
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        # We pack the log_prob and value into the info dictionary.
        # We keep them as tensors so we can backpropagate through them in observe().
        info = {
            "log_prob": dist.log_prob(action),
            "value": value
        }
    
        return action.item(), info

    def observe(self, state, action, reward, next_state, done, info):
        # 1. Extract values already computed in act() from the info dict
        # These are still tensors attached to the computational graph
        log_prob = info["log_prob"]
        state_value = info["value"]

        # 2. Prepare tensors for the update
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)

        # 3. Get next state value V(s')
        with torch.no_grad():
            _, next_state_value = self.model(next_state)
            # TD Target: r + gamma * V(s')
            target = reward + (1 - int(done)) * self.gamma * next_state_value

        # 4. Calculate TD Error (Advantage)
        # Use .detach() on the target to ensure we only train the critic to 
        # hit the target, not the target to move towards the critic.
        advantage = target - state_value

        # 5. Losses
        # Note: entropy isn't in info, so we can re-calculate it or add it to act()
        # For online AC, we use the detached advantage for the actor loss
        actor_loss = -log_prob * advantage.detach()
        
        # Critic loss: minimize MSE between V(s) and TD Target
        critic_loss = torch.nn.functional.mse_loss(state_value, target)

        total_loss = actor_loss + 0.5 * critic_loss

        # 6. Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

    def end_episode(self): 
            return
    
    def save(self, path):
        torch.save(self.model.state_dict(), f"{path}.pt")


    def load(self, path):
        self.model.load_state_dict(torch.load(f"{path}.pt"))