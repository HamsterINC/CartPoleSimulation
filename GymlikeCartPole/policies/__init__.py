# Optional: expose key policies for easier imports
from .base import BasePolicy
from .tabular import QLearningPolicy, SARSAPolicy
from .continuous import DQNPolicy
from .AC import OnlineActorCritic
from .ppo import PPO
# future: from .continuous import DQNPolicy, LinearPolicy

__all__ = ["BasePolicy", "QLearningPolicy", "SARSAPolicy", "DQNPolicy", "OnlineActorCritic", "PPO"]