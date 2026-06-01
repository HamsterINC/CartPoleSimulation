import gymnasium as gym
import numpy as np
import yaml
import torch
from GymlikeCartPole.policies import QLearningPolicy, SARSAPolicy, DQNPolicy, OnlineActorCritic, PPO
from GymlikeCartPole.runner import run_episode
from GymlikeCartPole.environments import CartPoleSwingUp, CartPoleEnv, FurutaEnv

# Map strings in YAML to Class objects
POLICY_MAP = {
    # "Q-learning": QLearningPolicy,
    # "SARSA": SARSAPolicy,
    # "DQN": DQNPolicy,
    # "OnlineActorCritic": OnlineActorCritic,
    "PPO": PPO
}

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_experiment():
    config = load_config()
    exp_cfg = config['experiment']
    
    best_policy = None
    best_score = -float("inf")

    # Iterate through policies defined in YAML
    for name, params in config['policies'].items():
        if name not in POLICY_MAP:
            continue
            
        print(f"\n=== Training {name} ===")
        
        # Initialize environment based on policy requirements
        match exp_cfg['env']:
            case "CartPole-v1":
                gym.make("CartPole-v1", render_mode="rgb_array")
                env = gym.make("CartPole-v1", render_mode="rgb_array")
            case "DemoCartpoleSwingup":
                env = CartPoleEnv(swingup=True)
                env.render_mode = "rgb_array"
            case "DemoCartpoleBalance":
                env = CartPoleEnv(swingup=False)
                env.render_mode = "rgb_array"
            case "CartpoleSwingup":
                env = CartPoleSwingUp(name['is_discrete'])
                env.render_mode = "rgb_array"
            case "inverted_pendulum":
                env = FurutaEnv()
                env.render_mode = "rgb_array"
        

        # Initialize Policy with kwargs from YAML
        policy = POLICY_MAP[name](
            action_dim=env.action_space.shape[0],
            state_dim=env.observation_space.shape[0],
            **params # Passes LR, Gamma, etc. directly
        )

        train_returns = []

        for ep in range(exp_cfg['n_train_episodes']):
            r, _ = run_episode(
                env, 
                policy, 
                seed=ep, 
                train=True, 
                render=exp_cfg['render_training'], 
                max_steps=exp_cfg['max_steps']
            )
            train_returns.append(r)

            if ep % 25 == 0:
                print(f"ep {ep:4d} | mean return {np.mean(train_returns[-25:]):6.1f}")
        
        eval_returns = [run_episode(env, policy, seed=exp_cfg['seed_offset'] + i, train=False)
            for i in range(exp_cfg['n_eval_episodes'])
        ]
        eval_returns = [r for r, _ in eval_returns]

        mean_eval = np.mean(eval_returns)
        print(f"Eval: {mean_eval:.1f} ± {np.std(eval_returns):.1f}")
        
        policy.save(f"Output/{name.lower()}_inverted_pendulum_policy")
        output_path = f"Output/{name.lower()}_inverted_pendulum_policy.txt"

        with open(output_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        
        env.close()

if __name__ == "__main__":
    run_experiment()