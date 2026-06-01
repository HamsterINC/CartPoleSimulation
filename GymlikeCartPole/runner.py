def run_episode(env, policy, seed=None, train=True, render=False, max_steps=500, visualize=False):
    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    t = 0
    done = False
    frames = []

    while t < max_steps:
        # Standardize act() to return the action and any optional metadata (log_probs, etc.)
        # The runner just passes the 'info' back to the policy during the observation phase.
        action, info = policy.act(state, train)
        if visualize:
            frame = env.render()
            frames.append(frame) 
        
        

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if train:
            # The policy decides internally whether to learn now (TD) or store for later (PPO)
            policy.observe(state, action, reward, next_state, done, info)

        if done:
            state, _ = env.reset()
        else:
            state = next_state

        total_reward += reward

        if render:
            env.render()
        t += 1

    return total_reward, frames 
