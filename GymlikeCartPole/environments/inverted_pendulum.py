import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FurutaEnv(gym.Env):
    """
    Minimal PPO-friendly Furuta pendulum (no curriculum).
    Stable swing-up + balancing via shaped reward.
    """

    def __init__(self):
        super().__init__()

        self.dt = 0.005

        # physical params
        self.m = 0.15
        self.l = 0.5
        self.g = 9.81
        self.Ip = 0.01
        self.Ia = 0.02

        # action space
        self.action_space = spaces.Box(-2.0, 2.0, (1,), dtype=np.float32)

        # observation (PPO-friendly)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # weights (important)
        self.wE = 1.0
        self.wA = 2.0
        self.wV = 0.1
        self.wT = 0.01

        self.state = None

    # -----------------------
    # helpers
    # -----------------------

    def wrap(self, x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    def obs(self):
        theta, theta_dot, alpha, alpha_dot = self.state

        return np.array([
            np.sin(theta),
            np.cos(theta),
            theta_dot / 10.0,
            np.sin(alpha),
            np.cos(alpha),
            alpha_dot / 10.0
        ], dtype=np.float32)

    # -----------------------
    # energy
    # -----------------------

    def energy(self, alpha, alpha_dot):
        return (
            0.5 * self.Ip * alpha_dot**2
            + self.m * self.g * self.l * (1 - np.cos(alpha))
        )

    # -----------------------
    # dynamics (stable simplified coupling)
    # -----------------------

    def step_dynamics(self, theta, theta_dot, alpha, alpha_dot, torque):

        s = np.sin(alpha)
        c = np.cos(alpha)

        alpha_ddot = (
            (self.g / self.l) * s
            - 0.05 * alpha_dot
            - 0.3 * theta_dot**2 * s * c
        )

        theta_ddot = (
            torque
            - 0.05 * theta_dot
            - 0.2 * alpha_ddot * c
        ) / self.Ia

        return theta_ddot, alpha_ddot

    # -----------------------
    # reset
    # -----------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = np.array([
            0.0,
            0.0,
            np.pi + np.random.uniform(-0.1, 0.1),
            0.0
        ], dtype=np.float32)

        return self.obs(), {}

    # -----------------------
    # step
    # -----------------------

    def step(self, action):
        torque = np.clip(action[0], -2.0, 2.0)

        theta, theta_dot, alpha, alpha_dot = self.state

        theta_ddot, alpha_ddot = self.step_dynamics(
            theta, theta_dot, alpha, alpha_dot, torque
        )

        # integrate
        theta_dot += self.dt * theta_ddot
        theta += self.dt * theta_dot

        alpha_dot += self.dt * alpha_ddot
        alpha += self.dt * alpha_dot

        alpha = self.wrap(alpha)

        self.state = np.array([theta, theta_dot, alpha, alpha_dot])

        # -----------------------
        # reward (key part)
        # -----------------------

        E = self.energy(alpha, alpha_dot)
        E_target = 2 * self.m * self.g * self.l

        energy_error = (E - E_target) ** 2

        angle_weight = np.exp(-10 * abs(alpha - np.pi))  # activates near upright
        angle_error = (alpha - np.pi) ** 2

        reward = (
            - energy_error
            - angle_weight * angle_error
            - 0.01 * torque**2
        )
        reward = reward / (self.m * self.g * self.l)  # normalize for easier learning

        # termination
        terminated = abs(alpha - np.pi) > 1.5
        truncated = False

        return self.obs(), reward, terminated, truncated, {}
    def render(self):
        theta, _, alpha, _ = self.state
        if self.render_mode == "human":
            print(f"theta={theta:.2f}, alpha={alpha:.2f}")