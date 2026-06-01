import gymnasium as gym
import numpy as np
from gymnasium import spaces

class CartPoleSwingUp(gym.Env):
    def __init__(self, discrete=False):
        super().__init__()
        # Actions: Force applied to cart
        self.discrete = discrete
        if self.discrete:
            self.action_space = spaces.Discrete(2)  # left or right
            self._get_force = self._get_force_discrete
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)  # Continious
            self._get_force = self._get_force_continuous
        
        # Observations: [x, x_dot, sin(theta), cos(theta), theta_dot]
        self.observation_space = spaces.Box(low=np.array([-2.4, -np.inf, -1.0, -1.0, -np.inf], dtype=np.float32), high=np.array([2.4, np.inf, 1.0, 1.0, np.inf], dtype=np.float32), dtype=np.float32)
        
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"
        
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4


        self.render_mode = "rgb_array"  # default to rgb_array for visualization

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state: np.ndarray | None = None
        self.render_fps = 30

        self.steps_beyond_terminated = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start pole hanging DOWN (theta = pi)
        self.state = np.array([0, 0, np.pi, 0]) 
        return self._get_obs(), {}

    # Define the specific logic in separate methods
    def _get_force_discrete(self, action):
        return self.force_mag if action == 1 else -self.force_mag

    def _get_force_continuous(self, action):
        return action * self.force_mag

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self._get_force(action)

        temp = (
            force + self.polemass_length * np.square(theta_dot) * np.sin(theta)
        ) / self.total_mass
        thetaacc = (self.gravity * np.sin(theta) - np.cos(theta) * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * np.square(np.cos(theta)) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * np.cos(theta) / self.total_mass
        
        # 3. Update state (Euler integration)

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        # (Similar updates for x and x_dot)
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        
        # --- Swing-up Reward Function ---
        # Reward is high when cos(theta) is 1 (upright) and velocity is low
        reward = (np.cos(theta) + 1) / 2  # Normalized 0 to 1
        reward -= 0.01 * (theta_dot**2)   # Penalty for excessive spinning
        reward -= 0.001 * force**2
        # No termination at 12 degrees anymore! 
        # Only terminate if cart goes off-screen
        terminated = bool(abs(x) > 2.4)
        
        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        x, x_dot, theta, theta_dot = self.state
        return np.array([x, x_dot, np.sin(theta), np.cos(theta), theta_dot], dtype=np.float32)
    
    def _get_n_actions(self):
        if self.discrete:
            return self.action_space.n
        else:
            return self.action_space.shape[0]
    
    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.render_fps)
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )