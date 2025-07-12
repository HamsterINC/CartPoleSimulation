import numpy as np
from CartPole.cartpole_equations import _cartpole_ode


class EKFCartPole:
    """
    Extended Kalman Filter for the cart-pole system.

    Estimates the full state vector [position, velocity, angle, angular_velocity]
    given noisy measurements of position and angle.

    Usage:
        ekf = EKFCartPole(dt, params, Q, R, P0)
        v_est, omega_est = ekf.step(position, angle, u)
        full_state = ekf.get_state()
        ekf.reset(x0, P0)
    """
    def __init__(self, dt, params, Q=None, R=None, P0=None):
        # Sampling interval
        self.dt = dt
        # Physical parameters dict must include: k, m_cart, m_pole, g, J_fric, M_fric, L
        self.params = params

        # Noise covariances (tune to your system)
        self.Q = Q if Q is not None else np.diag([1e-4]*4)
        self.R = R if R is not None else np.diag([1e-2, 1e-3])

        # Initialize state estimate and covariance
        self.x_est = np.zeros(4)
        self.P     = P0 if P0 is not None else np.eye(4) * 0.1

    def reset(self, x0=None, P0=None):
        """
        Reset filter to initial conditions.

        :param x0: optional 4-vector initial state; defaults to zeros
        :param P0: optional 4×4 initial covariance; defaults to eye*0.1
        """
        self.x_est = np.array(x0) if x0 is not None else np.zeros(4)
        self.P     = np.array(P0) if P0 is not None else np.eye(4) * 0.1

    def get_state(self):
        """Return current state estimate vector [x, x_dot, theta, theta_dot]."""
        return self.x_est.copy()

    def f_continuous(self, x, u):
        """
        Continuous-time dynamics: [x_dot, x_ddot, theta_dot, theta_ddot]
        Calls imported cartpole_ode which returns (angleDD, positionDD).
        """
        ca, sa = np.cos(x[2]), np.sin(x[2])
        angleDD, positionDD = _cartpole_ode(
            ca, sa, x[3], x[1], u,
            self.params['k'], self.params['m_cart'], self.params['m_pole'],
            self.params['g'], self.params['J_fric'], self.params['M_fric'], self.params['L']
        )
        return np.array([x[1], positionDD, x[3], angleDD])

    def F_jacobian(self, x, u, eps=1e-5):
        """
        Compute the discrete-time Jacobian of the state transition
        via central finite differences:
        ∂x_{k+1}/∂x_k ≈ I + (∂f/∂x) * dt
        """
        n = x.size
        F = np.zeros((n, n))
        # central difference for each state dimension
        for i in range(n):
            dx = np.zeros(n); dx[i] = eps
            f_plus  = self.f_continuous(x + dx, u)
            f_minus = self.f_continuous(x - dx, u)
            F[:, i] = (f_plus - f_minus) / (2 * eps)
        # Euler approximation for discrete-time
        return np.eye(n) + F * self.dt

    def H_jacobian(self):
        """Measurement Jacobian for y = [position, angle]."""
        return np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0]])

    def step(self, position, angle, u):
        """
        Perform one EKF predict-update cycle.

        :param position: measured cart position [m]
        :param angle: measured pole angle [rad]
        :param u: control force applied [N]
        :returns: (estimated velocity, estimated angular velocity)
        """
        # — Predict —
        F      = self.F_jacobian(self.x_est, u)
        x_pred = self.x_est + self.f_continuous(self.x_est, u) * self.dt
        P_pred = F @ self.P @ F.T + self.Q

        # — Update —
        H      = self.H_jacobian()
        y_pred = H @ x_pred
        y      = np.array([position, angle])
        S      = H @ P_pred @ H.T + self.R
        K      = P_pred @ H.T @ np.linalg.inv(S)

        self.x_est = x_pred + K @ (y - y_pred)
        self.P     = (np.eye(4) - K @ H) @ P_pred

        return self.x_est[1], self.x_est[3]
