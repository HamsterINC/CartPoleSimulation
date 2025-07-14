# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
from typing import Tuple
from CartPole.cartpole_equations import _cartpole_ode


class EKFCartPole:
    """
    Extended Kalman Filter for the cart–pole.

    State:      x = [cart_pos, cart_vel, pole_ang, pole_ang_vel]
    Measurement y = [cart_pos, pole_ang]   (noisy)
    """

    # ──────────────────────────  Construction / reset  ──────────────────────────
    def __init__(self, dt: float, params: dict,
                 Q: np.ndarray | None = None,
                 R: np.ndarray | None = None,
                 P0: np.ndarray | None = None,
                 fd_eps: float = 1e-6):
        self.dt     = dt                       # sampling interval
        self.params = params                   # physical constants
        self.fd_eps = fd_eps                   # finite-difference step

        # Process- and measurement-noise covariances (reasonable defaults)
        self.Q = Q if Q is not None else np.diag([5e-5, 3e-3, 1e-5, 1e-3])
        self.R = R if R is not None else np.diag([1e-3, 5e-4])

        # Initial state / covariance
        self.x_est = np.zeros(4)
        self.P     = P0 if P0 is not None else np.eye(4) * 0.05

        # Constant measurement matrix (same for every step)
        self.H = self.H_jacobian()

    # --------------------------------------------------------------------------
    def reset(self, x0: np.ndarray | None = None,
                    P0: np.ndarray | None = None) -> None:
        """Re-initialise the filter."""
        self.x_est = np.array(x0, dtype=float) if x0 is not None else np.zeros(4)
        self.P     = np.array(P0, dtype=float) if P0 is not None else np.eye(4) * 0.05

    def get_state(self) -> np.ndarray:
        """Return a copy of the current state estimate."""
        return self.x_est.copy()

    # ───────────────────────────  System dynamics  ──────────────────────────────
    def f_continuous(self, x: np.ndarray, u: float) -> np.ndarray:
        """
        Continuous-time derivative  ẋ = f(x,u).

        `_cartpole_ode` expects cosθ, sinθ, θ̇, ẋ, u and returns (θ̈, ẍ).
        We convert that to [ẋ, ẍ, θ̇, θ̈].
        """
        ca, sa = np.cos(x[2]), np.sin(x[2])
        theta_dd, x_dd = _cartpole_ode(
            ca, sa, x[3], x[1], u,
            self.params['k'], self.params['m_cart'], self.params['m_pole'],
            self.params['g'], self.params['J_fric'], self.params['M_fric'],
            self.params['L'],
        )
        return np.array([x[1], x_dd, x[3], theta_dd])

    # ────────────────  Discrete prediction (Runge–Kutta 4)  ────────────────────
    def _rk4(self, x: np.ndarray, u: float) -> np.ndarray:
        """One RK4 step – far more accurate than Euler at identical Δt."""
        k1 = self.f_continuous(x,               u)
        k2 = self.f_continuous(x + 0.5*self.dt*k1, u)
        k3 = self.f_continuous(x + 0.5*self.dt*k2, u)
        k4 = self.f_continuous(x +       self.dt*k3, u)
        return x + (self.dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    # ─────────────────────────────  Jacobians  ──────────────────────────────────
    def F_jacobian(self, x: np.ndarray, u: float) -> np.ndarray:
        """
        Discrete-time Jacobian ∂xₖ₊₁/∂xₖ via centred finite differences.
        Analytical Jacobians are faster but far less maintainable.
        """
        n, F = x.size, np.zeros((4, 4))
        for i in range(n):
            dx = np.zeros(4); dx[i] = self.fd_eps
            F[:, i] = (self._rk4(x + dx, u) - self._rk4(x - dx, u)) / (2*self.fd_eps)
        return F

    @staticmethod
    def H_jacobian() -> np.ndarray:
        """Linear measurement model  y = H x  ."""
        return np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0]])

    # ─────────────────────────────  EKF step  ───────────────────────────────────
    def step(self, position_meas: float, angle_meas: float, u: float) -> Tuple[float, float]:
        """
        One EKF predict-and-update cycle.

        Returns
        -------
        cart velocity estimate, pole angular velocity estimate
        """
        # — Predict —
        F      = self.F_jacobian(self.x_est, u)
        x_pred = self._rk4(self.x_est, u)
        P_pred = F @ self.P @ F.T + self.Q

        # — Innovation —
        z_pred = self.H @ x_pred
        z_meas = np.array([position_meas, angle_meas])
        y      = z_meas - z_pred

        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)       # Kalman gain

        # — Correct —
        self.x_est = x_pred + K @ y
        self.x_est[2] = self._wrap_angle(float(self.x_est[2]))  # keep θ ∈ (−π,π]

        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T      # Joseph form
        return float(self.x_est[1]), float(self.x_est[3])

    # ──────────────────────────────  Helpers  ───────────────────────────────────
    @staticmethod
    def _wrap_angle(theta_rad: float) -> float:
        """Map any angle to (−π, π]."""
        return float((theta_rad + np.pi) % (2*np.pi) - np.pi)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Smallest signed difference a − b wrapped to (−π, π]."""
        return float((a - b + np.pi) % (2*np.pi) - np.pi)


# ───────────────────  Innovation-Based Adaptive Estimation  ───────────────────
class _IAEQRUpdater:
    """
    Online adjustment of (Q,R) so that the **empirical** innovation covariance
    matches the Kalman-predicted one.  See Maybeck vol. I §8.7.
    """

    def __init__(self, dim_z: int,
                 lambda_: float = 0.97,
                 gamma_R: float = 0.90,
                 gamma_Q: float = 0.98,
                 beta: float = 1.0):
        self.lmb   = lambda_
        self.gR    = gamma_R
        self.gQ    = gamma_Q
        self.beta  = beta
        self.S_hat = np.zeros((dim_z, dim_z))          # EWMA of yyᵀ

    # -------------------------------------------------------------------------
    def update(self, y: np.ndarray, S_pred: np.ndarray, K: np.ndarray,
               R_prev: np.ndarray, Q_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return new (Q, R) given innovation vector y and predicted S.
        """
        # 1) EWMA of empirical innovation covariance
        self.S_hat = self.lmb * self.S_hat + (1 - self.lmb) * np.outer(y, y)

        # 2) Update R towards the mismatch (empirical − predicted)
        R_target = _psd(self.S_hat - S_pred + R_prev)
        R_new    = self.gR * R_prev + (1 - self.gR) * R_target

        # 3) Update Q along 'surprising' directions
        Q_incr = self.beta * K @ np.outer(y, y) @ K.T
        Q_new  = self.gQ * Q_prev + (1 - self.gQ) * _psd(Q_incr)
        return Q_new, R_new


def _psd(M: np.ndarray) -> np.ndarray:
    """
    Project symmetric matrix onto the PSD cone (eigenvalue flooring).
    Keeps (Q,R) numerical soundness without heavy optimisation.
    """
    eigval, eigvec = np.linalg.eigh(0.5*(M + M.T))
    eigval[eigval < 1e-12] = 1e-12
    return (eigvec * eigval) @ eigvec.T


# ────────────────────────  Adaptive EKF wrapper  ──────────────────────────────
class AdaptiveEKFCartPole(EKFCartPole):
    """
    Drop-in variant that **optionally** performs IAE adaptation of (Q,R).

    Example
    -------
    ekf = AdaptiveEKFCartPole(dt, params)                       # plain EKF
    ekf_adapt = AdaptiveEKFCartPole(dt, params,
                                    iae_params={'lambda_': 0.96})
    """

    def __init__(self, *args,
                 iae_params = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._iae = None if iae_params is None else _IAEQRUpdater(dim_z=2, **iae_params)

    # ---------------------------------------------------------------------
    def step(self, position_meas: float, angle_meas: float, u: float) -> Tuple[float, float]:
        """
        Same interface as parent; if IAE active, (Q,R) are updated each cycle.
        """
        # — Predict —
        F      = self.F_jacobian(self.x_est, u)
        x_pred = self._rk4(self.x_est, u)
        P_pred = F @ self.P @ F.T + self.Q

        # — Innovation —
        z_pred = self.H @ x_pred
        y = np.array([position_meas - z_pred[0],
                      self._angle_diff(angle_meas, z_pred[1])])

        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # — Correct —
        self.x_est = x_pred + K @ y
        self.x_est[2] = self._wrap_angle(float(self.x_est[2]))

        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)      # symmetrise

        # — Optional IAE update —
        if self._iae is not None:
            self.Q, self.R = self._iae.update(y, S, K, self.R, self.Q)

        return float(self.x_est[1]), float(self.x_est[3])
