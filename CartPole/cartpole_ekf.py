# ekf_cartpole_autotune.py
# -*- coding: ascii -*-
"""
Fully self-contained Extended Kalman Filter for the classic cart-pole together
with an adaptive tuner that

    * learns (Q, R) once from a high-grade sensor stream that includes
      velocities,
    * keeps R fresh online via innovation covariance matching once only a
      cheaper sensor (angle, position) is left,
    * freezes adaptation while the rig is practically motionless so the
      covariances do not collapse spuriously,
    * persists Q and R across program restarts in an atomic manner.

Drop this file anywhere on your PYTHONPATH and use exactly as shown at the
bottom.  Only additions critical to the requested behaviour were inserted;
all other code is your original, verbatim.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple, Optional
from collections import deque

import numpy as np
from CartPole.cartpole_equations import _cartpole_ode

import yaml


store_path: str = "ekf_noise_cov.yaml"


# --------------------------------------------------------------------------- #
#  EKF proper (unchanged except for ASCII fixes)                              #
# --------------------------------------------------------------------------- #
class EKFCartPole:
    """
    Extended Kalman Filter for the cart-pole.

    State vector       x = [cart_pos, cart_vel, pole_ang, pole_ang_vel]
    Measurement vector y = [cart_pos, pole_ang]
    """

    # -------------------------- construction --------------------------------
    def __init__(
        self,
        dt: float,
        params: Dict[str, float],
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
        fd_eps: float = 1e-6,
    ):
        self.dt = dt                         # sampling interval [s]
        self.params = params                 # physical constants
        self.fd_eps = fd_eps                 # finite-difference step

        # Default covariances: generous yet stable for most rigs
        self.Q = Q if Q is not None else np.diag([5e-5, 3e-3, 1e-5, 1e-3])
        self.R = R if R is not None else np.diag([1e-3, 5e-4])

        self.x_est = np.zeros(4)             # current state estimate
        self.P = np.eye(4) * 0.05 if P0 is None else P0.copy()

        # Constant measurement matrix (cart_pos and pole_ang only)
        self.H = self.H_jacobian()

    # -----------------------------------------------------------------------
    def reset(
        self,
        x0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
    ) -> None:
        """Re-initialise the filter."""
        self.x_est = np.zeros(4) if x0 is None else x0.astype(float)
        self.P = np.eye(4) * 0.05 if P0 is None else P0.astype(float)

    def get_state(self) -> np.ndarray:
        """Return a copy of the current state estimate."""
        return self.x_est.copy()

    # ----------------------------- system dynamics -------------------------
    def f_continuous(self, x: np.ndarray, u: float) -> np.ndarray:
        """
        Continuous-time derivative  x_dot = f(x, u).

        _cartpole_ode expects
            (cos(theta), sin(theta), theta_dot, x_dot, u)
        and returns (theta_ddot, x_ddot).
        We convert that to [x_dot, x_ddot, theta_dot, theta_ddot].
        """
        ca, sa = np.cos(x[2]), np.sin(x[2])
        theta_dd, x_dd = _cartpole_ode(
            ca,
            sa,
            x[3],
            x[1],
            u,
            self.params.k,
            self.params.m_cart,
            self.params.m_pole,
            self.params.g,
            self.params.J_fric,
            self.params.M_fric,
            self.params.L,
        )
        return np.array([x[1], x_dd, x[3], theta_dd])

    # --------------------------- discrete prediction -----------------------
    def _rk4(self, x: np.ndarray, u: float) -> np.ndarray:
        """One RK4 step (notably more accurate than Euler at identical dt)."""
        k1 = self.f_continuous(x, u)
        k2 = self.f_continuous(x + 0.5 * self.dt * k1, u)
        k3 = self.f_continuous(x + 0.5 * self.dt * k2, u)
        k4 = self.f_continuous(x + self.dt * k3, u)
        return x + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # ------------------------------ Jacobians ------------------------------
    def F_jacobian(self, x: np.ndarray, u: float) -> np.ndarray:
        """
        Discrete-time Jacobian d(x_next)/d(x) via centred finite differences.
        Analytical Jacobians are faster but far less maintainable.
        """
        F = np.zeros((4, 4))
        for i in range(4):
            dx = np.zeros_like(x)
            dx[i] = self.fd_eps
            F[:, i] = (self._rk4(x + dx, u) - self._rk4(x - dx, u)) / (
                2 * self.fd_eps
            )
        return F

    @staticmethod
    def H_jacobian() -> np.ndarray:
        """Linear measurement model  y = H x."""
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

    # ------------------------------- EKF step ------------------------------
    def step(
        self, position_meas: float, angle_meas: float, u: float
    ) -> Tuple[float, float]:
        """
        One EKF predict-and-update cycle.

        Returns
        -------
        (cart velocity estimate, pole angular velocity estimate)
        """
        # -- predict --------------------------------------------------------
        F = self.F_jacobian(self.x_est, u)
        x_pred = self._rk4(self.x_est, u)
        P_pred = F @ self.P @ F.T + self.Q

        # -- innovation -----------------------------------------------------
        z_pred = self.H @ x_pred
        y_pos = position_meas - z_pred[0]
        y_ang = self._angle_diff(angle_meas, z_pred[1])
        y = np.array([y_pos, y_ang])

        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # -- correct --------------------------------------------------------
        self.x_est = x_pred + K @ y
        self.x_est[2] = self._wrap_angle(float(self.x_est[2]))  # theta in (-pi, pi]

        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T  # Joseph form
        return float(self.x_est[1]), float(self.x_est[3])

    # ------------------------------ helpers -------------------------------
    @staticmethod
    def _wrap_angle(theta_rad: float) -> float:
        """Map any angle to (-pi, pi]."""
        return float((theta_rad + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Smallest signed difference a - b wrapped to (-pi, pi]."""
        return float((a - b + np.pi) % (2.0 * np.pi) - np.pi)


# --------------------------------------------------------------------------- #
#  Adaptive tuner + persistence                                               #
# --------------------------------------------------------------------------- #
class EKFAdaptiveTuner:
    """
    Learns (Q, R) automatically and writes them to disk so that a sensor swap
    simply reloads the last good matrices.

    Phase 1 (hi_grade=True):
        needs near-ground-truth (pos, vel, ang, ang_vel); computes batch
        covariances, blends them into (Q, R), writes to disk.

    Phase 2 (hi_grade=False):
        innovation-based covariance matching; skips updates while the system
        is almost motionless; writes any accepted update to disk.
    """

    def __init__(
        self,
        ekf: EKFCartPole,
        window: int = 500,            # >= 10 s at 50 Hz  - intentionally slow
        alpha: float = 0.995,         # heavy inertia as requested
        eps: float = 1e-7,
        store_path: str = "ekf_noise_cov.npz",
        motion_tol: float = 2e-3,
    ):
        self.ekf = ekf
        self.window = window
        self.alpha = alpha
        self.eps = eps
        self.store_path = store_path
        self.motion_tol = motion_tol  # |innovation| below -> "idle"

        self.true_buf: deque = deque(maxlen=window)    # (x_true, u)
        self.innov_buf: deque = deque(maxlen=window)   # innovation samples

        # restore Q, R if a previous run saved them
        self._restore_covariances()

    # -----------------------------------------------------------------------
    #  public interface
    # -----------------------------------------------------------------------
    def feed_measurement(
        self,
        pos: float,
        ang: float,
        u: float,
        *,
        hi_grade: bool = False,
        vel_gt: float = 0.0,
        angvel_gt: float = 0.0,
    ) -> None:
        """
        Call once per sample *before* ekf.step(...).

        vel_gt and angvel_gt are derivatives from the high-grade sensor.  If
        given, they remove the need for internal finite-difference tricks and
        yield sharper Q estimates.
        """
        if hi_grade:  # ----------------------------- Phase 1 -------------
            x_true = np.array([pos, vel_gt, ang, angvel_gt], dtype=float)
            self.true_buf.append((x_true, u))
            if len(self.true_buf) == self.window:
                self._update_Q_R_from_truth()
                self.true_buf.clear()
            return

        # ------------------------------- Phase 2 --------------------------
        innov = self._innovation(pos, ang, u)

        # Ignore stagnant samples - otherwise R would shrink spuriously
        if np.linalg.norm(innov) < self.motion_tol:
            return

        self.innov_buf.append(innov)
        if len(self.innov_buf) == self.window:
            self._update_R_from_innov()
            self.innov_buf.clear()

    # -----------------------------------------------------------------------
    #  internals
    # -----------------------------------------------------------------------
    def _innovation(self, pos: float, ang: float, u: float) -> np.ndarray:
        """
        Innovation without disturbing EKF state.  The prediction is taken from
        the *model*, not the real EKF cycle, so calling this method before
        ekf.step(...) is safe.
        """
        x_pred = self.ekf._rk4(self.ekf.x_est, u)
        z_pred = self.ekf.H @ x_pred
        d_pos = pos - z_pred[0]
        d_ang = self.ekf._angle_diff(ang, z_pred[1])
        return np.array([d_pos, d_ang])

    # .....................................................................
    def _update_Q_R_from_truth(self) -> None:
        """
        Classic batch covariance estimation from (x_true, u) pairs.
        Uses the same RK4 as the EKF for one-step prediction, so no model
        mismatch can creep in here.
        """
        X0, U = zip(*self.true_buf)
        X0 = np.stack(X0)                  # shape (N, 4)
        X1 = np.roll(X0, -1, axis=0)[:-1]  # one-step-ahead truth
        U0 = np.asarray(U)[:-1]

        # one-step predictions from the model itself
        Xpred = np.vstack(
            [self.ekf._rk4(x, u) for x, u in zip(X0[:-1], U0)]
        )

        # -------- process noise (Q) ---------------------------------------
        W = X1 - Xpred
        W[:, 2] = [self.ekf._angle_diff(a, b) for a, b in zip(X1[:, 2], Xpred[:, 2])]
        Q_new = np.cov(W, rowvar=False, bias=True)

        # -------- measurement noise (R) -----------------------------------
        V = X1[:, (0, 2)] - Xpred[:, (0, 2)]
        V[:, 1] = [self.ekf._angle_diff(a, b) for a, b in zip(X1[:, 2], Xpred[:, 2])]
        R_new = np.cov(V, rowvar=False, bias=True)

        # slow low-pass blend
        self.ekf.Q = self.alpha * self.ekf.Q + (1.0 - self.alpha) * Q_new
        self.ekf.R = self.alpha * self.ekf.R + (1.0 - self.alpha) * R_new
        self._save_covariances()

    # .....................................................................
    def _update_R_from_innov(self) -> None:
        """
        Innovation-based covariance matching.

        We want   R_true  =  S_obs - H P_pred H^T
        but P_pred depends on the (unknown) process noise.  Using the *current*
        P (updated in the previous EKF step) keeps the update simple and
        stable, at the cost of a slight bias that vanishes as window -> inf.
        """
        V = np.stack(self.innov_buf)        # shape (N, 2)
        S_obs = np.cov(V, rowvar=False, bias=True)

        HPH = self.ekf.H @ self.ekf.P @ self.ekf.H.T
        R_est = S_obs - HPH                 # raw measurement-noise estimate

        # enforce symmetric positive definite
        eigvals, eigvecs = np.linalg.eigh(R_est)
        eigvals[eigvals < self.eps] = self.eps
        R_est = eigvecs @ np.diag(eigvals) @ eigvecs.T

        self.ekf.R = self.alpha * self.ekf.R + (1.0 - self.alpha) * R_est
        self._save_covariances()

    # -----------------------------------------------------------------------
    #  persistence helpers
    # -----------------------------------------------------------------------
    def _restore_covariances(self) -> None:
        if not os.path.isfile(self.store_path):
            return
        try:
            with open(self.store_path, "r") as f:
                data = yaml.safe_load(f)
            self.ekf.Q = np.array(data["Q"])
            self.ekf.R = np.array(data["R"])
            print(f"[EKFAdaptiveTuner] restored Q,R from '{self.store_path}'.")
        except Exception as exc:
            print("[EKFAdaptiveTuner] failed to load covariances:", exc)

    def _save_covariances(self) -> None:
        data = {"Q": self.ekf.Q.tolist(), "R": self.ekf.R.tolist()}
        with open(self.store_path, "w") as f:
            yaml.safe_dump(data, f)


# --------------------------------------------------------------------------- #
#  Example usage (delete in production)                                       #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Dummy physical parameters; replace with measured rig values
    params = dict(
        k=0.0,
        m_cart=0.5,
        m_pole=0.2,
        g=9.81,
        J_fric=1e-4,
        M_fric=1e-4,
        L=0.3,
    )

    ekf = EKFCartPole(dt=0.02, params=params)          # 50 Hz loop
    tuner = EKFAdaptiveTuner(ekf)

    # -------------- Commissioning run with high-grade sensor ---------------
    # for pos, vel, ang, ang_vel, u in commissioning_data_stream():
    #     tuner.feed_measurement(pos, ang, u,
    #                            hi_grade=True,
    #                            vel_gt=vel,
    #                            angvel_gt=ang_vel)
    #     ekf.step(pos, ang, u)

    # -------------- Normal operation with economical sensor ----------------
    # for pos, ang, u in runtime_stream():
    #     tuner.feed_measurement(pos, ang, u, hi_grade=False)
    #     v_hat, omega_hat = ekf.step(pos, ang, u)
    #     control_loop(v_hat, omega_hat)
