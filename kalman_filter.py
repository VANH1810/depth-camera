# NEW (KF)
import numpy as np

class KalmanFilter3D:
    def __init__(self, dt, sigma_a=1.5, R_diag=(0.05, 0.05, 0.05)):
        """
        dt: thời gian giữa 2 frame (s)
        sigma_a: std gia tốc (m/s^2)
        R_diag: std đo lường (m) cho (X,Y,Z)
        """
        self.dt = float(dt)
        self.x  = np.zeros((6,1))  # [X,Y,Z,VX,VY,VZ]^T
        self.P  = np.eye(6) * 10.0
        self.P[3:,3:] *= 100.0     # vận tốc chưa biết → covariance lớn

        dt = self.dt
        self.F = np.array([
            [1,0,0,dt,0, 0],
            [0,1,0,0, dt,0],
            [0,0,1,0, 0, dt],
            [0,0,0,1, 0, 0],
            [0,0,0,0, 1, 0],
            [0,0,0,0, 0, 1],
        ], dtype=float)

        self.H = np.array([
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,1,0,0,0],
        ], dtype=float)

        q = sigma_a**2
        dt2 = dt*dt
        dt3 = dt2*dt
        dt4 = dt2*dt2
        self.Q = q * np.array([
            [dt4/4,    0,      0,    dt3/2,   0,      0],
            [0,     dt4/4,     0,       0,  dt3/2,   0],
            [0,        0,   dt4/4,      0,     0,  dt3/2],
            [dt3/2,   0,      0,      dt2,    0,      0],
            [0,     dt3/2,     0,       0,   dt2,     0],
            [0,        0,   dt3/2,      0,     0,    dt2],
        ], dtype=float)

        self.R = np.diag([r**2 for r in R_diag])  # var = std^2
        self.I = np.eye(6)
        self.maha_thresh = 7.81  # Chi-square(3 dof, ~95%)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z, adaptive_R_std=None):
        """
        z: (3,) hoặc (3,1) toạ độ đo (X,Y,Z). Nếu None → bỏ qua update.
        adaptive_R_std: tuple (sx,sy,sz) để thay R theo frame (tuỳ chọn)
        """
        if z is None:
            return self.x
        z = np.asarray(z, dtype=float).reshape(3,1)
        if adaptive_R_std is not None:
            self.R = np.diag([s**2 for s in adaptive_R_std])

        # Innovation
        y  = z - (self.H @ self.x)
        S  = self.H @ self.P @ self.H.T + self.R
        Sinv = np.linalg.inv(S)

        # Mahalanobis gating (loại outlier đo lường)
        maha = float(y.T @ Sinv @ y)
        if maha > self.maha_thresh:
            return self.x  # bỏ update nếu quá lệch

        K  = self.P @ self.H.T @ Sinv
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P
        return self.x
