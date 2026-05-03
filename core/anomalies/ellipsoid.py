from dataclasses import dataclass
import numpy as np
from .base import Anomaly

@dataclass
class EllipsoidAnomaly(Anomaly):
    center: tuple   # (cx, cy, cz) in meters
    axes: tuple     # (a, b, c) in meters
    R: np.ndarray   # 3x3 rotation matrix

    def mask(self, X, Y, Z):
        cx, cy, cz = self.center
        a, b, c = self.axes
        # shift
        P = np.stack([X-cx, Y-cy, Z-cz], axis=0)     # (3, z,y,x)
        # rotate: q = R.T @ p
        # Note: reshaping for matrix multiplication then reshaping back
        orig_shape = X.shape
        P_flat = P.reshape(3, -1)
        q = self.R.T @ P_flat
        qx, qy, qz = q[0], q[1], q[2]
        
        val = (qx/a)**2 + (qy/b)**2 + (qz/c)**2
        return (val <= 1.0).reshape(orig_shape)
