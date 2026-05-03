import numpy as np
from .anomalies.base import Anomaly

class DatasetBuilder:
    def __init__(self, dx, dy, dz):
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def make_grid(self, shape_xyz):
        nx, ny, nz = shape_xyz
        # Create coordinates arrays
        # Assuming origin at (0,0,0) for simplicity, can be adjustable
        x = np.arange(nx) * self.dx
        y = np.arange(ny) * self.dy
        z = np.arange(nz) * self.dz
        # Meshgrid: indexing='ij' gives (nx, ny, nz) order if we pass x, y, z
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        return X, Y, Z

    def inject_properties(self, props_bg: dict, anomalies: list[Anomaly]):
        """
        Inject anomalies into a multiphysics dictionary of background models.
        props_bg: dict like {'vp': vp_array, 'rho': rho_array, ...}
        Returns:
            props: The updated property dictionaries.
            label: The anomaly label mask (0 for background).
            X, Y, Z: The coordinate grids used.
        """
        # Assume all properties have the same shape, use one to get axes
        sample_key = list(props_bg.keys())[0]
        X, Y, Z = self.make_grid(props_bg[sample_key].shape)
        
        props_new = {k: v.copy() for k, v in props_bg.items()}
        label = np.zeros_like(props_bg[sample_key], dtype=np.int16)
        
        for k, anom in enumerate(anomalies, start=1):
            m = anom.soft_mask(X, Y, Z)
            
            # Apply all multiphysics logic simultaneously
            props_new = anom.apply_properties(props_new, X, Y, Z)
            
            label[m > 0.5] = k
            
        return props_new, label, X, Y, Z

    def inject_anomalies(self, vp_bg: np.ndarray, anomalies: list[Anomaly]):
        """
        Backward compatible wrapper for injecting only into a Vp background model.
        """
        props, label, X, Y, Z = self.inject_properties({'vp': vp_bg}, anomalies)
        return props['vp'], label, X, Y, Z
