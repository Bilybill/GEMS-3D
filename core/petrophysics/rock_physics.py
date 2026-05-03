import numpy as np
from scipy.ndimage import gaussian_filter


class PetrophysicsConverter:
    FACIES_SHALE = 0
    FACIES_SANDSTONE = 1
    FACIES_CARBONATE = 2

    FACIES_NAMES = {
        FACIES_SHALE: "shale",
        FACIES_SANDSTONE: "sandstone",
        FACIES_CARBONATE: "carbonate",
    }

    FACIES_PARAMS = {
        FACIES_SHALE: {
            "density_a": 0.255,
            "density_b": 0.275,
            "density_clip": (1.95, 2.90),
            "vm": 3950.0,
            "vf": 1500.0,
            "phi0": 0.12,
            "phi_min": 0.02,
            "phi_max": 0.16,
            "compaction": 3.2,
            "phi_blend": 0.18,
            "rw_surface": 0.65,
            "rw_deep": 0.32,
            "sw_surface": 0.94,
            "sw_deep": 0.98,
            "archie_a": 1.15,
            "archie_m": 2.05,
            "archie_n": 1.95,
            "clay_base": 0.34,
            "clay_span": 0.16,
            "clay_res_base": 8.0,
            "res_clip": (2.0, 120.0),
            "chi_mean": 8.0e-5,
            "chi_span": 5.0e-5,
        },
        FACIES_SANDSTONE: {
            "density_a": 0.292,
            "density_b": 0.255,
            "density_clip": (2.05, 2.85),
            "vm": 5400.0,
            "vf": 1500.0,
            "phi0": 0.22,
            "phi_min": 0.02,
            "phi_max": 0.26,
            "compaction": 4.0,
            "phi_blend": 0.35,
            "rw_surface": 1.10,
            "rw_deep": 0.48,
            "sw_surface": 0.64,
            "sw_deep": 0.88,
            "archie_a": 1.0,
            "archie_m": 2.05,
            "archie_n": 2.00,
            "clay_base": 0.02,
            "clay_span": 0.08,
            "clay_res_base": 18.0,
            "res_clip": (25.0, 2500.0),
            "chi_mean": 2.5e-5,
            "chi_span": 1.5e-5,
        },
        FACIES_CARBONATE: {
            "density_a": 0.308,
            "density_b": 0.248,
            "density_clip": (2.20, 3.00),
            "vm": 6400.0,
            "vf": 1500.0,
            "phi0": 0.08,
            "phi_min": 0.005,
            "phi_max": 0.12,
            "compaction": 5.2,
            "phi_blend": 0.30,
            "rw_surface": 1.60,
            "rw_deep": 0.72,
            "sw_surface": 0.38,
            "sw_deep": 0.72,
            "archie_a": 0.95,
            "archie_m": 2.35,
            "archie_n": 2.15,
            "clay_base": 0.0,
            "clay_span": 0.02,
            "clay_res_base": 40.0,
            "res_clip": (150.0, 15000.0),
            "chi_mean": 8.0e-6,
            "chi_span": 8.0e-6,
        },
    }

    def __init__(self, random_seed=7):
        self.rng = np.random.default_rng(random_seed)
        self._background_state = None
        self.last_background_qc = None

    def _generate_correlated_noise(self, shape, sigma=3.0):
        noise = self.rng.normal(0.0, 1.0, shape)
        filtered_noise = gaussian_filter(noise, sigma=sigma)
        std_val = np.std(filtered_noise)
        if std_val > 1e-8:
            filtered_noise = filtered_noise / std_val
        return filtered_noise.astype(np.float32)

    def _depth_norm(self, shape):
        nz = shape[2]
        if nz <= 1:
            return np.zeros(shape, dtype=np.float32)
        z = np.linspace(0.0, 1.0, nz, dtype=np.float32)
        return np.broadcast_to(z.reshape(1, 1, nz), shape)

    def _vp_norm(self, vp_model):
        q05, q95 = np.percentile(vp_model, [5.0, 95.0])
        scale = max(float(q95 - q05), 1.0)
        return np.clip((vp_model - q05) / scale, 0.0, 1.0).astype(np.float32)

    def _facies_from_score(self, score):
        facies = np.full(score.shape, self.FACIES_SANDSTONE, dtype=np.int16)
        facies[score < 0.38] = self.FACIES_SHALE
        facies[score >= 0.72] = self.FACIES_CARBONATE
        return facies

    def _layer_base_facies_from_score(self, score):
        facies = np.full(score.shape, self.FACIES_SANDSTONE, dtype=np.int16)
        facies[score < 0.34] = self.FACIES_SHALE
        facies[score >= 0.68] = self.FACIES_CARBONATE
        return facies

    def _refine_layer_facies(self, base_facies, local_score):
        cell_facies = np.full(local_score.shape, base_facies, dtype=np.int16)
        if base_facies == self.FACIES_SHALE:
            cell_facies[local_score > 0.62] = self.FACIES_SANDSTONE
        elif base_facies == self.FACIES_SANDSTONE:
            cell_facies[local_score < 0.28] = self.FACIES_SHALE
            cell_facies[local_score > 0.80] = self.FACIES_CARBONATE
        elif base_facies == self.FACIES_CARBONATE:
            cell_facies[local_score < 0.44] = self.FACIES_SANDSTONE
        return cell_facies

    def _infer_facies_model(self, vp_model, label_vol):
        depth_norm = self._depth_norm(vp_model.shape)
        vp_norm = self._vp_norm(vp_model)
        labels = np.unique(label_vol.astype(np.int32))
        label_summary = {}
        facies = np.full(vp_model.shape, self.FACIES_SANDSTONE, dtype=np.int16)
        layer_noise = self._generate_correlated_noise(vp_model.shape, sigma=12.0)

        if labels.size <= 1:
            global_score = 0.60 * vp_norm + 0.40 * depth_norm + 0.03 * layer_noise
            facies = self._layer_base_facies_from_score(global_score)
            label_summary[int(labels[0]) if labels.size == 1 else 0] = {
                "base_facies": "mixed",
                "median_vp_norm": float(np.median(vp_norm)),
                "median_depth_norm": float(np.median(depth_norm)),
                "score": float(np.median(global_score)),
            }
            return facies, label_summary, depth_norm, vp_norm

        for label in labels:
            mask = label_vol == label
            if not np.any(mask):
                continue

            label_vp = vp_norm[mask]
            label_depth = depth_norm[mask]
            label_vp_med = float(np.median(label_vp))
            label_depth_med = float(np.median(label_depth))
            label_score = 0.55 * label_vp_med + 0.45 * label_depth_med
            base_facies = int(self._layer_base_facies_from_score(np.array([label_score], dtype=np.float32))[0])
            local_score = (
                0.18 * vp_norm[mask]
                + 0.12 * depth_norm[mask]
                + 0.70 * label_score
                + 0.03 * layer_noise[mask]
            )
            cell_facies = self._refine_layer_facies(base_facies, local_score)

            facies[mask] = cell_facies
            label_summary[int(label)] = {
                "base_facies": self.FACIES_NAMES[base_facies],
                "median_vp_norm": label_vp_med,
                "median_depth_norm": label_depth_med,
                "score": label_score,
            }

        return facies, label_summary, depth_norm, vp_norm

    def _build_background_controls(self, vp_model, facies, depth_norm, vp_norm):
        shape = vp_model.shape
        phi_model = np.zeros(shape, dtype=np.float32)
        rw_model = np.zeros(shape, dtype=np.float32)
        sw_model = np.ones(shape, dtype=np.float32)
        rho_model = np.zeros(shape, dtype=np.float32)
        chi_model = np.zeros(shape, dtype=np.float32)

        noise_phi = self._generate_correlated_noise(shape, sigma=10.0)
        noise_rw = self._generate_correlated_noise(shape, sigma=14.0)
        noise_sw = self._generate_correlated_noise(shape, sigma=12.0)
        noise_rho = self._generate_correlated_noise(shape, sigma=8.0)
        noise_chi = self._generate_correlated_noise(shape, sigma=6.0)

        for facies_id, params in self.FACIES_PARAMS.items():
            mask = facies == facies_id
            if not np.any(mask):
                continue

            vp_local = vp_model[mask]
            depth_local = depth_norm[mask]
            vp_clipped = np.clip(vp_local, params["vf"] + 20.0, params["vm"] - 50.0)

            phi_depth = params["phi_min"] + (params["phi0"] - params["phi_min"]) * np.exp(-params["compaction"] * depth_local)
            phi_vp = (1.0 / vp_clipped - 1.0 / params["vm"]) / (1.0 / params["vf"] - 1.0 / params["vm"])
            phi_vp = np.clip(phi_vp, params["phi_min"], params["phi_max"])

            phi_local = params["phi_blend"] * phi_vp + (1.0 - params["phi_blend"]) * phi_depth
            phi_local = phi_local * np.exp(0.10 * noise_phi[mask])
            phi_model[mask] = np.clip(phi_local, params["phi_min"], params["phi_max"])

            rw_local = params["rw_surface"] + (params["rw_deep"] - params["rw_surface"]) * depth_local
            rw_local = rw_local * np.exp(0.08 * noise_rw[mask])
            rw_model[mask] = np.clip(rw_local, 0.03, 2.5)

            sw_local = params["sw_surface"] + (params["sw_deep"] - params["sw_surface"]) * depth_local
            sw_local = sw_local + 0.03 * noise_sw[mask]
            sw_model[mask] = np.clip(sw_local, 0.15, 1.0)

            rho_local = params["density_a"] * np.power(np.clip(vp_local, 1200.0, None), params["density_b"])
            rho_local = rho_local * np.exp(0.035 * noise_rho[mask])
            rho_model[mask] = np.clip(rho_local, *params["density_clip"])

            chi_local = params["chi_mean"] + params["chi_span"] * 0.35 * noise_chi[mask]
            chi_model[mask] = np.clip(chi_local, 0.0, None)

        return rho_model, rw_model, sw_model, phi_model, chi_model

    def _compute_resistivity_from_controls(self, facies, vp_norm, depth_norm, phi_model, sw_model, rw_model):
        res_model = np.zeros_like(phi_model, dtype=np.float32)

        for facies_id, params in self.FACIES_PARAMS.items():
            mask = facies == facies_id
            if not np.any(mask):
                continue

            phi = np.clip(phi_model[mask], params["phi_min"], params["phi_max"])
            sw = np.clip(sw_model[mask], 0.05, 1.0)
            rw = np.clip(rw_model[mask], 0.03, 2.5)
            vp_local = vp_norm[mask]
            depth_local = depth_norm[mask]

            sigma_clean = np.power(phi, params["archie_m"]) * np.power(sw, params["archie_n"]) / (params["archie_a"] * rw)
            clay_frac = params["clay_base"] + params["clay_span"] * (1.0 - vp_local) * (0.65 + 0.35 * (1.0 - depth_local))
            clay_frac = np.clip(clay_frac, 0.0, 0.95)
            rho_sh = params["clay_res_base"] * (1.0 + 0.35 * depth_local)
            sigma_clay = clay_frac / np.clip(rho_sh, 0.1, None)
            sigma_total = sigma_clean * (1.0 - 0.25 * clay_frac) + sigma_clay
            res_local = 1.0 / np.clip(sigma_total, 1.0e-6, None)
            res_model[mask] = np.clip(res_local, *params["res_clip"])

        return res_model

    def _summarize_background(self, res_model, facies, label_summary):
        positive = res_model[res_model > 0.0]
        if positive.size == 0:
            self.last_background_qc = None
            return

        summary = {
            "min": float(np.min(positive)),
            "p05": float(np.percentile(positive, 5.0)),
            "median": float(np.median(positive)),
            "p95": float(np.percentile(positive, 95.0)),
            "max": float(np.max(positive)),
            "mean": float(np.mean(positive)),
            "facies_fraction": {
                self.FACIES_NAMES[idx]: float(np.mean(facies == idx))
                for idx in self.FACIES_NAMES
            },
            "label_summary": label_summary,
        }
        self.last_background_qc = summary

        print("Background facies fractions:")
        for facies_name, fraction in summary["facies_fraction"].items():
            print(f"  - {facies_name}: {fraction:.2%}")
        print(
            "Background resistivity stats: "
            f"min={summary['min']:.3g}, p05={summary['p05']:.3g}, median={summary['median']:.3g}, "
            f"p95={summary['p95']:.3g}, max={summary['max']:.3g}, mean={summary['mean']:.3g}"
        )

    def get_last_background_state(self):
        return self._background_state

    def generate_background(self, vp_model, label_vol=None):
        print("Generating facies-aware background multi-physics properties...")
        vp_model = np.asarray(vp_model, dtype=np.float32)
        if label_vol is None:
            label_vol = np.zeros_like(vp_model, dtype=np.int32)
        else:
            label_vol = np.asarray(label_vol, dtype=np.int32)
            if label_vol.shape != vp_model.shape:
                raise ValueError("label_vol must have the same shape as vp_model")

        facies, label_summary, depth_norm, vp_norm = self._infer_facies_model(vp_model, label_vol)
        rho_model, rw_model, sw_model, phi_model, chi_model = self._build_background_controls(vp_model, facies, depth_norm, vp_norm)
        res_model = self._compute_resistivity_from_controls(facies, vp_norm, depth_norm, phi_model, sw_model, rw_model)

        self._background_state = {
            "facies": facies.copy(),
            "depth_norm": depth_norm,
            "vp_norm": vp_norm,
            "phi": phi_model.copy(),
            "rw": rw_model.copy(),
            "sw": sw_model.copy(),
            "rho": rho_model.copy(),
            "res": res_model.copy(),
            "chi": chi_model.copy(),
            "label_summary": label_summary,
        }
        self._summarize_background(res_model, facies, label_summary)
        return rho_model, res_model, chi_model

    def _generate_heterogeneous_prop(self, shape, mask, low, high, noise_level=0.05, sigma=4.0):
        base_val = self.rng.uniform(low, high)
        noise = self._generate_correlated_noise(shape, sigma=sigma)
        values = base_val + base_val * noise_level * noise[mask]
        return np.clip(values, low * 0.8, high * 1.2)

    def _generate_log_heterogeneous_prop(self, shape, mask, low, high, noise_level=0.12, sigma=4.0):
        log_low = np.log10(low)
        log_high = np.log10(high)
        log_base = self.rng.uniform(log_low, log_high)
        noise = self._generate_correlated_noise(shape, sigma=sigma)
        log_values = log_base + noise_level * noise[mask]
        log_values = np.clip(log_values, log_low - 0.2, log_high + 0.2)
        return np.power(10.0, log_values)

    def apply_anomaly(self, mask, anomaly_type, vp, rho, res, chi):
        mask = mask.astype(bool)
        if not np.any(mask):
            return vp, rho, res, chi

        vp = np.asarray(vp, dtype=np.float32)
        rho = np.asarray(rho, dtype=np.float32)
        res = np.asarray(res, dtype=np.float32)
        chi = np.asarray(chi, dtype=np.float32)
        shape = vp.shape

        state = self._background_state
        if state is None or state["facies"].shape != shape:
            facies, _, depth_norm, vp_norm = self._infer_facies_model(vp, np.zeros_like(vp, dtype=np.int32))
            phi = np.full(shape, 0.12, dtype=np.float32)
            rw = np.full(shape, 0.18, dtype=np.float32)
            sw = np.ones(shape, dtype=np.float32)
        else:
            facies = state["facies"].copy()
            depth_norm = state["depth_norm"]
            vp_norm = state["vp_norm"]
            phi = state["phi"].copy()
            rw = state["rw"].copy()
            sw = state["sw"].copy()

        noise = self._generate_correlated_noise(shape, sigma=6.0)

        if anomaly_type == "Gas":
            vp[mask] = np.clip(vp[mask] * np.clip(0.62 + 0.08 * noise[mask], 0.45, 0.82), 1400.0, None)
            rho[mask] = np.clip(rho[mask] * np.clip(0.90 + 0.04 * noise[mask], 0.82, 0.98), 1.40, None)
            facies[mask] = np.maximum(facies[mask], self.FACIES_SANDSTONE)
            phi[mask] = np.clip(phi[mask] * np.clip(1.10 + 0.05 * noise[mask], 0.95, 1.25), 0.05, 0.35)
            sw[mask] = np.clip(0.20 + 0.10 * (1.0 + noise[mask]), 0.05, 0.45)
            rw[mask] = np.clip(rw[mask] * np.clip(1.05 + 0.05 * noise[mask], 0.9, 1.2), 0.03, 2.5)
            chi[mask] = np.clip(-1.0e-5 + 2.0e-6 * noise[mask], -2.0e-5, 0.0)
            vp_norm = self._vp_norm(vp)
            res = self._compute_resistivity_from_controls(facies, vp_norm, depth_norm, phi, sw, rw)

        elif anomaly_type == "Hydrate":
            vp[mask] = np.clip(vp[mask] * np.clip(1.18 + 0.08 * noise[mask], 1.05, 1.35), None, 4500.0)
            rho[mask] = np.clip(rho[mask] * np.clip(0.93 + 0.03 * noise[mask], 0.86, 0.99), 1.50, None)
            facies[mask] = np.maximum(facies[mask], self.FACIES_SANDSTONE)
            phi[mask] = np.clip(phi[mask] * np.clip(0.86 + 0.05 * noise[mask], 0.70, 0.96), 0.02, 0.25)
            sw[mask] = np.clip(0.45 + 0.12 * (1.0 + noise[mask]), 0.20, 0.70)
            vp_norm = self._vp_norm(vp)
            res = self._compute_resistivity_from_controls(facies, vp_norm, depth_norm, phi, sw, rw)

        elif anomaly_type == "BrineFault":
            vp[mask] = np.clip(vp[mask] * np.clip(0.82 + 0.06 * noise[mask], 0.65, 0.96), 1400.0, None)
            rho[mask] = np.clip(rho[mask] * np.clip(0.92 + 0.04 * noise[mask], 0.80, 0.98), 1.50, None)
            facies[mask] = self.FACIES_SHALE
            phi[mask] = np.clip(phi[mask] * np.clip(1.22 + 0.10 * noise[mask], 1.05, 1.60), 0.08, 0.40)
            sw[mask] = np.clip(0.95 + 0.03 * noise[mask], 0.80, 1.0)
            rw[mask] = np.clip(rw[mask] * np.clip(0.22 + 0.06 * (1.0 + noise[mask]), 0.08, 0.40), 0.01, 0.5)
            vp_norm = self._vp_norm(vp)
            res = self._compute_resistivity_from_controls(facies, vp_norm, depth_norm, phi, sw, rw)

        elif anomaly_type == "Sulfide":
            vp[mask] = self._generate_heterogeneous_prop(shape, mask, 5500.0, 6800.0, noise_level=0.04)
            rho[mask] = self._generate_heterogeneous_prop(shape, mask, 3.50, 4.80, noise_level=0.05)
            res[mask] = self._generate_log_heterogeneous_prop(shape, mask, 0.001, 0.1, noise_level=0.22, sigma=3.0)
            chi[mask] = self._generate_heterogeneous_prop(shape, mask, 0.05, 0.2, noise_level=0.15)

        elif anomaly_type == "Igneous":
            vp[mask] = self._generate_heterogeneous_prop(shape, mask, 5000.0, 6500.0, noise_level=0.05)
            rho[mask] = self._generate_heterogeneous_prop(shape, mask, 2.70, 3.10, noise_level=0.03)
            res[mask] = self._generate_log_heterogeneous_prop(shape, mask, 2000.0, 10000.0, noise_level=0.15, sigma=6.0)
            chi[mask] = self._generate_heterogeneous_prop(shape, mask, 0.01, 0.08, noise_level=0.20)

        elif anomaly_type == "Serpentinized":
            vp[mask] = np.clip(vp[mask] * np.clip(0.76 + 0.08 * noise[mask], 0.55, 0.92), 1800.0, None)
            rho[mask] = np.clip(rho[mask] * np.clip(0.88 + 0.04 * noise[mask], 0.75, 0.97), 1.80, None)
            res_scale = np.clip(0.65 + 0.12 * noise[mask], 0.25, 0.95)
            res[mask] = np.clip(res[mask] * res_scale, 1.0, 200.0)
            chi[mask] = np.clip(chi[mask] + self._generate_heterogeneous_prop(shape, mask, 0.01, 0.05, noise_level=0.20), 0.0, 0.1)

        elif anomaly_type == "SaltDome":
            vp[mask] = self._generate_heterogeneous_prop(shape, mask, 4200.0, 5500.0, noise_level=0.02, sigma=14.0)
            rho[mask] = self._generate_heterogeneous_prop(shape, mask, 2.10, 2.25, noise_level=0.01, sigma=18.0)
            res[mask] = self._generate_log_heterogeneous_prop(shape, mask, 8000.0, 30000.0, noise_level=0.08, sigma=10.0)
            chi[mask] = np.clip(-1.0e-5 + 2.0e-6 * noise[mask], -2.0e-5, 0.0)

        else:
            print(f"Warning: Unknown anomaly type '{anomaly_type}'")

        self._background_state = {
            "facies": facies.copy(),
            "depth_norm": depth_norm,
            "vp_norm": self._vp_norm(vp),
            "phi": phi.copy(),
            "rw": rw.copy(),
            "sw": sw.copy(),
            "rho": rho.copy(),
            "res": res.copy(),
            "chi": chi.copy(),
            "label_summary": state["label_summary"] if state is not None and "label_summary" in state else {},
        }
        return vp, rho, res, chi
