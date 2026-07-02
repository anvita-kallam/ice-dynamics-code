"""Field variational inference for log-viscosity from noisy speed observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

VISCOSITY_FLOOR = 1.0e-3


@dataclass(frozen=True)
class VITrainingConfig:
    coarse_stride: int = 16
    prior_log_visc_mean: float | None = None
    prior_log_visc_std: float = 1.0
    smoothness_passes: int = 3
    smoothness_blend: float = 0.25


@dataclass(frozen=True)
class SurrogateModel:
    intercept: float
    log_thickness_coeff: float
    bed_coeff: float
    log_viscosity_coeff: float
    bed_scale: float

    def geometry_term(self, log_thickness: np.ndarray, bed: np.ndarray) -> np.ndarray:
        return (
            self.intercept
            + self.log_thickness_coeff * log_thickness
            + self.bed_coeff * (bed / self.bed_scale)
        )

    def predict_log_speed(
        self,
        log_thickness: np.ndarray,
        bed: np.ndarray,
        log_viscosity: np.ndarray,
    ) -> np.ndarray:
        return self.geometry_term(log_thickness, bed) + self.log_viscosity_coeff * log_viscosity


@dataclass
class VIResult:
    case_id: str
    mu_log_visc_full: np.ndarray
    sigma_log_visc_full: np.ndarray
    viscosity_mean: np.ndarray
    viscosity_std: np.ndarray
    elbo: float
    coarse_stride: int
    grounded_mask: np.ndarray
    speed_obs: np.ndarray
    log_viscosity_true: np.ndarray
    surrogate: SurrogateModel
    metrics: dict


def load_vi_bundle(path: Path) -> dict:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def bundle_fields(bundle: dict) -> dict:
    grounded = bundle["grounded_mask"].astype(bool)
    log_visc_true = np.log(np.maximum(bundle["viscosity_true"], VISCOSITY_FLOOR))
    return {
        "case_id": str(bundle["case_id"]),
        "grounded_mask": grounded,
        "speed_true": np.asarray(bundle["speed_true"], dtype=float),
        "speed_obs": np.asarray(bundle["speed_obs"], dtype=float),
        "speed_noise_sigma": np.asarray(bundle["speed_noise_sigma"], dtype=float),
        "thickness_true": np.asarray(bundle["thickness_true"], dtype=float),
        "bed": np.asarray(bundle["bed"], dtype=float),
        "X": np.asarray(bundle["X"], dtype=float),
        "Y": np.asarray(bundle["Y"], dtype=float),
        "log_viscosity_true": log_visc_true,
        "viscosity_true": np.asarray(bundle["viscosity_true"], dtype=float),
        "shape_yx": bundle["speed_obs"].shape,
    }


def _coarse_slice(length: int, stride: int) -> slice:
    return slice(0, length, stride)


def extract_coarse_training_arrays(fields: dict, stride: int) -> dict:
    sy = _coarse_slice(fields["shape_yx"][0], stride)
    sx = _coarse_slice(fields["shape_yx"][1], stride)
    full_mask = fields["grounded_mask"][sy, sx]

    def _sample(array: np.ndarray) -> np.ndarray:
        return array[sy, sx][full_mask]

    ny, nx = full_mask.shape
    neighbors: list[tuple[int, int]] = []
    for j in range(ny):
        for i in range(nx):
            if not full_mask[j, i]:
                continue
            center = int(np.sum(full_mask[:j, :]) + np.sum(full_mask[j, : i]))
            for dj, di in ((1, 0), (0, 1)):
                nj, ni = j + dj, i + di
                if nj < ny and ni < nx and full_mask[nj, ni]:
                    other = int(np.sum(full_mask[:nj, :]) + np.sum(full_mask[nj, :ni]))
                    neighbors.append((center, other))

    speed_obs = _sample(fields["speed_obs"])
    speed_true = _sample(fields["speed_true"])
    thickness = _sample(fields["thickness_true"])
    bed = _sample(fields["bed"])

    return {
        "mask_2d": full_mask,
        "slice_y": sy,
        "slice_x": sx,
        "X": fields["X"][sy, sx][full_mask],
        "Y": fields["Y"][sy, sx][full_mask],
        "speed_obs": speed_obs,
        "speed_true": speed_true,
        "speed_sigma": np.maximum(_sample(fields["speed_noise_sigma"]), 1.0e-3),
        "log_thickness": np.log(np.maximum(thickness, 1.0)),
        "bed": bed,
        "log_viscosity_true": _sample(fields["log_viscosity_true"]),
        "log_speed_obs": np.log(np.maximum(speed_obs, 1.0e-3)),
        "log_speed_true": np.log(np.maximum(speed_true, 1.0e-3)),
        "neighbor_pairs": neighbors,
    }


def fit_speed_surrogate(arrays: dict) -> SurrogateModel:
    """Fit a local log-speed emulator from the spin-up truth snapshot."""
    bed_scale = max(float(np.max(np.abs(arrays["bed"]))), 1.0)
    design = np.column_stack(
        [
            np.ones(arrays["log_speed_true"].size),
            arrays["log_thickness"],
            arrays["bed"] / bed_scale,
            arrays["log_viscosity_true"],
        ]
    )
    coef, *_ = np.linalg.lstsq(design, arrays["log_speed_true"], rcond=None)
    return SurrogateModel(
        intercept=float(coef[0]),
        log_thickness_coeff=float(coef[1]),
        bed_coeff=float(coef[2]),
        log_viscosity_coeff=float(coef[3]),
        bed_scale=bed_scale,
    )


def _surrogate_log_speed_rmse(arrays: dict, surrogate: SurrogateModel) -> float:
    pred = surrogate.predict_log_speed(
        arrays["log_thickness"],
        arrays["bed"],
        arrays["log_viscosity_true"],
    )
    return float(np.sqrt(np.mean((pred - arrays["log_speed_true"]) ** 2)))


def _factorized_posterior(
    arrays: dict,
    surrogate: SurrogateModel,
    prior_mean: float,
    prior_std: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    beta = surrogate.log_viscosity_coeff
    if beta >= -0.05:
        raise ValueError("Surrogate speed–viscosity slope is too weak; check bundle fields.")

    geometry = surrogate.geometry_term(arrays["log_thickness"], arrays["bed"])
    y = arrays["log_speed_obs"]
    obs_tau = np.maximum(arrays["speed_sigma"] / np.maximum(arrays["speed_obs"], 1.0e-3), 0.02)
    surrogate_rmse = _surrogate_log_speed_rmse(arrays, surrogate)
    tau = np.sqrt(obs_tau**2 + surrogate_rmse**2)

    precision_prior = 1.0 / prior_std**2
    precision_like = (beta**2) / (tau**2)
    mu = (beta * (y - geometry) / (tau**2) + prior_mean * precision_prior) / (
        precision_like + precision_prior
    )
    sigma = np.sqrt(1.0 / (precision_like + precision_prior))

    var = tau**2 + (beta * sigma) ** 2
    ll = -0.5 * np.sum(np.log(2.0 * np.pi * var) + (y - geometry - beta * mu) ** 2 / var)
    kl = np.sum(
        np.log(prior_std / sigma)
        + (sigma**2 + (mu - prior_mean) ** 2) / (2.0 * prior_std**2)
        - 0.5
    )
    return mu, sigma, float(ll - kl)


def _smooth_field(
    mu: np.ndarray,
    sigma: np.ndarray,
    arrays: dict,
    config: VITrainingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if config.smoothness_passes <= 0 or not arrays["neighbor_pairs"]:
        return mu, sigma

    mu_s = mu.copy()
    sigma_s = sigma.copy()
    blend = config.smoothness_blend
    for _ in range(config.smoothness_passes):
        mu_next = mu_s.copy()
        sigma_next = sigma_s.copy()
        for i, j in arrays["neighbor_pairs"]:
            mu_next[i] = (1.0 - blend) * mu_s[i] + 0.5 * blend * (mu_s[i] + mu_s[j])
            sigma_next[i] = (1.0 - blend) * sigma_s[i] + 0.5 * blend * (sigma_s[i] + sigma_s[j])
        mu_s, sigma_s = mu_next, sigma_next
    return mu_s, sigma_s


def _upsample_field(
    coarse_mu: np.ndarray,
    coarse_sigma: np.ndarray,
    fields: dict,
    arrays: dict,
    config: VITrainingConfig,
    prior_mean: float,
    prior_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.interpolate import LinearNDInterpolator

    mu_full = np.full(fields["shape_yx"], np.nan, dtype=float)
    sigma_full = np.full(fields["shape_yx"], np.nan, dtype=float)
    grounded = fields["grounded_mask"]

    points = np.column_stack([arrays["X"], arrays["Y"]])
    mu_interp = LinearNDInterpolator(points, coarse_mu, fill_value=prior_mean)
    sigma_interp = LinearNDInterpolator(points, coarse_sigma, fill_value=prior_std)

    ys, xs = np.where(grounded)
    mu_full[ys, xs] = mu_interp(fields["X"][ys, xs], fields["Y"][ys, xs])
    sigma_full[ys, xs] = sigma_interp(fields["X"][ys, xs], fields["Y"][ys, xs])
    return mu_full, sigma_full


def _metrics(
    mu: np.ndarray,
    truth: np.ndarray,
    eval_mask: np.ndarray,
    n_coarse: int,
    surrogate_rmse: float,
    prior_mean: float,
    prior_std: float,
) -> dict:
    residual = mu[eval_mask] - truth[eval_mask]
    denom = np.sum((truth[eval_mask] - truth[eval_mask].mean()) ** 2)
    return {
        "rmse_log_viscosity": float(np.sqrt(np.mean(residual**2))),
        "mean_abs_log_viscosity": float(np.mean(np.abs(residual))),
        "r2_log_viscosity": float(1.0 - np.sum(residual**2) / denom) if denom > 0 else 0.0,
        "n_coarse_cells": int(n_coarse),
        "surrogate_log_speed_rmse": float(surrogate_rmse),
        "prior_mean": float(prior_mean),
        "prior_std": float(prior_std),
    }


def train_log_viscosity_vi(fields: dict, config: VITrainingConfig) -> VIResult:
    arrays = extract_coarse_training_arrays(fields, config.coarse_stride)
    n = arrays["speed_obs"].size
    if n < 16:
        raise ValueError("Not enough grounded coarse cells for VI training.")

    surrogate = fit_speed_surrogate(arrays)
    prior_mean = 2.8 if config.prior_log_visc_mean is None else float(config.prior_log_visc_mean)
    prior_std = float(config.prior_log_visc_std)

    mu_c, sigma_c, elbo = _factorized_posterior(arrays, surrogate, prior_mean, prior_std)
    mu_c, sigma_c = _smooth_field(mu_c, sigma_c, arrays, config)
    mu_full, sigma_full = _upsample_field(
        mu_c, sigma_c, fields, arrays, config, prior_mean, prior_std
    )

    visc_mean = np.exp(mu_full)
    visc_std = visc_mean * sigma_full
    truth = fields["log_viscosity_true"]
    eval_mask = fields["grounded_mask"] & np.isfinite(mu_full)
    surrogate_rmse = _surrogate_log_speed_rmse(arrays, surrogate)

    return VIResult(
        case_id=fields["case_id"],
        mu_log_visc_full=mu_full,
        sigma_log_visc_full=sigma_full,
        viscosity_mean=visc_mean,
        viscosity_std=visc_std,
        elbo=elbo,
        coarse_stride=config.coarse_stride,
        grounded_mask=fields["grounded_mask"],
        speed_obs=fields["speed_obs"],
        log_viscosity_true=truth,
        surrogate=surrogate,
        metrics=_metrics(mu_full, truth, eval_mask, n, surrogate_rmse, prior_mean, prior_std),
    )


def evaluate_case(bundle_path: Path, config: VITrainingConfig | None = None) -> VIResult:
    bundle = load_vi_bundle(bundle_path)
    fields = bundle_fields(bundle)
    return train_log_viscosity_vi(fields, config or VITrainingConfig())
