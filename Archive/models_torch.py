#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def unpack_variables(params):
    return params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4]


def assemble_scale_tensors(norms, keys, dtype):
    values = np.array([norms[key].denom for key in keys], dtype=np.float64)
    W = np.diag(values)
    iW = np.diag(1.0 / values)
    b = np.array([norms[key].xmin for key in keys], dtype=np.float64).reshape(1, len(keys))
    return (
        torch.tensor(W, dtype=dtype),
        torch.tensor(iW, dtype=dtype),
        torch.tensor(b, dtype=dtype),
    )


def normalize_tensor(x, iW, b, pos=False):
    if pos:
        return (x - b) @ iW
    return 2.0 * ((x - b) @ iW) - 1.0


def inverse_normalize_tensor(xn, W, b, pos=False):
    if pos:
        return xn @ W + b
    return 0.5 * (xn + 1.0) @ W + b


def normal_log_prob(x, std):
    return -0.5 * math.log(2.0 * math.pi) - torch.log(std) - 0.5 * (x / std).square()

def grad(q, z):
    return torch.autograd.grad(
                q.sum(), z,
                create_graph=True,
                retain_graph=True
    )[0]


# CHANGED: icepack unit system (m, yr, MPa) for SSA physics aligned with spin-up.
def icepack_ssa_constants(pars, torch_dtype, device):
    """Return icepack-scaled constants used in IceStream SSA."""
    year = float(getattr(pars.prior, 'year', 3600 * 24 * 365.25))
    g = torch.tensor(9.81 * year ** 2, dtype=torch_dtype, device=device)
    rho_ice = torch.tensor(917.0 / year ** 2 * 1.0e-6, dtype=torch_dtype, device=device)
    rho_water = torch.tensor(1024.0 / year ** 2 * 1.0e-6, dtype=torch_dtype, device=device)
    glen_n = float(getattr(pars.prior, 'glen_exponent', 3.0))
    weertman_m = float(getattr(pars.prior, 'weertman_exponent', 3.0))
    eps_min = torch.tensor(
        float(getattr(pars.prior, 'strain_rate_min', 1.0e-5)),
        dtype=torch_dtype,
        device=device,
    )
    fluidity_A = torch.tensor(
        float(
            getattr(
                pars.prior,
                'fluidity_A',
                getattr(pars.prior, 'fluidity_a', 3.985e-13 * year * 1.0e18),
            )
        ),
        dtype=torch_dtype,
        device=device,
    )
    friction_C = torch.tensor(
        float(
            getattr(
                pars.prior,
                'friction_C',
                getattr(pars.prior, 'friction_c', 1.0),
            )
        ),
        dtype=torch_dtype,
        device=device,
    )
    return {
        'year': year,
        'g': g,
        'rho_ice': rho_ice,
        'rho_water': rho_water,
        'glen_n': glen_n,
        'weertman_m': weertman_m,
        'eps_min': eps_min,
        'fluidity_A': fluidity_A,
        'friction_C': friction_C,
    }


# CHANGED: spin-up notebooks use a regularized plastic basal law (not default Weertman).
def spinup_plastic_basal_drag(u, v, speed, tau_c, friction_C, weertman_m, speed_eps):
    """
    Basal drag from the spin-up ``friction`` potential
    phi = tau_c * ((u_c**(1/m+1) + |u|**(1/m+1))**(m/(m+1)) - u_c),
    u_c = (tau_c / C)**m.  Returns (tau_bx, tau_by) = d(phi)/d(u).
    """
    alpha = 1.0 / weertman_m + 1.0
    beta = weertman_m / (weertman_m + 1.0)
    u_c = (tau_c / friction_C).clamp_min(1.0e-30) ** weertman_m
    u_b = speed.clamp_min(speed_eps)
    u_hat_x = u / u_b
    u_hat_y = v / u_b
    inner_term = u_c ** alpha + u_b ** alpha
    coeff = tau_c * beta * alpha * inner_term.clamp_min(1.0e-30) ** (beta - 1.0) * u_b ** (alpha - 1.0)
    return coeff * u_hat_x, coeff * u_hat_y


# CHANGED: icepack Glen effective strain rate (viscosity._effective_strain_rate).
def glen_effective_strain_rate(u_x, u_y, v_x, v_y, eps_min):
    eps_xx = u_x
    eps_yy = v_y
    eps_xy = 0.5 * (u_y + v_x)
    trace_eps = u_x + v_y
    return torch.sqrt(
        0.5 * (
            eps_xx.square()
            + eps_yy.square()
            + 2.0 * eps_xy.square()
            + trace_eps.square()
        )
        + eps_min.square()
    )


# CHANGED: icepack Glen effective viscosity μ = 0.5 A^{-1/n} ε_e^{1/n-1} (MPa·yr).
def glen_effective_viscosity(eps_eff, fluidity_A, glen_n):
    return 0.5 * fluidity_A.pow(-1.0 / glen_n) * eps_eff.pow(1.0 / glen_n - 1.0)

class DenseNetwork(nn.Module):

    def __init__(self, layer_sizes, dtype=torch.float64):
        super().__init__()
        layers = []
        for in_features, out_features in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_features, out_features, dtype=dtype))
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs, activate_outputs=False):
        out = inputs
        for idx, layer in enumerate(self.layers):
            out = layer(out)
            if idx != len(self.layers) - 1:
                out = torch.tanh(out)
        if activate_outputs:
            out = torch.tanh(out)
        return out


class ResidualNetwork(nn.Module):

    def __init__(self, layer_sizes, dtype=torch.float64):
        super().__init__()
        layers = []
        for in_features, out_features in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_features, out_features, dtype=dtype))
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs, activate_outputs=False):
        x = inputs
        for idx, layer in enumerate(self.layers):
            z = layer(x)
            if 0 < idx < len(self.layers) - 1:
                x = x + torch.tanh(z)
            else:
                x = z
        if activate_outputs:
            x = torch.tanh(x)
        return x


class MeanNetwork(nn.Module):

    def __init__(self, norms, resnet=False, dtype=torch.float64):
        super().__init__()
        network_cls = ResidualNetwork if resnet else DenseNetwork

        # Coordinate-only state field:
        #   (x, y) -> (u_hat, v_hat, s_hat, H_hat)
        # The fourth output is ice thickness H, not bed elevation. Bed is
        # derived wherever it is needed as b_hat = s_hat - H_hat. Do not feed
        # observed u/v/s/b/H or masks into this branch; otherwise autograd
        # derivatives are derivatives of an observation-conditioned map rather
        # than spatial slopes.
        self.state_dense = network_cls([2, 128, 128, 128, 128, 4], dtype=dtype)

        W_coord, iW_coord, b_coord = assemble_scale_tensors(norms, ['x', 'y'], dtype)
        W_out, iW_out, b_out = assemble_scale_tensors(norms, ['u', 'v', 's', 'h'], dtype)

        self.register_buffer('W_coord', W_coord)
        self.register_buffer('iW_coord', iW_coord)
        self.register_buffer('b_coord', b_coord)
        self.register_buffer('W_out', W_out)
        self.register_buffer('iW_out', iW_out)
        self.register_buffer('b_out', b_out)

    @property
    def state_net(self):
        """Alias for the coordinate-only state network.

        The registered module name remains ``state_dense`` so compatible layers
        from old checkpoints can still be partially restored.
        """
        return self.state_dense

    def forward(
        self,
        x,
        y,
        u_in=None,
        v_in=None,
        s_in=None,
        b_in=None,
        uv_mask=None,
        inverse_norm=False,
    ):
        del u_in, v_in, s_in, b_in, uv_mask
        coords = torch.cat([x, y], dim=1)
        normalized_coords = normalize_tensor(coords, self.iW_coord, self.b_coord)
        params = self.state_net(normalized_coords)
        if inverse_norm:
            params = inverse_normalize_tensor(params, self.W_out, self.b_out)
        # Returns (u, v, s, H). Bed is not a network output; use b = s - H.
        return unpack_variables(params)


def build_inducing_points_physical(x, y, num_inducing_x, num_inducing_y, placement='ice_fps'):
    """
    Build inducing locations in physical (x, y) coordinates.

    placement:
      - 'bbox_grid': uniform mesh over the bounding box (may place points off-ice)
      - 'ice_fps': farthest-point sample on the provided ice coordinates (default)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size == 0 or y.size == 0:
        raise ValueError('Cannot build inducing points from empty ice coordinates')
    n_target = int(num_inducing_x) * int(num_inducing_y)
    if n_target <= 0:
        raise ValueError(f'Need positive inducing count, got {num_inducing_x} x {num_inducing_y}')

    placement = str(placement or 'ice_fps').lower()
    if placement == 'bbox_grid':
        x_ind = np.linspace(x.min(), x.max(), int(num_inducing_x))
        y_ind = np.linspace(y.min(), y.max(), int(num_inducing_y))
        x_ind, y_ind = [arr.ravel() for arr in np.meshgrid(x_ind, y_ind)]
        return np.column_stack((x_ind, y_ind))

    if placement not in ('ice_fps', 'fps', 'ice'):
        raise ValueError(
            f"Unknown inducing_placement={placement!r}; use 'ice_fps' or 'bbox_grid'")

    xy = np.column_stack((x, y))
    # Subsample dense ice clouds before FPS for O(N * M) cost control.
    max_candidates = max(n_target * 40, 8000)
    if xy.shape[0] > max_candidates:
        rng = np.random.default_rng(0)
        choose = rng.choice(xy.shape[0], size=max_candidates, replace=False)
        xy = xy[choose]

    n = xy.shape[0]
    m = min(n_target, n)
    # Seed near the ice centroid so the first points cover the main mass.
    seed = int(np.argmin(np.sum((xy - xy.mean(axis=0)) ** 2, axis=1)))
    selected = [seed]
    dists = np.full(n, np.inf, dtype=np.float64)
    for _ in range(1, m):
        last = xy[selected[-1]]
        d = np.sqrt(np.sum((xy - last) ** 2, axis=1))
        dists = np.minimum(dists, d)
        selected.append(int(np.argmax(dists)))
    return xy[np.asarray(selected, dtype=np.int64)]


class SparseVariationalGP(nn.Module):
    """
    Minimal sparse variational GP for the 2D latent fields used in joint training.
    This implementation exposes predictive mean/stddev for local batches and an
    inducing-space KL term that is stable under DDP. That KL is intentionally not
    the original batch-point KL used by the TensorFlow code.

    Optional kernel knobs (defaults preserve the historical RBF isotropic GP used
    by joint training):
      kernel_type: 'rbf' | 'matern12' | 'matern32' | 'matern52'
      anisotropic: separate length scales in x and y (normalized coords)
      learnable_length_scale: if False, length-scale params are frozen
    """

    def __init__(self, x, y, num_inducing_x, num_inducing_y, norms,
                 trainable_obs_variance=False, amplitude_init=0.2,
                 length_scale_init=0.3, noise_variance_init=1.0e-4,
                 jitter=1.0e-5, dtype=torch.float64, inducing_placement='ice_fps',
                 kernel_type='rbf', anisotropic=False, learnable_length_scale=True,
                 length_scale_y_init=None):
        super().__init__()
        X_phys = build_inducing_points_physical(
            x, y, num_inducing_x, num_inducing_y, placement=inducing_placement)
        X_ind = np.column_stack((norms['x'](X_phys[:, 0]), norms['y'](X_phys[:, 1])))
        num_inducing = X_ind.shape[0]

        self.register_buffer('inducing_index_points', torch.tensor(X_ind, dtype=dtype))
        self.raw_amplitude = nn.Parameter(torch.tensor(math.log(amplitude_init), dtype=dtype))
        self.kernel_type = str(kernel_type or 'rbf').strip().lower()
        self.anisotropic = bool(anisotropic)
        self.jitter = jitter
        self.inducing_placement = str(inducing_placement)

        ls_x = float(length_scale_init)
        ls_y = float(length_scale_y_init if length_scale_y_init is not None else length_scale_init)
        if self.anisotropic:
            self.raw_length_scale_x = nn.Parameter(torch.tensor(math.log(ls_x), dtype=dtype))
            self.raw_length_scale_y = nn.Parameter(torch.tensor(math.log(ls_y), dtype=dtype))
            # Keep an unused isotropic buffer for checkpoint compatibility with older loads.
            self.register_parameter(
                'raw_length_scale',
                nn.Parameter(torch.tensor(math.log(ls_x), dtype=dtype), requires_grad=False))
        else:
            self.raw_length_scale = nn.Parameter(torch.tensor(math.log(ls_x), dtype=dtype))

        if not learnable_length_scale:
            if self.anisotropic:
                self.raw_length_scale_x.requires_grad_(False)
                self.raw_length_scale_y.requires_grad_(False)
            else:
                self.raw_length_scale.requires_grad_(False)

        self.raw_noise_variance = nn.Parameter(
            torch.tensor(math.log(noise_variance_init), dtype=dtype),
            requires_grad=trainable_obs_variance
        )
        self.variational_inducing_loc = nn.Parameter(torch.zeros(num_inducing, dtype=dtype))
        self.raw_variational_inducing_scale = nn.Parameter(torch.eye(num_inducing, dtype=dtype))

    @property
    def amplitude(self):
        return torch.exp(self.raw_amplitude)

    @property
    def length_scale(self):
        """Isotropic length scale (or geometric mean if anisotropic)."""
        if self.anisotropic:
            return torch.sqrt(torch.exp(self.raw_length_scale_x) * torch.exp(self.raw_length_scale_y))
        return torch.exp(self.raw_length_scale)

    @property
    def length_scale_x(self):
        if self.anisotropic:
            return torch.exp(self.raw_length_scale_x)
        return torch.exp(self.raw_length_scale)

    @property
    def length_scale_y(self):
        if self.anisotropic:
            return torch.exp(self.raw_length_scale_y)
        return torch.exp(self.raw_length_scale)

    @property
    def observation_noise_variance(self):
        return torch.exp(self.raw_noise_variance)

    def variational_scale_tril(self):
        lower = torch.tril(self.raw_variational_inducing_scale, diagonal=-1)
        diag = F.softplus(torch.diagonal(self.raw_variational_inducing_scale)) + self.jitter
        return lower + torch.diag(diag)

    def _pairwise_dist_sq(self, xa, xb):
        if self.anisotropic:
            lx = self.length_scale_x.clamp_min(1.0e-12)
            ly = self.length_scale_y.clamp_min(1.0e-12)
            dx = (xa[:, 0:1] - xb[:, 0:1].T) / lx
            dy = (xa[:, 1:2] - xb[:, 1:2].T) / ly
            return dx.square() + dy.square()
        xa_sq = (xa ** 2).sum(dim=1, keepdim=True)
        xb_sq = (xb ** 2).sum(dim=1, keepdim=True).T
        dist_sq = xa_sq + xb_sq - 2.0 * xa @ xb.T
        ls = self.length_scale.clamp_min(1.0e-12)
        return dist_sq / ls.square()

    def kernel(self, xa, xb):
        # dist_sq is already scaled by length-scale(s) (Mahalanobis / isotropic).
        dist_sq = self._pairwise_dist_sq(xa, xb).clamp_min(0.0)
        amp2 = self.amplitude.square()
        kind = self.kernel_type
        if kind in ('rbf', 'sqexp', 'squared_exponential'):
            return amp2 * torch.exp(-0.5 * dist_sq)
        r = torch.sqrt(dist_sq + 1.0e-18)
        if kind in ('matern12', 'matern_12', 'exponential'):
            return amp2 * torch.exp(-r)
        if kind in ('matern32', 'matern_32'):
            sqrt3 = math.sqrt(3.0)
            return amp2 * (1.0 + sqrt3 * r) * torch.exp(-sqrt3 * r)
        if kind in ('matern52', 'matern_52'):
            sqrt5 = math.sqrt(5.0)
            return amp2 * (1.0 + sqrt5 * r + (5.0 / 3.0) * dist_sq) * torch.exp(-sqrt5 * r)
        raise ValueError(
            f"Unknown kernel_type={kind!r}; use 'rbf', 'matern12', 'matern32', or 'matern52'")

    def kernel_diagnostics(self):
        """Scalar kernel hyperparameters for logging (Python floats)."""
        return {
            'kernel_type': self.kernel_type,
            'anisotropic': float(self.anisotropic),
            'amplitude': float(self.amplitude.detach().cpu().item()),
            'length_scale': float(self.length_scale.detach().cpu().item()),
            'length_scale_x': float(self.length_scale_x.detach().cpu().item()),
            'length_scale_y': float(self.length_scale_y.detach().cpu().item()),
            'num_inducing': int(self.inducing_index_points.shape[0]),
        }

    def _kzz_chol(self):
        z = self.inducing_index_points
        eye = torch.eye(z.shape[0], dtype=z.dtype, device=z.device)
        return torch.linalg.cholesky(self.kernel(z, z) + (self.jitter * eye))

    def _solve_kzz(self, rhs):
        chol = self._kzz_chol()
        return torch.cholesky_solve(rhs, chol), chol

    def posterior_stats(self, index_points):
        kxz = self.kernel(index_points, self.inducing_index_points)
        kzz_inv_m, chol = self._solve_kzz(self.variational_inducing_loc.unsqueeze(1))
        mean = (kxz @ kzz_inv_m).squeeze(1)

        eye = torch.eye(self.inducing_index_points.shape[0], dtype=index_points.dtype, device=index_points.device)
        kzz_inv, _ = self._solve_kzz(eye)
        A = kxz @ kzz_inv

        prior_diag = torch.ones(
            (index_points.shape[0],),
            dtype=index_points.dtype,
            device=index_points.device
        ) * self.amplitude.square()
        proj_diag = torch.sum(A * kxz, dim=1)
        S = self.variational_scale_tril()
        S = S @ S.T
        extra_diag = torch.sum((A @ S) * A, dim=1)
        variance = torch.clamp(prior_diag - proj_diag + extra_diag, min=self.jitter)
        return mean, variance, A, S, chol

    def mean(self, index_points):
        return self.posterior_stats(index_points)[0]

    def stddev(self, index_points):
        return torch.sqrt(self.posterior_stats(index_points)[1])

    def sample(self, num_samples, index_points):
        mean, variance, A, _, _ = self.posterior_stats(index_points)
        scale_tril = self.variational_scale_tril()

        eps_u = torch.randn(
            num_samples, scale_tril.shape[0],
            dtype=index_points.dtype,
            device=index_points.device
        )
        u_samples = self.variational_inducing_loc.unsqueeze(0) + eps_u @ scale_tril.T
        correlated = u_samples @ A.T

        # This preserves the posterior marginal variance but treats the
        # conditional residual as diagonal for prediction-time sampling.
        eps_f = torch.randn(
            num_samples, index_points.shape[0],
            dtype=index_points.dtype,
            device=index_points.device
        )
        diag_residual = torch.clamp(variance - torch.sum((A @ scale_tril) ** 2, dim=1), min=self.jitter)
        return mean.unsqueeze(0) + correlated - (A @ self.variational_inducing_loc.unsqueeze(1)).T + eps_f * torch.sqrt(diag_residual).unsqueeze(0)

    def kl_divergence(self):
        mean = self.variational_inducing_loc.unsqueeze(1)
        scale_tril = self.variational_scale_tril()
        S = scale_tril @ scale_tril.T
        eye = torch.eye(self.inducing_index_points.shape[0], dtype=mean.dtype, device=mean.device)
        kzz_inv, chol = self._solve_kzz(eye)

        trace_term = torch.sum(kzz_inv * S)
        quad_term = (mean.T @ kzz_inv @ mean).squeeze()
        logdet_p = 2.0 * torch.sum(torch.log(torch.diagonal(chol)))
        logdet_q = 2.0 * torch.sum(torch.log(torch.diagonal(scale_tril)))
        m = mean.shape[0]
        return 0.5 * (trace_term + quad_term - m + logdet_p - logdet_q)

    @staticmethod
    def _inverse_softplus(y):
        """
        Inverse of softplus for positive y.
        softplus(x) = log(1 + exp(x)).
        """
        eps = torch.finfo(y.dtype).eps
        y = y.clamp_min(eps)
        return torch.where(y > 20.0, y, torch.log(torch.expm1(y)))

    def initialize_variational_to_prior(self, variance_scale=1.0):
        """
        Initialize q(u) close to p(u).

        variance_scale = 1.0 gives:
            q(u) = N(0, Kzz)
        so the initial inducing KL should be approximately zero.

        variance_scale < 1.0 gives a narrower initial q.
        variance_scale > 1.0 gives a broader initial q.
        """
        if variance_scale <= 0.0:
            raise ValueError("variance_scale must be positive")

        with torch.no_grad():
            # This is chol(Kzz + jitter * I), using the current amplitude
            # and length_scale parameters.
            prior_chol = self._kzz_chol()

            # Optional scaling: S_q = variance_scale * Kzz.
            q_chol = math.sqrt(float(variance_scale)) * prior_chol

            raw = torch.tril(q_chol).clone()

            # variational_scale_tril() computes:
            # diag = softplus(raw_diag) + jitter
            # so invert that transformation.
            diag_target = torch.diagonal(q_chol) - self.jitter
            diag_target = diag_target.clamp_min(torch.finfo(q_chol.dtype).eps)

            idx = torch.arange(raw.shape[0], device=raw.device)
            raw[idx, idx] = self._inverse_softplus(diag_target)

            self.variational_inducing_loc.zero_()
            self.raw_variational_inducing_scale.copy_(raw)


def make_sparse_vgp(x_ref, y_ref, norms, pars, field, dtype):
    """Build SparseVariationalGP from config; used by VI-only (+ predict) paths."""
    if field == 'eta':
        amp = pars.prior.std_eta
        ls_m = pars.prior.l_scale_eta
        ls_y_m = getattr(pars.prior, 'l_scale_eta_y', None)
    else:
        amp = pars.prior.std_lambda
        ls_m = pars.prior.l_scale_lambda
        ls_y_m = getattr(pars.prior, 'l_scale_lambda_y', None)

    def _norm_ls(length_m):
        dx = float(norms['x'].denom)
        dy = float(norms['y'].denom)
        domain = math.sqrt(dx * dy)
        return 2.0 * float(length_m) / domain

    ls = _norm_ls(ls_m)
    ls_y = _norm_ls(ls_y_m) if ls_y_m is not None else None
    return SparseVariationalGP(
        x_ref, y_ref,
        pars.prior.num_inducing_x, pars.prior.num_inducing_y, norms,
        trainable_obs_variance=pars.likelihood.trainable_obs_variance,
        amplitude_init=amp,
        length_scale_init=ls,
        length_scale_y_init=ls_y,
        dtype=dtype,
        inducing_placement=getattr(pars.prior, 'inducing_placement', 'ice_fps'),
        kernel_type=getattr(pars.prior, 'kernel_type', 'rbf'),
        anisotropic=bool(getattr(pars.prior, 'anisotropic', False)),
        learnable_length_scale=bool(getattr(pars.prior, 'learnable_length_scale', True)),
    )


class VariationalNaturalGradient:
    """Approximate natural-gradient updates for sparse-GP variational mean.

    Uses the current variational covariance S as a Fisher preconditioner:
        m <- m - lr * (S @ grad_m)
    Kernel / Cholesky scale parameters are left to a standard optimizer (AdamW).
    """

    def __init__(self, vgp_modules, learning_rate=1.0e-2):
        self.vgps = list(vgp_modules)
        self.lr = float(learning_rate)

    def zero_grad(self):
        for vgp in self.vgps:
            if vgp.variational_inducing_loc.grad is not None:
                vgp.variational_inducing_loc.grad = None

    @torch.no_grad()
    def step(self):
        for vgp in self.vgps:
            g = vgp.variational_inducing_loc.grad
            if g is None:
                continue
            scale_tril = vgp.variational_scale_tril()
            S = scale_tril @ scale_tril.T
            vgp.variational_inducing_loc.add_(-self.lr * (S @ g))


class JointModel(nn.Module):

    def __init__(self, mean_net, vgp_eta, vgp_lambda, dtype=None, mean_net_ref=None):
        super().__init__()
        self.mean_net = mean_net
        self.vgp_eta = vgp_eta
        self.vgp_lambda = vgp_lambda
        # Frozen pretrained MeanNetwork used as a soft state anchor.
        # Stored outside the module tree so it is not saved/loaded in state_dict
        # (predict builds JointModel without a ref copy).
        object.__setattr__(self, 'mean_net_ref', None)
        self.set_mean_net_ref(mean_net_ref)
        if dtype is None:
            dtype = next(mean_net.parameters()).dtype

        # Global, trainable offsets for the absolute latent fields.
        # They are intentionally not spatial reference fields; they only give
        # the dimensional log-viscosity and bounded sliding logit a learnable
        # global intercept so the GP does not have to pay KL cost just to move
        # the whole field away from zero.
        self.eta_log_shift = nn.Parameter(torch.tensor(0.0, dtype=dtype))
        self.lambda_logit_shift = nn.Parameter(torch.tensor(0.0, dtype=dtype))

    def set_mean_net_ref(self, mean_net_ref):
        """Attach a frozen PINN anchor without registering it as a submodule."""
        if mean_net_ref is not None:
            mean_net_ref.eval()
            for param in mean_net_ref.parameters():
                param.requires_grad_(False)
        object.__setattr__(self, 'mean_net_ref', mean_net_ref)

    def _distance_prior_stats(self, xb, yb, length_scale, prior_std, torch_dtype):
        x = xb.squeeze(-1)
        y = yb.squeeze(-1)
        dist_sq = (x[:, None] - x[None, :]).square() + (y[:, None] - y[None, :]).square()
        cov = (prior_std ** 2) * torch.exp(-dist_sq / (2.0 * (length_scale ** 2)))
        eye = torch.eye(x.shape[0], dtype=torch_dtype, device=xb.device)
        cov = cov + (1.0e-6 * eye)
        mean = torch.zeros_like(x)
        return mean, cov

    def _posterior_batch_stats(self, vgp, index_points, torch_dtype):
        mean, _, A, S, _ = vgp.posterior_stats(index_points)
        kxx = vgp.kernel(index_points, index_points)
        kxz = vgp.kernel(index_points, vgp.inducing_index_points)
        cov = kxx - (A @ kxz.T) + (A @ S @ A.T)
        eye = torch.eye(cov.shape[0], dtype=torch_dtype, device=index_points.device)
        cov = 0.5 * (cov + cov.T) + (1.0e-6 * eye)
        return mean, cov

    def _gaussian_kl_divergence(self, mean_q, cov_q, mean_p, cov_p):
        chol_p = torch.linalg.cholesky(cov_p)
        chol_q = torch.linalg.cholesky(cov_q)
        diff = (mean_p - mean_q).unsqueeze(1)

        solve_cov_q = torch.cholesky_solve(cov_q, chol_p)
        trace_term = torch.trace(solve_cov_q)
        quad_term = (diff.T @ torch.cholesky_solve(diff, chol_p)).squeeze()
        logdet_p = 2.0 * torch.log(torch.diagonal(chol_p)).sum()
        logdet_q = 2.0 * torch.log(torch.diagonal(chol_q)).sum()
        dim = mean_q.shape[0]
        return 0.5 * (trace_term + quad_term - dim + logdet_p - logdet_q)

    @staticmethod
    def _init_debug_stats():
        names = ('eta_log', 'theta_eta', 'eta', 'lambda_logit', 'lambda', 'rux', 'rvy', 'rh')
        return {name: {'min': float('inf'), 'max': float('-inf')} for name in names}

    @staticmethod
    def _update_debug_stats(stats, name, value):
        value_detached = value.detach()
        current_min = float(value_detached.amin().item())
        current_max = float(value_detached.amax().item())
        stats[name]['min'] = min(stats[name]['min'], current_min)
        stats[name]['max'] = max(stats[name]['max'], current_max)

    @staticmethod
    def _resolve_physics_approximation(pars):
        approx = getattr(pars.train, 'physics_approximation', getattr(pars.train, 'physics', 'SIA'))
        approx = str(approx).strip().upper()
        if approx not in ('SIA', 'SSA'):
            raise ValueError(
                f"Unsupported train.physics_approximation={approx!r}. "
                "Use either 'SIA' or 'SSA'."
            )
        return approx

    def _physics_common(self, batch, pars, torch_dtype, return_debug=False):
        x = batch['x'].detach().clone().requires_grad_(True)
        y = batch['y'].detach().clone().requires_grad_(True)
        debug_stats = self._init_debug_stats() if return_debug else None

        u, v, s, H = self.mean_net(x, y, inverse_norm=True)
        # H = torch.clamp(H, min=float(getattr(pars.prior, 'thickness_min', 0.0)))
        bed = s - H

        X = torch.cat([x, y], dim=1)
        Xn = normalize_tensor(X, self.mean_net.iW_coord, self.mean_net.b_coord)
        eta_loc = self.vgp_eta.mean(Xn).unsqueeze(1)
        eta_scale = self.vgp_eta.stddev(Xn).unsqueeze(1)
        lambda_loc = self.vgp_lambda.mean(Xn).unsqueeze(1)
        lambda_scale = self.vgp_lambda.stddev(Xn).unsqueeze(1)

        # with initial spatial reference
        eta_init = float(getattr(pars.prior, 'eta_init', math.sqrt(pars.prior.eta_min * pars.prior.eta_max)))
        eta_init = min(max(eta_init, float(pars.prior.eta_min)), float(pars.prior.eta_max))
        eta_log_center = torch.tensor(math.log(eta_init), dtype=torch_dtype, device=X.device)
        eta_log_min = torch.tensor(math.log(float(pars.prior.eta_min)), dtype=torch_dtype, device=X.device)
        eta_log_max = torch.tensor(math.log(float(pars.prior.eta_max)), dtype=torch_dtype, device=X.device)

        lambda_min = float(getattr(pars.prior, 'lambda_min', 0.5))
        lambda_max = float(getattr(pars.prior, 'lambda_max', 0.5))
        lambda_init = float(getattr(pars.prior, 'lambda_init', 0.5))
        lambda_init = min(max(lambda_init, 1.0e-6), 1.0 - 1.0e-6)
        lambda_logit_center = torch.tensor(
            math.log(lambda_init / (1.0 - lambda_init)),
            dtype=torch_dtype,
            device=X.device,
        )

        return {
            'x': x,
            'y': y,
            'u': u,
            'v': v,
            's': s,
            'bed': bed,
            'H': H,
            'X': X,
            'Xn': Xn,
            'eta_loc': eta_loc,
            'eta_scale': eta_scale,
            'lambda_loc': lambda_loc,
            'lambda_scale': lambda_scale,
            'eta_log_center': eta_log_center,
            'eta_log_min': eta_log_min,
            'eta_log_max': eta_log_max,
            'lambda_log_min': lambda_min,
            'lambda_log_max': lambda_max,
            'lambda_logit_center': lambda_logit_center,
            'debug_stats': debug_stats,
        }
        
    def _physics_nll_sia(self, batch, grid, weights, pars, torch_dtype, return_debug=False):
        common = self._physics_common(batch, pars, torch_dtype, return_debug=return_debug)
        x = common['x']
        y = common['y']
        u = common['u']
        v = common['v']
        s = common['s']
        H = common['H']
        X = common['X']
        Xn = common['Xn']
        debug_stats = common['debug_stats']
        
        H_x = grad(H, x)
        H_y = grad(H, y)
        s_x = grad(s, x)
        s_y = grad(s, y)
        tdx = -917 * 9.80665 * H * s_x
        tdy = -917 * 9.80665 * H * s_y
        A = (H ** 2) * tdx / 3.0
        B = (H ** 2) * tdy / 3.0
        A_x = grad(A, x)
        B_y = grad(B, y)

        weighted_ll = torch.zeros((), dtype=torch_dtype, device=X.device)
        momentum_ll = torch.zeros((), dtype=torch_dtype, device=X.device)
        continuity_ll = torch.zeros((), dtype=torch_dtype, device=X.device)
        sqrt2 = math.sqrt(2.0)
        rx_std = torch.tensor(pars.likelihood.rx_std, dtype=torch_dtype, device=X.device)
        ry_std = torch.tensor(pars.likelihood.ry_std, dtype=torch_dtype, device=X.device)
        rh_std = torch.tensor(pars.likelihood.rh_std, dtype=torch_dtype, device=X.device)
        u_x = grad(u, x)
        v_y = grad(v, y)

        for i in range(grid.shape[0]):
            theta_eta = sqrt2 * common['eta_scale'] * grid[i] + common['eta_loc']
            eta_log = common['eta_log_center'] + self.eta_log_shift + theta_eta # if assume no spatial variation
            eta_log = eta_log.clamp(min=common['eta_log_min'], max=common['eta_log_max'])
            eta = torch.exp(eta_log)
            # eta = eta_ref * torch.exp(theta_eta)
            # eta = eta.clamp(min=pars.prior.eta_min, max=pars.prior.eta_max)
            if return_debug:
                self._update_debug_stats(debug_stats, 'theta_eta', theta_eta)
                self._update_debug_stats(debug_stats, 'eta_log', eta_log)
                self._update_debug_stats(debug_stats, 'eta', eta)
            
            for j in range(grid.shape[0]):
                theta_lambda = sqrt2 * common['lambda_scale'] * grid[j] + common['lambda_loc']
                lambda_logit = common['lambda_logit_center'] + self.lambda_logit_shift + theta_lambda
                lam = torch.sigmoid(lambda_logit)
                # lam = lam.clamp(min=common['lambda_log_min'], max=common['lambda_log_max'])
                lam_x = grad(lam, x)
                lam_y = grad(lam, y)
                eta_x = grad(eta, x)
                eta_y = grad(eta, y)
                if return_debug:
                    self._update_debug_stats(debug_stats, 'lambda_logit', lambda_logit)
                    self._update_debug_stats(debug_stats, 'lambda', lam)

                rux = (1.0 - lam) * u - H * tdx / (2.0 * eta)
                rvy = (1.0 - lam) * v - H * tdy / (2.0 * eta)

                rh = H_x * lam * u + H * (lam_x * u + lam * u_x)
                rh = rh + (A_x / eta) - (A * eta_x / eta.square())
                rh = rh + H_y * lam * v + H * (lam_y * v + lam * v_y)
                rh = rh + (B_y / eta) - (B * eta_y / eta.square())
                if return_debug:
                    self._update_debug_stats(debug_stats, 'rux', rux)
                    self._update_debug_stats(debug_stats, 'rvy', rvy)
                    self._update_debug_stats(debug_stats, 'rh', rh)

                log_prob = normal_log_prob(rux, rx_std).mean()
                log_prob = log_prob + normal_log_prob(rvy, ry_std).mean()
                momentum_ll = momentum_ll + weights[i] * weights[j] * log_prob
                cont_prob = normal_log_prob(rh, rh_std).mean()
                continuity_ll = continuity_ll + weights[i] * weights[j] * cont_prob
                weighted_ll = weighted_ll + weights[i] * weights[j] * (log_prob + cont_prob)

        if return_debug:
            debug_stats['loss_components'] = {
                'momentum_nll': float((-momentum_ll).detach().item()),
                'continuity_nll': float((-continuity_ll).detach().item()),
            }
            return -weighted_ll, Xn, debug_stats
        return -weighted_ll, Xn

    def _physics_nll_ssa(self, batch, grid, weights, pars, torch_dtype, return_debug=False):
        common = self._physics_common(batch, pars, torch_dtype, return_debug=return_debug)
        x = common['x']
        y = common['y']
        u = common['u']
        v = common['v']
        s = common['s']
        H = common['H']
        X = common['X']
        Xn = common['Xn']
        debug_stats = common['debug_stats']

        s_x = grad(s, x)
        s_y = grad(s, y)
        u_x = grad(u, x)
        u_y = grad(u, y)
        v_x = grad(v, x)
        v_y = grad(v, y)

        # CHANGED: icepack constants (m, yr, MPa) instead of SI Pa·m/s².
        icepack = icepack_ssa_constants(pars, torch_dtype, X.device)
        rho_ice = icepack['rho_ice']
        rho_water = icepack['rho_water']
        gravity = icepack['g']
        glen_n = icepack['glen_n']
        weertman_m = icepack['weertman_m']
        eps_min = icepack['eps_min']
        fluidity_A = icepack['fluidity_A']
        friction_C = icepack['friction_C']

        # CHANGED: driving stress matches icepack IceStream.gravity (rho_I * g * h * grad(s)).
        tau_dx = rho_ice * gravity * H * s_x
        tau_dy = rho_ice * gravity * H * s_y

        # CHANGED: effective pressure and yield stress for spin-up plastic basal law.
        water_depth = torch.clamp(-(s - H), min=0.0)
        p_water = rho_water * gravity * water_depth
        p_ice = rho_ice * gravity * H
        effective_pressure = torch.clamp(p_ice - p_water, min=0.0)
        tau_c = 0.5 * effective_pressure

        # CHANGED: Glen effective strain rate (yr^-1) and viscosity scale (MPa·yr).
        eps_eff = glen_effective_strain_rate(u_x, u_y, v_x, v_y, eps_min)
        mu_glen = glen_effective_viscosity(eps_eff, fluidity_A, glen_n)
        use_inferred_eta = bool(getattr(pars.prior, 'ssa_use_inferred_eta', True))

        weighted_ll = torch.zeros((), dtype=torch_dtype, device=X.device)
        momentum_ll = torch.zeros((), dtype=torch_dtype, device=X.device)
        continuity_ll = torch.zeros((), dtype=torch_dtype, device=X.device)
        sqrt2 = math.sqrt(2.0)
        rx_std = torch.tensor(
            getattr(pars.likelihood, 'ssa_rx_std', pars.likelihood.rx_std),
            dtype=torch_dtype,
            device=X.device,
        )
        ry_std = torch.tensor(
            getattr(pars.likelihood, 'ssa_ry_std', pars.likelihood.ry_std),
            dtype=torch_dtype,
            device=X.device,
        )
        rh_std = torch.tensor(
            getattr(pars.likelihood, 'ssa_rh_std', pars.likelihood.rh_std),
            dtype=torch_dtype,
            device=X.device,
        )
        speed_eps = torch.tensor(getattr(pars.prior, 'speed_epsilon', 1.0), dtype=torch_dtype, device=X.device)
        speed = torch.sqrt(u.square() + v.square() + speed_eps.square())
        enforce_continuity = bool(getattr(pars.prior, 'ssa_enforce_continuity', False))

        # CHANGED: continuity residual optional (off by default for icepack diagnostic SSA).
        if enforce_continuity:
            H_x = grad(H, x)
            H_y = grad(H, y)
            rh = H_x * u + H * u_x + H_y * v + H * v_y
        else:
            rh = None

        for i in range(grid.shape[0]):
            theta_eta = sqrt2 * common['eta_scale'] * grid[i] + common['eta_loc']
            eta_log = common['eta_log_center'] + self.eta_log_shift + theta_eta
            eta_log = eta_log.clamp(min=common['eta_log_min'], max=common['eta_log_max'])
            eta = torch.exp(eta_log)
            if return_debug:
                self._update_debug_stats(debug_stats, 'theta_eta', theta_eta)
                self._update_debug_stats(debug_stats, 'eta_log', eta_log)
                self._update_debug_stats(debug_stats, 'eta', eta)

            # CHANGED: Glen membrane stress M = 2 μ (ε + tr(ε) I), μ in MPa·yr.
            # Default: μ = inferred η (inverse-problem effective viscosity).
            # Set prior.ssa_use_inferred_eta = False to use forward-model μ_Glen(A, ε).
            mu = eta if use_inferred_eta else mu_glen
            membrane_xx = 2.0 * H * mu * (2.0 * u_x + v_y)
            membrane_xy = H * mu * (u_y + v_x)
            membrane_yy = 2.0 * H * mu * (u_x + 2.0 * v_y)
            membrane_div_x = grad(membrane_xx, x) + grad(membrane_xy, y)
            membrane_div_y = grad(membrane_xy, x) + grad(membrane_yy, y)

            # CHANGED: spin-up plastic basal drag with fixed C (λ GP no longer used in SSA).
            basal_drag_x, basal_drag_y = spinup_plastic_basal_drag(
                u, v, speed, tau_c, friction_C, weertman_m, speed_eps
            )
            rux = membrane_div_x + tau_dx - basal_drag_x
            rvy = membrane_div_y + tau_dy - basal_drag_y
            if return_debug:
                self._update_debug_stats(debug_stats, 'rux', rux)
                self._update_debug_stats(debug_stats, 'rvy', rvy)
                if rh is not None:
                    self._update_debug_stats(debug_stats, 'rh', rh)

            log_prob = normal_log_prob(rux, rx_std).mean()
            log_prob = log_prob + normal_log_prob(rvy, ry_std).mean()
            momentum_ll = momentum_ll + weights[i] * log_prob
            weighted_ll = weighted_ll + weights[i] * log_prob
            if rh is not None:
                cont_prob = normal_log_prob(rh, rh_std).mean()
                continuity_ll = continuity_ll + weights[i] * cont_prob
                weighted_ll = weighted_ll + weights[i] * cont_prob

        if return_debug:
            debug_stats['loss_components'] = {
                'momentum_nll': float((-momentum_ll).detach().item()),
                'continuity_nll': float((-continuity_ll).detach().item()) if enforce_continuity else float('nan'),
            }
            return -weighted_ll, Xn, debug_stats
        return -weighted_ll, Xn

    def _physics_nll(self, batch, grid, weights, pars, torch_dtype, return_debug=False):
        approx = self._resolve_physics_approximation(pars)
        if approx == 'SIA':
            return self._physics_nll_sia(batch, grid, weights, pars, torch_dtype, return_debug=return_debug)
        if approx == 'SSA':
            return self._physics_nll_ssa(batch, grid, weights, pars, torch_dtype, return_debug=return_debug)
        raise AssertionError(f'Unhandled physics approximation: {approx}')

    
    def forward(self, batch_obs, batch_phys, grid, weights, pars, torch_dtype, return_debug=False):
        # up, vp, sp, hp = self.mean_net(
        #     batch_obs['x'], batch_obs['y'], inverse_norm=False
        # )
        mean_net_trainable = any(param.requires_grad for param in self.mean_net.parameters())
        if mean_net_trainable:
            up, vp, sp, hp = self.mean_net(
                batch_obs['x'], batch_obs['y'],
                batch_obs['u_in'], batch_obs['v_in'], batch_obs['s_in'], batch_obs['b_in'],
                batch_obs['uv_mask'],
                inverse_norm=False)
        else:
            # During the frozen-mean stage, the observation term is a monitor
            # only; it cannot update any parameter.  Avoid building this graph.
            with torch.no_grad():
                up, vp, sp, hp = self.mean_net(
                    batch_obs['x'], batch_obs['y'],
                    batch_obs['u_in'], batch_obs['v_in'], batch_obs['s_in'], batch_obs['b_in'],
                    batch_obs['uv_mask'],
                    inverse_norm=False)
        u_term = batch_obs['uv_mask'] * (up - batch_obs['u']).square() / batch_obs['u_err'].square()
        v_term = batch_obs['uv_mask'] * (vp - batch_obs['v']).square() / batch_obs['v_err'].square()
        s_term = (sp - batch_obs['s']).square() / batch_obs['s_err'].square()
        h_term = (hp - batch_obs['h']).square() / batch_obs['h_err'].square()
        obs_weight = 2.0 * batch_obs['uv_mask'].sum() + 2.0 * batch_obs['geom_mask'].sum()
        data_nll = (u_term + v_term + s_term + h_term).sum() / torch.clamp(obs_weight, min=1.0)

        # Soft anchor toward the pretrained PINN state. Penalizes drift in
        # (u,v,s,H) so PDE residuals must be explained by η rather than by
        # freely retuning the mean field.
        state_reg = torch.zeros((), dtype=torch_dtype, device=batch_obs['x'].device)
        state_reg_scale = float(getattr(pars.train, 'state_reg_scale', 0.0) or 0.0)
        if self.mean_net_ref is not None and state_reg_scale > 0.0:
            with torch.no_grad():
                ur, vr, sr, hr = self.mean_net_ref(
                    batch_obs['x'], batch_obs['y'],
                    batch_obs['u_in'], batch_obs['v_in'], batch_obs['s_in'], batch_obs['b_in'],
                    batch_obs['uv_mask'],
                    inverse_norm=False)
            state_u = batch_obs['uv_mask'] * (up - ur).square()
            state_v = batch_obs['uv_mask'] * (vp - vr).square()
            state_s = (sp - sr).square()
            state_h = (hp - hr).square()
            state_reg = (state_u + state_v + state_s + state_h).sum() / torch.clamp(obs_weight, min=1.0)

        physics_out = self._physics_nll(
            batch_phys, grid, weights, pars, torch_dtype, return_debug=return_debug
        )
        if return_debug:
            physics_nll, Xn, debug_stats = physics_out
        else:
            physics_nll, Xn = physics_out

        
        # eta_mean, eta_cov = self._posterior_batch_stats(self.vgp_eta, Xn, torch_dtype)
        # lambda_mean, lambda_cov = self._posterior_batch_stats(self.vgp_lambda, Xn, torch_dtype)
        # zero_eta, prior_eta_cov = self._distance_prior_stats(
        #     batch_phys['x'], batch_phys['y'],
        #     pars.prior.l_scale_eta, pars.prior.std_eta, torch_dtype)
        # zero_lambda, prior_lambda_cov = self._distance_prior_stats(
        #     batch_phys['x'], batch_phys['y'],
        #     pars.prior.l_scale_lambda, pars.prior.std_lambda, torch_dtype)
        # kl_eta = self._gaussian_kl_divergence(eta_mean, eta_cov, zero_eta, prior_eta_cov)
        # kl_lambda = self._gaussian_kl_divergence(lambda_mean, lambda_cov, zero_lambda, prior_lambda_cov)
        
        if getattr(pars.train, 'use_inducing_kl', True):
            # Memory-safe sparse-GP KL.  This avoids constructing dense
            # physics_batch_size x physics_batch_size posterior/prior covariance
            # matrices and Cholesky factors every minibatch.
            kl_eta = self.vgp_eta.kl_divergence()
            kl_lambda = self.vgp_lambda.kl_divergence()
        else:
            # Original batchwise posterior-vs-prior KL.  This is O(B^2) memory
            # and O(B^3) Cholesky work in physics_batch_size B.
            eta_mean, eta_cov = self._posterior_batch_stats(self.vgp_eta, Xn, torch_dtype)
            lambda_mean, lambda_cov = self._posterior_batch_stats(self.vgp_lambda, Xn, torch_dtype)

            zero_eta, prior_eta_cov = self._distance_prior_stats(
                batch_phys['x'], batch_phys['y'],
                pars.prior.l_scale_eta, pars.prior.std_eta, torch_dtype)
            zero_lambda, prior_lambda_cov = self._distance_prior_stats(
                batch_phys['x'], batch_phys['y'],
                pars.prior.l_scale_lambda, pars.prior.std_lambda, torch_dtype)
            kl_eta = self._gaussian_kl_divergence(eta_mean, eta_cov, zero_eta, prior_eta_cov)
            kl_lambda = self._gaussian_kl_divergence(lambda_mean, lambda_cov, zero_lambda, prior_lambda_cov)
        batch_n = torch.tensor(batch_phys['x'].shape[0], device=batch_phys['x'].device, dtype=torch_dtype)
        kl_eta_scale = torch.tensor(pars.prior.kl_eta, dtype=torch_dtype, device=batch_phys['x'].device) / batch_n.clamp(min=1.0)    
        kl_lambda_scale = torch.tensor(pars.prior.kl_lambda, dtype=torch_dtype, device=batch_phys['x'].device) / batch_n.clamp(min=1.0)    
        kl_value = kl_eta_scale * kl_eta + kl_lambda_scale * kl_lambda

        # Soft Gaussian anchor of log η toward log(eta_init).  Prevents the
        # physics term from collapsing η while mean_net is frozen (observed
        # when tight ssa_*_std alone preferred η → 0).
        eta_prior_scale = float(getattr(pars.train, 'eta_prior_scale', 0.0) or 0.0)
        eta_prior_std = float(getattr(pars.train, 'eta_prior_std', 1.0) or 1.0)
        eta_prior_reg = torch.zeros((), dtype=torch_dtype, device=batch_phys['x'].device)
        if eta_prior_scale > 0.0:
            eta_loc = self.vgp_eta.mean(Xn)
            # eta = exp(log(eta_init) + eta_log_shift + theta); pull offset to 0.
            log_eta_offset = self.eta_log_shift + eta_loc
            eta_prior_reg = 0.5 * torch.mean(
                (log_eta_offset / max(eta_prior_std, 1.0e-6)) ** 2)

        data_scale = torch.tensor(
            float(getattr(pars.train, 'data_scale', 1.0) or 1.0),
            dtype=torch_dtype,
            device=batch_phys['x'].device,
        )
        phys_scale = torch.tensor(
            float(getattr(pars.train, 'phys_scale', 1.0) or 1.0),
            dtype=torch_dtype,
            device=batch_phys['x'].device,
        )
        state_reg_scale_t = torch.tensor(
            state_reg_scale, dtype=torch_dtype, device=batch_phys['x'].device)
        eta_prior_scale_t = torch.tensor(
            eta_prior_scale, dtype=torch_dtype, device=batch_phys['x'].device)
        # Fold prior into the KL slot so the existing 4-term train loop is unchanged.
        kl_and_prior = kl_value + eta_prior_scale_t * eta_prior_reg
        if return_debug and isinstance(debug_stats, dict) and 'loss_components' in debug_stats:
            scale = float(phys_scale.item())
            debug_stats['loss_components'] = {
                key: float('nan') if not math.isfinite(value) else scale * float(value)
                for key, value in debug_stats['loss_components'].items()
            }
            debug_stats['loss_components']['state_reg'] = float(
                (state_reg_scale_t * state_reg).detach().item())
            debug_stats['loss_components']['eta_prior'] = float(
                (eta_prior_scale_t * eta_prior_reg).detach().item())
            debug_stats['loss_components']['kl_only'] = float(kl_value.detach().item())
        outputs = (
            data_scale * data_nll,
            phys_scale * physics_nll,
            kl_and_prior,
            state_reg_scale_t * state_reg,
        )
        if return_debug:
            return outputs + (debug_stats,)
        return outputs


def create_optimizer(optname, parameters, learning_rate=0.0005, **kwargs):
    name = str(optname or 'adam').strip().lower()
    # Callers may pass LBFGS-only kwargs unconditionally; strip for other opts.
    lbfgs_only = {
        'max_iter': kwargs.pop('max_iter', None),
        'history_size': kwargs.pop('history_size', None),
        'line_search_fn': kwargs.pop('line_search_fn', None),
    }
    if name == 'adam':
        return torch.optim.Adam(parameters, lr=learning_rate, **kwargs)
    if name == 'adamw':
        return torch.optim.AdamW(parameters, lr=learning_rate, **kwargs)
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.8, **kwargs)
    if name in ('lbfgs', 'l-bfgs'):
        return torch.optim.LBFGS(
            parameters,
            lr=learning_rate,
            max_iter=int(lbfgs_only['max_iter'] if lbfgs_only['max_iter'] is not None else 20),
            history_size=int(
                lbfgs_only['history_size'] if lbfgs_only['history_size'] is not None else 10),
            line_search_fn=(
                lbfgs_only['line_search_fn']
                if lbfgs_only['line_search_fn'] is not None else 'strong_wolfe'),
            **kwargs,
        )
    raise NotImplementedError(f'Unsupported optimizer: {optname!r}')
