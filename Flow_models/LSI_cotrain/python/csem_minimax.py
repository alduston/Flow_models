from __future__ import annotations
from torch._higher_order_ops import out_dtype
import math
import os
import random
import numpy as np
from typing import Any, Dict, Tuple

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision import transforms, utils as tv_utils

import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import pandas as pd


# ---------------------------------------------------------------------------
# Dataset Configuration
# ---------------------------------------------------------------------------
DATASET_INFO = {
    "MNIST": {"class": torchvision.datasets.MNIST, "num_classes": 10, "img_size": 28, "img_channels": 1},
    "FMNIST": {"class": torchvision.datasets.FashionMNIST, "num_classes": 10, "img_size": 28, "img_channels": 1},
    "EMNIST": {"class": torchvision.datasets.EMNIST, "num_classes": 47, "split": "balanced", "img_size": 28, "img_channels": 1},
    "KMNIST": {"class": torchvision.datasets.KMNIST, "num_classes": 10, "img_size": 28, "img_channels": 1},
    "GCIFAR": {"class": torchvision.datasets.CIFAR10, "num_classes": 10, "img_size": 32, "img_channels": 1, "grayscale": True},
    "CIFAR": {"class": torchvision.datasets.CIFAR10, "num_classes": 10, "img_size": 32, "img_channels": 3},
}


# ---------------------------------------------------------------------------
# Imports & Checks
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs): return iterable

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available, perceptual loss/metrics will be skipped")

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics not available, FID will be -1")

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)

def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    ensure_parent(path)
    torch.save(state, path)

import os
import shutil
from datetime import datetime

def zip_results_dir(results_dir: str, zip_path: str | None = None) -> str:
    """
    Zips the entire results_dir into a .zip file.
    Returns the path to the created zip.
    """
    results_dir = os.path.abspath(results_dir)

    if zip_path is None:
        parent = os.path.dirname(results_dir)
        base = os.path.basename(results_dir.rstrip(os.sep))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = os.path.join(parent, f"{base}_{stamp}.zip")
    else:
        zip_path = os.path.abspath(zip_path)
        if not zip_path.endswith(".zip"):
            zip_path += ".zip"

    # shutil.make_archive wants a path without ".zip"
    zip_base = zip_path[:-4]

    # Create the zip. This includes everything under results_dir.
    shutil.make_archive(zip_base, "zip", root_dir=results_dir)

    return zip_path

def make_group_norm(num_channels: int, num_groups: int = 16) -> nn.GroupNorm:
    best = min(num_groups, num_channels)
    while num_channels % best != 0 and best > 1:
        best -= 1
    return nn.GroupNorm(best, num_channels)

# ---------------------------------------------------------------------------
# Math Utilities
# ---------------------------------------------------------------------------

def sample_log_uniform_times(B: int, t_min: float, t_max: float, device: torch.device) -> torch.Tensor:
    u = torch.rand(B, device=device)
    log_t_min = math.log(t_min)
    log_t_max = math.log(t_max)
    return torch.exp(log_t_min + u * (log_t_max - log_t_min))

def get_ou_params(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """OU process: alpha(t) = exp(-t), sigma(t) = sqrt(1 - exp(-2t))."""
    alpha = torch.exp(-t)
    sigma = torch.sqrt(1.0 - torch.exp(-2.0 * t) + 1e-8)
    return alpha, sigma


# ---------------------------------------------------------------------------
# Flow Matching (LightningDiT-style)
# ---------------------------------------------------------------------------

def sample_logit_normal_times(B: int, t_min: float, t_max: float, device: torch.device) -> torch.Tensor:
    """Logit-normal time sampling on (t_min, t_max), where 0 <= t_min < t_max <= 1.

    Sample u ~ N(0,1), set t01 = sigmoid(u), then affine-map to [t_min, t_max].
    This concentrates mass around mid-times while still covering endpoints.
    """
    t_min = float(t_min); t_max = float(t_max)
    if not (0.0 <= t_min < t_max <= 1.0):
        raise ValueError(f"flow matching requires 0 <= t_min < t_max <= 1, got t_min={t_min}, t_max={t_max}")
    u = torch.randn(B, device=device)
    t01 = torch.sigmoid(u)
    t = t_min + (t_max - t_min) * t01
    return t.clamp(min=t_min, max=t_max)

def get_flow_params(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linear interpolant params: z_t = alpha(t) * z0 + sigma(t) * eps, with alpha=t and sigma=1-t."""
    alpha = t
    sigma = (1.0 - t).clamp_min(1e-8)
    return alpha, sigma

# ---------------------------------------------------------------------------
# Log-SNR <-> OU time conversion
# ---------------------------------------------------------------------------

def ou_logsnr(t: torch.Tensor) -> torch.Tensor:
    """log-SNR for the OU process: log(alpha^2 / sigma^2)."""
    e2t = torch.exp(-2.0 * t)
    return torch.log(e2t / (1.0 - e2t + 1e-12) + 1e-12)

def ou_time_from_logsnr(lam: torch.Tensor) -> torch.Tensor:
    """Invert log-SNR to OU time.

    From lambda = log(e^{-2t} / (1 - e^{-2t})):
      e^{-2t} = sigmoid(lambda)
      t = -0.5 * log(sigmoid(lambda))
    """
    sig = torch.sigmoid(lam)
    return -0.5 * torch.log(sig.clamp(1e-12, 1.0 - 1e-7))

# ---------------------------------------------------------------------------
# VP-Cosine schedule helpers (Nichol & Dhariwal, Improved DDPM)
# ---------------------------------------------------------------------------

def get_cosine_params(
    t_frac: torch.Tensor, cosine_s: float = 0.008,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Given t_frac in (0, 1), return (alpha, sigma, beta) for the VP-cosine schedule.

    alpha_bar(t) = cos^2(g(t)) / cos^2(g(0)),  g(t) = (t+s)/(1+s) * pi/2
    alpha = sqrt(alpha_bar),  sigma = sqrt(1 - alpha_bar)
    beta(t) = -d/dt log(alpha_bar) = 2 * tan(g(t)) * g'(t)
    """
    g = ((t_frac + cosine_s) / (1.0 + cosine_s)) * (math.pi / 2.0)
    g0 = (cosine_s / (1.0 + cosine_s)) * (math.pi / 2.0)
    alpha_bar = (torch.cos(g) / math.cos(g0)) ** 2
    alpha_bar = alpha_bar.clamp(1e-8, 1.0)
    alpha = torch.sqrt(alpha_bar)
    sigma = torch.sqrt((1.0 - alpha_bar).clamp_min(1e-8))
    g_prime = math.pi / (2.0 * (1.0 + cosine_s))
    beta = (2.0 * torch.tan(g.clamp(max=math.pi / 2.0 - 1e-4)) * g_prime).clamp(max=20.0)
    return alpha, sigma, beta


# ---------------------------------------------------------------------------
# Discrete schedule utilities (cosine/linear)
# ---------------------------------------------------------------------------

def extract_schedule(a: torch.Tensor, t_idx: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """Extract 1D schedule values at indices t_idx and reshape for broadcast to x_shape."""
    # a: [T], t_idx: [B] long
    out = a.gather(0, t_idx).float()
    return out.view(-1, *([1] * (len(x_shape) - 1)))

def make_beta_schedule(
    schedule: str,
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    cosine_s: float = 0.008,
) -> torch.Tensor:
    """Return betas[t] for a discrete VP/DDPM schedule (legacy helper)."""
    schedule = str(schedule).lower()
    if num_timesteps < 1:
        raise ValueError("num_timesteps must be >= 1")
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        return betas.clamp(1e-8, 0.999)
    if schedule == "cosine":
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps, dtype=torch.float32) / num_timesteps
        alphas_cumprod = torch.cos(((t + cosine_s) / (1 + cosine_s)) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-8, 0.999)
    raise ValueError(f"Unknown noise_schedule: {schedule}")



def canonicalize_time_schedule(schedule: Any, default: str = "log_snr") -> str:
    """Normalize user-facing schedule names to the internal canonical names."""
    if schedule is None:
        schedule = default
    s = str(schedule).strip().lower()
    aliases = {
        "logt": "log_t",
        "log_t": "log_t",
        "log-uniform": "log_t",
        "log_uniform": "log_t",
        "loguniform": "log_t",
        "snr": "log_snr",
        "logsnr": "log_snr",
        "log_snr": "log_snr",
        "linear": "linear",
        "lin": "linear",
        "linear_t": "linear",
        "t_linear": "linear",
        "flowmatching": "flow_matching",
        "flow_matching": "flow_matching",
        "frontier": "frontier",
        "adaptive": "frontier",
    }
    return aliases.get(s, s)


def canonicalize_init_mode(init_mode: Any, default: str = "prior") -> str:
    """Normalize sampler initialization mode names."""
    if init_mode is None:
        init_mode = default
    s = str(init_mode).strip().lower()
    if s in ("prior", "gaussian", "gauss", "normal"):
        return "prior"
    if s in ("oracle", "orcale"):
        return "oracle"
    raise ValueError(f"Unknown init_mode={init_mode!r}. Expected 'prior' or 'oracle'.")


def canonicalize_cfg_mode(cfg_mode: Any, default: str = "constant") -> str:
    """Normalize classifier-free guidance schedule mode names."""
    if cfg_mode is None:
        cfg_mode = default
    s = str(cfg_mode).strip().lower()
    aliases = {
        "const": "constant",
        "constant": "constant",
        "linear-ramp": "linear_ramp",
        "linear_ramp": "linear_ramp",
        "linearramp": "linear_ramp",
        "ramp": "linear_ramp",
    }
    out = aliases.get(s, s)
    if out not in ("constant", "linear_ramp"):
        raise ValueError(f"Unknown cfg_mode={cfg_mode!r}. Expected 'constant' or 'linear_ramp'.")
    return out


def get_alpha_sigma_for_schedule(
    t: torch.Tensor,
    schedule_type: str = "log_snr",
    cosine_s: float = 0.008,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (alpha, sigma) for the requested schedule family at time t."""
    stype = canonicalize_time_schedule(schedule_type, default="log_snr")
    if stype == "frontier":
        raise ValueError("'frontier' is a discretization strategy, not a diffusion family.")
    if stype in ("flow", "flow_matching"):
        return get_flow_params(t)
    if stype == "cosine":
        a, s, _ = get_cosine_params(t, cosine_s=cosine_s)
        return a, s
    return get_ou_params(t)


def sample_forward_latent_from_z0(
    z0: torch.Tensor,
    t_value: float,
    schedule_type: str = "log_snr",
    cosine_s: float = 0.008,
    noise: torch.Tensor | None = None,
    generator=None,
) -> torch.Tensor:
    """Sample z_t ~ q(z_t | z_0) for the requested schedule family."""
    if noise is None:
        if generator is None:
            noise = torch.randn_like(z0)
        else:
            noise = torch.randn(z0.shape, device=z0.device, dtype=z0.dtype, generator=generator)
    t = torch.full((z0.shape[0], 1, 1, 1), float(t_value), device=z0.device, dtype=z0.dtype)
    alpha, sigma = get_alpha_sigma_for_schedule(t, schedule_type=schedule_type, cosine_s=cosine_s)
    return alpha * z0 + sigma * noise


def _format_sampler_float_tag(x: float) -> str:
    s = f"{float(x):.4g}"
    return s.replace("-", "m").replace(".", "p")


def make_schedule(cfg: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """Build a unified discrete schedule of T timesteps.

    Config key ``time_schedule`` selects the grid type:
      - "flow" / "flow_matching" : linear flow-matching interpolant on [0,1] (alpha=t, sigma=1-t)
      - "linear"                  : OU process, grid uniform in OU time
      - "log_t"                   : OU process, grid log-uniform in OU time  (legacy)
      - "log_snr"                 : OU process, grid uniform in log-SNR
      - "cosine"                  : VP-cosine process (Nichol & Dhariwal), grid uniform in [0,1]

    Returns dict with keys:
        T, times, alpha, sigma, beta, schedule_type, cosine_s
    """
    stype = canonicalize_time_schedule(cfg.get("time_schedule", "log_snr"), default="log_snr")
    T = int(cfg.get("num_train_timesteps", 1000))

    if stype in ("flow", "flow_matching"):
        t_min = float(cfg.get("t_min", 1e-5))
        t_max = float(cfg.get("t_max", 0.99999))
        if not (0.0 <= t_min < t_max <= 1.0):
            raise ValueError(f"flow matching requires 0 <= t_min < t_max <= 1, got t_min={t_min}, t_max={t_max}")

        times = torch.linspace(t_min, t_max, T, device=device, dtype=torch.float32)
        alpha = times.clone()
        sigma = (1.0 - times).clamp_min(1e-8)
        beta_arr = torch.zeros(T, device=device, dtype=torch.float32)

        return {
            "T": torch.tensor(T, device=device, dtype=torch.long),
            "times": times, "alpha": alpha, "sigma": sigma, "beta": beta_arr,
            "schedule_type": stype, "cosine_s": 0.0,
        }

    if stype in ("linear", "log_t", "log_snr"):
        t_min = float(cfg.get("t_min", 2e-5))
        t_max = float(cfg.get("t_max", 2.0))

        if stype == "linear":
            times = torch.linspace(t_min, t_max, T, device=device, dtype=torch.float32)
        elif stype == "log_t":
            times = torch.logspace(
                math.log10(t_min), math.log10(t_max), T,
                device=device, dtype=torch.float32,
            )
        else:
            lam_max = ou_logsnr(torch.tensor(t_min, dtype=torch.float64)).item()
            lam_min = ou_logsnr(torch.tensor(t_max, dtype=torch.float64)).item()
            lam_grid = torch.linspace(lam_min, lam_max, T, device=device, dtype=torch.float64)
            times = ou_time_from_logsnr(lam_grid).float()
            times = times.flip(0).clamp(t_min, t_max)

        a, s = get_ou_params(times.view(T, 1, 1, 1))
        alpha = a.view(T).float()
        sigma = s.view(T).float()
        beta = torch.zeros(T, device=device, dtype=torch.float32)

        return {
            "T": torch.tensor(T, device=device, dtype=torch.long),
            "times": times, "alpha": alpha, "sigma": sigma, "beta": beta,
            "schedule_type": stype, "cosine_s": 0.0,
        }

    if stype == "cosine":
        cosine_s = float(cfg.get("cosine_s", 0.008))
        t_min_frac = float(cfg.get("cosine_t_min", 2e-4))
        t_max_frac = float(cfg.get("cosine_t_max", 0.9999))

        times = torch.linspace(t_min_frac, t_max_frac, T, device=device, dtype=torch.float32)
        a, s, b = get_cosine_params(times.view(T, 1, 1, 1), cosine_s=cosine_s)
        alpha = a.view(T).float()
        sigma = s.view(T).float()
        beta_arr = b.view(T).float()

        return {
            "T": torch.tensor(T, device=device, dtype=torch.long),
            "times": times, "alpha": alpha, "sigma": sigma, "beta": beta_arr,
            "schedule_type": stype, "cosine_s": cosine_s,
        }

    raise ValueError(f"Unknown time_schedule: {stype!r}. Expected 'flow', 'flow_matching', 'linear', 'log_t', 'log_snr', or 'cosine'.")


# Keep legacy alias
def make_ou_schedule(cfg: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    cfg_copy = dict(cfg); cfg_copy["time_schedule"] = "log_t"
    return make_schedule(cfg_copy, device)


# ---------------------------------------------------------------------------
# Reconstruction Frontier Tracker — Adaptive Time Sampling
# ---------------------------------------------------------------------------
# Concentrates score-training budget at the "information frontier":
# the region in t where |d_t log R(t)| is largest.
#
# R(t) = E[ ||D(z_t, t) - D(z_0, None)||^2 ]  (oracle TDD error)
# Adaptive weight:  w(t) ~ |d(log R) / d(log t)| + floor
#
# Feedback loop: CONVERGENT (negative). Improvement at the frontier
# reduces weight there; frontier walks outward to next region.
# ---------------------------------------------------------------------------

class ReconFrontierTracker:
    """Tracks decoder reconstruction frontier R(t) and provides adaptive
    time sampling concentrated at the information frontier."""

    def __init__(self, t_min, t_max, n_bins=100, ema_decay=0.99,
                 floor_weight=0.02, warmup_steps=500, min_counts_per_bin=5,
                 smooth_sigma=3.0, device=torch.device("cpu")):
        self.t_min = float(t_min); self.t_max = float(t_max)
        self.n_bins = n_bins; self.ema_decay = ema_decay
        self.floor_weight = floor_weight; self.warmup_steps = warmup_steps
        self.min_counts_per_bin = min_counts_per_bin; self.device = device
        self.smooth_sigma = smooth_sigma  # Gaussian kernel σ in bins for pre-diff smoothing
        log_edges = torch.linspace(math.log(self.t_min), math.log(self.t_max),
                                   n_bins + 1, device=self.device, dtype=torch.float32)
        self.bin_edges = torch.exp(log_edges)
        self.bin_centers = torch.exp(0.5 * (log_edges[:-1] + log_edges[1:]))
        self.bin_log_widths = log_edges[1:] - log_edges[:-1]
        self.R_ema = torch.ones(n_bins, device=self.device)
        self.counts = torch.zeros(n_bins, device=self.device, dtype=torch.long)
        self.total_updates = 0
        self._weights_cache = None; self._weights_dirty = True

    @property
    def is_active(self):
        return (self.total_updates >= self.warmup_steps
                and bool((self.counts >= self.min_counts_per_bin).all().item()))

    @torch.no_grad()
    def update(self, t_batch, mse_batch):
        t_batch = t_batch.detach().to(self.device)
        mse_batch = mse_batch.detach().float().to(self.device)
        bin_idx = torch.bucketize(t_batch, self.bin_edges[1:-1])
        bin_sums = torch.zeros(self.n_bins, device=self.device).scatter_add_(0, bin_idx, mse_batch)
        ones = torch.ones_like(bin_idx, dtype=torch.long)
        bin_counts = torch.zeros(self.n_bins, device=self.device, dtype=torch.long).scatter_add_(0, bin_idx, ones)
        active = bin_counts > 0
        if active.any():
            bin_means = bin_sums[active] / bin_counts[active].float()
            first_visit = self.counts[active] == 0
            self.R_ema[active] = torch.where(first_visit, bin_means,
                self.ema_decay * self.R_ema[active] + (1.0 - self.ema_decay) * bin_means)
            self.counts[active] += bin_counts[active]
        self.total_updates += 1; self._weights_dirty = True

    @staticmethod
    def _gaussian_smooth_1d(x: torch.Tensor, sigma: float) -> torch.Tensor:
        """1-D Gaussian smoothing with reflect-padding (preserves endpoints)."""
        if sigma <= 0:
            return x
        radius = int(math.ceil(3.0 * sigma))
        ks = 2 * radius + 1
        grid = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        kernel = torch.exp(-0.5 * (grid / sigma) ** 2)
        kernel = kernel / kernel.sum()
        # conv1d expects [B, C, L]
        x_pad = torch.nn.functional.pad(x.unsqueeze(0).unsqueeze(0),
                                         (radius, radius), mode="reflect")
        return torch.nn.functional.conv1d(x_pad, kernel.view(1, 1, ks)).squeeze()

    @torch.no_grad()
    def _compute_weights(self):
        if not self._weights_dirty and self._weights_cache is not None:
            return self._weights_cache
        log_R = torch.log(self.R_ema.clamp(min=1e-12))
        # --- smooth log R in bin-index space before finite-differencing ---
        log_R = self._gaussian_smooth_1d(log_R, self.smooth_sigma)
        log_t = torch.log(self.bin_centers)
        d_logR = log_R[1:] - log_R[:-1]
        d_logt = log_t[1:] - log_t[:-1]
        grad = (d_logR / (d_logt + 1e-12)).abs()
        w = torch.zeros(self.n_bins, device=self.device)
        w[0] = grad[0]; w[-1] = grad[-1]
        w[1:-1] = 0.5 * (grad[:-1] + grad[1:])
        w = w + self.floor_weight / self.n_bins
        w = w / w.sum()
        self._weights_cache = w; self._weights_dirty = False
        return w

    def sample(self, B, device):
        if not self.is_active:
            return sample_log_uniform_times(B, self.t_min, self.t_max, device)
        weights = self._compute_weights()
        bi = torch.multinomial(weights, B, replacement=True)
        log_lo = torch.log(self.bin_edges[bi])
        log_hi = torch.log(self.bin_edges[bi + 1])
        u = torch.rand(B, device=self.device)
        return torch.exp(log_lo + u * (log_hi - log_lo)).to(device)

    def sample_discrete(self, B, grid_times, device):
        T_grid = len(grid_times)
        if not self.is_active:
            return torch.randint(0, T_grid, (B,), device=device, dtype=torch.long)
        weights = self._compute_weights()
        gbi = torch.bucketize(grid_times.to(self.device), self.bin_edges[1:-1])
        gw = weights[gbi]; gw = gw / gw.sum()
        return torch.multinomial(gw, B, replacement=True).to(device)

    def sample_flow(self, B, t_min, t_max, device):
        if not self.is_active:
            return sample_logit_normal_times(B, t_min, t_max, device)
        return self.sample(B, device)

    def importance_weights(self, t_batch, reference="log_uniform"):
        if not self.is_active:
            return torch.ones_like(t_batch)
        weights = self._compute_weights()
        bi = torch.bucketize(t_batch.detach().to(self.device), self.bin_edges[1:-1])
        p_adapt = weights[bi] / (t_batch.to(self.device) * self.bin_log_widths[bi] + 1e-12)
        if reference == "log_uniform":
            p_ref = 1.0 / (t_batch.to(self.device) * math.log(self.t_max / self.t_min))
        elif reference == "uniform":
            p_ref = 1.0 / (self.t_max - self.t_min)
        else:
            raise ValueError(f"Unknown reference: {reference}")
        iw = p_ref / (p_adapt + 1e-12)
        iw = iw / (iw.mean() + 1e-12)
        return iw.detach().to(t_batch.device)

    @torch.no_grad()
    def make_adaptive_time_grid(self, N: int, device: torch.device,
                                descending: bool = True,
                                t_lo: float | None = None,
                                t_hi: float | None = None) -> torch.Tensor:
        """Build an N+1-point time grid by inverting the frontier-weight CDF.

        Each interval between consecutive grid points carries equal probability
        mass under the frontier weight distribution w(t).  This concentrates
        integration steps at the information frontier — exactly where
        |d log R / d log t| is large — matching training to inference.

        When the tracker is not yet active, falls back to a log-uniform grid.

        Parameters
        ----------
        N : int
            Number of integration *steps* (grid has N+1 points).
        device : torch.device
            Target device for the returned tensor.
        descending : bool
            If True (default, OU/cosine), grid runs high -> low.
            If False (flow matching), grid runs low -> high.
        t_lo : float | None
            Lower bound of the sub-interval (defaults to tracker's t_min).
        t_hi : float | None
            Upper bound of the sub-interval (defaults to tracker's t_max).
            When t_lo/t_hi are set, the CDF is restricted to [t_lo, t_hi]
            so that grid points only fall within the requested range while
            still respecting the adaptive weight profile.

        Returns
        -------
        Tensor of shape [N+1] — adaptive time grid.
        """
        t_lo = float(t_lo if t_lo is not None else self.t_min)
        t_hi = float(t_hi if t_hi is not None else self.t_max)

        if not self.is_active:
            # Fallback: log-uniform grid over the (possibly restricted) interval
            if descending:
                return torch.logspace(
                    math.log10(t_hi), math.log10(t_lo),
                    N + 1, device=device,
                )
            else:
                return torch.logspace(
                    math.log10(t_lo), math.log10(t_hi),
                    N + 1, device=device,
                )

        weights = self._compute_weights()                       # [n_bins]

        # Build piecewise-linear CDF over the bin *edges* in log-t space.
        # CDF(edge_0) = 0,  CDF(edge_k) = sum(weights[:k]),  CDF(edge_n) = 1
        cum_w = torch.cumsum(weights, dim=0)                    # [n_bins]
        cdf_edges = torch.zeros(self.n_bins + 1, device=self.device)
        cdf_edges[1:] = cum_w
        # Normalise (should already sum to 1, but be safe)
        cdf_edges = cdf_edges / cdf_edges[-1]

        log_edges = torch.log(self.bin_edges)                   # [n_bins+1]

        # --- Restrict to [t_lo, t_hi] sub-interval ---
        # Find CDF values at the sub-interval endpoints via linear interp
        log_t_lo = math.log(max(t_lo, self.t_min))
        log_t_hi = math.log(min(t_hi, self.t_max))

        def _interp_cdf(log_t_val):
            """Linearly interpolate the CDF at a single log-t value."""
            idx = torch.searchsorted(log_edges, torch.tensor(log_t_val, device=self.device))
            idx = idx.clamp(1, self.n_bins)
            lo_e = log_edges[idx - 1]; hi_e = log_edges[idx]
            frac = (log_t_val - lo_e) / (hi_e - lo_e + 1e-12)
            frac = float(frac.clamp(0.0, 1.0))
            return float(cdf_edges[idx - 1]) + frac * float(cdf_edges[idx] - cdf_edges[idx - 1])

        cdf_lo = _interp_cdf(log_t_lo)
        cdf_hi = _interp_cdf(log_t_hi)

        # Query quantiles: N+1 equally-spaced values in [cdf_lo, cdf_hi]
        if cdf_hi - cdf_lo < 1e-12:
            # Degenerate: no weight in this interval — fall back to log-uniform
            if descending:
                return torch.logspace(math.log10(t_hi), math.log10(t_lo), N + 1, device=device)
            else:
                return torch.logspace(math.log10(t_lo), math.log10(t_hi), N + 1, device=device)

        quantiles = torch.linspace(cdf_lo, cdf_hi, N + 1, device=self.device)

        # Invert CDF via searchsorted + linear interpolation
        idx = torch.searchsorted(cdf_edges, quantiles).clamp(1, self.n_bins)
        # Linear interp within each bin
        cdf_lo_vec = cdf_edges[idx - 1]
        cdf_hi_vec = cdf_edges[idx]
        log_t_lo_vec = log_edges[idx - 1]
        log_t_hi_vec = log_edges[idx]
        frac = (quantiles - cdf_lo_vec) / (cdf_hi_vec - cdf_lo_vec + 1e-12)
        frac = frac.clamp(0.0, 1.0)
        log_t_grid = log_t_lo_vec + frac * (log_t_hi_vec - log_t_lo_vec)

        t_grid = torch.exp(log_t_grid).clamp(t_lo, t_hi)

        if descending:
            t_grid = t_grid.flip(0)                             # high -> low

        return t_grid.to(device)

    @torch.no_grad()
    def perceptual_weights(
        self,
        t_batch: torch.Tensor,
        R_cutoff: float = 0.05,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Frontier-informed soft gate for perceptual losses (LPIPS, GAN).

        Returns per-sample weights in [0, 1]:
          - ≈ 1 where R(t) ≪ R_cutoff  (decoder output is close to clean → LPIPS meaningful)
          - ≈ 0 where R(t) ≫ R_cutoff  (decoder output is garbage → LPIPS would hallucinate)

        w(t) = sigmoid( (log R_cutoff - log R(t)) / τ )

        When the tracker is not yet active, falls back to SNR-based gamma weighting.

        Parameters
        ----------
        t_batch : Tensor [B]
            Per-sample time values.
        R_cutoff : float
            Reconstruction MSE threshold above which perceptual losses are gated off.
            For images in [-1,1], ~0.03-0.10 is the "blurry but recognizable" regime.
        temperature : float
            Controls sharpness of the sigmoid transition.
            τ=1.0 gives a soft gate; τ→0 gives a hard step.

        Returns
        -------
        Tensor [B] — weights in [0, 1], detached.
        """
        if not self.is_active:
            # Fallback: return ones (caller can combine with gamma if desired)
            return torch.ones(t_batch.shape[0], device=t_batch.device)

        t_batch = t_batch.detach().to(self.device)
        bi = torch.bucketize(t_batch, self.bin_edges[1:-1])
        R_at_t = self.R_ema[bi].clamp_min(1e-12)
        log_ratio = (math.log(max(R_cutoff, 1e-12)) - torch.log(R_at_t)) / max(temperature, 1e-8)
        weights = torch.sigmoid(log_ratio)
        return weights.detach().to(t_batch.device)

    def get_diagnostics(self):
        if not self.is_active:
            return {"frontier/active": 0.0, "frontier/peak_t": -1.0,
                    "frontier/entropy_ratio": -1.0, "frontier/max_weight": -1.0,
                    "frontier/coverage": float((self.counts > 0).sum().item()) / self.n_bins}
        w = self._compute_weights()
        pk = int(w.argmax().item())
        ent = -(w * torch.log(w + 1e-12)).sum().item()
        return {"frontier/active": 1.0, "frontier/peak_t": self.bin_centers[pk].item(),
                "frontier/entropy_ratio": ent / math.log(self.n_bins),
                "frontier/max_weight": w.max().item(),
                "frontier/R_at_peak": self.R_ema[pk].item(),
                "frontier/R_at_tmin": self.R_ema[0].item(),
                "frontier/R_at_tmax": self.R_ema[-1].item()}

    def get_R_curve(self):
        return self.bin_centers.cpu(), self.R_ema.cpu()

    def get_weight_curve(self):
        w = self._compute_weights() if self.is_active else (
            torch.ones(self.n_bins, device=self.device) / self.n_bins)
        return self.bin_centers.cpu(), w.cpu()


def plot_frontier_diagnostics(frontier_tracker, save_path, epoch):
    """Plot reconstruction curve R(t) and adaptive weight curve w(t)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    t_c, R_vals = frontier_tracker.get_R_curve()
    t_w, w_vals = frontier_tracker.get_weight_curve()
    ax1.loglog(t_c.numpy(), R_vals.numpy(), 'b.-', linewidth=1.5, markersize=2)
    ax1.set_xlabel('t'); ax1.set_ylabel(r'$R(t)$')
    ax1.set_title(f'Reconstruction Frontier (ep {epoch})')
    ax1.grid(True, alpha=0.3)
    ax2.semilogx(t_w.numpy(), w_vals.numpy(), 'r.-', linewidth=1.5, markersize=2)
    ax2.set_xlabel('t'); ax2.set_ylabel('w(t)')
    ax2.set_title(f'Frontier Weight (ep {epoch})')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0/len(w_vals), color='gray', ls='--', alpha=0.5, label='uniform')
    ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close(fig)


def t_idx_to_time(t_idx: torch.Tensor, cfg: Dict[str, Any], T: int) -> torch.Tensor:
    """Map discrete indices t_idx in [0, T-1] to a continuous scalar used ONLY for time embedding."""
    t_min = float(cfg.get("t_min", 2e-5))
    t_max = float(cfg.get("t_max", 2.0))
    if T <= 1:
        return torch.full_like(t_idx.float(), t_max)
    frac = t_idx.float() / float(T - 1)
    return t_min + frac * (t_max - t_min)



# ---------------------------------------------------------------------------
# Metrics: SW2, MMD, FID, Diversity
# ---------------------------------------------------------------------------

def compute_sw2(
    x: torch.Tensor,
    y: torch.Tensor,
    n_projections: int = 1000,
    theta: torch.Tensor | None = None,
) -> float:
    """Sliced Wasserstein-2 distance in latent space.

    If `theta` is provided, it must be a fixed bank of unit-norm projection vectors with shape [D, n_projections]
    (or [D, K] where K >= n_projections). This makes SW2 deterministic across eval calls.
    """
    x = x.detach().float()
    y = y.detach().float()
    device = x.device
    N, D = x.shape

    if theta is None:
        # Random projections
        theta = torch.randn(D, n_projections, device=device)
        theta = theta / torch.norm(theta, dim=0, keepdim=True).clamp_min(1e-12)
    else:
        # Allow CPU or GPU theta; move to device and slice to requested projections.
        theta = theta.to(device)
        assert theta.dim() == 2 and theta.shape[0] == D, f"theta must have shape [D, K] with D={D}; got {tuple(theta.shape)}"
        assert theta.shape[1] >= n_projections, f"theta has only {theta.shape[1]} projections; need >= {n_projections}"
        theta = theta[:, :n_projections]

    # Project
    proj_x = x @ theta
    proj_y = y @ theta

    # Sort along sample axis
    proj_x, _ = torch.sort(proj_x, dim=0)
    proj_y, _ = torch.sort(proj_y, dim=0)

    # Mean squared difference
    w2 = torch.mean((proj_x - proj_y) ** 2)
    return w2.item()

# NOTE: MMD computation removed (we only use fixed sliced-W2 in evaluation).
def compute_diversity(imgs: torch.Tensor, lpips_fn: Any) -> float:
    """
    Computes pairwise LPIPS diversity.
    Higher = more diverse samples. Near 0 = mode collapse.
    """
    if lpips_fn is None: return 0.0

    # Ensure we have enough images
    N = imgs.shape[0]
    if N < 2: return 0.0

    # Shuffle
    perm = torch.randperm(N)
    imgs = imgs[perm]

    # Split into two halves
    half = N // 2
    set1 = imgs[:half]
    set2 = imgs[half:2*half]

    # Adapt for LPIPS (Need 3 channels)
    if set1.shape[1] == 1:
        set1 = set1.repeat(1, 3, 1, 1)
        set2 = set2.repeat(1, 3, 1, 1)

    # Compute distance
    with torch.no_grad():
        dist = lpips_fn(set1, set2)

    return dist.mean().item()

def log_latent_stats(name, mu_z, logvar_z=None):
    with torch.no_grad():
        mu_mean_norm = mu_z.mean(0).norm().item()
        mu_std_mean = mu_z.std(0).mean().item()
        mu_max_val = mu_z.abs().max().item()
        msg = (
            f"  [{name}] Latent Stats | "
            f"mu Mean Norm: {mu_mean_norm:.4f} | mu Avg Std: {mu_std_mean:.4f} | mu Max: {mu_max_val:.4f}"
        )
        if logvar_z is not None:
            logvar_mean_norm = logvar_z.mean(0).norm().item()
            logvar_std_mean = logvar_z.std(0).mean().item()
            logvar_max_val = logvar_z.abs().max().item()
            msg += (
                f" | logvar Mean Norm: {logvar_mean_norm:.4f}"
                f" | logvar Avg Std: {logvar_std_mean:.4f}"
                f" | logvar Max: {logvar_max_val:.4f}"
            )
        print(msg)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LeNet Feature Extractor for FID (non-FMNIST datasets)
# ---------------------------------------------------------------------------

class LeNetFeatureExtractor(nn.Module):
    """
    LeNet-style CNN classifier for FID feature extraction.
    Feature dimension: 256 (penultimate layer)
    """
    def __init__(self, num_classes: int = 10, feature_dim: int = 256, img_size: int = 28, img_channels: int = 1):
        super().__init__()
        self.feature_dim = feature_dim
        self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        # After 3 pooling layers: img_size -> img_size//2 -> img_size//4 -> img_size//8
        # For img_size=28 (padded to 32): 32->16->8->4, so 4x4
        # For img_size=32: 32->16->8->4, so 4x4
        final_spatial = img_size // 8
        self.fc1 = nn.Linear(128 * final_spatial * final_spatial, feature_dim)
        self.bn_fc = nn.BatchNorm1d(feature_dim)
        self.fc2 = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.extract_features(x)
        return self.fc2(features)

    def extract_features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        return x


def train_fid_classifier(train_loader, num_classes, device, epochs=10, lr=1e-3, checkpoint_path=None, img_size=32, img_channels=1):
    """Train LeNet classifier for FID feature extraction."""
    model = LeNetFeatureExtractor(num_classes=num_classes, img_size=img_size, img_channels=img_channels).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Loading FID classifier from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        model.eval()
        return model

    print(f"  Training FID classifier ({epochs} epochs)...")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        correct, total = 0, 0
        for x, y in tqdm(train_loader, desc=f"  FID Clf Ep {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        print(f"    Epoch {epoch+1}: Acc={100.*correct/total:.2f}%")

    if checkpoint_path:
        ensure_parent(checkpoint_path)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  Saved FID classifier to {checkpoint_path}")

    model.eval()
    return model


def get_fid_model(dataset_key, train_loader, num_classes, device, ckpt_dir="checkpoints"):
    """Get appropriate FID model: Inception for FMNIST/GCIFAR/CIFAR, LeNet for others."""
    if dataset_key in ("FMNIST", "GCIFAR", "CIFAR"):
        print(f"--> Using Inception features for FID ({dataset_key})")
        return None, False

    # Get img_size from dataset info (default 32 for padded 28x28 datasets)
    img_size = DATASET_INFO.get(dataset_key, {}).get("img_size", 28)
    img_channels = DATASET_INFO.get(dataset_key, {}).get("img_channels", 1)
    # For datasets that get padded (28->32), use 32 for the classifier
    effective_img_size = 32 if img_size == 28 else img_size

    print(f"--> Training LeNet classifier for FID ({dataset_key})")
    checkpoint_path = os.path.join(ckpt_dir, f"fid_classifier_{dataset_key.lower()}.pt")
    model = train_fid_classifier(train_loader, num_classes, device, epochs=10, checkpoint_path=checkpoint_path, img_size=effective_img_size, img_channels=img_channels)
    return model, True


def extract_lenet_features(images, device, batch_size, lenet_model):
    """Extract features using trained LeNet classifier."""
    lenet_model.eval()
    features_list = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device)
            feat = lenet_model.extract_features(batch)
            features_list.append(feat.cpu())
    return torch.cat(features_list, 0), lenet_model


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# =============================================================================
# Helper Functions for Evaluation Metrics
# =============================================================================

def extract_inception_features(images, device, batch_size=100, inception_model=None):
    """
    Extract Inception features for FID/KID computation.
    Uses InceptionV3 with final FC layer removed to get 2048-dim features.

    Args:
        images: [N, C, H, W] tensor of images (in [-1, 1] range)
        device: torch device
        batch_size: batch size for feature extraction
        inception_model: optional pre-loaded inception model (for reuse)

    Returns:
        features: [N, 2048] tensor of Inception features
        inception_model: the loaded model (for reuse)
    """
    # Lazy load inception model
    if inception_model is None:
        from torchvision.models import inception_v3
        inception_model = inception_v3(pretrained=True, transform_input=False)
        inception_model.fc = torch.nn.Identity()  # Remove final FC layer
        inception_model = inception_model.to(device)
        inception_model.eval()

    features_list = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device)

            # Inception expects 3-channel images of size 299x299
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)

            # Resize to 299x299
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)

            # Normalize to ImageNet stats (Inception expects [0, 1] input, we have [-1, 1])
            batch = (batch + 1) / 2  # [-1, 1] -> [0, 1]
            batch = (batch - torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)) / \
                    torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

            feat = inception_model(batch)
            features_list.append(feat.cpu())

    return torch.cat(features_list, 0), inception_model


def compute_fid_from_features(real_features, fake_features):
    """
    Compute FID from pre-extracted Inception features.

    FID = ||mu_r - mu_f||^2 + Tr(Sigma_r + Sigma_f - 2*sqrt(Sigma_r @ Sigma_f))

    Args:
        real_features: [N, D] tensor of real image features
        fake_features: [M, D] tensor of fake image features

    Returns:
        fid: scalar FID value
    """
    # Convert to numpy for numerical stability in matrix sqrt
    real_features = real_features.cpu().numpy()
    fake_features = fake_features.cpu().numpy()

    mu_r = np.mean(real_features, axis=0)
    mu_f = np.mean(fake_features, axis=0)

    sigma_r = np.cov(real_features, rowvar=False)
    sigma_f = np.cov(fake_features, rowvar=False)

    # Compute sqrt(Sigma_r @ Sigma_f) via eigendecomposition
    # covmean = scipy.linalg.sqrtm(sigma_r @ sigma_f)
    from scipy import linalg

    diff = mu_r - mu_f

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r) + np.trace(sigma_f) - 2 * np.trace(covmean)

    return float(fid)


def compute_kid(real_features, fake_features, num_subsets=100, subset_size=1000):
    """
    Compute KID (Kernel Inception Distance) using polynomial kernel with subsampling.

    Uses polynomial kernel: k(x, y) = (x^T y / d + 1)^3
    KID = MMD^2 between real and fake Inception features

    Args:
        real_features: [N, D] tensor of real image features
        fake_features: [M, D] tensor of fake image features
        num_subsets: number of random subsets for averaging
        subset_size: size of each subset

    Returns:
        kid_mean: mean KID across subsets
    """
    # Ensure tensors
    if not isinstance(real_features, torch.Tensor):
        real_features = torch.tensor(real_features)
    if not isinstance(fake_features, torch.Tensor):
        fake_features = torch.tensor(fake_features)

    n_real = real_features.shape[0]
    n_fake = fake_features.shape[0]
    d = real_features.shape[1]

    # Ensure we have enough samples
    subset_size = min(subset_size, n_real, n_fake)

    def polynomial_kernel(x, y):
        # k(x, y) = (x^T y / d + 1)^3
        return ((x @ y.T) / d + 1) ** 3

    kid_values = []

    for _ in range(num_subsets):
        # Random subsets
        real_idx = torch.randperm(n_real)[:subset_size]
        fake_idx = torch.randperm(n_fake)[:subset_size]

        real_subset = real_features[real_idx]
        fake_subset = fake_features[fake_idx]

        # Compute kernel matrices
        k_rr = polynomial_kernel(real_subset, real_subset)
        k_ff = polynomial_kernel(fake_subset, fake_subset)
        k_rf = polynomial_kernel(real_subset, fake_subset)

        # MMD^2 unbiased estimator
        m = subset_size

        # Remove diagonal for unbiased estimate
        diag_rr = torch.diag(k_rr)
        diag_ff = torch.diag(k_ff)

        sum_rr = (k_rr.sum() - diag_rr.sum()) / (m * (m - 1))
        sum_ff = (k_ff.sum() - diag_ff.sum()) / (m * (m - 1))
        sum_rf = k_rf.mean()

        mmd2 = sum_rr + sum_ff - 2 * sum_rf
        kid_values.append(mmd2.item())

    return np.mean(kid_values)



def compute_lsi_gap(
    score_net,
    encoder_mus,
    encoder_logvars,
    cfg,
    device,
    labels: torch.Tensor | None = None,
    num_classes: int | None = None,
    num_samples: int = 5000,
    num_time_points: int = 50,
    batch_size: int = 128,
):
    """Compute your LSI-gap diagnostic.

    - For OU/cosine schedules (legacy): assumes the score_net predicts eps.
    - For flow-matching (time_schedule='flow'): assumes the score_net predicts eps directly.
    """
    if score_net is None:
        return 0.0

    score_net.eval()

    n_data = encoder_mus.shape[0]
    if labels is not None:
        assert labels.shape[0] == n_data, "labels must align with encoder_mus/logvars"

    num_samples = min(int(num_samples), int(n_data))
    sample_indices = torch.randperm(n_data, device="cpu")[:num_samples]

    stype = str(cfg.get("time_schedule", "log_snr")).lower()

    if stype in ("flow", "flow_matching"):
        t_min = float(cfg.get("t_min", 1e-5))
        t_max = float(cfg.get("t_max", 0.99999))
        t_grid = torch.linspace(t_min, t_max, int(num_time_points), device=device, dtype=torch.float32)

        total_lsi_gap = 0.0
        total_count = 0

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_indices = sample_indices[i:i + batch_size]
                batch_mu = encoder_mus[batch_indices].to(device)
                batch_logvar = encoder_logvars[batch_indices].to(device)

                batch_var = torch.exp(batch_logvar)
                batch_std = torch.exp(0.5 * batch_logvar)
                bsz = batch_mu.shape[0]

                # Labels for CFG-conditional nets
                y_batch = None
                if labels is not None:
                    y_batch = labels[batch_indices].to(device=device, dtype=torch.long).view(-1)
                    if num_classes is not None:
                        if (y_batch.min() < 0) or (y_batch.max() >= num_classes):
                            raise ValueError("labels out of range for num_classes")

                # Sample z0 ~ q(z0 | x)
                eps_0 = torch.randn_like(batch_mu)
                z0 = batch_mu + batch_std * eps_0

                for t_val in t_grid:
                    t = t_val.expand(bsz)  # [bsz]
                    alpha = t_val.view(1, 1, 1, 1)  # scalar broadcast
                    sigma = (1.0 - t_val).view(1, 1, 1, 1).clamp_min(1e-8)

                    noise = torch.randn_like(z0)
                    z_t = alpha * z0 + sigma * noise

                    mu_t = alpha * batch_mu
                    var_t = (alpha ** 2) * batch_var + (sigma ** 2)

                    # component conditional mean of eps
                    eps_target_lsi = sigma * ((z_t - mu_t) / (var_t + 1e-8))

                    # model predicts eps directly (eps-parameterization)
                    eps_pred = score_net(z_t, t, y_batch)

                    sigma_sq = sigma ** 2 + 1e-8
                    eps_diff_sq = (eps_pred - eps_target_lsi) ** 2

                    #score_gap_per_sample = (eps_diff_sq / sigma_sq).sum(dim=(1, 2, 3))  # [bsz]
                    score_gap_per_sample = eps_diff_sq.sum(dim=(1, 2, 3))


                    total_lsi_gap += score_gap_per_sample.sum().item()
                    total_count += bsz

        return total_lsi_gap / total_count if total_count > 0 else 0.0

    # ---------------- legacy OU / cosine (eps parameterization) ----------------
    ou_sched = make_schedule(cfg, device)
    T = int(ou_sched["T"].item())
    t_idx_grid = torch.linspace(0, T - 1, int(num_time_points), device=device).round().long()

    total_lsi_gap = 0.0
    total_count = 0

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_indices = sample_indices[i:i + batch_size]
            batch_mu = encoder_mus[batch_indices].to(device)
            batch_logvar = encoder_logvars[batch_indices].to(device)

            batch_var = torch.exp(batch_logvar)
            batch_std = torch.exp(0.5 * batch_logvar)
            bsz = batch_mu.shape[0]

            y_batch = None
            if labels is not None:
                y_batch = labels[batch_indices].to(device=device, dtype=torch.long).view(-1)

            eps_0 = torch.randn_like(batch_mu)
            z0 = batch_mu + batch_std * eps_0

            for t_idx_scalar in t_idx_grid:
                t_idx = t_idx_scalar.expand(bsz)

                t_val = ou_sched["times"].gather(0, t_idx_scalar.view(1)).view(1)
                t = t_val.expand(bsz).float()

                alpha = extract_schedule(ou_sched["alpha"], t_idx, z0.shape)
                sigma = extract_schedule(ou_sched["sigma"], t_idx, z0.shape)

                noise = torch.randn_like(z0)
                z_t = alpha * z0 + sigma * noise

                mu_t = alpha * batch_mu
                var_t = (alpha ** 2) * batch_var + (sigma ** 2)

                eps_target_lsi = sigma * ((z_t - mu_t) / (var_t + 1e-8))
                eps_pred = score_net(z_t, t, y_batch)

                sigma_sq = sigma ** 2 + 1e-8
                eps_diff_sq = (eps_pred - eps_target_lsi) ** 2

                #score_gap_per_sample = (eps_diff_sq / sigma_sq).sum(dim=(1, 2, 3))  # [bsz]
                score_gap_per_sample = eps_diff_sq.sum(dim=(1, 2, 3))

                total_lsi_gap += score_gap_per_sample.sum().item()
                total_count += bsz

    return total_lsi_gap / total_count if total_count > 0 else 0.0


def compute_mse_gap(
    score_net,
    oracle_model,
    encoder_mus: torch.Tensor,
    encoder_logvars: torch.Tensor,
    cfg: Dict[str, Any],
    device: torch.device,
    labels: torch.Tensor | None = None,
    num_classes: int | None = None,
    num_samples: int = 5000,
    batch_size: int = 128,
    space: str = "eps",
) -> float:
    """Integrated MSE between the learned score network and the oracle score.

    MSE_gap = E_{t, x}[ || eps_theta(z_t, t) - eps*(z_t, t) ||_2^2 ]

    where t is drawn *uniformly* over the full discrete training time grid
    (``num_train_timesteps`` points produced by ``make_schedule``), and
    z_t is obtained by

        x  -->  (mu_0, Sigma_0) = Enc(x)
        z_0  ~  N(mu_0, Sigma_0)
        z_t  =  alpha(t) z_0  +  sigma(t) eps,   eps ~ N(0,I)

    Parameters
    ----------
    score_net : nn.Module | OracleScoreModel
        Learned eps-prediction network (same call signature as DiTModel).
    oracle_model : OracleScoreModel
        Oracle (exact CSEM) eps-prediction model.
    encoder_mus, encoder_logvars : Tensor [N, C, H, W]
        Precomputed encoder outputs over the evaluation dataset (CPU).
    cfg : dict
        Experiment configuration (needs time_schedule, t_min, t_max, num_train_timesteps, …).
    device : torch.device
    labels : Tensor [N] or None
        Class labels; passed to both score_net and oracle_model.
    num_classes : int or None
    num_samples : int
        How many data points to average over.
    batch_size : int
    space : ``"eps"`` | ``"score"``
        ``"eps"``   – plain MSE in eps-prediction space (default).
        ``"score"`` – divides each term by sigma(t)^2 so that the comparison
                      is in score space  (score = -eps / sigma).

    Returns
    -------
    float  – the averaged MSE gap.
    """
    if score_net is None or oracle_model is None:
        return 0.0

    score_net.eval()
    oracle_model.eval()

    n_data = encoder_mus.shape[0]
    num_samples = min(int(num_samples), int(n_data))
    sample_indices = torch.randperm(n_data, device="cpu")[:num_samples]

    stype = str(cfg.get("time_schedule", "log_snr")).lower()

    # ------------------------------------------------------------------
    # Build the full discrete training time grid
    # ------------------------------------------------------------------
    if stype in ("flow", "flow_matching"):
        T = int(cfg.get("num_train_timesteps", 1000))
        t_min = float(cfg.get("t_min", 1e-5))
        t_max = float(cfg.get("t_max", 0.99999))
        t_grid = torch.linspace(t_min, t_max, T, device=device, dtype=torch.float32)

        total_mse = 0.0
        total_count = 0

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_indices = sample_indices[i:i + batch_size]
                batch_mu = encoder_mus[batch_indices].to(device)
                batch_logvar = encoder_logvars[batch_indices].to(device)
                batch_std = torch.exp(0.5 * batch_logvar)
                bsz = batch_mu.shape[0]

                y_batch = None
                if labels is not None:
                    y_batch = labels[batch_indices].to(device=device, dtype=torch.long).view(-1)

                # Sample z0 ~ q(z0 | x)
                eps_0 = torch.randn_like(batch_mu)
                z0 = batch_mu + batch_std * eps_0

                for t_val in t_grid:
                    t = t_val.expand(bsz)
                    alpha = t_val.view(1, 1, 1, 1)
                    sigma = (1.0 - t_val).view(1, 1, 1, 1).clamp_min(1e-8)

                    noise = torch.randn_like(z0)
                    z_t = alpha * z0 + sigma * noise

                    eps_pred = score_net(z_t, t, y_batch)
                    eps_oracle = oracle_model(z_t, t, y_batch)

                    diff_sq = (eps_pred - eps_oracle) ** 2
                    if space == "score":
                        sigma_sq = (sigma ** 2).clamp_min(1e-8)
                        diff_sq = diff_sq / sigma_sq

                    mse_per_sample = diff_sq.sum(dim=(1, 2, 3))  # [bsz]
                    total_mse += mse_per_sample.sum().item()
                    total_count += bsz

        return total_mse / total_count if total_count > 0 else 0.0

    # ------------------------------------------------------------------
    # OU / cosine (discrete schedule)
    # ------------------------------------------------------------------
    ou_sched = make_schedule(cfg, device)
    T = int(ou_sched["T"].item())
    # Use all T discrete timesteps
    t_idx_grid = torch.arange(T, device=device, dtype=torch.long)

    total_mse = 0.0
    total_count = 0

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_indices = sample_indices[i:i + batch_size]
            batch_mu = encoder_mus[batch_indices].to(device)
            batch_logvar = encoder_logvars[batch_indices].to(device)
            batch_std = torch.exp(0.5 * batch_logvar)
            bsz = batch_mu.shape[0]

            y_batch = None
            if labels is not None:
                y_batch = labels[batch_indices].to(device=device, dtype=torch.long).view(-1)

            eps_0 = torch.randn_like(batch_mu)
            z0 = batch_mu + batch_std * eps_0

            for t_idx_scalar in t_idx_grid:
                t_idx = t_idx_scalar.expand(bsz)

                t_val = ou_sched["times"].gather(0, t_idx_scalar.view(1)).view(1)
                t = t_val.expand(bsz).float()

                alpha = extract_schedule(ou_sched["alpha"], t_idx, z0.shape)
                sigma = extract_schedule(ou_sched["sigma"], t_idx, z0.shape)

                noise = torch.randn_like(z0)
                z_t = alpha * z0 + sigma * noise

                eps_pred = score_net(z_t, t, y_batch)
                eps_oracle = oracle_model(z_t, t, y_batch)

                diff_sq = (eps_pred - eps_oracle) ** 2
                if space == "score":
                    sigma_sq = (sigma ** 2).clamp_min(1e-8)
                    diff_sq = diff_sq / sigma_sq

                mse_per_sample = diff_sq.sum(dim=(1, 2, 3))  # [bsz]
                total_mse += mse_per_sample.sum().item()
                total_count += bsz

    return total_mse / total_count if total_count > 0 else 0.0


# ── Per-t MSE gap + new evaluation visualizations ────────────────────

# Module-level cache: stores y-limits from the first eval so subsequent
# evals use identical axes for comparability.
_mse_gap_ylim_cache: Dict[str, Tuple[float, float]] = {}


def compute_mse_gap_by_t(
    score_net,
    oracle_model,
    encoder_mus: torch.Tensor,
    encoder_logvars: torch.Tensor,
    cfg: Dict[str, Any],
    device: torch.device,
    labels: torch.Tensor | None = None,
    num_classes: int | None = None,
    num_samples: int = 5000,
    batch_size: int = 128,
    space: str = "eps",
    num_t_bins: int = 30,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Like compute_mse_gap but also returns a per-t-bin breakdown.

    Returns
    -------
    scalar_mse : float – overall mean MSE (same as compute_mse_gap).
    t_bin_centres : np.ndarray [num_t_bins] – bin centre t values.
    mse_by_bin : np.ndarray [num_t_bins] – mean MSE in each bin.
    """
    if score_net is None or oracle_model is None:
        return 0.0, np.zeros(num_t_bins), np.zeros(num_t_bins)

    score_net.eval()
    oracle_model.eval()

    n_data = encoder_mus.shape[0]
    num_samples = min(int(num_samples), int(n_data))
    sample_indices = torch.randperm(n_data, device="cpu")[:num_samples]

    stype = str(cfg.get("time_schedule", "log_snr")).lower()

    # ------------------------------------------------------------------
    # Build the full discrete training time grid
    # ------------------------------------------------------------------
    if stype in ("flow", "flow_matching"):
        T = int(cfg.get("num_train_timesteps", 1000))
        t_min = float(cfg.get("t_min", 1e-5))
        t_max = float(cfg.get("t_max", 0.99999))
        t_grid = torch.linspace(t_min, t_max, T, device=device, dtype=torch.float32)
    else:
        ou_sched = make_schedule(cfg, device)
        T_int = int(ou_sched["T"].item())
        t_grid = ou_sched["times"].to(device)  # [T_int]

    t_grid_np = t_grid.cpu().numpy()
    # Create log-spaced bin edges for a smooth curve
    t_lo, t_hi = float(t_grid_np.min()), float(t_grid_np.max())
    bin_edges = np.logspace(np.log10(max(t_lo, 1e-8)), np.log10(t_hi), num_t_bins + 1)
    bin_sums = np.zeros(num_t_bins, dtype=np.float64)
    bin_counts = np.zeros(num_t_bins, dtype=np.float64)
    t_bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    total_mse = 0.0
    total_count = 0

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_indices = sample_indices[i:i + batch_size]
            batch_mu = encoder_mus[batch_indices].to(device)
            batch_logvar = encoder_logvars[batch_indices].to(device)
            batch_std = torch.exp(0.5 * batch_logvar)
            bsz = batch_mu.shape[0]

            y_batch = None
            if labels is not None:
                y_batch = labels[batch_indices].to(device=device, dtype=torch.long).view(-1)

            eps_0 = torch.randn_like(batch_mu)
            z0 = batch_mu + batch_std * eps_0

            if stype in ("flow", "flow_matching"):
                for t_val in t_grid:
                    t = t_val.expand(bsz)
                    alpha = t_val.view(1, 1, 1, 1)
                    sigma = (1.0 - t_val).view(1, 1, 1, 1).clamp_min(1e-8)

                    noise = torch.randn_like(z0)
                    z_t = alpha * z0 + sigma * noise

                    eps_pred = score_net(z_t, t, y_batch)
                    eps_oracle = oracle_model(z_t, t, y_batch)

                    diff_sq = (eps_pred - eps_oracle) ** 2
                    if space == "score":
                        sigma_sq = (sigma ** 2).clamp_min(1e-8)
                        diff_sq = diff_sq / sigma_sq

                    mse_per_sample = diff_sq.sum(dim=(1, 2, 3))
                    batch_mse_mean = mse_per_sample.mean().item()
                    total_mse += mse_per_sample.sum().item()
                    total_count += bsz

                    # Bin assignment
                    tv = float(t_val.item())
                    bidx = int(np.searchsorted(bin_edges, tv, side="right")) - 1
                    bidx = max(0, min(bidx, num_t_bins - 1))
                    bin_sums[bidx] += batch_mse_mean * bsz
                    bin_counts[bidx] += bsz
            else:
                ou_sched_local = make_schedule(cfg, device)
                t_idx_grid = torch.arange(int(ou_sched_local["T"].item()), device=device, dtype=torch.long)
                for t_idx_scalar in t_idx_grid:
                    t_idx = t_idx_scalar.expand(bsz)
                    t_val = ou_sched_local["times"].gather(0, t_idx_scalar.view(1)).view(1)
                    t = t_val.expand(bsz).float()

                    alpha = extract_schedule(ou_sched_local["alpha"], t_idx, z0.shape)
                    sigma = extract_schedule(ou_sched_local["sigma"], t_idx, z0.shape)

                    noise = torch.randn_like(z0)
                    z_t = alpha * z0 + sigma * noise

                    eps_pred = score_net(z_t, t, y_batch)
                    eps_oracle = oracle_model(z_t, t, y_batch)

                    diff_sq = (eps_pred - eps_oracle) ** 2
                    if space == "score":
                        sigma_sq = (sigma ** 2).clamp_min(1e-8)
                        diff_sq = diff_sq / sigma_sq

                    mse_per_sample = diff_sq.sum(dim=(1, 2, 3))
                    batch_mse_mean = mse_per_sample.mean().item()
                    total_mse += mse_per_sample.sum().item()
                    total_count += bsz

                    tv = float(t_val.item())
                    bidx = int(np.searchsorted(bin_edges, tv, side="right")) - 1
                    bidx = max(0, min(bidx, num_t_bins - 1))
                    bin_sums[bidx] += batch_mse_mean * bsz
                    bin_counts[bidx] += bsz

    scalar_mse = total_mse / total_count if total_count > 0 else 0.0

    # Per-bin means (leave zero where no data fell)
    valid = bin_counts > 0
    mse_by_bin = np.zeros(num_t_bins, dtype=np.float64)
    mse_by_bin[valid] = bin_sums[valid] / bin_counts[valid]

    return scalar_mse, t_bin_centres, mse_by_bin


def _conditional_tweedie_readout(
    unet,
    z_t: torch.Tensor,
    t_vec: torch.Tensor,
    y: torch.Tensor | None,
    schedule_type: str,
    cosine_s: float = 0.008,
) -> torch.Tensor:
    """Compute z_hat_0 = (z_t - sigma * eps_cond) / alpha using single-model
    conditional prediction (no CFG).

    This matches the 'conditional' readout mode in UniversalSampler._apply_readout
    and is the correct pre-image estimate to feed into a TDD decoder that was
    trained with mse_mode='score' or 'score_detached'.

    Parameters
    ----------
    unet    : score network (in eval mode)
    z_t     : noisy latent [B, C, H, W]
    t_vec   : time values [B]
    y       : class labels [B] or None
    schedule_type : one of 'log_snr', 'log_t', 'cosine', 'flow', 'flow_matching'
    cosine_s : cosine schedule shift

    Returns
    -------
    z_hat_0 : Tweedie-denoised latent estimate [B, C, H, W]
    """
    B = z_t.shape[0]
    t_4d = t_vec.view(B, 1, 1, 1)

    stype = str(schedule_type).lower()
    if stype in ("flow", "flow_matching"):
        alpha, sigma = get_flow_params(t_4d)
    elif stype == "cosine":
        alpha, sigma, _ = get_cosine_params(t_4d, cosine_s=cosine_s)
    else:
        alpha, sigma = get_ou_params(t_4d)

    # Single-model conditional prediction — in-distribution for the decoder
    eps_cond = unet(z_t, t_vec, y)
    z_hat_0 = (z_t - sigma * eps_cond) / (alpha + 1e-8)
    return z_hat_0


def plot_recon_error_vs_t(
    vae,
    encoder_mus: torch.Tensor,
    encoder_logvars: torch.Tensor,
    real_imgs: torch.Tensor,
    cfg: Dict[str, Any],
    device: torch.device,
    save_path: str,
    num_t_points: int = 10,
    num_samples: int = 256,
    batch_size: int = 64,
    unet=None,
    real_labels: torch.Tensor | None = None,
):
    """Viz I: Reconstruction error (log) vs t (log).

    For each of ``num_t_points`` log-spaced t values in [t_min, t_max],
    encode x → z_0, form z_t = alpha(t)*z_0 + sigma(t)*eps, decode
    D(z_t, t), and record MSE vs x.

    When mse_mode is 'score' or 'score_detached', the decoder was trained on
    Tweedie-denoised z_hat_0, not raw z_t.  In that case we apply a
    single-model conditional Tweedie readout before decoding.
    """
    vae.eval()
    t_min = float(cfg["t_min"])
    t_max = float(cfg["t_max"])
    stype = str(cfg.get("time_schedule", "log_snr")).lower()
    mse_mode = str(cfg.get("mse_mode", "raw")).lower()
    use_tweedie = mse_mode in ("score", "score_detached") and unet is not None

    t_values = np.logspace(np.log10(t_min), np.log10(t_max), num_t_points)

    n_data = encoder_mus.shape[0]
    num_samples = min(num_samples, n_data)
    idx = torch.randperm(n_data)[:num_samples]

    mu_sub = encoder_mus[idx].to(device)
    logvar_sub = encoder_logvars[idx].to(device)
    std_sub = torch.exp(0.5 * logvar_sub)
    imgs_sub = real_imgs[idx].to(device)
    labels_sub = real_labels[idx].to(device) if real_labels is not None else None

    eps_0 = torch.randn_like(mu_sub)
    z0 = mu_sub + std_sub * eps_0

    mse_values = []

    with torch.no_grad():
        for tv in t_values:
            t_tensor = torch.tensor(tv, device=device, dtype=torch.float32)
            if stype in ("flow", "flow_matching"):
                alpha = t_tensor
                sigma = (1.0 - t_tensor).clamp_min(1e-8)
            elif stype == "cosine":
                a, s, _ = get_cosine_params(
                    t_tensor.view(1, 1), cosine_s=float(cfg.get("cosine_s", 0.008))
                )
                alpha, sigma = a.squeeze(), s.squeeze()
            else:
                a, s = get_ou_params(t_tensor.view(1, 1))
                alpha, sigma = a.squeeze(), s.squeeze()

            noise = torch.randn_like(z0)
            z_t = alpha * z0 + sigma * noise

            # Decode in mini-batches
            recon_list = []
            for j in range(0, num_samples, batch_size):
                z_batch = z_t[j:j + batch_size]
                bsz_j = z_batch.shape[0]
                t_vec_j = torch.full((bsz_j,), tv, device=device)
                y_j = labels_sub[j:j + batch_size] if labels_sub is not None else None

                # Conditional Tweedie readout when decoder trained on z_hat_0
                if use_tweedie:
                    z_dec = _conditional_tweedie_readout(
                        unet, z_batch, t_vec_j, y_j,
                        schedule_type=stype,
                        cosine_s=float(cfg.get("cosine_s", 0.008)),
                    )
                else:
                    z_dec = z_batch

                if getattr(vae, 'time_cond_decoder', False):
                    t_dec = t_vec_j
                else:
                    t_dec = None
                recon_list.append(vae.decode(z_dec, t=t_dec, y=y_j))
            recon = torch.cat(recon_list, 0)

            mse_val = ((recon - imgs_sub) ** 2).mean().item()
            mse_values.append(mse_val)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.loglog(t_values, mse_values, "o-", linewidth=2, markersize=5)
    ax.set_xlabel("t")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Reconstruction Error vs t")
    ax.set_ylim(5e-3, 0.5)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"    Saved recon-error-vs-t plot → {save_path}")


def plot_lpips_decay_vs_t(
    vae,
    encoder_mus: torch.Tensor,
    encoder_logvars: torch.Tensor,
    real_imgs: torch.Tensor,
    cfg: Dict[str, Any],
    device: torch.device,
    lpips_fn,
    save_path: str,
    num_t_points: int = 10,
    num_samples: int = 256,
    batch_size: int = 64,
    unet=None,
    real_labels: torch.Tensor | None = None,
):
    """Viz: LPIPS growth E[lpips(D(z_0, t_min), D(z_t,t))] vs t (log-log).

    Uses the same t-grid and decoding pathway as ``plot_recon_error_vs_t``.
    When mse_mode is 'score' or 'score_detached', the decoder was trained on
    Tweedie-denoised z_hat_0, not raw z_t. In that case we apply a
    single-model conditional Tweedie readout before decoding.
    """
    if lpips_fn is None:
        print("    [plot_lpips_decay_vs_t] LPIPS unavailable — skipping plot.")
        return

    vae.eval()
    lpips_fn.eval()
    t_min = float(cfg["t_min"])
    t_max = float(cfg["t_max"])
    stype = str(cfg.get("time_schedule", "log_snr")).lower()
    mse_mode = str(cfg.get("mse_mode", "raw")).lower()
    use_tweedie = mse_mode in ("score", "score_detached") and unet is not None

    t_values = np.logspace(np.log10(t_min), np.log10(t_max), num_t_points)

    n_data = encoder_mus.shape[0]
    num_samples = min(num_samples, n_data)
    idx = torch.randperm(n_data)[:num_samples]

    mu_sub = encoder_mus[idx].to(device)
    logvar_sub = encoder_logvars[idx].to(device)
    std_sub = torch.exp(0.5 * logvar_sub)
    imgs_sub = real_imgs[idx].to(device)
    labels_sub = real_labels[idx].to(device) if real_labels is not None else None

    eps_0 = torch.randn_like(mu_sub)
    z0 = mu_sub + std_sub * eps_0

    lpips_values = []

    with torch.no_grad():
        ref_recon_chunks = []
        for j in range(0, num_samples, batch_size):
            z0_batch = z0[j:j + batch_size]
            bsz_j = z0_batch.shape[0]
            t_ref_j = torch.full((bsz_j,), t_min, device=device)
            if getattr(vae, 'time_cond_decoder', False):
                t_ref_dec = t_ref_j
            else:
                t_ref_dec = None
            ref_recon_chunks.append(vae.decode(z0_batch, t=t_ref_dec, y=labels_sub[j:j + batch_size] if labels_sub is not None else None))
        ref_recon = torch.cat(ref_recon_chunks, 0)

        for tv in t_values:
            t_tensor = torch.tensor(tv, device=device, dtype=torch.float32)
            if stype in ("flow", "flow_matching"):
                alpha = t_tensor
                sigma = (1.0 - t_tensor).clamp_min(1e-8)
            elif stype == "cosine":
                a, s, _ = get_cosine_params(
                    t_tensor.view(1, 1), cosine_s=float(cfg.get("cosine_s", 0.008))
                )
                alpha, sigma = a.squeeze(), s.squeeze()
            else:
                a, s = get_ou_params(t_tensor.view(1, 1))
                alpha, sigma = a.squeeze(), s.squeeze()

            noise = torch.randn_like(z0)
            z_t = alpha * z0 + sigma * noise

            lpips_sum = 0.0
            lpips_count = 0
            for j in range(0, num_samples, batch_size):
                z_batch = z_t[j:j + batch_size]
                ref_batch = ref_recon[j:j + batch_size]
                bsz_j = z_batch.shape[0]
                t_vec_j = torch.full((bsz_j,), tv, device=device)
                y_j = labels_sub[j:j + batch_size] if labels_sub is not None else None

                if use_tweedie:
                    z_dec = _conditional_tweedie_readout(
                        unet, z_batch, t_vec_j, y_j,
                        schedule_type=stype,
                        cosine_s=float(cfg.get("cosine_s", 0.008)),
                    )
                else:
                    z_dec = z_batch

                if getattr(vae, 'time_cond_decoder', False):
                    t_dec = t_vec_j
                else:
                    t_dec = None
                recon = vae.decode(z_dec, t=t_dec, y=y_j)

                if ref_batch.shape[1] == 1:
                    ref_lp = ref_batch.repeat(1, 3, 1, 1)
                    recon_lp = recon.repeat(1, 3, 1, 1)
                else:
                    ref_lp = ref_batch
                    recon_lp = recon

                lp_batch = lpips_fn(ref_lp, recon_lp)
                lpips_sum += float(lp_batch.mean().item()) * bsz_j
                lpips_count += bsz_j

            lpips_values.append(max(lpips_sum / max(lpips_count, 1), 1e-8))

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.loglog(t_values, lpips_values, "o-", linewidth=2, markersize=5)
    ax.set_xlabel("t")
    ax.set_ylabel("LPIPS")
    ax.set_title("LPIPS Decay vs t")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"    Saved LPIPS-decay-vs-t plot → {save_path}")


def plot_mse_gap_by_t(
    t_bin_centres: np.ndarray,
    mse_by_bin: np.ndarray,
    save_path: str,
    cache_key: str = "default",
):
    """Viz II: MSE gap (eps-space) vs t (log-log), with consistent y-limits.

    On the first call for a given ``cache_key``, the y-limits are recorded
    from the data.  Subsequent calls reuse those limits for comparability.
    """
    global _mse_gap_ylim_cache

    valid = mse_by_bin > 0
    if not np.any(valid):
        print(f"    [plot_mse_gap_by_t] No valid bins — skipping plot.")
        return

    t_plot = t_bin_centres[valid]
    mse_plot = mse_by_bin[valid]

    if cache_key not in _mse_gap_ylim_cache:
        y_lo = float(mse_plot.min()) * 0.5
        y_hi = float(mse_plot.max()) * 2.0
        _mse_gap_ylim_cache[cache_key] = (y_lo, y_hi)

    y_lo, y_hi = _mse_gap_ylim_cache[cache_key]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.loglog(t_plot, mse_plot, "o-", linewidth=2, markersize=4, color="tab:red")
    ax.set_xlabel("t")
    ax.set_ylabel("MSE gap (score-space)")
    ax.set_title("Score MSE Gap vs t (score parameterization)")
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"    Saved mse-gap-by-t plot → {save_path}")


def plot_decoder_output_grid(
    vae,
    encoder_mus: torch.Tensor,
    encoder_logvars: torch.Tensor,
    real_imgs: torch.Tensor,
    cfg: Dict[str, Any],
    device: torch.device,
    save_path: str,
    num_rows: int = 5,
    num_cols: int = 5,
    t_upper: float = 1.0,
    t_values: list[float] | None = None,
    unet=None,
    real_labels: torch.Tensor | None = None,
):
    """Viz III: D(z_t, t) grid — rows = different x_0, columns = different t.

    ``num_cols`` t values are log-uniform from t_min to ``t_upper``.
    ``num_rows`` different x_0's are encoded and noised.

    When mse_mode is 'score' or 'score_detached', the decoder was trained on
    Tweedie-denoised z_hat_0, not raw z_t.  In that case we apply a
    single-model conditional Tweedie readout before decoding.
    """
    vae.eval()
    t_min = float(cfg["t_min"])
    stype = str(cfg.get("time_schedule", "log_snr")).lower()
    mse_mode = str(cfg.get("mse_mode", "raw")).lower()
    use_tweedie = mse_mode in ("score", "score_detached") and unet is not None

    if t_values is None:
      t_values = np.logspace(np.log10(t_min), np.log10(t_upper), num_cols)

    n_data = encoder_mus.shape[0]
    idx = torch.randperm(n_data)[:num_rows]

    mu_sub = encoder_mus[idx].to(device)
    logvar_sub = encoder_logvars[idx].to(device)
    std_sub = torch.exp(0.5 * logvar_sub)
    imgs_sub = real_imgs[idx]  # keep on CPU for display
    labels_sub = real_labels[idx].to(device) if real_labels is not None else None

    eps_0 = torch.randn_like(mu_sub)
    z0 = mu_sub + std_sub * eps_0

    # Collect decoded images: list of [num_rows, C, H, W]
    all_panels = []

    with torch.no_grad():
        for tv in t_values:
            t_tensor = torch.tensor(tv, device=device, dtype=torch.float32)
            if stype in ("flow", "flow_matching"):
                alpha = t_tensor
                sigma = (1.0 - t_tensor).clamp_min(1e-8)
            elif stype == "cosine":
                a, s, _ = get_cosine_params(
                    t_tensor.view(1, 1), cosine_s=float(cfg.get("cosine_s", 0.008))
                )
                alpha, sigma = a.squeeze(), s.squeeze()
            else:
                a, s = get_ou_params(t_tensor.view(1, 1))
                alpha, sigma = a.squeeze(), s.squeeze()

            noise = torch.randn_like(z0)
            z_t = alpha * z0 + sigma * noise

            # Conditional Tweedie readout when decoder trained on z_hat_0
            if use_tweedie:
                t_vec = torch.full((num_rows,), tv, device=device)
                z_dec = _conditional_tweedie_readout(
                    unet, z_t, t_vec, labels_sub,
                    schedule_type=stype,
                    cosine_s=float(cfg.get("cosine_s", 0.008)),
                )
            else:
                z_dec = z_t

            if getattr(vae, 'time_cond_decoder', False):
                t_dec = torch.full((num_rows,), tv, device=device)
            else:
                t_dec = None
            decoded = vae.decode(z_dec, t=t_dec, y=labels_sub).cpu()
            all_panels.append(decoded)

    # Build the grid: rows = x_0, cols = t.  Also prepend a column of originals.
    # Final grid order: [orig_0, D(z_t1,t1)_0, ..., D(z_tK,tK)_0, orig_1, ...]
    grid_imgs = []
    for row_i in range(num_rows):
        grid_imgs.append(imgs_sub[row_i])  # original x_0
        for col_j in range(num_cols):
            grid_imgs.append(all_panels[col_j][row_i])
    grid_tensor = torch.stack(grid_imgs, dim=0)

    ncol = num_cols + 1  # +1 for the original image column
    tv_utils.save_image(
        (grid_tensor + 1) / 2, save_path,
        nrow=ncol, padding=2, pad_value=0.5,
    )

    # Also add a small title annotation via matplotlib for the t-labels
    fig, ax = plt.subplots(1, 1, figsize=(2.0 * ncol, 2.0 * num_rows))
    from PIL import Image as _PILImage
    img = _PILImage.open(save_path)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    col_labels = ["x₀"] + [f"t={tv:.3g}" for tv in t_values]
    # Approximate pixel positions for column labels
    w = img.width
    cell_w = w / ncol
    for ci, lbl in enumerate(col_labels):
        ax.text(cell_w * (ci + 0.5), 4, lbl, ha="center", va="top",
                fontsize=8, color="white",
                bbox=dict(facecolor="black", alpha=0.6, pad=1))
    fig.tight_layout(pad=0.3)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"    Saved decoder-output grid → {save_path}")

def plot_reverse_trajectory_grid(
    vae,
    score_model,
    cfg: Dict[str, Any],
    device: torch.device,
    save_path: str,
    num_rows: int = 5,
    num_cols: int = 5,
    t_upper: float = 1.0,
    steps_per_leg: int = 10,
    cfg_scale: float = 3.0,
    class_label: int | list[int] | None = None,
    t_values: list[float] | None = None,
    frontier_tracker=None,
    save_movie: bool = False,
    plot_path_norms: bool = False,
):
    """Viz V: Reverse-trajectory D(z_t, t) grid.

    Works with either the learned score net or OracleScoreModel.

    Static grid:
        Columns correspond to the requested snapshot times in strictly
        descending order.

    Optional path-norm plot:
        If plot_path_norms=True, additionally saves
            <save_root>_path_norms.png
        showing per-row curves of
            ||D(z_t,t) - D(z_{t_min}, t_min)||_2
        vs t on log-log axes.

    Notes:
    - The reverse path is always traced all the way down to t_min so that
      D(z_{t_min}, t_min) is available as the reference for the norm plot.
    - The reference point itself is excluded from the displayed norm plot
      to avoid the artificial log-scale plunge caused by plotting zero.
    """
    if score_model is None:
        print("    [plot_reverse_trajectory_grid] No score model — skipping.")
        return

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image as _PILImage

    vae.eval()
    if hasattr(score_model, "eval"):
        score_model.eval()

    t_min = float(cfg["t_min"])
    t_max = float(cfg["t_max"])
    stype = str(cfg.get("time_schedule", "log_snr")).lower()
    mse_mode = str(cfg.get("mse_mode", "raw")).lower()
    use_tweedie = mse_mode in ("score", "score_detached")

    # ------------------------------------------------------------------
    # Resolve snapshot times for the static grid (descending)
    # ------------------------------------------------------------------
    if t_values is None:
        upper = min(float(t_upper), t_max)
        if upper <= t_min:
            raise ValueError(f"t_upper={t_upper} must exceed t_min={t_min}.")
        snapshot_t_stops = np.logspace(np.log10(upper), np.log10(t_min), num_cols).tolist()
    else:
        snapshot_t_stops = [float(t) for t in t_values if t_min <= float(t) <= t_max]
        if len(snapshot_t_stops) == 0:
            raise ValueError("No t_values remain after clipping to [t_min, t_max].")
        snapshot_t_stops = sorted(set(snapshot_t_stops), reverse=True)

    # ------------------------------------------------------------------
    # Trace times for the reverse path:
    # force inclusion of t_min for the path-norm reference.
    # Keep grid snapshots unchanged unless user explicitly included t_min.
    # ------------------------------------------------------------------
    trace_t_stops = sorted(set(snapshot_t_stops + [t_min]), reverse=True)

    # Construct strictly descending legs
    legs = []
    current_t = t_max
    for t_stop in trace_t_stops:
        if t_stop >= current_t:
            continue
        legs.append((current_t, t_stop))
        current_t = t_stop

    if len(legs) == 0:
        raise ValueError("No valid descending legs could be constructed.")

    snapshot_t_set = set(snapshot_t_stops)

    # ------------------------------------------------------------------
    # Initial noise
    # ------------------------------------------------------------------
    img_size = cfg.get("img_size", 32)
    latent_spatial = img_size // 4
    latent_shape = (num_rows, int(cfg["latent_channels"]), latent_spatial, latent_spatial)
    z = torch.randn(latent_shape, device=device)

    # ------------------------------------------------------------------
    # Resolve per-row class labels
    # ------------------------------------------------------------------
    num_classes = int(cfg.get("num_classes", 10))
    if class_label is None:
        row_labels = [int(i * num_classes / num_rows) % num_classes for i in range(num_rows)]
    elif isinstance(class_label, (list, tuple)):
        row_labels = [int(class_label[i % len(class_label)]) for i in range(num_rows)]
    else:
        row_labels = [int(class_label)] * num_rows
    y_batch = torch.tensor(row_labels, device=device, dtype=torch.long)

    # ------------------------------------------------------------------
    # Outputs / bookkeeping
    # ------------------------------------------------------------------
    snapshots_by_t = {}

    save_root, _ = os.path.splitext(save_path)
    frames_dir = f"{save_root}_frames"
    movie_path = f"{save_root}.mp4"
    gif_path = f"{save_root}.gif"
    path_norms_path = f"{save_root}_path_norms.png"
    frame_paths = []

    path_times: list[float] = []
    path_decoded: list[torch.Tensor] = []

    def _decode_for_viz(z_raw: torch.Tensor, t_scalar: float) -> torch.Tensor:
        t_vec = torch.full((num_rows,), float(t_scalar), device=device, dtype=z_raw.dtype)
        if use_tweedie:
            z_dec = _conditional_tweedie_readout(
                score_model,
                z_raw,
                t_vec,
                y_batch,
                schedule_type=stype,
                cosine_s=float(cfg.get("cosine_s", 0.008)),
            )
        else:
            z_dec = z_raw
        decoded = vae.decode(z_dec, t=t_vec, y=y_batch).cpu()
        return decoded

    def _step_sampler_once(sampler, x, t_curr, t_next):
        if sampler.method == "ddim":
            return sampler.step_ddim(
                x, t_curr=t_curr, t_next=t_next, unet=score_model,
                y=y_batch, cfg_scale=cfg_scale, generator=None,
            )
        elif sampler.method == "rk4_ode":
            return sampler.step_rk4_ode(x, t_curr, t_next, score_model, y=y_batch, cfg_scale=cfg_scale)
        elif sampler.method == "euler_ode":
            return sampler.step_euler_ode(x, t_curr, t_next, score_model, y=y_batch, cfg_scale=cfg_scale)
        elif sampler.method == "heun_ode":
            return sampler.step_heun_ode(x, t_curr, t_next, score_model, y=y_batch, cfg_scale=cfg_scale)
        elif sampler.method == "heun_sde":
            return sampler.step_heun_sde(x, t_curr, t_next, score_model, y=y_batch, cfg_scale=cfg_scale, generator=None)
        elif sampler.method == "exp_euler_ode":
            return sampler.step_exp_euler_ode(x, t_curr, t_next, score_model, y=y_batch, cfg_scale=cfg_scale)
        elif sampler.method == "exp_heun_ode":
            return sampler.step_exp_heun_ode(x, t_curr, t_next, score_model, y=y_batch, cfg_scale=cfg_scale)
        else:
            raise ValueError(f"Unknown sampling method: {sampler.method}")

    def _record_path(decoded_batch: torch.Tensor, t_value: float):
        path_times.append(float(t_value))
        path_decoded.append(decoded_batch.detach().cpu())

    def _save_movie_frame(decoded_batch: torch.Tensor, t_value: float, frame_idx: int) -> str:
        ensure_dir(frames_dir)
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}_t_{float(t_value):.6f}.png")

        tv_utils.save_image(
            (decoded_batch + 1) / 2,
            frame_path,
            nrow=1,
            padding=2,
            pad_value=0.5,
        )

        fig, ax = plt.subplots(1, 1, figsize=(2.4, 2.0 * num_rows))
        img = _PILImage.open(frame_path)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        h = img.height
        cell_h = h / num_rows

        ax.text(
            img.width * 0.5, 4, f"t={float(t_value):.4g}",
            ha="center", va="top",
            fontsize=10, color="white",
            bbox=dict(facecolor="black", alpha=0.65, pad=1),
        )
        for ri, yl in enumerate(row_labels):
            ax.text(
                4, cell_h * (ri + 0.5), f"y={yl}",
                ha="left", va="center",
                fontsize=8, color="white", rotation=90,
                bbox=dict(facecolor="black", alpha=0.6, pad=1),
            )

        fig.tight_layout(pad=0.3)
        fig.savefig(frame_path, dpi=150)
        plt.close(fig)
        return frame_path

    # ------------------------------------------------------------------
    # Integrate reverse path
    # ------------------------------------------------------------------
    with torch.no_grad():
        frame_idx = 0

        decoded0 = _decode_for_viz(z, t_max)
        if plot_path_norms:
            _record_path(decoded0, t_max)
        if save_movie:
            frame_paths.append(_save_movie_frame(decoded0, t_max, frame_idx))
            frame_idx += 1

        for (t_from, t_to) in legs:
            sampler = UniversalSampler(
                method="rk4_ode",
                num_steps=steps_per_leg,
                t_min=t_to,
                t_max=t_from,
                schedule_type=cfg.get("time_schedule", "log_snr"),
                cosine_s=cfg.get("cosine_s", 0.008),
                readout_mode="direct",
                frontier_tracker=frontier_tracker,
                cfg_mode=cfg.get("cfg_mode", "constant"),
            )

            ts = sampler._make_time_grid(device)

            ts_list = [float(t.item()) for t in ts]
            if any(ts_list[i + 1] >= ts_list[i] for i in range(len(ts_list) - 1)):
                raise RuntimeError(f"Sampler time grid is not strictly descending: {ts_list}")

            for step_i in range(sampler.num_steps):
                t_curr = ts[step_i]
                t_next = ts[step_i + 1]
                z = _step_sampler_once(sampler, z, t_curr, t_next)

                if save_movie or plot_path_norms:
                    decoded_step = _decode_for_viz(z, float(t_next.item()))
                    if plot_path_norms:
                        _record_path(decoded_step, float(t_next.item()))
                    if save_movie:
                        frame_paths.append(_save_movie_frame(decoded_step, float(t_next.item()), frame_idx))
                        frame_idx += 1

            # Decode exact leg endpoint for snapshot bookkeeping
            decoded_leg_end = _decode_for_viz(z, t_to)
            if t_to in snapshot_t_set:
                snapshots_by_t[float(t_to)] = decoded_leg_end

    # ------------------------------------------------------------------
    # Static endpoint grid (requested snapshot times only)
    # ------------------------------------------------------------------
    snapshot_times_present = [t for t in snapshot_t_stops if float(t) in snapshots_by_t]
    if len(snapshot_times_present) == 0:
        raise RuntimeError("No snapshot endpoints were collected for the static grid.")

    grid_imgs = []
    for row_i in range(num_rows):
        for t_snap in snapshot_times_present:
            grid_imgs.append(snapshots_by_t[float(t_snap)][row_i])
    grid_tensor = torch.stack(grid_imgs, dim=0)

    tv_utils.save_image(
        (grid_tensor + 1) / 2,
        save_path,
        nrow=len(snapshot_times_present),
        padding=2,
        pad_value=0.5,
    )

    fig, ax = plt.subplots(1, 1, figsize=(2.0 * len(snapshot_times_present), 2.0 * num_rows))
    img = _PILImage.open(save_path)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])

    col_labels = [f"t={tv:.3g}" for tv in snapshot_times_present]
    w = img.width
    h = img.height
    cell_w = w / len(snapshot_times_present)
    cell_h = h / num_rows

    for ci, lbl in enumerate(col_labels):
        ax.text(
            cell_w * (ci + 0.5), 4, lbl,
            ha="center", va="top",
            fontsize=8, color="white",
            bbox=dict(facecolor="black", alpha=0.6, pad=1),
        )

    for ri, yl in enumerate(row_labels):
        ax.text(
            4, cell_h * (ri + 0.5), f"y={yl}",
            ha="left", va="center",
            fontsize=8, color="white", rotation=90,
            bbox=dict(facecolor="black", alpha=0.6, pad=1),
        )

    fig.tight_layout(pad=0.3)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"    Saved reverse-trajectory grid -> {save_path}")

    # ------------------------------------------------------------------
    # Path norms:
    # plot ||D(z_t,t) - D(z_{t_min}, t_min)||_2 for all traced points,
    # excluding the final reference point itself to avoid a fake log-drop.
    # ------------------------------------------------------------------
    if plot_path_norms and len(path_decoded) >= 2:
        path_stack = torch.stack(path_decoded, dim=0)   # [K, R, C, H, W]
        t_arr = np.asarray(path_times, dtype=np.float64)

        # Use final traced point as the reference; by construction this is t_min.
        ref = path_stack[-1]  # [R, C, H, W]

        path_norms = (
            (path_stack - ref.unsqueeze(0))
            .flatten(2)
            .norm(dim=2)
            .cpu()
            .numpy()
        )  # [K, R]

        # Sort by t ascending for plotting
        order = np.argsort(t_arr)
        t_plot_all = t_arr[order]
        norm_plot_all = path_norms[order]

        # Exclude the reference point itself (the final t_min point).
        ref_tol = max(1e-14, 1e-8 * t_min)
        mask = t_plot_all > (t_min + ref_tol)

        t_plot = np.clip(t_plot_all[mask], 1e-12, None)
        norm_plot = np.clip(norm_plot_all[mask], 1e-12, None)

        if len(t_plot) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.8))
            for ri, yl in enumerate(row_labels):
                ax.loglog(t_plot, norm_plot[:, ri], linewidth=1.6, label=f"row {ri+1} (y={yl})")

            ax.set_xlabel("t")
            ax.set_ylabel(r"$\|D(z_t,t) - D(z_{t_{\min}}, t_{\min})\|_2$")
            ax.set_title("Reverse-path decoder continuity")
            ax.grid(True, which="both", alpha=0.3)
            ax.invert_xaxis()
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            fig.savefig(path_norms_path, dpi=150)
            plt.close(fig)
            print(f"    Saved reverse-path norm plot -> {path_norms_path}")
        else:
            print("    Warning: no valid non-reference points remained for path-norm plot.")

    # ------------------------------------------------------------------
    # Movie
    # ------------------------------------------------------------------
    if save_movie and len(frame_paths) > 0:
        fps = max(1, int(steps_per_leg))
        try:
            import imageio.v2 as imageio
            frames_np = [np.asarray(_PILImage.open(fp).convert("RGB")) for fp in frame_paths]
            imageio.mimsave(movie_path, frames_np, fps=fps)
            print(f"    Saved reverse-trajectory movie -> {movie_path}")
            print(f"    Saved per-step frames -> {frames_dir}")
        except Exception as e_mp4:
            try:
                frames_pil = [_PILImage.open(fp).convert("P", palette=_PILImage.ADAPTIVE) for fp in frame_paths]
                duration_ms = int(round(1000.0 / fps))
                frames_pil[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames_pil[1:],
                    duration=duration_ms,
                    loop=0,
                )
                print(f"    Warning: mp4 write failed ({e_mp4}). Saved GIF instead -> {gif_path}")
                print(f"    Saved per-step frames -> {frames_dir}")
            except Exception as e_gif:
                print(f"    Warning: failed to save movie. mp4 error: {e_mp4}; gif error: {e_gif}")
                print(f"    Per-step frames were still saved -> {frames_dir}")


# ── VAE ───────────────────────────────────────────────────────────────

"""
Refactored VAE with LDM-aligned architectural improvements.

Changes vs previous version (all gated by flags, defaults reproduce old behaviour):
  1. conv3x3_proj=True  : GN→SiLU→3×3 combined mu+logvar projection (encoder)
                          + 3×3 decoder input conv (replaces 1×1)
  2. decoder_extra_block : +1 VAEResBlock per decoder stage (LDM asymmetry)
  3. encoder_attn_half   : attention at half-res (16×16) in encoder
  4. use_tanh_out=False  : raw decoder output (no tanh saturation)
  5. clamp_logvar=True   : clamp logvar to [-30, 20]
  6. attn_zero_init=False: standard init on VAE attention proj (faster learning)
"""

# ── VAEAttentionBlock (updated: optional zero-init) ──────────────────────

class VAEAttentionBlock(nn.Module):
    """
    Multi-head self-attention with optional zero-init on output projection.

    zero_init=True  (default): proj starts as no-op — good for score nets.
    zero_init=False           : standard init — faster attention learning in VAEs.
    """
    def __init__(self, ch: int, num_heads: int = 4, zero_init: bool = True):
        super().__init__()
        ch = int(ch)
        num_heads = int(num_heads)

        num_heads = min(num_heads, ch)
        while num_heads > 1 and (ch % num_heads) != 0:
            num_heads -= 1

        self.ch = ch
        self.num_heads = num_heads
        self.head_dim = ch // num_heads

        self.norm = make_group_norm(ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, kernel_size=1)
        self.proj = nn.Conv2d(ch, ch, kernel_size=1)

        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        qkv = qkv.view(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = torch.matmul(q.transpose(-2, -1).float(), k.float())
        attn = attn * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1).to(q.dtype)

        out = torch.matmul(v, attn.transpose(-2, -1))
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


# ── VAEResBlock (unchanged) ───────────────────────────────────────────

class VAEResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            make_group_norm(in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            make_group_norm(out_ch), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x): return self.net(x) + self.skip(x)


# ── Decoder-specific time-conditioned modules ───────────────────────────

def decoder_time_embedding(t: torch.Tensor, dim: int, schedule_type: str = "log_t") -> torch.Tensor:
    """Compute a time embedding for the decoder using log-SNR as the base quantity.

    Log-SNR is approximately linear in "perceptual difficulty" of reconstruction,
    giving the sinusoidal embedding more uniform coverage of the regime the decoder
    needs to distinguish.

    For flow-matching schedules (alpha=t, sigma=1-t), we compute
        lambda = log(t^2 / (1-t)^2)
    which is the analogous SNR quantity.
    """
    if schedule_type in ("flow", "flow_matching"):
        # flow: alpha=t, sigma=1-t  =>  logSNR = log(t^2/(1-t)^2)
        t_clamp = t.clamp(1e-6, 1.0 - 1e-6)
        logsnr = torch.log(t_clamp ** 2 / ((1.0 - t_clamp) ** 2 + 1e-12) + 1e-12)
    elif schedule_type == "cosine":
        # For cosine VP, compute logSNR from alpha_bar
        t_clamp = t.clamp(1e-6, 1.0 - 1e-6)
        g = ((t_clamp + 0.008) / 1.008) * (math.pi / 2.0)
        g0 = (0.008 / 1.008) * (math.pi / 2.0)
        alpha_bar = (torch.cos(g) / math.cos(g0)) ** 2
        alpha_bar = alpha_bar.clamp(1e-8, 1.0 - 1e-8)
        logsnr = torch.log(alpha_bar / (1.0 - alpha_bar))
    else:
        # OU process (log_t, log_snr): use ou_logsnr directly
        logsnr = ou_logsnr(t)
    return timestep_embedding(logsnr, dim)


class TimeCondVAEResBlock(nn.Module):
    """Decoder ResBlock with per-block AdaGN time conditioning (FiLM on second norm).

    Mirrors the score network's ResBlock pattern: the time embedding produces
    (scale, shift) that modulate the second GroupNorm, giving every block
    its own time-dependent behavior.

    Zero-init on conv2 so the block starts as identity.
    """
    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.norm1 = make_group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.time_proj = nn.Linear(t_dim, 2 * out_ch)  # scale, shift
        self.norm2 = make_group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        # Zero-init conv2 so block starts as identity
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_proj(t_emb).chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class TimeCondAttentionBlock(nn.Module):
    """Multi-head self-attention with time-dependent temperature modulation.

    At small t (high SNR), attention should focus on local sharpening.
    At large t (low SNR), attention needs long-range coherence.
    The time embedding modulates the attention temperature (scale) and adds
    a residual gate, similar to adaLN-Zero in DiT blocks.
    """
    def __init__(self, ch: int, t_dim: int, num_heads: int = 4, zero_init: bool = True):
        super().__init__()
        ch = int(ch)
        num_heads = int(num_heads)
        num_heads = min(num_heads, ch)
        while num_heads > 1 and (ch % num_heads) != 0:
            num_heads -= 1

        self.ch = ch
        self.num_heads = num_heads
        self.head_dim = ch // num_heads

        self.norm = make_group_norm(ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, kernel_size=1)
        self.proj = nn.Conv2d(ch, ch, kernel_size=1)

        # Time-dependent modulation: (scale_factor, gate) for attention
        # scale_factor modulates attention temperature
        # gate controls residual contribution (adaLN-Zero style)
        self.time_attn_mod = nn.Linear(t_dim, 2)
        # Init: temperature scale=0 (so effective scale = 1.0), gate=0 (zero residual at init)
        nn.init.zeros_(self.time_attn_mod.weight)
        nn.init.zeros_(self.time_attn_mod.bias)

        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        qkv = qkv.view(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Time-dependent attention temperature and gate
        mod = self.time_attn_mod(t_emb)  # [B, 2]
        temp_scale = mod[:, 0].view(B, 1, 1, 1)  # modulates 1/sqrt(d)
        attn_gate = torch.sigmoid(mod[:, 1]).view(B, 1, 1, 1)  # residual gate

        # Effective scale: base scale * (1 + learned offset)
        effective_scale = (self.head_dim ** -0.5) * (1.0 + temp_scale)

        attn = torch.matmul(q.transpose(-2, -1).float(), k.float())
        attn = attn * effective_scale.float()
        attn = attn.softmax(dim=-1).to(q.dtype)

        out = torch.matmul(v, attn.transpose(-2, -1))
        out = out.reshape(B, C, H, W)
        return x + attn_gate * self.proj(out)


class VAE(nn.Module):
    """
    Convolutional VAE with two stride-2 downsamples (img_size → img_size/4 latent).

    Backward-compatible: every NEW parameter's default reproduces the prior
    architecture exactly.

    Recommended CIFAR call (with all improvements):
        VAE(latent_channels=8, base_ch=64, img_size=32, img_channels=3,
            num_res_blocks=2, decoder_attn_half=True, latent_proj_depth=2,
            encoder_attn_half=True, decoder_extra_block=True,
            conv3x3_proj=True, use_tanh_out=False,
            clamp_logvar=True, attn_zero_init=False,
            num_classes=10, cond_emb_dim=64, use_norm=True)
    """

    def __init__(
        self,
        latent_channels: int = 4,
        base_ch: int = 32,
        use_norm: bool = False,
        img_size: int = 28,
        img_channels: int = 1,
        # --- Optional conditional encoder ---
        num_classes: int | None = None,
        null_label: int | None = None,
        cond_emb_dim: int = 64,
        # --- Auxiliary encoder noise (mixture-of-Gaussians per x) ---
        aux_d: int = 0,
        # --- Capacity knobs (v1, defaults = original behaviour) ---
        num_res_blocks: int = 1,
        decoder_attn_half: bool = False,
        latent_proj_depth: int = 1,
        # --- v2 architectural knobs (defaults = v1 behaviour) ---
        encoder_attn_half: bool = False,     # [4] attention at half-res in encoder
        decoder_extra_block: bool = False,   # [3] +1 ResBlock per decoder stage
        conv3x3_proj: bool = False,          # [1,2] 3×3 latent projection + decoder input
        use_tanh_out: bool = True,           # [5] False = raw output, no tanh
        clamp_logvar: bool = False,          # [6] clamp logvar to [-30, 20]
        attn_zero_init: bool = True,         # [7] False = standard init on VAE attn
        # --- Time-dependent decoder (TDD) ---
        time_cond_decoder: bool = False,     # Enable FiLM time conditioning in decoder
        dec_time_emb_dim: int = 128,         # Dimension of decoder time embedding
        # --- Optional class-conditioned decoder ---
        class_decoder: bool = True,
        decoder_num_classes: int | None = None,
        decoder_null_label: int | None = None,
    ):
        super().__init__()

        # ---- bookkeeping ------------------------------------------------
        self.use_norm = use_norm
        self.img_size = img_size
        self.img_channels = int(img_channels)
        self.latent_channels = int(latent_channels)
        self.base_ch = int(base_ch)
        self.num_res_blocks = int(num_res_blocks)
        self.decoder_attn_half = decoder_attn_half
        self.latent_proj_depth = int(latent_proj_depth)
        self.aux_d = int(aux_d)
        self.encoder_attn_half = encoder_attn_half
        self.decoder_extra_block = decoder_extra_block
        self.conv3x3_proj = conv3x3_proj
        self.use_tanh_out = use_tanh_out
        self.clamp_logvar = clamp_logvar
        self.attn_zero_init = attn_zero_init

        self.num_classes = None if num_classes is None else int(num_classes)
        self.null_label = (
            None if self.num_classes is None
            else int(self.num_classes if null_label is None else null_label)
        )
        self.class_decoder = bool(class_decoder)
        self.decoder_num_classes = (
            None if (not self.class_decoder or decoder_num_classes is None)
            else int(decoder_num_classes)
        )
        self.decoder_null_label = (
            None if self.decoder_num_classes is None
            else int(self.decoder_num_classes if decoder_null_label is None else decoder_null_label)
        )

        ch1 = base_ch          # full res
        ch2 = base_ch * 2      # half res
        ch4 = base_ch * 4      # quarter res / bottleneck

        azero = self.attn_zero_init  # shorthand for attention blocks

        # ================================================================
        #  ENCODER
        # ================================================================
        self.enc_conv_in = nn.Conv2d(img_channels, ch1, 3, 1, 1)

        # Stage 0 — full-res → half-res
        enc_stage0 = [VAEResBlock(ch1, ch1) for _ in range(num_res_blocks)]
        enc_stage0.append(nn.Conv2d(ch1, ch2, 3, 2, 1))           # stride-2 down

        # Stage 1 — half-res → quarter-res
        enc_stage1: list[nn.Module] = [VAEResBlock(ch2, ch2) for _ in range(num_res_blocks)]
        if encoder_attn_half:                                       # [4] NEW
            enc_stage1.append(VAEAttentionBlock(ch2, zero_init=azero))
        enc_stage1.append(nn.Conv2d(ch2, ch4, 3, 2, 1))           # stride-2 down

        # Stage 2 — bottleneck (quarter-res, with attention)
        enc_stage2: list[nn.Module] = [VAEResBlock(ch4, ch4)]
        enc_stage2.append(VAEAttentionBlock(ch4, zero_init=azero))
        for _ in range(num_res_blocks):
            enc_stage2.append(VAEResBlock(ch4, ch4))

        self.enc_blocks = nn.ModuleList([
            nn.Sequential(*enc_stage0),
            nn.Sequential(*enc_stage1),
            nn.Sequential(*enc_stage2),
        ])

        enc_out_ch = ch4

        # ---- conditional bottleneck bias --------------------------------
        if self.num_classes is not None:
            self.y_emb = nn.Embedding(self.num_classes + 1, cond_emb_dim)
            self.cond_proj = nn.Linear(cond_emb_dim, enc_out_ch)
            nn.init.zeros_(self.cond_proj.weight)
            nn.init.zeros_(self.cond_proj.bias)
        else:
            self.y_emb = None
            self.cond_proj = None

        # ---- latent projection (enc_out_ch → latent_channels) -----------
        if latent_proj_depth >= 2:
            self.enc_pre_proj = VAEResBlock(enc_out_ch, enc_out_ch)
        else:
            self.enc_pre_proj = None

        proj_in_ch = enc_out_ch + self.aux_d

        if self.conv3x3_proj:
            # [1] NEW: terminal norm → activation → 3×3 combined mu+logvar
            self.enc_norm_out = make_group_norm(proj_in_ch)
            self.enc_conv_out = nn.Conv2d(proj_in_ch, 2 * latent_channels, 3, 1, 1)
            # no separate mu / logvar convs
            self.mu = None
            self.logvar = None
        else:
            # legacy: separate 1×1 convs
            self.enc_norm_out = None
            self.enc_conv_out = None
            self.mu = nn.Conv2d(proj_in_ch, latent_channels, 1)
            self.logvar = nn.Conv2d(proj_in_ch, latent_channels, 1)

        # aux zero-init (so aux channels are ignored at init)
        if self.aux_d > 0:
            with torch.no_grad():
                if self.conv3x3_proj:
                    self.enc_conv_out.weight[:, enc_out_ch:, :, :].zero_()
                else:
                    self.mu.weight[:, enc_out_ch:, :, :].zero_()
                    self.logvar.weight[:, enc_out_ch:, :, :].zero_()

        # GroupNorm on mu
        if self.use_norm:
            self.gn_mu = nn.GroupNorm(num_groups=1, num_channels=latent_channels, affine=False)

        # ================================================================
        #  DECODER
        # ================================================================

        # ---- latent back-projection (latent_channels → ch4) -------------
        dec_in_ks = 3 if self.conv3x3_proj else 1                  # [2] NEW
        if latent_proj_depth >= 2:
            self.dec_conv_in = nn.Conv2d(latent_channels, ch4, dec_in_ks, 1, dec_in_ks // 2)
            self.dec_post_proj = VAEResBlock(ch4, ch4)
        else:
            self.dec_conv_in = nn.Conv2d(latent_channels, ch4, dec_in_ks, 1, dec_in_ks // 2)
            self.dec_post_proj = None

        # number of ResBlocks per decoder stage
        dec_nrb = num_res_blocks + (1 if decoder_extra_block else 0)   # [3] NEW

        # ================================================================
        #  TIME-DEPENDENT DECODER (TDD) v2
        #  - Per-ResBlock AdaGN (rec 1)
        #  - Log-SNR time embedding (rec 2, applied in decode())
        #  - Scale-dependent time gating (rec 3)
        #  - Time-conditioned attention (rec 4)
        #  - Deeper 3-layer time MLP (rec 5)
        # ================================================================
        self.time_cond_decoder = time_cond_decoder
        self.dec_schedule_type = None  # set externally after construction
        # Default t value used by decode() when time_cond_decoder=True but
        # t=None is passed — avoids a hard error and keeps the TDD path
        # in-distribution.  Callers can override via vae._decode_t_default.
        self._decode_t_default = 1e-4

        if time_cond_decoder:
            self.dec_time_emb_dim = dec_time_emb_dim

            # [Rec 5] Deeper time MLP (3-layer) for richer time representations
            self.dec_time_mlp = nn.Sequential(
                nn.Linear(dec_time_emb_dim, 4 * dec_time_emb_dim),
                nn.SiLU(),
                nn.Linear(4 * dec_time_emb_dim, 4 * dec_time_emb_dim),
                nn.SiLU(),
                nn.Linear(4 * dec_time_emb_dim, dec_time_emb_dim),
            )

            # [Rec 1] Per-ResBlock AdaGN: TimeCondVAEResBlock replaces VAEResBlock
            # [Rec 4] TimeCondAttentionBlock replaces VAEAttentionBlock

            # Stage 0 — bottleneck (quarter-res, with time-conditioned attention)
            self.dec_stage0_res = nn.ModuleList()
            self.dec_stage0_attn = nn.ModuleList()
            self.dec_stage0_res.append(TimeCondVAEResBlock(ch4, ch4, dec_time_emb_dim))
            self.dec_stage0_attn.append(TimeCondAttentionBlock(ch4, dec_time_emb_dim, zero_init=azero))
            for _ in range(dec_nrb):
                self.dec_stage0_res.append(TimeCondVAEResBlock(ch4, ch4, dec_time_emb_dim))

            # Stage 1 — quarter-res → half-res
            self.dec_stage1_up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(ch4, ch2, 3, 1, 1))
            self.dec_stage1_res = nn.ModuleList()
            for _ in range(dec_nrb):
                self.dec_stage1_res.append(TimeCondVAEResBlock(ch2, ch2, dec_time_emb_dim))
            self.dec_stage1_attn = nn.ModuleList()
            if decoder_attn_half:
                self.dec_stage1_attn.append(TimeCondAttentionBlock(ch2, dec_time_emb_dim, zero_init=azero))

            # Stage 2 — half-res → full-res
            self.dec_stage2_up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(ch2, ch1, 3, 1, 1))
            self.dec_stage2_res = nn.ModuleList()
            for _ in range(dec_nrb):
                self.dec_stage2_res.append(TimeCondVAEResBlock(ch1, ch1, dec_time_emb_dim))

            # [Rec 3] Scale-dependent time gating
            # Stage 0 (quarter-res): always active (no gate)
            # Stage 1 (half-res): gated — biased toward on
            # Stage 2 (full-res): gated — biased toward on
            # At large t (low SNR), fine-res stages soft-bypass; at t≈0 all fully active
            self.dec_stage_gate1 = nn.Linear(dec_time_emb_dim, 1)
            nn.init.zeros_(self.dec_stage_gate1.weight)
            nn.init.constant_(self.dec_stage_gate1.bias, 4.0)  # sigmoid(4) ≈ 0.98
            self.dec_stage_gate2 = nn.Linear(dec_time_emb_dim, 1)
            nn.init.zeros_(self.dec_stage_gate2.weight)
            nn.init.constant_(self.dec_stage_gate2.bias, 4.0)

            # Sentinel so legacy code paths don't crash
            self.dec_blocks = None
            self.dec_film_layers = None
        else:
            # --- Legacy non-time-conditioned decoder (unchanged) ---
            # Stage 0 — bottleneck (quarter-res, with attention)
            dec_stage0: list[nn.Module] = [VAEResBlock(ch4, ch4)]
            dec_stage0.append(VAEAttentionBlock(ch4, zero_init=azero))
            for _ in range(dec_nrb):
                dec_stage0.append(VAEResBlock(ch4, ch4))

            # Stage 1 — quarter-res → half-res
            dec_stage1: list[nn.Module] = [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch4, ch2, 3, 1, 1),
            ]
            for _ in range(dec_nrb):
                dec_stage1.append(VAEResBlock(ch2, ch2))
            if decoder_attn_half:
                dec_stage1.append(VAEAttentionBlock(ch2, zero_init=azero))

            # Stage 2 — half-res → full-res
            dec_stage2: list[nn.Module] = [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch2, ch1, 3, 1, 1),
            ]
            for _ in range(dec_nrb):
                dec_stage2.append(VAEResBlock(ch1, ch1))

            self.dec_blocks = nn.ModuleList([
                nn.Sequential(*dec_stage0),
                nn.Sequential(*dec_stage1),
                nn.Sequential(*dec_stage2),
            ])
            self.dec_film_layers = None

        if self.decoder_num_classes is not None:
            self.dec_y_emb = nn.Embedding(self.decoder_num_classes + 1, cond_emb_dim)
            dec_cond_out_dim = dec_time_emb_dim if time_cond_decoder else ch4
            self.dec_label_proj = nn.Linear(cond_emb_dim, dec_cond_out_dim)
            nn.init.zeros_(self.dec_label_proj.weight)
            nn.init.zeros_(self.dec_label_proj.bias)
        else:
            self.dec_y_emb = None
            self.dec_label_proj = None

        self.dec_out = nn.Sequential(
            nn.GroupNorm(16, ch1), nn.SiLU(), nn.Conv2d(ch1, img_channels, 3, 1, 1)
        )

    # -----------------------------------------------------------------
    #  encode / decode / forward  — signatures IDENTICAL to before
    # -----------------------------------------------------------------

    def encode(self, x: torch.Tensor, y: torch.Tensor | None = None):
        h = self.enc_conv_in(x)
        for block in self.enc_blocks:
            h = block(h)

        # class-conditioning at bottleneck
        if self.num_classes is not None:
            B, C, H, W = h.shape
            if y is None:
                y = torch.full((B,), self.null_label, device=h.device, dtype=torch.long)
            else:
                y = y.to(device=h.device, dtype=torch.long).view(B)
            emb = self.y_emb(y)
            bias = self.cond_proj(emb).view(B, C, 1, 1)
            h = h + bias

        if self.enc_pre_proj is not None:
            h = self.enc_pre_proj(h)

        # Auxiliary stochastic conditioning
        if self.aux_d > 0:
            B, _, H, W = h.shape
            w = torch.randn(B, self.aux_d, H, W, device=h.device, dtype=h.dtype)
            h = torch.cat([h, w], dim=1)

        # ── latent projection ──
        if self.conv3x3_proj:
            # [1] combined: GN → SiLU → 3×3 → split
            h = F.silu(self.enc_norm_out(h))
            moments = self.enc_conv_out(h)
            mu, logvar = moments.chunk(2, dim=1)
        else:
            # legacy: separate 1×1 convs
            mu = self.mu(h)
            logvar = self.logvar(h)

        # [6] logvar clamping
        if self.clamp_logvar:
            logvar = logvar.clamp(-30.0, 20.0)

        if self.use_norm:
            mu = self.gn_mu(mu)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _decoder_label_ids(self, y: torch.Tensor | None, B: int, device: torch.device) -> torch.Tensor | None:
        if self.decoder_num_classes is None:
            return None
        if y is None:
            return torch.full((B,), int(self.decoder_null_label), device=device, dtype=torch.long)
        y_ids = y.to(device=device, dtype=torch.long).view(-1)
        if y_ids.shape[0] != B:
            if y_ids.numel() == 1:
                y_ids = y_ids.expand(B)
            else:
                raise ValueError(f"Decoder labels batch mismatch: got {y_ids.shape[0]} labels for batch size {B}.")
        return y_ids

    def decode(self, z, t=None, y: torch.Tensor | None = None):
        h = self.dec_conv_in(z)
        B = h.shape[0]
        y_ids = self._decoder_label_ids(y, B, h.device)
        if self.dec_post_proj is not None:
            h = self.dec_post_proj(h)

        if self.time_cond_decoder and t is not None:
            # [Rec 2] Log-SNR time embedding instead of raw sinusoidal
            stype = getattr(self, 'dec_schedule_type', None) or "log_t"
            raw_emb = decoder_time_embedding(t, self.dec_time_emb_dim, schedule_type=stype)
            t_emb = self.dec_time_mlp(raw_emb)
            if y_ids is not None and self.dec_y_emb is not None and self.dec_label_proj is not None:
                t_emb = t_emb + self.dec_label_proj(self.dec_y_emb(y_ids))

            # ── Stage 0: bottleneck (quarter-res) — always active ──
            for j, res_block in enumerate(self.dec_stage0_res):
                h = res_block(h, t_emb)
                # Insert attention after the first ResBlock (mirrors encoder)
                if j == 0 and len(self.dec_stage0_attn) > 0:
                    h = self.dec_stage0_attn[0](h, t_emb)

            # ── Stage 1: quarter-res → half-res — scale-gated ──
            # [Rec 3] Scale gating: at large t, skip fine-res refinement
            gate1 = torch.sigmoid(self.dec_stage_gate1(t_emb))  # [B, 1]
            h = self.dec_stage1_up(h)
            h_up1 = h  # upsampled but un-refined (the "skip" path)
            for res_block in self.dec_stage1_res:
                h = res_block(h, t_emb)
            for attn_block in self.dec_stage1_attn:
                h = attn_block(h, t_emb)
            # Gated combination: gate≈1 → use refined, gate≈0 → use raw upsample
            h = gate1[:, :, None, None] * h + (1.0 - gate1[:, :, None, None]) * h_up1

            # ── Stage 2: half-res → full-res — scale-gated ──
            gate2 = torch.sigmoid(self.dec_stage_gate2(t_emb))  # [B, 1]
            h = self.dec_stage2_up(h)
            h_up2 = h  # upsampled but un-refined
            for res_block in self.dec_stage2_res:
                h = res_block(h, t_emb)
            h = gate2[:, :, None, None] * h + (1.0 - gate2[:, :, None, None]) * h_up2

        elif self.dec_blocks is not None:
            # --- Legacy non-time-conditioned path (t silently ignored) ---
            if y_ids is not None and self.dec_y_emb is not None and self.dec_label_proj is not None:
                h = h + self.dec_label_proj(self.dec_y_emb(y_ids)).view(B, h.shape[1], 1, 1)
            for block in self.dec_blocks:
                h = block(h)
        else:
            # time_cond_decoder=True but t=None: fall back to t_min so
            # the TDD path still runs (most in-distribution default).
            import warnings
            _t_default = float(getattr(self, '_decode_t_default', 1e-4))
            warnings.warn(
                f"time_cond_decoder=True but t=None; defaulting to t={_t_default:.2e}",
                stacklevel=2,
            )
            return self.decode(z, t=torch.full((z.shape[0],), _t_default, device=z.device), y=y)

        out = self.dec_out(h)
        return torch.tanh(out) if self.use_tanh_out else out        # [5]

    def forward(self, x, y: torch.Tensor | None = None):
        mu, logvar = self.encode(x, y=y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y=y), mu, logvar



# ---------------------------------------------------------------------------
# PatchGAN Discriminator (Stable Diffusion / pix2pix style)
# ---------------------------------------------------------------------------

class PatchDiscriminator(nn.Module):
    """Lightweight NLayerDiscriminator (PatchGAN) as used in SD / taming-transformers."""
    def __init__(self, in_channels: int = 3, ndf: int = 64, n_layers: int = 2):
        super().__init__()
        layers = [nn.Conv2d(in_channels, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        ch = ndf
        for i in range(1, n_layers):
            ch_next = min(ch * 2, ndf * 8)
            layers += [nn.Conv2d(ch, ch_next, 4, 2, 1), nn.LeakyReLU(0.2, True)]
            ch = ch_next
        ch_next = min(ch * 2, ndf * 8)
        layers += [nn.Conv2d(ch, ch_next, 4, 1, 1), nn.LeakyReLU(0.2, True)]
        ch = ch_next
        layers += [nn.Conv2d(ch, 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    return 0.5 * (F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean())


def hinge_g_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    return -logits_fake.mean()


# ---------------------------------------------------------------------------
# Time-conditioned PatchGAN + Wiener-reference target for TDD GAN training
# ---------------------------------------------------------------------------

def wiener_reference_x0(
    x0: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    *,
    alpha_min: float = 1e-4,
    max_var: float = 1e3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Approximate \mathbb{E}[x_0 | x_t] for a pixel-space Gaussian corruption at the same (alpha,sigma) level.

    We interpret the pixel forward process as:
        x_t = alpha * x0 + sigma * eps,   eps ~ N(0, I)

    so the (scaled) noisy observation is:
        y := x_t / alpha = x0 + (sigma/alpha) * eps.

    As a first-pass approximation, we draw one noisy observation y and apply a per-image,
    frequency-domain Wiener filter with known noise variance.
    """
    B, C, H, W = x0.shape
    a = alpha.clamp_min(alpha_min)
    noise_var = ((sigma / a) ** 2).clamp(max=max_var)  # [B,1,1,1]
    noise_std = torch.sqrt(noise_var)

    # Sample one pixel-space corruption (y = x0 + n)
    y = x0 + noise_std * torch.randn_like(x0)

    # Unitary FFT => white noise stays white with the same variance under norm='ortho'
    Y = torch.fft.rfft2(y, norm="ortho")
    Syy = (Y.real ** 2 + Y.imag ** 2)

    # Estimate signal power by subtracting known noise power
    Sxx = (Syy - noise_var).clamp_min(0.0)
    Hf = Sxx / (Sxx + noise_var + eps)

    x_hat = torch.fft.irfft2(Y * Hf, s=(H, W), norm="ortho")
    return x_hat


class TimeCondPatchDiscriminator(nn.Module):
    """PatchGAN discriminator with FiLM time-conditioning.

    The discriminator receives (x, t) and uses a log-SNR-based embedding (same helper as the TDD decoder),
    projecting it to per-layer (scale, shift) FiLM parameters applied after each conv.
    """
    def __init__(
        self,
        in_channels: int = 3,
        ndf: int = 64,
        n_layers: int = 2,
        time_emb_dim: int = 128,
        schedule_type: str = "log_t",
    ):
        super().__init__()
        self.schedule_type = str(schedule_type).lower()
        self.time_emb_dim = int(time_emb_dim)

        # Time embedding (match decoder's log-SNR base embedding)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_emb_dim, 4 * self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(4 * self.time_emb_dim, 4 * self.time_emb_dim),
        )

        # Conv tower + FiLM projections per layer
        self.convs = nn.ModuleList()
        self.films = nn.ModuleList()
        self.act = nn.LeakyReLU(0.2, True)

        # Layer 0
        self.convs.append(nn.Conv2d(in_channels, ndf, 4, 2, 1))
        self.films.append(nn.Linear(4 * self.time_emb_dim, 2 * ndf))
        ch = ndf

        # Downsampling layers
        for _ in range(1, int(n_layers)):
            ch_next = min(ch * 2, ndf * 8)
            self.convs.append(nn.Conv2d(ch, ch_next, 4, 2, 1))
            self.films.append(nn.Linear(4 * self.time_emb_dim, 2 * ch_next))
            ch = ch_next

        # Final conv (stride 1)
        ch_next = min(ch * 2, ndf * 8)
        self.convs.append(nn.Conv2d(ch, ch_next, 4, 1, 1))
        self.films.append(nn.Linear(4 * self.time_emb_dim, 2 * ch_next))
        ch = ch_next

        # Output head
        self.conv_out = nn.Conv2d(ch, 1, 4, 1, 1)

        # Initialize FiLM to identity at start
        for film in self.films:
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() != 1:
            t = t.view(-1)

        raw = decoder_time_embedding(t, self.time_emb_dim, schedule_type=self.schedule_type)
        t_emb = self.time_mlp(raw)

        h = x
        for conv, film in zip(self.convs, self.films):
            h = conv(h)
            scale, shift = film(t_emb).chunk(2, dim=1)
            h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
            h = self.act(h)
        return self.conv_out(h)



# -*- coding: utf-8 -*-
"""New_DIT.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KpCdfJsS9N-fQSLz1h6CKp1IGqsTyYyv
"""

# -*- coding: utf-8 -*-
"""New_DIT.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KpCdfJsS9N-fQSLz1h6CKp1IGqsTyYyv
"""

import math
from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """
    Standard sinusoidal timestep embedding.
    t: [B] float (can be continuous)
    returns: [B, dim]
    """
    if t.dim() != 1:
        t = t.view(-1)
    half = dim // 2
    # exp(-log(max_period) * i/half)
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / max(half, 1)
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimeEmbedding(nn.Module):
    """
    Small change: use a more standard sinusoidal embedding implementation,
    keep the same MLP width (4*dim) as your reference.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, 4 * dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = timestep_embedding(t, self.dim)
        return self.mlp(emb)


class AttentionBlock(nn.Module):
    """
    Multi-head self-attention (defaults to a conservative head count).
    Output projection is zero-initialized for stability (starts as no-op residual).
    """
    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        ch = int(ch)
        num_heads = int(num_heads)

        # Make heads divide channels (fallback to fewer heads if needed).
        num_heads = min(num_heads, ch)
        while num_heads > 1 and (ch % num_heads) != 0:
            num_heads -= 1

        self.ch = ch
        self.num_heads = num_heads
        self.head_dim = ch // num_heads

        self.norm = make_group_norm(ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, kernel_size=1)
        self.proj = nn.Conv2d(ch, ch, kernel_size=1)

        # Zero-init proj => attention path starts off as exactly zero.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)  # [B, 3C, H, W]
        qkv = qkv.view(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each [B, heads, head_dim, N]

        # Attention in fp32 for stability; cast back after softmax.
        attn = torch.matmul(q.transpose(-2, -1).float(), k.float())  # [B, heads, N, N]
        attn = attn * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1).to(q.dtype)

        out = torch.matmul(v, attn.transpose(-2, -1))  # [B, heads, head_dim, N]
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class ResBlock(nn.Module):
    """
    Minimal-but-stronger ResBlock:
      - FiLM time conditioning: (1+scale)*GN(h) + shift
      - optional dropout before conv2
      - zero-init conv2 so block starts as identity-ish
      - optional attention at the end (also stabilized by zero-init proj)
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        t_dim: int,
        *,
        dropout: float = 0.0,
        use_attn: bool = False,
        attn_heads: int = 4,
        skip_scale: float = 1.0,
    ):
        super().__init__()
        in_ch = int(in_ch)
        out_ch = int(out_ch)
        t_dim = int(t_dim)

        self.skip_scale = float(skip_scale)
        self.dropout = float(dropout)

        self.norm1 = make_group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        # Produce (scale, shift) for FiLM.
        self.time_proj = nn.Linear(t_dim, 2 * out_ch)

        self.norm2 = make_group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # Zero-init conv2 => residual path starts at 0.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

        self.attn = AttentionBlock(out_ch, num_heads=attn_heads) if use_attn else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # First conv
        h = self.conv1(F.silu(self.norm1(x)))

        # FiLM on second norm
        scale_shift = self.time_proj(t_emb).type_as(h)  # [B, 2*out_ch]
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.norm2(h)
        h = h * (1.0 + scale) + shift
        h = F.silu(h)

        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h)

        # Residual + (optional) attention + gentle scaling
        h = h + self.skip(x)
        h = self.attn(h)
        return h * self.skip_scale


# ---------------------------------------------------------------------------
# DiT (Diffusion Transformer) — drop-in replacement for UNetModel
# ---------------------------------------------------------------------------
# Same forward API:  forward(x, t, y) -> eps_pred
#   x: [B, C, H, W]  latent tensor
#   t: [B]            continuous time scalars
#   y: [B] | None     class labels (None → null/unconditional token)
#
# Architecture:  patchify → transformer blocks with adaLN-Zero → unpatchify
# Reference: Peebles & Xie, "Scalable Diffusion Models with Transformers" (2023)
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """RMSNorm with learnable scale (LightningDiT style)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D]
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU MLP: (SiLU(a) * b)W3."""
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        dim = int(dim)
        mlp_ratio = float(mlp_ratio)
        hidden = int((2.0 / 3.0) * mlp_ratio * dim)  # LightningDiT default: 2/3 * 4 * dim
        self.w12 = nn.Linear(dim, 2 * hidden)
        self.w3 = nn.Linear(hidden, dim)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w12(x).chunk(2, dim=-1)
        x = F.silu(a) * b
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.w3(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero, RMSNorm, and SwiGLU."""
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads

        # --- Self-attention ---
        self.norm1 = RMSNorm(self.hidden_dim, eps=1e-6)
        self.attn_qkv = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)
        self.attn_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # --- Feed-forward ---
        self.norm2 = RMSNorm(self.hidden_dim, eps=1e-6)
        self.mlp = SwiGLU(self.hidden_dim, mlp_ratio=mlp_ratio, dropout=dropout)

        # --- adaLN-Zero modulation: produces 6 * hidden_dim from conditioning ---
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 6 * self.hidden_dim),
        )
        # Zero-init so block starts as identity
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D], c: [B, D]."""
        mod = self.adaLN_modulation(c).unsqueeze(1)  # [B, 1, 6D]
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        # --- Attention branch ---
        h = self.norm1(x)
        h = h * (1.0 + gamma1) + beta1

        B, N, D = h.shape
        qkv = self.attn_qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        h = (attn @ v).transpose(1, 2).reshape(B, N, D)
        h = self.attn_proj(h)
        x = x + alpha1 * h

        # --- MLP branch ---
        h = self.norm2(x)
        h = h * (1.0 + gamma2) + beta2
        h = self.mlp(h)
        x = x + alpha2 * h

        return x



class DiTModel(nn.Module):
    """Diffusion Transformer (DiT) configured for strong diffusion/flow training (eps prediction).

    - patchify -> transformer blocks with adaLN-Zero -> unpatchify
    - RMSNorm + SwiGLU (LightningDiT-style)
    - fixed 2D sin-cos positional embeddings
    - outputs an eps prediction ε̂(z_t, t)

    Forward signature stays: forward(x, t, y) -> [B,C,H,W]
    """
    def __init__(
        self,
        in_channels: int = 4,
        patch_size: int = 1,
        hidden_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_classes: int | None = None,
        *,
        dropout: float = 0.0,
        latent_size: int = 8,
        # accepted (ignored) for backward compat with UNetModel call sites
        base_channels: int = 32,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_levels: Optional[Tuple[int, ...]] = None,
        attn_heads: int = 4,
        skip_scale: float = 1.0,
        mid_attn: bool = True,
        factored_head: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.factored_head = bool(factored_head)

        self.num_classes = num_classes
        self.null_label = num_classes if num_classes is not None else None

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            self.in_channels, self.hidden_dim,
            kernel_size=self.patch_size, stride=self.patch_size,
        )

        # Time embedding (sinusoidal -> MLP)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Class embedding
        if num_classes is not None:
            self.label_emb = nn.Embedding(int(num_classes) + 1, self.hidden_dim)
        else:
            self.label_emb = None

        # Fixed 2D sin-cos positional embedding
        grid = int(latent_size // self.patch_size)
        num_tokens = grid * grid
        pos = self._build_2d_sincos_pos_embed(self.hidden_dim, grid)  # [1, N, D]
        self.register_buffer("pos_embed", pos, persistent=False)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(self.hidden_dim, int(num_heads), mlp_ratio=float(mlp_ratio), dropout=float(dropout))
            for _ in range(int(depth))
        ])

        # Final layer: adaLN modulation + projection back to pixels
        self.final_norm = RMSNorm(self.hidden_dim, eps=1e-6)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
        )

        patch_out_dim = self.in_channels * self.patch_size * self.patch_size
        if self.factored_head:
            # Factored Natural-Parameter head (λ, ν):
            #   prec_proj  -> predicts log(λ), where λ = σₜ · diag(Σₜ(x)⁻¹)
            #   nu_proj    -> predicts ν = λ ⊙ μₜ  (precision-weighted mean)
            #   output  ε̂ = λ ⊙ zₜ − ν
            # The aggregate score is always s*(z,t) = -Λ_eff z + ν_eff,
            # so this is the exact functional form with no approximation.
            self.nu_proj   = nn.Linear(self.hidden_dim, patch_out_dim)
            self.prec_proj = nn.Linear(self.hidden_dim, patch_out_dim)
        else:
            self.final_proj = nn.Linear(self.hidden_dim, patch_out_dim)

        self.initialize_weights()

    @staticmethod
    def _build_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
        """Return fixed 2D sin-cos positional embedding as torch Tensor [1, N, D]."""
        # Based on common MAE/ViT utilities
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # (w, h)
        grid = np.stack(grid, axis=0)  # [2, Gh, Gw]
        grid = grid.reshape(2, 1, grid_size, grid_size)

        def get_1d(embed_dim_1d: int, pos: np.ndarray) -> np.ndarray:
            assert embed_dim_1d % 2 == 0
            omega = np.arange(embed_dim_1d // 2, dtype=np.float32)
            omega /= (embed_dim_1d / 2.0)
            omega = 1.0 / (10000 ** omega)  # [D/2]
            pos = pos.reshape(-1)  # [M]
            out = np.einsum("m,d->md", pos, omega)  # [M, D/2]
            emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)  # [M, D]
            return emb

        assert embed_dim % 2 == 0, "embed_dim must be even for 2D sin-cos"
        emb_h = get_1d(embed_dim // 2, grid[1])  # [N, D/2]
        emb_w = get_1d(embed_dim // 2, grid[0])  # [N, D/2]
        pos_embed = np.concatenate([emb_w, emb_h], axis=1)  # [N, D]
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)  # [1, N, D]
        return pos_embed

    def initialize_weights(self) -> None:
        """LightningDiT-style init: Xavier on linears/convs, zero on adaLN/final."""
        def _init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if m.weight is not None and m.weight.dim() > 1:
                    nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None and m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)

        # Zero-init all adaLN modulations + final projection (identity start)
        for blk in self.blocks:
            nn.init.zeros_(blk.adaLN_modulation[1].weight)
            nn.init.zeros_(blk.adaLN_modulation[1].bias)
        nn.init.zeros_(self.final_modulation[1].weight)
        nn.init.zeros_(self.final_modulation[1].bias)
        if self.factored_head:
            nn.init.zeros_(self.nu_proj.weight)
            nn.init.zeros_(self.nu_proj.bias)
            nn.init.zeros_(self.prec_proj.weight)
            # Bias init: exp(-4) ≈ 0.018, so initial ε ≈ 0.018·zₜ
            # (near-zero output at init while keeping grad alive to both heads)
            nn.init.constant_(self.prec_proj.bias, -4.0)
        else:
            nn.init.zeros_(self.final_proj.weight)
            nn.init.zeros_(self.final_proj.bias)

    def unpatchify(self, x: torch.Tensor, H_tok: int, W_tok: int) -> torch.Tensor:
        p = self.patch_size
        C = self.in_channels
        x = x.reshape(-1, H_tok, W_tok, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(-1, C, H_tok * p, W_tok * p)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None, *, return_components: bool = False, detach_components: bool = False):
        """Return eps prediction ε̂(z_t, t).

        When factored_head=True the prediction uses natural parameters:
            ε̂ = λ ⊙ z_t − ν
        where λ = σₜ·Σₜ⁻¹ (diagonal precision) and ν = λ⊙μₜ.

        If return_components=True and factored_head=True, returns (eps, lam, nu).
        If detach_components=True, (lam, nu) are computed from detached trunk tokens
        so auxiliary losses can update only the head without backpropagating to the encoder.
        """
        B, C, H, W = x.shape
        z_t = x                      # keep original input for factored combination
        t = t.view(-1)

        tokens = self.patch_embed(x)  # [B, D, H_tok, W_tok]
        H_tok, W_tok = tokens.shape[2], tokens.shape[3]
        tokens = tokens.flatten(2).transpose(1, 2)  # [B, N, D]

        tokens = tokens + self.pos_embed

        # conditioning vector c = time_emb + class_emb
        t_emb = timestep_embedding(t, self.hidden_dim)
        c = self.time_mlp(t_emb)

        if self.label_emb is not None:
            if y is None:
                y_ids = torch.full((B,), int(self.null_label), device=x.device, dtype=torch.long)
            else:
                y_ids = y.to(device=x.device, dtype=torch.long).view(-1)
                if y_ids.shape[0] != B:
                    y_ids = y_ids.expand(B)
            c = c + self.label_emb(y_ids)

        for blk in self.blocks:
            tokens = blk(tokens, c)

        # final adaLN modulation
        gamma, beta = self.final_modulation(c).unsqueeze(1).chunk(2, dim=-1)
        tokens = self.final_norm(tokens) * (1.0 + gamma) + beta

        if self.factored_head:
            # Natural-parameter form: ε̂ = λ ⊙ zₜ − ν
            nu_hat  = self.unpatchify(self.nu_proj(tokens), H_tok, W_tok)
            log_lam = self.unpatchify(self.prec_proj(tokens), H_tok, W_tok)
            log_lam = log_lam.clamp(-20.0, 20.0)
            lam     = torch.exp(log_lam)
            out = lam * z_t - nu_hat

            if return_components:
                if detach_components:
                    tokens_aux = tokens.detach()
                    nu_aux  = self.unpatchify(self.nu_proj(tokens_aux), H_tok, W_tok)
                    log_lam_aux = self.unpatchify(self.prec_proj(tokens_aux), H_tok, W_tok)
                    log_lam_aux = log_lam_aux.clamp(-20.0, 20.0)
                    lam_aux = torch.exp(log_lam_aux)
                else:
                    lam_aux = lam
                    nu_aux  = nu_hat
                return out, lam_aux, nu_aux
        else:
            tokens = self.final_proj(tokens)
            out = self.unpatchify(tokens, H_tok, W_tok)
            if return_components:
                raise ValueError("return_components=True requires factored_head=True")

        return out


# Alias so that existing code referencing UNetModel still works
UNetModel = DiTModel

class UniversalSampler:
    def __init__(
        self,
        method: str = "heun_sde",
        num_steps: int = 20,
        t_min: float = 2e-5,
        t_max: float = 2.0,
        schedule_cfg: Dict[str, Any] | None = None,
        ddim_eta: float = 0.0,
        schedule_type: str = "log_snr",
        cosine_s: float = 0.008,
        readout_mode: str = "direct",
        frontier_tracker: "ReconFrontierTracker | None" = None,
        init_mode: str = "prior",
        cfg_mode: str = "constant",
    ):
        self.num_steps = int(num_steps)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.method = str(method).lower()
        self.schedule_type = canonicalize_time_schedule(schedule_type, default="log_snr")
        self.cosine_s = float(cosine_s)
        self.frontier_tracker = frontier_tracker
        self.init_mode = canonicalize_init_mode(init_mode, default="prior")
        self.cfg_mode = canonicalize_cfg_mode(cfg_mode, default="constant")

        # --- Composed-decoder readout mode ---
        # Controls how z_t at the final integration step is converted to the
        # latent fed into the decoder:
        #   "direct"       : return z_{t_min} as-is (current default behaviour).
        #                    Caller decodes at decode_time (≈ t_min).
        #   "conditional"  : apply single-model conditional Tweedie readout at
        #                    t_final using y (no CFG).  Returns (z_hat_0, t_final).
        #                    Recommended when transport uses CFG but decoder was
        #                    trained on single-model score predictions.
        #   "cfg"          : apply CFG Tweedie readout (same cfg_scale as transport).
        #                    Out-of-distribution for the decoder at high guidance
        #                    scales — use for ablation only.
        #   "unconditional": apply unconditional (y=None) Tweedie readout.
        #                    Useful as a baseline / for unconditional generation.
        assert readout_mode in ("direct", "conditional", "cfg", "unconditional"), \
            f"Unknown readout_mode={readout_mode!r}. Expected 'direct', 'conditional', 'cfg', or 'unconditional'."
        self.readout_mode = str(readout_mode)

        # For discrete samplers (DDIM), we need the same schedule used in training.
        self.schedule_cfg = schedule_cfg
        self.ddim_eta = float(ddim_eta)
        self._ddpm_schedule: Dict[str, torch.Tensor] | None = None

    # -------------------- schedule-aware helpers --------------------

    def _get_alpha_sigma(self, t_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (alpha, sigma) at continuous time t_vec, dispatching on schedule type."""
        if self.schedule_type in ("flow", "flow_matching"):
            # linear interpolant
            return get_flow_params(t_vec)
        if self.schedule_type == "cosine":
            a, s, _ = get_cosine_params(t_vec, cosine_s=self.cosine_s)
            return a, s
        # linear / log_t / log_snr — all use OU params
        return get_ou_params(t_vec)

    def _get_beta(self, t_vec: torch.Tensor) -> torch.Tensor:
        """Return instantaneous beta(t) — only meaningful for cosine VP."""
        if self.schedule_type == "cosine":
            _, _, b = get_cosine_params(t_vec, cosine_s=self.cosine_s)
            return b
        else:
            # For OU, beta is formally 2 (constant), but the ODE form is different.
            return torch.full_like(t_vec, 2.0)

    def _make_time_grid(self, device: torch.device) -> torch.Tensor:
        """Build a time grid for sampling.

        - If an active ``frontier_tracker`` is attached, build the grid by
          inverting its weight CDF so that integration steps concentrate at the
          information frontier.  This matches the adaptive training distribution.
        - OU / cosine samplers run **reverse-time**: t_max -> t_min (descending grid).
        - Flow matching runs **forward-time**: t_min -> t_max (ascending grid).
        """
        N = self.num_steps + 1

        # --- Adaptive frontier grid (overrides base schedule) ---
        if self.frontier_tracker is not None and self.frontier_tracker.is_active:
            descending = self.schedule_type not in ("flow", "flow_matching")
            return self.frontier_tracker.make_adaptive_time_grid(
                self.num_steps, device, descending=descending,
                t_lo=self.t_min, t_hi=self.t_max,
            )

        if self.schedule_type in ("flow", "flow_matching"):
            return torch.linspace(self.t_min, self.t_max, N, device=device)
        if self.schedule_type == "cosine":
            return torch.linspace(self.t_max, self.t_min, N, device=device)
        if self.schedule_type == "log_snr":
            lam_min_val = ou_logsnr(torch.tensor(self.t_max, dtype=torch.float64)).item()
            lam_max_val = ou_logsnr(torch.tensor(self.t_min, dtype=torch.float64)).item()
            lam_grid = torch.linspace(lam_min_val, lam_max_val, N, device=device, dtype=torch.float64)
            times = ou_time_from_logsnr(lam_grid).float()
            return times.clamp(self.t_min, self.t_max)
        if self.schedule_type == "linear":
            return torch.linspace(self.t_max, self.t_min, N, device=device)
        # log_t
        return torch.logspace(math.log10(self.t_max), math.log10(self.t_min), N, device=device)

    def _resolve_cfg_scale(self, cfg_scale: float | None, t_vec: torch.Tensor) -> torch.Tensor | None:
        """Return the effective per-sample CFG scale at time ``t_vec``."""
        if cfg_scale is None:
            return None

        cfg_scale_val = float(cfg_scale)
        if self.cfg_mode == "constant":
            return torch.full_like(t_vec, cfg_scale_val)

        denom = max(self.t_max - self.t_min, 1e-12)
        ramp = ((t_vec - self.t_min) / denom).clamp(0.0, 1.0)
        return 1.0 + (cfg_scale_val - 1.0) * ramp

    def _predict_eps(
        self,
        unet,
        x: torch.Tensor,
        t_vec: torch.Tensor,
        y: torch.Tensor | None = None,
        cfg_scale: float | None = None,
    ) -> torch.Tensor:
        """Classifier-Free Guidance in eps-parameterization, with optional time-varying CFG."""
        if cfg_scale is None or cfg_scale <= 0.0 or y is None:
            return unet(x, t_vec, y)

        cfg_scale_t = self._resolve_cfg_scale(cfg_scale, t_vec)
        eps_uncond = unet(x, t_vec, None)
        eps_cond = unet(x, t_vec, y)
        cfg_scale_t = cfg_scale_t.view(-1, *([1] * (x.ndim - 1)))
        return eps_uncond + cfg_scale_t * (eps_cond - eps_uncond)

    # ---------- factored-head helpers (exponential integrator) ----------

    def _predict_components(
        self,
        unet,
        x: torch.Tensor,
        t_vec: torch.Tensor,
        y: torch.Tensor | None = None,
        cfg_scale: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict (eps, lam, nu) from a factored-head model, with optional time-varying CFG.

        Requires unet.factored_head == True.  The CFG combination is exact
        because eps = lam * z - nu is bilinear in (lam, nu) for fixed z.
        """
        if not getattr(unet, "factored_head", False):
            raise ValueError(
                "Exponential integrator samplers (exp_euler_ode / exp_heun_ode) "
                "require a model with factored_head=True so that the precision "
                "and natural-mean heads are exposed."
            )

        if cfg_scale is None or cfg_scale <= 0.0 or y is None:
            return unet(x, t_vec, y, return_components=True)

        cfg_scale_t = self._resolve_cfg_scale(cfg_scale, t_vec)
        eps_u, lam_u, nu_u = unet(x, t_vec, None, return_components=True)
        eps_c, lam_c, nu_c = unet(x, t_vec, y, return_components=True)
        cfg_scale_t = cfg_scale_t.view(-1, *([1] * (x.ndim - 1)))
        eps = eps_u + cfg_scale_t * (eps_c - eps_u)
        lam = lam_u + cfg_scale_t * (lam_c - lam_u)
        nu = nu_u + cfg_scale_t * (nu_c - nu_u)
        return eps, lam, nu

    @staticmethod
    def _phi1(x: torch.Tensor) -> torch.Tensor:
        r"""Compute the entire function \varphi_1(x) = (e^x - 1) / x.

        Uses a 4th-order Taylor expansion for |x| < 1e-4 to avoid 0/0.
        """
        small = x.abs() < 1e-4
        # Taylor: 1 + x/2 + x^2/6 + x^3/24
        taylor = 1.0 + x * (0.5 + x * (1.0 / 6.0 + x / 24.0))
        # For |x| >= 1e-4 the direct formula is fine
        x_safe = torch.where(small, torch.ones_like(x), x)
        exact = torch.expm1(x_safe) / x_safe
        return torch.where(small, taylor, exact)

    def _exp_integrate(
        self,
        z: torch.Tensor,
        lam: torch.Tensor,
        nu: torch.Tensor,
        t_curr: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        r"""Exact diagonal exponential integration over one frozen-coefficient step.

        The probability-flow ODE with factored score takes the form (per channel)

            dz_i / dt  =  A_i \cdot z_i  +  C_i

        where A (diagonal) and C depend on the schedule type:

        * OU:     A_i = lam_i / sigma - 1,      C_i = -nu_i / sigma
        * Cosine: A_i = 0.5 beta (lam_i/sigma - 1),  C_i = -0.5 beta nu_i / sigma

        With A and C frozen, the exact solution is

            z_i(t + dt) = exp(A_i dt) z_i(t)  +  phi_1(A_i dt) dt C_i
        """
        B = z.shape[0]
        t_vec = t_curr.expand(B)

        _, sigma = self._get_alpha_sigma(t_vec.view(B, 1, 1, 1))
        inv_sigma = 1.0 / (sigma + 1e-10)

        if self.schedule_type in ("flow", "flow_matching"):
            raise ValueError(
                "Exponential integrator is not supported for flow matching schedules."
            )

        if self.schedule_type == "cosine":
            beta_t = self._get_beta(t_vec.view(B, 1, 1, 1))
            half_beta = 0.5 * beta_t
            A = half_beta * (lam * inv_sigma - 1.0)
            C = -half_beta * nu * inv_sigma
        else:
            # OU (log_t or log_snr): dz/dt = -z + eps/sigma
            # with eps = lam*z - nu  =>  dz/dt = (lam/sigma - 1)*z - nu/sigma
            A = lam * inv_sigma - 1.0
            C = -nu * inv_sigma

        Ah = A * dt
        return torch.exp(Ah) * z + self._phi1(Ah) * dt * C

    # ------------------------- Continuous-time samplers -------------------------

    def get_ode_derivative(self, x, t, unet, y=None, cfg_scale=None):
        """ODE derivative.

        - Flow matching: dz/dt = v(z,t)
        - OU/cosine: probability-flow ODE (eps parameterization)
        """
        B = x.shape[0]
        t_vec = t.expand(B)

        if self.schedule_type in ("flow", "flow_matching"):
            # Model predicts eps; convert to velocity via z_t = eps + t * v.
            eps_pred = self._predict_eps(unet, x, t_vec, y=y, cfg_scale=cfg_scale)
            t_b = t_vec.view(B, 1, 1, 1).clamp_min(1e-5)
            v_pred = (x - eps_pred) / t_b
            return v_pred

        eps_pred = self._predict_eps(unet, x, t_vec, y=y, cfg_scale=cfg_scale)
        alpha, sigma = self._get_alpha_sigma(t_vec.view(B, 1, 1, 1))
        inv_sigma = 1.0 / (sigma + 1e-10)

        if self.schedule_type == "cosine":
            beta_t = self._get_beta(t_vec.view(B, 1, 1, 1))
            return -0.5 * beta_t * (x - inv_sigma * eps_pred)
        # OU (log_t or log_snr)
        return -x + inv_sigma * eps_pred

    def get_rev_sde_drift(self, x, t, unet, y=None, cfg_scale=None):
        """Reverse-time SDE drift (OU/cosine only)."""
        if self.schedule_type in ("flow", "flow_matching"):
            raise ValueError("Reverse-time SDE drift is not defined for flow matching. Use an ODE sampler (rk4_ode/heun_ode/euler_ode).")

        B = x.shape[0]
        t_vec = t.expand(B)
        eps_pred = self._predict_eps(unet, x, t_vec, y=y, cfg_scale=cfg_scale)
        alpha, sigma = self._get_alpha_sigma(t_vec.view(B, 1, 1, 1))
        inv_sigma = 1.0 / (sigma + 1e-8)

        if self.schedule_type == "cosine":
            beta_t = self._get_beta(t_vec.view(B, 1, 1, 1))
            return -0.5 * beta_t * x + beta_t * inv_sigma * eps_pred
        return -x + 2.0 * inv_sigma * eps_pred

    def step_euler_ode(self, x, t_curr, t_next, unet, y=None, cfg_scale=None):
        dt = t_next - t_curr
        d_curr = self.get_ode_derivative(x, t_curr, unet, y=y, cfg_scale=cfg_scale)
        return x + dt * d_curr

    def step_rk4_ode(self, x, t_curr, t_next, unet, y=None, cfg_scale=None):
        dt = t_next - t_curr
        half_dt = dt * 0.5
        t_half = t_curr + half_dt

        k1 = self.get_ode_derivative(x, t_curr, unet, y=y, cfg_scale=cfg_scale)
        k2 = self.get_ode_derivative(x + half_dt * k1, t_half, unet, y=y, cfg_scale=cfg_scale)
        k3 = self.get_ode_derivative(x + half_dt * k2, t_half, unet, y=y, cfg_scale=cfg_scale)
        k4 = self.get_ode_derivative(x + dt * k3, t_next, unet, y=y, cfg_scale=cfg_scale)

        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def step_heun_ode(self, x, t_curr, t_next, unet, y=None, cfg_scale=None):
        B = x.shape[0]
        dt = t_next - t_curr

        d_curr = self.get_ode_derivative(x, t_curr, unet, y=y, cfg_scale=cfg_scale)
        x_proposed = x + dt * d_curr

        if t_next > self.t_min:
            d_next = self.get_ode_derivative(x_proposed, t_next, unet, y=y, cfg_scale=cfg_scale)
            x = x + 0.5 * dt * (d_curr + d_next)
        else:
            x = x_proposed

        return x

    def step_heun_sde(self, x, t_curr, t_next, unet, y=None, cfg_scale=None, generator=None):
        """Heun SDE step with schedule-aware diffusion coefficient.

        OU:     diffusion = sqrt(2)
        Cosine: diffusion = sqrt(beta(t))
        """
        dt = t_next - t_curr
        dt_abs = torch.abs(dt).clamp_min(1e-12)

        if generator is None:
            noise = torch.randn_like(x)
        else:
            noise = torch.randn(x.shape, device=x.device, generator=generator)

        B = x.shape[0]
        if self.schedule_type == "cosine":
            beta_t = self._get_beta(t_curr.expand(B).view(B, 1, 1, 1))
            diff_coeff = torch.sqrt(beta_t.clamp_min(1e-8))
        else:  # OU: constant diffusion sqrt(2)
            diff_coeff = math.sqrt(2.0)

        dW = diff_coeff * torch.sqrt(dt_abs) * noise

        b_curr = self.get_rev_sde_drift(x, t_curr, unet, y=y, cfg_scale=cfg_scale)
        x_hat = x + dt * b_curr + dW

        b_next = self.get_rev_sde_drift(x_hat, t_next, unet, y=y, cfg_scale=cfg_scale)
        x_new = x + 0.5 * dt * (b_curr + b_next) + dW
        return x_new

    # --------------- Diagonal exponential integrators ---------------

    def step_exp_euler_ode(self, x, t_curr, t_next, unet, y=None, cfg_scale=None):
        """First-order diagonal exponential integrator (1 NFE per step).

        Freezes the learned (lam, nu) at (x, t_curr) and exactly integrates
        the resulting constant-coefficient diagonal-linear ODE over [t_curr, t_next].

        Requires a model with ``factored_head=True``.
        """
        B = x.shape[0]
        t_vec = t_curr.expand(B)
        _eps, lam, nu = self._predict_components(unet, x, t_vec, y=y, cfg_scale=cfg_scale)
        dt = t_next - t_curr
        return self._exp_integrate(x, lam, nu, t_curr, dt)

    def step_exp_heun_ode(self, x, t_curr, t_next, unet, y=None, cfg_scale=None):
        """Second-order exponential Heun integrator (2 NFE per step).

        1. Evaluate (lam, nu) at (x, t_curr); exp-Euler predict x_hat at t_next.
        2. Evaluate (lam, nu) at (x_hat, t_next).
        3. Average the two sets of natural parameters.
        4. Re-integrate from x using the averaged coefficients.

        This exactly treats per-channel stiffness at the cost of two forward
        passes (same as standard Heun) and matches the second-order exponential
        Heun variant described in Section 5.1 of the precision-head note.

        Requires a model with ``factored_head=True``.
        """
        B = x.shape[0]
        dt = t_next - t_curr

        # --- predictor: exp-Euler from t_curr ---
        t_vec_curr = t_curr.expand(B)
        _eps_curr, lam_curr, nu_curr = self._predict_components(
            unet, x, t_vec_curr, y=y, cfg_scale=cfg_scale,
        )
        x_hat = self._exp_integrate(x, lam_curr, nu_curr, t_curr, dt)

        # At the terminal step just return the Euler prediction
        if (t_next <= self.t_min).item():
            return x_hat

        # --- corrector: evaluate at predicted point ---
        t_vec_next = t_next.expand(B)
        _eps_next, lam_next, nu_next = self._predict_components(
            unet, x_hat, t_vec_next, y=y, cfg_scale=cfg_scale,
        )

        # Average natural parameters and re-integrate from x
        lam_avg = 0.5 * (lam_curr + lam_next)
        nu_avg = 0.5 * (nu_curr + nu_next)
        return self._exp_integrate(x, lam_avg, nu_avg, t_curr, dt)

    # ------------------------- Discrete DDIM sampler -------------------------

    def _get_ddpm_schedule(self, device: torch.device) -> Dict[str, torch.Tensor]:
        if self._ddpm_schedule is None or self._ddpm_schedule["betas"].device != device:
            if self.schedule_cfg is None:
                raise ValueError("DDIM sampling requires schedule_cfg (pass the same cfg used for training).")
            self._ddpm_schedule = make_ddpm_schedule(self.schedule_cfg, device)
        return self._ddpm_schedule


    def step_ddim(
        self,
        x: torch.Tensor,
        t_curr: torch.Tensor,          # scalar tensor on device
        t_next: torch.Tensor,          # scalar tensor on device (smaller than t_curr)
        unet,
        y: torch.Tensor | None = None,
        cfg_scale: float | None = None,
        generator=None,
    ) -> torch.Tensor:
        """DDIM step using schedule-aware alpha/sigma (works for OU and cosine)."""
        B = x.shape[0]

        t_vec = t_curr.expand(B)
        eps_pred = self._predict_eps(unet, x, t_vec, y=y, cfg_scale=cfg_scale)

        alpha_t, sigma_t = self._get_alpha_sigma(t_vec.view(B, 1, 1, 1))
        alpha_t = alpha_t.to(x.dtype)
        sigma_t = sigma_t.to(x.dtype)

        # x0 prediction
        x0_pred = (x - sigma_t * eps_pred) / (alpha_t + 1e-8)

        # If we're at (or below) the terminal noise level, return the denoised prediction
        if (t_next <= self.t_min).item():
            return x0_pred

        t_next_vec = t_next.expand(B)
        alpha_next, sigma_next = self._get_alpha_sigma(t_next_vec.view(B, 1, 1, 1))
        alpha_next = alpha_next.to(x.dtype)
        sigma_next = sigma_next.to(x.dtype)

        # DDIM-style stochasticity using a(t) = alpha(t)^2
        a_t = alpha_t ** 2
        a_next = alpha_next ** 2

        eta = float(self.ddim_eta)
        if eta <= 0.0:
            sigma_ddim = torch.zeros_like(sigma_next)
        else:
            denom = (1.0 - a_t).clamp_min(1e-12)
            term1 = ((1.0 - a_next) / denom).clamp_min(0.0)
            term2 = (1.0 - (a_t / (a_next + 1e-12))).clamp_min(0.0)
            sigma_ddim = eta * torch.sqrt(term1 * term2)

        # direction coefficient
        dir_coeff = torch.sqrt((1.0 - a_next - sigma_ddim ** 2).clamp_min(0.0))

        if eta > 0.0:
            if generator is None:
                noise = torch.randn_like(x)
            else:
                noise = torch.randn(x.shape, device=x.device, generator=generator)
            x_next = alpha_next * x0_pred + dir_coeff * eps_pred + sigma_ddim * noise
        else:
            x_next = alpha_next * x0_pred + dir_coeff * eps_pred

        return x_next

    # --------------- Composed-decoder readout ---------------

    def _apply_readout(
        self,
        z_t: torch.Tensor,
        t_final: torch.Tensor,
        unet,
        y: torch.Tensor | None = None,
        cfg_scale: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the composed-decoder Tweedie readout at the final integration step.

        The composed decoder D_t(z_t) = D( (z_t - sigma*eps) / alpha, t ) was
        the *actual* training-time decoder objective.  This method computes the
        Tweedie z_hat_0 = (z_t - sigma*eps) / alpha using the score head with
        the mode specified by self.readout_mode.

        Transport (ODE/SDE integration) always uses cfg_scale for the velocity
        field.  The readout_mode controls ONLY the final projection:

          "direct"       → no Tweedie, return (z_t, None)
          "conditional"  → eps = unet(z_t, t, y)           [single-model, in-distribution]
          "cfg"          → eps via CFG at cfg_scale         [OOD at high w, ablation only]
          "unconditional"→ eps = unet(z_t, t, None)         [unconditional baseline]

        Returns (z_out, readout_t):
          - z_out:     the latent to pass to vae.decode
          - readout_t: the time to pass to the TDD decoder (t_final for composed,
                       None for direct → caller uses its default decode_time)
        """
        if self.readout_mode == "direct":
            return z_t, None

        B = z_t.shape[0]
        t_vec = t_final.expand(B)
        alpha, sigma = self._get_alpha_sigma(t_vec.view(B, 1, 1, 1))

        if self.readout_mode == "conditional":
            # Single-model conditional prediction — in-distribution for the decoder
            eps_readout = unet(z_t, t_vec, y)
        elif self.readout_mode == "cfg":
            # CFG prediction — same as transport; potentially OOD for decoder at high w
            eps_readout = self._predict_eps(unet, z_t, t_vec, y=y, cfg_scale=cfg_scale)
        elif self.readout_mode == "unconditional":
            # Unconditional prediction — y=None
            eps_readout = unet(z_t, t_vec, None)
        else:
            raise ValueError(f"Unknown readout_mode: {self.readout_mode!r}")

        z_hat_0 = (z_t - sigma * eps_readout) / (alpha + 1e-8)
        return z_hat_0, t_final

    def sample(
        self,
        unet,
        shape=None,
        device=None,
        x_init=None,
        generator=None,
        y: torch.Tensor | None = None,
        cfg_scale: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Run the full sampling loop and apply the composed-decoder readout.

        Returns
        -------
        (z, readout_t) : Tuple[Tensor, Tensor | None]
            z         – latent to feed to vae.decode.
                        For readout_mode='direct', this is z_{t_min} (raw ODE/SDE output).
                        For composed modes, this is z_hat_0 from Tweedie at t_final.
            readout_t – scalar tensor: the time at which the readout was applied.
                        None when readout_mode='direct' (caller should use its own decode_time).
                        For composed modes, this is the final time in the integration grid.
        """
        unet.eval()

        if x_init is None:
            if self.init_mode == "oracle":
                raise ValueError("UniversalSampler(init_mode='oracle') requires x_init to be provided.")
            assert shape is not None and device is not None
            x = torch.randn(shape, device=device, generator=generator)
        else:
            x = x_init
        device = x.device

        # Build descending time grid matching the training schedule type
        ts = self._make_time_grid(device)

        # --- Discrete DDIM path ---
        if self.method == "ddim":
            for i in range(self.num_steps):
                t_curr = ts[i]
                t_next = ts[i + 1]
                x = self.step_ddim(
                    x,
                    t_curr=t_curr,
                    t_next=t_next,
                    unet=unet,
                    y=y,
                    cfg_scale=cfg_scale,
                    generator=generator,
                )
            return self._apply_readout(x, ts[-1], unet, y=y, cfg_scale=cfg_scale)

        # --- Continuous ODE/SDE samplers ---
        for i in range(self.num_steps):
            t_curr = ts[i]
            t_next = ts[i + 1]

            if self.method == "rk4_ode":
                x = self.step_rk4_ode(x, t_curr, t_next, unet, y=y, cfg_scale=cfg_scale)
            elif self.method == "euler_ode":
                x = self.step_euler_ode(x, t_curr, t_next, unet, y=y, cfg_scale=cfg_scale)
            elif self.method == "heun_ode":
                x = self.step_heun_ode(x, t_curr, t_next, unet, y=y, cfg_scale=cfg_scale)
            elif self.method == "heun_sde":
                x = self.step_heun_sde(x, t_curr, t_next, unet, y=y, cfg_scale=cfg_scale, generator=generator)
            elif self.method == "exp_euler_ode":
                x = self.step_exp_euler_ode(x, t_curr, t_next, unet, y=y, cfg_scale=cfg_scale)
            elif self.method == "exp_heun_ode":
                x = self.step_exp_heun_ode(x, t_curr, t_next, unet, y=y, cfg_scale=cfg_scale)
            else:
                raise ValueError(f"Unknown sampling method: {self.method}")

        # Apply composed-decoder readout (Tweedie projection at t_final)
        return self._apply_readout(x, ts[-1], unet, y=y, cfg_scale=cfg_scale)

# ---------------------------------------------------------------------------
# Oracle Score Model (exact CSEM identity over full training set)
# ---------------------------------------------------------------------------

class OracleScoreModel:
    """Exact CSEM oracle score computed by importance-weighted sum over all
    training-set Gaussian components.

    Interface matches ``DiTModel.__call__(z_t, t, y) -> eps_pred`` so it can
    be used as a drop-in replacement inside ``UniversalSampler``.

    Parameters
    ----------
    all_mu : Tensor [N, C, H, W]   – encoder means (CPU)
    all_logvar : Tensor [N, C, H, W] – encoder log-variances (CPU)
    all_labels : Tensor [N]         – integer class labels (CPU)
    cfg : dict                      – experiment config (needs time_schedule, t_min, t_max, …)
    device : torch.device
    ref_chunk_size : int            – how many Gaussian components to load on GPU at once
    """

    def __init__(
        self,
        all_mu: torch.Tensor,
        all_logvar: torch.Tensor,
        all_labels: torch.Tensor,
        cfg: Dict[str, Any],
        device: torch.device,
        ref_chunk_size: int = 4096,
    ):
        self.spatial_shape = tuple(all_mu.shape[1:])                              # (C, H, W)
        self.N = all_mu.shape[0]
        self.D = int(np.prod(self.spatial_shape))
        self.cfg = cfg
        self.device = device
        self.num_classes = cfg.get("num_classes", None)
        self.null_label = self.num_classes if self.num_classes is not None else None

        # Pre-load ALL reference data on GPU (≈ 2 × N × D × 4 bytes).
        # For N=60k, D=512 this is ~240 MB — comfortably fits on any modern GPU.
        self.all_mu_flat = all_mu.reshape(self.N, -1).float().to(device)          # [N, D]
        self.all_var_flat = torch.exp(
            all_logvar.reshape(self.N, -1).float()
        ).to(device)                                                               # [N, D]
        self.all_labels = all_labels.long().to(device)                             # [N]

    # ------------------------------------------------------------------

    def eval(self):
        """No-op (compatibility with model.eval())."""
        return self

    def train(self, mode=True):
        """No-op (compatibility)."""
        return self

    def parameters(self):
        """No parameters (compatibility)."""
        return iter([])

    # ------------------------------------------------------------------

    def _get_alpha_sigma(self, t_scalar: torch.Tensor):
        """Return (alpha, sigma) scalars for a single time value."""
        stype = str(self.cfg.get("time_schedule", "log_snr")).lower()
        t = t_scalar.view(1, 1)
        if stype in ("flow", "flow_matching"):
            a, s = get_flow_params(t)
        elif stype == "cosine":
            a, s, _ = get_cosine_params(t, cosine_s=float(self.cfg.get("cosine_s", 0.008)))
        else:
            a, s = get_ou_params(t)
        return a.squeeze(), s.squeeze()

    # ------------------------------------------------------------------

    def _compute_eps(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        label_filter: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Core oracle computation (all reference data GPU-resident, no chunking).

        Parameters
        ----------
        z_t : [B, C, H, W]
        t   : [B]  (assumed constant across batch)
        label_filter : [B] int labels – if given, restrict the sum per query to
                       reference points whose label matches.  ``None`` → unconditional.

        Returns
        -------
        eps_oracle : [B, C, H, W]
        """
        B = z_t.shape[0]
        device = z_t.device

        t_scalar = t[0]
        alpha, sigma = self._get_alpha_sigma(t_scalar)
        alpha = alpha.to(device)
        sigma = sigma.to(device)
        alpha_sq = alpha * alpha
        sigma_sq = sigma * sigma

        z_flat = z_t.reshape(B, self.D)                          # [B, D]

        # Diffuse reference components to time t
        mu_t = alpha * self.all_mu_flat                           # [N, D]
        var_t = alpha_sq * self.all_var_flat + sigma_sq           # [N, D]
        one_over_var = 1.0 / var_t                                # [N, D]
        mu_over_var = mu_t * one_over_var                         # [N, D]

        # --- log-weights [B, N] ---
        log_var_sum = torch.log(var_t).sum(dim=1)                 # [N]
        mu_sq_over_var_sum = (mu_t * mu_over_var).sum(dim=1)      # [N]

        # log p(z_t | x_i) ∝ -0.5 [log|Σ_t| + z^2·(1/v) - 2 z·(μ/v) + μ^2/v]
        all_log_w = -0.5 * (
            log_var_sum.unsqueeze(0)                              # [1, N]
            + (z_flat * z_flat) @ one_over_var.T                  # [B, N]
            - 2.0 * z_flat @ mu_over_var.T                        # [B, N]
            + mu_sq_over_var_sum.unsqueeze(0)                     # [1, N]
        )

        # Conditional masking
        if label_filter is not None:
            mask = (self.all_labels.unsqueeze(0) == label_filter.unsqueeze(1))  # [B, N]
            all_log_w[~mask] = float('-inf')

        # Softmax weights
        w = torch.softmax(all_log_w, dim=1)                      # [B, N]

        # Weighted score: eps = sigma * (z * (w @ 1/var) - (w @ mu/var))
        A = w @ one_over_var                                      # [B, D]
        Bv = w @ mu_over_var                                      # [B, D]
        eps_flat = sigma * (z_flat * A - Bv)

        return eps_flat.reshape(B, *self.spatial_shape)

    # ------------------------------------------------------------------

    def __call__(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward call matching DiTModel interface.

        * ``y is None``  → unconditional oracle score (sum over all components).
        * ``y`` given     → class-conditional oracle score (sum over matching labels).
        """
        return self._compute_eps(z_t, t, label_filter=y)


def evaluate_current_state(
    epoch_idx,
    prefix,
    vae,
    unet,
    loader,
    cfg,
    device,
    lpips_fn,
    fixed_noise_bank=None,
    fixed_posterior_eps_bank_A=None,
    fixed_posterior_eps_bank_B=None,
    fixed_sw2_theta=None,
    results_dir=None,
    fid_model=None,
    use_lenet_fid=False,
    frontier_tracker=None,
):
    """Evaluate unconditional generation and (optionally) class-conditional generation with CFG.

    Conditional evaluation is controlled via:
      - cfg['eval_class_labels']: list[int] (e.g. [2] to evaluate the '2' class)
      - cfg['cfg_eval_scale']: float (e.g. 3.0)
    """

    print(f"\n--- Evaluation: {prefix} @ Ep {epoch_idx} ---")
    vae.eval()
    if unet is not None:
        unet.eval()

    target_count = len(loader.dataset)
    bs = cfg["batch_size"]
    # Compute latent spatial size from img_size (default 32 for backward compat)
    img_size = cfg.get("img_size", 32)
    latent_spatial = img_size // 4
    latent_shape = (cfg["latent_channels"], latent_spatial, latent_spatial)
    sw2_nproj = int(cfg.get("sw2_n_projections", 1000))
    _dt = cfg.get("decode_time", None)
    decode_time = float(_dt) if _dt is not None else float(cfg["t_min"])

    # Validate banks
    if fixed_noise_bank is not None:
        assert fixed_noise_bank.shape[0] >= target_count
        assert tuple(fixed_noise_bank.shape[1:]) == latent_shape
    if fixed_posterior_eps_bank_A is not None:
        assert fixed_posterior_eps_bank_A.shape[0] >= target_count
        assert tuple(fixed_posterior_eps_bank_A.shape[1:]) == latent_shape
    if fixed_posterior_eps_bank_B is not None:
        assert fixed_posterior_eps_bank_B.shape[0] >= target_count
        assert tuple(fixed_posterior_eps_bank_B.shape[1:]) == latent_shape

    # -----------------------------------------------------------------------
    # Collect data: latents, images, encoder outputs, labels, dataset indices
    # -----------------------------------------------------------------------
    real_latents_A, real_latents_B, real_imgs = [], [], []
    encoder_mus, encoder_logvars = [], []
    real_labels, sample_indices = [], []
    bank_idx = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            bsz = x.shape[0]

            use_cond_enc = bool(cfg.get("use_cond_encoder", False))
            mu, logvar = vae.encode(x, y=y.to(device=device, dtype=torch.long) if use_cond_enc else None)
            std = torch.exp(0.5 * logvar)

            encoder_mus.append(mu.cpu())
            encoder_logvars.append(logvar.cpu())
            real_labels.append(y.view(-1).cpu())

            idx = torch.arange(bank_idx, bank_idx + bsz, dtype=torch.long)
            sample_indices.append(idx)

            # Posterior samples (fixed if provided)
            epsA = fixed_posterior_eps_bank_A[bank_idx:bank_idx + bsz].to(device) \
                   if fixed_posterior_eps_bank_A is not None else torch.randn_like(std)
            zA = mu + std * epsA
            real_latents_A.append(zA.cpu())

            if fixed_posterior_eps_bank_B is not None:
                epsB = fixed_posterior_eps_bank_B[bank_idx:bank_idx + bsz].to(device)
                real_latents_B.append((mu + std * epsB).cpu())

            real_imgs.append(x.cpu())
            bank_idx += bsz
            if bank_idx >= target_count:
                break

    real_latents_A = torch.cat(real_latents_A, 0)[:target_count]
    real_imgs = torch.cat(real_imgs, 0)[:target_count]
    encoder_mus = torch.cat(encoder_mus, 0)[:target_count]
    encoder_logvars = torch.cat(encoder_logvars, 0)[:target_count]
    real_labels = torch.cat(real_labels, 0)[:target_count]
    sample_indices = torch.cat(sample_indices, 0)[:target_count]

    real_flat_A = real_latents_A.view(target_count, -1).to(device)

    if fixed_posterior_eps_bank_B is not None:
        real_latents_B = torch.cat(real_latents_B, 0)[:target_count]
        real_flat_B = real_latents_B.view(target_count, -1).to(device)
    else:
        real_flat_B = None

    # -----------------------------------------------------------------------
    # Pre-compute shared quantities (FID features, unconditional LSI gap)
    # -----------------------------------------------------------------------
    print("  Extracting features for FID...")
    if use_lenet_fid:
        real_features, fid_model = extract_lenet_features(
            real_imgs, device, batch_size=cfg.get("fid_batch_size", bs), lenet_model=fid_model
        )
    else:
        real_features, fid_model = extract_inception_features(
            real_imgs, device, batch_size=cfg.get("fid_batch_size", bs), inception_model=fid_model
        )
    real_features = real_features.to(device)

    '''
    lsi_gap_unet = compute_lsi_gap(
        unet,
        encoder_mus,
        encoder_logvars,
        cfg,
        device,
        labels=None,  # unconditional branch
        num_classes=cfg.get("num_classes", None),
        num_samples=min(2500, target_count),
        num_time_points=50,
        batch_size=bs,
    )
    '''

    # -----------------------------------------------------------------------
    # Oracle score model (exact CSEM over full training set)
    # -----------------------------------------------------------------------
    print("  Building OracleScoreModel from precomputed Gaussian components...")
    oracle_model = OracleScoreModel(
        all_mu=encoder_mus,
        all_logvar=encoder_logvars,
        all_labels=real_labels,
        cfg=cfg,
        device=device,
        ref_chunk_size=4096,
    )

    '''
    lsi_gap_oracle = compute_lsi_gap(
        oracle_model,
        encoder_mus,
        encoder_logvars,
        cfg,
        device,
        labels=None,  # unconditional branch
        num_classes=cfg.get("num_classes", None),
        num_samples=min(2500, target_count),
        num_time_points=50,
        batch_size=bs,
    )
    print(f"  Oracle (unconditional) LSI gap = {lsi_gap_oracle:.6f}")

    # -----------------------------------------------------------------------
    # MSE gap: learned score net vs oracle (eps-space) with per-t breakdown
    # -----------------------------------------------------------------------
    print("  Computing MSE gap (score-space) vs oracle (with per-t breakdown)...")
    mse_gap_eps, mse_t_centres, mse_by_t_bin = compute_mse_gap_by_t(
        unet, oracle_model,
        encoder_mus, encoder_logvars,
        cfg, device,
        labels=None,
        num_classes=cfg.get("num_classes", None),
        num_samples=min(2500, target_count),
        batch_size=bs,
        space="score",
        num_t_bins=30,
    )
    print(f"  MSE gap (score-space, uncond) = {mse_gap_eps:.6f}")
    mse_gap_score = mse_gap_eps  # single pass in score space
    '''

    mse_gap_eps = mse_gap_score = lsi_gap_unet = lsi_gap_oracle = 0.0


    # -----------------------------------------------------------------------
    # Set up per-epoch eval directory
    # -----------------------------------------------------------------------
    eval_dir = None
    if results_dir is not None:
        eval_dir = os.path.join(results_dir, "evals", f"eval_{epoch_idx}")
        os.makedirs(eval_dir, exist_ok=True)


    # ----------------------------------------------------------------------
    # Viz I: Reconstruction error vs t (log-log)
    # -----------------------------------------------------------------------
    if eval_dir is not None:
        print("  [Viz I] Plotting reconstruction error vs t ...")
        plot_recon_error_vs_t(
            vae, encoder_mus, encoder_logvars, real_imgs,
            cfg, device,
            save_path=os.path.join(eval_dir, f"{prefix}_recon_error_vs_t_ep{epoch_idx}.png"),
            num_t_points=10,
            num_samples=min(256, target_count),
            batch_size=bs,
            unet=unet,
            real_labels=real_labels,
        )

    # -----------------------------------------------------------------------
    # Viz II: LPIPS decay vs t (log-log)
    # -----------------------------------------------------------------------
    lpips_decay_plot_path = None
    if eval_dir is not None and lpips_fn is not None:
        lpips_decay_plot_path = os.path.join(eval_dir, f"{prefix}_lpips_decay_vs_t_ep{epoch_idx}.png")
        print("  [Viz II] Plotting LPIPS decay vs t ...")
        plot_lpips_decay_vs_t(
            vae, encoder_mus, encoder_logvars, real_imgs,
            cfg, device,
            lpips_fn=lpips_fn,
            save_path=lpips_decay_plot_path,
            num_t_points=10,
            num_samples=min(256, target_count),
            batch_size=bs,
            unet=unet,
            real_labels=real_labels,
        )

    # -----------------------------------------------------------------------
    # Viz III: MSE gap breakdown by t (log-log)
    # -----------------------------------------------------------------------
    '''
    if eval_dir is not None and unet is not None:
        print("  [Viz III] Plotting MSE gap by t (score-space) ...")
        plot_mse_gap_by_t(
            mse_t_centres, mse_by_t_bin,
            save_path=os.path.join(eval_dir, f"{prefix}_mse_gap_by_t_ep{epoch_idx}.png"),
            cache_key=prefix,
        )
    '''

    # -----------------------------------------------------------------------
    # Viz IV: D(z_t, t) decoder output grid
    # -----------------------------------------------------------------------
    if eval_dir is not None:
        print("  [Viz IV] Plotting decoder output grid ...")
        plot_decoder_output_grid(
            vae, encoder_mus, encoder_logvars, real_imgs,
            cfg, device,
            save_path=os.path.join(eval_dir, f"{prefix}_decoder_grid_ep{epoch_idx}.png"),
            num_rows=6, num_cols=8, t_upper=2.0, t_values = [.01, .05, .15, .4, .7, 1.0, 1.5, 2.0],
            unet=unet,
            real_labels=real_labels,
        )

    # -----------------------------------------------------------------------
    # Viz V: Reverse-trajectory D(z_t, t) grid
    # -----------------------------------------------------------------------
    cfg_eval_scale = float(cfg.get("cfg_eval_scale", 3.0))
    if eval_dir is not None and unet is not None:
        print("  [Viz V] Plotting reverse trajectory grid ...")
        _traj_labels = cfg.get("eval_class_labels", None) or None  # [] → None
        plot_reverse_trajectory_grid(
            vae, unet, cfg, device,
            save_path=os.path.join(eval_dir, f"{prefix}_reverse_traj_ep{epoch_idx}.png"),
            num_rows=6, num_cols=8, t_upper=2.00, t_values = [.01, .05, .15, .4, .7, 1.0, 1.5, 2.0],
            steps_per_leg=10, cfg_scale=3.0, #cfg_eval_scale,
            class_label=_traj_labels,
            frontier_tracker=frontier_tracker,
            save_movie=True,
            plot_path_norms=True,
        )

    # -----------------------------------------------------------------------
    # Viz Vb: Reverse-trajectory D(z_t, t) grid (oracle score)
    # -----------------------------------------------------------------------
    if eval_dir is not None:
        print("  [Viz Vb] Plotting oracle reverse trajectory grid ...")
        _traj_labels = cfg.get("eval_class_labels", None) or None  # [] -> None
        plot_reverse_trajectory_grid(
            vae, oracle_model, cfg, device,
            save_path=os.path.join(eval_dir, f"{prefix}_reverse_traj_oracle_ep{epoch_idx}.png"),
            num_rows=6, num_cols=8, t_upper=2.00,
            t_values=[.01, .05, .15, .4, .7, 1.0, 1.5, 2.0],
            steps_per_leg=10, cfg_scale=3.0,
            class_label=_traj_labels,
            frontier_tracker=frontier_tracker,
            save_movie=True,
            plot_path_norms=True,
        )
    # -----------------------------------------------------------------------
    # Sampler configurations (unconditional baseline)
    # -----------------------------------------------------------------------
    configs = [
        {"method": "VAE_Rec_eps", "steps": 0, "desc": "Recon (posterior z)", "use_rand_token": False},
    ]
    if unet is not None:
         configs.extend([
            #{"method": "rk4_ode",  "steps": 25, "desc": "RandToken (RK4)", "use_rand_token": True, "cfg_level": 0},
            #{"method": "rk4_ode",  "steps": 25, "desc": "RandToken (RK4)", "use_rand_token": True, "cfg_level": 1},
            #{"method": "rk4_ode",  "steps": 25, "desc": "RandToken (RK4)", "use_rand_token": True, "cfg_level": 1.5},
            #{"method": "rk4_ode",  "steps": 25, "desc": "RandToken (RK4)", "use_rand_token": True, "cfg_level": 2.0},
            #{"method": "exp_heun_ode",  "steps": 50, "desc": "RandToken (Heun-Exp)", "use_rand_token": True, "cfg_level": 3.0},
            #{"method": "exp_euler_ode",  "steps": 100, "desc": "RandToken (Euler-Exp)", "use_rand_token": True, "cfg_level": 3.0},
            #{"method": "heun_sde",  "steps": 100, "desc": "RandToken (Heun-SDE)", "use_rand_token": True, "cfg_level": 3.0, "readout_mode": "direct"},
            #{"method": "rk4_ode",  "steps": 30, "desc": "RandToken (RK4)", "use_rand_token": True,"time_schedule": "log_t",
                 #"init_mode": "oracle", "t_max": 2.45, "t_min": 1e-3, "cfg_level": 3.0, "readout_mode": "direct"},
            #{"method": "rk4_ode",  "steps": 30, "desc": "RandToken (RK4)", "use_rand_token": True,"time_schedule": "log_t",
                 #"init_mode": "prior", "t_max": 1.98, "t_min": 1e-4, "cfg_level": 1.0, "readout_mode": "direct",},
            {"method": "rk4_ode",  "steps": 30, "desc": "RandToken (RK4)", "use_rand_token": True,"time_schedule": "log_t",
                 "init_mode": "prior", "t_max": 1.98, "t_min": 1e-4, "cfg_level": 3.0, "readout_mode": "direct","cfg_mode": "linear_ramp" },
            {"method": "rk4_ode",  "steps": 30, "desc": "RandToken (RK4)", "use_rand_token": True,"time_schedule": "log_t",
                 "init_mode": "prior", "t_max": 1.98, "t_min": 1e-4, "cfg_level": 3.0, "readout_mode": "direct"},
        ])
    # Oracle sampler configs (same steps / CFG levels as the NN)
    configs.extend([
        #{"method": "rk4_ode", "steps": 25, "desc": "Oracle (RK4)", "use_rand_token": True, "cfg_level": 1,   "use_oracle": True},
        #{"method": "rk4_ode", "steps": 25, "desc": "Oracle (RK4)", "use_rand_token": True, "cfg_level": 1.5, "use_oracle": True},
        #{"method": "heun_sde", "steps": 50, "desc": "Oracle (Heun-SDE)", "use_rand_token": True, "cfg_level": 3.0, "use_oracle": True},
        #{"method": "rk4_ode", "steps":25, "desc": "Oracle (RK4)", "use_rand_token": True, "cfg_level": 3.0, "use_oracle": True, "readout_mode": "direct"},
        {"method": "rk4_ode",  "steps": 30, "desc": "RandToken (RK4)", "use_rand_token": True,"time_schedule": "log_t",
                 "init_mode": "prior", "t_max": 1.98, "t_min": 1e-3, "cfg_level": 3.0, "readout_mode": "direct", "use_oracle": True},
        #{"method": "rk4_ode",  "steps": 30, "desc": "RandToken (RK4)", "use_rand_token": True,"time_schedule": "log_t",
                 #"init_mode": "prior", "t_max": 1.98, "t_min": 1e-3, "cfg_level": 3.0, "readout_mode": "direct", "use_oracle": True},
        #{"method": "rk4_ode",  "steps": 25, "desc": "RandToken (RK4)", "use_rand_token": True,"time_schedule": "log_t",
                 #"init_mode": "oracle", "t_max": 2.45, "t_min": 1e-3, "cfg_level": 3.0, "readout_mode": "direct", "use_oracle": True},
    ])

    def _resolve_sampler_eval_settings(scfg_local: Dict[str, Any]) -> Dict[str, Any]:
        requested_schedule_raw = scfg_local.get("time_schedule", scfg_local.get("schedule", None))
        requested_schedule = None if requested_schedule_raw is None else canonicalize_time_schedule(
            requested_schedule_raw,
            default=cfg.get("time_schedule", "log_snr"),
        )
        use_frontier_grid = requested_schedule == "frontier"
        if use_frontier_grid:
            if frontier_tracker is None or not getattr(frontier_tracker, "is_active", False):
                raise ValueError("Sampler config requested time_schedule='frontier' but no active frontier_tracker is available.")
            resolved_schedule = canonicalize_time_schedule(cfg.get("time_schedule", "log_snr"), default="log_snr")
        elif requested_schedule is None:
            resolved_schedule = canonicalize_time_schedule(cfg.get("time_schedule", "log_snr"), default="log_snr")
        else:
            resolved_schedule = requested_schedule

        default_t_min = cfg.get("cosine_t_min", 2e-4) if resolved_schedule == "cosine" else cfg["t_min"]
        default_t_max = cfg.get("cosine_t_max", 0.9999) if resolved_schedule == "cosine" else cfg["t_max"]

        t_min_val = float(scfg_local.get(
            "t_min",
            scfg_local.get("terminal_time", scfg_local.get("t_terminal", default_t_min))
        ))
        t_max_val = float(scfg_local.get(
            "t_max",
            scfg_local.get("initial_time", scfg_local.get("t_init", scfg_local.get("T_init", scfg_local.get("T_max", default_t_max))))
        ))
        if not (t_min_val < t_max_val):
            raise ValueError(f"Expected sampler t_min < t_max, got t_min={t_min_val}, t_max={t_max_val} for config {scfg_local!r}")

        init_mode = canonicalize_init_mode(
            scfg_local.get("init_mode", scfg_local.get("terminal_mode", scfg_local.get("into_mode", "prior"))),
            default="prior",
        )

        has_overrides = (
            requested_schedule is not None
            or ("t_min" in scfg_local) or ("terminal_time" in scfg_local) or ("t_terminal" in scfg_local)
            or ("t_max" in scfg_local) or ("initial_time" in scfg_local) or ("t_init" in scfg_local) or ("T_init" in scfg_local) or ("T_max" in scfg_local)
            or ("init_mode" in scfg_local) or ("terminal_mode" in scfg_local) or ("into_mode" in scfg_local)
        )

        schedule_label = requested_schedule if requested_schedule is not None else resolved_schedule
        extra_suffix = ""
        if has_overrides and scfg_local.get("method") != "VAE_Rec_eps":
            extra_suffix += f"_sched{schedule_label}"
            if abs(t_min_val - float(default_t_min)) > 1e-12 or abs(t_max_val - float(default_t_max)) > 1e-12:
                extra_suffix += f"_T{_format_sampler_float_tag(t_max_val)}_t{_format_sampler_float_tag(t_min_val)}"
            if init_mode != "prior":
                extra_suffix += f"_{init_mode}"

        return {
            "requested_schedule": requested_schedule,
            "schedule_type": resolved_schedule,
            "t_min": t_min_val,
            "t_max": t_max_val,
            "init_mode": init_mode,
            "frontier_tracker": frontier_tracker if use_frontier_grid else None,
            "extra_suffix": extra_suffix,
        }

    results = []

    # -----------------------------------------------------------------------
    # Shared banks for comparability (noise + random labels)
    # -----------------------------------------------------------------------
    # Align fixed noise bank with the realized dataset order
    noise_bank_all = fixed_noise_bank[sample_indices] if fixed_noise_bank is not None else None

    # Fixed random token bank: used when sampler config has use_rand_token=True
    rand_token_bank_all = None
    if cfg.get("num_classes", None) is not None:
        num_classes = int(cfg["num_classes"])
        # Use an independent CPU RNG so this bank is stable and does not depend on global seeding state.
        g_tok = torch.Generator(device="cpu")
        g_tok.manual_seed(int(cfg.get("seed", 0)) + 1337)
        # Sample *valid class labels* uniformly: [0, num_classes-1]. (null label is `num_classes`.)
        rand_token_bank_all = torch.randint(
            low=0, high=num_classes, size=(target_count,), generator=g_tok, dtype=torch.long
        )
        rand_token_bank_all = rand_token_bank_all[sample_indices]

    # -----------------------------------------------------------------------
    # Unconditional evaluation sweep
    # -----------------------------------------------------------------------

    # cfg_eval_scale already read above (from cfg, default 3.0)
    for scfg in configs:
        method = scfg["method"]
        steps = int(scfg.get("steps", 0))
        desc = scfg.get("desc", "")
        use_rand_token = bool(scfg.get("use_rand_token", False))
        cfg_level = scfg.get("cfg_level", None)  # NEW: extract cfg_level
        cfg_mode = canonicalize_cfg_mode(scfg.get("cfg_mode", cfg.get("cfg_mode", "constant")), default="constant")
        use_oracle = bool(scfg.get("use_oracle", False))
        readout_mode = str(scfg.get("readout_mode", "direct"))
        sampler_eval_settings = _resolve_sampler_eval_settings(scfg)

        # Choose which score model to sample with
        score_model = oracle_model if use_oracle else unet

        # Build suffix for naming - include cfg_level and readout_mode if present
        oracle_tag = "_oracle" if use_oracle else ""
        readout_tag = f"_{readout_mode}" if readout_mode != "direct" else ""
        cfg_mode_tag = f"_{cfg_mode}" if cfg_mode != "constant" else ""
        if use_rand_token:
            if cfg_level is not None:
                config_suffix = f"_randtok_cfg{cfg_level}{cfg_mode_tag}{readout_tag}{oracle_tag}"
            else:
                config_suffix = f"_randtok{cfg_mode_tag}{readout_tag}{oracle_tag}"
        else:
            config_suffix = f"{cfg_mode_tag}{readout_tag}{oracle_tag}"
        config_suffix += sampler_eval_settings["extra_suffix"]
        config_name = f"{method}@{steps}{config_suffix}"

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            if method == "VAE_Rec_eps":
                fake_imgs = torch.cat([
                    vae.decode(real_latents_A[i:i + bs].to(device),
                               t=torch.full((min(bs, len(real_latents_A)-i),), decode_time, device=device) if getattr(vae, 'time_cond_decoder', False) else None,
                               y=real_labels[i:i + min(bs, len(real_latents_A)-i)].to(device)).cpu()
                    for i in range(0, len(real_latents_A), bs)
                ], 0)

                # SW2: aggregated posterior floor
                if real_flat_B is not None:
                    w2 = compute_sw2(real_flat_A, real_flat_B, n_projections=sw2_nproj, theta=fixed_sw2_theta)
                else:
                    perm = torch.randperm(real_flat_A.size(0), device=device)
                    half = real_flat_A.size(0) // 2
                    w2 = compute_sw2(real_flat_A[perm[:half]], real_flat_A[perm[half:2*half]],
                                     n_projections=sw2_nproj, theta=fixed_sw2_theta)
                lsi_gap = 0.0

            else:
                sampler_kwargs = dict(
                    method=method,
                    num_steps=steps,
                    t_min=sampler_eval_settings["t_min"],
                    t_max=sampler_eval_settings["t_max"],
                    schedule_type=sampler_eval_settings["schedule_type"],
                    cosine_s=cfg.get("cosine_s", 0.008),
                    readout_mode=readout_mode,
                    frontier_tracker=sampler_eval_settings["frontier_tracker"],
                    init_mode=sampler_eval_settings["init_mode"],
                    cfg_mode=cfg_mode,
                )
                if method == "ddim":
                    schedule_cfg_local = dict(cfg)
                    schedule_cfg_local["time_schedule"] = sampler_eval_settings["schedule_type"]
                    if sampler_eval_settings["schedule_type"] == "cosine":
                        schedule_cfg_local["cosine_t_min"] = sampler_eval_settings["t_min"]
                        schedule_cfg_local["cosine_t_max"] = sampler_eval_settings["t_max"]
                    else:
                        schedule_cfg_local["t_min"] = sampler_eval_settings["t_min"]
                        schedule_cfg_local["t_max"] = sampler_eval_settings["t_max"]
                    sampler_kwargs.update(
                        schedule_cfg=schedule_cfg_local,
                        ddim_eta=float(cfg.get("ddim_eta", 0.0)),
                    )
                sampler = UniversalSampler(**sampler_kwargs)
                fake_latents_list, fake_imgs_list = [], []

                for i in range(0, target_count, bs):
                    batch_sz = min(bs, target_count - i)

                    y_batch = None
                    if use_rand_token and rand_token_bank_all is not None:
                        y_batch = rand_token_bank_all[i:i + batch_sz].to(device)

                    g_scale = cfg_level if use_rand_token and cfg_level is not None else None

                    if sampler_eval_settings["init_mode"] == "oracle":
                        z0_batch = real_latents_A[i:i + batch_sz].to(device)
                        noise_batch = None
                        if noise_bank_all is not None:
                            noise_batch = noise_bank_all[i:i + batch_sz].to(device)
                        x_init = sample_forward_latent_from_z0(
                            z0_batch,
                            t_value=sampler_eval_settings["t_max"],
                            schedule_type=sampler_eval_settings["schedule_type"],
                            cosine_s=cfg.get("cosine_s", 0.008),
                            noise=noise_batch,
                        )
                        z_gen, readout_t = sampler.sample(score_model, x_init=x_init, y=y_batch, cfg_scale=g_scale)
                    elif noise_bank_all is not None:
                        xT = noise_bank_all[i:i + batch_sz].to(device)
                        z_gen, readout_t = sampler.sample(score_model, x_init=xT, y=y_batch, cfg_scale=g_scale)
                    else:
                        z_gen, readout_t = sampler.sample(score_model, shape=(batch_sz, *latent_shape), device=device, y=y_batch, cfg_scale=g_scale)

                    fake_latents_list.append(z_gen.cpu())
                    # Decode time: use readout_t from composed decoder if available,
                    # otherwise fall back to the configured decode_time (≈ t_min).
                    if readout_t is not None and getattr(vae, 'time_cond_decoder', False):
                        t_dec = torch.full((z_gen.shape[0],), readout_t.item(), device=device)
                    elif getattr(vae, 'time_cond_decoder', False):
                        t_dec = torch.full((z_gen.shape[0],), decode_time, device=device)
                    else:
                        t_dec = None
                    fake_imgs_list.append(vae.decode(z_gen, t=t_dec, y=y_batch).cpu())

                fake_latents = torch.cat(fake_latents_list, 0)
                fake_imgs = torch.cat(fake_imgs_list, 0)
                # Line 2977: change .view to .reshape

                fake_flat = fake_latents.reshape(fake_latents.shape[0], -1).to(device)
                #fake_flat = fake_latents.view(fake_latents.shape[0], -1).to(device)
                w2 = compute_sw2(real_flat_A, fake_flat, n_projections=sw2_nproj, theta=fixed_sw2_theta)
                lsi_gap = lsi_gap_oracle if use_oracle else lsi_gap_unet

        # Compute image metrics (unconditional)
        if use_lenet_fid:
            fake_features, fid_model = extract_lenet_features(
                fake_imgs, device, batch_size=cfg.get("fid_batch_size", bs), lenet_model=fid_model
            )
        else:
            fake_features, fid_model = extract_inception_features(
                fake_imgs, device, batch_size=cfg.get("fid_batch_size", bs), inception_model=fid_model
            )
        fake_features = fake_features.to(device)

        fid = compute_fid_from_features(real_features, fake_features)
        kid = compute_kid(real_features, fake_features, num_subsets=100, subset_size=1000)
        div = compute_diversity(fake_imgs.to(device), lpips_fn) if LPIPS_AVAILABLE else 0.0

        results.append({
            "config": config_name,
            "desc": desc,
            "fid": fid,
            "kid": kid,
            "w2": w2,
            "div": div,
            "lsi_gap": lsi_gap,
            "mse_gap_eps": mse_gap_eps if not use_oracle else 0.0,
            "mse_gap_score": mse_gap_score if not use_oracle else 0.0,
            "time_schedule": sampler_eval_settings["schedule_type"] if method != "VAE_Rec_eps" else None,
            "t_min": sampler_eval_settings["t_min"] if method != "VAE_Rec_eps" else None,
            "t_max": sampler_eval_settings["t_max"] if method != "VAE_Rec_eps" else None,
            "init_mode": sampler_eval_settings["init_mode"] if method != "VAE_Rec_eps" else None,
            "cfg_mode": cfg_mode if method != "VAE_Rec_eps" else None,
        })

        # Save sample panels for the main sweep
        if method in ("VAE_Rec_eps",) or "rk4" in method or "heun" in method or method == "ddim":
            if eval_dir is not None:
                save_path = os.path.join(eval_dir, f"{prefix}_{method}_{steps}{config_suffix}_ep{epoch_idx}.png")
            elif results_dir is not None:
                save_path = os.path.join(results_dir, "samples", f"{prefix}_{method}_{steps}{config_suffix}_ep{epoch_idx}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            else:
                save_path = os.path.join("samples", f"{prefix}_{method}_{steps}{config_suffix}_ep{epoch_idx}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            panel = fake_imgs[:16] if fake_imgs.shape[0] >= 16 else fake_imgs
            tv_utils.save_image((panel + 1) / 2, save_path, nrow=4, padding=2)

    # Print main results
    print(f"\n  >>> Sweep Results [{prefix}] <<<")
    print(f"  {'Config':<40} | {'Desc':<20} | {'FID':<8} | {'KID':<10} | {'SW2':<10} | {'Div':<8} | {'LSI Gap':<10} | {'MSE(eps)':<10} | {'MSE(score)':<10}")
    print("  " + "-" * 150)
    for r in results:
        print(f"  {r['config']:<40} | {r['desc']:<20} | {r['fid']:<8.2f} | {r['kid']:<10.4f} | "
              f"{r['w2']:<10.6f} | {r['div']:<8.4f} | {r['lsi_gap']:<10.4f} | "
              f"{r['mse_gap_eps']:<10.4f} | {r['mse_gap_score']:<10.4f}")
    print("  " + "-" * 150 + "\n")

    # Flatten for DataFrame logging
    output_dict: Dict[str, Any] = {}
    output_dict["lpips_decay_plot_uncond"] = lpips_decay_plot_path
    output_dict["lsi_gap_oracle_uncond"] = lsi_gap_oracle
    output_dict["lsi_gap_unet_uncond"] = lsi_gap_unet
    output_dict["mse_gap_eps_uncond"] = mse_gap_eps
    output_dict["mse_gap_score_uncond"] = mse_gap_score
    for r in results:
        config = r["config"]
        if "VAE_Rec_eps" in config:
            output_dict["fid_vae_recon"] = r["fid"]
            output_dict["kid_vae_recon"] = r["kid"]
            output_dict["sw2_vae_recon"] = r["w2"]
            output_dict["div_vae_recon"] = r["div"]
        elif "rk4" in config.lower():
            # Extract everything after @ (e.g., "20_randtok_cfg1.5")
            after_at = config.split("@")[1] if "@" in config else "10"
            col_suffix = after_at.replace(".", "_")  # dots to underscores for column names
            output_dict[f"fid_rk4_{col_suffix}"] = r["fid"]
            output_dict[f"kid_rk4_{col_suffix}"] = r["kid"]
            output_dict[f"sw2_rk4_{col_suffix}"] = r["w2"]
            output_dict[f"div_rk4_{col_suffix}"] = r["div"]
            output_dict[f"lsi_gap_rk4_{col_suffix}"] = r["lsi_gap"]
            output_dict[f"mse_gap_eps_rk4_{col_suffix}"] = r["mse_gap_eps"]
            output_dict[f"mse_gap_score_rk4_{col_suffix}"] = r["mse_gap_score"]
        elif "heun" in config.lower():
            after_at = config.split("@")[1] if "@" in config else "20"
            col_suffix = after_at.replace(".", "_")
            output_dict[f"fid_heun_{col_suffix}"] = r["fid"]
            output_dict[f"kid_heun_{col_suffix}"] = r["kid"]
            output_dict[f"sw2_heun_{col_suffix}"] = r["w2"]
            output_dict[f"div_heun_{col_suffix}"] = r["div"]
            output_dict[f"lsi_gap_heun_{col_suffix}"] = r["lsi_gap"]
            output_dict[f"mse_gap_eps_heun_{col_suffix}"] = r["mse_gap_eps"]
            output_dict[f"mse_gap_score_heun_{col_suffix}"] = r["mse_gap_score"]

    # -----------------------------------------------------------------------
    # Optional: class-conditional evaluation + CFG
    # -----------------------------------------------------------------------
    eval_class_labels = cfg.get("eval_class_labels", None)
    if eval_class_labels is not None and not isinstance(eval_class_labels, (list, tuple)):
        eval_class_labels = [int(eval_class_labels)]

    cfg_eval_scale = float(cfg.get("cfg_eval_scale", 3.0))
    cond_cfg_mode = canonicalize_cfg_mode(cfg.get("cfg_mode", "constant"), default="constant")

    if unet is not None and eval_class_labels:
        print(f"  Conditional eval on labels: {list(eval_class_labels)} (CFG scale={cfg_eval_scale:g}, cfg_mode={cond_cfg_mode})")
        noise_bank_all = None
        if fixed_noise_bank is not None:
            noise_bank_all = fixed_noise_bank[sample_indices]

        cond_results_by_label: Dict[int, Any] = {}

        for y0 in eval_class_labels:
            y0 = int(y0)
            mask = (real_labels == y0)
            n_y = int(mask.sum().item())
            output_dict[f"n_real_y{y0}"] = n_y
            if n_y < 2:
                print(f"    Skipping y={y0}: only {n_y} samples")
                continue

            # Same-class reals
            real_features_y = real_features[mask]
            real_imgs_y = real_imgs[mask]  # CPU tensor, [-1,1]
            real_latents_A_y = real_latents_A[mask]
            real_flat_A_y = real_latents_A_y.view(n_y, -1).to(device)

            # For posterior-floor SW2 in class case
            if fixed_posterior_eps_bank_B is not None:
                real_latents_B_y = real_latents_B[mask]
                real_flat_B_y = real_latents_B_y.view(n_y, -1).to(device)
            else:
                real_flat_B_y = None

            # Fixed noise bank aligned to dataset order, then class-filtered
            if noise_bank_all is not None:
                noise_bank_y = noise_bank_all[mask]
            else:
                noise_bank_y = None

            rows = []

            # ---------------------------------------------------------------
            # NEW: Class-conditional VAE recon metrics (encode->reparam->decode)
            # ---------------------------------------------------------------
            with torch.no_grad():
                fake_imgs_recon_y = torch.cat([
                    vae.decode(real_latents_A_y[i:i + bs].to(device),
                               t=torch.full((min(bs, n_y-i),), decode_time, device=device) if getattr(vae, 'time_cond_decoder', False) else None,
                               y=torch.full((min(bs, n_y - i),), y0, device=device, dtype=torch.long)).cpu()
                    for i in range(0, n_y, bs)
                ], 0)

            # Feature extraction for recon
            if use_lenet_fid:
                fake_features_recon_y, fid_model = extract_lenet_features(
                    fake_imgs_recon_y, device, batch_size=cfg.get("fid_batch_size", bs), lenet_model=fid_model
                )
            else:
                fake_features_recon_y, fid_model = extract_inception_features(
                    fake_imgs_recon_y, device, batch_size=cfg.get("fid_batch_size", bs), inception_model=fid_model
                )
            fake_features_recon_y = fake_features_recon_y.to(device)

            fid_recon_y = compute_fid_from_features(real_features_y, fake_features_recon_y)

            subset_size = min(1000, real_features_y.shape[0], fake_features_recon_y.shape[0])
            if subset_size < 2:
                kid_recon_y = -1.0
            else:
                kid_recon_y = compute_kid(real_features_y, fake_features_recon_y, num_subsets=50, subset_size=subset_size)

            # SW2 posterior floor, restricted to class
            if real_flat_B_y is not None:
                w2_recon_y = compute_sw2(real_flat_A_y, real_flat_B_y, n_projections=sw2_nproj, theta=fixed_sw2_theta)
            else:
                perm = torch.randperm(real_flat_A_y.size(0), device=device)
                half = max(1, real_flat_A_y.size(0) // 2)
                w2_recon_y = compute_sw2(
                    real_flat_A_y[perm[:half]],
                    real_flat_A_y[perm[half:2 * half]],
                    n_projections=sw2_nproj,
                    theta=fixed_sw2_theta,
                )

            div_recon_y = compute_diversity(fake_imgs_recon_y.to(device), lpips_fn) if LPIPS_AVAILABLE else 0.0

            rows.append({
                "config": "VAE_Rec_eps@0",
                "mode": "recon",
                "desc": "Recon (posterior z)",
                "fid": float(fid_recon_y),
                "kid": float(kid_recon_y),
                "w2": float(w2_recon_y),
                "div": float(div_recon_y),
            })

            # Log recon metrics per class (so existing unconditional keys remain unchanged)
            output_dict[f"fid_vae_recon_y{y0}"] = fid_recon_y
            output_dict[f"kid_vae_recon_y{y0}"] = kid_recon_y
            output_dict[f"sw2_vae_recon_y{y0}"] = w2_recon_y
            output_dict[f"div_vae_recon_y{y0}"] = div_recon_y

            # Optionally save a recon panel for the class
            if eval_dir is not None:
                save_path = os.path.join(eval_dir, f"{prefix}_VAE_Rec_eps_0_y{y0}_ep{epoch_idx}.png")
                panel = fake_imgs_recon_y[:16] if fake_imgs_recon_y.shape[0] >= 16 else fake_imgs_recon_y
                tv_utils.save_image((panel + 1) / 2, save_path, nrow=4, padding=2)

            # ---------------------------------------------------------------
            # Diffusion conditional + CFG methods
            # ---------------------------------------------------------------
            for method, steps, desc in [("heun_ode", 20, "Baseline (Heun)"), ("rk4_ode", 10, "Smoothness (RK4)")]:
                for g_scale in [0.0, cfg_eval_scale]:
                    if g_scale <= 0.0:
                        tag = "cond"
                        mode = "cond"
                    else:
                        cfg_mode_suffix = f"_{cond_cfg_mode}" if cond_cfg_mode != "constant" else ""
                        tag = f"cfg{g_scale:g}{cfg_mode_suffix}"
                        mode = f"cfg{g_scale:g}{cfg_mode_suffix}"

                    sampler_kwargs_cond = dict(
                        method=method,
                        num_steps=steps,
                        t_min=cfg["t_min"],
                        t_max=cfg["t_max"],
                        schedule_type=cfg.get("time_schedule", "log_snr"),
                        cosine_s=cfg.get("cosine_s", 0.008),
                        readout_mode="direct",  # Conditional eval uses direct readout by default
                        frontier_tracker=frontier_tracker,
                        cfg_mode=cond_cfg_mode,
                    )
                    if cfg.get("time_schedule", "log_snr") == "cosine":
                        sampler_kwargs_cond["t_min"] = cfg.get("cosine_t_min", 2e-4)
                        sampler_kwargs_cond["t_max"] = cfg.get("cosine_t_max", 0.9999)
                    sampler = UniversalSampler(**sampler_kwargs_cond)
                    fake_latents_list, fake_imgs_list = [], []

                    for i in range(0, n_y, bs):
                        batch_sz = min(bs, n_y - i)
                        y_batch = torch.full((batch_sz,), y0, device=device, dtype=torch.long)

                        if noise_bank_y is not None:
                            xT = noise_bank_y[i:i + batch_sz].to(device)
                            z_gen, readout_t = sampler.sample(
                                unet,
                                x_init=xT,
                                y=y_batch,
                                cfg_scale=(None if g_scale <= 0.0 else float(g_scale)),
                            )
                        else:
                            z_gen, readout_t = sampler.sample(
                                unet,
                                shape=(batch_sz, *latent_shape),
                                device=device,
                                y=y_batch,
                                cfg_scale=(None if g_scale <= 0.0 else float(g_scale)),
                            )

                        fake_latents_list.append(z_gen.cpu())
                        if readout_t is not None and getattr(vae, 'time_cond_decoder', False):
                            t_dec = torch.full((z_gen.shape[0],), readout_t.item(), device=device)
                        elif getattr(vae, 'time_cond_decoder', False):
                            t_dec = torch.full((z_gen.shape[0],), decode_time, device=device)
                        else:
                            t_dec = None
                        fake_imgs_list.append(vae.decode(z_gen, t=t_dec, y=y_batch).cpu())

                    fake_latents = torch.cat(fake_latents_list, 0)
                    fake_imgs = torch.cat(fake_imgs_list, 0)

                    # Feature extraction
                    if use_lenet_fid:
                        fake_features_y, fid_model = extract_lenet_features(
                            fake_imgs, device, batch_size=cfg.get("fid_batch_size", bs), lenet_model=fid_model
                        )
                    else:
                        fake_features_y, fid_model = extract_inception_features(
                            fake_imgs, device, batch_size=cfg.get("fid_batch_size", bs), inception_model=fid_model
                        )
                    fake_features_y = fake_features_y.to(device)

                    fid_y = compute_fid_from_features(real_features_y, fake_features_y)

                    subset_size = min(1000, real_features_y.shape[0], fake_features_y.shape[0])
                    if subset_size < 2:
                        kid_y = -1.0
                    else:
                        kid_y = compute_kid(real_features_y, fake_features_y, num_subsets=50, subset_size=subset_size)

                    fake_flat_y = fake_latents.view(fake_latents.shape[0], -1).to(device)
                    w2_y = compute_sw2(real_flat_A_y, fake_flat_y, n_projections=sw2_nproj, theta=fixed_sw2_theta)
                    div_y = compute_diversity(fake_imgs.to(device), lpips_fn) if LPIPS_AVAILABLE else 0.0

                    rows.append({
                        "config": f"{method}@{steps}",
                        "mode": mode,
                        "desc": desc,
                        "fid": float(fid_y),
                        "kid": float(kid_y),
                        "w2": float(w2_y),
                        "div": float(div_y),
                    })

                    # Log with suffixes to avoid breaking existing plots
                    steps_str = str(steps)
                    suffix = f"_y{y0}_{tag}"
                    if method == "rk4_ode":
                        output_dict[f"fid_rk4_{steps_str}{suffix}"] = fid_y
                        output_dict[f"kid_rk4_{steps_str}{suffix}"] = kid_y
                        output_dict[f"sw2_rk4_{steps_str}{suffix}"] = w2_y
                        output_dict[f"div_rk4_{steps_str}{suffix}"] = div_y
                    else:
                        output_dict[f"fid_heun_{steps_str}{suffix}"] = fid_y
                        output_dict[f"kid_heun_{steps_str}{suffix}"] = kid_y
                        output_dict[f"sw2_heun_{steps_str}{suffix}"] = w2_y
                        output_dict[f"div_heun_{steps_str}{suffix}"] = div_y

                    # Save panels for conditional methods
                    if eval_dir is not None:
                        save_path = os.path.join(eval_dir, f"{prefix}_{method}_{steps}_y{y0}_{tag}_ep{epoch_idx}.png")
                    elif results_dir is not None:
                        save_path = os.path.join(results_dir, "samples", f"{prefix}_{method}_{steps}_y{y0}_{tag}_ep{epoch_idx}.png")
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    else:
                        save_path = os.path.join("samples", f"{prefix}_{method}_{steps}_y{y0}_{tag}_ep{epoch_idx}.png")
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    panel = fake_imgs[:16] if fake_imgs.shape[0] >= 16 else fake_imgs
                    tv_utils.save_image((panel + 1) / 2, save_path, nrow=4, padding=2)

            cond_results_by_label[y0] = rows

        # Print conditional results (mirrors the unconditional sweep printout)
        if cond_results_by_label:
            print(f"  >>> Conditional Sweep Results [{prefix}] <<<")
            for y0 in sorted(cond_results_by_label.keys()):
                rows = cond_results_by_label[y0]
                if not rows:
                    continue
                n_y = int(output_dict.get(f"n_real_y{y0}", 0))
                print(f"  [y={y0}] (n_real={n_y})")
                print(f"  {'Config':<15} | {'Mode':<8} | {'Desc':<20} | {'FID':<8} | {'KID':<10} | {'SW2':<10} | {'Div':<8}")
                print("  " + "-" * 96)
                for r in rows:
                    print(
                        f"  {r['config']:<15} | {r['mode']:<8} | {r['desc']:<20} | "
                        f"{r['fid']:<8.2f} | {r['kid']:<10.4f} | {r['w2']:<10.6f} | {r['div']:<8.4f}"
                    )
                print("  " + "-" * 96)
            print("")

        print("  Conditional eval complete. (Metrics stored with suffix: _yK_cond / _yK_cfgS; "
              "and recon metrics as fid_vae_recon_yK, etc.)\n")

    return output_dict



def setup_run_results_dir(base_dir="run_results", wipe=True, preserve_checkpoints=True):
    if os.path.exists(base_dir) and wipe:
        if preserve_checkpoints:
            # delete everything except checkpoints
            for sub in ["plots", "samples", "dataframes"]:
                shutil.rmtree(os.path.join(base_dir, sub), ignore_errors=True)
        else:
            shutil.rmtree(base_dir)

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "dataframes"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)

    return base_dir



# ===========================================================================
# REFACTORED PLOTTING FUNCTIONS
# ===========================================================================
# These functions dynamically discover available metrics from the DataFrame
# instead of hardcoding column names like "fid_rk4_10" or "fid_heun_20".
#
# Replace your existing plotting functions with these.
# ===========================================================================

import os
import re
import pandas as pd
import matplotlib.pyplot as plt


def discover_metric_columns(eval_df, metric_prefix):
    """
    Discover all columns matching a metric prefix pattern.

    Args:
        eval_df: DataFrame with evaluation metrics
        metric_prefix: One of 'fid', 'kid', 'sw2', 'div', 'lsi_gap', 'mse_gap_eps', 'mse_gap_score'

    Returns:
        List of column names matching the pattern, excluding vae_recon variants
    """
    if metric_prefix == 'lsi_gap':
        # lsi_gap columns look like: lsi_gap_rk4_20_randtok
        pattern = re.compile(r'^lsi_gap_')
    elif metric_prefix == 'mse_gap_eps':
        # mse_gap_eps columns look like: mse_gap_eps_rk4_20_randtok
        pattern = re.compile(r'^mse_gap_eps_')
    elif metric_prefix == 'mse_gap_score':
        # mse_gap_score columns look like: mse_gap_score_rk4_20_randtok
        pattern = re.compile(r'^mse_gap_score_')
    else:
        # Other columns look like: fid_rk4_20_randtok (exclude fid_vae_recon)
        pattern = re.compile(rf'^{metric_prefix}_(?!vae_recon)')

    return [col for col in eval_df.columns if pattern.match(col)]

def parse_metric_column(col_name):
    """
    Parse a metric column name into its components.

    Examples:
        'fid_rk4_10' -> {'metric': 'fid', 'method': 'rk4', 'steps': '10', 'suffix': ''}
        'fid_rk4_20_randtok' -> {'metric': 'fid', 'method': 'rk4', 'steps': '20', 'suffix': '_randtok'}
        'lsi_gap_rk4_20_randtok' -> {'metric': 'lsi_gap', 'method': 'rk4', 'steps': '20', 'suffix': '_randtok'}
        'mse_gap_eps_rk4_25_randtok' -> {'metric': 'mse_gap_eps', 'method': 'rk4', 'steps': '25', 'suffix': '_randtok'}
        'mse_gap_score_rk4_25_randtok' -> {'metric': 'mse_gap_score', 'method': 'rk4', 'steps': '25', 'suffix': '_randtok'}
    """
    parts = col_name.split('_')

    # Handle mse_gap_eps / mse_gap_score specially (3-token metric name)
    if parts[0] == 'mse' and len(parts) > 2 and parts[1] == 'gap' and parts[2] in ('eps', 'score'):
        metric = f'mse_gap_{parts[2]}'
        remaining = parts[3:]  # Skip 'mse', 'gap', 'eps'/'score'
    # Handle lsi_gap specially (has underscore in metric name)
    elif parts[0] == 'lsi' and len(parts) > 1 and parts[1] == 'gap':
        metric = 'lsi_gap'
        remaining = parts[2:]  # Skip 'lsi' and 'gap'
    else:
        metric = parts[0]
        remaining = parts[1:]

    if len(remaining) < 2:
        return None  # Not a valid metric column (e.g., fid_vae_recon)

    method = remaining[0]  # rk4, heun, ddim, euler, etc.
    steps = remaining[1]   # 10, 20, 50, etc.

    # Everything after method_steps is the suffix
    suffix = '_'.join(remaining[2:]) if len(remaining) > 2 else ''
    if suffix:
        suffix = '_' + suffix

    return {
        'metric': metric,
        'method': method,
        'steps': steps,
        'suffix': suffix,
        'full_name': col_name,
    }


def get_metric_groups(eval_df):
    """
    Group all metric columns by (method, steps, suffix) for systematic plotting.

    Returns:
        Dict mapping (method, steps, suffix) -> dict of metric columns
        e.g., {('rk4', '20', '_randtok'): {'fid': 'fid_rk4_20_randtok', 'kid': 'kid_rk4_20_randtok', ...}}
    """
    groups = {}

    for metric_type in ['fid', 'kid', 'sw2', 'div', 'lsi_gap', 'mse_gap_eps', 'mse_gap_score']:
        cols = discover_metric_columns(eval_df, metric_type)
        for col in cols:
            parsed = parse_metric_column(col)
            if parsed is None:
                continue

            key = (parsed['method'], parsed['steps'], parsed['suffix'])
            if key not in groups:
                groups[key] = {}
            groups[key][metric_type] = col

    return groups


def plot_generic_metric(eval_df, metric_col, metric_name, ylabel, title, save_path,
                        use_log=False, include_vae_recon=True):
    """
    Generic plotting function for any metric column.
    Plots LSI vs Tweedie (blue vs red) with optional VAE recon baseline.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    lsi_df = eval_df[eval_df["tag"].str.contains("LSI", case=False)]
    ctrl_df = eval_df[eval_df["tag"].str.contains("Ctrl", case=False)]

    # VAE recon baseline (if available and requested)
    vae_recon_col = f"{metric_name}_vae_recon"
    if include_vae_recon and vae_recon_col in lsi_df.columns and not lsi_df[vae_recon_col].isna().all():
        ax.plot(lsi_df["epoch"], lsi_df[vae_recon_col],
                color="black", linestyle="--", linewidth=2, marker='o', markersize=4,
                label=f"VAE Recon {metric_name.upper()}")

    # LSI metric
    if metric_col in lsi_df.columns and not lsi_df[metric_col].isna().all():
        ax.plot(lsi_df["epoch"], lsi_df[metric_col],
                color="blue", linewidth=2, marker='o', markersize=4,
                label=f"LSI {metric_name.upper()}")

    # Tweedie metric
    if metric_col in ctrl_df.columns and not ctrl_df[metric_col].isna().all():
        ax.plot(ctrl_df["epoch"], ctrl_df[metric_col],
                color="red", linewidth=2, marker='s', markersize=4,
                label=f"Tweedie {metric_name.upper()}")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    if use_log:
        ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both' if use_log else 'major')
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def plot_gap_metric(eval_df, metric_col, metric_name, title, save_path):
    """
    Plot the gap between Tweedie and LSI for a given metric.
    Positive gap = LSI is better.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    lsi_df = eval_df[eval_df["tag"].str.contains("LSI", case=False)].copy()
    ctrl_df = eval_df[eval_df["tag"].str.contains("Ctrl", case=False)].copy()

    if len(lsi_df) > 0 and len(ctrl_df) > 0 and metric_col in lsi_df.columns and metric_col in ctrl_df.columns:
        # Merge on epoch
        merged = pd.merge(
            lsi_df[["epoch", metric_col]],
            ctrl_df[["epoch", metric_col]],
            on="epoch",
            suffixes=("_lsi", "_ctrl")
        )

        lsi_col = f"{metric_col}_lsi"
        ctrl_col = f"{metric_col}_ctrl"

        if lsi_col in merged.columns and ctrl_col in merged.columns:
            merged["gap"] = merged[ctrl_col] - merged[lsi_col]
            ax.plot(merged["epoch"], merged["gap"],
                    color="purple", linewidth=2, marker='o', markersize=4,
                    label=f"{metric_name.upper()} Gap (Tweedie - LSI)")

            ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
            if len(merged) > 0:
                ax.fill_between(merged["epoch"], 0, merged["gap"],
                                where=merged["gap"] > 0, alpha=0.3, color="green",
                                label="LSI Better")
                ax.fill_between(merged["epoch"], 0, merged["gap"],
                                where=merged["gap"] < 0, alpha=0.3, color="red",
                                label="Tweedie Better")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(f"{metric_name.upper()} Gap (Tweedie - LSI)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def format_config_label(method, steps, suffix):
    """Create a human-readable label for a sampler configuration."""
    method_names = {
        'rk4': 'RK4',
        'heun': 'Heun',
        'euler': 'Euler',
        'ddim': 'DDIM',
    }
    base = f"{method_names.get(method, method.upper())} {steps} Steps"

    if suffix:
        # Parse suffix for human readability
        suffix_clean = suffix.lstrip('_')
        if 'randtok' in suffix_clean:
            parts = []
            if 'cfg' in suffix_clean:
                cfg_match = re.search(r'cfg([\d.]+)', suffix_clean)
                if cfg_match:
                    parts.append(f"CFG={cfg_match.group(1)}")
            parts.insert(0, "RandTok")
            base += f" ({', '.join(parts)})"
        elif 'cfg' in suffix_clean:
            cfg_match = re.search(r'cfg([\d.]+)', suffix_clean)
            if cfg_match:
                base += f" (CFG={cfg_match.group(1)})"

    return base


def generate_all_visualizations(loss_df, eval_df, results_dir):
    """
    Generate visualization plots dynamically based on available metrics.
    """
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("\n--> Generating visualization suite...")

    # Discover all available metric columns
    metric_groups = {}
    for metric_type in ['fid', 'kid', 'sw2', 'lsi_gap', 'mse_gap_eps', 'mse_gap_score']:
        cols = discover_metric_columns(eval_df, metric_type)
        for col in cols:
            parsed = parse_metric_column(col)
            if parsed is None:
                continue

            key = (parsed['method'], parsed['steps'], parsed['suffix'])
            if key not in metric_groups:
                metric_groups[key] = {}
            metric_groups[key][metric_type] = col

    if metric_groups:
        print(f"    Found {len(metric_groups)} sampler configuration(s):")
        for (method, steps, suffix) in sorted(metric_groups.keys()):
            label = format_config_label(method, steps, suffix)
            available = list(metric_groups[(method, steps, suffix)].keys())
            print(f"      - {label}: {available}")

    plot_idx = 1

    # --- Loss plots (always generate these) ---
    plot_vae_recon_loss(loss_df, os.path.join(plots_dir, f"{plot_idx:02d}_vae_recon_loss.png"))
    plot_idx += 1
    plot_score_losses(loss_df, os.path.join(plots_dir, f"{plot_idx:02d}_score_losses.png"))
    plot_idx += 1

    if not metric_groups:
        print("--> Warning: No metric columns found. Skipping eval metric plots.")
        print(f"--> Visualization suite complete ({plot_idx - 1} plots generated)!")
        return

    # --- Metric plots for each sampler configuration ---
    for (method, steps, suffix) in sorted(metric_groups.keys()):
        group = metric_groups[(method, steps, suffix)]
        config_label = format_config_label(method, steps, suffix)
        config_tag = f"{method}_{steps}{suffix}".replace('.', '_')

        # FID plot
        if 'fid' in group:
            plot_generic_metric(
                eval_df, group['fid'], 'fid', 'FID',
                f"FID Comparison ({config_label})",
                os.path.join(plots_dir, f"{plot_idx:02d}_fid_{config_tag}.png"),
                use_log=False, include_vae_recon=True
            )
            plot_idx += 1

        # KID plot
        if 'kid' in group:
            plot_generic_metric(
                eval_df, group['kid'], 'kid', 'KID',
                f"KID Comparison ({config_label})",
                os.path.join(plots_dir, f"{plot_idx:02d}_kid_{config_tag}.png"),
                use_log=False, include_vae_recon=True
            )
            plot_idx += 1

        # SW2 plot (log scale)
        if 'sw2' in group:
            plot_generic_metric(
                eval_df, group['sw2'], 'sw2', 'SW2 (log scale)',
                f"Sliced-Wasserstein-2 ({config_label})",
                os.path.join(plots_dir, f"{plot_idx:02d}_sw2_{config_tag}.png"),
                use_log=True, include_vae_recon=True
            )
            plot_idx += 1

        # FID Gap plot
        if 'fid' in group:
            plot_gap_metric(
                eval_df, group['fid'], 'fid',
                f"LSI Advantage: FID Gap ({config_label})",
                os.path.join(plots_dir, f"{plot_idx:02d}_fid_gap_{config_tag}.png")
            )
            plot_idx += 1

        # LSI Gap Metric plot
        if 'lsi_gap' in group:
            plot_generic_metric(
                eval_df, group['lsi_gap'], 'lsi_gap', 'LSI Gap (lower = better)',
                f"LSI Gap Metric ({config_label})",
                os.path.join(plots_dir, f"{plot_idx:02d}_lsi_gap_{config_tag}.png"),
                use_log=False, include_vae_recon=False
            )
            plot_idx += 1

        # MSE Gap (eps-space) plot
        if 'mse_gap_eps' in group:
            plot_generic_metric(
                eval_df, group['mse_gap_eps'], 'mse_gap_eps', 'MSE Gap eps (lower = better)',
                f"MSE Gap vs Oracle, eps-space ({config_label})",
                os.path.join(plots_dir, f"{plot_idx:02d}_mse_gap_eps_{config_tag}.png"),
                use_log=False, include_vae_recon=False
            )
            plot_idx += 1

        # MSE Gap (score-space) plot
        if 'mse_gap_score' in group:
            plot_generic_metric(
                eval_df, group['mse_gap_score'], 'mse_gap_score', 'MSE Gap score (lower = better)',
                f"MSE Gap vs Oracle, score-space ({config_label})",
                os.path.join(plots_dir, f"{plot_idx:02d}_mse_gap_score_{config_tag}.png"),
                use_log=False, include_vae_recon=False
            )
            plot_idx += 1

    print(f"--> Visualization suite complete ({plot_idx - 1} plots generated)!")

def plot_vae_recon_loss(loss_df, save_path):
    """Plot VAE reconstruction loss (cotrain epochs only)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    cotrain_df = loss_df[loss_df["stage"] == "cotrain"]

    if len(cotrain_df) > 0:
        ax.plot(cotrain_df["epoch"], cotrain_df["recon"],
                color="black", linewidth=2, marker='o', markersize=4)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Reconstruction Loss (MSE)", fontsize=12)
    ax.set_title("VAE Reconstruction Loss (Co-training Phase)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def plot_score_losses(loss_df, save_path):
    """Plot LSI vs Tweedie score losses (log scale)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(loss_df["epoch"], loss_df["score_lsi"],
            color="blue", linewidth=2, marker='o', markersize=4, label="LSI Score Loss")
    ax.plot(loss_df["epoch"], loss_df["score_control"],
            color="red", linewidth=2, marker='s', markersize=4, label="Tweedie Score Loss")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Score Loss (MSE, log scale)", fontsize=12)
    ax.set_title("Score Network Losses: LSI vs Tweedie", fontsize=14)
    ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def generate_comparison_visualizations(eval_df_cotrain, eval_df_indep, results_dir):
    """
    Generate all comparison plots (4-way: cotrain vs indep, LSI vs Tweedie)
    dynamically based on available metric columns.
    """
    plots_dir = os.path.join(results_dir, "comparison_plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("\n--> Generating comparison visualization suite...")

    # Discover all available metric columns from both dataframes
    all_columns = set(eval_df_cotrain.columns) | set(eval_df_indep.columns)

    # Group columns by (method, steps, suffix)
    metric_groups = {}
    for metric_type in ['fid', 'kid', 'sw2', 'lsi_gap', 'mse_gap_eps', 'mse_gap_score']:
        cols = discover_metric_columns(eval_df_cotrain, metric_type)
        cols.extend(discover_metric_columns(eval_df_indep, metric_type))
        cols = list(set(cols))  # Remove duplicates

        for col in cols:
            parsed = parse_metric_column(col)
            if parsed is None:
                continue

            key = (parsed['method'], parsed['steps'], parsed['suffix'])
            if key not in metric_groups:
                metric_groups[key] = {}
            metric_groups[key][metric_type] = col

    if not metric_groups:
        print("--> Warning: No metric columns found. Skipping comparison plots.")
        return

    print(f"    Found {len(metric_groups)} sampler configuration(s) to compare:")
    for (method, steps, suffix) in sorted(metric_groups.keys()):
        label = format_config_label(method, steps, suffix)
        available = list(metric_groups[(method, steps, suffix)].keys())
        print(f"      - {label}: {available}")

    # Generate plots
    plot_idx = 1
    ylabel_map = {
        'fid': 'FID',
        'kid': 'KID',
        'sw2': 'SW2 (log scale)',
        'lsi_gap': 'LSI Gap (lower=better)',
        'mse_gap_eps': 'MSE Gap eps (lower=better)',
        'mse_gap_score': 'MSE Gap score (lower=better)',
    }

    for (method, steps, suffix) in sorted(metric_groups.keys()):
        group = metric_groups[(method, steps, suffix)]
        config_label = format_config_label(method, steps, suffix)
        config_tag = f"{method}_{steps}{suffix}".replace('.', '_')  # Safe filename

        for metric_type in ['fid', 'kid', 'sw2', 'lsi_gap', 'mse_gap_eps', 'mse_gap_score']:
            if metric_type not in group:
                continue

            metric_col = group[metric_type]
            use_log = (metric_type == 'sw2')
            ylabel = ylabel_map[metric_type]
            title = f"Co-trained vs Independent: {metric_type.upper()} ({config_label})"
            save_path = os.path.join(plots_dir, f"{plot_idx:02d}_comparison_{metric_type}_{config_tag}.png")

            plot_comparison_metric(eval_df_cotrain, eval_df_indep, metric_col, ylabel, title, save_path, use_log)
            plot_idx += 1

    print(f"--> Comparison visualization suite complete ({plot_idx - 1} plots generated)!")


import matplotlib.pyplot as plt
import numpy as np

def plot_comparison_metric(eval_df_cotrain, eval_df_indep, metric_col, ylabel, title, save_path, use_log=False):
    """
    4-way comparison plot for any metric.
    - Solid blue: Co-trained LSI
    - Solid red: Co-trained Tweedie
    - Dashed blue: Independent LSI
    - Dashed red: Independent Tweedie
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Co-trained data
    lsi_cotrain = eval_df_cotrain[eval_df_cotrain["tag"].str.contains("LSI", case=False)]
    ctrl_cotrain = eval_df_cotrain[eval_df_cotrain["tag"].str.contains("Ctrl", case=False)]

    # Independent data
    lsi_indep = eval_df_indep[eval_df_indep["tag"].str.contains("LSI", case=False)]
    ctrl_indep = eval_df_indep[eval_df_indep["tag"].str.contains("Ctrl", case=False)]

    # Plot Co-trained (solid lines)
    if metric_col in lsi_cotrain.columns and not lsi_cotrain[metric_col].isna().all():
        ax.plot(lsi_cotrain["epoch"], lsi_cotrain[metric_col],
                color="blue", linewidth=2, marker='o', markersize=4,
                linestyle="-", label="Co-trained LSI")

    if metric_col in ctrl_cotrain.columns and not ctrl_cotrain[metric_col].isna().all():
        ax.plot(ctrl_cotrain["epoch"], ctrl_cotrain[metric_col],
                color="red", linewidth=2, marker='s', markersize=4,
                linestyle="-", label="Co-trained Tweedie")

    # Plot Independent (dashed lines)
    if metric_col in lsi_indep.columns and not lsi_indep[metric_col].isna().all():
        ax.plot(lsi_indep["epoch"], lsi_indep[metric_col],
                color="blue", linewidth=2, marker='o', markersize=4,
                linestyle="--", label="Independent LSI")

    if metric_col in ctrl_indep.columns and not ctrl_indep[metric_col].isna().all():
        ax.plot(ctrl_indep["epoch"], ctrl_indep[metric_col],
                color="red", linewidth=2, marker='s', markersize=4,
                linestyle="--", label="Independent Tweedie")

    ax.set_xlabel("LDM Training Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    if use_log:
        ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both' if use_log else 'major')
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def save_dataframes(loss_df, eval_df, results_dir):
    """
    Save the dataframes to CSV files.
    """
    df_dir = os.path.join(results_dir, "dataframes")

    loss_path = os.path.join(df_dir, "loss_history.csv")
    eval_path = os.path.join(df_dir, "eval_metrics.csv")

    loss_df.to_csv(loss_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    print(f"--> Saved loss history to {loss_path}")
    print(f"--> Saved eval metrics to {eval_path}")

# ---------------------------------------------------------------------------
# Data & Training
# ---------------------------------------------------------------------------

def make_dataloaders(batch_size, num_workers, dataset_key="FMNIST"):
    """Create dataloaders for specified dataset."""
    if dataset_key not in DATASET_INFO:
        raise ValueError(f"Unknown dataset: {dataset_key}. Choose from {list(DATASET_INFO.keys())}")

    info = DATASET_INFO[dataset_key]
    dataset_cls = info["class"]
    num_classes = info["num_classes"]
    img_size = info.get("img_size", 28)
    img_channels = info.get("img_channels", 1)
    is_grayscale_cifar = info.get("grayscale", False)

    # Build transforms based on dataset
    if dataset_key == "CIFAR":
        # CIFAR-10 RGB: 32x32, 3 channels, no padding needed
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif dataset_key == "GCIFAR":
        # CIFAR-10 grayscale: 32x32, no padding needed
        tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        # MNIST-family datasets: 28x28, pad to 32x32
        tf = transforms.Compose([
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    if dataset_key == "EMNIST":
        train = dataset_cls("./data", split=info["split"], train=True, download=True, transform=tf)
        test = dataset_cls("./data", split=info["split"], train=False, download=True, transform=tf)
    else:
        train = dataset_cls("./data", train=True, download=True, transform=tf)
        test = dataset_cls("./data", train=False, download=True, transform=tf)

    tl = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    vl = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"--> Loaded {dataset_key}: {len(train)} train, {len(test)} test, {num_classes} classes, img_size={img_size}, img_channels={img_channels}")
    return tl, vl, num_classes


def train_vae_cotrained_cond(cfg):
    """
    Modified co-training function with DataFrame logging and visualization.

    Changes from original:
    - Wipes and recreates run_results directory at start
    - Logs per-epoch losses to a DataFrame
    - Logs evaluate_current_state results to a DataFrame
    - Generates visualization suite after training completes
    - Saves all results to run_results directory
    """
    # --- Setup Results Directory ---
    results_dir = setup_run_results_dir(cfg.get("results_dir", "run_results"),
                                   wipe=True, preserve_checkpoints=True)

    # Keep the original checkpoint directory name distinct from the run-specific save path.
    # Example: if cfg["ckpt_dir"] == "checkpoints", then by default we first try loading
    # from the literal top-level ./checkpoints directory, while still saving into
    # <results_dir>/checkpoints for this run.
    ckpt_dir_name = cfg.get("ckpt_dir", "checkpoints")
    cfg["ckpt_dir_name"] = ckpt_dir_name
    cfg["ckpt_dir"] = os.path.join(results_dir, ckpt_dir_name)
    eval_freq = cfg.get("eval_freq", 10)
    eval_freq_cotrain = cfg.get("eval_freq_cotrain", eval_freq)  # ADD THIS
    eval_freq_refine = cfg.get("eval_freq_refine", eval_freq)    # ADD THIS

    device = default_device()
    # --- Discrete diffusion schedule (train + sampling) ---
    # This switches training from continuous log-uniform OU times to discrete VP/DDPM timesteps.


    # --- Unified discrete schedule (log_t / log_snr / cosine depending on cfg) ---
    ou_sched = make_schedule(cfg, device)
    T = int(ou_sched["T"].item())
    noise_sched = ou_sched
    print(f"--> Time schedule: {ou_sched['schedule_type']} ({T} steps)")

    # --- Adaptive Frontier Time Sampler ---
    use_adaptive_time = bool(cfg.get("adaptive_time", False))
    frontier_tracker = None
    if use_adaptive_time:
        frontier_tracker = ReconFrontierTracker(
            t_min=float(cfg["t_min"]),
            t_max=float(cfg["t_max"]),
            n_bins=int(cfg.get("adaptive_time_nbins", 100)),
            ema_decay=float(cfg.get("adaptive_time_ema", 0.99)),
            floor_weight=float(cfg.get("adaptive_time_floor", 0.02)),
            warmup_steps=int(cfg.get("adaptive_time_warmup", 500)),
            min_counts_per_bin=int(cfg.get("adaptive_time_min_counts", 5)),
            smooth_sigma=float(cfg.get("adaptive_time_smooth_sigma", 3.0)),
            device=device,
        )
        print(f"--> Adaptive frontier time sampling: ON "
              f"({frontier_tracker.n_bins} bins, warmup={frontier_tracker.warmup_steps}, "
              f"smooth_σ={frontier_tracker.smooth_sigma})")
    else:
        print("--> Adaptive frontier time sampling: OFF")

    dataset_key = cfg.get("dataset", "FMNIST")
    train_l, test_l, num_classes = make_dataloaders(cfg["batch_size"], cfg["num_workers"], dataset_key)
    cfg["num_classes"] = num_classes  # for CFG label embedding / eval

    # Get image size from dataset info (for model initialization)
    img_size = DATASET_INFO.get(dataset_key, {}).get("img_size", 28)
    img_channels = DATASET_INFO.get(dataset_key, {}).get("img_channels", 1)
    # Effective size after padding (28->32 for MNIST-family, 32 for CIFAR variants)
    effective_img_size = 32 if img_size == 28 else img_size
    cfg["img_size"] = effective_img_size  # store for later use
    cfg["img_channels"] = img_channels    # store for later use
    latent_spatial = effective_img_size // 4  # latent H=W after 2 stride-2 downsamples

    # Get FID model (Inception for FMNIST/GCIFAR, LeNet for others)
    fid_model, use_lenet_fid = get_fid_model(dataset_key, train_l, num_classes, device, cfg["ckpt_dir"])

    use_ddim_times = cfg.get("use_ddim_times", False)
    if str(cfg.get("time_schedule", "")).lower() in ("flow", "flow_matching"):
        use_ddim_times = False

    vae = VAE(
                  latent_channels=cfg["latent_channels"],
                  base_ch=int(cfg.get("base_ch", 64)),
                  use_norm=cfg.get("use_latent_norm", False),
                  img_size=effective_img_size,
                  img_channels=img_channels,
                  num_classes=(num_classes if cfg.get("use_cond_encoder", False) else None),
                  null_label=int(num_classes),
                  cond_emb_dim=int(cfg.get("cond_emb_dim", 64)),
                  aux_d=int(cfg.get("aux_d", 0)),
                  # v1 knobs
                  num_res_blocks=int(cfg.get("num_res_blocks", 2)),
                  decoder_attn_half=cfg.get("decoder_attn_half", True),
                  latent_proj_depth=int(cfg.get("latent_proj_depth", 2)),
                  # v2 knobs
                  encoder_attn_half=cfg.get("encoder_attn_half", False),
                  decoder_extra_block=cfg.get("decoder_extra_block", False),
                  conv3x3_proj=cfg.get("conv3x3_proj", False),
                  use_tanh_out=cfg.get("use_tanh_out", True),
                  clamp_logvar=cfg.get("clamp_logvar", False),
                  attn_zero_init=cfg.get("attn_zero_init", True),
                  # TDD knobs
                  time_cond_decoder=cfg.get("time_cond_decoder", False),
                  dec_time_emb_dim=int(cfg.get("dec_time_emb_dim", 128)),
                  # Decoder class conditioning
                  class_decoder=cfg.get("class_decoder", True),
                  decoder_num_classes=(num_classes if cfg.get("class_decoder", True) else None),
                  decoder_null_label=int(num_classes),
              ).to(device)

    # Tell the TDD decoder which schedule type to use for log-SNR embedding
    if cfg.get("time_cond_decoder", False):
        vae.dec_schedule_type = str(cfg.get("time_schedule", "log_t")).lower()

    # Set the fallback decode time so that decode(z) without an explicit t
    # uses t_min (the most in-distribution default for generation).
    vae._decode_t_default = float(cfg.get("t_min", 1e-4))

    # --- Online Models ---
    dit_kwargs = dict(
        in_channels=int(cfg["latent_channels"]),
        patch_size=int(cfg.get("dit_patch_size", 1)),
        hidden_dim=int(cfg.get("dit_hidden_dim", 384)),
        depth=int(cfg.get("dit_depth", 12)),
        num_heads=int(cfg.get("dit_num_heads", 6)),
        mlp_ratio=float(cfg.get("dit_mlp_ratio", 4.0)),
        dropout=float(cfg.get("dit_dropout", 0.0)),
        num_classes=num_classes,
        latent_size=int(latent_spatial),
        factored_head=bool(cfg.get("factored_head", False)),
    )
    unet_lsi = UNetModel(**dit_kwargs).to(device)
    unet_control = UNetModel(**dit_kwargs).to(device)

    if cfg.get("load_from_checkpoint", False):
        explicit_ckpt_load_dir = cfg.get("ckpt_load_dir", None)
        run_ckpt_dir = cfg["ckpt_dir"]

        if explicit_ckpt_load_dir is not None:
            ckpt_search_dirs = [explicit_ckpt_load_dir]
        else:
            top_level_ckpt_dir = cfg.get("ckpt_dir_name", "checkpoints")
            ckpt_search_dirs = [top_level_ckpt_dir, run_ckpt_dir]

        # Preserve order while removing duplicates.
        ckpt_search_dirs = list(dict.fromkeys(ckpt_search_dirs))
        print(f"--> Loading checkpoints; search order: {ckpt_search_dirs}")

        def _resolve_ckpt_path(filename: str) -> str:
            for ckpt_dir_candidate in ckpt_search_dirs:
                ckpt_path_candidate = os.path.join(ckpt_dir_candidate, filename)
                if os.path.exists(ckpt_path_candidate):
                    return ckpt_path_candidate
            return os.path.join(ckpt_search_dirs[0], filename)

        try:
            vae.load_state_dict(torch.load(_resolve_ckpt_path("vae_cotrained.pt"), map_location=device), strict=False)
            print("    Loaded VAE.")
        except Exception as e:
            print(f"    Warning: Could not load VAE ({e})")

        try:
            unet_lsi.load_state_dict(torch.load(_resolve_ckpt_path("unet_lsi.pt"), map_location=device), strict=False)
            print("    Loaded UNet LSI.")
        except Exception as e:
            print(f"    Warning: Could not load UNet LSI ({e})")

        try:
            unet_control.load_state_dict(torch.load(_resolve_ckpt_path("unet_control.pt"), map_location=device), strict=False)
            print("    Loaded UNet Control.")
        except Exception as e:
            print(f"    Warning: Could not load UNet Control ({e})")


    # --- EMA Models (Score Heads Only) ---
    unet_lsi_ema = UNetModel(**dit_kwargs).to(device)
    unet_lsi_ema.load_state_dict(unet_lsi.state_dict())
    unet_lsi_ema.eval()
    for p in unet_lsi_ema.parameters(): p.requires_grad = False

    unet_control_ema = UNetModel(**dit_kwargs).to(device)
    unet_control_ema.load_state_dict(unet_control.state_dict())
    unet_control_ema.eval()
    for p in unet_control_ema.parameters(): p.requires_grad = False

    ema_decay = cfg.get("ema_decay", .999)

    # --- Asymmetric LR Settings ---
    score_w_vae = cfg.get("score_w_vae", cfg["score_w"])
    cotrain_head = cfg.get("cotrain_head", "lsi")
    aux_head_w = float(cfg.get("aux_head_w", 0.05))
    div_w = float(cfg.get("div_w", -0.001))
    freeze_score_in_cotrain = cfg.get("freeze_score_in_cotrain", False)
    if div_w != 0.0 and not bool(getattr(unet_lsi, "factored_head", False)):
        raise ValueError(
            "div_w != 0 requires factored_head=True because the raw diversity loss "
            "uses the factorized natural-parameter heads (nu / lambda)."
        )

    # --- Asymmetric decode MSE weights ---
    score_w_decode = float(cfg.get("score_w_decode", 0.5))
    decode_w = float(cfg.get("decode_w", 1.0))

    # --- MSE mode: 'raw', 'score', or 'score_detached' ---
    mse_mode = str(cfg.get("mse_mode", "raw")).lower()
    assert mse_mode in ("raw", "score", "score_detached"), \
        f"Unknown mse_mode={mse_mode!r}. Expected 'raw', 'score', or 'score_detached'."
    if mse_mode in ("score", "score_detached"):
        if cotrain_head != "lsi":
            raise ValueError(
                f"mse_mode={mse_mode!r} requires cotrain_head='lsi' (CSEM score head). "
                f"Got cotrain_head={cotrain_head!r}. Use mse_mode='raw' with tweedie/control heads."
            )
        if freeze_score_in_cotrain:
            raise ValueError(
                f"mse_mode={mse_mode!r} requires an active score head during co-training "
                f"(freeze_score_in_cotrain must be False)."
            )
    temporal_variance_scale = float(cfg.get("temporal_variance_scale", 0.0))
    temporal_perturb_type = str(cfg.get("temporal_perturb_type", "base")).strip().lower()
    if temporal_variance_scale < 0.0:
        raise ValueError("temporal_variance_scale must be >= 0.")
    if temporal_perturb_type not in ("base", "log"):
        raise ValueError(
            f"Unknown temporal_perturb_type={temporal_perturb_type!r}. Expected 'base' or 'log'."
        )
    print(f"--> MSE mode: {mse_mode}")
    if mse_mode == "score":
        print(f"    score_w_decode={score_w_decode}, decode_w={decode_w}")

    if freeze_score_in_cotrain:
        # Independent mode: VAE-only optimizer during cotrain phase
        # Score networks will only be trained during refine phase
        opt_joint = optim.AdamW(vae.parameters(), lr=cfg["lr_vae"], weight_decay=1e-4, betas=(0.9, float(cfg.get("adam_beta2", 0.95))))
        opt_tracking = None  # No tracking optimizer needed
        print("--> Independent mode: Score networks FROZEN during cotrain phase")
    else:
        # Standard co-training mode
        if cotrain_head == "lsi":
            opt_joint = optim.AdamW([
                {'params': vae.parameters(), 'lr': cfg["lr_vae"]},
                {'params': unet_lsi.parameters(), 'lr': cfg["lr_ldm"]/score_w_vae if score_w_vae > 0 else 0.0},
            ], weight_decay=1e-4, betas=(0.9, float(cfg.get("adam_beta2", 0.95))))
            opt_tracking = optim.AdamW(unet_control.parameters(), lr=cfg["lr_ldm"], weight_decay=1e-4, betas=(0.9, float(cfg.get("adam_beta2", 0.95))))
        else:  # cotrain_head == "control"
            opt_joint = optim.AdamW([
                {'params': vae.parameters(), 'lr': cfg["lr_vae"]},
                {'params': unet_control.parameters(), 'lr': cfg["lr_ldm"]/score_w_vae if score_w_vae > 0 else 0.0},
            ], weight_decay=1e-4, betas=(0.9, float(cfg.get("adam_beta2", 0.95))))
            opt_tracking = optim.AdamW(unet_lsi.parameters(), lr=cfg["lr_ldm"], weight_decay=1e-4, betas=(0.9, float(cfg.get("adam_beta2", 0.95))))

    # --- Cosine LR Schedulers (cotrain phase) ---
    sched_joint = CosineAnnealingLR(opt_joint, T_max=cfg["epochs_vae"], eta_min=1e-6)
    sched_tracking = CosineAnnealingLR(opt_tracking, T_max=cfg["epochs_vae"], eta_min=1e-6) if opt_tracking is not None else None

    lpips_fn = lpips.LPIPS(net='vgg').to(device) if LPIPS_AVAILABLE else None

    # --- PatchGAN Discriminator ---
    gan_w = float(cfg.get("gan_w", 0.0))
    gan_w_tdd_mult = float(cfg.get("gan_w_tdd_mult", 1.0))
    disc_start_epoch = int(cfg.get("disc_start_epoch", 0))
    gan_logit_clamp = float(cfg.get("gan_logit_clamp", 10.0))
    use_tdd_global = bool(cfg.get("time_cond_decoder", False))
    use_tdd_gan_global = use_tdd_global and bool(cfg.get("time_dependent_gan", True))
    gan_w_eff = gan_w * (gan_w_tdd_mult if use_tdd_gan_global else 1.0)

    if gan_w_eff > 0.0:
        if use_tdd_gan_global:
            disc = TimeCondPatchDiscriminator(
                in_channels=img_channels,
                ndf=int(cfg.get("disc_ndf", 64)),
                n_layers=int(cfg.get("disc_n_layers", 2)),
                time_emb_dim=int(cfg.get("disc_time_emb_dim", cfg.get("dec_time_emb_dim", 128))),
                schedule_type=str(cfg.get("time_schedule", "log_t")).lower(),
            ).to(device)
        else:
            disc = PatchDiscriminator(
                in_channels=img_channels,
                ndf=int(cfg.get("disc_ndf", 64)),
                n_layers=int(cfg.get("disc_n_layers", 2)),
            ).to(device)

        opt_disc = optim.Adam(disc.parameters(), lr=float(cfg.get("lr_disc", 5e-5)), betas=(0.5, 0.9))
    else:
        disc = None
        opt_disc = None

    # --- Fixed Evaluation Banks ---
    # Latent spatial size: img_size // 4 (two 2x downsampling in encoder)
    latent_spatial = effective_img_size // 4
    if cfg.get("use_fixed_eval_banks", True):
        N_test = len(test_l.dataset)
        latent_shape = (cfg["latent_channels"], latent_spatial, latent_spatial)
        seed = int(cfg.get("seed", 0))

        g_noise = torch.Generator(device="cpu").manual_seed(seed + 12345)
        fixed_noise_bank = torch.randn((N_test, *latent_shape), generator=g_noise)

        g_postA = torch.Generator(device="cpu").manual_seed(seed + 54321)
        g_postB = torch.Generator(device="cpu").manual_seed(seed + 98765)
        fixed_posterior_eps_bank_A = torch.randn((N_test, *latent_shape), generator=g_postA)
        fixed_posterior_eps_bank_B = torch.randn((N_test, *latent_shape), generator=g_postB)

        D = cfg["latent_channels"] * latent_spatial * latent_spatial
        K = int(cfg.get("sw2_n_projections", 1000))
        g_theta = torch.Generator(device="cpu").manual_seed(seed + 22222)
        theta = torch.randn((D, K), generator=g_theta)
        theta = theta / torch.norm(theta, dim=0, keepdim=True).clamp_min(1e-12)
        fixed_sw2_theta = theta
    else:
        fixed_noise_bank = None
        fixed_posterior_eps_bank_A = None
        fixed_posterior_eps_bank_B = None
        fixed_sw2_theta = None

    # --- Initialize Logging DataFrames ---
    loss_records = []
    eval_records = []

    print("--> Starting Dual Co-training...")
    for ep in range(cfg["epochs_vae"]):
        vae.train(); unet_lsi.train(); unet_control.train()
        metrics = {k: 0.0 for k in ["loss", "recon", "kl", "score_lsi", "score_control", "aux_lam", "aux_nu", "div", "perc", "stiff", "gan_d", "gan_g", "tdd"]}
        mu_stats = []
        logvar_stats = []

        for x, y in tqdm(train_l, desc=f"Ep {ep+1}", leave=False):
            x = x.to(device)
            y = y.to(device=device, dtype=torch.long)
            B = x.shape[0]

            # --- Classifier-Free Guidance (label dropout) ---
            p_uncond = float(cfg.get("cfg_label_dropout", 0.1))
            if p_uncond > 0.0:
                drop = (torch.rand(B, device=device) < p_uncond)
                y_in = y.clone()
                # reserve index num_classes as the unconditional "null" label
                y_in[drop] = int(cfg["num_classes"])
            else:
                y_in = y

            # --- VAE Encode ---
            mu_base, logvar_base = vae.encode(x)

            # --- Conditional geometry correction (ResidualGaussianAdapter) ---
            use_cond_enc = bool(cfg.get("use_cond_encoder", False))
            use_tdd =  bool(cfg.get("time_cond_decoder", False))
            use_tdd_gan = use_tdd and bool(cfg.get("time_dependent_gan", True))

            mu, logvar = vae.encode(x, y=y_in if use_cond_enc else None)
            logvar = torch.clamp(logvar, min=-30.0, max=20.0)

            # Reparameterize z0 from encoder
            z0 = vae.reparameterize(mu, logvar)

            if len(mu_stats) < 5:
                mu_stats.append(mu.detach())
                logvar_stats.append(logvar.detach())

            # --- Time / forward process (BEFORE decode) ---
            if frontier_tracker is not None and frontier_tracker.is_active:
                # Adaptive frontier sampling
                if use_ddim_times:
                    t_idx = frontier_tracker.sample_discrete(B, ou_sched["times"], device)
                    t = ou_sched["times"].gather(0, t_idx).float()
                    alpha = extract_schedule(ou_sched["alpha"], t_idx, z0.shape)
                    sigma = extract_schedule(ou_sched["sigma"], t_idx, z0.shape)
                elif str(cfg.get("time_schedule", "")).lower() in ("flow", "flow_matching"):
                    t = frontier_tracker.sample_flow(B, cfg["t_min"], cfg["t_max"], device)
                    alpha, sigma = get_flow_params(t.view(B, 1, 1, 1))
                else:
                    t = frontier_tracker.sample(B, device)
                    alpha, sigma = get_ou_params(t.view(B, 1, 1, 1))
            else:
                # Legacy sampling (or warmup fallback)
                if str(cfg.get("time_schedule", "")).lower() in ("flow", "flow_matching"):
                    t = sample_logit_normal_times(B, cfg["t_min"], cfg["t_max"], device)
                    alpha, sigma = get_flow_params(t.view(B, 1, 1, 1))
                else:
                    t = sample_log_uniform_times(B, cfg["t_min"], cfg["t_max"], device)
                    alpha, sigma = get_ou_params(t.view(B, 1, 1, 1))
                if use_ddim_times:
                    t_idx = torch.randint(0, T, (B,), device=device, dtype=torch.long)
                    t = ou_sched["times"].gather(0, t_idx).float()
                    alpha = extract_schedule(ou_sched["alpha"], t_idx, z0.shape)
                    sigma = extract_schedule(ou_sched["sigma"], t_idx, z0.shape)

            noise = torch.randn_like(z0)
            z_t = alpha * z0 + sigma * noise
            z_t_mse = z_t
            if temporal_variance_scale > 0.0 and mse_mode == "raw":
                delta_t = temporal_variance_scale * torch.randn_like(t)
                if temporal_perturb_type == "log":
                    tau = torch.exp(torch.log(t.clamp_min(float(cfg["t_min"]))) + delta_t)
                else:
                    tau = t + delta_t
                tau = tau.clamp(min=float(cfg["t_min"]), max=float(cfg["t_max"]))
                alpha_mse, sigma_mse = get_alpha_sigma_for_schedule(
                    tau.view(B, 1, 1, 1),
                    schedule_type=cfg.get("time_schedule", "log_snr"),
                    cosine_s=float(cfg.get("cosine_s", 0.008)),
                )
                z_t_mse = alpha_mse * z0 + sigma_mse * noise
            if cfg.get("train_on_mu", False):
                z_mu_t =  alpha * mu + sigma * noise
            else:
                z_mu_t = z_t

            frontier_iw = None  # importance weights from frontier tracker (if active)
            if frontier_tracker is not None and frontier_tracker.is_active:
                if bool(cfg.get("frontier_correct_score", False)):
                    ref = "uniform" if str(cfg.get("time_schedule", "")).lower() in ("flow", "flow_matching") else "log_uniform"
                    frontier_iw = frontier_tracker.importance_weights(t, reference=ref)

            var_0 = torch.exp(logvar)
            mu_t = alpha * mu
            var_t = (alpha**2) * var_0 + (sigma**2)

            # --- Decode moved below score prediction (needed for mse_mode='score') ---

            snr_downweight = bool(cfg.get("snr_downeight", False))
            if snr_downweight and use_tdd:
                snr_weight = (alpha.view(B, 1, 1, 1) ** 2).mean()  # scalar in [0,1]
            else:
                snr_weight = (alpha.view(B, 1, 1, 1) ** 2).mean()**0  # scalar in [0,1]

            # --- Perceptual loss (LPIPS) ---
            # Computed later to optionally use precision-masked weighting.
            perc = torch.tensor(0.0, device=device)
            # [NEW] Flexible KL Regularization on mu_t, var_t (time-dependent moments)
             # [NEW] Flexible KL Regularization on mu_t, var_t (time-dependent moments)
            reg_type = cfg.get("kl_reg_type", "mod") # 'normal', 'mod', or 'norm'
            logvar_t = torch.log(var_t + 1e-8)

            if reg_type == "normal":
                # Standard VAE KL applied to q(z_t|x): N(mu_t, var_t) vs N(0,1)
                kl = -0.5 * torch.mean(1 + logvar_t - mu_t.pow(2) - var_t)

            elif reg_type == "mod":
                # Modified KL: Energy/Trace control on z_t moments
                kl = -0.5 * torch.mean(1 - mu_t.pow(2) - var_t)

            elif reg_type == "norm":
                # Variance Anchor on z_t: penalize logvar_t deviation from 0
                kl = torch.mean(logvar_t.pow(2))

            elif reg_type == "vol":
                # Volume-preserving regularization on z_t
                logvar_t_clamped = torch.clamp(logvar_t, min=-10.0)
                log_det = torch.sum(logvar_t_clamped, dim=[1, 2, 3])  # [B]
                kl = - torch.mean(log_det)

            elif reg_type == "temporal":
                beta = float(cfg.get("temporal_beta", 2.0))
                kl = torch.mean(F.softplus(logvar * beta) / beta)

            else:
                raise ValueError(f"Unknown kl_reg_type: {reg_type}")

            lam_pred = None  # used for LPIPS precision masking (if enabled)
            nu_pred = None
            div_loss = torch.tensor(0.0, device=device)
            # --- Compute both score losses (for logging, even if frozen) ---
            cos_w = float(cfg.get("cosine_w", 1.0))
            resid = (z_t - mu_t) / (var_t + 1e-8)
            eps_target_lsi = sigma * resid  # E[eps | z_t, x]
            # Control head predicts the sampled eps directly.
            eps_target_control = noise

            if freeze_score_in_cotrain:
                score_loss_lsi = torch.tensor(0.0, device=device)
                score_loss_control = torch.tensor(0.0, device=device)
                aux_loss_lam = torch.tensor(0.0, device=device)
                aux_loss_nu  = torch.tensor(0.0, device=device)
            else:
                use_factored = bool(getattr(unet_lsi, "factored_head", False))
                if use_factored:
                    eps_pred_lsi, lam_pred, nu_pred = unet_lsi(
                        z_t, t, y_in,
                        return_components=True,
                        detach_components=True,
                    )
                    lam_tgt = (sigma / (var_t + 1e-8)).detach()
                    nu_tgt  = (lam_tgt * mu_t.detach())
                    aux_loss_lam = F.mse_loss(lam_pred, lam_tgt)
                    aux_loss_nu  = F.mse_loss(nu_pred,  nu_tgt)
                    if div_w != 0.0:
                        eps_safe = 1e-4
                        mu_consensus = nu_pred.detach() / (lam_pred.detach() + eps_safe)
                        div_loss = F.mse_loss(mu_t, mu_consensus)
                else:
                    eps_pred_lsi = unet_lsi(z_t, t, y_in)
                    aux_loss_lam = torch.tensor(0.0, device=device)
                    aux_loss_nu  = torch.tensor(0.0, device=device)

                # Score MSE with optional importance weighting
                if frontier_iw is not None:
                    per_sample_mse_lsi = (eps_pred_lsi - eps_target_lsi).pow(2).flatten(1).mean(1)
                    loss_mse_lsi = (frontier_iw * per_sample_mse_lsi).mean()
                else:
                    loss_mse_lsi = F.mse_loss(eps_pred_lsi, eps_target_lsi)
                loss_cos_lsi = (1.0 - F.cosine_similarity(eps_pred_lsi.flatten(1), eps_target_lsi.flatten(1), dim=1)).mean()
                score_loss_lsi = loss_mse_lsi + cos_w * loss_cos_lsi
                if use_factored:
                    #score_loss_lsi = score_loss_lsi + aux_head_w * (aux_loss_lam + aux_loss_nu)
                    score_loss_lsi = score_loss_lsi + aux_head_w * (aux_loss_lam)

                if cfg.get("train_on_mu", False):
                    eps_pred_control = unet_control(z_mu_t, t, y_in)
                else:
                    eps_pred_control = unet_control(z_t, t, y_in)
                # Control head: same IW treatment
                if frontier_iw is not None:
                    per_sample_mse_ctrl = (eps_pred_control - eps_target_control).pow(2).flatten(1).mean(1)
                    loss_mse_ctrl = (frontier_iw * per_sample_mse_ctrl).mean()
                else:
                    loss_mse_ctrl = F.mse_loss(eps_pred_control, eps_target_control)
                loss_cos_ctrl = (1.0 - F.cosine_similarity(eps_pred_control.flatten(1), eps_target_control.flatten(1), dim=1)).mean()
                score_loss_control = loss_mse_ctrl + cos_w * loss_cos_ctrl

            # --- Decode from z_t (or z_hat_0) with time-dependent decoder ---
            # The decoder input is consistent across ALL losses (MSE, LPIPS, GAN).
            # Only the gradient path through the score head differs by mode:
            #
            #   'raw':            D(z_t, t) everywhere.
            #   'score':          D(z_hat_0, t) everywhere.
            #                     Score head receives grad from MSE only (L2 info
            #                     channel identity) — NOT from LPIPS / GAN.
            #                     → two decoder fwd passes with different detach.
            #   'score_detached': D(z_hat_0_det, t) everywhere.
            #                     Score head sees no decoder gradients at all;
            #                     trained exclusively by CSEM score distillation.
            if mse_mode in ("score", "score_detached") and not freeze_score_in_cotrain:
                # Detached z_hat_0 (used by LPIPS/GAN in 'score', everything in 'score_detached')
                z_hat_0_det = (z_t - sigma * eps_pred_lsi.detach()) / (alpha + 1e-8)

                if mse_mode == "score":
                    # Attached path — score head feels MSE gradient only
                    z_hat_0_att = (z_t - sigma * eps_pred_lsi) / (alpha + 1e-8)
                    x_rec_mse = vae.decode(z_hat_0_att, t, y=y_in)
                    # Detached path — LPIPS / GAN (decoder still gets grad, score head shielded)
                    x_rec = vae.decode(z_hat_0_det, t, y=y_in)
                else:  # score_detached
                    x_rec = vae.decode(z_hat_0_det, t, y=y_in)
                    x_rec_mse = x_rec  # same tensor, score fully detached
            else:
                # raw mode — no score preprocessing
                x_rec = vae.decode(z_t, t, y=y_in)
                if temporal_variance_scale > 0.0:
                    x_rec_mse = vae.decode(z_t_mse, t, y=y_in)
                else:
                    x_rec_mse = x_rec

            # --- Asymmetric MSE weighting ---
            # In 'score' mode, the MSE couples both the score head (via z_hat_0_att)
            # and the decoder.  We want different gradient scales for each:
            #   score head  ←  score_w_decode * ∂MSE/∂score
            #   decoder     ←  decode_w       * ∂MSE/∂decoder
            #
            # Trick: recon_full (from x_rec_mse) sends grad to both score + decoder.
            #        recon_det  (from x_rec)     sends grad to decoder only.
            # Since the forward *values* are identical (detach doesn't change values),
            # the linear combination below yields the desired per-component scaling:
            #   score sees:   score_w_decode * grad
            #   decoder sees: score_w_decode * grad + (decode_w - score_w_decode) * grad = decode_w * grad
            if mse_mode == "score" and not freeze_score_in_cotrain:
                recon_full = F.mse_loss(x_rec_mse, x)
                recon_det  = F.mse_loss(x_rec, x)
                recon = score_w_decode * recon_full + (decode_w - score_w_decode) * recon_det
            else:
                recon = decode_w * F.mse_loss(x_rec_mse, x)

            # --- Stiffness penalty ---
            stiff_w = cfg.get("stiff_w", 0.0)
            if stiff_w > 0.0 and not freeze_score_in_cotrain:
                inv_var_t = 1.0 / (var_t + 1e-8)
                stiff_pen = inv_var_t.flatten(1).mean(dim=1).mean()
            else:
                stiff_pen = torch.tensor(0.0, device=device)

            # --- TDD loss removed: all reconstruction losses now applied directly to D(z_t, t) ---
            tdd_loss = torch.tensor(0.0, device=device)

            # --- Oracle TDD for frontier tracker ---
            # Measures the actual training-pipeline decode vs clean decode:
            #   raw mode:            D(z_t, t)       vs D(z_0, None)
            #   score/score_detached: D(z_hat_0, t)   vs D(z_0, None)
            # x_rec already reflects the correct decode for the active mse_mode.
            if frontier_tracker is not None:
                with torch.no_grad():
                    x_tdd_noisy = x_rec.detach()
                    x_tdd_clean = vae.decode(z0, y=y_in)  # D(z_0, y) with null-token fallback
                    oracle_tdd_mse = (x_tdd_noisy - x_tdd_clean).pow(2).flatten(1).mean(1)  # [B]
                    frontier_tracker.update(t.detach(), oracle_tdd_mse)

            # --- Perceptual loss (LPIPS) ---
            lpips_mode = str(cfg.get("lpips_mode", "snr")).lower()
            if LPIPS_AVAILABLE:
                x_3c = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
                x_rec_3c = x_rec.repeat(1, 3, 1, 1) if x_rec.shape[1] == 1 else x_rec
                lpips_per = lpips_fn(x_rec_3c, x_3c)
                # lpips returns [B, 1, 1, 1]; reduce to [B]
                lpips_per = lpips_per.view(B, -1).mean(dim=1) if lpips_per.dim() > 1 else lpips_per.view(B)
                if lpips_mode == "uniform":
                    perc = lpips_per.mean()
                elif lpips_mode == "snr":
                    # Legacy behavior: scalar SNR weight (optionally enabled by snr_downeight flag) times mean LPIPS.
                    perc = snr_weight * lpips_per.mean()
                elif lpips_mode == "frontier":
                    # Frontier-informed gating: w(t) = sigmoid((log R_cutoff - log R(t)) / τ)
                    # Active where R(t) is small (decoder output recognizable), off where R(t) is large.
                    # Falls back to gamma weighting during frontier warmup.
                    R_cutoff = float(cfg.get("frontier_R_cutoff", 0.05))
                    R_tau = float(cfg.get("frontier_R_tau", 1.0))
                    if frontier_tracker is not None and frontier_tracker.is_active:
                        frontier_gate = frontier_tracker.perceptual_weights(t, R_cutoff=R_cutoff, temperature=R_tau)
                        perc = (frontier_gate * lpips_per).mean()
                    else:
                        # Warmup fallback: use gamma (SNR-based) until frontier is active
                        gamma = (alpha ** 2) / ((alpha ** 2) + (sigma ** 2) + 1e-12)
                        gamma = gamma.view(B, -1).mean(dim=1)
                        perc = (gamma * lpips_per).mean()
                else:
                    # gamma(t) in [0,1], derived from SNR: gamma = alpha^2 / (alpha^2 + sigma^2) = sigmoid(logSNR).
                    gamma = (alpha ** 2) / ((alpha ** 2) + (sigma ** 2) + 1e-12)
                    gamma = gamma.view(B, -1).mean(dim=1)
                    if lpips_mode == "gamma":
                        perc = (gamma * lpips_per).mean()
                    elif lpips_mode == "prec_mask":
                        use_factored = bool(getattr(unet_lsi, "factored_head", False))
                        if (not freeze_score_in_cotrain) and use_factored and (lam_pred is not None):
                            # Detach precision head so LPIPS cannot be optimized by changing precision.
                            lam_mask = lam_pred.detach()
                            # Aggregate to a scalar per-sample, stabilize scale, batch-norm, then sigmoid-gate.
                            lam_s = lam_mask.float().mean(dim=(1, 2, 3))
                            lam_s = torch.log(lam_s.clamp_min(1e-8))
                            lam_bn = (lam_s - lam_s.mean()) / (lam_s.std(unbiased=False) + 1e-6)
                            gate = torch.sigmoid(lam_bn).detach()
                            perc = (gamma * gate * lpips_per).mean()
                        else:
                            # Fallback if factorized head not available: use gamma-only weighting.
                            perc = (gamma * lpips_per).mean()
                    else:
                        raise ValueError(f"Unknown lpips_mode: {lpips_mode!r}. Expected 'uniform', 'frontier', 'prec_mask', 'gamma', or 'snr'.")
            else:
                perc = torch.tensor(0.0, device=device)

            # --- PatchGAN adversarial loss ---
            use_gan = gan_w_eff > 0.0 and disc is not None and (ep + 1) >= disc_start_epoch

            if use_gan:
                if use_tdd_gan:
                    # Real target: pixel-space "best possible" denoising at the same time level (Wiener approximation)
                    with torch.no_grad():
                        x_real = wiener_reference_x0(
                            x,
                            alpha,
                            sigma,
                            alpha_min=float(cfg.get("wiener_alpha_min", 1e-4)),
                            max_var=float(cfg.get("wiener_max_var", 1e3)),
                        ).clamp(-1.0, 1.0)

                    # Discriminator step (detached reconstructions)
                    logits_real = disc(x_real, t)
                    logits_fake = disc(x_rec.detach(), t)
                    d_loss = hinge_d_loss(logits_real, logits_fake)
                    opt_disc.zero_grad()
                    d_loss.backward()
                    opt_disc.step()

                    # Generator loss (non-detached, logits clamped to bound grad magnitude)
                    g_logits = disc(x_rec, t).clamp(-gan_logit_clamp, gan_logit_clamp)
                    g_loss = hinge_g_loss(g_logits)
                else:
                    # Legacy non-time-conditioned discriminator against true x0
                    logits_real = disc(x)
                    logits_fake = disc(x_rec.detach())
                    d_loss = hinge_d_loss(logits_real, logits_fake)
                    opt_disc.zero_grad()
                    d_loss.backward()
                    opt_disc.step()
                    g_logits = disc(x_rec).clamp(-gan_logit_clamp, gan_logit_clamp)
                    g_loss = hinge_g_loss(g_logits)
                # --- GAN generator time weighting ---
                gan_time_mode = str(cfg.get("gan_time_weight", "uniform")).lower()
                if gan_time_mode != "uniform" and use_tdd:
                    if gan_time_mode == "frontier":
                        # Frontier-informed gating: same R(t) gate as LPIPS.
                        # Per-sample weighting: samples at high t (large R) get near-zero GAN weight.
                        R_cutoff = float(cfg.get("frontier_R_cutoff", 0.05))
                        R_tau = float(cfg.get("frontier_R_tau", 1.0))
                        if frontier_tracker is not None and frontier_tracker.is_active:
                            gan_frontier_gate = frontier_tracker.perceptual_weights(t, R_cutoff=R_cutoff, temperature=R_tau)
                            # Recompute g_loss with per-sample weighting
                            g_loss_per_sample = -g_logits.mean(dim=list(range(1, g_logits.dim())))  # [B]
                            g_loss = (gan_frontier_gate * g_loss_per_sample).mean()
                        else:
                            # Warmup fallback: use gamma (SNR-based) weighting
                            snr = (alpha ** 2) / (sigma ** 2 + 1e-12)
                            gan_tw = snr / (snr + 1.0)
                            g_loss = g_loss * gan_tw.view(B, -1).mean().detach()
                    else:
                        snr = (alpha ** 2) / (sigma ** 2 + 1e-12)
                        if gan_time_mode == "gamma":
                            gan_tw = snr / (snr + 1.0)            # sigmoid(logSNR), in [0,1]
                        elif gan_time_mode == "snr":
                            gan_tw = snr
                        elif gan_time_mode == "snr2":
                            gan_tw = snr ** 2
                        else:
                            raise ValueError(f"Unknown gan_time_weight: {gan_time_mode!r}. Expected 'uniform', 'frontier', 'gamma', 'snr', or 'snr2'.")
                        g_loss = g_loss * gan_tw.view(B, -1).mean().detach()
            else:
                d_loss = torch.tensor(0.0, device=device)
                g_loss = torch.tensor(0.0, device=device)

            # --- Joint loss ---
            if freeze_score_in_cotrain:
                # Independent mode: VAE-only loss (no score, no stiffness)
                loss_joint = recon + cfg["perc_w"]*perc + cfg["kl_w"]*kl + gan_w_eff*g_loss
            else:
                # Co-training mode: include score loss
                score_w_vae = cfg.get("score_w_vae", cfg["score_w"])
                if cotrain_head == "lsi":
                    loss_joint = recon + cfg["perc_w"]*perc + cfg["kl_w"]*kl + score_w_vae*score_loss_lsi + div_w*div_loss + stiff_w*stiff_pen + gan_w_eff*g_loss
                else:  # cotrain_head == "control"
                    loss_joint = recon + cfg["perc_w"]*perc + cfg["kl_w"]*kl + score_w_vae*score_loss_control + div_w*div_loss + stiff_w*stiff_pen + gan_w_eff*g_loss

            opt_joint.zero_grad()
            loss_joint.backward()
            if freeze_score_in_cotrain:
                nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            elif cotrain_head == "lsi":
                nn.utils.clip_grad_norm_(list(vae.parameters()) + list(unet_lsi.parameters()), 1.0)
            else:
                nn.utils.clip_grad_norm_(list(vae.parameters()) + list(unet_control.parameters()), 1.0)
            opt_joint.step()

            # --- EMA Update and Tracking (only if not frozen) ---
            if not freeze_score_in_cotrain:
                # --- EMA Update (co-trained head) ---
                with torch.no_grad():
                    if cotrain_head == "lsi":
                        for p_online, p_ema in zip(unet_lsi.parameters(), unet_lsi_ema.parameters()):
                            p_ema.data.mul_(ema_decay).add_(p_online.data, alpha=1 - ema_decay)
                    else:
                        for p_online, p_ema in zip(unet_control.parameters(), unet_control_ema.parameters()):
                            p_ema.data.mul_(ema_decay).add_(p_online.data, alpha=1 - ema_decay)

                # --- Tracking head (trains on detached latents) ---
                z_t_detached = z_t.detach()
                z_mu_t_detached = z_mu_t.detach()
                eps_target_control_det = eps_target_control.detach()
                eps_target_lsi_det = eps_target_lsi.detach()

                if cotrain_head == "lsi":
                    # Control head tracks the velocity baseline
                    x_in = z_mu_t_detached if cfg.get("train_on_mu", False) else z_t_detached
                    eps_pred_control_tracking = unet_control(x_in, t, y_in)
                    loss_mse = F.mse_loss(eps_pred_control_tracking, eps_target_control_det)
                    loss_cos = (1.0 - F.cosine_similarity(eps_pred_control_tracking.flatten(1), eps_target_control_det.flatten(1), dim=1)).mean()
                    tracking_loss = cfg["score_w"] * (loss_mse + cos_w * loss_cos)
                else:
                    # LSI head tracks the CSEM velocity target
                    use_factored = bool(getattr(unet_lsi, "factored_head", False))
                    if use_factored:
                        eps_pred_lsi_tracking, lam_pred, nu_pred = unet_lsi(
                            z_t_detached, t, y_in,
                            return_components=True,
                            detach_components=True,
                        )
                        lam_tgt = (sigma / (var_t + 1e-8)).detach()
                        nu_tgt  = (lam_tgt * mu_t.detach())
                        aux_loss_lam_tr = F.mse_loss(lam_pred, lam_tgt)
                        aux_loss_nu_tr  = F.mse_loss(nu_pred,  nu_tgt)
                    else:
                        eps_pred_lsi_tracking = unet_lsi(z_t_detached, t, y_in)
                        aux_loss_lam_tr = torch.tensor(0.0, device=device)
                        aux_loss_nu_tr  = torch.tensor(0.0, device=device)

                    loss_mse = F.mse_loss(eps_pred_lsi_tracking, eps_target_lsi_det)
                    loss_cos = (1.0 - F.cosine_similarity(eps_pred_lsi_tracking.flatten(1), eps_target_lsi_det.flatten(1), dim=1)).mean()
                    tracking_loss = cfg["score_w"] * (loss_mse + cos_w * loss_cos)
                    if use_factored:
                        #tracking_loss = tracking_loss + aux_head_w * (aux_loss_lam_tr + aux_loss_nu_tr)
                        tracking_loss = tracking_loss + aux_head_w * (aux_loss_lam_tr)


                opt_tracking.zero_grad()
                tracking_loss.backward()
                if cotrain_head == "lsi":
                    nn.utils.clip_grad_norm_(unet_control.parameters(), 1.0)
                else:
                    nn.utils.clip_grad_norm_(unet_lsi.parameters(), 1.0)
                opt_tracking.step()

                # --- EMA Update (tracking head) ---
                with torch.no_grad():
                    if cotrain_head == "lsi":
                        for p_online, p_ema in zip(unet_control.parameters(), unet_control_ema.parameters()):
                            p_ema.data.mul_(ema_decay).add_(p_online.data, alpha=1 - ema_decay)
                    else:
                        for p_online, p_ema in zip(unet_lsi.parameters(), unet_lsi_ema.parameters()):
                            p_ema.data.mul_(ema_decay).add_(p_online.data, alpha=1 - ema_decay)

            metrics["loss"] += loss_joint.item()
            metrics["recon"] += recon.item()
            metrics["kl"] += kl.item()
            metrics["score_lsi"] += score_loss_lsi.item()
            metrics["score_control"] += score_loss_control.item()
            metrics["aux_lam"] += aux_loss_lam.item()
            metrics["aux_nu"] += aux_loss_nu.item()
            metrics["div"] += div_loss.item()
            metrics["perc"] += perc.item()
            metrics["stiff"] += stiff_pen.item()
            metrics["gan_d"] += d_loss.item()
            metrics["gan_g"] += g_loss.item()
            metrics["tdd"] += tdd_loss.item()

        # --- Compute epoch averages and log ---
        n_batches = len(train_l)
        epoch_metrics = {
            "epoch": ep + 1,
            "stage": "cotrain",
            "loss": metrics["loss"] / n_batches,
            "recon": metrics["recon"] / n_batches,
            "kl": metrics["kl"] / n_batches,
            "score_lsi": metrics["score_lsi"] / n_batches,
            "score_control": metrics["score_control"] / n_batches,
            "aux_lam": metrics["aux_lam"] / n_batches,
            "aux_nu": metrics["aux_nu"] / n_batches,
            "div": metrics["div"] / n_batches,
            "perc": metrics["perc"] / n_batches,
            "stiff": metrics["stiff"] / n_batches,
            "gan_d": metrics["gan_d"] / n_batches,
            "gan_g": metrics["gan_g"] / n_batches,
            "tdd": metrics["tdd"] / n_batches,
        }
        loss_records.append(epoch_metrics)

        print(f"Ep {ep+1} | LSI: {epoch_metrics['score_lsi']:.4f} | Ctrl: {epoch_metrics['score_control']:.4f} | AuxLam: {epoch_metrics['aux_lam']:.4f} | AuxNu: {epoch_metrics['aux_nu']:.4f} | Div: {epoch_metrics['div']:.4f} | "
              f"Rec: {epoch_metrics['recon']:.4f} | KL: {epoch_metrics['kl']:.4f} | Perc: {epoch_metrics['perc']:.4f} | "
              f"Stiff: {epoch_metrics['stiff']:.4f} | GAN_d: {epoch_metrics['gan_d']:.4f} | GAN_g: {epoch_metrics['gan_g']:.4f} | TDD: {epoch_metrics['tdd']:.4f}")

        # --- Frontier tracker diagnostics ---
        if frontier_tracker is not None:
            ft_diag = frontier_tracker.get_diagnostics()
            for k, v in ft_diag.items():
                epoch_metrics[k] = v
            if frontier_tracker.is_active:
                print(f"  Frontier: peak_t={ft_diag['frontier/peak_t']:.4f} "
                      f"max_w={ft_diag['frontier/max_weight']:.3f} "
                      f"entropy_ratio={ft_diag['frontier/entropy_ratio']:.2f}")
            if (ep + 1) % max(eval_freq_cotrain, 10) == 0 and frontier_tracker.is_active:
                _frontier_eval_dir = os.path.join(results_dir, "evals", f"eval_{ep+1}")
                os.makedirs(_frontier_eval_dir, exist_ok=True)
                plot_frontier_diagnostics(
                    frontier_tracker,
                    os.path.join(_frontier_eval_dir, f"frontier_ep{ep+1}.png"),
                    ep + 1)

        if len(mu_stats) > 0:
            log_latent_stats("VAE_Train", torch.cat(mu_stats, 0), torch.cat(logvar_stats, 0))

        # --- Step cosine LR schedulers ---
        sched_joint.step()
        if sched_tracking is not None:
            sched_tracking.step()

        # --- Evaluation (skip if score frozen - no LDM to evaluate) ---
        should_eval_cotrain = (not freeze_score_in_cotrain) and ((ep + 1) % eval_freq_cotrain == 0)

        if should_eval_cotrain:
            # For cotrain, epoch = LDM training epoch (1-indexed)
            ldm_epoch = ep + 1

            # Evaluate LSI
            results_lsi = evaluate_current_state(
                ldm_epoch,
                "LSI_Diff",
                vae,
                unet_lsi_ema,
                test_l,
                cfg,
                device,
                lpips_fn,
                fixed_noise_bank=fixed_noise_bank,
                fixed_posterior_eps_bank_A=fixed_posterior_eps_bank_A,
                fixed_posterior_eps_bank_B=fixed_posterior_eps_bank_B,
                fixed_sw2_theta=fixed_sw2_theta,
                results_dir=results_dir,
                fid_model=fid_model,
                use_lenet_fid=use_lenet_fid,
                frontier_tracker=frontier_tracker,
            )
            if results_lsi is not None:
                results_lsi["epoch"] = ldm_epoch  # LDM epoch for comparison
                results_lsi["stage"] = "cotrain"
                results_lsi["tag"] = "LSI_Diff"
                eval_records.append(results_lsi)

            results_ctrl = results_lsi
            '''
            # Evaluate Control
            results_ctrl = evaluate_current_state(
                ldm_epoch,
                "Ctrl_Diff",
                vae,
                unet_control_ema,
                test_l,
                cfg,
                device,
                lpips_fn,
                fixed_noise_bank=fixed_noise_bank,
                fixed_posterior_eps_bank_A=fixed_posterior_eps_bank_A,
                fixed_posterior_eps_bank_B=fixed_posterior_eps_bank_B,
                fixed_sw2_theta=fixed_sw2_theta,
                results_dir=results_dir,
                fid_model=fid_model,
                use_lenet_fid=use_lenet_fid,
                frontier_tracker=frontier_tracker,
            )
            '''
            if results_ctrl is not None:
                results_ctrl["epoch"] = ldm_epoch  # LDM epoch for comparison
                results_ctrl["stage"] = "cotrain"
                results_ctrl["tag"] = "Ctrl_Diff"
                eval_records.append(results_ctrl)

            # --- Save Checkpoints at Evaluation ---
            print(f"  Saving checkpoints at cotrain eval (LDM epoch {ldm_epoch})...")
            save_checkpoint(vae.state_dict(), os.path.join(cfg["ckpt_dir"], "vae_cotrained.pt"))
            save_checkpoint(unet_lsi_ema.state_dict(), os.path.join(cfg["ckpt_dir"], "unet_lsi.pt"))
            save_checkpoint(unet_control_ema.state_dict(), os.path.join(cfg["ckpt_dir"], "unet_control.pt"))


    # ===========================================================================
    # REFINEMENT STAGE: Freeze VAE, train only score networks
    # ===========================================================================
    epochs_refine = cfg.get("epochs_refine", 20)
    lr_refine = cfg.get("lr_refine", 1e-4)

    if epochs_refine > 0:
        print(f"\n--> Starting Refinement Stage ({epochs_refine} epochs, lr={lr_refine})...")

        # Re-initialize frontier tracker for refinement (VAE now frozen)
        if frontier_tracker is not None:
            frontier_tracker = ReconFrontierTracker(
                t_min=float(cfg["t_min"]), t_max=float(cfg["t_max"]),
                n_bins=int(cfg.get("adaptive_time_nbins", 100)),
                ema_decay=float(cfg.get("adaptive_time_ema", 0.99)),
                floor_weight=float(cfg.get("adaptive_time_floor", 0.02)),
                warmup_steps=int(cfg.get("adaptive_time_warmup", 500)),
                min_counts_per_bin=int(cfg.get("adaptive_time_min_counts", 5)),
                smooth_sigma=float(cfg.get("adaptive_time_smooth_sigma", 3.0)),
                device=device,
            )
            print("--> Frontier tracker re-initialized for refinement phase")

        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False

        opt_lsi_refine = optim.AdamW(unet_lsi.parameters(), lr=lr_refine, weight_decay=1e-4, betas=(0.9, float(cfg.get("adam_beta2", 0.95))))
        opt_control_refine = optim.AdamW(unet_control.parameters(), lr=lr_refine, weight_decay=1e-4, betas=(0.9, float(cfg.get("adam_beta2", 0.95))))

        # --- Cosine LR Schedulers (refine phase) ---
        sched_lsi_refine = CosineAnnealingLR(opt_lsi_refine, T_max=epochs_refine, eta_min=1e-7)
        sched_control_refine = CosineAnnealingLR(opt_control_refine, T_max=epochs_refine, eta_min=1e-7)

        for ep in range(epochs_refine):
            unet_lsi.train(); unet_control.train()
            metrics_refine = {k: 0.0 for k in ["score_lsi", "score_control", "aux_lam", "aux_nu"]}

            for x, y in tqdm(train_l, desc=f"Refine Ep {ep+1}", leave=False):
                x = x.to(device)
                y = y.to(device=device, dtype=torch.long)
                B = x.shape[0]

                # --- Classifier-Free Guidance (label dropout) ---
                p_uncond = float(cfg.get("cfg_label_dropout", 0.1))
                if p_uncond > 0.0:
                    drop = (torch.rand(B, device=device) < p_uncond)
                    y_in = y.clone()
                    y_in[drop] = int(cfg["num_classes"])
                else:
                    y_in = y

                use_cond_enc = bool(cfg.get("use_cond_encoder", False))
                with torch.no_grad():
                    mu, logvar = vae.encode(x, y=y_in if use_cond_enc else None)
                    logvar = torch.clamp(logvar, min=-30.0, max=20.0)
                    z0 = vae.reparameterize(mu, logvar)

                # --- Time / forward process (adaptive or legacy) ---
                if frontier_tracker is not None and frontier_tracker.is_active:
                    if use_ddim_times:
                        t_idx = frontier_tracker.sample_discrete(B, ou_sched["times"], device)
                        t = ou_sched["times"].gather(0, t_idx).float()
                        alpha = extract_schedule(ou_sched["alpha"], t_idx, z0.shape)
                        sigma = extract_schedule(ou_sched["sigma"], t_idx, z0.shape)
                    elif str(cfg.get("time_schedule", "")).lower() in ("flow", "flow_matching"):
                        t = frontier_tracker.sample_flow(B, cfg["t_min"], cfg["t_max"], device)
                        alpha, sigma = get_flow_params(t.view(B, 1, 1, 1))
                    else:
                        t = frontier_tracker.sample(B, device)
                        alpha, sigma = get_ou_params(t.view(B, 1, 1, 1))
                else:
                    if str(cfg.get("time_schedule", "")).lower() in ("flow", "flow_matching"):
                        t = sample_logit_normal_times(B, cfg["t_min"], cfg["t_max"], device)
                        alpha, sigma = get_flow_params(t.view(B, 1, 1, 1))
                    else:
                        t = sample_log_uniform_times(B, cfg["t_min"], cfg["t_max"], device)
                        alpha, sigma = get_ou_params(t.view(B, 1, 1, 1))
                    if use_ddim_times:
                        t_idx = torch.randint(0, T, (B,), device=device, dtype=torch.long)
                        t = ou_sched["times"].gather(0, t_idx).float()
                        alpha = extract_schedule(ou_sched["alpha"], t_idx, z0.shape)
                        sigma = extract_schedule(ou_sched["sigma"], t_idx, z0.shape)

                noise = torch.randn_like(z0)
                z_t = alpha * z0 + sigma * noise
                if cfg.get("train_on_mu", False):
                    z_mu_t = alpha * mu + sigma * noise
                else:
                    z_mu_t = z_t

                # --- LSI Score Training (eps-parameterization) ---
                cos_w = float(cfg.get("cosine_w", 1.0))
                var_0 = torch.exp(logvar)
                mu_t = alpha * mu
                var_t = (alpha ** 2) * var_0 + (sigma ** 2)

                resid = (z_t - mu_t) / (var_t + 1e-8)
                eps_target_lsi = sigma * resid  # E[eps | z_t, x]

                # --- Oracle TDD for frontier tracker (refine: VAE frozen) ---
                frontier_iw_refine = None
                if frontier_tracker is not None:
                    with torch.no_grad():
                        x_rec_tdd = vae.decode(z_t, t, y=y_in)
                        x_clean_tdd = vae.decode(z0, y=y_in)
                        oracle_tdd_mse = (x_rec_tdd - x_clean_tdd).pow(2).flatten(1).mean(1)
                        frontier_tracker.update(t.detach(), oracle_tdd_mse)
                    if bool(cfg.get("frontier_correct_score", False)):
                        ref = "uniform" if str(cfg.get("time_schedule", "")).lower() in ("flow", "flow_matching") else "log_uniform"
                        frontier_iw_refine = frontier_tracker.importance_weights(t, reference=ref)

                use_factored = bool(getattr(unet_lsi, "factored_head", False))
                if use_factored:
                    eps_pred_lsi, lam_pred, nu_pred = unet_lsi(
                        z_t, t, y_in,
                        return_components=True,
                        detach_components=True,
                    )
                    lam_tgt = (sigma / (var_t + 1e-8)).detach()
                    nu_tgt  = (lam_tgt * mu_t.detach())
                    aux_loss_lam = F.mse_loss(lam_pred, lam_tgt)
                    aux_loss_nu  = F.mse_loss(nu_pred,  nu_tgt)
                else:
                    eps_pred_lsi = unet_lsi(z_t, t, y_in)
                    aux_loss_lam = torch.tensor(0.0, device=device)
                    aux_loss_nu  = torch.tensor(0.0, device=device)

                # Score MSE with optional importance weighting (refine)
                if frontier_iw_refine is not None:
                    per_sample_mse_lsi = (eps_pred_lsi - eps_target_lsi).pow(2).flatten(1).mean(1)
                    loss_mse_lsi = (frontier_iw_refine * per_sample_mse_lsi).mean()
                else:
                    loss_mse_lsi = F.mse_loss(eps_pred_lsi, eps_target_lsi)
                loss_cos_lsi = (1.0 - F.cosine_similarity(eps_pred_lsi.flatten(1), eps_target_lsi.flatten(1), dim=1)).mean()
                score_loss_lsi = loss_mse_lsi + cos_w * loss_cos_lsi
                if use_factored:
                    #score_loss_lsi = score_loss_lsi + aux_head_w * (aux_loss_lam + aux_loss_nu)
                    score_loss_lsi = score_loss_lsi + aux_head_w * (aux_loss_lam)

                opt_lsi_refine.zero_grad()
                score_loss_lsi.backward()
                nn.utils.clip_grad_norm_(unet_lsi.parameters(), 1.0)
                opt_lsi_refine.step()

                # --- EMA Update (LSI) ---
                with torch.no_grad():
                    for p_online, p_ema in zip(unet_lsi.parameters(), unet_lsi_ema.parameters()):
                        p_ema.data.mul_(ema_decay).add_(p_online.data, alpha=1 - ema_decay)

                # --- Control (eps baseline) ---
                eps_target_control = noise
                if cfg.get("train_on_mu", False):
                    eps_pred_control = unet_control(z_mu_t, t, y_in)
                else:
                    eps_pred_control = unet_control(z_t, t, y_in)

                if frontier_iw_refine is not None:
                    per_sample_mse_ctrl = (eps_pred_control - eps_target_control).pow(2).flatten(1).mean(1)
                    loss_mse_ctrl = (frontier_iw_refine * per_sample_mse_ctrl).mean()
                else:
                    loss_mse_ctrl = F.mse_loss(eps_pred_control, eps_target_control)
                loss_cos_ctrl = (1.0 - F.cosine_similarity(eps_pred_control.flatten(1), eps_target_control.flatten(1), dim=1)).mean()
                score_loss_control = loss_mse_ctrl + cos_w * loss_cos_ctrl

                opt_control_refine.zero_grad()
                score_loss_control.backward()
                nn.utils.clip_grad_norm_(unet_control.parameters(), 1.0)
                opt_control_refine.step()

                # --- EMA Update (Control) ---
                with torch.no_grad():
                    for p_online, p_ema in zip(unet_control.parameters(), unet_control_ema.parameters()):
                        p_ema.data.mul_(ema_decay).add_(p_online.data, alpha=1 - ema_decay)

                metrics_refine["score_lsi"] += score_loss_lsi.item()
                metrics_refine["score_control"] += score_loss_control.item()
                metrics_refine["aux_lam"] += aux_loss_lam.item()
                metrics_refine["aux_nu"] += aux_loss_nu.item()

            # --- Log Refinement Epoch ---
            n_batches = len(train_l)
            global_epoch = cfg["epochs_vae"] + ep + 1
            epoch_metrics = {
                "epoch": global_epoch,
                "stage": "refine",
                "loss": 0.0,  # No joint loss in refinement
                "recon": 0.0,
                "kl": 0.0,
                "score_lsi": metrics_refine["score_lsi"] / n_batches,
                "score_control": metrics_refine["score_control"] / n_batches,
                "aux_lam": metrics_refine["aux_lam"] / n_batches,
                "aux_nu": metrics_refine["aux_nu"] / n_batches,
                "perc": 0.0,
            }
            loss_records.append(epoch_metrics)

            print(f"Refine Ep {ep+1} | LSI: {epoch_metrics['score_lsi']:.4f} | "
                  f"Ctrl: {epoch_metrics['score_control']:.4f} | AuxLam: {epoch_metrics['aux_lam']:.4f} | AuxNu: {epoch_metrics['aux_nu']:.4f}")

            # --- Step cosine LR schedulers ---
            sched_lsi_refine.step()
            sched_control_refine.step()

            if (ep + 1) % eval_freq_refine == 0:
                # For refine phase:
                # - Cotrain mode: LDM epoch = epochs_vae + refine_epoch
                # - Indep mode: LDM epoch = refine_epoch (VAE phase had no LDM training)
                if freeze_score_in_cotrain:
                    ldm_epoch = ep + 1  # Indep: refine is the only LDM training
                else:
                    ldm_epoch = cfg["epochs_vae"] + ep + 1  # Cotrain: add cotrain epochs

                results_lsi = evaluate_current_state(
                        ldm_epoch,
                        "LSI_Diff_Refine",
                        vae,
                        unet_lsi_ema,
                        test_l,
                        cfg,
                        device,
                        lpips_fn,
                        fixed_noise_bank=fixed_noise_bank,
                        fixed_posterior_eps_bank_A=fixed_posterior_eps_bank_A,
                        fixed_posterior_eps_bank_B=fixed_posterior_eps_bank_B,
                        fixed_sw2_theta=fixed_sw2_theta,
                        results_dir=results_dir,
                        fid_model=fid_model,
                        use_lenet_fid=use_lenet_fid,
                        frontier_tracker=frontier_tracker,
                )
                if results_lsi is not None:
                    results_lsi["epoch"] = ldm_epoch
                    results_lsi["stage"] = "refine"
                    results_lsi["tag"] = "LSI_Diff_Refine"
                    eval_records.append(results_lsi)

                results_ctrl = evaluate_current_state(
                        ldm_epoch,
                        "Ctrl_Diff_Refine",
                        vae,
                        unet_control_ema,
                        test_l,
                        cfg,
                        device,
                        lpips_fn,
                        fixed_noise_bank=fixed_noise_bank,
                        fixed_posterior_eps_bank_A=fixed_posterior_eps_bank_A,
                        fixed_posterior_eps_bank_B=fixed_posterior_eps_bank_B,
                        fixed_sw2_theta=fixed_sw2_theta,
                        results_dir=results_dir,
                        fid_model=fid_model,
                        use_lenet_fid=use_lenet_fid,
                        frontier_tracker=frontier_tracker,
                )
                if results_ctrl is not None:
                    results_ctrl["epoch"] = ldm_epoch
                    results_ctrl["stage"] = "refine"
                    results_ctrl["tag"] = "Ctrl_Diff_Refine"
                    eval_records.append(results_ctrl)

                # --- Save Checkpoints at Evaluation ---
                print(f"  Saving checkpoints at refine eval (LDM epoch {ldm_epoch})...")
                save_checkpoint(vae.state_dict(), os.path.join(cfg["ckpt_dir"], "vae_cotrained.pt"))
                save_checkpoint(unet_lsi_ema.state_dict(), os.path.join(cfg["ckpt_dir"], "unet_lsi.pt"))
                save_checkpoint(unet_control_ema.state_dict(), os.path.join(cfg["ckpt_dir"], "unet_control.pt"))

    # --- Save Checkpoints ---
    save_checkpoint(vae.state_dict(), os.path.join(cfg["ckpt_dir"], "vae_cotrained.pt"))
    save_checkpoint(unet_lsi_ema.state_dict(), os.path.join(cfg["ckpt_dir"], "unet_lsi.pt"))
    save_checkpoint(unet_control_ema.state_dict(), os.path.join(cfg["ckpt_dir"], "unet_control.pt"))

    # --- Create DataFrames ---
    loss_df = pd.DataFrame(loss_records)
    eval_df = pd.DataFrame(eval_records) if eval_records else pd.DataFrame()

    # --- Save DataFrames ---
    save_dataframes(loss_df, eval_df, results_dir)

    # --- Generate Visualizations ---
    if len(loss_df) > 0 and len(eval_df) > 0:
        generate_all_visualizations(loss_df, eval_df, results_dir)
    else:
        print("--> Warning: Insufficient data for visualization generation")

    # --- Save run configuration ---
    cfg_path = os.path.join(results_dir, "config.txt")
    with open(cfg_path, "w") as f:
        for k, v in cfg.items():
            f.write(f"{k}: {v}\n")
    print(f"--> Saved configuration to {cfg_path}")


    # --- ZIP the entire results directory (DEFAULT) ---
    zip_path = zip_results_dir(results_dir)
    print(f"--> Zipped results to: {zip_path}")

    print(f"\n{'='*60}")
    print(f"Training complete! All results saved to: {results_dir}")
    print(f"{'='*60}")

    return loss_df, eval_df

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_cotrain_vs_indep_comparison(cfg_cotrain, cfg_indep):
    """
    Run both co-trained and independent training experiments,
    then generate comparison plots.

    Args:
        cfg_cotrain: Config for co-training experiment
        cfg_indep: Config for independent experiment

    Returns:
        dict containing all results and paths
    """
    print("=" * 70)
    print("RUNNING CO-TRAINED vs INDEPENDENT TRAINING COMPARISON")
    print("=" * 70)

    # Create master results directory
    master_results_dir = cfg_cotrain.get("master_results_dir", "run_results_comparison")
    if os.path.exists(master_results_dir):
        shutil.rmtree(master_results_dir)
    os.makedirs(master_results_dir, exist_ok=True)

    # Calculate total LDM epochs for each
    cotrain_ldm_epochs = cfg_cotrain["epochs_vae"] + cfg_cotrain["epochs_refine"]
    indep_ldm_epochs = cfg_indep["epochs_refine"]  # Only refine has LDM training

    print(f"\nExperiment setup:")
    print(f"  Co-trained: {cfg_cotrain['epochs_vae']} cotrain + {cfg_cotrain['epochs_refine']} refine = {cotrain_ldm_epochs} LDM epochs")
    print(f"  Independent: {cfg_indep['epochs_vae']} VAE-only + {cfg_indep['epochs_refine']} refine = {indep_ldm_epochs} LDM epochs")

    # --- EXPERIMENT 1: CO-TRAINED ---
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: CO-TRAINED (CSEM)")
    print("=" * 70)

    cfg_cotrain["results_dir"] = os.path.join(master_results_dir, "run_results_cotrain")

    seed_everything(cfg_cotrain["seed"])
    loss_df_cotrain, eval_df_cotrain = train_vae_cotrained_cond(cfg_cotrain)

    # --- EXPERIMENT 2: INDEPENDENT ---
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: INDEPENDENT (Two-Stage)")
    print("=" * 70)

    cfg_indep["results_dir"] = os.path.join(master_results_dir, "run_results_indep")

    seed_everything(cfg_indep["seed"])  # Reset seed for fair comparison
    loss_df_indep, eval_df_indep = train_vae_cotrained_cond(cfg_indep)

    # --- GENERATE COMPARISON PLOTS ---
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("=" * 70)

    generate_comparison_visualizations(eval_df_cotrain, eval_df_indep, master_results_dir)

    # --- Save Combined DataFrames ---
    combined_dir = os.path.join(master_results_dir, "combined_dataframes")
    os.makedirs(combined_dir, exist_ok=True)

    # Add source column to distinguish
    eval_df_cotrain_save = eval_df_cotrain.copy()
    eval_df_cotrain_save["experiment"] = "cotrained"
    eval_df_indep_save = eval_df_indep.copy()
    eval_df_indep_save["experiment"] = "independent"

    combined_eval_df = pd.concat([eval_df_cotrain_save, eval_df_indep_save], ignore_index=True)
    combined_eval_df.to_csv(os.path.join(combined_dir, "combined_eval_metrics.csv"), index=False)

    loss_df_cotrain_save = loss_df_cotrain.copy()
    loss_df_cotrain_save["experiment"] = "cotrained"
    loss_df_indep_save = loss_df_indep.copy()
    loss_df_indep_save["experiment"] = "independent"

    combined_loss_df = pd.concat([loss_df_cotrain_save, loss_df_indep_save], ignore_index=True)
    combined_loss_df.to_csv(os.path.join(combined_dir, "combined_loss_history.csv"), index=False)

    print(f"--> Saved combined eval metrics to {combined_dir}/combined_eval_metrics.csv")
    print(f"--> Saved combined loss history to {combined_dir}/combined_loss_history.csv")

    # --- ZIP the entire comparison results directory ---
    zip_path = zip_results_dir(master_results_dir)
    print(f"--> Zipped all comparison results to: {zip_path}")

    print(f"\n{'='*70}")
    print(f"COMPARISON COMPLETE!")
    print(f"All results saved to: {master_results_dir}")
    print(f"  - Co-trained results: {master_results_dir}/run_results_cotrain/")
    print(f"  - Independent results: {master_results_dir}/run_results_indep/")
    print(f"  - Comparison plots: {master_results_dir}/comparison_plots/")
    print(f"{'='*70}")

    return {
        "loss_df_cotrain": loss_df_cotrain,
        "eval_df_cotrain": eval_df_cotrain,
        "loss_df_indep": loss_df_indep,
        "eval_df_indep": eval_df_indep,
        "combined_eval_df": combined_eval_df,
        "combined_loss_df": combined_loss_df,
        "master_results_dir": master_results_dir,
        "zip_path": zip_path,
    }

# --- END NEW FUNCTIONS ---

def main():
    """
    Main function that runs co-trained vs independent comparison experiment.

    Comparison setup:
    - Co-trained: 200 cotrain epochs + 50 refine epochs = 250 LDM training epochs
    - Independent: 50 VAE-only epochs + 250 refine epochs = 250 LDM training epochs

    Both experiments have the same total LDM training epochs for fair comparison.
    X-axis on plots = "LDM Training Epoch" (1-250 for both)
    """

    # === SHARED CONFIG (base settings for both experiments) ===
    cfg_shared = {
        # --- Dataset ---
        "dataset": "CIFAR",
        "batch_size": 256,
        "num_workers": 2,

        # --- Model Architecture ---
        "latent_channels": 8,
        "cond_emb_dim": 32,

        # --- DiT / Transformer settings (LightningDiT-style) ---
        "dit_patch_size": 1,        # patch_size=1 => 8x8 latents -> 64 tokens
        "dit_hidden_dim": 384,
        "dit_depth": 12,
        "dit_num_heads": 6,
        "dit_mlp_ratio": 4.0,
        "dit_dropout": 0.0,

        # --- Optimizer ---
        "adam_beta2": 0.95,

        # --- Flow-matching loss ---
        "cosine_w": 0.0,

        # --- Aux gauge-fix losses for factored DiT head ---
        "aux_head_w": 0.0025,
        "div_w": -0.0008,

        # --- Auxiliary encoder noise channels (0 disables) ---
        "aux_d": 0,
        # --- Encoder architecture (v1 knobs, unchanged) ---
        "base_ch": 64,
        "num_res_blocks": 2,
        "decoder_attn_half": True,
        "latent_proj_depth": 2,
        # --- v2 architectural improvements ---
        "encoder_attn_half": True,       # [4] attention at 16×16 in encoder
        "decoder_extra_block": True,     # [3] +1 ResBlock per decoder stage
        "conv3x3_proj": True,            # [1,2] 3×3 latent proj + decoder input
        "use_tanh_out": False,           # [5] raw output, no tanh saturation
        "clamp_logvar": True,            # [6] clamp logvar to [-30, 20]
        "attn_zero_init": False,         # [7] standard init on VAE attention

        # --- Learning Rates ---
        "lr_vae": 5e-4,
        "lr_ldm": 2e-4,

        # --- KL and perceptual weights ---
        "kl_w": 1e-3,
        "perc_w": 0.85,
        "lpips_mode": "frontier",  # "uniform", "snr" (legacy), "gamma", "frontier", or "prec_mask"

        # --- Frontier-gated perceptual loss settings (used when lpips_mode="frontier" or gan_time_weight="frontier") ---
        "frontier_R_cutoff": 0.05,    # R(t) threshold: LPIPS/GAN gated off where R(t) > cutoff
        "frontier_R_tau": 1.0,        # Sigmoid temperature: smaller = sharper gate

        # --- PatchGAN discriminator ---
        "gan_w": 0.0025,
        "gan_w_tdd_mult": 4.0,
        "disc_time_emb_dim": 128,
        "wiener_alpha_min": 1e-4,
        "wiener_max_var": 1e3,
        "disc_start_epoch": 25,
        "disc_ndf": 64,
        "disc_n_layers": 2,
        "lr_disc": 1e-4,
        "gan_logit_clamp": 10.0,            # Clamp D logits in G loss to prevent divergence
        "gan_time_weight": "frontier",  # "uniform", "frontier", "gamma", "snr", or "snr2"

        # --- Diffusion Settings ---
        "time_schedule": "log_t",     # "flow", "log_t", "log_snr", or "cosine"
        "use_ddim_times": True,
        "t_min": 2.0e-5,
        "t_max": 2.0,
        "num_train_timesteps": 1000,
        "train_on_mu": False,
        "temporal_variance_scale": 0.0,
        "temporal_perturb_type": "log",  # "base" perturbs t, "log" perturbs log(t)

        # --- Cosine VP schedule settings (only used when time_schedule="cosine") ---
        "cosine_t_min": 2e-4,
        "cosine_t_max": 0.9999,
        "cosine_s": 0.008,

        # --- CFG ---
        "cfg_label_dropout": 0.1,
        "cfg_eval_scale": 3.0,
        "cfg_mode": "constant",   # "constant" or "linear_ramp"
        "eval_class_labels": [],
        "class_decoder": True,

        # --- Evaluation & Logging ---
        "use_fixed_eval_banks": True,
        "sw2_n_projections": 1000,
        "ema_decay": 0.9997,

        # --- Misc ---
        "seed": 42,
        "load_from_checkpoint": False,
        "ckpt_dir": "checkpoints",

        # --- Comparison Output ---
        "master_results_dir": "run_results_comparison",
    }


    # === CO-TRAINED CONFIG ===
    # 200 cotrain epochs (VAE + LDM joint) + 50 refine epochs = 250 LDM epochs total
    cfg_cotrain = cfg_shared.copy()
    cfg_cotrain.update({
        # Training schedule
        "epochs_vae": 800,          # Cotrain phase: VAE + LDM joint training
        "epochs_refine": 100,        # Refine phase: LDM-only on frozen VAE
        "lr_refine": 1.5e-5,

        # Score head gaussian factored param
        "factored_head": True,

        # Co-training specific settings
        "freeze_score_in_cotrain": False,  # Normal co-training
        "cotrain_head": "lsi",
        "mse_mode": "raw",               # 'raw', 'score', or 'score_detached'
        "use_latent_norm": True,
        "use_cond_encoder": False,
        "kl_reg_type": "temporal",
        "score_w_vae": 0.6,
        "stiff_w": 1e-6,
        "score_w": 1.0,
        "score_w_decode": 0.0,          # Gradient scale: score head ← MSE recon loss
        "decode_w": 1.0,                   # Gradient scale: decoder   ← MSE recon loss

        # Time-dependent decoder (TDD)
        "time_cond_decoder": True,
        "time_dependent_gan": False,
        "gan_time_weight": "frontier",  # "uniform", "gamma", "snr", or "snr2"
        #"w_decode_time": 0.1,
        "dec_time_emb_dim": 128,
        "decode_time": 1e-4,             # Decode at this t; defaults to t_min if None
        #"snr_downweight": True,

        # Adaptive frontier time sampling
        "adaptive_time": True,              # enable decoder-informed time weighting
        "adaptive_time_nbins": 200,         # log-spaced bins over [t_min, t_max]
        "adaptive_time_ema": 0.9975,          # EMA decay for R(t) tracker
        "adaptive_time_floor": 0.02,        # minimum weight fraction (prevents starvation)
        "adaptive_time_warmup": 500,        # update() calls before activation
        "adaptive_time_min_counts": 5,      # min samples per bin before activation
        "frontier_correct_score": False,     # IW-correct score loss back to log-uniform

        # Eval frequency (eval during both phases)
        "eval_freq_cotrain": 100,    # Eval every 10 epochs during cotrain
        "eval_freq_refine": 100,     # Eval every 10 epochs during refine
    })


    # === INDEPENDENT CONFIG ===
    # 50 VAE-only epochs (no eval) + 250 refine epochs = 250 LDM epochs total
    cfg_indep = cfg_shared.copy()
    cfg_indep.update({
        # Training schedule
        "epochs_vae": 500,           # VAE-only pretraining (no LDM)
        "epochs_refine": 900,       # LDM training on frozen VAE
        "lr_refine": 5e-4,
        "cfg_label_dropout": 0.1,
        "t_min": 3e-4,

        # Score head gaussian factored param
        "factored_head": False,

        # Independent mode settings
        "freeze_score_in_cotrain": True,   # Freeze score nets during VAE training
        "mse_mode": "raw",                    # Must be 'raw' when score is frozen
        "score_w_vae": 0.0,                # No score gradient (redundant but explicit)
        "stiff_w": 0.0,                    # No stiffness penalty
        "use_latent_norm": False,          # Standard VAE (no GroupNorm on mu)
        "use_cond_encoder": False,         # No conditional encoder
        "kl_reg_type": "temporal",           # Standard KL to N(0,I)
        "kl_w": 1e-3,
        "cotrain_head": "lsi",             # Doesn't matter when frozen
        "score_w": 1.0,
        "div_w": 0.0,

        # Time-dependent decoder (TDD) — disabled for independent baseline
        "time_cond_decoder": True,
        "time_dependent_gan": False,
        "gan_time_weight": "uniform",
        "w_decode_time": 0.0,

        # Adaptive frontier time sampling — disabled for independent baseline
        "adaptive_time": False,

        # Eval frequency (no eval during VAE phase, eval during refine)
        "eval_freq_cotrain": 999999,  # Effectively never (VAE phase has no LDM)
        "eval_freq_refine": 100,       # Eval every 10 epochs during refine
    })

    print("=" * 70)
    print("=== CO-TRAINED vs INDEPENDENT TRAINING COMPARISON ===")
    print("=" * 70)
    print("\nExperiment design:")
    print(f"  Co-trained: {cfg_cotrain['epochs_vae']} cotrain + {cfg_cotrain['epochs_refine']} refine = {cfg_cotrain['epochs_vae'] + cfg_cotrain['epochs_refine']} LDM epochs")
    print(f"  Independent: {cfg_indep['epochs_vae']} VAE-only + {cfg_indep['epochs_refine']} refine = {cfg_indep['epochs_refine']} LDM epochs")
    print("\nPlot legend:")
    print("  - Solid blue: Co-trained LSI")
    print("  - Solid red: Co-trained Tweedie")
    print("  - Dashed blue: Independent LSI")
    print("  - Dashed red: Independent Tweedie")
    print("=" * 70)

    results = run_cotrain_vs_indep_comparison(cfg_cotrain, cfg_indep)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Results directory: {results['master_results_dir']}")
    print(f"Zipped results: {results['zip_path']}")
    print("\nOutput structure:")
    print(f"  {results['master_results_dir']}/")
    print(f"  ├── run_results_cotrain/   (co-trained experiment)")
    print(f"  ├── run_results_indep/     (independent experiment)")
    print(f"  ├── comparison_plots/      (4-way comparison plots)")
    print(f"  └── combined_dataframes/   (merged CSV files)")


if __name__ == "__main__":
    main()
