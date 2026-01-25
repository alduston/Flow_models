import math
import os
import random
import numpy as np
from typing import Any, Dict, Tuple

import torch
from torch import nn, optim
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
    alpha = torch.exp(-t)
    sigma = torch.sqrt(1.0 - torch.exp(-2.0 * t) + 1e-8)
    return alpha, sigma

# ---------------------------------------------------------------------------
# Metrics: SW2, MMD, FID, Diversity
# ---------------------------------------------------------------------------

def compute_sw2(
    x: torch.Tensor,
    y: torch.Tensor,
    n_projections: int = 1000,
    theta: torch.Tensor | None = None,
) -> float:
    """Sliced Wasserstein-2 distance in latent space."""
    x = x.detach().float()
    y = y.detach().float()
    device = x.device
    N, D = x.shape

    if theta is None:
        theta = torch.randn(D, n_projections, device=device)
        theta = theta / torch.norm(theta, dim=0, keepdim=True).clamp_min(1e-12)
    else:
        theta = theta.to(device)
        assert theta.dim() == 2 and theta.shape[0] == D, f"theta must have shape [D, K] with D={D}; got {tuple(theta.shape)}"
        assert theta.shape[1] >= n_projections, f"theta has only {theta.shape[1]} projections; need >= {n_projections}"
        theta = theta[:, :n_projections]

    proj_x = x @ theta
    proj_y = y @ theta

    proj_x, _ = torch.sort(proj_x, dim=0)
    proj_y, _ = torch.sort(proj_y, dim=0)

    w2 = torch.mean((proj_x - proj_y) ** 2)
    return w2.item()

def compute_diversity(imgs: torch.Tensor, lpips_fn: Any) -> float:
    """Computes pairwise LPIPS diversity."""
    if lpips_fn is None: return 0.0

    N = imgs.shape[0]
    if N < 2: return 0.0

    perm = torch.randperm(N)
    imgs = imgs[perm]

    half = N // 2
    set1 = imgs[:half]
    set2 = imgs[half:2*half]

    # CIFAR is already 3 channels
    if set1.shape[1] == 1:
        set1 = set1.repeat(1, 3, 1, 1)
        set2 = set2.repeat(1, 3, 1, 1)

    with torch.no_grad():
        dist = lpips_fn(set1, set2)

    return dist.mean().item()

def evaluate_fid(real_loader, generated_imgs, device, batch_size=128):
    if not TORCHMETRICS_AVAILABLE:
        return -1.0

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    def to_01(x):
        x = (x * 0.5) + 0.5
        return x.clamp(0.0, 1.0)

    for batch, _ in real_loader:
        batch = batch.to(device).float()
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)
        fid.update(to_01(batch), real=True)

    n = generated_imgs.shape[0]
    for i in range(0, n, batch_size):
        batch = generated_imgs[i:i+batch_size].to(device).float()
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)
        fid.update(to_01(batch), real=False)

    return fid.compute().item()


def log_latent_stats(name, z):
    with torch.no_grad():
        mean_norm = z.mean(0).norm().item()
        std_mean = z.std(0).mean().item()
        max_val = z.abs().max().item()
        print(f"  [{name}] Latent Stats | Mean Norm: {mean_norm:.4f} | Avg Std: {std_mean:.4f} | Max: {max_val:.4f}")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def extract_inception_features(images, device, batch_size=100, inception_model=None):
    """Extract Inception features for FID/KID computation."""
    if inception_model is None:
        from torchvision.models import inception_v3
        inception_model = inception_v3(pretrained=True, transform_input=False)
        inception_model.fc = torch.nn.Identity()
        inception_model = inception_model.to(device)
        inception_model.eval()

    features_list = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device)

            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)

            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)

            batch = (batch + 1) / 2
            batch = (batch - torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)) / \
                    torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

            feat = inception_model(batch)
            features_list.append(feat.cpu())

    return torch.cat(features_list, 0), inception_model


def compute_fid_from_features(real_features, fake_features):
    """Compute FID from pre-extracted Inception features."""
    real_features = real_features.cpu().numpy()
    fake_features = fake_features.cpu().numpy()

    mu_r = np.mean(real_features, axis=0)
    mu_f = np.mean(fake_features, axis=0)

    sigma_r = np.cov(real_features, rowvar=False)
    sigma_f = np.cov(fake_features, rowvar=False)

    from scipy import linalg

    diff = mu_r - mu_f

    covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r) + np.trace(sigma_f) - 2 * np.trace(covmean)

    return float(fid)


def compute_kid(real_features, fake_features, num_subsets=100, subset_size=1000):
    """Compute KID (Kernel Inception Distance) using polynomial kernel with subsampling."""
    if not isinstance(real_features, torch.Tensor):
        real_features = torch.tensor(real_features)
    if not isinstance(fake_features, torch.Tensor):
        fake_features = torch.tensor(fake_features)

    n_real = real_features.shape[0]
    n_fake = fake_features.shape[0]
    d = real_features.shape[1]

    subset_size = min(subset_size, n_real, n_fake)

    def polynomial_kernel(x, y):
        return ((x @ y.T) / d + 1) ** 3

    kid_values = []

    for _ in range(num_subsets):
        real_idx = torch.randperm(n_real)[:subset_size]
        fake_idx = torch.randperm(n_fake)[:subset_size]

        real_subset = real_features[real_idx]
        fake_subset = fake_features[fake_idx]

        k_rr = polynomial_kernel(real_subset, real_subset)
        k_ff = polynomial_kernel(fake_subset, fake_subset)
        k_rf = polynomial_kernel(real_subset, fake_subset)

        m = subset_size

        diag_rr = torch.diag(k_rr)
        diag_ff = torch.diag(k_ff)

        sum_rr = (k_rr.sum() - diag_rr.sum()) / (m * (m - 1))
        sum_ff = (k_ff.sum() - diag_ff.sum()) / (m * (m - 1))
        sum_rf = k_rf.mean()

        mmd2 = sum_rr + sum_ff - 2 * sum_rf
        kid_values.append(mmd2.item())

    return np.mean(kid_values)


def compute_lsi_gap(score_net, encoder_mus, encoder_logvars, cfg, device,
                    num_samples=5000, num_time_points=50, batch_size=128):
    """Compute LSI gap metric in score parameterization."""
    if score_net is None:
        return 0.0

    score_net.eval()

    n_data = encoder_mus.shape[0]
    num_samples = min(num_samples, n_data)
    sample_indices = torch.randperm(n_data)[:num_samples]

    t_min, t_max = cfg["t_min"], cfg["t_max"]
    time_grid = torch.linspace(t_min, t_max, num_time_points, device=device)

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

            eps_0 = torch.randn_like(batch_mu)
            z0 = batch_mu + batch_std * eps_0

            for t_val in time_grid:
                t = t_val.expand(bsz)
                alpha, sigma = get_ou_params(t.view(bsz, 1, 1, 1))

                noise = torch.randn_like(z0)
                z_t = alpha * z0 + sigma * noise

                mu_t = alpha * batch_mu
                var_t = (alpha ** 2) * batch_var + (sigma ** 2)

                eps_target_lsi = sigma * ((z_t - mu_t) / (var_t + 1e-8))

                eps_pred = score_net(z_t, t)

                sigma_sq = sigma ** 2 + 1e-8
                eps_diff_sq = (eps_pred - eps_target_lsi) ** 2

                score_gap_per_sample = (eps_diff_sq / sigma_sq).sum(dim=(1, 2, 3))

                total_lsi_gap += score_gap_per_sample.sum().item()
                total_count += bsz

    return total_lsi_gap / total_count if total_count > 0 else 0.0


# ===========================================================================
# CIFAR-10 VAE: 3 input channels, 32x32 -> 8x8 latent
# ===========================================================================

class VAE(nn.Module):
    """
    VAE for CIFAR-10 (32x32x3).
    Downsamples 32 -> 16 -> 8, so latent is 8x8 spatial.
    """
    def __init__(self, latent_channels: int = 4, base_ch: int = 32, in_channels: int = 3, use_norm: bool = False):
        super().__init__()
        # Encoder: 32x32 -> 16x16 -> 8x8
        self.use_norm = use_norm
        self.enc_conv_in = nn.Conv2d(in_channels, base_ch, 3, 1, 1)
        self.enc_blocks = nn.ModuleList([
            nn.Sequential(VAEResBlock(base_ch, base_ch), nn.Conv2d(base_ch, base_ch*2, 3, 2, 1)),       # 32->16
            nn.Sequential(VAEResBlock(base_ch*2, base_ch*2), nn.Conv2d(base_ch*2, base_ch*4, 3, 2, 1)), # 16->8
            nn.Sequential(VAEResBlock(base_ch*4, base_ch*4), AttentionBlock(base_ch*4), VAEResBlock(base_ch*4, base_ch*4))
        ])
        self.mu = nn.Conv2d(base_ch*4, latent_channels, 1)
        self.logvar = nn.Conv2d(base_ch*4, latent_channels, 1)
        
        # [NEW] Conditional Group Norm (num_groups=1 is Spatial LayerNorm)
        if self.use_norm:
            # affine=False is critical to enforce the unit constraints hard
            self.gn_mu = nn.GroupNorm(num_groups=1, num_channels=latent_channels, affine=False)


        # Decoder: 8x8 -> 16x16 -> 32x32
        self.dec_conv_in = nn.Conv2d(latent_channels, base_ch*4, 1)
        self.dec_blocks = nn.ModuleList([
            nn.Sequential(VAEResBlock(base_ch*4, base_ch*4), AttentionBlock(base_ch*4), VAEResBlock(base_ch*4, base_ch*4)),
            nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(base_ch*4, base_ch*2, 3, 1, 1), VAEResBlock(base_ch*2, base_ch*2)), # 8->16
            nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(base_ch*2, base_ch, 3, 1, 1), VAEResBlock(base_ch, base_ch))        # 16->32
        ])
        self.dec_out = nn.Sequential(
            nn.GroupNorm(16, base_ch), nn.SiLU(), nn.Conv2d(base_ch, in_channels, 3, 1, 1)
        )

    
    def encode(self, x):
        h = self.enc_conv_in(x)
        for block in self.enc_blocks: h = block(h)
        mu = self.mu(h)
        logvar = self.logvar(h)
        # [NEW] Conditional Apply
        if self.use_norm:
            mu = self.gn_mu(mu)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_conv_in(z)
        for block in self.dec_blocks: h = block(h)
        return torch.tanh(self.dec_out(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAEResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            make_group_norm(in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            make_group_norm(out_ch), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x): return self.net(x) + self.skip(x)

class AttentionBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = make_group_norm(ch)
        self.qkv = nn.Conv2d(ch, ch*3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, -1).chunk(3, 1)
        attn = (q.transpose(-2, -1) @ k) * (C ** -0.5)
        attn = attn.softmax(dim=-1)
        h = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)
        return x + self.proj(h)

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, 4*dim), nn.SiLU(), nn.Linear(4*dim, 4*dim))
    def forward(self, t):
        half = self.mlp[0].in_features // 2
        freqs = torch.exp(torch.linspace(0, math.log(10000), half, device=t.device))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)
        

class UNetModel(nn.Module):
    def __init__(self, in_channels=4, base_channels=32, channel_mults=(1, 2, 6),
                 num_res_blocks=3, attn_levels=(1,)):   # <--- NEW
        super().__init__()
        self.time_embed = TimeEmbedding(base_channels)
        self.head = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.downs = nn.ModuleList()
        ch = base_channels
        chs = [ch]
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResBlock(ch, out_ch, base_channels*4))
                ch = out_ch
                chs.append(ch)
            # <--- NEW: attention at selected resolutions
            # Note: AttentionBlock does NOT add to chs - it doesn't create a skip connection
            if i in attn_levels:
                self.downs.append(AttentionBlock(ch))

            if i != len(channel_mults)-1:
                self.downs.append(nn.Conv2d(ch, ch, 3, 2, 1))
                chs.append(ch)
        self.mid = nn.ModuleList([
            ResBlock(ch, ch, base_channels*4),
            AttentionBlock(ch),
            ResBlock(ch, ch, base_channels*4)
        ])
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                skip = chs.pop()
                self.ups.append(ResBlock(ch + skip, out_ch, base_channels*4))
                ch = out_ch
            # <--- NEW: mirror attention on the way up (optional but usually helps)
            if i in attn_levels:
                self.ups.append(AttentionBlock(ch))
            if i != 0:
                self.ups.append(nn.Sequential(nn.Upsample(scale_factor=2),
                                              nn.Conv2d(ch, ch, 3, 1, 1)))
        self.out = nn.Sequential(make_group_norm(ch), nn.SiLU(),
                                 nn.Conv2d(ch, in_channels, 3, 1, 1))

    def forward(self, x, t):
        emb = self.time_embed(t)
        h = self.head(x)
        hs = [h]
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                h = layer(h, emb)
                hs.append(h)
            elif isinstance(layer, AttentionBlock):
                # Attention blocks don't create skip connections
                h = layer(h)
            else:
                # Downsampling conv
                h = layer(h)
                hs.append(h)
        for layer in self.mid:
            if isinstance(layer, ResBlock): h = layer(h, emb)
            else: h = layer(h)
        for layer in self.ups:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = layer(h, emb)
            else: h = layer(h)
        return self.out(h)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.block1 = nn.Sequential(make_group_norm(in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.time_proj = nn.Linear(t_dim, out_ch)
        self.block2 = nn.Sequential(make_group_norm(out_ch), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, t_emb):
        h = self.block1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        return self.block2(h) + self.skip(x)

class OldUNetModel(nn.Module):
    def __init__(self, in_channels=4, base_channels=64, channel_mults=(1, 2), num_res_blocks=2):
        super().__init__()
        self.time_embed = TimeEmbedding(base_channels)
        self.head = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.downs = nn.ModuleList()
        
        ch = base_channels
        chs = [ch]
        # Downsampling
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResBlock(ch, out_ch, base_channels * 4))
                ch = out_ch
                chs.append(ch)
            if i != len(channel_mults) - 1:
                self.downs.append(nn.Conv2d(ch, ch, 3, 2, 1))
                chs.append(ch)        
        # Middle
        self.mid = nn.ModuleList([
            ResBlock(ch, ch, base_channels * 4),
            AttentionBlock(ch),
            ResBlock(ch, ch, base_channels * 4)
        ])
        # Upsampling
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                skip = chs.pop()
                self.ups.append(ResBlock(ch + skip, out_ch, base_channels * 4))
                ch = out_ch
            if i != 0:
                self.ups.append(nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(ch, ch, 3, 1, 1)
                ))
        self.out = nn.Sequential(
            make_group_norm(ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_channels, 3, 1, 1)
        )

    def forward(self, x, t):
        emb = self.time_embed(t)
        h = self.head(x)
        hs = [h]
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                h = layer(h, emb)
            else:
                h = layer(h)
            hs.append(h)     
        for layer in self.mid:
            if isinstance(layer, ResBlock):
                h = layer(h, emb)
            else:
                h = layer(h)        
        for layer in self.ups:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = layer(h, emb)
            else:
                h = layer(h)
        return self.out(h)


class OldResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.block1 = nn.Sequential(
            make_group_norm(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        )
        self.time_proj = nn.Linear(t_dim, out_ch)
        self.block2 = nn.Sequential(
            make_group_norm(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        return self.block2(h) + self.skip(x)

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

class UniversalSampler:
    def __init__(self, method="heun_sde", num_steps=20, t_min=2e-5, t_max=2.0):
        self.num_steps = num_steps
        self.t_min = t_min
        self.t_max = t_max
        self.method = method

    def get_ode_derivative(self, x, t, unet):
        B = x.shape[0]
        t_vec = t.expand(B)
        eps_pred = unet(x, t_vec)
        _, sigma = get_ou_params(t_vec.view(B, 1, 1, 1))
        inv_sigma = 1.0 / (sigma + 1e-10)
        return -x + inv_sigma * eps_pred

    def get_rev_sde_drift(self, x, t, unet):
        B = x.shape[0]
        t_vec = t.expand(B)
        eps_pred = unet(x, t_vec)
        _, sigma = get_ou_params(t_vec.view(B, 1, 1, 1))
        inv_sigma = 1.0 / (sigma + 1e-8)
        return -x + 2.0 * inv_sigma * eps_pred

    def step_euler_ode(self, x, t_curr, t_next, unet):
        dt = t_next - t_curr
        d_curr = self.get_ode_derivative(x, t_curr, unet)
        return x + dt * d_curr

    def step_rk4_ode(self, x, t_curr, t_next, unet):
        dt = t_next - t_curr
        half_dt = dt * 0.5
        t_half = t_curr + half_dt

        k1 = self.get_ode_derivative(x, t_curr, unet)
        k2 = self.get_ode_derivative(x + half_dt * k1, t_half, unet)
        k3 = self.get_ode_derivative(x + half_dt * k2, t_half, unet)
        k4 = self.get_ode_derivative(x + dt * k3, t_next, unet)

        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def step_heun_ode(self, x, t_curr, t_next, unet):
        B = x.shape[0]
        dt = t_next - t_curr

        t_vec = t_curr.expand(B)
        eps_pred = unet(x, t_vec)
        _, sigma = get_ou_params(t_vec.view(B, 1, 1, 1))
        d_curr = -x + (1.0 / (sigma + 1e-10)) * eps_pred
        x_proposed = x + dt * d_curr

        if t_next > self.t_min:
            t_next_vec = t_next.expand(B)
            eps_next = unet(x_proposed, t_next_vec)
            _, sigma_next = get_ou_params(t_next_vec.view(B, 1, 1, 1))
            d_next = -x_proposed + (1.0 / (sigma_next + 1e-10)) * eps_next
            x = x + 0.5 * dt * (d_curr + d_next)
        else:
            x = x_proposed

        return x

        # ---- New: stochastic Heun predictor-corrector for reverse-time SDE ----
    def step_heun_sde(self, x, t_curr, t_next, unet, generator=None):
        B = x.shape[0]
        dt = t_next - t_curr                         # negative
        dt_abs = torch.abs(dt).clamp_min(1e-12)       # scalar tensor
        # dW ~ N(0, dt_abs I), and G = sqrt(2)
        if generator is None:
            noise = torch.randn_like(x)
        else:
            noise = torch.randn(x.shape, device=x.device, generator=generator)
        dW = torch.sqrt(2.0 * dt_abs) * noise

        b_curr = self.get_rev_sde_drift(x, t_curr, unet)
        x_hat = x + dt * b_curr + dW

        b_next = self.get_rev_sde_drift(x_hat, t_next, unet)
        x_new = x + 0.5 * dt * (b_curr + b_next) + dW
        return x_new

    def sample(self, unet, shape=None, device=None, x_init=None, generator=None):
        unet.eval()

        if x_init is None:
            assert shape is not None and device is not None
            x = torch.randn(shape, device=device, generator=generator)
            device = x.device
        else:
            x = x_init
            device = x.device

        ts = torch.logspace(
            math.log10(self.t_max),
            math.log10(self.t_min),
            self.num_steps + 1,
            device=device
        )

        for i in range(self.num_steps):
            t_curr = ts[i]
            t_next = ts[i + 1]

            if self.method == "rk4_ode":
                x = self.step_rk4_ode(x, t_curr, t_next, unet)
            elif self.method == "euler_ode":
                x = self.step_euler_ode(x, t_curr, t_next, unet)
            elif self.method == "heun_ode":
                x = self.step_heun_ode(x, t_curr, t_next, unet)
            elif self.method == "heun_sde":
                x = self.step_heun_sde(x, t_curr, t_next, unet, generator=generator)
            else:
                raise ValueError(f"Unknown sampling method: {self.method}")
        return x


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
):
    """Full test-set evaluation."""
    print(f"\n--- Evaluation: {prefix} @ Ep {epoch_idx} ---")
    vae.eval()
    if unet is not None:
        unet.eval()

    target_count = len(loader.dataset)
    bs = cfg["batch_size"]
    latent_shape = (cfg["latent_channels"], 8, 8)
    sw2_nproj = int(cfg.get("sw2_n_projections", 1000))

    if fixed_noise_bank is not None:
        assert fixed_noise_bank.shape[0] >= target_count
        assert tuple(fixed_noise_bank.shape[1:]) == latent_shape
    if fixed_posterior_eps_bank_A is not None:
        assert fixed_posterior_eps_bank_A.shape[0] >= target_count
        assert tuple(fixed_posterior_eps_bank_A.shape[1:]) == latent_shape
    if fixed_posterior_eps_bank_B is not None:
        assert fixed_posterior_eps_bank_B.shape[0] >= target_count
        assert tuple(fixed_posterior_eps_bank_B.shape[1:]) == latent_shape

    real_latents_A, real_latents_B, real_imgs = [], [], []
    encoder_mus, encoder_logvars = [], []
    bank_idx = 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            mu, logvar = vae.encode(x)
            std = torch.exp(0.5 * logvar)
            bsz = x.shape[0]

            encoder_mus.append(mu.cpu())
            encoder_logvars.append(logvar.cpu())

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
    real_flat_A = real_latents_A.view(target_count, -1).to(device)

    if fixed_posterior_eps_bank_B is not None:
        real_latents_B = torch.cat(real_latents_B, 0)[:target_count]
        real_flat_B = real_latents_B.view(target_count, -1).to(device)
    else:
        real_flat_B = None

    print("  Extracting Inception features...")
    real_features, inception_model = extract_inception_features(
        real_imgs, device, batch_size=cfg.get("fid_batch_size", bs)
    )
    real_features = real_features.to(device)

    lsi_gap_unet = compute_lsi_gap(
        unet, encoder_mus, encoder_logvars, cfg, device,
        num_samples=min(5000, target_count), num_time_points=50, batch_size=bs
    )

    configs = [("VAE_Rec_eps", 0, "Recon (posterior z)")]
    if unet is not None:
        configs.extend([
            ("heun_ode", 20, "Baseline (Heun)"),
            ("rk4_ode",  10, "Smoothness (RK4)"),
        ])

    results = []

    for method, steps, desc in configs:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            if method == "VAE_Rec_eps":
                fake_imgs = torch.cat([
                    vae.decode(real_latents_A[i:i + bs].to(device)).cpu()
                    for i in range(0, len(real_latents_A), bs)
                ], 0)

                if real_flat_B is not None:
                    w2 = compute_sw2(real_flat_A, real_flat_B, n_projections=sw2_nproj, theta=fixed_sw2_theta)
                else:
                    perm = torch.randperm(real_flat_A.size(0), device=device)
                    half = real_flat_A.size(0) // 2
                    w2 = compute_sw2(real_flat_A[perm[:half]], real_flat_A[perm[half:2*half]],
                                     n_projections=sw2_nproj, theta=fixed_sw2_theta)
                lsi_gap = 0.0

            else:
                sampler = UniversalSampler(method=method, num_steps=steps,
                                           t_min=cfg["t_min"], t_max=cfg["t_max"])
                fake_latents_list, fake_imgs_list = [], []

                for i in range(0, target_count, bs):
                    batch_sz = min(bs, target_count - i)
                    if fixed_noise_bank is not None:
                        xT = fixed_noise_bank[i:i + batch_sz].to(device)
                        z_gen = sampler.sample(unet, x_init=xT)
                    else:
                        z_gen = sampler.sample(unet, shape=(batch_sz, *latent_shape), device=device)
                    fake_latents_list.append(z_gen.cpu())
                    fake_imgs_list.append(vae.decode(z_gen).cpu())

                fake_latents = torch.cat(fake_latents_list, 0)
                fake_imgs = torch.cat(fake_imgs_list, 0)
                fake_flat = fake_latents.view(fake_latents.shape[0], -1).to(device)
                w2 = compute_sw2(real_flat_A, fake_flat, n_projections=sw2_nproj, theta=fixed_sw2_theta)
                lsi_gap = lsi_gap_unet

        fake_features, inception_model = extract_inception_features(
            fake_imgs, device, batch_size=cfg.get("fid_batch_size", bs),
            inception_model=inception_model
        )
        fake_features = fake_features.to(device)

        fid = compute_fid_from_features(real_features, fake_features)
        kid = compute_kid(real_features, fake_features, num_subsets=100, subset_size=1000)
        div = compute_diversity(fake_imgs.to(device), lpips_fn) if LPIPS_AVAILABLE else 0.0

        results.append({
            "config": f"{method}@{steps}",
            "desc": desc,
            "fid": fid,
            "kid": kid,
            "w2": w2,
            "div": div,
            "lsi_gap": lsi_gap,
        })

        if method in ("VAE_Rec_eps",) or "rk4" in method or "heun" in method:
            if results_dir is not None:
                samples_dir = os.path.join(results_dir, "samples")
                os.makedirs(samples_dir, exist_ok=True)
                grid = tv_utils.make_grid(fake_imgs[:64], nrow=8, normalize=True, value_range=(-1, 1))
                tv_utils.save_image(grid, os.path.join(samples_dir, f"{prefix}_ep{epoch_idx}_{method}_{steps}.png"))

    print(f"\n  {'Config':<20} {'FID':>8} {'KID':>10} {'SW2':>12} {'Div':>8} {'LSI Gap':>10}")
    print("  " + "-"*70)
    for r in results:
        print(f"  {r['config']:<20} {r['fid']:>8.2f} {r['kid']:>10.6f} {r['w2']:>12.6f} {r['div']:>8.4f} {r['lsi_gap']:>10.4f}")

    result_dict = {}
    for r in results:
        cfg_name = r['config'].replace('@', '_')
        result_dict[f"fid_{cfg_name}"] = r['fid']
        result_dict[f"kid_{cfg_name}"] = r['kid']
        result_dict[f"sw2_{cfg_name}"] = r['w2']
        result_dict[f"div_{cfg_name}"] = r['div']
        result_dict[f"lsi_gap_{cfg_name}"] = r['lsi_gap']

    return result_dict


# ---------------------------------------------------------------------------
# Visualization & Logging
# ---------------------------------------------------------------------------

def setup_run_results_dir(results_dir: str = "run_results") -> str:
    """Wipe and recreate the results directory."""
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)
    os.makedirs(os.path.join(results_dir, "checkpoints"))
    os.makedirs(os.path.join(results_dir, "samples"))
    os.makedirs(os.path.join(results_dir, "plots"))
    os.makedirs(os.path.join(results_dir, "dataframes"))
    print(f"--> Created fresh results directory: {results_dir}")
    return results_dir


def plot_vae_recon_loss(loss_df, save_path):
    """Plot VAE reconstruction loss."""
    ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cotrain_df = loss_df[loss_df["stage"] == "cotrain"]
    if len(cotrain_df) > 0 and "recon" in cotrain_df.columns:
        ax.plot(cotrain_df["epoch"], cotrain_df["recon"], 
                color="blue", linewidth=2, marker='o', markersize=3, label="Reconstruction Loss")
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("VAE Reconstruction Loss (Co-training Stage)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def plot_score_losses(loss_df, save_path):
    """Plot score losses LSI vs Tweedie (log scale)."""
    ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if "score_lsi" in loss_df.columns:
        ax.semilogy(loss_df["epoch"], loss_df["score_lsi"], 
                    color="blue", linewidth=2, marker='o', markersize=3, label="LSI Score Loss")
    if "score_control" in loss_df.columns:
        ax.semilogy(loss_df["epoch"], loss_df["score_control"], 
                    color="red", linewidth=2, marker='s', markersize=3, label="Tweedie Score Loss")
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Score Loss (log scale)", fontsize=12)
    ax.set_title("Score Matching Losses: LSI vs Tweedie", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def plot_fid_rk4(eval_df, save_path):
    """Plot FID comparison (RK4 10 steps)."""
    ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lsi_df = eval_df[eval_df["tag"].str.contains("LSI", case=False)].copy()
    ctrl_df = eval_df[eval_df["tag"].str.contains("Ctrl", case=False)].copy()
    
    if "fid_rk4_ode_10" in lsi_df.columns and not lsi_df["fid_rk4_ode_10"].isna().all():
        ax.plot(lsi_df["epoch"], lsi_df["fid_rk4_ode_10"], 
                color="blue", linewidth=2, marker='o', markersize=4, label="LSI")
    
    if "fid_rk4_ode_10" in ctrl_df.columns and not ctrl_df["fid_rk4_ode_10"].isna().all():
        ax.plot(ctrl_df["epoch"], ctrl_df["fid_rk4_ode_10"], 
                color="red", linewidth=2, marker='s', markersize=4, label="Tweedie")
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("FID (lower = better)", fontsize=12)
    ax.set_title("FID Comparison (RK4 ODE, 10 Steps)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def plot_fid_heun(eval_df, save_path):
    """Plot FID comparison (Heun PC 20 steps)."""
    ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lsi_df = eval_df[eval_df["tag"].str.contains("LSI", case=False)].copy()
    ctrl_df = eval_df[eval_df["tag"].str.contains("Ctrl", case=False)].copy()
    
    if "fid_heun_sde_20" in lsi_df.columns and not lsi_df["fid_heun_sde_20"].isna().all():
        ax.plot(lsi_df["epoch"], lsi_df["fid_heun_sde_20"], 
                color="blue", linewidth=2, marker='o', markersize=4, label="LSI")
    
    if "fid_heun_sde_20" in ctrl_df.columns and not ctrl_df["fid_heun_sde_20"].isna().all():
        ax.plot(ctrl_df["epoch"], ctrl_df["fid_heun_sde_20"], 
                color="red", linewidth=2, marker='s', markersize=4, label="Tweedie")
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("FID (lower = better)", fontsize=12)
    ax.set_title("FID Comparison (Heun PC, 20 Steps)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def plot_kid_rk4(eval_df, save_path):
    """Plot KID comparison (RK4 10 steps)."""
    ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lsi_df = eval_df[eval_df["tag"].str.contains("LSI", case=False)].copy()
    ctrl_df = eval_df[eval_df["tag"].str.contains("Ctrl", case=False)].copy()
    
    if "kid_rk4_ode_10" in lsi_df.columns and not lsi_df["kid_rk4_ode_10"].isna().all():
        ax.plot(lsi_df["epoch"], lsi_df["kid_rk4_ode_10"], 
                color="blue", linewidth=2, marker='o', markersize=4, label="LSI")
    
    if "kid_rk4_ode_10" in ctrl_df.columns and not ctrl_df["kid_rk4_ode_10"].isna().all():
        ax.plot(ctrl_df["epoch"], ctrl_df["kid_rk4_ode_10"], 
                color="red", linewidth=2, marker='s', markersize=4, label="Tweedie")
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("KID (lower = better)", fontsize=12)
    ax.set_title("KID Comparison (RK4 ODE, 10 Steps)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def plot_kid_heun(eval_df, save_path):
    """Plot KID comparison (Heun PC 20 steps)."""
    ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lsi_df = eval_df[eval_df["tag"].str.contains("LSI", case=False)].copy()
    ctrl_df = eval_df[eval_df["tag"].str.contains("Ctrl", case=False)].copy()
    
    if "kid_heun_sde_20" in lsi_df.columns and not lsi_df["kid_heun_sde_20"].isna().all():
        ax.plot(lsi_df["epoch"], lsi_df["kid_heun_sde_20"], 
                color="blue", linewidth=2, marker='o', markersize=4, label="LSI")
    
    if "kid_heun_sde_20" in ctrl_df.columns and not ctrl_df["kid_heun_sde_20"].isna().all():
        ax.plot(ctrl_df["epoch"], ctrl_df["kid_heun_sde_20"], 
                color="red", linewidth=2, marker='s', markersize=4, label="Tweedie")
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("KID (lower = better)", fontsize=12)
    ax.set_title("KID Comparison (Heun PC, 20 Steps)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def plot_sw2_rk4(eval_df, save_path):
    """Plot SW2 comparison (RK4 10 steps, log scale)."""
    ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lsi_df = eval_df[eval_df["tag"].str.contains("LSI", case=False)].copy()
    ctrl_df = eval_df[eval_df["tag"].str.contains("Ctrl", case=False)].copy()
    
    if "sw2_rk4_ode_10" in lsi_df.columns and not lsi_df["sw2_rk4_ode_10"].isna().all():
        ax.semilogy(lsi_df["epoch"], lsi_df["sw2_rk4_ode_10"], 
                    color="blue", linewidth=2, marker='o', markersize=4, label="LSI")
    
    if "sw2_rk4_ode_10" in ctrl_df.columns and not ctrl_df["sw2_rk4_ode_10"].isna().all():
        ax.semilogy(ctrl_df["epoch"], ctrl_df["sw2_rk4_ode_10"], 
                    color="red", linewidth=2, marker='s', markersize=4, label="Tweedie")
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("SW2 (log scale, lower = better)", fontsize=12)
    ax.set_title("SW2 Latent Distance (RK4 ODE, 10 Steps)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def plot_sw2_heun(eval_df, save_path):
    """Plot SW2 comparison (Heun PC 20 steps, log scale)."""
    ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lsi_df = eval_df[eval_df["tag"].str.contains("LSI", case=False)].copy()
    ctrl_df = eval_df[eval_df["tag"].str.contains("Ctrl", case=False)].copy()
    
    if "sw2_heun_sde_20" in lsi_df.columns and not lsi_df["sw2_heun_sde_20"].isna().all():
        ax.semilogy(lsi_df["epoch"], lsi_df["sw2_heun_sde_20"], 
                    color="blue", linewidth=2, marker='o', markersize=4, label="LSI")
    
    if "sw2_heun_sde_20" in ctrl_df.columns and not ctrl_df["sw2_heun_sde_20"].isna().all():
        ax.semilogy(ctrl_df["epoch"], ctrl_df["sw2_heun_sde_20"], 
                    color="red", linewidth=2, marker='s', markersize=4, label="Tweedie")
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("SW2 (log scale, lower = better)", fontsize=12)
    ax.set_title("SW2 Latent Distance (Heun PC, 20 Steps)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def plot_fid_gap_rk4(eval_df, save_path):
    """Plot FID Gap (RK4 10 steps)."""
    ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lsi_df = eval_df[eval_df["tag"].str.contains("LSI", case=False)].copy()
    ctrl_df = eval_df[eval_df["tag"].str.contains("Ctrl", case=False)].copy()
    
    if len(lsi_df) > 0 and len(ctrl_df) > 0:
        merged = pd.merge(
            lsi_df[["epoch", "fid_rk4_ode_10", "kid_rk4_ode_10", "sw2_rk4_ode_10"]],
            ctrl_df[["epoch", "fid_rk4_ode_10", "kid_rk4_ode_10", "sw2_rk4_ode_10"]],
            on="epoch",
            suffixes=("_lsi", "_ctrl")
        )
        
        if "fid_rk4_ode_10_lsi" in merged.columns and "fid_rk4_ode_10_ctrl" in merged.columns:
            merged["fid_gap"] = merged["fid_rk4_ode_10_ctrl"] - merged["fid_rk4_ode_10_lsi"]
            ax.plot(merged["epoch"], merged["fid_gap"], 
                    color="purple", linewidth=2, marker='o', markersize=4, 
                    label="FID Gap (Tweedie - LSI)")
        
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.fill_between(merged["epoch"], 0, merged["fid_gap"], 
                        where=merged["fid_gap"] > 0, alpha=0.3, color="green", 
                        label="LSI Better")
        ax.fill_between(merged["epoch"], 0, merged["fid_gap"], 
                        where=merged["fid_gap"] < 0, alpha=0.3, color="red", 
                        label="Tweedie Better")
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("FID Gap (Tweedie - LSI)", fontsize=12)
    ax.set_title("LSI Advantage: FID Gap (RK4 ODE 10 Steps)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def plot_fid_gap_heun(eval_df, save_path):
    """Plot FID Gap (Heun PC 20 steps)."""
    ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lsi_df = eval_df[eval_df["tag"].str.contains("LSI", case=False)].copy()
    ctrl_df = eval_df[eval_df["tag"].str.contains("Ctrl", case=False)].copy()
    
    if len(lsi_df) > 0 and len(ctrl_df) > 0:
        merged = pd.merge(
            lsi_df[["epoch", "fid_heun_sde_20", "kid_heun_sde_20", "sw2_heun_sde_20"]],
            ctrl_df[["epoch", "fid_heun_sde_20", "kid_heun_sde_20", "sw2_heun_sde_20"]],
            on="epoch",
            suffixes=("_lsi", "_ctrl")
        )
        
        if "fid_heun_sde_20_lsi" in merged.columns and "fid_heun_sde_20_ctrl" in merged.columns:
            merged["fid_gap"] = merged["fid_heun_sde_20_ctrl"] - merged["fid_heun_sde_20_lsi"]
            ax.plot(merged["epoch"], merged["fid_gap"], 
                    color="purple", linewidth=2, marker='o', markersize=4, 
                    label="FID Gap (Tweedie - LSI)")
        
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.fill_between(merged["epoch"], 0, merged["fid_gap"], 
                        where=merged["fid_gap"] > 0, alpha=0.3, color="green", 
                        label="LSI Better")
        ax.fill_between(merged["epoch"], 0, merged["fid_gap"], 
                        where=merged["fid_gap"] < 0, alpha=0.3, color="red", 
                        label="Tweedie Better")
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("FID Gap (Tweedie - LSI)", fontsize=12)
    ax.set_title("LSI Advantage: FID Gap (Heun PC 20 Steps)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def plot_lsi_gap_metric_rk4(eval_df, save_path):
    """Plot LSI Gap Metric (RK4 10 steps)."""
    ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lsi_df = eval_df[eval_df["tag"].str.contains("LSI", case=False)].copy()
    ctrl_df = eval_df[eval_df["tag"].str.contains("Ctrl", case=False)].copy()
    
    if "lsi_gap_rk4_ode_10" in lsi_df.columns and not lsi_df["lsi_gap_rk4_ode_10"].isna().all():
        ax.plot(lsi_df["epoch"], lsi_df["lsi_gap_rk4_ode_10"], 
                color="blue", linewidth=2, marker='o', markersize=4, 
                label="LSI Net - LSI Gap Metric")
    
    if "lsi_gap_rk4_ode_10" in ctrl_df.columns and not ctrl_df["lsi_gap_rk4_ode_10"].isna().all():
        ax.plot(ctrl_df["epoch"], ctrl_df["lsi_gap_rk4_ode_10"], 
                color="red", linewidth=2, marker='s', markersize=4, 
                label="Tweedie Net - LSI Gap Metric")
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("LSI Gap Metric (lower = better)", fontsize=12)
    ax.set_title("LSI Gap Metric: Score Network Alignment (RK4 10 Steps)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def plot_lsi_gap_metric_heun(eval_df, save_path):
    """Plot LSI Gap Metric (Heun PC 20 steps)."""
    ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lsi_df = eval_df[eval_df["tag"].str.contains("LSI", case=False)].copy()
    ctrl_df = eval_df[eval_df["tag"].str.contains("Ctrl", case=False)].copy()
    
    if "lsi_gap_heun_sde_20" in lsi_df.columns and not lsi_df["lsi_gap_heun_sde_20"].isna().all():
        ax.plot(lsi_df["epoch"], lsi_df["lsi_gap_heun_sde_20"], 
                color="blue", linewidth=2, marker='o', markersize=4, 
                label="LSI Net - LSI Gap Metric")
    
    if "lsi_gap_heun_sde_20" in ctrl_df.columns and not ctrl_df["lsi_gap_heun_sde_20"].isna().all():
        ax.plot(ctrl_df["epoch"], ctrl_df["lsi_gap_heun_sde_20"], 
                color="red", linewidth=2, marker='s', markersize=4, 
                label="Tweedie Net - LSI Gap Metric")
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("LSI Gap Metric (lower = better)", fontsize=12)
    ax.set_title("LSI Gap Metric: Score Network Alignment (Heun PC 20 Steps)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def generate_all_visualizations(loss_df, eval_df, results_dir):
    """Generate all 12 visualization plots."""
    plots_dir = os.path.join(results_dir, "plots")
    print("\n--> Generating visualization suite (12 plots)...")
    
    plot_vae_recon_loss(loss_df, os.path.join(plots_dir, "01_vae_recon_loss.png"))
    plot_score_losses(loss_df, os.path.join(plots_dir, "02_score_losses.png"))
    
    plot_fid_rk4(eval_df, os.path.join(plots_dir, "03_fid_rk4_10.png"))
    plot_fid_heun(eval_df, os.path.join(plots_dir, "04_fid_heun_20.png"))
    plot_kid_rk4(eval_df, os.path.join(plots_dir, "05_kid_rk4_10.png"))
    plot_kid_heun(eval_df, os.path.join(plots_dir, "06_kid_heun_20.png"))
    plot_sw2_rk4(eval_df, os.path.join(plots_dir, "07_sw2_rk4_10.png"))
    plot_sw2_heun(eval_df, os.path.join(plots_dir, "08_sw2_heun_20.png"))
    
    plot_fid_gap_rk4(eval_df, os.path.join(plots_dir, "09_fid_gap_rk4_10.png"))
    plot_fid_gap_heun(eval_df, os.path.join(plots_dir, "10_fid_gap_heun_20.png"))
    
    plot_lsi_gap_metric_rk4(eval_df, os.path.join(plots_dir, "11_lsi_gap_metric_rk4_10.png"))
    plot_lsi_gap_metric_heun(eval_df, os.path.join(plots_dir, "12_lsi_gap_metric_heun_20.png"))
    
    print("--> Visualization suite complete (12 plots generated)!")


def save_dataframes(loss_df, eval_df, results_dir):
    """Save the dataframes to CSV files."""
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

def make_dataloaders(batch_size, num_workers):
    """CIFAR-10 dataloaders. Images are 32x32x3, normalized to [-1,1]."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=tf)
    test = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=tf)
    tl = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    vl = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return tl, vl


def train_vae_cotrained(cfg):
    """Co-training function with DataFrame logging and visualization."""
    results_dir = setup_run_results_dir(cfg.get("results_dir", "run_results"))
    
    cfg["ckpt_dir"] = os.path.join(results_dir, "checkpoints")
    
    device = default_device()
    train_l, test_l = make_dataloaders(cfg["batch_size"], cfg["num_workers"])

    # VAE for CIFAR-10: 3 input channels
    # vae = VAE(latent_channels=cfg["latent_channels"], in_channels=3).to(device)
    vae = VAE(latent_channels=cfg["latent_channels"], use_norm=cfg.get("use_latent_norm", False)).to(device)
    eval_freq = cfg.get("eval_freq", 10)

    unet_lsi = UNetModel(in_channels=cfg["latent_channels"]).to(device)
    unet_control = UNetModel(in_channels=cfg["latent_channels"]).to(device)

    if cfg.get("load_from_checkpoint", False):
        ckpt_load_dir = cfg.get("ckpt_load_dir", cfg["ckpt_dir"])
        print(f"--> Loading checkpoints from {ckpt_load_dir}...")
        try:
            vae.load_state_dict(torch.load(os.path.join(ckpt_load_dir, "vae_cotrained.pt"), map_location=device))
            print("    Loaded VAE.")
        except Exception as e:
            print(f"    Warning: Could not load VAE ({e})")

        try:
            unet_lsi.load_state_dict(torch.load(os.path.join(ckpt_load_dir, "unet_lsi.pt"), map_location=device))
            print("    Loaded UNet LSI.")
        except Exception as e:
            print(f"    Warning: Could not load UNet LSI ({e})")

        try:
            unet_control.load_state_dict(torch.load(os.path.join(ckpt_load_dir, "unet_control.pt"), map_location=device))
            print("    Loaded UNet Control.")
        except Exception as e:
            print(f"    Warning: Could not load UNet Control ({e})")

    unet_lsi_ema = UNetModel(in_channels=cfg["latent_channels"]).to(device)
    unet_lsi_ema.load_state_dict(unet_lsi.state_dict())
    unet_lsi_ema.eval()
    for p in unet_lsi_ema.parameters(): p.requires_grad = False

    unet_control_ema = UNetModel(in_channels=cfg["latent_channels"]).to(device)
    unet_control_ema.load_state_dict(unet_control.state_dict())
    unet_control_ema.eval()
    for p in unet_control_ema.parameters(): p.requires_grad = False

    ema_decay = 0.999

    score_w_vae = cfg.get("score_w_vae", cfg["score_w"])
    opt_joint = optim.AdamW([
        {'params': vae.parameters(), 'lr': cfg["lr_vae"]},
        {'params': unet_lsi.parameters(), 'lr': cfg["lr_ldm"]/score_w_vae}
    ], weight_decay=1e-4)

    opt_control = optim.AdamW(unet_control.parameters(), lr=cfg["lr_ldm"], weight_decay=1e-4)

    lpips_fn = lpips.LPIPS(net='vgg').to(device) if LPIPS_AVAILABLE else None

    if cfg.get("use_fixed_eval_banks", True):
        N_test = len(test_l.dataset)
        latent_shape = (cfg["latent_channels"], 8, 8)
        seed = int(cfg.get("seed", 0))

        g_noise = torch.Generator(device="cpu").manual_seed(seed + 12345)
        fixed_noise_bank = torch.randn((N_test, *latent_shape), generator=g_noise)

        g_postA = torch.Generator(device="cpu").manual_seed(seed + 54321)
        g_postB = torch.Generator(device="cpu").manual_seed(seed + 98765)
        fixed_posterior_eps_bank_A = torch.randn((N_test, *latent_shape), generator=g_postA)
        fixed_posterior_eps_bank_B = torch.randn((N_test, *latent_shape), generator=g_postB)

        D = cfg["latent_channels"] * 8 * 8
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

    loss_records = []
    eval_records = []

    print("--> Starting Dual Co-training...")
    for ep in range(cfg["epochs_vae"]):
        vae.train(); unet_lsi.train(); unet_control.train()
        metrics = {k: 0.0 for k in ["loss", "recon", "kl", "score_lsi", "score_control", "perc", "stiff"]}
        mu_stats = []

        for x, _ in tqdm(train_l, desc=f"Ep {ep+1}", leave=False):
            x = x.to(device)
            B = x.shape[0]

            x_rec, mu, logvar = vae(x)
            if len(mu_stats) < 5: mu_stats.append(mu.detach())

            recon = F.mse_loss(x_rec, x)

            # CIFAR is already 3 channels, no repeat needed
            if LPIPS_AVAILABLE:
                perc = lpips_fn(x_rec, x).mean()
            else:
                perc = torch.tensor(0.0, device=device)

            # [NEW] Flexible KL Regularization Selector
            reg_type = cfg.get("kl_reg_type", "mod") # 'normal', 'mod', or 'norm'

            if reg_type == "normal":
                # Standard VAE KL: N(0,1) target
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            elif reg_type == "mod":
                # Modified KL: Energy/Trace control (No per-sample isotropy)
                kl = -0.5 * torch.mean(1 - mu.pow(2) - logvar.exp())    
            elif reg_type == "norm":
                # Variance Anchor: Only penalize logvar deviation (for use with GroupNorm)
                # Target logvar=0 (std=1). 0.1 is the spring constant.
                kl = 0.1 * torch.mean(logvar.pow(2))
            else:
                raise ValueError(f"Unknown kl_reg_type: {reg_type}")

            t = sample_log_uniform_times(B, cfg["t_min"], cfg["t_max"], device)
            z0 = vae.reparameterize(mu, logvar)
            alpha, sigma = get_ou_params(t.view(B,1,1,1))

            noise = torch.randn_like(z0)
            z_t = alpha * z0 + sigma * noise

            var_0 = torch.exp(logvar)
            mu_t = alpha * mu
            var_t = (alpha**2) * var_0 + (sigma**2)

            eps_target_lsi = sigma * ((z_t - mu_t) / (var_t + 1e-8))
            eps_pred_lsi = unet_lsi(z_t, t)
            score_loss_lsi = F.mse_loss(eps_pred_lsi, eps_target_lsi)

            stiff_w = cfg.get("stiff_w", 0.0)
            if stiff_w > 0.0:
                inv_var_t = 1.0 / (var_t + 1e-8)
                stiff_pen = inv_var_t.flatten(1).mean(dim=1).mean()
            else:
                stiff_pen = torch.tensor(0.0, device=device)
                

            score_w_vae = cfg.get("score_w_vae", cfg["score_w"])
            loss_joint = recon + cfg["perc_w"]*perc + cfg["kl_w"]*kl + score_w_vae*score_loss_lsi + stiff_w*stiff_pen


            opt_joint.zero_grad()
            loss_joint.backward()
            nn.utils.clip_grad_norm_(list(vae.parameters()) + list(unet_lsi.parameters()), 1.0)
            opt_joint.step()

            with torch.no_grad():
                for p_online, p_ema in zip(unet_lsi.parameters(), unet_lsi_ema.parameters()):
                    p_ema.data.mul_(ema_decay).add_(p_online.data, alpha=1 - ema_decay)

            z_t_detached = z_t.detach()
            eps_pred_control = unet_control(z_t_detached, t)
            score_loss_control = cfg["score_w"] * F.mse_loss(eps_pred_control, noise)

            opt_control.zero_grad()
            score_loss_control.backward()
            nn.utils.clip_grad_norm_(unet_control.parameters(), 1.0)
            opt_control.step()

            with torch.no_grad():
                for p_online, p_ema in zip(unet_control.parameters(), unet_control_ema.parameters()):
                    p_ema.data.mul_(ema_decay).add_(p_online.data, alpha=1 - ema_decay)

            metrics["loss"] += loss_joint.item()
            metrics["recon"] += recon.item()
            metrics["kl"] += kl.item()
            metrics["score_lsi"] += score_loss_lsi.item()
            metrics["score_control"] += (score_loss_control.item() / cfg["score_w"])
            metrics["perc"] += perc.item()
            metrics["stiff"] += stiff_pen.item() if isinstance(stiff_pen, torch.Tensor) else stiff_pen

        n_batches = len(train_l)
        epoch_metrics = {
            "epoch": ep + 1,
            "stage": "cotrain",
            "loss": metrics["loss"] / n_batches,
            "recon": metrics["recon"] / n_batches,
            "kl": metrics["kl"] / n_batches,
            "score_lsi": metrics["score_lsi"] / n_batches,
            "score_control": metrics["score_control"] / n_batches,
            "perc": metrics["perc"] / n_batches,
            "stiff": metrics["stiff"] / n_batches,
        }
        loss_records.append(epoch_metrics)

        print(f"Ep {ep+1} | Loss: {epoch_metrics['loss']:.4f} | Rec: {epoch_metrics['recon']:.4f} | "
              f"KL: {epoch_metrics['kl']:.4f} | LSI: {epoch_metrics['score_lsi']:.4f} | "
              f"Ctrl: {epoch_metrics['score_control']:.4f}")

        if len(mu_stats) > 0:
            log_latent_stats("VAE_Train", torch.cat(mu_stats, 0))

        if (ep + 1) % eval_freq == 0:
            results_lsi = evaluate_current_state(
                ep + 1,
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
            )
            if results_lsi is not None:
                results_lsi["epoch"] = ep + 1
                results_lsi["stage"] = "cotrain"
                results_lsi["tag"] = "LSI_Diff"
                eval_records.append(results_lsi)

            results_ctrl = evaluate_current_state(
                ep + 1,
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
            )
            if results_ctrl is not None:
                results_ctrl["epoch"] = ep + 1
                results_ctrl["stage"] = "cotrain"
                results_ctrl["tag"] = "Ctrl_Diff"
                eval_records.append(results_ctrl)

    # ===========================================================================
    # REFINEMENT STAGE
    # ===========================================================================
    epochs_refine = cfg.get("epochs_refine", 20)
    lr_refine = cfg.get("lr_refine", 1e-4)

    if epochs_refine > 0:
        print(f"\n--> Starting Refinement Stage ({epochs_refine} epochs, lr={lr_refine})...")

        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False

        opt_lsi_refine = optim.AdamW(unet_lsi.parameters(), lr=lr_refine, weight_decay=1e-4)
        opt_control_refine = optim.AdamW(unet_control.parameters(), lr=lr_refine, weight_decay=1e-4)

        for ep in range(epochs_refine):
            unet_lsi.train(); unet_control.train()
            metrics_refine = {k: 0.0 for k in ["score_lsi", "score_control"]}

            for x, _ in tqdm(train_l, desc=f"Refine Ep {ep+1}", leave=False):
                x = x.to(device)
                B = x.shape[0]

                with torch.no_grad():
                    _, mu, logvar = vae(x)
                    z0 = vae.reparameterize(mu, logvar)

                t = sample_log_uniform_times(B, cfg["t_min"], cfg["t_max"], device)
                alpha, sigma = get_ou_params(t.view(B,1,1,1))

                noise = torch.randn_like(z0)
                z_t = alpha * z0 + sigma * noise

                var_0 = torch.exp(logvar)
                mu_t = alpha * mu
                var_t = (alpha**2) * var_0 + (sigma**2)

                eps_target_lsi = sigma * ((z_t - mu_t) / (var_t + 1e-8))
                eps_pred_lsi = unet_lsi(z_t, t)
                score_loss_lsi = F.mse_loss(eps_pred_lsi, eps_target_lsi)

                opt_lsi_refine.zero_grad()
                score_loss_lsi.backward()
                nn.utils.clip_grad_norm_(unet_lsi.parameters(), 1.0)
                opt_lsi_refine.step()

                with torch.no_grad():
                    for p_online, p_ema in zip(unet_lsi.parameters(), unet_lsi_ema.parameters()):
                        p_ema.data.mul_(ema_decay).add_(p_online.data, alpha=1 - ema_decay)

                eps_pred_control = unet_control(z_t.detach(), t)
                score_loss_control = F.mse_loss(eps_pred_control, noise)

                opt_control_refine.zero_grad()
                score_loss_control.backward()
                nn.utils.clip_grad_norm_(unet_control.parameters(), 1.0)
                opt_control_refine.step()

                with torch.no_grad():
                    for p_online, p_ema in zip(unet_control.parameters(), unet_control_ema.parameters()):
                        p_ema.data.mul_(ema_decay).add_(p_online.data, alpha=1 - ema_decay)

                metrics_refine["score_lsi"] += score_loss_lsi.item()
                metrics_refine["score_control"] += score_loss_control.item()

            n_batches = len(train_l)
            global_epoch = cfg["epochs_vae"] + ep + 1
            epoch_metrics = {
                "epoch": global_epoch,
                "stage": "refine",
                "loss": 0.0,
                "recon": 0.0,
                "kl": 0.0,
                "score_lsi": metrics_refine["score_lsi"] / n_batches,
                "score_control": metrics_refine["score_control"] / n_batches,
                "perc": 0.0,
            }
            loss_records.append(epoch_metrics)

            print(f"Refine Ep {ep+1} | LSI: {epoch_metrics['score_lsi']:.4f} | "
                  f"Ctrl: {epoch_metrics['score_control']:.4f}")

            if (ep + 1) % eval_freq == 0:
                results_lsi = evaluate_current_state(
                        global_epoch,
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
                )
                if results_lsi is not None:
                    results_lsi["epoch"] = global_epoch
                    results_lsi["stage"] = "refine"
                    results_lsi["tag"] = "LSI_Diff_Refine"
                    eval_records.append(results_lsi)

                results_ctrl = evaluate_current_state(
                        global_epoch,
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
                )
                if results_ctrl is not None:
                    results_ctrl["epoch"] = global_epoch
                    results_ctrl["stage"] = "refine"
                    results_ctrl["tag"] = "Ctrl_Diff_Refine"
                    eval_records.append(results_ctrl)

    save_checkpoint(vae.state_dict(), os.path.join(cfg["ckpt_dir"], "vae_cotrained.pt"))
    save_checkpoint(unet_lsi_ema.state_dict(), os.path.join(cfg["ckpt_dir"], "unet_lsi.pt"))
    save_checkpoint(unet_control_ema.state_dict(), os.path.join(cfg["ckpt_dir"], "unet_control.pt"))

    loss_df = pd.DataFrame(loss_records)
    eval_df = pd.DataFrame(eval_records) if eval_records else pd.DataFrame()

    save_dataframes(loss_df, eval_df, results_dir)

    if len(loss_df) > 0 and len(eval_df) > 0:
        generate_all_visualizations(loss_df, eval_df, results_dir)
    else:
        print("--> Warning: Insufficient data for visualization generation")

    cfg_path = os.path.join(results_dir, "config.txt")
    with open(cfg_path, "w") as f:
        for k, v in cfg.items():
            f.write(f"{k}: {v}\n")
    print(f"--> Saved configuration to {cfg_path}")

    print(f"\n{'='*60}")
    print(f"Training complete! All results saved to: {results_dir}")
    print(f"{'='*60}")

    return loss_df, eval_df

def main():
    # CIFAR-10 Config - minimal changes from FashionMNIST
    cfg = {
        "batch_size": 128,
        "num_workers": 2,
        "use_latent_norm": True,
        "kl_reg_type": "norm",
        "score_w": 1.0,
        "lr_vae": 1e-3,
        "lr_ldm": 2e-4,
        "lr_refine": 7e-5,
        "epochs_vae": 300,
        "epochs_refine": 100,
        "latent_channels": 4,  # Bumped from 2 to 4 for CIFAR's RGB complexity
        "kl_w": 6e-4,
        "stiff_w": 1e-4,
        "score_w_vae": 0.45,
        "perc_w": 1.0,
        "t_min": 2e-5,
        "t_max": 2.0,
        "ckpt_dir": "checkpoints_cifar_comp",
        "seed": 42,
        "use_fixed_eval_banks": True,
        "sw2_n_projections": 1000,
        "load_from_checkpoint": False,
        "eval_freq": 10,
    }

    seed_everything(cfg["seed"])
    ensure_dir(cfg["ckpt_dir"])
    ensure_dir("samples")

    print("=== Dual Co-Training: LSI vs Control (Tweedie) - CIFAR-10 ===")
    train_vae_cotrained(cfg)

if __name__ == "__main__":
    main()
