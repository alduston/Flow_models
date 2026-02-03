from __future__ import annotations
from torch._higher_order_ops import out_dtype
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
# Dataset Configuration
# ---------------------------------------------------------------------------
DATASET_INFO = {
    "MNIST": {"class": torchvision.datasets.MNIST, "num_classes": 10, "img_size": 28},
    "FMNIST": {"class": torchvision.datasets.FashionMNIST, "num_classes": 10, "img_size": 28},
    "EMNIST": {"class": torchvision.datasets.EMNIST, "num_classes": 47, "split": "balanced", "img_size": 28},
    "KMNIST": {"class": torchvision.datasets.KMNIST, "num_classes": 10, "img_size": 28},
    "GCIFAR": {"class": torchvision.datasets.CIFAR10, "num_classes": 10, "img_size": 32, "grayscale": True},
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
    alpha = torch.exp(-t)
    sigma = torch.sqrt(1.0 - torch.exp(-2.0 * t) + 1e-8)
    return alpha, sigma


# ---------------------------------------------------------------------------
# Discrete VP / DDPM noise schedule utilities (cosine/linear)
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
    """Return betas[t] for a discrete VP/DDPM schedule.

    - 'cosine' matches the Improved DDPM cosine ᾱ(t) schedule (Nichol & Dhariwal).
    - 'linear' is a simple linear beta schedule.
    """
    schedule = str(schedule).lower()
    if num_timesteps < 1:
        raise ValueError("num_timesteps must be >= 1")

    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        return betas.clamp(1e-8, 0.999)

    if schedule == "cosine":
        # Build ᾱ for t in {0..T}
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps, dtype=torch.float32) / num_timesteps
        alphas_cumprod = torch.cos(((t + cosine_s) / (1 + cosine_s)) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        # Convert ᾱ to betas
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-8, 0.999)

    raise ValueError(f"Unknown noise_schedule: {schedule}")

def make_ou_schedule(cfg: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Precompute a discrete OU schedule:
      - T discrete times spanning [t_min, t_max]
      - alpha(t), sigma(t) from get_ou_params
    Choose log-spaced times so uniform index sampling approximates log-uniform t sampling.
    """
    T = int(cfg.get("num_train_timesteps", 1000))
    t_min = float(cfg.get("t_min", 2e-5))
    t_max = float(cfg.get("t_max", 2.0))

    # log-spaced grid from t_min -> t_max (monotone increasing)
    times = torch.logspace(
        math.log10(t_min),
        math.log10(t_max),
        T,
        device=device,
        dtype=torch.float32,
    )

    # get OU params on the grid; squeeze to [T]
    a, s = get_ou_params(times.view(T, 1, 1, 1))
    alpha = a.view(T).float()
    sigma = s.view(T).float()

    return {
        "T": torch.tensor(T, device=device, dtype=torch.long),
        "times": times,   # [T]
        "alpha": alpha,   # [T]
        "sigma": sigma,   # [T]
    }

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

def log_latent_stats(name, z):
    with torch.no_grad():
        mean_norm = z.mean(0).norm().item()
        std_mean = z.std(0).mean().item()
        max_val = z.abs().max().item()
        print(f"  [{name}] Latent Stats | Mean Norm: {mean_norm:.4f} | Avg Std: {std_mean:.4f} | Max: {max_val:.4f}")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LeNet Feature Extractor for FID (non-FMNIST datasets)
# ---------------------------------------------------------------------------

class LeNetFeatureExtractor(nn.Module):
    """
    LeNet-style CNN classifier for FID feature extraction on grayscale datasets.
    Feature dimension: 256 (penultimate layer)
    """
    def __init__(self, num_classes: int = 10, feature_dim: int = 256, img_size: int = 28):
        super().__init__()
        self.feature_dim = feature_dim
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
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


def train_fid_classifier(train_loader, num_classes, device, epochs=10, lr=1e-3, checkpoint_path=None, img_size=32):
    """Train LeNet classifier for FID feature extraction."""
    model = LeNetFeatureExtractor(num_classes=num_classes, img_size=img_size).to(device)

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
    """Get appropriate FID model: Inception for FMNIST/GCIFAR, LeNet for others."""
    if dataset_key in ("FMNIST", "GCIFAR"):
        print(f"--> Using Inception features for FID ({dataset_key})")
        return None, False

    # Get img_size from dataset info (default 32 for padded 28x28 datasets)
    img_size = DATASET_INFO.get(dataset_key, {}).get("img_size", 28)
    # For datasets that get padded (28->32), use 32 for the classifier
    effective_img_size = 32 if img_size == 28 else img_size
    
    print(f"--> Training LeNet classifier for FID ({dataset_key})")
    checkpoint_path = os.path.join(ckpt_dir, f"fid_classifier_{dataset_key.lower()}.pt")
    model = train_fid_classifier(train_loader, num_classes, device, epochs=10, checkpoint_path=checkpoint_path, img_size=effective_img_size)
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
    if score_net is None:
        return 0.0

    score_net.eval()

    n_data = encoder_mus.shape[0]
    if labels is not None:
        assert labels.shape[0] == n_data, "labels must align with encoder_mus/logvars"

    num_samples = min(int(num_samples), int(n_data))
    sample_indices = torch.randperm(n_data, device="cpu")[:num_samples]

    # --- Discrete OU schedule ---
    ou_sched = make_ou_schedule(cfg, device)
    T = int(ou_sched["T"].item())

    # Choose a fixed grid of discrete indices
    # Use round() so endpoints are included and avoid repeated zeros from long-cast.
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

            # Labels for CFG-conditional nets
            y_batch = None
            if labels is not None:
                y_batch = labels[batch_indices].to(device=device, dtype=torch.long).view(-1)
                if num_classes is not None:
                    if (y_batch.min() < 0) or (y_batch.max() >= num_classes):
                        raise ValueError("labels out of range for num_classes")

            # Sample z0 ~ q(z0 | x) using the encoder Gaussian
            eps_0 = torch.randn_like(batch_mu)
            z0 = batch_mu + batch_std * eps_0

            # Evaluate gap across timepoints
            for t_idx_scalar in t_idx_grid:
                # Expand scalar index to [bsz]
                t_idx = t_idx_scalar.expand(bsz)

                # time embedding: actual OU time at this index, expanded to [bsz]
                t_val = ou_sched["times"].gather(0, t_idx_scalar.view(1)).view(1)  # [1]
                t = t_val.expand(bsz).float()  # [bsz]

                # OU alpha/sigma at this index, broadcast to z0 shape
                alpha = extract_schedule(ou_sched["alpha"], t_idx, z0.shape)  # [bsz,1,1,1]
                sigma = extract_schedule(ou_sched["sigma"], t_idx, z0.shape)  # [bsz,1,1,1]

                # Forward diffuse z0 -> z_t
                noise = torch.randn_like(z0)
                z_t = alpha * z0 + sigma * noise

                # Conditional moments under encoder Gaussian pushed through OU
                mu_t = alpha * batch_mu
                var_t = (alpha ** 2) * batch_var + (sigma ** 2)

                # LSI target in eps-parameterization
                eps_target_lsi = sigma * ((z_t - mu_t) / (var_t + 1e-8))

                # Model prediction
                eps_pred = score_net(z_t, t, y_batch)

                # Weighted squared error (your gap definition)
                sigma_sq = sigma ** 2 + 1e-8
                eps_diff_sq = (eps_pred - eps_target_lsi) ** 2
                score_gap_per_sample = (eps_diff_sq / sigma_sq).sum(dim=(1, 2, 3))  # [bsz]

                total_lsi_gap += score_gap_per_sample.sum().item()
                total_count += bsz

    return total_lsi_gap / total_count if total_count > 0 else 0.0


class VAE(nn.Module):
    def __init__(
        self,
        latent_channels: int = 4,
        base_ch: int = 32,
        use_norm: bool = False,
        img_size: int = 28,
        # --- NEW: optional conditional encoder ---
        num_classes: int | None = None,
        null_label: int | None = None,
        cond_emb_dim: int = 64,
    ):
        super().__init__()
        # Encoder
        self.use_norm = use_norm
        self.img_size = img_size
        self.latent_channels = int(latent_channels)
        self.base_ch = int(base_ch)

        # If num_classes is provided, encoder becomes class-conditional.
        self.num_classes = None if num_classes is None else int(num_classes)
        self.null_label = None if self.num_classes is None else int(self.num_classes if null_label is None else null_label)

        self.enc_conv_in = nn.Conv2d(1, base_ch, 3, 1, 1)

        self.enc_blocks = nn.ModuleList([
            nn.Sequential(VAEResBlock(base_ch, base_ch), nn.Conv2d(base_ch, base_ch * 2, 3, 2, 1)),
            nn.Sequential(VAEResBlock(base_ch * 2, base_ch * 2), nn.Conv2d(base_ch * 2, base_ch * 4, 3, 2, 1)),
            nn.Sequential(VAEResBlock(base_ch * 4, base_ch * 4), AttentionBlock(base_ch * 4), VAEResBlock(base_ch * 4, base_ch * 4)),
        ])

        enc_out_ch = base_ch * 4

        # --- NEW: conditional bottleneck bias (zero-init) ---
        if self.num_classes is not None:
            # reserve num_classes as unconditional/null label (consistent with CFG dropout)
            self.y_emb = nn.Embedding(self.num_classes + 1, cond_emb_dim)
            self.cond_proj = nn.Linear(cond_emb_dim, enc_out_ch)
            nn.init.zeros_(self.cond_proj.weight)
            nn.init.zeros_(self.cond_proj.bias)
        else:
            self.y_emb = None
            self.cond_proj = None

        self.mu = nn.Conv2d(enc_out_ch, latent_channels, 1)
        self.logvar = nn.Conv2d(enc_out_ch, latent_channels, 1)

        # GroupNorm on eta (mu)
        if self.use_norm:
            self.gn_mu = nn.GroupNorm(num_groups=1, num_channels=latent_channels, affine=False)

        # Decoder (unchanged)
        self.dec_conv_in = nn.Conv2d(latent_channels, base_ch * 4, 1)
        self.dec_blocks = nn.ModuleList([
            nn.Sequential(VAEResBlock(base_ch * 4, base_ch * 4), AttentionBlock(base_ch * 4), VAEResBlock(base_ch * 4, base_ch * 4)),
            nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(base_ch * 4, base_ch * 2, 3, 1, 1), VAEResBlock(base_ch * 2, base_ch * 2)),
            nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(base_ch * 2, base_ch, 3, 1, 1), VAEResBlock(base_ch, base_ch)),
        ])
        self.dec_out = nn.Sequential(
            nn.GroupNorm(16, base_ch), nn.SiLU(), nn.Conv2d(base_ch, 1, 3, 1, 1)
        )

    def encode(self, x: torch.Tensor, y: torch.Tensor | None = None):
        h = self.enc_conv_in(x)
        for block in self.enc_blocks:
            h = block(h)

        # --- NEW: inject class-conditioning at bottleneck if enabled ---
        if self.num_classes is not None:
            B, C, H, W = h.shape
            if y is None:
                y = torch.full((B,), self.null_label, device=h.device, dtype=torch.long)
            else:
                y = y.to(device=h.device, dtype=torch.long).view(B)

            emb = self.y_emb(y)                         # [B, cond_emb_dim]
            bias = self.cond_proj(emb).view(B, C, 1, 1)  # [B, C, 1, 1]
            h = h + bias                                 # broadcast to [B, C, H, W]

        mu = self.mu(h)
        logvar = self.logvar(h)

        if self.use_norm:
            mu = self.gn_mu(mu)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_conv_in(z)
        for block in self.dec_blocks:
            h = block(h)
        return torch.tanh(self.dec_out(h))

    def forward(self, x, y: torch.Tensor | None = None):
        mu, logvar = self.encode(x, y=y)
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


'''
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
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 32,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        num_classes: int | None = None,
    ):
        super().__init__()
        self.time_embed = TimeEmbedding(base_channels)

        # Optional classifier-free class conditioning:
        # - if num_classes is provided, we reserve an extra "null" label at index num_classes
        # - passing y=None uses the null label
        self.num_classes = num_classes
        self.null_label = num_classes if num_classes is not None else None
        self.label_emb = nn.Embedding(num_classes + 1, base_channels * 4) if num_classes is not None else None

        self.head = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.downs = nn.ModuleList()
        ch = base_channels
        chs = [ch]
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResBlock(ch, out_ch, base_channels * 4))
                ch = out_ch
                chs.append(ch)
            if i != len(channel_mults) - 1:
                self.downs.append(nn.Conv2d(ch, ch, 3, 2, 1))
                chs.append(ch)

        self.mid = nn.ModuleList([
            ResBlock(ch, ch, base_channels * 4),
            AttentionBlock(ch),
            ResBlock(ch, ch, base_channels * 4),
        ])

        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                skip = chs.pop()
                self.ups.append(ResBlock(ch + skip, out_ch, base_channels * 4))
                ch = out_ch
            if i != 0:
                self.ups.append(nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(ch, ch, 3, 1, 1)))

        self.out = nn.Sequential(
            make_group_norm(ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        # t: [B], y: [B] (class ids), or y=None for unconditional branch
        emb = self.time_embed(t)

        if self.label_emb is not None:
            if y is None:
                y = torch.full((t.shape[0],), int(self.null_label), device=t.device, dtype=torch.long)
            else:
                if not torch.is_tensor(y):
                    y = torch.tensor(y, device=t.device)
                y = y.to(device=t.device, dtype=torch.long).view(-1)
                if y.shape[0] != t.shape[0]:
                    y = y.expand(t.shape[0])
            emb = emb + self.label_emb(y)

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
'''


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


class UNetModel(nn.Module):
    """
    Same external API as your current UNetModel, but slightly stronger blocks.
    """
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 32,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        num_classes: int | None = None,
        *,
        dropout: float = 0.0,
        # Add attention at exactly one additional level by default (second-to-last resolution).
        # For (1,2,4): level 1 corresponds to the "middle" spatial resolution.
        attn_levels: Optional[Tuple[int, ...]] = None,
        attn_heads: int = 4,
        # If you stack many blocks, sqrt(0.5) can help; default 1.0 keeps behavior closer to current.
        skip_scale: float = 1.0,
        mid_attn: bool = True,
    ):
        super().__init__()
        self.time_embed = TimeEmbedding(base_channels)

        # Optional classifier-free class conditioning:
        self.num_classes = num_classes
        self.null_label = num_classes if num_classes is not None else None
        self.label_emb = nn.Embedding(num_classes + 1, base_channels * 4) if num_classes is not None else None

        if attn_levels is None:
            # One extra attention level besides the mid block:
            # - if there are >=2 levels, use the second-to-last; else none.
            attn_levels = (max(len(channel_mults) - 2, 0),) if len(channel_mults) >= 2 else tuple()
        self.attn_levels = set(int(i) for i in attn_levels)

        self.head = nn.Conv2d(in_channels, base_channels, 3, 1, 1)

        # Down path
        self.downs = nn.ModuleList()
        ch = base_channels
        chs = [ch]
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for r in range(num_res_blocks):
                # Only add attention once per chosen level (on the last resblock of that level).
                use_attn = (level in self.attn_levels) and (r == num_res_blocks - 1)
                self.downs.append(
                    ResBlock(
                        ch, out_ch, base_channels * 4,
                        dropout=dropout, use_attn=use_attn, attn_heads=attn_heads, skip_scale=skip_scale
                    )
                )
                ch = out_ch
                chs.append(ch)
            if level != len(channel_mults) - 1:
                self.downs.append(nn.Conv2d(ch, ch, 3, 2, 1))
                chs.append(ch)

        # Mid
        mid_blocks = [
            ResBlock(ch, ch, base_channels * 4, dropout=dropout, use_attn=False, attn_heads=attn_heads, skip_scale=skip_scale),
        ]
        if mid_attn:
            mid_blocks.append(AttentionBlock(ch, num_heads=attn_heads))
        mid_blocks.append(
            ResBlock(ch, ch, base_channels * 4, dropout=dropout, use_attn=False, attn_heads=attn_heads, skip_scale=skip_scale)
        )
        self.mid = nn.ModuleList(mid_blocks)

        # Up path
        self.ups = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for r in range(num_res_blocks + 1):
                skip = chs.pop()
                # Again: only once per chosen level (on the last resblock before upsample).
                use_attn = (level in self.attn_levels) and (r == num_res_blocks)
                self.ups.append(
                    ResBlock(
                        ch + skip, out_ch, base_channels * 4,
                        dropout=dropout, use_attn=use_attn, attn_heads=attn_heads, skip_scale=skip_scale
                    )
                )
                ch = out_ch
            if level != 0:
                self.ups.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                              nn.Conv2d(ch, ch, 3, 1, 1)))

        self.out = nn.Sequential(
            make_group_norm(ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        # t: [B], y: [B] (class ids), or y=None for unconditional branch
        t = t.view(-1)
        emb = self.time_embed(t)

        if self.label_emb is not None:
            if y is None:
                y = torch.full((t.shape[0],), int(self.null_label), device=t.device, dtype=torch.long)
            else:
                if not torch.is_tensor(y):
                    y = torch.tensor(y, device=t.device)
                y = y.to(device=t.device, dtype=torch.long).view(-1)
                if y.shape[0] != t.shape[0]:
                    y = y.expand(t.shape[0])
            emb = emb + self.label_emb(y)

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

# ----------------------------------------------------------------------
# Suggested "minimal beef" settings to try first (no code changes needed):
#
# 1) Increase width a bit (smallest retune burden):
#    UNetModel(base_channels=48, channel_mults=(1,2,4), num_res_blocks=2, dropout=0.05, attn_levels=(1,))
#
# 2) Or keep width, add one more level (compute bump, still modest):
#    UNetModel(base_channels=32, channel_mults=(1,2,4,4), num_res_blocks=2, dropout=0.05, attn_levels=(2,))
#
# If joint training gets twitchy, set skip_scale=math.sqrt(0.5) and/or dropout=0.0.
# ----------------------------------------------------------------------


class UniversalSampler:
    def __init__(
        self,
        method: str = "heun_sde",
        num_steps: int = 20,
        t_min: float = 2e-5,
        t_max: float = 2.0,
        schedule_cfg: Dict[str, Any] | None = None,
        ddim_eta: float = 0.0,
    ):
        self.num_steps = int(num_steps)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.method = str(method).lower()

        # For discrete samplers (DDIM), we need the same schedule used in training.
        self.schedule_cfg = schedule_cfg
        self.ddim_eta = float(ddim_eta)
        self._ddpm_schedule: Dict[str, torch.Tensor] | None = None

    @staticmethod
    def _predict_eps(
        unet,
        x: torch.Tensor,
        t_vec: torch.Tensor,
        y: torch.Tensor | None = None,
        cfg_scale: float | None = None,
    ) -> torch.Tensor:
        """Classifier-Free Guidance in eps-parameterization."""
        if cfg_scale is None or cfg_scale <= 0.0 or y is None:
            return unet(x, t_vec, y)

        # unconditional branch uses y=None (UNetModel maps to null label internally)
        eps_uncond = unet(x, t_vec, None)
        eps_cond = unet(x, t_vec, y)
        return eps_uncond + float(cfg_scale) * (eps_cond - eps_uncond)

    # ------------------------- Continuous-time OU samplers -------------------------

    def get_ode_derivative(self, x, t, unet, y=None, cfg_scale=None):
        B = x.shape[0]
        t_vec = t.expand(B)
        eps_pred = self._predict_eps(unet, x, t_vec, y=y, cfg_scale=cfg_scale)
        _, sigma = get_ou_params(t_vec.view(B, 1, 1, 1))
        inv_sigma = 1.0 / (sigma + 1e-10)
        return -x + inv_sigma * eps_pred

    def get_rev_sde_drift(self, x, t, unet, y=None, cfg_scale=None):
        B = x.shape[0]
        t_vec = t.expand(B)
        eps_pred = self._predict_eps(unet, x, t_vec, y=y, cfg_scale=cfg_scale)
        _, sigma = get_ou_params(t_vec.view(B, 1, 1, 1))
        inv_sigma = 1.0 / (sigma + 1e-8)
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

        t_vec = t_curr.expand(B)
        eps_pred = self._predict_eps(unet, x, t_vec, y=y, cfg_scale=cfg_scale)
        _, sigma = get_ou_params(t_vec.view(B, 1, 1, 1))
        d_curr = -x + (1.0 / (sigma + 1e-10)) * eps_pred
        x_proposed = x + dt * d_curr

        if t_next > self.t_min:
            t_next_vec = t_next.expand(B)
            eps_next = self._predict_eps(unet, x_proposed, t_next_vec, y=y, cfg_scale=cfg_scale)
            _, sigma_next = get_ou_params(t_next_vec.view(B, 1, 1, 1))
            d_next = -x_proposed + (1.0 / (sigma_next + 1e-10)) * eps_next
            x = x + 0.5 * dt * (d_curr + d_next)
        else:
            x = x_proposed

        return x

    def step_heun_sde(self, x, t_curr, t_next, unet, y=None, cfg_scale=None, generator=None):
        dt = t_next - t_curr
        dt_abs = torch.abs(dt).clamp_min(1e-12)

        if generator is None:
            noise = torch.randn_like(x)
        else:
            noise = torch.randn(x.shape, device=x.device, generator=generator)
        dW = torch.sqrt(2.0 * dt_abs) * noise

        b_curr = self.get_rev_sde_drift(x, t_curr, unet, y=y, cfg_scale=cfg_scale)
        x_hat = x + dt * b_curr + dW

        b_next = self.get_rev_sde_drift(x_hat, t_next, unet, y=y, cfg_scale=cfg_scale)
        x_new = x + 0.5 * dt * (b_curr + b_next) + dW
        return x_new

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
        """
        OU-DDIM step: one UNet call at t_curr, update to t_next using OU (alpha(t), sigma(t)).
        Deterministic when ddim_eta=0.
        """
        B = x.shape[0]

        # UNet eval time is exactly the OU time (no clamping, no index mapping)
        t_vec = t_curr.expand(B)
        eps_pred = self._predict_eps(unet, x, t_vec, y=y, cfg_scale=cfg_scale)

        alpha_t, sigma_t = get_ou_params(t_vec.view(B, 1, 1, 1))
        alpha_t = alpha_t.to(x.dtype)
        sigma_t = sigma_t.to(x.dtype)

        # x0 prediction under OU parameterization
        x0_pred = (x - sigma_t * eps_pred) / (alpha_t + 1e-8)

        # If we're at (or below) the terminal noise level, return the denoised prediction
        if (t_next <= self.t_min).item():
            return x0_pred

        t_next_vec = t_next.expand(B)
        alpha_next, sigma_next = get_ou_params(t_next_vec.view(B, 1, 1, 1))
        alpha_next = alpha_next.to(x.dtype)
        sigma_next = sigma_next.to(x.dtype)

        # DDIM-style stochasticity using a(t) = alpha(t)^2
        a_t = alpha_t ** 2
        a_next = alpha_next ** 2

        eta = float(self.ddim_eta)
        if eta <= 0.0:
            sigma_ddim = torch.zeros_like(sigma_next)
        else:
            # same algebra as DDIM, but with continuous a(t)
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

    def sample(
        self,
        unet,
        shape=None,
        device=None,
        x_init=None,
        generator=None,
        y: torch.Tensor | None = None,
        cfg_scale: float | None = None,
    ):
        unet.eval()

        if x_init is None:
            assert shape is not None and device is not None
            x = torch.randn(shape, device=device, generator=generator)
        else:
            x = x_init
        device = x.device


        # --- Discrete DDIM path ---
        if self.method == "ddim":
            # Use the same OU log-universe schedule as the ODE/SDE solvers
            ts = torch.logspace(
                math.log10(self.t_max),
                math.log10(self.t_min),
                self.num_steps + 1,
                device=device
            )

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

            return x

        # --- Continuous OU samplers (existing path) ---
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
                x = self.step_rk4_ode(x, t_curr, t_next, unet, y=y, cfg_scale=cfg_scale)
            elif self.method == "euler_ode":
                x = self.step_euler_ode(x, t_curr, t_next, unet, y=y, cfg_scale=cfg_scale)
            elif self.method == "heun_ode":
                x = self.step_heun_ode(x, t_curr, t_next, unet, y=y, cfg_scale=cfg_scale)
            elif self.method == "heun_sde":
                x = self.step_heun_sde(x, t_curr, t_next, unet, y=y, cfg_scale=cfg_scale, generator=generator)
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
    fid_model=None,
    use_lenet_fid=False,
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

    lsi_gap_unet = compute_lsi_gap(
        unet,
        encoder_mus,
        encoder_logvars,
        cfg,
        device,
        labels=None,  # unconditional branch
        num_classes=cfg.get("num_classes", None),
        num_samples=min(5000, target_count),
        num_time_points=50,
        batch_size=bs,
    )

    # -----------------------------------------------------------------------
    # Sampler configurations (unconditional baseline)
    # -----------------------------------------------------------------------
    configs = [
        {"method": "VAE_Rec_eps", "steps": 0, "desc": "Recon (posterior z)", "use_rand_token": False},
    ]
    if unet is not None:
        configs.extend([
            {"method": "rk4_ode",  "steps": 20, "desc": "RandToken (RK4) CFG0", "use_rand_token": True, "cfg_level": 0},
            {"method": "rk4_ode",  "steps": 20, "desc": "RandToken (RK4) CFG1", "use_rand_token": True, "cfg_level": 1},
            {"method": "rk4_ode",  "steps": 20, "desc": "RandToken (RK4) CFG1.5", "use_rand_token": True, "cfg_level": 1.5},
            {"method": "rk4_ode",  "steps": 20, "desc": "RandToken (RK4) CFG2", "use_rand_token": True, "cfg_level": 2.0}, 
            {"method": "rk4_ode",  "steps": 20, "desc": "RandToken (RK4) CFG3", "use_rand_token": True, "cfg_level": 3.0}, 
        ])

    results = []

    # -----------------------------------------------------------------------
    # Shared banks for comparability (noise + random labels)
    # -----------------------------------------------------------------------
    # Align fixed noise bank with the realized dataset order
    noise_bank_all = fixed_noise_bank[sample_indices] if fixed_noise_bank is not None else None

    # Fixed random token bank: used when sampler config has use_rand_token=True
    rand_token_bank_all = None
    if unet is not None and cfg.get("num_classes", None) is not None:
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
    cfg_eval_scale = float(cfg.get("cfg_eval_scale", 2.0))
    for scfg in configs:
        method = scfg["method"]
        steps = int(scfg.get("steps", 0))
        desc = scfg.get("desc", "")
        use_rand_token = bool(scfg.get("use_rand_token", False))
        config_suffix = "_randtok" if use_rand_token else ""
        config_name = f"{method}@{steps}{config_suffix}"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            if method == "VAE_Rec_eps":
                fake_imgs = torch.cat([
                    vae.decode(real_latents_A[i:i + bs].to(device)).cpu()
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
                    t_min=cfg["t_min"],
                    t_max=cfg["t_max"],
                )
                if method == "ddim":
                    sampler_kwargs.update(
                        schedule_cfg=cfg,
                        ddim_eta=float(cfg.get("ddim_eta", 0.0)),
                    )
                sampler = UniversalSampler(**sampler_kwargs)
                fake_latents_list, fake_imgs_list = [], []

                for i in range(0, target_count, bs):
                    batch_sz = min(bs, target_count - i)

                    y_batch = None
                    if use_rand_token and rand_token_bank_all is not None:
                        y_batch = rand_token_bank_all[i:i + batch_sz].to(device)

                    g_scale = cfg_eval_scale if use_rand_token else None
                    if noise_bank_all is not None:
                        xT = noise_bank_all[i:i + batch_sz].to(device)
                        #z_gen = sampler.sample(unet, x_init=xT, y=y_batch, cfg_scale=None)
                        z_gen = sampler.sample(unet, x_init=xT, y=y_batch, cfg_scale=g_scale)
                    else:
                        #z_gen = sampler.sample(unet, shape=(batch_sz, *latent_shape), device=device, y=y_batch, cfg_scale=None)
                        z_gen = sampler.sample(unet, shape=(batch_sz, *latent_shape), device=device, y=y_batch, cfg_scale=g_scale)

                    fake_latents_list.append(z_gen.cpu())
                    fake_imgs_list.append(vae.decode(z_gen).cpu())

                fake_latents = torch.cat(fake_latents_list, 0)
                fake_imgs = torch.cat(fake_imgs_list, 0)
                fake_flat = fake_latents.view(fake_latents.shape[0], -1).to(device)
                w2 = compute_sw2(real_flat_A, fake_flat, n_projections=sw2_nproj, theta=fixed_sw2_theta)
                lsi_gap = lsi_gap_unet

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
        })

        # Save sample panels for the main sweep
        if method in ("VAE_Rec_eps",) or "rk4" in method or "heun" in method or method == "ddim":
            if results_dir is not None:
                samples_dir = os.path.join(results_dir, "samples")
                os.makedirs(samples_dir, exist_ok=True)
                save_path = os.path.join(samples_dir, f"{prefix}_{method}_{steps}{config_suffix}_ep{epoch_idx}.png")
            else:
                save_path = os.path.join("samples", f"{prefix}_{method}_{steps}{config_suffix}_ep{epoch_idx}.png")
            panel = fake_imgs[:16] if fake_imgs.shape[0] >= 16 else fake_imgs
            tv_utils.save_image((panel + 1) / 2, save_path, nrow=4, padding=2)

    # Print main results
    print(f"\n  >>> Sweep Results [{prefix}] <<<")
    print(f"  {'Config':<15} | {'Desc':<20} | {'FID':<8} | {'KID':<10} | {'SW2':<10} | {'Div':<8} | {'LSI Gap':<10}")
    print("  " + "-" * 100)
    for r in results:
        print(f"  {r['config']:<15} | {r['desc']:<20} | {r['fid']:<8.2f} | {r['kid']:<10.4f} | "
              f"{r['w2']:<10.6f} | {r['div']:<8.4f} | {r['lsi_gap']:<10.4f}")
    print("  " + "-" * 100 + "\n")

    # Flatten for DataFrame logging
    output_dict: Dict[str, Any] = {}
    for r in results:
        config = r["config"]
        if "VAE_Rec_eps" in config:
            output_dict["fid_vae_recon"] = r["fid"]
            output_dict["kid_vae_recon"] = r["kid"]
            output_dict["sw2_vae_recon"] = r["w2"]
            output_dict["div_vae_recon"] = r["div"]
        elif "rk4" in config.lower():
            steps_str = config.split("@")[1] if "@" in config else "10"
            output_dict[f"fid_rk4_{steps_str}"] = r["fid"]
            output_dict[f"kid_rk4_{steps_str}"] = r["kid"]
            output_dict[f"sw2_rk4_{steps_str}"] = r["w2"]
            output_dict[f"div_rk4_{steps_str}"] = r["div"]
            output_dict[f"lsi_gap_rk4_{steps_str}"] = r["lsi_gap"]
        elif "heun" in config.lower():
            steps_str = config.split("@")[1] if "@" in config else "20"
            output_dict[f"fid_heun_{steps_str}"] = r["fid"]
            output_dict[f"kid_heun_{steps_str}"] = r["kid"]
            output_dict[f"sw2_heun_{steps_str}"] = r["w2"]
            output_dict[f"div_heun_{steps_str}"] = r["div"]
            output_dict[f"lsi_gap_heun_{steps_str}"] = r["lsi_gap"]

    # -----------------------------------------------------------------------
    # Optional: class-conditional evaluation + CFG
    # -----------------------------------------------------------------------
    eval_class_labels = cfg.get("eval_class_labels", None)
    if eval_class_labels is not None and not isinstance(eval_class_labels, (list, tuple)):
        eval_class_labels = [int(eval_class_labels)]

    cfg_eval_scale = float(cfg.get("cfg_eval_scale", 3.0))

    if unet is not None and eval_class_labels:
        print(f"  Conditional eval on labels: {list(eval_class_labels)} (CFG scale={cfg_eval_scale:g})")
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
                    vae.decode(real_latents_A_y[i:i + bs].to(device)).cpu()
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
            if results_dir is not None:
                samples_dir = os.path.join(results_dir, "samples")
                os.makedirs(samples_dir, exist_ok=True)
                save_path = os.path.join(samples_dir, f"{prefix}_VAE_Rec_eps_0_y{y0}_ep{epoch_idx}.png")
                panel = fake_imgs_recon_y[:16] if fake_imgs_recon_y.shape[0] >= 16 else fake_imgs_recon_y
                tv_utils.save_image((panel + 1) / 2, save_path, nrow=4, padding=2)

            # ---------------------------------------------------------------
            # Diffusion conditional + CFG methods
            # ---------------------------------------------------------------
            for method, steps, desc in [("heun_ode", 20, "Baseline (Heun)"), ("rk4_ode", 10, "Smoothness (RK4)")]:
                for g_scale in [0.0, cfg_eval_scale]:
                    tag = "cond" if g_scale <= 0.0 else f"cfg{g_scale:g}"
                    mode = "cond" if g_scale <= 0.0 else f"cfg{g_scale:g}"

                    sampler = UniversalSampler(
                        method=method,
                        num_steps=steps,
                        t_min=cfg["t_min"],
                        t_max=cfg["t_max"],
                    )
                    fake_latents_list, fake_imgs_list = [], []

                    for i in range(0, n_y, bs):
                        batch_sz = min(bs, n_y - i)
                        y_batch = torch.full((batch_sz,), y0, device=device, dtype=torch.long)

                        if noise_bank_y is not None:
                            xT = noise_bank_y[i:i + batch_sz].to(device)
                            z_gen = sampler.sample(
                                unet,
                                x_init=xT,
                                y=y_batch,
                                cfg_scale=(None if g_scale <= 0.0 else float(g_scale)),
                            )
                        else:
                            z_gen = sampler.sample(
                                unet,
                                shape=(batch_sz, *latent_shape),
                                device=device,
                                y=y_batch,
                                cfg_scale=(None if g_scale <= 0.0 else float(g_scale)),
                            )

                        fake_latents_list.append(z_gen.cpu())
                        fake_imgs_list.append(vae.decode(z_gen).cpu())

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
                    if results_dir is not None:
                        samples_dir = os.path.join(results_dir, "samples")
                        os.makedirs(samples_dir, exist_ok=True)
                        save_path = os.path.join(samples_dir, f"{prefix}_{method}_{steps}_y{y0}_{tag}_ep{epoch_idx}.png")
                    else:
                        save_path = os.path.join("samples", f"{prefix}_{method}_{steps}_y{y0}_{tag}_ep{epoch_idx}.png")
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
        metric_prefix: One of 'fid', 'kid', 'sw2', 'div', 'lsi_gap'
    
    Returns:
        List of column names matching the pattern, e.g., 
        ['fid_rk4_20_randtok', 'fid_heun_20', 'fid_ddim_50_randtok_cfg2.0']
    """
    pattern = re.compile(rf'^{metric_prefix}_(?!vae_recon)')  # exclude vae_recon variants
    return [col for col in eval_df.columns if pattern.match(col)]


def parse_metric_column(col_name):
    """
    Parse a metric column name into its components.
    
    Examples:
        'fid_rk4_10' -> {'metric': 'fid', 'method': 'rk4', 'steps': '10', 'suffix': ''}
        'fid_rk4_20_randtok' -> {'metric': 'fid', 'method': 'rk4', 'steps': '20', 'suffix': '_randtok'}
        'kid_heun_20_randtok_cfg2.0' -> {'metric': 'kid', 'method': 'heun', 'steps': '20', 'suffix': '_randtok_cfg2.0'}
    """
    parts = col_name.split('_')
    metric = parts[0]  # fid, kid, sw2, div, lsi (note: lsi_gap has underscore)
    
    # Handle lsi_gap specially
    if metric == 'lsi' and len(parts) > 1 and parts[1] == 'gap':
        metric = 'lsi_gap'
        parts = [metric] + parts[2:]  # Rejoin and skip 'gap'
    
    if len(parts) < 3:
        return None  # Not a valid metric column
    
    method = parts[1]  # rk4, heun, ddim, euler, etc.
    steps = parts[2]   # 10, 20, 50, etc.
    
    # Everything after method_steps is the suffix
    suffix = '_'.join(parts[3:]) if len(parts) > 3 else ''
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
    
    for metric_type in ['fid', 'kid', 'sw2', 'div', 'lsi_gap']:
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
        'rk4': 'RK4 ODE',
        'heun': 'Heun ODE', 
        'euler': 'Euler ODE',
        'ddim': 'DDIM',
    }
    base = f"{method_names.get(method, method.upper())} {steps} steps"
    
    if suffix:
        # Parse suffix for human readability
        if '_randtok' in suffix:
            base += " (RandTok"
            if '_cfg' in suffix:
                cfg_match = re.search(r'_cfg([\d.]+)', suffix)
                if cfg_match:
                    base += f", CFG={cfg_match.group(1)}"
            base += ")"
        elif '_cfg' in suffix:
            cfg_match = re.search(r'_cfg([\d.]+)', suffix)
            if cfg_match:
                base += f" (CFG={cfg_match.group(1)})"
    
    return base


def generate_all_visualizations(loss_df, eval_df, results_dir):
    """
    Generate visualization plots dynamically based on available metrics.
    
    This replaces the hardcoded version that expected specific column names.
    """
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Discover all metric groups
    metric_groups = get_metric_groups(eval_df)
    
    if not metric_groups:
        print("--> Warning: No metric columns found in eval_df. Skipping metric plots.")
        # Still generate loss plots
        plot_idx = 1
        plot_vae_recon_loss(loss_df, os.path.join(plots_dir, f"{plot_idx:02d}_vae_recon_loss.png"))
        plot_idx += 1
        plot_score_losses(loss_df, os.path.join(plots_dir, f"{plot_idx:02d}_score_losses.png"))
        print(f"--> Visualization suite complete ({plot_idx} plots generated)!")
        return
    
    print(f"\n--> Generating visualization suite...")
    print(f"    Found {len(metric_groups)} sampler configurations:")
    for (method, steps, suffix) in sorted(metric_groups.keys()):
        label = format_config_label(method, steps, suffix)
        metrics_available = list(metric_groups[(method, steps, suffix)].keys())
        print(f"      - {label}: {metrics_available}")
    
    plot_idx = 1
    
    # --- Loss plots (always generate these) ---
    plot_vae_recon_loss(loss_df, os.path.join(plots_dir, f"{plot_idx:02d}_vae_recon_loss.png"))
    plot_idx += 1
    plot_score_losses(loss_df, os.path.join(plots_dir, f"{plot_idx:02d}_score_losses.png"))
    plot_idx += 1
    
    # --- Metric plots for each sampler configuration ---
    for (method, steps, suffix) in sorted(metric_groups.keys()):
        group = metric_groups[(method, steps, suffix)]
        config_label = format_config_label(method, steps, suffix)
        config_tag = f"{method}_{steps}{suffix}"  # For filenames
        
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
                f"Sliced-Wasserstein-2 Comparison ({config_label})",
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
                eval_df, group['lsi_gap'], 'lsi_gap', 'LSI Gap Metric (lower = better)',
                f"LSI Gap Metric: Score Alignment ({config_label})",
                os.path.join(plots_dir, f"{plot_idx:02d}_lsi_gap_{config_tag}.png"),
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


# ===========================================================================
# COMPARISON PLOTTING (for cotrain vs indep experiments)
# ===========================================================================

def generate_comparison_visualizations(eval_df_cotrain, eval_df_indep, results_dir):
    """
    Generate 4-way comparison plots dynamically based on available metrics.
    
    This replaces the hardcoded version that expected specific column names.
    """
    plots_dir = os.path.join(results_dir, "comparison_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("\n--> Generating comparison visualization suite...")
    
    # Discover metrics from both dataframes (they should have the same columns)
    metric_groups_cotrain = get_metric_groups(eval_df_cotrain)
    metric_groups_indep = get_metric_groups(eval_df_indep)
    
    # Use union of available configurations
    all_configs = set(metric_groups_cotrain.keys()) | set(metric_groups_indep.keys())
    
    if not all_configs:
        print("--> Warning: No metric columns found. Skipping comparison plots.")
        return
    
    print(f"    Found {len(all_configs)} sampler configurations to compare")
    
    plot_idx = 1
    
    for (method, steps, suffix) in sorted(all_configs):
        config_label = format_config_label(method, steps, suffix)
        config_tag = f"{method}_{steps}{suffix}"
        
        # Get available metrics for this config (from either df)
        metrics_cotrain = metric_groups_cotrain.get((method, steps, suffix), {})
        metrics_indep = metric_groups_indep.get((method, steps, suffix), {})
        all_metrics = set(metrics_cotrain.keys()) | set(metrics_indep.keys())
        
        for metric_type in ['fid', 'kid', 'sw2', 'lsi_gap']:
            if metric_type not in all_metrics:
                continue
            
            metric_col = metrics_cotrain.get(metric_type) or metrics_indep.get(metric_type)
            use_log = (metric_type == 'sw2')
            
            ylabel_map = {
                'fid': 'FID',
                'kid': 'KID', 
                'sw2': 'SW2 (log scale)',
                'lsi_gap': 'LSI Gap (lower=better)',
            }
            
            save_path = os.path.join(plots_dir, f"{plot_idx:02d}_comparison_{metric_type}_{config_tag}.png")
            title = f"Co-trained vs Independent: {metric_type.upper()} ({config_label})"
            
            plot_comparison_metric(
                eval_df_cotrain, eval_df_indep,
                metric_col, ylabel_map[metric_type], title, save_path, use_log
            )
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
    is_grayscale_cifar = info.get("grayscale", False)

    # Build transforms based on dataset
    if dataset_key == "GCIFAR":
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

    print(f"--> Loaded {dataset_key}: {len(train)} train, {len(test)} test, {num_classes} classes, img_size={img_size}")
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

    # Update checkpoint directory to be within results
    cfg["ckpt_dir"] = os.path.join(results_dir, "checkpoints")
    eval_freq = cfg.get("eval_freq", 10)
    eval_freq_cotrain = cfg.get("eval_freq_cotrain", eval_freq)  # ADD THIS
    eval_freq_refine = cfg.get("eval_freq_refine", eval_freq)    # ADD THIS

    device = default_device()
    # --- Discrete diffusion schedule (train + sampling) ---
    # This switches training from continuous log-uniform OU times to discrete VP/DDPM timesteps.


    # --- Discrete OU schedule (optional) ---
    # For use_ddim_times: discretize OU times (log-spaced) and use OU alpha/sigma.
    ou_sched = make_ou_schedule(cfg, device)
    T = int(ou_sched["T"].item())
    noise_sched = ou_sched

    dataset_key = cfg.get("dataset", "FMNIST")
    train_l, test_l, num_classes = make_dataloaders(cfg["batch_size"], cfg["num_workers"], dataset_key)
    cfg["num_classes"] = num_classes  # for CFG label embedding / eval
    
    # Get image size from dataset info (for model initialization)
    img_size = DATASET_INFO.get(dataset_key, {}).get("img_size", 28)
    # Effective size after padding (28->32 for MNIST-family, 32 for GCIFAR)
    effective_img_size = 32 if img_size == 28 else img_size
    cfg["img_size"] = effective_img_size  # store for later use

    # Get FID model (Inception for FMNIST/GCIFAR, LeNet for others)
    fid_model, use_lenet_fid = get_fid_model(dataset_key, train_l, num_classes, device, cfg["ckpt_dir"])

    use_ddim_times = cfg.get("use_ddim_times", False)

    vae = VAE(
        latent_channels=cfg["latent_channels"],
        use_norm=cfg.get("use_latent_norm", False),
        img_size=effective_img_size,
        # NEW: condition only if enabled
        num_classes=(num_classes if cfg.get("use_cond_encoder", False) else None),
        null_label=int(num_classes),  # reserve num_classes as unconditional token
        cond_emb_dim=int(cfg.get("cond_emb_dim", 64)),).to(device)

    # --- Online Models ---
    unet_lsi = UNetModel(in_channels=cfg["latent_channels"], num_classes=num_classes).to(device)
    unet_control = UNetModel(in_channels=cfg["latent_channels"], num_classes=num_classes).to(device)

    if cfg.get("load_from_checkpoint", False):
        ckpt_load_dir = cfg.get("ckpt_load_dir", cfg["ckpt_dir"])
        print(f"--> Loading checkpoints from {ckpt_load_dir}...")
        try:
            vae.load_state_dict(torch.load(os.path.join(ckpt_load_dir, "vae_cotrained.pt"), map_location=device), strict=False)
            print("    Loaded VAE.")
        except Exception as e:
            print(f"    Warning: Could not load VAE ({e})")

        try:
            unet_lsi.load_state_dict(torch.load(os.path.join(ckpt_load_dir, "unet_lsi.pt"), map_location=device), strict=False)
            print("    Loaded UNet LSI.")
        except Exception as e:
            print(f"    Warning: Could not load UNet LSI ({e})")

        try:
            unet_control.load_state_dict(torch.load(os.path.join(ckpt_load_dir, "unet_control.pt"), map_location=device), strict=False)
            print("    Loaded UNet Control.")
        except Exception as e:
            print(f"    Warning: Could not load UNet Control ({e})")


    # --- EMA Models (Score Heads Only) ---
    unet_lsi_ema = UNetModel(in_channels=cfg["latent_channels"], num_classes=num_classes).to(device)
    unet_lsi_ema.load_state_dict(unet_lsi.state_dict())
    unet_lsi_ema.eval()
    for p in unet_lsi_ema.parameters(): p.requires_grad = False

    unet_control_ema = UNetModel(in_channels=cfg["latent_channels"], num_classes=num_classes).to(device)
    unet_control_ema.load_state_dict(unet_control.state_dict())
    unet_control_ema.eval()
    for p in unet_control_ema.parameters(): p.requires_grad = False

    ema_decay = cfg.get("ema_decay", .999)

    # --- Asymmetric LR Settings ---
    score_w_vae = cfg.get("score_w_vae", cfg["score_w"])
    cotrain_head = cfg.get("cotrain_head", "lsi")
    freeze_score_in_cotrain = cfg.get("freeze_score_in_cotrain", False)
    
    if freeze_score_in_cotrain:
        # Independent mode: VAE-only optimizer during cotrain phase
        # Score networks will only be trained during refine phase
        opt_joint = optim.AdamW(vae.parameters(), lr=cfg["lr_vae"], weight_decay=1e-4)
        opt_tracking = None  # No tracking optimizer needed
        print("--> Independent mode: Score networks FROZEN during cotrain phase")
    else:
        # Standard co-training mode
        if cotrain_head == "lsi":
            opt_joint = optim.AdamW([
                {'params': vae.parameters(), 'lr': cfg["lr_vae"]},
                {'params': unet_lsi.parameters(), 'lr': cfg["lr_ldm"]/score_w_vae if score_w_vae > 0 else 0.0},
            ], weight_decay=1e-4)
            opt_tracking = optim.AdamW(unet_control.parameters(), lr=cfg["lr_ldm"], weight_decay=1e-4)
        else:  # cotrain_head == "control"
            opt_joint = optim.AdamW([
                {'params': vae.parameters(), 'lr': cfg["lr_vae"]},
                {'params': unet_control.parameters(), 'lr': cfg["lr_ldm"]/score_w_vae if score_w_vae > 0 else 0.0},
            ], weight_decay=1e-4)
            opt_tracking = optim.AdamW(unet_lsi.parameters(), lr=cfg["lr_ldm"], weight_decay=1e-4)

    lpips_fn = lpips.LPIPS(net='vgg').to(device) if LPIPS_AVAILABLE else None

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
        metrics = {k: 0.0 for k in ["loss", "recon", "kl", "score_lsi", "score_control", "perc", "stiff"]}
        mu_stats = []

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
            mu, logvar = vae.encode(x, y=y_in if use_cond_enc else None)
            logvar = torch.clamp(logvar, min=-30.0, max=20.0)

            # Decode from corrected geometry so all base losses see (mu+mu_r, Sigma+Sigma_r)
            z0 = vae.reparameterize(mu, logvar)
            x_rec = vae.decode(z0)

            if len(mu_stats) < 5: mu_stats.append(mu.detach())

            recon = F.mse_loss(x_rec, x)

            if LPIPS_AVAILABLE:
                x_3c = x.repeat(1, 3, 1, 1)
                x_rec_3c = x_rec.repeat(1, 3, 1, 1)
                perc = lpips_fn(x_rec_3c, x_3c).mean()
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
                kl = torch.mean(logvar.pow(2))

            elif reg_type == "vol":
                # Volume-preserving regularization: -λ·𝔼[log det Σ]
                logvar_clamped = torch.clamp(logvar, min=-10.0)  # σ² ≥ exp(-10) ≈ 4.5e-5
                log_det = torch.sum(logvar_clamped, dim=[1, 2, 3])  # [B] vector
                kl = - torch.mean(log_det)
                
            else:
                raise ValueError(f"Unknown kl_reg_type: {reg_type}")

            t = sample_log_uniform_times(B, cfg["t_min"], cfg["t_max"], device)
            alpha, sigma = get_ou_params(t.view(B,1,1,1))


            if use_ddim_times:
                t_idx = torch.randint(0, T, (B,), device=device, dtype=torch.long)

                # time embedding = actual OU time on the discrete grid
                t = ou_sched["times"].gather(0, t_idx).float()  # [B]

                # alpha/sigma = OU alpha(t), sigma(t) on the same grid
                alpha = extract_schedule(ou_sched["alpha"], t_idx, z0.shape)
                sigma = extract_schedule(ou_sched["sigma"], t_idx, z0.shape)


            noise = torch.randn_like(z0)
            z_t = alpha * z0 + sigma * noise
            
            var_0 = torch.exp(logvar)
            mu_t = alpha * mu
            var_t = (alpha**2) * var_0 + (sigma**2)
            
            # --- Compute both score losses (for logging, even if frozen) ---
            eps_target_lsi = sigma * ((z_t - mu_t) / (var_t + 1e-8))
            
            if freeze_score_in_cotrain:
                # In independent mode, don't compute score predictions during cotrain
                # Just log zeros for score losses
                score_loss_lsi = torch.tensor(0.0, device=device)
                score_loss_control = torch.tensor(0.0, device=device)
            else:
                eps_pred_lsi = unet_lsi(z_t, t, y_in)
                score_loss_lsi = F.mse_loss(eps_pred_lsi, eps_target_lsi)
                
                eps_pred_control = unet_control(z_t, t, y_in)
                score_loss_control = F.mse_loss(eps_pred_control, noise)

            # --- Stiffness penalty ---
            stiff_w = cfg.get("stiff_w", 0.0)
            if stiff_w > 0.0 and not freeze_score_in_cotrain:
                inv_var_t = 1.0 / (var_t + 1e-8)
                stiff_pen = inv_var_t.flatten(1).mean(dim=1).mean()
            else:
                stiff_pen = torch.tensor(0.0, device=device)

            # --- Joint loss ---
            if freeze_score_in_cotrain:
                # Independent mode: VAE-only loss (no score, no stiffness)
                loss_joint = recon + cfg["perc_w"]*perc + cfg["kl_w"]*kl
            else:
                # Co-training mode: include score loss
                score_w_vae = cfg.get("score_w_vae", cfg["score_w"])
                if cotrain_head == "lsi":
                    loss_joint = recon + cfg["perc_w"]*perc + cfg["kl_w"]*kl + score_w_vae*score_loss_lsi + stiff_w*stiff_pen
                else:  # cotrain_head == "control"
                    loss_joint = recon + cfg["perc_w"]*perc + cfg["kl_w"]*kl + score_w_vae*score_loss_control + stiff_w*stiff_pen

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
                if cotrain_head == "lsi":
                    # Control tracks
                    eps_pred_control_tracking = unet_control(z_t_detached, t, y_in)
                    tracking_loss = cfg["score_w"] * F.mse_loss(eps_pred_control_tracking, noise)
                else:
                    # LSI tracks
                    eps_pred_lsi_tracking = unet_lsi(z_t_detached, t, y_in)
                    tracking_loss = cfg["score_w"] * F.mse_loss(eps_pred_lsi_tracking, eps_target_lsi.detach())

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
            metrics["perc"] += perc.item()
            metrics["stiff"] += stiff_pen.item()

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
            "perc": metrics["perc"] / n_batches,
            "stiff": metrics["stiff"] / n_batches,
        }
        loss_records.append(epoch_metrics)

        print(f"Ep {ep+1} | LSI: {epoch_metrics['score_lsi']:.4f} | Ctrl: {epoch_metrics['score_control']:.4f} | "
              f"Rec: {epoch_metrics['recon']:.4f} | KL: {epoch_metrics['kl']:.4f} | Perc: {epoch_metrics['perc']:.4f} | "
              f"Stiff: {epoch_metrics['stiff']:.4f}")

        if len(mu_stats) > 0:
            log_latent_stats("VAE_Train", torch.cat(mu_stats, 0))

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
            )
            if results_lsi is not None:
                results_lsi["epoch"] = ldm_epoch  # LDM epoch for comparison
                results_lsi["stage"] = "cotrain"
                results_lsi["tag"] = "LSI_Diff"
                eval_records.append(results_lsi)

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
            )
            if results_ctrl is not None:
                results_ctrl["epoch"] = ldm_epoch  # LDM epoch for comparison
                results_ctrl["stage"] = "cotrain"
                results_ctrl["tag"] = "Ctrl_Diff"
                eval_records.append(results_ctrl)


    # ===========================================================================
    # REFINEMENT STAGE: Freeze VAE, train only score networks
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

                t = sample_log_uniform_times(B, cfg["t_min"], cfg["t_max"], device)
                alpha, sigma = get_ou_params(t.view(B,1,1,1))


                if use_ddim_times:
                    t_idx = torch.randint(0, T, (B,), device=device, dtype=torch.long)

                    # time embedding = actual OU time on the discrete grid
                    t = ou_sched["times"].gather(0, t_idx).float()  # [B]

                    # alpha/sigma = OU alpha(t), sigma(t) on the same grid
                    alpha = extract_schedule(ou_sched["alpha"], t_idx, z0.shape)
                    sigma = extract_schedule(ou_sched["sigma"], t_idx, z0.shape)
                

                noise = torch.randn_like(z0)
                z_t = alpha * z0 + sigma * noise

                # --- LSI Score Training ---
                var_0 = torch.exp(logvar)
                mu_t = alpha * mu
                var_t = (alpha**2) * var_0 + (sigma**2)

                eps_target_lsi = sigma * ((z_t - mu_t) / (var_t + 1e-8))
                eps_pred_lsi = unet_lsi(z_t, t, y_in)
                score_loss_lsi = F.mse_loss(eps_pred_lsi, eps_target_lsi)

                opt_lsi_refine.zero_grad()
                score_loss_lsi.backward()
                nn.utils.clip_grad_norm_(unet_lsi.parameters(), 1.0)
                opt_lsi_refine.step()

                # --- EMA Update (LSI) ---
                with torch.no_grad():
                    for p_online, p_ema in zip(unet_lsi.parameters(), unet_lsi_ema.parameters()):
                        p_ema.data.mul_(ema_decay).add_(p_online.data, alpha=1 - ema_decay)

                # --- Control (Tweedie/DSM) Score Training ---
                eps_pred_control = unet_control(z_t.detach(), t, y_in)
                score_loss_control = F.mse_loss(eps_pred_control, noise)

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
                "perc": 0.0,
            }
            loss_records.append(epoch_metrics)

            print(f"Refine Ep {ep+1} | LSI: {epoch_metrics['score_lsi']:.4f} | "
                  f"Ctrl: {epoch_metrics['score_control']:.4f}")
            
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
                )
                if results_ctrl is not None:
                    results_ctrl["epoch"] = ldm_epoch
                    results_ctrl["stage"] = "refine"
                    results_ctrl["tag"] = "Ctrl_Diff_Refine"
                    eval_records.append(results_ctrl)
                    
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



def generate_comparison_visualizations(eval_df_cotrain, eval_df_indep, results_dir):
    """
    Generate all comparison plots (4-way: cotrain vs indep, LSI vs Tweedie)
    for all relevant sampling modes.
    """
    plots_dir = os.path.join(results_dir, "comparison_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("\n--> Generating comparison visualization suite...")
    
    # Define all metrics to plot
    metrics = [
        # (metric_col, ylabel, title_suffix, use_log)
        ("fid_rk4_10", "FID", "FID (RK4 10 Steps)", False),
        ("fid_heun_20", "FID", "FID (Heun 20 Steps)", False),
        ("kid_rk4_10", "KID", "KID (RK4 10 Steps)", False),
        ("kid_heun_20", "KID", "KID (Heun 20 Steps)", False),
        ("sw2_rk4_10", "SW2 (log scale)", "SW2 (RK4 10 Steps)", True),
        ("sw2_heun_20", "SW2 (log scale)", "SW2 (Heun 20 Steps)", True),
        ("lsi_gap_rk4_10", "LSI Gap (lower=better)", "LSI Gap Metric (RK4 10 Steps)", False),
        ("lsi_gap_heun_20", "LSI Gap (lower=better)", "LSI Gap Metric (Heun 20 Steps)", False),
    ]
    
    for i, (metric_col, ylabel, title_suffix, use_log) in enumerate(metrics, 1):
        save_path = os.path.join(plots_dir, f"{i:02d}_comparison_{metric_col}.png")
        title = f"Co-trained vs Independent: {title_suffix}"
        plot_comparison_metric(eval_df_cotrain, eval_df_indep, metric_col, ylabel, title, save_path, use_log)
    
    print(f"--> Comparison visualization suite complete ({len(metrics)} plots generated)!")


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

# --- START REPLACEMENT main() ---
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
        "dataset": "GCIFAR",
        "batch_size": 128,
        "num_workers": 2,
        
        # --- Model Architecture ---
        "latent_channels": 5,
        "cond_emb_dim": 64,
        
        # --- Learning Rates ---
        "lr_vae": 5e-4,
        "lr_ldm": 1e-4,
        
        # --- KL and perceptual weights ---
        "kl_w": 1e-4,
        "perc_w": 1.0,
        
        # --- Diffusion Settings ---
        "use_ddim_times": True,
        "t_min": 2e-5,
        "t_max": 2.0,
        "num_train_timesteps": 1000,
        "noise_schedule": "cosine",
        "beta_start": 1e-4,
        "beta_end": 2e-2,
        "cosine_s": 0.008,
        "ddim_eta": 0.0,
        
        # --- CFG ---
        "cfg_label_dropout": 0.25,
        "cfg_eval_scale": 2.0,
        "eval_class_labels": [],
        
        # --- Evaluation & Logging ---
        "use_fixed_eval_banks": True,
        "sw2_n_projections": 1000,
        "ema_decay": 0.999,
        
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
        "epochs_vae": 400,          # Cotrain phase: VAE + LDM joint training
        "epochs_refine": 100,        # Refine phase: LDM-only on frozen VAE
        "lr_refine": 2e-5,
        
        # Co-training specific settings
        "freeze_score_in_cotrain": False,  # Normal co-training
        "cotrain_head": "lsi",
        "use_latent_norm": True,
        "use_cond_encoder": True,
        "kl_reg_type": "norm",
        "score_w_vae": 0.4,
        "stiff_w": 1e-4,
        "score_w": 1.0,
        
        # Eval frequency (eval during both phases)
        "eval_freq_cotrain": 10,    # Eval every 10 epochs during cotrain
        "eval_freq_refine": 10,     # Eval every 10 epochs during refine
    })
    
    # === INDEPENDENT CONFIG ===
    # 50 VAE-only epochs (no eval) + 250 refine epochs = 250 LDM epochs total
    cfg_indep = cfg_shared.copy()
    cfg_indep.update({
        # Training schedule
        "epochs_vae": 50,           # VAE-only pretraining (no LDM)
        "epochs_refine": 500,       # LDM training on frozen VAE
        "lr_refine": 1e-4,
        
        # Independent mode settings
        "freeze_score_in_cotrain": True,   # Freeze score nets during VAE training
        "score_w_vae": 0.0,                # No score gradient (redundant but explicit)
        "stiff_w": 0.0,                    # No stiffness penalty
        "use_latent_norm": False,          # Standard VAE (no GroupNorm on mu)
        "use_cond_encoder": False,         # No conditional encoder
        "kl_reg_type": "normal",           # Standard KL to N(0,I)
        "cotrain_head": "lsi",             # Doesn't matter when frozen
        "score_w": 1.0,
        
        # Eval frequency (no eval during VAE phase, eval during refine)
        "eval_freq_cotrain": 999999,  # Effectively never (VAE phase has no LDM)
        "eval_freq_refine": 10,       # Eval every 10 epochs during refine
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

