# =================== nd AVRC training + sampling (+ standard Rectified Flow) ===================
# Includes:
#   • PyTorch-native OT-Flow-style nd samplers
#   • AVRC (x1-only) trainer in R^D
#   • Standard Rectified Flow (RF) baseline trainer in R^D
#   • nd metrics (MMD, sliced-W2), utilities, and samplers
#
# Changelog (diagnostics):
#   • sliced_w2 now supports max_n subsampling for faster diagnostics
#   • (unchanged otherwise; logging/timing added in the viz/main cell)

import math, time, os
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------- device & dtype --------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TDTYPE = torch.float32


def _square_limits(xlim, ylim, c_exp = 1.15):
    """
    Make xlim, ylim square in data coordinates.
    We keep the centers the same, just expand the smaller span.
    """
    x0, x1 = float(xlim[0]), float(xlim[1])
    y0, y1 = float(ylim[0]), float(ylim[1])
    xc = 0.5 * (x0 + x1)
    yc = 0.5 * (y0 + y1)
    w = x1 - x0
    h = y1 - y0
    side = c_exp * max(w, h)
    xlim_sq = (xc - 0.5*side, xc + 0.5*side)
    ylim_sq = (yc - 0.5*side, yc + 0.5*side)
    return list(xlim_sq), list(ylim_sq)

# --------------------------------- random utils ---------------------------------------
def seed_everything(seed: int | None):
    if seed is None: return
    torch.manual_seed(seed); np.random.seed(seed)

# -------------------------------- time embedding --------------------------------------
def t_embed(t: torch.Tensor):
    """Simple 4-dim time features; t in [0,1], returns (B,4)."""
    return torch.cat([t, torch.sin(2*math.pi*t), torch.cos(2*math.pi*t), t*t], dim=1)

# ---------------------------------- MLP blocks ----------------------------------------
class MLP(nn.Module):
    def __init__(self, din, dout, hidden=128, depth=4, act=nn.SiLU):
        super().__init__()
        layers = []
        d = din
        for _ in range(depth-1):
            layers += [nn.Linear(d, hidden), act()]
            d = hidden
        layers += [nn.Linear(d, dout)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)

# ------------------------------ Enhanced MLP (RF) ------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding: maps t∈[0,1] (B,1) to (B,dim)."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        # frequencies: exponential range as in transformer pos-emb
        half = self.dim // 2
        exp_range = torch.arange(half).float()
        # register as buffer to keep on same device/dtype
        self.register_buffer("freqs", torch.exp(-math.log(10000.0) * exp_range / max(1, half)), persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,1) in [0,1]
        angles = t * self.freqs.view(1, -1)
        emb_sin = torch.sin(angles)
        emb_cos = torch.cos(angles)
        emb = torch.cat([emb_sin, emb_cos], dim=1)
        if emb.size(1) < self.dim:
            # pad to exact dim if odd
            emb = F.pad(emb, (0, self.dim - emb.size(1)))
        return emb

class EnhancedMLP(nn.Module):
    """
    Enhanced MLP for rectified flow velocity fields with:
    - sinusoidal time embedding
    - FiLM-style modulation per block (scale/shift from time embedding)
    - residual skip connections
    - zero-init final layer for stable training
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden: int = 256,
        depth: int = 6,
        time_embed_dim: int = 128,
        act: type[nn.Module] = nn.SiLU,
        dropout: float = 0.0,
        final_zero_init: bool = True,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden = int(hidden)
        self.depth = int(max(1, depth))
        self.act = act()
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.use_layer_norm = bool(use_layer_norm)

        # time embedding -> hidden
        self.time_emb = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        # per-block FiLM (scale, shift)
        self.film = nn.ModuleList([nn.Linear(hidden, 2*hidden) for _ in range(self.depth)])

        # input projection
        self.input_proj = nn.Linear(self.input_dim, hidden)
        # blocks
        self.blocks = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(self.depth)])
        self.norms  = nn.ModuleList([nn.LayerNorm(hidden) if self.use_layer_norm else nn.Identity() for _ in range(self.depth)])

        # output head
        self.output_proj = nn.Linear(hidden, self.output_dim)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        if final_zero_init:
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, D), t: (B,1)
        h = self.input_proj(x)
        # shared time embedding
        te = self.time_mlp(self.time_emb(t))
        for i in range(self.depth):
            res = h
            h = self.blocks[i](h)
            h = self.norms[i](h)
            # FiLM modulation
            s, b = self.film[i](te).chunk(2, dim=1)
            h = h * (1.0 + s) + b
            h = self.act(h)
            h = self.dropout(h)
            h = h + res
        return self.output_proj(h)

# ------------------------------ Encoders / heads (D-dim) ------------------------------
class GaussianHead(nn.Module):
    """
    Outputs [mu (D), logvar (D)] with last layer zeroed so initial mu=0, logvar=0.
    """
    def __init__(self, din: int, D: int, hidden=128, depth=4):
        super().__init__()
        self.D = D
        self.mlp = MLP(din, 2*D, hidden=hidden, depth=depth)
        # force constant zero init
        for m in reversed(list(self.mlp.modules())):
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias); break

    def forward(self, x):
        h = self.mlp(x)
        mu, logvar = h[:, :self.D], h[:, self.D:]
        return mu, logvar

def reparam(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + std*eps

def kl_normal_diag(mu, logvar):
    """
    E_x [ KL( N(mu, diag(exp(logvar))) || N(0,I) ) ] averaged over batch.
    """
    elem = mu.pow(2) + logvar.exp() - logvar - 1.0
    return 0.5 * torch.mean(elem.sum(dim=1))

class EncoderX1Only(nn.Module):
    """z0 = E(x1) (destination-only encoder) in D dims."""
    def __init__(self, D=2, hidden=128, depth=4):
        super().__init__()
        self.D = D
        self.head = GaussianHead(D, D, hidden=hidden, depth=depth)
    def forward(self, x1): return self.head(x1)

# --------------------- Enhanced Encoder (Pro) and components ---------------------
class RFF(nn.Module):
    def __init__(self, D, n_freq=0, freq_scale=1.0, trainable=False):
        super().__init__()
        self.n_freq = int(n_freq)
        if self.n_freq > 0:
            B = torch.randn(self.n_freq, D) * float(freq_scale)
            self.register_parameter("B", nn.Parameter(B, requires_grad=trainable))
        else:
            self.register_parameter("B", None)

    def forward(self, x):
        if self.n_freq <= 0:
            return x
        proj = x @ self.B.T
        s, c = torch.sin(proj), torch.cos(proj)
        return torch.cat([x, s, c], dim=1)

class GeGLU(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W = nn.Linear(d_in, 2*d_out)
    def forward(self, x):
        a, g = self.W(x).chunk(2, dim=-1)
        return a * F.gelu(g)

class ResBlock(nn.Module):
    def __init__(self, d, hidden, dropout=0.0, act=nn.SiLU, use_geglu=True):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.ff1  = (GeGLU(d, hidden) if use_geglu else nn.Sequential(
                        nn.Linear(d, hidden), act()))
        self.drop = nn.Dropout(dropout)
        self.ff2  = nn.Linear(hidden, d)
        nn.init.zeros_(self.ff2.bias)
        nn.init.normal_(self.ff2.weight, mean=0.0, std=1e-3)

    def forward(self, x):
        h = self.norm(x)
        h = self.ff1(h)
        h = self.drop(h)
        h = self.ff2(h)
        return x + h

class OrthoMix(nn.Module):
    def __init__(self, D, n_householder=4):
        super().__init__()
        self.v = nn.Parameter(torch.randn(n_householder, D))

    def forward(self, x):
        Qx = x
        for v in self.v:
            v = F.normalize(v, dim=0, eps=1e-12)
            Qx = Qx - 2.0 * (Qx @ v.unsqueeze(1)) * v.unsqueeze(0)
        return Qx

class GaussianHeadPro(nn.Module):
    def __init__(self, D, width=512, depth=8, dropout=0.0, ff_freqs=16,
                 ff_scale=1.0, use_geglu=True, residual_mu=True,
                 ortho_mix=False, n_householder=4,
                 logvar_min=-10.0, logvar_max=+6.0):
        super().__init__()
        self.D = int(D)
        self.residual_mu = bool(residual_mu)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

        self.rff = RFF(D, n_freq=ff_freqs, freq_scale=ff_scale, trainable=False)
        feat_dim = D + 2*ff_freqs

        self.mixer = OrthoMix(D, n_householder=n_householder) if ortho_mix else None

        blocks = [nn.Linear(feat_dim, width), nn.SiLU()]
        for _ in range(depth):
            blocks.append(ResBlock(width, width, dropout=dropout, use_geglu=use_geglu))
        self.trunk = nn.Sequential(*blocks)

        self.mu_head     = nn.Linear(width, D)
        self.logvar_head = nn.Linear(width, D)
        for m in (self.mu_head, self.logvar_head):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

        self.alpha = nn.Parameter(torch.tensor(0.0)) if self.residual_mu else None

    def forward(self, x):
        xin = self.mixer(x) if (self.mixer is not None) else x
        h   = self.rff(xin)
        h   = self.trunk(h)

        dmu = self.mu_head(h)
        if self.residual_mu and self.alpha is not None:
            mu = x + self.alpha.tanh() * dmu
        else:
            mu = dmu

        logv = self.logvar_head(h)
        logv = torch.clamp(logv, self.logvar_min, self.logvar_max)
        return mu, logv

class EncoderX1OnlyPro(nn.Module):
    def __init__(self, D=2, width=512, depth=8, dropout=0.05,
                 ff_freqs=16, ff_scale=1.0, use_geglu=True,
                 residual_mu=False,       # <--- default to classic VAE-style init
                 ortho_mix=False, n_householder=4,
                 logvar_min=-10.0, logvar_max=+6.0):

        super().__init__()
        self.D = int(D)
        self.head = GaussianHeadPro(
            D, width=width, depth=depth, dropout=dropout,
            ff_freqs=ff_freqs, ff_scale=ff_scale, use_geglu=use_geglu,
            residual_mu=residual_mu, ortho_mix=ortho_mix, n_householder=n_householder,
            logvar_min=logvar_min, logvar_max=logvar_max
        )

    def forward(self, x1):
        return self.head(x1)

class Velocity(nn.Module):
    """v(x,t) -> R^D (mean-field velocity in data space)."""
    def __init__(self, D=2, hidden=128, depth=4):
        super().__init__()
        self.D = D
        self.net = EnhancedMLP(input_dim=D, output_dim=D, hidden=hidden, depth=depth)
    def forward(self, x, t): return self.net(x, t)

# -------------------------------- Schedules / frames ----------------------------------
def lin_sched(a, b, step, total):
    s = min(max(step / max(total, 1), 0.0), 1.0)
    return a + (b - a) * s

def _make_log_stride_rounds(total_rounds: int,
                            first_dense: int = 20,
                            growth: float = 2.0,
                            stride0: int = 1,
                            max_stride: int | None = None) -> list[int]:
    frames: list[int] = []
    start = 1
    L = max(1, int(first_dense))
    stride = max(1, int(stride0))
    while start <= total_rounds:
        end = min(total_rounds, start + L - 1)
        use_stride = min(stride, max_stride) if (max_stride is not None) else stride
        frames.extend(range(start, end + 1, use_stride))
        start = end + 1
        L = max(1, int(round(L * growth)))
        stride = max(1, int(round(stride * growth)))
    return sorted(set(frames))


# ------------------------------ Projection helpers (NEW) ------------------------------
@torch.no_grad()
def _orthonormalize_cols(A: torch.Tensor) -> torch.Tensor:
    # A: (D, k); returns Q with orthonormal columns (thin Gram-Schmidt)
    Q = []
    for i in range(A.size(1)):
        v = A[:, i]
        for j in range(len(Q)):
            v = v - (Q[j].T @ v) * Q[j]
        v = v / (v.norm() + 1e-12)
        Q.append(v)
    return torch.stack(Q, dim=1)

@torch.no_grad()
def choose_projection_pairs(
    D: int,
    *,
    data_for_pca: torch.Tensor | None = None,   # (N, D) or None
    num_pairs: int = 3,
    mode: str = "pca",                           # {"pca","random"}
    seed: int | None = 1234
) -> list[torch.Tensor]:
    """
    Returns a list of P_i ∈ R^{D×2}. If mode='pca' and data is given,
    uses leading 2,4,6,... components as pairs. Otherwise random pairs.
    """
    pairs: list[torch.Tensor] = []
    if mode == "pca" and (data_for_pca is not None) and (data_for_pca.ndim == 2) and (data_for_pca.size(1) == D):
        X = data_for_pca - data_for_pca.mean(dim=0, keepdim=True)
        # torch.pca_lowrank is efficient and differentiable-free
        q = min(D, 2 * num_pairs)
        U, S, V = torch.pca_lowrank(X, q=q, center=False)  # V: (D, q)
        for i in range(num_pairs):
            a = 2*i
            b = 2*i + 1
            if b >= V.size(1): break
            P = V[:, a:b+1]                      # (D, 2)
            P = _orthonormalize_cols(P)          # be safe
            pairs.append(P.to(device=device, dtype=TDTYPE))
    else:
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed))
        else:
            gen = None
        for _ in range(num_pairs):
            R = torch.randn(D, 2, device=device, dtype=TDTYPE, generator=gen)
            P = _orthonormalize_cols(R)
            pairs.append(P)
    return pairs

@torch.no_grad()
def project_nd(X: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """
    X: (N, D), P: (D, 2) with orthonormal columns.
    Returns (N, 2).
    """
    return X @ P

# ===================== EMBEDDING: globals & helpers =====================
EMBEDDING_MODE = "identity"   # {"identity","linear","sine_wiggle","rff","radial"}
EMBEDDING_DIM  = 2            # target ambient dimension K after embedding

def set_embedding(mode: str = "identity", dim: int = 2):
    """Set global embedding mode and ambient dimension K."""
    global EMBEDDING_MODE, EMBEDDING_DIM
    EMBEDDING_MODE = str(mode).lower().strip()
    EMBEDDING_DIM  = int(dim)

# ---------- linear-algebra helpers ----------
def _qr_tall(k: int, m: int, *, device, dtype):
    """
    Return Q in R^{k×m} with orthonormal columns (k >= m).
    """
    assert k >= m and k > 0 and m > 0
    G = torch.randn(k, m, device=device, dtype=dtype)
    # reduced QR gives Q:(k×m), R:(m×m)
    Q, _ = torch.linalg.qr(G, mode='reduced')
    return Q  # (k,m)

def _qr_wide(m: int, k: int, *, device, dtype):
    """
    Return A in R^{m×k} with orthonormal columns (k <= m).
    """
    assert m >= k and k > 0 and m > 0
    G = torch.randn(m, k, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(G, mode='reduced')  # Q:(m,k), columns orthonormal
    return Q  # (m,k)

def _orthonormal_complement(Q: torch.Tensor, k: int, s: int, *, device, dtype):
    """
    Given Q (k×m) with orthonormal cols, return Q_perp (k×s) with orthonormal cols and Q^T Q_perp = 0.
    """
    if s <= 0:
        return None
    m = Q.shape[1]
    assert Q.shape == (k, m)
    G = torch.randn(k, s, device=device, dtype=dtype)
    G = G - Q @ (Q.transpose(0,1) @ G)     # project out span(Q)
    Qp, _ = torch.linalg.qr(G, mode='reduced')  # (k,s)
    return Qp[:, :s]

# ---------- embedding kernels ----------
@torch.no_grad()
def _embed_linear(X: torch.Tensor, k_dim: int) -> torch.Tensor:
    """
    Linear lift/reduce into R^{k_dim}. X: (n,m). Returns Z: (n,k_dim)
    """
    device, dtype = X.device, X.dtype
    n, m = X.shape
    if k_dim > m:
        Q = _qr_tall(k_dim, m, device=device, dtype=dtype)  # (k,m)
        Z = X @ Q.transpose(0,1)                            # (n,k)
    else:
        Z = X[:, :k_dim]
    return Z

@torch.no_grad()
def _embed_sine_wiggle(X: torch.Tensor, k_dim: int, amp: float = 0.25, freq: float = 1.0) -> torch.Tensor:
    """
    Linear lift plus small sinusoidal wiggle in orthogonal directions when k_dim>m.
    If k_dim<m, reduce then add intra-subspace sinusoid.
    """
    device, dtype = X.device, X.dtype
    n, m = X.shape

    if k_dim >= m:
        # base linear lift
        Q = _qr_tall(k_dim, m, device=device, dtype=dtype)      # (k,m)
        Zlin = X @ Q.transpose(0,1)                              # (n,k)
        # wiggle in orthogonal complement
        s = min(m, max(1, k_dim - m))                            # at least 1 if we truly lifted
        Q_perp = _orthonormal_complement(Q, k_dim, s, device=device, dtype=dtype)
        if Q_perp is not None and s > 0:
            R = torch.randn(s, m, device=device, dtype=dtype) * freq
            b = 2.0*math.pi*torch.rand(s, device=device, dtype=dtype)
            H = torch.sin(X @ R.transpose(0,1) + b)              # (n,s)
            Zwig = H @ Q_perp.transpose(0,1)                     # (n,k)
            Z = Zlin + amp * Zwig
        else:
            Z = Zlin
    else:
        # reduce then wiggle within the k-dim subspace
        A = _qr_wide(m, k_dim, device=device, dtype=dtype)       # (m,k)
        Zlin = X @ A                                             # (n,k)
        R = torch.randn(k_dim, m, device=device, dtype=dtype) * freq
        b = 2.0*math.pi*torch.rand(k_dim, device=device, dtype=dtype)
        H = torch.sin(X @ R.transpose(0,1) + b)                  # (n,k)
        Z = Zlin + amp * H

    return Z

@torch.no_grad()
def _embed_rff(X: torch.Tensor, k_dim: int, freq: float = 1.0) -> torch.Tensor:
    """
    Random Fourier Features lift into R^{k_dim}.
    """
    device, dtype = X.device, X.dtype
    n, m = X.shape
    M = max(1, k_dim // 2)
    W = torch.randn(M, m, device=device, dtype=dtype) * freq     # (M,m)
    b = 2.0*math.pi*torch.rand(M, device=device, dtype=dtype)    # (M,)

    C = torch.cos(X @ W.transpose(0,1) + b)  # (n,M)
    S = torch.sin(X @ W.transpose(0,1) + b)  # (n,M)
    Z = torch.cat([C, S], dim=1)             # (n,2M)

    if Z.shape[1] < k_dim:
        pad = X @ torch.randn(m, k_dim - Z.shape[1], device=device, dtype=dtype)  # linear pad
        Z = torch.cat([Z, pad], dim=1)
    else:
        Z = Z[:, :k_dim]
    return Z

@torch.no_grad()
def _embed_radial(X: torch.Tensor, k_dim: int) -> torch.Tensor:
    """
    Radial feature expansion (r, r^2, sin r, cos r, plus quadratic pads as needed).
    """
    device, dtype = X.device, X.dtype
    n, m = X.shape
    r = torch.linalg.norm(X, dim=1, keepdim=True)  # (n,1)
    feats = [X, r, r**2, torch.sin(r), torch.cos(r)]
    Z = torch.cat(feats, dim=1)  # (n, m+4)

    # pad with squared random linear forms if needed
    while Z.shape[1] < k_dim:
        w = torch.randn(m, 1, device=device, dtype=dtype)
        Z = torch.cat([Z, (X @ w)**2], dim=1)
    Z = Z[:, :k_dim]
    return Z

@torch.no_grad()
def embed_to_k(X2d: torch.Tensor, mode: str, k_dim: int) -> torch.Tensor:
    """
    Dispatch entry: X2d (n,2) --> Z (n,k_dim) according to mode.
    """
    mode = str(mode).lower().strip()
    if k_dim == 2 and mode in ("identity", "", None):
        return X2d

    if mode in ("identity", "linear"):
        return _embed_linear(X2d, k_dim)
    elif mode == "sine_wiggle":
        return _embed_sine_wiggle(X2d, k_dim, amp=0.25, freq=1.0)
    elif mode == "rff":
        return _embed_rff(X2d, k_dim, freq=1.0)
    elif mode == "radial":
        return _embed_radial(X2d, k_dim)
    else:
        raise ValueError(f"unknown embedding mode: {mode}")



# --------------------------------- nd OT samplers -------------------------------------
# All return torch.Tensor [n,2] on current device/dtype.
@torch.no_grad()
def sample_two_moons(n, gap=0.5, rad=1.0, noise=0.08):
    n1 = n//2; n2 = n - n1
    u1 = torch.rand(n1, device=device, dtype=TDTYPE) * math.pi
    u2 = torch.rand(n2, device=device, dtype=TDTYPE) * math.pi
    x1 = torch.stack([rad*torch.cos(u1), rad*torch.sin(u1)], dim=1)
    x2 = torch.stack([rad*(1.0-torch.cos(u2)), -rad*torch.sin(u2)-gap], dim=1)
    X  = torch.cat([x1, x2], dim=0)
    if noise>0: X = X + noise*torch.randn_like(X)
    return X

@torch.no_grad()
def sample_spiral(n, a=0.5, b=0.25, tmin=0.0, tmax=4.0*math.pi, noise=0.08):
    t = tmin + (tmax - tmin)*torch.rand(n, device=device, dtype=TDTYPE)
    r = a + b*t
    X = torch.stack([r*torch.cos(t), r*torch.sin(t)], dim=1)
    if noise>0: X = X + noise*torch.randn_like(X)
    return X

@torch.no_grad()
def sample_rings(n, radii=(1.0, 2.0, 3.0), sigma_r=0.08):
    radii = torch.tensor(radii, device=device, dtype=TDTYPE)
    idx = torch.randint(0, radii.numel(), (n,), device=device)
    R = radii[idx] + sigma_r*torch.randn(n, device=device, dtype=TDTYPE)
    th = 2*math.pi*torch.rand(n, device=device, dtype=TDTYPE)
    X  = torch.stack([R*torch.cos(th), R*torch.sin(th)], dim=1)
    return X

@torch.no_grad()
def sample_checker_grid(n, cells=4, fill_frac=0.98, jitter=0.01, span=4.0):
    i = torch.randint(0, cells, (n,), device=device)
    j = torch.randint(0, cells, (n,), device=device)
    parity = (i + j) % 2
    j = (j + parity) % cells  # shift into even parity
    cell = span / float(cells)
    centers = torch.stack([i, j], dim=1).to(TDTYPE) + 0.5
    U = (torch.rand(n, 2, device=device, dtype=TDTYPE) - 0.5) * (fill_frac * cell)
    X = (-span/2.0) + centers * cell + U
    if jitter>0: X = X + jitter*torch.randn_like(X)
    return X

@torch.no_grad()
def sample_checker_stripes(n, span=4.0, noise=0.08):
    x1 = (torch.rand(n, device=device, dtype=TDTYPE) - 0.5) * span
    x2 = (torch.rand(n, device=device, dtype=TDTYPE) - 0.5) * span
    x2 = x2 + ((torch.floor(x1) % 2) * (span/4.0))
    X  = torch.stack([x1, x2], dim=1)
    if noise>0: X = X + noise*torch.randn_like(X)
    return X

@torch.no_grad()
def sample_pinwheel(n, radial_std=0.25, tangential_std=0.05, n_arms=5, rate=0.25):
    k = torch.randint(0, n_arms, (n,), device=device)
    r = radial_std*torch.randn(n, device=device, dtype=TDTYPE) + 1.0
    base = k.to(TDTYPE) * (2.0*math.pi/n_arms) + rate*torch.randn(n, device=device, dtype=TDTYPE)
    X = torch.stack([r*torch.cos(base), r*torch.sin(base)], dim=1)
    noise = torch.randn(n, 2, device=device, dtype=TDTYPE)
    c, s = torch.cos(base), torch.sin(base)
    R = torch.stack([torch.stack([c, -s], dim=1),
                     torch.stack([s,  c], dim=1)], dim=1)  # (n,2,2)
    tang = torch.einsum('nij,nj->ni', R, noise) * tangential_std
    return X + tang

@torch.no_grad()
def sample_scurve(n, tmin=-math.pi, tmax=math.pi, noise=0.08):
    t = tmin + (tmax - tmin)*torch.rand(n, device=device, dtype=TDTYPE)
    x = t
    y = torch.sin(t) + 0.25*torch.sin(3.0*t)
    X = torch.stack([x, y], dim=1)
    if noise>0: X = X + noise*torch.randn_like(X)
    return X

@torch.no_grad()
def sample_eight_gaussians(n, radius=4.0, std=0.10, weights=None):
    ang = torch.linspace(0.0, 2.0*math.pi, 9, device=device, dtype=TDTYPE)[:-1]
    means = torch.stack([radius*torch.cos(ang), radius*torch.sin(ang)], dim=1)  # (8,2)
    if weights is None:
        w = torch.full((8,), 1/8, device=device, dtype=TDTYPE)
    else:
        w = torch.tensor(weights, device=device, dtype=TDTYPE)
        w = w / (w.sum() + 1e-12)
    comp = torch.multinomial(w, num_samples=n, replacement=True)  # (n,)
    mu = means[comp]                                             # (n,2)
    return mu + std*torch.randn(n, 2, device=device, dtype=TDTYPE)


# --------------------------- NEW: line target ---------------------------
@torch.no_grad()
def sample_line(
    n: int,
    *,
    length: float = 6.0,     # total extent along x, centered at 0: x ~ U[-length/2, length/2]
    fuzz_y: float = 0.02,    # tiny vertical noise -> almost all variance in x
    fuzz_x: float = 0.0,     # usually 0; set small (e.g. 0.01) if you want to break exact uniformity
):
    # uniform x on a long line
    x = (torch.rand(n, 1, device=device, dtype=TDTYPE) - 0.5) * length
    # tiny vertical fuzz
    y = fuzz_y * torch.randn(n, 1, device=device, dtype=TDTYPE)

    if fuzz_x > 0.0:
        x = x + fuzz_x * torch.randn_like(x)
    return torch.cat([x, y], dim=1)


def sample_rose_knot(
    n,
    k: int = 9,                 # number of petals (odd gives k petals; even gives 2k)
    R: float = 1.25,            # base radius (≈ Gaussian scale)
    alpha: float = 0.6,         # petal amplitude (0<alpha<1)
    turns: float = 2.0,         # how many full wraps around the origin
    noise: float = 0.06,        # overall noise magnitude
    aniso: float = 2.0          # tangential vs radial noise ratio (>1 => thinner petals)
):
    """
    Adversarial 'rose-knot' distribution to maximize independent-coupling crossings.
    Polar param: r(θ) = R * (1 + alpha * cos(k θ)).
    θ ~ Uniform[0, 2π * turns], then add anisotropic (tangent-heavy) noise.

    Typical radius range: R*(1-alpha) .. R*(1+alpha).
    Defaults give ~0.5 .. ~2.0, i.e., near standard Gaussian scale.
    """
    # angles across multiple wraps
    theta = (2.0 * math.pi * turns) * torch.rand(n, device=device, dtype=TDTYPE)

    # rose radius
    r = R * (1.0 + alpha * torch.cos(k * theta))

    # base points on the curve
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    base = torch.stack([x, y], dim=1)

    # unit radial & tangential directions
    ur = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)            # (n,2)
    ut = torch.stack([-torch.sin(theta), torch.cos(theta)], dim=1)           # (n,2)

    # anisotropic noise: thin in radial, fatter along tangent
    radial_std = noise
    tang_std   = noise * aniso
    eps_r = radial_std * torch.randn(n, 1, device=device, dtype=TDTYPE)
    eps_t = tang_std   * torch.randn(n, 1, device=device, dtype=TDTYPE)

    X = base + eps_r * ur + eps_t * ut
    return X


# --------------------------- NEW: Sierpiński triangle target ---------------------------
@torch.no_grad()
def sample_sierpinski(
    n: int,
    *,
    burn_in: int = 20,      # mixing steps (contractive, so this is plenty)
    iters: int = 20,        # additional iterations; final state is sampled
    scale: float = 2.8,     # centers to origin and scales to ~Gaussian radius
    noise: float = 0.03,    # small jitter to thicken filaments
):
    """
    Sierpiński triangle via a 3-map IFS:
      f_i(x) = 0.5*(x + v_i), i in {1,2,3}, with equilateral-triangle vertices.
    We take 'burn_in + iters' contractive steps in batch, then center & scale.

    The default (scale≈2.8) puts the outer radius ~1.6, i.e., close to N(0,I) scale.
    """
    # Equilateral triangle vertices (unit side)
    v = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, math.sqrt(3.0)/2.0],
    ], device=device, dtype=TDTYPE)

    # Start anywhere (zeros is fine; contraction kills init quickly)
    x = torch.zeros(n, 2, device=device, dtype=TDTYPE)

    steps = burn_in + iters
    for _ in range(steps):
        idx = torch.randint(0, 3, (n,), device=device)
        x = 0.5 * (x + v[idx])

    # Center at triangle centroid and scale to ~Gaussian-ish spread
    centroid = torch.tensor([0.5, math.sqrt(3.0)/6.0], device=device, dtype=TDTYPE)
    x = (x - centroid) * scale

    # Thin isotropic noise to avoid degenerate filaments
    if noise > 0:
        x = x + noise * torch.randn_like(x)

    return x



# ---------------------------- TARGET toggle & source ----------------------------
TARGET = "moons"  # {"moons","spiral","rings","checker",...,"sierpinski"}

def set_target(name: str):
    global TARGET
    TARGET = str(name).lower().strip()

def sample_source_torch(n, D=2):
    return torch.randn(n, D, device=device, dtype=TDTYPE)

# ---- base 2-D target (unchanged) ----
def _sample_target_2d(n):
    key = TARGET
    if key in ("moons", "two_moons", "two-moons"):      return sample_two_moons(n)
    if key == "spiral":                                 return sample_spiral(n)
    if key in ("rings","concentric"):                   return sample_rings(n)
    if key in ("checker","checkerboard","checker_grid"):return sample_checker_grid(n)
    if key in ("checker_stripes","checker-legacy"):     return sample_checker_stripes(n)
    if key == "pinwheel":                               return sample_pinwheel(n)
    if key in ("scurve","s-curve","s_curve"):           return sample_scurve(n)
    if key in ("8g","8gaussians","eight_gaussians"):    return sample_eight_gaussians(n)
    if key in ("rose_knot","rose","flower"):            return sample_rose_knot(n)
    if key in ("sierpinski","gasket","tri_gasket"):     return sample_sierpinski(n)
    if key in ("line","long_line","anisotropic_line"):   return sample_line(n)
    return sample_two_moons(n)

@torch.no_grad()
def sample_target_torch(n, *, embedding_mode: str = None, k_dim: int = None):
    """
    Sample the intrinsic 2-D target, then embed to R^{k_dim}.
    If embedding_mode/k_dim are None, use globals set by set_embedding(...).
    Returns torch.Tensor [n, k_dim].
    """
    mode = EMBEDDING_MODE if embedding_mode is None else embedding_mode
    K    = EMBEDDING_DIM  if k_dim is None          else int(k_dim)

    X2 = _sample_target_2d(n)  # (n,2) on current device/dtype
    if K == 2 and (mode in ("identity","",None)):
        return X2
    return embed_to_k(X2, mode=mode, k_dim=K)

def make_pairs_random(n):
    """
    Return (x0, x1, None) where x0 ~ N(0, I_K), x1 is embedded target in R^K.
    """
    K = EMBEDDING_DIM
    x0 = sample_source_torch(n, D=K)
    x1 = sample_target_torch(n)     # uses global EMBEDDING_MODE/DIM
    return x0, x1, None



# --------------------------------- Metrics (nd) ---------------------------------------
@torch.no_grad()
def mmd_rbf_nd(x: torch.Tensor, y: torch.Tensor, sigma=None, max_n:int = 8192):
    """
    Unbiased MMD with Gaussian kernel in R^d. Subsamples to avoid OOM.
    Returns scalar float.
    """
    x = x.reshape(x.size(0), -1); y = y.reshape(y.size(0), -1)
    if x.size(0) > max_n: x = x[torch.randint(0, x.size(0), (max_n,), device=x.device)]
    if y.size(0) > max_n: y = y[torch.randint(0, y.size(0), (max_n,), device=y.device)]
    n, m = x.size(0), y.size(0)

    if sigma is None:
        take = min(3000, n + m)
        xy = torch.cat([x, y], dim=0)
        sel = torch.randint(0, xy.size(0), (take,), device=xy.device)
        pd = torch.cdist(xy[sel], xy[sel], p=2)
        sigma = torch.median(pd[pd>0]).clamp(min=1e-4)
    gamma = 1.0 / (2.0 * sigma**2)

    Kxx = torch.exp(-gamma * torch.cdist(x, x, p=2).pow(2))
    Kyy = torch.exp(-gamma * torch.cdist(y, y, p=2).pow(2))
    Kxy = torch.exp(-gamma * torch.cdist(x, y, p=2).pow(2))
    mmd2 = (Kxx.sum() - torch.diagonal(Kxx).sum())/(n*(n-1) + 1e-12) \
         + (Kyy.sum() - torch.diagonal(Kyy).sum())/(m*(m-1) + 1e-12) \
         - 2.0 * Kxy.mean()
    return float(mmd2.clamp(min=0).sqrt().detach().cpu())

@torch.no_grad()
def sliced_w2(x: torch.Tensor, y: torch.Tensor, L: int = 128, max_n: int | None = None):
    """
    Sliced Wasserstein-2: average 1D W2 over random projections u ~ Unif(S^{d-1}).
    If max_n is provided, subsample both x and y to at most max_n points.
    """
    x = x.reshape(x.size(0), -1); y = y.reshape(y.size(0), -1)
    n = min(x.size(0), y.size(0))
    if (max_n is not None) and (n > max_n):
        idx = torch.randperm(n, device=x.device)[:max_n]
        x = x[idx]; y = y[idx]
        n = max_n
    else:
        x = x[:n]; y = y[:n]
    d = x.size(1)
    u = torch.randn(L, d, device=x.device, dtype=x.dtype)
    u = u / (u.norm(dim=1, keepdim=True) + 1e-12)
    xs = (x @ u.T).sort(dim=0).values     # (n,L)
    ys = (y @ u.T).sort(dim=0).values
    w2_per = torch.mean((xs - ys).pow(2), dim=0).sqrt()  # (L,)
    return float(w2_per.mean().detach().cpu())


# ---------------------------- 3D crossings (t on X-axis) ----------------------------
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _set_pane_color(ax, axis: str, rgba):
    try:
        getattr(ax, f"{axis}axis").pane.set_facecolor(rgba)  # mpl ≥ 3.7
    except Exception:
        getattr(ax, f"w_{axis}axis").set_pane_color(rgba)    # back-compat
        getattr(ax, f"w_{axis}axis").set_pane_color(rgba)



# -------------------- Crossings plot via projection (replaces *_nd) --------------------
@torch.no_grad()
def plot_crossings_proj(
    P: torch.Tensor,                                 # (D,2) projection
    model_or_sampler=None, *,
    pairs_mode="encoder",
    n_pairs=60_000,
    subset_lines=160,
    subset_strategy="pca",
    line_indices=None,
    plane_mode="scatter",
    bins=220,
    midplane=False, mid_t=0.5,
    mid_bins=180,
    density_gamma=0.4,
    cmap_ref="Blues", cmap_tgt="Oranges", cmap_mid="Purples",
    line_color_mode="target_angle_turbo",
    solid_line_color="#7CFC00",
    line_alpha=0.5, line_width=.5, line_glow=False,
    hsv_sat=0.95, hsv_val=0.95,
    bg="white",
    view_elev=12, view_azim=185,
    seed=None,
    pairs=None,
    title=None,
    save_path=None, show=True,
    name_for_console="crossings(P)",
    t_inset_ref: float = 0.0,
    t_inset_tgt: float = -0.02,
    mark_hits: bool = True,
    hit_ms: float = 1.0, hit_alpha: float = 0.8, hit_mew: float = 1.0, hit_color: str = "k",
    iso_extent_std: float = 3.5,
    iso_ring_levels: tuple = (1.0, 2.0, 3.0),
):
    """
    Identical look to the old nd function but first projects x_ref, x_tgt via P (D×2).
    """
    import numpy as np, matplotlib.pyplot as plt, matplotlib.cm as cm
    import matplotlib.patheffects as pe
    from matplotlib.colors import PowerNorm

    if seed is not None:
        np.random.seed(seed); torch.manual_seed(seed)

    # assemble pairs in R^D
    if pairs is not None:
        x_ref, x_tgt = pairs
    else:
        if pairs_mode == "encoder":
            assert hasattr(model_or_sampler, "Enc"), "encoder pairs require AVRC model"
            x1  = sample_target_torch(n_pairs)
            mu, logv = model_or_sampler.Enc(x1)
            eps = torch.randn_like(mu)
            x_ref, x_tgt = (mu + torch.exp(0.5*logv)*eps), x1
        elif pairs_mode == "gauss":
            D = P.size(0)
            x_ref = torch.randn(n_pairs, D, device=device, dtype=TDTYPE)
            x_tgt = sample_target_torch(n_pairs)
        else:
            raise ValueError("pairs_mode must be 'encoder' or 'gauss' or pass pairs=(x_ref,x_tgt)")

    # project to nd
    Xr = project_nd(x_ref, P).detach().cpu().numpy()
    Xt = project_nd(x_tgt, P).detach().cpu().numpy()

    # ---- the rest is exactly your old nd function (unchanged), operating on Xr, Xt ----
    # (I copy the stable part verbatim, omitting extra comments to keep this block compact.)

    from matplotlib import cm as _cm
    plt.style.use("default")
    fig = plt.figure(figsize=(11.5, 6.8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(bg)
    tickc = "white" if bg == "black" else "black"
    if bg == "black":
        fig.patch.set_facecolor("black")
        _set_pane_color(ax, 'x', (0, 0, 0, 0))
        _set_pane_color(ax, 'y', (0, 0, 0, 0))
        _set_pane_color(ax, 'z', (0, 0, 0, 0))
        for spine in ax.spines.values(): spine.set_color("white")

    ax.view_init(elev=view_elev, azim=view_azim)
    t_lo = min(0.0, 0.0 + t_inset_ref)
    t_hi = max(1.0, 1.0 + t_inset_tgt)
    ax.set_xlim(t_lo, t_hi)

    Q = 0.997
    x1t_min, x1t_max = np.quantile(Xt[:,0], 1-Q), np.quantile(Xt[:,0], Q)
    x2t_min, x2t_max = np.quantile(Xt[:,1], 1-Q), np.quantile(Xt[:,1], Q)
    R = float(iso_extent_std)
    x1min, x1max = min(-R, x1t_min), max(R, x1t_max)
    x2min, x2max = min(-R, x2t_min), max(R, x2t_max)

    ax.set_ylim(x1min, x1max); ax.set_zlim(x2min, x2max)
    ax.set_xlabel("t", color=tickc); ax.set_ylabel("$x_1$", color=tickc); ax.set_zlabel("$x_2$", color=tickc)
    ax.tick_params(colors=tickc)

    def _plane_heat(ax, t_const, X, cmap, bins, alpha=0.97):
        H, xedges, yedges = np.histogram2d(X[:,0], X[:,1], bins=bins,
                                           range=[[x1min, x1max],[x2min, x2max]], density=True)
        norm = PowerNorm(gamma=max(1e-3, density_gamma))
        C = _cm.get_cmap(cmap)(norm(H).T)
        yy, zz = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
        tt = np.full_like(yy, float(t_const))
        ax.plot_surface(tt, yy, zz, rstride=1, cstride=1,
                        facecolors=C, shade=False, antialiased=False, linewidth=0, alpha=alpha)
        return norm

    def _density_lookup(X, bins):
        H, xe, ye = np.histogram2d(
            X[:,0], X[:,1], bins=bins,
            range=[[x1min, x1max],[x2min, x2max]], density=True
        )
        i1 = np.clip(np.searchsorted(xe, X[:,0]) - 1, 0, H.shape[0]-1)
        i2 = np.clip(np.searchsorted(ye, X[:,1]) - 1, 0, H.shape[1]-1)
        d = H[i1, i2]
        return H, d

    if plane_mode == "heatmap":
        cnr = _plane_heat(ax, 0.0, Xr, cmap_ref, bins=bins, alpha=0.96)
        cnt = _plane_heat(ax, 1.0, Xt, cmap_tgt, bins=bins, alpha=0.96)
        cbr = fig.colorbar(_cm.ScalarMappable(norm=cnr, cmap=cmap_ref), ax=ax, fraction=0.026, pad=0.04)
        cbt = fig.colorbar(_cm.ScalarMappable(norm=cnt, cmap=cmap_tgt), ax=ax, fraction=0.026, pad=0.01)
        cbr.set_label("ref density @ t=0"); cbt.set_label("target density @ t=1")
    else:
        Href, d_ref = _density_lookup(Xr, bins)
        Htgt, d_tgt = _density_lookup(Xt, bins)
        cnr = PowerNorm(gamma=max(1e-3, density_gamma))
        cnt = PowerNorm(gamma=max(1e-3, density_gamma))
        ax.scatter(np.full(Xr.shape[0], 0.0), Xr[:,0], Xr[:,1],
                   c=d_ref, cmap=cmap_ref, norm=cnr, s=1.2, alpha=0.65,
                   depthshade=False, zorder=1, rasterized=True)
        cbr = fig.colorbar(_cm.ScalarMappable(norm=cnr, cmap=cmap_ref), ax=ax, fraction=0.026, pad=0.04)
        cbt = fig.colorbar(_cm.ScalarMappable(norm=cnt, cmap=cmap_tgt), ax=ax, fraction=0.026, pad=0.01)
        cbr.set_label("ref density @ t=0"); cbt.set_label("target density @ t=1")

    if midplane:
        M = 0.5*(Xr + Xt)
        Hm, xe, ye = np.histogram2d(M[:,0], M[:,1], bins=mid_bins,
                                    range=[[x1min, x1max],[x2min, x2max]], density=True)
        from matplotlib import cm as cm2
        from matplotlib.colors import PowerNorm as PN2
        norm_m = PN2(gamma=0.7 if density_gamma is None else density_gamma)
        Cm = cm2.get_cmap(cmap_mid)(norm_m(Hm).T)
        yy, zz = np.meshgrid(xe[:-1], ye[:-1], indexing="ij")
        tt = np.full_like(yy, float(mid_t))
        ax.plot_surface(tt, yy, zz, rstride=1, cstride=1,
                        facecolors=Cm, shade=False, antialiased=False, linewidth=0, alpha=0.60, zorder=2)

    if line_indices is not None:
        keep_idx = np.asarray(line_indices, dtype=int)
        keep_idx = keep_idx[(keep_idx >= 0) & (keep_idx < Xr.shape[0])]
        if keep_idx.size == 0:
            keep_idx = np.arange(min(subset_lines, Xr.shape[0]))
    else:
        disp  = Xt - Xr
        theta = np.arctan2(disp[:,1], disp[:,0]) % (2*np.pi)
        length= np.linalg.norm(disp, axis=1)
        if subset_lines >= Xr.shape[0]:
            keep_idx = np.arange(Xr.shape[0])
        elif subset_strategy == "angle_stratified":
            nb = max(8, int(np.sqrt(subset_lines)))
            bins_theta = np.linspace(0, 2*np.pi, nb+1)
            keep = []
            for b in range(nb):
                mask = (theta >= bins_theta[b]) & (theta < bins_theta[b+1])
                cand = np.where(mask)[0]
                if cand.size == 0: continue
                k = max(1, int(np.ceil(subset_lines/nb)))
                sel = cand[np.argsort(length[cand])[-k:]] if cand.size > k else cand
                keep.append(sel)
            keep_idx = np.unique(np.concatenate(keep))[:subset_lines]
        elif subset_strategy == "longest":
            keep_idx = np.argsort(length)[-subset_lines:]
        else:
            keep_idx = np.random.choice(Xr.shape[0], size=subset_lines, replace=False)

    turbo = cm.get_cmap("turbo")
    def _color_from_target(x_t):
        if line_color_mode == "solid":
            return solid_line_color
        elif line_color_mode == "target_angle_turbo":
            ang = (np.arctan2(x_t[1], x_t[0]) % (2*np.pi)) / (2*np.pi)
            return turbo(ang)
        elif line_color_mode == "target_angle_hsv":
            from matplotlib import colors as mcolors
            ang = (np.arctan2(x_t[1], x_t[0]) % (2*np.pi)) / (2*np.pi)
            return mcolors.hsv_to_rgb([ang, hsv_sat, hsv_val])
        else:
            u = (x_t[0] - x1min) / (x1max - x1min + 1e-9)
            v = (x_t[1] - x2min) / (x2max - x2min + 1e-9)
            c1 = cm.get_cmap("plasma")(u); c2 = cm.get_cmap("viridis")(v)
            rgb = 0.60*np.array(c1[:3]) + 0.40*np.array(c2[:3])
            return np.clip(rgb, 0, 1)

    pefx = [pe.Stroke(linewidth=line_width+1.2, foreground="k", alpha=0.85), pe.Normal()] if line_glow else None
    t0_draw = 0.0 + t_inset_ref
    t1_draw = 1.0 + t_inset_tgt

    target_hits_y, target_hits_z = [], []
    for i in keep_idx.tolist():
        x0, x1v = Xr[i], Xt[i]
        line_col = _color_from_target(x1v)
        ax.plot([t0_draw, t1_draw], [x0[0], x1v[0]], [x0[1], x1v[1]],
                color=line_col, alpha=line_alpha, lw=line_width,
                linestyle="--", dash_capstyle="round", path_effects=pefx, zorder=5, rasterized=True)
        if mark_hits:
            ax.plot([0.0], [x0[0]], [x0[1]],
                    marker='x', markersize=hit_ms, markeredgewidth=hit_mew,
                    color=line_col, alpha=hit_alpha, zorder=8)
            target_hits_y.append(x1v[0]); target_hits_z.append(x1v[1])

    if plane_mode == "scatter":
        ax.scatter(np.full(Xt.shape[0], 1.0), Xt[:,0], Xt[:,1],
                   c=None, cmap=cmap_tgt, s=1.0, alpha=0.85,
                   depthshade=False, zorder=20, rasterized=True)

    if mark_hits and len(target_hits_y) > 0:
        ax.plot([1.0]*len(target_hits_y), target_hits_y, target_hits_z,
                linestyle='None', marker='x', markersize=hit_ms, markeredgewidth=hit_mew,
                color=hit_color, alpha=hit_alpha, zorder=25)

    mr = Xr.mean(axis=0); mt = Xt.mean(axis=0)
    ax.scatter([0.0],[mr[0]],[mr[1]], s=70, c="#00e5ff", edgecolors="k",
               depthshade=False, label="ref mean", zorder=30)
    ax.scatter([1.0],[mt[0]],[mt[1]], s=70, c="#ffd400", edgecolors="k",
               depthshade=False, label="target mean", zorder=30)
    ax.legend(loc="upper left", facecolor="none", edgecolor=("white" if bg=="black" else "black"))
    ax.set_title(title or f"Crossings — projection", color=tickc)

    if iso_ring_levels and len(iso_ring_levels) > 0:
        tt = np.linspace(0, 2*np.pi, 600)
        for r in iso_ring_levels:
            yy = r * np.cos(tt); zz = r * np.sin(tt)
            ax.plot(np.zeros_like(tt), yy, zz, color=(1,1,1,0.95), lw=0.9, ls="--", zorder=3)

    Xproj = np.vstack([Xr, Xt]) - np.mean(np.vstack([Xr, Xt]), axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(Xproj, full_matrices=False)
    v = vh[0]
    p0, p1 = Xr @ v, Xt @ v
    m = min(4000, p0.size); I = np.random.choice(p0.size, size=m, replace=False)
    r0 = np.argsort(np.argsort(p0[I])); r1 = np.argsort(np.argsort(p1[I]))
    invs = np.sum(np.sign(r0[:,None]-r0[None,:]) != np.sign(r1[:,None]-r1[None,:]))
    print(f"[{name_for_console}] approx crossing ratio: {invs/(m*(m-1)):.4f} (m={m})")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show: plt.show()
    else:    plt.close(fig)



# ========================= NEW: RF-on-current-coupling probe ==========================
# Put this in the same cell as AVRC / utilities (after your imports).

# ------------------------------- Generic config (REPLACES) -------------------------------
@dataclass
class AVRCConfig:
    D: int = 2
    rounds: int = 400
    batch: int = 2
    log_every: int = 10

    init_default: str = "gauss"
    init_fuzz_eps: float = 1e-2

    enc_hidden: int = 128
    enc_depth:  int = 4
    vel_hidden: int = 128
    vel_depth:  int = 4
    enc_residual_mu: bool = False

    pretrain_steps: int = 10000
    critic_lr: float = 1e-3
    critic_weight_decay: float = 1e-5
    critic_clip: float = 1.0
    critic_huber_delta: float = 1e9

    critic_t_alpha: float = 1.0
    critic_t_beta:  float = 1.0
    critic_t_gamma: float = 0.0
    critic_match_rf_midpoints: bool = True
    critic_match_rf_K: int | None = None
    reset_opt_v_every_round: bool = True

    enc_lr: float = 1e-3
    enc_lr_decay: float = 1.0
    ed_clip: float = 1.0
    lam_disp_start: float = 0.0
    lam_disp_end:   float = 1.0
    lam_kl_start:   float = 1.0
    lam_kl_end:     float = 1.0
    unbiased_dispersion: bool = True
    lam_align_start: float | None = None
    lam_align_end:   float | None = None
    post_anneal_rounds: int = 0

    teacher_mode: str = "intra"
    intra_ema_decay: float = 0.98
    ema_decay: float = 0.98
    critic_adapt_min: int = 20
    critic_adapt_max: int = 1000
    critic_tol: float = 0.97

    log_k: int = 8
    log_trials: int = 50
    log_n: int = 8192
    log_mmd_max_n: int = 8192
    agg_kl_batch: int = 65536
    recon_k: int = 8
    recon_n: int = 8192

    # crossings / latent viz (now projection-aware)
    k_plot: int = 10
    viz_cross_dir: str = "viz_crossings3d"
    viz_subset_strategy: str = "random"
    viz_pairs: int = 100000
    viz_subset_lines: int = 160
    viz_bins: int = 160
    viz_plane_mode: str = "scatter"
    viz_density_gamma: float = 1.0
    viz_cmap_ref: str = "Blues"
    viz_cmap_tgt: str = "Oranges"
    viz_seed: int = 123
    viz_camera: tuple = (20, -30)

    # projection settings (NEW)
    viz_proj_mode: str = "pca"          # {"pca","random"}
    viz_num_projections: int = 2        # how many nd views to save each time
    viz_proj_source: str = "target"     # {"target","latent","both"} for picking PCA basis

    # RF probe
    test_rf_every: int = 10
    test_rf_steps: int = 25000
    test_rf_batch: int = 2048
    test_rf_lr: float = 1e-3
    test_rf_clip: float = 1.0
    test_rf_hidden: int = 128
    test_rf_depth:  int = 4
    test_rf_log_every: int = 10000
    test_rf_Ks: tuple = (1,2,4,8)
    test_rf_n: int = 80_000
    test_rf_mmd_max_n: int = 8192
    test_rf_bins: int = 200
    test_rf_outdir: str = "rf_snapshots"
    test_rf_seed: int | None = 777
    test_rf_proj_index: int = 0         # which cached projection to use for probe figs

    # latent viz
    viz_latent: bool = True
    viz_latent_color_mode: str = "hsvnd"
    viz_latent_bg: str = "black"
    viz_latent_s: float = 50.0
    viz_latent_alpha: float = 0.75
    viz_latent_rings: bool = True
    viz_latent_ring_levels: tuple = (1.0, 2.0, 3.0)
    viz_latent_points: int = 1000

    # dispersion-over-t
    viz_disp_pairs: int = 16384
    viz_disp_t_steps: int = 100
    viz_disp_enable: bool = True



    # --- NEW: temporal jitter amplitude (ε). 0.0 = off (exact baseline) ---
    jitter_lambda: float = 0.0

    device = "cuda" if torch.cuda.is_available() else "cpu"



@torch.no_grad()
def sample_init_points(
    n: int,
    mode: str = "gauss",
    *,
    D: int = 2,
    enc: nn.Module | None = None,
    eps: float = 1e-2
):
    """
    z0 ~ q0(. | x1) depending on `mode` in R^D.
    - "gauss"         : N(0, I_D)
    - "encoder"       : reparam(Enc(x1))  (needs enc)
    - "identity_fuzz" : N(x1, eps I_D)
    """
    if mode == "gauss":
        return torch.randn(n, D, device=device, dtype=TDTYPE)

    elif mode == "encoder":
        assert enc is not None, "encoder init requires enc"
        x1 = sample_target_torch(n)            # expected to output (n, D) when you swap in high-D targets
        mu, logv = enc(x1)
        return reparam(mu, logv)

    elif mode in ("identity_fuzz", "x_fuzz", "id_fuzz"):
        x1 = sample_target_torch(n)
        return x1 + math.sqrt(max(eps, 1e-12)) * torch.randn_like(x1)

    else:
        raise ValueError(f"unknown init mode: {mode}")




# ----------------------------- Dimension-agnostic velocity -----------------------------
class VelocityXD(nn.Module):
    """v(x,t): R^D × [0,1] → R^D"""
    def __init__(self, D=2, hidden=128, depth=4):
        super().__init__()
        self.D = D
        self.net = EnhancedMLP(input_dim=D, output_dim=D, hidden=hidden, depth=depth)
    def forward(self, x, t):
        return self.net(x, t)


@torch.no_grad()
def _disp_metrics_vec(pred: torch.Tensor, ell: torch.Tensor):
    eps = 1e-8
    resid2 = (pred-ell).pow(2).sum(dim=1)
    ell2   = ell.pow(2).sum(dim=1)
    mse  = float(resid2.mean().detach().cpu())
    nmse = float((resid2/(ell2+eps)).mean().detach().cpu())
    return {"mse": mse, "nmse": nmse}


# ---------------------- RF on arbitrary coupling pairs (generic D) ---------------------
def train_rectified_flow_on_pairs_nd(
    make_pairs_fn,
    *,
    D: int,
    steps=10_000, batch=2048, lr=1e-3, clip=1.0, hidden=128, depth=4, log_every=200,
    midpoints_K: int | None = None, weight_decay: float = 0.0, seed=None
):
    if seed is not None:
        torch_state = torch.random.get_rng_state()
        np_state = np.random.get_state()
        torch.manual_seed(seed); np.random.set_state(np_state)

    Vx = VelocityXD(D=D, hidden=hidden, depth=depth).to(device)
    opt = torch.optim.Adam(Vx.parameters(), lr=lr, betas=(0.9,0.99), weight_decay=weight_decay)

    t0 = time.time()
    for it in range(1, steps+1):
        x0, x1 = make_pairs_fn(batch)                 # (B, D), (B, D)
        if midpoints_K is None:
            t = torch.rand(batch,1, device=device, dtype=TDTYPE)
        else:
            idx = torch.randint(midpoints_K, (batch,), device=device)
            t = ((idx + 0.5)/float(midpoints_K)).view(batch,1).type_as(x0)

        xt  = (1.0 - t)*x0 + t*x1
        ell = (x1 - x0).detach()
        pred= Vx(xt, t)
        loss= F.mse_loss(pred, ell)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(Vx.parameters(), clip); opt.step()

        if (it % log_every) == 0:
            resid2 = (pred-ell).pow(2).sum(dim=1).mean()
            nmse   = resid2 / (ell.pow(2).sum(dim=1).mean() + 1e-8)
            print(f"[fresh-RF-D] step {it}/{steps}  loss={float(loss):.4f}  NMSE={float(nmse):.4f}  (+{time.time()-t0:.1f}s)")
            t0 = time.time()

    if seed is not None:
        torch.random.set_rng_state(torch_state); np.random.set_state(np_state)

    @torch.no_grad()
    def sampler(n: int, nfe: int, init="gauss", enc=None, fuzz_eps=1e-2):
        x = sample_init_points(n, init, D=D, enc=enc, eps=fuzz_eps)
        dt = 1.0/float(max(nfe,1))
        for i in range(nfe):
            t = torch.full((n,1), (i+0.5)*dt, device=device, dtype=TDTYPE)
            x = x + dt * Vx(x, t)
        return x
    return Vx, sampler



# --------------- RF probe grid via projection (REPLACES _plot_rf_probe_grid_nd) ---------------
@torch.no_grad()
def _plot_rf_probe_grid_proj(
    sampler, enc, Ks, n, P: torch.Tensor, bins=180, out_path=None,
    title="fresh RF probe (proj)",
    cmap="magma",
    gamma=0.42,
    vmax_percentile=99.7,
):
    import os, gc, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
    from matplotlib.colors import PowerNorm

    try: plt.close('all')
    except Exception: pass
    mpl.rcdefaults()

    with plt.rc_context({
        "figure.facecolor": "black",
        "axes.facecolor":   "black",
        "savefig.facecolor":"black",
        "axes.edgecolor":   "white",
        "axes.labelcolor":  "white",
        "xtick.color":      "white",
        "ytick.color":      "white",
    }):
        if out_path is not None:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Fixed target, then project
        x_tgt = sample_target_torch(n)
        X = project_nd(x_tgt, P).detach().cpu().numpy()

        q = 0.997
        xlim = (np.quantile(X[:,0], 1-q), np.quantile(X[:,0], q)); xlim = [val*1.2 for val in xlim]
        ylim = (np.quantile(X[:,1], 1-q), np.quantile(X[:,1], q)); ylim = [val*1.2 for val in ylim]
        xlim, ylim = _square_limits(xlim, ylim)

        y_edges = np.linspace(xlim[0], xlim[1], bins+1)
        z_edges = np.linspace(ylim[0], ylim[1], bins+1)
        Ht, *_ = np.histogram2d(X[:,0], X[:,1], bins=[y_edges, z_edges], density=True)

        panels = []
        for K in Ks:
            xe = sampler(n, K, init="encoder", enc=enc)
            He, *_ = np.histogram2d(
                project_nd(xe, P).detach().cpu().numpy()[:,0],
                project_nd(xe, P).detach().cpu().numpy()[:,1],
                bins=[y_edges, z_edges], density=True
            )
            panels.append((K, "init: encoder", He))

            xg = sampler(n, K, init="gauss", enc=enc)
            Hg, *_ = np.histogram2d(
                project_nd(xg, P).detach().cpu().numpy()[:,0],
                project_nd(xg, P).detach().cpu().numpy()[:,1],
                bins=[y_edges, z_edges], density=True
            )
            panels.append((K, "init: gauss", Hg))

        all_vals = np.concatenate([Ht.ravel()] + [H.ravel() for (_, _, H) in panels])
        vmax = np.percentile(all_vals, vmax_percentile)
        norm = PowerNorm(gamma=max(1e-3, float(gamma)), vmin=0.0, vmax=max(1e-9, vmax))

        fig, axs = plt.subplots(len(Ks), 3, figsize=(12.0, 3.0*len(Ks)), sharex=True, sharey=True)
        if len(Ks) == 1:
            axs = np.array([axs])

        for r, K in enumerate(Ks):
            # left target
            ax_t = axs[r, 0]
            ax_t.set_facecolor("black")
            ax_t.imshow(Ht.T, origin="lower", extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                        cmap=cmap, norm=norm, interpolation="bilinear", alpha=1.0, aspect="equal")
            ax_t.set_title("target", color='w', pad=3)
            if r == len(Ks)-1: ax_t.set_xlabel("proj x", color='w')
            ax_t.set_ylabel("proj y", color='w')
            ax_t.tick_params(color='w', labelcolor='w')

            # model columns
            for c, init_label in enumerate(("init: encoder", "init: gauss"), start=1):
                H = panels[2*r + (c-1)][2]
                ax = axs[r, c]
                ax.set_facecolor("black")
                ax.imshow(H.T, origin="lower", extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                          cmap=cmap, norm=norm, interpolation="bilinear", alpha=1.0, aspect="equal")
                ax.set_title(f"K={K}  ({init_label})", color='w', pad=3)
                if r == len(Ks)-1: ax.set_xlabel("proj x", color='w')
                if c == 1: ax.set_ylabel("proj y", color='w')
                ax.tick_params(color='w', labelcolor='w')

        fig.suptitle(title, y=0.995, color='w')
        fig.tight_layout()
        if out_path is not None:
            fig.savefig(out_path, dpi=190, bbox_inches="tight", facecolor='black')
        plt.close(fig)

    gc.collect()
    return out_path



# --------------- Encoder mean scatter via projection + optional density (REPLACES) ---------------
@torch.no_grad()
def plot_encoder_means_proj(
    Enc, x1, P: torch.Tensor, *,
    out_path=None,
    color_mode="hsvnd",
    bg="black",
    s=.5, alpha=0.85, edge_lw=0.0,
    add_gaussian_rings=True,
    ring_levels=(1.0, 2.0, 3.0),
    title=None,
    iso_extent_std: float = 3.2,
):
    import numpy as np, os, matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    import matplotlib.cm as cm

    mu, _ = Enc(x1)
    M2 = project_nd(mu, P).detach().cpu().numpy()
    X2 = project_nd(x1, P).detach().cpu().numpy()

    q = 0.997
    if color_mode == "target_angle_turbo":
        ang = (np.arctan2(X2[:,1], X2[:,0]) % (2*np.pi)) / (2*np.pi)
        colors = cm.get_cmap("turbo")(ang)[:, :3]
    else:
        x1min, x1max = np.quantile(X2[:,0], 1-q), np.quantile(X2[:,0], q)
        x2min, x2max = np.quantile(X2[:,1], 1-q), np.quantile(X2[:,1], q)
        u = np.clip((X2[:,0] - x1min) / (x1max - x1min + 1e-9), 0, 1)
        v = np.clip((X2[:,1] - x2min) / (x2max - x2min + 1e-9), 0, 1)
        hsv = np.stack([u, 0.35 + 0.65*v, np.full_like(u, 0.95)], axis=1)
        colors = mcolors.hsv_to_rgb(hsv)

    R = float(iso_extent_std)
    xlim = (-R, R); ylim = (-R, R)

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7.4, 8.2))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    tickc = "white" if bg == "black" else "black"
    for spine in ax.spines.values(): spine.set_color(tickc)

    ax.scatter(M2[:,0], M2[:,1], s=s, c=colors, alpha=alpha, linewidths=edge_lw, zorder=2)

    if add_gaussian_rings:
        t = np.linspace(0, 2*np.pi, 600)
        for r in ring_levels:
            yy = r * np.cos(t); zz = r * np.sin(t)
            ax.plot(yy, zz, color=(1,1,1,0.9), lw=0.9, ls="--", zorder=3)

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("$\\mu_{P,1}$", color=tickc); ax.set_ylabel("$\\mu_{P,2}$", color=tickc)
    ax.tick_params(colors=tickc)
    ax.set_title(title or "Encoder means μ(x) (projected)", color=tickc, pad=8)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path




# ------------------ Latent samples scatter via projection (REPLACES old) ------------------
@torch.no_grad()
def plot_encoder_latent_proj(
    Enc, x1, eps, P: torch.Tensor, *,
    out_path=None,
    color_mode="hsvnd",
    bg="black",
    s=.5, alpha=0.75, edge_lw=0.0,
    add_gaussian_rings=True,
    ring_levels=(1.0, 2.0, 3.0),
    title=None,
    iso_extent_std: float = 3.5,
):
    import numpy as np, os, matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    import matplotlib.cm as cm

    mu, logv = Enc(x1)
    z  = mu + torch.exp(0.5 * logv) * eps
    Z2 = project_nd(z,  P).detach().cpu().numpy()
    X2 = project_nd(x1, P).detach().cpu().numpy()

    q = 0.997
    R = float(iso_extent_std)
    xlim = (-R, R); ylim = (-R, R)

    if color_mode == "target_angle_turbo":
        ang = (np.arctan2(X2[:,1], X2[:,0]) % (2*np.pi)) / (2*np.pi)
        colors = cm.get_cmap("turbo")(ang)
    else:
        x1min, x1max = np.quantile(X2[:,0], 1-q), np.quantile(X2[:,0], q)
        x2min, x2max = np.quantile(X2[:,1], 1-q), np.quantile(X2[:,1], q)
        u = np.clip((X2[:,0] - x1min) / (x1max - x1min + 1e-9), 0, 1)
        v = np.clip((X2[:,1] - x2min) / (x2max - x2min + 1e-9), 0, 1)
        hsv = np.stack([u, 0.35 + 0.65*v, np.full_like(u, 0.95)], axis=1)
        colors = mcolors.hsv_to_rgb(hsv)

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7.4, 8.2))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    tickc = "white" if bg == "black" else "black"
    for spine in ax.spines.values(): spine.set_color(tickc)

    ax.scatter(Z2[:,0], Z2[:,1], s=s, c=colors, alpha=alpha, linewidths=edge_lw)

    if add_gaussian_rings:
        t = np.linspace(0, 2*np.pi, 600)
        for r in ring_levels:
            yy = r * np.cos(t); zz = r * np.sin(t)
            ax.plot(yy, zz, color=(1,1,1,0.9), lw=0.9, ls="--")

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(colors=tickc)
    ax.set_title(title or "Encoder latent (projected)", color=tickc, pad=8)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


@torch.no_grad()
def make_current_V_sampler(V: nn.Module, *, fuzz_eps: float = 1e-2):
    # Grab the model's ambient dimension (VelocityXD sets .D)
    D = getattr(V, "D", None)
    if D is None:
        raise ValueError("Velocity module must expose attribute .D (ambient dimension).")

    def _sampler(n: int, nfe: int, init: str = "gauss", enc: nn.Module | None = None):
        # Pass D explicitly so 'gauss' init uses the correct ambient dim
        x = sample_init_points(n, init, D=D, enc=enc, eps=fuzz_eps)
        if nfe <= 0:
            return x
        dt = 1.0 / float(max(nfe, 1))
        for i in range(nfe):
            t = torch.full((n, 1), (i + 0.5) * dt, device=device, dtype=TDTYPE)
            x = x + dt * V(x, t)
        return x

    return _sampler




# ----- RF vs V evaluation via projection (REPLACES _eval_probe_models_2x2) -----
@torch.no_grad()
def _eval_probe_models_proj(
    samplers: dict,             # {"RF": rf_sampler, "V": v_sampler}
    enc, Ks, n=80_000, P: torch.Tensor | None = None,
    mmd_max_n=8192,
    out_img=None, tag="RF vs V (proj)", bins=200,
    cmap="magma", gamma=0.42, vmax_percentile=99.7
):
    import os, gc, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
    from matplotlib.colors import PowerNorm

    x_tgt = sample_target_torch(n).to(device)          # (n, D)
    D = x_tgt.size(1)
    if P is None:
        # default: random projection if none supplied
        P = choose_projection_pairs(D, num_pairs=1, mode="random")[0]

    tbl = {m: {"gauss": {}, "encoder": {}} for m in samplers.keys()}

    def _run_one(model_key: str, init: str, K: int):
        x = samplers[model_key](n, K, init=init, enc=enc)          # (n, D)
        w2 = sliced_w2(x, x_tgt, L=128, max_n=20_000)              # ND metric
        mmd= mmd_rbf_nd(x, x_tgt, max_n=mmd_max_n)
        Xp = project_nd(x, P).detach().cpu().numpy()               # for hist only
        return float(w2), float(mmd), Xp

    yx = project_nd(x_tgt, P).detach().cpu().numpy()
    q  = 0.997
    xlim = (np.quantile(yx[:,0], 1-q), np.quantile(yx[:,0], q)); xlim = [val * 1.2 for val in xlim]
    ylim = (np.quantile(yx[:,1], 1-q), np.quantile(yx[:,1], q)); ylim = [val * 1.2 for val in ylim]
    xlim, ylim = _square_limits(xlim, ylim)

    y_edges = np.linspace(xlim[0], xlim[1], bins+1)
    z_edges = np.linspace(ylim[0], ylim[1], bins+1)

    Ht, *_ = np.histogram2d(yx[:,0], yx[:,1], bins=[y_edges, z_edges], density=True)

    order = [("RF","gauss"), ("RF","encoder"), ("V","gauss"), ("V","encoder")]
    perK_panels = {}
    perK_arrays = {}
    for K in Ks:
        panels = [("target", Ht)]
        flat_vals = [Ht.ravel()]
        for (model_key, init) in order:
            sw2, mmd, x_np = _run_one(model_key, init, K)
            tbl[model_key][init][K] = (sw2, mmd)
            H, *_ = np.histogram2d(x_np[:,0], x_np[:,1], bins=[y_edges, z_edges], density=True)
            panels.append((f"{model_key} ({init})", H))
            flat_vals.append(H.ravel())
        perK_panels[K] = panels
        perK_arrays[K] = np.concatenate(flat_vals)

    hdr = "model/init".ljust(14) + " | " + "  ".join([f"K={K:^3d}  SW2   MMD" for K in Ks])
    print("\n=== probe:", tag, "===\n" + hdr + "\n" + "-"*len(hdr))
    for (model_key, init) in order:
        s = f"{model_key}/{init}".ljust(14) + " | "
        for K in Ks:
            sw2, mmd = tbl[model_key][init][K]
            s += f" {sw2:5.3f} {mmd:5.3f} "
        print(s)
    print()

    try: plt.close('all')
    except Exception: pass
    mpl.rcdefaults()
    with plt.rc_context({
        "figure.facecolor": "black",
        "axes.facecolor":   "black",
        "savefig.facecolor":"black",
        "axes.edgecolor":   "white",
        "axes.labelcolor":  "white",
        "xtick.color":      "white",
        "ytick.color":      "white",
    }):
        if out_img is not None:
            os.makedirs(os.path.dirname(out_img), exist_ok=True)

        rows = len(Ks); cols = 5
        fig, axs = plt.subplots(rows, cols, figsize=(14.0, 2.9*rows), sharex=True, sharey=True)
        if rows == 1: axs = np.array([axs])

        for r, K in enumerate(Ks):
            vmax = np.percentile(perK_arrays[K], vmax_percentile)
            norm = PowerNorm(gamma=max(1e-3, float(gamma)), vmin=0.0, vmax=max(1e-9, vmax))
            for c, (title, H) in enumerate(perK_panels[K]):
                ax = axs[r, c]
                ax.imshow(H.T, origin='lower',
                          extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                          cmap=cmap, norm=norm, interpolation="bilinear", alpha=1.0, aspect='equal')
                ax.set_title(("target" if c==0 else f"K={K} — {title}"), color='w', pad=3)
                if r == rows-1: ax.set_xlabel("proj x", color='w')
                if c == 0:      ax.set_ylabel("proj y", color='w')
                ax.tick_params(color='w', labelcolor='w')

        fig.suptitle(tag + " — histograms (projected)", y=0.995, color='w')
        fig.tight_layout()
        if out_img is not None:
            fig.savefig(out_img, dpi=190, bbox_inches="tight", facecolor='black')
        plt.close(fig)

    gc.collect()
    return tbl



# -------------------------------- AVRC model (REPLACES) --------------------------------
class AVRC:
    def __init__(self, cfg=AVRCConfig()):
        self.cfg = cfg
        D = cfg.D
        # Switched to enhanced encoder; width/depth map to previous hidden/depth
        self.Enc = EncoderX1OnlyPro(
                                  D=D,
                                  width=cfg.enc_hidden,
                                  depth=cfg.enc_depth,
                                  residual_mu=cfg.enc_residual_mu,   # <--- key line
                              ).to(device)
        self.V   = VelocityXD(D=D,    hidden=cfg.vel_hidden,  depth=cfg.vel_depth).to(device)
        self.V_teacher = VelocityXD(D=D, hidden=cfg.vel_hidden, depth=cfg.vel_depth).to(device)
        self.V_teacher.load_state_dict(self.V.state_dict())
        for p in self.V_teacher.parameters(): p.requires_grad_(False)

        self.opt_v   = torch.optim.Adam(self.V.parameters(),   lr=cfg.critic_lr, betas=(0.9,0.99),
                                        weight_decay=self.cfg.critic_weight_decay)
        self.opt_enc = torch.optim.Adam(self.Enc.parameters(), lr=cfg.enc_lr,    betas=(0.9,0.99))

        self._round_ema_shadow = None
        self.history = {'critic': [], 'organizer': [], 'sw2_k': [], 'mmd_k': [],
                        'agg_kl': [], 'recon_mse': []}
        self._lam_disp = cfg.lam_disp_start
        self._lam_kl   = cfg.lam_kl_start
        self._lam_align = (cfg.lam_align_start
                           if cfg.lam_align_start is not None else cfg.lam_disp_start)

        # viz caches
        self._viz_ready = False
        self._viz_x1 = None
        self._viz_eps = None
        self._viz_keep_idx = None
        self._viz_small_idx = None
        self._viz_proj_pairs: list[torch.Tensor] = []   # list of (D,2) matrices

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'Training on {self.device}')
        self._disp_scale0 = None
        self.round_idx = 0

    
    @torch.no_grad()
    def _time_warp(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input:  t  (B,1) physical time in [0,1].
        Output: alpha(t;phi), alpha_dot(t;phi) with one random phase per sample.
        If jitter_lambda == 0, returns (t, ones_like(t)) exactly.
        """
        eps = float(getattr(self.cfg, "jitter_lambda", 0.0))
        if eps <= 0.0:
            return t, torch.ones_like(t)

        import math
        two_pi = 2.0 * math.pi

        # one random phase per sample (same shape/device/dtype as t)
        phi = two_pi * torch.rand_like(t)

        # α(t) = t + (ε / 2π) [ sin(2π t + φ) - sin φ ]
        alpha = t + (eps / two_pi) * (torch.sin(two_pi * t + phi) - torch.sin(phi))
        # α̇(t) = 1 + ε cos(2π t + φ)
        alpha_dot = 1.0 + eps * torch.cos(two_pi * t + phi)

        return alpha, alpha_dot


    @torch.no_grad()
    def _init_crossings_subset(self):
        if self._viz_ready: return
        os.makedirs(self.cfg.viz_cross_dir, exist_ok=True)

        torch.manual_seed(self.cfg.viz_seed); np.random.seed(self.cfg.viz_seed)
        self._viz_x1  = sample_target_torch(self.cfg.viz_pairs).detach()                 # (N,D)
        self._viz_eps = torch.randn(self.cfg.viz_pairs, self.cfg.D, device=device, dtype=TDTYPE).detach()

        # round-0 geometry to fix line subset
        mu0, logv0 = self.Enc(self._viz_x1)
        z0 = mu0 + torch.exp(0.5 * logv0) * self._viz_eps
        Xr0 = z0.detach().cpu().numpy(); Xt0 = self._viz_x1.detach().cpu().numpy()

        # pick line indices by projected geometry on first projection; we use a quick fallback P
        P_fallback = choose_projection_pairs(self.cfg.D, data_for_pca=self._viz_x1 if self.cfg.viz_proj_mode=="pca" else None,
                                             num_pairs=1, mode=self.cfg.viz_proj_mode, seed=self.cfg.viz_seed)[0]
        Xr0p = project_nd(z0, P_fallback).detach().cpu().numpy()
        Xt0p = project_nd(self._viz_x1, P_fallback).detach().cpu().numpy()

        N = Xr0p.shape[0]; L = min(self.cfg.viz_subset_lines, N)
        disp  = Xt0p - Xr0p
        theta = np.arctan2(disp[:,1], disp[:,0]) % (2*np.pi)
        length= np.linalg.norm(disp, axis=1)
        if L >= N:
            keep_idx = np.arange(N)
        elif self.cfg.viz_subset_strategy == "angle_stratified":
            nb = max(8, int(np.sqrt(L))); bins_theta = np.linspace(0, 2*np.pi, nb+1)
            picks = []
            for b in range(nb):
                mask = (theta >= bins_theta[b]) & (theta < bins_theta[b+1])
                cand = np.where(mask)[0]
                if cand.size == 0: continue
                k = max(1, int(np.ceil(L/nb)))
                sel = cand[np.argsort(length[cand])[-k:]] if cand.size > k else cand
                picks.append(sel)
            keep_idx = np.unique(np.concatenate(picks))[:L]
        elif self.cfg.viz_subset_strategy == "longest":
            keep_idx = np.argsort(length)[-L:]
        else:
            rng = np.random.default_rng(self.cfg.viz_seed ^ 0xA5A5A5)
            keep_idx = rng.choice(N, size=L, replace=False)
        self._viz_keep_idx = np.asarray(keep_idx, dtype=int)

        # small latent scatter subset
        M  = min(self.cfg.viz_latent_points, N)
        rng2 = np.random.default_rng(self.cfg.viz_seed ^ 0x5A5A5A)
        self._viz_small_idx = np.asarray(rng2.choice(N, size=M, replace=False), dtype=int)

        # choose & cache projection pairs (fixed for entire training)
        data_for_pca = None
        if self.cfg.viz_proj_mode == "pca":
            if self.cfg.viz_proj_source == "target":
                data_for_pca = self._viz_x1
            elif self.cfg.viz_proj_source == "latent":
                data_for_pca = z0
            else:  # "both"
                data_for_pca = torch.cat([self._viz_x1, z0], dim=0)
        self._viz_proj_pairs = choose_projection_pairs(
            self.cfg.D,
            data_for_pca=data_for_pca,
            num_pairs=self.cfg.viz_num_projections,
            mode=self.cfg.viz_proj_mode,
            seed=self.cfg.viz_seed
        )
        self._viz_ready = True

    # ------------------------------ NEW: pair generator ------------------------------
    def _make_pairs_encoder_current(self):
        @torch.no_grad()
        def gen(n):
            x1 = sample_target_torch(int(n))
            mu, logv = self.Enc(x1)
            x0 = reparam(mu, logv)
            return x0, x1
        return gen

    # ------------------------------ RF probe (uses ND + proj) -------------------------
    def _run_rf_probe(self, round_idx: int):
        os.makedirs(self.cfg.test_rf_outdir, exist_ok=True)
        tag = f"RF-on-coupling@r{round_idx:05d}"
        print(f"\n[probe] Training fresh RF on current coupling … ({tag})")

        Vx, rf_sampler = train_rectified_flow_on_pairs_nd(
            make_pairs_fn=self._make_pairs_encoder_current(),
            D=self.cfg.D,
            steps=self.cfg.pretrain_steps,
            batch=self.cfg.batch,
            lr=self.cfg.test_rf_lr,
            clip=self.cfg.test_rf_clip,
            hidden=self.cfg.vel_hidden,
            depth=self.cfg.vel_depth,
            log_every=5000,
            seed=self.cfg.test_rf_seed,
            midpoints_K=(self.cfg.critic_match_rf_K or self.cfg.recon_k),
            weight_decay=self.cfg.critic_weight_decay
        )

        v_sampler = make_current_V_sampler(self.V)

        # choose a cached projection (or the first)
        proj_idx = min(max(0, int(self.cfg.test_rf_proj_index)), max(0, len(self._viz_proj_pairs)-1))
        P = self._viz_proj_pairs[proj_idx] if self._viz_proj_pairs else choose_projection_pairs(self.cfg.D, num_pairs=1)[0]

        img = os.path.join(self.cfg.test_rf_outdir, f"rf_probe_{round_idx:05d}_p{proj_idx:02d}.png")
        tbl = _eval_probe_models_proj(
            samplers={"RF": rf_sampler, "V": v_sampler},
            enc=self.Enc,
            Ks=self.cfg.test_rf_Ks,
            n=self.cfg.test_rf_n,
            P=P,
            mmd_max_n=self.cfg.test_rf_mmd_max_n,
            out_img=img,
            tag=tag,
            bins=self.cfg.test_rf_bins
        )

        txt = os.path.join(self.cfg.test_rf_outdir, f"rf_probe_{round_idx:05d}.txt")
        with open(txt, "w") as f:
            f.write(f"{tag}\n")
            for model in ("RF","V"):
                for init in ("gauss","encoder"):
                    for K in self.cfg.test_rf_Ks:
                        sw2, mmd = tbl[model][init][K]
                        f.write(f"{model},{init},K={K},SW2={sw2:.6f},MMD={mmd:.6f}\n")
        print(f"[probe] saved: {img}  and {txt}")

    # ------------------------------ Crossings frames (projected) ----------------------
    @torch.no_grad()
    def _save_crossings_frame(self, round_idx: int):
        if not self._viz_ready:
            self._init_crossings_subset()
        mu, logv = self.Enc(self._viz_x1)
        z0 = mu + torch.exp(0.5 * logv) * self._viz_eps

        elev, azim = self.cfg.viz_camera
        for j, P in enumerate(self._viz_proj_pairs):
            out = os.path.join(self.cfg.viz_cross_dir, f"cross_{round_idx:05d}_p{j:02d}.png")
            plot_crossings_proj(
                P,
                pairs=(z0, self._viz_x1),
                plane_mode=self.cfg.viz_plane_mode,
                bins=self.cfg.viz_bins,
                subset_lines=len(self._viz_keep_idx),
                subset_strategy=self.cfg.viz_subset_strategy,
                line_indices=self._viz_keep_idx,
                density_gamma=self.cfg.viz_density_gamma,
                cmap_ref=self.cfg.viz_cmap_ref,
                cmap_tgt=self.cfg.viz_cmap_tgt,
                view_elev=elev, view_azim=azim,
                title=f"AVRC coupling — round {round_idx} — proj {j}",
                save_path=out,
                show=False
            )

    # ------------------------------ Latent scatter frames (projected) -----------------
    @torch.no_grad()
    def _save_latent_scatter_frame(self, round_idx: int):
        if not self._viz_ready:
            self._init_crossings_subset()
        idx = torch.as_tensor(self._viz_small_idx, device=device, dtype=torch.long)
        x1_small  = self._viz_x1[idx]
        eps_small = self._viz_eps[idx]

        for j, P in enumerate(self._viz_proj_pairs):
            out = os.path.join(self.cfg.viz_cross_dir, f"latent_{round_idx:05d}_p{j:02d}.png")
            plot_encoder_latent_proj(
                self.Enc, x1_small, eps_small, P,
                out_path=out,
                color_mode=self.cfg.viz_latent_color_mode,
                bg=self.cfg.viz_latent_bg,
                s=self.cfg.viz_latent_s,
                alpha=self.cfg.viz_latent_alpha,
                add_gaussian_rings=self.cfg.viz_latent_rings,
                ring_levels=self.cfg.viz_latent_ring_levels,
                title=f"Encoder latent (proj {j}) — round {round_idx}"
            )

    @torch.no_grad()
    def _save_latent_mu_frame(self, *, round_idx: int):
        if not self._viz_ready:
            self._init_crossings_subset()
        for j, P in enumerate(self._viz_proj_pairs):
            out_path = os.path.join(self.cfg.viz_cross_dir, f"latent_mu_{round_idx:05d}_p{j:02d}.png")
            plot_encoder_means_proj(
                Enc=self.Enc,
                x1=self._viz_x1,
                P=P,
                out_path=out_path,
                color_mode=self.cfg.viz_latent_color_mode,
                bg=self.cfg.viz_latent_bg,
                s=self.cfg.viz_latent_s,
                alpha=self.cfg.viz_latent_alpha,
                add_gaussian_rings=self.cfg.viz_latent_rings,
                ring_levels=self.cfg.viz_latent_ring_levels,
                iso_extent_std=3.2,
                title=f"Encoder means μ(x) — r={round_idx:05d} — proj {j}",
            )


    @torch.no_grad()
    def _save_dispersion_over_t_frame(self, round_idx: int):
        """
        Plot dispersion vs t for the current model snapshot:
        E[ || v(x_t, t) - (x1 - x0) ||_2 ] over t in [0,1].
        Resamples pairs every call (no fixed subset).
        """
        if not self.cfg.viz_disp_enable:
            return

        import os
        import matplotlib.pyplot as plt
        import numpy as np

        n = int(self.cfg.viz_disp_pairs)

        # fresh target samples
        x1 = sample_target_torch(n).to(self.device).to(TDTYPE)           # (n,2)

        # encoder -> z0
        mu, logv = self.Enc(x1)
        eps = torch.randn_like(mu)
        x0 = mu + torch.exp(0.5 * logv) * eps                            # (n,2)

        # label is constant in t for each pair
        label = x1 - x0                                                  # (n,2)

        # t grid
        t_vals = torch.linspace(0.0, 1.0, steps=int(self.cfg.viz_disp_t_steps),
                                device=self.device, dtype=TDTYPE)

        disp_vals = []
        for t in t_vals:
            # straight chord interpolation
            xt = (1.0 - t) * x0 + t * x1                                 # (n,2)

            # --- IMPORTANT: pass a Python float to torch.full (robust on all builds) ---
            ti = float(t.item())                                         # NEW
            tt = torch.full((n, 1), ti, device=self.device, dtype=TDTYPE) # NEW

            pred = self.V(xt, tt)                                        # (n,2)
            # E[ || pred - label ||_2 ]
            d = (pred - label).pow(2).sum(dim=1).sqrt().mean()
            disp_vals.append(d.item())

        # Convert once for plotting
        t_np   = t_vals.detach().cpu().numpy()
        disp_np= np.array(disp_vals, dtype=np.float64)

        # --- Lock the vertical scale across frames using the very first call's value at t=0 ---
        # (Exactly what you asked: use dispersion@t=0 as the anchor stored on the model.)
        if self._disp_scale0 is None:                                    # NEW
            self._disp_scale0 = float(disp_np[0] + 1e-12)                # NEW

        # figure
        out_dir = self.cfg.viz_cross_dir
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"dispersion_{round_idx:05d}.png")

        plt.figure(figsize=(6, 4), dpi=150)
        plt.plot(t_np, disp_np, linewidth=2, label="dispersion")

        # Fix y-limits to a stable band anchored at the first frame's t=0
        ymax = max(1.10 * self._disp_scale0, float(disp_np.max()) * 1.05)  # NEW
        plt.ylim(0.0, ymax)                                                 # NEW

        plt.xlabel("t")
        plt.ylabel(r"$\mathbb{E}\,[\|\,v(x_t,t) - (x_1-x_0)\|_2]$")
        plt.title(f"Dispersion vs t — round {round_idx:05d}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out)
        plt.close()

        # tiny console sanity line for endpoints (helps diagnose “0 @ t=0” claims)
        lab_norm = float(label.pow(2).sum(dim=1).sqrt().mean().detach().cpu())   # NEW
        print(f"[dispersion] r={round_idx}  d(0)={disp_np[0]:.4f}  d(1)={disp_np[-1]:.4f}  E||ℓ||={lab_norm:.4f}")  # NEW

    # ------------------------------ helpers -----------------------------------
    def _reset_critic_optimizer(self):
        self.opt_v = torch.optim.Adam(self.V.parameters(), lr=self.cfg.critic_lr, betas=(0.9,0.99),
                                      weight_decay=self.cfg.critic_weight_decay)

    @torch.no_grad()
    def _start_intra_ema(self):
        if self.cfg.teacher_mode == "intra":
            self._round_ema_shadow = [p.detach().clone() for p in self.V.parameters()]

    @torch.no_grad()
    def _accumulate_intra_ema(self):
        if self.cfg.teacher_mode != "intra" or self._round_ema_shadow is None: return
        d = self.cfg.intra_ema_decay
        for s, p in zip(self._round_ema_shadow, self.V.parameters()):
            s.mul_(d).add_(p.detach(), alpha=(1.0 - d))

    @torch.no_grad()
    def _commit_intra_ema_to_teacher(self):
        if self.cfg.teacher_mode == "intra":
            if self._round_ema_shadow is None:
                self.V_teacher.load_state_dict(self.V.state_dict())
            else:
                for pt, s in zip(self.V_teacher.parameters(), self._round_ema_shadow):
                    pt.copy_(s)
            self._round_ema_shadow = None
        elif self.cfg.teacher_mode == "hard":
            self.V_teacher.load_state_dict(self.V.state_dict())
        elif self.cfg.teacher_mode == "ema":
            for pt, ps in zip(self.V_teacher.parameters(), self.V.parameters()):
                pt.data.mul_(self.cfg.ema_decay).add_(ps.data, alpha=(1.0 - self.cfg.ema_decay))


       # ------------------------------ losses ------------------------------------
    def _critic_loss(self, z0, z1, t, w, alpha: torch.Tensor | None = None,
                     alpha_dot: torch.Tensor | None = None):
        """
        If alpha/alpha_dot are provided, use xt at alpha(t) and label = alpha_dot * (z1 - z0).
        Otherwise, fall back to the baseline (xt at t, label = z1 - z0).
        """
        zcoef = alpha if alpha is not None else t
        zt    = (1.0 - zcoef) * z0 + zcoef * z1                 # (B,D)

        # label
        ell = (z1 - z0)
        if alpha_dot is not None:
            # critic label has no grads
            lab = (alpha_dot * ell).detach()
        else:
            lab = ell.detach()

        pred = self.V(zt, t)                                    # (B,D) evaluated at physical t
        r    = pred - lab

        # Huber on vector residual (unchanged)
        r2  = (r**2).sum(dim=1, keepdim=True)
        a   = r.abs().sum(dim=1, keepdim=True)
        quad= 0.5 * r2
        lin = self.cfg.critic_huber_delta * (a - 0.5 * self.cfg.critic_huber_delta)
        hub = torch.where(a <= self.cfg.critic_huber_delta, quad, lin)
        return (w * hub).mean()


    def critic_step(self, B=None):
        B = B or self.cfg.batch
        with torch.no_grad():
            xt1 = sample_target_torch(B)
            mu0, logv0 = self.Enc(xt1)
            z0 = reparam(mu0, logv0)
            z1 = xt1

        if getattr(self.cfg, "critic_match_rf_midpoints", False):
            K = self.cfg.critic_match_rf_K or self.cfg.recon_k
            idx = torch.randint(0, K, (B,), device=device)
            t = ((idx + 0.5) / float(K)).view(B,1).type_as(z0)
            w = torch.ones_like(t)
        else:
            t = torch.distributions.Beta(self.cfg.critic_t_alpha, self.cfg.critic_t_beta)\
                .sample((B,1)).to(device=device, dtype=TDTYPE)
            w = ((1.0 - t) ** self.cfg.critic_t_gamma); w = w / (w.mean() + 1e-8)

        # --- NEW: apply temporal jitter (no-op if jitter_lambda == 0) ---
        alpha, alpha_dot = self._time_warp(t)

        loss = self._critic_loss(z0, z1, t, w, alpha=alpha, alpha_dot=alpha_dot)
        self.opt_v.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(self.V.parameters(), self.cfg.critic_clip)
        self.opt_v.step()
        self._accumulate_intra_ema()
        return float(loss.item())


    def organizer_step(self, B=None):
        B = B or self.cfg.batch
        xt1 = sample_target_torch(B)
        mu0_enc, logv0 = self.Enc(xt1)
        z0 = reparam(mu0_enc, logv0)
        z1 = xt1

        # K midpoints (physical time grid)
        K = self.cfg.recon_k
        reps = (B + K - 1)//K
        t_mid = (torch.arange(K, device=device, dtype=TDTYPE) + 0.5) / float(K)
        t = t_mid.repeat_interleave(reps)[:B].view(B,1)
        w = torch.full_like(t, 1.0/(K*K))

        # --- NEW: warp ---
        alpha, alpha_dot = self._time_warp(t)

        # warped state and instantaneous label
        zt  = (1.0 - alpha) * z0 + alpha * z1        # xt at α(t)
        ell = (z1 - z0)                              # (no detach; grads flow via z0)
        St  = alpha_dot * ell                        # instantaneous label α̇(t)ℓ

        if self.cfg.unbiased_dispersion:
            mu_t = self.V_teacher(zt, t).detach()
            t0   = torch.zeros_like(t)
            mu_0 = self.V_teacher(z0, t0).detach()

            # Dispersion + alignment (same algebra, ℓ -> St, and teacher is stop-grad)
            a = St - mu_t
            b = mu_t - mu_0

            a2     = (a * a).sum(dim=1, keepdim=True)
            b2_log = ((b * b).sum(dim=1, keepdim=True)).detach()
            cross  = (b * (St - mu_0)).sum(dim=1, keepdim=True)

            denom       = (w.mean() + 1e-12)
            disp_piece  = (w * a2).mean() / denom
            align_piece = (w * (2.0 * cross - b2_log)).mean() / denom
        else:
            with torch.no_grad():
                v_targ = self.V_teacher(zt, t)
            diff2 = (v_targ - St).pow(2).sum(dim=1, keepdim=True)
            disp_piece  = (w * diff2).mean() / (w.mean() + 1e-12)
            align_piece = torch.tensor(0.0, device=zt.device, dtype=zt.dtype)

        loss_kl = kl_normal_diag(mu0_enc, logv0)

        loss = (self._lam_disp  * disp_piece
              + self._lam_align * align_piece
              + self._lam_kl    * loss_kl)

        self.opt_enc.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(self.Enc.parameters(), self.cfg.ed_clip)
        self.opt_enc.step()
        ...
        return {'disp': float(disp_piece.item()),
                'align': float(align_piece.item()),
                'kl': float(loss_kl.item()),
                'total': float(loss.item())}


    # ------------------------------ integration & eval -------------------------
    @torch.no_grad()
    def _integrate_steps(self, z0: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0: return z0.clone()
        z = z0.clone()
        dt = 1.0 / float(k)
        n = z.size(0)
        for i in range(k):
            t = torch.full((n,1), (i+0.5)*dt, device=device, dtype=TDTYPE)
            z = z + dt * self.V(z, t)
        return z

    @torch.no_grad()
    def _eval_divs_k(self, k: int, n: int, trials: int = 1, init: str = "gauss", mmd_max_n: int = 8192):
        def _sample_z0(nn: int) -> torch.Tensor:
            if init == "gauss":
                return torch.randn(nn, self.cfg.D, device=device, dtype=TDTYPE)
            elif init == "encoder":
                x1 = sample_target_torch(nn)
                mu, logv = self.Enc(x1)
                return reparam(mu, logv)
            else:
                raise ValueError("init must be 'gauss' or 'encoder'.")
        trials = max(1, int(trials))
        mmd_vals, sw2_val = [], None
        for _ in range(trials):
            z0 = _sample_z0(n)
            zK = self._integrate_steps(z0, k)
            x_tgt = sample_target_torch(n)
            if sw2_val is None:
                sw2_val = sliced_w2(zK, x_tgt, L=128, max_n=20000)
            mmd_vals.append(mmd_rbf_nd(zK, x_tgt, max_n=mmd_max_n))
        return sw2_val, float(np.mean(mmd_vals))

    @torch.no_grad()
    def _estimate_aggregate_kl(self, N: int):
        x1 = sample_target_torch(N)
        mu, logv = self.Enc(x1)
        z = mu + torch.exp(0.5*logv) * torch.randn_like(mu)
        m = z.mean(dim=0, keepdim=True)               # (1,D)
        C = (z - m).T @ (z - m) / float(N) + 1e-6*torch.eye(self.cfg.D, device=device, dtype=TDTYPE)
        trC = torch.trace(C)
        mm  = (m @ m.T).squeeze()
        logdetC = torch.logdet(C)
        kl = 0.5*(trC + mm - self.cfg.D - logdetC)
        return float(kl.detach().cpu()), float(m.squeeze().norm().detach().cpu()), float(trC.detach().cpu()/self.cfg.D)

    @torch.no_grad()
    def _flow_reconstruction_mse(self, K: int, N: int):
        x1 = sample_target_torch(N)
        mu, _ = self.Enc(x1)
        zK = self._integrate_steps(mu, K)
        mse = F.mse_loss(zK, x1)
        return float(mse.detach().cpu())


    # ------------------------------ Train loop (wire new methods) ---------------------
    def train(self, rounds=None, progress=True, seed=None):
        seed_everything(seed)
        rounds = rounds or self.cfg.rounds
        anneal_T = rounds - self.cfg.post_anneal_rounds

        def schedule(start, end, r):
            if anneal_T <= 0: return end
            if r <= anneal_T: return lin_sched(start, end, r, anneal_T)
            return end

        self.pretrain_velocity(progress=progress)

        self._init_crossings_subset()
        try:
            self._save_crossings_frame(round_idx=0)
            self._save_dispersion_over_t_frame(round_idx=0)
            if self.cfg.viz_latent:
                self._save_latent_scatter_frame(round_idx=0)
                self._save_latent_mu_frame(round_idx=0)
        except Exception as e:
            print(f"[warn] viz @0 failed: {e}")

        if self.cfg.test_rf_every and self.cfg.test_rf_every > 0:
            self._run_rf_probe(round_idx=0)

        t0 = time.time()
        for r in range(1, rounds+1):
            self._lam_disp = schedule(self.cfg.lam_disp_start, self.cfg.lam_disp_end, r)
            self._lam_kl   = schedule(self.cfg.lam_kl_start,   self.cfg.lam_kl_end,   r)
            align_start = (self.cfg.lam_align_start if self.cfg.lam_align_start is not None else self.cfg.lam_disp_start)
            align_end   = (self.cfg.lam_align_end   if self.cfg.lam_align_end   is not None else self.cfg.lam_disp_end)
            self._lam_align = schedule(align_start, align_end, r)

            c_loss, c_steps = self._critic_round_adaptive()
            self.history['critic'].append(c_loss)
            stats = self.organizer_step()
            self.history['organizer'].append(stats)

            if (self.cfg.k_plot is not None) and (self.cfg.k_plot > 0) and (r % self.cfg.k_plot == 0):
                try:
                    self._save_crossings_frame(round_idx=r)
                    self._save_dispersion_over_t_frame(round_idx=r)
                    if self.cfg.viz_latent:
                        self._save_latent_scatter_frame(round_idx=r)
                        self._save_latent_mu_frame(round_idx=r)
                except Exception as e:
                    print(f"[warn] crossings/latent save failed at r={r}: {e}")

            if (self.cfg.test_rf_every is not None) and (self.cfg.test_rf_every > 0) and (r % self.cfg.test_rf_every == 0):
                try:
                    self._run_rf_probe(round_idx=r)
                except Exception as e:
                    print(f"[warn] RF probe failed at r={r}: {e}")

            if (r % self.cfg.log_every) == 0 and progress:
                k_eval      = self.cfg.log_k
                trials      = self.cfg.log_trials
                n_eval      = self.cfg.log_n
                mmd_max_n   = self.cfg.log_mmd_max_n

                sw2_g,  mmd_g = self._eval_divs_k(k=k_eval, n=n_eval, trials=trials, init="gauss",   mmd_max_n=mmd_max_n)
                sw2_e,  mmd_e = self._eval_divs_k(k=k_eval, n=n_eval, trials=trials, init="encoder", mmd_max_n=mmd_max_n)

                self.history['sw2_k'].append({'round': r, 'gauss': sw2_g, 'encoder': sw2_e})
                self.history['mmd_k'].append({'round': r, 'gauss': mmd_g, 'encoder': mmd_e})

                kl_agg, agg_mu_norm, avg_var = self._estimate_aggregate_kl(self.cfg.agg_kl_batch)
                recon_mse = self._flow_reconstruction_mse(self.cfg.recon_k, self.cfg.recon_n)
                self.history['agg_kl'].append({'round': r, 'kl_agg': kl_agg, 'mu_norm': agg_mu_norm, 'avg_var': avg_var})
                self.history['recon_mse'].append({'round': r, 'mse': recon_mse, 'K': self.cfg.recon_k})

                print(
                    f"[{r:05d}] critic {c_loss:.4f} (steps={c_steps}) | "
                    f"disp {stats['disp']:.4f} align {stats['align']:.4f} kl {stats['kl']:.4f} | "
                    f"lam_disp {self._lam_disp:.2f} lam_align {self._lam_align:.2f} lam_kl {self._lam_kl:.2f} | "
                    f"SW2@k={k_eval} N→*: {sw2_g:.4f}  MMD: {mmd_g:.4f} | "
                    f"SW2@k={k_eval} E→*: {sw2_e:.4f}  MMD: {mmd_e:.4f} | "
                    f"AGG-KL≈{kl_agg:.4f} (||μ||≈{agg_mu_norm:.3f}, avg var≈{avg_var:.3f}) | "
                    f"FlowRecon@K={self.cfg.recon_k} MSE≈{recon_mse:.5f}"
                )
                t0 = time.time()
            self.round_idx += 1

    # ------------------------------ critic rounds ------------------------------
    def _critic_round_adaptive(self):
        losses = []
        self._start_intra_ema()
        steps_used = 0
        for k in range(self.cfg.critic_adapt_max):
            steps_used += 1
            losses.append(self.critic_step())
            if k+1 >= self.cfg.critic_adapt_min:
                win = self.cfg.critic_adapt_min
                recent = np.mean(losses[-win:])
                early  = np.mean(losses[:win])
                if recent <= self.cfg.critic_tol * max(early, 1e-8): break
        self._commit_intra_ema_to_teacher()
        avg_loss = float(np.mean(losses[-self.cfg.critic_adapt_min:]))
        return avg_loss, steps_used

    def pretrain_velocity(self, steps=None, progress=True):
        steps = self.cfg.pretrain_steps if steps is None else steps
        if steps <= 0:
            self.V_teacher.load_state_dict(self.V.state_dict()); return
        self._start_intra_ema()
        t0 = time.time()
        for it in range(1, steps+1):
            B = self.cfg.batch
            z0 = sample_source_torch(B, D=self.cfg.D)
            z1 = sample_target_torch(B)
            t  = torch.rand(B,1, device=device, dtype=TDTYPE)
            w  = torch.ones_like(t)

            # --- NEW: warp (no-op if jitter_lambda == 0) ---
            alpha, alpha_dot = self._time_warp(t)

            loss = self._critic_loss(z0, z1, t, w, alpha=alpha, alpha_dot=alpha_dot)
            self.opt_v.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(self.V.parameters(), self.cfg.critic_clip)
            self.opt_v.step()
            self._accumulate_intra_ema()
            if progress and (it % 200 == 0):
                print(f"[pretrain v (indep)] {it}/{steps}  loss={float(loss):.4f}  ({time.time()-t0:.1f}s)")
                t0 = time.time()
        self._commit_intra_ema_to_teacher()


    # ------------------------------ Public sampler (unchanged API) --------------------
    @torch.no_grad()
    def sample(self, n: int, nfe: int = 8):
        z = torch.randn(n, self.cfg.D, device=device, dtype=TDTYPE)
        dt = 1.0 / float(max(nfe,1))
        for k in range(nfe):
            t = torch.full((n,1), (k+0.5)*dt, device=device, dtype=TDTYPE)
            z = z + dt * self.V(z, t)
        return z

# ----------------------------- AVRC samplers (ND wrappers) ----------------------------
@torch.no_grad()
def avrc_sample_torch_nd(model_or_sampler, n: int, nfe: int = 8) -> torch.Tensor:
    if hasattr(model_or_sampler, "V"):
        return model_or_sampler.sample(n, nfe)
    if callable(model_or_sampler):
        out = model_or_sampler(n, nfe)
        if not isinstance(out, torch.Tensor):
            raise TypeError("Sampler must return a torch.Tensor.")
        return out
    raise TypeError("Expected AVRC or callable (n,nfe)->Tensor.")

@torch.no_grad()
def avrc_oracle_sampler_torch_nd(model) -> callable:
    D = model.cfg.D
    def sampler(n: int, nfe: int):
        x1 = sample_target_torch(n)                 # expected shape (n, D)
        mu, logv = model.Enc(x1)
        z = reparam(mu, logv)
        if nfe <= 0: return z
        dt = 1.0/float(max(nfe,1))
        for k in range(nfe):
            t = torch.full((n,1), (k+0.5)*dt, device=device, dtype=TDTYPE)
            z = z + dt * model.V(z, t)
        return z
    return sampler

@torch.no_grad()
def batch_dispersion_metrics(pred: torch.Tensor, ell: torch.Tensor, t: torch.Tensor | None = None):
    eps = 1e-8
    resid = (pred - ell)
    mse  = torch.mean((resid**2).sum(dim=1))
    nmse = mse / (torch.mean((ell**2).sum(dim=1)) + eps)
    return {"mse": float(mse.detach().cpu()), "nmse": float(nmse.detach().cpu())}


# ----------------------------- Standard RF (independent) ND ---------------------------
def train_rectified_flow_nd(
    *,
    D: int,
    steps=10000, batch=2048, lr=1e-3, clip=1.0, log_every=200,
    hidden=128, depth=4,
    midpoints_K: int | None = None, weight_decay: float = 0.0, seed: int | None = None
):
    if seed is not None:
        torch_state = torch.random.get_rng_state(); np_state = np.random.get_state()
        torch.manual_seed(seed); np.random.set_state(np_state)

    Vx = VelocityXD(D=D, hidden=hidden, depth=depth).to(device)
    opt = torch.optim.Adam(Vx.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)

    for it in range(1, steps+1):
        x0 = sample_source_torch(batch, D=D)
        x1 = sample_target_torch(batch)
        if midpoints_K is None:
            t  = torch.rand(batch, 1, device=device, dtype=TDTYPE)
        else:
            idx = torch.randint(midpoints_K, (batch,), device=device)
            t   = ((idx + 0.5) / float(midpoints_K)).view(batch, 1).type_as(x0)
        xt  = (1.0 - t) * x0 + t * x1
        ell = (x1 - x0).detach()
        pred = Vx(xt, t)
        loss = F.mse_loss(pred, ell)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(Vx.parameters(), clip); opt.step()

        if (it % log_every) == 0:
            resid2 = (pred-ell).pow(2).sum(dim=1).mean()
            nmse   = resid2 / (ell.pow(2).sum(dim=1).mean() + 1e-8)
            print(f"[RF-D] step {it}/{steps}  loss={float(loss):.4f}  NMSE={float(nmse):.4f}")

    if seed is not None:
        torch.random.set_rng_state(torch_state); np.random.set_state(np_state)

    @torch.no_grad()
    def rf_sampler(n: int, nfe: int):
        x = torch.randn(n, D, device=device, dtype=TDTYPE)
        dt = 1.0/float(max(nfe,1))
        for i in range(nfe):
            t = torch.full((n,1), (i+0.5)*dt, device=device, dtype=TDTYPE)
            x = x + dt * Vx(x, t)
        return x

    return Vx, rf_sampler



# In[32]:


import os, shutil
for _d in ["viz_crossings3d", "rf_snapshots", "training_plots"]:
    shutil.rmtree(_d, ignore_errors=True)
for _d in ["viz_crossings3d", "viz_rf_snapshots", "training_plots"]:
    os.makedirs(_d, exist_ok=True)


# In[38]:


from logging import debug
# =============================== Main + nd Viz / Bench ================================
# Added detailed LOGGING and TIMING to diagnose stalls around benchmark_samplers_nd.
#
# What’s new:
#   • benchmark_samplers_nd now logs per-sampler/K timing (sampling, SW2, MMD), throughput,
#     and (if CUDA) memory stats; also supports chunked sampling to avoid spikes.
#   • _plot_heat_grid logs timing; KDE fitting uses capped subsampling (kde_max_n) and
#     contour grid is computed once and reused.
#   • Crossings plot unchanged except for optional logging.
#
# If you still see a stall, likely culprits are:
#   1) very large n with big Ks (lots of forward passes), or
#   2) KDE fit on huge x_tgt, or
#   3) SW2 on all n (now capped via sw2_max_n).
#
# You can lower n, use smaller Ks, or raise chunk_n to spread work.

import os, sys, math, time, numpy as np
import torch
import matplotlib.pyplot as plt
from time import perf_counter as _tic
from sklearn.neighbors import KernelDensity

# ---------- sync helpers ----------
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _gpu_mem():
    if not torch.cuda.is_available():
        return "(CPU)"
    alloc = torch.cuda.memory_allocated() / 1e9
    reserv= torch.cuda.memory_reserved() / 1e9
    return f"(GPU mem: alloc={alloc:.2f} GB, reserved={reserv:.2f} GB)"


@torch.no_grad()
def avrc_oracle_sampler_torch_nd(model) -> callable:
    D = model.cfg.D
    def sampler(n: int, nfe: int):
        x1 = sample_target_torch(n)                 # expected shape (n, D)
        mu, logv = model.Enc(x1)
        z = reparam(mu, logv)
        if nfe <= 0: return z
        dt = 1.0/float(max(nfe,1))
        for k in range(nfe):
            t = torch.full((n,1), (k+0.5)*dt, device=device, dtype=TDTYPE)
            z = z + dt * model.V(z, t)
        return z
    return sampler


# ----------------------------- AVRC samplers (ND wrappers) ----------------------------
@torch.no_grad()
def avrc_sample_torch_nd(model_or_sampler, n: int, nfe: int = 8) -> torch.Tensor:
    if hasattr(model_or_sampler, "V"):
        return model_or_sampler.sample(n, nfe)
    if callable(model_or_sampler):
        out = model_or_sampler(n, nfe)
        if not isinstance(out, torch.Tensor):
            raise TypeError("Sampler must return a torch.Tensor.")
        return out
    raise TypeError("Expected AVRC or callable (n,nfe)->Tensor.")

@torch.no_grad()
def evaluate_model_divergences_nd(model_or_sampler, n=200_000, nfe=8, mmd_max_n=8192, L=128, sw2_max_n=20000):
    x_model = avrc_sample_torch_nd(model_or_sampler, n, nfe)
    x_tgt   = sample_target_torch(n)
    sw2 = sliced_w2(x_model, x_tgt, L=L, max_n=sw2_max_n)
    mmd = mmd_rbf_nd(x_model, x_tgt, max_n=mmd_max_n)
    print(f"[Divergences ND]  SW2≈{sw2:.4f}   MMD≈{mmd:.4f}  (n={n}, K={nfe}, L={L})")
    return sw2, mmd

# ----------------------------- KDE (nd) utilities ------------------------------------
def _fit_kde_nd(X_np: np.ndarray, bw: float | None = None):
    n, d = X_np.shape
    if bw is None:
        std = X_np.std(axis=0, ddof=1) + 1e-8
        bw  = float((n ** (-1.0/(d+4))) * np.mean(std))
        bw  = max(bw, 1e-3)
    kde = KernelDensity(bandwidth=bw, kernel='gaussian'); kde.fit(X_np)
    return kde, bw

def _kde_grid_eval(kde: KernelDensity, xlim, ylim, gridsize=200):
    xs = np.linspace(xlim[0], xlim[1], gridsize)
    ys = np.linspace(ylim[0], ylim[1], gridsize)
    Xg, Yg = np.meshgrid(xs, ys, indexing='xy')
    pts = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
    logp = kde.score_samples(pts).reshape(gridsize, gridsize)
    return Xg, Yg, logp

# ------------------------- Simple wrappers to sample & project -------------------------
@torch.no_grad()
def avrc_sample_torch(model_or_sampler, n: int, nfe: int = 8) -> torch.Tensor:
    # keep exported name but route to ND
    return avrc_sample_torch_nd(model_or_sampler, n=n, nfe=nfe)

# -------------------------- nd heatmap of projected samples ---------------------------
@torch.no_grad()
def plot_output_heatmaps_proj(
    model_or_sampler, P: torch.Tensor, n=200_000, nfe=8, bins=180, target_n=None,
    xlim=None, ylim=None, cmap="magma", title="Model vs Target (proj)",
    gamma: float = 0.42, vmax_percentile: float = 99.7
):
    import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
    from matplotlib.colors import PowerNorm

    x_model = avrc_sample_torch(model_or_sampler, n=n, nfe=nfe)
    xm2 = project_nd(x_model, P).detach().cpu().numpy()
    x_tgt = sample_target_torch(n if target_n is None else target_n)
    xt2 = project_nd(x_tgt, P).detach().cpu().numpy()

    if xlim is None or ylim is None:
        q = 0.997
        xlim = xlim or (np.quantile(xt2[:,0], 1-q), np.quantile(xt2[:,0], q))
        xlim = [val * 1.2 for val in xlim]
        ylim = ylim or (np.quantile(xt2[:,1], 1-q), np.quantile(xt2[:,1], q))
        ylim = [val * 1.2 for val in ylim]
        xlim, ylim = _square_limits(xlim, ylim)


    y_edges = np.linspace(xlim[0], xlim[1], bins+1)
    z_edges = np.linspace(ylim[0], ylim[1], bins+1)

    Ht, *_ = np.histogram2d(xt2[:,0], xt2[:,1], bins=[y_edges, z_edges], density=True)
    Hm, *_ = np.histogram2d(xm2[:,0], xm2[:,1], bins=[y_edges, z_edges], density=True)

    all_vals = np.concatenate([Ht.ravel(), Hm.ravel()])
    vmax = np.percentile(all_vals, vmax_percentile)
    norm = PowerNorm(gamma=max(1e-3, float(gamma)), vmin=0.0, vmax=max(1e-9, vmax))

    try: plt.close('all')
    except Exception: pass
    mpl.rcdefaults()
    with plt.rc_context({
        "figure.facecolor": "black",
        "axes.facecolor":   "black",
        "savefig.facecolor":"black",
        "axes.edgecolor":   "white",
        "axes.labelcolor":  "white",
        "xtick.color":      "white",
        "ytick.color":      "white",
    }):
        fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=True, sharey=True)
        ax = axs[0]
        ax.imshow(Ht.T, origin="lower", extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                  cmap=cmap, norm=norm, interpolation="bilinear", alpha=1.0, aspect="equal")
        ax.set_title("target", color='w', pad=3)
        ax.set_xlabel("proj x", color='w'); ax.set_ylabel("proj y", color='w')
        ax.tick_params(color='w', labelcolor='w')

        ax = axs[1]
        ax.imshow(Hm.T, origin="lower", extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                  cmap=cmap, norm=norm, interpolation="bilinear", alpha=1.0, aspect="equal")
        ax.set_title(f"model (K={nfe})", color='w', pad=3)
        ax.set_xlabel("proj x", color='w')
        ax.tick_params(color='w', labelcolor='w')

        fig.suptitle(title, y=0.995, color='w')
        fig.tight_layout()
        plt.show()



# ---------------------------- 3D crossings: heatmap planes + cords ----------------------------
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (activates 3D proj)


# ----------------------------- Benchmark grid (3-model, timed) ------------------------
def _sample_in_chunks(sampler, n: int, nfe: int, chunk_n: int):
    """
    Call sampler in chunks to avoid big allocations; returns concatenated Tensor on device.
    """
    outs = []
    done = 0
    while done < n:
        take = min(chunk_n, n - done)
        out = avrc_sample_torch(sampler, n=take, nfe=nfe)  # (take,2)
        outs.append(out)
        done += take
    return torch.cat(outs, dim=0)


def _plot_heat_grid(
    samples_by_nameK: dict, x_tgt: torch.Tensor,
    Ks=(1,2,4,8,16), bins=160, xlim=None, ylim=None, cmap="magma",
    out_dir: str | None = None, prefix: str = "benchnd",
    gamma: float = 0.42, vmax_percentile: float = 99.7
):
    import os, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
    from matplotlib.colors import PowerNorm

    t0 = _tic()
    names = list(samples_by_nameK.keys())
    R, C = len(Ks), 1 + len(names)   # +1 for TARGET column

    x_tgt_np = x_tgt.detach().cpu().numpy()

    # Robust bounds from target (match probe style)
    if xlim is None or ylim is None:
        q = 0.997
        xlim = xlim or (np.quantile(x_tgt_np[:,0], 1-q), np.quantile(x_tgt_np[:,0], q))
        xlim = [val * 1.2 for val in xlim]
        ylim = ylim or (np.quantile(x_tgt_np[:,1], 1-q), np.quantile(x_tgt_np[:,1], q))
        ylim = [val * 1.2 for val in ylim]

    # 👉 force square data extent
    xlim, ylim = _square_limits(xlim, ylim)

    # Shared edges
    y_edges = np.linspace(xlim[0], xlim[1], bins+1)
    z_edges = np.linspace(ylim[0], ylim[1], bins+1)

    Ht, *_ = np.histogram2d(x_tgt_np[:,0], x_tgt_np[:,1], bins=[y_edges, z_edges], density=True)

    try: plt.close('all')
    except Exception: pass
    mpl.rcdefaults()
    with plt.rc_context({
        "figure.facecolor": "black",
        "axes.facecolor":   "black",
        "savefig.facecolor":"black",
        "axes.edgecolor":   "white",
        "axes.labelcolor":  "white",
        "xtick.color":      "white",
        "ytick.color":      "white",
    }):
        fig, axs = plt.subplots(R, C, figsize=(4.3*C, 3.1*R), sharex=True, sharey=True)
        if R == 1 and C == 1:
            axs = np.array([[axs]])
        elif R == 1:
            axs = np.array([axs])
        elif C == 1:
            axs = np.array([[ax] for ax in axs])

        for r, K in enumerate(Ks):
            row_histos = [Ht.ravel()]
            panels = [("target", Ht)]

            for name in names:
                x_model = samples_by_nameK[name][K].detach().cpu().numpy()
                Hm, *_ = np.histogram2d(x_model[:,0], x_model[:,1], bins=[y_edges, z_edges], density=True)
                panels.append((f"{name}", Hm))
                row_histos.append(Hm.ravel())

            vmax = np.percentile(np.concatenate(row_histos), vmax_percentile)
            norm = PowerNorm(gamma=max(1e-3, float(gamma)), vmin=0.0, vmax=max(1e-9, vmax))

            for c, (title, H) in enumerate(panels):
                ax = axs[r, c] if R > 1 else axs[0, c]
                ax.imshow(
                    H.T, origin="lower",
                    extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                    cmap=cmap, norm=norm, interpolation="bilinear", alpha=1.0, aspect="equal"
                )
                if c == 0:
                    ax.set_title("target", color='w', pad=3)
                else:
                    ax.set_title(f"K={K} — {title}", color='w', pad=3)
                if r == R-1: ax.set_xlabel("x", color='w')
                if c == 0:   ax.set_ylabel("y", color='w')
                ax.tick_params(color='w', labelcolor='w')

        fig.suptitle("nd histograms by sampler (target in left column)", y=0.995, color='w')
        fig.tight_layout()

        saved_path = None
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            saved_path = os.path.join(out_dir, f"{prefix}_hists_{R}x{C}.png")
            fig.savefig(saved_path, dpi=170, bbox_inches="tight", facecolor='black')
            print(f"[heat grid saved] {saved_path}")
        plt.show()

    t3 = _tic()
    print(f"[plot_grid] total draw {t3 - t0:.2f}s  {_gpu_mem()}")
    return saved_path if out_dir is not None else None



# ----------------------------- Benchmark via projection (REPLACES) -----------------------------
@torch.no_grad()
def benchmark_samplers_proj(
    name2sampler: dict,
    P: torch.Tensor,
    Ks=(1,2,4,8),
    n=120_000,
    mmd_max_n=8192,
    plot_hists: bool = True,
    hist_bins: int = 160,
    hist_out_dir: str | None = "bench_hists_proj",
    hist_prefix: str = "bench_proj",
    sw2_L: int = 128,
    sw2_max_n: int | None = 20000,
    sample_chunk_n: int = 40000,
    hist_cmap: str = "magma",
    hist_gamma: float = 0.42,
    hist_vmax_percentile: float = 99.7,
):
    import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
    from matplotlib.colors import PowerNorm

    print(f"\n=== Projected Sampling quality ===  n={n}, Ks={tuple(Ks)}, sw2_L={sw2_L}, sw2_max_n={sw2_max_n}")

    x_tgt = sample_target_torch(n)
    Xt2   = project_nd(x_tgt, P).detach().cpu().numpy()
    q = 0.997
    xlim = (np.quantile(Xt2[:,0], 1-q), np.quantile(Xt2[:,0], q)); xlim = [val*1.2 for val in xlim]
    ylim = (np.quantile(Xt2[:,1], 1-q), np.quantile(Xt2[:,1], q)); ylim = [val*1.2 for val in ylim]
    xlim, ylim = _square_limits(xlim, ylim)

    y_edges = np.linspace(xlim[0], xlim[1], hist_bins+1)
    z_edges = np.linspace(ylim[0], ylim[1], hist_bins+1)
    Ht, *_ = np.histogram2d(Xt2[:,0], Xt2[:,1], bins=[y_edges, z_edges], density=True)

    results = {}
    samples_for_plots = {name: {} for name in name2sampler.keys()}

    for name, sampler in name2sampler.items():
        row = name.ljust(34) + " | "
        entry = {}
        for K in Ks:
            x = []
            done = 0
            while done < n:
                take = min(sample_chunk_n, n - done)
                out = avrc_sample_torch_nd(sampler, n=take, nfe=K)   # (take, D)
                x.append(out); done += take
            X = torch.cat(x, dim=0)
            sw2 = sliced_w2(X, x_tgt, L=sw2_L, max_n=sw2_max_n)
            mmd = mmd_rbf_nd(X, x_tgt, max_n=mmd_max_n)
            results.setdefault(name, {})[K] = (sw2, mmd)
            row += f" {sw2:5.3f} {mmd:5.3f} "
            samples_for_plots[name][K] = project_nd(X, P).detach().cpu().numpy()
        print(row); print()

    if plot_hists:
        try: plt.close('all')
        except Exception: pass
        mpl.rcdefaults()
        with plt.rc_context({
            "figure.facecolor": "black",
            "axes.facecolor":   "black",
            "savefig.facecolor":"black",
            "axes.edgecolor":   "white",
            "axes.labelcolor":  "white",
            "xtick.color":      "white",
            "ytick.color":      "white",
        }):
            names = list(samples_for_plots.keys())
            R, C = len(Ks), 1 + len(names)
            fig, axs = plt.subplots(R, C, figsize=(4.3*C, 3.1*R), sharex=True, sharey=True)
            if R == 1 and C == 1: axs = np.array([[axs]])
            elif R == 1: axs = np.array([axs])
            elif C == 1: axs = np.array([[ax] for ax in axs])

            for r, K in enumerate(Ks):
                panels = [("target", Ht)]
                flats  = [Ht.ravel()]
                for name in names:
                    Xm2 = samples_for_plots[name][K]
                    Hm, *_ = np.histogram2d(Xm2[:,0], Xm2[:,1], bins=[y_edges, z_edges], density=True)
                    panels.append((f"{name}", Hm))
                    flats.append(Hm.ravel())
                vmax = np.percentile(np.concatenate(flats), hist_vmax_percentile)
                norm = PowerNorm(gamma=max(1e-3, float(hist_gamma)), vmin=0.0, vmax=max(1e-9, vmax))
                for c, (title, H) in enumerate(panels):
                    ax = axs[r, c] if R > 1 else axs[0, c]
                    ax.imshow(H.T, origin="lower", extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                              cmap=hist_cmap, norm=norm, interpolation="bilinear", alpha=1.0, aspect="equal")
                    if c == 0: ax.set_title("target", color='w', pad=3)
                    else:      ax.set_title(f"K={K} — {title}", color='w', pad=3)
                    if r == R-1: ax.set_xlabel("proj x", color='w')
                    if c == 0:   ax.set_ylabel("proj y", color='w')
                    ax.tick_params(color='w', labelcolor='w')

            fig.suptitle("Projected histograms by sampler (target in left column)", y=0.995, color='w')
            fig.tight_layout()
            if hist_out_dir is not None:
                os.makedirs(hist_out_dir, exist_ok=True)
                saved_path = os.path.join(hist_out_dir, f"{hist_prefix}_hists_{R}x{C}.png")
                fig.savefig(saved_path, dpi=170, bbox_inches="tight", facecolor='black')
                print(f"[heat grid saved] {saved_path}")
            plt.show()

    print("[bench] completed benchmark_samplers_proj.")
    return results


# --------------------------------- Chords helpers -------------------------------------
@torch.no_grad()
def chords_pairs_nd(model: AVRC, n=4096, mode="encoder"):
    x1 = sample_target_torch(n)
    if mode == "encoder":
        mu0, _ = model.Enc(x1); x_ref = mu0
    elif mode == "gauss":
        x_ref = torch.randn(n, 2, device=device, dtype=TDTYPE)
    else:
        raise ValueError("mode must be 'encoder' or 'gauss'")
    return x_ref, x1

def main1(target="checker", embedding="identity", K=2):
    """
    Train in ambient dimension K by embedding the intrinsic 2-D target using `embedding`.
    Valid embeddings: {"identity","linear","sine_wiggle","rff","radial"}.
    """
    set_target(target)
    set_embedding(embedding, K)

    avrc = AVRC(AVRCConfig(
        D=K,                          # <--- ambient train dimension
        init_default='gauss',
        rounds=400,
        critic_adapt_max=1000,
        post_anneal_rounds=300,
        batch=4096,
        pretrain_steps=5000,
        recon_k=16, recon_n=8192,
        agg_kl_batch=65536,
        log_every=1,
        k_plot=5,
        test_rf_every=20,
        viz_camera=(10, -40),
        lam_disp_start=1.0, lam_disp_end=1.0,
        lam_align_start=0.0, lam_align_end=1.0,
        enc_lr_decay=.975,
        viz_latent=True,
        viz_proj_mode="pca",
        viz_num_projections=2,
        viz_proj_source="target",
        jitter_lambda = .99
    ))
    avrc.train(progress=True, seed=0)
    print(f'USING jitter = {jitter_lambda}, disp_anneal = {[lam_disp_start, lam_disp_end]}, align_anneal = {[lam_align_start, lam_align_end]}, a_rounds = {rounds - post_anneal_rounds}, rounds = {rounds}')

    rf_model, rf_sampler = train_rectified_flow_nd(
        D=K,                                   # <--- ambient dimension
        steps= avrc.cfg.pretrain_steps,
        batch= avrc.cfg.batch,
        lr=1e-3,
        clip=1.0,
        log_every=5000,
        hidden=avrc.cfg.vel_hidden,
        depth=avrc.cfg.vel_depth,
        midpoints_K=(avrc.cfg.critic_match_rf_K or avrc.cfg.recon_k),
        weight_decay=avrc.cfg.critic_weight_decay,
        seed=avrc.cfg.test_rf_seed
    )

    avrc_oracle = avrc_oracle_sampler_torch_nd(avrc)

    # choose one cached projection for quick single-view plots
    P = avrc._viz_proj_pairs[0] if avrc._viz_proj_pairs else choose_projection_pairs(K, num_pairs=1)[0]

    samplers = {
        "AVRC (q(z) init)": avrc_oracle,
        "AVRC (N→* joint)": avrc,
        "Rectified Flow":   rf_sampler,
    }
    benchmark_samplers_proj(samplers, P=P, Ks=(2,4,8), n=100_000)
    plot_output_heatmaps_proj(avrc,        P=P, n=150_000, nfe=8, title=f"AVRC (N→*) on '{TARGET}' [{embedding}, K={K}]")
    plot_output_heatmaps_proj(avrc_oracle, P=P, n=150_000, nfe=8, title=f"AVRC (q(z) init) on '{TARGET}' [{embedding}, K={K}]")
    plot_output_heatmaps_proj(rf_sampler,  P=P, n=150_000, nfe=8, title=f"Rectified Flow on '{TARGET}' [{embedding}, K={K}]")


import os, re, glob, shutil, subprocess, sys
from pathlib import Path

def numeric_key(path: str):
    import os, re
    base = os.path.basename(path)
    # match patterns like cross_00010.png or cross_00010_p01.png
    m = re.search(r'_(\d+)(?:_p(\d+))?\.png$', base)
    if m:
        round_num = int(m.group(1))
        proj_num  = int(m.group(2)) if m.group(2) is not None else -1
        return (round_num, proj_num, base.lower())
    # generic fallback: first number, then name
    nums = re.findall(r'(\d+)', base)
    return (int(nums[0]) if nums else float('inf'), -1, base.lower())


def ensure_dir(p: Path):
    if p.exists():
        # fully clear any previous staging dir
        for child in p.iterdir():
            try:
                if child.is_symlink() or child.is_file():
                    child.unlink()
                else:
                    shutil.rmtree(child)
            except FileNotFoundError:
                pass
    else:
        p.mkdir(parents=True, exist_ok=True)

def link_or_copy(src: Path, dst: Path):
    try:
        os.symlink(os.path.realpath(src), dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)

def build_sequence(src_glob: str, stage_dir: Path) -> int:
    files = sorted(glob.glob(src_glob), key=numeric_key)
    files = [Path(f) for f in files if Path(f).is_file()]
    if not files:
        print(f"[SKIP] No files matched: {src_glob}")
        return 0

    ensure_dir(stage_dir)

    # Create contiguous sequence frame_000000.png, frame_000001.png, ...
    count = 0
    for i, f in enumerate(files):
        dst = stage_dir / f"frame_{i:06d}.png"
        link_or_copy(f, dst)
        count += 1

    # If only one frame, duplicate last so the clip isn't zero-length
    if count == 1:
        dst2 = stage_dir / f"frame_{1:06d}.png"
        link_or_copy(files[0], dst2)
        count = 2

    print(f"[OK] Sequenced {count} frames → {stage_dir}/frame_%06d.png")
    # Show a couple of examples for sanity
    head = list(sorted(stage_dir.glob("frame_*.png")))[:3]
    tail = list(sorted(stage_dir.glob("frame_*.png")))[-3:]
    if head:
        print("  first frames:", [p.name for p in head])
    if tail and len(tail) != len(head):
        print("  last  frames:", [p.name for p in tail])
    return count

import shutil, cv2
from pathlib import Path
import numpy as np

def encode_from_stage(stage_dir: Path, out_path: Path, fps: int):
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(Path(stage_dir) / "frame_%06d.png"),  # <-- was "%06d.png"
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out_path),
        ]
        print("[CMD]", " ".join(cmd))
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(proc.stdout)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {out_path}")
        return

    # Fallback: OpenCV VideoWriter
    imgs = sorted(Path(stage_dir).glob("*.png"))
    if not imgs:
        print(f"[warn] No frames found in {stage_dir}")
        return
    first = cv2.imread(str(imgs[0]))
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {imgs[0]}")
    h, w = first.shape[:2]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError("VideoWriter failed to open (mp4v).")
    for p in imgs:
        im = cv2.imread(str(p))
        if im is None:
            continue
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        vw.write(im)
    vw.release()
    print(f"[ok] Wrote {out_path} ({len(imgs)} frames @ {fps} fps)")

def make_movie(name: str, src_glob: str, out_path: str, fps: int):
    print(f"\n=== {name} ===")
    stage_dir = Path(out_path).with_suffix("").parent / (".seq_" + Path(out_path).stem)
    count = build_sequence(src_glob, stage_dir)
    if count == 0:
        return
    encode_from_stage(stage_dir, Path(out_path), fps)
    # Clean up staging to avoid clutter; set to False to keep for debugging
    cleanup = True
    if cleanup:
        try:
            shutil.rmtree(stage_dir)
        except Exception as e:
            print(f"[WARN] Could not remove stage dir {stage_dir}: {e}")


# In[33]:


# Make all training plots from a log.txt file with the session's format
# - Place log.txt in the current directory (or use common alternative names)
# - Outputs go to training_plots/, plus a training_plots.zip for download.

import os, re, sys, io, csv, json, math, shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Simplified log file finder - no Colab dependencies needed

# Matplotlib headless setup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- small IO helpers ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_text(path: str) -> list[str]:
    """Robust text reader: tries common encodings and falls back with replacement."""
    encs = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    for enc in encs:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            raise
    # Last resort: decode with replacement
    with open(path, "rb") as f:
        data = f.read()
    return data.decode("utf-8", errors="replace").splitlines(True)

def find_log_file(default_name: str = "log.txt") -> str:
    """
    Find a log file in the current directory.
    Looks for the default name first, then tries common alternatives.
    """
    p = Path(default_name)
    if p.exists():
        return str(p)

    # try some common names
    alts = ["log (1).txt", "logfile.txt", "training.log", "output.log"]
    for a in alts:
        if Path(a).exists():
            return a

    # Provide helpful error message
    available_files = [f.name for f in Path(".").glob("*.txt") if f.is_file()]
    if available_files:
        print(f"Available .txt files: {available_files}")
        print(f"Please rename one of them to '{default_name}' or specify the correct path.")

    raise FileNotFoundError(
        f"Could not find '{default_name}'. "
        "Please place your log file in the current directory "
        "and ensure it has one of these names: log.txt, log (1).txt, logfile.txt, training.log, output.log"
    )

# ---------- parsing ----------
FLOAT = r"[-+]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][-+]?\d+)?"

def parse_pretrain(lines: List[str]) -> List[Dict[str, Any]]:
    out = []
    rx = re.compile(r"\[pretrain v \(indep\)\]\s+(\d+)/(\d+)\s+loss=(" + FLOAT + ")")
    for ln in lines:
        m = rx.search(ln)
        if m:
            out.append({
                "step": int(m.group(1)),
                "total": int(m.group(2)),
                "loss": float(m.group(3)),
            })
    return out

def parse_periodic_round_lines(lines: List[str]) -> List[Dict[str, Any]]:
    out = []
    start = re.compile(r"^\[(\d{5})\]\s+critic\s+(" + FLOAT + r")(?:\s+\(steps=(\d+)\))?")
    for ln in lines:
        m = start.search(ln)
        if not m:
            continue
        d: Dict[str, Any] = {}
        d["round"] = int(m.group(1))
        d["critic"] = float(m.group(2))
        d["steps"] = int(m.group(3)) if m.group(3) else None

        def grab(rx: str, cast=float) -> Optional[float]:
            mm = re.search(rx, ln)
            return cast(mm.group(1)) if mm else None

        d["disp"]  = grab(r"\|\s*disp\s+(" + FLOAT + r")")
        d["align"] = grab(r"\s+align\s+(" + FLOAT + r")")
        d["kl"]    = grab(r"\s+kl\s+(" + FLOAT + r")")

        d["lam_disp"]  = grab(r"\|\s*lam_disp\s+(" + FLOAT + r")")
        d["lam_align"] = grab(r"\s+lam_align\s+(" + FLOAT + r")")
        d["lam_kl"]    = grab(r"\s+lam_kl\s+(" + FLOAT + r")")

        # SW2/MMD @ log k: N→*
        mN = re.search(r"SW2@k=(\d+)\s+N[^:]*:\s*(" + FLOAT + r")\s+MMD:\s*(" + FLOAT + r")", ln)
        if mN:
            d["sw2_logk_N"] = float(mN.group(2))
            d["mmd_logk_N"] = float(mN.group(3))
            d["log_k"]      = int(mN.group(1))

        # SW2/MMD @ log k: E→*
        mE = re.search(r"SW2@k=\d+\s+E[^:]*:\s*(" + FLOAT + r")\s+MMD:\s*(" + FLOAT + r")", ln)
        if mE:
            d["sw2_logk_E"] = float(mE.group(1))
            d["mmd_logk_E"] = float(mE.group(2))

        # AGG-KL, mu-norm, avg-var
        mA = re.search(r"AGG-KL≈(" + FLOAT + r")\s+\(\|\|[μu]?\|\|≈(" + FLOAT + r"),\s+avg var≈(" + FLOAT + r")\)", ln)
        if mA:
            d["agg_kl"]  = float(mA.group(1))
            d["mu_norm"] = float(mA.group(2))
            d["avg_var"] = float(mA.group(3))

        # Flow recon
        mF = re.search(r"FlowRecon@K=(\d+)\s+MSE≈(" + FLOAT + r")", ln)
        if mF:
            d["flow_K"]   = int(mF.group(1))
            d["flow_mse"] = float(mF.group(2))

        out.append(d)
    return out

def parse_dispersion_endpoints(lines: List[str]) -> List[Dict[str, Any]]:
    out = []
    rx_r   = re.compile(r"\[dispersion\]\s+r=(\d+)")
    rx_d0  = re.compile(r"d\(0\)=(" + FLOAT + r")")
    rx_d1  = re.compile(r"d\(1\)=(" + FLOAT + r")")
    rx_ell = re.compile(r"E\|\|.*?\|\|=(" + FLOAT + r")")
    for ln in lines:
        mr = rx_r.search(ln)
        if not mr:
            continue
        rd = {"round": int(mr.group(1))}
        m0 = rx_d0.search(ln)
        m1 = rx_d1.search(ln)
        me = rx_ell.search(ln)
        if m0: rd["d0"] = float(m0.group(1))
        if m1: rd["d1"] = float(m1.group(1))
        if me: rd["ell_norm"] = float(me.group(1))
        out.append(rd)
    return out

def parse_probe_tables(lines: List[str]) -> List[Dict[str, Any]]:
    """
    Parse any probe-style tabular block:
      'model/init | K= 2  SW2  MMD  K= 4  SW2  MMD ...'
    Returns tidy rows: round, model, init, K, SW2, MMD.
    """
    tidy = []
    current_probe_round: Optional[int] = None
    ks: List[int] = []

    probe_round_rx = re.compile(r"===\s*probe:.*@r(\d+)\s*===")
    header_rx      = re.compile(r"^\s*model/init\s*\|.*K\s*=\s*\d+")
    k_find_rx      = re.compile(r"K\s*=\s*(\d+)")

    i = 0
    while i < len(lines):
        ln = lines[i]

        pr = probe_round_rx.search(ln)
        if pr:
            current_probe_round = int(pr.group(1))

        if header_rx.search(ln):
            ks = [int(x) for x in k_find_rx.findall(ln)]
            j = i + 1
            while j < len(lines) and set(lines[j].strip()) <= set("-| "):
                j += 1

            for _ in range(12):  # guard
                if j >= len(lines): break
                row = lines[j].rstrip("\n")
                if not row or row.startswith("[") or row.startswith("==="):
                    break
                if "/" not in row or "|" not in row:
                    break
                left, right = row.split("|", 1)
                label = left.strip()
                if "/" not in label:
                    break
                model, init = [t.strip() for t in label.split("/", 1)]
                nums = re.findall(FLOAT, right)
                if len(nums) >= 2 * len(ks):
                    for k_idx, K in enumerate(ks):
                        sw2 = float(nums[2*k_idx])
                        mmd = float(nums[2*k_idx + 1])
                        tidy.append({
                            "round": current_probe_round,  # None if not inside a probe stanza
                            "model": model,
                            "init": init,
                            "K": K,
                            "SW2": sw2,
                            "MMD": mmd
                        })
                j += 1
            i = j
            continue
        i += 1

    return tidy

def extract_last_comparison_table(lines: List[str]) -> Optional[Dict[str, Any]]:
    """
    Return the LAST table block (header + rows) as structured data,
    even if it's outside a 'probe' banner.
    """
    header_rx = re.compile(r"^\s*model/init\s*\|.*K\s*=\s*\d+")
    k_find_rx = re.compile(r"K\s*=\s*(\d+)")
    blocks = []
    i = 0
    while i < len(lines):
        if header_rx.search(lines[i]):
            ks = [int(x) for x in k_find_rx.findall(lines[i])]
            j = i + 1
            while j < len(lines) and set(lines[j].strip()) <= set("-| "):
                j += 1
            rows = []
            # read a handful of lines after header, keep those that look like data
            for _ in range(12):
                if j >= len(lines): break
                row = lines[j].strip("\n")
                if not row or row.startswith("[") or row.startswith("==="):
                    break
                if "/" in row and "|" in row:
                    rows.append(row)
                j += 1
            if rows:
                blocks.append((i, ks, rows))
            i = j
        else:
            i += 1
    if not blocks:
        return None
    _, ks, rows = blocks[-1]
    parsed = []
    for row in rows:
        if "|" not in row or "/" not in row:
            continue
        left, right = row.split("|", 1)
        label = left.strip()
        if "/" not in label:
            continue
        model, init = [t.strip() for t in label.split("/", 1)]
        nums = re.findall(FLOAT, right)
        if len(nums) < 2*len(ks):
            continue
        for k_idx, K in enumerate(ks):
            sw2 = float(nums[2*k_idx])
            mmd = float(nums[2*k_idx + 1])
            parsed.append({"model": model, "init": init, "K": K, "SW2": sw2, "MMD": mmd})
    return {"K": ks, "rows": parsed}

# ---------- plotting ----------
def simple_curve(xs, ys, xlabel, ylabel, title, outpath, legend=None):
    plt.figure(figsize=(7.0, 4.2), dpi=140)
    if isinstance(ys, dict):
        for key, vals in ys.items():
            if xs and vals:
                plt.plot(xs, vals, label=str(key), linewidth=2)
        if legend is None:
            legend = True
    else:
        plt.plot(xs, ys, linewidth=2)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title); plt.grid(alpha=0.3)
    if legend: plt.legend(loc="best", fontsize=9)
    plt.tight_layout(); plt.savefig(outpath); plt.close()

def table_to_png(table_rows: List[Dict[str, Any]], ks: List[int], outpath: Path, title: str):
    cols = ["model/init"] + [f"K={k} SW2 | MMD" for k in ks]
    order = [("RF", "gauss"), ("RF", "encoder"), ("V", "gauss"), ("V", "encoder")]
    grid: List[List[str]] = []
    for m, init in order:
        row = [f"{m}/{init}"]
        for K in ks:
            hit = [r for r in table_rows if r["model"]==m and r["init"]==init and r["K"]==K]
            if hit:
                row.append(f"{hit[0]['SW2']:.3f} | {hit[0]['MMD']:.3f}")
            else:
                row.append("—")
        grid.append(row)

    fig, ax = plt.subplots(figsize=(min(14, 4 + 1.3*len(ks)), 1 + 0.6*len(grid)), dpi=150)
    ax.axis("off")
    tab = ax.table(cellText=grid, colLabels=cols, loc="center", cellLoc="center")
    tab.auto_set_font_size(False); tab.set_fontsize(9)
    tab.scale(1.1, 1.3)
    ax.set_title(title, pad=12)
    plt.tight_layout()
    plt.savefig(outpath); plt.close()

# ---------- orchestration ----------
def make_all_plots(log_path: str, out_dir: str = "training_plots") -> Path:
    outdir = Path(out_dir)
    ensure_dir(outdir)
    lines = read_text(log_path)

    pretrain = parse_pretrain(lines)
    rounds   = parse_periodic_round_lines(lines)
    disper   = parse_dispersion_endpoints(lines)
    probe_tidy = parse_probe_tables(lines)
    last_tbl = extract_last_comparison_table(lines)

    # Pretrain
    if pretrain:
        xs = [r["step"] for r in pretrain]
        ys = [r["loss"] for r in pretrain]
        simple_curve(xs, ys, "pretrain step", "loss",
                     "Pretrain velocity loss",
                     outdir/"pretrain_loss.png")

    # Periodic training lines
    if rounds:
        xs = [r["round"] for r in rounds]
        simple_curve(xs, [r["critic"] for r in rounds], "round", "critic loss",
                     "Critic loss vs round", outdir/"training_critic_loss.png")
        ys = {
            "disp":  [r.get("disp")  for r in rounds],
            "align": [r.get("align") for r in rounds],
            "kl":    [r.get("kl")    for r in rounds],
        }
        simple_curve(xs, ys, "round", "value",
                     "Dispersion / Alignment / KL vs round",
                     outdir/"training_disp_align_kl.png", legend=True)
        ys = {
            "lam_disp":  [r.get("lam_disp")  for r in rounds],
            "lam_align": [r.get("lam_align") for r in rounds],
            "lam_kl":    [r.get("lam_kl")    for r in rounds],
        }
        simple_curve(xs, ys, "round", "λ",
                     "Lambda schedules vs round",
                     outdir/"training_lambda_schedules.png", legend=True)

        if any(r.get("sw2_logk_N") is not None for r in rounds):
            simple_curve(xs, {
                "SW2 N→*": [r.get("sw2_logk_N") for r in rounds],
                "SW2 E→*": [r.get("sw2_logk_E") for r in rounds],
            }, "round", "SW2", "SW2 at log-k vs round", outdir/"training_sw2_logk_over_rounds.png", legend=True)
        if any(r.get("mmd_logk_N") is not None for r in rounds):
            simple_curve(xs, {
                "MMD N→*": [r.get("mmd_logk_N") for r in rounds],
                "MMD E→*": [r.get("mmd_logk_E") for r in rounds],
            }, "round", "MMD", "MMD at log-k vs round", outdir/"training_mmd_logk_over_rounds.png", legend=True)

        if any(r.get("agg_kl") is not None for r in rounds):
            simple_curve(xs, [r.get("agg_kl") for r in rounds], "round", "AGG-KL",
                         "Aggregate KL vs round", outdir/"agg_kl_over_rounds.png")
        if any(r.get("mu_norm") is not None for r in rounds):
            simple_curve(xs, [r.get("mu_norm") for r in rounds], "round", "||μ||",
                         "Mean norm vs round", outdir/"mu_norm_over_rounds.png")
        if any(r.get("avg_var") is not None for r in rounds):
            simple_curve(xs, [r.get("avg_var") for r in rounds], "round", "avg var",
                         "Average latent variance vs round", outdir/"avg_var_over_rounds.png")
        if any(r.get("flow_mse") is not None for r in rounds):
            simple_curve(xs, [r.get("flow_mse") for r in rounds], "round", "MSE",
                         "Flow reconstruction MSE vs round", outdir/"flow_recon_mse_over_rounds.png")

    # Dispersion endpoints & E||ℓ||
    if disper:
        xs = [d["round"] for d in disper]
        d0 = [d.get("d0") for d in disper]
        d1 = [d.get("d1") for d in disper]
        y_max = None
        if d0 and d0[0] is not None:
            y_max = d0[0] * 1.05
        plt.figure(figsize=(7.0, 4.2), dpi=140)
        plt.plot(xs, d0, label="d(0)", linewidth=2)
        plt.plot(xs, d1, label="d(1)", linewidth=2)
        if y_max is not None:
            plt.ylim(0, y_max)
        plt.xlabel("round"); plt.ylabel("dispersion endpoint")
        plt.title("Dispersion endpoints vs round")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(outdir/"dispersion_endpoints_over_rounds.png"); plt.close()

        ell = [d.get("ell_norm") for d in disper]
        simple_curve(xs, ell, "round", "E||ℓ||",
                     "Mean chord length E||ℓ|| vs round",
                     outdir/"ell_norm_over_rounds.png")

    # RF probe multi-line plots (V, init=gauss): MMD & SW2 by K
    tidy_csv_path = outdir/"rf_probe_metrics_all.csv"
    probe_tidy_rows = []
    if probe_tidy:
        with open(tidy_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["round","model","init","K","SW2","MMD"])
            w.writeheader()
            for r in probe_tidy:
                w.writerow(r)
                probe_tidy_rows.append(r)

        vgauss = [r for r in probe_tidy_rows if r["model"]=="V" and r["init"]=="gauss" and r["round"] is not None]
        if vgauss:
            rounds_sorted = sorted(sorted(set(r["round"] for r in vgauss)))
            Ks_sorted = sorted(sorted(set(r["K"] for r in vgauss)))
            y_mmd = {}
            y_sw2 = {}
            for K in Ks_sorted:
                kv = [r for r in vgauss if r["K"]==K]
                kv_by_round = {r["round"]: r for r in kv}
                y_mmd[K] = [kv_by_round[rr]["MMD"] if rr in kv_by_round else None for rr in rounds_sorted]
                y_sw2[K] = [kv_by_round[rr]["SW2"] if rr in kv_by_round else None for rr in rounds_sorted]

            plt.figure(figsize=(7.8, 4.6), dpi=150)
            for K in Ks_sorted:
                plt.plot(rounds_sorted, y_mmd[K], label=f"K={K}", linewidth=2)
            plt.xlabel("round"); plt.ylabel("MMD")
            plt.title("V model (init: Gaussian) — MMD vs round by step count K")
            plt.grid(alpha=0.3); plt.legend(title="Steps", ncol=min(5, len(Ks_sorted)))
            plt.tight_layout(); plt.savefig(outdir/"rf_probe_v_gauss_mmd_over_rounds_byK.png"); plt.close()

            plt.figure(figsize=(7.8, 4.6), dpi=150)
            for K in Ks_sorted:
                plt.plot(rounds_sorted, y_sw2[K], label=f"K={K}", linewidth=2)
            plt.xlabel("round"); plt.ylabel("SW2")
            plt.title("V model (init: Gaussian) — SW2 vs round by step count K")
            plt.grid(alpha=0.3); plt.legend(title="Steps", ncol=min(5, len(Ks_sorted)))
            plt.tight_layout(); plt.savefig(outdir/"rf_probe_v_gauss_sw2_over_rounds_byK.png"); plt.close()

    # Last comparison table (wherever the final block is)
    last_tbl = extract_last_comparison_table(lines)
    if last_tbl and last_tbl.get("rows"):
        with open(outdir/"last_comparison_table.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["model","init","K","SW2","MMD"])
            w.writeheader()
            for r in last_tbl["rows"]:
                w.writerow(r)
        table_to_png(last_tbl["rows"], last_tbl["K"], outdir/"last_comparison_table.png",
                     "Last comparison table: SW2 | MMD")

    return outdir

def run_plots_from_log(log_path: str, out_dir: str = "training_plots"):
    """Generate training_plots from a given log file and zip the folder."""
    outdir = None
    try:
        outdir = make_all_plots(log_path, out_dir=out_dir)
        print(f"[ok] Plots saved in: {outdir.resolve()}")
    except Exception as e:
        print(f"[warn] plotting from log failed: {e}")
        return None

    # Show a few key images inline (best-effort)
    try:
        from IPython.display import display, Image
    except Exception:
        def display(*args, **kwargs):
            return None
        class Image:
            def __init__(self, filename):
                self.filename = filename
    def _show_if_exists(name: str):
        p = outdir / name
        if p.exists():
            display(Image(filename=str(p)))
    _preview = [
        "training_critic_loss.png",
        "training_disp_align_kl.png",
        "training_sw2_logk_over_rounds.png",
        "training_mmd_logk_over_rounds.png",
        "rf_probe_v_gauss_mmd_over_rounds_byK.png",
        "last_comparison_table.png",
    ]
    for n in _preview:
        _show_if_exists(n)

    # Zip for convenience
    zip_path = outdir.with_suffix(".zip")
    shutil.make_archive(outdir.name, "zip", root_dir=outdir)
    print(f"[ok] Zipped folder: {zip_path}")
    print(f"[info] Find your plots in: {outdir.resolve()}")
    print(f"[info] Zipped archive: {zip_path}")
    return outdir


# In[ ]:


import os, sys, shutil, contextlib
from pathlib import Path


def _safe_rmtree(path: str | Path):
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass

def _safe_remove(path: str | Path):
    p = Path(path)
    try:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():  # file, symlink, etc.
            p.unlink(missing_ok=True)
    except Exception:
        pass

def _wipe_work_dirs():
    for d in [
        "viz_crossings3d",
        "viz_rf_snapshots",
        "rf_snapshots",
        "bench_hists_nd",
        "training_plots",
        "training_plots.zip",   # <- in case a previous zip is in the way
    ]:
        _safe_remove(d)


import re

def _run_label(target: str, embedding: str, K: int) -> str:
    """Filesystem-safe label for staging/zips."""
    name = f"{target}__{embedding}_K{int(K)}"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name)



def _stage_and_zip(example: str, log_path: Path, out_root: Path = Path("meta_runs")) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    stage = out_root / f"{example}_bundle"
    _safe_remove(stage)
    stage.mkdir(parents=True, exist_ok=True)

    # copy log
    shutil.copy2(log_path, stage / "log.txt")

    # copy outputs (dirs only; skip if missing)
    for d in ["viz_crossings3d", "rf_snapshots", "bench_hists_nd", "training_plots"]:
        p = Path(d)
        if not p.exists():
            continue
        dst = stage / d
        if p.is_dir():
            shutil.copytree(p, dst)
        else:
            # Be tolerant: if someone dropped a file with that name, just stage it as a file
            shutil.copy2(p, dst)

    # zip
    zip_base = out_root / f"{example}"
    shutil.make_archive(str(zip_base), "zip", root_dir=stage)

    # clean staging
    _safe_remove(stage)
    return zip_base.with_suffix(".zip")




def meta_run_examples(examples: list[tuple[str, str, int]]):
    """
    examples: list of (target, embedding, K) triples, e.g.
      [('checker','linear',3), ('moons','identity',2), ('rings','rff',8)]
    """
    import re, glob

    def _discover_proj_indices(pattern: str) -> list[int]:
        """
        Scan files matching `pattern` (e.g., 'viz_crossings3d/cross_*.png')
        and return sorted unique projection indices extracted from suffix '_pXX.png'.
        """
        files = glob.glob(pattern)
        idxs = set()
        rx = re.compile(r"_p(\d+)\.png$")
        for f in files:
            m = rx.search(f)
            if m:
                try:
                    idxs.add(int(m.group(1)))
                except ValueError:
                    pass
        return sorted(idxs)

    out_root = Path("meta_runs")
    out_root.mkdir(exist_ok=True)

    for ex in examples:
        try:
            target, embedding, K = ex
        except Exception:
            raise ValueError("Each example must be a 3-tuple: (target, embedding, K).")

        label = _run_label(str(target), str(embedding), int(K))
        print(f"\n=== META RUN: target={target} | embedding={embedding} | K={K} ===")
        _wipe_work_dirs()

        # Create ephemeral log.txt in CWD for this run
        log_path = Path("log.txt")
        try:
            log_path.unlink(missing_ok=True)
        except Exception:
            pass

        with open(log_path, "w", encoding="utf-8") as lf, \
             contextlib.redirect_stdout(lf), \
             contextlib.redirect_stderr(lf):
            # Full pipeline per example
            main1(target, embedding=str(embedding), K=int(K))

            # -------- Movies from generated frames (one movie per projection) --------
            # Crossings
            cross_dir = "viz_crossings3d"
            cross_glob_all = f"{cross_dir}/cross_*.png"
            cross_ps = _discover_proj_indices(cross_glob_all)
            if cross_ps:
                for p in cross_ps:
                    make_movie(f"Crossings movie (p={p:02d})",
                               f"{cross_dir}/cross_*_p{p:02d}.png",
                               f"{cross_dir}/crossings_p{p:02d}.mp4", 5)
            else:
                # Fallback (legacy, no projection suffix)
                make_movie("Crossings movie", cross_glob_all, f"{cross_dir}/crossings_evolution.mp4", 5)

            # Latent scatter
            latent_glob_all = f"{cross_dir}/latent_*.png"
            latent_ps = _discover_proj_indices(latent_glob_all)
            if latent_ps:
                for p in latent_ps:
                    make_movie(f"Latent movie (p={p:02d})",
                               f"{cross_dir}/latent_*_p{p:02d}.png",
                               f"{cross_dir}/latent_p{p:02d}.mp4", 5)
            else:
                make_movie("Latent movie", f"{cross_dir}/latent_[0-9]*.png",
                           f"{cross_dir}/latent_evolution.mp4", 5)

            # Latent means (mu)
            mu_glob_all = f"{cross_dir}/latent_mu_*.png"
            mu_ps = _discover_proj_indices(mu_glob_all)
            if mu_ps:
                for p in mu_ps:
                    make_movie(f"Latent means (μ) movie (p={p:02d})",
                               f"{cross_dir}/latent_mu_*_p{p:02d}.png",
                               f"{cross_dir}/latent_mu_p{p:02d}.mp4", 5)
            else:
                make_movie("Latent means (μ) movie", mu_glob_all,
                           f"{cross_dir}/latent_mu_evolution.mp4", 5)

            # Dispersion (no projection)
            make_movie("Dispersion movie",
                       f"{cross_dir}/dispersion_*.png",
                       f"{cross_dir}/dispersion_evolution.mp4", 5)

            # RF probe (may include projection suffixes)
            rf_dir = "rf_snapshots"
            rf_glob_all = f"{rf_dir}/rf_probe_*.png"
            rf_ps = _discover_proj_indices(rf_glob_all)
            if rf_ps:
                for p in rf_ps:
                    make_movie(f"RF probe movie (p={p:02d})",
                               f"{rf_dir}/rf_probe_*_p{p:02d}.png",
                               f"{rf_dir}/rf_probe_p{p:02d}.mp4", 3)
            else:
                make_movie("RF probe movie", rf_glob_all,
                           f"{rf_dir}/rf_probe_evolution.mp4", 3)
            # ------------------------------------------------------------------------

            # Training plots from this run's log.txt
            run_plots_from_log(str(log_path), out_dir="training_plots")

        # Stage outputs and log into meta zip, then clean up
        try:
            zip_path = _stage_and_zip(label, log_path, out_root=out_root)
            print(f"[ok] Saved results to {zip_path}")
        finally:
            try:
                log_path.unlink(missing_ok=True)
            except Exception:
                pass

        # wipe before next
        _wipe_work_dirs()



if __name__ == "__main__":

    import argparse, ast, sys
    parser = argparse.ArgumentParser(
        description="Run VAE_reflownd meta examples and produce zipped outputs."
    )
    parser.add_argument(
        "--examples",
        type=str,
        default="[('checker','identity', 2), ('rings','identity', 2), ('8g','identity', 2), ('spiral','identity', 2), ('moon','identity', 2)]",
        help="Python list of (target, embedding, K) triples."
    )
    # In notebooks, sys.argv includes the kernel’s -f flag; ignore unknowns.
    args, _unknown = parser.parse_known_args()
    

    try:
        raw = ast.literal_eval(args.examples)
        if not isinstance(raw, (list, tuple)):
            raise ValueError

        triples = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                t, e, k = item
                triples.append((str(t), str(e), int(k)))
            elif isinstance(item, str):
                # backward compat: target only → (target, 'identity', 2)
                triples.append((item, "identity", 2))
            else:
                raise ValueError
    except Exception:
        print("[warn] Could not parse --examples; using default [('checker','identity',2)].")
        triples = [('checker','identity',2)]

    meta_run_examples(triples)

