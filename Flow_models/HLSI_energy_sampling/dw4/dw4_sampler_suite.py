# -*- coding: utf-8 -*-
"""
dw4_hlsi_stress_test.py  —  HLSI score estimator stress test on DW-4 variants.

Base DW-4 benchmark: 4 particles in R^2, total dimension d = 8.
  U(x) = α · Σ_i (‖x_i‖² − 1)²  +  c · Σ_{i<j} exp(−‖x_i − x_j‖² / 2σ²)
  log p(x) = −U(x)  (unnormalized)

Target modifications (stress-testing HLSI's dependence on local GMM-ness):
  0. Baseline      : hardened DW-4 (α=8, c=3, σ²=0.0225)
  1. Corrugated    : angular cosine bumps → rapidly varying curvature within mode
  2. Soft radial   : α=0.5 → thick annulus, wide sloppy modes
  3. Banana        : cubic tilt → skewed modes (asymmetric non-Gaussianity)
  4. Heavy-tail    : Lorentzian-saturated well → fat radial tails
  5. Flat-bottomed : dead zone with zero curvature → locally uniform support

Ground-truth samples: MALA long-chain MCMC on parallel chains.
Scores / Hessians: autodiff via torch.func (vmap + grad / hessian).
Score reference for RMSE: high-N SNIS-Tweedie from a large reference pool.

Methods compared
────────────────
  Tweedie          : SNIS-weighted Tweedie denoiser
  Blended          : Per-coordinate soft blend of Tweedie & TSI
  HLSI / CE-HLSI / Leaf-CE-HLSI : original HLSI-family estimators
  Gamma-HLSI / Leaf-Gamma-HLSI   : Lambda-derived inverse-gate variants
  Surrogate-*                    : same gates with local-surrogate transition weights
"""

# ==============================================================
#  CONFIG — set before running
# ==============================================================
#   'single'  : baseline hardened DW-4 only (quick sanity check)
#   'full'    : all 6 stress-test variants
RUN_MODE = 'full'

# ==============================================================

try:
    import google.colab  # noqa: F401
    _IN_COLAB = True
except ImportError:
    _IN_COLAB = False

if _IN_COLAB:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', 'POT'])

import os
import math
import time
from collections import OrderedDict
import torch
import numpy as np
import matplotlib
if not _IN_COLAB:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.func import vmap, grad as fgrad, hessian as fhessian

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

# Use CUDA automatically when available.  Override with, e.g.,
#   TORCH_DEVICE=cpu python dw4_sampler_suite_gpu_updated.py
# or
#   TORCH_DEVICE=cuda:0 python dw4_sampler_suite_gpu_updated.py
DEVICE = torch.device(os.environ.get('TORCH_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(42)
print(f"Using device: {DEVICE}")

def _close_fig(fig):
    """Save-compatible close: show inline in Colab, then close."""
    if _IN_COLAB:
        plt.show()
    plt.close(fig)

P_LEAF = None   # Leaf-HLSI within-leaf precision for rescued non-PSD directions
D_TARGET = 8    # total dimension (4 particles × 2D)

# ==============================================================
# DW-4 target (base class)
# ==============================================================
class DW4Target:
    """
    4-particle double-well system in 2D.  x ∈ R^8.

    Energy:
      U(x) = α · Σ_i (‖x_i‖² − 1)²            (double-well per particle)
            + c · Σ_{i<j} exp(−‖x_i−x_j‖²/2σ²) (pairwise Gaussian repulsion)
    """
    N_PARTICLES = 4
    D_PARTICLE  = 2
    D           = 8

    def __init__(self, alpha=1.0, repul_c=1.0, repul_sigma=0.5):
        self.alpha = float(alpha)
        self.c     = float(repul_c)
        self.sig2  = float(repul_sigma) ** 2
        self._pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        self.variant_name = 'base'
        self.variant_desc = ''

    def _base_energy(self, x):
        """Base DW-4 energy (without modifications). Returns [B]."""
        r   = x.reshape(-1, 4, 2)
        r2  = (r * r).sum(-1)
        U   = self.alpha * ((r2 - 1.0) ** 2).sum(-1)
        diffs = r.unsqueeze(2) - r.unsqueeze(1)
        d2    = (diffs * diffs).sum(-1)
        for i, j in self._pairs:
            U = U + self.c * torch.exp(-d2[:, i, j] / (2.0 * self.sig2))
        return U

    def _modifier_energy(self, x):
        """Override in subclasses to add energy modifications. Returns [B]."""
        return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

    def log_prob(self, x):
        return -(self._base_energy(x) + self._modifier_energy(x))

    def score(self, x):
        x_ = x.detach().requires_grad_(True)
        with torch.enable_grad():
            self.log_prob(x_).sum().backward()
        return x_.grad.detach().clone()

    def hessian(self, x):
        def neg_lp_scalar(xi):
            return -self.log_prob(xi.unsqueeze(0)).squeeze(0)
        H_fn = vmap(fhessian(neg_lp_scalar))
        return H_fn(x.detach())

    def hessian_fd(self, x, eps=1e-4):
        """Symmetric finite-difference approximation of −∇² log p."""
        B, d = x.shape
        H = torch.zeros(B, d, d, dtype=x.dtype, device=x.device)
        for j in range(d):
            e = torch.zeros(d, dtype=x.dtype, device=x.device)
            e[j] = eps
            sp = self.score(x + e)
            sm = self.score(x - e)
            H[:, :, j] = -(sp - sm) / (2.0 * eps)
        return 0.5 * (H + H.transpose(-1, -2))

    def sample_mala(self, n, step_size=0.025, n_chains=32,
                    burnin=60_000, thin=40, verbose=True):
        d   = self.D
        x   = torch.randn(n_chains, d, dtype=torch.get_default_dtype(), device=DEVICE) * 0.5
        accept_sum, total = 0, 0
        samples = []
        needed_post = math.ceil(n / n_chains) * thin + 1

        def lp_score(xin):
            xin_ = xin.detach().requires_grad_(True)
            with torch.enable_grad():
                lp = self.log_prob(xin_)
                lp.sum().backward()
            return lp.detach(), xin_.grad.detach().clone()

        lp_x, sx = lp_score(x)
        if verbose:
            print(f"  MALA: {n_chains} chains, burnin={burnin:,}, "
                  f"thin={thin}, step_size={step_size}")

        for step in range(burnin + needed_post):
            noise   = torch.randn_like(x)
            x_prop  = x + step_size * sx + math.sqrt(2.0 * step_size) * noise
            lp_prop, sx_prop = lp_score(x_prop)
            log_q_fwd = -((x_prop - x  - step_size * sx     ) ** 2).sum(-1) / (4.0 * step_size)
            log_q_bwd = -((x      - x_prop - step_size * sx_prop) ** 2).sum(-1) / (4.0 * step_size)
            log_alpha = (lp_prop - lp_x + log_q_bwd - log_q_fwd).clamp(max=0.0)
            acc  = torch.rand(n_chains, dtype=x.dtype, device=x.device) < log_alpha.exp()
            mask = acc.unsqueeze(-1)
            x    = torch.where(mask, x_prop,  x)
            lp_x = torch.where(acc,  lp_prop, lp_x)
            sx   = torch.where(mask, sx_prop, sx)
            if step >= burnin:
                accept_sum += acc.sum().item()
                total      += n_chains
                if (step - burnin) % thin == 0:
                    samples.append(x.clone())
                    if len(samples) * n_chains >= n:
                        break
        if verbose and total > 0:
            print(f"  MALA acceptance (post-burnin): {100.0 * accept_sum / total:.1f}%")
        out = torch.cat(samples, dim=0)
        return out[torch.randperm(len(out), device=out.device)[:n]]

    @staticmethod
    def particles(x):
        return x.reshape(-1, 4, 2)

    @staticmethod
    def all_coords(x):
        return x.reshape(-1, 4, 2).reshape(-1, 2)


# ==============================================================
# Target modifications (subclasses that override _modifier_energy)
# ==============================================================

class CorrugatedDW4(DW4Target):
    """
    Mod 1: Angular corrugation.
    Adds β · Σ_i cos(k · θ_i)  where θ_i = atan2(x_{i,2}, x_{i,1}).
    Creates k bumps per ring with curvature swinging ±β·k² along the arc.
    Tests: does HLSI degrade when curvature varies rapidly within an
    SNIS-dominant neighborhood?
    """
    def __init__(self, beta=3.0, k=4, **kwargs):
        super().__init__(**kwargs)
        self.beta = float(beta)
        self.k    = int(k)
        self.variant_name = 'corrugated'
        self.variant_desc = f'β={self.beta}, k={self.k}'

    def _modifier_energy(self, x):
        r     = x.reshape(-1, 4, 2)
        theta = torch.atan2(r[..., 1], r[..., 0])        # [B, 4]
        return self.beta * torch.cos(self.k * theta).sum(-1)


class SoftRadialDW4(DW4Target):
    """
    Mod 2: Soft radial confinement (just lower α).
    Shallow wells → thick annulus, wide sloppy modes.
    Tests: does performance degrade when mode width exceeds the
    Hessian's radius of validity?
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.variant_name = 'soft_radial'
        self.variant_desc = f'α={self.alpha}'


class BananaDW4(DW4Target):
    """
    Mod 3: Banana distortion.
    Adds γ · Σ_i x_{i,1} · (‖x_i‖² − 1).
    Cubic-in-displacement tilt bends each ring into a kidney/banana shape.
    Produces asymmetric non-Gaussianity (skewness) that the symmetric
    two-leaf ansatz cannot capture.
    Tests: does the symmetric two-leaf ansatz fail when dominant
    non-Gaussianity is skewness rather than bimodality?
    """
    def __init__(self, gamma=3.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = float(gamma)
        self.variant_name = 'banana'
        self.variant_desc = f'γ={self.gamma}'

    def _modifier_energy(self, x):
        r  = x.reshape(-1, 4, 2)
        r2 = (r * r).sum(-1)
        return self.gamma * (r[..., 0] * (r2 - 1.0)).sum(-1)


class HeavyTailDW4(DW4Target):
    """
    Mod 4: Heavy-tail radial modification.
    Replaces quartic well α(r²−1)² with α·(r²−1)²/(1 + ν·(r²−1)²).
    Matches quartic near the ring but saturates to α/ν at large displacement,
    allowing occasional large radial excursions.
    Tests: does HLSI degrade when the target has heavier tails than
    the local Gaussian predicts?
    """
    def __init__(self, nu=0.5, **kwargs):
        super().__init__(**kwargs)
        self.nu = float(nu)
        self.variant_name = 'heavy_tail'
        self.variant_desc = f'ν={self.nu}'

    def _base_energy(self, x):
        """Override base energy to use Lorentzianized well."""
        r   = x.reshape(-1, 4, 2)
        r2  = (r * r).sum(-1)
        quartic = (r2 - 1.0) ** 2
        U   = self.alpha * (quartic / (1.0 + self.nu * quartic)).sum(-1)
        diffs = r.unsqueeze(2) - r.unsqueeze(1)
        d2    = (diffs * diffs).sum(-1)
        for i, j in self._pairs:
            U = U + self.c * torch.exp(-d2[:, i, j] / (2.0 * self.sig2))
        return U


class FlatBottomDW4(DW4Target):
    """
    Mod 5: Flat-bottomed well.
    Replaces quartic with α·max(0, |r²−1| − δ)².
    Creates an annular dead zone of width δ where energy is zero (Hessian
    is zero in the radial direction → locally uniform, maximally non-Gaussian).
    Tests: does HLSI add value when parts of the support have zero curvature?
    """
    def __init__(self, delta=0.3, **kwargs):
        super().__init__(**kwargs)
        self.delta = float(delta)
        self.variant_name = 'flat_bottom'
        self.variant_desc = f'δ={self.delta}'

    def _base_energy(self, x):
        """Override base energy to use flat-bottomed well."""
        r   = x.reshape(-1, 4, 2)
        r2  = (r * r).sum(-1)
        deviation = (r2 - 1.0).abs() - self.delta
        U   = self.alpha * (torch.clamp(deviation, min=0.0) ** 2).sum(-1)
        diffs = r.unsqueeze(2) - r.unsqueeze(1)
        d2    = (diffs * diffs).sum(-1)
        for i, j in self._pairs:
            U = U + self.c * torch.exp(-d2[:, i, j] / (2.0 * self.sig2))
        return U


# ==============================================================
# OU-process helpers
# ==============================================================
def at(t):
    return torch.exp(-t)

def vt(t):
    return 1.0 - torch.exp(-2.0 * t)


# ==============================================================
# SNIS weights
# ==============================================================
def snis_w(y, t, xr):
    a, v = at(t), vt(t)
    diff = y.unsqueeze(1) - a * xr.unsqueeze(0)
    lw   = -0.5 * (diff ** 2).sum(-1) / v
    lw   = lw - lw.max(1, keepdim=True).values
    w    = lw.exp()
    return w / w.sum(1, keepdim=True).clamp(min=1e-30)


def surrogate_transition_w(y, t, xr, precomp, leaf=False,
                           lmin=1e-4, lmax=1e6):
    """
    Exact transition responsibilities for the local Gaussian/leaf surrogate
    components used by HLSI.

    This is the controlled ablation requested in the OU-vs-surrogate-weight
    question: it changes only the transition weights.  The downstream score
    estimator is still the corresponding existing HLSI / Leaf-CE-HLSI routine.

    For ordinary HLSI, the component is the band-gated Laplace Gaussian
    implicit in est_hlsi: trusted positive Hessian directions have clean
    precision lambda, while untrusted directions fall back to a point mass at
    x_i and therefore transition precision 1/v_t.

    For leaf=True, the component is the leaf-rescued surrogate implicit in
    est_leaf_hlsi / est_leaf_ce_hlsi: trusted directions use the ordinary
    Hessian precision, negative directions use rescued_eig, and remaining
    directions fall back to the Tweedie point-mass transition.
    """
    a  = at(t); a2 = a ** 2; v = vt(t)
    V  = precomp['V']

    if leaf:
        mu          = precomp['mu']
        trusted_eig = precomp['trusted_eig']
        trusted     = precomp['trusted']
        is_non_psd  = precomp['is_non_psd']
        rescued_eig = precomp['rescued_eig']

        hlsi_eig = trusted_eig / (a2 + v * trusted_eig.clamp(min=1e-30))
        dual_eig = rescued_eig / (a2 + v * rescued_eig.clamp(min=1e-30))
        twd_eig  = torch.full_like(trusted_eig, 1.0 / v)
        sig_inv_eig = torch.where(
            trusted,
            hlsi_eig,
            torch.where(is_non_psd, dual_eig, twd_eig),
        )
    else:
        s0  = precomp['s0']
        lam = precomp['lam']
        ok  = (lam > lmin) & (lam <= lmax)
        lam_g = torch.where(ok, lam, torch.zeros_like(lam))
        s0e = torch.einsum('mji,mj->mi', V, s0)
        delta_eig = torch.where(
            ok,
            s0e / lam_g.clamp(min=1e-30),
            torch.zeros_like(s0e),
        )
        mu = xr + torch.einsum('mij,mj->mi', V, delta_eig)
        sig_inv_eig = torch.where(
            ok,
            lam_g / (a2 + v * lam_g.clamp(min=1e-30)),
            torch.full_like(lam, 1.0 / v),
        )

    # log N(y; a_t mu_i, Sigma_{i,t}) up to constants common to i.
    # Since Sigma_inv is diagonal in V_i, logdet contributes sum log eigs.
    disp     = y.unsqueeze(1) - a * mu.unsqueeze(0)
    disp_eig = torch.einsum('mji,nmj->nmi', V, disp)
    qform    = (sig_inv_eig.unsqueeze(0) * disp_eig.square()).sum(-1)
    logdet   = torch.log(sig_inv_eig.clamp(min=1e-30)).sum(-1)
    lw       = 0.5 * logdet.unsqueeze(0) - 0.5 * qform
    lw       = lw - lw.max(1, keepdim=True).values
    w        = lw.exp()
    return w / w.sum(1, keepdim=True).clamp(min=1e-30)


# ==============================================================
# Score estimators
# ==============================================================
def est_tweedie(y, t, xr, w):
    disp = y.unsqueeze(1) - at(t) * xr.unsqueeze(0)
    return -(w.unsqueeze(2) * disp).sum(1) / vt(t)


def est_tsi(y, t, xr, w, target, s0_ref=None):
    s0 = target.score(xr) if s0_ref is None else s0_ref
    return (w.unsqueeze(2) * s0.unsqueeze(0)).sum(1) / at(t)


def est_hlsi(y, t, xr, w, target, lmin=1e-4, lmax=1e6, precomp=None):
    a  = at(t); a2 = a ** 2; v = vt(t)
    if precomp is None:
        H  = target.hessian(xr)
        s0 = target.score(xr)
        lam, V = torch.linalg.eigh(H)
    else:
        s0  = precomp['s0']
        lam = precomp['lam']
        V   = precomp['V']
    ok    = (lam > lmin) & (lam <= lmax)
    lam_g = torch.where(ok, lam, torch.zeros_like(lam))
    s0e       = torch.einsum('mji,mj->mi', V, s0)
    delta_eig = torch.where(ok, s0e / lam_g.clamp(min=1e-30), torch.zeros_like(s0e))
    mu        = xr + torch.einsum('mij,mj->mi', V, delta_eig)
    sig_inv_eig = torch.where(
        ok,
        lam_g / (a2 + v * lam_g.clamp(min=1e-30)),
        torch.full_like(lam, 1.0 / v)
    )
    disp     = y.unsqueeze(1) - a * mu.unsqueeze(0)
    disp_eig = torch.einsum('mji,nmj->nmi', V, disp)
    sc_eig   = sig_inv_eig.unsqueeze(0) * disp_eig
    comp     = -torch.einsum('mij,nmj->nmi', V, sc_eig)
    return (w.unsqueeze(2) * comp).sum(1)


def est_ce_hlsi(y, t, xr, w, target, lmin=1e-4, lmax=1e6, precomp=None):
    a  = at(t); a2 = a ** 2; v = vt(t)
    s_twd = est_tweedie(y, t, xr, w)
    if precomp is None:
        s_tsi = est_tsi(y, t, xr, w, target)
        H     = target.hessian(xr)
        lam_r, V_r = torch.linalg.eigh(H)
        lam_r = lam_r.clamp(min=lmin, max=lmax)
        H_ref = torch.einsum('mij,mj,mkj->mik', V_r, lam_r, V_r)
    else:
        s_tsi = est_tsi(y, t, xr, w, target, s0_ref=precomp['s0'])
        H_ref = precomp['H_ce']
    Hb      = (w.unsqueeze(-1).unsqueeze(-1) * H_ref.unsqueeze(0)).sum(1)
    Hb      = 0.5 * (Hb + Hb.transpose(-1, -2))
    lam, V  = torch.linalg.eigh(Hb)
    lam     = lam.clamp(min=lmin, max=lmax)
    gate   = a2 / (a2 + v * lam + 1e-30)
    I_gate = 1.0 - gate
    def apply_diag_gate(g, s):
        se = torch.einsum('nij,nj->ni', V.transpose(-1, -2), s)
        return torch.einsum('nij,nj->ni', V, g * se)
    return apply_diag_gate(I_gate, s_twd) + apply_diag_gate(gate, s_tsi)


def compute_adaptive_p_leaf(target, xr, eigvals, eigvecs, lmin=1e-4,
                             delta_vals=(0.05, 0.1, 0.2),
                             p_min=0.1, p_max=100.0):
    M, d = xr.shape
    is_bad = eigvals < 0
    p_adaptive = torch.where(is_bad,
                             torch.full_like(eigvals, p_min),
                             eigvals)
    bad_i, bad_k = is_bad.nonzero(as_tuple=True)
    if len(bad_i) == 0:
        return p_adaptive
    bad_vecs = eigvecs[bad_i, :, bad_k]
    bad_lam  = eigvals[bad_i, bad_k]
    bad_xr   = xr[bad_i]
    D_accum = torch.zeros(len(bad_i), dtype=xr.dtype, device=xr.device)
    for delta in delta_vals:
        x_plus  = bad_xr + delta * bad_vecs
        x_minus = bad_xr - delta * bad_vecs
        s_both  = target.score(torch.cat([x_plus, x_minus], dim=0))
        s_plus  = s_both[:len(bad_i)]
        s_minus = s_both[len(bad_i):]
        q_plus  = (s_plus  * bad_vecs).sum(-1)
        q_minus = (s_minus * bad_vecs).sum(-1)
        D = (q_plus - q_minus + 2.0 * bad_lam * delta) / (2.0 * delta ** 3 + 1e-30)
        D_accum = D_accum + D
    D_mean = D_accum / len(delta_vals)
    p_heur = bad_lam + torch.sqrt((-3.0 * D_mean).clamp(min=0.0))
    p_heur = p_heur.clamp(min=p_min, max=p_max)
    p_adaptive[bad_i, bad_k] = p_heur
    return p_adaptive


def precompute_leaf_hlsi(target, xr, lmin=1e-4, lmax=1e6, p_leaf=P_LEAF,
                         use_fd_hessian=True, fd_eps=1e-4,
                         adaptive_delta_vals=(0.05, 0.1, 0.2),
                         adaptive_p_min=0.1, adaptive_p_max=100.0):
    if use_fd_hessian:
        H = target.hessian_fd(xr, eps=fd_eps)
    else:
        H = target.hessian(xr)
    H  = 0.5 * (H + H.transpose(1, 2))
    s0 = target.score(xr)
    eigvals, eigvecs = torch.linalg.eigh(H)
    trusted    = (eigvals >= lmin) & (eigvals <= lmax)
    is_non_psd = eigvals < 0
    trusted_eig = torch.where(trusted, eigvals, torch.zeros_like(eigvals))

    if p_leaf is None:
        p_per_dir = compute_adaptive_p_leaf(
            target, xr, eigvals, eigvecs, lmin=lmin,
            delta_vals=adaptive_delta_vals,
            p_min=adaptive_p_min, p_max=adaptive_p_max,
        )
        rescued_eig = torch.where(
            trusted, eigvals,
            torch.where(is_non_psd, p_per_dir, torch.zeros_like(eigvals))
        )
    else:
        p_per_dir = torch.full_like(eigvals, float(p_leaf))
        rescued_eig = torch.where(
            trusted, eigvals,
            torch.where(is_non_psd, p_per_dir, torch.zeros_like(eigvals))
        )

    s0e       = torch.einsum('mji,mj->mi', eigvecs, s0)
    delta_eig = torch.where(trusted,
                            s0e / trusted_eig.clamp(min=lmin),
                            torch.zeros_like(s0e))
    mu = xr + torch.einsum('mij,mj->mi', eigvecs, delta_eig)
    P_leaf_mat = torch.einsum('mij,mj,mkj->mik', eigvecs, rescued_eig, eigvecs)
    H_ce   = torch.einsum('mij,mj,mkj->mik', eigvecs,
                           eigvals.clamp(min=lmin, max=lmax), eigvecs)
    return {
        's0':          s0,
        'lam':         eigvals,
        'V':           eigvecs,
        'trusted_eig': trusted_eig,
        'rescued_eig': rescued_eig,
        'mu':          mu,
        'P_leaf':      P_leaf_mat,
        'trusted':     trusted,
        'is_non_psd':  is_non_psd,
        'H_ce':        H_ce,
    }


def est_leaf_hlsi(y, t, xr, w, target=None, precomp=None, lmin=1e-4, lmax=1e6, p_leaf=P_LEAF):
    if precomp is None:
        if target is None:
            raise ValueError('Need target or precomp')
        precomp = precompute_leaf_hlsi(target, xr, lmin, lmax, p_leaf)
    a  = at(t); a2 = a ** 2; v = vt(t)
    V           = precomp['V']
    trusted_eig = precomp['trusted_eig']
    trusted     = precomp['trusted']
    is_non_psd  = precomp['is_non_psd']
    mu          = precomp['mu']
    hlsi_eig = trusted_eig / (a2 + v * trusted_eig.clamp(min=1e-30))
    p_per_dir = precomp['rescued_eig']
    dual_eig  = p_per_dir / (a2 + v * p_per_dir.clamp(min=1e-30))
    twd_eig   = torch.full_like(trusted_eig, 1.0 / v)
    sigmainv_eig = torch.where(trusted, hlsi_eig,
                               torch.where(is_non_psd, dual_eig, twd_eig))
    disp   = y.unsqueeze(1) - a * mu.unsqueeze(0)
    disp_e = torch.einsum('mji,nmj->nmi', V, disp)
    comp   = -torch.einsum('mij,nmj->nmi', V,
                            sigmainv_eig.unsqueeze(0) * disp_e)
    return (w.unsqueeze(2) * comp).sum(1)


def est_leaf_ce_hlsi(y, t, xr, w, target=None, precomp=None,
                     lmin=1e-4, lmax=1e6, p_leaf=P_LEAF):
    if precomp is None:
        if target is None:
            raise ValueError('Need target or precomp')
        precomp = precompute_leaf_hlsi(target, xr, lmin, lmax, p_leaf)
    a = at(t); a2 = a ** 2; v = vt(t)
    s_twd = est_tweedie(y, t, xr, w)
    s_tsi = est_tsi(y, t, xr, w, target, s0_ref=precomp['s0'])
    Pbar = (w.unsqueeze(-1).unsqueeze(-1) * precomp['P_leaf'].unsqueeze(0)).sum(1)
    Pbar = 0.5 * (Pbar + Pbar.transpose(1, 2))
    lam, V = torch.linalg.eigh(Pbar)
    gate   = a2 / (a2 + v * lam + 1e-30)
    delta   = s_tsi - s_twd
    delta_e = torch.einsum('nji,nj->ni', V, delta)
    gated   = torch.einsum('nij,nj->ni', V, gate * delta_e)
    return s_twd + gated


def _precision_cache(precomp, xr, hessian_processing='base', lmin=1e-4, lmax=1e6):
    """
    Return a PSD clean-precision closure for a given Hessian-processing mode.

    hessian_processing='base' uses the CE-HLSI PSD clamp H_ce.
    hessian_processing in {'leaf', 'adaptive_leaf'} uses the leaf-repaired P_leaf.
    The cache is attached to the precompute dict because these quantities are
    fixed for the reference bank and are reused at every reverse-SDE step.
    """
    key = f"_precision_cache::{hessian_processing}::{float(lmin):.3e}::{float(lmax):.3e}"
    if key in precomp:
        return precomp[key]

    V  = precomp['V']
    s0 = precomp['s0']
    lam = precomp['lam']

    if hessian_processing == 'base':
        p_eig = lam.clamp(min=lmin, max=lmax)
    elif hessian_processing in ('leaf', 'adaptive_leaf'):
        p_eig = precomp['rescued_eig'].clamp(min=0.0)
    else:
        raise ValueError(f"Unknown hessian_processing={hessian_processing!r}")

    P_mat = torch.einsum('mij,mj,mkj->mik', V, p_eig, V)

    # Newton-shifted center mu_i = x_i + P_i^\dagger s_i.
    # Singular directions in the leaf closure get zero pseudoinverse gain.
    s0_eig = torch.einsum('mji,mj->mi', V, s0)
    inv_p_eig = torch.where(p_eig > 1e-30, 1.0 / p_eig.clamp(min=1e-30), torch.zeros_like(p_eig))
    mu = xr + torch.einsum('mij,mj->mi', V, inv_p_eig * s0_eig)
    P_mu = torch.einsum('mij,mj->mi', P_mat, mu)

    out = dict(P_eig=p_eig, P=P_mat, mu=mu, P_mu=P_mu, V=V)
    precomp[key] = out
    return out


def _project_lambda_ge_identity(Lambda, lambda_max=1e6):
    """Project a batch of Lambda matrices to {Lambda symmetric, Lambda >= I}."""
    Lambda = 0.5 * (Lambda + Lambda.transpose(-1, -2))
    eig, V = torch.linalg.eigh(Lambda)
    eig = eig.clamp(min=1.0, max=lambda_max)
    return torch.einsum('bij,bj,bkj->bik', V, eig, V)


def est_lambda_hlsi(y, t, xr, w, target=None, precomp=None,
                    hessian_processing='base', lmin=1e-4, lmax=1e6,
                    project_lambda=True, m_reg=1e-8, pinv_rtol=1e-6):
    """
    Lambda/Gamma-HLSI gate from the local surrogate regression closure.

    Given the standard CE-HLSI ingredients {x_i, s_i, P_i, rho_i}, this builds
    M_q = E_q[E E^T] and N_q = E_q[E D^T] using the surrogate Gaussian closure,
    solves Lambda_q^* = -N_q^T M_q^\dagger, and applies

        score = Tweedie + (Lambda_q^*)^\dagger (TSI - Tweedie).

    The public sampler names below use "Gamma-HLSI" for this inverse-Lambda
    gate, while "Lambda-HLSI" is retained as an alias in SAMPLER_CONFIGS.
    """
    if precomp is None:
        if target is None:
            raise ValueError('Need target or precomp')
        precomp = precompute_leaf_hlsi(target, xr, lmin=lmin, lmax=lmax, p_leaf=P_LEAF)

    a = at(t); v = vt(t)
    a2 = a ** 2
    tau = a2 / v
    B, d = y.shape
    dtype, device = y.dtype, y.device
    I = torch.eye(d, dtype=dtype, device=device)

    cache = _precision_cache(precomp, xr, hessian_processing=hessian_processing,
                             lmin=lmin, lmax=lmax)
    V = cache['V']
    p_eig = cache['P_eig']
    P_mat = cache['P']
    P_mu = cache['P_mu']

    # Standard Tweedie/TSI means under the chosen responsibilities rho_i.
    s_twd = est_tweedie(y, t, xr, w)
    s_tsi = est_tsi(y, t, xr, w, target, s0_ref=precomp['s0'])
    delta = s_tsi - s_twd

    # Posterior X_0 | Y_t=y, I=i under the local Gaussian surrogate:
    #   Sigma_i = (P_i + tau I)^(-1),
    #   m_i = Sigma_i (P_i mu_i + (a/v)y).
    sigma_eig = 1.0 / (p_eig + tau).clamp(min=1e-30)
    Sigma = torch.einsum('mij,mj,mkj->mik', V, sigma_eig, V)

    rhs = P_mu.unsqueeze(0) + (a / v) * y.unsqueeze(1)
    rhs_eig = torch.einsum('mji,bmj->bmi', V, rhs)
    post_mean = torch.einsum('mij,bmj->bmi', V, sigma_eig.unsqueeze(0) * rhs_eig)

    # m_i^E = E[B - s_t^q(y) | I=i] in the surrogate algebra; equivalently
    # the conditional mean Tweedie residual around the component posterior.
    mE = -(y.unsqueeze(1) - a * post_mean) / v
    Ebar = (w.unsqueeze(-1) * mE).sum(1)
    dE = mE - Ebar.unsqueeze(1)

    # M_q = within covariance + between covariance.
    M_within = (a2 / (v ** 2)) * torch.einsum('bm,mij->bij', w, Sigma)
    M_between = torch.einsum('bm,bmi,bmj->bij', w, dE, dE)
    M_q = M_within + M_between
    M_q = 0.5 * (M_q + M_q.transpose(-1, -2))

    # K_i m_i^E = (I + tau^{-1} P_i) m_i^E, avoiding explicit BxMxDxD tensors.
    mE_eig = torch.einsum('mji,bmj->bmi', V, mE)
    P_mE = torch.einsum('mij,bmj->bmi', V, p_eig.unsqueeze(0) * mE_eig)
    K_mE = mE + (1.0 / tau) * P_mE

    # K_{Pbar} Ebar = (I + tau^{-1} Pbar) Ebar.
    Ebar_eig = torch.einsum('mji,bj->bmi', V, Ebar)
    P_Ebar_components = torch.einsum('mij,bmj->bmi', V, p_eig.unsqueeze(0) * Ebar_eig)
    Pbar_Ebar = (w.unsqueeze(-1) * P_Ebar_components).sum(1)
    Kbar_Ebar = Ebar + (1.0 / tau) * Pbar_Ebar

    C_ED_bw = -torch.einsum('bm,bmi,bmj->bij',
                            w, dE, K_mE - Kbar_Ebar.unsqueeze(1))

    # N_q = -(a/v)I + C_ED_bw^T after the within-component simplification.
    N_q = -(a / v) * I.unsqueeze(0) + C_ED_bw.transpose(-1, -2)

    if m_reg is not None and m_reg > 0:
        scale = M_q.diagonal(dim1=-2, dim2=-1).mean(-1).clamp(min=1.0)
        M_solve = M_q + (m_reg * scale).view(B, 1, 1) * I.unsqueeze(0)
    else:
        M_solve = M_q

    M_pinv = torch.linalg.pinv(M_solve, rtol=pinv_rtol)
    Lambda = -torch.matmul(N_q.transpose(-1, -2), M_pinv)

    if project_lambda:
        Lambda = _project_lambda_ge_identity(Lambda)
        eig, U = torch.linalg.eigh(Lambda)
        gate = torch.einsum('bij,bj,bkj->bik', U, 1.0 / eig.clamp(min=1e-30), U)
    else:
        gate = torch.linalg.pinv(Lambda, rtol=pinv_rtol)

    return s_twd + torch.matmul(gate, delta.unsqueeze(-1)).squeeze(-1)


DEFAULT_METHODS = ['Leaf-CE-HLSI', 'CE-HLSI', 'HLSI', 'Leaf-Gamma-HLSI', 'Gamma-HLSI']

SAMPLER_CONFIGS = OrderedDict([
    # Requested comparison set.
    ('Leaf-CE-HLSI',      {'transition_weights': 'ou',        'hessian_processing': 'leaf',          'gate': 'ce'}),
    ('CE-HLSI',           {'transition_weights': 'ou',        'hessian_processing': 'base',          'gate': 'ce'}),
    ('HLSI',              {'transition_weights': 'ou',        'hessian_processing': 'base',          'gate': 'hlsi'}),
    ('Leaf-Gamma-HLSI',   {'transition_weights': 'ou',        'hessian_processing': 'leaf',          'gate': 'lambda', 'project_lambda': True}),
    ('Gamma-HLSI',        {'transition_weights': 'ou',        'hessian_processing': 'base',          'gate': 'lambda', 'project_lambda': True}),

    # Aliases and additional ablation axes kept available for the OU-vs-surrogate experiments.
    ('Leaf-Lambda-HLSI',  {'transition_weights': 'ou',        'hessian_processing': 'leaf',          'gate': 'lambda', 'project_lambda': True}),
    ('Lambda-HLSI',       {'transition_weights': 'ou',        'hessian_processing': 'base',          'gate': 'lambda', 'project_lambda': True}),
    ('Surrogate-HLSI',    {'transition_weights': 'surrogate', 'hessian_processing': 'base',          'gate': 'hlsi'}),
    ('Surrogate-CE-HLSI', {'transition_weights': 'surrogate', 'hessian_processing': 'base',          'gate': 'ce'}),
    ('Surrogate-Gamma-HLSI', {'transition_weights': 'surrogate', 'hessian_processing': 'base',       'gate': 'lambda', 'project_lambda': True}),
    ('Surrogate-Leaf-CE-HLSI', {'transition_weights': 'surrogate', 'hessian_processing': 'leaf',     'gate': 'ce'}),
    ('Surrogate-Leaf-Gamma-HLSI', {'transition_weights': 'surrogate', 'hessian_processing': 'leaf',  'gate': 'lambda', 'project_lambda': True}),
    ('Adaptive Leaf-CE',  {'transition_weights': 'ou',        'hessian_processing': 'adaptive_leaf', 'gate': 'ce'}),
    ('Adaptive Leaf-Gamma-HLSI', {'transition_weights': 'ou', 'hessian_processing': 'adaptive_leaf', 'gate': 'lambda', 'project_lambda': True}),
    ('Surrogate-Adaptive Leaf-CE', {'transition_weights': 'surrogate', 'hessian_processing': 'adaptive_leaf', 'gate': 'ce'}),
    ('Surrogate-Adaptive Leaf-Gamma-HLSI', {'transition_weights': 'surrogate', 'hessian_processing': 'adaptive_leaf', 'gate': 'lambda', 'project_lambda': True}),

    # Non-HLSI baselines retained for backwards compatibility.
    ('Tweedie',           {'transition_weights': 'ou',        'hessian_processing': 'base',          'gate': 'tweedie'}),
    ('TSI',               {'transition_weights': 'ou',        'hessian_processing': 'base',          'gate': 'tsi'}),
    ('Blended',           {'transition_weights': 'ou',        'hessian_processing': 'base',          'gate': 'blend'}),
    ('Leaf-HLSI',         {'transition_weights': 'ou',        'hessian_processing': 'leaf',          'gate': 'hlsi'}),
    ('Adaptive Leaf',     {'transition_weights': 'ou',        'hessian_processing': 'adaptive_leaf', 'gate': 'hlsi'}),
])


def _resolve_sampler_config(method):
    if isinstance(method, str):
        if method not in SAMPLER_CONFIGS:
            raise ValueError(f"Unknown sampler method {method!r}. Available: {list(SAMPLER_CONFIGS)}")
        return method, dict(SAMPLER_CONFIGS[method])
    if isinstance(method, dict):
        name = method.get('name') or method.get('label') or method.get('method') or 'custom'
        cfg = dict(method)
        cfg.pop('name', None); cfg.pop('label', None); cfg.pop('method', None)
        return name, cfg
    raise TypeError(f"Method entries must be strings or dicts, got {type(method)}")


def make_sampler_score_fn(method, target, xr, precomp, precomp_adaptive,
                          lmin=1e-4, lmax=1e6):
    """Build a score closure from factorized sampler ingredients."""
    name, cfg = _resolve_sampler_config(method)
    hproc = cfg.get('hessian_processing', 'base')
    gate = cfg.get('gate', 'hlsi')
    weight_mode = cfg.get('transition_weights', 'ou')

    if hproc == 'adaptive_leaf':
        pc = precomp_adaptive
        leaf_weights = True
    elif hproc == 'leaf':
        pc = precomp
        leaf_weights = True
    elif hproc == 'base':
        pc = precomp
        leaf_weights = False
    else:
        raise ValueError(f"Unknown hessian_processing={hproc!r} for {name}")

    def fn(y, t):
        t = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.get_default_dtype(), device=y.device)
        if weight_mode in ('ou', 'OU', 'snis', 'L'):
            w = snis_w(y, t, xr)
        elif weight_mode in ('surrogate', 'Surr', 'surr'):
            w = surrogate_transition_w(y, t, xr, pc, leaf=leaf_weights, lmin=lmin, lmax=lmax)
        else:
            raise ValueError(f"Unknown transition_weights={weight_mode!r} for {name}")

        if gate == 'tweedie':
            return est_tweedie(y, t, xr, w)
        if gate == 'tsi':
            return est_tsi(y, t, xr, w, target, s0_ref=pc['s0'])
        if gate == 'blend':
            return est_blended(y, t, xr, w, target, s0_ref=pc['s0'])
        if gate == 'hlsi':
            if hproc in ('leaf', 'adaptive_leaf'):
                return est_leaf_hlsi(y, t, xr, w, target, precomp=pc, lmin=lmin, lmax=lmax)
            return est_hlsi(y, t, xr, w, target, precomp=pc, lmin=lmin, lmax=lmax)
        if gate == 'ce':
            if hproc in ('leaf', 'adaptive_leaf'):
                return est_leaf_ce_hlsi(y, t, xr, w, target, precomp=pc, lmin=lmin, lmax=lmax)
            return est_ce_hlsi(y, t, xr, w, target, precomp=pc, lmin=lmin, lmax=lmax)
        if gate in ('lambda', 'gamma', 'Lambda-HLSI', 'Gamma-HLSI'):
            return est_lambda_hlsi(
                y, t, xr, w, target=target, precomp=pc,
                hessian_processing=hproc,
                lmin=lmin, lmax=lmax,
                project_lambda=cfg.get('project_lambda', True),
                m_reg=cfg.get('m_reg', 1e-8),
                pinv_rtol=cfg.get('pinv_rtol', 1e-6),
            )
        raise ValueError(f"Unknown gate={gate!r} for {name}")

    return name, fn


def est_blended(y, t, xr, w, target, s0_ref=None):
    a, v = at(t), vt(t)
    s0  = target.score(xr) if s0_ref is None else s0_ref
    tsi = s0.unsqueeze(0) / a
    twd = -(y.unsqueeze(1) - a * xr.unsqueeze(0)) / v
    am  = (w.unsqueeze(2) * tsi).sum(1)
    bm  = (w.unsqueeze(2) * twd).sum(1)
    ac  = tsi - am.unsqueeze(1)
    bc  = twd - bm.unsqueeze(1)
    va  = (w.unsqueeze(2) * ac ** 2).sum(1).clamp(min=1e-30)
    vb  = (w.unsqueeze(2) * bc ** 2).sum(1).clamp(min=1e-30)
    cab = (w.unsqueeze(2) * ac * bc).sum(1)
    den = (va + vb - 2 * cab).clamp(min=1e-20)
    g   = ((va - cab) / den).clamp(0, 1)
    return (1 - g) * am + g * bm


# ==============================================================
# Heun predictor–corrector reverse SDE
# ==============================================================
def heun_sde(score_fn, n, d, n_steps=200, t_max=3.0, t_min=0.015, device=None):
    device = DEVICE if device is None else torch.device(device)
    ts  = torch.linspace(t_max, t_min, n_steps + 1, dtype=torch.get_default_dtype(), device=device)
    y   = torch.randn(n, d, dtype=torch.get_default_dtype(), device=device)
    ms  = 0.0
    fail = False
    for i in range(n_steps):
        tc, tn = ts[i], ts[i + 1]
        h = tc - tn
        s1 = score_fn(y, tc)
        ms = max(ms, s1.abs().max().item())
        if not torch.isfinite(s1).all():
            fail = True; break
        d1    = y + 2.0 * s1
        noise = (2.0 * h).sqrt() * torch.randn_like(y)
        yh    = y + h * d1 + noise
        s2 = score_fn(yh, tn)
        if not torch.isfinite(s2).all():
            fail = True; break
        d2 = yh + 2.0 * s2
        y  = y + 0.5 * h * (d1 + d2) + noise
        if not torch.isfinite(y).all():
            fail = True; break
    if not fail:
        tf = torch.tensor(t_min, dtype=torch.get_default_dtype(), device=y.device)
        sf = score_fn(y, tf)
        ms = max(ms, sf.abs().max().item())
        if torch.isfinite(sf).all():
            y = (y + vt(tf) * sf) / at(tf)
        else:
            fail = True
    return y, ms, fail


# ==============================================================
# Metrics
# ==============================================================
def mmd(X, Y, bws=(0.5, 1.0, 2.0, 5.0, 10.0)):
    n  = min(len(X), 2000)
    m  = min(len(Y), 2000)
    X  = X[:n]; Y = Y[:m]
    xx = torch.cdist(X, X) ** 2
    yy = torch.cdist(Y, Y) ** 2
    xy = torch.cdist(X, Y) ** 2
    gs = [0.5 / (b ** 2) for b in bws]
    v  = sum(((-g * xx).exp().mean() + (-g * yy).exp().mean()
              - 2 * (-g * xy).exp().mean()) for g in gs)
    return (v / len(gs)).item()


def kl_energy_histogram(X, Y, target, bins=120, smoothing=1e-12):
    with torch.no_grad():
        Ux = (-target.log_prob(X)).cpu().numpy()
        Uy = (-target.log_prob(Y)).cpu().numpy()
    lo, hi = float(np.percentile(Ux, 0.5)), float(np.percentile(Ux, 99.5))
    edges  = np.linspace(lo, hi, bins + 1)
    px, _ = np.histogram(Ux, bins=edges, density=False)
    py, _ = np.histogram(Uy, bins=edges, density=False)
    px    = px + smoothing; py = py + smoothing
    px   /= px.sum();       py /= py.sum()
    return float(np.sum(px * (np.log(px) - np.log(py))))


def kl_pairwise_dist_histogram(X, Y, bins=120, smoothing=1e-12):
    def pairwise_dists(samples):
        r = samples.reshape(-1, 4, 2)
        dists = []
        for i in range(4):
            for j in range(i + 1, 4):
                d = ((r[:, i] - r[:, j]) ** 2).sum(-1).sqrt()
                dists.append(d)
        return torch.cat(dists, dim=0).cpu().numpy()
    with torch.no_grad():
        dx = pairwise_dists(X)
        dy = pairwise_dists(Y)
    lo, hi = 0.0, float(np.percentile(dx, 99.5))
    edges  = np.linspace(lo, hi, bins + 1)
    px, _  = np.histogram(dx, bins=edges, density=False)
    py, _  = np.histogram(dy, bins=edges, density=False)
    px     = px + smoothing; py = py + smoothing
    px    /= px.sum();       py /= py.sum()
    return float(np.sum(px * (np.log(px) - np.log(py))))


def ksd_rbf(samples, score_fn, bandwidth='median'):
    X = samples
    n = min(len(X), 1000)
    X = X[:n]
    if n < 5:
        return float('inf')
    with torch.no_grad():
        S    = score_fn(X)
        d    = X.shape[1]
        dmat = torch.cdist(X, X)
        med  = torch.median(dmat[dmat > 0]).item() if (dmat > 0).any() else 1.0
        h    = med if (np.isfinite(med) and med > 1e-12) else 1.0
        if bandwidth != 'median':
            h = float(bandwidth)
        d2   = dmat ** 2
        K    = torch.exp(-d2 / (2.0 * h ** 2))
        diff = X.unsqueeze(1) - X.unsqueeze(0)
        term1 = K * (S @ S.T)
        sdiff = S.unsqueeze(1) - S.unsqueeze(0)
        term2 = -(K.unsqueeze(-1) * sdiff * diff).sum(-1) / (h ** 2)
        term4 = K * (d / (h ** 2) - d2 / (h ** 4))
        U    = term1 + term2 + term4
        ksd2 = U.mean()
        return float(torch.sqrt(ksd2.clamp(min=0.0)).item())


def w2_distance(X, Y):
    import ot as pot
    if X.dim() > 2: X = X.reshape(X.shape[0], -1)
    if Y.dim() > 2: Y = Y.reshape(Y.shape[0], -1)
    a = pot.unif(X.shape[0])
    b = pot.unif(Y.shape[0])
    M = torch.cdist(X, Y) ** 2
    ret = pot.emd2(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    return float(math.sqrt(ret))


def ess_kde(samples, target, n_max=1000):
    from sklearn.neighbors import KernelDensity
    n    = min(len(samples), n_max)
    sc   = samples[torch.randperm(len(samples), device=samples.device)[:n]]
    sc_np = sc.cpu().double().numpy()
    d    = sc_np.shape[1]
    bw   = max(n ** (-1.0 / (d + 4)), 0.05)
    kde  = KernelDensity(kernel='gaussian', bandwidth=bw).fit(sc_np)
    log_q = kde.score_samples(sc_np)
    with torch.no_grad():
        log_p = target.log_prob(sc).cpu().double().numpy()
    log_w  = log_p - log_q
    log_w -= log_w.max()
    w      = np.exp(log_w)
    w     /= w.sum()
    return float(1.0 / (n * float((w ** 2).sum())))


def nll_kde(samples, test_points, n_fit=5000):
    from sklearn.neighbors import KernelDensity
    sc_np = samples.cpu().double().numpy()
    te_np = test_points.cpu().double().numpy()
    n, d  = len(sc_np), sc_np.shape[1]
    if n_fit is not None and n_fit < n:
        idx   = np.random.choice(n, n_fit, replace=False)
        sc_np = sc_np[idx]
        n     = len(sc_np)
    bw  = max(n ** (-1.0 / (d + 4)), 0.05)
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(sc_np)
    return float(-kde.score_samples(te_np).mean())


def copying_score(samples, xr_train, xt_test, n_gen=800, k_pool=None,
                  alert_ratio=0.8, method_name=''):
    n_gen  = min(n_gen, len(samples))
    k_pool = k_pool or min(len(xr_train), len(xt_test))
    k_pool = min(k_pool, len(xr_train), len(xt_test))
    if n_gen <= 0 or k_pool <= 0 or len(xr_train) == 0 or len(xt_test) == 0:
        return dict(mean_d_train=float('nan'), mean_d_test=float('nan'),
                    ratio=float('nan'), copying=False,
                    d_train_dist=np.array([]), d_test_dist=np.array([]))
    gen  = samples[torch.randperm(len(samples), device=samples.device)[:n_gen]]
    ref  = xr_train[torch.randperm(len(xr_train), device=xr_train.device)[:k_pool]]
    test = xt_test [torch.randperm(len(xt_test), device=xt_test.device) [:k_pool]]
    def min_dist(A, B, chunk=200):
        if len(A) == 0 or len(B) == 0:
            return torch.zeros(len(A), device=A.device)
        mins = []
        for i in range(0, len(A), chunk):
            d = torch.cdist(A[i:i+chunk].float(), B.float())
            mins.append(d.min(dim=1).values)
        return torch.cat(mins)
    with torch.no_grad():
        d_tr = min_dist(gen, ref)
        d_te = min_dist(gen, test)
    mean_tr = float(d_tr.mean())
    mean_te = float(d_te.mean())
    ratio   = mean_tr / (mean_te + 1e-30)
    copying = ratio < alert_ratio
    if copying:
        print(f"\n  ⚠️  ALERT {method_name} IS COPYING ⚠️")
        print(f"      d_train={mean_tr:.4f}  d_test={mean_te:.4f}  ratio={ratio:.3f}")
    return dict(mean_d_train=mean_tr, mean_d_test=mean_te,
                ratio=ratio, copying=copying,
                d_train_dist=d_tr.cpu().numpy(), d_test_dist=d_te.cpu().numpy())


def score_rmse_snis_ref(score_hat_fn, target, xr_ref_large, n_eval=2000,
                        t_min=0.015, t_max=3.0, n_time_grid=120):
    with torch.no_grad():
        x0 = xr_ref_large[torch.randperm(len(xr_ref_large), device=xr_ref_large.device)[:n_eval]]
        t_grid = torch.exp(torch.linspace(
            math.log(t_min), math.log(t_max), n_time_grid,
            dtype=torch.get_default_dtype(), device=x0.device))
        idx = torch.randint(0, n_time_grid, (n_eval,), device=x0.device)
        t   = t_grid[idx]
        a   = at(t).unsqueeze(-1)
        v   = vt(t).unsqueeze(-1)
        xt  = a * x0 + torch.sqrt(v) * torch.randn_like(x0)
        s_true = torch.empty_like(xt)
        for j in idx.unique(sorted=True):
            mask = (idx == j)
            tj   = t_grid[j]
            w_ref = snis_w(xt[mask], tj, xr_ref_large)
            s_true[mask] = est_tweedie(xt[mask], tj, xr_ref_large, w_ref)
        s_hat = torch.empty_like(xt)
        for j in idx.unique(sorted=True):
            mask = (idx == j)
            tj   = t_grid[j]
            s_hat[mask] = score_hat_fn(xt[mask], tj)
        err2 = ((s_true - s_hat) ** 2).sum(dim=1)
        return float(torch.sqrt(err2.mean()).item())


# ==============================================================
# Run  (parameterised by target and output directory)
# ==============================================================

# MALA tuning presets per variant (some modifications shift the energy
# landscape enough that the baseline MALA step_size underperforms)
MALA_PRESETS = {
    'base':        dict(step_size=0.008, n_chains=64, burnin=120_000, thin=80),
    'corrugated':  dict(step_size=0.006, n_chains=64, burnin=120_000, thin=80),
    'soft_radial': dict(step_size=0.015, n_chains=64, burnin=120_000, thin=80),
    'banana':      dict(step_size=0.006, n_chains=64, burnin=120_000, thin=80),
    'heavy_tail':  dict(step_size=0.008, n_chains=64, burnin=120_000, thin=80),
    'flat_bottom': dict(step_size=0.010, n_chains=64, burnin=120_000, thin=80),
}


def run(target, out_dir='outputs', methods=None):
    """
    Run the full benchmark pipeline for a single target.

    Parameters
    ----------
    target  : DW4Target (or subclass) instance
    out_dir : str, directory for outputs (created if needed)
    methods : list of str, score estimators to run

    Returns
    -------
    results : dict of method → metric dict
    xt      : ground-truth test samples (set 1)
    xr      : reference samples
    methods : list of method names
    target  : target object
    """
    if methods is None:
        methods = list(DEFAULT_METHODS)


    NR        = 1500
    NS        = 5000
    NT        = 3000
    NR_LARGE  = 1500
    N_STEPS   = 300
    T_MAX, T_MIN = 3.0, 0.0002
    N_TIME_GRID  = 300
    D = DW4Target.D

    os.makedirs(out_dir, exist_ok=True)

    vname = getattr(target, 'variant_name', 'unknown')
    vdesc = getattr(target, 'variant_desc', '')
    label = f"{vname}" + (f" ({vdesc})" if vdesc else "")

    print("\n" + "=" * 80)
    print(f"DW-4 benchmark  —  variant: {label}")
    print(f"  α={target.alpha},  c={target.c},  σ²={target.sig2}")
    if vdesc:
        print(f"  modifier: {vdesc}")
    print("=" * 80)

    # ---- Ground-truth samples via MALA ----
    print("\n[1] Generating ground-truth samples via MALA …")
    t0 = time.time()

    mala_kw = MALA_PRESETS.get(vname, MALA_PRESETS['base'])
    all_gt = target.sample_mala(NR + 2*NT + NR_LARGE, verbose=True, **mala_kw)
    print(f"    Done ({time.time()-t0:.0f}s),  got {len(all_gt)} samples")

    xr        = all_gt[:NR]
    xt        = all_gt[NR:NR+NT]
    x_gt2     = all_gt[NR+NT:NR+2*NT]
    xr_large  = all_gt[NR:NR+NR_LARGE]

    # ---- Precompute spectral data ----
    print("\n[2] Precomputing Hessian spectral data …")
    t0 = time.time()
    precomp = precompute_leaf_hlsi(target, xr, lmin=1e-4, lmax=1e6, p_leaf=P_LEAF)
    print(f"    Done ({time.time()-t0:.1f}s)")

    lam_np = precomp['lam'].cpu().numpy()
    print(f"    Hessian eigenvalues: min={lam_np.min():.3e}, "
          f"max={lam_np.max():.3e}, "
          f"frac_non_psd={100*(lam_np < 0).mean():.1f}%")

    print("\n[2b] Computing adaptive p_leaf …")
    t0 = time.time()
    precomp_adaptive = precompute_leaf_hlsi(
        target, xr, lmin=1e-4, lmax=1e6, p_leaf=None,
        adaptive_delta_vals=(0.05, 0.1, 0.2),
        adaptive_p_min=0.1, adaptive_p_max=100.0,
    )
    bad_mask = precomp_adaptive['is_non_psd']
    n_bad = bad_mask.sum().item()
    if n_bad > 0:
        p_np = precomp_adaptive['rescued_eig'][bad_mask].cpu().numpy()
        print(f"    Done ({time.time()-t0:.1f}s)  adaptive p: "
              f"min={p_np.min():.3f}, median={float(np.median(p_np)):.3f}, "
              f"max={p_np.max():.3f}  (n_bad_dirs={len(p_np)})")
    else:
        print(f"    Done ({time.time()-t0:.1f}s)  no non-PSD directions")

    # ---- Score estimator closures ----
    def make_fn(meth):
        _, fn = make_sampler_score_fn(meth, target, xr, precomp, precomp_adaptive,
                                      lmin=1e-4, lmax=1e6)
        return fn

    # ---- Sample + evaluate ----
    print("\n[3] Sampling and evaluating methods …")
    hdr = (f"{'Method':<26s} {'NLL':>10} {'ESS':>10} {'W2':>8}"
           f" {'MMD':>9} {'KL-E':>9} {'KL-D':>9}"
           f" {'KSD':>9} {'RMSE':>9} {'dTr':>8} {'dTe':>8}"
           f" {'Ratio':>7} {'|s|max':>8} {'NaN%':>5} {'t':>5}")
    print(hdr)
    print("-" * len(hdr))

    method_entries = list(methods)
    methods = [_resolve_sampler_config(m)[0] for m in method_entries]

    results = {}
    for entry, m in zip(method_entries, methods):
        fn = make_fn(entry)
        t0 = time.time()
        with torch.no_grad():
            samp, ms, fail = heun_sde(fn, NS, D, n_steps=N_STEPS,
                                      t_max=T_MAX, t_min=T_MIN)
        ok  = torch.isfinite(samp).all(dim=1)
        sc  = samp[ok]
        nv  = ok.sum().item()
        nan_pct = 100.0 * (1.0 - nv / NS)

        if nv >= 20:
            with torch.no_grad():
                mv   = mmd(sc, xt)
                klev = kl_energy_histogram(sc, xt, target)
                kldv = kl_pairwise_dist_histogram(sc, xt)
                ksdv = ksd_rbf(sc, target.score)
                rmsev = score_rmse_snis_ref(fn, target, xr_large,
                                            n_eval=1500,
                                            t_min=T_MIN, t_max=T_MAX,
                                            n_time_grid=N_TIME_GRID)
                copy = copying_score(sc, xr, xt,
                                     n_gen=min(nv, 800),
                                     k_pool=min(NR, NT),
                                     alert_ratio=0.50,
                                     method_name=m)
            w2v  = w2_distance(sc, xt)
            essv = ess_kde(sc, target, n_max=1000)
            nllv = nll_kde(sc, xt, n_fit=5000)
        else:
            mv = klev = kldv = ksdv = rmsev = float('inf')
            w2v = essv = nllv = float('inf')
            copy = dict(mean_d_train=float('nan'), mean_d_test=float('nan'),
                        ratio=float('nan'), copying=False,
                        d_train_dist=np.array([]), d_test_dist=np.array([]))

        dt = time.time() - t0
        results[m] = dict(samples=sc, mmd=mv, kl_energy=klev,
                          kl_dist=kldv, ksd=ksdv, score_rmse=rmsev,
                          w2=w2v, ess=essv, nll=nllv,
                          ms=ms, nan_pct=nan_pct, copy=copy)

        rat_str = f"{copy['ratio']:7.3f}" if np.isfinite(copy['ratio']) else "    N/A"
        tag     = "FAIL" if fail else "ok"
        nll_s   = f"{nllv:10.5f}" if np.isfinite(nllv)  else "   DIVERGED"
        ess_s   = f"{essv:10.5f}" if np.isfinite(essv)  else "   DIVERGED"
        w2_s    = f"{w2v:8.5f}"  if np.isfinite(w2v)   else "  DIVERGED"
        print(f"{m:<26s} {nll_s} {ess_s} {w2_s}"
              f" {mv:9.5f} {klev:9.5f} {kldv:9.5f}"
              f" {ksdv:9.5f} {rmsev:9.5f}"
              f" {copy['mean_d_train']:8.4f} {copy['mean_d_test']:8.4f} {rat_str}"
              f" {ms:8.1f} {nan_pct:5.1f} {dt:4.0f}s {tag}")

    # ---- GT floor ----
    print(f"\n[4] Computing GT floor (MALA set 2 vs set 1) …")
    t0 = time.time()
    with torch.no_grad():
        gt_mmd    = mmd(x_gt2, xt)
        gt_kl_e   = kl_energy_histogram(x_gt2, xt, target)
        gt_kl_d   = kl_pairwise_dist_histogram(x_gt2, xt)
        gt_ksd    = ksd_rbf(x_gt2, target.score)
        gt_copy   = copying_score(x_gt2, xr, xt,
                                   n_gen=min(NT, 800),
                                   k_pool=min(NR, NT),
                                   alert_ratio=0.50,
                                   method_name="GT floor")
    gt_w2   = w2_distance(x_gt2, xt)
    gt_ess  = ess_kde(x_gt2, target, n_max=1000)
    gt_nll  = nll_kde(x_gt2, xt, n_fit=5000)

    gt_floor = dict(samples=x_gt2, mmd=gt_mmd, kl_energy=gt_kl_e,
                    kl_dist=gt_kl_d, ksd=gt_ksd, score_rmse=float('nan'),
                    w2=gt_w2, ess=gt_ess, nll=gt_nll,
                    ms=0.0, nan_pct=0.0, copy=gt_copy)
    results['GT floor'] = gt_floor
    print(f"    Done ({time.time()-t0:.0f}s)")
    print(f"  {'GT floor':<26s}"
          f" {gt_nll:10.5f} {gt_ess:10.5f} {gt_w2:8.5f}"
          f" {gt_mmd:9.5f} {gt_kl_e:9.5f} {gt_kl_d:9.5f}"
          f" {gt_ksd:9.5f} {'N/A':>9}"
          f" {gt_copy['mean_d_train']:8.4f} {gt_copy['mean_d_test']:8.4f}"
          f" {gt_copy['ratio']:7.3f}")

    # ---- Summary table ----
    plot_methods = list(methods) + ['GT floor']
    print("\n" + "=" * 80)
    print(f"Summary: {label}")
    print(f"  {'Method':<26s} {'NLL':>12} {'ESS':>12} {'W2':>10}")
    for m in plot_methods:
        nv = results[m]['nll']
        ev = results[m]['ess']
        wv = results[m]['w2']
        nll_s = f"{nv:12.5f}" if np.isfinite(nv) else '     DIVERGED'
        ess_s = f"{ev:12.5f}" if np.isfinite(ev) else '     DIVERGED'
        w2_s  = f"{wv:10.5f}" if np.isfinite(wv) else '   DIVERGED'
        print(f"  {m:<26s} {nll_s} {ess_s} {w2_s}")

    for metric, mlabel in [('nll',        'NLL (KDE proxy)'),
                           ('ess',        'ESS (KDE proxy)'),
                           ('w2',         'W2'),
                           ('mmd',        'MMD'),
                           ('kl_energy',  'KL energy hist'),
                           ('kl_dist',    'KL pair-dist'),
                           ('ksd',        'KSD'),
                           ('score_rmse', 'Score RMSE')]:
        print(f"\n{mlabel}")
        print("-" * 50)
        for m in plot_methods:
            v = results[m][metric]
            tag = f"{v:12.6f}" if np.isfinite(v) else "         N/A"
            print(f"  {m:<26s}  {tag}")

    print(f"\nCopying score  (ratio < 0.50 → ALERT)")
    print("-" * 50)
    for m in plot_methods:
        cd  = results[m]['copy']
        print(f"  {m:<26s}  dTr={cd['mean_d_train']:7.4f}  "
              f"dTe={cd['mean_d_test']:7.4f}  ratio={cd['ratio']:6.3f}")

    print(f"\nHessian spectrum")
    print("-" * 50)
    print(f"  eigenvalue range: [{lam_np.min():.3e}, {lam_np.max():.3e}]")
    print(f"  frac_non_psd:     {100*(lam_np < 0).mean():.1f}%")
    print(f"  n_bad_dirs:       {n_bad}")
    print("=" * 80)

    return results, xt, xr, methods, target


# ==============================================================
# Plotting  (saves to out_dir)
# ==============================================================
COLORS = {
    'Tweedie':           '#1f77b4',
    'TSI':               '#d62728',
    'HLSI':              '#2ca02c',
    'Surrogate-HLSI':    '#98df8a',
    'Leaf-HLSI':         '#17becf',
    'Gamma-HLSI':        '#bcbd22',
    'Lambda-HLSI':       '#bcbd22',
    'Leaf-Gamma-HLSI':   '#e377c2',
    'Leaf-Lambda-HLSI':  '#e377c2',
    'Surrogate-Gamma-HLSI': '#dbdb8d',
    'Surrogate-Leaf-Gamma-HLSI': '#f7b6d2',
    'CE-HLSI':           '#9467bd',
    'Leaf-CE-HLSI':      '#8c564b',
    'Surrogate-Leaf-CE-HLSI': '#c49c94',
    'Blended':           '#ff7f0e',
    'Adaptive Leaf':     '#006400',
    'Adaptive Leaf-CE':  '#4b0082',
    'Surrogate-Adaptive Leaf-CE': '#9edae5',
    'Adaptive Leaf-Gamma-HLSI': '#7f7f7f',
    'Surrogate-Adaptive Leaf-Gamma-HLSI': '#c7c7c7',
    'GT floor':          '#888888',
}
MARKERS = {
    'Tweedie': 'o', 'TSI': 's', 'HLSI': '^', 'Surrogate-HLSI': '>',
    'Leaf-HLSI': 'P', 'CE-HLSI': 'D', 'Leaf-CE-HLSI': 'X',
    'Gamma-HLSI': '*', 'Lambda-HLSI': '*',
    'Leaf-Gamma-HLSI': 'p', 'Leaf-Lambda-HLSI': 'p',
    'Surrogate-Gamma-HLSI': '1', 'Surrogate-Leaf-Gamma-HLSI': '2',
    'Surrogate-Leaf-CE-HLSI': '<', 'Blended': 'v',
    'Adaptive Leaf': 'h', 'Adaptive Leaf-CE': 'H',
    'Adaptive Leaf-Gamma-HLSI': 'd',
    'Surrogate-Adaptive Leaf-CE': '8',
    'Surrogate-Adaptive Leaf-Gamma-HLSI': '3', 'GT floor': '*',
}


def plot(results, xt, xr, methods, target, out_dir='outputs'):
    C = COLORS
    os.makedirs(out_dir, exist_ok=True)

    vname = getattr(target, 'variant_name', 'unknown')
    vdesc = getattr(target, 'variant_desc', '')
    title_suffix = f" — {vname}" + (f" ({vdesc})" if vdesc else "")

    plot_methods = list(methods) + (['GT floor'] if 'GT floor' in results else [])
    gt_floor = results.get('GT floor')

    # ---- 1. Bar chart of metrics ----
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    axes = axes.flatten()
    metric_keys   = ['nll', 'ess', 'w2', 'mmd', 'kl_energy', 'ksd', 'score_rmse']
    metric_labels = ['NLL (KDE, ↓)', 'ESS (KDE, ↑)', 'W2 (↓)',
                      'MMD (↓)', 'KL energy (↓)', 'KSD (↓)', 'Score RMSE (↓)']
    for ax, mk, ml in zip(axes, metric_keys, metric_labels):
        vals  = [min(results[m][mk], 50.0) for m in methods]
        cols  = [C.get(m, '#333333') for m in methods]
        bars  = ax.bar(range(len(methods)), vals, color=cols)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
        ax.set_title(ml, fontsize=10)
        ax.set_ylabel(mk)
        ax.grid(alpha=0.3, axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7)
        if gt_floor is not None and mk in gt_floor and np.isfinite(gt_floor[mk]):
            ax.axhline(gt_floor[mk], color='#888888', ls='--', lw=1.5,
                       label='GT floor', zorder=0)
            ax.legend(fontsize=7, loc='best')
    fig.suptitle(f'DW-4 metrics{title_suffix}', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'metrics_bar.png'), dpi=150)
    _close_fig(fig)

    # ---- 2. 2D particle coordinate heatmaps ----
    with torch.no_grad():
        gt_coords  = DW4Target.all_coords(xt).cpu().numpy()
    lo = gt_coords.min(axis=0) - 0.2
    hi = gt_coords.max(axis=0) + 0.2
    xedges = np.linspace(lo[0], hi[0], 101)
    yedges = np.linspace(lo[1], hi[1], 101)
    def make_hist(coords):
        H, _, _ = np.histogram2d(coords[:, 0], coords[:, 1],
                                 bins=[xedges, yedges], density=True)
        return H.T
    H_true  = make_hist(gt_coords)
    vmax    = max(float(H_true.max()), 1e-12)
    extent  = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    nc = len(plot_methods) + 1
    fig2, axes2 = plt.subplots(1, nc, figsize=(3.8 * nc, 4.2), squeeze=False)
    axes2 = axes2[0]
    im = axes2[0].imshow(H_true, origin='lower', extent=extent,
                         aspect='equal', interpolation='nearest',
                         vmin=0.0, vmax=vmax)
    axes2[0].set_title('True (MALA)', fontsize=9)
    axes2[0].set_xlabel('x'); axes2[0].set_ylabel('y')
    for ci, m in enumerate(plot_methods):
        ax   = axes2[ci + 1]
        sc   = results[m]['samples']
        if len(sc) >= 10:
            coords = DW4Target.all_coords(sc).cpu().numpy()
            H = make_hist(coords)
        else:
            H = np.zeros_like(H_true)
        ax.imshow(H, origin='lower', extent=extent, aspect='equal',
                  interpolation='nearest', vmin=0.0, vmax=vmax)
        ax.set_title(m, fontsize=9)
        ax.set_xlabel('x')
    fig2.colorbar(im, ax=axes2[-1], fraction=0.046, label='density')
    fig2.suptitle(f'Single-particle marginal{title_suffix}', fontsize=12)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, 'heatmaps.png'), dpi=150, bbox_inches='tight')
    _close_fig(fig2)

    # ---- 3. Pairwise distance + energy + radial ----
    def get_pdists(x_tensor):
        r = x_tensor.reshape(-1, 4, 2)
        ds = []
        for i in range(4):
            for j in range(i + 1, 4):
                ds.append(((r[:, i] - r[:, j]) ** 2).sum(-1).sqrt())
        return torch.cat(ds).cpu().numpy()

    for hist_name, get_vals, xlabel in [
        ('pairwise_dist',
         lambda s: get_pdists(s),
         'Inter-particle distance'),
        ('radial',
         lambda s: (s.reshape(-1, 4, 2) ** 2).sum(-1).sqrt().reshape(-1).cpu().numpy(),
         'Particle radius ‖x_i‖'),
        ('energy',
         lambda s: (-target.log_prob(s)).cpu().numpy(),
         'Energy U(x)'),
    ]:
        with torch.no_grad():
            gt_vals = get_vals(xt)
        lo_h = float(np.percentile(gt_vals, 0.5))
        hi_h = float(np.percentile(gt_vals, 99.5))
        bins_h = np.linspace(lo_h, hi_h, 80)
        fig_h, ax_h = plt.subplots(figsize=(9, 4))
        ax_h.hist(gt_vals, bins=bins_h, density=True, alpha=0.35,
                  color='gray', label='True')
        for m in plot_methods:
            sc = results[m]['samples']
            if len(sc) >= 10:
                with torch.no_grad():
                    mv = get_vals(sc)
                ax_h.hist(mv, bins=bins_h, density=True, alpha=0.55,
                         color=C.get(m, '#333333'), label=m, histtype='step', lw=2)
        ax_h.set_xlabel(xlabel)
        ax_h.set_ylabel('Density')
        ax_h.set_title(f'{xlabel}{title_suffix}')
        ax_h.legend(fontsize=8, ncol=2)
        ax_h.grid(alpha=0.3)
        fig_h.tight_layout()
        fig_h.savefig(os.path.join(out_dir, f'{hist_name}.png'), dpi=150)
        _close_fig(fig_h)

    print(f"  Figures saved to {out_dir}/")


# ==============================================================
# Variant definitions for the stress-test suite
# ==============================================================

def make_variants():
    """
    Returns an ordered dict of (name → target) for the stress-test suite.
    All share the hardened base: α=8, c=3, σ=0.15 (except soft_radial).
    """
    base_kw = dict(alpha=8.0, repul_c=3.0, repul_sigma=0.15)

    variants = {}

    # 0. Baseline (hardened DW-4)
    t0 = DW4Target(**base_kw)
    t0.variant_name = 'baseline'
    t0.variant_desc = 'α=8, c=3, σ²=0.0225'
    variants['baseline'] = t0

    # 1. Angular corrugation
    variants['corrugated'] = CorrugatedDW4(beta=3.0, k=4, **base_kw)

    # 2. Soft radial (lower α — override base_kw)
    variants['soft_radial'] = SoftRadialDW4(alpha=0.5, repul_c=3.0, repul_sigma=0.15)

    # 3. Banana distortion
    variants['banana'] = BananaDW4(gamma=3.0, **base_kw)

    # 4. Heavy-tail
    variants['heavy_tail'] = HeavyTailDW4(nu=0.5, **base_kw)

    # 5. Flat-bottomed well
    variants['flat_bottom'] = FlatBottomDW4(delta=0.3, **base_kw)

    return variants


# ==============================================================
# Meta-run: iterate over all variants
# ==============================================================

def meta_run(root_dir='outputs_stress_test',
             methods=None,
             variants=None):
    """
    Run the full benchmark for each target variant, saving results
    to per-variant subdirectories under root_dir.

    Parameters
    ----------
    root_dir : str, root output directory
    methods  : list of str, score estimators (default: Adaptive Leaf-CE, Blended, Tweedie)
    variants : dict of name → DW4Target, or None to use make_variants()

    Returns
    -------
    all_results : dict of variant_name → (results, xt, xr, methods, target)
    """
    if variants is None:
        variants = make_variants()
    if methods is None:
        methods = list(DEFAULT_METHODS)

    os.makedirs(root_dir, exist_ok=True)
    all_results = {}

    # ---- Run each variant ----
    for vname, target in variants.items():
        print(f"\n{'#' * 80}")
        print(f"# VARIANT: {vname}")
        print(f"{'#' * 80}")

        out_dir = os.path.join(root_dir, vname)
        torch.manual_seed(42)  # reset seed per variant for reproducibility
        if DEVICE.type == 'cuda':
            torch.cuda.manual_seed_all(42)

        res, xt, xr, meths, tgt = run(target, out_dir=out_dir, methods=methods)
        plot(res, xt, xr, meths, tgt, out_dir=out_dir)

        all_results[vname] = (res, xt, xr, meths, tgt)

    # ---- Cross-variant comparison table ----
    print("\n\n" + "=" * 100)
    print("CROSS-VARIANT COMPARISON")
    print("=" * 100)

    # Header
    meths_plus = list(methods) + ['GT floor']
    print(f"\n{'Variant':<16s}", end="")
    for m in meths_plus:
        print(f"  {m:>18s}", end="")
    print()

    for metric, mlabel, direction in [
        ('w2',         'W2 (↓)',          'lower'),
        ('nll',        'NLL (↓)',         'lower'),
        ('ess',        'ESS (↑)',         'higher'),
        ('ksd',        'KSD (↓)',         'lower'),
        ('kl_energy',  'KL-E (↓)',        'lower'),
        ('score_rmse', 'RMSE (↓)',        'lower'),
    ]:
        print(f"\n  {mlabel}")
        print(f"  {'─' * 14}", end="")
        for _ in meths_plus:
            print(f"  {'─' * 18}", end="")
        print()
        for vname in variants:
            res = all_results[vname][0]
            print(f"  {vname:<14s}", end="")
            for m in meths_plus:
                v = res[m][metric]
                s = f"{v:18.4f}" if np.isfinite(v) else f"{'N/A':>18s}"
                print(f"  {s}", end="")
            print()

    # ---- Relative advantage table for the requested Gamma/Lambda comparison ----
    print(f"\n\n{'='*80}")
    print("GAMMA RELATIVE ADVANTAGE:  baseline / Gamma  (>1 means Gamma is better)")
    print("="*80)
    print(f"  {'Variant':<16s} {'Pair':<18s} {'W2':>8} {'KSD':>8} {'KL-E':>8} {'RMSE':>8} {'|s|max':>8}")
    print(f"  {'─'*16} {'─'*18} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    pairs = [('HLSI', 'Gamma-HLSI'), ('Leaf-CE-HLSI', 'Leaf-Gamma-HLSI'), ('CE-HLSI', 'Gamma-HLSI')]
    for vname in variants:
        res = all_results[vname][0]
        for base, gamma in pairs:
            if base not in res or gamma not in res:
                continue
            rb = res[base]
            rg = res[gamma]
            def ratio(key):
                b, g = rb[key], rg[key]
                if not np.isfinite(b) or not np.isfinite(g) or g < 1e-15:
                    return float('nan')
                return b / g
            r_ms = rb['ms'] / max(rg['ms'], 1e-15)
            print(f"  {vname:<16s} {base + '/' + gamma:<18s}"
                  f" {ratio('w2'):8.3f}"
                  f" {ratio('ksd'):8.3f}"
                  f" {ratio('kl_energy'):8.3f}"
                  f" {ratio('score_rmse'):8.3f}"
                  f" {r_ms:8.1f}")

    print(f"\n  Ratio > 1 → Gamma/Lambda gate improves that metric relative to the named baseline")
    print(f"  Ratio ≈ 1 → no material gain")
    print(f"  Ratio < 1 → Gamma/Lambda gate hurts that metric")
    print("=" * 80)

    return all_results


# ==============================================================
# Main — controlled by RUN_MODE at the top of the script
# ==============================================================
if RUN_MODE == 'single':
    # Quick smoke test: baseline hardened DW-4 only
    target = DW4Target(alpha=8.0, repul_c=3.0, repul_sigma=0.15)
    target.variant_name = 'baseline'
    target.variant_desc = 'α=8, c=3, σ²=0.0225'
    results, xt, xr, methods, target = run(target)
    plot(results, xt, xr, methods, target)

elif RUN_MODE == 'full':
    # Full 6-variant stress-test suite
    all_results = meta_run()

else:
    print(f"Unknown RUN_MODE='{RUN_MODE}'.  Set to 'single' or 'full'.")
