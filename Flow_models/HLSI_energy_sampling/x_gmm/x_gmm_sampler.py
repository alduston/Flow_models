import os
import math
import time
from collections import OrderedDict
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

# Leaf-HLSI / Leaf-CE-HLSI within-leaf precision for rescued non-PSD directions
# Set to None to estimate a per-reference, per-direction rescue precision from
# a symmetric finite-difference diagnostic of score nonlinearity.
P_LEAF = None

# Finite-difference settings for automatic leaf-precision estimation
FD_DELTA_SCALE = 0.25
FD_DELTA_MIN = 5e-3
FD_DELTA_MAX = 0.50
FD_P_MIN = 1e-3

# ==============================================================
# Batched 2x2 helpers
# ==============================================================
def bmv(A, v):
    """[...,2,2] x [...,2] -> [...,2]"""
    return torch.einsum('...ij,...j->...i', A, v)

def b_inv22(A):
    """Invert [...,2,2] symmetric matrices."""
    a, b, d = A[..., 0, 0], A[..., 0, 1], A[..., 1, 1]
    det = (a * d - b * b).clamp(min=1e-30)
    out = torch.empty_like(A)
    out[..., 0, 0] = d / det
    out[..., 0, 1] = -b / det
    out[..., 1, 0] = -b / det
    out[..., 1, 1] = a / det
    return out


# ==============================================================
# General 2D Gaussian mixtures
# ==============================================================
class GeneralGMM:
    def __init__(self, mus, covs, weights=None, name='General GMM'):
        mus = torch.as_tensor(mus, dtype=torch.get_default_dtype())
        covs = torch.as_tensor(covs, dtype=torch.get_default_dtype())

        assert mus.ndim == 2 and mus.shape[1] == 2
        assert covs.ndim == 3 and covs.shape[1:] == (2, 2)
        assert covs.shape[0] == mus.shape[0]

        self.mus = mus
        self.covs = covs
        self.cov_invs = b_inv22(covs)
        self.chols = torch.linalg.cholesky(covs)
        self.log_dets = torch.logdet(covs)
        self.K = mus.shape[0]
        self.name = name

        if weights is None:
            weights = torch.full((self.K,), 1.0 / self.K, dtype=torch.get_default_dtype())
        else:
            weights = torch.as_tensor(weights, dtype=torch.get_default_dtype())
            weights = weights / weights.sum().clamp(min=1e-30)
        self.weights = weights
        self.log_weights = torch.log(self.weights.clamp(min=1e-30))

    def sample(self, n):
        k = torch.multinomial(self.weights, num_samples=n, replacement=True)
        z = torch.randn(n, 2, dtype=torch.get_default_dtype())
        return self.mus[k] + torch.einsum('nij,nj->ni', self.chols[k], z)

    def _log_comp(self, x):
        diffs = x.unsqueeze(1) - self.mus.unsqueeze(0)                     # [N,K,2]
        maha = torch.einsum('nki,kij,nkj->nk', diffs, self.cov_invs, diffs)
        const = 2.0 * np.log(2.0 * np.pi)
        return self.log_weights.unsqueeze(0) - 0.5 * (const + self.log_dets.unsqueeze(0) + maha)

    def resp(self, x):
        return torch.softmax(self._log_comp(x), dim=-1)

    def component_scores(self, x):
        diffs = x.unsqueeze(1) - self.mus.unsqueeze(0)                     # [N,K,2]
        return -torch.einsum('kij,nkj->nki', self.cov_invs, diffs)         # [N,K,2]

    def log_prob(self, x):
        return torch.logsumexp(self._log_comp(x), dim=1)

    def score(self, x):
        w = self.resp(x)
        sk = self.component_scores(x)
        return (w.unsqueeze(-1) * sk).sum(1)

    def hessian(self, x):
        """
        Returns the positive precision-like Hessian
            H(x) = -∇ s(x) = E_r[P_k] - Cov_r[s_k(x)]
        where P_k = Σ_k^{-1} and r are posterior responsibilities.
        """
        w = self.resp(x)                                                   # [N,K]
        sk = self.component_scores(x)                                      # [N,K,2]
        s_bar = (w.unsqueeze(-1) * sk).sum(1)                              # [N,2]
        centered = sk - s_bar.unsqueeze(1)                                 # [N,K,2]
        cov_term = torch.einsum('nk,nki,nkj->nij', w, centered, centered)  # [N,2,2]
        mean_prec = torch.einsum('nk,kij->nij', w, self.cov_invs)          # [N,2,2]
        return mean_prec - cov_term

    def score_t(self, y, t):
        """
        Exact score of the OU-marginal p_t at time t.
        Supports scalar t or a vector t with one time per sample.
        """
        t = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=y.dtype, device=y.device)
        mus = self.mus.to(y.device)
        covs = self.covs.to(y.device)
        log_weights = self.log_weights.to(y.device)
        eye = torch.eye(2, dtype=y.dtype, device=y.device)
        const = 2.0 * np.log(2.0 * np.pi)

        if t.ndim == 0:
            a = at(t)
            v = vt(t)
            covs_t = (a ** 2) * covs + v * eye.unsqueeze(0)                 # [K,2,2]
            covs_t_inv = b_inv22(covs_t)                                    # [K,2,2]
            log_det_t = torch.logdet(covs_t)                                # [K]
            mus_t = a * mus                                                 # [K,2]

            diffs = y.unsqueeze(1) - mus_t.unsqueeze(0)                     # [N,K,2]
            maha = torch.einsum('nki,kij,nkj->nk', diffs, covs_t_inv, diffs)
            log_comp = log_weights.unsqueeze(0) - 0.5 * (const + log_det_t.unsqueeze(0) + maha)
            w = torch.softmax(log_comp, dim=-1)
            sk = -torch.einsum('kij,nkj->nki', covs_t_inv, diffs)
            return (w.unsqueeze(-1) * sk).sum(1)

        a = at(t).reshape(-1, 1, 1, 1)                                      # [N,1,1,1]
        v = vt(t).reshape(-1, 1, 1, 1)                                      # [N,1,1,1]
        covs_t = (a ** 2) * covs.unsqueeze(0) + v * eye.view(1, 1, 2, 2)   # [N,K,2,2]
        covs_t_inv = b_inv22(covs_t)                                        # [N,K,2,2]
        log_det_t = torch.logdet(covs_t)                                    # [N,K]
        mus_t = at(t).reshape(-1, 1, 1) * mus.unsqueeze(0)                  # [N,K,2]

        diffs = y.unsqueeze(1) - mus_t                                      # [N,K,2]
        maha = torch.einsum('nki,nkij,nkj->nk', diffs, covs_t_inv, diffs)
        log_comp = log_weights.unsqueeze(0) - 0.5 * (const + log_det_t + maha)
        w = torch.softmax(log_comp, dim=-1)
        sk = -torch.einsum('nkij,nkj->nki', covs_t_inv, diffs)
        return (w.unsqueeze(-1) * sk).sum(1)

# ==============================================================
# Specific targets
# ==============================================================
class RotatedGMM(GeneralGMM):
    def __init__(self, d=3.0, eps=0.01, angle_deg=45.0):
        th = angle_deg * np.pi / 180.0
        R = torch.tensor([[np.cos(th), -np.sin(th)],
                          [np.sin(th),  np.cos(th)]], dtype=torch.get_default_dtype())
        D = torch.diag(torch.tensor([1.0, eps], dtype=torch.get_default_dtype()))
        Sig = R @ D @ R.T

        e_sloppy = R[:, 0]
        mus = torch.stack([-d * e_sloppy, d * e_sloppy])
        covs = Sig.unsqueeze(0).repeat(2, 1, 1)

        super().__init__(mus=mus, covs=covs, weights=None, name='Rotated pair')
        self.eps = eps
        self.R = R
        self.primary_stiff_dir = R[:, 1]

class DoubleXGMM(GeneralGMM):
    """
    Two separated local X-shapes:
    start from the original two elongated components and add a crossed
    component at each mean with the opposite diagonal orientation.
    """
    def __init__(self, d=3.0, eps=0.01, angle_deg=45.0):
        th = angle_deg * np.pi / 180.0

        R_plus = torch.tensor([[np.cos(th), -np.sin(th)],
                               [np.sin(th),  np.cos(th)]], dtype=torch.get_default_dtype())
        R_minus = torch.tensor([[np.cos(-th), -np.sin(-th)],
                                [np.sin(-th),  np.cos(-th)]], dtype=torch.get_default_dtype())

        D = torch.diag(torch.tensor([1.0, eps], dtype=torch.get_default_dtype()))
        Sig_plus = R_plus @ D @ R_plus.T
        Sig_minus = R_minus @ D @ R_minus.T

        e_sep = R_plus[:, 0]
        means = torch.stack([-d * e_sep, d * e_sep])

        mus = torch.stack([
            means[0], means[0],
            means[1], means[1],
        ])
        covs = torch.stack([
            Sig_plus, Sig_minus,
            Sig_plus, Sig_minus,
        ])

        super().__init__(mus=mus, covs=covs, weights=None, name='Double-X')
        self.eps = eps
        self.R_plus = R_plus
        self.R_minus = R_minus
        self.primary_stiff_dir = R_plus[:, 1]

# ==============================================================
# OU helpers
# ==============================================================
def at(t):
    return torch.exp(-t)

def vt(t):
    return 1.0 - torch.exp(-2.0 * t)

def snis_w(y, t, xr):
    a = at(t)
    v = vt(t)
    diff = y.unsqueeze(1) - a * xr.unsqueeze(0)
    lw = -0.5 * (diff ** 2).sum(-1) / v
    lw = lw - lw.max(1, keepdim=True).values
    w = lw.exp()
    return w / w.sum(1, keepdim=True).clamp(min=1e-30)


def surrogate_transition_w(y, t, xr, precomp, leaf=False,
                           lmin=1e-4, lmax=1e6):
    """
    Exact transition responsibilities for the local Gaussian/leaf surrogate
    components used by HLSI.

    This changes only the transition weights.  The downstream estimator remains
    the corresponding existing HLSI / Leaf-CE-HLSI routine.

    For ordinary HLSI, trusted positive Hessian directions use the local
    Laplace precision lambda.  Untrusted directions fall back to the Tweedie
    point-mass transition, i.e. transition precision 1/v_t.

    For leaf=True, trusted directions use the ordinary Hessian precision,
    negative-Hessian directions use the rescued leaf precision, and remaining
    directions again use Tweedie point-mass fallback.
    """
    a = at(t)
    a2 = a ** 2
    v = vt(t)
    V = precomp['V']

    if leaf:
        mu = precomp['mu']
        trusted_eig = precomp['trusted_eig']
        trusted = precomp['trusted']
        is_non_psd = precomp['is_non_psd']
        rescued_eig = precomp['rescued_eig']

        hlsi_eig = trusted_eig / (a2 + v * trusted_eig.clamp(min=1e-30))
        dual_eig = rescued_eig / (a2 + v * rescued_eig.clamp(min=1e-30))
        twd_eig = torch.full_like(trusted_eig, 1.0 / v)
        sig_inv_eig = torch.where(
            trusted,
            hlsi_eig,
            torch.where(is_non_psd, dual_eig, twd_eig),
        )
    else:
        s0 = precomp['s0']
        lam = precomp['lam']
        ok = (lam > lmin) & (lam <= lmax)
        lam_g = torch.where(ok, lam, torch.zeros_like(lam))

        s0_eig = torch.einsum('mji,mj->mi', V, s0)
        delta_eig = torch.where(
            ok,
            s0_eig / lam_g.clamp(min=1e-30),
            torch.zeros_like(s0_eig),
        )
        mu = xr + torch.einsum('mij,mj->mi', V, delta_eig)
        sig_inv_eig = torch.where(
            ok,
            lam_g / (a2 + v * lam_g.clamp(min=1e-30)),
            torch.full_like(lam, 1.0 / v),
        )

    # log N(y; a_t mu_i, Sigma_{i,t}) up to constants common to i.
    # Sigma_inv is diagonal in V_i, so logdet contributes sum log eigs.
    disp = y.unsqueeze(1) - a * mu.unsqueeze(0)
    disp_eig = torch.einsum('mji,nmj->nmi', V, disp)
    qform = (sig_inv_eig.unsqueeze(0) * disp_eig.square()).sum(-1)
    logdet = torch.log(sig_inv_eig.clamp(min=1e-30)).sum(-1)
    lw = 0.5 * logdet.unsqueeze(0) - 0.5 * qform
    lw = lw - lw.max(1, keepdim=True).values
    w = lw.exp()
    return w / w.sum(1, keepdim=True).clamp(min=1e-30)

# ==============================================================
# Score estimators
# ==============================================================
def est_tweedie(y, t, xr, w):
    disp = y.unsqueeze(1) - at(t) * xr.unsqueeze(0)
    return -(w.unsqueeze(2) * disp).sum(1) / vt(t)

def est_tsi(y, t, xr, w, gmm, s0_ref=None):
    s0 = gmm.score(xr) if s0_ref is None else s0_ref
    return (w.unsqueeze(2) * s0.unsqueeze(0)).sum(1) / at(t)

def est_hlsi(y, t, xr, w, gmm, lmin=1e-4, lmax=1e6, precomp=None):
    """HLSI with full 2x2 Hessian eigenbasis gating."""
    a = at(t)
    a2 = a ** 2
    v = vt(t)

    if precomp is None:
        H = gmm.hessian(xr)
        s0 = gmm.score(xr)
        lam, V = torch.linalg.eigh(H)
    else:
        s0 = precomp['s0']
        lam = precomp['lam']
        V = precomp['V']

    ok = (lam > lmin) & (lam <= lmax)
    lam_g = torch.where(ok, lam, torch.zeros_like(lam))

    s0_eig = torch.einsum('mji,mj->mi', V, s0)
    delta_eig = torch.where(ok, s0_eig / lam_g.clamp(min=1e-30), torch.zeros_like(s0_eig))
    delta = torch.einsum('mij,mj->mi', V, delta_eig)
    mu = xr + delta

    sig_inv_eig = torch.where(
        ok,
        lam_g / (a2 + v * lam_g.clamp(min=1e-30)),
        torch.full_like(lam, 1.0 / v)
    )

    disp = y.unsqueeze(1) - a * mu.unsqueeze(0)
    disp_eig = torch.einsum('mji,nmj->nmi', V, disp)
    sc_eig = sig_inv_eig.unsqueeze(0) * disp_eig
    comp = -torch.einsum('mij,nmj->nmi', V, sc_eig)

    return (w.unsqueeze(2) * comp).sum(1)



def estimate_leaf_precision_fd(gmm, xr, eigvals, eigvecs,
                               fd_delta_scale=FD_DELTA_SCALE,
                               fd_delta_min=FD_DELTA_MIN,
                               fd_delta_max=FD_DELTA_MAX,
                               p_min=FD_P_MIN,
                               p_max=1e6):
    """
    Estimate a rescue precision for each negative-Hessian eigendirection using
    the symmetric finite-difference growth of the residual to the affine score
    model along that eigendirection.

    For a symmetric two-leaf model along coordinate u, the directional score
    satisfies q(u) = -lambda*u - c*u^3 + O(u^5) with c = (p-lambda)^2 / 3.
    Using the centered odd residual

        D(delta) = [q(delta) - q(-delta) + 2*lambda*delta] / (2*delta^3),

    we have D(delta) ≈ -c for small delta, so

        p ≈ lambda + sqrt(max(0, -3 D(delta))).

    The estimate is only used on negative-Hessian directions. If the cubic
    residual is too small/noisy to imply a positive precision, we fall back to
    a conservative positive value based on |lambda|.
    """
    M, D = xr.shape
    out = torch.zeros_like(eigvals)

    for j in range(D):
        lam_j = eigvals[:, j]
        bad = lam_j < 0
        if not bad.any():
            continue

        v_j = eigvecs[:, :, j]
        delta = fd_delta_scale / torch.sqrt(lam_j.abs() + 1.0)
        delta = delta.clamp(min=fd_delta_min, max=fd_delta_max)

        x_plus = xr + delta.unsqueeze(1) * v_j
        x_minus = xr - delta.unsqueeze(1) * v_j

        q_plus = (gmm.score(x_plus) * v_j).sum(dim=1)
        q_minus = (gmm.score(x_minus) * v_j).sum(dim=1)

        D_delta = (q_plus - q_minus + 2.0 * lam_j * delta) / (2.0 * delta.pow(3).clamp(min=1e-30))
        rad = (-3.0 * D_delta).clamp(min=0.0)
        p_est = lam_j + torch.sqrt(rad)

        fallback = (lam_j.abs() + 1.0).clamp(min=p_min, max=p_max)
        good = torch.isfinite(p_est) & (p_est > p_min)
        p_est = torch.where(good, p_est, fallback)
        p_est = p_est.clamp(min=p_min, max=p_max)

        out[:, j] = torch.where(bad, p_est, out[:, j])

    return out

def est_ce_hlsi(y, t, xr, w, gmm, lmin=1e-4, lmax=1e6, precomp=None):
    """CE-HLSI: full-matrix measurable gate from SNIS-mean Hessian."""
    a = at(t)
    a2 = a ** 2
    v = vt(t)

    s_twd = est_tweedie(y, t, xr, w)
    if precomp is None:
        s_tsi = est_tsi(y, t, xr, w, gmm)
        H = gmm.hessian(xr)
        lam_ref, V_ref = torch.linalg.eigh(H)
        lam_ref = lam_ref.clamp(min=lmin, max=lmax)
        H_ref = torch.einsum('mij,mj,mkj->mik', V_ref, lam_ref, V_ref)
    else:
        s_tsi = est_tsi(y, t, xr, w, gmm, s0_ref=precomp['s0'])
        H_ref = precomp['H_ce']

    Hb = (w.unsqueeze(-1).unsqueeze(-1) * H_ref.unsqueeze(0)).sum(1)
    Hb = 0.5 * (Hb + Hb.transpose(-1, -2))

    lam, V = torch.linalg.eigh(Hb)
    lam = lam.clamp(min=lmin, max=lmax)
    Hb = torch.einsum('nij,nj,nkj->nik', V, lam, V)

    eye = torch.eye(2, dtype=Hb.dtype, device=Hb.device).unsqueeze(0)
    M_mat = a2 * eye + v * Hb
    A = a2 * b_inv22(M_mat)

    I_A = eye - A
    return bmv(I_A, s_twd) + bmv(A, s_tsi)

def precompute_leaf_hlsi(gmm, xr, lmin=1e-4, lmax=1e6, p_leaf=P_LEAF):
    """Precompute spectral data for Leaf-HLSI / Leaf-CE-HLSI."""
    H = gmm.hessian(xr)
    H = 0.5 * (H + H.transpose(1, 2))
    s0 = gmm.score(xr)

    eigvals, eigvecs = torch.linalg.eigh(H)
    trusted = (eigvals >= lmin) & (eigvals <= lmax)
    is_non_psd = eigvals < 0

    trusted_eig = torch.where(trusted, eigvals, torch.zeros_like(eigvals))

    if p_leaf is None:
        auto_leaf_eig = estimate_leaf_precision_fd(gmm, xr, eigvals, eigvecs, p_max=lmax)
        leaf_precision_mode = 'finite-difference'
    else:
        auto_leaf_eig = torch.full_like(eigvals, float(p_leaf))
        leaf_precision_mode = 'fixed'

    rescued_eig = torch.where(
        trusted, eigvals,
        torch.where(is_non_psd, auto_leaf_eig, torch.zeros_like(eigvals))
    )

    s0e = torch.einsum('mji,mj->mi', eigvecs, s0)
    delta_eig = torch.where(
        trusted,
        s0e / trusted_eig.clamp(min=lmin),
        torch.zeros_like(s0e),
    )
    mu = xr + torch.einsum('mij,mj->mi', eigvecs, delta_eig)

    P_leaf_ref = torch.einsum('mij,mj,mkj->mik', eigvecs, rescued_eig, eigvecs)
    H_ce = torch.einsum('mij,mj,mkj->mik', eigvecs, eigvals.clamp(min=lmin, max=lmax), eigvecs)
    return {
        's0': s0,
        'lam': eigvals,
        'V': eigvecs,
        'trusted_eig': trusted_eig,
        'rescued_eig': rescued_eig,
        'mu': mu,
        'P_leaf': P_leaf_ref,
        'trusted': trusted,
        'is_non_psd': is_non_psd,
        'H_ce': H_ce,
        'leaf_precision_mode': leaf_precision_mode,
    }

def est_leaf_hlsi(y, t, xr, w, gmm=None, precomp=None, lmin=1e-4, lmax=1e6, p_leaf=P_LEAF):
    """Leaf-HLSI / dual-likelihood HLSI."""
    if precomp is None:
        if gmm is None:
            raise ValueError('est_leaf_hlsi needs either gmm or precomp')
        precomp = precompute_leaf_hlsi(gmm, xr, lmin=lmin, lmax=lmax, p_leaf=p_leaf)

    a = at(t)
    a2 = a ** 2
    v = vt(t)
    V = precomp['V']
    trusted_eig = precomp['trusted_eig']
    trusted = precomp['trusted']
    is_non_psd = precomp['is_non_psd']
    mu = precomp['mu']

    hlsi_eig = trusted_eig / (a2 + v * trusted_eig.clamp(min=1e-30))
    dual_base = precomp['rescued_eig']
    dual_eig = dual_base / (a2 + v * dual_base)
    twd_eig = torch.full_like(trusted_eig, 1.0 / v)
    sigmainv_eig = torch.where(trusted, hlsi_eig, torch.where(is_non_psd, dual_eig, twd_eig))

    disp = y.unsqueeze(1) - a * mu.unsqueeze(0)
    disp_e = torch.einsum('mji,nmj->nmi', V, disp)
    comp = -torch.einsum('mij,nmj->nmi', V, sigmainv_eig.unsqueeze(0) * disp_e)
    return (w.unsqueeze(2) * comp).sum(1)

def est_leaf_ce_hlsi(y, t, xr, w, gmm=None, precomp=None, lmin=1e-4, lmax=1e6, p_leaf=P_LEAF):
    """Measurable CE analogue of Leaf-HLSI."""
    if precomp is None:
        if gmm is None:
            raise ValueError('est_leaf_ce_hlsi needs either gmm or precomp')
        precomp = precompute_leaf_hlsi(gmm, xr, lmin=lmin, lmax=lmax, p_leaf=p_leaf)

    a = at(t)
    a2 = a ** 2
    v = vt(t)
    s_twd = est_tweedie(y, t, xr, w)
    s_tsi = est_tsi(y, t, xr, w, gmm, s0_ref=precomp['s0'])

    Pbar = (w.unsqueeze(-1).unsqueeze(-1) * precomp['P_leaf'].unsqueeze(0)).sum(1)
    Pbar = 0.5 * (Pbar + Pbar.transpose(1, 2))

    lam, V = torch.linalg.eigh(Pbar)
    gate = a2 / (a2 + v * lam + 1e-30)
    delta = s_tsi - s_twd
    delta_e = torch.einsum('nji,nj->ni', V, delta)
    gated_delta = torch.einsum('nij,nj->ni', V, gate * delta_e)
    return s_twd + gated_delta

def est_blended(y, t, xr, w, gmm, s0_ref=None):
    """Diagonal (per-coordinate) gate — can't resolve rotated eigendirections."""
    a = at(t)
    v = vt(t)
    s0 = gmm.score(xr) if s0_ref is None else s0_ref

    tsi = s0.unsqueeze(0) / a
    twd = -(y.unsqueeze(1) - a * xr.unsqueeze(0)) / v

    am = (w.unsqueeze(2) * tsi).sum(1)
    bm = (w.unsqueeze(2) * twd).sum(1)

    ac = tsi - am.unsqueeze(1)
    bc = twd - bm.unsqueeze(1)
    va = (w.unsqueeze(2) * ac ** 2).sum(1).clamp(min=1e-30)
    vb = (w.unsqueeze(2) * bc ** 2).sum(1).clamp(min=1e-30)
    cab = (w.unsqueeze(2) * ac * bc).sum(1)
    den = (va + vb - 2 * cab).clamp(min=1e-20)
    g = ((va - cab) / den).clamp(0, 1)
    return (1 - g) * am + g * bm

# ==============================================================
# Lambda/Gamma-HLSI gate and factorized sampler construction
# ==============================================================
def _precision_cache(precomp, xr, hessian_processing='base', lmin=1e-4, lmax=1e6):
    """
    Return the PSD local precision objects P_i used by the chosen closure.

    base           : band-clamped positive Hessian precision
    leaf/adaptive  : leaf-repaired precision, including rescued non-PSD dirs
    """
    key = f"_precision_cache::{hessian_processing}::{float(lmin):.3e}::{float(lmax):.3e}"
    if key in precomp:
        return precomp[key]

    V = precomp['V']
    s0 = precomp['s0']
    lam = precomp['lam']

    if hessian_processing == 'base':
        p_eig = lam.clamp(min=lmin, max=lmax)
    elif hessian_processing in ('leaf', 'adaptive_leaf'):
        p_eig = precomp['rescued_eig'].clamp(min=0.0, max=lmax)
    else:
        raise ValueError(f"Unknown hessian_processing={hessian_processing!r}")

    P_mat = torch.einsum('mij,mj,mkj->mik', V, p_eig, V)
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


def est_lambda_hlsi(y, t, xr, w, gmm=None, precomp=None,
                    hessian_processing='base', lmin=1e-4, lmax=1e6,
                    project_lambda=True, m_reg=1e-8, pinv_rtol=1e-6):
    """
    Lambda/Gamma-HLSI gate from the local surrogate regression closure.

    Given the standard CE-HLSI ingredients {x_i, s_i, P_i, rho_i}, this builds
    M_q = E_q[E E^T] and N_q = E_q[E D^T], solves
    Lambda_q^* = -N_q^T M_q^dagger, and applies
    score = Tweedie + (Lambda_q^*)^dagger (TSI - Tweedie).
    """
    if precomp is None:
        if gmm is None:
            raise ValueError('est_lambda_hlsi needs either gmm or precomp')
        precomp = precompute_leaf_hlsi(gmm, xr, lmin=lmin, lmax=lmax, p_leaf=P_LEAF)

    a = at(t)
    v = vt(t)
    a2 = a ** 2
    tau = a2 / v
    B, d = y.shape
    dtype, device = y.dtype, y.device
    I = torch.eye(d, dtype=dtype, device=device)

    cache = _precision_cache(precomp, xr, hessian_processing=hessian_processing,
                             lmin=lmin, lmax=lmax)
    V = cache['V']
    p_eig = cache['P_eig']
    P_mu = cache['P_mu']

    s_twd = est_tweedie(y, t, xr, w)
    s_tsi = est_tsi(y, t, xr, w, gmm, s0_ref=precomp['s0'])
    delta = s_tsi - s_twd

    sigma_eig = 1.0 / (p_eig + tau).clamp(min=1e-30)
    Sigma = torch.einsum('mij,mj,mkj->mik', V, sigma_eig, V)

    rhs = P_mu.unsqueeze(0) + (a / v) * y.unsqueeze(1)
    rhs_eig = torch.einsum('mji,bmj->bmi', V, rhs)
    post_mean = torch.einsum('mij,bmj->bmi', V, sigma_eig.unsqueeze(0) * rhs_eig)

    mE = -(y.unsqueeze(1) - a * post_mean) / v
    Ebar = (w.unsqueeze(-1) * mE).sum(1)
    dE = mE - Ebar.unsqueeze(1)

    M_within = (a2 / (v ** 2)) * torch.einsum('bm,mij->bij', w, Sigma)
    M_between = torch.einsum('bm,bmi,bmj->bij', w, dE, dE)
    M_q = M_within + M_between
    M_q = 0.5 * (M_q + M_q.transpose(-1, -2))

    mE_eig = torch.einsum('mji,bmj->bmi', V, mE)
    P_mE = torch.einsum('mij,bmj->bmi', V, p_eig.unsqueeze(0) * mE_eig)
    K_mE = mE + (1.0 / tau) * P_mE

    Ebar_eig = torch.einsum('mji,bj->bmi', V, Ebar)
    P_Ebar_components = torch.einsum('mij,bmj->bmi', V, p_eig.unsqueeze(0) * Ebar_eig)
    Pbar_Ebar = (w.unsqueeze(-1) * P_Ebar_components).sum(1)
    Kbar_Ebar = Ebar + (1.0 / tau) * Pbar_Ebar

    C_ED_bw = -torch.einsum('bm,bmi,bmj->bij',
                            w, dE, K_mE - Kbar_Ebar.unsqueeze(1))
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
    ('Leaf-CE-HLSI',      {'transition_weights': 'ou',        'hessian_processing': 'leaf',          'gate': 'ce'}),
    ('CE-HLSI',           {'transition_weights': 'ou',        'hessian_processing': 'base',          'gate': 'ce'}),
    ('HLSI',              {'transition_weights': 'ou',        'hessian_processing': 'base',          'gate': 'hlsi'}),
    ('Leaf-Gamma-HLSI',   {'transition_weights': 'ou',        'hessian_processing': 'leaf',          'gate': 'lambda', 'project_lambda': True}),
    ('Gamma-HLSI',        {'transition_weights': 'ou',        'hessian_processing': 'base',          'gate': 'lambda', 'project_lambda': True}),
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
        cfg.pop('name', None)
        cfg.pop('label', None)
        cfg.pop('method', None)
        return name, cfg
    raise TypeError(f"Method entries must be strings or dicts, got {type(method)}")


def make_sampler_score_fn(method, gmm, xr, precomp, precomp_adaptive=None,
                          lmin=1e-4, lmax=1e6):
    """Build a score closure from factorized sampler ingredients."""
    name, cfg = _resolve_sampler_config(method)
    hproc = cfg.get('hessian_processing', 'base')
    gate = cfg.get('gate', 'hlsi')
    weight_mode = cfg.get('transition_weights', 'ou')

    if hproc == 'adaptive_leaf':
        pc = precomp_adaptive if precomp_adaptive is not None else precomp
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
            return est_tsi(y, t, xr, w, gmm, s0_ref=pc['s0'])
        if gate == 'blend':
            return est_blended(y, t, xr, w, gmm, s0_ref=pc['s0'])
        if gate == 'hlsi':
            if hproc in ('leaf', 'adaptive_leaf'):
                return est_leaf_hlsi(y, t, xr, w, gmm=gmm, precomp=pc, lmin=lmin, lmax=lmax)
            return est_hlsi(y, t, xr, w, gmm, precomp=pc, lmin=lmin, lmax=lmax)
        if gate == 'ce':
            if hproc in ('leaf', 'adaptive_leaf'):
                return est_leaf_ce_hlsi(y, t, xr, w, gmm=gmm, precomp=pc, lmin=lmin, lmax=lmax)
            return est_ce_hlsi(y, t, xr, w, gmm, precomp=pc, lmin=lmin, lmax=lmax)
        if gate in ('lambda', 'gamma', 'Lambda-HLSI', 'Gamma-HLSI'):
            return est_lambda_hlsi(
                y, t, xr, w, gmm=gmm, precomp=pc,
                hessian_processing=hproc,
                lmin=lmin, lmax=lmax,
                project_lambda=cfg.get('project_lambda', True),
                m_reg=cfg.get('m_reg', 1e-8),
                pinv_rtol=cfg.get('pinv_rtol', 1e-6),
            )
        raise ValueError(f"Unknown gate={gate!r} for {name}")

    return name, fn

# ==============================================================
# Heun-PC reverse SDE + Tweedie denoising
# ==============================================================
def heun_sde(score_fn, n, n_steps=120, t_max=3.0, t_min=0.015):
    ts = torch.linspace(t_max, t_min, n_steps + 1, dtype=torch.get_default_dtype())
    y = torch.randn(n, 2, dtype=torch.get_default_dtype())
    ms = 0.0
    fail = False

    for i in range(n_steps):
        tc, tn = ts[i], ts[i + 1]
        h = tc - tn
        s1 = score_fn(y, tc)
        ms = max(ms, s1.abs().max().item())
        if torch.isnan(s1).any() or torch.isinf(s1).any():
            fail = True
            break
        d1 = y + 2 * s1
        z = torch.randn_like(y)
        noise = (2 * h).sqrt() * z
        yh = y + h * d1 + noise
        s2 = score_fn(yh, tn)
        if torch.isnan(s2).any() or torch.isinf(s2).any():
            fail = True
            break
        d2 = yh + 2 * s2
        y = y + 0.5 * h * (d1 + d2) + noise
        if torch.isnan(y).any() or torch.isinf(y).any():
            fail = True
            break

    if not fail:
        tf = torch.tensor(t_min, dtype=torch.get_default_dtype())
        sf = score_fn(y, tf)
        ms = max(ms, sf.abs().max().item())
        if torch.isnan(sf).any() or torch.isinf(sf).any():
            fail = True
        else:
            y = (y + vt(tf) * sf) / at(tf)
    return y, ms, fail

# ==============================================================
# Metrics
# ==============================================================
def mmd(X, Y, bws=(0.1, 0.5, 1.0, 2.0, 5.0)):
    n = min(len(X), 2000)
    m = min(len(Y), 2000)
    X = X[:n]
    Y = Y[:m]
    xx = torch.cdist(X, X) ** 2
    yy = torch.cdist(Y, Y) ** 2
    xy = torch.cdist(X, Y) ** 2
    gammas = [0.5 / (b ** 2) for b in bws]
    v = sum(((-g * xx).exp().mean() + (-g * yy).exp().mean() - 2 * (-g * xy).exp().mean())
            for g in gammas)
    return (v / len(gammas)).item()

def kl_histogram(X, Y, bins=180, smoothing=1e-12):
    """
    Approximate KL(P||Q) with 2D histograms.
    X defines P (ground truth), Y defines Q (model samples).
    Range and normalization are set from X only.
    """
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()

    if len(X_np) < 10 or len(Y_np) < 10:
        return float('inf')

    lo = X_np.min(axis=0)
    hi = X_np.max(axis=0)
    span = np.maximum(hi - lo, 1e-8)
    lo = lo - 0.05 * span
    hi = hi + 0.05 * span

    Hx, xedges, yedges = np.histogram2d(
        X_np[:, 0], X_np[:, 1],
        bins=bins,
        range=[[lo[0], hi[0]], [lo[1], hi[1]]],
        density=False,
    )
    Hy, _, _ = np.histogram2d(
        Y_np[:, 0], Y_np[:, 1],
        bins=[xedges, yedges],
        density=False,
    )

    px = Hx + smoothing
    py = Hy + smoothing
    px /= px.sum()
    py /= py.sum()
    return float(np.sum(px * (np.log(px) - np.log(py))))

def _rbf_gram(X, Y, h):
    d2 = torch.cdist(X, Y) ** 2
    return torch.exp(-d2 / (2 * h * h)), d2

def _median_positive_distance(X, Y=None):
    D = torch.cdist(X, X if Y is None else Y)
    pos = D[D > 0]
    if pos.numel() == 0:
        return 1.0
    med = torch.median(pos).item()
    return med if np.isfinite(med) and med > 1e-12 else 1.0

def _sinkhorn_cost_uniform(X, Y, reg=None, max_iter=100, tol=1e-6):
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        return float('inf')

    C = torch.cdist(X, Y) ** 2
    if reg is None:
        c_med = torch.median(C[C > 0]).item() if torch.any(C > 0) else 1.0
        reg = max(5e-3, 0.05 * c_med)

    log_a = torch.full((n,), -math.log(n), dtype=X.dtype, device=X.device)
    log_b = torch.full((m,), -math.log(m), dtype=X.dtype, device=X.device)
    f = torch.zeros(n, dtype=X.dtype, device=X.device)
    g = torch.zeros(m, dtype=X.dtype, device=X.device)

    for _ in range(max_iter):
        f_prev = f.clone()
        g_prev = g.clone()
        f = reg * (log_a - torch.logsumexp((g.unsqueeze(0) - C) / reg, dim=1))
        g = reg * (log_b - torch.logsumexp((f.unsqueeze(1) - C) / reg, dim=0))
        err = max((f - f_prev).abs().max().item(), (g - g_prev).abs().max().item())
        if err < tol:
            break

    log_pi = (f.unsqueeze(1) + g.unsqueeze(0) - C) / reg
    pi = torch.exp(log_pi)
    pi = pi / pi.sum().clamp(min=1e-30)
    return float((pi * C).sum().item())

def sinkhorn_w2_distance(X, Y, max_points=400, reg=None, max_iter=100):
    n = min(len(X), max_points)
    m = min(len(Y), max_points)
    if n < 5 or m < 5:
        return float('inf')

    X = X[:n]
    Y = Y[:m]
    cost_xy = _sinkhorn_cost_uniform(X, Y, reg=reg, max_iter=max_iter)
    cost_xx = _sinkhorn_cost_uniform(X, X, reg=reg, max_iter=max_iter)
    cost_yy = _sinkhorn_cost_uniform(Y, Y, reg=reg, max_iter=max_iter)
    sinkhorn2 = max(cost_xy - 0.5 * cost_xx - 0.5 * cost_yy, 0.0)
    return float(np.sqrt(sinkhorn2))

def target_nll(samples, target_log_prob_fn, max_points=2000):
    n = min(len(samples), max_points)
    X = samples[:n]
    if n < 5:
        return float('inf')
    with torch.no_grad():
        return float((-target_log_prob_fn(X)).mean().item())


def knn_entropy(samples, max_points=2000, k=5, jitter=1e-12):
    """
    Kozachenko-Leonenko kNN entropy estimator for differential entropy.
    Uses rho_i = 2 * distance to the k-th nearest neighbor under the Euclidean norm:

        H(q) ≈ ψ(n) - ψ(k) + log V_d + d E[log rho_i],

    where V_d is the unit-ball volume in R^d.
    """
    n = min(len(samples), max_points)
    X = samples[:n]
    if n <= k:
        return float('inf')

    d = X.shape[1]
    with torch.no_grad():
        dists = torch.cdist(X, X)
        mask = torch.eye(n, dtype=torch.bool, device=X.device)
        dists = dists.masked_fill(mask, float('inf'))
        eps = dists.kthvalue(k, dim=1).values.clamp(min=jitter)

        psi_n = torch.special.digamma(torch.tensor(float(n), dtype=X.dtype, device=X.device))
        psi_k = torch.special.digamma(torch.tensor(float(k), dtype=X.dtype, device=X.device))
        log_unit_ball = 0.5 * d * math.log(math.pi) - math.lgamma(0.5 * d + 1.0)
        H = psi_n - psi_k + log_unit_ball + d * torch.log(2.0 * eps).mean()
        return float(H.item())


def reverse_kl_knn(samples, target_log_prob_fn, max_points=2000, k=5):
    """
    Estimate KL(q || p) via

        KL(q || p) = -H(q) - E_q[log p(X)],

    using a kNN entropy estimator for H(q) and exact target log-density for log p.
    This is substantially more stable here than same-sample KDE evaluation.
    """
    n = min(len(samples), max_points)
    X = samples[:n]
    if n <= k:
        return float('inf')

    with torch.no_grad():
        Hq = knn_entropy(X, max_points=n, k=k)
        cross_ent = float((-target_log_prob_fn(X)).mean().item())
        return cross_ent - Hq

def ksd_imq_or_rbf(samples, score_fn, bandwidth='median'):
    """
    Kernel Stein discrepancy squared estimate with RBF kernel.
    score_fn: function mapping [N,2] -> [N,2], the target score ∇ log p.
    """
    X = samples
    n = min(len(X), 1500)
    X = X[:n]
    if n < 5:
        return float('inf')

    with torch.no_grad():
        S = score_fn(X)  # [n, d]
        d = X.shape[1]

        dists = torch.cdist(X, X)
        med = torch.median(dists[dists > 0]) if torch.any(dists > 0) else torch.tensor(1.0, dtype=X.dtype)
        h = med.item()
        if not np.isfinite(h) or h <= 1e-12:
            h = 1.0
        if bandwidth != 'median':
            h = float(bandwidth)

        K, d2 = _rbf_gram(X, X, h)                     # [n,n]
        diff = X.unsqueeze(1) - X.unsqueeze(0)         # [n,n,d]

        s_dot = S @ S.T                                # [n,n]
        term1 = K * s_dot

        sdiff = S.unsqueeze(1) - S.unsqueeze(0)        # [n,n,d]
        term2 = -(K.unsqueeze(-1) * sdiff * diff).sum(-1) / (h * h)

        term4 = K * (d / (h * h) - d2 / (h ** 4))

        U = term1 + term2 + term4
        ksd2 = U.mean()
        return float(torch.sqrt(torch.clamp(ksd2, min=0.0)).item())

def score_rmse_forward_process(score_hat_fn, gmm, n_eval=6000, t_min=0.015, t_max=3.0, n_time_grid=120):
    """
    Estimate E_{t, x_t} ||s*(x_t,t) - s_hat(x_t,t)|| where
    x_t is produced by applying the forward OU process to true samples x_0.
    The time index is sampled uniformly from a log-spaced grid on [t_min, t_max].
    """
    with torch.no_grad():
        x0 = gmm.sample(n_eval)
        t_grid = torch.exp(torch.linspace(math.log(t_min), math.log(t_max), n_time_grid, dtype=x0.dtype))
        idx = torch.randint(0, n_time_grid, (n_eval,))
        t = t_grid[idx]
        a = at(t).unsqueeze(-1)
        v = vt(t).unsqueeze(-1)
        xt = a * x0 + torch.sqrt(v) * torch.randn_like(x0)

        s_true = gmm.score_t(xt, t)
        s_hat = torch.empty_like(s_true)

        # group by time-grid index so the estimator interface can keep scalar t inputs
        for j in idx.unique(sorted=True):
            mask = idx == j
            tj = t_grid[j]
            s_hat[mask] = score_hat_fn(xt[mask], tj)

        err2 = ((s_true - s_hat) ** 2).sum(dim=1)
        return float(torch.sqrt(err2.mean()).item())

# ==============================================================
# Run
# ==============================================================
def run(methods=None):
    eps_list = [1.00, .5, 0.2, .08, .025, .01, .003] #, 1e-3, 2e-4]
    if methods is None:
        methods = list(DEFAULT_METHODS)
    method_entries = list(methods)
    methods = [_resolve_sampler_config(m)[0] for m in method_entries]

    NR, NS, NT = 200, 10000, 5000
    T_MAX = 3.0
    T_MIN = 0.00025
    N_TIME_GRID = 1205

    data = {e: {} for e in eps_list}
    true = {}

    os.makedirs('outputs', exist_ok=True)

    for eps in eps_list:
        gmm = DoubleXGMM(d=3.0, eps=eps, angle_deg=45.0)
        xr = gmm.sample(NR)
        xt = gmm.sample(NT)
        true[eps] = xt

        precomp = precompute_leaf_hlsi(gmm, xr, lmin=1e-4, lmax=1e6, p_leaf=P_LEAF)
        # In this X-GMM script, P_LEAF=None already means adaptive finite-difference
        # leaf precision. Keep a separate object so the config axis is explicit.
        precomp_adaptive = precompute_leaf_hlsi(gmm, xr, lmin=1e-4, lmax=1e6, p_leaf=None)

        if precomp['is_non_psd'].any():
            rescued_vals = precomp['rescued_eig'][precomp['is_non_psd']]
            print(f"    leaf precision mode: {precomp['leaf_precision_mode']}  mean={rescued_vals.mean().item():.3f}  med={rescued_vals.median().item():.3f}  min={rescued_vals.min().item():.3f}  max={rescued_vals.max().item():.3f}")

        print(f"\n{'=' * 88}")
        print(f"  eps={eps:.0e}  1/eps={1/eps:.0e}  (double-X target, ±45° arms)")
        print(f"  methods={methods}")
        print(f"{'=' * 88}")

        for method_entry, m in zip(method_entries, methods):
            _, fn = make_sampler_score_fn(method_entry, gmm, xr, precomp, precomp_adaptive,
                                          lmin=1e-4, lmax=1e6)
            t0 = time.time()
            samp, ms, fail = heun_sde(fn, NS, n_steps=N_TIME_GRID, t_max=T_MAX, t_min=T_MIN)
            dt = time.time() - t0

            ok = ~(torch.isnan(samp).any(1) | torch.isinf(samp).any(1))
            sc = samp[ok]
            nv = ok.sum().item()

            if nv > 20:
                mv = mmd(sc, xt)
                klv = kl_histogram(xt, sc)
                rklv = reverse_kl_knn(sc, gmm.log_prob, k=5)
                nllv = target_nll(sc, gmm.log_prob)
                sw2v = sinkhorn_w2_distance(sc, xt)
                ksdv = ksd_imq_or_rbf(sc, gmm.score)
                srmsev = score_rmse_forward_process(fn, gmm, n_eval=6000, t_min=T_MIN, t_max=T_MAX, n_time_grid=N_TIME_GRID)
            else:
                mv = float('inf')
                klv = float('inf')
                rklv = float('inf')
                nllv = float('inf')
                sw2v = float('inf')
                ksdv = float('inf')
                srmsev = float('inf')

            np_ = 100.0 * (1.0 - nv / NS)

            data[eps][m] = dict(
                samples=sc.detach(),
                mmd=mv,
                kl=klv,
                reverse_kl=rklv,
                target_nll=nllv,
                sinkhorn_w2=sw2v,
                ksd=ksdv,
                score_rmse=srmsev,
                ms=ms,
                nan=np_,
            )
            tag = "FAIL" if fail else "ok"
            print(
                f"  {m:<20s} "
                f"MMD={mv:10.6f}  KL={klv:10.6f}  rKL={rklv:10.6f}  NLL={nllv:10.6f}  sW2={sw2v:10.6f}  "
                f"KSD={ksdv:10.6f}  RMSE={srmsev:10.6f}  |s|={ms:10.1f}  NaN%={np_:5.1f}  [{dt:5.1f}s] {tag}"
            )

    return data, true, eps_list, methods

# ==============================================================
# Plotting helpers
# ==============================================================
def make_gt_hist2d(gt_np, bins=170):
    lo = gt_np.min(axis=0)
    hi = gt_np.max(axis=0)
    span = np.maximum(hi - lo, 1e-8)
    lo = lo - 0.08 * span
    hi = hi + 0.08 * span

    H, xedges, yedges = np.histogram2d(
        gt_np[:, 0], gt_np[:, 1],
        bins=bins,
        range=[[lo[0], hi[0]], [lo[1], hi[1]]],
        density=True,
    )
    return H.T, xedges, yedges, lo, hi

def hist2d_on_gt_grid(samples_np, xedges, yedges):
    H, _, _ = np.histogram2d(
        samples_np[:, 0], samples_np[:, 1],
        bins=[xedges, yedges],
        density=True,
    )
    return H.T

# ==============================================================
# Plot
# ==============================================================
def display_saved_figure(path, width=None):
    try:
        from IPython.display import Image, display
        print(f"\nDisplaying: {path}")
        display(Image(filename=path, width=width))
    except Exception:
        print(f"Saved figure: {path}")


def plot(data, true, eps_list, methods):
    C = {'Tweedie': '#1f77b4', 'TSI': '#d62728', 'HLSI': '#2ca02c',
         'Surrogate-HLSI': '#98df8a',
         'Leaf-HLSI': '#17becf', 'CE-HLSI': '#9467bd', 'Leaf-CE-HLSI': '#8c564b',
         'Gamma-HLSI': '#bcbd22', 'Lambda-HLSI': '#bcbd22',
         'Leaf-Gamma-HLSI': '#e377c2', 'Leaf-Lambda-HLSI': '#e377c2',
         'Surrogate-Gamma-HLSI': '#dbdb8d',
         'Surrogate-Leaf-CE-HLSI': '#c49c94',
         'Surrogate-Leaf-Gamma-HLSI': '#f7b6d2',
         'Adaptive Leaf-CE': '#7f7f7f', 'Adaptive Leaf-Gamma-HLSI': '#c7c7c7',
         'Surrogate-Adaptive Leaf-CE': '#9edae5',
         'Surrogate-Adaptive Leaf-Gamma-HLSI': '#17becf',
         'Blended': '#ff7f0e'}
    MK = {'Tweedie': 'o', 'TSI': 's', 'HLSI': '^', 'Surrogate-HLSI': '>',
          'Leaf-HLSI': 'P', 'CE-HLSI': 'D', 'Leaf-CE-HLSI': 'X',
          'Gamma-HLSI': '*', 'Lambda-HLSI': '*',
          'Leaf-Gamma-HLSI': 'p', 'Leaf-Lambda-HLSI': 'p',
          'Surrogate-Gamma-HLSI': '1',
          'Surrogate-Leaf-CE-HLSI': '<', 'Surrogate-Leaf-Gamma-HLSI': '2',
          'Adaptive Leaf-CE': 'd', 'Adaptive Leaf-Gamma-HLSI': 'h',
          'Surrogate-Adaptive Leaf-CE': '8',
          'Surrogate-Adaptive Leaf-Gamma-HLSI': '3',
          'Blended': 'v'}

    fig, axes = plt.subplots(3, 3, figsize=(22, 14))
    axes = axes.ravel()

    metric_specs = [
        ('mmd', 'MMD', 'MMD vs Singularity (double-X)', True),
        ('kl', 'Histogram KL', 'KL(P_true || P_model)', True),
        ('reverse_kl', 'Reverse KL', 'Approx. KL(P_model || P_true)', True),
        ('target_nll', 'Target NLL', r'$-E_{x\sim P_{model}}[\log p_{true}(x)]$', True),
        ('ksd', 'KSD', 'KSD vs Singularity (double-X)', True),
        ('score_rmse', 'Score RMSE', r'$E_{t,x_t}\|s^*(x_t,t)-\hat s(x_t,t)\|$', True),
        ('sinkhorn_w2', 'Sinkhorn W2', 'Sinkhorn W2 vs Singularity', True),
        ('ms', 'Max |score|', 'Score Magnitude Stability', True),
        ('nan', 'NaN %', 'Invalid sample fraction', False),
    ]

    for ax, (key, ylabel, title, logy) in zip(axes, metric_specs):
        for m in methods:
            vals = [data[e][m][key] for e in eps_list]
            if logy:
                vals = [min(v, 1e2) if np.isfinite(v) else 1e2 for v in vals]
            ax.plot(eps_list, vals, '-' + MK.get(m, 'o'), label=m, color=C.get(m, None), ms=8, lw=2)
        ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        ax.set(xlabel=r'$\epsilon$', ylabel=ylabel, title=title)
        ax.invert_xaxis()
        ax.grid(alpha=.3)

    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig('outputs/summary_double_x_metrics.png', dpi=150)
    plt.close(fig)

    # --- Heatmap grid in place of scatter grid ---
    show = [e for e in eps_list if e in [1.00, .5,  0.2 ,.08, .025, .01, .003]]
    nr = len(show)
    nc = len(methods) + 1
    fig2 = plt.figure(figsize=(3.5 * (nc + 0.25), 3.2 * nr))
    gs2 = fig2.add_gridspec(
        nr,
        nc + 1,
        width_ratios=[1.0] * nc + [0.045],
        wspace=0.18,
        hspace=0.12,
    )
    axes2 = np.empty((nr, nc), dtype=object)
    cbar_axes = []
    for r in range(nr):
        for c in range(nc):
            axes2[r, c] = fig2.add_subplot(gs2[r, c])
        cbar_axes.append(fig2.add_subplot(gs2[r, nc]))

    for r, eps in enumerate(show):
        gt_np = true[eps].detach().cpu().numpy()
        H_true, xedges, yedges, lo, hi = make_gt_hist2d(gt_np, bins=170)
        vmax = max(float(H_true.max()), 1e-12)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        for c, nm in enumerate(['True'] + methods):
            ax = axes2[r, c]
            if nm == 'True':
                H = H_true
            else:
                s = data[eps][nm]['samples']
                if len(s) >= 10:
                    H = hist2d_on_gt_grid(s.detach().cpu().numpy(), xedges, yedges)
                else:
                    H = np.zeros_like(H_true)

            im = ax.imshow(
                H,
                origin='lower',
                extent=extent,
                aspect='equal',
                interpolation='nearest',
                vmin=0.0,
                vmax=vmax,
            )

            if r == 0:
                ax.set_title(nm, fontsize=10)
            if c == 0:
                ax.set_ylabel(fr'$\epsilon$={eps:.0e}', fontsize=11)

            if nm != 'True':
                mv = data[eps][nm]['mmd']
                klv = data[eps][nm]['kl']
                rklv = data[eps][nm]['reverse_kl']
                sw2v = data[eps][nm]['sinkhorn_w2']
                ksdv = data[eps][nm]['ksd']
                rmsev = data[eps][nm]['score_rmse']
                ax.text(
                    0.03, 0.97,
                    f'MMD={mv:.3f}\nKL={klv:.3f}\nrKL={rklv:.3f}\nsW2={sw2v:.3f}\nKSD={ksdv:.3f}\nRMSE={rmsev:.3f}',
                    transform=ax.transAxes,
                    fontsize=6.5,
                    va='top',
                    bbox=dict(boxstyle='round', fc='w', alpha=0.75)
                )

            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.tick_params(labelsize=6)

        # dedicated colorbar axis at the far right of each row
        cax = cbar_axes[r]
        cbar = fig2.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label('density', fontsize=8)

    fig2.suptitle('2D Sample Heatmaps for the double-X target (range and normalization from ground-truth samples)', fontsize=13, y=0.995)
    fig2.subplots_adjust(left=0.05, right=0.985, top=0.94, bottom=0.05)
    fig2.savefig('outputs/heatmap_double_x.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # --- One-diagonal marginals ---
    R45 = torch.tensor([[np.cos(np.pi / 4), -np.sin(np.pi / 4)],
                        [np.sin(np.pi / 4),  np.cos(np.pi / 4)]], dtype=torch.get_default_dtype())
    stiff_dir = R45[:, 1].detach().cpu().numpy()

    fig3, axes3 = plt.subplots(len(show), 1, figsize=(10, 3 * len(show)))
    if len(show) == 1:
        axes3 = [axes3]
    for r, eps in enumerate(show):
        ax = axes3[r]
        xt = true[eps].detach().cpu().numpy()
        proj_true = xt @ stiff_dir
        yr = max(abs(proj_true).max() * 1.5, 0.1)
        bins = np.linspace(-yr, yr, 80)
        ax.hist(proj_true[:5000], bins=bins, density=True, alpha=.3, color='gray', label='True')
        for m in methods:
            s = data[eps][m]['samples'].detach().cpu().numpy()
            if len(s) > 10:
                proj = s @ stiff_dir
                ax.hist(proj, bins=bins, density=True, alpha=.4,
                        color=C.get(m, None), label=m, histtype='step', lw=2)
        ax.set_xlabel('Projection onto one diagonal arm', fontsize=11)
        ax.set_ylabel('Density')
        ax.set_title(fr'One-diagonal marginal, $\epsilon$={eps:.0e}')
        ax.legend(fontsize=8, ncol=3)
    fig3.tight_layout()
    fig3.savefig('outputs/diag_marginal_double_x.png', dpi=150)
    plt.close(fig3)

    # --- Tables ---
    print("\n" + "=" * 110)
    print("MMD  (double-X target, lower=better)")
    print("=" * 110)
    hdr = f"{'eps':>10}" + "".join(f"{m:>14}" for m in methods)
    print(hdr)
    print("-" * len(hdr))
    for e in eps_list:
        row = f"{e:10.1e}"
        for m in methods:
            v = data[e][m]['mmd']
            row += f"{v:14.6f}" if np.isfinite(v) else f"{'DIVERGED':>14}"
        print(row)

    print("\nHistogram KL(P_true || P_model)")
    print("=" * 110)
    print(hdr)
    print("-" * len(hdr))
    for e in eps_list:
        row = f"{e:10.1e}"
        for m in methods:
            v = data[e][m]['kl']
            row += f"{v:14.6f}" if np.isfinite(v) else f"{'DIVERGED':>14}"
        print(row)

    print("\nApprox Reverse KL(P_model || P_true)")
    print("=" * 110)
    print(hdr)
    print("-" * len(hdr))
    for e in eps_list:
        row = f"{e:10.1e}"
        for m in methods:
            v = data[e][m]['reverse_kl']
            row += f"{v:14.6f}" if np.isfinite(v) else f"{'DIVERGED':>14}"
        print(row)

    print("\nTarget NLL  -E_q[log p_true(x)]")
    print("=" * 110)
    print(hdr)
    print("-" * len(hdr))
    for e in eps_list:
        row = f"{e:10.1e}"
        for m in methods:
            v = data[e][m]['target_nll']
            row += f"{v:14.6f}" if np.isfinite(v) else f"{'DIVERGED':>14}"
        print(row)

    print("\nKSD")
    print("=" * 110)
    print(hdr)
    print("-" * len(hdr))
    for e in eps_list:
        row = f"{e:10.1e}"
        for m in methods:
            v = data[e][m]['ksd']
            row += f"{v:14.6f}" if np.isfinite(v) else f"{'DIVERGED':>14}"
        print(row)

    print("\nScore RMSE")
    print("=" * 110)
    print(hdr)
    print("-" * len(hdr))
    for e in eps_list:
        row = f"{e:10.1e}"
        for m in methods:
            v = data[e][m]['score_rmse']
            row += f"{v:14.6f}" if np.isfinite(v) else f"{'DIVERGED':>14}"
        print(row)

    print("\nMax |score|")
    print("=" * 110)
    print(hdr)
    print("-" * len(hdr))
    for e in eps_list:
        row = f"{e:10.1e}"
        for m in methods:
            row += f"{data[e][m]['ms']:14.1f}"
        print(row)
    print("=" * 110)
    display_saved_figure('outputs/heatmap_double_x.png', width=1400)

if __name__ == '__main__':
    print("Double-X singular GMM experiment (crossed ±45° components)")
    data, true, el, meth = run()
    plot(data, true, el, meth)
    print("\nDone.")
