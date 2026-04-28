import os
import math
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.func import vmap, hessian as fhessian

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

P_LEAF   = None   # None → adaptive p_leaf; scalar → fixed
D_TARGET = 39     # 13 particles × 3D


# ==============================================================
# LJ-13 target  (iDEM benchmark convention)
# ==============================================================
class LJ13Target:
    """
    13-particle Lennard-Jones system in 3D.
    U(x) = 4ε Σ_{i<j} [(σ/r_ij)^12 - (σ/r_ij)^6],  ε=σ=1.

    Three key differences from the first version that caused MALA failure:
      1. r_min_clip removed — only a 1e-8 numerical stabiliser, matching iDEM.
      2. _icosahedron_init: chains start at the known global minimum
         (energy ≈ -44.3ε), not a diffuse partial grid.
      3. sample_mala defaults: step_size=3e-5, burnin=100_000, thin=200.
         Exact MGF calculation: h=3e-5 → E[ΔU]=0.4 → ~67% acceptance.
         h=0.005 and h=0.001 both give ~0% acceptance (E[ΔU]≫1).
    """
    N_PARTICLES = 13
    D_PARTICLE  = 3
    D           = 39
    _TRI = torch.triu(torch.ones(13, 13, dtype=torch.bool), diagonal=1)

    def __init__(self, eps=1.0, sigma=1.0):
        self.eps  = float(eps)
        self.sig2 = float(sigma) ** 2

    # ------------------------------------------------------------------ #
    #  log p  [B, 39] → [B]                                               #
    # ------------------------------------------------------------------ #
    def log_prob(self, x):
        r     = x.reshape(-1, 13, 3)
        diffs = r.unsqueeze(2) - r.unsqueeze(1)           # [B,13,13,3]
        d2    = (diffs * diffs).sum(-1).clamp(min=1e-8)   # [B,13,13]
        d2i   = self.sig2 / d2
        # FAB-paper convention: U = Σ_{i<j} [(σ/r)^12 - 2(σ/r)^6]
        # Minimum per pair at r=σ, U_min=-1ε  (vs standard 4ε LJ: min at 2^{1/6}σ)
        lj    = d2i**6 - 2.0 * d2i**3                    # [B,13,13]
        mask  = self._TRI.to(x.device)
        return -lj[:, mask].sum(-1)                        # [B]

    # ------------------------------------------------------------------ #
    #  Score  [B, 39] → [B, 39]                                           #
    # ------------------------------------------------------------------ #
    def score(self, x):
        x_ = x.detach().requires_grad_(True)
        with torch.enable_grad():
            self.log_prob(x_).sum().backward()
        return x_.grad.detach().clone()

    # ------------------------------------------------------------------ #
    #  Hessian (autodiff)  [B, 39] → [B, 39, 39]                         #
    # ------------------------------------------------------------------ #
    def hessian(self, x):
        def neg_lp(xi):
            return -self.log_prob(xi.unsqueeze(0)).squeeze(0)
        return vmap(fhessian(neg_lp))(x.detach())

    # ------------------------------------------------------------------ #
    #  Hessian (FD, much faster for d=39)                                 #
    # ------------------------------------------------------------------ #
    def hessian_fd(self, x, eps=1e-4):
        B, d = x.shape
        H = torch.zeros(B, d, d, dtype=x.dtype, device=x.device)
        for j in range(d):
            e = torch.zeros(d, dtype=x.dtype, device=x.device); e[j] = eps
            H[:, :, j] = -(self.score(x + e) - self.score(x - e)) / (2.0 * eps)
        return 0.5 * (H + H.transpose(-1, -2))

    # ------------------------------------------------------------------ #
    #  Icosahedral init  (LJ-13 global minimum, energy ≈ −44.3ε)          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _icosahedron_init(n_chains, noise=0.05):
        """
        12 icosahedron vertices + 1 central particle.
        Scaled so nearest-neighbour distance = 2^{1/6} ≈ 1.122σ (LJ minimum).
        """
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        v = torch.tensor([
            [ 0.,  1.,  phi], [ 0., -1.,  phi],
            [ 0.,  1., -phi], [ 0., -1., -phi],
            [ 1.,  phi,  0.], [-1.,  phi,  0.],
            [ 1., -phi,  0.], [-1., -phi,  0.],
            [ phi,  0.,  1.], [-phi,  0.,  1.],
            [ phi,  0., -1.], [-phi,  0., -1.],
        ], dtype=torch.float64)
        pos = torch.cat([v, torch.zeros(1, 3, dtype=torch.float64)], dim=0)  # [13,3]
        pos = pos * (1.0 / 2.0)   # edge 2 → 1  (FAB equilibrium: nn surface-surface = 1)
        pos = pos - pos.mean(0)
        x0  = pos.reshape(1, 39).expand(n_chains, -1).contiguous()
        return x0 + noise * torch.randn(n_chains, 39, dtype=torch.float64)

    # ------------------------------------------------------------------ #
    #  MALA  (iDEM convention)                                             #
    # ------------------------------------------------------------------ #
    def sample_mala(self, n, step_size=1e-3, n_chains=32,
                    burnin=10_000, thin=50, verbose=True):
        """
        MALA with icosahedron init.

        Measured acceptance sweep at the icosahedral minimum:
          h=5e-3 →  0%   (noise-induced energy blow-up, stiff modes)
          h=1e-3 → 64%   ← default (good range, drift term compensates)
          h=3e-4 → 93%
          h=1e-4 → 98%   (essentially ULA)

        Note: theory predicts ~0% for h=0.005 even without singularity hits.
        The transition is sharp around h≈0.002; above that the stiff-mode
        (λ≈750) energy fluctuations dominate the MH correction.

        Correlation times at h=1e-3 (~64% acceptance):
          λ≈750 stiff mode: τ ≈ 3 steps   (well thermalized)
          λ≈1   soft mode:  τ ≈ 1500 steps (OK after burnin=10k)
          λ≈0.1 very soft:  τ ≈ 15k steps  (marginally at burnin=10k)
        """
        x = self._icosahedron_init(n_chains)

        def lp_score(xin):
            xin_ = xin.detach().requires_grad_(True)
            with torch.enable_grad():
                lp = self.log_prob(xin_)
                lp.sum().backward()
            return lp.detach(), xin_.grad.detach().clone()

        lp_x, sx = lp_score(x)
        accept_sum = total = 0
        samples = []
        needed_post = math.ceil(n / n_chains) * thin + 1

        if verbose:
            print(f"  MALA: {n_chains} chains, burnin={burnin:,}, "
                  f"thin={thin}, step_size={step_size}")

        for step in range(burnin + needed_post):
            noise  = torch.randn_like(x)
            x_prop = x + step_size * sx + math.sqrt(2.0 * step_size) * noise

            lp_prop, sx_prop = lp_score(x_prop)

            log_q_fwd = -((x_prop - x      - step_size * sx     )**2).sum(-1) / (4*step_size)
            log_q_bwd = -((x      - x_prop - step_size * sx_prop)**2).sum(-1) / (4*step_size)
            log_alpha = (lp_prop - lp_x + log_q_bwd - log_q_fwd).clamp(max=0.0)

            acc  = torch.rand(n_chains) < log_alpha.exp()
            mask = acc.unsqueeze(-1)
            x    = torch.where(mask, x_prop,  x)
            lp_x = torch.where(acc,  lp_prop, lp_x)
            sx   = torch.where(mask, sx_prop, sx)

            if step >= burnin:
                accept_sum += acc.sum().item(); total += n_chains
                if (step - burnin) % thin == 0:
                    samples.append(x.clone())
                    if len(samples) * n_chains >= n:
                        break

        if verbose and total > 0:
            print(f"  MALA acceptance (post-burnin): {100*accept_sum/total:.1f}%")

        out = torch.cat(samples, dim=0)
        return out[torch.randperm(len(out))[:n]]

    # ------------------------------------------------------------------ #
    #  ULA  (fast biased sampler for large reference pools)                #
    # ------------------------------------------------------------------ #
    def sample_ula(self, n, step_size=0.05, n_chains=64,
                   burnin=5_000, thin=25, verbose=False):
        x = self._icosahedron_init(n_chains, noise=0.1)
        samples = []
        needed  = burnin + math.ceil(n / n_chains) * thin + 1
        if verbose:
            print(f"  ULA: {n_chains} chains, burnin={burnin:,}, thin={thin}")
        for step in range(needed):
            sx = self.score(x)
            x  = x + step_size * sx + math.sqrt(2*step_size) * torch.randn_like(x)
            if step >= burnin and (step - burnin) % thin == 0:
                samples.append(x.clone())
                if len(samples) * n_chains >= n:
                    break
        out = torch.cat(samples, dim=0)
        return out[torch.randperm(len(out))[:n]]

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def particles(x):
        return x.reshape(-1, 13, 3)

    @staticmethod
    def center_of_mass(x):
        return x.reshape(-1, 13, 3).mean(dim=1)

    @staticmethod
    def radial_from_com(x):
        r   = x.reshape(-1, 13, 3)
        com = r.mean(dim=1, keepdim=True)
        return ((r - com)**2).sum(-1).sqrt().reshape(-1)

    @staticmethod
    def pairwise_dists(x):
        r = x.reshape(-1, 13, 3)
        return torch.cat([((r[:, i] - r[:, j])**2).sum(-1).sqrt()
                          for i in range(13) for j in range(i+1, 13)], dim=0)

    @staticmethod
    def com_centered_coords(x):
        """CoM-subtracted positions.  [B,39] → [13B,3].  Used for 2D heatmaps."""
        r   = x.reshape(-1, 13, 3)
        com = r.mean(dim=1, keepdim=True)
        return (r - com).reshape(-1, 3)


# ==============================================================
# OU helpers
# ==============================================================
def at(t): return torch.exp(-t)
def vt(t): return 1.0 - torch.exp(-2.0 * t)


# ==============================================================
# SNIS weights
# ==============================================================
def snis_w(y, t, xr):
    a, v = at(t), vt(t)
    lw = -0.5 * ((y.unsqueeze(1) - a * xr.unsqueeze(0))**2).sum(-1) / v
    lw = lw - lw.max(1, keepdim=True).values
    w  = lw.exp()
    return w / w.sum(1, keepdim=True).clamp(min=1e-30)


def surrogate_transition_w(y, t, xr, precomp, leaf=False,
                           lmin=1e-4, lmax=1e6):
    """
    Exact transition responsibilities for the local Gaussian/leaf surrogate
    components used by the corresponding HLSI estimator.

    This is the controlled OU-vs-surrogate-weight ablation: only the transition
    weights are changed.  The downstream score estimator is still the existing
    HLSI / Leaf-CE-HLSI routine.

    Ordinary HLSI: trusted positive Hessian directions use clean precision
    lambda; untrusted directions fall back to the Tweedie point-mass transition
    precision 1/v_t.

    Leaf modes: trusted directions use the ordinary Hessian precision, negative
    directions use rescued_eig, and the remaining directions fall back to the
    Tweedie point-mass transition.
    """
    a = at(t); a2 = a**2; v = vt(t)
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
    disp = y.unsqueeze(1) - a * mu.unsqueeze(0)
    disp_eig = torch.einsum('mji,nmj->nmi', V, disp)
    qform = (sig_inv_eig.unsqueeze(0) * disp_eig.square()).sum(-1)
    logdet = torch.log(sig_inv_eig.clamp(min=1e-30)).sum(-1)
    lw = 0.5 * logdet.unsqueeze(0) - 0.5 * qform
    lw = lw - lw.max(1, keepdim=True).values
    w = lw.exp()
    return w / w.sum(1, keepdim=True).clamp(min=1e-30)


# ==============================================================
# Score estimators  (d-general)
# ==============================================================
def est_tweedie(y, t, xr, w):
    return -(w.unsqueeze(2) * (y.unsqueeze(1) - at(t) * xr.unsqueeze(0))).sum(1) / vt(t)

def est_tsi(y, t, xr, w, target, s0_ref=None):
    s0 = target.score(xr) if s0_ref is None else s0_ref
    return (w.unsqueeze(2) * s0.unsqueeze(0)).sum(1) / at(t)

def est_hlsi(y, t, xr, w, target, lmin=1e-4, lmax=1e6, precomp=None):
    a = at(t); a2 = a**2; v = vt(t)
    s0, lam, V = (precomp['s0'], precomp['lam'], precomp['V']) if precomp else \
                 (target.score(xr), *torch.linalg.eigh(target.hessian(xr))[::1])
    if precomp is None: lam, V = torch.linalg.eigh(target.hessian(xr))

    ok  = (lam > lmin) & (lam <= lmax)
    lg  = torch.where(ok, lam, torch.zeros_like(lam))
    s0e = torch.einsum('mji,mj->mi', V, s0)
    mu  = xr + torch.einsum('mij,mj->mi', V,
            torch.where(ok, s0e / lg.clamp(1e-30), torch.zeros_like(s0e)))
    si  = torch.where(ok, lg / (a2 + v*lg.clamp(1e-30)), torch.full_like(lam, 1/v))
    de  = torch.einsum('mji,nmj->nmi', V, y.unsqueeze(1) - a*mu.unsqueeze(0))
    return (w.unsqueeze(2) * (-torch.einsum('mij,nmj->nmi', V, si.unsqueeze(0)*de))).sum(1)

def est_ce_hlsi(y, t, xr, w, target, lmin=1e-4, lmax=1e6, precomp=None):
    """
    CE-HLSI with corrected zero-mode handling.

    The bug: clamping lam to [lmin,lmax] before computing the gate left zero
    modes (translational/rotational invariance eigenvalues ≈ 0) with gate ≈ 1
    (fully TSI).  Since TSI ≈ 0 at any LJ minimum, the reverse-SDE drift in
    those 6 directions became +y (unstable), producing samples ~28 units away.

    Fix: average the UNCLAMPED per-sample Hessians (H_raw), decompose the
    result, then gate only trusted eigenvalues [lmin, lmax].  Zero modes
    (lam ≈ 0) and non-PSD modes (lam < 0) fall back to Tweedie — exactly
    mirroring how est_hlsi treats non-trusted directions.
    """
    a = at(t); a2 = a**2; v = vt(t)
    s_twd = est_tweedie(y, t, xr, w)
    if precomp is None:
        s_tsi = est_tsi(y, t, xr, w, target)
        lam_r, V_r = torch.linalg.eigh(target.hessian(xr))
        H_ref = torch.einsum('mij,mj,mkj->mik', V_r, lam_r, V_r)   # UNCLAMPED
    else:
        s_tsi = est_tsi(y, t, xr, w, target, s0_ref=precomp['s0'])
        H_ref = precomp['H_raw']                                      # UNCLAMPED

    Hb = (w.unsqueeze(-1).unsqueeze(-1) * H_ref.unsqueeze(0)).sum(1)
    Hb = 0.5 * (Hb + Hb.transpose(-1,-2))
    lam, V = torch.linalg.eigh(Hb)

    # Zero modes (|lam| < lmin) and non-PSD (lam < 0): gate = 0 → Tweedie
    # Trusted range: CE gate.  Too-stiff (lam > lmax): gate = 0 → Tweedie.
    ok   = (lam >= lmin) & (lam <= lmax)
    gate = torch.where(ok, a2 / (a2 + v * lam + 1e-30), torch.zeros_like(lam))

    def gated(g, s):
        se = torch.einsum('nij,nj->ni', V.transpose(-1,-2), s)
        return torch.einsum('nij,nj->ni', V, g * se)
    return gated(1.0 - gate, s_twd) + gated(gate, s_tsi)

def est_blended(y, t, xr, w, target, s0_ref=None):
    a, v = at(t), vt(t)
    s0  = target.score(xr) if s0_ref is None else s0_ref
    tsi = s0.unsqueeze(0) / a
    twd = -(y.unsqueeze(1) - a*xr.unsqueeze(0)) / v
    am  = (w.unsqueeze(2)*tsi).sum(1); bm = (w.unsqueeze(2)*twd).sum(1)
    ac  = tsi - am.unsqueeze(1);       bc = twd - bm.unsqueeze(1)
    va  = (w.unsqueeze(2)*ac**2).sum(1).clamp(1e-30)
    vb  = (w.unsqueeze(2)*bc**2).sum(1).clamp(1e-30)
    cab = (w.unsqueeze(2)*ac*bc).sum(1)
    g   = ((va - cab) / (va + vb - 2*cab).clamp(1e-20)).clamp(0,1)
    return (1-g)*am + g*bm


# ==============================================================
# Adaptive within-leaf precision
# ==============================================================
def compute_adaptive_p_leaf(target, xr, eigvals, eigvecs, lmin=1e-4,
                             delta_vals=(0.05,0.1,0.2), p_min=0.1, p_max=100.0):
    is_bad = eigvals < 0
    p = torch.where(is_bad, torch.full_like(eigvals, p_min), eigvals)
    bi, bk = is_bad.nonzero(as_tuple=True)
    if len(bi) == 0: return p
    bv = eigvecs[bi,:,bk]; bl = eigvals[bi,bk]; bx = xr[bi]
    Da = torch.zeros(len(bi), dtype=xr.dtype, device=xr.device)
    for delta in delta_vals:
        s = target.score(torch.cat([bx + delta*bv, bx - delta*bv]))
        sp, sm = s[:len(bi)], s[len(bi):]
        Da += ((sp*bv).sum(-1) - (sm*bv).sum(-1) + 2*bl*delta) / (2*delta**3 + 1e-30)
    p_h = (bl + torch.sqrt((-3*Da/len(delta_vals)).clamp(0))).clamp(p_min, p_max)
    p[bi, bk] = p_h
    return p


def precompute_leaf_hlsi(target, xr, lmin=1e-4, lmax=1e6, p_leaf=P_LEAF,
                         use_fd_hessian=True, fd_eps=1e-4,
                         adaptive_delta_vals=(0.05,0.1,0.2),
                         adaptive_p_min=0.1, adaptive_p_max=100.0):
    H  = target.hessian_fd(xr, eps=fd_eps) if use_fd_hessian else target.hessian(xr)
    H  = 0.5*(H + H.transpose(1,2))
    s0 = target.score(xr)
    ev, V = torch.linalg.eigh(H)
    trusted    = (ev >= lmin) & (ev <= lmax)
    is_non_psd = ev < 0
    te = torch.where(trusted, ev, torch.zeros_like(ev))

    if p_leaf is None:
        pp = compute_adaptive_p_leaf(target, xr, ev, V, lmin,
                                     adaptive_delta_vals, adaptive_p_min, adaptive_p_max)
        re = torch.where(trusted, ev, torch.where(is_non_psd, pp, torch.zeros_like(ev)))
    else:
        pp = torch.full_like(ev, float(p_leaf))
        re = torch.where(trusted, ev, torch.where(is_non_psd, pp, torch.zeros_like(ev)))

    s0e = torch.einsum('mji,mj->mi', V, s0)
    de  = torch.where(trusted, s0e / te.clamp(lmin), torch.zeros_like(s0e))
    mu  = xr + torch.einsum('mij,mj->mi', V, de)
    return dict(s0=s0, lam=ev, V=V, trusted_eig=te, rescued_eig=re, mu=mu,
                P_leaf=torch.einsum('mij,mj,mkj->mik', V, re, V),
                H_ce=torch.einsum('mij,mj,mkj->mik', V, ev.clamp(lmin,lmax), V),
                # H_raw: unclamped per-sample Hessian (includes true zero-mode
                # eigenvalues ≈ 0 from translational/rotational invariance).
                # Used by est_ce_hlsi so the SNIS-averaged Hbar retains genuine
                # zero modes, enabling ok-gating to send them to Tweedie fallback.
                H_raw=torch.einsum('mij,mj,mkj->mik', V, ev, V),
                trusted=trusted, is_non_psd=is_non_psd)


def est_leaf_hlsi(y, t, xr, w, target=None, precomp=None,
                  lmin=1e-4, lmax=1e6, p_leaf=P_LEAF):
    if precomp is None:
        precomp = precompute_leaf_hlsi(target, xr, lmin, lmax, p_leaf)
    a = at(t); a2 = a**2; v = vt(t)
    V, mu = precomp['V'], precomp['mu']
    te, re = precomp['trusted_eig'], precomp['rescued_eig']
    tr, np_ = precomp['trusted'], precomp['is_non_psd']
    si = torch.where(tr, te/(a2+v*te.clamp(1e-30)),
          torch.where(np_, re/(a2+v*re.clamp(1e-30)), torch.full_like(te,1/v)))
    de = torch.einsum('mji,nmj->nmi', V, y.unsqueeze(1) - a*mu.unsqueeze(0))
    return (w.unsqueeze(2) * (-torch.einsum('mij,nmj->nmi', V, si.unsqueeze(0)*de))).sum(1)


def est_leaf_ce_hlsi(y, t, xr, w, target=None, precomp=None,
                     lmin=1e-4, lmax=1e6, p_leaf=P_LEAF):
    """
    Leaf-CE-HLSI with corrected zero-mode handling.

    P_leaf for zero modes (not trusted, not non-PSD) has rescued_eig = 0,
    so Pbar eigenvalues in those directions are exactly 0.  Without ok-gating,
    gate = a²/(a²+v*0+1e-30) ≈ 1 (TSI) → score ≈ 0 → unstable reverse SDE.
    Fix: gate = 0 for lam < lmin (Tweedie fallback), same as HLSI convention.
    """
    if precomp is None:
        precomp = precompute_leaf_hlsi(target, xr, lmin, lmax, p_leaf)
    a = at(t); a2 = a**2; v = vt(t)
    s_twd = est_tweedie(y, t, xr, w)
    s_tsi = est_tsi(y, t, xr, w, target, s0_ref=precomp['s0'])
    Pb = (w.unsqueeze(-1).unsqueeze(-1) * precomp['P_leaf'].unsqueeze(0)).sum(1)
    Pb = 0.5*(Pb + Pb.transpose(1,2))
    lam, V = torch.linalg.eigh(Pb)
    # ok: zero modes (lam≈0, rescued_eig=0) and large modes → gate=0 → Tweedie
    ok   = (lam >= lmin) & (lam <= lmax)
    gate = torch.where(ok, a2 / (a2 + v * lam + 1e-30), torch.zeros_like(lam))
    de = torch.einsum('nji,nj->ni', V, s_tsi - s_twd)
    return s_twd + torch.einsum('nij,nj->ni', V, gate * de)


# ==============================================================
# Heun reverse SDE
# ==============================================================
def heun_sde(score_fn, n, d, n_steps=200, t_max=3.0, t_min=0.015):
    ts = torch.linspace(t_max, t_min, n_steps+1, dtype=torch.get_default_dtype())
    y  = torch.randn(n, d, dtype=torch.get_default_dtype())
    ms = 0.0; fail = False
    for i in range(n_steps):
        tc, tn = ts[i], ts[i+1]; h = tc - tn
        s1 = score_fn(y, tc); ms = max(ms, s1.abs().max().item())
        if not torch.isfinite(s1).all(): fail = True; break
        d1 = y + 2*s1; noise = (2*h).sqrt() * torch.randn_like(y); yh = y + h*d1 + noise
        s2 = score_fn(yh, tn)
        if not torch.isfinite(s2).all(): fail = True; break
        y = y + 0.5*h*(d1 + yh + 2*s2) + noise
        if not torch.isfinite(y).all(): fail = True; break
    if not fail:
        tf = torch.tensor(t_min, dtype=torch.get_default_dtype())
        sf = score_fn(y, tf); ms = max(ms, sf.abs().max().item())
        if torch.isfinite(sf).all(): y = (y + vt(tf)*sf) / at(tf)
        else: fail = True
    return y, ms, fail


# ==============================================================
# Metrics
# ==============================================================
def mmd(X, Y, bws=(0.5,1.0,2.0,5.0,10.0)):
    n = min(len(X),2000); m = min(len(Y),2000); X=X[:n]; Y=Y[:m]
    xx=torch.cdist(X,X)**2; yy=torch.cdist(Y,Y)**2; xy=torch.cdist(X,Y)**2
    gs=[0.5/b**2 for b in bws]
    return (sum((-g*xx).exp().mean()+(-g*yy).exp().mean()-2*(-g*xy).exp().mean()
                for g in gs)/len(gs)).item()

def kl_energy_histogram(X, Y, target, bins=120, sm=1e-12):
    with torch.no_grad():
        ux=(-target.log_prob(X)).cpu().numpy(); uy=(-target.log_prob(Y)).cpu().numpy()
    lo,hi=float(np.percentile(ux,.5)),float(np.percentile(ux,99.5))
    e=np.linspace(lo,hi,bins+1)
    px,_=np.histogram(ux,bins=e); py,_=np.histogram(uy,bins=e)
    px=px+sm; py=py+sm; px/=px.sum(); py/=py.sum()
    return float(np.sum(px*(np.log(px)-np.log(py))))

def kl_pairwise_dist_histogram(X, Y, bins=120, sm=1e-12, n_particles=13, d_particle=3):
    def pd(s):
        r=s.reshape(-1,n_particles,d_particle)
        return torch.cat([((r[:,i]-r[:,j])**2).sum(-1).sqrt()
                          for i in range(n_particles) for j in range(i+1,n_particles)]).cpu().numpy()
    with torch.no_grad(): dx=pd(X); dy=pd(Y)
    lo,hi=0.,float(np.percentile(dx,99.5)); e=np.linspace(lo,hi,bins+1)
    px,_=np.histogram(dx,bins=e); py,_=np.histogram(dy,bins=e)
    px=px+sm; py=py+sm; px/=px.sum(); py/=py.sum()
    return float(np.sum(px*(np.log(px)-np.log(py))))

def ksd_rbf(samples, score_fn, bandwidth='median'):
    n=min(len(samples),1000); X=samples[:n]
    if n<5: return float('inf')
    with torch.no_grad():
        S=score_fn(X); d=X.shape[1]; dm=torch.cdist(X,X)
        med=torch.median(dm[dm>0]).item() if (dm>0).any() else 1.
        h=float(bandwidth) if bandwidth!='median' else max(med,1e-12)
        d2=dm**2; K=torch.exp(-d2/(2*h**2))
        diff=X.unsqueeze(1)-X.unsqueeze(0)
        U=K*(S@S.T) - (K.unsqueeze(-1)*(S.unsqueeze(1)-S.unsqueeze(0))*diff).sum(-1)/h**2 \
          + K*(d/h**2 - d2/h**4)
        return float(torch.sqrt(U.mean().clamp(0)).item())


def w2_distance_old(X, Y, max_n=400):
    """
    2-Wasserstein distance W2(X, Y) via linear assignment on a subsample.

    Matches the iDEM paper metric (Table 2 / §F.2): Euclidean ground distance,
    Hungarian algorithm (scipy) on equal-size subsamples.  Lower is better.
    Paper DW-4 reference: iDEM 2.13±0.04, FAB 2.15±0.02.
    """
    from scipy.optimize import linear_sum_assignment
    n   = min(len(X), len(Y), max_n)
    Xn  = X[torch.randperm(len(X))[:n]].cpu().numpy().astype(np.float32)
    Yn  = Y[torch.randperm(len(Y))[:n]].cpu().numpy().astype(np.float32)
    diff = Xn[:, None, :] - Yn[None, :, :]   # [n, n, d]
    C    = (diff ** 2).sum(-1)                # [n, n]
    row_ind, col_ind = linear_sum_assignment(C)
    return float(np.sqrt(C[row_ind, col_ind].mean()))


def w2_distance(X, Y):
    """
    2-Wasserstein distance W2(X, Y) matching the DEM repo implementation.

    Uses exact OT with uniform marginals via POT's emd2 on the full
    squared Euclidean cost matrix, then returns the square root.
    """
    import math
    import ot as pot

    if X.dim() > 2:
        X = X.reshape(X.shape[0], -1)
    if Y.dim() > 2:
        Y = Y.reshape(Y.shape[0], -1)

    a = pot.unif(X.shape[0])
    b = pot.unif(Y.shape[0])

    M = torch.cdist(X, Y) ** 2
    ret = pot.emd2(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    return float(math.sqrt(ret))



def ess_kde(samples, target, n_max=800):
    """
    Normalised Effective Sample Size using KDE as the proposal density.

      ESS = (Σ wi)² / (n · Σ wi²)  ∈ [0, 1]
      wi ∝ exp(−E(xi)) / q_KDE(xi)

    The iDEM paper (§F.2) uses the trained model density for q.  Since our
    score estimators have no explicit density we fit a Gaussian KDE on the
    generated samples as a proxy.  Higher is better.
    Paper LJ-13 reference: iDEM 0.231±0.005, FAB 0.101±0.059, pDEM 0.044±0.013.

    Note: KDE in d=39 is rough; treat values as relative comparisons,
    not direct matches to the OT-CFM-density ESS in the paper.
    """
    from sklearn.neighbors import KernelDensity
    n     = min(len(samples), n_max)
    sc    = samples[torch.randperm(len(samples))[:n]]
    sc_np = sc.cpu().double().numpy()
    d     = sc_np.shape[1]
    bw    = max(n ** (-1.0 / (d + 4)), 0.01)   # Silverman's rule
    kde   = KernelDensity(kernel='gaussian', bandwidth=bw).fit(sc_np)
    log_q = kde.score_samples(sc_np)
    with torch.no_grad():
        log_p = target.log_prob(sc).cpu().double().numpy()
    log_w  = log_p - log_q
    log_w -= log_w.max()
    w      = np.exp(log_w)
    w     /= w.sum()
    return float(1.0 / (n * float((w ** 2).sum())))


def nll_kde(samples, test_points, n_fit=3000):
    """
    Approximate NLL  −mean_{x~p_true} log q_KDE(x),  lower is better.

    q_KDE fit on generated samples, evaluated on held-out GT test points.
    The iDEM paper trains an OT-CFM flow for NLL (§F.2); KDE is a tractable
    proxy (rougher in d=39 than d=8, but gives valid relative rankings).
    Paper LJ-13 reference: iDEM 17.68±0.14, FAB 17.52±0.17, pDEM 18.80±0.48.
    """
    from sklearn.neighbors import KernelDensity
    sc_np = samples.cpu().double().numpy()
    te_np = test_points.cpu().double().numpy()
    n, d  = len(sc_np), sc_np.shape[1]
    if n_fit is not None and n_fit < n:
        idx   = np.random.choice(n, n_fit, replace=False)
        sc_np = sc_np[idx]
        n     = len(sc_np)
    bw  = max(n ** (-1.0 / (d + 4)), 0.01)
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(sc_np)
    return float(-kde.score_samples(te_np).mean())


def copying_score(samples, xr_train, xt_test, n_gen=800, k_pool=None,
                  alert_ratio=0.8, method_name=''):
    n_gen=min(n_gen,len(samples))
    k_pool=min(k_pool or min(len(xr_train),len(xt_test)), len(xr_train),len(xt_test))
    gen=samples[torch.randperm(len(samples))[:n_gen]]
    ref=xr_train[torch.randperm(len(xr_train))[:k_pool]]
    tst=xt_test [torch.randperm(len(xt_test)) [:k_pool]]
    def md(A,B,ch=200):
        return torch.cat([torch.cdist(A[i:i+ch].float(),B.float()).min(1).values
                          for i in range(0,len(A),ch)])
    with torch.no_grad(): dtr=md(gen,ref); dte=md(gen,tst)
    mtr=float(dtr.mean()); mte=float(dte.mean())
    ratio=mtr/(mte+1e-30); copying=ratio<alert_ratio
    if copying:
        print(f"  COPYING ALERT: {method_name}  dTr={mtr:.4f} dTe={mte:.4f} ratio={ratio:.3f}")
    return dict(mean_d_train=mtr,mean_d_test=mte,ratio=ratio,copying=copying,
                d_train_dist=dtr.cpu().numpy(),d_test_dist=dte.cpu().numpy())

def score_rmse_snis_ref(score_hat_fn, target, xr_ref_large,
                        n_eval=2000, t_min=0.015, t_max=3.0, n_time_grid=120):
    with torch.no_grad():
        x0=xr_ref_large[torch.randperm(len(xr_ref_large))[:n_eval]]
        tg=torch.exp(torch.linspace(math.log(t_min),math.log(t_max),n_time_grid,
                                    dtype=torch.get_default_dtype()))
        idx=torch.randint(0,n_time_grid,(n_eval,)); t=tg[idx]
        a=at(t).unsqueeze(-1); v=vt(t).unsqueeze(-1)
        xt=a*x0 + torch.sqrt(v)*torch.randn_like(x0)
        st=torch.empty_like(xt); sh=torch.empty_like(xt)
        for j in idx.unique(sorted=True):
            mask=(idx==j); tj=tg[j]
            st[mask]=est_tweedie(xt[mask],tj,xr_ref_large,snis_w(xt[mask],tj,xr_ref_large))
            sh[mask]=score_hat_fn(xt[mask],tj)
        return float(torch.sqrt(((st-sh)**2).sum(1).mean()).item())




def _rand_subset(x, n):
    n = min(int(n), len(x))
    return x[torch.randperm(len(x))[:n]].clone()


def _apply_sym_eig(V, eigvals, vec):
    proj = torch.einsum('bij,bj->bi', V.transpose(-1, -2), vec)
    return torch.einsum('bij,bj->bi', V, eigvals * proj)


def _gaussian_logpdf_eigh(x, mean, V, cov_eigvals):
    diff = x - mean
    proj = torch.einsum('bij,bj->bi', V.transpose(-1, -2), diff)
    maha = (proj.pow(2) / cov_eigvals.clamp_min(1e-30)).sum(-1)
    logdet = torch.log(cov_eigvals.clamp_min(1e-30)).sum(-1)
    d = x.shape[1]
    return -0.5 * (d * math.log(2.0 * math.pi) + logdet + maha)


def _metric_bundle(samples, xt, xr_train, target, method_name=''):
    if len(samples) < 20:
        return dict(
            nll=float('inf'), ess=float('inf'), w2=float('inf'),
            mmd=float('inf'), kl_energy=float('inf'), kl_dist=float('inf'),
            ksd=float('inf'),
            copy=dict(mean_d_train=float('nan'), mean_d_test=float('nan'),
                      ratio=float('nan'), copying=False,
                      d_train_dist=np.array([]), d_test_dist=np.array([]))
        )
    with torch.no_grad():
        mv   = mmd(samples, xt)
        klev = kl_energy_histogram(samples, xt, target)
        kldv = kl_pairwise_dist_histogram(samples, xt, n_particles=13, d_particle=3)
        ksdv = ksd_rbf(samples, target.score)
        copy = copying_score(samples, xr_train, xt,
                             n_gen=min(len(samples), 800),
                             k_pool=min(len(xr_train), len(xt)),
                             alert_ratio=0.50,
                             method_name=method_name)
    return dict(
        nll=nll_kde(samples, xt, n_fit=3000),
        ess=ess_kde(samples, target, n_max=800),
        w2=w2_distance(samples, xt),
        mmd=mv, kl_energy=klev, kl_dist=kldv, ksd=ksdv,
        copy=copy,
    )


def sample_mala_exact_from_init(target, x_init, step_size=1e-3, n_steps=4, verbose=False):
    x = x_init.clone()

    def lp_score(xin):
        xin_ = xin.detach().requires_grad_(True)
        with torch.enable_grad():
            lp = target.log_prob(xin_)
            lp.sum().backward()
        return lp.detach(), xin_.grad.detach().clone()

    lp_x, sx = lp_score(x)
    accept_sum = total = 0

    if verbose:
        print(f"    Exact MALA: {len(x)} chains, steps={n_steps}, step_size={step_size}")

    for _ in range(int(n_steps)):
        noise = torch.randn_like(x)
        mean_fwd = x + step_size * sx
        x_prop = mean_fwd + math.sqrt(2.0 * step_size) * noise

        lp_prop, sx_prop = lp_score(x_prop)
        mean_bwd = x_prop + step_size * sx_prop

        log_q_fwd = -((x_prop - mean_fwd) ** 2).sum(-1) / (4.0 * step_size)
        log_q_bwd = -((x - mean_bwd) ** 2).sum(-1) / (4.0 * step_size)
        log_alpha = (lp_prop - lp_x + log_q_bwd - log_q_fwd).clamp(max=0.0)

        acc = torch.rand(len(x), dtype=x.dtype, device=x.device) < log_alpha.exp()
        mask = acc.unsqueeze(-1)
        x = torch.where(mask, x_prop, x)
        lp_x = torch.where(acc, lp_prop, lp_x)
        sx = torch.where(mask, sx_prop, sx)

        accept_sum += acc.sum().item()
        total += len(x)

    return x, dict(accept_rate=accept_sum / max(total, 1), n_steps=int(n_steps))


def sample_hessian_precond_mala_exact_from_init(
    target,
    x_init,
    step_size=0.005,
    n_steps=4,
    use_fd_hessian=True,
    fd_eps=1e-4,
    hess_lmin=1e-4,
    hess_lmax=1e4,
    fallback_mobility=0.02,
    mob_min=5e-4,
    mob_max=0.05,
    verbose=False,
):
    x = x_init.clone()

    def proposal_stats(xin):
        xin_ = xin.detach().requires_grad_(True)
        with torch.enable_grad():
            lp = target.log_prob(xin_)
            lp.sum().backward()
        score = xin_.grad.detach().clone()
        with torch.no_grad():
            H = target.hessian_fd(xin.detach(), eps=fd_eps) if use_fd_hessian else target.hessian(xin.detach())
            H = 0.5 * (H + H.transpose(-1, -2))
            lam, V = torch.linalg.eigh(H)
            prec_eff = torch.where(
                lam > hess_lmin,
                lam.clamp(max=hess_lmax),
                torch.full_like(lam, 1.0 / fallback_mobility),
            )
            mob = (1.0 / prec_eff).clamp(min=mob_min, max=mob_max)
            drift = _apply_sym_eig(V, mob, score)
            mean = xin.detach() + step_size * drift
            cov_eig = (2.0 * step_size * mob).clamp_min(1e-30)
        return dict(lp=lp.detach(), score=score, H=H, lam=lam, V=V,
                    mob=mob, mean=mean, cov_eig=cov_eig)

    stats_x = proposal_stats(x)
    accept_sum = total = 0

    if verbose:
        print(f"    Hessian-precond exact MALA: {len(x)} chains, steps={n_steps}, step_size={step_size}, fd_hessian={use_fd_hessian}")

    for _ in range(int(n_steps)):
        noise = torch.randn_like(x)
        prop_noise = torch.einsum('bij,bj->bi', stats_x['V'], torch.sqrt(stats_x['mob']) * noise)
        x_prop = stats_x['mean'] + math.sqrt(2.0 * step_size) * prop_noise

        stats_prop = proposal_stats(x_prop)
        log_q_fwd = _gaussian_logpdf_eigh(x_prop, stats_x['mean'], stats_x['V'], stats_x['cov_eig'])
        log_q_bwd = _gaussian_logpdf_eigh(x, stats_prop['mean'], stats_prop['V'], stats_prop['cov_eig'])
        log_alpha = (stats_prop['lp'] - stats_x['lp'] + log_q_bwd - log_q_fwd).clamp(max=0.0)

        acc = torch.rand(len(x), dtype=x.dtype, device=x.device) < log_alpha.exp()
        mask = acc.unsqueeze(-1)
        x = torch.where(mask, x_prop, x)
        for key in list(stats_x.keys()):
            if torch.is_tensor(stats_x[key]):
                tensor_mask = acc.reshape(-1, *([1] * (stats_x[key].ndim - 1)))
                stats_x[key] = torch.where(tensor_mask, stats_prop[key], stats_x[key])
            else:
                stats_x[key] = stats_prop[key]

        accept_sum += acc.sum().item()
        total += len(x)

    return x, dict(accept_rate=accept_sum / max(total, 1), n_steps=int(n_steps))


def run_hessian_mcmc_sanity_check(
    target,
    train_pool,
    test_pool,
    n_init=64,
    n_test=512,
    plain_step_size=1e-3,
    plain_steps=4,
    hess_step_size=0.005,
    hess_steps=4,
    use_fd_hessian=True,
    fd_eps=1e-4,
):
    x_init = _rand_subset(train_pool, n_init)
    xt_eval = _rand_subset(test_pool, n_test)

    print("\n[1b] Hessian exact-MCMC sanity check …")
    print(f"    Using {len(x_init)} train initialisations and {len(xt_eval)} held-out test samples")

    t0 = time.time()
    baseline = _metric_bundle(x_init, xt_eval, train_pool, target, method_name='Train baseline')
    plain_samples, plain_info = sample_mala_exact_from_init(
        target, x_init, step_size=plain_step_size, n_steps=plain_steps, verbose=False)
    plain_metrics = _metric_bundle(plain_samples, xt_eval, train_pool, target,
                                   method_name='Plain MALA from train init')
    hess_samples, hess_info = sample_hessian_precond_mala_exact_from_init(
        target, x_init, step_size=hess_step_size, n_steps=hess_steps,
        use_fd_hessian=use_fd_hessian, fd_eps=fd_eps, verbose=False)
    hess_metrics = _metric_bundle(hess_samples, xt_eval, train_pool, target,
                                  method_name='Hessian-precond MALA from train init')

    sanity = {
        'Train baseline': dict(metrics=baseline, info=dict(accept_rate=float('nan'), n_steps=0)),
        'Plain MALA from train init': dict(metrics=plain_metrics, info=plain_info),
        'Hessian-precond MALA from train init': dict(metrics=hess_metrics, info=hess_info),
    }

    header = (f"{'Reference':<34} {'Acc%':>7} {'NLL':>10} {'ESS':>10} {'W2':>9} {'MMD':>9} "
              f"{'KL-E':>9} {'KL-D':>9} {'KSD':>9} {'Ratio':>8}")
    print(header)
    print('-' * len(header))
    lines = [header, '-' * len(header)]
    for name in ['Train baseline', 'Plain MALA from train init', 'Hessian-precond MALA from train init']:
        met = sanity[name]['metrics']
        info = sanity[name]['info']
        acc = info['accept_rate']
        acc_s = f"{100*acc:6.1f}" if np.isfinite(acc) else '   N/A '
        line = (f"{name:<34} {acc_s:>7} {met['nll']:10.5f} {met['ess']:10.5f} {met['w2']:9.5f} "
                f"{met['mmd']:9.5f} {met['kl_energy']:9.5f} {met['kl_dist']:9.5f} {met['ksd']:9.5f} "
                f"{met['copy']['ratio']:8.3f}")
        print(line)
        lines.append(line)

    base = sanity['Train baseline']['metrics']
    plain = sanity['Plain MALA from train init']['metrics']
    hess = sanity['Hessian-precond MALA from train init']['metrics']
    issues = []
    for mk in ['nll', 'w2', 'mmd', 'kl_energy', 'kl_dist', 'ksd']:
        b, p, h = base[mk], plain[mk], hess[mk]
        if np.isfinite(h):
            thresh = max(2.0 * b if np.isfinite(b) else -np.inf,
                         1.5 * p if np.isfinite(p) else -np.inf)
            if h > thresh and h > min(b, p) + 1e-8:
                issues.append(mk)
    if np.isfinite(hess['ess']):
        b_ess, p_ess, h_ess = base['ess'], plain['ess'], hess['ess']
        if h_ess < min(0.5 * b_ess if np.isfinite(b_ess) else np.inf,
                       0.67 * p_ess if np.isfinite(p_ess) else np.inf):
            issues.append('ess')

    if issues:
        msg = f"    Potential Hessian red flag on: {', '.join(issues)}"
    else:
        msg = "    No obvious Hessian-specific metric blow-up relative to train/test or plain MALA baselines."
    print(msg)
    lines.append(msg)

    out_path = 'outputs/lj13_hessian_mcmc_sanity.txt'
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"    Saved {out_path}  ({time.time()-t0:.1f}s)")
    return sanity


# ==============================================================
# Run
# ==============================================================
def run(train_npy=None, test_npy=None):
    """
    train_npy / test_npy : paths to pre-computed ground-truth .npy files
        (shape [N, 39], float32).  If provided, MALA is skipped entirely.
        Expected: FAB-convention LJ-13 samples, CoM-centred per sample.
        E.g. 'train_split_LJ13-1000.npy' (100k) and 'test_split_LJ13-1000.npy' (10k).
    """
    NR=500; NS=800; NT=2000; NR_LARGE=1200
    N_STEPS=200; T_MAX,T_MIN=2.5,0.01; N_TIME_GRID=200
    D=LJ13Target.D

    methods=['Surrogate-Adaptive Leaf-CE','Surrogate-Leaf-CE-HLSI','Surrogate-HLSI',
             'Adaptive Leaf-CE','Adaptive Leaf','Leaf-CE-HLSI','CE-HLSI',
             'Leaf-HLSI','HLSI','Blended','Tweedie','TSI']

    os.makedirs('outputs',exist_ok=True)
    target=LJ13Target(eps=1.0,sigma=1.0)

    print("="*80)
    print("LJ-13 benchmark  (13 particles × R³,  d=39)")
    print("MALA: icosahedron init, step_size=3e-5  (targeting ~67% acceptance)")
    print("="*80)

    print("\n[1] Ground-truth samples …")
    t0=time.time()
    if train_npy is not None and test_npy is not None:
        all_train = torch.from_numpy(np.load(train_npy)).double()
        all_test  = torch.from_numpy(np.load(test_npy)).double()
        xr       = all_train[torch.randperm(len(all_train))[:NR]]
        xr_large = all_train[torch.randperm(len(all_train))[:NR_LARGE]]
        xt       = all_test [torch.randperm(len(all_test)) [:NT]]
        print(f"    Loaded {len(all_train):,} train / {len(all_test):,} test samples "
              f"({time.time()-t0:.1f}s)")
        E_xt = (-target.log_prob(xt[:500])).cpu()
        print(f"    Energy (test, FAB): min={E_xt.min():.2f}  mean={E_xt.mean():.2f}  "
              f"max={E_xt.max():.2f}")
    else:
        print("    No data files provided — running MALA (slow, may need tuning).")
        all_gt=target.sample_mala(NT+NR+NR_LARGE, step_size=1e-3, n_chains=32,
                                   burnin=10_000, thin=50, verbose=True)
        E_all = (-target.log_prob(all_gt)).cpu()
        print(f"    Done ({time.time()-t0:.0f}s),  {len(all_gt)} samples")
        print(f"    Energy: min={E_all.min():.2f}  mean={E_all.mean():.2f}  max={E_all.max():.2f}")
        xr=all_gt[:NR]; xt=all_gt[NR:NR+NT]; xr_large=all_gt[NR:NR+NR_LARGE]

    diag_train_pool = all_train if train_npy is not None and test_npy is not None else xr
    diag_test_pool  = all_test  if train_npy is not None and test_npy is not None else xt

    _ = run_hessian_mcmc_sanity_check(
        target,
        diag_train_pool,
        diag_test_pool,
        n_init=min(64, len(diag_train_pool), len(diag_test_pool)),
        n_test=min(512, len(diag_test_pool)),
        plain_step_size=1e-3,
        plain_steps=4,
        hess_step_size=0.005,
        hess_steps=4,
        use_fd_hessian=True,
        fd_eps=1e-4,
    )

    print("\n[2] Precomputing Hessian spectral data (FD, eps=1e-4) …")
    t0=time.time()
    precomp=precompute_leaf_hlsi(target,xr,lmin=1e-4,lmax=1e6,
                                  p_leaf=P_LEAF,use_fd_hessian=True,fd_eps=1e-4)
    print(f"    Done ({time.time()-t0:.1f}s)")
    ln=precomp['lam'].cpu().numpy()
    print(f"    Hessian: min={ln.min():.3e}  max={ln.max():.3e}  "
          f"frac_non_psd={100*(ln<0).mean():.1f}%")

    print("\n[2b] Adaptive p_leaf diagnostic …")
    t0=time.time()
    precomp_adaptive=precompute_leaf_hlsi(
        target,xr,lmin=1e-4,lmax=1e6,p_leaf=None,use_fd_hessian=True,fd_eps=1e-4,
        adaptive_delta_vals=(0.05,0.1,0.2),adaptive_p_min=0.1,adaptive_p_max=100.0)
    pn=precomp_adaptive['rescued_eig'][precomp_adaptive['is_non_psd']].cpu().numpy()
    if len(pn)>0:
        print(f"    Done ({time.time()-t0:.1f}s)  adaptive p: "
              f"min={pn.min():.3f} median={float(pn.mean()):.3f} max={pn.max():.3f} "
              f"(n_bad={len(pn)})")
    else:
        print(f"    Done ({time.time()-t0:.1f}s)  no non-PSD directions")

    def make_fn(meth):
        def fn(y,t):
            t=t if isinstance(t,torch.Tensor) else torch.tensor(t,dtype=torch.get_default_dtype())
            w=snis_w(y,t,xr)
            if   meth=='Tweedie':         return est_tweedie(y,t,xr,w)
            elif meth=='TSI':             return est_tsi(y,t,xr,w,target,s0_ref=precomp['s0'])
            elif meth=='HLSI':            return est_hlsi(y,t,xr,w,target,precomp=precomp)
            elif meth=='Surrogate-HLSI':
                w_surr = surrogate_transition_w(y,t,xr,precomp,leaf=False)
                return est_hlsi(y,t,xr,w_surr,target,precomp=precomp)
            elif meth=='CE-HLSI':         return est_ce_hlsi(y,t,xr,w,target,precomp=precomp)
            elif meth=='Leaf-HLSI':       return est_leaf_hlsi(y,t,xr,w,target,precomp=precomp)
            elif meth=='Leaf-CE-HLSI':    return est_leaf_ce_hlsi(y,t,xr,w,target,precomp=precomp)
            elif meth=='Surrogate-Leaf-CE-HLSI':
                w_surr = surrogate_transition_w(y,t,xr,precomp,leaf=True)
                return est_leaf_ce_hlsi(y,t,xr,w_surr,target,precomp=precomp)
            elif meth=='Blended':         return est_blended(y,t,xr,w,target,s0_ref=precomp['s0'])
            elif meth=='Adaptive Leaf':   return est_leaf_hlsi(y,t,xr,w,target,precomp=precomp_adaptive)
            elif meth=='Adaptive Leaf-CE':return est_leaf_ce_hlsi(y,t,xr,w,target,precomp=precomp_adaptive)
            elif meth=='Surrogate-Adaptive Leaf-CE':
                w_surr = surrogate_transition_w(y,t,xr,precomp_adaptive,leaf=True)
                return est_leaf_ce_hlsi(y,t,xr,w_surr,target,precomp=precomp_adaptive)
            raise ValueError(meth)
        return fn

    print("\n[3] Sampling and evaluating methods …")
    print(f"{'Method':<30} {'NLL(KDE)':>10} {'ESS(KDE)':>10} {'W2':>8}"
          f" {'MMD':>9} {'KL-E':>9} {'KL-D':>9} {'KSD':>9}"
          f" {'RMSE':>9} {'dTr':>8} {'dTe':>8} {'Ratio':>7} {'|s|max':>8} {'NaN%':>5} t")
    print("-"*150)

    results={}
    for m in methods:
        fn=make_fn(m); t0=time.time()
        with torch.no_grad():
            samp,ms,fail=heun_sde(fn,NS,D,n_steps=N_STEPS,t_max=T_MAX,t_min=T_MIN)
        ok=torch.isfinite(samp).all(dim=1); sc=samp[ok]; nv=ok.sum().item()
        nan_pct=100*(1-nv/NS)

        if nv>=20:
            with torch.no_grad():
                mv   =mmd(sc,xt)
                klev =kl_energy_histogram(sc,xt,target)
                kldv =kl_pairwise_dist_histogram(sc,xt,n_particles=13,d_particle=3)
                ksdv =ksd_rbf(sc,target.score)
                rmse =score_rmse_snis_ref(fn,target,xr_large,n_eval=1200,
                                          t_min=T_MIN,t_max=T_MAX,n_time_grid=N_TIME_GRID)
                copy =copying_score(sc,xr,xt,n_gen=min(nv,800),
                                    k_pool=min(NR,NT),alert_ratio=0.50,method_name=m)
            # iDEM-paper metrics (Table 2 / §F.2)
            w2v  = w2_distance(sc, xt)
            essv = ess_kde(sc, target, n_max=800)
            nllv = nll_kde(sc, xt, n_fit=3000)
        else:
            mv=klev=kldv=ksdv=rmse=float('inf')
            w2v = essv = nllv = float('inf')
            copy=dict(mean_d_train=float('nan'),mean_d_test=float('nan'),ratio=float('nan'),
                      copying=False,d_train_dist=np.array([]),d_test_dist=np.array([]))

        dt=time.time()-t0
        results[m]=dict(samples=sc,mmd=mv,kl_energy=klev,kl_dist=kldv,
                        ksd=ksdv,score_rmse=rmse,w2=w2v,ess=essv,nll=nllv,
                        ms=ms,nan_pct=nan_pct,copy=copy)
        r_s=f"{copy['ratio']:7.3f}" if np.isfinite(copy['ratio']) else "    N/A"
        nll_s = f"{nllv:10.5f}" if np.isfinite(nllv) else "   DIVERGED"
        ess_s = f"{essv:10.5f}" if np.isfinite(essv) else "   DIVERGED"
        w2_s  = f"{w2v:8.5f}"  if np.isfinite(w2v)  else "  DIVERGED"
        print(f"{m:<30} {nll_s} {ess_s} {w2_s}"
              f" {mv:9.5f} {klev:9.5f} {kldv:9.5f} {ksdv:9.5f} {rmse:9.5f}"
              f" {copy['mean_d_train']:8.4f} {copy['mean_d_test']:8.4f} {r_s}"
              f" {ms:8.1f} {nan_pct:5.1f} {dt:3.0f}s {'FAIL' if fail else 'ok'}"
              f"{' COPY' if copy['copying'] else ''}")

    nc=copying_score(xt,xr,xt[NR:],n_gen=min(NT,800),k_pool=min(NR,NT-NR),
                     alert_ratio=0.50,method_name="GT-null")
    print(f"\n  GT-null  dTr={nc['mean_d_train']:.4f}  dTe={nc['mean_d_test']:.4f}  "
          f"ratio={nc['ratio']:.3f}")

    return results,xt,xr,methods,target


# ==============================================================
# Plotting
# ==============================================================
COLORS={
    'Tweedie':'#1f77b4','TSI':'#d62728','HLSI':'#2ca02c',
    'Surrogate-HLSI':'#98df8a',
    'Leaf-HLSI':'#17becf','CE-HLSI':'#9467bd','Leaf-CE-HLSI':'#8c564b',
    'Surrogate-Leaf-CE-HLSI':'#c49c94',
    'Blended':'#ff7f0e','Adaptive Leaf':'#006400','Adaptive Leaf-CE':'#4b0082',
    'Surrogate-Adaptive Leaf-CE':'#9edae5',
}


def _make_2d_hist(coords, edges):
    """Normalised 2D density.  coords: [N,2], edges shared for both axes."""
    H,_,_=np.histogram2d(coords[:,0],coords[:,1],bins=[edges,edges],density=False)
    H=H+1e-30
    return (H/H.sum()).T   # transpose: rows=y, cols=x


def plot_2d_heatmaps(results, xt, methods, proj=(0,1), proj_name='XY'):
    """
    2D heatmap grid showing CoM-centred particle positions projected onto two
    coordinate axes.  13 particles per sample are overlaid in a single density.

    With a proper MALA reference:
      - The icosahedral shell appears as a ring at r ≈ 1.07σ from CoM.
      - The central particle appears as a single spot at the origin.
    """
    C=COLORS; i0,i1=proj
    with torch.no_grad():
        gt_com=LJ13Target.com_centered_coords(xt).cpu().numpy()
    gp=gt_com[:,[i0,i1]]

    # Shared axis limits from ground truth
    lo=float(np.percentile(gp,0.5)); hi=float(np.percentile(gp,99.5))
    pad=(hi-lo)*0.06; lo-=pad; hi+=pad
    edges=np.linspace(lo,hi,81); ext=[edges[0],edges[-1],edges[0],edges[-1]]

    H_gt=_make_2d_hist(gp,edges); vmax=float(H_gt.max())
    n_panels=len(methods)+1
    fig,axes=plt.subplots(1,n_panels,figsize=(3.4*n_panels,3.8),squeeze=False)
    axes=axes[0]

    # Ground-truth panel
    im=axes[0].imshow(H_gt,origin='lower',extent=ext,aspect='equal',
                       interpolation='bilinear',vmin=0,vmax=vmax,cmap='viridis')
    axes[0].set_title('True (MALA)',fontsize=9,fontweight='bold')
    axes[0].set_xlabel('xyz'[i0]); axes[0].set_ylabel('xyz'[i1])

    # One panel per method
    for ci,m in enumerate(methods):
        ax=axes[ci+1]; sc=results[m]['samples']
        if len(sc)>=10:
            with torch.no_grad():
                mc=LJ13Target.com_centered_coords(sc).cpu().numpy()[:,[i0,i1]]
            H=_make_2d_hist(mc,edges)
            ax.imshow(H,origin='lower',extent=ext,aspect='equal',
                      interpolation='bilinear',vmin=0,vmax=vmax,cmap='viridis')
            mv=results[m]['mmd']; kle=results[m]['kl_energy']
            ax.text(0.03,0.97,f'MMD={mv:.3f}\nKL-E={kle:.2f}',
                    transform=ax.transAxes,fontsize=6.5,va='top',color='white',
                    bbox=dict(boxstyle='round',fc='k',alpha=0.55))
        else:
            ax.set_facecolor('#111'); ax.text(0.5,0.5,'FAILED',
                transform=ax.transAxes,ha='center',va='center',color='red',fontsize=11)
        ax.set_title(m,fontsize=8,color=C.get(m,'black')); ax.set_xlabel('xyz'[i0])

    fig.colorbar(im,ax=axes[-1],fraction=0.046,label='density (norm.)')
    lbl={'XY':'X–Y','XZ':'X–Z','YZ':'Y–Z'}.get(proj_name,proj_name)
    fig.suptitle(f'LJ-13  CoM-centred particle positions  ({lbl} projection)\n'
                 f'13 particles per sample overlaid  ·  icosahedral shell visible as ring',
                 fontsize=10)
    fig.tight_layout()
    path=f'outputs/lj13_2d_heatmaps_{proj_name.lower()}.png'
    fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig)
    print(f"    Saved {path}")


def plot(results,xt,xr,methods,target):
    C=COLORS
    print("\n[4] Generating plots …")

    # ---- 1. Metric bars ----
    fig,axes=plt.subplots(2,4,figsize=(24,10))
    axes=axes.flatten()
    for ax,mk,ml in zip(axes,
                         ['nll','ess','w2','mmd','kl_energy','ksd','score_rmse'],
                         ['NLL (KDE, ↓)','ESS (KDE, ↑)','W2 (↓)',
                          'MMD (↓)','KL energy hist (↓)','KSD (↓)','Score RMSE (↓)']):
        vals=[min(results[m][mk],50.) for m in methods]
        bars=ax.bar(range(len(methods)),vals,color=[C[m] for m in methods])
        ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods,rotation=30,ha='right',fontsize=8)
        ax.set_title(ml,fontsize=10); ax.set_ylabel(mk); ax.grid(alpha=.3,axis='y')
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),
                    f'{v:.3f}',ha='center',va='bottom',fontsize=7)
    for unused_ax in axes[7:]: unused_ax.set_visible(False)
    fig.suptitle('LJ-13 benchmark metrics',fontsize=13)
    fig.tight_layout(); fig.savefig('outputs/lj13_metrics_bar.png',dpi=150); plt.close(fig)

    # ---- 2. 2D heatmaps — three orthogonal projections ----
    # The icosahedral shell at radius ≈1.07σ appears as a ring; the central
    # particle sits at the CoM origin.  Comparing GT vs each method reveals
    # whether the generated samples concentrate on the correct cluster structure.
    for proj,name in [((0,1),'XY'),((0,2),'XZ'),((1,2),'YZ')]:
        plot_2d_heatmaps(results,xt,methods,proj=proj,proj_name=name)

    # ---- 3. Pairwise distance distribution ----
    with torch.no_grad(): pdist_gt=LJ13Target.pairwise_dists(xt).cpu().numpy()
    r_star=2**(1/6)
    bins_d=np.linspace(0,float(np.percentile(pdist_gt,99.5)),100)
    fig2,ax2=plt.subplots(figsize=(10,5))
    ax2.hist(pdist_gt,bins=bins_d,density=True,alpha=.35,color='gray',label='True (MALA)')
    for m in methods:
        sc=results[m]['samples']
        if len(sc)>=10:
            with torch.no_grad(): pd=LJ13Target.pairwise_dists(sc).cpu().numpy()
            ax2.hist(pd,bins=bins_d,density=True,alpha=.55,color=C[m],label=m,histtype='step',lw=2)
    ax2.axvline(r_star,ls='--',color='k',lw=1.2,label=f'r*=2^{{1/6}}≈{r_star:.3f}')
    ax2.set_xlabel('Inter-particle distance  r_ij',fontsize=11); ax2.set_ylabel('Density')
    ax2.set_title('LJ-13  pairwise distance distribution  (78 pairs pooled)',fontsize=11)
    ax2.legend(fontsize=8,ncol=2); ax2.grid(alpha=.3)
    fig2.tight_layout(); fig2.savefig('outputs/lj13_pairwise_dist.png',dpi=150); plt.close(fig2)

    # ---- 4. Radial from CoM ----
    with torch.no_grad(): rad_gt=LJ13Target.radial_from_com(xt).cpu().numpy()
    bins_r=np.linspace(0,float(np.percentile(rad_gt,99.5)),80)
    fig3,ax3=plt.subplots(figsize=(9,4))
    ax3.hist(rad_gt,bins=bins_r,density=True,alpha=.35,color='gray',label='True (MALA)')
    for m in methods:
        sc=results[m]['samples']
        if len(sc)>=10:
            with torch.no_grad(): rr=LJ13Target.radial_from_com(sc).cpu().numpy()
            ax3.hist(rr,bins=bins_r,density=True,alpha=.55,color=C[m],label=m,histtype='step',lw=2)
    ax3.set_xlabel('‖x_i − CoM‖',fontsize=11); ax3.set_ylabel('Density')
    ax3.set_title('LJ-13  per-particle radial distance from CoM')
    ax3.legend(fontsize=8,ncol=2); ax3.grid(alpha=.3)
    fig3.tight_layout(); fig3.savefig('outputs/lj13_radial_from_com.png',dpi=150); plt.close(fig3)

    # ---- 5. Energy distribution ----
    with torch.no_grad(): E_gt=(-target.log_prob(xt)).cpu().numpy()
    bins_e=np.linspace(float(np.percentile(E_gt,.5)),float(np.percentile(E_gt,99.5)),80)
    fig4,ax4=plt.subplots(figsize=(9,4))
    ax4.hist(E_gt,bins=bins_e,density=True,alpha=.35,color='gray',label='True (MALA)')
    for m in methods:
        sc=results[m]['samples']
        if len(sc)>=10:
            with torch.no_grad(): Ec=(-target.log_prob(sc)).cpu().numpy()
            ax4.hist(Ec,bins=bins_e,density=True,alpha=.55,color=C[m],label=m,histtype='step',lw=2)
    ax4.set_xlabel('Energy  U(x) = -log p(x)',fontsize=11); ax4.set_ylabel('Density')
    ax4.set_title('LJ-13  energy distribution')
    ax4.legend(fontsize=8,ncol=2); ax4.grid(alpha=.3)
    fig4.tight_layout(); fig4.savefig('outputs/lj13_energy_dist.png',dpi=150); plt.close(fig4)

    # ---- 6. Copying detector ----
    fig5,axes5=plt.subplots(1,len(methods),figsize=(3.2*len(methods),4.2))
    for ax,m in zip(axes5,methods):
        cd=results[m]['copy']; dtr=cd['d_train_dist']; dte=cd['d_test_dist']
        if len(dtr)==0: ax.set_title(m+'\n(no samples)',fontsize=8); continue
        hi=float(np.percentile(np.concatenate([dtr,dte]),98))
        edges=np.linspace(0,hi,60)
        ax.hist(dtr,bins=edges,density=True,alpha=.6,color='#d62728',label='→ train')
        ax.hist(dte,bins=edges,density=True,alpha=.6,color='#1f77b4',label='→ test')
        ax.axvline(cd['mean_d_train'],color='#d62728',lw=1.8,ls='--')
        ax.axvline(cd['mean_d_test'], color='#1f77b4',lw=1.8,ls='--')
        ratio=cd['ratio']; alert='\nCOPYING' if cd['copying'] else ''
        ax.set_title(f"{m}\nratio={ratio:.3f}{alert}",fontsize=8,
                     color='#d62728' if cd['copying'] else 'black',
                     fontweight='bold' if cd['copying'] else 'normal')
        ax.set_xlabel('min NN dist',fontsize=8); ax.grid(alpha=.3)
        if ax is axes5[0]: ax.legend(fontsize=7)
    fig5.suptitle('Copying: min-NN dist to train (red) vs test (blue)',fontsize=10)
    fig5.tight_layout(); fig5.savefig('outputs/lj13_copying.png',dpi=150,bbox_inches='tight'); plt.close(fig5)

    # ---- 7. Copying ratio bar ----
    ratios=[results[m]['copy']['ratio'] for m in methods]
    fig6,ax6=plt.subplots(figsize=(9,4))
    ax6.bar(range(len(methods)),ratios,color=['#d62728' if results[m]['copy']['copying'] else C[m] for m in methods],edgecolor='k',lw=.7)
    ax6.axhline(1.,color='k',ls='--',lw=1.2,label='ratio=1'); ax6.axhline(.8,color='orange',ls=':',lw=1.5,label='alert=0.80')
    ax6.set_xticks(range(len(methods))); ax6.set_xticklabels(methods,rotation=25,ha='right',fontsize=9)
    ax6.set_ylabel('d_train / d_test'); ax6.set_title('Copying score  (red = ALERT)')
    ax6.set_ylim(0,max(max(ratios)*1.15,1.2)); ax6.legend(fontsize=9); ax6.grid(alpha=.3,axis='y')
    for i,v in enumerate(ratios): ax6.text(i,v+.01,f'{v:.3f}',ha='center',va='bottom',fontsize=8)
    fig6.tight_layout(); fig6.savefig('outputs/lj13_copying_ratio.png',dpi=150); plt.close(fig6)

    # ---- Summary table ----
    print("\n"+"="*90)
    # ---- iDEM paper comparison (Table 2, LJ-13) ----
    print('\n' + '='*90)
    print('iDEM paper reference  LJ-13 (d=39)  — Table 2, mean±std over 3 seeds')
    print(f"  {'Method':<10} {'NLL':>15} {'ESS':>15} {'W2':>15}")
    for mname, vals in [
        ('FAB',   ('17.52±0.17',  '0.101±0.059', '4.35±0.01')),
        ('PIS',   ('47.05±12.46', '0.004±0.002', '4.67±0.11')),
        ('DDS',   ('DIVERGED',    'DIVERGED',     'DIVERGED')),
        ('pDEM',  ('18.80±0.48',  '0.044±0.013', '4.21±0.06')),
        ('iDEM',  ('17.68±0.14',  '0.231±0.005', '4.26±0.03')),
    ]:
        nll_p, ess_p, w2_p = vals
        print(f"  {mname:<10} {nll_p:>15} {ess_p:>15} {w2_p:>15}")
    print('\nOur methods (NLL/ESS via KDE proxy; W2 via Hungarian assignment):')
    print(f"  {'Method':<16} {'NLL(KDE)':>12} {'ESS(KDE)':>12} {'W2':>10}")
    for m in methods:
        nv=results[m]['nll']; ev=results[m]['ess']; wv=results[m]['w2']
        nll_s=f"{nv:12.5f}" if np.isfinite(nv) else '     DIVERGED'
        ess_s=f"{ev:12.5f}" if np.isfinite(ev) else '     DIVERGED'
        w2_s =f"{wv:10.5f}" if np.isfinite(wv) else '   DIVERGED'
        print(f"  {m:<16} {nll_s} {ess_s} {w2_s}")
    print('='*90)

    for metric,label in [('nll','NLL (KDE proxy, ↓)  iDEM paper metric'),
                          ('ess','ESS (KDE proxy, ↑)  iDEM paper metric'),
                          ('w2', 'W2 (↓)              iDEM paper metric'),
                          ('mmd','MMD'),('kl_energy','KL energy'),('kl_dist','KL pair-dist'),
                          ('ksd','KSD'),('score_rmse','Score RMSE')]:
        print(f"\n{label}"); print("-"*60)
        for m in methods:
            v=results[m][metric]
            print(f"  {m:<16}  {v:12.6f}" if np.isfinite(v) else f"  {m:<16}     DIVERGED")
    print("\nCopying (ratio<0.50→ALERT)"); print("-"*60)
    for m in methods:
        cd=results[m]['copy']
        print(f"  {m:<16}  dTr={cd['mean_d_train']:7.4f}  dTe={cd['mean_d_test']:7.4f}  "
              f"ratio={cd['ratio']:6.3f}{'  <-- COPYING' if cd['copying'] else ''}")
    print("\n"+"="*90)


# ==============================================================
# Main
# ==============================================================
if __name__=='__main__':
    import os
    # In Jupyter, sys.argv contains kernel flags ('-f kernel.json') — don't use it.
    # Instead check whether the standard-named files exist next to the notebook.
    _train = 'train_split_LJ13-1000.npy'
    _test  = 'test_split_LJ13-1000.npy'
    train_p = _train if os.path.exists(_train) else None
    test_p  = _test  if os.path.exists(_test)  else None
    print("LJ-13 HLSI benchmark")
    results,xt,xr,methods,target=run(train_npy=train_p, test_npy=test_p)
    plot(results,xt,xr,methods,target)
    print("\nDone — figures saved to outputs/")
