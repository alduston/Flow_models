#!/usr/bin/env python3
"""Two-stage coarse-to-fine DIRECT-LATENT CSEM cotraining, Fashion-MNIST.

Direct conditional hierarchical coordinates (the same stochastic codes are
modeled by the score heads and consumed by the decoder):
  z1: 4 channels x 4 x 4 = 64 coordinates, coarse root;
  z2: 3 channels x 8 x 8 = 192 coordinates, fine child.
  q(z1|x) and q(z2|x,z1) are diagonal Gaussians.

Child inference fuses bottom-up image features with top-down CLEAN root features.
Both the conditional CSEM teacher and the child score head see the same clean
root sample z1; the score head never receives image-conditioned posterior means,
variances, or x as inputs. Those moments are used solely for analytic targets.
The decoder consumes z2 DIRECTLY, with no decoder-only b2(z1)+d2(z1)*u2
transformation. Root/child interactions therefore occur in the encoder's
conditional q(z2|x,z1), both CSEM-prior factors, and deterministic decoder skips,
not through a new unregularized child coordinate transformation.

Default denoising reconstruction uses D(y1_tau,y2_tau;tau), with BOTH sampled
latents independently OU-corrupted at one shared time tau (= the sampled CSEM
training t in the default recipe). ALL decoder root paths are computed from
y1_tau rather than cached clean z1, preventing a clean-root bypass. The child
posterior and conditional score target still use CLEAN z1. --recon-root clean
is retained as an explicit, non-default ablation only. At generation, completed
sequential reverse solves return (z1,z2), decoded at --decode-time (~zero).

Other model sizes, losses, optimizers, conditional score-head architecture,
per-coordinate aggregation, B_tmin penalty, evaluation protocols, and gradient
routing are left as in the supplied FMNIST script. The flat reference remains
unchanged. This is an ARCHITECTURE BREAK: old residual cascade checkpoints
cannot resume, even if their tensor sizes otherwise match.

--csem-rep-weight controls ONLY CSEM gradients into the encoder/VAE, including
child->root paths. The score-head coefficient and optimizer are unchanged.
Diagnostic CSVs include posterior-vs-generated-child, ablation and gradient
checks. --self-test exercises both flat and hierarchical computation offline.

Usage:
  python train_hier_csem_fmnist.py --self-test
  python train_hier_csem_fmnist.py --mode hier --epochs 700 --output runs/hier_direct_s42
  python train_hier_csem_fmnist.py --mode flat --epochs 700 --output runs/flat_s42
  python train_hier_csem_fmnist.py --mode hier --tmin-band-weight 0 --output runs/hier_direct_noB_s42

Requires a local fmnist_reference.py next to this file (no edits required).
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import save_image

try:
    import fmnist_reference as ref
except ImportError as e:
    raise RuntimeError("Place fmnist_reference.py (shipped in the zip) beside this script.") from e


ARCHITECTURE_VERSION = 'coarse_fine_direct_z2_v2'


class CascadeResBlock(nn.Module):
    """Normalized residual feature processing; never normalizes latent means."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(math.gcd(in_channels, 16), in_channels), nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(math.gcd(out_channels, 16), out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.skip = (nn.Identity() if in_channels == out_channels else
                     nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x):
        return self.skip(x) + self.net(x)


class HierarchicalFMNISTVAE(nn.Module):
    """Two spatial stages with conditional Gaussian direct child code z2.

    q(z1|x)=N(mu1,diag(v1)); q(z2|x,z1)=N(mu2,diag(v2)).
    Both conditional CSEM and the image decoder consume the SAME child z2.
    Decoder expands z1 at 4x4, injects z2 at 8x8, then performs 8->16->32.
    Image features enter posterior inference only. No image skip enters decode.
    """
    def __init__(self, z1_channels=4, z2_channels=3, **vae_kwargs):
        super().__init__()
        self.z1_channels, self.z2_channels = int(z1_channels), int(z2_channels)
        self.latent_shapes = ((self.z1_channels, 4, 4), (self.z2_channels, 8, 8))
        backbone = ref.VAE(latent_channels=4, **vae_kwargs)
        width = 4 * int(vae_kwargs['base_ch'])
        # Retain actual reference feature stages, not unused posterior projections.
        self.enc_conv_in = backbone.enc_conv_in
        self.enc_blocks = backbone.enc_blocks
        self.enc_pre_proj = backbone.enc_pre_proj
        self.enc_coarse = nn.Sequential(
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            CascadeResBlock(width, width),
            ref.VAEAttentionBlock(width, zero_init=False),
            CascadeResBlock(width, width))
        self.root_posterior = nn.Sequential(
            nn.GroupNorm(16, width), nn.SiLU(),
            nn.Conv2d(width, 2*self.z1_channels, 3, padding=1))
        self.root_in = nn.Conv2d(self.z1_channels, width, 3, padding=1)
        self.root_blocks = nn.Sequential(
            CascadeResBlock(width, width),
            ref.VAEAttentionBlock(width, zero_init=False),
            CascadeResBlock(width, width))
        self.root_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(width, width, 3, padding=1),
            CascadeResBlock(width, width))
        # Fuse bottom-up image evidence and top-down ancestral predictions.
        self.condition = nn.Sequential(
            CascadeResBlock(2*width, width), CascadeResBlock(width, width),
            nn.GroupNorm(16, width), nn.SiLU(),
            nn.Conv2d(width, 2*self.z2_channels, 3, padding=1))
        nn.init.normal_(self.condition[-1].weight, std=1e-3)
        nn.init.zeros_(self.condition[-1].bias)
        self.child_in = nn.Conv2d(self.z2_channels, width, 3, padding=1)
        self.child_fusion = CascadeResBlock(width, width)
        self.dec_blocks, self.dec_out = backbone.dec_blocks, backbone.dec_out
        self.use_tanh_out = backbone.use_tanh_out
        self.time_cond_decoder = backbone.time_cond_decoder
        if self.time_cond_decoder:
            self.dec_time_emb_dim = backbone.dec_time_emb_dim
            self.dec_time_mlp = backbone.dec_time_mlp
            self.dec_film_layers = backbone.dec_film_layers

    def encode_root(self, x):
        h = self.enc_conv_in(x)
        for block in self.enc_blocks:
            h = block(h)
        if self.enc_pre_proj is not None:
            h = self.enc_pre_proj(h)
        mu1, lv1 = self.root_posterior(self.enc_coarse(h)).chunk(2, 1)
        return mu1, lv1.clamp(-30, 20), h

    def ancestor_features(self, z1):
        return self.root_up(self.root_blocks(self.root_in(z1)))

    def encode_conditional(self, image_features, z1, top_down=None):
        if top_down is None:
            top_down = self.ancestor_features(z1)
        mu2, lv2 = self.condition(torch.cat([image_features, top_down], 1)).chunk(2, 1)
        return mu2, lv2.clamp(-30, 20)

    def encode_sample(self, x, *, return_context=False):
        mu1, lv1, image_features = self.encode_root(x)
        z1 = mu1 + torch.exp(0.5 * lv1) * torch.randn_like(mu1)
        top_down = self.ancestor_features(z1)
        mu2, lv2 = self.encode_conditional(image_features, z1, top_down)
        z2 = mu2 + torch.exp(0.5 * lv2) * torch.randn_like(mu2)
        result = (z1, z2), ((mu1, lv1), (mu2, lv2))
        return (*result, top_down) if return_context else result

    def decode(self, z1, z2, t=None, *, top_down=None):
        # z2 is already the CSEM-modeled stochastic child coordinate. Decoder
        # receives it directly; only deterministic root-dependent feature skips
        # are permitted. During noisy reconstruction top_down MUST be None.
        h = self.ancestor_features(z1) if top_down is None else top_down
        h = self.child_fusion(h + self.child_in(z2))
        t_emb = None
        if self.time_cond_decoder and t is not None:
            t_emb = self.dec_time_mlp(ref.timestep_embedding(t, self.dec_time_emb_dim))
        for i, block in enumerate(self.dec_blocks):
            h = block(h)
            if t_emb is not None:
                scale, shift = self.dec_film_layers[i](t_emb).chunk(2, 1)
                h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        out = self.dec_out(h)
        return torch.tanh(out) if self.use_tanh_out else out


class FlatFMNISTVAE(nn.Module):
    """Exactly the original 4-channel VAE; used as within-script control."""
    def __init__(self, **vae_kwargs):
        super().__init__()
        self.vae = ref.VAE(latent_channels=4, **vae_kwargs)
        self.latent_shapes = ((4, 8, 8),)
    def encode_sample(self, x):
        mu, lv = self.vae.encode(x)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        return (z,), ((mu, lv),)
    def decode(self, *z, t=None):
        return self.vae.decode(torch.cat(z, 1), t=t)


class ConditionalDiT(ref.DiTModel):
    """Direct-child conditional DiT with full spatial root access at each layer.

    Diffuses z2 itself; context is CLEAN z1 (not its OU-corrupted version).
    Important: the score head does not see x. Classifier-free label dropout is
    inherited from the original label embedding; context is never dropped.
    """
    def __init__(self, context_channels: int, **kwargs):
        super().__init__(**kwargs)
        self.context_channels = int(context_channels)
        self.context_proj = nn.Conv2d(context_channels, self.hidden_dim, 1)
        nn.init.normal_(self.context_proj.weight, std=0.02)
        nn.init.zeros_(self.context_proj.bias)
        self.register_buffer('context_pos', self._build_2d_sincos_pos_embed(self.hidden_dim, 4),
                             persistent=False)
        self.context_attn = nn.ModuleList([
            nn.MultiheadAttention(self.hidden_dim, kwargs['num_heads'], batch_first=True)
            for _ in self.blocks])
        self.context_norm = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in self.blocks])
        self.context_gate = nn.Parameter(torch.full((len(self.blocks),), 0.1))

    def prepare_context(self, context):
        if context.ndim != 4 or context.shape[1:] != (self.context_channels, 4, 4):
            raise ValueError(f'Expected clean root [B,{self.context_channels},4,4], got {context.shape}')
        features = self.context_proj(context)
        # Root projection is computed once per child trajectory, reused for CFG.
        spatial = F.interpolate(features, size=(8, 8), mode='nearest')
        tokens = features.flatten(2).transpose(1, 2) + self.context_pos
        return spatial, tokens

    def forward(self, x, t, y=None, *, context=None, context_cache=None):
        B, C, H, W = x.shape
        spatial, context_tokens = (self.prepare_context(context) if context_cache is None
                                   else context_cache)
        if spatial.shape[0] != B:
            raise ValueError('Context batch does not match the child batch')
        tokens = self.patch_embed(x) + spatial
        Htok, Wtok = tokens.shape[2:]
        tokens = tokens.flatten(2).transpose(1, 2) + self.pos_embed
        t = t.view(-1)
        cond = self.time_mlp(ref.timestep_embedding(t, self.hidden_dim))
        if self.label_emb is not None:
            if y is None:
                y = torch.full((B,), self.null_label, dtype=torch.long, device=x.device)
            cond = cond + self.label_emb(y.to(x.device).long().view(-1))
        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens, cond)
            update = self.context_attn[i](self.context_norm[i](tokens), context_tokens,
                                          context_tokens, need_weights=False)[0]
            tokens = tokens + torch.tanh(self.context_gate[i]) * update
        gamma, beta = self.final_modulation(cond).unsqueeze(1).chunk(2, dim=-1)
        tokens = self.final_norm(tokens) * (1 + gamma) + beta
        if self.factored_head:
            raise NotImplementedError("Optimized FMNIST uses non-factored DiT heads")
        return self.unpatchify(self.final_proj(tokens), Htok, Wtok)


class BoundConditionalHead(nn.Module):
    """Adapts ConditionalDiT to the original UniversalSampler API."""
    def __init__(self, head, context):
        super().__init__()
        self.head = head
        self.context_cache = head.prepare_context(context)
    def forward(self, x, t, y=None):
        return self.head(x, t, y, context_cache=self.context_cache)


def build(args, device):
    preset = ref.DATASET_PRESETS['fmnist_reference']
    opts = dict(
        base_ch=preset['base_ch'], use_norm=False, img_size=32, img_channels=1,
        num_res_blocks=preset['num_res_blocks'], decoder_attn_half=preset['decoder_attn_half'],
        latent_proj_depth=preset['latent_proj_depth'],
        encoder_attn_half=preset['encoder_attn_half'],
        decoder_extra_block=preset['decoder_extra_block'],
        conv3x3_proj=preset['conv3x3_proj'],
        use_tanh_out=preset['use_tanh_out'], clamp_logvar=preset['clamp_logvar'],
        attn_zero_init=preset['attn_zero_init'],
        time_cond_decoder=True, dec_time_emb_dim=128)
    if args.mode == 'hier':
        vae = HierarchicalFMNISTVAE(args.z1_channels, args.z2_channels, **opts)
        channels = (args.z1_channels, args.z2_channels)
    else:
        vae = FlatFMNISTVAE(**opts)
        channels = (4,)
    heads = []
    for i, c in enumerate(channels):
        depth = 8 if args.score_budget == 'per_head' or args.mode == 'flat' else 4
        cfg = dict(in_channels=c, latent_size=vae.latent_shapes[i][-1], patch_size=1,
                   hidden_dim=192, depth=depth, num_heads=6, mlp_ratio=4.0,
                   dropout=0.0, num_classes=10, factored_head=False)
        if i == 0:
            heads.append(ref.DiTModel(**cfg))
        else:
            heads.append(ConditionalDiT(channels[0], **cfg))
    return vae.to(device), nn.ModuleList(heads).to(device), channels


def terminal_kl(mu, logvar, alpha_T, sigma_T):
    var = alpha_T**2 * torch.exp(logvar) + sigma_T**2
    return 0.5 * (alpha_T**2 * mu.square() + var - 1 - torch.log(var.clamp_min(1e-10))).mean()


def tmin_band_floor(schedule):
    """log(sigma0^2/alpha0^2) at the smallest training time (the B_tmin floor)."""
    times = schedule['times'].flatten()
    i0 = int(torch.argmin(times))
    a0 = schedule['alpha'].flatten()[i0].double()
    s0 = schedule['sigma'].flatten()[i0].double()
    return float(torch.log(s0**2) - torch.log(a0**2))


def tmin_band_penalty(logvar, log_floor):
    """Per-coordinate mean of B = 1/2 log(1 + sigma0^2/(alpha0^2 exp(logvar))).

    Upper bound on I(X;Z_0) - I(X;Z_{t_min}) for a Gaussian component (the rate
    dissipated on (0, t_min) that the truncated CSEM integral never sees).
    Written as a softplus for stability at clamped logvars; depends only on the
    encoder (no t sampling), so it adds no Monte Carlo variance.
    """
    return 0.5 * F.softplus(log_floor - logvar).mean()


def tmin_band_coefficient(args, schedule):
    """Resolve --tmin-band-weight.

    'auto': the canonical-equivalent weight the script's unit-epsilon CSEM loss
    applies at t_min.  Uniform index sampling on N grid points gives
    E||eps_hat - eps_CSEM||^2 = sum_i (1/N) sigma_i^2 ||s_hat - s_x||^2, i.e. physical
    time weight lambda(t_i) = sigma_i^2 / (N * dt_i).  B continues that integral
    below t_min, so it gets csem_rep_weight * lambda(t_0). On a log-t grid
    lambda(t) ~ 2/log(t_max/t_min) for small t.
    """
    spec = str(args.tmin_band_weight).strip().lower()
    if spec in ('off', 'none'):
        return 0.0
    if spec != 'auto':
        w = float(spec)
        if w < 0:
            raise ValueError('--tmin-band-weight must be >= 0 or auto')
        return w
    times, order = torch.sort(schedule['times'].detach().double().flatten())
    N = times.numel()
    s0 = schedule['sigma'].detach().double().flatten()[order[0]]
    dt0 = float(times[1] - times[0]) if N > 1 else 0.0
    if not dt0 > 0:  # degenerate grid: fall back to log-uniform density
        t0, t1 = float(times[0]), float(times[-1])
        dt0 = t0 * math.log(t1 / t0) / max(N - 1, 1) if t1 > t0 > 0 else 0.0
    if not dt0 > 0:
        raise ValueError('Cannot infer the t_min quadrature mass; pass --tmin-band-weight explicitly')
    lam0 = float(s0**2) / (N * dt0)
    return representation_weight(args) * lam0


def representation_weight(args):
    """Absolute encoder CSEM coefficient; None preserves historical coupling."""
    weight = getattr(args, 'csem_rep_weight', None)
    return float(args.score_weight if weight is None else weight)


def reconstruction_root_mode(args):
    """The flat control always reconstructs from its corrupted latent."""
    return getattr(args, 'recon_root', 'noisy') if args.mode == 'hier' else 'noisy'


def scale_representation_gradient(value, scale):
    """Identity forward, scaled backward at each encoder->CSEM boundary.

    Apply to noisy inputs, analytic targets and clean ancestors independently.
    Do not scale the scalar loss: that would also change score-head gradients.
    Reconstruction and terminal-KL graphs use the original tensors.
    """
    if scale == 1.0:
        return value
    return value.detach() + scale * (value - value.detach())


def score_and_losses(args, vae, heads, x, labels, schedule):
    """Differentiable through encoder, conditional contexts, and target precision.

    details['band_loss'] is the dimension-weighted B_tmin over all blocks; it is
    returned unweighted and added to the joint loss by the caller.
    Loss values retain the score-head convention; --csem-rep-weight scales only
    their encoder backward route. csem_routing.csv records the encoder coefficient.
    """
    if args.mode == 'hier':
        latents, params, top_down = vae.encode_sample(x, return_context=True)
    else:
        latents, params = vae.encode_sample(x)
    B = x.shape[0]
    rep_scale = representation_weight(args) / args.score_weight
    idx = torch.randint(len(schedule['times']), (B,), device=x.device)
    t = schedule['times'][idx]
    alpha = schedule['alpha'][idx].view(B, 1, 1, 1)
    sigma = schedule['sigma'][idx].view(B, 1, 1, 1)
    zt, eps, targets = [], [], []
    scores, kls = [], []
    bands, below, deep = [], [], []
    y_drop = labels.clone()
    mask = torch.rand(B, device=x.device) < args.label_dropout
    y_drop[mask] = 10
    alpha_T = schedule['alpha'][-1]
    sigma_T = schedule['sigma'][-1]
    log_floor = schedule.get('_band_log_floor')
    if log_floor is None:  # cache: avoids a device sync every step
        log_floor = schedule['_band_log_floor'] = tmin_band_floor(schedule)
    for i, (z, (mu, lv)) in enumerate(zip(latents, params)):
        # B_tmin for this block (conditional on z_{<i} through lv in hier mode).
        bands.append(tmin_band_penalty(lv, log_floor))
        with torch.no_grad():
            # Fraction of coordinates with sigma_phi < sigma0/alpha0 (losing >=50%
            # of the truncated objective's anti-sharpening gradient) and with
            # sigma_phi < sigma0/(10 alpha0) (losing >=99%).
            below.append((lv < log_floor).float().mean())
            deep.append((lv < log_floor - 2 * math.log(10.0)).float().mean())
        noise = torch.randn_like(z)
        noisy = alpha * z + sigma * noise
        var_t = alpha**2 * torch.exp(lv) + sigma**2
        # E[epsilon | y_t, X, z_{<i}] for a Gaussian component:
        target = sigma * (noisy - alpha*mu) / var_t.clamp_min(1e-10)
        score_input = scale_representation_gradient(noisy, rep_scale)
        score_target = scale_representation_gradient(target, rep_scale)
        if i == 0:
            estimate = heads[i](score_input, t, y_drop)
        else:
            context = scale_representation_gradient(latents[0], rep_scale)
            estimate = heads[i](score_input, t, y_drop, context=context)
        loss_i = F.mse_loss(estimate, score_target)
        scores.append(loss_i)
        kls.append(terminal_kl(mu, lv, alpha_T, sigma_T))
        zt.append(noisy)
        eps.append(noise)
        targets.append(target)
    # Weight by COORDINATES, including spatial dimensions: 64:192 by default.
    dims = [z[0].numel() for z in latents]
    weights = [n / sum(dims) for n in dims]
    score_loss = sum(w * L for w, L in zip(weights, scores))
    kl_loss = sum(w * K for w, K in zip(weights, kls))
    band_loss = sum(w * Bb for w, Bb in zip(weights, bands))
    # The conditional Gaussian q(z2|x,z1) and child CSEM context use CLEAN z1.
    # The image-denoising reconstruction is a separate objective at the sampled
    # time tau=t: decoder sees y1_tau, y2_tau (NO clean root feature cache), and
    # recomputes all root/child fusion features from these noisy latents.
    recon_mode = reconstruction_root_mode(args)
    if args.mode == 'hier' and recon_mode == 'clean':
        recon_root = latents[0]
        reconstruction = vae.decode(recon_root, zt[1], t=t, top_down=top_down)
    else:
        recon_root = zt[0]
        reconstruction = vae.decode(*zt, t=t)
    details = {
        'score_per_block': scores, 'kl_per_block': kls,
        'block_dimensions': dims, 'block_weights': weights,
        'band_loss': band_loss, 'band_per_block': bands,
        'below_per_block': below, 'deep_per_block': deep,
        'band_log_floor': log_floor,
        'latents': latents, 'params': params,
        'noisy': zt, 'target_eps': targets, 'time': t,
        'y_drop': y_drop, 'reconstruction': reconstruction,
        'reconstruction_root': recon_root, 'reconstruction_root_mode': recon_mode,
    }
    return score_loss, kl_loss, details


@torch.no_grad()
def ema_update(online, target, decay):
    for p, q in zip(online.parameters(), target.parameters()):
        q.mul_(decay).add_(p, alpha=1-decay)
    # no parameter buffers in DiT except fixed sincos positional embeddings


def make_sampler(args):
    return ref.UniversalSampler('rk4_ode', args.rk4_steps,
                                t_min=args.t_min, t_max=args.t_max,
                                schedule_type='log_t')


@torch.no_grad()
def sample_blocks(args, heads, B, device, y=None, root=None, bank=None):
    sampler = make_sampler(args)
    if bank is None:
        bank = [None] * len(heads)
    if root is None:
        z1 = sampler.sample(heads[0], shape=(B, *latent_shapes(args)[0]),
                            device=device, x_init=bank[0], y=y, cfg_scale=args.cfg)
    else:
        z1 = root
    if args.mode == 'flat':
        return (z1,)
    conditional = BoundConditionalHead(heads[1], z1)
    z2 = sampler.sample(conditional, shape=(B, *latent_shapes(args)[1]),
                        device=device,x_init=bank[1], y=y, cfg_scale=args.cfg)
    return z1,z2


def latent_shapes(args):
    return ((args.z1_channels, 4, 4), (args.z2_channels, 8, 8)) if args.mode == 'hier' else ((4, 8, 8),)


def to_grid(images, path, nrow=8):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_image((images[:64].clamp(-1,1).float()+1)*0.5, str(path),nrow=nrow)


def prep_fid(args, loader, device, out):
    """Use the identical legacy FMNIST Inception features by default."""
    if args.eval_samples < 2:
        return None
    xs, ys = [], []
    for x, y in loader:
        need = args.eval_samples-sum(q.shape[0] for q in xs)
        if need <= 0: break
        xs.append(x[:need]); ys.append(y[:need])
    real = torch.cat(xs); labels = torch.cat(ys)
    to_grid(real, out/'target_real.png')
    np.save(out/'target_real_first256.npy', real[:256].numpy())
    if args.fid_backend == 'inception-legacy':
        feats, inception = ref.extract_inception_features(real,device,args.feature_batch)
        return (real,labels,feats,inception)
    else:
        ckpt = out/'fid_classifier_fmnist.pt'
        if args.fid_ckpt is not None:
            ckpt = Path(args.fid_ckpt)
            if not ckpt.exists(): raise FileNotFoundError(ckpt)
        clf = ref.train_fid_classifier(
            loader,10,device,epochs=args.fid_classifier_epochs,
            checkpoint_path=str(ckpt),img_size=32,img_channels=1)
        feats = []
        for start in range(0,len(real),args.feature_batch):
            feats.append(clf.extract_features(real[start:start+args.feature_batch].to(device)).cpu())
        return real,labels,torch.cat(feats),clf


def fid_features(images, ref_model, args, device):
    if args.fid_backend == 'inception-legacy':
        return ref.extract_inception_features(images,device,args.feature_batch,
                                              inception_model=ref_model)[0]
    feats=[]
    with torch.no_grad():
        for off in range(0,len(images),args.feature_batch):
            feats.append(ref_model.extract_features(images[off:off+args.feature_batch].to(device)).cpu())
    return torch.cat(feats)


def fid_numeric(a,b):
    # Numerical fallback compatible with newer SciPy versions (sqrtm disp removed).
    try:
        return ref.compute_fid_from_features(a,b)
    except TypeError as e:
        if 'disp' not in str(e): raise
        from scipy.linalg import sqrtm
        X,Y = a.cpu().numpy().astype('float64'),b.cpu().numpy().astype('float64')
        cr,cf=np.cov(X,rowvar=False),np.cov(Y,rowvar=False)
        sm=sqrtm(cr@cf)
        if np.iscomplexobj(sm): sm=sm.real
        return float(np.square(X.mean(0)-Y.mean(0)).sum()+np.trace(cr)+np.trace(cf)-2*np.trace(sm))


@torch.no_grad()
def evaluate(args, epoch, vae, heads, test_loader, fid_info, device, out, *,
             online_heads=None, schedule=None, lpips_net=None, disc=None):
    # Diagnostic draws must not change the subsequent training trajectory.
    modules = {m: m.training for model in (vae, heads, online_heads, lpips_net, disc)
               if model is not None for m in model.modules()}
    numpy_state, python_state = np.random.get_state(), random.getstate()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(args.seed + 12003)
            if devices:
                with torch.cuda.device(devices[0]): torch.cuda.manual_seed(args.seed + 12003)
            np.random.seed(args.seed + 12003); random.seed(args.seed + 12003)
            return _evaluate(args, epoch, vae, heads, fid_info, device, out,
                             online_heads, schedule, lpips_net, disc)
    finally:
        np.random.set_state(numpy_state); random.setstate(python_state)
        for module, training in modules.items(): module.training = training


def _evaluate(args, epoch, vae, heads, fid_info, device, out,
              online_heads, schedule, lpips_net, disc):
    real, labs, real_feat, feat_model = fid_info
    vae.eval(); heads.eval()
    N=len(real)
    generated=[]; rec=[]; ancestral=[]
    checks = getattr(args, 'eval_checks', True)
    check_n = min(N, getattr(args, 'eval_check_samples', 1024)) if checks else 0
    usage_n = min(N, getattr(args, 'eval_usage_samples', 128)) if checks and args.mode == 'hier' else 0
    if checks and getattr(args, 'eval_grad_batch', 16) > 0 and (online_heads is None or schedule is None):
        raise ValueError('Eval gradient checks require online_heads and training schedule; '
                         'pass them or set --eval-grad-batch 0')
    keep_n = max(check_n, usage_n)
    starts = [[] for _ in latent_shapes(args)]
    posteriors = [[] for _ in latent_shapes(args)]
    # Fixed starts with a different seed stream for each block and checkpoint.
    # Same starts/labels across eval epochs and both modes for comparability.
    root_seed=args.seed+12001; child_seed=args.seed+12002
    g1=torch.Generator(device='cpu').manual_seed(root_seed)
    g2=torch.Generator(device='cpu').manual_seed(child_seed)
    for off in range(0,N,args.eval_batch):
        y=labs[off:off+args.eval_batch].to(device)
        b=len(y)
        bank=[torch.randn((b, *latent_shapes(args)[0]),generator=g1).to(device)]
        if args.mode=='hier':
            bank.append(torch.randn((b, *latent_shapes(args)[1]),generator=g2).to(device))
        keep = max(0, min(b, keep_n-off))
        if keep:
            for dest, value in zip(starts, bank): dest.append(value[:keep].cpu().clone())
        z=sample_blocks(args,heads,b,device,y=y,bank=[q.clone() for q in bank])
        t_eval=torch.full((b,),args.decode_time,device=device)
        fake=vae.decode(*z,t=t_eval).clamp(-1,1)
        generated.append(fake.cpu())
        # True posterior reconstructions and encoded-root / generated-descendant
        # diagnostics are conditioned on the same real examples and labels.
        x=real[off:off+b].to(device)
        zpost,_=vae.encode_sample(x)
        if keep:
            for dest, value in zip(posteriors, zpost): dest.append(value[:keep].cpu())
        rec.append(vae.decode(*zpost,t=t_eval).clamp(-1,1).cpu())
        if args.mode=='hier':
            anc_bank=[None,bank[1].clone()]
            z_anc=sample_blocks(args,heads,b,device,y=y,root=zpost[0],bank=anc_bank)
            ancestral.append(vae.decode(*z_anc,t=t_eval).clamp(-1,1).cpu())
    generated=torch.cat(generated)
    rec=torch.cat(rec)
    if ancestral: ancestral=torch.cat(ancestral)
    fid_gen=fid_numeric(real_feat,fid_features(generated,feat_model,args,device))
    fid_rec=fid_numeric(real_feat,fid_features(rec,feat_model,args,device))
    fid_ancestral=float('nan')
    if isinstance(ancestral,torch.Tensor):
        fid_ancestral=fid_numeric(real_feat,fid_features(ancestral,feat_model,args,device))
    epdir=out/'samples'/f'epoch_{epoch:04d}'
    epdir.mkdir(parents=True,exist_ok=True)
    to_grid(generated,epdir/'generated_cfg.png')
    to_grid(rec,epdir/'posterior_reconstruction.png')
    to_grid(real,epdir/'target_real.png')
    if isinstance(ancestral,torch.Tensor):
        to_grid(ancestral,epdir/'encoded_root_generated_child.png')
    for name, v in [('generated',generated),('posterior_recon',rec),('encoded_root_child',ancestral)]:
        if isinstance(v,torch.Tensor):
            np.save(epdir/f'{name}_first{args.save_array_samples}.npy',
                    v[:args.save_array_samples].numpy())
    row={'epoch':epoch, 'fid_generated':fid_gen,'fid_posterior_recon':fid_rec,
         'fid_encoded_root_generated_child':fid_ancestral,
         'fid_backend':args.fid_backend,'cfg':args.cfg,'rk4_steps':args.rk4_steps,
         'num_eval':N}
    with (out/'eval.csv').open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row));
        if f.tell()==0: writer.writeheader()
        writer.writerow(row)
    print('[EVAL]',json.dumps(row,default=str),flush=True)
    if keep_n:
        starts = [torch.cat(q) for q in starts]
        posteriors = [torch.cat(q) for q in posteriors]
    if check_n >= 2:
        eval_sampling_checks(args, epoch, vae, heads, real, labs, real_feat,
                             feat_model, generated, ancestral, starts, posteriors,
                             check_n, device, out, epdir)
    if usage_n:
        eval_child_usage(args, epoch, vae, heads, real, labs, rec, ancestral,
                         posteriors, usage_n, device, out, epdir, lpips_net)
    if checks and getattr(args, 'eval_grad_batch', 16) > 0:
        eval_gradient_checks(args, epoch, vae, online_heads, real, labs,
                             schedule, device, out, lpips_net, disc)
    return row


@torch.no_grad()
def eval_sampling_checks(args, epoch, vae, heads, real, labs, real_feat,
                         feat_model, generated, ancestral, starts, posteriors,
                         count, device, out, epdir):
    configs = list(dict.fromkeys([(args.cfg, args.rk4_steps),
                                 (1.0, args.rk4_steps),
                                 (args.cfg, 2*args.rk4_steps),
                                 (1.0, 2*args.rk4_steps)]))
    for cfg, steps in configs:
        settings = copy.copy(args); settings.cfg = cfg; settings.rk4_steps = steps
        if (cfg, steps) == (args.cfg, args.rk4_steps):
            images = generated[:count]
            children = ancestral[:count] if args.mode == 'hier' else None
        else:
            images, children = [], []
            for off in range(0, count, args.eval_batch):
                end = min(off+args.eval_batch, count); b = end-off
                y = labs[off:end].to(device)
                bank = [q[off:end].to(device).clone() for q in starts]
                z = sample_blocks(settings, heads, b, device, y=y, bank=bank)
                t = torch.full((b,), args.decode_time, device=device)
                images.append(vae.decode(*z, t=t).clamp(-1, 1).cpu())
                if args.mode == 'hier':
                    zc = sample_blocks(settings, heads, b, device, y=y,
                                       root=posteriors[0][off:end].to(device),
                                       bank=[None, starts[1][off:end].to(device).clone()])
                    children.append(vae.decode(*zc, t=t).clamp(-1, 1).cpu())
            images = torch.cat(images)
            children = torch.cat(children) if children else None
        row = {'epoch': epoch, 'cfg': cfg, 'rk4_steps': steps, 'num_eval': count,
               'fid_backend': args.fid_backend,
               'fid_generated': fid_numeric(real_feat[:count], fid_features(images, feat_model, args, device)),
               'fid_encoded_root_generated_child': (fid_numeric(real_feat[:count], fid_features(children, feat_model, args, device))
                                                     if children is not None else float('nan')),
               'mse_generated_vs_default': float(F.mse_loss(images, generated[:count])),
               'mse_child_vs_default': (float(F.mse_loss(children, ancestral[:count]))
                                        if children is not None else float('nan'))}
        write_csv(out/'eval_sampling_checks.csv', row)
        suffix = f'cfg_{cfg:g}_steps_{steps}'
        to_grid(images, epdir/f'check_generated_{suffix}.png')
        if children is not None: to_grid(children, epdir/f'check_encoded_root_child_{suffix}.png')
        print('[EVAL_SAMPLING]', json.dumps(row), flush=True)


@torch.no_grad()
def eval_child_usage(args, epoch, vae, heads, real, labs, rec, ancestral,
                     posteriors, count, device, out, epdir, lpips_net):
    generator = torch.Generator(device='cpu').manual_seed(args.seed+12004)
    second_start = torch.randn((count, *latent_shapes(args)[1]), generator=generator)
    # Shuffle direct child codes across examples, keeping each recipient root fixed.
    donor = torch.randperm(count, generator=generator)
    variants = {'posterior_child': rec[:count], 'generated_child_1': ancestral[:count],
                'zero_child': [], 'shuffled_child': [], 'generated_child_2': []}
    for off in range(0, count, args.eval_batch):
        end = min(off+args.eval_batch, count); b = end-off
        root, child = [q[off:end].to(device) for q in posteriors]
        t = torch.full((b,), args.decode_time, device=device)
        variants['zero_child'].append(vae.decode(root, torch.zeros_like(child), t=t).clamp(-1, 1).cpu())
        shuffled = posteriors[1][donor[off:end]].to(device)
        variants['shuffled_child'].append(vae.decode(root, shuffled, t=t).clamp(-1, 1).cpu())
        z = sample_blocks(args, heads, b, device, y=labs[off:end].to(device), root=root,
                          bank=[None, second_start[off:end].to(device).clone()])
        variants['generated_child_2'].append(vae.decode(*z, t=t).clamp(-1, 1).cpu())
    variants = {name: torch.cat(value) if isinstance(value, list) else value
                for name, value in variants.items()}
    target, reference = real[:count], rec[:count]
    for name, images in variants.items():
        perceptual = [float('nan'), float('nan')]
        if lpips_net is not None:
            sums = [0.0, 0.0]
            for off in range(0, count, args.eval_batch):
                end = min(off+args.eval_batch, count)
                image = images[off:end].to(device).repeat(1, 3, 1, 1)
                for i, comparison in enumerate((target, reference)):
                    other = comparison[off:end].to(device).repeat(1, 3, 1, 1)
                    sums[i] += float(lpips_net(image, other).sum())
            perceptual = [value/count for value in sums]
        row = {'epoch': epoch, 'variant': name, 'num_eval': count,
               'mse_to_input': float(F.mse_loss(images, target)),
               'mse_to_posterior_recon': float(F.mse_loss(images, reference)),
               'lpips_to_input': perceptual[0], 'lpips_to_posterior_recon': perceptual[1],
               'mse_to_generated_child_1': float(F.mse_loss(images, variants['generated_child_1']))}
        write_csv(out/'eval_child_usage.csv', row)
        print('[EVAL_CHILD_USAGE]', json.dumps(row), flush=True)
    # Each ROW is the same example/root: real, posterior, zero, shuffled, draw1, draw2.
    columns = [target, reference, variants['zero_child'], variants['shuffled_child'],
               variants['generated_child_1'], variants['generated_child_2']]
    paired = torch.stack([q[:min(count, 10)] for q in columns], dim=1).flatten(0, 1)
    to_grid(paired, epdir/'child_usage_real_posterior_zero_shuffled_draw1_draw2.png', nrow=6)


@torch.enable_grad()
def eval_gradient_checks(args, epoch, vae, heads, real, labs, schedule,
                         device, out, lpips_net=None, disc=None):
    """Weighted gradients at one fixed probe batch; never writes parameter .grad.

    Online heads are essential: frozen EMA parameters would hide head gradients.
    Modules use eval mode, so stochastic layer dropout is disabled, but the
    training schedule, label-dropout draws and all encoder routes are retained.
    """
    vae.eval(); heads.eval()
    if not any(p.requires_grad for p in heads.parameters()):
        raise ValueError('Gradient audit needs trainable online heads, not frozen EMA heads')
    if disc is not None: disc.eval()
    count = min(len(real), args.eval_grad_batch)
    x, y = real[:count].to(device), labs[:count].to(device)
    _, kl, details = score_and_losses(args, vae, heads, x, y, schedule)
    pixel = F.mse_loss(details['reconstruction'], x)
    terms = [('pixel', pixel)]
    if lpips_net is not None:
        terms.append(('perceptual', args.perc_weight * lpips_net(
            details['reconstruction'].repeat(1, 3, 1, 1), x.repeat(1, 3, 1, 1)).mean()))
    if disc is not None and epoch >= args.disc_start:
        terms.append(('gan_generator', args.gan_weight * ref.hinge_g_loss(disc(details['reconstruction']))))
    terms.extend((f'csem_{i+1}', args.score_weight * w * loss)
                 for i, (w, loss) in enumerate(zip(details['block_weights'], details['score_per_block'])))
    terms.extend([('terminal', args.terminal_weight*kl),
                  ('band', tmin_band_coefficient(args, schedule)*details['band_loss'])])
    named = [('vae.'+n, p) for n, p in vae.named_parameters() if p.requires_grad]
    named += [('heads.'+n, p) for n, p in heads.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    groups = {
        'vae': [i for i, (n, _) in enumerate(named) if n.startswith('vae.')],
        'heads': [i for i, (n, _) in enumerate(named) if n.startswith('heads.')],
        'root_posterior': [i for i, (n, _) in enumerate(named) if n.startswith('vae.root_posterior.')],
        'child_posterior': [i for i, (n, _) in enumerate(named) if n.startswith('vae.condition.')],
        'image_encoder': [i for i, (n, _) in enumerate(named) if n.startswith(('vae.enc_', 'vae.encoder'))],
        'root_top_down': [i for i, (n, _) in enumerate(named) if n.startswith(('vae.root_in.', 'vae.root_blocks.', 'vae.root_up.'))],
    }
    def squared(grads, indices):
        return sum(float(grads[i].square().sum()) for i in indices if grads[i] is not None)
    def cosine(left, right, indices):
        denom = math.sqrt(squared(left, indices)*squared(right, indices))
        if denom == 0: return float('nan')
        return sum(float((left[i]*right[i]).sum()) for i in indices
                   if left[i] is not None and right[i] is not None)/denom
    def accumulate(dest, grads, indices):
        for i in indices:
            if grads[i] is not None:
                if dest[i] is None: dest[i] = grads[i].detach().clone()
                else: dest[i].add_(grads[i].detach())
    all_indices = list(range(len(params)))
    total, reconstruction, csem = [[None]*len(params) for _ in range(3)]
    rows = []; total_value = 0.0
    for name, loss in terms:
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        row = {'epoch': epoch, 'term': name, 'num_eval': count,
               'head_source': 'online', 'model_mode': 'eval',
               'encoder_csem_weight': representation_weight(args), 'head_csem_weight': args.score_weight,
               'weighted_loss_value': float(loss.detach())}
        row.update({f'grad_norm_{group}': math.sqrt(squared(grads, indices))
                    for group, indices in groups.items()})
        rows.append(row); total_value += float(loss.detach())
        accumulate(total, grads, all_indices)
        if name in ('pixel', 'perceptual', 'gan_generator'):
            accumulate(reconstruction, grads, groups['vae'])
        if name.startswith('csem_'): accumulate(csem, grads, groups['vae'])
        del grads
    row = dict(rows[0]); row.update(term='joint', weighted_loss_value=total_value)
    row.update({f'grad_norm_{group}': math.sqrt(squared(total, indices))
                for group, indices in groups.items()})
    rows.append(row)
    joint_norm = math.sqrt(squared(total, all_indices))
    for row in rows:
        row.update(joint_unclipped_norm=joint_norm,
                   predicted_joint_clip_factor=min(1.0, 1.0/(joint_norm+1e-6)),
                   cosine_reconstruction_csem_vae=cosine(reconstruction, csem, groups['vae']),
                   cosine_reconstruction_csem_root_posterior=cosine(reconstruction, csem, groups['root_posterior']),
                   reconstruction_root_posterior_norm=math.sqrt(squared(reconstruction, groups['root_posterior'])),
                   csem_root_posterior_norm=math.sqrt(squared(csem, groups['root_posterior'])))
        write_csv(out/'eval_gradient_checks.csv', row)
        print('[EVAL_GRADIENT]', json.dumps(row), flush=True)


def write_csv(path,row):
    with path.open('a',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(row))
        if f.tell()==0: wr.writeheader()
        wr.writerow(row)


def save_checkpoint(out,epoch,vae,heads,ema,opt,sch,disc,opt_disc,args):
    torch.save({'epoch':epoch,'vae':vae.state_dict(),'heads':heads.state_dict(),
                'ema':ema.state_dict(),'optimizer':opt.state_dict(),
                'scheduler':sch.state_dict(),
                'disc':disc.state_dict() if disc else None,
                'disc_optimizer':opt_disc.state_dict() if opt_disc else None,
                'args':vars(args), 'rng_torch':torch.get_rng_state(),
                'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'rng_numpy':np.random.get_state(), 'rng_python':random.getstate()},
               out/'latest.pt')


def train(args):
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else
                        'cpu' if args.device=='auto' else args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
    train_loader,test_loader,_=ref.make_dataloaders(args.batch,args.num_workers,'FMNIST')
    # Historical code uses a common trained feature backend for FMNIST, not LeNet.
    fid_info=prep_fid(args,train_loader if args.fid_backend=='lenet' else test_loader,device,out)
    # Inception backend passes test_loader above; LeNet needs train_loader for training.
    if args.fid_backend=='lenet':
        # Load/fit classifier from train_loader but evaluate against test examples.
        _,_,_,clf=fid_info
        xs=[];ys=[]
        for x,y in test_loader:
            b=max(0,args.eval_samples-sum(a.shape[0] for a in xs))
            if not b: break
            xs.append(x[:b]);ys.append(y[:b])
        real=torch.cat(xs); y=torch.cat(ys)
        real_feat=fid_features(real,clf,args,device)
        fid_info=(real,y,real_feat,clf)
    # Match recovered source: classifier preparation must not change model init.
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    vae,heads,channels=build(args,device)
    args.architecture_version = ARCHITECTURE_VERSION if args.mode == 'hier' else 'flat_reference_v1'
    args.latent_shapes = [list(shape) for shape in vae.latent_shapes]
    args.child_latent_coordinates = ('direct_z2' if args.mode == 'hier' else 'flat_z')
    ema=copy.deepcopy(heads).eval().requires_grad_(False)
    S=dict(t_min=args.t_min,t_max=args.t_max,num_train_timesteps=1000,time_schedule='log_t')
    schedule=ref.make_schedule(S,device)
    band_w=tmin_band_coefficient(args,schedule)
    band_floor=tmin_band_floor(schedule)
    args.tmin_band_weight_resolved=band_w
    args.tmin_band_logvar_floor=band_floor
    args.csem_rep_weight_resolved=representation_weight(args)
    print(f'CSEM: encoder weight={args.csem_rep_weight_resolved:g}; '
          f'head weight={args.score_weight:g}; '
          f'block weights={[math.prod(s)/sum(math.prod(q) for q in vae.latent_shapes) for s in vae.latent_shapes]}',flush=True)
    print(f'B_tmin: weight={band_w:.4g} ({args.tmin_band_weight}); logvar floor '
          f'log(sigma0^2/alpha0^2)={band_floor:.3f} (sigma_phi floor {math.exp(0.5*band_floor):.3g})',flush=True)
    # Matched FMNIST optim: VAE lr=5e-4, score effective lr=2e-4;
    # due to multiplication by loss coefficient 0.6, original script used
    # lr_ldm/score_w_vae=2e-4/0.6.
    score_lr=args.lr_head/args.score_weight
    opt=AdamW([{'params':vae.parameters(),'lr':args.lr_vae},
               {'params':heads.parameters(),'lr':score_lr}],
              weight_decay=1e-4,betas=(0.9,0.95))
    sch=CosineAnnealingLR(opt,T_max=args.epochs,eta_min=1e-6)
    disc=ref.PatchDiscriminator(in_channels=1,ndf=64,n_layers=2).to(device) if args.gan_weight>0 else None
    opt_disc=AdamW(disc.parameters(),lr=args.lr_disc,betas=(0.5,0.9),weight_decay=0) if disc else None
    if args.perc_weight>0 and not ref.LPIPS_AVAILABLE:
        raise RuntimeError('Optimized FMNIST recipe requires lpips. Install `pip install lpips` or explicitly use --no-lpips for a nonmatched smoke test.')
    lpips_net=ref.lpips.LPIPS(net='vgg').to(device).eval() if args.perc_weight>0 else None
    if lpips_net is not None: lpips_net.requires_grad_(False)
    first=0
    if not args.resume and (out/'latest.pt').exists():
        raise FileExistsError(f'{out}/latest.pt already exists. Use a NEW --output directory '
                              'for an independent direct-z2 experiment, or --resume '
                              'only with a matching direct-z2 checkpoint.')
    if args.resume:
        ck=torch.load(args.resume,map_location=device,weights_only=False)
        old = ck['args']
        if old.get('mode') != args.mode:
            raise ValueError('Cannot resume a checkpoint with a different mode')
        if args.mode == 'hier' and (old.get('architecture_version') != ARCHITECTURE_VERSION or
                old.get('z1_channels') != args.z1_channels or old.get('z2_channels') != args.z2_channels):
            raise ValueError('Direct-z2 hierarchy requires a matching v2 checkpoint. '
                             'Old residual (u2->raw_z2) checkpoints are incompatible; start a fresh run.')
        vae.load_state_dict(ck['vae']);heads.load_state_dict(ck['heads'])
        ema.load_state_dict(ck['ema']);opt.load_state_dict(ck['optimizer']);sch.load_state_dict(ck['scheduler'])
        if disc is not None and ck.get('disc') is not None: disc.load_state_dict(ck['disc']);opt_disc.load_state_dict(ck['disc_optimizer'])
        first=int(ck['epoch'])
        if args.mode == 'hier':
            previous_recon_root = old.get('recon_root', 'clean')
            if previous_recon_root != args.recon_root:
                print(f'Resume: reconstruction root changes {previous_recon_root} -> {args.recon_root}; '
                      'child CSEM conditioning remains clean.', flush=True)
        torch.set_rng_state(ck['rng_torch']);np.random.set_state(ck['rng_numpy']);random.setstate(ck['rng_python'])
        if torch.cuda.is_available() and ck['rng_cuda'] is not None: torch.cuda.set_rng_state_all(ck['rng_cuda'])
    (out/'config.json').write_text(json.dumps(vars(args),indent=2))
    print(f'Reconstruction root: {reconstruction_root_mode(args)}; '
          'child posterior/score root: clean; sampling: sequential.', flush=True)
    print(f'Models: VAE={sum(p.numel() for p in vae.parameters()):,}; '
          f'score_heads={[sum(p.numel() for p in h.parameters()) for h in heads]}; '
          f'latent_shapes={vae.latent_shapes}; device={device}',flush=True)
    for epoch in range(first,args.epochs):
        start=time.time(); vae.train();heads.train();
        if disc is not None: disc.train()
        sums={k:0.0 for k in ['total','recon','score','terminal','band','lpips','gan_d','gan_g','score_1','score_2','kl_1','kl_2','var_1','var_2',
                               'band_1','band_2','below_1','below_2','deep_1','deep_2']}
        count=0
        for it,(x,y) in enumerate(train_loader):
            if args.max_batches and it>=args.max_batches: break
            x=x.to(device);y=y.to(device)
            ls,kl,d=score_and_losses(args,vae,heads,x,y,schedule)
            rec=F.mse_loss(d['reconstruction'],x)
            if lpips_net is not None:
                xx=x.repeat(1,3,1,1); xr=d['reconstruction'].repeat(1,3,1,1)
                perc=lpips_net(xr,xx).mean()
            else: perc=x.new_zeros(())
            active_disc=disc is not None and epoch+1>=args.disc_start
            if active_disc:
                opt_disc.zero_grad(set_to_none=True)
                dloss=ref.hinge_d_loss(disc(x),disc(d['reconstruction'].detach()))
                dloss.backward();opt_disc.step()
                gloss=ref.hinge_g_loss(disc(d['reconstruction']))
            else: dloss=x.new_zeros(());gloss=x.new_zeros(())
            band=d['band_loss']
            loss=rec+args.perc_weight*perc+args.terminal_weight*kl+args.score_weight*ls+args.gan_weight*gloss
            if band_w>0: loss=loss+band_w*band
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite joint loss: '+str([q.item() for q in (rec,ls,kl,perc,band)]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(vae.parameters())+list(heads.parameters()),1.0)
            opt.step()
            ema_update(heads,ema,args.ema)
            count+=1
            vals={'total':loss,'recon':rec,'score':ls,'terminal':kl,'band':band,'lpips':perc,'gan_d':dloss,'gan_g':gloss}
            for i,((mu,lv),si,ki,bi,fb,fd) in enumerate(zip(d['params'],d['score_per_block'],d['kl_per_block'],
                                                          d['band_per_block'],d['below_per_block'],d['deep_per_block'])):
                vals[f'score_{i+1}']=si
                vals[f'kl_{i+1}']=ki
                vals[f'var_{i+1}']=torch.exp(lv).mean()
                vals[f'band_{i+1}']=bi
                vals[f'below_{i+1}']=fb
                vals[f'deep_{i+1}']=fd
            for k,v in vals.items(): sums[k]+=float(v.detach())
            if args.smoke_test: break
        sch.step()
        row={'epoch':epoch+1,'seconds':round(time.time()-start,2)}
        row.update({k:(v/count if count else float('nan')) for k,v in sums.items()})
        write_csv(out/'loss.csv',row)
        # Separate file keeps loss.csv append-compatible with existing runs.
        write_csv(out/'csem_routing.csv',{'epoch':epoch+1,
            'encoder_csem_weight':args.csem_rep_weight_resolved,
            'head_csem_weight':args.score_weight,'band_weight':band_w,
            'encoder_csem_value':args.csem_rep_weight_resolved*row['score']})
        write_csv(out/'reconstruction_routing.csv',{'epoch':epoch+1,
            'recon_root':reconstruction_root_mode(args),
            'child_score_context':'clean' if args.mode=='hier' else 'none',
            'child_latent_coordinates':args.child_latent_coordinates,
            'encoder_csem_weight':args.csem_rep_weight_resolved})
        print('[TRAIN]',json.dumps(row),flush=True)
        if (epoch+1)%args.checkpoint_every==0 or epoch+1==args.epochs:
            save_checkpoint(out,epoch+1,vae,heads,ema,opt,sch,disc,opt_disc,args)
        if (epoch+1)%args.eval_every==0 or epoch+1==args.epochs:
            if fid_info is not None:
                evaluate(args,epoch+1,vae,ema,test_loader,fid_info,device,out,
                         online_heads=heads, schedule=schedule, lpips_net=lpips_net, disc=disc)
        if args.smoke_test: break
    # Save a single deterministic batch of exact supervision arrays to check
    # that both heads see the intended CSEM targets, including z1 context.
    vae.eval();heads.eval()
    x,y=next(iter(test_loader));x=x[:16].to(device);y=y[:16].to(device)
    with torch.no_grad():
        _,_,d=score_and_losses(args,vae,heads,x,y,schedule)
    np.savez_compressed(out/'csem_target_samples.npz',
        **{'x':x.cpu().numpy(), 'labels':y.cpu().numpy(),
           **{f'z{i+1}':z.cpu().numpy() for i,z in enumerate(d['latents'])},
           **{f'noisy{i+1}':z.cpu().numpy() for i,z in enumerate(d['noisy'])},
           **{f'csem_epsilon_target{i+1}':z.cpu().numpy() for i,z in enumerate(d['target_eps'])},
           **{f'mu{i+1}':m.cpu().numpy() for i,(m,v) in enumerate(d['params'])},
           **{f'logvar{i+1}':v.cpu().numpy() for i,(m,v) in enumerate(d['params'])},
           'times':d['time'].cpu().numpy(),
           'reconstruction_root_mode':np.array(d['reconstruction_root_mode']),
           'latent_coordinates':np.array('z1,z2_direct' if args.mode=='hier' else 'z_flat')})
    plot_history(out)
    print(f'Done: outputs in {out.resolve()}',flush=True)


def plot_history(out):
    """Save training and FID curves without requiring pandas or seaborn."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def read(name):
        p=out/name
        if not p.exists(): return []
        with p.open() as f: return list(csv.DictReader(f))
    L=read('loss.csv')
    if L:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for k in ('recon','score','terminal','lpips'):
            if k in L[0]:
                ax.plot([int(v['epoch']) for v in L],[float(v[k]) for v in L],label=k)
        ax.set(xlabel='Epoch',ylabel='Loss',title='FMNIST two-block CSEM training')
        ax.legend();fig.tight_layout();fig.savefig(out/'loss_curves.png',dpi=150);plt.close(fig)
        keys=[k for k in ('below_1','deep_1','below_2','deep_2') if k in L[0]]
        if keys:
            fig,ax=plt.subplots(figsize=(8,4.5))
            for k in keys:
                ys=[float(v[k]) for v in L]
                if k.endswith('_2') and not any(ys): continue  # flat mode
                ax.plot([int(v['epoch']) for v in L],ys,label=k.replace('below','σ_φ<σ_tmin').replace('deep','σ_φ<σ_tmin/10'))
            ax.set(xlabel='Epoch',ylabel='Fraction of latent coordinates',
                   title='Posterior sharpening below the t_min cutoff')
            ax.legend();fig.tight_layout();fig.savefig(out/'tmin_band_diagnostics.png',dpi=150);plt.close(fig)
    E=read('eval.csv')
    if E:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for k,lab in [('fid_generated','Full generated'),
                      ('fid_posterior_recon','Posterior reconstruction'),
                      ('fid_encoded_root_generated_child','Encoded root + generated child')]:
            if k in E[0] and any(math.isfinite(float(r[k])) for r in E):
                ax.plot([int(r['epoch']) for r in E],[float(r[k]) for r in E],label=lab,marker='o',ms=3)
        ax.set(xlabel='Epoch',ylabel='Legacy Inception FID' if E[0]['fid_backend']=='inception-legacy' else 'LeNet feature FID',
               title='FMNIST FID versus epoch')
        ax.legend();fig.tight_layout();fig.savefig(out/'fid_curves.png',dpi=150);plt.close(fig)


def self_test():
    """Offline executable test of both paths, gradients, and hierarchical CSEM."""
    from types import SimpleNamespace
    device=torch.device('cpu')
    torch.set_num_threads(min(torch.get_num_threads(),2))
    for mode in ('flat','hier'):
        a=SimpleNamespace(mode=mode,z1_channels=4,z2_channels=3,score_budget='matched_total',
                          label_dropout=0.1,score_weight=0.6,t_min=2e-5,t_max=2.0,
                          tmin_band_weight='auto',rk4_steps=2,cfg=1.5)
        torch.manual_seed(42)
        vae,heads,cs=build(a,device)
        sch=ref.make_schedule({'time_schedule':'log_t','t_min':a.t_min,
                               't_max':a.t_max,'num_train_timesteps':1000},device)
        bw=tmin_band_coefficient(a,sch)
        assert bw>0 and math.isfinite(bw)
        x=torch.rand(2,1,32,32)*2-1
        y=torch.tensor([2,5])
        sl,kl,d=score_and_losses(a,vae,heads,x,y,sch)
        band=d['band_loss']
        assert len(d['band_per_block'])==len(cs)
        expected_shapes = latent_shapes(a)
        assert vae.latent_shapes == expected_shapes
        assert [tuple(z.shape[1:]) for z in d['latents']] == list(expected_shapes)
        assert sum(d['block_dimensions']) == 256
        assert d['reconstruction'].shape == x.shape
        # Check epsilon targets against an independent log-density derivative,
        # holding Gaussian parameters and the ancestor fixed while differentiating y.
        at = sch['alpha'][torch.searchsorted(sch['times'], d['time'])].view(2,1,1,1)
        st = sch['sigma'][torch.searchsorted(sch['times'], d['time'])].view(2,1,1,1)
        for noisy, (mu, lv), target in zip(d['noisy'], d['params'], d['target_eps']):
            point = noisy.detach().clone().requires_grad_(True)
            variance = (at.square()*lv.detach().exp()+st.square()).clamp_min(1e-10)
            log_density = -0.5*((point-at*mu.detach()).square()/variance+variance.log()).sum()
            component_score = torch.autograd.grad(log_density, point)[0]
            torch.testing.assert_close(target.detach(), -st*component_score)
        rec=F.mse_loss(d['reconstruction'],x)
        loss=rec+0.6*sl+kl+bw*band
        loss.backward()
        assert all(torch.isfinite(z).all() for z in [loss,sl,kl,band])
        # B_tmin: nonnegative, ~0 far above the floor, and slope -1/2 per unit
        # logvar far below it (restoring the full rate gradient).
        fl=tmin_band_floor(sch)
        lv=torch.tensor([fl+12.0,fl-12.0],dtype=torch.float64,requires_grad=True)
        per=0.5*F.softplus(fl-lv)
        per.sum().backward()
        assert per[0].item()<1e-5 and abs(lv.grad[1].item()+0.5)<1e-4 and (per>=0).all()
        for p in [*vae.parameters(), *heads.parameters()]:
            assert p.grad is None or torch.isfinite(p.grad).all()
        assert all(any(p.grad is not None and p.grad.abs().sum()>0 for p in h.parameters()) for h in heads)
        if mode=='hier':
            # Paired objective ablation: only reconstruction changes. Also track
            # actual top-down calls to catch accidental reuse of the clean cache.
            top_down_inputs = []
            original_ancestor = vae.ancestor_features
            def track_ancestor(z1):
                top_down_inputs.append(z1.detach().clone())
                return original_ancestor(z1)
            vae.ancestor_features = track_ancestor
            a.recon_root = 'noisy'
            try:
                with torch.no_grad():
                    torch.manual_seed(1300)
                    noisy_score, noisy_kl, noisy_audit = score_and_losses(a,vae,heads,x,y,sch)
                    assert len(top_down_inputs) == 2  # inference clean, reconstruction noisy
                    torch.testing.assert_close(top_down_inputs[0], noisy_audit['latents'][0])
                    torch.testing.assert_close(top_down_inputs[1], noisy_audit['noisy'][0])
                    assert not torch.equal(top_down_inputs[0], top_down_inputs[1])
                    torch.testing.assert_close(noisy_audit['reconstruction'],
                                               vae.decode(*noisy_audit['noisy'],t=noisy_audit['time']))
                    assert noisy_audit['reconstruction_root'] is noisy_audit['noisy'][0]
                    top_down_inputs.clear()
                    a.recon_root = 'clean'; torch.manual_seed(1300)
                    clean_score, clean_kl, clean_audit = score_and_losses(a,vae,heads,x,y,sch)
                    assert len(top_down_inputs) == 1  # deliberately reuse cache for control
                    torch.testing.assert_close(clean_score, noisy_score)
                    torch.testing.assert_close(clean_kl, noisy_kl)
                    for key in ('latents','noisy','target_eps','score_per_block'):
                        for clean, noisy in zip(clean_audit[key],noisy_audit[key]):
                            torch.testing.assert_close(clean,noisy)
                    assert not torch.equal(clean_audit['reconstruction'], noisy_audit['reconstruction'])
            finally:
                vae.ancestor_features = original_ancestor
                a.recon_root = 'noisy'
            # Both latent codes must directly influence reconstruction; child
            # image injection cannot bypass the conditional CSEM variable.
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in vae.child_in.parameters())
            # One actual optimizer step checks trainability of the new paths.
            AdamW([*vae.parameters(), *heads.parameters()],lr=1e-4).step()
            # The terminal/middle head must send gradients back to the root.
            vae.zero_grad(set_to_none=True);heads.zero_grad(set_to_none=True)
            # Separate forward because previous graph was freed.
            _sl,_kl,_d=score_and_losses(a,vae,heads,x,y,sch)
            _d['score_per_block'][1].backward()
            root_params=list(vae.root_posterior.parameters())
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in root_params)
            # The condition adapter must also get a gradient from block-2 CSEM.
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in vae.condition.parameters())
            # Block-2 B_tmin must reach the conditional adapter (it depends on z1).
            vae.zero_grad(set_to_none=True);heads.zero_grad(set_to_none=True)
            _sl,_kl,_d=score_and_losses(a,vae,heads,x,y,sch)
            _d['band_per_block'][1].backward()
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in vae.condition.parameters())
            assert all(p.grad is None or p.grad.abs().sum()==0 for p in heads.parameters())
            assert d['block_weights'] == [0.25,0.75]
            # No affine child transform exists: same z2 reaches child CSEM and decoder.
            with torch.no_grad():
                z1, z2 = d['latents']
                assert not hasattr(vae, 'child_baseline')
                # Exercise direct decoder injection (no implicit child recentering).
                x_direct = vae.decode(z1, z2)
                x_zero = vae.decode(z1, torch.zeros_like(z2))
                assert (x_direct-x_zero).abs().max() > 1e-6
                # Avoid vacuous context tests with DiT's zero-output init.
                nn.init.normal_(heads[1].final_proj.weight, std=0.01)
                pred = heads[1](d['noisy'][1],d['time'],y,context=z1)
                bound = BoundConditionalHead(heads[1],z1)
                torch.testing.assert_close(pred, bound(d['noisy'][1],d['time'],y))
                changed = heads[1](d['noisy'][1],d['time'],y,context=z1+1)
                assert (changed-pred).abs().max()>1e-6
                # Decoder cannot consume image skips: latents alone suffice.
                assert vae.decode(z1,z2).shape == x.shape
            # Real two-solve sampler, plus encoded-root child-only path.
            samples = sample_blocks(a,heads,2,device,y=y)
            assert [tuple(z.shape[1:]) for z in samples] == list(expected_shapes)
            assert all(torch.isfinite(z).all() for z in samples)
            anchored = sample_blocks(a,heads,2,device,y=y,root=d['latents'][0])
            assert anchored[0] is d['latents'][0]
        # Paired RNG/weights: only encoder CSEM gradients change, including the
        # child->root conditioning route. Forward values/head gradients coincide.
        parameters = [*vae.parameters(), *heads.parameters()]
        n_vae = len(list(vae.parameters()))
        baseline_grads = None
        baseline_value = None
        for multiplier in (1.0, 0.0, 3.0):
            a.csem_rep_weight = multiplier*a.score_weight
            torch.manual_seed(1200)
            routed, _, _ = score_and_losses(a,vae,heads,x,y,sch)
            value = a.score_weight*routed
            gradients = torch.autograd.grad(value,parameters,allow_unused=True)
            if baseline_grads is None:
                baseline_grads = [None if g is None else g.detach().clone() for g in gradients]
                baseline_value = value.detach().clone()
            else:
                torch.testing.assert_close(value.detach(),baseline_value)
                for j,(gradient,baseline,param) in enumerate(zip(gradients,baseline_grads,parameters)):
                    expected = (torch.zeros_like(param) if baseline is None else
                                baseline*(multiplier if j<n_vae else 1.0))
                    actual = torch.zeros_like(param) if gradient is None else gradient
                    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-4)
            assert math.isclose(tmin_band_coefficient(a,sch),multiplier*bw)
        a.csem_rep_weight = None
        # Independent direct loss reproduces the pre-flag joint cotraining path.
        routed, term, audit = score_and_losses(a,vae,heads,x,y,sch)
        direct_blocks = []
        for i,(noisy,target) in enumerate(zip(audit['noisy'],audit['target_eps'])):
            pred = (heads[i](noisy,audit['time'],audit['y_drop']) if i==0 else
                    heads[i](noisy,audit['time'],audit['y_drop'],context=audit['latents'][0]))
            direct_blocks.append(F.mse_loss(pred,target))
        direct = sum(w*loss_i for w,loss_i in zip(audit['block_weights'],direct_blocks))
        routed_grads = torch.autograd.grad(a.score_weight*routed,parameters,allow_unused=True,retain_graph=True)
        direct_grads = torch.autograd.grad(a.score_weight*direct,parameters,allow_unused=True,retain_graph=True)
        for lhs,rhs,param in zip(routed_grads,direct_grads,parameters):
            torch.testing.assert_close(torch.zeros_like(param) if lhs is None else lhs,
                                       torch.zeros_like(param) if rhs is None else rhs)
        # Reconstruction, K_T and B must not accidentally train score heads.
        non_score = F.mse_loss(audit['reconstruction'],x)+term+bw*audit['band_loss']
        assert all(g is None for g in torch.autograd.grad(non_score,list(heads.parameters()),
                                                        allow_unused=True,retain_graph=True))
        if mode=='hier':
            root_parameters = list(vae.root_posterior.parameters())
            noisy, target = audit['noisy'][1], audit['target_eps'][1]
            context, times, labels = audit['latents'][0], audit['time'], audit['y_drop']
            # Test the three routes separately. Detaches here are test controls,
            # not part of training: isolate input Jacobian, target, and context.
            input_only = F.mse_loss(heads[1](noisy,times,labels,context=context.detach()),target.detach())
            target_only = F.mse_loss(heads[1](noisy.detach(),times,labels,context=context.detach()).detach(),target)
            context_only = F.mse_loss(heads[1](noisy.detach(),times,labels,context=context),target.detach())
            for name,channel_loss in [('input Jacobian',input_only),('analytic target',target_only),
                                      ('clean-root context',context_only)]:
                channel_grads = torch.autograd.grad(channel_loss,root_parameters,allow_unused=True,retain_graph=True)
                assert any(g is not None and g.abs().sum()>0 for g in channel_grads), name
        print(f'[SELF-TEST] {mode}: rec={rec.item():.4f} score={sl.item():.4f} K_T={kl.item():.4f} '
              f'B_tmin={band.item():.4g} (w={bw:.4g}); gradients/routing OK',flush=True)
    print('Offline two-block CSEM self-test passed.')


def main():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--mode',choices=['hier','flat'],default='hier')
    p.add_argument('--z1-channels',type=int,default=4,help='Coarse root channels on a 4x4 grid')
    p.add_argument('--z2-channels',type=int,default=3,help='Direct Gaussian child z2 channels on an 8x8 grid; defaults total 256 coordinates')
    p.add_argument('--score-budget',choices=['per_head','matched_total'],default='per_head',
                   help='8 DiT blocks per head or 4 per hier head; matched_total matches trunk layers only, not FLOPs/parameters')
    p.add_argument('--epochs',type=int,default=700)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--lr-vae',type=float,default=5e-4)
    p.add_argument('--lr-head',type=float,default=2e-4)
    p.add_argument('--score-weight',type=float,default=0.6)
    p.add_argument('--recon-root',choices=['noisy','clean'],default='noisy',
                   help='Hierarchical reconstruction root: noisy corrupts both blocks and all decoder root paths; '
                        'clean is a non-default bypass ablation only. Child posterior/score context stays clean. '
                        'Flat mode always uses noisy reconstruction.')
    p.add_argument('--csem-rep-weight',type=float,default=None,
                   help='Absolute CSEM coefficient into the encoder/VAE only; default inherits --score-weight (0.6). '
                        '0 disables CSEM representation gradients. Score-head training stays unchanged; auto B weight follows this coefficient.')
    p.add_argument('--terminal-weight',type=float,default=1.0)
    p.add_argument('--tmin-band-weight',type=str,default='auto',
                   help="Weight on the small-time band closure B_tmin. 'auto' = encoder CSEM weight * "
                        "sigma0^2/(N*dt0), the script's own CSEM weight at t_min; a float sets it; "
                        "0 recovers the previous objective")
    p.add_argument('--perc-weight',type=float,default=0.85)
    p.add_argument('--gan-weight',type=float,default=0.0025)
    p.add_argument('--disc-start',type=int,default=25)
    p.add_argument('--lr-disc',type=float,default=1e-4)
    p.add_argument('--label-dropout',type=float,default=0.1)
    p.add_argument('--ema',type=float,default=0.9997)
    p.add_argument('--t-min',type=float,default=2e-5)
    p.add_argument('--t-max',type=float,default=2.0)
    p.add_argument('--decode-time',type=float,default=1e-4)
    p.add_argument('--eval-every',type=int,default=50)
    p.add_argument('--eval-samples',type=int,default=10000)
    p.add_argument('--eval-batch',type=int,default=128)
    p.add_argument('--eval-checks',action=argparse.BooleanOptionalAction,default=True,
                   help='Run paired sampling, child-usage and online-gradient checks at each eval')
    p.add_argument('--eval-check-samples',type=int,default=1024,
                   help='Paired FID subset for CFG=1 and doubled RK4 steps; 0 skips sampling checks')
    p.add_argument('--eval-usage-samples',type=int,default=128,
                   help='Fixed-root child replacement/repeated-draw probes in hier mode; 0 skips')
    p.add_argument('--eval-grad-batch',type=int,default=16,
                   help='Online-head objective gradient probe batch before joint clipping; 0 skips')
    p.add_argument('--rk4-steps',type=int,default=20)
    p.add_argument('--cfg',type=float,default=3.0)
    p.add_argument('--fid-backend',choices=['inception-legacy','lenet'],default='inception-legacy')
    p.add_argument('--fid-ckpt',type=str,default=None)
    p.add_argument('--fid-classifier-epochs',type=int,default=10)
    p.add_argument('--feature-batch',type=int,default=128)
    p.add_argument('--save-array-samples',type=int,default=256)
    p.add_argument('--checkpoint-every',type=int,default=50)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--max-batches',type=int,default=0)
    p.add_argument('--output',type=str,default='runs/fmnist_hier_direct_z2_s42')
    p.add_argument('--resume',default=None)
    p.add_argument('--no-lpips',action='store_true')
    p.add_argument('--smoke-test',action='store_true',help='1 minibatch, 1 epoch on FashionMNIST; no FID')
    p.add_argument('--self-test',action='store_true',help='offline synthetic forward/backprop test; downloads nothing')
    a=p.parse_args()
    if a.self_test:
        self_test(); return
    if a.z1_channels<1 or a.z2_channels<1: p.error('--z1-channels and --z2-channels must be positive')
    if min(a.eval_check_samples,a.eval_usage_samples,a.eval_grad_batch)<0:
        p.error('Evaluation diagnostic sample counts must be nonnegative')
    if a.eval_check_samples==1: p.error('--eval-check-samples must be 0 or at least 2 for FID')
    if a.eval_batch<1 or a.rk4_steps<1: p.error('--eval-batch and --rk4-steps must be positive')
    if a.score_weight<=0: p.error('--score-weight must be positive')
    if a.csem_rep_weight is not None and (not math.isfinite(a.csem_rep_weight) or a.csem_rep_weight<0):
        p.error('--csem-rep-weight must be finite and nonnegative')
    if a.tmin_band_weight.strip().lower() not in ('auto','off','none'):
        try:
            if float(a.tmin_band_weight)<0: raise ValueError
        except ValueError:
            p.error("--tmin-band-weight must be 'auto', 'off', or a float >= 0")
    if a.no_lpips: a.perc_weight=0.0
    if a.smoke_test:
        a.epochs=1;a.max_batches=1;a.gan_weight=0;a.perc_weight=0
        a.eval_samples=0;a.eval_every=1;a.num_workers=0
    train(a)

if __name__=='__main__':
    main()
