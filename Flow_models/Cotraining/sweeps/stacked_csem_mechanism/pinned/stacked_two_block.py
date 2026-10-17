#!/usr/bin/env python3
"""FMNIST experiment: one joint score head, naive stacked CSEM component targets.

Encoder: q(z1|x) q(z2|x,z1). Decoder: D(z1,z2), identical to the direct-z2 baseline.
One head observes both OU-noised blocks at a common t. Targets retain the
original two conditional Gaussian precision residuals, in epsilon units.
All training paths through noisy inputs, target moments and ancestry remain
fully differentiable. Generation uses one joint RK4 solve, not two cascaded solves.
This is an empirical experiment; no FID failure or improvement is presumed.
Requires fmnist_reference.py alongside this file.

Hypothesis diagnostics (enabled by default; do not change the training objective):
  --diagnostic-every 25: fixed held-out representation/score probes, also at start.
  --reference-every 50: smaller, expensive aggregate-score importance audit.
  --diagnostic-every 0 --reference-every 0: original evaluation cost.
  --eval-only --resume ... --eval-samples 0: diagnostics without Inception/FID.
See diagnostics/README.md in the run directory for definitions and interpretation.
The architecture/checkpoint version is unchanged; original v1 checkpoints resume.
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
from contextlib import contextmanager
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
except ModuleNotFoundError as exc:
    if exc.name == 'fmnist_reference':
        raise RuntimeError('Place fmnist_reference.py beside this script.') from exc
    raise
ARCHITECTURE_VERSION = 'fmnist_joint_stacked_component_targets_v1'
BASELINE_SHA256 = '11db5d5ca81d9f51f5f361f86e029ecc7e2c83a2b41d424249b3076ee42c568f'

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


def to_grid(images, path, nrow=8):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_image((images[:64].clamp(-1,1).float()+1)*0.5, str(path),nrow=nrow)


def write_csv(path,row):
    with path.open('a',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(row))
        if f.tell()==0: wr.writeheader()
        wr.writerow(row)


def fid_numeric(a,b):
    """The same sample-covariance FID, using real PSD algebra at singular ranks.

    Avoid sqrtm(Cr@Cf), its spurious imaginary parts, and SciPy's removed disp
    API. No diagonal ridge is added, so the metric is not redefined. For the
    small diagnostic subsets, the equivalent sample-space nuclear norm is
    cheaper than decomposing a full 2048-dimensional covariance.
    """
    from scipy import linalg
    X,Y = a.detach().cpu().numpy().astype('float64'),b.detach().cpu().numpy().astype('float64')
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1] or min(len(X),len(Y)) < 2:
        raise ValueError('FID needs >=2 samples and matching [N,D] feature arrays')
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):
        raise ValueError('FID features must be finite')
    mean_term = np.square(X.mean(0)-Y.mean(0)).sum()
    X = (X-X.mean(0)) / math.sqrt(len(X)-1)
    Y = (Y-Y.mean(0)) / math.sqrt(len(Y)-1)
    traces = np.square(X).sum()+np.square(Y).sum()
    if max(len(X),len(Y)) < X.shape[1]:
        cross_trace = linalg.svdvals(X@Y.T,check_finite=False).sum()
    else:
        cr,cf = X.T@X,Y.T@Y
        values,vectors = linalg.eigh(cr,check_finite=False)
        root = (vectors*np.sqrt(np.maximum(values,0)))@vectors.T
        middle = root@cf@root
        values = linalg.eigvalsh((middle+middle.T)*0.5,check_finite=False)
        cross_trace = np.sqrt(np.maximum(values,0)).sum()
    value = float(mean_term+traces-2*cross_trace)
    if value < -1e-6*max(1.0,float(mean_term+traces)):
        raise FloatingPointError(f'FID is materially negative: {value}')
    return max(0.0,value)


class JointStackDiT(ref.DiTModel):
    """One head; sees only both noisy blocks, one time, and the class label."""
    def __init__(self, args):
        super().__init__(in_channels=args.z1_channels // 4 + args.z2_channels,
                         latent_size=8, patch_size=1, hidden_dim=args.joint_width,
                         depth=args.joint_depth, num_heads=args.joint_heads,
                         mlp_ratio=4.0, dropout=0.0, num_classes=10,
                         factored_head=False)


def pack_blocks(z1, z2):
    """A coordinate permutation: [B,4,4,4] -> [B,1,8,8], then concatenate.

    No interpolation, replication, padding, scaling, or lost coordinates.
    IID isotropic OU noise and coordinate-mean MSE commute with this packing.
    """
    if z1.ndim != 4 or z2.ndim != 4 or z1.shape[-2:] != (4, 4) or z2.shape[-2:] != (8, 8):
        raise ValueError('Expected root on 4x4 and child on 8x8')
    if z1.shape[0] != z2.shape[0] or z1.shape[1] % 4:
        raise ValueError('Matching batches and root channels divisible by 4 required')
    return torch.cat((F.pixel_shuffle(z1, 2), z2), dim=1)


def unpack_blocks(packed, root_channels=4):
    n = root_channels // 4
    return F.pixel_unshuffle(packed[:, :n], 2), packed[:, n:]


def build(args, device):
    preset = ref.DATASET_PRESETS['fmnist_reference']
    opts = dict(base_ch=preset['base_ch'], use_norm=False, img_size=32, img_channels=1,
                num_res_blocks=preset['num_res_blocks'], decoder_attn_half=preset['decoder_attn_half'],
                latent_proj_depth=preset['latent_proj_depth'], encoder_attn_half=preset['encoder_attn_half'],
                decoder_extra_block=preset['decoder_extra_block'], conv3x3_proj=preset['conv3x3_proj'],
                use_tanh_out=preset['use_tanh_out'], clamp_logvar=preset['clamp_logvar'],
                attn_zero_init=preset['attn_zero_init'], time_cond_decoder=True, dec_time_emb_dim=128)
    # This class is copied verbatim from the original DIRECT-z2 cascade.
    vae = HierarchicalFMNISTVAE(args.z1_channels, args.z2_channels, **opts)
    return vae.to(device), JointStackDiT(args).to(device)


def score_and_losses(args, vae, head, x, labels, schedule, *, indices=None):
    """Exactly one coherent ancestral draw, followed by independent OU noise.

    The requested naive target is retained: two componentwise precision
    residuals, stacked and multiplied by sigma_t (the original epsilon units).
    There is no clean-root input to the head and no encoder stop-gradient.
    """
    latents, params = vae.encode_sample(x)
    batch = len(x)
    if indices is None:
        indices = torch.randint(len(schedule['times']), (batch,), device=x.device)
    t = schedule['times'][indices]
    alpha = schedule['alpha'][indices].view(batch, 1, 1, 1)
    sigma = schedule['sigma'][indices].view(batch, 1, 1, 1)
    noisy, targets, raw_precision_residuals = [], [], []
    noises, variances = [], []
    kls, bands = [], []
    log_floor = schedule.get('_band_log_floor')
    if log_floor is None:
        log_floor = schedule['_band_log_floor'] = tmin_band_floor(schedule)
    for z, (mu, logvar) in zip(latents, params):
        noise = torch.randn_like(z)
        point = alpha * z + sigma * noise
        variance = alpha.square() * logvar.exp() + sigma.square()
        precision_residual = (point - alpha * mu) / variance.clamp_min(1e-10)
        # Analytic Gaussian score = -precision_residual.
        # UniversalSampler expects epsilon = -sigma*score, hence PLUS below.
        target = sigma * precision_residual
        noisy.append(point); targets.append(target)
        raw_precision_residuals.append(precision_residual)
        noises.append(noise); variances.append(variance)
        kls.append(terminal_kl(mu, logvar, schedule['alpha'][-1], schedule['sigma'][-1]))
        bands.append(tmin_band_penalty(logvar, log_floor))

    y_drop = labels.clone()
    y_drop[torch.rand(batch, device=x.device) < args.label_dropout] = 10
    joint_input = pack_blocks(*noisy)
    joint_target = pack_blocks(*targets)
    prediction = head(joint_input, t, y_drop)  # ONE joint forward, no clean context
    predictions = unpack_blocks(prediction, args.z1_channels)
    block_losses = [F.mse_loss(p, target) for p, target in zip(predictions, targets)]
    dimensions = [z[0].numel() for z in latents]
    weights = [n / sum(dimensions) for n in dimensions]
    score_loss = F.mse_loss(prediction, joint_target)
    terminal = sum(w * loss for w, loss in zip(weights, kls))
    band = sum(w * loss for w, loss in zip(weights, bands))
    # Both decoder blocks are OU-corrupted, exactly as in the old recipe.
    # Recompute root features from noisy[0]; never cache a clean-root bypass.
    reconstruction = vae.decode(*noisy, t=t)
    details = dict(latents=latents, params=params, noisy=noisy, target_eps=targets,
                   precision_residuals=raw_precision_residuals, noises=noises,
                   variances=variances, alpha=alpha, sigma=sigma, time=t,
                   y_drop=y_drop, joint_input=joint_input, joint_target=joint_target,
                   prediction=prediction, predictions=predictions,
                   score_per_block=block_losses, kl_per_block=kls,
                   band_per_block=bands, band_loss=band,
                   block_dimensions=dimensions, block_weights=weights,
                   reconstruction=reconstruction)
    return score_loss, terminal, details


@torch.no_grad()
def ema_update(online, target, decay):
    for p, q in zip(online.parameters(), target.parameters()):
        q.mul_(decay).add_(p, alpha=1-decay)
    for source, dest in zip(online.buffers(), target.buffers()):
        dest.copy_(source)


@torch.no_grad()
def sample_joint(args, head, batch, device, *, labels=None, starts=None):
    sampler = ref.UniversalSampler('rk4_ode', args.rk4_steps, t_min=args.t_min,
                                   t_max=args.t_max, schedule_type='log_t')
    packed = sampler.sample(head, shape=(batch, args.z1_channels//4+args.z2_channels, 8, 8),
                            device=device, x_init=starts, y=labels, cfg_scale=args.cfg)
    # Both latent blocks evolve together throughout this SINGLE reverse solve.
    return unpack_blocks(packed, args.z1_channels)


@contextmanager
def preserved_state(*models):
    states = {m: m.training for model in models if model is not None for m in model.modules()}
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        yield
    finally:
        torch.set_rng_state(torch_state)
        if cuda_state is not None: torch.cuda.set_rng_state_all(cuda_state)
        np.random.set_state(numpy_state); random.setstate(python_state)
        for module, training in states.items(): module.training = training


def prepare_fid(args, loader, device, out):
    if args.eval_samples == 0:
        return None
    images, labels, count = [], [], 0
    for x, y in loader:
        take = min(len(x), args.eval_samples-count)
        if take <= 0: break
        images.append(x[:take]); labels.append(y[:take]); count += take
    real, labels = torch.cat(images), torch.cat(labels)
    if len(real) < 2: raise ValueError('FID requires at least two images')
    features, inception = ref.extract_inception_features(real, device, args.feature_batch)
    to_grid(real, out/'target_real.png')
    return real, labels, features, inception


def fid_features(images, model, args, device):
    return ref.extract_inception_features(images, device, args.feature_batch, inception_model=model)[0]


@torch.no_grad()
def evaluate(args, epoch, vae, head, fid_info, device, out):
    with preserved_state(vae, head):
        vae.eval(); head.eval()
        torch.manual_seed(args.seed+22000)
        np.random.seed(args.seed+22000); random.seed(args.seed+22000)
        real, labels, real_features, inception = fid_info
        # Independent fixed CPU streams retain paired starts across epochs and batches.
        root_rng = torch.Generator().manual_seed(args.seed+12001)
        child_rng = torch.Generator().manual_seed(args.seed+12002)
        generated, reconstructed = [], []
        for start in range(0, len(real), args.eval_batch):
            x = real[start:start+args.eval_batch].to(device)
            y = labels[start:start+len(x)].to(device)
            initial = pack_blocks(
                torch.randn(len(x), args.z1_channels, 4, 4, generator=root_rng),
                torch.randn(len(x), args.z2_channels, 8, 8, generator=child_rng)).to(device)
            z = sample_joint(args, head, len(x), device, labels=y, starts=initial)
            times = torch.full((len(x),), args.decode_time, device=device)
            fake = vae.decode(*z, t=times)
            if not torch.isfinite(fake).all():
                raise FloatingPointError('Non-finite generated images; no FID is reported')
            generated.append(fake.clamp(-1, 1).cpu())
            posterior, _ = vae.encode_sample(x)
            reconstructed.append(vae.decode(*posterior, t=times).clamp(-1, 1).cpu())
        generated, reconstructed = torch.cat(generated), torch.cat(reconstructed)
        row = dict(epoch=epoch,
                   fid_generated=fid_numeric(real_features, fid_features(generated, inception, args, device)),
                   fid_posterior_recon=fid_numeric(real_features, fid_features(reconstructed, inception, args, device)),
                   fid_backend='inception-legacy', cfg=args.cfg, rk4_steps=args.rk4_steps,
                   sample_t_min=args.t_min, decode_time=args.decode_time,
                   head_source='ema', vae_source='online', num_eval=len(real),
                   prior='joint_stacked_component_targets')
        epdir = out/'samples'/f'epoch_{epoch:04d}'
        epdir.mkdir(parents=True, exist_ok=True)
        for name, values in [('generated_cfg', generated), ('posterior_reconstruction', reconstructed), ('target_real', real)]:
            to_grid(values, epdir/f'{name}.png')
            np.save(epdir/f'{name}_first{args.save_array_samples}.npy', values[:args.save_array_samples].numpy())
        write_csv(out/'eval.csv', row)
        print('[EVAL]', json.dumps(row), flush=True)
        plot_history(out)
        return row


def plot_history(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name, keys, ylabel in [
            ('loss', ('recon','score','score_1','score_2','terminal','lpips'), 'Training loss'),
            ('eval', ('fid_generated','fid_posterior_recon'), 'Legacy Inception FID')]:
        path = out/(name+'.csv')
        if not path.exists(): continue
        with path.open() as stream: rows = list(csv.DictReader(stream))
        if not rows: continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for key in keys:
            ax.plot([int(r['epoch']) for r in rows], [float(r[key]) for r in rows], label=key)
        ax.set(xlabel='Epoch', ylabel=ylabel, title='FMNIST: single joint head, stacked component targets')
        ax.legend(); fig.tight_layout(); fig.savefig(out/(name+'_curves.png'), dpi=150); plt.close(fig)


def atomic_torch_save(payload, path):
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def checkpoint_payload(args, epoch, vae, head, ema, opt, scheduler, disc, opt_disc, best_fid):
    return dict(architecture_version=ARCHITECTURE_VERSION, epoch=epoch,
                vae=vae.state_dict(), head=head.state_dict(), ema=ema.state_dict(),
                optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
                disc=disc.state_dict() if disc is not None else None,
                disc_optimizer=opt_disc.state_dict() if opt_disc is not None else None,
                args=vars(args).copy(), best_fid=best_fid,
                rng_torch=torch.get_rng_state(), rng_numpy=np.random.get_state(),
                rng_python=random.getstate(),
                rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(checkpoint):
    torch.set_rng_state(checkpoint['rng_torch'].cpu())
    np.random.set_state(checkpoint['rng_numpy']); random.setstate(checkpoint['rng_python'])
    if torch.cuda.is_available() and checkpoint.get('rng_cuda') is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in checkpoint['rng_cuda']])


def save_targets(args, vae, head, x, labels, schedule, out):
    with preserved_state(vae, head), torch.no_grad():
        torch.manual_seed(args.seed+31000)
        vae.eval(); head.eval()
        _, _, d = score_and_losses(args, vae, head, x[:16], labels[:16], schedule)
        arrays = dict(x=x[:16].cpu().numpy(), labels=labels[:16].cpu().numpy(),
                      times=d['time'].cpu().numpy(),
                      alpha=d['alpha'].cpu().numpy(), sigma=d['sigma'].cpu().numpy(),
                      packed_input=d['joint_input'].cpu().numpy(),
                      packed_epsilon_target=d['joint_target'].cpu().numpy(),
                      prediction=d['prediction'].cpu().numpy(),
                      reconstruction=d['reconstruction'].cpu().numpy())
        for i in range(2):
            arrays.update({f'z{i+1}':d['latents'][i].cpu().numpy(),
                           f'noisy{i+1}':d['noisy'][i].cpu().numpy(),
                           f'mu{i+1}':d['params'][i][0].cpu().numpy(),
                           f'logvar{i+1}':d['params'][i][1].cpu().numpy(),
                           f'noise{i+1}':d['noises'][i].cpu().numpy(),
                           f'component_score{i+1}':(-d['precision_residuals'][i]).cpu().numpy(),
                           f'epsilon_target{i+1}':d['target_eps'][i].cpu().numpy()})
        np.savez_compressed(out/'csem_target_samples.npz', **arrays)


# ---------------------------------------------------------------------------
# Read-only hypothesis tests. All scores below use epsilon = -sigma * score.
# Never send these diagnostic teachers into the encoder's training objective.
# ---------------------------------------------------------------------------
DIAGNOSTIC_VERSION = 1
DIAGNOSTIC_GUIDE = r'''# Stacked CSEM hypothesis tests

All probes use online VAE, both online/EMA heads, CFG=1, and fixed held-out
images/random streams across epochs. Conditional labels and null label 10 are
tested separately. The recipe and gradients of training are unchanged. Files:
metrics.csv (long format), epoch_*.json (same rows plus config), curves.png, and
epoch_*_probes.npz (individual image/query observations for paired reanalysis;
keys are metric|t=time|conditioning|head, in fixed probe order).
SEs are across images/queries after averaging within-image root replicas, not
across scalar coordinates. These are descriptive fixed-probe SEs, not repeated
training-seed uncertainty. Finite-bank reference SEs exclude reference bias.

## Does q(B|X,A) become q(B|X)?
representation/conditional_symkl_per_dim is half the two directional Gaussian
KLs between q(B|x,A) and q(B|x,A'), for independent roots at the SAME image.
Zero is the decoupled limit (including variance independence). This is NOT MI.
mean_parent_snr and logvar_parent_variance detect mean and variance dependence.
cross_corr_rms and child_offdiag_corr_rms use law-of-total-covariance over root
replicas, including E[diag(v2)]. The root/child cross block and child off-diagonal
block test departure from a diagonal per-image Gaussian; they do not certify
Gaussianity. parent_mu_jac_energy / parent_logvar_jac_energy are Hutchinson
estimates of whitened-root sensitivity, detecting nonlinear dependence which
linear covariance can miss. Aggregate cross_corr_rms is separate: shared X can
correlate A and B even when the per-image joint is diagonal Gaussian.
Finite root samples cause positive covariance-squared and variance noise floors;
compare trajectories and increase --diagnostic-particles for confirmation.

decoder/root_swap_delta and child_swap_delta replace one latent by another
independent draw at the SAME image, holding the other fixed. child_reparent_delta
changes B's conditional moments to A' with the SAME child innovation. These
paired reconstruction-MSE changes measure use/compatibility, not causality of
the training objective. Posterior moment summaries expose variance inflation or
collapse that could mimic independence.

## Which representation pressure favors dependence?
pressure/*_gate_derivative differentiates through a diagnostic scalar g in
mu2(x, mu1+g*(A-mu1)), v2(...), at g=1. The decoder still receives A.
Positive derivative means locally reducing stochastic-parent coupling lowers
that objective, negative means increasing it lowers it. Records include score,
reconstruction MSE, terminal and band derivatives, and their training-weighted
sum. LPIPS/GAN derivatives join that sum when available and active in training.
This is a local intervention, NOT a universal scaling
of every dependency or the full training gradient. score_image_gradient_cosine
uses actual shared encoder parameters and all available image losses; negative
means conflicting gradients.

## Is the learned score accurate, and how large is missing-root supervision?
Each time bin logs raw DSM MSE and corrected-target MSE per block, but their
noise floors differ: NEVER compare raw teacher MSE to corrected MSE as score
errors. excess_* is risk(head,T)-risk(0,T) = E[h^2-2hT]. With a valid teacher,
this equals score error minus true-score energy. Across heads at the SAME
encoder/time it compares score accuracy without an aggregate-density reference.
The paired online_minus_ema_excess has an image-level SE.
raw_vs_corrected_excess_gap should average to zero (teacher calibration).
The analytic root correction is sigma*J*G, J=alpha*v1/(alpha^2*v1+sigma^2),
G=partial_A log q2,t(Y2|X,A) with Y2 and image features FIXED; both mean and
variance derivatives are retained. Ccorr1=C1-sigma*J*G, Ccorr2=C2.
correction_energy is an upper bound on population squared missing-field bias
by Jensen; it is NOT itself the squared bias. mean/variance contributions and
their cross term are logged. A large pathwise correction can average away.
teacher_defect_witness_* estimates E[(C1-eps1) f(Y)] with fixed bounded random
sine witnesses f depending on the full noisy input. A nonzero mean relative to
its SE witnesses root-target bias; zero for finitely many witnesses is inconclusive.
head_residual_witness_* replaces C1 by head-Ccorr1: necessary calibration only.
curl_* tests the epsilon field (divide energies by sigma^2 for score units).
Small curl alone cannot establish score accuracy.

## Numerical aggregate-score reference (reference/*)
Independent TRAIN-image bank, held-out test queries (never insert the query's
generating image). For each query, sample A|x,Y1 analytically and weight by
q1,t(Y1|x)*q2,t(Y2|x,A). Average corrected/root and valid child teachers for the
finite-bank reference; average naive root teacher separately. Class-conditioned
audits restrict the image bank to the query's class; unconditional uses all.
Two independent root-particle panels, nested half-image/half-particle checks,
image/joint ESS and max weight diagnose resolution. reference_screen_pass needs
all joint/root/child relative RMS changes < --reference-tolerance and minimum
query image ESS >= 2;
this is only a heuristic screen, NOT certification of full-population scores.
head_reference_mse, stack_reference_mse, root_correction_gain and relative RMS
remain explicitly finite-bank estimates even when this screen passes. Gain is
undefined when correction energy < 1e-12. Its separate correction_screen_pass
also requires the correction itself (not merely the larger score) to resolve.
Increase image bank AND particles;
particles cannot fix missing image coverage. Report all times, including failures.

The attached report may predate its proposed section 8. These tests address its
stated representation-coadaptation hypothesis; no mechanism is presumed proven.
'''


def _flat(x):
    return x.flatten(1)


def _per_image(x):
    return _flat(x).mean(1)


class DiagnosticLog:
    def __init__(self, epoch):
        self.epoch, self.rows, self.observations = epoch, [], {}

    def add(self, metric, values, *, time_value='', conditioning='all', head='none', unit=''):
        values = torch.as_tensor(values).detach().double().flatten().cpu()
        # Undefined ratios remain explicit nulls, never disguised as zero.
        n = len(values)
        valid = bool(n and torch.isfinite(values).all())
        key = f'{metric}|t={time_value}|{conditioning}|{head}'
        self.observations.setdefault(key, []).append(values.numpy().copy())
        self.rows.append(dict(epoch=self.epoch, time=time_value, conditioning=conditioning,
                              head=head, metric=metric,
                              mean=float(values.mean()) if valid else None,
                              se=float(values.std(unbiased=True)/math.sqrt(n)) if valid and n > 1 else None,
                              n=n, unit=unit))


def diagnostic_subset(dataset, count, seed):
    """Sample without replacement, without iterating/advancing training loader."""
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=gen)[:min(count, len(dataset))].tolist()
    pairs = [dataset[i] for i in indices]
    return torch.stack([p[0] for p in pairs]), torch.tensor([int(p[1]) for p in pairs]), indices


def prepare_diagnostics(args, train_loader, test_loader, vae, head):
    if not (args.diagnostic_every or args.reference_every):
        return None
    with preserved_state(vae, head):
        seed_everything(args.seed+44000)
        x, y, indices = diagnostic_subset(test_loader.dataset, args.diagnostic_samples, args.seed+44001)
        bank, by, bi = diagnostic_subset(train_loader.dataset, args.reference_bank, args.seed+44002)
    times = (np.geomspace(args.t_min, args.t_max, 8).tolist() if args.diagnostic_times == 'auto'
             else [float(v) for v in args.diagnostic_times.split(',')])
    return dict(x=x, labels=y, bank=bank, bank_labels=by, indices=indices,
                bank_indices=bi, times=times)


def analytic_root_correction(vae, features, a, y2, mu1, lv1, alpha, sigma):
    """Exact single-chain valid teacher correction; partial derivative, not total.

    No parameter .grad is written, no second derivatives, and no graph escapes.
    Split G into mean/variance derivatives, using detached likelihood coefficients.
    """
    with torch.enable_grad():
        root = a.detach().requires_grad_(True)
        mu2, lv2 = vae.encode_conditional(features.detach(), root)
        v2 = lv2.exp()
        s2 = alpha**2*v2 + sigma**2
        r2 = y2.detach()-alpha*mu2.detach()
        cm = (alpha*r2/s2.detach()).detach()
        cv = (.5*alpha**2*(r2.square()/s2.detach().square()-1/s2.detach())).detach()
        gm = torch.autograd.grad((cm*mu2).sum(), root, retain_graph=True)[0]
        gv = torch.autograd.grad((cv*v2).sum(), root)[0]
    j = alpha*lv1.exp()/(alpha**2*lv1.exp()+sigma**2)
    return (sigma*j*gm).detach(), (sigma*j*gv).detach()


def _symkl(mu, lv, other_mu, other_lv):
    # Half Jeffreys, per coordinate; exact zero iff the two diagonal Gaussians agree.
    delta = (mu-other_mu).square()
    return .25*((lv-other_lv).exp()+(other_lv-lv).exp()-2
                +delta*(torch.exp(-lv)+torch.exp(-other_lv)))


def representation_diagnostics(args, vae, data, device, log):
    k = args.diagnostic_particles
    aa, bb = [], []
    for start in range(0, len(data['x']), args.diagnostic_batch):
        x = data['x'][start:start+args.diagnostic_batch].to(device)
        n = len(x)
        with torch.no_grad():
            m1, l1, h = vae.encode_root(x)
            a = m1[:, None]+torch.exp(.5*l1[:, None])*torch.randn(n, k, *m1.shape[1:], device=device)
            hrep = h[:, None].expand(-1, k, *h.shape[1:]).reshape(n*k, *h.shape[1:])
            m2, l2 = vae.encode_conditional(hrep, a.flatten(0, 1))
            m2, l2 = m2.reshape(n, k, *m2.shape[1:]), l2.reshape(n, k, *l2.shape[1:])
            b = m2+torch.exp(.5*l2)*torch.randn_like(m2)
            m = m2.flatten(2).double(); v = l2.exp().flatten(2).double()
            ac = a.flatten(2).double(); ac = ac-ac.mean(1, keepdim=True)
            mc = m-m.mean(1, keepdim=True)
            cm = mc.transpose(1, 2)@mc/(k-1)
            av = ac.square().sum(1)/(k-1)
            bv = cm.diagonal(dim1=1, dim2=2)+v.mean(1)
            cross = ac.transpose(1, 2)@mc/(k-1)
            cross = cross/(av[:, :, None]*bv[:, None, :]).clamp_min(1e-24).sqrt()
            corr2 = cm/(bv[:, :, None]*bv[:, None, :]).clamp_min(1e-24).sqrt()
            d2 = m.shape[-1]
            off = corr2.square().sum((1, 2))-corr2.diagonal(dim1=1, dim2=2).square().sum(1)
            metrics = {
                'conditional_symkl_per_dim': _symkl(m2[:, 0::2], l2[:, 0::2], m2[:, 1::2], l2[:, 1::2]).flatten(1).mean(1),
                'mean_parent_snr': (cm.diagonal(dim1=1, dim2=2)/v.mean(1).clamp_min(1e-20)).mean(1),
                'logvar_parent_variance': l2.flatten(2).var(1, unbiased=True).mean(1),
                'cross_corr_rms': cross.square().mean((1, 2)).sqrt(),
                'child_offdiag_corr_rms': (off.clamp_min(0)/(d2*(d2-1))).sqrt(),
                'root_var': _per_image(l1.exp()), 'child_var': v.mean((1, 2)),
                'root_mean_energy': _per_image(m1.square()), 'child_mean_energy': m.square().mean((1, 2)),
            }
            for name, value in metrics.items():
                log.add('representation/'+name, value)
            aa.append(a[:, 0].cpu()); bb.append(b[:, 0].cpu())
            t = torch.full((n,), args.decode_time, device=device)
            baseline = _per_image((vae.decode(a[:, 0], b[:, 0], t=t)-x).square())
            # Hold child innovation fixed while changing its parent moments.
            innovation = (b[:, 0]-m2[:, 0])*torch.exp(-.5*l2[:, 0])
            reparent = m2[:, 1]+torch.exp(.5*l2[:, 1])*innovation
            for name, r, c in [('root_swap', a[:, 1], b[:, 0]),
                               ('child_swap', a[:, 0], b[:, 1]),
                               ('child_reparent', a[:, 0], reparent)]:
                delta = _per_image((vae.decode(r, c, t=t)-x).square())-baseline
                log.add('decoder/'+name+'_delta', delta)
            log.add('decoder/posterior_mse', baseline)
        # Whitened local Jacobian norms: E||J_mu diag(sd1)||F^2 / sum(v2).
        with torch.enable_grad():
            r = a[:, 0].detach().requires_grad_(True)
            mu, lv = vae.encode_conditional(h.detach(), r)
            vm = torch.empty_like(mu).bernoulli_(.5).mul_(2).sub_(1)
            vl = torch.empty_like(lv).bernoulli_(.5).mul_(2).sub_(1)
            jm = torch.autograd.grad((mu*vm).sum(), r, retain_graph=True)[0]
            jl = torch.autograd.grad((lv*vl).sum(), r)[0]
        log.add('representation/parent_mu_jac_energy',
                _flat(jm.square()*l1.exp()).sum(1)/_flat(lv.detach().exp()).sum(1).clamp_min(1e-20))
        log.add('representation/parent_logvar_jac_energy',
                _flat(jl.square()*l1.exp()).sum(1)/lv[0].numel())
    a, b = _flat(torch.cat(aa)).double(), _flat(torch.cat(bb)).double()
    a, b = a-a.mean(0), b-b.mean(0)
    corr = (a.T@b)/(a.square().sum(0)[:, None]*b.square().sum(0)[None, :]).clamp_min(1e-24).sqrt()
    log.add('aggregate/cross_corr_rms', corr.square().mean().sqrt().reshape(1))


def pressure_diagnostics(args, vae, head, data, device, log, lpips_net=None, disc=None, gan_active=False):
    """Local coupling intervention and actual encoder-gradient conflict."""
    with torch.enable_grad():
        x = data['x'][:args.diagnostic_batch].to(device)
        labels = data['labels'][:len(x)].to(device)
        m1, l1, h = vae.encode_root(x)
        a = m1+torch.exp(.5*l1)*torch.randn_like(m1)
        gate = x.new_tensor(1., requires_grad=True)
        m2, l2 = vae.encode_conditional(h, m1+gate*(a-m1))
        b = m2+torch.exp(.5*l2)*torch.randn_like(m2)
        t = torch.as_tensor(data['times'], device=device, dtype=x.dtype)
        t = t[torch.linspace(0, len(t)-1, len(x), device=device).long()]
        al = torch.exp(-t)[:, None, None, None]; si = torch.sqrt(-torch.expm1(-2*t))[:, None, None, None]
        y1, y2 = al*a+si*torch.randn_like(a), al*b+si*torch.randn_like(b)
        c1 = si*(y1-al*m1)/(al.square()*l1.exp()+si.square())
        c2 = si*(y2-al*m2)/(al.square()*l2.exp()+si.square())
        score = F.mse_loss(head(pack_blocks(y1, y2), t, labels), pack_blocks(c1, c2))
        image = vae.decode(y1, y2, t=t)
        recon = F.mse_loss(image, x)
        w1 = a[0].numel()/(a[0].numel()+b[0].numel())
        at, st = math.exp(-args.t_max), math.sqrt(-math.expm1(-2*args.t_max))
        terminal = w1*terminal_kl(m1, l1, at, st)+(1-w1)*terminal_kl(m2, l2, at, st)
        schedule = ref.make_schedule(dict(time_schedule='log_t', t_min=args.t_min,
                                         t_max=args.t_max, num_train_timesteps=1000), device)
        floor = tmin_band_floor(schedule)
        band = w1*tmin_band_penalty(l1, floor)+(1-w1)*tmin_band_penalty(l2, floor)
        objectives = [('score', score), ('recon', recon), ('terminal', terminal), ('band', band)]
        image_objective = recon
        if lpips_net is not None:
            perc = lpips_net(image.repeat(1,3,1,1), x.repeat(1,3,1,1)).mean()
            objectives.append(('lpips', perc)); image_objective = image_objective+args.perc_weight*perc
        if disc is not None and gan_active:
            gan = ref.hinge_g_loss(disc(image))
            objectives.append(('gan', gan)); image_objective = image_objective+args.gan_weight*gan
        total = args.score_weight*score+image_objective+args.terminal_weight*terminal+tmin_band_coefficient(args, schedule)*band
        objectives.append(('weighted_total', total))
        for name, loss in objectives:
            derivative = torch.autograd.grad(loss, gate, retain_graph=True, allow_unused=True)[0]
            log.add('pressure/'+name+'_gate_derivative', (derivative if derivative is not None else gate*0).reshape(1))
        params = [p for name, p in vae.named_parameters()
                  if p.requires_grad and name.startswith(('enc_', 'root_posterior.', 'condition.'))]
        gs = torch.autograd.grad(score, params, retain_graph=True, allow_unused=True)
        gr = torch.autograd.grad(image_objective, params, allow_unused=True)
        dot = sum((u*v).sum() for u, v in zip(gs, gr) if u is not None and v is not None)
        ns = sum(u.square().sum() for u in gs if u is not None)
        nr = sum(v.square().sum() for v in gr if v is not None)
        log.add('pressure/score_image_gradient_cosine', (dot/(ns*nr).sqrt().clamp_min(1e-30)).reshape(1))


def curl_diagnostics(args, head, packed, times, labels):
    """Two independent Rademacher probes of antisymmetric Jacobian energy."""
    with torch.enable_grad():
        point = packed.detach().requires_grad_(True)
        pred = head(point, times, labels)
        u = torch.empty_like(point).bernoulli_(.5).mul_(2).sub_(1)
        v = torch.empty_like(point).bernoulli_(.5).mul_(2).sub_(1)
        ju = torch.autograd.grad((pred*u).sum(), point, retain_graph=True)[0]
        jv = torch.autograd.grad((pred*v).sum(), point, retain_graph=True)[0]
        nroot = args.z1_channels//4
        ur = u.clone(); ur[:, nroot:] = 0
        vc = v.clone(); vc[:, :nroot] = 0
        jr = torch.autograd.grad((pred*ur).sum(), point, retain_graph=True)[0]
        jc = torch.autograd.grad((pred*vc).sum(), point)[0]
    dim = point[0].numel()
    return dict(curl_all_energy=(_flat(ju*v-jv*u).sum(1)).square()/dim,
                curl_cross_energy=(_flat(jr*vc-jc*ur).sum(1)).square()/dim,
                jacobian_energy=_flat(ju.square()).sum(1)/dim)


def score_diagnostics(args, vae, heads, data, device, log):
    for tv in data['times']:
        al, si = math.exp(-tv), math.sqrt(-math.expm1(-2*tv))
        for start in range(0, len(data['x']), args.diagnostic_batch):
            x = data['x'][start:start+args.diagnostic_batch].to(device)
            labels = data['labels'][start:start+len(x)].to(device)
            with torch.no_grad():
                m1, l1, features = vae.encode_root(x)
                a = m1+torch.exp(.5*l1)*torch.randn_like(m1)
                m2, l2 = vae.encode_conditional(features, a)
                b = m2+torch.exp(.5*l2)*torch.randn_like(m2)
                e1, e2 = torch.randn_like(a), torch.randn_like(b)
                y1, y2 = al*a+si*e1, al*b+si*e2
                c1 = si*(y1-al*m1)/(al**2*l1.exp()+si**2)
                c2 = si*(y2-al*m2)/(al**2*l2.exp()+si**2)
            cm, cv = analytic_root_correction(vae, features, a, y2, m1, l1, al, si)
            corr = cm+cv
            valid = [c1-corr, c2]
            packed = pack_blocks(y1, y2)
            ts = torch.full((len(x),), tv, device=device)
            for name, value in [('total', corr.square()), ('mean', cm.square()),
                                ('variance', cv.square()), ('cross', 2*cm*cv)]:
                log.add('score/correction_'+name+'_energy', _per_image(value), time_value=tv)
            for name, teacher in [('naive_root', c1), ('corrected_root', valid[0]), ('child', c2)]:
                log.add('score/'+name+'_target_energy', _per_image(teacher.square()), time_value=tv)
            # Fixed bounded test functions see both noisy blocks. No density bank.
            gen = torch.Generator(device=device).manual_seed(args.seed+44008)
            directions = torch.randn(4, packed[0].numel(), generator=gen, device=device)/math.sqrt(packed[0].numel())
            root_directions = torch.randn(4, a[0].numel(), generator=gen, device=device)/math.sqrt(a[0].numel())
            f = torch.sin(_flat(packed)@directions.T)
            for w in range(4):
                witness = ((_flat(c1-e1)*root_directions[w]).sum(1))*f[:, w]
                log.add('score/teacher_defect_witness_'+str(w), witness, time_value=tv)
            for cond, lab in [('class', labels), ('null', torch.full_like(labels, 10))]:
                predictions = {}
                for hn, head in heads.items():
                    with torch.no_grad():
                        predictions[hn] = unpack_blocks(head(packed, ts, lab), args.z1_channels)
                    for i, (pred, raw, naive, correct) in enumerate(zip(predictions[hn], [e1, e2], [c1, c2], valid), 1):
                        metrics = dict(dsm_mse=_per_image((pred-raw).square()),
                                       corrected_mse=_per_image((pred-correct).square()),
                                       stacked_mse=_per_image((pred-naive).square()),
                                       excess_dsm=_per_image(pred.square()-2*pred*raw),
                                       excess_corrected=_per_image(pred.square()-2*pred*correct),
                                       raw_vs_corrected_excess_gap=_per_image(2*pred*(correct-raw)),
                                       prediction_energy=_per_image(pred.square()))
                        for name, value in metrics.items():
                            log.add(f'score/block{i}/'+name, value, time_value=tv, conditioning=cond, head=hn)
                    for w in range(4):
                        witness = (_flat(predictions[hn][0]-valid[0])*root_directions[w]).sum(1)*f[:, w]
                        log.add('score/head_residual_witness_'+str(w), witness, time_value=tv, conditioning=cond, head=hn)
                    # Derivative cost limited to first minibatch at each time.
                    if start == 0:
                        for name, value in curl_diagnostics(args, head, packed, ts, lab).items():
                            log.add('score/'+name, value, time_value=tv, conditioning=cond, head=hn)
                if 'online' in predictions and 'ema' in predictions:
                    for i in range(2):
                        p, q, c = predictions['online'][i], predictions['ema'][i], valid[i]
                        log.add(f'score/block{i+1}/online_minus_ema_excess',
                                _per_image(p.square()-q.square()-2*(p-q)*c), time_value=tv, conditioning=cond)


def _finish_records(log):
    """Combine minibatches with exact pooled sample variance, not mean-of-SEs."""
    grouped = {}
    for r in log.rows:
        key = tuple(r[k] for k in ('epoch', 'time', 'conditioning', 'head', 'metric', 'unit'))
        grouped.setdefault(key, []).append(r)
    rows = []
    for group in grouped.values():
        row = group[0].copy(); n = sum(r['n'] for r in group)
        if any(r['mean'] is None for r in group):
            row.update(mean=None, se=None, n=n)
        else:
            mean = sum(r['n']*r['mean'] for r in group)/n
            ss = sum((r['se'] or 0)**2*r['n']*max(0, r['n']-1)+r['n']*(r['mean']-mean)**2 for r in group)
            row.update(mean=mean, se=math.sqrt(max(0., ss)/(n-1)/n) if n > 1 else None, n=n)
        rows.append(row)
    return rows


def finite_bank_reference(args, vae, bank, bank_labels, y1, y2, label, tv):
    """One query, one panel. Keep X and observed Y2 fixed in likelihood gradient.

    Log weights and reduction are float64; encoder remains in its native dtype.
    Chunking bounds activation memory by --diagnostic-batch * root particles.
    """
    al, si = math.exp(-tv), math.sqrt(-math.expm1(-2*tv))
    select = torch.ones(len(bank), dtype=torch.bool) if label == 10 else bank_labels == label
    images = bank[select]
    if len(images) < 4:
        return None
    k, device = args.reference_particles, y1.device
    weights, teachers = [], []
    for start in range(0, len(images), args.diagnostic_batch):
        x = images[start:start+args.diagnostic_batch].to(device)
        n = len(x)
        with torch.no_grad():
            m1, l1, feat = vae.encode_root(x)
            v1 = l1.exp(); s1 = al**2*v1+si**2
            post_m = m1+al*v1/s1*(y1-al*m1)
            post_v = v1*si**2/s1
            a = post_m[:, None]+post_v[:, None].sqrt()*torch.randn(n, k, *m1.shape[1:], device=device)
            h = feat[:, None].expand(-1, k, *feat.shape[1:]).reshape(n*k, *feat.shape[1:])
            m1r = m1[:, None].expand(-1, k, *m1.shape[1:]).flatten(0, 1)
            l1r = l1[:, None].expand(-1, k, *l1.shape[1:]).flatten(0, 1)
            m2, l2 = vae.encode_conditional(h, a.flatten(0, 1))
            s2 = al**2*l2.exp()+si**2
            # Constant log(2*pi) cancels across equal-dimensional components.
            lw1 = -.5*_flat((y1-al*m1).double().square()/s1.double()+s1.double().log()).sum(1)
            lw2 = -.5*_flat((y2-al*m2).double().square()/s2.double()+s2.double().log()).sum(1)
            c1 = si*(y1-al*m1r)/(al**2*l1r.exp()+si**2)
            c2 = si*(y2-al*m2)/s2
        cm, cv = analytic_root_correction(vae, h, a.flatten(0, 1), y2, m1r, l1r, al, si)
        teachers.append(torch.cat([_flat(c1-cm-cv), _flat(c2), _flat(c1)], 1).double().reshape(n, k, -1))
        weights.append(lw1[:, None]+lw2.reshape(n, k))
    lw, vals = torch.cat(weights), torch.cat(teachers)
    def average(w, v):
        w = torch.softmax(w.flatten(), 0).reshape(w.shape)
        return (w[:, :, None]*v).sum((0, 1))
    full = average(lw, vals)
    w = torch.softmax(lw.flatten(), 0).reshape(lw.shape)
    iw = w.sum(1)
    return dict(full=full, half_bank=average(lw[:len(lw)//2], vals[:len(lw)//2]),
                half_particles=average(lw[:, :k//2], vals[:, :k//2]),
                image_ess=float(1/iw.square().sum()), joint_ess=float(1/w.square().sum()),
                max_weight=float(w.max()), bank_images=len(images))


def reference_diagnostics(args, vae, heads, data, device, log):
    n = min(args.reference_queries, len(data['x']))
    for tv in data['times']:
        al, si = math.exp(-tv), math.sqrt(-math.expm1(-2*tv))
        with torch.no_grad():
            x = data['x'][:n].to(device); labels = data['labels'][:n].to(device)
            (a, b), _ = vae.encode_sample(x)
            y1, y2 = al*a+si*torch.randn_like(a), al*b+si*torch.randn_like(b)
        for cond in ('class', 'null'):
            labs = labels if cond == 'class' else torch.full_like(labels, 10)
            bucket = {key: [] for key in ('repeat', 'half_bank', 'half_particles', 'image_ess', 'joint_ess', 'max_weight', 'bank_images')}
            cbucket = {key: [] for key in ('repeat', 'half_bank', 'half_particles')}
            refs, naive, good_indices = [], [], []
            d1, d2 = a[0].numel(), b[0].numel()
            for i in range(n):
                panels = [finite_bank_reference(args, vae, data['bank'], data['bank_labels'], y1[i:i+1], y2[i:i+1], int(labs[i]), tv) for _ in range(2)]
                if any(p is None for p in panels):
                    continue
                p, q = panels
                full = .5*(p['full']+q['full'])
                refs.append(full[:d1+d2]); naive.append(full[d1+d2:]); good_indices.append(i)
                correction = full[:d1]-full[d1+d2:]
                pcor = p['full'][:d1]-p['full'][d1+d2:]
                qcor = q['full'][:d1]-q['full'][d1+d2:]
                cbucket['repeat'].append((pcor-qcor).square().mean())
                bucket['repeat'].append((p['full'][:d1+d2]-q['full'][:d1+d2]).square())
                for key in ('half_bank', 'half_particles'):
                    small = .5*(p[key]+q[key])
                    bucket[key].append((small[:d1+d2]-full[:d1+d2]).square())
                    cbucket[key].append((small[:d1]-small[d1+d2:]-correction).square().mean())
                for key in ('image_ess', 'joint_ess', 'max_weight', 'bank_images'):
                    bucket[key].append(.5*(p[key]+q[key]))
            log.add('reference/queries_resolved', [len(refs)], time_value=tv, conditioning=cond)
            if not refs:
                log.add('reference/reference_screen_pass', [0.], time_value=tv, conditioning=cond)
                continue
            true, stack = torch.stack(refs), torch.stack(naive)
            changes = []
            for key in ('repeat', 'half_bank', 'half_particles'):
                for block, sl in [('joint', slice(None)), ('root', slice(0, d1)), ('child', slice(d1, d1+d2))]:
                    ratio = torch.stack(bucket[key])[:, sl].mean().div(true[:, sl].square().mean().clamp_min(1e-24)).sqrt()
                    changes.append(float(ratio))
                    log.add('reference/'+block+'_'+key+'_relative_rms', ratio.reshape(1), time_value=tv, conditioning=cond)
            for key in ('image_ess', 'joint_ess', 'max_weight', 'bank_images'):
                log.add('reference/'+key, bucket[key], time_value=tv, conditioning=cond)
            log.add('reference/min_image_ess', [min(bucket['image_ess'])], time_value=tv, conditioning=cond)
            log.add('reference/fraction_image_ess_below2', [np.mean(np.asarray(bucket['image_ess']) < 2)], time_value=tv, conditioning=cond)
            passed = max(changes) < args.reference_tolerance and min(bucket['image_ess']) >= 2 and len(refs) == n
            log.add('reference/reference_screen_pass', [float(passed)], time_value=tv, conditioning=cond)
            missing = true[:, :d1]-stack
            correction_energy = missing.square().mean()
            correction_changes = []
            for key in cbucket:
                ratio = (torch.stack(cbucket[key]).mean()/correction_energy).sqrt() if correction_energy > 1e-12 else correction_energy.new_tensor(float('nan'))
                correction_changes.append(float(ratio))
                log.add('reference/correction_'+key+'_relative_rms', ratio.reshape(1), time_value=tv, conditioning=cond)
            cpass = passed and all(math.isfinite(v) and v < args.reference_tolerance for v in correction_changes)
            log.add('reference/correction_screen_pass', [float(cpass)], time_value=tv, conditioning=cond)
            log.add('reference/stack_root_mse', missing.square().mean(1), time_value=tv, conditioning=cond)
            log.add('reference/stack_root_relative_rms',
                    (missing.square().mean()/true[:, :d1].square().mean().clamp_min(1e-24)).sqrt().reshape(1),
                    time_value=tv, conditioning=cond)
            for hn, head in heads.items():
                with torch.no_grad():
                    # Packed root has a pixel permutation; compare in unpacked order.
                    pp = unpack_blocks(head(pack_blocks(y1, y2), torch.full((n,), tv, device=device), labs), args.z1_channels)
                    pred = torch.cat([_flat(pp[0]), _flat(pp[1])], 1)[good_indices].double()
                for block, sl in [('root', slice(0, d1)), ('child', slice(d1, d1+d2)), ('joint', slice(None))]:
                    err = (pred[:, sl]-true[:, sl]).square().mean(1)
                    log.add('reference/'+block+'_head_mse', err, time_value=tv, conditioning=cond, head=hn)
                    log.add('reference/'+block+'_head_relative_rms',
                            (err.mean()/true[:, sl].square().mean().clamp_min(1e-24)).sqrt().reshape(1),
                            time_value=tv, conditioning=cond, head=hn)
                energy = missing.square().mean()
                log.add('reference/head_minus_stack_root_mse', (pred[:, :d1]-stack).square().mean(1),
                        time_value=tv, conditioning=cond, head=hn)
                gain = ((pred[:, :d1]-stack)*missing).mean()/energy if energy > 1e-12 else energy.new_tensor(float('nan'))
                log.add('reference/root_correction_gain', gain.reshape(1), time_value=tv, conditioning=cond, head=hn)


def plot_diagnostics(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (path/'metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    specs = [
        ('representation/conditional_symkl_per_dim', 'Per-image conditional dependence'),
        ('representation/parent_mu_jac_energy', 'Whitened mean Jacobian energy'),
        ('representation/child_offdiag_corr_rms', 'Per-image child off-diagonal correlation'),
        ('score/correction_total_energy', 'Pathwise correction energy (bias upper bound)'),
        ('score/block1/excess_corrected', 'Root score excess risk (moving encoder)'),
        ('reference/root_head_relative_rms', 'Finite-bank root error (check resolution)'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (metric, title) in zip(axes.flat, specs):
        groups = {}
        for r in rows:
            if r['metric'] != metric or not r['mean'] or r['conditioning'] not in ('all', 'class') or r['head'] not in ('none', 'ema'):
                continue
            groups.setdefault(r['time'], []).append(r)
        for tv, group in groups.items():
            ax.plot([int(r['epoch']) for r in group], [float(r['mean']) for r in group],
                    marker='.', label=f't={float(tv):.3g}' if tv else 'representation')
        ax.set(title=title, xlabel='Epoch'); ax.grid(alpha=.2)
        if groups: ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path/'curves.png', dpi=140); plt.close(fig)


def evaluate_diagnostics(args, epoch, vae, head, ema, data, device, out, *, run_core=True, run_reference=False,
                         lpips_net=None, disc=None):
    if data is None:
        return
    path = out/'diagnostics'; path.mkdir(exist_ok=True)
    (path/'README.md').write_text(DIAGNOSTIC_GUIDE)
    log = DiagnosticLog(epoch); started = time.time()
    with preserved_state(vae, head, ema, lpips_net, disc):
        vae.eval(); head.eval(); ema.eval()
        if disc is not None: disc.eval()
        seed_everything(args.seed+45000)
        if run_core:
            representation_diagnostics(args, vae, data, device, log)
            pressure_diagnostics(args, vae, head, data, device, log, lpips_net, disc,
                                 gan_active=epoch >= args.disc_start)
            score_diagnostics(args, vae, {'online': head, 'ema': ema}, data, device, log)
        if run_reference:
            seed_everything(args.seed+46000)
            reference_diagnostics(args, vae, {'online': head, 'ema': ema}, data, device, log)
    rows = _finish_records(log)
    # Replace matching epoch/metric keys when re-evaluating or resuming, so an
    # at-start audit does not silently double-weight that epoch in the plots.
    keys = ('epoch', 'time', 'conditioning', 'head', 'metric')
    identity = lambda r: tuple(str(r[k]) for k in keys)
    updates = {identity(r) for r in rows}
    history = []
    if (path/'metrics.csv').exists():
        with (path/'metrics.csv').open() as f:
            history = [r for r in csv.DictReader(f) if identity(r) not in updates]
    temporary = path/'metrics.csv.tmp'
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(history+rows)
    os.replace(temporary, path/'metrics.csv')
    payload = dict(version=DIAGNOSTIC_VERSION, epoch=epoch, seconds=time.time()-started,
                   run_core=run_core, run_reference=run_reference, test_indices=data['indices'],
                   train_bank_indices=data['bank_indices'], times=data['times'],
                   settings={k: v for k, v in vars(args).items() if k.startswith(('diagnostic_', 'reference_'))}, rows=rows)
    report = path/f'epoch_{epoch:04d}.json'
    if report.exists():
        previous = json.loads(report.read_text())
        payload['rows'] = [r for r in previous['rows'] if identity(r) not in updates]+rows
    report.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    probes = path/f'epoch_{epoch:04d}_probes.npz'
    arrays = {}
    if probes.exists():
        with np.load(probes, allow_pickle=False) as prior:
            arrays = {key: prior[key] for key in prior.files}
    arrays.update({key: np.concatenate(value) for key, value in log.observations.items()})
    np.savez_compressed(probes, **arrays)
    plot_diagnostics(path)
    selected = {r['metric']: r['mean'] for r in rows if r['time'] == ''}
    print('[DIAGNOSTICS]', json.dumps(dict(epoch=epoch, seconds=round(payload['seconds'], 2),
          core=run_core, reference=run_reference, metrics=len(rows), **selected)), flush=True)


def train(args, checkpoint=None):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if args.cpu_threads: torch.set_num_threads(args.cpu_threads)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    seed_everything(args.seed)
    if not args.resume and not args.eval_only and (out/'latest.pt').exists():
        raise FileExistsError('Output already has latest.pt. Use --resume or choose a new --output.')
    if args.synthetic_smoke_test:
        # Only software plumbing: no real-data FID and no claim of model quality.
        generator = torch.Generator().manual_seed(1700)
        dataset = torch.utils.data.TensorDataset(torch.rand(8,1,32,32,generator=generator)*2-1, torch.arange(8)%10)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch)
    else:
        train_loader, test_loader, _ = ref.make_dataloaders(args.batch, args.num_workers, 'FMNIST')
    fid_info = prepare_fid(args, test_loader, device, out)
    # Original order: feature preparation must not alter model initialization.
    seed_everything(args.seed)
    vae, head = build(args, device)
    ema = copy.deepcopy(head).eval().requires_grad_(False)
    schedule = ref.make_schedule(dict(time_schedule='log_t', t_min=args.t_min, t_max=args.t_max,
                                      num_train_timesteps=1000), device)
    band_weight = tmin_band_coefficient(args, schedule)
    opt = AdamW([dict(params=vae.parameters(), lr=args.lr_vae),
                 dict(params=head.parameters(), lr=args.lr_head/args.score_weight)],
                weight_decay=1e-4, betas=(.9,.95))
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    disc = ref.PatchDiscriminator(in_channels=1, ndf=64, n_layers=2).to(device) if args.gan_weight else None
    opt_disc = AdamW(disc.parameters(), lr=args.lr_disc, betas=(.5,.9), weight_decay=0) if disc is not None else None
    if args.perc_weight and not ref.LPIPS_AVAILABLE:
        raise RuntimeError('Original recipe needs lpips: pip install lpips. Use --no-lpips only for a separate ablation or smoke test.')
    lpips_net = ref.lpips.LPIPS(net='vgg').to(device).eval().requires_grad_(False) if args.perc_weight else None
    first, best_fid = 0, float('inf')
    if checkpoint is not None:
        vae.load_state_dict(checkpoint['vae']); head.load_state_dict(checkpoint['head'])
        ema.load_state_dict(checkpoint['ema'])
        if not args.eval_only:
            opt.load_state_dict(checkpoint['optimizer']); scheduler.load_state_dict(checkpoint['scheduler'])
            if disc is not None:
                if checkpoint.get('disc') is None: raise ValueError('Resume discriminator configuration differs from checkpoint')
                disc.load_state_dict(checkpoint['disc']); opt_disc.load_state_dict(checkpoint['disc_optimizer'])
        first = int(checkpoint['epoch']); best_fid = float(checkpoint.get('best_fid', float('inf')))
        restore_rng(checkpoint)
    config = vars(args).copy()
    config.update(architecture_version=ARCHITECTURE_VERSION,
                  diagnostic_version=DIAGNOSTIC_VERSION,
                  posterior_chain='q(z1|x) q(z2|x,z1), one shared sampled z1',
                  head_inputs='packed(y1_t,y2_t),t,dropped_class_label',
                  target='packed(sigma*(y1-alpha*mu1)/S1, sigma*(y2-alpha*mu2(x,z1))/S2)',
                  sampling='one joint RK4 solve for both coordinates',
                  vae_parameters=sum(p.numel() for p in vae.parameters()),
                  head_parameters=sum(p.numel() for p in head.parameters()),
                  band_weight_resolved=band_weight,
                  source_baseline_sha256=BASELINE_SHA256,
                  original_recipe='online VAE + EMA head; all-time LPIPS/GAN; joint clipping; constant disc LR')
    (out/('eval_config.json' if args.eval_only else 'config.json')).write_text(json.dumps(config, indent=2)+'\n')
    print('[CONFIG]', json.dumps(config), flush=True)
    diagnostic_data = prepare_diagnostics(args, train_loader, test_loader, vae, head)
    if args.eval_only:
        if fid_info is None and diagnostic_data is None:
            raise ValueError('--eval-only requires FID or hypothesis diagnostics')
        if diagnostic_data is not None:
            evaluate_diagnostics(args, first, vae, head, ema, diagnostic_data, device, out,
                                 run_core=bool(args.diagnostic_every), run_reference=bool(args.reference_every),
                                 lpips_net=lpips_net, disc=disc)
        if fid_info is not None:
            return evaluate(args, first, vae, ema, fid_info, device, out)
        return
    if diagnostic_data is not None and args.diagnostic_at_start:
        evaluate_diagnostics(args, first, vae, head, ema, diagnostic_data, device, out,
                             run_core=bool(args.diagnostic_every), run_reference=bool(args.reference_every),
                             lpips_net=lpips_net, disc=disc)
    last_batch = None
    for epoch in range(first, args.epochs):
        vae.train(); head.train()
        if disc is not None: disc.train()
        start_time, count, sums = time.time(), 0, {}
        for iteration, (x, labels) in enumerate(train_loader):
            if args.max_batches and iteration >= args.max_batches: break
            x, labels = x.to(device), labels.to(device)
            last_batch = (x, labels)
            score, terminal, d = score_and_losses(args, vae, head, x, labels, schedule)
            recon = F.mse_loss(d['reconstruction'], x)
            perc = (lpips_net(d['reconstruction'].repeat(1,3,1,1), x.repeat(1,3,1,1)).mean()
                    if lpips_net is not None else x.new_zeros(()))
            active_disc = disc is not None and epoch+1 >= args.disc_start
            if active_disc:
                opt_disc.zero_grad(set_to_none=True)
                dloss = ref.hinge_d_loss(disc(x), disc(d['reconstruction'].detach()))
                dloss.backward(); opt_disc.step()
                # Freeze discriminator parameters for G only; gradients to images remain.
                disc.requires_grad_(False)
                gloss = ref.hinge_g_loss(disc(d['reconstruction']))
                disc.requires_grad_(True)
            else:
                dloss = x.new_zeros(()); gloss = x.new_zeros(())
            loss = (recon + args.perc_weight*perc + args.score_weight*score
                    + args.terminal_weight*terminal + band_weight*d['band_loss'] + args.gan_weight*gloss)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # The original recipe clips VAE and prior together.
            norm = nn.utils.clip_grad_norm_(list(vae.parameters())+list(head.parameters()), 1.0)
            if not torch.isfinite(norm): raise FloatingPointError('Non-finite gradient norm')
            opt.step(); ema_update(head, ema, args.ema)
            values = dict(total=loss, recon=recon, score=score, terminal=terminal,
                          band=d['band_loss'], lpips=perc, gan_d=dloss, gan_g=gloss,
                          gradient_norm=norm)
            for i in range(2):
                values.update({f'score_{i+1}':d['score_per_block'][i],
                               f'kl_{i+1}':d['kl_per_block'][i],
                               f'var_{i+1}':d['params'][i][1].exp().mean(),
                               f'band_{i+1}':d['band_per_block'][i]})
            for name, value in values.items(): sums[name] = sums.get(name, 0.0)+float(value.detach())
            count += 1
        if count == 0: raise ValueError('Training loader produced zero batches')
        scheduler.step()
        row = dict(epoch=epoch+1, seconds=round(time.time()-start_time,2))
        row.update({name:value/count for name,value in sums.items()})
        write_csv(out/'loss.csv', row); print('[TRAIN]', json.dumps(row), flush=True)
        evaluation = None
        if fid_info is not None and ((epoch+1)%args.eval_every == 0 or epoch+1 == args.epochs):
            evaluation = evaluate(args, epoch+1, vae, ema, fid_info, device, out)
        core_due = bool(args.diagnostic_every and ((epoch+1)%args.diagnostic_every == 0 or epoch+1 == args.epochs))
        reference_due = bool(args.reference_every and ((epoch+1)%args.reference_every == 0 or epoch+1 == args.epochs))
        if core_due or reference_due:
            evaluate_diagnostics(args, epoch+1, vae, head, ema, diagnostic_data, device, out,
                                 run_core=core_due, run_reference=reference_due, lpips_net=lpips_net, disc=disc)
        improved = evaluation is not None and evaluation['fid_generated'] < best_fid
        if improved: best_fid = evaluation['fid_generated']
        if improved or (epoch+1)%args.checkpoint_every == 0 or epoch+1 == args.epochs:
            payload = checkpoint_payload(args, epoch+1, vae, head, ema, opt, scheduler, disc, opt_disc, best_fid)
            atomic_torch_save(payload, out/'latest.pt')
            if improved: atomic_torch_save(payload, out/'best.pt')
            if args.archive_checkpoints: atomic_torch_save(payload, out/f'epoch_{epoch+1:04d}.pt')
    if last_batch is not None: save_targets(args, vae, head, *last_batch, schedule, out)
    plot_history(out)
    print('Done:', out.resolve(), flush=True)


def seed_everything(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)


def self_test():
    """Offline implementation checks; these tests do not assert that FID fails."""
    import tempfile
    torch.set_num_threads(2)
    args = make_parser().parse_args([])
    args.joint_depth, args.joint_width, args.joint_heads = 2, 48, 6
    args.batch, args.rk4_steps, args.label_dropout = 2, 2, 0.
    args.perc_weight, args.gan_weight = 0., 0.
    device = torch.device('cpu')
    seed_everything(42)
    vae, head = build(args, device)
    schedule = ref.make_schedule(dict(time_schedule='log_t', t_min=args.t_min,
                                      t_max=args.t_max, num_train_timesteps=1000), device)
    x, labels = torch.rand(2,1,32,32)*2-1, torch.tensor([2,5])
    observed_roots = []
    original_conditional = vae.encode_conditional
    def observe(image_features, root, top_down=None):
        observed_roots.append(root)
        return original_conditional(image_features, root, top_down)
    vae.encode_conditional = observe
    score, terminal, d = score_and_losses(args, vae, head, x, labels, schedule,
                                         indices=torch.tensor([100,700]))
    vae.encode_conditional = original_conditional
    assert len(observed_roots) == 1 and observed_roots[0] is d['latents'][0]
    assert [tuple(z.shape) for z in d['latents']] == [(2,4,4,4),(2,3,8,8)]
    assert d['joint_input'].shape == (2,4,8,8)
    for original, recovered in zip(d['noisy'], unpack_blocks(d['joint_input'])):
        torch.testing.assert_close(original, recovered, rtol=0, atol=0)
    assert d['reconstruction'].shape == x.shape
    torch.testing.assert_close(score, .25*d['score_per_block'][0]+.75*d['score_per_block'][1])
    torch.testing.assert_close(d['joint_input'].square().sum(), sum(z.square().sum() for z in d['noisy']))
    # Recompute child moments from the SAME root and the same image evidence.
    with torch.no_grad():
        _, _, image_features = vae.encode_root(x)
        mu2, lv2 = vae.encode_conditional(image_features, d['latents'][0])
        torch.testing.assert_close(mu2, d['params'][1][0])
        torch.testing.assert_close(lv2, d['params'][1][1])
    for point, (mu, lv), target in zip(d['noisy'], d['params'], d['target_eps']):
        independent = point.detach().clone().requires_grad_(True)
        variance = d['alpha'].square()*lv.detach().exp()+d['sigma'].square()
        log_density = -.5*((independent-d['alpha']*mu.detach()).square()/variance+variance.log()).sum()
        analytic_score = torch.autograd.grad(log_density, independent)[0]
        torch.testing.assert_close(target.detach(), -d['sigma']*analytic_score)
    # Requested training objective has real gradients into both posterior branches.
    posterior_modules = [vae.root_posterior, vae.condition]
    for name, module in zip(('root_posterior','child_posterior'), posterior_modules):
        gradients = torch.autograd.grad(score, list(module.parameters()), retain_graph=True, allow_unused=True)
        norm = sum(float(g.square().sum()) for g in gradients if g is not None)**.5
        assert math.isfinite(norm) and norm > 0, name
        print('[SELF-TEST] score gradient', name, norm, flush=True)
    # Descendant target alone must backpropagate to the same root sample.
    descendant_gradient = torch.autograd.grad(d['target_eps'][1].square().mean(), d['latents'][0],
                                             retain_graph=True)[0]
    assert descendant_gradient.abs().sum() > 0
    total = F.mse_loss(d['reconstruction'],x)+args.score_weight*score+terminal+tmin_band_coefficient(args,schedule)*d['band_loss']
    total.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    assert vae.child_in.weight.grad is not None and vae.child_in.weight.grad.abs().sum()>0
    # Test nonzero input Jacobians after the reference zero-init output projection.
    with torch.no_grad(): head.final_proj.weight.normal_(std=.02)
    head.zero_grad(set_to_none=True)
    packed = d['joint_input'].detach().requires_grad_(True)
    prediction = head(packed, d['time'], labels)
    lhs = torch.autograd.grad(prediction[:, :1].square().mean(), packed, retain_graph=True)[0]
    rhs = torch.autograd.grad(prediction[:, 1:].square().mean(), packed)[0]
    assert lhs[:,1:].abs().sum()>0 and rhs[:,:1].abs().sum()>0
    # The ONE sampler advances a packed joint tensor and returns both blocks.
    vae.eval(); head.eval()
    z = sample_joint(args, head, 2, device, labels=labels)
    assert [tuple(q.shape) for q in z] == [(2,4,4,4),(2,3,8,8)]
    assert all(torch.isfinite(q).all() for q in z)
    with torch.no_grad(): assert vae.decode(*z, t=torch.full((2,),args.decode_time)).shape == x.shape
    # Evaluation/diagnostics cannot consume training RNG or change modes.
    vae.train(); head.train(); old_rng=torch.get_rng_state().clone()
    with preserved_state(vae,head):
        vae.eval();head.eval();torch.randn(9)
    assert vae.training and head.training and torch.equal(old_rng,torch.get_rng_state())
    # Save/load must preserve head and VAE tensors and optimizer/scheduler state.
    optimizer = AdamW(list(vae.parameters())+list(head.parameters()), lr=1e-4)
    optimizer.step()
    scheduler = CosineAnnealingLR(optimizer,T_max=2)
    ema = copy.deepcopy(head).eval().requires_grad_(False)
    payload = checkpoint_payload(args,1,vae,head,ema,optimizer,scheduler,None,None,float('inf'))
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'latest.pt'; atomic_torch_save(payload,path)
        restored=torch.load(path,map_location='cpu',weights_only=False)
        vae2,head2=build(args,device)
        vae2.load_state_dict(restored['vae']);head2.load_state_dict(restored['head'])
        for a,b in zip(head.state_dict().values(),head2.state_dict().values()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert restored['architecture_version']==ARCHITECTURE_VERSION
    features=torch.randn(7,32,dtype=torch.float64)
    assert fid_numeric(features,features)<1e-8
    assert abs(fid_numeric(features,features+2)-128)<1e-8
    diagnostic_self_test(args, vae, head, x, labels)
    print('[SELF-TEST] PASSED: coherent ancestry, target signs, coordinate weights, full gradients, '
          'cross-block dependence, joint sampling, RNG isolation and checkpoint roundtrip.',flush=True)


def diagnostic_self_test(args, vae, head, x, labels):
    """Analytic counterexamples plus a real-network diagnostic integration check."""
    class ToyConditional(nn.Module):
        def __init__(self, beta=1.3, slope=.2, nonlinear=False):
            super().__init__(); self.beta, self.slope, self.nonlinear = beta, slope, nonlinear
        def encode_conditional(self, features, a):
            return self.beta*(a.sin() if self.nonlinear else a), math.log(.6)+self.slope*a

    # Both mean and heteroscedastic variance derivatives; fixed observed Y2.
    a = torch.tensor([-.7, .3, 1.1], dtype=torch.float64).reshape(3, 1, 1, 1)
    y = torch.tensor([.9, -.2, 1.4], dtype=torch.float64).reshape_as(a)
    mu1, lv1 = a*0, a*0+math.log(.8)
    al, si = math.exp(-.2), math.sqrt(-math.expm1(-.4))
    for model in (ToyConditional(nonlinear=True), ToyConditional(beta=0., slope=0.)):
        cm, cv = analytic_root_correction(model, a*0, a, y, mu1, lv1, al, si)
        point = a.clone().requires_grad_(True)
        mu, lv = model.encode_conditional(a*0, point)
        s = al**2*lv.exp()+si**2
        g = torch.autograd.grad((-.5*((y-al*mu).square()/s+s.log())).sum(), point)[0]
        expected = si*al*lv1.exp()/(al**2*lv1.exp()+si**2)*g
        torch.testing.assert_close(cm+cv, expected, rtol=1e-10, atol=1e-10)
        assert not cm.requires_grad and not cv.requires_grad
        if model.beta == 0: assert torch.count_nonzero(cm+cv) == 0
    # Linear-Gaussian counterexample: analytic posterior mean suffices because
    # Ccorr is affine in A. It equals true joint epsilon, while naive root differs.
    model = ToyConditional(slope=0.)
    v, w, beta = .8, .6, model.beta
    y1, y2 = .4, -.8
    s1, s2 = al**2*v+si**2, al**2*w+si**2
    vp = 1/(1/v+al**2/si**2+(al*beta)**2/s2)
    am = vp*(al*y1/si**2+al*beta*y2/s2)
    root = a[:1]*0+am
    cm, cv = analytic_root_correction(model, root*0, root, root*0+y2, root*0, root*0+math.log(v), al, si)
    c1 = si*y1/s1
    cov = torch.tensor([[s1, al**2*beta*v], [al**2*beta*v, al**2*(beta**2*v+w)+si**2]], dtype=torch.float64)
    exact = si*torch.linalg.solve(cov, torch.tensor([y1, y2], dtype=torch.float64))[0]
    torch.testing.assert_close((c1-cm-cv).flatten()[0], exact)
    assert abs(c1-float(exact)) > .01
    # Pooled means/SEs must agree with direct image-level reduction.
    log = DiagnosticLog(0)
    log.add('test', [1., 2.]); log.add('test', [3., 7., 9.])
    merged = _finish_records(log)[0]
    assert abs(merged['mean']-4.4) < 1e-12
    assert abs(merged['se']-float(torch.tensor([1., 2., 3., 7., 9.]).double().std()/math.sqrt(5))) < 1e-12
    import tempfile
    small = copy.copy(args)
    small.diagnostic_samples=2; small.diagnostic_batch=2; small.diagnostic_particles=4
    small.reference_particles=4; small.reference_queries=2; small.reference_bank=8
    data = dict(x=x, labels=labels, bank=x.repeat(4, 1, 1, 1), bank_labels=labels.repeat(4),
                times=[.2], indices=[0, 1], bank_indices=list(range(8)))
    vae.train(); head.train()
    ema = copy.deepcopy(head).eval().requires_grad_(False)
    # Preserve preexisting .grad as well as tensors, modes and random streams.
    params = list(vae.parameters())+list(head.parameters())
    tensors = [p.detach().clone() for p in params]
    grads = [p.grad.clone() if p.grad is not None else None for p in params]
    rng = torch.get_rng_state().clone()
    with tempfile.TemporaryDirectory() as tmp:
        evaluate_diagnostics(small, 0, vae, head, ema, data, torch.device('cpu'), Path(tmp), run_reference=True)
        saved = json.loads((Path(tmp)/'diagnostics'/'epoch_0000.json').read_text())
        metrics = {r['metric'] for r in saved['rows']}
        assert 'reference/root_head_relative_rms' in metrics
        assert 'representation/conditional_symkl_per_dim' in metrics
        assert 'pressure/score_gate_derivative' in metrics
        assert all(r['mean'] is None or math.isfinite(r['mean']) for r in saved['rows'])
    assert vae.training and head.training and not ema.training
    assert torch.equal(rng, torch.get_rng_state())
    for p, value, grad in zip(params, tensors, grads):
        torch.testing.assert_close(p, value, rtol=0, atol=0)
        if grad is None: assert p.grad is None
        else: torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)
    print('[SELF-TEST] PASSED: nonlinear correction derivatives, decoupled zero correction, '
          'Gaussian missing-term identity, pooled uncertainty, core/reference output and evaluation isolation.', flush=True)


def make_parser():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs',type=int,default=500)
    parser.add_argument('--batch',type=int,default=128)
    parser.add_argument('--num-workers',type=int,default=2)
    parser.add_argument('--z1-channels',type=int,default=4)
    parser.add_argument('--z2-channels',type=int,default=3)
    parser.add_argument('--joint-width',type=int,default=192)
    parser.add_argument('--joint-depth',type=int,default=18,
                        help='18 blocks nearly matches the parameter count of the original two 8-block heads')
    parser.add_argument('--joint-heads',type=int,default=6)
    parser.add_argument('--lr-vae',type=float,default=5e-4)
    parser.add_argument('--lr-head',type=float,default=2e-4)
    parser.add_argument('--score-weight',type=float,default=.6)
    parser.add_argument('--terminal-weight',type=float,default=1.)
    parser.add_argument('--tmin-band-weight',default='auto')
    parser.add_argument('--perc-weight',type=float,default=.85)
    parser.add_argument('--gan-weight',type=float,default=.0025)
    parser.add_argument('--disc-start',type=int,default=25)
    parser.add_argument('--lr-disc',type=float,default=1e-4)
    parser.add_argument('--label-dropout',type=float,default=.1)
    parser.add_argument('--ema',type=float,default=.9997)
    parser.add_argument('--t-min',type=float,default=2e-5)
    parser.add_argument('--t-max',type=float,default=2.)
    parser.add_argument('--decode-time',type=float,default=1e-4)
    parser.add_argument('--rk4-steps',type=int,default=20)
    parser.add_argument('--cfg',type=float,default=3.)
    parser.add_argument('--eval-every',type=int,default=50)
    parser.add_argument('--eval-samples',type=int,default=10000)
    parser.add_argument('--eval-batch',type=int,default=128)
    dg = parser.add_argument_group('Hypothesis diagnostics (read-only; epsilon score units)')
    dg.add_argument('--diagnostic-every', type=int, default=25, help='Core probe interval; 0 disables')
    dg.add_argument('--diagnostic-at-start', action=argparse.BooleanOptionalAction, default=True)
    dg.add_argument('--diagnostic-samples', type=int, default=64, help='Fixed held-out images')
    dg.add_argument('--diagnostic-batch', type=int, default=8, help='Image chunk size; reference uses batch*particles activations')
    dg.add_argument('--diagnostic-particles', type=int, default=8, help='Even root draws per image for dependence tests')
    dg.add_argument('--diagnostic-times', default='auto', help='auto = 8 geometric times, or comma-separated explicit times')
    dg.add_argument('--reference-every', type=int, default=50, help='Expensive aggregate-score audit interval; 0 disables')
    dg.add_argument('--reference-bank', type=int, default=128, help='Independent fixed train-image bank; class audit uses matching subset')
    dg.add_argument('--reference-queries', type=int, default=16, help='Held-out queries per time/conditioning')
    dg.add_argument('--reference-particles', type=int, default=8, help='Root proposal particles per bank image, per independent panel')
    dg.add_argument('--reference-tolerance', type=float, default=.15, help='Resolution screening threshold, not an accuracy guarantee')
    parser.add_argument('--feature-batch',type=int,default=128)
    parser.add_argument('--save-array-samples',type=int,default=256)
    parser.add_argument('--checkpoint-every',type=int,default=50)
    parser.add_argument('--archive-checkpoints',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    parser.add_argument('--cpu-threads',type=int,default=0)
    parser.add_argument('--max-batches',type=int,default=0)
    parser.add_argument('--output',default='runs/fmnist_joint_stacked_s42')
    parser.add_argument('--resume')
    parser.add_argument('--eval-only',action='store_true')
    parser.add_argument('--no-lpips',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--smoke-test',action='store_true',help='One REAL FashionMNIST minibatch, no FID/LPIPS/GAN')
    parser.add_argument('--synthetic-smoke-test',action='store_true',help='Synthetic software test only; no data download or FID')
    return parser


def main(argv=None):
    parser=make_parser()
    arguments=list(sys.argv[1:] if argv is None else argv)
    args=parser.parse_args(arguments)
    if args.self_test: self_test(); return
    explicit={arg.split('=',1)[0] for arg in arguments if arg.startswith('--')}
    checkpoint=None
    if args.resume:
        checkpoint=torch.load(args.resume,map_location='cpu',weights_only=False)
        if checkpoint.get('architecture_version')!=ARCHITECTURE_VERSION:
            parser.error('Resume requires a checkpoint from THIS joint-stack script. Cascade checkpoints have different heads.')
        old=checkpoint['args']
        # Preserve the full recipe, not just architecture, unless explicitly overridden.
        for key,value in old.items():
            if not hasattr(args,key) or key in ('resume','eval_only','self_test','smoke_test','synthetic_smoke_test'): continue
            if '--'+key.replace('_','-') not in explicit and '--no-'+key.replace('_','-') not in explicit:
                setattr(args,key,value)
        for key in ('z1_channels','z2_channels','joint_width','joint_depth','joint_heads'):
            if getattr(args,key)!=old[key]: parser.error('Resume architecture differs: '+key)
        if args.epochs!=old['epochs'] and not args.eval_only:
            parser.error('Keep --epochs equal to the checkpoint total to preserve the cosine schedule')
    if args.eval_only and checkpoint is None: parser.error('--eval-only requires --resume')
    if args.z1_channels<4 or args.z1_channels%4 or args.z2_channels<1:
        parser.error('Root channels must be a positive multiple of 4; child channels must be positive')
    if args.joint_width<4 or args.joint_heads<1 or args.joint_width%4 or args.joint_width%args.joint_heads:
        parser.error('Joint width must be positive and divisible by 4 and attention heads')
    for name in ('epochs','batch','joint_depth','eval_every','eval_batch','feature_batch','rk4_steps','checkpoint_every','disc_start'):
        if getattr(args,name)<1: parser.error(name+' must be positive')
    for name in ('score_weight','lr_vae','lr_head','lr_disc'):
        if not math.isfinite(getattr(args,name)) or getattr(args,name)<=0: parser.error(name+' must be finite and positive')
    for name in ('perc_weight','gan_weight','terminal_weight','cfg','decode_time'):
        if not math.isfinite(getattr(args,name)) or getattr(args,name)<0: parser.error(name+' must be finite and nonnegative')
    if not 0<args.t_min<args.t_max or not math.isfinite(args.t_max): parser.error('Require finite 0 < t_min < t_max')
    if args.decode_time>args.t_max: parser.error('decode-time must not exceed t-max')
    if not 0<=args.ema<1 or not 0<=args.label_dropout<1: parser.error('EMA and label dropout must be in [0,1)')
    if min(args.max_batches,args.cpu_threads,args.num_workers,args.save_array_samples)<0: parser.error('Counts must be nonnegative')
    if args.eval_samples==1 or args.eval_samples<0: parser.error('eval-samples must be 0 or >=2')
    if min(args.diagnostic_every, args.reference_every) < 0: parser.error('Diagnostic intervals must be nonnegative')
    for name in ('diagnostic_samples', 'diagnostic_batch', 'reference_bank', 'reference_queries'):
        if getattr(args, name) < 1: parser.error(name+' must be positive')
    if args.diagnostic_samples < 2: parser.error('diagnostic-samples must be >=2')
    if args.diagnostic_particles < 4 or args.diagnostic_particles%2:
        parser.error('diagnostic-particles must be even and >=4')
    if args.reference_particles < 2: parser.error('reference-particles must be >=2')
    if not math.isfinite(args.reference_tolerance) or args.reference_tolerance <= 0:
        parser.error('reference-tolerance must be finite and positive')
    if args.diagnostic_times != 'auto':
        try: times = [float(v) for v in args.diagnostic_times.split(',')]
        except ValueError: parser.error('diagnostic-times must be auto or comma-separated numbers')
        if not times or any(not math.isfinite(t) or not args.t_min <= t <= args.t_max for t in times):
            parser.error('diagnostic-times must lie in [t-min,t-max]')
    if args.tmin_band_weight.lower() not in ('auto','off','none'):
        try: value=float(args.tmin_band_weight)
        except ValueError: parser.error('Invalid tmin-band-weight')
        if not math.isfinite(value) or value<0: parser.error('Invalid tmin-band-weight')
    if args.no_lpips: args.perc_weight=0.
    if args.smoke_test:
        if args.resume or args.eval_only: parser.error('smoke-test must use a fresh output')
        args.epochs=1;args.max_batches=1;args.perc_weight=0.;args.gan_weight=0.;args.eval_samples=0;args.num_workers=0
    if args.synthetic_smoke_test:
        args.eval_samples=0;args.perc_weight=0.;args.num_workers=0
        if not args.cpu_threads: args.cpu_threads=2
    # No representation-only scale, warmup, target detachment, or DSM correction.
    args.csem_rep_weight=None
    train(args,checkpoint)


if __name__=='__main__':
    main()
