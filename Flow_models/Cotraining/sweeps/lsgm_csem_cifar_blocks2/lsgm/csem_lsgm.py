"""CSEM surgery on NVlabs/LSGM. Channels follow NVAE's decoder group order.

The representation route uses likelihood weighting, live posterior moments and
live clean ancestors. The prior route uses detached inputs/moments/ancestors.
All losses are sums over latent coordinates, in LSGM's nats-per-image units.
"""
import copy
import csv
import json
import math
import os
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def posterior_moments(vae):
    return (torch.cat([d.mu for d in vae.last_q], 1).float(),
            torch.cat([d.log_sigma for d in vae.last_q], 1).float())


def csem_eps_target(eps_t, mu, log_sig, m_t, var_t):
    v = m_t.square() * torch.exp(2 * log_sig) + var_t
    return torch.sqrt(var_t) * (eps_t.float() - m_t * mu) / v


def terminal_component_kl(mu, log_sig, m_K, var_K):
    v = m_K.square() * torch.exp(2 * log_sig) + var_K
    return 0.5 * (m_K.square() * mu.square() + v - 1 - torch.log(v))


def band_term(log_sig, m_e, var_e):
    # log-sigma, not log-variance. Stable even for very narrow posteriors.
    return 0.5 * F.softplus(torch.log(var_e) - 2 * torch.log(m_e) - 2 * log_sig)


def diffusion_endpoints(diffusion, args, device):
    te = torch.tensor(args.time_eps, device=device, dtype=torch.float32)
    tk = torch.tensor(getattr(args, 'rep_horizon', 1.0), device=device, dtype=torch.float32)
    return diffusion.e2int_f(te), diffusion.var(te), diffusion.e2int_f(tk), diffusion.var(tk)


def q_times_rep(diffusion, size, time_eps, t_K, device='cuda'):
    t = time_eps + torch.rand(size, device=device) * (t_K - time_eps)
    var, m, g2 = diffusion.var(t), diffusion.e2int_f(t), diffusion.g2(t)
    w = g2 / (2 * var) * (t_K - time_eps)
    return t, var[:, None, None, None], m[:, None, None, None], w[:, None, None, None]


def representation_terms(vae, mu, log_sig, rep, diffusion, args):
    me, ve, mk, vk = diffusion_endpoints(diffusion, args, mu.device)
    K = (torch.zeros_like(rep) if getattr(args, 'no_csem_terminal', False) else
         getattr(args, 'csem_lambda_K', 1.0) * terminal_component_kl(mu, log_sig, mk, vk))
    B = torch.zeros_like(rep) if getattr(args, 'no_csem_band', False) else band_term(log_sig, me, ve)
    per_group = vae.decompose_eps(rep + K + B)
    kl_all = [g.sum((1, 2, 3)) for g in per_group]
    vals = torch.stack(kl_all, 1).mean(0)
    diag = [g.sum((2, 3)).mean(0) for g in per_group]
    return kl_all, vals, diag, K, B


class _ContextInputConv(nn.Module):
    def __init__(self, base, ctx_ch, nf):
        super().__init__()
        self.base = base
        self.proj = nn.Conv2d(ctx_ch, nf, 3, padding=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.context = None

    def forward(self, x):
        if self.context is None or self.context.shape[0] != x.shape[0]:
            raise ValueError('Conditional head requires matching clean-ancestor context')
        return self.base(x) + self.proj(self.context)


class ConditionalNCSNpp(nn.Module):
    def __init__(self, args, in_ch, ctx_ch, nf):
        super().__init__()
        from score_sde.ncsnpp import NCSNpp
        a = copy.copy(args)
        a.num_channels_dae = nf
        self.net = NCSNpp(a, in_ch)
        self.ctx_ch = ctx_ch
        if ctx_ch:
            if a.progressive_input != 'none':
                raise ValueError('Cascade requires progressive_input=none')
            self.net.all_modules[2] = _ContextInputConv(self.net.all_modules[2], ctx_ch, nf)

    def forward(self, x, t, context=None):
        if not self.ctx_ch:
            return self.net(x, t)
        conv = self.net.all_modules[2]
        conv.context = context
        try:
            return self.net(x, t)
        finally:
            conv.context = None


class CascadeDAE(nn.Module):
    def __init__(self, args, group_channels, nf):
        super().__init__()
        self.group_channels = list(group_channels)
        self.heads = nn.ModuleList([
            ConditionalNCSNpp(args, c, sum(group_channels[:g]), nf)
            for g, c in enumerate(group_channels)])
        self.mixed_prediction = args.mixed_prediction
        self.num_input_channels = sum(group_channels)
        self.input_size = self.heads[0].net.input_size
        self.is_active = None

    @property
    def mixing_logit(self):
        return torch.cat([h.net.mixing_logit for h in self.heads], 1)

    def forward(self, x, t, context):
        if context.shape != x.shape:
            raise ValueError('Full clean latent context must match the noised latent shape')
        outs, off = [], 0
        for head, c in zip(self.heads, self.group_channels):
            outs.append(head(x[:, off:off+c], t, context[:, :off] if off else None))
            off += c
        return torch.cat(outs, 1)


class _BoundHead(nn.Module):
    def __init__(self, head, context):
        super().__init__()
        self.head = head
        self.context = context
        self.mixed_prediction = head.net.mixed_prediction

    @property
    def mixing_logit(self):
        return self.head.net.mixing_logit

    def forward(self, x, t):
        return self.head(x, t, self.context)


def sample_cascade_ode(diffusion, dae, num_samples, ode_eps, tol, enable_autocast, temp, noise=None):
    if noise is not None and list(noise.shape) != [num_samples, dae.num_input_channels, dae.input_size, dae.input_size]:
        raise ValueError('Noise bank has the wrong cascade shape')
    zs, nfes, secs, off = [], [], 0.0, 0
    for head, c in zip(dae.heads, dae.group_channels):
        ctx = torch.cat(zs, 1) if zs else None
        n = None if noise is None else noise[:, off:off+c]
        z, k, s = diffusion.sample_model_ode(
            _BoundHead(head, ctx), num_samples, [c, dae.input_size, dae.input_size],
            ode_eps, tol, enable_autocast, temp, n)
        zs.append(z)
        nfes.append(float(k))
        secs += s
        off += c
    dae.last_sampling_nfe_per_head = nfes
    return torch.cat(zs, 1), sum(nfes), secs


def cascade_ode_nll(diffusion, dae, eps, **kw):
    # Repeat the entire conditional likelihood estimator for honest joint std.
    num_samples = kw.pop('num_samples', 1)
    report_std = kw.pop('report_std', False)
    if len(dae.heads) == 1:
        return diffusion.compute_ode_nll(
            dae=_BoundHead(dae.heads[0], None), eps=eps,
            num_samples=num_samples, report_std=report_std, **kw)
    estimates, counts = [], []
    for _ in range(num_samples):
        total, nfe, off = 0.0, 0.0, 0
        for head, c in zip(dae.heads, dae.group_channels):
            ctx = eps[:, :off].detach() if off else None
            val, k, _, _ = diffusion.compute_ode_nll(
                dae=_BoundHead(head, ctx), eps=eps[:, off:off+c],
                num_samples=1, report_std=False, **kw)
            total, nfe, off = total + val, nfe + k, off + c
        estimates.append(total)
        counts.append(nfe)
    vals = torch.stack(estimates, 1)
    std = vals.std(1).mean() if num_samples > 1 and report_std else None
    return vals.mean(1), sum(counts)/len(counts), std, None if std is None else std/math.sqrt(num_samples)


def build_prior(args, vae):
    from util import utils
    if getattr(args, 'prior_factorization', 'joint') == 'joint':
        return utils.get_dae_model(args, vae.latent_structure()[0])
    G = vae.num_groups_per_scale
    nf = getattr(args, 'dae_channels_per_head', 0) or 32 * int(args.num_channels_dae/math.sqrt(G)//32)
    return CascadeDAE(args, [vae.num_latent_per_group]*G, nf)


def validate_args(args):
    csem = getattr(args, 'csem', False)
    cascade = getattr(args, 'prior_factorization', 'joint') == 'cascade'
    if csem or cascade:
        if args.num_nf != 0 or args.num_latent_scales != 1:
            raise ValueError('CSEM/cascade requires num_nf=0 and one latent scale')
        if getattr(args, 'disjoint_training', False) or getattr(args, 'update_q_ema', False):
            raise ValueError('Surgery supports the joint trainer only')
    if csem:
        if args.iw_sample_q not in {'ll_iw', 'll_uniform'}:
            raise ValueError('CSEM requires a separately sampled likelihood q-route')
        if args.sde_type != 'vpsde' or args.sigma2_0 != 0:
            raise ValueError('This surgery is calibrated for VPSDE with sigma2_0=0')
        if args.num_groups_per_scale > 1 and not cascade:
            raise ValueError('Multiple nonlinear NVAE groups require a CSEM cascade')
        if not args.time_eps < args.rep_horizon <= 1 or args.csem_lambda_K < 0:
            raise ValueError('Require time_eps < rep_horizon <= 1 and lambda_K >= 0')
    elif getattr(args, 'rep_horizon', 1.0) != 1.0:
        raise ValueError('rep_horizon is a CSEM-only option')
    if cascade and getattr(args, 'drop_inactive_var', False):
        raise ValueError('In-place mixing_logit writes are unsupported for cascade')


def parameter_counts(vae, dae):
    count = lambda m: sum(p.numel() for p in m.parameters())
    return {'vae_parameters': count(vae), 'prior_parameters': count(dae),
            'prior_parameters_per_head': [count(h) for h in dae.heads] if isinstance(dae, CascadeDAE) else [count(dae)]}


def scalar(v):
    return v.detach().cpu().item() if isinstance(v, torch.Tensor) else v


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=scalar) + '\n')
    os.replace(tmp, path)


def append_csv(path, row, key=None):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
    row = {k: scalar(v) for k, v in row.items()}
    if key:
        rows = [r for r in rows if str(r.get(key)) != str(row[key])]
    rows.append(row)
    columns = list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def group_diagnostics(vae, dae, log_sig, rep, K, B, diffusion, args):
    with torch.no_grad():
        me, ve, _, _ = diffusion_endpoints(diffusion, args, log_sig.device)
        floor = torch.sqrt(ve)/me
        row = {}
        for i, (ls, r, k, b) in enumerate(zip(*(vae.decompose_eps(v) for v in (log_sig, rep, K, B)))):
            sig = ls.exp()
            row.update({f'rep_{i}': r.sum((1,2,3)).mean(), f'K_{i}': k.sum((1,2,3)).mean(),
                        f'B_{i}': b.sum((1,2,3)).mean(), f'below_{i}': (sig < floor).float().mean(),
                        f'deep_{i}': (sig < floor/10).float().mean()})
            for q, val in zip((0.01, 0.5, 0.99), torch.quantile(sig.flatten(), torch.tensor([0.01,0.5,0.99], device=sig.device))):
                row[f'sigma_q{int(100*q):02d}_{i}'] = val
        for i, m in enumerate(vae.decompose_eps(torch.sigmoid(dae.mixing_logit))):
            row[f'mixing_{i}'] = m.mean()
        return row
