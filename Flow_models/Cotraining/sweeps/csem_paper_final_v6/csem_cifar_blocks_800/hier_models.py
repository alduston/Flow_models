"""Autoregressive Gaussian encoder factors and conditional diffusion priors.

All image information reaches the decoder through sampled, modeled latents.
Every factor has an exact Gaussian-component CSEM target at fixed clean parents.
Gaussian starts occur only at diffusion endpoints, never at a clean latent layer.
"""
from __future__ import annotations

import torch
from torch import nn
import core


class HierarchicalVAE(core.VAE):
    def __init__(self, *, block_channels, **kwargs):
        self.block_channels = tuple(int(c) for c in block_channels)
        if sum(self.block_channels) != int(kwargs['latent_channels']):
            raise ValueError('Block channels must sum to the original latent width')
        self.block_anchor_mode = kwargs.get('latent_anchor_mode', 'none')
        self.block_anchor_time = float(kwargs.get('latent_anchor_time', 0.0))
        if kwargs.get('use_norm', False):
            raise ValueError('This experiment uses OU-partial, not hard GroupNorm')
        # An ancestor-dependent global normalization after sampling would change
        # already-sampled root parameters. Apply the same OU-partial rule locally
        # to each Gaussian factor before its own draw instead.
        kwargs['latent_anchor_mode'] = 'none'
        super().__init__(**kwargs)
        self.conditions = nn.ModuleList()
        for i in range(1, len(self.block_channels)):
            adapter = nn.Sequential(
                nn.Conv2d(sum(self.block_channels[:i]), 64, 3, padding=1),
                nn.SiLU(), nn.Conv2d(64, 2*self.block_channels[i], 3, padding=1),
            )
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)
            self.conditions.append(adapter)

    def encode_sample(self, x, *, eps=None):
        """Return concatenated conditional moments and the SAME ancestral draw.

        The moment of block i is computed using the realized clean parents in
        this draw. Resampling a full Gaussian from the returned concatenation
        would silently break that coupling and is deliberately avoided.
        eps supplies a fixed full-width posterior noise bank during evaluation.
        """
        base_mu, base_lv = self.encode(x)
        mus = base_mu.split(self.block_channels, dim=1)
        lvs = base_lv.split(self.block_channels, dim=1)
        if eps is not None and eps.shape != base_mu.shape:
            raise ValueError('Posterior noise bank has the wrong shape')
        eps_blocks = eps.split(self.block_channels, dim=1) if eps is not None else None
        zs, out_mu, out_lv = [], [], []
        for i, (mu, lv) in enumerate(zip(mus, lvs)):
            if i:
                context = torch.cat(zs, dim=1)
                dmu, dlv = self.conditions[i-1](context).chunk(2, dim=1)
                mu, lv = mu + dmu, lv + dlv
            lv = lv.clamp(-30.0, 20.0)
            if self.block_anchor_mode == 'ou_partial':
                mu = core.ou_partial_groupnorm(mu, self.block_anchor_time, eps=1e-5)
            noise = eps_blocks[i] if eps_blocks is not None else torch.randn_like(mu)
            z = mu + torch.exp(0.5*lv)*noise
            out_mu.append(mu); out_lv.append(lv); zs.append(z)
        return torch.cat(out_mu, 1), torch.cat(out_lv, 1), torch.cat(zs, 1)


class ConditionalDiT(core.DiTModel):
    """Original factored CIFAR DiT with a clean-parent context token projection."""
    def __init__(self, *, context_channels, **kwargs):
        super().__init__(**kwargs)
        self.context_channels = int(context_channels)
        self.context_proj = nn.Conv2d(self.context_channels, self.hidden_dim,
                                      self.patch_size, stride=self.patch_size)
        nn.init.zeros_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)

    def forward(self, x, t, y=None, *, context, return_components=False,
                detach_components=False):
        B, _, H, W = x.shape
        if context.shape != (B, self.context_channels, H, W):
            raise ValueError(f'Clean-parent context shape {context.shape} does not match {x.shape}')
        tokens = self.patch_embed(x) + self.context_proj(context)
        ht, wt = tokens.shape[2:]
        tokens = tokens.flatten(2).transpose(1, 2) + self.pos_embed
        c = self.time_mlp(core.timestep_embedding(t.view(-1), self.hidden_dim))
        if self.label_emb is not None:
            ids = (torch.full((B,), int(self.null_label), device=x.device, dtype=torch.long)
                   if y is None else y.to(device=x.device, dtype=torch.long).view(-1))
            if ids.shape[0] != B:
                ids = ids.expand(B)
            c = c + self.label_emb(ids)
        for block in self.blocks:
            tokens = block(tokens, c)
        gamma, beta = self.final_modulation(c).unsqueeze(1).chunk(2, dim=-1)
        tokens = self.final_norm(tokens)*(1 + gamma) + beta
        if not self.factored_head:
            return self.unpatchify(self.final_proj(tokens), ht, wt)
        nu = self.unpatchify(self.nu_proj(tokens), ht, wt)
        lam = torch.exp(self.unpatchify(self.prec_proj(tokens), ht, wt).clamp(-20.0, 20.0))
        out = lam*x - nu
        if return_components:
            if detach_components:
                aux_tokens = tokens.detach()
                nu_aux = self.unpatchify(self.nu_proj(aux_tokens), ht, wt)
                lam_aux = torch.exp(self.unpatchify(self.prec_proj(aux_tokens), ht, wt).clamp(-20.0, 20.0))
                return out, lam_aux, nu_aux
            return out, lam, nu
        return out


class HierarchicalHeads(nn.Module):
    """Concatenation adapter for the original coordinate-mean routed objectives.

    Concatenated MSE/KL gives dimension-weighted block losses automatically.
    Only *preceding clean latents* are visible to the conditional head.
    The caller explicitly detaches context on the independent inner route.
    """
    def __init__(self, *, block_channels, score_depths, **kwargs):
        super().__init__()
        self.block_channels = tuple(int(v) for v in block_channels)
        self.score_depths = tuple(int(v) for v in score_depths)
        self.in_channels = sum(self.block_channels)
        self.factored_head = bool(kwargs.get('factored_head', False))
        self.num_classes = kwargs.get('num_classes')
        self.null_label = self.num_classes
        if len(self.block_channels) != len(self.score_depths):
            raise ValueError('One DiT depth is required per block')
        self.heads = nn.ModuleList()
        for i, (width, depth) in enumerate(zip(self.block_channels, self.score_depths)):
            opts = dict(kwargs, in_channels=width, depth=depth)
            head = (core.DiTModel(**opts) if i == 0 else
                    ConditionalDiT(context_channels=sum(self.block_channels[:i]), **opts))
            self.heads.append(head)

    def forward(self, x, t, y=None, *, context, return_components=False,
                detach_components=False):
        if x.shape != context.shape or x.shape[1] != self.in_channels:
            raise ValueError('Full latent and clean-parent bank shapes must agree')
        inputs = x.split(self.block_channels, dim=1)
        outputs = []
        offset = 0
        for i, (head, noisy, width) in enumerate(zip(self.heads, inputs, self.block_channels)):
            opts = dict(return_components=return_components, detach_components=detach_components)
            if i:
                opts['context'] = context[:, :offset]
            outputs.append(head(noisy, t, y, **opts))
            offset += width
        if return_components and self.factored_head:
            return tuple(torch.cat([result[j] for result in outputs], 1) for j in range(3))
        return torch.cat(outputs, 1)


class BoundConditionalHead(nn.Module):
    def __init__(self, head, context):
        super().__init__()
        self.head = head
        self.context = context

    def forward(self, x, t, y=None):
        return self.head(x, t, y, context=self.context)


@torch.no_grad()
def sample_hierarchy(heads, *, noise, y, cfg, method='rk4_ode', steps=20,
                     cfg_scale=2.5, root=None):
    """Generate every modeled factor using its learned diffusion prior.

    root optionally provides an encoded root for the decoder-gap diagnostic.
    All ancestors remain fixed while a descendant undergoes reverse diffusion.
    """
    bank = noise.split(heads.block_channels, dim=1)
    zs = []
    for i, (head, start) in enumerate(zip(heads.heads, bank)):
        if i == 0 and root is not None:
            if root.shape != start.shape:
                raise ValueError('Encoded root has the wrong shape')
            zs.append(root)
            continue
        model = head if i == 0 else BoundConditionalHead(head, torch.cat(zs, 1))
        sampler = core.UniversalSampler(method, int(steps), t_min=float(cfg['t_min']),
                                        t_max=float(cfg['t_max']), schedule_type=cfg['time_schedule'])
        zs.append(sampler.sample(model, x_init=start, y=y, cfg_scale=float(cfg_scale)))
    return torch.cat(zs, 1)
