"""FMNIST-informed CIFAR two-stage residual cascade, architecture v3.

Root: eight channels at 4x4 (128 coords); child: six innovation channels at
8x8 (384 coords). Root pixel shuffle is an exact isometric packing to the
pinned trainer's 2+6 channels at 8x8. Both training reconstruction and
sampling ALWAYS recompute top-down features from the supplied (possibly noisy)
root. Child CSEM inference/score use the same CLEAN posterior root instead.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
import core
from hier_models import HierarchicalHeads


def pack_root(root):
    return F.pixel_shuffle(root, 2)


def unpack_root(packed):
    return F.pixel_unshuffle(packed, 2)


class OptimizedHierarchicalVAE(core.VAE):
    def __init__(self, *, block_channels, **kwargs):
        self.block_channels = tuple(map(int, block_channels))
        if self.block_channels != (2, 6) or int(kwargs['latent_channels']) != 8:
            raise ValueError('FMNIST-inspired cascade expects packed 2:6 channels (512 coordinates)')
        if int(kwargs.get('img_size', 32)) != 32 or kwargs.get('aux_d', 0):
            raise ValueError('Only CIFAR-32, aux_d=0 is supported')
        if kwargs.get('use_norm', False) or kwargs.get('num_classes') is not None:
            raise ValueError('Use conditional OU-partial posterior means and unconditional VAE encoder')
        self.block_anchor_mode = kwargs.get('latent_anchor_mode', 'none')
        self.block_anchor_time = float(kwargs.get('latent_anchor_time', 0.0))
        kwargs['latent_anchor_mode'] = 'none'
        super().__init__(**kwargs)
        ch = 4 * self.base_ch
        # Remove unused posterior maps and replace with two actual spatial stages.
        self.enc_norm_out = self.enc_conv_out = self.mu = self.logvar = None
        self.root_down = nn.Sequential(
            nn.Conv2d(ch, ch, 3, stride=2, padding=1),
            core.VAEResBlock(ch, ch), core.VAEAttentionBlock(ch, zero_init=False),
            core.VAEResBlock(ch, ch))
        self.root_moments = nn.Sequential(core.make_group_norm(ch), nn.SiLU(), nn.Conv2d(ch, 16, 3, padding=1))
        self.root_in = nn.Conv2d(8, ch, 3, padding=1)
        self.root_blocks = nn.Sequential(
            core.VAEResBlock(ch, ch), core.VAEAttentionBlock(ch, zero_init=False),
            core.VAEResBlock(ch, ch))
        self.root_up = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                     nn.Conv2d(ch, ch, 3, padding=1), core.VAEResBlock(ch, ch))
        self.child_moments = nn.Sequential(
            core.VAEResBlock(2*ch, ch), core.VAEResBlock(ch, ch),
            core.make_group_norm(ch), nn.SiLU(), nn.Conv2d(ch, 12, 3, padding=1))
        nn.init.normal_(self.child_moments[-1].weight, std=1e-3)
        nn.init.zeros_(self.child_moments[-1].bias)
        self.child_baseline = nn.Conv2d(ch, 12, 3, padding=1)
        nn.init.zeros_(self.child_baseline.weight)
        nn.init.zeros_(self.child_baseline.bias)
        self.child_in = nn.Conv2d(6, ch, 3, padding=1)
        self.child_fusion = core.VAEResBlock(ch, ch)
        # The original decoder starts at 8x8: now fed by the fused 4->8 path.
        self.dec_conv_in = None

    def ancestor_features(self, root):
        return self.root_up(self.root_blocks(self.root_in(root)))

    def baseline(self, features):
        location, raw_log_scale = self.child_baseline(features).chunk(2, dim=1)
        return location, 2.0 * torch.tanh(raw_log_scale / 2.0)

    def anchor(self, mean):
        if self.block_anchor_mode == 'ou_partial':
            return core.ou_partial_groupnorm(mean, self.block_anchor_time, eps=1e-5)
        return mean

    def encode_sample(self, x, *, eps=None):
        h = self.enc_conv_in(x)
        for block in self.enc_blocks:
            h = block(h)
        if self.enc_pre_proj is not None:
            h = self.enc_pre_proj(h)
        mr, lr = self.root_moments(self.root_down(h)).chunk(2, dim=1)
        mr, lr = self.anchor(mr), lr.clamp(-30., 20.)
        if eps is not None:
            if eps.shape != (len(x), 8, 8, 8):
                raise ValueError('Posterior noise bank must be B x 8 x 8 x 8')
            er = unpack_root(eps[:, :2])
        else:
            er = torch.randn_like(mr)
        root = mr + (0.5*lr).exp()*er
        # Crucial: conditional inference is given the CLEAN sampled root.
        top = self.ancestor_features(root)
        mc, lc = self.child_moments(torch.cat((h, top), dim=1)).chunk(2, dim=1)
        mc, lc = self.anchor(mc), lc.clamp(-30., 20.)
        ec = eps[:, 2:] if eps is not None else torch.randn_like(mc)
        child = mc + (0.5*lc).exp()*ec
        return torch.cat((pack_root(mr), mc), 1), torch.cat((pack_root(lr), lc), 1), torch.cat((pack_root(root), child), 1)

    def encode(self, x, y=None):
        raise RuntimeError('Conditional child moments require encode_sample with the same clean root draw')

    def decode(self, packed, t=None):
        root, innovation = unpack_root(packed[:, :2]), packed[:, 2:]
        # DO NOT cache clean-root features here. Training supplies (z1_t,u2_t).
        top = self.ancestor_features(root)
        loc, log_scale = self.baseline(top)
        raw_child = loc + log_scale.exp()*innovation
        h = self.child_fusion(top + self.child_in(raw_child))
        t_emb = None
        if self.time_cond_decoder and t is not None:
            t_emb = self.dec_time_mlp(core.timestep_embedding(t, self.dec_time_emb_dim))
        for i, block in enumerate(self.dec_blocks):
            h = block(h)
            if t_emb is not None:
                scale, shift = self.dec_film_layers[i](t_emb).chunk(2, dim=1)
                h = h * (1. + scale[:, :, None, None]) + shift[:, :, None, None]
        out = self.dec_out(h)
        return torch.tanh(out) if self.use_tanh_out else out

    def forward(self, x, y=None):
        mu, lv, z = self.encode_sample(x)
        return self.decode(z), mu, lv


class PackedRootDiT(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = core.DiTModel(**dict(kwargs, in_channels=8, latent_size=4))
        self.num_classes = self.model.num_classes
        self.null_label = self.model.null_label
        self.factored_head = self.model.factored_head

    def forward(self, x, t, y=None, **kwargs):
        answer = self.model(unpack_root(x), t, y, **kwargs)
        return tuple(pack_root(v) for v in answer) if isinstance(answer, tuple) else pack_root(answer)


class OptimizedConditionalDiT(core.DiTModel):
    """Full-root spatial injection plus per-block root-token cross-attention.

    Maintains the original factored natural-parameter epsilon prediction and
    detached auxiliary (lambda,nu) branch required by CIFAR gradient routing.
    """
    def __init__(self, **kwargs):
        super().__init__(**dict(kwargs, in_channels=6, latent_size=8))
        self.context_channels = 8
        self.context_proj = nn.Conv2d(8, self.hidden_dim, 1)
        nn.init.normal_(self.context_proj.weight, std=.02)
        nn.init.zeros_(self.context_proj.bias)
        self.register_buffer('context_pos', self._build_2d_sincos_pos_embed(self.hidden_dim, 4), persistent=False)
        self.context_norm = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in self.blocks])
        self.context_attn = nn.ModuleList([
            nn.MultiheadAttention(self.hidden_dim, kwargs['num_heads'], batch_first=True)
            for _ in self.blocks])
        self.context_gate = nn.Parameter(torch.full((len(self.blocks),), .1))

    def prepare_context(self, packed_clean_root):
        if packed_clean_root.ndim != 4 or packed_clean_root.shape[1:] != (2, 8, 8):
            raise ValueError('Expected packed clean ancestor [B,2,8,8]')
        root = unpack_root(packed_clean_root)
        features = self.context_proj(root)
        return (F.interpolate(features, size=(8, 8), mode='nearest'),
                features.flatten(2).transpose(1, 2) + self.context_pos)

    def forward(self, x, t, y=None, *, context, context_cache=None,
                return_components=False, detach_components=False):
        B, _, _, _ = x.shape
        spatial, root_tokens = self.prepare_context(context) if context_cache is None else context_cache
        if spatial.shape[0] != B:
            raise ValueError('Mismatched clean-ancestor batch')
        tokens = self.patch_embed(x) + spatial
        ht, wt = tokens.shape[2:]
        tokens = tokens.flatten(2).transpose(1, 2) + self.pos_embed
        c = self.time_mlp(core.timestep_embedding(t.view(-1), self.hidden_dim))
        if self.label_emb is not None:
            ids = (torch.full((B,), self.null_label, device=x.device, dtype=torch.long)
                   if y is None else y.to(x.device).long().view(-1))
            c = c + self.label_emb(ids)
        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens, c)
            update = self.context_attn[i](self.context_norm[i](tokens), root_tokens,
                                          root_tokens, need_weights=False)[0]
            tokens = tokens + torch.tanh(self.context_gate[i])*update
        gamma, beta = self.final_modulation(c).unsqueeze(1).chunk(2, dim=-1)
        tokens = self.final_norm(tokens)*(1.+gamma) + beta
        if not self.factored_head:
            if return_components:
                raise ValueError('Auxiliary outputs require factored head')
            return self.unpatchify(self.final_proj(tokens), ht, wt)
        nu = self.unpatchify(self.nu_proj(tokens), ht, wt)
        lam = self.unpatchify(self.prec_proj(tokens), ht, wt).clamp(-20., 20.).exp()
        out = lam*x - nu
        if return_components:
            if detach_components:
                aux = tokens.detach()
                nu_aux = self.unpatchify(self.nu_proj(aux), ht, wt)
                lam_aux = self.unpatchify(self.prec_proj(aux), ht, wt).clamp(-20., 20.).exp()
                return out, lam_aux, nu_aux
            return out, lam, nu
        return out


class OptimizedHierarchicalHeads(HierarchicalHeads):
    def __init__(self, *, block_channels, score_depths, **kwargs):
        nn.Module.__init__(self)
        self.block_channels = tuple(map(int, block_channels))
        self.score_depths = tuple(map(int, score_depths))
        if self.block_channels != (2, 6) or len(self.score_depths) != 2:
            raise ValueError('Expected 2:6 widths and two DiT depths')
        self.in_channels = 8
        self.factored_head = bool(kwargs.get('factored_head', False))
        self.num_classes = kwargs.get('num_classes')
        self.null_label = self.num_classes
        self.heads = nn.ModuleList([
            PackedRootDiT(**dict(kwargs, depth=self.score_depths[0])),
            OptimizedConditionalDiT(**dict(kwargs, depth=self.score_depths[1])),
        ])
