"""Two-resolution residual cascade with the original 512-coordinate budget.

Root: 8 x 4 x 4 (128 coordinates). Child innovation: 6 x 8 x 8 (384).
An orthogonal pixel permutation packs the root as 2 x 8 x 8 for the pinned
trainer's tensor interface. The root network and decoder actually operate at
4 x 4. Gaussian targets, coordinate means, noise banks and KL are invariant
under this permutation. It introduces no stochastic posterior dependency.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
import core
from hier_models import HierarchicalHeads, ConditionalDiT


def pack_root(root):
    return F.pixel_shuffle(root, 2)


def unpack_root(packed):
    return F.pixel_unshuffle(packed, 2)


class OptimizedHierarchicalVAE(core.VAE):
    def __init__(self, *, block_channels, **kwargs):
        self.block_channels = tuple(int(c) for c in block_channels)
        if self.block_channels != (2, 6) or int(kwargs['latent_channels']) != 8:
            raise ValueError('Optimized cascade requires packed channels 2:6, total width 8')
        if int(kwargs.get('img_size', 32)) != 32 or kwargs.get('aux_d', 0):
            raise ValueError('Optimized cascade is the CIFAR-32 non-mixture encoder')
        if kwargs.get('use_norm', False) or kwargs.get('num_classes') is not None:
            raise ValueError('Use OU-partial means and the original unconditional encoder')
        self.block_anchor_mode = kwargs.get('latent_anchor_mode', 'none')
        self.block_anchor_time = float(kwargs.get('latent_anchor_time', 0.0))
        kwargs['latent_anchor_mode'] = 'none'
        super().__init__(**kwargs)
        ch = 4 * self.base_ch
        # Replace the original flat posterior and decoder input projections.
        self.enc_norm_out = self.enc_conv_out = self.mu = self.logvar = None
        self.root_down = nn.Sequential(nn.Conv2d(ch, ch, 3, 2, 1), core.VAEResBlock(ch, ch))
        self.root_moments = nn.Sequential(core.make_group_norm(ch), nn.SiLU(), nn.Conv2d(ch, 16, 3, padding=1))
        self.dec_conv_in = nn.Conv2d(8, ch, 3, padding=1)
        self.root_decode = core.VAEResBlock(ch, ch)
        self.root_upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(ch, ch, 3, padding=1))
        self.child_moments = nn.Sequential(core.make_group_norm(2 * ch), nn.SiLU(),
            nn.Conv2d(2 * ch, ch, 3, padding=1), nn.SiLU(), nn.Conv2d(ch, 12, 3, padding=1))
        self.child_baseline = nn.Conv2d(ch, 12, 3, padding=1)
        nn.init.zeros_(self.child_baseline.weight)
        nn.init.zeros_(self.child_baseline.bias)
        self.child_inject = nn.Conv2d(6, ch, 3, padding=1)

    def coarse_features(self, root):
        h = self.dec_conv_in(root)
        if self.dec_post_proj is not None:
            h = self.dec_post_proj(h)
        return self.root_upsample(self.root_decode(h))

    def child_location_scale(self, context_features):
        location, log_scale = self.child_baseline(context_features).chunk(2, dim=1)
        return location, log_scale.clamp(-4.0, 4.0).exp()

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
        mr, lr = self.anchor(mr), lr.clamp(-30.0, 20.0)
        if eps is None:
            er = torch.randn_like(mr)
        else:
            if eps.shape != (len(x), 8, 8, 8):
                raise ValueError('Posterior noise bank must have packed shape B x 8 x 8 x 8')
            er = unpack_root(eps[:, :2])
        root = mr + (0.5 * lr).exp() * er
        context = self.coarse_features(root)
        mc, lc = self.child_moments(torch.cat((h, context), dim=1)).chunk(2, dim=1)
        mc, lc = self.anchor(mc), lc.clamp(-30.0, 20.0)
        ec = torch.randn_like(mc) if eps is None else eps[:, 2:]
        child = mc + (0.5 * lc).exp() * ec
        return (torch.cat((pack_root(mr), mc), dim=1),
                torch.cat((pack_root(lr), lc), dim=1),
                torch.cat((pack_root(root), child), dim=1))

    def encode(self, x, y=None):
        raise RuntimeError('Use encode_sample: child moments require the SAME clean root draw')

    def decode(self, packed, t=None):
        root, innovation = unpack_root(packed[:, :2]), packed[:, 2:]
        h = self.coarse_features(root)
        location, scale = self.child_location_scale(h)
        child = location + scale * innovation
        # Root enters at 4 x 4, child at 8 x 8, before the inherited 8->16->32 path.
        h = h + self.child_inject(child)
        t_emb = None
        if self.time_cond_decoder and t is not None:
            t_emb = self.dec_time_mlp(core.timestep_embedding(t, self.dec_time_emb_dim))
        for i, block in enumerate(self.dec_blocks):
            h = block(h)
            if t_emb is not None:
                scale_shift = self.dec_film_layers[i](t_emb)
                scale_t, shift_t = scale_shift.chunk(2, dim=1)
                h = h * (1.0 + scale_t[:, :, None, None]) + shift_t[:, :, None, None]
        out = self.dec_out(h)
        return torch.tanh(out) if self.use_tanh_out else out

    def forward(self, x, y=None):
        mu, lv, z = self.encode_sample(x)
        return self.decode(z), mu, lv


class PackedRootDiT(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        opts = dict(kwargs, in_channels=8, latent_size=4)
        self.model = core.DiTModel(**opts)
        self.num_classes = self.model.num_classes
        self.null_label = self.model.null_label
        self.factored_head = self.model.factored_head

    def forward(self, x, t, y=None, **kwargs):
        result = self.model(unpack_root(x), t, y, **kwargs)
        if isinstance(result, tuple):
            return tuple(pack_root(value) for value in result)
        return pack_root(result)


class ResidualConditionalDiT(ConditionalDiT):
    def __init__(self, **kwargs):
        super().__init__(context_channels=8, **kwargs)

    def forward(self, x, t, y=None, *, context, **kwargs):
        # Preserve the entire clean root; nearest upsampling is injective.
        parents = F.interpolate(unpack_root(context), size=x.shape[-2:], mode='nearest')
        return super().forward(x, t, y, context=parents, **kwargs)


class OptimizedHierarchicalHeads(HierarchicalHeads):
    def __init__(self, *, block_channels, score_depths, **kwargs):
        nn.Module.__init__(self)
        self.block_channels = tuple(int(c) for c in block_channels)
        self.score_depths = tuple(int(d) for d in score_depths)
        if self.block_channels != (2, 6) or len(self.score_depths) != 2:
            raise ValueError('Optimized heads require two blocks with packed widths 2:6')
        self.in_channels = 8
        self.factored_head = bool(kwargs.get('factored_head', False))
        self.num_classes = kwargs.get('num_classes')
        self.null_label = self.num_classes
        self.heads = nn.ModuleList([
            PackedRootDiT(**dict(kwargs, depth=self.score_depths[0])),
            ResidualConditionalDiT(**dict(kwargs, in_channels=6, latent_size=8, depth=self.score_depths[1])),
        ])
