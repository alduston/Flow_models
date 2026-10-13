"""Direct stochastic child cascade from the supplied 4.898-FID architecture note.

Baseline root: 8 x 4 x 4 = 128 coordinates; child: 6 x 8 x 8 = 384.
Posterior, conditional CSEM, terminal KL, B and decoder all use the SAME z2.
No root-dependent affine child location/scale or innovation coordinates exist.
Root packing is an exact pixel permutation solely for the inherited CIFAR IO.
Reconstruction recomputes every root feature from the noisy root supplied to it.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
import core
from hier_models import HierarchicalHeads


def pack_root(root, root_size=4):
    return F.pixel_shuffle(root, 8//root_size) if root_size == 4 else root


def unpack_root(packed, root_size=4):
    return F.pixel_unshuffle(packed, 8//root_size) if root_size == 4 else packed


class DirectHierarchicalVAE(core.VAE):
    def __init__(self, *, block_channels, root_size=4, **kwargs):
        self.block_channels = tuple(map(int, block_channels))
        if len(self.block_channels) != 2 or min(self.block_channels)<1 or sum(self.block_channels)!=8 or int(kwargs['latent_channels'])!=8:
            raise ValueError('Direct cascade requires two positive packed widths totaling eight')
        self.root_size = int(root_size)
        if self.root_size not in (4,8): raise ValueError('Root size must be 4 or 8')
        self.root_packed_channels, self.child_channels = self.block_channels
        self.root_channels = self.root_packed_channels*(8//self.root_size)**2
        self.latent_shapes = ((self.root_channels,self.root_size,self.root_size),(self.child_channels,8,8))
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
            nn.Conv2d(ch, ch, 3, stride=8//self.root_size, padding=1),
            core.VAEResBlock(ch, ch), core.VAEAttentionBlock(ch, zero_init=False),
            core.VAEResBlock(ch, ch))
        self.root_moments = nn.Sequential(core.make_group_norm(ch), nn.SiLU(), nn.Conv2d(ch, 2*self.root_channels, 3, padding=1))
        self.root_in = nn.Conv2d(self.root_channels, ch, 3, padding=1)
        self.root_blocks = nn.Sequential(
            core.VAEResBlock(ch, ch), core.VAEAttentionBlock(ch, zero_init=False),
            core.VAEResBlock(ch, ch))
        self.root_up = nn.Sequential(nn.Upsample(scale_factor=8//self.root_size, mode='nearest'),
                                     nn.Conv2d(ch, ch, 3, padding=1), core.VAEResBlock(ch, ch))
        self.child_moments = nn.Sequential(
            core.VAEResBlock(2*ch, ch), core.VAEResBlock(ch, ch),
            core.make_group_norm(ch), nn.SiLU(), nn.Conv2d(ch, 2*self.child_channels, 3, padding=1))
        nn.init.normal_(self.child_moments[-1].weight, std=1e-3)
        nn.init.zeros_(self.child_moments[-1].bias)
        self.child_in = nn.Conv2d(self.child_channels, ch, 3, padding=1)
        self.child_fusion = core.VAEResBlock(ch, ch)
        # The original decoder starts at 8x8: now fed by the fused 4->8 path.
        self.dec_conv_in = self.dec_post_proj = None

    def ancestor_features(self, root):
        return self.root_up(self.root_blocks(self.root_in(root)))

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
            er = unpack_root(eps[:, :self.root_packed_channels], self.root_size)
        else:
            er = torch.randn_like(mr)
        root = mr + (0.5*lr).exp()*er
        # Crucial: conditional inference is given the CLEAN sampled root.
        top = self.ancestor_features(root)
        mc, lc = self.child_moments(torch.cat((h, top), dim=1)).chunk(2, dim=1)
        mc, lc = self.anchor(mc), lc.clamp(-30., 20.)
        ec = eps[:, self.root_packed_channels:] if eps is not None else torch.randn_like(mc)
        child = mc + (0.5*lc).exp()*ec
        return (torch.cat((pack_root(mr,self.root_size), mc), 1),
                torch.cat((pack_root(lr,self.root_size), lc), 1),
                torch.cat((pack_root(root,self.root_size), child), 1))

    def encode(self, x, y=None):
        raise RuntimeError('Conditional child moments require encode_sample with the same clean root draw')

    def decode(self, packed, t=None):
        root = unpack_root(packed[:, :self.root_packed_channels],self.root_size)
        child = packed[:, self.root_packed_channels:]
        # The very same modeled child goes straight into the decoder.
        # Recompute root features from the argument; never reuse a clean cache.
        top = self.ancestor_features(root)
        h = self.child_fusion(top + self.child_in(child))
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
    def __init__(self, root_packed_channels=2, root_size=4, **kwargs):
        super().__init__()
        self.root_size = int(root_size)
        self.model = core.DiTModel(**dict(kwargs, in_channels=root_packed_channels*(8//self.root_size)**2, latent_size=self.root_size))
        self.num_classes = self.model.num_classes
        self.null_label = self.model.null_label
        self.factored_head = self.model.factored_head

    def forward(self, x, t, y=None, **kwargs):
        answer = self.model(unpack_root(x,self.root_size), t, y, **kwargs)
        return tuple(pack_root(v,self.root_size) for v in answer) if isinstance(answer, tuple) else pack_root(answer,self.root_size)


class DirectConditionalDiT(core.DiTModel):
    """Full-root spatial injection plus per-block root-token cross-attention.

    Maintains the original factored natural-parameter epsilon prediction and
    detached auxiliary (lambda,nu) branch required by CIFAR gradient routing.
    """
    def __init__(self, root_packed_channels=2, root_size=4, child_channels=6, **kwargs):
        super().__init__(**dict(kwargs, in_channels=child_channels, latent_size=8))
        self.root_size = int(root_size)
        self.root_packed_channels = int(root_packed_channels)
        self.context_channels = self.root_packed_channels*(8//self.root_size)**2
        self.context_proj = nn.Conv2d(self.context_channels, self.hidden_dim, 1)
        nn.init.normal_(self.context_proj.weight, std=.02)
        nn.init.zeros_(self.context_proj.bias)
        self.register_buffer('context_pos', self._build_2d_sincos_pos_embed(self.hidden_dim, self.root_size), persistent=False)
        self.context_norm = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in self.blocks])
        self.context_attn = nn.ModuleList([
            nn.MultiheadAttention(self.hidden_dim, kwargs['num_heads'], batch_first=True)
            for _ in self.blocks])
        self.context_gate = nn.Parameter(torch.full((len(self.blocks),), .1))

    def prepare_context(self, packed_clean_root):
        if packed_clean_root.ndim != 4 or packed_clean_root.shape[1:] != (self.root_packed_channels,8,8):
            raise ValueError('Packed clean ancestor has wrong dimensions')
        root = unpack_root(packed_clean_root,self.root_size)
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


class DirectHierarchicalHeads(HierarchicalHeads):
    def __init__(self, *, block_channels, score_depths, root_size=4, **kwargs):
        nn.Module.__init__(self)
        self.block_channels = tuple(map(int, block_channels))
        self.score_depths = tuple(map(int, score_depths))
        if len(self.block_channels)!=2 or sum(self.block_channels)!=8 or len(self.score_depths)!=2:
            raise ValueError('Expected two packed widths totaling eight and two DiT depths')
        self.root_size = int(root_size)
        root_packed_channels, child_channels = self.block_channels
        self.in_channels = 8
        self.factored_head = bool(kwargs.get('factored_head', False))
        self.num_classes = kwargs.get('num_classes')
        self.null_label = self.num_classes
        self.heads = nn.ModuleList([
            PackedRootDiT(root_packed_channels=root_packed_channels,root_size=self.root_size,**dict(kwargs,depth=self.score_depths[0])),
            DirectConditionalDiT(root_packed_channels=root_packed_channels,root_size=self.root_size,child_channels=child_channels,**dict(kwargs,depth=self.score_depths[1])),
        ])
