"""Direct three-block VAE from the supplied cluster bundle; score prior replaced elsewhere."""
import math
import torch
from torch import nn
import fmnist_reference as ref
from pinned.stacked_two_block import CascadeResBlock

class HierarchicalFMNISTVAE(nn.Module):
    """Three Gaussian direct-latent blocks, two spatial scales, zero bypass.

    q1(z1|x) on 4x4; q2(z2|x,z1) and q3(z3|x,z1,z2) on 8x8.
    The third posterior receives bottom-up evidence plus two clean ancestral
    top-down features. Both score heads receive exactly those sampled ancestors.
    """
    def __init__(self, z1_channels=4, z2_channels=1, z3_channels=2, **vae_kwargs):
        super().__init__()
        self.z1_channels, self.z2_channels, self.z3_channels = (
            int(z1_channels), int(z2_channels), int(z3_channels))
        self.latent_shapes = ((self.z1_channels, 4, 4),
                              (self.z2_channels, 8, 8),
                              (self.z3_channels, 8, 8))
        backbone = ref.VAE(latent_channels=4, **vae_kwargs)
        width = 4 * int(vae_kwargs['base_ch'])
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
        self.condition = nn.Sequential(
            CascadeResBlock(2*width, width), CascadeResBlock(width, width),
            nn.GroupNorm(16, width), nn.SiLU(),
            nn.Conv2d(width, 2*self.z2_channels, 3, padding=1))
        nn.init.normal_(self.condition[-1].weight, std=1e-3)
        nn.init.zeros_(self.condition[-1].bias)
        # Third posterior uses bottom-up, root-top-down and sampled z2. Its
        # z2 dependence is deterministic conditioning, not an unmodeled code.
        self.z2_to_posterior3 = nn.Sequential(
            nn.Conv2d(self.z2_channels, width, 3, padding=1),
            CascadeResBlock(width, width))
        self.condition3 = nn.Sequential(
            CascadeResBlock(3*width, width), CascadeResBlock(width, width),
            nn.GroupNorm(16, width), nn.SiLU(),
            nn.Conv2d(width, 2*self.z3_channels, 3, padding=1))
        nn.init.normal_(self.condition3[-1].weight, std=1e-3)
        nn.init.zeros_(self.condition3[-1].bias)
        # Decoder injection uses each stochastic code DIRECTLY, in ancestor
        # order. No affine child baseline or clean-root skip.
        self.child_in = nn.Conv2d(self.z2_channels, width, 3, padding=1)
        self.child_fusion = CascadeResBlock(width, width)
        self.child3_in = nn.Conv2d(self.z3_channels, width, 3, padding=1)
        self.child3_fusion = CascadeResBlock(width, width)
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

    def encode_conditional3(self, image_features, z1, z2, top_down=None):
        if top_down is None:
            top_down = self.ancestor_features(z1)
        second = self.z2_to_posterior3(z2)
        mu3, lv3 = self.condition3(torch.cat([image_features, top_down, second], 1)).chunk(2, 1)
        return mu3, lv3.clamp(-30, 20)

    def encode_sample(self, x, *, return_context=False):
        mu1, lv1, image_features = self.encode_root(x)
        z1 = mu1 + torch.exp(0.5 * lv1) * torch.randn_like(mu1)
        top_down = self.ancestor_features(z1)
        mu2, lv2 = self.encode_conditional(image_features, z1, top_down)
        z2 = mu2 + torch.exp(0.5 * lv2) * torch.randn_like(mu2)
        mu3, lv3 = self.encode_conditional3(image_features, z1, z2, top_down)
        z3 = mu3 + torch.exp(0.5 * lv3) * torch.randn_like(mu3)
        result = (z1,z2,z3), ((mu1,lv1),(mu2,lv2),(mu3,lv3))
        return (*result, top_down) if return_context else result

    def decode(self, z1, z2, z3, t=None, *, top_down=None):
        # For reconstruction top_down MUST be None, forcing all decoder
        # ancestor features to be recomputed from the corrupted z1 input.
        h = self.ancestor_features(z1) if top_down is None else top_down
        h = self.child_fusion(h + self.child_in(z2))
        h = self.child3_fusion(h + self.child3_in(z3))
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


