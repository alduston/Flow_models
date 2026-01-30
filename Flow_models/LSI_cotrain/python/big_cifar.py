import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import numpy as np
import math
from functools import partial
from typing import Optional, Tuple, List, Dict, Any
from tqdm import tqdm
import os
from collections import OrderedDict

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    """Extract values from a at indices t and reshape for broadcasting."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

# ============================================================================
# ATTENTION & NORMALIZATION MODULES
# ============================================================================

class GroupNorm32(nn.GroupNorm):
    """GroupNorm with float32 precision for stability."""
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def Normalize(in_channels, num_groups=32):
    return GroupNorm32(num_groups=min(num_groups, in_channels), num_channels=in_channels, eps=1e-6, affine=True)

class SelfAttention(nn.Module):
    """Self-attention module for image features."""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = Normalize(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head attention
        q = q.view(b, self.num_heads, self.head_dim, h * w)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.contiguous().view(b, c, h, w)

        return x + self.proj_out(out)

class LinearAttention(nn.Module):
    """Efficient linear attention for larger feature maps."""
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            Normalize(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, self.heads, -1, h * w), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = out.contiguous().view(b, c, h, w)

        return self.to_out(out) + x

# ============================================================================
# RESIDUAL BLOCKS
# ============================================================================

class ResnetBlock(nn.Module):
    """Residual block with optional time embedding."""
    def __init__(self, in_channels, out_channels=None, time_emb_dim=None, dropout=0.0):
        super().__init__()
        out_channels = default(out_channels, in_channels)

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        else:
            self.time_mlp = None

        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.conv1(F.silu(self.norm1(x)))

        if self.time_mlp is not None and time_emb is not None:
            h = h + self.time_mlp(time_emb)[:, :, None, None]

        h = self.conv2(self.dropout(F.silu(self.norm2(h))))

        return h + self.skip_conv(x)

# ============================================================================
# VAE ENCODER
# ============================================================================

class Encoder(nn.Module):
    """
    VAE Encoder that maps images to latent distributions.
    Outputs mean and log-variance for diagonal Gaussian.

    Architecture follows Stable Diffusion style with:
    - ResNet blocks
    - Self-attention at lower resolutions
    - Downsampling via strided convolutions
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        latent_channels: int = 4,
        dropout: float = 0.0,
        use_attention: bool = True,
        attention_resolutions: Tuple[int, ...] = (16,),  # Resolutions at which to use attention
        double_z: bool = True,  # Output both mean and logvar
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.double_z = double_z

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        ch = base_channels
        current_res = 32  # CIFAR-10 resolution

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult

            for j in range(num_res_blocks):
                self.down_blocks.append(ResnetBlock(ch, out_ch, dropout=dropout))
                ch = out_ch
                channels.append(ch)

                if use_attention and current_res in attention_resolutions:
                    self.down_blocks.append(SelfAttention(ch))

            if i != len(channel_mults) - 1:
                self.down_blocks.append(Downsample(ch))
                channels.append(ch)
                current_res //= 2

        # Middle
        self.mid_block1 = ResnetBlock(ch, ch, dropout=dropout)
        self.mid_attn = SelfAttention(ch)
        self.mid_block2 = ResnetBlock(ch, ch, dropout=dropout)

        # Output
        self.norm_out = Normalize(ch)
        out_channels = 2 * latent_channels if double_z else latent_channels
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)

        for block in self.down_blocks:
            if isinstance(block, (ResnetBlock, SelfAttention)):
                h = block(h)
            else:  # Downsample
                h = block(h)

        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)

        return h

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

# ============================================================================
# VAE DECODER
# ============================================================================

class Decoder(nn.Module):
    """
    VAE Decoder that maps latent codes to images.
    """
    def __init__(
        self,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        latent_channels: int = 4,
        dropout: float = 0.0,
        use_attention: bool = True,
        attention_resolutions: Tuple[int, ...] = (16,),
    ):
        super().__init__()

        # Calculate channels
        ch = base_channels * channel_mults[-1]

        # Initial convolution from latent
        self.conv_in = nn.Conv2d(latent_channels, ch, 3, padding=1)

        # Middle
        self.mid_block1 = ResnetBlock(ch, ch, dropout=dropout)
        self.mid_attn = SelfAttention(ch)
        self.mid_block2 = ResnetBlock(ch, ch, dropout=dropout)

        # Upsampling
        self.up_blocks = nn.ModuleList()
        current_res = 32 // (2 ** (len(channel_mults) - 1))  # Starting resolution

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult

            for j in range(num_res_blocks + 1):
                self.up_blocks.append(ResnetBlock(ch, out_ch, dropout=dropout))
                ch = out_ch

                if use_attention and current_res in attention_resolutions:
                    self.up_blocks.append(SelfAttention(ch))

            if i != 0:
                self.up_blocks.append(Upsample(ch))
                current_res *= 2

        # Output
        self.norm_out = Normalize(ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)

        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        for block in self.up_blocks:
            if isinstance(block, (ResnetBlock, SelfAttention)):
                h = block(h)
            else:  # Upsample
                h = block(h)

        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)

        return h

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

# ============================================================================
# DIAGONAL GAUSSIAN DISTRIBUTION
# ============================================================================

class DiagonalGaussian:
    """Diagonal Gaussian distribution for VAE."""
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self):
        if self.deterministic:
            return self.mean
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self):
        """KL divergence from N(0, I)."""
        return 0.5 * torch.sum(
            self.mean.pow(2) + self.var - 1.0 - self.logvar,
            dim=[1, 2, 3]
        )

    def mode(self):
        return self.mean

# ============================================================================
# COMPLETE VAE
# ============================================================================

class AutoencoderKL(nn.Module):
    """
    Complete Variational Autoencoder with KL regularization.
    Stable Diffusion style with diagonal Gaussian latent.

    For CIFAR-10 (32x32):
    - With default f=4 downsampling: latent is 8x8
    - With f=2 downsampling: latent is 16x16
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,  # Smaller for CIFAR-10
        channel_mults: Tuple[int, ...] = (1, 2, 4),  # f=4 downsampling (32->8)
        num_res_blocks: int = 2,
        latent_channels: int = 4,
        dropout: float = 0.0,
        use_attention: bool = True,
        attention_resolutions: Tuple[int, ...] = (8,),
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            latent_channels=latent_channels,
            dropout=dropout,
            use_attention=use_attention,
            attention_resolutions=attention_resolutions,
            double_z=True,
        )

        self.decoder = Decoder(
            out_channels=out_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            latent_channels=latent_channels,
            dropout=dropout,
            use_attention=use_attention,
            attention_resolutions=attention_resolutions,
        )

        # Scaling factor (as in Stable Diffusion)
        # This helps normalize latents to approximately unit variance
        self.scale_factor = 0.18215

    def encode(self, x):
        """Encode image to latent distribution."""
        h = self.encoder(x)
        return DiagonalGaussian(h)

    def decode(self, z):
        """Decode latent to image."""
        return self.decoder(z)

    def forward(self, x, sample_posterior=True):
        """Full forward pass."""
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        reconstruction = self.decode(z)
        return reconstruction, posterior

    def get_latent(self, x, deterministic=False):
        """Get scaled latent for diffusion training."""
        posterior = self.encode(x)
        if deterministic:
            z = posterior.mode()
        else:
            z = posterior.sample()
        return z * self.scale_factor

    def decode_latent(self, z):
        """Decode from scaled latent."""
        z = z / self.scale_factor
        return self.decode(z)

# ============================================================================
# LPIPS PERCEPTUAL LOSS
# ============================================================================

class LPIPS(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) loss.
    Uses VGG16 features by default.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights

        if pretrained:
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        else:
            vgg = vgg16()

        self.features = nn.ModuleList([
            nn.Sequential(*list(vgg.features[:4])),   # relu1_2
            nn.Sequential(*list(vgg.features[4:9])),  # relu2_2
            nn.Sequential(*list(vgg.features[9:16])), # relu3_3
            nn.Sequential(*list(vgg.features[16:23])), # relu4_3
            nn.Sequential(*list(vgg.features[23:30])), # relu5_3
        ])

        for param in self.features.parameters():
            param.requires_grad = False

        # Learned weights for each layer
        self.weights = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.1) for _ in range(5)
        ])

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        """Normalize from [-1, 1] to ImageNet scale."""
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, x, y):
        """Compute LPIPS distance between x and y."""
        x = self.normalize(x)
        y = self.normalize(y)

        loss = 0.0
        fx, fy = x, y

        for i, layer in enumerate(self.features):
            fx = layer(fx)
            fy = layer(fy)

            # Normalize features
            fx_norm = fx / (fx.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)
            fy_norm = fy / (fy.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)

            loss += self.weights[i] * (fx_norm - fy_norm).pow(2).mean()

        return loss

# ============================================================================
# PATCHGAN DISCRIMINATOR
# ============================================================================

class NLayerDiscriminator(nn.Module):
    """
    PatchGAN discriminator for adversarial VAE training.
    Outputs a grid of predictions rather than single value.
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        ch = base_channels
        for i in range(1, n_layers):
            out_ch = min(ch * 2, 512)
            layers += [
                nn.Conv2d(ch, out_ch, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            ch = out_ch

        out_ch = min(ch * 2, 512)
        layers += [
            nn.Conv2d(ch, out_ch, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, 1, 4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ============================================================================
# VAE LOSS FUNCTIONS
# ============================================================================

class VAELoss(nn.Module):
    """
    Complete VAE loss combining:
    - Reconstruction loss (L1 or L2)
    - KL divergence (with small weight for mild regularization)
    - LPIPS perceptual loss
    - Optional discriminator loss (hinge)
    """
    def __init__(
        self,
        kl_weight: float = 1e-6,  # Very small KL weight (Stable Diffusion style)
        lpips_weight: float = 1.0,
        recon_weight: float = 1.0,
        disc_weight: float = 0.5,
        disc_start_step: int = 50001,  # Start discriminator after this step
        use_lpips: bool = True,
        use_disc: bool = True,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.lpips_weight = lpips_weight
        self.recon_weight = recon_weight
        self.disc_weight = disc_weight
        self.disc_start_step = disc_start_step
        self.use_lpips = use_lpips
        self.use_disc = use_disc

        if use_lpips:
            self.lpips = LPIPS()
        if use_disc:
            self.discriminator = NLayerDiscriminator()

    def calc_adaptive_weight(self, nll_loss, g_loss, last_layer):
        """Calculate adaptive weight for GAN loss (from taming-transformers)."""
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight * self.disc_weight

    def forward(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        posterior: DiagonalGaussian,
        global_step: int,
        last_layer: Optional[nn.Parameter] = None,
        split: str = "train"
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE losses.

        Args:
            x: Original images
            reconstruction: Reconstructed images
            posterior: Latent distribution
            global_step: Current training step
            last_layer: Last decoder layer for adaptive weight
            split: "train" or "val"
        """
        # Reconstruction loss (L1)
        rec_loss = F.l1_loss(reconstruction, x)

        # KL loss
        kl_loss = posterior.kl().mean()

        # LPIPS loss
        if self.use_lpips:
            lpips_loss = self.lpips(reconstruction, x)
        else:
            lpips_loss = torch.tensor(0.0, device=x.device)

        # Total non-adversarial loss
        nll_loss = self.recon_weight * rec_loss + self.lpips_weight * lpips_loss

        # Discriminator loss
        if self.use_disc and global_step >= self.disc_start_step:
            # Generator loss
            logits_fake = self.discriminator(reconstruction)
            g_loss = -torch.mean(logits_fake)  # Hinge loss

            if last_layer is not None:
                try:
                    d_weight = self.calc_adaptive_weight(nll_loss, g_loss, last_layer)
                except RuntimeError:
                    d_weight = torch.tensor(1.0)
            else:
                d_weight = torch.tensor(1.0)

            loss = nll_loss + self.kl_weight * kl_loss + d_weight * g_loss

            # Discriminator loss (for separate update)
            logits_real = self.discriminator(x.detach())
            logits_fake_d = self.discriminator(reconstruction.detach())

            d_loss_real = torch.mean(F.relu(1. - logits_real))
            d_loss_fake = torch.mean(F.relu(1. + logits_fake_d))
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
        else:
            loss = nll_loss + self.kl_weight * kl_loss
            g_loss = torch.tensor(0.0, device=x.device)
            d_loss = torch.tensor(0.0, device=x.device)

        return {
            'loss': loss,
            'rec_loss': rec_loss,
            'kl_loss': kl_loss,
            'lpips_loss': lpips_loss,
            'g_loss': g_loss,
            'd_loss': d_loss,
        }

# ============================================================================
# UNET FOR DIFFUSION (Class-Conditional for CIFAR-10)
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal time embeddings."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

class DownBlock(nn.Module):
    """Downsampling block for UNet."""
    def __init__(self, in_ch, out_ch, time_emb_dim, has_attn=False, dropout=0.1):
        super().__init__()
        self.res1 = ResnetBlock(in_ch, out_ch, time_emb_dim, dropout)
        self.res2 = ResnetBlock(out_ch, out_ch, time_emb_dim, dropout)
        self.attn = SelfAttention(out_ch) if has_attn else nn.Identity()
        self.downsample = Downsample(out_ch)

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.res2(x, t)
        x = self.attn(x)
        skip = x
        x = self.downsample(x)
        return x, skip

class UpBlock(nn.Module):
    """Upsampling block for UNet."""
    def __init__(self, in_ch, out_ch, time_emb_dim, has_attn=False, dropout=0.1):
        super().__init__()
        self.upsample = Upsample(in_ch)
        self.res1 = ResnetBlock(in_ch + out_ch, out_ch, time_emb_dim, dropout)
        self.res2 = ResnetBlock(out_ch, out_ch, time_emb_dim, dropout)
        self.attn = SelfAttention(out_ch) if has_attn else nn.Identity()

    def forward(self, x, skip, t):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t)
        x = self.res2(x, t)
        x = self.attn(x)
        return x

class UNet(nn.Module):
    """
    UNet for latent diffusion.
    Operates on latent space (e.g., 8x8x4 for CIFAR-10 with f=4 VAE).
    Class-conditional via embedding.
    """
    def __init__(
        self,
        in_channels: int = 4,  # Latent channels
        out_channels: int = 4,
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        num_classes: int = 10,  # CIFAR-10 classes
        dropout: float = 0.1,
        attention_resolutions: Tuple[int, ...] = (4, 2),  # Attention at these latent resolutions
    ):
        super().__init__()

        time_dim = base_channels * 4

        # Time embedding
        self.time_pos_emb = SinusoidalPosEmb(base_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Class embedding (for CFG)
        self.class_emb = nn.Embedding(num_classes + 1, time_dim)  # +1 for unconditional
        self.null_class = num_classes  # Use this for unconditional

        # Initial conv
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        ch = base_channels
        current_res = 8  # Starting latent resolution for 32x32 image with f=4

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            has_attn = current_res in attention_resolutions

            self.down_blocks.append(
                DownBlock(ch, out_ch, time_dim, has_attn, dropout)
            )
            ch = out_ch
            channels.append(ch)
            current_res //= 2

        # Middle
        self.mid_res1 = ResnetBlock(ch, ch, time_dim, dropout)
        self.mid_attn = SelfAttention(ch)
        self.mid_res2 = ResnetBlock(ch, ch, time_dim, dropout)

        # Upsampling
        self.up_blocks = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            has_attn = current_res in attention_resolutions

            self.up_blocks.append(
                UpBlock(ch, out_ch, time_dim, has_attn, dropout)
            )
            ch = out_ch
            current_res *= 2

        # Output
        self.norm_out = Normalize(ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x, t, y=None, drop_class_prob=0.0):
        """
        Args:
            x: Noisy latent [B, C, H, W]
            t: Timesteps [B]
            y: Class labels [B], optional
            drop_class_prob: Probability of dropping class for CFG training
        """
        # Time embedding
        t = t.float()                      # (B,) long -> float
        t_emb = self.time_pos_emb(t)       # (B,) -> (B, base_channels)
        t_emb = self.time_mlp(t_emb)       # (B, base_channels) -> (B, time_dim)

        # Class embedding
        if y is not None:
            # Randomly drop class for CFG training
            if self.training and drop_class_prob > 0:
                mask = torch.rand(y.shape[0], device=y.device) < drop_class_prob
                y = torch.where(mask, self.null_class * torch.ones_like(y), y)
            c_emb = self.class_emb(y)
            t_emb = t_emb + c_emb
        else:
            # Unconditional
            c_emb = self.class_emb(torch.full((x.shape[0],), self.null_class, device=x.device, dtype=torch.long))
            t_emb = t_emb + c_emb

        # UNet
        h = self.conv_in(x)

        skips = []
        for block in self.down_blocks:
            h, skip = block(h, t_emb)
            skips.append(skip)

        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)

        for block in self.up_blocks:
            skip = skips.pop()
            h = block(h, skip, t_emb)

        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)

        return h

# ============================================================================
# DIFFUSION PROCESS
# ============================================================================

class GaussianDiffusion(nn.Module):
    """
    DDPM/DDIM diffusion process for latent space.
    """
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = 'cosine',
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        # Noise schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance', torch.log(posterior_variance.clamp(min=1e-20)))

        self.register_buffer('posterior_mean_coef1',
                             betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x_0 from x_t and predicted noise."""
        return (
            extract(1.0 / self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """Posterior distribution q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(self, x_t, t, y=None, clip_denoised=True, cfg_scale=1.0):
        """Model prediction for p(x_{t-1} | x_t)."""
        # Predict noise
        if cfg_scale > 1.0 and y is not None:
            # Classifier-free guidance
            noise_cond = self.model(x_t, t, y)
            noise_uncond = self.model(x_t, t, None)
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = self.model(x_t, t, y)

        # Predict x_0
        x_start = self.predict_start_from_noise(x_t, t, noise_pred)

        if clip_denoised:
            x_start = torch.clamp(x_start, -1.0, 1.0)

        # Get posterior
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start, x_t, t)

        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x_t, t, y=None, cfg_scale=1.0):
        """Sample from p(x_{t-1} | x_t) using DDPM."""
        model_mean, _, model_log_variance, _ = self.p_mean_variance(
            x_t, t, y, clip_denoised=True, cfg_scale=cfg_scale
        )

        noise = torch.randn_like(x_t)
        # No noise at t=0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))

        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def sample_ddpm(self, shape, y=None, cfg_scale=1.0, device='cuda'):
        """Full DDPM sampling."""
        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, y, cfg_scale)

        return x

    @torch.no_grad()
    def sample_ddim(self, shape, y=None, cfg_scale=1.0, ddim_steps=50, eta=0.0, device='cuda'):
        """DDIM sampling with fewer steps."""
        batch_size = shape[0]

        # DDIM timesteps
        step_size = self.timesteps // ddim_steps
        timesteps = list(range(0, self.timesteps, step_size))
        timesteps = list(reversed(timesteps))

        x = torch.randn(shape, device=device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict noise
            if cfg_scale > 1.0 and y is not None:
                noise_cond = self.model(x, t_batch, y)
                noise_uncond = self.model(x, t_batch, None)
                noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = self.model(x, t_batch, y)

            # Predict x_0
            alpha_t = extract(self.alphas_cumprod, t_batch, x.shape)
            x_0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

            # Get previous timestep
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                t_prev_batch = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)
                alpha_t_prev = extract(self.alphas_cumprod, t_prev_batch, x.shape)
            else:
                alpha_t_prev = torch.ones_like(alpha_t)

            # DDIM step
            sigma = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)

            pred_dir = torch.sqrt(1 - alpha_t_prev - sigma ** 2) * noise_pred
            noise = torch.randn_like(x) if i < len(timesteps) - 1 else 0

            x = torch.sqrt(alpha_t_prev) * x_0_pred + pred_dir + sigma * noise

        return x

    def loss(self, x_start, t=None, y=None, noise=None):
        """Compute diffusion training loss."""
        batch_size = x_start.shape[0]
        device = x_start.device

        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        if noise is None:
            noise = torch.randn_like(x_start)

        # Forward diffusion
        x_t = self.q_sample(x_start, t, noise)

        # Predict noise
        noise_pred = self.model(x_t, t, y, drop_class_prob=0.1)  # 10% unconditional for CFG

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)

        return loss

# ============================================================================
# COMPLETE LATENT DIFFUSION MODEL
# ============================================================================

class LatentDiffusion(nn.Module):
    """
    Complete Latent Diffusion Model combining VAE and Diffusion.
    """
    def __init__(
        self,
        vae: AutoencoderKL,
        diffusion: GaussianDiffusion,
    ):
        super().__init__()
        self.vae = vae
        self.diffusion = diffusion

    def encode(self, x):
        """Encode images to scaled latents."""
        return self.vae.get_latent(x, deterministic=False)

    def decode(self, z):
        """Decode scaled latents to images."""
        return self.vae.decode_latent(z)

    @torch.no_grad()
    def sample(self, batch_size, y=None, cfg_scale=1.5, ddim_steps=50, device='cuda'):
        """Sample images."""
        # Sample latents
        latent_shape = (batch_size, 4, 8, 8)  # For CIFAR-10 with f=4
        z = self.diffusion.sample_ddim(latent_shape, y, cfg_scale, ddim_steps, device=device)

        # Decode
        images = self.decode(z)
        images = torch.clamp(images, -1.0, 1.0)

        return images

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def get_cifar10_dataloader(batch_size=128, num_workers=4, train=True):
    """Get CIFAR-10 dataloader."""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
    ])

    dataset = datasets.CIFAR10(
        root='./data',
        train=train,
        download=True,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )

    return dataloader

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_vae(
    vae: AutoencoderKL,
    loss_fn: VAELoss,
    train_loader: DataLoader,
    num_epochs: int = 100,
    lr: float = 4.5e-6,
    save_dir: str = './checkpoints',
    device: str = 'cuda',
):
    """Train the VAE."""
    os.makedirs(save_dir, exist_ok=True)

    vae = vae.to(device)
    loss_fn = loss_fn.to(device)

    # Optimizers
    opt_ae = torch.optim.Adam(
        list(vae.encoder.parameters()) + list(vae.decoder.parameters()),
        lr=lr,
        betas=(0.5, 0.9),
    )

    if loss_fn.use_disc:
        opt_disc = torch.optim.Adam(
            loss_fn.discriminator.parameters(),
            lr=lr,
            betas=(0.5, 0.9),
        )

    ema = EMA(vae, decay=0.999)
    global_step = 0

    for epoch in range(num_epochs):
        vae.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)

            # VAE forward
            recon, posterior = vae(images)

            # Compute losses
            last_layer = vae.decoder.conv_out.weight
            losses = loss_fn(images, recon, posterior, global_step, last_layer)

            # Update VAE
            opt_ae.zero_grad()
            losses['loss'].backward(retain_graph=loss_fn.use_disc and global_step >= loss_fn.disc_start_step)
            opt_ae.step()

            # Update discriminator
            if loss_fn.use_disc and global_step >= loss_fn.disc_start_step:
                opt_disc.zero_grad()

                # Recompute for discriminator
                with torch.no_grad():
                    recon_d, _ = vae(images)

                logits_real = loss_fn.discriminator(images)
                logits_fake = loss_fn.discriminator(recon_d)

                d_loss_real = torch.mean(F.relu(1. - logits_real))
                d_loss_fake = torch.mean(F.relu(1. + logits_fake))
                d_loss = 0.5 * (d_loss_real + d_loss_fake)

                d_loss.backward()
                opt_disc.step()

            ema.update()
            global_step += 1

            pbar.set_postfix({
                'rec': f"{losses['rec_loss'].item():.4f}",
                'kl': f"{losses['kl_loss'].item():.6f}",
                'lpips': f"{losses['lpips_loss'].item():.4f}" if loss_fn.use_lpips else 'N/A',
            })

        # Save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            ema.apply_shadow()
            torch.save({
                'vae_state_dict': vae.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
            }, f'{save_dir}/vae_epoch_{epoch+1}.pt')
            ema.restore()

            # Save sample reconstructions
            vae.eval()
            with torch.no_grad():
                sample_imgs = images[:16]
                recon_imgs, _ = vae(sample_imgs)
                comparison = torch.cat([sample_imgs, recon_imgs])
                save_image(comparison * 0.5 + 0.5, f'{save_dir}/vae_recon_epoch_{epoch+1}.png', nrow=16)

    return vae

def train_diffusion(
    ldm: LatentDiffusion,
    train_loader: DataLoader,
    test_loader: DataLoader = None,
    num_epochs: int = 500,
    lr: float = 1e-4,
    save_dir: str = './checkpoints',
    device: str = 'cuda',
    eval_every: int = 50,
    fid_num_samples: int = 10000,
):
    """Train the diffusion model (with frozen VAE)."""
    os.makedirs(save_dir, exist_ok=True)

    ldm = ldm.to(device)

    # Freeze VAE
    for param in ldm.vae.parameters():
        param.requires_grad = False
    ldm.vae.eval()

    # Optimizer for diffusion model
    optimizer = torch.optim.AdamW(
        ldm.diffusion.model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(train_loader))
    ema = EMA(ldm.diffusion.model, decay=0.9999)

    global_step = 0

    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            # Encode to latent
            with torch.no_grad():
                z = ldm.encode(images)

            # Diffusion loss
            loss = ldm.diffusion.loss(z, y=labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ldm.diffusion.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ema.update()

            global_step += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Evaluate FID every eval_every epochs
        if (epoch + 1) % eval_every == 0:
            # Apply EMA weights for evaluation
            ema.apply_shadow()

            # Save checkpoint
            torch.save({
                'diffusion_state_dict': ldm.diffusion.model.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
            }, f'{save_dir}/diffusion_epoch_{epoch+1}.pt')

            # Evaluate FID
            if test_loader is not None:
                fid_score = evaluate_fid(
                    ldm, test_loader,
                    num_samples=fid_num_samples,
                    batch_size=128,
                    device=device
                )
                print(f"Epoch {epoch+1}: FID = {fid_score:.2f}" if fid_score is not None else f"Epoch {epoch+1}: FID not calculated (torchmetrics[image] not installed)")

            # Generate and save sample images
            ldm.eval()
            with torch.no_grad():
                labels = torch.arange(10, device=device).repeat(5)[:16]
                samples = ldm.sample(16, y=labels, cfg_scale=2.0, ddim_steps=50, device=device)
                save_image(samples * 0.5 + 0.5, f'{save_dir}/samples_epoch_{epoch+1}.png', nrow=8)

            # Restore original weights for continued training
            ema.restore()

    # Final evaluation with EMA
    ema.apply_shadow()
    if test_loader is not None:
        final_fid = evaluate_fid(ldm, test_loader, num_samples=fid_num_samples, batch_size=128, device=device)
        print(f"Final FID: {final_fid:.2f}" if final_fid is not None else "Final FID not calculated (torchmetrics[image] not installed)")

    return ldm

# ============================================================================
# FID CALCULATION
# ============================================================================

def calculate_fid(real_images, fake_images, device='cuda'):
    """
    Calculate FID score between real and fake images.
    Uses torchmetrics FID if available, otherwise uses manual calculation.
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance

        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

        # Convert from [-1, 1] to [0, 255] uint8
        real_uint8 = ((real_images * 0.5 + 0.5) * 255).clamp(0, 255).to(torch.uint8)
        fake_uint8 = ((fake_images * 0.5 + 0.5) * 255).clamp(0, 255).to(torch.uint8)

        fid.update(real_uint8, real=True)
        fid.update(fake_uint8, real=False)

        return fid.compute().item()
    except ImportError:
        print("torchmetrics not available. Please install: pip install torchmetrics[image]")
        return None

@torch.no_grad()
def evaluate_fid(
    ldm,
    dataloader,
    num_samples=10000,
    batch_size=128,
    device='cuda',
    fid_batch=32,   # <-- smaller = less VRAM (try 16/32/64)
    verbose=False
):
    """Evaluate FID on CIFAR-10 in a memory-safe streaming way."""
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        print("torchmetrics not available. Please install: pip install torchmetrics[image]")
        return None

    ldm.eval()

    # Optional: reduce fragmentation before the eval spike
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # -------------------
    # 1) Update with real images (stream from dataloader)
    # -------------------
    seen = 0
    for images, _ in dataloader:
        if seen >= num_samples:
            break

        images = images[: max(0, num_samples - seen)]  # trim last batch
        b = images.size(0)
        if b == 0:
            break

        real_uint8 = ((images * 0.5 + 0.5) * 255).clamp(0, 255).to(torch.uint8).to(device)

        # Update FID in small chunks to keep Inception activations small
        for i in range(0, b, fid_batch):
            fid.update(real_uint8[i:i + fid_batch], real=True)

        seen += b

    # -------------------
    # 2) Update with fake images (generate + update in chunks)
    # -------------------
    generated = 0
    while generated < num_samples:
        cur = min(batch_size, num_samples - generated)

        labels = torch.randint(0, 10, (cur,), device=device)
        samples = ldm.sample(cur, y=labels, cfg_scale=2.0, ddim_steps=50, device=device)

        fake_uint8 = ((samples * 0.5 + 0.5) * 255).clamp(0, 255).to(torch.uint8)

        for i in range(0, cur, fid_batch):
            fid.update(fake_uint8[i:i + fid_batch], real=False)

        # aggressively release temp tensors
        del samples, fake_uint8, labels
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        generated += cur

    return fid.compute().item()

def _fid_uint8_from_minus1_1(x: torch.Tensor) -> torch.Tensor:
    # x: float in [-1, 1]
    return ((x * 0.5 + 0.5) * 255.0).clamp(0, 255).to(torch.uint8)

def _make_fid_metric(device: str):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except Exception:
        return None
    return FrechetInceptionDistance(feature=2048, normalize=True).to(device)

@torch.no_grad()
def evaluate_vae_recon_fid(
    vae,
    dataloader,
    num_samples: int = 10000,
    fid_batch_size: int = 32,
    device: str = "cuda",
    deterministic: bool = False,  # False => z ~ N(mu_x, Sigma_x); True => z = mu_x
):
    vae.eval()
    fid = _make_fid_metric(device)
    if fid is None:
        print("torchmetrics not available. Install with: pip install torchmetrics[image]")
        return None

    seen = 0
    for images, _ in dataloader:
        if seen >= num_samples:
            break
        images = images.to(device)

        for i in range(0, images.shape[0], fid_batch_size):
            if seen >= num_samples:
                break
            x = images[i:i + fid_batch_size]
            if seen + x.shape[0] > num_samples:
                x = x[:(num_samples - seen)]

            # Real
            fid.update(_fid_uint8_from_minus1_1(x), real=True)

            # Recon: z is sampled unless deterministic=True
            z = vae.get_latent(x, deterministic=deterministic)
            xhat = vae.decode_latent(z).clamp(-1.0, 1.0)
            fid.update(_fid_uint8_from_minus1_1(xhat), real=False)

            seen += x.shape[0]

    return float(fid.compute().item())
    
# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training script."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 128
    vae_epochs = 100
    diffusion_epochs = 500

    # Data
    train_loader = get_cifar10_dataloader(batch_size=batch_size, train=True)
    test_loader = get_cifar10_dataloader(batch_size=batch_size, train=False)

    # ================
    # Stage 1: Train VAE
    # ================
    print("\n" + "="*50)
    print("Stage 1: Training VAE")
    print("="*50)

    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4),  # f=4 downsampling: 32->8
        num_res_blocks=2,
        latent_channels=4,
        attention_resolutions=(8,),
    )

    vae_loss = VAELoss(
        kl_weight=1e-6,
        lpips_weight=1.0,
        recon_weight=1.0,
        disc_weight=0.5,
        disc_start_step=30000,  # Start discriminator after 30k steps
        use_lpips=True,
        use_disc=True,
    )

    print(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")

    vae = train_vae(
        vae=vae,
        loss_fn=vae_loss,
        train_loader=train_loader,
        num_epochs=vae_epochs,
        lr=4.5e-6,
        save_dir='./checkpoints/vae',
        device=device,
    )

    print("\n" + "-"*50)
    print("Evaluating VAE reconstruction FID (posterior samples)")
    print("-"*50)
    
    recon_fid = evaluate_vae_recon_fid(
        vae=vae,
        dataloader=test_loader,
        num_samples=10000,
        fid_batch_size=32,   # drop to 16/8 if you still see OOM
        device=device,
        deterministic=False, # IMPORTANT: posterior samples
    )
    print(f"VAE Recon FID (sample z): {recon_fid:.2f}" if recon_fid is not None else "Recon FID not computed")
    
    # optional reference: mean-code recon
    recon_fid_mu = evaluate_vae_recon_fid(vae, test_loader, 10000, 32, device, deterministic=True)
    print(f"VAE Recon FID (mean mu):  {recon_fid_mu:.2f}" if recon_fid_mu is not None else "")

    # ================
    # Stage 2: Train Diffusion
    # ================
    print("\n" + "="*50)
    print("Stage 2: Training Diffusion Model")
    print("="*50)

    unet = UNet(
        in_channels=4,
        out_channels=4,
        base_channels=128,
        channel_mults=(1, 2, 4),
        num_classes=10,
        dropout=0.1,
        attention_resolutions=(4, 2),
    )

    diffusion = GaussianDiffusion(
        model=unet,
        timesteps=1000,
        beta_schedule='cosine',
    )

    ldm = LatentDiffusion(vae=vae, diffusion=diffusion)

    print(f"UNet parameters: {sum(p.numel() for p in unet.parameters()):,}")

    ldm = train_diffusion(
        ldm=ldm,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=diffusion_epochs,
        lr=1e-4,
        save_dir='./checkpoints/diffusion',
        device=device,
        eval_every=50,
        fid_num_samples=10000,
    )

    return ldm

if __name__ == '__main__':
    main()
