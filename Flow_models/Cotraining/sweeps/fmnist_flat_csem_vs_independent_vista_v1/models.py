"""Capacity-matched posterior depth, anchored to b2_full_r0.5.

`blocks` is the storage factor count used by inherited probes and sampling.
`stochastic_blocks` is stochastic depth. Decoupled factors together form ONE
image-conditional diagonal Gaussian over all 256 coordinates. The multiscale
image decoder is retained deliberately to control architecture and capacity.
"""
import math
import torch
from torch import nn
from pinned import stacked_two_block as baseline
import original_models as original
from original_models import (pack, unpack, flat, per_sample, coeff, noise_like, opts,
    Posterior, draw_noisy, corrupt_draw, corrected_teacher, scaled_gradient,
    training_losses, loss_from_draw)

VERSION = 'fmnist_capacity_matched_v1'
# Selected by analytic parameter arithmetic, without training or FID selection.
THREE_WIDTHS = (87, 75)


def conditioner(inputs, hidden, channels):
    result = nn.Sequential(
        baseline.CascadeResBlock(inputs, hidden),
        baseline.CascadeResBlock(hidden, hidden),
        nn.GroupNorm(math.gcd(hidden, 16), hidden), nn.SiLU(),
        nn.Conv2d(hidden, 2*channels, 3, padding=1))
    nn.init.normal_(result[-1].weight, std=1e-3)
    nn.init.zeros_(result[-1].bias)
    return result


class BudgetThreeVAE(baseline.HierarchicalFMNISTVAE):
    """Reallocate the reference child-posterior budget into two conditioners.

    Root: 64 coordinates at 4x4. Children: 64 and 128 coordinates at 8x8.
    The third conditioner sees X features, root features, and the sampled second
    code. All decoder tensors/operations are the original two-block decoder,
    receiving the concatenated child codes. No unused parameters are added.
    """
    def __init__(self, reference):
        nn.Module.__init__(self)
        for name, module in reference.named_children():
            self.add_module(name, module)
        for name in ('use_tanh_out','time_cond_decoder','dec_time_emb_dim'):
            if hasattr(reference, name):setattr(self, name, getattr(reference, name))
        self.z1_channels, self.z2_channels, self.z3_channels = 4, 1, 2
        self.latent_shapes = ((4,4,4),(1,8,8),(2,8,8))
        width = reference.root_in.out_channels
        self.condition = conditioner(2*width, THREE_WIDTHS[0], 1)
        self.condition3 = conditioner(2*width+1, THREE_WIDTHS[1], 2)

    def encode_conditional3(self, image_features, z1, z2, top_down=None):
        if top_down is None:top_down = self.ancestor_features(z1)
        mu, lv = self.condition3(torch.cat((image_features,top_down,z2),1)).chunk(2,1)
        return mu, lv.clamp(-30,20)

    def decode(self, z1, z2, z3, t=None, *, top_down=None):
        return super().decode(z1,torch.cat((z2,z3),1),t=t,top_down=top_down)


def args_for_reference(args):
    import copy
    value = copy.copy(args)
    value.blocks = 2
    value.topology = 'full'
    return value


def build(args, device):
    # Preserve reference VAE/head initialization and the following RNG state.
    scaffold, head = original.build(args_for_reference(args), device)
    if args.blocks == 2:
        vae = Posterior(scaffold.net, 2, args.topology).to(device)
    elif args.blocks == 3:
        devices = [device.index or 0] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(args.seed+71001)
            net = BudgetThreeVAE(scaffold.net).to(device)
        vae = Posterior(net, 3, args.topology).to(device)
    else:
        raise ValueError('Storage factor count must be 2 or 3')
    vae.stochastic_blocks = args.stochastic_blocks
    return vae, head


def parameter_audit(vae, head):
    groups = dict(bottom_up_encoder=0, conditional_posteriors=0,
                  shared_root_features=0, decoder=0)
    for name,p in vae.net.named_parameters():
        if name.startswith(('enc_','root_posterior')):key='bottom_up_encoder'
        elif name.startswith('condition'):key='conditional_posteriors'
        elif name.startswith(('root_in','root_blocks','root_up')):key='shared_root_features'
        else:key='decoder'
        groups[key] += p.numel()
    groups['vae'] = sum(p.numel() for p in vae.parameters())
    groups['score_head'] = sum(p.numel() for p in head.parameters())
    groups['total_vae_plus_score'] = groups['vae']+groups['score_head']
    groups['stochastic_blocks'] = vae.stochastic_blocks
    groups['storage_factors'] = vae.blocks
    groups['latent_coordinates'] = sum(vae.dimensions)
    return groups
