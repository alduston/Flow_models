"""One joint score head; coherent two/three-block Gaussian posterior DAGs.

All configurations have 256 latent coordinates packed by a permutation into
4x8x8: 64+192 or 64+64+128. Posterior topology is varied; the joint head always
observes ALL noisy coordinates and one class label. This is not a cascade prior.
"""
from __future__ import annotations
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from pinned import stacked_two_block as baseline
from vae_three import HierarchicalFMNISTVAE as VAE3

VERSION = 'joint_stacked_mechanism_v1'


def pack(blocks):
    return torch.cat([F.pixel_shuffle(blocks[0], 2), *blocks[1:]], 1)


def unpack(value, blocks=2):
    return [F.pixel_unshuffle(value[:, :1], 2), value[:, 1:]] if blocks == 2 else [
        F.pixel_unshuffle(value[:, :1], 2), value[:, 1:2], value[:, 2:]]


def flat(value):
    return value.flatten(1)


def per_sample(value):
    return flat(value).mean(1)


def coeff(t):
    return torch.exp(-t)[:, None, None, None], torch.sqrt(-torch.expm1(-2*t))[:, None, None, None]


def noise_like(value):
    return torch.randn_like(value)


def opts():
    p = baseline.ref.DATASET_PRESETS['fmnist_reference']
    keys = ('base_ch', 'num_res_blocks', 'decoder_attn_half', 'latent_proj_depth',
            'encoder_attn_half', 'decoder_extra_block', 'conv3x3_proj', 'use_tanh_out',
            'clamp_logvar', 'attn_zero_init')
    return dict({k:p[k] for k in keys}, use_norm=False, img_size=32, img_channels=1,
                time_cond_decoder=True, dec_time_emb_dim=128)


class Posterior(nn.Module):
    def __init__(self, net, blocks=2, topology='full'):
        super().__init__()
        self.net, self.blocks, self.topology = net, blocks, topology
        self.shapes = [(4,4,4),(3,8,8)] if blocks == 2 else [(4,4,4),(1,8,8),(2,8,8)]
        self.dimensions = [math.prod(s) for s in self.shapes]
        self.default_gates = {'full':(1.,1.,1.), 'decoupled':(0.,0.,0.),
                              'markov':(1.,0.,1.), 'siblings':(1.,1.,0.)}[topology]

    def root(self, x):
        return self.net.encode_root(x)

    @staticmethod
    def mix(parent, anchor, gate):
        # Preserve the original arithmetic exactly for full/absent edges.
        if not torch.is_tensor(gate):
            if gate == 1: return parent
            if gate == 0: return anchor
        return anchor+gate*(parent-anchor)

    def moment(self, index, features, root_mu, root_lv, parents, gates=None):
        if index == 0: return root_mu, root_lv
        g12,g13,g23 = self.default_gates if gates is None else gates
        if index == 1:
            a = self.mix(parents[0], root_mu, g12)
            return self.net.encode_conditional(features, a)
        a = self.mix(parents[0], root_mu, g13)
        b = parents[1]
        if torch.is_tensor(g23) or g23 != 1:
            anchor, _ = self.net.encode_conditional(features, a)
            b = self.mix(b, anchor, g23)
        return self.net.encode_conditional3(features, a, b)

    def draw(self, x, innovations=None, gates=None):
        mu, lv, features = self.root(x)
        zs, params, noises = [], [], []
        for i in range(self.blocks):
            m, l = self.moment(i, features, mu, lv, zs, gates)
            e = noise_like(m) if innovations is None else innovations[i]
            zs.append(m+torch.exp(.5*l)*e); params.append((m,l)); noises.append(e)
        return dict(z=zs, params=params, innovations=noises, features=features,
                    root_mu=mu, root_lv=lv, x=x)

    def decode(self, zs, t=None):
        return self.net.decode(*zs, t=t)


def build(args, device):
    # Head initialization matches the delivered two-block run in every arm.
    head_scaffold=copy.copy(args)
    head_scaffold.z1_channels=4;head_scaffold.z2_channels=3
    original, head = baseline.build(head_scaffold, device)
    if args.blocks == 2:
        net = original
    else:
        del original
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            torch.manual_seed(args.seed+71001)
            net = VAE3(4,1,2, **opts()).to(device)
    return Posterior(net, args.blocks, args.topology).to(device), head


def draw_noisy(vae, x, t, *, innovations=None, diffusion=None, gates=None):
    d = vae.draw(x, innovations, gates)
    return corrupt_draw(d,t,diffusion=diffusion)


def corrupt_draw(d, t, diffusion=None, alpha=None, sigma=None):
    a,s = coeff(t) if alpha is None else (alpha,sigma)
    eps = [noise_like(z) for z in d['z']] if diffusion is None else diffusion
    ys = [a*z+s*e for z,e in zip(d['z'],eps)]
    cs = [s*((y-a*m)/(a.square()*lv.exp()+s.square()).clamp_min(1e-10))
          for y,(m,lv) in zip(ys,d['params'])]
    d.update(y=ys, eps=eps, naive=cs, time=t, alpha=a, sigma=s, packed=pack(ys))
    return d


def corrected_teacher(vae, d, *, components=False):
    """Valid epsilon teacher by sequential conditional-posterior transport.

    Hold X, observed Y, and inferred posterior innovations fixed. Reconstruct
    each clean parent with Z_i|X,Z_<i,Y_i (Gaussian), then differentiate the sum
    of noisy Gaussian log factors with respect to Y. Reparameterization accounts
    for ALL descendant mean/variance paths, including the indirect 1->2->3 path.
    E[Tcorr|Y,C] = E[eps|Y,C]. No encoder training uses these derivatives.
    """
    with torch.enable_grad():
        ys = [y.detach().requires_grad_(True) for y in d['y']]
        mu,lv = d['root_mu'].detach(),d['root_lv'].detach()
        feat = d['features'].detach()
        a,s = d['alpha'].detach(),d['sigma'].detach()
        parents, logs = [], []
        for i,y in enumerate(ys):
            m,l = vae.moment(i,feat,mu,lv,parents)
            v = l.exp(); variance = a.square()*v+s.square()
            logs.append(-.5*(flat((y-a*m).square()/variance+variance.log()).sum(1)))
            if i < vae.blocks-1:
                post_m = m+a*v/variance*(y-a*m)
                post_v = v*s.square()/variance
                eta = (d['z'][i].detach()-post_m.detach())/post_v.detach().sqrt()
                parents.append(post_m+post_v.sqrt()*eta)
        result = [-s*g for g in torch.autograd.grad(sum(q.sum() for q in logs),ys,
                                                    retain_graph=components)]
        contributions = {}
        if components:
            for j in range(1,vae.blocks):
                gs = torch.autograd.grad(logs[j].sum(), ys[:j], retain_graph=True, allow_unused=True)
                for i,g in enumerate(gs):
                    contributions[f'{i+1}_from_{j+1}'] = torch.zeros_like(ys[i]) if g is None else (-s*g).detach()
    result = [r.detach() for r in result]
    return (result, contributions) if components else result


def scaled_gradient(value, multiplier):
    if multiplier == 1: return value
    return value.detach()+multiplier*(value-value.detach())


def training_losses(args, vae, head, x, labels, schedule, *, indices=None):
    d = vae.draw(x)
    if indices is None:
        indices = torch.randint(len(schedule['times']), (len(x),), device=x.device)
    t = schedule['times'][indices]
    d = corrupt_draw(d,t,alpha=schedule['alpha'][indices].view(-1,1,1,1),
                     sigma=schedule['sigma'][indices].view(-1,1,1,1))
    return loss_from_draw(args,vae,head,d,labels,schedule)


def loss_from_draw(args, vae, head, d, labels, schedule):
    drop = labels.clone()
    drop[torch.rand(len(labels),device=labels.device) < args.label_dropout] = 10
    factor = args.rep_multiplier
    # Scale ALL input/target/ancestry routes into the VAE while leaving head
    # derivatives and its optimizer LR untouched. Only used at this loss boundary.
    yy = scaled_gradient(d['packed'],factor)
    target = scaled_gradient(pack(d['naive']),factor)
    prediction = head(yy,d['time'],drop)
    score = F.mse_loss(prediction,target)
    perblock = [F.mse_loss(p,c) for p,c in zip(unpack(prediction,vae.blocks),d['naive'])]
    weights = [n/256 for n in vae.dimensions]
    terminal = sum(w*baseline.terminal_kl(m,l,schedule['alpha'][-1],schedule['sigma'][-1])
                   for w,(m,l) in zip(weights,d['params']))
    floor = baseline.tmin_band_floor(schedule)
    band = sum(w*baseline.tmin_band_penalty(l,floor) for w,(m,l) in zip(weights,d['params']))
    image = vae.decode(d['y'],d['time'])
    d.update(prediction=prediction,score_per_block=perblock,reconstruction=image)
    return score,terminal,band,d
