"""Coherent direct hierarchical posterior; one joint epsilon field in packed coordinates."""
import torch
from torch import nn
import core
from direct_models import DirectHierarchicalVAE,pack_root,unpack_root

class StackedVAE(DirectHierarchicalVAE):
    def __init__(self,*,topology='full',**kwargs):
        super().__init__(**kwargs);self.topology=topology
    def root(self,x):
        h=self.enc_conv_in(x)
        for b in self.enc_blocks:h=b(h)
        if self.enc_pre_proj is not None:h=self.enc_pre_proj(h)
        m,l=self.root_moments(self.root_down(h)).chunk(2,1)
        return self.anchor(m),l.clamp(-30,20),h
    def child(self,h,root,root_mean):
        context=root if self.topology=='full' else root_mean
        m,l=self.child_moments(torch.cat((h,self.ancestor_features(context)),1)).chunk(2,1)
        return self.anchor(m),l.clamp(-30,20)
    def encode_sample(self,x,*,eps=None):
        m,l,h=self.root(x)
        e=torch.randn((len(x),8,8,8),device=x.device,dtype=x.dtype) if eps is None else eps
        r=m+(.5*l).exp()*unpack_root(e[:,:2])
        cm,cl=self.child(h,r,m);c=cm+(.5*cl).exp()*e[:,2:]
        return torch.cat((pack_root(m),cm),1),torch.cat((pack_root(l),cl),1),torch.cat((pack_root(r),c),1)
    def draw(self,x):
        m,l,h=self.root(x);r=m+(.5*l).exp()*torch.randn_like(m)
        cm,cl=self.child(h,r,m);c=cm+(.5*cl).exp()*torch.randn_like(cm)
        return dict(z=torch.cat((pack_root(r),c),1),mu=torch.cat((pack_root(m),cm),1),
                    lv=torch.cat((pack_root(l),cl),1),h=h,root_mu=m,root_lv=l)

class FlatVAE(core.VAE):
    def encode_sample(self,x,*,eps=None):
        m,l=self.encode(x);l=l.clamp(-30,20)
        e=torch.randn_like(m) if eps is None else eps
        return m,l,m+(.5*l).exp()*e
    def draw(self,x):
        m,l,z=self.encode_sample(x);return dict(mu=m,lv=l,z=z)

def vae_factory(cfg,**kwargs):
    if cfg['blocks']==1:return FlatVAE(**kwargs)
    # Consume exactly the flat baseline construction RNG before any head is built.
    # Hierarchy construction runs under a CPU RNG fork (all modules initialize on CPU).
    scaffold=FlatVAE(**kwargs);del scaffold
    with torch.random.fork_rng(devices=[]):
        return StackedVAE(block_channels=[2,6],root_size=4,topology=cfg['topology'],**kwargs)

def noisy(vae,x,t,*,draw=None,noise=None):
    d=vae.draw(x) if draw is None else draw
    a,s=core.get_ou_params(t[:,None,None,None]);e=torch.randn_like(d['z']) if noise is None else noise
    y=a*d['z']+s*e;S=a.square()*d['lv'].exp()+s.square()
    d.update(y=y,eps=e,time=t,alpha=a,sigma=s,naive=s*(y-a*d['mu'])/(S+1e-8))
    return d

def corrected(vae,d):
    """Fisher identity with conditional posterior transport, copied algebraically from FMNIST.

    Valid conditional mean in epsilon units; not a finite-bank exact score.
    Only used on frozen representations or held-out probes.
    """
    if not isinstance(vae,StackedVAE):return d['naive'].detach()
    with torch.enable_grad():
        y=d['y'].detach().requires_grad_(True);a=d['alpha'].detach();s=d['sigma'].detach()
        yr,yc=unpack_root(y[:,:2]),y[:,2:]
        m,l=d['root_mu'].detach(),d['root_lv'].detach();v=l.exp();S=a.square()*v+s.square()
        pm=m+a*v/S*(yr-a*m);pv=v*s.square()/S
        eta=(unpack_root(d['z'][:,:2]).detach()-pm.detach())/pv.detach().sqrt()
        r=pm+pv.sqrt()*eta
        cm,cl=vae.child(d['h'].detach(),r,m);CS=a.square()*cl.exp()+s.square()
        log=-.5*((yr-a*m).square()/S+S.log()).flatten(1).sum(1)
        log=log-.5*((yc-a*cm).square()/CS+CS.log()).flatten(1).sum(1)
        answer=-s*torch.autograd.grad(log.sum(),y)[0]
    return answer.detach()

def build_models(cfg,device):
    # Extract the pinned trainer's constructor expressions to avoid a second recipe.
    import ast,inspect
    tree=ast.parse(inspect.getsource(core.train_vae_cotrained_cond));assign={}
    for n in ast.walk(tree):
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            if n.targets[0].id in ('vae','dit_kwargs'):assign[n.targets[0].id]=n.value
    ns=dict(core.__dict__,cfg=cfg,device=device,effective_img_size=32,img_channels=3,num_classes=10,latent_spatial=8,
            VAE=lambda **kw:vae_factory(cfg,**kw))
    vae=eval(compile(ast.Expression(assign['vae']),'<pinned-vae>','eval'),ns)
    kw=eval(compile(ast.Expression(assign['dit_kwargs']),'<pinned-dit>','eval'),ns)
    return vae,core.DiTModel(**kw).to(device)
