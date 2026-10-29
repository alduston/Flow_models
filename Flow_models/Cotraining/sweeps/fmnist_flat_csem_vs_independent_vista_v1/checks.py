"""Meaningful protocol, gradients, freezing, replay, and inference checks."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
import torch
import models
import protocol
from pinned import stacked_two_block as old
from train_stages import run as train,prior_loss,streams,kl0,state_hash

def small(base,name,stage='joint',source=None):
    a=protocol.resolve(dict(name=name,stage=stage,seed=42,beta0=.03,group='check'),base,device='cpu')
    a.joint_width=48;a.joint_depth=2;a.batch=2;a.num_workers=0;a.cpu_threads=2
    a.epochs=2;a.max_batches=1;a.perc_weight=0.;a.gan_weight=0.
    a.eval_samples=4;a.eval_batch=2;a.feature_batch=2;a.rk4_steps=2;a.eval_every=1
    a.archive_epochs=[1,2];a.kid_subsets=0
    if source and stage.startswith('prior'):a.source=source
    return a

def tensor_equal(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:tensor_equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):tensor_equal(x,y)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    else:assert a==b

def model_checks(base,device):
    torch.set_num_threads(2);dev=torch.device(device)
    args=argparse.Namespace(**protocol.common(base));args.seed=42
    old.seed_everything(args.seed);vae,head=models.build(args,dev)
    audit=models.parameter_audit(vae,head)
    assert audit['vae']==6368817 and audit['score_head']==12159556
    assert vae.stochastic_blocks==1 and not head.factored_head
    # Verify deterministic parent anchors, independent of sampled ancestors.
    x=torch.randn(2,1,32,32,device=dev);labels=torch.tensor([0,9],device=dev)
    d1=vae.draw(x);d2=vae.draw(x)
    tensor_equal(d1['params'],d2['params'])
    z=torch.arange(512.,device=dev).reshape(2,4,8,8)
    assert torch.equal(models.pack(models.unpack(z,3)),z)
    schedule=old.ref.make_schedule(dict(time_schedule='log_t',t_min=args.t_min,t_max=args.t_max,num_train_timesteps=1000),dev)
    # CSEM formula and exact time-zero KL, including unequal storage-group sizes.
    t=torch.tensor([.01,1.],device=dev);d=models.draw_noisy(vae,x,t)
    a,s=models.coeff(t)
    for y,(m,lv),c in zip(d['y'],d['params'],d['naive']):
        torch.testing.assert_close(c,s*(y-a*m)/(a*a*lv.exp()+s*s),rtol=1e-5,atol=1e-6)
    teacher=models.corrected_teacher(vae,d)
    for u,v in zip(teacher,d['naive']):torch.testing.assert_close(u,v.detach(),rtol=2e-5,atol=2e-6)
    mu=torch.cat([m.flatten(1) for m,l in d['params']],1)
    lv=torch.cat([l.flatten(1) for m,l in d['params']],1)
    torch.testing.assert_close(kl0(d['params'],vae.dimensions),.5*(mu.square()+lv.exp()-1-lv).mean())
    # Activate head input gradients (the inherited final projection starts at zero).
    with torch.no_grad():head.final_proj.weight.normal_(0,.002)
    gradients=[]
    for r in (1.,.5):
        args.rep_multiplier=r;vae.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True)
        old.seed_everything(900)
        score,*_=models.training_losses(args,vae,head,x,labels,schedule)
        score.backward()
        gradients.append(([p.grad.clone() if p.grad is not None else None for p in vae.parameters()],
                          [p.grad.clone() if p.grad is not None else None for p in head.parameters()],float(score)))
    assert gradients[0][2]==gradients[1][2]
    for g,h in zip(gradients[0][0],gradients[1][0]):
        if g is None:assert h is None
        else:torch.testing.assert_close(h,.5*g,rtol=5e-5,atol=1e-7)
    tensor_equal(gradients[0][1],gradients[1][1])
    # Frozen objective pair sees identical data/time/eta/eps/dropout, fresh equal heads.
    vae.eval().requires_grad_(False);vae.zero_grad(set_to_none=True)
    before=state_hash(vae.state_dict());draws=[]
    for stage in ('prior_dsm','prior_csem'):
        args.stage=stage;head.zero_grad(set_to_none=True)
        loss,draw,drop=prior_loss(args,vae,head,x,labels,schedule,streams(42,dev),True)
        loss.backward();assert not any(p.grad is not None for p in vae.parameters())
        draws.append((draw['packed'],draw['eps'],draw['time'],drop))
    tensor_equal(draws[0],draws[1]);assert state_hash(vae.state_dict())==before
    return dict(parameters=audit,encoder_routing='VAE score gradients exactly half; head gradients identical',
                analytic_teacher='exact flat Gaussian',posterior_ancestry='sample independent',frozen_vae='all tensors and gradients verified')

def resume_checks(base,device):
    # Temporary results are intentionally excluded from production and final compilation.
    with tempfile.TemporaryDirectory(prefix='fmnist-check-',dir=base) as temp:
        root=Path(temp)
        # Sources/recipe lookup is rooted at bundle, results may be temporary.
        stages={}
        parent_name=None
        parent_path=None
        try:
            for stage in ('vae_pretrain','joint','prior_dsm','prior_csem'):
                name=f'check_{stage}'
                a=small(base,name,stage,parent_name);a.device=device
                a.output=str(root/name)
                full=copy.deepcopy(a);full.output=str(root/(name+'_uninterrupted'))
                assert train(full,base,synthetic=True)==0
                assert train(a,base,synthetic=True,stop_after=1)==75
                assert train(a,base,synthetic=True,resume=True)==0
                snapshot=Path(a.output)/'snapshots/epoch_0002.pt'
                frozen_snapshot=protocol.sha(snapshot)
                assert train(a,base,synthetic=True,resume=True)==0
                assert protocol.sha(snapshot)==frozen_snapshot
                one=torch.load(Path(full.output)/'latest.pt',map_location='cpu',weights_only=False)
                two=torch.load(Path(a.output)/'latest.pt',map_location='cpu',weights_only=False)
                for key in ('vae','head','ema','optimizer','scheduler','named_rng','data_rng','rng'):
                    tensor_equal(one[key],two[key])
                stages[stage]='uninterrupted and resumed model/optimizer/EMA/RNG tensors identical; reloaded inference verified'
                if stage=='vae_pretrain':
                    # Match production: children use a real immutable parent directory.
                    parent_name=f'check_source_{root.name}';parent_path=Path(base)/'results'/parent_name
                    shutil.copytree(Path(a.output),parent_path)
        finally:
            if parent_path is not None and parent_path.exists():shutil.rmtree(parent_path)
        return stages

def run(base,device='cpu',models_only=False):
    base=Path(base);result=dict(model_checks=model_checks(base,device))
    if not models_only:result['resume_checks']=resume_checks(base,device)
    result['device']=device;result['production_gpu_training_performed']=False
    protocol.atomic(base/'validation'/f'checks_{device}.json',result)
    print(json.dumps(result,indent=2));return 0

if __name__=='__main__':run(protocol.ROOT)
