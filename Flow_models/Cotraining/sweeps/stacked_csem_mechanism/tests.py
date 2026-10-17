"""Analytic identities and real-network software checks; no claims of trained FID."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import tempfile
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiments
import models
import trainer
import measurements
import witness
import reference
from pinned import stacked_two_block as old
from gaussian_sandbox import matrices


def small(row,base,device='cpu'):
    a=experiments.configuration(row,base,device=device)
    a.joint_width=48;a.joint_depth=2;a.batch=2;a.num_workers=0;a.cpu_threads=2
    a.epochs=2;a.max_batches=1;a.perc_weight=0.;a.gan_weight=0.
    a.eval_samples=4;a.eval_batch=2;a.feature_batch=2;a.rk4_steps=2
    a.diagnostic_every=1;a.diagnostic_samples=2;a.diagnostic_particles=4;a.diagnostic_batch=2
    a.diagnostic_times=[.2];a.witness_every=2;a.witness_train=12;a.witness_test=8;a.witness_features=4
    a.sampling_check_every=0;a.checkpoint_every=1;a.archive_epochs=[1,2]
    return a


def same(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for u,v in zip(a,b):same(u,v)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    else:assert a==b


class Toy(nn.Module):
    def __init__(self,beta,variances,nonlinear=False):
        super().__init__();self.beta=torch.as_tensor(beta,dtype=torch.float64);self.variances=variances
        self.blocks=len(variances);self.nonlinear=nonlinear
    def moment(self,i,features,mu,lv,parents):
        if i==0:return mu,lv
        m=sum(self.beta[i,j]*parents[j] for j in range(i))
        l=torch.zeros_like(m)+math.log(self.variances[i])
        if self.nonlinear:
            m=m+.13*parents[0].sin()+.08*parents[0].square()
            l=l+.17*parents[-1]
        return m,l


def analytic_teachers():
    for blocks,top in [(2,'full'),(2,'decoupled'),(3,'full'),(3,'markov'),(3,'siblings'),(3,'decoupled')]:
        beta=np.zeros((blocks,blocks));beta[1,0]=.9
        if blocks==3:
            if top in ('full','siblings'):beta[2,0]=.5
            if top in ('full','markov'):beta[2,1]=.7
        if top=='decoupled':beta[:]=0
        variances=[.8,.6,.9][:blocks];t=.35
        cov,noisy,b,m=matrices(beta,variances,t)
        alpha=math.exp(-t);sigma=math.sqrt(-math.expm1(-2*t))
        yy=torch.tensor([[.4,-.8,.6][:blocks],[-.7,.3,.2][:blocks]],dtype=torch.float64)
        zm=yy@torch.tensor(alpha*cov@np.linalg.inv(noisy),dtype=torch.float64).T
        model=Toy(beta,variances)
        ys=[yy[:,i:i+1,None,None] for i in range(blocks)]
        zs=[zm[:,i:i+1,None,None] for i in range(blocks)]
        mu=torch.zeros_like(zs[0]);lv=mu+math.log(variances[0])
        d=dict(y=ys,z=zs,root_mu=mu,root_lv=lv,features=mu,
               alpha=mu+alpha,sigma=mu+sigma)
        valid=models.corrected_teacher(model,d)
        got=torch.cat([v.flatten(1) for v in valid],1)
        expected=yy@torch.tensor(m).T
        torch.testing.assert_close(got,expected,rtol=1e-10,atol=1e-10)
        assert np.max(np.abs(b[-1]-m[-1]))<1e-12
        if top=='decoupled':assert np.max(np.abs(b-m))<1e-12
        if blocks==3 and top=='siblings':assert np.max(np.abs(b[1]-m[1]))<1e-12
        if blocks==3 and top=='markov':assert np.linalg.norm(b[0]-m[0])>.001
    # Nonlinear means AND variances, checked against independent prior-space
    # Gauss-Hermite integration and a finite difference of the full joint density.
    beta=[[0,0,0],[.7,0,0],[.4,.8,0]];vv=[.8,.6,.9]
    model=Toy(beta,vv,True);t=.6;alpha=math.exp(-t);sigma=math.sqrt(-math.expm1(-2*t))
    nodes,weights=np.polynomial.hermite.hermgauss(48)
    n1,n2=np.meshgrid(nodes,nodes,indexing='ij');w1,w2=np.meshgrid(weights,weights,indexing='ij')
    a=torch.tensor((math.sqrt(2*vv[0])*n1).reshape(-1,1,1,1))
    mu=torch.zeros_like(a);lv=mu+math.log(vv[0])
    mb,lb=model.moment(1,mu,mu,lv,[a]);b=mb+torch.exp(.5*lb)*torch.tensor((math.sqrt(2)*n2).reshape(-1,1,1,1))
    mc,lc=model.moment(2,mu,mu,lv,[a,b]);s3=alpha**2*lc.exp()+sigma**2
    logw=torch.tensor(np.log(w1*w2/math.pi).reshape(-1))
    query=torch.tensor([.4,-.3,.7],dtype=torch.float64)
    def likelihood(q):
        return logw-.5*models.flat((q[0]-alpha*a).square()/sigma**2+(q[1]-alpha*b).square()/sigma**2+(q[2]-alpha*mc).square()/s3+s3.log()).sum(1)
    weights=torch.softmax(likelihood(query),0)
    d=dict(y=[mu+query[i] for i in range(3)],z=[a,b,mc],root_mu=mu,root_lv=lv,features=mu,alpha=mu+alpha,sigma=mu+sigma)
    valid=models.corrected_teacher(model,d)
    means=torch.stack([(v.flatten()*weights).sum() for v in valid])
    finite=[]
    for i in range(3):
        step=torch.zeros(3,dtype=torch.float64);step[i]=1e-5
        finite.append(-sigma*(torch.logsumexp(likelihood(query+step),0)-torch.logsumexp(likelihood(query-step),0))/(2e-5))
    torch.testing.assert_close(means,torch.stack(finite),rtol=2e-5,atol=2e-5)
    print('[PASS] corrected two/three-block teacher identities, DAG zero controls, nonlinear variance derivatives')


def parity_and_scaling(base):
    a=small(experiments.rows()[4],base);a.label_dropout=0.
    old.seed_everything(42);vae,head=models.build(a,torch.device('cpu'))
    original=copy.deepcopy(vae.net);h0=copy.deepcopy(head)
    with torch.no_grad():head.final_proj.weight.normal_(std=.02);h0.load_state_dict(head.state_dict())
    x=torch.rand(2,1,32,32)*2-1;labels=torch.tensor([2,5])
    schedule=old.ref.make_schedule(dict(time_schedule='log_t',t_min=a.t_min,t_max=a.t_max,num_train_timesteps=1000),'cpu')
    idx=torch.tensor([100,700]);old.seed_everything(44)
    so,ko,do=old.score_and_losses(a,original,h0,x,labels,schedule,indices=idx)
    old.seed_everything(44);sn,kn,bn,dn=models.training_losses(a,vae,head,x,labels,schedule,indices=idx)
    for u,v in [(so,sn),(ko,kn),(do['band_loss'],bn),(do['joint_input'],dn['packed']),(do['reconstruction'],dn['reconstruction'])]:
        torch.testing.assert_close(u,v,rtol=0,atol=0)
    go=torch.autograd.grad(so,list(original.parameters())+list(h0.parameters()),allow_unused=True)
    gn=torch.autograd.grad(sn,list(vae.parameters())+list(head.parameters()),allow_unused=True)
    for u,v in zip(go,gn):
        if u is None:assert v is None
        else:torch.testing.assert_close(u,v,rtol=0,atol=0)
    fixed=copy.deepcopy(vae.state_dict());hf=copy.deepcopy(head.state_dict())
    for factor in (0.,.5,2.,4.):
        a.rep_multiplier=factor;vae.load_state_dict(fixed);head.load_state_dict(hf);old.seed_everything(44)
        sc,*_=models.training_losses(a,vae,head,x,labels,schedule,indices=idx)
        gs=torch.autograd.grad(sc,list(vae.parameters())+list(head.parameters()),allow_unused=True)
        split=len(list(vae.parameters()))
        for j,(u,v) in enumerate(zip(gn,gs)):
            if u is None:assert v is None
            else:torch.testing.assert_close(v,u*(factor if j<split else 1),rtol=2e-4,atol=1e-7)
    print('[PASS] exact original two-block targets/losses/gradients; encoder-only scale leaves head gradients unchanged')


def zero_controls(base):
    for blocks in (2,3):
        row=next(r for r in experiments.rows() if r['stage']=='joint' and r['config']['blocks']==blocks and r['config']['topology']=='decoupled')
        args=small(row,base);old.seed_everything(42);vae,head=models.build(args,torch.device('cpu'))
        x=torch.rand(2,1,32,32)*2-1
        with torch.no_grad():d=models.draw_noisy(vae,x,torch.full((2,),.2))
        corrected=models.corrected_teacher(vae,d)
        for a,b in zip(corrected,d['naive']):torch.testing.assert_close(a,b,rtol=2e-5,atol=1e-6)
        zs=d['z'];torch.testing.assert_close(models.pack(zs).square().sum(),sum(z.square().sum() for z in zs))
        for a,b in zip(zs,models.unpack(models.pack(zs),blocks)):torch.testing.assert_close(a,b,rtol=0,atol=0)
    print('[PASS] real-network decoupled zero correction; lossless 256-coordinate packing in both depths')


def witness_contract():
    from types import SimpleNamespace
    a=SimpleNamespace(seed=42,witness_features=8,witness_ridge=.001)
    gen=torch.Generator().manual_seed(43)
    x=torch.randn(1024,256,generator=gen,dtype=torch.float64)
    y=torch.randn(4096,256,generator=gen,dtype=torch.float64)
    def test(coupling):
        tr=dict(y=x,delta=coupling*x+.05*torch.randn(x.shape,generator=gen,dtype=x.dtype),labels=torch.zeros(len(x),dtype=torch.long))
        te=dict(y=y,delta=coupling*y+.05*torch.randn(y.shape,generator=gen,dtype=y.dtype),labels=torch.zeros(len(y),dtype=torch.long))
        w=witness.fit_predict(tr,te,a,'null');score=(2*te['delta']*w-w.square()).mean(1)
        return float(score.mean()),float(score.std()/math.sqrt(len(y)))
    mean,se=test(.2);assert mean-1.96*se>.02 and mean<.041
    mean,se=test(0.);assert mean-1.96*se<=0
    print('[PASS] independent learned-witness lower bound detects known bias and respects zero-bias control')


def resume_contract(base):
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);row=experiments.rows()[4];a=small(row,base)
        a.epochs=3;a.eval_samples=0;a.diagnostic_at_start=False;a.diagnostic_every=0;a.witness_every=0
        a.gan_weight=.0025;a.disc_start=1;a.archive_epochs=[3]
        a.output=str(tmp/'continuous');assert trainer.train(a,base,synthetic=True)==0
        b=copy.copy(a);b.output=str(tmp/'resumed')
        assert trainer.train(b,base,synthetic=True,stop_after=1)==75
        # Stray uncommitted row is discarded on resume.
        with (Path(b.output)/'loss.csv').open('a') as f:f.write('99,0'+',0'*18+'\n')
        assert trainer.train(b,base,resume=True,synthetic=True)==0
        u=torch.load(Path(a.output)/'latest.pt',weights_only=False,map_location='cpu')
        v=torch.load(Path(b.output)/'latest.pt',weights_only=False,map_location='cpu')
        for key in ('vae','head','ema','disc','optimizer','scheduler','disc_optimizer','rng_torch','rng_numpy','rng_python'):same(u[key],v[key])
        (Path(a.output)/'snapshots'/'epoch_0003.pt').unlink()
        (Path(a.output)/'complete.json').unlink()
        assert trainer.train(a,base,resume=True,synthetic=True)==0
        assert (Path(a.output)/'snapshots'/'epoch_0003.pt').exists()
        # Frozen corrected continuation really runs, leaves VAE untouched, and
        # resumes with the same state. Source function redirected ONLY in test.
        source=Path(a.output)/'snapshots'/'epoch_0003.pt';old_source=trainer.source_snapshot
        trainer.source_snapshot=lambda *_:source
        try:
            frozen=small(next(r for r in experiments.rows() if r['stage']=='frozen' and r['config']['teacher']=='corrected'),base)
            frozen.epochs=2;frozen.eval_samples=0;frozen.diagnostic_every=0;frozen.witness_every=0;frozen.diagnostic_at_start=False
            frozen.output=str(tmp/'frozen_continuous');assert trainer.train(frozen,base,synthetic=True)==0
            fr=copy.copy(frozen);fr.output=str(tmp/'frozen_resumed')
            assert trainer.train(fr,base,synthetic=True,stop_after=1)==75
            assert trainer.train(fr,base,resume=True,synthetic=True)==0
            u=torch.load(Path(frozen.output)/'latest.pt',weights_only=False);v=torch.load(Path(fr.output)/'latest.pt',weights_only=False)
            for key in ('vae','head','ema','optimizer','scheduler','rng_torch'):same(u[key],v[key])
            same(u['vae'],torch.load(source,weights_only=False)['vae'])
        finally:trainer.source_snapshot=old_source
    print('[PASS] bitwise joint/GAN and frozen-corrected interrupted resume; log rollback and final-snapshot crash recovery')


def smoke(base,row,device='cpu'):
    if row['stage']!='joint':
        row=experiments.rows()[row['source']]
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as tmp:
        a=small(row,base,device);a.output=str(Path(tmp)/'run');a.epochs=1;a.witness_every=1;a.archive_epochs=[1]
        assert trainer.train(a,base,synthetic=True)==0
        report=json.loads((Path(a.output)/'diagnostics'/'epoch_0001.json').read_text())
        assert any(r['metric']=='score/pathwise_defect_energy' for r in report['rows'])
        assert (Path(a.output)/'witness'/'epoch_0001_probes.npz').exists()
        assert (Path(a.output)/'samples'/'epoch_0001'/'cfg1'/'generated.png').exists()
    print('[PASS] full synthetic training/evaluation/witness/checkpoint pipeline:',row['name'])
    return 0


def isolation(base):
    args=small(experiments.rows()[4],base);old.seed_everything(42)
    vae,head=models.build(args,torch.device('cpu'));ema=copy.deepcopy(head).eval().requires_grad_(False)
    vae.train();head.train();head.zero_grad();params=list(vae.parameters())+list(head.parameters())
    for p in params:p.grad=torch.ones_like(p)*.001
    states=[p.clone() for p in params];grads=[p.grad.clone() for p in params]
    tr,te=trainer.loaders(args,True)
    schedule=old.ref.make_schedule(dict(time_schedule='log_t',t_min=args.t_min,t_max=args.t_max,num_train_timesteps=1000),'cpu')
    rng=torch.get_rng_state().clone()
    with tempfile.TemporaryDirectory() as tmp:
        measurements.core(args,0,vae,head,ema,tr.dataset,te.dataset,schedule,torch.device('cpu'),tmp)
        witness.run(args,0,vae,head,ema,tr.dataset,te.dataset,torch.device('cpu'),tmp)
    assert vae.training and head.training and not ema.training
    assert torch.equal(rng,torch.get_rng_state())
    for p,s,g in zip(params,states,grads):same(p,s);same(p.grad,g)
    print('[PASS] core and learned-witness evaluations preserve weights, parameter gradients, RNG and modes')


def reference_pipeline(base):
    args=small(experiments.rows()[12],base);args.epochs=1
    old.seed_everything(42);vae,head=models.build(args,torch.device('cpu'))
    gen=torch.Generator().manual_seed(321)
    ds=torch.utils.data.TensorDataset(torch.rand(40,1,32,32,generator=gen)*2-1,torch.arange(40)%10)
    loader=torch.utils.data.DataLoader(ds,batch_size=2)
    original=trainer.loaders
    trainer.loaders=lambda *_:(loader,loader)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            args.output=tmp;source=Path(tmp)/'source.pt'
            trainer.save_snapshot(source,1,args,vae,head,head)
            rows=reference.audit(args,base,source,banks=(1,2),particles=2,queries=2,times=[.2])
            assert any(r['metric']=='reference/bank2/head_mse' for r in rows)
            assert (Path(tmp)/'reference_audit'/'epoch_0001_probes.npz').exists()
    finally:trainer.loaders=original
    print('[PASS] three-block finite-bank audit output, both conditioning modes and nested banks')


def default_capacity(base):
    two=experiments.configuration(experiments.rows()[4],base,device='cpu')
    three=experiments.configuration(experiments.rows()[12],base,device='cpu')
    states=None
    for args in (two,three):
        old.seed_everything(42);vae,head=models.build(args,torch.device('cpu'))
        assert sum(p.numel() for p in head.parameters())==12159556
        if states is None:states={k:v.clone() for k,v in head.state_dict().items()}
        else:
            for k,v in head.state_dict().items():same(v,states[k])
        x=torch.rand(2,1,32,32)*2-1;labels=torch.tensor([2,5])
        schedule=old.ref.make_schedule(dict(time_schedule='log_t',t_min=args.t_min,t_max=args.t_max,num_train_timesteps=1000),'cpu')
        score,terminal,band,d=models.training_losses(args,vae,head,x,labels,schedule,indices=torch.tensor([100,700]))
        loss=score+terminal+band+F.mse_loss(d['reconstruction'],x);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in list(vae.parameters())+list(head.parameters()))
        del vae,head,d,loss,score,terminal,band
    print('[PASS] full 192x18 joint-head two/three-block forward/backward and identical head initialization')


def run(base,models=False):
    torch.set_num_threads(2)
    rows=experiments.rows();assert len(rows)==54
    assert len(experiments.selected('core'))==22 and len(experiments.selected('frozen'))==24
    assert len({r['name'] for r in rows})==54
    for r in rows:
        c=r['config'];assert c['score_weight']==.6 and c['lr_head']==2e-4
        if r['source'] is not None:assert rows[r['source']]['stage']=='joint'
    analytic_teachers();witness_contract();parity_and_scaling(base);zero_controls(base)
    if models:
        isolation(base);resume_contract(base)
        for cid in (4,10,12,16,18,20):smoke(base,rows[cid])
        reference_pipeline(base);default_capacity(base)
    print('[VALIDATION] PASS. Analytic/software checks only; no real-data training or pretrained FID measured locally.')
    return 0
