"""Optional expensive finite-image/particle score audit, separate from training.

This is explicitly a finite-bank reference. Learned variational witnesses in
witness.py are the primary density-bank-free test for bias. Do not infer a
population score error from a failed resolution screen.
"""
import argparse
import json
import math
from pathlib import Path
import torch
from models import build,coeff,flat,pack,unpack,draw_noisy,corrected_teacher
from measurements import Log,subset
from pinned import stacked_two_block as old


def balanced_bank(dataset,per_class,seed):
    if hasattr(dataset,'targets'):labels=torch.as_tensor(dataset.targets)
    elif hasattr(dataset,'tensors'):labels=dataset.tensors[1]
    else:labels=torch.tensor([int(dataset[i][1]) for i in range(len(dataset))])
    gen=torch.Generator().manual_seed(seed);ids=[]
    for c in range(10):
        indices=torch.where(labels==c)[0]
        ids.extend(indices[torch.randperm(len(indices),generator=gen)[:per_class]].tolist())
    xs=torch.stack([dataset[i][0] for i in ids]);ys=labels[ids]
    return xs,ys,ids


def panel(args,vae,bank,query,tv,particles,device):
    weights,targets,naives=[],[],[]
    for off in range(0,len(bank),args.diagnostic_batch):
        x=bank[off:off+args.diagnostic_batch].to(device)
        x=x.repeat_interleave(particles,0);n=len(x)
        t=torch.full((n,),tv,device=device);a,s=coeff(t)
        with torch.no_grad():
            mu,lv,feat=vae.root(x);zs,params,ys,cs,logw=[],[],[],[],torch.zeros(n,device=device,dtype=torch.float64)
            for i in range(vae.blocks):
                m,l=vae.moment(i,feat,mu,lv,zs);v=l.exp();y=query[i].expand(n,*query[i].shape[1:]);ss=a.square()*v+s.square()
                logw+=-.5*flat((y-a*m).double().square()/ss.double()+ss.double().log()).sum(1)
                post=m+a*v/ss*(y-a*m);postv=v*s.square()/ss
                zs.append(post+postv.sqrt()*torch.randn_like(post));params.append((m,l));ys.append(y);cs.append(s*(y-a*m)/ss)
            d=dict(z=zs,params=params,y=ys,naive=cs,alpha=a,sigma=s,features=feat,root_mu=mu,root_lv=lv)
        valid=corrected_teacher(vae,d)
        targets.append(torch.cat([flat(v) for v in valid],1).double().reshape(-1,particles,256))
        naives.append(torch.cat([flat(v) for v in cs],1).double().reshape(-1,particles,256))
        weights.append(logw.reshape(-1,particles))
    lw=torch.cat(weights);tt=torch.cat(targets);cc=torch.cat(naives)
    def average(w,t):
        p=torch.softmax(w.flatten(),0).reshape(w.shape)
        return (p[:,:,None]*t).sum((0,1))
    w=torch.softmax(lw.flatten(),0).reshape(lw.shape);iw=w.sum(1)
    return dict(true=average(lw,tt),naive=average(lw,cc),
                half_particles=average(lw[:,:max(1,particles//2)],tt[:,:max(1,particles//2)]),
                ess=float(1/w.square().sum()),image_ess=float(1/iw.square().sum()),max_weight=float(w.max()))


def audit(args,base,checkpoint,*,banks=(64,256),particles=16,queries=16,times=None):
    from trainer import loaders
    device=torch.device(args.device);vae,head=build(args,device)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    vae.load_state_dict(saved['vae']);head.load_state_dict(saved['ema']);vae.eval().requires_grad_(False);head.eval().requires_grad_(False)
    tr,te=loaders(args)
    times=args.diagnostic_times if times is None else times
    log=Log(saved['epoch'])
    with old.preserved_state(vae,head):
        old.seed_everything(args.seed+84000)
        bank,by,bank_ids=balanced_bank(tr.dataset,max(banks),args.seed+84001)
        x,labels,query_ids=subset(te.dataset,queries,args.seed+84002)
        sizes=[int((by==i).sum()) for i in range(10)]
        if min(sizes)<max(banks):raise ValueError('Not enough bank images per class')
        for tv in times:
            with torch.no_grad():
                d=draw_noisy(vae,x.to(device),torch.full((len(x),),tv,device=device))
            for condition in ('class','null'):
                labels0=labels.to(device) if condition=='class' else torch.full_like(labels,10).to(device)
                with torch.no_grad():pp=unpack(head(d['packed'],d['time'],labels0),vae.blocks);pred=torch.cat([flat(v) for v in pp],1).double()
                for qi in range(len(x)):
                    prior=None
                    query=[v[qi:qi+1] for v in d['y']]
                    for count in banks:
                        indices=[]
                        classes=[int(labels[qi])] if condition=='class' else range(10)
                        for c in classes:indices.extend(torch.where(by==c)[0][:count].tolist())
                        panels=[panel(args,vae,bank[indices],query,tv,particles,device) for _ in range(2)]
                        p,q=panels;truth=(p['true']+q['true'])/2;naive=(p['naive']+q['naive'])/2
                        prefix=f'reference/bank{count}/'
                        changes=[];offset=0
                        for bi,width in enumerate(vae.dimensions):
                            sl=slice(offset,offset+width);offset+=width
                            energy=truth[sl].square().mean().clamp_min(1e-24)
                            repeat=((p['true'][sl]-q['true'][sl]).square().mean()/energy).sqrt()
                            half=(((p['half_particles'][sl]+q['half_particles'][sl])/2-truth[sl]).square().mean()/energy).sqrt()
                            bank_change=((truth[sl]-prior[sl]).square().mean()/energy).sqrt() if prior is not None else energy.new_tensor(float('nan'))
                            changes.extend([float(repeat),float(half),float(bank_change)])
                            for name,value in dict(head_mse=(pred[qi,sl]-truth[sl]).square().mean(),
                                stack_mse=(naive[sl]-truth[sl]).square().mean(),head_relative_rms=((pred[qi,sl]-truth[sl]).square().mean()/energy).sqrt(),
                                particle_repeat_relative_rms=repeat,half_particle_relative_rms=half,nested_bank_relative_rms=bank_change).items():
                                log.add(prefix+name,[value],time=tv,conditioning=condition,block=bi+1)
                        min_ess=min(p['image_ess'],q['image_ess'])
                        passed=all(math.isfinite(c) and c<.15 for c in changes) and min_ess>=2
                        log.add(prefix+'resolution_screen',[float(passed)],time=tv,conditioning=condition)
                        for name in ('ess','image_ess','max_weight'):
                            log.add(prefix+name,[(p[name]+q[name])/2],time=tv,conditioning=condition)
                        prior=truth
    dest=Path(args.output)/'reference_audit'
    return log.save(dest,dict(checkpoint=str(checkpoint),banks_per_class=list(banks),particles=particles,
                             query_ids=query_ids,bank_ids=bank_ids,reference_scope='finite bank; screen does not certify population accuracy'))
