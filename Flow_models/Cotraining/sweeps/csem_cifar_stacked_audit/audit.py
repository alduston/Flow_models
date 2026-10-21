"""Held-out proper risks, coupling, signed witness bounds and paired raw arrays."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
from stacked import noisy,corrected,StackedVAE
from suite import fixed_metric_rng


def subset(ds,count,seed):
    ids=torch.randperm(len(ds),generator=torch.Generator().manual_seed(seed))[:min(count,len(ds))].tolist()
    p=[ds[i] for i in ids];return torch.stack([q[0] for q in p]),torch.tensor([int(q[1]) for q in p]),ids

def slices(vae):return [slice(0,2),slice(2,8)] if isinstance(vae,StackedVAE) else [slice(0,8)]

def summary(values,**meta):
    v=np.asarray(values,dtype=float)
    return dict(meta,n=len(v),mean=float(v.mean()),se=float(v.std(ddof=1)/math.sqrt(len(v))) if len(v)>1 else float('nan'))

def collect(vae,heads,ds,n,tv,cfg,device,seed):
    x,labels,ids=subset(ds,n,seed);out={k:[] for k in ('y','delta','valid','dsm','naive','mu','lv')};preds={h:[] for h in heads}
    for off in range(0,len(x),cfg['audit_batch']):
        xx=x[off:off+cfg['audit_batch']].to(device);lab=labels[off:off+len(xx)].to(device)
        with torch.no_grad():d=noisy(vae,xx,torch.full((len(xx),),tv,device=device))
        v=corrected(vae,d)
        for k,z in dict(y=d['y'],delta=d['naive']-v,valid=v,dsm=d['eps'],naive=d['naive'],mu=d['mu'],lv=d['lv']).items():out[k].append(z.cpu())
        for name,head in heads.items():
            with torch.no_grad():preds[name].append(torch.stack([head(d['y'],d['time'],q).cpu() for q in (lab,torch.full_like(lab,10))],1))
    return dict(**{k:torch.cat(v) for k,v in out.items()},pred={k:torch.cat(v) for k,v in preds.items()},labels=labels,ids=ids)

def witness(tr,te,seed,conditioning):
    a=tr['y'].flatten(1).double();b=te['y'].flatten(1).double();D=a.shape[1]
    m=a.mean(0);s=a.std(0,unbiased=False).clamp_min(.05);a=(a-m)/s;b=(b-m)/s
    g=torch.Generator().manual_seed(seed+82003);w=torch.randn(D,64,generator=g,dtype=torch.float64)/math.sqrt(D)
    phase=2*math.pi*torch.rand(64,generator=g,dtype=torch.float64)
    def phi(x):return torch.cat((torch.ones(len(x),1),x/math.sqrt(D),math.sqrt(2/64)*torch.cos(x@w+phase)),1)
    a,b=phi(a),phi(b);delta=tr['delta'].flatten(1).double();answer=torch.zeros_like(te['delta'].flatten(1),dtype=torch.float64)
    for label in (range(10) if conditioning=='class' else [10]):
        ii=tr['labels']==label if conditioning=='class' else torch.ones(len(a),dtype=torch.bool)
        jj=te['labels']==label if conditioning=='class' else torch.ones(len(b),dtype=torch.bool)
        if ii.any() and jj.any():
            aa=a[ii];coef=torch.linalg.solve(aa.T@aa/len(aa)+.001*torch.eye(aa.shape[1]),aa.T@delta[ii]/len(aa))
            answer[jj]=b[jj]@coef
    return answer.reshape_as(te['delta'])

def risk_rows(data,vae,epoch,tv,raw):
    rr=[]
    for name,ps in data['pred'].items():
        for ci,cond in enumerate(('class','null')):
            for block,sl in enumerate(slices(vae),1):
                p=ps[:,ci,sl];valid=data['valid'][:,sl];eps=data['dsm'][:,sl];naive=data['naive'][:,sl]
                metrics=dict(valid_risk=p.square()-2*p*valid,dsm_risk=p.square()-2*p*eps,
                    naive_mse=(p-naive).square(),defect_energy=(naive-valid).square())
                for metric,value in metrics.items():
                    vals=value.flatten(1).mean(1).numpy();key=f'{name}|{cond}|{tv:g}|b{block}|{metric}'
                    raw[key]=vals;rr.append(summary(vals,epoch=epoch,time=tv,head=name,conditioning=cond,block=block,metric=metric))
    return rr

def run_audit(epoch,vae,head,train_ds,test_ds,cfg,device,*,heads=None,out_name='audit'):
    out=Path(cfg['results_dir'])/out_name;out.mkdir(parents=True,exist_ok=True)
    rows=[];raw={};heads={'ema':head} if heads is None else heads
    modes={m:m.training for m in [vae,*heads.values()]}
    try:
        for m in modes:m.eval()
        with fixed_metric_rng(torch,int(cfg['seed'])+81000):
            for tv in cfg['audit_times']:
                te=collect(vae,heads,test_ds,cfg['audit_samples'],tv,cfg,device,cfg['seed']+81001)
                rows+=risk_rows(te,vae,epoch,tv,raw)
                if tv==cfg['audit_times'][0]:
                    for block,sl in enumerate(slices(vae),1):
                        for metric,value in {'posterior_variance':te['lv'][:,sl].exp(),'mean_energy':te['mu'][:,sl].square()}.items():
                            vals=value.flatten(1).mean(1).numpy();raw[f'representation|b{block}|{metric}']=vals
                            rows.append(summary(vals,epoch=epoch,time=0,head='',conditioning='given_image',block=block,metric=metric))
            if isinstance(vae,StackedVAE):
                x,_,_=subset(test_ds,64,cfg['seed']+81001);scores=[]
                with torch.no_grad():
                    for off in range(0,len(x),cfg['audit_batch']):
                        xx=x[off:off+cfg['audit_batch']].to(device);m,l,h=vae.root(xx)
                        parents=[m+(.5*l).exp()*torch.randn_like(m) for _ in range(8)]
                        pars=[vae.child(h,p,m) for p in parents]
                        ms=torch.stack([p[0] for p in pars]);ls=torch.stack([p[1] for p in pars])
                        a,b=ms[::2],ms[1::2];u,v=ls[::2],ls[1::2]
                        sk=.25*((u-v).exp()+(v-u).exp()-2+(a-b).square()*((-u).exp()+(-v).exp()))
                        scores.extend(sk.mean((0,2,3,4)).cpu().tolist())
                rows.append(summary(scores,epoch=epoch,time=0,head='',conditioning='given_image',block=2,metric='conditional_symkl_per_dim'))
                raw['coupling|conditional_symkl_per_dim']=np.asarray(scores)
            # Witness fitting is held out by image; never infer bias from pathwise defect alone.
            if epoch==0 or epoch%200==0 or cfg['stage']=='frozen':
                for tv in cfg['witness_times']:
                    tr=collect(vae,{},train_ds,cfg['witness_train'],tv,cfg,device,cfg['seed']+82001)
                    te=collect(vae,heads,test_ds,cfg['witness_test'],tv,cfg,device,cfg['seed']+82002)
                    for cond in ('class','null'):
                        w=witness(tr,te,cfg['seed'],cond)
                        for block,sl in enumerate(slices(vae),1):
                            delta=te['delta'][:,sl];f=w[:,sl]
                            for metric,v in dict(bias_squared_lower=2*delta*f-f.square(),pathwise_bias_upper=delta.square(),defect_projection=delta*f).items():
                                vals=v.flatten(1).mean(1).numpy();raw[f'witness|{cond}|{tv:g}|b{block}|{metric}']=vals
                                rows.append(summary(vals,epoch=epoch,time=tv,head='',conditioning=cond,block=block,metric=metric))
                            ci=0 if cond=='class' else 1
                            for name,ps in te['pred'].items():
                                vals=((ps[:,ci,sl]-te['valid'][:,sl])*f).flatten(1).mean(1).numpy()
                                raw[f'witness|{name}|{cond}|{tv:g}|b{block}|head_error_projection']=vals
                                rows.append(summary(vals,epoch=epoch,time=tv,head=name,conditioning=cond,block=block,metric='head_error_projection'))
    finally:
        for m,training in modes.items():m.train(training)
    pd.DataFrame(rows).to_csv(out/f'epoch_{epoch:04d}.csv',index=False)
    np.savez_compressed(out/f'epoch_{epoch:04d}_paired.npz',**raw)
    return rows
