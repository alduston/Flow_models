"""Independent learned witnesses give variational LOWER bounds on score bias.

delta = C_naive-T_corrected, b(Y,c)=E[delta|Y,c]. For any fitted function w,
2 E[delta.w]-E||w||^2 <= E||b||^2 <= E||delta||^2 (Jensen).
We fit w on train-image draws and evaluate it on untouched test-image draws.
No density importance bank and no claim that a negative lower bound proves
zero bias. Pointwise 95% intervals are descriptive; no multiple-test selection.
"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from models import draw_noisy,corrected_teacher,flat,unpack
from measurements import Log,subset
from pinned import stacked_two_block as old


def collect(args,vae,heads,dataset,count,tv,device,seed):
    x,y,ids=subset(dataset,count,seed)
    ys,defects,targets,preds=[],[],[],{h:[] for h in heads}
    for off in range(0,len(x),args.diagnostic_batch):
        xx=x[off:off+args.diagnostic_batch].to(device);lab=y[off:off+len(xx)].to(device)
        t=torch.full((len(xx),),tv,device=device)
        with torch.no_grad():d=draw_noisy(vae,xx,t)
        correct=corrected_teacher(vae,d)
        ys.append(flat(d['packed']).cpu())
        defects.append(torch.cat([flat(a-b) for a,b in zip(d['naive'],correct)],1).cpu())
        targets.append(torch.cat([flat(q) for q in correct],1).cpu())
        for hn,head in heads.items():
            with torch.no_grad():
                # Both conditioning modes share the identical held-out draws.
                ps=[]
                for labs in (lab,torch.full_like(lab,10)):
                    p=unpack(head(d['packed'],t,labs),vae.blocks)
                    ps.append(torch.cat([flat(q) for q in p],1).cpu())
                preds[hn].append(torch.stack(ps,1))
    return dict(y=torch.cat(ys).double(),delta=torch.cat(defects).double(),
                teacher=torch.cat(targets).double(),labels=y,ids=ids,
                preds={h:torch.cat(v).double() for h,v in preds.items()})


def fit_predict(train,test,args,conditioning='null'):
    a,b=train['y'],test['y']
    center=a.mean(0);scale=a.std(0,unbiased=False).clamp_min(.05)
    a=(a-center)/scale;b=(b-center)/scale
    gen=torch.Generator().manual_seed(args.seed+82003)
    directions=torch.randn(256,args.witness_features,generator=gen,dtype=torch.float64)/16
    phase=2*math.pi*torch.rand(args.witness_features,generator=gen,dtype=torch.float64)
    def features(v):
        return torch.cat([torch.ones(len(v),1,dtype=v.dtype),v/16,
            math.sqrt(2/args.witness_features)*torch.cos(v@directions+phase)],1)
    aa,bb=features(a),features(b)
    out=torch.zeros_like(test['delta'])
    groups=range(10) if conditioning=='class' else [10]
    for group in groups:
        ii=train['labels']==group if conditioning=='class' else torch.ones(len(aa),dtype=torch.bool)
        jj=test['labels']==group if conditioning=='class' else torch.ones(len(bb),dtype=torch.bool)
        if not ii.any() or not jj.any():continue
        x=aa[ii];z=train['delta'][ii];n=len(x)
        gram=x.T@x/n+args.witness_ridge*torch.eye(x.shape[1],dtype=x.dtype)
        coef=torch.linalg.solve(gram,x.T@z/n)
        out[jj]=bb[jj]@coef
    return out


def run(args,epoch,vae,head,ema,train_dataset,test_dataset,device,out):
    log=Log(epoch);metadata={}
    with old.preserved_state(vae,head,ema):
        vae.eval();head.eval();ema.eval()
        for ti,tv in enumerate(args.diagnostic_times):
            old.seed_everything(args.seed+82000+ti)
            tr=collect(args,vae,{},train_dataset,args.witness_train,tv,device,args.seed+82001)
            te=collect(args,vae,{'online':head,'ema':ema},test_dataset,args.witness_test,tv,device,args.seed+82002)
            metadata=dict(train_ids=tr['ids'],test_ids=te['ids'],test_labels=te['labels'].tolist(),features=args.witness_features,
                          ridge=args.witness_ridge,test_population='heldout empirical encoder aggregate')
            for ci,cond in enumerate(('class','null')):
                w=fit_predict(tr,te,args,cond)
                offset=0
                for i,width in enumerate(vae.dimensions):
                    sl=slice(offset,offset+width);offset+=width
                    d,f=te['delta'][:,sl],w[:,sl]
                    lower=(2*d*f-f.square()).mean(1)
                    energy=f.square().mean(1)
                    log.add('witness/bias_squared_lower',lower,time=tv,conditioning=cond,block=i+1)
                    log.add('witness/pathwise_bias_upper',d.square().mean(1),time=tv,conditioning=cond,block=i+1)
                    log.add('witness/fitted_field_energy',energy,time=tv,conditioning=cond,block=i+1)
                    log.add('witness/defect_projection',(d*f).mean(1),time=tv,conditioning=cond,block=i+1)
                    for hn,ps in te['preds'].items():
                        residual=ps[:,ci,sl]-te['teacher'][:,sl]
                        log.add('witness/head_error_projection',(residual*f).mean(1),time=tv,conditioning=cond,block=i+1,head=hn)
    rows=log.save(Path(out)/'witness',metadata)
    summary=[]
    for r in rows:
        if r['metric']=='witness/bias_squared_lower':
            summary.append(dict(time=r['time'],block=r['block'],conditioning=r['conditioning'],
                                lower=r['mean'],se=r['se'],lcb95=r['mean']-1.96*r['se'] if r['se'] is not None else None))
    print('[BIAS_WITNESS]',json.dumps(dict(epoch=epoch,results=summary)),flush=True)
    return rows
