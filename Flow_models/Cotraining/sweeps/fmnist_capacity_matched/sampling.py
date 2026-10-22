"""Matched joint reverse solves; score probes use CFG 1, deployment also CFG 3."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from models import pack,unpack,coeff,draw_noisy
from pinned import stacked_two_block as old


@torch.no_grad()
def generate(args,head,n,device,labels,starts=None,*,cfg=None,steps=None):
    sampler=old.ref.UniversalSampler('rk4_ode',steps or args.rk4_steps,t_min=args.t_min,t_max=args.t_max,schedule_type='log_t')
    value=sampler.sample(head,shape=(n,4,8,8),device=device,x_init=starts,y=labels,
                         cfg_scale=args.cfg if cfg is None else cfg)
    return unpack(value,args.blocks)


def upsert(path,row,keys):
    path=Path(path);prior=[]
    if path.exists():
        with path.open() as f:prior=[r for r in csv.DictReader(f) if any(str(r[k])!=str(row[k]) for k in keys)]
    with path.with_suffix('.tmp').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(prior+[row])
    path.with_suffix('.tmp').replace(path)


@torch.no_grad()
def evaluate(args,epoch,vae,head,fid_info,device,out,*,checks=False,fake_features=False):
    if fid_info is None:return []
    out=Path(out)
    real,labels,features,inception=fid_info
    def extract(x):
        if fake_features:return x.flatten(1)[:,:16].double()
        return old.ref.extract_inception_features(x,device,args.feature_batch,inception_model=inception)[0]
    results=[]
    with old.preserved_state(vae,head):
        vae.eval();head.eval()
        old.seed_everything(args.seed+83000)
        # Starts and posterior draws are generated once so changing evaluation
        # minibatch size does not change paired starts/innovations.
        n=len(real)
        starts=torch.randn(n,4,8,8)
        reconstructed=[];oracle=[]
        for off in range(0,n,args.eval_batch):
            x=real[off:off+args.eval_batch].to(device)
            t=torch.full((len(x),),args.t_max,device=device)
            d=draw_noisy(vae,x,t)
            ts=torch.full_like(t,args.decode_time)
            reconstructed.append(vae.decode(d['z'],ts).clamp(-1,1).cpu())
            oracle.append(pack(d['y']).cpu())
        rec=torch.cat(reconstructed);oracle=torch.cat(oracle)
        fid_rec=old.fid_numeric(features,extract(rec))
        settings=[(1.,args.rk4_steps,'gaussian',n),(3.,args.rk4_steps,'gaussian',n)]
        if checks:
            count=min(n,args.sampling_check_samples)
            settings += [(g,s,init,count) for g in (1.,3.) for s in (20,50) for init in ('gaussian','oracle_qT')]
        for cfg,steps,init,count in settings:
            if any((r['cfg'],r['steps'],r['init'],r['num_eval'])==(cfg,steps,init,count) for r in results):continue
            generated=[]
            for off in range(0,count,args.eval_batch):
                stop=min(count,off+args.eval_batch);lab=labels[off:stop].to(device)
                start=(starts if init=='gaussian' else oracle)[off:stop].to(device)
                zs=generate(args,head,stop-off,device,lab,start,cfg=cfg,steps=steps)
                ts=torch.full((stop-off,),args.decode_time,device=device)
                image=vae.decode(zs,ts)
                if not torch.isfinite(image).all():raise FloatingPointError('Nonfinite generated sample')
                generated.append(image.clamp(-1,1).cpu())
            images=torch.cat(generated)
            row=dict(epoch=epoch,cfg=cfg,steps=steps,init=init,num_eval=count,
                     fid_generated=old.fid_numeric(features[:count],extract(images)),
                     fid_posterior_recon=fid_rec if count==n else old.fid_numeric(features[:count],extract(rec[:count])),
                     backend='synthetic-smoke-only' if fake_features else 'inception-legacy')
            upsert(out/'eval.csv',row,('epoch','cfg','steps','init','num_eval'))
            results.append(row);print('[EVAL]',json.dumps(row),flush=True)
            if count==n and init=='gaussian' and steps==args.rk4_steps:
                dest=out/'samples'/f'epoch_{epoch:04d}'/f'cfg{cfg:g}';dest.mkdir(parents=True,exist_ok=True)
                old.to_grid(images,dest/'generated.png');old.to_grid(rec,dest/'posterior.png')
                np.save(dest/'generated_first.npy',images[:args.save_array_samples].numpy())
    return results
