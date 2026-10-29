"""One legacy Inception/FID contract for every stage; isolated fixed RNG."""
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
import models
from pinned import stacked_two_block as old
from protocol import atomic,object_sha

def upsert(path,row):
    path=Path(path);prior=[];keys=('epoch','cfg','kind','eval_split')
    if path.exists():
        with path.open() as f:
            prior=[x for x in csv.DictReader(f) if any(str(x[k])!=str(row[k]) for k in keys)]
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(prior+[row])
    os.replace(tmp,path)

def prepare(args,loader,device,out,synthetic=False):
    images=[];labels=[];count=0
    for x,y in loader:
        n=min(len(x),args.eval_samples-count)
        if n<=0:break
        images.append(x[:n]);labels.append(y[:n]);count+=n
    real=torch.cat(images);labels=torch.cat(labels)
    if len(real)!=args.eval_samples:raise ValueError('Reference has fewer images than the evaluation contract')
    if synthetic:return real,labels,real.flatten(1)[:,:16].double(),None
    # Instantiate exactly the inherited feature extractor, never a new FID backend.
    features,net=old.ref.extract_inception_features(real,device,args.feature_batch)
    old.to_grid(real,Path(out)/'target_real.png')
    return real,labels,features,net

def kid(real,fake,subsets,size,seed):
    """Unbiased polynomial MMD^2; raw KID units, subset SD is not seed SD."""
    x=real.detach().cpu().double().numpy();y=fake.detach().cpu().double().numpy()
    rng=np.random.default_rng(seed);n=min(size,len(x),len(y));values=[]
    if n<2:raise ValueError('KID subset needs at least two examples')
    for _ in range(subsets):
        a=x[rng.choice(len(x),n,replace=False)];b=y[rng.choice(len(y),n,replace=False)]
        aa=(a@a.T/a.shape[1]+1)**3;bb=(b@b.T/b.shape[1]+1)**3;ab=(a@b.T/a.shape[1]+1)**3
        values.append(float((aa.sum()-np.trace(aa)+bb.sum()-np.trace(bb))/(n*(n-1))-2*ab.mean()))
    return float(np.mean(values)),float(np.std(values,ddof=1)) if len(values)>1 else 0.

class CountHead(nn.Module):
    def __init__(self,head):super().__init__();self.head=head;self.calls=0
    def forward(self,*a,**kw):self.calls+=1;return self.head(*a,**kw)

@torch.no_grad()
def generate(args,head,starts,labels,device,cfg):
    sampler=old.ref.UniversalSampler('rk4_ode',args.rk4_steps,t_min=args.t_min,t_max=args.t_max,schedule_type='log_t')
    value=sampler.sample(head,shape=tuple(starts.shape),device=device,x_init=starts.to(device),y=labels.to(device),cfg_scale=cfg)
    return models.unpack(value,3)

def balanced_grid(images,labels,path):
    ids=[]
    for c in range(10):ids.extend(torch.where(labels==c)[0][:10].tolist())
    if ids:
        from torchvision.utils import save_image
        save_image((images[ids]+1)/2,path,nrow=10)

@torch.no_grad()
def evaluate(args,epoch,vae,head,fid_info,device,out,timing,synthetic=False):
    real,labels,features,inception=fid_info;out=Path(out)
    extract=(lambda x:x.flatten(1)[:,:16].double()) if synthetic else (
        lambda x:old.ref.extract_inception_features(x,device,args.feature_batch,inception_model=inception)[0])
    start=time.monotonic();rows=[]
    with old.preserved_state(vae,head):
        vae.eval()
        if head is not None:head.eval()
        # CPU streams are independent of batch order, training seed, teacher and checkpoint.
        innovations=[torch.randn((len(real),*s),generator=torch.Generator().manual_seed(args.eval_seed+10+i)) for i,s in enumerate(vae.shapes)]
        starts=torch.randn(len(real),4,8,8,generator=torch.Generator().manual_seed(args.eval_seed))
        rec=[];diag={};count=0
        for off in range(0,len(real),args.eval_batch):
            x=real[off:off+args.eval_batch].to(device);n=len(x)
            d=vae.draw(x,innovations=[v[off:off+n].to(device) for v in innovations])
            image=vae.decode(d['z'],torch.full((n,),args.decode_time,device=device))
            if not torch.isfinite(image).all():raise FloatingPointError('Nonfinite reconstruction')
            rec.append(image.clamp(-1,1).cpu())
            for key,value in dict(kl0=__import__('train_stages').kl0(d['params'],vae.dimensions),
                posterior_variance=sum(q/256*l.exp().mean() for q,(m,l) in zip(vae.dimensions,d['params'])),
                latent_second_moment=sum(q/256*(m.square()+l.exp()).mean() for q,(m,l) in zip(vae.dimensions,d['params']))).items():
                diag[key]=diag.get(key,0.)+float(value)*n
            count+=n
        rec=torch.cat(rec);rec_f=extract(rec);recon_fid=old.fid_numeric(features,rec_f)
        atomic(out/f'diagnostics_epoch_{epoch:04d}.json',{k:v/count for k,v in diag.items()}|dict(epoch=epoch,eval_split=args.eval_split))
        settings=[0.] if head is None else [1.,3.]
        for cfg in settings:
            counted=CountHead(head) if head is not None else None;minibatches=0
            if head is not None:
                images=[]
                for off in range(0,len(real),args.eval_batch):
                    n=min(args.eval_batch,len(real)-off)
                    zs=generate(args,counted,starts[off:off+n],labels[off:off+n],device,cfg)
                    image=vae.decode(zs,torch.full((n,),args.decode_time,device=device))
                    if not torch.isfinite(image).all():raise FloatingPointError('Nonfinite generated images')
                    images.append(image.clamp(-1,1).cpu());minibatches+=1
                images=torch.cat(images);fake=extract(images);fid=old.fid_numeric(features,fake)
            else:images=rec;fake=rec_f;fid=''
            km=ks=''
            if epoch==args.epochs and args.kid_subsets and head is not None:
                km,ks=kid(features,fake,args.kid_subsets,args.kid_subset_size,args.kid_seed)
            row=dict(epoch=epoch,regime=args.regime,seed=args.seed,eval_split=args.eval_split,
                cfg=cfg,kind='generation' if head is not None else 'reconstruction',steps=args.rk4_steps,
                score_calls_per_sample=counted.calls/minibatches if head is not None else 0,
                score_forward_calls_all_batches=counted.calls if head is not None else 0,
                init='gaussian',temperature=1.,decode_time=args.decode_time,num_eval=len(real),
                fid_generated=fid,fid_posterior_recon=recon_fid,kid=km,kid_subset_sd=ks,
                kid_subsets=args.kid_subsets if km!='' else 0,kid_subset_size=min(args.kid_subset_size,len(real)),
                backend='synthetic-smoke-only' if synthetic else 'inception-legacy',
                training_seconds=timing['training_seconds'],source_training_seconds=timing['source_training_seconds'],
                end_to_end_training_gpu_hours=(timing['training_seconds']+timing['source_training_seconds'])/3600)
            upsert(out/'eval.csv',row);rows.append(row);print('[EVAL]',json.dumps(row),flush=True)
            dest=out/'samples'/f'epoch_{epoch:04d}'/f'cfg{cfg:g}';dest.mkdir(parents=True,exist_ok=True)
            balanced_grid(images,labels,dest/'generated.png');balanced_grid(rec,labels,dest/'posterior.png')
            if epoch==args.epochs:
                np.savez_compressed(dest/'prespecified_samples.npz',generated=images[:args.save_array_samples].numpy(),
                                    posterior=rec[:args.save_array_samples].numpy(),labels=labels[:args.save_array_samples].numpy())
    return rows

@torch.no_grad()
def reload_parity(args,vae,head,path,device):
    """Same saved inputs: reload, entire reverse solve, decoder and encoder moments."""
    from train_stages import state_hash
    with old.preserved_state(vae,head):
        vae.eval()
        if head is not None:head.eval()
        saved=torch.load(path,map_location='cpu',weights_only=False)
        v,h=models.build(args,device);v.load_state_dict(saved['vae']);v.eval()
        if head is not None:h.load_state_dict(saved['ema']);h.eval()
        assert state_hash(v.state_dict())==state_hash(vae.state_dict())
        if head is not None:assert state_hash(h.state_dict())==state_hash(head.state_dict())
        g=torch.Generator().manual_seed(args.eval_seed+99)
        x=torch.randn(2,1,32,32,generator=g).to(device).clamp(-1,1)
        eta=[torch.randn(2,*s,generator=g).to(device) for s in vae.shapes]
        d1=vae.draw(x,innovations=eta);d2=v.draw(x,innovations=eta)
        t=torch.full((2,),args.decode_time,device=device)
        a=vae.decode(d1['z'],t);b=v.decode(d2['z'],t);err=float((a-b).abs().max())
        if head is not None:
            z=torch.randn(2,4,8,8,generator=g);labels=torch.tensor([0,9])
            for cfg in (1.,3.):
                z1=generate(args,head,z,labels,device,cfg);z2=generate(args,h,z,labels,device,cfg)
                err=max(err,float((vae.decode(z1,t)-v.decode(z2,t)).abs().max()))
        if err>1e-6:raise AssertionError(f'Checkpoint-reload evaluation drift: {err}')
        return dict(passed=True,max_abs_error=err,inputs_seed=args.eval_seed+99,
                    checks=['tensor_identity','posterior_reconstruction','RK4 generation at CFG1 and CFG3'] if head is not None else ['tensor_identity','posterior_reconstruction'])
