"""Original update recipe plus encoder-only gradient scaling and durable resume."""
from __future__ import annotations
import copy
import csv
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader,TensorDataset
import models
import measurements
import witness
import sampling
from models import build,training_losses,draw_noisy,corrected_teacher,pack
from pinned import stacked_two_block as old
from paper_io import load_prepared_pair,prepare_dataset


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,default=str)+'\n')
    os.replace(temp,path)


def prepare_data(cache):
    return prepare_dataset(old.ref.DATASET_INFO['FMNIST']['class'],cache,'FMNIST')


def loaders(args,synthetic=False):
    if synthetic:
        gen=torch.Generator().manual_seed(1700)
        dataset=TensorDataset(torch.rand(16,1,32,32,generator=gen)*2-1,torch.arange(16)%10)
        tr,te=dataset,dataset
    else:
        tf=transforms.Compose([transforms.Pad(2),transforms.ToTensor(),transforms.Normalize((.5,),(.5,))])
        tr,te=load_prepared_pair(old.ref.DATASET_INFO['FMNIST']['class'],args.data_root,'FMNIST',transform=tf)
    return (DataLoader(tr,batch_size=args.batch,shuffle=True,num_workers=args.num_workers,drop_last=True),
            DataLoader(te,batch_size=args.batch,shuffle=False,num_workers=args.num_workers))


def source_snapshot(args,base):
    from experiments import rows
    r=rows()[args.source]
    return Path(base)/'results'/r['name']/'snapshots'/f'epoch_{r["config"]["epochs"]:04d}.pt'


def capture_rng():
    return dict(rng_torch=torch.get_rng_state(),rng_numpy=np.random.get_state(),rng_python=random.getstate(),
                rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def save_checkpoint(path,epoch,args,vae,head,ema,opt,scheduler,disc,opt_disc,source_sha=None):
    payload=dict(version=models.VERSION,epoch=epoch,args=vars(args),vae=vae.state_dict(),head=head.state_dict(),
                 ema=ema.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
                 disc=disc.state_dict() if disc is not None else None,
                 disc_optimizer=opt_disc.state_dict() if opt_disc is not None else None,
                 source_sha256=source_sha,**capture_rng())
    old.atomic_torch_save(payload,Path(path))


def save_snapshot(path,epoch,args,vae,head,ema):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    old.atomic_torch_save(dict(version=models.VERSION,epoch=epoch,args=vars(args),
                              vae=vae.state_dict(),head=head.state_dict(),ema=ema.state_dict()),Path(path))


def trim_uncommitted(out,epoch):
    out=Path(out)
    for path in out.rglob('*.csv'):
        if path.name=='resume_recipe.csv':continue
        with path.open() as f:
            rr=csv.DictReader(f);fields=rr.fieldnames;values=list(rr)
        if not fields or 'epoch' not in fields:continue
        keep=[r for r in values if int(float(r['epoch']))<=epoch]
        with path.with_suffix('.tmp').open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(keep)
        path.with_suffix('.tmp').replace(path)
    for folder in ('diagnostics','witness','snapshots','samples'):
        for path in (out/folder).glob('epoch_*'):
            try:number=int(path.name.split('_')[1].split('.')[0])
            except ValueError:continue
            if number>epoch:
                if path.is_dir():
                    import shutil
                    shutil.rmtree(path)
                else:path.unlink()


class StopFlag:
    def __init__(self):self.stop=False;self.old={}
    def __enter__(self):
        for sig in (signal.SIGUSR1,signal.SIGTERM):
            self.old[sig]=signal.signal(sig,self.handler)
        return self
    def handler(self,*_):
        self.stop=True
        print('[SIGNAL] stop requested; committing the current epoch before exit.',flush=True)
    def __exit__(self,*_):
        for sig,handler in self.old.items():signal.signal(sig,handler)


def eval_epoch(args,epoch,vae,head,ema,train_loader,test_loader,schedule,device,out,fid_info,
               lpips_net=None,disc=None,synthetic=False):
    final=epoch==args.epochs
    if args.eval_samples and (epoch==0 and args.stage=='frozen' or epoch>0 and (epoch%args.eval_every==0 or final)):
        checks=bool(args.sampling_check_every and epoch>0 and (epoch%args.sampling_check_every==0 or final))
        sampling.evaluate(args,epoch,vae,ema,fid_info,device,out,checks=checks,fake_features=synthetic)
    if args.diagnostic_every and (epoch==0 or epoch%args.diagnostic_every==0 or final):
        measurements.core(args,epoch,vae,head,ema,train_loader.dataset,test_loader.dataset,schedule,device,out,lpips_net,disc)
    if args.witness_every and epoch>0 and (epoch%args.witness_every==0 or final):
        witness.run(args,epoch,vae,head,ema,train_loader.dataset,test_loader.dataset,device,out)


def train(args,base,*,resume=False,synthetic=False,stop_after=None,diagnostics_override=None):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    if args.cpu_threads:torch.set_num_threads(args.cpu_threads)
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA GPU is not visible')
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    old.seed_everything(args.seed)
    train_loader,test_loader=loaders(args,synthetic)
    if synthetic and args.eval_samples:
        x,y,_=measurements.subset(test_loader.dataset,args.eval_samples,42)
        fid_info=(x,y,x.flatten(1)[:,:16].double(),None)
    else:fid_info=old.prepare_fid(args,test_loader,device,out)
    old.seed_everything(args.seed)
    vae,head=build(args,device)
    source_sha=None
    checkpoint=out/'latest.pt'
    if args.stage=='frozen':
        import hashlib
        source=source_snapshot(args,base)
        if not source.is_file():raise FileNotFoundError(f'Complete the parent joint run first: {source}')
        source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
        parent=torch.load(source,map_location='cpu',weights_only=False)
        if parent['epoch']!=parent['args']['epochs']:raise ValueError('Frozen branches require the final joint snapshot')
        vae.load_state_dict(parent['vae']);head.load_state_dict(parent['ema'])
        vae.eval().requires_grad_(False)
    ema=copy.deepcopy(head).eval().requires_grad_(False)
    schedule=old.ref.make_schedule(dict(time_schedule='log_t',t_min=args.t_min,t_max=args.t_max,num_train_timesteps=1000),device)
    band_weight=old.tmin_band_coefficient(args,schedule)*(args.rep_multiplier if args.band_mode=='scaled' else 1)
    if args.stage=='joint':
        opt=torch.optim.AdamW([dict(params=vae.parameters(),lr=args.lr_vae),dict(params=head.parameters(),lr=args.lr_head/args.score_weight)],weight_decay=1e-4,betas=(.9,.95))
    else:opt=torch.optim.AdamW(head.parameters(),lr=args.lr_frozen,weight_decay=1e-4,betas=(.9,.95))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=1e-6 if args.stage=='joint' else 2e-6)
    disc=old.ref.PatchDiscriminator(in_channels=1,ndf=64,n_layers=2).to(device) if args.gan_weight and args.stage=='joint' else None
    opt_disc=torch.optim.AdamW(disc.parameters(),lr=args.lr_disc,betas=(.5,.9),weight_decay=0) if disc is not None else None
    if args.perc_weight and args.stage=='joint' and not old.ref.LPIPS_AVAILABLE:raise RuntimeError('Install lpips or use synthetic smoke')
    lpips_net=old.ref.lpips.LPIPS(net='vgg').to(device).eval().requires_grad_(False) if args.perc_weight and args.stage=='joint' else None
    first=0
    if resume and checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['version']!=models.VERSION:raise ValueError('Wrong checkpoint format')
        if saved.get('source_sha256')!=source_sha:raise ValueError('Frozen source snapshot changed')
        vae.load_state_dict(saved['vae']);head.load_state_dict(saved['head']);ema.load_state_dict(saved['ema'])
        opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        if disc is not None:
            disc.load_state_dict(saved['disc']);opt_disc.load_state_dict(saved['disc_optimizer'])
        first=int(saved['epoch']);trim_uncommitted(out,first);old.restore_rng(saved)
    elif checkpoint.exists():raise FileExistsError('Existing latest.pt needs --resume')
    elif (out/'loss.csv').exists():raise FileExistsError('Partial run without checkpoint; use --restart to archive it')
    config=dict(vars(args),version=models.VERSION,latent_dimensions=vae.dimensions,latent_shapes=vae.shapes,
                vae_parameters=sum(p.numel() for p in vae.parameters()),head_parameters=sum(p.numel() for p in head.parameters()),
                band_weight_resolved=band_weight,source_sha256=source_sha,
                encoder_score_coefficient=args.score_weight*args.rep_multiplier if args.stage=='joint' else 0.,
                head_score_coefficient=args.score_weight if args.stage=='joint' else 1.,
                head_optimizer_lr=args.lr_head/args.score_weight if args.stage=='joint' else args.lr_frozen,
                vae_trainable=args.stage=='joint',image_losses_active=args.stage=='joint')
    config['parameter_audit']=models.parameter_audit(vae,head)
    import platform
    import torchvision
    config['runtime']=dict(python=platform.python_version(),torch=torch.__version__,
        torchvision=torchvision.__version__,cuda=torch.version.cuda,
        device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None)
    config['posterior_interpretation']=('single image-conditional diagonal Gaussian' if args.stochastic_blocks==1
                                       else f'{args.stochastic_blocks} ancestral Gaussian blocks')
    atomic_json(out/'parameter_audit.json',config['parameter_audit'])
    atomic_json(out/'config.json',config);print('[CONFIG]',json.dumps(config),flush=True)
    evaluate=eval_epoch if diagnostics_override is None else diagnostics_override
    if first==0 and args.diagnostic_at_start:
        evaluate(args,0,vae,head,ema,train_loader,test_loader,schedule,device,out,fid_info,lpips_net,disc,synthetic)
    # An epoch-zero checkpoint makes an early disconnection resumable too.
    if not checkpoint.exists():save_checkpoint(checkpoint,first,args,vae,head,ema,opt,scheduler,disc,opt_disc,source_sha)
    # Recover a crash between committing an archived epoch and writing its
    # immutable model snapshot, including the final epoch before complete.json.
    if first and (first in args.archive_epochs or first==args.epochs):
        snapshot=out/'snapshots'/f'epoch_{first:04d}.pt'
        if not snapshot.exists():save_snapshot(snapshot,first,args,vae,head,ema)
    with StopFlag() as stopping:
        for epoch in range(first,args.epochs):
            head.train()
            if args.stage=='joint':vae.train()
            if disc is not None:disc.train()
            sums={};count=0;started=time.time()
            for iteration,(x,labels) in enumerate(train_loader):
                if args.max_batches and iteration>=args.max_batches:break
                x,labels=x.to(device),labels.to(device)
                if args.stage=='joint':
                    score,terminal,band,d=training_losses(args,vae,head,x,labels,schedule)
                    recon=F.mse_loss(d['reconstruction'],x)
                    perc=lpips_net(d['reconstruction'].repeat(1,3,1,1),x.repeat(1,3,1,1)).mean() if lpips_net is not None else x.new_zeros(())
                    if disc is not None and epoch+1>=args.disc_start:
                        opt_disc.zero_grad(set_to_none=True)
                        dl=old.ref.hinge_d_loss(disc(x),disc(d['reconstruction'].detach()));dl.backward();opt_disc.step()
                        disc.requires_grad_(False);gl=old.ref.hinge_g_loss(disc(d['reconstruction']));disc.requires_grad_(True)
                    else:dl=x.new_zeros(());gl=x.new_zeros(())
                    loss=recon+args.perc_weight*perc+args.score_weight*score+args.terminal_weight*terminal+band_weight*band+args.gan_weight*gl
                    values=dict(recon=recon,score=score,terminal=terminal,band=band,lpips=perc,gan_d=dl,gan_g=gl)
                    for i,(m,l) in enumerate(d['params']):values.update({f'score_{i+1}':d['score_per_block'][i],f'var_{i+1}':l.exp().mean()})
                else:
                    with torch.no_grad():
                        t=schedule['times'][torch.randint(1000,(len(x),),device=device)]
                        d=draw_noisy(vae,x,t)
                    target={'naive':lambda:d['naive'],'dsm':lambda:d['eps'],
                            'corrected':lambda:corrected_teacher(vae,d)}[args.teacher]()
                    drop=labels.clone();drop[torch.rand(len(x),device=device)<args.label_dropout]=10
                    loss=F.mse_loss(head(d['packed'],t,drop),pack(target))
                    values=dict(score=loss)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                opt.zero_grad(set_to_none=True);loss.backward()
                vp=[p for p in vae.parameters() if p.requires_grad and p.grad is not None]
                hp=[p for p in head.parameters() if p.grad is not None]
                vn=sum((p.grad.detach().square().sum() for p in vp),torch.zeros((),device=device)).sqrt()
                hn=sum((p.grad.detach().square().sum() for p in hp),torch.zeros((),device=device)).sqrt()
                if args.clip_mode=='separate' and args.stage=='joint':
                    nn.utils.clip_grad_norm_(vp,1.);nn.utils.clip_grad_norm_(hp,1.)
                else:nn.utils.clip_grad_norm_(vp+hp,1.)
                if not torch.isfinite(vn+hn):raise FloatingPointError('Nonfinite gradients')
                opt.step();old.ema_update(head,ema,args.ema)
                joint_factor=torch.clamp(1/(torch.sqrt(vn.square()+hn.square())+1e-6),max=1)
                vfactor=torch.clamp(1/(vn+1e-6),max=1) if args.clip_mode=='separate' else joint_factor
                hfactor=torch.clamp(1/(hn+1e-6),max=1) if args.clip_mode=='separate' else joint_factor
                values.update(total=loss,vae_grad_norm=vn,head_grad_norm=hn,
                              joint_clip_factor=joint_factor,vae_clip_factor=vfactor,head_clip_factor=hfactor)
                for key,value in values.items():sums[key]=sums.get(key,0.)+float(value.detach())
                count+=1
            if count==0:raise ValueError('No minibatches')
            scheduler.step();ep=epoch+1
            row=dict(epoch=ep,seconds=round(time.time()-started,3),**{k:v/count for k,v in sums.items()})
            sampling.upsert(out/'loss.csv',row,('epoch',));print('[TRAIN]',json.dumps(row),flush=True)
            evaluate(args,ep,vae,head,ema,train_loader,test_loader,schedule,device,out,fid_info,lpips_net,disc,synthetic)
            if ep%args.checkpoint_every==0 or ep==args.epochs or stopping.stop or ep==stop_after:
                save_checkpoint(checkpoint,ep,args,vae,head,ema,opt,scheduler,disc,opt_disc,source_sha)
            if ep in args.archive_epochs or ep==args.epochs:
                save_snapshot(out/'snapshots'/f'epoch_{ep:04d}.pt',ep,args,vae,head,ema)
            if (stopping.stop or ep==stop_after) and ep<args.epochs:return 75
    atomic_json(out/'complete.json',dict(epoch=args.epochs,version=models.VERSION,source_sha256=source_sha))
    return 0
