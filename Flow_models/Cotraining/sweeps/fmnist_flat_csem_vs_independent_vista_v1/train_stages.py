"""Explicit joint, independent VAE, and fresh frozen-prior stages."""
from __future__ import annotations
import copy
import csv
import hashlib
import math
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from torch.nn import functional as F
import models
from models import pack,coeff
from pinned import stacked_two_block as old
from protocol import VERSION,atomic,sha,object_sha,resolved_contract
from data_protocol import loaders

def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def kl0(params,dimensions):
    """(1/(2 B D)) sum_bj [mu_bj^2 + exp(lv_bj) - 1 - lv_bj]."""
    total=sum(dimensions)
    return sum(n/total*.5*(m.square()+l.exp()-1-l).mean() for n,(m,l) in zip(dimensions,params))

def streams(seed,device):
    return {k:torch.Generator(device=device).manual_seed(seed+offset) for k,offset in
            dict(time=20001,posterior=20002,diffusion=20003,label=20004).items()}

def forward_draw(vae,x,schedule,rng):
    t=schedule['times'][torch.randint(len(schedule['times']),(len(x),),device=x.device,generator=rng['time'])]
    innovations=[torch.randn((len(x),*shape),device=x.device,generator=rng['posterior']) for shape in vae.shapes]
    diffusion=[torch.randn((len(x),*shape),device=x.device,generator=rng['diffusion']) for shape in vae.shapes]
    return models.draw_noisy(vae,x,t,innovations=innovations,diffusion=diffusion)

def prior_loss(args,vae,head,x,labels,schedule,rng,return_draw=False):
    with torch.no_grad():d=forward_draw(vae,x,schedule,rng)
    # The decoupled posterior makes the inherited naive component target exact.
    target=pack(d['eps'] if args.stage=='prior_dsm' else d['naive']).detach()
    drop=labels.clone();drop[torch.rand(len(labels),device=x.device,generator=rng['label'])<args.label_dropout]=10
    loss=F.mse_loss(head(d['packed'].detach(),d['time'],drop),target)
    return (loss,d,drop) if return_draw else loss

def capture_rng():
    return dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

def restore_rng(rng):
    torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy']);random.setstate(rng['python'])
    if rng['cuda'] is not None:torch.cuda.set_rng_state_all(rng['cuda'])

class StopFlag:
    def __init__(self):self.stop=False;self.handlers={}
    def __enter__(self):
        for sig in (signal.SIGUSR1,signal.SIGTERM):self.handlers[sig]=signal.signal(sig,self.handler)
        return self
    def handler(self,*_):
        self.stop=True;print('[SIGNAL] Checkpointing at the next epoch boundary.',flush=True)
    def __exit__(self,*_):
        for sig,h in self.handlers.items():signal.signal(sig,h)

def upsert(path,row):
    path=Path(path);prior=[]
    if path.exists():
        with path.open() as f:prior=[r for r in csv.DictReader(f) if int(r['epoch'])!=row['epoch']]
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(prior+[row])
    os.replace(tmp,path)

def trim(out,epoch):
    for name in ('loss.csv','eval.csv'):
        path=out/name
        if not path.exists():continue
        with path.open() as f:r=csv.DictReader(f);fields=r.fieldnames;rows=[x for x in r if int(x['epoch'])<=epoch]
        with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    for path in (out/'snapshots').glob('epoch_*.pt'):
        if int(path.stem.split('_')[1])>epoch:path.unlink()

def run(args,base,*,resume=True,synthetic=False,stop_after=None):
    from evaluate import prepare,evaluate,reload_parity
    base=Path(base);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device);torch.set_num_threads(args.cpu_threads)
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA GPU is not visible')
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    old.seed_everything(args.seed)
    tr,ref,data_rng,split=loaders(args,synthetic)
    fid_info=prepare(args,ref,device,out,synthetic)
    old.seed_everything(args.seed)
    vae,head=models.build(args,device)
    audit=models.parameter_audit(vae,head)
    assert vae.stochastic_blocks==1 and vae.topology=='decoupled' and not head.factored_head
    if args.joint_width==192 and args.joint_depth==18:
        assert (audit['vae'],audit['score_head'])==(6368817,12159556),audit
    initial_head_sha=state_hash(head.state_dict())
    source_sha=None;source_cost=0.;source_record=None
    if args.stage.startswith('prior'):
        parent=base/'results'/args.source
        from campaign import validate_complete
        source_record=validate_complete(parent)
        source=parent/'snapshots'/f'epoch_{args.epochs:04d}.pt'
        source_sha=sha(source);saved=torch.load(source,map_location='cpu',weights_only=False)
        pc=saved['args']
        for key in ('seed','beta0','split','eval_split','split_seed','validation_per_class','epochs'):
            if pc[key]!=getattr(args,key):raise ValueError(f'VAE parent mismatch: {key}')
        if pc['stage']!='vae_pretrain':raise ValueError('Independent priors require a true VAE-pretrain parent')
        vae.load_state_dict(saved['vae']);vae.eval().requires_grad_(False)
        source_cost=float(saved['timing']['training_seconds'])
    frozen_hash=state_hash(vae.state_dict()) if args.stage.startswith('prior') else None
    if args.stage=='vae_pretrain':head=None;ema=None
    else:ema=copy.deepcopy(head).eval().requires_grad_(False)
    schedule=old.ref.make_schedule(dict(time_schedule='log_t',t_min=args.t_min,t_max=args.t_max,num_train_timesteps=1000),device)
    band=old.tmin_band_coefficient(args,schedule) if args.stage=='joint' else 0.
    if args.stage=='joint':
        opt=torch.optim.AdamW([dict(params=vae.parameters(),lr=args.lr_vae),
                              dict(params=head.parameters(),lr=args.lr_head/args.score_weight)],betas=(.9,.95),weight_decay=1e-4)
    else:opt=torch.optim.AdamW((vae if head is None else head).parameters(),lr=args.lr_vae if head is None else args.lr_prior,betas=(.9,.95),weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=1e-6)
    image_stage=args.stage in ('joint','vae_pretrain')
    disc=old.ref.PatchDiscriminator(in_channels=1,ndf=64,n_layers=2).to(device) if image_stage and args.gan_weight else None
    opt_disc=torch.optim.AdamW(disc.parameters(),lr=args.lr_disc,betas=(.5,.9),weight_decay=0) if disc is not None else None
    if image_stage and args.perc_weight and not old.ref.LPIPS_AVAILABLE:raise RuntimeError('LPIPS must be installed in the existing hlsi environment')
    percept=old.ref.lpips.LPIPS(net='vgg').to(device).eval().requires_grad_(False) if image_stage and args.perc_weight else None
    rng=streams(args.seed,device)
    config=dict(vars(args),parameter_audit=audit,band_weight_resolved=band,split_metadata=split,
                source_sha256=source_sha,source_training_seconds=source_cost,source_recipe_sha256=source_record['recipe_sha256'] if source_record else None,
                stochastic_blocks=1,initial_head_sha256=initial_head_sha,synthetic=synthetic,
                vae_initial_frozen_sha256=frozen_hash,kl0_formula='0.5 * mean over batch and all 256 coordinates of (mu^2+exp(logvar)-1-logvar)',
                head_optimizer_lr=args.lr_head/args.score_weight if args.stage=='joint' else args.lr_prior if head is not None else None,
                encoder_score_coefficient=args.score_weight*args.rep_multiplier if args.stage=='joint' else 0.,
                evaluation_contract=resolved_contract(args),checkpoint_version=VERSION)
    # Runtime metadata does not enter the resume recipe; exact sources and settings do.
    import torchvision,platform
    config['runtime']=dict(torch=torch.__version__,torchvision=torchvision.__version__,python=platform.python_version(),
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),allocated_nodes=os.environ.get('SLURM_JOB_NUM_NODES','1'),
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    recipe=object_sha(dict(args=vars(args),sources=__import__('protocol').sources(base),source_sha256=source_sha,split=split,synthetic=synthetic))
    config['recipe_sha256']=recipe
    config['source_hashes']=__import__('protocol').sources(base)
    if (out/'complete.json').exists():
        from campaign import validate_complete
        done=validate_complete(out)
        if done['recipe_sha256']!=recipe:raise ValueError('Completed run recipe/source mismatch')
        print('[SKIP verified completed stage]',args.name,flush=True)
        return 0
    ck=out/'latest.pt';first=0
    timing=dict(training_seconds=0.,evaluation_seconds=0.,optimizer_steps=0,data_passes=0.,examples_seen=0,
                source_training_seconds=source_cost,allocation_seconds=0.)
    if ck.exists():
        if not resume:raise FileExistsError('Use resume for existing outputs')
        saved=torch.load(ck,map_location='cpu',weights_only=False)
        if saved['recipe_sha256']!=recipe:raise ValueError('Checkpoint recipe/source/split mismatch')
        vae.load_state_dict(saved['vae'])
        if head is not None:head.load_state_dict(saved['head']);ema.load_state_dict(saved['ema'])
        opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        if disc is not None:disc.load_state_dict(saved['disc']);opt_disc.load_state_dict(saved['disc_optimizer'])
        for k,g in rng.items():g.set_state(saved['named_rng'][k])
        data_rng.set_state(saved['data_rng']);restore_rng(saved['rng'])
        first=saved['epoch'];timing=saved['timing'];trim(out,first)
    elif (out/'loss.csv').exists():raise RuntimeError('Output without checkpoint: archive this attempt before restarting')
    atomic(out/'config.json',config)
    allocation_started=time.monotonic();previous_allocation=timing['allocation_seconds']
    def checkpoint(epoch):
        timing['allocation_seconds']=previous_allocation+time.monotonic()-allocation_started
        old.atomic_torch_save(dict(checkpoint_version=VERSION,recipe_sha256=recipe,epoch=epoch,args=vars(args),
            vae=vae.state_dict(),head=head.state_dict() if head is not None else None,ema=ema.state_dict() if ema is not None else None,
            optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),disc=disc.state_dict() if disc is not None else None,
            disc_optimizer=opt_disc.state_dict() if opt_disc is not None else None,named_rng={k:g.get_state() for k,g in rng.items()},
            data_rng=data_rng.get_state(),rng=capture_rng(),timing=timing,source_sha256=source_sha),ck)
    def snapshot(epoch):
        dest=out/'snapshots'/f'epoch_{epoch:04d}.pt';dest.parent.mkdir(exist_ok=True)
        old.atomic_torch_save(dict(checkpoint_version=VERSION,recipe_sha256=recipe,epoch=epoch,args=vars(args),
            vae=vae.state_dict(),head=head.state_dict() if head is not None else None,
            ema=ema.state_dict() if ema is not None else None,timing=timing,source_sha256=source_sha),dest)
    if not ck.exists():checkpoint(0)
    with StopFlag() as flag:
        for epoch in range(first,args.epochs):
            if head is not None:head.train()
            if image_stage:vae.train()
            else:vae.eval()
            if disc is not None:disc.train()
            sums={};count=0;examples=0
            if device.type=='cuda':torch.cuda.synchronize()
            started=time.monotonic()
            for iteration,(x,labels) in enumerate(tr):
                if args.max_batches and iteration>=args.max_batches:break
                x=x.to(device);labels=labels.to(device)
                if image_stage:
                    if args.stage=='joint':
                        score,terminal,penalty,d=models.training_losses(args,vae,head,x,labels,schedule)
                        regularizer=args.score_weight*score+args.terminal_weight*terminal+band*penalty
                        values=dict(score=score,terminal=terminal,band=penalty,kl0=kl0(d['params'],vae.dimensions))
                    else:
                        d=forward_draw(vae,x,schedule,rng)
                        d['reconstruction']=vae.decode(d['y'],d['time'])
                        rate=kl0(d['params'],vae.dimensions);regularizer=args.beta0*rate
                        values=dict(score=x.new_zeros(()),terminal=x.new_zeros(()),band=x.new_zeros(()),kl0=rate)
                    recon=F.mse_loss(d['reconstruction'],x)
                    perc=percept(d['reconstruction'].repeat(1,3,1,1),x.repeat(1,3,1,1)).mean() if percept is not None else x.new_zeros(())
                    if disc is not None and epoch+1>=args.disc_start:
                        opt_disc.zero_grad(set_to_none=True)
                        dl=old.ref.hinge_d_loss(disc(x),disc(d['reconstruction'].detach()));dl.backward();opt_disc.step()
                        disc.requires_grad_(False);gl=old.ref.hinge_g_loss(disc(d['reconstruction']));disc.requires_grad_(True)
                    else:dl=x.new_zeros(());gl=x.new_zeros(())
                    loss=recon+args.perc_weight*perc+args.gan_weight*gl+regularizer
                    values.update(recon=recon,lpips=perc,gan_d=dl,gan_g=gl,
                        posterior_variance=sum(n/256*l.exp().mean() for n,(m,l) in zip(vae.dimensions,d['params'])),
                        latent_rms=sum(n/256*(m.square()+l.exp()).mean() for n,(m,l) in zip(vae.dimensions,d['params'])).sqrt())
                else:
                    loss=prior_loss(args,vae,head,x,labels,schedule,rng);values=dict(score=loss)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                opt.zero_grad(set_to_none=True);loss.backward()
                params=[p for g in opt.param_groups for p in g['params'] if p.grad is not None]
                gn=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                opt.step()
                if head is not None:old.ema_update(head,ema,args.ema)
                values.update(total=loss,gradient_norm=gn)
                for key,value in values.items():sums[key]=sums.get(key,0.)+float(value.detach())
                count+=1;examples+=len(x)
            if not count:raise ValueError('No minibatches')
            if device.type=='cuda':torch.cuda.synchronize()
            elapsed=time.monotonic()-started;ep=epoch+1
            timing['training_seconds']+=elapsed;timing['optimizer_steps']+=count
            timing['examples_seen']+=examples;timing['data_passes']+=examples/len(tr.dataset)
            scheduler.step()
            if frozen_hash is not None:
                if any(p.grad is not None for p in vae.parameters()) or state_hash(vae.state_dict())!=frozen_hash:
                    raise AssertionError('Frozen VAE parameters/state changed')
            row=dict(epoch=ep,seconds=elapsed,training_seconds=timing['training_seconds'],optimizer_steps=timing['optimizer_steps'],
                     data_passes=timing['data_passes'],**{k:v/count for k,v in sums.items()})
            upsert(out/'loss.csv',row);print('[TRAIN]',__import__('json').dumps(row),flush=True)
            if ep%args.eval_every==0 or ep==args.epochs:
                started=time.monotonic();evaluate(args,ep,vae,ema,fid_info,device,out,timing,synthetic)
                timing['evaluation_seconds']+=time.monotonic()-started
            if ep in args.archive_epochs or ep==args.epochs:snapshot(ep)
            checkpoint(ep)
            if (flag.stop or ep==stop_after) and ep<args.epochs:return 75
    # Recover a crash after final checkpoint but before completion publication.
    snapshot(args.epochs)
    parity=reload_parity(args,vae,ema,out/'snapshots'/f'epoch_{args.epochs:04d}.pt',device)
    atomic(out/'reload_parity.json',parity)
    if frozen_hash is not None and state_hash(vae.state_dict())!=frozen_hash:raise AssertionError('Frozen state drift')
    complete=dict(version=VERSION,epoch=args.epochs,stage=args.stage,seed=args.seed,regime=args.regime,
        recipe_sha256=recipe,source_sha256=source_sha,vae_sha256=state_hash(vae.state_dict()),
        initial_head_sha256=initial_head_sha,split_hash=split['hash'],synthetic=synthetic,timing=timing,
        snapshot_sha256=sha(out/'snapshots'/f'epoch_{args.epochs:04d}.pt'),
        evaluation_contract=resolved_contract(args),parity=parity)
    atomic(out/'complete.json',complete)
    from campaign import validate_complete
    validate_complete(out)
    return 0
