"""Paired fixed-encoder teacher interventions, with full optimizer/RNG resume."""
import copy, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import core
from stacked import build_models,noisy,corrected
from audit import run_audit
from evaluation import evaluate
from design import rows,config_for


def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def save(path,value):
    torch.save(value,str(path)+'.tmp');Path(str(path)+'.tmp').replace(path)

def train(row,base,data_root=None,resume=False):
    source=rows()[int(row['source'])];source_dir=Path(base)/'results'/source['result_name']
    path=source_dir/'checkpoints/training_state_latest.pt'
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['epoch']!=800 or state.get('pending_eval') or state.get('stage')!='joint':raise ValueError('Frozen source must be final evaluated epoch 800')
    scfg=config_for(source,base,data_root)
    if state['cfg']['bundle_recipe_hash']!=scfg['bundle_recipe_hash']:raise ValueError('Source code/config changed')
    cfg=config_for(row,base,data_root);cfg.update(num_classes=10,img_size=32,img_channels=3)
    out=Path(cfg['results_dir']);out.mkdir(parents=True,exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
    total_epochs=int(row['epochs_joint'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    core.seed_everything(cfg['seed']+91000)
    vae,head=build_models(scfg,device);vae.load_state_dict(state['vae']);head.load_state_dict(state['score_ema'])
    vae.eval()
    for p in vae.parameters():p.requires_grad_(False)
    source_head=copy.deepcopy(head).eval()
    for p in source_head.parameters():p.requires_grad_(False)
    ema=copy.deepcopy(source_head)
    train_loader,test_loader,_=core.make_dataloaders(cfg['batch_size'],cfg['num_workers'],'CIFAR',cfg['data_root'])
    fid_model,use_lenet=core.get_fid_model('CIFAR',train_loader,10,device,str(out/'checkpoints'),use_bespoke_classifier=False)
    # Use the recovered CIFAR refinement LR, no FMNIST LR/EMA substitution.
    opt=torch.optim.AdamW(head.parameters(),lr=cfg['lr_refine'],betas=(.9,.95),weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,total_epochs,eta_min=1e-6)
    cfg.update(frozen_lr=cfg['lr_refine'],frozen_auxiliary_loss=0.,frozen_ema=cfg['ema_decay'],
               frozen_source_epoch=800,source_weight_hash=file_digest(path))
    ckpt=out/'checkpoints/frozen_latest.pt';start=0;losses=[];evals=[];pending=False
    from rng_state import capture_rng_state,restore_rng_state
    if resume and ckpt.exists():
        s=torch.load(ckpt,map_location=device,weights_only=False)
        for key in ('bundle_recipe_hash','source_weight_hash'):
            if s['cfg'][key]!=cfg[key]:raise ValueError('Frozen resume mismatch: '+key)
        head.load_state_dict(s['head']);ema.load_state_dict(s['ema']);opt.load_state_dict(s['optimizer']);sched.load_state_dict(s['scheduler'])
        start=s['epoch'];losses=s['losses'];evals=s['evals'];pending=s['pending'];restore_rng_state(s['rng'])
    else:core.seed_everything(cfg['seed']+92000)
    def checkpoint(ep,pending=False):
        save(ckpt,dict(epoch=ep,cfg=cfg,head=head.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
            scheduler=sched.state_dict(),losses=losses,evals=evals,pending=pending,rng=capture_rng_state()))
    def evaluation(ep):
        value=evaluate(ep,'Frozen',vae,ema,test_loader,cfg,device,None,results_dir=str(out),fid_model=fid_model,
                       use_lenet_fid=use_lenet,oracle_reference_loader=None)
        value.update(epoch=ep,stage='frozen',tag='LSI_Diff');evals.append(value)
        run_audit(ep,vae,ema,train_loader.dataset,test_loader.dataset,cfg,device,
                  heads={'source':source_head,'ema':ema},out_name='audit')
        pd.DataFrame(evals).to_csv(out/'eval_metrics.csv',index=False)
    if start==0 and not evals:evaluation(0);checkpoint(0)
    if pending:evaluation(start);checkpoint(start)
    schedule=core.make_schedule(cfg,device)
    for ep in range(start,total_epochs):
        head.train();total=0.;n=0
        for x,y in train_loader:
            x=x.to(device);y=y.to(device)
            idx=torch.randint(len(schedule['times']),(len(x),),device=device);t=schedule['times'][idx].float()
            with torch.no_grad():d=noisy(vae,x,t)
            target=d['naive'] if row['teacher']=='naive' else d['eps'] if row['teacher']=='dsm' else corrected(vae,d)
            y=y.clone();y[torch.rand(len(y),device=device)<cfg['cfg_label_dropout']]=10
            loss=(head(d['y'],t,y)-target).square().mean()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
            with torch.no_grad():
                for p,q in zip(ema.parameters(),head.parameters()):p.lerp_(q,1-cfg['ema_decay'])
            total+=loss.item();n+=1
        sched.step();losses.append(dict(epoch=ep+1,head_loss=total/max(n,1)))
        checkpoint(ep+1,pending=ep+1==total_epochs)
        if ep+1==total_epochs:evaluation(total_epochs);checkpoint(total_epochs)
        pd.DataFrame(losses).to_csv(out/'loss_history.csv',index=False)
    from launcher_base import atomic_json
    atomic_json(out/'paper_cell_config.json',cfg)
    return cfg
