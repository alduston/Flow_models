"""Fixed held-out probes of dependence, score risk, curl and gradient pressure."""
from __future__ import annotations
import csv
import json
import math
import os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from models import (pack,unpack,flat,per_sample,coeff,draw_noisy,corrected_teacher)
from pinned import stacked_two_block as old


class Log:
    def __init__(self,epoch):self.epoch,self.values=epoch,{}
    def add(self,metric,values,*,time='',conditioning='all',head='none',block=0):
        key=(str(time),conditioning,head,int(block),metric)
        arr=torch.as_tensor(values).detach().double().cpu().flatten().numpy()
        self.values.setdefault(key,[]).append(arr)
    def rows(self):
        rows=[]
        for (tv,c,h,b,m),chunks in self.values.items():
            v=np.concatenate(chunks);good=len(v)>0 and np.isfinite(v).all()
            rows.append(dict(epoch=self.epoch,time=tv,conditioning=c,head=h,block=b,metric=m,
                mean=float(v.mean()) if good else None,
                se=float(v.std(ddof=1)/np.sqrt(len(v))) if good and len(v)>1 else None,n=len(v)))
        return rows
    def save(self,out,settings):
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        rows=self.rows();file=out/'metrics.csv'
        def ident(r):return tuple(str(r[k]) for k in ('epoch','time','conditioning','head','block','metric'))
        updates={ident(r) for r in rows};prior=[]
        if file.exists():
            with file.open() as f:prior=[r for r in csv.DictReader(f) if ident(r) not in updates]
        if rows:
            with file.with_suffix('.tmp').open('w',newline='') as f:
                wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(prior+rows)
            os.replace(file.with_suffix('.tmp'),file)
        report=out/f'epoch_{self.epoch:04d}.json'
        if report.exists():
            prev=json.loads(report.read_text());saved=[r for r in prev['rows'] if ident(r) not in updates]+rows
        else:saved=rows
        report.write_text(json.dumps(dict(settings=settings,rows=saved),indent=2,allow_nan=False)+'\n')
        arrays={f'{m}|t={t}|{c}|{h}|b={b}':np.concatenate(v) for (t,c,h,b,m),v in self.values.items()}
        raw=out/f'epoch_{self.epoch:04d}_probes.npz'
        if raw.exists():
            with np.load(raw,allow_pickle=False) as f:arrays={**{k:f[k] for k in f.files},**arrays}
        np.savez_compressed(raw,**arrays)
        return rows


def subset(dataset,n,seed):
    gen=torch.Generator().manual_seed(seed)
    ids=torch.randperm(len(dataset),generator=gen)[:min(n,len(dataset))].tolist()
    pairs=[dataset[i] for i in ids]
    return torch.stack([p[0] for p in pairs]),torch.tensor([int(p[1]) for p in pairs]),ids


def symkl(m,l,n,k):
    return .25*((l-k).exp()+(k-l).exp()-2+(m-n).square()*((-l).exp()+(-k).exp()))


def dependence(args,vae,images,device,log):
    k=args.diagnostic_particles
    agg=[]
    for off in range(0,len(images),args.diagnostic_batch):
        x=images[off:off+args.diagnostic_batch].to(device);n=len(x)
        with torch.no_grad():
            xx=x[:,None].expand(-1,k,*x.shape[1:]).flatten(0,1)
            d=vae.draw(xx)
            zs=[z.reshape(n,k,-1).double() for z in d['z']]
            agg.append(torch.cat([z[:,0] for z in zs],1).cpu())
            for i,(m,l) in enumerate(d['params']):
                m=m.reshape(n,k,-1).double();l=l.reshape(n,k,-1).double();v=l.exp()
                log.add('representation/variance',v.mean((1,2)),block=i+1)
                log.add('representation/mean_energy',m.square().mean((1,2)),block=i+1)
                if i==0:continue
                mc=m-m.mean(1,keepdim=True);cov=mc.transpose(1,2)@mc/(k-1)
                bv=cov.diagonal(dim1=1,dim2=2)+v.mean(1)
                corr=cov/(bv[:,:,None]*bv[:,None,:]).clamp_min(1e-24).sqrt()
                offdiag=corr.square().sum((1,2))-corr.diagonal(dim1=1,dim2=2).square().sum(1)
                width=m.shape[-1]
                log.add('representation/conditional_symkl_per_dim',symkl(m[:,::2],l[:,::2],m[:,1::2],l[:,1::2]).mean((1,2)),block=i+1)
                log.add('representation/mean_parent_snr',(cov.diagonal(dim1=1,dim2=2)/v.mean(1).clamp_min(1e-20)).mean(1),block=i+1)
                log.add('representation/logvar_parent_variance',l.var(1,unbiased=True).mean(1),block=i+1)
                log.add('representation/child_offdiag_corr_rms',(offdiag.clamp_min(0)/(width*(width-1))).sqrt(),block=i+1)
                for parent in range(i):
                    a=zs[parent]-zs[parent].mean(1,keepdim=True)
                    cross=a.transpose(1,2)@mc/(k-1)
                    av=a.square().sum(1)/(k-1)
                    cross=cross/(av[:,:,None]*bv[:,None,:]).clamp_min(1e-24).sqrt()
                    log.add(f'representation/parent{parent+1}_cross_corr_rms',cross.square().mean((1,2)).sqrt(),block=i+1)
            # Decoder interventions preserve X; zero masks are additional capacity controls.
            native=[z.reshape(n,k,*s) for z,s in zip(d['z'],vae.shapes)]
            first=[z[:,0] for z in native]
            ts=torch.full((n,),args.decode_time,device=device)
            baseline=per_sample((vae.decode(first,ts)-x).square())
            log.add('decoder/posterior_mse',baseline)
            for i in range(vae.blocks):
                for kind,replacement in [('same_image_swap',native[i][:,1]),('zero',torch.zeros_like(first[i]))]:
                    repl=list(first);repl[i]=replacement
                    delta=per_sample((vae.decode(repl,ts)-x).square())-baseline
                    log.add('decoder/'+kind+'_delta',delta,block=i+1)
        # Per-edge LOCAL derivatives hold every other sampled parent fixed.
        with torch.no_grad():d0=vae.draw(x)
        for i in range(1,vae.blocks):
            with torch.enable_grad():
                parents=[z.detach().requires_grad_(True) for z in d0['z'][:i]]
                m,l=vae.moment(i,d0['features'].detach(),d0['root_mu'].detach(),d0['root_lv'].detach(),parents)
                for kind,output in [('mean',m),('logvar',l)]:
                    probe=torch.empty_like(output).bernoulli_(.5).mul_(2).sub_(1)
                    gs=torch.autograd.grad((output*probe).sum(),parents,retain_graph=True,allow_unused=True)
                    for j,g in enumerate(gs):
                        value=torch.zeros(n,device=device) if g is None else flat(g.square()*d0['params'][j][1].exp()).sum(1)
                        denom=flat(l.detach().exp()).sum(1) if kind=='mean' else output[0].numel()
                        log.add(f'representation/local_parent{j+1}_{kind}_jac_energy',value/denom,block=i+1)
    values=torch.cat(agg);values=values-values.mean(0)
    cov=values.T@values/max(1,len(values)-1)
    diag=cov.diagonal().clamp_min(1e-24)
    corr=cov/(diag[:,None]*diag[None,:]).sqrt()
    log.add('aggregate/offdiag_corr_rms',((corr.square().sum()-corr.diagonal().square().sum())/(256*255)).sqrt().reshape(1))


def pressures(args,vae,head,images,labels,schedule,device,log,lpips_net=None,disc=None,epoch=0):
    if not any(p.requires_grad for p in vae.parameters()):return
    with torch.enable_grad():
        x=images[:args.diagnostic_batch].to(device);lab=labels[:len(x)].to(device)
        gs=x.new_ones(3,requires_grad=True)
        gates=tuple(gs[j]*g for j,g in enumerate(vae.default_gates))
        times=schedule['times']
        t=times[torch.linspace(0,len(times)-1,len(x),device=device).long()]
        d=draw_noisy(vae,x,t,gates=gates)
        prediction=head(d['packed'],t,lab)
        score=F.mse_loss(prediction,pack(d['naive']))
        # Exact local chain-rule split of score pressure. Mean-dependent
        # teacher terms can cancel their Y route under coherent reparameterized
        # draws; a raw gate sign is not automatically a variance incentive.
        a,s=d['alpha'],d['sigma']
        variances=[a.square()*l.exp()+s.square() for _,l in d['params']]
        target_y=[s*(y-a*m.detach())/v.detach() for y,(m,l),v in zip(d['y'],d['params'],variances)]
        target_mean=[s*(y.detach()-a*m)/v.detach() for y,(m,l),v in zip(d['y'],d['params'],variances)]
        target_var=[s*(y.detach()-a*m.detach())/v for y,(m,l),v in zip(d['y'],d['params'],variances)]
        routes=dict(score_input_path=F.mse_loss(prediction,pack(d['naive']).detach()),
                    score_target_y_path=F.mse_loss(prediction.detach(),pack(target_y)),
                    score_target_mean_path=F.mse_loss(prediction.detach(),pack(target_mean)),
                    score_target_variance_path=F.mse_loss(prediction.detach(),pack(target_var)))
        image=vae.decode(d['y'],t);recon=F.mse_loss(image,x)
        image_loss=recon;objectives={'score':score,'recon':recon}
        if lpips_net is not None:
            lp=lpips_net(image.repeat(1,3,1,1),x.repeat(1,3,1,1)).mean()
            objectives['lpips']=lp;image_loss=image_loss+args.perc_weight*lp
        if disc is not None and epoch>=args.disc_start:
            gan=old.ref.hinge_g_loss(disc(image));objectives['gan']=gan;image_loss=image_loss+args.gan_weight*gan
        weights=[n/256 for n in vae.dimensions]
        terminal=sum(w*old.terminal_kl(m,l,schedule['alpha'][-1],schedule['sigma'][-1]) for w,(m,l) in zip(weights,d['params']))
        band=sum(w*old.tmin_band_penalty(l,old.tmin_band_floor(schedule)) for w,(m,l) in zip(weights,d['params']))
        bw=old.tmin_band_coefficient(args,schedule)*(args.rep_multiplier if args.band_mode=='scaled' else 1)
        objectives.update(terminal=terminal,band=band,weighted_total=image_loss+args.score_weight*args.rep_multiplier*score+args.terminal_weight*terminal+bw*band,**routes)
        derivs={}
        for name,loss in objectives.items():
            g=torch.autograd.grad(loss,gs,retain_graph=True,allow_unused=True)[0]
            if g is None:g=torch.zeros_like(gs)
            derivs[name]=g.detach()
            for j,edge in enumerate(('1to2','1to3','2to3')[:1 if vae.blocks==2 else 3]):
                log.add('pressure/'+name+'_gate_'+edge,g[j:j+1])
        residual=sum(derivs[name] for name in routes)-derivs['score']
        for j,edge in enumerate(('1to2','1to3','2to3')[:1 if vae.blocks==2 else 3]):
            log.add('pressure/score_chainrule_residual_'+edge,residual[j:j+1])
        params=[p for name,p in vae.named_parameters() if p.requires_grad and any(k in name for k in ('enc_','root_posterior','condition','z2_to_posterior'))]
        a=torch.autograd.grad(score,params,retain_graph=True,allow_unused=True)
        b=torch.autograd.grad(image_loss,params,allow_unused=True)
        dot=sum((u*v).sum() for u,v in zip(a,b) if u is not None and v is not None)
        aa=sum(u.square().sum() for u in a if u is not None);bb=sum(v.square().sum() for v in b if v is not None)
        log.add('pressure/score_image_gradient_cosine',(dot/(aa*bb).sqrt().clamp_min(1e-30)).reshape(1))


def score_probes(args,vae,heads,images,labels,device,log):
    for tv in args.diagnostic_times:
        for off in range(0,len(images),args.diagnostic_batch):
            x=images[off:off+args.diagnostic_batch].to(device);lab=labels[off:off+len(x)].to(device)
            t=torch.full((len(x),),tv,device=device)
            with torch.no_grad():d=draw_noisy(vae,x,t)
            valid,parts=corrected_teacher(vae,d,components=True)
            for i,(naive,target) in enumerate(zip(d['naive'],valid)):
                defect=naive-target
                log.add('score/pathwise_defect_energy',per_sample(defect.square()),time=tv,block=i+1)
                log.add('score/valid_target_energy',per_sample(target.square()),time=tv,block=i+1)
            for name,part in parts.items():
                log.add('score/descendant_correction_'+name,per_sample(part.square()),time=tv)
            for cond,y in [('class',lab),('null',torch.full_like(lab,10))]:
                preds={}
                for hn,head in heads.items():
                    with torch.no_grad():ps=unpack(head(d['packed'],t,y),vae.blocks);preds[hn]=ps
                    for i,(p,v,c,e) in enumerate(zip(ps,valid,d['naive'],d['eps'])):
                        for name,value in dict(dsm_mse=(p-e).square(),corrected_mse=(p-v).square(),
                            stacked_mse=(p-c).square(),excess_valid=p.square()-2*p*v,
                            excess_dsm=p.square()-2*p*e,teacher_calibration_gap=2*p*(v-e),
                            prediction_energy=p.square()).items():
                            log.add('score/'+name,per_sample(value),time=tv,block=i+1,conditioning=cond,head=hn)
                    if off==0:
                        with torch.enable_grad():
                            point=d['packed'].detach().requires_grad_(True);f=head(point,t,y)
                            u=torch.empty_like(f).bernoulli_(.5).mul_(2).sub_(1);v=torch.empty_like(f).bernoulli_(.5).mul_(2).sub_(1)
                            ju=torch.autograd.grad((f*u).sum(),point,retain_graph=True)[0]
                            jv=torch.autograd.grad((f*v).sum(),point,retain_graph=True)[0]
                            ranges=[(0,1),(1,4)] if vae.blocks==2 else [(0,1),(1,2),(2,4)]
                            pairs={}
                            for i in range(vae.blocks):
                                for j in range(i+1,vae.blocks):
                                    ur=torch.zeros_like(u);vc=torch.zeros_like(v)
                                    ur[:,ranges[i][0]:ranges[i][1]]=u[:,ranges[i][0]:ranges[i][1]]
                                    vc[:,ranges[j][0]:ranges[j][1]]=v[:,ranges[j][0]:ranges[j][1]]
                                    jr=torch.autograd.grad((f*ur).sum(),point,retain_graph=True)[0]
                                    jc=torch.autograd.grad((f*vc).sum(),point,retain_graph=True)[0]
                                    pairs[f'{i+1}_{j+1}']=flat(jr*vc-jc*ur).sum(1).square()/256
                        log.add('score/curl_epsilon_energy',flat(ju*v-jv*u).sum(1).square()/256,time=tv,conditioning=cond,head=hn)
                        for pair,value in pairs.items():
                            log.add('score/cross_curl_'+pair,value,time=tv,conditioning=cond,head=hn)
                if 'online' in preds and 'ema' in preds:
                    for i,(a,b,c) in enumerate(zip(preds['online'],preds['ema'],valid)):
                        log.add('score/online_minus_ema_valid_risk',per_sample(a.square()-b.square()-2*(a-b)*c),time=tv,block=i+1,conditioning=cond)


def core(args,epoch,vae,head,ema,train_dataset,test_dataset,schedule,device,out,lpips_net=None,disc=None):
    log=Log(epoch)
    with old.preserved_state(vae,head,ema,lpips_net,disc):
        vae.eval();head.eval();ema.eval()
        if disc is not None:disc.eval()
        old.seed_everything(args.seed+81000)
        x,y,ids=subset(test_dataset,args.diagnostic_samples,args.seed+81001)
        dependence(args,vae,x,device,log)
        pressures(args,vae,head,x,y,schedule,device,log,lpips_net,disc,epoch)
        score_probes(args,vae,{'online':head,'ema':ema},x,y,device,log)
    rows=log.save(Path(out)/'diagnostics',dict(version=1,test_ids=ids,test_labels=y.tolist(),
        pressure_grid='fixed batch stratified over training log-time indices'))
    # Print selected time-resolved score energies as well as dependence; full
    # table/raw probes remain primary. No repeat of the earlier console omission.
    show=[r for r in rows if (not r['time'] and r['metric'].startswith(('representation/conditional','pressure/score_gate')))
          or r['metric']=='score/pathwise_defect_energy']
    print('[DIAGNOSTICS]',json.dumps(dict(epoch=epoch,metrics=len(rows),summary=show)),flush=True)
    return rows
