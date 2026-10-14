"""Read-only hybrid-FID and finite-reference conditional CSEM diagnostics.

Risk estimates use analytic Gaussian components with importance weights for
observed ancestors. They approximate population Bayes risk, not certify it.
"""
from __future__ import annotations
import itertools
import math
from pathlib import Path
import numpy as np
import torch

_IMAGE_BANK = {}


def add_cli(p):
    p.add_argument('--eval-all-combinations', action=__import__('argparse').BooleanOptionalAction, default=True)
    p.add_argument('--eval-risk-samples', type=int, default=128)
    p.add_argument('--eval-risk-bank-sizes', default='64:256', help='Reference components PER CLASS, nested banks')
    p.add_argument('--eval-risk-time-bins', type=int, default=8)
    p.add_argument('--eval-block-cfg-samples', type=int, default=1024)
    p.add_argument('--eval-block-cfg-scales', default='1:3:5', help='Change one block at a time on a paired subset')


def validate_cli(a, p):
    try: sizes = [int(n) for n in a.eval_risk_bank_sizes.split(':')]
    except ValueError: p.error('Risk bank sizes must be positive colon-separated integers')
    if not sizes or min(sizes) < 2 or sizes != sorted(set(sizes)):
        p.error('Risk bank sizes must be unique increasing integers >=2')
    if a.eval_risk_samples < 0 or a.eval_risk_time_bins < 1:
        p.error('Risk samples >=0 and time bins >=1 are required')
    try: cfg_scales=[float(n) for n in a.eval_block_cfg_scales.split(':')]
    except ValueError: p.error('Block guidance scales must be numeric colon-separated values')
    if not cfg_scales or any(not math.isfinite(n) or n<0 for n in cfg_scales):
        p.error('Block guidance scales must be finite and nonnegative')
    if a.eval_block_cfg_samples<0 or a.eval_block_cfg_samples==1:
        p.error('Block guidance sample count must be 0 or >=2')


def missing_masks(args, heads):
    if not getattr(args, 'eval_all_combinations', True): return ()
    k=len(heads)
    native={'G'*k, 'E'*k}
    if k>=2: native.add('E'+'G'*(k-1))
    if k==3: native.add('EEG')
    return tuple(''.join(m) for m in itertools.product('EG', repeat=k) if ''.join(m) not in native)


@torch.no_grad()
def hybrid_batch(engine, args, vae, heads, encoded, bank, labels, device, masks):
    """Encoded entries keep the same posterior draw; generated entries use realized parents."""
    devices=[device.index or 0] if device.type=='cuda' else []
    outputs={}
    with torch.random.fork_rng(devices=devices):
        sampler=engine.make_sampler(args)
        for mask in masks:
            latents=[]
            for i, choice in enumerate(mask):
                if choice=='E': value=encoded[i]
                else:
                    head=heads[i]
                    if i:
                        context=(latents[0] if len(heads)==2 else
                                 tuple(latents[j] for j in engine.conditioning_indices(args,i)))
                        head=engine.BoundConditionalHead(head,context)
                    value=sampler.sample(head,shape=(len(labels),*engine.latent_shapes(args)[i]),
                        device=device,x_init=bank[i].clone(),y=labels,
                        cfg_scale=(getattr(args,'cfg_per_block',None) or [args.cfg]*len(heads))[i])
                latents.append(value)
            t=torch.full((len(labels),),args.decode_time,device=device)
            outputs[mask]=vae.decode(*latents,t=t).clamp(-1,1).cpu()
    return outputs


@torch.no_grad()
def write_combinations(engine,args,epoch,vae,heads,extra,primary,fid_info,device,out):
    _,_,real_features,feature_model=fid_info
    k=len(heads)
    native={'G'*k:primary['fid_generated'],'E'*k:primary['fid_posterior_recon']}
    if k>=2: native['E'+'G'*(k-1)]=primary['fid_encoded_root_generated_child']
    if k==3: native['EEG']=primary['fid_encoded_z1_z2_generated_z3']
    rows=[]
    for mask in sorted(set(native)|set(extra)):
        fid=native.get(mask)
        if fid is None:
            images=torch.cat(extra[mask])
            fid=engine.fid_numeric(real_features,engine.fid_features(images,feature_model,args,device))
            sample_dir=Path(out)/'samples'/f'epoch_{epoch:04d}'
            engine.to_grid(images,sample_dir/f'hybrid_{mask}.png')
            np.save(sample_dir/f'hybrid_{mask}_first{args.save_array_samples}.npy',images[:args.save_array_samples].numpy())
        # Generated ancestors with encoded descendants are interventions, not
        # the normal ancestral generative distribution.
        stress='GE' in mask
        row=dict(epoch=epoch,combination=mask,kind='conditioning_stress' if stress else 'ancestral_prefix',
                 fid=float(fid),fid_backend=args.fid_backend,cfg=args.cfg,rk4_steps=args.rk4_steps,
                 num_eval=len(fid_info[0]),conditioning=getattr(args,'conditioning','full' if k>1 else 'none'))
        engine.write_csv(Path(out)/'eval_fid_combinations.csv',row); rows.append(row)
    print('[EVAL_COMBINATIONS]',__import__('json').dumps(rows),flush=True)


@torch.no_grad()
def evaluate_block_guidance(engine,args,epoch,vae,heads,fid_info,device,out):
    """Paired subset: one guidance coordinate changes, ancestors/starts are fixed."""
    import copy
    count=min(len(fid_info[0]),getattr(args,'eval_block_cfg_samples',1024))
    if count<2: return
    scales=[float(s) for s in args.eval_block_cfg_scales.split(':')]
    real,labels,real_features,feature_model=fid_info;k=len(heads)
    settings=[(0,args.cfg,[args.cfg]*k)]
    for block in range(k):
        for scale in scales:
            if scale==args.cfg: continue
            cfg=[args.cfg]*k;cfg[block]=scale;settings.append((block+1,scale,cfg))
    masks=['G'*k]
    if k>=2: masks.append('E'+'G'*(k-1))
    if k==3: masks.append('EEG')
    outputs={(tuple(cfg),mask):[] for _,_,cfg in settings for mask in masks}
    generators=[torch.Generator(device='cpu').manual_seed(args.seed+22001+i) for i in range(k)]
    for off in range(0,count,args.eval_batch):
        x=real[off:min(off+args.eval_batch,count)].to(device)
        y=labels[off:off+len(x)].to(device)
        encoded,_=vae.encode_sample(x)
        bank=[torch.randn(len(x),*shape,generator=g).to(device)
              for shape,g in zip(engine.latent_shapes(args),generators)]
        for _,_,cfg in settings:
            settings_args=copy.copy(args);settings_args.cfg_per_block=cfg
            batch=hybrid_batch(engine,settings_args,vae,heads,encoded,bank,y,device,masks)
            for mask,images in batch.items(): outputs[tuple(cfg),mask].append(images)
    fid_values={};reference={}
    for changed,scale,cfg in settings:
        for mask in masks:
            image=torch.cat(outputs[tuple(cfg),mask])
            features=engine.fid_features(image,feature_model,args,device)
            fid=engine.fid_numeric(real_features[:count],features);fid_values[tuple(cfg),mask]=fid
            if changed==0: reference[mask]=image
            baseline=reference[mask]
            row=dict(epoch=epoch,changed_block=changed,scale=scale,
                cfg_vector=':'.join(map(str,cfg)),combination=mask,fid=fid,
                mse_to_default=float((image-baseline).square().mean()),
                num_eval=count,rk4_steps=args.rk4_steps,fid_backend=args.fid_backend)
            engine.write_csv(Path(out)/'eval_block_cfg.csv',row)
    print('[EVAL_BLOCK_CFG]',__import__('json').dumps(dict(epoch=epoch,num_eval=count,settings=len(settings))),flush=True)


def log_density(point,mu,lv):
    """Unnormalized Gaussian log density; omitted constant cancels across components."""
    return -.5*((point-mu).square()*torch.exp(-lv)+lv).flatten(1).sum(1)


def mixture_oracle(point, mu, lv, log_weights, alpha, sigma):
    variance=(alpha.square()*lv.exp()+sigma.square()).clamp_min(1e-10)
    log_joint=log_weights-.5*((point-alpha*mu).square()/variance+variance.log()).flatten(1).sum(1)
    resp=torch.softmax(log_joint.double(),dim=0).to(mu.dtype)
    teacher=sigma*(point-alpha*mu)/variance
    oracle=(resp.view(-1,1,1,1)*teacher).sum(0,keepdim=True)
    ess=float((1/resp.double().square().sum()).detach())
    prior_resp=torch.softmax(log_weights.double(),dim=0)
    return oracle,ess,float(resp.max().detach()),float((1/prior_resp.square().sum()).detach())


def image_bank(engine,args,max_bank):
    key=(str(args.data_root),args.seed,max_bank)
    if key not in _IMAGE_BANK:
        # Cache hits and misses leave the same RNG state for the risk probe.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed+31001)
            loader,_,_=engine.ref.make_dataloaders(args.batch,args.num_workers,'FMNIST')
            images=[[] for _ in range(10)]
            for x,y in loader:
                for xi,yi in zip(x,y):
                    bucket=images[int(yi)]
                    if len(bucket)<max_bank: bucket.append(xi.cpu())
                if all(len(v)==max_bank for v in images): break
        if any(len(v)<max_bank for v in images): raise ValueError('Insufficient training reference images per class')
        _IMAGE_BANK[key]=torch.stack([torch.stack(v) for v in images])
    return _IMAGE_BANK[key]


@torch.no_grad()
def reference_components(engine,args,vae,reference,query_latents,block):
    """Integrate hidden ancestors by ancestral component draws; weight visible ancestors.

    q2|z1: weights q1(z1|x). q3|z1,z2: weights q1*q2.
    q3|z2: sample hidden z1~q1(.|x), weights q2(z2|x,z1).
    q3|z1: sample hidden z2~q2(.|x,z1), weights q1(z1|x).
    """
    mu1,lv1,features,z1_draw=reference
    n=len(mu1); device=mu1.device
    weights=torch.zeros(n,device=device)
    if block==0: return mu1,lv1,weights
    observed_root=(block==1 or getattr(args,'conditioning','full')!='markov')
    z1=query_latents[0].expand_as(mu1) if observed_root else z1_draw
    if observed_root: weights=log_density(z1,mu1,lv1)
    top=vae.ancestor_features(z1)
    mu2,lv2=vae.encode_conditional(features,z1,top)
    if block==1: return mu2,lv2,weights
    topology=getattr(args,'conditioning','full')
    if topology=='siblings':
        z2=mu2+(.5*lv2).exp()*torch.randn_like(mu2)
    else:
        z2=query_latents[1].expand_as(mu2)
        weights=weights+log_density(z2,mu2,lv2)
    mu3,lv3=vae.encode_conditional3(features,z1,z2,top)
    return mu3,lv3,weights


@torch.no_grad()
def _evaluate_risk(engine,args,epoch,vae,heads,online_heads,fid_info,schedule,device,out):
    count=min(len(fid_info[0]),getattr(args,'eval_risk_samples',128))
    if not count: return
    sizes=[int(n) for n in args.eval_risk_bank_sizes.split(':')]; maximum=max(sizes)
    images=image_bank(engine,args,maximum)
    vae.eval();heads.eval()
    if online_heads is not None: online_heads.eval()
    references=[]
    for class_images in images:
        params=[]
        for batch in class_images.split(args.eval_batch):
            if len(heads)==1:
                mu,lv=vae.vae.encode(batch.to(device));features=None
            else: mu,lv,features=vae.encode_root(batch.to(device))
            params.append((mu,lv,features))
        mu=torch.cat([v[0] for v in params]);lv=torch.cat([v[1] for v in params])
        feat=torch.cat([v[2] for v in params]) if len(heads)>1 else None
        z1=mu+(.5*lv).exp()*torch.randn_like(mu)
        references.append((mu,lv,feat,z1))
    # Fixed stratified grid-index draws match the uniform log-grid training measure.
    real,labs,_,_=fid_info
    records=[]
    bins=args.eval_risk_time_bins; grid=len(schedule['times'])
    sources=[('ema',heads)]+([('online',online_heads)] if online_heads is not None else [])
    for j in range(count):
        x=real[j:j+1].to(device);label=labs[j:j+1].to(device)
        latents,params=vae.encode_sample(x)
        time_bin=j%bins
        lo=time_bin*grid//bins;hi=(time_bin+1)*grid//bins
        index=int(torch.randint(lo,hi,(1,),device=device))
        t=schedule['times'][index:index+1];alpha=schedule['alpha'][index];sigma=schedule['sigma'][index]
        reference=references[int(label)]
        for block,(latent,(mu,lv)) in enumerate(zip(latents,params)):
            noisy=alpha*latent+sigma*torch.randn_like(latent)
            target=sigma*(noisy-alpha*mu)/(alpha.square()*lv.exp()+sigma.square()).clamp_min(1e-10)
            # Hidden-ancestor integration must not move the next query's RNG.
            devices=[device.index or 0] if device.type=='cuda' else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(args.seed+530001+100*j+block)
                cmu,clv,weights=reference_components(engine,args,vae,reference,latents,block)
            predictions=[]
            for source,model in sources:
                if block==0: pred=model[block](noisy,t,label)
                else:
                    context=latents[0] if len(heads)==2 else tuple(latents[i] for i in engine.conditioning_indices(args,block))
                    pred=model[block](noisy,t,label,context=context)
                predictions.append((source,pred))
            for n in sizes:
                oracle,ess,max_resp,ancestor_ess=mixture_oracle(noisy,cmu[:n],clv[:n],weights[:n],alpha,sigma)
                half=n//2
                left,*_=mixture_oracle(noisy,cmu[:half],clv[:half],weights[:half],alpha,sigma)
                right,*_=mixture_oracle(noisy,cmu[half:n],clv[half:n],weights[half:n],alpha,sigma)
                for source,pred in predictions:
                    risk=float((pred-target).square().mean())
                    bayes=float((oracle-target).square().mean())
                    excess=float((pred-oracle).square().mean())
                    cross=float(2*((pred-oracle)*(oracle-target)).mean())
                    row=dict(epoch=epoch,query=j,label=int(label),block=block+1,time_bin=time_bin,
                        time=float(t),bank_size=n,head_source=source,raw_csem_risk=risk,
                        bayes_risk_proxy=bayes,excess_risk_proxy=excess,orthogonality_residual=cross,
                        decomposition_error=risk-bayes-excess-cross,
                        cross_fitted_excess_proxy=float(((pred-left)*(pred-right)).mean()),
                        split_oracle_disagreement=float((left-right).square().mean()),
                        ess=ess,ancestor_ess=ancestor_ess,max_responsibility=max_resp,
                        low_ess=int(ess<8),conditioning=getattr(args,'conditioning','full' if len(heads)>1 else 'none'),
                        estimator='finite_reference_importance_mixture',conditioning_labels='class',cfg=1.0)
                    engine.write_csv(Path(out)/'eval_csem_risk_queries.csv',row);records.append(row)
    # Equal coordinate weighting within each block. Report individual time bins
    # as well as the uniform-grid average; std errors are across query draws.
    metrics=['raw_csem_risk','bayes_risk_proxy','excess_risk_proxy','orthogonality_residual',
             'cross_fitted_excess_proxy','split_oracle_disagreement','ess','ancestor_ess',
             'max_responsibility','low_ess','decomposition_error']
    for block in range(1,len(heads)+1):
        for n in sizes:
            for source,_ in sources:
                for time_bin in [-1]+list(range(bins)):
                    subset=[r for r in records if r['block']==block and r['bank_size']==n
                            and r['head_source']==source and (time_bin==-1 or r['time_bin']==time_bin)]
                    if not subset: continue
                    row=dict(epoch=epoch,block=block,bank_size=n,head_source=source,time_bin=time_bin,
                             num_eval=len(subset),estimator='finite_reference_importance_mixture')
                    for metric in metrics:
                        vals=np.array([r[metric] for r in subset]);row[metric]=float(vals.mean())
                        row[metric+'_stderr']=float(vals.std(ddof=1)/math.sqrt(len(vals))) if len(vals)>1 else float('nan')
                    row['coverage_flag']='low_ess' if row['low_ess']>.25 else 'inspect_bank_stability'
                    engine.write_csv(Path(out)/'eval_csem_risk.csv',row)
    print('[EVAL_CSEM_RISK]',__import__('json').dumps(dict(epoch=epoch,queries=count,
          banks_per_class=sizes,time_bins=bins,estimator='finite_reference_importance_mixture')),flush=True)


@torch.no_grad()
def evaluate_risk(engine,args,epoch,vae,heads,online_heads,fid_info,schedule,device,out):
    devices=[device.index or 0] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(args.seed+32001)
        return _evaluate_risk(engine,args,epoch,vae,heads,online_heads,fid_info,schedule,device,out)
