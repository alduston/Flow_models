"""Offline contracts, baseline parity and full-size model/diagnostic smoke."""
from __future__ import annotations
import ast, csv, hashlib, importlib, importlib.util, json, sys
from pathlib import Path
from tempfile import TemporaryDirectory


def run(base):
    from suite import manifest,validate_manifest,cli_args
    from paper import select_cells,parse_slurm_job_id
    base=Path(base);rows=manifest(base);validate_manifest(rows)
    for p in base.rglob('*.py'): ast.parse(p.read_text(),filename=str(p))
    for name in ['paper.py','core.py','suite.py','paper_io.py','diagnostics.py','compile_results.py',
                 'train_hier_csem_fmnist.py','train_hier_csem_fmnist_3block.py','fmnist_reference.py',
                 'job.slurm','manifest.csv','README.md','requirements.txt','.gitignore']:
        assert (base/name).is_file(),name
    provenance=json.loads((base/'provenance.json').read_text())
    for name,digest in provenance['bundle_source_sha256'].items():
        assert hashlib.sha256((base/name).read_bytes()).hexdigest()==digest,name
    assert [int(r['cell_id']) for r in select_cells('baseline',rows)]==[0,1,2]
    assert [int(r['cell_id']) for r in select_cells('variants',rows)]==[3,4]
    assert [int(r['cell_id']) for r in select_cells('three',rows)]==[2,3,4]
    for msg in ['1067000','Submitted batch job 1067000','site banner\n1067000;vista']:
        assert parse_slurm_job_id(msg)=='1067000'
    for row in rows:
        name='train_hier_csem_fmnist_3block.py' if int(row['blocks'])==3 else 'train_hier_csem_fmnist.py'
        tree=ast.parse((base/name).read_text())
        flags={node.args[0].value for node in ast.walk(tree) if isinstance(node,ast.Call)
            and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument'
            and node.args and isinstance(node.args[0],ast.Constant)}
        flags.update(node.args[0].value for node in ast.walk(ast.parse((base/'diagnostics.py').read_text()))
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
            and node.func.attr=='add_argument' and node.args and isinstance(node.args[0],ast.Constant))
        command=cli_args(row,base)
        assert all(command[i] in flags for i in range(0,len(command),2))
    print('OK: five recipes, 256-coordinate budgets, topology-only contrasts, CLI/Slurm and source hashes')
    return 0


def engine_for(row):
    return importlib.import_module('train_hier_csem_fmnist_3block' if int(row['blocks'])==3 else 'train_hier_csem_fmnist')


def torch_smoke(base,cell_id,device='cpu'):
    import torch
    from torch.nn import functional as F
    from suite import manifest,config_for
    import diagnostics as diag
    torch.set_num_threads(min(torch.get_num_threads(),2))
    row=manifest(base)[cell_id];args=config_for(row,base);args.device=device
    engine=engine_for(row);dev=torch.device(device)
    torch.manual_seed(1024);vae,heads,_=engine.build(args,dev)
    x=torch.rand(2,1,32,32,device=dev)*2-1;y=torch.tensor([2,5],device=dev)
    schedule=engine.ref.make_schedule(dict(time_schedule='log_t',t_min=args.t_min,t_max=args.t_max,
                                         num_train_timesteps=1000),dev)
    seen={}
    for i,head in enumerate(heads):
        if i:
            original=head.prepare_context
            def record(context,i=i,original=original):
                seen[i]=context;return original(context)
            head.prepare_context=record
    sl,kl,detail=engine.score_and_losses(args,vae,heads,x,y,schedule)
    assert vae.latent_shapes==engine.latent_shapes(args)
    assert sum(detail['block_dimensions'])==256
    assert [len(h.blocks) for h in heads]==list(map(int,row['score_depths'].split(':')))
    if len(heads)==2: assert seen[1] is detail['latents'][0]
    if len(heads)==3:
        assert seen[1][0] is detail['latents'][0]
        idx=engine.conditioning_indices(args,2)
        assert all(a is detail['latents'][j] for a,j in zip(seen[2],idx))
        assert heads[2].context_specs==tuple((engine.latent_shapes(args)[j][0],engine.latent_shapes(args)[j][-1]) for j in idx)
    idx=torch.searchsorted(schedule['times'],detail['time'])
    alpha=schedule['alpha'][idx].view(2,1,1,1);sigma=schedule['sigma'][idx].view(2,1,1,1)
    for noisy,(mu,lv),target in zip(detail['noisy'],detail['params'],detail['target_eps']):
        point=noisy.detach().clone().requires_grad_(True)
        variance=alpha.square()*lv.detach().exp()+sigma.square()
        density=-.5*((point-alpha*mu.detach()).square()/variance+variance.log()).sum()
        score=torch.autograd.grad(density,point)[0]
        torch.testing.assert_close(target.detach(),-sigma*score)
    loss=F.mse_loss(detail['reconstruction'],x)+args.score_weight*sl+args.terminal_weight*kl+engine.tmin_band_coefficient(args,schedule)*detail['band_loss']
    loss.backward()
    for module in [vae,*heads]:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
    opt=torch.optim.AdamW([*vae.parameters(),*heads.parameters()],lr=1e-4)
    torch.nn.utils.clip_grad_norm_([*vae.parameters(),*heads.parameters()],1);opt.step()
    args.rk4_steps=2;args.cfg=1.;vae.eval();heads.eval()
    with torch.no_grad():
        bank=[torch.randn(2,*s,device=dev) for s in vae.latent_shapes]
        sample=engine.sample_blocks(args,heads,2,dev,y=y,bank=[z.clone() for z in bank])
        image=vae.decode(*sample,t=torch.full((2,),args.decode_time,device=dev))
        assert image.shape==x.shape and torch.isfinite(image).all()
        rng=torch.get_rng_state().clone()
        extra=diag.hybrid_batch(engine,args,vae,heads,detail['latents'],bank,y,dev,diag.missing_masks(args,heads))
        assert torch.equal(rng,torch.get_rng_state())
        assert len(extra)==(0,1,4)[len(heads)-1]
        assert all(torch.isfinite(v).all() and v.shape==x.shape for v in extra.values())
    # Risk and gradient diagnostics, end-to-end with tiny synthetic train banks.
    args.eval_risk_samples=2;args.eval_risk_bank_sizes='2:4';args.eval_risk_time_bins=2
    args.eval_grad_batch=2
    synthetic=torch.rand(10,4,1,32,32)*2-1
    oldbank=diag.image_bank;diag.image_bank=lambda *a:synthetic
    try:
        with TemporaryDirectory() as tmp:
            engine.eval_gradient_checks(args,50,vae,heads,x.cpu(),y.cpu(),schedule,dev,Path(tmp))
            diag.evaluate_risk(engine,args,50,vae,heads,heads,(x.cpu(),y.cpu(),None,None),schedule,dev,Path(tmp))
            with (Path(tmp)/'eval_csem_risk_queries.csv').open() as f:records=list(csv.DictReader(f))
            assert len(records)==2*len(heads)*2*2
            assert all(abs(float(r['decomposition_error']))<1e-5 for r in records)
            assert all(float(r['ess'])>=1-1e-5 for r in records)
            if len(heads)==3:
                with (Path(tmp)/'eval_gradient_checks.csv').open() as f:gradients=list(csv.DictReader(f))
                joint=next(r for r in gradients if r['term']=='joint')
                assert abs(float(joint['cosine_csem2_csem3_z2_posterior']))<=1+1e-6
        # Entire evaluation path with a lightweight feature surrogate. This tests
        # the plumbing and RNG isolation, not pretrained Inception or FMNIST FID.
        args.eval_block_cfg_samples=2;args.eval_block_cfg_scales='1:3'
        args.eval_check_samples=2;args.eval_usage_samples=2
        old_features=engine.fid_features
        engine.fid_features=lambda images,*a:images.flatten(1)[:,:8].cpu()
        try:
            vae.train();heads.train()
            before=torch.get_rng_state().clone()
            old_grads=[None if p.grad is None else p.grad.clone() for p in [*vae.parameters(),*heads.parameters()]]
            with TemporaryDirectory() as tmp:
                features=engine.fid_features(x)
                engine.evaluate(args,50,vae,heads,None,(x.cpu(),y.cpu(),features,None),dev,Path(tmp),
                                online_heads=heads,schedule=schedule)
                with (Path(tmp)/'eval_fid_combinations.csv').open() as f:combinations=list(csv.DictReader(f))
                assert len(combinations)==2**len(heads)
                assert all(int(r['num_eval'])==2 for r in combinations)
                with (Path(tmp)/'eval_block_cfg.csv').open() as f:cfg_rows=list(csv.DictReader(f))
                assert len(cfg_rows)==(len(heads)+1)*len(heads)
            assert torch.equal(before,torch.get_rng_state()) and vae.training and heads.training
            assert all((old is None and p.grad is None) or torch.equal(old,p.grad)
                       for old,p in zip(old_grads,[*vae.parameters(),*heads.parameters()]))
        finally: engine.fid_features=old_features
    finally: diag.image_bank=oldbank
    report=dict(cell=cell_id,device=device,loss=float(loss.detach()),latent_shapes=vae.latent_shapes,
        vae_parameters=sum(p.numel() for p in vae.parameters()),
        score_parameters_per_head=[sum(p.numel() for p in h.parameters()) for h in heads],
        forward_backward=True,sequential_rk4=True,hybrid_combinations=True,risk_diagnostics=True,
        gradient_alignment=True,conditioning=row['conditioning'])
    print('OK real FMNIST smoke:',json.dumps(report),flush=True)
    return report


def baseline_parity(base,cell_id):
    """Pinned direct 2/full-3 sources vs launchable versions: bitwise computations."""
    import torch
    from suite import manifest,config_for
    torch.set_num_threads(2)
    row=manifest(base)[cell_id];args=config_for(row,base);args.device='cpu'
    engine=engine_for(row)
    path=Path(base)/'pinned'/('train_hier_csem_fmnist_3block_original.py' if cell_id==2 else 'train_hier_csem_fmnist_direct_original.py')
    name=f'pinned_reference_{cell_id}'
    spec=importlib.util.spec_from_file_location(name,path);pin=importlib.util.module_from_spec(spec)
    sys.modules[name]=pin;spec.loader.exec_module(pin)
    torch.manual_seed(91);v1,h1,_=pin.build(args,'cpu');rng=torch.get_rng_state().clone()
    torch.manual_seed(91);v2,h2,_=engine.build(args,'cpu');assert torch.equal(rng,torch.get_rng_state())
    for a,b in [(v1,v2),(h1,h2)]:
        assert a.state_dict().keys()==b.state_dict().keys()
        assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    x=torch.rand(2,1,32,32)*2-1;y=torch.tensor([2,5])
    schedule=engine.ref.make_schedule(dict(time_schedule='log_t',t_min=args.t_min,t_max=args.t_max,num_train_timesteps=1000),'cpu')
    torch.manual_seed(92);s1,k1,d1=pin.score_and_losses(args,v1,h1,x,y,schedule);rng=torch.get_rng_state().clone()
    torch.manual_seed(92);s2,k2,d2=engine.score_and_losses(args,v2,h2,x,y,schedule)
    assert torch.equal(rng,torch.get_rng_state())
    for a,b in [(s1,s2),(k1,k2),(d1['band_loss'],d2['band_loss']),(d1['reconstruction'],d2['reconstruction'])]: assert torch.equal(a,b)
    params1=[*v1.parameters(),*h1.parameters()];params2=[*v2.parameters(),*h2.parameters()]
    l1=d1['reconstruction'].square().mean()+.6*s1+k1+.1*d1['band_loss']
    l2=d2['reconstruction'].square().mean()+.6*s2+k2+.1*d2['band_loss']
    g1=torch.autograd.grad(l1,params1,allow_unused=True);g2=torch.autograd.grad(l2,params2,allow_unused=True)
    assert all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)) for a,b in zip(g1,g2))
    args.rk4_steps=2;args.cfg=1.;v1.eval();v2.eval();h1.eval();h2.eval()
    bank=[torch.randn(2,*s) for s in engine.latent_shapes(args)]
    with torch.no_grad():
        z1=pin.sample_blocks(args,h1,2,'cpu',y=y,bank=[q.clone() for q in bank])
        z2=engine.sample_blocks(args,h2,2,'cpu',y=y,bank=[q.clone() for q in bank])
        assert all(torch.equal(a,b) for a,b in zip(z1,z2))
    report=dict(cell=cell_id,initialization=True,rng=True,losses=True,reconstruction=True,
                all_parameter_gradients=True,sequential_sampling=True,comparison='bitwise',pinned_source=path.name)
    print('OK baseline parity:',json.dumps(report),flush=True);return report


def oracle_contract():
    import torch
    from diagnostics import mixture_oracle
    torch.set_num_threads(2)
    mu=torch.tensor([-.8,.9],dtype=torch.float64).view(2,1,1,1)
    lv=torch.tensor([-.2,.3],dtype=torch.float64).view(2,1,1,1)
    point=torch.tensor([.4],dtype=torch.float64).view(1,1,1,1).requires_grad_(True)
    alpha=torch.tensor(.8,dtype=torch.float64);sigma=torch.tensor(.6,dtype=torch.float64)
    log_weights=torch.tensor([.2,-.4],dtype=torch.float64)
    oracle,*_=mixture_oracle(point,mu,lv,log_weights,alpha,sigma)
    var=alpha.square()*lv.exp()+sigma.square()
    log_mix=torch.logsumexp(log_weights-.5*((point-alpha*mu).square()/var+var.log()).flatten(),0)
    score=torch.autograd.grad(log_mix,point)[0]
    torch.testing.assert_close(oracle,-sigma*score,atol=1e-12,rtol=1e-12)
    print('OK: mixture oracle equals analytic score from differentiated mixture density')
    return True


def resume_contract(base,cell_id):
    """Interrupt after one committed epoch; resumed two-epoch state must match."""
    import torch
    from torch.utils.data import TensorDataset,DataLoader
    from suite import manifest,config_for
    import core
    torch.set_num_threads(2)
    row=manifest(base)[cell_id];engine=engine_for(row)
    generator=torch.Generator().manual_seed(910)
    data=TensorDataset(torch.rand(4,1,32,32,generator=generator)*2-1,torch.tensor([1,2,3,4]))
    old_loader=engine.ref.make_dataloaders
    old_save=engine.save_checkpoint
    engine.ref.make_dataloaders=lambda *a:(DataLoader(data,batch_size=2,shuffle=True),DataLoader(data,batch_size=2),10)
    def args_for(dest):
        a=config_for(row,base);a.device='cpu';a.output=str(dest);a.epochs=2;a.max_batches=1
        a.perc_weight=0.;a.gan_weight=0.;a.eval_samples=0;a.checkpoint_every=1
        return a
    try:
        with TemporaryDirectory() as tmp:
            tmp=Path(tmp);core.train(args_for(tmp/'continuous'),synthetic=True)
            def interrupt_after_commit(*a,**kw):
                old_save(*a,**kw)
                if a[1]==1: raise RuntimeError('synthetic interruption after commit')
            engine.save_checkpoint=interrupt_after_commit
            try: core.train(args_for(tmp/'interrupted'),synthetic=True)
            except RuntimeError as exc: assert 'synthetic interruption' in str(exc)
            else: raise AssertionError('Expected the simulated interruption')
            engine.save_checkpoint=old_save
            # Add an uncommitted log row to verify rollback of diagnostics.
            with (tmp/'interrupted'/'partial_probe.csv').open('w') as f:f.write('epoch,value\n1,3\n2,9\n')
            resumed=args_for(tmp/'interrupted');resumed.resume=str(tmp/'interrupted'/'latest.pt')
            core.train(resumed,synthetic=True)
            with (tmp/'interrupted'/'partial_probe.csv').open() as f:assert len(list(csv.DictReader(f)))==1
            a=torch.load(tmp/'continuous'/'latest.pt',map_location='cpu',weights_only=False)
            b=torch.load(tmp/'interrupted'/'latest.pt',map_location='cpu',weights_only=False)
            for key in ('vae','heads','ema'):
                assert all(torch.equal(v,b[key][k]) for k,v in a[key].items()),key
            assert a['scheduler']==b['scheduler']
            assert torch.equal(a['rng_torch'],b['rng_torch'])
            assert len(list(csv.DictReader((tmp/'interrupted'/'loss.csv').open())))==2
    finally:
        engine.ref.make_dataloaders=old_loader;engine.save_checkpoint=old_save
    report=dict(cell=cell_id,resume_matches_uninterrupted=True,atomic_checkpoint=True,log_rollback=True)
    print('OK resume contract:',json.dumps(report),flush=True);return report
