"""Offline contracts and real, full-size PyTorch model smoke checks."""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def run(base):
    from suite import manifest, validate_manifest, cli_args
    from paper import select_cells, parse_slurm_job_id, validate_evaluation
    from tempfile import TemporaryDirectory
    base=Path(base); rows=manifest(base); validate_manifest(rows)
    for p in base.rglob('*.py'): ast.parse(p.read_text(),filename=str(p))
    required=['paper.py','job.slurm','core.py','suite.py','paper_io.py','compile_results.py',
              'train_hier_csem_fmnist.py','fmnist_reference.py','tests.py','manifest.csv',
              'README.md','requirements.txt','.gitignore','provenance.json']
    assert all((base/p).is_file() for p in required)
    provenance=json.loads((base/'provenance.json').read_text())
    for name,digest in provenance['bundle_source_sha256'].items():
        assert hashlib.sha256((base/name).read_bytes()).hexdigest()==digest, name
    assert [int(r['cell_id']) for r in select_cells('paired',rows)]==[0,1,2,3]
    assert [int(r['cell_id']) for r in select_cells('variants',rows)]==[4,5,6,7]
    assert [int(r['cell_id']) for r in select_cells('1,4-6',rows)]==[1,4,5,6]
    for message in ('1067000','Submitted batch job 1067000','site banner\n1067000;vista'):
        assert parse_slurm_job_id(message)=='1067000'
    tree=ast.parse((base/'train_hier_csem_fmnist.py').read_text())
    flags={node.args[0].value for node in ast.walk(tree) if isinstance(node,ast.Call)
           and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument'
           and node.args and isinstance(node.args[0],ast.Constant)}
    for row in rows:
        command=cli_args(row,base)
        assert all(command[i] in flags for i in range(0,len(command),2))
    with TemporaryDirectory() as tmp:
        dest=Path(tmp)
        evaluations=[dict(epoch=ep,fid_generated=6,fid_posterior_recon=4,
                          fid_encoded_root_generated_child=4.5,fid_backend='inception-legacy',
                          cfg=3,rk4_steps=20,num_eval=10000) for ep in range(50,501,50)]
        def write(name,rs):
            with (dest/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
        write('eval.csv',evaluations);write('loss.csv',[dict(epoch=500,total=.1)])
        assert int(validate_evaluation(rows[1],dest)['epoch'])==500
        write('eval.csv',evaluations[:-1])
        try: validate_evaluation(rows[1],dest)
        except ValueError: pass
        else: raise AssertionError('Incomplete endpoints cannot be completed')
    print('OK: eight arms, fixed recipes/budgets, source hashes, CLI/Slurm and endpoint contracts')
    return 0


def torch_smoke(base, cell_id, device='cpu'):
    import copy
    import math
    import torch
    from torch.nn import functional as F
    from suite import manifest, config_for
    import train_hier_csem_fmnist as e
    torch.set_num_threads(min(torch.get_num_threads(),2))
    dev=torch.device(device)
    row=manifest(base)[cell_id]; a=config_for(row,base); a.device=device
    torch.manual_seed(1024)
    vae,heads,_=e.build(a,dev)
    x=torch.rand(2,1,32,32,device=dev)*2-1; y=torch.tensor([2,5],device=dev)
    schedule=e.ref.make_schedule({'time_schedule':'log_t','t_min':a.t_min,'t_max':a.t_max,
                                 'num_train_timesteps':1000},dev)
    called=[]
    if a.mode=='hier':
        original=vae.ancestor_features
        def ancestor(z):
            called.append(z)
            return original(z)
        vae.ancestor_features=ancestor
    sl,kl,d=e.score_and_losses(a,vae,heads,x,y,schedule)
    assert vae.latent_shapes==e.latent_shapes(a)
    assert sum(d['block_dimensions'])==256
    assert [len(h.blocks) for h in heads]==list(map(int,row['score_depths'].split(':')))
    assert d['reconstruction'].shape==x.shape
    if a.mode=='hier':
        assert len(called)==2
        assert called[0] is d['latents'][0] and called[1] is d['noisy'][0]
        assert d['reconstruction_root'] is d['noisy'][0]
        # Child conditional parameters depend on clean root only for topdown arm.
        mu,lv,feat=vae.encode_root(x)
        probe=torch.randn_like(mu,requires_grad=True)
        mean,logvar=vae.encode_conditional(feat.detach(),probe)
        root_grad=torch.autograd.grad((mean+logvar).sum(),probe,allow_unused=True)[0]
        if a.child_posterior=='bottomup': assert root_grad is None
        else: assert root_grad is not None and root_grad.abs().sum()>0
    weights=[n/256 for n in d['block_dimensions']]
    torch.testing.assert_close(d['band_loss'],sum(w*b for w,b in zip(weights,d['band_per_block'])))
    idx=torch.searchsorted(schedule['times'],d['time'])
    at=schedule['alpha'][idx].view(2,1,1,1); st=schedule['sigma'][idx].view(2,1,1,1)
    for z,(mu,lv),target in zip(d['noisy'],d['params'],d['target_eps']):
        point=z.detach().clone().requires_grad_(True)
        variance=(at.square()*lv.detach().exp()+st.square()).clamp_min(1e-10)
        density=-.5*((point-at*mu.detach()).square()/variance+variance.log()).sum()
        score=torch.autograd.grad(density,point)[0]
        torch.testing.assert_close(target.detach(),-st*score)
    band_grads=torch.autograd.grad(d['band_loss'],list(vae.parameters())+list(heads.parameters()),
                                   allow_unused=True,retain_graph=True)
    nvae=sum(1 for _ in vae.parameters())
    assert any(g is not None and g.abs().sum()>0 for g in band_grads[:nvae])
    assert all(g is None for g in band_grads[nvae:])
    band_weight=e.tmin_band_coefficient(a,schedule)
    assert 0<band_weight<.6 and math.isfinite(band_weight)
    floor=e.tmin_band_floor(schedule)
    v=torch.tensor([floor+12,floor-12],device=dev,requires_grad=True)
    band=e.tmin_band_penalty(v,floor)
    grad=torch.autograd.grad(band,v)[0]
    torch.testing.assert_close(grad[1],torch.tensor(-.25,device=dev),atol=2e-6,rtol=0)
    loss=F.mse_loss(d['reconstruction'],x)+a.score_weight*sl+a.terminal_weight*kl+band_weight*d['band_loss']
    opt=torch.optim.AdamW([*vae.parameters(),*heads.parameters()],lr=1e-4)
    opt.zero_grad();loss.backward()
    for h in heads:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in h.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in [*vae.parameters(),*heads.parameters()])
    torch.nn.utils.clip_grad_norm_([*vae.parameters(),*heads.parameters()],1.0);opt.step()
    # Actual RK4 ancestral solves, with updated full-size model weights.
    a.rk4_steps=2; a.cfg=1.0
    vae.eval();heads.eval()
    with torch.no_grad():
        z=e.sample_blocks(a,heads,2,dev,y=y)
        fake=vae.decode(*z,t=torch.full((2,),a.decode_time,device=dev))
    assert fake.shape==x.shape and torch.isfinite(fake).all()
    if a.mode=='hier':
        root=torch.randn(2,*e.latent_shapes(a)[0],device=dev)
        bank=[None,torch.randn(2,*e.latent_shapes(a)[1],device=dev)]
        z=e.sample_blocks(a,heads,2,dev,y=y,root=root,bank=bank)
        assert z[0] is root
    report=dict(cell=cell_id,device=device,loss=float(loss.detach()),
                latent_shapes=vae.latent_shapes,vae_parameters=sum(p.numel() for p in vae.parameters()),
                score_parameters_per_head=[sum(p.numel() for p in h.parameters()) for h in heads],
                band_weight=band_weight,forward_backward=True,sequential_rk4=True)
    print('OK real FMNIST smoke:',json.dumps(report),flush=True)
    return report


def default_parity(base):
    """Original 5.341 trainer vs extended trainer: weights, RNG and loss math."""
    import gc
    import torch
    import train_hier_csem_fmnist as e
    torch.set_num_threads(2)
    spec=importlib.util.spec_from_file_location('pinned_fmnist',Path(base)/'pinned'/'train_hier_csem_fmnist_original.py')
    pin=importlib.util.module_from_spec(spec);sys.modules[spec.name]=pin;spec.loader.exec_module(pin)
    a=e.parse_args(['--mode','hier','--score-depths','8:8'])
    torch.manual_seed(90); v1,h1,_=pin.build(a,'cpu'); state1=torch.get_rng_state()
    torch.manual_seed(90); v2,h2,_=e.build(a,'cpu'); state2=torch.get_rng_state()
    assert torch.equal(state1,state2)
    for m,n in ((v1,v2),(h1,h2)):
        assert m.state_dict().keys()==n.state_dict().keys()
        assert all(torch.equal(p,n.state_dict()[k]) for k,p in m.state_dict().items())
    x=torch.rand(2,1,32,32)*2-1;y=torch.tensor([2,5])
    sched=e.ref.make_schedule({'time_schedule':'log_t','t_min':a.t_min,'t_max':a.t_max,'num_train_timesteps':1000},'cpu')
    torch.manual_seed(93);s1,k1,d1=pin.score_and_losses(a,v1,h1,x,y,sched)
    torch.manual_seed(93);s2,k2,d2=e.score_and_losses(a,v2,h2,x,y,sched)
    for p,q in ((s1,s2),(k1,k2),(d1['band_loss'],d2['band_loss']),(d1['reconstruction'],d2['reconstruction'])):
        torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert pin.tmin_band_coefficient(a,sched)==e.tmin_band_coefficient(a,sched)
    print('OK: default cascade has bit-identical initialization, RNG, score/KL/B and reconstruction')


def trainer_resume_test(base):
    """Real optimizer/scheduler training: interrupted+resumed equals uninterrupted."""
    import copy
    from tempfile import TemporaryDirectory
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import core
    import train_hier_csem_fmnist as e
    torch.set_num_threads(2)
    generator=torch.Generator().manual_seed(99)
    dataset=TensorDataset(torch.rand(2,1,32,32,generator=generator)*2-1,torch.tensor([2,5]))
    def loaders(batch,workers,key):
        return DataLoader(dataset,batch_size=2,shuffle=True),DataLoader(dataset,batch_size=2),10
    original_loader=e.ref.make_dataloaders;original_checkpoint=e.save_checkpoint
    e.ref.make_dataloaders=loaders
    with TemporaryDirectory() as tmp:
        def arguments(name):
            return e.parse_args(['--mode','hier','--epochs','2','--batch','2','--num-workers','0',
                '--max-batches','1','--eval-samples','0','--no-eval-checks','--checkpoint-every','1',
                '--eval-every','1','--perc-weight','0','--gan-weight','0','--device','cpu',
                '--output',str(Path(tmp)/name)])
        try:
            core.train(arguments('continuous'),synthetic=True)
            def interrupt(out,epoch,*args):
                original_checkpoint(out,epoch,*args)
                if epoch==1: raise InterruptedError('Synthetic interruption after committed epoch 1')
            e.save_checkpoint=interrupt
            try: core.train(arguments('resumed'),synthetic=True)
            except InterruptedError: pass
            else: raise AssertionError('Interruption did not execute')
            e.save_checkpoint=original_checkpoint
            # An uncommitted log row must be removed before continuation.
            with (Path(tmp)/'resumed'/'loss.csv').open('a') as f:
                fields=(Path(tmp)/'resumed'/'loss.csv').read_text().splitlines()[0].split(',')
                f.write(','.join('99' if col=='epoch' else '0' for col in fields)+'\n')
            a=arguments('resumed');a.resume=str(Path(tmp)/'resumed'/'latest.pt')
            core.train(a,synthetic=True)
            continuous=torch.load(Path(tmp)/'continuous'/'latest.pt',map_location='cpu',weights_only=False)
            resumed=torch.load(Path(tmp)/'resumed'/'latest.pt',map_location='cpu',weights_only=False)
            for module in ('vae','heads','ema'):
                assert all(torch.equal(v,resumed[module][k]) for k,v in continuous[module].items()), module
            assert continuous['epoch']==resumed['epoch']==2
            assert continuous['scheduler']==resumed['scheduler']
            assert torch.equal(continuous['rng_torch'],resumed['rng_torch'])
            with (Path(tmp)/'resumed'/'loss.csv').open(newline='') as f: rows=list(csv.DictReader(f))
            assert [int(r['epoch']) for r in rows]==[1,2]
            assert all(float(r['band_weight'])>0 for r in rows)
            old=json.loads((Path(tmp)/'resumed'/'resume_recipe.json').read_text())
            altered=dict(old,root_size=8)
            try: core.check_resume(old,altered)
            except ValueError: pass
            else: raise AssertionError('Resume failed to reject changed architecture')
            print('OK: real two-epoch training, atomic checkpoint, log rollback, and bit-identical resume')
        finally:
            e.ref.make_dataloaders=original_loader;e.save_checkpoint=original_checkpoint


def evaluation_io_test(base):
    """Exercise real generation/reconstruction/probes with synthetic features.

    This is an IO/autograd check, not an Inception/FashionMNIST FID measurement.
    """
    import copy
    import math
    import random
    from tempfile import TemporaryDirectory
    import numpy as np
    import torch
    from suite import manifest,config_for
    import train_hier_csem_fmnist as e
    torch.set_num_threads(2)
    original=e.fid_features
    def features(images,*args):
        return images.flatten(1)[:,:12].cpu()
    e.fid_features=features
    try:
        with TemporaryDirectory() as tmp:
            for cid in (0,6,7):
                a=config_for(manifest(base)[cid],base);a.device='cpu';a.rk4_steps=2
                a.eval_batch=2;a.eval_check_samples=4;a.eval_usage_samples=4;a.eval_grad_batch=2
                a.save_array_samples=4
                torch.manual_seed(100+cid)
                vae,heads,_=e.build(a,'cpu');ema=copy.deepcopy(heads).requires_grad_(False)
                real=torch.rand(4,1,32,32)*2-1;labs=torch.tensor([2,5,0,1])
                sched=e.ref.make_schedule({'time_schedule':'log_t','t_min':a.t_min,'t_max':a.t_max,'num_train_timesteps':1000},'cpu')
                out=Path(tmp)/str(cid);out.mkdir()
                fid_info=(real,labs,features(real),None)
                before=torch.get_rng_state().clone();before_np=np.random.get_state();before_py=random.getstate()
                row=e.evaluate(a,50,vae,ema,None,fid_info,torch.device('cpu'),out,
                               online_heads=heads,schedule=sched)
                assert torch.equal(before,torch.get_rng_state())
                assert np.array_equal(before_np[1],np.random.get_state()[1])
                assert before_py==random.getstate()
                assert row['num_eval']==4 and math.isfinite(row['fid_generated'])
                assert row['generation_decode_seconds']>0
                assert row['total_rk4_stages_per_image']==8*int(manifest(base)[cid]['blocks'])
                assert (out/'eval_gradient_checks.csv').is_file()
                with (out/'eval_sampling_checks.csv').open(newline='') as f: checks=list(csv.DictReader(f))
                assert len(checks)==(5 if a.mode=='hier' else 4)
                if a.mode=='hier':
                    assert (out/'eval_child_usage.csv').is_file()
                    assert math.isfinite(row['fid_encoded_root_generated_child'])
                    assert any(int(r['rk4_steps'])==1 for r in checks)
                assert all(p.grad is None for p in [*vae.parameters(),*heads.parameters()])
            print('OK: flat/same-scale/bottom-up evaluation, timing, child/sampling/gradient CSVs and preserved RNG')
    finally:
        e.fid_features=original
