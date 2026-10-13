"""CIFAR recipe contracts and real direct-child model/gradient/sampling checks."""
from __future__ import annotations
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def run(base):
    import paper
    import tempfile
    base=Path(base);rows=paper.manifest(base)
    for p in base.rglob('*.py'): ast.parse(p.read_text(),filename=str(p))
    required=['core.py','suite.py','paper_io.py','paper.py','job.slurm','manifest.csv',
              'direct_models.py','hier_models.py','hier_engine.py','hier_eval.py',
              'band_regularizer.py','compile_results.py','README.md','.gitignore','requirements.txt']
    assert all((base/p).exists() for p in required)
    info=json.loads((base/'provenance.json').read_text())
    for name,digest in info['source_sha256'].items():
        assert hashlib.sha256((base/name).read_bytes()).hexdigest()==digest,name
    # Core and suite stay exactly at the original optimized CIFAR recipe.
    for name in ('core.py','suite.py'):
        assert hashlib.sha256((base/name).read_bytes()).hexdigest()==info['original_cifar_sha256'][name]
    assert [int(r['cell_id']) for r in paper.select_cells('baseline',rows)]==[0,1]
    assert [int(r['cell_id']) for r in paper.select_cells('variants',rows)]==[4,5,6,7]
    for message in ('1070000','site banner\n1070000;vista','Submitted batch job 1070000'):
        assert paper.parse_slurm_job_id(message)=='1070000'
    with tempfile.TemporaryDirectory() as td:
        for r in rows:
            endpoint=dict(epoch=800,stage='cotrain',tag='LSI_Diff')
            for cfg in (2.5,3.0):
                for method,steps in (('rk4',20),('rk4',25),('heun',20)):
                    endpoint[f'fid_{method}_{steps}_randtok_cfg{str(cfg).replace(".","_")}_temp1_initgaussianT1p35']=9
                endpoint[f'fid_encoded_root_generated_children_cfg{str(cfg).replace(".","_")}']=7
            records=[dict(endpoint,epoch=ep) for ep in range(50,801,50)]
            paper.validate_evaluation(r,records)
            try:paper.validate_evaluation(r,records[:-1])
            except ValueError:pass
            else:raise AssertionError('Missing endpoint accepted')
    print('OK: eight-arm manifest, original CIFAR sources, source hashes, CLI/Slurm and 800-epoch endpoints')
    return 0


def recipe_checks(base):
    import paper
    from suite import resolve_config
    from hier_engine import adapt_trainer_source
    from band_regularizer import adapt_band_source
    import core,inspect
    old=list(csv.DictReader((Path(base)/'pinned/original_cifar_manifest.csv').open()))[5]
    before=resolve_config(old,Path(base));rows=paper.manifest(base)
    omit={'seed','results_dir','master_results_dir','ckpt_dir','paper_cell_id','paper_result_name'}
    for r in rows:
        current=paper.config_for(r,Path(base))
        assert {k:v for k,v in before.items() if k not in omit}=={k:current[k] for k in before if k not in omit}
        assert current['epochs_vae']==current['lr_schedule_epochs']==800
        assert current['score_w_vae']==current['resolved_tmin_band_weight']==.05
        assert current['kl_w']==.6 and current['score_head_loss_w']==1
        assert current['T_terminal']==1.05 and current['t_max']==1.35
        assert current['batch_size']==256 and current['base_ch']==64
        assert current['dit_hidden_dim']==384 and current['dit_num_heads']==6
        assert current['factored_head'] and current['aux_head_w']==.0025
        assert current['score_time_weighting']=='canonical'
        assert current['score_head_time_weighting']=='unweighted-eps'
        assert current['lr_vae']==.00025 and current['lr_score_head']==.0001
        assert current['latent_anchor_mode']=='ou_partial' and current['latent_anchor_time']==1.05
        assert current['perc_w']==.85 and current['gan_w']==.0025
        assert current['paper_cfg_grid']==[2.5,3.0]
        dims=[int(c) for c in r['block_coordinates'].split(':')]
        assert sum(dims)==512 and sum(map(int,r['score_depths'].split(':')))==12
    source=inspect.getsource(core.train_vae_cotrained_cond)
    adapted,counts=adapt_trainer_source(source)
    compile(adapted,'<direct-hierarchy>','exec')
    for call in [n for n in ast.walk(ast.parse(adapted)) if isinstance(n,ast.Call)
                 and isinstance(n.func,ast.Name) and n.func.id in ('unet_lsi','unet_control')]:
        context=next(k.value for k in call.keywords if k.arg=='context')
        if ast.unparse(call.args[0]) in ('z_t_for_score','z_mu_t_for_score'):
            assert 'canonical_stability_scale_gradient(z0, encoder_csem_gradient_multiplier)' in ast.unparse(context)
        else:assert ast.unparse(context)=='z0.detach()'
    for src in (source,adapted):
        result,band_counts=adapt_band_source(src);compile(result,'<band>','exec')
        assert band_counts==dict(setup=1,penalty=1,loss=2,epoch=1,resume=1)
        assert 'band_objective = active_score_w_vae * band_penalty' in result
        assert 'replace_parameter_gradients' in result and 'loss_joint.backward()' in result
    print('OK: every nonarchitectural setting matches original CIFAR; canonical B and independent head routing retained')


def build_models(cfg,device='cpu',small=False):
    import core
    from direct_models import DirectHierarchicalVAE,DirectHierarchicalHeads
    opts={k:cfg[k] for k in ('latent_channels','base_ch','num_res_blocks','decoder_attn_half',
           'latent_proj_depth','encoder_attn_half','decoder_extra_block','conv3x3_proj',
           'use_tanh_out','clamp_logvar','attn_zero_init','time_cond_decoder','dec_time_emb_dim')}
    opts.update(img_size=32,img_channels=3,use_norm=False,latent_anchor_mode='ou_partial',latent_anchor_time=1.05)
    if small:opts['base_ch']=16
    hops=dict(in_channels=8,hidden_dim=48 if small else 384,depth=2 if small else 12,num_heads=6,
              patch_size=1,latent_size=8,num_classes=10,mlp_ratio=4,dropout=0,factored_head=True)
    if cfg.get('hierarchy_variant')=='direct_child_v1':
        v=DirectHierarchicalVAE(block_channels=cfg['block_channels'],root_size=cfg['root_size'],**opts)
        h=DirectHierarchicalHeads(block_channels=cfg['block_channels'],root_size=cfg['root_size'],
                                  score_depths=[1,1] if small else cfg['score_depths'],**hops)
    else:v=core.VAE(**opts);h=core.DiTModel(**hops)
    return v.to(device),h.to(device)


def architecture_note_parity(base):
    """Copy shared modules into the note's actual VAE and compare direct decode.

    CIFAR's mean anchor is kept at the encoder boundary and is deliberately not
    part of this decoder parity check. Channel/image sizes are scaled to CIFAR.
    """
    import math,types
    import torch
    from torch import nn
    from torch.nn import functional as F
    import core,paper
    torch.set_num_threads(2)
    tree=ast.parse((Path(base)/'pinned/train_hier_csem_fmnist_direct.py').read_text())
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('CascadeResBlock','HierarchicalFMNISTVAE')]
    ns=dict(nn=nn,torch=torch,F=F,math=math,ref=core)
    exec(compile(ast.Module(body=classes,type_ignores=[]),'<actual_direct_FMNIST_model>','exec'),ns)
    cfg=paper.config_for(paper.manifest(base)[0],Path(base))
    vae,heads=build_models(cfg,'cpu',small=True)
    opts={k:cfg[k] for k in ('num_res_blocks','decoder_attn_half','latent_proj_depth','encoder_attn_half',
          'decoder_extra_block','conv3x3_proj','use_tanh_out','clamp_logvar','attn_zero_init','time_cond_decoder','dec_time_emb_dim')}
    opts.update(base_ch=16,use_norm=False,img_size=32,img_channels=3)
    note=ns['HierarchicalFMNISTVAE'](8,6,**opts)
    mapping={'enc_conv_in':'enc_conv_in','enc_blocks':'enc_blocks','enc_pre_proj':'enc_pre_proj',
             'enc_coarse':'root_down','root_posterior':'root_moments','root_in':'root_in',
             'root_blocks':'root_blocks','root_up':'root_up','condition':'child_moments',
             'child_in':'child_in','child_fusion':'child_fusion','dec_blocks':'dec_blocks','dec_out':'dec_out',
             'dec_time_mlp':'dec_time_mlp','dec_film_layers':'dec_film_layers'}
    for dest,source in mapping.items():getattr(note,dest).load_state_dict(getattr(vae,source).state_dict())
    from direct_models import pack_root
    root=torch.randn(2,8,4,4);child=torch.randn(2,6,8,8);t=torch.tensor([.01,.5])
    image=torch.randn(2,3,32,32)
    note.eval();vae.eval()
    with torch.no_grad():
        a=note.decode(root,child,t=t)
        b=vae.decode(torch.cat((pack_root(root),child),1),t=t)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        _,_,features=note.encode_root(image)
        m,l=note.encode_conditional(features,root)
        direct=vae.child_moments(torch.cat((features,vae.ancestor_features(root)),1)).chunk(2,1)
        torch.testing.assert_close(m,direct[0],atol=0,rtol=0)
        torch.testing.assert_close(l,direct[1].clamp(-30,20),atol=0,rtol=0)
    print('OK: direct decoder and raw conditional posterior match the note\'s actual architecture bit for bit')


def trainer_io_test(base):
    """Run the ACTUAL guarded CIFAR trainer on two synthetic minibatches."""
    import copy,tempfile,types
    import torch
    from torch.utils.data import TensorDataset,DataLoader
    import core,paper,hier_eval
    from hier_engine import train_hierarchy
    torch.set_num_threads(2)
    originals={k:getattr(core,k) for k in ('make_dataloaders','get_fid_model','LPIPS_AVAILABLE')}
    original_eval=hier_eval.evaluate_hierarchy
    gen=torch.Generator().manual_seed(9)
    dataset=TensorDataset(torch.randn(2,3,32,32,generator=gen).clamp(-1,1),torch.tensor([1,7]))
    def loaders(batch,workers,key,data_root=None):
        return DataLoader(dataset,batch_size=2,shuffle=True),DataLoader(dataset,batch_size=2),10
    def evaluate(*args,**kwargs):
        result={}
        for cfg in (2.5,3.0):
            for method,steps in (('rk4',20),('rk4',25),('heun',20)):
                result[f'fid_{method}_{steps}_randtok_cfg{str(cfg).replace(".","_")}_temp1_initgaussianT1p35']=9
            result[f'fid_encoded_root_generated_children_cfg{str(cfg).replace(".","_")}']=7
        return result
    core.make_dataloaders=loaders;core.get_fid_model=lambda *a,**k:(None,False);core.LPIPS_AVAILABLE=False
    hier_eval.evaluate_hierarchy=evaluate
    try:
        with tempfile.TemporaryDirectory() as td:
            cfg=paper.config_for(paper.manifest(base)[0],Path(td))
            cfg.update(epochs_vae=2,lr_schedule_epochs=2,base_ch=16,dit_hidden_dim=48,
                       score_depths=[1,1],batch_size=2,num_workers=0,eval_max_samples=2,
                       eval_freq_cotrain=1,paper_checkpoint_every=1,gan_w=0,perc_w=0,
                       num_train_timesteps=32,sw2_n_projections=4,generate_visualizations=False)
            core.seed_everything(42);loss,ev=train_hierarchy(cfg)
            assert len(loss)==2 and len(ev)==2
            assert (loss.band_weight==.05).all() and (loss.band_objective>=0).all()
            path=Path(cfg['ckpt_dir'])/'training_state_latest.pt'
            state=torch.load(path,map_location='cpu',weights_only=False)
            assert state['epoch']==2 and not state['pending_eval']
            assert state['cfg']['child_coordinates'].startswith('direct Z2')
            assert 'vae' in state and 'joint_optimizer' in state and 'joint_scheduler' in state
            # Resume the same completed recipe and verify state/history is preserved.
            resume=dict(cfg,paper_resume_state=str(path))
            core.seed_everything(99);again,eval_again=train_hierarchy(resume)
            after=torch.load(path,map_location='cpu',weights_only=False)
            assert len(again)==len(loss) and len(eval_again)==len(ev)
            assert all(torch.equal(v,after['vae'][k]) for k,v in state['vae'].items())
            assert all(torch.equal(v,after['score'][k]) for k,v in state['score'].items())
            assert state['joint_scheduler']==after['joint_scheduler']
            print('OK: real two-step guarded trainer, coefficient/routing logs, full checkpoint and completed-state resume')
    finally:
        for k,v in originals.items():setattr(core,k,v)
        hier_eval.evaluate_hierarchy=original_eval


def evaluator_io_test(base):
    """Real ancestral evaluation with tiny synthetic features, not real FID."""
    import tempfile
    import torch
    from torch.utils.data import TensorDataset,DataLoader
    import core,paper,hier_eval
    torch.set_num_threads(2)
    old=core.extract_inception_features;old_plan=hier_eval.sampler_configs
    def features(images,device,batch_size=2,inception_model=None):
        return images.mean(dim=(2,3)).cpu(),None
    core.extract_inception_features=features
    hier_eval.sampler_configs=lambda cfg:[dict(method='rk4_ode',steps=20,cfg_level=v,init_mode='gaussian-T') for v in (2.5,3.)]
    try:
        with tempfile.TemporaryDirectory() as td:
            cfg=paper.config_for(paper.manifest(base)[0],Path(td))
            cfg.update(batch_size=2,eval_max_samples=4,eval_lsi_gap_samples=4,
                       eval_lsi_gap_time_points=2,kid_num_subsets=2,kid_subset_size=4,
                       sw2_n_projections=4,num_classes=10)
            v,h=build_models(cfg,'cpu',small=True)
            dataset=TensorDataset(torch.randn(4,3,32,32).clamp(-1,1),torch.tensor([1,7,0,2]))
            loader=DataLoader(dataset,batch_size=2)
            before=torch.get_rng_state().clone()
            metrics=hier_eval.evaluate_hierarchy(800,'LSI_Diff',v,h,loader,cfg,torch.device('cpu'),None,
                                                results_dir=td)
            assert torch.equal(before,torch.get_rng_state())
            assert metrics['latent_coordinates']==512 and metrics['root_coordinates']==128
            assert metrics['child_coordinates_definition'].startswith('direct Z2')
            assert metrics['child_use_zero_delta']>0 and metrics['child_generated_draws_mse']>0
            for level in ('2_5','3_0'):
                assert metrics[f'fid_encoded_root_generated_children_cfg{level}']>=0
                assert metrics[f'sampling_seconds_rk4_20_randtok_cfg{level}_temp1_initgaussianT1p35']>0
            assert all(p.grad is None for p in [*v.parameters(),*h.parameters()])
            print('OK: real direct-cascade evaluator, fixed-root generation, interventions, timing and preserved RNG')
    finally:
        core.extract_inception_features=old;hier_eval.sampler_configs=old_plan


def torch_smoke(base,blocks=None,device='cpu',cell_id=None):
    import math
    import torch
    import core,paper
    from direct_models import pack_root,unpack_root
    from hier_models import sample_hierarchy
    from band_regularizer import tmin_band_penalty,tmin_band_logvar_floor
    torch.set_num_threads(min(torch.get_num_threads(),2));dev=torch.device(device)
    reports=[]
    for row in paper.manifest(base):
        cid=int(row['cell_id'])
        if cell_id is not None and cid!=cell_id:continue
        if blocks is not None and int(row['blocks'])!=blocks:continue
        cfg=paper.config_for(row,Path(base));core.seed_everything(int(row['seed']))
        vae,heads=build_models(cfg,dev);hier=int(row['blocks'])==2
        x=torch.randn(2,3,32,32,device=dev).clamp(-1,1);y=torch.tensor([1,7],device=dev)
        eps=torch.randn(2,8,8,8,device=dev)
        calls=[]
        if hier:
            assert not hasattr(vae,'child_baseline') and not hasattr(vae,'baseline')
            assert not hasattr(vae,'child_location_scale')
            ancestor=vae.ancestor_features
            def track(root):calls.append(root);return ancestor(root)
            vae.ancestor_features=track
            mu,lv,z=vae.encode_sample(x,eps=eps)
            torch.testing.assert_close(z,mu+(.5*lv).exp()*eps)
            r=cfg['block_channels'][0]
            assert calls[0].shape[1:]==vae.latent_shapes[0]
            assert [len(h.model.blocks) if hasattr(h,'model') else len(h.blocks) for h in heads.heads]==cfg['score_depths']
            root=unpack_root(z[:,:r],cfg['root_size'])
            torch.testing.assert_close(pack_root(root,cfg['root_size']),z[:,:r])
            torch.testing.assert_close(root.square().sum(),z[:,:r].square().sum())
        else:mu,lv=vae.encode(x);z=mu+(.5*lv).exp()*eps;r=8
        t=torch.tensor([.005,.7],device=dev)
        a,s=core.get_ou_params(t[:,None,None,None]);noisy=a*z+s*torch.randn_like(z)
        variance=a.square()*lv.exp()+s.square()+1e-8
        target=s*(noisy-a*mu)/variance
        point=noisy.detach().clone().requires_grad_(True)
        ld=-.5*((point-a*mu.detach()).square()/variance.detach()+variance.detach().log()).sum()
        gradient=torch.autograd.grad(ld,point)[0]
        torch.testing.assert_close(target.detach(),-s*gradient)
        ctx={'context':z} if hier else {}
        pred,lam,nu=heads(noisy,t,y,return_components=True,detach_components=True,**ctx)
        rec=vae.decode(noisy,t=t)
        if hier:
            assert len(calls)==2
            torch.testing.assert_close(calls[1],unpack_root(noisy[:,:r],cfg['root_size']))
            # The child injected by the decoder is the untransformed modeled z2.
            seen=[];handle=vae.child_in.register_forward_pre_hook(lambda _,args:seen.append(args[0]))
            vae.decode(noisy,t=t);handle.remove()
            torch.testing.assert_close(seen[0],noisy[:,r:])
        rep_cfg=dict(cfg,t_max=cfg['T_terminal']);schedule=core.make_schedule(rep_cfg,dev)
        weights=core.score_time_importance_weights(t=t,sigma=s,cfg=rep_cfg,schedule=None,t_idx=None,weighting_mode='canonical')
        mse,_=core.weighted_score_prediction_losses(pred,target,weights)
        floor=tmin_band_logvar_floor(schedule);band=.05*tmin_band_penalty(lv,floor)
        parameters=tuple(vae.parameters())+tuple(heads.parameters())
        gradients=torch.autograd.grad(band,parameters,allow_unused=True,retain_graph=True)
        nvae=len(tuple(vae.parameters()))
        assert any(g is not None and g.abs().sum()>0 for g in gradients[:nvae])
        assert all(g is None for g in gradients[nvae:])
        split=lv.split(cfg['block_channels'] if hier else [8],dim=1)
        torch.testing.assert_close(band,sum(q.shape[1]/8*.05*tmin_band_penalty(q,floor) for q in split))
        outer=(rec-x).square().mean()+.05*mse+.6*core.component_kl_at_ou_time(mu,lv,1.05)+band
        # Independently regressed score path uses detached moments AND context.
        tin=torch.tensor([.1,1.2],device=dev);ai,si=core.get_ou_params(tin[:,None,None,None])
        ni=ai*z.detach()+si*torch.randn_like(z);vi=ai.square()*lv.detach().exp()+si.square()+1e-8
        ci=si*(ni-ai*mu.detach())/vi
        opts={'context':z.detach()} if hier else {}
        pi,li,_=heads(ni,tin,y,return_components=True,detach_components=True,**opts)
        inner=(pi-ci).square().mean()+.0025*(li-si/vi).square().mean()
        inner_to_vae=torch.autograd.grad(inner,tuple(vae.parameters()),allow_unused=True,retain_graph=True)
        assert all(g is None for g in inner_to_vae)
        head_parameters=tuple(heads.parameters())
        head_grad=torch.autograd.grad(inner,head_parameters,allow_unused=True,retain_graph=False)
        outer.backward();core.replace_parameter_gradients(head_parameters,head_grad)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in vae.parameters())
        optimizer=torch.optim.AdamW([{'params':vae.parameters(),'lr':cfg['lr_vae']},
                                    {'params':heads.parameters(),'lr':cfg['lr_score_head']}],betas=(.9,.95),weight_decay=1e-4)
        torch.nn.utils.clip_grad_norm_(parameters,1);optimizer.step()
        vae.eval();heads.eval()
        if hier:
            probe=torch.randn_like(z,requires_grad=True)
            mm,ll,_=vae.encode_sample(x,eps=probe)
            ancestor_gradient=torch.autograd.grad(mm[:,r:].square().mean(),probe)[0]
            assert ancestor_gradient[:,:r].abs().sum()>0
            context=z[:,:r].detach().clone().requires_grad_(True)
            child_pred=heads.heads[1](noisy[:,r:].detach(),t,y,context=context)
            context_grad=torch.autograd.grad(child_pred.square().mean(),context)[0]
            assert context_grad.abs().sum()>0
        with torch.no_grad():
            if hier:
                zz=sample_hierarchy(heads,noise=torch.randn_like(z),y=y,cfg=cfg,steps=2,cfg_scale=2.5)
                held=sample_hierarchy(heads,noise=torch.randn_like(z),y=y,cfg=cfg,steps=2,cfg_scale=2.5,root=z[:,:r])
                torch.testing.assert_close(held[:,:r],z[:,:r])
            else:zz=core.UniversalSampler('rk4_ode',2,t_min=cfg['t_min'],t_max=cfg['t_max'],schedule_type='log_t').sample(heads,x_init=torch.randn_like(z),y=y,cfg_scale=2.5)
            fake=vae.decode(zz,t=t)
            assert fake.shape==x.shape and torch.isfinite(fake).all()
        report=dict(cell=cid,device=device,vae_parameters=sum(p.numel() for p in vae.parameters()),
                    score_parameters=sum(p.numel() for p in heads.parameters()),
                    latent_coordinates=512,block_coordinates=list(map(int,row['block_coordinates'].split(':'))),
                    direct_child=hier,band_weight=.05,canonical_representation=True,
                    detached_inner_gradient_routing=True,forward_backward=True,sequential_sampling=True)
        reports.append(report);print('OK real CIFAR smoke:',json.dumps(report),flush=True)
        del vae,heads,optimizer
        if dev.type=='cuda':torch.cuda.empty_cache()
    return 0
