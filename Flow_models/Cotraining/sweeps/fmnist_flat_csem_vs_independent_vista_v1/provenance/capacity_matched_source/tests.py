"""Protocol, capacity, routing, reference-equivalence, resume and compiler checks."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import torch
from torch.nn import functional as F
import experiments
import models
import original_models
import original_experiments
from pinned import stacked_two_block as old


def small(row,base,device='cpu'):
    a=experiments.configuration(row,base,device=device)
    a.joint_width=48;a.joint_depth=2;a.batch=2;a.num_workers=0;a.cpu_threads=2
    a.epochs=2;a.max_batches=1;a.perc_weight=0.;a.gan_weight=0.
    a.eval_samples=4;a.eval_batch=2;a.feature_batch=2;a.rk4_steps=2;a.eval_every=1
    a.diagnostic_every=1;a.diagnostic_samples=2;a.diagnostic_particles=2;a.diagnostic_batch=2
    a.diagnostic_times=[.2];a.witness_every=0;a.sampling_check_every=0
    a.checkpoint_every=1;a.archive_epochs=[1,2]
    return a


def same(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:same(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for u,v in zip(a,b):same(u,v)
    else:assert a==b


def state_hash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def protocol_checks(base):
    rr=experiments.rows();assert len(rr)==20
    original=original_experiments.rows()[3]['config']
    current=next(r['config'] for r in rr if r['name']=='stack2_s43')
    for key,value in original.items():assert current[key]==value,(key,current[key],value)
    metadata={'seed','blocks','topology','z2_channels','z3_channels','stochastic_blocks','capacity_family','arm'}
    common={k:v for k,v in rr[0]['config'].items() if k not in metadata}
    for row in rr:
        assert {k:v for k,v in row['config'].items() if k not in metadata}==common
        assert row['config']['rep_multiplier']==.5
    assert {r['config']['seed'] for r in rr}==set(experiments.SEEDS)
    for blocks in (2,3):
        value=torch.arange(512.).reshape(2,4,8,8)
        assert torch.equal(models.pack(models.unpack(value,blocks)),value)
    experiments.export(base)
    return dict(jobs=20,seeds=list(experiments.SEEDS),two_block_reference_config='exact match',packing='exact permutation')


def model_checks(base):
    torch.set_num_threads(2)
    audits={};hashes={};routes={}
    for row in experiments.rows()[:4]:
        args=experiments.configuration(row,base,device='cpu')
        old.seed_everything(args.seed);vae,head=models.build(args,torch.device('cpu'))
        audits[args.arm]=models.parameter_audit(vae,head)
        assert not head.factored_head
        assert audits[args.arm]['score_head']==12159556
        assert audits[args.arm]['latent_coordinates']==256
        state=vae.state_dict()
        hashes[args.arm]=dict(head=state_hash(head.state_dict()),
            common_vae=state_hash({k:v for k,v in state.items() if not k.startswith('net.condition')}),
            full_vae=state_hash(state),rng=state_hash({'rng':torch.get_rng_state()}))
        x=torch.rand(2,1,32,32)*2-1
        mu,lv,features=vae.root(x)
        root=mu.detach().requires_grad_(True)
        m,l=vae.moment(1,features,mu,lv,[root])
        grad=torch.autograd.grad(m.sum()+l.sum(),root,allow_unused=True,retain_graph=True)[0]
        if args.stochastic_blocks==1:assert grad is None or grad.abs().max()==0
        else:assert grad is not None and grad.abs().max()>0
        if args.blocks==3:
            child=m.detach().requires_grad_(True)
            m3,l3=vae.moment(2,features,mu,lv,[root,child])
            grads=torch.autograd.grad(m3.sum()+l3.sum(),(root,child),allow_unused=True)
            if args.stochastic_blocks==1:assert all(g is None or g.abs().max()==0 for g in grads)
            else:assert all(g is not None and g.abs().max()>0 for g in grads)
        schedule=old.ref.make_schedule(dict(time_schedule='log_t',t_min=args.t_min,t_max=args.t_max,num_train_timesteps=1000),torch.device('cpu'))
        score,terminal,band,d=models.training_losses(args,vae,head,x,torch.tensor([0,1]),schedule)
        objective=.6*score+terminal+old.tmin_band_coefficient(args,schedule)*band+F.mse_loss(d['reconstruction'],x)
        objective.backward()
        unused=[n for n,p in vae.named_parameters() if p.grad is None]
        assert not unused,unused
        routes[args.arm]=dict(no_unused_vae_parameters=True,posterior_parent_routes='absent' if args.stochastic_blocks==1 else 'present')
        if args.stochastic_blocks==1:
            # Flat CSEM has no stacked descendant correction.
            valid=models.corrected_teacher(vae,d)
            for a,b in zip(valid,d['naive']):torch.testing.assert_close(a,b.detach(),rtol=2e-5,atol=2e-6)
        del vae,head
    assert len({v['head'] for v in hashes.values()})==1
    assert len({v['common_vae'] for v in hashes.values()})==1
    assert len({v['rng'] for v in hashes.values()})==1
    assert hashes['flat']['full_vae']==hashes['stack2']['full_vae']
    assert hashes['flat_three_encoder']['full_vae']==hashes['stack3']['full_vae']
    assert audits['flat']['vae']==6368847
    assert audits['stack3']['vae']==6368817
    assert audits['flat']['vae']-audits['stack3']['vae']==30
    # Saved at build time, never inferred from labels or padded parameter counts.
    (Path(base)/'parameter_audit.json').write_text(json.dumps(dict(counts=audits,initialization_hashes=hashes),indent=2)+'\n')
    return dict(counts=audits,routes=routes,shared_head_decoder_encoder='tensor-identical at seed42')


def reference_equivalence(base):
    torch.set_num_threads(2)
    row=next(r for r in experiments.rows() if r['name']=='stack2_s43')
    args=experiments.configuration(row,base,device='cpu')
    # Full production VAE and 18-layer score head, not reduced smoke models.
    old.seed_everything(args.seed);a,ha=models.build(args,torch.device('cpu'))
    old.seed_everything(args.seed);b,hb=original_models.build(args,torch.device('cpu'))
    same(a.state_dict(),b.state_dict());same(ha.state_dict(),hb.state_dict())
    # Activate head input derivatives rather than relying on zero-output init.
    with torch.no_grad():
        gen=torch.Generator().manual_seed(703)
        ha.final_proj.weight.copy_(torch.randn(ha.final_proj.weight.shape,generator=gen)*.002)
        hb.load_state_dict(ha.state_dict())
    x=torch.rand(2,1,32,32,generator=torch.Generator().manual_seed(704))*2-1
    labels=torch.tensor([2,7]);schedule=old.ref.make_schedule(dict(time_schedule='log_t',t_min=args.t_min,t_max=args.t_max,num_train_timesteps=1000),torch.device('cpu'))
    old.seed_everything(705);rng=torch.get_rng_state()
    outputs=[];opts=[]
    for vae,head,loss_fn in [(a,ha,models.training_losses),(b,hb,original_models.training_losses)]:
        torch.set_rng_state(rng)
        score,terminal,band,d=loss_fn(args,vae,head,x,labels,schedule)
        loss=F.mse_loss(d['reconstruction'],x)+.6*score+terminal+old.tmin_band_coefficient(args,schedule)*band
        loss.backward();outputs.append((score.detach(),terminal.detach(),band.detach(),d['reconstruction'].detach()))
        opt=torch.optim.AdamW([dict(params=vae.parameters(),lr=args.lr_vae),dict(params=head.parameters(),lr=args.lr_head/args.score_weight)],weight_decay=1e-4,betas=(.9,.95))
        torch.nn.utils.clip_grad_norm_(list(vae.parameters())+list(head.parameters()),1.)
        opts.append(opt)
    same(outputs[0],outputs[1])
    for left,right in [(a,b),(ha,hb)]:
        for p,q in zip(left.parameters(),right.parameters()):same(p.grad,q.grad)
    for opt in opts:opt.step()
    same(a.state_dict(),b.state_dict());same(ha.state_dict(),hb.state_dict())
    return 'bitwise identical production initialization, sampled losses/reconstruction, clipped gradients, and AdamW update (MSE+score+KL+band; LPIPS/GAN disabled for this local check)'


def compiler_checks(base):
    import compile_results
    with tempfile.TemporaryDirectory(prefix='fmnist-compile-') as temp:
        root=Path(temp);(root/'provenance').mkdir()
        (root/'provenance'/'baseline_4p3.json').write_text((Path(base)/'provenance'/'baseline_4p3.json').read_text())
        def put(name,epoch,fid,**kw):
            out=root/'results'/name;out.mkdir(parents=True,exist_ok=True)
            cell=next(r for r in experiments.rows() if r['name']==name)
            (out/'config.json').write_text(json.dumps(cell['config']))
            row=dict(epoch=epoch,cfg=3.,steps=20,init='gaussian',num_eval=10000,
                     fid_generated=fid,fid_posterior_recon=2.,backend='inception-legacy',**kw)
            with (out/'eval.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
        put('flat_s42',50,7.);put('stack2_s42',100,6.)
        compile_results.run(root)
        assert not compile_results.read_csv(root/'compiled'/'paired_latest.csv')
        put('stack2_s42',50,6.);compile_results.run(root)
        pairs=compile_results.read_csv(root/'compiled'/'paired_latest.csv')
        assert len(pairs)==1 and float(pairs[0]['a_minus_b'])==-1.
        put('stack2_s42',50,6.)
        wrong=root/'results'/'stack2_s42'/'config.json'
        wrong_cfg=json.loads(wrong.read_text());wrong_cfg['rep_multiplier']=1.;wrong.write_text(json.dumps(wrong_cfg))
        compile_results.run(root)
        assert not compile_results.read_csv(root/'compiled'/'paired_latest.csv')
        assert json.loads((root/'compiled'/'summary.json').read_text())['errors']
        for seed in experiments.SEEDS:
            put(f'flat_s{seed}',500,5.+.01*seed)
            put(f'stack2_s{seed}',500,4.+.01*seed)
        compile_results.run(root)
        result=json.loads((root/'compiled'/'summary.json').read_text())
        summaries=[r for r in result['endpoint_pairs'] if r['comparison']=='stack2-minus-flat' and r['cfg']==3 and r['metric']=='a_minus_b']
        assert {r['n'] for r in summaries}=={3,5}
        assert all(abs(r['mean']+1)<1e-12 for r in summaries)
        assert compile_results.run(root,strict=True)==1
    return 'live CSVs, matched epochs, delta sign, endpoint cohorts and strict completion'


def run(base,models=False):
    result=dict(protocol=protocol_checks(base),compiler=compiler_checks(base))
    if models:
        result['models']=model_checks(base)
        result['reference_equivalence']=reference_equivalence(base)
    (Path(base)/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('[VALIDATE]',json.dumps(result),flush=True)
    return 0


def smoke(base,row,device='cpu'):
    import trainer
    with tempfile.TemporaryDirectory(prefix='fmnist-smoke-') as temp:
        args=small(row,Path(temp),device)
        rc=trainer.train(args,Path(temp),synthetic=True)
        assert rc==0
        out=Path(args.output)
        assert json.loads((out/'complete.json').read_text())['epoch']==2
        assert (out/'diagnostics'/'epoch_0002.json').exists()
        assert len(__import__('compile_results').read_csv(out/'eval.csv'))==4
        result=dict(arm=args.arm,device=device,epochs=2,synthetic=True,
                    sampling='CFG1/3 generation and posterior recon',diagnostics=True)
        dest=Path(base)/'validation';dest.mkdir(exist_ok=True)
        (dest/f'smoke_{args.arm}_{device}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('[SMOKE]',json.dumps(result),flush=True)
    return 0


def resume_check(base):
    import trainer
    with tempfile.TemporaryDirectory(prefix='fmnist-resume-') as temp:
        root=Path(temp);row=experiments.rows()[1]
        a=small(row,root/'uninterrupted');b=small(row,root/'resumed')
        # Full diagnostic and generation path is exercised by smoke; this check
        # isolates checkpoint RNG/optimizer/scheduler restoration.
        for arg in (a,b):arg.diagnostic_every=0;arg.diagnostic_at_start=False;arg.eval_samples=0
        assert trainer.train(a,root,synthetic=True)==0
        assert trainer.train(b,root,synthetic=True,stop_after=1)==75
        assert trainer.train(b,root,resume=True,synthetic=True)==0
        left=torch.load(Path(a.output)/'latest.pt',map_location='cpu',weights_only=False)
        right=torch.load(Path(b.output)/'latest.pt',map_location='cpu',weights_only=False)
        for key in ('vae','head','ema','optimizer','scheduler','rng_torch','rng_python'):
            same(left[key],right[key])
    result=dict(bitwise_resume=True,epochs=2,optimizer_scheduler_rng=True)
    dest=Path(base)/'validation';dest.mkdir(exist_ok=True)
    (dest/'resume.json').write_text(json.dumps(result,indent=2)+'\n')
    print('[RESUME]',json.dumps(result),flush=True)
    return result
