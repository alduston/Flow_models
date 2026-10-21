"""Recipe parity, real graphs, analytic Gaussian teacher and full-state resume checks."""
import ast,csv,hashlib,json,inspect,tempfile
from pathlib import Path


def validate(base):
    from design import rows,config_for,export
    info=json.loads((base/'pinned/direct_cifar_provenance.json').read_text())
    for name in ('core.py','suite.py','paper_io.py','direct_models.py','hier_models.py','band_regularizer.py'):
        expected=info['source_sha256'][name]
        assert hashlib.sha256((base/name).read_bytes()).hexdigest()==expected,name
    for p in base.glob('*.py'):ast.parse(p.read_text(),filename=p.name)
    rr=rows();assert len(rr)==24
    assert sum(r['stage']=='joint' for r in rr)==12
    assert sum(r['stage']=='frozen' for r in rr)==12
    from suite import resolve_config
    original=list(csv.DictReader((base/'pinned/direct_cifar_manifest.csv').open()))[2]
    before=resolve_config(original,base)
    exclusions={'seed','results_dir','master_results_dir','ckpt_dir','paper_cell_id','paper_result_name',
                'paper_cfg_grid','skip_lsi_gap','eval_tk_vs_t_comparison'}
    for r in rr[:12]:
        cfg=config_for(r,base)
        for k,v in before.items():
            if k not in exclusions:assert cfg[k]==v,(r['cell_id'],k,cfg[k],v)
        assert cfg['dit_hidden_dim']==384 and cfg['dit_depth']==12 and cfg['batch_size']==256
        assert cfg['lr_score_head']==1e-4 and cfg['score_head_loss_w']==1
        assert cfg['T_terminal']==1.05 and cfg['t_max']==1.35
        assert cfg['fixed_band_weight']==.05 and cfg['score_w_vae']==.05
    import core
    from engine import adapt
    from band_regularizer import adapt_band_source
    src,_=adapt(inspect.getsource(core.train_vae_cotrained_cond));src,_=adapt_band_source(src);compile(src,'<adapted>','exec')
    from evaluation import make_evaluator
    make_evaluator()
    import launcher_base
    for message in ('1070000','site banner\n1070000;vista','Submitted batch job 1070000'):
        assert launcher_base.parse_slurm_job_id(message)=='1070000'
    export(base);print('PASS: immutable CIFAR recipe; 24 cells; guarded trainer/evaluator; Slurm parser')
    return 0


def small_cfg(row,base):
    from design import config_for
    cfg=config_for(row,base);cfg.update(base_ch=16,dit_hidden_dim=24,dit_depth=1,dit_num_heads=3,
        num_res_blocks=1,latent_proj_depth=1,encoder_attn_half=False,decoder_attn_half=False,
        decoder_extra_block=False,batch_size=2,num_workers=0,num_classes=10,img_size=32,img_channels=3,
        num_train_timesteps=32,epochs_vae=2,epochs_refine=0,lr_schedule_epochs=2,
        eval_freq_cotrain=1,eval_max_samples=4,kid_num_subsets=1,kid_subset_size=4,sw2_n_projections=4,
        audit_samples=4,witness_train=8,witness_test=8,audit_batch=2,audit_times=[.2],witness_times=[.2],
        gan_w=0.,perc_w=0.,save_eval_sample_panels=False)
    return cfg


def smoke(base,cell_id=6,device='cpu'):
    import copy,torch
    import core
    from design import rows
    from stacked import build_models,noisy,corrected,StackedVAE
    from suite import fixed_metric_rng
    from rng_state import capture_rng_state,restore_rng_state
    torch.set_num_threads(2);dev=torch.device(device)
    cfg=(__import__('design').config_for(rows()[cell_id],base) if device=='cuda' else small_cfg(rows()[cell_id],base));cfg.update(num_classes=10,img_size=32,img_channels=3);core.seed_everything(17)
    vae,head=build_models(cfg,dev)
    if cell_id==6:
        core.seed_everything(17);flat_cfg=dict(cfg,blocks=1);fv,fh=build_models(flat_cfg,dev)
        assert all(torch.equal(a,b) for a,b in zip(head.state_dict().values(),fh.state_dict().values()))
        del fv,fh
    x=torch.randn(2,3,32,32,device=dev);lab=torch.tensor([1,2],device=dev);t=torch.full((2,),.2,device=dev)
    d=noisy(vae,x,t);prediction=head(d['y'],t,lab);v=corrected(vae,d)
    assert prediction.shape==v.shape==(2,8,8,8) and torch.isfinite(v).all()
    rec=vae.decode(d['y'],t);assert rec.shape==x.shape
    # r scales every VAE route, leaving head gradients identical on common draws.
    from torch.nn import functional as F
    def grads(r):
        mm=copy.deepcopy(vae);hh=copy.deepcopy(head)
        with fixed_metric_rng(torch,101):dd=noisy(mm,x,t)
        scale=lambda z:z.detach()+r*(z-z.detach())
        loss=F.mse_loss(hh(scale(dd['y']),t,lab),scale(dd['naive']))
        return torch.autograd.grad(loss,[*mm.parameters(),*hh.parameters()],allow_unused=True),len(list(mm.parameters()))
    g1,n=grads(1);g0,_=grads(0);g4,_=grads(4)
    for a,b,c in zip(g1[:n],g0[:n],g4[:n]):
        if a is not None:assert torch.allclose(b,torch.zeros_like(b),atol=1e-8) and torch.allclose(c,4*a,atol=1e-5,rtol=1e-4)
    for a,b,c in zip(g1[n:],g0[n:],g4[n:]):
        if a is not None:assert torch.equal(a,b) and torch.equal(a,c)
    if isinstance(vae,StackedVAE):
        from direct_models import pack_root,unpack_root
        z=torch.randn(2,8,4,4,device=dev);assert torch.equal(unpack_root(pack_root(z)),z)
        cfg2=dict(cfg,topology='decoupled');dec,dh=build_models(cfg2,dev)
        dd=noisy(dec,x,t);assert torch.allclose(corrected(dec,dd),dd['naive'],atol=2e-5,rtol=2e-5)
        # Corrected supervision cannot update the frozen VAE.
        for p in vae.parameters():p.requires_grad_(False)
        dd=noisy(vae,x,t);ct=corrected(vae,dd)
        assert not ct.requires_grad
    from evaluation import make_evaluator
    make_evaluator()
    from audit import run_audit
    from torch.utils.data import TensorDataset
    ds=TensorDataset(torch.randn(8,3,32,32),torch.arange(8)%10)
    with tempfile.TemporaryDirectory() as tmp:
        cc=dict(cfg,results_dir=tmp);result=run_audit(0,vae,head,ds,ds,cc,dev)
        assert result and list(Path(tmp).rglob('*paired.npz'))
    s=capture_rng_state();a=torch.randn(4);restore_rng_state(s);assert torch.equal(a,torch.randn(4))
    print('PASS: real posterior/decoder/DiT; r=0/1/4 gradient routing; decoupled zero defect; audit/witness raw exports; RNG')
    return 0


def gaussian_teacher_check():
    import torch
    from stacked import StackedVAE,corrected
    from direct_models import pack_root,unpack_root
    torch.set_num_threads(2);torch.manual_seed(29)
    class Linear(StackedVAE):
        def __init__(self):torch.nn.Module.__init__(self)
        def child(self,h,r,m):return .7*pack_root(r).repeat(1,3,1,1),torch.full((len(r),6,8,8),.4,dtype=r.dtype).log()
    # Four coordinates: one root and three conditionally independent child coordinates.
    v=.7;c=.7;w=.4;tv=.2;a=torch.exp(torch.tensor(-tv,dtype=torch.float64));s=torch.sqrt(1-a*a)
    L=torch.eye(4,dtype=torch.float64);L[1:,0]=c
    C=L@torch.diag(torch.tensor([v,w,w,w],dtype=torch.float64))@L.T
    V=a*a*C+s*s*torch.eye(4,dtype=torch.float64);y=torch.tensor([.2,-.7,.3,.8],dtype=torch.float64)
    M=a*C@torch.linalg.inv(V);P=C-a*a*C@torch.linalg.solve(V,C)
    N=4096;z=(M@y)+torch.randn(N,4,dtype=torch.float64)@torch.linalg.cholesky(P).T
    root=z[:,0,None,None,None].expand(N,8,4,4)
    clean=torch.cat((pack_root(root),z[:,1:,None,None,None].expand(N,3,2,8,8).reshape(N,6,8,8)),1)
    yy=torch.cat((y[:1].expand(N,2,8,8),y[1:,None,None,None,None].expand(3,N,2,8,8).permute(1,0,2,3,4).reshape(N,6,8,8)),1)
    d=dict(y=yy,z=clean,alpha=a.expand(N,1,1,1),sigma=s.expand(N,1,1,1),
           root_mu=torch.zeros_like(root),root_lv=torch.full_like(root,v).log(),h=torch.zeros(N,1,dtype=root.dtype))
    answer=corrected(Linear(),d);got=torch.stack((answer[:,:2].mean(),answer[:,2:4].mean(),answer[:,4:6].mean(),answer[:,6:].mean()))
    exact=s*torch.linalg.solve(V,y)
    assert torch.allclose(got,exact,atol=.035,rtol=.035),(got,exact)
    print('PASS: analytic correlated-Gaussian conditional-mean identity',got.tolist(),exact.tolist())


def trainer_check(base):
    """Execute pinned training updates and compare uninterrupted/resumed model states."""
    import torch,copy
    from torch.utils.data import TensorDataset,DataLoader
    import core,evaluation,audit,engine,band_regularizer
    from design import rows
    torch.set_num_threads(2)
    ds=TensorDataset(torch.randn(4,3,32,32,generator=torch.Generator().manual_seed(1)),torch.arange(4))
    old=(core.make_dataloaders,core.get_fid_model,core.LPIPS_AVAILABLE,core.evaluate_current_state,evaluation.evaluate,audit.run_audit)
    def loaders(*a,**k):return DataLoader(ds,batch_size=2,shuffle=True),DataLoader(ds,batch_size=2),10
    stopping=[False]
    def ev(*a,**kw):
        if stopping[0] and a[0]==1:raise InterruptedError('synthetic stop after committed epoch')
        return {'fid_fake_test_feature_only':1.}
    core.make_dataloaders=loaders;core.get_fid_model=lambda *a,**k:(None,False);core.LPIPS_AVAILABLE=False
    core.evaluate_current_state=evaluation.evaluate=ev;audit.run_audit=lambda *a,**k:None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            def run(label,epochs=2,resume=None,flat=False,original=False,stop=False):
                cfg=small_cfg(rows()[0 if flat else 6],base);out=Path(tmp)/label
                cfg.update(results_dir=str(out),master_results_dir=str(out),ckpt_dir=str(out/'checkpoints'),epochs_vae=epochs)
                if resume:cfg['paper_resume_state']=str(resume)
                core.seed_everything(42);stopping[0]=stop
                try:(band_regularizer.train_flat_band if original else engine.train)(cfg)
                except InterruptedError:
                    if not stop:raise
                stopping[0]=False
                return torch.load(out/'checkpoints/training_state_latest.pt',map_location='cpu',weights_only=False)
            full=run('full');one=run('partial',stop=True)
            resume=Path(tmp)/'partial/checkpoints/training_state_latest.pt'
            resumed=run('partial',2,resume)
            for group in ('vae','score','score_ema'):
                for key in full[group]:assert torch.equal(full[group][key],resumed[group][key]),(group,key)
            # New flat path must exactly reproduce original flat+B update on common seeds.
            oldflat=run('oldflat',flat=True,original=True);newflat=run('newflat',flat=True)
            for group in ('vae','score','score_ema'):
                for key in oldflat[group]:assert torch.equal(oldflat[group][key],newflat[group][key]),('flat parity',group,key)
    finally:
        core.make_dataloaders,core.get_fid_model,core.LPIPS_AVAILABLE,core.evaluate_current_state,evaluation.evaluate,audit.run_audit=old
    print('PASS: actual routed trainer updates; exact uninterrupted/resume equality; exact flat+B baseline parity (reduced capacity synthetic RGB)')



def evaluator_check(base):
    """Exercise coherent A/B banks and real joint RK4 calls; use synthetic features."""
    import torch,core,evaluation
    from torch.utils.data import TensorDataset,DataLoader
    from stacked import build_models
    from design import rows
    old=(core.extract_inception_features,core.compute_kid,core.compute_fid_from_features,evaluation._RAW)
    def features(images,device,batch_size=1,inception_model=None):return images.flatten(1)[:,:8].float(),None
    core.extract_inception_features=features;core.compute_kid=lambda *a,**k:0.
    core.compute_fid_from_features=lambda a,b:float((a.mean(0)-b.mean(0)).square().sum())
    try:
        for cid in (0,6):
            cfg=small_cfg(rows()[cid],base);cfg.update(paper_cfg_grid=[],eval_rk4_steps_t=2,eval_rk4_steps_tk=2,sw2_n_projections=4)
            vae,head=build_models(cfg,torch.device('cpu'))
            ds=TensorDataset(torch.randn(4,3,32,32),torch.arange(4));loader=DataLoader(ds,batch_size=2)
            bank=torch.randn(4,8,8,8)
            evaluation._RAW=None
            result=evaluation.evaluate(1,'Synthetic',vae,head,loader,cfg,torch.device('cpu'),None,
                fixed_noise_bank=bank,fixed_posterior_eps_bank_A=bank,fixed_posterior_eps_bank_B=-bank,
                oracle_reference_loader=None)
            assert result['joint_reverse_solves']==1 and result['csem_gap_score_uncond']!=result['csem_gap_score_uncond']
            assert any(k.startswith('fid_rk4_2') for k in result)
    finally:core.extract_inception_features,core.compute_kid,core.compute_fid_from_features,evaluation._RAW=old
    print('PASS: coherent posterior A/B evaluation; joint RK4; inherited metric columns; invalid joint-Gaussian gap disabled (synthetic features only)')


def frozen_check(base):
    """Execute teacher branches on a tiny fixed representation; verify resume equality."""
    import torch,copy
    from torch.utils.data import TensorDataset,DataLoader
    import core,frozen
    from design import rows
    from stacked import build_models
    torch.set_num_threads(2)
    ds=TensorDataset(torch.randn(4,3,32,32,generator=torch.Generator().manual_seed(77)),torch.arange(4))
    old=(frozen.config_for,core.make_dataloaders,core.get_fid_model,frozen.evaluate,frozen.run_audit,frozen.save)
    def config(row,root,data_root=None):return small_cfg(row,root)
    def loaders(*a,**k):return DataLoader(ds,batch_size=2,shuffle=True),DataLoader(ds,batch_size=2),10
    frozen.config_for=config;core.make_dataloaders=loaders;core.get_fid_model=lambda *a,**k:(None,False)
    frozen.evaluate=lambda *a,**k:{'fid_synthetic_feature_only':1.};frozen.run_audit=lambda *a,**k:None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scfg=config(rows()[6],root);vae,head=build_models(scfg,torch.device('cpu'))
            source=root/'results'/rows()[6]['result_name']/'checkpoints';source.mkdir(parents=True)
            torch.save(dict(epoch=800,stage='joint',pending_eval=False,cfg=scfg,vae=vae.state_dict(),score_ema=head.state_dict()),source/'training_state_latest.pt')
            def run(label,teacher,interrupt=False,resume=False):
                row=dict(rows()[12],result_name=label,teacher=teacher,epochs_joint='2')
                def saving(path,value):
                    old[-1](path,value)
                    if interrupt and value['epoch']==1:raise InterruptedError('stop after frozen checkpoint')
                frozen.save=saving
                try:frozen.train(row,root,resume=resume)
                except InterruptedError:
                    if not interrupt:raise
                return torch.load(root/'results'/label/'checkpoints/frozen_latest.pt',map_location='cpu',weights_only=False)
            full=run('full','corrected');run('partial','corrected',interrupt=True);resumed=run('partial','corrected',resume=True)
            for name in ('head','ema'):
                for key in full[name]:assert torch.equal(full[name][key],resumed[name][key]),(name,key)
            for teacher in ('naive','dsm'):assert run(teacher,teacher)['epoch']==2
    finally:
        frozen.config_for,core.make_dataloaders,core.get_fid_model,frozen.evaluate,frozen.run_audit,frozen.save=old
    print('PASS: naive/corrected/DSM frozen optimizer updates and exact frozen resume (reduced capacity synthetic RGB)')




def compiler_check():
    import numpy as np,pandas as pd
    from design import rows
    import compile_results as compiler
    with tempfile.TemporaryDirectory() as tmp:
        base=Path(tmp)
        for row in rows():
            root=base/'results'/row['result_name'];root.mkdir(parents=True)
            values=[]
            for ep in (range(50,801,50) if row['stage']=='joint' else (0,50)):
                v=dict(epoch=ep,tag='LSI_Diff',stage='cotrain' if row['stage']=='joint' else 'frozen',fid_vae_recon=3.)
                for level in (1.,2.5,3.):
                    for method,n in (('rk4',20),('rk4',25),('heun',20)):
                        v[f'fid_{method}_{n}_randtok_cfg{str(level).replace(".","_")}_temp1_initgaussianT1p35']=10.+int(row['cell_id'])/100
                values.append(v)
            folder=root/('dataframes' if row['stage']=='joint' else '')
            folder.mkdir(exist_ok=True);pd.DataFrame(values).to_csv(folder/'eval_metrics.csv',index=False)
            if row['stage']=='frozen':
                (root/'audit').mkdir()
                a=np.arange(4,dtype=float);offset=0 if row['teacher']=='naive' else -1
                np.savez(root/'audit/epoch_0050_paired.npz',**{'source|class|0.2|b1|valid_risk':a,
                       'ema|class|0.2|b1|valid_risk':a+offset,'ema|class|0.2|b1|dsm_risk':a+offset})
        assert compiler.compile(base,strict=True)==0
        risk=pd.read_csv(base/'compiled/paired_valid_risk_contrasts.csv');assert len(risk)==16 and (risk['delta_risk']==-1).all()
        summary=pd.read_csv(base/'compiled/summary.csv');assert summary['complete'].all()
        p=base/'results'/rows()[0]['result_name']/'dataframes/eval_metrics.csv'
        d=pd.read_csv(p);d[d['epoch']<800].to_csv(p,index=False)
        try:compiler.compile(base,strict=True)
        except SystemExit:pass
        else:raise AssertionError('Compiler accepted missing endpoint')
        assert not pd.read_csv(base/'compiled/summary.csv').iloc[0]['complete']
    print('PASS: endpoint/paired-risk compiler and missing-endpoint rejection (fabricated fixture, no experimental results)')


if __name__=='__main__':
    base=Path(__file__).resolve().parent
    validate(base);smoke(base,0);smoke(base,6);gaussian_teacher_check();trainer_check(base);frozen_check(base);evaluator_check(base);compiler_check()
