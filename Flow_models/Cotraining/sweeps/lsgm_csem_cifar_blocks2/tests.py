"""Scientific, source, command and real-model graph checks; no training results.

CPU checks adapt CUDA allocation and NVAE's BN only inside this test process.
The delivered trainers always use CUDA and the original BN implementation.
"""
import argparse
import ast
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import warnings


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parser_for(path):
    tree=ast.parse(Path(path).read_text())
    block=next(n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=="__name__ == '__main__'")
    body=[]
    for n in block.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in n.targets): break
        body.append(n)
    scope={'argparse':argparse}
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),scope)
    return scope['parser']


def offline_checks(base):
    from paper import manifest, command_for, select, parse_slurm_job_id, prerequisite, config_hash, atomic_json
    base=Path(base)
    provenance=json.loads((base/'provenance.json').read_text())
    assert provenance['upstream_commit']=='5eae2f385c014f2250c3130152b6be711f6a3a5a'
    for name,rec in provenance['upstream_files'].items():
        assert sha(base/'lsgm'/name)==rec['patched_sha256'], 'Source hash mismatch: '+name
        if not rec['modified']: assert rec['upstream_sha256']==rec['patched_sha256']
    allowed={'util/datasets.py','thirdparty/functions.py','fid/fid_score.py','nvae.py',
             'training_obj_joint.py','train_vada.py','evaluate_diffusion.py','evaluate_vada.py',
             'score_sde/op/upfirdn2d.py','score_sde/op/fused_act.py','util/ema.py','util/utils.py'}
    assert set(provenance['modified_files'])==allowed
    for name,digest in provenance['new_files_sha256'].items():
        assert sha(base/name)==digest, 'Bundle hash mismatch: '+name
    assert sha(base/'paper_io.py')==provenance['paper_io_sha256']
    assert sha(base/'reference'/'training_obj_joint.py')==provenance['upstream_files']['training_obj_joint.py']['upstream_sha256']
    for p in base.rglob('*.py'): ast.parse(p.read_text(),filename=str(p))
    print('OK pinned source, allowed patch list, unchanged paper_io, Python syntax')
    rows=manifest(base)
    for row in rows:
        cmd=command_for(row,base)
        parser=parser_for(base/'lsgm'/cmd[2])
        parsed=parser.parse_args(cmd[3:])
        assert parsed.seed==42 and parsed.num_process_per_node==1
        if row['phase']=='B':
            assert parsed.epochs==1000 and parsed.batch_size==128
            assert parsed.iw_sample_p=='drop_all_iw' and parsed.iw_sample_q=='ll_iw'
            assert parsed.learning_rate_dae==parsed.learning_rate_min_dae==1e-4
            assert parsed.time_eps==.01 and parsed.num_cell_per_scale_dae==2
            assert Path(parsed.vae_checkpoint).parent.name==row['vae_cell']
        if row['phase']=='A': assert parsed.epochs==400 and parsed.batch_size==256
        if row['phase']=='C': assert parsed.num_fid_samples==50000 and parsed.ode_eps==1e-6
    assert [r['cell_id'] for r in select(rows,'csem','B')]==['B1-C','B2-C','B4-C']
    # Apart from arm flags and paths, paired stage-2 commands are identical.
    arm_keys={'csem','prior_factorization','dae_channels_per_head','no_csem_band','no_csem_terminal',
              'rep_horizon','csem_lambda_K','save','eval_csv','diagnostics_csv'}
    for g in (1,2,4):
        values=[]
        for suffix in ('L','C'):
            r=next(r for r in rows if r['cell_id']==f'B{g}-{suffix}')
            values.append(vars(parser_for(base/'lsgm'/'train_vada.py').parse_args(command_for(r,base)[3:])))
        assert {k:v for k,v in values[0].items() if k not in arm_keys}=={k:v for k,v in values[1].items() if k not in arm_keys}
    for text in ('12345\n','12345;vista\n','notice\n12345;vista\n','Submitted batch job 12345\n'):
        assert parse_slurm_job_id(text)=='12345'
    try: parse_slurm_job_id('warning: invalid account 12345')
    except ValueError: pass
    else: raise AssertionError('Accepted a warning as job ID')
    with tempfile.TemporaryDirectory() as tmp:
        dep=next(r for r in rows if r['cell_id']=='A1')
        row=next(r for r in rows if r['cell_id']=='B1-C')
        assert prerequisite(tmp,row,rows)
        atomic_json(Path(tmp)/'status'/'A1.json',{'state':'completed','returncode':0,'config_hash':config_hash(dep)})
        path=Path(tmp)/'results'/'A1';path.mkdir(parents=True);(path/'checkpoint.pt').touch()
        assert prerequisite(tmp,row,rows) is None
    parameter_count_checks(base)
    scheduler_checks(base)
    print('OK all 26 CLI configurations, 1/2/4 groups, paired parity, SLURM parser and phase gating')


def parameter_count_checks(base):
    # Execute the exact helper without needing torch/GPU imports on login nodes.
    import math
    path=Path(base)/'lsgm'/'util'/'utils.py'
    tree=ast.parse(path.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='count_parameters_in_M')
    scope={}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),scope)
    count=scope['count_parameters_in_M']
    class Parameter:
        def __init__(self,shape): self.shape=shape
        def numel(self): return math.prod(self.shape)
    class Model:
        def __init__(self,entries): self.entries=entries
        def named_parameters(self): return iter(self.entries)
    entries=[('weight',Parameter((2,3))),('bias',Parameter((3,))),
             ('scalar',Parameter(())),('empty',Parameter((0,5))),
             ('auxiliary.weight',Parameter((100,100)))]
    assert count(Model(entries)) == 10/1e6
    assert count(Model([])) == 0
    # Reject the same NumPy generator pattern anywhere in the vendored code.
    for p in (Path(base)/'lsgm').rglob('*.py'):
        tree=ast.parse(p.read_text())
        numpy_names={a.asname or a.name for n in ast.walk(tree) if isinstance(n,ast.Import)
                     for a in n.names if a.name=='numpy'}
        for n in ast.walk(tree):
            if (isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                    and isinstance(n.func.value,ast.Name) and n.func.value.id in numpy_names):
                assert not any(isinstance(arg,ast.GeneratorExp) for arg in n.args), \
                    'NumPy generator argument: '+str(p)+':'+str(n.lineno)
    print('OK parameter logging count, auxiliary exclusion, scalar/empty parameters, NumPy generator scan')


def scheduler_checks(base):
    """Expired queue records must not block resubmission or mask outages."""
    import subprocess
    from unittest.mock import patch
    from contextlib import redirect_stdout
    from io import StringIO
    import paper
    expired = subprocess.CompletedProcess([], 1, '',
        'slurm_load_jobs error: Invalid job id specified\n')
    with patch.object(paper.shutil, 'which', return_value='/usr/bin/squeue'):
        for response, expected in (
                (expired, False),
                (subprocess.CompletedProcess([], 1, '',
                    'squeue: slurm_load_jobs error: Invalid job id specified\n'), False),
                (subprocess.CompletedProcess([], 0, '', ''), False),
                (subprocess.CompletedProcess([], 0, '12345\n', ''), True)):
            with patch.object(paper.subprocess, 'run', return_value=response):
                assert paper.job_active('12345') is expected
        for error in ('Unable to contact slurm controller', 'Permission denied',
                      'slurm_load_jobs error: Invalid job id specified\nConnection refused'):
            response = subprocess.CompletedProcess([], 1, '', error)
            with patch.object(paper.subprocess, 'run', return_value=response):
                try: paper.job_active('12345')
                except RuntimeError: pass
                else: raise AssertionError('Scheduler error was masked: '+error)

        rows = paper.manifest(base)
        chosen = paper.select(rows, 'csem', 'A')
        with tempfile.TemporaryDirectory() as tmp:
            for i, row in enumerate(chosen):
                paper.atomic_json(paper.status_path(tmp, row), {
                    'state': 'submitted', 'job_id': str(100+i),
                    'config_hash': paper.config_hash(row), 'returncode': None})
            calls = []
            def scheduler(cmd, **kw):
                calls.append(cmd)
                if cmd[0] == 'squeue':
                    if cmd[3] == '101':
                        return subprocess.CompletedProcess(cmd, 0, '101\n', '')
                    return expired
                return subprocess.CompletedProcess(cmd, 0, str(900+len(calls))+'\n', '')
            with patch.object(paper.subprocess, 'run', side_effect=scheduler), redirect_stdout(StringIO()):
                paper.submit(tmp, chosen, rows)
                # B/C remain gated until prerequisites actually complete.
                paper.submit(tmp, paper.select(rows, 'csem', 'B'), rows)
                paper.submit(tmp, paper.select(rows, 'csem', 'C'), rows)
            assert len([cmd for cmd in calls if cmd[0] == 'sbatch']) == 2
            assert paper.status(tmp, chosen[0])['job_id'] != '100'
            assert paper.status(tmp, chosen[1])['job_id'] == '101'
            assert paper.status(tmp, chosen[2])['job_id'] != '102'
    print('OK expired SLURM IDs resubmit, active jobs skip, outages fail, B/C gating preserved')


@contextmanager
def cpu_test_adapter():
    """Only a test adapter; never loaded by paper.py run or lsgm_runner.py."""
    import torch
    import torch.nn.functional as F
    from unittest.mock import patch
    from contextlib import ExitStack
    from thirdparty.inplaced_sync_batchnorm import SyncBatchNormSwish
    from thirdparty.swish import Swish
    def bn(self,x):
        if self.training and self.track_running_stats: self.num_batches_tracked.add_(1)
        return Swish.apply(F.batch_norm(x,self.running_mean,self.running_var,self.weight,self.bias,
                                       self.training or not self.track_running_stats,self.momentum,self.eps))
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self))
        stack.enter_context(patch.object(torch.nn.Module,'cuda',lambda self,*a,**kw:self))
        stack.enter_context(patch.object(SyncBatchNormSwish,'forward',bn))
        for name in ('tensor','rand','randn','ones','zeros','empty','ones_like','zeros_like','randn_like'):
            original=getattr(torch,name)
            def factory(*a,_f=original,**kw):
                if str(kw.get('device','')).startswith('cuda'): kw['device']='cpu'
                return _f(*a,**kw)
            stack.enter_context(patch.object(torch,name,factory))
        yield


def load_torch_modules(base):
    sys.path.insert(0,str(Path(base)/'lsgm'))
    import torch
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    warnings.filterwarnings('ignore',category=FutureWarning)
    import csem_lsgm
    return torch,csem_lsgm


def numeric_checks(base):
    torch,c=load_torch_modules(base)
    torch.manual_seed(81)
    # Conditional Gaussian z0|zt; draw the implied noise at fixed zt.
    mu=torch.tensor([.7,-1.2],dtype=torch.float64)
    sig=torch.tensor([.15,.9],dtype=torch.float64)
    zt=torch.tensor([.3,-.4],dtype=torch.float64)
    m=torch.tensor(.83,dtype=torch.float64);v=torch.tensor(.31,dtype=torch.float64)
    comp=m*m*sig*sig+v
    mean0=mu+m*sig*sig/comp*(zt-m*mu)
    variance0=sig*sig*v/comp
    z0=mean0+variance0.sqrt()*torch.randn(150000,2,dtype=torch.float64)
    noise=(zt-m*z0)/v.sqrt()
    target=c.csem_eps_target(zt,mu,sig.log(),m,v)
    torch.testing.assert_close(noise.mean(0),target,atol=.009,rtol=0)
    # RB gradient variance gap equals alpha^2 sigma_q^2 / component variance.
    torch.testing.assert_close(noise.var(0),m*m*sig*sig/comp,atol=.008,rtol=0)
    # Band bound on exact Gaussian-channel mutual information.
    sig=torch.exp(torch.empty(20000).uniform_(-7,3))
    signal=torch.exp(torch.empty(20000).uniform_(-8,5))
    m=torch.tensor(.999);v=torch.tensor(.001995)
    lost=.5*torch.log1p(signal/sig.square())-.5*torch.log1p(m.square()*signal/(m.square()*sig.square()+v))
    B=c.band_term(sig.log(),m,v)
    assert torch.all(lost<=B+3e-6)
    ls=torch.tensor([-15.,-3.,3.],requires_grad=True)
    b=c.band_term(ls,m,v);b.sum().backward()
    assert torch.isfinite(b).all() and torch.isfinite(ls.grad).all()
    assert abs(ls.grad[0].item()+1)<1e-5 and b[-1]<3e-6
    # A population identity at the component oracle, modulo fixed constants.
    m1=torch.tensor(__import__('math').exp(-5.025),dtype=torch.float64)
    v1=1-m1*m1;me=torch.tensor(__import__('math').exp(-.0009975),dtype=torch.float64);ve=1-me*me
    variance=torch.tensor([.03,.7,4.],dtype=torch.float64)
    mean=torch.tensor([.2,1.,-3.],dtype=torch.float64)
    vc_e=me*me*variance+ve;vc_1=m1*m1*variance+v1
    lsgm=.5*torch.log(v1*vc_e/(vc_1*variance))
    csem=c.band_term(.5*variance.log(),me,ve)+c.terminal_component_kl(mean,.5*variance.log(),m1,v1)
    expected=.5*m1*m1*(mean.square()+variance-1)-.5*torch.log(me*me*v1)
    torch.testing.assert_close(csem-lsgm,expected,atol=2e-12,rtol=0)
    print('OK RB target/variance, 20,000 band bounds, band derivatives, finite-horizon population identity')


def cascade_contract_checks(base):
    torch,c=load_torch_modules(base)
    from types import SimpleNamespace
    class Head(torch.nn.Module):
        def __init__(self,ch):
            super().__init__();self.net=SimpleNamespace(mixed_prediction=True,mixing_logit=torch.zeros(1,ch,1,1))
        def forward(self,x,t,context=None): return x if context is None else x+context.mean(1,keepdim=True)
    class Diffusion:
        def __init__(self): self.calls=[]
        def sample_model_ode(self,head,n,shape,eps,tol,amp,temp,noise=None):
            self.calls.append(None if head.context is None else head.context.clone())
            return head(noise,torch.tensor(.5)),7,0.1
        def compute_ode_nll(self,dae,eps,**kw):
            self.calls.append(None if dae.context is None else dae.context.clone())
            return eps.square().sum((1,2,3)),5,None,None
    dae=SimpleNamespace(heads=[Head(1),Head(2),Head(1)],group_channels=[1,2,1],num_input_channels=4,input_size=2)
    noise=torch.arange(16).reshape(1,4,2,2).float()
    d=Diffusion();out,nfe,_=c.sample_cascade_ode(d,dae,1,.001,1e-5,False,1,noise)
    assert d.calls[0] is None
    torch.testing.assert_close(d.calls[1],out[:,:1]);torch.testing.assert_close(d.calls[2],out[:,:3])
    assert nfe==21 and dae.last_sampling_nfe_per_head==[7,7,7]
    d2=Diffusion();out2,_,_=c.sample_cascade_ode(d2,dae,1,.001,1e-5,False,1,noise)
    torch.testing.assert_close(out,out2,atol=0,rtol=0)
    single=SimpleNamespace(heads=[Head(4)],group_channels=[4])
    kwargs=dict(ode_eps=.001,ode_solver_tol=1e-5,enable_autocast=False,no_autograd=False,num_samples=1,report_std=False)
    original=Diffusion().compute_ode_nll(c._BoundHead(single.heads[0],None),noise,**kwargs)
    got=c.cascade_ode_nll(Diffusion(),single,noise,**kwargs)
    torch.testing.assert_close(got[0],original[0]);assert got[1:]==original[1:]
    vals=c.cascade_ode_nll(Diffusion(),dae,noise,**kwargs)
    torch.testing.assert_close(vals[0],noise.square().sum((1,2,3)))
    print('OK cascade call order, generated ancestors, deterministic noise bank, G=1 and joint NLL')


def model_args(base,row,full_size=False):
    from paper import command_for,manifest
    source=row
    if row['phase']=='C': source=next(r for r in manifest(base) if r['cell_id']==row['vae_cell'])
    if source['phase']=='A':
        a=parser_for(Path(base)/'lsgm'/'train_vae.py').parse_args(command_for(source,base)[3:])
        # Add SGM fields for smoke only; training A remains train_vae.py unchanged.
        prior=next(r for r in manifest(base) if r['phase']=='B' and r['vae_cell']==source['cell_id'])
    else:
        prior=source
        ar=next(r for r in manifest(base) if r['cell_id']==source['vae_cell'])
        a=parser_for(Path(base)/'lsgm'/'train_vae.py').parse_args(command_for(ar,base)[3:])
    p=parser_for(Path(base)/'lsgm'/'train_vada.py').parse_args(command_for(prior,base)[3:])
    from util import utils
    # Exactly the checkpoint architecture override performed by train_vada.py.
    class Log:
        def info(self,*a,**kw): pass
    utils.override_architecture_fields(p,a,Log())
    p.mixed_prediction=True
    p.distributed=False;p.global_rank=0;p.num_total_iter=10000
    p.batch_size=2;p.autocast_train=False;p.autocast_eval=False
    p.max_steps_per_epoch=5;p.diagnostics_csv='';p.current_epoch=1
    if not full_size:
        p.num_channels_enc=p.num_channels_dec=16
        p.num_preprocess_cells=p.num_postprocess_cells=1
        p.num_cell_per_cond_enc=p.num_cell_per_cond_dec=1
        p.num_latent_per_group=2
        p.num_channels_dae=32;p.dae_channels_per_head=32
        p.num_cell_per_scale_dae=1
    return p


def captured_trainer(module):
    source=inspect.getsource(module.train_vada_joint)
    marker='    # backpropagate q_loss for vae and update vae params, if trained'
    # Capture the real function's first graph before its two backward passes.
    source=source.replace(marker,'    return q_loss, p_loss\n'+marker)
    scope=dict(module.__dict__)
    exec(compile(source,'captured_'+module.__name__,'exec'),scope)
    return scope['train_vada_joint']


def torch_smoke(base,row,device='cuda',full_size=False):
    torch,c=load_torch_modules(base)
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA smoke needs a visible CUDA GPU')
    from contextlib import nullcontext
    from util import utils
    from nvae import NVAE
    from diffusion_continuous import make_diffusion
    import training_obj_joint as trainer
    from util.sr_utils import SpectralNormCalculator
    from torch.cuda.amp import GradScaler
    ctx=cpu_test_adapter() if device=='cpu' else nullcontext()
    initialized=False
    if device=='cuda' and not torch.distributed.is_initialized():
        import tempfile
        init=Path(tempfile.gettempdir())/('lsgm-smoke-'+str(os.getpid()))
        torch.distributed.init_process_group('nccl',init_method='file://'+str(init),rank=0,world_size=1)
        initialized=True
    start=time.time()
    try:
        with ctx:
            args=model_args(base,row,full_size)
            if device=='cuda':
                args.autocast_train=True
            c.validate_args(args)
            torch.manual_seed(42)
            vae=NVAE(args,utils.get_arch_cells(args.arch_instance,args.use_se)).to(device)
            dae=c.build_prior(args,vae).to(device)
            # Exercise the shared startup logging helper used by A/B/C trainers.
            for model in (vae,dae):
                expected=sum(p.numel() for name,p in model.named_parameters() if 'auxiliary' not in name)/1e6
                assert utils.count_parameters_in_M(model)==expected
            diffusion=make_diffusion(args)
            x=torch.rand(args.batch_size,3,32,32,device=device)
            vae(x*2-1)
            if args.num_nf==0:
                # Distribution round-trip and group splitting at the realized samples.
                _,_,zs=vae(x*2-1)
                for z,q in zip(zs,vae.last_q):
                    rho=(z-q.mu)/q.sigma
                    torch.testing.assert_close(z,q.sample_given_rho(rho),atol=2e-6,rtol=2e-6)
                for z,r in zip(zs,vae.decompose_eps(vae.concat_eps_per_scale(zs)[0])):
                    torch.testing.assert_close(z,r,atol=0,rtol=0)
            class Writer:
                def add_scalar(self,*a,**kw): pass
                def add_histogram(self,*a,**kw): pass
            class Log:
                def info(self,*a,**kw): pass
            def sn(v,d):
                vs=SpectralNormCalculator(num_power_iter=1,custom_conv=True);vs.add_conv_layers(v);vs.add_bn_layers(v)
                ds=SpectralNormCalculator(num_power_iter=1,custom_conv=False);ds.add_conv_layers(d);ds.add_bn_layers(d)
                return ds,vs
            def call(fn,v,d,steps=1):
                ds,vs=sn(v,d)
                vo=torch.optim.Adam(v.parameters(),lr=1e-4,eps=1e-3,weight_decay=3e-4)
                from util.ema import EMA
                do=EMA(torch.optim.Adam(d.parameters(),lr=1e-4,eps=1e-4,weight_decay=3e-4),ema_decay=.9999)
                scaler=GradScaler(2**10,enabled=device=='cuda')
                return fn([(x,None)]*steps,diffusion,d,do,v,vo,scaler,10,5,Writer(),Log(),ds,vs,args)
            torch.manual_seed(91)
            q_loss,p_loss=call(captured_trainer(trainer),vae,dae)
            assert torch.isfinite(q_loss) and torch.isfinite(p_loss)
            p_to_vae=torch.autograd.grad(p_loss,list(vae.parameters()),retain_graph=True,allow_unused=True)
            assert all(g is None or torch.count_nonzero(g)==0 for g in p_to_vae), 'p-route reaches VAE'
            q_grads=torch.autograd.grad(q_loss,list(vae.enc_sampler.parameters()),retain_graph=True,allow_unused=True)
            assert any(g is not None and torch.count_nonzero(g)>0 for g in q_grads), 'q-route misses encoder'
            expected=torch.autograd.grad(p_loss,list(dae.parameters()),allow_unused=True)
            # The default joint LSGM path must retain the exact upstream first losses.
            if not args.csem and args.prior_factorization=='joint':
                spec=importlib.util.spec_from_file_location('reference_joint',Path(base)/'reference'/'training_obj_joint.py')
                ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
                rv=NVAE(args,utils.get_arch_cells(args.arch_instance,args.use_se)).to(device)
                rd=c.build_prior(args,rv).to(device)
                rv.load_state_dict(vae.state_dict());rd.load_state_dict(dae.state_dict())
                torch.manual_seed(91)
                refq,refp=call(captured_trainer(ref),rv,rd)
                torch.testing.assert_close(refq,q_loss,atol=0,rtol=0)
                torch.testing.assert_close(refp,p_loss,atol=0,rtol=0)
            torch.manual_seed(91)
            # First step reproduces the captured p gradients, despite q backward.
            call(trainer.train_vada_joint,vae,dae,steps=1)
            for got,want in zip(dae.parameters(),expected):
                if want is not None: torch.testing.assert_close(got.grad,want,atol=2e-5,rtol=2e-5)
            call(trainer.train_vada_joint,vae,dae,steps=4)
            assert all(torch.isfinite(p).all() for p in list(vae.parameters())+list(dae.parameters()))
            if isinstance(dae,c.CascadeDAE):
                live=torch.randn(2,dae.num_input_channels,dae.input_size,dae.input_size,device=device,requires_grad=True)
                pred=dae(torch.randn_like(live),torch.full((2,),.4,device=device),live)
                anc_grad=torch.autograd.grad(pred.square().sum(),live)[0]
                assert torch.count_nonzero(anc_grad[:,:-args.num_latent_per_group])>0, 'Clean ancestor route is inactive'
                assert torch.count_nonzero(anc_grad[:,-args.num_latent_per_group:])==0, 'Head sees itself/future groups in context'
            # Optimizer EMA / module state roundtrip, including conditional adapters.
            from util.ema import EMA
            opt=EMA(torch.optim.Adam(dae.parameters(),lr=1e-4),ema_decay=.9999)
            opt.step(); saved=copy.deepcopy(opt.state_dict())
            opt.load_state_dict(saved); opt.swap_parameters_with_ema(True);opt.swap_parameters_with_ema(True)
            clone=copy.deepcopy(dae);clone.load_state_dict(dae.state_dict())
            result={'cell':row['cell_id'],'device':device,'full_size':full_size,
                    'five_training_steps':True,'gradient_routing':True,'prior_gradients_p_only':True,
                    'q_loss_first':q_loss.item(),'p_loss_first':p_loss.item(),
                    'seconds':time.time()-start,**c.parameter_counts(vae,dae)}
            print('OK real NVAE/NCSN++ smoke:',json.dumps(result))
            return result
    finally:
        if initialized: torch.distributed.destroy_process_group()


def torch_checks(base):
    numeric_checks(base);cascade_contract_checks(base)
    from paper import manifest
    results=[]
    for cid in ('B1-L','B1-C','B2-L','B2-C','B4-L','B4-C','B2-LC','B4-LC','B0','B1-C2H','B1-Cnb'):
        row=next(r for r in manifest(base) if r['cell_id']==cid)
        results.append(torch_smoke(base,row,'cpu'))
    real_ode_checks(base)
    print('OK numeric and reduced-width real-model CPU checks; CUDA and full CIFAR training are separate')
    return results


def real_ode_checks(base):
    torch,c=load_torch_modules(base)
    from paper import manifest
    from util import utils
    from nvae import NVAE
    from diffusion_continuous import make_diffusion
    row=next(r for r in manifest(base) if r['cell_id']=='B2-C')
    with cpu_test_adapter():
        args=model_args(base,row)
        torch.manual_seed(82)
        vae=NVAE(args,utils.get_arch_cells(args.arch_instance,args.use_se))
        dae=c.build_prior(args,vae).eval();diffusion=make_diffusion(args)
        bank=torch.randn(1,dae.num_input_channels,16,16)
        with torch.no_grad():
            z,k,s=c.sample_cascade_ode(diffusion,dae,1,.02,2e-3,False,1,bank)
            z2,k2,_=c.sample_cascade_ode(diffusion,dae,1,.02,2e-3,False,1,bank)
            torch.testing.assert_close(z,z2,atol=0,rtol=0)
            assert k==k2 and torch.isfinite(z).all()
            nll,kn,_,_=c.cascade_ode_nll(diffusion,dae,z,ode_eps=.02,ode_solver_tol=.05,
                                       enable_autocast=False,no_autograd=False,num_samples=1,report_std=False)
            assert nll.shape==(1,) and torch.isfinite(nll).all() and kn>0
            image=vae.sample(1,1.,vae.decompose_eps(z),False)
            assert image.shape==(1,3,32,32) and torch.isfinite(image).all()
        print('OK real conditional RK45 sampling, prior likelihood and NVAE decoding:',{'nfe':k,'nll_nfe':kn})
