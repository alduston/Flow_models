"""CPU regression checks of actual config, sampler, I/O, checkpoint and reporting paths.
Fixtures are temporary; no synthetic values are delivered as experimental results.
"""
import ast, csv, copy, importlib.util, json, math, multiprocessing, os, pickle
import random, sys, tempfile, types
from pathlib import Path
import numpy as np
import pandas as pd
from paper_io import prepare_dataset, load_prepared_pair, normalize_metrics, sampler_configs, validate_resume_config

class TinyDataset:
    """Emulates integrity checks and dataset deserialization on a shared filesystem."""
    def __init__(self, root, train, download=False, transform=None):
        root=Path(root);root.mkdir(parents=True,exist_ok=True)
        file=root/('train.pkl' if train else 'test.pkl')
        if download and not file.exists():
            with file.open('wb') as f:pickle.dump([1,2,3],f)
        with file.open('rb') as f:self.data=pickle.load(f)
        if self.data != [1,2,3]:raise RuntimeError('corrupt split')
    def __len__(self):return len(self.data)
    def _check_integrity(self):return self.data==[1,2,3]

class RetryDataset(TinyDataset):
    failures=0
    def __init__(self,root,train,download=False,transform=None):
        if download and train and type(self).failures<1:
            type(self).failures+=1
            (Path(root)/'train.pkl').write_bytes(b'partial')
            raise RuntimeError('interrupted download')
        super().__init__(root,train,download,transform)

def prepare_worker(root):prepare_dataset(TinyDataset,root,'CIFAR')

class FakeObject:
    def __init__(self,name):self.name=name;self.restored=None
    def state_dict(self):return {'name':self.name}
    def load_state_dict(self,state):self.restored=state
class RNG:
    def cpu(self):return self
class FakeTorch:
    cuda=types.SimpleNamespace(is_available=lambda:False)
    @staticmethod
    def get_rng_state():return RNG()
    @staticmethod
    def set_rng_state(state):pass
    @staticmethod
    def save(state,path):
        with open(path,'wb') as f:pickle.dump(state,f)
    @staticmethod
    def load(path,**kw):
        with open(path,'rb') as f:return pickle.load(f)


def extract(node, namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)],type_ignores=[])), '<actual_source>', 'exec'),namespace)


def config_checks(base,tree):
    # Resolve real presets/suite code without importing the unavailable CUDA runtime.
    stub=types.ModuleType('core')
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'DATASET_PRESETS','DATASET_TO_PRESET'} for t in node.targets):extract(node,stub.__dict__)
        if isinstance(node,ast.FunctionDef) and node.name=='resolve_model_preset':extract(node,stub.__dict__)
    stub.resolve_bespoke_fid_classifier=lambda dataset,override:False
    previous=sys.modules.get('core');sys.modules['core']=stub
    try:
        spec=importlib.util.spec_from_file_location('_tested_suite',base/'suite.py')
        suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)
        rows=list(csv.DictReader((base/'manifest.csv').open()))
        assert [int(r['cell_id']) for r in rows]==list(range(22))
        assert len({r['result_name'] for r in rows})==22
        assert not any(any(ord(c)<32 for c in r['result_name']) for r in rows)
        independent=[]
        for r in rows:
            cfg=suite.resolve_config(r,base)
            assert cfg['lr_score_head']==float(r['lr_score_head'])
            assert cfg['t_max']>=cfg['T_terminal']>=0
            plan=sampler_configs(cfg)
            if r['mode']=='independent_pair':
                independent.append((r['dataset'],int(r['seed'])))
                assert cfg['freeze_score_in_cotrain'] and cfg['train_tracking_head']
                assert cfg['score_w_vae']==cfg['T_terminal']==0
                assert cfg['epochs_refine']==int(r['epochs_refine'])>0
                assert cfg['lr_ldm']==cfg['lr_refine']==cfg['lr_score_head']
                assert not any(c['init_mode'].startswith('oracle-') for c in plan)
                assert set(cfg['paper_cfg_grid'])==({3.,1.5} if r['dataset']=='FMNIST' else {2.5,3.})
            if r['mode']=='fmnist_aug13_gaussian':
                assert cfg['training_engine']=='aug13_joint'
                assert cfg['epochs_vae']==700 and cfg['T_terminal']==cfg['t_max']==2
                assert cfg['kl_w']==1 and cfg['score_w_vae']==.6
                assert math.isclose(cfg['lr_ldm'],2e-4) and cfg['score_head_loss_w']==.6
            print(f"OK cell {r['cell_id']:>2}: {r['dataset']} / {r['outputs']} / seed {r['seed']}")
        assert set(independent)=={('CIFAR',42),('CIFAR',43),('FMNIST',42),('FMNIST',43)}
        # Legacy module remains byte-for-byte the recovered training source.
        assert any(n.name=='train_vae_cotrained_cond' for n in ast.parse((base/'legacy_fmnist.py').read_text()).body if isinstance(n,ast.FunctionDef))
        return rows
    finally:
        if previous is None:sys.modules.pop('core',None)
        else:sys.modules['core']=previous


def sampler_checks(tree):
    cfg=dict(t_max=2.,T_terminal=0.,t_min=2e-5,cfg_eval_scale=3.,eval_sampling_init='gaussian',
             eval_tk_vs_t_comparison=False,paper_cfg_grid=[3.,1.5],paper_solver_suite=True)
    plan=sampler_configs(cfg)
    assert all(not c['init_mode'].startswith('oracle-') for c in plan)
    assert len({(c['method'],c['steps'],c.get('cfg_level'),c['init_mode']) for c in plan})==len(plan)
    for g in [3.,1.5]:
        assert any(c['method']=='rk4_ode' and c['steps']==20 and c.get('cfg_level')==g for c in plan)
        assert any(c['method']=='heun_sde' and c['steps']==20 and c.get('cfg_level')==g for c in plan)
    cfg['eval_tk_vs_t_comparison']=True
    assert {c['init_mode'] for c in sampler_configs(cfg) if c['init_mode'].startswith('oracle-')}=={'oracle-qT-class'}
    cfg['T_terminal']=1.05
    assert {c['init_mode'] for c in sampler_configs(cfg) if c['init_mode'].startswith('oracle-')}=={'oracle-qT-class','oracle-qtk-class'}
    cfg['deployment_only']=True
    assert not any(c['init_mode'].startswith('oracle-') for c in sampler_configs(cfg))
    # Execute actual evaluator planning assignments; independent banks must agree with the requested plan.
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_current_state')
    scope={'cfg':dict(cfg,deployment_only=False,eval_tk_vs_t_comparison=False,T_terminal=0.),'unet':object(),'sampler_configs':sampler_configs}
    for node in fn.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'planned_samplers','needs_oracle_qt_init','configs'} for t in node.targets):extract(node,scope)
    assert scope['needs_oracle_qt_init'] is False
    assert not any(c['init_mode'].startswith('oracle-') for c in scope['configs'])


def header_checks():
    f=pd.DataFrame([{'fid_rk4_20_randtok_cfg3_0':5.,'fid_rk4_20_randtok_cfg1_5':6.}])
    g,_=normalize_metrics(f)
    assert list(g)==['fid_rk4_20_randtok_cfg3_0_temp1_initgaussianT2','fid_rk4_20_randtok_cfg1_5_temp1_initgaussianT2']
    damaged=pd.DataFrame([[5.,6.]],columns=['fid_rk4_20\x01_temp1_initgaussianT2','fid_rk4_20\x01_temp1_initgaussianT2.1'])
    recovered,_=normalize_metrics(damaged,recover_v5=True)
    assert np.array_equal(recovered.values,damaged.values)
    assert list(recovered)==list(g)
    heun,_=normalize_metrics(pd.DataFrame([{'fid_heun_50\x01_temp1_initgaussianT2':5.}]),recover_v5=True)
    assert 'fid_heun_50_randtok_cfg3_0_temp1_initgaussianT2' in heun
    for bad,kwargs in [(damaged,{}),(damaged.iloc[:,:1],{'recover_v5':True})]:
        try:normalize_metrics(bad,**kwargs)
        except ValueError:pass
        else:raise AssertionError('Ambiguous/corrupt name accepted')


def data_checks():
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        try:load_prepared_pair(TinyDataset,root,'CIFAR')
        except RuntimeError:pass
        else:raise AssertionError('Missing data accepted')
        # Fresh directory staging recovers a real truncated pickle.
        cache=root/'CIFAR';cache.mkdir(exist_ok=True);(cache/'train.pkl').write_bytes(b'\x80\x04broken')
        prepare_dataset(TinyDataset,root,'CIFAR')
        assert tuple(map(len,load_prepared_pair(TinyDataset,root,'CIFAR')))==(3,3)
        assert len(list(root.glob('.CIFAR-corrupt-*')))==1
        (cache/'test.pkl').write_bytes(b'bad')
        try:load_prepared_pair(TinyDataset,root,'CIFAR')
        except RuntimeError:pass
        else:raise AssertionError('Corrupt data accepted by training')
        prepare_dataset(TinyDataset,root,'CIFAR')
        RetryDataset.failures=0
        prepare_dataset(RetryDataset,root,'FMNIST')
        assert RetryDataset.failures==1 and not list(root.glob('.*-stage-*'))
        # Multiple independent preparers exercise the actual file-lock path.
        workers=[multiprocessing.get_context('fork').Process(target=prepare_worker,args=(str(root/'parallel'),)) for _ in range(3)]
        for process in workers:process.start()
        for process in workers:process.join(15);assert process.exitcode==0
        assert tuple(map(len,load_prepared_pair(TinyDataset,root/'parallel','CIFAR')))==(3,3)


def checkpoint_checks(tree):
    train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_vae_cotrained_cond')
    save=next(n for n in ast.walk(train) if isinstance(n,ast.FunctionDef) and n.name=='save_paper_training_state')
    evaluator=next(n for n in ast.walk(train) if isinstance(n,ast.FunctionDef) and n.name=='evaluate_prior_epoch')
    resume=next(n for n in train.body if isinstance(n,ast.If) and ast.unparse(n.test)=='resume_path')
    with tempfile.TemporaryDirectory() as temp:
        Path(temp,'dataframes').mkdir()
        scope=dict(torch=FakeTorch,os=os,np=np,math=math,pd=pd,freeze_score_in_cotrain=True,
                   cfg=dict(paper_cfg_grid=[3.,1.5],cfg_eval_scale=3.,ckpt_dir=temp,epochs_vae=700,epochs_refine=700),
                   disc=None,opt_disc=None,loss_records=[dict(epoch=750,stage='refine')],eval_records=[],
                   global_cotrain_batch=0,results_dir=temp,train_tracking_head=True)
        for name in ['vae','unet_lsi','unet_lsi_ema','unet_control','unet_control_ema','opt_joint','sched_joint',
                     'opt_lsi_refine','sched_lsi_refine','opt_control_refine','sched_control_refine']:
            scope[name]=FakeObject(name)
        for name in ['test_l','device','lpips_fn','fixed_noise_bank','fixed_posterior_eps_bank_A','fixed_posterior_eps_bank_B',
                     'fixed_sw2_theta','fid_model','use_lenet_fid','train_l']:scope.setdefault(name,None)
        extract(save,scope);scope['save_paper_training_state'](700,stage='vae')
        checkpoint=Path(temp,'training_state_latest.pt')
        state=FakeTorch.load(checkpoint);assert state['stage']=='vae' and state['epoch']==700
        scope['save_paper_training_state'](50,stage='prior',pending_eval=True)
        state=FakeTorch.load(checkpoint)
        assert state['stage']=='prior' and state['pending_eval']
        assert state['prior_optimizer']['name']=='opt_lsi_refine'
        assert state['control_prior_optimizer']['name']=='opt_control_refine'
        calls=[]
        def fails_second(epoch,tag,*args,**kw):
            calls.append(tag)
            if tag=='Ctrl_Diff_Refine':raise RuntimeError('evaluation interrupted')
            return {'fid_rk4_20_randtok_cfg3_0_temp1_initgaussianT2':5.}
        scope['evaluate_current_state']=fails_second;extract(evaluator,scope)
        try:scope['evaluate_prior_epoch'](50)
        except RuntimeError:pass
        else:raise AssertionError('Injected evaluation failure did not happen')
        # Saved state predates evaluation and preserves both trained priors.
        assert FakeTorch.load(checkpoint)['eval_records']==[]
        scope.update(resume_path=str(checkpoint),start_epoch=0,refine_start_epoch=0)
        extract(resume,scope)
        assert scope['start_epoch']==700 and scope['refine_start_epoch']==50
        assert scope['unet_control'].restored==state['control']
        scope['evaluate_current_state']=lambda epoch,tag,*a,**k:{'fid_rk4_20_randtok_cfg3_0_temp1_initgaussianT2':5.}
        # A retry removes any partial first-head record and writes both paired heads exactly once.
        scope['evaluate_prior_epoch'](50);scope['evaluate_prior_epoch'](50)
        frame=pd.read_csv(Path(temp,'dataframes','eval_metrics_in_progress.csv'))
        assert set(frame.tag)=={'LSI_Diff_Refine','Ctrl_Diff_Refine'} and len(frame)==2
        assert set(frame.epoch)=={50} and set(frame.stage)=={'refine'}
        scope['save_paper_training_state'](50,stage='prior')
        assert not FakeTorch.load(checkpoint)['pending_eval']
        try:validate_resume_config({'kl_w':.1},{'kl_w':.2})
        except ValueError:pass
        else:raise AssertionError('Resume allowed changed objective')
    # Frozen VAE is outside both prior gradient routes in actual training source.
    refine=next(n for n in train.body if isinstance(n,ast.If) and ast.unparse(n.test)=='epochs_refine > 0')
    assert any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='requires_grad' for t in n.targets) and isinstance(n.value,ast.Constant) and n.value.value is False for n in ast.walk(refine))
    assert any(isinstance(n,ast.With) and any(ast.unparse(i.context_expr)=='torch.no_grad()' for i in n.items) and any(isinstance(c,ast.Call) and ast.unparse(c.func)=='vae.encode' for c in ast.walk(n)) for n in ast.walk(refine))


def report_checks(base,rows):
    import paper
    moving=pd.DataFrame([dict(tag='LSI_Diff',stage='cotrain',epoch=50),dict(tag='Ctrl_Diff',stage='cotrain',epoch=50)])
    assert paper._main_regime(moving,'independent_pair')==[]
    mixed=pd.Series({'fid_rk4_20_randtok_cfg3_0_initoracleqT2':1.,'fid_rk4_20_randtok_cfg3_0_temp1_initgaussianTK2':2.,
                    'fid_rk4_25_randtok_cfg3_0_temp1_initgaussianT2':3.,'fid_rk4_20_randtok_cfg3_0_temp1_initgaussianT2':5.,
                    'fid_rk4_20_randtok_cfg1_5_temp1_initgaussianT2':np.nan})
    assert paper._find_metric(mixed,'fid_rk4_',20,cfg=3.)[1]==5.
    assert math.isnan(paper._find_metric(mixed,'fid_rk4_',20,cfg=1.5)[1])
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/'manifest.csv').write_bytes((base/'manifest.csv').read_bytes())
        for row in rows[:8]:
            independent=row['mode']=='independent_pair';epoch=int(row['epochs_refine'] if independent else row['epochs_joint'])
            records=[]
            for e in range(50,epoch+1,50):
                for tag in ['LSI_Diff_Refine','Ctrl_Diff_Refine'] if independent else ['LSI_Diff']:
                    r=dict(epoch=e,tag=tag,stage='refine' if independent else 'cotrain',csem_gap_score_uncond=1.,fid_vae_recon=1.)
                    for g in [float(row['cfg_scale']),1.5 if row['dataset']=='FMNIST' else 3.]:
                        for prefix in ['fid','kid','feature_sw2','sw2']:
                            r[f'{prefix}_rk4_20_randtok_cfg{str(g).replace(".","_")}_temp1_initgaussianT2']=5. if g==float(row['cfg_scale']) else 6.
                    records.append(r)
            df=pd.DataFrame(records)
            paper.validate_evaluation(row,df)
            frames=root/'results'/row['result_name']/'dataframes';frames.mkdir(parents=True)
            df.to_csv(frames/'eval_metrics_in_progress.csv',index=False)
            loss_name='loss_history_in_progress.csv' if int(row['cell_id'])==7 else 'loss_history.csv'
            pd.DataFrame([dict(epoch=1,stage='cotrain',recon=1.,score_mse_weighted=1.)]).to_csv(frames/loss_name,index=False)
            paper.write_json(root/'status'/f"cell_{int(row['cell_id']):03d}.json",dict(row,returncode=0,config_hash=paper.config_hash(row)))
            if int(row['cell_id'])!=7:df.to_csv(frames/'eval_metrics.csv',index=False)
            else:paper.write_json(root/'status'/'cell_007.json',dict(row,returncode=None))
        paper.compile_results(['--base-dir',str(root)])
        curves=pd.read_csv(root/'compiled'/'main_epoch_curves_long.csv')
        final=pd.read_csv(root/'compiled'/'main_final_metrics_by_seed.csv')
        assert set(curves.regime)=={'Co-trained CSEM','Independent CSEM','Independent Tweedie'}
        assert len(final)==10 # Last incomplete paired job is visible in curves, absent from final table.
        assert curves.paper_fid.notna().all() and set(curves.paper_steps)=={20}
        paper.plot_results(['--base-dir',str(root)])
        coverage=json.loads((root/'figures'/'figure_coverage.json').read_text())
        assert coverage['missing_regimes']==[]
        assert (root/'figures'/'main_old_paper_epoch_comparison.pdf').stat().st_size>1000
        # Empty recompilation must clear prior endpoints rather than leave stale tables.
        shutil=__import__('shutil');shutil.rmtree(root/'results');shutil.rmtree(root/'status')
        paper.compile_results(['--base-dir',str(root)])
        assert paper.read_frame(root/'compiled'/'main_final_metrics_aggregate.csv').empty


def submission_checks(base, rows):
    import paper
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/'manifest.csv').write_bytes((base/'manifest.csv').read_bytes())
        paper.write_json(root/'status'/'cell_006.json',dict(returncode=None,slurm_job_id='123'))
        paper.write_json(root/'status'/'cell_007.json',dict(returncode=None,slurm_job_id='124'))
        calls=[];next_id=2000
        original=paper.subprocess.run
        def mocked(cmd,**kw):
            nonlocal next_id
            calls.append(cmd)
            if cmd[0]=='squeue':return types.SimpleNamespace(returncode=0,stdout='123\n',stderr='')
            assert cmd[0]=='sbatch';next_id+=1
            return types.SimpleNamespace(returncode=0,stdout=str(next_id)+'\n',stderr='')
        paper.subprocess.run=mocked
        try:paper.submit_jobs(root,'independent')
        finally:paper.subprocess.run=original
        sbatch=[c for c in calls if c[0]=='sbatch']
        assert len(sbatch)==4 # One preparation + three priors, with live cell 6 skipped.
        assert any('TASK=prepare' in c for c in sbatch[0])
        assert all('--dependency=afterok:2001' in c for c in sbatch[1:])
        assert {int(c.split('CELL_ID=')[1].split(',')[0]) for cmd in sbatch[1:] for c in cmd if 'CELL_ID=' in c}=={2,3,7}
        assert paper.read_status(root,rows[6])['slurm_job_id']=='123'
        assert paper.read_status(root,rows[7])['slurm_job_id']=='2004'


def run(base=None):
    base=Path(base or __file__).resolve()
    if base.is_file():base=base.parent
    for path in base.glob('*.py'):ast.parse(path.read_text())
    tree=ast.parse((base/'core.py').read_text())
    rows=config_checks(base,tree)
    sampler_checks(tree);header_checks();data_checks();checkpoint_checks(tree);report_checks(base,rows);submission_checks(base,rows)
    print('PASS: 22 resolved configs; four paired baseline jobs; actual evaluator plan; CFG recovery; concurrent dataset preparation; corrupt-cache repair; VAE/prior checkpoints and interrupted paired evaluation; complete/partial compilation; all three paper curves and PNG/PDF rendering.')
    print('CPU contract tests passed. GPU training, torchvision downloads and numerical FID reproduction must be checked on the cluster; they were not run here.')
    return 0
if __name__=='__main__':raise SystemExit(run())
