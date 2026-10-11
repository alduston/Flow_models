#!/usr/bin/env python3
"""Matched LSGM/CSEM CIFAR-10 sweep with locked phase gating and resume."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

VERSION = 'lsgm-csem-cifar-v1'


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(tmp, path)


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def manifest(base):
    with (Path(base)/'manifest.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    ids = [r['cell_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate cell IDs')
    for r in rows:
        g = int(r['groups'])
        if g not in (1,2,4) or g*int(r['latent_per_group']) != 180:
            raise ValueError('Expected 180 total channels in 1/2/4 groups')
        if r['phase'] not in ('A','B','C') or (r['vae_cell'] and r['vae_cell'] not in ids):
            raise ValueError('Invalid phase/dependency')
    return rows


def config_hash(row):
    data = {k:v for k,v in row.items() if k not in ('notes','result_name')}
    return hashlib.sha256(json.dumps({'version': VERSION, 'science': data}, sort_keys=True).encode()).hexdigest()


def command_for(row, base, data_root=None, *, python=None):
    """Pure command construction; no imports of torch and no side effects."""
    base = Path(base).resolve()
    cache = Path(data_root or base/'data_cache').resolve()
    results = base/'results'
    prefix = [python or sys.executable, str(base/'lsgm_runner.py')]
    phase = row['phase']
    common = ['--data', str(cache/'CIFAR'), '--root', str(results), '--save', row['result_name'],
              '--num_process_per_node', '1', '--seed', row['seed']]
    if phase == 'A':
        return prefix+['train_vae.py']+common+[
            '--dataset','cifar10','--num_channels_enc','256','--num_channels_dec','256',
            '--num_postprocess_cells','3','--num_preprocess_cells','3','--num_latent_scales','1',
            '--num_cell_per_cond_enc','3','--num_cell_per_cond_dec','3','--num_preprocess_blocks','1',
            '--num_postprocess_blocks','1','--num_latent_per_group',row['latent_per_group'],
            '--num_groups_per_scale',row['groups'],'--epochs',row['epochs'],'--batch_size',row['batch_size'],
            '--weight_decay_norm','1e-2','--num_nf',row['num_nf'],'--kl_anneal_portion','1.0',
            '--kl_max_coeff','1.0','--channel_mult','1','2','--arch_instance','res_bnswish','--use_se']
    if phase == 'C':
        return prefix+['evaluate_vada.py']+common+[
            '--eval_mode','evaluate','--checkpoint',str(results/row['vae_cell']/'checkpoint.pt'),
            '--fid_dir',str(base/'fid_stats'),'--fid_ode_eval','--nll_ode_eval','--num_fid_samples','50000',
            '--ode_eps','1e-6','--ode_solver_tol','1e-5','--batch_size',row['batch_size']]
    cmd = prefix+['train_vada.py']+common+[
        '--fid_dir',str(base/'fid_stats'),'--vae_checkpoint',str(results/row['vae_cell']/'checkpoint.pt'),
        '--train_vae','--apply_sqrt2_res','--fir','--cont_kl_anneal','--dae_arch','ncsnpp',
        '--embedding_scale','1000','--dataset','cifar10','--learning_rate_dae','1e-4',
        '--learning_rate_min_dae','1e-4','--epochs',row['epochs'],'--dropout','0.2',
        '--batch_size',row['batch_size'],'--num_channels_dae','512','--num_scales_dae','3',
        '--num_cell_per_scale_dae','2','--sde_type','vpsde','--beta_start','0.1','--beta_end','20.0',
        '--sigma2_0','0.0','--weight_decay_norm_dae','1e-2','--weight_decay_norm_vae','1e-2',
        '--time_eps','0.01','--train_ode_eps','1e-6','--eval_ode_eps','1e-6',
        '--train_ode_solver_tol','1e-5','--eval_ode_solver_tol','1e-5',
        '--iw_sample_p','drop_all_iw','--iw_sample_q','ll_iw','--use_se','--skip_final_eval',
        '--eval_csv',str(results/row['result_name']/'eval_in_progress.csv'),
        '--diagnostics_csv',str(results/row['result_name']/'diagnostics.csv')]
    if row['prior_factorization'] == 'cascade':
        cmd += ['--prior_factorization','cascade','--dae_channels_per_head',row['dae_channels_per_head']]
    if row['objective'] == 'csem':
        cmd += ['--csem','--rep_horizon',row['rep_horizon'],'--csem_lambda_K',row['csem_lambda_K']]
        if row['csem_band'] == '0': cmd += ['--no_csem_band']
        if row['csem_terminal'] == '0': cmd += ['--no_csem_terminal']
    return cmd


def select(rows, cells, phase=None):
    families = {'main':{'main'}, 'csem':{'main'}, 'baselines':{'main'},
                'tier1':{'main','calibration'}, 'calibration':{'calibration'},
                'cascade-controls':{'cascade_control'}, 'ablations':{'ablation'},
                'all':{'main','calibration','cascade_control','ablation'}}
    if cells in families:
        chosen = [r for r in rows if r['family'] in families[cells]]
        if cells in ('csem','baselines'):
            obj = 'csem' if cells == 'csem' else 'lsgm'
            chosen = [r for r in chosen if r['phase']=='A' or r['objective']==obj]
    else:
        wanted = set(cells.split(','))
        known = {r['cell_id'] for r in rows}
        if wanted-known: raise ValueError('Unknown cells: '+','.join(sorted(wanted-known)))
        chosen = [r for r in rows if r['cell_id'] in wanted]
    return [r for r in chosen if phase is None or r['phase']==phase]


def status_path(base, row):
    return Path(base)/'status'/(row['cell_id']+'.json')


def status(base, row):
    return read_json(status_path(base,row))


def prerequisite(base, row, rows):
    if not row['vae_cell']: return None
    dep = next(r for r in rows if r['cell_id']==row['vae_cell'])
    state = status(base,dep)
    if state.get('state') != 'completed' or state.get('returncode') != 0 or state.get('config_hash') != config_hash(dep):
        return f"{row['cell_id']} waits for successful {dep['cell_id']}"
    if not (Path(base)/'results'/dep['result_name']/'checkpoint.pt').exists():
        return f"{dep['cell_id']} checkpoint is missing"
    return None


def parse_slurm_job_id(stdout):
    for line in reversed(stdout.strip().splitlines()):
        m = re.fullmatch(r'(\d+)(?:;[\w.-]+)?', line.strip())
        if m: return m.group(1)
        m = re.fullmatch(r'Submitted batch job (\d+)', line.strip())
        if m: return m.group(1)
    raise ValueError('sbatch did not return a parseable job ID: '+repr(stdout))


def job_active(job_id):
    if not job_id or not shutil.which('squeue'): return False
    p = subprocess.run(['squeue','-h','-j',str(job_id),'-o','%i'], capture_output=True,text=True)
    if p.returncode != 0: raise RuntimeError('squeue failed: '+p.stderr.strip())
    return any(line.strip()==str(job_id) for line in p.stdout.splitlines())


def check_config(base,row):
    state = status(base,row)
    saved = read_json(Path(base)/'results'/row['result_name']/'config.json')
    for old in (state,saved):
        if old and old.get('config_hash') not in (None,config_hash(row)):
            raise ValueError('Configuration mismatch for '+row['cell_id']+'; use an explicitly archived restart')


def validate_outputs(base,row):
    output = Path(base)/'results'/row['result_name']
    marker=read_json(output/'execution_complete.json')
    if marker.get('returncode') != 0:
        raise ValueError('Missing successful entrypoint marker')
    if row['phase'] == 'C':
        data=read_json(output/'final_metrics.json')
        for k in ('ode_fid_50000','ode_nll_bpd','sampling_nfe_total','sampling_seconds'):
            if not math.isfinite(float(data.get(k,float('nan')))): raise ValueError('Missing/nonfinite '+k)
        if data.get('num_fid_samples') != 50000: raise ValueError('Final FID must use 50,000 samples')
        return data
    if marker.get('requested_epochs') != int(row['epochs']): raise ValueError('Epoch budget mismatch')
    import torch
    cp=torch.load(output/'checkpoint.pt',map_location='cpu',weights_only=False)
    ep=int(cp['epoch'])
    if row['phase']=='A':
        if not 0 < ep <= int(row['epochs']): raise ValueError('Invalid best-validation VAE checkpoint epoch')
    else:
        if ep != int(row['epochs']): raise ValueError('Stage B has not reached its final epoch')
        with (output/'eval_in_progress.csv').open(newline='') as f: evaluations=list(csv.DictReader(f))
        if not evaluations or int(evaluations[-1]['epoch'])!=ep: raise ValueError('Missing final evaluation row')
        for k in ('ode_fid_2500','ode_nll_bpd'):
            if not math.isfinite(float(evaluations[-1][k])): raise ValueError('Nonfinite '+k)
    return {'checkpoint_epoch':ep}


def archive_incomplete(base,row):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+str(time.time_ns())
    dest=Path(base)/'failed_attempts'/(row['cell_id']+'_'+stamp); dest.mkdir(parents=True)
    output=Path(base)/'results'/row['result_name']
    if output.exists(): shutil.move(str(output),str(dest/'results'))
    sp=status_path(base,row)
    if sp.exists(): shutil.move(str(sp),str(dest/'status.json'))


def prepare_data(base,data_root=None):
    from torchvision.datasets import CIFAR10
    from paper_io import prepare_dataset
    cache=Path(data_root or Path(base)/'data_cache').resolve()
    print('Prepared CIFAR-10:',prepare_dataset(CIFAR10,cache,'CIFAR'),flush=True)


def prepare_fid(base,data_root=None,weights_only=False):
    from paper_io import dataset_lock
    base=Path(base).resolve()
    prepare_data(base,data_root)
    dest=base/'fid_stats'; dest.mkdir(exist_ok=True)
    with dataset_lock(dest,'preparation'):
        import torch
        url='https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'
        weights=dest/url.rsplit('/',1)[-1]
        if not weights.exists():
            tmp=weights.with_name(weights.name+'.download')
            torch.hub.download_url_to_file(url,str(tmp),hash_prefix='6726825d')
            torch.load(tmp,map_location='cpu',weights_only=True)
            os.replace(tmp,weights)
        if weights_only: return
        if not torch.cuda.is_available(): raise RuntimeError('FID statistics require a CUDA node; weights are downloaded')
        cmd=[sys.executable,str(base/'lsgm_runner.py'),'scripts/precompute_fid_statistics.py',
             '--dataset','cifar10','--data',str(Path(data_root or base/'data_cache').resolve()/'CIFAR'),
             '--fid_dir',str(dest),'--batch_size','256','--max_samples','50000']
        subprocess.run(cmd,cwd=base/'lsgm',check=True)


def run(base,row,rows,*,data_root=None,resume=True,restart=False):
    base=Path(base).resolve(); sp=status_path(base,row); sp.parent.mkdir(exist_ok=True)
    with sp.with_suffix('.lock').open('a') as lock:
        try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Cell is already running: '+row['cell_id'])
        old=status(base,row)
        if old.get('state')=='completed':
            check_config(base,row); print('Already completed:',row['cell_id']); return
        reason=prerequisite(base,row,rows)
        if reason: raise RuntimeError(reason)
        if restart: archive_incomplete(base,row)
        check_config(base,row)
        output=base/'results'/row['result_name']
        output.mkdir(parents=True,exist_ok=True)
        from paper_io import load_prepared_pair
        from torchvision.datasets import CIFAR10
        load_prepared_pair(CIFAR10,Path(data_root or base/'data_cache'),'CIFAR')
        if row['phase'] != 'A':
            for file in ('cifar10.npz','pt_inception-2015-12-05-6726825d.pth'):
                if not (base/'fid_stats'/file).exists(): raise RuntimeError('Run paper.py prepare-fid first: '+file+' missing')
        cmd=command_for(row,base,data_root)
        if row['phase']!='C' and (output/'checkpoint.pt').exists():
            if not resume: raise RuntimeError('Checkpoint exists; select --resume or --restart')
            cmd+=['--cont_training']
        cfg={'config_hash':config_hash(row),'row':row,'command':cmd}
        atomic_json(output/'config.json',cfg)
        state={**cfg,'state':'running','returncode':None,'started':datetime.now(timezone.utc).isoformat(),
               'job_id':os.environ.get('SLURM_JOB_ID'), 'pid':os.getpid()}
        atomic_json(sp,state)
        print(shlex.join(cmd),flush=True)
        try:
            with (output/'console.log').open('a') as log:
                proc=subprocess.Popen(cmd,cwd=base/'lsgm',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in proc.stdout:
                    print(line,end='',flush=True); log.write(line); log.flush()
                rc=proc.wait()
            if rc: raise subprocess.CalledProcessError(rc,cmd)
            info=validate_outputs(base,row)
            state.update(state='completed',returncode=0,output_validation=info)
        except BaseException as exc:
            state.update(state='failed',returncode=getattr(exc,'returncode',1),error=repr(exc))
            raise
        finally:
            state['finished']=datetime.now(timezone.utc).isoformat()
            atomic_json(sp,state)


def submit(base,chosen,rows,*,dry_run=False,restart=False,data_root=None):
    base=Path(base).resolve()
    (base/'slurm_logs').mkdir(exist_ok=True); (base/'status').mkdir(exist_ok=True)
    with (base/'status'/'submit.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        for row in chosen:
            s=status(base,row)
            if not (restart and s.get('state') != 'completed'):
                check_config(base,row)
            if s.get('state')=='completed': print('[done]',row['cell_id']); continue
            if job_active(s.get('job_id')): print('[active]',row['cell_id'],s['job_id']); continue
            reason=prerequisite(base,row,rows)
            if reason: print('[gated]',reason); continue
            if ',' in str(base) or (data_root and ',' in str(data_root)):
                raise ValueError('SLURM export paths cannot contain commas')
            exports=f'ALL,CELL_ID={row["cell_id"]},BASE_DIR={base},RESTART={int(restart)}'
            if data_root: exports+=',CSEM_DATA_ROOT='+str(Path(data_root).resolve())
            cmd=['sbatch','--parsable','--job-name=lsgm_'+row['cell_id'],
                 '--export='+exports,str(base/'job.slurm')]
            print(shlex.join(cmd),flush=True)
            if dry_run: continue
            p=subprocess.run(cmd,cwd=base,capture_output=True,text=True)
            if p.returncode: raise RuntimeError('sbatch failed: '+p.stderr)
            jid=parse_slurm_job_id(p.stdout)
            atomic_json(status_path(base,row),{'state':'submitted','job_id':jid,'config_hash':config_hash(row),
                        'returncode':None,'submitted':datetime.now(timezone.utc).isoformat()})


def report(base,rows):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from paper_io import save_csv_atomic
    base=Path(base); dest=base/'compiled'; dest.mkdir(exist_ok=True)
    latest,curves=[],[]
    for row in rows:
        output=base/'results'/row['result_name']
        rec={k:row[k] for k in ('cell_id','phase','groups','objective','prior_factorization')}
        rec['state']=status(base,row).get('state','not_started')
        metrics=read_json(output/'final_metrics.json')
        path=output/'eval_in_progress.csv'
        if path.exists():
            with path.open(newline='') as f: history=list(csv.DictReader(f))
            for v in history: curves.append({**rec,**v})
            if history: metrics.update(history[-1])
        rec.update(metrics); latest.append(rec)
    save_csv_atomic(pd.DataFrame(latest),dest/'latest.csv')
    frame=pd.DataFrame(curves)
    save_csv_atomic(frame,dest/'curves.csv')
    if not frame.empty:
        fig,ax=plt.subplots(figsize=(9,5))
        for cid,g in frame.groupby('cell_id',sort=False):
            ax.plot(pd.to_numeric(g['epoch']),pd.to_numeric(g['ode_fid_2500']),label=cid)
        ax.set(xlabel='Stage-2 epoch',ylabel='LSGM ODE FID (2,500 samples)')
        ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
        for ext in ('png','pdf'): fig.savefig(dest/('fid_curves.'+ext))
        plt.close(fig)
    print('Report:',dest/'latest.csv')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    for action in ('validate','prepare-data','prepare-fid','submit','run','smoke','report','commands'):
        p=sub.add_parser(action)
        p.add_argument('--base-dir',default=str(Path(__file__).resolve().parent))
        if action in ('prepare-data','prepare-fid','submit','run','commands'): p.add_argument('--data-root')
        if action in ('submit','commands'):
            p.add_argument('--phase',choices=['A','B','C'],default='A')
            p.add_argument('--cells',default='main')
        if action=='submit':
            p.add_argument('--dry-run',action='store_true')
            p.add_argument('--restart-incomplete',action='store_true')
        if action in ('run','smoke'): p.add_argument('--cell-id',required=True)
        if action=='run':
            p.add_argument('--resume',action=argparse.BooleanOptionalAction,default=True)
            p.add_argument('--restart',action='store_true')
        if action=='smoke':
            p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
            p.add_argument('--full-size',action='store_true',help='Use production network widths (batch 2)')
        if action=='validate': p.add_argument('--torch',action='store_true',help='Also run numeric/model CPU checks')
        if action=='prepare-fid': p.add_argument('--weights-only',action='store_true')
    args=parser.parse_args(argv); base=Path(args.base_dir).resolve(); rows=manifest(base)
    if args.action=='validate':
        import tests
        tests.offline_checks(base)
        if args.torch: tests.torch_checks(base)
    elif args.action=='prepare-data': prepare_data(base,args.data_root)
    elif args.action=='prepare-fid': prepare_fid(base,args.data_root,args.weights_only)
    elif args.action in ('submit','commands'):
        chosen=select(rows,args.cells,args.phase)
        if args.action=='submit': submit(base,chosen,rows,dry_run=args.dry_run,restart=args.restart_incomplete,data_root=args.data_root)
        else:
            for row in chosen: print(row['cell_id']+': '+shlex.join(command_for(row,base,args.data_root)))
    elif args.action in ('run','smoke'):
        match=[r for r in rows if r['cell_id']==args.cell_id]
        if not match: parser.error('Unknown cell ID: '+args.cell_id)
        row=match[0]
        if args.action=='run': run(base,row,rows,data_root=args.data_root,resume=args.resume,restart=args.restart)
        else:
            import tests
            tests.torch_smoke(base,row,args.device,args.full_size)
    else: report(base,rows)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
