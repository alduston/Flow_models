#!/usr/bin/env python3
"""Focused CIFAR stacked CSEM audit, optimized CIFAR recipe, paired seeds."""
import argparse,csv,json,os,subprocess,sys
from pathlib import Path
import launcher_base as launch
from design import rows,select,config_for,source_hash


def manifest(base):return rows()
def config_hash(row):return config_for(row,Path(__file__).resolve().parent)['bundle_recipe_hash']

def validate_evaluation(row,values):
    target=int(row['epochs_joint'])
    data=[v for v in values if v.get('tag')=='LSI_Diff']
    final=[v for v in data if int(float(v['epoch']))==target]
    if not final:raise ValueError('Missing final scheduled endpoint')
    if row['stage']=='joint':
        seen={int(float(v['epoch'])) for v in data}
        if not set(range(50,801,50))<=seen:raise ValueError('Missing scheduled evaluations')
    import math
    for level in (1.,2.5,3.):
        for method,n in (('rk4',20),('rk4',25),('heun',20)):
            col=launch.fid_column(final[-1],level=level,steps=n,method=method)
            if not math.isfinite(float(final[-1][col])):raise ValueError('Nonfinite FID '+col)
    return final[-1]

def run_job(base,cid,resume=False,data_root=None):
    import fcntl
    row=rows()[cid];cfg=config_for(row,base,data_root);out=Path(cfg['results_dir'])
    sp=launch.status_path(base,row);sp.parent.mkdir(exist_ok=True)
    with sp.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=launch.status(base,row)
        if old.get('config_hash') not in (None,config_hash(row)):raise ValueError('Code/config changed; extract fresh bundle')
        if old.get('returncode')==0:
            file=out/('eval_metrics.csv' if row['stage']=='frozen' else 'dataframes/eval_metrics.csv')
            validate_evaluation(row,launch.read_evaluation(file));print('Already complete');return 0
        payload=dict(row,config_hash=config_hash(row),returncode=None,slurm_job_id=os.environ.get('SLURM_JOB_ID'))
        launch.atomic_json(sp,payload)
        try:
            import core
            if not core.torch.cuda.is_available():raise RuntimeError('Full run requires CUDA')
            if not core.LPIPS_AVAILABLE:raise RuntimeError('Install lpips to retain CIFAR loss')
            launch.prepare_data(base,data_root)
            if row['stage']=='frozen':
                from frozen import train
                cfg=train(row,base,data_root,resume=resume);file=out/'eval_metrics.csv'
            else:
                ckpt=out/'checkpoints/training_state_latest.pt'
                if out.exists():
                    if not resume or not ckpt.exists():raise FileExistsError('Existing run requires --resume and full checkpoint')
                    cfg['paper_resume_state']=str(ckpt)
                out.mkdir(parents=True,exist_ok=True);launch.atomic_json(out/'paper_cell_config.json',cfg)
                core.seed_everything(cfg['seed'])
                from engine import train
                loss,evaluation=train(cfg)
                from paper_io import save_csv_atomic
                save_csv_atomic(loss,out/'dataframes/loss_history.csv');save_csv_atomic(evaluation,out/'dataframes/eval_metrics.csv')
                file=out/'dataframes/eval_metrics.csv';launch.atomic_json(out/'paper_cell_config.json',cfg)
            validate_evaluation(row,launch.read_evaluation(file));payload.update(returncode=0)
        except Exception:
            import traceback
            payload.update(returncode=1,error=traceback.format_exc());print(payload['error'],file=sys.stderr)
        launch.atomic_json(sp,payload);return payload['returncode']


def submit(base,spec,dry_run=False,data_root=None,time_limit='24:00:00',account=None,partition='gh'):
    selected=select(spec);active=None
    for row in selected:
        old=launch.status(base,row)
        if old.get('config_hash') not in (None,config_hash(row)):raise ValueError('Configuration changed')
        if old.get('returncode')==0:print('[skip complete]',row['cell_id']);continue
        if old.get('slurm_job_id') and old.get('returncode') is None:
            if active is None:
                active=set(subprocess.check_output(['squeue','-h','-u',launch.getpass.getuser(),'-o','%i'],text=True).split())
            if old['slurm_job_id'] in active:print('[skip active]',row['cell_id']);continue
        dep=None
        if row['stage']=='frozen':
            source=rows()[int(row['source'])];ss=launch.status(base,source)
            if ss.get('returncode')!=0:
                if not ss.get('slurm_job_id'):
                    if dry_run:dep='<source-job-'+row['source']+'>'
                    else:raise ValueError('Submit core/source before frozen cell '+row['cell_id'])
                else:dep=ss['slurm_job_id']
        exports={'CELL_ID':row['cell_id'],'BASE_DIR':str(Path(base).resolve())}
        if data_root:exports['CSEM_DATA_ROOT']=str(Path(data_root).resolve())
        if any(',' in str(v) or '\n' in str(v) for v in exports.values()):raise ValueError('Invalid export path')
        cmd=['sbatch','--parsable','--job-name=cifar_stack_c'+row['cell_id'],f'--time={time_limit}',f'--partition={partition}',
             f'--chdir={Path(base).resolve()}','--export=ALL,'+','.join(f'{k}={v}' for k,v in exports.items())]
        if account:cmd.append('--account='+account)
        if dep:cmd.append('--dependency=afterok:'+dep)
        cmd.append(str(Path(base)/'job.slurm'));print(launch.shlex.join(cmd))
        if not dry_run:
            (Path(base)/'slurm_logs').mkdir(exist_ok=True)
            jid=launch.parse_slurm_job_id(subprocess.check_output(cmd,text=True))
            # Preserve a worker's state if the scheduler already started it.
            st=launch.status(base,row)
            if st.get('slurm_job_id')!=jid:launch.atomic_json(launch.status_path(base,row),dict(row,returncode=None,slurm_job_id=jid,config_hash=config_hash(row)))
    return 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);ss=p.add_subparsers(dest='cmd',required=True)
    for cmd in ('validate','smoke','prepare-data','submit','run','report'):
        q=ss.add_parser(cmd);q.add_argument('--base-dir',type=Path,default=Path(__file__).resolve().parent)
        if cmd in ('submit','run','prepare-data'):q.add_argument('--data-root',type=Path,default=os.environ.get('CSEM_DATA_ROOT'))
        if cmd in ('smoke','run'):q.add_argument('--cell-id',type=int,default=6)
        if cmd=='smoke':q.add_argument('--device',choices=('cpu','cuda'),default='cpu')
        if cmd=='run':q.add_argument('--resume',action='store_true')
        if cmd=='submit':
            q.add_argument('--cells',default='core');q.add_argument('--dry-run',action='store_true');q.add_argument('--time',default='24:00:00');q.add_argument('--account');q.add_argument('--partition',default='gh')
        if cmd=='report':q.add_argument('--strict',action='store_true')
    a=p.parse_args(argv);base=a.base_dir.resolve()
    if a.cmd=='validate':
        import tests;return tests.validate(base)
    if a.cmd=='smoke':
        import tests;return tests.smoke(base,a.cell_id,a.device)
    if a.cmd=='prepare-data':return 0 if launch.prepare_data(base,a.data_root) else 1
    if a.cmd=='submit':return submit(base,a.cells,a.dry_run,a.data_root,a.time,a.account,a.partition)
    if a.cmd=='run':return run_job(base,a.cell_id,a.resume,a.data_root)
    import compile_results;return compile_results.compile(base,a.strict)

if __name__=='__main__':raise SystemExit(main())
