#!/usr/bin/env python3
"""Prepare, validate, submit, resume, audit, and compile the mechanism study."""
import argparse
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import traceback
from datetime import datetime,timezone
import experiments


def now():return datetime.now(timezone.utc).isoformat()


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,default=str)+'\n');os.replace(temp,path)


def signature(row,base):
    return hashlib.sha256(json.dumps(dict(row=row,sources=experiments.digest(base)),sort_keys=True).encode()).hexdigest()


def status(base,row):
    p=base/'status'/f'cell_{row["cell_id"]:03d}.json'
    return json.loads(p.read_text()) if p.exists() else {}


def run(base,row,resume=False,restart=False,data_root=None):
    import trainer
    args=experiments.configuration(row,base,data_root)
    sp=base/'status'/f'cell_{row["cell_id"]:03d}.json';sp.parent.mkdir(exist_ok=True)
    out=Path(args.output);digest=signature(row,base)
    with sp.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        st=status(base,row)
        if st.get('signature') not in (None,digest) and not restart:raise ValueError('Source/recipe changed; archive with --restart or restore original sources')
        if (out/'complete.json').exists() and not restart:
            guard=out/'resume_recipe.json'
            expected=dict(config=vars(args),source_hashes=experiments.digest(base))
            if not guard.exists() or json.loads(guard.read_text())!=expected:raise ValueError('Completed run source/configuration changed')
            validate_complete(row,out);print('[SKIP] completed',row['name']);return 0
        if restart and out.exists():
            archive=base/'archived_attempts'/f'{row["name"]}_{time.time_ns()}'
            archive.parent.mkdir(exist_ok=True);os.replace(out,archive)
        guard=out/'resume_recipe.json'
        recipe=dict(config=vars(args),source_hashes=experiments.digest(base))
        if guard.exists():
            if json.loads(guard.read_text())!=recipe:raise ValueError('Resume configuration/source mismatch')
        elif out.exists() and any(out.iterdir()):raise ValueError('Existing output lacks its resume guard')
        else:atomic(guard,recipe)
        payload=dict(cell_id=row['cell_id'],name=row['name'],signature=digest,started=now(),
                     slurm_job_id=os.environ.get('SLURM_JOB_ID'),returncode=None)
        atomic(sp,payload)
        try:
            rc=trainer.train(args,base,resume=resume)
            if rc==0:validate_complete(row,out)
        except BaseException as exc:
            rc=1;payload.update(error=repr(exc),traceback=traceback.format_exc());print(payload['traceback'],file=sys.stderr)
        payload.update(returncode=rc,finished=now());atomic(sp,payload)
        return rc


def validate_complete(row,out):
    import csv
    cfg=row['config'];complete=json.loads((out/'complete.json').read_text())
    if complete['epoch']!=cfg['epochs']:raise ValueError('Wrong endpoint')
    if not (out/'snapshots'/f'epoch_{cfg["epochs"]:04d}.pt').exists():raise ValueError('Final model snapshot missing')
    guard=out/'resume_recipe.json'
    if guard.exists():
        saved_guard=json.loads(guard.read_text())
        if saved_guard['source_hashes']!=experiments.digest(out.parent.parent):raise ValueError('Completed run source hashes differ')
    with (out/'loss.csv').open() as f:losses=list(csv.DictReader(f))
    if {int(r['epoch']) for r in losses}!=set(range(1,cfg['epochs']+1)):raise ValueError('Missing/duplicate training epochs')
    if cfg['eval_samples']:
        with (out/'eval.csv').open() as f:ev=list(csv.DictReader(f))
        expected=set(range(cfg['eval_every'],cfg['epochs']+1,cfg['eval_every']))|{cfg['epochs']}
        if row['stage']=='frozen':expected.add(0)
        keys={(int(r['epoch']),float(r['cfg'])) for r in ev if r['init']=='gaussian' and int(r['steps'])==cfg['rk4_steps'] and int(r['num_eval'])==cfg['eval_samples']}
        if not {(e,g) for e in expected for g in (1.,3.)}<=keys:raise ValueError('Missing paired CFG1/3 primary FIDs')
    for folder,interval in [('diagnostics',cfg['diagnostic_every']),('witness',cfg['witness_every'])]:
        if interval:
            expected=set(range(interval,cfg['epochs']+1,interval))|{cfg['epochs']}
            if folder=='diagnostics' and cfg['diagnostic_at_start']:expected.add(0)
            for e in expected:
                if not (out/folder/f'epoch_{e:04d}.json').exists():raise ValueError(f'Missing {folder} epoch {e}')
    return True


def submit(base,spec,dry_run=False,partition='gh',time_limit='48:00:00',account=None,data_root=None,dependencies=False,audit_epoch=None):
    selected=experiments.selected(spec);allrows=experiments.rows()
    (base/'status').mkdir(exist_ok=True);(base/'slurm_logs').mkdir(exist_ok=True)
    with (base/'status'/'submit.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        active=set()
        if not dry_run:
            result=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i'],check=True,text=True,capture_output=True)
            active=set(result.stdout.split())
        queued={};
        for row in selected:
            st=status(base,row);out=base/'results'/row['name'];digest=signature(row,base)
            if st.get('signature') not in (None,digest):raise ValueError(f'Source/recipe changed for cell {row["cell_id"]}')
            if audit_epoch is None and (out/'complete.json').exists():validate_complete(row,out);print('[SKIP] complete',row['cell_id']);continue
            if audit_epoch is None and str(st.get('slurm_job_id')) in active:
                print('[SKIP] active',row['cell_id']);queued[row['cell_id']]=str(st['slurm_job_id']);continue
            dep=None
            if row['source'] is not None and audit_epoch is None:
                parent=allrows[row['source']];parent_out=base/'results'/parent['name']
                if not (parent_out/'complete.json').exists():
                    ps=status(base,parent);dep=queued.get(parent['cell_id'])
                    if dep is None and str(ps.get('slurm_job_id')) in active:dep=str(ps['slurm_job_id'])
                    if dep is None and dry_run and dependencies:dep=f'<parent-cell-{parent["cell_id"]}>'
                    if not dependencies or dep is None:
                        print('[WAIT] source incomplete; submit parent/core first, or use --dependencies with queued parents:',row['cell_id']);continue
            exports={'CELL_ID':str(row['cell_id']),'BASE_DIR':str(base)}
            if data_root:exports['CSEM_DATA_ROOT']=str(Path(data_root).resolve())
            if audit_epoch is not None:exports['AUDIT_EPOCH']=str(audit_epoch)
            if any(',' in v or '\n' in v for v in exports.values()):raise ValueError('Slurm export values cannot contain comma/newline')
            cmd=['sbatch','--parsable',f'--job-name=stack_c{row["cell_id"]}',f'--partition={partition}',f'--time={time_limit}',f'--chdir={base}',
                 '--export=ALL,'+','.join(f'{k}={v}' for k,v in exports.items())]
            if account:cmd.append(f'--account={account}')
            if dep:cmd.append(f'--dependency=afterok:{dep}')
            cmd.append(str(base/'job.slurm'));print(shlex.join(cmd),flush=True)
            if not dry_run:
                q=subprocess.run(cmd,check=True,text=True,capture_output=True,cwd=base)
                match=re.search(r'^(\d+)(?:;[^\s]+)?$',q.stdout.strip(),re.M)
                if not match:raise RuntimeError('sbatch succeeded but returned an unrecognized job ID; inspect squeue before retry')
                jid=match.group(1);queued[row['cell_id']]=jid
                if audit_epoch is None:
                    current=status(base,row)
                    if not (current.get('slurm_job_id')==jid and current.get('started')):
                        atomic(base/'status'/f'cell_{row["cell_id"]:03d}.json',dict(signature=digest,slurm_job_id=jid,queued=now(),returncode=None))
    return 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for command in ('plan','validate','prepare-data','submit','run','smoke','audit','report'):
        q=sub.add_parser(command);q.add_argument('--base-dir',type=Path,default=Path(__file__).resolve().parent)
        if command in ('prepare-data','run','submit','audit'):q.add_argument('--data-root',type=Path,default=os.environ.get('CSEM_DATA_ROOT'))
        if command in ('run','smoke','audit'):q.add_argument('--cell-id',type=int,default=4 if command=='smoke' else None,required=command in ('run','audit'))
        if command in ('run','audit'):q.add_argument('--resume',action='store_true')
        if command=='run':q.add_argument('--restart',action='store_true')
        if command=='smoke':q.add_argument('--device',default='cpu',choices=('cpu','cuda'))
        if command=='validate':q.add_argument('--models',action='store_true')
        if command=='prepare-data':q.add_argument('--weights',action='store_true')
        if command=='submit':
            q.add_argument('--cells',default='core');q.add_argument('--dry-run',action='store_true');q.add_argument('--dependencies',action='store_true')
            q.add_argument('--partition',default='gh');q.add_argument('--time',default='48:00:00');q.add_argument('--account')
            q.add_argument('--audit-epoch',type=int)
        if command=='audit':
            q.add_argument('--epoch',type=int,default=500);q.add_argument('--banks',default='64,256');q.add_argument('--particles',type=int,default=16)
            q.add_argument('--queries',type=int,default=16);q.add_argument('--times',default=None);q.add_argument('--device',default='cuda',choices=('cpu','cuda'))
        if command=='report':q.add_argument('--strict',action='store_true')
    a=p.parse_args(argv);base=a.base_dir.resolve();rr=experiments.rows()
    if a.command=='plan':
        for r in rr:print(f'{r["cell_id"]:2d} {r["group"]:10s} {r["name"]} source={r["source"]}')
        return 0
    if a.command in ('validate','smoke'):
        import tests
        return tests.run(base,models=a.models) if a.command=='validate' else tests.smoke(base,rr[a.cell_id],a.device)
    if a.command=='prepare-data':
        import trainer
        root=a.data_root or base/'data_cache_fmnist';print(trainer.prepare_data(root))
        if a.weights:
            from torchvision.models import inception_v3,Inception_V3_Weights
            inception_v3(weights=Inception_V3_Weights.DEFAULT)
            if not trainer.old.ref.LPIPS_AVAILABLE:raise RuntimeError('Install lpips in hlsi')
            trainer.old.ref.lpips.LPIPS(net='vgg')
        return 0
    if a.command=='submit':return submit(base,a.cells,a.dry_run,a.partition,a.time,a.account,a.data_root,a.dependencies,a.audit_epoch)
    if a.command=='run':return run(base,rr[a.cell_id],a.resume,a.restart,a.data_root)
    if a.command=='audit':
        import reference
        args=experiments.configuration(rr[a.cell_id],base,a.data_root,a.device)
        path=Path(args.output)/'snapshots'/f'epoch_{a.epoch:04d}.pt'
        if not path.exists():raise FileNotFoundError(path)
        lock_dir=Path(args.output)/'reference_audit';lock_dir.mkdir(exist_ok=True)
        with (lock_dir/'audit.lock').open('a') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            return 0 if reference.audit(args,base,path,banks=tuple(map(int,a.banks.split(','))),particles=a.particles,queries=a.queries,
                                       times=list(map(float,a.times.split(','))) if a.times else None) is not None else 1
    import compile_results
    return compile_results.run(base,a.strict)


if __name__=='__main__':raise SystemExit(main())
