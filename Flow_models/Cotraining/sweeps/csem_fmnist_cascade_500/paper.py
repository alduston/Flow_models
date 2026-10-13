#!/usr/bin/env python3
"""Validate, prepare, submit, resume, and report the eight-run FMNIST sweep."""
from __future__ import annotations

import argparse
import csv
import fcntl
import getpass
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

from suite import manifest, config_for, config_hash, validate_manifest


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str)+'\n')
    os.replace(tmp, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def status_path(base, row):
    return Path(base)/'status'/f'cell_{int(row["cell_id"]):03d}.json'


def status(base, row):
    path = status_path(base,row)
    return json.loads(path.read_text()) if path.exists() else {}


def read_csv(path):
    if not Path(path).is_file(): return []
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))


def validate_evaluation(row, dest):
    values = read_csv(Path(dest)/'eval.csv')
    expected = set(range(int(row['eval_every']), int(row['epochs_joint'])+1, int(row['eval_every'])))
    expected.add(int(row['epochs_joint']))
    valid = {}
    for r in values:
        if (r['fid_backend'] != row['fid_backend'] or float(r['cfg']) != float(row['cfg'])
                or int(r['rk4_steps']) != int(row['rk4_steps']) or int(r['num_eval']) != int(row['eval_samples'])):
            raise ValueError('Evaluation recipe differs from manifest')
        metrics = ['fid_generated','fid_posterior_recon']
        if int(row['blocks']) == 2: metrics += ['fid_encoded_root_generated_child']
        if not all(math.isfinite(float(r.get(k,'nan'))) for k in metrics):
            raise ValueError(f'Nonfinite/missing FID at epoch {r.get("epoch")}')
        valid[int(float(r['epoch']))] = r
    if not expected.issubset(valid):
        raise ValueError(f'Missing evaluated epochs: {sorted(expected-set(valid))}')
    losses = read_csv(Path(dest)/'loss.csv')
    if not losses or max(int(float(r['epoch'])) for r in losses) != int(row['epochs_joint']):
        raise ValueError('Training has not reached its 500-epoch endpoint')
    return valid[int(row['epochs_joint'])]


def select_cells(spec, rows):
    if spec in ('all','main','missing'): return rows
    if spec in ('baseline','paired'): return rows[:4]
    if spec in ('new','variants'): return rows[4:]
    ids = set()
    for part in spec.split(','):
        if '-' in part:
            a,b = map(int,part.split('-',1)); ids.update(range(a,b+1))
        else: ids.add(int(part))
    if ids - {int(r['cell_id']) for r in rows}: raise ValueError('Only cells 0..7 exist')
    return [r for r in rows if int(r['cell_id']) in ids]


def parse_slurm_job_id(output):
    for line in reversed(output.splitlines()):
        m = re.fullmatch(r'(?:Submitted batch job\s+)?([0-9]+)(?:;[^\s;]+)?',line.strip())
        if m: return m.group(1)
    raise RuntimeError(f'sbatch succeeded but job ID is unrecognized: {output!r}; inspect squeue before retrying')


def run_job(base, cid, *, resume=False, restart=False, data_root=None):
    import core
    rows = manifest(base); validate_manifest(rows)
    row = next(r for r in rows if int(r['cell_id']) == cid)
    sp = status_path(base,row); sp.parent.mkdir(exist_ok=True)
    with sp.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
        st = status(base,row); digest = config_hash(row,base)
        dest = Path(base)/'results'/row['result_name']
        if st.get('returncode') == 0:
            if st.get('config_hash') != digest: raise ValueError('Completed source/config changed')
            validate_evaluation(row,dest); print(f'[skip] completed cell {cid}'); return 0
        if st.get('config_hash') not in (None,digest) and not restart:
            raise ValueError('Incomplete source/config changed; use --restart for a fresh archived attempt')
        if dest.exists() and restart:
            archive = Path(base)/'failed_attempts'/f'{row["result_name"]}_{time.time_ns()}'
            archive.parent.mkdir(exist_ok=True); os.replace(dest,archive)
        ckpt = dest/'latest.pt'
        if dest.exists() and any(dest.iterdir()) and not (resume and ckpt.exists()):
            raise FileExistsError(f'Existing run needs --resume with latest.pt, or --restart: {dest}')
        payload = dict(row,returncode=None,config_hash=digest,
                       slurm_job_id=os.environ.get('SLURM_JOB_ID'),started_utc=now())
        atomic_json(sp,payload)
        try:
            if not core.torch.cuda.is_available(): raise RuntimeError('CUDA is required for full sweep training')
            args = config_for(row,base,data_root)
            if resume and ckpt.exists(): args.resume = str(ckpt)
            print('[RUN]',json.dumps(vars(args)),flush=True)
            core.train(args)
            validate_evaluation(row,dest)
            payload.update(returncode=0,resolved_config=vars(args)); rc=0
        except Exception as exc:
            payload.update(returncode=1,error=repr(exc),traceback=traceback.format_exc()); rc=1
            print(payload['traceback'],file=sys.stderr)
        payload['finished_utc'] = now(); atomic_json(sp,payload)
        return rc


def submit_jobs(base, spec, *, dry_run=False, restart=False, data_root=None,
                partition='gh', time_limit='48:00:00', account=None):
    rows = manifest(base); validate_manifest(rows)
    (Path(base)/'status').mkdir(exist_ok=True)
    (Path(base)/'slurm_logs').mkdir(exist_ok=True)
    with (Path(base)/'status'/'submit.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        active_jobs = None; pending=[]
        for row in select_cells(spec,rows):
            st = status(base,row); digest=config_hash(row,base)
            if st.get('config_hash') not in (None,digest) and (st.get('returncode') == 0 or not restart):
                raise ValueError(f'Cell {row["cell_id"]}: source/config changed')
            if st.get('returncode') == 0:
                validate_evaluation(row,Path(base)/'results'/row['result_name'])
                print(f'[skip] completed cell {row["cell_id"]}'); continue
            if st.get('returncode') is None and st.get('slurm_job_id'):
                if active_jobs is None:
                    q = subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i'],
                                       capture_output=True,text=True,check=True)
                    active_jobs=set(q.stdout.split())
                if str(st['slurm_job_id']) in active_jobs:
                    print(f'[skip] active cell {row["cell_id"]}: {st["slurm_job_id"]}'); continue
            pending.append(row)
        print(f'Selected {len(pending)} training jobs; no dependency on a preparation job.')
        for row in pending:
            exports = {'CELL_ID':row['cell_id'],'BASE_DIR':str(Path(base).resolve()),'RESTART':str(int(restart))}
            if data_root: exports['CSEM_DATA_ROOT']=str(Path(data_root).resolve())
            if any(',' in v or '\n' in v for v in exports.values()):
                raise ValueError('Slurm export values cannot contain commas or newlines')
            cmd=['sbatch','--parsable',f'--job-name=fmnist_c{row["cell_id"]}_s{row["seed"]}',
                 f'--partition={partition}',f'--time={time_limit}',f'--chdir={Path(base).resolve()}',
                 '--export=ALL,'+','.join(f'{k}={v}' for k,v in exports.items())]
            if account: cmd += [f'--account={account}']
            cmd += [str(Path(base)/'job.slurm')]
            print(shlex.join(cmd),flush=True)
            if not dry_run:
                q=subprocess.run(cmd,cwd=base,text=True,capture_output=True,check=True)
                jobid=parse_slurm_job_id(q.stdout)
                current=status(base,row)
                if not (str(current.get('slurm_job_id')) == jobid and current.get('started_utc')):
                    atomic_json(status_path(base,row),dict(row,returncode=None,slurm_job_id=jobid,
                                                         config_hash=config_hash(row,base),queued_utc=now()))
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('validate','prepare-data','submit','run','smoke','report'):
        p=commands.add_parser(name)
        p.add_argument('--base-dir',type=Path,default=Path(__file__).resolve().parent)
        if name in ('prepare-data','submit','run'):
            p.add_argument('--data-root',type=Path,default=os.environ.get('CSEM_DATA_ROOT') or None)
        if name=='prepare-data': p.add_argument('--weights',action='store_true',help='Pre-cache LPIPS and Inception weights on login node')
        if name=='submit':
            p.add_argument('--cells',default='all'); p.add_argument('--dry-run',action='store_true')
            p.add_argument('--restart-incomplete',action='store_true')
            p.add_argument('--partition',default='gh'); p.add_argument('--time',default='48:00:00')
            p.add_argument('--account',default=None)
        if name=='run':
            p.add_argument('--cell-id',type=int,choices=range(8),required=True)
            p.add_argument('--resume',action='store_true'); p.add_argument('--restart',action='store_true')
        if name=='smoke':
            p.add_argument('--cell-id',type=int,choices=range(8),default=None)
            p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
        if name=='report': p.add_argument('--strict',action='store_true')
    a=parser.parse_args(argv); base=a.base_dir.resolve()
    if a.command=='validate':
        import tests
        return tests.run(base)
    if a.command=='smoke':
        import tests
        for row in select_cells('all' if a.cell_id is None else str(a.cell_id),manifest(base)):
            tests.torch_smoke(base,int(row['cell_id']),a.device)
        return 0
    if a.command=='prepare-data':
        import core
        root=a.data_root.resolve() if a.data_root else base/'data_cache_fmnist'
        print(f'Prepared FMNIST: {core.prepare_data(root)}')
        if a.weights:
            from torchvision.models import inception_v3, Inception_V3_Weights
            inception_v3(weights=Inception_V3_Weights.DEFAULT)
            if not core.engine.ref.LPIPS_AVAILABLE: raise RuntimeError('Install lpips in hlsi')
            core.engine.ref.lpips.LPIPS(net='vgg')
            print('Pretrained Inception and LPIPS/VGG weights cached')
        return 0
    if a.command=='submit':
        return submit_jobs(base,a.cells,dry_run=a.dry_run,restart=a.restart_incomplete,
                           data_root=a.data_root,partition=a.partition,time_limit=a.time,account=a.account)
    if a.command=='run': return run_job(base,a.cell_id,resume=a.resume,restart=a.restart,data_root=a.data_root)
    return subprocess.call([sys.executable,str(base/'compile_results.py'),'--base-dir',str(base)]
                           + (['--strict'] if a.strict else []))


if __name__=='__main__':
    raise SystemExit(main())
