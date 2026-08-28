#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, shlex, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/tk_anchor_sweep')
MANIFEST = 'cifar_tk_anchor_compare_v1_manifest.csv'
TARGET = 'csem_tk_anchor_compare_cifar_v1.py'
RESULTS_ROOT = 'cifar_tk_anchor_compare_v1_results'
LOG_ROOT = 'cifar_tk_anchor_compare_v1_logs'
STATUS_ROOT = 'cifar_tk_anchor_compare_v1_status'

def utc_now(): return datetime.now(timezone.utc).isoformat()

def load_row(path: Path, cell_id: int):
    with path.open(newline='') as f: rows=list(csv.DictReader(f))
    hits=[r for r in rows if int(r['cell_id'])==cell_id]
    if len(hits)!=1: raise RuntimeError(f'Expected one row for cell {cell_id}, found {len(hits)}')
    return hits[0]

def tee(cmd, log_path: Path, cwd: Path):
    log_path.parent.mkdir(parents=True,exist_ok=True); t0=time.monotonic(); started=utc_now()
    with log_path.open('w',buffering=1) as log:
        log.write('COMMAND:\n'+shlex.join(cmd)+'\n\n')
        p=subprocess.Popen(cmd,cwd=str(cwd),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        assert p.stdout is not None
        for line in p.stdout: log.write(line); sys.stdout.write(line); sys.stdout.flush()
        rc=p.wait()
    return rc,time.monotonic()-t0,started,utc_now()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cell-id',type=int,required=True); ap.add_argument('--base-dir',type=Path,default=BASE); ap.add_argument('--python',default=None); a=ap.parse_args()
    base=a.base_dir.resolve(); row=load_row(base/MANIFEST,a.cell_id); result_root=base/RESULTS_ROOT/row['result_name']
    expected=result_root/'run_terminal_kl'/'dataframes'/'eval_metrics.csv'; status=base/STATUS_ROOT/f"{row['result_name']}.json"
    if expected.is_file() and status.is_file():
        try:
            if int(json.loads(status.read_text()).get('returncode',999))==0:
                print(f"[skip] complete: {row['result_name']}"); return 0
        except Exception: pass
    if result_root.exists():
        raise SystemExit(f"[blocked] Existing/partial result directory: {result_root}\nRename/remove only this cell before retrying; no overwrite is performed.")
    py=a.python or sys.executable
    cmd=[py,'-u',str(base/TARGET), '--dataset','CIFAR','--model-preset','auto','--arms','terminal_kl',
         '--seed',row['seed'], '--latent-anchor-mode',row['anchor_mode'], '--ou-visible-anchor-w',row['ou_visible_anchor_w'],
         '--score-time-weighting','canonical','--score-head-time-weighting','unweighted-eps','--csem-w',row['csem_w'],
         '--terminal-kl-w',row['terminal_kl_w'],'--score-head-loss-w','1.0','--T-terminal',row['T_K'],'--T',row['T_full'],
         '--epochs',row['epochs'],'--refine-epochs',row['refine_epochs'],'--eval-every','0','--eval-samples',row['eval_samples'],
         '--cfg-strength',row['cfg'],'--canonical-lr-scale','1.0','--encoder-score-warmup-epochs','0','--csem-ramp-epochs','0',
         '--score-tracking-steps','0','--grad-diagnostics-every','0','--logvar-min=-30.0','--logvar-max','20.0','--no-fail-on-nonfinite',
         '--no-bespoke-fid-classifier','--no-eval-oracle-diagnostics','--no-eval-oracle-transport-decomposition',
         '--master-results-dir',str(result_root)]
    print('='*100); print('CIFAR CSEM T_K-ANCHOR COMPARISON V1'); print(f"cell={row['cell_id']} arm={row['arm_id']} seed={row['seed']}")
    print(f"anchor={row['anchor_mode']} anchor_w={row['ou_visible_anchor_w']} terminal_KL_w={row['terminal_kl_w']}")
    print(f"fixed geometry: T_K={row['T_K']} T={row['T_full']} DeltaT={row['delta_T']} wC={row['csem_w']}")
    print(f"training: {row['epochs']} joint / {row['refine_epochs']} refine | eval: CFG={row['cfg']} RK4={row['rk4_steps']} steps=NFE{row['nfe']} samples={row['eval_samples']}")
    print('fresh training; no checkpoint reload'); print('='*100)
    rc,elapsed,started,finished=tee(cmd,base/LOG_ROOT/f"{row['result_name']}.log",base)
    payload={**row,'returncode':rc,'elapsed_seconds':elapsed,'started_utc':started,'finished_utc':finished,'result_dir':str(result_root),'expected_table':str(expected),'command':cmd,'checkpoint_reload_used':False}
    status.parent.mkdir(parents=True,exist_ok=True); status.write_text(json.dumps(payload,indent=2)+'\n')
    if rc==0 and not expected.is_file(): print(f'[error] expected eval table missing: {expected}'); return 3
    return rc
if __name__=='__main__': raise SystemExit(main())
