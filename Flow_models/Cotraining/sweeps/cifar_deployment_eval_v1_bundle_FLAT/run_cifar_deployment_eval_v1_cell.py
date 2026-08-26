#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, shlex, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/eval_sweep")
MANIFEST = "cifar_deployment_eval_v1_manifest.csv"
TARGET = "csem_deployment_eval_grid_v1.py"
SOURCE_RESULTS_ROOT = "cifar_detached_tail_downward_v1_results"
RESULTS_ROOT = "cifar_deployment_eval_v1_results"
LOG_ROOT = "cifar_deployment_eval_v1_logs"
STATUS_ROOT = "cifar_deployment_eval_v1_status"

def utc_now(): return datetime.now(timezone.utc).isoformat()
def load_cell(path, cell_id):
    with path.open(newline="") as f: rows=list(csv.DictReader(f))
    hit=[r for r in rows if int(r["cell_id"])==cell_id]
    if len(hit)!=1: raise RuntimeError(f"Expected one cell {cell_id}, found {len(hit)}")
    return hit[0]
def tee(cmd, log_path, cwd):
    log_path.parent.mkdir(parents=True,exist_ok=True)
    t0=time.monotonic(); started=utc_now()
    with log_path.open('w',buffering=1) as log:
        log.write('COMMAND:\n'+shlex.join(cmd)+'\n\n')
        p=subprocess.Popen(cmd,cwd=str(cwd),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            log.write(line); sys.stdout.write(line); sys.stdout.flush()
        rc=p.wait()
    return rc,time.monotonic()-t0,started,utc_now()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cell-id',type=int,required=True); ap.add_argument('--base-dir',type=Path,default=BASE_DIR); args=ap.parse_args()
    base=args.base_dir.resolve(); row=load_cell(base/MANIFEST,args.cell_id); target=base/TARGET
    ckpt=base/SOURCE_RESULTS_ROOT/row['source_result_name']/'training'/'run_terminal_kl'/'checkpoints'
    required=[ckpt/'vae_cotrained.pt',ckpt/'unet_lsi.pt']; missing=[str(p) for p in required if not p.is_file()]
    if missing: raise SystemExit('[missing checkpoint]\n  '+'\n  '.join(missing))
    result_root=base/RESULTS_ROOT/row['result_name']; expected=result_root/'run_terminal_kl'/'dataframes'/f"deployment_grid_ep{row['epoch_label']}.csv"; status_path=base/STATUS_ROOT/f"{row['result_name']}.json"
    if expected.is_file() and status_path.is_file():
        try:
            if int(json.loads(status_path.read_text()).get('returncode',999))==0: print(f"[skip] complete: {row['result_name']}"); return 0
        except Exception: pass
    if result_root.exists(): raise SystemExit(f"[blocked] partial result directory exists: {result_root}\nRemove/rename only this eval cell before retrying.")
    cmd=[sys.executable,'-u',str(target),'--dataset','CIFAR','--model-preset','auto','--arms','terminal_kl','--seed',row['seed'],'--score-time-weighting','canonical','--score-head-time-weighting','unweighted-eps','--csem-w',row['csem_w'],'--terminal-kl-w',row['terminal_kl_w'],'--score-head-loss-w','1.0','--T-terminal',row['T_K'],'--T',row['T_full'],'--cfg-strength','3.0','--canonical-lr-scale','1.0','--logvar-min=-30.0','--logvar-max','20.0','--no-fail-on-nonfinite','--no-bespoke-fid-classifier','--epochs','1','--refine-epochs','0','--eval-every','0','--eval-samples',row['eval_samples'],'--evaluation-only-checkpoint-dir',str(ckpt),'--oracle-eval-epoch-label',row['epoch_label'],'--deployment-only','--deployment-cfg-grid',row['cfg_grid'],'--deployment-temperature-grid',row['temperature_grid'],'--deployment-rk4-step-grid',row['rk4_steps'],'--skip-lsi-gap','--no-save-eval-sample-panels','--no-eval-oracle-diagnostics','--no-eval-oracle-transport-decomposition','--master-results-dir',str(result_root)]
    print('='*100); print('CIFAR DEPLOYMENT-ONLY EVAL V1'); print(f"cell={row['cell_id']} group={row['sweep_group']} geometry={row['geometry']} seed={row['seed']}"); print(f"T={row['T_full']} TK={row['T_K']} delta={row['delta_T']}  RK4={row['rk4_steps']} steps / {row['nfe']} NFE"); print(f"CFG={row['cfg_grid']}  temp={row['temperature_grid']}  samples/config={row['eval_samples']}"); print(f"checkpoint={ckpt}"); print('='*100)
    rc,elapsed,started,finished=tee(cmd,base/LOG_ROOT/f"{row['result_name']}.log",base)
    payload={'returncode':rc,'cell_id':int(row['cell_id']),'sweep_group':row['sweep_group'],'geometry':row['geometry'],'seed':int(row['seed']),'rk4_steps':int(row['rk4_steps']),'nfe':int(row['nfe']),'source_result_name':row['source_result_name'],'checkpoint_dir':str(ckpt),'result_dir':str(result_root),'elapsed_seconds':elapsed,'started_utc':started,'finished_utc':finished,'command':cmd}
    status_path.parent.mkdir(parents=True,exist_ok=True); status_path.write_text(json.dumps(payload,indent=2)+'\n')
    if rc==0 and not expected.is_file(): print(f"[error] process returned 0 but expected table missing: {expected}"); return 3
    return rc
if __name__=='__main__': raise SystemExit(main())
