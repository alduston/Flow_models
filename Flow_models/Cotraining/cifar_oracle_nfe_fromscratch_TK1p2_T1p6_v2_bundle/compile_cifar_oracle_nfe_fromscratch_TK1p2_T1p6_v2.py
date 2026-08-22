#!/usr/bin/env python3
"""Compile five from-scratch representations and their paired 4-point NFE curves."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np, pandas as pd

BASE_DIR=Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0")
MANIFEST="cifar_oracle_nfe_fromscratch_TK1p2_T1p6_v2_manifest.csv"
RESULTS_ROOT="cifar_oracle_nfe_fromscratch_TK1p2_T1p6_v2_results"
STATUS_ROOT="cifar_oracle_nfe_fromscratch_TK1p2_T1p6_v2_status"
OUT_ROOT="cifar_oracle_nfe_fromscratch_TK1p2_T1p6_v2_compiled"
META=["cell_id","rep_id","rep_role","csem_w","terminal_kl_w","T_K","T_full"]


def read_csv(p:Path):
    if not p.is_file(): return None
    try:return pd.read_csv(p)
    except Exception as e: print(f"[warning] failed reading {p}: {e}",file=sys.stderr); return None


def stamp(df,row):
    o=df.copy()
    for c in reversed(META):
        if c in o.columns:o=o.drop(columns=[c])
        o.insert(0,c,row[c])
    return o


def wide_table(curves:pd.DataFrame)->pd.DataFrame:
    rows=[]; metrics=["fid","kid","sw2","diversity"]
    for keys,g in curves.groupby(META,dropna=False,sort=True):
        base=dict(zip(META,keys if isinstance(keys,tuple) else (keys,)))
        direct=g[g["mode"]=="direct_q0_train"]
        for steps in sorted(pd.to_numeric(g["steps"],errors="coerce").dropna().astype(int).unique()):
            if steps<=0: continue
            sg=g[pd.to_numeric(g["steps"],errors="coerce")==steps]
            for h in ("TK","T"):
                q=sg[sg["horizon_name"].astype(str)==h]
                if q.empty: continue
                rec=dict(base); rec["steps"]=steps; rec["nfe"]=4*steps; rec["horizon_name"]=h; rec["horizon_t"]=float(q["horizon_t"].iloc[0])
                modes={
                    "oracle_qh":q[(q["score_source"]=="oracle")&(q["init_mode"]=="q_h")],
                    "learned_qh":q[(q["score_source"]=="learned")&(q["init_mode"]=="q_h")],
                    "oracle_gaussian":q[(q["score_source"]=="oracle")&(q["init_mode"]=="gaussian")],
                    "learned_gaussian":q[(q["score_source"]=="learned")&(q["init_mode"]=="gaussian")],
                }
                for label,fr in modes.items():
                    if len(fr):
                        z=fr.iloc[0]
                        for m in metrics: rec[f"{m}_{label}"]=z.get(m,np.nan)
                        for m in ["endpoint_rms_to_maxnfe","endpoint_rel_rms_to_maxnfe","endpoint_rms_learned_vs_oracle","endpoint_rms_gaussian_vs_qh"]:
                            rec[f"{m}_{label}"]=z.get(m,np.nan)
                if len(direct):
                    z=direct.iloc[0]
                    for m in metrics:rec[f"{m}_direct_q0"]=z.get(m,np.nan)
                for m in metrics:
                    oq=rec.get(f"{m}_oracle_qh",np.nan); lq=rec.get(f"{m}_learned_qh",np.nan); og=rec.get(f"{m}_oracle_gaussian",np.nan); lg=rec.get(f"{m}_learned_gaussian",np.nan)
                    rec[f"{m}_score_model_gap_qh"]=lq-oq
                    rec[f"{m}_terminal_init_gap_oracle"]=og-oq
                    rec[f"{m}_terminal_init_gap_learned"]=lg-lq
                rows.append(rec)
    return pd.DataFrame(rows)


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--base-dir",type=Path,default=BASE_DIR); ap.add_argument("--allow-incomplete",action="store_true"); args=ap.parse_args()
    base=args.base_dir.resolve(); manifest=pd.read_csv(base/MANIFEST,dtype={"cell_id":str}); manifest["cell_id"]=manifest["cell_id"].str.zfill(2)
    out=base/OUT_ROOT; out.mkdir(parents=True,exist_ok=True)
    statuses=[]; problems=[]; curves=[]; profiles=[]; train_eval=[]; train_loss=[]
    for _,row in manifest.iterrows():
        root=base/RESULTS_ROOT/row["result_name"]; train=root/"training"; mech=root/"mechanism_eval"; statusp=base/STATUS_ROOT/row["result_name"]/"overall.json"
        curvep=mech/"run_terminal_kl"/"dataframes"/"oracle_sampling_decomposition_ep600.csv"
        profilep=mech/"run_terminal_kl"/"dataframes"/"oracle_score_time_profile_ep600.csv"
        tevalp=train/"combined_dataframes"/"combined_eval_metrics.csv"; tlossp=train/"combined_dataframes"/"combined_loss_history.csv"
        st={"cell_id":row["cell_id"],"rep_id":row["rep_id"],"status_exists":statusp.is_file(),"curve_exists":curvep.is_file(),"profile_exists":profilep.is_file(),"training_eval_exists":tevalp.is_file(),"training_loss_exists":tlossp.is_file(),"returncode":None}
        if statusp.is_file():
            try:st["returncode"]=json.loads(statusp.read_text()).get("returncode")
            except Exception as e:st["status_parse_error"]=repr(e)
        statuses.append(st); ok=st["status_exists"] and st["returncode"]==0 and st["curve_exists"] and st["profile_exists"]
        if not ok:problems.append(st)
        for path,arr in [(curvep,curves),(profilep,profiles),(tevalp,train_eval),(tlossp,train_loss)]:
            d=read_csv(path)
            if d is not None:arr.append(stamp(d,row))
    pd.DataFrame(statuses).to_csv(out/"run_status.csv",index=False); pd.DataFrame(problems).to_csv(out/"missing_or_failed.csv",index=False)
    curves_df=pd.concat(curves,ignore_index=True,sort=False) if curves else pd.DataFrame(); profiles_df=pd.concat(profiles,ignore_index=True,sort=False) if profiles else pd.DataFrame(); teval_df=pd.concat(train_eval,ignore_index=True,sort=False) if train_eval else pd.DataFrame(); tloss_df=pd.concat(train_loss,ignore_index=True,sort=False) if train_loss else pd.DataFrame()
    if not curves_df.empty:
        curves_df.to_csv(out/"oracle_nfe_curve_long.csv",index=False); wide_table(curves_df).to_csv(out/"oracle_nfe_mechanism_wide.csv",index=False)
    if not profiles_df.empty: profiles_df.to_csv(out/"oracle_field_profile_all.csv",index=False)
    if not teval_df.empty: teval_df.to_csv(out/"training_eval_all.csv",index=False)
    if not tloss_df.empty: tloss_df.to_csv(out/"training_loss_all.csv",index=False)
    (out/"README.txt").write_text("""CIFAR CSEM oracle-score NFE mechanism sweep — trained from scratch\n\nPrimary outputs:\n  oracle_nfe_curve_long.csv       all paired sampling modes and NFE points\n  oracle_nfe_mechanism_wide.csv  one row per representation x NFE x horizon\n  oracle_field_profile_all.csv    time-resolved exact-field and learned-oracle diagnostics\n  training_eval_all.csv           ordinary training/evaluation trajectories\n  training_loss_all.csv           training loss/latent diagnostics\n\nDirect paired latent diagnostics:\n  endpoint_rms_to_maxnfe          finite-integration error relative to same-channel 50-step endpoint\n  endpoint_rms_learned_vs_oracle  propagated learned-score error at identical init/NFE\n  endpoint_rms_gaussian_vs_qh     propagated terminal-initialization mismatch at identical score/NFE\n\nFor exact oracle + q_h, endpoint_rms_to_maxnfe is the cleanest direct probe of optimal-field dynamical difficulty.\n""")
    print(f"Compilation output: {out}"); print(f"curve rows: {len(curves_df)}; problems: {len(problems)}")
    if problems and not args.allow_incomplete:return 2
    return 0

if __name__=="__main__": raise SystemExit(main())
