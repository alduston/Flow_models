#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Dict

import core


def _preset(dataset: str):
    name, p = core.resolve_model_preset(dataset, "auto")
    return name, p


def build_base_config(*, dataset: str, seed: int, epochs: int, refine_epochs: int,
                      T_K: float, T_full: float, csem_w: float, terminal_kl_w: float,
                      eval_every: int, eval_samples: int, cfg_scale: float,
                      results_dir: str, score_time_weighting: str = "canonical",
                      score_head_time_weighting: str = "unweighted-eps",
                      rk4_steps: int = 25, paper_solver_suite: bool = False,
                      paper_solver_steps: int = 20,
                      latent_anchor_mode: str = "none",
                      lr_score_head_override: float | None = None,
                      lr_refine_override: float | None = None) -> Dict[str, Any]:
    preset_name, preset = _preset(dataset)
    t_min = float(preset.get("t_min", 3e-5))
    if T_K < 0 or T_full < T_K:
        raise ValueError(f"Require 0 <= T_K <= T; got {T_K}, {T_full}")
    if 0 < T_K <= t_min:
        raise ValueError(f"Nonzero T_K must exceed t_min={t_min}")

    base_lr_vae = float(preset["lr_vae"])
    base_lr_ldm = float(preset["lr_ldm"])
    lr_refine = float(lr_refine_override) if lr_refine_override is not None else float(preset["lr_refine"])
    lr_schedule_epochs = int(preset["lr_schedule_epochs"]) if preset.get("lr_schedule_epochs") is not None else int(epochs)
    use_bespoke = core.resolve_bespoke_fid_classifier(dataset, None)

    cfg = {
        "dataset": dataset,
        "batch_size": int(preset["batch_size"]),
        "num_workers": 2,
        "latent_channels": int(preset["latent_channels"]),
        "cond_emb_dim": int(preset["cond_emb_dim"]),
        "dit_patch_size": int(preset["dit_patch_size"]),
        "dit_hidden_dim": int(preset["dit_hidden_dim"]),
        "dit_depth": int(preset["dit_depth"]),
        "dit_num_heads": int(preset["dit_num_heads"]),
        "dit_mlp_ratio": float(preset["dit_mlp_ratio"]),
        "dit_dropout": float(preset["dit_dropout"]),
        "adam_beta2": 0.95,
        "vae_grad_clip": 1.0,
        "score_grad_clip": 1.0,
        "disc_grad_clip": 1.0,
        "encoder_score_warmup_epochs": 0,
        "csem_ramp_epochs": 0,
        "score_tracking_steps": 0,
        "score_tracking_every": 1,
        "grad_diagnostics_every": 0,
        "time_diagnostic_bins": 4,
        # Match the certified fresh-training audit: do not abort merely because a
        # diagnostic scalar goes nonfinite; the training code still logs it.
        "fail_on_nonfinite": False,
        "cosine_w": 0.0,
        "aux_head_w": 0.0025,
        "score_w_vae": float(csem_w),
        "score_head_loss_w": 1.0,
        "aux_d": 0,
        "base_ch": int(preset["base_ch"]),
        "num_res_blocks": int(preset["num_res_blocks"]),
        "decoder_attn_half": bool(preset["decoder_attn_half"]),
        "latent_proj_depth": int(preset["latent_proj_depth"]),
        "encoder_attn_half": bool(preset["encoder_attn_half"]),
        "decoder_extra_block": bool(preset["decoder_extra_block"]),
        "conv3x3_proj": bool(preset["conv3x3_proj"]),
        "use_tanh_out": bool(preset["use_tanh_out"]),
        "clamp_logvar": bool(preset["clamp_logvar"]),
        "attn_zero_init": bool(preset["attn_zero_init"]),
        "logvar_min": -30.0,
        "logvar_max": 20.0,
        "base_lr_vae": base_lr_vae,
        "base_lr_ldm": base_lr_ldm,
        "canonical_lr_scale": 1.0,
        "lr_vae": base_lr_vae,
        "lr_ldm": base_lr_ldm,
        "lr_score_head": float(lr_score_head_override) if lr_score_head_override is not None else base_lr_ldm,
        "kl_w": float(terminal_kl_w),
        "perc_w": 0.85,
        "gan_w": 0.0025,
        "disc_start_epoch": 25,
        "disc_ndf": 64,
        "disc_n_layers": 2,
        "lr_disc": 1.0e-4,
        "time_schedule": str(preset.get("time_schedule", "log_t")),
        "use_ddim_times": bool(preset.get("use_ddim_times", True)),
        "t_min": t_min,
        "T_terminal": float(T_K),
        "t_max": float(T_full),
        "eval_tk_vs_t_comparison": True,
        "deployment_only": False,
        "deployment_cfg_grid": [float(cfg_scale)],
        "deployment_temperature_grid": [1.0],
        "deployment_rk4_step_grid": [int(rk4_steps)],
        "skip_lsi_gap": False,
        "save_eval_sample_panels": True,
        "eval_rk4_steps_tk": int(rk4_steps),
        "eval_rk4_steps_t": int(rk4_steps),
        "num_train_timesteps": int(preset.get("num_train_timesteps", 1000)),
        "score_time_weighting": score_time_weighting,
        "score_head_time_weighting": score_head_time_weighting,
        "train_on_mu": False,
        "cosine_t_min": 2e-4,
        "cosine_t_max": 0.9999,
        "cosine_s": 0.008,
        "cfg_label_dropout": 0.1,
        "cfg_eval_scale": float(cfg_scale),
        "eval_class_labels": [],
        "use_fixed_eval_banks": True,
        "sw2_n_projections": 1000,
        "ema_decay": 0.9997,
        "eval_max_samples": int(eval_samples),
        "eval_lsi_gap_samples": min(2500, int(eval_samples)),
        "eval_lsi_gap_time_points": 50,
        "eval_oracle": False,
        "eval_sampling_init": "gaussian",
        "eval_oracle_diagnostics": False,
        "eval_oracle_full_train_reference": True,
        "oracle_profile_query_samples": 256,
        "oracle_profile_time_points": 24,
        "oracle_profile_batch_size": 16,
        "oracle_reference_batch_size": 2048,
        "oracle_sampling_samples": 512,
        "oracle_sampling_batch_size": 32,
        "oracle_sampling_steps": 25,
        "oracle_sampling_step_grid": [20, 40, 100],
        "oracle_sampling_method": "rk4_ode",
        "eval_oracle_transport_decomposition": False,
        "eval_oracle_standard_samplers": False,
        "kid_num_subsets": 100,
        "kid_subset_size": min(1000, int(eval_samples)),
        "use_bespoke_fid_classifier": use_bespoke,
        "generate_visualizations": False,
        "mechanism_diagnostics": False,
        "seed": int(seed),
        "load_from_checkpoint": False,
        "ckpt_load_dir": None,
        "evaluation_only": False,
        "oracle_nfe_eval_only": False,
        "oracle_eval_epoch_label": 0,
        "ckpt_dir": str(Path(results_dir) / "checkpoints"),
        "master_results_dir": str(results_dir),
        "overwrite_results": False,
        "model_preset": preset_name,
        "epochs_vae": int(epochs),
        "epochs_refine": int(refine_epochs),
        "lr_schedule_epochs": lr_schedule_epochs,
        "lr_refine": lr_refine,
        "factored_head": True,
        "freeze_score_in_cotrain": False,
        "cotrain_head": "lsi",
        "use_latent_norm": False,
        "latent_anchor_mode": str(latent_anchor_mode),
        "latent_anchor_time": float(T_K),
        "use_cond_encoder": False,
        "kl_reg_type": "terminal",
        "stiff_w": 0.0,
        "score_w": 1.0,
        "train_tracking_head": False,
        "time_cond_decoder": True,
        "dec_time_emb_dim": 128,
        "decode_time": preset.get("decode_time", None),
        "eval_freq_cotrain": int(eval_every) if int(eval_every) > 0 else int(epochs),
        "eval_freq_refine": int(eval_every) if int(eval_every) > 0 else max(1, int(refine_epochs)),
        "results_dir": str(results_dir),
        "comparison_arm": "paper",
        "paper_solver_suite": bool(paper_solver_suite),
        "paper_solver_steps": int(paper_solver_steps),
    }
    return cfg


def configure_mode(cfg: Dict[str, Any], mode: str, *, beta0: float = 0.07) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    mode = str(mode)
    cfg["paper_mode"] = mode
    cfg["comparison_arm"] = mode

    if mode in {"cotrained_csem", "anchor_terminal", "sensitivity", "raw_terminal_control", "fmnist_aug18_recovered", "fmnist_aug13_gaussian"}:
        cfg.update({
            "use_latent_norm": False,
            "kl_reg_type": "terminal",
            "freeze_score_in_cotrain": False,
            "cotrain_head": "lsi",
            "train_tracking_head": False,
        })
    elif mode == "cotrained_csem_norm":
        # Historical optimized FMNIST treatment used by the old CSEM-main lineage:
        # hard per-example one-group GroupNorm on the encoder mean is the scale
        # gauge; no terminal KL is active.  This is intentionally distinct from
        # the modern CIFAR OU-partial+K_TK recipe.
        cfg.update({
            "use_latent_norm": True,
            "latent_anchor_mode": "none",
            "kl_reg_type": "normal",
            "kl_w": 0.0,
            "freeze_score_in_cotrain": False,
            "cotrain_head": "lsi",
            "train_tracking_head": False,
        })
    elif mode == "anchor_norm":
        cfg.update({
            "use_latent_norm": True,
            "latent_anchor_mode": "none",
            "kl_reg_type": "terminal",
            "kl_w": 0.0,
            "freeze_score_in_cotrain": False,
            "cotrain_head": "lsi",
            "train_tracking_head": False,
        })
    elif mode == "anchor_none":
        cfg.update({
            "use_latent_norm": False,
            "latent_anchor_mode": "none",
            "kl_reg_type": "terminal",
            "kl_w": 0.0,
            "freeze_score_in_cotrain": False,
            "cotrain_head": "lsi",
            "train_tracking_head": False,
        })
    elif mode == "independent_pair":
        # Paper baseline: VAE-only stage, then both priors on the same frozen VAE.
        cfg.update({
            "T_terminal": 0.0,
            "epochs_refine": int(cfg["epochs_refine"] or cfg["epochs_vae"]),
            "score_w_vae": 0.0,
            "kl_reg_type": "terminal",
            "kl_w": float(beta0),
            "use_latent_norm": False,
            "latent_anchor_mode": "none",
            "freeze_score_in_cotrain": True,
            "cotrain_head": "lsi",
            "train_tracking_head": True,
            "factored_head": True,
        })
    elif mode == "naive_tweedie_cotrain":
        # Standard DSM/Tweedie epsilon target is the active encoder-shaping loss.
        # No CSEM-specific terminal anchor is used here: this is the collapse
        # control requested by the paper, not a proposed stabilized DSM method.
        cfg.update({
            "T_terminal": float(cfg["t_max"]),
            "score_time_weighting": "unweighted-eps",
            "score_head_time_weighting": "unweighted-eps",
            "score_w_vae": 1.0,
            "score_head_loss_w": 1.0,
            "use_latent_norm": False,
            "latent_anchor_mode": "none",
            "kl_reg_type": "normal",
            "kl_w": 0.0,
            "freeze_score_in_cotrain": False,
            "cotrain_head": "control",
            "train_tracking_head": False,
            "factored_head": True,
        })
    else:
        raise ValueError(f"Unknown paper mode: {mode}")
    if mode == "fmnist_aug13_gaussian":
        cfg.update({
            "training_engine": "aug13_joint",
            "first_arm": "norm",
            "resolved_outer_score_time_weighting": "unweighted-eps",
            "resolved_score_head_time_weighting": "unweighted-eps",
            "resolved_split_score_gradient_routing": False,
            "score_head_loss_w": float(cfg["score_w_vae"]),
            "lr_ldm": float(cfg["lr_score_head"]) * float(cfg["score_w_vae"]),
            "base_lr_ldm": float(cfg["lr_score_head"]) * float(cfg["score_w_vae"]),
            "evaluation_contract": "original_gaussian_cfg3_heun50_rk4_25_plus_paper_rk4_20_cfg3_1p5_heun20_and_feature_sw2_score_gap",
            "paper_solver_suite": False,
            "eval_tk_vs_t_comparison": False,
            "generate_visualizations": False,
        })
    elif mode == "fmnist_aug18_recovered":
        cfg["score_head_sample_mode"] = "shared"
    if mode == "independent_pair":
        # Equal tracking LR for the two independently trained heads.
        cfg["lr_ldm"] = cfg["lr_score_head"]
        cfg["lr_refine"] = cfg["lr_score_head"]
        cfg["eval_tk_vs_t_comparison"] = False
    return cfg


def resolve_config(row: Dict[str, str], base_dir: Path):
    result_root = base_dir / "results" / row["result_name"]

    mode = row["mode"]
    cfg = build_base_config(
        dataset=row["dataset"], seed=int(row["seed"]),
        epochs=int(row["epochs_joint"]), refine_epochs=int(row["epochs_refine"]),
        T_K=float(row["T_K"]), T_full=float(row["T_full"]),
        csem_w=float(row["csem_w"]), terminal_kl_w=float(row["terminal_kl_w"]),
        eval_every=int(row["eval_every"]), eval_samples=int(row["eval_samples"]),
        cfg_scale=float(row["cfg_scale"]), results_dir=str(result_root),
        score_time_weighting=row["score_time_weighting"],
        score_head_time_weighting=row["score_head_time_weighting"],
        rk4_steps=int(row["rk4_steps"]),
        paper_solver_suite=row.get("paper_solver_suite", "0") in {"1", "true", "True"},
        paper_solver_steps=int(row.get("paper_solver_steps", "20") or 20),
        latent_anchor_mode=row.get("latent_anchor_mode", "none") or "none",
        lr_score_head_override=(float(row["lr_score_head"]) if row.get("lr_score_head", "").strip() else None),
        lr_refine_override=(float(row["lr_refine"]) if row.get("lr_refine", "").strip() else None),
    )
    cfg = configure_mode(cfg, mode, beta0=float(row.get("beta0", 0.07) or 0.07))
    cfg["paper_cfg_grid"] = [float(row["cfg_scale"])]
    if row["family"] == "main" or (mode == "raw_terminal_control" and row["family"] == "scale_anchor"):
        cfg["paper_cfg_grid"].append(1.5 if row["dataset"] == "FMNIST" else 3.0)
    cfg["paper_cell_id"] = int(row["cell_id"])
    cfg["paper_family"] = row["family"]
    cfg["paper_result_name"] = row["result_name"]
    cfg["data_root"] = str(base_dir / "data_cache_v6")

    cfg['paper_checkpoint_every'] = 10
    return cfg


def run_cell(row: Dict[str, str], base_dir: Path, resume=False):
    cfg = resolve_config(row, base_dir)
    result_root = Path(cfg['results_dir'])
    mode = row['mode']
    if result_root.exists():
        checkpoint = result_root / 'checkpoints' / 'training_state_latest.pt'
        if not resume or not checkpoint.is_file():
            raise FileExistsError(f'Existing run needs --resume and a complete-state checkpoint: {result_root}')
        if mode == 'fmnist_aug13_gaussian':
            raise ValueError('The historical engine has no complete-state resume; import completed results instead')
        cfg['paper_resume_state'] = str(checkpoint)
    result_root.parent.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(exist_ok=True)
    (result_root / 'paper_cell_config.json').write_text(json.dumps(cfg, indent=2, default=str) + '\n')
    core.seed_everything(int(cfg["seed"]))
    if mode == "fmnist_aug13_gaussian":
        import importlib.util
        source = base_dir / "legacy_fmnist.py"
        spec = importlib.util.spec_from_file_location("csem_fmnist_aug13", source)
        historical = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(historical)
        # Reuse the dataset-lock fix without changing transforms/minibatches.
        historical.make_dataloaders = lambda batch_size, num_workers, dataset_key="FMNIST": core.make_dataloaders(
            batch_size, num_workers, dataset_key, cfg["data_root"])
        if historical.torch.cuda.is_available():
            historical.torch.backends.cuda.matmul.allow_tf32 = True
            historical.torch.backends.cudnn.allow_tf32 = True
            historical.torch.set_float32_matmul_precision("high")
        install_historical_evaluator(historical, source)
        loss_df, eval_df = historical.train_vae_cotrained_cond(cfg)
        # The recovered source starts from N(0,I) but uses legacy column names.
        # Explicitly label that contract for the existing paper compiler.
        from paper_io import normalize_metrics
        eval_df, _ = normalize_metrics(eval_df)
        eval_df.to_csv(result_root / "dataframes" / "eval_metrics.csv", index=False)
    else:
        loss_df, eval_df = core.train_vae_cotrained_cond(cfg)
    (result_root / "paper_cell_config.json").write_text(json.dumps(cfg, indent=2, default=str) + "\n")
    return loss_df, eval_df, cfg


"""Paper-only evaluation additions; original August-13 training source stays intact."""
import ast
import copy
import inspect
import random
from contextlib import contextmanager
import numpy as np

@contextmanager
def fixed_metric_rng(torch, seed):
    py_state=random.getstate(); np_state=np.random.get_state()
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed)); random.seed(int(seed)); np.random.seed(int(seed))
        try: yield
        finally: random.setstate(py_state); np.random.set_state(np_state)

def feature_sw2(real, fake, cfg, compute):
    import torch
    # Feature-space distance is distinct from the legacy latent-space SW2.
    n=int(cfg.get('feature_sw2_n_projections',256))
    g=torch.Generator(device='cpu').manual_seed(int(cfg.get('seed',42))+33333)
    theta=torch.randn(real.shape[1],n,generator=g).to(real.device,real.dtype)
    theta=theta/theta.norm(dim=0,keepdim=True).clamp_min(1e-12)
    return max(float(compute(real,fake,n_projections=n,theta=theta)),0.0)**0.5

_gap_cache={}
def score_unit_gap(original, net, mus, logvars, cfg, device):
    import torch
    if original not in _gap_cache:
        tree=ast.parse(inspect.getsource(original))
        fn=tree.body[0];fn.name='_paper_score_unit_gap'
        count=0
        for n in ast.walk(fn):
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='score_gap_per_sample' for t in n.targets):
                n.value=ast.parse('(eps_diff_sq / sigma_sq).sum(dim=(1,2,3))',mode='eval').body;count+=1
        if count!=2: raise RuntimeError('Unexpected CSEM diagnostic source; refuse silent epsilon-gap substitution')
        ns=dict(original.__globals__);exec(compile(ast.fix_missing_locations(tree),'<paper_score_gap>','exec'),ns)
        _gap_cache[original]=ns[fn.name]
    with fixed_metric_rng(torch,int(cfg.get('seed',42))+44444):
        return _gap_cache[original](net,mus,logvars,cfg,device,labels=None,
            num_classes=cfg.get('num_classes'),num_samples=int(cfg.get('eval_lsi_gap_samples',2500)),
            num_time_points=int(cfg.get('eval_lsi_gap_time_points',50)),batch_size=int(cfg['batch_size']))

def install_historical_evaluator(module, source_path):
    original=module.evaluate_current_state
    tree=ast.parse(source_path.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_current_state')
    fn=copy.deepcopy(fn);fn.name='_paper_extra_evaluate'
    # Replace only the evaluator's sampler list and its unet extension block.
    for j,n in enumerate(fn.body):
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='configs' for t in n.targets):
            fn.body[j]=ast.parse("configs = [{'method':'VAE_Rec_eps','steps':0,'desc':'Recon (posterior z)','use_rand_token':False}] + cfg['_paper_extra_samplers']").body[0]
            if not (isinstance(fn.body[j+1],ast.If) and ast.unparse(fn.body[j+1].test)=='unet is not None'):
                raise RuntimeError('Unexpected historical evaluation structure')
            del fn.body[j+1];break
    code=ast.unparse(ast.fix_missing_locations(fn))
    code=code.replace('fid = compute_fid_from_features(real_features, fake_features)',
        'feature_w2 = feature_sw2(real_features, fake_features, cfg, compute_sw2)\n        fid = compute_fid_from_features(real_features, fake_features)')
    code=code.replace("'w2': w2,", "'w2': w2, 'feature_w2': feature_w2,")
    for typ in ['rk4','heun']:
        line=f"output_dict[f'sw2_{typ}_{{col_suffix}}'] = r['w2']"
        code=code.replace(line,line+f"\n            output_dict[f'feature_sw2_{typ}_{{col_suffix}}'] = r['feature_w2']")
    code=code.replace("output_dict['sw2_vae_recon'] = r['w2']", "output_dict['sw2_vae_recon'] = r['w2']\n            output_dict['feature_sw2_vae_recon'] = r['feature_w2']")
    code=code.replace("output_dict['lsi_gap_unet_uncond'] = lsi_gap_unet", "output_dict['lsi_gap_unet_uncond'] = lsi_gap_unet\n    output_dict['csem_gap_score_uncond'] = score_unit_gap(compute_lsi_gap, unet, encoder_mus, encoder_logvars, cfg, device)")
    ns=module.__dict__;ns.update(feature_sw2=feature_sw2,score_unit_gap=score_unit_gap)
    exec(compile(code,'<aug13_paper_eval_only>','exec'),ns)
    extra=ns['_paper_extra_evaluate'];signature=inspect.signature(original)
    def evaluate(*args,**kwargs):
        result=original(*args,**kwargs)
        bound=signature.bind(*args,**kwargs);bound.apply_defaults();kw=dict(bound.arguments)
        cfg=copy.deepcopy(kw['cfg'])
        cfg['_paper_extra_samplers']=[dict(method=method,steps=steps,desc='Paper comparison',use_rand_token=True,cfg_level=g)
            for g in dict.fromkeys(float(x) for x in cfg['paper_cfg_grid'])
            for method,steps in [('rk4_ode',20),('rk4_ode',25),('heun_sde',20)]]
        kw['cfg']=cfg
        # Extra metrics cannot consume the training RNG or alter original metrics.
        with fixed_metric_rng(module.torch,int(cfg.get('seed',42))+55555): additions=extra(**kw)
        result.update({k:v for k,v in additions.items() if k not in result})
        from paper_io import normalize_metrics
        import pandas as pd
        normalized, _ = normalize_metrics(pd.DataFrame([result]))
        return normalized.iloc[0].to_dict()
    module.evaluate_current_state=evaluate
