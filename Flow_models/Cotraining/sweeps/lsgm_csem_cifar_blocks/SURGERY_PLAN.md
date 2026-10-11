# LSGM → CSEM surgery plan: 1-, 2- and 4-group CIFAR-10

*Plan for converting LSGM's own CIFAR-10 code into CSEM co-training (1 group) and hierarchical-CSEM co-training (2 and 4 groups). The changes are kept as small as possible, and LSGM baselines are run on identical backbones. Sources: NVlabs/LSGM at a pinned commit, the LSGM paper and supplement, and `csem_cifar_blocks_800.zip`. Every LSGM file, function and flag named below was checked against the pinned commit.*

---

## 0. Summary

**Principle.** Only the second-stage training objective changes. That means three things: the score target, the encoder-side (q) objective, and, for 2 and 4 groups, the factorization of the latent prior, which CSEM requires. Everything else stays as LSGM ships it: the NVAE backbone, the NCSN++ prior, the VPSDE, the time-sampling and importance-weighting modes, the optimizers and learning rates, EMA, AMP, spectral regularization, two-stage training, the epoch budget, and the FID/NLL evaluation code.

**Key decisions**

1. **Backbone.** Use LSGM's CIFAR-10 "best NLL" NVAE (Table 7; README `vae3`) with **`--num_nf 0`**, at 1, 2 and 4 groups. Total latent width is held at 180 channels at 16×16 (180/G channels per group). It is the only published LSGM CIFAR backbone with 4 or fewer groups, and it is the backbone of LSGM's ablations: FID 5.19 end-to-end vs. 9.00 with the VAE frozen (App. G.5.2). Normalizing flows must go because they make q(z|x) non-Gaussian and destroy the analytic CSEM target.
2. **Training protocol.** Use LSGM's *ablation* protocol, not the full one (App. G.5.1–G.5.2): 1000 epochs, global batch 128, NCSN++ with 2 residual cells per scale, VPSDE, w_un prior weighting with importance sampling (`drop_all_iw`), and a separate likelihood-weighted encoder batch (`ll_iw`). This is the manageable-cost protocol that produced the published small-group numbers.
3. **Matched LSGM baselines on identical backbones.** We retrain LSGM itself at nf=0 with 1, 2 and 4 groups. Its published 4-group numbers had nf=2, so they are a reference, not the baseline. One extra cell reproduces the published nf=2 configuration to validate the port before any comparison is trusted.
4. **The 1-group comparison isolates the estimator.** With 1 group, LSGM's objective and CSEM + 𝒦 + B (𝒦 is the terminal KL, B the small-time band term; both defined in §4.3) are the same population objective up to a term of order 10⁻⁵ (§6.2). The difference between the two arms is therefore the Rao–Blackwellized estimator.
5. **For 2 and 4 groups, CSEM needs a cascade.** NVAE's posterior q(z_g | x, z_<g) is Gaussian given the realized earlier groups, so each group gets a conditional NCSN++ head conditioned on clean ancestors. This is the same design as `hier_models.py` in the zip. An optional **LSGM-cascade** control (LSGM objective, cascade prior) separates the estimator effect from the factorization effect.
6. **Keep the dual routing LSGM already has.** LSGM's joint trainer already routes a likelihood-weighted q-objective to the VAE and a separately sampled, detached, unweighted p-objective to the prior. That is CSEM's dual routing. The surgery swaps targets inside this structure and does not rebuild it.
7. **Add B at LSGM's `time_eps`.** LSGM trains on t ≥ 0.01, where σ² ≈ 2×10⁻³ and σ ≈ 0.045. Without B, CSEM cannot see posterior sharpening below σ ≈ 0.045, while LSGM charges it exactly through its encoder entropy. With B, the 1-group objectives match.
8. **Unconditional generation; LSGM's FID protocol.** No class labels or CFG. FID is 50k ODE samples against training-set statistics, using LSGM's `fid/` code. The zip's ~8.85 FID is class-conditional with CFG, measured on 10k test images with a different Inception backend, and is **not comparable**.
9. **Cluster scaffolding follows the zip.** That means `paper.py validate/submit/run/report/smoke/prepare-data`, `job.slurm` on `gh` with one node, one task and 48 h, status JSONs, locks, resume, and a manifest. `paper_io.py` is reused unchanged. `core.py`, `suite.py` and `hier_*.py` are not used, because LSGM's code is the trainer.

---

## 1. Sources and pinning

| What | Where | Notes |
|---|---|---|
| LSGM code | `git clone https://github.com/NVlabs/LSGM.git && git checkout 5eae2f385c014f2250c3130152b6be711f6a3a5a` | HEAD as of 2021-12-02 ("added PWC tags"). Vendor it into the sweep as `lsgm/`, keep its `LICENSE` (NVIDIA Source Code License, non-commercial research), and record per-file SHA-256 in `provenance.json`, as the zip does for `core.py`. |
| Backbone and protocol | Paper Table 2, Table 6, Sec. 5.2; supplement App. G.1, G.5.1–G.5.3, Table 7 | Supplement: `https://proceedings.neurips.cc/paper/2021/file/5dca4c6b9e244d24a30b4c45601d9720-Supplemental.pdf` |
| Training commands | LSGM `README.md`: `vae3` (stage 1) and `lsgm3`, the best-NLL stage-2 command | One `lsgm3` flag (`--arch_instance_dae res_ho_attn`) does not exist in `train_vada.py`'s parser and must be dropped (§4.1). |
| Inception weights for FID | `lsgm/fid/inception.py` → `pt_inception-2015-12-05-6726825d.pth` (pytorch-fid release) | Download once into `$FID_DIR` from a node with internet access. |
| CIFAR-10 | torchvision, prepared with the zip's `paper_io.prepare_dataset` under its lock | LSGM's `dset.CIFAR10(root=…)` reads the same layout. Point `--data` at `<cache>/CIFAR`. |
| CSEM sweep conventions | `csem_cifar_blocks_800.zip`: `paper.py`, `job.slurm`, `manifest.csv`, `paper_io.py`, `tests.py`, `README.md`, `hier_models.py`, `hier_engine.py`, and `core.py` for CSEM semantics | Used for the CSEM hyperparameter semantics (§5) and the sweep and submission machinery (§8). Its trainer and models are **not** used. |

---

## 2. Architectures

### 2.1 Why this backbone

| LSGM CIFAR-10 backbone | Groups × channels | Flows | Status |
|---|---|---|---|
| best FID (`vae2`) / balanced (`vae1`) | 20 × 9 | 0 | Too many groups: a CSEM cascade would need 20 sequential solves. |
| **best NLL (`vae3`)** | **4 × 45** | **2** | **Used, with flows set to 0.** Its ablations give FID 4.99–5.39 (Table 6), 5.19 end-to-end vs. 9.00 frozen. |
| single-group ablation (G.5.3) | 1 × ? | ? | FID 7.60 with the mixed score. Width and flows are unstated, so we instantiate it as `vae3` with G=1. |

**This backbone is a good hierarchy testbed.** In NVAE, group g's posterior comes from `enc_sampler[g]` applied to features that depend on the realized earlier groups through the decoder state (`nvae.py` `forward`, roughly lines 264–289). It is therefore exactly **conditionally Gaussian** with nf=0, which is what CSEM needs. NVAE also injects group g at decoder depth g (`DecCombinerCell`), so the architecture breaks the symmetry between groups that the Fashion-MNIST 2+2 script lacked.

### 2.2 Configurations

All use one latent scale at 16×16, 256 encoder/decoder channels, 3 cells per conditional, 3 pre/post-processing cells, channel multipliers (1, 2), `res_bnswish` cells with squeeze-excitation, and a discretized mixture-of-logistics decoder with 8 bits.

| ID | Groups | Channels per group | Total latent | nf | Used by |
|---|---|---|---|---|---|
| `A1` | 1 | 180 | 180×16×16 | 0 | LSGM-1, CSEM-1 (+ ablations) |
| `A2` | 2 | 90 | 180×16×16 | 0 | LSGM-2, CSEM-2, LSGM-cascade-2 |
| `A4` | 4 | 45 | 180×16×16 | 0 | LSGM-4, CSEM-4, LSGM-cascade-4 |
| `A0` | 4 | 45 | 180×16×16 | **2** | reproduction gate only (§7, cell `B0`) |

Holding total latent width at 180 mirrors the zip's rule of fixed total latent channels (8 = 4+4 = 3+3+2). If you would rather match G.5.3's possibly 45-channel single group, add an `A1'` with 45 channels. The paper does not say which it used.

### 2.3 Prior networks

| Arm | Prior | Channels per NCSN++ (`num_channels_dae`) | Parameters vs. LSGM's prior |
|---|---|---|---|
| LSGM joint, G ∈ {1, 2, 4} | one NCSN++ over all 180 channels | 512 | 1× |
| CSEM-1 | one NCSN++ (no context) | 512 | 1× |
| CSEM cascade / LSGM cascade, G=2 | 2 conditional NCSN++ (90 channels each) | **352** (512/√2 rounded down to a multiple of 32, for GroupNorm) | ≈0.95× |
| CSEM cascade / LSGM cascade, G=4 | 4 conditional NCSN++ (45 channels each) | **256** | ≈1.0× |

Prior parameters scale roughly as channels², so this matches *total* prior parameters, the analogue of the zip's matched total DiT depth (12 = 6+6 = 4+4+4). Print and save the actual parameter counts, as the zip does. Sampling remains sequential across groups (G ODE solves). Report the number of function evaluations (NFE) both summed over heads and per head.

### 2.4 Published reference points (for calibration, not baselines)

| Published LSGM, CIFAR-10 | FID | NELBO (bpd) |
|---|---|---|
| 4-group (nf=2), ablation protocol, w_un with importance sampling, end-to-end (Sec. 5.2) | 5.19 | 2.98 |
| same, VAE frozen in stage 2 | 9.00 | 3.03 |
| same backbone, best Table 6 cell (linear VPSDE, w_re)* | 4.99 | 2.99 |
| 1-group, mixed score (Sec. 5.2) | 7.60 | 3.29 |
| LSGM (NLL), full protocol, 4-group (nf=2) | 6.89 | ≤2.87 |

\*Table 6's columns don't extract cleanly, so the w_re assignment is reconstructed. Check it against the PDF before citing.

---

## 3. LSGM protocol held fixed

### 3.1 Stage 1: NVAE pretraining with an N(0, I) prior per group (`train_vae.py`, unchanged)

This stage is identical for the LSGM and CSEM arms, and its checkpoints are **shared**: every LSGM/CSEM pair at a given G starts from the same stage-1 weights. In CSEM terms this stage is the β-VAE corner, (T_K, T) = (0, 0).

| Setting | Value | Source |
|---|---|---|
| Architecture flags | as in `vae3` | README |
| `--num_nf` | **0** (A1/A2/A4); 2 (A0) | surgery / `vae3` |
| `--epochs` | 400 | `vae3` |
| Global batch | 256 (`vae3`: 64/GPU × 4) → **`--batch_size 256 --num_process_per_node 1`** | `vae3` |
| Optimizer | Adamax, lr 1e-2, cosine to 1e-4, eps 1e-3, weight decay 3e-4, 5 warmup epochs | `train_vae.py` defaults |
| Spectral/BN regularization | `--weight_decay_norm 1e-2` | `vae3` |
| KL schedule | `--kl_anneal_portion 1.0 --kl_max_coeff 1.0`, KL balancing on | G.5.1: ablations anneal KL to **1.0** (`vae3` itself used 0.7) |
| Precision | AMP (`autocast_train` defaults to True) | default |
| Seed | 42 (the sweep convention; LSGM's default is 1) | sweep |
| Checkpoint handed to stage 2 | `checkpoint.pt`, which is the **best-validation** snapshot: written only when validation log p improves, checked every `epochs // 40` = 10 epochs | `train_vae.py` lines 102–129 (LSGM behavior; shared by both arms) |

### 3.2 Stage 2: end-to-end (`train_vada.py` + `training_obj_joint.py`)

| Setting | Value | Source |
|---|---|---|
| Score network | NCSN++: `--dae_arch ncsnpp --num_channels_dae 512 --num_scales_dae 3 --num_cell_per_scale_dae 2 --dropout 0.2 --embedding_scale 1000 --fir --apply_sqrt2_res` (no `--custom_conv_dae`) | `lsgm3`, with 2 cells per scale per G.5.1 |
| Mixed score | on (`args.mixed_prediction = True` is hard-coded); `--mixing_logit_init` default −3 | `train_vada.py` line 73 |
| SDE | `--sde_type vpsde --beta_start 0.1 --beta_end 20.0 --sigma2_0 0.0 --time_eps 0.01` | G.5.2 (VPSDE); `time_eps` from `lsgm1`/`lsgm2` |
| Time sampling | `--iw_sample_p drop_all_iw --iw_sample_q ll_iw` | G.5.2: w_un with importance sampling for the prior, a separate r_ll(t) batch for the encoder |
| VAE optimizer | Adam (FusedAdam if Apex is installed), lr 1e-4 cosine to 1e-5, eps 1e-3, weight decay 3e-4 | `train_vada.py` defaults |
| Prior optimizer | Adam lr 1e-4 (constant: min = max), eps 1e-4, wrapped in EMA 0.9999 | `lsgm3`; `train_vada.py` lines 93–95 |
| Warmup / clipping | 5 epochs of linear warmup for both; no gradient clipping (`grad_clip_max_norm 0`) | defaults |
| Spectral regularization | `--weight_decay_norm_vae 1e-2 --weight_decay_norm_dae 1e-2` | `lsgm3` |
| KL coefficient | `--cont_kl_anneal` (0.7 → 1.0 over the first 10% of iterations), applied **identically** to the CSEM q-objective | `lsgm3`; see risk R3 |
| Epochs / batch | `--epochs 1000`; global batch 128 → `--batch_size 128 --num_process_per_node 1` | G.5.1 |
| Precision | AMP + `GradScaler(2**10)` | `train_vada.py` |
| In-training eval | every `epochs // 20` epochs: 2,500-sample ODE FID and ODE NLL on 10 validation batches; `checkpoint.pt` every `epochs // 100` epochs | `train_vada.py` |
| Final eval | `--skip_final_eval` in training; a separate Phase C runs `evaluate_vada.py` identically for every arm (§8) | |

### 3.3 Evaluation (Phase C, `evaluate_vada.py`, cascade-aware per §4.6)

FID: `--fid_ode_eval --num_fid_samples 50000 --ode_eps 1e-6 --ode_solver_tol 1e-5` (RK45), against `cifar10.npz` training-set statistics from `scripts/precompute_fid_statistics.py`. NLL: `--nll_ode_eval` (encoder entropy + ODE prior likelihood + reconstruction). This evaluation does not depend on the training objective, so it is a fair comparison. Use `checkpoint.pt` at the final epoch (EMA swapped in automatically) for headline numbers. Optionally also report `checkpoint_fid.pt`, selected identically for all arms.

---

## 4. The code surgery

### 4.0 Overview

| File | Change | Arms affected | Changes numerics? |
|---|---|---|---|
| `util/datasets.py` | `_accumulate` import fix; `download=False` | all | no |
| `thirdparty/functions.py` | SyncBN backward signature for torch ≥ 1.9 | all | no |
| `fid/fid_score.py` | `sqrtm(..., disp=False)` fallback for new SciPy | all | no |
| `nvae.py` | stash per-group posterior distributions (1 line) | all (unused by LSGM-joint) | no |
| **`csem_lsgm.py` (new)** | CSEM target, 𝒦, B, conditional heads, cascade wrapper, cascade sampling and NLL, prior builder | CSEM / cascade | yes (new code) |
| **`training_obj_joint.py`** | target swap; CSEM q-objective; cascade context | CSEM / cascade | **yes (the surgery)** |
| `train_vada.py` | flags; prior built through `csem_lsgm.build_prior`; eval CSV hook; smoke-step limit | all | no for LSGM-joint |
| `evaluate_diffusion.py`, `evaluate_vada.py` | cascade-aware sampling, NLL and prior construction | cascade | no for joint |
| `training_obj_disjoint.py`, `diffusion_*.py`, `score_sde/*`, `layers/*`, `train_vae.py` | **unchanged** | — | — |

Invariant: with no new flags set, the patched code must reproduce LSGM bit-for-bit given a fixed seed on the same hardware. Test this in the smoke stage by comparing first-step losses with and without the patch.

### 4.1 Compatibility patches (no numerical effect; needed on Vista's torch 2.x / Python 3.11 / aarch64 stack)

1. **`util/datasets.py` line 20:** `from torch._utils import _accumulate` was removed in torch 2.x. Replace it with `from itertools import accumulate as _accumulate`.
2. **`thirdparty/functions.py`, around line 115, in the in-place SyncBatchNorm backward:** for torch ≥ 1.9, `torch.batch_norm_backward_elemt` takes `(grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count)` instead of `mean_dy, mean_dy_xmu`. Pass `sum_dy, sum_dy_xmu, count_tensor.to(torch.int32)` and drop the manual division. NVAE's `SyncBatchNormSwish` routes through this function even at world size 1.
3. **`fid/fid_score.py` line 198:** `linalg.sqrtm(..., disp=False)` breaks on recent SciPy. Use the same try/fallback as the user's Fashion-MNIST script's `fid_numeric`.
4. **`util/datasets.py`, `get_loaders_eval`, CIFAR branch:** set `download=False`. The sweep prepares the cache under its lock, and training must never download (the zip's rule).
5. **Custom CUDA ops (`score_sde/op`, used through `--fir`):** these are JIT-compiled with `torch.utils.cpp_extension.load` on first import, which needs `nvcc` and `ninja`. In `job.slurm`, `module load cuda/<12.x>` and set `TORCH_EXTENSIONS_DIR=$SCRATCH/torch_extensions/lsgm`; the `smoke` step compiles them once. Fallback if the build fails on aarch64: route `upfirdn2d` to the pure-PyTorch `upfirdn2d_native` already in `score_sde/op/upfirdn2d.py`, then record and log the swap.
6. **README `lsgm3` command:** drop `--arch_instance_dae res_ho_attn`. That flag is not defined in `train_vada.py`'s parser and would abort `parse_args()`.
7. **Environment additions to the `hlsi` venv:** `tensorboardX`, `lmdb` (imported unconditionally by `util/datasets.py`), `torchdiffeq` (`method="scipy_solver"` needs ≥ 0.2), `ninja`. Apex is optional; without it LSGM falls back to `torch.optim.Adam`. Log which one was used.

### 4.2 `nvae.py`: expose posterior moments (1 line)

In `NVAE.forward`, just before `return logits, all_log_q, all_eps` (line 299), add `self.last_q = all_q`. `all_q` is the list of per-group `Normal(mu_q, log_sig_q)` built at lines 270–273, after `soft_clamp5`/`soft_clamp`. No call signature changes. With `--num_nf 0`, each `eps` equals `mu_q + sigma_q * ε` exactly; assert this in the smoke test.

### 4.3 New `csem_lsgm.py`

```python
import copy, math, torch, torch.nn as nn, torch.nn.functional as F
from score_sde.ncsnpp import NCSNpp
from util import utils

def posterior_moments(vae):
    """Concatenated (mu, log_sigma) of q(z_g | x, z_<g) for all groups; fp32."""
    mu = torch.cat([d.mu for d in vae.last_q], 1).float()
    log_sig = torch.cat([d.log_sigma for d in vae.last_q], 1).float()
    return mu, log_sig

def csem_eps_target(eps_t, mu, log_sig, m_t, var_t):
    """E[noise | eps_t, x, z_<g] for the VPSDE kernel eps_t = m_t z0 + sqrt(var_t) noise
    applied to the Gaussian component N(mu, sigma^2). Replaces `noise` in LSGM's l2 term."""
    var_comp = m_t ** 2 * torch.exp(2 * log_sig) + var_t
    return torch.sqrt(var_t) * (eps_t.float() - m_t * mu) / var_comp

def terminal_component_kl(mu, log_sig, m_K, var_K):
    """Per-coordinate KL(q_tK(.|x, z_<g) || N(0, I)), with m_K, var_K at the representation horizon."""
    v = m_K ** 2 * torch.exp(2 * log_sig) + var_K
    return 0.5 * (m_K ** 2 * mu ** 2 + v - 1.0 - torch.log(v))

def band_term(log_sig, m_e, var_e):
    """Per-coordinate B = 1/2 log(1 + var_e / (m_e^2 sigma^2)) at t = time_eps.
    Exact upper bound on I(X;Z_0) - I(X;Z_eps); equals the piece of LSGM's exact
    encoder entropy that truncation at time_eps would otherwise drop."""
    return 0.5 * F.softplus(math.log(var_e) - math.log(m_e ** 2) - 2 * log_sig)

class _ContextInputConv(nn.Module):
    """Wraps NCSN++'s input conv: base(x) + zero-initialised projection of clean ancestors."""
    def __init__(self, base, ctx_ch, nf):
        super().__init__()
        self.base, self.proj, self.context = base, nn.Conv2d(ctx_ch, nf, 3, padding=1), None
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
    def forward(self, x):
        return self.base(x) + self.proj(self.context)

class ConditionalNCSNpp(nn.Module):
    def __init__(self, args, in_ch, ctx_ch, nf):
        super().__init__()
        a = copy.copy(args); a.num_channels_dae = nf
        self.net, self.ctx_ch = NCSNpp(a, in_ch), ctx_ch
        if ctx_ch:
            assert a.progressive_input == 'none'          # then all_modules[2] is the input conv
            self.net.all_modules[2] = _ContextInputConv(self.net.all_modules[2], ctx_ch, nf)
    def forward(self, x, t, context=None):
        if self.ctx_ch:
            self.net.all_modules[2].context = context
        return self.net(x, t)

class CascadeDAE(nn.Module):
    """One conditional NCSN++ per NVAE group; looks like LSGM's `dae` to the trainer,
    the EMA optimizer, the spectral-norm calculator and checkpointing."""
    def __init__(self, args, group_channels, nf):
        super().__init__()
        self.group_channels = list(group_channels)
        self.heads = nn.ModuleList([ConditionalNCSNpp(args, c, sum(group_channels[:g]), nf)
                                    for g, c in enumerate(group_channels)])
        self.mixed_prediction = args.mixed_prediction
        self.num_input_channels = sum(group_channels)
        self.input_size = self.heads[0].net.input_size
        self.is_active = None
    @property
    def mixing_logit(self):
        return torch.cat([h.net.mixing_logit for h in self.heads], 1)
    def forward(self, x, t, context):
        outs, off = [], 0
        for h, c in zip(self.heads, self.group_channels):
            outs.append(h(x[:, off:off + c], t, context[:, :off] if off else None)); off += c
        return torch.cat(outs, 1)

class _BoundHead(nn.Module):
    """Adapts head g with fixed clean ancestors to LSGM's `dae(x=..., t=...)` interface."""
    def __init__(self, head, context):
        super().__init__(); self.head, self.context = head, context
        self.mixed_prediction, self.mixing_logit = head.net.mixed_prediction, head.net.mixing_logit
    def forward(self, x, t):
        return self.head(x, t, self.context)

def sample_cascade_ode(diffusion, dae, num_samples, ode_eps, tol, enable_autocast, temp, noise=None):
    zs, nfe, secs, off = [], 0, 0.0, 0
    for head, c in zip(dae.heads, dae.group_channels):
        ctx = torch.cat(zs, 1) if zs else None
        n = None if noise is None else noise[:, off:off + c]
        z, k, s = diffusion.sample_model_ode(_BoundHead(head, ctx), num_samples,
                                             [c, dae.input_size, dae.input_size],
                                             ode_eps, tol, enable_autocast, temp, n)
        zs.append(z); nfe += k; secs += s; off += c
    return torch.cat(zs, 1), nfe, secs

def cascade_ode_nll(diffusion, dae, eps, **kw):
    """-log p(z) = sum_g -log p(z_g | z_<g), with clean ancestors taken from the posterior sample."""
    total, nfe, off = 0.0, 0, 0
    for head, c in zip(dae.heads, dae.group_channels):
        ctx = eps[:, :off] if off else None
        nll, k, _, _ = diffusion.compute_ode_nll(dae=_BoundHead(head, ctx), eps=eps[:, off:off + c], **kw)
        total, nfe, off = total + nll, nfe + k, off + c
    return total, nfe, None, None

def q_times_rep(diffusion, size, time_eps, t_K):
    """q-route times on [time_eps, t_K) for rep_horizon < 1 (uniform; canonical weight / density)."""
    t = time_eps + torch.rand(size, device='cuda') * (t_K - time_eps)
    var_t, m_t, g2_t = diffusion.var(t), diffusion.e2int_f(t), diffusion.g2(t)
    w = g2_t / (2.0 * var_t) * (t_K - time_eps)
    return t, var_t.view(-1, 1, 1, 1), m_t.view(-1, 1, 1, 1), w.view(-1, 1, 1, 1)

def build_prior(args, vae):
    if getattr(args, 'prior_factorization', 'joint') == 'joint':
        return utils.get_dae_model(args, vae.latent_structure()[0])     # LSGM, unchanged
    G = vae.num_groups_per_scale
    nf = args.dae_channels_per_head or 32 * int(args.num_channels_dae / math.sqrt(G) // 32)
    return CascadeDAE(args, [vae.num_latent_per_group] * G, nf)
```

Notes:
- Group order. `all_eps` is appended in decoder (top-down) order, and `concat_eps_per_scale` and `decompose_eps` use that same order. So "ancestors of g" are channels `[:off]`, and `vae.decompose_eps` splits per-group terms consistently.
- `CascadeDAE.mixing_logit` returns a concatenation, so in-place writes to it go nowhere. Assert `not args.drop_inactive_var` for cascade runs; CIFAR doesn't use it.
- The context projection is a plain `nn.Conv2d`, so LSGM's spectral-norm calculator (`custom_conv=False` for NCSN++) regularizes it like every other prior conv. That is consistent behavior.

### 4.4 `training_obj_joint.py`: the actual surgery

Changes inside `train_vada_joint` (line references are to the pinned file):

```python
# after the vae forward (lines 36-42)
if args.csem:
    mu_q, log_sig_q = csem_lsgm.posterior_moments(vae)               # requires num_nf == 0

# (lines 49-57 unchanged: t_p/var_t_p/m_t_p from iw_sample_p; t_q/var_t_q/m_t_q from iw_sample_q)
# Under --csem require iw_sample_q in {'ll_uniform','ll_iw'} so p and q use separate batches.
# If args.rep_horizon < 1: draw the q-batch from csem_lsgm.q_times_rep(...) on [time_eps, rep_horizon)
#   with weight g2/(2 var) * (rep_horizon - time_eps); the p-batch keeps LSGM's [time_eps, 1].

# replace the dae call (lines 68-71)
cascade = args.prior_factorization == 'cascade'
ctx = torch.cat([eps.detach(), eps], 0) if cascade else None        # p-route context DETACHED
pred_params = dae(eps_t, t, ctx) if cascade else dae(eps_t, t)
params = utils.get_mixed_prediction(dae.mixed_prediction, pred_params, dae.mixing_logit, mixing_component)

# replace the target (line 72)
if args.csem:
    with autocast(enabled=False):
        tgt_p = csem_lsgm.csem_eps_target(eps_t_p, mu_q.detach(), log_sig_q.detach(), m_t_p, var_t_p)
        tgt_q = csem_lsgm.csem_eps_target(eps_t_q, mu_q, log_sig_q, m_t_q, var_t_q)
    target = torch.cat([tgt_p, tgt_q], 0)
    l2_term = torch.square(params.float() - target)
else:
    l2_term = torch.square(params - noise)      # LSGM line 72 verbatim (shared noise, fp16 under autocast)

# p_objective (lines 76-77) unchanged: sum(obj_weight_t_p * l2_term_p) -> Rao-Blackwellised head regression

# replace the q-route KL assembly (lines 78-88) when args.csem:
rep_per_var = obj_weight_t_q * l2_term_q                              # canonical = LSGM's likelihood weight
K_per_var = (args.csem_lambda_K * csem_lsgm.terminal_component_kl(mu_q, log_sig_q, m_K, var_K)
             if not args.no_csem_terminal else torch.zeros_like(rep_per_var))
B_per_var = (csem_lsgm.band_term(log_sig_q, m_eps, var_eps)
             if not args.no_csem_band else torch.zeros_like(rep_per_var))
per_group = [r + k + b for r, k, b in zip(*(vae.decompose_eps(v)
                                            for v in (rep_per_var, K_per_var, B_per_var)))]
kl_all_list = [g.sum(dim=[1, 2, 3]) for g in per_group]
kl_vals_per_group = torch.stack(kl_all_list, 1).mean(0)
kl_diag_list = [g.sum(dim=[2, 3]).mean(0) for g in per_group]
# (LSGM path unchanged otherwise: cross_entropy_const + kl_per_group_vada(all_log_q, all_neg_log_p))

# lines 91-139 unchanged: kl_coeff schedule, kl_balancer, nelbo_loss = balanced_kl + recon,
# q_loss.backward(retain_graph=True) -> VAE step; dae.zero_grad(); p_loss.backward() -> prior step.
```

Constants come from the diffusion object once per run:
- `m_eps = e2int_f(time_eps)`, `var_eps = var(time_eps)`
- `m_K = e2int_f(rep_horizon)`, `var_K = var(rep_horizon)`

At the defaults: m_eps² = 0.998, var_eps = 1.99×10⁻³, B's σ_q floor ≈ 0.045. At `rep_horizon = 1`, m_1² = 4.3×10⁻⁵.

**What this does, route by route**
- **q-route (VAE).** recon + kl_coeff · Σ_g [likelihood-weighted CSEM over [ε, t_K] + λ_K 𝒦_{t_K} + B_ε]. Gradients reach the VAE through the noised sample, through the analytic target (μ_q, σ_q), and, for cascades, through the clean-ancestor context. This is the zip's "representation route."
- **Prior heads.** Gradients come only from p_loss: w_un weighting with importance sampling, detached latents, detached context, an independently sampled t-batch, and RB targets. LSGM's existing `zero_grad` + `p_loss.backward()` already implements the zip's "independent full-horizon head gradients replace head grads."
- **The existing LSGM paths are untouched.** LSGM-joint is the unchanged code. LSGM-cascade is `--prior_factorization cascade` without `--csem`, which uses the DSM target with entropy plus cross-entropy constant, through conditional heads.

Logging additions: per group, the mean of `rep`, `K` and `B`, and the fraction of coordinates with σ_q below the B floor and below a tenth of it. These are the `below_i` / `deep_i` diagnostics from the Fashion-MNIST script. Also σ_q quantiles, and the mixing coefficients sigmoid(`mixing_logit`), which LSGM reports as < 0.02 at convergence.

### 4.5 `train_vada.py`

- **New arguments:**
  - `--csem` (store_true)
  - `--prior_factorization {joint,cascade}` (default `joint`)
  - `--no_csem_band` (store_true; B is on by default under `--csem`)
  - `--no_csem_terminal` (store_true; 𝒦 is on by default under `--csem`)
  - `--rep_horizon` (default 1.0)
  - `--csem_lambda_K` (default 1.0)
  - `--dae_channels_per_head` (0 = auto)
  - `--eval_csv` (path)
  - `--max_steps_per_epoch` (smoke only; default 0 = off)
- **Guards:**
  - `--csem` or `cascade` ⇒ `args.num_nf == 0` (checked after `override_architecture_fields`, which copies `num_nf` from the stage-1 checkpoint)
  - `--csem` ⇒ `iw_sample_q in {ll_uniform, ll_iw}`
  - `cascade` ⇒ `not drop_inactive_var`, `num_latent_scales == 1`
- **Prior construction:** line 75 becomes `dae = csem_lsgm.build_prior(args, vae)`. Optimizer, EMA, scheduler, spectral norm, checkpointing and resume (lines 93–140, 259–268) are unchanged, because `CascadeDAE` is one `nn.Module`.
- **Evaluation CSV:** in the eval block (lines 189–257), also append `{epoch, ode_fid_2500, ode_nll_bpd, nfe, sampling_seconds}` to `--eval_csv` atomically, using `paper_io.save_csv_atomic` semantics. The `report` command reads this, like the zip's `eval_metrics_in_progress.csv`.

### 4.6 `evaluate_diffusion.py` and `evaluate_vada.py` (cascade-aware evaluation)

- `create_generator_vada` (line 21) and `generate_samples_vada` (line 36): if `isinstance(dae, CascadeDAE)`, use `sample_cascade_ode` on the ODE path. The discretized ancestral path raises `NotImplementedError` for cascades. It isn't used for headline numbers, and Phase C uses ODE evaluation for all arms.
- `elbo_evaluation` (line 67), ODE branch: if cascade, replace `diffusion.compute_ode_nll` with `cascade_ode_nll`.
- `evaluate_vada.py` line 74: `dae = csem_lsgm.build_prior(args, vae)`. `args` comes from the checkpoint, so the new flags propagate automatically, with defaults through `getattr`. The EMA swap for `checkpoint.pt` (lines 81–91) works unchanged.

### 4.7 Deliberately not changed

- NVAE's `soft_clamp5` on μ and `soft_clamp` on log σ.
- The mixed-score parameterization. It is the constant-precision special case of the zip's factored head, so it is kept for parity, and the factored head with its auxiliary loss is not ported.
- The shared noise between p and q batches (`"note that this noise value is currently shared!"`).
- KL balancing (off in stage 2), the warmup, `GradScaler`, and the importance-sampling distributions. With RB targets the encoder no longer needs importance sampling for variance control, but keeping it keeps the arms identical.

---

## 5. CSEM hyperparameters: what transfers from the zip and what does not

| Sweep setting (`manifest.csv`, `suite.build_base_config`) | Sweep value | In the LSGM surgery | Why |
|---|---|---|---|
| Representation time weighting | `canonical` | LSGM `obj_weight_t_q` with `ll_iw` (likelihood weighting g²/(2σ_t²)) | Same physical-time Girsanov weight; LSGM's estimator is unbiased for it. |
| Head time weighting | `unweighted-eps` | LSGM `drop_all_iw` (w_un with importance sampling) | Same pointwise oracle; keeps LSGM's prior metric. |
| Dual routing, independent head time sample, detached head latents | on | already LSGM's joint trainer (`ll_iw` + `different_p_q_objectives`) | Structural match; nothing to add. |
| Clean-parent context, gradients on the representation route, detached on the head route | `hier_engine.py` (`z0` vs. `z0.detach()`) | `ctx = cat([eps.detach(), eps])` | Same contract. |
| `csem_w` / `terminal_kl_w` | 0.05 / 0.60 | **primary:** λ_CSEM = λ_K = kl_coeff (canonical) | Sweep weights are relative to per-pixel MSE + LPIPS + GAN and don't transfer numerically. LSGM's natural scale is the NELBO. The ratio λ_K/λ_CSEM = 12 is dimensionless and is used in ablation `B1-C2H`. |
| T_K / T | 1.05 / 1.35 (OU) | **primary:** t_K = T = 1 (VPSDE, LSGM's horizon). **Ablation:** t_K = **0.4544**, with T = 1 | 0.4544 matches the sweep's α_K² = e^{-2.1} = 0.1225 under LSGM's VPSDE. The sweep's T corresponds to VPSDE t = 0.516, but the sampling horizon stays LSGM's. |
| Terminal KL 𝒦 | `kl_reg_type terminal` | `--csem_terminal` (on) | At t_K = 1 it is about 4×10⁻⁵ × moments: harmless, and kept for exactness. |
| B (band term) | not in the zip | `--csem_band` (on) | Required for parity at `time_eps = 0.01` (§0, item 7). |
| OU-partial mean anchor | `ou_partial` at T_K | not ported in primary (≈ identity at t_K = 1); optional with the t_K = 0.4544 ablation | It changes the encoder parameterization, so it doesn't count as minimal surgery. |
| `t_min`, log-t grid, 1000 nodes | 3e-5, OU | LSGM `time_eps = 0.01`, continuous VPSDE with importance sampling | LSGM setup kept; B closes the cutoff gap. |
| Factored head, `aux_head_w` | on, 0.0025 | not ported (mixed score kept) | LSGM architecture parity. |
| VAE / head learning rates, AdamW (0.9, 0.95), norm-1 joint clip, EMA 0.9997 | 2.5e-4 / 1e-4 … | LSGM: Adam 1e-4 → 1e-5 / 1e-4 constant, eps 1e-3 / 1e-4, no clip, EMA 0.9999 | User requirement: LSGM optimizers. |
| LPIPS 0.85, GAN 0.0025, time-dependent decoding | on | **off** (LSGM's DML likelihood decoder) | These directly change FID; they would break the comparison. |
| Class labels, CFG 2.5/3.0, label dropout 0.1 | on | **off** (unconditional) | LSGM's CIFAR-10 is unconditional. |
| Seed | 42 | 42 in every cell | Paired arms share seeds and stage-1 checkpoints. |

---

## 6. Run matrix

### 6.1 Cells

Phase A trains the stage-1 VAEs, Phase B the stage-2 priors, and Phase C evaluates. Every Phase B cell has a matching Phase C cell `C·` that runs §3.3 on its `checkpoint.pt`.

| Cell | Phase | Backbone | Objective | Prior | Extra flags | Purpose |
|---|---|---|---|---|---|---|
| A0 | A | 4×45, nf=2 | NVAE | N(0, I) | — | reproduction backbone |
| A1 / A2 / A4 | A | 1×180 / 2×90 / 4×45, nf=0 | NVAE | N(0, I) | — | shared stage 1 |
| **B0** | B | A0 | LSGM | joint | — | **reproduction gate**: expect FID ≈ 5.2–5.4 (paper: 5.19 / 5.39) |
| **B1-L** | B | A1 | LSGM | joint | — | baseline, G=1 |
| **B1-C** | B | A1 | CSEM | flat | `--csem` | **estimator-only comparison** |
| **B2-L** | B | A2 | LSGM | joint | — | baseline, G=2 |
| **B2-C** | B | A2 | CSEM | cascade | `--csem --prior_factorization cascade` | hierarchical CSEM |
| **B4-L** | B | A4 | LSGM | joint | — | baseline, G=4 |
| **B4-C** | B | A4 | CSEM | cascade | `--csem --prior_factorization cascade` | hierarchical CSEM |
| B2-LC | B | A2 | LSGM | cascade | `--prior_factorization cascade` | Tier 2: separates factorization from estimator |
| B4-LC | B | A4 | LSGM | cascade | `--prior_factorization cascade` | Tier 2 |
| B1-Cnb | B | A1 | CSEM | flat | `--csem --no_csem_band` | Tier 3: tests the small-time hole at `time_eps = 0.01` |
| B1-C2H | B | A1 | CSEM | flat | `--csem --rep_horizon 0.4544 --csem_lambda_K 12` | Tier 3: transfer of the sweep's two-horizon recipe |

Tier 1 is A0–A4, B0, the six B·-L/B·-C cells, and their C cells: 4 stage-1 runs and 7 stage-2 runs. Tier 2 adds 2 runs and Tier 3 another 2.

### 6.2 What each comparison isolates

- **B1-C vs. B1-L: the estimator alone.** The 1-group LSGM objective is −H + CE_DSM over [ε, 1]. Its population value is CSEM_[ε,1] + B_ε − h(Z_1|X) + const. The CSEM arm's is CSEM_[ε,1] + B_ε + 𝒦_1. They differ by CE(q_1(·|x) ‖ γ) − const = ½ α_1² (E‖Z₀‖² − D), with α_1² = 4.3×10⁻⁵, so the population objectives are effectively identical. Any difference comes from the Rao–Blackwellized estimator. That is the clean test of CSEM's main claim.
- **B{2,4}-C vs. B{2,4}-LC:** estimator at a fixed cascade factorization.
- **B{2,4}-LC vs. B{2,4}-L:** joint vs. cascade prior under LSGM's objective.
- **B{2,4}-C vs. B{2,4}-L:** the headline comparison.
- **Trends with G within each family:** the hierarchy question.
- **B0:** confirms the port, hardware and torch version reproduce LSGM before any comparison is trusted.
- **B1-Cnb vs. B1-C:** whether the encoder exploits the cutoff hole. Expect `below_i` to rise without B.

### 6.3 Cost (rough; measure in the smoke test)

On one GH200, stage 2 at batch 128 is an NCSN++ forward/backward on 256 latents (p and q batches) plus an NVAE forward/backward. That plausibly takes about 0.2–0.4 s per iteration × 391 iterations per epoch, or roughly 1–2 days for 1000 epochs: 1–2 walltime segments. Stage 1 is plausibly under a day. Phase C (50k ODE samples at about 120–140 NFE, LSGM's App. H.3) takes a few GPU-hours per cell. Cascades cost about the same FLOPs at matched parameters but G× the sequential latency.

---

## 7. Validation (most important first)

1. **Reproduction gate (B0).** If B0's FID lands well outside 5.2–5.4, stop and debug the port (torch 2.x, the SyncBN patch, the JIT ops, the single-GPU batch mapping) before reading any CSEM result.
2. **Gradient-routing assertions (CUDA smoke, per arm):**
   - `p_loss` has no gradient path to any VAE parameter (`torch.autograd.grad(..., allow_unused=True)` all None or zero).
   - `q_loss` reaches the encoder samplers, the combiner cells, and (for cascades) the decoder-state path that carries ancestors.
   - The prior's final gradients equal the gradients of `p_loss` alone.
   - LSGM-joint first-step losses are identical with and without the patch.
3. **Exact-target checks (offline, numeric):**
   - Monte Carlo E[noise | eps_t, x] matches `csem_eps_target`.
   - With nf=0, `eps` equals `mu_q + sigma_q * rho`.
   - `vae.decompose_eps(cat(all_eps))` equals `all_eps`.
   - The B bound holds on random Gaussian channels; it held over 20k random cases in an earlier check.
   - B → 0 for σ_q ≫ 0.045.
4. **Cascade sampling and NLL checks:**
   - Group g's ODE is conditioned on exactly the earlier generated groups (call-order test, as in the zip's `sampling_checks`).
   - `cascade_ode_nll` with G=1 equals `compute_ode_nll`.
   - Sampling with a provided noise bank is reproducible.
5. **Configuration parity:** the LSGM command lines built for paired cells differ only in the allowed CSEM flags, and every cell's stage-2 command points at the right stage-1 checkpoint.
6. **Provenance:** vendored files hash to upstream except the patch list, and the patch touches only the functions listed in §4.0. This mirrors the zip's `source_checks`.

---

## 8. Sweep and cluster integration (following the zip)

### 8.1 Layout

```
lsgm_csem_cifar_blocks/
  README.md              # run procedure, as in the zip
  paper.py               # validate | prepare-data | prepare-fid | submit | run | smoke | report
  job.slurm              # zip's header + CUDA/JIT env; runs smoke then run
  manifest.csv           # cells A0-A4, B*, C*
  paper_io.py            # copied UNCHANGED from the zip (locks, prepare_dataset, atomic CSV)
  tests.py               # §7 offline tests + torch_smoke
  provenance.json        # LSGM commit, upstream/patched file hashes, zip file hashes
  patches/lsgm_csem.patch
  lsgm/                  # vendored NVlabs/LSGM @ 5eae2f3 with the patch applied (+ csem_lsgm.py)
  data_cache/ fid_stats/ results/<cell>/ status/ slurm_logs/ compiled/ failed_attempts/
```

### 8.2 Manifest schema

`cell_id, phase, family, groups, latent_per_group, num_nf, vae_cell, objective (lsgm|csem), prior_factorization, csem_band, csem_terminal, rep_horizon, csem_lambda_K, dae_channels_per_head, epochs, batch_size, seed, result_name, notes`

`paper.py` turns each row into an LSGM command line with a pure function: a fixed stage-specific base flag set (Appendix) plus the row's arm flags. The scientific columns are hashed into `config_hash`, as in the zip.

### 8.3 `job.slurm` (diff from the zip)

Keep `#SBATCH -p gh -N 1 -n 1 -t 48:00:00`, the module and venv lines, and the CUDA visibility check. Add:

```bash
module load cuda/12.x                                  # nvcc for score_sde/op JIT (match the venv's torch CUDA)
export TORCH_EXTENSIONS_DIR="$SCRATCH/torch_extensions/lsgm"
export PYTHONPATH="$BASEDIR/lsgm:${PYTHONPATH:-}"
"$PY" -u paper.py smoke --cell-id "$CELL_ID" --device cuda   # compiles ops, runs 5 steps, routing asserts
"$PY" -u paper.py run "${ARGS[@]}"                        # ARGS as in the zip: --cell-id, --base-dir, --resume, ...
```

LSGM's single-process path (`--num_process_per_node 1`) calls `utils.init_processes` with NCCL, world size 1, `MASTER_ADDR` 127.0.0.1 and the hard-coded port 6020. That is fine on Vista's exclusive `gh` nodes. Global batch is mapped to `--batch_size` on one GPU. NVAE uses SyncBatchNorm, so BatchNorm statistics match the original multi-GPU global batch.

### 8.4 `paper.py` behavior (same contract as the zip)

- **`run --cell-id`:**
  - Takes the status lock and refuses to start if the cell is completed with a different `config_hash`.
  - For phase B/C, refuses unless the referenced `vae_cell` (or `B` cell) has `returncode == 0`.
  - Builds the command line. If `results/<cell>/checkpoint.pt` exists, appends `--cont_training` (LSGM's resume, which restores optimizer, scheduler, GradScaler and spectral-norm state but not RNG; note this in the README).
    - Phase B writes `checkpoint.pt` every 10 epochs, so resume loses at most 10 epochs.
    - Phase A writes it **only when validation improves**, so a Phase A resume can roll back to the last improvement. Give Phase A enough walltime to finish in one segment, or accept the rollback, the same way for every A cell.
  - Runs it as a **subprocess** with `cwd=lsgm/`, writes the status JSON with started/finished times, the job ID and the return code, then validates outputs: final epoch reached, `checkpoint.pt` present, eval CSV rows present.
- **`submit [--phase A|B|C] [--cells …] [--dry-run] [--restart-incomplete]`:** the zip's logic (skip completed or active jobs, archive failures to `failed_attempts/`) plus **phase gating** (B cells submit only once their A cell is complete; C cells only once their B cell is). The zip deliberately avoids `--dependency=`; keep that.
- **`prepare-data`:** `paper_io.prepare_dataset(torchvision.datasets.CIFAR10, cache, 'CIFAR')`. LSGM gets `--data <cache>/CIFAR`.
- **`prepare-fid`:** run once on a node with internet access to fetch the Inception weights into `fid_stats/`. Runs `PYTHONPATH=lsgm python lsgm/scripts/precompute_fid_statistics.py --dataset cifar10 --data <cache>/CIFAR --fid_dir fid_stats --batch_size 256` → `fid_stats/cifar10.npz`.
- **`report`:** reads every cell's eval CSV and Phase C results, then writes `compiled/latest.csv` and `compiled/curves.csv` plus FID-vs-epoch PNG/PDF plots, keyed by (groups, objective, factorization). Missing values stay missing, as in the zip.

---

## 9. What to report

Per Phase C cell:
- 50k ODE FID against training statistics
- ODE NLL in bpd
- NFE, total and per head
- Sampling seconds per 50k
- Prior parameter count, total and per head
- Training seconds per iteration

Per group:
- Mean of `rep`, 𝒦 and B, and the fractions below the B floor
- σ_q quantiles
- Mixing coefficients
- Number of active units (LSGM's `kl_active_step` uses a per-channel KL threshold of 0.1)

Headline tables:
1. **G=1:** B1-L vs. B1-C, the estimator effect.
2. **G=2 and G=4:** L / LC / C for each, separating factorization from estimator.
3. **Calibration:** B0 against the published 5.19 / 5.39.

Report sequential latency next to FID for cascades. Keep LSGM-protocol FIDs separate from the zip's legacy, class-conditional FIDs.

---

## 10. Risks and open decisions (most important first)

- **R1. nf=0 changes LSGM's published 4-group backbone.** This is why the baselines are retrained (B·-L). B0 shows the cost of the change against the paper.
- **R2. Fairness of cascade capacity.** Parameter matching (352 / 256 channels) is the primary choice. If a cascade loses, check whether a per-head 512-channel variant (G× parameters) changes the conclusion before attributing it to factorization.
- **R3. `--cont_kl_anneal` in the ablation protocol is ambiguous.** G.5.1 anneals stage-1 KL to 1.0 but doesn't say whether stage 2 re-anneals from 0.7. Choose once (default: keep `lsgm3`'s flag) and apply it identically to all arms. It affects only the first 10% of iterations.
- **R4. The `time_eps = 0.01` cutoff is large.** B is essential for parity, and B1-Cnb tests it. If posteriors stay well above σ ≈ 0.045, B will be inactive. Report `below_i` either way.
- **R5. Torch 2.x / aarch64 port.** The SyncBN signature, JIT ops, `_accumulate`, SciPy `sqrtm`, the Apex fallback and AMP deprecation warnings are known. Unknown issues surface in the smoke test, and B0 is the numeric check.
- **R6. FID selection bias.** LSGM's `checkpoint_fid.pt` is selected on a 2,500-sample FID. Headline numbers use the final `checkpoint.pt` for every arm.
- **R7. Compute.** If 1000 epochs at batch 128 does not fit the walltime budget, cut epochs **uniformly** across all Tier-1 cells, keep B0 at 1000, and say so in the results.

---

## Appendix: base command lines

**Phase A** (`results/A{G}/`; G ∈ {1, 2, 4}; `A0` uses `--num_latent_per_group 45 --num_groups_per_scale 4 --num_nf 2`):

```bash
python train_vae.py --data $CACHE/CIFAR --root $RESULTS --save A{G} --dataset cifar10 \
  --num_channels_enc 256 --num_channels_dec 256 --num_postprocess_cells 3 --num_preprocess_cells 3 \
  --num_latent_scales 1 --num_cell_per_cond_enc 3 --num_cell_per_cond_dec 3 --num_preprocess_blocks 1 \
  --num_postprocess_blocks 1 --num_latent_per_group $((180/G)) --num_groups_per_scale {G} \
  --epochs 400 --batch_size 256 --weight_decay_norm 1e-2 --num_nf 0 --kl_anneal_portion 1.0 \
  --kl_max_coeff 1.0 --channel_mult 1 2 --seed 42 --arch_instance res_bnswish \
  --num_process_per_node 1 --use_se
```

**Phase B** (`results/B…/`; append the arm flags from §6.1):

```bash
python train_vada.py --fid_dir $FID --data $CACHE/CIFAR --root $RESULTS --save B{cell} \
  --vae_checkpoint $RESULTS/A{G}/checkpoint.pt --train_vae --apply_sqrt2_res --fir --cont_kl_anneal \
  --dae_arch ncsnpp --embedding_scale 1000 --dataset cifar10 --learning_rate_dae 1e-4 \
  --learning_rate_min_dae 1e-4 --epochs 1000 --dropout 0.2 --batch_size 128 --num_channels_dae 512 \
  --num_scales_dae 3 --num_cell_per_scale_dae 2 --sde_type vpsde --beta_start 0.1 --beta_end 20.0 \
  --sigma2_0 0.0 --weight_decay_norm_dae 1e-2 --weight_decay_norm_vae 1e-2 --time_eps 0.01 \
  --train_ode_eps 1e-6 --eval_ode_eps 1e-6 --train_ode_solver_tol 1e-5 --eval_ode_solver_tol 1e-5 \
  --iw_sample_p drop_all_iw --iw_sample_q ll_iw --num_process_per_node 1 --use_se --seed 42 \
  --skip_final_eval --eval_csv $RESULTS/B{cell}/eval_in_progress.csv
```

**Phase C:**

```bash
python evaluate_vada.py --data $CACHE/CIFAR --root $RESULTS --save C{cell} --eval_mode evaluate \
  --checkpoint $RESULTS/B{cell}/checkpoint.pt --fid_dir $FID --num_process_per_node 1 \
  --fid_ode_eval --nll_ode_eval --num_fid_samples 50000 --ode_eps 1e-6 --ode_solver_tol 1e-5 --batch_size 128
```

*Inferred rather than stated in the paper:*
- the VPSDE `time_eps = 0.01` for the ablation protocol (taken from `lsgm1`/`lsgm2`)
- the group count of the G.5.1/G.5.2 ablation model (stated only as "the best-NLL setup", which has 4 groups)
- the width and flows of G.5.3's single group
- the w_re assignment of the 4.99 entry in Table 6

Everything else is taken from the code at the pinned commit or from the LSGM README and supplement.
