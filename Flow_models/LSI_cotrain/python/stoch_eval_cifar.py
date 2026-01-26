# ============================
# Stochastic Heun-PC evaluation cell (MNIST/FMNIST/EMNIST/KMNIST)
# Compatible with batch_norm_lsi.py
# ============================

# ===========================================================================
# Standard Library Imports
# ===========================================================================
from __future__ import annotations
import os
import math
import torch
from torchvision import utils as tv_utils


# ---------------------------------------------------------------------------
# Imports & Checks
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs): return iterable

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available, perceptual loss/metrics will be skipped")

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics not available, FID will be -1")



# ===========================================================================
# Imports from lsi_cotrain.py (batch_norm_lsi.py)
# ===========================================================================
from lsi_cotrain_cifar import (
    # Device & Utils
    default_device,
    
    # Math utilities
    get_ou_params,
    
    # Metrics
    compute_sw2,
    compute_diversity,
    compute_fid_from_features,
    compute_kid,
    compute_lsi_gap,
    
    # Feature extraction
    extract_inception_features,
    
    # Models
    VAE,
    UNetModel,
    
    # Data
    make_dataloaders,
    
    # Constants
    LPIPS_AVAILABLE,
)

import os, math
import torch
import torchvision
from torchvision import utils as tv_utils
# ------------------------------------------------------------
# New sampler (does NOT overwrite your existing UniversalSampler)
# ------------------------------------------------------------
class UniversalSamplerV2:
    """
    Adds a true stochastic reverse-time SDE sampler:
      - method = "heun_pc_stoch": stochastic Heun predictor-corrector (strong order ~1 for additive noise)
    Keeps existing ODE methods for convenience.
    """
    def __init__(self, method="heun_pc_stoch", num_steps=20, t_min=1e-4, t_max=3.0):
        self.num_steps = int(num_steps)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.method = str(method)

    # ---- ODE drift (probability-flow) ----
    def get_ode_derivative(self, x, t, unet):
        B = x.shape[0]
        t_vec = t.expand(B)
        eps_pred = unet(x, t_vec)  # epsilon-pred network
        _, sigma = get_ou_params(t_vec.view(B, 1, 1, 1))
        inv_sigma = 1.0 / (sigma + 1e-8)
        # probability-flow ODE drift for OU
        return -x + inv_sigma * eps_pred

    # ---- Reverse-time SDE drift ----
    def get_rev_sde_drift(self, x, t, unet):
        """
        Forward OU: dx = -x dt + sqrt(2) dW
        Reverse-time SDE drift: b_rev = f - g^2 * score = -x - 2*score
        With epsilon parameterization: score = -(1/sigma) * eps  =>  -2*score = + 2*(1/sigma)*eps
        Hence b_rev = -x + 2*(1/sigma)*eps_pred.
        """
        B = x.shape[0]
        t_vec = t.expand(B)
        eps_pred = unet(x, t_vec)
        _, sigma = get_ou_params(t_vec.view(B, 1, 1, 1))
        inv_sigma = 1.0 / (sigma + 1e-8)
        return -x + 2.0 * inv_sigma * eps_pred

    def step_euler_ode(self, x, t_curr, t_next, unet):
        dt = t_next - t_curr
        d_curr = self.get_ode_derivative(x, t_curr, unet)
        return x + dt * d_curr

    def step_rk4_ode(self, x, t_curr, t_next, unet):
        dt = t_next - t_curr
        half_dt = dt * 0.5
        t_half = t_curr + half_dt

        k1 = self.get_ode_derivative(x, t_curr, unet)
        k2 = self.get_ode_derivative(x + half_dt * k1, t_half, unet)
        k3 = self.get_ode_derivative(x + half_dt * k2, t_half, unet)
        k4 = self.get_ode_derivative(x + dt * k3, t_next, unet)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # ---- New: stochastic Heun predictor-corrector for reverse-time SDE ----
    def step_heun_pc_stoch(self, x, t_curr, t_next, unet, generator=None):
        """
        Stochastic Heun / PC scheme for additive noise SDE:
          x_hat = x + dt*b(x,t) + G*dW
          x_new = x + 0.5*dt*(b(x,t) + b(x_hat,t_next)) + G*dW
        Uses SAME noise increment dW in predictor and corrector.

        Here: G = sqrt(2) I (OU diffusion); dt is negative since we integrate t_max -> t_min.
        """
        dt = t_next - t_curr                         # negative
        dt_abs = torch.abs(dt).clamp_min(1e-12)       # scalar tensor
        # dW ~ N(0, dt_abs I), and G = sqrt(2)
        if generator is None:
            noise = torch.randn_like(x)
        else:
            noise = torch.randn(x.shape, device=x.device, generator=generator)
        dW = torch.sqrt(2.0 * dt_abs) * noise

        b_curr = self.get_rev_sde_drift(x, t_curr, unet)
        x_hat = x + dt * b_curr + dW

        b_next = self.get_rev_sde_drift(x_hat, t_next, unet)
        x_new = x + 0.5 * dt * (b_curr + b_next) + dW
        return x_new

    def sample(self, unet, shape=None, device=None, x_init=None, generator=None):
        unet.eval()
        if x_init is None:
            assert shape is not None and device is not None
            x = torch.randn(shape, device=device, generator=generator)
        else:
            x = x_init
        device = x.device

        # same log time grid as your current sampler
        ts = torch.logspace(
            math.log10(self.t_max),
            math.log10(self.t_min),
            self.num_steps + 1,
            device=device
        )

        for i in range(self.num_steps):
            t_curr = ts[i]
            t_next = ts[i + 1]

            if self.method == "rk4_ode":
                x = self.step_rk4_ode(x, t_curr, t_next, unet)
            elif self.method == "euler_ode":
                x = self.step_euler_ode(x, t_curr, t_next, unet)
            elif self.method == "heun_pc_stoch":
                x = self.step_heun_pc_stoch(x, t_curr, t_next, unet, generator=generator)
            else:
                raise ValueError(f"Unknown method={self.method}")

        return x


# ------------------------------------------------------------
# New evaluation function using heun_pc_stoch at 20/40/80 steps
# (does NOT overwrite your existing evaluate_current_state)
# ------------------------------------------------------------
def evaluate_current_state_stoch(
    epoch_idx,
    prefix,
    vae,
    unet,
    loader,
    cfg,
    device,
    lpips_fn,
    fixed_noise_bank=None,
    fixed_posterior_eps_bank_A=None,
    fixed_posterior_eps_bank_B=None,
    fixed_sw2_theta=None,
    results_dir=None,
    also_run_baselines=False,   # set True if you also want rk4_ode@10 alongside stoch runs
):
    """Full test-set evaluation with stochastic Heun-PC sampler at multiple step counts."""
    print(f"\n--- Evaluation (stoch): {prefix} @ Ep {epoch_idx} ---")
    vae.eval()
    if unet is not None:
        unet.eval()

    target_count = len(loader.dataset)
    bs = cfg["batch_size"]
    latent_shape = (cfg["latent_channels"], 8, 8)
    sw2_nproj = int(cfg.get("sw2_n_projections", 1000))

    # bank shape checks (same as your original)
    if fixed_noise_bank is not None:
        assert fixed_noise_bank.shape[0] >= target_count
        assert tuple(fixed_noise_bank.shape[1:]) == latent_shape
    if fixed_posterior_eps_bank_A is not None:
        assert fixed_posterior_eps_bank_A.shape[0] >= target_count
        assert tuple(fixed_posterior_eps_bank_A.shape[1:]) == latent_shape
    if fixed_posterior_eps_bank_B is not None:
        assert fixed_posterior_eps_bank_B.shape[0] >= target_count
        assert tuple(fixed_posterior_eps_bank_B.shape[1:]) == latent_shape

    real_latents_A, real_latents_B, real_imgs = [], [], []
    encoder_mus, encoder_logvars = [], []
    bank_idx = 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            mu, logvar = vae.encode(x)
            std = torch.exp(0.5 * logvar)
            bsz = x.shape[0]

            encoder_mus.append(mu.cpu())
            encoder_logvars.append(logvar.cpu())

            epsA = fixed_posterior_eps_bank_A[bank_idx:bank_idx + bsz].to(device) \
                   if fixed_posterior_eps_bank_A is not None else torch.randn_like(std)
            zA = mu + std * epsA
            real_latents_A.append(zA.cpu())

            if fixed_posterior_eps_bank_B is not None:
                epsB = fixed_posterior_eps_bank_B[bank_idx:bank_idx + bsz].to(device)
                real_latents_B.append((mu + std * epsB).cpu())

            real_imgs.append(x.cpu())
            bank_idx += bsz
            if bank_idx >= target_count:
                break

    real_latents_A = torch.cat(real_latents_A, 0)[:target_count]
    real_imgs = torch.cat(real_imgs, 0)[:target_count]
    encoder_mus = torch.cat(encoder_mus, 0)[:target_count]
    encoder_logvars = torch.cat(encoder_logvars, 0)[:target_count]
    real_flat_A = real_latents_A.view(target_count, -1).to(device)

    if fixed_posterior_eps_bank_B is not None:
        real_latents_B = torch.cat(real_latents_B, 0)[:target_count]
        real_flat_B = real_latents_B.view(target_count, -1).to(device)
    else:
        real_flat_B = None

    print("  Extracting Inception features...")
    real_features, inception_model = extract_inception_features(
        real_imgs, device, batch_size=cfg.get("fid_batch_size", bs)
    )
    real_features = real_features.to(device)

    if unet is not None:
        lsi_gap_unet = compute_lsi_gap(
            unet, encoder_mus, encoder_logvars, cfg, device,
            num_samples=min(5000, target_count), num_time_points=50, batch_size=bs
        )
    else:
        lsi_gap_unet = 0.0

    # configs: always include recon; if unet provided include stochastic sampler at 20/40/80
    configs = [("VAE_Rec_eps", 0, "Recon (posterior z)")]
    if unet is not None:
        configs.extend([
            ("heun_pc_stoch", 40, "Stoch Heun-PC (40)"),
            ("heun_pc_stoch", 80, "Stoch Heun-PC (80)"),
            ("heun_pc_stoch", 120, "Stoch Heun-PC (120)"),
        ])
        if also_run_baselines:
            configs.extend([
                ("rk4_ode", 10, "RK4 ODE (10)"),
            ])

    results = []

    for method, steps, desc in configs:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            if method == "VAE_Rec_eps":
                fake_imgs = torch.cat([
                    vae.decode(real_latents_A[i:i + bs].to(device)).cpu()
                    for i in range(0, len(real_latents_A), bs)
                ], 0)

                if real_flat_B is not None:
                    w2 = compute_sw2(real_flat_A, real_flat_B, n_projections=sw2_nproj, theta=fixed_sw2_theta)
                else:
                    perm = torch.randperm(real_flat_A.size(0), device=device)
                    half = real_flat_A.size(0) // 2
                    w2 = compute_sw2(
                        real_flat_A[perm[:half]],
                        real_flat_A[perm[half:2*half]],
                        n_projections=sw2_nproj,
                        theta=fixed_sw2_theta
                    )
                lsi_gap = 0.0

            else:
                sampler = UniversalSamplerV2(
                    method=method,
                    num_steps=steps,
                    t_min=cfg["t_min"],
                    t_max=cfg["t_max"],
                )
                fake_latents_list, fake_imgs_list = [], []

                for i in range(0, target_count, bs):
                    batch_sz = min(bs, target_count - i)
                    if fixed_noise_bank is not None:
                        xT = fixed_noise_bank[i:i + batch_sz].to(device)
                        z_gen = sampler.sample(unet, x_init=xT)
                    else:
                        z_gen = sampler.sample(unet, shape=(batch_sz, *latent_shape), device=device)
                    fake_latents_list.append(z_gen.cpu())
                    fake_imgs_list.append(vae.decode(z_gen).cpu())

                fake_latents = torch.cat(fake_latents_list, 0)
                fake_imgs = torch.cat(fake_imgs_list, 0)
                fake_flat = fake_latents.view(fake_latents.shape[0], -1).to(device)
                w2 = compute_sw2(real_flat_A, fake_flat, n_projections=sw2_nproj, theta=fixed_sw2_theta)
                lsi_gap = lsi_gap_unet

        fake_features, inception_model = extract_inception_features(
            fake_imgs, device, batch_size=cfg.get("fid_batch_size", bs),
            inception_model=inception_model
        )
        fake_features = fake_features.to(device)

        fid = compute_fid_from_features(real_features, fake_features)
        kid = compute_kid(real_features, fake_features, num_subsets=100, subset_size=1000)
        div = compute_diversity(fake_imgs.to(device), lpips_fn) if LPIPS_AVAILABLE else 0.0

        results.append({
            "config": f"{method}@{steps}",
            "desc": desc,
            "fid": fid,
            "kid": kid,
            "w2": w2,
            "div": div,
            "lsi_gap": lsi_gap,
        })

        # save sample grids if requested
        if results_dir is not None:
            samples_dir = os.path.join(results_dir, "samples")
            os.makedirs(samples_dir, exist_ok=True)
            grid = tv_utils.make_grid(fake_imgs[:64], nrow=8, normalize=True, value_range=(-1, 1))
            tv_utils.save_image(grid, os.path.join(samples_dir, f"{prefix}_ep{epoch_idx}_{method}_{steps}.png"))

    print(f"\n  {'Config':<20} {'FID':>8} {'KID':>10} {'SW2':>12} {'Div':>8} {'LSI Gap':>10}")
    print("  " + "-"*70)
    for r in results:
        print(f"  {r['config']:<20} {r['fid']:>8.2f} {r['kid']:>10.6f} {r['w2']:>12.6f} {r['div']:>8.4f} {r['lsi_gap']:>10.4f}")

    result_dict = {}
    for r in results:
        cfg_name = r['config'].replace('@', '_')
        result_dict[f"fid_{cfg_name}"] = r['fid']
        result_dict[f"kid_{cfg_name}"] = r['kid']
        result_dict[f"sw2_{cfg_name}"] = r['w2']
        result_dict[f"div_{cfg_name}"] = r['div']
        result_dict[f"lsi_gap_{cfg_name}"] = r['lsi_gap']

    return result_dict


# ------------------------------------------------------------
# Main: load EMA checkpoints and run stochastic evaluation
# ------------------------------------------------------------
def _load_state_dict_flex(path, device):
    obj = torch.load(path, map_location=device)
    # support either raw state_dict or {"state_dict": ...}
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj

def run_stoch_eval_from_checkpoints(
    ckpt_dir="./checkpoints_cifar_compt",
    epoch_label=0,
    results_dir="stoch_eval_results",
    also_run_baselines=False,
    cfg = {},
    train_mode = 'cotrained'
):
    os.makedirs(results_dir, exist_ok=True)

    device = default_device()
    print("Device:", device)

    # data
    _, test_l = make_dataloaders(cfg["batch_size"], cfg.get("num_workers", 2))

    # models
    vae = VAE(latent_channels=cfg["latent_channels"], in_channels=3).to(device)
    unet_lsi = UNetModel(in_channels=cfg["latent_channels"]).to(device)
    unet_ctrl = UNetModel(in_channels=cfg["latent_channels"]).to(device)

    # load EMA weights (assumed saved as state_dicts)
    if train_mode == 'cotrained':
        vae_path = os.path.join(ckpt_dir, "vae_cotrained.pt")
    elif train_mode == 'indep':
        vae_path = os.path.join(ckpt_dir, "vae_pretrained.pt")
    else:
        raise ValueError(f"Unknown train_mode={train_mode}")
    lsi_path = os.path.join(ckpt_dir, "unet_lsi.pt")
    ctrl_path = os.path.join(ckpt_dir, "unet_control.pt")

    vae.load_state_dict(_load_state_dict_flex(vae_path, device))
    unet_lsi.load_state_dict(_load_state_dict_flex(lsi_path, device))
    unet_ctrl.load_state_dict(_load_state_dict_flex(ctrl_path, device))
    vae.eval(); unet_lsi.eval(); unet_ctrl.eval()

    # lpips
    lpips_fn = lpips.LPIPS(net='vgg').to(device) if LPIPS_AVAILABLE else None

    # fixed eval banks (recreate if not already defined)
    N_test = len(test_l.dataset)
    latent_shape = (cfg["latent_channels"], 8, 8)
    seed = int(cfg.get("seed", 0))

    g_noise = torch.Generator(device="cpu").manual_seed(seed + 12345)
    fixed_noise_bank = torch.randn((N_test, *latent_shape), generator=g_noise)

    g_postA = torch.Generator(device="cpu").manual_seed(seed + 54321)
    g_postB = torch.Generator(device="cpu").manual_seed(seed + 98765)
    fixed_posterior_eps_bank_A = torch.randn((N_test, *latent_shape), generator=g_postA)
    fixed_posterior_eps_bank_B = torch.randn((N_test, *latent_shape), generator=g_postB)

    D = cfg["latent_channels"] * 8 * 8
    K = int(cfg.get("sw2_n_projections", 1000))
    g_theta = torch.Generator(device="cpu").manual_seed(seed + 22222)
    theta = torch.randn((D, K), generator=g_theta)
    theta = theta / torch.norm(theta, dim=0, keepdim=True).clamp_min(1e-12)
    fixed_sw2_theta = theta

    # 1) VAE-only recon eval
    res_vae = evaluate_current_state_stoch(
        epoch_label, "VAE_ONLY", vae, None, test_l, cfg, device, lpips_fn,
        fixed_noise_bank=fixed_noise_bank,
        fixed_posterior_eps_bank_A=fixed_posterior_eps_bank_A,
        fixed_posterior_eps_bank_B=fixed_posterior_eps_bank_B,
        fixed_sw2_theta=fixed_sw2_theta,
        results_dir=results_dir,
        also_run_baselines=also_run_baselines,
    )

    # 2) LSI prior eval (stoch sampler @ 20/40/80)
    res_lsi = evaluate_current_state_stoch(
        epoch_label, "LSI_Diff", vae, unet_lsi, test_l, cfg, device, lpips_fn,
        fixed_noise_bank=fixed_noise_bank,
        fixed_posterior_eps_bank_A=fixed_posterior_eps_bank_A,
        fixed_posterior_eps_bank_B=fixed_posterior_eps_bank_B,
        fixed_sw2_theta=fixed_sw2_theta,
        results_dir=results_dir,
        also_run_baselines=also_run_baselines,
    )

    # 3) Control (Tweedie) prior eval (stoch sampler @ 20/40/80)
    res_ctrl = evaluate_current_state_stoch(
        epoch_label, "Ctrl_Diff", vae, unet_ctrl, test_l, cfg, device, lpips_fn,
        fixed_noise_bank=fixed_noise_bank,
        fixed_posterior_eps_bank_A=fixed_posterior_eps_bank_A,
        fixed_posterior_eps_bank_B=fixed_posterior_eps_bank_B,
        fixed_sw2_theta=fixed_sw2_theta,
        results_dir=results_dir,
        also_run_baselines=also_run_baselines,
    )

    return {"vae": res_vae, "lsi": res_lsi, "ctrl": res_ctrl}

def main():
      cfg = {
        "batch_size": 128,
        "num_workers": 2,
        "use_latent_norm": True,
        "kl_reg_type": "norm",
        "score_w": 1.0,
        "lr_vae": 1e-3,
        "lr_ldm": 2e-4,
        "lr_refine": 7e-5,
        "epochs_vae": 300,
        "epochs_refine": 100,
        "latent_channels": 4,  # Keeping 4 channels even for grayscale to test capacity
        "kl_w": 1e-4,
        "stiff_w": 1e-4,
        "score_w_vae": 0.4,
        "perc_w": 1.0,
        "t_min": 2e-5,
        "t_max": 2.0,
        "ckpt_dir": "checkpoints_cifar_grayscale",
        "seed": 42,
        "use_fixed_eval_banks": True,
        "sw2_n_projections": 1000,
        "load_from_checkpoint": False,
        "eval_freq": 10,
    }
  # --------- run it ----------
  # Assumes your checkpoints are in ./checkpoints_cifar_compt/{vae_cotrained.pt, unet_lsi.pt, unet_control.pt}
  stoch_results = run_stoch_eval_from_checkpoints(
        ckpt_dir="./run_results/checkpoints",
        epoch_label=999,                # just a label for filenames/logs
        results_dir="stoch_eval_results",
        also_run_baselines=True,        # set True to also compute rk4_ode@10 in the same run
        cfg=cfg,
        train_mode='cotrained',         # 'cotrained' or 'indep'
    )

    print("\n" + "="*60)
    print("STOCHASTIC EVALUATION RESULTS SUMMARY")
    print("="*60)
    for key, val in stoch_results.items():
        print(f"\n{key.upper()}:")
        for metric, value in val.items():
            print(f"  {metric}: {value:.6f}")

if __name__ == "__main__":
    main()
