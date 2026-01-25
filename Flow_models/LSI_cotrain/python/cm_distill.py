"""
Consistency Model Distillation for MNIST/FMNIST

This script trains consistency models via distillation from pre-trained score networks.
It compares LSI co-trained latent representations against standard two-stage Tweedie baselines.

Compatible with batch_norm_lsi.py - includes proper VAE initialization with use_norm parameter.

Place this cell directly below batch_norm_lsi.py in a Google Colab notebook.
All imports and definitions from batch_norm_lsi.py are assumed to be available.

Expected checkpoint structure:
    cotrained_nets/vae_cotrained.pt
    cotrained_nets/unet_lsi_cotrained.pt
    indep_trained_nets/vae_indep.pt
    indep_trained_nets/unet_control_indep.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import pandas as pd
from tqdm import tqdm
from torchvision import utils as tv_utils

# ===========================================================================
# 1. Consistency Model Wrappers & Helpers
# ===========================================================================

def get_consistency_schedule(num_steps, t_min=2e-5, t_max=2.0, rho=7.0, device="cpu"):
    """Karras time discretization schedule."""
    rho_inv = 1.0 / rho
    steps = torch.arange(num_steps, device=device, dtype=torch.float32)
    t_max_rho = t_max ** rho_inv
    t_min_rho = t_min ** rho_inv
    ts = (t_max_rho + steps / (num_steps - 1) * (t_min_rho - t_max_rho)) ** rho
    return ts


class BatchedEulerSolver:
    """Helper to perform ODE steps with batched time deltas."""
    def __init__(self, t_min=2e-5, t_max=2.0):
        self.t_min = t_min
        self.t_max = t_max

    def get_ode_derivative(self, x, t, unet):
        B = x.shape[0]
        t_view = t.view(B, 1, 1, 1)
        eps_pred = unet(x, t)
        # VP-ODE Drift: -x + (1/sigma)*eps
        _, sigma = get_ou_params(t_view)
        inv_sigma = 1.0 / (sigma + 1e-8)
        return -x + inv_sigma * eps_pred

    def step(self, x, t_from, t_to, unet, n_substeps=5):
        """Perform ODE integration from t_from to t_to with substeps."""
        # linearly interpolate times for substeps
        ts = torch.linspace(0, 1, n_substeps + 1, device=x.device)
        t_grid = t_from[:, None] + (t_to - t_from)[:, None] * ts[None, :]
        x_curr = x
        for k in range(n_substeps):
            t_k = t_grid[:, k]
            t_k1 = t_grid[:, k + 1]
            dt = t_k1 - t_k
            d = self.get_ode_derivative(x_curr, t_k, unet)
            x_curr = x_curr + dt.view(-1, 1, 1, 1) * d
        return x_curr


class Z0Denoiser(nn.Module):
    """Wraps epsilon-prediction UNet to output z0 directly: z0 = (z_t - sigma*eps)/alpha."""
    def __init__(self, base_unet):
        super().__init__()
        self.net = base_unet

    def forward(self, x, t):
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        eps_pred = self.net(x, t)
        t_view = t.view(-1, 1, 1, 1)
        alpha, sigma = get_ou_params(t_view)
        z0_pred = (x - sigma * eps_pred) / (alpha + 1e-8)
        return z0_pred


class ConsistencySampler:
    """Multistep sampler for the trained Consistency Model."""
    def __init__(self, model, t_min=2e-5, t_max=2.0):
        self.model = model
        self.t_min = t_min
        self.t_max = t_max

    def sample(self, shape, device, steps=1, generator=None):
        self.model.eval()
        x = torch.randn(shape, device=device, generator=generator)
        ts = get_consistency_schedule(steps + 1, t_min=self.t_min, t_max=self.t_max, device=device)
        with torch.no_grad():
            t_start = ts[0].expand(shape[0])
            x = self.model(x, t_start)
            for i in range(steps - 1):
                t_next = ts[i + 1]
                t_next_vec = t_next.expand(shape[0]).view(-1, 1, 1, 1)
                alpha, sigma = get_ou_params(t_next_vec)
                noise = torch.randn_like(x)
                z_re_noised = alpha * x + sigma * noise
                x = self.model(z_re_noised, t_next.expand(shape[0]))
        return x


# ===========================================================================
# 2. Evaluation Function for Consistency Models (MNIST/FMNIST compatible)
# ===========================================================================

def evaluate_consistency_mnist(
    exp_name,
    vae,
    cm_model,
    loader,
    device,
    cfg,
    results_dir,
    epoch_label=None,
    fid_model=None,
    use_lenet_fid=False,
):
    """
    Evaluate consistency model on MNIST/FMNIST datasets.

    Uses the same FID computation approach as evaluate_current_state from batch_norm_lsi.py:
    - LeNet features for MNIST/KMNIST/EMNIST
    - Inception features for FMNIST
    
    Args:
        exp_name: Experiment name for logging
        vae: VAE model (with use_norm parameter support)
        cm_model: Trained consistency model (Z0Denoiser)
        loader: Test dataloader
        device: torch device
        cfg: Configuration dictionary
        results_dir: Directory for saving results
        epoch_label: Optional epoch number for logging
        fid_model: Pre-loaded FID feature extractor (LeNet or None for Inception)
        use_lenet_fid: Whether to use LeNet (True) or Inception (False) for FID
    
    Returns:
        metrics_list: List of metric dictionaries
        fid_model: The FID model (possibly loaded/updated)
    """
    prefix = f"{exp_name}_ep{epoch_label}" if epoch_label is not None else exp_name
    print(f"\n--- Evaluating Consistency Model: {prefix} ---")

    sampler = ConsistencySampler(cm_model, t_min=cfg["t_min"], t_max=cfg["t_max"])
    vae.eval()
    cm_model.eval()

    configs = [(2, "2-Step"), (4, "4-Step"), (6, "6-Step")]
    metrics_list = []
    latent_shape = (cfg["latent_channels"], 8, 8)
    batch_size = cfg["batch_size"]

    # Extract Real Features once
    real_imgs = []
    for x, _ in loader:
        real_imgs.append(x.to(device))
        if len(real_imgs) * x.shape[0] >= 2000:
            break
    real_imgs = torch.cat(real_imgs)[:2000]

    # Extract features based on dataset type
    if use_lenet_fid and fid_model is not None:
        real_feats, fid_model = extract_lenet_features(
            real_imgs, device, batch_size=batch_size, lenet_model=fid_model
        )
    else:
        real_feats, fid_model = extract_inception_features(
            real_imgs, device, batch_size=batch_size, inception_model=fid_model
        )

    for steps, desc in configs:
        fake_imgs = []
        num_gen = 2000

        with torch.no_grad():
            for i in range(0, num_gen, batch_size):
                sz = min(batch_size, num_gen - i)
                z_gen = sampler.sample((sz, *latent_shape), device, steps=steps)
                img_gen = vae.decode(z_gen)
                fake_imgs.append(img_gen.cpu())

        fake_imgs = torch.cat(fake_imgs)

        # Extract features for generated images
        if use_lenet_fid and fid_model is not None:
            fake_feats, _ = extract_lenet_features(
                fake_imgs, device, batch_size=batch_size, lenet_model=fid_model
            )
        else:
            fake_feats, fid_model = extract_inception_features(
                fake_imgs, device, batch_size=batch_size, inception_model=fid_model
            )

        fid = compute_fid_from_features(real_feats, fake_feats)
        kid = compute_kid(real_feats, fake_feats)

        metrics_list.append({
            "Method": exp_name,
            "Epoch": epoch_label if epoch_label else "Final",
            "Steps": steps,
            "FID": fid,
            "KID": kid
        })

        # Save sample grid
        samples_dir = os.path.join(results_dir, "samples_cm")
        os.makedirs(samples_dir, exist_ok=True)
        grid = tv_utils.make_grid(fake_imgs[:64], nrow=8, normalize=True, value_range=(-1, 1))
        tv_utils.save_image(grid, os.path.join(samples_dir, f"{prefix}_steps{steps}.png"))

        print(f"   {desc} | FID: {fid:.2f} | KID: {kid:.5f}")

    return metrics_list, fid_model


# ===========================================================================
# 3. Training Logic
# ===========================================================================

def train_consistency_distillation_mnist(
    teacher_unet,
    vae,
    student_model,
    target_model,
    optimizer,
    loader,
    test_loader,
    cfg,
    device,
    exp_name,
    fid_model=None,
    use_lenet_fid=False,
):
    """
    Train a consistency model via distillation from a teacher score network.

    Args:
        teacher_unet: Pre-trained epsilon-prediction UNet (frozen)
        vae: Pre-trained VAE for encoding/decoding (frozen, supports use_norm)
        student_model: Z0Denoiser wrapper around student UNet (trained)
        target_model: EMA copy of student model
        optimizer: Optimizer for student_model
        loader: Training dataloader
        test_loader: Test dataloader for evaluation
        cfg: Configuration dict
        device: torch device
        exp_name: Name for logging
        fid_model: Pre-loaded FID feature extractor
        use_lenet_fid: Whether to use LeNet (True) or Inception (False) for FID
    
    Returns:
        loss_history: List of per-epoch average losses
        all_eval_metrics: List of evaluation metric dictionaries
        fid_model: The FID model (possibly updated)
    """
    teacher_unet.eval()
    vae.eval()
    student_model.train()
    target_model.train()

    solver = BatchedEulerSolver(t_min=cfg["t_min"], t_max=cfg["t_max"])
    ema_decay = cfg.get("ema_decay_cm", 0.99)
    loss_history = []
    all_eval_metrics = []

    N_discrete = cfg.get("N_discrete", 20)
    full_schedule = get_consistency_schedule(
        N_discrete, t_min=cfg["t_min"], t_max=cfg["t_max"], device=device
    )

    epochs_cm = cfg.get("epochs_cm", 90)
    eval_every = cfg.get("eval_every_cm", 30)

    print(f"\n--> Starting Consistency Distillation: {exp_name} ({epochs_cm} Epochs)")

    for ep in range(epochs_cm):
        epoch_loss = 0.0
        student_model.train()

        for x, _ in tqdm(loader, desc=f"CD {exp_name} Ep {ep+1}", leave=False):
            x = x.to(device)
            B = x.shape[0]

            with torch.no_grad():
                _, mu, logvar = vae(x)
                z0 = vae.reparameterize(mu, logvar)

            n = torch.randint(0, N_discrete - 1, (B,), device=device)
            t_current = full_schedule[n]
            t_next = full_schedule[n + 1]

            # Re-noise to current time
            t_curr_view = t_current.view(B, 1, 1, 1)
            alpha_cur, sigma_cur = get_ou_params(t_curr_view)
            noise = torch.randn_like(z0)
            z_current = alpha_cur * z0 + sigma_cur * noise

            # Teacher Step
            with torch.no_grad():
                z_teacher_next = solver.step(z_current, t_current, t_next, teacher_unet)

            # Student & Target Predictions
            pred_student = student_model(z_current, t_current)
            with torch.no_grad():
                pred_target = target_model(z_teacher_next, t_next)

            loss = F.mse_loss(pred_student, pred_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA Update
            with torch.no_grad():
                for p_stud, p_targ in zip(student_model.parameters(), target_model.parameters()):
                    p_targ.data.mul_(ema_decay).add_(p_stud.data, alpha=1 - ema_decay)

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"   Ep {ep+1} | CD Loss: {avg_loss:.5f}")

        # Evaluation checkpoint
        if (ep + 1) % eval_every == 0:
            metrics, fid_model = evaluate_consistency_mnist(
                exp_name, vae, target_model, test_loader, device, cfg,
                cfg["results_dir"], epoch_label=ep + 1,
                fid_model=fid_model, use_lenet_fid=use_lenet_fid
            )
            all_eval_metrics.extend(metrics)

            # Save checkpoint
            ckpt_dir = os.path.join(cfg["results_dir"], "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"{exp_name}_ep{ep+1}.pt")
            torch.save(target_model.state_dict(), ckpt_path)

    return loss_history, all_eval_metrics, fid_model


# ===========================================================================
# 4. Main Execution
# ===========================================================================

def run_consistency_experiment_mnist(dataset_key="FMNIST", use_latent_norm=False):
    """
    Run consistency model distillation experiment on MNIST/FMNIST.

    Compares:
    - CM_LSI: Consistency model distilled from LSI co-trained score network
    - CM_Control: Consistency model distilled from independent Tweedie score network

    Args:
        dataset_key: "MNIST", "FMNIST", "EMNIST", or "KMNIST"
        use_latent_norm: Whether VAE uses GroupNorm on latents (must match training)
    
    Returns:
        df: DataFrame with evaluation metrics
        loss_df: DataFrame with training loss history
    """
    cfg = {
        "dataset": dataset_key,
        "batch_size": 128,
        "num_workers": 2,
        "latent_channels": 2,  # MNIST/FMNIST uses 2 channels
        "t_min": 2e-5,
        "t_max": 2.0,
        "epochs_cm": 160,
        "lr_cm": 2e-4,
        "N_discrete": 20,
        "eval_every_cm": 40,
        "ema_decay_cm": 0.99,
        "results_dir": f"run_results_cm_{dataset_key.lower()}",
        "sw2_n_projections": 1000,
        "seed": 42,
        "use_latent_norm": use_latent_norm,  # Pass through to VAE
    }

    epochs_cm = cfg["epochs_cm"]

    device = default_device()
    print(f"Device: {device}")
    print(f"Dataset: {dataset_key}")
    print(f"Use Latent Norm: {use_latent_norm}")

    # Setup directories
    os.makedirs(cfg["results_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["results_dir"], "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(cfg["results_dir"], "samples_cm"), exist_ok=True)

    # Load data
    train_l, test_l, num_classes = make_dataloaders(
        cfg["batch_size"], cfg["num_workers"], dataset_key
    )

    # Setup LPIPS (optional, for sanity checks)
    lpips_fn = None
    if LPIPS_AVAILABLE:
        lpips_fn = lpips.LPIPS(net='vgg').to(device)

    # Setup FID model
    # For FMNIST: use Inception (use_lenet_fid=False)
    # For MNIST/others: use LeNet (use_lenet_fid=True)
    fid_model, use_lenet_fid = get_fid_model(
        dataset_key, train_l, num_classes, device, cfg["results_dir"]
    )

    # Fixed Banks for deterministic sanity check
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

    # ---------------------------------------------------------
    # EXPERIMENT A: LSI Co-trained
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT A: Consistency Distillation on LSI Co-trained Latent")
    print("=" * 60)

    # Initialize VAE with use_norm parameter (matching batch_norm_lsi.py)
    vae_cotrained = VAE(
        latent_channels=cfg["latent_channels"],
        use_norm=cfg["use_latent_norm"]
    ).to(device)
    teacher_lsi = UNetModel(in_channels=cfg["latent_channels"]).to(device)
    ckpt_dir_cot = "cotrained_nets"

    try:
        vae_cotrained.load_state_dict(
            torch.load(os.path.join(ckpt_dir_cot, "vae_cotrained.pt"), map_location=device)
        )
        teacher_lsi.load_state_dict(
            torch.load(os.path.join(ckpt_dir_cot, "unet_lsi_cotrained.pt"), map_location=device)
        )
        print("--> Loaded LSI Teachers successfully.")

        # SANITY CHECK: Evaluate teacher before distillation
        print("\n--- Sanity Check: LSI Teacher Performance ---")
        results_lsi_teacher = evaluate_current_state(
            0, "Sanity_LSI_Teacher", vae_cotrained, teacher_lsi, test_l, cfg, device, lpips_fn,
            fixed_noise_bank=fixed_noise_bank,
            fixed_posterior_eps_bank_A=fixed_posterior_eps_bank_A,
            fixed_posterior_eps_bank_B=fixed_posterior_eps_bank_B,
            fixed_sw2_theta=fixed_sw2_theta,
            results_dir=cfg["results_dir"],
            fid_model=fid_model,
            use_lenet_fid=use_lenet_fid,
        )
    except Exception as e:
        print(f"!! Warning: Could not load LSI checkpoints ({e}).")
        print("!! Please ensure cotrained_nets/vae_cotrained.pt and cotrained_nets/unet_lsi_cotrained.pt exist.")
        return None, None

    # Create student/target models for LSI
    student_unet_lsi = UNetModel(in_channels=cfg["latent_channels"]).to(device)
    student_unet_lsi.load_state_dict(teacher_lsi.state_dict())

    cm_lsi = Z0Denoiser(student_unet_lsi).to(device)
    target_lsi = copy.deepcopy(cm_lsi).to(device)
    opt_lsi = torch.optim.AdamW(cm_lsi.parameters(), lr=cfg["lr_cm"])

    loss_history_lsi, eval_metrics_lsi, fid_model = train_consistency_distillation_mnist(
        teacher_lsi, vae_cotrained, cm_lsi, target_lsi, opt_lsi,
        train_l, test_l, cfg, device, "CM_LSI",
        fid_model=fid_model, use_lenet_fid=use_lenet_fid
    )

    # ---------------------------------------------------------
    # EXPERIMENT B: Independent Control (Tweedie)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT B: Consistency Distillation on Independent Tweedie Latent")
    print("=" * 60)

    # Initialize VAE with use_norm parameter (matching batch_norm_lsi.py)
    vae_indep = VAE(
        latent_channels=cfg["latent_channels"],
        use_norm=cfg["use_latent_norm"]
    ).to(device)
    teacher_control = UNetModel(in_channels=cfg["latent_channels"]).to(device)
    indep_dir = "indep_trained_nets"

    try:
        vae_indep.load_state_dict(
            torch.load(os.path.join(indep_dir, "vae_indep.pt"), map_location=device)
        )
        teacher_control.load_state_dict(
            torch.load(os.path.join(indep_dir, "unet_control_indep.pt"), map_location=device)
        )
        print("--> Loaded Control Teachers successfully.")

        # SANITY CHECK: Evaluate teacher before distillation
        print("\n--- Sanity Check: Control Teacher Performance ---")
        results_ctrl_teacher = evaluate_current_state(
            0, "Sanity_Indep_Teacher", vae_indep, teacher_control, test_l, cfg, device, lpips_fn,
            fixed_noise_bank=fixed_noise_bank,
            fixed_posterior_eps_bank_A=fixed_posterior_eps_bank_A,
            fixed_posterior_eps_bank_B=fixed_posterior_eps_bank_B,
            fixed_sw2_theta=fixed_sw2_theta,
            results_dir=cfg["results_dir"],
            fid_model=fid_model,
            use_lenet_fid=use_lenet_fid,
        )
    except Exception as e:
        print(f"!! Warning: Could not load Indep Teachers ({e}). Using LSI weights as fallback.")
        vae_indep = vae_cotrained
        teacher_control = teacher_lsi

    # Create student/target models for Control
    student_unet_ctrl = UNetModel(in_channels=cfg["latent_channels"]).to(device)
    student_unet_ctrl.load_state_dict(teacher_control.state_dict())

    cm_ctrl = Z0Denoiser(student_unet_ctrl).to(device)
    target_ctrl = copy.deepcopy(cm_ctrl).to(device)
    opt_ctrl = torch.optim.AdamW(cm_ctrl.parameters(), lr=cfg["lr_cm"])

    loss_history_ctrl, eval_metrics_ctrl, fid_model = train_consistency_distillation_mnist(
        teacher_control, vae_indep, cm_ctrl, target_ctrl, opt_ctrl,
        train_l, test_l, cfg, device, "CM_Control",
        fid_model=fid_model, use_lenet_fid=use_lenet_fid
    )

    # ---------------------------------------------------------
    # Results Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"FINAL RESULTS: {epochs_cm} Epoch Consistency Model Training ({dataset_key})")
    print("=" * 60)

    all_metrics = eval_metrics_lsi + eval_metrics_ctrl
    df = pd.DataFrame(all_metrics)
    print(df)

    # Save results
    csv_path = os.path.join(cfg["results_dir"], "cm_training_trajectory.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n--> Saved metrics to {csv_path}")

    # Save loss histories
    loss_df = pd.DataFrame({
        "epoch": list(range(1, len(loss_history_lsi) + 1)),
        "CM_LSI_loss": loss_history_lsi,
        "CM_Control_loss": loss_history_ctrl
    })
    loss_path = os.path.join(cfg["results_dir"], "cm_loss_history.csv")
    loss_df.to_csv(loss_path, index=False)
    print(f"--> Saved loss history to {loss_path}")

    # Final evaluation summary
    print("\n" + "-" * 60)
    print("FINAL EVALUATION SUMMARY")
    print("-" * 60)
    
    # Get final epoch results
    final_lsi = [m for m in eval_metrics_lsi if m["Epoch"] == epochs_cm]
    final_ctrl = [m for m in eval_metrics_ctrl if m["Epoch"] == epochs_cm]
    
    if final_lsi:
        print("\nCM_LSI (Final):")
        for m in final_lsi:
            print(f"  {m['Steps']}-Step: FID={m['FID']:.2f}, KID={m['KID']:.5f}")
    
    if final_ctrl:
        print("\nCM_Control (Final):")
        for m in final_ctrl:
            print(f"  {m['Steps']}-Step: FID={m['FID']:.2f}, KID={m['KID']:.5f}")

    return df, loss_df


# ===========================================================================
# 5. Alternative: Run with Stochastic Evaluation (uses evaluate_current_state_stoch)
# ===========================================================================

def run_consistency_experiment_with_stoch_eval(dataset_key="FMNIST", use_latent_norm=False):
    """
    Same as run_consistency_experiment_mnist but also runs stochastic evaluation
    on the teacher networks before distillation for comprehensive comparison.
    
    Requires evaluate_current_state_stoch from stoch_eval_mnist_fmnist.py
    """
    cfg = {
        "dataset": dataset_key,
        "batch_size": 128,
        "num_workers": 2,
        "latent_channels": 2,
        "t_min": 2e-5,
        "t_max": 2.0,
        "epochs_cm": 160,
        "lr_cm": 2e-4,
        "N_discrete": 20,
        "eval_every_cm": 40,
        "ema_decay_cm": 0.99,
        "results_dir": f"run_results_cm_{dataset_key.lower()}",
        "sw2_n_projections": 1000,
        "seed": 42,
        "use_latent_norm": use_latent_norm,
    }

    device = default_device()
    print(f"Device: {device}")
    print(f"Dataset: {dataset_key}")

    # Setup directories
    os.makedirs(cfg["results_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["results_dir"], "checkpoints"), exist_ok=True)

    # Load data
    train_l, test_l, num_classes = make_dataloaders(
        cfg["batch_size"], cfg["num_workers"], dataset_key
    )

    # Setup FID model
    fid_model, use_lenet_fid = get_fid_model(
        dataset_key, train_l, num_classes, device, cfg["results_dir"]
    )

    # Setup LPIPS
    lpips_fn = lpips.LPIPS(net='vgg').to(device) if LPIPS_AVAILABLE else None

    # Fixed Banks
    N_test = len(test_l.dataset)
    latent_shape = (cfg["latent_channels"], 8, 8)
    seed = cfg["seed"]

    g_noise = torch.Generator(device="cpu").manual_seed(seed + 12345)
    fixed_noise_bank = torch.randn((N_test, *latent_shape), generator=g_noise)

    g_postA = torch.Generator(device="cpu").manual_seed(seed + 54321)
    g_postB = torch.Generator(device="cpu").manual_seed(seed + 98765)
    fixed_posterior_eps_bank_A = torch.randn((N_test, *latent_shape), generator=g_postA)
    fixed_posterior_eps_bank_B = torch.randn((N_test, *latent_shape), generator=g_postB)

    D = cfg["latent_channels"] * 8 * 8
    K = cfg["sw2_n_projections"]
    g_theta = torch.Generator(device="cpu").manual_seed(seed + 22222)
    theta = torch.randn((D, K), generator=g_theta)
    theta = theta / torch.norm(theta, dim=0, keepdim=True).clamp_min(1e-12)
    fixed_sw2_theta = theta

    # Load teachers and run stochastic eval
    vae_cotrained = VAE(latent_channels=cfg["latent_channels"], use_norm=cfg["use_latent_norm"]).to(device)
    teacher_lsi = UNetModel(in_channels=cfg["latent_channels"]).to(device)

    try:
        vae_cotrained.load_state_dict(torch.load("cotrained_nets/vae_cotrained.pt", map_location=device))
        teacher_lsi.load_state_dict(torch.load("cotrained_nets/unet_lsi_cotrained.pt", map_location=device))
        
        print("\n" + "=" * 60)
        print("STOCHASTIC EVALUATION: LSI Teacher (Before Distillation)")
        print("=" * 60)
        
        # Check if evaluate_current_state_stoch is available
        if 'evaluate_current_state_stoch' in dir():
            stoch_results_lsi = evaluate_current_state_stoch(
                0, "LSI_Teacher_Stoch", vae_cotrained, teacher_lsi, test_l, cfg, device, lpips_fn,
                fixed_noise_bank=fixed_noise_bank,
                fixed_posterior_eps_bank_A=fixed_posterior_eps_bank_A,
                fixed_posterior_eps_bank_B=fixed_posterior_eps_bank_B,
                fixed_sw2_theta=fixed_sw2_theta,
                results_dir=cfg["results_dir"],
                also_run_baselines=True,
                fid_model=fid_model,
                use_lenet_fid=use_lenet_fid,
            )
        else:
            print("!! evaluate_current_state_stoch not available. Run standard eval instead.")
            
    except Exception as e:
        print(f"!! Could not load LSI teacher: {e}")

    # Continue with consistency training...
    print("\n--> Proceeding to consistency model distillation...")
    return run_consistency_experiment_mnist(dataset_key, use_latent_norm)


# ===========================================================================
# 6. Entry Point
# ===========================================================================

if __name__ == "__main__":
    # Configuration
    DATASET = "FMNIST"  # Options: "MNIST", "FMNIST", "EMNIST", "KMNIST"
    USE_LATENT_NORM = False  # Set True if your VAE was trained with use_norm=True
    
    # Run the experiment
    df, loss_df = run_consistency_experiment_mnist(
        dataset_key=DATASET,
        use_latent_norm=USE_LATENT_NORM
    )
    
    if df is not None:
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)
        print(f"\nResults saved to: run_results_cm_{DATASET.lower()}/")
