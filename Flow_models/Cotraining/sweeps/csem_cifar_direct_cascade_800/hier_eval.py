"""CIFAR evaluation with learned conditional priors at every latent block.

Uses the baseline's legacy torchvision Inception features, test images, Gaussian
noise/label banks and metric functions. Evaluate full-prior generation separately
from posterior reconstruction and encoded-root/generated-descendant decoding.
No joint-Gaussian oracle initialization is asserted for the non-Gaussian joint.
"""
from __future__ import annotations
from pathlib import Path
import json
import time
import torch
import core
from suite import feature_sw2, fixed_metric_rng
from paper_io import sampler_configs
from hier_models import sample_hierarchy


def metric_suffix(method, steps, level, horizon):
    short = {'rk4_ode': 'rk4', 'heun_sde': 'heun'}[method]
    level_token = str(float(level)).replace('.', '_')
    time_token = f'{float(horizon):g}'.replace('.', 'p')
    return f'{short}_{int(steps)}_randtok_cfg{level_token}_temp1_initgaussianT{time_token}'


@torch.no_grad()
def conditional_score_gap(heads, mu, logvar, z0, cfg, device):
    """Factorwise CSEM risk in score units, evaluated on clean-parent contexts.

    This is a conditional-risk diagnostic, not the flat joint-score functional.
    It includes both intrinsic Gaussian-component variance and approximation
    error, as does a CSEM component-target residual.
    """
    N = min(len(z0), int(cfg.get('eval_lsi_gap_samples', 2500)))
    schedule = core.make_schedule(cfg, device)
    indices = torch.linspace(0, int(schedule['T'].item())-1,
                             int(cfg.get('eval_lsi_gap_time_points', 50)), device=device).round().long()
    bs = int(cfg['batch_size'])
    total, count = 0.0, 0
    for offset in range(0, N, bs):
        z = z0[offset:offset+bs].to(device)
        m = mu[offset:offset+bs].to(device)
        lv = logvar[offset:offset+bs].to(device)
        b = len(z)
        for idx in indices:
            t = schedule['times'][idx].expand(b).float()
            a = schedule['alpha'][idx]
            s = schedule['sigma'][idx]
            noisy = a*z + s*torch.randn_like(z)
            target = s*(noisy-a*m)/(a.square()*lv.exp()+s.square()+1e-8)
            pred = heads(noisy, t, None, context=z)
            risk = ((pred-target).square()/(s.square()+1e-8)).flatten(1).sum(1)
            total += risk.sum().item(); count += b
    return total/max(count, 1)


@torch.no_grad()
def evaluate_hierarchy(epoch_idx, prefix, vae, unet, loader, cfg, device, lpips_fn,
                       fixed_noise_bank=None, fixed_posterior_eps_bank_A=None,
                       fixed_posterior_eps_bank_B=None, fixed_sw2_theta=None,
                       results_dir=None, fid_model=None, use_lenet_fid=False,
                       oracle_reference_loader=None):
    core.validate_terminal_time_contract(cfg)
    if cfg['dataset'] != 'CIFAR' or use_lenet_fid:
        raise ValueError('This comparison is RGB CIFAR with the original Inception backend')
    vae.eval(); unet.eval()
    started = time.perf_counter()
    N = min(len(loader.dataset), int(cfg['eval_max_samples']))
    bs = int(cfg['batch_size'])
    decode_time = float(cfg['decode_time']) if cfg.get('decode_time') is not None else float(cfg['t_min'])
    out = Path(results_dir)/'samples'/f'epoch_{int(epoch_idx):04d}'
    out.mkdir(parents=True, exist_ok=True)
    rows = {}

    # Additional diagnostics and sampler-count changes cannot consume training
    # random draws. All checkpoint evaluations reuse fixed banks.
    with fixed_metric_rng(torch, int(cfg['seed'])+55555):
        images, labels, latents, mus, lvs, recs = [], [], [], [], [], []
        offset = 0
        for x, y in loader:
            b = min(len(x), N-offset)
            if b <= 0:
                break
            x = x[:b].to(device)
            eps = (fixed_posterior_eps_bank_A[offset:offset+b].to(device)
                   if fixed_posterior_eps_bank_A is not None else None)
            mu, lv, z = vae.encode_sample(x, eps=eps)
            t_dec = torch.full((b,), decode_time, device=device)
            recs.append(vae.decode(z, t=t_dec).cpu())
            images.append(x.cpu()); labels.append(y[:b].cpu())
            latents.append(z.cpu()); mus.append(mu.cpu()); lvs.append(lv.cpu())
            offset += b
        real = torch.cat(images); labels = torch.cat(labels)
        zpost = torch.cat(latents); mu = torch.cat(mus); lv = torch.cat(lvs)
        reconstructed = torch.cat(recs)
        if len(real) != N:
            raise RuntimeError('Evaluator did not collect the requested sample count')
        # Quantify posterior-child dependence, not merely generated FID.
        # The child here is the SAME direct Z2 used by the posterior and score.
        if cfg.get('hierarchy_variant') == 'direct_child_v1':
            m = min(128, N)
            original = zpost[:m].to(device)
            root = original[:, :unet.block_channels[0]]
            child = original[:, unet.block_channels[0]:]
            t_probe = torch.full((m,), decode_time, device=device)
            reference = reconstructed[:m].to(device)
            target = real[:m].to(device)
            def mse(a, b):
                return float((a-b).square().mean().item())
            baseline_mse = mse(reference, target)
            zero_rec = vae.decode(torch.cat((root, torch.zeros_like(child)), dim=1), t=t_probe)
            shuffle_rec = vae.decode(torch.cat((root, child.roll(1, 0)), dim=1), t=t_probe)
            noise_rec = vae.decode(torch.cat((root, torch.randn_like(child)), dim=1), t=t_probe)
            rows.update(child_use_probe_samples=m, child_use_posterior_mse=baseline_mse,
                        child_use_zero_mse=mse(zero_rec, target),
                        child_use_shuffle_mse=mse(shuffle_rec, target),
                        child_use_noise_mse=mse(noise_rec, target),
                        child_use_zero_delta=mse(zero_rec, reference),
                        child_use_shuffle_delta=mse(shuffle_rec, reference),
                        child_use_noise_delta=mse(noise_rec, reference))
            print(f'[CHILD USE] epoch={epoch_idx} posterior_mse={baseline_mse:.6f} '
                  f'zero_delta={rows["child_use_zero_delta"]:.6f} '
                  f'shuffle_delta={rows["child_use_shuffle_delta"]:.6f}', flush=True)
        feature_bs = int(cfg.get('fid_batch_size', bs))
        real_features, fid_model = core.extract_inception_features(real, device, feature_bs, fid_model)

        g = torch.Generator(device='cpu').manual_seed(int(cfg['seed'])+12345)
        noise = (fixed_noise_bank[:N] if fixed_noise_bank is not None else
                 torch.randn((N, sum(unet.block_channels), 8, 8), generator=g))
        g_labels = torch.Generator(device='cpu').manual_seed(int(cfg['seed'])+1337)
        gen_labels = torch.randint(0, int(cfg['num_classes']), (N,), generator=g_labels)

        def measure(fake, latent=None):
            fake_features, _ = core.extract_inception_features(fake, device, feature_bs, fid_model)
            values = {
                'fid': core.compute_fid_from_features(real_features, fake_features),
                'kid': core.compute_kid(real_features, fake_features,
                        num_subsets=int(cfg['kid_num_subsets']), subset_size=int(cfg['kid_subset_size'])),
                'feature_sw2': feature_sw2(real_features.to(device), fake_features.to(device), cfg, core.compute_sw2),
            }
            if latent is not None:
                values['sw2'] = core.compute_sw2(zpost.flatten(1).to(device), latent.flatten(1).to(device),
                                    n_projections=int(cfg['sw2_n_projections']), theta=fixed_sw2_theta)
            return values

        def save_panel(fake, name):
            core.tv_utils.save_image(((fake[:64]+1)/2).clamp(0, 1), str(out/f'{name}.png'), nrow=8)

        for key, value in measure(reconstructed, zpost).items():
            rows[f'{key}_vae_recon'] = float(value)
        save_panel(real, 'real'); save_panel(reconstructed, 'posterior_reconstruction')
        plan = [s for s in sampler_configs(cfg) if s.get('init_mode') == 'gaussian-T']
        for index, scfg in enumerate(plan):
            method, steps = scfg['method'], int(scfg['steps'])
            level = float(scfg['cfg_level'])
            suffix = metric_suffix(method, steps, level, cfg['t_max'])
            if device.type == 'cuda': torch.cuda.synchronize(device)
            sampling_started = time.perf_counter()
            generated, generated_latents = [], []
            for off in range(0, N, bs):
                start = noise[off:off+bs].to(device)
                y = gen_labels[off:off+bs].to(device)
                z = sample_hierarchy(unet, noise=start, y=y, cfg=cfg,
                                     method=method, steps=steps, cfg_scale=level)
                t_dec = torch.full((len(z),), decode_time, device=device)
                generated.append(vae.decode(z, t=t_dec).cpu()); generated_latents.append(z.cpu())
            fake, zfake = torch.cat(generated), torch.cat(generated_latents)
            if device.type == 'cuda': torch.cuda.synchronize(device)
            rows[f'sampling_seconds_{suffix}'] = time.perf_counter() - sampling_started
            for key, value in measure(fake, zfake).items():
                rows[f'{key}_{suffix}'] = float(value)
            save_panel(fake, suffix)
            print(f'[EVAL] epoch={epoch_idx} blocks={len(unet.block_channels)} {suffix} '
                  f'FID={rows["fid_"+suffix]:.6f}', flush=True)
            if method == 'rk4_ode' and steps == 20:
                descendants = []
                for off in range(0, N, bs):
                    root = zpost[off:off+bs, :unet.block_channels[0]].to(device)
                    z = sample_hierarchy(unet, noise=noise[off:off+bs].to(device),
                                         y=labels[off:off+bs].to(device), cfg=cfg,
                                         method=method, steps=steps, cfg_scale=level, root=root)
                    descendants.append(vae.decode(z, t=torch.full((len(z),), decode_time, device=device)).cpu())
                descendants = torch.cat(descendants)
                diag = f'encoded_root_generated_children_cfg{str(level).replace(".", "_")}'
                for key, value in measure(descendants).items():
                    rows[f'{key}_{diag}'] = float(value)
                save_panel(descendants, diag)
                if level == float(cfg['cfg_eval_scale']):
                    # The second draw uses the same roots with independent child
                    # starts. Generation need not reconstruct the selected x.
                    m=min(128,N);rr=zpost[:m,:unet.block_channels[0]].to(device)
                    starts=torch.randn_like(noise[:m]).to(device)
                    z_second=sample_hierarchy(unet,noise=starts,y=labels[:m].to(device),cfg=cfg,
                                              method=method,steps=steps,cfg_scale=level,root=rr)
                    second=vae.decode(z_second,t=torch.full((m,),decode_time,device=device)).cpu()
                    rows['child_generated_draws_mse']=float((second-descendants[:m]).square().mean())
                    rows['child_generated_vs_posterior_mse']=float((descendants[:m]-reconstructed[:m]).square().mean())

        rows['conditional_csem_gap_score_uncond'] = conditional_score_gap(unet, mu, lv, zpost, cfg, device)
        rows['eval_samples'] = N
        rows['blocks'] = len(unet.block_channels)
        rows['hierarchy_variant'] = cfg.get('hierarchy_variant', 'standard')
        rows['latent_coordinates'] = int(zpost[0].numel())
        rows['vae_parameters'] = sum(p.numel() for p in vae.parameters())
        rows['score_parameters'] = sum(p.numel() for p in unet.parameters())
        rows['root_coordinates'] = int(unet.block_channels[0]*64)
        rows['child_coordinates'] = int(unet.block_channels[1]*64)
        rows['child_coordinates_definition'] = 'direct Z2; no affine innovation map'
        rows['block_spatial_sizes'] = ':'.join(str(v) for v in cfg.get('block_spatial_sizes', [8]*len(unet.block_channels)))
        rows['evaluation_seconds'] = time.perf_counter()-started
        rows['sampling_definition'] = 'sequential learned conditional diffusion; fixed clean parents'
        rows['fid_backend'] = 'legacy torchvision Inception; same as flat'
        rows['diffusion_steps_per_block'] = 20
        (out/'metrics.json').write_text(json.dumps(rows, indent=2)+'\n')
    return rows
