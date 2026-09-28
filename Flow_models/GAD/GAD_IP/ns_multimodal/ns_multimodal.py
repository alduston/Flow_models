#!/usr/bin/env python3
"""Rotation-ambiguous Navier--Stokes filtering benchmark for transport-only GAD.

Run beside the supplied sampling.py:
    python ns_multimodal.py
    NSM_VALIDATE_ONLY=1 python ns_multimodal.py  # forward/symmetry check, no GAD

Adapted from Baptista, Hosseini & Hsu, "Approximating Measures on Function
Spaces: Transport and Truncation", Section 5.3 / Figure 8.  As in that
experiment: periodic 2-D vorticity, viscosity 5e-4, Matérn-5/2 initial field,
time-independent random band-limited forcing, T=15, 60 observation times, and
pointwise vorticity noise 0.05.  This is a deliberately reduced 32x32, 48-KL
inverse problem, rather than a reproduction of their 128x128, 16-sensor,
learned-conditional experiment.

The essential controlled difference is four point sensors at the fixed points
of the half-turn x -> -x (mod 2*pi).  Forcing AND initial vorticity are unknown.
Their Gaussian Fourier coefficients have paired cosine/sine modes; the half-turn
flips all sine coefficients.  Navier--Stokes commutes with this rotation and
every sensor is fixed, so G(z)=G(Rz) EXACTLY while the terminal fields generally
differ.  The Gaussian prior is invariant as well: posterior modes are intrinsic
to this nonlinear inverse problem, with no mixture inserted into the prior.
Four spatial sensors, rather than the paper's 16, are necessary for this exact
pointwise symmetry; all 60 temporal observations and the paper's noise remain.

Use NSM_N_REF, NSM_N_GEN, NSM_ROUNDS, NSM_STEPS for work budget; NSM_N=32
controls grid resolution (even >= 16).  The script prints an actual posterior
midpoint barrier, a Gauss--Newton curvature spectrum, and mode masses.  A
visible two-mode Figure-8-style plot is not, by itself, proof of GAD mixing.
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")

import json
import random
from collections import OrderedDict
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sampling import (GaussianPrior, configure_sampling, get_valid_samples,
                      init_run_results, make_physics_likelihood,
                      run_standard_sampler_pipeline, save_reproducibility_log,
                      summarize_sampler_run, zip_run_results_dir)

jax.config.update("jax_enable_x64", True)

SEED = int(os.environ.get("NSM_SEED", "291"))
N = int(os.environ.get("NSM_N", "32"))
if N < 16 or N % 2:
    raise ValueError("NSM_N must be even and at least 16")
N_INITIAL, N_FORCING = 32, 16
DIM = N_INITIAL + N_FORCING
VISCOSITY = 5e-4
T = 15.0
N_TIMES = 60
DT = 0.025
STEPS_PER_OBS = 10
NOISE_STD = 0.05
LENGTH_SCALE = np.pi / 5
FORCING_POINT_STD = float(os.environ.get("NSM_FORCING_STD", "0.10"))
ODD_TRUTH_SCALE = float(os.environ.get("NSM_ODD_TRUTH_SCALE", "0.003"))
N_REF = int(os.environ.get("NSM_N_REF", "256"))
N_GEN = int(os.environ.get("NSM_N_GEN", str(N_REF)))
ROUNDS = int(os.environ.get("NSM_ROUNDS", "3"))
FLOW_STEPS = int(os.environ.get("NSM_STEPS", "128"))
VALIDATE_ONLY = os.environ.get("NSM_VALIDATE_ONLY", "0") == "1"
if min(N_REF, N_GEN, ROUNDS, FLOW_STEPS) < 1:
    raise ValueError("All work-budget settings must be positive")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def frequency_pairs(count):
    """Independent real Fourier pairs (one representative per +/- k)."""
    pairs = [(kx, ky) for kx in range(0, N // 3)
             for ky in range(-N // 3 + 1, N // 3)
             if (kx > 0 or (kx == 0 and ky > 0))]
    return sorted(pairs, key=lambda k: (k[0] ** 2 + k[1] ** 2,
                                        abs(k[0]) + abs(k[1]), k[0], k[1]))[:count]


def fourier_kl(n_coeff, amplitudes):
    """Columns normalized for prescribed pointwise prior variance."""
    x = 2 * np.pi * np.arange(N) / N
    xx, yy = np.meshgrid(x, x, indexing="ij")
    pairs = frequency_pairs(n_coeff // 2)
    if len(pairs) != n_coeff // 2:
        raise ValueError("Too few resolved modes")
    amps = np.asarray([amplitudes(kx, ky) for kx, ky in pairs])
    amps /= np.sqrt(np.sum(amps ** 2) / 2.0)
    cols = []
    for (kx, ky), a in zip(pairs, amps):
        phase = kx * xx + ky * yy
        cols.extend([a * np.cos(phase), a * np.sin(phase)])
    return jnp.asarray(np.stack(cols), dtype=jnp.float64), pairs


# On a 2-D torus, Matérn nu=5/2 has spectral *power* exponent nu+d/2=7/2.
# Its Fourier coefficient standard deviations thus decay as power -7/4.
INITIAL_BASIS, INITIAL_PAIRS = fourier_kl(
    N_INITIAL,
    lambda kx, ky: (1.0 + LENGTH_SCALE ** 2 * (kx*kx + ky*ky) / 5.0) ** (-1.75),
)
FORCING_BASIS, FORCING_PAIRS = fourier_kl(
    N_FORCING, lambda kx, ky: 1.0 / (1.0 + 0.15 * (kx*kx + ky*ky)),
)

freq = jnp.fft.fftfreq(N, d=2*np.pi/N) * (2*np.pi)
kx, ky = jnp.meshgrid(freq, freq, indexing="ij")
k2 = kx*kx + ky*ky
k2_safe = jnp.where(k2 == 0, 1.0, k2)
dealias = (jnp.abs(kx) < N/3) & (jnp.abs(ky) < N/3)

# Four fixed points under x -> -x on the 2*pi torus. Flattened row-major.
sensor_rc = np.asarray([(0, 0), (0, N//2), (N//2, 0), (N//2, N//2)])
sensor_indices = jnp.asarray(sensor_rc[:, 0] * N + sensor_rc[:, 1])


def rotate_latent(z):
    """Half-turn in coefficient space: cos unchanged, sin sign reversed."""
    return z * jnp.asarray([1., -1.] * (DIM // 2))


def fields_at_zero(z):
    omega = jnp.einsum("c,cij->ij", z[:N_INITIAL], INITIAL_BASIS)
    forcing = FORCING_POINT_STD * jnp.einsum(
        "c,cij->ij", z[N_INITIAL:], FORCING_BASIS)
    return omega, forcing


def ns_rhs(omega_hat, forcing_hat):
    omega_hat = omega_hat * dealias
    psi_hat = -omega_hat / k2_safe
    psi_hat = jnp.where(k2 == 0, 0.0, psi_hat)
    vx = jnp.fft.ifftn(1j * ky * psi_hat).real
    vy = jnp.fft.ifftn(-1j * kx * psi_hat).real
    dx = jnp.fft.ifftn(1j * kx * omega_hat).real
    dy = jnp.fft.ifftn(1j * ky * omega_hat).real
    adv_hat = jnp.fft.fftn(vx * dx + vy * dy) * dealias
    return (-adv_hat - VISCOSITY * k2 * omega_hat + forcing_hat) * dealias


@jax.checkpoint
def advance_observation(omega_hat, forcing_hat):
    def rk4(_, w):
        a = ns_rhs(w, forcing_hat)
        b = ns_rhs(w + 0.5 * DT * a, forcing_hat)
        c = ns_rhs(w + 0.5 * DT * b, forcing_hat)
        d = ns_rhs(w + DT * c, forcing_hat)
        out = (w + DT * (a + 2*b + 2*c + d) / 6.0) * dealias
        return jnp.where(k2 == 0, 0.0, out)
    return jax.lax.fori_loop(0, STEPS_PER_OBS, rk4, omega_hat)


def trajectory(z):
    omega0, force = fields_at_zero(z)
    force_hat = jnp.fft.fftn(force)

    def one_observation(w, _):
        w = advance_observation(w, force_hat)
        field = jnp.fft.ifftn(w).real
        return w, (field.reshape(-1)[sensor_indices], field)

    _, (observations, fields) = jax.lax.scan(
        one_observation, jnp.fft.fftn(omega0), None, length=N_TIMES)
    return observations, fields[-1]


@jax.jit
def solve_forward(z):
    return trajectory(z)[0].reshape(-1)


@jax.jit
def final_field(z):
    return trajectory(z)[1]


def log_posterior(z, y):
    residual = solve_forward(z) - y
    return -0.5 * jnp.vdot(z, z) - 0.5 * jnp.vdot(residual, residual) / NOISE_STD**2


def choose_truth():
    """Choose a supported Gaussian-prior state with a resolvable two-mode valley.

    The small odd component curates the synthetic trajectory for a *moderate*
    barrier at T=15; a generic draw has barriers of order 1e5 here.  No change
    is made to the Gaussian prior used by GAD or to the likelihood.
    """
    rng = np.random.default_rng(SEED)
    best = None
    for _ in range(int(os.environ.get("NSM_TRUTH_CANDIDATES", "8"))):
        z = rng.standard_normal(DIM)
        z[1::2] *= ODD_TRUTH_SCALE
        mid = 0.5 * (z + np.asarray(rotate_latent(z)))
        clean = np.asarray(solve_forward(jnp.asarray(z)))
        center = np.asarray(solve_forward(jnp.asarray(mid)))
        if not np.all(np.isfinite(clean)) or not np.all(np.isfinite(center)):
            continue
        # Includes the prior's preference for the symmetric midpoint.
        barrier = (np.sum((center-clean)**2) / (2 * NOISE_STD**2)
                   - 0.5 * (np.sum(z*z) - np.sum(mid*mid)))
        if barrier > 5 and (best is None or abs(np.log(barrier/25)) <
                            abs(np.log(best[0]/25))):
            best = (barrier, z, clean)
    if best is None:
        raise RuntimeError("No separated posterior pair found. Try another NSM_SEED "
                           "or increase NSM_TRUTH_CANDIDATES / NSM_FORCING_STD.")
    return best


def stiffness_and_pair_diagnostics(z, y):
    zr = np.asarray(rotate_latent(z))
    midpoint = (z + zr) / 2
    lp = np.asarray([log_posterior(jnp.asarray(a), y)
                     for a in (z, zr, midpoint)])
    jac = np.asarray(jax.jit(jax.jacfwd(solve_forward))(jnp.asarray(z)))
    ev = np.linalg.eigvalsh(np.eye(DIM) + jac.T @ jac / NOISE_STD**2)
    f0 = np.asarray(final_field(jnp.asarray(z)))
    fr = np.asarray(final_field(jnp.asarray(zr)))
    out = dict(logpost_pair_difference=float(abs(lp[0]-lp[1])),
               logpost_midpoint_barrier=float(min(lp[0], lp[1])-lp[2]),
               terminal_pair_rel_distance=float(np.linalg.norm(f0-fr) /
                                                max(np.linalg.norm(f0), 1e-12)),
               gn_min_eigenvalue=float(ev[0]), gn_max_eigenvalue=float(ev[-1]),
               gn_eigenvalue_ratio=float(ev[-1]/ev[0]),
               gn_eigenvalues=ev.tolist())
    return out


def main():
    truth_barrier, z_true, y_clean = choose_truth()
    rng = np.random.default_rng(SEED + 1)
    y_obs = y_clean + rng.normal(0, NOISE_STD, size=y_clean.shape)
    y_jax = jnp.asarray(y_obs)
    other = np.asarray(rotate_latent(z_true))
    equiv = np.max(np.abs(np.asarray(solve_forward(jnp.asarray(other))) - y_clean))
    if equiv > 1e-7:
        raise AssertionError(f"Half-turn observation symmetry failed: {equiv}")
    diagnostics = stiffness_and_pair_diagnostics(z_true, y_jax)
    diagnostics.update(dict(clean_midpoint_barrier=float(truth_barrier),
                            symmetry_observation_max_error=float(equiv),
                            dim=DIM, grid=N, n_sensors=4, n_times=N_TIMES,
                            odd_truth_scale=ODD_TRUTH_SCALE,
                            noise_std=NOISE_STD, seed=SEED))
    print(json.dumps(diagnostics, indent=2))
    if diagnostics['logpost_midpoint_barrier'] < 3:
        print("WARNING: noise made the midpoint valley shallow for this truth.")
    if VALIDATE_ONLY:
        return

    configure_sampling(active_dim=DIM, default_n_gen=N_GEN,
                       hess_min=1e-6, hess_max=1e9)
    run = init_run_results("ns_multimodal")
    outdir = Path(run['run_results_dir'])
    (outdir / "model_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    np.savez_compressed(outdir / "synthetic_data.npz", z_true=z_true,
                        z_rotated=other, y_clean=y_clean, y_obs=y_obs,
                        sensor_rc=sensor_rc, initial_pairs=INITIAL_PAIRS,
                        forcing_pairs=FORCING_PAIRS)

    prior = GaussianPrior(dim=DIM)
    likelihood, _ = make_physics_likelihood(
        solve_forward, y_obs, NOISE_STD, use_gauss_newton_hessian=True,
        log_batch_size=8, grad_batch_size=4, hess_batch_size=1)
    configs = OrderedDict()
    for k in range(1, ROUNDS+1):
        label = f"GAD{k}"
        configs[label] = dict(node="transport", score="lfgi", n_samples=N_GEN,
                              n_ref=N_REF, n_gate=N_REF, bank_coupling="shared",
                              steps=FLOW_STEPS, t_min=0.002, t_max=2.5,
                              log_mean_ess=True, display_name=f"GAD round {k}")
        if k > 1:
            configs[label]['ref_source'] = f"GAD{k-1}"

    pipeline = run_standard_sampler_pipeline(prior, likelihood, configs, n_ref=N_REF)
    summarize_sampler_run(pipeline['sampler_run_info'])
    true_final = np.asarray(final_field(jnp.asarray(z_true)))
    odd = z_true - other
    odd /= np.linalg.norm(odd)

    # Histograms expose both symmetry-related basins and mode collapse.
    fig, axes = plt.subplots(1, ROUNDS, figsize=(4.3*ROUNDS, 3.5), squeeze=False)
    stats = {}
    for ax, (label, samples) in zip(axes[0], pipeline['samples'].items()):
        z = np.asarray(get_valid_samples(samples))
        if len(z) == 0:
            raise RuntimeError(f"No finite particles in {label}")
        projection = z @ odd
        # Sign halves only indicate basin occupancy; they do not certify
        # posterior weights or equilibrium within either basin.
        stats[label] = dict(n_valid=int(len(z)), positive_fraction=float(np.mean(projection>0)),
                            negative_fraction=float(np.mean(projection<0)),
                            projection_mean=float(np.mean(projection)),
                            projection_std=float(np.std(projection)))
        ax.hist(projection, bins=35, density=True, color="#417ea3", alpha=.8)
        ax.axvline(float(z_true @ odd), color="black", ls="--", label="truth")
        ax.axvline(float(other @ odd), color="orange", ls="--", label="rotation")
        ax.set(title=label, xlabel="rotation-odd latent projection", ylabel="density")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "mode_occupancy.png", dpi=180)
    plt.close(fig)

    # Figure 8 analogue: truth, two draws, posterior mean, std, |mean-truth|.
    for label, samples in pipeline['samples'].items():
        z = np.asarray(get_valid_samples(samples))
        f = []
        for a in z:
            f.append(np.asarray(final_field(jnp.asarray(a))))
        fields = np.asarray(f)
        mean, std = fields.mean(axis=0), fields.std(axis=0)
        pick = [int(np.argmin(z @ odd)), int(np.argmax(z @ odd))]
        panels = [true_final, fields[pick[0]], fields[pick[1]],
                  mean, std, abs(mean-true_final)]
        titles = ["True terminal", "Draw: negative basin", "Draw: positive basin",
                  "Posterior mean", "Posterior std", "|mean - truth|"]
        clim = max(np.max(abs(a)) for a in panels[:4])
        fig, axes = plt.subplots(1, 6, figsize=(20, 3.7), constrained_layout=True)
        for i, (ax, image, title) in enumerate(zip(axes, panels, titles)):
            im = ax.imshow(image, origin="lower", cmap="RdBu_r" if i<4 else "magma",
                           vmin=-clim if i<4 else 0,
                           vmax=clim if i<4 else max(float(np.max(image)), 1e-9))
            ax.scatter(sensor_rc[:, 1], sensor_rc[:, 0], s=24,
                       facecolors="none", edgecolors="lime", linewidths=1.4)
            ax.set(title=title, xticks=[], yticks=[])
            fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
        fig.suptitle(f"{label}: terminal vorticity | {len(z)} particles")
        fig.savefig(outdir / f"figure8_{label}.png", dpi=170)
        plt.close(fig)

    (outdir / "mode_occupancy.json").write_text(json.dumps(stats, indent=2))
    save_reproducibility_log(
        title="Rotation-ambiguous Navier--Stokes GAD benchmark",
        config=dict(seed=SEED, grid=N, latent_dim=DIM, initial_dim=N_INITIAL,
                    forcing_dim=N_FORCING, forcing_std=FORCING_POINT_STD,
                    odd_truth_scale=ODD_TRUTH_SCALE,
                    n_ref=N_REF, n_gen=N_GEN, rounds=ROUNDS, steps=FLOW_STEPS,
                    viscosity=VISCOSITY, T=T, dt=DT, n_times=N_TIMES,
                    n_sensors=4, noise_std=NOISE_STD, diagnostics=diagnostics,
                    sampler_configs=configs))
    zip_path = zip_run_results_dir()
    print("Mode occupancy:", json.dumps(stats, indent=2))
    print("Results:", outdir, "archive:", zip_path)


if __name__ == "__main__":
    main()
