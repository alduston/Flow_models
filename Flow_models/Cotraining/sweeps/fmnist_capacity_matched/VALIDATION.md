# Validation

Local validation uses PyTorch 2.14.1+cpu and torchvision 0.29.1+cpu. These are
validation-environment versions, not a request to upgrade the cluster runtime.
No full FMNIST training or GPU FID reproduction has been performed locally.

`python paper.py validate --models` checks:

- All 20 cells share the 4.3 run's science settings. The entire original
  `b2_full_r0.5_s43` configuration matches `stack2_s43`.
- Production-size 18-layer score heads, common encoder and decoder tensors
  initialize identically at the same seed. Flat/stacked pairs have identical
  entire VAE initialization within each encoder family.
- VAE budgets are 6,368,847 and 6,368,817; head budget is 12,159,556; all arms
  retain 256 coordinates and reversible permutation packing.
- Flat moments have no derivative with respect to sampled parent codes;
  stacked second/third moments have the intended nonzero parent derivatives.
- Every VAE parameter receives an objective gradient; no dummy parameters.
- The valid corrected teacher equals naive CSEM for the flat controls.
- At production capacity, reference and new two-block initialization, sampled
  losses/reconstruction, clipped gradients and one AdamW update are bitwise
  identical. This local comparison disables LPIPS/GAN and uses
  MSE+score+terminal+band; production retains unchanged LPIPS/GAN training code.
- Partial compilation pairs only common epochs, preserves A-minus-B sign,
  separates the five-seed and three-new-seed endpoint summaries, and refuses
  strict completion when jobs are incomplete.

Synthetic two-epoch smoke checks exercise every arm through optimization,
CFG1/3 reverse generation, posterior reconstruction, diagnostics, model saves
and completion. These use the full VAE with a reduced 48-wide/two-layer score
head, batch two, one minibatch/epoch, and artificial 16-dimensional features.
Their FIDs are software checks, not scientific results. Real LPIPS/GAN training
and 2048-dimensional Inception evaluation require the cluster environment.

An interrupted/resumed synthetic two-block training check compares VAE, score,
EMA, AdamW, scheduler, Torch RNG and Python RNG bitwise with uninterrupted
training. Sampling/diagnostics preserve the original RNG state.

Run checks:

```bash
python paper.py validate --models
python paper.py smoke --cell-id 0
python paper.py smoke --cell-id 1
python paper.py smoke --cell-id 2
python paper.py smoke --cell-id 3
python -c 'from pathlib import Path; import tests; tests.resume_check(Path.cwd())'
```

`validation.json`, `parameter_audit.json` and `validation/*.json` retain the
executed check results. The 20-job Slurm dry-run, Python syntax compilation,
and Bash syntax check are included in `validation/package_checks.json`.
