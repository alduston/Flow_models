# Validation performed for this bundle

Local CPU checks passed with PyTorch2.6.0+cpu and torchvision0.21.0+cpu. These
packages were installed into an isolated scratch runtime solely for validation;
they are not included in the ZIP and are not installed on Vista by this bundle.
Use the existing working hlsi CUDA Torch/torchvision pair on Vista.

- Production architecture audit:6,368,817 VAE parameters,12,159,556 score
  parameters,256 latent coordinates, one stochastic Gaussian block, ordinary
  epsilon head. Exact packing/unpacking checked.
- Deterministic posterior moments do not change when ancestor samples change.
  Analytic teacher agrees with the exact flat Gaussian formula and the inherited
  corrected teacher. Time-zero KL agrees with the packed coordinate-mean formula.
- Encoder score gradients at r=.5 are half those at r=1; score-head gradients
  and scalar score loss agree. Both input and target routes remain active.
- Frozen prior objectives use identical named time/posterior/diffusion/dropout
  streams and leave all VAE state tensors and gradients unchanged.
- All four stages ran reduced synthetic two-epoch training, uninterrupted and
  stopped/resumed. VAE/head/EMA/optimizer/scheduler and global/named/loader RNG
  tensors were identical. Completion re-entry leaves immutable snapshots intact.
- Checkpoint-reloaded reconstruction and full reverse solves agree with in-process
  inference on fixed inputs. CFG1 and CFG3 paths are both checked.
- Offline scheduler fixture verifies all48 jobs, exact afterok dependencies,
  and duplicate-submission prevention. Selection fixtures verify the common
  beta/LR criterion and immutable selections.
- Compiler fixtures verify five-seed Student-t intervals, known paired differences,
  PDF/SVG/PNG generation, duplicate rejection, missing-seed rejection, and removal
  of stale completion markers. All numerical fixtures live in temporary folders
  and are excluded from production results and the archive.
- Architecture/loss/sampler source files are byte-identical to the supplied
  baseline. Config audit confirms joint optimization/loss settings are unchanged.
- Python syntax compilation, shell syntax, Slurm dry-run construction and ZIP
  CRC/SHA256 integrity checks were run.

The CPU smoke path disables LPIPS/GAN, uses a reduced DiT and synthetic features.
It does not certify a real Inception FID or production CUDA performance. The
submitted Vista pilot repeats structural/resume checks on CUDA and times real
data, full models, LPIPS/GAN and legacy Inception evaluation before production
submission. No new500-epoch run or new FID is claimed by this deliverable.

Machine-readable records are in `validation/checks_cpu.json`,
`validation/source_audit.json`, and `validation/workflow_checks.json`.
