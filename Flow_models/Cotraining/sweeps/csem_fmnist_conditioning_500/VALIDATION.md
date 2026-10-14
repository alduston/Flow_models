# Validation performed before packaging

Environment: CPU PyTorch 2.14.1; full model sizes. No training-result FIDs are
included in this bundle.

Passed:

1. Five manifest arms, 500 epochs, 256 latent coordinates, seed 42, correct
   per-block score depths, common loss/evaluation recipe and CLI/Slurm contracts.
2. Bitwise parity against recovered direct two-block and full-ancestry three-block
   sources: initialization/state dictionaries, RNG state, analytic targets and
   losses, reconstruction, all parameter gradients, and sequential RK4 sampling.
3. All five full-size models: forward/backward, finite gradients, an optimizer
   update, conditional input routing and actual sequential RK4 solves.
4. All latent hybrid masks and complete evaluation plumbing, including per-block
   CFG probes. The evaluation smoke uses an eight-dimensional image-feature
   surrogate; it does not validate pretrained Inception weights or actual FID.
5. Online/EMA finite-reference CSEM risk probes for every block/topology, sampled
   decomposition identity, ESS fields and cross-block gradient alignment.
6. Mixture-oracle epsilon predictions against the score obtained by differentiating
   an explicit Gaussian-mixture log density.
7. Diagnostics preserve RNG, training/eval modes and existing parameter gradients.
8. Synthetic two-epoch runs for the reference models: interruption after epoch one,
   atomic checkpoint recovery, uncommitted CSV rollback, and resumed model/EMA/
   scheduler/RNG states equal to an uninterrupted run.
9. Synthetic five-cell result fixtures: all completion contracts, rejection of
   missing hybrid rows, and strict CSV/Markdown/PDF report generation.
10. Archive integrity, fresh-extraction source/hash validation, and five-job
    dry-run submission.

The real 500-epoch runs, CUDA numerical behavior and pretrained Inception/LPIPS
evaluations are performed on the cluster. `job.slurm` automatically runs the
selected full-size CUDA smoke in a separate process before training.

Machine-readable model checks are in `smoke_validation.json`.

```bash
python paper.py validate
python paper.py validate --models
python paper.py smoke --cell-id 3 --device cuda
```
