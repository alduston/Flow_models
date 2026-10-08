# Source lineage and decisions

This bundle was assembled from the late-August CSEM code lineage rather than the much older paper-comparison driver.

Primary recovered code lineage:
- `cifar_cfg_nfe_fresh500_v2_bundle_fixed_slurm.zip`: fresh-training evaluator certification and the current short-horizon CIFAR operating point.
- `cifar_tk_anchor_compare_v1_bundle.zip` and `cifar_TKxCSEM_fine_T1p75_v1`: two-horizon / terminal-anchor routing and later representation-horizon sweep machinery.
- Earlier `csem_split_metric.py` / `csem_new(2).py`: the independent two-stage freeze-score/refinement mechanism and the paired CSEM/Tweedie score-head design.

The final driver in this bundle keeps the current paper's stated final method: raw encoder mean, no normalization, CSEM co-training plus terminal component KL. The later OU-partial+KL result was not promoted into the final paper arm because the consolidated knowledge report explicitly calls its absolute headline result provisional and the current paper specifies the raw-mean terminal-anchor method.

The independent comparison is implemented as the exact `T_K=0` limit of the current two-horizon code. The VAE is trained with the score networks frozen; then CSEM and Tweedie heads are trained on the same frozen VAE. This removes a historical apples-to-oranges problem in which independent baselines could come from a different driver generation.

Two paper-specific code patches are intentionally small:
1. co-training evaluation now evaluates the active score head (`control` for the naive Tweedie control, `lsi` otherwise);
2. an optional matched-20-step Heun-SDE / RK4-ODE evaluation pair is added only for cells that populate the sampler-order placeholder.

The full dense `T x lambda_K` sweep was intentionally reduced to a compact cross around the certified CIFAR point. Everything else requested by the current paper is represented in the manifest.


## V2 correction (October 8, 2026)

The late-August state-of-knowledge report identifies the leading CIFAR matched candidate as **OU-partial GroupNorm at T_K + terminal KL w_K=.60**, not the raw-mean KL-only arm used as V1's main cell. V2 therefore uses OU-partial on the two main CIFAR seeds and all CIFAR T/lambda_K sensitivity cells, while adding raw-mean+KL controls on the same code path. The historical hard GN0 and no-anchor cells remain ablations. V2 also fixes the independent-pair oracle loader and concurrent dataset-download race.
