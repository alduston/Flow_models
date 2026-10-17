# Questions, interventions, and falsifiable readings

The working explanation is a compromise: image reconstruction can benefit from
stochastic-parent-dependent posterior means; naive stacked score supervision
may favor a representation whose per-image posterior is closer to a product
of diagonal Gaussians. The observed modest decline of coupling after early
learning establishes neither negligible score bias nor a theorem forcing
decoupling. This study tests those missing steps.

## 1. Is CSEM pressure suppressing dependence, rather than merely fitting an easier moving target?

Compare two-block encoder multipliers 0, 0.5, 1, 2, 4 at matched seeds. A multiplier
of 0 removes score gradients to the encoder, while the head still learns the
same naive target on its changing representation. Head coefficient and learning
rate, terminal coefficient and primary boundary B coefficient remain fixed.

Predictions to test: higher pressure lowers per-image conditional KL/Jacobian
energy and the learned-bias bound, while decoder swaps quantify the
reconstruction value sacrificed. An easier teacher can lower the raw loss
without improving score fitting; inspect target energies and common valid
excess risks, not raw losses alone. Compare dependence at equal epochs and, as
a secondary descriptive analysis, at similar reconstruction levels.

Pressure gates are local infinitesimal interventions, not proof that gradient
descent follows a global scalar coupling coordinate. We log objective-specific
gate derivatives and actual encoder gradient conflict. Global clipping can
change head updates when encoder gradients increase: the clipping controls
and recorded norm/factor trajectories expose this mediation. Co-scaling B is
separate because it would otherwise confound the CSEM-only intervention.

The score gate derivative is also split exactly into head-input, target-Y,
target-mean and target-variance chain-rule routes. In a coherent draw, the
mean in Y can cancel the mean in its own Gaussian teacher; these paths should
not be interpreted separately without their sum. This decomposition helps
distinguish pressure from teacher variance, altered score predictability and
the head's current fitting error. It is not a decomposition into exact Bayes
and excess risk; that would require knowing the full conditional regression.

## 2. How much of the coupling actually becomes score bias?

Distinguish three quantities: dependence of the encoder, pathwise teacher
correction, and the conditional mean of that correction. A large pathwise
correction can average away. The variational learned witness gives a resolved
projection lower bound on conditional-mean bias, while pathwise correction
energy supplies an upper bound. A wide interval is an identifiability limitation,
not evidence for zero bias. Fixed witnesses can miss high-complexity defects.

The richer finite-bank audit is optional corroboration. It must resolve both
image coverage and particle integration. Do not discard failed low-time bins,
and do not normalize a tiny correction by an unresolved reference and interpret
the resulting ratio as a mechanism.

## 3. What does a third block change?

Every prior remains one joint 256-dimensional head; block allocation changes
from 64+192 to 64+64+128. The three-block VAE grows, so depth alone is not isolated
from decoder/inference capacity. Use within-three-block contrasts and the
decoupled controls to interpret that confound.

| Posterior dependency | Naive ancestor targets generically biased |
|---|---|
| Two-block full | A |
| Two-block decoupled given X | none |
| Three-block full: A -> B, (A,B) -> C | A, B |
| Three-block Markov given X: A -> B -> C | A, B (A still has an indirect path) |
| Three-block siblings given X: A -> B, A -> C | A |
| Three-block decoupled given X | none |

All statements allow special cancellations/zero couplings. The final target is
valid in every DAG. The siblings/decoupled cases are mathematical negative
controls for the measured middle/ancestor defects. A failure of those controls
indicates a software/numerical problem before any substantive conclusion.

The corrected teacher is decomposed by descendant log-likelihood factor.
For root A, the C contribution includes direct and indirect paths; it is not
identified as a direct-edge-only effect. Squared contribution energies are
not additive because cross terms can cancel or reinforce.

## 4. Does removing bias help generation once the encoder is fixed?

For each depth at pressure 1 and 4, replicate naive/corrected/DSM head-only
continuations from the identical final source EMA. This isolates target choice
from representation movement. Compare all three branches against both their
matched naive continuation and the unchanged epoch-zero source.

Use the same held-out valid teacher to compare risk and identical generated
starts at CFG1/3. If corrected and DSM agree on improved valid risk but neither
improves FID, score bias is not established as the dominant operating-point
error. If a corrected branch helps at CFG1 and hurts at CFG3, guidance was a
confound. If DSM lags corrected, teacher variance/optimization is plausible;
fifty epochs still does not prove attainment of either population optimum.
Head risk improving without FID improvement is a substantive possible result,
not a failed experiment to optimize away.

These frozen corrections establish valid regression supervision. They do NOT
establish a corrected joint CSEM representation/information objective for
end-to-end encoder training. That separate mathematical question is not hidden
inside this study.

## 5. Could guidance, time integration, initialization or decoder robustness explain apparent success?

CFG1 directly tests the trained conditional field; CFG3 intentionally changes
it. Matched RK4-20/50 checks reveal solver sensitivity. Encoder-OU q_T versus
Gaussian initialization tests terminal-prior mismatch separately from the head.
Decoder same-image swaps/zeros measure whether a block matters; they do not
represent exact conditional samples under the joint prior. No clamped joint
diffusion is advertised as an exact conditional sampler.

## Analysis commitments

- Two paired seeds, all fixed 500-epoch joint endpoints; final versus partial
  status is explicit. Do not pick an arm/epoch because it has the best FID.
- Present time bins and class/null conditions separately before averaging.
- Compare raw losses only when teacher/encoder are held fixed; otherwise use
  the recorded target energies, valid-risk probes, and encoder diagnostics.
- Treat image-level SEs, seed replication and reference resolution as different
  sources of uncertainty. Two seeds are a check of reproducibility, not a strong
  statistical-power claim.
- Negative witness lower bounds and unresolved finite-bank references leave
  uncertainty. They are not claims that the true bias is absent.
- An effect of increasing pressure does not automatically mean a pure
  irreducible-variance incentive: target bias, finite head capacity, shared
  encoder pathways, regularization and clipping can all contribute.
- End-to-end pressure results are associations under a controlled intervention;
  the frozen stage supplies a cleaner intervention on score supervision itself.
