# What the diagnostic teachers and bounds mean

For a fixed encoder, suppress the class label. Ancestral blocks have diagonal
Gaussian conditionals q_i(z_i | x,z_<i); absent DAG parents are ignored.
Y_i = alpha Z_i + sigma eps_i, with independent eps_i. The joint head receives
Y and class information, never X or any clean latent.

The naive datum is

    C_i = sigma (Y_i-alpha mu_i(X,Z_<i)) / S_i,
    S_i = alpha^2 v_i(X,Z_<i) + sigma^2.

Its regression optimum is b(Y)=E[C|Y]. The valid epsilon score is
m(Y)=E[eps|Y]=-sigma grad_Y log q_t(Y). A leaf's datum is valid. A nonleaf's
datum can omit descendant likelihood terms; covariance alone does not measure
their conditional mean.

## Corrected single-chain teacher for any included DAG

Define a sequential Gaussian posterior proposal

    r_i(z_i | x,z_<i,Y_i) = N(M_i,V_i),
    M_i = mu_i + alpha v_i/S_i (Y_i-alpha mu_i),
    V_i = v_i sigma^2/S_i.

Bayes' rule factor by factor gives

    q_t(Y|x) = integral [product_i q_i,t(Y_i|x,z_<i)]
                           [product_i r_i(dz_i|x,z_<i,Y_i)].

After reparameterizing Z_i=M_i+sqrt(V_i) eta_i, the proposal measure is a fixed
standard Gaussian measure. Differentiating the noisy log factors while holding
eta and X fixed yields a valid teacher

    T = -sigma grad_Y sum_i log q_i,t(Y_i|x,Z_<i(Y,eta)).

For an actual coherent training chain, recover eta_i from its clean Z_i and the
corresponding M_i,V_i, then hold these inferred innovations fixed during the
derivative. The chain conditioned on Y has exactly the likelihood-weighted
proposal law, so E[T|Y,c]=m(Y,c). All derivatives through descendant means AND
variances, and through proposal variances, are required.

For two blocks, this reduces to

    T_1 = C_1 - sigma J_1 G_2,  T_2=C_2,
    J_1 = alpha v_1/S_1,
    G_2 = partial_A log q_2,t(Y_2|X,A).

For three blocks, root correction includes descendant 2 and 3 factors; middle
correction includes factor 3. The latter vanishes in the siblings DAG because
C does not depend on B given (X,A). Root correction can remain in the Markov
DAG through A's effect on the Y_2-conditioned B proposal, even without a direct
A-to-C link. This is why simply adding pairwise clean-chain score corrections
would be wrong. The implementation tests the complete derivative against both
exact linear-Gaussian scores and nonlinear prior-space quadrature.

The helper detaches all finished teachers. It is used for evaluation and frozen
head training. It is not used to assert an encoder information-bound identity.
Differentiating through observed Y as though it were generated inside the
likelihood derivative would also be wrong: Y is an independent argument there.

## Density-bank-free learned witness

Let delta=C-T and beta(Y,c)=E[delta|Y,c]=b-m. For any fitted field w,

    2 E[delta.w(Y,c)] - E||w(Y,c)||^2
      = E||beta||^2 - E||beta-w||^2
      <= E||beta||^2
      <= E||delta||^2.

All squared norms in the tables are means per scalar coordinate. The fitted
field uses standardized linear features plus fixed random Fourier features,
with ridge regression. Class-conditioned fits are separate by class; the null
fit pools labels. Fit images/draws and test images/draws are independent. The
bound concerns the held-out encoder aggregate used by the test distribution;
generalization differences between empirical train/test aggregates can remain.

Recorded SEs are over image draws after reduction over coordinates. We do not
clip negative lower-bound estimates to zero. Positive held-out lower bounds
above their uncertainty support a detected bias component; failure to detect
it can reflect weak witnesses. The fixed feature/ridge choice must not be tuned
by selecting the largest test lower bound without a new independent test set.

The field-error projection E[(h-T).w] equals E[(h-m).w]. It tests whether the
head residual aligns with the same detected defect direction. A small projection
does not prove small full-dimensional score error.

## Common valid score risk

For a valid datum T and any head h(Y,c),

    E[||h-T||^2 - ||T||^2] = E||h-m||^2 - E||m||^2.

Thus `excess_valid = h^2-2hT` compares heads at the SAME fixed encoder/time.
The unknown subtractive true-score energy changes when the encoder changes;
cross-epoch changes in this number alone are not changes in absolute score
accuracy. Raw DSM and corrected targets share their conditional mean but can
have very different variance. Their raw MSE floors cannot be compared as head
errors. `teacher_calibration_gap` should average to zero in expectation for
raw DSM versus corrected supervision of the same head.

## Numerical reference

For a selected bank, sequential proposals sample each Z_i from r_i. Importance
weights are proportional to the product of noisy Gaussian factors. Weighted
averages of corrected and naive teachers estimate m and b for that finite
bank. Self-normalized finite-particle estimates are biased; two panels do not
remove finite-image bias. The audit exposes nested-bank/half-particle/repeat
changes and ESS separately. Passing heuristics is not a full-population proof.
