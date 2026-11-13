# Similarity profiles through a hierarchy of observables

This program studies what a representation makes measurable, how much structure each observable retains, and which task distinctions a decision rule can recover. Polymer, liquid-crystal and glass ideas motivate observables. Physical dynamics and thermodynamic behavior require separate evidence and definitions. The calculations below concern finite vector configurations, periodic scalar fields and ordered sequences.

## 1. Configuration and reference geometry

An object is represented by

```math
\mathcal C_A=\{(r_i,h_i,w_i)\}_{i=1}^{N_A},\qquad
h_i\in\mathbb R^d,\quad w_i\geq0,\quad\sum_iw_i=1.
```

Positions r_i live in an explicitly specified domain, such as a periodic grid, open sequence or graph. Features h_i share a coordinate system across compared objects. For h_i ≠ 0, define u_i = h_i / ||h_i||. Feature norms, zero-vector mass and occupancy can be separate observables. Direction normalization intentionally discards magnitude.

Token positions and embedding coordinates define distinct geometries. Sequence lag operates on token positions. A trajectory through embedding space instead defines bonds h_(i+1) − h_i. Selecting one determines the question answered by its correlations.

The prototype experiments use no learned embeddings. Their choices are equal weights for abstract vectors, gradient-magnitude weights for image orientations, and periodic boundaries. Domain-specific labels remain outside the observable definition.

## 2. Observable and comparison are separate choices

Let O_j map a configuration into a specified vector or tensor space. Define

```math
\mathbf S(A,B)=\big(s_j(O_j(A),O_j(B))\big)_{j\in J}.
```

The program implements a descriptive agreement convention

```math
s_j(X,Y)=1-\frac{\|X-Y\|}{\|X\|+\|Y\|}.
```

The norm is Euclidean, or its Frobenius/complex analogue. Triangle inequality gives 0 ≤ s_j ≤ 1. Identical nonzero observables give 1. Numerically zero total norm (≤ 10⁻¹² in the fixed experiment units) gives 1 by convention. This convention must be accompanied by the observable magnitudes, because agreement between two absent signals carries different information from agreement between strong signals. No metric axiom or calibrated probability interpretation is assumed for this descriptive ratio. A different reporting scale can preserve or change task rankings and must be named.

A scalar decision rule f(S) is an additional layer. The foundational retrieval experiment evaluates each observable separately. There are no optimized profile weights.

## 3. Moments and angular harmonics

For unit directions, define

```math
M_A^{(k)}=\sum_i w_i u_i^{\otimes k},\quad
\mu_A=M_A^{(1)},\quad
Q_A=M_A^{(2)}-\frac{I_d}{d}.
```

Q is symmetric and traceless. Its eigenvalues lie between −1/d and 1−1/d. The directional covariance is M²−μμᵀ; Q subtracts an isotropic reference instead. These observables answer different questions.

The independent-pair contraction identity follows by distributivity and the tensor-product inner product,

```math
\langle M_A^{(k)},M_B^{(k)}\rangle
=\sum_{i,j}w_i v_j (u_i^Tz_j)^k.
```

Raw k-th tensor moments include angular harmonic degrees k, k−2, … . A hierarchy of irreducible angular orders avoids counting those lower-degree pieces repeatedly.

For unit directions u,z and t = uᵀz,

```math
\langle Q(u),Q(z)\rangle=t^2-1/d,\quad
\|Q(u)\|_F^2=1-1/d,\quad
H_2^{(d)}(t)=\frac{dt^2-1}{d-1}.
```

Proof follows by expanding tr[(uuᵀ−I/d)(zzᵀ−I/d)] and using tr(uuᵀ)=1. On t ≥ 0, H₂ is strictly increasing, so transforming a single nonnegative cosine in this way preserves retrieval order. Over signed t, H₂ identifies t and −t.

For d > 2 the normalized zonal harmonic of order l is

```math
H_l^{(d)}(t)=\frac{C_l^{(d-2)/2}(t)}{C_l^{(d-2)/2}(1)}.
```

In d=3 these are Legendre polynomials. In d=2 use H_l(t)=cos(l arccos t). The functions H_l are positive-semidefinite kernels on the corresponding unit sphere by the spherical-harmonic addition theorem. Their measure-level comparisons are

```math
K_l(A,B)=E_{u\sim A,z\sim B}[H_l^{(d)}(u^Tz)],
\quad D_l^2=K_l(A,A)+K_l(B,B)-2K_l(A,B).
```

This is the squared Hilbert-space distance between mean harmonic embeddings of order l. It can vanish for distinct distributions. The linear kernel uses μ_Aᵀμ_B. Cosine between pooled means introduces a further division by their norms and becomes undefined when either mean vanishes.

On the circle, write m_l = E[exp(ilθ)]. Then K_l(A,B)=Re(m_l(A) conjugate(m_l(B))) and D_l²=|m_l(A)−m_l(B)|². Moreover,

```math
Q=\frac12
\begin{pmatrix}
\Re m_2 & \Im m_2\\
\Im m_2 & -\Re m_2
\end{pmatrix}.
```

Even orders are invariant under individually reversing any direction; odd orders carry polarity. A common orthogonal change of feature coordinates preserves inner-product comparisons. Q itself transforms as RQRᵀ. Invariance under rotating both objects together differs from invariance under independently rotating either object.

A finite truncation can have collisions. On the compact sphere the complete collection of harmonic coefficients determines the directional probability measure. The finite-sample estimation problem still grows with order and dimension.

For periodic image central differences g_i, magnitude weighting gives
E_w[u] = Σ_i ||g_i|| g_i/||g_i|| / Σ_i ||g_i|| = Σ_i g_i / Σ_i ||g_i|| = 0.
The sum telescopes in each spatial direction. Thus the global polar component in the image experiment is forced to zero by the representation. Its constant agreement column carries no discrimination. This identity does not apply to arbitrary vector configurations or each subregion separately.

## 4. Two exact angular counterexamples

1. Half +e_x and half −e_x versus half +e_y and half −e_y. Both means vanish; Q differs.
2. Uniform mass on the four cardinal directions versus uniform mass on four directions rotated by π/4. Both means and Q vanish. Their fourth circular moments are +1 and −1, so D₄²=4.

Thus vanishing second-order anisotropy alone does not certify an isotropic distribution. Normalizing Q by its norm near zero creates an unstable direction estimate.

## 5. Position, correlation and structure factor

For a periodic vector sequence x_i of length N, the implementation uses uncentered

```math
C(\ell)=\frac1N\sum_i x_i x_{i+\ell}^T.
```

A connected correlation replaces x_i with x_i−mean(x). For an open sequence the divisor becomes N−l and no index wraps. These conventions give different results near boundaries.

With F(k)=Σ_i x_i exp(−2πiki/N), define the Hermitian matrix spectrum S(k)=F(k)F(k)†/N. The inverse DFT recovers the corresponding periodic directed correlations with the lag-sign/transposition determined by the DFT convention. The scalar trace gives Σ_channels |F_channel|²/N and discards cross-channel terms.

For a scalar grid field with M sites, P(k)=|FFT(x−mean(x))|²/M, and inverse FFT(P) equals the mean-centered circular autocorrelation. In particular C(0)=mean[(x−mean(x))²] and Σ_k P(k)/M=C(0). The implementation tests this equality against direct sums at selected lags.

Full power is invariant under periodic integer shifts and global sign reversal. It retains orientation in the wavevector index. Radially pooling shell powers discards wavevector direction. Shells use rounded radius in Fourier-index units in the implementation.

A local field Q(r) and its correlations provide another combination of position and orientation. Fixed-grid local moments depend on registration and grid alignment.

## 6. Homometry and third-order information

On the cyclic group Z₁₂, binary occupancy sets A={0,1,4,6} and B={0,1,3,7} have identical pair-difference counts and hence identical autocorrelations and powers. Exhausting all 12 shifts and their reflections verifies that the two configurations are inequivalent under those operations.

The cyclic third-order occupancy correlation

```math
T(r,s)=\frac1N\sum_i x_i x_{i+r}x_{i+s}
```

and its Fourier counterpart

```math
B(k,l)=F(k)F(l)\overline{F((k+l)\bmod N)}/N
```

separate this pair. A translation multiplies F(k) by a phase; the three phases in B cancel. Contrast inversion changes the sign of B. Reconstruction guarantees need additional hypotheses on the group, spectral zeros and admissible equivalences. We establish separation and translation invariance for this particular example.

## 7. Finite-sample order and reference distributions

For iid equal-weight unit vectors, let q(u)=uuᵀ−I/d. Its squared norm is always c=1−1/d. Expanding the squared sample mean gives

```math
E\|\widehat Q\|^2
=\frac{c}{N}+\frac{N-1}{N}\|E q(u)\|^2.
```

Under isotropy, E q=0. For N>1 an unbiased estimator of population squared order is

```math
\widehat A=\frac{N\|\widehat Q\|^2-c}{N-1}.
```

It may be negative. This reflects estimation uncertainty. Clipping it creates bias. Independent directions with fixed unequal normalized weights replace 1/N by Σ_i w_i² in the expectation; data-dependent weights and correlated samples need separate analysis.

Reference centering may use E_reference[uuᵀ] instead of I/d. Estimate that reference separately from evaluation objects. Its error must enter uncertainty if substantive conclusions depend on it.

Repeated measurements of the same direction do not increase independent sample size. The retrieval benchmark therefore resamples independent base queries, within prespecified prototype strata, retaining their repeated noise views.

## 8. A deliberate hierarchy with branching information

The hierarchy is partially ordered. For a fixed representation, later moments can reveal new distinctions while positional pooling and angular pooling remove different information. High order is not synonymous with high usefulness. Full power and autocorrelation are equivalent information in different coordinates. Fourth angular order does not contain all the distinctions of second angular order. Retaining orders jointly addresses that particular loss.

An observable is adequate for a controlled label only when its equivalence classes respect that label. If O(A)=O(B) but labels differ, any deterministic rule using only O must assign the same output to both. This gives an exact obstruction independent of model optimization.

For explanation, report a change in the measured property. Associating the property with category, identity or style requires independent labels or controlled interventions.

## 9. Research stages and decision criteria

The delivered foundation establishes analytical identities, designed collisions, finite-sample nulls, and a fixed-rule retrieval example. The next stages are:

- Repeat the same interventions across representations while preserving the same configurations and labels.
- Add orientation–position cross-correlations and progressively richer angular/spatial orders.
- Evaluate correlations among profile components to identify redundant evidence.
- Fit any normalization, reference distribution or decision weights only on independent development configurations.
- Test generalization on held-out pattern families, encoder settings and nuisance distributions.
- For learned representations, compare with pooled cosine, fixed histogram/HOG descriptors, bilinear/second-moment pooling, and spectral baselines.
- Evaluate explanation by known property interventions and quantify unwanted cross-responses.
- Treat success as a specified gain in resolving power, calibrated measurement or held-out decision performance. A physical-phase interpretation would require a separate ensemble and dynamics.

## Sources and connections

- [NIST DLMF, orthogonal polynomials and spherical harmonics](https://dlmf.nist.gov/18.38) supplies the dimension-dependent harmonic context.
- [SciPy Gegenbauer definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gegenbauer.html) documents the implemented polynomial family.
- [Grimm and Baake, Homometric Point Sets and Inverse Problems](https://arxiv.org/abs/0808.0094) discusses ambiguity under equal pair correlations.
- [Kakarala, The bispectrum as a source of phase-sensitive invariants](https://arxiv.org/abs/0902.0196) supplies the broader invariant-theory context.
- [Muandet et al., Kernel Mean Embedding of Distributions](https://arxiv.org/abs/1605.09522) connects distributional observables to kernel comparisons.
- [Lin et al., Bilinear CNN Models for Fine-Grained Visual Recognition](https://openaccess.thecvf.com/content_iccv_2015/html/Lin_Bilinear_CNN_Models_ICCV_2015_paper.html) is relevant prior work on local-feature outer-product pooling.

The finite examples, their numerical values and implementation conventions are derived and verified in this project. No novelty claim is made for the individual observables or their established mathematical connections.
