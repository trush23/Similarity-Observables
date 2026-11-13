# Stage 2 — representations, spatial order and nuisance equivalence

This stage extends the foundation using a fixed inventory of local motifs. The construction provides exact controls over what global pooling discards. All statements concern the supplied finite periodic configurations.

## 1. Fixed inventory, different organization

Let T₀ be an 8 × 8 cosine texture multiplied by a separable Hann window, with its four borders explicitly zero. Let T₁=T₀ᵀ. Four 64 × 64 images place 32 copies of each motif on an 8 × 8 lattice:

- domains: label 1 occupies the right half;
- alternating: label is column index modulo 2;
- checker: label is row-plus-column modulo 2;
- diagonal bands: label is floor((row+column)/2) modulo 2.

Every aligned patch collection is the same multiset. For any pointwise patch map φ, every permutation-invariant sum Σφ(patch) is therefore equal across the four images. The conclusion applies to any finite patch moment and to the entire empirical patch distribution. The explicit experiments use normalized patch mean and traceless second moment.

The zero borders also isolate periodic central-difference gradients at neighboring motifs. Rearranging motifs preserves the multiset of local gradients. Hence the magnitude-weighted global gradient Q also agrees. Positional correlations can change despite these identical inventories.

## 2. Three representation levels

**Whole image.** Flatten signed intensities into x ∈ R⁴⁰⁹⁶. A single-vector Q comparison is equivalent in ranking to cos²(x,y). The implementation uses distance 1−cos², avoiding an explicit 4096 × 4096 tensor. This removes global sign while preserving most positional dependence.

**Patch collection.** Flatten each aligned patch into R⁶⁴ and normalize each nonzero patch independently. Pool its mean and Q, using equal patch weights. Zero patches, when present, are excluded from the orientational distribution. Their presence should be reported in applications where empty mass varies. All supplied clean motifs have nonzero norm.

**Gradient field.** Central differences give two-dimensional vectors at each pixel. Magnitude weights and second circular moments define local nematic order. The position of each field element is retained before any subsequent pooling.

The full orthonormal DCT of a patch is a coordinate change. It preserves the Gram matrix, directional cosine, and distances between means or Q tensors when applied consistently. Truncation, nonlinear maps and learned encoders change the information content; those remain subsequent experiments.

## 3. Local Q at a spatial scale

Let g(r)=gₓ(r)+i gᵧ(r), m(r)=|g(r)|. The sliding local second moment is

```math
z_w(r)=\frac{\sum_{s\in W_w(r)} g(s)^2/m(s)}
                  {\sum_{s\in W_w(r)}m(s)}.
```

Terms with zero magnitude contribute zero; a zero denominator gives z=0. A uniform periodic 8 × 8 window is used in the correlation experiment. The local Q tensor is

```math
Q_w(r)=\frac12
\begin{pmatrix}
\Re z_w(r)&\Im z_w(r)\\
\Im z_w(r)&-\Re z_w(r)
\end{pmatrix}.
```

The fixed-grid alternative partitions the entire image into g × g cells for g=1,2,4,8,16 and computes the same magnitude-weighted moment separately in each cell. The gradient is computed before partitioning, avoiding artificial cell-boundary gradients.

Grid refinement retains more location information. Its sensitivity to a shift depends on grid alignment. Dividing an image into fixed raw patches also changes which pixels share a descriptor after a subpatch translation. That partition sensitivity can be mistaken for an intrinsic change in organization unless translation controls are included.

## 4. Position–orientation correlation

Define the real contraction

```math
C_Q(\delta)=\Re\left[\frac1M\sum_r z_w(r)\overline{z_w(r+\delta)}\right].
```

Since tr[Q(r)Q(s)] = Re[z(r) conjugate(z(s))]/2, this is twice the average tensor contraction. It uses uncentered local order, so C_Q(0) includes its full local energy. Centering would define a different observable.

Periodic translation reindexes the sum, leaving C_Q unchanged. It can be computed as Re[ifft2(|fft2(z)|²)]/M. The real part discards the imaginary antisymmetric component of the complex correlation. That choice is appropriate for this trace observable and creates an additional possible loss of information. The direct and FFT formulations are tested independently.

As established in the foundation, equal pair correlations can conceal different structures. This stage demonstrates separation for the chosen four configurations, without asserting injectivity for all Q-fields.

## 5. Explicit nuisance equivalence

For two fields a,b, the registered relative distance is

```math
D_{\mathrm{shift}}(a,b)=
\frac{\min_t\|a-T_tb\|}{\|a\|+\|b\|}.
```

For real or complex arrays,

```math
\min_t\|a-T_tb\|^2
=\|a\|^2+\|b\|^2-2\max_t\Re\langle a,T_tb\rangle.
```

FFT cross-correlation evaluates every integer shift exactly on the periodic lattice. The pixel baseline additionally minimizes over global sign, replacing the maximum real correlation with maximum absolute real correlation.

Rotation acts on positions and feature directions. A physical angle α multiplies the local nematic phase by exp(2iα), in addition to rotating its coordinates. The code recomputes each representation from all four quarter-turned versions of the image and minimizes over them. This includes the phase transformation and avoids assuming that independently rotating the array of Q values implements the full action. Even-sized local windows have an origin convention; explicit image rotation plus translation search handles that origin in registered comparisons.

Correlation maps minimize over the same four image rotations. Spatial registration handles shifts for Q-fields and pixels. Correlation comparison gains translation invariance from its definition.

These operations deliberately identify all translations and quarter turns. A recognition label that distinguishes absolute location or absolute orientation would conflict with that equivalence. The supplied arrangement label treats those operations as nuisances.

## 6. Controlled discrimination and uncertainty

The gallery contains the four fixed clean prototypes. For each, 24 independent perturbation trajectories are generated with seed 9021. Each trajectory produces five conditions:

1. clean;
2. shifts by multiples of the eight-pixel cell size;
3. integer shifts of one through seven pixels independently on each axis;
4. a quarter turn by 90°, 180° or 270°;
5. the same subcell shift and quarter turn, random global sign, and independent Gaussian noise with standard deviation 0.15 times the clean image standard deviation.

Every method sees the same gallery and views. The whole-image, patch and local-grid methods compare the supplied coordinates. The registration-aware baselines use the explicit nuisance group. No transform is inferred from the correct label or known perturbation parameters.

Ties within absolute distance 10⁻⁷ share credit, giving a chance baseline of 1/4. The uniform accuracy of pooled patch methods follows exactly from the equal-gallery-observable construction, even when a transformed query changes its own patch partition.

Intervals use 2,000 stratified bootstrap resamples of perturbation trajectories, preserving the four prototype strata. Paired contrasts use 5,000 draws and shared indices for methods. These intervals quantify random-perturbation variation conditional on the fixed quartet. There are four underlying prototypes, not 96 independently sampled image contents. Degenerate intervals occur when every perturbation trajectory has the same score.

The combined condition uses modest noise. Success there cannot establish resistance to high noise, nonperiodic boundaries, new motif inventories or novel arrangement families. Those are explicit next tests.

## 7. Interpretation of outcomes

- Clean pooled collisions establish an information limitation of global aggregation.
- Local-grid discrimination establishes resolving power at a chosen scale and alignment.
- Degradation under translation or rotation quantifies the cost of coordinate specificity.
- Recovery by registration shows the role of the decision rule's equivalence relation.
- Recovery by correlations shows that some spatial information can survive removal of absolute position.
- Comparison with registered pixels assesses whether the Q-field supplies additional predictive value in this controlled setting.
- Explanation remains attached to local orientation, spatial organization, alignment and correlation structure. Semantic interpretation would require independent labels.

The delivered code includes tests for exact motif inventory, pooled-moment equality, orthogonal-coordinate equivalence, direct-versus-FFT registration, correlation identities, contrast/translation invariance, and correct orbit retrieval.
