# Stage 5 — pair profiles and a transparent decision rule

The main object is an image-only profile S(A,B). A decision rule can combine its coordinates without obscuring the individual measurements. This stage tests that separation with six synthetic arrangement families and a family-disjoint evaluation.

## Pair interface

`similarity_profile(A,B)` accepts two finite 64 × 64 grayscale arrays. It returns six named agreements. The function receives no labels, reference motif identities, or gallery. All spatial shifts are periodic and the tile frame is fixed. The experiment aligns over integer tile shifts in both directions.

For each coordinate, write Sⱼ=1−dⱼ. Distances are bounded by 1 using the sum of the two norms. The convention for two zero observables is agreement 1. A zero field has no preferred orientation. Its reported agreement should be interpreted in light of that absence.

| Component | Observable and comparison | Invariance or information loss |
|---|---|---|
| Orientation histogram | Sixteen-bin marginal of the real tile nematic field | Ignores tile positions and tile contrast sign. |
| Orientation pairs | Full real periodic autocorrelation of that field | Ignores absolute position and can have homometric collisions. |
| Orientation triplets | 8 × 8 array of three-point moments at fixed directional lag pairs | Retains selected three-point organization while pooling absolute position. |
| Aligned orientation | Minimum relative field distance across the 64 cyclic tile shifts | Allows a tile translation while preserving the registered field. |
| Intensity power | Full mean-centered periodic power of the image | Removes translation phase and retains frequency direction. Independent tile polarity can change the spectrum. |
| Aligned sign-free pixels | Minimum relative pixel distance across the 64 cyclic tile shifts and independent sign choices for each tile | Matches the known lattice and contrast nuisance. Keeps pixel structure within each tile. |

For the last component, compare tiles using min over sign of squared residual, ||x||²+||y||²−2|xᵀy|. Sum over tiles for a given shift and minimize over shifts. The shift search is symmetric and contains the identity. The core implementation uses actual tiles from both arrays. It is not restricted to the synthetic motif library.

## Arrangement benchmark

Each 8 × 8 layout contains 32 tiles of each of two orientations. Each gallery contains one source and three distractors created by exchanging 2, 6 and 12 pairs of opposite sites. Candidate order is randomized. All 768 gallery arrangements are distinct modulo cyclic shifts, including across splits.

- Train on independent and compact arrangements.
- Use the diagonal family to select a single-component comparator.
- Hold out elongated, checker-biased and separable arrangements for the final evaluation.

Each family supplies 32 galleries. Each gallery supplies six views of its source: wave, ridge, bar, ridge with Gaussian noise SD equal to signal RMS, bar with noise SD twice signal RMS, and bar with lower noise plus a blank 3 × 3 tile block. Each view independently draws a cyclic tile shift and independent tile signs. The clean reference gallery uses wave motifs. Seeds and split definitions are saved in the fit manifest.

There are 64 training, 32 development, and 96 test galleries, for 1,152 queries total and 4,608 query–candidate profiles. The test panel contains 576 queries, grouped by 96 galleries. Ties within an absolute distance tolerance of 10⁻⁷ share credit. Chance is 25%.

## Decision and explanation

Given a nonnegative simplex weight vector w, the score is

```math
s_w(A,B)=\sum_j w_jS_j(A,B)=1-\sum_j w_j d_j(A,B).
```

For a positive candidate p and a negative n, the ranking margin is wᵀ(dₙ−dₚ). Fit w by averaging the logistic loss log(1+exp(−margin/0.05)) over the training query–negative pairs, with a 0.02 squared-distance penalty toward the uniform vector. These coefficients are fixed. The simplex constraint keeps the scalar score on the same bounded scale. The score is not a calibrated match probability.

The scale diagnostic uses only training data to fix a scale τⱼ for each component, the median range across candidates within a query. Replace dⱼ with dⱼ/(dⱼ+τⱼ). A zero scale maps that component to zero distance. Positive scales preserve each component's rank but can alter the distances and margins used by a combined rule. The full diagnostic repeats the same constrained fit after this transform.

To explain a decision between B and C, report wⱼ[Sⱼ(A,B)−Sⱼ(A,C)] for every component. The sum equals the scalar score gap. Also report the unweighted gaps to expose components that favor a different candidate. This explains the decision algebraically under the chosen features. The link to identity, category or human perception requires independently defined evidence.

## Result and scope

The raw and scaled fits both put all material weight on aligned orientation. Aligned orientation had 99.2% positive training pair margins, with mean raw margin 0.2304 and logistic pair loss 0.1354. Sign-aware pixel alignment had 100% positive margins, with mean raw margin 0.0837 and loss 0.3342. The ranking loss preferred the larger margins despite the errors. The training-only scale diagnostic leaves this preference intact, so the experiment does not isolate raw units as the sole cause.

Sign-aware pixel alignment won development selection and scored 100% on the 576 test queries. Raw and scaled fits each scored 96.0%. The raw fit trailed by 4.0 percentage points, with a paired 95% interval of −5.4 to −2.6 points. Uniform fusion scored 79.3%. The bootstrap uses 4,000 resamples of galleries within the three fixed held-out generators, retaining every view and method together. These intervals condition on the generators, transformations and fit.

The strong single comparator has the known tile frame and exact permitted shift and sign search. The result addresses transfer across synthetic layout generators under those assumptions. It gives no estimate for unknown registration, natural images or semantic labels.

## Bridge to standard image data

The next implementation step is to replace the supplied periodic tile frame with a documented image adapter, including open boundaries, feature sampling, foreground coverage and explicit alignment policy. Keep the pair profile and the decision weights separate. Start with identity retrieval under labeled perturbations and an independently sourced distractor gallery. Use a separate semantic endpoint when testing class relevance. Split by source before transformation, choose models and any weighting by the actual development endpoint, and export both pair profiles and decision contributions for inspection. The new synthetic results argue for retaining strong single observables in that selection.
