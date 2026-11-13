# Stage 7 — redundancy and intervention response on MNIST

This stage addresses roadmap items 3 and 7 together. It uses the frozen Stage 6 candidate distances, gallery membership and intervention definitions. No weights, methods or nuisance search settings are fitted on these test results. The audit manifest records the SHA-256 digests of the Stage 6 inputs.

## What it means for coordinates to agree

The rank audit uses all 13 recorded distance coordinates, including the two magnitude and strength controls. For each source and each view, omit the source from its gallery and compute Spearman correlations across the 19 remaining candidate distances. Round distances to ten decimal places to suppress numerical differences and use average ranks for ties. An all-tied component has no defined correlation for that query. Export the mean correlation and the count of defined queries for each component pair and condition. The condition means are descriptive. They are averaged with equal weight when reporting an across-condition figure.

This asks whether two coordinates impose similar orderings on the same alternatives. It differs from correlation across pooled images, where gallery membership and condition shifts can dominate the signal. It also differs from agreement at the very top of a ranking.

For each endpoint, a hard miss means the target is absent from the top tie set. The hard-miss overlap is intersection divided by union across the eight transformed conditions. Fractional top ties are counted separately. Identity has the original as target. Class retrieval removes the original and targets the one other example of the same digit. Shared misses are descriptive and can be influenced by a method's overall error rate.

Fixed and aligned local Q had mean rank correlation 0.835 over the nine conditions. In exact-source retrieval, 28 of the 283 views missed by at least one were missed by both. For same-digit retrieval, the corresponding counts were 1,145 of 1,623. Open Q pair and triplet profiles had mean rank correlation 0.088. These observations distinguish overall rank redundancy, decision overlap and shared mathematical construction.

## Intervention responses

Each perturbed query is paired with its clean view of the same source and scored against the same gallery. For a given endpoint, let p be the distance to the designated target and n be the smallest eligible impostor distance. The audit reports two changes, perturbed minus clean:

- Δp measures how the intervention moves the target score.
- Δ(n−p) measures how the intervention changes the target's separation from its nearest impostor.

The candidate achieving n can change. Thus the margin is a decision statistic with a fixed candidate set, not a direct measurement of one fixed pair. In class retrieval, p is the distance to a distinct handwriting example and is generally positive before any perturbation.

The paired intervals use 4,000 percentile bootstrap draws of whole galleries. This preserves common distractors and repeated views. The panel has 16 galleries, 320 sources, eight transformed views per source and one clean view. The intervals condition on this design and are marginal across coordinates and conditions.

The mean exact-source distance for a pure shift rises by 0.488 for fixed local Q and by 0.028 for aligned local Q. Aligned pixels retain distance zero to the source. The fixed-Q same-digit margin drops by 0.071 on shift. Aligned Q's mean margin change is +0.005. These are overlap-dependent comparisons with the support convention from Stage 6.

## Synthetic controls and cross-response

The synthetic controls manipulate a known property on finite images. The open ramp has nonzero net gradient. It makes polarity, gain and a quarter-turn check interpretable. The compact shape fits within the translated canvas. Identical tile motifs are rearranged to separate global inventory from position.

- Additive offset changes intensity while preserving gradients and centered DCT power.
- Positive gain preserves normalized global angular orders. Damped local Q responds to gradient amplitude because its denominator contains an additive reliability term. A gain of 0.55 changes mean exact-source local-Q distance by 0.089 on MNIST.
- DCT power scales as gain squared. At gain 0.55 its relative distance is (1−0.55²)/(1+0.55²)=0.536. The synthetic control and MNIST agree on this value.
- Contrast reversal negates the first angular moment and preserves the second and fourth. A quarter turn on the ramp negates its second angular moment and preserves its fourth.
- Rearrangement of equal-inventory motifs leaves global Q unchanged while changing the aligned local Q field. An open boundary can change polar order through its edge bands.

These experiments explain measured component responses under specific interventions. A component's label alone cannot establish specificity. Sensitivity to a known change may be useful for one task and interfere with another. The same-label target does not supply a perceptual judgment.

## Leave-one-component diagnostic

The pre-existing uniform profile averages 11 distances. Remove each coordinate in turn and divide the remaining sum by ten. Keep all candidate scores, support rules and top-tie handling fixed. The orientation-removal controls were outside the original uniform profile and are outside this diagnostic. Report top-1 change versus the full uniform profile and a paired gallery-bootstrap interval.

Omitting open Q triplets raises same-digit retrieval from 30.3% to 33.0%, a 2.7-point gain with a pointwise interval of 1.1–4.5 points. Identity rises from 92.3% to 94.0%. This is a test-panel diagnostic for one decision rule. Eleven omissions were evaluated, and the intervals have no multiplicity correction. The results provide hypotheses for independent development and test data. The exact synthetic separation by triplets remains valid.

## Reproduce

Use the saved Stage 6 outputs under `results/stage6/`. Run `python -m observables.stage7` to regenerate the audit tables, then `python execute_notebooks.py --pattern '2[5-7]*.ipynb'` to regenerate the notebook figures. The audit uses the base dependencies and requires no raw dataset download.
