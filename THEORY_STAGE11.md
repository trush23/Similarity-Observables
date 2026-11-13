# Stage 11 — CIFAR-10 scorecards and frozen encoder fusion

## Panels

The grayscale audit reuses all 12 Stage 10 galleries and reconstructs the same eight perturbations. It also adds the clean query. The held-out RGB panel uses seed 11427 to sample 12 new balanced galleries of 40 from the official test split. The new sampler excludes the earlier 480 test IDs and exact pixel duplicates of those images. Each gallery holds four per category. Every query excludes its own source, leaving three same-category matches among 39 candidates. The nine views provide 4,320 queries per panel. Clean and transformed queries are reported separately. All transformations and labels are shared across the methods within a panel.

## Common-offset profile

For each pair, select an integer translation by minimizing pixel ZNCC distance. The budget is ±3 pixels per axis and at least 65% shared valid canvas. Numerical ties within 10⁻¹² keep the first offset in order of squared displacement and then lexicographic order. Both local-Q comparisons use the selected offset. The audit uses the Stage 10 grayscale linear Q field and the fresh panel uses its RGB analogue. The offset can differ across candidates.

Write the aligned local fields as a·exp(2iθ) and b·exp(2iφ). The squared, normalized magnitude distance is the sum of (a−b)² divided by the square of the sum of the two field norms. The squared phase distance adds 2ab[1−cos(2(θ−φ))] at each site with the same denominator. These two maps sum exactly to the squared phase distance. A candidate margin subtracts the positive candidate's contribution from the distractor's. Positive margins favor the same-category candidate. The panels use each candidate's own eligible overlap and normalization, so differences can also reflect which sites are supported. Missing sites contribute zero to that candidate's map. The patch maps explain the measured score in query coordinates. They do not identify causal or semantic object regions.

The audit reports category top-1, AP, and unique top-1 switches. A rescue is a wrong-category magnitude winner replaced by a same-category phase winner. A harm is the reverse. If either top rank ties within 10⁻⁸, the example is excluded from switch counts but included in top-1 with fractional credit. Notebook 36 chooses the lower median by absolute angular margin for each direction among transformed switches. It reconstructs those pair margins from the source pixels and stored offsets.

## Frozen encoder comparison

The same ResNet-50 ImageNet V1 weights used in Stage 10 encode both RGB runs. The 56 pipeline bilinearly resizes the full image to 56 × 56, as before. The standard checkpoint transform first resizes to 256 and center-crops to 224, using the same channel normalization. The comparison therefore changes size and crop together. It does not isolate input resolution. Both runs take the global pooled feature before the classifier, normalize to unit L2 norm, and compare cosine distance. The model remains frozen. [TorchVision ResNet-50 V1 inference transform](https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.resnet50.html)

One fixed fusion ranks the 39 eligible candidates separately by standard-preprocessing encoder distance and phase-aware Q distance, then averages those two ranks with equal weights. Average ranks handle ties. The ensemble uses only within-gallery order, and its scores are candidate-set dependent. No weight, threshold, or encoder is selected on these galleries. Encoder-only retrieval is the primary comparator. Phase-only, magnitude-only, and the prior 56 × 56 encoder are reference rows.

Top-1 splits ties within absolute tolerance 10⁻⁸. Average precision averages precision at the ranks of the three eligible positives, taking the expectation over permutations within exact score ties. Report mAP as the mean of these query AP values. Both primary endpoints compare fusion with the standard encoder on transformed queries. The gallery bootstrap uses 4,000 paired resamples with pointwise 95% intervals. Clean outcomes are reported separately. These endpoints measure category retrieval within the sampled galleries.

## Findings

With a common offset, phase increased transformed grayscale category top-1 from 10.47% to 13.62%, a gain of 3.15 points (95% CI 1.09 to 5.49). It improved mAP by 3.06 points (1.78 to 4.48). There were 395 unique rescues and 274 unique harms across 3,840 transformed queries. On clean queries, phase top-1 was 16.67% and magnitude top-1 was 12.71%, showing that same-category retrieval was already difficult before perturbation.

On the fresh RGB galleries, standard ResNet-50 preprocessing raised transformed top-1 from 21.48% to 32.14% (+10.65 points, 95% CI 8.49 to 12.50) and mAP from 27.63% to 35.79% (+8.17 points, 7.05 to 9.26). The fixed fusion reduced transformed top-1 to 26.30% (−5.84 points, −8.10 to −3.56) and mAP to 30.57% (−5.23 points, −6.08 to −4.35). It also reduced both clean endpoints. The fusion result applies to this equal-weight ranking rule and these galleries.

## Reproduce

Use the benchmark requirements and the same official CIFAR-10 test file as Stage 10. The original gallery membership and checkpoint manifest must remain available. Standard preprocessing is the most expensive CPU step. Both encoder commands also accept a compatible Torch device through `--device`.

```bash
python -m observables.stage11 audit
python -m observables.stage11 fresh
python -m observables.stage11 56
python -m observables.stage11 224
python -m observables.stage11 summarize
python execute_notebooks.py --pattern '3[67]*.ipynb'
```
