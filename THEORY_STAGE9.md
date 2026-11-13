# Stage 9 — transfer to Fashion-MNIST

The experiment carries the Stage 8 retrieval protocol from original MNIST to Fashion-MNIST. It freezes the representations, local window, alignment budget, feature layers, query transformations and decision rules. The primary effect is the same-category difference between aligned local Q and aligned Q magnitude. The conventional normalized structure tensor is the principal established orientation comparator.

## Sources and exclusions

Fashion-MNIST contains 28 × 28 grayscale clothing and accessory images with ten categories. The benchmark reads the official gzipped 10,000-image test split. It checks the two published MD5 digests and IDX headers before parsing pixels or labels. [Fashion-MNIST data and label mapping](https://github.com/zalandoresearch/fashion-mnist)

The exclusion file records 3,080 distinct previously used test IDs. It combines the 20 sources from the earlier ten-class smoke panel with all 3,060 query and candidate IDs from the distractor experiment. The three-class smoke sources are included in the ten-class panel. The earlier smoke manifest used a 60,000 offset for test IDs, which is removed here. All IDs in this stage are zero-based test row indices. The sampler excludes exact pixel duplicates of prior IDs and of other newly sampled images. The exclusion file and its digest are retained for regeneration.

Seed 9327 generates 12 disjoint galleries of 40 images each, with four images per class. All 480 sources serve as queries under eight transformations, yielding 3,840 transformed queries and 153,600 query–candidate pairs. The matching MNIST comparison uses the fresh Stage 8 40-image galleries, with the same counts and target definition. The two panels have independent source draws. In this stage the term transfer means applying the fixed Stage 8 method to new Fashion-MNIST sources. It does not imply that the research program had never examined any Fashion-MNIST images.

## Frozen comparisons

Exact-source top-1 has one target among 40. Same-category top-1 removes the source and accepts any of three same-label images among 39. Chance is 2.5% and 7.69%, respectively. The eight Stage 6 recipes cover shift, gain, noise, blur, rotation, crop, occlusion and shift plus noise. Clean views form the encoder gallery and do not enter the reported averages.

The classical field methods use the same 49 possible open shifts and minimum 65% shared overlap. Each chooses its own eligible offset. ZNCC-registered HOG uses the offset found by ZNCC. Both frozen backbones, all feature layers and the full-frame 56 × 56 preprocessing are identical to Stage 8. Checkpoint hashes are compared in Notebook 32. The original uniform and omit-triplets profiles are fixed secondary comparisons. No weights, thresholds, search parameters or model settings are fitted on Fashion-MNIST. See [Stage 8 definitions](THEORY_STAGE8.md).

Both datasets receive the same absolute noise SD, 0.23 on the [0,1] intensity scale. Their original-image RMS distributions differ. The relative noise burden and the variability of same-class shape can differ across datasets even under the same protocol. The data cannot isolate those mechanisms from other dataset differences.

## Inference and challenge subsets

Compute paired method effects using 4,000 gallery bootstrap draws. For cross-dataset comparisons, independently resample each panel's galleries and subtract paired effect estimates. The primary result is the within-Fashion-MNIST phase contrast. The between-dataset interval describes whether the observed gain differs across the two tested panels. Other contrasts and per-category breakdowns are pointwise descriptive. The intervals condition on these datasets, gallery designs and transformation recipes.

Each same-size distractor challenge keeps the three legitimate same-category candidates. It adds either 16 random wrong-category images or the 16 wrong-category images nearest to the clean source on coarse 7 × 7 thumbnails. This construction uses the original only to select the candidate subset. Every method scores the transformed query and the identical subset. The result is a stress test for this coarse-pixel notion of hardness, not a generic measure of the difficulty of all distractors.

## Findings

Aligned local Q reached 90.86% exact-source and 53.41% same-category top-1. Keeping orientation phase improved on the matched Q-magnitude control by 18.93 percentage points for exact source (95% CI 17.73 to 20.08) and 12.47 points for same category (10.52 to 14.53). The same-category gain was 10.86 points on MNIST. The Fashion-MNIST minus MNIST difference in that gain was 1.61 points (−1.74 to 4.90). The larger Fashion-MNIST exact-source gain was concentrated in noise and shift plus noise.

The aligned conventional structure tensor reached 91.43% exact-source and 53.36% same-category top-1. For category retrieval, the aligned Q advantage over this tensor was 0.05 points (−0.86 to 1.02), compared with 2.29 points on MNIST. Against aligned pixels, the Fashion-MNIST category gain was 8.88 points (5.76 to 12.21). These contrasts separate the evidence for retaining phase from the evidence for the particular Q weighting.

At fixed challenge size, aligned Q scored 64.40% with random distractors and 54.66% with coarse-pixel hard distractors. Its category top-1 across the full galleries ranged from 24.0% for shirts to 82.8% for ankle boots. Omitting triplets from the uniform profile changed same-category top-1 by −0.08 points (−0.89 to 0.76), compared with +4.14 points on MNIST.

## Reproduce

Install `requirements-benchmarks.txt` and use the original test files from the Fashion-MNIST authors. The CLI reads `~/.keras/datasets/fashion-mnist` by default. Use `FASHION_MNIST_DIR` or the `--data` option to supply another cache. The saved exclusion CSV must be present.

```bash
python -m observables.stage9 classical
python -m observables.stage9 frozen
python -m observables.stage9 summarize
python execute_notebooks.py --pattern '3[1-3]*.ipynb'
```

The classical and frozen outputs use identical pair keys. The benchmark writes the sources, candidate distances, query outcomes and summary under `results/stage9/`. The notebooks use those results, plus the original data for illustrations. Existing MNIST artifacts are preserved.
