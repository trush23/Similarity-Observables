# Stage 8 — matched baselines and a fresh MNIST panel

Stage 8 expands the manuscript comparison and tests whether the orientation-phase gain survives a fresh set of MNIST sources. The existing 20-candidate pilot is retained as a separate panel. The fresh panel uses 40 candidates. The experiment compares both identity and same-digit retrieval.

## Panels and endpoints

- **Pilot.** Replay the Stage 6 source IDs, gallery membership and all eight transformed views. Reuse the saved Stage 6 profile distances. Compute the additional baselines on the same query images and gallery images. There are 16 galleries, 320 sources and 2,560 transformed queries.
- **Fresh replication.** Fix seed 8319. Sample 12 disjoint galleries of 40 originals from the official test split, with four examples per digit. Exclude pilot test IDs and exact image duplicates. There are 480 sources and 3,840 transformed queries. Apply the same eight transformation recipes as in the pilot. The generation seed and the sampled source IDs determine the entire run.

Exact-source top-1 has one relevant candidate in each full gallery. Same-digit top-1 removes the source and gives credit for any remaining same-label image. It has one relevant candidate among 19 in the pilot, and three among 39 in the replication. Their chance rates are 1/19 and 3/39. Tied top ranks split credit. Clean views anchor the frozen feature gallery but are excluded from the reported accuracies. The panels have different candidate counts and relevance counts, so their absolute scores should be read separately.

The primary planned replication contrast is **aligned local Q versus aligned Q magnitude for same-digit retrieval**. Prespecified comparison rows include aligned pixels, the normalized structure tensor, ZNCC and registered HOG. A secondary frozen Stage 7 check compares the existing uniform profile with the same profile omitting open Q triplets. There is no selection of new weights on either panel.

## Classical baselines and support

Every field search uses the same 49 integer offsets within ±3 pixels per axis, open boundaries, and at least 65% shared canvas coverage. Pixel, Q, normalized tensor, and ZNCC scores may choose different eligible offsets. Report each policy in the table.

- **ZNCC.** For each eligible offset, center query and candidate intensities on the shared overlap. Score (1−r)/2, where r is the normalized centered dot product. For a degenerate pair of equal centered arrays the correlation convention is 1, and otherwise it is 0 when the denominator vanishes. The fixed baseline uses offset zero, and the aligned baseline minimizes the distance over offsets.
- **Standard HOG.** Use scikit-image's block-normalized HOG with 9 orientation bins, 7 × 7 cells, 2 × 2 cell blocks and L2-Hys normalization. Compare descriptors by relative Euclidean distance. The registered variant takes the ZNCC-selected shift, places both images on the query canvas on common support, fills the rest with zeros, then recomputes HOG. This is a two-step registration baseline with the same allowed shift budget. The fixed variant uses the whole supplied canvas with missing query intensities filled by zero. [scikit-image HOG specification](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog)
- **Quadratic-gradient structure tensor.** Form the local traceless tensor by averaging g² for complex gradient g=gₓ+igᵧ in the 5 × 5 window. Divide by the local trace, average |g|², plus 0.05² times the observed window fraction. This is the trace-normalized traceless part of the conventional second-moment gradient tensor, encoded as a complex number. The scale, support and search match the tested Q field, which instead averages g²/|g| and divides by local |g| plus its own floor. The contrast tests the two full normalization choices. It does not isolate an exponent independently of the denominator. [Structure tensor literature](https://lmbweb.informatik.uni-freiburg.de/Publications/2006/Bro06c/)

## Frozen learned baselines

Use torchvision ResNet-50 ImageNet-1K V1 and EfficientNet-B0 ImageNet-1K V1, in evaluation mode. Resize the entire grayscale canvas to 56 × 56 by bilinear interpolation, broadcast to RGB, and use each checkpoint's channel normalization. The input keeps the full canvas and uses no classification center crop. Missing intensities are zero. No MNIST fine-tuning, rotation compensation or translation search is applied to the frozen rows.

For each architecture compare cosine between final globally averaged feature vectors. Separately, compare the first and traceless second moments of unit local features from its stride-eight intermediate grid. Exclude zero local vectors from the second moment's count. Unit-normalize the flattened moment before cosine comparison. The first-order and second-order comparisons use the same feature grid, whereas the final embedding comes from a later layer. The input size is deliberately fixed across architectures and is below the standard classification recipe. The frozen rows characterize this setting. [ResNet-50 weights and preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html) · [EfficientNet-B0 weights and preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html)

## Equal-size distractor test

Within the fresh panel, construct two secondary same-digit retrieval subsets of 19 candidates per query. Both retain the three same-digit exemplars. The random subset adds 16 wrong-label images sampled with a fixed per-source seed. The hard subset adds the 16 wrong-label images nearest to the clean source in Euclidean distance on 7 × 7 average-pooled pixels. The clean source is available only to construct the challenge. Actual candidate scores use the transformed query. All methods are scored against exactly the same subsets. The challenge measures difficulty under this coarse-pixel selection rule.

## Inference and reproducibility

Paired differences and percentile intervals resample whole galleries with 4,000 draws. They condition on the MNIST source and gallery design and these eight perturbations. Secondary contrasts have pointwise intervals without multiplicity adjustment. The test does not expand to a different dataset or different perturbation distributions.

Install `requirements-benchmarks.txt`. The original MNIST archive and frozen weights are read from the caches described in Stages 4 and 6. To regenerate the baseline scores and manuscript table, run:

```bash
python -m observables.stage8 classical
python -m observables.stage8 frozen
python -m observables.stage8 summarize
python execute_notebooks.py --pattern '2[89]*.ipynb'
python execute_notebooks.py --pattern '30*.ipynb'
```

The two scoring commands have separate outputs. The summary requires their pair keys to match one to one and reconciles pilot keys with the existing Stage 6 run. Scoring times and checkpoint digests are recorded separately. The reported timings are aggregate work for descriptor groups, including extraction and scoring as indicated, rather than a per-method latency comparison.

## Findings

The fresh-panel primary contrast replicated the pilot phase gain. Aligned local Q achieved 64.79% same-digit retrieval versus 53.93% for aligned Q magnitude, a difference of 10.86 points (paired 95% interval 8.12–13.78). The pilot difference was 10.94 points. Because the relevant-candidate count changes across panels, compare the within-panel contrasts.

The aligned conventional tensor was the strongest new classical comparator. On the fresh panel it scored 95.16% exact source and 62.50% same digit, versus 95.26% and 64.79% for aligned local Q. The exact-source difference was +0.10 points (−0.13 to +0.36). The same-digit difference was +2.29 points (0.94–3.54). Aligned pixels scored 93.91% and 52.27%. Aligned ZNCC scored 92.81% and 55.70%.

With 19 candidates and the same relevant examples, aligned Q fell from 75.44% with random wrong-digit examples to 68.39% with thumbnail-selected hard ones. The conventional aligned tensor fell from 73.20% to 66.28%. These are equal-size gallery comparisons rather than a penalty for adding candidates.

Within each frozen encoder's intermediate feature grid, second-order pooling improved fresh-panel same-digit accuracy over the matched first-order mean by 5.36 points. The paired intervals were 4.43–6.35 for ResNet-50 and 4.32–6.48 for EfficientNet-B0. EfficientNet's final pooled cosine still scored higher than its intermediate Q on this endpoint, at 40.81% versus 36.28%. Layer choice matters alongside pooling.

The Stage 7 triplet omission also replicated for the frozen uniform rule. Fresh same-digit top-1 rose from 45.70% to 49.84%, a gain of 4.14 points (2.37–5.89). This remains below the single local-Q scores. The result validates the omission for this specific uniform rule, and the synthetic third-order collision result remains a separate statement of resolving power.
