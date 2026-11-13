# Stage 12 — hierarchical CIFAR-100 retrieval

## Gallery design

Use the official CIFAR-100 Python archive. Verify its published MD5 before reading the test and label metadata members. Record the SHA-256 of the test member. The 10,000 test images have 100 fine labels nested in 20 superclasses. No training samples are used to set parameters. [CIFAR-100 dataset and download](https://www.cs.toronto.edu/~kriz/cifar.html)

Seed 12427 constructs 12 galleries of 40 RGB images at their native 32 × 32 resolution. In each of three rounds, randomly partition the 20 superclasses into four groups of five. In each appearance of a superclass, omit a different one of its five fine classes and sample two images from each of the other four. Every superclass appears in three galleries and all fine classes appear at least twice. The 480 sources are disjoint across galleries by row ID and exact pixel content. Each source contributes a clean query and eight fixed Stage 6 transformations, making 480 clean and 3,840 transformed queries.

## Two retrieval tasks on the same score matrix

| Task | Eligible candidates | Positives | Random top-1 |
|---|---:|---:|---:|
| Fine discrimination | 7 | 1 | 14.29% |
| Coarse resemblance | 38 | 6 | 15.79% |

Fine discrimination searches only the same superclass. The source is removed and the other image with the same fine label is the sole target. Coarse resemblance removes both images with the query's fine label. The six images in the other fine classes of the same superclass are targets, and the 32 other-superclass images are distractors. Use exactly these masks for all methods. The labels provide explicit retrieval targets, with no perceptual judgments.

## Frozen scoring and inference

Use the Stage 11 RGB linear Q field and its magnitude, with a shared offset chosen by pixel ZNCC independently for each pair. ZNCC searches ±3 pixels per axis with at least 65% common observed canvas. Aligned ZNCC is itself a baseline. The frozen ImageNet V1 ResNet-50 uses the same verified checkpoint and standard resize-to-256 then center-crop-to-224 preprocessing. It compares global pooled features by cosine distance. There is no fusion or fitting.

Separate the clean controls from the eight transformed conditions. The primary effects are phase minus magnitude on transformed top-1, separately at each label level. Secondary endpoints are mean average precision and within-query pair accuracy. Pair accuracy averages positive-versus-negative wins, gives half credit for ties, and has chance 0.5 on both candidate sets. The exploratory coarse-minus-fine phase contrast uses that endpoint. It compares effects across two different candidate constructions, so it cannot isolate label granularity from the change in negatives. Top-1 splits numerical ties and AP averages over exact score ties as in Stage 11. Report pointwise 95% intervals from 4,000 paired gallery resamples. The intervals condition on the gallery design and the represented superclasses.

## Findings

On transformed queries, phase increased fine top-1 from 16.15% to 18.83%, a gain of 2.68 points (95% CI −0.49 to 5.65). It increased coarse top-1 from 18.80% to 21.74%, a gain of 2.94 points (0.36 to 5.60). Within-query pair accuracy increased by 1.77 points for fine retrieval and 0.29 for coarse retrieval. The coarse-minus-fine phase effect was −1.48 points (−5.60 to 2.52), leaving the relative strength of the phase effect unresolved. Both transformed mAP intervals for phase minus magnitude include zero.

Aligned ZNCC reached 20.26% fine and 23.18% coarse top-1 on transformed queries. Standard-preprocessing ResNet-50 led at both levels, with 28.67% fine and 41.09% coarse top-1. Its clean controls reached 32.92% and 56.04%, respectively.

## Reproduce

Install the benchmark dependencies. Supply the verified archive with `CIFAR100_ARCHIVE` or `--data`. The default path is `~/.keras/datasets/cifar-100-python.tar.gz`. The saved notebooks replay the tables and the first gallery preview using only the project artifacts.

```bash
export CIFAR100_ARCHIVE=/path/to/cifar-100-python.tar.gz
python -m observables.stage12 classical
python -m observables.stage12 encoder
python -m observables.stage12 summarize
python execute_notebooks.py --pattern '3[89]*.ipynb'
```
