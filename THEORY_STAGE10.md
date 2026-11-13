# Stage 10 — phase, gradient weighting, and CIFAR-10

## Design

The MNIST and Fashion-MNIST experiments contrasted aligned local Q with its magnitude and with a conventional structure tensor. Here the four matched fields are the Cartesian product of two gradient weightings and two representations. For gradient magnitude `m`, direction `θ`, exponent `p ∈ {1, 2}`, and `n` observed gradient samples in the 5 × 5 window, the damped complex field is

```text
q_p = sum(m**p * exp(2j*theta)) / (sum(m**p) + n * 0.05**p).
```

The magnitude control uses `abs(q_p)`. In grayscale, the phase fields reproduce local Q for `p=1` and the normalized traceless conventional structure tensor for `p=2`. An undamped variant sets the floor to zero and returns zero at undefined sites. For a fixed `p`, changing the floor rescales a nonzero complex field by a positive real number. Its local axis cannot change. Retrieval may change because the distance weights locations differently after rescaling.

Every aligned method minimizes its own field distance over the same 49 integer translations and at least 65% shared canvas. A phase and magnitude field can select different offsets. Thus each retrieval contrast compares full matching procedures under a common search budget. The distance for both field types is the norm of the difference divided by the sum of field norms on the common mask. Pixel distance and zero-normalized cross-correlation use the same eligible offsets.

At a fixed offset, write the two local fields as `a exp(2iθ_A)` and `b exp(2iθ_B)`. The squared field difference at a site is `(a − b)² + 2ab[1 − cos(2(θ_A − θ_B))]`. The magnitude control retains the first term. The phase-aware comparison adds an angular disagreement weighted by the product of the two local magnitudes. The normalizer is identical at the same offset. The extra term can improve rankings when it penalizes distractors more than the intended match.

## Synthetic tests

The analytic two-gradient calculation in Notebook 34 establishes how increasing `p` favors the stronger orientation at fixed sample counts. The rendered crossing-edge test adds two tanh ramps to a gray background. It measures axial error at the center relative to the first edge's known normal. The second edge is nuisance by construction, even when it is stronger. Because image gradients add where the ramps overlap, the rendered case differs from a window containing two disjoint populations of gradient vectors.

The orientation sweep pairs 160 randomized base angles and noise realizations across two crossing angles (45° and 90°), six relative crossing strengths (0 to 1.25), and three noise levels (SD 0, 0.03, and 0.08 on [0,1]). The companion retrieval experiment uses 12 independent galleries with 24 edge images each, four per orientation at 30° spacing. Edge width, offset and strength vary within the gallery. The query has a crossing edge, noise, both, or a dimmed target with noise. Same-orientation retrieval excludes the source, leaving three matches among 23. The damped and undamped comparisons use identical queries.

For clean 90° crossings at a 0.5 nuisance-to-target strength ratio, quadratic weighting reduced mean axis error from 23.21° to 19.16°. With a stronger nuisance edge at ratio 1.25, quadratic weighting increased the error from 53.08° to 55.44°. In retrieval, keeping phase raised same-orientation top-1 by 20.66 points for linear weighting (95% CI 18.40 to 23.00) and 17.88 points for quadratic weighting (15.80 to 19.97). The two damped phase fields were close at 81.51% and 81.34%, with aligned ZNCC higher at 91.67%. Removing the floor reduced phase-field top-1 to 38.28% and 49.74%, respectively, while preserving the estimated local axes.

## CIFAR-10 transfer

Use the official 10,000-image Python test batch at native 32 × 32 resolution. The loader verifies the test file MD5 and layout, and records its SHA-256. Use 12 disjoint galleries of 40 exact-pixel-distinct originals, four per class. The seed is 10427. Each of 480 images is queried under the eight fixed Stage 6 perturbations, yielding 3,840 transformed queries per panel. Exact source has one target of 40. Category has three targets of 39 after source exclusion. Split top-1 credit across ties within absolute tolerance 10⁻⁸. No data from the training split or CIFAR-10 labels are used to fit a method. [CIFAR-10 dataset and test batch format](https://www.cs.toronto.edu/~kriz/cifar.html)

The primary panel projects RGB into grayscale with fixed weights `[0.299, 0.587, 0.114]` and then applies the frozen perturbations. The separate RGB panel uses the same image IDs, transformations, and random draws, including a common additive noise draw across RGB channels. Clipping is per channel in RGB and scalar in grayscale. RGB fields sum both gradient moments and their normalizing mass across channels before division, with the floor counting every observed channel sample. Replicating one gray channel three times therefore reproduces the same field. RGB pixels and ZNCC compare the channel values jointly. The color analysis changes the representation and can also change clipping under noise.

The frozen encoder baselines reuse Stage 8's ResNet-50 and EfficientNet-B0 ImageNet V1 checkpoints, intermediate mean and Q summaries, final cosine, and 56 × 56 full-frame preprocessing. For grayscale each encoder receives three replicated channels before its channel normalization. The RGB panel supplies the original three perturbed channels. The fixed 5 × 5 window and ±3-pixel search have slightly smaller relative extent on 32 × 32 CIFAR-10 than on 28 × 28 MNIST or Fashion-MNIST.

The primary contrasts are same-category phase minus magnitude at each exponent in grayscale. Secondary contrasts include exact source, the linear-versus-quadratic phase contrast, and the phase-by-weighting interaction. The RGB minus grayscale phase contrasts pair galleries by ID. Report pointwise 95% intervals from 4,000 paired gallery bootstrap draws. The intervals reflect the 12 sampled galleries under the fixed perturbation design. Category labels are retrieval targets, and these gallery results are separate from full-test classification accuracy or human similarity judgments.

## CIFAR-10 findings

In grayscale, phase increased exact-source top-1 by 19.92 points with linear weighting (95% CI 18.57 to 21.20) and 19.84 points with quadratic weighting (18.18 to 21.33). Same-category gains were 4.69 points (2.24 to 7.14) and 5.36 points (2.71 to 8.26), respectively. Most of the linear exact-source gain came from noise, shift plus noise, and blur. The grayscale linear-minus-quadratic phase contrast was −0.55 points for category (−1.25 to 0.16). In RGB, the category phase gains remained close at 4.48 and 4.77 points.

The grayscale linear phase gain for category was 6.17 points smaller than on MNIST (95% CI for CIFAR-10 minus MNIST −10.00 to −2.45) and 7.79 points smaller than on Fashion-MNIST (−11.09 to −4.56). These intervals independently resample galleries within each dataset.

Aligned ZNCC led exact-source retrieval at 98.54% in grayscale and 99.58% in RGB. The strongest same-category result among the tested methods was 20.68% for RGB ResNet-50 final cosine with the fixed 56 × 56 inputs. The grayscale phase fields reached 15.36% and 15.91% for category, against gallery chance of 7.69%.

## Reproduce

The synthetic study uses the base requirements. Install `requirements-benchmarks.txt` for the frozen encoder run. The CIFAR-10 test file defaults to `~/.keras/datasets/cifar-10-batches-py/test_batch`. Set `CIFAR10_TEST_BATCH` or pass `--data` for another location.

```bash
python -m observables.stage10 synthetic
python -m observables.stage10 classical
python -m observables.stage10 frozen
python -m observables.stage10 summarize
python execute_notebooks.py --pattern '3[45]*.ipynb'
```
