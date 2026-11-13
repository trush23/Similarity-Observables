# Stage 6 — finite image support and original MNIST

The profile now operates on a finite grayscale canvas. Image values and valid support are separate inputs. This makes an observed zero, such as dark background or an opaque black occluder, distinguishable from an unobserved pixel outside a crop or translated canvas.

## Gradients and orientation

Compute central differences when both adjacent samples are observed. Use a one-sided difference when only one neighbor is observed. Across an axis with neither neighbor, use zero derivative. There is no derivative between an observed and a missing pixel. Treat images as grayscale intensities in [0,1] to give the reliability constants fixed units.

For complex gradient g=gₓ+igᵧ and magnitude m=|g|, use magnitude-weighted moments mₖ=Σm(g/m)ᵏ/Σm for k=1,2,4. Zero gradients carry zero weight. The real and imaginary parts of m₂ are the two independent entries of the 2D traceless orientation tensor up to a factor 1/2. Polar order retains gradient sign. Orders 2 and 4 identify opposite directions. The global polar moment need not vanish with these finite boundary conventions.

The local field q is computed on a 5 × 5 sliding window. Divide the sum of g²/|g| by the sum of |g| plus 0.05 times the number of observed pixels in the window. Set q to zero at unobserved centers. The denominator damps weak-gradient measurements. The comparison therefore retains both orientation and local strength. Two aligned controls separate some of this information: |q| retains the magnitude of local order without its phase, and the scalar gradient-strength field uses Σ|g|/(Σ|g|+0.05n) and retains edge strength regardless of local order.

## Pair profile

The profile reports 11 named agreements. Feature comparisons use 1 minus the norm of the difference divided by the sum of norms. Two zero features give agreement 1 by convention. The aligned magnitude and gradient-strength controls are additional benchmark endpoints. The uniform profile averages the original 11 agreements only.

| Agreement | Representation |
|---|---|
| Pixels | Fixed-canvas intensities on shared observed support |
| Polar m1, global Q, angular m4 | Magnitude-weighted complex angular orders 1, 2 and 4 |
| HOG | 4 × 4 spatial grid of 9 unsigned orientation bins, gradient weighted with damped local normalization |
| Local Q | Complex nematic field on shared observed support |
| Open Q pairs | Eight finite-lag averages of Re[q(x) conjugate(q(x+r))] |
| Open Q triplets | 64 finite three-point averages of products of Re q at the anchor and two lags |
| DCT power | Squared coefficients of the orthonormal DCT, after subtracting the observed-pixel mean and filling missing values with centered zero |
| Aligned pixels, aligned local Q | Minimum field distance over the same 49 integer shifts within ±3 pixels per axis |

The lag list is (0,1), (1,0), (1,1), (1,−1), (0,2), (2,0), (0,4), (4,0). A lag average uses only anchors where every requested site is inside the canvas and observed. An empty set contributes zero in this implementation. Pair statistics keep the real contraction. Triplets use the real nematic channel in the common image frame. These choices preserve a fixed orientation frame and introduce projection ambiguities. DCT power uses the finite cosine basis. Its missing-support fill and loss of coefficient signs affect what it measures.

Each alignment compares only jointly observed samples. A candidate offset is eligible if the overlap covers at least 65% of the full canvas. The symmetric search contains both signs of every offset. The two fields can choose different best offsets. The pair API returns the offsets and coverage for pixels and local Q. An insufficient-overlap pair receives zero registered agreement with zero coverage. The distances do not assign any disagreement to material outside the observed intersection. Perfect overlap agreement can therefore coexist with missing strokes.

## Original MNIST protocol

The dataset is the Keras-distributed original MNIST NPZ with SHA-256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`. The run uses the cached archive and verifies its digest before loading. Digits remain at native 28 × 28 resolution. [Keras source and dataset provenance](https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py)

Six development galleries sample the official training split, and 16 test galleries sample the official test split. Each gallery has two originals per digit, and no original is reused within its split. Development chooses the best observed method for each endpoint by mean transformed top-1 accuracy. The transformed test comparisons have no fitted weights. A fixed equal-weight profile is included.

Every source produces clean plus eight controlled views: shift by a signed 2 or 3 pixels on each axis, gain 0.55, Gaussian noise SD 0.23 with clipping, Gaussian blur SD 1.25 pixels, rotation ±18° with bilinear interpolation, a missing six-pixel side, a black 7 × 7 occluder, and shift plus noise. Open shifts and crops expose a valid-support mask. Rotation marks nearest-neighbor support and fills unsupported image samples with zero. Other corruptions retain full support. The perturbation seed and gallery membership are saved.

- **Identity retrieval** finds the original from 20 candidates. Chance is 5%.
- **Same-digit retrieval** removes the query's original and seeks the one other sample with the same label among 19 candidates. Chance is 1/19.

There are 320 distinct test originals and 2,560 transformed test queries. Clean identity results serve as a control and are excluded from transformed averages. Top-1 ties split credit within tolerance 10⁻⁸. Paired 95% intervals resample the 16 galleries, retaining common distractors and all associated views. The results address this retrieval design with limited balanced galleries. Same-digit labels provide an explicit class target and do not supply perceptual judgments. Full-dataset classifier accuracy is a different endpoint.

## Reproduce

Run `python -m observables.stage6` from the project using the base dependencies. It reads `~/.keras/datasets/mnist.npz` by default. Set `MNIST_NPZ` to a verified copy elsewhere or pass `--data` to the benchmark. The official Keras archive is available at [mnist.npz](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz). The project does not bundle the dataset.

Run `python execute_notebooks.py --pattern '2[2-4]*.ipynb'` for the three Stage 6 notebooks. Notebook 23 reads the saved run by default. Set its RECOMPUTE flag to regenerate it. The synthetic boundary checks in Notebook 22 require no dataset. The raw candidate profiles and named scorecard in Notebook 24 support inspection of individual decisions.

## Findings on the held-out galleries

Aligned local Q achieved 97.7% transformed identity and 45.3% same-digit retrieval. Aligned pixels achieved 95.9% and 37.5%. The paired gallery-bootstrap gains for aligned Q are 1.8 points (95% interval 1.3–2.4) and 7.8 points (4.4–11.1), respectively. The identity advantage comes entirely from the ±18° rotation condition. Both registered methods recovered all sources in the other seven transformed conditions. Translation search reduced identity accuracy on rotation relative to each field's fixed-coordinate score.

With the same search, aligned Q magnitude achieved 34.3% same-digit retrieval, and aligned gradient strength achieved 21.6%. Relative to Q magnitude, retaining the complex orientation phase raised same-digit retrieval by 10.9 points (paired 95% interval 6.8–14.8). This comparison conditions on the Q field and the common search. Unregistered local Q achieved 46.6% same-digit retrieval. Its 1.3-point difference from aligned local Q has a paired interval from −1.6 to +3.8 points.

The clean polar moment was numerically zero for 281 of 320 originals. The summed gradients are sensitive to the boundary bands and can cancel despite strong internal strokes. The uniform profile reached 30.3% same-digit retrieval. The scorecard illustrates a blurred 7 for which local Q favors the other 7 while the global fourth moment and DCT power favor a 6 strongly enough to change the combined decision.
