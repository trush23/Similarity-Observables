# Stage 4 — what frozen features preserve

This stage uses a frozen encoder on the synthetic controls from Stage 3. It separates changes in the representation from changes in the observable applied to it.

## Encoder and input geometry

The encoder is torchvision ResNet-50 with the ImageNet-1K V1 checkpoint, in evaluation mode. The manifest records the checkpoint digest, package versions and clipping rate. The feature comparisons use no classification head.

A common affine map sends signed grayscale samples to 0.5 plus a fixed multiple of the intensity. The peak of the clean wave motif maps to an excursion of 0.25. Samples outside [0,1] are clipped. The grayscale array is broadcast to three channels and normalized with the checkpoint's channel statistics. The controlled spatial preprocessing differs from the standard ImageNet resize and center crop so that every known tile remains present. [Torchvision ResNet-50 specification](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)

- **Isolated patches.** Resize each 8 × 8 tile to 32 × 32 with bilinear interpolation and encode it independently. Extract the unit-normalized pooled layer 4 embedding. The sign-symmetric representation averages the unit vectors for the two polarities and normalizes the result. Thus changing an entire tile's sign exchanges two summands and preserves its descriptor.
- **Contextual features.** Repeat the 8 × 96 strip vertically to make a 96 × 96 image, repeating its noise as well as its motifs. Use the network at this resolution. Layer 1 yields a 24 × 24 map and layer 4 a 3 × 3 map. Normalize the global average of each map to define the whole-image embeddings. The whole-image path uses the supplied signed intensities. It has no per-tile sign symmetrization.

The same checkpoint is used for both paths. Their input sizes, access to context and polarity treatment differ. Comparing the two paths therefore compares complete pipelines. The controlled observable comparisons occur within each path. This stage uses one pretrained checkpoint and gives no estimate of the effect of pretraining itself.

## Exactly what independent encoding preserves

If images have the same tile multiset, every deterministic independent encoding produces the same descriptor multiset. Hence every order of globally pooled tensor moments agrees. For the two-state homometric pair, the entire cyclic two-state table at each lag is also equal. Applying any fixed map independently to the two states leaves this two-point collision intact.

For unit vectors in each patch collection, write Q = mean(uuᵀ) − I/d. Its Frobenius distance to another Q can be computed without constructing a large tensor. Expand the squared difference and use the inner product of rank-one matrices:

```math
\|Q_A-Q_B\|_F^2 =
E_{u,u'\sim A}(u^T u')^2 + E_{v,v'\sim B}(v^T v')^2
-2 E_{u\sim A,v\sim B}(u^T v)^2 .
```

All draws in these products are independent uniform draws from the respective finite descriptor collections. The isotropic subtractions cancel. Patch order remains available until a pooling choice removes it.

To compare scalar pair and triplet correlations, choose the axis defined by the two sign-symmetric wave reference embeddings a and b. Project a descriptor z as

```math
p(z)=\frac{2(z-(a+b)/2)^T(b-a)}{\|b-a\|^2}.
```

This maps a to −1 and b to +1. The orientation identities of the two wave reference tiles are available when defining the gallery. There is no fitted projection using query motifs, noise or outcomes. A transferred motif can land on either side of zero and can exceed the reference scale. The triplet comparison uses all cyclic lag pairs, while the ordered alignment baseline uses the full vectors over all twelve shifts.

For two noiseless transferred motif values p₀,p₁, write p=β+γs where s=2x−1 and γ=(p₁−p₀)/2. Expanding the triple product shows that the difference between the two arrangements is γ³(T_A−T_B). The lower terms cancel because their single-site counts and every two-site table agree. Thus a smaller separation along the reference axis attenuates the absolute third-order contrast cubically. Notebook 16 verifies this identity on both transferred motif families. A distance-based retrieval rule can still succeed with small noiseless contrast.

## Context and coordinate dependence

When an encoder sees the entire strip, neighboring tiles can affect a feature at a location where the tile itself is identical. The receptive field, padding and strides all contribute to the behavior of the implemented encoder. A measured difference after contextual encoding is therefore evidence that the independent-tile argument no longer applies. It does not establish that the representation isolates arrangement as a nuisance-invariant quantity.

Compare a fixed phase reference with a phase bank. The bank contains all twelve allowed cyclic tile shifts, each encoded from pixels. Both candidate classes receive the same search. Shifting a feature map is a different operation from shifting and re-encoding the image, especially when stride and padding matter. The whole-image phase bank tests the latter directly.

## Retrieval and uncertainty

Reuse the first 16 trajectory IDs in each of the two Stage 3 prototype strata, generated with the original 48-trajectory generator to preserve the exact random draws. Each trajectory has 15 views from three motif families and five perturbation conditions. The retrieval experiment thus has 32 trajectories, 480 views, and 4,320 model-based query-method outcomes. The matched pixel control evaluates those same 480 views.

The gallery uses only the clean wave motifs. Whole-tile cyclic shifts, independent tile sign reversals, and Gaussian noise at SD 0.35, 1 or 2 times the clean signal RMS are the controlled perturbations. Phase search uses no known test shift. The match label concerns the arrangement of axes. Near ties within absolute distance 10⁻⁷ split credit. The control retains its Stage 3 tolerance of 10⁻⁹.

The 4,000 stratified resamples reuse the same draws across methods. Intervals condition on two fixed arrangements, one frozen model, a supplied tile frame and the stated perturbation generator. They quantify variability of these synthetic queries. Performance on unseen arrangement families, unknown tile layouts, and natural images requires a separate design.

## Reproduce

Install `requirements-learned.txt` in the project Python environment. A first run needs the ResNet-50 V1 weights in the Torch cache or access to download them. Use `python -m observables.stage4` to run the benchmark, and `python execute_notebooks.py --pattern '1[6-8]*.ipynb'` to run the three notebooks. Use `--keep` on the benchmark command to select a smaller diagnostic run. That command overwrites the model-only benchmark tables and manifest for its selected sample size. Notebook 18 combines the saved model scores with the matched pixel control. Set its RECOMPUTE flag to True to regenerate model scores from the checkpoint.
