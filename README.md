# Similarity through a hierarchy of observables

This is a runnable foundation for a research program built around the profile S(A,B). Its two objectives are discrimination of known structural differences and explanation of agreement between representations. Eight notebooks progress from definitions and exact identities to designed ambiguities, uncertainty and controlled retrieval.

## Start and run

Use Python 3.11 or 3.12. All configurations are generated locally. No dataset download, model weights or GPU is required. From the extracted project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python execute_notebooks.py
python -m jupyter lab notebooks
```

On Windows, activate with `.venv\Scripts\activate`. The notebook runner uses the interpreter that invokes it. Each notebook also runs independently from the project root or notebook directory, and regenerates its own tables and figures. Execution overwrites corresponding results and notebook snapshots. To regenerate only the retrieval tables, run `python -m observables.benchmark`.

The delivered notebook files contain executed results. The formal definitions, proofs and assumptions are in [THEORY.md](THEORY.md). The experiment code is under `observables/`; the notebook cells retain the experimental choices and plotting instructions.

## Notebook sequence

| Notebook | Question | Main evidence |
|---|---|---|
| [00 · Research contract](notebooks/00_Research_contract.ipynb) | What is the object being compared? | Configuration, observable and decision layers; checkerboards, stripes and patches. |
| [01 · Angular order](notebooks/01_From_cosine_to_angular_order.ipynb) | How does cosine extend to higher angular order in dimension d? | Q identity, Legendre/Gegenbauer distinction, tensor–kernel contraction. |
| [02 · Moment collisions](notebooks/02_Moment_collisions.ipynb) | When do equal lower moments conceal different order? | Equal means with different Q; equal means and Q with different fourth moments. |
| [03 · Position and sequence](notebooks/03_Position_and_sequence.ipynb) | What does global pooling erase? | Identical feature multisets with different lag correlations and spectra. |
| [04 · Homometry](notebooks/04_Homometry_and_higher_correlations.ipynb) | Can power and pair correlation still miss structure? | A cyclic binary pair with equal power and autocorrelation, separated by third-order correlations. |
| [05 · Similarity profiles](notebooks/05_Similarity_profiles.ipynb) | Which properties agree in controlled images? | Nine-component profiles, local/global orientation, full/radial power and a representation blind spot. |
| [06 · Null models](notebooks/06_Uncertainty_and_null_models.ipynb) | How much empirical order is sampling noise? | 3,600 random configurations and an exact finite-sample expectation; repeated-direction dependence control. |
| [07 · Retrieval and program](notebooks/07_Controlled_retrieval_and_research_program.ipynb) | Which invariances preserve the chosen labels? | Eight prototypes, 240 independent base queries, four noise levels, five observable comparisons and bootstrap intervals. |

## Findings established in the delivered run

- Four cardinal directions and four diagonal directions have equal zero first and second angular moments. Their fourth-order squared harmonic distance is 4.
- Blocked and alternating sequences contain identical direction multisets and therefore agree in every global angular moment. Their nearest-neighbor Cxx correlations are 0.875 and −1.
- The cyclic occupancy sets {0,1,4,6} and {0,1,3,7} have equal power and autocorrelation to floating-point precision. Exhaustive shifts/reflections confirm distinct structures. Their bispectra and selected third-order correlations differ.
- For iid isotropic directions, the expected squared empirical Q norm is (1−1/d)/N. The Monte Carlo experiment verifies the finite-sample baseline at d = 2, 3, 16 and N = 8, 32, 128.
- Periodic central-difference image gradients have an exactly vanishing global first moment under magnitude weighting. A single-pixel checkerboard also has zero central-difference gradient everywhere despite nonzero intensity variance. These are representation constraints, which a profile must disclose.
- In the noiseless, randomly phase-shifted and sign-reversed stripe benchmark, Q recovers the axis label on 100% of queries but the scale label on 50%. Radial power recovers scale on 100% but axis on 50%. Full power recovers both on 100%. Fourth angular order identifies perpendicular axes, giving 50% axis accuracy. These controlled rates reflect chosen symmetries and labels.

## Experimental choices and scope

Synthetic images are 64 × 64 periodic fields, with signed intensities. Angular unit-vector fields and symbolic sequences provide dimension-independent controls. Image orientations use central differences, gradient-magnitude weights, and normals to edges. Nine profile components compare mean intensity, variance, gradient energy, global polar/nematic/fourth moments, grid-local nematic order, full power, and radial power.

The comparison convention is 1 − ||a−b||/(||a||+||b||), with both numerically zero treated as agreement 1. Observable magnitudes must accompany this score. An agreement of 1 between two vanishing quantities conveys absent signal. This reporting convention is not fitted. The mathematically derived harmonic kernel distance is also implemented separately.

Retrieval queries use four axes, two wavevector scales, 30 independent phase/sign draws per combination and Gaussian noise SD 0, 0.1, 0.3, 0.6. The same query and gallery serve every observable. Ties within absolute distance tolerance 10⁻⁸ split credit. Intervals use 2,000 stratified resamples of base sources, keeping the eight designed prototype types equally weighted. They condition on the generator, gallery and representation. No fitted classifier or real-world semantic interpretation is evaluated.

The program gives a concrete foundation and subsequent research stages. Learned patch features, pretrained whole-object embeddings, natural-image recognition and language tasks are planned extensions. The symbolic-sequence experiments test ordering itself. Their units have no linguistic semantics. Novelty of the broader combination remains a literature-review question.

## Validation and outputs

All eight notebooks executed top to bottom. Nineteen tests check analytical identities, dimensional conventions, moment collisions, shared-coordinate equivariance, homometry, direct versus Fourier autocorrelation, bispectral translation invariance, periodic-gradient constraints and invalid directions. Notebook assertions add experiment-specific checks. Twelve generated figures were rendered and visually inspected.

Results include PNG figures and CSV tables for image profiles, moment collisions, isotropic nulls and every retrieval outcome. Environment and package hashes are recorded in `results/validation.json`. The lightweight project was tested on CPU with Python 3.12.2, NumPy 1.26.4, SciPy 1.13.1, pandas 2.2.3 and Matplotlib 3.10.0.

Primary mathematical references and neighboring research are linked in THEORY.md. They guide the definitions; the controlled numerical results are generated by the accompanying code.

## Stage 2 — spatial representations

Notebooks 08–11 extend the foundation to arrangements with the same patch inventory, multiscale Q-fields, spatial correlations and registration-aware retrieval. The definitions and group actions are in [THEORY_STAGE2.md](THEORY_STAGE2.md).

## Stage 3 — what the observables discard

| Notebook | Controlled experiment |
|---|---|
| [12 · Homometric orientation fields](notebooks/12_Homometric_orientation_fields.ipynb) | Realized tile Q-fields share every cyclic two-point statistic. A third-order lag separates them. |
| [13 · Vector correlations and trace](notebooks/13_Vector_correlations_and_trace.ipynb) | Equal traces hide opposite director rotation. A signed off-diagonal projection reveals it at the same scalar budget. |
| [14 · Open boundaries and cropping](notebooks/14_Open_boundaries_and_cropping.ipynb) | Enumerate all cuts and distinguish boundary information from ambiguity in the observed window itself. |
| [15 · Motif transfer and noise](notebooks/15_Motif_transfer_and_noise.ipynb) | Transfer from wave motifs to ridges and bars under controlled noise and compare with sign-aware pixel alignment. |

See [THEORY_STAGE3.md](THEORY_STAGE3.md) for definitions, proofs and sampling assumptions. Run `python -m observables.stage3` to regenerate the tables, or `python execute_notebooks.py --pattern '1[2-5]*.ipynb'` to regenerate the four notebooks and figures. Results are in `results/stage3/`. These experiments use synthetic tiles and a supplied tile frame. Learned representations are the next stage.

## Stage 4 — frozen encoders on controlled inputs

| Notebook | Controlled experiment |
|---|---|
| [16 · Frozen patch geometry](notebooks/16_Frozen_patch_geometry.ipynb) | Compare orientation and contrast polarity in isolated tiles, verify exact pooled collisions and measure transfer of a fixed orientation axis. |
| [17 · Context and whole-image embeddings](notebooks/17_Context_and_whole_image.ipynb) | Hold a central tile fixed while changing context. Compare local features and whole-image pooling across all 12 phases. |
| [18 · Frozen-encoder retrieval](notebooks/18_Frozen_encoder_retrieval.ipynb) | Hold each representation fixed while varying pooling or phase search. Compare with a matched sign-aware pixel baseline. |

See [THEORY_STAGE4.md](THEORY_STAGE4.md) for the checkpoint, preprocessing, derivations and scope. Install `requirements-learned.txt` to run Stage 4. The first inference requires cached ResNet-50 ImageNet-1K V1 weights or access to download them. Run `python execute_notebooks.py --pattern '1[6-8]*.ipynb'` for the new notebooks. Notebook 18 reads the included frozen run by default and supports fresh inference with `RECOMPUTE=True`. A direct fresh benchmark is available via `python -m observables.stage4`. Saved scores, figures, and weight provenance are in `results/stage4/`.

The base requirements suffice for Notebooks 00–15. Use `--pattern '0*.ipynb'` and `--pattern '1[0-5]*.ipynb'` to execute those notebooks without the optional encoder dependencies.

## Stage 5 — interpretable pair profiles and family transfer

| Notebook | Experiment |
|---|---|
| [19 · Pair profiles and family split](notebooks/19_Pair_profiles_and_family_split.ipynb) | Build an image-only S(A,B), check symmetry, and construct six disjoint synthetic arrangement families. |
| [20 · Fit and held-out evaluation](notebooks/20_Fit_and_heldout_evaluation.ipynb) | Fit nonnegative weights with and without training-only scale calibration. Compare against a development-selected single observable. |
| [21 · Decision explanation and image bridge](notebooks/21_Explain_decisions_and_image_bridge.ipynb) | Attribute a held-out error exactly and specify the protocol for a standard-image pilot. |

[THEORY_STAGE5.md](THEORY_STAGE5.md) defines the profile and decision rule. The Python entry points are `observables.stage5.similarity_profile` and `observables.stage5.weighted_similarity`. The base requirements suffice. Run `python -m observables.stage5` for the benchmark or `python execute_notebooks.py --pattern '19*.ipynb'` and `python execute_notebooks.py --pattern '2[01]*.ipynb'` for the notebooks. Tables, figures, and the fitted weights are in `results/stage5/`.

## Stage 6 — finite boundaries and original MNIST

| Notebook | Experiment |
|---|---|
| [22 · Open image boundaries](notebooks/22_Open_image_boundaries.ipynb) | Validate finite shifts, missing support, gradient boundaries, and shared-overlap alignment. |
| [23 · MNIST finite image benchmark](notebooks/23_MNIST_finite_image_benchmark.ipynb) | Separate exact-source and same-digit retrieval. Compare pixel and local Q alignment and orientation-removal controls. |
| [24 · MNIST scorecard](notebooks/24_MNIST_profile_scorecard.ipynb) | Reconstruct a held-out digit query and attribute its decision to the component agreements. |

See [THEORY_STAGE6.md](THEORY_STAGE6.md) for the finite profile, perturbation design and data provenance. Use the base Python dependencies and a verified original MNIST NPZ. Run `python -m observables.stage6` to regenerate the tables. The notebooks and saved results are under `notebooks/` and `results/stage6/`. The pair API is `observables.open_images.similarity_profile`.

## Stage 7 — MNIST redundancy and intervention audit

| Notebook | Experiment |
|---|---|
| [25 · Component redundancy](notebooks/25_MNIST_component_redundancy.ipynb) | Compare within-query rank correlations and shared miss sets. |
| [26 · Intervention response](notebooks/26_MNIST_intervention_response.ipynb) | Compare paired clean-to-perturbed distances and margins, with synthetic invariance checks. |
| [27 · Profile ablation](notebooks/27_MNIST_profile_ablation.ipynb) | Diagnose the existing uniform decision by omitting one coordinate at a time. |

See [THEORY_STAGE7.md](THEORY_STAGE7.md). Run `python -m observables.stage7` using the saved Stage 6 pair profiles. Audit tables, figures and input hashes are in `results/stage7/`. No new model is fitted.

## Stage 8 — matched baselines and fresh MNIST replication

| Notebook | Experiment |
|---|---|
| [28 · Matched MNIST baselines](notebooks/28_MNIST_matched_baselines.ipynb) | Replay the pilot and add ZNCC, standard HOG, a normalized structure tensor, and frozen ImageNet embeddings. |
| [29 · Fresh replication and hard distractors](notebooks/29_MNIST_replication_and_hard_distractors.ipynb) | Test 12 disjoint 40-image galleries and equal-size hard versus random wrong-digit subsets. |
| [30 · Frozen encoders and comparison table](notebooks/30_MNIST_frozen_encoders_and_table.ipynb) | Compare first- and second-order pooling within two encoders and regenerate the manuscript draft table. |

See [THEORY_STAGE8.md](THEORY_STAGE8.md) for baseline definitions, matched search policies and regeneration commands. The optional dependencies are in `requirements-benchmarks.txt`. The pilot rows use the exact existing Stage 6 galleries. Both panels, including individual pair scores, are under `results/stage8/`.

## Stage 9 — Fashion-MNIST transfer

| Notebook | Experiment |
|---|---|
| [31 · Transfer protocol](notebooks/31_Fashion_MNIST_transfer_protocol.ipynb) | Verify prior-source exclusions, illustrate a fresh gallery, and compare relative noise strength. |
| [32 · Transfer results](notebooks/32_Fashion_MNIST_transfer_results.ipynb) | Compare phase, structure-tensor and frozen-encoder effects under the matched MNIST/Fashion-MNIST protocol. |
| [33 · Categories and distractors](notebooks/33_Fashion_MNIST_categories_and_distractors.ipynb) | Diagnose category variation, matched-size distractors and disagreements between Q and its magnitude. |

See [THEORY_STAGE9.md](THEORY_STAGE9.md) for data provenance, exclusions and regeneration commands. Stage 9 uses the optional benchmark dependencies and the official Fashion-MNIST test archive. The results and comparison figure are in `results/stage9/`.

## Stage 10 — weighting controls and CIFAR-10

| Notebook | Experiment |
|---|---|
| [34 · Synthetic phase and weighting](notebooks/34_Synthetic_phase_and_gradient_weighting.ipynb) | Start from two gradient populations, measure axial error on rendered crossings, and separate phase, gradient weighting, and damping in retrieval. |
| [35 · CIFAR-10 phase transfer](notebooks/35_CIFAR10_phase_transfer.ipynb) | Compare the four frozen fields with aligned pixel, ZNCC, and frozen ImageNet baselines on matched grayscale and RGB galleries. |

See [THEORY_STAGE10.md](THEORY_STAGE10.md) for formulas, sampling, provenance, and run commands. Data tables, pair distances, and figures are in `results/stage10/`.

## Stage 11 — natural-image scorecards and encoder fusion

| Notebook | Experiment |
|---|---|
| [36 · CIFAR-10 angular scorecards](notebooks/36_CIFAR10_angular_scorecards.ipynb) | Lock phase and magnitude to the same offset, compare clean and transformed retrieval, and reconstruct patches behind category rescues and harms. |
| [37 · CIFAR-10 encoder fusion](notebooks/37_CIFAR10_encoder_fusion.ipynb) | Compare 56 × 56 and standard frozen ResNet-50 preprocessing on disjoint RGB galleries, then test one equal-rank fusion using top-1 and mAP. |

See [THEORY_STAGE11.md](THEORY_STAGE11.md) for the definitions and run commands. The scored pairs, galleries and figures are in `results/stage11/`.

## Stage 12 — hierarchical CIFAR-100 retrieval

| Notebook | Experiment |
|---|---|
| [38 · CIFAR-100 gallery and label protocol](notebooks/38_CIFAR100_hierarchy_protocol.ipynb) | Check balanced superclass coverage and the exact candidate masks for fine and coarse retrieval. |
| [39 · CIFAR-100 hierarchy results](notebooks/39_CIFAR100_phase_by_label_level.ipynb) | Compare frozen phase, magnitude, ZNCC and encoder scores at both levels and quantify paired phase effects. |

See [THEORY_STAGE12.md](THEORY_STAGE12.md) for data provenance and reproduction. Results, scored pairs and figures are in `results/stage12/`.
