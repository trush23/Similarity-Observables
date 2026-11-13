# Stage 3 — pair collisions, boundary information, and transfer

Stage 3 tests three sources of disagreement in a similarity profile. A scalar projection can discard information in a matrix. An entire correlation order can fail to distinguish two arrangements. A boundary can change which statistics are identifiable.

## Two-state orientation fields

Take the cyclic occupation sequences A={0,1,4,6} and B={0,1,3,7} in Z₁₂ from Notebook 04. Render site 0 as an 8 × 8 texture and site 1 as its transpose. Each image is a row of 12 known tiles. The local representation is the real second angular moment, computed separately on each tile using the periodic gradient within that tile. Transposition exchanges the two derivative axes and reverses the sign of this component. The clean field is therefore qᵢ=α(1−2xᵢ) for a common nonzero α.

The sites have the same one-point counts. For each cyclic lag, the count of 1–1 pairs also agrees. Those counts determine the complete 2 × 2 table of joint states, since each cyclic lag gives the same marginals. Any fixed function of the two motif states therefore has the same average at that lag. This includes a full matrix of channel products and its scalar trace. Choosing a richer encoding of each of the two states cannot resolve this particular two-point collision.

Let s=2x−1 for a normalized proof example. The third-order statistic T(r,t)=meanᵢ[sᵢsᵢ₊ᵣsᵢ₊ₜ] is invariant under cyclic shifts. At (r,t)=(1,4), A gives 2/3 and B gives 0. The full triplet array separates the two arrangements, and the bispectrum gives the corresponding Fourier information. The retrieval benchmark uses the measured q values, with no access to the true state labels when extracting features.

These are third-order correlations across **positions** of a field that is itself second order in orientation. That distinction is essential when interpreting the hierarchy.

## Matrix correlation and scalar projections

For a real vector field vᵢ, the directed lag matrix is M(r)=meanᵢ[vᵢvᵢ₊ᵣᵀ]. The trace measures the mean inner product. In a fixed two-channel frame, M₁₂−M₂₁ measures the mean oriented area. These projections each consume one real number per lag.

For vᵢ=(cos ωi, sin ωi) and its reflected counterpart (cos ωi, −sin ωi), with ω=2π/24, both traces are cos ωr. Their antisymmetric projections are ±sin ωr. The distinction is defined relative to a shared handed coordinate frame and a directed sequence. A task that identifies reversals could deliberately pool it away. This example uses 24 lag samples for both scalar projections and isolates channel choice from the lag budget. The full matrix stores four values per lag.

## Open windows

The crop census retains a length-L contiguous observation for each possible cut of each cyclic prototype. All 24 class-and-cut combinations have equal weight in the report. An open pair moment at lag r averages the L−r observed products. A triple at r,t averages L−max(r,t) products. A window is ambiguous when the opposite prototype has any cut with the same signature. Exact observed binary windows supply the information ceiling for this noiseless task. No unobserved pixels or wraparound products enter the open signature.

At length 12, cutting the cycle makes the open pair signatures distinguish all tested class-and-cut combinations. The excluded pairs depend on the cut, so this signature uses boundary information that cyclic pooling removes. At length six, eight of 24 exact windows are already ambiguous, and 15 of 24 are ambiguous under either correlation profile. This is a finite census for these prototypes. High open lags may rely on one product and can be unstable under noise.

## Matched nuisance controls

The reference gallery uses windowed waves. Query tiles use either the same family or one of two held-out families, a smooth ridge and a hard bar. Each family contains a motif and its transpose, matched in per-tile norm. The tile frame is supplied. The task allows cyclic whole-tile translations and independent tile sign reversals. It keeps the gradient axis in the common image frame. No rotation search or tile detection is involved.

Five methods compare a cyclic Q pair profile, a cyclic Q triplet profile, an aligned Q sequence, aligned raw pixels, and aligned pixels with independent sign freedom per tile. All alignment methods consider all 12 tile shifts. For a query tile x and gallery tile y, the sign-free pixel baseline uses min over signs of ||x ± y||² = ||x||²+||y||²−2|xᵀy|. This baseline accounts for the same sign nuisance that Q removes.

There are 48 seeded trajectories per arrangement. Each supplies one shift, one vector of tile signs, and one noise field, shared across the compared families and levels. The clean condition uses only the shift. The remaining conditions apply tile signs and noise standard deviation equal to 0, 0.35, 1, or 2 times signal RMS. Noise is Gaussian at the pixel level. The resulting 1,440 views each yield five decisions, for 7,200 query-method rows. Exact and numerical ties within 10⁻⁹ split credit. The two-class chance rate is 50%.

Intervals use 4,000 stratified resamples of the 96 perturbation trajectories with common indices across methods and conditions. They describe sampling over the perturbation generator for two fixed arrangements. The held-out motifs test a prescribed change of local shape. Generalization to new arrangement families, unknown tile frames, or real images remains an open experiment.

The stress result separates exact resolving power from estimation. At noise SD twice signal RMS, triplet retrieval ranges from 66.7% to 85.4%. The sign-aware patch alignment control ranges from 97.9% to 100%. At lower noise, all triplet runs in the supplied seed score 100%. The stronger baseline shows why an apparent Q advantage over signed pixel alignment needs to be interpreted in light of the nuisance symmetry.
