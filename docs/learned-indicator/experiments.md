# Experiment Design: a Fault Indicator on Masklets with a Video-Native Backbone

This note designs the experiments for a fault indicator whose unit is a surface region followed
through a posed video (a masklet), whose geometric evidence is the motion of that region and the
violation of predicted motion in the feature space of a frozen video encoder, and whose learned
part is a light probe on frozen spatiotemporal tokens. One monocular depth model is used per
frame; the surface is a set of 2D Gaussian surfels fitted to those depth maps. The note first
turns each part of the conception into a quantity that can be measured, corrects the parts that do
not hold as stated, and then gives a staged cheap test that can stop the idea early. Notation
follows the main document (`docs/learned-indicator.md`, Section 7); new symbols are in Section 0.

Status of every number: *(repo)* is quoted from the repository; *(checked)* was read on the
model's or dataset's own page or paper while writing; *(recalled)* was not; everything else is a
design choice or an estimate. Nothing has been measured, and every statement about what an
experiment will show is a prediction.

## 0. Setting, symbols, models

**Symbols added to the main document's notation.**

| Symbol | Meaning |
| --- | --- |
| $m$; $S_{m,t}$ | a masklet; its pixel support in frame $t$ |
| $\boldsymbol\theta \in \mathbb R^3$ | plane in inverse depth: $\xi(u) = \boldsymbol\theta^\top \mathbf h(u)$ with $\mathbf h(u) = [u_x, u_y, 1]^\top$ |
| $\boldsymbol\theta_{M,m}$ | plane fitted to the aligned monocular inverse depth on the masklet |
| $\mathbf r_m$, $\Lambda_m$ | masklet gap (witness plane minus monocular plane) and its $3 \times 3$ information |
| $\boldsymbol\nu$, $\sigma_b$ | unit normal of a mask boundary point; boundary position noise in pixels |
| $\alpha_m$, $\beta_{m,t}$, $\gamma_t(u)$ | components of the monocular error: constant on the masklet, per frame of the masklet, within the masklet |
| $\kappa$ | persistence: share of masklet-level error variance that is constant over frames |
| $E(\cdot)$ | frozen video encoder, clip to token grid |
| $D_m(\delta\boldsymbol\theta)$ | latent discrepancy of the depth-stabilised clip on masklet $m$ under a plane correction |
| $M_{\mathrm{blind}}$ | harm-weighted share of faulty surface on which no geometric witness speaks |

**Data and code that exist** *(repo)*. MatrixCity `small_city` aerial block 3: 185 training
frames in strips, $1920 \times 1080$, exact depth, median depth 338 m; water is 46% of the ground
area, flat at $-3.4$ m. A depth cache (four views cached; Depth Pro 3-5% and Depth Anything V2
1-2% median relative error after alignment; Depth Pro 0.54 s and 6.3 GB per view).
`depthcache.align.align_depth` for sparse references; loaders with strip keys
(`experiments/matrixcity_io.py`); an OpenMVS wrapper; the consensus gate. No segmenter, tracker
or video encoder is in the repository. One 8 GB GPU; a 24 GB card has been used for depth models.

**Frame spacing is not known for the target data.** Block 3 and the planned real set (UseGeo,
80% forward overlap *(repo)*) are survey stills: the image shifts by about a fifth of its length
between frames and a ground point is seen in about five frames of a strip. Every experiment is
specified for both survey spacing and dense video; dense video with exact depth has to be
obtained (TartanAir has exact depth, pose and flow, an ocean environment and reflective scenes
*(checked)*, but ground-robot and free-flight motion, not survey motion).

**Foundation models, as verified.**

| Model | What it is and gives | Verified | Not verified, or does not do |
| --- | --- | --- | --- |
| V-JEPA 2 | self-supervised video encoder; ViT-L, H, g (0.3, 0.6, 1 B parameters) at 256 or 384 px; mostly MIT licence | *(checked)* | not described as suited to dense tasks; hub entries return encoders only |
| V-JEPA 2.1 | same family trained for dense features; ViT-B, L, g, G (80 M to 2 B) at 384 px; tokens are $16 \times 16$ px by 2 frames; a separate image path with $16 \times 16$ patches; dense loss on visible and masked tokens with deep supervision; frozen-feature results on depth, segmentation and video object segmentation; trained on a mix of curated images and SSv2, Kinetics, HowTo100M, YT-1B | *(checked, paper and repository)* | no aerial video named in the training mix; behaviour under large inter-frame motion not discussed in what was read; memory not stated |
| V-JEPA 2-AC | predictor conditioned on robot actions, post-trained on robot-arm data | *(checked)* | **no predictor conditioned on camera pose exists**; whether the unconditioned predictor weights are released was not confirmed |
| SAM 2 / 2.1 | promptable video segmentation with streaming memory; clicks and boxes as video prompts; automatic mask generation **on images**; four sizes, 39 to 224 M; Apache 2.0 | *(checked)* | unprompted segmentation of a whole video is not offered; input resolution and memory for long clips not stated on the page |
| SAM 3 / 3.1 | detects, segments and tracks all instances of a concept given a short noun phrase or image exemplars, with identities through a video; detector and memory tracker on one backbone; 848 M parameters; 3.1 adds joint multi-object tracking with shared memory; "SAM License"; checkpoints gated on request | *(checked)* | whether amorphous regions (water) and nadir aerial views work; licence terms were not read; memory and speed not stated; the documented video workflow is prompt-driven |

Consequences for the conception, before any experiment:

1. *There is no model that predicts future tokens from a camera pose.* "The depth and the poses
   predict how every token should move" cannot be delegated to a released predictor. The
   prediction has to be made in pixels, by warping with depth and pose, and only the comparison
   happens in feature space. Section 1.3 gives a form of this that also removes the large-motion
   problem.
2. *Masklets do not come for free.* Both segmenters need prompts. A masklet inventory of a whole
   clip needs seeds (automatic masks on key frames, or a fixed concept vocabulary) and an
   association step, and neither has been shown at survey spacing. Section 1.2 gives an
   association that uses the poses.
3. *The licences differ.* SAM 2.1 and V-JEPA are permissive; SAM 3 is gated under its own
   licence. The default design uses SAM 2.1 and V-JEPA 2.1; SAM 3 is an option for strata.

## 1. The conception as measurable quantities

### 1.1 A fault as a property of a surface: the error anatomy

After the per-frame affine alignment, the error of the monocular inverse depth is decomposed over
masklets and frames,

```math
\epsilon_M(u, t) = \alpha_m + \beta_{m,t} + \gamma_t(u), \qquad u \in S_{m,t}, \quad \sum_t \beta_{m,t} = 0, \quad \sum_{u \in S_{m,t}} \gamma_t(u) = 0 ,
```

with $\alpha_m$ the part every frame repeats, $\beta_{m,t}$ the per-frame deviation and $\gamma$
the relief inside the masklet. The claim that a repeated fault belongs to the surface is the claim
that fault energy sits in $\alpha$. Three numbers test it, all harm-weighted (each pixel weighted
by $1 - \varsigma$ of the main document):

- the **between-masklet share** $\sum_m n_m (\alpha_m + \beta_{m,t})^2 / \sum \epsilon_M^2$ restricted
  to faulty pixels: how much of the fault a verdict per masklet and frame can express at all;
- the **persistence**
  $\kappa = \operatorname{Var}(\alpha_m) / \bigl(\operatorname{Var}(\alpha_m) + \overline{\operatorname{Var}_t(\beta_{m,t})}\bigr)$,
  an intraclass correlation over frames, with the sampling variance of masklet means removed. It
  is the repetition coefficient $\varrho$ of the main document expressed per entity: $\kappa = 1$
  means every frame repeats the masklet's error, and consensus over those frames is blind to it;
- the **pooling efficiency** of masklets as test regions. A regional mean shift $\delta$ over
  $n_{\mathrm{eff}}$ independent pixels is detected at $z = \delta \sqrt{n_{\mathrm{eff}}} / \sigma$.
  A masklet that a fault covers only in part dilutes the shift. Efficiency is the squared ratio of
  the $z$ reached with masklets to the $z$ reached with the true fault region.

Each is compared, at an equal number of regions, with regular tiles (the consensus gate's region
scan uses $4 \times 6$ tiles *(repo)*), SLIC superpixels and clusters of frozen image features.
The masklet is the right unit only if it beats these; that it "follows objects and materials" is
a hypothesis about where monocular depth fails, and the alternative, that faults follow
illumination, range or image position, is equally plausible before measurement.

The unit of evaluation changes with the unit of the method, and that can flatter the method.
Every method is therefore scored twice: on masklets (area- and harm-weighted) and on the
pixel-level fault mask of the main document, with masklet verdicts painted onto pixels.

### 1.2 Masklet motion as parallax

The inverse depth of a plane is affine in pixel coordinates, $\xi(u) = \boldsymbol\theta^\top \mathbf h(u)$,
so a planar masklet has three unknowns given the poses, and the comparison with the monocular
depth is linear. With $\mathbf b_i(u) = \partial \mathbf p_i / \partial \xi$ the parallax gain of
the main document, a boundary point predicted at $\mathbf p_i(u; \boldsymbol\theta_M)$ and
observed at $\mathbf q_i$ gives, along the boundary normal $\boldsymbol\nu$,

```math
c_i(u) = \boldsymbol\nu^\top \bigl( \mathbf q_i - \mathbf p_i(u; \boldsymbol\theta_M) \bigr) = \bigl(\boldsymbol\nu^\top \mathbf b_i(u)\bigr)\, \mathbf h(u)^\top \mathbf r_m + \text{noise}, \qquad
\Lambda_m = \sum_{i,\,u \in \partial S} \frac{\bigl(\boldsymbol\nu^\top \mathbf b_i(u)\bigr)^2}{\sigma_b^2}\, \mathbf h(u) \mathbf h(u)^\top ,
```

and $\mathbf r_m = \Lambda_m^{-1} \sum (\boldsymbol\nu^\top \mathbf b_i)\, \mathbf h\, c_i / \sigma_b^2$
is the masklet gap. This is the parallax gap of the main document with boundary points in place
of interior tracks. Four things in the statement "its outline fixes its plane" need care, and
each is an experiment.

- **Aperture.** A boundary point informs only through $\boldsymbol\nu^\top \mathbf b$. In a
  straight strip $\mathbf b$ points along the flight line, so boundary segments parallel to the
  flight line carry nothing, and an elongated region aligned with the strip (a road, a canal) is
  poorly conditioned however long its outline. $\Lambda_m$ reports this; a cross strip repairs it.
- **Occluding contours.** The outline of a roof lies on the roof; a shoreline lies on the water.
  The outline of ground seen between buildings is drawn by the roofs in front of it and moves
  with *their* parallax. A boundary segment may enter $\Lambda_m$ only if the masklet owns it,
  that is, if it is the nearer side or the two sides meet. Ownership is decided without labels
  from the depth order across the boundary in the monocular map (ordinal depth at edges is what
  such models do best) and checked against the neighbour's own gap; segments on the image border
  are never used. Truth for this test comes from exact depth.
- **Boundary noise.** Mask boundaries jitter by a few pixels and the jitter is correlated along
  the boundary. With boundary length $\ell$, correlation length $c$ and parallax $p$, the relative
  precision of the level is about $\sigma_b / (p \sqrt{\ell / 2c})$. For $\sigma_b = 2$ px,
  $\ell / c = 20$ and $p = 216$ px (adjacent survey frames) this is 0.3%: an order of magnitude
  below the monocular error. These inputs are assumed; $\sigma_b$ and $c$ are measured in A4.
- **Planarity.** The model is a plane; a non-planar masklet (terrain, canopy, a hipped roof) is
  wrong for it. The residual of the monocular surface against its own plane fit, and the residual
  of the boundary fit, flag this without labels; such masklets abstain from the plane witness.

**Association at survey spacing.** A memory tracker trained on video may not follow a region that
jumps by a fifth of the image. The poses give a remedy: warp the mask of frame $j$ into frame $i$
with the monocular plane $\boldsymbol\theta_M$ and use the warped mask (its box, or points
sampled inside it) as the prompt in frame $i$. The segmenter snaps to the image evidence. The
displacement between the prompt and the snapped mask *is* $c_i$: association and gap come from
the same operation. The monocular depth is used only as the point of linearisation, as in the
main document; a prior so wrong that the prompt misses the region is itself a detection.

**What this does to the argument for the prior.** If the estimate above is near the truth, a
planar masklet with an owned, well-oriented outline has its depth from its own motion at
sub-percent precision, texture or none. Water and flat roofs then leave the blind regime, and
they also leave the set on which the monocular prior matters, because the plane can replace it.
The prior, and with it any learned indicator, is left with: masklets that are untextured *and*
non-planar or without an owned outline; relief inside masklets where interior tracking fails; and
short windows of dense video where no witness is yet precise. The harm-weighted share of faulty
surface in the first two is $M_{\mathrm{blind}}$. It is the single number that says whether a
learned head has a job, and it can be measured before any head exists (A3).

### 1.3 Violation of expectation: the depth-stabilised clip

Since no released predictor takes a camera pose, the expectation is formed in pixels. Two
choices decide whether the comparison in feature space means anything.

**Stabilise instead of predict.** Warp frame $i$ back into frame $j$ with the monocular depth of
frame $j$ and the poses, and form the two-frame clip (frame $j$, warped frame $i$). If the depth
is right the clip is still; if it is wrong, what remains is motion along the epipolar direction
of size $b_i\, \epsilon_M$ pixels. A token of V-JEPA 2.1 spans two frames and
$16 \times 16$ pixels *(checked)*, so residual motion is exactly what a token encodes. This also
answers the worry about encoders trained on small motion: at survey spacing the raw shift is
hundreds of pixels, but after stabilisation by a depth that is right to a few percent the
residual is a few pixels, which is the regime the encoder was trained in. The reference is the
still clip (frame $j$, frame $j$).

**Sweep instead of score.** A distance between token grids has no unit and no null: it responds
to resampling blur, disocclusion, exposure and view-dependent appearance as well as to depth. It
becomes a measurement when the depth is perturbed and the discrepancy minimised. On a masklet,

```math
D_m(\delta\boldsymbol\theta) = \sum_{\text{tokens } k \subset S_{m,j}} \bigl\lVert E_k\bigl(I_j,\ \mathcal W_{i \to j}[I_i;\ \boldsymbol\theta_M + \delta\boldsymbol\theta]\bigr) - E_k(I_j, I_j) \bigr\rVert^2, \qquad
\mathbf r^{\mathrm{lat}}_m = \arg\min_{\delta\boldsymbol\theta} D_m, \qquad
A^{\mathrm{lat}}_m = \min_{\delta\boldsymbol\theta} D_m ,
```

with the sweep in practice over the level (one parameter, nine values) and the full plane only
for large masklets. The minimiser is a gap in the units of the explicit parallax gap; the
curvature at the minimum, scaled by the null level of $D$, is its information; and the minimum
value is an ancillary statistic in the sense of the main document: a masklet whose stabilised
clip is not still under *any* depth has left the static rigid model (ripples, movers,
view-dependent appearance) and abstains. This is a plane sweep in the feature space of a video
encoder. It is offered as the corrected form of "violation of expectation", and the uncorrected
form (raw discrepancy at $\delta\boldsymbol\theta = 0$) is kept as an ablation.

**What it should and should not be expected to add.** It cannot see motion where there is no
image structure, so it does not extend coverage onto untextured interiors; there the outline of
Section 1.2 is the witness. Its position noise is tied to a 16-pixel token, so per token it is
far coarser than a point tracker; pooled over a masklet it may still be adequate. With parallax
of 216 px, a token-scale sensitivity of a few pixels is a relative depth error of a few percent,
the same order as the monocular error: at survey spacing the latent witness would sit near
information ratio $\vartheta \approx 1$, the regime in which a witness can judge a prior and not
replace it. Whether that is a virtue or merely a worse tracker is decided by B2.

**Reflections.** A planar reflection is a rigid scene behind the surface. Stabilised at the
correct surface depth, the reflected content still moves; the sweep finds a clean minimum at the
virtual depth and the ancillary passes. The latent witness inherits the reflection problem
unchanged. The masklet gives the handle the pixel did not: outline and interior of one entity are
two witnesses of one plane, and a significant difference between the boundary gap and the
interior gap, with the interior behind, marks a reflective or transparent surface without labels.
That test needs both to be informative, so it applies to textured-looking water and glass only.

### 1.4 The probe, and what it can copy

The head is a linear or two-layer probe on frozen tokens pooled per masklet (mean and spread over
the tokens of $S_{m,t}$, concatenated over frames), trained on masklets where a geometric witness
gives a label, applied where none does. Which tokens it sees decides what it can learn.

| Input to the probe | Can it read the witness? | Expected use |
| --- | --- | --- |
| image-path tokens of single frames | no | appearance and material; the clean transfer baseline |
| tokens of the raw clip | it sees motion, not the monocular depth | adds parallax cues the encoder has formed |
| tokens of the depth-stabilised clip | yes: residual motion is the label's cause | can reproduce the witness, including its errors |

A probe on stabilised-clip tokens that agrees with the witness has learned nothing that
transfers, and on reflective masklets it will repeat the witness's false accusation. The copy
test (B3) scores each input on masklets where the witness is known from exact depth to be wrong.
Before measurement the prediction is that motion-aware tokens help on witnessed masklets, where
help is not needed, and add nothing on blind ones, where there is no motion signal to encode.

**Strata from concept prompts.** A fixed vocabulary (water, glass roof, glass facade, flat roof,
road, bare ground, vegetation, vehicle, shadow) prompted through SAM 3 would give material strata
with no fault labels. It is zero-label and not zero-knowledge: the vocabulary encodes where
faults are expected. Whether the model returns amorphous regions and works on nadir views is
unverified; B1 scores its water masks against the exact water mask.

## 2. The cheap test, in two stages

Exact depth is used for scoring only, and in stage A also to *stand in for parts of the pipeline
that do not exist yet*, so that each premise is tested separately from the engineering needed to
exploit it. Where exact depth plays that role it is named.

### Stage A: premises, with a depth model and image masks only

**Inputs.** One Depth Pro pass and one Depth Anything V2 pass per frame (about 3 minutes of GPU
together). SAM 2.1 automatic masks on every frame (image mode, the documented path). Masks are
associated across frames by *exact* reprojection, so stage A is independent of video tracking.
Geometry from `transforms.json`: frames per strip, overlap, parallax per ground point.

- **A1, error anatomy.** Between-masklet share, persistence $\kappa$ with a bootstrap interval,
  and the same for tiles, superpixels and feature clusters at equal region count
  (Section 1.1). Reported per surface class (water, ground, roof, facade, vegetation where
  derivable from exact data).
- **A2, pooling efficiency** of each partition against the true fault regions, and the resulting
  detection rate of an ideal regional test at 5% false alarms.
- **A3, blind mass.** Each masklet is classified with exact data as: interior trackable (local
  texture above a threshold, to be replaced by a real tracker later); planar (exact depth within
  the monocular noise of a plane); outline owned and conditioned ($\Lambda_m$ from exact boundary
  ownership above a threshold). $M_{\mathrm{blind}}$ is the harm-weighted share of faulty surface
  in masklets that are neither trackable nor planar-with-owned-outline, plus fault energy in
  $\gamma$ inside untracked masklets. Reported next to the same share when only interior tracking
  is allowed, so that the reduction claimed for outline parallax is a measured ratio.
- **A4, outline parallax with real masks.** With association still exact, the plane of each
  planar masklet is estimated from the automatic masks of its frames by the estimator of
  Section 1.2; the error against the exact plane is compared with $\Lambda_m^{-1}$; $\sigma_b$ and
  its correlation length are estimated from the fit residuals; ownership decided from the
  monocular depth order is scored against true ownership.

**Stage A stops the line if any of:**

- **KA1, the unit is not justified.** Masklets explain no more fault energy than superpixels or
  tiles at equal count (difference in between-region share below 0.05), or pooling efficiency is
  below 0.5.
- **KA2, faults are not persistent on entities.** $\kappa$ has upper bound below 0.3 on fault
  masklets: errors change frame to frame, consensus already sees them, and a verdict per masklet
  with frame deviations as a second-order term is the wrong model.
- **KA3, nothing is left, or nothing is gained.** $M_{\mathrm{blind}} < 0.05$: geometry covers
  the faults and no learned head is needed; the work becomes a geometric method and is re-scoped
  as such. Or outline parallax reduces the blind share by less than a quarter relative to
  interior tracking alone: the second element of the conception does not deliver.
- **KA4, outlines do not give planes.** On planar, untextured masklets the standardised plane
  error has robust standard deviation above 2 after calibrating $\sigma_b$, or the relative depth
  error of the outline plane exceeds the monocular error scale on more than half of them.

Stage A needs about 370 depth passes, 185 mask-generation passes and CPU work; an estimate of 600
lines of new code.

### Stage B: the video-native parts

- **B1, masklets from a frozen segmenter.** (i) SAM 2.1 video propagation from automatic masks on
  key frames; (ii) the pose-guided prompt-and-snap association of Section 1.2; (iii) SAM 3
  concept prompts with the fixed vocabulary, if access is granted. Scored against the exact
  association of stage A: purity and fragmentation of masklets, lifetime in frames, coverage of
  the image, at survey spacing and, on dense synthetic video, at strides 1 to 16. Water masks
  against the exact water mask.
- **B2, psychometrics of the latent witness.** Depth-stabilised clips are built with *exact*
  depth plus an injected plane error, in a ladder of residual parallax from 0.25 to 32 px. For
  each surface class: the discrepancy at zero injection (the null, which measures what warping
  artefacts, disocclusion and view-dependent appearance cost); the smallest residual parallax
  detected at AUC 0.9 against the null, per token and pooled per masklet; the bias and spread of
  $\mathbf r^{\mathrm{lat}}_m$ against the injected value and against its curvature-based error
  bar; the response of $A^{\mathrm{lat}}_m$ to injected non-rigid motion. Then the same with the
  monocular depth in place of exact depth, scored against the true masklet error. Encoder:
  V-JEPA 2.1 ViT-L, frames tiled at 384 px; the explicit parallax gap from a point tracker or
  matcher is run on the same masklets as the calibrated reference. Images degraded and poses
  perturbed as in Section 4.
- **B3, the copy test.** Probes on the three inputs of Section 1.4, trained on witness labels,
  scored against truth on (a) witnessed masklets, (b) masklets where the witness is wrong
  (virtual images, movers; from exact depth), (c) blind masklets. Comparators without labels: the
  fault rate per feature cluster; texture energy; the consensus gap. With labels: a linear probe
  on the same tokens trained on true masks of disjoint ground tiles.
- **B4, is the prior worth keeping on blind masklets.** Relative depth error there of the
  monocular depth restricted to truly fault-free masklets (what a perfect indicator keeps),
  against interpolation from neighbouring witnessed masklets, OpenMVS, and a feed-forward
  multi-view model where one fits in memory.

**Stage B stops the line if any of:**

- **KB1, no masklets at the data's spacing.** With every association method, mean purity below
  0.8 or median lifetime below three frames.
- **KB2, the latent witness is not a measurement.** At zero injection the discrepancy exceeds
  the response to a residual parallax equal to the monocular error scale (artefacts dominate);
  or $\mathbf r^{\mathrm{lat}}_m$ is biased by more than its error bar; or its masklet-level
  detection threshold is more than three times that of the explicit gap while its coverage is no
  larger. In the last case it is a costlier and coarser tracker, and the explicit gap is used.
- **KB3, the probe copies.** On masklets where the witness is wrong, every probe that beats the
  per-cluster rate agrees with the witness more often than with the truth.
- **KB4, no transfer.** On blind masklets the best probe has AUC with lower bound at or below
  0.65, or does not exceed the per-cluster fault rate by 0.05 average precision.
- **KB5, the prior is not worth keeping.** On blind masklets, in every surface class with at
  least five fault regions, an alternative without monocular depth is at least as accurate as the
  monocular depth under a perfect indicator.

**It is supported if all of:** masklets beat the other partitions by 0.10 in between-region
share with pooling efficiency of at least 0.7; $\kappa \ge 0.6$ on fault masklets;
$0.15 \le M_{\mathrm{blind}}$ and outline parallax halves the blind share; outline planes are
calibrated; the latent gap is unbiased with a masklet-level threshold within a factor of two of
the explicit gap or with larger coverage; and on blind masklets a probe that does not see the
stabilised clip reaches AUC 0.75 and a true-positive rate of 0.30 at 5% false positives on
masklets with $\kappa \ge 2/3$. All numbers are conventions fixed before measurement.

**Uncertainty.** Cluster bootstrap over $100 \times 100$ m ground tiles; the number of distinct
fault masklets is shown beside every interval. Block 3 is about half water, which is likely one
or a few masklets cut by image borders: statistics "per masklet" on water rest on very few
entities, and are reported as such.

**Cost of stage B** (estimates; nothing measured). V-JEPA 2.1 ViT-L has 0.3 B parameters, about
1.2 GB of weights in single precision; a two-frame clip at 384 px is 576 tokens, and a frame
needs about 15 tiles, so a nine-value sweep over the four neighbours of 185 frames is about
100 000 small forward passes, under an hour by a rough estimate. ViT-g (1 B) should fit 8 GB in
half precision; ViT-G (2 B) probably not with activations. SAM 2.1 large is 224 M *(checked)*;
SAM 3 at 848 M is about 3.4 GB of weights before the memory of tracked objects, unverified on
8 GB. About 1000 further lines of code.

## 3. Follow-on experiments, in order

| | Question | Data | Metric | Baseline | Falsifiable hypothesis |
| --- | --- | --- | --- | --- | --- |
| F1 | How do masklets, outline planes and the latent gap depend on frame spacing? | dense synthetic video at strides 1 to 16; block 3 | B1 and B2 per stride | interior tracking; plane-sweep patch correlation | prompt-and-snap keeps purity above 0.8 at every stride where memory propagation falls below it; the latent threshold in pixels of residual parallax is constant over strides once clips are stabilised |
| F2 | Is there a judge regime: a witness too coarse to replace the prior that still detects its regional faults? | dense video, causal windows of growing length | per-pixel error of witness depth against monocular; masklet-level AUC; delay in frames | pixel-level witness; consensus over the same window | a range of windows exists where the masklet gap reaches AUC 0.9 while per-pixel witness error exceeds the monocular error; falsified if the two crossovers lie within a factor of two |
| F3 | What transfers to blind masklets, and can transfer be measured without ground truth? | blocks and sequences with water and glass | probe AUC on blind masklets per feature set and per propensity decile; error of the counterfactual-abstention estimate | per-cluster rate; supervised probe from a disjoint scene | image-path tokens transfer as well as clip tokens on blind masklets (within 0.02); withholding whole masklets of the lowest propensity predicts the true blind-masklet AUC within 0.05 |
| F4 | What do true labels buy? | 0, 1, 5, 20 labelled frames, one labelled scene, from scenes never evaluated | average precision and calibration on blind masklets | linear probe on the same tokens at equal budget | labels halve the calibration error at 5 frames and change ranking by less than 0.02; with synthetic labels only, the supervised probe loses 0.05 on real data and the witness-trained probe does not |
| F5 | Faults every frame repeats (headline) | masklets with $\kappa \ge 2/3$, natural, and constructed by making one frame's fault consistent over all frames | true-positive rate at 5% false positives, on witnessed and on blind masklets separately | consensus gate, robust loss, frame removal (predicted at chance); depth-model confidence; single-image baselines (disagreement between depth models, instability under input transformations) | consensus within 0.05 of chance on both; outline or latent gap above AUC 0.9 on witnessed; the probe at 0.30 true-positive rate on blind |
| F6 | Does a masklet verdict improve the surfel surface? | blocks and sequences; held-out frames by block | held-out depth error and surface F-score on witnessed and blind masklets; precision and recall separately | no gate; pixel-level gate; witness planes fused as depth messages with the prior ungated; the prior down-weighted uniformly on blind masklets; no prior at all; oracle | on fault masklets in the blind set the gate closes half the gap between "planes fused, prior ungated" and the oracle; falsified if that baseline is already at the oracle |
| F7 | Is the judge specific to monocular depth? | depth from a feed-forward multi-view model in place of the monocular map | stage A and B | the monocular case | persistence on blind masklets is as high for the multi-view prior, and the probe transfers within 0.05 AUC |
| F8 | Real aerial data | UseGeo stills with laser depth; a real aerial video with poses and laser reference has not been identified | stages A and B on laser-covered surface; counterfactual abstention elsewhere | as F6 | outline planes stay calibrated once pose covariance enters $\Lambda_m$; $\kappa$ estimated without reference (from consensus and witness gaps, as the main document's $\hat\varrho$) agrees with the laser-based value within its interval |

Frozen feature sets compared in F3: V-JEPA 2.1 image-path tokens, raw-clip tokens and
stabilised-clip tokens (ViT-L and ViT-g); the image encoder of the segmenter, which is run
anyway; the encoder of the depth model, likewise; a self-supervised image encoder as the
reference.

## 4. What a rendered survey block can and cannot show

- **Water** dominates block 3 and is cut by image borders, so its outline is mostly not owned
  and A3 may place it in the blind set for a reason that a real lake seen whole would not share.
  Whether the renderer draws reflections on it is unknown; if not, the virtual-image case cannot
  appear here and is tested on rendered reflective scenes elsewhere.
- **Glass** at 340 m range in nadir views is too small to form masklets of useful size.
- **Masks on rendered frames** may be cleaner than on real imagery; $\sigma_b$ is re-estimated on
  real frames (label-free, from fit residuals) before any real-data claim.
- **Flattery.** Exact poses and noise-free frames favour every witness. Decisions are read with
  frames degraded (gain and gamma jitter of 20%, noise of 3/255, blur of 0.7 px, JPEG quality 75)
  and poses perturbed at bundle-adjustment level (`experiments/h7_pose_check.py`).
- **Domain.** The video encoder's training mix names no aerial footage *(checked)*; its dense
  features on nadir frames are untested. B2's null level is where this shows first.

## 5. Protocol

1. No exact depth in any witness, alignment, association, label or probe of the method. Where
   stage A uses exact depth as a stand-in for a missing component, the table says so, and the
   stand-in is replaced in stage B.
2. Evaluation masks, classes, true planes and true association live in `results/k0/eval_only/`;
   a unit test fails if a fitting script opens it.
3. Thresholds of the witnesses (conditioning of $\Lambda_m$, ownership, planarity, ancillary
   quantiles) come from nominal values and fit residuals, never from a detection metric. The
   concept vocabulary is fixed before any mask is read.
4. The probe is fitted on the scene it is applied to, from witness labels (transductive,
   declared). Labelled data, when used, come from scenes never evaluated; real labelled frames
   are separated on the ground from evaluation frames by one image footprint.
5. Splits by block or scene. Block 3 serves the cheap test and is a development block afterwards.
6. Backbones, segmenter and depth model are used with released weights and are not fine-tuned.
7. Every method is scored on masklets and on the common pixel-level mask.
8. Thresholds of Section 2 and hypotheses of Section 3 are committed and tagged as
   `prereg/TAGGING.md` prescribes before the scoring script is first run.

## 6. Implementation plan for the cheap test

Scripts under `experiments/` with prefix `k0_`; outputs under `results/k0/`; nothing in `src/`
changes.

| # | Stage | Script | Inputs | Outputs | Reuses |
| --- | --- | --- | --- | --- | --- |
| 1 | A | `k0_geometry.py` | `transforms.json` | strips, overlaps, neighbours, parallax gains | `matrixcity_io.load_block` |
| 2 | A | `k0_cache.py` | frames | depth cache, float32 | `depthcache.run.run` |
| 3 | A | `k0_masks.py` | frames | automatic masks per frame, image-encoder features | SAM 2.1 image mode |
| 4 | A | `k0_truth.py` | exact depth, depth cache, masks | `eval_only/`: error field, fault mask, classes, exact association of masks into masklets, exact planes, boundary ownership | `matrixcity_io.project`, `align_depth`, `WATER_Z` of `surface_dsm.py`, `baselines/_common.depth_to_normals` |
| 5 | A | `k0_anatomy.py` | step 4 | A1, A2, A3 tables for masklets, tiles, superpixels, clusters | scoring side only |
| 6 | A | `k0_outline.py` | masks, poses, depth cache; association from step 4 | $\mathbf r_m$, $\Lambda_m$, $\hat\sigma_b$, ownership from monocular depth order | `align_depth` for the per-frame alignment against witnessed planes |
| 7 | B | `k0_masklets.py` | frames, poses, depth cache | masklets by memory propagation, by prompt-and-snap, by concept prompts | SAM 2.1 video predictor; SAM 3 if available |
| 8 | B | `k0_stabilise.py` | frames, poses, a depth source, plane perturbations | depth-stabilised two-frame clips, tiled; visibility masks | `matrixcity_io.pixel_rays_world`, `project` |
| 9 | B | `k0_latent.py` | clips | token grids, $D_m$ over the sweep, $\mathbf r^{\mathrm{lat}}_m$, $A^{\mathrm{lat}}_m$ | V-JEPA 2.1 encoder |
| 10 | B | `k0_probe.py` | pooled tokens, witness labels | masklet verdicts for each probe input; counterfactual-abstention scores | none; no path to `eval_only/` |
| 11 | both | `k0_score.py` | all | tables A1 to A4, B1 to B4, one line per criterion with value, interval, verdict | `eval/metrics.detection_pr`; a tile-resampling wrapper around `eval/stats.bootstrap_ci`, which resamples one value per scene |
| 12 | both | `tests/test_k0_isolation.py` | - | fails if scripts 6 to 10 open `eval_only/` (script 6 receives the association as a file in stage A, and that exception is listed) | - |

Order: 1 to 5 first. They need no video model and answer KA1 to KA3.

## 7. Contribution and nearest prior work, as seen from the experiments

**What could carry a paper.**

1. *The error anatomy of monocular depth in posed video* (A1 to A3): how much of the error is a
   property of a surface entity, how persistent it is across frames, and how much faulty surface
   no geometric witness reaches. This is a measurement that stands whether or not the method
   works, and the persistence $\kappa$ is the quantity that makes "a fault every frame repeats"
   a number.
2. *The depth-stabilised clip with a sweep* (Section 1.3, B2): a way to make a frozen video
   encoder act as a geometric witness with a unit, an error bar and an abstention statistic,
   without a pose-conditioned predictor, and with residual motion kept in the encoder's training
   regime at any frame spacing. The psychometric protocol (how many pixels of unexplained
   parallax a frozen video encoder can see, per surface type) is reusable beyond this method.
3. *Prompt-and-snap* (Section 1.2): association of regions across widely spaced frames and the
   regional parallax gap from one operation of a frozen segmenter.

**Nearest prior work** *(all recalled; no literature search was made for this note)*. Piecewise
planar and superpixel-based multi-view stereo, which fit planes to regions with photometric
costs; plane-sweep stereo on deep features and feature-metric alignment in bundle adjustment,
which are the image-encoder relatives of the latent sweep; lifting of segment-anything masks to
3D with known poses, which associates masks geometrically; violation-of-expectation studies that
read surprise from a JEPA predictor, which use the model's own predictor where this design uses
geometry; residual-supervised distractor masks on frozen features in radiance-field training,
which learn from the residual of the model being fitted. The combination of a region-level
geometric judge of a *learned depth prior*, with persistence as the target and a label-free
account of where the judge is blind, was not found in memory; that is weak evidence of novelty
and needs a search.

**Honest reading.** If $M_{\mathrm{blind}}$ is small, the result is a geometric method with a
segmenter and there is no learned indicator to speak of. If $M_{\mathrm{blind}}$ is large and
transfer fails, there is a measurement paper. The learned indicator is a contribution only in
the band between.

## 8. What this design cannot settle

- **Whether masklets exist at survey spacing** with frozen segmenters; stage A sidesteps this
  with exact association and stage B may fail on it.
- **Few entities.** Persistence and transfer are statistics over masklets; a block that is half
  water has few independent ones on the surface that matters most.
- **Reflections** remain a source of confident false accusation for every motion witness; the
  outline-against-interior test covers only surfaces that show trackable content.
- **Calibration on blind masklets** is not identified without labels; ranking may transfer, the
  fault rate does not.
- **Dense aerial video with exact depth** is not on disk; the judge regime and all stride
  ladders depend on obtaining it.
- **Unverified about the models:** memory and speed of every model on 8 GB; SAM 3 on water and
  nadir views, its licence terms and access; release of unconditioned V-JEPA predictor weights;
  V-JEPA 2.1 dense features on aerial frames and its sensitivity to sub-token motion.
- **Thresholds are conventions** fixed in advance, not derived from a cost model.
