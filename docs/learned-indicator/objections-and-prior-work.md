# Judging a Monocular Depth Prior per Surface Region through Time: Objections, Prior Work, Verdict

Nothing in this note has been measured. Numerical examples are worked from stated assumptions and
are labelled as such. Statements about existing models are marked *verified* (checked against the
paper, its arXiv page or the official repository) or *recalled* (from memory, not checked).

## 0. The conception under examination

**Setting.** A static scene is filmed from the air in strips, with known poses. The frame spacing is
not fixed in advance: it may be dense video, or survey stills at about 80% forward overlap, where the
image shifts by a fifth of its length per frame. One monocular depth map $\hat d_i$ per frame from one
frozen model. A surface of 2D Gaussian surfels is fitted. A fault is a connected region whose depth
deviates from the measurement model by many $\sigma_i$; the faults of interest are repeated by every
view of the region.

**Four proposals.**

1. *Unit.* The verdict belongs to a surface region followed through the clip (a masklet from a video
   segmenter), not to a pixel in a frame. Per-frame deviations are a second-order term.
2. *Witness without texture.* With known poses, the motion of a region's outline and interior between
   frames fixes its plane, since a plane-induced homography has three unknowns.
3. *Evidence in feature space.* The monocular depth and the poses predict how the clip should move.
   The clip warped by that prediction is compared with the real clip inside a frozen video encoder.
   Disagreement is the evidence. The explicit parallax gap with its error bar remains as the
   calibrated special case.
4. *Head.* A dense probe on spatiotemporal tokens; concept prompts of a promptable segmenter give
   material strata without labels.

**Notation.** Masklet $m$ has mask $M_{m,i}$ in frame $i$. A plane in the camera frame of a reference
view $r$ is written by its inverse-depth vector $\boldsymbol\theta = \mathbf n / d_\pi \in \mathbb R^3$,
so that the inverse depth at pixel $u$ is $\rho(u) = \boldsymbol\theta^\top K^{-1} \tilde u$. For a
second view with $X' = R_t X + \mathbf t_t$ the induced homography is

```math
H_t(\boldsymbol\theta) = K \bigl( R_t + \mathbf t_t\, \boldsymbol\theta^\top \bigr) K^{-1} .
```

In a straight strip ($R_t = I$, $\mathbf t_t = (b_t, 0, 0)$) this reduces to a shift along the flight
direction, $x' = x + f b_t\, \rho(u)$, with $\rho$ affine in the pixel coordinates. A region's motion
therefore carries three numbers: a mean disparity and two disparity gradients.

## Part 1. What the models are

This part decides which of the four proposals can be built from existing frozen models.

| Model | Verified | Recalled or not verified | Consequence |
| --- | --- | --- | --- |
| V-JEPA 2 (Assran et al., arXiv 2506.09985, 2025) | Self-supervised video encoder, ViT-L/H/g with patch 16; tubelets of 2 frames by 16 by 16 pixels; 256 or 384 pixel input; the ViT-g encoder gives a 16 by 16 token grid per tubelet step at 256 pixels; pretrained on about a million hours of internet video (instructional, action, egocentric) plus ImageNet; code under MIT. | Licence of the weights is not stated on the repository page. Pretraining frame rate not found. Whether the masked-prediction predictor is released was not established. | Tokens are coarse (16 pixels at 256 to 384 input) and the training motion is ground-level, small between frames. |
| V-JEPA 2-AC | An action-conditioned predictor of about 300 M parameters on the frozen encoder, post-trained on about 62 hours of robot video (DROID); actions are 7-D changes of end-effector state; 16 frames at 256 pixels and 4 frames per second. | | **It is not conditioned on camera pose.** No released V-JEPA model predicts latent features under a given camera motion. A pose-conditioned predictor would have to be trained, and at 300 M parameters it is not a light head. |
| V-JEPA 2.1 (Mur-Labadia et al., arXiv 2603.14482, 2026) | Adds a dense predictive loss over all tokens and deep self-supervision; models ViT-B to ViT-G at 384 pixels; reports NYUv2 depth with a linear probe and describes temporally consistent dense features. | No tracking or segmentation result with frozen features appears in the abstract; tubelet size not confirmed for this version. | The most suitable released encoder for a dense probe on video tokens. Its sensitivity to small residual motion is unknown. |
| SAM 2 (Ravi et al., arXiv 2408.00714, 2024) | Promptable segmentation in images and video with a streaming memory; `propagate_in_video` returns masks per object and frame (masklets); code and checkpoints under Apache 2.0; **automatic mask generation is provided for images only**, video needs a prompt per object. | Input resolution of 1024 pixels. Behaviour under large inter-frame shifts not found in the sources read. | Masklets for "everything" require image-level proposals in some frame and one tracked object per proposal. Cost grows with objects times frames. |
| SAM 3 (Carion et al., arXiv 2511.16719; listed in the ICLR 2026 proceedings) | Promptable concept segmentation: short noun phrases or image exemplars, returning masks and identities of all matching instances in images and video; a detector and a SAM 2-style tracker on a shared encoder; 848 M parameters; video latency grows with the number of objects, near real time for about five; videos of 30 seconds or less in the paper's setting; released under the "SAM License" with gated checkpoints. | Terms of the SAM License were not read. No statement on aerial imagery, water or glass was found. | Concept prompts exist. Whether "water" or "glass roof" work on nadir imagery is unknown. Dense masklets over hundreds of small aerial regions are far outside the stated operating point. |
| SAM on glass and mirrors | SAM often fails to detect mirrors and transparent objects (Han et al., arXiv 2305.00278, 2023); SAM 2 is reported as suboptimal for video mirror detection (Jie, arXiv 2412.19293, 2024). | Whether SAM 3 with a text prompt improves this. | The segmenter is weakest on a surface class that the method most needs as a unit. |

Two findings matter for the conception.

- **Proposal 3 cannot use a released latent predictor.** It has to be built as "warp in pixel space,
  encode both, compare", or a pose-conditioned predictor has to be trained. The first is a
  feature-metric reprojection error, which has a long history (Part 4).
- **Proposal 1 cannot use a video segmenter as released for all regions of an aerial frame** at
  acceptable cost, and a memory tracker is unproven at survey spacing. With known poses there is a
  cheaper association (W9).

Comparable backbones that do what the conception needs: for dense per-frame features with sharp
tokens, an image-native self-supervised encoder applied per frame (DINOv3, recalled); for
motion-aware dense features, V-JEPA 2.1; for geometry from several frames with a confidence output,
feed-forward multi-view models such as VGGT (title and venue verified; outputs recalled), which are
then also a baseline that the method must beat.

## Part 2. Objections

Each objection is given with the reason it could be fatal, a test, and the change that answers it.

### W1. Where a region's plane is witnessed, the monocular prior is redundant at the region level

**Why it may be fatal.** For a strip, each outline sample $c$ with unit normal $\mathbf n_c$ and
localisation error $\sigma_c$ constrains the disparity along the flight direction. The mean inverse
depth of the region has

```math
\frac{\sigma_d}{d} \approx \frac{\sigma_c}{p\, \sqrt{n_{\mathrm{eff}}}}, \qquad
n_{\mathrm{eff}} = \sum_c n_{c,x}^2 , \qquad p = \frac{f\, b}{d} ,
```

with $p$ the parallax in pixels. Worked example, not a measurement: adjacent survey frames with
$p = 800$ pixels, boundary repeatability $\sigma_c = 4$ pixels, $n_{\mathrm{eff}} = 25$ give 0.1% of
the flying height. A monocular depth at 2% is twenty times worse. If the witnessed planes enter the
surface, gating the monocular message changes the region's placement by very little. The harm of a
monocular fault is small exactly where its detectability is high.

**Test.** With exact depth, surface error for: monocular only; witnessed planes only; both ungated;
both with an oracle gate.

**Change.** Split the monocular message of a region into two parts (S2 in Part 3): its plane, three
numbers per masklet and frame, and its residual shape inside the region. Parallax judges and replaces
the first. The second is what the prior is kept for: roof detail, vegetation, terrain undulation,
which a plane does not carry. The role of the prior is then stated precisely, and the reconstruction
gain must be shown on residual shape and on regions without a witnessed plane.

### W2. A masklet is not a fault region

**Why it may be fatal.** A segmenter groups by object and appearance. A fault follows the depth
model's misreading, which may cover half a roof, or a roof together with its shadow on the ground. If
a fraction $q$ of a masklet is faulty by $\Delta$, the plane-level statistic sees roughly
$q \Delta$, and a region test loses power in proportion. Segmentation also has no unique scale: part,
object and group are all valid masks.

**Test.** With exact depth, decompose the true monocular error into between-masklet plane terms and
within-masklet residual, at two or three segmentation scales. Report the share of fault harm at each
level. If most of it is within masklets and not planar, the unit is wrong.

**Change.** Keep the pixel-level parallax gap as the second-order term with a defined role: after the
masklet plane is removed, the residual is tested per pixel where texture allows. Use a two-level
hierarchy of masks and take the level that maximises the standardised statistic, with the multiplicity
accounted for in calibration.

### W3. Outline motion is valid only for outlines the region owns

**Why it may be fatal.** Three cases break "the outline moves with the region's plane".

1. *Occluding contours.* Where a roof hides the ground, the shared boundary lies on the roof. The
   ground region's outline there moves with the roof's parallax, and a plane fitted to it is biased
   upward. In nadir views every building does this to its surroundings.
2. *Edges parallel to the flight direction.* They have $n_{c,x} = 0$ and carry no disparity. A road
   or canal running along the strip is constrained only at its ends, which are usually out of frame.
3. *Unbounded regions.* A lake or field larger than the frame has no closed outline, and the frame
   border is not a surface boundary.

A fourth case is not a bias but a noise term: mask boundaries flicker between frames, and at shadows,
wet margins or glint the boundary is not a material edge at all.

**Test.** Without ground truth: fit $\boldsymbol\theta$ per masklet from outlines, and measure the
residual outline distance after the fit ($\sigma_c$), the share of image area in masklets with a
well-conditioned fit, and the share of boundary length lost to cases 1 to 3. With ground truth: error
of the outline plane by surface class.

**Change.** Resolve ownership jointly. Fit planes for all masklets, assign each shared boundary to the
region that is nearer to the camera under the current planes, refit the other region without it, and
iterate. Accretion and deletion of visible area along the flight direction is a second ownership cue.
Use a second strip in another direction for case 2. Declare case 3 unwitnessed.

### W4. The outline is immune to reflections; the interior is not; and the segmenter may not see the surface

**Why it may be fatal, and why it is also an opportunity.** The reflection of a static scene in a flat
mirror is a static virtual point. Interior correspondences on calm water or glass follow the virtual
scene and satisfy every epipolar constraint. Any interior witness, by tracks or by features, then
testifies to the wrong surface with full confidence. The physical outline of the pane or the
shoreline does not: it lies on the real surface. This is the strongest argument for the region as
unit.

The risk is that the segmenter does not return the pane. If it segments the reflected content
instead, the masklets are virtual objects, their outlines move with virtual parallax, and the outline
witness inherits the reflection problem.

**Test.** On scenes with water and glazing: does the segmenter return the physical region; plane
error of outline and of interior witness against truth; frequency of masklets whose plane lies below
the surrounding terrain.

**Change.** Use both witnesses and read their disagreement (S1 in Part 3). A masklet whose own plane
lies well below an enclosing region's plane is treated as a virtual object and merged into its
enclosure.

### W5. Comparison in latent space adds no information where pixels carry none

**Why it may be fatal.**

1. *Texture-free interiors.* If the region's pixels are constant, every warp of it gives the same
   image, and no encoder can tell warps apart. Latent comparison is blind in the same regime as
   tracking. It may be more tolerant on weak texture; it cannot be informative on none.
2. *Coarse tokens.* A depth error $\Delta d$ misplaces the warp by $p\, \Delta d / d$ pixels. Features
   are tolerant to shifts well below a token. If the detectable shift is $s$ pixels at model
   resolution, the detectable error is $\Delta d / d \gtrsim s / p_{\mathrm{model}}$. Worked example:
   a whole frame reduced to 384 pixels with a fifth of the image as parallax gives
   $p_{\mathrm{model}} \approx 77$; with $s = 4$ this is 5% of the flying height. At native resolution
   in tiles, $p = 800$ and the same $s$ gives 0.5%. The method must run in tiles, and its cost is
   tiles times frame pairs times an encoder of up to a billion parameters. The value of $s$ is not
   known.
3. *No error bar.* A feature distance has no units and no null distribution. It also has no sign and
   no magnitude unless several depth hypotheses are compared, which is a plane sweep.
4. *Warp artefacts.* Disocclusion holes, resampling blur and stretched pixels are themselves
   violations of expectation for an encoder trained on natural video. The encoder may respond to the
   warp and not to the geometry.
5. *Reflections and view-dependent appearance.* Specular highlights violate expectation at the correct
   depth. On a flat mirror the comparison is satisfied by the virtual depth. The reflection problem is
   unchanged.
6. *Frame spacing.* A tubelet that spans two survey stills contains a shift of a fifth of the image,
   far from the motion in the encoder's training video.

**Test.** An injected-shift response curve, with no ground truth. Take pairs of real aerial frames,
register them with the best available depth, add a known residual shift of $k$ pixels along the
flight direction inside a region, and record the feature distance against $k$, per texture level and
per tile scale. This gives $s$, the noise floor, and the response on texture-free regions. If the
curve is flat below several pixels at native resolution, or flat on weak texture, the latent witness
is dead.

**Change.** Three changes make the latent witness a defined estimator (S3 in Part 3): stabilise the
clip before encoding, so the encoder sees small residual motion; compare two warps with the same
artefacts, not a warp with the raw clip; and sweep the three plane parameters per masklet, so that a
minimum and a curvature exist. With these it is a feature-space plane sweep per region, and the
explicit parallax gap is the same computation with pixel correlation.

### W6. A head on motion-aware tokens can copy the witness

**Why it may be fatal.** Spatiotemporal tokens of the raw clip encode parallax, including virtual
parallax on reflections. A head trained on labels derived from parallax and fed those tokens can
reproduce the label computation, errors included. Its agreement with the witness is then not
evidence.

**Test.** Train the head on tokens of the raw clip and, separately, on per-frame appearance tokens
pooled over the masklet. Compare on masklets where the witness is known to be wrong (reflections with
ground truth). If the motion-token head follows the witness there and the appearance head does not,
copying is shown.

**Change.** Keep the roles apart. Motion-aware tokens are evidence and enter through the witness.
The head is a prior and reads appearance pooled over the masklet, the masklet's shape descriptors, and
the depth model's frozen encoder. The motion-token head is an ablation.

### W7. Concept prompts give material classes, and a class is a propensity

**Why it may be fatal.** "Water" and "glass" say where the depth model tends to fail. They do not say
that it failed here. Precision on a stratum is capped at the fault rate within it. Concept prompts
also bring an open-vocabulary detector with its own errors and an unknown domain gap to nadir views,
and on glass the segmenter family is known to be weak (Part 1).

**Test.** Recall of each prompted concept on nadir frames against exact classes; fault rate within
each stratum per depth model.

**Change.** Use strata for three bounded purposes only: pooling the noise scale, switching the sign
convention on reflective strata, and reporting. Do not let a stratum raise a verdict.

### W8. Selective labels shrink, and what remains is harder

**Why it may be fatal.** With outlines as witness, most bounded regions get a plane label whether or
not they have texture, so the tracked set grows and the head has less to extrapolate. The unwitnessed
set is now: unbounded regions, non-planar untextured regions, regions whose outline is owned by
others or runs along the flight direction, and surfaces the segmenter does not return. Membership in
that set still shares causes with faults (large water bodies, glazing). The conditions for transfer,
that the fault is independent of witness availability given the features and that every feature
region has some witnessed examples, still have to be checked, and bounded lakes may not be
informative about open water.

**Test.** Without ground truth: a classifier that separates witnessed from unwitnessed masklets on
frozen features (an AUC near one means no support); the count of witnessed masklets per feature
cluster. With ground truth: head accuracy on unwitnessed masklets against the stratum base rate.

**Change.** Weight training by inverse propensity of being witnessed; return a third state,
*unsupported*, below a propensity floor; if a few synthetic labels are used, spend them on unwitnessed
surface types.

### W9. Association through time need not come from a memory tracker, and should not come from the judged depth

**Why it may be fatal.** A video segmenter's tracker is unproven at survey spacing and its cost
grows with the number of regions. The obvious alternative, lifting per-frame masks onto the surfel
map, uses the depth under judgment: a region with wrong depth lands in the wrong place and is matched
to the wrong region or to none.

**Test.** Association accuracy of three schemes at dense and at survey spacing: memory tracker on raw
frames; memory tracker on stabilised frames; geometric association.

**Change.** Stabilise every frame to the reference by the homography of the dominant ground plane,
which needs poses and a mean height only. Residual motion is then proportional to height above the
plane and is small for most of the scene, at any frame spacing. Segment frames in image mode and
associate masks by overlap on the stabilised frames, refining with the masklet's own plane as it is
estimated. A failed match at any plausible plane is itself a flag. The memory tracker is a fallback
for dense video.

### W10. Simpler pipelines that a reviewer will ask for

- *Per-segment re-anchoring of the monocular depth* to sparse triangulated points, with no witness
  theory at all.
- *Depth completion* conditioned on triangulated points (OMNI-DC, Marigold-DC, Prior Depth Anything).
- *Segmentation-guided multi-view stereo*, which already uses segment masks to carry depth into
  texture-free regions (W-table in Part 4).
- *A feed-forward multi-view model's* depth and confidence in place of the monocular prior.
- *The trivial rule*: distrust the prior wherever no interior witness exists.

Each must be in the comparison. The second and third are the closest competitors on texture-free
planar regions.

### W11. The learned part becomes small

With a plane witness for most bounded regions, the verdict there is computed, not learned. The head is
a prior for the unwitnessed remainder. A paper titled as a learned indicator would then have its
weight in an estimator. That is acceptable if stated; it is a problem if the abstract promises
learning.

### Summary of the attack

| Objection | Fatal if | Cheapest test | Needs exact depth |
| --- | --- | --- | --- |
| W1 redundancy | planes-only surface matches the oracle-gated fusion everywhere | four-fit comparison | yes |
| W2 unit | most fault harm is within masklets and non-planar | error decomposition by level | yes |
| W3 outline validity | boundary repeatability is many pixels, or few regions own a well-conditioned outline | outline fit residual and coverage | no |
| W4 segmenter on glass and water | the physical region is not returned | inspect masks on such scenes | partly |
| W5 latent witness | response curve flat below several native pixels or on weak texture | injected-shift response | no |
| W6 copying | motion-token head follows the witness on reflections | two-head comparison | yes |
| W8 support | witnessed and unwitnessed masklets separate, with no witnessed examples in hard clusters | domain classifier, counts | no |
| W10 alternatives | re-anchoring or completion matches the method | baseline runs | yes |

The three tests that need no ground truth (outline fit, injected-shift response, support) can be run
on one posed aerial sequence with a frozen segmenter and a frozen encoder.

## Part 3. What is sharp here, and a better form of the conception

The conception as stated has one strong idea (outline parallax), one weak one (latent comparison as a
new source of evidence), and one that is unbuildable from released models (a pose-aware latent
predictor). The following is the form that survives the objections.

### S1. Two witnesses with different physics: the outline and the interior

For each masklet estimate the plane twice.

- *Outline witness* $\hat{\boldsymbol\theta}^{\partial}_m$: from owned boundary samples only.
- *Interior witness* $\hat{\boldsymbol\theta}^{\circ}_m$: from interior appearance (features, pixel
  correlation or tracks).

Both come from one direct-alignment objective with three unknowns,

```math
E_m(\boldsymbol\theta) = \sum_{t} \Biggl[ \lambda_\partial \sum_{c \in \partial M_{m,r}^{\mathrm{own}}} \frac{D_{m,t}\bigl(H_t(\boldsymbol\theta)\, c\bigr)^2}{\sigma_c^2}
\; + \; \lambda_\circ \sum_{u \in M_{m,r}} \bigl\lVert \Phi_t\bigl(H_t(\boldsymbol\theta)\, u\bigr) - \Phi_r(u) \bigr\rVert^2 \Biggr] ,
```

with $D_{m,t}$ the distance to the masklet's boundary in frame $t$ and $\Phi$ dense frozen features;
the two witnesses are the minimisers with $\lambda_\circ = 0$ and $\lambda_\partial = 0$. Each has a
covariance from the curvature of its term, rescaled by the injected-shift calibration of W5.

Their comparison is a classification with physical meaning:

| Outline | Interior | Reading | Verdict on the monocular plane |
| --- | --- | --- | --- |
| defined | agrees | opaque textured surface | test against either; most precise case |
| defined | flat (no curvature) | untextured surface | test against the outline |
| defined | defined, behind the outline | reflection or transparency | test against the outline; the interior is a virtual scene |
| defined | defined, in front | outline not owned (occluder), or a mover | abstain; revisit ownership |
| undefined | defined | unbounded textured region | test against the interior |
| undefined | flat | unwitnessed | prior only; third state |

The third row is the point. It turns the reflection problem from an undetectable error of the witness
into a detected class, without labels and without a material classifier. The second row carries
parallax into texture-free regions. Neither is available to a per-pixel formulation.

The monocular plane of masklet $m$ in frame $i$, $\boldsymbol\theta^{\mathrm{mono}}_{m,i}$, is the
least-squares affine fit to the aligned inverse depth over $M_{m,i}$. The masklet gap is

```math
T_{m,i} = \bigl(\boldsymbol\theta^{\mathrm{mono}}_{m,i} - \hat{\boldsymbol\theta}_m\bigr)^{\!\top}
\bigl(\Sigma_{\hat\theta, m} + \Sigma_{\mathrm{mono}, m, i}\bigr)^{-1}
\bigl(\boldsymbol\theta^{\mathrm{mono}}_{m,i} - \hat{\boldsymbol\theta}_m\bigr) ,
```

referred to $\chi^2_3$ only after the covariances are calibrated. The repetition of a fault is read
from the same quantities: with $\bar{\boldsymbol\theta}^{\mathrm{mono}}_m$ the mean over frames,

```math
\varrho_m = \frac{\lVert \bar{\boldsymbol\theta}^{\mathrm{mono}}_m - \hat{\boldsymbol\theta}_m \rVert^2_{\Sigma}}
{\lVert \bar{\boldsymbol\theta}^{\mathrm{mono}}_m - \hat{\boldsymbol\theta}_m \rVert^2_{\Sigma} + \tfrac{1}{n_m} \sum_i \lVert \boldsymbol\theta^{\mathrm{mono}}_{m,i} - \bar{\boldsymbol\theta}^{\mathrm{mono}}_m \rVert^2_{\Sigma}} ,
```

the share of the monocular plane error that all frames have in common. Depth consensus sees the
second term of the denominator and is blind to the numerator.

What is identified: three plane parameters per masklet with at least two frames and enough owned
boundary not parallel to the flight direction, or enough interior texture. What is not: anything
about a masklet in the last row; the interior shape of an untextured non-planar region; a reflection
whose pane the segmenter does not return and whose virtual depth is plausible.

### S2. Judge three numbers, keep the shape

Write the aligned monocular inverse depth on a masklet as plane plus residual shape,
$\rho^{\mathrm{mono}}_i(u) = \boldsymbol\theta^{\mathrm{mono}\top}_{m,i} K^{-1} \tilde u + g_{m,i}(u)$.
The message of view $i$ to the surfels of the masklet splits accordingly: a component along the
three-dimensional subspace of common plane displacements, and the remainder. The gate acts on the
first component only, and the witnessed plane replaces it:

```math
\rho^{\mathrm{fused}}_i(u) = \bigl[(1 - \zeta_{m,i})\, \boldsymbol\theta^{\mathrm{mono}}_{m,i} + \zeta_{m,i}\, \hat{\boldsymbol\theta}_m\bigr]^{\!\top} K^{-1} \tilde u + g_{m,i}(u) ,
```

or the information-weighted equivalent. A flagged region keeps its internal relief and loses only its
placement. This answers three objections at once: gating no longer removes the only constraint on a
region; the prior's role is stated (shape within regions, everything on unwitnessed regions); and
detection and repair are one operation with one statistic. The residual $g$ is tested per pixel
where interior texture allows, which is the second-order term of proposal 1 with a definition.

Re-anchoring monocular depth per segment is not new in itself (Part 4). What is new is doing it in
the three-parameter plane subspace against a texture-free witness, with a calibrated test deciding
whether to do it.

### S3. The latent witness as a calibrated plane sweep on a stabilised clip

Stabilise all frames to the reference by the dominant ground plane. For a masklet and a candidate
$\boldsymbol\theta$, warp its region by $H_t(\boldsymbol\theta)$, encode, and compare with the
reference tokens; this is the interior term of $E_m$. Evaluate at the monocular plane, at the outline
plane, and on a small grid between and beyond. The two endpoints share warp artefacts, so their
difference isolates geometry. The injected-shift curve converts feature distance into pixels and
supplies the noise floor. With these three steps the latent comparison has a minimum, a curvature
and a unit, and the frame-spacing problem is removed because the encoder only ever sees residual
motion.

Whether a video encoder is better here than an image encoder applied per frame is open. Image tokens
are spatially sharper; tubelet tokens see sub-token motion in their first layer. The response curve
decides. No argument from first principles favours the video encoder for alignment; its natural role
is the stabilised-clip variant, in which "anything still moving" is the evidence.

### What is not a contribution

- Latent comparison as an independent source of evidence. It sees what pixels carry, with coarser
  resolution and no native error bar.
- A learned latent predictor conditioned on camera pose. None exists frozen, and training one is
  outside a light head.
- Concept prompts as a detector of faults.
- The masklet from a video tracker as such. With poses, association is geometric.

## Part 4. Prior work

### 4.1 Regions, planes and layers from motion

| Work | What it does | Relation |
| --- | --- | --- |
| Plane-plus-parallax and layered motion (Irani and Anandan; Sawhney; Wang and Adelson, 1990s). *Recalled; no citation details verified.* | Register a dominant plane, read residual motion as height; describe a scene as moving layers with ownership of boundaries. | The stabilised clip, residual parallax and boundary ownership used here are these ideas. A paper must cite them and not present them as new. |
| "MSP-MVS: Multi-Granularity Segmentation Prior Guided Multi-View Stereo", arXiv 2407.19323. *Title verified from a search listing; content recalled only in outline.* | Uses segment masks at several scales to guide patch-match stereo into texture-free regions. | The closest existing use of foundation segmentation to carry multi-view depth across untextured regions. It reconstructs; it does not judge a prior, and it does not separate outline from interior. |
| "SuperPrimitive: Scene Reconstruction at a Primitive Level", arXiv 2312.05889; planar splatting methods (Planar Gaussian Splatting, WACV 2025; PlanarSplatting, arXiv 2412.03451). *Titles from search listings only.* | Reconstruction with segment-level or planar primitives, some using monocular normals or depth per segment. | Region-level primitives with monocular cues exist. None was verified to test the monocular cue against region parallax. |
| Zhang et al., "MonoInstance", arXiv 2503.18363, 2025. *Verified (arXiv).* | Aligns per-instance monocular depth across views and derives uncertainty from the spread. | Region-level treatment of monocular depth, by consensus of the depth maps. Blind to errors that all views repeat. |
| Ye et al., "Gaussian Grouping: Segment and Edit Anything in 3D Scenes", ECCV 2024. *Verified (title, venue).* | Lifts 2D masks to Gaussians with consistent identities. | Mechanism for carrying masklet identity on surfels; relies on the fitted geometry (W9). |

### 4.2 Feature-space reprojection and expectation violation

| Work | What it does | Relation |
| --- | --- | --- |
| Shu et al., "Feature-metric Loss for Self-supervised Learning of Depth and Egomotion", ECCV 2020. *Verified (title, venue).* | Replaces photometric reprojection error with error in a learned feature space. | Proposal 3 is this error with a frozen encoder, used for testing and not training. |
| Feature-metric refinement in structure-from-motion and feature-metric bundle adjustment. *Recalled.* | Align views by minimising differences of deep features. | Same computation as the interior term of $E_m$. |
| Kulhanek et al., "WildGaussians", arXiv 2407.08447. *Verified (arXiv).* | Uncertainty from similarity of frozen features between render and image. | Feature-space comparison of prediction and observation for masking, on colour. |
| Garrido et al., "Intuitive physics understanding emerges from self-supervised pretraining on natural videos", arXiv 2502.11831, 2025. *Verified (arXiv abstract).* | Measures surprise of latent video prediction models on physically impossible clips; representation-space predictors score above chance. | The precedent for "violation of expectation" with this model family. It uses the model's own predictor on ordinary video; nothing shows sensitivity to small geometric inconsistency. |
| Assran et al., V-JEPA 2; Mur-Labadia et al., V-JEPA 2.1. *Verified as in Part 1.* | Video encoders; an action-conditioned predictor for robot arms. | No camera-pose conditioning. |

### 4.3 Segmentation foundation models

SAM 2 and SAM 3 as in Part 1. Han et al. (arXiv 2305.00278) and Jie (arXiv 2412.19293) on glass and
mirrors, verified. No evaluation of SAM 2 or SAM 3 masklets on nadir aerial sequences, or at survey
frame spacing, was found; the search was brief.

### 4.4 Witnesses and labels from multi-view geometry

| Work | What it does | Relation |
| --- | --- | --- |
| Mostegel et al., CVPR 2016; Tosi et al., BMVC 2017. *Titles and venues verified.* | Train stereo confidence from geometric self-contradiction without ground truth. | Ancestor of "a geometric contradiction is a free label". |
| Pataki, Sarlin, Schönberger, Pollefeys, "MP-SfM", arXiv 2504.20040, 2025. *Verified (arXiv).* | Monocular depth and normal priors inside structure-from-motion with uncertainty propagation. | Same pairing of prior and parallax. How its prior uncertainty is obtained was not read and must be before a novelty claim. |
| Bae, Budvytis, Cipolla, MaGNet, CVPR 2022; Guo et al., Murre, CVPR 2025. *Verified.* | Fuse or condition monocular depth with multi-view geometry. | The "repair" alternative. |
| OMNI-DC (arXiv 2411.19278), Marigold-DC (arXiv 2412.13389), Prior Depth Anything (arXiv 2505.10565). *Titles verified.* | Zero-shot depth completion from sparse metric depth. | Baselines of W10. |
| Karaev et al., CoTracker3, arXiv 2410.11831; Harley et al., AllTracker, ICCV 2025. *Verified.* | Frozen point and dense trackers. | The interior witness by tracks; kept as the calibrated special case. |
| Goli et al., RoMo, ICCV 2025. *Title and venue verified.* | Motion masks from flow and epipolar cues. | Epipolar violation as a mover flag; silent on reflections that satisfy the constraint. |

### 4.5 Residual-supervised heads and depth-prior reliability in splatting

NeRF On-the-go (CVPR 2024), SpotLessSplats (arXiv 2406.20055) and WildGaussians supervise small heads
on frozen features by a photometric residual; verified. CDGS (arXiv 2502.14684), "In Depth We Trust"
(arXiv 2604.05715), a photometric-mask method (arXiv 2607.02554) and ConfidentSplat
(arXiv 2509.16863) weight or restrict monocular depth supervision by fixed cues or by agreement of
depth maps; verified at the level of their arXiv pages. None was found to use region outlines as a
witness or to separate a region's plane from its shape.

### 4.6 Fault detection by redundancy

Data snooping and solution separation (Baarda 1968, recalled; Blanch et al., ION GNSS 2012 and IEEE
TAES 2015, titles and venues verified) state that a fault common to all redundant observations needs
an independent source, and that the independent source protects only where it observes. The outline
witness enlarges where it observes.

**Summary of the gap.** No work was found that (a) estimates a region's plane from the parallax of
its owned outline to judge a monocular prior, (b) uses disagreement between outline and interior
parallax to detect reflection, or (c) gates a depth prior in the plane subspace of a region while
keeping its shape. The search was not exhaustive. The segmentation-guided stereo literature and the
planar-primitive reconstruction literature were seen only as titles and are the most likely place
for an overlap.

## Part 5. Verdict

**Is it a contribution at a top venue?** The outline witness with ownership, the outline-interior
comparison, and the plane-subspace gate together are a coherent method with a clear physical argument
and, as far as was checked, no direct precedent. That is enough for a strong robotics paper and
plausible for a vision venue if the reflection result holds on real data. The latent comparison and
the video encoder are not what would carry it; they are an interior witness among others and should be
presented as such. If the paper leads with "violation of expectation in a video world model", it
leads with its weakest part.

**The single claim.** *The parallax of a region's own outline judges a monocular depth prior on
surfaces where no interior correspondence exists or where interior correspondence is deceived by
reflection; the verdict concerns three plane parameters per region, the prior's shape within the
region is kept, and regions without a witness are reported as such.*

**Cheapest tests, in order.**

1. *No ground truth, one posed aerial sequence.* Segment frames, stabilise by the ground plane,
   associate, fit outline planes with ownership. Report boundary repeatability, the share of area with
   a well-conditioned outline plane, and the share of masklets in each row of the table in S1.
2. *No ground truth.* Injected-shift response curves for one video encoder and one image encoder.
3. *Exact depth.* Outline-plane error by surface class; whether water and glazing are returned as
   regions; decomposition of fault harm into plane and shape; the four-fit comparison of W1; the
   baselines of W10.

**Kill criteria.** Abandon the region formulation if any of the following holds:

- boundary repeatability after the plane fit is so poor that outline planes are no more precise than
  the monocular depth, or fewer than a small share of regions own a well-conditioned outline;
- the segmenter does not return water and glazing as regions on nadir imagery, by image mode or by
  concept prompt;
- most fault harm lies in shape within regions and not in their planes;
- per-segment re-anchoring to sparse triangulated points, or zero-shot completion, matches the method
  on texture-free regions.

Abandon the latent witness alone, keeping the rest, if the response curve is flat below several
native pixels or on weak texture.

**Parts to drop.**

- A pose-conditioned latent predictor.
- A head fed motion-aware tokens as the main head.
- Concept strata as evidence.
- Masklets from a memory tracker at survey spacing as the default association.

**What stays unknown.** Whether a frozen segmenter returns the physical extent of water and glass in
nadir aerial frames with boundaries repeatable to a few pixels. The outline witness, the reflection
test and the plane-subspace gate all rest on that, and it can be checked without ground truth on the
first sequence available.

## Sources consulted

- https://arxiv.org/abs/2506.09985 and https://arxiv.org/html/2506.09985
- https://arxiv.org/abs/2603.14482
- https://cdn.jsdelivr.net/gh/facebookresearch/vjepa2@main/README.md
- https://arxiv.org/abs/2408.00714 and https://github.com/facebookresearch/sam2
- https://arxiv.org/html/2511.16719v2 and https://github.com/facebookresearch/sam3
- https://proceedings.iclr.cc/paper_files/paper/2026/hash/e0982cbc81401df3430ee1ff780dc7a2-Abstract-Conference.html
- https://arxiv.org/abs/2305.00278
- https://arxiv.org/abs/2412.19293v1
- https://arxiv.org/abs/2502.11831
- https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3301_ECCV_2020_paper.php
- https://arxiv.org/abs/2312.00732
- https://arxiv.org/html/2407.19323v6
- https://arxiv.org/html/2312.05889v2
- https://arxiv.org/abs/2503.18363
- https://arxiv.org/abs/2407.08447v2
- https://openaccess.thecvf.com/content_cvpr_2016/html/Mostegel_Using_Self-Contradiction_to_CVPR_2016_paper.html
- https://bmva-archive.org.uk/bmvc/2017/papers/paper133/index.html
- https://arxiv.org/abs/2504.20040
- https://openaccess.thecvf.com/content/CVPR2022/html/Bae_Multi-View_Depth_Estimation_by_Fusing_Single-View_Depth_Probability_With_Multi-View_CVPR_2022_paper.html
- https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Multi-view_Reconstruction_via_SfM-guided_Monocular_Depth_Estimation_CVPR_2025_paper.html
- https://arxiv.org/pdf/2411.19278, https://arxiv.org/abs/2412.13389, https://arxiv.org/abs/2505.10565
- https://arxiv.org/abs/2410.11831v1 and https://arxiv.org/abs/2506.07310
- https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html
- https://openaccess.thecvf.com/content/ICCV2025/html/Goli_RoMo_Robust_Motion_Segmentation_Improves_Structure_from_Motion_ICCV_2025_paper.html
- https://openaccess.thecvf.com/content/CVPR2024/html/Ren_NeRF_On-the-go_Exploiting_Uncertainty_for_Distractor-free_NeRFs_in_the_Wild_CVPR_2024_paper.html
- https://arxiv.org/abs/2406.20055, https://arxiv.org/abs/2502.14684, https://arxiv.org/abs/2604.05715, https://arxiv.org/abs/2607.02554, https://arxiv.org/abs/2509.16863v1
- https://web.stanford.edu/group/scpnt/gpslab/pubs/papers/Blanch_et_al_IONGNSS_2012_v2.pdf and https://dblp.uni-trier.de/rec/journals/taes/BlanchWELPRSK15.html
