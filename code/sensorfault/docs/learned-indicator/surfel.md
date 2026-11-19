# The Map as Geometry, Appearance and Information: the Surfel Formulation

## 0. Scope and summary

The setting is large-area mapping from posed aerial video in strips, with one monocular depth map per
frame and 2D Gaussian surfels as the scene representation. This note defines what the representation
stores beyond geometry and appearance, how evidence from many views reaches it without
correspondences, what exactly is measured when geometry and appearance are said to be aligned, and how
pose error, errors of the monocular model and errors of the rendering model enter the same
information model. It gives the argument in plain sentences first (Section 1) and the supporting
equations after. Notation follows `docs/learned-indicator.md`: placement
$\boldsymbol\theta = (\nu, \tau_1, \tau_2)$ of a surfel or entity, depth Jacobian row $\mathbf j_{uk}$,
image sensitivity $\mathbf g_{uk}$, entity $m$, tension $\mathbf t_m$ and $T_m$, dispersion
$\hat\varphi$, retained fraction $\kappa_m$, generalised eigenvalues $\vartheta$, surface term
$\boldsymbol\mu_m$ and per-frame term $\boldsymbol\alpha_{mj}$ of the monocular error, repetition
coefficient $\varrho$.

Nothing in this note has been measured and no experiment was run for it. Statements about behaviour
are predictions.

Summary.

1. A map is geometry, appearance and information. The information is stored per evidence source as a
   small symmetric block on the placement of each primitive and of each surface entity. Opacity is
   not confidence, and a single merged block would not allow any of the tests below.
2. For a planar primitive both evidence sources act on the same three placement numbers through the
   same geometric row. The image information is the stiffness of the coupling between appearance and
   geometry; the misfit between the two sources is the stretch; the test statistic is the stored
   energy, and half of it is a log-likelihood ratio in nats.
3. "Correspondence-free" has an exact meaning: no association between images is estimated. The
   primitive is the correspondence. This scales with frames instead of frame pairs and degrades to
   zero information on weak texture instead of failing. Its price is a capture range of about one
   footprint, and the monocular depth is what brings the surface inside it. Where the prior is
   grossly wrong, a search over three numbers per entity replaces it.
4. Nuisances fall into four classes that must not be confused: marginalised with a stated loss of
   information (pose, per-frame depth scale and offset, exposure, free appearance); estimated as
   shared parameters (distance bias, camera model); turned into variance (per-frame depth error,
   rendering misfit); and not absorbable, hence detected (the depth error every frame repeats,
   reflections, motion along the strip).
5. Honest depth information saturates. With a surface term of covariance $\Sigma_\mu$, the
   information of any number of repeating frames is bounded by $\Sigma_\mu^{-1}$. A representation
   that adds nominal blocks grows without bound and outvotes the pixels; one that stores the bound
   listens to the pixels at the right rate without any gate. The gate is needed only for the tail
   beyond that law.

## 1. The argument

Each step is a premise, what follows from it, and what it means for the design.

1. *A map that is used must say how well each part of it is known.* Large-area mapping fuses
   thousands of frames, and downstream use (measurement, change detection, planning) needs to know
   where the surface is determined and by what. Gaussian splatting stores where things are and how
   they look. Its opacity says how much a primitive occludes, not how well it is placed. **Design:**
   store information explicitly, next to geometry and appearance.

2. *Evidence has provenance, and tests are comparisons between provenances.* The monocular prior and
   the images can each be wrong in ways the other is not. If their information is merged into one
   number, the map can no longer say whether they agree. **Design:** one information block per
   source, and within a source per block of frames, so that any subset can be removed by subtraction
   and any two subsets compared.

3. *The comparison needs a common coordinate.* Depth evidence speaks about where a surface is along
   rays. Image evidence speaks about how appearance moves between views. For a planar primitive these
   are statements about the same three numbers: offset along the normal and two tilts. The derivative
   of the rendered image with respect to them is the derivative of the ray depth times a scalar.
   **Design:** planar primitives, and all information on placement. This is the job of the surfel.

4. *The image's knowledge of geometry is a coupling, not a property of the image.* How strongly
   appearance constrains placement is texture contrast times the spread of viewing directions. A
   blank surface or a single view gives none. **Design:** the image information is computed, not
   assumed, and it is zero where it should be; abstention is not a rule but a value.

5. *Cross-view evidence can enter without matching.* Rendering the shared primitives into every view
   and comparing with that view is a comparison across views, because the same primitive is rendered
   in all of them. Nothing is associated between images. **Design:** analysis by synthesis, with
   visibility and occlusion from the rasteriser.

6. *A local comparison only works near the answer.* The image residual is linear in placement for
   shifts below a footprint. **Consequence:** direct mapping needs a starting surface inside that
   range. **Design:** the monocular depth places the surface; the images then refine and judge it.
   This is why the prior is kept even where the pixels are informative, and why its gross faults are
   the cases that need a search.

7. *A fault moves a surface, not a primitive.* One primitive moved alone tears the surface and
   measures coverage, not parallax. **Design:** the unit of placement, information and test is the
   surface entity, a set of primitives that segment masks have grouped.

8. *What disturbs the comparison must be in the model, each in the way that fits it.* A camera pose
   that is slightly wrong moves everything in a frame together; a depth map has its own scale; the
   renderer is never exact. If these are ignored, the test detects them instead of faults. If they are
   given unlimited freedom, they explain faults away. **Design:** low-order nuisances with stated
   priors, marginalised, and the information they cost is reported.

9. *Information must be honest to be used as a weight.* Frames that repeat an error do not add
   knowledge. **Design:** the stored depth information includes the error that frames share, so that
   fusion by information is right on average, and a test handles what is left: surfaces where the
   prior is wrong beyond its ordinary scatter.

10. *What the map cannot judge it must say it cannot judge.* Where the pixels are blind and the prior
    is the only source, no test exists. **Design:** a learned prior on frozen features carries the
    verdict there, with an explicit unsupported state where it has nothing to carry.

## 2. What the representation stores

Per primitive $k$: geometry (centre, orientation, scales, opacity) and appearance $\mathbf a_k$ in each
rendered channel, as in any surfel map; and, in addition,

```math
\bigl\{ (\Lambda^{s}_{k},\ \boldsymbol\eta^{s}_{k}) \bigr\}_{s \in \mathcal S}, \qquad \Lambda^s_k \in \mathbb S^{3}_{+}, \quad \boldsymbol\eta^s_k \in \mathbb R^3 ,
```

information and information vector on the placement $\boldsymbol\theta_k$, one pair per source $s$.
Sources are the depth family and the image family, each subdivided by block of frames and, for the
image family, by channel (colour, frozen features, labels). Per entity $m$ the same objects are stored
on the three entity placement parameters, together with the dispersion of each source and the frozen
feature of the entity. A symmetric $3\times 3$ block is six numbers, so the overhead per primitive is
nine numbers per source, small against the appearance coefficients a splat already carries.

Stored this way the information is additive and local. Fusing a new frame is an addition. Removing a
block of frames is a subtraction. Tiles of a large area are fused independently and joined by adding
blocks on shared primitives. Marginalising frames that have left the working window keeps their
information on the primitives, as in an information filter.

**The spring picture, made exact.** For an entity with image information $\Lambda^I_m$ and depth
information $\Lambda^D_m$:

| Picture | Quantity | Units |
| --- | --- | --- |
| stiffness of the coupling between appearance and geometry | $\Lambda^I_m / \hat\varphi$ | inverse squared metres and radians |
| stiffness of the monocular prior | $\Lambda^D_m$ (honest form, Section 6.3) | the same |
| stretch | $\mathbf t_m = \hat{\boldsymbol\theta}^D_m - \hat{\boldsymbol\theta}^I_m$ | metres, radians |
| stored energy | $\tfrac12 T_m$ | nats |
| what the pixels add to what the prior knows | $\tfrac12 \sum_a \log(1 + \vartheta_a)$ | nats |

The last row is the reduction of differential entropy of the placement when the image information is
added to the depth information, with $\vartheta_a$ the generalised eigenvalues of
$(\Lambda^I_m/\hat\varphi, \Lambda^D_m)$. It is the sense in which the representation is
information-theoretic: stiffness is Fisher information, and its logarithm, relative to what was
already known, is information gained.

## 3. Two sources on the same three numbers

**Depth.** With compositing weights $w_{uk}$ and monocular noise $\sigma_i(u)$, frame $i$ sends

```math
\Lambda^D_{ik} = \sum_u w_{uk}\, \frac{a_i^2\, \mathbf j_{uk}^\top \mathbf j_{uk}}{\sigma_i^2(u)}, \qquad
\boldsymbol\eta^D_{ik} = \sum_u w_{uk}\, \frac{a_i\, \mathbf j_{uk}^\top e^D_i(u)}{\sigma_i^2(u)},
\qquad
\mathbf j_{uk} = \frac{\bigl[\,1,\ -\mathbf t_1^\top(\mathbf x_{uk} - \boldsymbol\mu_k),\ -\mathbf t_2^\top(\mathbf x_{uk} - \boldsymbol\mu_k)\bigr]}{\mathbf n^\top \mathbf w_u} .
```

The weight is $w_{uk}$ and not its square. Since the normalised weights sum to one per pixel, Jensen's
inequality makes this block-diagonal matrix a majoriser of the frame's Hessian, so a step taken with
it cannot overshoot; blocks from squared weights overshoot the common mode of overlapping primitives
by $\sum_u w_{uk} / \sum_u w_{uk}^2$.

**Image.** Placement enters the rendered appearance only through the depth at which a ray meets the
plane, so

```math
\frac{\partial \hat{\mathbf c}_i(u)}{\partial \boldsymbol\theta_k} = \mathbf g_{uk}\, \mathbf j_{uk}, \qquad
\mathbf g_{uk} = - w_{uk}\, (\mathbf a_k - \mathbf b_{uk})\, (\mathbf s_{uk}^\top \mathbf m_{uk}) ,
```

with $\mathbf b_{uk}$ the appearance composited behind $k$, $\mathbf s_{uk}$ the intersection in the tangent
frame and $\mathbf m_{uk}$ the tangential component of the viewing ray per unit scale. Contrast with
what lies behind, times how far the footprint slides per unit depth. For an entity moved coherently
through $P_{km}$,

```math
\Lambda^I_{im} = \sum_u \frac{J^I_{im}(u)^\top J^I_{im}(u)}{\sigma_I^2}, \qquad J^I_{im}(u) = \sum_{k \in \mathcal K_m} \mathbf g_{uk}\, \mathbf j_{uk}\, P_{km} .
```

One view gives no information on the offset, because an offset is confounded with a tangential shift
of the texture. Eliminating that shift, the offset information is texture energy $E_i$ times the
spread of slides over the views,

```math
\lambda^I_\nu = \Bigl( \sum_i E_i \Bigr) \operatorname{Var}_E(m_i) ,
```

the stiffness of point 4 of the argument. It is the precision of a triangulation, reached as a Schur
complement and without a correspondence.

## 4. Correspondence-free: the exact sense, the gain and the price

**Sense.** No variable of the model associates a pixel, feature or region of one image with one of
another image, and none is estimated. Every residual is between one image and a rendering of the
shared primitives into that image. Views are related only because the same primitive appears in each.
The data association problem of matching is replaced by a visibility computation, which the
rasteriser solves by depth order.

Two qualifications keep the statement honest. Segment masks of different frames are merged when they
land on the same primitives; that is association through the geometry, not between images, and it
inherits the geometry's errors. And the profile of Section 4's last paragraph is a search.

**Gain.**

- *Scale.* Cost grows with the number of frames, not of overlapping frame pairs. A primitive seen in
  thirty frames is rendered thirty times; pairwise matching would consider several hundred pairs.
- *Weak texture.* A matcher on weak texture returns nothing or a wrong match. Here the information
  $\Lambda^I$ falls continuously to zero and the prior takes over in proportion. There is no failure
  event, only a stiffness.
- *Occlusion.* A residual is formed only where the rasteriser says the primitive is visible.
- *Uniform treatment of channels.* Colour, frozen features and segment labels are rendered channels
  with the same Jacobian. The label channel gives parallax of an entity's outline when its interior
  has nothing to match.
- *Any camera.* A primitive does not know which camera views it. Frames of another strip, an oblique
  camera or a later pass are further renderings. This covers the second reading of cross-view
  evidence, with two conditions: appearance must be comparable (per-camera gain and offset; the
  feature channel where illumination or season changed), and visibility must be one-sided, which
  planar surfels provide. A cross strip also supplies the direction of parallax that one strip lacks,
  so tilt across the strip and outlines along the flight line become informative. Because sources are
  stored separately, the tension between two passes is defined and is a registration check between
  them.

**Price.** The residual is linear in placement for image shifts below about one footprint. Explicit
matching has no such limit. Consequences:

- The starting surface must lie within that range. The monocular depth, after alignment, is expected
  to put most of the scene there; this is its first job in the method and the logical reason it is
  not replaced by the pixels.
- Where the prior is wrong by more, the local pull is weak and its direction arbitrary. The estimate
  is then a profile: move the entity over a range of its three parameters, re-render, and take the
  minimum, coarse to fine in image scale. This is a plane sweep on entities. It associates nothing
  between images, but it is a search, and its cost is renders per entity per hypothesis.
- Repetitive texture has several minima within the range of plausible faults; the prior chooses
  among them, and a wrong prior chooses wrongly.
- Precision is that of a dense direct method; at fine texture and with exact poses a feature-based
  bundle adjustment may place individual points better.

## 5. Measuring the alignment of geometry and appearance

**Measure.** For an entity, each source alone would place it at
$\hat{\boldsymbol\theta}^s_m = (\Lambda^s_m)^{-1}\boldsymbol\eta^s_m$. The misalignment is the vector

```math
\mathbf t_m = \hat{\boldsymbol\theta}^D_m - \hat{\boldsymbol\theta}^I_m \quad \text{(metres along the normal; radians of tilt)},
```

and its size against what both sources can resolve is

```math
T_m = \mathbf t_m^\top \bigl( (\Lambda^D_m)^{-1} + \hat\varphi\, (\Lambda^I_m)^{-1} \bigr)^{-1} \mathbf t_m .
```

$T_m$ is dimensionless. It is the increase of the total misfit when the two sources are forced to
share one placement, and $\tfrac12 T_m$ is the log-likelihood ratio, in nats, between "one placement"
and "two placements" under the Gaussian model.

**Null.** If both sources are unbiased with independent errors and the stated information is honest,
$T_m \sim \chi^2_3$ (fewer degrees where $\Lambda^I_m$ is rank-deficient). Honesty is the condition
that carries weight and it is checked, not assumed: the dispersion $\hat\varphi$ of the image family
comes from the scatter of per-frame image placements within the entity; the depth information is
taken in its saturated form (Section 6.3); placebo tensions between split halves of one source give
the null scale directly; and moving an entity by a known amount and re-rendering must return that
amount with the stated variance.

**Where it is evaluated.** At a joint optimum the pulls are equal and opposite by construction, and
free appearance can relieve the stretch. The measure is therefore taken at the surface placed by
depth alone, with appearance projected from source frames and judged on disjoint frames. There
$\boldsymbol\eta^D_m = 0$ and $T_m$ is a score statistic in the image pull.

**Decomposition.** With per-frame messages, the total misfit of an entity about a common placement is
the sum of the scatter among depth frames, the scatter among image frames, and $T_m$. The first
contains only the per-frame monocular terms $\boldsymbol\alpha_{mj}$, the last contains the surface
term $\boldsymbol\mu_m$, and the middle one contains neither. A fault that every frame repeats is a
between-source effect and nothing else; no comparison among depth maps can contain it.

**What a large value means.** Not that the prior is wrong. It means the sources disagree. Attribution
uses the within-image scatter and the split of the image message into outline and interior: an
interior that asks for a placement behind an outline that agrees with the prior is a reflection or a
view through glass, not a fault of the prior.

## 6. How errors enter the model

"Absorbed" is used here in one sense only: a nuisance is given parameters or a variance inside the
same information model, and the placement information that this costs is computed. It never means
that the error has no effect.

### 6.1 Marginalised, with a stated cost

Let $\mathbf n$ collect nuisance parameters with prior information $\Lambda_{\mathbf n}^0$. Placement
information after marginalisation is the Schur complement

```math
\Lambda^{\mathrm{marg}}_{\theta\theta} = \Lambda_{\theta\theta} - \Lambda_{\theta \mathbf n} \bigl( \Lambda_{\mathbf n \mathbf n} + \Lambda^0_{\mathbf n} \bigr)^{-1} \Lambda_{\mathbf n \theta},
\qquad
\kappa = \frac{\operatorname{tr} \Lambda^{\mathrm{marg}}_{\theta\theta}}{\operatorname{tr} \Lambda_{\theta\theta}} \in [0, 1] ,
```

and $\kappa$ is reported per entity and per nuisance.

- *Pose of each frame* (six parameters, with the covariance of the navigation solution as prior).
  The structure is that of bundle adjustment with entities in place of points: eliminating the
  entities leaves a reduced system in the poses, banded along a strip and coupled across strips. What
  is lost: a placement pattern common to all entities of a frame is confounded with its pose. A
  uniform offset along the viewing direction trades against camera height, a common tilt against
  rotation, and an error of the baseline along the track against a common scale of depth. What
  survives is the placement of an entity relative to the others in the frame. A regional fault
  survives; a fault covering most of a frame does not, and with a tight pose prior it reappears as an
  implausible pose correction, which is itself reported.
- *Scale and offset of each depth map.* They remove from each frame's depth message the component of
  placement that is affine in depth over the whole frame. By definition that component is not a
  fault. The cost is largest over flat terrain seen from nadir, where depth varies little and scale
  and offset are nearly collinear.
- *Exposure gain and offset per frame and channel.* Cheap: they do not resemble a slide.
- *Free appearance.* If appearance is fitted, its freedom is marginalised the same way. One diffuse
  value per primitive seen by many frames keeps $\kappa$ near one. View-dependent colour in a strip,
  where viewing direction and parallax vary together, leaves almost nothing. This is why the test
  uses projected appearance with no free parameters.

### 6.2 Estimated as shared parameters

Parameters common to all frames are estimated once and cost a negligible share of information per
entity, but they must be in the model or they appear as tension with a signature: the *distance bias*
of the monocular model (tension that grows with range), and *residual camera calibration and rolling
shutter* (tension that depends on image position). Both signatures are diagnostics before they are
corrections.

### 6.3 Turned into variance

- *The per-frame term of the monocular error.* A random effect with covariance $\Sigma_\alpha$ per
  frame, or per block of frames that share it.
- *The ordinary surface term.* Frames repeat part of their error even on healthy surfaces. With
  covariance $\Sigma_\mu$ for the inlier class, the honest depth information of an entity is

  ```math
  \Lambda^{D,\mathrm{hon}}_m = \Bigl( \Sigma_\mu + \Bigl[ \sum_j \bigl( \Sigma_\alpha + (\Lambda^D_{mj})^{-1} \bigr)^{-1} \Bigr]^{-1} \Bigr)^{-1} \ \preceq\ \Sigma_\mu^{-1} .
  ```

  However many frames observe an entity, the prior's information on its placement cannot exceed
  $\Sigma_\mu^{-1}$. The nominal sum $\sum_j \Lambda^D_{mj}$ grows linearly and, used as a weight,
  silences the pixels on exactly the long strips where frames repeat each other. Both covariances are
  estimated without ground truth from the decomposition of Section 5, per appearance stratum; their
  ratio is the repetition coefficient $\varrho$.
- *Rendering model error.* Residuals at a fit are misfit, not sensor noise, and are correlated in
  space. A dispersion $\hat\varphi$ per stratum and channel, from the within-image scatter with frames
  as units, divides the image information. This corrects the scale of the uncertainty. It does not
  correct a bias.

### 6.4 Not absorbable: detected

- *The surface term beyond its ordinary law*: a prior that places a surface wrongly in every frame.
  This is the fault. Giving it a parameter per entity would be the same as discarding the prior's
  placement everywhere. It is tested by $T_m$ and, on a positive verdict, the placement component of
  the entity's depth message is gated while its shape is kept.
- *A second layer*: reflection in still water or flat glass, or a view through glass. The image
  family is consistent and wrong. Detected by outline against interior where an attached outline is
  in view; otherwise not.
- *Motion along the strip at constant speed.* Consistent with a wrong height. Detected only as a
  small entity leaving a well-determined surface, or by a cross strip.
- *Errors of shape inside an entity.* The entity model is three numbers. A deformed interior is seen
  only where primitives carry image information of their own.
- *Bias of the renderer that resembles a slide.* Not known to exist in general; injection measures it
  where it does.

### 6.5 Summary

| Error | Treatment | Cost or limit |
| --- | --- | --- |
| frame pose | marginalised under navigation prior | frame-wide placement patterns lost |
| depth scale and offset per frame | marginalised | frame-wide affine depth component lost |
| exposure | marginalised | negligible |
| free appearance | marginalised, or avoided by projection | up to all image information if view-dependent |
| distance bias, camera model | estimated, shared | signature in range or image position if omitted |
| per-frame monocular term | variance $\Sigma_\alpha$ | depth information reduced |
| ordinary surface term | variance $\Sigma_\mu$ | depth information saturates |
| rendering misfit | dispersion $\hat\varphi$ | image information divided; bias untouched |
| surface term in the tail | detected, gated on placement | needs image information on the entity |
| reflection, transparency | detected by layer test | needs an attached outline in view |
| mover along the strip | partly detected | needs a cross strip in general |
| pixel-blind entity with a faulty prior | not detected; learned prior or unsupported | no label-free check |

## 7. What the representation makes possible

- *Fusion by information.* With honest blocks, the fused placement weights the pixels by
  $\vartheta/(1+\vartheta)$ per direction. No schedule or hand-set loss weight between depth and
  photometric terms is involved.
- *Listening rule.* Pixels dominate where $\vartheta \gg 1$; the prior is nearly irrelevant there and
  its verdict is a label. Both are fused and tested where $\vartheta \approx 1$. The prior stands
  alone where $\vartheta \ll 1$. For a fault of given harm, detectability rises with the pixels'
  share and vanishes with it, so evaluation is weighted by harm.
- *Two gates.* On the placement component of an entity's depth message, and on the interior image
  message of an entity attributed to a second layer. A depth gate on an entity without image
  information leaves it free in placement, and the map says so.
- *Labels for learning without annotation.* The verdicts, with noise known through the stored
  information, train light heads on frozen features lifted onto the same primitives: which surfaces
  make the depth model fail, which mislead the pixels, and whether an appearance is ever judged. The
  last defines the unsupported state.
- *A map that reports its own knowledge.* Per entity: information by source, information gained from
  the pixels in nats, retained fractions after each nuisance, tension, and verdict.

## 8. Existing code and what must be built

*Exists:* a depth-only surfel fit whose default path is its own ray-plane compositor; the closed-form
row $\mathbf j_{uk}$, lumped depth blocks, fusion, leave-out by subtraction and the support rules; a
reference renderer exposing weights, depths and kernels per pixel and surfel; a separate photometric
trainer on its own primitives, not connected to any of this.

*To build:* storage of blocks per source and block of frames; projective appearance with source and
judging frames; the sensitivity $\mathbf g_{uk}$ in the reference compositor, all of whose factors
are present or one accumulation away; entity blocks along three directions by finite differences of a
group move; the reduced system in poses and per-frame depth parameters; the saturated depth
information and its covariances; lifting of masks and features; profile and injection as group moves.

*Cost, as estimates from counts:* storage is nine numbers per primitive and source. Entity blocks
need six extra renders per judging frame. The reduced nuisance system has eight parameters per frame
and is banded. A profile is a few tens of renders per window at reduced resolution. All are small
against a fit. The GPU rasteriser was not available where this note was written; whether it renders
feature channels in its 2DGS path or exposes per-fragment quantities is unchecked.

## 9. Occupied ground, as far as checked for this note

Searched and read only in part. Fisher information of splat parameters for view selection and
uncertainty exists (FisherRF, arXiv 2311.17874; abstract read). Second-order optimisation of splats
with matrix-free products exists (arXiv 2409.12892, 2504.12905; abstracts read). Selective
monocular-depth supervision in splatting exists (arXiv 2604.05715, method read: a mask from
inconsistency of rendered depth, no test; arXiv 2607.02554 and 2503.18363, abstracts read). Seen by
title only in a search for the phrase: information-driven exploration with splats, next-best-view
with optimality criteria on splats, and an uncertainty-estimation method for splats. No work was found
under the exact name "information-theoretic Gaussian splatting" in that one search, which does not
establish that the name is free. Recalled and unchecked: Gaussian-map SLAM with covariance, pose-free
and pose-refining splatting, direct and photometric bundle adjustment, large-scale aerial splatting.

What this note would defend as not duplicated, subject to that reading: information stored *by
provenance* on a placement coordinate shared by a learned prior and the pixels; the saturation of the
prior's information by its repeated error; an alignment measure with a stated null; and an explicit
account of which nuisances are marginalised at what cost and which must be detected. A claim of having
introduced uncertainty or Fisher information to splatting would be false.

## 10. What would defeat it, cheapest test first; and what is open

Exact depth from a synthetic scene is used for evaluation only.

1. *Is the stiffness honest?* Place surfels from exact depth, move entities by known amounts, and
   compare the scatter of recovered placements with the stored image information after dispersion.
   If the ratio varies by an order of magnitude across materials, the information field is not a
   usable weight and nothing built on it holds. Needs a renderer and exact depth, no depth network.
2. *Is the prior inside the capture range?* The distribution of residual parallax, in footprints, of
   the aligned monocular surface against exact depth, by material. If most of the scene is outside,
   the method is a sweep everywhere and loses its cost argument.
3. *Does saturation matter?* Fusion with nominal against honest depth blocks, on the harm-weighted
   error. If they agree, the provenance machinery buys nothing for fusion and remains only a test.
4. *What do the nuisances cost?* Retained fractions after pose and per-frame depth parameters, per
   entity. If frame-wide marginalisation leaves little information for entities that fill a large
   part of a frame (a lake, a large roof), those are unjudged for a new reason.
5. *Is the null usable?* The distribution of $T_m$ with exact depth as the depth source.

Open.

- Entities with no image information and a faulty prior are not detected by anything in the
  representation.
- The entity model is three numbers; errors of shape are outside it.
- Visibility, entity membership and ownership are computed on the geometry under test.
- The dispersion corrects scale, not bias, of the image family.
- Marginalising pose removes frame-wide patterns; a fault of that extent is invisible or appears as a
  pose correction.
- Per-primitive image curvature at scale needs stochastic probes or changes inside the rasteriser;
  the entity-level blocks do not.
- Not verified: every prediction above; the rasteriser's capabilities; the prior work listed as
  recalled or seen by title.
