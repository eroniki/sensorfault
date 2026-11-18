# Listen to the Depth Prior or to the Pixels? A Rendering-Based Consistency Test on Gaussian Surfels

Supporting derivations are in `docs/learned-indicator/`: the statistical theory (`theory.md`), the
definition of the evidence (`cues.md`), the surfel-space formulation (`surfel.md`), objections and
prior work (`objections-and-prior-work.md`), and the experiment design (`experiments.md`).

## Abstract

Gaussian surfel reconstruction from posed aerial video is commonly regularised by one monocular depth
map per frame. These maps fail on whole surfaces and repeat the failure in every frame, which hides
it from any gate built on agreement between depth maps. We place the depth prior and the images on
the same primitives and test them against each other through the differentiable renderer. For a
planar surfel, the Jacobian of the rendered image with respect to the surfel's placement factors
through the Jacobian of its depth: both kinds of evidence are quadratic forms on the same three
coordinates, offset and two tilts, and differ by a scalar weight per pixel. The unit of the test is a
surface entity, a set of surfels onto which per-frame segment masks have been fused. The total misfit
of an entity splits exactly into disagreement among depth maps, disagreement among images, and a
tension between the two families. A fault that every frame repeats is, exactly, a between-family
effect, which is why no reweighting of depth maps against each other can see it. The tension is
computed as a score test at the surface placed by depth alone, with appearance projected from source
frames and judged on disjoint frames, so that nothing in the model can explain a misplacement away.
Outline parallax needs no interior texture and arises from rendering the entity's label, with
boundary ownership decided by occlusion in the rasteriser; outline against interior identifies
reflecting and transparent surfaces. Moving an entity's surfels by a known amount and re-rendering
calibrates every instrument, including a frozen video encoder, and measures how honest the image
evidence is where it claims to be informative. Light heads on frozen features say which surfaces make
the depth model fail and which mislead the pixels, and carry the verdict to entities the pixels
cannot judge. Nothing in this document has been measured; a staged test with thresholds fixed in
advance decides whether the approach holds.

## 1. Introduction

A static scene is observed by a posed aerial video flown in strips. An off-the-shelf monocular model
gives one depth map per frame, and a surface of 2D Gaussian surfels is fitted with those maps as
supervision. Most depth pixels are accurate up to a per-frame scale and offset. A minority, grouped in
regions that follow objects and materials (water, glass, texture-free roofs), are wrong by many
standard deviations.

The usual defence is consistency between depth maps. Every such *consensus gate* shares a blind spot:
if the frames that observe a surface make the same error, their residuals against each other are
small and the error is fused. Images are the independent evidence, and in splatting they are already
present as the rendering loss. Current practice combines the two by a fixed or decaying weight on the
depth term, or by masks derived from one of them. That leaves the question this work is about:

*For each part of the surface, when should the primitives listen to the depth prior, and when to the
pixels?*

The answer by default is: in proportion to information, direction by direction. That default is wrong
in two cases, when the depth prior is faulty and when the pixels are misled (a reflection, a moving
object), and it must be changed there and only there. This requires three things that the fusion of
two loss terms does not provide: the information each family really has about a surface, a test of
whether they are compatible, and an attribution of which one is wrong.

The differentiable renderer supplies all three on the primitives themselves. It also gives the
primitives a definite role. A per-frame pipeline that warps images through depth maps can test one
frame at a time; it cannot hold both families as forms on the same coordinates with occlusion
resolved, fuse segments and features in the same place, or move a surface and see the consequence in
every view.

**Claim.** (i) On a surface entity the misfit decomposes into within-depth, within-image and
between-family parts, and a fault repeated by every frame appears only in the last. (ii) The
between-family tension, computed in a model where appearance cannot absorb misplacement and
calibrated by injection, is a test with a usable null and an honest error bar, including on
untextured entities through their outline. (iii) Gating placement on the attributed verdict, with
heads on frozen features for entities the pixels cannot judge, improves the surface where
information-weighted fusion, a decaying depth weight, a consensus gate and a robust loss do not.

**Scope of the learning.** Self-supervised on frozen models: a frozen segmenter, frozen dense image
features, a frozen video encoder and the frozen monocular depth model. The test is a statistic. What
is learned are light heads on frozen features and a few dispersion parameters per appearance stratum.
No fault labels are needed for training; a small labelled set from a scene with exact depth is
optional, for testing assumptions and for calibration on entities the pixels cannot judge.

**Contributions.**

1. The factorisation of the image Jacobian through the depth Jacobian for planar surfels, which makes
   depth and image evidence comparable on three placement coordinates per entity (Section 3).
2. The exact decomposition of entity misfit, identifying the repeated fault with the between-family
   term, and the repetition coefficient estimable without ground truth (Section 4.2).
3. A test that cannot be explained away: a score test at the depth-placed surface with projective
   appearance and disjoint judging frames, and the retained-information fraction as the measure of
   what any appearance model leaves of the test (Section 4.3).
4. Outline parallax as the label channel of the same renderer, with ownership by occlusion, and the
   outline-against-interior test for a second layer (Section 4.4).
5. Calibration by injection through the renderer: capture range, honesty of stated information, and
   the fraction of a fault that appearance absorbs, as measured quantities (Section 4.5).
6. Attribution with two gates, acting on placement and keeping the monocular shape, and the transfer
   to pixel-blind entities with an explicit unsupported state (Sections 4.6 to 4.8).
7. A staged decision protocol that includes the outcomes "fusion already answers the question" and
   "the prior is not worth keeping" (Section 5).

## 2. Related Work

Statements about specific papers are from reading where marked *(read)* in the supporting notes and
otherwise recalled; the search was not exhaustive.

**Depth-regularised splatting and neural surfaces.** Monocular depth with scale alignment is a
standard regulariser, usually with a fixed or decaying weight. Recent work makes the supervision
selective: masks from a virtual-stereo check on rendered depth, masks from the photometric error,
heuristic confidence sums, and uncertainty from the consistency of monocular depth across views. These
are the nearest systems. None of those read tests the prior against the pixels with an error bar,
separates the repeated part of a fault, or attributes a disagreement.

**Information and uncertainty in splatting.** Per-Gaussian Fisher information from the rendering loss
is used for view selection, pruning and uncertainty; second-order optimisers build Gauss-Newton
systems on Gaussians. They provide the machinery for image information on primitives. They use the
diagonal or the per-primitive block, for the image loss alone.

**Lifting segments and features onto Gaussians.** Segment identities and frozen features are fused
onto primitives by rendering weights, in closed form or by optimisation. Entities and features here
are lifted in that way.

**Fault detection by redundancy.** The tension between two independent estimates of the same quantity
is solution separation in navigation integrity and a Hausman-type specification test in statistics.
The statistic is not new; its construction for depth against pixels on surfels, its null and its
attribution are the work.

**Plane plus parallax, layered motion and feature-metric reprojection.** Stabilising by a reference
geometry and reading the residual motion is classical, as are motion layers and losses that compare
warped and real images or features. The stabilised clip of one frame is the single-message special
case of the test.

**Video foundation models.** Joint-embedding predictive video encoders provide spatiotemporal tokens;
promptable segmenters provide regions. No released video predictor is conditioned on camera pose, and
the segmenters have no promptless video mode, so the expectation is formed by rendering and entities
by lifting.

## 3. Two Messages on One Primitive

### 3.1 Depth message

Let $\boldsymbol\theta_k = (\nu, \tau_1, \tau_2)$ be the placement of surfel $k$: offset along its
normal and two tilts. The depth at which the ray of pixel $u$ meets the plane of $k$ is $d_k$, with

```math
\mathbf j_{uk} = \frac{\partial d_k}{\partial \boldsymbol\theta_k} = \frac{\bigl[\,1,\ -\mathbf t_1^\top(\mathbf x_{uk} - \boldsymbol\mu_k),\ -\mathbf t_2^\top(\mathbf x_{uk} - \boldsymbol\mu_k)\bigr]}{\mathbf n^\top \mathbf w_u} ,
```

for ray direction $\mathbf w_u$, intersection $\mathbf x_{uk}$, tangents $\mathbf t_1, \mathbf t_2$ and
normal $\mathbf n$. The rendered depth is $\hat D_i(u) = \sum_k w_{uk} d_k$ with compositing weights
$w_{uk}$. The monocular depth is modelled as
$\hat d_i(u) = a_i(d_i(u) + \beta d_i(u)^2) + b_i + \varepsilon_i(u)$ with per-frame scale and offset,
a shared distance bias and nominal noise $\sigma_i(u)$. Frame $i$ sends surfel $k$

```math
\Lambda^D_{ik} = \sum_u w_{uk}\, \frac{a_i^2\, \mathbf j_{uk}^\top \mathbf j_{uk}}{\sigma_i^2(u)}, \qquad
\boldsymbol\eta^D_{ik} = \sum_u w_{uk}\, \frac{a_i\, \mathbf j_{uk}^\top e^D_i(u)}{\sigma_i^2(u)} ,
```

with $e^D_i$ the depth residual. The information uses $w_{uk}$ and not its square: the lumped
block-diagonal matrix majorises the Hessian of the frame's loss, so a step cannot overshoot, and
leaving out a block of frames is a subtraction.

### 3.2 Image message

The rendered appearance is $\hat{\mathbf c}_i(u) = \sum_k w_{uk} \mathbf a_k$ with
$w_{uk} = \alpha_k G_k(\mathbf s_{uk}) T_{uk}$, Gaussian kernel $G_k$, transmittance $T_{uk}$ and
$\mathbf s_{uk}$ the intersection in the surfel's tangent frame in units of its scales. Two steps give
the derivative with respect to placement.

*Contrast.* A change of the kernel of $k$ changes its own weight and, through the transmittance, the
weights of everything behind it:
$\partial \hat{\mathbf c}_i(u) / \partial \log G_k = w_{uk}(\mathbf a_k - \mathbf b_{uk})$, where
$\mathbf b_{uk}$ is the appearance composited behind $k$. A primitive matters to the image only through
its contrast with what lies behind it.

*Slide.* Moving the plane along the ray by $\partial d_k$ moves the intersection in the tangent frame
by $\mathbf m_{uk} = [\mathbf t_1^\top \mathbf w_u / \sigma_1,\ \mathbf t_2^\top \mathbf w_u / \sigma_2]^\top$,
the tangential component of the viewing ray. Seen along its normal a surfel does not slide; the
difference of slides between two frames is the parallax.

Together,

```math
\frac{\partial \hat{\mathbf c}_i(u)}{\partial \boldsymbol\theta_k} = \mathbf g_{uk}\, \mathbf j_{uk}, \qquad
\mathbf g_{uk} = - w_{uk}\, \bigl( \mathbf a_k - \mathbf b_{uk} \bigr)\, \bigl( \mathbf s_{uk}^\top \mathbf m_{uk} \bigr) .
```

**The image Jacobian with respect to placement factors through the depth Jacobian.** Both families act
on the same three coordinates through the same geometric row $\mathbf j_{uk}$ and differ by a scalar
weight per pixel and primitive: $w_{uk} a_i^2 / \sigma_i^2$ for depth, $\lVert \mathbf g_{uk} \rVert^2 / \sigma_I^2$
for the image. This holds because a surfel is a plane, and it is the precise sense in which the
primitive is where the two meet. The pull of the pixels on every primitive is one backward pass of the
image loss per frame, with visibility and occlusion supplied by the rasteriser.

### 3.3 The unit is the entity

The per-primitive image block is not a parallax error bar. A single surfel moved alone tears the
surface, and its message measures contrast with whatever shows through the gap: information about
coverage. One view gives no offset information at all, since offset is confounded with a tangential
shift of the texture. And a fault moves a surface, not a surfel.

Let an *entity* $m$ be a set of surfels $\mathcal K_m$ (Section 4.1) with three placement parameters
$\boldsymbol\theta_m$ that move its surfels coherently through $P_{km}$. Its image Jacobian and
information are

```math
J^I_{im}(u) = \sum_{k \in \mathcal K_m} \mathbf g_{uk}\, \mathbf j_{uk}\, P_{km}, \qquad
\Lambda^I_{im} = \sum_u \frac{J^I_{im}(u)^\top J^I_{im}(u)}{\sigma_I^2} .
```

Inside the entity the tears of neighbouring surfels cancel and what remains is texture gradient times
slide: a blank interior carries no information. At the outline the contrast with the surface behind
does not cancel. The entity block contains the cross terms between its surfels and is computed along
the three entity directions only, by Jacobian-vector products or by finite differences of a group
move. Its size is texture energy times the spread of viewing directions.

The working hypothesis, stated for the error of the prior and not for the surface, is that monocular
error on an entity is low-order: an offset and two tilts, with a remainder. The surface itself may be
curved. Whether this holds is the first thing to measure.

## 4. Method

### 4.1 Entities, features and labels on the primitives

Per-frame masks of a frozen segmenter are fused onto surfels by accumulated compositing weights,
$A_{k\ell} = \sum_{i,u} w_{uk} \mathbb 1[\mathrm{mask}_i(u) = \ell]$; identities of different frames
are merged when they claim the same surfels. No 2D tracking of masks is needed, which matters because
available segmenters do not segment video without prompts. Frozen dense features are lifted by the
same weights, giving one feature per primitive and, pooled, per entity. Nested masks give a tree of
entities. Lifting uses the geometry under test; under a placement fault masks land on displaced
surfels by the residual parallax, which is small against all but the smallest entities and is itself
the signal of Section 4.4.

### 4.2 Tension, and what each family sees

Linearised at the current state, each family alone would place entity $m$ at
$\hat{\boldsymbol\theta}^D_m = (\Lambda^D_m)^{-1} \boldsymbol\eta^D_m$ and
$\hat{\boldsymbol\theta}^I_m = (\Lambda^I_m)^{-1} \boldsymbol\eta^I_m$. The **tension** is their
difference in units of their joint uncertainty,

```math
\mathbf t_m = \hat{\boldsymbol\theta}^D_m - \hat{\boldsymbol\theta}^I_m, \qquad
T_m = \mathbf t_m^\top \bigl( (\Lambda^D_m)^{-1} + \hat\varphi\, (\Lambda^I_m)^{-1} \bigr)^{-1} \mathbf t_m ,
```

$\chi^2_3$ under the null if both families are unbiased with independent errors; $\hat\varphi$ is the
dispersion of the image family (below). $T_m$ is the increase in misfit caused by forcing the two
families to share one placement. The pulls alone are not a statistic: at any joint optimum they are
equal and opposite by construction. The vector $\mathbf t_m$ is an offset fault and two tilt faults in
metric units.

With per-frame messages the total misfit of an entity about a common placement splits exactly:

```math
\chi^2_{\mathrm{total}} = \underbrace{\sum_j \bigl\lVert \hat{\boldsymbol\theta}^D_{mj} - \hat{\boldsymbol\theta}^D_m \bigr\rVert^2_{\Lambda^D_{mj}}}_{\text{within depth: consensus}}
+ \underbrace{\sum_i \bigl\lVert \hat{\boldsymbol\theta}^I_{mi} - \hat{\boldsymbol\theta}^I_m \bigr\rVert^2_{\Lambda^I_{mi}}}_{\text{within image: ancillary}}
+ \underbrace{T_m}_{\text{between families}} .
```

Write the monocular placement error of frame $j$ on entity $m$ as
$\boldsymbol\mu_m + \boldsymbol\alpha_{mj}$: a surface term every frame repeats and a per-frame term.

- The within-depth term contains only the $\boldsymbol\alpha_{mj}$. It is what a consensus gate tests,
  and the surface term cancels in it.
- The between term contains $\boldsymbol\mu_m$ plus the average of the $\boldsymbol\alpha$. It is the
  only place the repeated fault appears.
- The within-image term contains neither. It is ancillary for the monocular error.

**A fault that every frame repeats is exactly a between-family effect.** The repetition coefficient
$\varrho = \operatorname{tr}\Sigma_\mu / (\operatorname{tr}\Sigma_\mu + \operatorname{tr}\Sigma_\alpha)$,
the share of monocular error variance in the surface term, is estimated from the first and third
components without ground truth, as a function of the left-out block. A plateau above zero is what no
consensus method can see.

**Null distribution.** Image residuals at a fit are model misfit, not sensor noise, and are spatially
correlated, so the nominal image information is far too large. The within-image term supplies the
correction: $\hat\varphi = \chi^2_{\text{within image}} / (3(n_m - 1))$ pooled per appearance stratum,
a sandwich variance with frames as independent units. Two further calibrations need no labels: on
entities with strong image information the bulk of $T_m$ must follow $\chi^2_3$, and *placebo
tensions* between split halves of one family (depth against depth, image against image) give the null
scale directly. Pose error moves all entities of a frame together and is removed as a low-order field
per frame.

### 4.3 A test that cannot be explained away

A joint fit of geometry and appearance does not have to sit at a compromise. It can relieve tension
with degrees of freedom that are not placement: view-dependent colour that paints the parallax,
semi-transparent duplicates, floaters, lowered opacity. At convergence both pulls are small and the
fault is still there. Tension at an unrestricted joint fit is not a test.

**Score test at the depth-placed surface.** Fit the geometry to the depth messages alone. There
$\boldsymbol\eta^D_m = 0$ and

```math
T_m = \boldsymbol\eta^{I\top}_m\, (\Lambda^I_m)^{-1} \bigl( (\Lambda^D_m)^{-1} + \hat\varphi\,(\Lambda^I_m)^{-1} \bigr)^{-1} (\Lambda^I_m)^{-1}\, \boldsymbol\eta^I_m :
```

the pull of the pixels on a surface the prior placed. The geometry has never seen an image, so there
is nothing for it to have explained away. A surface fused from all frames' depth carries the surface
term and only the average of the per-frame terms, so this is directly the test of the repeated fault.

**Projective appearance.** The test model has no free appearance. Each surfel is textured from
*source* frames by projection at its current placement,
$\mathbf a_k(\boldsymbol\theta) = I_r(\pi_r(\mathbf x_k(\boldsymbol\theta)))$, and disjoint *judging*
frames form the residual. The Jacobian of a judging frame then carries the difference of slides
between judging and source view, the parallax between them, times the texture gradient. Source and
judging frames are interleaved folds, so the judging baseline is not shortened, and every frame
judges once. Rendering one frame's depth map into its neighbours, the depth-stabilised clip, is the
special case with a single depth message; rendering adds the fused geometry and occlusion decided by
the rasteriser.

**Retained information.** If appearance is fitted instead, its freedom is accounted for by the Schur
complement,

```math
\Lambda^{I,\mathrm{prof}}_m = \Lambda_{\theta\theta} - \Lambda_{\theta a} \Lambda_{aa}^{-1} \Lambda_{a\theta}, \qquad
\kappa_m = \frac{\operatorname{tr} \Lambda^{I,\mathrm{prof}}_m}{\operatorname{tr} \Lambda_{\theta\theta}} \in [0, 1] ,
```

the fraction of placement information that survives the freedom of appearance. One diffuse colour per
surfel seen by many frames keeps it near one; view-dependent colour in a strip, where viewing
direction and parallax vary together, leaves almost nothing. "Appearance can hide wrong geometry"
becomes a number per entity.

**Range.** Gradient and curvature are valid within about a surfel footprint of residual parallax. A
depth fault of a few percent at survey spacing shifts content by many pixels, where the pull is weak
and its direction arbitrary. The score test is therefore a screen; the estimate for gross faults is a
profile over the three placement parameters by moving the entity and re-rendering (Section 4.5),
coarse to fine in image scale.

### 4.4 Channels, outline parallax and the layer test

The image message is defined for any rendered channel, each with its own dispersion: *colour*, fine
but sensitive to exposure and specularity; *frozen features*, reduced to a few tens of channels,
tolerant to exposure and sensitive to weak large-scale structure; and *labels*, the entity identity
rendered and compared with each frame's own mask.

**Outline parallax is the label channel.** There the contrast is non-zero only where an entity ends in
front of another, and the message measures the misalignment between rendered and segmented outline in
every frame, with three parameters per entity and no need for interior texture. A lake has nothing to
match inside, but its shoreline moves. Ownership needs no rule: moving the farther entity does not
move an occlusion edge, because its surface continues behind the occluder, so its Jacobian there is
zero and only the nearer entity receives the message. Limits: outline segments parallel to the flight
line carry nothing; the depth order must be right in the model; a curved occluder's rim is not
attached to the surface; and the segmenter must draw the outline at the same physical place in every
view, which is testable on entities that pass the tension test.

**Layer test.** Split the image message of an entity into outline and interior. Their difference, a
three-degree statistic, is zero for an opaque diffuse surface. An outline that agrees with the depth
placement and an interior that asks for a placement *behind* is a second layer: a reflection or a view
through glass. This addresses the principal failure of image evidence, that the mirror image of a
static scene in still water or flat glass is a rigid scene and triangulates consistently at the wrong
depth. A reflector with no attached outline in view, and an object moving along the strip at constant
speed, remain image-consistent and unresolved: they are to the image family what the repeated fault
is to the depth family.

### 4.5 Injection through the renderer

Move the surfels of an entity by a known $\boldsymbol\delta$ and re-render every frame. With
projective appearance, any statistic $S$ of rendered against real frames satisfies the exchange
identity

```math
S(\boldsymbol\delta) = \mathcal S\bigl( \boldsymbol\epsilon_m + \boldsymbol\delta \bigr) :
```

the response function translated by the unknown placement error, with occlusion changes included.

- *Estimator.* The minimiser over $\boldsymbol\delta$ estimates the placement error in metric units
  for any black-box statistic, including the distance between frozen video-encoder tokens of rendered
  and real clips.
- *Information.* The curvature at the minimum is the instrument's own sensitivity on this entity; zero
  curvature is abstention by measurement.
- *Honesty.* On surfaces where the image family has no real information it may still report some: a
  photometric fit places open water metres from its true plane with confidence. The ratio of nominal
  information to the information realised under injection measures this, per stratum, and corrects it.
- *Absorption.* The fraction of an injected fault that an appearance model absorbs is measured for
  each appearance regime.
- *Supervision.* Injection responses are free labels for a readout head on encoder tokens and for a
  head that predicts where stated image information can be trusted.

A frozen video encoder compared on rendered and real clips is an instrument, not new information: it
is as blind as any matcher where the image has no structure. It is admitted only if its calibrated
estimate agrees with the photometric and feature channels where both are informative, and it is of
interest where it has a larger capture range.

### 4.6 Attribution and when to listen

Tension is symmetric. Three hypotheses per entity:

| Hypothesis | Tension | Within-image scatter | Outline against interior |
| --- | --- | --- | --- |
| both right | nominal | nominal | agree |
| depth placement wrong | large, either sign | nominal | agree with each other, both against depth |
| pixels misled (second layer, mover) | large; for a reflection the offset is behind | often inflated | outline agrees with depth, interior does not |

The verdict is the posterior over the three, with priors $\pi^D(\boldsymbol\phi_m)$ and
$\pi^I(\boldsymbol\phi_m)$ from heads on frozen features and the likelihoods of the three statistics.
It yields two gates: $1 - \zeta^D_m$ on the placement component of the depth messages and
$1 - \zeta^I_m$ on the image messages of the entity.

Let $\vartheta_1, \vartheta_2, \vartheta_3$ be the generalised eigenvalues of
$(\Lambda^I_m, \Lambda^D_m)$. Along each direction the fused placement weights the pixels by
$\vartheta / (1 + \vartheta)$; a depth fault $\Delta$ moves the fused surface by its harm
$\Delta / (1 + \vartheta)$ and is detectable with noncentrality
$\Delta^2 \Lambda^D \vartheta / (1 + \vartheta)$.

| Regime | Rule |
| --- | --- |
| $\vartheta \gg 1$, pixels dominate | listen to pixels; the prior is nearly irrelevant in this direction and its verdict is a training label |
| $\vartheta \approx 1$ | fuse by information; test; on tension, attribute and gate |
| $\vartheta \ll 1$, pixels blind | listen to depth, with the prior head as the only check; unsupported if the head has no support |

An entity can be pixel-dominated in offset and pixel-blind in a tilt, and the rule applies per
direction. For a fault of given harm, detectability rises with the pixels' share and vanishes with it,
so the indicator's own contribution is confined to the second and third rows, and evaluation is
weighted by harm.

### 4.7 What is learned, and transfer to pixel-blind entities

No backbone is trained. The learned parts are:

1. **Prior heads** $\pi^D(\boldsymbol\phi_m)$ and $\pi^I(\boldsymbol\phi_m)$: which surfaces make the
   depth model fail and which mislead the pixels. Inputs are frozen dense image features and
   video-encoder tokens of the *raw* clip pooled over the entity, never the residuals. They are fitted
   by the marginal likelihood of the tension statistics, which is learning from labels whose noise is
   known per entity through the informations; an entity contributes in proportion to its label
   information. They are what reaches pixel-blind entities.
2. **Support head** $e(\boldsymbol\phi)$: whether an entity of this appearance ever has image
   information. It needs no fault labels and separates interpolation from extrapolation. In the latter
   the output is *unsupported*, not a probability.
3. **Dispersions** per stratum and channel, and the honesty correction.
4. **Readout by injection** for the video encoder (optional).

Transfer rests on two assumptions: among entities that look alike, those the pixels can judge fail as
often as those they cannot, and every appearance the head is applied to is sometimes judged. The first
fails when judgeable parts of an unjudgeable material are atypical. Two label-free checks exist.
*Counterfactual blindness:* judged entities are made to play blind ones with their verdict kept, by
withholding feature clusters, shortening the window, hiding the outline or blurring the interior,
which gives accuracy against distance from the training support. *Consensus as examiner:* the
within-depth term exists without texture, so among pixel-blind entities its outlier rate must rise
with the head's score. What stays beyond every label-free check is a surface-term fault on a
pixel-blind entity.

Material strata can come from noun-phrase prompts of a promptable segmenter. As covariates they cost
only efficiency if wrong; where they switch the model they must be confirmed by the layer test.

### 4.8 Use in reconstruction

1. **Placement gate.** The depth message of an entity is split into its component along the three
   placement directions and the remainder. The gate acts on the placement component only, so a faulty
   entity keeps the shape of its monocular depth and takes its placement from the pixels, or, if
   pixel-blind, is left free in placement and reported as such. Detection and repair are one
   operation.
2. **Image gate.** The interior image messages of an entity attributed to a second layer are removed;
   its outline messages are kept.
3. **Nuisance parameters.** Per-frame scale and offset and the distance bias are estimated with the
   gates applied, against image-witnessed placement where the pixels dominate, so that no nuisance
   parameter absorbs a fault.
4. **The reconstruction is a separate fit.** After the verdict, geometry and appearance are fitted to
   both families under the gates, with the indicator held fixed. The test model with projective
   appearance is not the final model.
5. **Fits compared.** Pixels only; depth only; both with a fixed weight; both with a decaying depth
   weight; both by information without gates; both with gates.

### 4.9 Why primitives

The primitives have a job if, and only if, these are used: both families as quadratic forms on the
same placement coordinates with occlusion resolved, so that information ratios, tension and gates are
defined per direction; the factorisation of Section 3.2; entities, features and labels fused in one
place by the same weights; and injection as a move of a group with correct occlusion. Any explicit
surface with a differentiable renderer would serve. The planar surfel is the one for which placement
is three numbers and the factorisation is exact.

### 4.10 What exists and what must be built

*Exists in the code:* a depth-only surfel fit with placeholder colour and its own ray-plane depth; a
reference renderer giving per-pixel, per-surfel weights and depth Jacobians, with the lumped depth
information, the cavity render and its variance; a frame-level gate with leave-block-out by
subtraction; a separate photometric trainer with view-dependent colour and densification, not
connected to the depth fit or the gate; fault injectors on depth maps.

*To build:* projective texturing and a linear diffuse fit on the depth-placed state; the image pull
and entity information along three directions; lifting of masks and features; the misfit
decomposition, dispersion, attribution and gates; injection as a group move.

*Not verified:* the rasteriser's interface for per-pixel weights, feature channels and forward-mode
products; memory and speed of the foundation models; their behaviour on nadir aerial imagery.

## 5. Experiments

### 5.1 Evaluation targets

On data with exact depth: the true placement error of every entity per frame and its split into
surface and per-frame terms; the true-fault mask; surface classes; the information ratio $\vartheta$
per entity and direction; and true second-layer entities (water, glazing).

### 5.2 The deciding measurement, in stages

**Stage 0, on a CPU.** A synthetic entity with known offset and tilt on the reference renderer: does
the image message recover the injected placement, and how large and how stable is its variance
inflation? On exact depth of a synthetic aerial block, with single-image masks: how much monocular
fault energy is an offset and tilt per mask, against tiles of equal count; and the variance components
$\Sigma_\mu$, $\Sigma_\alpha$ and $\varrho(B)$ from true errors.

**Stage A: is the image message a measurement?** On an existing photometric model and the
depth-placed surface, by injection: capture range per image scale and channel; the honesty ratio per
surface class; the fraction of an injected fault absorbed under four appearance regimes (projective;
diffuse fitted; view-dependent; view-dependent with densification).

**Stage B: the tension against real monocular faults.** Null calibration from placebo tensions;
detection of true placement faults, overall and for the surface term; attribution on water and
glazing; the gated fit against the alternatives of Section 4.8; transfer of the prior heads to
pixel-blind entities.

**The approach is abandoned if any of:**

- K1, wrong unit: masks explain no more fault energy than tiles of equal count.
- K2, nothing repeated: $\varrho(B)$ falls to zero within a few frames.
- K3, not a measurement: on textured land no scale or channel recovers injected faults of realistic
  size; or on blank and water classes stated information exceeds realised information by more than a
  factor of three for every estimator and no correction is stable.
- K4, always explained away: no appearance regime absorbs less than half of an injected fault while
  rendering healthy surface nearly as well as the unrestricted model.
- K5, no usable null: healthy entities have a tension whose robust scale is far from nominal after
  the label-free corrections.
- K6, rendering adds nothing: the tension matches a hand-warped per-frame parallax gap.
- K7, fusion already answers the question: the gated fit does not beat information-weighted fusion
  without gates, or a decaying depth weight, on the harm-weighted metric.
- K8, not better than existing selective supervision on repeated faults.
- K9, no transfer, or the prior is not worth keeping: on pixel-blind entities the heads do not exceed
  the per-stratum fault rate, or reconstruction without the prior does as well as with a perfect gate.

Thresholds for each are fixed in the experiment design before measurement.

### 5.3 Baselines

No depth term; fixed depth weight; decaying depth weight; robust depth loss; removal of whole frames;
the consensus gate; the depth model's own confidence; disagreement between several depth models and
instability under input transformations; masks from a virtual-stereo check, from photometric error and
from cross-view depth consistency; heuristic confidence sums; per-Gaussian Fisher weighting;
information-weighted fusion without a test; the hand-warped parallax gap per frame; the per-stratum
fault rate; a linear probe on frozen features at equal label budget; multi-view stereo and a
feed-forward multi-view model without any monocular prior. Bounds: an oracle gate with true masks.

### 5.4 Protocol

Evaluation quantities are written to a store only the scoring code reads. Thresholds, strata and
appearance regime are chosen by held-out likelihood, by injection, or on labelled scenes disjoint from
evaluation. Splits are by ground tile or scene; heads are cross-fitted by ground tile. Intervals are
cluster-bootstrap intervals over ground tiles with the number of entities beside each. A rendered
scene has perfect poses and synthetic materials: images are degraded, poses perturbed, and verdicts
repeated on real scenes, which decide where the two disagree.

## 6. Limitations

- **The image null is the weakest point.** Rendering residuals are model misfit. If their dispersion
  at surfel resolution swamps real fault sizes, the renderer cannot judge.
- **Pixel-blind entities** (blank interior and no owned outline in view) are judged by heads alone,
  and a surface-term fault on them is beyond every label-free check.
- **Image-consistent wrong geometry**, a reflector without attached outline or constant-speed motion
  along the strip, passes every image test.
- **The test is silent where pixels are silent**, so a decaying depth weight may match it; this is a
  stated kill criterion.
- **The entity may be the wrong unit**, and frozen segmenters may not form repeatable regions on
  water and glazing in nadir views.
- **Gross faults are outside the linear range** and need a profile by re-rendering.
- **Independence of the two families** is an assumption once pose error is removed; they share poses
  and the surfel discretisation.
- **The statistic is classical.** The contribution is its construction, null and attribution for
  depth against pixels on surfels, and overlap with selective depth supervision in splatting must be
  examined further.

## 7. Notation

| Symbol | Meaning |
| --- | --- |
| $\boldsymbol\theta_k = (\nu, \tau_1, \tau_2)$ | placement of surfel $k$: normal offset and two tilts |
| $w_{uk}$, $\mathbf j_{uk}$ | compositing weight; Jacobian of ray-plane depth with respect to placement |
| $\mathbf a_k$, $\mathbf b_{uk}$, $\mathbf m_{uk}$, $\mathbf g_{uk}$ | appearance of $k$; appearance behind $k$; slide; image weight |
| $\Lambda^D$, $\boldsymbol\eta^D$; $\Lambda^I$, $\boldsymbol\eta^I$ | depth and image information and pull |
| $m$, $\mathcal K_m$, $P_{km}$, $\boldsymbol\theta_m$ | entity, its surfels, coherent-move map, its placement |
| $\hat{\boldsymbol\theta}^D_m$, $\hat{\boldsymbol\theta}^I_m$, $\mathbf t_m$, $T_m$ | placements by each family, tension vector and statistic |
| $\boldsymbol\mu_m$, $\boldsymbol\alpha_{mj}$; $\Sigma_\mu$, $\Sigma_\alpha$; $\varrho$ | surface and per-frame terms; their covariances; repetition coefficient |
| $\hat\varphi$, $\kappa_m$ | image dispersion; retained-information fraction |
| $\boldsymbol\delta$, $S(\boldsymbol\delta)$ | injected placement change; statistic under injection |
| $\vartheta$ | generalised eigenvalue of image against depth information |
| $\zeta^D_m$, $\zeta^I_m$; $\pi^D$, $\pi^I$; $e(\boldsymbol\phi)$ | verdicts on depth and on pixels; prior heads; support head |
| $\boldsymbol\phi_m$ | frozen features pooled over the entity |
| $a_i, b_i, \beta$; $\sigma_i$, $\sigma_I$ | per-frame scale and offset, distance bias; depth and image noise |
