# Information-Theoretic Gaussian Splatting: a Map That Stores What It Knows, for Correspondence-Free Large-Area Mapping with Monocular Depth

Supporting derivations are in `docs/learned-indicator/`: the statistical theory (`theory.md`), the
definition of the evidence (`cues.md`), the surfel-space formulation (`surfel.md`), objections and
prior work (`objections-and-prior-work.md`), and the experiment design (`experiments.md`).

## Abstract

A map built from posed aerial video and one monocular depth map per frame looks complete, yet it can
be confidently wrong over whole surfaces, identically in every frame. Gaussian splatting cannot say
so, because it stores geometry and appearance and nothing about how well either is known. We add a
third quantity to the same primitives: information, the stiffness with which each source of evidence
holds a surface in place, kept separately for the depth prior and for the pixels. For a planar
Gaussian surfel the derivative of the rendered image with respect to the surfel's placement factors
through the derivative of its depth, so prior and pixels are quadratic forms on the same three
coordinates. Their disagreement is then a stored energy with units and a null distribution: the
alignment of geometry and appearance becomes measurable. Cross-view evidence enters only by rendering
the shared primitives into each frame and comparing; no features are matched, no tracks kept, no
points triangulated. The monocular prior brings every surface within the short capture range of that
comparison, and the pixels audit the prior in return. Stored per source, information splits the
misfit of a surface exactly into disagreement among depth maps, disagreement among images, and
disagreement between the two; a depth error that every frame repeats is the last term and is
invisible to any consensus among depth maps. Nuisance is given a place in the same model: pose,
per-frame depth scale and offset, and exposure are absorbed at a stated cost in information;
rendering misfit and ordinary prior error deflate the stated stiffness; regional faults and
reflections can be neither and are detected. The information is calibrated on the scene from frame
replicates and from displacements injected through the renderer. Light heads on frozen foundation
features speak only where the pixels have no stiffness, and the map says so. Nothing in this document
has been measured; the first experiment asks whether a depth-only map is overconfident by the amount
the theory predicts.

## 1. Introduction

### 1.1 The argument

Each step is a premise, its consequence, and what it means for the design. The equations follow in
Sections 3 and 4.

1. **A map is geometry, appearance and information.** Geometry and appearance describe the scene.
   Information describes what the data said about it. Gaussian splatting stores the first two.
   Opacity is not confidence: an opaque surfel on a blank roof is drawn with the same certainty as one
   on a textured wall, although the images hardly constrain the first. A large-area map that is used
   for measurement or change detection must say where the surface is determined, and by what.
2. **Pixels inform geometry only through a coupling.** An image says where a surface is only if
   moving the surface would change the image. That needs texture contrast and views from different
   directions. The strength of the coupling is a stiffness. It is zero on blank surface and zero from
   one viewpoint, and it can only be computed from the renderer, which is where geometry and
   appearance meet.
3. **The monocular depth is a second spring on the same surface.** It pulls each surface towards
   where the depth network put it. For a planar surfel both springs act on the same three
   coordinates, an offset and two tilts, and this is exact, not an analogy (Section 3.2).
4. **Two springs on one body store energy when they disagree.** The stretch is the misfit, in metres
   and radians. The energy is dimensionless, in nats, and has a null distribution. Alignment of
   geometry and appearance is therefore measurable. This is what the representation is for.
5. **The energy can only be read if the springs are kept apart.** A single summed confidence cannot
   say which source put it there. A fault that every depth map repeats is invisible in the
   disagreement among depth maps and visible only between depth and pixels. Information must be stored
   per source.
6. **Information must be honest.** A stiffness is a claim about scatter. Rendering residuals are
   model misfit, not sensor noise, and counting pixels as independent overstates stiffness by orders
   of magnitude. The same holds for the prior: frames that share an error do not average it out, so
   adding their information without limit lets the prior outvote the pixels. Both must be corrected
   on the scene itself.
7. **Evidence enters by rendering, not by matching.** Each frame is compared with a rendering of the
   shared primitives. Weak texture lowers stiffness smoothly instead of breaking a matcher, every
   pixel contributes, and occlusion is decided once. The price is a short capture range: the
   comparison works only if the surface is already nearly in place.
8. **The monocular prior is what makes step 7 possible.** It puts every surface within reach of the
   pixels, including where nothing could be matched. The pixels refine it where they are stiff and
   audit it where both are comparable. Each covers the other's weakness, and the information field
   says, direction by direction, which one is speaking.
9. **Nuisance has three fates.** A disturbance with a known low-dimensional form (a frame's pose
   error, a depth map's scale and offset, exposure) is *absorbed*: its directions are removed from the
   evidence and the map stops claiming knowledge in them. A disturbance with a level but no form
   (discretisation, resampling) is *deflated*: stiffness is reduced by a measured factor. A
   disturbance that is structured and local (a region of wrong depth, a reflection) can be neither and
   must be *detected*. Absorbing an error never makes it vanish; it converts it into a stated loss of
   information.
10. **What follows.** Store, per surface entity and per source, an information matrix and a pull on
    three placement coordinates, additive over frames and map tiles. Compute them from the
    differentiable renderer. Calibrate them. Read alignment as energy. Gate a source where the energy
    says the two disagree and the attribution says which is wrong. Use learned heads only where the
    pixels have no stiffness, and say so in the map.

### 1.2 Claim, scope, contributions

**Claim.** (i) A depth-only map of this kind is overconfident by an amount set by the share of the
monocular error that all frames repeat. (ii) Storing calibrated information per source on shared
Gaussian surfels makes geometry-appearance alignment a statistic with a usable null, and a repeated
fault exactly a between-source effect. (iii) A map built this way, by rendering alone, is better
calibrated and more accurate than depth-only, image-only and loss-weighted alternatives, and stays
calibrated under pose and prior errors, with its stated information falling as predicted.

**Scope of the learning.** Self-supervised on frozen models: the monocular depth model, a frozen
segmenter, frozen dense image features and a frozen video encoder. The representation and the test
are statistics computed through the renderer. What is learned are light heads on frozen features and
a few dispersion parameters per appearance stratum. No fault labels are needed; a small labelled set
from a scene with exact depth is optional, for checking assumptions and for calibration where the
pixels are blind.

**Contributions.**

1. A scene representation: Gaussian surfels carrying, per surface entity and per evidence source, a
   calibrated information record on placement, additive over frames and tiles (Section 3).
2. The factorisation that puts depth prior and pixels on the same coordinates, and the alignment
   measure built on it: stretch, energy, null (Sections 3.2 and 4.2).
3. The exact three-way split of misfit, the repetition coefficient, and the consequence that honest
   prior information saturates (Sections 4.2 and 4.6).
4. A reading of the energy that the model cannot hide: a score test at the depth-placed surface with
   projective appearance, and the retained-information fraction (Section 4.3).
5. A ledger of nuisance, absorbed, deflated or detected, each with its cost (Section 4.6).
6. Correspondence-free mapping stated exactly, with its gain and its price, including outline
   parallax from the rendered label and the layer test for reflections (Sections 4.4 and 4.7).
7. An evaluation by a proper score of the true surface under the map, and a staged protocol whose
   first stage is nearly free (Section 5).

**On the name.** The content is Fisher information in information form. Information-theoretic and
Fisher-information methods for view selection, pruning and exploration with Gaussian splats exist, as
do Bayesian and variational splatting; a short search found no method with this exact name, which is
weak evidence that it is free. The distinguishing content is information about placement, per entity,
per source, calibrated, and used to compare sources.

## 2. Related Work

Statements about specific papers are marked as read or recalled in the supporting notes; the search
was not exhaustive.

**Uncertainty and information in Gaussian splatting.** Per-Gaussian Fisher information from the
rendering loss is used for view selection, pruning and active mapping; Bayesian and variational
formulations keep a posterior over Gaussian parameters; SLAM systems with Gaussian maps carry
uncertainty for tracking and keyframe selection. These keep one posterior or one score per primitive,
for the image loss. None found keeps information per evidence source in order to compare sources, or
calibrates it against realised scatter.

**Depth-regularised splatting and selective depth supervision.** Monocular depth with scale alignment
is a standard regulariser with a fixed or decaying weight. Recent work masks or weights it by a
virtual-stereo check on rendered depth, by photometric error, by heuristic confidence, or by
cross-view consistency of the depth itself. These are the nearest systems for the fault side. None of
those read tests the prior against the pixels with an error bar or separates the repeated part.

**Pose-free and pose-refining splatting; direct methods.** Splatting without structure-from-motion
poses already uses monocular depth to initialise, and direct methods in visual odometry align images
without features. "Correspondence-free" here is in their tradition and inherits their capture-range
limit. What differs is that the evidence is stored in information form on a shared map and that the
prior is audited.

**Large-scale aerial splatting.** Tiled and hierarchical systems reconstruct city-scale scenes from
structure-from-motion inputs. They are baselines for accuracy and completeness and do not store or
evaluate what the map knows.

**Fault detection by redundancy.** Data snooping, solution separation and specification tests compare
estimates of one quantity from different evidence. The energy between two sources is of this family;
the statistic is classical and its construction on surfels is the work.

**Lifting segments and features; video foundation models.** Segment identities and frozen features
are fused onto Gaussians by rendering weights. No released video predictor is conditioned on camera
pose and promptable segmenters have no promptless video mode, so expectations are formed by rendering
and entities by lifting.

## 3. The Representation

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

### 3.4 The record

Per entity $m$ and per source $s$ (the depth family; the colour, frozen-feature and label channels of
the image family; a weak regulariser) the map stores

```math
\bigl( \Lambda^s_m,\ \boldsymbol\eta^s_m,\ \hat\varphi^s \bigr), \qquad
\Lambda^s_m = \sum_i \Lambda^s_{im}, \quad \boldsymbol\eta^s_m = \sum_i \boldsymbol\eta^s_{im} ,
```

a symmetric $3 \times 3$ matrix, a 3-vector and a dispersion: ten numbers per source. Three properties
make this a representation for large areas.

- *Additive.* Frames add. Tiles add. Leaving out a block of frames is a subtraction. Two maps of the
  same ground merge by adding records, source by source.
- *Local.* An entity is coupled to its neighbours through shared pixels and to other entities only
  through shared per-frame nuisance. The structure is the sparse entity-by-frame pattern of bundle
  adjustment, and its size grows linearly with area at fixed overlap.
- *Separable.* Because sources are not summed, any source can be removed, reweighted or tested
  afterwards without revisiting the images.

Derived fields per entity: *knowledge*, $\tfrac12 \log\det \Lambda_m$ in nats; *what appearance says
about geometry*, the mutual information
$\tfrac12 \log\det(\mathbf I + (\Lambda^{D}_m + \Lambda^0_m)^{-1} \Lambda^I_m)$ in nats; *who is speaking*,
the generalised eigenvalues $\vartheta$ of $(\Lambda^I_m, \Lambda^D_m)$ per placement direction; and
*surprise*, the energy of Section 4.2.

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

**Units.** The stretch $\mathbf t_m$ is in metres of offset and radians of tilt; the stiffness in
their inverse squares; the energy $E_m = T_m / 2$ is dimensionless, in nats, and is the
log-likelihood ratio of "the two sources see one surface" against "they see two". Over $M$ tested
entities, $\sum_m T_m / (3M)$ is one number that should be one.

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

### 4.5 Honest information: injection through the renderer

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

### 4.6 Nuisance: absorbed, deflated or detected

A nuisance that spans a known low-dimensional subspace of the evidence is projected out, and the map
loses exactly the information in that subspace. For a nuisance block $\mathbf q$ with prior
information $\Sigma_q^{-1}$,

```math
\Lambda^{\mathrm{marg}}_{\theta\theta} = \Lambda_{\theta\theta} - \Lambda_{\theta q} \bigl( \Lambda_{qq} + \Sigma_q^{-1} \bigr)^{-1} \Lambda_{q\theta} ,
```

and the retained fraction $\operatorname{tr}\Lambda^{\mathrm{marg}} / \operatorname{tr}\Lambda$ is the
stated cost.

| Nuisance | Fate | What is lost or assumed |
| --- | --- | --- |
| Pose of a frame, with a prior from navigation or an earlier adjustment | absorbed in the image family | a pose error moves all entities of a frame together; projecting it out removes the image evidence about that common mode. Absolute placement is known no better than the poses; relative placement between entities is kept. An entity that fills the frame retains almost nothing |
| Scale and offset of a depth map, shared distance bias | absorbed in the depth family | the prior says nothing about a frame-wide affine error; only shape and relative placement within a frame remain |
| Exposure per frame; appearance | absorbed in the image family | cost measured by $\kappa_m$ |
| Rendering misfit: surfel resolution, resampling, residual view dependence | deflated by the dispersion $\hat\varphi$ | all stiffness of the source is reduced by a measured factor, and with it the power of every test |
| Per-frame term of the prior error | deflated: variance $\Sigma_\alpha$ | assumed independent beyond the left-out block |
| Ordinary surface term of the prior error | deflated: variance $\Sigma_\mu$ | the prior's information saturates (below) |
| Segmentation into entities, depth order, surfel scales | assumed | flagged in part by the within-entity remainder and outline consistency |
| Surface term in the tail: a regional fault | detected by the between-source energy | cannot be absorbed: it has no global form |
| Planar reflection, object moving along the strip | detected in part, by outline against interior and by cross strips | not detected without an attached outline |
| Gross pose blunder, scene change between passes | outside the model | shows as inflated dispersion of a frame or pass |

**Honest prior information saturates.** With a surface term of covariance $\Sigma_\mu$ shared by all
frames and per-frame terms of covariance $\Sigma_\alpha$, the information that $n$ frames of
monocular depth carry about an entity's placement is

```math
\Lambda^{D,\mathrm{honest}}_m = \bigl( \Sigma_\mu + \Sigma_\alpha / n \bigr)^{-1} \ \le\ \Sigma_\mu^{-1} ,
```

however many frames repeat. Nominal blocks added per frame grow without bound and silence the pixels;
the saturated form lets the pixels speak at the right rate with no gate at all. The fused error
variance of a depth-only map is about $\varrho + (1 - \varrho)/n$ of a single frame's, where the
nominal map states $1/n$: this is the predicted overconfidence, and it is measurable before any image
evidence is used. The gate is needed only for the tail beyond that law.

Two limits follow. A fault that lies in an absorbed subspace is invisible by construction, so a fault
filling most of a frame cannot be told from a pose or scale error. And the fates compete: every
nuisance absorbed lowers the stiffness available for detection, so each must earn its place by a
measured retained fraction.

### 4.7 Correspondence-free: the exact sense, the gain, the price

**The sense.** No discrete association between image points is estimated, stored or trusted. A pixel
is associated with the primitives it renders, softly, by the compositing weights, and the association
is recomputed whenever the map changes. The primitive is the correspondence. If the input poses come
from a feature-based adjustment, the claim covers the mapping stage; it covers the pipeline only when
poses come from navigation sensors and are corrected inside the model.

**The gain.** Weak texture lowers stiffness continuously instead of breaking a matcher, so abstention
is a value of the stored field and not a missing datum. Every pixel contributes, including outlines
of untextured regions through the label channel. Occlusion is consistent across all evidence. Cost is
linear in pixels with no track management, and the evidence arrives in information form.

**The price.** The comparison is local. The pull is a valid estimate only while the misplacement
shifts the image by less than the correlation length of the texture, a few pixels, or of a feature
channel, a few patches. Beyond that the gradient has no direction: detection survives, because a
decorrelated region has a large saturated misfit, but estimation does not. Three things widen the
range: coarse channels, a one-dimensional sweep of an entity's offset through the renderer, and above
all the prior, which starts every surface near its place. Explicit matching has a far larger capture
range and remains the better tool where the prior is absent or grossly wrong.

**Cross view.** Evidence is cross-view in that every frame speaks about the same primitives. Views
from other strips, cameras or passes (nadir and oblique, crossing lines, repeated flights) enter the
same way. They stiffen the placement directions one strip leaves soft and expose objects moving along
a strip. They also strain the assumptions, since diffuse appearance fails sooner under large changes
of viewing angle and illumination changes between passes; the model carries a dispersion and an
exposure term per pass, and the within-image misfit splits into within-pass and between-pass parts.
Under strong appearance change the label channel, outline parallax, is the evidence that survives.
Ground-to-aerial matching is outside what is claimed.

### 4.8 Attribution and when to listen

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

### 4.9 What is learned, and transfer to pixel-blind entities

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

### 4.10 Mapping

1. **Initial placement** from the monocular depth maps alone, with per-frame scale and offset and the
   shared distance bias absorbed. This is the step that needs no pixels and brings every surface
   within the capture range.
2. **Records.** Depth and image records are accumulated per entity through the renderer, with
   projective appearance from source frames and disjoint judging frames, and calibrated.
3. **Fusion by information.** With honest records the fused placement weights the pixels by
   $\vartheta / (1 + \vartheta)$ per direction. No schedule or hand-set weight between depth and
   photometric terms is involved.
4. **Gates.** Where the energy is in the tail, the attribution gates the placement component of the
   depth record, keeping the monocular shape, or the interior image record of a second-layer entity.
   Detection and repair are one operation. Pixel-blind entities are governed by the prior heads or
   reported as unsupported.
5. **Refit.** Geometry and appearance are fitted to both sources under the gates with the indicator
   held fixed. The test model with projective appearance is not the final model.
6. **Large area.** Records add across tiles; a tile is processed with its frames, and overlapping
   tiles merge by adding records source by source. Per-frame nuisance couples the tiles a frame sees.

Maps compared: pixels only; depth only; both with a fixed weight; both with a decaying depth weight;
both by information without gates; both with gates.

### 4.11 Why primitives

The primitives have a job if, and only if, these are used: both families as quadratic forms on the
same placement coordinates with occlusion resolved, so that information ratios, tension and gates are
defined per direction; the factorisation of Section 3.2; entities, features and labels fused in one
place by the same weights; and injection as a move of a group with correct occlusion. Any explicit
surface with a differentiable renderer would serve. The planar surfel is the one for which placement
is three numbers and the factorisation is exact.

### 4.12 What exists and what must be built

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

### 5.1 What must be shown, and how it is scored

A stored information is a forecast of error, so the primary experiment is calibration against exact
geometry. The headline metric is the **surface log-score**: the negative log density of the true
surface under the map, in nats per square metre. It is a proper scoring rule over geometry and
information jointly: it cannot be improved by inflating or deflating the stated information, and it
falls when either the geometry or its error bar improves. Reported with it: the robust scale of the
standardised placement error (one if honest, above one if overconfident), coverage of the 68% and 95%
regions, accuracy and completeness, each per surface class and per provenance.

### 5.2 Stages, cheapest decisive first

**Stage 0, on a CPU.** A synthetic entity with known offset and tilt on the reference renderer: does
the image record recover the injected placement, and how large and how stable is its dispersion?

**Stage A. Is a depth-only information map dishonest, and by how much?** The existing depth-only
pipeline with real monocular depth on a synthetic aerial block with exact depth. Scale of the
standardised error and log-score per class; the same after the consensus gate; the repetition
coefficient against the left-out block, and the overconfidence it predicts,
$\varrho + (1 - \varrho)/n$ against the stated $1/n$. One depth pass and CPU work.

**Stage B. Is the pixel stiffness honest, and how far does it reach?** By injection on the
depth-placed surface: capture range per channel and image scale; stated against realised information
per surface class; the fraction of an injected fault that each appearance regime absorbs; the same
against the residual parallax the prior actually leaves.

**Stage C. Does the two-source map win?** The maps of Section 4.10 on the log-score, calibration,
accuracy and completeness. Distribution of the energy on healthy entities; detection of true
placement faults by repetition coefficient; attribution on water and glazing.

**Stage D. Nuisance.** Ladders of injected pose noise, depth scale and offset error, and exposure
change: calibration must stay flat and stated information must fall as the retained fraction
predicts. Regional faults and reflections must not be absorbed; they must be detected.

**Stage E. Large area.** Additivity of records across tiles; completeness on weak texture against
matching pipelines; cost against area.

**The line stops if any of:**

- S1. The depth-only map is already honest: no overconfidence, or none that the repetition
  coefficient explains. The problem does not exist on that data.
- S2. The image record is not a measurement: no channel or scale recovers injected displacements of
  the size the prior leaves, or its dispersion is so large that realistic faults carry no energy.
- S3. Always explained away: no appearance regime keeps most of an injected fault while rendering
  healthy surface nearly as well as the unrestricted model.
- S4. Stated covariance does not predict placement error after the label-free corrections.
- S5. Fusion already answers it: gates add nothing over information-weighted fusion with saturated
  prior information, or a decaying depth weight matches every gain.
- S6. Matching wins at every texture level on accuracy and completeness, so being correspondence-free
  buys nothing.
- S7. Calibration breaks under injected pose or prior error instead of information falling.
- S8. The entity is the wrong unit: masks explain no more fault energy than tiles of equal count.

Thresholds for each are fixed in the experiment design before measurement.

### 5.3 Baselines

Depth only; pixels only; fixed and decaying depth weights; robust depth loss; removal of whole frames;
the consensus gate; selective depth supervision by virtual-stereo masks, photometric-error masks,
cross-view depth consistency and heuristic confidence; per-Gaussian Fisher uncertainty and Bayesian
splatting as alternative information fields, scored on the same log-score; structure-from-motion with
multi-view stereo; large-scale aerial splatting on matched inputs; a feed-forward multi-view model.
Bounds: an oracle gate with true masks.

### 5.4 Protocol

Evaluation quantities are written to a store only the scoring code reads. Thresholds, strata and
appearance regime are chosen by held-out likelihood, by injection, or on labelled scenes disjoint from
evaluation. Splits are by ground tile or scene; heads are cross-fitted by ground tile. Intervals are
cluster-bootstrap intervals over ground tiles with the number of entities beside each. A rendered
scene has perfect poses and synthetic materials: images are degraded, poses perturbed, and verdicts
repeated on real scenes, which decide where the two disagree.

## 6. Limitations

- **Honest image information may be small.** If dispersion from surfel-resolution misfit is large,
  the image spring is too soft to judge, and the correct output of the map is "little is known".
- **The capture range is short.** Mapping by rendering alone depends on the prior placing surfaces
  within a few pixels of residual parallax; where it does not, a sweep or matching is needed.
- **Absorbed directions are blind.** Absolute placement is no better than the poses, and a fault
  filling most of a frame is confounded with pose and scale.
- **Independence of the two sources is an assumption.** Both read the same image: the depth network
  from one frame, the renderer from several.
- **Pixel-blind entities** are judged by heads alone, and a repeated fault on them is beyond every
  label-free check.
- **Image-consistent wrong geometry**, a reflector without attached outline or constant-speed motion
  along the strip, passes every image test.
- **The entity may be the wrong unit**, and frozen segmenters may not form repeatable regions on
  water and glazing in nadir views.
- **The statistic and the information are classical.** The contribution is the representation, per
  source and calibrated, and what it makes measurable; overlap with uncertainty-aware splatting and
  selective depth supervision must be examined further.

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
| $\Lambda^s_m$, $\boldsymbol\eta^s_m$, $\hat\varphi^s$ | the record of source $s$ on entity $m$: information, pull, dispersion |
| $E_m$ | stored energy $T_m / 2$, in nats |
| $\mathbf q$, $\Sigma_q$ | nuisance block and its prior covariance |
