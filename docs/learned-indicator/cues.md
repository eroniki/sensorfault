# Evidence Definitions: Depth Messages and Image Messages on the Same Primitives

This note defines the evidence for a fault indicator on monocular depth when the surface is a set of
2D Gaussian surfels fitted to posed aerial frames, with one monocular depth map per frame, and the
differentiable renderer is used for both kinds of evidence. Every primitive, and every surface entity
made of primitives, receives two families of messages about its placement: one from the depth maps,
one from the pixels through the renderer. The fault test is the tension between the two families. A
fault that every frame repeats is visible to it, because all depth messages pull one way and the
pixels the other.

The note works out: the image message in information form from the rasteriser's Jacobian; why it is
defined on groups of primitives and not on single ones; how appearance is restricted so that it cannot
hide a misplaced surface; the tension statistic, its null and its calibration; what is identified;
how circularity is avoided; how entities and frozen features are lifted onto primitives; what remains
to be learned; what exists in the code; the closest prior work; and what would end the idea.

Notation follows the main document where it exists: frame $i$, pixel $u$, surfel $k$ with centre
$\boldsymbol\mu_k$, frame $[\mathbf t_{k1}\ \mathbf t_{k2}\ \mathbf n_k]$ and geometric perturbation
(normal offset $\nu$, tilts $\tau_1, \tau_2$); depth information blocks $\Lambda_k^{(i)}$ and vectors
$\boldsymbol\eta_{ik}$; compositing weight $w_{uk}$; information ratio $\vartheta$ and share
$\varsigma$; repetition coefficient $\varrho$; minimum detectable fault $\delta_{\min}$.

**Status.** Nothing here has been measured, and no code was run. Statements about the project's code
were checked against the source. Statements about third-party work are marked *checked* (read from the
paper or project page) or "(recalled)". The rasteriser library is not installed in the environment
that was inspected, so its interface is known only from how the project's scripts call it.

## 1. What the Primitives Are For

A depth-only surfel map with images as an outside judge gives the primitives no role that a depth map
per frame could not play. In the present formulation they have three, and the note should be read as a
test of whether these are real.

1. **A common coordinate system for evidence.** Depth messages and image messages are both expressed
   as information about the same few placement parameters. Messages add; the comparison of the two
   families is a comparison of two estimates of one quantity.
2. **Visibility for free.** Which frames see a piece of surface, what occludes it, and which side owns
   a silhouette are decided by the rasteriser, consistently for prediction, Jacobian and perturbation.
3. **A carrier for everything lifted from frozen models.** Segment labels and features attached to
   primitives are fused over frames and can be rendered into any view.

None of this requires Gaussians in particular. Any differentiable surface renderer with local planar
elements would serve. Surfels are used because an element has a plane (so "offset along the normal"
and "tilt" are its natural coordinates), because the depth information blocks already exist for them,
and because rendering is fast enough to re-render every view under perturbation.

## 2. Placement Parameters

### 2.1 Primitive

The code carries a ten-dimensional perturbation per surfel (position, rotation, log-scales, opacity
logit), of which depth observes three combinations: $\nu = \mathbf n^\top \delta\boldsymbol\mu$ and the
two tilts. Scales are frozen and opacity is frozen after a warm-up. The image message is defined on
the same three coordinates $\boldsymbol\theta_k = (\nu, \tau_1, \tau_2)$.

### 2.2 Entity

A surface entity $m$ is a set of primitives $\mathcal K_m$ (Section 9) with a mean frame and tangent
coordinates $\mathbf y_k$ of its members. Its placement is three numbers
$\boldsymbol\delta_m = (\nu_m, \tau_{m1}, \tau_{m2})$, acting rigidly:

```math
\boldsymbol\theta_k = B_k\, \boldsymbol\delta_m, \qquad
B_k = \begin{bmatrix} 1 & -y_{k1} & -y_{k2} \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}, \quad k \in \mathcal K_m ,
```

up to the rotation between the member's frame and the entity's. This is the affine fault field of a
region (a planar surface put on a wrong plane) written on primitives. The remaining degrees of freedom
of the members, $\boldsymbol\theta_k - B_k \boldsymbol\delta_m$, are the *shape* of the entity.

## 3. The Depth Message

Unchanged. Frame $i$ contributes to surfel $k$ the block $\Lambda_k^{(i)}$ and the vector
$\boldsymbol\eta_{ik}$ (weighted Jacobian of the composited ray-plane depth times the inverse noise
variance, and times the residual). In the code the blocks exist in a lumped form, built from
$\sqrt{w_{uk}}$-weighted Jacobians, which is block-diagonal and majorises the true Hessian. Projected
on the entity,

```math
\Lambda^{D}_m = \sum_i \sum_{k \in \mathcal K_m} B_k^\top \Lambda_k^{(i)} B_k, \qquad
\boldsymbol\eta^{D}_m = \sum_i \sum_{k \in \mathcal K_m} B_k^\top \boldsymbol\eta_{ik}, \qquad
\hat{\boldsymbol\delta}^{D}_m = \bigl(\Lambda^D_m\bigr)^{-1} \boldsymbol\eta^D_m ,
```

with the per-frame terms $\boldsymbol\eta^D_{m,i}$ kept separately. $\hat{\boldsymbol\delta}^D_m$ is
where the depth maps alone would put the entity, relative to the current surface.

## 4. The Image Message

### 4.1 Jacobian of the rendered colour for one surfel

A pixel ray with origin $\mathbf c$ and direction $\mathbf w$ hits the plane of surfel $k$ at depth
$d = \mathbf n^\top(\boldsymbol\mu - \mathbf c) / \mathbf n^\top \mathbf w$, at local coordinates
$\mathbf s = (\mathbf t_1^\top \mathbf x, \mathbf t_2^\top \mathbf x)$ with
$\mathbf x = \mathbf c + d\,\mathbf w - \boldsymbol\mu$. The footprint is
$G = \exp(-\tfrac12 \mathbf s^\top L^{-2} \mathbf s)$ with $L$ the diagonal matrix of the two scales
(a screen-space low-pass term of the published method is left out here; recalled). The pixel colour is
$\hat C(u) = \sum_l w_{ul}\, \mathbf c_l$ with $w_{ul} = a_{ul} T_{ul}$, $a_{ul} = \alpha_l G_{ul}$ and
$T$ the transmittance in front.

Moving the surfel along its normal by $\nu$ moves the hit point within the plane:

```math
\frac{\partial \mathbf s}{\partial \nu} = \frac{1}{\mathbf n^\top \mathbf w} \begin{bmatrix} \mathbf t_1^\top \mathbf w \\ \mathbf t_2^\top \mathbf w \end{bmatrix} =: \mathbf t_{uk} ,
```

the tangent of the angle of incidence, resolved in the surfel's axes. A tilt acts at the hit point as a
normal offset that grows with distance from the centre, $\nu_{\mathrm{eff}} = -(\tau_1 s_1 + \tau_2 s_2)$,
exactly as for depth. With $\kappa_{uk} = \partial \log G_{uk} / \partial \nu = - \mathbf s_{uk}^\top L_k^{-2}\, \mathbf t_{uk}$
and the compositing derivative $\partial \hat C / \partial a_k = T_k (\mathbf c_k - \mathbf C^{>k})$,
where $\mathbf C^{>k}_u$ is the colour composited behind $k$ at that pixel,

```math
\mathbf J_{uk} = \frac{\partial \hat C(u)}{\partial (\nu, \tau_1, \tau_2)_k}
= w_{uk}\ \kappa_{uk}\ \bigl( \mathbf c_k - \mathbf C^{>k}_u \bigr)\ \bigl[\, 1,\ -s_1,\ -s_2 \,\bigr]_{uk} .
```

Three factors: the weight of the surfel in the pixel, the rate at which its footprint slides, and the
colour contrast with what lies behind. Three properties follow at once.

- **No contrast, no information.** On blank surface $\mathbf c_k = \mathbf C^{>k}$ and the Jacobian
  vanishes, whatever the geometric error. Abstention is automatic.
- **No obliquity, no information.** $\kappa$ contains $\mathbf t_{uk}$: for a ray along the normal a
  normal offset moves nothing. For a nadir strip over flat ground the information comes from off-axis
  rays.
- **Silhouettes are included, with ownership.** At the outline of an entity the contrast is with the
  surface behind, and only the surfels in front have weight there. The motion of an occluding contour
  is credited to the occluder, as it must be. Parallax of an untextured roof against the ground is
  therefore part of the image message without a separate construction.

Not differentiated: changes of depth order between surfels, which are discontinuous.

### 4.2 Why a single primitive has no usable image message

Moving one surfel while its neighbours stay tears the surface: it uncovers what is behind on one side
and overlaps its neighbour on the other. The response of the image to that is real, but it is not the
response to a displaced surface, and the Jacobian blocks of neighbouring surfels are strongly coupled
through the pixels they share, far more than for depth. A block-diagonal (lumped) image information
per primitive would misstate the evidence in both directions. The image message is therefore defined
for a group moved together. For a locally opaque group moved rigidly the sum of the member Jacobians
reduces to the brightness-constancy form,

```math
\mathbf J_{um} = \sum_{k \in \mathcal K_m} \mathbf J_{uk} B_k \ \approx\ - \nabla \hat C(u)^\top \mathbf M_{u}, \qquad
\mathbf M_u = \frac{\partial \mathbf p(u)}{\partial \boldsymbol\delta_m} ,
```

the image gradient of the rendering times the image motion of the surface point under the placement
change, inside the entity, plus the silhouette terms at its outline. The entity message has three
unknowns and many pixels; it is the unit on which a test with a reference distribution can be built.
Where a per-primitive map is wanted, an entity can be a small disc of neighbouring surfels.

### 4.3 Restricting appearance: the reduced message

Appearance can explain away a misplaced surface. The effect can be stated exactly. With geometry fixed,
the rendered colour is linear in the surfel colours, $\hat{\mathbf C} = A(\boldsymbol\delta)\, \mathbf a$,
where $\mathbf a$ stacks all appearance parameters (a constant colour per surfel, plus a gain and
offset per frame). Eliminating $\mathbf a$ by least squares at each geometry (variable projection) gives
the image loss as a function of placement only, with Gauss-Newton Jacobian and information

```math
\mathbf J^{\mathrm{red}}_m = \bigl( I - A (A^\top A)^{-1} A^\top \bigr)\, \mathbf J_m, \qquad
\Lambda^{I}_m = \frac{1}{\sigma_c^2}\ \mathbf J^{\mathrm{red}\,\top}_m \mathbf J^{\mathrm{red}}_m, \qquad
\boldsymbol\eta^{I}_m = \frac{1}{\sigma_c^2}\ \mathbf J^{\mathrm{red}\,\top}_m \mathbf r ,
```

with $\mathbf r$ the image residual. The image information about placement is what is left of the
Jacobian after projecting out everything appearance can absorb.

- With one colour per surfel and $L$ frames, the projection removes the mean over frames of the image
  motion of each surface element. What remains is motion relative to the mean view,
  $\mathbf M_{u,i} - \bar{\mathbf M}_u$: parallax. The reduced information is gradient energy times
  $\sum_i \lVert \mathbf M_{u,i} - \bar{\mathbf M}_u \rVert^2$, the same structure as the information of
  a track.
- With appearance free per frame (a per-frame colour, or view-dependent colour of an order comparable
  to the number of distinct viewing directions), the projection removes everything and
  $\Lambda^I_m = 0$. A strip sees each point from a narrow fan of directions along one line, so even
  low-order view dependence is close to this case. Published work reports that the energy of
  higher-order view-dependent coefficients marks primitives whose geometry is poorly constrained or
  whose appearance has absorbed an error (*checked*), which is the same phenomenon seen from the
  other side.

The appearance model *for the test* is therefore fixed: constant colour per surfel, per-frame gain and
offset, opacity and scales frozen, no densification or pruning. This is a judging model, separate from
whatever is used to render final images.

**Sample splitting.** The reduced message removes absorption but still fits colour on the frames it
judges. The alternative is to fit colour on a fold of frames and take the image message from the
others, exchanged across folds of contiguous frames. No projection is then needed on the judging
frames. Colour taken from a single frame and judged in the others is the hand-warped compensated clip:
that construction is the special case of rendering one frame's texture through the surface. Both
variants are defined; the reduced message uses all frames and is the default, the split version is the
check on it.

### 4.4 Feature-metric variant

Frozen dense features of each frame, attached to surfels by the same linear fit and rendered like
colour, give a second image message with the same algebra. It tolerates exposure and weak texture
better and has a wider linear range; it is coarser (features live on a grid of 14 to 16 pixels,
*checked* for the video encoder, recalled for image encoders) and features are not exactly
view-independent, which the constant-per-surfel model then reads as residual. Whether the library
renders more than three channels in its 2D mode was not checked.

### 4.5 How it is computed

A backward pass gives $\boldsymbol\eta^I$ but not $\Lambda^I$, which needs Jacobian products. For three
parameters per entity the simplest exact route is finite differences through the renderer: perturb the
placement of a set of entities by $\pm h$ in one parameter, re-render every view, difference. Entities
that share no pixel in a view can be perturbed in the same render, so with a colouring of the entity
adjacency graph into $q$ classes the cost is $6q$ renders per view; pixels near an outline respond to
the entity in front, which the rasteriser handles. Forward-mode differentiation would give the same in
$3q$ passes if the rasteriser supports it (not checked). The projection of Section 4.3 is a sparse
least-squares solve per colour channel. This perturb-and-re-render loop is also the injection used for
calibration (Section 5.3): the same code serves both.

## 5. Null and Calibration

### 5.1 Why the nominal information is not an error bar

$\Lambda^I_m$ with independent pixel noise is far too large: residuals are spatially correlated and
dominated by model error (resampling, unmodelled shading, small pose error), not sensor noise. Used as
it stands, every entity would fail any test.

### 5.2 Variance from the dispersion of per-frame scores

Let $\mathbf g_{m,i}$ be the part of $\boldsymbol\eta^I_m$ contributed by the pixels of frame $i$,
evaluated at the image optimum. Under the hypothesis that one placement explains all frames they
scatter around zero, and their dispersion, not the nominal noise, estimates the variance:

```math
\widehat V_m = \sum_i \mathbf g_{m,i}\, \mathbf g_{m,i}^\top, \qquad
\widehat{\operatorname{Var}}\bigl(\hat{\boldsymbol\delta}^I_m\bigr) = \bigl(\Lambda^I_m\bigr)^{-1} \widehat V_m \bigl(\Lambda^I_m\bigr)^{-1} ,
```

the cluster-robust (sandwich) form with frames as clusters. It uses neither the depth maps nor labels.
With five to ten frames per entity it is noisy for a single entity; it is pooled over entities of a
stratum as a variance inflation factor $\widehat V_m \approx \phi_s \Lambda^I_m$. The depth family is
calibrated in the same way from its per-frame terms $\boldsymbol\eta^D_{m,i}$, but there the dispersion
only measures the non-repeated part of the monocular error: a repeated fault leaves it untouched. That
asymmetry is the reason the image family is needed.

### 5.3 Ancillary statistics, re-expressed

- **Along the parallax direction.** A pure placement error $\boldsymbol\epsilon$ produces per-frame
  scores $\mathbf g_{m,i} = \Lambda^I_{m,i} \boldsymbol\epsilon$, a fixed pattern over frames set by
  their gains. The misfit of the observed scores to that pattern,
  $A^{\parallel}_m = \sum_i \lVert \mathbf g_{m,i} - \Lambda^I_{m,i} \hat{\boldsymbol\delta}^I_m \rVert^2_{\widehat V_{m,i}^{-1}}$,
  tests whether a single placement explains the frames.
- **Across it.** The placement parameters cannot produce image motion across the epipolar direction.
  A two-parameter image shift of the rendered entity in each frame is estimated by one registration
  step on the entity's pixels, with the rasteriser's visibility mask; its component across the epipolar
  direction must vanish for a static rigid scene, and its component along it reproduces the explicit
  parallax measurement with gain $b_i$. This keeps the calibrated explicit reading inside the renderer.
- **By injection.** Moving an entity by a known placement and re-rendering must change
  $\hat{\boldsymbol\delta}^I_m$ by exactly that amount. The slope of recovered against injected
  placement, per stratum and as a function of injected size, gives the linear range and corrects the
  scale of the message. It is available on any posed sequence without labels.

### 5.4 Linear range and the sweep

Gradient and curvature describe the image loss within about a pixel or two of residual image motion
(more for features, more on an image pyramid). A gross fault moves the surface by many pixels of
parallax; evaluated at the depth-only surface, the gradient then has arbitrary direction and the
curvature is not an error bar. For such entities the message is obtained from a **sweep**: the reduced
image loss is evaluated on a grid of offsets $\nu_m$ by re-rendering, coarse to fine, and
$\hat{\boldsymbol\delta}^I_m$ and its information are taken at the minimum of the profile. This is plane
sweeping carried out by the renderer on an entity, with occlusion handled. An entity whose profile has
no distinct minimum abstains. The derivative form is the cheap special case for small tension.

## 6. The Tension Statistic

Two estimates of one placement, with calibrated covariances
$\Sigma^D_m$ and $\Sigma^I_m$:

```math
\mathbf d_m = \hat{\boldsymbol\delta}^D_m - \hat{\boldsymbol\delta}^I_m, \qquad
T_m = \mathbf d_m^\top \bigl( \Sigma^D_m + \Sigma^I_m \bigr)^{-1} \mathbf d_m \ \sim\ \chi^2_{3} \quad \text{under no fault.}
```

At the joint fit $\boldsymbol\delta^\ast = (\Lambda^D + \Lambda^I)^{-1}(\boldsymbol\eta^D + \boldsymbol\eta^I)$
the two families pull with equal and opposite force
$\mathbf g = \boldsymbol\eta^D - \Lambda^D \boldsymbol\delta^\ast = \bigl( (\Lambda^D)^{-1} + (\Lambda^I)^{-1} \bigr)^{-1} \mathbf d$,
so with nominal covariances $T_m = \mathbf g^\top \bigl( (\Lambda^D)^{-1} + (\Lambda^I)^{-1} \bigr) \mathbf g$:
the squared pull, normalised by the sum of the two inverse informations. Four remarks.

1. **It is the parallax gap.** $\mathbf d_m$ is the error of the prior minus the error of the image
   estimate, pooled over the entity; the statistic is the compatibility test of two measurement groups
   that estimation theory has long used. What the renderer adds is how the second estimate is formed:
   with visibility, silhouettes and appearance elimination handled by one model.
2. **The pull alone is not the statistic.** A large pull with large information on both sides is
   expected noise. During optimisation, before either family is at a stationary point, the difference
   form must be used, not the gradients.
3. **Repeated faults.** If all frames share an error $\boldsymbol\epsilon$,
   $\hat{\boldsymbol\delta}^D_m$ is displaced by $\boldsymbol\epsilon$ with a small nominal covariance,
   and $T_m$ is large. The between-frame contrasts of the depth scores $\boldsymbol\eta^D_{m,i}$ are
   depth consensus and contain no repeated term; only the difference of the two family means does.
4. **Minimum detectable fault.** Along a direction $\mathbf h$ of placement,
   $\delta_{\min}(m, \mathbf h) = (z_{1-\alpha/2} + z_{1-\beta_{\mathrm p}}) \sqrt{\mathbf h^\top (\Sigma^D_m + \Sigma^I_m) \mathbf h}$.
   Abstention is graded by it. With $\varsigma = \Lambda^I/(\Lambda^I + \Lambda^D)$ the main document's relation
   between harm and detectability carries over unchanged: for a fault of given harm to the fused
   surface, detectability rises steeply with the image share and vanishes with it.

### 6.1 Splitting the image family: interior against outline

The pixels of an entity divide into its interior and a band along its owned outline, and the image
message divides with them into $\hat{\boldsymbol\delta}^{I,\mathrm{int}}_m$ and
$\hat{\boldsymbol\delta}^{I,\partial}_m$. Three pairwise tensions result.

| Depth vs outline | Depth vs interior | Interior vs outline | Reading |
| --- | --- | --- | --- |
| low | low | low | prior confirmed |
| high | high | low | prior misplaced; fault |
| low | high | high | something is seen in or through the surface: reflection, transparency, or a real recess |
| high | low | high | outline misattributed, or prior and interior share the virtual depth |

The third row is the layer test: the interior texture belongs to a layer behind the surface that the
outline defines. From geometry alone a planar mirror and an opening cannot be told apart. Two further
checks use the same primitives: a virtual point is the reflection of a real one, so interior evidence
reflected across the outline plane should land on reconstructed surface with matching appearance; and
the appearance of a reflector changes with the frame while a diffuse surface does not, which shows as
a large residual of the constant-colour model with no placement that reduces it.

**One-sidedness at contact outlines.** Where an entity meets its neighbour without a depth step (a
lake and its shore), raising it would occlude the neighbour and is strongly visible; lowering it hides
its edge behind the neighbour and is visible only where a gap opens. The outline message is then
asymmetric and not Gaussian, and must come from the sweep, not from curvature.

## 7. What Is Identified

| Entity | Image message | Status |
| --- | --- | --- |
| Textured, diffuse | interior and outline, full rank | tension test; often $\varsigma \to 1$, prior nearly irrelevant |
| Untextured, with owned outline crossing the flight direction | outline only | tension test on outline; rank can drop for entities elongated along the strip |
| Planar reflector with reconstructed surroundings | outline; interior withdrawn by the layer test | tested, with a material attribute |
| Untextured, contact outline only | one-sided | bounds from one side |
| Untextured and unbounded in the views, or every outline owned by neighbours | none | blind |
| Non-rigid, moving | fails the ancillaries | abstains |

Not identified by any message: a whole-frame error in the nuisance group of the depth model (it is
absorbed by the per-frame alignment); the shape of a blind entity; whether an unexplained deeper
interior is real. Shape *within* a tested entity is not tested by the three-parameter statistic;
residual per-member tension is a second-order term with weak power.

## 8. Listening to the Prior or to the Pixels

For each entity the placement part of the depth message receives a gate $w_m \in [0, 1]$, the
posterior probability of no fault given $T_m$ and a prior probability $\pi_m$ from the head
(Section 10); the shape part is kept:

```math
\Lambda_k \leftarrow \Lambda_0 + \sum_i \Bigl[ \Lambda_k^{(i)} - (1 - w_m)\, P_k^\top \Lambda_k^{(i)} P_k \Bigr] + \Lambda^I_k ,
```

with $P_k$ the projector onto the entity placement at member $k$. A gated entity is placed by the
pixels (or by its outline) and keeps the relief that the monocular map gives it. Three regimes follow
from $\varsigma$: where the image share is near one the gate hardly matters; where the shares are
comparable the test decides; where the image share is near zero the entity is placed by the prior, the
test is silent, and the gate is the head's prior alone, reported with an explicit *unsupported* state
when the entity lies outside the support of the labelled ones.

## 9. Lifting Entities and Features onto Primitives

**Entities.** Video segmenters have no promptless mode (*checked*); automatic mask generation exists
for single images (*checked*). Masks are generated per frame and attached to primitives through the
compositing weights. Two masks in different frames belong to one entity when they claim the same
primitives; entities are the connected components of that agreement, at a chosen overlap threshold.
Association through time is thus done by the surface, not by a tracker. The dependence is circular in
one place: a misplaced surface projects a mask onto shifted primitives, by the fault's parallax. For
regions large against that shift the assignment is stable; for small regions it is not, and the
entity set should be recomputed after gating. Lifting segmenter masks onto Gaussians is established
(*checked* that such work exists; details recalled).

**Features.** Frozen dense features are fused per primitive by the same linear fit as colour. They
serve the feature-metric message (Section 4.4) and the head's input. The difference between the fused
feature and the per-frame feature is appearance that changes with the view, a label-free cue for
reflective material.

**A video encoder as reader.** Rendered and real frames, resampled to a common view per entity, form
two clips whose tokens can be compared with the finite-difference response to a known placement
change, as a third reading of the same tension. It is optional here: the renderer already supplies a
calibrated message, and the video encoder would have to show response where the feature-metric message
has none.

## 10. What Is Learned

Nothing in Sections 3 to 8 is learned; it is estimation with a renderer. Learning enters in three
places, all light heads on frozen features, with no backbone updated.

1. **Fault prior for entities the test cannot reach.** A probe on features fused per entity is fitted
   to the outcomes $(T_m, \delta_{\min})$ of tested entities and applied where $\Lambda^I_m$ is small.
   Labels are selective: an entity is labelled when it has texture or an owned outline. A second probe
   predicts testability itself from the same features, without fault labels, and delimits the support.
2. **Null calibration.** The inflation factors $\phi_s$ per stratum, with strata from frozen features
   or concept prompts.
3. **Material attribute.** The layer test labels reflectors; a probe carries the attribute to entities
   without reconstructed surroundings.

The head must not receive rendered-minus-real residuals or any quantity computed from the depth map;
those are the test.

## 11. Circularity

| Shared element | Risk | Treatment |
| --- | --- | --- |
| Colour fitted on the judged frames | absorbs misplacement | constant colour, elimination by projection; cross-fitted folds as check |
| Opacity, scale, new primitives | explain misplacement by floaters or blur | frozen in the judging model |
| Linearisation point from the depth fit | image message invalid far from it | sweep; injection slope |
| Primitive set created from the depth maps | a faulty region has surfels only where the fault put them | entity placement moves them as a group; the sweep range is set by scene bounds, not by the prior |
| Entity definition through projected masks | boundary members misassigned under a fault | recompute after gating; exclude a boundary band from the interior message |
| Poses | both families | per-frame registration shift exposes pose error as a field common to all entities of a frame |
| Per-frame depth alignment | a regional fault biases scale and offset | robust alignment; alignment re-estimated with the gate |
| Tracks or pixels used both to fit and to judge | the judge is part of the judged surface | the test compares family estimates, each formed without the other; fusion happens after the gate |

## 12. Code: What Exists and What Must Be Built

| Item | Exists | To build |
| --- | --- | --- |
| Depth fit of surfels | `gs/fit.py`: optimiser on a ray-plane depth loss; colours are a constant placeholder; opacity frozen after warm-up, scales frozen | nothing for the depth family |
| Depth information blocks and vectors | `eval/gaussian_uncertainty.view_information` (lumped option), closed-form depth Jacobian $[1, -s_1, -s_2]/(\mathbf n^\top \mathbf w)$, opacity Jacobian of the compositor | projection onto entity placement |
| Compositor with per-pixel weights | numpy reference in `gs/reference.py`, torch port in `gs/fit.py` (depth only) | colour channel; the factor $\kappa$ and the contrast term of Section 4.1 are a few lines on the same quantities |
| Photometric 2D surfel training | `experiments/rgb_train_2dgs.py`: separate script, library rasteriser with view-dependent colour, densification, structural-similarity and regulariser losses | a judging mode: constant colour, no densification, frozen opacity and scale, linear colour solve, per-frame gain |
| Images in the data path | `depthcache/datasets.View.load_rgb`; `experiments/rgb_prepare.py` (half-resolution images of the synthetic block) | images in the frames used by `gs/` |
| Leave-out and gate driver | `gs/gate.ViewGateTest` (depth only, per frame) | entity-level statistics; image family |
| Entities, features | nothing | mask and feature lifting; frozen-model wrappers in the pattern of `depthcache/` |
| Perturb-and-re-render | nothing | entity placement transform, graph colouring, finite differences, sweep |

The two fitting paths do not share state: the depth fit uses its own ray-plane compositor because the
library's blended depth does not see surfel tilt (noted in the code), while the photometric script
uses the library rasteriser. The image Jacobian must be taken from the same footprint model as the
depth Jacobian, or the two families refer to slightly different surfaces. The reference compositor can
supply both on small scenes without a GPU.

## 13. Closest Prior Work

*Checked* (read from the papers):

| Work | What it does | Difference |
| --- | --- | --- |
| Reliable monocular depth supervision by a depth-inconsistency mask | per-pixel mask from re-projection of rendered depth between a view and a synthetic stereo partner; depth loss applied only where rendered geometry is inconsistent | decides where geometry needs help, not whether the prior is right; no information weighting, no test, no repeated-fault case |
| Reliability-aware depth supervision from an RGB-only baseline | per-pixel mask by thresholding photometric error of a model trained without depth | photometric evidence as a threshold on error; no placement estimate, no error bar |
| Confidence-aware depth regularisation | per-pixel confidence from image edges, texture and depth gradients | heuristic, does not compare prior with images |
| Fisher information for splatting | diagonal Fisher of the colour loss over all Gaussian parameters, for view selection and relative pixel uncertainty | one family only, diagonal, no depth prior, no test between terms |
| Photometric ambiguity in splatting surface reconstruction | marks primitives by the energy of view-dependent colour coefficients; applies normal priors to them | an indicator of appearance absorbing geometry, by percentile; no curvature or test |

Existence checked, content recalled: depth and normal priors in splatting with fixed or scheduled
weights; lifting of segmenter masks onto Gaussians; distillation of 2D features into Gaussians.
Recalled only: per-ray Laplace uncertainty for radiance fields; variable projection for separable
least squares; feature-metric alignment; plane sweeping; compatibility tests between observation
groups in adjustment theory.

What remains new, to the best of this search: (i) a calibrated test between the depth family and the
image family on shared placement parameters, with the image family's variance taken from per-frame
score dispersion and from injection; (ii) the reduced image message, which states exactly how much
placement information an appearance model leaves, and why view-dependent colour destroys it in a
strip; (iii) entity-level messages with the interior-against-outline split as a layer test;
(iv) explicit treatment of the fault that every frame repeats, which none of the checked methods
addresses and which consistency-based masks cannot see. The search was not exhaustive, the area is
crowded, and a per-primitive trust weight between priors and pixels may exist in work not found.

## 14. What Would End the Idea, and the Cheapest Tests

1. **The image family cannot be calibrated.** $T_m$ on fault-free entities is far from $\chi^2_3$ after
   inflation, or the inflation varies so much between entities that no stratum is stable.
   *Test A, CPU, reference compositor:* a small synthetic textured scene with exact geometry; form the
   reduced image message, the per-frame scores and the injection slope; check the null distribution of
   $T$ and the recovery of injected placements as a function of their size.
2. **Appearance elimination leaves nothing.** With constant colour and realistic strip geometry the
   reduced information is a small fraction of the nominal one. *Computed in test A* as the ratio of the
   two, per entity, for the actual camera fan.
3. **The linear range is too small and the sweep has no distinct minimum** on real textures at the
   fault sizes that occur. *Test B, CPU:* from cached monocular depth and exact depth, the residual
   parallax of true faults in pixels at working resolution, against the range found in test A.
4. **Per-entity image messages are dominated by neighbours** (coupling across entity boundaries),
   so that entity statistics are not approximately independent.
5. **Entities are unstable**: lifted masks fragment or merge under faults.
6. **Harmful faults sit in the blind rows of Section 7.**
7. **Prior work already contains the test.**

Test A needs no GPU and no data beyond a synthetic scene; it decides items 1 and 2, which are the
premises of everything else.

## 15. Not Known

- The size of the variance inflation of the image family and its stability across strata.
- The fraction of placement information that survives appearance elimination in strip geometry.
- The linear range of colour and feature messages in pixels of parallax on aerial texture.
- Whether the library rasteriser supports forward-mode differentiation or more than three channels in
  its 2D mode; its interface was read only from the project's scripts.
- Whether ownership as decided by the rasteriser is reliable when the depth of the occluder itself is
  the fault.
- How often outlines cross the flight direction with enough length for rank three.
- Cost per view of the perturb-and-re-render loop with realistic entity counts.
