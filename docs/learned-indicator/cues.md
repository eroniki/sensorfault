# Evidence Definitions: What the Map Knows, from Which Source, and How Honestly

This note defines the evidence carried by an information-bearing Gaussian surfel map built from posed
aerial frames in strips, with one monocular depth map per frame and no correspondences between images.
The map stores geometry, appearance and *information*: how strongly each source of evidence constrains
each piece of surface. The note gives the argument first, in plain sentences, and then the definitions
that support it: the two families of evidence, the measure of alignment between geometry and
appearance, the sense in which the method is correspondence-free, how nuisance (poses, depth-model
errors, rendering and appearance model error) enters the same information model, what is identified,
and what must be shown for the definitions to be believed.

Notation follows the main document: surfel $k$ with placement $(\nu, \tau_1, \tau_2)$ (offset along the
normal, two tilts); entity $m$, a set of surfels $\mathcal K_m$ with three placement parameters
$\boldsymbol\theta_m$; depth and image information $\Lambda^D_m$, $\Lambda^I_m$ and score vectors
$\boldsymbol\eta^D_m$, $\boldsymbol\eta^I_m$; dispersion $\hat\varphi$; retained fraction $\kappa_m$;
information ratio $\vartheta$; repetition coefficient $\varrho$.

**Status.** Nothing here has been measured and no code was run. Statements about the project's code
were checked against the source. Statements about third-party work are marked *checked* (read) or
"(recalled)".

## 1. The Argument

1. **A map that is used must say what it knows.** A reconstruction is geometry and appearance. A user
   of the map (a planner, a change detector, the next fusion step) also needs to know how well each
   part is determined and by what. Gaussian splatting stores the first two; opacity is coverage, not
   confidence. So the map needs a third field, information.
2. **Information has a physical meaning here.** A piece of surface is constrained by the images to the
   degree that moving it would change them. That degree is texture contrast times the spread of
   viewing directions. It is a stiffness: how hard appearance pulls geometry back when it is displaced.
3. **The monocular depth is a second spring on the same surface.** Each depth map pulls the surface
   towards where the network put it, with a stiffness set by the network's noise. Without it, blank
   surface is unconstrained; with it, every surface has a position.
4. **The two springs must be kept apart in the books.** If their stiffnesses are summed and only the
   sum is stored, a disagreement between them is lost. Kept by source, the disagreement is a
   measurement: the stretch between where each alone would put the surface.
5. **The energy stored in that stretch is the test.** Two springs pulled apart store an energy that
   depends on the stretch and on both stiffnesses. In statistical units it is a chi-square variable
   when both sources are right. This is the measure of alignment between geometry and appearance.
6. **A fault that every frame repeats is a stretch between the families and nothing else.** Within the
   depth family all frames agree, so no comparison among depth maps shows it. Only the image family
   can disagree with all of them at once.
7. **The stiffness must be honest, or the energy means nothing.** Image residuals are dominated by
   model misfit, not sensor noise; the nominal stiffness is far too large. It is corrected from the
   scatter of the image evidence among frames, which does not involve the depth prior.
8. **Appearance must not be allowed to carry the stretch.** A free appearance model can paint parallax
   as view-dependent colour. The information that remains about geometry after appearance has taken
   what it can is computable, and it is the number to store.
9. **No correspondences are needed, because the surface is the association.** Each view is compared
   with its own rendering of the shared primitives. This scales with pixels, pools evidence over an
   entity, and uses outlines and weak texture that a matcher discards. Its weakness is capture range,
   and the monocular depth is what supplies the starting point inside it.
10. **Nuisance enters the same accounts.** Whatever is frame-wide and low-order (pose, depth scale and
    offset, exposure) can be marginalised, and the price is paid in information that is then no longer
    claimed. Whatever is local to an entity and repeats in every frame cannot be marginalised, because
    it is indistinguishable from the thing being estimated. It can only be detected, by the energy.
11. **The unit is the entity.** One surfel moved alone tears the surface; a fault moves a surface. The
    books are kept per entity in three placement parameters, with the surfels carrying shape.
12. **What remains to learn is small.** Where the image spring is absent, the energy is silent; a head
    on frozen features carries verdicts from tested entities to untested ones and says "unsupported"
    where it has no basis.

The rest of the note makes each step exact.

## 2. The Map: Geometry, Appearance, Information

Per entity $m$ the map stores:

| Field | Content | Units |
| --- | --- | --- |
| Geometry | member surfels; placement $\boldsymbol\theta_m$ | m, rad |
| Appearance | constant colour and frozen-feature vector per surfel | |
| Information, depth | $\Lambda^D_m$ and per-frame terms; after marginalising per-frame alignment | m$^{-2}$, rad$^{-2}$ |
| Information, image | $\Lambda^I_m$ per channel (colour, feature, outline), after appearance and pose are accounted for; dispersion $\hat\varphi$ | same |
| Derived | retained fraction $\kappa_m$; ratio spectrum $\vartheta_{1..3}$; energy $T_m$; verdict | dimensionless |

Two requirements define "information" as opposed to a weight.

- **Provenance.** Information is additive over independent evidence and is stored per source, so that
  any source can be subtracted (leave-out) or compared with the rest.
- **Honesty.** The inverse of a stored information must predict the squared error of the estimate it
  belongs to. This is a testable property (Section 9), not a convention.

**The spring picture, exactly.** For a displacement $\mathbf x$ of the entity from where a source
alone would place it, the source's misfit is $\tfrac12 \mathbf x^\top \Lambda \mathbf x$: $\Lambda$ is a
stiffness. Two sources that would place the entity at positions differing by $\mathbf t$ act as two
springs in series stretched by $\mathbf t$; compliances add, and twice the stored energy is

```math
T_m = \mathbf t_m^\top \bigl( (\Lambda^D_m)^{-1} + \hat\varphi\, (\Lambda^I_m)^{-1} \bigr)^{-1} \mathbf t_m .
```

## 3. Two Families of Evidence

### 3.1 Depth

For the ray of pixel $u$ meeting the plane of surfel $k$ at depth $d_k$,
$\mathbf j_{uk} = \partial d_k / \partial(\nu, \tau_1, \tau_2) = [1, -s_1, -s_2] / (\mathbf n^\top \mathbf w_u)$
with $(s_1, s_2)$ the hit point in the tangent frame. With the depth model
$\hat d_i = a_i (d + \beta d^2) + b_i + \varepsilon_i$ and compositing weights $w_{uk}$, frame $i$ sends

```math
\Lambda^D_{ik} = \sum_u w_{uk}\, \frac{a_i^2\, \mathbf j_{uk}^\top \mathbf j_{uk}}{\sigma_i^2(u)}, \qquad
\boldsymbol\eta^D_{ik} = \sum_u w_{uk}\, \frac{a_i\, \mathbf j_{uk}^\top e^D_i(u)}{\sigma_i^2(u)} .
```

### 3.2 Image

Rendered appearance is $\hat{\mathbf c}_i(u) = \sum_k w_{uk} \mathbf a_k$. Its derivative with respect to
placement is a scalar times the same geometric row:

```math
\frac{\partial \hat{\mathbf c}_i(u)}{\partial \boldsymbol\theta_k} = \mathbf g_{uk}\, \mathbf j_{uk}, \qquad
\mathbf g_{uk} = - w_{uk}\, \bigl( \mathbf a_k - \mathbf b_{uk} \bigr)\, \bigl( \mathbf s_{uk}^\top \mathbf m_{uk} \bigr),
```

with $\mathbf b_{uk}$ the appearance composited behind $k$ (contrast) and $\mathbf m_{uk}$ the tangential
component of the viewing ray in units of the surfel's scales (slide). Both families act on the same
three coordinates through $\mathbf j_{uk}$ and differ by a weight. This is why the stiffness of the
image spring is contrast times spread of viewing directions: contrast is $\mathbf a_k - \mathbf b_{uk}$,
and after appearance is accounted for only *differences* of slide between frames remain (Section 3.4).

### 3.3 Entity

A single surfel moved alone tears the surface, and one view cannot separate offset from a tangential
shift of texture. The image message is defined for the entity,

```math
J^I_{im}(u) = \sum_{k \in \mathcal K_m} \mathbf g_{uk}\, \mathbf j_{uk}\, P_{km}, \qquad
\Lambda^I_{im} = \sum_u \frac{J^I_{im}(u)^\top J^I_{im}(u)}{\sigma_I^2}, \qquad
\boldsymbol\eta^I_{im} = \sum_u \frac{J^I_{im}(u)^\top \mathbf r_i(u)}{\sigma_I^2} ,
```

computed along three directions by moving the group and re-rendering. Inside the entity it reduces to
texture gradient times slide; at an outline the contrast with the surface behind remains, credited by
the rasteriser to the surface in front.

### 3.4 What appearance leaves

*Projective appearance* (the test model): surfels take their texture from source frames by projection
at the current placement; disjoint judging frames form the residual. The Jacobian then contains the
slide difference between judging and source view: parallax.

*Fitted appearance*: its freedom is removed by the Schur complement, and the fraction that survives is
stored,

```math
\Lambda^{I,\mathrm{prof}}_m = \Lambda_{\theta\theta} - \Lambda_{\theta a} \Lambda_{aa}^{-1} \Lambda_{a\theta}, \qquad
\kappa_m = \frac{\operatorname{tr} \Lambda^{I,\mathrm{prof}}_m}{\operatorname{tr} \Lambda_{\theta\theta}} \in [0, 1] .
```

One diffuse colour per surfel seen by many frames keeps $\kappa$ near one. View-dependent colour in a
strip, where viewing direction and parallax vary together, leaves almost nothing. Published work marks
poorly constrained primitives by the energy of their view-dependent coefficients (*checked*); $\kappa$
is the quantity behind that heuristic.

### 3.5 Channels

The same algebra holds for any rendered channel, each with its own dispersion: colour; frozen dense
features attached to surfels; and entity labels rendered against each frame's own segment mask. In the
label channel contrast exists only at outlines, so its message is outline parallax, and it needs no
interior texture and no photometric constancy.

## 4. The Measure of Alignment between Geometry and Appearance

**Statement.** For entity $m$, let $\hat{\boldsymbol\theta}^D_m$ and $\hat{\boldsymbol\theta}^I_m$ be the
placements each family alone supports. The alignment measure is the pair

```math
\mathbf t_m = \hat{\boldsymbol\theta}^D_m - \hat{\boldsymbol\theta}^I_m \quad [\text{m},\ \text{rad},\ \text{rad}], \qquad
T_m = \mathbf t_m^\top \bigl( (\Lambda^D_m)^{-1} + \hat\varphi\, (\Lambda^I_m)^{-1} \bigr)^{-1} \mathbf t_m \quad [\text{dimensionless}] .
```

- **Meaning.** $\mathbf t_m$ is the misalignment in metric units: an offset and two tilts. $T_m$ is the
  increase in total misfit caused by forcing both families onto one placement.
- **Null.** If both families are unbiased with independent errors and honest information,
  $T_m \sim \chi^2_3$. Along one direction the square root is a standard normal score.
- **Resolution.** The smallest misalignment detectable along a direction $\mathbf h$ at size $\alpha$
  and power $1 - \beta_{\mathrm p}$ is
  $\delta_{\min} = (z_{1-\alpha/2} + z_{1-\beta_{\mathrm p}}) \sqrt{\mathbf h^\top ((\Lambda^D_m)^{-1} + \hat\varphi (\Lambda^I_m)^{-1}) \mathbf h}$.
  It is stored with the measure; an entity with large $\delta_{\min}$ is not "aligned", it is untested.

**Where it is evaluated.** At an unrestricted joint fit the measure is void: the fit relieves stretch
with opacity, duplicates and view-dependent colour. It is evaluated at the surface placed by the depth
family alone, where $\boldsymbol\eta^D_m = 0$ and the stretch is the pull of the pixels on a surface
that has never seen an image:

```math
T_m = \boldsymbol\eta^{I\top}_m (\Lambda^I_m)^{-1} \bigl( (\Lambda^D_m)^{-1} + \hat\varphi\,(\Lambda^I_m)^{-1} \bigr)^{-1} (\Lambda^I_m)^{-1} \boldsymbol\eta^I_m .
```

**Decomposition.** With per-frame messages the misfit about a common placement splits exactly into
three parts,

```math
\chi^2_{\mathrm{total}} = \underbrace{\sum_j \bigl\lVert \hat{\boldsymbol\theta}^D_{mj} - \hat{\boldsymbol\theta}^D_m \bigr\rVert^2_{\Lambda^D_{mj}}}_{\text{within depth}}
+ \underbrace{\sum_i \bigl\lVert \hat{\boldsymbol\theta}^I_{mi} - \hat{\boldsymbol\theta}^I_m \bigr\rVert^2_{\Lambda^I_{mi}}}_{\text{within image}}
+ \underbrace{T_m}_{\text{between}} .
```

Writing the monocular placement error of frame $j$ as a surface term plus a per-frame term,
$\boldsymbol\mu_m + \boldsymbol\alpha_{mj}$: the first part contains only the per-frame terms (it is
depth consensus); the third contains the surface term; the second contains neither and is ancillary.
The repetition coefficient $\varrho$, the share of monocular error variance in the surface term, is
estimated from the first and third without ground truth.

**Honesty of the image stiffness.** The within-image part gives the dispersion,
$\hat\varphi = \chi^2_{\text{within image}} / (3 (n_m - 1))$, pooled per appearance stratum: a sandwich
variance with frames as units. Two label-free checks: tensions between split halves of one family must
follow the null, and moving an entity by a known placement and re-rendering must move
$\hat{\boldsymbol\theta}^I_m$ by exactly that amount.

**Range.** Gradient and curvature hold within about a surfel footprint of residual parallax. Beyond
it the measure is taken from a profile: the entity is moved over a grid of placements and re-rendered,
coarse to fine, and the image placement and its curvature are read at the minimum. An entity whose
profile has no distinct minimum has no image spring.

## 5. In What Sense Correspondence-Free

**Definition.** No association between points of different images is computed or stored at any stage
of mapping: no detected features, no matches, no tracks, no triangulated points. The unknowns are
surface and appearance parameters. Each frame is compared only with its own rendering of the shared
primitives. The association between views is implicit in the surface, continuous, and changes with the
estimate. This is analysis by synthesis, the principle of direct methods (recalled), on entities.

**What the claim does not cover.** Poses are an input. If they come from navigation sensors, the whole
pipeline is correspondence-free. If they come from structure from motion, correspondences were used
upstream; the claim then concerns mapping and pose refinement only, and should be worded so.

**What it buys.**

1. *Evidence that matchers discard.* A matcher needs a point that is distinctive in two dimensions.
   The image message needs gradient along the parallax direction only, summed over the entity: edges,
   weak texture, shading, and outlines all count, and there is no detection threshold below which
   evidence is thrown away.
2. *Scale.* Cost grows with pixels times frames. There is no pairwise matching graph, no track
   management across strips, and adding a frame is adding its messages.
3. *One visibility model.* Occlusion and silhouette ownership are decided once, by the rasteriser, for
   prediction, Jacobian and perturbation alike.
4. *Uncertainty where it is wanted.* The information is about the surface parameter, not about a match
   that must then be propagated.

**Where it is weaker than explicit matching.**

1. *Capture range.* The pull is valid within about a footprint; a matcher finds correspondences across
   hundreds of pixels. The design answers this in two ways: the monocular depth and the poses place
   the surface near the answer, which is the prior's first job, and the profile over placements
   extends the range at the cost of renders.
2. *Local minima* on repetitive texture; the profile must be read for several minima and abstain.
3. *No per-match outlier rejection.* A mover or a reflection contaminates an entity's message; it is
   caught by the within-image scatter and the layer test, at entity resolution.
4. *Dependence on an appearance model.* Exposure, specularity and resampling enter as model error,
   paid for through $\hat\varphi$ and $\kappa$.
5. *Initial geometry must exist.* Primitives are created from the depth maps; where the prior puts no
   surface, nothing is tested.

**Cross view, both readings.** Read as *several views of the same primitives*, cross-view evidence is
the image family as defined. Read as *views from different cameras, strips or passes* (nadir and
oblique, cross strips, repeat flights), the formulation covers it without change in form, and the
terms say what changes. A wider spread of viewing directions raises the slide differences and so the
stiffness; a cross strip adds the second parallax direction and repairs rank for entities elongated
along a strip; oblique views give information on facades. Against that, appearance differs between
passes. If each pass needs its own appearance, the Schur complement removes the cross-pass colour
information and $\kappa$ records the loss; what survives across passes is the feature channel, as far
as the features are stable, and the label channel, which needs no photometric constancy at all. Outline
parallax is thus the evidence that carries across the widest change of view.

## 6. Nuisance in the Same Information Model

**Rule.** A nuisance can be marginalised when its footprint in the scores is low-dimensional and
separable from entity placement, typically because it is common to a whole frame while placement is
local. The price is a Schur complement: the information along the confounded directions is removed
from the books. A nuisance whose effect on an entity is the same in every frame is not separable from
the entity's placement. It cannot be absorbed; it must be detected, or it is a stated blind spot.
"Absorbed" never means the error has gone: it means the map no longer claims the information that the
nuisance could have faked.

For a nuisance $\mathbf q$ with prior information $\Lambda_q^0$ and cross information
$\Lambda_{\theta q}$ with the entity,

```math
\Lambda_m^{\mathrm{marg}} = \Lambda_{\theta\theta} - \Lambda_{\theta q} \bigl( \Lambda_{qq} + \Lambda_q^0 \bigr)^{-1} \Lambda_{q\theta} ,
```

where $\Lambda_{qq}$ collects the evidence of *all* entities of the frame on $\mathbf q$. An entity that
is a small part of a well-textured frame loses little; an entity that fills the frame loses the
confounded directions entirely.

| Nuisance | Treatment | Price, and what is not covered |
| --- | --- | --- |
| Depth scale and offset per frame, $(a_i, b_i)$ | marginalised (projected out of each frame's depth residual, robustly) | an entity covering most of a frame loses its offset information from that frame; a fault that is affine over a whole frame is not a fault by definition and is invisible |
| Shared distance bias $\beta$ | estimated, one scalar, with leave-out | negligible information; a fault class tied to range everywhere is absorbed with it |
| Spatial correlation of depth noise | estimated deflation of depth information | none beyond the deflation; if wrong, depth information is dishonest |
| Monocular per-frame term $\boldsymbol\alpha_{mj}$ | averaged over frames; tested by depth consensus | enters the fused surface at $1/L$ |
| Monocular surface term $\boldsymbol\mu_m$ | **cannot be absorbed**: identical to a placement of the entity | detected by the between-family energy where an image spring exists; otherwise a blind spot handed to the head |
| Pose of a frame | marginalised with its prior, or estimated as a low-order field of image displacement common to the frame's entities | Schur price above; a global similarity and slow deformations along a strip that stay photo-consistent are not observable by either family |
| Intrinsics, lens, shutter | as pose: frame-wide low-order fields | same |
| Exposure, white balance | estimated per frame (gain, offset) | two parameters per frame and channel |
| Rendering model error (resampling, footprint approximation, residual shading) | dispersion $\hat\varphi$: information divided by it | valid for misfit that behaves like zero-mean scatter between frames; a bias common to all frames is not covered |
| Appearance freedom | projective appearance, or Schur complement with $\kappa_m$ stored | $1 - \kappa_m$ of the image information |
| View-dependent appearance, reflections, transparency | **not absorbed** | detected: interior against outline (layer test), within-image scatter, mirror consistency |
| Moving and non-rigid content | **not absorbed** | detected by within-image scatter; motion along the strip at constant speed mimics parallax and is not |
| Linearisation of the image family | profile over placements | renders; abstention where no minimum |

Pose deserves one more sentence, because it touches both families. A pose error moves every entity of
the frame together, by a pattern fixed by the camera geometry; it is therefore separable from the
placement of any one entity, estimable from the frame as a whole, and the stored information of each
entity is reduced by its share. What pose error cannot be separated from is a deformation of the whole
scene that is consistent with the erroneous poses. That is a property of the data, not of the
representation.

## 7. What Is Identified

| Entity | Image spring | Outcome |
| --- | --- | --- |
| Textured, diffuse | interior and outline, rank three | measured; usually pixel-dominated, the prior supplies shape and starting point |
| Untextured, owned outline crossing the parallax direction | outline (label channel) | measured through the outline; rank may drop along a single strip |
| Planar reflector, surroundings reconstructed | outline; interior withdrawn by the layer test | measured, with a material attribute |
| Untextured, contact outline only | one-sided | bounded from one side |
| Untextured and unbounded, or all outlines owned by neighbours | none | blind: placed by the prior; verdict from the head or "unsupported" |

Not identified by any evidence: errors inside the per-frame alignment group; scene deformations
consistent with the poses; whether an unexplained deeper interior is real; shape within an entity
beyond the three placement parameters.

## 8. What Is Learned

Sections 2 to 7 are estimation with a renderer. Learned components are light heads on frozen features
fused per entity, with no backbone updated: a prior on depth faults and one on misleading pixels,
fitted to the outcomes of tested entities and applied where the image spring is absent; a head that
predicts testability, which delimits the support and produces the "unsupported" state; and the
dispersion per appearance stratum. No head receives a residual or any quantity computed from the
depth map.

## 9. What Must Be Shown about the Evidence

In order of cost; the first decides whether the rest is worth doing.

1. **Honest information on a synthetic scene (CPU, reference compositor).** With exact geometry and
   synthetic texture: (a) the squared placement error of each family against the inverse of its stored
   information, per direction, should lie on the diagonal after $\hat\varphi$; (b) $T_m$ on fault-free
   entities against $\chi^2_3$; (c) injected placements recovered with slope one, as a function of
   size, which gives the capture range; (d) $\kappa_m$ for constant and for view-dependent appearance
   in strip geometry.
2. **Nuisance accounting.** Perturb poses, depth scale and exposure by known amounts: the realised
   loss of accuracy should equal the Schur prediction, and the null of $T_m$ should hold. Perturb by a
   photo-consistent scene deformation: nothing should be detected, as stated.
3. **Residual parallax of real faults against the capture range (CPU, cached depth and exact depth).**
4. **The three-way decomposition on real monocular depth**: $\hat\varrho$ against the value from exact
   depth.
5. **Coverage of the image spring**: share of surface area, and of harmful faults, per row of
   Section 7.
6. **Cross-view reading**: cross strips and oblique frames; rank repair and the survival of the label
   channel under changed appearance.

## 10. Occupied Ground, as Far as Checked

*Checked* (papers read): diagonal Fisher information of the colour loss over Gaussian parameters for
view selection and relative pixel uncertainty; per-pixel masks for monocular depth supervision from
rendered-depth inconsistency, from the photometric error of a model trained without depth, and from
image heuristics; an indicator of geometric ambiguity from view-dependent colour energy. None keeps
information by source, calibrates it, or tests one source against another.

Titles seen in search, content not read: several works use Fisher information or uncertainty in
Gaussian maps for active mapping, exploration and tracking, and at least one variational treatment of
splat uncertainty exists. The name "information-theoretic Gaussian splatting" was not found as a title
in a short search; that search was not exhaustive, and information-driven splatting for exploration is
close enough in wording that the name needs a qualifier (information by source, calibrated, used as a
test). Recalled only: direct and photometric bundle adjustment, covariance recovery in adjustment
theory, compatibility tests between observation groups, variable projection.

What the evidence definitions add: information stored per source on shared placement parameters; an
alignment measure in metric units with a null and a resolution; a calibration of the image family
that does not use the prior; an explicit ledger of what marginalisation costs; and the repeated fault
as a between-family effect.

## 11. What Would End It

1. The image information cannot be made honest: no stable dispersion per stratum, or $T_m$ far from
   its null on clean entities.
2. $\kappa_m$ is small in strip geometry even with constant appearance.
3. Real faults lie outside the capture range and profiles have no distinct minimum.
4. The image spring covers only surface where the prior is irrelevant, and harmful faults lie in the
   blind rows.
5. Marginalising poses removes most of the image information of typical entities.
6. An existing method already stores calibrated information by source and tests between sources.

## 12. Not Known

- The dispersion of the image family on aerial imagery and its stability.
- The retained fraction in real strip geometry.
- The capture range in pixels of parallax for colour and for features.
- The information price of pose marginalisation for realistic pose priors.
- How much surface, by area and by harm, has an image spring.
- Whether the rasteriser library supports the needed channels and differentiation modes; it is not
  installed in the inspected environment.
