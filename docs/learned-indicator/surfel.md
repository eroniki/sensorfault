# The Fault Indicator on Surface Entities: Masklets, the Depth-Stabilised Clip, and the Surfel Map

## 0. Scope, notation, and summary of findings

The setting is posed aerial video in strips, one monocular depth map per frame, and a
depth-supervised 2D Gaussian surfel reconstruction. Depth maps fail regionally, and neighbouring
frames repeat the failure. The indicator is built on frozen backbones that are native to video and to
dense prediction: a video segmenter that follows regions through a clip (masklets), and a video
encoder with spatiotemporal tokens. No backbone is fine-tuned; the learned part is a light probe.

This note works out the part that lives on the surface: what the unit of judgement is once a masklet
is lifted onto the surfel map, what error model a masklet supports, how evidence is formed from the
motion of a masklet and from a frozen video encoder, how that evidence relates to depth consensus, and
how the verdict enters the fit. Notation follows `docs/learned-indicator.md` (inverse depth $\xi$,
monocular error $\epsilon_M$, parallax gap $r_T$ with information $\Lambda_T$, ancillaries $A_\perp$,
$A_\parallel$, repetition coefficient $\varrho$, parallax share $\varsigma$, harm $h$). Additions:

| Symbol | Meaning |
| --- | --- |
| $S$ | a surface entity: a masklet lifted onto surfels (Section 1) |
| $M_{S,i}(u)$ | mask of $S$ in frame $i$ |
| $\boldsymbol\psi(u) = [u_1, u_2, 1]^\top$ | affine basis in the pixel coordinates of the reference view |
| $\boldsymbol\Delta_S \in \mathbb R^3$ | common error of the fused surface on $S$, affine in inverse depth |
| $\boldsymbol\delta_{Si}$ | deviation of frame $i$ from the common error |
| $\mathbf b_j(u)$ | parallax gain: image motion in frame $j$ per unit inverse depth at reference pixel $u$ |
| $V^{\mathrm{st}}$ | depth-stabilised clip (Section 3) |
| $o_c \in [0,1]$ | ownership weight of a contour sample $c$ |
| $\Lambda^{\partial}_S$, $\Lambda^{\phi}_S$, $\Lambda^{T}_S$ | information on $\boldsymbol\Delta_S$ from contour, latent and point-track readouts |
| $w_{uk}$, $n_{ik}$ | compositing weight of surfel $k$ at pixel $u$; $n_{ik} = \sum_{u\in i} w_{uk}$ |
| $(\Lambda_{ik}, \eta_{ik})$ | lumped depth message of frame $i$ to surfel $k$ |

Nothing in this note has been measured. Statements about behaviour are predictions.

Findings.

1. **The unit is a surface entity with a three-parameter error, not a plane.** A masklet lifted
   through the compositing weights is a set of surfels. The error model it supports is "the monocular
   shape is right and its placement is wrong": an error that is affine in inverse depth over the
   entity. A planar region is the special case. The assumption concerns the error, not the surface,
   so it covers curved roofs and terrain.
2. **Witness and consensus are complementary by construction.** At an entity the frame errors split
   into their information-weighted mean and their contrasts. Depth consensus tests the contrasts and
   is blind to the mean; a witness applied to the *fused* surface tests the mean and nothing else.
   Under the Gaussian model the two are independent and together complete. "The verdict belongs to the
   masklet, per-frame deviations are second order" becomes an identity: frame error equals witnessed
   mean plus consensus contrast. One witness per entity suffices.
3. **Stabilise, then look for motion.** Warping every frame into a reference view through the fused
   surface gives a clip that is static wherever the surface is right. Residual motion is the depth
   error times the parallax gain. This removes the dependence on frame spacing: a survey sequence
   that shifts by a fifth of the image per frame and a dense video both become small-motion clips,
   which is what video segmenters and video encoders are built for.
4. **Latent distance is not evidence; the location of its minimum is.** A frozen encoder's distance
   between an observed and a predicted clip has no units and responds to warping artefacts. Sweeping
   the three error parameters and taking the minimiser gives an estimate in units of parallax, whose
   variance comes from split halves of the clip. The explicit parallax gap is the same estimator with a
   point tracker as readout, and calibrates it.
5. **A masklet outline carries parallax only where the entity owns it and only across the epipolar
   direction.** Contour segments that belong to a nearer occluder report the occluder's depth, and
   segments parallel to the flight line report nothing. Both are computable weights, not caveats.
6. **Latent evidence does not see inside a featureless region either.** Warping a uniform interior by
   a wrong depth produces the same image. On water and glass the information comes from the outline
   and from whatever the interior reflects, and the second is a virtual image. Disagreement between
   the contour estimate and the interior estimate of one entity is the signature of a reflecting or
   transparent surface.
7. **The blind regime** is an entity that is featureless inside and whose owned outline is absent,
   cut by the image border, or parallel to the flight line, or whose monocular shape is itself wrong.

## 1. The unit: a masklet lifted onto the surface

### 1.1 Lifting

A video segmenter returns, for a region it follows, masks $M_{S,i}(u)$ in the frames $i$ where it sees
the region. With the pooling operator of the surfel map, the membership of surfel $k$ in the entity is

```math
m_{Sk} = \frac{\sum_i \sum_{u \in i} w_{uk}\, M_{S,i}(u)}{\sum_i n_{ik}} \in [0, 1] ,
```

and the entity is the set of surfels with $m_{Sk}$ above one half, with the per-frame memberships
$m_{Sk}^{(i)} = (\mathsf P_i M_{S,i})_k$ kept for diagnostics. Three properties matter.

- *Two associations, compared.* The segmenter associates frames by appearance and memory, without
  depth. The surfel map associates frames by geometry. If both are right, the masks of all frames
  land on the same surfels and $m^{(i)}_{Sk}$ agrees across frames. Disagreement is the masklet form
  of parallax evidence (Section 4.1). The segmenter's association must therefore stay depth-free; a
  3D segmentation obtained by clustering masks *through* the surfel map would use the judged depth
  to define the unit.
- *Masklets are not a partition.* Promptable segmenters return overlapping regions at several
  granularities and leave pixels uncovered. Entities are taken at the coarsest granularity that
  passes the tests of Section 4.4, a surfel may belong to nested entities, and uncovered surface
  falls back to fixed ground tiles as entities.
- *Resolution.* Pooling and rendering are adjoint under the measure $n_{ik}$ and their round trip
  blurs at footprint width. Entity boundaries on the surfel map are no sharper than a footprint, so
  contour evidence is formed in the image and only its result is stored on the entity.

### 1.2 The error model of an entity

A monocular network is ambiguous up to scale and shift, and its regional failures are, to a first
approximation, a region with the right shape in the wrong place: a roof floating above or sunk below
its true height, a water surface at the depth of what it reflects. In inverse depth a relative
placement error of a region is low order. The model is

```math
\hat\xi_{M,i}(u) - \xi(u) = \boldsymbol\psi(u)^\top \bigl(\boldsymbol\Delta_S + \boldsymbol\delta_{Si}\bigr) + \epsilon_i(u), \qquad u \in M_{S,i},
```

in the coordinates of a reference view, with $\boldsymbol\Delta_S$ common to all frames,
$\boldsymbol\delta_{Si}$ a per-frame deviation, and $\epsilon_i$ noise with the nominal variance.
Three parameters describe an offset and two gradients of the inverse-depth error.

This sharpens the statement that a region's motion "fixes its plane". A plane is affine in inverse
depth, so for a planar entity whose monocular depth is also planar the model is exact. But the three
unknowns are those of the *error*. For a curved entity the warp uses the monocular shape and estimates
its misplacement; planarity of the surface is not required. What is required, and not guaranteed, is
that the monocular shape is right within the entity. Whether true monocular errors are well described
as affine in inverse depth per masklet is the first thing to measure (Section 9), because it tests the
choice of unit and of error model at once and needs only single-frame masks and exact depth.

The fault state is a property of the entity. With the nominal covariance $C_S$ of
$\hat{\boldsymbol\Delta}_S$, the entity is faulty when $\boldsymbol\Delta_S$ is displaced by more than the
fault definition at some point of the region, and a frame is additionally faulty when its
$\boldsymbol\delta_{Si}$ is.

## 2. Mean and contrasts: what consensus tests and what a witness tests

Express the error of each frame on the entity in the three-dimensional sub-block that depth observes
(normal offset and tilts of the surfels of $S$ tied together, or equivalently
$\boldsymbol\psi$-coefficients). Let $\boldsymbol\epsilon_{Si} = \boldsymbol\Delta_S + \boldsymbol\delta_{Si} + \text{noise}$ be the
error of the message of frame $i$, with lumped information $\Lambda_{Si} = \sum_{k \in S} \Lambda_{ik}$ restricted
to the sub-block, and $\Lambda_S = \sum_i \Lambda_{Si}$. The fused surface carries the information-weighted mean,

```math
\bar{\boldsymbol\epsilon}_S = \Lambda_S^{-1} \sum_i \Lambda_{Si}\, \boldsymbol\epsilon_{Si} ,
```

and every consensus statistic, with or without leave-block-out, is a function of the contrasts
$\boldsymbol\epsilon_{Si} - \bar{\boldsymbol\epsilon}_S$ (the messages at a surfel sum to zero at the fit). The contrasts
are invariant under $\boldsymbol\epsilon_{Si} \to \boldsymbol\epsilon_{Si} + \mathbf c$ for all $i$: a common error is
exactly what they cannot contain. Two statistics exhaust the information:

```math
Q_S = \sum_i (\boldsymbol\epsilon_{Si} - \bar{\boldsymbol\epsilon}_S)^\top \Lambda_{Si}\, (\boldsymbol\epsilon_{Si} - \bar{\boldsymbol\epsilon}_S) \sim \chi^2_{3(n_S - 1)},
\qquad
T_S = \mathbf r_S^\top \bigl(\Lambda_S^{-1} + C^{W}_S\bigr)^{-1} \mathbf r_S \sim \chi^2_3 ,
```

where $\mathbf r_S = \bar{\boldsymbol\epsilon}_S - \boldsymbol\epsilon^{W}_S$ is the gap between the fused surface and an external
witness with error covariance $C^W_S$. $Q_S$ is computable from the depth maps alone. $T_S$ needs
something that measures the fused surface against the world. Under the Gaussian model with these
weights the mean and the contrasts are independent, so the two tests do not share noise.

Consequences for the design.

1. *The witness is applied to the fused surface, once per entity.* It does not need to judge each
   frame's depth map. The estimate of a frame's error is assembled:
   $\hat{\boldsymbol\epsilon}_{Si} = \hat{\bar{\boldsymbol\epsilon}}_S + (\boldsymbol\epsilon_{Si} - \bar{\boldsymbol\epsilon}_S)$, witnessed mean plus
   consensus contrast. This is the exact form of a verdict that belongs to the entity with per-frame
   deviations as a correction, and it costs one witness per entity instead of one per frame.
2. *The repetition coefficient is the variance share of the mean.* With
   $\varrho_S = \operatorname{Var}(\boldsymbol\Delta_S)/(\operatorname{Var}(\boldsymbol\Delta_S) + \operatorname{Var}(\boldsymbol\delta_{Si}))$ along any direction,
   $\varrho_S \to 1$ is the fault every frame repeats; the consensus gap carries the fraction
   $1 - \varrho$ of the error and the witness gap carries the mean. The covariance estimator of
   $\varrho$ in the main document is the pixel-level version of this decomposition.
3. *Blocks, not frames, are the independent units of the contrasts.* Frames of a block share their
   deviation; the contrast statistic is formed between blocks, and its degrees of freedom count blocks.
   This is unchanged.

The decomposition relies on the fused surface being the information-weighted mean of the messages.
An optimiser that has not converged, compositing weights that change with the surface, and surfels
that a fault created for itself all break it to some degree; the closed-form fused estimate from the
lumped blocks, not the raw optimiser state, is the reference where this matters.

## 3. The depth-stabilised clip

### 3.1 Construction

Fix a reference view $i$ and a window of frames $j \in \mathcal J$. Render the depth of the fused surface
in the reference view, $\hat D_i(u)$, and form for each $j$ the image of frame $j$ resampled into the
reference view through that depth:

```math
V^{\mathrm{st}}_j(u) = I_j\bigl(\mathbf p_j(u, \hat\xi_i(u))\bigr), \qquad \hat\xi_i(u) = 1/\hat D_i(u),
\qquad \text{valid}_j(u) = \bigl[\ \lvert \hat D_j(\mathbf p_j) - z_j(u) \rvert < \text{tol}\ \bigr] ,
```

with $\mathbf p_j$ the projection into frame $j$, $z_j(u)$ the depth of the back-projected point in
frame $j$, and the validity mask an occlusion test against the surface rendered in frame $j$. Invalid
pixels are filled with a neutral value and their tokens are excluded; filling them from the reference
frame would manufacture evidence of stillness. The surfel map supplies what a per-pixel warp cannot:
occlusion-aware visibility through compositing, and one surface for all frames.

If the surface is right and the scene is static and diffuse, $V^{\mathrm{st}}$ shows no motion. If the
inverse depth is wrong by $\boldsymbol\psi(u)^\top \boldsymbol\Delta_S$, the content at $u$ moves by

```math
\mathbf m_j(u) = \mathbf b_j(u)\, \boldsymbol\psi(u)^\top \boldsymbol\Delta_S ,
```

along the epipolar direction and in proportion to the baseline of frame $j$. Evidence is any motion in
the stabilised clip that has this form. This is the classical plane-plus-parallax decomposition
(residual motion after registering a dominant plane; recalled from the motion-analysis literature)
with the plane replaced by the surface under test, so that the residual is the error and not the
relief.

### 3.2 Why stabilise

- *Frame spacing stops mattering.* A survey sequence with 80% forward overlap displaces the image by a
  fifth of its width per frame; a dense video by a few pixels. After stabilisation both show only the
  residual. As an illustration with assumed values, a frame step that shifts the ground by 800 pixels
  leaves a residual of 16 pixels for a 2% depth error. Video backbones trained on small motion then
  operate in their own regime, whatever the acquisition.
- *The temporal stride becomes a design variable.* The residual grows with baseline. Subsampling the
  stabilised clip so that the smallest fault of interest moves by a few pixels over the window, about
  the scale the readout resolves, is a matched filter; several strides cover several fault sizes.
- *Association becomes easy.* A video segmenter run on the stabilised clip follows regions that
  barely move. Its association is still by appearance; the depth only removed the bulk motion.
- *A static reference exists.* The reference frame repeated over the window is the clip the surface
  predicts. Evidence is distance from static, which a video encoder represents natively.

Costs: resampling stretches content seen obliquely, changes its sharpness, and leaves holes at
disocclusions. These artefacts are present whether or not the depth is right, which is why raw
distances are not used as evidence (Section 4.2).

### 3.3 Which depth

Stabilising with the fused surface tests the mean of Section 2, which is the quantity consensus cannot
see. Stabilising with each frame's own monocular depth would test that frame's message; it costs one
clip per frame and repeats what the contrasts already provide. The fused surface is the default; the
per-frame form is the fallback where the fused surface is not a usable summary (surfels that only one
block supports, semi-transparent blends of two depths).

## 4. Evidence on an entity

Three readouts estimate the same $\boldsymbol\Delta_S$ from the stabilised clip. Each is an M-estimator
with its own information and its own failure cases.

### 4.1 Contour readout: the motion of the mask

Sample the outline of $M_{S,i}$ at points $c$ with unit normals $\mathbf n_c$, spaced at the correlation
length of the segmenter's boundary error. In stabilised frame $j$ the observed outline lies at signed
normal distance $g_{jc}$ from the reference outline. Under the error model,

```math
g_{jc} = \mathbf n_c^\top \mathbf b_j(c)\ \boldsymbol\psi(c)^\top \boldsymbol\Delta_S + \varepsilon_{jc}, \qquad
\Lambda^{\partial}_S = \sum_{j} \sum_{c} o_c\, \frac{\bigl(\mathbf n_c^\top \mathbf b_j(c)\bigr)^2}{\sigma_c^2}\ \boldsymbol\psi(c)\, \boldsymbol\psi(c)^\top ,
\qquad
\hat{\boldsymbol\Delta}^{\partial}_S = \bigl(\Lambda^{\partial}_S\bigr)^{-1} \sum_{j,c} o_c\, \frac{\mathbf n_c^\top \mathbf b_j(c)}{\sigma_c^2}\, \boldsymbol\psi(c)\, g_{jc} .
```

A featureless interior contributes nothing and is not needed. The formula makes three restrictions
explicit.

- *Aperture.* Only the component of motion normal to the outline is observed, and motion is along the
  epipolar direction. A segment parallel to that direction has $\mathbf n_c^\top \mathbf b_j = 0$ and carries
  no information. In a straight strip the epipolar direction is the flight direction for all pixels,
  so the edges of an entity that run along the flight line are silent, and an elongated entity
  aligned with the strip (a road, a canal) is poorly determined. A cross strip supplies the missing
  direction.
- *Rank.* $\Lambda^\partial_S$ has rank three only if the informative outline spans two dimensions in the
  image. For informative samples spread evenly on a ring of radius $R$, the predictive variance of
  the error at a point at distance $\rho$ from the centroid is $\frac{1}{\Lambda}(1 + 2\rho^2/R^2)$ with
  $\Lambda$ the total scalar information: inside the outline it is at most three times the variance at
  the centroid. An entity cut by the image border loses the outline on that side and the bound with it.
- *Ownership.* An outline is the image of a surface boundary only if the entity owns it. Three cases:
  an attached boundary on one surface (a shoreline at water level, a pane in its frame, a marking)
  moves with the entity; the edge of an entity that occludes what lies behind it (a roof against the
  ground) also moves with the entity; but where a nearer object hides part of the entity (ground
  beside a building, water under overhanging trees) the outline is the occluder's edge and moves with
  the occluder's depth. The weight $o_c$ is one where the depth step across the outline, rendered from
  the surfel map on both sides, is small or has the entity on the near side, and zero otherwise. Only
  the sign and rough size of the step are used, so the judged depth enters weakly. Ignoring ownership
  biases the estimate toward the occluder: a tree-lined lake is placed at canopy height.

The boundary error $\sigma_c$ of the segmenter is not taken from the model. The component of outline
displacement across the epipolar direction does not depend on depth, and its spread estimates
$\sigma_c$ on the scene, as the across-line residual does for a point track.

### 4.2 Latent readout: a sweep, not a distance

Let $\Phi$ be a frozen video encoder with tokens $\tau$, and let $V^{\boldsymbol\Delta}$ be the stabilised clip
with the additional warp $-\mathbf b_j(u)\, \boldsymbol\psi(u)^\top \boldsymbol\Delta$ applied inside the entity, which is what
stabilisation would have produced had the surface been corrected by $\boldsymbol\Delta$. Define

```math
D_S(\boldsymbol\Delta) = \sum_{\tau \in S} \omega_\tau\, \bigl\lVert \Phi_\tau\bigl(V^{\boldsymbol\Delta}\bigr) - \Phi_\tau\bigl(V^{\mathrm{ref}}\bigr) \bigr\rVert^2,
\qquad
\hat{\boldsymbol\Delta}^{\phi}_S = \arg\min_{\boldsymbol\Delta} D_S(\boldsymbol\Delta) ,
```

with $V^{\mathrm{ref}}$ the static reference clip and $\omega_\tau$ the mask and validity coverage of a
token. The readout is the minimiser. The value $D_S(\mathbf 0)$, the raw "violation of expectation",
is not used as evidence, for three reasons: it has no units; it is raised by resampling artefacts,
exposure and view-dependent appearance wherever the depth is right; and its scale differs between
textures. These nuisances shift $D_S$ by an amount that varies slowly with $\boldsymbol\Delta$, so they
largely cancel in the location of the minimum. That cancellation is an assumption: the estimator is
consistent if the expected distance from static is minimised at the true error. It is checked where a
calibrated readout exists (Section 4.3).

*Procedure.* A seven-point design (centre and $\pm$ one step along each parameter) gives the gradient
and the diagonal of the curvature by finite differences; two or three Gauss-Newton iterations with
step sizes matched to a few pixels of residual motion suffice if the response is smooth. All entities
of a clip are perturbed in the same passes, each with its own $\boldsymbol\Delta$; attention couples
neighbouring entities, and alternating the signs of the perturbations between passes separates an
entity's own response from its neighbours'.

*Variance.* The curvature of $D_S$ is in latent units and is not an information. The error covariance
of $\hat{\boldsymbol\Delta}^\phi_S$ is estimated from split halves: the frames before and after the reference
view (or odd and even frames) give two estimates from disjoint images, and half their squared
difference, pooled over entities of a stratum and regressed on the geometric information
$\sum_j \lVert \mathbf b_j \rVert^2$, gives a variance model. The two halves also provide the third measurement
that identifies the error laws without labels, as the half-tracks do in the main document.

*What the latent readout cannot do.* It does not see inside a uniform region: a wrong depth warps a
featureless interior onto itself. Its information on water, glass and blank roofs comes from tokens
that straddle the outline, which is the contour readout through a different instrument, and from
whatever structure the interior shows, which on these materials is a reflection. The claim for the
latent readout is therefore robustness (illumination, weak and repetitive texture read with context),
density, and freedom from a tracker, not sight where there is nothing to see. Its resolution is set
by the token grid: with patches of 16 pixels, residual motion well below a patch may leave the tokens
nearly unchanged, and the stride of Section 3.2 has to bring the residual up to that scale. Whether
the frozen tokens respond smoothly and monotonically to sub-patch motion is unknown and decides the
usefulness of this readout.

### 4.3 Point readout: the calibrated special case

A point tracker or flow model applied to the stabilised clip measures the residual motion directly.
For a point $u$ with displacement $\mathbf q_j - u$ in stabilised frame $j$, the along-line component over
the gain is the parallax gap of the main document, $r_T$ with information
$\Lambda_T = \sum_j b_j^2/\sigma_j^2$, and the across-line component and the along-line misfit are its
ancillaries $A_\perp$ and $A_\parallel$. Pooled over the entity,

```math
\Lambda^{T}_S = \sum_{t \in S} \Lambda_{T,t}\ \boldsymbol\psi(u_t)\, \boldsymbol\psi(u_t)^\top, \qquad
\hat{\boldsymbol\Delta}^{T}_S = -\bigl(\Lambda^{T}_S\bigr)^{-1} \sum_{t \in S} \Lambda_{T,t}\ \boldsymbol\psi(u_t)\, r_{T,t} .
```

The sign follows the convention $r_T = \epsilon_M - \epsilon_T$ applied to the fused surface. This
readout has an error bar from geometry and a noise scale estimated from its own ancillaries. Its role
is to be the reference on textured entities: there, $\hat{\boldsymbol\Delta}^\phi_S$ and
$\hat{\boldsymbol\Delta}^\partial_S$ are compared with it, and their bias and their split-half variance model are
accepted or rejected. A readout that fails this comparison on textured surface is not trusted on
untextured surface. On the stabilised clip the tracker also works in the small-motion regime.

### 4.4 Tests an entity must pass, and what they cannot see

The ancillary statistics of a track have entity-level counterparts, none of which depends on the
depth error:

- outline motion across the epipolar direction (rigidity of the scene and stability of the
  segmenter's boundary);
- the misfit of the outline displacements after estimating $\boldsymbol\Delta_S$, with the number of informative
  samples times frames minus three degrees of freedom (adequacy of the three-parameter model on the
  boundary);
- agreement of the split halves;
- agreement between frames of $m^{(i)}_{Sk}$ after correction by $\hat{\boldsymbol\Delta}_S$ (the two associations
  of Section 1.1 coincide).

An entity that fails is split at a finer granularity or abstains. An error of the witness that is
proportional to the parallax gain passes all of them, exactly as for a single track. The cases carry
over and one is added:

- *Reflections.* The mirror image of a static scene in still water or flat glass is rigid and lies
  behind the surface. Interior structure of such an entity moves consistently in the stabilised clip
  and reports the virtual depth.
- *Motion along the strip.* A vehicle on a road parallel to the flight line produces residual motion
  proportional to the baseline.
- *Borrowed outline.* An unowned contour segment that is given weight reports the occluder.
- *Appearance-defined outlines.* Where the boundary of a mask is drawn by something that changes with
  the viewpoint (the edge of a reflected treeline on a lake, a glint, a shadow that moved between
  strips), the outline moves without the surface. A segmenter can also return the reflection as an
  entity of its own, which is then a consistent rigid entity at the virtual depth.

### 4.5 Combining the readouts, and the reflection signature

Where the readouts have independent errors their information adds. They are not independent in
general: all read the same pixels. The combination therefore uses the point readout where
$\Lambda^T_S$ is adequate, the contour readout otherwise, and the latent readout as a third estimate
whose weight comes from its measured split-half variance.

More useful than their sum is their difference. The contour readout uses owned and attached
boundaries, which are real structure on the surface. The interior readouts (point or latent,
restricted to tokens away from the outline) use whatever the interior shows. On a diffuse surface they
agree. On a reflecting or transparent surface the interior reports a depth behind the surface while
the outline reports the surface:

```math
\mathbf d_S = \hat{\boldsymbol\Delta}^{\mathrm{int}}_S - \hat{\boldsymbol\Delta}^{\partial}_S, \qquad
T^{\mathrm{refl}}_S = \mathbf d_S^\top \bigl(C^{\mathrm{int}}_S + C^{\partial}_S\bigr)^{-1} \mathbf d_S ,
```

with the sign of the offset component of $\mathbf d_S$ telling behind from in front. A significant
difference toward greater depth marks the entity as reflecting or transparent. Its interior evidence
is then discarded as a judge of the monocular depth, the verdict rests on the contour, and the mark
itself is a material attribute obtained without labels, which is an input the probe can use. Sky
reflected in water is the extreme case: the interior reports an inverse depth near zero. An isolated
point that departs from a well-determined entity, on a road, is the mover case. Without an owned
outline the conflict between an interior estimate and the monocular depth cannot be attributed, and
the entity is labelled unknown.

## 5. Harm and detectability at the level of the entity

Let $\Lambda^M_S$ be the information of the monocular messages on the offset of the entity and
$\Lambda^W_S$ that of the combined witness. With the parallax share
$\varsigma_S = \Lambda^W_S/(\Lambda^W_S + \Lambda^M_S)$, a common error $\Delta$ would move a surface fused from
both by $h = \Delta(1 - \varsigma_S)$, and the noncentrality of the test $T_S$ is

```math
\delta^2 = \Delta^2\, \Lambda^M_S\, \varsigma_S = h^2\, \Lambda^M_S\, \frac{\varsigma_S}{(1 - \varsigma_S)^2} .
```

For a given harm, detectability rises steeply with the share and vanishes with it. The entity changes
the accounting in one respect: $\Lambda^W_S$ now includes the contour, so a featureless entity with an
owned, two-dimensional outline across the flight direction is no longer at $\varsigma = 0$. The blind
regime is the set of entities for which all of $\Lambda^\partial_S$, $\Lambda^\phi_S$, $\Lambda^T_S$ are small or
untrusted:

- featureless interior, and outline that is unowned, cut by the image border, or parallel to the
  flight line;
- reflecting interior with no owned outline;
- entities whose monocular *shape* is wrong, so that the three-parameter error model does not hold
  and the misfit test rejects, or, worse, does not;
- entities smaller than a few tokens or a few contour correlation lengths.

Open water running to the image border and a glazed facade without visible frames remain there. The
objection to keeping the monocular prior has the same answer as before, in entity terms: where the
witness share is high the prior is redundant and its faults are easy to find; the prior earns its
place on entities with a low share, and the method is worth more than a multi-view check only through
what it does for those: the contour, and the probe of Section 6. Evaluation is weighted by
$1 - \varsigma_S$ or reported by regime.

## 6. Tokens on the surface, the probe, and the transfer

### 6.1 Lifting tokens

Token features lift like any pixel field. With bilinear weights $b_{u\tau}$ from the token grid of a
clip to pixels of frame $i$,

```math
\boldsymbol\phi_{ik} = \frac{1}{n_{ik}} \sum_\tau \Bigl(\sum_{u \in i} w_{uk}\, b_{u\tau}\Bigr) \boldsymbol\phi_\tau,
\qquad
\bar{\boldsymbol\phi}_S = \frac{\sum_{k \in S} \sum_i n_{ik}\, \boldsymbol\phi_{ik}}{\sum_{k \in S}\sum_i n_{ik}} .
```

The fused feature is the lumped least-squares solution of feature-field distillation, one
majorisation step by the same Jensen argument that justifies lumped depth messages (weights
$w_{uk}$, not $w_{uk}^2$). A token of 16 pixels and two frames is coarser than a surfel footprint; the
lifted field has token resolution. The entity feature $\bar{\boldsymbol\phi}_S$ averages over many
tokens and is where the coarseness matters least, which is a further reason for the entity as unit.

### 6.2 Keeping the probe from copying the witness

Tokens of a video encoder encode motion. A probe on tokens of the stabilised clip would read the
residual motion, that is, the evidence itself, reflections included; a probe on tokens of the raw clip
would read the raw parallax, from which the same can be inferred. The separation is made by the input
clip, with one backbone in two modes:

- *Prior.* The probe reads tokens of a clip with no parallax in it: the reference frame alone, in the
  encoder's image mode or repeated over the window. It sees appearance and nothing of the witness.
- *Evidence.* The stabilised clip enters only through the readouts of Section 4.

A third input is admissible: the difference between tokens of the stabilised clip and of the static
clip on entities that *passed* as correct, pooled per stratum. It describes how a material's
appearance changes with viewpoint when geometry is right, which is material information and not
evidence about a particular entity. The material mark of Section 4.5 is admissible for the same
reason, with the caution that it is produced by the witness and must not be an input where the
target is the witness's own verdict on the same entity.

The probe is a linear or two-layer map on $[\boldsymbol\phi_{ik}, \bar{\boldsymbol\phi}_S, \text{geometry scalars}]$
(range, incidence, entity size, witness share), trained toward the posterior of the entity-level
verdict with a weight equal to its label information, and cross-fitted by ground area: an entity is
scene-fixed and seen from many frames, so the probe applied to an entity is fitted on entities of
other ground tiles. The unit that counts for sample size and calibration is the entity, not the pixel.

### 6.3 Strata without labels

A promptable segmenter that accepts text returns the instances of a named concept. Prompts such as
water, glass roof, solar panel or skylight give material strata and candidate entities without fault
labels. Two limits: the vocabulary is a human choice, which is a prior on where faults live and must
be fixed before any evaluation; and the strata cover only what was named, so an unprompted stratum
formed by clustering $\bar{\boldsymbol\phi}_S$ remains necessary. Behaviour of concept prompts on nadir
aerial imagery is unverified.

### 6.4 The transfer

Let $e(\boldsymbol\phi)$ be the probability that an entity with features $\boldsymbol\phi$ receives an
informative verdict. If informativeness is determined by what the features encode, verdicts are
missing at random given $\boldsymbol\phi$ and the probe transfers wherever $e(\boldsymbol\phi) > 0$; where
$e = 0$ it extrapolates with an error that the video cannot estimate. The entity moves mass in three
ways. Contour evidence gives verdicts on featureless entities with owned outlines, so water, glazing
and blank roofs are no longer at $e = 0$ as classes. These verdicts are of the *kind* of fault that
matters there (misplacement of a featureless region), which protects against a probe that learns only
the faults of textured surface. And the transfer becomes testable on the video itself: train the probe
on entities judged by the point readout, test it on entities judged by the contour readout. What
remains at $e = 0$ is the blind regime of Section 5, and for it a synthetic scene with exact depth is
the only examiner.

## 7. Use in reconstruction

The indicator delivered to the fit is a pixel field, assembled from the entity and frame terms:
$\zeta_i(u)$ is constant over $M_{S,i}$ up to the frame deviation, and falls back to the probe where no
entity verdict exists. Gates enter the lumped messages,
$\Lambda_{ik}[w_i] = c_{\mathrm{eff}} \sum_u w_i(u)\, a_i^2 w_{uk}\, \mathbf j_{uk}^\top \mathbf j_{uk}/\sigma_i^2(u)$ with
$w_i = 1 - \zeta_i$, and gated fusion is closed form to first order. The lumped form is required
because it majorises the Hessian, so a gating step cannot overshoot; blocks built from squared
weights overshoot the common mode by $\sum_u w_{uk}/\sum_u w_{uk}^2$.

- *Repair.* An entity with a trusted estimate $\hat{\boldsymbol\Delta}_S$ and covariance $C_S$ does not have
  to be gated to nothing. The correction is a message to its surfels in the same sub-block, with
  variance $\boldsymbol\psi(u)^\top C_S\, \boldsymbol\psi(u)$ mapped to the normal offset, and it replaces the gated
  monocular messages. Detection becomes repair for entities with an informative witness. This is
  reported separately from detection.
- *Unconstrained regions.* An entity that is gated on the probe's word alone has no replacement. The
  fit reports it as unconstrained.
- *Refit.* A warm-started fit with the weight map frozen for the round. Closed-form fusion holds
  compositing weights and the set of surfels fixed and cannot predict what appears when a surfel
  loses its support.
- *Nuisance parameters.* Per-frame scale and offset and the shared distance bias are fitted with
  flagged entities down-weighted. Where point readouts exist they anchor the alignment independently
  of the surface. A view-wide residual in the stabilised clip, across the epipolar direction, is a
  pose or timing error of that frame and is removed at view level before entities are judged.
- *One pass.* The stabilised clip depends on the surface, so after a refit that moves an entity the
  evidence can be recomputed; a second pass is a check that the residual motion is gone, not a
  requirement.

## 8. Frozen models: what was checked and what is recalled

**Video encoder.** Checked against the public repository and paper pages: V-JEPA 2 is released with
ViT-L (300M parameters), ViT-H (600M) and ViT-g (1B) at resolution 256 and ViT-g at 384; V-JEPA 2.1
with ViT-B (80M), ViT-L (300M), ViT-g (1B) and ViT-G (2B) at 384; the code is mostly under the MIT
licence. V-JEPA 2.1 (arXiv 2603.14482) states dense features as its aim, through a dense predictive
loss, self-supervision at intermediate layers, and tokenizers for both images and video, and reports
a linear probe for depth among its evaluations. The action-conditioned model is described as
post-trained on robot trajectories. Not checked: the tubelet size (recalled as two frames by
$16 \times 16$ pixels), the number of frames per clip, the shape of the token output, memory use, and
any result on aerial imagery. Consequences for the design: there is, as far as checked, no released
predictor conditioned on camera pose, so "what the geometry predicts" cannot be asked of the model in
latent space and is constructed in pixel space by the stabilised clip; the image tokenizer of
V-JEPA 2.1 is what the two-mode use of Section 6.2 relies on; the ViT-B and ViT-L models are the ones
plausible on a small GPU, which is an estimate from parameter counts.

**Video segmenter.** Checked: SAM 3 (arXiv 2511.16719) detects, segments and tracks in images and
video, accepts short text phrases, exemplar boxes, points and masks as prompts, has 848M parameters,
is under its own licence with access-gated checkpoints, and has a successor release for joint tracking
of many objects. Not checked: the terms of the licence, input resolution, memory use with hundreds of
entities per clip, whether an exhaustive "segment everything" mode exists for video, boundary
accuracy, behaviour under large inter-frame displacement, and behaviour on nadir aerial imagery. The
word masklet for a region followed through a clip is recalled from the preceding model's description.
Running the segmenter on the stabilised clip removes the dependence on large displacement but not the
other unknowns.

**Point readout.** A long-range point tracker (CoTracker3, arXiv 2410.11831: public, with per-point
visibility and an online mode, mostly non-commercial licence) or a dense tracker (AllTracker, arXiv
2506.07310: dense correspondence over hundreds of frames, 16M parameters, reported at 768 by 1024 on
a 40 GB GPU, so reduced resolution or tiles would be needed on a small one). These facts were checked
on the papers' and repository pages; their accuracy on stabilised aerial clips was not.

**Cost.** Per reference view, the stabilised window is cut into crops at the encoder's resolution; at
half the native resolution of a 12-megapixel frame that is about twenty crops. The seven-point sweep
with two iterations is about fifteen encoder passes per crop, a few hundred passes per reference view,
with reference views every few frames. The segmenter runs once per stabilised window. The contour and
point readouts, the lifting and the entity algebra are accumulations of $3\times 3$ blocks and are
negligible. These are estimates from sizes; nothing was run.

## 9. Contribution, nearest prior work, kill tests, and what is unresolved

**What is sharp here.**

1. *The mean-contrast decomposition with a witness on the fused surface* (Section 2). It says exactly
   what depth consensus cannot see, shows that one witness per surface entity completes it, and turns
   the entity-level verdict from a design preference into an identity. The algebra is that of a
   one-way random-effects model; the contribution is identifying the fused surface, not each depth
   map, as what the video should judge.
2. *The depth-stabilised clip with a sweep readout* (Sections 3 and 4.2). It makes frozen
   video-native backbones usable on survey-spaced imagery, converts an uncalibrated latent distance
   into an estimate in units of parallax with a variance from split halves, and puts the explicit
   parallax gap, the mask outline and the latent encoder on one footing as three readouts of one
   three-parameter quantity.
3. *Ownership-weighted contour parallax and the contour-interior difference* (Sections 4.1, 4.5):
   parallax evidence for featureless regions, and a label-free signature of reflecting surfaces.

**Nearest prior work**, all recalled and to be checked before any claim of novelty: plane-plus-parallax
motion analysis, of which the stabilised clip is a direct descendant; photometric reprojection as a
training loss for self-supervised monocular depth, and feature-metric plane sweeping in learned
multi-view stereo, which use the same warp to *estimate* depth where this uses it to *test* a prior
and to locate its error; planar priors from sparse triangulated points for textureless regions in
multi-view stereo; lifting of segmentation masks and features onto Gaussian splats; violation of
expectation measured in the latent space of a video-prediction model; region-level scale and shift
correction of monocular depth with segmentation masks. One closely related paper was found by search,
"In Depth We Trust: Reliable Monocular Depth Supervision for Gaussian Splatting" (arXiv 2604.05715),
which by its abstract applies monocular depth regularisation selectively by isolating ill-posed
geometry; only its abstract was read. If region-wise affine correction of monocular depth from
multi-view evidence is already standard, items 2 and 3 reduce to a careful instance of it and the
claim rests on item 1 and on the indicator's treatment of the blind regime.

**What would kill it, cheapest first.** Exact depth from a synthetic scene is used for evaluation only.

1. *Is the error model right?* Segment single frames, fit the true inverse-depth error of the
   monocular depth by an affine function per mask, and report the share of fault energy explained,
   by material. If true faults are not affine per mask (they cut across masks, or are errors of shape),
   the unit and the three-parameter model are wrong. Needs masks and exact depth only.
2. *Does the encoder respond to residual parallax?* Stabilise clips with exact depth, inject a known
   affine inverse-depth error on a region, and measure bias and spread of the sweep minimiser against
   injected size, texture and stride. A flat or non-monotone response below one patch of motion
   removes the latent readout; the contour and point readouts would remain.
3. *Does the outline carry it?* On featureless regions with exact depth: the fraction of outline that
   is owned and across the flight direction, the boundary noise of the segmenter on stabilised clips,
   and the error of the contour estimate. If water and glazing rarely have owned outlines inside the
   frame, the contour moves little mass out of the blind regime.
4. *Reflections.* How often the interior of glass and water entities yields a consistent estimate
   behind the surface, and whether the contour-interior difference detects it.
5. *Redundancy of the prior.* Fits from the witness alone, the monocular prior alone, both, and the
   gated prior, on the harm-weighted metric.

**Unresolved.**

- *The blind regime is smaller, not empty*, and its size on real scenes is unknown: featureless
  entities without owned outlines, and entities whose monocular shape is wrong.
- *Shape errors.* The model judges placement. A region whose monocular shape is wrong but whose
  outline fits passes the boundary tests; only interior evidence can object, and on featureless
  entities there is none.
- *Appearance-defined outlines* move without the surface and satisfy the epipolar form. The across-line
  spread detects unstable boundaries, not ones that slide along the flight direction.
- *Dependence of the readouts.* Contour, latent and point readouts read the same pixels. Their
  agreement on textured surface does not certify the contour readout on featureless surface, where
  the other two are silent.
- *Circularity through the surface.* Stabilisation, ownership and lifting all use the surface under
  test. Each uses it weakly (bulk motion, the sign of a depth step, association at region scale), but
  where a fault created its own surfels or two depths are blended, the reference surface is not a
  usable summary and the per-frame fallback applies.
- *Entities are not given.* Granularity, coverage and temporal stability of masklets on aerial strips
  are assumptions about a model that has not been run on such data.
- *Not verified:* every numerical illustration; all model properties listed as not checked in
  Section 8; the prior-work attributions.
