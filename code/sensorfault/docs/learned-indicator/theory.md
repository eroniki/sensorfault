# Theory: a Map That Stores What It Knows. Information-Form Gaussian Surfels for Correspondence-Free Mapping with a Monocular Depth Prior

This note gives the statistical model behind one claim: a large-area map built from posed aerial video
and one monocular depth map per frame should store, on the same Gaussian primitives, the geometry, the
appearance, and the *information* each source of evidence has contributed about the geometry. With
that third quantity the agreement between geometry and appearance can be measured in exact units, the
depth prior can be tested against the pixels, and nuisance such as pose error can be accounted for
instead of ignored.

The argument comes first, in plain sentences (Section 1). The equations that support each step follow
(Sections 2 to 8), then what is identified, what is learned, and the limits. Nothing here has been
measured. Derivations are first-order and have not been checked numerically. Statements about prior
work are marked as read or recalled in Section 12.

## 1. The Argument

**1. A map is geometry, appearance and information.** Geometry and appearance describe the scene.
Information describes what the data said about it. Standard Gaussian splatting stores the first two.
Opacity is not confidence: an opaque surfel on a blank roof is drawn with the same certainty as one on
a textured wall, although the images constrain the first hardly at all.

**2. Pixels inform geometry only through a coupling.** An image says something about where a surface
is only if moving the surface would change the image. That requires texture contrast, and it requires
views from different directions. The strength of this coupling is a stiffness: how hard the appearance
resists a displacement of the geometry. It is zero on blank surface and zero from a single viewpoint.
So the information belongs to the *pair* of geometry and appearance, and it has to be computed from
the renderer, which is the only place that knows how one depends on the other.

**3. The monocular depth is a second spring on the same surface.** It pulls each surface towards
where the depth network put it, with a stiffness set by the network's accuracy. For a planar surfel
both springs act on the same three coordinates, offset and two tilts. This is not an analogy chosen
for convenience: the derivative of the rendered image with respect to a surfel's placement factors
through the derivative of its depth (Section 2).

**4. Two springs on one body store energy when they disagree.** The stretch is the misfit, the stored
energy is the test statistic, and its unit is the nat. Geometry-appearance alignment is therefore
measurable, with a null distribution (Section 3). This is what the representation is for.

**5. The energy can only be read if the springs are kept apart.** A map that stores one summed
confidence cannot say which source put it there, and cannot test one source against another. The
information must be stored per source. A fault that every depth map repeats is invisible in the
disagreement among depth maps and visible only between depth and pixels; without provenance it cannot
even be stated.

**6. Information must be honest.** A stiffness is a claim about scatter: if the map says an offset is
known to five centimetres, repeated evidence must scatter by five centimetres. Rendering residuals are
dominated by model misfit, not sensor noise, so the claim has to be checked and corrected on the scene
itself, from redundancy the data already contain (Section 5).

**7. Evidence enters by rendering, not by matching.** Each frame is compared with a rendering of the
shared primitives. No features are matched, no tracks are kept, no points are triangulated. Where
texture is weak the information goes smoothly to zero instead of a matcher failing; every pixel
contributes; occlusion is decided once, by the rasteriser. The price is a short capture range: the
comparison is local, and it only works if the surface is already nearly in place.

**8. The monocular prior is what makes item 7 possible.** It puts the surface within reach of the
pixels everywhere, including where nothing could be matched. The pixels then refine it where they are
stiff and audit it where both are comparable. Prior and pixels are not rivals; each covers the other's
weakness, and the information field says, direction by direction, which one is speaking.

**9. Nuisance has three fates.** A disturbance with a known low-dimensional form (a frame's pose
error, a depth map's scale and offset, exposure) is *absorbed*: its directions are removed from the
evidence, and the map stops claiming knowledge in them. A disturbance with no form but a level
(discretisation, resampling) is *deflated*: all stiffness is reduced by a measured factor. A
disturbance that is structured and local (a region of wrong depth, a reflection) can be neither; it
must be *detected*. Absorbing an error never makes it vanish. It converts it into a stated loss of
information, and the art is to know which fate applies (Section 6).

**10. What follows for the design.** Store, per surface entity and per source, an information matrix
and a pull on the three placement coordinates, additive over frames and over map tiles. Compute them
from the differentiable renderer. Calibrate them. Read alignment as energy. Gate a source where the
energy says the two disagree and the attribution says which one is wrong. Use light heads on frozen
features only where the pixels have no stiffness at all, and say so in the map.

## 2. The Two Couplings

### 2.1 Coordinates

Surfel $k$ has centre $\boldsymbol\mu_k$, frame $[\mathbf t_1\ \mathbf t_2\ \mathbf n]$, scales
$(\sigma_1, \sigma_2)$, opacity $\alpha_k$ and appearance $\mathbf a_k \in \mathbb R^C$. Its *placement*
is $\boldsymbol\theta_k = (\nu, \tau_1, \tau_2)$: offset along the normal and two tilts. A *surface
entity* $m$ is a set of surfels $\mathcal K_m$ onto which per-frame segment masks have been fused by
accumulating compositing weights; its placement $\boldsymbol\theta_m \in \mathbb R^3$ moves its surfels
coherently through $P_{km}$.

### 2.2 Depth coupling

The ray of pixel $u$ meets the plane of surfel $k$ at depth $d_k$, with

```math
\mathbf j_{uk} = \frac{\partial d_k}{\partial \boldsymbol\theta_k} = \frac{\bigl[\,1,\ -\mathbf t_1^\top(\mathbf x_{uk} - \boldsymbol\mu_k),\ -\mathbf t_2^\top(\mathbf x_{uk} - \boldsymbol\mu_k)\bigr]}{\mathbf n^\top \mathbf w_u} .
```

The monocular depth is modelled as $\hat d_i(u) = a_i (d_i(u) + \beta d_i(u)^2) + b_i + \varepsilon_i(u)$.
With compositing weights $w_{uk}$ and depth residual $e^D_i$, frame $i$ contributes

```math
\Lambda^D_{ik} = \sum_u w_{uk}\, \frac{a_i^2\, \mathbf j_{uk}^\top \mathbf j_{uk}}{\sigma_i^2(u)}, \qquad
\boldsymbol\eta^D_{ik} = \sum_u w_{uk}\, \frac{a_i\, \mathbf j_{uk}^\top e^D_i(u)}{\sigma_i^2(u)} .
```

### 2.3 Image coupling

The rendered appearance is $\hat{\mathbf c}_i(u) = \sum_k w_{uk} \mathbf a_k$ with
$w_{uk} = \alpha_k G_k(\mathbf s_{uk}) T_{uk}$. Two facts give its derivative with respect to placement.
*Contrast:* a primitive changes the pixel only by its difference from what is composited behind it,
$\partial \hat{\mathbf c}_i(u) / \partial \log G_k = w_{uk} (\mathbf a_k - \mathbf b_{uk})$. *Slide:*
moving the plane along the ray by $\partial d_k$ moves the intersection in the tangent frame by
$\mathbf m_{uk} = [\mathbf t_1^\top \mathbf w_u / \sigma_1,\ \mathbf t_2^\top \mathbf w_u / \sigma_2]^\top$,
the tangential part of the viewing ray. Together

```math
\frac{\partial \hat{\mathbf c}_i(u)}{\partial \boldsymbol\theta_k} = \mathbf g_{uk}\, \mathbf j_{uk}, \qquad
\mathbf g_{uk} = - w_{uk}\, (\mathbf a_k - \mathbf b_{uk})\, (\mathbf s_{uk}^\top \mathbf m_{uk}) .
```

The image Jacobian factors through the depth Jacobian. Both sources are quadratic forms on the same
coordinates through the same row $\mathbf j_{uk}$ and differ by a scalar weight per pixel and
primitive. This holds because a surfel is a plane.

### 2.4 The entity is the unit

A single surfel moved alone tears the surface; its block measures coverage, not parallax. A fault
moves a surface. For entity $m$,

```math
J^I_{im}(u) = \sum_{k \in \mathcal K_m} \mathbf g_{uk}\, \mathbf j_{uk}\, P_{km}, \qquad
\Lambda^I_{im} = \sum_u \frac{J^I_{im}(u)^\top J^I_{im}(u)}{\sigma_I^2}, \qquad
\boldsymbol\eta^I_{im} = \sum_u \frac{J^I_{im}(u)^\top \mathbf e^I_i(u)}{\sigma_I^2} .
```

Inside the entity the tears cancel and texture gradient times slide remains; at its outline the
contrast with the surface behind remains. **Stiffness is texture contrast times the spread of viewing
directions**, summed over the pixels of the entity. One view alone gives none once appearance is
accounted for (Section 4). The working hypothesis for the prior is that its error on an entity is
low-order, an offset and two tilts plus a remainder; the surface itself may be curved.

### 2.5 The record

Per entity and per source $s$ (colour, frozen-feature and label channels of the image family; the
depth family; a weak regulariser), the map stores

```math
\bigl( \Lambda^s_m,\ \boldsymbol\eta^s_m,\ \hat\varphi^s \bigr), \qquad
\Lambda^s_m = \sum_i \Lambda^s_{im}, \quad \boldsymbol\eta^s_m = \sum_i \boldsymbol\eta^s_{im} ,
```

a symmetric $3 \times 3$ matrix, a 3-vector and a dispersion: ten numbers per source. Three properties
make this a representation for large areas and not only a diagnostic.

- *Additive.* Frames add. Tiles add. Leaving out a block of frames is a subtraction. Two maps of the same ground are merged by adding their records, source by source.
- *Local.* An entity is coupled to its image neighbours through shared pixels and to other entities only through shared per-frame nuisance. The structure is the sparse entity-by-frame pattern of bundle adjustment, and its size grows linearly with area at fixed overlap.
- *Separable.* Because sources are not summed, any source can be removed, reweighted or tested after the fact without revisiting the images.

Derived fields, per entity: *knowledge* $\tfrac12 \log\det \Lambda_m$ in nats; *what appearance says
about geometry*, the mutual information $\tfrac12 \log\det(\mathbf I + (\Lambda^{D}_m + \Lambda^0_m)^{-1} \Lambda^I_m)$
in nats; *who is speaking*, the generalised eigenvalues $\vartheta$ of $(\Lambda^I_m, \Lambda^D_m)$ per
placement direction; and *surprise*, the energy of Section 3.

## 3. Alignment as Stored Energy

### 3.1 The measure, its units, its null

Each source alone would place the entity at $\hat{\boldsymbol\theta}^s_m = (\Lambda^s_m)^{-1} \boldsymbol\eta^s_m$,
with stated covariance $\hat\varphi^s (\Lambda^s_m)^{-1}$. Two springs with rest positions
$\hat{\boldsymbol\theta}^D$, $\hat{\boldsymbol\theta}^I$ and stiffnesses $\Lambda^D$, $\Lambda^I$ (dispersions
absorbed) attached to one body settle at the information-weighted mean and store the energy

```math
E_m = \tfrac12\, \mathbf t_m^\top \bigl( (\Lambda^D_m)^{-1} + (\Lambda^I_m)^{-1} \bigr)^{-1} \mathbf t_m, \qquad
\mathbf t_m = \hat{\boldsymbol\theta}^D_m - \hat{\boldsymbol\theta}^I_m .
```

- *Stretch* $\mathbf t_m$: metres of offset and radians of tilt. It is the placement the prior asks for minus the placement the pixels ask for.
- *Stiffness* $\Lambda$: inverse square metres and inverse square radians.
- *Energy* $E_m$: dimensionless, in nats. It is the log-likelihood ratio of "the two sources see one surface" against "they see two".
- *Null.* If both sources are unbiased with independent errors and honest information, $T_m = 2 E_m \sim \chi^2_3$. Over a map of $M$ tested entities, $\sum_m T_m / (3M)$ is one number that should be one.

The same form measures the alignment of the *current* geometry with one source: replace
$\hat{\boldsymbol\theta}^D$ by the stored placement and drop $(\Lambda^D)^{-1}$. "Geometry and appearance
are aligned to within what the data can tell" is the statement $2E \sim \chi^2_3$, entity by entity.

### 3.2 Three kinds of misfit

With per-frame contributions, the total misfit of an entity about a common placement splits exactly:

```math
\chi^2_{\mathrm{total}} = \underbrace{\sum_j \bigl( \hat{\boldsymbol\theta}^D_{mj} - \hat{\boldsymbol\theta}^D_m \bigr)^\top \Lambda^D_{mj} \bigl( \cdot \bigr)}_{\text{within depth}}
+ \underbrace{\sum_i \bigl( \hat{\boldsymbol\theta}^I_{mi} - \hat{\boldsymbol\theta}^I_m \bigr)^\top \Lambda^I_{mi} \bigl( \cdot \bigr)}_{\text{within image}}
+ \underbrace{T_m}_{\text{between}} .
```

Write the prior's placement error on entity $m$ in frame $j$ as $\boldsymbol\mu_m + \boldsymbol\alpha_{mj}$:
a surface term that every frame repeats and a per-frame term. *Within depth* contains only the
per-frame terms; it is what agreement among depth maps tests. *Between* contains the surface term; it
is the only place a repeated fault appears. *Within image* contains neither; it is free of the prior's
error and is used to make the image information honest. The repetition coefficient
$\varrho = \operatorname{tr}\Sigma_\mu / (\operatorname{tr}\Sigma_\mu + \operatorname{tr}\Sigma_\alpha)$
is the share of the prior's error that no consensus among depth maps can see, and it is measurable
from the first and third terms on real flights.

### 3.3 Energy is symmetric: attribution, and when to listen

Energy says that two sources disagree, not which is wrong. Three hypotheses per entity are separated
by three statistics: the energy, the within-image scatter, and outline against interior (Section 7).
Depth wrong: high energy, images mutually consistent, outline and interior agree. Pixels misled (a
second layer, a mover): high energy, outline agrees with depth while the interior does not, and for a
reflection the stretch has one sign, behind. The verdict is a posterior over the three, with priors
from heads on frozen features, and yields one gate per source.

When to listen follows from the stored record, per placement direction. Where $\vartheta \gg 1$ the
pixels decide and the prior is nearly irrelevant. Where $\vartheta \approx 1$ the sources are fused by
information and tested; a prior fault of size $\Delta$ would move the surface by $\Delta/(1+\vartheta)$
and has non-centrality $\Delta^2 \Lambda^D \vartheta / (1 + \vartheta)$, so this is where testing pays.
Where $\vartheta \ll 1$ the prior decides, unchecked by pixels; the map marks the entity
*unsupported* unless a head with support in feature space says otherwise. The gate acts on placement
only, so a faulty entity keeps the shape of its monocular depth and takes its position from the
pixels.

## 4. Reading the Energy Without Letting the Model Hide It

A joint fit can relieve the stretch with freedoms that are not placement: view-dependent colour,
semi-transparent duplicates, floaters, lowered opacity. At convergence the energy is small and the
fault remains. The energy is therefore read in a restricted model.

**Score test.** Place the geometry by the depth source alone. There $\boldsymbol\eta^D_m = 0$,
$\mathbf t_m = -(\Lambda^I_m)^{-1} \boldsymbol\eta^I_m$, and the energy is the pull of the pixels on a
surface the prior placed. The geometry has never seen an image. A surface fused from all frames'
depth carries the surface term and only the mean of the per-frame terms, so this reads the repeated
fault directly.

**Appearance with no freedom.** Texture each surfel from *source* frames by projection at its current
placement and let disjoint *judging* frames, interleaved so the baseline is kept, supply the
residuals. Nothing can be explained away. If appearance is fitted instead, its freedom is charged
through the Schur complement, and the retained fraction

```math
\kappa_m = \frac{\operatorname{tr}\bigl( \Lambda_{\theta\theta} - \Lambda_{\theta a} \Lambda_{aa}^{-1} \Lambda_{a\theta} \bigr)}{\operatorname{tr} \Lambda_{\theta\theta}} \in [0, 1]
```

says how much stiffness survives. One diffuse colour seen by many frames keeps nearly all of it;
high-order view dependence, which can imitate parallax, keeps little; a single frame keeps none.

## 5. Honest Information

**Definition.** The information of a source is honest if its stated covariance equals the realised
scatter of its estimate, $\mathbb E\,(\hat{\boldsymbol\theta}^s - \boldsymbol\theta)(\cdot)^\top = \hat\varphi^s (\Lambda^s)^{-1}$.

**Why it fails by default.** $\sigma_I$ is not sensor noise. Rendering residuals come from finite
surfel resolution, resampling and unmodelled appearance, and are correlated across pixels. Counting
pixels as independent overstates stiffness by orders of magnitude.

**Three corrections, none needing ground truth.**

1. *Dispersion from replicates.* Frames are the independent units. With $n_m$ frames on an entity, $\hat\varphi^I = \chi^2_{\text{within image}} / (3(n_m - 1))$, pooled per appearance stratum and per channel. This is a sandwich variance.
2. *Response to injection.* Move an entity's surfels by a known $\boldsymbol\delta$ and re-render. With projective appearance the response of any statistic is the response to a true error of that size, $S(\boldsymbol\delta) = \mathcal S(\boldsymbol\epsilon_m + \boldsymbol\delta)$. The curvature at the minimum is the realised stiffness, occlusion changes included, and must match the stated one. The same device turns a frozen video encoder into an instrument with units.
3. *The bulk of the energies.* On entities with stiff image coupling the $T_m$ must follow $\chi^2_3$ in their bulk; the tail is the faults.

A stated false-alarm rate needs a further calibration with entities or scenes as units.

## 6. Nuisance: Absorbed, Deflated, or Detected

The rule: a nuisance that spans a known low-dimensional subspace of the evidence is projected out, and
the map loses exactly the information in that subspace. For a nuisance block $\mathbf q$ with prior
information $\Sigma_q^{-1}$,

```math
\Lambda^{\mathrm{marg}}_{\theta\theta} = \Lambda_{\theta\theta} - \Lambda_{\theta q} \bigl( \Lambda_{qq} + \Sigma_q^{-1} \bigr)^{-1} \Lambda_{q\theta} ,
```

and the retained fraction $\operatorname{tr}\Lambda^{\mathrm{marg}} / \operatorname{tr}\Lambda$ is the
stated cost. Marginalising a per-frame block couples all entities of that frame; the coupling is the
sparse pattern of Section 2.5.

| Nuisance | Fate | What is lost or assumed |
| --- | --- | --- |
| Pose of a frame (six parameters, prior from navigation or a previous adjustment) | **Absorbed** by marginalisation in the image family | A pose error moves all entities of a frame together, as a low-order field. Projecting it out removes the image evidence about that common mode: absolute placement is known no better than the poses; relative placement between entities is kept. An entity that fills the frame is confounded with pose and retains almost nothing |
| Scale and offset of a depth map, shared distance bias | **Absorbed** in the depth family | The prior then says nothing about a frame-wide affine error. Such an error is by definition not a fault. Only shape and relative placement within a frame remain as prior evidence |
| Exposure gain and offset per frame; appearance | **Absorbed** in the image family | Cost measured by $\kappa_m$ |
| Rendering model error: surfel resolution, resampling, residual view dependence | **Deflated** by the dispersion $\hat\varphi$ | Not removed. All stiffness of the source is reduced by a measured factor, and with it the power of every test. If the factor is large the correct output is "little is known" |
| Per-frame term of the prior error | **Deflated**: a variance component $\Sigma_\alpha$ in the stated covariance of the depth source | Assumes it is independent beyond the left-out block |
| Dispersions, variance components, prevalence | **Estimated and plugged in** | Their own uncertainty is not propagated; tolerable because each is a few numbers from many entities |
| Segmentation into entities, depth order, surfel scales | **Assumed** | Errors here are neither absorbed nor tested by the energy; the within-entity remainder and outline consistency flag some |
| Surface term of the prior error on a region | **Detected** by the between-source energy | Cannot be absorbed: it has no global form, and deflating for it would discard the prior everywhere |
| Image-consistent wrong geometry: planar reflection, object moving along the strip | **Detected** in part, by outline against interior and by cross strips | Without an attached outline, not detected |
| Gross pose blunder, scene change between passes | **Outside the model** | Beyond linearisation; shows as inflated dispersion of a whole frame or pass |

Two consequences deserve to be stated as limits. A fault that lies in an absorbed subspace is invisible
by construction; that is what absorbing means. And the three fates compete: every nuisance absorbed
lowers the stiffness available for detection, so the list of absorbed nuisances must be short and
each must earn its place by a measured retained fraction.

## 7. Correspondence-Free: the Exact Sense, the Gain, the Price

**The sense.** No discrete association between image points is estimated, stored or trusted. A pixel
is associated with the primitives it renders, softly, by the weights $w_{uk}$, and the association is
a function of the current map, recomputed whenever the map changes. Cross-view evidence exists only as
the sum of per-frame contributions on shared primitives. If the poses themselves come from a
feature-based adjustment, the claim covers the mapping stage; it covers the whole pipeline only when
poses come from navigation sensors and are corrected inside the model as in Section 6.

**The gain.** Weak texture lowers stiffness continuously instead of breaking a matcher; abstention is
a value of the stored field, not a missing datum. Every pixel contributes, including outlines of
untextured regions through the label channel: rendering an entity's identity and comparing it with
each frame's segment mask gives three placement parameters without interior texture, and the
rasteriser assigns an occlusion edge to the nearer entity without a rule. Occlusion is consistent
across all evidence. Cost is linear in pixels, with no track management. And the evidence arrives
already in information form.

**The price.** The comparison is local. The pull is a valid estimate only while the misplacement
shifts the image by less than the correlation length of the texture, a few pixels, or of the feature
channel, a few patches. Beyond that the gradient carries no direction. *Detection* survives, because
a decorrelated region has a large, saturated misfit; *estimation* does not. Three things widen the
range: coarse channels (frozen features, the label channel), a one-dimensional sweep of the entity's
offset through the renderer, which is exhaustive in the one direction parallax acts, and above all
the prior, which starts every surface near its place. Explicit matching has a far larger capture
range and remains the better tool where the prior is absent or grossly wrong.

**The second reading of "cross view".** Views from different strips, cameras or passes (nadir and
oblique, crossing lines, repeated flights) enter the same way, as further frames on the same
primitives. They help: slides in a second direction stiffen the placement directions a single strip
leaves soft and expose objects moving along a strip. They also strain the assumptions: diffuse
appearance fails sooner under large changes of viewing angle, and illumination and shadows change
between passes. The formulation covers them with a dispersion and an exposure term per pass, and the
within-image term then splits into within-pass and between-pass parts, the latter measuring scene or
appearance change. Ground-to-aerial matching is outside what is claimed.

## 8. Layers

Splitting an entity's image evidence into outline and interior gives two placements. They agree for
an opaque diffuse surface. An outline that agrees with the prior and an interior that asks for a
placement behind it is a second layer: a reflection or a view through glass. The entity keeps its
depth evidence, its interior image evidence is gated, and the material attribute is stored.

## 9. What Is Identified, and on What It Rests

| Quantity | From | Rests on |
| --- | --- | --- |
| dispersion per source, channel and stratum | within-source scatter over frames | frames independent; static scene |
| stiffness of the image coupling; retained fractions | Jacobians or injected curvature; Schur complements | restricted appearance; linearisation within capture range |
| per-frame component of the prior error | within-depth scatter, block left out | entity association |
| law of the surface term, fault prevalence per stratum | deconvolution of $\mathbf t_m$ over entities with the stated covariances | image errors independent of prior errors; no image-consistent wrong geometry |
| repetition coefficient | variance components | as above |
| second layer | outline against interior | an attached outline in view |
| absolute placement of the map | not from images beyond the pose prior | |
| anything in a direction with no image stiffness and no feature support | not identified | |

The sample size of every per-stratum estimate is the number of entities.

## 10. What Is Learned

No backbone is trained. Light heads on frozen dense image features and frozen video-encoder tokens,
lifted onto entities by the compositing weights, supply what the energy cannot: prior probabilities
that a surface of this appearance makes the depth model fail, or misleads the pixels; and a support
head that predicts whether such a surface ever has image stiffness, which needs no fault labels and
separates interpolation from extrapolation. They are fitted by the marginal likelihood of the
energies, each entity weighted by how informative its record is. Their inputs never include the
residuals. The transfer to pixel-blind entities assumes that, among entities that look alike, the
witnessed ones fail as often as the others; where no look-alike is ever witnessed the map says
*unsupported*.

## 11. What Would Refute This, Cheapest First

1. **The pull is not a measurement.** On a synthetic scene with exact geometry, offset one entity by a known amount, texture it projectively from source frames, and compute pull, stiffness and energy from judging frames. Required: the stretch recovers the offset inside the capture range, and energies of unperturbed entities follow $\chi^2_3$ after the dispersion. If the dispersion needed is so large that offsets of the size of real prior faults are insignificant, the image coupling is too soft at surfel resolution to judge anything. Runs on a processor with the existing reference renderer.
2. **Information is not honest.** Stated covariance against realised error of entity placement on exact geometry: a calibration curve of the information field. A map whose stated uncertainty does not track its error has no claim to the third component.
3. **No between-source variance.** If the surface term of the prior error is negligible, consensus among depth maps suffices and the provenance argument loses its example.
4. **Absorbing pose costs everything.** Retained fraction after marginalising pose at realistic pose uncertainty. If typical entities keep little, pose must be estimated first by other means.
5. **Out of range.** Share of true prior faults whose image shift exceeds the capture range of every channel and of the sweep.
6. **A pixel mask does as well.** If thresholding photometric error of an image-only fit matches the gates on a harm-weighted metric, the representation is not earning its place.

## 12. Prior Work Bearing on the Theory

*Read (abstract or text).* A method that keeps the monocular depth loss only where rendered depth
passes a virtual stereo check (arXiv 2604.05715); one that keeps it where an image-only fit has low
photometric error and verifies against lidar that photometric error separates good from bad
monocular depth (arXiv 2607.02554); a per-Gaussian ambiguity indicator from spherical-harmonic energy
(arXiv 2605.12494); depth supervision weighted by the depth model's own uncertainty (arXiv
2405.19657); per-Gaussian block Fisher information from the rendering Jacobian, used for pruning
(arXiv 2406.10219); closed-form lifting of masks onto Gaussians by accumulated weights (arXiv
2409.08270). None stores information per source, states a null distribution, or separates within-source
from between-source misfit.

*Titles seen in search, not read.* Information-driven exploration with Gaussian maps, optimal-design
view selection, observability-aware Gaussian SLAM, variational uncertainty in splatting. The phrase
"information-theoretic Gaussian splatting" was not seen as a title in one search; the neighbouring
ground is occupied, and whether the name is free must be settled by a proper search.

*Recalled.* Fisher information for view selection in radiance fields; post-hoc spatial uncertainty;
direct photometric alignment and its Hessian, which is the image stiffness here; the information
filter and sparse information-form mapping, of which additivity and marginalisation are textbook
properties; score tests and prior-data conflict diagnostics.

*What is claimed as new in this part.* Provenance: information kept per source on shared placement
coordinates, with the exact three-way split of misfit and the repeated fault as its between-source
term. The energy read as a score test in a model that cannot explain it away, with retained fractions
as the stated cost of appearance and of nuisance. The taxonomy of absorbed, deflated and detected. The
factorisation of the image Jacobian through the depth Jacobian, and outline parallax as a rendered
label channel. Per-primitive Fisher information, lifting, and information-form fusion are not new.

## 13. Limits

1. Where pixels have no stiffness in any direction, the record is a prior on appearance and is marked unsupported.
2. Image-consistent wrong geometry without an attached outline passes every internal check.
3. Absolute placement is no better than the poses; faults inside an absorbed subspace are invisible.
4. Estimation by rendering has a short capture range; gross faults are detected, not located, without a sweep.
5. Honesty is checked, not guaranteed: dispersion deflates for misfit at a cost in power, and plug-in variance components carry unpropagated uncertainty.
6. Independence of the two sources' errors is assumed; they share poses and discretisation.
7. Attribution can fail when both sources are wrong on one entity.
8. The record describes the restricted test model. The final joint reconstruction must be shown separately not to undo the gates.
