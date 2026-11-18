# Theory: Depth Prior and Pixels as Two Message Families on the Same Gaussian Primitives

This note develops the statistical model of the fault indicator when image evidence and depth evidence
are fused on the same 2D Gaussian surfels through a differentiable renderer. Every frame of a posed
aerial video sends each primitive two messages about where the surface is: one from its monocular
depth map, one from its pixels. The fault test is the disagreement between the two families,
normalised by their information. The question the model answers per primitive is when to listen to
the depth prior and when to the pixels.

The note derives the image message from the rasteriser's Jacobian (Section 2), defines the tension
statistic, its decomposition and its null distribution (Section 3), treats the two ways the test can
lie, appearance that explains wrong geometry away and a judge that shares a model with the judged
(Section 4), re-expresses outline parallax, reflections and injection in renderer terms (Sections 5
and 6), states what is identified and what is still learned (Sections 7 and 8), and ends with the
existing code, prior work, and what would refute the approach.

Nothing here has been measured. The derivations are first-order and have not been checked against
automatic differentiation. Statements about prior work and software are marked verified or recalled
in Section 12.

## 1. Setting and Symbols

Surfel $k$ has centre $\boldsymbol\mu_k$, frame $[\mathbf t_1\ \mathbf t_2\ \mathbf n]$, scales
$(\sigma_1, \sigma_2)$, opacity $\alpha_k$ and an appearance vector $\mathbf a_k \in \mathbb R^C$ (colour,
or a frozen feature, or a label). Its *placement* is the normal offset and the two tilts,
$\boldsymbol\theta_k = (\nu, \tau_1, \tau_2)$, the coordinates that depth observes. The other
geometric coordinates (in-plane shift and rotation, scales) move a footprint within the surface and
are not what a depth fault changes.

A *surface entity* $m$ is a set of surfels $\mathcal K_m$ obtained by lifting per-frame segment masks
(Section 5). Its placement $\boldsymbol\theta_m \in \mathbb R^3$ is a common offset along the entity's
mean normal and two tilts about its centroid; $P_{km}$ maps it to the placement of member $k$.

| Symbol | Meaning |
| --- | --- |
| $w_{uk}$ | compositing weight of surfel $k$ at pixel $u$ |
| $\mathbf j_{uk}$ | depth Jacobian $\partial d_k / \partial \boldsymbol\theta_k$ on the ray of $u$ |
| $\mathbf g_{uk}$ | appearance change at $u$ per unit depth change of surfel $k$ |
| $\Lambda^D, \boldsymbol\eta^D$; $\Lambda^I, \boldsymbol\eta^I$ | information and pull of the depth family and of the image family |
| $\hat{\boldsymbol\theta}^D$, $\hat{\boldsymbol\theta}^I$ | placement each family would choose alone |
| $\mathbf t_m$, $T_m$ | tension vector and statistic of entity $m$ |
| $\hat\varphi$ | dispersion of the image family, estimated from its own scatter |
| $\kappa_m$ | fraction of image information retained after profiling appearance |
| $\boldsymbol\mu_m$, $\boldsymbol\alpha_{mj}$ | surface term and per-frame term of the monocular placement error |
| $\varrho$ | repetition coefficient |
| $\vartheta$ | generalised eigenvalues of $(\Lambda^I, \Lambda^D)$: information ratio per placement direction |
| $\zeta^D_m$, $\zeta^I_m$ | posterior that the depth family, respectively the image family, is at fault |

## 2. The Two Messages

### 2.1 Depth message

The rendered depth is $\hat D_i(u) = \sum_k w_{uk}\, d_k$ with $d_k$ the depth at which the ray of $u$
meets the plane of surfel $k$, and

```math
\mathbf j_{uk} = \frac{\partial d_k}{\partial (\nu, \tau_1, \tau_2)} = \frac{\bigl[\,1,\ -\mathbf t_1^\top(\mathbf x_{uk} - \boldsymbol\mu_k),\ -\mathbf t_2^\top(\mathbf x_{uk} - \boldsymbol\mu_k)\bigr]}{\mathbf n^\top \mathbf w_u} ,
```

with $\mathbf w_u$ the ray direction and $\mathbf x_{uk}$ the intersection point. Frame $i$ sends

```math
\Lambda^D_{ik} = \sum_u w_{uk}\, \frac{a_i^2\, \mathbf j_{uk}^\top \mathbf j_{uk}}{\sigma_i^2(u)}, \qquad
\boldsymbol\eta^D_{ik} = \sum_u w_{uk}\, \frac{a_i\, \mathbf j_{uk}^\top e^D_i(u)}{\sigma_i^2(u)} ,
```

with $e^D_i$ the depth residual and $(a_i, b_i)$ the per-frame alignment. These lumped blocks exist.

### 2.2 Image message: the rasteriser's Jacobian

The rendered appearance is $\hat{\mathbf c}_i(u) = \sum_k w_{uk}\, \mathbf a_k$, with
$w_{uk} = \alpha_k G_k(\mathbf s_{uk})\, T_{uk}$, $G_k(\mathbf s) = \exp(-\tfrac12 \lVert \mathbf s \rVert^2)$,
$T_{uk}$ the transmittance in front of $k$, and $\mathbf s_{uk}$ the coordinates of the ray-plane
intersection in the surfel's tangent frame, in units of its scales:
$s_a = \mathbf t_a^\top(\mathbf x_{uk} - \boldsymbol\mu_k) / \sigma_a$.

*Step 1: the kernel.* The weight of $k$ changes through its kernel, and the weights of everything
behind $k$ change through the transmittance. Collecting both,

```math
\frac{\partial \hat{\mathbf c}_i(u)}{\partial \log G_k} = w_{uk}\, \bigl( \mathbf a_k - \mathbf b_{uk} \bigr), \qquad
\mathbf b_{uk} = \frac{\sum_{l \text{ behind } k} w_{ul}\, \mathbf a_l}{T_{uk}\,(1 - \alpha_k G_k)} ,
```

where $\mathbf b_{uk}$ is the appearance composited behind $k$: what the pixel would show if $k$ were
removed. A primitive matters to the image only through its *contrast* with what lies behind it.

*Step 2: the slide.* Moving the plane of $k$ along the ray by $\partial d_k$ moves the intersection
point by $\mathbf w_u\, \partial d_k$, hence the tangent coordinates by

```math
\frac{\partial \mathbf s_{uk}}{\partial d_k} = \mathbf m_{uk} = \Bigl[ \frac{\mathbf t_1^\top \mathbf w_u}{\sigma_1},\ \frac{\mathbf t_2^\top \mathbf w_u}{\sigma_2} \Bigr]^\top ,
```

the tangential component of the viewing ray. Seen along its normal a surfel does not slide; the slide
grows with the obliquity of the view, and the difference of slides between two frames is the parallax.

*Result.* With $\partial \log G_k / \partial \mathbf s = -\mathbf s^\top$,

```math
\frac{\partial \hat{\mathbf c}_i(u)}{\partial \boldsymbol\theta_k} = \mathbf g_{uk}\, \mathbf j_{uk}, \qquad
\mathbf g_{uk} = - w_{uk}\, \bigl( \mathbf a_k - \mathbf b_{uk} \bigr)\, \bigl( \mathbf s_{uk}^\top \mathbf m_{uk} \bigr) \in \mathbb R^{C} .
```

**The image Jacobian with respect to placement factors through the depth Jacobian.** Both families act
on the same three coordinates through the same geometric row $\mathbf j_{uk}$; they differ by a scalar
weight per pixel and primitive, $w_{uk} a_i^2 / \sigma_i^2$ for depth and
$\lVert \mathbf g_{uk} \rVert^2 / \sigma_I^2$ for the image. This is the precise sense in which the
primitive is where the two meet, and it holds because a surfel is a plane. The Gauss-Newton message of
frame $i$ is

```math
\Lambda^I_{ik} = \sum_u \frac{\lVert \mathbf g_{uk} \rVert^2}{\sigma_I^2}\, \mathbf j_{uk}^\top \mathbf j_{uk}, \qquad
\boldsymbol\eta^I_{ik} = \sum_u \frac{\mathbf j_{uk}^\top\, \mathbf g_{uk}^\top \mathbf e^I_i(u)}{\sigma_I^2} ,
```

with $\mathbf e^I_i = \mathbf c_i - \hat{\mathbf c}_i$ the image residual. The pull $\boldsymbol\eta^I$ for
all primitives is one backward pass of the image loss per frame; the rasteriser supplies visibility and
occlusion.

### 2.3 Per primitive is the wrong level; per entity is right

Two corrections to the idea that the per-primitive curvature is the parallax error bar and vanishes on
blank surface.

- *A single surfel moved alone tears the surface.* Its message measures contrast with whatever shows through the gap. On a uniformly coloured, watertight surface the contrast $\mathbf a_k - \mathbf b_{uk}$ is zero and so is the information. Where coverage is incomplete, the background shows and the per-primitive information is not zero; it is information about coverage, not about parallax.
- *A fault moves a surface, not a surfel.* The relevant direction is a coherent move of an entity, with Jacobian

```math
J^I_{im}(u) = \sum_{k \in \mathcal K_m} \mathbf g_{uk}\, \mathbf j_{uk}\, P_{km}, \qquad
\Lambda^I_{im} = \sum_u \frac{J^I_{im}(u)^\top J^I_{im}(u)}{\sigma_I^2} .
```

  Inside the entity the tears of neighbouring surfels cancel and what remains is the texture gradient times the slide: blank interior, no information, as it should be. At the outline of the entity the contrast with the surface behind does not cancel: that is outline parallax, and it appears without being asked for (Section 5).

The entity block is not the sum of the per-primitive blocks; it contains their cross terms. It is
computed along the three entity directions only.

### 2.4 Lumped, exact, and cost

- *Pull.* One render and two backward passes per frame (depth loss, image loss) give $\boldsymbol\eta^D$ and $\boldsymbol\eta^I$ for every primitive; entity pulls are $\sum_k P_{km}^\top \boldsymbol\eta_k$.
- *Entity information by finite differences.* Move an entity by $\pm\boldsymbol\delta$ along each of its three directions and re-render the frames that see it. Entities that do not touch in any image are moved in the same render (a colouring of the image-adjacency graph of entities, a handful of colours). This is exact in the entity direction, includes occlusion changes, and is the injection sweep of Section 6. Order of cost: six renders per colour and frame, small against a fit.
- *Closed form.* Accumulate $\mathbf g_{uk} \mathbf j_{uk} P_{km}$ per pixel in the chunked reference renderer, which already produces $w_{uk}$ and $\mathbf j_{uk}$; it needs the back-to-front sums for $\mathbf b_{uk}$ in addition. Memory is of the order of primitives times chunk.
- *Probing.* For random sign images $\mathbf v$, a backward pass of $\mathbf v^\top \hat{\mathbf c}$ returns $J^\top \mathbf v$, and the mean of its outer products estimates $J^\top J$. A few probes per frame give all blocks at once.
- *Lumped blocks.* The majorising trick of the depth message ($w$ instead of $w^2$) does not carry over, because $\mathbf g_{uk}$ is signed. Per-primitive image blocks are kept for optimisation with damping; the test uses entity blocks.

## 3. The Tension Test

### 3.1 Statistic

For entity $m$, linearised at the current state, each family alone would place the entity at

```math
\hat{\boldsymbol\theta}^D_m = (\Lambda^D_m)^{-1} \boldsymbol\eta^D_m, \qquad \hat{\boldsymbol\theta}^I_m = (\Lambda^I_m)^{-1} \boldsymbol\eta^I_m ,
```

sums over frames understood. The tension is their difference,

```math
\mathbf t_m = \hat{\boldsymbol\theta}^D_m - \hat{\boldsymbol\theta}^I_m, \qquad
T_m = \mathbf t_m^\top \bigl( (\Lambda^D_m)^{-1} + \hat\varphi\, (\Lambda^I_m)^{-1} \bigr)^{-1} \mathbf t_m \ \sim\ \chi^2_3 \quad \text{under } H_0 ,
```

if both families are unbiased with independent errors. In the linear model $\mathbf t_m$ does not depend
on the state at which it is evaluated. At a joint fit the pulls cancel,
$\boldsymbol\eta^D_m = -\boldsymbol\eta^I_m = \boldsymbol\eta$, and, with the dispersion absorbed into
$\Lambda^I_m$,

```math
T_m = \boldsymbol\eta^\top \bigl( (\Lambda^D_m)^{-1} + (\Lambda^I_m)^{-1} \bigr)\, \boldsymbol\eta :
```

two large opposite pulls, each measured in units of its own information. $T_m$ equals the increase in
total misfit caused by forcing the two families to share one placement; it is the likelihood-ratio
statistic for "depth and pixels see different surfaces". The vector $\mathbf t_m$ is the entity gap of
the parallax formulation, monocular placement minus image-witnessed placement, computed by the
renderer; its three components are an offset fault and two tilt faults.

### 3.2 Decomposition: consensus is within, repetition is between

With per-frame messages, the total misfit of an entity about a common placement splits exactly:

```math
\chi^2_{\mathrm{total}} = \underbrace{\sum_j \bigl( \hat{\boldsymbol\theta}^D_{mj} - \hat{\boldsymbol\theta}^D_m \bigr)^\top \Lambda^D_{mj} \bigl( \cdot \bigr)}_{\text{within depth: consensus}}
+ \underbrace{\sum_i \bigl( \hat{\boldsymbol\theta}^I_{mi} - \hat{\boldsymbol\theta}^I_m \bigr)^\top \Lambda^I_{mi} \bigl( \cdot \bigr)}_{\text{within image: ancillary}}
+ \underbrace{T_m}_{\text{between families}} .
```

Write the monocular placement error of frame $j$ on entity $m$ as
$\boldsymbol\mu_m + \boldsymbol\alpha_{mj}$, a surface term every frame repeats and a per-frame term.

- The *within-depth* term contains only the $\boldsymbol\alpha_{mj}$. It is what leave-out depth consensus tests. The surface term cancels in it.
- The *between* term contains $\boldsymbol\mu_m$ plus the average of the $\boldsymbol\alpha$. It is the only place the repeated fault appears.
- The *within-image* term contains neither. It is ancillary for the monocular error and plays the role that the across-epipolar and along-epipolar track residuals played for a tracker.

A fault that every frame repeats is therefore, exactly, a between-family effect, and no reweighting of
depth messages against each other can see it. The repetition coefficient is the share of the
monocular error variance in the surface term,
$\varrho = \operatorname{tr}\Sigma_\mu / (\operatorname{tr}\Sigma_\mu + \operatorname{tr}\Sigma_\alpha)$
for per-frame terms independent beyond the left-out block, estimated from the first and third
components of the decomposition.

### 3.3 Null distribution

The $\chi^2_3$ law needs three things that do not hold by default.

1. **A noise level for pixels.** Image residuals at a fit are dominated by model misfit (finite surfel resolution, resampling, exposure), not by sensor noise, and are spatially correlated. $\sigma_I$ is not known. The within-image term supplies it: with $n_m$ frames seeing the entity, $\hat\varphi = \chi^2_{\text{within image}} / (3 (n_m - 1))$, pooled per appearance stratum, is the dispersion that makes the image family's stated information honest. This is a sandwich variance with frames as the independent units.
2. **Independence of the two families' errors.** They share poses and the surfel discretisation. Pose error moves all entities of a frame together and is removed as a per-frame low-order field; what remains is an assumption.
3. **A model in which the tension cannot leak.** Section 4.

Calibration on the scene itself: on entities with strong image information and passing ancillaries,
$T_m$ should follow $\chi^2_3$ in its bulk; the fitted scale of the bulk is the last empirical
correction, and the tail is the faults. A stated false-alarm rate needs a separate calibration with
entities or scenes as units.

### 3.4 Tension is symmetric: attribution

$T_m$ says that the families disagree, not which is wrong. Three hypotheses per entity:

| Hypothesis | Tension | Within-image scatter | Outline against interior (Section 5) |
| --- | --- | --- | --- |
| $H_0$ both right | $\chi^2_3$ | nominal | agree |
| $H_D$ depth placement wrong | large, any sign | nominal | agree with each other, both against depth |
| $H_I$ pixels misled (second layer, mover) | large; for a reflection the offset has one sign, behind | often inflated | outline agrees with depth, interior does not |

The verdict is the posterior over the three, with priors $\pi^D(\boldsymbol\phi_m)$ and
$\pi^I(\boldsymbol\phi_m)$ from heads on frozen features (Section 8) and the likelihoods of the three
statistics. It yields two gates, $1 - \zeta^D_m$ on the placement component of the depth messages and
$1 - \zeta^I_m$ on the image messages of the entity. An entity with no image information in any
direction has no tension and no test; its state is *unsupported* and its gate is the prior alone.

### 3.5 When to listen to which

Let $\vartheta_1, \vartheta_2, \vartheta_3$ be the generalised eigenvalues of $(\Lambda^I_m, \Lambda^D_m)$.
Along each eigen-direction the fused placement weights the pixels by $\vartheta / (1 + \vartheta)$. A
depth fault of size $\Delta$ along that direction moves the fused surface by $\Delta / (1 + \vartheta)$
(its harm) and has non-centrality $\Delta^2 \Lambda^D \vartheta / (1 + \vartheta)$ (its detectability).

| Regime | Rule |
| --- | --- |
| $\vartheta \gg 1$: pixels dominate | listen to pixels; the prior is nearly irrelevant in this direction and its verdict is a training label |
| $\vartheta \approx 1$ | fuse by information; test; on tension, attribute and gate |
| $\vartheta \ll 1$: pixels blind | listen to depth, with the prior head as the only check; report as unsupported if the head has no support |

The answer to the paper question is thus: by default in proportion to information, direction by
direction; the indicator changes the default only where the two disagree beyond their error bars, and
then according to which family the attribution blames. Because both informations live on the same
three coordinates, an entity can be pixel-dominated in offset and pixel-blind in a tilt, and the rule
applies per direction. The gate acts on placement only, so a faulty entity keeps the shape of its
monocular depth and takes its placement from the pixels.

## 4. Two Ways the Test Can Lie, and the Restricted Model

### 4.1 Explaining away

A joint fit does not have to sit at a compromise. It can relieve the tension with degrees of freedom
that are not placement: view-dependent colour that paints the parallax, semi-transparent duplicates
(one layer for the depth loss, one for the pixels), floaters, lowered opacity. At convergence both
pulls are small, $T_m \approx 0$, and the fault is still there. Tension measured at an unrestricted
joint fit is not a test.

### 4.2 The score test at the depth-placed surface

The test is therefore computed in a restricted model and without letting the pixels shape the
geometry. Fit the geometry to the depth messages alone. At that state $\boldsymbol\eta^D_m = 0$ and

```math
T_m = \boldsymbol\eta^{I\top}_m\, (\Lambda^I_m)^{-1} \bigl( (\Lambda^D_m)^{-1} + \hat\varphi\,(\Lambda^I_m)^{-1} \bigr)^{-1} (\Lambda^I_m)^{-1}\, \boldsymbol\eta^I_m ,
```

a score test: the pull of the pixels on a surface that the depth prior placed. The geometry has never
seen an image, so there is nothing for it to have explained away. A surface fused from all frames'
depth carries the surface term and only the average of the per-frame terms, so this is directly the
test of the repeated fault. The stabilised clip, one frame's depth map rendered into its neighbours,
is the special case with a single depth message.

### 4.3 Appearance: profiled, or not free at all

Appearance still has to come from the images. Two treatments.

**Profiled appearance.** Fit appearance on the fixed geometry (with fixed geometry the rendered image
is linear in the $\mathbf a_k$, so this is a linear least-squares problem) and account for it in the
information by the Schur complement,

```math
\Lambda^{I,\mathrm{prof}}_m = \Lambda_{\theta\theta} - \Lambda_{\theta a} \Lambda_{aa}^{-1} \Lambda_{a\theta}, \qquad
\kappa_m = \frac{\operatorname{tr} \Lambda^{I,\mathrm{prof}}_m}{\operatorname{tr} \Lambda_{\theta\theta}} \in [0, 1] .
```

$\kappa_m$ is the fraction of placement information that survives the freedom of appearance. With one
diffuse colour per surfel seen by many frames it is close to one. With high-order view dependence it
falls, because such an appearance can imitate parallax; with a single frame it is zero, because one
image fixes no depth. "Appearance can hide wrong geometry" becomes a number per entity, and the
appearance model for the test is chosen to keep it high: diffuse colour, a per-frame exposure gain and
offset, no densification, opaque surfels.

**Projective appearance.** Give the test model no free appearance at all: texture each surfel from a
set of *source* frames by projection at its current placement, $\mathbf a_k(\boldsymbol\theta) = I_r(\pi_r(\mathbf x_k(\boldsymbol\theta)))$,
and let the remaining frames judge. The Jacobian of a judging frame then carries the difference of the
slides in the judging and the source view, which is the parallax between them, times the texture
gradient. Nothing can be explained away, $\kappa_m = 1$ by construction, and the dependence on the
source is averaged by rotating it. This is the stabilised clip with fused geometry and with occlusion
decided by the rasteriser. It is the recommended test model; profiled appearance is the ablation.

### 4.4 Circularity

- *Geometry.* In the score test the pixels do not enter the tested geometry. After the verdict, the reconstruction is a different fit that fuses both families under the gates; the indicator is held fixed during it.
- *Appearance.* Source frames and judging frames are disjoint, in interleaved folds (alternate frames) so that the judging baseline is not shortened, and every frame judges once.
- *Depth family.* The within-depth term uses leave-block-out, as before, because neighbouring frames repeat per-frame errors.
- *Alignment.* The per-frame scale and offset of a depth map and the shared distance bias are estimated with the gates applied, and against image-witnessed placement where $\vartheta \gg 1$, so that no nuisance parameter absorbs a fault.
- *Heads.* Cross-fitted by ground tile.

## 5. Entities, Channels, Outlines and Layers

### 5.1 Lifting

Per-frame segment masks are fused onto surfels by accumulating compositing weights,
$A_{k\ell} = \sum_{i,u} w_{uk}\, \mathbb 1[\text{mask}_i(u) = \ell]$, and taking the largest; mask
identities of different frames are merged when they claim the same surfels. An entity is a set of
surfels, and no 2D tracking of masks is needed. Frozen features are lifted the same way,
$\boldsymbol\phi_k = \sum_{i,u} w_{uk} \boldsymbol\phi_i(u) / \sum_{i,u} w_{uk}$, giving one feature per
primitive and, pooled, per entity. Both are closed-form accumulations in one pass. Lifting uses the
geometry under test: under a placement fault the masks land on displaced surfels, by the residual
parallax, which is small against the entity for all but the smallest ones and is itself the signal of
Section 5.3.

### 5.2 Channels

The image message is defined for any rendered channel. Each has its own residual, its own dispersion
$\hat\varphi_c$ from its own within-family scatter, and adds to $\Lambda^I$ and $\boldsymbol\eta^I$:

- *colour:* the classical photometric message; fine but sensitive to exposure and specularity;
- *frozen features:* reduced to a few tens of channels, rendered from lifted or projected features and compared with each frame's feature map; coarse, tolerant to exposure, sensitive to weak large-scale structure;
- *labels:* render the entity identity and compare with each frame's own mask.

### 5.3 Outline parallax is the label channel, and the rasteriser decides ownership

In the label channel the contrast $\mathbf a_k - \mathbf b_{uk}$ is non-zero only where an entity ends
in front of another. The message measures the misalignment between the rendered outline and the
segmenter's outline in every frame: outline parallax, with three parameters per entity, needing no
interior texture. Ownership needs no rule. Moving the *farther* entity does not move an occlusion
edge, because its surface continues behind the occluder, so its Jacobian there is zero; only the
nearer entity receives the message. An edge where two entities meet at the same depth informs both.
The limits are unchanged: the depth order must be right in the model, a curved occluder's rim is not
attached to the surface, and the segmenter must draw an outline at the same physical place in every
view.

### 5.4 Layer test

Split the image message of an entity into the part from its outline and the part from its interior,
with estimates $\hat{\boldsymbol\theta}^{\partial}_m$ and $\hat{\boldsymbol\theta}^{\mathrm{int}}_m$. Their
difference, a three-degree $\chi^2$ with the dispersions, is zero for an opaque diffuse surface. An
outline that agrees with the depth placement and an interior that asks for a placement *behind* is a
second layer: a reflection or a view through glass. In renderer terms it is the statement that the
misfit is relieved by an additional additive layer behind the entity and not by moving the entity.
Such an entity is assigned to $H_I$: the depth message is kept, the interior image message is gated,
and "second layer" is recorded as a material attribute. A reflector with no attached outline in view
remains unresolved.

## 6. Injection Through the Renderer

Move the surfels of an entity by a known $\boldsymbol\delta$ and re-render every frame. If the model's
response to moving the entity equals the world's, any statistic $S$ of rendered against real frames
satisfies the exchange identity

```math
S(\boldsymbol\delta) = \mathcal S\bigl( \boldsymbol\epsilon_m + \boldsymbol\delta \bigr) :
```

the observed statistic under injection is the response function translated by the unknown placement
error. Three uses.

- *Estimator.* The minimiser of $S$ over $\boldsymbol\delta$ estimates $-\boldsymbol\epsilon_m$ in metric units for any black-box $S$, including the distance between frozen video-encoder tokens of the rendered clip and of the real clip. No calibration of the encoder is needed for the location.
- *Information.* The curvature of $S$ at its minimum is the instrument's own sensitivity at this entity. Zero curvature is abstention by measurement. For the photometric loss it reproduces $\Lambda^I_m$ exactly, occlusion changes included.
- *Supervision.* A readout head on encoder tokens of (rendered, real) clips is trained to return the injected $\boldsymbol\delta$; the label is exact and costs nothing.

The identity holds with projective appearance, where appearance follows the placement. With fitted
appearance it holds only if appearance is refitted at each $\boldsymbol\delta$; otherwise colours fitted
at the wrong placement are blurred and the response is not a pure translate. Compared with warping
one frame by hand, rendering adds the fused geometry and correct occlusion, and removes the
dependence on a single reference depth map.

## 7. What Is Identified

| Quantity | From | Rests on |
| --- | --- | --- |
| image dispersion $\hat\varphi_c$ per channel and stratum | within-image scatter over frames | frames as independent units; static scene |
| image information $\Lambda^I_m$, retained fraction $\kappa_m$ | Jacobians or injected curvature; Schur complement | restricted appearance model |
| per-frame component $\Sigma_\alpha$ | within-depth scatter (leave-block-out) | entity association |
| law of the surface term $\boldsymbol\mu_m$, prevalence per stratum | deconvolution of $\mathbf t_m$ over the entities of a stratum with kernel $(\Lambda^D_m)^{-1} + \hat\varphi (\Lambda^I_m)^{-1}$; interleaved frame folds as independent channels | image errors independent of depth errors; no image-consistent wrong geometry |
| repetition $\varrho(B)$ | variance components | as above |
| second layer | outline against interior | an attached outline in view |
| which family is wrong | attribution of Section 3.4 | priors from features; sign rule |
| anything in a direction with $\vartheta \approx 0$ and no feature support | not identified | |

The units of every per-stratum estimate are entities. Faults that are image-consistent (a planar
reflection without outline, an object moving along the strip at constant speed) bias $\mathbf t_m$ and
pass the within-image test; the blind spot of the image family is the counterpart of the repeated
fault of the depth family.

## 8. What Is Still Learned

The test is a statistic. What is learned are light heads on frozen foundation features, lifted onto
entities; no backbone is trained.

1. **Prior heads $\pi^D(\boldsymbol\phi_m)$, $\pi^I(\boldsymbol\phi_m)$:** which surfaces make the depth model fail, and which mislead the pixels. Inputs are frozen dense image features and video-encoder tokens of the *raw* clip, pooled over the entity; never the residuals. Trained by the marginal likelihood of the tension statistics, which is learning from labels whose noise is known per entity through the informations; an entity contributes in proportion to its label information. They are what reaches pixel-blind entities.
2. **Support head $e(\boldsymbol\phi)$:** predicts whether an entity of this appearance ever has image information. It needs no fault labels and separates interpolation from extrapolation; in the latter the output is *unsupported*, not a probability. The transfer rests on two assumptions: among entities that look alike, the witnessed ones fail as often as the others, and every appearance the head is applied to is sometimes witnessed.
3. **Dispersion per stratum,** and the bulk scale of $T$: a handful of numbers.
4. **Readout head by injection** (optional): amortises the sweep of a frozen video encoder.
5. **Strata:** clusters of frozen features, or concept prompts of a promptable segmenter confirmed by the layer test.

Depth consensus remains the examiner of the prior head on pixel-blind entities, for the per-frame
term only. A surface-term fault on a pixel-blind entity is beyond every label-free check.

## 9. Why Primitives

The primitives have a job if, and only if, the following are used: (i) both families are quadratic
forms on the same placement coordinates, with occlusion resolved, so that information ratios, tension
and gates are defined per direction; (ii) the factorisation of Section 2.2, which makes the image
message a reweighting of the depth message; (iii) entities, features and labels fused in one place by
the same weights; (iv) injection as a move of a group with correct occlusion. A per-frame pipeline
with warped depth maps can imitate (i) and (iv) for one reference frame at a time and cannot do (iii).
Any explicit surface with a differentiable renderer would serve; the planar surfel is the one for
which placement is three numbers and the factorisation is exact. If the test is run only as a
stabilised clip per frame, the primitives have no job and the note should say so.

## 10. What Exists in the Code and What Must Be Built

Read in the repository; nothing was run.

- *Exists.* A depth-only surfel fit on the 2DGS rasteriser with placeholder colour and its own ray-plane depth, because the rasteriser's blended depth uses surfel centres and does not see tilt. A reference renderer producing per-pixel, per-surfel weights and depth Jacobians, with the lumped depth information, the cavity render and its variance. A view-level gate with leave-block-out by subtraction. A separate photometric 2DGS trainer with spherical harmonics, densification, normal-consistency and distortion terms, not connected to the depth fit or to the gate. Fault injectors on depth maps.
- *To build.* Appearance channels on the depth-fitted state (projective texturing; linear diffuse fit). The image pull per primitive and entity. Entity information by finite differences with graph colouring, or by extending the reference renderer with the behind-appearance sums. Lifting of masks and features. The decomposition of Section 3.2, the dispersion, the attribution, the two gates. Injection as a group move.
- *Not available here.* The rasteriser library and the tensor library are not installed in the environment in which this note was written, so the rasteriser's interface for per-pixel weights and feature channels was not inspected.

## 11. What Would Refute the Approach, and the Cheapest Tests

1. **The pull is not a measurement.** On a synthetic scene with exact geometry, place an entity with a known offset, texture projectively from source frames, and compute the image pull and $T_m$ from judging frames with the reference renderer. Required: $\hat{\boldsymbol\theta}^I$ recovers the offset; $T_m$ on unperturbed entities follows $\chi^2_3$ after the dispersion. If the dispersion from model misfit is so large that offsets of the size of real faults are not significant, the image family is too noisy at surfel resolution to judge anything. Runs on a processor.
2. **No between-family variance.** From exact depth, the variance components of the monocular placement error per entity. If the surface term is negligible, consensus suffices.
3. **Errors do not follow entities.** Share of true monocular error that is a placement error of a lifted entity.
4. **Faults sit where pixels are blind.** Distribution of $\vartheta$ and $\kappa$ over true fault entities, and the share that is unsupported.
5. **A threshold on photometric error does as well.** The nearest prior methods mask monocular depth where an image-only fit has high photometric error, or where rendered depth fails a virtual stereo check. If such a pixel mask matches the indicator on the harm-weighted metric, including on repeated faults and on reflective entities, the statistical machinery is not earning its place.
6. **Attribution fails.** On reflective entities with correct depth, the fraction blamed on the depth prior.

Test 1 is the cheapest and decides whether the renderer can be a judge at all; tests 2 and 5 decide
whether there is a paper.

## 12. Prior Work: What Was Checked, and What Remains New

**Read for this note (abstract or full text of the cited preprint).**

| Work | What it does | Difference |
| --- | --- | --- |
| "In Depth We Trust: Reliable Monocular Depth Supervision for Gaussian Splatting" (arXiv 2604.05715) | Applies the monocular depth loss only on pixels where the rendered depth fails a virtual stereo consistency check with a fixed threshold; for 3D and 2D Gaussian splatting | Pixel mask from rendered geometry alone; no image evidence in the decision, no error bar, no treatment of errors repeated across views |
| "Reliability-Aware Monocular Depth Supervision for Sparse-View Neural Reconstruction" (arXiv 2607.02554) | Trains an image-only baseline, keeps the monocular depth loss where its photometric error is below a threshold chosen on validation data; verifies statistically, against lidar, that photometric error separates good from bad monocular depth | The closest in premise: pixels judge the depth prior. A fixed pixel mask from a separate fit; no information weighting, no test with a null distribution, no entities, no attribution |
| "Revisiting Photometric Ambiguity for Accurate Gaussian-Splatting Surface Reconstruction" (arXiv 2605.12494) | Per-Gaussian ambiguity indicator from the norm of higher-order spherical-harmonic coefficients, used to select Gaussians for prior-based regularisation | An appearance heuristic for where pixels are ambiguous; related to $\kappa_m$ in intent, not derived from information |
| "Uncertainty-guided Optimal Transport in Depth Supervised Sparse-View 3D Gaussian" (arXiv 2405.19657) | Weights depth supervision by an uncertainty taken from the depth diffusion model itself | Trust comes from the depth network, not from the images |
| "PUP 3D-GS" (arXiv 2406.10219) | Block-diagonal Fisher information per Gaussian over mean and scale from the rendering Jacobian, log-determinant score for pruning; computed over all training views in seconds with a dedicated kernel | Per-primitive image information exists and is cheap. Used for pruning, on the image loss only, not combined with a prior, and at primitive, not entity, level |
| "FlashSplat" (arXiv 2409.08270) | Lifts 2D masks to Gaussians in closed form by accumulating compositing weights per label | The lifting of Section 5.1 is this; it is not a contribution |
| "LUDVIG" (arXiv 2410.14462) | Learning-free uplifting of 2D features to Gaussians | Title and abstract level only; the feature lifting of Section 5.1 is of this kind |

**Recalled, not checked.** Depth-supervised radiance fields with depth uncertainty (DS-NeRF; dense
depth priors); depth-regularised Gaussian splatting (DNGaussian, DN-Splatter, FSGS); Fisher information
for view selection in splatting (FisherRF) and post-hoc spatial uncertainty (Bayes' Rays);
Levenberg-Marquardt optimisation of splats; feature and segment lifting (Feature 3DGS, LangSplat,
Gaussian Grouping); distractor masks from photometric residuals (RobustNeRF, SpotLessSplats,
WildGaussians); gradient surgery for conflicting losses (PCGrad); direct photometric bundle adjustment,
whose Hessian is the same object as $\Lambda^I$; score and Hausman tests, and prior-data conflict
diagnostics, in statistics. A search for a per-primitive or per-entity consistency test between a depth
prior and photometric evidence in splatting, in information form, returned none; the search was not
exhaustive.

**What remains new, in the part treated here.**

1. *The between-family test with its decomposition* (Section 3). Depth messages and image messages as information-form quantities on the same placement coordinates; the exact split of the misfit into consensus (within depth), ancillary (within image) and tension (between); the repeated fault identified as the between term. The nearest works threshold a pixel residual and have no null distribution.
2. *The score test in a model that cannot explain the fault away* (Section 4): tension evaluated at the depth-placed surface, with projective appearance or with the retained-information fraction $\kappa_m$ as the quantitative form of "appearance hides geometry".
3. *Attribution and two gates* (Section 3.4): three hypotheses per entity, the layer test, and a rule per placement direction for when to listen to which family.
4. *The factorisation of the image Jacobian through the depth Jacobian* (Section 2.2) and the observation that outline parallax with ownership is the label channel of the same message (Section 5.3). These are derivations, small but apparently unstated.

Items 1 and 2 carry the contribution. The premise that pixels can tell where monocular depth is wrong
is already supported in the literature, which lowers the novelty of the idea and raises its odds.

## 13. What This Theory Cannot Deliver

1. A test where the pixels are blind in every direction; there the indicator is a prior on appearance.
2. A verdict on image-consistent wrong geometry without an attached outline.
3. A calibrated null without the dispersion step, and a stated error rate without separate calibration.
4. Independence of the two families' errors, which share poses and discretisation.
5. Correct attribution when both families fail on the same entity.
6. Robustness to a wrong depth order in the model, on which ownership depends.
7. Entities smaller than a few surfels, for which the three placement parameters are not separately informed.
8. Any guarantee for the unrestricted joint fit: the test lives in the restricted model, and the final reconstruction must be shown separately not to reintroduce what the gates removed.
