# Two Messages on One Primitive: the Image Message from the Differentiable Renderer and the Tension Test

## 0. Scope, notation, and summary of findings

The setting is posed aerial video in strips, one monocular depth map per frame, and a 2D Gaussian
surfel reconstruction. Depth maps fail regionally and neighbouring frames repeat the failure. The
formulation worked out here fuses the two sources of geometric evidence on the same primitives: each
surfel receives a depth message from every frame's monocular map and an image message from every
frame's pixels, the second obtained through the differentiable renderer. The fault test is the tension
between the two families. Frozen foundation models supply segments and features, which are lifted
onto the same primitives; no backbone is fine-tuned.

This note derives the image message for a planar Gaussian primitive, states what it identifies and
where its linearisation is invalid, defines the tension statistic and its null, treats circularity and
the restriction of appearance, re-expresses on primitives what was established for surface entities
(common and per-frame error, outline parallax, reflections, harm and detectability, the placement
gate), and says what exists in the code and what must be built. Notation follows
`docs/learned-indicator.md`; additions:

| Symbol | Meaning |
| --- | --- |
| $\boldsymbol\vartheta_k = (\nu, \tau_1, \tau_2)$ | normal offset and tilts of surfel $k$, the sub-block $G_k\theta_k$ of its ten-dimensional perturbation |
| $\mathbf w_u$ | ray direction of pixel $u$ in world coordinates, scaled so that depth is $z$-depth |
| $\mathbf s_{uk} \in \mathbb R^2$ | local coordinates of the ray-surfel intersection, in units of the surfel's scales |
| $a_{uk}$, $T_{uk}$, $w_{uk} = a_{uk} T_{uk}$ | own alpha of surfel $k$ at pixel $u$, transmittance in front of it, compositing weight |
| $\mathbf q_{uk} \in \mathbb R^{1 \times 3}$ | derivative of the ray-surfel depth $d_{uk}$ with respect to $\boldsymbol\vartheta_k$ (`depth_jacobian_subblock`) |
| $\boldsymbol\kappa_{uk} \in \mathbb R^2$ | derivative of $\mathbf s_{uk}$ with respect to $d_{uk}$ |
| $\mathbf c_k$, $\mathbf B_{uk}$ | appearance (colour or lifted feature) of surfel $k$; what is composited behind it at pixel $u$ |
| $(\Lambda^D_k, \eta^D_k)$, $(\Lambda^I_k, \eta^I_k)$ | depth and image message of all frames to surfel $k$, in information form |
| $\mathcal A$, $\mathcal J$ | frames that supply appearance, frames that judge |
| $m$, $\Psi_m$ | a surface entity (a set of surfels), and its three placement directions |
| $\varsigma$ | share of the image message in the information on placement |

Nothing in this note has been measured, and no experiment was run for it. Statements about behaviour
are predictions.

Findings.

1. **The image message factorises through the depth Jacobian.** For a planar Gaussian the rendered
   colour depends on normal offset and tilt only through the depth at which each ray meets the plane.
   The image Jacobian with respect to $\boldsymbol\vartheta_k$ is $\mathbf g_{uk}\mathbf q_{uk}$, with the same row
   $\mathbf q_{uk}$ that the depth message uses and a colour sensitivity
   $\mathbf g_{uk} = -w_{uk}\, (\mathbf s_{uk}^\top \boldsymbol\kappa_{uk})\, (\mathbf c_k - \mathbf B_{uk})$. Both messages live in the same
   three-dimensional sub-block. This is the job of the planar primitive: it is the object on which
   depth and pixels constrain the same three numbers.
2. **One view carries no information on the offset; the spread of viewing directions does.** In one
   view a normal offset is confounded with a tangential shift of the surfel. After eliminating the
   tangential position, the information on the offset is texture energy times the weighted variance of
   $\boldsymbol\kappa$ across views. That is triangulation, obtained as a Schur complement, with
   occlusion handled by the rasteriser. It vanishes on blank surface, since $\mathbf c_k = \mathbf B_{uk}$ there.
3. **The local gradient is a test only inside a narrow basin.** The photometric linearisation holds
   for image shifts below a footprint. A regional depth fault of a few percent moves content by many
   footprints. At such a primitive the image gradient is not a large opposite pull; it is small and
   arbitrary. The per-primitive tension is therefore a screening statistic, and the test proper is a
   three-parameter profile per entity, obtained by moving the entity's surfels and re-rendering.
4. **Appearance must be lifted, not fitted, and judged on frames that did not supply it.** At a joint
   optimum the image pull equals minus the depth pull by stationarity, and fitted appearance bakes the
   compromise in. Appearance taken in closed form from frames $\mathcal A$ at the current geometry and
   judged on disjoint frames $\mathcal J$ has no free parameter with which to hide a misplaced surface, and
   it makes the injection identity exact.
5. **The gate is a projection in information form.** A flagged entity loses the depth information
   along its three placement directions and keeps the rest. Pixels then place the entity and the
   monocular map shapes it. Combined with the fault posterior, this is the answer to when a primitive
   should listen to which source: information decides the weights when both are trusted, and the
   indicator decides the trust.
6. **What is learned is small**: a probe on frozen features lifted to primitives, a noise scale per
   stratum, and a null calibrated by injection. Most of the method is estimation.
7. **The area is crowded and the claim must be narrow.** Fusing depth priors and pixels in Gaussian
   splatting is established, as is masking unreliable depth. What was not found in the work read is
   the two-source consistency test in information form with a stated null, the split of appearance
   and judging frames as its safeguard, and the placement-subspace gate.

## 1. The image message of a planar Gaussian

### 1.1 Rendering model

The ray of pixel $u$ of a frame meets the plane of surfel $k$ at depth
$d_{uk} = \mathbf n_k^\top(\boldsymbol\mu_k - \mathbf c)/\mathbf n_k^\top \mathbf w_u$ and at the point
$\mathbf x_{uk} = \mathbf c + d_{uk}\mathbf w_u$. Its local coordinates are
$s_{uk,a} = \mathbf t_a^\top(\mathbf x_{uk} - \boldsymbol\mu_k)/\sigma_{k,a}$, the own alpha is
$a_{uk} = \alpha_k \exp(-\tfrac12 \lVert \mathbf s_{uk} \rVert^2)$, and the rendered colour or feature is

```math
\mathbf C(u) = \sum_k a_{uk}\, T_{uk}\, \mathbf c_k + T_{u,\infty}\, \mathbf c_{\mathrm{bg}}, \qquad T_{uk} = \prod_{l \prec k} (1 - a_{ul}) ,
```

with $l \prec k$ the surfels in front of $k$ on the ray. Depth is rendered by the same weights.

### 1.2 The Jacobian with respect to offset and tilt

Perturb the surfel by a normal offset $\nu$ and tilts $\tau_a$ about its centre. The centre moves along
the normal and the tangent basis rotates, but because $\mathbf t_a^\top \mathbf n = 0$ and
$\mathbf n^\top(\mathbf x_{uk} - \boldsymbol\mu_k) = 0$, both contributions to $\mathbf s_{uk}$ cancel to first order. What remains is
the motion of the intersection point along the ray:

```math
\frac{\partial \mathbf s_{uk}}{\partial \boldsymbol\vartheta_k} = \boldsymbol\kappa_{uk}\, \mathbf q_{uk}, \qquad
\boldsymbol\kappa_{uk} = \frac{\partial \mathbf s_{uk}}{\partial d_{uk}} = \begin{bmatrix} \mathbf t_1^\top \mathbf w_u / \sigma_{k,1} \\ \mathbf t_2^\top \mathbf w_u / \sigma_{k,2} \end{bmatrix}, \qquad
\mathbf q_{uk} = \frac{\bigl[\,1,\ -\mathbf t_1^\top(\mathbf x_{uk} - \boldsymbol\mu_k),\ -\mathbf t_2^\top(\mathbf x_{uk} - \boldsymbol\mu_k)\bigr]}{\mathbf n_k^\top \mathbf w_u} .
```

The geometry of the plane enters the image only through the ray depth, and $\mathbf q_{uk}$ is the row
that the code already uses for the depth message. With
$\partial a_{uk}/\partial \mathbf s_{uk} = -a_{uk}\, \mathbf s_{uk}^\top$ and
$\partial \mathbf C(u)/\partial a_{uk} = T_{uk}(\mathbf c_k - \mathbf B_{uk})$, where $\mathbf B_{uk}$ is the colour composited
behind surfel $k$ as seen through it (the remaining surfels and the background, without the factor
$1 - a_{uk}$),

```math
\frac{\partial \mathbf C(u)}{\partial \boldsymbol\vartheta_k} = \mathbf g_{uk}\, \mathbf q_{uk}, \qquad
\mathbf g_{uk} = \frac{\partial \mathbf C(u)}{\partial d_{uk}} = -\, w_{uk}\, \bigl(\mathbf s_{uk}^\top \boldsymbol\kappa_{uk}\bigr)\, \bigl(\mathbf c_k - \mathbf B_{uk}\bigr) .
```

Each factor has a meaning. $w_{uk}$: the surfel must be visible, which the rasteriser decides.
$\mathbf c_k - \mathbf B_{uk}$: its appearance must differ from what would show in its place; on blank surface the
difference vanishes and with it the message, without an abstention rule. $\mathbf s_{uk}^\top\boldsymbol\kappa_{uk}$: the
signal sits on the flanks of the footprint, and only for rays oblique to the plane in a tangent
direction. The derivation holds for any per-surfel appearance vector, so a feature-metric loss has the
same Jacobian with $\mathbf c_k$ a lifted feature (Section 5).

The dependence of the weights on a surfel's *own* geometry was dropped from the depth message
(compositing weights held fixed). In the image message it is the whole signal.

### 1.3 One view is not enough: the offset as a Schur complement

A tangential shift $\delta\boldsymbol\mu_t$ of the surfel changes the local coordinates by
$-\mathbf t_a^\top \delta\boldsymbol\mu_t/\sigma_a$ at every pixel. A normal offset changes them by
$\boldsymbol\kappa_{uk}\nu/\mathbf n^\top\mathbf w_u$, which is nearly constant over a small footprint. In one view the
two are indistinguishable: a surfel pushed along its normal and pulled back sideways renders the same.
The image message must therefore be formed on the joint block of position and tilt, and the
information on the offset is what remains after eliminating the tangential position.

In one tangent direction, with $E_i = \sum_{u \in i} \lVert w_{uk}\, s_{uk} (\mathbf c_k - \mathbf B_{uk}) \rVert^2/\sigma_I^2$ the
texture energy of the surfel in view $i$, and $\kappa_i$ its value of $\kappa/(\mathbf n^\top \mathbf w)$,

```math
\Lambda_{(\nu, \mu_t)} = \begin{bmatrix} \sum_i E_i \kappa_i^2 & -\sum_i E_i \kappa_i \\ -\sum_i E_i \kappa_i & \sum_i E_i \end{bmatrix}
\quad\Longrightarrow\quad
\lambda^I_\nu = \sum_i E_i \kappa_i^2 - \frac{\bigl(\sum_i E_i \kappa_i\bigr)^2}{\sum_i E_i} = \Bigl(\sum_i E_i\Bigr) \operatorname{Var}_E(\kappa) .
```

The information on the offset is the texture energy times the energy-weighted variance of the tangent
of the viewing angle over the views. This is the precision of a triangulated point, $\sum_i b_i^2/\sigma_i^2$
with the reference position eliminated, reached without correspondences: the role of the tracked
position is played by the surfel's tangential coordinates. Three consequences.

- The image message is taken on the nine geometric coordinates and reduced to $\boldsymbol\vartheta_k$ by a
  Schur complement. Using only the $3\times 3$ block on $\boldsymbol\vartheta_k$ would overstate the
  information by the term subtracted above.
- Appearance is a further nuisance. With one colour per surfel and one gain and offset per image the
  subtraction is small. With appearance free to vary by view (spherical harmonics of higher degree,
  per-view colours) the subtracted term grows until nothing remains: a pattern that is free in every
  view cannot contradict any geometry. Section 4 fixes this.
- A strip gives a spread of $\kappa$ along the flight direction only. Tilt across the strip is
  determined through the variation of $\mathbf q_{uk}$ over the footprint and over the entity, not through a
  second baseline, and is weaker.

### 1.4 Exact, diagonal, and lumped

The Gauss-Newton matrix of the image loss has cross-blocks between surfels that share pixels,
$H^I_{kl} = \sum_u \mathbf q_{uk}^\top\, \mathbf g_{uk}^\top \mathbf g_{ul}\, \mathbf q_{ul}/\sigma_I^2$. Keeping only the diagonal blocks is not a
bound and overshoots when overlapping surfels need a common shift, as for the depth message. The
Jensen argument used for depth does not carry over, because the sensitivities $\mathbf g_{uk}$ are not
non-negative weights that sum to one. A majoriser follows from the Cauchy-Schwarz inequality with
weights proportional to the magnitude of the sensitivities:

```math
\Lambda^{I}_{k} = \sum_i \sum_{u \in i} \frac{\lVert \mathbf g_{uk} \rVert \, \sum_l \lVert \mathbf g_{ul} \rVert}{\sigma_I^2}\ \mathbf q_{uk}^\top \mathbf q_{uk}
\ \succeq\ \text{the corresponding blocks of } H^I \text{ in the sense } \theta^\top H^I \theta \le \sum_k \theta_k^\top \Lambda^I_k \theta_k .
```

A step with these blocks cannot overshoot. They overstate the information of differential modes, so
they are conservative for moves and optimistic as covariances. At the level of an entity none of this
is needed: the reduced information on three placement parameters is a $3\times 3$ matrix computed
exactly from three directional derivatives of the render (Section 3.2).

### 1.5 Where the linearisation fails

The Jacobian describes image shifts small against the footprint of a surfel. A surfel of a few pixels
at a point where a 2% depth error moves content by more than ten pixels between distant frames is far
outside that range. There the gradient reflects whatever unrelated texture the misplaced footprint
overlaps, the curvature is the curvature of the wrong basin, and neither is an error bar. The picture
of a faulty primitive held at a compromise by two large opposite pulls is correct for small faults and
for coarse appearance; for the regional faults of interest the image pull at the compromise is
typically weak. Two remedies, used together: appearance with a wide basin (features at token
resolution, or blurred images, for the first pass), and a profile over a finite range of placement
per entity in place of a local gradient (Section 3.2). The per-primitive quantities remain useful for
screening and for the final, local stage.

## 2. The depth message, and what the two messages share

The depth message of frame $i$ to surfel $k$ under pixel weights $\omega_i$ is the lumped block of the
main document,

```math
\Lambda^D_{ik}[\omega_i] = c_{\mathrm{eff}} \sum_{u \in i} \omega_i(u)\, \frac{a_i^2\, \hat w_{uk}}{\sigma_i^2(u)}\ \mathbf q_{uk}^\top \mathbf q_{uk}, \qquad
\eta^D_{ik}[\omega_i] = c_{\mathrm{eff}} \sum_{u \in i} \omega_i(u)\, \frac{a_i\, \hat w_{uk}}{\sigma_i^2(u)}\ \mathbf q_{uk}^\top e_i(u) ,
```

with $\hat w_{uk}$ the normalised weights and $e_i$ the depth residual; the lumped form majorises the
Hessian by Jensen's inequality, and blocks built from squared weights overshoot the common mode by
$\sum_u \hat w_{uk}/\sum_u \hat w_{uk}^2$. Both messages are quadratic forms in the same rows $\mathbf q_{uk}$, weighted
by $a_i^2 \hat w_{uk}/\sigma_i^2$ for depth and by the squared colour sensitivity for the image. They differ
in three respects that matter for a test.

| | Depth message | Image message |
| --- | --- | --- |
| nuisance per frame | scale, offset, shared distance bias | gain and offset of exposure; appearance of the surfels |
| one frame alone | informs offset and tilt | informs neither; only the spread of views does |
| blank surface | informative | silent |
| a fault every frame repeats | all frames pull the same way | unaffected |
| reflecting surface | wrong or right, depending on the network | consistent pull toward the virtual image |

**Common and per-frame error.** Let $\boldsymbol\epsilon_{mi}$ be the placement error of the depth message of frame
$i$ on entity $m$, with information $\Lambda^D_{mi}$. The fused depth placement is the
information-weighted mean $\bar{\boldsymbol\epsilon}_m$, and depth consensus is a function of the contrasts
$\boldsymbol\epsilon_{mi} - \bar{\boldsymbol\epsilon}_m$ alone, because the depth messages at a primitive sum to zero at a
depth-only fit. The image message does not enter that sum. Its comparison with the fused depth
placement tests the mean, which is the component consensus cannot see, and the two tests are
independent under the Gaussian model. The repetition coefficient is the share of the mean in the
variance of $\boldsymbol\epsilon_{mi}$; an estimate of a frame's error is the image-judged mean plus the consensus
contrast. One image test per entity completes the per-frame consensus tests.

## 3. The tension test

### 3.1 Per primitive

Let $\mathbf m^D_k = (\Lambda^D_k)^{-1}\eta^D_k$ and $\mathbf m^I_k = (\Lambda^I_k)^{+}\eta^I_k$ be the placements the
two families ask for, relative to the current state, with $\Lambda^I_k$ the reduced block of Section
1.3. The consistency statistic of two independent estimates of one quantity is

```math
T_k = \bigl(\mathbf m^D_k - \mathbf m^I_k\bigr)^\top \bigl((\Lambda^D_k)^{-1} + (\Lambda^I_k)^{+}\bigr)^{+} \bigl(\mathbf m^D_k - \mathbf m^I_k\bigr)
\ \sim\ \chi^2_{\operatorname{rank} \Lambda^I_k} \quad \text{under the nominal model.}
```

At a joint fit with both families in the objective, stationarity gives $\eta^I_k = -\eta^D_k$ there (up
to the prior), and the same statistic is

```math
T_k = \eta^{D\top}_k \bigl((\Lambda^D_k)^{-1} + (\Lambda^I_k)^{+}\bigr)\, \eta^D_k ,
```

the squared pull of the depth family measured in the metric of both. It can be accumulated during
fitting at the cost of the information blocks. As $\Lambda^I_k \to 0$ the fit follows the depth,
$\eta^D_k \to 0$ faster than the metric grows, and $T_k \to 0$: a primitive without image information is
never flagged by tension. It is in the state *unsupported*, which is distinct from *consistent*.

The nominal law is not to be trusted at this level. The image noise is not Gaussian or independent
across pixels; the rendering model is wrong at every surfel to some degree (finite footprints,
aliasing, exposure); the reduced image block is estimated; and outside the basin of Section 1.5 the
statistic is meaningless. $T_k$ is a ranking for screening. Its scale is set per appearance stratum
from the bulk of primitives, which are healthy, and its use is to propose entities and to localise a
fault inside one.

### 3.2 Per entity: a profile with re-lifted appearance

For an entity $m$, let $\Psi_m$ map three placement parameters $\boldsymbol\delta$ (offset and two gradients of
the inverse-depth error over the entity, in a reference parametrisation) to the normal offsets and
tilts of its surfels. Moving the entity by $\boldsymbol\delta$ and re-rendering every judging view is an exact
operation of the renderer, occlusion changes included. With the appearance of the surfels taken from
frames $\mathcal A$ at the moved geometry (Section 4), define the image profile and its estimate

```math
\ell^I_m(\boldsymbol\delta) = \sum_{j \in \mathcal J} \sum_{u} \rho\Bigl( \frac{\mathbf I_j(u) - \mathbf C_j\bigl(u;\ \boldsymbol\vartheta + \Psi_m \boldsymbol\delta,\ \mathbf c[\mathcal A, \boldsymbol\delta]\bigr)}{\sigma_I} \Bigr),
\qquad
\hat{\boldsymbol\delta}^I_m = \arg\min_{\boldsymbol\delta} \ell^I_m(\boldsymbol\delta), \qquad
\Lambda^I_m = \nabla^2_{\boldsymbol\delta}\, \ell^I_m \big|_{\hat{\boldsymbol\delta}^I_m} ,
```

over pixels that the entity covers in some rendering of the sweep, with a robust loss $\rho$. A coarse
sweep of the offset over the range of plausible faults, in appearance with a wide basin, locates the
minimum; Gauss-Newton on three parameters refines it, with the exact $3\times 3$ information from three
directional derivatives of the render (finite differences: six extra renders per judging view). The
depth family gives $\hat{\boldsymbol\delta}^D_m = (\Psi_m^\top \Lambda^D_m \Psi_m)^{-1}\Psi_m^\top \eta^D_m$ with information
$\Psi_m^\top\Lambda^D_m\Psi_m$ in closed form. The entity statistic is

```math
T_m = \bigl(\hat{\boldsymbol\delta}^D_m - \hat{\boldsymbol\delta}^I_m\bigr)^\top \bigl( V^D_m + V^I_m \bigr)^{-1} \bigl(\hat{\boldsymbol\delta}^D_m - \hat{\boldsymbol\delta}^I_m\bigr) .
```

**Injection identity.** Because moving surfels and re-rendering is exact, an entity displaced by a
known $\boldsymbol\delta_0$ must return $\hat{\boldsymbol\delta}^I_m - \boldsymbol\delta_0$. Displacing entities by known amounts and
reading the estimates gives, without labels: the bias of the readout; its variance $V^I_m$ as a
function of the geometric information, replacing the curvature where the curvature is unreliable; the
range of displacements from which the minimum is recovered (the capture range); and an abstention
rule for entities on which the injected displacement is not recovered. This holds for any appearance
channel, including tokens of a frozen encoder, because only the location of a minimum is used.

**Null.** Under the nominal model $T_m$ is $\chi^2_3$. In practice the law of $T_m$ on healthy entities
is estimated per stratum from the bulk, with entities as the exchangeable units, and checked by
injection: displaced healthy entities must be detected at the rate their noncentrality predicts.

**Ancillaries.** Quantities that do not depend on the placement error accompany each profile:
the per-frame image registration of the entity *across* the epipolar direction (rigidity, pose and
timing, stability of the appearance); the residual of per-frame registrations *along* it about the
fitted placement (departure from a single placement, movers); the agreement of the estimates from two
halves of the judging frames; and the heterogeneity of the per-primitive estimates inside the entity,

```math
Q^I_m = \sum_{k \in m} \bigl(\mathbf m^I_k - \Psi_{mk} \hat{\boldsymbol\delta}^I_m\bigr)^\top \Lambda^I_k \bigl(\mathbf m^I_k - \Psi_{mk} \hat{\boldsymbol\delta}^I_m\bigr),
```

which tests that one placement serves the whole entity.

### 3.3 What is identified

| Quantity | Identified by |
| --- | --- |
| common placement error of the depth prior on an entity, textured or with an owned outline | $\hat{\boldsymbol\delta}^D_m - \hat{\boldsymbol\delta}^I_m$ |
| per-frame deviations | consensus contrasts between blocks of frames |
| repetition coefficient | variance of the mean against variance of the contrasts |
| noise of the image readout, its capture range | injection, split halves |
| error of shape inside an entity | $Q^I_m$ where primitives carry image information; otherwise not |
| image errors proportional to parallax (virtual images, motion along the strip) | not within a strip; partly by the layer test of Section 6 |
| anything on an entity with no image information | not identified; the state is unsupported |

## 4. Circularity and the restriction of appearance

Three mechanisms let a joint model hide a misplaced surface, and each has a specific counter.

**Fitted appearance bakes in the compromise.** If colours are optimised at the joint optimum, they are
whatever makes the render match there: a blur of misregistered views. The image profile then has its
minimum where the surface already is. *Counter:* appearance is not optimised. It is lifted in closed
form from the frames $\mathcal A$ at the geometry under test,

```math
\mathbf c_k[\mathcal A, \boldsymbol\delta] = \frac{\sum_{i \in \mathcal A} \sum_{u \in i} w_{uk}(\boldsymbol\delta)\, \bigl(\mathbf I_i(u) - \beta_i\bigr)/\gamma_i}{\sum_{i \in \mathcal A} \sum_{u \in i} w_{uk}(\boldsymbol\delta)} ,
```

with exposure gain $\gamma_i$ and offset $\beta_i$ per image, and it is re-lifted for every placement.
Lifting is the adjoint of rendering under the measure $\sum_u w_{uk}$ and is the lumped least-squares
solution for appearance. With nothing fitted, the comparison of the render with the frames $\mathcal J$ is a
photo-consistency test between two disjoint sets of frames through the surface, which is zero only if
the surface is right or the appearance is uninformative.

**View-dependent appearance, opacity and floaters explain parallax.** *Counter:* for the test the
appearance is one vector per surfel (no view dependence), opacity is frozen, and no primitive is
created or removed. View dependence that is real then appears as tension and is handled as a
material question (Section 6), not absorbed.

**Judge and judged share a model.** Three dependencies remain and are treated differently. The
judging frames $\mathcal J$ never supply appearance; $\mathcal A$ and $\mathcal J$ are interleaved along the strip so that both
span the baseline, and exchanging their roles gives the second half-estimate. The depth maps of the
judging frames do enter the depth family, which is intended: the test is between families, not between
frames. And the surface used for visibility, entity membership and ownership is the surface under
test. That dependence is weak (bulk motion, region-scale association, the sign of a depth step) but
not absent; where a fault created its own surfels, or two depths are blended, visibility itself is
wrong and the entity is reported as unsupported.

**Which surface is tested.** Three variants are possible: the depth-only fit judged by the pixels
(pixels are a pure witness, as in a depth-stabilised clip, which is the special case of lifting from
one frame through one depth map); the joint fit with the tension read from the pulls; and a fit in
which one half of the frames contributes pixels and the other half judges. The first keeps the families
independent and is the cleanest test. The second is what a production fit would run. The recommended
protocol tests on the first and reconstructs with the second, gated.

## 5. Entities and features on the same primitives

**Entities.** Segmenters applied per frame return masks without identities across frames. Each mask is
lifted to a soft set of surfels, $(\mathsf P_i M)_k = \sum_{u \in i} w_{uk} M(u)/\sum_{u \in i} w_{uk}$, and masks of different
frames that land on the same surfels are merged; an entity is a set of surfels. This uses the judged
geometry for association, which is admissible at region scale and fails when the depth error is
comparable to the size of the region. The safeguard is the same split: membership lifted from $\mathcal A$ is
rendered into $\mathcal J$ and compared with the masks found there.

**Outline parallax as a rendered channel.** Membership is an appearance channel like any other. Its
rendering loss has the Jacobian of Section 1.2 with $\mathbf c_k$ the membership and $\mathbf B_{uk}$ the
membership behind, so it is non-zero exactly at the outline of the entity and gives the parallax of
the outline for an interior without texture. Ownership needs no hand-set weights: where a nearer
object hides part of the entity, the rendered outline is produced by the occluder's surfels at the
occluder's depth and does not move with the entity. This is correct to the extent that the occluder is
correctly placed. The aperture limit remains: outline segments parallel to the flight direction carry
no information, and an outline cut by the image border is absent.

**Features.** Frozen dense features lift by the same operator and render to any view, so the profile
of Section 3.2 can be feature-metric. Coarse features widen the capture range and lower the precision;
the two are used in sequence. The lifted field has the resolution of the backbone's patches.

**A frozen video encoder** can compare a rendered clip with the real one in its latent space, and the
injection identity makes the location of the minimum usable. With a renderer and lifted dense features
available, this adds an instrument without adding information, and renders with one appearance vector
per surfel are outside the encoder's training distribution. It is kept as a comparison, not as part of
the method.

## 6. Reflections, movers and the layer test

A virtual image behind flat glass or still water is a rigid scene. On primitives it has a precise
form: the image family asks for a placement behind the surface, consistently over frames, and passes
every ancillary. Three readings distinguish it from a faulty depth prior.

- *Outline against interior.* The membership channel and owned edges place the entity at the surface;
  the interior appearance places it behind. A significant difference toward greater depth marks the
  entity as reflecting or transparent. Its interior image evidence is then discarded as a judge of
  the depth prior.
- *Heterogeneity.* $Q^I_m$ is large, with the per-primitive estimates forming two groups by position
  (rim and interior).
- *Two layers.* Allowing a second set of primitives behind the entity, with the first made partly
  transparent, lowers the image loss on the judging frames for a reflecting surface and not for a
  diffuse one. The gain in held-out loss of the two-layer model is a layer test; it is the rendering
  form of a ray that pierces a surface.

A material mark obtained this way is an attribute of the entity that costs no labels. Without an
owned outline the conflict cannot be attributed and the entity is unknown. A vehicle moving along the
strip appears as a small entity whose image placement leaves a well-determined road; dense traffic on a
blank road is not caught.

## 7. Listening to the prior or to the pixels

**Harm and detectability.** With $\varsigma_m = \Lambda^I_m/(\Lambda^I_m + \Lambda^D_m)$ on the offset, a common
error $\Delta$ of the depth prior moves a jointly fused entity by $h = \Delta(1 - \varsigma_m)$ and gives the
tension the noncentrality

```math
\delta^2 = \Delta^2\, \Lambda^D_m\, \varsigma_m = h^2\, \Lambda^D_m\, \frac{\varsigma_m}{(1 - \varsigma_m)^2} .
```

In a joint fit the first relation is no longer hypothetical: where the pixels are informative they
outvote a faulty prior and the fault does little harm without any test. Where they are not, the fault
does full harm and the tension is blind. The indicator changes the reconstruction in between and,
through the probe, at entities without image information.

**The gate.** Under the model that the monocular shape of an entity is right and its placement may be
wrong, a flagged entity should lose the depth information on placement and keep the rest. In
information form, with trust $\omega_m = 1 - \zeta_m$,

```math
\Lambda^D_m \ \longrightarrow\ \Lambda^D_m - (1 - \omega_m)\, \Lambda^D_m \Psi_m \bigl(\Psi_m^\top \Lambda^D_m \Psi_m\bigr)^{-1} \Psi_m^\top \Lambda^D_m ,
```

and correspondingly for $\eta^D_m$. The subtracted term is the projection onto the placement directions
in the metric of the depth information, so for $\omega_m = 0$ the depth family says nothing about where
the entity is and everything it knew about its shape. The fused placement is then

```math
\hat{\boldsymbol\delta}_m = \bigl(\omega_m \Psi_m^\top\Lambda^D_m\Psi_m + \Lambda^I_m\bigr)^{-1} \bigl(\omega_m \Psi_m^\top \Lambda^D_m \Psi_m\, \hat{\boldsymbol\delta}^D_m + \Lambda^I_m\, \hat{\boldsymbol\delta}^I_m\bigr) .
```

This is the answer the formulation gives to the question of when a primitive should listen to which
source. If both are trusted, information decides, and no indicator is needed. The indicator exists
because the depth message is biased with a probability that depends on the surface, and that
probability multiplies its placement information. An entity gated with $\Lambda^I_m \approx 0$ has no
placement at all and is reported as unconstrained.

## 8. What is learned

- *A probe on frozen features.* Features of a dense image encoder are lifted to primitives and pooled
  per entity. A linear or two-layer map on them and on geometry scalars (range, incidence, entity
  size, $\varsigma_m$) gives the prior probability that the depth placement of an entity is faulty.
  It is trained toward the posterior from $T_m$, weighted by label information, on entities where the
  image family is informative, and applied to unsupported entities. It does not receive the tension
  or any motion-aware feature of the entity it judges, and it is cross-fitted by ground tile, since an
  entity is fixed in the scene and seen from many frames.
- *Scales.* The image noise scale and the inlier spread of the placement gap per appearance stratum,
  from the bulk of entities.
- *Calibration.* The injection curves of Section 3.2. With a labelled synthetic scene, the same
  quantities are checked against exact depth; labels are not needed to fit them.

The transfer from informative to unsupported entities is a selective-labels problem. Outline parallax
through the membership channel gives verdicts on blank entities with owned outlines, so those
appearance classes are represented among the labels, and training on entities judged by interior
texture and testing on entities judged by outline alone measures the transfer on the video itself.
Entities that are blank, without an owned outline in the frame, or wrong in shape receive no verdict of
either kind.

## 9. What exists in the code and what must be built

Read in the repository:

- `src/sensorfault/gs/fit.py` fits surfels with a depth loss only. Its default path is a torch port of
  a dense ray-plane compositor; its alternative calls the 2DGS rasteriser with constant colours and
  reads depth, and the module notes that this path blends per-surfel centre depth and does not see
  tilt. Opacity is trained in a warm-up and frozen. Rotation is parametrised by a body-frame vector,
  so derivatives with respect to tilt are directly available.
- `src/sensorfault/eval/gaussian_uncertainty.py` has the closed-form depth Jacobian
  $\mathbf q_{uk}$, lumped information blocks, fusion, leave-out and the support rules. The reference backend
  exposes per-(pixel, surfel) arrays of weights, depths and kernels.
- `experiments/rgb_train_2dgs.py` is a separate photometric trainer (L1 and D-SSIM, spherical
  harmonics, normal-consistency and distortion regularisers) on its own primitives.

To build:

1. One set of primitives carrying geometry and a lifted appearance vector, with the closed-form
   lifting and the split into $\mathcal A$ and $\mathcal J$.
2. The image sensitivity $\mathbf g_{uk}$ in the reference compositor. All its factors
   ($w_{uk}$, $\mathbf s_{uk}$, $\boldsymbol\kappa_{uk}$, the composite behind) are present or one accumulation away.
   This gives exact per-primitive image blocks on small scenes and the reference against which
   anything faster is checked.
3. The entity profile by re-rendering: a sweep and six finite-difference renders per judging view,
   with any rasteriser that renders $N$-channel appearance. Entities that do not overlap in any view
   are perturbed in the same pass.
4. On the GPU rasteriser, the gradient $\eta^I_k$ comes from one backward pass. The curvature does not.
   Options are random probes (the outer product of $J^\top z$ for random sign vectors $z$ over pixels
   estimates the exact diagonal block, at one backward pass per probe and a relative error of order
   $\sqrt{2/M}$ for $M$ probes), or a per-fragment accumulation inside the rasteriser, which the
   lumped form of Section 1.4 requires.

The rasteriser library is not installed in the environment in which this note was written. Its
interface was read only as used in the two files above; whether it renders feature channels in its 2DGS
path, exposes per-fragment quantities, or supports forward-mode derivatives was not checked.

Cost, as estimates from counts: lifting is one pass per frame of $\mathcal A$; a sweep of $L$ offsets is $L$
renders per judging frame at reduced resolution; refinement is a few times seven renders per judging
frame. For some tens of judging frames per window this is hundreds of renders per window, small next
to a fit.

## 10. Prior work, what remains new, what would kill it, what is unresolved

**Read for this note** (abstracts, and for one paper the method section):

- *In Depth We Trust* (arXiv 2604.05715). Method read: a mask from the inconsistency of the *rendered*
  depth between the training camera and a virtual camera shifted sideways decides where an absolute
  monocular depth loss applies; a gradient-of-depth loss applies everywhere. No statistical test, no
  information, no comparison of photometric and depth gradients. It applies the prior where geometry
  is ill-posed, which is the complement of the question of where the prior is wrong.
- *Reliability-Aware Monocular Depth Supervision* (arXiv 2607.02554). Abstract: photometric masks from
  an RGB-only baseline select regions where depth supervision is applied.
- *MonoInstance* (arXiv 2503.18363). Abstract: monocular depths of segmented instances are aligned
  across views in 3D and the density of the resulting clusters is an uncertainty per instance, used
  to modulate the prior. This is the closest precedent for a per-entity judgement of a monocular
  prior, and it is consensus between depth maps.
- *FisherRF* (arXiv 2311.17874). Abstract: Fisher information of radiance-field parameters for view
  selection and pixel-wise uncertainty. Recalled, not read: a diagonal approximation over Gaussian
  parameters from the rendering loss.
- *3DGS-LM* (arXiv 2409.12892) and matrix-free second-order optimisation of splats (arXiv 2504.12905).
  Abstracts: Levenberg-Marquardt with matrix-free products exploiting the sparsity of the Jacobian.
  They establish that Gauss-Newton quantities of the rendering loss are computable at scale.

**Recalled, not checked:** depth-regularised and normal-regularised splatting (several methods);
multi-view photometric constraints on planar Gaussians through plane-induced homographies;
confidence-weighted depth fusion in splatting-based SLAM; feature splatting and language-embedded
splats; lifting of 2D segments to Gaussians with identity matching; post-hoc spatial uncertainty of
radiance fields; gradient-surgery methods for conflicting losses; the plane-plus-parallax literature;
the classical test of consistency between two information sources in adjustment theory. Each must be
read before a claim is fixed.

**What remains new, as far as this reading goes.**

1. The image message of a planar Gaussian in the same sub-block as the depth message, with the
   factorisation $\mathbf g_{uk}\mathbf q_{uk}$ and the offset information as a Schur complement equal to texture
   energy times the spread of viewing directions. The ingredients are standard; the statement that
   both families constrain the same three parameters of one primitive, and its use, were not found.
2. Tension as a two-source consistency test with a stated null, entity-level profile, injection
   calibration and ancillaries, in place of a mask from a threshold.
3. Lifted, never fitted, appearance with disjoint supplying and judging frames as the safeguard that
   makes the test valid inside a joint model.
4. The placement-subspace gate.

If per-primitive photometric and depth gradients have already been compared as a reliability signal,
item 2 narrows to its statistical treatment. Item 1 is a derivation, not a result. The work stands as
a paper on items 2 to 4 together with the evidence that faults repeated by every frame are found and
repaired; none of that evidence exists yet.

**What would kill it, cheapest first.** Exact depth from a synthetic scene is used for evaluation only.

1. *Capture range and bias of the entity profile.* Place surfels from exact depth, displace entities
   by known amounts, lift from $\mathcal A$, judge on $\mathcal J$, and read the recovered placement by material and
   texture. If the minimum is not recovered over the range of real faults, or is biased by blur and
   exposure, there is no image witness. This needs a renderer and exact depth, no depth network.
2. *The null.* With exact depth as the depth family, the distribution of $T_m$ over entities. If it is
   overdispersed by an order of magnitude from rendering-model error, tension measures the renderer
   and not the prior.
3. *Where the harm is.* The share of fault area, weighted by $1 - \varsigma_m$, on entities with image
   information, with outline information only, and with neither. If almost all weighted harm sits in
   the last group, the test judges what the joint fit already corrects and the method rests on the
   probe alone.
4. *Hiding.* The same tension with appearance fitted and view-dependent against lifted and
   restricted. If the difference is small, the safeguard of Section 4 is unnecessary; if tension
   vanishes under fitted appearance, the per-primitive statistic read during an ordinary joint fit is
   useless and only the split protocol remains.
5. *Redundancy of the prior.* Pixels alone, prior alone, joint, and gated joint, on the harm-weighted
   metric.

**Unresolved.**

- *Entities without image information* remain unjudged: blank interior with no owned outline in the
  frame, or with the outline along the flight direction. The probe's error there cannot be estimated
  from the video.
- *Errors of shape.* The gate and the profile assume that an entity is misplaced, not deformed. Shape
  errors are visible only through $Q^I_m$ and only where primitives have texture.
- *The null of the image family* rests on a noise model for rendering residuals that does not exist
  yet; calibration by injection covers the readout, not unmodelled appearance change between strips.
- *Visibility from the surface under test.* Occlusion, membership and ownership are computed on the
  geometry being judged.
- *Per-primitive curvature at scale* needs either stochastic probes of limited accuracy or changes
  inside the rasteriser.
- *Not verified:* every numerical illustration; the rasteriser's capabilities beyond what the
  repository's own calls show; all prior work listed as recalled; the methods of the papers of which
  only the abstract was read.
