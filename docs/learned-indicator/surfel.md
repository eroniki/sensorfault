# The Fault Indicator in Surfel Space: Monocular Depth Judged by Motion Parallax

## 0. Scope, notation, and summary of findings

The setting is posed aerial video in strips, one monocular depth map per frame, and a
depth-supervised 2D Gaussian surfel reconstruction. The indicator judges the monocular depth by
motion parallax from the same video: a frozen point tracker or flow model gives correspondences, the
known poses turn each correspondence into a depth with an uncertainty set by its parallax, and a light
head on frozen features carries the verdict to pixels where tracking abstains. No backbone is
fine-tuned; labels from a synthetic scene are optional calibration.

This note develops the part of that formulation that lives on the surfel map: how a track becomes a
message to a surfel in the same information form as a depth map, what the comparison of the two
identifies, where it cannot help, how sparse track evidence is extended over untextured surface, how
labels and frozen features move between pixels and surfels, and how the indicator enters the fit.
Notation follows `docs/learned-indicator.md`; additions are below. Nothing in this note has been
measured. Statements about expected behaviour are predictions.

| Symbol | Meaning |
| --- | --- |
| $\theta_k \in \mathbb R^{10}$ | perturbation of surfel $k$: $[\delta\mu\ (3),\ \delta\phi\ (3),\ \delta\log s\ (3),\ \delta\operatorname{logit}\alpha\ (1)]$; the first nine are geometric |
| $G_k \in \mathbb R^{3\times 10}$ | map to normal offset and tilt, $(\nu, \tau_1, \tau_2) = G_k\theta_k$ (`subblock_map`) |
| $w_{uk}$ | normalised compositing weight of surfel $k$ at pixel $u$ (`w_hat`); $\sum_k w_{uk} = 1$ on covered pixels |
| $\mathbf q_{uk} \in \mathbb R^{1\times 3}$ | derivative of the ray-surfel depth with respect to $(\nu,\tau_1,\tau_2)$ (`depth_jacobian_subblock`); $\mathbf j_{uk} = \mathbf q_{uk} G_k$ |
| $(\Lambda^{m}_{ik}, \eta^{m}_{ik})$ | message of the monocular depth map of view $i$ to surfel $k$ |
| $t$, $u_t$, $\mathcal J_t$ | a track, its pixel in its reference frame, the frames in which it is observed |
| $\mathbf x_{tj}$, $\Sigma_{tj}$ | tracked position in frame $j$ and its $2\times 2$ covariance |
| $d^{\mathrm{tr}}_t$, $\lambda_t$ | parallax depth of track $t$ along its reference ray, and its precision |
| $(\Lambda^{\mathrm{tr}}_{k}, \eta^{\mathrm{tr}}_{k})$ | summed track messages to surfel $k$ |
| $\varsigma_k$ | parallax share of the information of surfel $k$ (Section 4) |
| $n_{ik} = \sum_{u \in i} w_{uk}$ | effective number of pixels of view $i$ rendered by surfel $k$ |
| $w_i(u) = 1 - \zeta_i(u)$ | gate on a pixel |

Findings.

1. **A track is a depth message.** Along its reference ray a track is a depth observation with
   precision $\lambda_t$ equal to its squared parallax sensitivity over the tracker noise. It enters the
   existing lumped information blocks unchanged, as a second, sparse depth map with a per-pixel variance
   and *no* per-view scale, offset or distance bias. Each surfel then holds two posteriors of different
   provenance that are compared and, during judging, never mixed.
2. **No leave-out is needed against parallax.** The parallax posterior does not contain any depth map,
   so the cavity construction is unnecessary for this comparison, and a fault that every view repeats is
   as visible as any other. Leave-block-out remains for the depth-consensus evidence only.
3. **Harm and detectability are opposed.** For a fault of given harm to the fused surface, the
   noncentrality of the parallax test scales as $\varsigma/(1-\varsigma)^2$. Where parallax is strong
   the monocular prior hardly matters and its faults are easy to see; where the prior is the only
   constraint its faults are invisible to parallax. This is the precise form of "why keep the monocular
   prior", and it says where the head must carry the method.
4. **Anchored planes extend the witness over untextured surface.** Tracks on the trackable boundary of
   a planar region (shoreline, window frame, roof edge) define a plane in information form whose
   predictive variance inside the convex hull of the anchors is at most three times the variance at
   their centroid. This yields parallax-derived labels on water, glazing and flat roofs, with a stated
   planarity assumption, and it separates a monocular fault from a virtual image behind glass.
5. **The epipolar self-check does not catch everything it is credited with.** A static scene reflected
   in a flat pane is a rigid virtual scene: its tracks stay on their epipolar lines and triangulate
   consistently behind the glass. Objects moving along the flight direction also stay on the epipolar
   line. Both produce confident, wrong labels unless handled by the surfel map (Section 5.3).
6. **The indicator stays a pixel field**; the surfel map supplies association, the two posteriors,
   anchored planes, gated fusion and nuisance statistics. Track evidence fixes the per-view alignment
   without reference to the surface.

## 1. Two kinds of message at a surfel

### 1.1 The monocular message (lumped form)

Linearise the rendered depth of pixel $u$ of view $i$ at the fitted surface with compositing weights
held fixed, $\hat D_i(u;\theta) = \sum_k w_{uk}(\bar d_{uk} + \mathbf j_{uk}\theta_k)$. Depth observes three of
the ten coordinates of a surfel, its normal offset and two tilts; a message is a weighted plane fit of
the depth map over the surfel's footprint. With a pixel gate $w_i(u)$ and the residual
$e_i(u) = \hat d_i(u) - (a_i\tilde D_i(u) + b_i)$,

```math
\Lambda^{m}_{ik}[w_i] = c_{\mathrm{eff}} \sum_{u \in i} w_i(u)\, \frac{a_i^2\, w_{uk}}{\sigma_i^2(u)}\ \mathbf j_{uk}^\top \mathbf j_{uk},
\qquad
\eta^{m}_{ik}[w_i] = c_{\mathrm{eff}} \sum_{u \in i} w_i(u)\, \frac{a_i\, w_{uk}}{\sigma_i^2(u)}\ \mathbf j_{uk}^\top e_i(u) .
```

The weight inside $\Lambda$ is $w_{uk}$, not $w_{uk}^2$ (`view_information` with `lumped=True`). Because
the weights sum to one per pixel, Jensen's inequality gives
$\theta^\top H^{(i)}\theta \le \sum_k \theta_k^\top \Lambda^m_{ik}\theta_k$ for the true Gauss-Newton Hessian $H^{(i)}$:
the lumped block-diagonal matrix majorises the Hessian, and a step $\Lambda^{-1}\eta$ cannot overshoot.
Blocks built from $w_{uk}^2$ delete the cross-surfel terms and overshoot the common mode, in which all
overlapping surfels need the same shift, by $\sum_u w_{uk}/\sum_u w_{uk}^2$. Removing or gating a view is such
a common-mode correction, so gating must use the lumped form. A gate on a (view, surfel) pair is the
special case of a pixel gate that is constant over the footprint; the exact form above costs the same.

### 1.2 The track message

A track $t$ has a reference frame $i$, a reference pixel $u_t$, and observations $\mathbf x_{tj}$ in frames
$j \in \mathcal J_t$. A point at depth $d$ on the reference ray is $X(d) = \mathbf c_i + d\, R_i \mathbf r_{u_t}$ and
projects to $\pi_j(X(d))$. Its image motion per unit depth,

```math
\mathbf p_{tj}(d) = \frac{\partial \pi_j\bigl(X(d)\bigr)}{\partial d} \in \mathbb R^2 ,
```

points along the epipolar line of $u_t$ in frame $j$, and for a baseline component $b^\perp_{ij}$
perpendicular to the ray has magnitude close to $f\, b^\perp_{ij}/d^2$. The maximum-likelihood depth on
the ray is a one-dimensional Gauss-Newton problem:

```math
\lambda_t = \sum_{j \in \mathcal J_t} \mathbf p_{tj}^\top \Sigma_{tj}^{-1}\, \mathbf p_{tj},
\qquad
d^{\mathrm{tr}}_t = d_0 + \lambda_t^{-1} \sum_{j \in \mathcal J_t} \mathbf p_{tj}^\top \Sigma_{tj}^{-1} \bigl(\mathbf x_{tj} - \pi_j(X(d_0))\bigr),
```

iterated from $d_0$, the monocular or rendered depth. $\lambda_t$ is the precision of the parallax
depth. Two properties make this the right parametrisation. The noise is modelled where it is
homogeneous, in pixels, so the depth uncertainty follows from the geometry and needs no "well
conditioned" switch: a track with little parallax simply has small $\lambda_t$. And the estimate is a
depth along a ray of a known camera, which is exactly what the surfel machinery consumes.

The track message to surfel $k$ is therefore a depth message with unit scale and no offset,

```math
\Lambda^{\mathrm{tr}}_{k} = \sum_{t} w_{u_t k}\, \lambda_t\ \mathbf j_{u_t k}^\top \mathbf j_{u_t k},
\qquad
\eta^{\mathrm{tr}}_{k} = \sum_{t} w_{u_t k}\, \lambda_t\ \mathbf j_{u_t k}^\top \bigl(d^{\mathrm{tr}}_t - \bar D(u_t)\bigr),
```

computed by the same accumulation as the monocular message, with the per-pixel variance
$1/\lambda_t$ and an abstention mask. Each physical track sends one message: a scene point tracked from
several reference frames must not be counted as several independent observations.

**Precision in a strip.** With independent localisation noise $\sigma_x$ per frame,
$\sigma_{\mathrm{tr}} = \lambda_t^{-1/2} = d^2 \sigma_x / \bigl(f \sqrt{\sum_j (b^\perp_{ij})^2}\bigr)$. For frames at
uniform spacing $\Delta$ and a track of $L$ frames, $\sum_j b_j^2 \approx \Delta^2 L^3/3$, so the precision of
the depth grows as $L^3$: track length is baseline. This is the gain of video over a fixed set of
neighbours, and it holds only for independent noise. Point trackers drift, which makes the errors
along a track correlated and growing; with a random-walk component the gain is slower. $\Sigma_{tj}$
should therefore grow with the distance from the reference frame, with a rate estimated from the
across-epipolar residuals below, which see the same drift without any dependence on depth. As an
illustration with assumed values, $f = 3000$ px, a total baseline of half the flying height and
$\sigma_x = 1$ px give $\sigma_{\mathrm{tr}}/d \approx 0.07\%$; one frame step at a base-to-height ratio of
0.13 gives 0.26%. If monocular depth is accurate to one or a few percent (recalled as typical, not
measured here), parallax on a well-tracked point is the better measurement by an order of magnitude.
Section 4 takes up what follows from that.

### 1.3 The epipolar self-check, and what it does not check

With unit vector $\mathbf e^\perp_{tj}$ across the epipolar line,

```math
E_t = \sum_{j \in \mathcal J_t} \frac{\bigl(\mathbf e^{\perp\top}_{tj}(\mathbf x_{tj} - \pi_j(X(d^{\mathrm{tr}}_t)))\bigr)^2}{\sigma^2_{\perp,tj}} \ \sim\ \chi^2_{|\mathcal J_t|}
```

under a static scene, correct poses and a working tracker. It does not depend on depth, so it is a
check that costs no information. The along-line misfit after estimating the depth is a second
statistic with $|\mathcal J_t| - 1$ degrees of freedom. A track that fails either abstains. A residual across
the line that is coherent over a whole frame is a pose or timing error of that frame, not a property
of any track, and is handled at view level.

The check is blind to three things, and each yields a confident wrong depth:

- *Flat reflectors.* A static scene mirrored in a planar pane is a rigid virtual scene behind the
  pane. Its tracks obey the epipolar constraint and triangulate consistently to the virtual point.
  The check catches curved reflectors, moving highlights and waves, not flat glass.
- *Motion along the epipolar direction.* In a straight strip the epipolar lines of a nadir camera run
  along the flight direction. A vehicle driving along it at constant speed produces image motion
  proportional to the baseline, exactly as a change of height does.
- *Consistent tracker bias.* On weak texture a learned tracker fills in motion from context. The
  filled-in motion is smooth and can stay on the epipolar line while being wrong along it.

The first two are addressed with the surfel map in Section 5.3. The third is the reason tracker
confidence, not only the epipolar test, must enter $\Sigma_{tj}$, and it is not solved here.

## 2. Judging the monocular depth

### 2.1 Pixel score and minimum detectable bias

Let $d^{\mathrm{al}}_i(u)$ be the monocular depth mapped to metric depth by the inverse of the alignment and
bias model. At a track pixel,

```math
z^{\mathrm{par}}_i(u) = \frac{d^{\mathrm{al}}_i(u) - d^{\mathrm{tr}}_t}{\sqrt{\sigma_i^2(u)/a_i^2 + 1/\lambda_t + v^{\mathrm{pose}}_t}} ,
```

with $v^{\mathrm{pose}}_t$ the depth variance induced by pose uncertainty. It is, to first order, the score
test of the hypothesis that the monocular depth explains the observed parallax, and it is valid at any
$\lambda_t$: a poorly conditioned track gives a valid test with no power.

Power decides what can be called a label. In the terms of reliability theory, the smallest bias
detectable at level $\alpha$ with power $1 - \beta_0$ is

```math
\mathrm{MDB}_i(u) = \delta_0(\alpha, \beta_0)\, \sqrt{\sigma_i^2(u)/a_i^2 + 1/\lambda_t + v^{\mathrm{pose}}_t} ,
```

with $\delta_0$ the noncentrality that the chosen error rates require (about 4.1 for $\alpha = 0.1\%$ and
80% power; recalled). Labels are three-valued:

- *faulty*: $|z^{\mathrm{par}}|$ above the upper threshold;
- *clean*: $|z^{\mathrm{par}}|$ below the lower threshold **and** $\mathrm{MDB}_i(u)$ no larger than the fault size
  of interest. Without the second condition "no evidence of a fault" would be recorded as "no fault";
- *unknown*: everything else, including abstentions.

This replaces a threshold on triangulation angle by a statement in the units of the fault definition,
and it makes the asymmetry explicit: positive labels can come from weak tracks if the fault is large,
clean labels cannot.

### 2.2 The comparison at the surfel

Per surfel, the parallax-only posterior and the monocular messages are compared in the
three-dimensional sub-block. With $C^{\mathrm{tr}}_k = (\Lambda^{\mathrm{tr}}_0 + \Lambda^{\mathrm{tr}}_k)^{-1}$ and
$\delta^{\mathrm{tr}}_k = C^{\mathrm{tr}}_k \eta^{\mathrm{tr}}_k$ (a weak prior $\Lambda^{\mathrm{tr}}_0$ regularises the tilt when fewer than
three non-collinear tracks fall in the footprint),

```math
\mathbf r_{ik} = \eta^{m}_{ik} - \Lambda^{m}_{ik}\, \delta^{\mathrm{tr}}_k, \qquad
T_{ik} = \mathbf r_{ik}^\top \bigl(\Lambda^{m}_{ik} + \Lambda^{m}_{ik}\, C^{\mathrm{tr}}_k\, \Lambda^{m}_{ik}\bigr)^{+} \mathbf r_{ik}
\ \sim\ \chi^2_{\operatorname{rank}\Lambda^m_{ik}} .
```

Three differences from the depth-consensus test are structural.

- *No cavity.* $\delta^{\mathrm{tr}}_k$ contains no depth map. Nothing has to be left out, no Newton step is
  taken, and the validity gauge of the one-step cavity is irrelevant. The messages of all views at a
  surfel sum to zero at the fit, which makes any consensus statistic blind to a shared bias; the
  parallax posterior is outside that sum.
- *Every view of the surfel is judged*, not only the reference frame of a track. A triangulated point
  is a fact about the surface. Through the compositing weights it judges the monocular message of
  each view that renders the surfel, including views in which the tracker lost the point.
- *The footprint is judged, not the track pixel.* The surfel carries a sparse track grid to all pixels
  it renders, with predictive variance $\mathbf q_{uk} C^{\mathrm{tr}}_k \mathbf q_{uk}^\top$ that grows with the lever arm
  from the tracks.

The second and third points are what the surfel map adds to a per-pixel comparison of two depth
values: coverage. Whether the added coverage is large depends on track density relative to footprint
size and has to be counted.

The surfel positions and weights used for association come from the monocular fit. Where that fit is
far off, the association is to the wrong surfel. The pixel score of 2.1 does not use the surfel map
and is the primary evidence; the surfel form is used for association, extension and fusion.

### 2.3 Alignment and bias anchored on parallax

Triangulated depth is metric given the poses: it has no scale, offset or distance bias. The nuisance
parameters of each depth map can therefore be fitted against parallax instead of against the surface,

```math
(a_i, b_i) = \arg\min_{a,b} \sum_{t:\ \mathrm{ref}(t) = i} \rho_t\, \frac{\bigl(\hat d_i(u_t) - a\,(d^{\mathrm{tr}}_t + \beta\, d^{\mathrm{tr}\,2}_t) - b\bigr)^2}{\sigma_i^2(u_t) + a^2/\lambda_t} ,
```

with robust weights $\rho_t$, and $\beta$ from the summed normal equations with $(a_i,b_i)$ eliminated.
This removes two weaknesses of alignment against a cavity: the alignment no longer depends on the
surface it is used to judge, and a fault that captures the surface cannot capture the alignment.
Two cautions. The regressor is noisy, so ordinary least squares attenuates $a_i$ by about
$\operatorname{Var}(d)/(\operatorname{Var}(d) + \overline{1/\lambda})$; the effective-variance weighting above or a Deming fit corrects it.
And over flat terrain seen from nadir the depth range within a view is small, so $a_i$ and $b_i$ are
nearly collinear whatever the reference; only their combination at the mean depth is determined, which
is sufficient for the score but should be reported as such. The breakdown point of the robust fit
remains: a fault over most of the tracked pixels of a view captures the alignment.

## 3. What parallax identifies, and what it does not

| Situation | Depth consensus | Parallax at the surfel |
| --- | --- | --- |
| Fault in one block, textured surface | fires if the block is left out | fires |
| Fault repeated by every view, textured | silent | fires |
| Fault on untextured surface inside a trackable planar boundary | silent if repeated | fires through an anchored plane (Section 5), under planarity |
| Fault on untextured surface without such a boundary | silent if repeated | abstains; the head extrapolates |
| Correct monocular depth on flat glass with a visible reflection | silent | false alarm from the virtual image unless disambiguated (5.3) |
| Vehicle moving along the strip | fires or not | wrong depth with a passing epipolar check |
| Pose error of a frame | view-wide residual | view-wide across-line residual; detectable and separable |

Parallax identifies a depth error wherever $\lambda$ is large enough, independently of how many views
repeat it. It identifies nothing where tracking abstains, and it is wrong, not silent, in the two
cases of Section 1.3. Shared causes between parallax and the surface remain: both use the same poses,
so a pose error moves both; its signature is view-wide and across the epipolar line, which neither a
regional fault nor a track failure produces.

## 4. Why keep the monocular prior: the harm-detectability relation

If parallax on a tracked point is an order of magnitude more precise than monocular depth, the prior is
redundant there. The surfel information makes the question quantitative. Define the parallax share of
a surfel,

```math
\varsigma_k = \frac{\mathbf g_k^\top \Lambda^{\mathrm{tr}}_k\, \mathbf g_k}{\mathbf g_k^\top \bigl(\Lambda^{\mathrm{tr}}_k + \Lambda^{m}_k\bigr)\, \mathbf g_k} \in [0, 1],
```

with $\mathbf g_k$ the row of $G_k$ for the normal offset. Take the offset alone, with monocular precision
$\lambda^m$, parallax precision $\lambda^{\mathrm{tr}}$, and a bias $\Delta$ in the monocular messages (a fault
that all views repeat). If both were fused, the surface would move by

```math
h = \Delta\, \frac{\lambda^m}{\lambda^m + \lambda^{\mathrm{tr}}} = \Delta\,(1 - \varsigma) ,
```

the *harm* of the fault. The noncentrality of the comparison of the two posteriors is

```math
\delta^2 = \frac{\Delta^2}{1/\lambda^m + 1/\lambda^{\mathrm{tr}}} = \Delta^2\, \lambda^m\, \varsigma
\qquad\Longrightarrow\qquad
\delta^2 = h^2\, \lambda^m\, \frac{\varsigma}{(1 - \varsigma)^2} .
```

For a fault of given harm, detectability by parallax rises steeply with the parallax share and
vanishes as the share goes to zero. Three regimes follow.

- $\varsigma \to 1$ (textured, long tracks). Parallax decides the surface. The monocular prior adds
  nothing, its faults are easy to detect and, if tracks are fused, harmless. Detection here has one
  use: it is a free label.
- Intermediate $\varsigma$ (weak texture, short tracks, sparse tracks under a large footprint). Both
  sources matter, faults are harmful and detectable. This is where a directly witnessed gate changes
  the reconstruction.
- $\varsigma \to 0$ (water, glass, texture-free surface, occlusion boundaries). The prior is the only
  constraint, its faults do the most harm, and parallax says nothing.

The honest answer to the objection is therefore not that triangulation is too noisy to replace the
prior. Where it exists it is better. The prior is kept for the surfels with small $\varsigma$, and
the method is worth more than "check monocular depth with multi-view stereo, then use the stereo" only
if it does something for those surfels. Two mechanisms can: the anchored planes of Section 5, which
raise $\varsigma$ for untextured planar regions, and the frozen-feature head of Section 7, which
carries labels from high to low $\varsigma$. Two consequences for the design:

1. *Evaluation must be weighted by harm.* A detection rate over all faulty pixels is dominated by the
   regime where detection is easy and useless. The relevant figure weights each faulty pixel by
   $1 - \varsigma$ of its surfel, or reports the three regimes separately.
2. *Whether tracks enter the fit is a separate decision.* Keeping parallax as a judge only is clean
   but discards the better measurement in the first two regimes. Fusing it is optimal there and makes
   the judge part of the judged surface. Sample splitting resolves this: tracks are divided into a
   fitting set and a judging set (by spatial tile, so that neighbouring tracks with shared error stay
   together), and only the judging set produces evidence. The fused surface, track-only surface,
   monocular-only surface and gated surface are then four baselines of one experiment, and "the prior
   adds nothing" has a direct test: the track-only fit matches the gated fit on the harm-weighted
   metric.

## 5. Anchored planes: parallax evidence on untextured surface

### 5.1 Construction

Water bodies, glazing and flat roofs are untextured in their interior and bounded by structure that
tracks well: shoreline, frames and mullions, parapets and roof edges. If the interior is planar, the
boundary tracks determine it. Let $S$ be a connected set of surfels proposed as one planar region
(Section 5.4), with plane coordinates $\mathbf y \in \mathbb R^2$ and offset along the plane normal. The anchors
are the tracks on $S$ and on a rim around it that pass the checks of Section 1.3. In information form
on the plane parameters $\boldsymbol\vartheta_S = (\nu, \tau_1, \tau_2)$,

```math
\Lambda^{\mathrm{tr}}_S = \sum_{t \in \mathrm{anchors}} \lambda^{n}_t\ \mathbf h_t^\top \mathbf h_t, \qquad
\eta^{\mathrm{tr}}_S = \sum_{t} \lambda^{n}_t\ \mathbf h_t^\top \rho_t, \qquad
\mathbf h_t = [\,1,\ -y_{t,1},\ -y_{t,2}\,],
```

with $\rho_t$ the signed distance of the triangulated point from the current plane and $\lambda^n_t$ its
precision along the normal. This is the track message of Section 1.2 with the surfels of $S$ tied to
one plane. The predictive variance of the plane at an interior point is

```math
v_S(\mathbf y) = [\,1,\ -\mathbf y^\top] \bigl(\Lambda^{\mathrm{tr}}_S\bigr)^{-1} [\,1,\ -\mathbf y^\top]^\top + \varsigma^2_{\mathrm{pl}} ,
```

where $\varsigma^2_{\mathrm{pl}}$ is an allowance for departure from planarity. For $n$ anchors of equal
precision $\lambda$ spread evenly on a circle of radius $R$ around the centroid, the first term is
$\frac{1}{n\lambda}\bigl(1 + 2\lVert\mathbf y\rVert^2/R^2\bigr)$: inside the ring it never exceeds three times the
variance at the centroid. Interpolation inside the convex hull of the anchors is cheap; extrapolation
beyond it grows quadratically, and the formula says by how much. The monocular depth of every pixel
rendered by $S$ is then judged as in Section 2.1 with $1/\lambda_t$ replaced by $v_S$ mapped to depth along
the ray.

For water the plane is horizontal, and the vertical is known from georeferenced poses. One parameter
remains, the level, with variance $1/\sum_t \lambda^n_t$ independent of position; a few shoreline tracks judge
the whole water body.

### 5.2 What is assumed and what can be checked

The construction turns the regime $\varsigma \to 0$ into an intermediate one for regions that are
planar and bounded by trackable structure. It assumes planarity of the interior, and that assumption
is only partly testable:

- the plane-fit residual of the anchors, $\chi^2$ with $n - 3$ degrees of freedom, tests that the
  *boundary* is planar;
- the monocular depth itself is evidence about shape. A fault of offset or scale leaves the
  monocular surface planar; if the monocular surfels of $S$ are strongly non-planar, either the region
  is not a plane or the fault is one of shape, and the region abstains. That monocular depth
  preserves planarity where its offset is wrong is a conjecture;
- interior tracks, where any exist and pass the checks, test the interior directly.

Pitched glass roofs, domes, waves and vegetation mats violate the assumption without a boundary
signature. The allowance $\varsigma_{\mathrm{pl}}$ bounds the claim: an anchored plane can call a monocular
depth faulty only if it departs from the plane by several $\sqrt{v_S}$, so shallow curvature produces
neither label.

### 5.3 Virtual images and along-track movers

The anchored plane also resolves the first two blind cases of the epipolar check.

- *Virtual image.* Interior tracks on a pane that triangulate *behind* the anchored plane by many
  $\sqrt{v_S}$ are reflections or a view through the glass. In the terms of ray-based visibility, the
  anchored plane is a surface that these rays pierce. Such tracks are removed as judges of the
  monocular depth, and their presence is recorded as a material attribute of the region
  (transparent or reflective), which is a useful input to the head and costs nothing. Without an
  anchored plane the conflict between a monocular depth on the pane and a triangulated depth behind it
  cannot be attributed; those pixels are labelled unknown, not faulty.
- *Along-track movers.* A triangulated point that lies below or above a well-anchored ground plane
  inside a road segment, while its neighbours lie on it, is a mover. This catches isolated vehicles;
  dense traffic on an untracked road surface is not caught.

The same three-way ray test (confirmed, pierced, hidden) applies between triangulated points and
surfels in general: a triangulated point seen *through* a surfel that only monocular messages support
is the parallax form of a free-space violation and identifies surfels that a fault created for itself.
Hidden, a point in front of the surfel, stays neutral.

### 5.4 Where the regions come from

Candidates for $S$ are connected sets of surfels with homogeneous lifted features (Section 7.1), low
track density and co-planar monocular surfels, or the masks of a frozen segmentation model lifted to
surfels with the pooling operator. The choice affects coverage, not validity, since each region is
tested by 5.2 before it produces labels. Over-segmentation costs anchors per region;
under-segmentation fails the planarity tests.

Plane priors from triangulated sparse points are established in multi-view stereo for textureless
regions (recalled; planar-prior patch-match methods and piecewise-planar reconstruction). What differs
here is the use: the plane is a judge with a stated variance and a source of labels, not a depth
estimate that replaces the prior, and it is only trusted where its own tests pass.

## 6. Moving evidence and labels between pixels and surfels

For view $i$ let $W_i$ hold the compositing weights and $N_i = \operatorname{diag}(n_{ik})$.

```math
\text{render:}\quad (\mathsf R_i g)(u) = \sum_k w_{uk}\, g_k, \qquad
\text{pool:}\quad (\mathsf P_i s)_k = \frac{1}{n_{ik}} \sum_u w_{uk}\, s(u),
\qquad
\langle \mathsf R_i g,\ s\rangle_{U} = \langle g,\ \mathsf P_i s\rangle_{N_i} .
```

The two are adjoint under the measure $n_{ik}$, both preserve constants, and pooling is computable with
a stock rasteriser as the vector-Jacobian product of a rendered dummy scalar. The round trip
$\mathsf R_i\mathsf P_i = W_i N_i^{-1} W_i^\top$ is a smoothing kernel of footprint width, not a projection; variation
within a footprint, including the position of a fault boundary, is lost, and footprints are expected to
be largest on flat, untextured surface. Three uses in this formulation:

- *Sparse scores.* Track scores exist at track pixels only. Pooling them uses the weights restricted to
  those pixels, and the pooled value has the track count of the pair as its effective sample size.
- *Labels to all views.* A surfel-level verdict (Section 2.2 or an anchored plane) is rendered into
  every view that sees the surfel, which gives labels at pixels without tracks. A label carried this way
  judges the *message of that view*: the rendered quantity is the reference depth, and each view's own
  $z$ is formed against it. Rendering a single fault flag to all views would assert that all views
  repeat the fault, which is the thing to be found out.
- *Log-likelihood ratios* add, deflated for spatial correlation; probability-like scores average.

Labels inherited through one surfel are not independent samples. The unit for counting labels, for
cross-fitting and for calibration is the ground tile or the planar region, not the pixel or the pair.

## 7. Frozen features on surfels, and the transfer to untextured surface

### 7.1 Lifting

Frozen patch features $\mathbf f_i(p)$ of frame $i$, reduced by a fixed unsupervised linear map to some tens
of channels, lift with the same operator. With bilinear weights $b_{up}$ from patches to pixels,

```math
\mathbf f_{ik} = \frac{1}{n_{ik}} \sum_p \Bigl(\sum_{u \in i} w_{uk}\, b_{up}\Bigr) \mathbf f_i(p),
\qquad
\bar{\mathbf f}_k = \frac{\sum_i n_{ik}\, \mathbf f_{ik}}{\sum_i n_{ik}} .
```

$\mathbf f_{ik}$ is what view $i$ saw at the surfel; $\bar{\mathbf f}_k$ is the lumped least-squares solution of
feature-field distillation, one majorisation step by the same Jensen argument as in Section 1.1. Sums
are stored per block of frames so that any leave-block-out version is a subtraction. The lifted field
has the resolution of the backbone patch, not of the surfel.

### 7.2 Which frozen models, and a caution on video features

| Component | Frozen model | Why |
| --- | --- | --- |
| Tracks | a long-range point tracker or dense flow model | correspondences with visibility; Section 8 |
| View-specific appearance $\mathbf f_{ik}$ | the encoder of the depth model that produced the map, where accessible, else a self-supervised image encoder | closest observable to the failure state of that model in that view; a conjecture |
| Surface context $\bar{\mathbf f}_k$ | a self-supervised image encoder | material of the surface, averaged over views |
| Temporal or multi-view features | a video encoder or a feed-forward multi-view geometry model | see below |
| Region proposals | a frozen segmentation model, or clustering of $\bar{\mathbf f}_k$ | Section 5.4 |

A video or multi-view backbone sees parallax inside its features. That is useful and it changes the
role of the head. On appearance features the head is a prior, and its errors are plausibly independent
of the track evidence. On parallax-aware features the head is partly a second reading of the same
motion, so prior and evidence are no longer separable, and the gain over the witness has to be shown
on pixels where the witness abstains. There a parallax-aware backbone faces the same absence of
texture as the tracker; whether its features still carry more than appearance on such surface is an
open empirical question and the specific reason to try it. Temporal features also respond to
view-dependent appearance (moving highlights, changing reflections) that a per-image encoder cannot
see, and that is a direct signature of the materials in question.

### 7.3 The head

Per pixel or pair: $[\,\mathbf f_{ik},\ \bar{\mathbf f}_k,\ \text{material attributes from 5.3},\ \text{geometry scalars}\,]$ into a linear
layer or a two-layer perceptron of width about 128, some $10^4$ to $10^5$ parameters, giving the prior
logit. Geometry scalars are range, incidence, $n_{ik}$, $\varsigma_k$ and track density; the head may
know how much witness a pixel had. It is trained on the three-valued labels of Section 2.1 (faulty,
clean, with unknown ignored), carried to pairs and pixels as in Section 6. The posterior combines the
head's prior with the pixel's own evidence by Bayes' rule where evidence exists; where it does not, the
indicator is the prior.

Two structural rules. *The head does not see what its target was made from*: its inputs are features
and geometry, never $z^{\mathrm{par}}$ or the depth residual at that location. If evidence context from the
surroundings is added as input, it is taken outside a radius matched to the correlation length of the
tracker and depth noise. *Cross-fitting is by ground area*: surfels are fixed in the scene and seen
from many frames, so a head could memorise a location from one frame and replay it for another. The
head for a ground tile is fitted on other tiles, with a margin.

For the depth-consensus evidence a stronger form of the first rule holds. Because the messages at a
surfel sum to zero at the fit, a consensus feature of another view at the same surfel is a linear
function of the message being judged; consensus context for the gate of $(i,k)$ must be computed with
the block of view $i$ removed. In information form an arbitrary leave-out set is a subtraction and a
$3\times 3$ solve per surfel.

### 7.4 The transfer in the terms of this note

The head is trained where labels exist and applied where they do not, and the variable that separates
the two, trackability, is correlated with the fault mechanism. Stated as a missing-data problem: let
$e(\boldsymbol\phi)$ be the probability that a pixel with features $\boldsymbol\phi$ receives a label.

- If trackability is determined by what the features encode, labels are missing at random given
  $\boldsymbol\phi$. Then $\Pr(f \mid \boldsymbol\phi)$ is the same on labelled and unlabelled pixels, the shift is
  a covariate shift, and weighting by $1/e(\boldsymbol\phi)$ or simply a well-specified head handles it,
  **where $e(\boldsymbol\phi) > 0$**.
- Where $e(\boldsymbol\phi) = 0$, no reweighting helps. The head's output is an extrapolation whose error
  is not estimable from the video. This is the situation for the interior of water and glass under the
  pixel-level witness alone.

Three things move mass from the second case to the first, in order of how much they assume:

1. *Long tracks.* Since precision grows with track length, weakly textured surface that defeats a
   two-frame match can still receive labels from a long track. This widens the support of
   $e(\boldsymbol\phi)$ without assumptions beyond those of Section 1.2.
2. *Anchored planes.* They give labels inside untextured regions, under planarity. For these pixels
   $e(\boldsymbol\phi) > 0$, and the transfer from textured to untextured surface becomes a measurable
   quantity on the video itself: train the head on track-pixel labels only, test it on anchored-plane
   labels. No ground truth is needed for this test, which is its value.
3. *A synthetic scene with exact depth.* It supplies labels everywhere, including where both
   mechanisms abstain, and so measures the extrapolation error directly. This is the role left for
   labels: not to train the detector, but to validate and calibrate it where the video is silent.

Selection remains in each: long tracks favour unoccluded, slowly changing surface; anchored planes
favour planar, bounded regions; a synthetic scene has synthetic materials. A fault mechanism confined
to unanchored, unbounded, untextured surface (open water to the image border, a fully glazed facade
without visible frames) is reached by none of the first two.

A second, different risk is a shift in the *kind* of fault. On textured surface monocular depth errs
mostly at depth edges, thin structures and scale; on glass and water it errs by seeing through or by
hallucinating a surface. If the features that predict the first kind do not predict the second, a
head trained on witnessed labels learns the wrong detector, however many labels it has. Anchored-plane
labels are the only video-internal protection, since they are labels of the second kind.

## 8. Frozen trackers and cost

What was checked against the published sources, and what is recalled:

- *CoTracker3* (arXiv 2410.11831). Checked: it exists with public code and weights, has an offline
  mode over a whole clip and an online sliding-window mode for long streams, returns per-point
  visibility, and most of the code is under a non-commercial licence. Recalled, not checked: that it
  also returns a per-point confidence, and its accuracy on aerial imagery.
- *AllTracker* (arXiv 2506.07310, listed at ICCV 2025). Checked: it produces dense long-range
  correspondence from a query frame to hundreds of frames as flow fields, has about 16 million
  parameters, and its abstract reports 768 by 1024 pixels on a 40 GB GPU. That resolution will not fit
  in 8 GB as stated; operation at reduced resolution or in tiles is assumed, not verified. Whether it
  outputs visibility or confidence was not checked.
- *VGGT* (CVPR 2025). Checked: the paper exists. Recalled: it is a feed-forward multi-view model with
  depth, camera and point-track outputs, usable as a frozen multi-view backbone; its memory use per
  frame was not checked.
- Recalled only: TAPIR-family trackers, RAFT-family flow models, DINO-family image encoders, video
  encoders of the masked or joint-embedding type, and promptable segmentation models as candidates for
  the corresponding rows of the table in 7.2.
- Closest related work found: "In Depth We Trust: Reliable Monocular Depth Supervision for Gaussian
  Splatting" (arXiv 2604.05715), which by its abstract applies monocular depth regularisation
  selectively by isolating ill-posed geometry. Only the abstract was read; how it decides where the
  prior applies, and whether it uses tracks or triangulation, was not checked and must be before any
  claim of novelty.

Cost. The tracker is run once per clip and cached. A query grid of stride 8 on a frame of about
$1000 \times 750$ pixels is about $1.2 \times 10^4$ tracks per reference frame; with a reference frame every
few frames and each ground point in view for a bounded number of frames, tracks number some
$10^5$ to $10^6$ per strip. Triangulation is a scalar iteration per track; track messages and anchored
planes are accumulations of $3\times 3$ blocks; the comparison of Section 2.2 is one small solve per pair.
All of this is negligible next to the tracker and the surfel fit. Lifted features at 32 channels in
half precision are 64 bytes per pair. These are estimates from sizes, not measurements. No tracker
has been run on the project's data.

## 9. Use in reconstruction

The indicator is a pixel field $\zeta_i(u)$ and enters through $w_i(u) = 1 - \zeta_i(u)$.

- **Gated fusion in closed form.** $\Lambda_k = \Lambda_0 + \sum_i \Lambda^m_{ik}[w_i]\ (+\ \Lambda^{\mathrm{tr}}_k$ of the
  fitting tracks, if tracks are fused$)$, $\delta_k = \Lambda_k^{-1}\eta_k$ on the nine geometric coordinates.
  The sensitivity to a pair gate is $\partial\delta_k/\partial w_{ik} = \Lambda_k^{-1}(\eta^m_{ik} - \Lambda^m_{ik}\delta_k)$. Valid
  where the gated-out share of a surfel's information is small; it holds compositing weights, order and
  the set of surfels fixed and cannot predict what becomes visible when a surfel loses its support.
- **Refit.** A warm-started fit with the weight map frozen for the round. A weight that moves with the
  surfels during optimisation is unsound: the optimiser can cover high-residual pixels with gated
  surfels. Surfels that only gated messages support, or that triangulated points pierce, are pruned
  first.
- **Nuisance parameters.** Alignment and bias come from Section 2.3, with flagged pixels down-weighted.
  Views with too few tracks fall back to alignment against the gated surface, whose sufficient
  statistics decompose over pairs ($2\times 2$ moments of $[\tilde D, 1]$ per pair), so a change of gates
  re-solves them without revisiting pixels.
- **Gated regions.** Where the prior is gated and nothing else constrains the surface, the fit reports
  the region as unconstrained. An anchored plane that passed its tests is the one case with a
  replacement at hand: its message $(\Lambda^{\mathrm{tr}}_S, \eta^{\mathrm{tr}}_S)$ with variance $v_S$ can stand in
  for the gated prior. That turns detection into repair for planar regions and is an extension, to be
  reported separately from detection.
- **One pass** (fit, tracks, evidence, head, refit) is the default. Parallax evidence does not depend
  on the surface, so it does not need rounds; only the association and the consensus evidence change
  when the surface does.
- **Soft against hard.** Soft weights are stable and keep the clean part of a view; they down-weight
  inliers by their own $\zeta$ and invalidate $\Lambda_k^{-1}$ as a covariance. Confident gates are snapped
  to zero or one at the end and final variances are computed on the retained set.

## 10. What is new here, what would kill it, and what is unresolved

**What is sharp in this part.** Two ideas, each with precedents that limit the claim.

1. *The harm-detectability relation and the parallax share* (Section 4). It turns the objection to
   keeping the monocular prior into a per-surfel quantity, predicts where a parallax judge can change
   the reconstruction, and prescribes a harm-weighted evaluation under which "run multi-view stereo
   and compare" gets no credit for the easy regime. The algebra is two lines of reliability theory;
   the contribution is the framing and the metric, not the mathematics.
2. *Anchored planes as a judge with variance* (Section 5). They put parallax-derived labels on water,
   glazing and flat roofs, make the transfer from textured to untextured surface measurable without
   ground truth, and separate monocular faults from virtual images. Plane priors from sparse
   triangulated points are known in multi-view stereo; using them as a tested, variance-carrying
   witness and label source for a learned indicator is, as far as recalled, not.

The track-as-depth-message formulation (Section 1.2) and the alignment anchored on parallax (2.3) are
clean and useful but are engineering consequences of standard triangulation. If neither of the two
ideas above survives its test, this part of the work is a multi-view check on monocular depth with a
learned extrapolation, and should be described as that.

**What would kill it, cheapest test first.** All use the exact depth of a synthetic scene for
evaluation only.

1. *Oracle transfer, no tracker needed.* Train the head on exact fault labels restricted to textured
   pixels and test it on untextured pixels. This is the ceiling of what witnessed labels can transfer.
   If it is near chance, the kind of fault differs between the regimes and no witness on textured
   surface can train the detector. Cost: features and a linear probe.
2. *Where the harm is.* From poses and exact depth, compute the best-case $\lambda$ per pixel under an
   assumed tracker noise, then the share of fault area and of harm-weighted fault area in each regime
   of $\varsigma$. If nearly all harm sits at $\varsigma \approx 0$, the pixel-level witness judges
   only what does not matter. Cost: geometry only; then a tracker on a few clips to replace the
   assumed noise by the measured trackable fraction.
3. *Anchor coverage.* The fraction of untextured fault area that lies in a region passing the tests of
   Section 5.2 with at least three well-spread anchors, and the error of the anchored plane against
   exact depth. If coverage is a few percent, the second idea is a curiosity.
4. *Virtual images.* The rate at which interior tracks on glass pass the epipolar check and
   triangulate behind the pane, and how often the monocular depth there is on the pane. If common and
   unanchored, the witness manufactures false fault labels on exactly the target material.
5. *Redundancy of the prior.* Track-only, monocular-only, fused and gated fits on the harm-weighted
   metric. If the track-only fit matches the gated fit, the prior, and with it the indicator, is
   unnecessary for this data.

**Unresolved.**

- *The zero-overlap region.* Untextured, unbounded or non-planar surface receives no parallax label of
  either kind. The head's error there cannot be estimated from the video, only from a synthetic scene
  whose materials may not match.
- *Confident tracker errors on weak texture.* A learned tracker interpolates motion where it cannot
  see it, and the interpolated motion can satisfy the epipolar constraint. Tracker confidence is the
  only defence proposed, and its calibration on aerial data is unknown.
- *Correlated track noise.* The growth of precision with track length assumes independent noise; the
  drift model that replaces it is not specified beyond a growing $\Sigma_{tj}$.
- *Planarity inside a region* is tested only at its boundary and by the monocular shape.
- *Association through a wrong surface.* Carrying verdicts through compositing weights assumes the
  monocular fit is roughly in place; at surfels a fault created, the association itself is in doubt.
- *Shared poses.* Parallax and the surface use the same poses; only the view-wide, across-line
  signature of a pose error separates it from a fault.
- *Not verified:* every numerical illustration in this note; the behaviour and outputs of the frozen
  models beyond what Section 8 lists as checked; the literature attributions marked as recalled.
