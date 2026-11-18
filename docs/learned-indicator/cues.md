# Fault Cues: Definitions, Nulls, Abstention and Predicted Dependence

This note defines the detectors ("cues") that a label-free combination model would take as input. Each
cue is specified as a per-pixel score with a stated nuisance alignment, a null distribution that is known
or learnable, an abstention rule, and a prediction of how it behaves on each fault class. The last part is
the predicted dependence structure between cues, which is what decides whether a
conditional-independence label model (Dawid-Skene type) is applicable. Notation follows the main
document: view $j$, pixel $u$, depth map $\hat d_j$, per-view scale and offset $(a_j, b_j)$,
shared distance bias $\beta$, cavity depth $\hat D_{-j}$, block $\mathcal B_j$, testable set $\Omega_j$,
latent fault $f_j(u)$, surfel information $\Lambda_k$, $\Lambda_k^{(i)}$.

Statements about the internals of the depth models and of third-party methods are recalled from memory
and marked "(recalled)". Nothing in this note has been measured; every entry of the behaviour table is a
prediction to be tested.

## 1. Common Form of a Cue

### 1.1 Target

The label is the fault of the depth map that is fused: $f_j(u) = 1$ when the aligned error of the
*primary* model $m_0$ at $(j, u)$ exceeds $3\sigma$ inside a connected region of at least 2% of the view.
A whole-view affine error is absorbed by $(a_j, b_j)$ and is not a fault. Two consequences hold for every
cue below.

- Every cue compares a depth with something else after removing a nuisance transformation, so every cue
  is blind to an error that lies in its nuisance group. This is by construction and consistent with the
  definition of a fault.
- Faults are regional. Edge mislocalisation by a few pixels and thin structures are not faults under the
  definition, but all cues respond to them. They are the main source of *shared false alarms*.

### 1.2 Output of a cue

A cue $c$ delivers three fields per view:

```math
q^{c}_j(u) \in \mathbb R, \qquad A^{c}_j(u) \in \{0, 1\}, \qquad \mathbf x^{c}_j(u),
```

a standardised score, an abstention flag ($A = 1$: the cue carries no information at this pixel), and the
covariates on which its null was conditioned. The score is standardised so that, on non-faulty pixels
with $A = 0$, it has a known reference distribution $F^c_0$ (standard normal for signed scores, a fitted
law for unsigned ones). The discrete vote used by a label model is

```math
\lambda^{c}_j(u) = \begin{cases} \varnothing & A^{c}_j(u) = 1,\\ \mathbb 1\bigl[\, 1 - F^{c}_0\bigl(\lvert q^{c}_j(u)\rvert\bigr) < \alpha_c \,\bigr] & \text{otherwise.} \end{cases}
```

### 1.3 Nuisance alignment

Let $\mathcal G$ be a group of monotone maps acting on depth values. A comparison between two depth
fields $X$ and $Y$ on a common pixel domain $\mathcal D$ is the residual after a robust fit,

```math
g^\ast = \arg\min_{g \in \mathcal G} \sum_{u \in \mathcal D} \rho\bigl( \tau(X(u)) - \tau(g \cdot Y(u)) \bigr), \qquad \Delta(u) = \tau(X(u)) - \tau(g^\ast \cdot Y(u)),
```

with $\tau$ a parametrisation in which $\mathcal G$ acts linearly and $\rho$ a redescending loss (or a
consensus fit). Three groups occur.

| Group | Action | Linear in | Applies to |
| --- | --- | --- | --- |
| $\mathcal G_{\mathrm s}$ | $d \mapsto a d$ | $\tau = \log d$ (one offset) | pairs of metric models (Depth Pro, Metric3D v2, UniDepthV2) |
| $\mathcal G_{\mathrm{ai}}$ | $1/d \mapsto \alpha/d + \gamma$ | $\tau = 1/d$ | any pair that involves Depth Anything V2, whose output is affine-invariant inverse depth |
| $\mathcal G_{\mathrm{ad}}$ | $d \mapsto a(d + \beta d^2) + b$ | $\tau = d$ | depth map against a rendered surface (the main document's measurement model) |

The groups are not interchangeable. An offset in inverse depth is not an offset in depth:

```math
\frac{1}{\alpha/d + \gamma} = \frac{d}{\alpha} - \frac{\gamma}{\alpha^2} d^2 + O(\gamma^2 d^3),
```

so a Depth Anything V2 map aligned with $\mathcal G_{\mathrm{ai}}$ and then compared under
$\mathcal G_{\mathrm{ad}}$ carries a *per-view* quadratic term, which the *shared* $\beta$ cannot absorb.
With a relief of tens of metres at a flying height of a few hundred metres the term is small but
systematic over the whole view; the consistent choice is to align that model in inverse depth wherever
it is involved.

Requirements for the fit to be well-posed: (i) breakdown point above the largest fault fraction of
interest (a least-median or consensus fit tolerates just under 50%; the affine-inverse consensus fit in
the code is documented to hold up to 40% corrupted reference pixels); (ii) enough depth range in
$\mathcal D$ to separate scale from offset, since on a flat nadir scene $\alpha$ and $\gamma$ are nearly
collinear and the fit is ill-conditioned, in which case the group must be reduced to scale only; (iii) a
report of the inlier count, used for abstention. When the fault covers more than the breakdown fraction,
the alignment locks onto the fault and the cue is inverted on the whole view. This failure is common to
all cues that share an alignment.

### 1.4 Learning the null

Each raw residual $\Delta$ is divided by a scale $s^c(\mathbf x^c)$ estimated from the bulk of the data
with a robust estimator (median absolute deviation, or the lower half of the distribution for unsigned
scores), stratified by covariates that describe geometry only: range, local depth slope, distance to
the nearest depth edge, number of supporting views. This is the main document's scale head applied per cue.
It rests on the same assumption: within a stratum, faults are a minority. A fault class tied to a
stratum is absorbed into the null.

### 1.5 Spatial support

Faults are regional, noise is not. Each signed score is pooled over a window of the fault scale before
voting, $\bar q(u) = \sum_{u' \in W(u)} q(u') / \sqrt{n_{\mathrm{eff}}}$, and unsigned scores by a local
median. All cues are pooled at the same scale so that their votes refer to the same support. Votes at
neighbouring pixels are not independent samples; the unit of evidence for estimating detector accuracies
is a pooled cell, not a pixel.

### 1.6 Transfer to (view, surfel) pairs

A pixel score becomes a score on the pair $(i, k)$ by splatting with the compositing weights,
$q^c_{ik} = \sum_u w_{uk}\, q^c_i(u) / \sum_u w_{uk}$, with abstention when the non-abstaining weight is
below a threshold. Cues c2 and c3 and the plane-sweep form of c4 are defined on rays and need no surfel
map; they therefore also exist for surfels that a fault spawned for itself.

## 2. Cue c1: Leave-Block-Out Depth Residual

**Definition.** The standardised cavity residual of the main document,

```math
q^{1}_j(u) = z_j(u) = \frac{\hat d_j(u) - a_j \tilde D_{-j}(u) - b_j}{\sqrt{v_j(u)}}, \qquad \tilde D_{-j} = \hat D_{-j} + \beta_{-j} \hat D_{-j}^2 ,
```

with nuisance group $\mathcal G_{\mathrm{ad}}$ fitted robustly on $\Omega_j$. Null: zero mean, scale
$s_j(u)$ learned from geometry; the nominal value is one.

**Ray form (c1r).** The cavity residual exists only where the cavity surface covers the pixel. A fault
can create surfels that only faulty views support; they are dropped from the cavity and the pixel
becomes untestable, or the cavity renders the surface behind them. The conflict is then along the ray.
With $X_j(u)$ the back-projection of pixel $u$ at its aligned depth, $u_i = \pi_i(X_j(u))$ its
projection into a view $i \notin \mathcal B_j$, and $\delta_i$ the depth of $X_j(u)$ in view $i$,

```math
e_{j \to i}(u) = \frac{d^{\mathrm{al}}_i(u_i) - \delta_i\bigl(X_j(u)\bigr)}{\sqrt{\sigma_i^2 + \sigma_j^2}}, \qquad
q^{1r}_j(u) = \operatorname{med}_{i \notin \mathcal B_j}\ \max\bigl(e_{j \to i}(u),\ 0\bigr).
```

A positive value means view $i$ sees through the point (free-space violation); a negative value means the
point is occluded in $i$, which is no evidence. This is the visibility test of depth-map fusion. It
needs aligned depth $d^{\mathrm{al}}_i$ for all views, so it depends on the same per-view alignments as
c1.

**Abstains** when $u \notin \Omega_j$ (accumulated cavity opacity below threshold), when the removed
information share $\Phi_{-j}(u)$ is above the support threshold, when the alignment has too few inliers,
and for c1r when fewer than two views outside the block see the point unoccluded.

**Responds to** disagreement between the depth of view $j$ and the depth of views outside its block,
whatever the cause: depth-model error in $j$, depth-model error in the others, pose error of $j$,
unrepresented structure in the surfel map, linearisation error of the one-step cavity.

**Dependence between c1 and c1r.** Same information (depth maps of other views, same poses, same
alignments). They are two channels of one detector, not two detectors, and must enter a label model as
one cue (for example the maximum of the two tail probabilities).

**Cost and code.** The machinery exists: per-view information blocks
(`eval/gaussian_uncertainty.view_information`, `SurfelBackend.view_information`), subtraction and Newton
step (`leave_view_out_newton`, `leave_block_out_newton`, `block_for_view`), support rules, cavity render
with variance (`render_cavity`), distance-bias downdate, all driven by `gs/gate.ViewGateTest`, which
accepts a block map. Missing: (a) `ViewGateTest._stats` forms the residual and its weight per pixel but
returns only the energy and the region-scan values, so the map $z_j(u)$ and the mask must be exported;
(b) the nominal noise is a single scalar `sigma`, not a field $\sigma_j(u)$; (c) the robust (reweighted)
alignment: the gate projects the affine nuisance out by weighted least squares; (d) the rendered removed
share $\Phi_{-j}$; (e) the ray form, for which `baselines/merrell_visibility.stability_counts` and
`baselines/consistency_voting.support_counts` already contain the projection and free-space logic but
return counts, not standardised per-pixel values. Cost per tested view: one information pass per view of
the block (dense, $O(K \cdot \text{chunk})$ memory in the reference implementation) and one cavity
render; the ray form is one projection per neighbour.

## 3. Cue c2: Cross-Model Disagreement

**Setting.** $M$ depth models run on the *same* image, so the pixel grids coincide (all wrappers return
depth at input resolution) and no warp is needed. The nuisance is one element of $\mathcal G$ per pair
and view.

**Pairwise residual.** For models $m, m'$,

```math
\Delta^{mm'}_j(u) = \tau\bigl(\hat d^{m}_j(u)\bigr) - \tau\bigl(g^\ast_{mm'j} \cdot \hat d^{m'}_j(u)\bigr), \qquad
\mathcal G = \begin{cases} \mathcal G_{\mathrm s},\ \tau = \log & \text{both metric},\\ \mathcal G_{\mathrm{ai}},\ \tau(d) = 1/d & \text{otherwise.}\end{cases}
```

In inverse depth the residual is divided by $1/\hat d^m_j(u)$ so that both cases are relative errors.

**Attribution to the primary model.** A pairwise residual is symmetric: it does not say which model is
wrong. The score for $m_0$ is its residual against the consensus of the others, the analogue of the
cavity in model space,

```math
q^{2}_j(u) = \frac{1}{s^{2}_j(u)}\ \operatorname{med}_{m' \neq m_0} \Delta^{m_0 m'}_j(u).
```

With three other models the median is wrong only if two of them share an error.

**Null scale without labels.** If the model errors were mutually uncorrelated, the per-model variances
would follow from the pairwise variances (three-cornered hat),

```math
\operatorname{Var}(\epsilon^{a}) = \tfrac12 \bigl( V_{ab} + V_{ac} - V_{bc} \bigr), \qquad V_{mm'} = \operatorname{Var}\bigl(\Delta^{mm'}\bigr),
```

evaluated with robust variances per geometry stratum. This is the continuous counterpart of estimating
detector accuracies from agreement. The assumption is doubtful: a negative estimate is the signature of
correlated errors and must be reported, not clipped. Independently of that, $s^2$ can always be set to
the robust scale of the consensus residual itself.

**Abstains** where fewer than two other models are valid (saturation at Metric3D's depth clamp, invalid
or sky pixels), where an alignment failed or has few inliers (`AlignmentError`, `n_inliers`), and in a
band around depth edges if edge sharpness differs between models.

**Responds to** content on which the models' learned priors differ. It is a statement about the models,
not about the scene.

**Limits.** (i) Common-mode error: all four models use a ViT encoder pretrained in the DINOv2 family and
overlapping training data (recalled; to be verified per checkpoint), so on glass and water a shared
error is plausible. Then the consensus is wrong with $m_0$ and c2 misses. How large this share is, is the
first thing the MatrixCity measurement must answer. (ii) One model is poor after alignment: on the cached
MatrixCity block, Metric3D v2 keeps 9.5 to 14% median relative error after per-view scaling, against
under 1% for UniDepthV2. Its residuals widen the null; the median over three and the variance estimate
limit the damage. (iii) The error of the *other* models leaks into the score of $m_0$: where two others
fail together, c2 false-alarms on a correct $m_0$.

**Cost and code.** No new inference: the cache holds raw outputs of all four models
(`depthcache/cache.py`), 0.22 to 1.35 s per view and model on the recorded GPU run. The alignment
exists: `depthcache/align.align_depth(result, ref_depth)` accepts any reference depth, so passing another
model's depth as reference gives $\mathcal G_{\mathrm s}$ (median ratio, `fit_scale`) or
$\mathcal G_{\mathrm{ai}}$ (consensus fit plus least squares, `fit_affine_inverse`) directly. Missing: a
function that forms the six pairwise residuals and the consensus score, the stratified scales, and the
edge band. Storage: the default cache precision is float16 (a 0.25 m step at 256 to 512 m), coarse next
to small disagreements; float32 is needed. Cost per view: a few robust fits on subsampled pixels, CPU
seconds.

## 4. Cue c3: Instability under Input Transformations

**Setting.** One model $F$, a set $\mathcal T$ of $S$ input transformations $T$ with geometric action
$\psi_T$ on pixel coordinates: horizontal and vertical flip, rotation by multiples of 90 degrees,
centred crop (less context), rescale by a factor $\rho$ (other resolution), and photometric changes
(gamma, colour balance) with $\psi_T = \mathrm{id}$. The prediction transported back is

```math
\hat d^{T}_j(u) = F\bigl(T[I_j]\bigr)\bigl(\psi_T(u)\bigr), \qquad u \in \mathcal D_T = \{u : \psi_T(u) \text{ inside the transformed image}\}.
```

**Nuisance.** Each forward pass has its own gauge. A metric model's scale depends on the assumed field of
view; under a rescale by $\rho$ the focal length in pixels must be passed as $\rho f$, under a crop it
stays $f$ with a shifted principal point, under a flip the principal point is mirrored. Even with correct
intrinsics the predicted scale moves, and for the affine-invariant model the gauge is arbitrary per
pass. The comparison is therefore made after alignment on the overlap,

```math
\Delta^{T}_j(u) = \tau\bigl(\hat d_j(u)\bigr) - \tau\bigl(g^\ast_{T} \cdot \hat d^{T}_j(u)\bigr), \quad u \in \mathcal D_T ,
```

with the same group as for the model in c2. Resampling is needed for crop and rescale; to keep sub-pixel
shifts at depth edges out of the score, the residual is taken with a one-pixel tolerance,
$\Delta^{T}_{\mathrm{tol}}(u) = \min_{\lVert u' - u \rVert_\infty \le 1} \lvert \tau(\hat d_j(u)) - \tau(g^\ast_T \cdot \hat d^T_j(u')) \rvert$.

**Score.** The dispersion over transformations, standardised by a learned scale,

```math
q^{3}_j(u) = \frac{1}{s^{3}_j(u)} \Bigl( \frac{1}{S_u} \sum_{T :\, u \in \mathcal D_T} \Delta^{T}_{\mathrm{tol}}(u)^2 \Bigr)^{1/2}.
```

The score is unsigned; its null law is fitted empirically per stratum and per transformation family,
because the transformations are not exchangeable: a flip of a nadir image is nearly a symmetry of the
scene statistics, whereas a model trained mostly on ground-level images has a gravity prior, and
rotations may change clean predictions by a large systematic amount. If so, that family has no
usable null and is discarded.

**Abstains** where fewer than two transformations cover the pixel, where an alignment failed, and in the
edge band.

**Responds to** predictions that are weakly determined by local evidence and therefore depend on
position, context or resolution. This is *ambiguity*, not *error*. A confident systematic error, which
is the description of the glass failure (the model reads the reflected or transmitted scene), is
equivariant and produces no score. Instability on correct predictions is a false alarm. Whether
ambiguity predicts error in this domain is an empirical question; the evidence from test-time
augmentation in other tasks is that it captures part of the error and misses confident mistakes
(recalled).

**By-product.** The spread of the fitted gauges $g^\ast_T$ over crops measures how well the whole-view
scale is determined. It is a view-level quantity about the nuisance, not a fault cue.

**Cost and code.** $S$ additional forward passes per view and model (four passes with UniDepthV2: about
0.9 s per view on the recorded run). Nothing exists: the wrappers expose
`predict(rgb, focal_px)` with a centred principal point, so off-centre crops cannot be expressed and
crops must be centred or the shift left to the alignment; `View.load_rgb` gives the image. To add: the
transformation and inverse-warp functions, focal handling, cache entries per transformation (the cache
accepts any model identifier matching `[A-Za-z0-9._-]+`, but `depthcache/run.py` validates against the
four known names), and the score.

## 5. Cue c4: Photometric Evidence

Colour is used as a judge. It never moves geometry.

### 5.1 Variant (i): multi-view photo-consistency of the depth

**Why a single-depth score is not a cue.** Reprojecting pixel $u$ of view $j$ through its depth into a
neighbour and scoring the patch similarity gives a number with no reference distribution: on a
texture-free surface the patch matches at every depth, on repeated texture at many. The well-posed
quantity is the *profile* of the similarity along the ray, compared at the hypothesised depth and at its
alternatives.

**Definition.** Let $\mathcal N_j$ be neighbouring views with sufficient baseline. For a depth
hypothesis $d$ and a local plane through $X_j(u, d)$ (fronto-parallel, or with the normal of the depth
map), let $H_{ji}(u, d)$ be the plane-induced homography and

```math
\rho_{ji}(u, d) = \operatorname{ZNCC}\bigl( I_j[\mathcal P(u)],\ I_i[H_{ji}(u, d)\, \mathcal P(u)] \bigr), \qquad
\bar\rho_j(u, d) = \frac{1}{\kappa} \sum_{i \in \text{best-}\kappa \text{ of } \mathcal N_j} \rho_{ji}(u, d),
```

the zero-mean normalised cross-correlation on a patch $\mathcal P$, averaged over the $\kappa$ best
neighbours to tolerate occlusion. Normalisation removes gain and offset of brightness per patch, which
is the photometric nuisance. The profile is evaluated on $H$ hypotheses uniform in inverse depth over
the depth range of the view, at a spacing of about one pixel of disparity. Let
$d^\star(u) = \arg\max_d \bar\rho_j(u, d)$, refined by a parabola. The score is the standardised
difference between the aligned monocular depth and the photometric optimum,

```math
q^{4}_j(u) = \frac{\log d^{\mathrm{al}}_j(u) - \log d^\star(u)}{\sqrt{\sigma^2_{\mathrm{ph}}(u) + \sigma_j^2(u) / d^{\mathrm{al}}_j(u)^2}}, \qquad
\sigma^2_{\mathrm{ph}}(u) \propto \Bigl( - \partial^2_{\log d}\, \bar\rho_j(u, d) \big|_{d^\star} \Bigr)^{-1} .
```

The curvature of the profile at its peak is the product of the image gradient energy along the epipolar
direction and the squared disparity sensitivity $f B / d$; it plays the role of the rendered variance in
c1. The proportionality constant is learned as part of the null scale. Variant (i) is thus the residual
of the monocular depth against a local triangulation, a measurement from a different physical principle.

**Alignment.** $d^{\mathrm{al}}_j$ needs $(a_j, b_j)$. Taking them from the cavity alignment of c1 would
couple the two cues through a shared nuisance estimate. The cue has its own: the robust regression of
$\hat d_j$ on $d^\star$ over pixels with a distinctive peak,

```math
(a^{\mathrm{ph}}_j, b^{\mathrm{ph}}_j) = \arg\min_{a, b} \sum_{u :\, A^4_j(u) = 0} \rho\bigl( \hat d_j(u) - a\, d^\star(u) - b \bigr),
```

so that variant (i) uses images, poses and the depth map of view $j$ and nothing else: no surfel map, no
other depth map.

**Abstains** when the peak is not distinctive (ratio of best to second-best local maximum below a
threshold, or $\bar\rho(d^\star)$ below a floor), when the epipolar gradient energy of the patch is below
a floor, when fewer than two neighbours see the point, and when the alignment has too few inliers.

**Responds to** a displacement of the surface along the ray that changes the appearance in other views:
it requires texture, a Lambertian surface, a static scene and correct poses.

**Failure modes.** On specular and transparent surfaces the photometric optimum is the depth of the
virtual image, not of the surface; if the monocular depth is correct there, the cue false-alarms, and if
both are wrong they are wrong differently. On water, glint and waves leave no stable peak and the cue
abstains. On texture-free surfaces it abstains. So on the three fault classes that motivate the work,
variant (i) mostly abstains. Its value is elsewhere: it is the only multi-view cue that does not use the
other views' depth maps, so it can see an error that every view repeats, provided the surface is
textured (for example building heights flattened consistently in nadir views).

**Abstention is informative.** The cue abstains where texture is absent, which is where monocular faults
are more frequent. Abstention is not missing at random, and a label model must carry class-conditional
abstention rates or condition on the texture covariate.

**Cost and code.** $H \cdot \lvert \mathcal N_j \rvert$ image warps per view at reduced resolution; with
tens of hypotheses and four neighbours this is of the order of a second on a GPU and feasible on a CPU at
quarter resolution. Nothing exists on the image side: `types.DepthFrame` has no image field, the surfel
fit renders constant colours, and the only image access is `depthcache/datasets.View.load_rgb`, which is
not connected to the frames used by `gs/`. Available building blocks: `baselines/_common`
(`backproject_cam`, `project`, `project_into`, `depth_to_normals`). To add: an image-carrying frame or a
side loader keyed by view identifier, the plane sweep, the peak statistics, the alignment.

### 5.2 Variant (ii): influence through the surfel map

**Setup.** Linearise the depth likelihood of view $i$ at the fitted surface $\bar\theta$. With
$\delta_k$ the perturbation of the geometric coordinates of surfel $k$ and the block-diagonal
approximation of the main document,

```math
\mathcal L_i(\bar\theta + \delta) \approx \mathcal L_i(\bar\theta) - \sum_k \eta_{ik}^\top \delta_k + \tfrac12 \sum_k \delta_k^\top \Lambda_{ik} \delta_k, \qquad
\eta_{ik} = \sum_u \frac{a_i\, \mathbf g_{uk}^\top e_i(u)}{\sigma_i^2(u)}, \quad \Lambda_{ik} = \Lambda_k^{(i)},
```

where $e_i(u) = \hat d_i(u) - a_i \tilde D_i(u) - b_i$ is the residual against the *full* surface. Give
each message a gate $w_{ik}$. The gated estimate is in closed form,

```math
\Lambda_k(w) = \Lambda_0 + \sum_i w_{ik} \Lambda_{ik}, \qquad \delta_k(w) = \Lambda_k(w)^{-1} \sum_i w_{ik}\, \eta_{ik},
```

with the prior centred at $\bar\theta_k$. Differentiating,

```math
\frac{\partial \delta_k}{\partial w_{ik}} = \Lambda_k^{-1} \bigl( \eta_{ik} - \Lambda_{ik} \delta_k \bigr) = \Lambda_k^{-1} \eta_{ik} \quad \text{at } w = \mathbf 1,\ \delta_k = 0 ,
```

because at the fitted surface $\sum_i \eta_{ik} = 0$. Removing the message entirely gives
$\delta_k^{(-ik)} = -(\Lambda_k - \Lambda_{ik})^{-1} \eta_{ik}$, the cavity step, of which the derivative
is the first-order form.

**Influence.** Let $\mathcal L^{\mathrm{rgb}}_j(\theta, \mathbf c)$ be the photometric loss of a held-out
view $j$, with appearance parameters $\mathbf c$ fitted on other views with the geometry frozen. Then

```math
I(i \to j, k) = \frac{\partial \mathcal L^{\mathrm{rgb}}_j}{\partial w_{ik}} = \nabla_k \mathcal L^{\mathrm{rgb}\,\top}_j\ \Lambda_k^{-1}\ \eta_{ik} .
```

A positive value says that giving the message more weight makes the held-out image worse.

**Pixel form.** Substituting $\eta_{ik}$ and summing over surfels,

```math
\sum_k I(i \to j, k) = - \sum_u \frac{a_i\, e_i(u)}{\sigma_i^2(u)}\ \delta D^{\mathrm{rgb}, j}_i(u), \qquad
\delta D^{\mathrm{rgb}, j}_i(u) = \sum_k \mathbf g_{uk}\, \delta\theta^{\mathrm{rgb}, j}_k, \quad \delta\theta^{\mathrm{rgb}, j}_k = - \Lambda_k^{-1} \nabla_k \mathcal L^{\mathrm{rgb}}_j .
```

The influence of pixel $u$ of view $i$ is the product of two things: its depth residual against the full
surface, and the change of rendered depth at that pixel if the surface followed the colour gradient,
preconditioned by the depth information. It is positive when the depth datum pulls the surface one way
and colour pulls it the other way.

**Three structural consequences.**

1. *Zero sum.* Since $\sum_i \eta_{ik} = 0$, the influences of the views on a surfel sum to zero,
   $\sum_i I(i \to j, k) = 0$. The influence ranks messages against each other; it is not an absolute
   statement about the surface. When two views disagree one gets a positive and the other a negative
   value. When all views agree, every $\eta_{ik}$ is near zero and every influence is near zero, however
   large the photometric gradient is. As a per-message score, variant (ii) is therefore blind to an
   error that all views repeat, exactly like depth consensus. The information about such an error is in
   $\nabla_k \mathcal L^{\mathrm{rgb}}_j$ alone.
2. *It contains the non-cavity residual.* The factor $e_i(u)$ is measured against a surface that view
   $i$ helped to fit. A view that dominates a surfel has a small residual there and a small influence.
   The self-confirmation that the leave-out removes from c1 is present in variant (ii).
3. *No known null as it stands, but one is available.* The influence is a loss derivative with units
   that depend on contrast and noise. Under the hypothesis that the geometry is correct and the
   photometric noise is independent with variance $\sigma_c^2$, the gradient
   $\nabla_k \mathcal L^{\mathrm{rgb}}_j$ has zero mean and covariance equal to the photometric Fisher
   information $\mathcal F_k = \sigma_c^{-2} \sum_{u'} \mathbf J_{u'k}^\top \mathbf J_{u'k}$, with
   $\mathbf J_{u'k}$ the Jacobian of the rendered colour. Conditional on the depth data,

```math
\frac{I(i \to j, k)}{\sqrt{\mathbf v_{ik}^\top \mathcal F_k\, \mathbf v_{ik}}} \sim \mathcal N(0, 1), \quad \mathbf v_{ik} = \Lambda_k^{-1} \eta_{ik}; \qquad
T_k = \nabla_k \mathcal L^{\mathrm{rgb}\,\top} \mathcal F_k^{-1} \nabla_k \mathcal L^{\mathrm{rgb}} \sim \chi^2_{\operatorname{rank} \mathcal F_k}.
```

   The first is a score test of the photometric fit along the direction in which message $i$ pushes
   surfel $k$. The second is the omnibus score test on the surfel, with no reference to any message; it
   is the form that can see an error repeated by all views. Both need $\mathcal F_k$, and both are
   exact only for a correct appearance model.

**What the gradient contains.** With rendered colour $\hat C(u') = \sum_l w_{u'l}\, \mathbf c_l$ and
front-to-back compositing,

```math
\nabla_k \mathcal L^{\mathrm{rgb}}_j = \sum_{u'} \underbrace{\rho'\bigl(\hat C(u') - I_j(u')\bigr)}_{\text{photometric residual}}\ \underbrace{\bigl(\mathbf c_k - \hat C^{\,\mathrm{behind}}_{k}(u')\bigr)}_{\text{colour contrast}}\ \underbrace{T_{u'k}\, \nabla_k\bigl(\alpha_k G_k(u')\bigr)}_{\text{footprint change}} ,
```

where $\hat C^{\mathrm{behind}}_k$ is the colour composited behind surfel $k$, $T_{u'k}$ the
transmittance in front of it and $G_k$ its footprint. The footprint depends on the geometry through the
intersection of the pixel ray with the surfel plane. A displacement $\nu$ along the normal shifts that
intersection in the tangent plane by $\nu \tan\vartheta$, with $\vartheta$ the angle between ray and
normal. Hence:

- **Zero without contrast.** Where a surfel has the colour of its surroundings the gradient vanishes
  whatever the geometric error. Detected by a small $\mathcal F_k$; the cue abstains there.
- **Zero at normal incidence.** For a held-out view that looks along the surfel normal ($\vartheta = 0$,
  the centre of a nadir image over flat ground) a normal offset produces no image motion. The
  information comes from oblique rays only, so it depends on the angular spread of the held-out views.
- **Local validity.** The gradient is a first-order statement, correct while the image motion caused by
  the geometric error is below the correlation length of the texture, in practice one to two pixels of
  parallax. A fault of many $\sigma$ moves the projection by many pixels; the gradient then has a large
  norm and an arbitrary direction. Its sign, and so the sign of the influence, is unreliable for gross
  faults, which are the faults of interest. An image pyramid extends the range at the cost of
  resolution.
- **Appearance absorbs geometry.** The colours are fitted on images rendered through the wrong
  geometry. A constant colour per surfel becomes a blur; a view-dependent colour can represent the
  parallax of a misplaced surface as view dependence, and the held-out residual of a nearby view is then
  small with wrong geometry. The judge must use view-independent colour, and the held-out view must be
  outside the block of views that fitted the colour.
- **Non-Lambertian surfaces.** At the true geometry of a specular or transparent surface the residual is
  not zero. The gradient is non-zero for correct geometry and points towards the virtual image: a false
  alarm with a misleading direction.
- **Exposure.** Brightness differences between frames produce residuals at correct geometry. A
  per-view gain and offset, or a normalised patch loss, is needed: one more nuisance alignment.
- **Self-supported surfels.** A surfel that only faulty views support takes its colour from them and
  occludes the true surface in a held-out view. The loss is high, but the relevant derivative is with
  respect to existence (opacity), not position; the geometric gradient says nothing useful.

**A finite-difference form.** The sum over surfels $\sum_k I(i \to j, k)$ is the directional derivative
of the held-out loss along the field $\mathbf v_i$. Its finite form is the difference of two renders,

```math
\Delta \mathcal L^{(-\mathcal B_i)}_j(u') = \ell_j\bigl(u';\ \theta^{(-\mathcal B_i)}\bigr) - \ell_j\bigl(u';\ \theta\bigr),
```

the per-pixel photometric loss of the held-out view rendered from the cavity surface without the block
of view $i$, minus that from the full surface. It asks whether removing the depth of the block improves
the prediction of an image, it is valid beyond the linear range, and it handles self-supported surfels
because the cavity drops them. It has no analytic null; its scale would be learned.

**Cost and code.** Everything is missing: images in the frames, an appearance model (the fit calls the
rasteriser with constant colours), the colour fit with frozen geometry (linear least squares for
view-independent colour, one solve per fold of held-out views), the geometric gradient of the colour
render in the gate's nine local coordinates (the fit already parametrises rotation by a local
perturbation, so the chain rule is in place for that path), and $\mathcal F_k$, which needs per-pixel
Jacobians: either a diagonal estimate by random probing (a forward-mode and a reverse-mode pass per probe
and held-out view) or the empirical Fisher from per-view gradients. Cost per view: one backward pass
through the colour renderer per held-out view for the gradient, several more for $\mathcal F_k$; the
finite-difference form costs one colour render of the cavity per pair $(i, j)$. This is the most
expensive cue and the one with the most new code.

### 5.3 Relation between the two variants

Both ask whether a depth is compatible with the images. Variant (i) searches along the ray, has a wide
range of validity, a natural abstention rule and no dependence on the surfel map. Variant (ii) is the
local derivative of the same criterion through the surfel map: it lives on (view, surfel) pairs, which is
the domain the gate needs, but it is valid only near the correct surface, it inherits the map's model
error and the appearance model's ambiguities, and as an influence it is blind to common errors.

## 6. Cues from Frozen Foundation-Model Features

The learned component is a light head on frozen features, so it is natural to ask which cues can be
computed in feature space, and whether that makes them more or less independent of the others. Let
$\phi(I)$ be a frozen patch-feature map (a DINO-family image encoder, or the encoder of the depth model
itself), upsampled to the working resolution.

**c4f: cross-view feature consistency.** Replace the patch correlation of variant (i) by the cosine
similarity of features, $\rho^{\phi}_{ji}(u, d) = \cos\bigl(\phi(I_j)(u),\ \phi(I_i)(\pi_i X_j(u, d))\bigr)$,
with the same profile, peak and abstention statistics.

- *Gains.* Invariance to exposure and partly to specular highlights; a wider basin; some signal on
  weakly textured surfaces, because features encode context.
- *Losses.* Resolution: features live on a patch grid of the order of 14 pixels (recalled), so the
  disparity precision is several pixels against a fraction of a pixel for patch correlation, and
  $\sigma_{\mathrm{ph}}$ grows by the same factor. The cue remains useful only for faults whose parallax
  exceeds a few feature cells; whether regional faults at aerial baselines do, has to be computed from
  the geometry of each data set before implementing. Features also carry positional information and are
  not viewpoint invariant, so the similarity has a bias that depends on image position; it must be
  learned as part of the null.
- *Independence.* Less independent of c2 and c3 than the pixel version. If the depth models' encoders
  are of the same family as $\phi$ (recalled, to be verified), the regions where the features are
  ambiguous are the regions where the depth prediction is weakly determined: c4f degrades or abstains
  where c3 fires and where the models tend to fail. The pixel version shares only the image with the
  depth models. With respect to c1 both versions are equally independent, since neither uses depth maps
  of other views. The pixel version stays the primary judge; c4f is a coarse complement that narrows
  the abstention region, entered as a separate channel of the same cue, not as a further detector.

**c3f: feature-space instability.** With the depth model's own frozen encoder $\phi_F$,

```math
q^{3f}_j(u) = \frac{1}{s^{3f}_j(u)} \Bigl( \frac{1}{S_u} \sum_{T} \bigl[ 1 - \cos\bigl( \phi_F(I_j)(u),\ \phi_F(T[I_j])(\psi_T(u)) \bigr) \bigr] \Bigr) .
```

Features have no scale or offset gauge, so no nuisance alignment is needed, which removes the weakest
step of c3 (the alignment per pass) and its breakdown. The price is a null that is far from zero: a
vision transformer with positional embeddings is not equivariant to flips, crops or rescaling by
construction, and the patch grid shifts under crop and rescale. The null is learned per transformation
as for c3. As a detector c3f is *not* independent of c3: the depth output is a function of these
features, so the two share their cause. It is a replacement for c3 or a second channel of it, with one
distinct property: it can show instability that the decoder smooths away, and it can be insensitive to
instability the decoder creates. The same forward passes serve both, so computing both costs only a
hook on the encoder.

**c2f: cross-model feature disagreement.** Feature spaces of different models are not comparable
directly. A linear map fitted without labels (ridge regression of $\phi^{m'}$ on $\phi^{m}$ over all
pixels of a scene) gives a residual that marks content the two models encode differently. Its relation
to a depth fault is indirect, and it is strongly dependent on c2 and c3f. Not recommended as a cue.

**Features of a frozen multi-view model.** A frozen two-view or multi-view geometry model that outputs
point maps (recalled: such models exist; none is in the code) would give a depth for view $j$ that uses
its neighbours' images; the residual of the monocular depth against it, after alignment, is a learned
counterpart of variant (i) that does not abstain on texture-free surfaces. Its errors there come from
learned priors of the same kind as the monocular models', so it is predicted to be dependent on c2
exactly where it adds coverage. It should be judged by the same measurement as the other cues before
any use.

**Features as covariates.** The uncontroversial use of frozen features is as the appearance input
$\mathbf x^a$ of the head and of the distilled network, and as stratification variables of the nulls.
Two cautions. Conditional independence is then required given the fault *and* the features, which is a
weaker and more plausible requirement than unconditional independence. And a null that is stratified by
appearance can absorb a fault class tied to appearance (all glass), which the main document avoids by giving the
scale head geometry only. Appearance features may inform priors and abstention, not null scales.

**Summary of frozen-feature use.**

| Component | Frozen features | Purpose | Independence against pixel version |
| --- | --- | --- | --- |
| c4f | image encoder (DINO family) | coarse cross-view consistency | lower against c2, c3; equal against c1 |
| c3f | encoder of the depth model | alignment-free instability | same cause as c3: a channel, not a detector |
| c2f | encoders of two depth models | not recommended | dependent on c2 and c3f |
| head, student | image encoder | priors, abstention, distillation | not a cue |

## 7. Predicted Behaviour per Fault Class

F: fires on a true fault. M: misses. A: abstains. FA: fires where the primary depth is correct. Entries
are predictions. "c4(ii)" refers to the per-message influence; the omnibus score test $T_k$ behaves like
c4(i) within its linear range.

| Fault class | c1 cavity (and c1r) | c2 cross-model | c3 instability | c4(i) plane sweep | c4(ii) influence |
| --- | --- | --- | --- | --- | --- |
| Specular, glass | F if the error changes with viewpoint beyond the block; M if consistent | F if models differ; M on common-mode error (share unknown) | M for confident errors; partial F at region borders | A (weak peak) or FA (optimum at the virtual image while the depth is right) | FA, misleading direction |
| Water | F if inconsistent across views; M within a block | partial F | F (weakly determined prediction) | A (no stable texture) | A (no contrast) or FA (glint) |
| Texture-free | M within a strip (context repeats); F across strips | partial F | F: the best case | A, detected by gradient energy | A, detected by small $\mathcal F_k$ |
| Thin structures | FA (the surfel map cannot represent them); M if every view omits them | FA (sharpness differs between models) | FA under rescale | A or FA (patch straddles) | FA, large gradients |
| Depth discontinuities | FA unless the scale head absorbs it | FA in the edge band | FA from resampling | A or FA (occlusion) | FA |
| Whole-view scale or offset | M by construction (in the nuisance) | M by construction | M by construction; visible only in the spread of gauges | M by construction | M: the alignment enters $e_i$ |
| Repeated in neighbouring frames | M unless the block is longer than the repetition | as for a single view; its own errors repeat too | as for a single view | F if textured and Lambertian | M: $\eta_{ik}$ is small for all views of the run |
| Repeated by all views | M | F unless common to the models | as for a single view | F if textured and Lambertian: the only multi-view cue that can | M (zero sum); $T_k$ can fire within its linear range |

Three readings of the table.

- No cue fires reliably on glass or water. c1 depends on view dependence of the error, c2 on diversity of
  the models, c3 on the error being unconfident, and c4 abstains or misleads. The coverage of the union
  on these classes is an open empirical question, not a property of the design.
- On thin structures and discontinuities every cue false-alarms. These are not faults by definition, so
  they are joint false alarms on the negative class.
- A whole-view affine error is missed by all cues by construction. For Depth Anything V2 the nuisance is
  not affine in depth, so a cue aligned with the wrong group reports a view-wide residual on a clean
  view.

## 8. Predicted Violations of Conditional Independence

A label model that estimates detector accuracies from agreement assumes that the votes are independent
given the latent class. Where this is predicted to fail:

1. **Edges and thin structures: all cues, given $f = 0$.** A common cause (a depth edge within the
   pooling window) raises the false-alarm probability of every cue together. Left untreated, the label
   model reads the agreement as a fault class of its own and may attach the "fault" label to it. Remedy:
   a common abstention band around depth edges for all cues, or estimation of accuracies stratified by
   distance to the nearest edge.
2. **c2 and c3 (and c3f), given $f = 1$.** Both are functions of depth networks on the same image.
   Faults come in at least two kinds: weakly determined predictions, on which both fire, and confident
   systematic errors, on which both miss if the error is shared between models. A hidden sub-class
   induces positive dependence within the fault class. With binary classes the pair is expected to
   count closer to one detector than to two.
3. **c1 and c1r.** The same data. One cue.
4. **c1 and c4, given $f = 0$.** Both use the pose of view $j$ and the poses of its neighbours; a pose
   error gives both a regional residual. If c4 took its alignment from the cavity, an alignment error
   would also be shared; the separate photometric alignment of Section 5.1 removes that path. Their
   measurement noise is otherwise from different sources (depth maps of other views against images of
   other views).
5. **c1 and c4(ii).** The influence contains the full-surface depth residual $e_i$ and the information
   blocks $\Lambda_k$; it is a function of the same quantities as c1 multiplied by a photometric factor.
   Strong dependence by construction.
6. **c4 against the fault class, through abstention.** Texture-free and non-Lambertian regions make
   monocular faults likelier and make c4 abstain. If abstention is treated as uninformative, the
   accuracy of c4 is estimated on textured regions and extrapolated to regions where it never votes.
7. **Pairwise c2 residuals.** The six residuals share models pairwise and are dependent by construction;
   only the consensus score per target model is a cue.
8. **Shared alignment breakdown.** A fault covering more than the breakdown fraction of a view inverts
   every cue that aligns on that view (c1, c2, c3, c4), jointly.
9. **Spatial dependence.** Votes of neighbouring pixels are not independent replicates. This does not
   bias accuracy estimates but makes their uncertainty far larger than the pixel count suggests.

The pairs for which approximate conditional independence is defensible are (c1, c2), (c1, c3) and
(c2, c4), (c3, c4) outside the edge band: in each pair one cue uses other views and the other does not,
or one uses depth networks and the other does not. A triple with pairwise independence needs one cue
from each of three groups: multi-view depth (c1), single-image network (c2 or c3, not both as separate
detectors), multi-view image (c4(i)). That is exactly three groups, the minimum for identifiability, and
the third is the one with the largest abstention region. Where c4 abstains, only two independent groups
remain and the accuracies are not identifiable from agreement alone at those pixels; they must be
carried over from pixels where all three vote, which assumes that accuracies do not depend on the
texture that governs abstention. That assumption is unlikely to hold exactly.

**What a small labelled set buys for the cues.** With a few labelled views (exact-depth fault masks of
one synthetic scene) and no change to the cue definitions: the null scales $s^c$ can be checked on
known-clean pixels in place of the minority assumption; the class-conditional firing and abstention
rates of each cue can be anchored, which fixes the label-swap ambiguity and the problem of point 6; the
pairwise dependence of the cues given the true class can be measured directly, which decides whether c2
and c3 are merged; and the orientation of model confidence maps, which is not consistent between models
or even between views for one of them in the recorded run, can be fixed. Rates anchored on a synthetic
scene transfer to real data only as far as the fault mechanisms do.

## 9. Further Candidate Cues

- **Model confidence.** Two of the four models output a confidence. On the cached block one reads as
  predicted error and the other changes direction between views. It is a model-specific input for the
  head, not a cue with a null.
- **Sparse triangulated points.** Where poses come from bundle adjustment, the tie points give
  residuals $\hat d_j(u_p) - a_j d_p - b_j$ at sparse textured pixels. This is a sparse, cheap instance
  of c4(i) with an exactly known variance, absent in synthetic data with given poses. It belongs to the
  c4 group.
- **Cross-strip residual.** c1 with the whole strip as block. It is a setting of c1, and the direct
  answer to errors repeated along a strip, at the price of a smaller testable set.

None of these adds a fourth independent group.

## 10. Cost and Code Summary

| Cue | Per view | Exists | To add |
| --- | --- | --- | --- |
| c1 | information pass per block view, one cavity render | blocks, subtraction, Newton step, support rules, cavity render, bias downdate, gate driver | export of the residual map; noise field; robust alignment; removed share; ray form from the visibility baselines |
| c2 | CPU seconds; inference already cached (0.2 to 1.4 s per model) | four cached models; scale and affine-inverse robust alignment against any reference depth | pairwise residuals, consensus score, stratified scales, edge band; float32 cache |
| c3, c3f | $S$ forward passes (about 0.2 to 1.4 s each) | wrappers with `predict(rgb, focal_px)`, image loader | transformations and inverse warps, focal handling, cache keys, encoder hook, score |
| c4(i), c4f | tens of warps times neighbours; one encoder pass for c4f | projection helpers, normals from depth | images in frames, plane sweep, peak statistics, own alignment |
| c4(ii) | colour fit per fold; backward passes per held-out view; more for $\mathcal F_k$ | local rotation parametrisation in the fit | appearance model, colour fit, geometric gradient of colour, Fisher estimate |

## 11. Recommendation

**Carry forward:** c1 (with the ray form as a channel), c2, c4(i), and c3 on probation.

**Order of implementation.**

1. **c2.** No inference and no new geometry; the alignment code exists. It is the cheapest test of the
   whole approach: on the scene with exact depth it answers at once how much of the models' error is
   common-mode. If the consensus of the other models fails wherever the primary model fails, the
   single-image group contributes little and the plan must change before anything else is built.
2. **c1.** The computation exists; the work is exporting the per-pixel residual, making the alignment
   robust and adding the ray form. With c1 and c2 the first pairwise dependence measurement is possible.
3. **c4(i).** The third independent group and the only one that can see errors common to all depth
   maps. It needs the image path, which the head and the distilled network need anyway. Its abstention
   map should be measured before the score: if it abstains on nearly all true faults, it serves as a
   check on textured regions and not as a third voter.
4. **c3 with c3f.** Implemented last and kept only if, given the true class, it is not redundant with
   c2 and it has a usable null for at least one transformation family. Otherwise it is merged into the
   single-image group as a channel.

**Drop:** c4(ii) as a cue. As a per-message influence it sums to zero over views, contains the
non-cavity depth residual, is valid only within a pixel or two of parallax, and needs an appearance
model that can hide the error it should reveal; it is dependent on c1 by construction and is the most
expensive to build. Its sound core is the score test $T_k$ and the finite-difference form on the cavity
render; these are a later, surfel-domain refinement of c4(i), worth building only if c4(i) proves
informative.

**Not known.** The share of common-mode error between depth models on glass and water; whether
instability predicts error in nadir imagery; the abstention rate of c4(i) on true faults; whether
parallax at the given baselines is large enough for feature-resolution matching; whether any triple of
cues is close enough to conditional independence for the accuracies to be identifiable. All five are
decided by the measurement on exact depth, and none by argument.
