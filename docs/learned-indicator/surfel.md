# The Fault Indicator in Surfel Space, and How It Enters Reconstruction

## 0. Scope, notation, and summary of findings

This note develops the formulation in which the fault indicator is attached to (view, surfel) pairs,
examines where that formulation is strong and where it fails, and ends with a recommendation. Notation
follows the notation table of `docs/learned-indicator.md`; additions are listed here.

| Symbol | Meaning |
| --- | --- |
| $\theta_k \in \mathbb R^{10}$ | perturbation of surfel $k$: $[\delta\mu\ (3),\ \delta\phi\ (3),\ \delta\log s\ (3),\ \delta\operatorname{logit}\alpha\ (1)]$; the first nine are the geometric coordinates |
| $G_k \in \mathbb R^{3\times 10}$ | map to normal offset and tilt, $(\nu, \tau_1, \tau_2) = G_k\theta_k$ (`subblock_map`) |
| $w_{uk}$ | normalised compositing weight of surfel $k$ at pixel $u$ (pixel $u$ belongs to one view; $\sum_k w_{uk} = 1$ on covered pixels). `w_hat` in the code |
| $a_{uk}$ | the surfel's own alpha at the pixel, opacity times kernel, before transmittance |
| $\mathbf q_{uk} \in \mathbb R^{1\times 3}$ | derivative of the ray-surfel depth with respect to $(\nu,\tau_1,\tau_2)$ (`depth_jacobian_subblock`); $\mathbf j_{uk} = \mathbf q_{uk} G_k$ |
| $e_i(u)$ | residual at the fitted surface, $\hat d_i(u) - (a_i \tilde D_i(u) + b_i)$ |
| $(\Lambda_{ik}, \eta_{ik})$ | message of view $i$ to surfel $k$; $\Lambda_k^{(i)}$ in the main document |
| $n_{ik} = \sum_{u \in i} w_{uk}$ | effective number of pixels of view $i$ rendered by surfel $k$ |
| $w_i(u)$, $w_{ik}$ | gate on a pixel and on a pair; $w = 1 - \zeta$. Distinguished from compositing weights by their indices |
| $c_{\mathrm{eff}}$ | information deflation $n_{\mathrm{eff}}/n$ |

Findings, stated up front so that the derivations can be read against them.

1. The information the gate multiplies must be the *lumped* block ($w_{uk}$, not $w_{uk}^2$). Only then is
   the closed-form shift under a change of gates a majorisation step that cannot overshoot.
2. Pooling to pairs and rendering back are adjoint under the measure $n_{ik}$. Their composition is a
   smoothing operator of footprint width, not a projection. The loss matters most on large surfels, and
   large surfels sit on flat, texture-free surface, where faults are most likely.
3. The see-through problem has a clean solution, but the solution is the pixel-space cavity residual
   pooled with full-surface weights. The cross-ray free-space score adds coverage and a pruning rule; it
   does not add an independent source of evidence, because it is again depth consensus.
4. Multi-modal messages at a surfel and see-through between two surfels are the same event in two
   representations. A detector at the surfel can conclude *that* a group of views is inconsistent, and
   can say *which* only when independent blocks (not frames) on one side outnumber the other.
5. A pair gate is the special case of a pixel gate that is constant over a footprint, and the exact
   gated message for an arbitrary pixel gate costs the same to compute. There is therefore no reason to
   restrict the indicator to pair resolution. The recommendation is a hybrid: the indicator is a pixel
   field; the surfel structure supplies association across views, arbitrary leave-out sets, gated
   fusion and nuisance statistics in closed form.
6. Frozen image features lift onto surfels with the same pooling operator, in closed form. The learned
   part is then a light head on per-pair and per-surfel features. Cross-view feature disagreement at a
   surfel is the one surfel-space quantity that does not reduce to depth consensus, but it is weakest on
   homogeneous surface, and it speaks about the surfel more than about a particular view.

## 1. Messages, gates, and gated fusion

### 1.1 The message of a view to a surfel

Linearise the rendered depth of pixel $u$ of view $i$ at the fitted surface, compositing weights held
fixed:

```math
\hat D_i(u;\theta) = \sum_k w_{uk}\bigl(\bar d_{uk} + \mathbf j_{uk}\theta_k\bigr),
\qquad \mathbf j_{uk} = \mathbf q_{uk} G_k,
\qquad \mathbf q_{uk} = \frac{\bigl[\,1,\ -\mathbf t_1^\top(\mathbf x_{uk} - \boldsymbol\mu_k),\ -\mathbf t_2^\top(\mathbf x_{uk} - \boldsymbol\mu_k)\bigr]}{\mathbf n_k^\top R_c\,\mathbf r_u} .
```

Depth observes three of the ten coordinates of a surfel: its offset along the normal and its two tilts.
The scale columns are zero to first order and the opacity is frozen after warm-up, so a message is a
$3\times 3$ object embedded by $G_k$; the other six geometric coordinates keep their prior. In words,
the message is a weighted plane fit of the depth map of view $i$ over the footprint of surfel $k$,
expressed as offset and tilt relative to the surfel.

The Gauss-Newton Hessian of the depth loss of view $i$ is not block diagonal:

```math
H^{(i)}_{kl} = c_{\mathrm{eff}} \sum_{u \in i} \frac{a_i^2}{\sigma_i^2(u)}\ w_{uk} w_{ul}\ \mathbf j_{uk}^\top \mathbf j_{ul} .
```

Two block-diagonal surrogates exist, and the code implements both (`view_information`, flag `lumped`):

```math
\Lambda^{\mathrm{sq}}_{ik} = c_{\mathrm{eff}} \sum_{u \in i} \frac{a_i^2\, w_{uk}^2}{\sigma_i^2(u)}\ \mathbf j_{uk}^\top \mathbf j_{uk},
\qquad
\Lambda_{ik} = c_{\mathrm{eff}} \sum_{u \in i} \frac{a_i^2\, w_{uk}}{\sigma_i^2(u)}\ \mathbf j_{uk}^\top \mathbf j_{uk},
```

with the same information vector in both cases,

```math
\eta_{ik} = c_{\mathrm{eff}} \sum_{u \in i} \frac{a_i\, w_{uk}}{\sigma_i^2(u)}\ \mathbf j_{uk}^\top e_i(u) = -\nabla_{\theta_k} \mathcal L_i .
```

(The opacity row and column, built from $\partial \hat D/\partial \operatorname{logit}\alpha_k$, are kept
in the $10 \times 10$ block and used only for the support rule; they are not gated separately.)

### 1.2 Why the lumped form is the one to gate

*Majorisation.* For every pixel the weights sum to one, so by Jensen's inequality
$\bigl(\sum_k w_{uk}\, \mathbf j_{uk}\theta_k\bigr)^2 \le \sum_k w_{uk}\, (\mathbf j_{uk}\theta_k)^2$. Summing over
pixels with the noise weights,

```math
\theta^\top H^{(i)} \theta\ \le\ \sum_k \theta_k^\top \Lambda_{ik}\, \theta_k \qquad \text{for all } \theta .
```

The lumped block-diagonal matrix majorises the true Hessian. A step $\Lambda^{-1}\eta$ with a majoriser
is a majorise-minimise step: it decreases the quadratic model and cannot overshoot. On partially covered
pixels the normalised weights still sum to one over the surfels present, so the bound holds wherever a
depth is rendered.

*Overshoot of the squared form.* $\Lambda^{\mathrm{sq}}$ is the diagonal of $H$ with the off-diagonal
blocks deleted; it is neither an upper nor a lower bound. Take the common mode, in which all surfels
that overlap in a region need the same offset $x$ and the residual is the constant $e$. The correct
shift is $x = e/a_i$. The block-wise step of surfel $k$ is

```math
x^{\mathrm{sq}}_k = \frac{\sum_u w_{uk}\, e}{a_i \sum_u w_{uk}^2} = \frac{e}{a_i} \cdot \frac{\sum_u w_{uk}}{\sum_u w_{uk}^2},
\qquad
x^{\mathrm{lump}}_k = \frac{\sum_u w_{uk}\, e}{a_i \sum_u w_{uk}} = \frac{e}{a_i} .
```

The squared form overshoots by the overlap factor $\sum_u w_{uk} / \sum_u w_{uk}^2 \ge 1$ (equal to $m$ when
$m$ surfels share each pixel equally), because every surfel behaves as if it alone had to absorb the
residual. The lumped step is exact for the common mode. A leave-out or gating step is exactly such a
common-mode correction: the removed views pulled a patch of surface as a whole.

*Price.* The lumped form overstates the curvature of differential modes (neighbouring surfels moving
against each other), so those modes converge more slowly, and the lumped covariance
$\Lambda_k^{-1}$ understates their uncertainty. For the quantity that is actually rendered the lumped
form is the better calibrated one in an idealised case: with $m$ equally weighted surfels per pixel and
footprints of $F$ pixels, $n_{ik} = F/m$, and the rendered variance $\sum_k w_{uk}^2\, \mathbf j \Lambda_k^{-1} \mathbf j^\top$
evaluates to $\sigma^2/F$ per view with lumped blocks, which is the variance of a local mean over the
footprint, and to $m\,\sigma^2/F$ with squared blocks. This is a back-of-envelope statement for equal
weights and offset only; it has not been checked against an exact refit. The remaining overconfidence
(within-view noise correlation, shared pose error, cross-surfel covariance) is untouched by the choice.

*Consequence for gating.* The gate multiplies $(\Lambda_{ik}, \eta_{ik})$ in lumped form. A gate applied
to squared blocks would move a surfel too far when a view is removed, and the excess would then be read
as evidence against the *remaining* views in the next round.

### 1.3 Gated fusion and its first-order behaviour

With gates $w_{ik} = 1 - \zeta_{ik} \in [0,1]$ and the prior centred at the linearisation point,

```math
\Lambda_k(w) = \Lambda_0 + \sum_i w_{ik}\, \Lambda_{ik}, \qquad
\eta_k(w) = \sum_i w_{ik}\, \eta_{ik}, \qquad
\delta_k(w) = \Lambda_k(w)^{-1} \eta_k(w)
```

on the nine geometric coordinates (opacity frozen: the $9\times 9$ block is the conditional one, as in
`posterior`). This is what the code already does with one scalar per view (`ViewInfo(vi.Lambda * s, ...)`);
the pair gate replaces the scalar by a vector over surfels. If the linearisation point is the fit with
all gates one, $\sum_i \eta_{ik} = 0$ there and

```math
\delta_k(w) = -\Bigl(\Lambda_k - \sum_i \zeta_{ik} \Lambda_{ik}\Bigr)^{-1} \sum_i \zeta_{ik}\, \eta_{ik} ,
```

which contains leave-view-out ($\zeta_{jk} = 1$ for one view), leave-block-out, and every soft
intermediate. An optimiser that has not converged leaves a non-zero total gradient; the first form,
which sums the retained messages, removes it as well and is the one to use (`cavity_gradient="others"`).

The sensitivity to one gate is

```math
\frac{\partial \delta_k}{\partial w_{ik}} = \Lambda_k(w)^{-1} \bigl(\eta_{ik} - \Lambda_{ik}\, \delta_k(w)\bigr),
\qquad
\frac{\partial \Lambda_k^{-1}}{\partial w_{ik}} = -\Lambda_k^{-1} \Lambda_{ik} \Lambda_k^{-1} .
```

The influence of a message is its information-weighted disagreement with the current estimate, scaled
by the posterior covariance. At the all-gates-one fit it reduces to $\Lambda_k^{-1}\eta_{ik}$. Two
remarks follow. First, the influence is large where a message disagrees *and* the surfel has little
other support, which is exactly the phantom-surfel regime of Section 3; there the linearisation is least
trustworthy. Second, the validity of the closed form is measured by the removed share
$\phi_k(w) = \operatorname{tr}\bigl(\sum_i \zeta_{ik}\Lambda_{ik}\bigr) / \operatorname{tr}(\Lambda_k - \Lambda_0)$,
the soft generalisation of $\phi_k^{(j)}$.

### 1.4 The cavity test of a message, without rendering

Let $\mathcal S$ be a set of views to exclude (a block). The cavity estimate and the message's own
estimate are compared in the $3$-dimensional sub-block. With $C_k^{-\mathcal S}$ the cavity covariance
and $\delta_k^{-\mathcal S}$ the cavity mean,

```math
\mathbf r_{ik} = \eta_{ik} - \Lambda_{ik}\, \delta_k^{-\mathcal S}, \qquad
T_{ik} = \mathbf r_{ik}^\top \bigl(\Lambda_{ik} + \Lambda_{ik}\, C_k^{-\mathcal S} \Lambda_{ik}\bigr)^{+} \mathbf r_{ik} ,
```

which is $\chi^2$ with $\operatorname{rank}(\Lambda_{ik}) \le 3$ degrees of freedom under the nominal
model. The pseudo-inverse form avoids inverting a message of deficient rank (fewer than three
non-collinear pixels). A signed scalar $z_{ik}$ for the offset alone is the first whitened component.
This is the surfel-space form of the cavity residual: it is the within-footprint whitened sum of the
pixel residuals, the same quantity that `residual_precision_apply` forms by Woodbury. What it does not
contain is in Section 3.

### 1.5 What the pair gate approximates

A pixel gate $w_i(u)$ induces the exact gated message

```math
\Lambda_{ik}[w_i] = c_{\mathrm{eff}} \sum_{u \in i} w_i(u)\, \frac{a_i^2\, w_{uk}}{\sigma_i^2(u)}\ \mathbf j_{uk}^\top \mathbf j_{uk},
\qquad
\eta_{ik}[w_i] = c_{\mathrm{eff}} \sum_{u \in i} w_i(u)\, \frac{a_i\, w_{uk}}{\sigma_i^2(u)}\ \mathbf j_{uk}^\top e_i(u) .
```

A pair gate $w_{ik}\Lambda_{ik}$ equals this if and only if $w_i(u)$ is constant over the footprint of
$k$ in view $i$ (up to pixels with negligible weight). Otherwise the pair gate changes the weight of the
plane fit but not its *shape*: a fault boundary through a footprint biases the tilt of the gated
message, where the pixel gate would remove exactly the faulty side. Computing the exact form is one
extra factor in the accumulation (`inv` multiplied by $w_i(u)$ per contribution) and costs nothing.

## 2. Pooling pixel scores to pairs and rendering pair values to pixels

For one view $i$, collect the normalised weights in $W_i \in \mathbb R^{U_i \times K}$,
$(W_i)_{uk} = w_{uk}$, and let $N_i = \operatorname{diag}(n_{ik})$, $n_{ik} = \sum_u w_{uk}$.

```math
\text{render:}\quad (\mathsf R_i g)(u) = \sum_k w_{uk}\, g_k, \qquad
\text{pool:}\quad (\mathsf P_i s)_k = \frac{1}{n_{ik}} \sum_u w_{uk}\, s(u),
\qquad \mathsf R_i = W_i,\quad \mathsf P_i = N_i^{-1} W_i^\top .
```

**Adjointness.** With the counting measure on pixels and the measure $n_{ik}$ on pairs,

```math
\langle \mathsf R_i g,\ s\rangle_{U} = \sum_u \sum_k w_{uk}\, g_k\, s(u) = \sum_k n_{ik}\, g_k\, (\mathsf P_i s)_k = \langle g,\ \mathsf P_i s\rangle_{N_i} .
```

Both operators are row-stochastic on covered pixels ($\mathsf R_i \mathbf 1 = \mathbf 1$,
$\mathsf P_i \mathbf 1 = \mathbf 1$): a constant score is preserved in both directions, and total mass
is preserved, $\sum_u (\mathsf R_i g)(u) = \sum_k n_{ik} g_k$. The pooling can be computed with a stock
rasteriser and no custom kernel: render a dummy per-surfel scalar $g$ and take the vector-Jacobian
product, $\partial \bigl(\sum_u s(u)\, (\mathsf R_i g)(u)\bigr) / \partial g_k = \sum_u w_{uk}\, s(u)$; a
second pass with $s = 1$ gives $n_{ik}$.

**What is lost.** The round trip is

```math
\mathsf R_i \mathsf P_i = W_i N_i^{-1} W_i^\top ,
```

a symmetric, doubly stochastic smoothing kernel of rank at most the number of visible surfels, with
eigenvalues in $[0,1]$. It is a projection only if footprints do not overlap; in general
$(\mathsf R_i\mathsf P_i)^2 \ne \mathsf R_i\mathsf P_i$, so each pool-render cycle blurs again. The
opposite composition $\mathsf P_i\mathsf R_i = N_i^{-1} W_i^\top W_i$ mixes overlapping surfels. Lost
specifically:

- every variation of the score within a footprint (the null space of $W_i^\top$), including the position
  of a fault boundary to better than the footprint size;
- by Jensen, energy: $\lVert \mathsf P_i s\rVert_{N_i}^2 \le \lVert s \rVert^2$, so a pooled magnitude
  understates a mixed footprint; a pair that is half faulty looks half as deviant, not half faulty;
- sign structure: a signed residual that changes sign inside a footprint pools to zero. Pool
  $\lvert s \rvert$ or $s^2$ alongside $s$ when this matters;
- pairs with small $n_{ik}$ (surfels mostly hidden behind others in view $i$) receive a pooled value
  dominated by a few low-weight pixels. Pairs with $n_{ik}$ below a floor are treated as unobserved.

The resolution loss is not uniform. A depth-supervised fit places large surfels on planar,
texture-free surface, and that is where the depth models fail; the footprint is widest where the
boundary of a fault most needs to be resolved. This is an expectation from how densification behaves,
not a measured fact for these scenes.

**Which pooling for which quantity.** Three variants differ in what they preserve.

1. Probability-like scores (a cue's fault probability, model-disagreement magnitude): the mean
   $\mathsf P_i$ above.
2. Log-likelihood ratios: evidence adds, so the pooled quantity is the deflated sum
   $\ell_{ik} = c_{\mathrm{eff}} \sum_u w_{uk}\, \ell_i(u) = c_{\mathrm{eff}}\, n_{ik}\, (\mathsf P_i \ell_i)_k$. Without
   the deflation a pair of 200 correlated pixels claims 200 independent observations.
3. Gates that will multiply a message: the information-weighted mean
   $w_{ik} = \operatorname{tr}\bigl(\Lambda_{ik}[w_i]\bigr) / \operatorname{tr}(\Lambda_{ik})$, the scalar that reproduces the
   trace of the exact gated message of Section 1.5. It equals the exact gating only for a footprint-constant gate.

Pooling uses the compositing weights of the *full* fitted surface, including surfels that only the
view under test supports. Section 3 shows why this matters.

## 3. The see-through problem

### 3.1 Statement

A regional fault that is too near, repeated by a block of frames, is not a bad message to an existing
surfel. Densification gives it its own surfels, $k^\ast$, in front of the true surface. Every message to
$k^\ast$ comes from the faulty block, the messages agree with each other, and $T_{ik^\ast}$ of Section 1.4
is small or undefined: with the block removed, $\Lambda_{k^\ast}$ falls to the prior and the surfel is
"unsupported". A test that only compares messages arriving at one surfel sees nothing. The conflict is
between $k^\ast$ and the surfels behind it on the same rays.

### 3.2 The pair-level evidence: cavity residual pooled with full-surface weights

The cavity render of the main document already resolves this in pixel space: surfels that the block created or
mostly supports are dropped, the ray continues to the surface the other views support, and the residual
$z_j(u)$ of the faulty view against that surface is large. The surfel-space counterpart is to pool that
pixel residual with the weights of the full surface:

```math
\bar z_{jk} = \bigl(\mathsf P_j^{\mathrm{full}} z_j\bigr)_k = \frac{1}{n_{jk}} \sum_{u \in j} w^{\mathrm{full}}_{uk}\, z_j(u),
\qquad
\ell_{jk} = c_{\mathrm{eff}} \sum_{u \in j} w^{\mathrm{full}}_{uk}\, \ell\bigl(z_j(u)\bigr).
```

The residual is measured against the cavity; the attribution uses the surface that view $j$ actually
informed. Two cases show that the attribution is the right one.

- *Fault too near.* In the full surface the phantom $k^\ast$ dominates pixel $u$ of view $j$, so the
  evidence lands on the pair $(j, k^\ast)$: the message that created the phantom.
- *Fault too far.* The true surfel $k$ is in front and opaque, the faulty depth lies behind it. In the
  full surface $k$ dominates the pixel, and the evidence lands on $(j, k)$: the message through which
  view $j$ drags the true surfel backwards.

In both cases the pair that receives the evidence is the pair through which the faulty measurement
entered the fit, which is the pair a gate must close. This quantity requires the pixel to be testable
($O_{-j}(u)$ above threshold). It is the cue (c1) expressed on pairs; nothing new is measured.

### 3.3 The surfel-level evidence: free-space score from the other views

A second quantity looks at surfel $k$ from the views that do *not* support it. Map the measurement of
view $j$ into metric depth along the ray, $\check d_j(u)$ (invert the per-view alignment and the distance
bias, both fitted without the supporters of $k$ where possible). For each fragment $(u, k)$ that the
rasteriser visits, with ray-surfel depth $d_{uk}$ and own alpha $a_{uk}$, define the standardised depth
difference and three soft outcomes:

```math
t_{juk} = \frac{\check d_j(u) - d_{uk}}{\sqrt{\sigma_j^2(u)/a_j^2 + \mathbf q_{uk}\, C_k\, \mathbf q_{uk}^\top}},
\qquad
\psi^{-} = \Phi(t - t_0)\ \text{(pierced)},\quad
\psi^{+} = \Phi(t + t_0) - \Phi(t - t_0)\ \text{(confirmed)},\quad
\psi^{0} = \Phi(-t - t_0)\ \text{(hidden)},
```

with $C_k$ the sub-block covariance of the surfel on $(\nu, \tau_1, \tau_2)$ and $t_0$ a margin of a few
units. Accumulate per pair

```math
F^{\bullet}_{jk} = \frac{\sum_{u \in j} a_{uk}\, \psi^{\bullet}(t_{juk})}{\sum_{u \in j} a_{uk}}, \qquad \bullet \in \{-, +, 0\} .
```

$F^-_{jk}$ is the fraction of the footprint of $k$ in view $j$ through which view $j$ claims to see.
The weight is the surfel's own alpha $a_{uk}$, not the transmittance-weighted $w_{uk}$, and this is
deliberate: a measurement that lies behind $k$ asserts free space along the whole ray up to the
measurement, whatever else is on the ray. An occluder in front of $k$ would produce a *nearer*
measurement, which counts as "hidden" and is uninformative about $k$. The outcome "hidden" must stay
neutral; counting it against $k$ would penalise every legitimately occluded surfel.

Aggregated over views outside the supporters' blocks, with one vote per independent block $b$ (not per
frame; Section 4),

```math
n^{-}_k = \sum_{b}\ \max_{j \in b} F^{-}_{jk}, \qquad n^{+}_k = \sum_{b}\ \max_{j \in b} F^{+}_{jk} .
```

The pair $(n^+_k, n^-_k)$ is a per-surfel count of independent confirmations and free-space violations.
It is distributed to pairs by support share, $\pi_{ik} = \mathbf g_k^\top \Lambda_{ik}\, \mathbf g_k / \mathbf g_k^\top (\Lambda_k - \Lambda_0)\, \mathbf g_k$
with $\mathbf g_k$ the first row of $G_k$ (the normal-offset information that `supported` uses): the
supporters of a pierced surfel share the blame in proportion to what they contributed.

*Computation.* The accumulation needs, per fragment, $a_{uk}$, $d_{uk}$ and a scatter-add into two
per-surfel buffers. The numpy reference backend exposes these arrays per ray. Whether the GPU rasteriser
exposes per-fragment depth and alpha, or needs a small custom accumulation, has not been verified. An
approximation that needs only standard rendering: render view $j$ with fragments restricted to
$d_{uk} < \check d_j(u) - \text{margin}$; the accumulated opacity $O^{<}_j(u)$ is the pixel-level
free-space violation, and its vector-Jacobian product with respect to a dummy per-surfel scalar
attributes it to surfels with transmittance weighting, which blames the front-most blocker most. One
pass per view either way.

### 3.4 Relation to visibility constraints in depth-map fusion

The construction is the classical free-space constraint. The following is recalled from the literature
and not re-checked against the sources. Volumetric range-image integration (Curless and Levoy) carves
the space between the sensor and each measurement as empty. Visibility-based depth-map fusion (Merrell
et al.) scores a depth hypothesis by the number of views it occludes and the number of free-space
violations it incurs, and keeps hypotheses for which these balance. Visibility-consistent surface
extraction on a Delaunay tetrahedralisation (Labatut et al.; Jancosek and Pollefeys for weakly
supported surfaces) puts the ray from camera to point into a graph-cut energy as an "outside" vote.
Occupancy grids update log-odds of free space along each ray. $F^-$, $F^+$, $F^0$ are the same three
events evaluated per surfel footprint and softened by the measurement and surfel uncertainty.

### 3.5 What the ray conflict can and cannot contribute

- *It is symmetric.* A pierce of $k$ by view $j$ says that the supporters of $k$ and view $j$ disagree.
  For a too-near fault the supporters are wrong; for a too-far fault (a depth model that reports the
  interior behind a glass facade, or a reflection) view $j$ is wrong and $k$ is the true surface. The
  score alone does not say which. The direction is settled only by counting independent blocks on each
  side, by the fault prior, or by cues that do not come from depth.
- *It is not an independent cue.* It is computed from the same depth maps as (c1) and compares them
  across views. Its errors are correlated with those of (c1) by construction. In a label model that
  assumes conditionally independent detectors, the cavity residual and the ray conflict must enter as
  *one* detector (for instance the sum of their log-likelihood ratios, or their maximum), not as two;
  counted twice, they would manufacture agreement and inflate the estimated accuracy of depth consensus.
- *What it adds.* Coverage and robustness, not information of a new kind. It scores surfels whose only
  supporters are in one block; it gives a pruning criterion for the fused surface ($n^-_k$ large,
  $n^+_k$ small from outside the supporters' blocks); and it does not depend on a linearised Newton
  step, so it is valid where $\phi_k$ is near one and the cavity is not.
- *False positives.* Thin structures, vegetation and other semi-transparent geometry, Gaussian tails of
  large surfels (threshold $a_{uk}$), grazing rays, moving objects, and above all a wrong alignment
  $(a_j, b_j)$ of the piercing view, which pierces or hides coherently over the whole view. The
  alignment must be fitted under the gates before the score is read (Section 6.4).
- *Blind where everything is blind.* If every view repeats the error, the surface is consistently
  wrong, no ray pierces it, and both quantities are zero.

## 4. Errors repeated by several views

### 4.1 How they appear

Let a block of frames repeat an error over a region. Depending on how the fit and the densification
resolved the conflict, the same data appear in one of two forms.

- *One surfel, multi-modal messages.* The surfel sits at the information-weighted mean of two groups of
  messages; the estimates $\mathbf m_{ik} = \Lambda_{ik}^{+}\eta_{ik}$ cluster at two offsets, and every
  message has a non-zero residual.
- *Two surfels on the ray, each with uni-modal support.* The see-through configuration of Section 3,
  possibly with semi-transparent surfels that share the pixel.

These are one event in two representations, and a detector must treat them alike. The natural unit is
therefore not the surfel but the *column*: the set of surfels that overlap along the rays of the views
that see them. Messages to any surfel of a column are votes for a depth along that column.

### 4.2 What a detector at the surfel can conclude

*That the messages are inconsistent.* The heterogeneity statistic

```math
Q_k = \sum_i \bigl(\mathbf m_{ik} - \bar{\mathbf m}_k\bigr)^\top \Lambda_{ik} \bigl(\mathbf m_{ik} - \bar{\mathbf m}_k\bigr)
```

has, under the nominal model with independent views, a $\chi^2$ distribution with
$\sum_i \operatorname{rank}\Lambda_{ik} - 3$ degrees of freedom. It tests that all views describe one plane.
Because frames of a block share errors even without a fault (similar viewpoint, similar depth-model
input), the correct unit is the block: a random-effects model

```math
\mathbf m_{ik} = \theta_k + \Delta_{b(i),k} + \boldsymbol\epsilon_{ik}, \qquad
\boldsymbol\epsilon_{ik} \sim \mathcal N\bigl(0, \Lambda_{ik}^{-1}\bigr),
```

with one shared offset $\Delta_{b,k}$ per (block, surfel), small for clean blocks and large for a faulty
one. Leave-block-out is the device that makes $\Delta_{b,k}$ visible: it removes the within-block
agreement that would otherwise count as confirmation.

*Which group is wrong, only under an asymmetry.* The groups are exchangeable under a relabelling of
"true surface", so identification needs one of:

1. a majority of *independent blocks*. The breakdown point is one half in units of blocks, not frames.
   Five frames of one strip that agree are one vote;
2. the prior that faults are rare, which decides a tie only weakly and is then self-fulfilling;
3. evidence that does not come from depth consensus: cross-model disagreement, equivariance failure,
   photometric evidence. This is where those cues are indispensable and not merely additive;
4. spatial context: the group that agrees with the consensus of neighbouring surfels. Weak, since a
   real depth discontinuity looks the same.

*What it cannot conclude.*

- With two groups of equal independent support (two strips, each internally consistent) the faulty
  group is not identifiable from depth. Typical aerial blocks have high forward overlap and modest side
  overlap, so many ground points are seen by several frames of one strip and by one or two strips; the
  number of independent blocks per surfel is then two or three, and a majority is fragile. The actual
  distribution of the number of independent blocks per surfel in the project's scenes has not been
  tabulated and should be, because it bounds what any consensus method can decide.
- With one group only, the messages are uni-modal and wrong, $Q_k$ is nominal, and nothing is detected.
- Multi-modality is not specific to faults. A surfel straddling a depth edge receives two legitimate
  groups; so does thin structure; a per-view alignment or pose error shifts all messages of that view
  coherently over the whole view (a view-level effect, to be removed before surfel-level tests);
  view-dependent depth on reflective surface spreads the messages continuously with viewing angle
  instead of clustering them.

## 5. A network over the pair structure

### 5.1 Structure

Nodes are pairs $(i,k)$ with $n_{ik}$ above a floor. With $K$ of order $10^5$ to $10^6$ surfels and
some tens of views each, there are $10^6$ to $10^7$ pairs. Two neighbourhood relations matter, plus the
column relation of Section 4:

- *surfel-surfel within a view*: pairs $(i,k)$, $(i,k')$ with $k'$ near $k$ on the surface. A fault is a
  connected region of a view, so this is where regional coherence is read;
- *view-view at a surfel*: pairs $(i,k)$, $(i',k)$, ordered by strip and frame index. This is where
  repetition along the strip and the count of independent blocks are read;
- *along the ray*: pairs $(i,k)$, $(i,l)$ with $k$, $l$ overlapping in view $i$ at different depths.

### 5.2 Inputs per pair

| Group | Features | Masking |
| --- | --- | --- |
| Evidence | signed standardised cavity discrepancy in offset and tilts (from $\mathbf r_{ik}$), $T_{ik}$, pooled pixel residual $\bar z_{ik}$ with its pooled magnitude, $F^{-}_{ik}$ of view $i$ against other surfels, and for the surfel $n^{\pm}_k$ from views outside $\mathcal B_i$ | blind, see 5.3 |
| Geometry | $\log n_{ik}$, range, incidence, footprint size in pixels, eigenvalues of the $3\times 3$ message, support share $\pi_{ik}$, opacity share, removed share $\phi_k$, number of independent supporting blocks, surfel scale and opacity, position in strip | none |
| Other cues, pooled by $\mathsf P_i$ | cross-model disagreement, equivariance instability, photometric score; each as mean and mean magnitude | none for the context; see 5.3 for the target |
| Appearance | frozen backbone features lifted to the pair and to the surfel (Section 8), feature disagreement $\Delta_{ik}$; depth-model confidence | none as context; $\Delta_{ik}$ is leave-block-out by construction |

All geometric quantities are expressed in the surfel's own frame or as scalars, so that the network is
invariant to the world frame.

### 5.3 The blind rule on pairs

The rule in pixel space: the prior at a pixel does not see the residuals within radius $R$ of that
pixel. On pairs the rule has to be stronger, for a reason specific to fusion. At the fit,
$\sum_i \eta_{ik} = 0$: the messages at a surfel sum to zero. Any feature of another view $i'$ at the same
surfel that was computed against a surface containing view $i$ is a linear function of the message of
view $i$, with opposite sign. A network with a view-view layer could reconstruct $\eta_{ik}$ exactly
from the other views' discrepancies and so see what it is forbidden to see. The rule is therefore:

> The gate of message $(i,k)$ may depend only on quantities computed with every message of the block
> $\mathcal B_i$ to the surfels within the blind radius of $k$ removed from the surface.

Concretely, with $\mathcal M_{ik} = \{(i',k') : i' \in \mathcal B_i,\ \lVert \pi_i(\boldsymbol\mu_{k'}) - \pi_i(\boldsymbol\mu_k) \rVert \le R\}$
the masked set ($\pi_i$ the projection into view $i$, $R$ the blind radius in pixels of view $i$, chosen
from the noise correlation length as in the main document):

1. evidence features of pairs in $\mathcal M_{ik}$ are not inputs to the gate of $(i,k)$;
2. evidence features of other views at $k$ are recomputed against the doubly-left-out cavity,
   $\mathcal S = \mathcal B_i \cup \mathcal B_{i'}$ in Section 1.4. This is the place where information form
   pays: an arbitrary leave-out set is a subtraction and a $3\times 3$ solve per surfel, so all
   block-by-block cavities of a surfel with $n_b$ supporting blocks cost $O(n_b^2)$ small solves;
3. evidence of view $i$ at surfels beyond $R$ (the ring) is allowed; it carries the regional coherence;
4. the pair's own evidence enters afterwards, through Bayes' rule with the pair's likelihood ratio, as
   in the main document, never through the network.

What item 2 legitimately tells the gate of $(i,k)$ is whether the other blocks agree among themselves
at $k$: whether the cavity that $(i,k)$ is judged against deserves trust. That is information about the
null, and it belongs with the geometry features of the scale head.

The same rule applies to the other cues when they are used as targets or as detectors: a detector's
vote on $(i,k)$ must not be an input to the network that predicts that vote.

### 5.4 Cross-fitting on pairs

Folds of frames alone do not prevent memorisation here. A surfel is scene-fixed and is seen from
several folds; its pooled appearance is nearly the same from each. A network trained on the pairs
$(i',k)$ of other folds can learn "surfel $k$ is flagged" and replay it for $(i,k)$, and by the
sum-to-zero relation the targets of the other folds at $k$ depend on the residual of view $i$. The unit
of cross-fitting in surfel space is therefore *ground area*: the scene is cut into tiles larger than
the receptive field plus $R$, tiles are assigned to $S$ folds, and the gates of pairs whose surfel lies
in a tile are produced by a network fitted on the other folds, with a margin between training and
application tiles. What such a network knows about glass in a tile it has learned from glass elsewhere.
Frame folds can be added on top ($F \times S$ networks) and are needed only if view-level inputs could
be memorised; no view identity is given as input.

### 5.5 Architecture that fits in 8 GB

A general graph network over $10^7$ pair nodes with three edge types is possible but awkward: the
blind ring needs geodesic exclusion on an irregular graph, and edge lists of several times $10^7$
entries do not fit comfortably. A factorised design uses the fact that the surfels visible in one view
form a sheet parametrised by that view's pixels:

1. *Within-view layers as image convolutions.* Render the pair features of view $i$ with
   $\mathsf R_i$, apply a small U-Net at half or quarter resolution with ring-masked kernels on the
   evidence channels (as in the main document), pool back with $\mathsf P_i$. Surface neighbourhood within a
   view is image neighbourhood, and the blind radius is exact in pixels. Along-ray relations are
   captured by rendering two layers (front-most and next) or by the free-space channels.
2. *Cross-view layer per surfel.* For each surfel, a small set network over its supporting views
   (masked attention or a 1D convolution along the strip index, at most 16 to 32 views, one token per
   view), with the block of the query view masked and with the doubly-left-out evidence of 5.3. Its
   output per pair is a context vector.
3. Alternate 1 and 2 two or three times; render the final context back to each view.

Memory: pair features at 32 channels in half precision are 64 bytes per pair, 0.64 GB for $10^7$ pairs,
kept on the CPU or memory-mapped. The U-Net sees one view (or one crop) at a time; the cross-view layer
processes surfels in chunks of $10^5$ (at 16 views and 64 channels, about 0.4 GB of activations per
layer in single precision). Training samples a ground tile, its surfels, and crops of at most 16 views
of that tile at $256^2$; this fits in 8 GB. The backbone that supplies appearance is frozen and run
once per image; its features are cached at reduced dimension, so no backbone activations are held
during training. Because frozen patch features already carry wide spatial context, the trainable part
can be much smaller than a U-Net (Section 8.3). These figures are estimates from tensor sizes, not
measurements.

The heads and their constraints carry over: a prior per pair or per pixel from the blind network, a
scale from geometry features only, the posterior by Bayes' rule. The training target is independent of
this structure; it can be the pseudo-likelihood of the contamination model or the combined posterior of
several detectors.

## 6. Use in reconstruction

### 6.1 Closed-form gated fusion against gated refit

*Closed form* ($\delta_k(w)$ of Section 1.3). One $9\times 9$ (effectively $3 \times 3$) solve per surfel;
differentiable in the gates; with lumped blocks a majorisation step on the linearised model. It is the
right tool for cavities, for the E-step, for previewing the effect of a gate, and for any objective that
needs the surface as a function of the gates.

It is not a reconstruction. It holds compositing weights, depth order, opacity, and the set of surfels
fixed. The case that matters most breaks all four: a fully gated phantom surfel has
$\Lambda_k(w) \to \Lambda_0$; it does not move, it must be *removed* (the support rule), and then the
surfels behind it become visible, their compositing weights jump, and pixels that no remaining surfel
covers become holes that only densification can fill. The validity gauge is $\phi_k(w)$: the closed
form is trustworthy where the gated-out share is small, and Section 3 says faults concentrate where it
is not.

*Gated refit.* A warm-started optimisation of the depth loss with weights, with pruning and
densification active. The loss is a sum over pixels, so what the refit consumes is a pixel weight map
$w_i(u)$. If the indicator lives on pairs, the map is the render $w_i = \mathsf R_i w_{i\cdot}$, and
there are two ways to use it:

- *frozen per round*: render once with the surface at the start of the round and hold the map fixed;
- *attached to surfels*: evaluate $\sum_k w_{uk} w_{ik}$ with the current weights during the fit.

The attached form is unsound. The optimiser can lower the loss by rearranging surfels so that gated
surfels cover high-residual pixels, even with a stop-gradient on the weight, and split or cloned
surfels need inheritance rules for their gates. The frozen pixel map has neither problem. The refit is
therefore a pixel-space operation whichever space the indicator is defined in.

*Recommendation.* Closed form inside rounds (cavities, E-step, nuisance updates), a gated refit with a
frozen pixel map between rounds and at the end, and pruning of surfels by the support rule and by the
free-space counts before the refit.

### 6.2 One pass against rounds

One pass (fit, gate, refit) suffices when the faults are a small share of the support of every affected
surfel, so that the cavities of clean views are themselves clean. Rounds are needed for masking and
swamping: a cavity contaminated by a second faulty block makes a clean view look deviant and a faulty
one look moderate; once the first block is gated the second becomes visible. Each round recomputes the
messages at the new surface (linearisation point, compositing weights, alignments). Because the E-step
uses cavities, the iteration does not increase a single objective; the damping
$w^{(t+1)} = (1-\eta)\,w^{(t)} + \eta\,(1-\zeta^{(t+1)})$ of the main document and a round limit are the practical
safeguards. With frozen pixel maps the damping is a pixel-wise average; with pair gates it needs a
correspondence between the surfels of consecutive rounds, which pruning and densification destroy.

### 6.3 Hard against soft

Soft gates use the clean part of a faulty view and are stable under small changes of $\zeta$. Their
costs: every inlier is down-weighted by its own $\zeta$, an efficiency loss of about the mean of
$\zeta$ over inliers; and $\Lambda_k(w)^{-1}$ is no longer the covariance of the estimate (a weighted
estimator has a sandwich covariance), so the rendered variance of later cavities is off by an unknown
factor. Hard gates restore an ordinary Gaussian model on the retained data and a valid variance, at the
cost of discontinuity and of losing data near the threshold. A defensible combination: soft during the
rounds for stability; at the end, snap confident gates ($\zeta < 0.1$ to weight one, $\zeta > 0.9$ to
zero), keep the soft value in between, and compute final variances from the retained set. Whole-view
removal is the special case of a view whose gated support is too small to fit its alignment. The
thresholds are placeholders to be fixed by held-out likelihood, not by a detection metric.

### 6.4 Nuisance parameters under the gates

The per-view alignment and the distance bias are global to a view or to the scene and couple all
surfels. Under gates the structure is that of bundle adjustment, with surfels as points and views as
cameras with two parameters each, plus one shared scalar.

*Per-view alignment.* With pixel gates,

```math
(a_i, b_i) = \arg\min_{a,b} \sum_{u \in i} w_i(u)\, \frac{\bigl(\hat d_i(u) - a\, \tilde D_i(u) - b\bigr)^2}{\sigma_i^2(u)} ,
```

a $2\times 2$ solve. The sufficient statistics decompose over pairs,

```math
S_{ik} = \sum_{u \in i} \frac{w_{uk}}{\sigma_i^2(u)} \begin{bmatrix} \tilde D_i(u)^2 & \tilde D_i(u) \\ \tilde D_i(u) & 1 \end{bmatrix},
\qquad
\mathbf t_{ik} = \sum_{u \in i} \frac{w_{uk}}{\sigma_i^2(u)} \begin{bmatrix} \tilde D_i(u) \\ 1 \end{bmatrix} \hat d_i(u),
```

so that with pair gates $\bigl(\sum_k w_{ik} S_{ik}\bigr) (a_i, b_i)^\top = \sum_k w_{ik}\, \mathbf t_{ik}$: a change of
gates re-solves the alignment without revisiting pixels.

*Distance bias.* Extend the per-pair moments to the regressors $[\tilde D, 1, \hat D^2]$ ($3\times 3$ plus
a 3-vector), sum them under the gates per view, eliminate $(a_i, b_i)$ by a Schur complement (this is
the projection of $\hat D^2$ against $[\hat D, 1]$, now under the gates), and sum the scalar complements
over views to obtain $\beta$ and its variance. Leave-block-out of $\beta$ remains a subtraction.

*Joint step.* The cross terms between surfel $k$ and the alignment of view $i$ are $3\times 2$ blocks
per pair. Eliminating the surfel blocks gives a reduced system of size $2N + 1$, which for some hundreds
of views is a trivial dense solve. A full gated Gauss-Newton step over surfels, alignments and bias thus
costs $O(\text{pairs})$. Alternating (surfels given nuisances, nuisances given surfels) converges to the
same point but slowly along the scale direction, where surface and alignments trade off.

*Cautions.* (i) The alignment used to form the evidence of pair $(i,k)$ is fitted on view $i$ itself,
including that pair; with two parameters against thousands of pixels the leak is of order
$1/n_{\mathrm{eff}}$ and is ignored, but it is a leak. (ii) The gated alignment has a breakdown point: a
fault over more than about half of the gated support of a view can capture $(a_i, b_i)$, after which the
*clean* part looks faulty. (iii) An alignment error mimics a view-wide fault in every cue of Sections 3
and 4, so the alignment and the gates must be alternated, as in the main document. (iv) Metric poses fix the
global scale; if poses carry scale uncertainty, the mean of $\log a_i$ is a gauge freedom shared with
the surface and must be fixed by convention.

## 7. What pixel space does better, and a recommendation

### 7.1 Pixel space is better at

- **Resolution.** A pixel gate resolves a fault boundary; a pair gate resolves it to the footprint,
  which is coarsest on flat, texture-free surface (Section 2).
- **Where the loss lives.** The depth loss, the alignment and the bias are sums over pixels. A pixel
  gate enters them directly; a pair gate must be rendered first (Section 6.1).
- **See-through.** The cavity render with the block's surfels dropped handles occlusion, compositing
  and phantom surfels by construction. In surfel space the same result needs the pooled cavity residual
  and the free-space score (Section 3), both of which call the renderer anyway.
- **Stable domain.** The pixel grid does not change between rounds. Pairs are created and destroyed by
  pruning and densification, so damping, hysteresis and convergence checks on pair gates need a
  correspondence that does not exist.
- **Native cues and features.** Cross-model disagreement, equivariance instability, depth-model
  confidence and image features are pixel fields; convolutional networks and pretrained backbones apply
  directly; the noise correlation that defines the blind radius is an image-space quantity.
- **Untestable pixels.** An image-only student extends the indicator where no other view sees the
  scene; there is no pair-space analogue without the image.
- **Independence from the surface approximation.** Pixel evidence is not tied to the block-diagonal
  information model; pair messages inherit its overconfidence.

### 7.2 Surfel space is better at

- **Association across views.** Two views meet at a surfel without correspondence search. Counting
  independent blocks, reading mode structure and repetition along the strip are natural there and
  awkward in image space.
- **Arbitrary leave-out sets in closed form**, which the blind rule across views requires (5.3) and
  which rendering one cavity per set cannot afford.
- **Fusion and nuisances as functions of the gates**, with their derivatives (1.3, 6.4).
- **Scene-fixed cross-fitting** by ground tiles (5.4).
- **Aggregation.** A pair pools $n_{ik}$ pixels, so its evidence has a higher signal-to-noise ratio
  and a roughly known effective sample size.

### 7.3 Recommendation: a specific hybrid

Define the indicator as a pixel field $\zeta_i(u)$ and use the surfel structure as the association and
fusion machinery. The pair gate is not a separate object but the footprint-constant special case of the
pixel gate, and the exact gated message for a pixel gate is available at the same cost (Section 1.5).

0. **Frozen features, light heads.** No backbone is trained. Appearance enters as cached frozen
   features lifted to pairs and surfels (Section 8); the trainable parts are the small heads.
1. **Evidence in pixel space.** Per-pixel cue scores, with the depth-consensus cue taken from the
   cavity render in which the block's surfels are dropped.
2. **Pool to pairs** with full-surface weights ($\mathsf P_i$ by vector-Jacobian product), in the three
   variants of Section 2. Add the free-space counts $F^{\pm}$, merged with the cavity residual into one
   depth-consensus detector.
3. **Cross-view reasoning on surfels**: block counts, doubly-left-out discrepancies, the per-surfel set
   layer of 5.5 under the blind rule of 5.3.
4. **Render the cross-view context back** with $\mathsf R_i$ as additional channels of view $i$, and
   let a pixel-resolution head (ring-masked on evidence, cross-fitted by ground tile) produce the prior;
   the posterior $\zeta_i(u)$ follows by Bayes' rule with the pixel's own likelihood ratio.
5. **Enter reconstruction through pixel-weighted lumped messages** $\Lambda_{ik}[w_i]$, $\eta_{ik}[w_i]$
   for closed-form fusion, cavities and nuisance statistics, and through the frozen pixel map
   $w_i(u) = 1 - \zeta_i(u)$ for the refit. Prune surfels by the support rule and the free-space counts.

This keeps what surfel space is uniquely good at and gives up the claim that the indicator *lives* on
pairs. The claim does not survive the derivations: every reconstruction step consumes either a pixel
weight or a message computed from pixel weights, and restricting the gate to pair resolution only
discards information.

### 7.4 What remains unresolved

- **Attribution with few independent blocks.** Every quantity in Sections 1 to 4 is depth consensus,
  evaluated at a surfel or along a ray. When two groups of equal independent support disagree, or when
  all views repeat an error, no arrangement of these quantities identifies the faulty side. The
  formulation relies on the non-depth cues exactly there, and their information content and error
  correlation with depth consensus are not yet measured.
- **Validity of the block-diagonal linearisation at phantom surfels.** Closed-form fusion cannot predict
  what becomes visible when a surfel is removed. The pooled cavity residual sidesteps this for the
  evidence; for the fused surface only a refit does. How often faults fall in the regime
  $\phi_k \to 1$ is unknown.
- **Calibration of pooled evidence.** The deflation $c_{\mathrm{eff}}$ is one number per scene; the
  effective sample size of a pair depends on footprint shape relative to the noise correlation length.
  Pair-level log-likelihood ratios are therefore rankings until calibrated.
- **Feature disagreement where it is needed.** The feature cue of Section 8.4 needs parallax larger
  than a backbone patch and features that vary over the surface. Glass, water and texture-free roofs
  offer little of the second. Whether the material signal alone carries the cue there is unmeasured.
- **Rasteriser support.** The free-space accumulation needs per-fragment alpha and depth; whether the
  GPU backend exposes them without a custom kernel has not been checked.
- **Unverified estimates in this note.** The variance comparison of lumped and squared blocks (Section
  1.2), the memory figures (5.5), the claim about surfel size on texture-free surface (Section 2), and
  the literature attributions (3.4, 8.1), and the parallax example and head sizes of Section 8 are
  reasoned or recalled, not tested.

## 8. Frozen features on surfels

The learned components are restricted to light heads on a frozen backbone; the backbone is never
fine-tuned. Labels are a variable: none, a few (for example exact-depth fault masks of a synthetic
scene, or a handful of annotated views), never enough to train a large model. This section says how
frozen image features reach the pair structure, what the heads look like, and what cross-view feature
disagreement is evidence of.

### 8.1 Lifting features through the compositing weights

Let $\mathbf f_i(p) \in \mathbb R^{D_0}$ be the frozen patch features of view $i$ (patch stride of the
order of 14 pixels for DINOv2-type backbones; recalled, not checked), reduced by a fixed linear map to
$D$ of 32 to 64 channels. The reduction is a PCA fitted without labels and then frozen. With $b_{up}$
the bilinear interpolation weights from patches to pixels, the pixel feature is
$\mathbf f_i(u) = \sum_p b_{up}\, \mathbf f_i(p)$.

*Per-(view, surfel) feature.* Apply the pooling operator of Section 2 channel-wise:

```math
\mathbf f_{ik} = \bigl(\mathsf P_i\, \mathbf f_i\bigr)_k = \frac{1}{n_{ik}} \sum_{u \in i} w_{uk}\, \mathbf f_i(u)
= \frac{1}{n_{ik}} \sum_p \Bigl(\sum_{u \in i} w_{uk}\, b_{up}\Bigr) \mathbf f_i(p) .
```

The second form shows that no upsampled feature map is needed: accumulate the surfel-to-patch weights
$\sum_u w_{uk} b_{up}$ once (each surfel touches a handful of patches) and apply them to the cached
patch features. Alternatively, the vector-Jacobian product of a rendered $D$-channel dummy gives the
same sums, in chunks of channels.

*Per-surfel feature fused across views.* Fusing the pair features with their pixel masses,

```math
\bar{\mathbf f}_k = \frac{\sum_i n_{ik}\, \mathbf f_{ik}}{\sum_i n_{ik}},
\qquad
\bar{\mathbf f}_k^{(-\mathcal S)} = \frac{\sum_{i \notin \mathcal S} n_{ik}\, \mathbf f_{ik}}{\sum_{i \notin \mathcal S} n_{ik}} .
```

This is feature-field distillation in closed form. The least-squares problem
$\min_{\{\mathbf f_k\}} \sum_i \sum_u \lVert \mathbf f_i(u) - \sum_k w_{uk}\, \mathbf f_k \rVert^2$ has a Hessian with the
same cross-surfel structure as in Section 1.1, and the same Jensen bound applies: lumping gives the
normal equations $\bigl(\sum_i n_{ik}\bigr)\mathbf f_k = \sum_i \sum_u w_{uk}\, \mathbf f_i(u)$, whose solution is
$\bar{\mathbf f}_k$, one majorisation step from zero. Methods that optimise a feature per Gaussian by
gradient descent reach a sharper solution of the same problem (recalled from the feature-splatting
literature, not checked); the closed form is adequate here because the feature is a context, not a
rendering target. Storing the sums $\sum_{i \in b} n_{ik} \mathbf f_{ik}$ and $\sum_{i \in b} n_{ik}$ per block
$b$ makes every leave-block-out version a subtraction, exactly as for the geometric messages. A second
moment $\sum_i n_{ik}\, \mathbf f_{ik}\mathbf f_{ik}^\top$ (or its diagonal) gives the cross-view dispersion.

*What the lift loses.* The backbone patch is usually larger than a surfel footprint, so neighbouring
surfels receive nearly the same feature: the lifted field has patch resolution, not surfel resolution.
Patch features also depend on viewpoint, scale and surrounding context, so $\mathbf f_{ik}$ varies across
views of a correctly placed surfel; this baseline dispersion has to be modelled before disagreement is
read as evidence (8.4). Pairs with small $n_{ik}$ inherit the feature of whatever occludes them.

### 8.2 Which frozen features, and for what

| Component | Frozen features | Why |
| --- | --- | --- |
| Fault prior of a pair or pixel | view-specific $\mathbf f_{ik}$ (or $\mathbf f_i(u)$) from the frozen encoder of the depth model that produced the depth map, where accessible; otherwise a general self-supervised backbone | a fault is a property of what *this* model saw in *this* view; the model's own encoder is the closest observable to its failure state. This is a conjecture to be tested against a general backbone |
| Surface context | fused $\bar{\mathbf f}_k^{(-\mathcal B_i)}$ from a general self-supervised backbone | material or semantic class of the surface ("water", "glazing"), denoised across views and independent of the block under test |
| Cross-view evidence | $\mathbf f_{ik}$ against $\bar{\mathbf f}_k^{(-\mathcal B_i)}$, general backbone | must be comparable across views; an encoder tuned for depth may suppress exactly the appearance differences that matter |
| Inlier scale | none | the scale head stays geometry-only, for the identifiability argument of the main document |
| Untestable pixels (student) | $\mathbf f_i(u)$ alone | no surfel context exists there |

Colour and features act as judges only. They enter the indicator and never the geometric fit.

### 8.3 The light head

With frozen features the within-view network of Section 5.5 shrinks, because the patch features already
encode wide spatial context. The trainable parts are:

1. *Appearance head.* Per pair, the input
   $[\,\mathbf f_{ik},\ \bar{\mathbf f}_k^{(-\mathcal B_i)},\ \Delta_{ik},\ \text{geometry scalars}\,]$ goes through a linear layer or
   a two-layer perceptron of width about 128, some $10^4$ to $10^5$ parameters, to a prior logit. At
   pixel resolution the same head takes $\mathbf f_i(u)$ and the rendered surfel context
   $\mathsf R_i\, \bar{\mathbf f}^{(-\mathcal B_i)}$.
2. *Evidence context.* One or two ring-masked convolution layers on the rendered evidence channels
   only (blind radius as before), a few thousand parameters.
3. *Cross-view layer.* One masked attention layer over the tokens of a surfel's supporting views
   (Section 5.3), or, as the zero-parameter fallback, the leave-block-out means and dispersions
   themselves.
4. *Scale head.* The low-dimensional geometry network or lookup of the main document.

The blind rule and the tile-wise cross-fitting are unchanged. A head this small cannot memorise
individual patches from appearance to the extent a U-Net can, which weakens the memorisation concern
but does not remove it for a perceptron on 100-dimensional inputs with millions of training pairs;
cross-fitting stays.

*Training as a function of the amount of labels.*

- *None.* The head is fitted to the label-free target: the contamination pseudo-likelihood, or the
  posterior of a label model over several detectors. Nothing anchors the direction of the latent class
  except the assumption that faults are the minority and that detectors are better than chance.
- *A few* (synthetic masks, a handful of annotated views). In order of how little is needed: (i) fix the
  label-swap and the class-conditional accuracies of the detectors; (ii) calibrate the intercept of the
  head and the decision threshold; (iii) add a supervised term for the head on labelled pairs to the
  unlabelled objective, with confident pseudo-labels on the rest. A linear probe on $D \approx 100$
  frozen dimensions is well-posed with some hundreds of labelled pairs from distinct regions; pairs
  from one region are not independent samples. Synthetic-to-real shift of the features is a risk that
  frozen self-supervised features reduce but do not remove; it is untested here.
- *Consistency terms available without labels.* The pixel head and the pair head should agree under
  pooling, $\mathsf P_i\, \zeta_i \approx \zeta_{i\cdot}$. Predictions for the same surfel from frames of
  one block may be tied together, since a block shares its errors; predictions across blocks must not
  be tied, because that would assume the answer of Section 4.

### 8.4 Cross-view feature disagreement as evidence

Define, for each pair, the disagreement between the pair's feature and the fused feature of the surfel
without the pair's block,

```math
\Delta_{ik} = 1 - \cos\bigl(\mathbf f_{ik},\ \bar{\mathbf f}_k^{(-\mathcal B_i)}\bigr),
\qquad
\tilde\Delta_{ik} = \frac{\Delta_{ik} - m(\mathbf x^g_{ik})}{s_\Delta(\mathbf x^g_{ik})} ,
```

where $m$ and $s_\Delta$ are the median and a robust spread of $\Delta$ as a function of geometry
(difference in viewing direction and in scale between view $i$ and the other supporters, $n_{ik}$),
estimated without labels from the scene, since most pairs are clean. The standardisation removes the
baseline dispersion of 8.1. A surfel-level version is the between-block dispersion
$\Delta_k = \sum_b n_{bk}\, \lVert \bar{\mathbf f}_{bk} - \bar{\mathbf f}_k \rVert^2 / \sum_b n_{bk}$.

It responds to three different things.

1. *Misregistration.* If surfel $k$ is at the wrong depth, its footprints in different views cover
   different points of the true scene, and the pooled features differ. This is photo-consistency in
   feature space, with the surfels providing the correspondence. It judges the *geometry of the surfel*,
   whoever supplied it, and so it can respond where depth consensus cannot: when all views repeat an
   error that is consistent in three dimensions, the surface is uni-modally wrong, no residual and no
   ray conflict exists, yet the features sampled through the wrong surface disagree between blocks.
   Requirements: the image displacement must exceed a fraction of the patch stride, and the features
   must vary over that displacement. The displacement between two views is about
   $f\,(b/d)\,(\delta d/d)$ pixels for focal length $f$, baseline-to-range ratio $b/d$ and relative depth
   error $\delta d/d$. As an illustration with assumed values, $f = 3000$ px, $b/d = 0.3$ and a 2% depth
   error give 18 px, of the order of one patch; adjacent frames of a strip with $b/d = 0.1$ give 6 px,
   below it. The cue therefore works between strips or distant frames and for large errors, and it is
   silent in the interior of a homogeneous region, whatever the depth error.
2. *View-dependent appearance.* Specular and transparent surfaces look different from different
   directions even when the surfel is correct. This is not a geometric inconsistency, but these are the
   materials on which depth models fail, so the response is a useful material indicator. It is a
   statement about the surface, not about any one view.
3. *Nuisances.* Occlusion boundaries, thin structures, pooling over few pixels, moving objects,
   illumination changes between strips, and pose error. These raise $\Delta$ without a depth fault.

*What kind of evidence it is.*

- *Attribution.* The surfel-level dispersion says that a surfel is suspect, not which message is wrong;
  like the ray conflict, it is symmetric between the groups. The pair-level $\tilde\Delta_{ik}$ singles
  out a view only when that view's feature is the outlier against several independent blocks.
- *Independence.* The feature cue is computed from images through a backbone that never saw the depth
  maps, so its noise is independent of the depth noise. Its *signal* in mode 1 depends on the fused
  surface, and hence on the depth maps: where depth consensus has already moved a surfel to the wrong
  place, both cues fire together. Conditional on the true fault state this is the dependence a label
  model wants (agreement because of the fault), but the nuisances of item 3 are partly shared with the
  cavity residual (occlusion boundaries, thin structure, pose), so conditional independence is an
  assumption to be checked, not a given. It is still the best candidate in surfel space for a detector
  whose errors differ from those of depth consensus.
- *Blind rule.* $\Delta_{ik}$ compares view $i$ with a fused feature that excludes its block, so it is
  admissible as context for the gate of $(i,k)$. If it is used as a detector whose vote is a training
  target, the same quantity must not also be an input of the head that predicts that vote.
- *Use.* As an input to the appearance head (material and risk context); as one detector in a label
  model, entered separately from the depth-consensus detector; and, in its surfel-level form, as a
  third criterion next to the support rule and the free-space counts when surfels are pruned.
