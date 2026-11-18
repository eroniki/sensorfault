# Detecting Depth-Prior Faults That Every View Repeats: a Semi-Supervised Indicator on Frozen Features

Supporting derivations are in `docs/learned-indicator/`: the latent-class theory (`theory.md`), the
cue definitions (`cues.md`), the surfel-space formulation (`surfel.md`), objections and prior work
(`objections-and-prior-work.md`), and the experiment design (`experiments.md`).

## Abstract

Depth-supervised Gaussian surfel reconstruction trusts one monocular depth map per image. These maps
fail regionally, and because neighbouring views see the same surface in the same way, the failure is
often repeated in most of the views that observe a region. Any gate built on agreement between depth
maps, whether a statistical test, a robust loss or a learned reweighting, is blind to such faults: a
bias shared by all redundant observations is undetectable from their residuals. We ask what
information, independent of depth consensus, reveals them. The indicator we propose is the posterior
of a latent fault class given three *groups* of evidence of different provenance: the leave-block-out
depth residual (multi-view depth), disagreement and instability of single-image depth networks, and
multi-view image consistency. A light head on a frozen foundation model supplies the prior; the cue
groups supply the likelihood. We show that with the dependence structure these cues actually have,
their accuracies are not identifiable from agreement patterns alone, and that a small labelled set
from a synthetic scene supplies exactly the missing quantities, after which only the prevalence needs
to be adapted on the unlabelled target scene. The indicator is a per-pixel field; the surfel map
provides cross-view association, closed-form leave-out, gated fusion and the statistics of nuisance
parameters. Whether the approach has anything to add is decided by one measurement, specified with
thresholds fixed in advance: how much each cue group knows about repeated faults beyond what the
surface type already tells.

## 1. Introduction

A static scene is observed by posed aerial images flown in strips. An off-the-shelf monocular model
gives one depth map per image, and a surface of 2D Gaussian surfels is fitted to the depth maps. Most
depth pixels are accurate up to a per-view scale and offset. A minority, grouped in regions that follow
appearance (water, glass, texture-free ground), are wrong by many standard deviations.

The established defence is consistency. A view is compared with the surface that the other views
support, and a large residual marks a fault. We call every method of this kind a *consensus gate*:
the leave-view-out test of geodesy and navigation, robust losses, residual-supervised masks, and
learned reweightings of the same residual all belong to it. A consensus gate has a blind spot that no
refinement removes. If the views that observe a region share the error, their residuals against each
other are small, and the error is fused into the surface. In reliability theory this is the statement
that a bias in the direction common to all redundant observations has no minimal detectable size. In
satellite navigation the same problem, a fault common to a whole constellation, is covered only by a
second, independent constellation.

Monocular depth is prone to exactly this. The model sees each image alone, and frames of a strip show
the same surface under nearly the same viewpoint and context, so a misreading tends to recur. Leaving
out a block of neighbouring frames widens what consensus can see to the length of the block and no
further.

**Claim.** Depth-prior faults that every nearby view repeats are invisible to any consensus gate.
Evidence from information families that are independent of depth consensus (the behaviour of
single-image depth networks, and colour across views) detects a useful share of them, and gating on
that evidence improves the reconstructed surface where a consensus gate and a robust loss do not.

The claim concerns a stratum of faults and a property of evidence families. It is falsifiable by a
measurement that needs no new method, and the paper stands or falls with it.

**Scope of the learning.** Self-supervised and semi-supervised learning on frozen backbones only.
Every learned component is a light head on frozen features, or a model with tens of parameters. A small
labelled set (fault masks of a synthetic scene with exact depth, or a few annotated views) may be used
together with a large unlabelled set. Fine-tuning of a backbone and fully supervised training of a
detector are excluded.

**Contributions.**

1. The repeated-fault stratum as an evaluation target, with a definition computable from exact depth,
   and the argument that consensus gates are blind on it.
2. A fault indicator as the posterior of a latent class given three evidence groups of different
   provenance, with a frozen-feature head as prior (Section 4).
3. An identifiability analysis for the dependence structure these cues have: what agreement patterns
   identify, what they cannot, and which quantities a few labels supply (Section 4.5).
4. A hybrid formulation in which the indicator is a pixel field and the surfel map carries
   association, leave-out, gated fusion and nuisance statistics, using lumped information blocks that
   cannot overshoot (Sections 3 and 4.8).
5. A decision protocol with thresholds fixed in advance, including the reference that the method must
   beat: a linear probe on the same frozen features at the same label budget (Section 5).

## 2. Related Work

**Fault detection by redundancy.** Baarda's data snooping tests each observation by its normalised
residual and defines reliability through minimal detectable biases; solution separation in receiver
autonomous integrity monitoring compares the full solution with subset solutions, one per fault
hypothesis. Leave-block-out is solution separation with blocks as fault modes. Both literatures state
the limit we start from: common-mode faults need independent redundancy.

**Label models and unsupervised ensembles.** Dawid and Skene fit annotator error rates by
expectation-maximisation. Parisi et al. (PNAS 2014) show that the off-diagonal covariance of
conditionally independent binary classifiers has rank one; Fu et al. (ICML 2020) give closed-form
triplet estimators; data programming (Ratner et al., NeurIPS 2016) trains an end model on the label
model's posterior. Jaffe et al. (AISTATS 2016) treat dependent classifiers, and Mazzetto et al.
(AISTATS 2021) aggregate dependent sources with a small labelled set. The estimators we use are
theirs. Our contribution is not an estimator but the analysis of the cue families of this problem and
the stratum on which they matter.

**Masks and uncertainty in radiance fields and splatting.** RobustNeRF trims the photometric loss.
NeRF On-the-go (CVPR 2024), SpotLessSplats and WildGaussians predict masks or uncertainty with small
heads on frozen features, supervised by the rendering residual. The head we use is of this kind. The
difference is the supervision: one residual against a model that the masked view helped to fit, in
their case; several evidence groups, one of them a leave-out residual, in ours.

**Reliability of monocular depth priors.** CDGS (2025) weights the depth loss by a hand-weighted sum of
single-image heuristics. Recent work restricts depth supervision by photometric masks, by rendered
self-consistency, or by cross-view alignment of monocular depth. These are single cues or fixed
combinations, and the consensus-based ones share the blind spot. We found no work that reports
detection of view-repeated depth errors as a separate stratum; the search was not exhaustive.

**Single-image uncertainty.** Variance under flips and augmentations and disagreement of ensembles are
established uncertainty estimates for monocular depth (Poggi et al., CVPR 2020), with moderate
correlation to error.

**Frozen features.** DINOv2 and DINOv3 features support dense tasks with linear heads; PatchCore-style
methods detect anomalies from distances of frozen patch features without training; feature fields lift
frozen features onto 3D primitives.

## 3. Background: Surface, Messages and Leave-Block-Out Residual

**Measurement model.** With $d_i(u)$ the true depth of pixel $u$ in view $i$,

```math
\hat d_i(u) = a_i\bigl(d_i(u) + \beta\, d_i(u)^2\bigr) + b_i + \varepsilon_i(u),
```

with per-view scale and offset $(a_i, b_i)$, a distance bias $\beta$ shared by all views, and noise of
nominal standard deviation $\sigma_i(u)$. A *fault* is a connected region of a view whose depth deviates
from this model by many $\sigma$. A whole-view affine error is absorbed by $(a_i, b_i)$ and is not a
fault.

**Messages.** The rendered depth of pixel $u$ is $\hat D_i(u) = \sum_k w_{uk}\, d_k$ with normalised
compositing weights $w_{uk}$ and per-surfel depth Jacobians $\mathbf j_{uk}$. With a pixel weight map
$\omega_i(u) \in [0, 1]$, view $i$ sends surfel $k$ the message

```math
\Lambda_{ik}[\omega_i] = \sum_u \omega_i(u)\, w_{uk}\, \frac{a_i^2\, \mathbf j_{uk}^\top \mathbf j_{uk}}{\sigma_i^2(u)}, \qquad
\boldsymbol\eta_{ik}[\omega_i] = \sum_u \omega_i(u)\, w_{uk}\, \frac{a_i\, \mathbf j_{uk}^\top e_i(u)}{\sigma_i^2(u)},
```

where $e_i$ is the depth residual of view $i$ at the fitted surface. The information uses $w_{uk}$ and
not $w_{uk}^2$. Since the weights of a pixel sum to one, Jensen's inequality gives
$\theta^\top H^{(i)} \theta \le \sum_k \theta_k^\top \Lambda_{ik} \theta_k$: the *lumped* block-diagonal
matrix majorises the Hessian of the view's loss, so a step $\Lambda^{-1}\boldsymbol\eta$ decreases the
quadratic model and cannot overshoot. Blocks built from $w_{uk}^2$ delete the cross-surfel terms and
overshoot a common displacement by the overlap factor $\sum_u w_{uk} / \sum_u w_{uk}^2$, which is the
displacement that removing a view produces.

**Fusion and leave-block-out.** The fused information is $\Lambda_k = \Lambda_0 + \sum_i \Lambda_{ik}$.
For a tested view $j$ with a block $\mathcal B_j$ of $B$ neighbouring frames of its strip,

```math
\Lambda_k^{(-j)} = \Lambda_k - \sum_{i \in \mathcal B_j} \Lambda_{ik}, \qquad
\boldsymbol\mu_k^{(-j)} = \boldsymbol\mu_k - \bigl(\Lambda_k^{(-j)}\bigr)^{-1} \sum_{i \in \mathcal B_j} \boldsymbol\eta_{ik} .
```

Surfels that the block mostly supported are dropped, and the result is rendered into view $j$: the
*cavity* depth $\hat D_{-j}$, its variance $V_{-j}$, and accumulated opacity $O_{-j}$. Pixels with low
$O_{-j}$ are *untestable* by depth consensus. The distance bias is downdated in the same way, and
$\tilde D_{-j} = \hat D_{-j} + \beta_{-j} \hat D_{-j}^2$.

**Cavity residual.** With $(a_j, b_j)$ fitted by iteratively reweighted least squares on the testable
pixels,

```math
r_j(u) = \hat d_j(u) - \bigl(a_j \tilde D_{-j}(u) + b_j\bigr), \qquad z_j(u) = \frac{r_j(u)}{\sqrt{v_j(u)}},
```

where $v_j$ collects measurement noise, cavity variance, and the uncertainty of the distance bias and
of the pose of view $j$.

## 4. Method

### 4.1 Overview

For a pixel $u$ of view $j$ let $f \in \{0, 1\}$ be the latent fault state. The indicator is its
posterior given frozen features $\boldsymbol\phi$ and the scores $x_1, x_2, x_3$ of three evidence
groups:

```math
\zeta_j(u) = \Pr\bigl(f = 1 \mid \boldsymbol\phi, x_1, x_2, x_3\bigr), \qquad
\operatorname{logit} \zeta = \underbrace{\operatorname{logit} \pi(\boldsymbol\phi)}_{\text{prior: head on frozen features}}
+ \sum_{g \in \mathcal O} \underbrace{\log \frac{g_g^{1}(x_g \mid s)}{g_g^{0}(x_g \mid s)}}_{\text{likelihood ratio of group } g} .
```

$\mathcal O$ is the set of groups that do not abstain at this pixel, $s = s(\boldsymbol\phi)$ is a
stratum (a cluster of the frozen features), and $g_g^{f}$ is the class-conditional density of group
$g$. The prior says where faults are likely from appearance; the groups say whether there is evidence
at this pixel. Combination is by Bayes' rule, not by a network.

### 4.2 Evidence Groups

A cue is a standardised per-pixel score with a known or learnable null and an explicit abstention
flag. Each cue has its *own* nuisance alignment, so that no two cues share an estimated scale or
offset. Cues of common provenance are merged into one group with an unrestricted joint
class-conditional distribution; independence is assumed between groups only.

**Group 1, multi-view depth (consensus).** The cavity residual $z_j(u)$ of Section 3, merged with a
free-space score: the weighted count of other views' rays that pass through a surfel in front of their
own measured depth, or stop in front of it. The free-space score covers surfels that a fault created
for itself, which no opposing message reaches. Both are depth consensus and form one detector.
Abstains on untestable pixels.

**Group 2, single-image networks.** Two channels.

- *Cross-model disagreement.* Several depth models run on the same image. For the primary model $m_0$,
  the residual against the median of the others after a per-pair, per-view alignment (in log depth for
  metric models, in inverse depth when one is affine-invariant):

  ```math
  x^{\mathrm{cm}}_j(u) = \frac{1}{s^{\mathrm{cm}}_j(u)}\ \operatorname{med}_{m' \ne m_0} \Bigl[ \tau\bigl(\hat d^{m_0}_j(u)\bigr) - \tau\bigl(g^\ast_{m_0 m' j} \cdot \hat d^{m'}_j(u)\bigr) \Bigr] .
  ```

- *Instability under input transformations.* One model, a set of transformations $T$ (flips, crop,
  rescale, photometric change) with pixel maps $\psi_T$; the dispersion of the re-aligned predictions,
  with a one-pixel tolerance at depth edges:

  ```math
  x^{\mathrm{in}}_j(u) = \frac{1}{s^{\mathrm{in}}_j(u)} \Bigl( \frac{1}{S_u} \sum_{T} \Delta^{T}_{\mathrm{tol}}(u)^2 \Bigr)^{1/2} .
  ```

  Transformation families that change clean predictions systematically (a rotation, for a model with a
  gravity prior) have no usable null and are discarded.

The two channels share the image and the network family and are silent together on a confident,
stable, model-independent error. They count as one group.

**Group 3, multi-view image.** Plane-sweep photo-consistency. For depth hypotheses $d$ along the ray,
the zero-mean normalised cross-correlation of a patch with its plane-induced reprojection into the
$\kappa$ best neighbouring images gives a profile $\bar\rho_j(u, d)$ with optimum $d^\star(u)$. The
score is the standardised difference between the aligned monocular depth and the optimum,

```math
x^{\mathrm{ph}}_j(u) = \frac{\log d^{\mathrm{al}}_j(u) - \log d^\star(u)}{\sqrt{\sigma^2_{\mathrm{ph}}(u) + \sigma_j^2(u) / d^{\mathrm{al}}_j(u)^2}}, \qquad
\sigma^2_{\mathrm{ph}}(u) \propto \Bigl( - \partial^2_{\log d}\, \bar\rho_j(u, d) \big|_{d^\star} \Bigr)^{-1} ,
```

with its own alignment by robust regression of $\hat d_j$ on $d^\star$. It uses images, poses and the
depth map of view $j$ only: no surfel map and no other depth map. It abstains where the peak is not
distinctive (texture-free surface, repeated texture, occlusion). A second channel replaces patch
correlation by cosine similarity of frozen features; it is coarser and narrows the abstention region.
This is the only multi-view evidence that can see an error common to all depth maps.

**What was considered and dropped.** The influence of a depth message on the colour loss of held-out
views, $\nabla_k \mathcal L_j^{\mathrm{rgb}\top} \Lambda_k^{-1} \boldsymbol\eta_{ik}$, is not a cue.
Because the messages of all views sum to zero at the fit, the influences sum to zero over views, so
the quantity is blind to errors that all views repeat; it contains the non-cavity depth residual, and
its sign is valid only within a pixel or two of parallax.

**Predicted behaviour** (F fires, M misses, A abstains, FA false alarm; predictions, not measurements):

| Fault class | Group 1 depth consensus | Group 2 single-image | Group 3 multi-view image |
| --- | --- | --- | --- |
| Specular, glass | F if the error changes with viewpoint beyond the block, else M | F if models differ or the prediction is unstable; M on a confident common error | A (weak peak) or FA (optimum at the virtual image) |
| Water | F if inconsistent across views; M within a block | partial F | A |
| Texture-free | M within a strip; F across strips | F: the best case | A |
| Thin structures, depth edges | FA | FA | A or FA |
| Repeated in neighbouring frames | M unless the block is longer than the repetition | as for a single view | F if textured and Lambertian |
| Repeated by all views | M | F unless common to the models | F if textured and Lambertian |

Two readings matter. No group is predicted to fire reliably on glass and water, and Group 3 abstains
exactly there, so on those surfaces two groups remain. And every group false-alarms at thin structures
and depth edges: a nuisance common to all groups, which is excluded by a band around rendered depth
discontinuities and reported as abstention.

### 4.3 Latent-Class Model

Let $y_g \in \{0, 1\}$ be a binarised group score, $p_g = \Pr(y_g = 1 \mid f = 1)$,
$q_g = \Pr(y_g = 1 \mid f = 0)$, $\delta_g = p_g - q_g$, and $\pi = \Pr(f = 1)$. If the groups are
conditionally independent given $f$ (assumption A1), then for $g \ne h$

```math
C_{gh} = \operatorname{Cov}(y_g, y_h) = \pi(1 - \pi)\, \delta_g \delta_h = u_g u_h :
```

the off-diagonal covariance has rank one. With three groups the loadings follow in closed form, and the
third central moment separates accuracy from prevalence:

```math
u_1^2 = \frac{C_{12}\, C_{13}}{C_{23}}, \qquad
\kappa = \frac{\mathbb E\bigl[\prod_g (y_g - \mu_g)\bigr]}{u_1 u_2 u_3} = \frac{1 - 2\pi}{\sqrt{\pi(1 - \pi)}}, \qquad
p_g = \mu_g + (1 - \pi)\, \delta_g, \quad q_g = \mu_g - \pi\, \delta_g .
```

Seven parameters from seven free cell probabilities: the model is exactly identified, up to an exchange
of the classes that is broken by rarity ($\pi < \tfrac12$) or by the known polarity of a cue. For
continuous scores the class-conditional densities follow without a shape assumption from a $2 \times 2$
unmixing of the conditional distributions of one group given the votes of the others.

### 4.4 Frozen Features: Prior, Strata and Head

Let $\boldsymbol\phi_j(u)$ be frozen patch features of the image (a self-supervised image encoder, and
separately the frozen encoder of the depth model, whose forward pass is run anyway), upsampled to
working resolution. They enter in three places.

- **Strata.** A clustering of $\boldsymbol\phi$ gives appearance strata $s$. Prevalence and group
  accuracies are per stratum, $\pi_s$, $p_{gs}$, $q_{gs}$. This removes the dependence between groups
  that is mediated by surface type, which is the dominant violation of A1.
- **Prior head.** $\pi(\boldsymbol\phi) = \operatorname{sigm}(\mathbf w^\top \boldsymbol\phi + w_0)$, or a
  two-layer head. It is trained on the posterior of the label model over all unlabelled pixels, and on
  labels where they exist. To prevent the head from reproducing the evidence it is trained towards, the
  target at a pixel is computed without that pixel's own scores inside a radius matched to the noise
  correlation, and the head for a ground tile is fitted on other tiles (cross-fitting by ground tile,
  not only by frame, since all views of a tile share its faults).
- **Untestable pixels.** Where all groups abstain, the indicator is the prior. This is the only
  information available there and it is appearance alone; such pixels weight the reconstruction but
  never decide about a view.

The head generalises over appearance and space. It cannot add information that the groups do not
contain.

### 4.5 Identifiability: What Agreement Identifies, and What Labels Supply

**The latent class has no anchor.** Without labels the model identifies the dominant binary cause
that renders the groups conditionally independent. Nothing in unlabelled data says that this is "depth
error above $3\sigma$ in a connected region". A surface type or the neighbourhood of a depth
discontinuity are rival causes.

**Dependence biases the estimate.** With class-conditional covariance $e_{gh}$,
$C_{gh} = u_g u_h + e_{gh}$, and the triplet estimate of the independent cue becomes
$\hat u_1^2 = u_1^2\, u_2 u_3 / (u_2 u_3 + e_{23})$. For rare faults $u_2 u_3 = \pi(1-\pi)\delta_2\delta_3$
is small: at 2% prevalence a false-alarm correlation of 0.1 between two cues produces as much covariance
as the whole fault signal. A dependent pair is credited with accuracy it does not have, because
agreement with each other is read as agreement with the fault.

**Two blocks are not enough.** By provenance, the four raw cues form two dependent blocks: multi-view
(cavity residual, photo-consistency: shared poses, occlusion and baselines) and single-image
(disagreement, instability: shared image and network family). With independence only across the two
blocks, the cross covariances $C_{12} = u_1 u_2$, $C_{13} = u_1 u_3$, $C_{42} = u_4 u_2$,
$C_{43} = u_4 u_3$ are invariant under $u_1, u_4 \to t\, u_1, t\, u_4$ and
$u_2, u_3 \to u_2/t, u_3/t$. Relative informativeness within a block and the product of block strengths
are identified; how informativeness divides between the blocks is a free parameter. Identification
needs a third independent group, or strata that remove the within-block dependence, or labels. The
design of Section 4.2 aims at three groups by giving each cue its own alignment and by separating depth
consensus from image consistency, but whether three conditionally independent groups exist after
stratification is empirical. Tetrad constraints with four cues can reject independence and localise it
to a pairing; they cannot confirm it.

**Semi-supervised likelihood.** With unlabelled pixels $\mathcal U$ and a small labelled set
$\mathcal L$ from scenes that are never evaluated,

```math
\mathcal J = - \sum_{n \in \mathcal U} \omega \log \sum_{f} \Pr(f \mid s_n) \prod_{g \in \mathcal O_n} g_g^{f}(x_{gn} \mid s_n)
\ -\ \lambda \sum_{n \in \mathcal L} \Bigl[ \log \Pr(f_n \mid s_n) + \sum_{g \in \mathcal O_n} \log g_g^{f_n}(x_{gn} \mid s_n) \Bigr] ,
```

with $\omega$ a deflation for spatial correlation. What the labels provide, in order of how few are
needed:

1. *The meaning of the class.* A handful of labelled fault regions ties the latent class to the fault
   definition and fixes the exchange of classes.
2. *Anchored accuracies.* $p_g$, $q_g$ are estimated directly. The binding number is the count of
   independent labelled fault regions, tens in one synthetic scene.
3. *Dependence.* The class-conditional covariances are observable on $\mathcal L$, which fixes the free
   parameter $t$ and selects the dependence structure by an ordinary test.
4. *Calibration.* A monotone recalibration of $\zeta$, and conformal scores from labelled clean pixels.
5. *A test of the label-free model.* Accuracies estimated from $\mathcal L$ and from $\mathcal U$ must
   agree.

**Division of labour.** What is hard to identify without labels (class meaning, dependence structure,
shapes of the class-conditional densities) is transported from the labelled domain. What the unlabelled
likelihood identifies given those is re-estimated on the target scene: the prevalence per stratum, for
which a single cue suffices, $\pi_s = (\mu_{gs} - q_{gs}) / \delta_{gs}$, and, where data allow,
main-effect accuracies. This is the label-shift setting, and it is where the method differs from a
classifier trained on the labelled scene: it adapts to the target without target labels.

| Labels | Role of the label model | Rests on |
| --- | --- | --- |
| none | defines the indicator from agreement patterns | independence between at least three groups; ablation only |
| few (tens of fault regions) | labels fix meaning, dependence and calibration; unlabelled data fix prevalence and adapt accuracies | transport of dependence from labelled to target domain |
| many | class-conditional joint estimated directly; the latent class is a regulariser and adaptation device | similarity of domains |

### 4.6 What Several Groups Buy on Repeated Faults

With Group 1 silent, the posterior exceeds one half only if the other groups carry it:

```math
\frac{p_2\, p_3}{q_2\, q_3} > \frac{1 - \pi}{\pi} \cdot \frac{p_1}{q_1}\Big|_{y_1 = 0} ,
```

which for $\pi = 0.05$ and a moderately sensitive consensus cue requires a combined likelihood ratio of
the order of 50 from Groups 2 and 3: two decent *and* independent detectors, or one strong detector
with a raised prior from the head. A fault on which every group is silent has a posterior below the
prior. The model moves the blind spot from "repeated by all views" to "missed by all groups"; it does
not remove it.

### 4.7 Relation to Simpler Models

With one group the model is a contamination mixture on the cavity residual, identified by the shape of
the inlier density alone; with a constant prior and scale its decision rule is a pixel-wise test with a
robustly fitted scale, and with a shifted-Gaussian fault class its region statistic is the generalised
likelihood ratio of a region scan. With three binary groups the optimal rule is one of eighteen
monotone voting rules, of which majority vote and the single best cue are two. The combination rule is
therefore not where the contribution lies, and the simplest rule that reaches the supervised ceiling on
the cue vector is the one to use: if that is a rank average, it is reported as such.

### 4.8 Use in Reconstruction

The indicator is a pixel field. A gate on a (view, surfel) pair is the special case of a pixel gate that
is constant over the surfel's footprint, and the exact gated message for a pixel gate costs the same,
so nothing is gained by restricting the gate to pairs. The surfel map does four things.

1. **Association.** Pixel scores are pooled onto pairs with the full-surface compositing weights, and
   cross-view context (counts of independent blocks, discrepancies between doubly-left-out cavities) is
   rendered back as channels. Pooling the cavity residual with full-surface weights also lands the
   evidence on the surfel through which a faulty depth entered, which handles surfels that a fault
   created.
2. **Gated fusion.** With $\omega_i(u) = 1 - \zeta_i(u)$ the messages $\Lambda_{ik}[\omega_i]$,
   $\boldsymbol\eta_{ik}[\omega_i]$ of Section 3 give the gated surface in closed form to first order,
   $\Lambda_k = \Lambda_0 + \sum_i \Lambda_{ik}[\omega_i]$, and all cavities under the gates.
3. **Refit.** The surface is refitted with the frozen weight map $\omega_i(u)$ in the depth loss.
   Closed-form fusion cannot predict what becomes visible when a surfel is removed; only a refit does.
4. **Nuisance parameters.** $(a_i, b_i)$ and the normal equations of $\beta$ are accumulated with the
   same weights, so no nuisance parameter is fitted to flagged pixels.

One pass (fit, evidence, indicator, refit) is the default. Further rounds are an ablation; they carry
no guarantee of monotone improvement, since the evidence of a view is computed on its cavity and not on
the full model. The indicator is held fixed while the surface is fitted.

**A caution on gating.** Water and glass are where the depth prior is most often wrong and also where
nothing else constrains the surface. Removing the prior there removes the surface. The gated fit
therefore reports gated regions as unconstrained rather than filling them from neighbours, and
reconstruction quality is evaluated with and without those regions.

### 4.9 Calibration

Under residual dependence the posterior is overconfident and is treated as a ranking. A threshold with
a stated false-alarm rate comes from split-conformal calibration on labelled clean pixels or regions of
the labelled set, with regions (not pixels) as the exchangeable unit, and with covariate weighting for
acquisition differences between labelled and target scenes.

## 5. Experiments

### 5.1 Evaluation Targets

**True-fault mask** (evaluation, or labels on scenes excluded from evaluation). After the best affine
alignment of the depth map to exact depth, a pixel is faulty if its relative error exceeds
$3\hat\sigma_j$ (robust scale) inside a connected region of at least 2% of the view, clean below
$2\hat\sigma_j$, and ignored otherwise and within three cells of a depth discontinuity.

**Repetition strata.** Each faulty pixel is back-projected to its ground point $X$; $\varrho(X)$ is the
share of views observing $X$ that are faulty there with the same sign. *Isolated*: $\varrho < 1/3$.
*Common-mode*: $\varrho \ge 2/3$. The common-mode stratum is where consensus is blind by construction
and where the method must earn its keep.

**Surface classes** from exact data and the image: water, ground, roof, facade, edge and thin,
texture-free.

### 5.2 The Deciding Measurement

On a synthetic aerial block with exact depth, with truth used for scoring only:

- (a) AUC and average precision of each cue against the fault mask, overall, per surface class and per
  repetition stratum, with abstention rates.
- (b) Class-conditional dependence of each pair of cues (log odds ratios given fault and given no
  fault), before and after stratification by frozen-feature cluster, with cluster-bootstrap intervals.
- (c) Gain $G$ of the label-free combination over the best single cue, against trivial combinations
  (maximum, rank average).
- (d) Headroom $H$: a cross-validated logistic regression on the cue vector with true labels, against
  the best single cue.

**Precondition.** Prevalence between 1% and 30% for at least one depth model, at least 20 distinct
fault regions in ground space, and at least 10% of faulty pixels in the common-mode stratum. Otherwise
the scene carries no information about the claim and the measurement moves to another block and to real
data.

**The approach is abandoned if any of:**

- K1. $H < 0.05$ in average precision overall and on the common-mode stratum.
- K2. On the common-mode stratum no cue outside Group 1 has AUC with lower bound above 0.60.
- K3. Fewer than three cue groups have overall AUC with lower bound above 0.60 at coverage of at least
  50%.
- K4. The fitted fault class predicts a surface class better than it predicts faults (its average
  precision against the water mask exceeds that against the fault mask by more than 0.10).
- K5. A linear probe on frozen features with a few labelled views is not improved by appending the cue
  scores, within surface strata.

**It is supported if all of:** at least three groups have AUC of 0.70 or more and one group outside
Group 1 reaches 0.70 on the common-mode stratum; some triple of cues has absolute log odds ratio at most
one in both classes; the label-free combination gains at least 0.05 in average precision and at least
half the headroom; and on the common-mode stratum the true-positive rate at 5% false positives is at
least 0.30 and at least twice that of Group 1. The numerical values are conventions fixed before the
measurement; they are stated so that the outcome cannot be reinterpreted.

The measurement needs cached depth maps of several models, including transformed inputs, the cavity
residual, a plane sweep and one frozen backbone: no label model and no head.

**Rendered data.** A rendered scene has perfect poses and synthetic materials. Images are degraded
(noise, blur, exposure change, pose perturbation) so that photo-consistency is not flattered, and the
verdict on (a) and (b) is repeated on real scenes with laser depth; where the two disagree, the real
scenes decide.

### 5.3 Follow-On Experiments

| | Question | Hypothesis |
| --- | --- | --- |
| F1 | Label-free against anchored label model | A few labelled regions remove every class exchange and halve the error of the estimated accuracies |
| F2 | Label budget and backbones | At five labelled views or fewer, the method exceeds a linear probe on the same frozen features by 0.05 average precision; appending cues to the probe helps at every budget |
| F3 | Violation of independence | Stratification by feature cluster reduces class-conditional dependence; grouping dependent cues removes the bias of the triplet estimate |
| F4 | Distilled head against label model | The head matches the label model on testable pixels and exceeds the prior of a labelled-only probe on untestable ones |
| F5 | Repeated faults (headline) | On the common-mode stratum the indicator detects faults that the cavity residual, a robust loss and view removal miss |
| F6 | Reconstruction | Gating improves held-out depth error and surface F-score on scenes with faults against no gate, the consensus gate, a Geman-McClure loss and the depth model's own confidence, and changes clean scenes by less than 2% |
| F7 | Synthetic to real | With labels from a synthetic scene only, target-scene adaptation of prevalence retains the gain on real aerial data with laser depth, where the labelled-only probe loses it |

Label budgets are 0, 1, 5 and 20 labelled views and one whole labelled scene. Frozen backbones compared:
a self-supervised image encoder at several sizes, a variant pretrained on overhead imagery where
available, and the frozen encoder of the depth model.

### 5.4 Baselines and Bounds

No gate; robust depth loss; removal of whole views; the consensus gate (cavity residual alone); the
depth model's own confidence; best single cue; maximum and rank average of cues; residual-supervised
masks on frozen features; a linear probe on frozen features at equal label budget; a variance head on
the frozen depth encoder. Bounds: supervised combination of the cues, and an oracle gate with true
masks. All methods receive the same poses, depth maps, alignment model, surfel initialisation and
schedule.

### 5.5 Protocol

Labelled scenes are disjoint from evaluation scenes. Evaluation masks, classes and strata are written
to a store that only the scoring code reads. Thresholds for binarisation, strata, exclusion bands and
the dependence structure are chosen on the labelled set or by held-out unlabelled likelihood, never by
a detection metric on evaluation scenes. Splits are by block or scene. Intervals are cluster-bootstrap
intervals over ground regions, since the effective sample is the number of fault regions and not of
pixels.

## 6. Limitations

- **Faults silent on every group remain invisible**, and nothing reveals their number.
- **No group is predicted to fire reliably on glass and water**, and image consistency abstains there.
  If single-image evidence on those surfaces is no better than surface type, a frozen-feature probe
  with a few labels is the best available method and the cue machinery adds nothing. The deciding
  measurement tests this.
- **Independence can be rejected, never confirmed.** The label-free model is an ablation for this
  reason.
- **Transported quantities are as good as the domain match.** Dependence structure and class meaning
  come from a labelled synthetic scene; depth-model behaviour, pose error and materials differ on real
  data.
- **Frozen backbones are trained mostly on ground-level images.** Their features may separate surface
  materials poorly in nadir views.
- **The effective sample is small.** Faults are rare and regional; per-scene estimation counts regions.
- **Gating removes constraint where it is scarce.** Detection and repair are different problems; this
  work addresses detection and reports gated regions as unconstrained.
- **Attribution with few independent blocks.** When two groups of views with equal independent support
  disagree at a surfel, depth alone says that they disagree and not which is wrong.

## 7. Notation

| Symbol | Meaning |
| --- | --- |
| $\hat d_i(u)$; $a_i, b_i, \beta$; $\sigma_i(u)$ | depth map; per-view scale and offset, shared distance bias; nominal noise |
| $w_{uk}$, $\mathbf j_{uk}$ | compositing weight and depth Jacobian of surfel $k$ at pixel $u$ |
| $\Lambda_{ik}[\omega]$, $\boldsymbol\eta_{ik}[\omega]$ | lumped message of view $i$ to surfel $k$ under pixel weights $\omega$ |
| $\mathcal B_j$, $B$ | block of frames left out with view $j$, its size |
| $\hat D_{-j}$, $V_{-j}$, $O_{-j}$ | cavity depth, its variance, accumulated opacity |
| $r_j$, $z_j$ | cavity residual, standardised |
| $f$, $\zeta_j(u)$ | latent fault state, indicator (posterior fault probability) |
| $\boldsymbol\phi$, $s$ | frozen features, appearance stratum |
| $x_g$, $y_g$ | score and binarised vote of evidence group $g$ |
| $\pi$, $p_g$, $q_g$, $\delta_g$ | prevalence, detection and false-alarm rates of group $g$, their difference |
| $C_{gh}$, $u_g$, $e_{gh}$ | covariance of votes, loading, class-conditional excess covariance |
| $\mathcal U$, $\mathcal L$, $\lambda$ | unlabelled and labelled sets, their trade-off |
| $\varrho(X)$ | share of observing views that repeat a fault at ground point $X$ |
| $G$, $H$ | gain of the label-free combination, headroom of the supervised combination |
