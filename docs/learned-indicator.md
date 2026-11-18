# A Learned Fault Indicator for Depth-Supervised Gaussian Surfel Reconstruction

## Abstract

Monocular depth models fail regionally: on glass, water and texture-free surfaces a connected part of a
depth map can be wrong by many standard deviations, and a reconstruction that trusts it is corrupted.
Leave-view-out consistency tests can find such regions, but their decision rule rests on a Gaussian noise
model with one scale per scene and on an exact surface model, and it fails when either is wrong. We keep
the leave-block-out prediction of a view from its neighbours as the evidence and replace the decision
rule by a learned indicator function. The indicator is the posterior of a per-pixel contamination model
whose outlier prior and inlier scale are produced by a network and fitted by maximum likelihood on the
residuals themselves, with no fault labels. Structural constraints make the model identifiable: the
prior at a pixel never sees the residuals in that pixel's neighbourhood, the inlier scale sees geometry
but not appearance, and the prior of a view is produced by a network fitted on other views. The fixed-scale, constant-prior special case is a classical pixel-wise test, so the
statistical gate is a limit of the model. A second, image-only network is distilled from the indicator
and supplies a fault prior where no other view observes the scene. The indicator weights the depth loss
of the surfel fit, which turns reconstruction into an expectation-maximisation loop whose E-step is
protected from self-confirmation by the leave-out. A conformal threshold gives a stated false-alarm rate.

## 1. Introduction

Depth-supervised Gaussian splatting fuses one depth map per image into a shared surface. The depth maps
come from an off-the-shelf monocular model, and their errors are not noise in the usual sense. Most
pixels are accurate up to a per-view scale and offset; a minority, grouped in regions that follow
appearance, are wrong by a large amount. Because neighbouring frames of an aerial strip see the same
ground from almost the same direction, the same region tends to be wrong in several frames at once.

A consistency test can expose such regions. View $j$ is removed from the fused surface together with a
block of its neighbours, the remaining surface is rendered into view $j$, and the depth map of view $j$
is compared with that render. The comparison is sound because the tested view has no influence on the
surface it is compared with. What is fragile is the step from the comparison to a decision. A test with
a nominal level needs the residual distribution under the no-fault hypothesis, and that distribution is
assumed: Gaussian, with a covariance shape calibrated offline and one free scale per scene. Three things
break this assumption in practice.

1. **Noise level.** A scale that is a few tens of percent off turns a test with thousands of degrees of
   freedom into a detector of the scale error.
2. **Noise shape.** Real depth-model residuals have heavier tails than a Gaussian and a spread that
   varies with range and incidence in ways the offline shape does not capture.
3. **Model error.** A surfel map cannot represent thin structures and depth discontinuities exactly. The
   leave-out prediction is then wrong where the data are right, and the test fires on clean views.

All three are properties of the *inlier* distribution that can be learned from the residuals of the
scene, because most pixels are inliers. This is the starting point of the paper: learn the null, and the
indicator follows as a posterior.

We restrict ourselves to self-supervised and unsupervised learning. No fault labels are used for training
or model selection; ground truth is used for evaluation only. The restriction is practical, since
labelled depth faults on real aerial data do not exist at scale, and it allows the model to be fitted on
the very scene it is applied to.

**Contributions.**

1. A contamination model of leave-block-out depth residuals whose outlier prior and inlier scale are
   amortised by a network and fitted without labels (Section 4.2 to 4.4).
2. Three structural constraints, a blind spot on the prior that is wider than the noise correlation, a
   geometry-only scale, and cross-fitting over frames, with an argument for why they prevent the
   degenerate solutions of an unsupervised mixture (Section 4.5).
3. The classical pixel-wise test and the region scan as limit cases of the model (Section 4.6).
4. Geometry-to-appearance distillation, which extends the indicator to pixels no other view observes
   (Section 4.8).
5. An expectation-maximisation form of gated reconstruction in which the E-step uses the leave-out
   posterior and the nuisance parameters are fitted with the same weights (Section 4.10).

## 2. Related Work

**Fault detection and data snooping.** Geodesy and navigation test single observations against the
solution computed without them: Baarda's data snooping, the detection-identification-adaptation
procedure of Teunissen, and solution separation in receiver autonomous integrity monitoring. These tests
give minimum detectable biases and stated error rates under a Gaussian model. Our evidence is of this
kind; our decision rule is not.

**Robust estimation with contamination models.** Huber's $\epsilon$-contamination model, mixture
likelihoods with a uniform outlier component in bundle adjustment and stereo, and iteratively reweighted
least squares are the classical unsupervised treatment of outliers. They use one global outlier fraction
and one scale. We amortise both over pixels with a network.

**Transient and distractor masks in radiance fields.** RobustNeRF trims the photometric loss;
NeRF On-the-go and SpotLessSplats predict per-pixel uncertainty or masks from image features, supervised
by the training residual; WildGaussians learns an uncertainty from DINO features. These methods are
label-free like ours, but the residual they learn from is computed against a model that the masked view
itself helped to fit, and they do not control a false-alarm rate.

**Uncertainty in Gaussian splatting.** FisherRF uses Fisher information for view selection; Bayes' Rays
derives a post-hoc spatial uncertainty. We use per-surfel information blocks to subtract a view exactly
and to render the variance of the leave-out prediction.

**Self-supervised denoising.** Blind-spot networks (Noise2Void, Noise2Self) predict a pixel from its
context and never from itself, so they cannot learn the identity. Our prior head uses the same device
for a different purpose: it may not look at the residual it is asked to judge.

**Conformal prediction.** Split-conformal calibration gives distribution-free marginal error control for
any score function and is our route to a stated false-alarm rate.

## 3. Background: Shared Surface and Leave-Block-Out Residual

**Measurement model.** A static scene is observed by $N$ posed images with one monocular depth map
$\hat d_i$ each. With $d_i(u)$ the true depth of pixel $u$ in view $i$,

```math
\hat d_i(u) = a_i\bigl(d_i(u) + \beta\, d_i(u)^2\bigr) + b_i + \varepsilon_i(u),
```

where $(a_i, b_i)$ are a per-view scale and offset, $\beta$ is a distance bias shared by all views, and
$\varepsilon_i(u)$ has nominal standard deviation $\sigma_i(u)$, a function of range, texture and incidence from an offline calibration. A *fault* is a connected region of a view whose depth
deviates from this model by many $\sigma$. A whole-view affine error is absorbed by $(a_i, b_i)$ and is
not a fault.

**Surface.** The surface is a set of 2D Gaussian surfels fitted to the depth maps. Surfel $k$ carries an
information block $\Lambda_k$ over its geometric perturbation, accumulated at the fitted surface as

```math
\Lambda_k = \Lambda_0 + \sum_i \Lambda_k^{(i)}, \qquad
\Lambda_k^{(i)} = \sum_u \frac{a_i^2\, \mathbf g_{uk}^\top \mathbf g_{uk}}{\sigma_i^2(u)},
```

with $\mathbf g_{uk}$ the Jacobian of the rendered depth of pixel $u$ with respect to surfel $k$.

**Distance bias.** The bias $\beta$ is a single scalar, estimated jointly with the surface by least
squares over all views; it is not absorbed into the surfel geometry. Its normal equations are a sum of
per-view terms, in which the regressor $\hat D_i^2$ is first projected against $[\hat D_i, 1]$ so that
$\beta$ is not confounded with the per-view scale and offset. We write
$\tilde D = \hat D + \beta \hat D^2$ for a rendered depth mapped into measurement space.

**Leave-block-out.** For a tested view $j$, let $\mathcal B_j$ be a block of $B$ neighbouring frames of
the same strip that contains $j$. The surface without the block follows from one subtraction and one
Newton step,

```math
\Lambda_k^{(-j)} = \Lambda_k - \sum_{i \in \mathcal B_j} \Lambda_k^{(i)}, \qquad
\boldsymbol\mu_k^{(-j)} = \boldsymbol\mu_k + \bigl(\Lambda_k^{(-j)}\bigr)^{-1} \sum_{i \in \mathcal B_j} \nabla_k \mathcal L_i ,
```

which is exact for a linear Gaussian model. The distance bias is downdated in the same way: the block's
terms are subtracted from its normal equations, giving $\beta_{-j}$ and its variance
$\operatorname{Var}(\beta_{-j})$, so the tested view does not inform the bias it is corrected with.
Surfels that the block created or mostly supported are dropped. Rendering the result into view $j$ gives the *cavity* depth $\hat D_{-j}(u)$, its variance
$V_{-j}(u)$ and the accumulated opacity $O_{-j}(u)$. Pixels with $O_{-j}(u)$ below a threshold are
*untestable*: no view outside the block observes them. Let $\Omega_j$ be the testable pixels.

**Accuracy of the one-step cavity.** The surfel parameters enter the rendered depth nonlinearly, so the
single step is an approximation whose error grows with the share of a surfel's information that the
block removes,

```math
\phi_k^{(j)} = \frac{\operatorname{tr}\bigl(\sum_{i \in \mathcal B_j} \Lambda_k^{(i)}\bigr)}{\operatorname{tr}\bigl(\Lambda_k - \Lambda_0\bigr)} \in [0, 1] .
```

For large blocks and surfels seen at grazing incidence, $\phi_k^{(j)}$ approaches one, the step is long,
and both the cavity depth and its variance can be off. We handle this in three ways. Surfels with
$\phi_k^{(j)}$ above the support threshold are dropped, as above. For the remaining surfels touched by
the block, the step can be replaced by up to $K$ damped Gauss-Newton iterations on the loss of the views
outside the block, with all other surfels held fixed; $K = 1$ is the plain step. And the rendered
removed share $\Phi_{-j}(u) = \sum_k w_{uk}\, \phi_k^{(j)}$ is passed to the learned model as a geometric
feature, so that the residual error of the approximation is learned as part of the null rather than
mistaken for a fault. Section 5 tests the approximation against exact refits.

**Residual.** The cavity depth in measurement space is
$\tilde D_{-j} = \hat D_{-j} + \beta_{-j} \hat D_{-j}^2$. The alignment $(a_j, b_j)$ is fitted on
$\Omega_j$ by iteratively reweighted least squares,

```math
(a_j, b_j) = \arg\min_{a, b} \sum_{u \in \Omega_j} \rho_j(u)\, \frac{\bigl(\hat d_j(u) - a\, \tilde D_{-j}(u) - b\bigr)^2}{v_j(u)} ,
```

with robustness weights $\rho_j(u)$. Plain least squares ($\rho_j = 1$) would let a large regional
fault pull the alignment towards itself and bias the residual on the clean remainder of the view. The
weights are therefore Huber weights in a first pass and $\rho_j(u) = 1 - \zeta_j(u)$ once the indicator
of Section 4.4 is available; alignment and indicator are alternated a few times per view, each
alignment being a closed-form $2 \times 2$ solve. The residual and its nominal variance are

```math
r_j(u) = \hat d_j(u) - \bigl(a_j \tilde D_{-j}(u) + b_j\bigr), \qquad
z_j(u) = \frac{r_j(u)}{\sqrt{v_j(u)}} ,
```

```math
v_j(u) = \sigma_j^2(u) + a_j^2 \bigl(1 + 2\beta_{-j} \hat D_{-j}(u)\bigr)^2 V_{-j}(u) + a_j^2\, \hat D_{-j}(u)^4 \operatorname{Var}(\beta_{-j}) + \mathbf j_q(u)^\top \Sigma_q\, \mathbf j_q(u) .
```

The four terms are the measurement noise, the uncertainty of the cavity surface, the uncertainty of the
distance bias, and the uncertainty of the pose of view $j$, with $\mathbf j_q(u)$ the derivative of the
rendered depth with respect to the pose and $\Sigma_q$ the pose covariance obtained without the depth
of view $j$.

Under the nominal model and without a fault, $z_j(u)$ has zero mean and unit variance. The statistical
gate tests this hypothesis. The cavity is kept; the remainder of the paper replaces the test.

## 4. Method

### 4.1 Overview

For every tested view the cavity yields a stack of per-pixel features. A network maps the stack to two
fields: a contamination prior $c_j(u)$ and an inlier scale $s_j(u)$. Together with a fixed outlier
density they define a two-component mixture for the standardised residual. The network is fitted by
maximising the mixture likelihood of all residuals of a scene, or of several scenes. The indicator is
the posterior probability of the outlier component.

### 4.2 Features

The feature stack of view $j$ is split by what it may inform.

| Group | Channels | Symbol |
| --- | --- | --- |
| Residual | $z_j$, and $\lvert z_j\rvert$ | $\mathbf x^{z}_j$ |
| Geometry | $\sqrt{V_{-j}}$, $\sigma_j$, $O_{-j}$, removed information share $\Phi_{-j}$, range, incidence angle, number of supporting views | $\mathbf x^{g}_j$ |
| Appearance | RGB of view $j$ or frozen image features; depth-model confidence where available | $\mathbf x^{a}_j$ |

### 4.3 Contamination Model

Let $f_j(u) \in \{0, 1\}$ be the latent fault indicator of pixel $u$ in view $j$. We model

```math
\begin{aligned}
f_j(u) &\sim \mathrm{Bernoulli}\bigl(c_j(u)\bigr),\\
z_j(u) \mid f_j(u) = 0 &\sim \mathcal N\bigl(0,\ s_j(u)^2\bigr),\\
z_j(u) \mid f_j(u) = 1 &\sim p_{\mathrm{out}}, \qquad p_{\mathrm{out}}(z) = \frac{1}{2Z}\ \text{ for } \lvert z\rvert \le Z ,
\end{aligned}
```

The outlier density is fixed, has no parameters and is bounded: it states only that a faulty depth can
be anywhere in a wide range. A residual beyond the range, $\lvert z \rvert > Z$, is a fault by
definition; it is assigned $\zeta = 1$ and left out of the likelihood, so no probability mass piles up
at the bounds. $Z$ is chosen well above $s_{\max}$. The two fields come from two heads,

```math
c_j = \operatorname{sigm}\Bigl( G_\theta\bigl[\mathbf x^{z}_j;\ \mathbf x^{g}_j,\ \mathbf x^{a}_j\bigr] \Bigr), \qquad
\frac{\partial c_j(u)}{\partial \mathbf x^{z}_j(u')} = 0 \ \text{ for } \lVert u' - u \rVert \le R ,
```

```math
s_j(u) = s_{\min} + (s_{\max} - s_{\min}) \operatorname{sigm}\Bigl( h_\psi\bigl(\mathbf x^{g}_j\bigr)(u) \Bigr),
```

where $G_\theta$ is a convolutional operator on whole fields whose kernels on the residual channels are
masked, so that its output at $u$ does not depend on the residuals inside a disc of radius $R$ around
$u$: the receptive field on the residual channels is a ring. The prior at a pixel is thus computed from
the residuals of its surroundings, never from its own. The geometry and appearance channels are not
masked. The scale head $h_\psi$ receives geometry only.

**Blind-spot radius.** Depth residuals are spatially correlated, with the calibrated correlation
$C(\lVert u - u' \rVert)$ of length scale $\ell$. A one-pixel blind spot is not enough: the adjacent
pixels would reveal the noise at $u$ through this correlation, and the prior could again follow the
pixel's own residual. $R$ is therefore set to the distance at which the calibrated correlation falls
below a small value, $C(R) \le 0.1$. It must stay well below the extent of the faults of interest, or
the ring no longer overlaps the faulty region; with a minimum fault area of 2% of the view this leaves a
wide admissible range, and $R$ is ablated.

The mixture density of one residual is

```math
p\bigl(z_j(u)\bigr) = \bigl(1 - c_j(u)\bigr)\, \mathcal N\bigl(z_j(u);\ 0,\ s_j(u)^2\bigr) + c_j(u)\, p_{\mathrm{out}}\bigl(z_j(u)\bigr) .
```

Because the prior at $u$ depends on residuals outside the disc, this is a conditional density of
$z_j(u)$ given its surroundings, and the product over pixels is a pseudo-likelihood, not a joint
density of the residual field. The inlier component ignores the conditioning, which is justified by the
choice of $R$: beyond the blind spot the noise correlation is small, so the conditional distribution of
an inlier is close to its marginal. A fault, in contrast, is coherent over distances larger than $R$,
and the conditioning is informative exactly for it.

### 4.4 Indicator and Objective

The learned indicator is the posterior of the outlier component,

```math
\zeta_j(u) = \Pr\bigl(f_j(u) = 1 \mid z_j(u)\bigr)
= \frac{c_j(u)\, p_{\mathrm{out}}\bigl(z_j(u)\bigr)}{\bigl(1 - c_j(u)\bigr)\, \mathcal N\bigl(z_j(u); 0, s_j(u)^2\bigr) + c_j(u)\, p_{\mathrm{out}}\bigl(z_j(u)\bigr)} .
```

The prior carries the context (appearance, geometry, neighbouring residuals); the likelihood ratio
carries the pixel's own evidence. They are combined by Bayes' rule and not by the network, which is what
keeps the indicator interpretable.

The parameters are fitted by minimising the negative log pseudo-likelihood with two regularisers,

```math
\mathcal J(\theta, \psi) = - \sum_{j} \sum_{u \in \Omega_j} \omega\, \log p\bigl(z_j(u)\bigr)
\ -\ \lambda_c \sum_j \log \mathrm{Beta}\bigl(\bar c_j;\ \alpha_0, \beta_0\bigr)
\ +\ \lambda_s \sum_j \bigl\lVert \nabla \log s_j \bigr\rVert_1 ,
```

with $\bar c_j$ the mean prior of view $j$. The Beta term states that faults are rare
($\alpha_0 < \beta_0$) and that no view is expected to be entirely faulty; the last term keeps the scale
field smooth. Residuals are spatially correlated, so the pixel-wise sum overcounts the evidence; the
weight $\omega = n_{\mathrm{eff}} / n$ deflates it, with $n_{\mathrm{eff}}$ the effective sample size
under the calibrated spatial correlation.

The gradient of the likelihood term has the expectation-maximisation form: with $\zeta$ held fixed,

```math
- \nabla \log p(z) = - (1 - \zeta)\, \nabla \log\bigl[(1 - c)\, \mathcal N(z; 0, s^2)\bigr] - \zeta\, \nabla \log c ,
```

so the scale is fitted on pixels weighted by $1 - \zeta$, and the prior is regressed onto $\zeta$. Plain
stochastic gradient descent on $\mathcal J$ performs both steps at once.

The model can be fitted in two regimes. *Amortised*: one network fitted on many scenes and applied to
new ones without further training. *Per scene*: the network, or only its last layers, fitted on the
residuals of the scene under test. Since no labels are involved, the second regime is legitimate at test
time.

### 4.5 Identifiability

A two-component mixture with free, pixel-wise parameters has two degenerate solutions, and fitting it
on the scene under test adds a third. The model is built to exclude all three.

**Collapse to noise.** The scale grows wherever residuals are large, and every fault is explained as an
inlier with a large variance. This is excluded because the scale head sees only geometry. A fault caused
by appearance (glass, water) at a given range and incidence shares its geometric features with the
far more numerous clean pixels at the same range and incidence, so the scale that maximises the
likelihood there is the clean one. The bound $s_{\max}$ and the smoothness term close the remaining gap.
What the scale *can* learn is the part of the null that depends on geometry: a larger spread at long
range or grazing incidence, and the model error of the surfel map near depth discontinuities and on
poorly supported surface, both of which are visible in $\sqrt{V_{-j}}$, $O_{-j}$ and the view count.

**Collapse to outliers.** The prior approaches one wherever it helps, and the outlier component absorbs
ordinary noise. Two things oppose this. First, the outlier density is fixed at $1/(2Z)$, which is lower
than the inlier density for every moderate residual, so calling an inlier an outlier costs likelihood.
Second, without the blind spot the prior could copy the pixel's own residual, and the mixture would
degenerate to the pointwise maximum of the two densities; a blind spot narrower than the noise
correlation length would allow the same through the neighbours. With a blind spot of radius $R$, the
prior must predict
whether a pixel is faulty from its surroundings, which is possible exactly for the faults of interest:
regional, and correlated with appearance.

**Memorisation in per-scene fitting.** The blind spot constrains the input, not the training signal.
When the network is fitted on the scene it is applied to, a prior head with enough capacity can learn
to recognise individual image patches and reproduce, from appearance alone, the posterior it was
trained towards at those very pixels. The prior then again follows the pixel's own residual. We exclude
this by cross-fitting. The views are split into $F$ folds of contiguous frames, each longer than the
block size $B$; the prior of a view is produced by a network fitted on the other folds only. What such
a network knows about a glass roof in view $j$, it has learned from the residuals of other views of
glass, which is legitimate evidence. Capacity is further limited by early stopping on the
pseudo-likelihood of the held-out fold. The scale head has a low-dimensional input and is fitted on all
views.

**Decision boundary.** The indicator exceeds one half when the outlier term exceeds the inlier term,
that is, when

```math
\lvert z \rvert > z^\ast(c, s) = s\, \sqrt{\, 2 \log \frac{(1 - c)\, 2Z}{c\, s \sqrt{2\pi}} \,} .
```

A boundary exists while the argument of the logarithm exceeds one, $(1 - c)\, 2Z > c\, s \sqrt{2\pi}$; for a
prior beyond that, every residual is flagged, which the Beta term and $Z \gg s_{\max}$ keep out of reach
in practice. The boundary is a threshold on the standardised residual that the context moves: where the prior is
low, a pixel needs a large residual to be flagged; inside a region whose neighbours are deviant and
whose appearance is suspicious, the threshold drops. For $c = 0.01$, $s = 1$ and $Z = 30$ it is
$z^\ast \approx 3.9$; for $c = 0.5$ it is $z^\ast \approx 2.5$.

**What is not identifiable.** An error that every view of a region repeats produces no residual and is
invisible to any cross-view signal. Leave-block-out removes repetition within a block of $B$ frames and
no further. This limit is shared with the statistical gate.

### 4.6 Classical Tests as Limit Cases

**Pixel-wise test.** With a constant prior $c$ and a constant scale $s = \gamma$, the rule
$\zeta > \tfrac12$ is $\lvert z \rvert / \gamma > z^\ast(c, \gamma)/\gamma$: a two-sided test of each
standardised residual with a scene temperature $\gamma$ fitted robustly, since the fit weights pixels by
$1 - \zeta$. The fitted $c$ plays the role of the test level.

**Region scan.** Replace the uniform outlier density by a shifted inlier,
$z_j(u) \mid f_j(u) = 1 \sim \mathcal N(\delta_{\mathcal R}, s^2)$, with one unknown shift per region
$\mathcal R$. The generalised log likelihood ratio of the region is then

```math
\max_{\delta}\ \sum_{u \in \mathcal R} \log \frac{\mathcal N\bigl(z_j(u); \delta, s^2\bigr)}{\mathcal N\bigl(z_j(u); 0, s^2\bigr)}
= \frac{1}{2}\, \frac{\bigl(\sum_{u \in \mathcal R} z_j(u)\bigr)^2}{\lvert \mathcal R \rvert\, s^2}
= \tfrac12\, z_{\mathcal R}^2 ,
```

the squared region statistic of the statistical gate for uncorrelated residuals. The learned model
generalises this in three directions: the regions are not fixed segments but emerge from the prior
field, the scale varies over the image, and the alternative is not restricted to a constant shift. The
shifted-Gaussian outlier component is kept as a variant in the ablations.

### 4.7 View Score

Decisions about whole views use a pooled score. Let $M_j = \{u \in \Omega_j : \zeta_j(u) > \tfrac12\}$
and let $\mathrm{cc}(M_j)$ be its connected components. The score is the expected faulty area of the
largest component, relative to the testable area,

```math
S_j = \max_{\mathcal R \in \mathrm{cc}(M_j)}\ \frac{1}{\lvert \Omega_j \rvert} \sum_{u \in \mathcal R} \zeta_j(u) .
```

The score is computed on testable pixels only. The output of the student of Section 4.8 on untestable
pixels weights the reconstruction but does not enter the decision about a view, since it carries no
geometric evidence.

### 4.8 Geometry-to-Appearance Distillation

The indicator exists only on testable pixels. An image-only student $t_\phi$ extends it. It is trained
to predict the indicator from appearance alone,

```math
\mathcal J_{\mathrm{dist}}(\phi) = \sum_j \sum_{u \in \Omega_j} \bigl\lvert 2\zeta_j(u) - 1 \bigr\rvert\ \mathrm{BCE}\Bigl( t_\phi\bigl(\mathbf x^{a}_j\bigr)(u),\ \operatorname{sg}\bigl[\zeta_j(u)\bigr] \Bigr),
```

where $\operatorname{sg}$ stops the gradient and the weight discounts undecided pixels. The supervision
is produced by multi-view geometry, so the student is self-supervised: it learns which appearances make
this depth model fail on this kind of scene. On an untestable pixel, where no residual exists, the
indicator is set to the student's output, $\zeta_j(u) = t_\phi(\mathbf x^a_j)(u)$. The student is not fed
back into the prior head, which would let an early mistake confirm itself.

### 4.9 Threshold with a Stated False-Alarm Rate

The score $S_j$ is turned into a decision by a threshold $\hat t$, obtained in one of two ways.

**Split-conformal.** Given scores $S_{(1)} \le \dots \le S_{(n)}$ of calibration views from scenes taken
to be fault-free, set $\hat t = S_{(\lceil (n+1)(1-\alpha) \rceil)}$. If a clean test view is
exchangeable with the calibration views, then $\Pr(S > \hat t) \le \alpha$. Calibrating the maximum of
$S_j$ over a scene gives a scene-level rate in the same way. The assumption of fault-free calibration
scenes is at scene level and involves no pixel labels; violations make the threshold conservative.
Views of one scene are not independent: neighbouring frames share ground, surface and noise. The
calibration unit is therefore a scene or a strip, with one score per unit (its maximum or a randomly
drawn view), and $n$ counts units, not views.

**Sensitivity to shift.** The conformal guarantee is marginal over the calibration distribution. A test
flight at another altitude, overlap or scene type is not exchangeable with the calibration scenes, and
the realised rate can then differ from $\alpha$ in either direction. Three measures limit this. The
score is a relative area on standardised residuals, so its clean distribution depends on acquisition
only through what the learned null fails to absorb. Calibration can be weighted for covariate shift
(weighted conformal prediction) with acquisition covariates: ground sampling distance, number of
supporting views, block size, and testable fraction. And the model null below is specific to the test
view, so a large gap between the two thresholds is itself a warning of shift. Cross-dataset transfer of
the threshold is tested directly (Section 5.2).

**Model null.** Sample residual fields from the fitted inlier model, $z \sim \mathcal N(0, s_j^2)$ with
the calibrated spatial correlation, pass them through the indicator and the pooling, and take the
$(1-\alpha)$ quantile of the resulting scores. This needs no calibration scenes and is specific to the
view, but it is only as good as the fitted null.

### 4.10 Use in Reconstruction

Under the contamination model, the expected complete-data negative log-likelihood of the surface, given
the indicators, is a weighted depth loss:

```math
\mathcal L_{\mathrm{depth}} = \sum_i \sum_u \bigl(1 - \zeta_i(u)\bigr)\, \frac{\bigl(\hat d_i(u) - a_i \tilde D_i(u) - b_i\bigr)^2}{2\, \sigma_i(u)^2} ,
```

and the information that view $i$ contributes to surfel $k$ is weighted alike,

```math
\Lambda_k^{(i)} = \sum_u \bigl(1 - \zeta_i(u)\bigr)\, \frac{a_i^2\, \mathbf g_{uk}^\top \mathbf g_{uk}}{\sigma_i^2(u)} .
```

The same weights enter the closed-form updates of $(a_i, b_i)$ and the normal equations of $\beta$, so
no nuisance parameter is fitted to faulty pixels. Reconstruction therefore has the form of expectation-maximisation
on the joint model, with two deliberate deviations.

*The E-step uses the cavity.* The posterior of view $j$ is computed against the surface without its
block, not against the full surface. A view cannot lower its own fault posterior by having shaped the
surface it is compared with. This is the cavity construction of expectation propagation applied to an
outlier model. The price is that the iteration is no longer guaranteed to increase a single likelihood;
we monitor the held-out pseudo-likelihood over rounds and damp the weights,
$w^{(t+1)} = (1 - \eta)\, w^{(t)} + \eta\, (1 - \zeta^{(t+1)})$.

*The learned scale stays out of the surface fit.* The scale $s$ is fitted to the standardised cavity
residual, whose variance mixes measurement noise with the error of the cavity itself: linearisation,
unrepresented structure, pose. Only the first belongs in the weight of a depth observation, and the
residual alone cannot separate the two. Folding $s$ into $\Lambda_k$ would also feed back into
$V_{-j}$ and hence into the next round's $z$, so that $s$ would chase its own effect. The surface fit
therefore keeps the nominal $\sigma$, and $s$ acts only in the decision. Weighting by
$1/(s^2 \sigma^2)$ is an ablation.

Two variants are studied. *Soft*: the weights $1 - \zeta$ are used as above, so the clean parts of a
faulty view still contribute. *Hard*: the view with the largest score above $\hat t$ is removed, one
view per round. In both, the indicator is recomputed between rounds and held fixed while the surface is
fitted, so the surface fit cannot reduce its loss by raising $\zeta$.

**Algorithm.**

1. Fit the surfel surface, $(a_i, b_i)$ and $\beta$ to all depth maps with weights $w_i(u) = 1$.
2. Accumulate the information blocks $\Lambda_k$ and the normal equations of $\beta$.
3. For each view $j$: subtract the block $\mathcal B_j$ (surfels and $\beta$), take up to $K$ steps, render
   the cavity, align robustly, form $z_j$ and the feature stack.
4. Fit $(\theta, \psi)$ by minimising $\mathcal J$ with cross-fitting over folds of frames (per-scene
   regime), or apply the amortised network.
5. Compute $\zeta_j$ on testable pixels, alternating with the alignment of view $j$; take the student's
   output on untestable pixels; compute $S_j$.
6. Soft: move $w_j(u)$ towards $1 - \zeta_j(u)$ with damping $\eta$. Hard: remove the view with the largest $S_j > \hat t$.
7. Refit the surface, warm-started. Repeat from step 2 until the weights change by less than a tolerance
   or a round limit is reached.

### 4.11 Architecture and Cost

The prior head is a small U-Net over the concatenated channels, with the blind spot of radius $R$
implemented by masked (ring-shaped) convolutions on the residual channels, or by masking discs of the
input and restricting the loss to their centres. The appearance channels can be frozen self-supervised image features. The scale head is a
shallow network on the geometry channels; because its input is low-dimensional it can also be a
monotone lookup in range, incidence and $\sqrt{V_{-j}}$. One forward pass per tested view is negligible
next to the $B$ backward passes that build the cavity.

## 5. Experiments

### 5.1 Data

| Data | Training input | Evaluation only |
| --- | --- | --- |
| MatrixCity aerial blocks with off-the-shelf depth models (Depth Pro, Metric3D v2, UniDepthV2, Depth Anything V2) | Cavity feature stacks of all views | Exact depth, giving true error maps |
| MatrixCity with injected faults: constant and range-dependent bias, multipath surrogate, transients | The same stacks | Injected masks; sweep over area fraction and magnitude |
| Stanford bunny and synthetic relief scenes | Thin-structure and edge-of-coverage regime | Exact masks |
| UseGeo, ETH3D | Per-scene fitting | Laser depth |

A pixel counts as faulty in evaluation when its aligned error exceeds $3\sigma$ inside a connected
region of at least 2% of the view. Evaluation masks are stored apart from the training data. Splits are
by block or scene, never by view, because neighbouring frames repeat depth errors.

### 5.2 Hypotheses

- **E1 Recovery.** On injected faults, the indicator recovers the withheld masks and matches the
  statistical gate's view detection rate at a matched false-alarm rate for faults of $2\sigma$ and more.
- **E2 Robust null.** With the noise level misspecified by 30%, the realised false-alarm rate stays
  within $[0.5\alpha, 1.5\alpha]$ with no scene temperature, where the fixed-scale test rejects every
  view.
- **E3 Model error.** On scenes with thin structures, the scale head raises $s$ near discontinuities and
  the false alarms of the statistical gate on clean views disappear, at no loss of detection elsewhere.
- **E4 Real errors.** On real depth-model errors, the indicator exceeds the statistical gate and the
  depth model's own confidence in pixel average precision. This is the decisive experiment: if it fails,
  the method is reported as a negative result.
- **E5 Untestable pixels.** In low-overlap scenes, the distilled student detects faults on untestable
  pixels above chance and above depth-model confidence.
- **E6 Reconstruction.** Soft weighting improves held-out AbsRel and F-score over hard removal, over the
  statistical gate and over robust-loss baselines on scenes with faults, and changes clean scenes by less
  than 2%.
- **E7 Fitting regime.** Per-scene fitting matches or exceeds the amortised network on scenes unlike the
  training scenes (synthetic to real).
- **E8 Cavity fidelity.** Against exact refits without the block, on a subset of views including thin
  structures and grazing surfels and for every block size, the one-step cavity depth error stays below
  the nominal noise and the rendered variance $V_{-j}$ matches the squared error within a factor of two.
  Where it does not, $K \in \{2, 3\}$ steps restore it; the smallest sufficient $K$ is used throughout.
- **E9 Robust alignment.** With a fault of $20\sigma$ over 25% of a view, the reweighted alignment keeps
  the standardised residual on the clean remainder at zero mean and unit variance, where plain least
  squares does not.
- **E10 Threshold transfer.** A conformal threshold calibrated on synthetic scenes keeps the realised
  false-alarm rate on clean views of real scenes, and across flight altitudes and overlaps, within
  $[0.5\alpha, 1.5\alpha]$; the weighted variant and the model null are compared where it does not.

### 5.3 Baselines

The statistical gate (energy statistic and region scan with a Holm step-down); aligned RMS per view; the
depth model's confidence; a global contamination model with one prior and one scale per scene; robust
depth losses (Huber, Geman-McClure); RobustNeRF; SpotLessSplats with depth supervision; no gate; and an
oracle gate with true masks as an upper bound. All methods receive the same posed cameras, the same
depth maps, the same per-view alignment and distance-bias model, and the same surfel initialisation,
densification and iteration schedule, so that differences are due to the treatment of faults and not to
the training recipe.

### 5.4 Ablations

No blind spot on the prior head; appearance given to the scale head; no leave-out (residual against the
full surface); no rendered variance among the features; shifted-Gaussian outlier component; a density
model of residual patches (normalising flow) in place of the mixture; a student trained on the
statistical gate's flags in place of the indicator; cross-depth-model disagreement as an additional
input; conformal threshold against model null, with and without covariate weighting; block size
$B \in \{1, 3, 5, 9\}$; blind-spot radius $R$ from one pixel to twice the correlation length; number of
cavity steps $K \in \{1, 2, 3\}$; plain against reweighted alignment; cross-fitting against fitting on all
views; learned scale folded into the surface weights.

### 5.5 Metrics

Pixel average precision and F1 against the evaluation masks. View detection rate at 5% per-view false
alarm. Realised false-alarm rate on clean views against the target. Held-out AbsRel and F-score of the
fused surface. Calibration of $\zeta$ as a probability (reliability diagram, expected calibration
error). Transfer across depth models and from synthetic to real scenes.

### 5.6 Protocol

Hyperparameters ($Z$, $R$, $F$, $\eta$, $s_{\min}$, $s_{\max}$, $\alpha_0$, $\beta_0$, $\lambda_c$, $\lambda_s$,
architecture) are selected by held-out pseudo-likelihood on residuals of views not used for fitting,
never by a detection metric. The selection rule is fixed before the evaluation masks are read.

## 6. Limitations

- Errors repeated by every view of a region are undetectable; the block size bounds what leave-out can
  separate.
- The pseudo-likelihood treats correlated residuals as independent up to a global deflation, so $\zeta$
  is a ranking that needs calibration before it is read as a probability.
- Conformal control is marginal over scenes and requires exchangeability between calibration and test
  scenes; covariate weighting corrects only for shift in the covariates it is given.
- The cavity is a linearisation. Where a block removes most of a surfel's support, the affected pixels
  are dropped as untestable rather than tested against an unreliable prediction, which lowers coverage
  for large blocks.
- The identifiability argument assumes that faults are a minority among pixels that share geometric
  features. A fault class that is tied to geometry (for example, every pixel beyond a certain range) is
  absorbed by the scale head and reported as noise, not as a fault.
- The reweighted alignment has a breakdown point: when faults cover more than about half of the testable
  pixels of a view, the alignment can lock onto the fault. Such a view is reported as untestable as a
  whole when the two alignments, from the clean and from the flagged pixels, are both self-consistent.
- Untestable pixels lie at image borders, behind occlusions and at the ends of strips, so the student is
  applied where its training distribution is thinnest.
- The student learns only the failure classes that occur on testable pixels of the fitted scenes.

## 7. Notation

| Symbol | Meaning |
| --- | --- |
| $\hat d_i(u)$ | depth map of view $i$ at pixel $u$ |
| $a_i, b_i, \beta$ | per-view scale and offset, shared distance bias |
| $\tilde D$ | rendered depth in measurement space, $\hat D + \beta \hat D^2$ |
| $\rho_j(u)$ | robustness weight of the alignment |
| $\sigma_i(u)$ | nominal noise standard deviation from offline calibration |
| $\Lambda_k$, $\Lambda_k^{(i)}$ | information block of surfel $k$, and the part contributed by view $i$ |
| $\mathcal B_j$, $B$ | block of neighbouring frames left out with view $j$, and its size |
| $\hat D_{-j}$, $V_{-j}$, $O_{-j}$ | cavity depth, its variance, accumulated opacity |
| $\beta_{-j}$ | distance bias estimated without the block |
| $\phi_k^{(j)}$, $\Phi_{-j}$ | share of a surfel's information removed with the block, and its render |
| $K$ | number of Gauss-Newton steps of the cavity |
| $\Omega_j$ | testable pixels of view $j$ |
| $r_j$, $v_j$, $z_j$ | residual, its nominal variance, standardised residual |
| $f_j(u)$ | latent fault indicator |
| $c_j(u)$, $s_j(u)$ | contamination prior, inlier scale |
| $G_\theta$, $h_\psi$, $t_\phi$ | prior head (blind-spot operator), scale head, image-only student |
| $R$ | blind-spot radius on the residual channels |
| $F$ | number of folds of frames in cross-fitting |
| $\eta$ | damping of the reconstruction weights between rounds |
| $\mathbf j_q$, $\Sigma_q$ | pose Jacobian of the rendered depth, pose covariance of the tested view |
| $p_{\mathrm{out}}$, $Z$ | outlier density and its half-range |
| $\zeta_j(u)$ | learned indicator: posterior fault probability |
| $S_j$, $\hat t$, $\alpha$ | view score, threshold, target false-alarm rate |
