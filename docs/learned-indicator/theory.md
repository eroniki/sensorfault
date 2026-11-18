# Theory of the Multi-Cue Fault Indicator: a Latent-Class Model of Several Noisy Fault Detectors

This section develops the statistical model in which several cheap signals ("cues") are treated as noisy
detectors of one latent fault variable and are combined with few or no fault labels. It states the
assumptions, derives when the model is identifiable, gives closed-form and likelihood estimators,
analyses the violations that are expected for depth-supervised surfel reconstruction, and lists what the
theory cannot deliver. Bibliographic attributions are recalled from memory and are collected, marked as
unverified, in Section 12. One small simulation (Section 6.1) was run to check the algebra; no other
claim here is an empirical result.

## 1. Setting

**Units.** A *unit* $n$ is the object that carries one fault variable. Two choices are considered, and
the model is the same for both.

- *Pixel units:* $n = (j, u)$, pixel $u$ of view $j$, with $f_n = f_j(u)$ as in the contamination model.
- *Message units:* $n = (i, k)$, the information-form message $(\Lambda_k^{(i)}, \boldsymbol\eta_{ik})$ that view $i$ sends to surfel $k$. The indicator then gates the fusion in closed form, $\Lambda_k(w) = \Lambda_0 + \sum_i w_{ik} \Lambda_k^{(i)}$ with $w_{ik} = 1 - \zeta_{ik}$.

Every cue must be expressed on the chosen unit. A pixel-level cue $x_i(u)$ is carried to a message unit
by the splatting weights, $x_{ik} = \sum_u w_{uk}\, x_i(u) / \sum_u w_{uk}$. A fault can spawn surfels
that only faulty views support; the conflict is then along rays (a surface seen through) and not at any
single surfel, so on message units the geometric cue must include a free-space violation term, or the
affected messages have no geometric cue at all (an abstention, Section 6.4).

**Fault variable.** $f_n \in \{0, 1\}$, with $f_n = 1$ when the depth evidence of the unit deviates from
the measurement model by many $\sigma$ inside a connected region. A whole-view affine error is absorbed
by $(a_i, b_i)$ and is not a fault.

**Cues.** There are $M$ cues. Cue $m$ produces a real score $x_{mn}$, larger meaning more suspicious,
and optionally a binarised vote $y_{mn} = \mathbb 1[x_{mn} > t_m]$, or an abstention $y_{mn} = \varnothing$.
The cues considered are:

| Cue | Score | Information used |
| --- | --- | --- |
| c1 geometric consensus | $\lvert z_j(u) \rvert$, the standardised leave-block-out residual | depth maps of views outside the block $\mathcal B_j$ |
| c2 cross-model disagreement | disagreement of two depth models on the same image after per-view alignment | one image, two networks |
| c3 equivariance failure | instability of one depth model under flip, crop, rescale | one image, one network |
| c4 photometric evidence | multi-view photo-consistency, or the influence of a depth message on held-out views' colour loss | colours of other views, never fitted to geometry |

**Covariates.** Each unit has observed covariates that are not cues: geometry $\mathbf x^g_n$ (range,
incidence, $\sqrt{V_{-j}}$, $O_{-j}$, $\Phi_{-j}$, number of supporting views), and frozen image features
$\mathbf e_n$ from a foundation model that is never fine-tuned (Section 8).

**Symbols added to the existing notation.** $\pi = \Pr(f = 1)$ is the prevalence (the constant-prior
case of $c_j(u)$). For a binarised cue, $p_m = \Pr(y_m = 1 \mid f = 1)$ is its detection rate and
$q_m = \Pr(y_m = 1 \mid f = 0)$ its false-alarm rate; $\delta_m = p_m - q_m$ is its informedness;
$\mu_m = \mathbb E\, y_m$. For a continuous cue, $g_m^1$ and $g_m^0$ are the class-conditional densities
of $x_m$. $\zeta_n$ remains the posterior fault probability, the indicator.

## 2. Model

### 2.1 Binarised cues

```math
f_n \sim \mathrm{Bernoulli}(\pi), \qquad
\Pr\bigl(y_{1n}, \dots, y_{Mn} \mid f_n\bigr) = \prod_{m=1}^{M} \Pr\bigl(y_{mn} \mid f_n\bigr), \qquad
\Pr(y_{mn} = 1 \mid f_n) = \begin{cases} p_m & f_n = 1\\ q_m & f_n = 0 \end{cases}
```

The model has $2M + 1$ parameters. The observable distribution of the $M$ votes is a two-component
mixture of product distributions,

```math
\Pr(y_1, \dots, y_M) = \pi \prod_m p_m^{y_m} (1 - p_m)^{1 - y_m} + (1 - \pi) \prod_m q_m^{y_m} (1 - q_m)^{1 - y_m} ,
```

and has $2^M - 1$ free cell probabilities.

### 2.2 Continuous cues

```math
p(x_1, \dots, x_M) = \pi \prod_m g_m^1(x_m) + (1 - \pi) \prod_m g_m^0(x_m) .
```

### 2.3 The assumption

**A1 (class-conditional independence).** For every unit and every pair $l \ne m$,
$x_{ln} \perp x_{mn} \mid f_n$. In words: once it is known whether the unit is faulty, the error of one
cue (its miss or its false alarm) says nothing about the error of another. A1 is a statement about the
*errors* of the cues. The cues themselves are strongly dependent, which is the point: all their
dependence is to be explained by the fault.

A1 does not require the cues to be independent of covariates, nor units to be independent of each other
(Sections 6.2 and 6.3). It does require, with $M$ fixed, that there be no second common cause of cue
errors: no shared blind spot, no shared nuisance. Section 6.1 argues that this is false for the four
cues as listed and derives the consequences.

**A2 (informativeness).** $0 < \pi < 1$ and $\delta_m \ne 0$ for at least three cues (for continuous
cues, $g_m^1 \ne g_m^0$ for at least three).

**A3 (polarity).** Faults are the minority, $\pi < \tfrac12$, or at least one designated cue is better
than chance in the stated direction, $\delta_m > 0$.

## 3. Identifiability from the Cues Alone

### 3.1 Second moments: a one-factor structure

Write $y_m - \mu_m = \delta_m (f - \pi) + \xi_m$, where $\xi_m = y_m - \mathbb E[y_m \mid f]$ has zero
mean given $f$. Under A1 the $\xi_m$ are uncorrelated across cues, so for $l \ne m$

```math
C_{lm} = \operatorname{Cov}(y_l, y_m) = \pi (1 - \pi)\, \delta_l \delta_m = u_l u_m , \qquad u_m = \sqrt{\pi(1-\pi)}\; \delta_m .
```

The off-diagonal part of the covariance matrix of the votes has rank one. Dividing by the standard
deviations $\sqrt{\mu_m (1 - \mu_m)}$ gives the same statement for correlations,

```math
\rho_{lm} = \rho_{lf}\, \rho_{mf} , \qquad \rho_{mf} = \operatorname{Corr}(y_m, f) = \frac{u_m}{\sqrt{\mu_m (1 - \mu_m)}} :
```

the correlation of two cues is the product of their correlations with the fault. This is Spearman's
one-factor model with a binary factor.

### 3.2 Three cues: the triplet solution

With three cues the three pairwise covariances determine the three loadings up to one common sign,

```math
u_1^2 = \frac{C_{12}\, C_{13}}{C_{23}}, \qquad u_2 = \frac{C_{12}}{u_1}, \qquad u_3 = \frac{C_{13}}{u_1} ,
```

provided $C_{23} \ne 0$, which is A2. The loadings mix accuracy and prevalence. They are separated by
the third central moment. Since the $\xi_m$ are mutually independent given $f$ and have zero conditional
mean,

```math
T_{123} = \mathbb E\bigl[(y_1 - \mu_1)(y_2 - \mu_2)(y_3 - \mu_3)\bigr] = \delta_1 \delta_2 \delta_3\, \mathbb E (f - \pi)^3 = \delta_1 \delta_2 \delta_3\, \pi (1 - \pi)(1 - 2\pi) ,
```

so that

```math
\kappa = \frac{T_{123}}{u_1 u_2 u_3} = \frac{1 - 2\pi}{\sqrt{\pi (1 - \pi)}} , \qquad
\pi = \frac12 \Bigl( 1 - \frac{\kappa}{\sqrt{\kappa^2 + 4}} \Bigr) ,
```

using $\kappa^2 + 4 = 1 / (\pi(1-\pi))$. Finally

```math
\delta_m = \frac{u_m}{\sqrt{\pi (1 - \pi)}}, \qquad p_m = \mu_m + (1 - \pi)\, \delta_m, \qquad q_m = \mu_m - \pi\, \delta_m .
```

All $2M + 1 = 7$ parameters are thus explicit functions of the $2^3 - 1 = 7$ free cell probabilities:
with three cues the model is exactly identified and has no testable implication beyond inequalities
(Section 6.5). This is the constructive form of the classical result that a latent class model with
three conditionally independent variables is identifiable up to relabelling of the classes; in the
Kruskal-rank formulation the condition for two classes and binary variables is that each of the three
variables has Kruskal rank two, which is $\delta_m \ne 0$ again.

### 3.3 The label-swap ambiguity and what breaks it

The solution above fixes $u$ only up to $u \to -u$. The flip sends $\kappa \to -\kappa$,
$\pi \to 1 - \pi$, $\delta_m \to -\delta_m$ and exchanges $p_m \leftrightarrow q_m$: it renames the
classes and leaves the observable distribution unchanged. No amount of unlabelled data resolves it. Two
pieces of prior knowledge do, and here both are available:

- *Rarity.* $\pi < \tfrac12$ selects the sign with $\kappa > 0$, that is $\operatorname{sign}(u_1 u_2 u_3) = \operatorname{sign}(T_{123})$.
- *Polarity.* $\delta_m > 0$ for a cue whose direction is known by construction (a large residual means fault) fixes the sign of $u_m$ and, through the signs of $C_{lm}$, of all others.

The two rules must agree; disagreement is evidence against the model. Rarity fails locally: in a
stratum of units dominated by faults (a view that is mostly water) the minority class is the clean one,
so polarity is the rule to use whenever the model is fitted per stratum (Section 6.2).

The relabelling ambiguity is the only one the mathematics leaves. It is not the only one that matters.
The model identifies *the binary variable that renders the cues conditionally independent*. Nothing in
the unlabelled data says that this variable is "depth error above $3\sigma$ in a connected region". If
the strongest common binary cause of the cues is something else, for instance "the pixel lies on a depth
discontinuity", that is what $f$ will be (Section 6.1).

### 3.4 Two cues

With $M = 2$ there are five parameters and three free cell probabilities $(\mu_1, \mu_2, C_{12})$. The
model is not identifiable; the solution set is a two-parameter family. What is known:

- $\rho_{12} = \rho_{1f}\, \rho_{2f}$: the product of the two cue-fault correlations. Since each is at most one in magnitude, $\lvert \rho_{mf} \rvert \ge \lvert \rho_{12} \rvert$. Two cues that agree strongly are both good; two cues that agree weakly may be one good and one useless, or two mediocre ones.
- The sign of $C_{12}$ says whether the cues point the same way.

Two cues become identifiable with additional structure, each of which is an assumption of the same kind
as A1:

- *A third view in disguise.* A covariate that changes the prevalence but not the accuracies (two populations with different $\pi$ and common $p_m, q_m$: the Hui-Walter design) supplies the missing equations. Any variable that depends on the cues only through $f$ is a third cue.
- *Symmetry.* Equal accuracies, or $p_m = 1 - q_m$.
- *Parametric shape for continuous scores.* A mixture of two Gaussians per cue is identifiable for any $M \ge 1$, purely because of the assumed shape. This is the kind of identifiability the single-cue contamination model has (Section 7), and it is the kind that fails when the shape is wrong.

For continuous cues without shape assumptions the same boundary holds: two-component mixtures of
products are nonparametrically identifiable for $M \ge 3$ and not for $M = 2$.

### 3.5 More than three cues

For $M \ge 4$ the model is over-identified: $2M + 1 < 2^M - 1$. Only three informative, mutually
conditionally independent cues are needed for identification; the excess is what makes the assumption
testable (Section 6.5) and what leaves room for dependence terms (Section 6.1). With $M = 4$ there are
$15 - 9 = 6$ spare degrees of freedom.

## 4. Estimators and the Indicator

### 4.1 Method of moments

Replace $\mu_m$, $C_{lm}$, $T_{lmn}$ by sample moments and apply Section 3.2. For $M > 3$, fit the
rank-one structure to all pairs at once,

```math
\hat u = \arg\min_u \sum_{l < m} \bigl( \hat C_{lm} - u_l u_m \bigr)^2 ,
```

or, when all $\hat C_{lm}$ have the consistent sign pattern, solve the linear least-squares problem
$\log \lvert \hat C_{lm} \rvert = \log \lvert u_l \rvert + \log \lvert u_m \rvert$. The prevalence follows
from the average of $\hat T_{lmn} / (\hat u_l \hat u_m \hat u_n)$ over triples. The estimate is
closed-form, has no local optima, and serves as the initial value of the likelihood fit.

**Sampling error.** The estimator divides by a covariance. By the delta method, for weakly correlated
cues, the relative variance of a sample covariance is about $1 / (n_{\mathrm{eff}}\, \rho_{lm}^2)$, so

```math
\operatorname{Var}\bigl(\log \hat u_1^2\bigr) \approx \frac{1}{n_{\mathrm{eff}}} \Bigl( \frac{1}{\rho_{12}^2} + \frac{1}{\rho_{13}^2} + \frac{1}{\rho_{23}^2} \Bigr) ,
```

neglecting the covariances between the three terms. The error is governed by the *weakest pair*. Two
illustrative cases with $\pi = 0.05$ and three equal cues: for $p = 0.6$, $q = 0.05$ the pairwise
correlation is $\rho_{lm} \approx 0.20$ and a 10% relative error on $u^2$ needs
$n_{\mathrm{eff}} \approx 7\,500$; for $p = 0.3$, $q = 0.05$ it is $\rho_{lm} \approx 0.05$ and the
same precision needs $n_{\mathrm{eff}} \approx 1.2 \times 10^5$. The third-moment estimate of $\pi$ is
noisier still. $n_{\mathrm{eff}}$ counts independent units; because faults and cue errors are regional,
it is closer to the number of regions than to the number of pixels (Section 6.3). Rare faults and weak
cues therefore make the label-free estimate a data-hungry one, and pooling over scenes is the realistic
regime.

### 4.2 Maximum likelihood by expectation-maximisation

The log-likelihood of the votes is $\sum_n \omega \log \Pr(y_{1n}, \dots, y_{Mn})$, with the deflation
weight $\omega = n_{\mathrm{eff}} / n$ as in the contamination model. EM alternates

```math
\text{E:}\quad \operatorname{logit} \zeta_n = \operatorname{logit} \pi + \sum_{m} \Bigl[ y_{mn} \log \frac{p_m}{q_m} + (1 - y_{mn}) \log \frac{1 - p_m}{1 - q_m} \Bigr] ,
```

```math
\text{M:}\quad \pi = \frac{1}{n} \sum_n \zeta_n, \qquad p_m = \frac{\sum_n \zeta_n\, y_{mn}}{\sum_n \zeta_n}, \qquad q_m = \frac{\sum_n (1 - \zeta_n)\, y_{mn}}{\sum_n (1 - \zeta_n)} .
```

This is the Dawid-Skene estimator for two classes. The likelihood is not concave; it has the mirrored
optimum of Section 3.3 and can have others. Initialising at the moment estimate and constraining
$\delta_m > 0$ for the anchor cue removes the mirror. A Beta prior on $\pi$ with $\alpha_0 < \beta_0$, as
already used for $\bar c_j$, encodes rarity.

### 4.3 The indicator

The indicator is the E-step posterior, $\zeta_n = \Pr(f_n = 1 \mid \text{cues of unit } n)$: the prior
log-odds plus one log-likelihood ratio per cue. Each firing cue adds $\log(p_m / q_m)$, each silent cue
adds $\log((1 - p_m)/(1 - q_m)) < 0$, an abstaining cue adds nothing. This additivity is A1. If A1 is
false, the sum counts shared evidence more than once and $\zeta$ is overconfident in both directions.

## 5. Continuous Cues

Binarisation discards the size of a score and needs a threshold chosen without labels. Three
alternatives, in increasing order of assumption.

**Nonparametric unmixing.** Fix a reference triple of binarised cues, estimate $\pi$ and one reference
cue's $(p_r, q_r)$ by Section 3.2. For any other cue $l$ that is conditionally independent of the
reference, the two observable conditional densities of its score are known mixtures of the two unknown
class-conditional densities:

```math
h_1(x) = p(x_l \mid y_r = 1) = w_1\, g_l^1(x) + (1 - w_1)\, g_l^0(x), \qquad w_1 = \frac{\pi\, p_r}{\mu_r},
```

```math
h_0(x) = p(x_l \mid y_r = 0) = w_0\, g_l^1(x) + (1 - w_0)\, g_l^0(x), \qquad w_0 = \frac{\pi (1 - p_r)}{1 - \mu_r},
```

with $w_1 - w_0 = \pi(1-\pi)\,\delta_r / (\mu_r (1 - \mu_r)) \ne 0$. Solving the $2 \times 2$ system,

```math
g_l^1 = \frac{(1 - w_0)\, h_1 - (1 - w_1)\, h_0}{w_1 - w_0}, \qquad g_l^0 = \frac{w_1\, h_0 - w_0\, h_1}{w_1 - w_0} .
```

The class-conditional densities of a score are obtained as histograms with no shape assumption. Their
ratio $g_l^1 / g_l^0$ is the label-free likelihood-ratio curve of the cue and yields its ROC curve.
Estimated densities that go negative beyond sampling error falsify A1 for the pair $(l, r)$.

**Semiparametric EM.** Maximise the mixture likelihood of Section 2.2 with flexible one-dimensional
class-conditional densities: weighted histograms or kernel estimates in the M-step, optionally with a
monotone likelihood ratio ($g_m^1 / g_m^0$ non-decreasing in $x_m$, fitted by isotonic regression),
which encodes polarity for every cue. The posterior is

```math
\operatorname{logit} \zeta_n = \operatorname{logit} \pi + \sum_m \log \frac{g_m^1(x_{mn})}{g_m^0(x_{mn})} .
```

**Parametric mixture on the score vector.** A two-component mixture on $(x_1, \dots, x_M)$ after
monotone transforms, for example Gaussian for the signed residual and log-normal for non-negative
disagreement scores. With diagonal class covariances this is A1 with a shape; with full class
covariances it allows dependence between cues, but identifiability then rests entirely on the assumed
shape. It is useful as a comparison, not as the model.

Continuous cues also make the model testable with fewer cues: for any pair,
$p(x_l, x_m) - p(x_l)\, p(x_m) = \pi (1 - \pi) (g_l^1 - g_l^0)(x_l)\, (g_m^1 - g_m^0)(x_m)$ is a rank-one
function of two arguments, which can be checked on a two-dimensional histogram even for $M = 2$, although
$M = 2$ remains unidentified.

## 6. Violations

### 6.1 Cues that share blind spots or causes

Let $\epsilon_{lm}^f = \operatorname{Cov}(y_l, y_m \mid f)$ and
$e_{lm} = \pi\, \epsilon_{lm}^1 + (1 - \pi)\, \epsilon_{lm}^0$. Without A1,

```math
C_{lm} = u_l u_m + e_{lm} .
```

**Effect on the triplet estimate.** Suppose only cues 2 and 3 are dependent, $e_{23} > 0$. Then

```math
\hat u_1^2 = \frac{C_{12}\, C_{13}}{C_{23}} = u_1^2\, \frac{u_2 u_3}{u_2 u_3 + e_{23}} , \qquad
\hat u_2^2 = \frac{C_{12}\, C_{23}}{C_{13}} = u_2^2 \Bigl( 1 + \frac{e_{23}}{u_2 u_3} \Bigr) .
```

The dependent pair is credited with accuracy it does not have, because agreement with each other is read
as agreement with the fault, and the independent cue is discredited. The bias is large when $e_{23}$ is
comparable with $u_2 u_3 = \pi(1-\pi)\delta_2\delta_3$, a small number for rare faults: a modest shared
nuisance is enough. In a simulation with $\pi = 0.05$, $(p, q) = (0.6, 0.02), (0.7, 0.1), (0.5, 0.08)$
and a nuisance present on 10% of units that fires cues 2 and 3 with probability 0.6 each, the estimate
of $u_1$ was 0.068 against a true 0.126, and that of $u_2$ was 0.228 against 0.123. In the limit, the
latent class that the model finds is the shared nuisance and not the fault.

**Expected dependence structure of the four cues.** By provenance:

- *c2 and c3 share the depth network and the image.* Both respond to whatever makes single-image depth hard: thin structures, depth edges, far range, repeated texture. Both can be silent on a confident, stable, model-independent error. Dependence is expected in both classes.
- *c1 and c4 share the multi-view geometry.* Both depend on poses, on the surfel model, on occlusion handling and on the number and baseline of the other views. A pose error of view $j$ displaces its cavity residual and its reprojected colours alike. Dependence is expected mainly in the clean class (common false alarms).
- *All four respond to depth discontinuities,* where a small misregistration produces a large score for any of them. A nuisance common to all cues is the most dangerous case: it is a rank-one addition to the covariance and competes with the fault for the role of the latent class.
- *Glass and water.* c1 fires only if views disagree; c4 is unreliable because the surface is not Lambertian; c2 and c3 may or may not fire. Cue behaviour here is tied to the appearance class, which is a covariate effect (Section 6.2) on top of the dependence.

**The two-block case, analysed.** Take the structure that the provenance suggests: dependence within
$\{1, 4\}$ and within $\{2, 3\}$, independence across. The four cross covariances are

```math
C_{12} = u_1 u_2, \quad C_{13} = u_1 u_3, \quad C_{42} = u_4 u_2, \quad C_{43} = u_4 u_3 ,
```

and they are invariant under $u_1, u_4 \to t\, u_1, t\, u_4$ and $u_2, u_3 \to u_2 / t, u_3 / t$ for any
$t > 0$, while $C_{14}$ and $C_{23}$ are free because $e_{14}$ and $e_{23}$ are. Hence: the *relative*
informativeness inside each block is identified ($u_1 / u_4 = C_{12} / C_{42}$,
$u_2 / u_3 = C_{12} / C_{13}$), and the *product* of the block strengths is identified, but how the
informativeness divides between the multi-view block and the single-image block is not. This is the
two-cue situation of Section 3.4 at the level of groups; in the full distribution the two $4$-state
groups give a $4 \times 4$ table of rank two, whose factorisation is non-unique. The inequalities
$u_m^2 \le \mu_m (1 - \mu_m)$ bound $t$ to an interval and nothing more. **Under the dependence
structure that is most plausible for these cues, the label-free model is therefore not identified.**
Identification needs a third block that is conditionally independent of both.

**Ways to model dependence.**

1. *Grouping.* Replace a dependent set of cues by one composite cue with an unrestricted class-conditional joint distribution (for two binary cues, a four-state cue with $2 \times 3$ parameters; for scores, a two-dimensional density), and assume A1 between groups only. Identification needs three informative groups. Four binary cues in groups $\{1\}, \{2, 3\}, \{4\}$ use $1 + 2 + 6 + 2 = 11$ of 15 degrees of freedom and stay testable.
2. *Pairwise terms.* A class-conditional log-linear model, $\Pr(y \mid f) \propto \exp\bigl( \sum_m \theta_m^f y_m + \sum_{(l,m) \in E} \theta_{lm}^f y_l y_m \bigr)$, with an edge set $E$ fixed in advance from provenance. One class-specific edge is equivalent to grouping that pair. With $M = 4$ at most six extra parameters fit, and not every edge set is identified (the two-block set is not). The edge set cannot be learned from four cues; methods that recover it from the data separate a sparse from a low-rank part of the inverse covariance and need many more cues than are available here.
3. *A second latent variable.* A nuisance class (clean, hard-but-correct, faulty) or a continuous random effect shared by a subset of cues. A three-class model needs richer observations than four binary votes; with continuous or multi-level cues in three independent groups it is generically identifiable, but the third class multiplies the sampling problem of Section 4.1.
4. *Removing the cause.* Condition on the covariate that mediates the dependence (Section 6.2), or exclude the units where the nuisance acts (a band around rendered depth discontinuities) and report them as abstentions.

**Designing for independence.** Dependence that comes from shared inputs can be reduced by splitting the
inputs, which is cheaper than modelling it:

- compute c1 from one set of other views and c4 from a disjoint set, so that at least their noise is independent, even though poses remain shared;
- with several depth models available, compute c3 on the model whose depth is used and c2 between two *other* models: c2 then measures the difficulty of the image region for depth networks in general and does not share the realised error of the model under test;
- keep colour out of the surface fit, so that c4 does not inherit the error of c1 through the geometry it judges.

Splitting removes shared noise. It does not remove shared systematic causes, which is what frozen
features as covariates are for (Section 8).

### 6.2 Accuracy that depends on covariates

Cue accuracy varies with range, incidence, texture and the number of views; prevalence varies with
appearance. Let $w$ denote covariates, with $\pi(w)$, $p_m(w)$, $q_m(w)$.

**Pooling creates dependence.** Suppose A1 holds given $(f, w)$. Marginally over $w$ within a class,

```math
\operatorname{Cov}(y_l, y_m \mid f = 1) = \operatorname{Cov}_w\bigl( p_l(w),\, p_m(w) \mid f = 1 \bigr) ,
```

which is non-zero as soon as two cues are both better (or both worse) on the same kind of unit. A pooled
fit then suffers exactly the bias of Section 6.1. Heterogeneity is the generic source of
class-conditional dependence, and it is the removable part.

**What is identifiable.** Conditionally on $w$ the argument of Section 3 applies unchanged. The
functions $\pi(w)$, $p_m(w)$, $q_m(w)$ are identified at every $w$ at which (i) A1 holds given $(f, w)$,
(ii) at least three cues or groups are informative, and (iii) enough effective units exist. Nothing is
identified about how accuracy depends on a covariate that is not observed; only its average is fitted,
with the bias above.

**Parametrisation.** Condition (iii) forbids a free function of a high-dimensional $w$. The model is
therefore structured as in the contamination model, where the prior sees everything and the scale sees
geometry only:

```math
\operatorname{logit} \pi_n = G_\theta(\mathbf e_n, \mathbf x^g_n), \qquad
\operatorname{logit} p_{mn} = \mathbf a_m^\top \mathbf w_{mn}, \qquad
\operatorname{logit} q_{mn} = \mathbf b_m^\top \mathbf w_{mn} ,
```

where $\mathbf w_{mn}$ is a short, declared list per cue (for c1: number of supporting views, $\sqrt{V_{-j}}$, $\Phi_{-j}$; for c4: local texture energy, number of views; for c2 and c3: range, distance to the nearest rendered discontinuity, and an appearance stratum, Section 8). EM is unchanged except that the M-step is a weighted logistic regression per cue and class and a regression of the prior head on $\zeta$.

Three cautions.

- *The prior must not see the cues it is combined with.* If the covariates of $\pi_n$ determine a cue value of unit $n$, the prior can copy that cue and the cue is counted twice. This is the blind-spot condition in general form; cross-fitting over folds of frames remains necessary when the head is fitted on the scene it is applied to.
- *A variable that enters only the prior is one more cue.* If appearance changes prevalence but not accuracy, it acts as the extra view of Section 3.4 and adds identifying information. If it also changes accuracy (it does, for c4 through texture and for c2, c3 through material), it must enter both, and it then identifies less.
- *Label swap per stratum.* Rarity may fail within a stratum; polarity of an anchor cue, and continuity of the parameter functions in $w$, keep the labelling consistent.

### 6.3 Spatial dependence between units

Faults are regions, and cue noise is spatially correlated with length scale $\ell$; neighbouring frames
repeat both. Units are not independent. Three consequences.

1. *Point estimates survive.* The single-unit marginal distribution of the cue vector is the mixture of Section 2 whatever the dependence between units. The sum of single-unit log-likelihoods is a composite likelihood whose maximiser is consistent under stationarity; the moment estimator likewise. Dependence between units does not bias the parameters.
2. *Uncertainty does not.* Standard errors and test levels must use $n_{\mathrm{eff}}$, or a block bootstrap whose blocks are regions, views or strips. The tests of Section 6.5 are invalid with pixel counts.
3. *The single-unit posterior is valid and wasteful.* Neighbours carry information about $f_n$. It can be used by pooling scores over surfels or superpixels before the model, or by a spatial prior: a Potts field on $f$ with mean-field EM gives $\operatorname{logit} \pi_n = \operatorname{logit} \pi + \beta_s \sum_{n' \in \mathcal N(n)} (2 \zeta_{n'} - 1)$, of which the blind-spot prior head is the learned generalisation.

Spatial coherence is also a source of identification: the same cue at two units of one region, further
apart than the noise correlation, is two conditionally independent views of one $f$. A single cue with
spatial replicates therefore satisfies the three-view condition, which is what the blind-spot model
relies on. The condition is weaker than it looks. The *systematic* errors of a cue are regional too: a
false alarm of c2 covers a whole tree canopy, not a pixel. For such errors no blind-spot radius
separates noise from fault, and spatial replication cannot substitute for cues with different
mechanisms.

### 6.4 Cues that abstain or are uninformative on some units

**Declared abstention.** c1 abstains on untestable pixels ($O_{-j}$ below threshold), c4 where local
texture energy is below a threshold. With $\mathcal O_n$ the set of cues present,

```math
\Pr\bigl(y_{\mathcal O_n}\bigr) = \sum_{f} \pi_n^f (1 - \pi_n)^{1-f} \prod_{m \in \mathcal O_n} \Pr(y_{mn} \mid f) ,
```

and an absent cue contributes no term to the likelihood or to $\operatorname{logit} \zeta_n$. This is
correct if abstention is a function of covariates and the model conditions on them. It is not correct to
assume that prevalence is the same where a cue abstains: texture-free surfaces are where c4 abstains
*and* where depth models fail. The prevalence must be free per abstention pattern. Given transported
accuracies it is identified by any single remaining cue, $\pi = (\mu_m - q_m) / \delta_m$, and with two
remaining cues the three observables against one free parameter leave two degrees of freedom to test
the transport.

**The transport assumption.** In a stratum with fewer than three active groups, accuracies are not
identified from that stratum. They are carried over from strata where three are active, on the
assumption that the accuracy of a cue does not depend on whether another cue abstains. For c2 and c3 on
texture-free surfaces this is doubtful and cannot be checked without labels beyond the two-degree test
above.

**Undeclared uninformativeness.** A cue can have $\delta_m = 0$ on a set of units without knowing it:
c1 where all views outside the block repeat the error; c4 on surfaces whose colour does not change under
the wrong geometry. If a covariate predicts the condition (few views, small baseline, low texture), it
goes into $\mathbf w_m$ and the fitted $p_m(w)$ falls towards $q_m(w)$ there, so that silence of the cue
stops counting as evidence of cleanness. If no covariate predicts it, the model uses the average
accuracy and is overconfident that silent units are clean.

### 6.5 Testing the independence assumption without labels

With $M \ge 4$ (or $M \ge 3$ continuous cues) A1 has consequences that can be checked on the cues alone.

**Tetrad constraints.** Under A1, $C_{lm} = u_l u_m$, so for any four cues

```math
C_{12}\, C_{34} = C_{13}\, C_{24} = C_{14}\, C_{23} .
```

For $M = 4$ these are two independent constraints (six covariances, four loadings). The pattern of
violation localises the dependence: each product corresponds to one way of pairing the four cues. If
$C_{12} C_{34} = C_{13} C_{24} \ne C_{14} C_{23}$, the excess dependence lies inside the pair $\{1, 4\}$
or inside $\{2, 3\}$, or both; the constraint cannot tell which. In the simulation of Section 6.1 the
three products were $1.16$, $1.14$, $1.14 \times 10^{-4}$ under A1 and $1.02$, $1.01$, $3.45 \times 10^{-4}$
with the shared nuisance on cues 2 and 3.

**Flattening ranks.** For a split of the cues into two groups $A \mid B$, arrange the joint probabilities
as a matrix with rows indexed by the states of $A$ and columns by those of $B$. If $A \perp B \mid f$,
its rank is at most two. For $M = 4$ the three $4 \times 4$ flattenings $\{12 \mid 34\}$,
$\{13 \mid 24\}$, $\{14 \mid 23\}$ all have rank two under A1; under the two-block structure only
$\{14 \mid 23\}$ does. Testing the third and fourth singular values, with a block bootstrap, tests
which grouping is consistent with independence between groups. For continuous cues the same holds for
binned scores, and the rank-one test on the pairwise dependence function (Section 5) is available for
every pair.

**Agreement between triples.** Each of the $\binom{M}{3}$ triples yields its own $\hat\pi$ and its own
$(\hat p_m, \hat q_m)$. Under A1 they agree within sampling error. For continuous cues, $\hat\pi$ must
also be invariant to the binarisation thresholds. This is the tetrad information in a readable form: the
triple that excludes a dependent pair gives different numbers from the triples that contain it.

**Feasibility.** $C_{12} C_{13} C_{23} = (u_1 u_2 u_3)^2 \ge 0$; $u_m^2 \le \mu_m(1-\mu_m)$; all
$p_m, q_m \in [0, 1]$; unmixed densities non-negative; rarity and polarity select the same labelling.
These are the only checks available with three binary cues.

**Likelihood-ratio fit.** The deviance of the fitted model against the saturated multinomial on the
$2^M$ vote patterns ($6$ degrees of freedom for $M = 4$), and bivariate residuals of each pair, with
$n_{\mathrm{eff}}$ in place of $n$.

**What the tests cannot detect.**

- A dependence that is itself rank-one and proportional to the loadings, $e_{lm} \propto u_l u_m$: an error shared by all cues in proportion to their sensitivity. It inflates every accuracy and passes every test. Errors common to all cues are invisible to the model in the same way that errors repeated by all views are invisible to depth consensus.
- Whether the latent class is the fault. A model whose latent class is "depth discontinuity" can fit perfectly.
- With $M = 4$ there are two tetrads. A pass is weak evidence; a failure is strong evidence.

## 7. Relation to the Contamination Mixture on the Depth Residual

**The contamination model is the class-conditional model of c1.** Take $x_1 = z_j(u)$ and

```math
g_1^0(z \mid \mathbf x^g) = \mathcal N\bigl(z;\ 0,\ s(\mathbf x^g)^2\bigr), \qquad g_1^1(z) = p_{\mathrm{out}}(z) = \frac{1}{2Z}, \qquad \pi_n = c_j(u) .
```

With $M = 1$ the indicator of Section 5 is the contamination posterior. With more cues,

```math
\operatorname{logit} \zeta_n = \underbrace{\operatorname{logit} c_j(u) + \log \frac{p_{\mathrm{out}}(z_j(u))}{\mathcal N\bigl(z_j(u); 0, s_j(u)^2\bigr)}}_{\text{contamination model}} + \sum_{m \ge 2} \log \frac{g_m^1(x_{mn})}{g_m^0(x_{mn})} .
```

Seen from the latent-class model, the single-cue model is identified by three things only: the assumed
shapes (a Gaussian core against a flat outlier density), spatial replicates beyond the blind spot acting
as further views of the same cue (Section 6.3), and appearance entering the prior as an extra cue
(Section 6.2). All of its evidence is depth consensus.

**What several cues add in principle.**

1. *Identification without a shape.* The class-conditional densities of $z$ can be estimated by the unmixing formula of Section 5 with another cue as reference, and compared with the Gaussian-plus-uniform assumption. The scale head can be fitted with weights from the *leave-one-cue-out* posterior $\zeta^{(-1)}$, computed from the other cues only; under A1 this posterior is independent of the noise of $z$ given $f$, so it plays the role of the blind spot without a spatial argument. The cavity principle, never judge an observation with something it influenced, reappears at the level of cues.
2. *A fault class that is wider than "outlier in $z$".* In the single-cue model the latent variable is by construction "$z$ is not from the inlier density". A fault that every view outside the block repeats has an inlier-like $z$ and belongs to the clean class; the model cannot even represent it. In the multi-cue model the fault class is defined by the common factor of all cues, and c1 has a detection rate $p_1 < 1$ that the model estimates: the fraction of faults that are visible to consensus. A unit with a small residual can then still receive a high posterior.
3. *A quantified requirement for that to happen.* For a unit on which c1 is silent and c2, c3 fire (c4 abstaining), $\zeta > \tfrac12$ requires

```math
\frac{p_2\, p_3}{q_2\, q_3} > \frac{(1 - \pi)(1 - q_1)}{\pi (1 - p_1)} .
```

   With $\pi = 0.05$, $p_1 = 0.6$, $q_1 = 0.02$ the right side is about $47$: each of the two single-image cues needs a positive likelihood ratio near $7$ (for example $p = 0.7$, $q = 0.1$), *and* the two must be conditionally independent for the ratios to multiply. If c2 and c3 are dependent, their joint ratio is far below the product and the consensus-blind fault stays below threshold.

**Conditions under which the gain is real.** Let the fault class consist of consensus-visible and
consensus-blind faults.

- If c2, c3, c4 respond to both subtypes alike, the subtype is private to c1, A1 still holds given $f$, and everything above applies. This is plausible when the repetition is caused by viewing geometry (neighbouring frames of a strip see the surface from the same direction), which single-image cues do not know about.
- Faults repeated within a strip but not across strips are visible to c1 once the block is left out. They are the units on which the accuracies of c2 to c4 for repeated-type faults are learned, and from which they are transported to the units where c1 is blind.
- If the repetition is caused by the material itself (water, mirror glass: every view and possibly every depth model fails the same way, c3 is stable because the network is confidently wrong, c4 abstains or is unreliable), then all cues are silent together. Such units get a posterior below the prior; the fault is undetectable by this model, the estimated prevalence excludes it, and no test of Section 6.5 reveals it. Only the prior head can help, by generalising from appearance: if the same material is detectably faulty elsewhere, frozen features carry that to the silent units. This is extrapolation, not evidence.
- On units where c1 is uninformative, the remaining cues are one single-image block and a frequently abstaining c4. By Section 6.1 that is not enough to *identify* anything locally; the posterior there rests entirely on transported parameters.

## 8. Frozen Foundation Features as Covariates

The learned parts are light heads on features of a frozen backbone; nothing is fine-tuned. In this model
the features $\mathbf e_n$ are covariates, never cues, and enter in two places.

**Prevalence.** $\operatorname{logit} \pi_n = G_\theta(\mathbf e_n, \mathbf x^g_n)$, a linear or small
convolutional head on frozen self-supervised image features (DINOv2-type patch features, upsampled to the
unit). This head *is* the image-only student: it learns which appearances fail. Its M-step is a
regression onto $\zeta$. The features should come from a network unrelated to the depth models, so that
they carry the systematic, appearance-determined part of the fault probability and none of the realised
noise of any cue.

**Accuracy.** Letting $p_m, q_m$ depend on the full feature vector would destroy identification
(condition (iii) of Section 6.2). Features therefore enter accuracy through a bottleneck fixed before
any cue is seen: a partition of feature space into $K_e$ appearance strata by clustering the frozen
features (or a projection onto a few principal directions). Each cue has $p_m(h, \mathbf w_m)$ and
$q_m(h, \mathbf w_m)$ per stratum $h \in \{1, \dots, K_e\}$, with shrinkage towards the pooled value, and the
triplet argument holds within each stratum. This is what turns appearance-mediated dependence between
cues (c2 and c3 both fire on vegetation; c4 fails on glass where c1 also behaves differently) into a
modelled covariate effect instead of a violation of A1. The tetrad and rank tests are run per stratum:
dependence that disappears within strata was heterogeneity, dependence that stays is shared noise.

**Why frozen matters for the theory.** A trainable backbone could shape the strata, and the prior, to
the cue values of the scene and reproduce the memorisation failure of per-scene fitting. With frozen
features only the head adapts, its capacity is small, and the strata are fixed functions of the image.
Cross-fitting of the head over folds of frames is still required when it is fitted on the scene under
test.

**Which features must not be used where.** The frozen encoder of the depth model itself is a strong
predictor of that model's failures, and equally of the scores of c2 and c3, which are functions of the
same forward pass. As an input of the prior head it lets the prior decode those cues and count them
twice. It is admissible as an accuracy covariate of c2 and c3, or as the prior's input if c2 and c3 are
then removed as cues. Frozen multi-view features stand in the same relation to c1 and c4.

## 9. What a Few Labels Buy

Let $\mathcal U$ be the unlabelled units and $\mathcal L$ a small labelled set (exact-depth fault masks
of a synthetic scene, or a few annotated views). Labelled scenes are disjoint from evaluation scenes.

**Semi-supervised likelihood.**

```math
\mathcal J_{\mathrm{semi}} = - \sum_{n \in \mathcal U} \omega \log \sum_{f \in \{0,1\}} \Pr(f \mid w_n) \prod_{m \in \mathcal O_n} g_m^f(x_{mn} \mid w_n)
\ -\ \lambda \sum_{n \in \mathcal L} \omega_{\mathcal L} \Bigl[ \log \Pr(f_n \mid w_n) + \sum_{m \in \mathcal O_n} \log g_m^{f_n}(x_{mn} \mid w_n) \Bigr] .
```

EM is unchanged with $\zeta_n$ clamped to $f_n$ on $\mathcal L$. With dependence terms the product is
replaced by the grouped or log-linear class-conditional. The weight $\lambda$ trades the two sources;
$\lambda \to \infty$ with fixed accuracies is the plug-in classifier, $\lambda = 0$ the label-free model.

**What the labels provide, in order of how few are needed.**

1. *Labelling of the classes.* A handful of labelled faults fixes the sign of Section 3.3 and, more importantly, ties the latent class to the intended definition of a fault, which no unlabelled quantity can do.
2. *Anchored accuracies.* $p_m$ and $q_m$ are estimated directly, with variance $p_m(1-p_m)/n_1$ and $q_m(1-q_m)/n_0$, where $n_1$, $n_0$ count *independent* labelled faulty and clean regions. Faults are rare and regional, so $n_1$ is the binding number: one synthetic scene may contain a few dozen fault regions. The unlabelled term then mainly estimates prevalence, for which, given accuracies, a single cue suffices: $\pi = (\mu_m - q_m)/\delta_m$ per stratum and per scene. This is the label-shift setting: accuracies transported, prevalence re-estimated on the target.
3. *Dependence.* The class-conditional covariances $\epsilon_{lm}^f$ are directly observable on $\mathcal L$. This resolves what Section 6.1 showed to be unidentifiable: with $e_{14}$ and $e_{23}$ known, $C_{14} - e_{14} = u_1 u_4$ supplies the missing equation and fixes $t$. It also selects the edge set $E$ by an ordinary test instead of by provenance. The full class-conditional joint of four binary cues has $2 \times 15$ parameters and is estimable from a few hundred independent labelled regions per class; A1 is then not needed at all on the labelled domain.
4. *Calibration.* A monotone recalibration of $\zeta$ (isotonic or logistic) on $\mathcal L$ repairs the overconfidence caused by residual dependence, and labelled clean units supply conformal calibration scores without assuming whole scenes to be fault-free.
5. *A test of the label-free model.* Accuracies estimated from $\mathcal L$ alone and from $\mathcal U$ alone must agree. A significant difference means either that A1 fails or that the domains differ; the tetrad tests on $\mathcal U$ separate the two cases to the extent of their power.

**What the labels do not provide.** Labels from a synthetic domain anchor accuracies *for that domain*.
Cue accuracies depend on the depth model's behaviour on real imagery, on real pose error and on real
materials, and can shift. The natural hybrid is to transport from $\mathcal L$ what is hard to identify
without labels (the dependence structure, the class semantics, the shapes of $g_m^f$) and to re-estimate
on the target what the unlabelled likelihood identifies given those (prevalence, and main-effect
accuracies per stratum). That the main effects are identified when the excess covariances are
transported follows from the covariance equations above; for interactions transported as log-odds
ratios the corresponding statement is plausible by parameter count ($9$ unknowns against $15$ degrees of
freedom for $M = 4$) but is not proven here. Under a misspecified model the unlabelled term can make
the fit worse than labelled data alone; item 5 is the safeguard, and $\lambda$ should be chosen by
held-out labelled likelihood when $\mathcal L$ is large enough to split, by held-out unlabelled
likelihood otherwise.

**Regimes.**

| Labels | Role of the label model | Rests on |
| --- | --- | --- |
| none | defines the indicator; all parameters from agreement patterns | A1 between at least three groups, A2, A3; tests of Section 6.5 |
| few (tens of fault regions) | labels fix semantics, dependence structure and calibration; unlabelled data fix prevalence and adapt accuracies | transport of dependence from labelled to target domain |
| many | class-conditional joint estimated directly; the latent-class structure is a regulariser and a domain-adaptation device | similarity of domains |

In every regime the learned components remain heads on frozen features; the labels are used to fit a
model with tens of parameters, and to calibrate, not to train a large detector.

## 10. Quantities That Decide Whether the Model Applies

The theory reduces the question of whether the approach can work to a small set of numbers, measurable
on any scene with exact depth, used for evaluation only (or as $\mathcal L$ in the semi-supervised
regime, on scenes then excluded from evaluation):

- the ROC curve of each cue against true faults, and $(p_m, q_m)$ at the operating thresholds, per appearance stratum;
- the class-conditional dependence of each pair, $\epsilon_{lm}^0$ and $\epsilon_{lm}^1$, relative to $u_l u_m$, before and after conditioning on strata;
- the fraction of true faults on which c1 is silent, and the detection rates of c2 to c4 on exactly those;
- the fraction of true faults on which every cue is silent;
- the gap between label-free and labelled estimates of $\pi$, $p_m$, $q_m$, and whether the label-free tests of Section 6.5 flag the scenes on which the gap is large.

If $e_{lm}$ is comparable with $u_l u_m$ for all pairs, or if fewer than three conditionally independent
groups exist after stratification, the label-free version of the model is not usable and only the
semi-supervised version remains.

## 11. What This Theory Cannot Deliver

1. **The meaning of the latent class.** Without labels the model identifies the dominant shared binary cause of the cues. That it coincides with the fault definition is a hypothesis, and it competes with nuisances that all cues share, depth discontinuities above all.
2. **Identification with the cues as listed.** The four cues plausibly form two dependent blocks, multi-view (c1, c4) and single-image network (c2, c3). For two blocks the division of accuracy between them is not identified (Section 6.1). A third conditionally independent block, covariates that remove the within-block dependence, or labels are required. Whether stratification by frozen features removes enough dependence is an empirical question.
3. **Detection of errors common to all cues.** A fault on which every cue is silent has a posterior below the prior, is excluded from the estimated prevalence, and is not revealed by any test. The model moves the blind spot from "repeated by all views" to "missed by all cues"; it does not remove it.
4. **A proof of independence.** The tests can reject A1; they cannot confirm it. With four cues there are two tetrad constraints, and a dependence proportional to the loadings is undetectable in principle.
5. **Calibrated probabilities.** $\zeta$ is a probability only if A1 and the covariate model hold. Residual dependence makes it overconfident. A false-alarm guarantee still needs a separate calibration step with its own exchangeability assumption.
6. **Precision at small samples.** The moment estimator divides by covariances of order $\pi(1-\pi)\delta_l\delta_m$; with rare faults, weak cues and $n_{\mathrm{eff}}$ counted in regions, per-scene fitting may be too noisy, and pooling over scenes assumes accuracies common to those scenes.
7. **Accuracy as a function of unobserved or high-dimensional covariates.** Only low-dimensional, declared dependence is identified; the rest is averaged, with bias.
8. **Label-free choice of thresholds and structure.** Binarisation thresholds, strata, exclusion bands and the edge set must be chosen by criteria computed without labels (held-out likelihood, tetrad fit). Selecting for fit to the independence constraints can select a model whose latent class is a nuisance, because nuisances can satisfy the constraints better than faults do.
9. **Information beyond the cues.** The combination cannot exceed the joint information of the cues; the distilled head adds only interpolation over appearance and space.
10. **Transfer of labelled accuracies.** In the semi-supervised regime, the anchoring is as good as the similarity between the labelled and the target domain, which the theory does not provide.
11. **Message units when faults build their own surfels.** The per-unit model assumes that every unit has a well-defined fault state and cues. Where a fault is represented by surfels supported only by faulty views, the model has units without a geometric cue and no representation of the conflict along rays.

## 12. Attributions (recalled from memory, not verified)

The following attributions are given from memory; titles, years and exact statements must be checked
before citation.

- Latent class estimation of observer error rates by EM: Dawid and Skene (1979).
- Identifiability of latent class models with three conditionally independent variables through Kruskal's theorem: Allman, Matias and Rhodes (2009); Kruskal (1977).
- Nonparametric identifiability of two-component mixtures of products for three or more coordinates, and failure for two: Hall and Zhou (2003).
- Two tests in two populations with different prevalence: Hui and Walter (1980).
- Rank-one off-diagonal covariance of conditionally independent classifiers, spectral estimation, and use of the third-moment tensor for class imbalance: Parisi, Strino, Nadler and Kluger (2014); Jaffe, Nadler and Kluger (2015).
- Triplet method-of-moments label models and dependency structure from the inverse covariance in weak supervision: Ratner et al. (2016 to 2019); Fu et al. (2020); Varma et al. (2019).
- Tetrad constraints for one-factor models: Spearman (1904).
- Latent class models with random effects for conditionally dependent diagnostic tests: Qu, Tan and Kutner (1996).
- Semiparametric EM for mixtures of products: Benaglia, Chauveau and Hunter (2009).
- Prevalence estimation under label shift from a confusion matrix: Lipton, Wang and Smola (2018).
- Unlabelled data degrading a misspecified generative classifier: Cozman and Cohen (2002).

## 13. Notation Added

| Symbol | Meaning |
| --- | --- |
| $n$ | unit: pixel $(j, u)$ or message $(i, k)$ |
| $M$, $m$ | number of cues, cue index |
| $x_{mn}$, $y_{mn}$, $t_m$ | score of cue $m$ on unit $n$, its binarised vote, the threshold |
| $\mathcal O_n$ | cues that do not abstain on unit $n$ |
| $\pi$, $\pi_n$ | fault prevalence; covariate-dependent prior, equal to $c_j(u)$ on pixel units |
| $p_m$, $q_m$, $\delta_m$ | detection rate, false-alarm rate, informedness $p_m - q_m$ |
| $\mu_m$, $C_{lm}$, $T_{lmn}$ | mean, covariance and third central cross-moment of the votes |
| $u_m$ | loading $\sqrt{\pi(1-\pi)}\,\delta_m$ |
| $\rho_{lm}$, $\rho_{mf}$ | correlation between two cues, and between a cue and the fault |
| $\kappa$ | $T_{123}/(u_1 u_2 u_3)$, the skewness of the fault variable |
| $h$, $K_e$ | appearance stratum from clustering frozen features, number of strata |
| $g_m^1$, $g_m^0$ | class-conditional densities of the score of cue $m$ |
| $\epsilon_{lm}^f$, $e_{lm}$ | class-conditional covariance of two votes, and its prevalence-weighted mean |
| $t$ | free scale between the two blocks in the two-block model (Section 6.1) |
| $\mathbf e_n$ | frozen foundation-model features of unit $n$ |
| $\mathbf w_{mn}$, $\mathbf a_m$, $\mathbf b_m$ | accuracy covariates of cue $m$ and their coefficients |
| $\mathcal U$, $\mathcal L$, $\lambda$ | unlabelled units, labelled units, weight of the labelled term |
| $\zeta^{(-m)}$ | posterior computed without cue $m$ |
