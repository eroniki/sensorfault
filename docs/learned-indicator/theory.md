# Theory: a Monocular Depth Prior Judged by Motion Parallax

This note develops the statistical model of the fault indicator for posed aerial video. One monocular
depth map per image is the dense prior of a 2D Gaussian surfel reconstruction. A frozen point tracker
or optical-flow model supplies correspondences; with known poses each track is a triangulated depth
with an error bar set by its parallax. The gap between the monocular depth and the triangulated depth
is the evidence. A light head on frozen features is trained on that evidence without fault labels and
carries it to pixels where tracking abstains.

The note states the measurement model, derives what is identified without labels and under which
assumptions, gives the estimators and their cost, analyses the two standing objections (why keep the
monocular prior; whether the head transfers from textured to untextured surfaces), and lists what
would refute the approach. Nothing here has been measured. Numerical examples are illustrations with
assumed inputs and are marked as such. Attributions and statements about available models are
collected in Section 12 with a note on what was checked.

## 1. Setting and Symbols

A static scene is observed by $N$ posed frames in strips. Frame $j$ has a monocular depth map
$\hat d_j$, aligned per view by $(a_j, b_j)$ and corrected by the shared distance bias $\beta$. A
*fault* is a connected region where the aligned depth deviates from the truth by many $\sigma$.
$f_j(u) \in \{0, 1\}$ is the latent fault state of pixel $u$ in view $j$, $\zeta_j(u)$ its posterior
(the indicator), $\boldsymbol\phi_j(u)$ frozen features, $s = s(\boldsymbol\phi)$ an appearance stratum
(a cluster of frozen features), and $\pi(\boldsymbol\phi)$ the prior fault probability produced by the
head. The cavity residual $z_j(u)$ (the depth map of view $j$ against the surface fused from the views
outside its block $\mathcal B_j$) remains available as depth-consensus evidence.

All depth quantities in this note are expressed in **inverse depth** $\xi = 1/d$ along the ray of the
reference pixel, because image motion is linear in it. For a depth error $\delta d$ at depth $d$,
$\delta \xi \approx -\delta d / d^2$.

| Symbol | Meaning |
| --- | --- |
| $\xi$, $\hat\xi_M$, $\epsilon_M$ | true inverse depth, aligned monocular value, its error $\hat\xi_M - \xi$ |
| $\mathbf q_i$, $\boldsymbol\epsilon_i$, $\sigma_i$ | tracked position in frame $i$, its error, nominal tracker noise (pixels) |
| $\mathbf b_i$, $b_i$ | parallax gain $\partial \mathbf p_i / \partial \xi$ and its length (pixels per unit inverse depth) |
| $e_i$, $o_i$ | track residual along and across the epipolar line |
| $r_T$, $\epsilon_T$, $\Lambda_T$ | parallax gap, witness error, parallax information |
| $A_\perp$, $A_\parallel$ | ancillary statistics of a track |
| $\Lambda_M$, $\vartheta$ | information of the monocular depth $1/\sigma_M^2$, information ratio $\Lambda_T / \Lambda_M$ |
| $r_C$, $\epsilon_C$ | consensus gap (cavity residual in inverse depth), error of the cavity depth |
| $\varrho$ | repetition coefficient: share of a monocular error that the cavity repeats |
| $\iota$ | label information of a pixel, in $[0, 1]$ |
| $e(\boldsymbol\phi)$ | propensity: probability that a pixel with features $\boldsymbol\phi$ has informative parallax |

## 2. The Witness: Parallax Residual of a Track Against the Monocular Depth

### 2.1 Measurement model

Let $\tilde u$ be the homogeneous reference pixel in frame $j$, and $(R_{ij}, \mathbf t_{ij})$ the known
pose of frame $i$ relative to $j$. A point at inverse depth $\xi$ on the reference ray projects to

```math
\mathbf p_i(\xi) = \operatorname{proj}\bigl( \mathbf a_i + \xi\, \mathbf c_i \bigr), \qquad
\mathbf a_i = K_i R_{ij} K_j^{-1} \tilde u, \qquad \mathbf c_i = K_i \mathbf t_{ij} ,
```

which traces the epipolar line of $u$ in frame $i$ as $\xi$ varies. The tracker reports
$\mathbf q_i = \mathbf p_i(\xi) + \boldsymbol\epsilon_i$ for the frames $i \in \mathcal T(u)$ in which it
follows the point, $L = \lvert \mathcal T(u) \rvert$. Linearising at the monocular value,

```math
\mathbf q_i - \mathbf p_i(\hat\xi_M) \approx \mathbf b_i\, (\xi - \hat\xi_M) + \boldsymbol\epsilon_i, \qquad
\mathbf b_i = \frac{\partial \mathbf p_i}{\partial \xi}\Big|_{\hat\xi_M}, \qquad b_i = \lVert \mathbf b_i \rVert ,
```

and with $\hat{\mathbf l}_i = \mathbf b_i / b_i$ the unit epipolar direction and $\hat{\mathbf n}_i$ its
normal, each frame gives one scalar that depends on depth and one that does not:

```math
e_i = \hat{\mathbf l}_i^\top \bigl( \mathbf q_i - \mathbf p_i(\hat\xi_M) \bigr) = - b_i\, \epsilon_M + \epsilon_i^{\parallel}, \qquad
o_i = \hat{\mathbf n}_i^\top \bigl( \mathbf q_i - \mathbf p_i(\hat\xi_M) \bigr) = \epsilon_i^{\perp} .
```

For lateral camera motion $b_i \approx f B_i$, focal length in pixels times the baseline across the
ray. The linearisation is exact in $\xi$ for motion parallel to the image plane, which is the nadir
strip to first order; otherwise one or two Gauss-Newton steps from $\hat\xi_M$ remove it.

### 2.2 The parallax gap and its error bar

Weighted least squares on the along-line residuals gives the **parallax gap**

```math
r_T = - \frac{\sum_{i} b_i\, e_i / \sigma_i^2}{\Lambda_T} = \epsilon_M - \epsilon_T, \qquad
\Lambda_T = \sum_{i \in \mathcal T(u)} \frac{b_i^2}{\sigma_i^2}, \qquad \operatorname{Var}(\epsilon_T) = \Lambda_T^{-1} .
```

No explicit triangulation is needed: the witness is the mean displacement of the track from where the
monocular depth says it should be, in units of parallax. Three properties follow.

- *The error bar is per pixel and comes from geometry.* In depth units $\sigma_{d,T} = d^2 \sigma_{\mathrm{tr}} / (f B_{\mathrm{eff}})$ with $B_{\mathrm{eff}}^2 = \sum_i B_i^2$. In aerial terms, with ground sampling distance $g = d / f$, this is $\sigma_{d,T} = \sigma_{\mathrm{tr}}\, g\, d / B_{\mathrm{eff}}$: tracker noise in ground units times the height-to-baseline ratio.
- *Track length enters as the baseline it realises.* For $L$ frames at spacing $\Delta$ centred on the reference and independent tracker errors, $\sum_i B_i^2 \approx \Delta^2 L^3 / 12$: information grows with the cube of track length. If the tracker error is a random walk along the track (drift), with per-frame increment variance $\sigma_w^2$, the information grows only as $L^2$, and $\Lambda_T$ must be computed by generalised least squares with that covariance. Which law holds is an empirical property of the tracker.
- *A snap offset cancels on centred tracks.* If the tracker locks onto a feature displaced by a constant $\mathbf c$ from the query point, $e_i$ gains the constant $\hat{\mathbf l}_i^\top \mathbf c$. Fitting an intercept with the slope (regressing $e_i$ on $b_i$ signed along the strip) removes it; for tracks centred on the reference the intercept is orthogonal to the slope and costs nothing.

### 2.3 Ancillary statistics: the witness validates itself

A track of $L$ frames supplies $2L$ scalars. One linear combination carries the depth. The remaining
$2L - 1$ are, under the model, free of both the depth and the monocular error:

```math
A_\perp = \sum_{i} \frac{o_i^2}{\sigma_i^2} \sim \chi^2_{L}, \qquad
A_\parallel = \sum_{i} \frac{(e_i + b_i\, r_T)^2}{\sigma_i^2} \sim \chi^2_{L-1} .
```

They are *ancillary* for the judged quantity: their distribution depends on the tracker and the scene
dynamics only. This is the property that depth consensus never had, where every residual mixes the
error of the judged depth map with the errors of the maps it is compared with. The ancillaries serve
three purposes, none of which uses the monocular depth.

1. **Abstention.** A track with $A_\perp + A_\parallel$ above a $\chi^2_{2L-1}$ quantile has left the static rigid model: a moving object, a non-rigid reflection (rippled water, a highlight on a curved surface), an occlusion, or a tracker failure. The pixel abstains.
2. **Noise scale.** $\hat\sigma_{\mathrm{tr}}^2 = \sigma^2 (A_\perp + A_\parallel) / (2L - 1)$, pooled per stratum, calibrates $\Lambda_T$ on the scene itself. The error bar of the witness is estimated, not assumed.
3. **Isotropy test.** $\bigl(A_\parallel / (L-1)\bigr) / \bigl(A_\perp / L\bigr)$ is $F$-distributed under isotropic tracker noise. In a strip all epipolar lines are nearly parallel to the flight direction, so a ratio above one in a stratum says that the tracker is less certain *along* the epipolar direction there (the aperture problem on edges parallel to the flight line), which is exactly the component the depth depends on.

### 2.4 What the ancillaries cannot see

Any track error proportional to the parallax gain, $\boldsymbol\epsilon_i = \mathbf b_i\, c$, lies in the
column space of the depth regressor. It shifts $r_T$ by $c$ and leaves $A_\perp$ and $A_\parallel$
untouched. Such an error is, for the witness, what a fault repeated by every view is for depth
consensus. Its physical sources are specific:

- **Planar reflections.** The mirror image of a static scene in a plane (still water, a glass facade) is itself a static rigid scene. Tracks on it satisfy the epipolar constraint exactly and triangulate consistently, behind the reflecting surface. The epipolar check does not flag them. If the monocular depth places the surface correctly, the witness accuses a correct prior.
- **Motion along the epipolar direction at constant speed.** With constant camera speed, an object moving along the flight direction produces a displacement proportional to the baseline: a wrong depth that fits perfectly. Traffic on a road parallel to the strip is the aerial case.
- **Borrowed parallax.** On a texture-free patch, or on an edge parallel to the epipolar line, a learned tracker fills in the motion from its surroundings. The track then carries the parallax of the neighbouring surface, consistently across frames. At an occlusion edge this is foreground fattening; inside a uniform roof it is a planar interpolation of the border depth, which is often right and is in any case a prior, not a measurement.
- **Pose error with a depth-like pattern.** A baseline scale error rescales all inverse depths of a view and is absorbed by the per-view alignment; other pose errors appear in $A_\perp$ as a smooth field and are removed by a low-order fit per frame pair before the ancillaries are formed.

Two partial remedies exist without labels. A cross strip changes the epipolar direction and the time
gap: borrowed parallax along one direction and constant-speed movers become inconsistent between the
two strips' gaps, and are flagged; a planar reflection passes this test too. And the sign: a virtual
image always lies *behind* the reflecting surface, $r_T > 0$ in inverse depth for every view, so on
strata with reflective appearance a one-sided fault model discounts exactly that sign. Whether the
remaining false accusations are tolerable is an empirical question; they are the first thing a small
labelled set should measure (Section 9).

### 2.5 Assumptions

- **T1** The scene is static and rigid and the poses are correct up to their stated covariance.
- **T2** Tracker errors have a known covariance structure up to scale (independent across frames, or a random walk), isotropic in the image.
- **T3** No tracker error is proportional to the parallax gain (no source of Section 2.4).
- **T4** The witness error $\epsilon_T$ is independent of the monocular error $\epsilon_M$ given the stratum.

T1 and T2 are tested per track and per stratum by the ancillaries. T3 is tested only across strips and
not at all for planar reflections. T4 is an assumption about two different frozen networks (a depth
network and a tracker) with different inputs and training; it is violated to the extent that both fail
on the same appearance in a correlated *direction*, and it would be violated strongly if the judged
depth came from a multi-view network that uses the same parallax.

## 3. What Is Identified Without Labels

### 3.1 The requirement of three independent views is met inside the track

A latent quantity observed through noisy channels is identified, in general, when at least three
channels have mutually independent errors. For binary detectors this is the latent-class result (the
covariance of two detectors' votes is the product of their loadings, $C_{gh} = u_g u_h$, and three
detectors give $u_1^2 = C_{12} C_{13} / C_{23}$); for continuous measurements it is the three-cornered
hat. A formulation with two evidence groups of unknown accuracy fails this requirement, and no
label-free estimate exists for it: only the product of the two accuracies is identified.

The video supplies the missing channel through temporal redundancy. Split the track at the reference
frame into its forward and backward halves, with gaps $r_T^{+}$, $r_T^{-}$ and informations
$\Lambda_T^{+}$, $\Lambda_T^{-}$. Under T2 to T4,

```math
r_T^{+} = \epsilon_M - \epsilon_T^{+}, \qquad r_T^{-} = \epsilon_M - \epsilon_T^{-}, \qquad
\epsilon_M,\ \epsilon_T^{+},\ \epsilon_T^{-}\ \text{mutually independent} .
```

The monocular depth, the forward half-track and the backward half-track are three measurements of
$\xi$ with independent errors. Consequences:

```math
\operatorname{Cov}\bigl(r_T^{+}, r_T^{-}\bigr) = \operatorname{Var}(\epsilon_M), \qquad
\operatorname{Var}\bigl(r_T^{+} - r_T^{-}\bigr) = \operatorname{Var}(\epsilon_T^{+}) + \operatorname{Var}(\epsilon_T^{-}) .
```

The second identity contains no monocular quantity. It checks the predicted
$1/\Lambda_T^{+} + 1/\Lambda_T^{-}$ and is a third calibration of the witness besides the ancillaries.
The first gives the variance of the monocular error per stratum. More is true: two measurements of one
latent variable with mutually independent errors identify the *entire distribution* of the latent
variable and of both errors, without a shape assumption (Kotlarski's lemma; conditions: zero-mean
errors and non-vanishing characteristic functions). Hence:

**Proposition (label-free identification of the monocular error law).** Under T1 to T4, for every
stratum $s$ with enough two-sided tracks, the distribution of the monocular depth error
$p_s(\epsilon_M)$ is identified from the joint distribution of $(r_T^{+}, r_T^{-})$. In particular the
fault prevalence $\pi_s = \Pr(\lvert \epsilon_M \rvert > \tau \mid s)$ is identified for any threshold
$\tau$, with no fault labels, no assumed inlier shape and no assumed outlier density.

This replaces the two things the contamination model had to assume (a Gaussian core and a flat
outlier density) and the thing the two-group model could not supply (a third independent channel).
The fault class also has a fixed meaning: it is an event of the signed depth error, not "whatever
binary variable makes the cues independent".

### 3.2 The practical version: deconvolution with a known kernel

Kotlarski's construction is statistically inefficient. In practice the witness noise is treated as
known, $\epsilon_T \sim \mathcal N(0, \Lambda_T^{-1})$ with the scale calibrated by Section 2.3, and
the full-track gap is modelled as the monocular error law convolved with a known, per-pixel kernel:

```math
p\bigl(r_T \mid s, \Lambda_T\bigr) = \int p_s(\epsilon)\, \mathcal N\bigl(r_T;\ \epsilon,\ \Lambda_T^{-1}\bigr)\, d\epsilon .
```

With the contamination form $p_s(\epsilon) = (1 - \pi_s)\, \mathcal N(0, \varsigma_s^2) + \pi_s\, p_{\mathrm{out}}(\epsilon)$
the two class-conditional densities of the gap are

```math
g^{0}(r_T \mid s, \Lambda_T) = \mathcal N\bigl(r_T;\ 0,\ \varsigma_s^2 + \Lambda_T^{-1}\bigr), \qquad
g^{1}(r_T \mid s, \Lambda_T) = \bigl(p_{\mathrm{out}} * \mathcal N(0, \Lambda_T^{-1})\bigr)(r_T) ,
```

where $\varsigma_s$ is the inlier spread of the monocular depth in stratum $s$. The difference from a
contamination mixture on a consensus residual is that the heteroscedastic part of the null,
$\Lambda_T^{-1}(u)$, is known pixel by pixel. A mixture with a known, varying noise level is identified
by that variation: pixels with large $\Lambda_T$ show the monocular error law almost directly, pixels
with small $\Lambda_T$ show it blurred, and both must be explained by the same $p_s$. The split-track
identities are kept as the test of the kernel.

### 3.3 The continuous latent, and why binary conditional independence is the wrong model here

The consensus gap and the parallax gap are both residuals of the same monocular depth, so they share
its error as a continuous quantity. Given only the binary fault state they remain correlated; given
the signed error they are independent. The correct latent variable is $\epsilon_M$:

```math
p(r_T, r_C \mid s) = \int p_s(\epsilon)\ \mathcal N\bigl(r_T;\ \epsilon,\ \Lambda_T^{-1}\bigr)\ p\bigl(r_C \mid \epsilon\bigr)\ d\epsilon ,
```

and the indicator is a functional of the posterior of $\epsilon_M$,
$\zeta = \Pr(\lvert \epsilon_M \rvert > \tau \mid r_T, r_C, \boldsymbol\phi)$, or the posterior weight
of the outlier component. The latent-class formulas are recovered when each residual is reduced to a
vote, at the price of a class-conditional dependence that is then built in by construction.

### 3.4 The repetition coefficient is identified

Write the consensus gap in inverse depth as $r_C = \hat\xi_M - \hat\xi_C = \epsilon_M - \epsilon_C$,
with $\hat\xi_C$ the cavity depth. Define the repetition coefficient by projection,

```math
\epsilon_C = \varrho\, \epsilon_M + \eta_C, \qquad \varrho = \frac{\operatorname{Cov}(\epsilon_C, \epsilon_M)}{\operatorname{Var}(\epsilon_M)}, \qquad \eta_C \perp \epsilon_M .
```

$\varrho = 0$ means the other views do not share the error of view $j$; $\varrho = 1$ means that the
surface fused from the other views carries the same error, and consensus sees nothing:
$r_C = (1 - \varrho)\, \epsilon_M - \eta_C$. With $\epsilon_T$ independent of $\epsilon_M$ and $\eta_C$,

```math
\operatorname{Cov}(r_C, r_T) = (1 - \varrho) \operatorname{Var}(\epsilon_M), \qquad
\operatorname{Var}(r_T) = \operatorname{Var}(\epsilon_M) + \overline{\Lambda_T^{-1}} ,
```

```math
1 - \hat\varrho = \frac{\widehat{\operatorname{Cov}}(r_C, r_T)}{\widehat{\operatorname{Var}}(r_T) - \overline{\Lambda_T^{-1}}} ,
```

an errors-in-variables slope with known error variance. For a single region $\mathcal R$ with a
significant mean parallax gap, $1 - \hat\varrho_{\mathcal R} = \bar r_{C, \mathcal R} / \bar r_{T, \mathcal R}$.

This turns the premise of the work into a measurable quantity. "Depth consensus is blind to faults
that every view repeats" becomes: the share $\varrho$ of each fault, as a function of the left-out
block size $B$, estimated on real data without ground truth. A curve $\hat\varrho(B)$ that falls to
zero once the block exceeds a few frames says that repetition is local to a strip and that a longer
block cures it. A plateau above zero is the fault that every view repeats, and its height is the
fraction of the error that no consensus method can see. On a scene with exact depth the same curve is
computed from true errors and checks the estimator.

### 3.5 Summary of identification

| Quantity | Identified from | Rests on |
| --- | --- | --- |
| tracker noise scale, per stratum | ancillaries $A_\perp$, $A_\parallel$ | T1, covariance structure of T2 |
| anisotropy along the epipolar direction | ratio $A_\parallel / A_\perp$ | T1 |
| witness error variance | $\operatorname{Var}(r_T^{+} - r_T^{-})$ | T2 |
| monocular error law $p_s(\epsilon_M)$, prevalence $\pi_s$ | $(r_T^{+}, r_T^{-})$, or $r_T$ with known kernel | T2, T3, T4 |
| repetition coefficient $\varrho(B)$ | $\operatorname{Cov}(r_C, r_T)$ | T3, T4 |
| fault posterior on tracked pixels | Bayes' rule with the above | all |
| epipolar-consistent witness bias (Section 2.4) | not identified within a strip; partly across strips | |
| anything on pixels with no track and no feature overlap | not identified | |

## 4. Indicator and Estimators

### 4.1 Per-pixel sufficient statistics and cost

The witness needs five numbers per reference pixel, accumulated in one pass over the frames of its
track: $\sum b_i^2/\sigma_i^2$, $\sum b_i e_i/\sigma_i^2$, $\sum e_i^2/\sigma_i^2$,
$\sum o_i^2/\sigma_i^2$ and $L$ (ten if forward and backward halves are kept apart, plus the sums for
an intercept). Tracks need not be stored. The geometry costs one projection Jacobian per pixel and
frame. The tracker's forward passes dominate; everything else is linear in pixels times track length.

### 4.2 Indicator on tracked pixels

```math
\operatorname{logit} \zeta = \operatorname{logit} \pi(\boldsymbol\phi)
+ \log \frac{g^{1}(r_T \mid s, \Lambda_T)}{g^{0}(r_T \mid s, \Lambda_T)} \quad \text{(parallax, where a track exists and passes the ancillary test)} ,
```

optionally with the consensus residual through the joint density of Section 3.3 and the estimated
$\varrho$. As $\Lambda_T \to 0$ the likelihood ratio tends to one and $\zeta \to \pi(\boldsymbol\phi)$:
there is no hard boundary between tested and untested pixels, and an abstaining pixel is the limit of
an uninformative one.

### 4.3 Self-supervised fit of the head: pseudo-labels with known noise

The head on frozen features is fitted by maximising the marginal likelihood of the gaps,

```math
\mathcal J(\theta, \varsigma) = - \sum_{n} \omega\, \log \Bigl[ \bigl(1 - \pi_\theta(\boldsymbol\phi_n)\bigr)\, g^{0}(r_{T,n} \mid s_n, \Lambda_{T,n}) + \pi_\theta(\boldsymbol\phi_n)\, g^{1}(r_{T,n} \mid s_n, \Lambda_{T,n}) \Bigr] ,
```

with $\omega$ a deflation for spatial correlation. Its gradient with respect to the head's logit is
$\zeta_n - \pi_n$: the head is regressed onto the posterior. Three remarks.

- *This is learning from noisy labels whose noise is known per instance.* A thresholded gap is a label with miss and false-alarm rates that depend on $\Lambda_T$ through $g^0$, $g^1$. Because those rates are known, the marginal likelihood is the consistent (forward-corrected) objective; no clean subset has to be selected by hand.
- *Selection is soft and automatic.* The Fisher information of pixel $n$ about its prior logit is $\mathbb E(\zeta_n - \pi_n)^2$. Define the label information

```math
\iota_n = \frac{\mathbb E\,(\zeta_n - \pi_n)^2}{\pi_n (1 - \pi_n)} \in [0, 1] ,
```

  zero for an uninformative witness and one for a perfect one. A pixel contributes to the fit in proportion to $\iota_n$. "Well-conditioned triangulation" is $\iota_n$ near one; no threshold on parallax is needed, and the question of selection bias is the question of how $\iota$ depends on the features (Section 6).
- *The head cannot copy its target.* It receives frozen features only, never the gap. The degenerate solution of an amortised mixture, in which the prior reproduces the pixel's own residual, requires the features to encode the *noise* of the witness. This is excluded when the features come from a model that does not share the tracker's computation, and it must be re-examined for video or multi-view features (Section 7). Cross-fitting by ground tile remains necessary for per-scene fitting, because all views of a tile share its faults.

The inlier spread $\varsigma_s$ is fitted jointly per stratum. A nonparametric $p_s$ (a histogram
deconvolved with the known kernel) is the check on the contamination form.

### 4.4 Optional labels

A small labelled set from a scene with exact depth is not needed for identification under T1 to T4. It
is needed to test T3 and T4 themselves: the false-accusation rate of the witness per stratum
(reflective strata above all), the dependence between witness and monocular errors, and the
calibration of $\zeta$. In the semi-supervised likelihood the labelled pixels enter with $\zeta$
clamped; their weight is chosen on held-out labelled likelihood.

## 5. Why Keep the Monocular Prior: the Leverage-Power Relation

The objection is that a witness good enough to judge the prior is good enough to replace it. The model
gives a quantitative answer, and it is not entirely comfortable.

**How good the witness is.** Illustration with assumed inputs: ground sampling distance $g = 5$ cm,
tracker noise $\sigma_{\mathrm{tr}} = 1$ px, a point visible in five frames with baselines
$\pm 0.2, \pm 0.4$ of the flying height, so $B_{\mathrm{eff}} \approx 0.63\, d$. Then
$\sigma_{d,T} = \sigma_{\mathrm{tr}}\, g\, d / B_{\mathrm{eff}} \approx 8$ cm. An aligned monocular depth
with an inlier error of one percent of a 100 m flying height has $\sigma_{d,M} = 1$ m. The information
ratio is $\vartheta = \Lambda_T / \Lambda_M \approx 160$. On a tracked, static, diffuse pixel of a
normal aerial block the witness is two orders of magnitude more informative than the prior. The
statement that triangulation is "noisy at small baseline" applies to short tracks, far range and strip
ends, not to the typical tracked pixel.

**Where a fault in the prior matters.** If the triangulated depth is fused with the prior, a fault of
size $\delta$ in the prior that goes undetected biases the fused depth by $\delta / (1 + \vartheta)$:
the prior's leverage is $1 / (1 + \vartheta)$.

**Where the witness can detect it.** The non-centrality of the gap is
$\delta^2 / (\Lambda_T^{-1} + \Lambda_M^{-1}) = \delta^2 \Lambda_M\, \vartheta / (1 + \vartheta)$.

**The product.** The squared error avoided by detecting the fault is $\delta^2 / (1 + \vartheta)^2$.
For small faults, where power is proportional to non-centrality, the expected gain of direct testing is
proportional to

```math
\frac{\vartheta}{(1 + \vartheta)^3} , \qquad \text{maximal at } \vartheta = \tfrac12 .
```

For a large fault of $k$ inlier standard deviations, detection at non-centrality $n_\alpha$ needs
$\vartheta \ge n_\alpha / (k^2 - n_\alpha)$, and the gain $\delta^2/(1+\vartheta)^2$ decays above
$\vartheta \approx 1$. With $k = 10$ and $n_\alpha = 16$ the useful band is roughly
$0.2 \le \vartheta \le 3$. **Direct testing of the prior by the witness pays only where the two are
about equally informative.** Below the band the witness has no power; above it the prior has no
leverage.

This sorts pixels into three regimes, and it locates the contribution.

| Regime | Where | Role of the witness | Role of the prior |
| --- | --- | --- | --- |
| $\vartheta \gg 1$ | tracked, diffuse, good baseline | supplies labels at almost no noise ($\iota \approx 1$) | nearly irrelevant if the witness is fused |
| $\vartheta \approx 1$ | short tracks, far range, strip ends, low overlap | tests the prior directly | matters, and can be tested |
| $\vartheta \ll 1$ or no track | texture-free, reflective, occluded, single coverage | none | the only depth evidence |

The prior is kept because of the third row: it is the only dense depth where tracking abstains, and
those are also the surfaces where monocular models fail most. The witness is strongest in the first
row, where the prior is least needed. The method therefore stands or falls with *transfer*: learning
in the first row where to distrust the prior, and applying it in the third. A paper that reports
detection quality on tracked pixels has measured the regime in which the answer does not matter. The
evaluation has to be stratified by $\vartheta$, and the headline number is the one for $\vartheta \ll 1$.

**Judge or data.** Estimation theory says to fuse the triangulated depth as a sparse heteroscedastic
observation and to gate the prior with leverage $1/(1+\vartheta)$. The case for using the witness as a
judge only rests on its error structure, not on its variance: the errors of Section 2.4 are gross,
systematic and pass the witness's own checks. As a judge, a false witness costs a gated prior, which
is a loss of constraint; as data, it costs a wrong surface. That argument justifies caution on
reflective strata. It does not justify discarding centimetre-level depth on diffuse texture, and a
design that does so should expect the objection to stand. The consistent position is: fuse the witness
where its ancillaries pass and the stratum is not reflective, judge everywhere, and report the
reconstruction gain separately for the regions the witness does not reach.

## 6. Transfer from Tracked to Untracked Pixels

### 6.1 A selective-labels problem

The head is trained where $\iota$ is large and used where it is small. Let $o = \mathbb 1[\iota > \iota_0]$
and $e(\boldsymbol\phi) = \Pr(o = 1 \mid \boldsymbol\phi)$ the propensity. The target
$\Pr(f = 1 \mid \boldsymbol\phi)$ learned on $o = 1$ equals the one needed on $o = 0$ under two
conditions.

- **S1 (labels missing at random given features).** $f \perp o \mid \boldsymbol\phi$: among pixels that look the same to the frozen encoder, those the tracker can follow are faulty as often as those it cannot.
- **S2 (overlap).** $e(\boldsymbol\phi) > 0$ wherever the head is applied.

Under S1 and S2 the risk on untracked pixels is an importance-weighted risk on tracked ones, with
weights $(1 - e(\boldsymbol\phi)) / e(\boldsymbol\phi)$, and held-out likelihood can be reported for the
untracked distribution without labels.

### 6.2 A label-free map of where the head extrapolates

The propensity is estimated by a second head on the same frozen features, trained to predict $o$. It
needs no fault labels. It partitions the untracked pixels:

- *interpolation zone,* $e(\boldsymbol\phi) \ge e_{\min}$: appearances that are sometimes tracked. Transfer is supported by data, conditional on S1.
- *extrapolation zone,* $e(\boldsymbol\phi) < e_{\min}$: appearances that are essentially never tracked. The head's output there is determined by the geometry of the frozen feature space and by nothing in the training signal.

The effective sample size of the importance weights, $(\sum w)^2 / \sum w^2$, states how many tracked
pixels stand behind the prediction for the untracked ones. The method can thus report, per scene and
without ground truth, what fraction of its untracked area is interpolation. Open water and large glass
are expected in the extrapolation zone; weakly textured roofs and asphalt, where a learned tracker
follows some pixels and not others, in the interpolation zone. Results must be reported per zone, and
a claim for the extrapolation zone can only be empirical.

### 6.3 Where S1 fails

S1 fails when trackability and faultiness are linked beyond what the features encode.

- *Trackable pixels inside an untrackable material are atypical.* On a glass facade the tracker holds on frames, dirt and reflections. The first are not glass, the last are virtual images (Section 2.4). The pseudo-labels of the stratum are then drawn from the wrong population, in the direction of accusing the prior.
- *Occlusion.* Short tracks cluster at depth edges, where monocular depth is also worst. Distance to the nearest edge of the monocular depth map is available on every pixel and should be a covariate of the head for this reason.
- *Feature resolution.* Patch features mix a textured and an untextured surface inside one patch; the stratum then contains both, and labels come from the textured part.

### 6.4 Depth consensus as the examiner of the transferred head

The two kinds of evidence abstain in different places. Parallax abstains without texture; consensus
needs no texture and is blind only to the repeated share $\varrho$ of a fault. On untracked pixels the
cavity residual still exists. This gives a label-free test of the transfer where it is most in doubt:

- among untracked pixels, the rate of consensus outliers must increase with the head's score;
- quantitatively, with $\varrho$ transported from tracked pixels of neighbouring strata, the expected rate is the head's probability times the consensus detection rate $p_1(\varrho)$ plus the false-alarm rate; a head that scores untracked surfaces at random fails this check.

The head is trained by the witness that sees repeated faults and examined by the witness that works
without texture. What stays beyond any label-free check is the intersection: a fault that is repeated
by all views *and* lies on a surface that is never tracked. For that case the claim rests on the
transfer assumption and on labelled evaluation alone.

## 7. Frozen Features: Which, and What They May See

**Image features for the prior head.** Frozen self-supervised patch features of the reference frame.
They encode surface type and are computed by a network unrelated to both the depth model and the
tracker, so they satisfy the no-copy condition of Section 4.3.

**Video or multi-view features.** A video encoder, or the encoder of a feed-forward multi-view geometry
model, has parallax inside its features. Two different uses must be kept apart.

- As input to the *prior head* $\pi(\boldsymbol\phi)$, such features let the head recognise motion patterns that go with tracker or depth failure (non-rigid water, specular flicker). The risk is that they also encode the tracker-like displacement of the pixel, and with it a correlate of the witness noise; then the head can partially copy its target. The test is the usual one: cross-fitted held-out likelihood must not improve when the head is given the pixel's own track statistics.
- As input to a *gap head* that also receives the monocular depth and regresses the signed gap, $\hat r_\theta(u) \approx \mathbb E[\epsilon_M \mid \text{features}]$, trained with the heteroscedastic loss $\sum_n \Lambda_{T,n}\, (r_{T,n} - \hat r_\theta)^2$ plus the outlier component. This predicts sign and size of the error, not only its presence, and extends the witness itself to untracked pixels. It is also, in effect, a depth head on a multi-view model. If it works well, the honest conclusion is that the feed-forward multi-view depth should replace the monocular prior, and T4 fails for that prior because witness and judged depth then use the same parallax.

**The depth model's own encoder.** A strong predictor of that model's failures and admissible for the
prior head, since the tracker does not share its computation.

**Strata.** Clusters of the image features, fixed before any gap is seen, carry $\pi_s$,
$\varsigma_s$, the tracker noise scale and the anisotropy ratio. Frozen features cannot adapt their
strata to the evidence, which is what keeps per-stratum estimates honest.

## 8. Is There a New Idea Here

Stated plainly: checking a monocular depth against multi-view triangulation is not new, and
confidence estimators trained on self-generated consistency labels exist for stereo and multi-view
stereo (Section 12). What this formulation adds, in the part of the problem treated here, is the
following; novelty is asserted relative to the literature as recalled and has not been searched
systematically.

1. **A witness with ancillary statistics, and identification that follows from them.** Per track, $2L - 1$ residuals are free of the judged quantity. They calibrate the witness on the scene, and the forward and backward half-tracks supply the third independent channel that makes the law of the monocular error, and the fault prevalence, identified without labels and without an assumed shape (Section 3.1). Robust reweighting of depth consensus has no counterpart to this: its residual has no component that is free of the hypothesis under test.
2. **The repetition coefficient as a label-free measurement** (Section 3.4). The share of a monocular fault that the other views repeat, as a function of the left-out block, measured on real flights. It quantifies the blind spot of every consensus-based gate, including robust losses and distractor masks, and is of interest independently of the detector.
3. **The leverage-power relation and the complementary-abstention test** (Sections 5 and 6.4). The first says where judging a prior with a witness can matter at all and forces the evaluation onto the untracked regime. The second uses consensus to examine, without labels, the transfer that the method depends on.

The first two are sharp and, to the extent recalled, not in the literature in this form. The third is
an argument about evaluation more than a method. If the experiments show that $\varrho$ is negligible
at practical block sizes, item 2 becomes a negative result about the premise, which is still a
finding, and the detector loses its reason.

## 9. What Would Refute the Approach, and the Cheapest Tests

All tests use a scene with exact depth for evaluation; the first needs no tracker.

1. **Repetition is negligible.** From exact depth, compute the monocular error of every view, associate errors of the same ground point across views, and compute $\varrho(B)$. If $\varrho(B) \approx 0$ for a block of a few frames on real depth-model errors, consensus with a longer block suffices. Cost: hours on cached depth maps. This is the first test.
2. **Faults lie where nothing is tracked and nothing overlaps.** Synthesise ideal tracks from exact depth with added noise $\sigma_{\mathrm{tr}}$ and a texture-based visibility rule; then run a real frozen tracker on one strip. Measure the fraction of true fault area with $\iota > \iota_0$, in the interpolation zone, and in the extrapolation zone. If most fault area is in the extrapolation zone, the method reduces to an appearance probe.
3. **Transfer fails.** Train a linear probe on frozen features with parallax pseudo-labels on tracked pixels; evaluate on untracked true faults; compare with the same probe trained on true labels (the ceiling) and with a probe on consensus pseudo-labels. No gain over consensus pseudo-labels on untracked pixels means the witness adds nothing where it matters.
4. **The witness accuses correct priors.** Precision of gap-based flags per stratum against exact depth, with the sign of the gap; a reflective stratum dominated by one-sided false accusations refutes T3 there and requires the one-sided model or labels.
5. **The prior adds nothing once triangulation is fused.** Reconstruct with triangulated depth alone, with the prior added ungated, and with the prior under an oracle gate. If the oracle gate does not improve the untracked regions, no indicator can.
6. **The identification assumptions fail their own tests.** $\operatorname{Var}(r_T^{+} - r_T^{-})$ far from its prediction, an anisotropy ratio far from one, or a head whose held-out likelihood improves with the pixel's own track statistics.

Tests 1 and 5 can end the project before a tracker is run.

## 10. What This Theory Cannot Deliver

1. **Epipolar-consistent witness errors.** Planar reflections, constant-speed motion along the strip and borrowed parallax bias the gap and pass every internal check. The blind spot moves from "repeated by all depth maps" to "repeated by all frames of a track"; it is smaller and physically specific, and it is not gone.
2. **Evidence on untracked, never-tracked appearance.** In the extrapolation zone the indicator is an appearance prior with no supporting data. Glass and water are the expected content of that zone and also the expected content of the fault set.
3. **Verification of S1 and T4 without labels.** Both are independence assumptions about what is not observed. Section 6.4 tests transfer only for the non-repeated share of faults.
4. **Independence of tracker errors across frames.** Learned trackers smooth over time. The forward-backward split tolerates drift; it does not tolerate an error common to both halves, which is the case of item 1.
5. **Calibrated probabilities.** $\zeta$ is a probability under the model; a stated false-alarm rate needs a separate calibration on labelled clean regions or conformal scores, with regions as units.
6. **Precision per scene.** Prevalence and the tails of $p_s$ are estimated from the number of independent fault regions per stratum, not from the number of pixels.
7. **A reason to keep a monocular prior where tracking works.** By Section 5 there is none beyond densification; the contribution has to be demonstrated where tracking fails.
8. **Robustness to pose error beyond the stated covariance.** A wrong pose is a wrong witness for every pixel of the frame; the ancillaries detect it only through its across-line component.

## 11. Relation to the Earlier Models

- *Contamination mixture on the cavity residual.* Same form, with the consensus residual in place of the parallax gap. Its null is not known pixel by pixel and its identification rests on the assumed inlier shape; it is blind to the repeated share $\varrho$. It is retained as the second witness and as the examiner of Section 6.4.
- *Two evidence groups anchored by labels.* The latent-class structure with two groups has no label-free mode because only $u_1 u_2$ is identified. Here the second group is the parallax gap, its accuracy is not a free parameter but follows from $\Lambda_T$, and the third channel is supplied by the split track; labels move from a requirement to a test of assumptions.
- *Plane-sweep photo-consistency.* A matching cost profile gives a depth optimum with a curvature-based variance but no ancillary residuals: there is no component of the evidence that is free of depth. Tracks have $2L - 1$.

## 12. Attributions and Availability

**Statistical results, recalled from memory and not checked against the sources.** Identification of
the distribution of a latent variable from two measurements with independent errors: Kotlarski (1967);
the existence of the lemma and its use in measurement-error models was confirmed by a literature
search, its exact conditions were not. Three-cornered hat in metrology and triple collocation in
remote sensing (Stoffelen 1998) for error variances of three systems. Latent-class identifiability
with three conditionally independent detectors: Dawid and Skene (1979); Allman, Matias and Rhodes
(2009). Learning with class-conditional label noise and forward loss correction: Natarajan et al.
(2013); Patrini et al. (2017). Propensity weighting under missing-at-random selection: Rosenbaum and
Rubin (1983). Self-supervised confidence labels for stereo from multi-view contradiction: Mostegel et
al. (2016); Tosi et al. (2017). Triangulation uncertainty proportional to $d^2 / (f B)$ is textbook.

**Frozen models.** A search confirmed that the following exist as published work with public
repositories; their accuracy on aerial imagery, their behaviour at the large inter-frame displacements
of photogrammetric blocks as opposed to video, their licences and their memory cost at full resolution
were *not* verified.

| Role | Candidates | Checked |
| --- | --- | --- |
| point tracker | CoTracker3; TAPNext; AllTracker (dense, high resolution) | existence confirmed by search; properties recalled |
| optical flow | RAFT, SEA-RAFT | recalled only |
| image features | DINOv2, DINOv3 | recalled; DINOv3 seen in search results only through secondary sources |
| video encoder | V-JEPA 2 | repository confirmed by search; suitability of its patch features for dense heads recalled |
| multi-view geometry encoder | VGGT; MapAnything; MASt3R | VGGT (CVPR 2025) and MapAnything confirmed by search; MASt3R recalled |

Whether any of these trackers reports a calibrated per-point uncertainty is not known here; the theory
does not need one, since the noise scale is estimated from the ancillaries. Whether the exact-depth
aerial scenes available for evaluation contain physically rendered reflections on glass and water, on
which Section 2.4 depends, was not checked.
