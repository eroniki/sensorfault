# Theory: a Monocular Depth Prior Judged on Surface Entities in Depth-Stabilised Video

This note develops the statistical model of the fault indicator for posed aerial video, on backbones
that are native to video and dense prediction. One monocular depth map per frame is the dense prior of
a 2D Gaussian surfel reconstruction. A frozen video segmenter supplies *masklets*, regions followed
through a clip; a frozen video encoder supplies spatiotemporal tokens. The unit that receives a verdict
is the masklet, a surface entity, and the per-frame, per-pixel deviation is a second-order term.

The note does four things. It states which parts of this conception hold as stated and which need
repair (Section 2). It defines one test object, the depth-stabilised clip, of which explicit parallax,
masklet motion and latent comparison are three readings (Sections 3 to 5). It gives the hierarchical
model, what is identified without labels and the estimators (Sections 6 and 7). And it restates the
standing difficulties, harm against detectability, transfer, reflections, in the new units (Sections 8
to 10), ending with what would refute the approach and what was verified about the models.

Nothing here has been measured. Numerical examples use assumed inputs and are marked. Statements about
models and prior work are collected in Section 13 with what was checked.

## 1. Setting and Symbols

Frames $j = 1, \dots, N$ with known poses, in strips. $\hat\xi_{M,j}(u)$ is the aligned monocular
*inverse depth* of pixel $u$ in frame $j$ and $\epsilon_{M,j}(u) = \hat\xi_{M,j}(u) - \xi_j(u)$ its
error. For a point on the ray of $u$, the projection into frame $i$ is $\mathbf p_i(\xi)$, with parallax
gain $\mathbf b_i = \partial \mathbf p_i / \partial \xi$ of length $b_i$ (focal length times the
baseline across the ray, for lateral motion), unit epipolar direction $\hat{\mathbf l}_i$ and normal
$\hat{\mathbf n}_i$. $\mathbf k_u = K^{-1} \tilde u$ is the normalised homogeneous pixel.

| Symbol | Meaning |
| --- | --- |
| $m$, $\mathcal M_{mj}$, $\partial \mathcal M_{mj}$ | masklet, its mask in frame $j$, its boundary |
| $\tilde I_{i \to j}$ | frame $i$ stabilised into frame $j$ by the monocular depth of $j$ |
| $\boldsymbol\epsilon_{mj} \in \mathbb R^3$ | affine coefficients of the monocular error on masklet $m$ in frame $j$ |
| $\boldsymbol\mu_m$, $\boldsymbol\alpha_{mj}$, $\eta_j(u)$ | persistent (surface) term, per-frame term, within-masklet remainder |
| $r_T(u)$, $\Lambda_T(u)$ | pixel parallax gap and its information |
| $\hat{\boldsymbol\epsilon}_{mj}$, $\Lambda_{mj}$ | masklet gap (3-vector) and its information matrix |
| $A_\perp$, $A_\parallel$, $A_m$, $A_\partial$ | ancillary statistics: across-line, along-line, within-masklet, boundary |
| $\mathcal D(\cdot)$, $D(\boldsymbol\delta)$ | latent discrepancy as a function of residual error, and as observed under an injected perturbation $\boldsymbol\delta$ |
| $\vartheta$ | information ratio of witness to prior |
| $\varrho$ | repetition coefficient |
| $\iota$ | label information of a unit, in $[0,1]$ |
| $F_m$, $\zeta_m$ | fault state of masklet $m$ and its posterior |
| $e(\boldsymbol\phi)$ | propensity of a unit with features $\boldsymbol\phi$ to have an informative witness |

## 2. The Conception, Examined

| Claim | Status | Repair |
| --- | --- | --- |
| The verdict belongs to a surface entity; a fault every frame repeats is a property of the surface | Holds, and has an exact statistical form: a random effect per masklet. The repeated fault is the masklet-level term, the per-frame deviation is the residual, and the repetition coefficient is an intraclass correlation (Section 6) | The unit is only right if monocular errors are *low-order within masklets*. That is a hypothesis with a label-free test and a cheap ground-truth test (Sections 6.4, 11) |
| Masklet motion is parallax that needs no interior texture; a plane-induced homography has three unknowns | Half right. The outline gives one scalar per boundary point and frame (normal displacement), enough for three unknowns. But a boundary carries the parallax of the surface that *owns* it, which is the nearer one. A lake fringed by trees has an outline that moves with the canopy | Use only owned or co-planar boundary segments, test their mutual consistency, and prefer a weaker assumption: planarity of the *error*, not of the surface (Section 4) |
| Evidence is violation of expectation in latent space: warp by the predicted motion and compare inside a frozen video encoder | The available encoders have no pose-conditioned predictor (Section 13), so the expectation must be imposed in pixel space. A latent distance has no unit, no sign and no error bar. And to first order it is blind without image structure: a uniform surface produces the same tokens whatever its depth | Stabilise the clip with the judged depth; the null is then "a still image". Turn the latent distance into an estimator by a sweep, and measure its information by injected perturbations (Section 5) |
| The head is a dense probe on spatiotemporal tokens | Two different heads are being conflated: a *readout* of the witness, which must see motion, and a *prior*, which must not see the witness | Train the readout by injection, where the label is exact; train the prior on verdicts, on tokens that cannot contain the residual motion (Section 7) |
| Survey stills at 80% overlap move too much for video models | True for the raw clip. Not true for the stabilised clip, whose motion is the residual parallax of the depth error only | Run encoder, flow and mask propagation on the stabilised clip (Section 3) |

## 3. The Test Object: the Depth-Stabilised Clip

### 3.1 Definition and null hypothesis

For a reference frame $j$ and a window of frames $i \in \mathcal W_j$, resample every frame into the
reference view through the monocular depth of the reference:

```math
\tilde I_{i \to j}(u) = I_i\bigl( \mathbf p_i(\hat\xi_{M,j}(u)) \bigr) , \qquad i \in \mathcal W_j .
```

This uses the poses and the one depth map under test, nothing else. It is a backward warp: no holes,
and at occlusions a duplicated foreground.

**H0.** If the depth of $u$ is correct and the surface is static, diffuse and visible in frame $i$,
then $\tilde I_{i \to j}(u) = I_j(u)$ up to image noise and resampling. *The stabilised clip of a
correct depth map is a still image repeated.*

**Under a depth error.** Sampling at $\mathbf p_i(\hat\xi_M) = \mathbf p_i(\xi) + \mathbf b_i\, \epsilon_M$
shows content that belongs, in the reference, to a pixel displaced by about $\mathbf b_i \epsilon_M$:

```math
\tilde I_{i \to j}(u) \approx I_j\bigl( u + \mathbf b_i\, \epsilon_{M,j}(u) \bigr) .
```

The residual motion of the stabilised clip at $u$ is $-\mathbf b_i\, \epsilon_{M,j}(u)$: along the
epipolar direction, proportional to the signed baseline and to the depth error. A faulty region slides
back and forth in step with the camera, while everything correct stands still.

### 3.2 Consequences

1. **A sharp null for any instrument.** Every reading below asks one question, whether a region of the stabilised clip is static. Departures have a known direction (epipolar) and a known time course (the baseline), which separates them from illumination change and from independently moving objects.
2. **Frame spacing stops mattering to the frozen models.** Survey stills shift by a fifth of the image per frame; dense video by a few pixels. After stabilisation both show only the residual parallax, which is small wherever the depth is roughly right. Flow networks, video encoders and mask propagation then operate in the small-motion regime they were trained in. What frame spacing still sets is the baseline per frame, hence the information per frame, and the amount of occlusion.
3. **Perturbation is exchangeable with error.** Stabilising with $\hat\xi_M + \delta$ adds exactly $-\mathbf b_i \delta$ to the residual motion. For any statistic computed from the stabilised clip,

```math
D(\delta) = \mathcal D\bigl( \epsilon_M + \delta \bigr) ,
```

   the observed statistic under an injected perturbation is the response function $\mathcal D$ translated by the unknown error. This identity is exact in the geometry and holds for any black-box instrument. It is what allows a frozen encoder to be calibrated without labels (Section 5).

### 3.3 What breaks H0 without a depth fault

Movers; view-dependent appearance (specular highlights, reflections, transparency); occlusion (a
duplicated foreground next to depth edges); pose error (a smooth residual field over the frame);
exposure change. The first three are the nuisances of every multi-view witness. They are treated by
the ancillary statistics, by the boundary-interior comparison of Section 4.4, and by excluding a band
at rendered depth discontinuities, which is reported as abstention.

## 4. Reading One and Two: Explicit Parallax, per Pixel and per Masklet

### 4.1 Pixel parallax gap

A frozen flow or tracking model applied to the stabilised clip returns residual displacements
$\mathbf q_i(u)$. Along and across the epipolar line,

```math
e_i = \hat{\mathbf l}_i^\top \mathbf q_i = - b_i\, \epsilon_M + \epsilon_i^{\parallel}, \qquad o_i = \hat{\mathbf n}_i^\top \mathbf q_i = \epsilon_i^{\perp} ,
```

and weighted least squares gives the pixel gap with a geometric error bar,

```math
r_T = - \frac{\sum_i b_i\, e_i / \sigma_i^2}{\Lambda_T} = \epsilon_M - \epsilon_T, \qquad
\Lambda_T = \sum_i \frac{b_i^2}{\sigma_i^2}, \qquad \operatorname{Var}(\epsilon_T) = \Lambda_T^{-1} .
```

A track of $L$ frames supplies $2L$ scalars, of which one carries depth. The others are ancillary, free
of the depth and of the monocular error under a static rigid scene:

```math
A_\perp = \sum_i \frac{o_i^2}{\sigma_i^2} \sim \chi^2_L, \qquad A_\parallel = \sum_i \frac{(e_i + b_i r_T)^2}{\sigma_i^2} \sim \chi^2_{L-1} .
```

They give abstention (a pixel that left the static rigid model), the noise scale of the instrument per
stratum, and an anisotropy test along the flight direction. The forward and backward halves of the
window give two gaps $r_T^{+}$, $r_T^{-}$ with independent witness errors, and
$\operatorname{Var}(r_T^{+} - r_T^{-})$ checks the predicted error bar without any monocular quantity.
A witness error proportional to the parallax gain passes all of these: planar reflections, motion along
the strip at constant speed, parallax borrowed from a neighbouring surface.

### 4.2 The masklet-affine error model

The inverse depth of a plane is affine in the pixel: for a plane $\mathbf N^\top X = 1$,
$\xi(u) = \mathbf N^\top \mathbf k_u$. Monocular networks tend to get the shape of an object roughly
right and its placement wrong. The working hypothesis is therefore stated for the *error*, not for the
surface:

```math
\epsilon_{M,j}(u) = \mathbf k_u^\top \boldsymbol\epsilon_{mj} + \eta_j(u), \qquad u \in \mathcal M_{mj} ,
```

three coefficients per masklet and frame (an offset and two tilts in inverse depth) and a remainder
$\eta$. The surface itself may be curved. Under this model a few informative locations anywhere in the
masklet determine the error of all its pixels, textured or not.

### 4.3 The masklet gap

**Interior.** Pooling the pixel gaps with the affine basis,

```math
\hat{\boldsymbol\epsilon}_{mj} = \Lambda_{mj}^{-1} \sum_{u \in \mathcal M_{mj}} \Lambda_T(u)\, \mathbf k_u\, r_T(u), \qquad
\Lambda_{mj}^{\mathrm{int}} = \omega \sum_{u \in \mathcal M_{mj}} \Lambda_T(u)\, \mathbf k_u \mathbf k_u^\top ,
```

with $\omega$ a deflation for the spatial correlation of the flow errors.

**Boundary.** In the stabilised clip the mask of a faulty masklet moves with its content. Segmenting
each stabilised frame (or propagating the mask through the stabilised clip) gives, for a boundary point
$u$ with outward normal $\hat{\boldsymbol\nu}(u)$, a normal displacement

```math
c_i(u) = - \bigl( \hat{\boldsymbol\nu}(u)^\top \mathbf b_i \bigr)\, \mathbf k_u^\top \boldsymbol\epsilon_{mj} + \text{noise}, \qquad
\Lambda_{mj}^{\partial} = \sum_{i} \sum_{u \in \partial^{\mathrm{own}} \mathcal M_{mj}} \frac{(\hat{\boldsymbol\nu}^\top \mathbf b_i)^2}{\sigma_\partial^2}\, \mathbf k_u \mathbf k_u^\top .
```

Only the component of the boundary normal along the epipolar direction counts: in a strip, the leading
and trailing edges of a region carry the information and edges parallel to the flight line carry none.
A closed outline always has both. The total information is
$\Lambda_{mj} = \Lambda_{mj}^{\mathrm{int}} + \Lambda_{mj}^{\partial}$, and the gap is identified when it
has rank three: three informative, non-collinear locations. A horizontal water surface needs one.

*Illustration, assumed inputs.* A 30 m region at 5 cm ground sampling has a perimeter of about 2400
pixels. If half of it has a usable normal component, boundary errors are correlated over 20 pixels and
are 2 pixels in size, one frame gives about 60 effective points, a mean boundary displacement known to
about 0.26 pixel, and with a height-to-baseline ratio of 1.6 an offset known to about 2 cm in depth.
The noise is not the limit. The limit is systematic: whether the segmenter draws the boundary of a
water surface or a glass pane at the same physical place in every view.

**Ownership.** A boundary segment between two regions belongs to the nearer one. For the farther region
it is an occlusion edge that moves with the occluder's parallax, and using it yields the occluder's
depth. Segments are therefore classified as *attached* (a marking or a crease: both sides at the same
depth, usable by both), *owned* (this masklet is the nearer side) or *foreign* (excluded). The
classification needs only depth order across the boundary, which the monocular depth supplies
reliably where its metric value does not, and it is checked by the boundary ancillary below. Curved
occluders, whose rim slides over the surface with the viewpoint, are a remaining bias.

### 4.4 Masklet ancillaries and the layer test

With $n$ informative locations and three unknowns, a masklet has redundancy of its own.

- **Within-masklet scatter.** $A_m = \sum_u \Lambda_T(u)\,(r_T(u) - \mathbf k_u^\top \hat{\boldsymbol\epsilon}_{mj})^2$ measures the remainder $\eta$: the part of the monocular error that does not follow the masklet. Pooled over a stratum it yields the share of error energy that is masklet-affine, the label-free test of the unit itself.
- **Boundary consistency.** $A_\partial$, the weighted residual of the boundary displacements about the fitted three parameters. A masklet whose boundary segments do not agree with one affine error has mixed ownership or is not one surface; it abstains or is split.
- **Boundary against interior.** The two estimates $\hat{\boldsymbol\epsilon}^{\partial}$ and $\hat{\boldsymbol\epsilon}^{\mathrm{int}}$ use different image evidence. Their difference, a three-degree-of-freedom $\chi^2$, is zero for an opaque diffuse surface. A static outline with a moving interior, the interior lying *behind*, is the signature of a reflecting or transparent surface: the outline is attached to the pane or the shoreline, the interior content is a virtual image or a view through. This is layered motion, and it is identified per masklet without labels.

The third statistic settles the main open problem of pixel-level parallax for bounded reflectors. The
verdict about the *surface* is taken from the boundary; the interior is recorded as a material
attribute ("second layer present") and removed as a judge. What remains unresolved is a reflector with
no attached boundary inside the view.

## 5. Reading Three: the Latent Witness

### 5.1 What the encoder is asked

A frozen video encoder $E$ maps a clip to tokens on a grid of patches and tubelets. The latent
discrepancy of a token $\tau$ is the distance between the tokens of the stabilised clip and those of
the reference frame repeated over the same length,

```math
D_\tau = \bigl\lVert E\bigl( \tilde I_{\mathcal W_j \to j} \bigr)_\tau - E\bigl( I_j, \dots, I_j \bigr)_\tau \bigr\rVert ,
```

pooled over the tokens of a masklet for the masklet statistic. The repeated-still reference cancels
what the encoder does to any clip of that content (positional structure, temporal attention) and
leaves what the residual motion adds. No tracker and no triangulation are involved.

### 5.2 What the latent distance is not

- *Not a measurement.* It has no unit, no sign and no error bar.
- *Not texture-free.* To first order $D \propto \lVert J_E\, \nabla I^\top \mathbf b_i\, \epsilon_M \rVert$: without image structure inside the receptive field the stabilised clip is static whatever the depth, for any encoder. What a learned encoder adds over patch correlation is tolerance to exposure and resampling, use of weak and large-scale structure, and sensitivity to a moving *outline*, which is the boundary term of Section 4.3 seen natively.
- *Not free of the reflection problem.* A virtual image wobbles in the stabilised clip of a correct depth exactly as a fault does.
- *Not fine.* Tokens cover 16 by 16 pixels and two frames. A residual motion well below a patch barely moves a token, and a masklet narrower than two tokens has no interior tokens.

### 5.3 From distance to estimator: the latent sweep

By the exchange identity of Section 3.2, $D(\boldsymbol\delta) = \mathcal D(\boldsymbol\epsilon_{mj} + \boldsymbol\delta)$
for a perturbation $\boldsymbol\delta$ of the three masklet coefficients (or of the offset alone). If
the response $\mathcal D$ is minimal at zero residual motion, which is H0 itself, then

```math
\hat{\boldsymbol\epsilon}_{mj}^{\,\mathrm{lat}} = - \arg\min_{\boldsymbol\delta} D(\boldsymbol\delta)
```

is a consistent estimate of the masklet error: a plane sweep in latent space with three parameters per
masklet. It returns sign and size in metric units, because the perturbation is metric. No calibration
of the encoder is needed for the location of the minimum; only the noise of the minimiser has to be
estimated.

### 5.4 Information by injection

The curvature of $D$ at its minimum, obtained from the same sweep by finite differences, is the
observed sensitivity of the instrument at this masklet:

```math
\mathcal I_{mj} = \frac{D(+\boldsymbol\delta_0) + D(-\boldsymbol\delta_0) - 2 D(\mathbf 0)}{\lVert \boldsymbol\delta_0 \rVert^2} \quad \text{(at the minimiser; one direction shown)} .
```

A masklet on which an injected fault of the size of interest does not move the statistic has
$\mathcal I \approx 0$, and the latent witness abstains there *by its own measurement*. This replaces
the geometric $\Lambda_T$ for an instrument that has no noise model. It also disposes of the hope that
the latent version sees what explicit parallax cannot: if it does, the curvature will show it; if it
does not, the curvature is zero and the unit is reported as unwitnessed.

The noise of the minimiser is calibrated in two label-free ways: the scatter between the forward and
backward half-windows, $\hat{\boldsymbol\epsilon}^{\mathrm{lat},+} - \hat{\boldsymbol\epsilon}^{\mathrm{lat},-}$,
and the regression of the latent estimate on the explicit gap where both exist. The explicit gap, with
its error bar and ancillaries, is the calibrated special case; the test of honesty for the latent
witness is that its estimate agrees with the explicit one within the stated errors where both are
informative, and that its claimed information vanishes where the injected response does.

### 5.5 Cost

One stabilised clip per reference frame and window; one encoder pass for the statistic and, for a
sweep, two passes per swept direction (offset only: three passes in all; offset and tilts: seven). All
masklets of a tile are perturbed in the same pass with independent signs. Survey frames must be tiled
at native resolution, because downsampling a frame to the encoder's input size divides the residual
motion by the same factor. A cheaper variant warps token grids of independently encoded frames instead
of pixels and saves the re-encoding, at the price of position-dependent tokens that are not
equivariant to sub-patch shifts; it is an ablation.

## 6. Hierarchical Model and What Is Identified

### 6.1 Variance components

After transport to a common frame by the known poses,

```math
\boldsymbol\epsilon_{mj} = \boldsymbol\mu_m + \boldsymbol\alpha_{mj}, \qquad
\boldsymbol\mu_m \mid s \sim (1 - \pi_s)\, \mathcal N(\mathbf 0, \Sigma_{0s}) + \pi_s\, p_{\mathrm{out}}, \qquad
\boldsymbol\alpha_{mj} \sim (\mathbf 0, \Sigma_{\alpha s}) ,
```

with $s$ the stratum of the masklet. $\boldsymbol\mu_m$ is the error that every frame makes on this
surface; $\boldsymbol\alpha_{mj}$ is what frame $j$ adds; $\eta_j(u)$ is what does not follow the
masklet. The masklet fault state $F_m$ is the outlier component of $\boldsymbol\mu_m$.

### 6.2 Repetition is an intraclass correlation

The cavity depth of frame $j$ is fused from the other frames' depth maps, so on masklet $m$ its error
is $\boldsymbol\mu_m$ plus an average of other frames' $\boldsymbol\alpha$. The consensus gap of frame
$j$ is therefore $\boldsymbol\alpha_{mj} - \bar{\boldsymbol\alpha}_{m,-\mathcal B_j}$ plus noise, with
no $\boldsymbol\mu_m$ in it:

**Depth consensus observes the per-frame term. Only parallax observes the surface term.**

The repetition coefficient, defined as the share of a frame's error that the cavity repeats, is

```math
\varrho = \frac{\operatorname{tr} \Sigma_\mu + \operatorname{tr} \operatorname{Cov}(\boldsymbol\alpha_{mj}, \bar{\boldsymbol\alpha}_{m,-\mathcal B_j})}{\operatorname{tr} \Sigma_\mu + \operatorname{tr} \Sigma_\alpha} ,
```

which for per-frame terms independent beyond the block is the intraclass correlation
$\operatorname{tr}\Sigma_\mu / (\operatorname{tr}\Sigma_\mu + \operatorname{tr}\Sigma_\alpha)$. Its
dependence on the block size $B$ is the temporal correlation of $\boldsymbol\alpha$; its plateau is the
surface term. As before it is estimated without labels as an errors-in-variables slope,
$1 - \hat\varrho = \widehat{\operatorname{Cov}}(r_C, r_T) / (\widehat{\operatorname{Var}}(r_T) - \overline{\Lambda_T^{-1}})$,
now with masklet gaps in place of pixel gaps, or directly from the two variance components below.

### 6.3 Identification chain

| Quantity | Identified from | Rests on |
| --- | --- | --- |
| instrument noise, per stratum | $A_\perp$, $A_\parallel$; half-window scatter | static rigid scene, poses |
| witness information $\Lambda_{mj}$, or $\mathcal I_{mj}$ for the latent witness | geometry and noise scale; injected response | exchange identity |
| per-frame component $\Sigma_{\alpha s}$ | scatter of the monocular masklet coefficients across frames, $\sum_j (\hat{\boldsymbol\epsilon}_{mj} - \bar{\boldsymbol\epsilon}_m)(\cdot)^\top$, in which the witness of the masklet cancels when it is formed once per masklet | masklet association |
| law of the surface term $\boldsymbol\mu_m$, prevalence $\pi_s$ | deconvolution of $\bar{\boldsymbol\epsilon}_m$ over the masklets of a stratum with the known kernel $\Lambda_m^{-1} + \Sigma_\alpha / J_m$; forward and backward half-windows as two independent channels | witness error independent of monocular error; no witness error proportional to parallax |
| share of error that follows masklets | $A_m$ | none beyond the above |
| second layer (reflection, transparency) | boundary against interior | attached boundary in view |
| repetition $\varrho(B)$ | variance components, or the slope against the consensus gap | as above |
| anything on a masklet with rank-deficient $\Lambda_m$ and no feature overlap | not identified | |

The exchangeable units of the deconvolution are masklets, not pixels. The sample size of every
per-stratum estimate is the number of masklets in the stratum, which is the honest count: faults are
regional, and pixel counts overstated the evidence by the area of a region.

### 6.4 The unit hypothesis and its test

Everything above uses the hypothesis that monocular error is low-order within masklets. If faults cut
across masklets, or cover a small part of one, a masklet verdict dilutes them: a shift $\delta$ on a
fraction $a$ of the masklet has a mean shift $a\delta$ and a non-centrality reduced by the factor $a$
relative to a test on the right region. Three safeguards:

- the remainder statistic $A_m$ flags masklets whose interior does not follow the affine model, and the pixel-level posterior is kept as the second-order term, $\operatorname{logit} \zeta_j(u) = \operatorname{logit} \Pr(f \mid \zeta_m) + \log \bigl( g^1 / g^0 \bigr)(r_T(u) - \mathbf k_u^\top \hat{\boldsymbol\epsilon}_{mj})$;
- the segmenter's nested masks give a tree, and the verdict is placed at the coarsest node whose children's gaps are homogeneous;
- the share of true error energy that is masklet-affine is measured once on a scene with exact depth (Section 11). If it is low, the masklet is the wrong unit and the pixel formulation stands.

A fixed segmentation has one more consequence: the number of tests is the number of masklets. A scan
over all connected regions pays a multiplicity price that a given partition does not.

## 7. Verdict, Readout and Prior

### 7.1 Masklet verdict

```math
\bar{\boldsymbol\epsilon}_m = \Bigl( \sum_j \Lambda_{mj} \Bigr)^{-1} \sum_j \Lambda_{mj}\, \hat{\boldsymbol\epsilon}_{mj}, \qquad
\operatorname{logit} \zeta_m = \operatorname{logit} \pi(\boldsymbol\phi_m)
+ \log \frac{\bigl( p_{\mathrm{out}} * \mathcal N(\mathbf 0, V_m) \bigr)(\bar{\boldsymbol\epsilon}_m)}{\mathcal N\bigl( \bar{\boldsymbol\epsilon}_m;\ \mathbf 0,\ \Sigma_{0s} + V_m \bigr)} ,
```

with $V_m = \Lambda_m^{-1} + \Sigma_{\alpha s} / J_m$. The verdict is one number per surface entity and
is the same in every frame by construction. As the information of the witness goes to zero the
likelihood ratio goes to one and $\zeta_m \to \pi(\boldsymbol\phi_m)$. The three coefficients are also
reported separately: an offset fault, a tilt fault and, through $A_m$, a shape fault are different
failures of the prior and call for different handling in the surface fit.

### 7.2 Readout head: supervised by injection

A light head $w_\theta$ on the tokens of the stabilised clip (or on their difference to the repeated
still) regresses the masklet error. Its training signal is exact and free: perturb the monocular depth
of random masklets by known $\boldsymbol\delta$, stabilise, and require

```math
w_\theta\bigl( \mathrm{stab}(\hat\xi_M + \boldsymbol\delta) \bigr) - w_\theta\bigl( \mathrm{stab}(\hat\xi_M) \bigr) = \boldsymbol\delta ,
```

an equivariance constraint that does not need the true error. By the exchange identity an injected
perturbation produces the same residual motion as a real error of that size, so there is no domain gap
in the signal; only the spatial pattern of real faults can differ, and perturbations are drawn on
masklets for that reason. The constraint fixes $w_\theta$ up to its value at zero perturbation, which
is anchored by the explicit gap where that is informative. The readout amortises the sweep to one
pass. It is a learned *witness*, and it inherits every limit of the witness, reflections included.

### 7.3 Prior head: what it may see

The prior $\pi(\boldsymbol\phi_m)$ says which surfaces fail, from what they look like, and it is what
reaches masklets without a witness. It is fitted by the marginal likelihood of the masklet gaps (the
regression of the prior logit onto $\zeta_m$), which is learning from labels whose noise is known per
unit through $V_m$. The label information of a masklet,
$\iota_m = \mathbb E(\zeta_m - \pi_m)^2 / (\pi_m (1 - \pi_m))$, weights it automatically.

Its input must not contain the residual motion. Tokens of the stabilised clip do, by construction.
Tokens of the *raw* clip contain the parallax but not the depth under test, so the gap cannot be
computed from them; they do let the head learn motion signatures of surfaces (rippling, flicker,
layered motion) and are admissible. The risk that remains is not copying noise but learning the
witness's *bias*: if reflective masklets are labelled faulty because their interior is a virtual
image, a head on motion-aware tokens will learn that layered motion means fault. The remedy is in the
labels, not in the head: verdicts on masklets with a detected second layer come from the boundary
only. The check is that held-out likelihood does not improve when the head is additionally given the
masklet's own witness statistics. Pooling tokens over the masklet's mask and frames gives one feature
per surface entity; cross-fitting is by ground tile.

**Strata from concept prompts.** A promptable segmenter that accepts noun phrases returns the masklets
of "water", "glass roof" or "solar panel" without labels. Used as covariates of prevalence and of the
instrument's noise, wrong strata cost efficiency only. Used to switch the model, for example the
one-sided fault model on reflective material, they must be confirmed by the layer test, which is
geometric and does not depend on the segmenter's vocabulary. Whether noun-phrase prompts work on nadir
aerial imagery is not known.

## 8. Harm Against Detectability, in Masklets

Let $\vartheta_m$ be the ratio of witness information to the information of the prior on the masklet's
coefficients. If both are fused, a fault of size $\Delta$ in the prior moves the surface by
$\Delta / (1 + \vartheta_m)$, while the non-centrality of the comparison is
$\Delta^2 \Lambda_M\, \vartheta_m / (1 + \vartheta_m)$. Direct testing pays where the two are of the
same order (for small faults the expected gain is proportional to
$\vartheta / (1 + \vartheta)^3$, largest at $\vartheta = \tfrac12$; for a fault of ten inlier standard
deviations the useful band is roughly $0.2 \le \vartheta \le 3$). The three regimes remain, and the
masklet formulation moves surfaces between them.

| Regime | Pixel formulation | Masklet formulation |
| --- | --- | --- |
| replace, $\vartheta \gg 1$ | tracked, diffuse, good baseline | textured masklets; their verdicts are the training labels |
| judge, $\vartheta \approx 1$ | short tracks, far range, strip ends | in addition: untextured masklets with a few interior anchors or an attached boundary, since three coefficients are estimated from all the evidence of the region |
| blind | texture-free, reflective, occluded | masklets with rank-deficient information: untextured with foreign or missing boundaries, unbounded surfaces cut by the frame, and masklets whose error is not low-order |

The reason to keep the monocular prior is the same as before and is now narrower: it is the only dense
depth inside masklets whose witness has rank three but little information per pixel, and on blind
masklets. On textured masklets the witness replaces it. The gain of the masklet unit is that water
with a shoreline, glazing with frames and flat roofs with parapets move from blind to judged, which
are the surfaces where monocular depth fails most. Evaluation is stratified by $\vartheta_m$ and
weighted by harm; detection on textured masklets measures the regime in which the answer does not
matter.

## 9. Transfer, in Masklets

Training on witnessed masklets and applying to unwitnessed ones is learning from selectively observed
labels. With $o_m = \mathbb 1[\iota_m > \iota_0]$ and propensity $e(\boldsymbol\phi) = \Pr(o = 1 \mid \boldsymbol\phi)$,
the transfer is valid if (S1) among masklets that look alike to the encoder, the witnessed ones are
faulty as often as the others, and (S2) $e(\boldsymbol\phi) > 0$ where the prior is applied. A second
head predicts $o$ from the same features, needs no fault label, and divides unwitnessed masklets into
an interpolation zone and an extrapolation zone; in the latter the output is reported as unsupported,
not as a probability.

Two things change with the unit. Overlap improves: a material that is never tracked pixel by pixel is
often witnessed as a region, so water and glass can enter the interpolation zone through their bounded
instances and be transferred to their unbounded ones, which is transfer within a material instead of
from textured to untextured surface. And S1 acquires a specific failure: bounded and unbounded
instances of a material differ in size and context (a pond against the sea, a skylight against a
curtain wall), and monocular error may depend on exactly that.

Depth consensus remains the examiner on unwitnessed masklets. It needs no texture and observes the
per-frame term, so the rate of consensus outliers must rise with the prior there. By Section 6.2 this
examines the prior only on the per-frame component. A surface-term fault on an unwitnessed masklet is
beyond every label-free check, and the claim there rests on transfer and on labelled evaluation.

## 10. What This Theory Cannot Deliver

1. **Witness errors proportional to parallax, without an attached boundary.** Reflectors that fill the view, constant-speed motion along the strip, and curved occluders whose rim slides.
2. **A verdict when the unit is wrong.** Faults that cut across masklets are diluted; masklet identity errors (a mask that covers different surfaces in different frames) corrupt the per-frame component. Lifting masks onto the surfel map gives a geometric check of identity, at the cost of using the surface under test.
3. **Boundary placement.** The whole boundary term assumes that the segmenter draws a boundary at the same physical place in every view. For shorelines, wet margins and specular edges this is doubtful and can only be measured.
4. **Latent evidence without image structure.** No encoder turns a uniform interior into evidence; the injected response will say so.
5. **Calibrated probabilities and stated error rates.** $\zeta_m$ is a probability under the model; a false-alarm rate needs a separate calibration with masklets or scenes as units.
6. **Verification of S1 and of the independence of witness and prior errors without labels.**
7. **Small entities.** A masklet below about two tokens across has no latent interior, and below a few tens of boundary pixels no boundary term; such entities fall back to the pixel gap.
8. **Robustness to pose error beyond its covariance,** which makes the whole stabilised frame move.

## 11. What Would Refute the Approach, and the Cheapest Tests

On a scene with exact depth, used for evaluation only. The first two need no video model.

1. **Errors do not follow masklets.** Segment single frames, fit the affine model to the *true* error of the monocular depth per mask, and report the share of error energy and of fault area explained. If small, the masklet is the wrong unit. Cost: hours.
2. **No surface term.** From the same fits over frames, the variance components $\Sigma_\mu$ and $\Sigma_\alpha$ and the curve $\varrho(B)$. If the surface term is negligible at a block of a few frames, consensus suffices and the premise falls.
3. **The boundary is not a measurement.** Estimate the plane of water, glazing and flat roofs from mask outlines alone, with exact poses, and compare with the true plane; count ownership errors. If the systematic error exceeds the monocular inlier error, the boundary term is dropped and untextured surfaces return to the blind regime.
4. **The latent witness adds nothing.** Stabilise with exact depth plus injected perturbations and record the dose-response of the token distance, of residual flow and of plain photometric difference, by texture level and at both frame spacings. If the encoder's response is not stronger than residual flow anywhere, the latent reading is dropped and stabilisation, masklets and explicit parallax remain. This test also shows whether stabilised aerial clips are usable inputs for a model trained on ground-level video.
5. **Masklets do not survive.** Propagate masks through raw and stabilised clips at survey spacing and at video rate; measure identity switches against the surfel association.
6. **Transfer fails,** and **the prior adds nothing once the witness is fused** (oracle-gate ceiling on blind masklets), as in the pixel formulation.

Tests 1 and 2 can end the masklet formulation, and test 2 the project, before any video model is run.

## 12. What Is New, and the Nearest Prior Work

Named plainly: stabilising a sequence by a plane or a depth map and examining the residual motion is
plane-plus-parallax; estimating one plane per segment is piecewise-planar stereo; separating a
reflection from the surface behind it by their different motions is layered motion analysis; comparing
warped views in a learned feature space is the feature-metric loss of self-supervised depth; reading
surprise from a video encoder's latent prediction error has been done for intuitive physics; training
a detector on synthetic defects is standard in anomaly detection. None of these is new.

What the formulation adds, in the part treated here, and to the extent recalled without a systematic
search:

1. **The exchange identity and information by injection** (Sections 3.2, 5.3, 5.4). For a depth prior judged in a stabilised clip, a known perturbation is exchangeable with the unknown error. Any frozen, uncalibrated video model thereby becomes an estimator with sign and size, a self-reported information, and an exactly supervised readout, with no labels. This is what makes "violation of expectation in a foundation model" a measurement instead of a score, and it applies to any future encoder unchanged.
2. **The variance-component view of repeated faults** (Section 6). The surface term is seen only by parallax, the per-frame term only by consensus, and the repetition coefficient is their intraclass correlation, measurable on real flights. The masklet is the unit that makes this decomposition, and the sample size of every estimate, explicit.
3. **Judging with three parameters per entity under an error-planarity assumption, with the boundary-interior layer test** (Section 4). It moves bounded untextured and reflective surfaces from blind to judged and identifies the second layer per entity.

Items 1 and 2 are the candidates for a contribution at the level aimed for. Item 3 is an application of
known geometry whose value depends entirely on test 3 of Section 11.

## 13. Models and Attributions: What Was Checked

**Checked against the authors' paper or repository pages.**

| Model | Verified | Not verified |
| --- | --- | --- |
| V-JEPA 2 (arXiv 2506.09985) | Self-supervised video encoder; tubelets of 2 by 16 by 16; pretraining at 256 pixels, one variant at 384; clips of 16 to 64 frames; ViT-L about 0.3, ViT-H 0.6, ViT-g about 1 billion parameters; objective is L1 regression of masked token representations against an averaged teacher; trained on web and instructional video plus ImageNet. Checkpoints are loadable through a public repository | Code and weight licence; memory at 64 frames and 384 pixels; behaviour on aerial imagery |
| V-JEPA 2-AC | An action-conditioned predictor of about 0.3 billion parameters on the frozen encoder, conditioned on 7-dimensional robot end-effector actions and states, trained on about 62 hours of table-top manipulation at 4 frames per second | There is no camera-pose-conditioned predictor in this family. The conception's "expected motion given poses" is not available from the model and has to be imposed by warping |
| V-JEPA 2.1 (arXiv 2603.14482) | Trained for dense features: a dense predictive loss on visible and masked tokens, deep self-supervision, separate image and video tokenizers; ViT-g about 1 and ViT-G about 2 billion parameters with distilled ViT-L and ViT-B; cool-down at 64 frames and 384 pixels; linear-probe depth, segmentation and video object segmentation reported; released in the same repository. This, not V-JEPA 2, is the member of the family intended for dense probes | Token quality at sub-patch motion; licence of weights; no result on aerial or wide-baseline input was seen |
| Latent surprise as violation of expectation (arXiv 2502.11831) | Surprise is the error of the pretraining predictor on future-frame representations given past frames, used without adaptation | The distance used at evaluation |
| SAM 3 (arXiv 2511.16719) | Promptable concept segmentation: noun phrases, image exemplars or both; returns masks with identities for all matching instances in images and video; a detector and a memory-based tracker on a shared backbone; 848 million parameters; released under its own "SAM License"; a later "SAM 3.1" checkpoint for faster multi-object tracking exists | Input resolution; memory and speed with hundreds of masklets per clip; limits on the number of tracked objects; quality of noun-phrase prompts on nadir aerial imagery; whether it yields an exhaustive partition without prompts |
| SAM 2 (arXiv 2408.00714) | Existence and title only | Licence, architecture and behaviour at large inter-frame motion are recalled, not checked |

Two consequences for the design follow from what was verified. The encoder side is served by V-JEPA
2.1, with the expectation supplied by stabilisation because no pose-conditioned predictor exists. The
segmenter side is served by the SAM family for masklets and concept strata, but whether its masklets
survive survey spacing, and whether its boundaries are view-consistent on water and glass, are open
and are tests 3 and 5 of Section 11.

**Recalled, not checked.** Plane-plus-parallax (Sawhney 1994; Irani and Anandan 1996). Piecewise-planar
stereo (Sinha, Steedly and Szeliski 2009; Gallup, Frahm and Pollefeys 2010). Layer extraction with
reflections and transparency (Szeliski, Avidan and Anandan 2000; Xue et al. 2015). Feature-metric loss
for self-supervised depth (Shu et al. 2020). Synthetic-defect self-supervision for anomaly detection
(Li et al. 2021). Identification from two measurements with independent errors (Kotlarski 1967);
variance components and intraclass correlation are textbook. Other video-native dense backbones that
could take the encoder's place (video masked autoencoders, multi-view geometry transformers) were not
examined for this note. Whether the exact-depth aerial scenes available for evaluation render
reflections on glass and water physically, on which test 3 and the layer test depend, was not checked.
