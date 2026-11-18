# Evidence Definitions: Monocular Depth Judged by Motion Parallax

This note defines the evidence that the fault indicator is built on, for the setting in which the input
is a posed image sequence flown in strips and there is one monocular depth map per image. There are two
sources of evidence. The first is depth consensus between views (the leave-block-out residual of the
main document). The second, the *motion witness*, is new here: a frozen tracker or matcher gives
correspondences, the known poses turn each correspondence into a triangulated depth with a stated
uncertainty, and the gap between monocular and triangulated depth is a fault label that costs nothing
and does not depend on what the other depth maps say. A light head on frozen features is trained on
those labels and carries the result to pixels where the witness abstains.

For each source the note gives the score, its nuisance alignment, how its null is obtained, when it
abstains and how that is detected, what it physically responds to, and how the two sources depend on
each other. It then states what is identified and what is not, what the code already contains, what
would end the idea, and the cheapest test.

Notation follows the main document: view $j$, pixel $u$, depth map $\hat d_j$, per-view scale and
offset $(a_j, b_j)$, shared distance bias $\beta$, cavity depth $\hat D_{-j}$, block $\mathcal B_j$,
testable set $\Omega_j$, latent fault $f_j(u)$, nominal noise $\sigma_j(u)$.

**Status of the statements.** Nothing in this note has been measured. All behaviour tables are
predictions. Properties of third-party models (architectures, training data, outputs, licences) are
recalled from memory and marked "(recalled)"; none could be checked, because no model code or weights
are installed in the working environment and the model hubs were not reachable from it. Statements
about the project's own code were checked against the source.

## 1. Setting

### 1.1 Target

$f_j(u) = 1$ when the aligned error of the depth map at $(j, u)$ exceeds $3\sigma$ inside a connected
region of at least 2% of the view. A whole-view error inside the nuisance group of the depth model
(scale for a metric model, affine in inverse depth for a relative one) is absorbed by the alignment
and is not a fault. Edge mislocalisation and thin structures are not faults under this definition, but
every source of evidence responds to them; they are excluded by a band around depth discontinuities.

### 1.2 What "video" means for a survey flight

The usefulness of a tracker is governed by one number, the image displacement between consecutive
frames, $\Delta_1 = f B_1 / d$ pixels for a frame spacing $B_1$, focal length $f$ in pixels and depth
$d$. Two regimes must be kept apart.

- *Dense video*: $\Delta_1$ of a few to a few tens of pixels. Point trackers and optical-flow models
  are trained for this regime (recalled).
- *Survey stills*: frames are exposed for a fixed forward overlap. At 80% forward overlap consecutive
  frames are displaced by 20% of the image length, which is hundreds of pixels, and a ground point is
  seen in about five consecutive frames of a strip and again in the neighbouring strip at 60% side
  overlap. One of the real data sets configured in the repository is of this kind (nadir, about 80 m,
  80/60 overlap). This is the regime of wide-baseline dense matching, not of video tracking.

The definitions below are written for an abstract *correspondence provider* that returns, for a query
pixel, positions in other frames with a visibility flag and optionally a confidence. Which frozen model
plays this role depends on $\Delta_1$ and has to be decided per data set (Section 7). The frame spacing
of the synthetic aerial block was not checked: its images are not on disk in the working environment.

The same number sets the precision. With a frame spacing of 20 m, $f = 4000$ px and $d = 80$ m
(illustrative values consistent with the configured survey: 2 cm ground sampling distance at 80 m; the
frame spacing is assumed), one pixel of matching error on one pair is $d^2/(f B_1) = 0.08$ m of depth,
against an expected monocular error of the order of a metre. Where a correspondence exists, the witness
is one to two orders of magnitude sharper than the depth it judges. The question for this evidence is
never precision; it is coverage and the honesty of its abstention.

## 2. The Motion Witness

### 2.1 A track as a one-dimensional measurement plus a check

Let $\mathbf x = K_j^{-1} \tilde u$ be the ray of pixel $u$ (unit $z$), $\xi = 1/d$ its inverse depth,
and $(R_i, \mathbf t_i)$ the known motion from camera $j$ to camera $i$. The point at inverse depth
$\xi$ projects to

```math
\mathbf p_i(\xi) = \pi\bigl( K_i ( R_i \mathbf x + \xi\, \mathbf t_i ) \bigr), \qquad
\mathbf m_i(\xi) = \frac{\partial \mathbf p_i}{\partial \xi}, \qquad \mathbf e_i = \frac{\mathbf m_i}{\lVert \mathbf m_i \rVert}, \quad \mathbf n_i \perp \mathbf e_i .
```

As $\xi$ varies, $\mathbf p_i$ moves along the epipolar line of $u$ in view $i$, whose direction
$\mathbf e_i$ is fixed. The *parallax vector* $\mathbf m_i$ is the image motion per unit inverse depth;
for a sideways translation $\lVert \mathbf m_i \rVert = f B_i$ with $B_i$ the baseline. A track gives
observed positions $\hat{\mathbf p}_i$ for $i$ in a set $\mathcal S(u)$ of $n$ other views. Each
observation splits into two scalars,

```math
\nu_i = \mathbf n_i^\top \bigl( \hat{\mathbf p}_i - \mathbf p_i(\xi) \bigr), \qquad
a_i(\xi) = \mathbf e_i^\top \bigl( \hat{\mathbf p}_i - \mathbf p_i(\xi) \bigr).
```

The across-line component $\nu_i$ does not depend on $\xi$ at all: it is fixed by the poses. The
along-line component is the measurement of depth. So a track of $n$ observations is $2n$ numbers with
one unknown: $n$ equations for the depth, and $2n - 1$ degrees of freedom of redundancy that say
whether the track is what it claims to be.

This is the structural difference from a plane sweep, the classical image-based alternative. A
plane sweep searches along the epipolar line only. It cannot fail the epipolar test, because it never
looks off the line, and its uncertainty is the curvature of a correlation profile up to an unknown
constant. A correspondence found by an unconstrained two-dimensional search spends its second
dimension on a check.

### 2.2 Estimator and uncertainty

With along-line standard deviations $\sigma_{a,i}$ and a robust loss $\rho$,

```math
\hat\xi(u) = \arg\min_{\xi} \sum_{i \in \mathcal S(u)} \rho\Bigl( \frac{a_i(\xi)}{\sigma_{a,i}} \Bigr), \qquad
\operatorname{Var}\bigl(\hat\xi\bigr) = \Bigl( \sum_{i \in \mathcal S(u)} \frac{\lVert \mathbf m_i \rVert^2}{\sigma_{a,i}^2} \Bigr)^{-1},
```

a scalar problem solved by a few Gauss-Newton steps from the two-view solution of the longest
baseline. In depth, $\sigma_{\mathrm{tri}}(u) = d^2 \sqrt{\operatorname{Var}(\hat\xi)}$. For a track
over $n_f$ equally spaced frames with the query frame in the middle and a common $\sigma_a$,

```math
\sum_i \lVert \mathbf m_i \rVert^2 = f^2 B_1^2\, \frac{n_f (n_f^2 - 1)}{12}, \qquad
\sigma_{\mathrm{tri}} = \frac{d^2\, \sigma_a}{f B_1} \sqrt{\frac{12}{n_f (n_f^2 - 1)}} ,
```

so the uncertainty falls as $n_f^{-3/2}$: the length of the track, not a fixed set of neighbours,
decides how much baseline a pixel has, and it differs from pixel to pixel through occlusion and through
the position in the strip. The error is close to Gaussian in $\xi$, not in $d$; scores are therefore
formed in inverse depth.

### 2.3 The null is estimated from the track's own redundancy

The tracker's localisation noise is not known in advance and differs by surface type, displacement
and model. It does not have to be assumed.

- *Across-line noise.* Under the hypothesis of a static point, correct poses and a correct
  correspondence, $\nu_i$ has zero mean and its spread is the tracker's localisation noise across the
  line. The robust scale of $\nu_i$ over all tracks of a stratum is an estimate of $\sigma_\nu$ that
  uses no depth and no label.
- *Along-line noise.* For tracks with $n \ge 2$ observations the along-line misfit after estimating
  $\xi$ has $n - 1$ degrees of freedom. Its robust scale over a stratum estimates $\sigma_a$.
- *Anisotropy.* The two differ because of the aperture effect: a correspondence on an edge parallel to
  the epipolar line is sharp across the line and loose along it. The strata are therefore indexed by
  the angle between the local gradient orientation and $\mathbf e_i$, by gradient energy along
  $\mathbf e_i$, by displacement magnitude and by the tracker's confidence where it gives one.
  Assuming $\sigma_a = \sigma_\nu$ would make the witness overconfident exactly where it is weakest.

With the scales in hand, two test statistics have a known reference distribution,

```math
E(u) = \sum_{i \in \mathcal S(u)} \frac{\nu_i^2}{\sigma_{\nu,i}^2} \ \sim\ \chi^2_{n}, \qquad
C(u) = \sum_{i \in \mathcal S(u)} \frac{a_i(\hat\xi)^2}{\sigma_{a,i}^2} \ \sim\ \chi^2_{n-1} .
```

$E$ tests the track against the poses; $C$ tests whether one depth explains the along-line positions
in all views. $C$ needs at least two other views and is what catches a jump by one period on repeated
texture, since the jump is the same number of pixels at every baseline while a depth change is not.
Pose error adds a component to $\nu$ that is smooth over the image for a given pair; it is estimated
per pair by a low-order fit and removed, or carried as $\mathbf j_q^\top \Sigma_q \mathbf j_q$ in the
variance. The minority assumption applies as everywhere: within a stratum, most tracks are sound.

### 2.4 Score and its alignment

The monocular depth is brought to scene units by its own robust alignment against the triangulated
depths of well-conditioned tracks, in the nuisance group of the depth model: one scale for a metric
model, an affine map in inverse depth for a relative one. No other depth map and no surface enters. The
score is

```math
x^{\mathrm M}_j(u) = \frac{\xi^{\mathrm{al}}_j(u) - \hat\xi(u)}{\sqrt{\operatorname{Var}(\hat\xi(u)) + \sigma_j^2(u)\, \xi^{\mathrm{al}}_j(u)^4}} ,
```

with $\xi^{\mathrm{al}}_j = 1/d^{\mathrm{al}}_j$ and the second term the nominal monocular noise mapped
to inverse depth. Under the null it is standard normal up to a scale that is learned from geometry
strata, as for the consensus residual. For a regional decision the signed score is pooled over a
window with the effective sample size.

### 2.5 Abstention, graded by the minimum detectable fault

A binary "well conditioned / abstain" mask throws information away and hides the selection that the
head must later cross. The witness is characterised per pixel by the smallest fault it can detect. For
a two-sided test of size $\alpha$ and power $1 - \beta_{\mathrm p}$,

```math
\delta_{\min}(u) = \bigl( z_{1 - \alpha/2} + z_{1 - \beta_{\mathrm p}} \bigr)\ \sigma_x(u), \qquad
\sigma_x^2(u) = \sigma_{\mathrm{tri}}^2(u) + \sigma_j^2(u),
```

the minimum detectable bias of the geodetic testing literature applied to one pixel. The label
delivered by the witness is then not "fault / no fault" but a pair

```math
y_j(u) = \mathbb 1\bigl[\, \lvert x^{\mathrm M}_j(u) \rvert > z_{1-\alpha/2} \,\bigr], \qquad \delta_{\min}(u),
```

read as "a fault larger than $\delta_{\min}(u)$ is present / is excluded with power
$1 - \beta_{\mathrm p}$". Its false-positive rate is $\alpha$ by construction once the null is
calibrated, and its miss rate for a fault of magnitude $\delta$ is
$\Phi\bigl(z_{1-\alpha/2} - \delta/\sigma_x(u)\bigr)$. The labels come with a known, per-pixel error
model, which the head's loss can use directly (Section 5.1).

Hard abstention ($\delta_{\min} = \infty$) is declared when any of the following holds: no
correspondence, or the provider reports the point occluded; $E(u)$ or $C(u)$ above its $\chi^2$
quantile; forward-backward inconsistency of the correspondence; the track is prior-driven
(Section 2.6); the alignment of the view has too few inliers; the pixel lies in the discontinuity band.

### 2.6 What the epipolar check does not catch

The check $E$ is necessary for a sound label and far from sufficient. Three cases pass it with a wrong
depth, and each is tied to a surface class that matters.

**Planar reflections and transparent surfaces.** The virtual image of a static scene in a planar
mirror is itself a rigid static scene, the mirror image. A track on a reflection in a flat window, a
glass roof or calm water therefore stays on its epipolar line in every view and triangulates
consistently, at the virtual depth behind the reflector. The same holds for the interior seen through
clear glass, and for a sun glint on a flat surface, which triangulates at infinity. Only *non-rigid*
virtual images leave the line: rippled water, curved reflectors. From parallax alone a planar mirror
cannot be distinguished from an opening onto a mirrored scene; this is not a weakness of a particular
tracker but a non-identifiability. The consequence for labels: on a flat reflector whose monocular
depth is correct (the reflector's plane), the witness reports a confident fault. The false label is
systematic and attached to exactly the appearance class to which the head is supposed to carry the
result. A one-sided plausibility test (triangulated depth far behind a surface established by
surrounding tracks) reduces the damage but cannot be exact, because a real pit or courtyard has the
same geometry. Whether the exact depth of a synthetic scene records the reflector or what lies behind a
translucent material is also not known and must be checked before such scenes are used for evaluation.

**Motion along the epipolar line.** A vehicle driving parallel to the flight line stays on the line and
triangulates at a wrong depth. With three or more views at unequal time-to-baseline ratios $C$ catches
it; on a strip flown at constant speed and constant frame interval the object's displacement is
proportional to the baseline, exactly like parallax, and $C$ does not.

**Prior-driven tracks.** A learned tracker "holds" on weak texture by propagating the motion of the
surrounding structure. On a texture-free surface the reported correspondence is then an interpolation,
not a measurement. An interpolated motion field is epipolar-consistent whenever its support is, so $E$
and $C$ pass. The triangulated depth is the depth of the interpolant (in effect, the plane through the
textured surroundings), and $\operatorname{Var}(\hat\xi)$ computed from the stratum's localisation noise
understates its error. This is the same kind of inference from context that the monocular model makes,
which is the opposite of what a witness should be. It is detected by measuring how much of the
along-line information the local appearance actually supports. Let $\varphi$ be the feature map the
correspondence was computed on (or the image) and

```math
\mathcal E_i(u) = \sum_{u' \in \mathcal P(u)} \bigl( \mathbf e_i^\top \nabla \varphi(u') \bigr)^2, \qquad
\epsilon(u) = \frac{\min_{i \in \mathcal S(u)} \mathcal E_i(u)}{\mathcal E_{\mathrm{ref}}} ,
```

the feature gradient energy along the epipolar direction in a patch, relative to a reference level at
which the stratum's noise was calibrated. A complementary diagnostic is the similarity profile along
the epipolar line around the reported position: if it is flat, the position was not chosen by the
data. Tracks with small $\epsilon$ form their own stratum, are excluded from label generation by
default, and their triangulation error against exact depth is one of the first things to measure.
Qualified in this way, the statement that learned tracking extends coverage to weak texture may hold;
unqualified, it converts a prior into a label.

### 2.7 What the witness responds to

A displacement of the surface along the ray that moves its image in other views, on a surface that is
static, has appearance that a correspondence can lock onto, and is seen in at least one other view with
baseline. It does not use any other depth map, so an error that every depth map repeats is as visible
as one that a single view makes. That property is the reason for the witness.

## 3. Depth Consensus (unchanged)

The standardised leave-block-out residual

```math
z_j(u) = \frac{\hat d_j(u) - a_j \tilde D_{-j}(u) - b_j}{\sqrt{v_j(u)}}, \qquad \tilde D_{-j} = \hat D_{-j} + \beta_{-j} \hat D_{-j}^2 ,
```

together with a free-space score along rays for surfels that only the tested block supports, forms one
detector: both channels are depth maps of other views compared through the same poses. It abstains
outside $\Omega_j$. It responds to disagreement between depth maps whatever the cause and is blind to an
error that the views outside the block repeat.

## 4. Dependence between the Two Sources

### 4.1 Predicted behaviour

F fires on a true fault, M misses, A abstains, FA fires where the depth is correct. Predictions only.

| Case | Depth consensus | Motion witness | Head on frozen features |
| --- | --- | --- | --- |
| Fault on textured, diffuse surface, single view | F | F | not needed |
| Fault on textured, diffuse surface, repeated by all views | M | F: the case the witness exists for | not needed |
| Fault repeated within a block only | M unless the block is longer | F | not needed |
| Weak texture, track supported by the data at coarse scale | M if repeated | F for faults above a large $\delta_{\min}$ | interpolates in $\delta_{\min}$ |
| Texture-free, prior-driven track | M if repeated | A if detected by $\epsilon$; otherwise an unfounded label | extrapolates |
| Flat glass, calm water, depth map correct | silent | FA at the virtual depth; passes $E$ and $C$ | trained on false labels |
| Flat glass, calm water, depth map wrong | F if view dependent | fires, but for the wrong reason and at the wrong depth | label right by accident |
| Rippled water, curved reflector | F if inconsistent | A through $E$ | extrapolates |
| Moving object across the line | silent or F | A through $E$ | not needed |
| Moving object along the line, constant speed | silent or F | FA | none |
| Thin structure, depth edge | FA | A or FA | excluded by the band |
| Whole-view error in the nuisance group | M by construction | M by construction | M |
| Pose error of the view | FA, regional | FA unless removed by the per-pair fit; visible in $\nu$ | none |

### 4.2 Where independence is defensible and where it is not

The measurement noises come from different sources: depth networks applied to other images on one side,
correspondence positions on the other. Three shared causes remain.

1. **Poses.** Both sources use the pose of view $j$ and of its neighbours. The witness has the
   advantage that pose error shows in $\nu$ and can be estimated per pair; the consensus residual has
   no such channel.
2. **The image of view $j$.** The depth map is a function of it and the track starts from it. Where
   the image carries little information, the depth is more often wrong *and* the witness more often
   abstains or is prior-driven. This couples the fault to the abstention of the witness, not to its
   error; it is the selection problem of Section 5.
3. **Shared learned priors.** If the correspondence model and the depth model use encoders of the same
   family (recalled as likely for several candidates; not verified), their regions of weak evidence
   coincide. With the $\epsilon$ rule this again affects coverage, not the validity of the labels that
   remain.

With two sources, agreement cannot identify their accuracies; the witness is therefore not treated as
one noisy voter among several but as a calibrated measurement with its own null (Section 2.3). That is
the role the self-calibration plays in the overall argument: it replaces the third independent detector
that identifiability from agreement would require.

## 5. From Labels to an Indicator: Selection and Transfer

### 5.1 Labels with a known error model

On the labelled set the head $h$ on frozen features $\phi_j(u)$ is fitted to $y_j(u)$ with the
per-pixel error rates of Section 2.5: false-positive rate $\alpha$, and a miss rate that depends on
$\delta_{\min}(u)$. A fault indicator at a stated magnitude $\delta$ uses only labels with
$\delta_{\min}(u) \le \delta$; smaller magnitudes have fewer labels. The head never sees the score or
the depth residual of the pixel it labels, only features, so it cannot copy the label function.

### 5.2 The selection is on the variable that matters

Whether a pixel is labelled is decided by trackability, and trackability is decided by appearance. The
features are functions of the same appearance. The unlabelled pixels therefore do not differ from the
labelled ones by chance: they occupy a region of feature space that the labelled set does not reach.
Three statements follow.

- *Abstention is observed everywhere.* The propensity $\pi(u) = \Pr\bigl(\delta_{\min}(u) \le \delta \mid \phi_j(u)\bigr)$
  can be fitted on all pixels without any label. It measures how far a pixel is from the labelled
  support.
- *Reweighting cannot bridge a gap in support.* Importance weights $1/\pi$ correct a shift only where
  $\pi > 0$. On glass and water $\pi$ is near zero and there is nothing to reweight.
- *What can be tested is the trend.* $\delta_{\min}$ orders the pixels from sharply labelled to barely
  labelled. A head fitted on pixels with small $\delta_{\min}$ and evaluated on pixels with large
  $\delta_{\min}$, using their coarse labels, measures transfer across the conditioning axis inside the
  data, without ground truth. If accuracy falls towards chance as $\delta_{\min}$ grows, extrapolation
  to $\delta_{\min} = \infty$ is not supported, and the indicator on such pixels must be reported as
  prior-only with $\pi(u)$ attached. If it holds, the claim is still an extrapolation, and only exact
  depth can confirm it.

Weakly textured surfaces are the bridge. They are matchable at coarse scale with a large localisation
error, and long tracks and cross-strip pairs compensate: $\sigma_{\mathrm{tri}}$ scales with
$\sigma_a / (f \sqrt{\sum B_i^2})$. Such pixels can be labelled for large faults, and regional faults of
many $\sigma$ are large. How much of the texture-free area this recovers is a question of geometry and
of the correspondence model, to be measured.

### 5.3 Video and multi-view features cut both ways

A frozen video or multi-view encoder sees parallax in its features. For the head this is an advantage
on labelled pixels and a hazard for transfer: such features can encode "local depth from parallax
disagrees with the single-image impression", which is the label function itself, and that content is
absent exactly where the witness abstains. A head on multi-view features may thus score well on the
labelled set by a route that does not exist on the unlabelled set. The per-image features of the query
frame do not have this property. The comparison of the two feature sets across the $\delta_{\min}$ axis
is the direct test: multi-view features should win at small $\delta_{\min}$ and may lose their margin,
or fall below per-image features, at large $\delta_{\min}$.

## 6. Why Keep the Monocular Depth

In terms of the quantities above, the objection is sharp: where $\delta_{\min}(u)$ is small, the
triangulated depth is far more precise than the monocular one and makes it redundant; where
$\delta_{\min}(u)$ is large, the witness cannot judge. The monocular depth is needed only where it
cannot be checked. The answers that survive are these.

1. **Coverage.** The monocular depth is dense; the witness exists on a subset. The quantity that
   decides the matter is the fraction of the surface, by area, whose final geometry rests mainly on
   monocular depth. For a survey with 80/60 overlap over textured terrain this fraction may be small,
   and then the method addresses a small residual of the scene. For low overlap, single strips,
   corridor flights, forward motion (no parallax near the focus of expansion) and scenes dominated by
   glass, water or uniform surfaces it is large. The contribution is defined on that fraction and
   should be reported against it.
2. **Judging is cheaper than replacing.** A sparse, noisy witness suffices to *rate* a dense prior:
   the pooled regional test needs many fewer correspondences than a dense reconstruction, and a rating
   with a stated $\delta_{\min}$ is useful where a depth with that uncertainty is not.
3. **The label is the product.** The triangulated depths train an indicator that applies where
   triangulation is absent. This holds only as far as Section 5 holds.

A tension has to be resolved in the design and is noted here because it concerns the evidence. If the
triangulated depths are also fused into the surface, they are the best measurements available, but the
witness is then no longer independent of the surface it helps to judge. If they are kept out of the
fusion, good data are discarded. The clean construction is to split the correspondences (for instance
by alternating frames) into a part that is fused as sparse depth messages with variance
$\sigma_{\mathrm{tri}}^2$ and a part that judges, with roles exchanged across folds.

## 7. Frozen Models

All entries are recalled and unverified, including availability, outputs and licences. Several of the
models are recalled to be released under non-commercial licences.

| Role | Candidates (recalled) | Fits which regime | Open point |
| --- | --- | --- | --- |
| Point tracker | CoTracker family, TAPIR family | dense video; visibility and, for some, confidence output | behaviour at displacements of hundreds of pixels; cost of dense queries |
| Dense optical flow | RAFT family | dense video, pairwise | large displacement; no visibility output |
| Dense wide-baseline matcher | RoMa, LoFTR family | survey stills; dense warp with a certainty map | pairwise only, so tracks must be chained; uses frozen image-encoder features |
| Multi-view geometry model | DUSt3R and MASt3R, VGGT | survey stills; point maps and matching features | its depth is a learned prior as much as a measurement, so only its correspondences may serve as witness |
| Features for the head | per-image encoder of the DINO family; video encoder; multi-view model encoder | per-image always; video encoders need small inter-frame motion | Section 5.3 |

Two rules follow from Section 2. The correspondence must come from an unconstrained search, so that the
across-line residual is a test; a model that is given the poses and searches along the line has no
check. And the depth output of a multi-view model must not be used in place of triangulation: its
uncertainty is not set by parallax, and its errors on texture-free surfaces come from priors of the
same kind as the monocular model's.

## 8. Cost and Code

| Item | Exists in the code | To add |
| --- | --- | --- |
| Poses and intrinsics per view | `types.DepthFrame` (`K`, `T_wc`) | nothing |
| Images | `depthcache/datasets.View.load_rgb`; scripts under `experiments/` that prepare the synthetic block's images at half resolution | an image path into the frames used by `gs/`, which carry depth only |
| Projection, back-projection | `baselines/_common` (`backproject_cam`, `project`, `project_into`, `world_to_cam`) | epipolar split, parallax vectors |
| Alignment of a depth map to sparse reference depth | `depthcache/align.align_depth`: robust scale for metric depth, consensus fit affine in inverse depth for relative depth; written for sparse references with invalid pixels | call it with triangulated depth as reference; export inlier sets |
| Correspondence provider | nothing; no deep-learning framework is installed in the checked environment | a wrapper and an on-disk cache in the pattern of `depthcache/`; chaining of pairwise matches into tracks |
| Triangulation, $E$, $C$, strata, $\delta_{\min}$, $\epsilon$, propensity | nothing | a small array-level module; no learning except the propensity and the head |
| Consensus residual | `gs/gate.ViewGateTest` with block leave-out, cavity render, bias downdate | export of the per-pixel residual, which is formed internally but reduced to test statistics; the noise level is a single scalar, not a field |

The triangulation and its statistics are scalar problems per track and negligible in cost. The cost of
the witness is the cost of the correspondence model: for pairwise matchers, one forward pass per pair,
with about $2 n_f$ pairs per view for a track over $n_f$ frames plus cross-strip pairs. No timing is
known; for reference, the depth models cost 0.2 to 1.4 s per view on the recorded run.

## 9. What Is New, and What Is Not

Stated plainly: triangulating tracks with known poses, the across-epipolar residual as an outlier
test, and uncertainty from parallax are textbook material, and using multi-view geometry to supervise
or check monocular depth is established practice. The following points are, to the best of recalled
knowledge, not standard, and are where the contribution of this part lies.

1. **A witness that calibrates itself.** The epipolar split gives a depth measurement with $2n - 1$
   degrees of freedom of redundancy, from which the tracker's anisotropic noise is estimated per
   stratum with no labels. The resulting fault labels carry a per-pixel false-positive rate and a
   per-pixel minimum detectable fault. This replaces both the unknown constant of a correlation profile
   and the third independent detector that agreement-based identification would need.
2. **The minimum detectable fault as the axis of transfer.** Abstention is graded, the propensity is
   learned from frozen features without labels, and transfer from textured to untextured surfaces is
   measured inside the data along $\delta_{\min}$ before any claim is made beyond the labelled support.
3. **Limits of the epipolar self-check, with remedies.** Planar reflections, motion along the line and
   prior-driven tracks pass the check; the first is a non-identifiability, the third is detected by the
   evidence fraction $\epsilon$. A method that takes "on the epipolar line" for "trustworthy" trains
   its indicator on systematic false labels on reflective surfaces.

Whether the first two are sufficient for a strong venue depends on the measurements below; the third is
a necessary correction, not a contribution on its own.

## 10. What Would End the Idea, and the Cheapest Test

1. **No labelled faults of the kind that matter.** If faults that every view repeats occur almost only
   where $\delta_{\min} = \infty$, the witness labels nothing that consensus cannot already see, and the
   whole weight rests on extrapolation.
2. **Transfer fails along $\delta_{\min}$.** Head accuracy decays to chance as conditioning worsens.
3. **Prior-driven tracks cannot be separated.** Triangulation error on low-$\epsilon$ tracks is large
   and no diagnostic predicts it, so labels on weak texture are unfounded.
4. **Reflective surfaces poison the labels.** False labels at virtual depth dominate the positive class
   in scenes with glass and water.
5. **No frozen correspondence model works at the frame spacing of the data.**
6. **The monocular share is negligible.** The area whose geometry rests on monocular depth is a few
   percent on the target data.

**Cheapest test, no network and no GPU.** On the synthetic block with exact depth, poses and cached
monocular depth, build *oracle* tracks by reprojecting each pixel through the exact depth into the
other frames, with occlusion from the exact depth maps. This gives the true track geometry
$\mathcal S(u)$ and $\sum_i \lVert \mathbf m_i \rVert^2$. For assumed localisation noise of 0.5, 1 and 2
pixels compute $\delta_{\min}(u)$, and from the images the evidence fraction $\epsilon(u)$ with plain
gradients. Then tabulate the true faults of the cached monocular depth, split into "repeated by all
views" and the rest, against $\delta_{\min}$ and $\epsilon$. This is an upper bound on coverage: it
answers item 1 and item 6 directly, bounds item 3, and costs array arithmetic on data that are already
cached. If repeated faults lie almost entirely at low $\epsilon$, the idea is reduced to its
extrapolation before any model is run. The second step, only if the first passes, is one frozen
matcher on a single strip, to compare real tracks with the oracle tracks: coverage, calibration of $E$
and $C$, and triangulation error against $\epsilon$.

## 11. Not Known

- The frame spacing and overlap of the synthetic block, and so which correspondence regime it is in.
- Whether any frozen tracker or matcher delivers usable correspondences at survey frame spacing, and at
  what cost per view.
- The localisation noise of such a model on aerial imagery, its anisotropy, and whether the strata of
  Section 2.3 make it stationary enough for a $\chi^2$ reference.
- How often tracks on weak texture are prior-driven, and whether $\epsilon$ predicts their error.
- How the exact depth of the synthetic data treats glass and water.
- Whether faults repeated by all views exist in useful number on surfaces the witness can label.
- Whether accuracy along $\delta_{\min}$ says anything about accuracy at $\delta_{\min} = \infty$.
