# Monocular Depth Judged by Motion Parallax: Objections, Prior Work, Verdict

Nothing in this note has been measured. Numerical examples are worked from stated assumptions and are
labelled as such. Statements about what existing models output or cost are marked *verified* (checked
against the paper's arXiv or proceedings page) or *recalled* (from memory, not checked).

## 0. The formulation under examination

**Setting.** A static scene is filmed from the air in strips, with dense frames and known poses. One
monocular depth map $\hat d_i$ per frame, from one frozen depth model. A surface of 2D Gaussian surfels
is fitted to the depth maps. A fault is a connected region of a view whose depth deviates from the
measurement model $\hat d_i = a_i(d_i + \beta d_i^2) + b_i + \varepsilon_i$ by many $\sigma_i$. The
faults of interest are those that every view of a region repeats, which no gate built on agreement
between depth maps can see.

**Witness.** A frozen point tracker or optical-flow model gives, for pixel $u$ of frame $j$, positions
$\mathbf x_t(u)$ in frames $t \in \mathcal T_j(u)$. With the poses, the track is triangulated to a
depth $d^{\mathrm{tri}}_j(u)$ with variance $\sigma^2_{\mathrm{tri}}(u)$. The track is accepted only if
it stays near its epipolar lines and the triangulation is well conditioned; otherwise the witness
*abstains*. Let $A_j(u) \in \{0,1\}$ be the acceptance flag, $\mathcal T = \{A = 1\}$ the tracked set
and $\mathcal U = \{A = 0\}$ the untracked set.

**Label.** With $(a_j, b_j)$ fitted robustly on the tracked pixels of frame $j$,

```math
\delta_j(u) = \frac{\hat d_j(u) - a_j\bigl(d^{\mathrm{tri}}_j(u) + \beta\, d^{\mathrm{tri}}_j(u)^2\bigr) - b_j}{\sqrt{\sigma_j^2(u) + a_j^2\, \sigma^2_{\mathrm{tri}}(u)}}, \qquad
y_j(u) = \mathbb 1\bigl[\lvert \delta_j(u) \rvert > \tau\bigr] \quad \text{on } \mathcal T .
```

**Head.** A light head $h_\theta$ on frozen features $\boldsymbol\phi_j(u)$ is trained on
$\{(\boldsymbol\phi, y)\}$ over $\mathcal T$ and evaluated everywhere. The indicator is $y$ (or its
soft version) on $\mathcal T$ and $h_\theta(\boldsymbol\phi)$ on $\mathcal U$. Labels from a synthetic
scene are optional calibration.

**Triangulation precision in a strip.** For translation parallel to the image plane, a point at depth
$d$ seen from camera positions $b_t$ along the strip has image positions $x_t = x_0 + f b_t / d$. With
independent localisation errors of standard deviation $\sigma_p$ per frame, the least-squares inverse
depth has

```math
\frac{\sigma_{\mathrm{tri}}}{d} = \frac{\sigma_p}{p_{\mathrm{tot}}}\, \sqrt{\frac{12\,(N-1)}{N\,(N+1)}}\; , \qquad p_{\mathrm{tot}} = \frac{f\, b_{\mathrm{tot}}}{d} ,
```

for $N$ equally spaced frames spanning total baseline $b_{\mathrm{tot}}$ ($N = 2$ gives the familiar
$\sqrt 2\, \sigma_p / p$). The independence assumption is optimistic: tracker errors drift and are
correlated along a track, so the gain with $N$ is smaller than the formula says. The total parallax
$p_{\mathrm{tot}}$, the track's travel across the image in pixels, is the quantity that matters.

## Part 1. Objections

Each objection is given with the reason it could be fatal, a test, and the change that answers it.

### V1. Where the witness speaks, the monocular prior is redundant

**Why it may be fatal.** In a nadir strip a ground point stays in view while the camera moves most of
an image footprint. Worked example, not a measurement: an image 4000 pixels long in the flight
direction, a point followed over 3200 pixels of travel in 5 frames, $\sigma_p = 1$ pixel. The formula
gives $\sigma_{\mathrm{tri}}/d \approx 0.04\%$. A monocular depth with $\sigma_j / d$ of 2% is fifty
times less precise, and a "many $\sigma$" fault is hundreds of triangulation standard deviations.
"Triangulation is sparse and noisy at small baseline" describes adjacent video frames. It does not
describe a full track, and a method that has the full track would use it.

Two consequences follow.

1. On $\mathcal T$ the label is close to ground truth. Nothing is learned there; the fault is measured.
2. If the triangulated depths enter the surface, as any reviewer will insist they should, their
   information exceeds that of the monocular message by the ratio
   $\sigma_j^2 / (a_j^2 \sigma^2_{\mathrm{tri}})$, of order $10^3$ in the example. Gating the monocular
   message on $\mathcal T$ then changes the surface by a part in a thousand. The verdict is decisive
   where it is worthless.

The whole benefit to the reconstruction therefore lies on $\mathcal U$, where there is no witness and
the indicator is the head's extrapolation. The paper cannot rest on detection scores over
$\mathcal T$.

**Test.** On a scene with exact depth: (a) the share of true fault area that lies in $\mathcal U$;
(b) surface error on $\mathcal T$ and on $\mathcal U$ for four fits: monocular only, triangulation
plus monocular ungated, triangulation plus monocular gated by the oracle mask, triangulation only with
no monocular depth at all. If the last fit matches the third everywhere, the monocular prior has no
role and neither does a gate on it.

**Change.** State the role of the prior as *coverage*, not precision: it constrains $\mathcal U$
(texture-free ground, water, glass, occlusion shadows of tall structures, strip ends, low-overlap
edges). Put triangulated depth into the surface as a second message type with its own information.
Define the indicator's job as deciding, on $\mathcal U$, whether the only available constraint should
be trusted. Report every result separately on $\mathcal T$ and $\mathcal U$.

### V2. The labels are missing not at random, and the head is asked to extrapolate off its support

**Why it may be fatal.** The head estimates $\Pr(f = 1 \mid \boldsymbol\phi, A = 1)$. What is needed is
$\Pr(f = 1 \mid \boldsymbol\phi, A = 0)$. The two coincide only under two conditions:

```math
\text{(ignorability)}\quad f \perp A \mid \boldsymbol\phi , \qquad\qquad
\text{(positivity)}\quad \Pr(A = 1 \mid \boldsymbol\phi) > 0 \ \text{ wherever } \mathcal U \text{ has mass} .
```

Both are doubtful. Abstention is caused by missing texture and by specular or transparent surfaces,
which are also the textbook causes of monocular depth failure, so $A$ and $f$ share a cause. If the
frozen features encode that cause, ignorability may hold approximately, but then positivity fails:
features of open water never occur with $A = 1$, the head has no training example near them, and its
output there is set by the regulariser. Nothing guarantees even the sign.

The mechanisms also differ. Faults on $\mathcal T$ are faults on textured, diffuse surfaces: relief
errors on roofs, vegetation, shadows mistaken for depth edges. Faults on $\mathcal U$ are caused by
absence of texture and by reflection. A head that learns "what makes this depth model fail on textured
surfaces" has learned about a different failure.

**Test.** Three tests, the first two without any ground truth.

1. *Support.* Train a classifier to separate features of $\mathcal T$ from features of $\mathcal U$.
   An AUC near 1 means the two sets do not overlap in feature space and the head extrapolates. Report
   the estimated propensity $e(\boldsymbol\phi) = \Pr(A = 1 \mid \boldsymbol\phi)$ as a map.
2. *Graded holdout.* Order tracked pixels by a texture or track-confidence score. Train the head on
   the well-textured half, test on the poorly textured tracked pixels, and plot accuracy against the
   score. A curve that falls towards chance as texture vanishes predicts failure on $\mathcal U$.
3. *Direct.* With exact depth, average precision of the head on $\mathcal U$, per surface class.

**Change.** See S1 and S2 in Part 2. In short: find witnesses inside hard regions so that positivity
holds at least sparsely, weight training by $1/e(\boldsymbol\phi)$, and let the indicator return a
third state, *unsupported*, where $e(\boldsymbol\phi)$ is below a floor.

### V3. On untracked pixels the head predicts propensity, not the event

**Why it may be fatal.** On $\mathcal U$ nothing observes whether this region's depth is wrong in this
view. An image-only head can say "this is water, and this depth model is usually wrong on water". Its
precision is capped at $\Pr(f = 1 \mid \text{surface class})$. If the depth model is wrong on 40% of
water, the gate discards a correct prior 60% of the time, on surfaces where nothing else constrains
the geometry. The claim about faults that every view repeats is fully met on $\mathcal T$, where it
does not matter (V1), and is met on $\mathcal U$ only as a class prior.

**Test.** $\Pr(f = 1 \mid \text{class})$ per surface class and per depth model on $\mathcal U$, with
exact depth. Within-class average precision of the head on $\mathcal U$: above the class base rate, or
not.

**Change.** Give the head inputs that carry information about the event and not only the class: the
frozen encoder of the depth model itself (its state may differ when it errs), and the monocular depth
map's own geometry relative to tracked surroundings (S2). If within-class precision stays at the base
rate, say so and present the output as a propensity.

### V4. The trivial rule "the tracker abstains, so distrust the prior" may match the head

**Why it may be fatal.** If faults concentrate on surfaces that defeat tracking, the flag $1 - A$ is
itself a fault detector on exactly the pixels that matter, with no learning at all. A refinement
costs nothing: split abstention by cause (no texture, epipolar violation, occlusion, short track). The
head must beat this on $\mathcal U$. If it does not, the learned component is decoration.

**Test.** Average precision on the whole image of: $1 - A$; abstention by cause; the head; the head
with abstention cause as an input.

**Change.** Use abstention cause as a feature and report the increment of the head over it.

### V5. The epipolar check certifies the wrong coordinate

**Why it may be fatal.** In a straight strip the epipolar lines run along the flight direction. The
check measures the track's cross-track deviation. Depth is encoded in the along-track position. A
track can pass the check and carry any depth error. Four cases pass:

1. *Static virtual images.* The reflection of a static scene in a planar mirror is a static virtual
   point. It satisfies every epipolar constraint and triangulates consistently, at the virtual depth.
   On glass and calm water the witness does not abstain; it testifies with confidence to the wrong
   surface, and the label then calls a correct monocular depth faulty.
2. *Objects moving along the flight direction.* A vehicle on a road parallel to the strip stays on its
   epipolar line and triangulates to a wrong height.
3. *Periodic structure along the flight direction* (lane dashes, parked cars, panel rows, row houses):
   a slip by one period is along the line.
4. *Edges parallel to the flight direction*: the aperture problem leaves the along-line position
   unconstrained.

The claim that a track leaving its epipolar line is "a reflection, a moving object or a tracker
failure" is true. Its converse, which the label relies on, is false.

**Test.** On synthetic scenes with mirrors, moving vehicles and periodic roofs: rate of accepted
tracks whose triangulated depth is wrong by more than $3\sigma_{\mathrm{tri}}$, by cause.

**Change.** Four additional checks, each cheap.

- *Along-line linearity* for $N \ge 3$ frames: $x_t - x_0$ must be proportional to $b_t$. Catches
  period slips; catches moving objects only if their speed varies.
- *Cross-strip agreement*: a point triangulated in two strips flown in different directions or at
  different times must agree. Catches moving objects.
- *Terrain envelope*: in nadir views a virtual image lies below the reflecting surface, so a
  triangulated point far below the surrounding tracked surface is flagged as specular. Sky reflected
  in water or a glass roof triangulates near infinity and is caught the same way.
- *Two layers*: two consistent depth populations inside one feature-homogeneous region (frames on the
  surface, virtual scene behind it) mark reflection or transparency.

None of these removes case 1 for a reflection whose virtual depth is plausible, such as a glass facade
reflecting the building opposite. That case stays as systematic label noise on glass.

### V6. This is a multi-view check on monocular depth, and that exists

**Why it may be fatal.** Each component has a direct ancestor. Fitting a per-view scale and offset of
monocular depth to triangulated points is standard practice in depth-regularised splatting; the
residual of that fit is the label proposed here. Training a confidence predictor from geometric
self-contradiction, with no ground truth, is established for stereo (Mostegel et al., CVPR 2016; Tosi
et al., BMVC 2017). Residual-supervised heads on frozen features exist for photometric outliers.
Tight fusion of monocular priors with multi-view geometry, with uncertainty, exists in
structure-from-motion (MP-SfM) and in multi-view depth (MaGNet). A reviewer will summarise the paper
as "an SfM-alignment residual, thresholded, distilled into a DINO head".

**Test.** Not empirical.

**Change.** The paper needs a claim that none of the ancestors makes. Candidates are in Part 2; the
only one with content is about *transfer of failure prediction to surfaces without parallax*, with
its conditions stated and tested.

### V7. Three frozen alternatives may make the indicator unnecessary

**Why it may be fatal.**

- *Depth completion conditioned on the triangulated points.* Frozen models take an image and sparse
  metric depth and return dense depth (OMNI-DC, Marigold-DC, Prior Depth Anything; all verified to
  exist as zero-shot methods, performance on aerial data not known). This uses the witness to
  *repair* the prior, which is strictly more than judging it. A fault indicator falls out for free as
  the gap between the plain and the conditioned prediction.
- *A feed-forward multi-view geometry model* (VGGT and related) returns depth and a confidence from
  several frames at once. If its features serve as the "video backbone", a reviewer will ask why its
  depth and confidence are not used directly in place of the monocular prior.
- *Dense multi-view stereo* is the standard product of aerial photogrammetry and is accurate on
  $\mathcal T$.

**Test.** Baselines, on $\mathcal U$ and overall: surface from completion-conditioned depth with no
gate; surface from the multi-view model's depth weighted by its confidence; gap-to-completion as an
indicator; against the proposed gate and the oracle gate.

**Change.** If completion wins, the right method is "condition, then gate the remaining
disagreement", and the indicator becomes a diagnostic of where conditioning changed the answer. The
honest position is that the indicator must add calibrated abstention to whatever dense depth is best,
not compete with it.

### V8. Gating on untracked pixels removes the only constraint

**Why it may be fatal.** On $\mathcal U$ a gated pixel has no depth message of any kind. The surface
there is whatever regularisation and neighbouring surfels produce.

This may be acceptable, and it should be tested and not assumed. For the classic fault classes the
fill from tracked surroundings is physically reasonable: still water is a level plane through its
shoreline, a glass facade is a plane through its frames. A wrong monocular depth is then worse than
no depth.

**Test.** Surface error on gated regions of $\mathcal U$: ungated monocular, gated with fill, oracle
gate with fill, completion model.

**Change.** Pair the gate with an explicit fill and report gated regions as filled, with their own
uncertainty.

### V9. The tracker is itself out of domain and too coarse

**Why it may be fatal.** Point trackers are trained largely on synthetic ground-level video, with
pseudo-labelled real video in recent models (CoTracker3, verified from its abstract). Nadir imagery
has small repeated objects and no dominant foreground. Trackers work at modest resolution: AllTracker
reports 768 by 1024 on a 40 GB GPU (verified from its abstract), which is far below an aerial frame.
Tiling restores resolution at a cost that grows with frames times tiles, and $\sigma_p$ in full-image
pixels grows by the downsampling factor if tiling is avoided. If the accepted set is thin or
$\sigma_p$ is several pixels, the label is neither dense nor clean.

"Weak texture where patch correlation abstains" is also where a learned tracker interpolates from
context, which is a smoothness prior and not a measurement. A tracker that reports confident tracks
on a texture-free roof has moved the abstention problem inside a network.

**Test.** On an aerial sequence with exact or laser depth: accepted fraction, $\sigma_p$ from
reprojection residuals, and error of $d^{\mathrm{tri}}$ against truth, as functions of local texture
and tile size, for one tracker and one flow model. Compare with plane-sweep correlation at equal cost.

**Change.** Choose the correspondence model by this test, not by benchmark rank. Keep plane-sweep
correlation as the reference witness.

### V10. Poses and tracks are not independent

**Why it may be fatal.** If the poses come from structure-from-motion on the same frames, pose error
produces a smooth, image-wide bias in triangulated depth (drift along a strip, doming over a block).
The per-view alignment $(a_j, b_j)$ and $\beta$ absorb part of it and convert the rest into regional
"faults" at image borders. The same absorption hides real errors: in nadir views all depths are close
to the flying height, so $a_j$ sets the scale of relief almost freely, and a depth model that
flattens every building by the same factor is perfectly aligned and never flagged.

**Test.** Perturb poses by their stated covariance and measure the change in labels. Report the
fitted $a_j$ against the true relief scale.

**Change.** Propagate pose covariance into $\sigma_{\mathrm{tri}}$. State that uniform relief
compression is outside the fault definition.

### V11. The fault definition and the label threshold are tied to the witness

The label is a threshold on $\delta$. Where $\sigma_{\mathrm{tri}}$ is comparable to $\sigma_j$ (short
tracks), the same true error gives a smaller $\lvert\delta\rvert$, so the positive rate depends on
track length, which depends on occlusion and position in the strip. The head can learn "long track"
from context. Train on pixels with $\sigma_{\mathrm{tri}} \ll \sigma_j$ only, or regress the signed
normalised error and threshold afterwards.

### V12. The effective sample is regions, not pixels

Faults are regional, frames of a strip repeat them, and the head sees the same roof in dozens of
frames. Splits and intervals must be by ground region. A head fitted and tested on the same scene
needs cross-fitting by ground tile.

### Summary of the attack

| Objection | Fatal if | Cheapest test | Needs exact depth |
| --- | --- | --- | --- |
| V1 redundancy | triangulation-only surface matches the oracle-gated one everywhere | four-fit comparison | yes |
| V1, V3 fault location | most fault area is in $\mathcal U$ and the class base rate there is moderate | fault area by set and class | yes |
| V2 support | tracked and untracked features separate with AUC near 1 and no witnesses fall in hard regions | domain classifier, witness count per feature cluster | no |
| V2 transfer | graded holdout falls to chance | graded holdout | no |
| V4 trivial rule | head does not beat abstention-by-cause on $\mathcal U$ | four-way comparison | yes |
| V5 label validity | accepted tracks on specular surfaces are often wrong | accepted-track error by cause | yes |
| V7 alternatives | completion-conditioned depth matches the gated surface | baseline runs | yes |
| V9 tracker | accepted fraction small or $\sigma_p$ of several full-resolution pixels | tracker audit | partly |

The two tests that need no ground truth (support and graded holdout) can be run on any posed aerial
video with a frozen tracker and a frozen backbone, and they bear on the objection most likely to be
fatal.

## Part 2. What could make this more than a multi-view check

Stated plainly: the witness and the label are not new. Using the residual between aligned monocular
depth and triangulated points as supervision for a confidence head is an application of a known
recipe to a new pair of models. If the paper is that, it is a workshop paper or a component of a
systems paper.

Two ideas in this area have some sharpness. Neither is established; both are cheap to test.

### S1. Treat witness abstention as a missing-data problem and make the support explicit

The standard recipe trains on pixels where geometry speaks and applies the result everywhere,
silently. The honest formulation is selection: labels are observed with propensity
$e(\boldsymbol\phi) = \Pr(A = 1 \mid \boldsymbol\phi)$, the target is identified on $\mathcal U$ only
under ignorability and positivity (V2), and both conditions have observable consequences.

- Estimate $e(\boldsymbol\phi)$ with the same frozen features; this is the domain classifier of V2.
- Train the head with weights $1/e(\boldsymbol\phi)$, clipped, so that rare tracked examples in
  feature regions typical of $\mathcal U$ count for what they represent.
- Output three states, not two: *trusted*, *faulty*, and *unsupported* where $e(\boldsymbol\phi)$ is
  below a floor and the head would extrapolate. For a downstream user the third state is information:
  it says the reconstruction there rests on an unverified prior.
- Use the graded holdout as a label-free check of transfer.
- If synthetic labels are used at all, spend them where self-supervision is silent: on surfaces of
  $\mathcal U$. That is a precise answer to what a few labels buy, and it differs from general
  calibration.

What is new here is the combination of a propensity-aware head with a stated support, for a
depth-prior failure predictor. Inverse-propensity weighting itself is textbook. The contribution
would be showing that it changes the result on parallax-free surfaces, with the conditions under which
it can.

### S2. Sparse witnesses inside hard regions, and a regional verdict

Positivity fails only if hard regions contain no accepted track at all. They often contain a few:
window frames and mullions on a glass facade, the shoreline and floating objects on water, lane
markings and stains on texture-free asphalt, seams on a metal roof. These triangulate to the true
surface. The fault is regional, so a handful of witnesses can carry a verdict for a whole region.

Let $\mathcal R$ be a region from a clustering of frozen features, with $n$ accepted witnesses. The
region statistic of the pixel-wise labels is

```math
z_{\mathcal R} = \frac{1}{\sqrt n} \sum_{u \in \mathcal R \cap \mathcal T} \delta_j(u) ,
```

which for a common offset of $m$ standard deviations has mean $m \sqrt n$: ten witnesses detect a
$1.5\sigma$ regional offset at about $4.7$. This assumes independent witnesses and a constant offset;
both are approximations. Where the interior has no witness, a weaker test remains: fit a smooth
surface (a plane for water and facades) to the witnesses on the region's border and compare the
monocular depth's interior against it. This is a model assumption and must be labelled as one, and it
is close to what a depth-completion model does implicitly (V7).

The regional verdicts serve three purposes: they are indicator values on part of $\mathcal U$ that
are measured and not extrapolated; they are the rare training examples that restore positivity for
S1; and they answer V3 in part, since they observe the event for this region and not the class.

Whether such witnesses exist in useful number is an empirical question about real aerial video. It is
answered by counting accepted tracks per feature cluster, without ground truth.

### What is not a contribution

- The epipolar self-check (V5 shows it is necessary and not sufficient; the extra checks are standard
  multi-view reasoning).
- A video or multi-view backbone as a feature extractor. It may help, but if its features already
  encode parallax, the natural question is why its own depth and confidence are not the baseline
  (V7).
- The combination rule. On $\mathcal T$ there is one witness and no combination.

## Part 3. Prior work

"Verified" means title, authors and venue or arXiv identifier were confirmed against a publisher,
proceedings or arXiv page. "Recalled" means from memory and not checked.

### 3.1 Self-supervised confidence from geometric contradiction

| Work | What it does | Relation |
| --- | --- | --- |
| Mostegel, Rumpler, Fraundorfer, Bischof, "Using Self-Contradiction to Learn Confidence Measures in Stereo Vision", CVPR 2016. *Verified (title, venue); co-authors recalled.* | Generates training labels for a stereo confidence classifier from contradictions between depth maps of different viewpoints, without ground truth. | The direct ancestor of "a geometric contradiction is a free label for a confidence head". The present formulation changes the judged quantity (monocular depth), the witness (tracks) and the features (frozen). |
| Tosi, Poggi, Tonioni, Di Stefano, Mattoccia, "Learning confidence measures in the wild", BMVC 2017. *Verified (title, venue); author list recalled.* | Self-labels confident and unconfident stereo pixels from agreement of conventional confidence measures, then trains a confidence network. | Same recipe. Its known weakness, labels available only where the cues are decisive, is V2. |
| Poggi, Aleotti, Tosi, Mattoccia, "On the Uncertainty of Self-Supervised Monocular Depth Estimation", CVPR 2020. *Verified (title, venue).* | Compares uncertainty estimators for monocular depth. | Single-image uncertainty; baselines only. |

### 3.2 Monocular priors combined with multi-view geometry

| Work | What it does | Relation |
| --- | --- | --- |
| Pataki, Sarlin, Schönberger, Pollefeys, "MP-SfM: Monocular Surface Priors for Robust Structure-from-Motion", arXiv 2504.20040, 2025. *Verified (arXiv); CVPR 2025 from a proceedings link, not checked further.* | Integrates monocular depth and normal priors into structure-from-motion with uncertainty propagation; priors help reject wrong matches. | The opposite direction of judgment (the prior judges matches) inside the same pairing of prior and triangulation. How its prior uncertainty is obtained was not verified; it must be read before any novelty claim. |
| Bae, Budvytis, Cipolla, "Multi-View Depth Estimation by Fusing Single-View Depth Probability with Multi-View Geometry" (MaGNet), CVPR 2022. *Verified.* | Uses a per-pixel monocular depth distribution to guide and weight multi-view matching. | Fusion with per-pixel monocular uncertainty, trained with supervision. Shows the fusion alternative to gating. |
| Guo et al., "Multi-view Reconstruction via SfM-guided Monocular Depth Estimation" (Murre), CVPR 2025. *Verified (title, venue).* | Conditions a monocular depth model on sparse SfM points to get multi-view consistent depth. | "Repair, do not judge" (V7), with a trained model. |
| Zhu and Liu, "Revisit Self-supervised Depth Estimation with Local Structure-from-Motion", arXiv 2407.19166, 2024. *Verified (arXiv).* | Uses triangulated sparse depth from a few frames as self-supervision to improve depth models. | Triangulation as teacher for depth itself; requires training the depth model, which is out of scope here. |
| Li et al., "MegaSaM: Accurate, Fast, and Robust Structure and Motion from Casual Dynamic Videos", arXiv 2412.04463, 2024. *Verified (arXiv).* | Camera and depth from casual video by a deep SLAM system adapted to low parallax and dynamics; uses monocular depth priors and learned motion probabilities (the last two points recalled). | A system that already weighs monocular priors against parallax by conditioning. |
| Depth-regularised splatting with per-view scale and offset fitted to SfM points (several systems). *Recalled.* | Aligns monocular depth to triangulated points, then uses it as a loss. | The alignment residual that these systems discard is the label proposed here. |

### 3.3 Zero-shot depth completion (the "repair" baseline)

| Work | What it does | Relation |
| --- | --- | --- |
| Zuo et al., "OMNI-DC: Highly Robust Depth Completion with Multiresolution Depth Integration", arXiv 2411.19278. *Title verified; authors recalled.* | Depth completion that generalises across sparsity patterns and domains. | Baseline for V7. |
| Viola et al., "Marigold-DC: Zero-Shot Monocular Depth Completion with Guided Diffusion", arXiv 2412.13389. *Title verified; authors recalled.* | Guides a diffusion depth model with sparse depth at test time, no retraining. | Baseline for V7; a frozen model conditioned on the witness. |
| Wang, Chen, Yang, Wang, Zhang, Zhao, Zhao, "Depth Anything with Any Prior", arXiv 2505.10565, 2025. *Verified (arXiv).* | Pre-fills sparse metric depth using a monocular prediction, then refines with a conditioned monocular model; zero-shot. | Closest existing "monocular plus sparse metric" fusion. The gap between its output and the plain monocular depth is a ready-made indicator. |
| Yu et al., "Large Depth Completion Model from Sparse Observations", arXiv 2605.30115, 2026. *Verified (arXiv).* | Transformer for metric depth and point maps from one view and sparse depth. | Same baseline family. |

### 3.4 Frozen correspondence and multi-view models

| Model | What was checked | Relevance |
| --- | --- | --- |
| Karaev, Makarov, Wang, Neverova, Vedaldi, Rupprecht, "CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos", arXiv 2410.11831, 2024. | *Verified:* authors, title, training on pseudo-labelled real video, tracking through occlusion. *Recalled:* outputs per-point visibility and confidence. | Candidate tracker. |
| Harley et al., "AllTracker: Efficient Dense Point Tracking at High Resolution", arXiv 2506.07310, ICCV 2025. | *Verified:* dense all-pixel tracks from flow between a query frame and every other frame; 768 by 1024 on a 40 GB GPU; 16 M parameters. *Not verified:* visibility or confidence outputs. | Candidate dense tracker; the stated resolution bounds cost on aerial frames (V9). |
| RAFT (Teed and Deng, ECCV 2020), SEA-RAFT (ECCV 2024), TAPIR (ICCV 2023), TAPNext (2025). | *Recalled.* | Flow and tracking alternatives. |
| Wang et al., "VGGT: Visual Geometry Grounded Transformer", CVPR 2025. | *Verified:* title, venue. *Recalled:* predicts cameras, depth, point maps and tracks with confidence from many views; memory limits the number of frames. | Candidate multi-view backbone, and the baseline of V7. |
| DUSt3R (CVPR 2024), MASt3R (ECCV 2024), Depth Anything 3 (2025), V-JEPA 2 (2025). | *Recalled.* | Multi-view geometry models with confidence; a video encoder without geometry output. |
| Goli et al., "RoMo: Robust Motion Segmentation Improves Structure from Motion", ICCV 2025. | *Verified:* title, venue. *Recalled:* combines optical flow with epipolar cues to mask moving regions. | Prior use of epipolar violation as a motion mask; says nothing about cases that pass the check (V5). |

No aerial evaluation of any tracker listed here was found. Song et al., "AerialMetric", arXiv
2606.29716 (ECCV 2026 per its arXiv comments; verified) reports a significant domain gap for metric
depth models on aerial viewpoints, which makes diffuse, prevalent error a live possibility.

### 3.5 Residual-supervised heads on frozen features, and reliability of depth priors in splatting

| Work | What it does | Relation |
| --- | --- | --- |
| Ren et al., "NeRF On-the-go", CVPR 2024. *Verified.* Sabour et al., "SpotLessSplats", arXiv 2406.20055. *Verified (arXiv).* Kulhanek et al., "WildGaussians", arXiv 2407.08447. *Verified (arXiv).* | Small heads or clusterings on frozen features, supervised by the photometric rendering residual. | The head architecture and the "residual as weak label" recipe. Their residual is available at every pixel; the present label is not, which is what V2 is about. |
| Zhang, Wysocki, Urban, Jutzi, "CDGS", arXiv 2502.14684, 2025. *Verified (arXiv).* | Hand-weighted sum of image and depth-gradient cues plus sparse SfM depth as confidence for depth regularisation. | Hand-built confidence using SfM depth; nothing learned. |
| Xiao et al., "In Depth We Trust: Reliable Monocular Depth Supervision for Gaussian Splatting", arXiv 2604.05715, 2026. *Verified (arXiv).* | Applies absolute monocular depth only where rendered depth is self-inconsistent; a gradient loss elsewhere. | Uses the prior exactly where geometry is ill-posed, with no check of the prior there. The complement of the present proposal, and a baseline. |
| Chu, Gopalan, Yuan, "Reliability-Aware Monocular Depth Supervision for Sparse-View Neural Reconstruction", arXiv 2607.02554, 2026. *Verified (arXiv).* | Restricts depth supervision by photometric masks from an RGB-only model. | A multi-view image witness used as a mask, without a learned extension to silent pixels. |
| Zhang et al., "MonoInstance", arXiv 2503.18363, 2025; Dufera and Cai, "ConfidentSplat", arXiv 2509.16863, 2025. *Verified (arXiv).* | Confidence of monocular depth from cross-view agreement of the depth maps. | Consensus of the priors themselves; blind to repeated errors. |

No work was found that trains a failure predictor for a frozen monocular depth model from track
triangulation and evaluates it on surfaces without parallax, or that treats the witness's abstention
as a selection problem. The search was not exhaustive; MP-SfM and the stereo-confidence literature
after 2017 were not read in full, and absence here is weak evidence.

### 3.6 Fault detection by redundancy

Baarda's data snooping (1968, recalled) and solution separation in receiver autonomous integrity
monitoring (Blanch et al., ION GNSS 2012 and IEEE TAES 2015; titles and venues verified) state the
limit this work starts from: a fault common to all redundant observations needs an independent source.
Motion parallax is such a source for monocular depth. The same literature states its price: the
independent source protects only where it observes.

### 3.7 Label models

With one witness there is no agreement pattern to model, and the latent-class literature
(Dawid and Skene 1979, recalled; Parisi et al., PNAS 2014; Fu et al., ICML 2020; Mazzetto et al.,
AISTATS 2021; the last three verified) no longer applies to the method. It remains relevant only if
depth consensus and the witness are combined on $\mathcal T$, which V1 argues is pointless.

## Part 4. Verdict

**Is it publishable?** The detection result on tracked pixels is not: it is a measurement, and a
known one. The formulation is publishable at a robotics venue as a system if the reconstruction on
untracked surfaces improves against the baselines of V7. It reaches a top vision venue only if the
transfer result of S1 and S2 is positive: a failure predictor trained from parallax alone that is
shown, with ground truth, to work on surfaces where parallax is silent, together with the conditions
under which it can and a support map that says where it cannot. If transfer fails, the honest paper is
a negative result with the three-state indicator, and that is a weaker paper.

**The single claim.** *A frozen monocular depth prior can be audited by motion parallax from the same
video at no labelling cost, and the audit extends to surfaces without parallax to the extent that
sparse witnesses inside them restore support; where it does not extend, the indicator says so.*

**Answer to "why keep the monocular prior".** Not because triangulation is noisy; over a full track it
is far more precise. The prior is kept because it is the only dense constraint on $\mathcal U$. The
method's value is confined to $\mathcal U$ and must be shown there.

**Cheapest tests, in order.**

1. *No ground truth, one posed aerial video, one frozen tracker, one frozen backbone.* Accepted
   fraction; accepted tracks per feature cluster; domain-classifier AUC between $\mathcal T$ and
   $\mathcal U$; graded holdout curve.
2. *Exact depth, no head.* Fault area in $\mathcal T$ against $\mathcal U$; fault base rate per
   surface class on $\mathcal U$; accepted-track error by cause; the four-fit comparison of V1.
3. *Exact depth, with head.* Head on $\mathcal U$ against abstention-by-cause, a probe with a few
   labels, the gap to a completion model, and the oracle.

**Kill criteria.** Abandon the learned indicator if any of the following holds:

- the triangulation-only surface matches the oracle-gated surface on $\mathcal U$ as well as on
  $\mathcal T$ (the prior has no role);
- hard regions contain no accepted witnesses and the domain classifier separates the sets (no
  support, so nothing can be learned for $\mathcal U$ without labels);
- the head does not beat abstention-by-cause on $\mathcal U$;
- completion-conditioned depth with no gate matches the gated surface;
- the fault base rate on the surface classes of $\mathcal U$ is moderate and within-class precision
  of the head stays at that base rate (the gate would discard as much good prior as bad).

**Variants to drop.**

- Reporting detection on tracked pixels as the headline.
- The epipolar check as a sufficient validity test.
- A multi-view geometry model used only as a feature extractor, without its own depth and confidence
  as a baseline.
- Combining depth consensus with the witness through a latent-class model.

**What stays unknown.** Whether real aerial video offers witnesses inside glass, water and
texture-free regions in useful number, and whether a head trained on them predicts the event and not
only the surface class. Everything else in the formulation is either known to work or irrelevant to
the result.

## Sources consulted

- https://openaccess.thecvf.com/content_cvpr_2016/html/Mostegel_Using_Self-Contradiction_to_CVPR_2016_paper.html
- https://bmva-archive.org.uk/bmvc/2017/papers/paper133/index.html
- https://openaccess.thecvf.com/content_CVPR_2020/html/Poggi_On_the_Uncertainty_of_Self-Supervised_Monocular_Depth_Estimation_CVPR_2020_paper.html
- https://arxiv.org/abs/2504.20040
- https://openaccess.thecvf.com/content/CVPR2022/html/Bae_Multi-View_Depth_Estimation_by_Fusing_Single-View_Depth_Probability_With_Multi-View_CVPR_2022_paper.html
- https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Multi-view_Reconstruction_via_SfM-guided_Monocular_Depth_Estimation_CVPR_2025_paper.html
- https://arxiv.org/abs/2407.19166v2
- https://arxiv.org/abs/2412.04463
- https://arxiv.org/pdf/2411.19278
- https://arxiv.org/abs/2412.13389
- https://arxiv.org/abs/2505.10565
- https://arxiv.org/abs/2605.30115
- https://arxiv.org/abs/2410.11831v1
- https://arxiv.org/abs/2506.07310 and https://mlanthology.org/iccv/2025/harley2025iccv-alltracker
- https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html
- https://openaccess.thecvf.com/content/ICCV2025/html/Goli_RoMo_Robust_Motion_Segmentation_Improves_Structure_from_Motion_ICCV_2025_paper.html
- https://arxiv.org/abs/2606.29716
- https://openaccess.thecvf.com/content/CVPR2024/html/Ren_NeRF_On-the-go_Exploiting_Uncertainty_for_Distractor-free_NeRFs_in_the_Wild_CVPR_2024_paper.html
- https://arxiv.org/abs/2406.20055
- https://arxiv.org/abs/2407.08447v2
- https://arxiv.org/abs/2502.14684
- https://arxiv.org/abs/2604.05715
- https://arxiv.org/abs/2607.02554
- https://arxiv.org/abs/2503.18363
- https://arxiv.org/abs/2509.16863v1
- https://pubmed.ncbi.nlm.nih.gov/24474744/
- https://arxiv.org/pdf/2002.11955.pdf
- https://proceedings.mlr.press/v130/mazzetto21a.html
- https://web.stanford.edu/group/scpnt/gpslab/pubs/papers/Blanch_et_al_IONGNSS_2012_v2.pdf and https://dblp.uni-trier.de/rec/journals/taes/BlanchWELPRSK15.html
