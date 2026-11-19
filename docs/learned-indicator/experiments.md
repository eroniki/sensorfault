# Experiment Design: Information-Theoretic Gaussian Splatting as a Map for Correspondence-Free Aerial Mapping

This note says what has to be shown, and in what order, for a paper whose claim is a
representation and a mapping method: a large-area map built from posed aerial video and one
monocular depth map per frame, without feature matching, in which every Gaussian surfel stores
geometry, appearance and *information*, the last kept separately for each source of evidence.
The argument comes first in plain sentences; the definitions needed for scoring follow; then the
experiments, cheapest decisive test first. Notation follows the main document
(`docs/learned-indicator.md`, Section 7): placement $\boldsymbol\theta$ (offset along the normal,
two tilts), entity $m$, depth and image information $\Lambda^D$, $\Lambda^I$, tension $T_m$,
dispersion $\hat\varphi$, retained-information fraction $\kappa_m$, repetition coefficient
$\varrho$.

Status of numbers: *(repo)* is quoted from the repository; *(checked)* was read on a paper's or
project's own page; *(recalled)* was not; the rest is design choice or estimate. Nothing has been
measured. Every statement about what an experiment will show is a prediction.

## 1. The argument, and what it obliges the experiments to show

**1. A stored information is a forecast of error, so it can be scored.**
If the map says "this surface is known to 0.3 m along its normal", that is a prediction about the
distance between the mapped and the true surface. A prediction can be right or wrong
independently of whether the surface is accurate. *Consequence:* the first-class experiment of a
paper about information in the map is calibration of that information against exact geometry, and
the headline metric must reward geometry and information jointly. *Design:* the surface
log-score of Section 2.1.

**2. Information adds across sources only if the sources are honest and their errors are
independent.**
Every frame's monocular depth is produced by the same network looking at the same surface, so the
frames share an error. Adding their information counts that error as if it averaged out.
*Consequence:* a depth-only information map must be overconfident, by an amount set by the share
of the monocular error that every frame repeats. *Design:* this is predicted before any image
evidence is used, costs almost nothing to measure, and is the cheapest decisive test (Stage A).
If it fails, that is, if depth-only information is already honest, the paper has no problem to
solve on that data.

**3. A second source on the same primitives exposes what the first cannot see in itself.**
The pixels constrain the same three placement coordinates through rendering, with stiffness
equal to texture contrast times spread of viewing directions. The stretch between the two
springs is the only place where an error common to all depth maps appears. *Consequence:* keeping
information per source is not bookkeeping; it is what makes the between-source test possible, and
what tells a user *why* a surface is known. *Design:* calibration is reported per provenance
(depth only, image only, fused, fused after the gate), and the map must be best calibrated in the
last.

**4. Without correspondences, image evidence is local.**
Analysis by synthesis compares a render with a frame and follows the gradient. That gradient is
meaningful only while rendered and observed content overlap, within about a surfel footprint of
residual parallax. Explicit matching has no such limit. *Consequence:* a correspondence-free
mapper needs something else to place the surface inside the basin, and that is the role of the
monocular depth. The prior is kept because it is what replaces matching, and the pixels are
what make the prior honest. Neither is redundant. *Design:* the basin experiment (how much
surface the prior places within capture range, and what happens to the rest), and comparison
with matching-based pipelines on weak texture, where matches are missing, and on textured
surface, where they are strong and this method is expected to be the weaker one.

**5. "Absorbed" means marginalised, and marginalising costs information.**
Per-frame scale and offset, a shared distance bias, pose error per frame, the misfit of the
rendering model and the freedom of appearance can each be carried as nuisance in the same
information model. Doing so removes information in the directions the nuisance can mimic. What
is absorbed does not vanish: it reappears as a wider error bar. *Consequence:* the experimental
signature of absorption is not "the map stays accurate under pose error". It is "the map stays
*calibrated* under pose error, and its stated information falls by the predicted amount".
*Design:* injection ladders for each nuisance, scored on calibration and on predicted against
realised loss of information (Section 4).

**6. What a nuisance model cannot mimic must be detected.**
A wrong surface placement that every frame's depth repeats is not a scale, an offset, a pose or
a dispersion. No marginalisation removes it. *Consequence:* the fault test is the complement of
the nuisance model, and its false-alarm rate depends on the nuisance model being adequate.
*Design:* the gate is evaluated after absorption, and each nuisance left unmodelled is shown to
produce false alarms that modelling removes.

**7. A map that knows where it is uninformed can say so.**
On blank surface the image spring has no stiffness; on surface the prior fails on, the depth
spring is wrong. Where both hold, nothing in the data decides. *Consequence:* the map needs a
third state besides known and faulty, and that is where a head on frozen features has its only
job. *Design:* transfer is evaluated on exactly that surface, with an unsupported state.

## 2. What is measured: measures, units, nulls

### 2.1 Honesty of the information field

For entity $m$ with true placement error $\boldsymbol\epsilon_m$ (mapped minus exact, in metres
and radians) and stored covariance $\Sigma_m = \Lambda_m^{-1}$:

```math
z^2_m = \boldsymbol\epsilon_m^\top \Lambda_m\, \boldsymbol\epsilon_m \ \sim\ \chi^2_3 \ \text{ if the information is honest}, \qquad
\mathcal S = \frac{1}{\sum_m A_m} \sum_m A_m \Bigl[ \tfrac12 z^2_m - \tfrac12 \log\det \Lambda_m \Bigr] ,
```

with $A_m$ the entity's area. $\mathcal S$ is the negative log density of the true surface under
the map, in nats per entity on the signed normal error: the **surface log-score**. It is a proper scoring rule: it
cannot be improved by inflating or deflating the stated information, and it falls when either
the geometry or its error bar improves. Reported with it: the robust scale of $z_m$ (one if
honest; above one is overconfidence), coverage of the 68% and 95% regions, and the same per
surface class and per provenance. The scalar version along the normal is reported per surfel.

### 2.2 Alignment of geometry and appearance

The measure is the tension $T_m$ of the main document: the increase in misfit when the depth
family and the image family are made to share one placement. It is dimensionless; the gap
$\mathbf t_m$ behind it is in metres and radians; its null is $\chi^2_3$ after the image
information is deflated by the dispersion $\hat\varphi$. Three checks need no ground truth
(bulk of $T_m$ on image-strong entities, placebo tensions between halves of one family, response
to injected displacement) and one does (the distribution on truly healthy entities).

### 2.3 Honesty of the stiffness itself

Moving an entity by a known displacement and re-rendering gives the realised response of the
image misfit. Three numbers per surface class: the ratio of nominal to realised information; the
capture range in pixels of residual parallax over which the gradient has the right sign; the
retained-information fraction $\kappa_m$ under each appearance model.

### 2.4 In what sense correspondence-free, and how that is tested

No keypoints, descriptors, matches, tracks or triangulated points are computed or stored. No
pairwise structure between frames exists: each frame talks only to the primitives it sees, so
cost is linear in frames and a new strip is added by adding its messages. Correspondence is
implicit and local: two frames are related only through rendering the same primitives. This buys
additivity over large areas, evidence on weak or repetitive texture in proportion to its actual
contrast instead of all-or-nothing matching, and occlusion handled by the rasteriser. It costs
the basin: beyond the capture range the method is blind where matching is not.

"Cross view" has a second reading: views that differ widely, such as strips flown in different
headings, nadir against oblique, or passes at different times. The formulation covers different
headings directly, and they are valuable: they change the epipolar direction and so the
directions in which outlines and texture are informative. It covers widely different viewing
angles only with diffuse appearance, since projective texture from one view rendered into a very
different one meets foreshortening and view-dependent reflectance; and different times only if
the scene and lighting are unchanged. The block-3 loader already keys strips by heading
*(repo)*; whether the block's headings and tilt give a real cross-heading test is to be read
from its poses.

## 3. Experiments, cheapest decisive first

Data and code that exist *(repo)*: MatrixCity aerial block 3 (185 frames in strips,
$1920 \times 1080$, exact depth, 46% water); a depth cache with four views (Depth Pro 3-5%
median relative error after alignment, 0.54 s per frame); a depth-only surfel fit with
information blocks and a leave-block-out consensus gate (`gs/fit.py`, `gs/gate.py`), run so far
with synthetic depth noise at 96 by 54 pixels and 81 152 surfels (2.5 hours on CPU), where the
information was calibrated by construction; a separate photometric 2DGS trainer (1.5 million
Gaussians, 31 minutes, 3.2 GB, test PSNR 23.1 dB) whose surface is within 1 m on 81% of land and
5% of water cells; a surface comparison (`experiments/surface_dsm.py`); pose perturbation and
spline machinery (`experiments/h7_pose_check.py`); an OpenMVS wrapper. No joint depth-and-image
fit, no entity lifting, no feature encoder.

### Stage A. Is a depth-only information map dishonest, and by how much?

*Logic:* argument 2. *Cost:* one Depth Pro pass over 185 frames (under two minutes of GPU), then
the existing depth-only pipeline on CPU with real depth in place of synthetic noise, with the
per-frame alignment fitted against the fused surface.

- **A1.** Robust scale of $z$ and the surface log-score of the depth-only map against exact
  geometry, per surface class.
- **A2.** The same after the consensus gate has removed what it can.
- **A3.** The repetition coefficient $\varrho$ as a function of the left-out block, from true
  errors. The predicted overconfidence of the fused map follows from it: with $n$ frames on a
  surface and independent per-frame terms, the variance of the fused error is about
  $\varrho + (1 - \varrho)/n$ of a single frame's, where the map states $1/n$.

*Stops the line if:* the depth-only map is honest (scale of $z$ within $[0.8, 1.25]$) or the
consensus gate makes it so. Then on this data there is no shared error for a second source to
expose, and the test moves to another block and to real data before anything is built.
*Supports if:* overconfidence by a factor of two or more that the consensus gate does not remove
and that the formula above predicts within its interval.

### Stage B. Is the pixel stiffness honest, and how far does it reach?

*Logic:* arguments 3 and 4. *Cost:* the existing photometric model, one retrained copy with
restricted appearance, a few thousand renders; an estimate of two GPU-hours.

- **B1.** Injection ladder (0.25 to 32 px of residual parallax, offset and tilts) on groups of
  surfels: nominal against realised information; capture range; at three image scales; for
  colour and for frozen dense features as the rendered quantity.
- **B2.** Retained-information fraction $\kappa$ for: free view-dependent colour with free
  opacity; one diffuse colour per surfel with frozen opacity; projective appearance from source
  frames.
- **B3.** Where stiffness should be nil (water, shadow, uniform roofs): is it stated as nil?
  The existing photometric model places water a median 5.7 m above the true plane *(repo)*, so
  the unrestricted image family does not abstain there.

*Stops the line if:* no appearance model has both $\kappa > 0.5$ and nominal information within
a factor of three of realised on blank classes after dispersion; or the capture range, at every
scale, is below the residual parallax that the prior leaves on textured land (measured in A1).
In the second case the correspondence-free claim fails in its strict form and the method needs a
sweep of renders, which is a plane sweep by rasteriser.

### Stage C. Does the two-source map win on the surface log-score?

*Logic:* arguments 3, 6, 7. *Cost:* the joint model; the bulk of the new code.

- **C1, four maps.** Depth only; image only (from the same initial placement); both fused by
  information without a gate; both with the gate on the prior's placement component. Surface
  log-score, scale of $z$, accuracy and completeness, per provenance and per class.
- **C2, alignment measure.** Distribution of $T_m$ on healthy entities; detection of true
  placement faults, by repetition coefficient; against the consensus gate and against published
  devices for unreliable monocular depth in splatting (Section 5).
- **C3, the uninformed remainder.** Share of surface, weighted by harm, where both sources are
  silent or wrong; the head on frozen features there, with its unsupported state.

*Stops the line if:* fusion without a gate matches the gated map on the log-score (the
representation then needs information, not a test); or the fused map is not better calibrated
than the depth-only map; or image-only matches fused everywhere it converges and converges
wherever the prior does (the prior then adds nothing but an initial guess, which a coarser
method could give).

### Stage D. Nuisance: absorbed, estimated, or detected

One ladder per nuisance, each scored on accuracy, on calibration, and on predicted against
realised loss of information (Section 4).

### Stage E. Large area

- **E1, additivity.** Maps of adjacent tiles and of separate strips built independently and
  merged by adding messages, against a map built at once: difference in surface and in stated
  information.
- **E2, scale.** Several blocks; time and memory against area and frames; the claim is linear
  cost in frames and no global pairwise step.
- **E3, cross heading.** One heading held out as judge of a map built from the others.
- **E4, comparisons.** Matching-based structure from motion followed by splatting; multi-view
  stereo (OpenMVS); depth-regularised splatting with a confidence weight; a feed-forward
  multi-view model where it fits in memory; large-scale aerial splatting systems *(recalled;
  which have public code that runs on 8 GB is not checked)*. For baselines that state an
  uncertainty (Fisher-based or variational), the same calibration metrics. Baselines without one
  are scored on geometry only, and the log-score is reported with the best constant variance
  fitted for them on a disjoint block, which is generous to them.
- **E5, real data.** Aerial stills with laser depth (UseGeo; reader is a stub and the download
  pending *(repo)*): everything above on laser-covered surface, with real pose error.

### Follow-on table

| | Question | Metric | Baseline | Falsifiable hypothesis |
| --- | --- | --- | --- | --- |
| F1 | Basin: how much surface does the prior place within capture range? | share of surface, per class; convergence of placement from the prior against from a perturbed truth | initial placement from matching | at least 80% of textured land is inside the basin at the coarsest scale; faults outside it are found by the entity sweep, not by the gradient |
| F2 | Weak texture | log-score and completeness on the lowest-contrast quintile of land | matching-based pipelines | the method is more complete there and no less calibrated; on the highest-contrast quintile matching-based geometry is at least as accurate |
| F3 | Per-source information as explanation | agreement between stated provenance share and the realised change when a source is removed | none | removing the image family changes placement by an amount predicted by the image share within its interval |
| F4 | Frame spacing | all of Stage B and C on dense synthetic video at strides 1 to 16 | survey stills | capture range in pixels is constant over strides; the image share crosses one half at a predictable stride |
| F5 | Repeated faults | detection by repetition coefficient, where the image family is informative and where it is not | consensus gate, robust loss, frame removal, single-image baselines | consensus-based methods near chance at high repetition; the tension at AUC 0.9 where informative |
| F6 | Label budget for the head | calibration and ranking on uninformed surface at 0, 1, 5, 20 labelled frames | linear probe at equal budget | labels improve calibration there, not ranking |

Thresholds named in this section are conventions fixed before measurement.

## 4. Nuisance: what each ladder must show

| Nuisance | How it enters | Status | Cost in information | Experiment and its signature |
| --- | --- | --- | --- | --- |
| Per-frame scale and offset of the depth map | two parameters per frame, profiled out of the depth message | absorbed | each frame's depth carries no information on absolute level or uniform slope of what it sees; those come from the poses through the image family and from overlap | perturb scales and offsets per frame: geometry and calibration unchanged; stated information unchanged |
| Shared distance bias | one scalar, estimated jointly, projected against scale and offset | estimated | one direction in the whole map | inject a bias: recovered within its stated variance |
| Pose error per frame | six parameters per frame under a GNSS/IMU prior, or a low-order field along the strip; marginalised, $\Lambda_{\mathrm{eff}} = \Lambda_{\theta\theta} - \Lambda_{\theta p} (\Lambda_{pp} + \Sigma_p^{-1})^{-1} \Lambda_{p\theta}$, or refined on fitting frames | absorbed up to its prior, then estimated | modes in which a frame moves all its entities together; entity-to-entity relations are kept | ladder of pose noise from zero to several times the prior: with marginalisation the scale of $z$ stays near one and stated information falls as predicted; without it, overconfidence and false alarms of the tension grow |
| Rendering model misfit (finite primitives, resampling, correlated residuals) | dispersion $\hat\varphi$ per appearance stratum from the within-image scatter | absorbed as scale | divides the image information | $\hat\varphi$ from frames against the factor needed for honest $z$ on image-only maps |
| Appearance freedom | Schur complement over appearance; projective appearance for the test | absorbed | $1 - \kappa_m$ of the image information | Stage B2 |
| Exposure and gain per frame | affine colour per frame, or a feature-space loss | absorbed | contrast-free directions only | gain and gamma jitter: calibration unchanged |
| Error of the prior that every frame repeats on a surface | not a nuisance of any modelled kind | **must be detected** | none can be charged | Stage C2 |
| Reflections, transparency, moving objects | bias of the image family, consistent across frames in the worst cases | **partly detected** (outline against interior; within-image scatter), otherwise a stated limit | none can be charged | rendered reflective scenes; movers along and across the strip |
| Common error of all poses (datum) | invisible to every internal check | not absorbed, not detected | not applicable | stated; scored only against external reference |

The table is the paper's answer to "pose errors and model errors are absorbed": five rows
absorbed at a stated price, one estimated, three that the nuisance model cannot carry.

## 5. Prior devices the experiments must contain or beat

*(checked)* A method that supervises with monocular depth only where a rendered view and a
virtual stereo partner disagree; a method that scores monocular depth per segmented instance by
point density after multi-view fusion; a method that recalibrates monocular depth block-wise
against rendered depth in a visibility-based trusted region; per-Gaussian Fisher information of
the reconstruction loss used for pruning. The first three decide where to trust the prior from
depth or model self-consistency and are blind, by construction, to an error every view repeats;
the fourth computes an information of one source and does not test it. *(recalled)* Fisher
information for view selection, post-hoc spatial uncertainty, variational or Bayesian splatting,
splatting SLAM with pose refinement, large-scale aerial splatting. None of these was found to be
scored for calibration of surface uncertainty against exact geometry by source; that statement
rests on a limited search.

## 6. What a rendered survey block can and cannot show

- Exact poses, no exposure change and no noise favour the image family; decisions are read with
  degraded frames and perturbed poses.
- Survey spacing gives large parallax, so on textured land the image share is near one and the
  prior matters as initial placement only. Intermediate shares need dense video with exact
  depth, which is not on disk.
- Water is half the block, few entities, cut by frame borders; glass is too small at this range.
- A depth model may have seen similar renders in training; Stage A reports prevalence of gross
  error before anything else is concluded.

## 7. Protocol

1. Exact depth enters scoring and the construction of injected perturbations only.
2. Frames are split by ground tile into source, fitting and judging sets before any fit.
3. The appearance model for the test is fixed by Stage B before Stage C is scored.
4. Heads are trained on the scene's own tension labels and injection responses; labelled data,
   when used, come from scenes never evaluated. Depth model, segmenter and encoders are frozen.
5. Baselines receive the same frames and poses; constants fitted for them come from a disjoint
   block.
6. Intervals are cluster bootstraps over ground tiles, with the count of entities shown.
7. Evaluation data live in `results/k0/eval_only/`; thresholds and hypotheses are committed and
   tagged as `prereg/TAGGING.md` prescribes before the scoring script is first run.

## 8. Implementation: what exists and what must be built

| Stage | Needs | Exists | To build |
| --- | --- | --- | --- |
| A | depth cache; depth-only surfel map with information; scoring | `depthcache.run`, `gs/fit.py`, `gs/gate.py`, the E1 pipeline, `matrixcity_io`, `surface_dsm.py` | real depth in place of synthetic noise; alignment against the fused surface; $z$, log-score, $\varrho$ (about 300 lines) |
| B | photometric model, group moves, renders | `rgb_train_2dgs.py`, `rgb_prepare.py` | restricted and projective appearance; entity directions; information estimators (about 500 lines) |
| C | joint map at image resolution; entities; gate; head | parts of both trainers | information from the renderer's backward pass in place of the dense numpy path; mask and feature lifting; score test; four fits (about 1500 lines) |
| D | perturbation ladders | `h7_pose_check.py` | pose marginalisation in the entity information |
| E | several blocks, baselines, real data | OpenMVS wrapper | tile merge; baseline runners; UseGeo reader |

## 9. What this design cannot settle

- Whether the gradient reaches faults of realistic size at survey spacing; if not, detection is
  a sweep and the differentiable part serves repair only.
- Whether the gate adds anything to information-weighted fusion; Stage C can answer no.
- Bias of the image family (reflections, movers) is detected only in bounded cases.
- Calibration on surface where no source speaks is not identified without labels.
- Memory and time of a joint map on 8 GB, and which large-scale baselines run there, are
  unverified; the comparison set in E4 is partly recalled.
- Thresholds are conventions, not derived from a cost model.
