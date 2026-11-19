# Experiment Design: Protocol, Hypotheses and Decision Thresholds

This document fixes what `docs/learned-indicator.md` (Section 5) leaves open: thresholds of the stop
criteria S1 to S8, data splits, metrics, fault protocols, baselines, work packages and empty result
tables. Every number is a convention fixed before measurement or a cost estimate; none is a result.
"pred." marks a theoretical prediction. Assumptions about data not yet on disk are stated with the step
that checks them.

## 1. Decision rules common to all hypotheses

- Each hypothesis has one primary metric, one threshold $\tau$ and one direction. It is **supported** if
  the 95% two-sided bootstrap interval of the metric lies wholly on the supported side of $\tau$,
  **refuted** if wholly on the other side, and **inconclusive** otherwise. A stop criterion fires only on
  *refuted*. On *inconclusive* the single permitted action is to add held-out blocks or scenes (up to the
  cap in Section 2) and recompute once; the result after that is final.
- Thresholds, strata, appearance regime and block size $B$ are fixed on dev data and committed with the
  tag of `prereg/TAGGING.md` before scoring on test data.
- $n$ is the number of frames that see an entity, $z^2_m=\boldsymbol\epsilon_m^\top\Lambda_m\boldsymbol\epsilon_m$,
  $s_z=\sqrt{\operatorname{median}(z^2)/2.366}$, $\sigma_1$ the single-frame robust error scale of the
  depth model on land. $\chi^2_3$ quantiles: 0.683 at 3.53, 0.95 at 7.815, 0.99 at 11.345.

## 2. Datasets and splits

### 2.1 What each dataset serves

| Dataset | Role | Stages | What it cannot test |
| --- | --- | --- | --- |
| MatrixCity aerial blocks (synthetic, exact depth and poses) | exact truth for placement error, injection, calibration | 0, A to E | real pose and exposure error, real materials, illumination change between passes; the depth model may have seen similar renders |
| UseGeo (real UAV, LiDAR depth maps) | real replication of A, C, D on laser-covered land | A, C, D, E | exact depth on water and glass; faults below LiDAR accuracy; small frame spacing |
| ETH3D, 13 scenes, 454 views | real images, laser depth, exact poses; stiffness on real texture; real repetition of depth error | A, B, small C | nadir altitude, strips, large area, additivity at scale |
| Proprietary flight | qualitative figure only, if publishable | none | everything quantitative |

### 2.2 MatrixCity

*Assumption.* Only block 3 (185 training frames in strips, 19 off-grid test views, 1920 by 1080, 46%
water) is on disk. **Dev:** block 3. **Test:** the three lowest-indexed other aerial blocks with float32
depth and RGB, chosen before any is opened; downloads need approval. If fewer are obtained the test set
shrinks, not the rules; with only block 3, scoring is on held-out tile folds with a one-tile buffer and
every table is marked "same scene".

*Tiles.* 64 m square ground tiles (assumption: at least three times the largest fault footprint and above
the monocular error correlation length, which Stage A checks) are the unit of splitting, bootstrap and
cross-fitting. Block 3 is split by checkerboard into **calibration tiles** (fit $\hat\varphi$, honesty
corrections, heads, single-frame temperature) and **held-out tiles** (scored); judging frames follow the
source and judging fold rule of the plan, Section 4.3.

*Strips.* Strip key is the `(yaw, cross-track coordinate)` of `experiments/matrixcity_io.py`; WP1 prints
the strip table per block. *Same-strip separation* $s$ is the number of frames between two frames of one
key. *Cross strip:* same heading, different key. *Cross heading:* yaw differs by 90 degrees or more; if a
block has none the cross-heading tests are not run on it. *Held-out pass (E3):* one heading judges a map
built from the others. Appearance change between passes is synthesised (Section 5); no real illumination
change exists.

*Classes* (evaluation only, from exact depth): water (planar, below $-3$ m), roof (upward normal, above
6 m), ground (other land), and contrast quintiles of land by mean gradient energy of source frames over
the entity. A class with fewer than 50 entities gets no interval claim.

### 2.3 UseGeo

*Assumptions, checked in WP7 before use:* three sets of 8 strips (224, 328, 277 images, about 2 cm
sampling, 80% and 60% overlap, LiDAR depth per image), adjacent strips flown in opposite directions, no
crossing lines. **Dev:** set 1. **Test:** sets 2 and 3 (if sets overlap on the ground, split by tile with
buffer instead). Images downsampled to about 1600 by 1070. Cross strip means adjacent strips; no cross
heading exists, so the rank-repair test of the label channel is MatrixCity only. Natural fault regions
are connected regions of at least 1% of an image where aligned depth error exceeds $3\hat\sigma_1$
against LiDAR, classed (water, glazing, blank roof, other) by manual inspection without model output;
entities whose true error is below $3\times$ the LiDAR scatter are not labelled faults.

### 2.4 ETH3D

All 13 scenes with the four-model depth cache. Dev: courtyard, facade, office, relief (every third scene
of `configs/datasets/eth3d.yaml`); test: the other nine. **Pseudo-strip:** views sorted along the first
principal axis of camera centres, cut where spacing exceeds twice the median. *Other pass:* optical axis
more than 30 degrees from the reference view. Laser depth is sparse and in distorted geometry, so it is
warped with the stored calibration and entities are restricted to laser-covered pixels.

## 3. Hypotheses

Stages and stop criteria are those of the plan. "Pred." marks a theoretical prediction used as the
threshold's source.

| ID | Stage | Hypothesis (falsifiable) | Primary metric | Threshold and justification | Stop |
| --- | --- | --- | --- | --- | --- |
| H1 | A | A depth-only information map is overconfident. | $s_z$ of the depth-only map after single-frame calibration, entities with $n\ge 5$, Depth Pro, held-out tiles | Supported if lower bound $>1.25$: $z^2$ inflates by $1+(n-1)\varrho$ (pred.), so at $n\approx6$ a repetition share of 10% gives $s_z\approx1.22$ and 1.25 is the smallest effect of that size. Refuted if upper bound $\le1.25$ after the consensus gate | S1 |
| H2 | A | Overconfidence is explained by repetition: realised variance of the fused error equals $\varrho+(1-\varrho)/n$ times the single-frame variance. | ratio $R_{\rm real}/R_{\rm pred}$ per $n$-bin, $\varrho$ estimated beyond block $B$ | Supported if the interval lies in $[0.67,1.5]$. Refuted if $\hat\varrho$ has upper bound $<0.1$ or the ratio lies outside $[0.5,2]$ | S1 |
| H3 | 0, B | The image record recovers an injected placement. | median relative error of recovered offset, per channel and scale | Supported if $\le0.25$ for $p\le P^\star$, with $P^\star\ge$ the 90th percentile of residual parallax the prior leaves on textured land (Stage A); 25% keeps a 4 sigma fault within one sigma of its recovered size | S2 |
| H4 | B, C | Realistic faults carry energy. | noncentrality $\lambda=\Delta^2\Lambda^D\vartheta/(1+\vartheta)$ at the 75th percentile surface term of A, median textured-land entity, after $\hat\varphi$ | Supported if $\lambda\ge11$ (80% power at 5% size, $\chi^2_3$; 15.5 at 1%). Refuted if $<5$ | S2 |
| H5 | B | Some appearance regime keeps a fault and renders healthy surface well. | absorbed fraction of an injected fault; judging-frame PSNR loss against the unrestricted model | Supported by a regime with absorbed fraction $\le0.5$, $\kappa\ge0.5$, PSNR loss $\le1$ dB (12% in squared error). Refuted if none | S3 |
| H6 | B | Stated image stiffness is honest on blank surface after dispersion. | ratio nominal to realised information on water, roofs in the lowest contrast quintile | Supported if in $[1/3,3]$ (the unrestricted model places water 5.7 m off). Not a stop | S4 |
| H7 | C | The fused map is calibrated after label-free corrections. | $s_z$, coverage at 0.683 and 0.95, ECE, on held-out tiles | Supported if $s_z\in[0.8,1.25]$ (interval inside $[0.7,1.4]$), coverages within $\pm0.08$ and $\pm0.04$ of nominal (three to four binomial standard errors at 400 tiles), ECE $\le0.05$. Refuted if $s_z>1.5$ or log-score no better than depth-only | S4 |
| H8 | C | Gates add to saturated information fusion. | share of fault-affected entities with $z^2>11.345$ after fusion, gated over ungated-saturated | Supported if ratio $\le0.5$ with upper bound $<0.8$. Refuted if upper bound $\ge1$ or the best decaying-weight map has ratio $\le$ that of the gated map on at least half of the fault cells | S5 |
| H9 | C | Repeated faults are visible to the tension but not to consensus. | AUROC of $T_m$ and of the consensus statistic on repeated faults | Supported if AUROC$(T)\ge0.9$ where $\vartheta\ge0.5$, and consensus AUROC $\le0.6$ at $\varrho\ge0.8$. Refuted if consensus AUROC $\ge0.85$ | S5 |
| H10 | C | Energy has its stated null. | mean $T/3$ and exceedance of 7.815 on healthy entities | Supported if mean in $[0.85,1.15]$ and exceedance in $[0.035,0.075]$; placebo (depth half against depth half) same bounds | S4 |
| H11 | E | Matching is not better everywhere. | completeness at $\tau$ and median accuracy by contrast quintile, against SfM+MVS with the same poses | Supported if completeness exceeds matching by $\ge 5$ points in the lowest quintile with accuracy ratio $\le 1.5$. Refuted if matching has completeness $\ge$ ours and accuracy $\le$ ours in all five quintiles on all test data | S6 |
| H12 | D | Calibration holds under nuisance. | $s_z$ at each ladder level; ratio of realised to predicted retained information | Supported if $s_z\in[0.8,1.25]$ up to 2 times the pose prior, 0.1 log scale error, gain jitter 0.2, and realised/predicted retained fraction in $[0.67,1.5]$. Refuted if $s_z>1.5$ at 2 times the prior in two of three ladders | S7 |
| H13 | C | Entities beat tiles. | AUROC and captured fault noncentrality, mask entities against tiles of equal count and area | Supported if AUROC gain $\ge0.05$. Refuted if upper bound $<0.03$ on natural faults and on injected faults that are not aligned to masks | S8 |
| H14 | E | Records are additive. | relative Frobenius difference, merged against joint information | $\le10^{-6}$ (float64), $\le 10^{-3}$ (float32): arithmetic, not statistics | none |
| H15 | B | The prior places most textured land inside the capture range. | share of textured-land entities with residual parallax below $R_{80}$ at the coarsest scale | $\ge0.8$ | S2 |
| H16 | C | Heads transfer to pixel-blind entities. | AUROC on counterfactually blinded entities against distance from support | Supported if $\ge0.7$ at median distance and *unsupported* beyond the support threshold; refuted if no better than a linear probe | none |

## 4. Metrics

**Placement error.** The exact points of all exact depth maps inside the footprint of entity $m$ are
fitted by least squares as a plane correction to the mapped surface in the entity frame; its three
coefficients (offset, and tilts times the footprint half-width, all in metres) are $\boldsymbol\epsilon_m$,
so $\Lambda_m$ is in m$^{-2}$ for every method. The fit residual is the *shape remainder*, reported
separately. Stage A entities are blocks of $4\times4$ surfels (16 m); later stages use mask entities.

**Surface log-score.** $\mathcal S=\sum_m A_m[\tfrac12 z_m^2-\tfrac12\log\det\Lambda_m]/\sum A_m$, nats per
square metre without the constant $\tfrac32\log2\pi$; also per entity. With $c=\overline{z^2}/3$,
rescaling $\Lambda$ by $1/c$ lowers $\mathcal S$ by $\tfrac32(c-1-\log c)$, the **calibration loss**; the
remainder is sharpness. Comparisons are paired differences per tile.

**Standardised-error scale.** $s_z$ as defined; also $\sqrt{\overline{z^2}/3}$ for tail sensitivity.

**Calibration.** $u_m=F_{\chi^2_3}(z^2_m)$ is uniform if honest. The calibration curve is the empirical
CDF of $u_m$ at 19 levels 0.05 to 0.95; $\mathrm{ECE}$ is the mean absolute deviation from the diagonal.
Coverage is the share of entities with $z^2\le3.53$ and $\le7.815$.

**Repetition coefficient.** With errors $\boldsymbol\epsilon_{mj}$ of frame $j$ on entity $m$ (from that
frame's aligned depth, against exact points), the pair correlation at same-strip separation $s$ is
$r(s)=\langle\boldsymbol\epsilon_{mj},\boldsymbol\epsilon_{mk}\rangle/\langle\|\boldsymbol\epsilon_{mj}\|^2\rangle$
pooled over entities; $\hat\varrho(B)$ is the mean of $r$ over separations above $B$; cross-strip $r$
is separate. The realised fused variance ratio is
$R_{\rm real}=\overline{\|\bar{\boldsymbol\epsilon}_m\|^2}/\overline{\|\boldsymbol\epsilon_{mj}\|^2}$ with
$\bar{\boldsymbol\epsilon}_m$ the error of the depth-only fused entity, and $R_{\rm pred}=\hat\varrho+(1-\hat\varrho)/n$
(stated: $1/n$). A second, label-free estimate uses the three terms of the misfit split: $\hat a=W_D/(3(n-1))$
from the within-depth term, $\hat m=T'/3-\hat a/n-\hat\imath$ from the between term with image scatter
$\hat\imath$ from the within-image term, $\hat\varrho_{\rm lf}=\hat m/(\hat m+\hat a)$, all in units of
single-frame variance. Their agreement is reported per $B$.

**Energy null.** On healthy entities (true error below $2\sigma_1$ and not in an injected region): mean
$T/3$; fraction above 7.815 and 11.345; a QQ curve against $\chi^2_3$ at 99 quantiles. Placebo tensions: depth halves against each other, image halves against each other.

**Detection.** An entity is a *fault* if it lies in an injected or labelled natural region and its true
error exceeds $2\sigma_1$. Reported: AUROC, AUPRC, TPR at 5% FPR on healthy entities of disjoint tiles,
harm-weighted recall with harm $h_m=A_m\|\boldsymbol\epsilon^{\rm fused}_{\rm fault}-\boldsymbol\epsilon^{\rm fused}_{\rm clean}\|$,
the power curve against noncentrality, and false alarms per 100 entities on fault-free scenes.

**Accuracy and completeness.** Digital surface models as in `experiments/surface_dsm.py` (highest point
per cell). MatrixCity: 2 m cells, $\tau\in\{0.5,1,2\}$ m. UseGeo: 0.25 m cells (at least three LiDAR
points per cell), $\tau\in\{0.1,0.25,0.5\}$ m. Accuracy is the median and 95th percentile of absolute
height error on covered cells; completeness the share of reference cells within $\tau$.

**Capture range.** $R_{80}$, per channel and scale, is the largest injected parallax $p$ for which at least
80% of entities in the stratum recover the correct sign with error below 50% of $|\delta|$. Parallax is
the root mean square over judging frames of the difference between judging and source frame of the image
displacement of the entity centroid, in pixels.

**Uncertainty.** The resampling unit is the 64 m ground tile. For multi-block results resample blocks (or
scenes or sets), then tiles within block, 2000 replicates, percentile intervals; seeds and frames are
averaged within a tile first. Paired comparisons use the same resampled tiles. With at most four blocks
the between-block interval is wide: per-block values are always shown and no generalisation beyond them
is claimed.

**Multiple comparisons.** The 16 hypotheses are separate pre-specified decisions, each with its own
threshold and an interval rule instead of a $p$ value, so no pooled correction applies. Secondary
per-class tables use Benjamini and Hochberg at $q=0.1$ and are marked descriptive; per-entity detection
within a scene controls the false discovery rate at 5%; ablation rows are reported as paired effect sizes.

## 5. Fault and nuisance protocols

**Injected faults at depth level.** A region $\mathcal R$ of $K\in\{16,64,256\}$ surfels (about 2%, 8%,
32% of a frame footprint). In every frame that sees $\mathcal R$ the depth inside its projection is
replaced by that of the plane displaced by $\boldsymbol\delta$. *Repeated:* same $\boldsymbol\delta$ in all
frames. *Independent:* $\boldsymbol\delta_j$ drawn per frame with equal norm. *Partial:* same
$\boldsymbol\delta$ in a random 50% of frames. Magnitudes $\|\boldsymbol\delta\|\in\{0.5,1,2,4,8\}\sigma_1$;
types offset, tilt, combined (tilts giving the same displacement at the region edge). A cell is size
(3) $\times$ magnitude (5) $\times$ type (3) = 45 per mode. One fit holds the 45 faults of one mode,
spaced by at least three region radii and covering at most 15% of the area; 5 seeds per mode, 3 modes:
15 fits per block. Healthy tiles supply false alarms.

**Injected faults at map level (Stage B).** After the depth-placed fit, the surfels of a group of
$K\in\{16,64,256\}$ move coherently by $\boldsymbol\delta$ and every frame that sees the group is
re-rendered. Residual parallax $p\in\{0.25,0.5,1,2,4,8,16,32\}$ pixels; offset, tilt 1, tilt 2, both
signs; 40 groups per cell; image scales full, half, quarter; channels colour and frozen dense features
(32 principal components, projection fitted on the dev block). The recovered displacement is the
minimiser over $\boldsymbol\delta$ of the statistic and the curvature there is the realised information.

**Natural faults.** Water, glazing, blank roofs and ETH3D regions as labelled in Section 2; their shape
is not chosen by the design, so they decide H13.

**Nuisance ladders**, one nuisance at a time on block 3 and one test block, five seeds, level 0 clean:

| Ladder | Levels | Form |
| --- | --- | --- |
| Pose | $k\times$ prior, $k\in\{0,0.5,1,2,4\}$, prior $(0.25^\circ, 5\ \mathrm{GSD})$ (assumption; navigation specification on real sets) | independent per frame; smooth along the strip (spline, 5-frame correlation); blunder variant: 3% of frames at $10\times$ |
| Depth scale, offset | scale log-sd $\{0,0.02,0.05,0.1,0.2\}$; offset sd $\{0,0.5,1,2,5\}\%$ of altitude; bias $\{0,\pm2\%\}$ | per frame |
| Exposure | gain log-sd $\{0,0.1,0.2,0.4\}$; gamma $\{0.8,1,1.25\}$; per pass: gain $\times1.3$ and white-balance shift | per frame, per pass |
| Combined | pose $1\times$, scale 0.05, gain 0.2 | together |

Predicted retained information is $\operatorname{tr}\Lambda^{\rm marg}/\operatorname{tr}\Lambda$ (plan, Section
4.6); realised is the ratio of median stated information with and without the nuisance. Detection at fixed
5% false alarm is re-measured at each level.

## 6. Baselines and ablations

Each row says what it isolates and which claim falls if it matches the method.

| Baseline or ablation | Isolates | Falls if it matches |
| --- | --- | --- |
| Depth-only fit (`gs/fit.py`, `NoGate`) | any image evidence | claim (iii); H7 |
| Image-only from the same initial placement | the prior beyond initialisation | the prior is a basin-finder only |
| Depth plus image, fixed weight | information weighting | H8 |
| Decaying depth weight (schedule fitted on dev) | calibration against a schedule | S5 |
| Information fusion, nominal blocks, no gate | saturation | H8 |
| Saturated honest prior, no gate | what the gate adds | H8 |
| Consensus gate among depth maps (`gs/gate.py`, $B\in\{1,3,5,9\}$) | the between-source test | H9 |
| Robust depth loss; removal of whole frames | cheap robustness | H8 |
| Selective depth supervision (virtual-stereo mask, photometric-error mask, cross-view depth consistency, heuristic confidence), reimplemented from their descriptions | published selection devices | H8, H9 |
| Single-image fault detector, trained on injected faults of the dev block (baseline only) | learned appearance detector against multi-view test | detection claim |
| Tiles of equal count and area | entity as unit | H13, S8 |
| Single summed confidence | per-source storage | contributions 1, 2; H9 |
| SfM+MVS, same poses (COLMAP, OpenMVS via `baselines/openmvs.py`) | correspondence-free against matching | H11, S6 |
| Fisher and Bayes' Rays style information fields | alternative information, same log-score | novelty of per-source information |
| Oracle gate with true masks and faults | upper bound | none |
| No honesty correction ($\hat\varphi=1$); no pose marginalisation; single image channel; no layer test; no support head; $B=1$ | each part | its row of the ledger |
| Depth model swap (Depth Pro, Metric3D v2, UniDepthV2, Depth Anything V2) | model dependence | scope of H1, H2 |

Baselines with no stated uncertainty are scored on geometry, and on log-score with the best constant
variance fitted on calibration tiles.

## 7. Execution plan

GPU hours are estimates for one 24 GB GPU scaled from earlier repository runs (31 minutes for the
photometric trainer on 185 frames at 960 by 540; 0.54 s per frame for Depth Pro; 2.5 CPU hours for the
depth-only fit at 96 by 54), not measurements on this hardware.

| WP | Content | Code to build or extend | GPU h | Wall-clock | Needs | Stop that can fire |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Stage 0: synthetic entity | `gs/reference.py` (colour render, image Jacobian), new `gs/image_info.py`, `tests/core/test_image_info.py`, `experiments/stage0_entity.py` | 0 (CPU) | 3 days | none | S2 (early signal) |
| 1 | Data: strip tables, RGB loader, Depth Pro cache, other three models, 64 m tile map, class labels | `experiments/matrixcity_io.py` (RGB, tiles, strips), `depthcache/run.py`, `eval/calibration.py` | 1 | 1 day | block 3 on disk; approval for more blocks | none |
| 2 | Stage A on block 3 | `gs/fit.py` (real depth, alignment to fused surface, per-entity records), `eval/calibration.py`, `experiments/stage_a_honesty.py` | 0 (about 20 CPU h) | 2 days with 8 cores | 1 | **S1** |
| 3 | Stage B: injection ladder, appearance regimes | `experiments/rgb_train_2dgs.py` (restricted, projective appearance), new `gs/inject.py` | 14 | 2 days | 0, 1 | S2, S3 |
| 4 | Stage C joint map and fault grid | `gs/fit.py` (information from the renderer's backward pass), `gs/entities.py` (SAM 2 mask and feature lifting), `gs/fusers.py` (saturated, decaying, fixed, image-only), `gs/gate.py` (between-source test, attribution), `faults/injectors.py` (region faults, three modes), `eval/stats.py` (tile bootstrap, paired, BH) | 80 | 6 days | 2, 3, more blocks | S4, S5, S8 |
| 5 | Stage D ladders | `experiments/h7_pose_check.py` pattern; pose marginalisation | 35 | 3 days | 4 | S7 |
| 6 | Baselines | `baselines/_colmap.py` (extend the model writer to a full triangulation and MVS runner), `baselines/openmvs.py`, new `baselines/detector.py`, Fisher and Bayes' Rays fields | 30 | 4 days (parallel with 4) | 1 | S6 (block 3) |
| 7 | UseGeo download, reader, real-data repeats of A and C | `depthcache/datasets.py` (reader), `experiments/` runners | 40 | 5 days | approval, 4 | S1, S4, S5, S6 on real data |
| 8 | ETH3D stiffness and repetition | cache exists; `experiments/stage_a_honesty.py` adapter | 10 | 2 days | 3 | none |
| 9 | Stage E: additivity, cost vs area, cross heading | tile merge in `gs/entities.py` | 12 | 2 days | 4 | none |

Total about 220 GPU hours (about 10 days of continuous single-GPU time); with inspection and reruns plan
five to seven weeks. Earliest firing: S2 (early signal) after WP0, S1 after WP2 (about one week after
start), S3 after WP3, S6 on block 3 after WP6, S4, S5, S8 after the first clean and fault fits of WP4,
S7 after WP5.

### 7.1 Stage 0, to start immediately

*Scene.* A planar entity of $20\times20$ surfels at 1 m spacing, textured by band-limited noise (spectrum
slope $-2$, contrast 5%, 15%, 40% of mean), nine nadir frames at $H=100$ m, focal 512 px, 256 by 256
pixels (0.195 m per pixel), 10 m spacing (80% overlap), exact poses.
1. **Jacobian check.** Analytic $\mathbf g_{uk}\mathbf j_{uk}$ of `image_info.py` against central finite
   differences of the rendered colour for all 400 surfels; pass if median relative error $\le10^{-2}$.
2. **Recovery.** Inject $(\nu,\tau_1,\tau_2)$ ($\nu\in\{\pm0.1,\pm0.3,\pm1\}$ m, tilts with equal edge
   displacement); texture from frames 1, 3, 5, 7, 9, judge on 2, 4, 6, 8; one Gauss-Newton step from
   $\Lambda^I,\boldsymbol\eta^I$. Pass if the slope of recovered on injected is in $[0.8,1.2]$ with
   $R^2\ge0.9$ for $p\le1$ pixel.
3. **Dispersion.** 200 replicates (texture seed, pixel noise 2/255, surfel scale 1.0 and 0.5 of spacing,
   gain jitter 0.1). Pass if the coefficient of variation of $\hat\varphi$ is $\le0.3$ and mean $z^2/3$ of
   recovered against injected error is in $[0.8,1.25]$.

Stage 0 refutes S2 early if recovery fails at every contrast for $p\le1$ pixel, or $\hat\varphi>100$ at
contrast 40% (information divided by two orders of magnitude, so H4 cannot be met).

### 7.2 Stage A, to start immediately

Block 3, 185 training and 19 test views; Depth Pro depth point-sampled at step 20 (96 by 54) as in E1,
and at step 10 as a resolution check.
1. `depthcache/run.py` for Depth Pro (about 2 minutes) and the other three models (about 10 minutes).
2. Per-frame scale and offset fitted without ground truth against the frame's rendering of the current
   fused depth-only map (Huber loss, five alternations); shared distance bias jointly.
3. Depth-only surfel map (`gs/fit.py`, `NoGate`), per-frame $\sigma_i$ scaled on calibration tiles so
   that single-frame $s_z=1$; record per entity $\Lambda_m$, $\boldsymbol\epsilon_{mj}$, $\boldsymbol\epsilon_m$.
4. Consensus gate with $B=3$ (primary; 1, 5, 9 secondary), refit.
5. On held-out tiles: $s_z$, log-score, calibration loss, coverage, ECE per class and model (H1);
   $r(s)$, $\hat\varrho(B)$ for $B\in\{0,1,2,3,5,9\}$, $R_{\rm real}/R_{\rm pred}$ per $n$-bin and
   $\hat\varrho_{\rm lf}$ (H2); the 90th percentile of residual parallax of the aligned depth in pixels
   (input of H3, H15).
6. Decide H1 and H2 on Depth Pro; the other models set the stated scope only.

## 8. Tables and figure data the paper needs

Empty tables (column headers only).

| T1 Stage A | model | class | $n$ bin | entities | $s_z$ | coverage 0.683 | coverage 0.95 | ECE | log-score | calibration loss | $\hat\varrho$ | $R_{\rm real}$ | $R_{\rm pred}$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| T2 Stiffness | channel | scale | regime | stratum | nominal/realised | $R_{80}$ (px) | $\kappa$ | absorbed fraction | PSNR loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| T3 Maps | dataset | map | provenance | $s_z$ | coverage 0.95 | log-score | accuracy median | accuracy P95 | completeness at $\tau_1$, $\tau_2$, $\tau_3$ | entities |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| T4 Faults | dataset | mode | size | magnitude | method | AUROC | TPR at 5% FPR | harm-weighted recall | false alarms per 100 | noncentrality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| T5 Natural faults | dataset | class | entities | method | AUROC | attribution accuracy | unsupported share | $s_z$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

| T6 Nuisance | ladder | level | $s_z$ | realised retained | predicted retained | TPR at 5% FPR | false alarms per 100 |
| --- | --- | --- | --- | --- | --- | --- | --- |

| T7 Against matching | dataset | contrast quintile | method | accuracy median | completeness at $\tau$ | $s_z$ | time (h) |
| --- | --- | --- | --- | --- | --- | --- | --- |

| T8 Ablations | row | $s_z$ | log-score | AUROC repeated | AUROC independent | accuracy | stop affected |
| --- | --- | --- | --- | --- | --- | --- | --- |

| T9 Cost and additivity | tiles | frames | area (km$^2$) | time (h) | memory (GB) | merged/joint difference |
| --- | --- | --- | --- | --- | --- | --- |

Figure data (one file each): F1 `B, s, r, lo, hi, class, model` and `n, R_pred, R_real, lo, hi`; F2
`provenance, p, ecdf, lo, hi, class`; F3 `channel, scale, p, delta_true, delta_hat, curvature` and
`channel, scale, class, R80`; F4 `quantile, T_quantile, chi2_quantile, kind`; F5 `method, mode, fpr, tpr`
and `noncentrality, power, method`; F6 `ladder, level, s_z, retained_pred, retained_real, tpr`; F7
`entity_id, x, y, vartheta, share_depth, share_image, verdict, tension`; F8 `tiles, frames, time, memory`.

## 9. Threats to validity

| Threat | Control |
| --- | --- |
| Synthetic-to-real gap: exact poses and exposure, synthetic materials | Decisions repeated on UseGeo and ETH3D; pose, exposure and scale ladders; claims without a real replicate are labelled synthetic-only |
| Calibration and scoring on one scene | Dispersions, corrections, heads and temperature fitted on calibration tiles, scored on buffered held-out tiles, cross-fitted by tile; heads see raw features only; test blocks opened after the tag; exact depth read only by scoring code and injectors |
| LiDAR error and missing returns on water and glass | LiDAR scatter is the noise floor; water and glazing scored on MatrixCity; UseGeo scored on laser-covered land only |
| Monocular model trained on test data | Four depth models reported; prevalence of gross error reported first; UseGeo and ETH3D are real and likely unseen; claims conditioned on the model |
| Faults too easy or aligned to entities | Three repetition modes, regions not aligned to masks, natural faults decide H13, fault-free scenes |
| Few blocks | Per-block results shown; bootstrap over blocks; no generalisation beyond them claimed |
| Baseline handicap | Same frames, poses and tuning budget on dev; reimplementations checked on a toy case |
| Thresholds chosen after results | Tag before test; changes logged as deviations; one enlargement only |
| Absorbed directions hide faults | Faults filling most of a frame are reported as indistinguishable from pose or scale |
| Silent data corruption | Inputs hashed before and after each run |
