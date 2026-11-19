# Stage A task list: is a depth-only information map overconfident, and by the predicted amount?

Stage A decides hypotheses H1 and H2 of the protocol (`docs/learned-indicator/experiment-design.md`, §3 and §7.2;
config `configs/stage_a_depth_only_honesty.yaml`). It is the first go/no-go of the plan (`PLAN.md` §6, milestone
M0): if a depth-only map is already honest, the problem the paper addresses does not exist on this data (stop
rule S1).

- **H1.** After each frame's depth noise is calibrated so that a single frame is honest, the fused depth-only map
  is still overconfident: the robust scale $s_z$ of the standardised placement error exceeds 1.25.
- **H2.** The excess is the repeated share: the realised variance of the fused error is
  $\varrho + (1-\varrho)/n$ times the single-frame variance, where $n$ is the number of frames that see an
  entity and $\varrho$ the share of error variance common to them.

Everything here uses the depth prior only. No image term, no renderer backward pass and no new method code are
needed, which is why this stage comes first.

## Inputs and conventions

| Item | Value |
|---|---|
| Data | MatrixCity small-city aerial block 3: 185 training frames in strips, exact depth and poses |
| Depth model | Depth Pro (primary). Metric3D v2, UniDepthV2, Depth Anything V2 repeat the analysis and set the scope only |
| Resolution | depth point-sampled at step 20 (96 by 54), as in experiment E1; step 10 as a resolution check |
| Entity | ground tile of 8 m side, split by depth-normal class; placement = normal offset and two tilts |
| Split unit | 64 m ground tile; checkerboard of calibration and held-out tiles with a one-tile buffer |
| Classes | water, roof, ground, and contrast quintiles of land; from exact depth and source images, for reporting only |
| Statistics | 95% bootstrap over held-out 64 m tiles, 10 000 resamples |
| Where it runs | A2 needs a GPU for one depth pass (about 2 minutes for Depth Pro, 10 for the other three); everything else is CPU |

Ground truth is used for evaluation and for the single-frame noise scale on calibration tiles. It never enters
the per-frame alignment or the map.

## Tasks

Each task lists what is built, where, and the check that closes it. Order is the order of work; A1 to A4 can run
in parallel with A5.

### Data and bookkeeping

**A1. Strip table and tile map.** Extend `experiments/matrixcity_io.py` to return, per frame, its strip key
(heading and cross-track coordinate) and its index along the strip; add a tile map that assigns ground points to
8 m entities and 64 m split tiles, and the checkerboard split with buffer.
*Done when:* a table of strips (count, frames per strip, spacing, overlap) is written to
`results/stage_a/strips.md`; every frame has exactly one strip; unit tests cover the tile assignment and show
that no held-out tile touches a calibration tile.

**A2. Monocular depth for block 3.** Run `scripts/depth_cache.py` for the 185 training frames with Depth Pro,
then the other three models. The reader `depthcache/datasets.py::matrixcity_views` currently lists the 19 test
frames; extend it to the training split.
*Done when:* the cache manifest reports 185 views with status `ok` per model, and a smoke test loads one view
per model through `load_depth_frames`.

**A3. Class labels.** From exact depth: water (planar, below the water level), roof (upward normal above 6 m),
ground (other land). From the source images: contrast quintile of land entities by mean gradient energy.
*Done when:* every entity has one class; class areas are reported; classes with fewer than 50 entities are
flagged as carrying no interval claim.

**A4. Calibration module.** New `src/sensorfault/eval/calibration.py`: robust scale
$s_z=\sqrt{\operatorname{median}(z^2)/2.366}$ for three degrees of freedom, coverage at 0.683 and 0.95,
expected calibration error, surface log-score per entity on the signed normal error, and the tile bootstrap.
*Done when:* unit tests on synthetic Gaussian placements recover $s_z=1$ within 3% at 5 000 entities, the
nominal coverages within 1 point, and an interval that covers the truth in about 95% of repeated draws.

### The depth-only map

**A5. Label-free per-frame alignment.** Fit each frame's scale and offset, and the shared distance term,
against that frame's rendering of the current fused depth-only map (Huber loss, five alternations), starting
from a fit to the median of overlapping frames. `depthcache/align.py` has the closed-form affine fit; the
alternation and the shared term go next to `gs/fit.py`.
*Done when:* on a synthetic scene with known scale and offset per frame the alternation recovers them to 1%; on
block 3 the alignment converges (change below 0.1% in the last alternation) and the residual against exact depth
is reported next to that of an oracle alignment, so the cost of having no ground truth is known.

**A6. Depth-only surfel map from real depth.** Run the depth-only fit (`gs/fit.py`, fuser `NoGate`) on the
aligned Depth Pro maps. With the reference backend at 96 by 54 this is CPU work of a few hours.
*Done when:* the fit completes on all 185 frames, and accuracy and completeness against exact depth are reported
per class.

**A7. Per-entity records.** For every entity and frame: the lumped depth information $\Lambda^D_{mi}$ and the
placement error of that frame alone, $\boldsymbol\epsilon_{mi}$, against exact depth; and for the fused map,
$\Lambda^D_m$ and $\boldsymbol\epsilon_m$. Placement error is the plane correction (offset and two tilts) that
best maps the entity onto the exact points; the shape remainder is stored separately.
*Done when:* records are written to `results/stage_a/records.npz` with a schema test; merging the records of two
halves of the frames reproduces the joint record to $10^{-6}$ relative (additivity).

**A8. Single-frame calibration.** On calibration tiles, scale each frame's $\sigma_i$ so that the single-frame
standardised error has $s_z=1$. This removes ordinary miscalibration of the depth noise, so that what remains in
the fused map can only come from frames sharing their error.
*Done when:* single-frame $s_z$ on held-out tiles lies in $[0.9, 1.1]$ for land. If it does not, the noise model
(not the fusion) is at fault and is fixed before going on.

### The two measurements

**A9. H1: overconfidence of the fused map.** On held-out tiles, entities seen by at least five frames: $s_z$,
coverage, expected calibration error and log-score of the depth-only map, per class, with intervals.
*Decision:* supported if the lower bound of $s_z$ exceeds 1.25; refuted if the upper bound is at most 1.25.

**A10. Repetition.** Correlation $r(s)$ of single-frame placement errors of one entity between frames $s$ apart
along a strip, and between strips; from it the repeated share $\hat\varrho(B)$ using only pairs more than $B$
frames apart, for $B \in \{0,1,2,3,5,9\}$, per class.
*Done when:* $r(s)$ and $\hat\varrho(B)$ are tabulated with intervals, and a permutation of frame labels gives
$\hat\varrho$ indistinguishable from zero (the estimator is not biased upward).

**A11. H2: the law.** Bin entities by $n$. Per bin, the realised variance of the fused error over the
single-frame variance, against the predicted $\hat\varrho + (1-\hat\varrho)/n$ and the nominal $1/n$.
*Decision:* supported if the interval of realised over predicted lies in $[0.67, 1.5]$; refuted if the upper
bound of $\hat\varrho$ is below 0.1 or the ratio lies outside $[0.5, 2]$.

### Controls and outputs

**A12. Consensus gate.** Repeat A9 after the depth-consensus gate (`gs/gate.py`, block $B=3$ primary; 1, 5, 9
secondary) has removed what it flags and the map is refitted. H1 is refuted only if the map is honest after this
step too.
*Done when:* the A9 table has a second row, with the number of frames the gate removed.

**A13. Synthetic control.** Replace the monocular depth by exact depth plus independent noise of the calibrated
level, and by exact depth plus noise with a planted repeated share of 0.3. The pipeline must report
$s_z \approx 1$, $\hat\varrho \approx 0$ in the first case and recover 0.3 in the second.
*Done when:* both are within their intervals. Without this control a positive H1 could be an artefact of the
pipeline.

**A14. Scope and resolution.** Repeat A5 to A11 for the other three depth models and for step 10.
*Done when:* one table per model; conclusions are stated for Depth Pro only and the others are labelled scope.

**A15. Residual parallax.** The 90th percentile, on textured land, of the image displacement in pixels between
neighbouring frames that the aligned depth leaves against exact depth. This is the capture range the pixel
information must reach in the next stage.
*Done when:* the number is reported per image scale.

**A16. Report and figure data.** `experiments/stage_a_honesty.py` runs A5 to A15 from the cache and writes
`results/stage_a/summary.md`, the JSON behind it, and the data of the law figure (variance ratio against $n$;
$\hat\varrho$ per class).
*Done when:* a clean checkout with the cache reproduces the summary with one command and fixed seeds.

## Order and effort

| Step | Tasks | Effort (estimate) | Needs |
|---|---|---|---|
| 1 | A1, A3, A4, A13 scaffold | 2 days | nothing; CPU |
| 2 | A2 | 15 minutes of GPU, plus setup | the GPU workstation and the block-3 training images |
| 3 | A5, A6, A7 | 3 days, a few CPU hours of fitting | step 2 |
| 4 | A8, A9, A10, A11 | 2 days | step 3 |
| 5 | A12, A13, A14, A15 | 2 days | step 4 |
| 6 | A16 | 1 day | all |

## What the outcome means

| Outcome | Consequence |
|---|---|
| H1 and H2 supported | The paper's first claim stands on this block; go on to the pixel information |
| H1 supported, H2 not | The map is overconfident for a reason other than repetition; find it (A13 and A8 are the first suspects) before any further stage |
| H1 refuted | Stop rule S1. Repeat on real scenes with laser depth before concluding; if it holds there too, the paper becomes a negative result |
| A8 fails | The single-frame noise model is wrong; fix it and rerun, since H1 is undefined without it |

## Open choices to fix before scoring held-out tiles

- The water level and the roof height that define the classes.
- Entity side (8 m) and split tile (64 m): the split tile must exceed the correlation length that A10 measures; if
  it does not, enlarge it and redo the split before any held-out number is read.
- The Huber threshold of A5.
- Whether block 3 alone is used (tables marked "same scene") or further blocks are obtained as a test set.
