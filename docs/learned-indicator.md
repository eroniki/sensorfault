# Learned fault indicator (research track)

Status: scoping, 2026-11-18. Nothing built yet. Branch `research/learned-indicator`, kept apart from
`main` and from the RA-L letter, which keeps the statistical gate. Full scoping doc (editable):
https://claude.ai/artifact/EFe4tF4ns3xNsJbWdQaU8j

**Scope (M, 2026-11-18): self-supervised and unsupervised learning only.** No fault labels in training
or model selection. Exact depth, injected masks and LiDAR are for evaluation only; select models by
held-out likelihood, never by detection rate.

## Idea

Replace the statistical / information-theoretic decision of the view gate with a learned indicator
function, fitted without labels. Keep the leave-block-out cavity (fusion, block subtraction, Newton step,
support rules) as the evidence; it is also the self-supervision (predict view `j` from the other views).

- **Output:** a per-pixel fault probability `zeta_j(u)` for the tested view `j`, pooled to a view score.
- **Inputs (feature stack):** cavity residual `r_j` (aligned, divided by `sigma(range, texture, incidence)`),
  whitened residual, rendered sd `sqrt(V_{-j})`, accumulated opacity, untestable mask, range and incidence,
  RGB of view `j`, depth-model confidence where it exists.
- **Replaces:** scene temperature gamma, heterogeneity omega, n_eff degrees of freedom, energy statistic,
  region scan + Bonferroni, Holm step-down.
- **Keeps a guarantee:** split-conformal threshold on development scenes assumed clean at scene level (as in
  H5), or a threshold from the fitted inlier model's own null. Open.

## Formulation options (no labels)

| Option | Learned | Note |
| --- | --- | --- |
| U1 (start here) | amortised contamination model: net gives outlier prior `pi(u)` and inlier scale `s(u)`; maximise sum over views of log[(1-pi) N(r; 0, s^2 (sigma^2 + V)) + pi p_out(r)]; `zeta` = outlier posterior | statistical gate is the limit case (pi const, s = gamma) |
| U2 | image-only student distilled from U1's `zeta` | fault prior on untestable pixels |
| U3 | density model (flow / autoencoder, trimmed loss) of residual patches | alternative |
| U4 | self-training on the statistical gate's flags | baseline |
| U5 | cross-depth-model disagreement as input or pretext | blind to shared errors |

Design rules for U1: the scale head `s(u)` sees geometric inputs only (range, incidence, rendered sd), never
RGB, or it can explain glass faults away as noise; `pi(u)` may see RGB. `p_out` fixed and normalised over a
bounded residual range; Beta prior on mean `pi`; cap on flagged fraction per view.

## Data

- MatrixCity block 3 + depth cache (Depth Pro, Metric3D v2, UniDepthV2, DAv2): residual stacks of all 185
  views for training; exact depth held back for evaluation.
- Injected faults (`sensorfault.faults.injectors`): same, masks withheld; size and magnitude sweep.
- Bunny / relief scenes (`sensorfault.sim.bunny`): unit tests, thin-structure regime.
- UseGeo, ETH3D: no labels needed, so per-scene test-time fitting is allowed; report both trained-elsewhere
  and fitted-on-scene. LiDAR for evaluation.
- Evaluation label rule: aligned error > 3 sigma in a connected region >= 2% of the view (H1). Evaluation
  masks stored apart from training data. Splits by block or scene (H2).

## Integration

1. Hard: Algorithm 1 unchanged, Holm replaced by "drop the highest view above the threshold".
2. Soft: depth loss weighted by `w_j(u) = 1 - zeta_j(u)` each round (soft EM). Net refitted between rounds,
   frozen during the surface fit.

## Plan

| Step | Question | Blocked on |
| --- | --- | --- |
| L0 | dump feature stacks per view; evaluation masks in a separate store | nothing |
| L1 | U1 on injections: recover withheld masks, match the gate at >= 2 sigma, calm at noise scale 1.3 without temperature | nothing |
| L2 | U1 on real depth-model errors vs gate and depth confidence (**go / no-go**) | depth cache on all 185 views (TITAN) |
| L3 | U2 distillation on untestable pixels (low-overlap breakdown) | L2 |
| L4 | hard vs soft inside the 2DGS fit | working GPU fit (PR 4 / gate-fit-floor) |
| L5 | UseGeo / ETH3D, trained elsewhere vs fitted on scene | downloads (M's yes) |
| L6 | ablations: U3, U4, U5, no cavity, RGB in the scale head | L4 |

Stress regimes where the statistical gate fails today: noise scale 1.3 without temperature (100% FA in
E1), common-mode three-frame repetition, bunny edge-of-arc and ear (thin-structure) views, low overlap.

Baselines: statistical gate (E1 and 2DGS paths), aligned RMS, depth confidence, robust depth loss,
RobustNeRF, SpotLessSplats, no gate, oracle gate (evaluation bound only).
