# Learned fault indicator (research track)

Status: scoping, 2026-11-18. Nothing built yet. Branch `research/learned-indicator`, kept apart from
`main` and from the RA-L letter, which keeps the statistical gate. Full scoping doc (editable):
https://claude.ai/artifact/EFe4tF4ns3xNsJbWdQaU8j

## Idea

Replace the statistical / information-theoretic decision of the view gate with a learned indicator
function. Keep the leave-block-out cavity (fusion, block subtraction, Newton step, support rules) as the
evidence; learn only the decision.

- **Output:** a per-pixel fault probability `zeta_j(u)` for the tested view `j`, pooled to a view score.
- **Inputs (feature stack):** cavity residual `r_j` (aligned, divided by `sigma(range, texture, incidence)`),
  whitened residual, rendered sd `sqrt(V_{-j})`, accumulated opacity, untestable mask, range and incidence,
  RGB of view `j`, depth-model confidence where it exists.
- **Replaces:** scene temperature gamma, heterogeneity omega, n_eff degrees of freedom, energy statistic,
  region scan + Bonferroni, Holm step-down.
- **Keeps a guarantee:** split-conformal threshold on clean development views (per view, or on the scene
  maximum for a scene-level rate).

## Formulation options

| Option | Learned | Note |
| --- | --- | --- |
| F1 (start here) | decision on cavity features | keeps self-exclusion; small net |
| F2 | same net, residual against fused surface (no leave-out) | ablation: shows masking without the cavity |
| F3 | per-pixel loss weights in the 2DGS fit, trained jointly | needs a mask budget; later |
| F4 | net that predicts the gate's p-value | speed only; not pursued |

## Supervision

- **S1 (main):** real depth-model errors on MatrixCity (Depth Pro, Metric3D v2, UniDepthV2, DAv2 from the
  depth cache) against exact depth. Label: aligned error > 3 sigma in a connected region >= 2% of the view
  (the H1 definition); ignore band near the threshold. Alignment leave-view-out.
- **S2:** injected faults (`sensorfault.faults.injectors`), exact masks; controlled sweep.
- **S3:** bunny / relief scenes (`sensorfault.sim.bunny`); unit tests, thin-structure regime.
- **S4:** UseGeo, ETH3D: test only.
- Split by block or scene, never by view (H2: neighbours repeat errors). Focal loss per pixel + view term.

## Integration

1. Hard: Algorithm 1 unchanged, Holm replaced by "drop the highest view above the conformal threshold".
2. Soft: depth loss weighted by `w_j(u) = 1 - zeta_j(u)` each round (soft EM). Net frozen during the fit.

## Plan

| Step | Question | Blocked on |
| --- | --- | --- |
| L0 | dump feature stack + labels per view (block 3 E1 cavities, bunny) | nothing |
| L1 | F1 on injections; match the gate on clean faults, beat it in stress regimes | nothing |
| L2 | F1 on natural depth-model errors vs gate and depth confidence (**go / no-go**) | depth cache on all 185 views (TITAN) |
| L3 | hard vs soft inside the 2DGS fit | working GPU fit (PR 4 / gate-fit-floor) |
| L4 | transfer to UseGeo / ETH3D and more MatrixCity blocks | downloads (M's yes) |
| L5 | ablations, F3 | L3 |

Stress regimes where the statistical gate fails today: noise scale 1.3 without temperature (100% FA in
E1), common-mode three-frame repetition, bunny edge-of-arc and ear (thin-structure) views, low overlap.

Baselines: statistical gate (E1 and 2DGS paths), aligned RMS, depth confidence, robust depth loss,
RobustNeRF, SpotLessSplats, no gate, oracle gate.
