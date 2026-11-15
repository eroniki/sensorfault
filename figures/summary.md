# Summary of results on disk

All numbers come from results already produced (no new runs for this table). Figures are in this folder; each carries its data provenance in its footer. Everything marked **toy** is synthetic, one seed, with power standard errors of about 0.05; nothing here involves a trained depth model.

| # | Experiment | Data and provenance | Result | Status / caveat | Figure |
|---|---|---|---|---|---|
| 1 | Leave-one-view-out 2DGS, held-out view vs laser depth | REAL ETH3D pipes, 3 held-out views, 1 run each; surface from the other 13 views' sparse ground-truth depth (`docs/pipes-2dgs-fit.json`) | view 2: coverage 98.9->99.8%, median error 0.10->0.08%, p90 0.44->0.30%; view 6: coverage 87.7->94.7%, median error 0.43->0.19%, p90 0.82->0.63%; view 10: coverage 97.7->99.6%, median error 0.14->0.09%, p90 0.47->0.43% | 'before' = initialised surfels, **not optimised**; 'after' = 2000 its (RGB+depth L1, no densification). Uses ground-truth depth as a stand-in for model predictions: tests the machinery, not a depth model | fig1 |
| 2 | Leave-out guards, synthetic room | toy point surface (5 views, CPU) and gsplat 2DGS (5 views, GPU), every view held out once (`figures/data/guards.json`) | toy surface: poisoned 5/5, invariant 5/5, sensitive 5/5; 2DGS: poisoned 5/5, bit-identical 5/5, sensitive 5/5 | pass | fig2 |
| 3 | Mutation check: a fit that ignores the exclusion | deliberately leaky fit, views 0/2/4 | caught in 3/3 views (poisoned-view check fails 3/3, perturbation check fails 3/3) | the guard can detect a leak | fig2 |
| 4 | Leave-out guards, real optimised 2DGS | REAL ETH3D pipes, 3 held-out views, 300-iteration fits | poisoned view untouched 3/3; median absolute difference with j perturbed 2e-04, 2e-05, 2e-04 m vs control 2e-04, 2e-05, 2e-04 m | perturbation effect equals the run-to-run noise floor (CUDA non-determinism); floaters give 2-3 m max differences in the control as well | fig2 |
| 5 | Toy statistic: minimum detectable fault (80% power) | **toy**, 16x16, M=10, 1 seed, 80 trials/cell (`docs/toy-results.json`) | square: known 0.89; oracle 0.88; kernel 2.24; kernel_nr 2.21; shrunk 1.46. warp: known 2.15; oracle 2.07; kernel >4; kernel_nr >4; shrunk 1.21. edge: known 0.89; oracle 0.84; kernel 2.38; kernel_nr 2.42; shrunk 1.70 (sd units; '>' = not reached by 4) | fitted (s,t) costs ~0 with the true covariance (0.88 vs 0.89); the estimated covariance dominates the loss | fig3, fig4 |
| 6 | Toy statistic: false alarms at the chi2(n-2) threshold | **toy**, 200 clean trials | known 0.07; oracle 0.06; kernel 0.27; kernel_nr 0.27; shrunk 0.14 (nominal 0.05; mean T/(n-2): known 1.01, oracle 0.96, kernel 1.31, kernel_nr 1.31, shrunk 0.94) | chi2 is unusable for estimated covariances; the empirical null controls the level by construction (assumes exchangeable clean views, untested) | fig4 |
| 7 | Toy statistic: global faults, kappa=8 | **toy** | constant shift power known 1.00, oracle 0.05, kernel 0.04, kernel_nr 0.04, shrunk 0.03; scale error power known 1.00, oracle 1.00, kernel 0.91, kernel_nr 0.91, shrunk 1.00 | only a constant offset is invisible; a scale error is flagged because the fitted scale also rescales the sample covariance | fig5 |
| 8 | Toy statistic: power vs fault area (band) | **toy**; kappa 2.5 and 4; areas to 0.95 (`docs/toy-followups.md`) | kappa 2.5, oracle: 0.05:1.00, 0.15:1.00, 0.3:1.00, 0.5:1.00, 0.7:1.00, 0.85:1.00, 0.95:0.03; estimated kernel: 0.05:0.23, 0.15:0.67, 0.3:0.68, 0.5:0.57, 0.7:0.68, 0.85:0.57, 0.95:0.05 | collapses only at 0.95 (near-constant offset); the statistic detects inconsistency but cannot say which side is wrong | fig5 |
| 9 | Toy statistic: miscalibrated posterior | **toy**, spread = 0.6x / 1.0x / 1.6x the truth's (`docs/toy-followups.md`) | MDF square/warp: x0.6: oracle 0.87/2.07, kernel 1.58/3.65, kernel_nr 1.47/3.63; x1.0: oracle 0.89/2.06, kernel 2.25/>4.0, kernel_nr 2.22/>4.0; x1.6: oracle 0.91/2.11, kernel 2.31/>4.0, kernel_nr 2.27/>4.0 | scale error of the spread is absorbed by the empirical threshold; a wrong covariance *shape* (estimated kernel) costs power | fig6 |

## Not in this table (not run yet)

- Stage 0 (SDXL LoRA sizing) on the TITAN RTX: not run; no SDXL memory or speed number exists.
- Any trained or fine-tuned diffusion model, any calibration against ground truth for a model, any fault-injection result on real model samples.
- Real-data 2DGS fed with predicted (not ground-truth) depth.

## Files

| Figure | File (png and pdf) | Data |
|---|---|---|
| fig1 | `fig1_pipes_2dgs` | real ETH3D pipes, 2DGS before/after |
| fig2 | `fig2_leave_out_guards` | synthetic guards + real pipes guards |
| fig3 | `fig3_toy_power_vs_magnitude` | synthetic toy |
| fig4 | `fig4_toy_mdf_by_variant` | synthetic toy |
| fig5 | `fig5_toy_area_and_global` | synthetic toy |
| fig6 | `fig6_toy_miscalibration` | synthetic toy |

Regenerate: `python scripts/run_guard_checks.py` (seconds; GPU optional), then `python scripts/make_figures.py`.
