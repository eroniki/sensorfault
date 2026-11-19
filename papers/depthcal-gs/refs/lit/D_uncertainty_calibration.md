# D. Uncertainty and calibration (all verified via arXiv abs page unless noted)

1. kendall2017uncertainties - aleatoric vs epistemic uncertainty for vision (depth, segmentation). Ours: justifies reading sample spread as (mostly epistemic/model) covariance. https://arxiv.org/abs/1703.04977. Venue NIPS 2017 from arXiv comments.
2. lakshminarayanan2017ensembles - deep ensembles for predictive uncertainty. Ours: diffusion samples act as an ensemble. https://arxiv.org/abs/1612.01474. NIPS 2017 from comments.
3. kuleshov2018calibrated - quantile-based calibration/recalibration of regression. Ours: PIT/coverage metrics. https://arxiv.org/abs/1807.00263. ICML 2018 from comments.
4. levi2019evaluating - stddev calibration for regression (ENCE-style, rescaling). Ours: regression ECE / variance recalibration. https://arxiv.org/abs/1905.11659. arXiv-only on page (v3 2020); published venue not confirmed here.
5. angelopoulos2021gentle - conformal prediction tutorial. Ours: possible distribution-free alternative to Gaussian covariance. https://arxiv.org/abs/2107.07511. arXiv-only on page (v6 2022); a Foundations and Trends version exists but was not verified.
6. gneiting2007scoring - strictly proper scoring rules incl. CRPS. Ours: CRPS and NLL justification. Verified via Crossref API https://api.crossref.org/works/10.1198/016214506000001437 (publisher page 403, JSTOR failed). JASA 102(477):359-378.
7. ilg2018uncertainty - sparsification plots / oracle for uncertainty evaluation. Ours: sparsification error/AUSE. https://arxiv.org/abs/1802.07095. ECCV 2018 from comments.
8. poggi2020uncertainty - AUSE/AURG evaluation of depth uncertainty (self-supervised mono depth). Ours: sparsification protocol for depth. https://arxiv.org/abs/2005.06209. CVPR 2020 from comments.
9. jazbec2025generative - uncertainty for diffusion models via Bayesian posterior over weights, pixel-wise generative uncertainty. Ours: closest diffusion-uncertainty method. https://arxiv.org/abs/2502.20946. arXiv-only on page.
10. ke2024marigold - diffusion depth with ensembling of samples. Ours: sample-ensemble baseline for diffusion depth; no calibration evaluation claimed. https://arxiv.org/abs/2312.02145. CVPR 2024 from comments.
11. gui2025depthfm - flow-matching depth; ensemble std as uncertainty (per search-result summary; not read in full). https://arxiv.org/abs/2403.13788. AAAI 2025 from comments. Caveat: the "uncalibrated" claim should be re-checked in the PDF before citing.
12. xiang2023measuring - critiques sparsification/AUSE for depth uncertainty, proposes Spearman correlation. Ours: caveat on sparsification metric. https://arxiv.org/abs/2307.09929. arXiv-only on page (claim from search summary; check).
13. landgraf2025critical - survey of UQ and foundation models in mono depth. Ours: related-work context. https://arxiv.org/abs/2501.08188. arXiv-only.

## UNVERIFIED
- None cited. Not found: any prior work explicitly evaluating calibration (PIT/CRPS) of diffusion depth ensembles; search did not surface one.
