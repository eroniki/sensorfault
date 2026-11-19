# G: AffineGS and close rivals (all arXiv abstract pages fetched 2026-11-16)

Format: key | does | differs from ours | URL | caveats

1. tse2026affinegs | Spatially varying, edge-aware affine fields map monocular disparity to consistent metric depth for 3DGS, plus a normal-aware regularizer coupling rectified disparity with 3DGS depth/normals. | It rectifies (corrects) the depth-network output deterministically, with no calibrated spread. We sample depths with calibrated uncertainty and leave the network output uncorrected, tested leave-view-out against 2DGS. | https://arxiv.org/abs/2611.07709 | Verified authors Tse, Peng, Chen, Yao; submitted 5 Nov 2026. Venue is only the arXiv comment "Accpeted to NeurIPS 2026" (sic); no proceedings yet, so cite as arXiv with note. Closest rival.
2. turkulainen2025dnsplatter | Depth and normal priors in Gaussian splatting for meshing; evaluated on MuSHRoom. | Global-ish prior alignment/regularization, no per-pixel calibrated spread. | https://arxiv.org/abs/2403.17822 | Venue: arXiv comment "To be published in WACV 2025"; page revised v3 Nov 2024. I did not check the WACV proceedings page.
3. yu2022monosdf | Monocular depth and normal cues as supervision for neural implicit surfaces (scale/shift-invariant depth loss). | Per-image scale-shift alignment into the loss, not a sampled depth with calibrated spread. | https://arxiv.org/abs/2206.00665 | arXiv page lists no venue, so entered as arXiv-only. (It was NeurIPS 2022 from memory; unverified, add after a check.)
4. wang2023dust3r | Feed-forward pairwise pointmap regression with global alignment for uncalibrated multi-view geometry; outputs confidence maps. | Learned confidence is not calibrated against held-out views, and depth is not sampled. | https://arxiv.org/abs/2312.14132 | arXiv comment gives no venue; entered as arXiv-only (CVPR 2024 per memory, unverified).
5. leroy2024mast3r | Adds dense matching features to DUSt3R for better correspondences and metric-scale pointmaps. | As for DUSt3R. | https://arxiv.org/abs/2406.09756 | No venue on arXiv page; arXiv-only (ECCV 2024 per memory, unverified).
6. wang2025vggt | Feed-forward transformer predicting cameras, depth, pointmaps and tracks from many views, with confidence. | Deterministic point estimate with heuristic confidence; no calibration test. | https://arxiv.org/abs/2503.11651 | Venue CVPR 2025 per arXiv comment.
7. yang2024depthanythingv2 | Strong monocular relative-depth foundation model, the typical prior that AffineGS/DN-Splatter align. | Deterministic relative depth; we output calibrated samples. | https://arxiv.org/abs/2406.09414 | Venue NeurIPS 2024 per arXiv comment.
8. li2024megasam | Video SfM with monocular depth priors, uncertainty-aware bundle adjustment for casual dynamic videos. | Uses depth priors and uncertainty inside BA for poses/depth; no calibrated depth distribution tested against a held-out render. | https://arxiv.org/abs/2412.04463 | No venue on arXiv page (CVPR 2025 per memory, unverified).
9. xu2025depthsplat | Feed-forward Gaussian splatting coupled with depth, with multi-view and monocular features. | Feed-forward deterministic depth, no spread calibration. | https://arxiv.org/abs/2410.13862 | CVPR 2025 per arXiv comment.
10. yang2024depthanyvideo | Video diffusion-based consistent depth with synthetic data (generative, temporal consistency). | Generative but targets temporal consistency, not multi-view-consistent calibrated sampling. | https://arxiv.org/abs/2410.10815 | No venue on arXiv page; arXiv-only. Not an in-reconstruction fusion method; relevance moderate.
11. ren2023mushroom | MuSHRoom benchmark for joint reconstruction and novel view synthesis. | Benchmark we may use. | https://arxiv.org/abs/2311.02778 | No venue on arXiv page; arXiv-only (check publication before submission).
12. yeshwanth2023scannetpp | ScanNet++ high-fidelity indoor dataset. | Benchmark. | https://arxiv.org/abs/2308.11417 | ICCV 2023 per arXiv comment.

## UNVERIFIED (not in .bib)
- Metric3D / Metric3Dv2 (metric depth in reconstruction): not fetched.
- ETH3D benchmark (Schops et al., CVPR 2017): not fetched; no arXiv version expected.
- Marigold (diffusion depth), NeuralRecon-W: not fetched.
- Venues from memory noted above (MonoSDF, DUSt3R, MASt3R, MegaSaM) are unverified.
