# Evidence log: part_gs.bib (checked 2026-11-13)

Sources: Crossref REST record for each DOI (https://api.crossref.org/works/<DOI>, mirrors the publisher/IEEE/ACM record) for venue, volume/issue, pages, authors; arXiv API/abs page (https://arxiv.org/abs/<id>) for arXiv title, authors, comments, abstract. DBLP, CVF open access and ACM DL were blocked (bot check / 403), not used. ACM/Crossref give no separate article number for the ACM items, so none is given.

| Key | Verified URLs | Notes |
|---|---|---|
| gs_3dgs | https://doi.org/10.1145/3592433 ; https://arxiv.org/abs/2308.04079 | ACM TOG 42(4), pp. 1-14 per Crossref. Real article number (139) NOT seen, omitted. Author "Leimkühler" from arXiv. |
| gs_2dgs | https://doi.org/10.1145/3641519.3657428 ; https://arxiv.org/abs/2403.17888 | SIGGRAPH '24 Conference Papers, pp. 1-11. |
| gs_mcmc | https://doi.org/10.52202/079017-2573 ; https://arxiv.org/abs/2404.09591 | NeurIPS 37, pp. 80965-80986. Crossref author is "Yang-Che Tseng"; arXiv lists "Jeff Tseng". Used the published name. |
| gs_splatam | https://doi.org/10.1109/CVPR52733.2024.02018 ; https://arxiv.org/abs/2312.02126 | CVPR 2024 pp. 21357-21366. |
| gs_spotless | https://doi.org/10.1145/3727143 ; https://arxiv.org/abs/2406.20055 | Published: ACM TOG 44(2), 2025 (arXiv page itself lists no venue). Crossref "David Fleet" vs arXiv "David J. Fleet". Title is "SpotLessSplats" (capital L); the arXiv API title string is "SpotlessSplats". |
| gs_robustsplat | https://doi.org/10.1109/ICCV51701.2025.02518 ; https://arxiv.org/abs/2506.02751 | ICCV 2025 pp. 27126-27136. |
| gs_tofsplatting | https://doi.org/10.1109/ICCV51701.2025.02632 ; https://arxiv.org/abs/2504.16545 | arXiv page lists no venue, but Crossref shows ICCV 2025 pp. 28344-28353. |
| gs_droidsplat | https://doi.org/10.1109/ICCVW69036.2025.00291 ; https://arxiv.org/abs/2411.17660 | ICCV 2025 Workshops (not main conference); arXiv page lists no venue. Crossref title lacks the colon. |
| gs_3dgscalib | https://doi.org/10.1109/IROS58592.2024.10801360 ; https://arxiv.org/abs/2403.11577 | IROS 2024 (oral), pp. 8315-8321. |
| gs_bayesiangsslam | https://doi.org/10.1109/LRA.2026.3741776 ; https://arxiv.org/abs/2609.24140 | arXiv page (submitted 21 Sep 2026) lists no venue. Crossref lists IEEE RA-L 11(12), Dec 2026, pp. 13693-13700. Flag: arXiv v1 is only ~2 months old and the issue date is in the future; consider whether to cite the preprint instead. |
| gs_yogo | https://arxiv.org/abs/2604.21400 | arXiv-only (no venue on page, no Crossref record found). v4 revised 15 Jul 2026; 15 pages, 5 figures. |
| gs_multisensor_implicit | https://doi.org/10.1109/ICCVW60793.2023.00488 ; https://arxiv.org/abs/2306.11048 | arXiv 2306.11048 is NOT a multi-sensor paper: it is UncLe-SLAM (Sandström, Ta, Van Gool, Oswald), ICCV 2023 Workshops pp. 4539-4550. Caller's description does not match. |
| gs_gsfusion | https://doi.org/10.1109/LRA.2024.3502065 ; https://arxiv.org/abs/2408.12677 | IEEE RA-L 9(12), Dec 2024, pp. 11865-11872. arXiv lists no venue. |
| gs_dnsplatter | https://doi.org/10.1109/WACV61041.2025.00241 ; https://arxiv.org/abs/2403.17822 | WACV 2025 pp. 2421-2431. |

## One-line descriptions (from abstracts, <=25 words)
- gs_3dgs: Represents scenes as anisotropic 3D Gaussians with interleaved density control and a visibility-aware rasterizer for real-time (>=30 fps, 1080p) novel-view synthesis.
- gs_2dgs: Collapses 3D Gaussians into oriented 2D planar disks with ray-splat intersection, depth-distortion and normal-consistency terms for geometrically accurate surface reconstruction.
- gs_mcmc: Treats Gaussians as MCMC samples, turning updates into SGLD and densification/pruning into state transitions, with relocalization instead of cloning.
- gs_splatam: Dense RGB-D SLAM from a single unposed camera using explicit 3D Gaussians, online tracking/mapping and a silhouette mask.
- gs_spotless: Uses pre-trained general-purpose features and robust optimization to ignore transient distractors in 3DGS reconstruction of casual captures.
- gs_robustsplat: Delays Gaussian growth until static structure is optimized and uses scale-cascaded mask bootstrapping to estimate transient masks, reducing transient artifacts in 3DGS.
- gs_tofsplatting: First 3DGS-based SLAM pipeline for very sparse ToF depth, with a multi-frame module merging sparse ToF, monocular colour and multi-view geometry into dense depth.
- gs_droidsplat: SLAM system pairing an end-to-end tracker (DROID-style) with a 3DGS renderer; reports state-of-the-art tracking/rendering and works on in-the-wild monocular data without known intrinsics.
- gs_3dgscalib: Targetless multimodal spatiotemporal sensor calibration using 3D Gaussian Splatting, faster than implicit-representation methods; evaluated on KITTI-360.
- gs_bayesiangsslam: 3DGS SLAM estimating colour/depth predictive uncertainty (sensor noise plus opacity-induced), used to weight tracking residuals, augment mapping and select keyframes.
- gs_yogo: Budget-aware controller for predictable 3DGS resource use, multi-sensor registration protocol, and the ultra-dense indoor Immersion v1.0 dataset.
- gs_multisensor_implicit (UncLe-SLAM): Learns online, self-supervised pixel-wise depth uncertainty to weight tracking and mapping losses in dense neural SLAM; lower ATE on 7-Scenes and TUM-RGBD.
- gs_gsfusion: Embeds 3D Gaussians in a volumetric (TSDF) mapping system, using an image quadtree to cut initialized splats, giving a compact Gaussian map and volumetric map online.
- gs_dnsplatter: Adds depth and normal priors (adaptive depth loss, smoothness, monocular normal networks) to Gaussian splatting for indoor scenes, enabling direct mesh extraction.

## NOT VERIFIED
None dropped. Caveats: 3DGS article number (139) not seen; ACM article numbers for gs_2dgs/gs_spotless not seen; gs_yogo has only arXiv evidence; gs_bayesiangsslam publication status rests on Crossref only.
