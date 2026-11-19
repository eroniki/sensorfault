# How each reference was checked (2026-11-14)

- ke2024marigold: arXiv HTML v2 read in full, plus README and Hugging Face model cards. Author list from memory of the CVPR paper: verify against the PDF.
- guizilini2025mvgd: arXiv HTML v1 read in full (CVPR camera-ready not read). Authors and pages from the arXiv/CVF listing.
- depthsync2025: arXiv HTML v2 read in full. **Author list not verified**; fill from the arXiv page before use. No code link was found.
- tse2026affinegs: arXiv HTML v1 read (appendices A-G, most of H); authors and year from the arXiv listing (submitted 5 Nov 2026).
Statements in `sections/related.tex` summarise what those readings showed; they are not claims about later versions.

# Bibliography check (2026-11-17)

- Checked against the publisher or arXiv page: depthsync2025 (authors confirmed; no venue on arXiv, so the "accepted by ICCV 2025" note was removed),
  tse2026affinegs (arXiv 2611.07709 says accepted to NeurIPS 2026; NeurIPS 2026 poster page lists the same four authors),
  garciamartin2025e2edepth (second author is Karim Abou Zeid, not Karim Knaebel; WACV pp. 753-762), guizilini2025mvgd (authors, pp. 764-776),
  xu2025pixelperfect (authors, NeurIPS 2025), landgraf2025critical (arXiv only), teunissen2024network (DOI 10.59490/tb.100, TU Delft OPEN 2024),
  levi2019evaluating (arXiv v3, no journal-ref), schoeps2017eth3d (new entry; CVPR 2017 pp. 3260-3269).
- arXiv-only entries promoted to their published venue from known proceedings (not re-fetched in this pass): Lotus (ICLR 2025), GeoWizard (ECCV 2024),
  DepthCrafter, RollingDepth (CVPR 2025), UniDepth, ReconFusion, SuGaR, DUSt3R (CVPR 2024), CAT3D (NeurIPS 2024), MVDream, SDXL (ICLR 2024), LoRA (ICLR 2022),
  FSGS, MASt3R (ECCV 2024), RobustNeRF (CVPR 2023), NeRF-W (CVPR 2021), MonoSDF (NeurIPS 2022), 2DGS (SIGGRAPH 2024 Conference Papers).
- Left as arXiv because the venue could not be confirmed here: bansal2023universal, he2023mpgd, ye2024gsplat, yu2024gof, chen2024pgsr, xiang2023measuring,
  levi2019evaluating, chung2023depthreg, depthsync2025.
