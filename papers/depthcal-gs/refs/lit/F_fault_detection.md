# F: fault detection / leave-one-out / robust multi-view (12 verified)

Format per item: key; what; relation; URL; caveats.

1. baarda1968testing: data snooping, reliability, marginally detectable error. Ours: ancestor of whitened-residual test and MDB analysis. URL https://ncgeo.nl/publicatie/a-testing-procedure-for-use-in-geodetic-networks/ (DOI 10.54419/t8w4sg). Caveat: sources disagree on series numbering (ncgeo says "Publications on Geodesy 9"; others Vol 2 No 5); volume omitted from bib, check before camera-ready.
2. teunissen1998mdb: closed-form minimal detectable biases. Ours: MDB analysis of fault classes. URL https://api.crossref.org/works/10.1007/s001900050163 (DOI 10.1007/s001900050163). Caveat: GPS-specific models.
3. teunissen2024network: book, testing/DIA, reliability. Ours: DIA framework (detection-identification-adaptation). URL https://books.open.tudelft.nl/home/catalog/book/181. Caveat: 2024 open-access edition; the original print edition (Delft University Press, 2000) was not verified. Check it contains DIA chapter.
4. blanch2012araim: ARAIM user algorithm, solution separation, fault detection and exclusion. Ours: leave-one-out subset solutions vs. full solution is the same test logic. URL https://web.stanford.edu/group/scpnt/gpslab/pubs/papers/Blanch_IONGNSS_2012_B5_nr7_rev2.pdf (title/authors read from PDF; venue/year from search result only). Caveat: this is the 2012 ION GNSS paper, not the later TAES paper.
5. fischler1981ransac: RANSAC. Ours: classical outlier rejection in geometric fitting. URL https://api.crossref.org/works/10.1145/358669.358692. Caveat: none.
6. schonberger2016sfm: COLMAP SfM with geometric verification/outlier filtering. Ours: pipeline whose poses/focals are tested. URL https://api.crossref.org/works?query.bibliographic=Structure-from-Motion+Revisited. Caveat: none.
7. sabour2023robustnerf: robust losses ignore distractors. Ours: per-pixel robust down-weighting vs. our view-level leave-out test. URL https://arxiv.org/abs/2302.00833. Caveat: published at CVPR 2023 per memory; arXiv page names no venue, so typed @misc.
8. sabour2024spotless: distractor masks for 3DGS. Ours: related splatting outlier handling. URL https://arxiv.org/abs/2406.20055. Caveat: arXiv page title casing "SpotlessSplats"; published venue not verified.
9. martinbrualla2020nerfw: transient/appearance handling. Ours: transient modelling baseline. URL https://arxiv.org/abs/2008.02268. Caveat: CVPR 2021 venue not shown on arXiv page; typed @misc.
10. vovk2022algorithmic: conformal prediction book. Ours: empirical, distribution-free thresholds. URL https://api.crossref.org/works?query.bibliographic=Algorithmic+Learning+in+a+Random+World (DOI 10.1007/978-3-031-06649-8). Caveat: this is the 2022 second edition; the 2005 first edition authors not confirmed on page.
11. angelopoulos2021gentle: conformal tutorial. Ours: threshold calibration reference. URL https://arxiv.org/abs/2107.07511. Caveat: arXiv v6 (Dec 2022); no journal named.
12. barber2021jackknifeplus: jackknife+ (leave-one-out predictive inference). Ours: leave-one-out with calibrated coverage. URL https://api.crossref.org/works?query.bibliographic=Predictive+inference+with+the+jackknife%2B (also arXiv:1905.02928). Caveat: none.

## UNVERIFIED (not in .bib)
- Brown, RAIM / solution separation (Global Positioning System: Theory and Applications vol. II, 1996): no authoritative page found.
- Blanch et al. "Baseline Advanced RAIM user algorithm and possible improvements" (IEEE TAES 2015): search could not find it under this title; Crossref query rate-limited.
- Parkinson and Axelrad 1988 (pseudorange residual RAIM, Navigation): not checked.
- Cross-validation / leave-one-view-out NVS evaluation, SLAM/BA fault exclusion papers: not searched for lack of budget.
