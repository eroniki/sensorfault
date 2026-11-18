# Positioning: Related Work, Novelty, Reviewer Objections, Baselines

This note positions the method of `docs/learned-indicator.md` (per-source information on Gaussian surfels, calibrated without labels, used to test a monocular depth prior against pixels) against prior work.

**Verification status.** Each work carries one of three marks. *Read*: the arXiv abstract page, project page or repository page was fetched and the stated content checked (abstract-level only unless said otherwise). *Listed*: the title and identifier were seen in a search listing but the page was not read. *Recalled*: named from memory; venue and details not checked. Only *read* entries support a statement of the form "the paper does X". In the differentiation table, "not stated" means the fetched abstract or page does not mention the property; that is weaker than "the paper does not do it", and the cell is never turned into a "no" without reading the full text.

**Cost assumptions.** One GPU with 24 GB; an aerial strip is 300 to 600 posed frames of 1 to 2 megapixels; times marked *estimate* are unmeasured guesses.

## 1. Related-work map

### 1.1 Group (i): uncertainty and Fisher information in radiance fields and splatting

| Work | Status | What it does | Where this method differs |
| --- | --- | --- | --- |
| Jiang, Lei, Daniilidis, FisherRF, ECCV 2024, arXiv 2311.17874 | read (abstract, project page) | Observed Fisher information on radiance-field parameters, used for next-best-view selection and pixel-wise rendering uncertainty; runs on a 3DGS backend. | One information score for the image loss on all parameters, no depth source, no second source to compare with. Here: information on three placement coordinates of a surface entity, per source, calibrated against realised scatter, used to test a prior. |
| Hanson et al., PUP 3D-GS, arXiv 2406.10219 | read (abstract) | Second-order sensitivity of reconstruction error to each Gaussian's spatial parameters, for pruning. | A per-Gaussian score for image loss, used to delete primitives. Here: per entity, per source, about placement of a surface. |
| Xie et al., GauSS-MI, arXiv 2504.21067 | read (abstract) | Per-Gaussian visual uncertainty and Shannon mutual information for next-best-view planning. | Visual (appearance) uncertainty for active view choice; no geometry-placement information, no test of a prior. Occupies the words "information-theoretic" with splatting. |
| Li et al., OUGS, arXiv 2511.09397 | read (abstract) | Uncertainty from Gaussian parameters propagated through the rendering Jacobian, with segmentation masks for object-aware view selection. | Closest on "Jacobian plus masks": uses the renderer Jacobian and semantic masks. No per-source records, no depth prior, no calibration claim. |
| Goli et al., Bayes' Rays, CVPR 2024, arXiv 2309.03185 | read (page); NeRF only, with a third-party 3DGS port seen in a search listing | Post-hoc Laplace uncertainty on a spatial perturbation field of a trained NeRF. | A Laplace approximation of one image-loss posterior. Here: two sources on shared primitives; the contribution is their comparison and honesty correction. |
| Van de Maele et al., Variational Bayes Gaussian Splatting, arXiv 2410.03592 | read (abstract) | Closed-form variational updates of Gaussian-splat parameters for continual learning. | A posterior used for streaming; not about sources, geometry honesty or priors. |
| Zhu et al., VBGS-SLAM, arXiv 2604.02696 | read (abstract) | Joint probabilistic model of poses and splat parameters with uncertainty. | Single posterior. Which parameters carry uncertainty was not read. |
| Klasson et al., Sources of Uncertainty in 3D Scene Reconstruction, arXiv 2409.06407 | read (abstract) | Taxonomy of uncertainty sources in NeRF and 3DGS, with learned and ensemble estimators. | Per-pixel or per-field uncertainty by cause; not per-source information on placement. |

Also relevant: VarSplat (arXiv 2603.09673, learned appearance variance per splat) and BayesSDF (arXiv 2507.06269, Laplace over SDF weights).

### 1.2 Group (ii): monocular depth in splatting and failure handling

| Work | Status | What it does | Where this method differs |
| --- | --- | --- | --- |
| Xiao et al., In Depth We Trust, arXiv 2604.05715 | read (HTML) | Scale and shift fitted by least squares to sparse SfM depth; a gradient-matching loss; a virtual-stereo mask flags pixels where rendered depth is inconsistent between a training camera and a pseudo-camera offset along x, and the absolute depth loss is applied only outside them (threshold user-defined). Tested on ScanNet++, Mip-NeRF 360, Tanks and Temples; no aerial data in the text read. | The gate tests rendered geometry against itself, not the prior against pixels; the threshold is hand-set and no null is given; scale needs SfM points, which this plan does not have. Closest competitor for the "fault side"; must be a baseline. |
| Chu et al., Reliability-Aware Monocular Depth Supervision, arXiv 2607.02554 | read (abstract) | Depth Anything V2 scaled with LiDAR and COLMAP anchors; supervision only where photometric masks from an RGB-only baseline indicate reliable regions; gains vary by setting. | A photometric mask is a heuristic test of the prior against pixels. Here: stiffness-weighted, with a null and an attribution. |
| Zhang et al., CDGS, arXiv 2502.14684 | read (abstract) | Confidence maps from several cues, combined with sparse SfM depth, set the depth weight per pixel. | Heuristic weight; cues not read; no information model. |
| Turkulainen et al., DN-Splatter, arXiv 2403.17822 | read (abstract, repository) | Depth and normal priors with an adaptive loss driven by colour gradients; the repository aligns monocular depth to COLMAP SfM points (or skips alignment). Indoor. | Colour-gradient weighting is a fixed heuristic form of "weight by texture". Here the weight is a derived information. Apache-2.0 code. |
| Chung et al., Depth-Regularized 3DGS for Few-Shot, arXiv 2311.13398 | read (abstract) | Monocular depth aligned in scale and offset to sparse COLMAP points, as a regulariser. | The standard scale-alignment recipe; depends on SfM. |
| DNGaussian, arXiv 2403.06912 | read (abstract) | Hard and soft depth regularisation with global-local depth normalisation for sparse views. | Normalisation instead of alignment; no reliability model. |
| Song et al., EnerGS, arXiv 2604.26238 | read (abstract) | Partial LiDAR geometry as a continuous energy field guiding large outdoor splatting. | A soft prior on a patchy source; no test of it. |
| Song et al., AerialMetric, arXiv 2606.29716 | read (abstract) | Benchmark showing monocular metric depth degrades from aerial viewpoints; fine-tuned model. | Supports the premise that faults exist on aerial data. Not a splatting method. |

The representation itself, 2D Gaussian Splatting (arXiv 2403.17888, read abstract), supplies the ray-splat intersection depth whose Jacobian the plan uses; the factorisation of Section 3.2 is a specialisation of its renderer, not a new renderer.

### 1.3 Group (iii): pose-free, correspondence-free and feed-forward reconstruction

| Work | Status | What it does | In or out of "correspondence-free" in this plan's sense |
| --- | --- | --- | --- |
| Fu et al., COLMAP-Free 3DGS, CVPR 2024, arXiv 2312.07504 | read (abstract) | Sequential video, Gaussians added frame by frame while estimating poses by photometric optimisation. | In (no SfM); estimates poses, which this plan takes as input. Its use of monocular depth is from memory, not in the abstract. |
| InstantSplat, arXiv 2403.20309 | read (abstract) | Dense initial points from a pretrained geometric foundation model, then photometric refinement of splats and poses with Gaussian bundle adjustment. | Hybrid: the initialisation is learned, matching-like; the refinement is photometric. Same capture-range argument as here. |
| DUSt3R, arXiv 2312.14132 | read (abstract) | Regresses pointmaps from image pairs without calibration or poses; global alignment merges pairs; correspondences and poses are derivable outputs. | Out: a learned matcher/triangulator. It replaces correspondences by a network, not by rendering. |
| Leroy et al., MASt3R, arXiv 2406.09756 | read (abstract) | Adds dense local features and reciprocal matching to DUSt3R. | Out: matching is its purpose. |
| VGGT, CVPR 2025, arXiv 2503.11651 | read (abstract, repository) | One feed-forward pass predicts cameras, depth, point maps and tracks for one to hundreds of images. | Out: it predicts tracks. It is also a depth source with a different error mode (multi-view, not monocular); see objection 5. |
| MVSplat, ECCV 2024, arXiv 2403.14627 | read (abstract) | Feed-forward Gaussians from a plane-sweep cost volume on sparse views. | Out: cost volume is cross-view matching; trained on RealEstate10K and ACID, small baselines. |
| Correspondence-Guided SfM-Free 3DGS, arXiv 2408.08723; ICP-3DGS, arXiv 2506.21629 | listed | Pose-free variants that reintroduce correspondences or registration. | Evidence, if read, that pure photometric alignment lacks capture range. Not to be cited as such until read. |

The plan's sense of the term is narrow: this method estimates, stores and trusts no discrete association between pixels. The feed-forward family does not meet it, because correspondence is an output or an internal operation. It meets it only in the weaker sense that no hand-built matcher runs.

### 1.4 Group (iv): large-area aerial splatting

| Work | Status | What it does | Where this method differs |
| --- | --- | --- | --- |
| VastGaussian (Lin et al.), arXiv 2402.17427 | read (abstract) | Progressive partition into cells with an airspace-aware visibility criterion, parallel optimisation, merge, decoupled appearance. | Rendering quality from SfM inputs; nothing stored about what is known. |
| CityGaussian, arXiv 2404.01133 | read (abstract, repository) | Global-prior-guided partitioned training, merge, level-of-detail compression. Repository lists MatrixCity Aerial among its datasets; CC BY-NC-SA. | Same. A baseline on the MatrixCity block already used. |
| Hierarchical 3D Gaussians, arXiv 2406.12080 | read (abstract, repository) | Chunked training merged into a hierarchy with level-of-detail; repository offers optional monocular-depth regularisation (Depth Anything V2) and recommends it for textureless roads. | Large-scale splatting with a monocular prior exists, as a regulariser without a test. |
| Su et al., HUG, ICCV 2025, arXiv 2504.16606 | read (abstract) | Visibility-based partitioning and hierarchical weighted training for aerial scenes. | Quality and cost; no uncertainty. |
| Windisch et al., A LoD of Gaussians, arXiv 2507.01110 | read (abstract) | Unpartitioned out-of-core training and rendering with a level-of-detail structure; 60M-Gaussian city model on a 24 GB card; evaluated on MatrixCity and Campus. | Scale by streaming; no uncertainty. Shows the large-area bar on MatrixCity. |
| Yao et al., ARSGaussian, arXiv 2412.18380 | listed (abstract read earlier) | LiDAR constraints for aerial remote-sensing splatting. | LiDAR is an input source, not a tested prior. |
| Visibility-Driven Large-Scale 3DGS for Aerial Scene Reconstruction (VDGS), arXiv 2609.23049 | listed | Aerial large-scale splatting. | Not read. |

### 1.5 Group (v): classical foundations

| Work | Status | What the method inherits | Where this method differs |
| --- | --- | --- | --- |
| Information-form filters and smoothers in SLAM (additive fusion, Schur-complement marginalisation) | recalled | Sections 3.4 and 4.6 are their algebra. | Applied to surface-entity placement on splat primitives, per source. |
| Julier and Uhlmann, covariance intersection, American Control Conference 1997 | listed (search listing and encyclopaedic page) | Fusion of estimates with unknown correlation without overconfidence. | The plan's saturated prior information $(\Sigma_\mu + \Sigma_\alpha/n)^{-1}$ is a parametric alternative with an estimated repeated share; covariance intersection is the model-free comparator. |
| Blanch et al., solution separation (ION GNSS 2012; IEEE TAES 2015) | read in earlier notes (titles and venues) | Compare estimates from subsets of evidence to detect a faulty one. | The between-source statistic is of this family. |
| Hausman, Specification Tests in Econometrics, Econometrica 1978 | listed | Difference of two estimators, one efficient and one robust, scaled by the variance difference. | Tension $T_m$ is a Hausman-type statistic. |
| Baarda, data snooping, 1968 | recalled | Outlier tests in geodetic adjustment; internal and external reliability. | The "harm versus detectability" analysis of Section 4.8 is reliability theory. |
| Gneiting and Raftery, Strictly Proper Scoring Rules, JASA 2007 | listed | Proper scores reward honest forecasts. | The surface log-score of Section 5.1. |
| Engel, Koltun, Cremers, Direct Sparse Odometry, TPAMI 2018, arXiv 1607.02565 | listed | Photometric alignment without features, with a small basin and coarse-to-fine remedy. | The capture-range limit of Section 4.7 is theirs. |

## 2. Novelty statement and differentiation

**Novelty statement (115 words).** We store, on each planar Gaussian surfel entity, a separate Fisher-information record per evidence source (monocular depth; rendered pixels) on the same three placement coordinates, which is possible because the image Jacobian of a planar surfel factors through its ray-plane depth Jacobian. The records are calibrated on the scene without labels, by split-frame placebo comparisons and displacements injected through the renderer, and prior information saturates at the share of monocular error that all frames repeat. Their difference, with a null distribution, tests a depth prior against the pixels. Prior splatting work keeps one uncertainty per primitive for the image loss, or masks depth by rendered-depth consistency; we found none that separates sources or calibrates them.

**Differentiation table.** Columns: A, stores uncertainty on primitives; B, per-source information; C, uses the image Jacobian for geometry uncertainty; D, handles correlated (repeated) prior error; E, calibrated on the scene without labels; F, tests the prior against pixels; G, large area.

| Work | A | B | C | D | E | F | G |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FisherRF | yes (information on field parameters) | not stated | not verified | not stated | not stated | not stated | not stated |
| PUP 3D-GS | yes (per-Gaussian sensitivity) | not stated | not verified | not stated | not stated | not stated | not stated |
| GauSS-MI | yes (per-Gaussian visual uncertainty) | not stated | not verified | not stated | not stated | not stated | not stated |
| OUGS | yes | not stated | yes (covariance through the rendering Jacobian) | not stated | not stated | not stated | not stated |
| Bayes' Rays | no (field-level perturbation, NeRF) | not stated | not verified | not stated | not stated | not stated | not stated |
| VBGS-SLAM | yes (scene parameters) | not stated | not verified | not stated | not stated | not stated | not stated |
| In Depth We Trust | not stated | not stated | not stated | not stated | not stated | partial: rendered-depth inconsistency, not pixels | not stated; no aerial data in the text read |
| CDGS | not stated | not stated | not stated | not stated | not stated | not verified (cues not read) | not stated |
| **This method (planned)** | yes | yes | yes | yes | yes | yes | to be shown |

The last row is a plan, not a result. Emptiness of columns B, D and E reflects abstract-level reading; recheck in full text before claiming it.

## 3. The ten most likely reviewer objections

Venue tags: C = CVPR, R = IROS/RA-L.

**1. "This is a Laplace approximation, or FisherRF, with two terms." (C, R)** *Strongest form:* per-source Gauss-Newton information is the Fisher information that FisherRF, PUP 3D-GS and Bayes' Rays compute; summing a depth term and an image term is a two-term Hessian. *Honest answer:* the information is classical and the plan says so. What is not in the read abstracts is (a) a factorisation that puts both on one three-coordinate placement per entity, (b) separate storage and a between-source statistic with a null, (c) calibration and saturation of the prior. *Must exist:* a head-to-head on the surface log-score where FisherRF-style per-Gaussian Fisher and a summed two-term Laplace on the same primitives are scored as alternative information fields (the plan's Section 5.3 names this); the result must show that the per-source split changes calibration or detection, not only presentation.

**2. "Photometric Hessians are known to be badly calibrated." (C, R)** *Strongest form:* rendering residuals are model misfit, spatially correlated; Gauss-Newton curvature overstates information by orders of magnitude. *Answer:* agreed; Section 4.2 and 4.5 correct it, but the correction may not suffice. *Must exist:* Stage B reported as a reliability diagram of stated against realised information per texture stratum, before and after the dispersion $\hat\varphi$, with the ratio of nominal to realised information tabulated per scene (target: the ratio after correction within a stated factor of one; the target is a prediction, not a result). If the corrected information is still miscalibrated, the stop rule S4 applies.

**3. "The chi-square null will not hold for rendering residuals." (C, R)** *Strongest form:* residuals are heavy-tailed and correlated; $T_m \sim \chi^2_3$ fails, so thresholds are meaningless. *Answer:* the null is not assumed; it is checked. *Must exist:* placebo tensions (depth against depth over split frame halves; image against image) with a quantile-quantile plot against $\chi^2_3$ on healthy entities; the false-alarm rate at the nominal 5% level per stratum; a fallback empirical null (the placebo distribution) and the detection result reported under both nulls.

**4. "Just down-weight depth late in training." (C, R)** *Strongest form:* a decaying depth weight, or masking where rendered and monocular depth disagree (In Depth We Trust, CDGS, DN-Splatter), recovers any benefit. *Answer:* it may on textured surface; the claim is for where it cannot: weak texture, repeated faults, and a calibrated statement of what is unknown. *Must exist:* the maps of Section 4.10 including decaying weight, In Depth We Trust masking and CDGS-style confidence, compared on accuracy, completeness and log-score, stratified by texture energy, with injected repeated faults. Stop rule S5 states the kill condition.

**5. "Feed-forward multi-view models make monocular priors obsolete." (C, R)** *Strongest form:* VGGT or MASt3R on the strip gives multi-view depth and poses in a pass; monocular depth is an obsolete prior. *Answer:* feed-forward models may be better depth sources; the plan's machinery applies to any depth source with a repeated-error structure, and they too err identically across views. They are also trained on mostly ground-level or indoor data and limited in frames per pass (VGGT states no cap, only that the frame budget grew 2 to 3 times in an update). *Must exist:* VGGT depth as an additional prior in the same pipeline (fault test applied to it), and as a baseline on the same strips; the repetition coefficient $\varrho$ reported for each depth source, so that "the prior is obsolete" becomes "which source has the lower $\varrho$ on aerial strips".

**6. "No real aerial data with ground truth." (C, R)** *Strongest form:* MatrixCity is synthetic, ETH3D is ground-based; the claim is about aerial mapping. *Answer:* UseGeo (UAV with LiDAR) is licensed, not yet downloaded; the proprietary flight is at most a qualitative demo and may not be publishable. *Must exist:* at least one UseGeo evaluation of calibration and accuracy against the LiDAR surface; if it cannot be done, the claims must be restricted to "synthetic aerial and ground-based real" in title and abstract.

**7. "SAM entities are arbitrary." (C)** *Strongest form:* masks are not repeatable across views or scales; entities have no physical meaning. *Answer:* this is a stated risk (stop rule S8). *Must exist:* entities from masks against equal-count regular tiles and against random superpixels on fault energy detected and on log-score; mask repeatability across frames (fraction of surfel mass assigned to the same entity in two views); oracle masks as an upper bound.

**8. "The method cannot see faults where texture is absent, which is where monocular depth fails." (C, R)** *Strongest form:* blank roofs, water and glass give zero image stiffness exactly where the prior is least reliable, so the detector is blind where needed. *Answer:* partly true and stated (Section 6, and the regime table 4.8). The label channel (outline parallax) gives some stiffness on blank bounded surfaces; the prior heads cover the rest only by extrapolation. *Must exist:* a fault-detection result stratified by pixel stiffness (the $\vartheta$ regimes) and by harm, reporting recall in each; the fraction of total fault harm on pixel-blind entities; and the counterfactual-blindness check of Section 4.9. The paper should state the fraction of harm that is detectable, not imply all.

**9. "The two sources are not independent: both read the same image." (C, R)** *Strongest form:* a shadow edge misleads the depth network and also carries contrast; the null fails by shared cause. *Answer:* real; placebo tensions do not reveal it. *Must exist:* on exact-depth data, the correlation between the monocular placement error and the image placement error per entity, and tension recalibrated when it is nonzero.

**10. "Why Gaussian surfels; any surface and a differentiable renderer would do." (C)** *Strongest form:* Section 4.11 concedes this. *Answer:* the planar surfel is where the factorisation is exact and placement is three numbers; the paper must show it is more than convenience. *Must exist:* an ablation replacing surfel entities by a mesh patch or depth-map tile with the same three parameters and the same records; if the result is equal, the claim shifts from "Gaussians" to "per-source information on a placement parameterisation", and the title should follow.

## 4. Claims in `docs/learned-indicator.md` that prior work contradicts, weakens or already makes

| Section | Quoted claim | Problem | Change |
| --- | --- | --- | --- |
| Abstract | "Gaussian splatting cannot say so, because it stores geometry and appearance and nothing about how well either is known." | Contradicted for splatting with uncertainty: per-Gaussian uncertainty or Fisher information is stored by GauSS-MI, PUP 3D-GS, OUGS, VBGS-SLAM. | "Standard splatting stores...; uncertainty-aware variants keep one score per primitive for the image loss." |
| 1.1 step 1 | "Gaussian splatting stores the first two." | Same. | "Standard 3DGS and 2DGS store..." |
| 1.2, On the name | "a short search found no method with this exact name, which is weak evidence that it is free" | GauSS-MI occupies "Shannon mutual information with splatting"; the content is Fisher information. | Drop "information-theoretic" from the title; use "information-carrying surfels" or "per-source information on surfels", as the supporting notes already suggest. |
| 2, uncertainty | "None found keeps information per evidence source in order to compare sources, or calibrates it against realised scatter." | Holds for the abstracts read; OUGS also propagates covariance through the rendering Jacobian. | Keep, but add "at abstract level" and cite OUGS beside the Jacobian claim; do not call the use of the rendering Jacobian new. |
| 2, depth regularisation | "None of those read tests the prior against the pixels with an error bar or separates the repeated part." | Weakened: Chu et al. (2607.02554) use photometric masks to decide where depth is trusted, and In Depth We Trust uses a hand-thresholded rendered-depth test. | State precisely: the existing tests are thresholded heuristics without a null or attribution. |
| 2, depth regularisation | "Monocular depth with scale alignment is a standard regulariser" | Correct, but the standard alignment uses SfM points (In Depth We Trust, DN-Splatter, Chung et al.), which the plan does not have. | State that the plan's per-frame scale and offset, absorbed in the depth family (4.6), replace SfM-based alignment, and that this is a cost to be measured. |
| 2, large-scale | "Tiled and hierarchical systems reconstruct city-scale scenes from structure-from-motion inputs. They ... do not store or evaluate what the map knows." | Hierarchical 3D Gaussians offers optional monocular depth regularisation for large scenes; a LoD of Gaussians runs on MatrixCity without partitions. | Say "regularise with monocular depth without testing it" and cite both. |
| 2, pose-free | "Splatting without structure-from-motion poses already uses monocular depth to initialise" | Read abstracts do not say it for COLMAP-Free 3DGS; InstantSplat initialises from a geometric foundation model. | Cite InstantSplat for the verified claim; read the COLMAP-Free paper before keeping the other. |
| 3.2 | "it is the precise sense in which the primitive is where the two meet" | True for a plane; but the rendering Jacobian chain is not new (OUGS), and 2DGS already supplies the intersection depth. | Present as a specialisation of the renderer chain rule to planar surfels. |
| 4.6 | "Honest prior information saturates ... the saturated form lets the pixels speak at the right rate with no gate at all" | This is fusion with correlated errors, the problem covariance intersection solves model-free; the saturating form is the standard result for a shared bias term. | Cite covariance intersection as comparator; add it as a baseline of the fusion rule (Section 5); claim only the estimation of $\varrho$ without labels. |
| 4.2 | "the log-likelihood ratio of 'the two sources see one surface' against 'they see two'" | It is Hausman-type and solution-separation-type; the plan already says the statistic is classical in 2 and in the notes. | Cite Hausman 1978 and Blanch et al. in Section 4.2 itself. |
| 4.7 | "No discrete association between image points is estimated, stored or trusted." | Holds for the mapping stage only; the sweep over entity offset is a search, outline parallax uses masks, and many comparison systems are marketed on the same word. | Keep the three qualifications of the supporting notes in the paper body, and call the feed-forward family "matching by network". |
| 5.1 | "the negative log density of the true surface under the map, in nats per square metre" | The log of a density has the units of its reference measure; a surface point has a one-dimensional density along the normal (per metre), so "nats per square metre" mixes a density with an area average. | Define the score as the Gaussian negative log predictive density of the signed normal error per entity (a function of metres through a stated reference scale), area-weighted, and report a reference-scale-free comparison (a difference of scores between maps). |
| 6, last bullet | "overlap with uncertainty-aware splatting and selective depth supervision must be examined further." | Now examined at abstract level here. | Replace by the contrast of Section 2 of this note after full-text reading of FisherRF, OUGS and In Depth We Trust. |

## 5. Baseline recommendations

Costs below are for one strip, with the stated assumptions; every time is an *estimate* unless a source is cited. Code status comes from fetched pages.

| Baseline | Must it be run | Code | Aerial-strip cost |
| --- | --- | --- | --- |
| Depth-only, pixels-only, fixed and decaying weight, robust loss, frame removal, consensus gate | Yes, the core of S5 | In-repo | Low: same fit code with a different weight; hours of GPU per strip (estimate). |
| In Depth We Trust (2604.05715) | Yes, nearest fault-side competitor | Not verified; the page read links the public 3DGS code base only. If no release exists, re-implement the virtual-stereo mask (about 100 lines on the plan's renderer, since it needs rendered depth and reprojection) | Low if re-implemented. SfM points for scale alignment are not available; use the plan's per-frame alignment for all methods, and note the departure. |
| CDGS (2502.14684) and Chu et al. (2607.02554) confidence or photometric masks | Yes, as masking variants | Not verified | Low if re-implemented as masks. |
| DN-Splatter (2403.17822) | Yes, as the depth-regularised splatting reference | Public, Apache-2.0, nerfstudio with gsplat; has a mono-only option that skips SfM alignment; indoor-oriented | Moderate: nerfstudio data conversion of aerial strips; training on the order of an hour per 500 frames (estimate). |
| FisherRF (2311.17874) as an information field | Yes, for objection 1 | Public per project page; backend for 3DGS | Moderate: compute Fisher scores on the same primitives; no training change. Score it on the log-score; it has no source split, so report it as a one-source field. |
| PUP 3D-GS (2406.10219) and OUGS (2511.09397) as information fields | Yes, at least PUP as a second sensitivity field | Not verified for either | Low to moderate; PUP's score is a second-order quantity cheap once a model is trained. |
| Bayes' Rays (2309.03185) | Optional; NeRF only (a third-party 3DGS port was seen in a search listing, unverified) | Public, nerfstudio, supports Nerfacto, Instant-NGP, Mip-NeRF | High for aerial (NeRF training); include only if a splat port is verified, otherwise cite and exclude with the reason. |
| Large-scale splatting: CityGaussian, Hierarchical 3D Gaussians, HUG, A LoD of Gaussians | Yes, at least two, on MatrixCity | CityGaussian public (CC BY-NC-SA; MatrixCity Aerial among datasets); Hierarchical 3D Gaussians public (licence not read; toy example of 1,500 images measured on an A6000 at about 47 min calibration, 95 min chunk generation, 171 min optimisation, per its repository); HUG and LoD code not verified | High: COLMAP-based preprocessing and partitioned training, tens of GPU-hours for a multi-block area (estimate). |
| SfM plus multi-view stereo | Yes, the matching reference for F3 and S6 | Standard open implementations (recalled, not checked) | Moderate to high on dense strips; CPU-heavy (estimate). Its role is the error-against-texture-energy curve. |
| VGGT (2503.11651) | Yes, as feed-forward baseline and as an alternative depth source | Public; the original checkpoint is non-commercial and gated by an application | Moderate: memory grows with frame count; run in overlapping windows of a few tens to a hundred frames (estimate) and align windows. |
| DUSt3R, MASt3R | Optional, for pose and pointmap comparison | Not verified here | Moderate; quadratic in pairs unless windowed. |
| MVSplat | No: small-baseline, sparse-view training distribution | Not verified | Not recommended. |
| Covariance intersection fusion of depth records | Yes, a cheap fusion-rule baseline for the saturation claim | Closed form, no code | Negligible. |

**Order of effort.** First the in-repo schedule and gate baselines and the information-field comparison (FisherRF, PUP): these decide S4 and S5 on MatrixCity block 3 at low cost. Then the masking baselines. The large-scale and SfM-plus-MVS baselines are needed only for Stage E and the F3 curve; running them on one block is enough for credibility, not on every scene.
