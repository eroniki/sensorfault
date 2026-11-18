# Information-Carrying Gaussian Surfels for Correspondence-Free Aerial Mapping with Monocular Depth: Objections, Prior Work, Verdict

Nothing in this note has been measured. Statements about prior work are marked *verified* (checked
against the paper, its arXiv page or proceedings page) or *recalled* (from memory, not checked).

## 0. The framing under examination

The paper is about a representation and a mapping method. A map stores geometry, appearance and
*information*: what is known about the geometry, per source of evidence. The scene is a set of
Gaussian surfels grouped into surface entities. Two sources speak about the placement of each entity:
the monocular depth maps, and the images of all views rendered through the same primitives. The fault
indicator, "listen to the depth prior or to the pixels", is something the representation makes
possible. Five claims are attached:

- C1. The map carries information, kept per source and calibrated.
- C2. The method is correspondence-free: no matching, no tracks, no triangulated points.
- C3. Alignment of geometry and appearance can be measured probabilistically.
- C4. Pose errors and model errors are absorbed by the same model.
- C5. It scales to large areas from posed aerial video in strips.

## Part 1. The argument, and where each link can break

The chain is given in plain sentences. Each link is followed by the condition under which it fails.

1. *A monocular depth map is dense but can be wrong over a whole surface, and wrong in the same way
   in every frame.* Agreement among depth maps cannot reveal this. Hence a second, independent source
   is needed. **Breaks if** such repeated faults are rare or small on real aerial data.
2. *The images are that source, through parallax.* Parallax is usually extracted by matching points.
   Matching fails or abstains on weak and repeated texture and discards most pixels. Hence use every
   pixel without matching: render the shared surface into each view and compare. **Breaks if** the
   comparison has no capture range at the size of the faults (F3).
3. *Rendering has a small capture range.* It refines; it does not search. Hence something must bring
   the surface within range. That is the logical role of the monocular depth: it replaces the
   wide-range search that matching would perform. **Breaks if** the prior is outside the range
   exactly when it is faulty, which is the case of interest; a search over entity placement is then
   required, and the method is no longer purely local.
4. *How strongly the images speak about a surface varies by orders of magnitude,* with texture
   contrast and the spread of viewing directions. A fixed weight between depth and image losses is
   therefore wrong almost everywhere. Hence store, per entity and per source, how much each source
   knows: its information. **Breaks if** the stored information is not honest (F4).
5. *With information stored per source, three things follow by arithmetic:* fusion (add), consistency
   (compare), and attribution (which source is the outlier). The fault indicator is the consistency
   check. **Breaks if** the sources are not independent given the surface, or the test is evaded by
   free appearance.
6. *Nuisances that move many entities together are cheap to marginalise;* a fault that moves one
   entity is not a nuisance. Hence pose and per-frame depth calibration are absorbed at a stated cost
   and surface faults are detected. **Breaks if** a fault looks like a nuisance (F2).
7. *Information adds.* Blocks and strips merge by summation, and regions where neither source knows
   anything are visible in the map as such. Hence the representation suits large areas. **Breaks if**
   information is tied to a linearisation point that moves when blocks are merged.

Links 3 and 6 carry the two claims most likely to be overstated, C2 and C4. They are treated first.

## Part 2. The three statements made exact

### 2.1 In what sense correspondence-free

*Sense.* No association between image points is estimated or stored. Two pixels in two views are
related only by being rendered from the same primitive. Visibility and occlusion are decided by the
rasteriser. Evidence is a residual between a rendered and an observed view.

*What it buys.* Every pixel contributes in proportion to its local contrast, so weak texture adds
its small information over a whole entity instead of being rejected by a matching threshold.
Repeated texture produces no wrong matches inside the capture range. Cost is linear in pixels and
views. One probabilistic model covers all evidence.

*Where it is weaker.* The capture range is about one surfel footprint of residual parallax. Explicit
matching searches over hundreds of pixels. At survey spacing, a depth error of a few percent of the
flying height is outside the range.

*Three qualifications that the paper must state.*

1. "Posed" video: if the poses come from structure-from-motion, correspondences were used upstream.
   The claim holds for poses from satellite positioning and inertial sensing, refined by rendering.
2. The profile over an entity's three placement parameters, needed for gross faults, is a search. It
   is a plane sweep per entity. It searches over surfaces and not over point matches, which is a real
   difference, but a reviewer may call it correspondence by another name.
3. Outline parallax uses per-frame segment masks. Their identities are fused on the primitives, not
   matched in 2D, so the claim survives; the dependence on a segmenter must be stated.

*Second reading of "cross view".* Views from different strips, passes, or nadir and oblique cameras
enter the same way, as further frames rendering the same primitives. They add directions of
parallax, which resolves outlines parallel to one flight line and objects moving along it. They also
bring appearance change between passes. An appearance term per pass absorbs it, at a cost in retained
placement information that the model can state. Larger viewpoint change shrinks the capture range
further. The formulation covers this reading in principle; nothing suggests it is easy.

### 2.2 The measure of alignment between geometry and appearance

For entity $m$ with placement $\boldsymbol\theta_m$ (offset in metres, two tilts in radians), each
source $s \in \{D, I\}$ supplies an estimate $\hat{\boldsymbol\theta}^s_m$ and an information matrix
$\Lambda^s_m$ (inverse squared metres and radians).

| Quantity | Symbol | Units | Meaning |
| --- | --- | --- | --- |
| stiffness | $\Lambda^I_m$ | 1/m², 1/rad² | how strongly appearance constrains placement: texture energy times spread of viewing directions |
| stretch | $\mathbf t_m = \hat{\boldsymbol\theta}^D_m - \hat{\boldsymbol\theta}^I_m$ | m, rad | disagreement of the two sources |
| stored energy | $T_m = \mathbf t_m^\top \bigl((\Lambda^D_m)^{-1} + \hat\varphi (\Lambda^I_m)^{-1}\bigr)^{-1} \mathbf t_m$ | none | the alignment measure |

Null: $T_m \sim \chi^2_3$ if both sources are unbiased with independent errors and the dispersion
$\hat\varphi$ and the between-frame correlation of the depth source are estimated correctly. Both
conditions are empirical. The null is therefore established by placebo comparisons within one source
(split halves of the frames) and by injection of known displacements, not assumed.

The measure is a difference of two estimators in units of their joint uncertainty. It is the same
object as solution separation, the Hausman test, and parameter-shift tension statistics. The paper
claims the two inputs and their calibration, not the statistic.

### 2.3 What "absorbed" means

Marginalising a nuisance $\boldsymbol\nu$ with prior information $\Lambda_{\nu 0}$ leaves

```math
\Lambda^{\mathrm{marg}}_{\theta\theta} = \Lambda_{\theta\theta} - \Lambda_{\theta\nu}\bigl(\Lambda_{\nu\nu} + \Lambda_{\nu 0}\bigr)^{-1}\Lambda_{\nu\theta} \ \preceq\ \Lambda_{\theta\theta} .
```

Absorption is a subtraction of information. An error is absorbed when the remaining information is
still honest; it does not vanish. The ledger:

| Error | Treatment | Cost | What escapes |
| --- | --- | --- | --- |
| Pose of a frame | Marginalised, six parameters per frame or a low-order field along the strip, with the navigation prior | Each frame loses the placement information that a common motion of all its entities could explain; small when a frame sees many entities | A drift coherent along a whole strip bends surface and trajectory together. Neither source sees it. It is a gauge, fixed only by absolute positioning or ground control |
| Per-frame scale and offset, shared distance bias of the depth model | Estimated with gates applied | Any depth error affine over a frame is not a fault. In nadir views this includes a uniform compression of relief | A fault covering most of a frame biases the fit |
| Per-frame term of monocular error | Averaged by fusion; tested by consensus | None | Correlation between neighbouring frames inflates the null unless modelled |
| Surface term of monocular error (repeated fault) | **Cannot be absorbed.** Detected by tension | | Undetected where the image source is blind |
| Rendering model error | Dispersion $\hat\varphi$ per stratum and channel | Image information divided by $\hat\varphi$ | Structured misfit that is not noise-like: reflections, movers |
| Appearance freedom | Projective appearance in the test, or the Schur complement | Retained fraction $\kappa_m \in [0,1]$ | View-dependent colour in a strip leaves little; this is stated, not hidden |
| Reflection, transparency | **Cannot be absorbed.** Layer test (outline against interior) | | A reflector with no outline in view |
| Object moving along the strip | **Cannot be absorbed** in one strip | | Needs a second strip |

The paper should use "accounted for" and give this table. "Absorbed" invites the reading that errors
disappear.

## Part 3. Objections

### F1. The name promises more than the content, and the ground is occupied

"Information-theoretic" suggests entropy and mutual information. The content is Fisher information
in Gauss-Newton form. Information per Gaussian has been computed before (Part 4): for view selection,
pruning and active reconstruction; uncertainty-aware and variational Bayesian splatting keep variances
or posteriors per primitive. A reviewer will say that a map with geometry, appearance and uncertainty
exists.

*What remains.* Four properties were not found together or, as far as was checked, separately: the
information is about *placement of surface entities*, not per-primitive parameters; it is kept *per
source* so that sources can be compared; it is *calibrated* by placebo and injection; and it is used
for *consistency and attribution*, not only weighting or view selection.

*Change.* Prefer a name that says what is stored, such as information-form or information-carrying
surfels. Define information on the first page as Fisher information of placement per source.

### F2. A fault that looks like a nuisance is absorbed, and a nuisance that is mis-modelled looks like a fault

Per-frame affine alignment hides relief compression. A low-order pose field hides a tilted or bowed
surface fault of strip size. Conversely, an unmodelled lens or rolling-shutter effect moves entities
coherently near the image border and raises tension there. The boundary between nuisance and fault is
a modelling choice and must be stated as one.

*Test.* Inject frame-wide and entity-wide errors of each kind; report which are absorbed, which are
flagged, and the information lost.

### F3. Without correspondences there is no capture range, and with good texture correspondences are better

On textured surface, matching with a wide search is more robust and the standard aerial pipeline is
accurate. The method's advantage is confined to weak and repeated texture, to uniform probabilistic
treatment, and to cost. On weak texture the information is small by definition, so the advantage
must be shown as a gain in accuracy over entities, not asserted.

*Test.* Entity-level placement error against stated uncertainty, for the method and for a
matching-based reference, as a function of texture energy. The crossing point, if any, is the result.

### F4. Stored information is only useful if it is honest and stable

Nominal Gauss-Newton curvature overstates image information by a large factor, because residuals are
model misfit and spatially correlated. Information is also evaluated at a linearisation point and
under an appearance model; it changes when either changes. A stored field that is stale or optimistic
is worse than none.

*Test.* Calibration of the information field: on scenes with exact geometry, the standardised
placement error of entities against their stated covariance, per source and fused, with coverage of
nominal intervals. This is the decisive experiment for a representation paper.

*Change.* Define the stored information at a fixed protocol (depth-placed surface, projective
appearance, dispersion applied) and recompute it after gating.

### F5. Free appearance evades the measure

A joint fit relieves tension through view-dependent colour, opacity and floaters. The measure is
therefore taken as a score test at the depth-placed surface with projective appearance from disjoint
source frames, and the retained fraction $\kappa_m$ states what any fitted appearance costs. The
objection is answered only if the paper keeps the test model and the delivered model apart.

### F6. Independence of the two sources

The depth model reads the same image whose pixels form the image source. A shadow edge can mislead
the depth model and also carry high contrast. Errors of the two sources may share a cause even though
one uses a single image and the other parallax. Placebo tensions do not reveal this. Exact depth
does: the correlation between the monocular placement error and the image placement error per entity.

### F7. Large area is claimed but only block-level reasoning is given

Additivity of information supports merging blocks and strips. It does not address the gauge between
blocks, the cost of the entity sweep at scale, or memory for lifted features. The claim should be
"block-wise with additive merging", demonstrated on several blocks, with cost reported.

### F8. A decaying depth weight remains the baseline to beat

Where pixels are informative a schedule already lets them win. The representation must show gains
where a schedule loses: correct prior retained on weak texture; blank bounded surface placed by
outline parallax; reflections not baked into geometry; and a calibrated statement of what is unknown.

### F9. Five claims are too many for one paper

C1 and C3 are one claim (calibrated per-source information makes alignment measurable). C2 is a
property with qualifications. C4 is a ledger. C5 is a demonstration. The paper should lead with the
first, state the second and third precisely, and treat the fifth as evidence.

### Summary

| Objection | Fatal if | Cheapest test |
| --- | --- | --- |
| F4 honesty | stated covariance does not predict placement error after calibration | coverage on one block with exact depth |
| F3 capture and advantage | no texture regime where the method is at least as accurate as matching | error against texture energy |
| F2 nuisance boundary | common fault types are absorbed silently | typed injections |
| F6 independence | placement errors of the two sources are strongly correlated | correlation with exact depth |
| F8 schedule | a decaying weight matches every reported gain | fit comparison |

## Part 4. Prior work and what the name would be accused of duplicating

### 4.1 Is the name taken

A search for "information-theoretic Gaussian splatting" as the name of a method returned none. The
search was through a general engine and is weak evidence. The nearest titled uses of information
theory with splatting are below.

| Work | What it does | What it pre-empts |
| --- | --- | --- |
| Xie, Cai, Zhang, Yang, Pan, "GauSS-MI: Gaussian Splatting Shannon Mutual Information for Active 3D Reconstruction", arXiv 2504.21067, 2025. *Verified (abstract).* | A probabilistic model of visual uncertainty per Gaussian and a mutual-information criterion for next-best view. | "Information-theoretic" in the Shannon sense with splatting; per-Gaussian uncertainty. |
| Jiang, Lei, Daniilidis, "FisherRF". *Title verified; details recalled.* Wilson et al., "POp-GS", arXiv 2503.07819. *Verified.* | Fisher information of rendered colour with respect to Gaussian parameters, diagonal or per-ellipsoid blocks, for view selection. | Fisher information from the renderer's Jacobian. |
| Hanson et al., "PUP 3D-GS", arXiv 2406.10219. *Verified.* | Per-Gaussian Fisher blocks on position and scale for pruning. | Spatial Fisher information stored per primitive. |
| Goli et al., "Bayes' Rays", CVPR 2024. *Verified (title, venue).* | Post-hoc Laplace uncertainty on a spatial perturbation field. | Uncertainty of collective spatial displacement. |

### 4.2 Uncertainty-aware and Bayesian splatting, and maps with covariance

| Work | What it does | Difference |
| --- | --- | --- |
| Zhu, Zhang, Xu, Ren, "VBGS-SLAM: Variational Bayesian Gaussian Splatting SLAM", arXiv 2604.02696, 2026. *Verified (abstract).* | Posterior over poses and splat parameters with closed-form variational updates. | A single posterior, not separated by source; no consistency test between sources. Which parameters carry uncertainty was not read. |
| Tran and Kosecka, "VarSplat", arXiv 2603.09673, 2026. *Verified (abstract).* | A learned appearance variance per splat, composited to a per-pixel uncertainty that weights tracking and registration. | Uncertainty of appearance, learned; not information about geometry. |
| Hu et al., "CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-aware 3D Gaussian Field", ECCV 2024. *Title and venue verified; content recalled.* | Depth uncertainty in a Gaussian field for RGB-D tracking and mapping. | Sensor depth, not a monocular prior under test. |
| Direct photometric tracking against a Gaussian map (monocular and RGB-D Gaussian SLAM). *Recalled.* | Poses by rendering and comparing, without feature matching. | Correspondence-free pose estimation on Gaussian maps exists. |
| Information-form filters and smoothers in SLAM. *Recalled.* | Maps and trajectories kept as information matrices; additive fusion; marginalisation by Schur complement. | The algebra of Sections 2.2 and 2.3 is theirs. |

### 4.3 Correspondence-free and pose-free splatting

| Work | What it does | Difference |
| --- | --- | --- |
| Fu et al., "COLMAP-Free 3D Gaussian Splatting", CVPR 2024. *Title and venue verified; content recalled.* | Sequential video, monocular depth to initialise local Gaussians, poses by photometric alignment. | Monocular depth as initialiser in a matching-free pipeline already exists. No information model and no test of the prior. |
| "Correspondence-Guided SfM-Free 3D Gaussian Splatting for NVS", arXiv 2408.08723; "ICP-3DGS: SfM-Free 3D Gaussian Splatting", arXiv 2506.21629. *Titles from a search listing only.* | Pose-free variants that reintroduce correspondences or registration for robustness. | Evidence that pure photometric alignment lacks capture range, which is F3. |
| Classical direct methods (dense and semi-dense direct odometry, 2010s). *Recalled.* | Photometric alignment without features, with a known small basin and coarse-to-fine remedies. | The capture-range limit and its standard mitigation. |

### 4.4 Large-scale aerial splatting

"CityGaussian" (arXiv 2404.01133, title verified); Su et al., "HUG: Hierarchical Urban Gaussian
Splatting with Block-Based Reconstruction for Large-Scale Aerial Scenes", ICCV 2025 (verified);
Yu et al., "VDGS", arXiv 2609.23049 (verified from a listing); Yao et al., "ARSGaussian",
arXiv 2412.18380 (verified; LiDAR constraints). These partition by block and target rendering
quality, with structure-from-motion poses and points. None was found to use monocular depth as the
geometric source with a test of it, or to carry information per source. Song et al., "AerialMetric",
arXiv 2606.29716 (verified) reports that monocular metric depth degrades on aerial viewpoints, which
supports link 1.

### 4.5 Depth priors in splatting, lifting, and the statistic

Fixed or heuristic reliability weights for monocular depth (DN-Splatter, WACV 2025; CDGS, arXiv
2502.14684; "In Depth We Trust", arXiv 2604.05715; arXiv 2607.02554; ConfidentSplat, arXiv
2509.16863; all verified at abstract level); decoupled geometric and photometric gradients with a
LiDAR prior (EnerGS, arXiv 2604.26238, verified); masks and features lifted to Gaussians (Gaussian
Grouping, ECCV 2024; feature field distillation, NeurIPS 2022; verified). Solution separation
(Blanch et al., titles and venues verified), the Hausman test and tension statistics between data
sets (recalled) are the ancestors of the alignment measure.

### 4.6 What remains new

Not found in any of the above: (a) information about entity placement stored *per source* on shared
primitives; (b) its calibration without labels by within-source placebo comparisons and renderer
injection; (c) a consistency measure between a monocular prior and rendered views with a stated null
and stated blind regions; (d) a ledger of what marginalisation costs for each nuisance; (e) outline
parallax through rendered labels and the outline-against-interior layer test. Several neighbours
were read as abstracts only, and one title on decoupling depth and photometric gradients could not be
opened.

## Part 5. Verdict

**Is the framing an improvement?** Yes. A representation with honest per-source information is a
larger and more defensible claim than a fault gate, and the gate, the fusion rule and the report of
unknown regions follow from it by arithmetic. The framing is also easier to attack on novelty,
because every single ingredient has a neighbour. The paper stands on the assembly and on one
experiment: that the information field is calibrated.

**The single claim.** *A Gaussian surfel map that stores, for each surface entity and each source of
evidence, how much that source knows about the entity's placement, calibrated without labels, makes
fusion, consistency and attribution between a monocular depth prior and rendered views a matter of
arithmetic, without point correspondences.*

**What must be shown, cheapest first.**

1. *Calibration of the information field* on one block with exact geometry: standardised entity
   placement error against stated covariance, per source and fused.
2. *Capture range and advantage:* error against texture energy and against size of prior error, with
   a matching-based reference.
3. *Nuisance ledger by injection:* pose, per-frame affine, entity fault, reflection.
4. *Mapping quality on several blocks:* surface accuracy against pixels only, depth only, fixed
   weight, decaying weight, information fusion without and with gates; clean-scene change.
5. *Robustness curves:* accuracy against pose noise and against prior fault rate.
6. *A second pass or oblique views,* if the cross-view reading is claimed.

**Kill criteria.** Abandon the representation claim if, after calibration, stated covariance does
not predict placement error; or if no texture regime exists where rendering-based evidence is at
least as accurate as matching; or if a decaying depth weight reproduces every reported gain.

**Wording to avoid.** "Information-theoretic" without definition; "absorbed" without the ledger;
"correspondence-free" without the three qualifications; "large-area" without multi-block evidence.

**What stays unknown.** Whether image information, once made honest, is large enough on real aerial
strips to say something useful about most entities, and whether the two sources err independently.

## Sources consulted

- https://arxiv.org/abs/2504.21067
- https://arxiv.org/abs/2604.02696
- https://arxiv.org/abs/2603.09673
- https://arxiv.org/abs/2403.16095 and https://mlanthology.org/eccv/2024/hu2024eccv-cgslam/
- https://arxiv.org/html/2406.10219v3, https://arxiv.org/html/2503.07819v2, https://jiangwenpl.github.io/FisherRF
- https://openaccess.thecvf.com/content/CVPR2024/html/Goli_Bayes_Rays_Uncertainty_Quantification_for_Neural_Radiance_Fields_CVPR_2024_paper.html
- https://openaccess.thecvf.com/content/CVPR2024/papers/Fu_COLMAP-Free_3D_Gaussian_Splatting_CVPR_2024_paper.pdf
- https://arxiv.org/pdf/2408.08723 and https://arxiv.org/html/2506.21629v1 (listings only)
- https://arxiv.org/pdf/2404.01133 (listing only)
- https://openaccess.thecvf.com/content/ICCV2025/html/Su_HUG_Hierarchical_Urban_Gaussian_Splatting_with_Block-Based_Reconstruction_for_Large-Scale_ICCV_2025_paper.html
- https://www.alphaxiv.org/abs/2609.23049
- https://arxiv.org/abs/2412.18380v2
- https://arxiv.org/abs/2606.29716
- https://arxiv.org/abs/2403.17822, https://arxiv.org/abs/2502.14684, https://arxiv.org/abs/2604.05715, https://arxiv.org/abs/2607.02554, https://arxiv.org/abs/2509.16863v1
- https://arxiv.org/html/2604.26238
- https://arxiv.org/abs/2312.00732 and https://mlanthology.org/neurips/2022/kobayashi2022neurips-decomposing
- https://web.stanford.edu/group/scpnt/gpslab/pubs/papers/Blanch_et_al_IONGNSS_2012_v2.pdf and https://dblp.uni-trier.de/rec/journals/taes/BlanchWELPRSK15.html
