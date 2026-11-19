# Publication Plan: What the Paper Is, and How It Is Staged

The method plan (`docs/learned-indicator.md`, about 7,600 words, twelve method subsections) is a programme, not a paper. This document fixes the claim, the cuts, two concrete paper designs, the figures and tables, the evidence each sentence needs, the fallbacks, the calendar, and the vocabulary. Nothing has been measured; every number in square brackets is a placeholder, and every threshold is a proposal to be fixed in the preregistration before the scoring script is first run. Dates were looked up on 2026-11-18; entries marked *(verify)* come from third-party pages or are assumed.

## 1. Claim, contributions, cuts

**Claim (one sentence).** A Gaussian-surfel map built from posed aerial video and monocular depth is overconfident by the share of depth error that all frames repeat, and storing a calibrated Fisher information on placement separately for the depth prior and for the pixels, on the same three coordinates, exposes that repeated error as a between-source statistic with a null distribution and yields a map whose stated uncertainty predicts its realised error.

**Contributions (three, each checkable).**

| # | Contribution | How a reviewer checks it |
| --- | --- | --- |
| C1 | Per-source information records on planar surfels: the image Jacobian with respect to placement factors through the depth Jacobian, so prior and pixels are quadratic forms on one offset and two tilts; records add over frames and tiles | Finite-difference test of the factorisation on the renderer (relative error below $10^{-3}$, proposed); tile-merged record equals the joint record to numerical precision; shown in a released unit test |
| C2 | The overconfidence law $\varrho + (1-\varrho)/n$ for depth-only fusion, and the score test at the depth-placed surface that detects repeated faults; both against a depth-consensus gate | Measured against predicted variance ratio on exact depth; detection rate at fixed false-alarm rate against baselines (Fig. 3, Fig. 5, Table 2) |
| C3 | A label-free calibration of the pixel information (dispersion from frame replicates, honesty factor from displacement injected through the renderer), scored by a proper rule, the surface log-score | Coverage and log-score of six maps (Table 1); stated against realised information per surface class (Fig. 4) |

**Cut and deferred.** Each cut leaves the claim intact because the claim needs only the depth family, the image colour channel, one entity definition, the score test and the calibration.

| Cut item (plan section) | Why the claim survives | Where it goes |
| --- | --- | --- |
| Frozen-feature and label channels, outline parallax, layer test (4.4) | The claim concerns repeated depth faults and calibration, not reflections or blank surfaces; colour alone carries the test | Follow-up paper on pixel-blind entities |
| Learned prior, support and verdict heads (4.8, 4.9) | The gate can be a threshold on $E_m$; no learned part is needed to show overconfidence or detection. The no-fault-labels constraint is met trivially | Follow-up; one paragraph of future work |
| Segmenter-lifted entities (4.1) | Entities are fixed ground tiles of [8 m] side with a depth-normal split; the unit-of-analysis question (S8) becomes an ablation, not a premise | Supplement, ablation row of Table 3 |
| Video-encoder readout by injection (4.5, last bullet) | Injection is used only to calibrate colour information | Dropped |
| Retained fraction $\kappa_m$ and full nuisance ledger (4.3, 4.6) | Only pose and per-frame depth scale/offset are absorbed, with their retained fraction reported; exposure, passes and the rest are listed as assumptions | Supplement for the two absorbed nuisances; rest deferred |
| Cross-strip, oblique, repeated-pass evidence (4.7) | Single-pass strips are enough for the law and the test | Deferred |
| Large-area tiling beyond an additivity check (Stage E) | Additivity is shown as an identity test on two tiles, not as a scaling claim; the word "large-area" is removed from the title | Deferred; needs multi-block data |
| Matching baselines (SfM+MVS) and correspondence-free claim of Section 4.7 | The paper says "no matches are used by the map", not "better than matching" (S6 is not tested) | Supplement table, if time |
| Seven-point contribution list; attribution posterior (4.8) | Folded into C1 to C3; attribution is reduced to the sign of the stretch | Deferred |

The word "learned" in the file name no longer describes the paper. The paper is self-supervised in the sense that no fault labels exist at any stage; frozen monocular depth models are the only learned components.

## 2. Two paper designs

### 2.1 Design (a): CVPR-style, 8 pages plus references, supplement allowed

**Title.** *Two Springs on One Surface: Per-Source Information on Gaussian Surfels for Auditing Monocular Depth in Aerial Mapping.*

| Section | Pages | Content |
| --- | --- | --- |
| 1 Introduction | 1.0 | Fig. 1; problem, idea, claim, three contributions |
| 2 Related work | 0.6 | Information and uncertainty in splatting; depth-regularised splatting; solution-separation tests |
| 3 Method | 2.4 | 3.1 Records on surfels (0.9); 3.2 Tension, split, repetition law (0.8); 3.3 Calibration and fusion with gate (0.7); Fig. 2 |
| 4 Experiments | 3.4 | Setup 0.5; overconfidence law 0.7; honesty of pixel information 0.6; detection 0.6; mapping 0.6; real data 0.4 |
| 5 Limitations and conclusion | 0.6 | Capture range, blind directions, independence assumption |
| Total | 8.0 | References extra; supplement: derivations, ledger, ablations, extra scenes |

Equations in the main text: *placement-depth Jacobian* (inline), *depth record*, *image Jacobian factorisation*, *entity image information*, *tension and stretch*, *misfit split*, *score test at the depth-placed surface*, *honest prior saturation* (the overconfidence law), *dispersion estimator*, *injection exchange identity*, *surface log-score*. Supplement: Schur retained fraction, marginalisation of nuisance, outline parallax, heads.

### 2.2 Design (b): RA-L / IROS, 6 pages (8 allowed at charge; do not use), no supplement

**Title.** *Auditing Monocular Depth with Per-Source Information on Gaussian Surfels.*

| Section | Pages | Content |
| --- | --- | --- |
| 1 Introduction | 0.6 | Fig. 1 at one-column width |
| 2 Related work | 0.3 | Merged into the introduction if short |
| 3 Method | 1.7 | Records and factorisation; tension and score test; honest prior saturation; calibration by injection (one paragraph) |
| 4 Experiments | 2.4 | Exact-depth block (law, honesty, detection, log-score); real scenes with laser truth (one table) |
| 5 Limitations and conclusion | 0.3 | |
| References | 0.7 | About 25 entries |
| Total | 6.0 | Video attachment: teaser sequence of the map and stated uncertainty |

Equations: *image Jacobian factorisation*, *tension*, *score test*, *honest prior saturation*; the depth record, the dispersion and the injection identity are stated in words or inline; the surface log-score is defined in one inline sentence. Everything else is cited to the supplement-free preprint note or omitted. Design (b) drops the misfit split, entity details, the fusion-with-gate map comparison, and ablations except the entity-versus-tile row.

### 2.3 Recommendation and submission rules

**First submission: design (b), RA-L with the IROS 2027 presentation option.** This is the chosen route; design (a) follows it. The reasons are calendar and risk. The CVPR 2027 deadline (submission 2026-11-16) has passed. The next vision deadlines are ICCV 2027 (projected 2027-03-01 registration and 03-08 submission, from a third-party tracker, *verify*; the official ICCV 2027 page lists only conference dates, 2027-10-03 to 10-07) and CVPR 2028 (assumed mid-November 2027, *verify*). A March paper would need the joint photometric and depth fit, which is mostly stubs, plus UseGeo, in about fifteen weeks on one GPU. Design (b) needs only the depth-only pipeline with real monocular depth (Stage A), the injection study on the depth-placed surface (Stage B) and a small mapping comparison, and RA-L accepts submissions on a rolling basis: papers submitted between 2026-08-01 and 2027-04-30 can be presented at IROS 2027 (Florence, 2027-09-26 to 10-01; IROS direct deadline 2027-03-01), with a decision stated to arrive within six months.

**One body of experiments, two papers.** Concurrent submission of substantially similar work is not allowed. CVPR 2027's stated rule is that no substantially similar paper may be under review elsewhere during its review window; the same holds in spirit for RA-L, whose own prior-publication and preprint wording could not be retrieved (*assumption to verify*: arXiv preprints are tolerated, a paper accepted at RA-L counts as prior publication that a later paper must cite and exceed, and each paper may be presented at only one conference). So the two designs must be sequential and differ in claim, not only in length. Design (b) owns C1, C2 and the law with calibration on exact depth plus one real-data table. Design (a), for CVPR 2028, owns what (b) lacks: the gated map and its comparison with schedules, UseGeo and several blocks, entity-unit evidence, nuisance ladders, and (if the follow-up succeeds) the pixel-blind channels and heads. The shared Stage A and B data are cited in (a) as published, with new scenes and new conditions, never as a second presentation of the same table. Posting a preprint of (b) is allowed by CVPR as prior work provided it is cited; whether to post before RA-L reviews end is an RA-L question to verify.

## 3. Figure and table plan

Sizes: *1c* is one column, *2c* is full text width.

### 3.1 CVPR design: seven figures, five tables

| ID | Size | Shows | Data needed | Single message |
| --- | --- | --- | --- | --- |
| Fig. 1 teaser | 2c | See layout below | Block 3 crop with a depth failure; stages A and C outputs | A depth-only map is confidently wrong; the per-source record says where and by how much |
| Fig. 2 overview | 2c | Pipeline, left to right: frames and depth maps; depth-placed surfels; two records per entity; tension and score test; fused map with information field | None (diagram) | Both sources live on the same three coordinates; evidence enters by rendering |
| Fig. 3 law | 2c, two panels | Left: measured variance ratio of depth-only fusion against number of fused frames $n$, with nominal $1/n$ and predicted $\varrho + (1-\varrho)/n$; right: $\varrho$ per surface class | Block 3, real monocular depth (Depth Pro, others as supplement), exact depth | Nominal information grows without bound; the real error plateaus where theory says |
| Fig. 4 honesty | 2c, two panels | Left: stated against realised information per surface class (injection); right: recovery of injected displacement against size (capture range) per image scale | Injection runs on depth-placed surface, [5] seeds | Pixel information is real only within a few pixels of parallax; the honesty factor corrects it |
| Fig. 5 detection | 1c | ROC (log false-alarm axis) of score test, consensus gate, photometric mask, virtual-stereo mask, per fault size | Injected repeated faults of [3] sizes on block 3 | Repeated faults are invisible to consensus and visible to the score test |
| Fig. 6 qualitative | 2c | Rows: depth-only, decaying weight, information fusion, information fusion with gate; columns: block 3 land, block 3 water, one real scene; error map with stated-sigma contour | Fitted maps | Gains concentrate where the prior is wrong and the pixels are stiff enough |
| Fig. 7 nuisance | 1c | Calibration (coverage) and stated information against injected pose noise and per-frame scale error | Ladders, [5] levels each | Calibration stays flat while stated information falls |
| Table 1 | 2c | Six maps (pixels only, depth only, fixed weight, decaying weight, information fusion, with gate): log-score, robust scale of standardised error, coverage 68% and 95%, accuracy, completeness, per class (land, water, all) | Stage C | Calibration and accuracy move together only for the full map |
| Table 2 | 1c | Detection at 5% false alarms by fault type and size: true-positive rate for six detectors | Fig. 5 data | Same message as Fig. 5 with numbers |
| Table 3 | 1c | Ablations: tiles against segmenter entities, saturated against nominal prior, with and without dispersion, with and without honesty factor, projective against fitted appearance | Stage B and C reruns | Each calibration step is needed |
| Table 4 | 2c | Real scenes with laser truth (ETH3D [13] scenes; UseGeo [N] tiles): overconfidence ratio, coverage, detection of natural failures | Cached depth, UseGeo once downloaded | The synthetic findings reappear on real imagery, or do not |
| Table 5 | 1c | Cost: seconds and memory per frame and per hectare | Timing | The records are cheap to accumulate |

**Fig. 1 layout (exact).** Full width, two rows. Top row, three equal square panels sharing one ground footprint of about [120 by 80] m on block 3 where a monocular depth failure occurs (water or a roof; use an injected region if no natural failure is clear): (a) the frame with the failing region outlined; (b) depth-only map, signed placement error in metres on a diverging scale with the stated 2-sigma contour drawn as a thin line, annotated "stated [x] m, actual [y] m"; (c) our map, energy $E_m$ in nats as a sequential colour over the same footprint, flagged region outlined. Bottom row, two panels: left, wide, a transect profile through the failing region showing true surface (black), depth-only surface with its stated band (grey), and ours with its band (colour), the depth-only band narrow and wrong, ours wide where unknown; right, narrow, a reliability diagram of coverage at nominal 68% and 95% for the two maps.

### 3.2 RA-L design: four figures, three tables

Fig. 1 teaser at 1c with the top row stacked as (b) and (c) only and the transect profile below; Fig. 2 overview (2c, shortened); Fig. 3 law and honesty merged (2c: law panel, stated against realised information panel); Fig. 4 detection ROC (1c) with a qualitative map inset. Table 1 (merged six-map log-score and coverage, 1c, land/water/all only); Table 2 (detection at 5% false alarms); Table 3 (real scenes). Ablations go to a three-line paragraph.

## 4. Claims-to-evidence matrix

Sentences are those of the RA-L abstract draft (Section 5.1), in order.

| ID | Abstract sentence (short) | Evidence | If the experiment fails, the sentence becomes |
| --- | --- | --- | --- |
| S-a | Depth-only maps state variance [X] times too small | Fig. 3, Stage A; threshold: ratio above 1.5 (proposed) | "On [data], the depth-only map is already honest" and the paper is a negative result (stop S1) |
| S-b | The shortfall matches $\varrho + (1-\varrho)/n$ | Fig. 3 right panel; prediction within [30]% (proposed) | "Repeated error accounts for [Q]% of the shortfall" with the law as an upper bound |
| S-c | Image Jacobian factors through depth Jacobian | Unit test (C1) | Cannot fail for planar surfels; fails only if the renderer's weights are not as assumed, which is then a bug |
| S-d | Score test detects repeated faults at [Z]% against [W]% for consensus | Fig. 5, Table 2 | "Detects faults above [size] only" or "does not beat photometric masks" (stop S2/S5) |
| S-e | Calibration by injection gives coverage [C]% | Fig. 4, Table 1 | "Pixel information is too soft to judge; the map reports little is known" (limitation 1 of the plan) |
| S-f | Log-score improves from [S0] to [S] | Table 1 | "Improves calibration, not accuracy" or, with a decaying weight matching it, "no gain over schedule" (stop S5) |
| S-g | Real scenes [agree] | Table 3 / Table 4 | "Real scenes are mixed: [which behaviour]" |
| S-h (CVPR only) | Calibration holds under pose and scale error | Fig. 7 | "Calibration degrades beyond [level]" (stop S7 if it breaks at small levels) |
| S-i (CVPR only) | Tiles and segmenter entities differ by [Δ] | Table 3 | "Entity definition is not the source of the gain" |

**Fallback papers.**

1. *Detection of repeated faults works, mapping gains are small.* The paper is the audit: C1, C2, the law and the score test, with the mapping table reported honestly and a statement that a decaying weight matches fusion on log-score. Venue: RA-L as in design (b); title unchanged; abstract drops the log-score sentence.
2. *Overconfidence holds but image information is weak (S2).* The paper is "monocular depth maps are overconfident, here is the law and the saturated information that corrects it". It uses only the depth family and the injection study as a negative result on pixel information (capture range and dispersion), and the output of the map is stated abstention. Venue: RA-L or a workshop; the CVPR design is not viable.
3. *Only synthetic results are strong.* Claims are restricted to "exact-depth rendered aerial data"; real scenes appear as a failure analysis. Venue: a workshop or RA-L with a limitations-forward abstract; the contribution becomes the protocol (surface log-score, injection, repeated-fault injectors) released as a benchmark. CVPR is not viable without real support.

## 5. Abstracts and introduction

### 5.1 RA-L abstract (limit 1,200 characters)

> Maps built from posed aerial video and monocular depth state no confidence and can be confidently wrong over whole surfaces identically in every frame. We store on each planar Gaussian surfel patch a Fisher information record on its placement, an offset and two tilts, separately for the depth prior and for the pixels. Because the surfel is a plane, the image Jacobian factors through the depth Jacobian, so both records are quadratic forms on the same coordinates and their disagreement is a statistic with a null distribution. Other frames enter by rendering the shared surfels; no features are matched. On a synthetic aerial block with exact depth, a depth-only map states an error variance [X] times too small, close to the [Y] predicted from the share of depth error that all frames repeat. The pixel-against-depth test detects repeated faults at [Z]% true-positive rate at 5% false alarms where depth-consensus gating reaches [W]%. Calibrating pixel information by displacement injected through the renderer gives [C]% coverage at the nominal 95% and a surface log-score of [S] nats per entity against [S0] for depth only. Checks on [N] real scenes with laser truth [agree].

The text is 1,181 characters long including spaces (counted by script; keep it under 1,200 after any edit).

### 5.2 CVPR abstract (about 190 words)

> Gaussian splatting stores geometry and appearance but not how well either is known, and a map built from monocular depth can be confidently wrong over whole surfaces, identically in every frame. We add information, the stiffness with which each source of evidence holds a surface in place, to planar Gaussian surfels, kept separately for the depth prior and for the pixels. Because a surfel is a plane, the derivative of the rendered image with respect to its placement factors through the derivative of its depth: both sources are quadratic forms on one offset and two tilts, and their disagreement is a stored energy with a null distribution. Cross-view evidence enters by rendering shared surfels, with no matches or tracks. We show that a depth-only map is overconfident by [X] times, matching the law $\varrho + (1-\varrho)/n$ with $\varrho$ the share of error that frames repeat; that a score test at the depth-placed surface detects repeated faults at [Z]% against [W]% for consensus gating; and that calibrating pixel information by injection through the renderer gives coverage [C]% and improves the surface log-score from [S0] to [S] nats per entity. Gating on the energy gives [A] accuracy points on failure regions and changes clean scenes by [B]. Results on [N] real scenes with laser truth [agree].

### 5.3 Introduction, paragraph structure

1. *A map that looks complete says nothing about what it knows.* Aerial mapping from video and monocular depth; opacity is not confidence; a repeated error looks like agreement.
2. *Why depth consensus cannot help.* Frames share the error of the depth model on a surface; averaging reduces only the per-frame part; the law $\varrho + (1-\varrho)/n$.
3. *Pixels are the only independent witness, but they speak only through a coupling.* Stiffness is zero on blank surface and from one viewpoint; it can be computed only through the renderer.
4. *The two sources meet on one surfel.* Factorisation; both are quadratic forms on three coordinates; stretch, energy, null.
5. *Information must be honest.* Dispersion from frame replicates, honesty from injection, saturated prior; evidence enters by rendering.
6. *What we show.* Three contributions, with the headline numbers as placeholders.
7. *What we do not claim.* No better-than-matching claim; capture range is short; pixel-blind surfaces are reported as unknown.

## 6. Timeline, working backwards

**CVPR 2027 is closed** (registration 2026-11-10, submission 2026-11-16, supplement 2026-11-23, decisions 2027-02-25, from the official dates page). The remaining targets:

| Venue | Date | Status |
| --- | --- | --- |
| IROS 2027 (direct) | 2027-03-01 | Official page via tracker; not targeted |
| ICCV 2027 | registration 2027-03-01, submission 2027-03-08 | Projected from the previous cycle by a third-party page; official dates not yet posted *(verify)*; targeted only if milestone M3 is met by 2027-01-31 |
| RA-L with IROS 2027 option | submit by 2027-04-30; rolling review, decision within six months | Official RA-L page; **primary target, internal submission 2027-03-31** |
| CVPR 2028 | mid-November 2027 | Assumed *(verify)*; target for design (a) |

**Milestones** (each a go/no-go; thresholds proposed and fixed in the preregistration first).

| Date | Milestone | Go criterion | If no-go |
| --- | --- | --- | --- |
| 2026-12-04 | M0: Stage 0 on CPU and Stage A with real Depth Pro depth on block 3: measured variance ratio, $\varrho$ against exact depth | Ratio above 1.5 and $\varrho$ above 0.2 on land (stop S1) | Fallback paper 3 or negative-result note |
| 2026-12-23 | M1: Stage B injection on the depth-placed surface; capture range per image scale; stated against realised information | Injected displacement of the size the prior leaves is recovered on land; dispersion not so large that faults carry no energy (stop S2, S3) | Fallback paper 2 |
| 2027-01-15 | M2: ETH3D Stage A on [13] scenes with four depth models; UseGeo download approved and reader written | Repetition visible on at least [half] of scenes | Synthetic-only paper (fallback 3) |
| 2027-01-31 | M3: Stage C, six maps on block 3 with gsplat information blocks | Gate or information fusion beats decaying weight on log-score on failure regions (stop S5) | Fallback paper 1 |
| 2027-02-15 | Go/no-go for RA-L; ICCV is declined unless M3 and the Fig. 7 ladders exist | All of M0 to M3 met | Resize claim |
| 2027-03-07 | Table 1 to Table 3 frozen | | |
| 2027-03-21 | Real-scene table (Table 3, design b) | | |
| 2027-03-31 | Submission to RA-L | | |
| Apr to Oct 2027 | Extend for design (a): UseGeo blocks, ladders, tile additivity, optional channels | | |
| 2027-11 | CVPR 2028 submission | | |

Writing runs in parallel: introduction and method text by 2027-01-31, results sections as each milestone closes, figures final one week before submission. The single GPU is a risk for M3: the joint fit at image resolution must fit within roughly [8] hours per block; if it does not, the six-map comparison is run at reduced resolution and said so.

## 7. Naming and terminology

**Method name.** *Two-Spring Surfels* (TSS). The title above avoids "information-theoretic". Verified neighbours using that wording or close ideas: GauSS-MI (Shannon mutual information with splatting for active reconstruction, arXiv 2504.21067), FisherRF and POp-GS (Fisher information for view selection), PUP 3D-GS (Fisher blocks for pruning), Bayes' Rays (post-hoc uncertainty), VBGS-SLAM and VarSplat (posterior or variance per splat). A reviewer who reads "information-theoretic Gaussian splatting" will expect entropy, view selection or one posterior per Gaussian and will find none of it. The overlap risk is therefore in the name, not only in the idea; the safe wording is "Fisher information on placement, per source", defined on the first page. The comparison table must include at least one per-Gaussian Fisher field scored on the same log-score.

**Glossary (fixed).**

| Term | Meaning, used in no other sense |
| --- | --- |
| placement | $(\nu, \tau_1, \tau_2)$: offset along the normal and two tilts |
| source | one family of evidence: depth (prior) or image |
| information | Fisher information matrix $\Lambda$ on placement, one per source, in inverse squared metres and radians; never used for scalar confidence |
| stiffness | information along one direction; a scalar, used only in the spring analogy |
| pull | the vector $\boldsymbol\eta$: gradient of a source's loss at the current placement |
| entity | a ground tile of fixed size carrying one shared placement; not a segment |
| record | the stored triple (information, pull, dispersion) per entity and source |
| stretch | the vector $\mathbf t_m$, difference of the two sources' placements |
| tension | the scalar statistic $T_m$, the stretch in units of its joint uncertainty |
| energy | $E_m = T_m / 2$, in nats; no other use of the word |
| repetition coefficient | $\varrho$, share of depth error variance common to all frames |
| overconfidence ratio | measured over stated error variance of a map |
| absorbed, deflated, detected | the three fates of a nuisance: removed with a stated loss of information, reduced by a measured factor, or flagged |
| fault | a region where the depth prior is wrong beyond its stated uncertainty |
| surface log-score | negative log density of the true surface under the map, nats per entity |

"Depth prior" always means the monocular network output, never a sensor. "Honest" means calibrated against realised error.
