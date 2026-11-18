# Depth Prior and Pixels Meeting on the Same Gaussian Primitives: Objections, Prior Work, Verdict

Nothing in this note has been measured. Numerical statements are worked from stated assumptions or
quoted from the cited papers. Statements about prior work and software are marked *verified* (checked
against the paper, its arXiv page, the official repository, or the code of this repository) or
*recalled* (from memory, not checked).

## 0. The conception under examination

**Setting.** Posed aerial video in strips. One monocular depth map per frame from one frozen model. A
surface of 2D Gaussian surfels. A fault is a connected region whose monocular depth deviates from the
measurement model by many $\sigma$; the faults of interest are those that every frame repeats.

**Conception.**

1. Each primitive receives two families of messages about its placement
   $\boldsymbol\xi_k = (\delta_k, \boldsymbol\omega_k)$, the offset along its normal and its tilt:
   a *depth message* from every frame's monocular map, and an *image message* from every frame's
   pixels, obtained from the differentiable renderer as the gradient and Gauss-Newton curvature of
   the rendering loss with respect to $\boldsymbol\xi_k$.
2. The fault test is the *tension* between the two families at the joint fit, normalised by
   information. It is available per primitive or per surface entity, and it sees a fault that every
   frame repeats.
3. Segment masks and frozen features are lifted onto the primitives. An entity is a set of
   primitives. An injection is "move this set by a known offset and tilt and re-render".
4. The question for the paper: for each primitive, when should it listen to the depth prior and when
   to the pixels?

**The tension statistic.** Let the two families give, for one primitive or one entity, independent
estimates $\mathbf x_D \sim \mathcal N(\mathbf x, \Lambda_D^{-1})$ and
$\mathbf x_I \sim \mathcal N(\mathbf x, \Lambda_I^{-1})$ of the same placement. The fused estimate is
$\hat{\mathbf x} = (\Lambda_D + \Lambda_I)^{-1}(\Lambda_D \mathbf x_D + \Lambda_I \mathbf x_I)$. The
gradients of the two terms at $\hat{\mathbf x}$ are equal and opposite,

```math
\mathbf g_D = \Lambda_D(\hat{\mathbf x} - \mathbf x_D) = \Lambda_\parallel (\mathbf x_I - \mathbf x_D) = -\,\mathbf g_I ,
\qquad \Lambda_\parallel = \bigl(\Lambda_D^{-1} + \Lambda_I^{-1}\bigr)^{-1} ,
```

and the statistic can be written in either form,

```math
T = (\mathbf x_I - \mathbf x_D)^\top \Lambda_\parallel\, (\mathbf x_I - \mathbf x_D)
  = \mathbf g_D^\top \bigl(\Lambda_D^{-1} + \Lambda_I^{-1}\bigr)\, \mathbf g_D \ \sim\ \chi^2_{\dim \mathbf x} \ \text{ under no fault.}
```

Three facts follow and are used below.

- "Large opposite pulls" is not a finding; the pulls are equal and opposite at every joint optimum.
  The content is in their size relative to $\Lambda_\parallel$.
- A fault $\boldsymbol\Delta$ in the depth family has noncentrality
  $\boldsymbol\Delta^\top \Lambda_\parallel \boldsymbol\Delta$, bounded by the *weaker* family. It
  moves the fused surface by $(\Lambda_D + \Lambda_I)^{-1} \Lambda_D \boldsymbol\Delta$. Harm is large
  when the pixels are weak, and detectability is then small. This is the harm-detectability relation
  in the present terms.
- This is a difference-of-two-estimators test. It is the same object as solution separation in
  integrity monitoring, the Hausman test in econometrics, and parameter-shift tension statistics
  between data sets in cosmology (Part 4). The statistic is not new. Its two inputs are.

## Part 1. What exists in the code and in software

| Item | Status |
| --- | --- |
| `src/sensorfault/gs/fit.py` | *Verified.* Depth-only Adam fit. The rasteriser is called with constant colours. The default depth is a torch ray-plane compositor; the module's own note says that the rasteriser's RGB+D mode blends per-surfel centre depth and does not see tilt. |
| `view_information`, `render_cavity`, `gs/gate.py` | *Verified.* Closed-form depth Jacobians and lumped depth information blocks; a view-level gate with leave-out. No image term anywhere. |
| `experiments/rgb_train_2dgs.py` | *Verified.* A separate photometric 2DGS trainer: L1 plus 0.2 D-SSIM, normal-consistency and distortion regularisers, spherical harmonics, free opacities. No depth term. |
| `gsplat` in the present environment | *Verified:* not installed here (import fails), so the rasteriser's interface was not inspected. *Recalled:* it renders arbitrary per-primitive channels, which is what mask and feature splatting need; its backward pass gives gradients, not per-primitive Gauss-Newton blocks, and custom CUDA autograd functions do not support forward-mode products. |

To be built: a joint fit in which one set of primitives carries colour and receives both losses;
image information for placement (Part 2, R1, says in which form); mask and feature channels; entity
labels on primitives; the entity sweep. The two existing fits already give a first experiment with no
new code: fit the surface from depth alone and from pixels alone and compare (Part 5).

## Part 2. Objections

Each objection is given with the reason it could be fatal, a test, and the change that answers it.

### R1. Per-primitive image curvature does not vanish on blank surface

**Why it may be fatal.** The claim "the curvature vanishes on blank surface, so abstention is
automatic" holds for the exact Hessian over all primitives of the surface. It fails for a
per-primitive block. A single Gaussian of a uniformly coloured roof, moved along its normal, shifts
its footprint against its neighbours and changes the image at its edges, so its own block
$\Lambda^I_{kk} = \sum_u \mathbf J_{uk}^\top W \mathbf J_{uk}$ is positive. Only when all primitives
of the roof move together do the changes cancel, and the cancellation lives in the off-diagonal
blocks that a block-diagonal approximation deletes.

For the depth family, lumping majorises the Hessian and gives cautious steps. For the image family
the same approximation overstates what the pixels know about the surface, which makes the witness
overconfident and the test fire on blank surface.

There is a second coupling, with appearance. Placement and colour are estimated together. The
information about placement after colour is accounted for is the Schur complement
$\Lambda^I_{\xi\xi} - \Lambda^I_{\xi a} (\Lambda^I_{aa})^{-1} \Lambda^I_{a \xi}$, which is smaller
than $\Lambda^I_{\xi\xi}$.

**Test.** On a synthetic uniform plane and a textured plane: per-primitive block, entity-level
curvature, and the actual loss change under a common displacement. The first is expected to be
positive on the uniform plane and the other two near zero.

**Change.** Define the image message for a *collective* displacement. For an entity $m$ with three
placement parameters (offset and two tilts of its plane), the Jacobian is the directional derivative
$\mathbf J_{u,m} = \sum_{k \in m} \mathbf J_{uk}\, \partial \boldsymbol\xi_k / \partial \boldsymbol\theta_m$,
and $\Lambda^I_m = \sum_u \mathbf J_{u,m}^\top W \mathbf J_{u,m}$ with colour either marginalised or
fitted on other frames (R3). In practice this is a finite difference: move the entity, re-render,
difference the images. The image information is then the curvature of the injection sweep, and the
two are one computation. A per-primitive statistic remains possible only inside a textured entity,
as a second-order term.

### R2. A score test at the joint fit has no power against gross faults

**Why it may be fatal.** The depth term is quadratic in placement. The rendering loss is not: it has
a basin about as wide as the texture's correlation length in pixels of parallax, and it is flat or
erratic outside. A fault of many $\sigma$ displaces the surface by many pixels of parallax. If the
depth weight holds the primitive there, the image gradient is small or random, since the loss is
outside its basin, and the Gauss-Newton curvature evaluated there says nothing about where the
minimum is. The tension is then small for the largest faults. Small and moderate faults are detected;
gross ones are not, which is the opposite of what is wanted.

**Test.** The entity sweep on real aerial frames: rendering loss against offset, for several entities
and textures. Read off basin width in pixels of parallax and compare with the parallax of faults of
3, 10 and 30 $\sigma$.

**Change.** Use the difference-of-estimators form, not the score form. Obtain $\mathbf x_I$ by
profiling the rendering loss over the entity's placement on a coarse-to-fine grid (a sweep with three
unknowns), or from a pixels-only fit, and compare with $\mathbf x_D$. A feature-metric loss on
splatted frozen features widens the basin at the cost of resolution. The score form is kept as the
cheap monitor during fitting for small faults.

### R3. Appearance can explain away a misplaced surface

**Why it may be fatal.** With view-dependent colour, free opacity and densification, a surface at
the wrong depth can reproduce the images: colour that varies with direction imitates parallax of
low-frequency texture over the narrow range of directions in a strip, and semi-transparent primitives
or floaters carry the rest. The image family then agrees with the faulty depth. This is not a corner
case; it is how a radiance field normally absorbs geometric error.

**Test.** Place an entity at a known wrong offset, refit appearance only, and record the rendering
loss on fitted frames and on held-out frames, for diffuse colour and for spherical harmonics of
degree three.

**Change.** Two measures, both needed.

- *Restricted appearance for the test.* Diffuse colour only, opacity fixed, no densification, colour
  marginalised by the Schur complement or refitted. View-dependent residuals then appear as lack of
  fit, which is an ancillary statistic for specular surface.
- *Cross-fitting over frames.* Fit colour on one half of the frames and evaluate the placement
  profile on the other half. Colour cannot then adapt to the frames that judge. The price is half the
  baseline for the judging set.

The reconstruction that is delivered may use any appearance model; the restriction applies to the
test.

### R4. The null distribution is not $\chi^2$ without two corrections

**Why it may be fatal.**

- *Depth family.* $\Lambda_D = \sum_i \Lambda_{ik}$ assumes independent errors across frames. Ordinary
  monocular noise is correlated between neighbouring frames. With equal correlation $\varrho$ among
  $n$ frames the variance of the mean is inflated by $1 + (n-1)\varrho$, and with the nominal
  $\Lambda_D$ every well-observed primitive would be flagged. The repetition coefficient therefore
  enters the null, not only the alternative.
- *Image family.* Photometric residuals are neither white nor Gaussian. The weight $W$ has an
  unknown scale; blur, exposure and resampling add structured misfit.

**Test and change (one design).** Split-half placebo tensions. Split the frames in two interleaved
halves and compute four tensions per entity: depth half against depth half, image half against image
half, and depth against image in both pairings. The first two are null by construction for the
faults of interest (a repeated depth fault is common to both depth halves; a planar reflection is
common to both image halves). Their empirical distributions give the scale of each family's
covariance and the shape of the null, without labels. The cross tension is then referred to that
null. Injection of known displacements checks the image family's scale independently, since the loss
change under a known move must match $\tfrac12 \boldsymbol\theta^\top \Lambda^I \boldsymbol\theta$
inside the basin.

### R5. A decaying depth weight already answers "when to listen", and the test has power only where it does

**Why it may be fatal.** Common practice in depth-regularised splatting is a depth weight that decays
during training (recalled), so that the prior shapes the start and the pixels decide the end. Where
pixels are informative, this washes out a wrong prior with no test at all. Where pixels are silent,
the prior stays, and there the tension test abstains as well, since $\Lambda_I \to 0$ gives zero
noncentrality. On the reconstruction, the proposed method and the schedule would then differ nowhere.

**Test.** Surface error for: fixed depth weight; decaying depth weight; information-weighted fusion
without a gate; with the gate; with an oracle gate. Report separately where pixels are strong, weak
and silent.

**Change.** State where the method can differ from the schedule and show it there.

1. *Weak pixels, good prior.* A schedule discards a correct prior on weak texture, and floaters and
   depth noise return. Information weighting keeps it. This is the "listen to the depth" half of the
   question and is as important as the gate.
2. *Silent pixels, bounded entity.* A third message family from rendered segment masks (S2 in Part 3)
   carries outline parallax to blank surface, which no schedule uses.
3. *The product.* A calibrated map of where the prior was wrong, with an unsupported state, which a
   schedule does not give.

If the gain is confined to the third item, the paper is a diagnostic, not a reconstruction method.

### R6. Reflections become geometry behind the surface, and the test blames the depth prior

**Why it may be fatal.** A static scene seen in still water or flat glass is fitted by primitives at
the virtual depth, consistently across frames and across both image halves. If the monocular model
places the pane on its true surface, the two families are in strong, well-calibrated tension, and the
depth prior, which is right, is flagged. The image family has its own repeated fault.

**Test.** On scenes with water and glazing: sign and size of $\mathbf x_I - \mathbf x_D$ against
truth; share of flagged entities whose depth prior is correct.

**Change.** The layer test, now inside the renderer: an entity whose outline (from the mask family)
lies on one plane and whose colour-fitted interior lies behind it is a reflection or a view through
glass. On such entities the image family is discounted and the mask family judges.

### R7. Entities lifted through the geometry under judgment

**Why it may be fatal.** Segment masks reach the primitives by rendering. A primitive at the wrong
depth receives the labels of whatever it projects onto, consistently in the reference frame and
inconsistently elsewhere. Entity membership is then wrong exactly where the fault is, and segmenters
have no automatic video mode to supply an independent association.

**Test.** Label purity of primitives (agreement of the labels a primitive receives across frames) as
a function of injected displacement.

**Change.** Use label inconsistency across frames as evidence in its own right, which is what the
mask family does. Form entities from a first joint fit and refine once after the gate.

### R8. Cost

Exact per-pixel, per-primitive Jacobians are expensive. A Levenberg-Marquardt optimiser for splatting
needs a gradient cache averaging 53 GB and never forms $J^\top J$ (3DGS-LM, verified). Per-Gaussian
Fisher blocks have been computed by patch-wise backward passes at reduced resolution (PUP 3D-GS,
verified). The entity form of R1 avoids both: three placement parameters per entity, a few sweep
steps, and entities that do not overlap in the image perturbed in the same render. The cost is of
order (frames) times (colour classes of entities) times (steps), each a forward render. This is an
estimate, not a measurement.

### R9. What is still learned

With the tension computed, little remains for a head. The learned components that survive are: the
noise scales of both families per appearance stratum; a head on frozen features splatted onto
primitives that predicts the fault state on entities where no family but the prior speaks, trained on
the entities where the tension is decisive, with propensity weighting and an unsupported state; and
possibly a smooth gate. If the title promises a learned indicator, the abstract must say that the
evidence is computed and the learning extends it.

### R10. The area is crowded

Depth-regularised splatting, confidence-weighted depth losses, per-Gaussian Fisher information,
feature splatting and segment lifting all exist (Part 4). A reviewer will read the method as
"depth-regularised 2DGS with a per-Gaussian weight". The paper must make clear that the weight is the
outcome of a calibrated two-source test with a stated null and stated blind regions, and it must
compare against the nearest methods on their terms.

### Summary of the attack

| Objection | Fatal if | Cheapest test |
| --- | --- | --- |
| R1 blank surface | entity-level image curvature is not near zero on uniform surface, or only the per-primitive form is affordable | uniform against textured plane |
| R2 basin | faults of interest lie outside the basin and no wide-basin loss recovers them | entity sweep on real frames |
| R3 appearance | held-out frames do not penalise a misplaced entity under restricted appearance | wrong-offset refit |
| R4 null | placebo tensions are not stable across scenes | split-half tensions |
| R5 schedule | decaying depth weight matches the gated fusion in every regime | five-fit comparison |
| R6 reflections | most flags on water and glass are correct priors | sign test with exact depth |
| R7 lifting | label purity collapses under small displacement | purity curve |

## Part 3. What is sharp here

### S1. Two-source consistency on shared primitives, with a label-free null

The combination that was not found in prior work: an information-form depth message and an
information-form image message on the same placement parameters, compared by a difference test whose
null is calibrated by split-half placebo tensions and by injection. Each ingredient exists. Their
assembly into a test for a wrong prior, with a stated blind region
($\Lambda_\parallel \to 0$), was not found.

### S2. A third message family from rendered masks

Render the entity labels as channels and compare with the per-frame segment masks. The gradient of
this mask loss with respect to an entity's placement is outline parallax, and the rasteriser's
occlusion handling makes the nearer entity own a shared boundary without a separate ownership rule.
The family has information on bounded blank surface, where the image family has none, and it is not
deceived by a reflection inside the outline. With three families the pairwise tensions classify:

| Depth vs mask | Image vs mask | Reading |
| --- | --- | --- |
| consistent | consistent | healthy |
| tension | consistent | depth fault on a textured surface |
| tension | image silent | depth fault on a blank bounded surface |
| consistent | image behind mask | reflection or transparency; the prior is right |
| tension | image behind mask | reflection, and the prior followed the virtual scene or is otherwise wrong |
| mask silent | tension with depth | unbounded textured surface; image judges alone |
| mask silent | image silent | unsupported |

The mask family depends on a frozen segmenter returning the physical region with repeatable
boundaries, which is unverified for water and glass in nadir views and is the main risk.

### S3. The gate acts in the entity's placement subspace

The depth message of an entity is split into its component along the three collective placement
parameters and the remainder. The gate removes or down-weights the first and keeps the second, so a
flagged entity keeps the monocular relief and loses only its placement. In the joint fit this is a
projection of the depth gradient, which the renderer makes straightforward.

### What is not a contribution

- The tension statistic as such.
- Per-primitive image Fisher information as such.
- Lifting masks or features onto Gaussians.
- "Large opposite gradients" as a criterion; unnormalised gradient conflict is a multi-task
  optimisation notion and carries no error rate.

## Part 4. Prior work

### 4.1 Per-primitive information in splatting

| Work | What it does | Relation |
| --- | --- | --- |
| Hanson, Tu, Singla, Jayawardhana, Zwicker, Goldstein, "PUP 3D-GS: Principled Uncertainty Pruning", arXiv 2406.10219. *Verified.* | Per-Gaussian Fisher block of an L2 reconstruction loss with respect to mean and scale, block-diagonal, computed patch-wise; used to prune. | The image information block on position already exists. It is block-diagonal (R1) and is not compared with any other source. |
| Jiang, Lei, Daniilidis, "FisherRF". *Title verified; details recalled.* Wilson et al., "POp-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality", arXiv 2503.07819. *Verified.* | Diagonal or per-ellipsoid block information $J^\top J$ of rendered colour with respect to Gaussian parameters; used for view selection. | Same quantity, used for acquisition. POp-GS reports that geometric parameters dominate the information. |
| Höllein, Božič, Zollhöfer, Nießner, "3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt", ICCV 2025. *Verified.* | Gauss-Newton normal equations for splatting solved by preconditioned conjugate gradients with a per-pixel, per-splat gradient cache. | Shows the cost of exact Jacobians (53 GB average cache) and that $J^\top J$ products are feasible only matrix-free. |
| Goli et al., "Bayes' Rays", CVPR 2024. *Verified (title, venue).* | Post-hoc Laplace uncertainty of a radiance field on a spatial perturbation field. | Collective spatial perturbations, close in spirit to the entity displacement of R1; no second source. |

### 4.2 Depth priors in splatting and their reliability

| Work | What it does | Relation |
| --- | --- | --- |
| Turkulainen et al., "DN-Splatter", WACV 2025. *Verified.* | Depth and normal priors with a depth loss adapted by image gradients. | Fixed heuristic weighting; no test of the prior. |
| Sun et al., "Uncertainty-guided Optimal Transport in Depth Supervised Sparse-View 3D Gaussian", arXiv 2405.19657. *Verified (abstract).* | Depth priors with uncertainty and a patch-wise transport loss. | Uncertainty comes with the prior, not from the images. |
| Zhang et al., CDGS, arXiv 2502.14684; Xiao et al., "In Depth We Trust", arXiv 2604.05715; Chu et al., arXiv 2607.02554; Dufera and Cai, ConfidentSplat, arXiv 2509.16863. *Verified (arXiv pages).* | Confidence maps from fixed cues; depth applied only where rendered geometry is self-inconsistent; photometric masks from an RGB-only model; multi-view consistency weights. | The nearest neighbours in purpose. All are per-pixel masks or weights from one cue, with no information-form comparison and no error rate. The RGB-only-model mask is the closest to "pixels judge depth" and is the first baseline. |
| Song et al., "EnerGS: Energy-Based Gaussian Splatting with Partial Geometric Priors", arXiv 2604.26238. *Verified.* | LiDAR prior as an energy; positions move only under the geometric gradient, appearance only under the photometric loss. | Decouples the two gradients instead of comparing them; defines no conflict measure and does not detect wrong priors. |
| Decaying depth-regularisation weight in standard splatting pipelines. *Recalled.* | Prior early, pixels late. | The baseline of R5. |
| Pataki et al., MP-SfM, arXiv 2504.20040. *Verified (abstract).* | Monocular priors with uncertainty propagation inside structure-from-motion. | Same pairing of prior and multi-view evidence at the level of points; its uncertainty model was not read. |

### 4.3 Gradient conflict

Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020 (title and venue verified),
projects conflicting task gradients. A search for gradient conflict between depth and photometric
losses in splatting returned one item that could not be opened ("Depth-Gradient-Guided Decoupled
Optimization in 3D Gaussian Splatting for Sparse-View Reconstruction", title only) and optimiser
papers that do not concern priors (Ding et al., "A Step to Decouple Optimization in 3DGS",
arXiv 2601.16736, verified). No work was found that normalises the conflict by information and
reads it as a test. The unopened item is the most likely overlap and must be read.

### 4.4 Lifting masks and features onto primitives

Ye et al., "Gaussian Grouping", ECCV 2024 (verified): identity encodings on Gaussians supervised by
rendered masks. Kobayashi, Matsumoto, Sitzmann, "Decomposing NeRF for Editing via Feature Field
Distillation", NeurIPS 2022 (verified); Zhou et al., "Feature 3DGS", arXiv 2312.03203 (title
verified): frozen features rendered from the field. Kulhanek et al., WildGaussians (verified):
uncertainty from frozen-feature similarity of render and image. These supply the mechanisms of
conception point 3. None uses the mask loss's gradient with respect to placement as geometric
evidence, as far as was checked.

### 4.5 Reflections

Mirror-aware splatting methods (for example "Mirror-3DGS", arXiv 2404.01168, title only) model a
mirror plane and a mirrored scene, usually with mirror masks as input (recalled). They confirm that
an unconstrained fit places reflected content behind the surface. None was checked for a label-free
detection rule.

### 4.6 The statistic

Solution separation in integrity monitoring (Blanch et al., ION GNSS 2012 and IEEE TAES 2015; titles
and venues verified) and data snooping (Baarda 1968, recalled) compare a full solution with subset
solutions. The Hausman specification test (1978, recalled) compares two estimators of the same
parameter, one efficient under the null. Tension statistics between data sets in cosmology use the
parameter shift normalised by the summed covariances (surveys seen by title: "Concordance and
Discordance in Cosmology", arXiv 1806.04649; "Tensions in cosmology: a discussion of statistical
tools to determine inconsistencies", arXiv 2312.08542). Prior-data conflict checks in Bayesian
statistics (Evans and Moshonov, 2006, recalled) are the same idea with a prior as one source. The
paper should cite these and claim the application, not the test.

### 4.7 Summary of the gap

Not found: (a) an image message for collective placement compared in information form with a depth
prior on the same primitives; (b) a null for that comparison calibrated by split-half placebo
tensions; (c) a rendered-mask family used as geometric evidence with pairwise tensions that separate
depth faults from reflections; (d) a gate confined to an entity's placement subspace. The search
covered arXiv listings and proceedings pages through a general search engine and was not exhaustive.
One possibly overlapping title could not be opened, and several neighbours were read as abstracts
only.

## Part 5. Verdict

**Does the primitive have a real job?** Yes, under one condition. It is the common parameter on which
two or three independent families of evidence are expressed, with visibility and occlusion handled
once. That job exists only for collective placement of an entity (R1) and only with appearance
restricted or cross-fitted for the test (R3). A per-primitive tension from block-diagonal curvature
with free appearance would be wrong in both directions.

**Is it a contribution at a top venue?** The assembly of S1 to S3 is coherent, has a clear
statistical statement, and was not found in prior work. Its weak point is R5: on reconstruction
quality alone, a decaying depth weight may be hard to beat, and the test is silent where the prior
matters most unless the mask family works. The paper is strong if it shows (i) a calibrated fault
map with honest blind regions, (ii) retained good prior on weak texture, and (iii) correct handling
of reflections through the mask family. It is weak if it reports only that gated fusion beats a fixed
depth weight.

**The single claim.** *On shared Gaussian primitives, a monocular depth prior and the images are two
sources about the same placement; their tension, computed for surface entities and calibrated without
labels, tells where the prior is wrong, where the pixels are deceived, and where neither can say.*

**Cheapest tests, in order.**

1. *With the two existing fits.* Fit the surface from depth alone and from pixels alone on one
   block; compare rendered depths; with exact depth, check whether the difference localises true
   monocular faults and what it does on water and glazing. This needs no new code.
2. *Entity sweep on a few entities.* Basin width, curvature against texture, and the wrong-offset
   refit under diffuse and view-dependent appearance, on fitted and held-out frames (R2, R3).
3. *Uniform against textured plane*, per-primitive against entity curvature (R1).
4. *Split-half placebo tensions* on a scene without known faults (R4).
5. *Five-fit comparison* against the decaying schedule and the oracle gate (R5).

**Kill criteria.** Abandon the formulation if any of the following holds:

- under restricted, cross-fitted appearance, held-out frames do not penalise a misplaced entity on
  ordinary textured surface;
- the basin is narrower than the parallax of the faults of interest and neither a coarse-to-fine
  sweep nor a feature-metric loss recovers them;
- placebo tensions vary so much between scenes that no null can be fixed;
- a decaying depth weight matches the gated fusion on every regime and the oracle gate adds little.

Drop the mask family alone, keeping the rest, if a frozen segmenter does not return repeatable
physical regions on nadir frames; the method then abstains on blank surface and cannot separate
reflections from depth faults.

**Parts to drop now.**

- Per-primitive tension from block-diagonal image curvature.
- The score form as the main test.
- Free appearance in the judging step.
- "Opposite gradients" as the description of the criterion.

**What stays unknown.** Whether the image family, once appearance is restricted and colour
cross-fitted, still carries enough information about entity placement at the frame spacing and
texture of real aerial strips to give the test useful power, and whether the mask family works on
water and glass. Both are settled by the first three tests.

## Sources consulted

- Repository code: `src/sensorfault/gs/fit.py`, `src/sensorfault/gs/gate.py`, `experiments/rgb_train_2dgs.py`
- https://arxiv.org/html/2406.10219v3
- https://arxiv.org/html/2503.07819v2
- https://arxiv.org/html/2409.12892v2 and https://mlanthology.org/iccv/2025/hollein2025iccv-3dgslm
- https://jiangwenpl.github.io/FisherRF
- https://openaccess.thecvf.com/content/CVPR2024/html/Goli_Bayes_Rays_Uncertainty_Quantification_for_Neural_Radiance_Fields_CVPR_2024_paper.html
- https://arxiv.org/abs/2403.17822 and https://openaccess.thecvf.com/content/WACV2025/papers/Turkulainen_DN-Splatter_Depth_and_Normal_Priors_for_Gaussian_Splatting_and_Meshing_WACV_2025_paper.pdf
- https://arxiv.org/abs/2405.19657
- https://arxiv.org/abs/2502.14684, https://arxiv.org/abs/2604.05715, https://arxiv.org/abs/2607.02554, https://arxiv.org/abs/2509.16863v1
- https://arxiv.org/html/2604.26238
- https://arxiv.org/abs/2504.20040
- https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf
- https://arxiv.org/abs/2601.16736
- https://www.researchgate.net/publication/404047818_Depth-Gradient-Guided_Decoupled_Optimization_in_3D_Gaussian_Splatting_for_Sparse-View_Reconstruction (listing only)
- https://arxiv.org/abs/2312.00732, https://mlanthology.org/neurips/2022/kobayashi2022neurips-decomposing, https://arxiv.org/abs/2312.03203, https://arxiv.org/abs/2407.08447v2
- https://arxiv.org/abs/2404.01168v2 (listing only)
- https://arxiv.org/abs/2403.17888
- https://arxiv.org/pdf/1806.04649 and https://arxiv.org/pdf/2312.08542 (listings only)
- https://web.stanford.edu/group/scpnt/gpslab/pubs/papers/Blanch_et_al_IONGNSS_2012_v2.pdf and https://dblp.uni-trier.de/rec/journals/taes/BlanchWELPRSK15.html
