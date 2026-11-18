# Label-Free Combination of Fault Cues on (View, Surfel) Pairs: Objections, Prior Work, Verdict

## 0. The formulation under examination

**Setting.** Posed aerial images, one monocular depth map $\hat d_i$ per view, a surface of 2D Gaussian surfels. View $i$ sends surfel $k$ an information-form message $(\Lambda_{ik}, \eta_{ik})$; gated fusion is $\Lambda_k(w) = \Lambda_0 + \sum_i w_{ik} \Lambda_{ik}$. A fault is a connected region of a view whose depth deviates from the nominal model by many $\sigma$.

**Formulation B.** For each pair $(i,k)$ there is a latent fault indicator $f_{ik} \in \{0,1\}$ with prevalence $\pi = \Pr(f_{ik}=1)$, and $M$ cheap cues $x^m_{ik}$, binarised to $q^m_{ik} \in \{0,1\}$ where a discrete model is used:

| Cue | Signal | Information family |
| --- | --- | --- |
| c1 | leave-block-out standardised depth residual $z$ | cross-view geometry |
| c2 | disagreement between depth models on one image, after per-view alignment | single image, several networks |
| c3 | instability of one depth model under flip / crop / rescale | single image, one network |
| c4 | photometric evidence: photo-consistency, or the influence of a depth message on held-out RGB loss | cross-view colour |

The cues are treated as noisy detectors. Their accuracies are estimated from agreement patterns alone, under conditional independence given $f$ (a latent class / Dawid-Skene model, identifiable for $M \ge 3$). The posterior $\zeta_{ik} = \Pr(f_{ik}=1 \mid q_{ik})$ is the gate, $w_{ik} = 1 - \zeta_{ik}$, and is distilled into a light head on frozen backbone features. The learning is self-, semi- or unsupervised; the backbone is never fine-tuned; the number of labels is a variable (0, few, many).

Two facts about the latent class model are used throughout. Write $\mathrm{TPR}_m = \Pr(q^m=1 \mid f=1)$, $\mathrm{FPR}_m = \Pr(q^m=1 \mid f=0)$ and $\delta_m = \mathrm{TPR}_m - \mathrm{FPR}_m$ (Youden's index).

*Rank-one covariance.* Under conditional independence, for $a \ne b$,

```math
\operatorname{Cov}(q^a, q^b) = \pi (1-\pi)\, \delta_a \delta_b ,
```

so the off-diagonal covariance matrix has rank one and

```math
\delta_a^2 = \frac{1}{\pi(1-\pi)}\, \frac{\operatorname{Cov}(q^a,q^b)\operatorname{Cov}(q^a,q^c)}{\operatorname{Cov}(q^b,q^c)} .
```

Second moments give $\delta_m$ only up to the factor $\sqrt{\pi(1-\pi)}$ and a global sign; $\pi$ needs third moments or outside knowledge.

*Optimal rule.* Under conditional independence the Bayes rule is a weighted vote,

```math
\operatorname{logit} \zeta = \operatorname{logit} \pi + \sum_m \Bigl[ q^m \log \tfrac{\mathrm{TPR}_m}{\mathrm{FPR}_m} + (1-q^m) \log \tfrac{1-\mathrm{TPR}_m}{1-\mathrm{FPR}_m} \Bigr] .
```

## Part 1. Objections

Each objection is given with the reason it could be fatal, a test, and the change that answers it.

### O1. Conditional independence is false by construction

**Why it may be fatal.** The cues are not caused by "depth is wrong here". They are caused by the surface: glass, water and texture-free areas. The causal graph is surface type $s$ $\to$ each cue, and separately $s \to f$. Conditioning on $f$ does not block the path through $s$:

- c4 fails on specular, transparent and texture-free surfaces whether or not the depth is right. Photo-consistency has no minimum on a uniform roof; reflections on water move with the camera.
- c2 and c3 measure the *ambiguity* of the single-image problem. A depth model can be unstable on water and still land near the right plane, or stable and wrong.
- c1 raises false alarms at depth discontinuities and thin structures, where the surfel model is wrong. The same pixels are where depth models disagree with each other (edge bleeding) and where photo-consistency breaks (occlusion).

So the cues' false alarms coincide inside the clean class, and their misses coincide inside the faulty class. The second coincidence is the worse one. Where every view repeats an error, c1 is silent. The four depth models share training data and, to my recollection (not verified here), DINOv2-initialised ViT encoders, so on a shared failure c2 is silent too, and c3 is one of the same networks.

The dependence need not be large to matter, because faults are rare. Let two cues have $\mathrm{FPR} = 0.05$ and clean-class correlation $\rho_0$ between their false alarms. The spurious covariance is about $(1-\pi)\, \rho_0 \cdot 0.0475$. With $\pi = 0.02$ and $\delta_a = \delta_b = 0.5$ the entire signal is $\pi(1-\pi)\delta_a\delta_b = 0.0049$. A clean-class correlation of $\rho_0 = 0.1$ contributes $0.0047$: as much as the signal. The label model then fits a latent class that is not "fault".

**What the model recovers instead.** A two-class latent model returns the binary variable that best explains co-activation. The likely answer is "hard surface present" or "depth edge", not "depth wrong". The gate would then remove correct depth on every water body and roof edge, which is where a depth prior is needed most.

**Test.** On a scene with exact depth and per-pixel material or semantic class, tabulate the cue correlation matrix conditional on $(f, s)$ and conditional on $f$ alone, separately for the clean and faulty class. Report the residual of the rank-one fit to the covariance matrix. Then compare the fitted latent class with (a) the true fault mask and (b) the surface-type mask; if agreement with (b) exceeds agreement with (a), the formulation in this form is dead.

**Change that answers it.** Three options, in order of preference.

1. Condition on the confounder. Fit accuracies per stratum of a frozen-feature cluster $s$ (k-means on backbone patch features), so independence is only required given $(f, s)$. This costs samples per stratum (see O4).
2. Model the dependence. Group cues by family (c2 and c3 as one block, c1 and c4 as cross-view blocks) and allow within-block dependence; identifiability then needs three *blocks* that are independent, which the present cue set barely provides.
3. Drop the generative claim. Use worst-case aggregation that makes no independence assumption but needs bounds on each cue's error from a few labels (see O8 and Part 2.2).

### O2. The latent class has no anchor to the definition of a fault

**Why it may be fatal.** A fault is defined metrically: aligned error above $3\sigma$ in a region of at least 2% of the view. Only c1 measures depth error in units of $\sigma$. c2 measures spread between models, c3 spread under transformations, c4 colour. Nothing in an agreement-based fit ties the latent class to the $3\sigma$ line. The class is also subject to label swap, which is usually resolved by "the rare class is the fault", and that fails as soon as the cues fire on a common nuisance that is rarer or commoner than faults.

**Test.** Calibration of $\zeta$ against true fault frequency, and the realised error magnitude distribution inside the flagged set: is it concentrated above $3\sigma$?

**Change.** Anchor the class with a few labels (class-conditional rates of at least one cue), or make c1 the definitional cue and let the other cues enter only as a prior on where c1's threshold sits. The second option is the contamination model with an appearance prior, which is a different and simpler paper.

### O3. With three or four binary cues, the method selects one of eighteen rules

**Why it may be fatal.** The optimal rule under the model is a monotone weighted vote. On three binary cues there are exactly 18 non-constant monotone Boolean functions (8 up to relabelling: one cue, AND of two, OR of two, AND of three, OR of three, majority, $q^1 \wedge (q^2 \vee q^3)$, $q^1 \vee (q^2 \wedge q^3)$), and every one of them is a threshold function. All the estimation machinery does is choose among these 18. If the cues are of similar quality the answer is majority vote. If one cue dominates, its weight exceeds the sum of the others and the answer is that single cue. A reviewer will ask for the selected rule and will see one of these two.

**Test.** Report the fitted weights and the induced Boolean rule per scene and per depth model. Report how often it differs from majority and from the best single cue, and the change in average precision when it does.

**Change.** Keep the cues continuous. The model is then a mixture with conditionally independent continuous coordinates, which carries more information (a $7\sigma$ residual outweighs two mild votes). The price is that class-conditional densities must be estimated, and the dependence problem of O1 grows, since continuous scores are correlated through the scale of the scene as well. Alternatively accept the finding: if the rule is majority vote, say so and do not present a label model.

### O4. Faults are rare and regional, so the effective sample is small

**Why it may be fatal.** There are millions of $(i,k)$ pairs but they are not independent. A fault is one physical object seen in a run of frames. The independent unit is the object (or the object-strip pair), and a scene has perhaps tens of them. The estimator of $\delta_a$ is a ratio of covariances. With $\mathrm{FPR} = 0.05$ the standard error of a sample covariance is about $0.0475/\sqrt{n_{\mathrm{eff}}}$ against a signal of $0.005$: signal-to-noise one at $n_{\mathrm{eff}} \approx 90$, and 10% relative error at $n_{\mathrm{eff}} \approx 9000$. When a denominator covariance is near zero (a weak cue, such as c3 on a stable model), the triplet estimate is unbounded or imaginary. Per-scene fitting is then not credible, and stratifying by surface type (the remedy for O1) divides the sample further.

**Test.** Block bootstrap over spatial regions or strips, not over pairs. Report the interval of each $\delta_m$ and of $\pi$, and how often the triplet solution is infeasible. Simulate with the measured prevalence and region count.

**Change.** Pool accuracies across scenes (amortised regime), with one parameter set per depth model and surface stratum. Put a prior on $\pi$. Use a few labels to fix the accuracies and leave only $\pi$ per scene. Each of these moves the method away from "fitted on the scene under test without labels".

### O5. Abstention is not at random on surfels

**Why it may be fatal.** c1 exists only where views outside the block observe the surfel. A fault can spawn surfels supported only by faulty views; these are dropped from the cavity, so c1 is missing exactly where the fault is self-consistent. c4 as an influence needs $\Lambda_k^{-1}$, which is ill-conditioned for the same surfels. Missingness therefore depends on $f$. Label models that treat abstention as uninformative are biased, and the pairs that matter most have one or two cues, below the identifiability minimum of three.

**Test.** Fault rate among pairs by number of available cues. Detection rate on the stratum "fault repeated by all views in a block", which is the stratum the formulation was introduced to reach.

**Change.** Treat abstention as a third cue value with its own class-conditional rate. Add a ray-based cue that does not need a matching surfel: free-space violation, where the surfel of view $i$ lies in front of surfaces that other views see through. Without such a cue, B does not reach the see-through case.

### O6. "Combine several detectors" is an ensemble, not a contribution

**Why it may be fatal.** The parts are known: latent class models (1979), data programming (2016), triplet estimators (2020), distractor masks from frozen features supervised by residuals (2023 to 2024), hand-weighted multi-cue depth confidence in splatting (2025), cross-view consistency of monocular depth (many). A 3D vision reviewer will describe the paper as "Snorkel on four heuristics". Hand-weighted cue sums already exist for depth-regularised splatting (CDGS, Part 2.4); learned accuracy weights in place of hand weights is an increment unless it changes a result that matters.

**Test.** Not empirical; it is a framing question. The empirical counterpart is O7.

**Change.** The paper must claim a property, not a pipeline. The one property with content is this: *agreement between information families detects faults that every view repeats, which no consensus-based gate can detect*. That is a falsifiable claim about a stratum where the present gate has recall zero by construction. The label model is then the means, and may be replaced by something simpler without loss to the paper.

### O7. Simple baselines may match it

**Why it may be fatal.** The following cost nothing to implement and each has a real chance:

- mean, or maximum, of the standardised cues;
- rank average of the cues (label-free, scale-free, robust to one bad cue);
- the best single cue, in particular c1 alone and c2 alone;
- the depth model's own confidence output where it has one (UniDepthV2 and Metric3D v2 expose one, to my recollection; Depth Pro and Depth Anything V2 do not);
- a robust depth loss (Huber, Geman-McClure) with no gate at all;
- a linear probe on frozen features with a few labels (O9).

Under conditional independence with similar accuracies the weighted vote *is* the mean, so equality with the mean is the expected outcome, not an accident.

**Test.** Before any label model is built, compute the ceiling: a logistic regression on the cue vector trained *with* the true labels, cross-validated by scene. This is an upper bound on every combination rule that B could produce. If the ceiling does not beat the best single cue and the rank average by a clear margin (say five points of average precision, on the repeated-fault stratum in particular), no label-free method can, and B should be dropped at that point.

**Change.** If the ceiling is high but the mean matches it, publish the mean and the analysis. If the ceiling is high and only a fitted combination reaches it, B has a result.

### O8. The label-free constraint does little work once a few labels are allowed

**Why it may be fatal.** Synthetic scenes with exact depth exist, and a small labelled set is in scope. Selecting among 18 rules, or fitting four logistic weights, needs on the order of tens of independent labelled regions. A reviewer will ask why the accuracies were estimated from agreement patterns under an assumption known to be false, when one synthetic block gives them directly. The honest answers are narrow:

- cue accuracies differ between synthetic and real imagery (rendered glass and water are not real glass and water), so labelled accuracies do not transfer;
- prevalence $\pi$ varies by scene and has to be estimated on the target;
- a label-free fit can be run on the scene under test.

Each is an empirical claim about *transfer*, and each must be shown.

**Test.** A label-budget curve: detection quality on real scenes (laser depth) against the number of labelled synthetic regions, for (a) supervised weights on the cues, (b) agreement-fitted weights with labels used only to fix sign and $\pi$, (c) agreement-fitted weights with no labels. If (a) with twenty regions matches (c), the unsupervised label model is decoration.

**Change.** Recast B as semi-supervised from the start. A few labels do four specific jobs: they break the label swap, anchor the latent class to the $3\sigma$ definition (O2), measure the conditional dependence so it can be corrected (O1), and calibrate the threshold. The unlabelled target scene then adapts only $\pi$ and, if the data support it, a per-stratum shift. That is a defensible division of labour.

### O9. A linear probe on frozen features may already solve it

**Why it may be fatal.** If faults are a function of surface type, and surface type is linearly readable from frozen DINO patch features, then a logistic probe trained on one synthetic scene with exact fault masks predicts faults from the image alone, with no cues, no cavity and no label model. The distilled head of B is the same object trained on a noisier target. The cue machinery is then a lossy route to a classifier that labels give directly.

What the probe cannot do, in principle:

- it is image-only, so it predicts *propensity* ("this is water"), not the event ("this water region's depth is wrong in this view"). If the depth model is right on 60% of water, the probe's precision on water is capped at 40%;
- it is specific to one depth model, since its labels are that model's errors;
- it inherits the synthetic-to-real gap of its labels.

The cues are scene-specific evidence and can separate right from wrong within a surface type, but only if they carry information beyond surface type. That is the same question as O1 seen from the other side: cues that are independent given $f$ are useful and satisfy the model; cues that are functions of surface type violate the model and add nothing to the probe.

A probe on the frozen encoder of the depth model itself is a second variant and a stronger one, since those features encode what the model was uncertain about. It is close to a learned confidence head and should be treated as a baseline, not as a novelty.

**Test.** Average precision of (a) the probe on frozen image features with $n$ synthetic labelled regions, (b) the probe on the frozen depth encoder, (c) cues alone, (d) probe plus cues, measured *within* surface strata, on real scenes. B survives only if (d) exceeds (a) and (b) within strata.

**Change.** Make the probe the prior and the cues the likelihood: $\operatorname{logit}\zeta = \text{probe}(\text{features}) + \text{cue evidence}$, which is the contamination model with an appearance prior and a multi-cue likelihood. State the claim as the increment over the probe.

### O10. A frozen backbone trained on ground-level images may not transfer to nadir views

**Why it may be fatal.** The backbone was trained mostly on web photographs. Nadir aerial images have no horizon, no canonical "up", small objects and repetitive roofs. Two consequences. First, patch features may not separate water from asphalt from glass roofs at the ground sampling distance in use, and the head has nothing to read. Second, the monocular depth models are themselves out of distribution: a recent benchmark reports a significant domain gap for metric depth models on aerial viewpoints (AerialMetric, Part 2.7). Faults may then be common and diffuse, not rare and regional, which breaks the rarity assumption that resolves label swap and the "connected region" definition.

There is also correlated blindness. If the depth model and the head share a backbone family, the head is weakest exactly where the depth model fails for representational reasons.

**Test.** Linear-probe accuracy for material or land-cover class on nadir tiles from frozen features, as a function of ground sampling distance and patch size. Rotation test: aerial images have no preferred orientation, so head output should be equivariant to 90-degree rotations; measure the violation. Fraction of pixels above $3\sigma$ and the size distribution of connected error regions for each depth model on nadir views.

**Change.** Compare backbones, including one pretrained on overhead imagery (DINOv3 was released with a satellite-pretrained variant, to my recollection; not verified here). Average the head over rotations. If faults turn out diffuse, the problem is noise-level estimation, not fault detection, and B is the wrong tool.

### O11. Over-engineered relative to dropping bad views or using a robust loss

**Why it may be fatal.** With high overlap, a view is redundant. Hard removal of the worst views, or a Geman-McClure depth loss, may give the same surface. The gain of a pair-level gate over these is confined to two cases: low overlap, where clean parts of a faulty view are needed, and repeated faults, where neither robust loss nor view removal helps because the majority is wrong.

**Test.** Held-out depth error and F-score against overlap, for no gate, robust loss, view removal, pixel gate from c1 alone, B, and an oracle gate. If the oracle gate gains little over robust loss at the overlaps of interest, the whole line of work has no headroom and the detection metric is beside the point.

**Change.** Report the oracle headroom first. Restrict the claim to the regime where headroom exists.

### O12. Colour as a judge is not independent of the surface it judges

**Why it may be fatal.** The influence form $I(i \to j, k) = \nabla_k \mathcal L_j^{\mathrm{rgb}\,\top} \Lambda_k^{-1} \eta_{ik}$ uses the colour gradient at the current surface, which the suspect message helped place. It is a first-order quantity, valid for small moves, while faults are many $\sigma$. On texture-free surfaces the colour gradient is near zero, so the cue is uninformative where faults live; on specular surfaces it has the wrong sign. It also costs a backward pass per held-out view.

**Test.** Rank correlation of $I$ with the true change in held-out RGB loss under exact removal of the message, stratified by surface type.

**Change.** Use plain patch photo-consistency (normalised cross-correlation at the surfel, across views outside the block) with an explicit "no texture, abstain" outcome. Drop the influence form.

### O13. Distillation target and student share errors

**Why it may be fatal.** The student is trained on $\zeta$ and applied where cues are missing. If $\zeta$ tracks surface type (O1), the student learns "water is a fault" cleanly and confidently, and the error becomes invisible, since the student is smoother and looks better than its teacher. Feeding the student back would close a confirmation loop.

**Test.** Student against teacher against true masks, within surface strata. Student trained on $\zeta$ against a student trained on few true labels (the probe of O9).

**Change.** Keep the student out of the E-step. Train it with a loss that is robust to class-conditional label noise, using the estimated $\mathrm{TPR}$ and $\mathrm{FPR}$ of the teacher, and report it as a prior only.

### O14. Posterior probabilities from a misspecified independence model are overconfident

Naive-Bayes posteriors count correlated evidence twice. $\zeta$ will sit near 0 or 1. Any stated false-alarm rate must come from a separate calibration (conformal, with scene or strip as the unit), not from $\zeta$. This is not fatal, but the paper cannot present $\zeta$ as a probability.

### Summary of the attack

| Objection | Fatal if | Cheapest test |
| --- | --- | --- |
| O1 dependence | latent class matches surface type better than fault | conditional correlation tables given $(f,s)$ |
| O3 trivial rule | fitted rule is majority or one cue | print the rule |
| O4 variance | block-bootstrap intervals on $\delta_m$ span zero | block bootstrap |
| O5 abstention | repeated-fault stratum has fewer than three cues | cue availability by stratum |
| O7 baselines | supervised ceiling on cues is no better than best cue or rank average | logistic regression with true labels |
| O8, O9 labels and probe | probe with a few labels matches cues within strata | label-budget curve |
| O10 transfer | features do not separate materials at nadir | material probe, rotation test |
| O11 headroom | oracle gate gains little over robust loss | oracle run |

All eight tests need only cached depth maps, the cavity residual, exact depth for evaluation, and one frozen backbone. None needs the label model to exist.

## Part 2. Prior work

Verification status is given per entry. "Verified" means title, authors and venue or arXiv identifier were confirmed against a publisher, proceedings or arXiv page. "From memory" means the item is recalled and was not checked; its details may be wrong.

### 2.1 Label models and weak supervision

| Work | What it does | Relation to B |
| --- | --- | --- |
| Dawid and Skene, "Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm", Applied Statistics, 1979. *From memory.* | Latent class model of several annotators with unknown confusion matrices, fitted by EM. | B is this model with cues as annotators and $(i,k)$ pairs as items. No difference in the estimator. |
| Ratner, De Sa, Wu, Selsam, Ré, "Data Programming: Creating Large Training Sets, Quickly", NeurIPS 2016. *Verified (title, venue); author list from memory.* | Generative model over labelling functions, then a discriminative model trained on its soft labels. | B's "combine, then distil" is this two-stage recipe. The domain and the cue set differ, not the method. |
| Ratner, Bach, Ehrenberg, Fries, Wu, Ré, "Snorkel: Rapid Training Data Creation with Weak Supervision", VLDB 2017. *Verified.* | System built on data programming; models some dependencies between labelling functions. | Same. Its dependency modelling is the relevant part for O1. |
| Ratner, Hancock, Dunnmon, Sala, Pandey, Ré, "Training Complex Models with Multi-Task Weak Supervision", AAAI 2019. *From memory.* | Recovers accuracies from the inverse covariance of the sources given a known dependency graph. | B would need this, not plain Dawid-Skene, once cue dependence is admitted; the dependency graph must then be supplied. |
| Fu, Chen, Sala, Hooper, Fatahalian, Ré, "Fast and Three-rious: Speeding Up Weak Supervision with Triplet Methods", ICML 2020. *Verified.* | Closed-form accuracies from triplets of conditionally independent sources. | The triplet formula in Section 0 is theirs. Its failure when a pairwise moment is near zero is O4. |
| Zhang et al., "WRENCH: A Comprehensive Benchmark for Weak Supervision", NeurIPS 2021 Datasets and Benchmarks. *Verified (title, venue).* | Benchmarks label models and end models across tasks. | Its general finding, as I recall it (not verified), is that majority vote is hard to beat consistently. That is O3 and O7 as an empirical regularity. |

### 2.2 Unsupervised and semi-supervised ensemble learning

| Work | What it does | Relation to B |
| --- | --- | --- |
| Parisi, Strino, Nadler, Kluger, "Ranking and combining multiple predictors without labeled data", PNAS 2014. *Verified.* | Shows the off-diagonal covariance of conditionally independent binary classifiers is rank one, with leading eigenvector giving balanced accuracies; spectral meta-learner. | The rank-one identity of Section 0. B's accuracy estimation is this method. |
| Jaffe, Fetaya, Nadler, Jiang, Kluger, "Unsupervised Ensemble Learning with Dependent Classifiers", AISTATS 2016. *Verified.* | Detects and models groups of dependent classifiers through a second latent layer. | The principled answer to O1. It needs many classifiers to find groups; four cues are too few to estimate a group structure. |
| Platanios, Blum, Mitchell, "Estimating Accuracy from Unlabeled Data", UAI 2014. *Verified.* | Estimates classifier error rates from agreement rates, with and without independence. | Same aim as B's accuracy step; makes explicit that agreement identifies accuracy only under assumptions on error dependence. |
| Balsubramani and Freund, "Optimally Combining Classifiers Using Unlabeled Data", COLT 2015. *Verified.* | Minimax-optimal aggregation given bounds on each classifier's error, with no independence assumption. | The alternative to a generative label model: needs error bounds, which a few labels supply. Fits the semi-supervised scope. |
| Arachie and Huang, "Adversarial Label Learning", AAAI 2019. *Verified.* | Trains a model against the worst-case labelling consistent with error bounds on weak signals. | Same idea as above, with the end model in the loop. A candidate replacement for Dawid-Skene plus distillation. |
| Mazzetto, Sam, Park, Upfal, Bach, "Semi-Supervised Aggregation of Dependent Weak Supervision Sources With Performance Guarantees", AISTATS 2021. *Verified.* | Uses a small labelled set to bound source accuracies, then aggregates dependent sources with guarantees. | This is the semi-supervised version of B stated as a general method. B would be an application of it. |
| Anandkumar, Ge, Hsu, Kakade, Telgarsky, "Tensor Decompositions for Learning Latent Variable Models", JMLR 2014; Zhang, Chen, Zhou, Jordan, "Spectral Methods Meet EM", NeurIPS 2014; Hall and Zhou, nonparametric identifiability of mixtures with three conditionally independent coordinates, Annals of Statistics 2003. *All from memory.* | Identifiability and moment estimators for conditionally independent views, discrete and continuous. | The theory behind "three cues suffice", including the continuous-cue variant proposed under O3. |

### 2.3 Outlier and uncertainty masks in radiance fields and splatting

| Work | What it does | Relation to B |
| --- | --- | --- |
| Sabour et al., "RobustNeRF: Ignoring Distractors with Robust Losses". *Title verified; venue (CVPR 2023) from memory.* | Trimmed photometric loss with spatial smoothing of the inlier mask. | One cue (photometric residual against the fitted model), no depth, no cross-cue reasoning. A baseline. |
| Ren et al., "NeRF On-the-go: Exploiting Uncertainty for Distractor-free NeRFs in the Wild", CVPR 2024. *Verified.* | Per-pixel uncertainty predicted from frozen DINOv2 features (as I recall), supervised by the rendering residual. | Light head on frozen features, label-free, residual-supervised: the architecture of B's distilled head already exists for photometric outliers. |
| Sabour, Goli, Kopanas, Matthews, Lagun, Guibas, Jacobson, Fleet, Tagliasacchi, "SpotLessSplats: Ignoring Distractors in 3D Gaussian Splatting", arXiv 2406.20055. *Verified (arXiv); journal publication from memory.* | Clusters frozen Stable Diffusion features, or trains a small MLP on them, with photometric-residual quantiles as weak labels. | Weak labels from one cue into a head on frozen features. B differs by having several cues, a depth target, and a leave-out residual. The head is not new. |
| Kulhanek, Peng, Kukelova, Pollefeys, Sattler, "WildGaussians: 3D Gaussian Splatting in the Wild", arXiv 2407.08447. *Verified (arXiv); NeurIPS 2024 from memory.* | Uncertainty from DINO feature similarity between render and image, plus appearance embeddings. | Frozen-feature uncertainty for colour. No depth supervision, no cue combination. |
| Goli et al., "Bayes' Rays: Uncertainty Quantification for Neural Radiance Fields", CVPR 2024. *Verified.* Jiang, Lei, Daniilidis, "FisherRF". *Title verified; venue (ECCV 2024) from memory.* | Post-hoc Laplace-type spatial uncertainty; Fisher information for view selection. | Epistemic uncertainty of the fitted model, not detection of wrong inputs. Supplies $\Lambda_k$-type quantities that B uses. |

### 2.4 Reliability of monocular depth priors in splatting and neural surfaces (2024 to 2026)

| Work | What it does | Relation to B |
| --- | --- | --- |
| Zhang, Wysocki, Urban, Jutzi, "CDGS: Confidence-Aware Depth Regularization for 3D Gaussian Splatting", arXiv 2502.14684, 2025. *Verified (arXiv).* | Depth-loss weight from a hand-weighted sum of three cues (edge map, Laplacian texture, depth gradient). | The closest "multi-cue confidence" precedent. Weights are fixed by hand, cues are single-image heuristics, nothing is estimated. B's increment is learned weights and cross-view cues. |
| Xiao, Goan, Santa Cruz, Ahmedt-Aristizabal, Salvado, Fookes, Lebrat, "In Depth We Trust: Reliable Monocular Depth Supervision for Gaussian Splatting", arXiv 2604.05715, 2026. *Verified (arXiv).* | Applies absolute monocular depth only where rendered depth is self-inconsistent under a virtual stereo reprojection; gradient loss elsewhere. | Opposite policy: trust the prior where geometry is ill-posed. Addresses where the prior is *needed*, not where it is *wrong*. A direct baseline, and a reminder that gating the prior on water and glass removes it where nothing else constrains the surface. |
| Chu, Gopalan, Yuan, "Reliability-Aware Monocular Depth Supervision for Sparse-View Neural Reconstruction", arXiv 2607.02554, 2026. *Verified (arXiv).* | Depth supervision restricted by photometric masks from an RGB-only baseline model. | One photometric cue used as a reliability mask: B's c4 used alone. |
| Feng, Shan, Cheng, Wang, "Leveraging Geometric Prior Uncertainty and Complementary Constraints for High-Fidelity Neural Indoor Surface Reconstruction" (GPU-SDF), arXiv 2602.23926, 2026. *Verified (arXiv).* | Self-supervised uncertainty of geometric priors, uncertainty-weighted loss, extra multi-view constraints where uncertainty is high. | Self-supervised prior uncertainty with soft weighting already exists for SDF surfaces. Indoor, no label model. |
| Zhang, Yang, Huang, Han, Shi, Liu, Han, "MonoInstance: Enhancing Monocular Priors via Multi-view Instance Alignment for Neural Rendering and Reconstruction", arXiv 2503.18363, 2025. *Verified (arXiv).* | Aligns per-instance monocular depth across views; uncertainty is the density of the aligned point cloud. | Cross-view consensus of monocular depth as uncertainty: B's c1 at instance level. Blind to repeated errors, like every consensus cue. |
| Dufera and Cai, "ConfidentSplat: Confidence-Weighted Depth Fusion for Accurate 3D Gaussian Splatting SLAM", arXiv 2509.16863, 2025. *Verified (arXiv).* | Weights multi-view depth against monocular priors by multi-view geometric consistency. | Consensus-based confidence again. |

No work was found that fits a label model over heterogeneous fault cues for depth priors, or that reports detection of view-repeated depth errors as a separate stratum. The search was not exhaustive, and absence here is weak evidence.

### 2.5 Test-time-augmentation uncertainty for monocular depth

| Work | What it does | Relation to B |
| --- | --- | --- |
| Poggi, Aleotti, Tosi, Mattoccia, "On the Uncertainty of Self-Supervised Monocular Depth Estimation", CVPR 2020. *Verified (title, venue); co-authors from memory.* | Compares uncertainty estimators for monocular depth, including flip-based post-processing variance, ensembles and learned variants. | B's c3 is the flip and augmentation estimator; B's c2 is the ensemble estimator. Both are established as uncertainty measures, with known moderate correlation to error. |
| Landgraf, Qin, Ulrich, "A Critical Synthesis of Uncertainty Quantification and Foundation Models in Monocular Depth Estimation", arXiv 2501.08188, 2025. *Verified (arXiv).* | Evaluates five uncertainty methods on Depth Anything V2; finds fine-tuning with a Gaussian likelihood loss the most promising. | Says the strong single-image baseline is a learned variance head. That requires fine-tuning and is out of scope here, but a variance head on the frozen encoder is in scope and is the baseline of O9. |

### 2.6 Multi-view consistency checks for depth

| Work | What it does | Relation to B |
| --- | --- | --- |
| Geometric consistency filtering in learned multi-view stereo (forward-backward reprojection of per-view depth maps across at least a few views), as in Yao et al., "MVSNet", ECCV 2018. *From memory.* | Keeps a depth pixel only if it reprojects consistently into several other views. | B's c1 without the leave-out and without variance. The standard cheap baseline. |
| Luo et al., "Consistent Video Depth Estimation", SIGGRAPH 2020. *From memory.* | Test-time optimisation of a monocular depth network for geometric consistency across frames. | Repairs inconsistency by fine-tuning; out of scope, and it does not detect repeated errors. |
| Depth-regularised splatting and neural surfaces with scale-aligned monocular depth (MonoSDF, NeurIPS 2022; DNGaussian, CVPR 2024; DN-Splatter, WACV 2025). *All from memory.* | Use monocular depth and normals as losses, at most with global or patch-wise normalisation. | Treat the prior as uniformly reliable. Baselines for "no gate". |

### 2.7 Frozen foundation features: probes, anomaly heads, feature lifting, aerial transfer

| Work | What it does | Relation to B |
| --- | --- | --- |
| Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", arXiv 2304.07193. *Title verified.* Siméoni et al., "DINOv3", arXiv 2508.10104. *Title verified; satellite-pretrained variant from memory.* | Self-supervised ViT features that support dense tasks with linear heads. | The frozen backbone for the head. An overhead-pretrained variant is the control for O10. |
| Danier et al., "DepthCues: Evaluating Monocular Depth Perception in Large Vision Models", arXiv 2411.17385. *Title verified; authors and venue from memory.* | Probes frozen vision models for depth-cue understanding. | Evidence on what frozen features encode about depth; relevant to whether a probe can read depth *reliability*. No work was found that probes frozen features for depth-error prediction specifically. |
| Roth et al., "Towards Total Recall in Industrial Anomaly Detection" (PatchCore), CVPR 2022. *Verified.* Damm et al., "AnomalyDINO: Boosting Patch-Based Few-Shot Anomaly Detection with DINOv2", WACV 2025. *Verified.* | Anomaly maps from nearest-neighbour distances of frozen patch features to a memory bank of normal patches; no training. | A label-free, training-free head on frozen features. Applicable as a cue ("this patch is unlike patches where c1 says the depth is fine"), and a baseline for the distilled head. |
| Kobayashi, Matsumoto, Sitzmann, "Decomposing NeRF for Editing via Feature Field Distillation", NeurIPS 2022. *Verified.* Zhou et al., "Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields", arXiv 2312.03203. *Title verified; venue (CVPR 2024) from memory.* Kerr et al., "LERF", ICCV 2023; Qiu et al., "Feature Splatting", ECCV 2024. *Both from memory.* | Lift frozen 2D features into a 3D field or onto Gaussians by rendering-based distillation. | The mechanism to place frozen features on surfels, so the head can act on $(i,k)$ pairs. Caveat: lifting uses the geometry that the faulty depth helped build, so features on fault-spawned surfels are placed by the fault. |
| Song et al., "AerialMetric: Benchmarking and Adapting UAV Monocular Metric Depth Estimation in the Real World", arXiv 2606.29716, ECCV 2026. *Verified (arXiv page, venue from its comments field).* | Benchmark of metric depth models on aerial views; reports a significant domain gap and closes it by fine-tuning. | Supports O10: off-the-shelf depth is out of distribution on aerial views, so errors may be diffuse and prevalent. Also a source of real aerial depth for evaluation. |

### 2.8 Influence functions and data valuation in 3D reconstruction

| Work | What it does | Relation to B |
| --- | --- | --- |
| Jung, Yun, Park, Kim, Yang, "PruNeRF: Segment-Centric Dataset Pruning via 3D Spatial Consistency", arXiv 2406.00798, 2024. *Verified (arXiv).* | Scores pixel-level distraction with influence functions, checks 3D consistency by depth reprojection, refines to segments, prunes. | The nearest precedent for B's c4 in influence form, combined with a consistency cue and a segment prior. Targets photometric distractors, not depth priors; the combination is a pipeline, not an estimated model. |

No verified work was found that applies influence functions or data valuation to depth-prior messages in splatting. The influence form of c4 would be new, but O12 argues it is weak where it is needed.

### 2.9 Classical fault detection

| Work | What it does | Relation to B |
| --- | --- | --- |
| Baarda, "A Testing Procedure for Use in Geodetic Networks", Netherlands Geodetic Commission, 1968. *From memory.* | Data snooping: test each observation by its normalised residual; minimal detectable bias and reliability. | c1 is a w-test with a learned null. Baarda's reliability theory states that a bias in the direction shared by all redundant observations is undetectable: the formal version of "repeated faults are invisible to consensus". |
| Blanch, Walter, Enge et al., "Advanced RAIM User Algorithm Description: Integrity Support Message Processing, Fault Detection, Exclusion, and Protection Level Calculation", ION GNSS 2012; Blanch et al., "Baseline Advanced RAIM User Algorithm and Possible Improvements", IEEE Transactions on Aerospace and Electronic Systems, 2015. *Verified (title and venue; full author lists not checked).* | Solution separation: compare the all-in-view solution with subset solutions, one per fault hypothesis with a prior probability, including multi-satellite and constellation-wide faults. | Leave-block-out is solution separation with blocks as fault modes. The relevant lesson is the handling of *common-mode* (constellation) faults: they are covered only by a second independent constellation. B's cross-family agreement is the same remedy, and it works only to the extent that the families are independent. |

## Part 3. Verdict

**Is B publishable?** Not as stated. "Fit a Dawid-Skene model to four fault cues and distil it" has no new estimator, rests on an independence assumption that the physics of the problem contradicts, selects among at most a handful of voting rules, and is likely to be matched by a rank average of the cues or by a linear probe with a few synthetic labels. A robotics or 3D vision venue would reject it as an ensemble.

There is a publishable idea inside it, with one condition attached. The condition is empirical and cheap to check.

**The single claim.** *Depth-prior faults that every nearby view repeats are invisible to any consensus gate; evidence from an independent information family (single-image instability across models, and colour across views) detects them, and gating (view, surfel) messages on that evidence improves the reconstructed surface where a consensus gate and a robust loss do not.*

The claim is about a stratum of faults and a property of cue families. It does not depend on the label model. The combination rule should be the simplest one that reaches the supervised ceiling; if that is a rank average, the paper says so.

**Recommended form of the method.** Semi-supervised, three parts:

1. a prior from a logistic head on frozen features (image backbone, and separately the depth model's frozen encoder), trained on a few labelled synthetic regions and then on confident pseudo-labels of the target scene;
2. a likelihood from the cues, with accuracies anchored by the same few labels, stratified by frozen-feature cluster, and only the prevalence adapted on the unlabelled target scene;
3. a ray-based free-space cue for fault-spawned surfels, without which the pair-level gate cannot reach the see-through case.

**Minimum convincing experiment.** One synthetic aerial dataset with exact depth and material classes, at least two depth models, and one real dataset with laser depth for transfer.

1. *Stratified detection.* Split true faults into "isolated" (clean in most covering views) and "repeated" (wrong in at least, say, 80% of covering views). Report average precision per cue, per stratum. The claim requires c1 near chance on the repeated stratum and at least one other cue clearly above it.
2. *Ceiling.* Cross-validated logistic regression on the cue vector with true labels. Compare with best single cue, rank average, depth-model confidence. Proceed only if the ceiling beats all of them on the repeated stratum.
3. *Dependence.* Cue correlation matrices conditional on $(f, s)$; agreement of the fitted latent class with the fault mask against the surface-type mask.
4. *Label budget.* Probe on frozen features, cues alone, and probe plus cues, against the number of labelled regions (0, 5, 20, 100), evaluated within surface strata on the real dataset.
5. *Downstream.* Held-out depth error and F-score of the surface for: no gate, Geman-McClure loss, view removal, c1 gate, the proposed gate, oracle gate. Report on the repeated stratum and overall, with clean-scene degradation.

Steps 1 to 3 need no new method and decide whether the rest is worth building.

**Kill criteria.** Abandon B if any of the following holds: no non-consensus cue exceeds chance on the repeated stratum; the supervised ceiling on cues does not beat the rank average and the best single cue; the probe with twenty labelled regions matches probe plus cues within surface strata; the oracle gate gains little over a robust loss at the overlaps of interest.

**Variants to kill now.**

- *The fully unsupervised binary latent class model as the headline.* It is the least defensible part (O1 to O4), and a few labels remove the need for it. Keep it only as an ablation row, "zero labels".
- *The influence form of c4.* First-order, expensive, uninformative on texture-free surfaces and wrong-signed on specular ones (O12). Replace it with patch photo-consistency with explicit abstention.
- *Treating c2 and c3 as two independent cues.* They are one family. Count them once, or model them as a block.

**What stays unknown.** Whether any cue carries information about repeated faults beyond surface type. Every argument above reduces to that question. If the answer is no, the best available method is a frozen-feature probe that predicts fault propensity, and the multi-cue formulation has nothing to add.

## Sources consulted

- https://proceedings.mlr.press/r12/university14g.html and https://mlanthology.org/uai/2014/platanios2014uai-estimating
- https://arxiv.org/pdf/2002.11955.pdf and https://www.icml.cc/virtual/2020/poster/6736
- https://pubmed.ncbi.nlm.nih.gov/24474744/ and https://arxiv.org/pdf/1303.3257
- https://proceedings.mlr.press/v51/jaffe16.html
- https://proceedings.mlr.press/v40/Balsubramani15.html
- https://mlanthology.org/aaai/2019/arachie2019aaai-adversarial
- https://proceedings.mlr.press/v130/mazzetto21a.html
- https://arxiv.org/abs/2109.11377
- https://arxiv.org/abs/1605.07723v1 and https://arxiv.org/abs/1711.10160 (via alphaxiv listing)
- https://openaccess.thecvf.com/content/CVPR2024/html/Ren_NeRF_On-the-go_Exploiting_Uncertainty_for_Distractor-free_NeRFs_in_the_Wild_CVPR_2024_paper.html
- https://arxiv.org/abs/2406.20055
- https://arxiv.org/abs/2407.08447v2
- https://openaccess.thecvf.com/content/CVPR2024/html/Goli_Bayes_Rays_Uncertainty_Quantification_for_Neural_Radiance_Fields_CVPR_2024_paper.html
- https://jiangwenpl.github.io/FisherRF
- https://arxiv.org/abs/2502.14684
- https://arxiv.org/abs/2604.05715
- https://arxiv.org/abs/2607.02554
- https://arxiv.org/abs/2602.23926
- https://arxiv.org/abs/2503.18363
- https://arxiv.org/abs/2509.16863v1
- https://openaccess.thecvf.com/content_CVPR_2020/html/Poggi_On_the_Uncertainty_of_Self-Supervised_Monocular_Depth_Estimation_CVPR_2020_paper.html
- https://arxiv.org/abs/2501.08188
- https://arxiv.org/pdf/2304.07193 and https://arxiv.org/pdf/2508.10104
- https://arxiv.org/html/2411.17385
- https://mlanthology.org/cvpr/2022/roth2022cvpr-total
- https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html
- https://mlanthology.org/neurips/2022/kobayashi2022neurips-decomposing
- https://arxiv.org/abs/2312.03203
- https://arxiv.org/abs/2606.29716
- https://arxiv.org/abs/2406.00798
- https://web.stanford.edu/group/scpnt/gpslab/pubs/papers/Blanch_et_al_IONGNSS_2012_v2.pdf and https://dblp.uni-trier.de/rec/journals/taes/BlanchWELPRSK15.html
