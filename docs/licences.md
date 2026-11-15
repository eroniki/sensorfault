# Licences and access terms (checked 2026-11-14; not legal advice)

"Read" = primary text read by a reader agent or by me. Anything else is marked unverified.

## Backbones and baselines
| Item | Licence as read | Use limits / notes |
|---|---|---|
| **Stable Diffusion 2.x** | **STILL UNAVAILABLE (retried 2026-11-14).** The HF API returns HTTP 401 for `stabilityai/stable-diffusion-2`, `-2-1`, `-2-1-base` and `-2-base`, while `stable-diffusion-xl-base-1.0` returns 200 from the same host; no HF token is configured on this machine; the Stability-AI/stablediffusion GitHub LICENSE-MODEL and README raw URLs return 404. Marigold's v1-1 card links the SD2 LICENSE-MODEL and labels it "CreativeML Open RAIL++-M" (label only). | Cannot tell whether gated, moved or removed. **Decision (author): SDXL is the first backbone**; SD 2.x only as a later comparison arm if an authenticated check clears it. |
| **SDXL base 1.0 (first backbone)** | Read in full (LICENSE.md, "CreativeML Open RAIL++-M License dated July 26, 2023"); obligations for derivatives in the next section. | Fine-tuning expressly allowed. No research-only clause (the card's "intended for research" is a statement of intent, not a licence term). |
| SD 3.5 Medium | Read: Stability AI Community License (July 5, 2024), gated click-through | Free for research/non-commercial; registration only for commercial use; revenue ($1M) clause is in the commercial grant. If distributing it or derivatives: licence copy, Notice file, "Powered by Stability AI". Must not be used to create or improve a *foundational generative AI model*; AUP incorporated (AUP not read). |
| FLUX.1 schnell | Card tag Apache-2.0, card text allows scientific use (LICENSE file not read) | Few-step distilled model. |
| FLUX.1 dev | Read: FLUX.1 [dev] Non-Commercial License v1.1.1 | Non-commercial only; restrictions pass to derivatives; excludes surveillance-related R&D. Avoid. |
| Marigold code | Apache-2.0 (README) | |
| Marigold weights | **Conflicting:** README says RAIL++-M; v1-0 card says Apache-2.0; v1-1 card says OpenRAIL++. LICENSE-MODEL not read. | Baseline only. |
| Depth Anything V2 | Cards: Small Apache-2.0; Base, Large CC-BY-NC-4.0; Giant not reachable (401). Licence texts not read. | Not a dependency; at most a baseline. |
| Apple Depth Pro | Read: Apple custom licence (LICENSE), weights "released under the LICENSE terms"; no research-only limit found; ACKNOWLEDGEMENTS.md not read | Not a dependency in v3. |
| AffineGS (2611.07709) | Paper CC BY-NC-SA 4.0 (arXiv badge). **No code link on the abstract page**; project page not read. | Baseline must be re-implemented unless code appears. |
| MVGD, DepthSync | No code or weights found | Cannot be run as baselines for now. |

## Datasets (see also docs section in PLAN.md)
| Dataset | Licence / terms read | Scenes with GT | Independent-researcher fit |
|---|---|---|---|
| **ETH3D** | No licence or terms text on any page read (datasets, documentation, overview, about, imprint). Only: "Users utilize the contents of this website at their own risk." Paper not read. | **High-res DSLR: 13 training scenes with public GT** (14-76 images each; laser scans + rendered depth maps, 4-byte float binaries), **12 test scenes without GT**. Low-res rig: 5 train / 5 test. | Open download, no login. **Decision (author): use with citation, no redistribution.** Absence of a licence is not a grant: keep data and any derived GT-based files out of git and out of any release; ask the maintainers before releasing derived data. |
| **ScanNet++ v2** | **Kept in by the author's decision (2026-11-14); application drafted, nothing submitted.** Terms read in full: see the verbatim section below and `docs/scannetpp-application-draft.md`. | 1006 scenes; splits 856 train, 50 val; test scenes have no GT. Laser scan + iPhone LiDAR depth. | Open points: clause 7, the "established researcher" rule, institutional email on the registration form. |
| Matterport3D | Read (EULA for Academic Use): non-commercial academic only; trained models are "Derived Information"; recipients must give academic institution | 90 buildings | Ruled out (academic institution). |
| Hypersim | CC BY-SA 3.0 (README). Code licence separate. | 461 scenes, synthetic | Usable; ShareAlike may apply to a released derived dataset (weights unclear): legal check before release. |
| TartanAir | CC BY 4.0 | 30 sim environments | Usable, synthetic. |
| BlendedMVS / BlendedMVG | CC BY 4.0 | 113 + 389 scenes; depth rendered from textured meshes; metric scale not stated | Usable; real-image/mesh-rendered depth, good mixed pool. |
| Replica | Replica Dataset Research Terms (non-commercial research, no institution requirement) | 18 scenes | Usable; small, synthetic-like. |
| DTU MVS | No licence text found ("freely available") | 124 scenes, lab, structured-light GT | Usable with caution; object-scale. |
| Tanks and Temples | Page contradictory: CC BY 4.0 headline but grant "sole purpose of non-commercial scientific research", no redistribution | laser scans; counts not read | Evaluation only, no redistribution. |
| ARKitScenes | Apple licence, non-commercial | 1661 scenes | Non-commercial; terms DATA.md not read. |
| 7-Scenes | Microsoft non-commercial | 7 | Small; GT from KinectFusion (not laser). |
| KITTI | CC BY-NC-SA 3.0, academic use | outdoor | Outdoor only. |
| Waymo | Non-commercial; institution-representation clause | outdoor | Ruled out for now. |
| ScanNet v2 | Unverified (host TLS error) | | Unknown. |
| Co3D | Code CC BY-NC 4.0; dataset licence not found; no metric GT claim found | | Not suitable (no metric GT). |
| MVS-Synth, MuSHRoom | Not fetched | | Unchecked. |

### Shortlist if ScanNet++ is out
Labelled training pool: **BlendedMVG/BlendedMVS (CC BY 4.0, 500 scenes)** as the main real-ish pool, **Hypersim** and **TartanAir** (synthetic, as Marigold uses synthetic data), Replica for small indoor tests. Evaluation: **ETH3D 13 scenes** (primary, with leave-one-scene-out), DTU as a second object-scale test. Unlabelled pool for the semi-supervised term: any posed RGB, including ETH3D's 12 test scenes (images only) and BlendedMVS images used without their depth. Real indoor laser-GT at ScanNet++ scale has no permissive replacement in this list.


## SDXL OpenRAIL++-M: what derivatives require (read from LICENSE.md, 2026-11-14; not legal advice)

"Derivatives of the Model" includes "all modifications to the Model, works based on the Model" and any model "initialized by transfer of patterns of the weights, parameters, activations or output of the Model ... including ... distillation methods ... or methods based on the generation of synthetic data by the Model for training the other model". So LoRA/fine-tuned weights, and any model trained on SDXL-generated pseudo-labels (our EMA-teacher chain starts from the fine-tuned model anyway), are Derivatives.

Research use itself: "Use may include creating any content with, finetuning, updating, running, training, evaluating and/or reparametrizing the Model." Preprint, code and evaluation need no extra step. The obligations bite on **Distribution** ("any transmission, reproduction, publication or other sharing of the Model or Derivatives of the Model to a third party, including providing the Model as a hosted service"), i.e. if LoRA or fine-tuned weights are ever released:
1. "Use-based restrictions as referenced in paragraph 5 MUST be included as an enforceable provision by You in any type of legal agreement (e.g. a license) governing the use and/or distribution of the Model or Derivatives of the Model", and recipients must be notified they are subject to paragraph 5 (Attachment A).
2. "You must give any Third Party recipients of the Model or Derivatives of the Model a copy of this License".
3. "You must cause any modified files to carry prominent notices stating that You changed the files".
4. "You must retain all copyright, patent, trademark, and attribution notices".
5. Own terms may be added "respecting paragraph 4.a." (the use restrictions stay).
Outputs: "Licensor claims no rights in the Output You generate"; "You are accountable for the Output you generate and its subsequent uses".
The Complementary Material (code and scripts) is exempt from the use-restriction pass-through ("This provision does not apply to the use of Complementary Material"), so releasing our code needs no such clause.

Attachment A restrictions that must be passed on (abridged): violating law; exploiting minors; verifiably false content to harm; PII to harm; defamation/harassment; fully automated decisions adversely affecting legal rights; discrimination or exploitation of vulnerable groups; medical advice; administration of justice, law enforcement, immigration or asylum. None blocks depth estimation research, but a released depth model would carry them.

Practical rule for this project: do not release SDXL-derived weights without a LICENSE carrying Attachment A, a copy of the CreativeML Open RAIL++-M text and a change notice; releasing only code, configs and the paper has no pass-through duty. Marigold's own licence labelling conflict (see table) does not matter while Marigold is a baseline only.

## ScanNet++ Terms of Use: verbatim (PDF decoded locally with pdftotext, 2026-11-14)

Source: https://scannetpp.mlsg.cit.tum.de/scannetpp/static/scannetpp-terms-of-use.pdf

Preamble: "After approval, you (the "Researcher") receive permission to use the ScanNet++ database (the "Database") at the Technical University of Munich."

> 1. Researcher shall use the Database only for non-commercial research and educational purposes. Commercial use is strictly prohibited.
> 4. Researcher may only provide research associates and colleagues with the data, after the receiving entity has also agreed to and signed these terms and conditions. Sharing the data otherwise is strictly prohibited.
> 5. Following GDPR, Researcher must ensure that they can delete all person-specific data of a person upon request.
> 6. Technical University of Munich reserves the right to terminate Researcher's access to the Database or parts thereof at any time.
> **7. If Researcher is employed by a for-profit, commercial entity, Researcher's employer shall also be bound by these terms and conditions, and Researcher hereby represents that he or she is fully authorized to enter into this agreement on behalf of such employer.**

Form fields: "Requester's name / Requester's email / PI/Supervisor's[1,2] name / PI/Supervisor's email / Affiliation (Full University / Company Name) / Place, Date / Requester's signature[3] / Place, Date / PI/Supervisor's signature".

Footnotes, verbatim:
> 1. Principal investigator (PI) is the holder of an independent grant and the lead researcher for the grant project. For doctoral candidates and postdoctoral fellows, this is usually the supervising professor or research group leader. For researchers in for-profit commercial entities, the supervisor is a person who can legally represent the Researcher such as a manager.
> 2. For independent applicants, fill in the same details as the Requester here. Only established researchers with a publicly verifiable research profile (postdocs, junior professors, senior researchers in companies) can apply independently.
> 3. Signatures must be handwritten. Digital signatures cannot be accepted.

### How applicants with no supervisor are handled (public sources only)
- The PDF's footnote 2 is the only rule found: an independent applicant repeats their own details in the PI/Supervisor fields, and only "established researchers with a publicly verifiable research profile" are eligible. Its examples are postdocs, junior professors and senior researchers in companies; **an applicant with no institution or company is not listed**, and no page defines "publicly verifiable" (Google Scholar, ORCID and DBLP are not mentioned anywhere read).
- Registration form (https://scannetpp.mlsg.cit.tum.de/scannetpp/register): fields Email ("Please use an institutional email address"), Password, First/Last name ("full name ... english characters only"), Affiliation, and an "I accept the terms and conditions" box. No alternative is given for people without an institutional email.
- Process: "create an account, login and create an application. Once your application is approved, you will receive a personalized token". The application form itself is behind login (not read). How and where the handwritten-signed PDF is submitted is not on any public page; review time is not stated.
- Contact for access questions: scannetpp@googlegroups.com (contact page). GitHub issues show nothing on access.
- Not addressed anywhere readable: training models on the data, publishing weights or derived pseudo-labels (clause 4 forbids sharing the data itself; a model trained on it is not mentioned).

### Clause 7, read plainly
The clause turns on **employment status, not on whose time or resources are used**: it applies to a researcher "employed by a for-profit, commercial entity", and has no exception for work done privately. If that describes the applicant, signing includes a representation of authority to bind the employer; if the employer is not willing to be bound, that representation could not truthfully be made. I cannot settle this; it is the author's decision, ideally after asking TUM in writing (draft email in the application draft). If the applicant is not employed by a for-profit entity, clause 7 does not apply.
