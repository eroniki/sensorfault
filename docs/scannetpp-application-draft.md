# ScanNet++ access: draft for the author to review and submit personally

**Nothing has been submitted, signed or registered by me.** The form is behind a login and requires handwritten signatures
(Terms of Use, footnote 3), which only you can provide. Bracketed items are yours to fill or decide. Terms text: `docs/licences.md`.

## Before applying: three things the public pages leave open
1. **Eligibility.** Footnote 2 allows independent applicants only if they are "established researchers with a publicly verifiable
   research profile" (listed: postdocs, junior professors, senior researchers in companies). An unaffiliated applicant is not listed.
   Have a verifiable profile ready (e.g. ORCID, Google Scholar or DBLP page, publication list) and say so in the application.
2. **Institutional email.** The registration page asks for one. Decide which address you will use; do not use an address that
   implies an affiliation you do not have.
3. **Clause 7** (employment by a for-profit entity binds the employer and requires you to represent authority to bind them).
   Choose the statement that is true for you in the employment line below. If it is not the first one, ask TUM first (email below).

## Optional email to the maintainers (scannetpp@googlegroups.com) before applying
> Subject: Access question: independent researcher, no supervisor
>
> Dear ScanNet++ team,
>
> I would like to apply for ScanNet++ access for non-commercial research as an independent researcher ([ORCID / Scholar / DBLP link]).
> Two questions about the Terms of Use before I apply:
> 1. I have no supervisor or institution. Footnote 2 says independent applicants repeat their own details in the PI/supervisor
>    fields and must be established researchers with a publicly verifiable research profile. Does my profile above qualify, and which
>    email address should I register with?
> 2. [Choose one that is true.] (a) I am not employed by a for-profit entity, so clause 7 does not apply. / (b) I am employed by a
>    for-profit company in an unrelated role; this research is done independently, on my own time and resources, and the company has no
>    role in it. Does clause 7 apply to me, and if so what do you require from the employer?
>
> Thank you,
> [Name], Independent Researcher

## Registration fields
- Email: [address you will use]
- Name: [First] [Last] (English characters only)
- Affiliation: Independent Researcher

## Application text (adapt to the form's fields)
> **Requester:** [Name], Independent Researcher. **PI/Supervisor:** same as requester (independent applicant, per footnote 2 of the Terms of Use).
> **Research profile:** [ORCID / Google Scholar / DBLP / personal research page URL].
>
> **Purpose.** Non-commercial research on calibrated multi-view depth estimation: fine-tuning a pretrained image diffusion model to
> predict affine-invariant per-view depth, with sampling-time guidance from a Gaussian-splatting surface built from other views, and
> evaluating the calibration of its sample uncertainty and a leave-one-view-out fault test. ScanNet++ would be used for training (a labelled
> subset with laser/iPhone depth and an unlabelled subset without) and for evaluation on held-out scenes.
>
> **Intended output.** A research preprint and code. I will not redistribute any ScanNet++ data, images, depths, meshes or derived
> per-scene pseudo-labels. [Optional: I will ask the maintainers before releasing any trained weights, since the terms do not address trained models.] Results will cite ScanNet++.
>
> **Data handling.** [Confirm each statement is true for your setup:] stored on a personal access-controlled workstation [and on rented GPU storage, deleted after each job], not in any public repository or
> shared bucket; no person-specific data will be extracted or retained, and I can delete any person-specific data on request (clause 5).
>
> **Commercial use.** None. The work is non-commercial research done on my own time and resources.
>
> **Employment (clause 7).** [Choose the one that is true: (a) I am not employed by a for-profit, commercial entity. / (b) I am employed
> by a for-profit entity in an unrelated role; I request TUM's written confirmation of how clause 7 applies before I sign.]
>
> **Agreement.** I have read and accept the ScanNet++ Terms of Use, including clauses 1 (non-commercial only), 4 (no sharing of the data) and 6.

## Signature step (yours)
Print the Terms of Use PDF (https://scannetpp.mlsg.cit.tum.de/scannetpp/static/scannetpp-terms-of-use.pdf), fill requester and PI/supervisor
fields with your details (same person), and sign by hand in both signature places; digital signatures are not accepted. How to submit the
signed copy is not stated on public pages: look for it in the application form after login, otherwise ask scannetpp@googlegroups.com.

## Storage note for this project
ScanNet++ data stays outside the repository (`.gitignore` has `data/`); only manifests and hashes may be committed.
