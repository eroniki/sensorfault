# Evidence log: part_flt.bib (checked 2026-11-13)

## Method and limits
- Bibliographic fields (exact title, full author list, venue, year, volume/issue/pages/article no., DOI) were read from the **DOI.org/Crossref REST record**, `https://api.crossref.org/works/<DOI>`. Quoted below as "Crossref".
- Publisher pages (Wiley, IEEE Xplore, CVF, Project Euclid, ACM) returned 403 or empty content to automated fetch. DBLP served a bot challenge. No publisher HTML was read.
- arXiv API (`https://export.arxiv.org/api/query?id_list=<id>`) was used for Replica, Skoltech3D (author spelling) and US3D (abstract).
- "Line" = one factual line. Where Crossref/arXiv/Semantic Scholar had no abstract, the line restates the verified record, marked (record). No abstract was invented.
- Where Crossref lists initials only for the journal record, the .bib uses the initials seen. Full names appear in other Crossref records and were not substituted.

## Verified (in .bib: 23 of 23)

| Key | URL used | Line |
|---|---|---|
| flt_ambarkutuk | https://api.crossref.org/works/10.3390/s23239309 | Four-author Sensors 23(23):9309 paper; vibration-based occupant localization fusing per-sensor likelihoods, with Byzantine sensor elimination (Crossref abstract). |
| flt_lamport | https://api.crossref.org/works/10.1145/357172.357176 | Generals must agree on a plan despite traitors; reliable systems must handle components giving conflicting information (abstract via Semantic Scholar). |
| flt_marzullo | https://api.crossref.org/works/10.1145/128733.128735 | Transforms process-control programs to tolerate sensor failure; sensor-averaging accuracy and complexity analyzed over a hierarchy of failure models (Crossref abstract). |
| flt_brooksiyengar | https://api.crossref.org/works/10.1109/2.507632 | Computer 29(6):53-60, 1996, authors R.R. Brooks and S.S. Iyengar (record; no abstract retrieved). |
| flt_raim_parkinson | https://api.crossref.org/works/10.1002/j.2161-4296.1988.tb00955.x | Navigation 35(2):255-274, 1988; record lists Bradford W. Parkinson and Penina Axelrad (record; no abstract retrieved). |
| flt_raim_brown | https://api.crossref.org/works/10.1002/j.2161-4296.1992.tb02278.x | Navigation 39(3):301-316, 1992; sole author R. Grover Brown (record; no abstract retrieved). |
| flt_raim_joerger | https://api.crossref.org/works/10.1002/navi.71 | Navigation 61(4):273-291, Dec 2014; Joerger, Chan, Pervan (record; no abstract retrieved). |
| flt_raim_pervan | https://api.crossref.org/works/10.1002/j.2161-4296.1998.tb02372.x | Exact title is "A Multiple Hypothesis Approach to Satellite Navigation Integrity", Navigation 45(1):61-71, 1998 (record; no abstract retrieved). |
| flt_ransac | https://api.crossref.org/works/10.1145/358669.358692 | Introduces RANSAC for fitting models to data with a significant percentage of gross errors (Crossref abstract). |
| flt_msac | https://api.crossref.org/works/10.1006/cviu.1999.0832 | MLESAC generalizes RANSAC, choosing the solution that maximizes likelihood rather than inlier count (abstract via Semantic Scholar). |
| flt_huber | https://api.crossref.org/works/10.1214/aoms/1177703732 | Ann. Math. Statist. 35(1):73-101, 1964, Peter J. Huber (record; no abstract retrieved). |
| noise_khoshelham | https://api.crossref.org/works/10.3390/s120201437 | Calibrates the Kinect and analyses accuracy and resolution of its depth data, with a theoretical error model from disparity (Crossref abstract). |
| noise_nguyen | https://api.crossref.org/works/10.1109/3dimpvt.2012.84 | 3DIMPVT 2012, pp. 524-530; Nguyen, Izadi, Lovell (record; no abstract retrieved). |
| noise_lange | https://api.crossref.org/works/10.1109/3.910448 | IEEE J. Quantum Electron. 37(3):390-397, 2001; authors R. Lange and P. Seitz (record; no abstract retrieved). |
| noise_sarbolandi | https://api.crossref.org/works/10.1016/j.cviu.2015.05.006 | Compares Kinect structured-light and Kinect time-of-flight cameras with a set of nine tests, five of them novel (abstract via Semantic Scholar). |
| noise_matthies | https://api.crossref.org/works/10.1109/jra.1987.1087097 | IEEE J. Robot. Autom. 3(3):239-248, 1987; authors L. Matthies and S. Shafer (record; no abstract retrieved). |
| noise_freedman | https://api.crossref.org/works/10.1007/978-3-319-10590-1_16 | Sparse Reflections Analysis removes ToF multipath interference, including three or more paths and diffuse Lambertian multipath (abstract via Semantic Scholar). |
| data_skoltech3d | https://api.crossref.org/works/10.1109/cvpr52729.2023.02049 ; https://export.arxiv.org/api/query?id_list=2203.06111 | Dataset of about 1.4 million images of 107 scenes, 100 viewing directions, 14 lighting conditions, from multiple RGB and depth sensors (arXiv abstract). |
| data_replica | https://export.arxiv.org/api/query?id_list=1906.05797 | Replica: 18 photo-realistic indoor scene reconstructions with dense mesh, HDR textures, and semantic class and instance information (arXiv abstract). |
| data_us3d | https://api.crossref.org/works/10.1109/wacv.2019.00167 ; https://export.arxiv.org/api/query?id_list=1811.08739 | Public multi-view, multi-band satellite dataset with geometric and semantic ground truth for two large cities, plus baselines (arXiv abstract). |
| data_dfc2019 | https://api.crossref.org/works/10.1109/mgrs.2019.2949679 | IEEE GRSM 7(4):33-36, Dec 2019; the Data Fusion Contest series is organized by the GRSS IADF Technical Committee (Semantic Scholar snippet). |
| data_gongfritsch | https://api.crossref.org/works/10.5194/isprs-archives-xlii-2-363-2018 | Proposes a pipeline turning the JHU/APL multiple-view stereo satellite benchmark data into digital surface models (Crossref abstract). |
| data_mvs3dm | https://api.crossref.org/works/10.1109/aipr.2016.8010543 | 2016 AIPR Workshop, pp. 1-9; Bosch, Kurtz, Hagstrom, Brown (record; no abstract retrieved). |

## Notes and caveats per entry
- flt_ransac: Crossref gives title "Random sample consensus" plus subtitle "a paradigm for model fitting with applications to image analysis and automated cartography". The .bib joins them.
- flt_brooksiyengar, noise_lange, noise_matthies, flt_msac, data_gongfritsch: Crossref lists initials only. Initials kept.
- data_skoltech3d: Crossref spells "Bozhcnko". arXiv v4 (camera-ready) spells "Bozhenko". The .bib uses the arXiv spelling. The 16-author list is otherwise identical.
- data_dfc2019: Crossref title ends "[Technical Committees]", a magazine section tag. It was dropped from the .bib title. Entry is a 4-page column, not a research article.
- data_us3d: citation fields from Crossref (WACV 2019, pp. 1524-1532); arXiv 1811.08739 has the same title.
- data_replica: arXiv-only, as requested. arXiv lists 30 authors, all included.
- data_gongfritsch: Crossref stores the title in all caps; the .bib uses title case. Publication is ISPRS Archives XLII-2 (2018).
- noise_nguyen: Crossref booktitle was "...Visualization &amp; Transmission", decoded to `\&`.
- Month fields omitted for all entries (Crossref gives them only for some).
- Per-item abstract line was not retrievable for several older items. A reviewer who wants the abstract should open the DOI.

## NOT VERIFIED
None. All 23 keys met the field requirements via the Crossref record, with the caveats above (initials-only authors; "(record)" lines lack an abstract).
