# H: foundations and datasets (verified 2026-11-16 via arXiv abstract pages)

All arXiv pages link only arXiv's non-exclusive distribution licence 1.0 ("Rights to this article") unless noted. Venue strings come from the arXiv "Comments" field; exact proceedings names were not on the pages, so booktitles are the short official names (check before camera-ready). Where no venue is on the page, entry is @misc.

| Key | Role in pipeline | URL | Caveats |
|---|---|---|---|
| ho2020ddpm | Diffusion training/forward process basis | https://arxiv.org/abs/2006.11239 | Page lists no venue (NeurIPS 2020 is not on the page, so not stated in bib) |
| song2021ddim | DDIM sampler | https://arxiv.org/abs/2010.02502 | Page: "ICLR 2021" |
| rombach2022latent | Latent diffusion / VAE latent space | https://arxiv.org/abs/2112.10752 | Page: "CVPR 2022" |
| podell2023sdxl | Frozen fp16 base model | https://arxiv.org/abs/2307.01952 | Page shows CC BY 4.0 icon; no venue on page |
| hu2021lora | LoRA fine-tuning | https://arxiv.org/abs/2106.09685 | No venue on page (v2) |
| salimans2022progressive | v-prediction parameterisation | https://arxiv.org/abs/2202.00512 | Page: "ICLR 2022" |
| lin2023common | Zero terminal SNR, trailing spacing | https://arxiv.org/abs/2305.08891 | CC BY-SA 4.0 icon on page; no venue on page |
| ho2022classifierfree | CFG, only if guidance/dropout used | https://arxiv.org/abs/2207.12598 | Page: short version at NeurIPS 2021 Workshop DGMs; cite as arXiv |
| yao2020blendedmvs | Training data (BlendedMVS/BlendedMVG) | https://arxiv.org/abs/1911.10127 | Page: "Accepted to CVPR2020". Check dataset licence separately. BlendedMVG name not checked on page |
| roberts2021hypersim | Training data | https://arxiv.org/abs/2011.02523 | Page: ICCV 2021. arXiv year 2020, bib uses 2021 (venue year). Check dataset licence separately |
| wang2020tartanair | Training data | https://arxiv.org/abs/2003.14338 | Page: IROS 2020 (IEEE/RSJ). Check dataset licence separately |
| yang2024depthanythingv2 | Synthetic-vs-real training-data discussion (optional) | https://arxiv.org/abs/2406.09414 | Page: NeurIPS 2024 |

## UNVERIFIED
- ETH3D benchmark paper (suggested key schops2017eth3d): "A Multi-View Stereo Benchmark with High-Resolution Images and Multi-Camera Videos", CVPR 2017. Confirmed only via https://www.eth3d.net/ (title, venue CVPR 2017, authors as initials: T. Schöps, J. L. Schönberger, S. Galliani, T. Sattler, K. Schindler, M. Pollefeys, A. Geiger). CVF open-access page returned 403 and IEEE Xplore (doi 10.1109/CVPR.2017.272 -> ieeexplore 8099755) returned empty, so full given names and proceedings name are not verified; not in .bib. Verify manually on openaccess.thecvf.com.
