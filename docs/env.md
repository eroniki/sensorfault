# Development environment (what worked on the RTX 2070 host, 2026-11-14)

CPU-only work (toy, statistic, ETH3D interface, manifests): Python 3.10-3.12 with `numpy scipy pytest pillow py7zr`.

GPU work (2DGS adapter, stage-0 harness): the stack that worked is the one `sensorfault/environment.yml` recommends for this host:
- Python **3.10** (created with `uv venv --python 3.10`; the system python3.10 lacks `ensurepip`),
- `torch==2.4.1` from `https://download.pytorch.org/whl/cu121`,
- `gsplat==1.4.0` (compiles its CUDA extension at first use), `ninja`, `setuptools<81`, `wheel`,
- system `nvcc` 12.0 (`CUDA_HOME=/usr`, `TORCH_CUDA_ARCH_LIST=7.5`, `MAX_JOBS=4`); first build takes about 40 minutes with `nice`,
- for the harness: `diffusers==0.30.3 peft==0.12.0 accelerate==0.34.2 transformers==4.44.2 "numpy<2"` (newer diffusers/peft need torch >= 2.6).

Tried and failed: Python 3.12 + torch 2.15 (cu13.2) + pip `nvidia-cuda-nvcc` 13.x: header/compiler mismatch, then nvcc segfault and a `ptxas` error while building gsplat.

Run: `PYTHONPATH=src:tests python -m pytest -q` (gsplat and harness tests skip themselves without CUDA, torch or diffusers).
Real-data ETH3D test: `ETH3D_PIPES=/path/to/eth3d/pipes/pipes`. Data live outside the repository.
