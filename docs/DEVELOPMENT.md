# Development conventions

## Setup
- Conda route: `make env && conda activate sensorfault` (Python 3.10). A newer system Python
  (e.g. 3.12) is fine for CPU work: `python -m venv .venv && make install`.
- Heavy extras are optional: `.[tsdf]` (Open3D), `.[gs]` (torch + gsplat), `.[learned]`.
  Install torch with the wheel matching your CUDA driver first. Dev GPU target is 8 GB VRAM.
- `make lint`, `make test` must pass before sending changes. Pre-commit runs ruff + ruff-format.

## Seeds and determinism
- Every stochastic component takes an explicit seed; never use global RNG state
  (`np.random.default_rng(seed)`, not `np.random.seed`).
- Fault injection is deterministic: the per-(scene, sensor, fault, frame) seed is derived by
  hashing the master seed with those identifiers (e.g. SHA-256 truncated to 64 bits), so adding
  or reordering sensors/scenes does not change other runs' faults.
- Use at least 5 master seeds for reported results. Every run writes a JSON manifest (config,
  seeds, git revision, package versions) next to its outputs.
- GPU nondeterminism: record `torch.backends.cudnn` flags and the device name in the manifest;
  report seed variance, do not claim bitwise reproducibility on GPU.

## Data hygiene
- No data, renderings, checkpoints or third-party code are ever committed
  (`data/`, `outputs/`, `third_party/`, `checkpoints/` are git-ignored).
- Only public datasets, public sensor models and public code.
- Respect dataset licences (README table). Replica renderings must not be redistributed: publish
  the generator and seeds instead.

## Dev / test scene split
- Scenes are split once, by a fixed list committed under configs/, before any tuning.
- Hyperparameters and thresholds are tuned on dev scenes only. Test scenes are run once per
  final configuration; report scene-level bootstrap CIs and paired Wilcoxon tests.
- Never tune on test results; if you must revisit, record it as a deviation from the prereg.

## Code
- Shared contracts live in `sensorfault/types.py` and `sensorfault/baselines/base.py`; change
  them only with a note to all module owners.
- Wrappers for heavy external tools must implement `available()` and degrade gracefully.
- Tests: CPU only, offline. Mark GPU/data/slow tests with `@pytest.mark.gpu|data|slow`.
- Ruff, line length 100.
