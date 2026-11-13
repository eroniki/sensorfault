"""Baselines. Importing this package registers every baseline whose module imports.

Each module imports only numpy at module level; heavy deps (open3d, torch, external
binaries) are imported lazily and reported through ``Fuser.available()``. The
try/except keeps the package importable if a module has an unexpected import problem.
"""

from sensorfault.baselines.base import BaseFuser, get_fuser, list_fusers, register

for _m in (
    "median_fusion",
    "loo_chi2",
    "subset_msac",
    "consistency_voting",
    "merrell_visibility",
    "tvl1_range",
    "tsdf_open3d",
    "openmvs",
    "_external",
):
    try:
        __import__(f"{__name__}.{_m}")
    except ImportError:  # pragma: no cover
        pass
