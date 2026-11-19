from __future__ import annotations

import numpy as np
import pytest

from sensorfault.baselines.base import BaseFuser, get_fuser, list_fusers, register
from sensorfault.types import FusionResult


def test_register_and_get(clean_registry):
    @register("dummy")
    class Dummy(BaseFuser):
        def fuse(self, frames, cfg):
            return FusionResult(points=np.zeros((len(frames), 3), np.float32))

    assert "dummy" in list_fusers()
    fuser = get_fuser("dummy")
    assert isinstance(fuser, Dummy)
    assert fuser.name == "dummy"


def test_fuse_runs(clean_registry, frame):
    @register("dummy2")
    class Dummy(BaseFuser):
        def fuse(self, frames, cfg):
            return FusionResult(points=np.zeros((len(frames), 3), np.float32))

    out = get_fuser("dummy2").fuse([frame, frame], {})
    assert out.points.shape == (2, 3)


def test_unknown_fuser_lists_known(clean_registry):
    @register("known_one")
    class K(BaseFuser):
        pass

    with pytest.raises(KeyError, match="known_one"):
        get_fuser("nope")


def test_list_is_sorted(clean_registry):
    for n in ("b_f", "a_f"):
        register(n)(type(n, (BaseFuser,), {}))
    names = list_fusers()
    assert names == sorted(names)


def test_kwargs_forwarded(clean_registry):
    @register("kw")
    class KW(BaseFuser):
        def __init__(self, alpha=1):
            self.alpha = alpha

    assert get_fuser("kw", alpha=3).alpha == 3


def test_base_fuse_not_implemented(frame):
    with pytest.raises(NotImplementedError):
        BaseFuser().fuse([frame], {})


def test_available_default():
    assert BaseFuser.available() == (True, "")
