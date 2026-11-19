from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from sensorfault.types import FusionResult


def test_depth_frame_fields(frame):
    assert frame.depth.shape == (4, 6)
    assert frame.K.shape == (3, 3)
    assert frame.T_wc.shape == (4, 4)
    assert frame.conf is None


def test_depth_frame_is_frozen(frame):
    with pytest.raises(dataclasses.FrozenInstanceError):
        frame.sensor_id = "other"


def test_depth_frame_replace_keeps_original(frame):
    moved = dataclasses.replace(frame, T_wc=np.diag([1.0, 1, 1, 1]) + np.eye(4) * 0)
    assert moved is not frame
    assert moved.sensor_id == frame.sensor_id


def test_fusion_result_defaults():
    r = FusionResult(points=np.zeros((0, 3), np.float32))
    assert r.normals is None
    assert r.excluded_sensors == {}
    assert r.sensor_scores == {}
    assert r.runtime_s == 0.0
    assert r.meta == {}


def test_fusion_result_mutable_defaults_not_shared():
    a = FusionResult(points=np.zeros((0, 3), np.float32))
    b = FusionResult(points=np.zeros((0, 3), np.float32))
    a.excluded_sensors["s0"] = True
    a.meta["k"] = 1
    assert b.excluded_sensors == {}
    assert b.meta == {}
