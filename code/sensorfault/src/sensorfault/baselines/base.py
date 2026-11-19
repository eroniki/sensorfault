"""Registry + base class for fusers. Baselines register with @register("name")."""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Sequence, Type

from sensorfault.types import DepthFrame, FusionResult

_REGISTRY: Dict[str, Type["BaseFuser"]] = {}


def register(name: str) -> Callable[[Type["BaseFuser"]], Type["BaseFuser"]]:
    def deco(cls: Type["BaseFuser"]) -> Type["BaseFuser"]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def get_fuser(name: str, **kwargs) -> "BaseFuser":
    if name not in _REGISTRY:
        raise KeyError(f"unknown fuser {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_fusers() -> list[str]:
    return sorted(_REGISTRY)


class BaseFuser:
    name: str = "base"

    def fuse(self, frames: Sequence[DepthFrame], cfg: Mapping) -> FusionResult:
        raise NotImplementedError

    @classmethod
    def available(cls) -> tuple[bool, str]:
        """(ok, reason). Wrappers around heavy external tools return False with a reason
        when the tool is not installed, so the suite degrades gracefully."""
        return True, ""
