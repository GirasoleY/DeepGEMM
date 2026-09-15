"""Resolve the NCCL Device API headers used by the opt-in GIN backend."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


REQUIRED_NCCL_VERSION_CODE = 23007


@dataclass(frozen=True)
class GinNcclBuildConfig:
    root: Path
    include_dir: Path
    header_fingerprint: str

    @property
    def compile_definitions(self) -> tuple[str, ...]:
        return (
            "DG_MEGAMOE_GIN=1",
            f"DG_NCCL_HEADERS_FINGERPRINT=0x{self.header_fingerprint}ULL",
        )


def _read_decimal_define(header: str, name: str) -> int:
    match = re.search(
        rf"^\s*#\s*define\s+{re.escape(name)}\s+(\d+)\b",
        header,
        flags=re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"NCCL header does not define {name} as an integer")
    return int(match.group(1))


def _device_header_fingerprint(include_dir: Path) -> str:
    device_dir = include_dir / "nccl_device"
    paths = [include_dir / "nccl.h", include_dir / "nccl_device.h"]
    # Quoted Device API includes are private implementation details and do not
    # consistently use a header suffix. Fingerprint every regular file so a
    # change to any transitively consumed input invalidates the JIT cache.
    paths.extend(path for path in device_dir.rglob("*") if path.is_file())

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(include_dir).as_posix()):
        digest.update(path.relative_to(include_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def resolve_gin_nccl_config(
    enabled: bool,
    env: Mapping[str, str] = os.environ,
) -> Optional[GinNcclBuildConfig]:
    if not enabled:
        return None

    root_value = env.get("DG_NCCL_ROOT") or env.get("NCCL_ROOT")
    if not root_value:
        raise RuntimeError(
            "DG_MEGAMOE_GIN=1 requires DG_NCCL_ROOT "
            "(NCCL_ROOT is accepted as a fallback)"
        )

    root = Path(root_value).expanduser().resolve()
    include_dir = root / "include"
    required_headers = (
        include_dir / "nccl.h",
        include_dir / "nccl_device.h",
        include_dir / "nccl_device" / "core.h",
        include_dir / "nccl_device" / "gin.h",
    )
    missing = [str(path) for path in required_headers if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"incomplete NCCL Device API headers under {include_dir}: "
            f"missing {', '.join(missing)}"
        )

    nccl_header = required_headers[0].read_text(encoding="utf-8")
    declared_version = _read_decimal_define(nccl_header, "NCCL_VERSION_CODE")
    if declared_version != REQUIRED_NCCL_VERSION_CODE:
        major = declared_version // 10000
        minor = declared_version // 100 % 100
        patch = declared_version % 100
        raise RuntimeError(
            "MegaMoE GIN requires exactly NCCL 2.30.7; "
            f"found {major}.{minor}.{patch} under {root}"
        )

    return GinNcclBuildConfig(
        root=root,
        include_dir=include_dir,
        header_fingerprint=_device_header_fingerprint(include_dir),
    )
