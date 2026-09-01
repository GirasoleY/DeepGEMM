"""Validation helpers for the opt-in MegaMoE NCCL Device API build.

This module deliberately uses only the Python standard library so its behavior
can be tested without importing PyTorch or initializing CUDA.
"""

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
    library_dir: Path
    library_path: Path
    version_code: int
    header_fingerprint: int


def _read_version_component(header: str, name: str) -> int:
    match = re.search(
        rf"^\s*#\s*define\s+{re.escape(name)}\s+(\d+)\b",
        header,
        flags=re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"NCCL header does not define {name}")
    return int(match.group(1))


def _fingerprint_headers(include_dir: Path) -> int:
    digest = hashlib.sha256()
    files = sorted(path for path in include_dir.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"NCCL include directory is empty: {include_dir}")
    for path in files:
        digest.update(path.relative_to(include_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    # A 64-bit content identity is sufficient for the JIT cache signature and
    # can be passed to the compiler as an integer macro without shell quoting.
    return int.from_bytes(digest.digest()[:8], byteorder="big")


def resolve_gin_nccl_config(
    enabled: bool,
    env: Mapping[str, str] = os.environ,
) -> Optional[GinNcclBuildConfig]:
    """Resolve and validate the NCCL installation used by a GIN build.

    ``DG_NCCL_ROOT`` is authoritative. ``NCCL_ROOT`` is accepted only as a
    compatibility fallback. The returned paths are absolute so the extension's
    RUNPATH cannot silently change with the process working directory.
    """

    if not enabled:
        return None

    root_value = env.get("DG_NCCL_ROOT") or env.get("NCCL_ROOT")
    if not root_value:
        raise RuntimeError(
            "DG_MEGAMOE_GIN=1 requires DG_NCCL_ROOT (NCCL_ROOT is accepted "
            "as a fallback)"
        )

    root = Path(root_value).expanduser().resolve()
    include_dir = root / "include"
    required_headers = (
        include_dir / "nccl.h",
        include_dir / "nccl_device.h",
        include_dir / "nccl_device" / "core.h",
        include_dir / "nccl_device" / "gin.h",
    )
    missing_headers = [str(path) for path in required_headers if not path.is_file()]
    if missing_headers:
        raise RuntimeError(
            "NCCL Device API headers are incomplete under "
            f"{include_dir}: missing {', '.join(missing_headers)}"
        )

    nccl_header = required_headers[0].read_text(encoding="utf-8")
    major = _read_version_component(nccl_header, "NCCL_MAJOR")
    minor = _read_version_component(nccl_header, "NCCL_MINOR")
    patch = _read_version_component(nccl_header, "NCCL_PATCH")
    version_code = major * 10000 + minor * 100 + patch
    declared_version_code = _read_version_component(nccl_header, "NCCL_VERSION_CODE")
    if version_code != declared_version_code:
        raise RuntimeError(
            "Inconsistent NCCL version macros in "
            f"{required_headers[0]}: computed {version_code}, declared "
            f"{declared_version_code}"
        )
    if version_code != REQUIRED_NCCL_VERSION_CODE:
        raise RuntimeError(
            "MegaMoE GIN prototype requires exactly NCCL 2.30.7; found "
            f"{major}.{minor}.{patch} under {root}"
        )

    compatibility_checks = (
        (required_headers[0], nccl_header, "ncclCommWindowRegister"),
        (
            required_headers[2],
            required_headers[2].read_text(encoding="utf-8"),
            "ncclDevCommCreate",
        ),
        (
            required_headers[3],
            required_headers[3].read_text(encoding="utf-8"),
            "flushAsync",
        ),
        (
            required_headers[3],
            required_headers[3].read_text(encoding="utf-8"),
            "NCCL_DEVICE_INLINE void get",
        ),
    )
    for path, text, symbol in compatibility_checks:
        if symbol not in text:
            raise RuntimeError(
                f"NCCL header {path} lacks required GIN API token {symbol!r}"
            )

    library_path = None
    library_dir = None
    for candidate_dir in (root / "lib", root / "lib64"):
        candidate = candidate_dir / "libnccl.so.2"
        if candidate.is_file():
            library_dir = candidate_dir
            library_path = candidate
            break
    if library_path is None or library_dir is None:
        raise RuntimeError(
            f"DG_NCCL_ROOT must contain lib/libnccl.so.2 or "
            f"lib64/libnccl.so.2: {root}"
        )

    return GinNcclBuildConfig(
        root=root,
        include_dir=include_dir,
        library_dir=library_dir,
        library_path=library_path,
        version_code=version_code,
        header_fingerprint=_fingerprint_headers(include_dir),
    )
