import tempfile
import unittest
from pathlib import Path

from scripts.gin_build_config import resolve_gin_nccl_config


class GinBuildConfigTest(unittest.TestCase):
    def _make_nccl_root(self, parent: Path, version=(2, 30, 7)) -> Path:
        root = parent / "nccl"
        include_dir = root / "include"
        device_dir = include_dir / "nccl_device"
        device_dir.mkdir(parents=True)

        major, minor, patch = version
        version_code = major * 10000 + minor * 100 + patch
        (include_dir / "nccl.h").write_text(
            f"#define NCCL_VERSION_CODE {version_code}\n",
            encoding="utf-8",
        )
        (include_dir / "nccl_device.h").write_text(
            '#include "nccl_device/core.h"\n#include "nccl_device/gin.h"\n',
            encoding="utf-8",
        )
        (device_dir / "core.h").write_text(
            "// private Device API input\n",
            encoding="utf-8",
        )
        (device_dir / "gin.h").write_text(
            "// private GIN input\n",
            encoding="utf-8",
        )
        return root

    def test_disabled_build_needs_no_nccl_installation(self):
        self.assertIsNone(resolve_gin_nccl_config(False, {}))

    def test_enabled_build_requires_nccl_root(self):
        with self.assertRaisesRegex(RuntimeError, "requires DG_NCCL_ROOT"):
            resolve_gin_nccl_config(True, {})

    def test_missing_device_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._make_nccl_root(Path(temp_dir))
            (root / "include/nccl_device/gin.h").unlink()
            with self.assertRaisesRegex(RuntimeError, "nccl_device/gin.h"):
                resolve_gin_nccl_config(True, {"DG_NCCL_ROOT": str(root)})

    def test_only_exact_validated_version_is_accepted(self):
        for version in ((2, 30, 6), (2, 31, 0)):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temp_dir:
                root = self._make_nccl_root(Path(temp_dir), version)
                with self.assertRaisesRegex(RuntimeError, "exactly NCCL 2.30.7"):
                    resolve_gin_nccl_config(
                        True, {"DG_NCCL_ROOT": str(root)}
                    )

    def test_header_content_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._make_nccl_root(Path(temp_dir))
            first = resolve_gin_nccl_config(
                True, {"DG_NCCL_ROOT": str(root)}
            )
            core_header = root / "include/nccl_device/core.h"
            core_header.write_text(
                core_header.read_text(encoding="utf-8") + "// ABI change\n",
                encoding="utf-8",
            )
            second = resolve_gin_nccl_config(
                True, {"DG_NCCL_ROOT": str(root)}
            )
            self.assertNotEqual(first.header_fingerprint, second.header_fingerprint)

    def test_valid_config_has_exact_compile_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._make_nccl_root(Path(temp_dir))
            config = resolve_gin_nccl_config(
                True, {"DG_NCCL_ROOT": str(root)}
            )
            self.assertEqual(config.root, root.resolve())
            self.assertEqual(config.include_dir, root.resolve() / "include")
            self.assertRegex(config.header_fingerprint, r"^[0-9a-f]{16}$")
            self.assertEqual(
                config.compile_definitions,
                (
                    "DG_MEGAMOE_GIN=1",
                    "DG_NCCL_HEADERS_FINGERPRINT="
                    f"0x{config.header_fingerprint}ULL",
                ),
            )


if __name__ == "__main__":
    unittest.main()
