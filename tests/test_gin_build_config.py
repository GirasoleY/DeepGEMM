import tempfile
import unittest
from pathlib import Path

from scripts.gin_build_config import resolve_gin_nccl_config


class GinBuildConfigTest(unittest.TestCase):
    def _make_nccl_root(self, parent: Path, version=(2, 30, 7)) -> Path:
        root = parent / "nccl"
        include = root / "include"
        device = include / "nccl_device"
        library = root / "lib"
        device.mkdir(parents=True)
        library.mkdir(parents=True)
        major, minor, patch = version
        version_code = major * 10000 + minor * 100 + patch
        (include / "nccl.h").write_text(
            f"""#define NCCL_MAJOR {major}
#define NCCL_MINOR {minor}
#define NCCL_PATCH {patch}
#define NCCL_VERSION_CODE {version_code}
ncclResult_t ncclCommWindowRegister();
""",
            encoding="utf-8",
        )
        (include / "nccl_device.h").write_text(
            '#include "nccl_device/core.h"\n#include "nccl_device/gin.h"\n',
            encoding="utf-8",
        )
        (device / "core.h").write_text(
            "struct ncclDevCommRequirements {};\n"
            "ncclResult_t ncclDevCommCreate();\n",
            encoding="utf-8",
        )
        (device / "gin.h").write_text(
            "NCCL_DEVICE_INLINE void get();\n"
            "NCCL_DEVICE_INLINE void flushAsync();\n",
            encoding="utf-8",
        )
        (library / "libnccl.so.2").touch()
        return root

    def test_disabled_build_does_not_require_nccl(self):
        self.assertIsNone(resolve_gin_nccl_config(False, {}))

    def test_enabled_build_requires_a_root(self):
        with self.assertRaisesRegex(RuntimeError, "requires DG_NCCL_ROOT"):
            resolve_gin_nccl_config(True, {})

    def test_valid_device_api_installation_is_resolved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._make_nccl_root(Path(temp_dir))
            config = resolve_gin_nccl_config(True, {"DG_NCCL_ROOT": str(root)})
            self.assertEqual(config.version_code, 23007)
            self.assertEqual(config.library_path.name, "libnccl.so.2")
            self.assertGreater(config.header_fingerprint, 0)

    def test_nccl_root_is_only_a_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._make_nccl_root(Path(temp_dir))
            config = resolve_gin_nccl_config(
                True,
                {"DG_NCCL_ROOT": str(root), "NCCL_ROOT": str(root / "missing")},
            )
            self.assertEqual(config.root, root.resolve())

    def test_old_nccl_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._make_nccl_root(Path(temp_dir), version=(2, 30, 2))
            with self.assertRaisesRegex(RuntimeError, "exactly NCCL 2.30.7"):
                resolve_gin_nccl_config(True, {"DG_NCCL_ROOT": str(root)})

    def test_newer_nccl_is_rejected_until_its_device_abi_is_validated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._make_nccl_root(Path(temp_dir), version=(2, 31, 0))
            with self.assertRaisesRegex(RuntimeError, "exactly NCCL 2.30.7"):
                resolve_gin_nccl_config(True, {"DG_NCCL_ROOT": str(root)})

    def test_required_device_api_surface_is_checked(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._make_nccl_root(Path(temp_dir))
            (root / "include/nccl_device/gin.h").write_text(
                "NCCL_DEVICE_INLINE void get();\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "flushAsync"):
                resolve_gin_nccl_config(True, {"DG_NCCL_ROOT": str(root)})

    def test_header_content_changes_jit_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._make_nccl_root(Path(temp_dir))
            first = resolve_gin_nccl_config(True, {"DG_NCCL_ROOT": str(root)})
            (root / "include/nccl_device/gin.h").write_text(
                "NCCL_DEVICE_INLINE void get();\n"
                "NCCL_DEVICE_INLINE void flushAsync();\n"
                "// changed\n",
                encoding="utf-8",
            )
            second = resolve_gin_nccl_config(True, {"DG_NCCL_ROOT": str(root)})
            self.assertNotEqual(first.header_fingerprint, second.header_fingerprint)


if __name__ == "__main__":
    unittest.main()
