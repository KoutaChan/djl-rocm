"""CPU-only producer checks: python3 -m unittest discover -s rocm/hipblaslt."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("bundle", ROOT / "prepare-bundle.py")
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


class BundleProfilesTest(unittest.TestCase):
    def write_sdk(self, root, version, profile):
        (root / ".info").mkdir()
        (root / ".info/version").write_text(version + "-123-build")
        headers = root / "include/hipblaslt"
        headers.mkdir(parents=True)
        fields = dict(zip(("MAJOR", "MINOR", "PATCH"), profile["hipblaslt_version"].split(".")))
        fields["TWEAK"] = profile["source_revision"][:10]
        (headers / "hipblaslt-version.h").write_text("".join(f"#define HIPBLASLT_VERSION_{key} {value}\n" for key, value in fields.items()))

    def test_select_every_pinned_sdk_and_reject_cross_flavor(self):
        for flavor, settings in bundle.PROFILES.items():
            for version, profile in settings["sdk_versions"].items():
                with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    self.write_sdk(root, version, profile)
                    self.assertEqual(bundle.sdk_profile(root, flavor), (flavor, version, profile))
                    with self.assertRaisesRegex(RuntimeError, "Requested"):
                        bundle.sdk_profile(root, "rocm0.0")
                    for patch in profile["patches"]:
                        self.assertTrue((ROOT / "patches" / patch).is_file())

    def test_reject_same_version_with_different_vendor_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = bundle.PROFILES["rocm7.2"]["sdk_versions"]["7.2.2"]
            self.write_sdk(root, "7.2.2", profile)
            header = root / "include/hipblaslt/hipblaslt-version.h"
            header.write_text(header.read_text().replace(profile["source_revision"][:10], "0123456789"))
            with self.assertRaisesRegex(RuntimeError, "source differs"):
                bundle.sdk_profile(root, "rocm7.2")

    def test_flat_metadata_preserves_all_architectures_and_global_mappings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            libraries = root / "lib/hipblaslt/library"
            libraries.mkdir(parents=True)
            names = ("TensileLibrary_lazy_gfx1100.dat", "TensileLibrary_lazy_gfx942.dat",
                     "TensileLiteLibrary_lazy_Mapping.dat", "hipblasltExtOpLibrary.dat")
            for name in names:
                (libraries / name).write_text(name)
            targets, metadata = bundle.sdk_metadata(root)
            self.assertEqual(targets, ["gfx1100", "gfx942"])
            self.assertEqual(len(metadata), 4)
            targets, metadata = bundle.sdk_metadata(root, "gfx1100")
            self.assertEqual(targets, ["gfx1100"])
            self.assertEqual(len(metadata), 3)
            self.assertIn("lib/hipblaslt/library/TensileLiteLibrary_lazy_Mapping.dat", metadata)
            with self.assertRaisesRegex(RuntimeError, "does not support"):
                bundle.sdk_metadata(root, "gfx950")

    def test_nested_sdk_metadata_and_compressed_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for target in ("gfx1100", "gfx942"):
                kernels = root / "lib/hipblaslt/library" / target
                kernels.mkdir(parents=True)
                (kernels / "TensileLibrary.dat.zlib").write_bytes(b"compressed-table")
                (kernels / "device.co").write_bytes(b"not-metadata")
            targets, metadata = bundle.sdk_metadata(root, "gfx942")
            self.assertEqual(targets, ["gfx942"])
            self.assertEqual(list(metadata), ["lib/hipblaslt/library/gfx942/TensileLibrary.dat.zlib"])


if __name__ == "__main__":
    unittest.main()
