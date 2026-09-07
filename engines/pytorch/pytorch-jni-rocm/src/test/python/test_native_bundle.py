"""Exercise the real JAR task with synthetic files; no native code is executed."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def properties(data):
    return dict(line.split("=", 1) for line in data.decode().splitlines() if line)


class NativeBundlePackagingTest(unittest.TestCase):
    def test_bundle_integrity_and_flavor_isolation(self):
        repo = Path(__file__).resolve().parents[6]
        gradle = repo / ("gradlew.bat" if os.name == "nt" else "gradlew")
        profiles = json.loads((repo / "engines/pytorch/pytorch-native/rocm/hipblaslt/profiles.json").read_text())
        self.assertEqual(set(profiles), {"rocm6.4", "rocm7.0", "rocm7.1", "rocm7.2", "rocm10.0"})
        with tempfile.TemporaryDirectory(prefix="djl-native-bundle-") as temporary:
            root = Path(temporary)
            vendor = root / "vendor"
            (vendor / "licenses").mkdir(parents=True)
            native = root / "libdjl_torch.so"
            native.write_bytes(b"synthetic JNI library v1")
            loader = root / "libdjl_rocm_loader.so"
            loader.write_bytes(b"synthetic runtime loader v1")
            notice = vendor / "licenses/hipblaslt-LICENSE.txt"
            notice.write_text("Synthetic packaging test notice\n", encoding="utf-8")
            metadata = {
                "msgpack_revision": "8c602e8579c7e7d65d6f9c6703c9699db3fb0488",
                "patch_sha256": "1" * 64,
                "classifier": "linux-x86_64",
                "sdk_library_sha256": "2" * 64,
                "kernel_metadata_sha256": "3" * 64,
                "gpu_targets": "gfx1100",
                "kernel_metadata.lib/hipblaslt/library/TensileLibrary_lazy_gfx1100.dat": "5" * 64,
            }

            def select_profile(flavor, sdk_version):
                profile = profiles[flavor]["sdk_versions"][sdk_version]
                metadata.update({key: profile[key] for key in ("source_repository", "source_revision", "hipblaslt_version", "soname")})
                metadata.update(flavor=flavor, rocm_version=sdk_version, library=profile["soname"])
                library = vendor / metadata["library"]
                library.write_bytes(b"synthetic hipBLASLt host library v1")
                return library

            vendor_library = select_profile("rocm10.0", "10.0.0")

            def save_vendor_metadata():
                metadata["sha256"] = sha256(vendor_library.read_bytes())
                (vendor / "hipblaslt.properties").write_text(
                    "".join(f"{key}={value}\n" for key, value in sorted(metadata.items())),
                    encoding="utf-8",
                )

            save_vendor_metadata()
            init_script = root / "isolated-build.gradle"
            output = root / "build"
            init_script.write_text(
                "allprojects { if (path == ':engines:pytorch:pytorch-jni-rocm') "
                f"{{ layout.buildDirectory.set(file('{output.as_posix()}')); "
                "tasks.withType(Jar).configureEach { outputs.upToDateWhen { false } } } }\n",
                encoding="utf-8",
            )

            def build(flavor="rocm10.0", succeeds=True):
                result = subprocess.run(
                    [
                        str(gradle),
                        ":engines:pytorch:pytorch-jni-rocm:jar",
                        "--offline",
                        "--console=plain",
                        "--init-script",
                        str(init_script),
                        "-Ppt_version=2.11.0",
                        f"-Pflavor={flavor}",
                        "-Pclassifier=linux-x86_64",
                        f"-Pjni_library={native}",
                        f"-Phipblaslt_bundle_dir={vendor}",
                        f"-Procm_loader_library={loader}",
                    ],
                    cwd=repo,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                if not succeeds:
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    return result.stdout
                self.assertEqual(result.returncode, 0, result.stdout)
                artifact = next((output / "libs").glob(f"pytorch-jni-{flavor}-linux-x86_64-*.jar"))
                with zipfile.ZipFile(artifact) as jar:
                    entries = {name: jar.read(name) for name in jar.namelist() if not name.endswith("/")}
                return artifact.read_bytes(), entries

            def assert_bundle(entries):
                prefix = f"jnilib/linux-x86_64/{metadata['flavor']}/"
                manifest_data = entries[prefix + "native-bundle.properties"]
                manifest = properties(manifest_data)
                self.assertEqual(manifest["libraries"], f"libdjl_torch.so,{vendor_library.name},{loader.name}")
                self.assertEqual(manifest["rocmLoaderLibrary"], loader.name)
                self.assertEqual(entries[prefix + loader.name], loader.read_bytes())
                self.assertEqual(manifest["hipblasltLibrary"], vendor_library.name)
                self.assertEqual(manifest["flavor"], metadata["flavor"])
                self.assertEqual(manifest["pytorchVersion"], "2.11.0")
                self.assertTrue(manifest["djlVersion"])
                self.assertEqual(manifest["hipblasltSoname"], metadata["soname"])
                self.assertEqual(manifest["rocmVersion"], metadata["rocm_version"])
                self.assertEqual(manifest["hipblasltRevision"], metadata["source_revision"])
                self.assertEqual(manifest["hipblasltSourceRepository"], metadata["source_repository"])
                self.assertEqual(manifest["hipblasltKernelMetadata.lib/hipblaslt/library/TensileLibrary_lazy_gfx1100.dat"], "5" * 64)
                for name in manifest["files"].split(","):
                    self.assertEqual(manifest[f"sha256.{name}"], sha256(entries[prefix + name]))
                self.assertEqual(
                    {name for name in entries if name.startswith("jnilib/")},
                    {"jnilib/pytorch.properties", prefix + "native-bundle.properties"}
                    | {prefix + name for name in manifest["files"].split(",")},
                    "Only the current flavor and its selected SONAME must enter the JAR",
                )
                cache_key = properties(entries["jnilib/pytorch.properties"])["jni_cache_key"]
                self.assertTrue(cache_key.endswith("-" + sha256(manifest_data)))
                return cache_key

            for flavor, family in profiles.items():
                for sdk_version in family["sdk_versions"]:
                    with self.subTest(flavor=flavor, sdk=sdk_version):
                        vendor_library = select_profile(flavor, sdk_version)
                        save_vendor_metadata()
                        assert_bundle(build(flavor)[1])

            vendor_library = select_profile("rocm10.0", "10.0.0")
            save_vendor_metadata()
            prefix = "jnilib/linux-x86_64/rocm10.0/"
            original_jar, entries = build()
            original_key = assert_bundle(entries)
            self.assertEqual(entries[prefix + native.name], native.read_bytes())
            self.assertEqual(entries[prefix + vendor_library.name], vendor_library.read_bytes())
            self.assertEqual(build()[0], original_jar, "Identical input must produce identical JAR bytes")

            self.assertIn("hipBLASLt bundle flavor mismatch", build("rocm7.2", succeeds=False))
            for key, invalid, message in (
                ("classifier", "linux-aarch64", "bundle classifier mismatch"),
                ("rocm_version", "7.2.0", "SDK version 7.2.0 is not supported for rocm10.0"),
                ("source_revision", "0" * 40, "bundle source_revision does not match"),
                ("soname", "libhipblaslt.so.999", "bundle soname does not match"),
                ("library", "../libhipblaslt.so.1", "Unexpected hipBLASLt library name"),
            ):
                with self.subTest(invalid_metadata=key):
                    original = metadata[key]
                    metadata[key] = invalid
                    save_vendor_metadata()
                    self.assertIn(message, build(succeeds=False))
                    metadata[key] = original
            save_vendor_metadata()

            native.write_bytes(b"synthetic JNI library v2")
            jni_key = assert_bundle(build()[1])
            self.assertNotEqual(jni_key, original_key)

            loader.write_bytes(b"synthetic runtime loader v2")
            loader_key = assert_bundle(build()[1])
            self.assertNotEqual(loader_key, jni_key)

            vendor_library.write_bytes(b"synthetic hipBLASLt host library v2")
            self.assertIn("hipBLASLt library does not match", build(succeeds=False))
            save_vendor_metadata()
            vendor_key = assert_bundle(build()[1])
            self.assertNotEqual(vendor_key, loader_key)

            notice.write_text("Changed synthetic notice\n", encoding="utf-8")
            notice_key = assert_bundle(build()[1])
            self.assertNotEqual(notice_key, vendor_key)

            metadata["sdk_library_sha256"] = "4" * 64
            save_vendor_metadata()
            self.assertNotEqual(assert_bundle(build()[1]), notice_key)

            _, cpu_entries = build("cpu")
            self.assertEqual(
                {name for name in cpu_entries if name.startswith("jnilib/")},
                {"jnilib/pytorch.properties", "jnilib/linux-x86_64/cpu/libdjl_torch.so"},
            )
            self.assertEqual(
                properties(cpu_entries["jnilib/pytorch.properties"])["jni_cache_key"].split("-")[-1],
                sha256(native.read_bytes()),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
