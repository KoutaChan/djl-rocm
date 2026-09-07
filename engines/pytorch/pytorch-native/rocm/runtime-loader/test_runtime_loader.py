"""Prove private ELF overlays survive JNI loading, using only CPU fixtures."""

import os
import platform
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@unittest.skipUnless(platform.system() == "Linux", "ELF overlays require Linux")
class RuntimeLoaderTest(unittest.TestCase):
    def test_overlay_and_canonicalization_control(self):
        def run(*args):
            subprocess.run(args, check=True)

        with tempfile.TemporaryDirectory(prefix="djl-loader-test-") as temporary:
            work = Path(temporary)
            for name in ("original", "patched", "overlay", "classes"):
                (work / name).mkdir()
            answer = work / "answer.c"
            answer.write_text("int answer(void) { return VALUE; }\n")
            consumer = work / "consumer.c"
            consumer.write_text(
                "extern int answer(void); int call_answer(void) { return answer(); }\n"
            )
            compiler = os.environ.get("CC", "cc")
            basename = "libdjl_loader_sample.so"
            for directory, soname, value in (
                ("original", basename, "2"),
                ("patched", basename + ".1", "1"),
            ):
                run(
                    compiler,
                    "-shared",
                    "-fPIC",
                    f"-DVALUE={value}",
                    f"-Wl,-soname,{soname}",
                    str(answer),
                    "-o",
                    str(work / directory / soname),
                )
            (work / "patched" / basename).symlink_to(basename + ".1")
            run(
                compiler,
                "-shared",
                "-fPIC",
                str(consumer),
                "-L" + str(work / "original"),
                "-ldjl_loader_sample",
                "-Wl,-rpath,$ORIGIN/original",
                "-o",
                str(work / "libconsumer.so"),
            )
            (work / "overlay/libconsumer.so").symlink_to(work / "libconsumer.so")
            (work / "overlay/original").symlink_to(work / "patched")
            run("bash", str(ROOT / "build.sh"), str(work / "libdjl_rocm_loader.so"))
            loader = (
                ROOT.parents[2]
                / "pytorch-engine/src/main/java/ai/djl/pytorch/jni/RocmLibraryLoader.java"
            )
            run(
                "javac",
                "-d",
                str(work / "classes"),
                str(loader),
                str(ROOT / "tests/RuntimeLoaderProbe.java"),
            )
            # Separate JVMs prevent the negative control's loaded original
            # library from contaminating the private-loader verification.
            for mode in ("canonical", "overlay"):
                run(
                    "java",
                    "-cp",
                    str(work / "classes"),
                    "ai.djl.pytorch.jni.RuntimeLoaderProbe",
                    mode,
                    str(work),
                    str(work / "patched" / basename),
                    str(work / "overlay/libconsumer.so"),
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
