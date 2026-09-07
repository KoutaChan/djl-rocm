# PyTorch native JAR packaging

This module packages an already built DJL JNI library. For Linux x86-64 ROCm
6.4, 7.0, 7.1, 7.2, and 10.0, packaging also prepares the bounded-cache hipBLASLt
host library automatically. Each flavor builds from its matching upstream SDK
source; libraries are never shared between SDK versions. Supported patch-level
SDK releases are pinned in [`profiles.json`](../pytorch-native/rocm/hipblaslt/profiles.json).
These libraries travel in the same Maven artifact:

```text
pytorch-jni-rocm10.0-linux-x86_64-<version>.jar
└── jnilib/
    ├── pytorch.properties
    └── linux-x86_64/rocm10.0/
        ├── libdjl_torch.so
        ├── libhipblaslt.so.<SDK SONAME>
        ├── libdjl_rocm_loader.so
        ├── native-bundle.properties
        └── licenses/
```

## Build

After building the JNI library with `pytorch-native/build.sh`, run:

```bash
ROCM_PATH=/path/to/matching/sdk ./gradlew :engines:pytorch:pytorch-jni-rocm:jar \
    -Ppt_version=2.11.0 -Pflavor=rocm10.0 -Pclassifier=linux-x86_64
```

The `prepareHipblaslt` dependency invokes
[`prepare-bundle.sh`](../pytorch-native/rocm/hipblaslt/prepare-bundle.sh), which reuses
its validated build cache. The native build also prepares this bundle, so the
subsequent JAR task normally only validates and reuses it. No SDK files are
replaced. Consumers receive the patched host library through their usual DJL
dependency; they do not run the patch or build scripts. The matching ROCm SDK
and its existing GPU kernels remain runtime requirements.

The artifact's flavor names the SDK's major/minor line. Each individual JAR is
bound to the exact SDK version recorded as `rocmVersion` in its manifest. For
example, a `rocm7.2` JAR built for SDK 7.2.0 cannot be used with SDK 7.2.2. The
7.2.1 and 7.2.2 profiles enable builds against those SDKs; they are not additional
runtimes hidden inside a 7.2.0 JAR. The current CI matrix builds one SDK per
flavor, and consumers need the artifact built for their matching SDK release.

Explicit inputs support packaging on another machine or CI artifact stages:

- `-Pjni_library=/path/to/libdjl_torch.so` selects an already built JNI library.
- `-Phipblaslt_bundle_dir=/path/to/rocm-runtime` selects the complete output of
  `prepare-bundle.sh` and skips its build. Library hashes and required metadata
  are still checked before packaging.
- `-Procm_loader_library=/path/to/libdjl_rocm_loader.so` supplies the small runtime
  loader. Normally `prepareRocmLoader` compiles it automatically using the active
  JDK and C compiler; it does not require a ROCm compiler or GPU.

Without the explicit JNI input, the ROCm native build identity must match the
requested PyTorch version, flavor, classifier, and JNI bytes. A stale native
build cannot silently enter another flavor's artifact. The staging root is
replaced for every packaging operation, so switching flavors also removes old
bundle files. Supplied vendor bundles must also match the requested flavor,
platform, SDK version, pinned source revision, hipBLASLt version, and SONAME.
CPU, CUDA, macOS, and Windows retain their existing JNI-only packaging.

## Integrity and cache identity

`native-bundle.properties` records the bundle schema, PyTorch version, platform,
flavor, vendor revision and patch, SDK compatibility metadata, and SHA-256 of
every bundled file, including redistribution notices. `jni_cache_key` in
`pytorch.properties` combines the DJL version with the manifest's SHA-256. A
change to any bundled library, a notice, or compatibility metadata therefore selects
a new runtime cache directory even if the Maven snapshot version is unchanged.
The manifest is sorted and timestamp-free, and the JAR uses reproducible entry
ordering and timestamps.

The loader checks and extracts this complete bundle before loading LibTorch,
then preloads the included hipBLASLt library before rocBLAS/hipBLAS. It does not
modify the SDK installation. It verifies the SDK's full version and Tensile
mapping hashes, and links the cache's `hipblaslt/library` directory to the SDK's
kernel directory. The canonical SDK and LibTorch paths are also part of the
cache identity, so installations cannot replace each other's runtime links.
No additional hipBLASLt environment variable is needed; existing explicit
Tensile path overrides still take precedence inside hipBLASLt.
The small CPU-only loader passes private symlink paths directly to `dlopen`.
This preserves the overlay's ELF `$ORIGIN`, so older LibTorch dependencies with
unversioned hipBLASLt names resolve to the patched library. SDK and LibTorch
libraries are linked into this overlay without copying their large files.
See the [vendor patch notes](../pytorch-native/rocm/hipblaslt/README.md)
for the patch's scope and native regression coverage.

## Packaging regression test

```bash
python engines/pytorch/pytorch-jni-rocm/src/test/python/test_native_bundle.py
```

This test runs the real Gradle JAR task in an isolated temporary build directory
with synthetic library bytes for every pinned SDK profile. It verifies payloads,
deterministic output, content-based invalidation, incompatible-bundle rejection,
corrupt-input rejection, and switching between ROCm and JNI-only packaging.
It does not load native code, require a GPU, or claim ABI or
numerical validation. Gradle dependencies must already be available locally;
the test uses `--offline` and never publishes an artifact.

The Linux CPU loader regression requires a JDK and a C compiler:

```bash
python engines/pytorch/pytorch-native/rocm/runtime-loader/test_runtime_loader.py
```

It builds small ELF fixtures and the real bootstrap. Separate JVMs demonstrate
that `System.load` resolves the symlink to the original dependency, while the
bootstrap loads only the patched dependency through the private overlay. It
also checks that a missing library reports `UnsatisfiedLinkError`. No ROCm SDK,
LibTorch runtime, GPU, or network access is needed. CI runs this together with
the producer's profile/metadata unit tests and the JAR packaging regression.
