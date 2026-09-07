# Bounded hipBLASLt solution cache

Each supported Linux x86_64 ROCm native JAR includes the patched host library
for its matching SDK and reuses that SDK's existing Tensile kernels.
`profiles.json` defines build and packaging compatibility:

| JAR flavor | SDK versions | hipBLASLt | Exact vendor revision | ELF SONAME |
| --- | --- | --- | --- | --- |
| `rocm6.4` | 6.4.0 | 0.12.0 | `5051e24def137fb02672e20bb8248305ae0cf784` | `libhipblaslt.so.0` |
| `rocm7.0` | 7.0.0 | 1.0.0 | `976b9c4a87a60a862ffb0e88e1fc0016e93c9961` | `libhipblaslt.so.1` |
| `rocm7.1` | 7.1.0 | 1.1.0 | `de5c1aebb641af098d9310a9fcca5591a7c066c8` | `libhipblaslt.so.1` |
| `rocm7.2` | 7.2.0 | 1.2.1 | `5b515cf1bca9959fb434c2414cf79b42fe25e93b` | `libhipblaslt.so.1` |
| `rocm7.2` | 7.2.1, 7.2.2 | 1.2.2 | `dabb6df2b988f8eabed1e2fecefaaf4e818bc7ef` | `libhipblaslt.so.1` |
| `rocm10.0` | 10.0.0 | 1.4.1 | `8d1ae90eff7d022f26019ec55b2ec6a7674b3112` | `libhipblaslt.so.1` |

ROCm 6.4 uses [ROCm/hipBLASLt](https://github.com/ROCm/hipBLASLt); the other
profiles use [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries).
The producer verifies the installed version header, including its source
revision suffix, and actual ELF SONAME before compiling. In particular, the
standalone repository's `rocm-7.0.0` tag is not the source shipped by SDK 7.0.0.
An unlisted SDK patch release fails with its version instead of receiving
another release's binary; add and verify its profile before publishing it.

## Cache behavior

DJL separates compact exact-geometry algorithm results from reusable matrix
descriptor families. The vendor's process-wide `CachingLibrary` also owns
complete `ContractionProblemGemm` keys, so bounding DJL descriptors alone
cannot bound host memory retention.

All profile patches limit each `CacheMap` to **4096 outer problem keys** with
LRU eviction. Hardware-specific values remain grouped under their problem
key. The owning hash map stores each problem once; the recency list holds
stable pointers to those keys. The existing mutex protects lookup, insertion
and eviction; returned values remain valid after eviction.

ROCm 10's solutions and completion flag share one entry so concurrent access
or eviction cannot separate them. Completion comes from the current result
count. ROCm 6.4 and 7.x have no separate completion-flag cache; their patch
changes only `CacheMap` retention and preserves their lookup contract.
Repeated insertion in those versions keeps the first cached value, as in
the original SDK, while refreshing its recency.

Eviction changes whether an exact heuristic result is cached. It does not
change ranking, supported algorithms, arithmetic, model parameters or device
kernels. DJL's larger bounded cache of compact algorithm results avoids
repeating vendor selection for its common shapes.

## Build and distribution

All listed Linux x86_64 ROCm flavors automatically prepare the vendor bundle
from the native build and again as a dependency of JAR packaging:

```bash
export ROCM_PATH=/path/to/matching/sdk
engines/pytorch/pytorch-native/build.sh 2.11.0 rocm7.2 cxx11 amd64
./gradlew :engines:pytorch:pytorch-jni-rocm:jar \
    -Ppt_version=2.11.0 -Pflavor=rocm7.2 -Pclassifier=linux-x86_64
```

For only the host runtime:

```bash
ROCM_PATH=/path/to/matching/sdk \
    engines/pytorch/pytorch-native/rocm/hipblaslt/prepare-bundle.sh \
    /absolute/output/directory --flavor rocm7.2
```

The producer stages the library under its actual SONAME, provenance, and
required license notices. The SDK is never overwritten. The JAR loader
extracts the private runtime and resolves the matching SDK kernel files;
users do not apply patches, rebuild libraries, or configure Tensile paths.
The corresponding SDK is still required. The multi-gigabyte GPU kernel set
is not duplicated in the JAR.

ROCm 6.4 and 7.0 use separate host-only CMake patches. They bypass Python and
Tensile/device generation while compiling the original host sources,
including extension and matrix-transform APIs. ROCm 7.1 onward has native
host-only switches. When the installed hipBLASLt depends on RocRoller or
profiling markers, the build preserves these dependencies and links the
SDK's existing runtime. The RocRoller development interface requires fmt
11.1.3 and yaml-cpp 0.8.0; pinned private copies satisfy this interface when
needed. yaml-cpp remains static, and dependency notices are included.

The producer supports flat ROCm 6/7 kernel layouts and architecture-specific
ROCm 10 directories. It records hashes for metadata, compressed tables, and
global mapping tables. By default it retains every architecture present in
the SDK metadata. `GPU_TARGET` explicitly selects a semicolon-separated
subset; the JNI's architecture list is not narrowed. `BUILD_JOBS` defaults
to 8. CMake 3.25.2 or newer and the matching development SDK are required.

`HIPBLASLT_CACHE_DIR` selects the private source/build/artifact cache, default
`pytorch-native/build/hipblaslt-cache`. A hit requires matching profile,
patches, scripts, compiler, SDK library, metadata, build flags, and output
hashes. A process lock serializes preparation; the completion manifest is
published after the referenced files. SDK discovery also supports
`ROCM_HOME`, `rocm-sdk path --root`, and `/opt/rocm`.

Source fetching configures Git's promisor remote before sparse checkout,
selects only host sources and required build files, and excludes generated
kernel logic. It does not fetch or cache the entire rocm-libraries monorepo.
MessagePack C++ 6.1.0 uses pinned revision
`8c602e8579c7e7d65d6f9c6703c9699db3fb0488`; fmt 11.1.3 uses
`9cf9f38eded63e5e0fb95cd536ba51be601d7fa2`; yaml-cpp 0.8.0 uses
`f7320141120f720aecc4c32be25586e7da9eb978`. All are built privately.

CPU producer checks cover exact SDK/source selection, cross-flavor rejection,
and both kernel metadata layouts:

```bash
python3 -m unittest discover -s engines/pytorch/pytorch-native/rocm/hipblaslt
```

## GitHub Actions build time

Before compiler caching was added,
[run 33941245727](https://github.com/KoutaChan/djl-rocm/actions/runs/33941245727)
took 52m47s overall. ROCm 7.2's native compilation alone took 37m12s on
two CPUs. ROCm 10 spent 9m56s compiling, about 2m35s installing/initializing
the SDK, 2m56s in Gradle, and 3m11s in disk cleanup.

The Linux matrix now uses pinned ccache 4.13.3 with compiler-content checks,
1 GiB per ROCm flavor (512 MiB per CPU/CUDA flavor), and a shared container
Gradle cache. The small patched hipBLASLt host artifact has its own cache;
multi-GiB SDK wheels and device kernels are not duplicated in Actions caches.
Pip's redundant download cache is disabled, CPU jobs skip accelerator disk
cleanup, and the container no longer recursively changes ownership throughout
the workspace and libtorch tree. CMake retains compatible local objects instead
of deleting its build tree every invocation; target/SDK changes invalidate it.

`DJL_BUILD_PHASE` timings and ccache statistics are emitted on each run. These
changes primarily improve repeated builds. A new SDK/toolchain or changed kernel
still requires compilation; no GPU target, optimization level or distributed
training feature was removed to reduce build time. A post-change Actions run
is required before assigning a measured overall speedup.

## Regression coverage

The ROCm 10 patch adds `CacheMap_test.cpp` to the vendor test target. It checks the
4096-entry bound, recency updates on hits and writes, shared capacity across
hardware keys, returned-value lifetime after eviction, concurrent access,
and consistent solution/completion snapshots through eviction and updates.
With the upstream test dependencies installed:

```bash
ROCM_PATH=/path/to/matching/sdk bash build.sh \
    /absolute/path/to/rocm-libraries /absolute/path/to/hipblaslt-host-build \
    -DTENSILELITE_BUILD_TESTING=ON -DBUILD_TESTING=ON
cmake --build /absolute/path/to/hipblaslt-host-build --target tensilelite-tests
ctest --test-dir /absolute/path/to/hipblaslt-host-build \
    --output-on-failure -R '^CacheMapTest\.'
```

DJL's `PtRocmMatmulCacheTest` separately covers changing matrix geometries,
descriptor eviction, multiple streams, and graph replay. Memory and throughput
comparisons must use the same workload and distinguish initial cache warmup
from repeated steady-state rounds.

On two RX 7900 XTX GPUs with ROCm 10, three consecutive 32,768-game epsilon
Decision collections measured the following post-collection native allocation:

| Runtime | Round 1 | Round 2 | Round 3 | Combined games/s |
| --- | ---: | ---: | ---: | ---: |
| Original | 6.152 GiB | 8.599 GiB | 10.347 GiB | 287.60 |
| Bounded DJL and vendor caches | 0.990 GiB | 1.006 GiB | 1.025 GiB | 287.76 |

Allocation means glibc `mallinfo2().uordblks + hblkhd`, measured after each
collection closes its inference handles. It is not total RSS, Java heap, or
VRAM. No forced GC or allocator trimming was used between rounds. Each DJL
context retained 48 descriptor families; its algorithm cache reached the
262,144-entry limit and evicted older entries. The CPU cache tests passed all
seven cases, and the GPU regressions passed all five cases without skips.
