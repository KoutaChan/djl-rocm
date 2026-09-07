# Bounded hipBLASLt solution cache

This patch targets **hipBLASLt 1.4.1**, ROCm/rocm-libraries revision
[`8d1ae90eff7d022f26019ec55b2ec6a7674b3112`](https://github.com/ROCm/rocm-libraries/tree/8d1ae90eff7d022f26019ec55b2ec6a7674b3112/projects/hipblaslt).
It rebuilds the host library and reuses the matching SDK's existing Tensile
kernel files. PyTorch, model parameters, and device kernels are unchanged.

## Why both cache layers need bounds

DJL's ROCm matmul path now separates compact, exact-geometry algorithm results
from reusable matrix descriptor families. That removes heavyweight hipBLASLt
descriptors for every observed row count and batch shape.

The matching vendor library also retains each complete `ContractionProblemGemm`
in its process-wide `CachingLibrary`. `findTopSolutions` inserts the problem
into both the solution-vector cache and the all-solutions flag cache. Those
copies survive descriptor destruction. The loader installs the caching wrapper
unconditionally, and this revision exposes no cache-capacity or bypass setting.

Relevant upstream sources:

- [CacheMap and findTopSolutions](https://github.com/ROCm/rocm-libraries/blob/8d1ae90eff7d022f26019ec55b2ec6a7674b3112/projects/hipblaslt/tensilelite/include/Tensile/CachingLibrary.hpp)
- [Library loader](https://github.com/ROCm/rocm-libraries/blob/8d1ae90eff7d022f26019ec55b2ec6a7674b3112/projects/hipblaslt/tensilelite/include/Tensile/Serialization/SolutionLibrary.hpp)

The patch limits each vendor `CacheMap` to **4096 outer problem keys** with LRU
eviction. Hardware-specific values remain grouped under their problem key.
The owning hash map stores the problem once; the recency list holds pointers
to those stable keys. The existing mutex protects map and recency changes,
and returned values are copied while locked so eviction cannot invalidate them.
Solutions and their completion flag share one cache entry, so concurrent access
or eviction cannot separate them. Completion is derived from the current
result count rather than another caller's shared last-result flag.

Eviction changes only whether an exact heuristic result is cached. It does not
change the heuristic, ranking, supported algorithms, arithmetic, or GPU work.
DJL retains a much larger bounded cache of compact algorithm results, so its
repeated shapes do not need to repeat vendor selection after vendor eviction.

## Build

Use a separate checkout of the pinned vendor revision:

```bash
git clone --filter=blob:none --no-checkout https://github.com/ROCm/rocm-libraries.git rocm-libraries
git -C rocm-libraries sparse-checkout init --cone
git -C rocm-libraries sparse-checkout set projects/hipblaslt shared/origami
git -C rocm-libraries checkout 8d1ae90eff7d022f26019ec55b2ec6a7674b3112

ROCM_PATH=/path/to/matching/sdk bash build.sh \
    /absolute/path/to/rocm-libraries /absolute/path/to/hipblaslt-host-build
```

`build.sh` verifies the revision, applies the patch once, and builds only the
host library. It does not install or replace the SDK. `GPU_TARGET` defaults to
`gfx1100`; `BUILD_JOBS` defaults to `8`. Additional arguments are forwarded to
CMake. The default host build disables RocRoller and profiling markers; the
validated gfx1100 path uses the SDK's Tensile kernels. Builds needing other
backends can override the corresponding CMake options and provide their
dependencies.

The script fetches the header-only MessagePack C++ 6.1.0 dependency at revision
`8c602e8579c7e7d65d6f9c6703c9699db3fb0488` into the build directory's `deps`
subdirectory and installs its headers and CMake package there. Boost, tests,
examples, and documentation are disabled. No system packages or SDK files are
changed. Set `HIPBLASLT_DEPS_DIR` to reuse a separate private dependency directory.
The build also exposes the SDK headers to Origami's standalone C++ header
checks, which intentionally omit normal target compiler flags.

Run validation against the private library before selecting it for an
application. Point `HIPBLASLT_TENSILE_LIBPATH` at the matching SDK's existing
`lib/hipblaslt/library/gfx1100` directory. Keep that SDK's other runtime
libraries available through the existing launch environment.
Place the rebuilt library and its SONAME symlink in `PYTORCH_LIBRARY_PATH`.
The ROCm 10 loader prefers libraries in that directory and preloads hipBLASLt
before hipBLAS/rocBLAS, preventing their SDK RPATH from selecting another copy.
Verify the running process maps only the intended hipBLASLt file before
accepting memory or throughput results.

## Regression coverage

The patch adds `CacheMap_test.cpp` to the vendor test target. It checks the
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
