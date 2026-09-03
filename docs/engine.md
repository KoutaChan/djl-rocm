# Engines

The [Engine](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/engine/Engine.html) is one of
the most fundamental classes in DJL. Most of the core functionality in DJL including NDArrays,
NDManager, and Models are only interfaces. They form a tree of interfaces with the root as the Engine class.

The implementations of these interfaces are provided by the various engines using a Java service
loader. This means that DJL is able to take advantage of much of the performance optimization and
hardware support work which has gone on in these engines. As they are updated, DJL can take
advantage of the updates. And, DJL is able to freely switch between engines to keep up with
performance advancements.

In addition, the engines are very useful for production systems. Models trained in those engines
in python can often be imported and run in Java through DJL. This makes it much easier to integrate
into existing Java servers or any of the powerful Java production ecosystem. Because they are run
with the same engine they are trained in, there shouldn't be any loss in performance or accuracy
either.

For training in DJL, the choice of engine is less important. Any engine that fully implements
the DJL specification would have similar results. As you are encouraged to write engine agnostic
code, you can even switch between the engines as easily as switching dependencies. In general,
you should use the recommended engine (below) unless you have a good reason to use a different one.

## Supported Engines

Currently, the engines that are supported by DJL are:

- [MXNet](../engines/mxnet/README.md) - full support
- [PyTorch](../engines/pytorch/README.md) - full support
- [TensorFlow](../engines/tensorflow/README.md) - supports inference and NDArray operations
- [ONNX Runtime](../engines/onnxruntime/onnxruntime-engine/README.md) - supports basic inference
- [XGBoost](../engines/ml/xgboost/README.md) - supports basic inference
- [LightGBM](../engines/ml/lightgbm/README.md) - supports basic inference

## Setup

In order to choose an engine, it must be added into the Java classpath. Usually this means
additional Maven or Gradle dependencies. Many engines require multiple dependencies be added,
so look at the engine README for your desired engine to learn what dependencies are necessary.

It is also possible to load multiple engines simultaneously. When DJL starts up, it chooses a
default engine from the available engines. Most of the API that requires an engine such as
`NDManager.newBaseManager()` or `Model.newInstance()` will internally use the default engine.
DJL chooses the default engine based on a ranking of how much we recommend the engine, but before
that it was chosen at random. For those calls, you can also choose the engine manually by getting
an engine with `Engine.getEngine(engineName)` or calling the equivalent method such as
`NDManager.newBaseManager(engineName)`.

Some calls will also take advantage of all possible engines. For example, model loading will try
all of the engines available to see if any work for the model you are trying to load.

You can also choose the default engine manually. Each engine has a name which can be found in the
engine's javadoc or README. You can set the default engine by setting either the 
"DJL_DEFAULT_ENGINE" environment variable or the "ai.djl.default_engine" Java property. 
Either one should be set to the name of the desired default engine.

## Fusion storage planning

Engines that implement `FusionCompiler` may reduce accelerator memory by sharing command scratch,
reusing storage after a computed value's last consumer, and executing certified operations in place.
These optimizations are enabled by default. They preserve public output storage until its output
lease is released and never share storage between session buffer slots.

The following environment variables control the planner phases:

| Environment variable | Default | Description |
| --- | --- | --- |
| `DJL_FUSION_SCRATCH_PLANNER` | `true` | Shares command-local scratch storage according to its command-internal lifetime. |
| `DJL_FUSION_INTERMEDIATE_PLANNER` | `true` | Reuses computed intermediate storage after its final consumer. |
| `DJL_FUSION_INPLACE_PLANNER` | `true` | Hands dead intermediate storage to operations with a verified in-place execution path. Requires intermediate planning. |

Set a variable to `0` or `false` before the process starts to disable that phase for diagnostics or
performance comparison. Accepted enabled values are `1` and `true`; other values are rejected.
Disabling intermediate planning also disables in-place planning.

Planner statistics are available through `FusionPlan.getCompilationReport()`, including persistent,
workspace, exported-output, and arena bytes as well as logical allocation, backing allocation,
alias-view, and in-place reuse counts.

`getPersistentStorageBytes()` is the peak payload for an active slot. Only exported-output storage
is pinned for the full slot lifetime. Planner arenas are acquired on the submission stream and
released after work is enqueued, allowing the engine's device-wide caching allocator to reuse that
workspace across plans and sessions once stream work completes. Consequently, idle Fusion plans do
not each retain their reported arena capacity; concurrent submissions still receive independent
storage. When scratch and intermediate planning are both enabled, temporary tensors of different
data types share one alignment-safe byte arena according to their lifetimes.

The settings are engine-neutral. Currently, the native planner is implemented by the PyTorch CUDA
and ROCm Fusion backend. Other engines may adopt the same settings when they implement equivalent
storage lifetime guarantees.
