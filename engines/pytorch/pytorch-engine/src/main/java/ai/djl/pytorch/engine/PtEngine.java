/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.pytorch.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Autocast;
import ai.djl.engine.Engine;
import ai.djl.engine.EngineException;
import ai.djl.engine.InferenceMode;
import ai.djl.engine.fusion.FusionCompiler;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.nn.SymbolBlock;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.pytorch.jni.LibUtils;
import ai.djl.training.DistributedTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.GradientCollectorMode;
import ai.djl.training.ParameterServer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.util.Utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The {@code PtEngine} is an implementation of the {@link Engine} based on the <a
 * href="https://pytorch.org/">PyTorch Deep Learning Framework</a>.
 *
 * <p>To get an instance of the {@code PtEngine} when it is not the default Engine, call {@link
 * Engine#getEngine(String)} with the Engine name "PyTorch".
 */
public final class PtEngine extends Engine {

    private static final Logger logger = LoggerFactory.getLogger(PtEngine.class);

    public static final String ENGINE_NAME = "PyTorch";
    static final int RANK = 2;

    private final boolean useNativeInferenceMode;

    private PtEngine() {
        useNativeInferenceMode = JniUtils.getGpuCount() > 0;
    }

    @SuppressWarnings("PMD.AvoidRethrowingException")
    static Engine newInstance() {
        try {
            LibUtils.loadLibrary();
            JniUtils.setGradMode(false);
            if (Integer.getInteger("ai.djl.pytorch.num_interop_threads") != null) {
                JniUtils.setNumInteropThreads(
                        Integer.getInteger("ai.djl.pytorch.num_interop_threads"));
            }
            if (Integer.getInteger("ai.djl.pytorch.num_threads") != null) {
                JniUtils.setNumThreads(Integer.getInteger("ai.djl.pytorch.num_threads"));
            }
            // for ConvNN related model speed up
            if (Boolean.getBoolean("ai.djl.pytorch.cudnn_benchmark")) {
                JniUtils.setBenchmarkCuDNN(true);
            }
            if ("true".equals(System.getProperty("ai.djl.pytorch.graph_optimizer", "true"))) {
                logger.info(
                        "PyTorch graph executor optimizer is enabled, this may impact your"
                            + " inference latency and throughput. See:"
                            + " https://docs.djl.ai/master/docs/development/inference_performance_optimization.html#graph-executor-optimization");
            }
            logger.info("Number of inter-op threads is {}", JniUtils.getNumInteropThreads());
            logger.info("Number of intra-op threads is {}", JniUtils.getNumThreads());

            String paths = Utils.getEnvOrSystemProperty("PYTORCH_EXTRA_LIBRARY_PATH");
            if (paths != null) {
                String[] files = paths.split(",");
                for (String file : files) {
                    Path path = Paths.get(file);
                    if (Files.notExists(path)) {
                        throw new FileNotFoundException("PyTorch extra Library not found: " + file);
                    }
                    System.load(path.toAbsolutePath().toString()); // NOPMD
                }
            }
            return new PtEngine();
        } catch (EngineException e) {
            throw e;
        } catch (Throwable t) {
            throw new EngineException("Failed to load PyTorch native library", t);
        }
    }

    /** {@inheritDoc} */
    @Override
    public Engine getAlternativeEngine() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public String getEngineName() {
        return ENGINE_NAME;
    }

    /** {@inheritDoc} */
    @Override
    public int getRank() {
        return RANK;
    }

    /** {@inheritDoc} */
    @Override
    public String getVersion() {
        return LibUtils.getVersion();
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasCapability(String capability) {
        return JniUtils.getFeatures().contains(capability);
    }

    /** {@inheritDoc} */
    @Override
    public int getGpuCount() {
        return JniUtils.getGpuCount();
    }

    /**
     * Returns PyTorch caching allocator memory statistics for a GPU device.
     *
     * <p>The returned values describe memory managed by the PyTorch allocator, not total process or
     * device memory.
     *
     * @param device the GPU device
     * @return an immutable snapshot of allocator memory statistics
     * @throws IllegalArgumentException if the device is not a GPU device
     */
    public PtMemoryStats getMemoryStats(Device device) {
        if (!device.isGpu()) {
            throw new IllegalArgumentException("Memory statistics require a GPU device.");
        }
        return new PtMemoryStats(JniUtils.getMemoryStats(device.getDeviceId()));
    }

    /**
     * Resets PyTorch caching allocator peak memory statistics for a GPU device.
     *
     * <p>Each peak is reset to the corresponding current value. This method does not synchronize
     * the device or release cached memory.
     *
     * @param device the GPU device
     * @throws IllegalArgumentException if the device is not a GPU device
     */
    public void resetPeakMemoryStats(Device device) {
        if (!device.isGpu()) {
            throw new IllegalArgumentException("Memory statistics require a GPU device.");
        }
        JniUtils.resetPeakMemoryStats(device.getDeviceId());
    }

    /** {@inheritDoc} */
    @Override
    public SymbolBlock newSymbolBlock(NDManager manager) {
        return new PtSymbolBlock((PtNDManager) manager);
    }

    /** {@inheritDoc} */
    @Override
    public Model newModel(String name, Device device) {
        return new PtModel(name, device);
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager() {
        return PtNDManager.getSystemManager().newSubManager();
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager(Device device) {
        return PtNDManager.getSystemManager().newSubManager(device);
    }

    /** {@inheritDoc} */
    @Override
    public GradientCollector newGradientCollector() {
        return new PtGradientCollector();
    }

    /** {@inheritDoc} */
    @Override
    public GradientCollectorMode getGradientCollectorMode() {
        return GradientCollectorMode.THREAD_CONFINED;
    }

    /** {@inheritDoc} */
    @Override
    public InferenceMode newInferenceMode() {
        return new PtInferenceMode(useNativeInferenceMode);
    }

    /**
     * Opens a thread-local compute stream selected from PyTorch's stream pool.
     *
     * @param device accelerator device whose stream should become current
     * @return scope that restores the previous stream when closed
     */
    public PtStreamScope newStreamScope(Device device) {
        return new PtStreamScope(device);
    }

    /**
     * Creates a reusable device stream selected from PyTorch's stream pool.
     *
     * @param device device on which work is enqueued
     * @return a reusable device stream
     */
    public PtStream newStream(Device device) {
        return new PtStream(device);
    }

    /**
     * Creates a reusable device event.
     *
     * @param device device on which the event is recorded
     * @return a reusable device event
     */
    public PtEvent newEvent(Device device) {
        return new PtEvent(device);
    }

    /**
     * Creates a reusable accelerator graph for a fixed-shape workload.
     *
     * @param device accelerator device on which capture and replay execute
     * @return accelerator graph owned by the caller
     */
    public PtAcceleratorGraph newAcceleratorGraph(Device device) {
        return new PtAcceleratorGraph(device);
    }

    /** {@inheritDoc} */
    @Override
    public FusionCompiler newFusionCompiler(Device device) {
        return new PtFusionCompiler(device);
    }

    /** {@inheritDoc} */
    @Override
    public ParameterServer newParameterServer(Optimizer optimizer, TrainingConfig trainingConfig) {
        Optional<DistributedTrainingConfig> config = trainingConfig.getDistributedTrainingConfig();
        if (config.isPresent() && config.get().getWorldSize() > 1) {
            if (!hasCapability("NATIVE_DISTRIBUTED_NCCL")) {
                throw new EngineException(
                        "Native distributed training is not available in this PyTorch JNI build. "
                                + "Rebuild djl_torch with NCCL/RCCL development headers, or remove "
                                + "DistributedTrainingConfig to use the local parameter server.");
            }
            return new PtDistributedParameterServer(optimizer, config.get());
        }
        return super.newParameterServer(optimizer, trainingConfig);
    }

    /** {@inheritDoc} */
    @Override
    public boolean supportsAutocast() {
        return true;
    }

    /** {@inheritDoc} */
    @Override
    public Autocast newAutocast(Device device, DataType dataType, boolean cacheEnabled) {
        return new PtAutocast(device, dataType, true, cacheEnabled);
    }

    /** {@inheritDoc} */
    @Override
    public boolean unscaleGradients(NDList gradients, float inverseScale) {
        if (inverseScale <= 0f || !Float.isFinite(inverseScale)) {
            throw new IllegalArgumentException("inverseScale must be positive and finite.");
        }

        Map<Device, Map<DataType, List<PtNDArray>>> grouped = new ConcurrentHashMap<>();
        for (NDArray gradient : gradients) {
            if (!(gradient instanceof PtNDArray)) {
                throw new IllegalArgumentException(
                        "PyTorch GradScaler requires PyTorch gradient arrays.");
            }
            switch (gradient.getDataType()) {
                case FLOAT16:
                case BFLOAT16:
                case FLOAT32:
                case FLOAT64:
                    break;
                default:
                    throw new IllegalArgumentException(
                            "GradScaler gradients must use a real floating data type.");
            }
            grouped.computeIfAbsent(gradient.getDevice(), key -> new EnumMap<>(DataType.class))
                    .computeIfAbsent(gradient.getDataType(), key -> new ArrayList<>())
                    .add((PtNDArray) gradient);
        }

        boolean gradientsFinite = true;
        for (Map<DataType, List<PtNDArray>> byDataType : grouped.values()) {
            for (List<PtNDArray> group : byDataType.values()) {
                boolean groupFinite = JniUtils.unscaleGradients(group, inverseScale);
                gradientsFinite &= groupFinite;
            }
        }
        return gradientsFinite;
    }

    /** {@inheritDoc} */
    @Override
    public void setRandomSeed(int seed) {
        super.setRandomSeed(seed);
        JniUtils.setSeed(seed);
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(200);
        sb.append(getEngineName()).append(':').append(getVersion()).append(", capabilities: [\n");
        for (String feature : JniUtils.getFeatures()) {
            sb.append("\t").append(feature).append(",\n"); // NOPMD
        }
        sb.append("]\nPyTorch Library: ").append(LibUtils.getLibtorchPath());
        return sb.toString();
    }
}
