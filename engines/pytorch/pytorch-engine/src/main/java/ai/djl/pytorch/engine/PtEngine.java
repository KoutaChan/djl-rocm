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
import java.util.Optional;

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

    private PtEngine() {}

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
    public ParameterServer newParameterServer(
            Optimizer optimizer, TrainingConfig trainingConfig) {
        Optional<DistributedTrainingConfig> config =
                trainingConfig.getDistributedTrainingConfig();
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
    public Autocast newAutocast(Device device, DataType dtype, boolean cacheEnabled) {
        return new PtAutocast(device, dtype, true, cacheEnabled);
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
