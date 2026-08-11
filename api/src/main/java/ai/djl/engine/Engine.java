/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.nn.SymbolBlock;
import ai.djl.training.GradientCollector;
import ai.djl.training.GradientCollectorMode;
import ai.djl.training.LocalParameterServer;
import ai.djl.training.ParameterServer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.util.Ec2Utils;
import ai.djl.util.RandomUtils;
import ai.djl.util.Utils;
import ai.djl.util.cuda.CudaUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.lang.management.MemoryUsage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.Properties;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;

/**
 * The {@code Engine} interface is the base of the provided implementation for DJL.
 *
 * <p>Any engine-specific functionality should be provided through this class. In general, it should
 * contain methods to detect information about the usable machine hardware and to create a new
 * {@link NDManager} and {@link Model}.
 *
 * @see <a href="https://docs.djl.ai/master/docs/engine.html">Engine Guide</a>
 * @see EngineProvider
 * @see <a href="https://docs.djl.ai/master/docs/development/cache_management.html">The guide on
 *     resource and engine caching</a>
 */
public abstract class Engine {

    private static final Logger logger = LoggerFactory.getLogger(Engine.class);

    private static final Map<String, EngineProvider> ALL_ENGINES = new ConcurrentHashMap<>();

    private static final String DEFAULT_ENGINE = initEngine();
    private static final Pattern PATTERN =
            Pattern.compile("KEY|TOKEN|PASSWORD", Pattern.CASE_INSENSITIVE);

    private Device defaultDevice;

    // use object to check if it's set
    private Integer seed;

    private static synchronized String initEngine() {
        ServiceLoader<EngineProvider> loaders = ServiceLoader.load(EngineProvider.class);
        for (EngineProvider provider : loaders) {
            registerEngine(provider);
        }

        if (ALL_ENGINES.isEmpty()) {
            logger.debug("No engine found from EngineProvider");
            return null;
        }

        String def = System.getProperty("ai.djl.default_engine");
        String defaultEngine = Utils.getenv("DJL_DEFAULT_ENGINE", def);
        boolean aarch64 = "aarch64".equals(System.getProperty("os.arch"));
        if (defaultEngine == null || defaultEngine.isEmpty()) {
            int rank = Integer.MAX_VALUE;
            for (EngineProvider provider : ALL_ENGINES.values()) {
                if (aarch64 && "MXNet".equals(provider.getEngineName())) {
                    // exclude MXNet on aarch64 machine
                    continue;
                }
                if (provider.getEngineRank() < rank) {
                    defaultEngine = provider.getEngineName();
                    rank = provider.getEngineRank();
                }
            }
        } else if (!ALL_ENGINES.containsKey(defaultEngine)) {
            throw new EngineException("Unknown default engine: " + defaultEngine);
        }
        logger.debug("Found default engine: {}", defaultEngine);
        Ec2Utils.callHome(defaultEngine);
        return defaultEngine;
    }

    /**
     * Returns the alternative {@code engine} if available.
     *
     * @return the alternative {@code engine}
     */
    public abstract Engine getAlternativeEngine();

    /**
     * Returns the name of the Engine.
     *
     * @return the name of the engine
     */
    public abstract String getEngineName();

    /**
     * Return the rank of the {@code Engine}.
     *
     * @return the rank of the engine
     */
    public abstract int getRank();

    /**
     * Returns the default Engine name.
     *
     * @return the default Engine name
     */
    public static String getDefaultEngineName() {
        return System.getProperty("ai.djl.default_engine", DEFAULT_ENGINE);
    }

    /**
     * Returns the default Engine.
     *
     * @return the instance of {@code Engine}
     * @see EngineProvider
     */
    public static Engine getInstance() {
        if (DEFAULT_ENGINE == null) {
            throw new EngineException(
                    "No deep learning engine found."
                            + System.lineSeparator()
                            + "Please refer to"
                            + " https://github.com/deepjavalibrary/djl/blob/master/docs/development/troubleshooting.md"
                            + " for more details.");
        }
        return getEngine(getDefaultEngineName());
    }

    /**
     * Returns if the specified engine is available.
     *
     * @param engineName the name of Engine to check
     * @return {@code true} if the specified engine is available
     * @see EngineProvider
     */
    public static boolean hasEngine(String engineName) {
        return ALL_ENGINES.containsKey(engineName);
    }

    /**
     * Registers a {@link EngineProvider} if not registered.
     *
     * @param provider the {@code EngineProvider} to be registered
     */
    public static void registerEngine(EngineProvider provider) {
        logger.debug("Registering EngineProvider: {}", provider.getEngineName());
        ALL_ENGINES.putIfAbsent(provider.getEngineName(), provider);
    }

    /**
     * Returns a set of engine names that are loaded.
     *
     * @return a set of engine names that are loaded
     */
    public static Set<String> getAllEngines() {
        return ALL_ENGINES.keySet();
    }

    /**
     * Returns the {@code Engine} with the given name.
     *
     * @param engineName the name of Engine to retrieve
     * @return the instance of {@code Engine}
     * @see EngineProvider
     */
    public static Engine getEngine(String engineName) {
        EngineProvider provider = ALL_ENGINES.get(engineName);
        if (provider == null) {
            throw new IllegalArgumentException("Deep learning engine not found: " + engineName);
        }
        return provider.getEngine();
    }

    /**
     * Returns the version of the deep learning engine.
     *
     * @return the version number of the deep learning engine
     */
    public abstract String getVersion();

    /**
     * Returns whether the engine has the specified capability.
     *
     * @param capability the capability to retrieve
     * @return {@code true} if the engine has the specified capability
     */
    public abstract boolean hasCapability(String capability);

    /**
     * Returns the engine's default {@link Device}.
     *
     * @return the engine's default {@link Device}
     */
    public Device defaultDevice() {
        if (defaultDevice == null) {
            if (hasCapability(StandardCapabilities.CUDA) && getGpuCount() > 0) {
                defaultDevice = Device.gpu();
            } else {
                defaultDevice = Device.cpu();
            }
        }
        return defaultDevice;
    }

    /**
     * Returns an array of devices.
     *
     * <p>If GPUs are available, it will return an array of {@code Device} of size
     * \(min(numAvailable, maxGpus)\). Else, it will return an array with a single CPU device.
     *
     * @return an array of devices
     */
    public Device[] getDevices() {
        return getDevices(Integer.MAX_VALUE);
    }

    /**
     * Returns an array of devices given the maximum number of GPUs to use.
     *
     * <p>If GPUs are available, it will return an array of {@code Device} of size
     * \(min(numAvailable, maxGpus)\). Else, it will return an array with a single CPU device.
     *
     * @param maxGpus the max number of GPUs to use. Use 0 for no GPUs.
     * @return an array of devices
     */
    public Device[] getDevices(int maxGpus) {
        int count = getGpuCount();
        if (maxGpus <= 0 || count <= 0) {
            return new Device[] {Device.cpu()};
        }

        count = Math.min(maxGpus, count);

        Device[] devices = new Device[count];
        for (int i = 0; i < count; ++i) {
            devices[i] = Device.gpu(i);
        }
        return devices;
    }

    /**
     * Returns the number of GPUs available in the system.
     *
     * @return the number of GPUs available in the system
     */
    public int getGpuCount() {
        if (hasCapability(StandardCapabilities.CUDA)) {
            return CudaUtils.getGpuCount();
        }
        return 0;
    }

    /**
     * Construct an empty SymbolBlock for loading.
     *
     * @param manager the manager to manage parameters
     * @return Empty {@link SymbolBlock} for static graph
     */
    public SymbolBlock newSymbolBlock(NDManager manager) {
        throw new UnsupportedOperationException("Not supported.");
    }

    /**
     * Constructs a new model.
     *
     * @param name the model name
     * @param device the device that the model will be loaded onto
     * @return a new Model instance using the network defined in block
     */
    public abstract Model newModel(String name, Device device);

    /**
     * Creates a new top-level {@link NDManager}.
     *
     * <p>{@code NDManager} will inherit default {@link Device}.
     *
     * @return a new top-level {@code NDManager}
     */
    public abstract NDManager newBaseManager();

    /**
     * Creates a new top-level {@link NDManager} with specified {@link Device}.
     *
     * @param device the default {@link Device}
     * @return a new top-level {@code NDManager}
     */
    public abstract NDManager newBaseManager(Device device);

    /**
     * Returns a new instance of {@link GradientCollector}.
     *
     * @return a new instance of {@link GradientCollector}
     */
    public GradientCollector newGradientCollector() {
        throw new UnsupportedOperationException("Not supported.");
    }

    /**
     * Returns how this engine scopes {@link GradientCollector} instances.
     *
     * @return the {@link GradientCollectorMode}
     */
    public GradientCollectorMode getGradientCollectorMode() {
        return GradientCollectorMode.GLOBAL_SINGLETON;
    }

    /**
     * Opens a thread-local inference scope that disables automatic-gradient recording.
     *
     * <p>The default implementation is a no-op so engine-agnostic inference code may always use
     * try-with-resources. Engines with thread-local gradient recording should override this method
     * and restore the previous state when the returned scope is closed.
     *
     * @return an {@link InferenceMode} scope
     */
    public InferenceMode newInferenceMode() {
        return NoOpInferenceMode.INSTANCE;
    }

    /**
     * Returns whether this engine implements automatic mixed-precision
     * ("autocast") scopes. When {@code false}, {@link #newAutocast} returns a
     * no-op guard, so callers can still wrap their forward pass in
     * {@code try-with-resources} without a feature gate.
     *
     * @return {@code true} if {@link #newAutocast} performs real autocasting
     */
    public boolean supportsAutocast() {
        return false;
    }

    /**
     * Opens an autocast scope on the given {@link Device}. Heavy matmul / conv
     * / attention ops inside the scope are cast to {@code dtype}; numerically
     * sensitive ops stay in FP32. The previous autocast state (enabled flag,
     * dtype, cache flag) is saved on entry and restored on {@link
     * Autocast#close()}, so scopes nest safely.
     *
     * <p>The default implementation is a no-op guard, which lets callers write
     * engine-agnostic code ({@code try (Autocast ac = engine.newAutocast(...))
     * { ... }}). Engines that implement autocast must override this method.
     *
     * @param device the device to autocast on (typically a GPU)
     * @param dtype the lower-precision dtype ({@link DataType#BFLOAT16} or
     *     {@link DataType#FLOAT16})
     * @param cacheEnabled whether to enable the op-result cache inside the
     *     scope (matches PyTorch's {@code cache_enabled} flag)
     * @return an {@link Autocast} guard whose {@code close()} restores state
     */
    public Autocast newAutocast(Device device, DataType dtype, boolean cacheEnabled) {
        throw new UnsupportedOperationException("Not supported.");
    }

    /**
     * Opens an autocast scope with the op-result cache enabled. Shortcut for
     * {@link #newAutocast(Device, DataType, boolean)} with {@code cacheEnabled
     * = true}.
     *
     * @param device the device to autocast on
     * @param dtype the lower-precision dtype
     * @return an {@link Autocast} guard
     */
    public Autocast newAutocast(Device device, DataType dtype) {
        return newAutocast(device, dtype, true);
    }

    private enum NoOpInferenceMode implements InferenceMode {
        INSTANCE;

        @Override
        public void close() {}
    }

    /**
     * Returns a new instance of {@link ParameterServer}.
     *
     * @param optimizer the optimizer to update
     * @return a new instance of {@link ParameterServer}
     */
    public ParameterServer newParameterServer(Optimizer optimizer) {
        return new LocalParameterServer(optimizer);
    }

    /**
     * Returns a new instance of {@link ParameterServer}.
     *
     * @param optimizer the optimizer to update
     * @param trainingConfig the training configuration
     * @return a new instance of {@link ParameterServer}
     */
    public ParameterServer newParameterServer(
            Optimizer optimizer, TrainingConfig trainingConfig) {
        return newParameterServer(optimizer);
    }

    /**
     * Seeds the random number generator in DJL Engine.
     *
     * <p>This will affect all {@link Device}s and all operators using Engine's random number
     * generator.
     *
     * @param seed the seed to be fixed in Engine
     */
    public void setRandomSeed(int seed) {
        this.seed = seed;
        RandomUtils.RANDOM.setSeed(seed);
    }

    /**
     * Returns the random seed in DJL Engine.
     *
     * @return seed the seed to be fixed in Engine
     */
    public Integer getSeed() {
        return seed;
    }

    /**
     * Returns the DJL API version.
     *
     * @return seed the seed to be fixed in Engine
     */
    public static String getDjlVersion() {
        String version = Engine.class.getPackage().getSpecificationVersion();
        if (version != null) {
            return version;
        }
        try (InputStream is = Engine.class.getResourceAsStream("api.properties")) {
            Properties prop = new Properties();
            prop.load(is);
            return prop.getProperty("djl_version");
        } catch (IOException e) {
            throw new AssertionError("Failed to open api.properties", e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return getEngineName() + ':' + getVersion();
    }

    /** Prints debug information about the environment for debugging environment issues. */
    @SuppressWarnings("PMD.SystemPrintln")
    public static void debugEnvironment() {
        System.out.println("----------- System Properties -----------");
        System.getProperties().forEach((k, v) -> print((String) k, v));

        System.out.println();
        System.out.println("--------- Environment Variables ---------");
        Utils.getenv().forEach(Engine::print);

        System.out.println();
        System.out.println("-------------- Directories --------------");
        try {
            Path temp = Paths.get(System.getProperty("java.io.tmpdir"));
            System.out.println("temp directory: " + temp);
            Path tmpFile = Files.createTempFile("test", ".tmp");
            Files.delete(tmpFile);

            Path cacheDir = Utils.getCacheDir();
            System.out.println("DJL cache directory: " + cacheDir.toAbsolutePath());

            Path path = Utils.getEngineCacheDir();
            System.out.println("Engine cache directory: " + path.toAbsolutePath());
            Files.createDirectories(path);
            if (!Files.isWritable(path)) {
                System.out.println("Engine cache directory is not writable!!!");
            }
        } catch (Throwable e) {
            e.printStackTrace(System.out);
        }

        System.out.println();
        System.out.println("------------------ CUDA -----------------");
        int gpuCount = CudaUtils.getGpuCount();
        System.out.println("GPU Count: " + gpuCount);
        if (gpuCount > 0) {
            System.out.println("CUDA: " + CudaUtils.getCudaVersionString());
            System.out.println("ARCH: " + CudaUtils.getComputeCapability(0));
        }
        for (int i = 0; i < gpuCount; ++i) {
            Device device = Device.gpu(i);
            MemoryUsage mem = CudaUtils.getGpuMemory(device);
            System.out.println("GPU(" + i + ") memory used: " + mem.getCommitted() + " bytes");
        }

        System.out.println();
        System.out.println("----------------- Engines ---------------");
        System.out.println("DJL version: " + getDjlVersion());
        System.out.println("Default Engine: " + Engine.getInstance());
        System.out.println("Default Device: " + Engine.getInstance().defaultDevice());
        for (EngineProvider provider : ALL_ENGINES.values()) {
            System.out.println(provider.getEngineName() + ": " + provider.getEngineRank());
        }
    }

    private static void print(String key, Object value) {
        if (PATTERN.matcher(key).find()) {
            value = "*********";
        }
        System.out.println(key + ": " + value); // NOPMD
    }
}
