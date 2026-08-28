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
package ai.djl.training;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Autocast;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.nn.UninitializedParameterException;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.training.listener.EpochTrainingListener;
import ai.djl.training.listener.EvaluatorTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.function.Consumer;

/**
 * The {@code Trainer} interface provides a session for model training.
 *
 * <p>{@code Trainer} provides an easy, and manageable interface for training. {@code Trainer} is
 * not thread-safe.
 *
 * <p>See the tutorials on:
 *
 * <ul>
 *   <li><a
 *       href="https://docs.djl.ai/master/docs/demos/jupyter/tutorial/02_train_your_first_model.html">Training
 *       your first model</a>
 *   <li><a
 *       href="https://docs.djl.ai/master/docs/demos/jupyter/transfer_learning_on_cifar10.html">Training
 *       using transfer learning</a>
 *   <li><a href="https://docs.djl.ai/master/docs/demos/jupyter/load_mxnet_model.html">Inference
 *       with an MXNet model</a>
 * </ul>
 *
 * @see <a href="https://docs.djl.ai/master/docs/development/memory_management.html">The guide on
 *     memory management</a>
 */
@SuppressWarnings("try") // Autocast resources are used for their scope side effects.
public class Trainer implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(Trainer.class);

    private Model model;
    private NDManager manager;
    private Metrics metrics;
    private List<TrainingListener> listeners;
    private Device[] devices;
    private ParameterStore parameterStore;
    private List<Evaluator> evaluators;
    private Loss loss;
    private ExecutorService executorService;
    private DataType autocastDataType;
    private boolean autocastCacheEnabled;
    private GradScaler gradScaler;
    private final ThreadLocal<MixedPrecisionGradientCollector> activeMixedPrecisionCollector =
            new ThreadLocal<>();

    private boolean gradientsChecked;

    /**
     * Creates an instance of {@code Trainer} with the given {@link Model} and {@link
     * TrainingConfig}.
     *
     * @param model the model the trainer will train on
     * @param trainingConfig the configuration used by the trainer
     */
    @SuppressWarnings("this-escape")
    public Trainer(Model model, TrainingConfig trainingConfig) {
        this.model = model;
        manager = model.getNDManager().newSubManager();
        manager.setName("trainer");
        devices = trainingConfig.getDevices();
        loss = trainingConfig.getLossFunction();
        Objects.requireNonNull(loss, "You must specify a loss for the trainer");
        evaluators = new ArrayList<>(trainingConfig.getEvaluators());
        evaluators.add(loss); // track loss as an evaluator by default
        executorService = trainingConfig.getExecutorService();
        autocastDataType = trainingConfig.getAutocastDataType().orElse(null);
        if (autocastDataType != null
                && autocastDataType != DataType.FLOAT16
                && autocastDataType != DataType.BFLOAT16) {
            throw new IllegalArgumentException(
                    "Training autocast data type must be FLOAT16 or BFLOAT16.");
        }
        autocastCacheEnabled = trainingConfig.isAutocastCacheEnabled();
        gradScaler = trainingConfig.getGradScaler().orElse(null);
        if (autocastDataType == DataType.FLOAT16 && gradScaler == null) {
            gradScaler = GradScaler.builder().build();
        }

        ParameterServer parameterServer =
                manager.getEngine()
                        .newParameterServer(trainingConfig.getOptimizer(), trainingConfig);

        parameterStore = new ParameterStore(manager, false);
        parameterStore.setParameterServer(parameterServer, devices);
        initializeParameterStore(false);

        listeners = trainingConfig.getTrainingListeners();
        notifyListeners(listener -> listener.onTrainingBegin(this));
    }

    /**
     * Initializes the {@link Model} that the {@code Trainer} is going to train.
     *
     * @param shapes an array of {@code Shape} of the inputs
     */
    public void initialize(Shape... shapes) {
        model.getBlock().initialize(model.getNDManager(), model.getDataType(), shapes);
        initializeParameterStore(true);
    }

    /**
     * Returns how this trainer scopes {@link GradientCollector} instances.
     *
     * @return the {@link GradientCollectorMode}
     */
    public GradientCollectorMode getGradientCollectorMode() {
        return manager.getEngine().getGradientCollectorMode();
    }

    /**
     * Fetches an iterator that can iterate through the given {@link Dataset}.
     *
     * @param dataset the dataset to iterate through
     * @return an {@link Iterable} of {@link Batch} that contains batches of data from the dataset
     * @throws IOException for various exceptions depending on the dataset
     * @throws TranslateException if there is an error while processing input
     */
    public Iterable<Batch> iterateDataset(Dataset dataset) throws IOException, TranslateException {
        return dataset.getData(getManager(), executorService);
    }

    private void initializeParameterStore(boolean failOnUninitialized) {
        // Materialize parameter mirrors before the first training split so
        // multi-threaded training reads a stable set of device-local tensors.
        model.getBlock()
                .getParameters()
                .forEach(
                        pair -> {
                            for (Device device : devices) {
                                try {
                                    parameterStore.getValue(pair.getValue(), device, true);
                                } catch (UninitializedParameterException e) {
                                    if (failOnUninitialized) {
                                        throw new IllegalStateException(
                                                "Failed to initialize parameter: "
                                                        + pair.getKey()
                                                        + ".\n"
                                                        + "If you are defining a Block extending"
                                                        + " AbstractBlock, check that you are"
                                                        + " initializing all child blocks as part"
                                                        + " of the overload for"
                                                        + " AbstractBlock.initializeChildBlocks().",
                                                e);
                                    }
                                    break;
                                }
                            }
                        });
    }

    /**
     * Returns a new instance of {@link GradientCollector}.
     *
     * @return a new instance of {@link GradientCollector}
     */
    public GradientCollector newGradientCollector() {
        GradientCollector collector = manager.getEngine().newGradientCollector();
        if (autocastDataType == null && gradScaler == null) {
            return collector;
        }
        return new MixedPrecisionGradientCollector(collector);
    }

    /**
     * Opens this trainer's configured autocast scope for a device.
     *
     * <p>When autocast is disabled, the returned guard is a no-op. The guard must be opened on the
     * thread that executes the operations because backend autocast state is thread-local.
     *
     * @param device the device on which operations will execute
     * @return the configured autocast guard
     */
    public Autocast newAutocast(Device device) {
        if (autocastDataType == null) {
            return NoOpAutocast.INSTANCE;
        }
        return manager.getEngine().newAutocast(device, autocastDataType, autocastCacheEnabled);
    }

    /**
     * Applies the forward function of the model once on the given input {@link NDList}.
     *
     * @param input the input {@link NDList}
     * @return the output of the forward function
     */
    public NDList forward(NDList input) {
        long begin = System.nanoTime();
        try {
            parameterStore.prepareForForward();
            try (Autocast ignored = newAutocast(executionDevice(input))) {
                NDList output = model.getBlock().forward(parameterStore, input, true);
                parameterStore.prepareForBackward(output);
                return output;
            }
        } finally {
            addMetric("forward", begin);
        }
    }

    /**
     * Applies the forward function of the model once with both data and labels.
     *
     * @param data the input data {@link NDList}
     * @param labels the input labels {@link NDList}
     * @return the output of the forward function
     */
    public NDList forward(NDList data, NDList labels) {
        long begin = System.nanoTime();
        try {
            parameterStore.prepareForForward();
            try (Autocast ignored = newAutocast(executionDevice(data))) {
                NDList output = model.getBlock().forward(parameterStore, data, labels, null);
                parameterStore.prepareForBackward(output);
                return output;
            }
        } finally {
            addMetric("forward", begin);
        }
    }

    /**
     * Evaluates function of the model once on the given input {@link NDList}.
     *
     * @param input the input {@link NDList}
     * @return the output of the predict function
     */
    public NDList evaluate(NDList input) {
        parameterStore.prepareForForward();
        try (Autocast ignored = newAutocast(executionDevice(input))) {
            return model.getBlock().forward(parameterStore, input, false, null);
        }
    }

    /** Updates all of the parameters of the model once. */
    public void step() {
        MixedPrecisionGradientCollector collector = activeMixedPrecisionCollector.get();
        if (collector == null) {
            stepWithoutAutocast();
            return;
        }

        collector.closeAutocastScopes();
        try {
            stepWithoutAutocast();
        } catch (RuntimeException | Error e) {
            try {
                collector.openAutocastScopes();
            } catch (RuntimeException | Error openException) {
                e.addSuppressed(openException);
            }
            throw e;
        }
        collector.openAutocastScopes();
    }

    private void stepWithoutAutocast() {
        parameterStore.finalizeGradients(gradScaler != null);
        if (gradScaler == null) {
            try {
                if (!gradientsChecked) {
                    checkGradients();
                }
                long begin = System.nanoTime();
                parameterStore.updateAllParameters();
                addMetric("step", begin);
            } finally {
                parameterStore.releaseGradientStep();
            }
            return;
        }

        try (NDList gradients = parameterStore.getGradients()) {
            try {
                if (!gradientsChecked) {
                    checkGradients(gradients);
                }
                long begin = System.nanoTime();
                boolean gradientsFinite = gradScaler.unscale(gradients);
                if (!gradientsFinite) {
                    for (NDArray gradient : gradients) {
                        gradient.fillI(0);
                    }
                }
                if (gradientsFinite) {
                    parameterStore.updateAllParameters();
                }
                gradScaler.update();
                addMetric("step", begin);
            } finally {
                parameterStore.releaseGradientStep();
            }
        }
    }

    /** Returns the configured autocast data type. */
    public Optional<DataType> getAutocastDataType() {
        return Optional.ofNullable(autocastDataType);
    }

    /** Returns this trainer's gradient scaler. */
    public Optional<GradScaler> getGradScaler() {
        return Optional.ofNullable(gradScaler);
    }

    /**
     * Saves optimizer state.
     *
     * @param path the file to save optimizer state to
     * @throws IOException if failed to save optimizer state
     */
    public void saveOptimizerState(Path path) throws IOException {
        parameterStore.saveOptimizerState(path);
    }

    /**
     * Loads optimizer state.
     *
     * @param path the file to load optimizer state from
     * @throws IOException if failed to load optimizer state
     */
    public void loadOptimizerState(Path path) throws IOException {
        parameterStore.loadOptimizerState(path);
    }

    /**
     * Saves the configured {@link GradScaler} state.
     *
     * @param path the file to save scaler state to
     * @throws IllegalStateException if this trainer does not use a scaler or a step is active
     * @throws IOException if the state cannot be written
     */
    public void saveGradScalerState(Path path) throws IOException {
        requireGradScaler().saveState(path);
    }

    /**
     * Restores the configured {@link GradScaler} state.
     *
     * @param path the scaler state file to load
     * @throws IllegalStateException if this trainer does not use a scaler or a step is active
     * @throws IOException if the state is invalid, incompatible, or cannot be read
     */
    public void loadGradScalerState(Path path) throws IOException {
        requireGradScaler().loadState(path);
    }

    /**
     * Returns the Metrics param used for benchmarking.
     *
     * @return the the Metrics param used for benchmarking
     */
    public Metrics getMetrics() {
        return metrics;
    }

    /**
     * Attaches a Metrics param to use for benchmarking.
     *
     * @param metrics the Metrics class
     */
    public void setMetrics(Metrics metrics) {
        this.metrics = metrics;
    }

    /**
     * Returns the devices used for training.
     *
     * @return the devices used for training
     */
    public Device[] getDevices() {
        return devices;
    }

    /**
     * Gets the training {@link Loss} function of the trainer.
     *
     * @return the {@link Loss} function
     */
    public Loss getLoss() {
        return loss;
    }

    /**
     * Returns the model used to create this trainer.
     *
     * @return the model associated with this trainer
     */
    public Model getModel() {
        return model;
    }

    /**
     * Returns the {@link ExecutorService}.
     *
     * @return the {@link ExecutorService}
     */
    public Optional<ExecutorService> getExecutorService() {
        return Optional.ofNullable(executorService);
    }

    /**
     * Gets all {@link Evaluator}s.
     *
     * @return the evaluators used during training
     */
    public List<Evaluator> getEvaluators() {
        return evaluators;
    }

    /**
     * Executes a method on each of the {@link TrainingListener}s.
     *
     * @param listenerConsumer a consumer that executes the method
     */
    public final void notifyListeners(Consumer<TrainingListener> listenerConsumer) {
        listeners.forEach(listenerConsumer);
    }

    /**
     * Returns the {@link TrainingResult}.
     *
     * @return the {@code TrainingResult}
     */
    public TrainingResult getTrainingResult() {
        TrainingResult result = new TrainingResult();
        for (TrainingListener listener : listeners) {
            if (listener instanceof EpochTrainingListener) {
                result.setEpoch(((EpochTrainingListener) listener).getNumEpochs());
            } else if (listener instanceof EvaluatorTrainingListener) {
                EvaluatorTrainingListener l = (EvaluatorTrainingListener) listener;
                result.setEvaluations(l.getLatestEvaluations());
            }
        }
        return result;
    }

    /**
     * Gets the {@link NDManager} from the model.
     *
     * @return the {@link NDManager}
     */
    public NDManager getManager() {
        return manager;
    }

    /** {@inheritDoc} */
    @SuppressWarnings("deprecation")
    @Override
    protected void finalize() throws Throwable {
        if (manager.isOpen()) {
            if (logger.isDebugEnabled()) {
                logger.warn("Trainer for {} was not closed explicitly.", model.getName());
            }
            close();
        }
        super.finalize();
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        notifyListeners(listener -> listener.onTrainingEnd(this));

        try {
            parameterStore.sync();
        } finally {
            try {
                parameterStore.close();
            } finally {
                manager.close();
            }
        }
    }

    /**
     * Checks if all gradients are zeros. This prevent users from calling step() without running
     * {@code backward}.
     */
    private void checkGradients() {
        List<NDArray> grads = new ArrayList<>();
        model.getBlock().getParameters().values().stream()
                .filter(Parameter::requiresGradient)
                .forEach(
                        param ->
                                grads.add(
                                        parameterStore
                                                .getValue(param, devices[0], true)
                                                .getGradient()));

        try (NDManager scoped = manager.newSubManager()) {
            NDList gradients = new NDList(grads);
            scoped.tempAttachAll(gradients);
            checkGradients(gradients);
        }
    }

    private void checkGradients(NDList gradients) {
        boolean hasNonZeroGradient = false;
        for (NDArray gradient : gradients) {
            if (hasNonZeroGradient(gradient)) {
                hasNonZeroGradient = true;
                break;
            }
        }

        if (!hasNonZeroGradient) {
            throw new IllegalStateException(
                    "Gradient values are all zeros, please call gradientCollector.backward() on"
                            + "your target NDArray (usually loss), before calling step() ");
        }

        gradientsChecked = true;
    }

    private boolean hasNonZeroGradient(NDArray gradient) {
        if (gradient.isSparse()) {
            try (NDArray absoluteGradient = gradient.abs();
                    NDArray absoluteSum = absoluteGradient.sum()) {
                return absoluteSum.getFloat() != 0f;
            }
        }

        try (NDArray nonZero = gradient.neq(0);
                NDArray anyNonZero = nonZero.any()) {
            return anyNonZero.getBoolean();
        }
    }

    private Device executionDevice(NDList input) {
        if (input.isEmpty()) {
            return devices[0];
        }
        return input.head().getDevice();
    }

    private GradScaler requireGradScaler() {
        if (gradScaler == null) {
            throw new IllegalStateException("This trainer does not use a GradScaler.");
        }
        return gradScaler;
    }

    /**
     * Helper to add a metric for a time difference.
     *
     * @param metricName the metric name
     * @param begin the time difference start (this method is called at the time difference end)
     */
    public void addMetric(String metricName, long begin) {
        if (metrics != null && begin > 0L) {
            metrics.addMetric(metricName, System.nanoTime() - begin);
        }
    }

    private final class MixedPrecisionGradientCollector implements GradientCollector {

        private final GradientCollector delegate;
        private final Thread ownerThread;
        private final List<Autocast> autocastScopes;
        private boolean closed;

        private MixedPrecisionGradientCollector(GradientCollector delegate) {
            this.delegate = delegate;
            ownerThread = Thread.currentThread();
            autocastScopes = new ArrayList<>();
            try {
                openAutocastScopes();
            } catch (RuntimeException | Error e) {
                try {
                    delegate.close();
                } catch (RuntimeException | Error closeException) {
                    e.addSuppressed(closeException);
                }
                throw e;
            }
            if (activeMixedPrecisionCollector.get() != null) {
                closeAutocastScopes();
                delegate.close();
                throw new IllegalStateException(
                        "Nested mixed-precision GradientCollectors are not supported.");
            }
            activeMixedPrecisionCollector.set(this);
        }

        @Override
        public void backward(NDArray target) {
            validateThread();
            validateOpen();
            closeAutocastScopes();
            NDArray converted = null;
            NDArray scaled = null;
            try {
                NDArray backwardTarget = target;
                DataType dataType = target.getDataType();
                if (dataType == DataType.FLOAT16 || dataType == DataType.BFLOAT16) {
                    converted = target.toType(DataType.FLOAT32, false);
                    backwardTarget = converted;
                }
                if (gradScaler != null) {
                    scaled = gradScaler.scale(backwardTarget);
                    backwardTarget = scaled;
                }
                delegate.backward(backwardTarget);
            } finally {
                if (scaled != null) {
                    scaled.close();
                }
                if (converted != null) {
                    converted.close();
                }
                openAutocastScopes();
            }
        }

        @Override
        public void zeroGradients() {
            validateThread();
            validateOpen();
            delegate.zeroGradients();
        }

        @Override
        public void close() {
            validateThread();
            if (closed) {
                return;
            }
            closed = true;
            try {
                closeAutocastScopes();
            } finally {
                try {
                    delegate.close();
                } finally {
                    if (activeMixedPrecisionCollector.get() == this) {
                        activeMixedPrecisionCollector.remove();
                    }
                }
            }
        }

        private void openAutocastScopes() {
            if (autocastDataType == null || closed || !autocastScopes.isEmpty()) {
                return;
            }

            List<String> deviceTypes = new ArrayList<>();
            try {
                for (Device device : devices) {
                    if (!deviceTypes.contains(device.getDeviceType())) {
                        autocastScopes.add(newAutocast(device));
                        deviceTypes.add(device.getDeviceType());
                    }
                }
            } catch (RuntimeException | Error e) {
                closeAutocastScopes(e);
                throw e;
            }
        }

        private void closeAutocastScopes() {
            Throwable failure = closeAutocastScopes(null);
            if (failure instanceof RuntimeException) {
                throw (RuntimeException) failure;
            }
            if (failure != null) {
                throw (Error) failure;
            }
        }

        private Throwable closeAutocastScopes(Throwable failure) {
            for (int i = autocastScopes.size() - 1; i >= 0; --i) {
                try {
                    autocastScopes.get(i).close();
                } catch (RuntimeException | Error e) {
                    if (failure == null) {
                        failure = e;
                    } else {
                        failure.addSuppressed(e);
                    }
                }
            }
            autocastScopes.clear();
            return failure;
        }

        private void validateThread() {
            if (Thread.currentThread() != ownerThread) {
                throw new IllegalStateException(
                        "Mixed-precision GradientCollector can only be used from the thread that"
                                + " created it.");
            }
        }

        private void validateOpen() {
            if (closed) {
                throw new IllegalStateException(
                        "Mixed-precision GradientCollector has already been closed.");
            }
        }
    }

    private enum NoOpAutocast implements Autocast {
        INSTANCE;

        @Override
        public void close() {}
    }
}
