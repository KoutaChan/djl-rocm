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
package ai.djl.training.optimizer;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/**
 * An {@code Optimizer} updates the weight parameters to minimize the loss function. {@code
 * Optimizer} is an abstract class that provides the base implementation for optimizers.
 *
 * @see <a href="https://d2l.djl.ai/chapter_optimization/index.html">The D2L chapters on
 *     optimization algorithms</a>
 */
public abstract class Optimizer {

    private static final String STATE_MAGIC = "DJL_OPTIMIZER_STATE";
    private static final int STATE_VERSION = 1;

    protected float rescaleGrad;
    protected float clipGrad;
    private float weightDecays;
    private int beginNumUpdate;
    private int numUpdate;
    private Map<String, Integer> updateCounts = new ConcurrentHashMap<>();

    /**
     * Creates a new instance of {@code Optimizer}.
     *
     * @param builder the builder used to create an instance of {@code Optimizer}
     */
    public Optimizer(OptimizerBuilder<?> builder) {
        this.rescaleGrad = builder.rescaleGrad;
        this.weightDecays = builder.weightDecays;
        this.clipGrad = builder.clipGrad;
        this.beginNumUpdate = builder.beginNumUpdate;
    }

    /**
     * Returns a new instance of {@link ai.djl.training.optimizer.Sgd.Builder} that can build an
     * {@link Sgd} optimizer.
     *
     * @return the {@link Sgd} {@link ai.djl.training.optimizer.Sgd.Builder}
     */
    public static Sgd.Builder sgd() {
        return new Sgd.Builder();
    }

    /**
     * Returns a new instance of {@link ai.djl.training.optimizer.Nag.Builder} that can build an
     * {@link Nag} optimizer.
     *
     * @return the {@link Nag} {@link ai.djl.training.optimizer.Nag.Builder}
     */
    public static Nag.Builder nag() {
        return new Nag.Builder();
    }

    /**
     * Returns a new instance of {@link ai.djl.training.optimizer.Adam.Builder} that can build an
     * {@link Adam} optimizer.
     *
     * @return the {@link Adam} {@link ai.djl.training.optimizer.Adam.Builder}
     */
    public static Adam.Builder adam() {
        return new Adam.Builder();
    }

    /**
     * Returns a new instance of {@link ai.djl.training.optimizer.AdamW.Builder} that can build an
     * {@link AdamW} optimizer.
     *
     * @return the {@link AdamW} {@link ai.djl.training.optimizer.AdamW.Builder}
     */
    public static AdamW.Builder adamW() {
        return new AdamW.Builder();
    }

    /**
     * Returns a new instance of {@link RmsProp.Builder} that can build an {@link RmsProp}
     * optimizer.
     *
     * @return the {@link RmsProp} {@link RmsProp.Builder}
     */
    public static RmsProp.Builder rmsprop() {
        return new RmsProp.Builder();
    }

    /**
     * Returns a new instance of {@link ai.djl.training.optimizer.Adagrad.Builder} that can build an
     * {@link Adagrad} optimizer.
     *
     * @return the {@link Adagrad} {@link ai.djl.training.optimizer.Adagrad.Builder}
     */
    public static Adagrad.Builder adagrad() {
        return new Adagrad.Builder();
    }

    /**
     * Returns a new instance of {@link ai.djl.training.optimizer.Adadelta.Builder} that can build
     * an {@link Adadelta} optimizer.
     *
     * @return the {@link Adadelta} {@link ai.djl.training.optimizer.Adadelta.Builder}
     */
    public static Adadelta.Builder adadelta() {
        return new Adadelta.Builder();
    }

    /**
     * Gets the value of weight decay.
     *
     * @return the value of weight decay
     */
    protected float getWeightDecay() {
        return weightDecays;
    }

    protected int updateCount(String parameterId) {
        // if index exists, increment update count, if not, use begin number of update + 1
        int count =
                updateCounts.compute(
                        parameterId, (key, val) -> (val == null) ? beginNumUpdate + 1 : val + 1);
        numUpdate = Math.max(numUpdate, count);
        return numUpdate;
    }

    /**
     * Updates the parameters according to the gradients.
     *
     * @param parameterId the parameter to be updated
     * @param weight the weights of the parameter
     * @param grad the gradients
     */
    public abstract void update(String parameterId, NDArray weight, NDArray grad);

    /**
     * Saves this optimizer's state.
     *
     * <p>The saved state includes update counters and optimizer-specific tensor state such as Adam
     * moments. Hyperparameters are not saved and should be restored by constructing the same
     * optimizer configuration before loading the state.
     *
     * @param path the file to save the optimizer state to
     * @throws IOException if failed to save optimizer state
     */
    public void saveState(Path path) throws IOException {
        Path parent = path.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (DataOutputStream os = new DataOutputStream(Files.newOutputStream(path));
                NDList arrays = new NDList()) {
            os.writeUTF(STATE_MAGIC);
            os.writeInt(STATE_VERSION);
            os.writeUTF(getClass().getName());
            os.writeInt(beginNumUpdate);
            os.writeInt(numUpdate);
            os.writeInt(updateCounts.size());
            for (Map.Entry<String, Integer> entry : updateCounts.entrySet()) {
                os.writeUTF(entry.getKey());
                os.writeInt(entry.getValue());
            }

            for (String stateName : getStateNames()) {
                Map<String, Map<Device, NDArray>> state = getState(stateName);
                for (Map.Entry<String, Map<Device, NDArray>> parameterEntry : state.entrySet()) {
                    for (Map.Entry<Device, NDArray> deviceEntry :
                            parameterEntry.getValue().entrySet()) {
                        NDArray array = deviceEntry.getValue().toDevice(Device.cpu(), true);
                        array.setName(
                                encodeName(
                                        stateName,
                                        parameterEntry.getKey(),
                                        deviceEntry.getKey()));
                        arrays.add(array);
                    }
                }
            }

            try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
                arrays.encode(baos, NDList.Encoding.NPZ);
                byte[] bytes = baos.toByteArray();
                os.writeInt(bytes.length);
                os.write(bytes);
            }
        }
    }

    /**
     * Loads this optimizer's state.
     *
     * @param manager the manager to create state arrays with
     * @param path the file to load the optimizer state from
     * @throws IOException if failed to load optimizer state
     */
    public void loadState(NDManager manager, Path path) throws IOException {
        if (!Files.exists(path)) {
            return;
        }
        try (DataInputStream is = new DataInputStream(Files.newInputStream(path))) {
            String magic = is.readUTF();
            if (!STATE_MAGIC.equals(magic)) {
                throw new IOException("Invalid optimizer state file: " + path);
            }
            int version = is.readInt();
            if (version != STATE_VERSION) {
                throw new IOException("Unsupported optimizer state version: " + version);
            }
            String optimizerClass = is.readUTF();
            if (!getClass().getName().equals(optimizerClass)) {
                throw new IOException(
                        "Optimizer state was saved for "
                                + optimizerClass
                                + " but current optimizer is "
                                + getClass().getName());
            }
            beginNumUpdate = is.readInt();
            numUpdate = is.readInt();
            int updateCountSize = is.readInt();
            updateCounts = new ConcurrentHashMap<>();
            for (int i = 0; i < updateCountSize; i++) {
                updateCounts.put(is.readUTF(), is.readInt());
            }

            int byteLength = is.readInt();
            if (byteLength < 0) {
                throw new IOException("Invalid optimizer state byte length: " + byteLength);
            }
            byte[] bytes = new byte[byteLength];
            is.readFully(bytes);
            Map<String, Map<String, Map<Device, NDArray>>> states = new HashMap<>();
            try (NDList arrays = NDList.decode(manager, new ByteArrayInputStream(bytes))) {
                for (NDArray array : arrays) {
                    StateKey key = decodeName(array.getName());
                    NDArray stateArray = array.toDevice(key.device, true);
                    stateArray.detach();
                    states.computeIfAbsent(key.stateName, k -> new ConcurrentHashMap<>())
                            .computeIfAbsent(key.parameterId, k -> new ConcurrentHashMap<>())
                            .put(key.device, stateArray);
                }
            }
            for (String stateName : getStateNames()) {
                setState(stateName, states.getOrDefault(stateName, new ConcurrentHashMap<>()));
            }
        }
    }

    /**
     * Returns optimizer-specific state names.
     *
     * @return optimizer-specific state names
     */
    protected Set<String> getStateNames() {
        return Collections.emptySet();
    }

    /**
     * Returns optimizer-specific state arrays.
     *
     * @param stateName the state name
     * @return optimizer-specific state arrays
     */
    protected Map<String, Map<Device, NDArray>> getState(String stateName) {
        return Collections.emptyMap();
    }

    /**
     * Sets optimizer-specific state arrays.
     *
     * @param stateName the state name
     * @param state optimizer-specific state arrays
     */
    protected void setState(String stateName, Map<String, Map<Device, NDArray>> state) {}

    protected static Set<String> stateNames(String... names) {
        return new HashSet<>(Arrays.asList(names));
    }

    protected NDArray withDefaultState(
            Map<String, Map<Device, NDArray>> state,
            String key,
            Device device,
            Function<String, NDArray> defaultFunction) {
        Map<Device, NDArray> arrayMap =
                state.computeIfAbsent(
                        key,
                        k -> {
                            Map<Device, NDArray> map = new ConcurrentHashMap<>();
                            NDArray s = defaultFunction.apply(k);
                            // TODO attach s to the NDManager of ParameterStore
                            s.detach(); // s is detached because it would be put into the optimizer
                            // callback manager and closed after the optimizer callback
                            // when using the MxParameterServer. For now, this will let it be closed
                            // by the
                            // GC when the optimizer is out of scope. Ideally, it should be put into
                            // the
                            // trainer manager instead.
                            map.put(device, s);
                            return map;
                        });
        return arrayMap.computeIfAbsent(
                device, k -> arrayMap.values().iterator().next().toDevice(device, true));
    }

    private static String encodeName(String stateName, String parameterId, Device device) {
        return encode(stateName)
                + '.'
                + encode(parameterId)
                + '.'
                + encode(device.getDeviceType())
                + '.'
                + device.getDeviceId();
    }

    private static StateKey decodeName(String name) throws IOException {
        String[] tokens = name.split("\\.", 4);
        if (tokens.length != 4) {
            throw new IOException("Invalid optimizer state array name: " + name);
        }
        return new StateKey(
                decode(tokens[0]),
                decode(tokens[1]),
                decodeDevice(decode(tokens[2]), Integer.parseInt(tokens[3])));
    }

    private static Device decodeDevice(String deviceType, int deviceId) {
        if ("cpu".equals(deviceType)) {
            return Device.cpu();
        }
        if ("gpu".equals(deviceType)) {
            return Device.gpu(deviceId);
        }
        return Device.of(deviceType, deviceId);
    }

    private static String encode(String value) {
        return Base64.getUrlEncoder()
                .withoutPadding()
                .encodeToString(value.getBytes(StandardCharsets.UTF_8));
    }

    private static String decode(String value) {
        return new String(Base64.getUrlDecoder().decode(value), StandardCharsets.UTF_8);
    }

    private static final class StateKey {
        String stateName;
        String parameterId;
        Device device;

        StateKey(String stateName, String parameterId, Device device) {
            this.stateName = stateName;
            this.parameterId = parameterId;
            this.device = device;
        }
    }

    /** The Builder to construct an {@link Optimizer}. */
    @SuppressWarnings("rawtypes")
    public abstract static class OptimizerBuilder<T extends OptimizerBuilder> {

        private float rescaleGrad = 1.0f;
        private float weightDecays;
        private float clipGrad = -1;
        private int beginNumUpdate;

        protected OptimizerBuilder() {}

        /**
         * Sets the value used to rescale the gradient. This is used to alleviate the effect of
         * batching on the loss. Usually, the value is set to \( 1/batch_size \). Defaults to 1.
         *
         * @param rescaleGrad the value used to rescale the gradient
         * @return this {@code Builder}
         */
        public T setRescaleGrad(float rescaleGrad) {
            this.rescaleGrad = rescaleGrad;
            return self();
        }

        /**
         * Sets the value of weight decay. Weight decay augments the objective function with a
         * regularization term that penalizes large weights.
         *
         * @param weightDecays the value of weight decay to be set
         * @return this {@code Builder}
         */
        public T optWeightDecays(float weightDecays) {
            this.weightDecays = weightDecays;
            return self();
        }

        /**
         * Sets the value of the \(clipGrad\). Clips the gradient to the range of \([-clipGrad,
         * clipGrad]\). If \(clipGrad \lt 0\), gradient clipping is turned off.
         *
         * <p>\(grad = max(min(grad, clipGrad), -clipGrad)\)
         *
         * @param clipGrad the value of \(clipGrad\)
         * @return this {@code Builder}
         */
        public T optClipGrad(float clipGrad) {
            this.clipGrad = clipGrad;
            return self();
        }

        /**
         * Sets the initial value of the number of updates.
         *
         * @param beginNumUpdate the initial value of the number of updates
         * @return this {@code Builder}
         */
        public T optBeginNumUpdate(int beginNumUpdate) {
            this.beginNumUpdate = beginNumUpdate;
            return self();
        }

        protected abstract T self();
    }
}
