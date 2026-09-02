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
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

/** An interface for a key-value store to store parameters, and their corresponding gradients. */
public interface ParameterServer extends AutoCloseable {

    /**
     * Validates devices that will be used by the {@link ParameterStore}.
     *
     * @param devices the devices to create mirrored parameters on
     */
    default void validateDevices(Device[] devices) {}

    /**
     * Initializes the {@code ParameterStore} for the given parameter.
     *
     * @param parameterId the parameter ID
     * @param value the values to be set for the given parameter
     */
    void init(String parameterId, NDArray[] value);

    /**
     * Updates the parameter of a key from Parameter Server.
     *
     * @param parameterId the key to identify the parameter
     * @param params the parameter NDArrays in different devices to be updated.
     */
    default void update(String parameterId, NDArray[] params) {
        NDArray[] grads = Arrays.stream(params).map(NDArray::getGradient).toArray(NDArray[]::new);
        update(parameterId, grads, params);
        Arrays.stream(grads).forEach(NDArray::close);
    }

    /**
     * Updates the parameter of a key from Parameter Server.
     *
     * @param parameterId the key to identify the parameter
     * @param grads the gradient NDArrays in different devices to apply the update.
     * @param params the parameter NDArrays in different devices to be updated.
     */
    void update(String parameterId, NDArray[] grads, NDArray[] params);

    /**
     * Prepares the parameter server for a backward pass.
     *
     * @param outputs the training forward outputs
     */
    default void prepareForBackward(NDList outputs) {}

    /**
     * Completes any parameter-server initialization required before a forward pass.
     *
     * <p>This hook is invoked after parameter mirrors have been materialized and immediately before
     * the model reads them. Implementations must make repeated calls safe.
     */
    default void prepareForForward() {}

    /** Finalizes asynchronous or distributed gradients before they are inspected or updated. */
    default void finalizeGradients() {}

    /**
     * Returns whether this parameter server needs per-parameter gradient preparation before finite
     * checks.
     *
     * @return {@code true} when {@link #prepareGradients(String, NDArray[])} must be called
     */
    default boolean requiresGradientPreparation() {
        return false;
    }

    /**
     * Prepares one parameter's gradients for finite checking and the optimizer step.
     *
     * <p>Parameter servers that reduce gradients during {@link #update(String, NDArray[],
     * NDArray[])} must perform that reduction here as well, so mixed-precision overflow detection
     * observes the exact gradient that the optimizer will consume.
     *
     * @param parameterId the key that identifies the parameter
     * @param gradients the parameter gradients on each training device
     */
    default void prepareGradients(String parameterId, NDArray[] gradients) {}

    /**
     * Releases any state retained while preparing the current gradient step.
     *
     * <p>This method is called after an applied step, a skipped step, or a failed step.
     */
    default void finishGradientStep() {}

    /**
     * Saves optimizer state.
     *
     * @param path the file to save optimizer state to
     * @throws IOException if failed to save optimizer state
     */
    default void saveOptimizerState(Path path) throws IOException {}

    /**
     * Loads optimizer state.
     *
     * @param manager the manager to create state arrays with
     * @param path the file to load optimizer state from
     * @throws IOException if failed to load optimizer state
     */
    default void loadOptimizerState(NDManager manager, Path path) throws IOException {}

    /** {@inheritDoc} */
    @Override
    void close();
}
