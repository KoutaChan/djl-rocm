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

package ai.djl.mxnet.engine;

import ai.djl.Device;
import ai.djl.mxnet.jna.JnaUtils;
import ai.djl.mxnet.jna.MxnetLibrary;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.ParameterServer;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.util.NativeResource;

import com.sun.jna.Pointer;

import java.util.Arrays;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/** {@code MxParameterServer} is the MXNet implementation of {@link ParameterServer}. */
public class MxParameterServer extends NativeResource<Pointer> implements ParameterServer {

    @SuppressWarnings("PMD.SingularField")
    // use class field to hold the OptimizerCallback which prevent it from being gc.
    private OptimizerCallback callback;

    private int priority;
    private Set<String> preparedGradients;

    /**
     * Constructs a new {@code MxParameterServer}.
     *
     * @param optimizer the optimizer to use for the parameter server updates
     */
    @SuppressWarnings("this-escape")
    public MxParameterServer(Optimizer optimizer) {
        super(createdKVStore());
        callback = new OptimizerCallback(optimizer);
        JnaUtils.parameterStoreSetUpdater(getHandle(), null, callback, null);
        priority = 0;
        preparedGradients = ConcurrentHashMap.newKeySet();
    }

    /** {@inheritDoc} */
    @Override
    public void init(String parameterId, NDArray[] values) {
        String[] keys = new String[values.length];
        Arrays.fill(keys, parameterId);
        NDList vals = new NDList(values);
        JnaUtils.parameterStoreInit(getHandle(), values.length, keys, vals);
    }

    /** {@inheritDoc} */
    @Override
    public void update(String parameterId, NDArray[] grads, NDArray[] params) {
        boolean prepared = preparedGradients.remove(parameterId);
        int gradientCount = prepared ? 1 : grads.length;
        String[] gradKeys = new String[gradientCount];
        String[] paramKeys = new String[params.length];
        Arrays.fill(gradKeys, parameterId);
        Arrays.fill(paramKeys, parameterId);
        NDList gradients = prepared ? new NDList(grads[0]) : new NDList(grads);
        JnaUtils.parameterStorePushPull(
                getHandle(),
                gradientCount,
                gradKeys,
                params.length,
                paramKeys,
                gradients,
                new NDList(params),
                -priority);
        priority++;
    }

    /** {@inheritDoc} */
    @Override
    public void prepareGradients(String parameterId, NDArray[] gradients) {
        Device firstDevice = gradients[0].getDevice();
        for (int i = 1; i < gradients.length; ++i) {
            try (NDArray gradientCopy = gradients[i].toDevice(firstDevice, true)) {
                gradients[0].addi(gradientCopy);
            }
        }
        preparedGradients.add(parameterId);
    }

    /** {@inheritDoc} */
    @Override
    public boolean requiresGradientPreparation() {
        return true;
    }

    /** {@inheritDoc} */
    @Override
    public void finishGradientStep() {
        preparedGradients.clear();
    }

    private static Pointer createdKVStore() {
        return JnaUtils.parameterStoreCreate("device");
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        finishGradientStep();
        Pointer pointer = handle.getAndSet(null);
        if (pointer != null) {
            JnaUtils.parameterStoreClose(pointer);
        }
    }

    /** A helper to wrap the optimizer so it can be called by the MXNet KVStore. */
    private static final class OptimizerCallback implements MxnetLibrary.MXKVStoreStrUpdater {

        private Optimizer optimizer;

        OptimizerCallback(Optimizer optimizer) {
            this.optimizer = optimizer;
        }

        /** {@inheritDoc} */
        @Override
        public void apply(String parameterId, Pointer recv, Pointer local, Pointer handle) {
            // updater callback arguments order is: index, gradient, weight.
            try (NDManager manager = MxNDManager.getSystemManager().newSubManager()) {
                MxNDManager m = (MxNDManager) manager;
                MxNDArray grad = m.create(recv);
                MxNDArray weight = m.create(local);
                optimizer.update(parameterId, weight, grad);
            }
        }
    }
}
