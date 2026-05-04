/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.engine.EngineException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.DistributedTrainingConfig;
import ai.djl.training.ParameterServer;
import ai.djl.training.optimizer.Optimizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/** PyTorch native distributed parameter server backed by c10d ProcessGroupNCCL. */
final class PtDistributedParameterServer implements ParameterServer {

    private Optimizer optimizer;
    private DistributedTrainingConfig config;
    private Map<String, PtNDArray> parameters;
    private long reducerHandle;
    private boolean prepared;
    private boolean finalized = true;

    PtDistributedParameterServer(Optimizer optimizer, DistributedTrainingConfig config) {
        this.optimizer = optimizer;
        this.config = config;
        parameters = new LinkedHashMap<>();
    }

    /** {@inheritDoc} */
    @Override
    public void validateDevices(Device[] devices) {
        if (devices.length != 1) {
            throw new IllegalArgumentException(
                    "Native distributed training expects one GPU device per process.");
        }
        if (!devices[0].isGpu()) {
            throw new IllegalArgumentException(
                    "Native distributed training requires a CUDA or ROCm GPU device.");
        }
        if (config.isFindUnusedParameters()) {
            throw new UnsupportedOperationException(
                    "findUnusedParameters is not supported by the PyTorch native reducer yet.");
        }
    }

    /** {@inheritDoc} */
    @Override
    public synchronized void init(String parameterId, NDArray[] value) {
        if (value.length != 1) {
            throw new IllegalArgumentException(
                    "Native distributed training expects one local parameter tensor.");
        }
        if (!(value[0] instanceof PtNDArray)) {
            throw new IllegalArgumentException(
                    "Native distributed training is only supported by the PyTorch engine.");
        }
        parameters.putIfAbsent(parameterId, (PtNDArray) value[0]);
    }

    /** {@inheritDoc} */
    @Override
    public synchronized void prepareForBackward(NDList outputs) {
        ensureReducer();
        JniUtils.distributedPrepareForBackward(reducerHandle);
        prepared = true;
        finalized = false;
    }

    /** {@inheritDoc} */
    @Override
    public synchronized void update(String parameterId, NDArray[] params) {
        finishBackward();
        try (NDArray grad = params[0].getGradient()) {
            optimizer.update(parameterId, params[0], grad);
        }
    }

    /** {@inheritDoc} */
    @Override
    public synchronized void update(String parameterId, NDArray[] grads, NDArray[] params) {
        finishBackward();
        optimizer.update(parameterId, params[0], grads[0]);
    }

    /** {@inheritDoc} */
    @Override
    public void saveOptimizerState(Path path) throws IOException {
        optimizer.saveState(path);
    }

    /** {@inheritDoc} */
    @Override
    public void loadOptimizerState(NDManager manager, Path path) throws IOException {
        optimizer.loadState(manager, path);
    }

    /** {@inheritDoc} */
    @Override
    public synchronized void close() {
        if (reducerHandle != 0L) {
            try {
                finishBackward();
            } finally {
                JniUtils.distributedDeleteReducer(reducerHandle);
                reducerHandle = 0L;
            }
        }
    }

    private void ensureReducer() {
        if (reducerHandle != 0L) {
            return;
        }
        if (parameters.isEmpty()) {
            throw new EngineException(
                    "No trainable parameters were initialized for native distributed training.");
        }
        long[] handles = new long[parameters.size()];
        int index = 0;
        for (PtNDArray parameter : parameters.values()) {
            handles[index++] = parameter.getHandle();
        }
        reducerHandle =
                JniUtils.distributedCreateReducer(
                        handles,
                        config.getMasterHost(),
                        config.getMasterPort(),
                        config.getRank(),
                        config.getWorldSize(),
                        config.getLocalRank(),
                        config.getBucketCapMb(),
                        config.isStaticGraph(),
                        config.isFindUnusedParameters(),
                        config.isAverageGradients());
    }

    private void finishBackward() {
        if (prepared && !finalized) {
            JniUtils.distributedFinalizeBackward(reducerHandle);
            finalized = true;
            prepared = false;
        }
    }
}
