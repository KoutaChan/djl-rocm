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
import ai.djl.ndarray.NDManager;
import ai.djl.training.optimizer.Optimizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/** {@code LocalParameterServer} is an implementation of the {@code ParameterServer} interface. */
public class LocalParameterServer implements ParameterServer {

    private Optimizer optimizer;
    private Set<String> preparedGradients;

    /**
     * Create a new instance of {@code LocalParameterServer} for the given optimizer.
     *
     * @param optimizer an optimizer
     */
    public LocalParameterServer(Optimizer optimizer) {
        this.optimizer = optimizer;
        preparedGradients = ConcurrentHashMap.newKeySet();
    }

    /** {@inheritDoc} */
    @Override
    public void init(String parameterId, NDArray[] value) {}

    /** {@inheritDoc} */
    @Override
    public void update(String parameterId, NDArray[] grads, NDArray[] params) {
        if (!preparedGradients.remove(parameterId)) {
            reduceGradients(grads);
        }
        Device firstDevice = params[0].getDevice();
        // update weights on different devices with reduced gradient
        // use duplicate because after the first optimizer.update
        // PyTorch optimizer will zero grads[0]
        // the second copy is to move the grads[0] to the device the weight is on
        try (NDArray aggregatedGrad = grads[0].duplicate()) {
            for (NDArray param : params) {
                if (param.getDevice().equals(firstDevice)) {
                    optimizer.update(parameterId, param, aggregatedGrad);
                } else {
                    try (NDArray gradSumCopy = aggregatedGrad.toDevice(param.getDevice(), true)) {
                        optimizer.update(parameterId, param, gradSumCopy);
                    }
                }
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public void prepareGradients(String parameterId, NDArray[] gradients) {
        reduceGradients(gradients);
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
    public void close() {
        finishGradientStep();
    }

    private static void reduceGradients(NDArray[] gradients) {
        Device firstDevice = gradients[0].getDevice();
        for (int i = 1; i < gradients.length; ++i) {
            try (NDArray gradientCopy = gradients[i].toDevice(firstDevice, true)) {
                gradients[0].addi(gradientCopy);
            }
        }
    }
}
