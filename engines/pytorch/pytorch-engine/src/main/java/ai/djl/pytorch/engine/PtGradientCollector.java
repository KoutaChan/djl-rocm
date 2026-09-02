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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.GradientCollector;

/** {@code PtGradientCollector} is the PyTorch implementation of {@link GradientCollector}. */
public final class PtGradientCollector implements GradientCollector {

    private static final ThreadLocal<PtGradientCollector> ACTIVE = new ThreadLocal<>();

    private final Thread ownerThread;
    private final boolean gradMode;
    private final PtFlatGradientAccumulator flatGradientAccumulator;
    private boolean closed;

    /** Constructs a new {@code PtGradientCollector} instance. */
    public PtGradientCollector() {
        this(null);
    }

    PtGradientCollector(PtFlatGradientAccumulator flatGradientAccumulator) {
        ownerThread = Thread.currentThread();
        if (ACTIVE.get() != null) {
            throw new IllegalStateException("Nested PtGradientCollectors are not supported.");
        }

        this.flatGradientAccumulator = flatGradientAccumulator;
        gradMode = JniUtils.isGradMode();
        ACTIVE.set(this);
        JniUtils.setGradMode(true);

        // TODO Currently has performance implications and so has been disabled
        // Should fix and re-enable support for PyTorch gradient accumulation
        // See https://github.com/deepjavalibrary/djl/pull/2304
        // zeroGradients();
    }

    /** {@inheritDoc} */
    @Override
    public void backward(NDArray target) {
        validateThread();
        validateOpen();
        // TODO manager should create the new NDArray on the same device
        NDArray grad =
                target.getManager()
                        .ones(target.getShape(), target.getDataType())
                        .toDevice(target.getDevice(), false);
        backward(target, grad, false, false);
    }

    /**
     * Computes the gradients of the NDArray w.r.t variables.
     *
     * @param target the target/head array to run backward on
     * @param grad The “vector” in the Jacobian-vector product, usually gradients w.r.t. each
     *     element of corresponding tensors
     * @param keepGraph whether to retain the computation graph for another backward pass on the
     *     same graph. By default the computation history is cleared.
     * @param createGraph If true, graph of the derivative will be constructed, allowing to compute
     *     higher order derivative products. Defaults to false.
     */
    private void backward(NDArray target, NDArray grad, boolean keepGraph, boolean createGraph) {
        if (flatGradientAccumulator == null) {
            JniUtils.backward((PtNDArray) target, (PtNDArray) grad, keepGraph, createGraph);
        } else {
            flatGradientAccumulator.backward((PtNDArray) target, (PtNDArray) grad);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void zeroGradients() {
        validateThread();
        validateOpen();
        if (flatGradientAccumulator != null) {
            flatGradientAccumulator.zeroGradients();
            return;
        }
        NDManager systemManager = PtNDManager.getSystemManager();
        for (NDArray array : systemManager.getManagedArrays()) {
            if (array.hasGradient()) {
                // getGradient() allocates a new NDArray wrapper around the same native
                // tensor on every call, so the wrapper must be closed to avoid leaking
                // PtNDArray handles. fillI(0) overwrites the buffer without reading it,
                // so the gradient recovers cleanly even if it currently holds NaN/Inf.
                try (NDArray gradient = array.getGradient()) {
                    gradient.fillI(0);
                }
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        validateThread();
        if (closed) {
            return;
        }
        closed = true;
        try {
            JniUtils.setGradMode(gradMode);
        } finally {
            ACTIVE.remove();
            if (flatGradientAccumulator != null) {
                flatGradientAccumulator.collectorClosed();
            }
        }
        // TODO: do some clean up if necessary
    }

    private void validateThread() {
        if (Thread.currentThread() != ownerThread) {
            throw new IllegalStateException(
                    "PtGradientCollector can only be used from the thread that created it.");
        }
    }

    private void validateOpen() {
        if (closed) {
            throw new IllegalStateException("PtGradientCollector has already been closed.");
        }
    }
}
