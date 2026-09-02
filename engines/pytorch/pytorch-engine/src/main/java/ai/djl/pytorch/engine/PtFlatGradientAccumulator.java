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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.GradientCollector;
import ai.djl.util.NativeResource;

import java.util.Objects;

/**
 * Accumulates gradients for a fixed ordered parameter list directly into a flat tensor.
 *
 * <p>The accumulator uses {@code torch::autograd::grad} and therefore does not create or update the
 * leaf parameters' {@code .grad} tensors. Each parameter owns one contiguous, non-overlapping slice
 * of the destination in list order. Unused parameters leave their slices unchanged.
 *
 * <p>Parameters must be distinct dense leaf tensors that require gradients. All parameters and the
 * contiguous rank-1 destination must use the same PyTorch device and data type. Sparse gradients
 * are not supported.
 *
 * <p>On an accelerator, the first backward or zero operation binds this accumulator to the current
 * device stream. Later operations must use that stream. Use a separate accumulator for each stream.
 * The returned flat gradient must also be consumed on the bound stream unless the caller
 * establishes an explicit stream dependency.
 *
 * <p>The parameter arrays and destination remain caller-owned and must outlive this object. One
 * accumulator supports at most one active {@link GradientCollector} at a time.
 */
public final class PtFlatGradientAccumulator extends NativeResource<Long> {

    private final PtNDArray gradient;
    private boolean collectorActive;

    PtFlatGradientAccumulator(NDList parameters, NDArray gradient) {
        super(create(parameters, gradient));
        this.gradient = (PtNDArray) gradient;
    }

    /**
     * Opens a gradient collector that adds each backward result to the flat destination.
     *
     * @return the flat-gradient collector
     * @throws IllegalStateException if another collector from this accumulator is active
     */
    public synchronized GradientCollector newGradientCollector() {
        getHandle();
        if (collectorActive) {
            throw new IllegalStateException(
                    "PtFlatGradientAccumulator already has an active gradient collector.");
        }
        collectorActive = true;
        try {
            return new PtGradientCollector(this);
        } catch (RuntimeException | Error e) {
            collectorActive = false;
            throw e;
        }
    }

    /**
     * Returns the caller-owned flat gradient tensor.
     *
     * @return the flat gradient tensor passed to the factory
     */
    public NDArray getGradient() {
        getHandle();
        return gradient;
    }

    /** Sets the flat gradient tensor to zero without changing the plan. */
    public synchronized void zeroGradients() {
        JniUtils.zeroFlatGradientAccumulator(getHandle());
    }

    synchronized void backward(PtNDArray target, PtNDArray targetGradient) {
        JniUtils.backwardFlatGradientAccumulator(getHandle(), target, targetGradient);
    }

    synchronized void collectorClosed() {
        collectorActive = false;
    }

    /** {@inheritDoc} */
    @Override
    public synchronized void close() {
        if (collectorActive) {
            throw new IllegalStateException(
                    "Cannot close PtFlatGradientAccumulator while its collector is active.");
        }
        onClose();
        Long pointer = handle.getAndSet(null);
        if (pointer != null) {
            JniUtils.deleteFlatGradientAccumulator(pointer);
        }
    }

    private static long create(NDList parameters, NDArray gradient) {
        Objects.requireNonNull(parameters, "parameters");
        Objects.requireNonNull(gradient, "gradient");
        PtNDArray[] parameterArrays = new PtNDArray[parameters.size()];
        for (int index = 0; index < parameters.size(); ++index) {
            NDArray parameter = Objects.requireNonNull(parameters.get(index), "parameter");
            if (!(parameter instanceof PtNDArray)) {
                throw new IllegalArgumentException(
                        "Flat gradient parameters must be PyTorch NDArrays.");
            }
            parameterArrays[index] = (PtNDArray) parameter;
        }
        if (!(gradient instanceof PtNDArray)) {
            throw new IllegalArgumentException(
                    "The flat gradient destination must be a PyTorch NDArray.");
        }
        return JniUtils.createFlatGradientAccumulator(parameterArrays, (PtNDArray) gradient);
    }
}
