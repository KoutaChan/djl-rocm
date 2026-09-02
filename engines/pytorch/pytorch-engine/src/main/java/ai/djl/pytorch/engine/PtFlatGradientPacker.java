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
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.util.NativeResource;

import java.util.Objects;

/**
 * Packs a fixed ordered parameter list's leaf gradients into a flat destination tensor.
 *
 * <p>Each parameter owns one contiguous, non-overlapping slice of the destination in list order.
 * Packing can either replace or accumulate into those slices. A successful operation clears the
 * parameter {@code .grad} tensors after their values have been enqueued. The destination may use a
 * different floating-point data type from the parameters.
 *
 * <p>Parameters must be distinct dense PyTorch tensors. The destination must be a contiguous,
 * floating-point rank-1 PyTorch tensor on the same device, and its element count must equal the
 * total parameter element count. Sparse gradients are not supported.
 *
 * <p>On an accelerator, the first operation binds this packer to the current device stream. Later
 * operations must use that stream. The destination must also be consumed on the bound stream unless
 * the caller establishes an explicit stream dependency.
 *
 * <p>The parameter arrays and destination remain caller-owned and must outlive this object.
 */
public final class PtFlatGradientPacker extends NativeResource<Long> {

    /** Defines how a packing operation handles a parameter without a gradient. */
    public enum MissingGradientPolicy {
        /** Rejects the operation without modifying the destination or parameter gradients. */
        ERROR,

        /** Treats a missing gradient as zero. */
        ZERO
    }

    private final PtNDArray destination;

    PtFlatGradientPacker(NDList parameters, NDArray destination) {
        super(create(parameters, destination));
        this.destination = (PtNDArray) destination;
    }

    /**
     * Overwrites the destination with the current parameter gradients and clears those gradients.
     *
     * <p>This overload rejects missing gradients. Use {@link #packAndClear(MissingGradientPolicy)}
     * to treat them as zero instead.
     */
    public void packAndClear() {
        packAndClear(MissingGradientPolicy.ERROR);
    }

    /**
     * Overwrites the destination with the current parameter gradients and clears those gradients.
     *
     * <p>Each gradient is flattened into its parameter's slice in parameter-list order and is
     * converted to the destination data type when necessary. With {@link
     * MissingGradientPolicy#ERROR}, validation is completed before the destination or any gradient
     * is changed.
     *
     * @param missingGradientPolicy policy for parameters without gradients
     */
    public synchronized void packAndClear(MissingGradientPolicy missingGradientPolicy) {
        Objects.requireNonNull(missingGradientPolicy, "missingGradientPolicy");
        JniUtils.packAndClearFlatGradientPacker(
                getHandle(), missingGradientPolicy == MissingGradientPolicy.ZERO);
    }

    /**
     * Adds the current parameter gradients to the destination and clears those gradients.
     *
     * <p>This overload rejects missing gradients. Use {@link
     * #accumulateAndClear(MissingGradientPolicy)} to treat them as zero instead.
     */
    public void accumulateAndClear() {
        accumulateAndClear(MissingGradientPolicy.ERROR);
    }

    /**
     * Adds the current parameter gradients to the destination and clears those gradients.
     *
     * <p>Each gradient is flattened into its parameter's slice in parameter-list order and is
     * converted to the destination data type when necessary. With {@link
     * MissingGradientPolicy#ERROR}, validation is completed before the destination or any gradient
     * is changed.
     *
     * @param missingGradientPolicy policy for parameters without gradients
     */
    public synchronized void accumulateAndClear(MissingGradientPolicy missingGradientPolicy) {
        Objects.requireNonNull(missingGradientPolicy, "missingGradientPolicy");
        JniUtils.accumulateAndClearFlatGradientPacker(
                getHandle(), missingGradientPolicy == MissingGradientPolicy.ZERO);
    }

    /** Sets the destination tensor to zero without changing parameter gradients. */
    public synchronized void zeroDestination() {
        JniUtils.zeroFlatGradientPackerDestination(getHandle());
    }

    /** Clears all bound parameter gradients without changing the destination tensor. */
    public synchronized void clearParameterGradients() {
        JniUtils.clearFlatGradientPackerParameterGradients(getHandle());
    }

    /**
     * Returns the caller-owned destination tensor.
     *
     * @return the destination tensor passed to the factory
     */
    public NDArray getDestination() {
        getHandle();
        return destination;
    }

    /** {@inheritDoc} */
    @Override
    public synchronized void close() {
        onClose();
        Long pointer = handle.getAndSet(null);
        if (pointer != null) {
            JniUtils.deleteFlatGradientPacker(pointer);
        }
    }

    private static long create(NDList parameters, NDArray destination) {
        Objects.requireNonNull(parameters, "parameters");
        Objects.requireNonNull(destination, "destination");
        if (!(destination instanceof PtNDArray)) {
            throw new IllegalArgumentException(
                    "The flat gradient destination must be a PyTorch NDArray.");
        }
        PtNDArray ptDestination = (PtNDArray) destination;
        validateDestination(ptDestination);

        if (parameters.isEmpty()) {
            throw new IllegalArgumentException("A flat gradient packer requires parameters.");
        }
        PtNDArray[] parameterArrays = new PtNDArray[parameters.size()];
        long parameterElements = 0;
        Device device = ptDestination.getDevice();
        for (int index = 0; index < parameters.size(); ++index) {
            NDArray parameter = Objects.requireNonNull(parameters.get(index), "parameter");
            if (!(parameter instanceof PtNDArray)) {
                throw new IllegalArgumentException(
                        "Flat gradient parameters must be PyTorch NDArrays.");
            }
            PtNDArray ptParameter = (PtNDArray) parameter;
            if (!device.equals(ptParameter.getDevice())) {
                throw new IllegalArgumentException(
                        "Flat gradient parameters and destination must use the same device.");
            }
            parameterElements = Math.addExact(parameterElements, ptParameter.getShape().size());
            parameterArrays[index] = ptParameter;
        }
        if (parameterElements != ptDestination.getShape().size()) {
            throw new IllegalArgumentException(
                    "The flat gradient destination size must equal the total parameter size.");
        }
        return JniUtils.createFlatGradientPacker(parameterArrays, ptDestination);
    }

    private static void validateDestination(PtNDArray destination) {
        if (destination.getShape().dimension() != 1) {
            throw new IllegalArgumentException(
                    "The flat gradient destination must be a rank-1 tensor.");
        }
        if (!destination.getDataType().isFloating()) {
            throw new IllegalArgumentException(
                    "The flat gradient destination must have a floating-point data type.");
        }
        if (!JniUtils.isContiguous(destination)) {
            throw new IllegalArgumentException("The flat gradient destination must be contiguous.");
        }
    }
}
