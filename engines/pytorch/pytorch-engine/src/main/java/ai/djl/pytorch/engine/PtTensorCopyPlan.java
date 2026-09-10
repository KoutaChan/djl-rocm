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
import ai.djl.util.NativeResource;

/**
 * Copies a fixed list of tensor pairs without rebuilding views or JNI handle lists.
 *
 * <p>All tensors must be dense and on one device; each source/destination pair must have the same
 * shape and data type. Sources must not overlap destination storage, and destinations must not
 * overlap one another. Tensor shape, storage, and device must remain unchanged while this plan is
 * open. Values may be updated in place between executions.
 *
 * <p>Copies do not record an autograd graph. Accelerator execution is confined to the current
 * stream of the first copy. Callers establish dependencies on other streams. Arrays remain
 * caller-owned; close this plan before closing them.
 */
public final class PtTensorCopyPlan extends NativeResource<Long> {

    PtTensorCopyPlan(NDList sources, NDList destinations) {
        super(JniUtils.createTensorCopyPlan(handles(sources), handles(destinations)));
    }

    /** Copies current source values into the existing destination tensors. */
    public synchronized void copy() {
        JniUtils.copyTensorCopyPlan(getHandle());
    }

    /** {@inheritDoc} */
    @Override
    public synchronized void close() {
        onClose();
        Long pointer = handle.getAndSet(null);
        if (pointer != null) {
            JniUtils.deleteTensorCopyPlan(pointer);
        }
    }

    private static long[] handles(NDList arrays) {
        long[] handles = new long[arrays.size()];
        for (int index = 0; index < arrays.size(); index++) {
            NDArray array = arrays.get(index);
            if (!(array instanceof PtNDArray)) {
                throw new IllegalArgumentException("Tensor copy plans require PyTorch NDArrays.");
            }
            handles[index] = ((PtNDArray) array).getHandle();
        }
        return handles;
    }
}
