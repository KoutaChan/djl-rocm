/*
 * Copyright 2025 KoutaChan.
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
import ai.djl.engine.Autocast;
import ai.djl.ndarray.types.DataType;
import ai.djl.pytorch.jni.JniUtils;

/**
 * PyTorch-backed {@link Autocast} guard. Wraps libtorch's {@code at::autocast} thread-local flags:
 * on construction, saves the previous (enabled, dtype, cache_enabled) triple + bumps the nesting
 * counter, then flips the flags to the requested values. On {@link #close()} it decrements the
 * nesting counter, clears the op-result cache when nesting reaches zero, and restores the saved
 * previous state.
 *
 * <p>Mirrors the {@code __enter__} / {@code __exit__} semantics of PyTorch's Python-level {@code
 * torch.autocast} context manager, so nested scopes and {@code enabled=false} sub-regions behave
 * identically to the reference implementation.
 */
final class PtAutocast implements Autocast {

    private final int deviceType;
    private final boolean prevEnabled;
    private final int prevDtype;
    private final boolean prevCacheEnabled;
    private final Thread ownerThread;
    private boolean closed;

    PtAutocast(Device device, DataType dataType, boolean enabled, boolean cacheEnabled) {
        if (enabled && dataType != DataType.FLOAT16 && dataType != DataType.BFLOAT16) {
            throw new IllegalArgumentException(
                    "PyTorch autocast data type must be FLOAT16 or BFLOAT16.");
        }
        this.deviceType = PtDeviceType.toDeviceType(device);
        this.prevEnabled = JniUtils.autocastIsEnabled(deviceType);
        this.prevDtype = JniUtils.autocastGetDtype(deviceType);
        this.prevCacheEnabled = JniUtils.autocastIsCacheEnabled();
        this.ownerThread = Thread.currentThread();

        // Match PyTorch's ordering: set dtype before enabled so the first op
        // inside the scope sees a consistent (enabled, dtype) pair. The
        // nesting counter bumps regardless of enabled so the cache clear at
        // the outermost exit still fires symmetrically.
        if (dataType != null) {
            JniUtils.autocastSetDtype(deviceType, dataType.ordinal());
        }
        JniUtils.autocastSetEnabled(deviceType, enabled);
        JniUtils.autocastIncrementNesting();
        JniUtils.autocastSetCacheEnabled(cacheEnabled);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (Thread.currentThread() != ownerThread) {
            throw new IllegalStateException(
                    "PyTorch autocast scope can only be closed from the thread that created it.");
        }
        if (closed) {
            return;
        }
        closed = true;
        if (JniUtils.autocastDecrementNesting() == 0) {
            JniUtils.autocastClearCache();
        }
        JniUtils.autocastSetEnabled(deviceType, prevEnabled);
        JniUtils.autocastSetDtype(deviceType, prevDtype);
        JniUtils.autocastSetCacheEnabled(prevCacheEnabled);
    }
}
