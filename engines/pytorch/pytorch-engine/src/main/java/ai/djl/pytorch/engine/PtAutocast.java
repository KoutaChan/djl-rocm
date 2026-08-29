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

/** PyTorch implementation of {@link Autocast}. */
final class PtAutocast implements Autocast {

    private final int deviceType;
    private final boolean previousEnabled;
    private final int previousDataType;
    private final boolean previousCacheEnabled;
    private final Thread ownerThread;
    private boolean closed;

    PtAutocast(Device device, DataType dataType, boolean enabled, boolean cacheEnabled) {
        if (enabled && dataType != DataType.FLOAT16 && dataType != DataType.BFLOAT16) {
            throw new IllegalArgumentException(
                    "PyTorch autocast data type must be FLOAT16 or BFLOAT16.");
        }
        deviceType = PtDeviceType.toDeviceType(device);
        previousEnabled = JniUtils.autocastIsEnabled(deviceType);
        previousDataType = JniUtils.autocastGetDataType(deviceType);
        previousCacheEnabled = JniUtils.autocastIsCacheEnabled();
        ownerThread = Thread.currentThread();

        // Set the data type before enabling autocast so the first operation sees consistent state.
        if (dataType != null) {
            JniUtils.autocastSetDataType(deviceType, dataType.ordinal());
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
        JniUtils.autocastSetEnabled(deviceType, previousEnabled);
        JniUtils.autocastSetDataType(deviceType, previousDataType);
        JniUtils.autocastSetCacheEnabled(previousCacheEnabled);
    }
}
