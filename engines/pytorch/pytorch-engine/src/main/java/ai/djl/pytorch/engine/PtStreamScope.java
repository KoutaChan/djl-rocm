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
import ai.djl.pytorch.jni.JniUtils;

/** A thread-local PyTorch accelerator stream scope. */
public final class PtStreamScope implements AutoCloseable {

    private final Thread ownerThread;
    private final PtStream stream;
    private long handle;
    private boolean closed;

    PtStreamScope(Device device) {
        ownerThread = Thread.currentThread();
        stream = null;
        handle = JniUtils.openStreamScope(device);
    }

    PtStreamScope(PtStream stream) {
        ownerThread = Thread.currentThread();
        this.stream = stream;
        handle = JniUtils.openDeviceStream(stream.getHandle());
    }

    /** Restores the stream that was current when this scope was opened. */
    @Override
    public void close() {
        if (Thread.currentThread() != ownerThread) {
            throw new IllegalStateException(
                    "PtStreamScope must be closed by the thread that opened it.");
        }
        if (!closed) {
            closed = true;
            try {
                if (handle != 0) {
                    JniUtils.closeStreamScope(handle);
                }
            } finally {
                handle = 0;
                if (stream != null) {
                    stream.closeScope();
                }
            }
        }
    }
}
