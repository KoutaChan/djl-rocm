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

import ai.djl.engine.InferenceMode;
import ai.djl.pytorch.jni.JniUtils;

/** PyTorch thread-local inference scope. */
final class PtInferenceMode implements InferenceMode {

    private final Thread ownerThread;
    private final boolean nativeInferenceMode;
    private final boolean previousGradMode;
    private long handle;
    private boolean closed;

    PtInferenceMode(boolean nativeInferenceMode) {
        ownerThread = Thread.currentThread();
        this.nativeInferenceMode = nativeInferenceMode;
        if (nativeInferenceMode) {
            previousGradMode = false;
            handle = JniUtils.openInferenceMode();
        } else {
            previousGradMode = JniUtils.isGradMode();
            JniUtils.setGradMode(false);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (Thread.currentThread() != ownerThread) {
            throw new IllegalStateException(
                    "PtInferenceMode must be closed by the thread that opened it.");
        }
        if (closed) {
            return;
        }
        closed = true;
        if (nativeInferenceMode) {
            long nativeHandle = handle;
            handle = 0;
            JniUtils.closeInferenceMode(nativeHandle);
        } else {
            JniUtils.setGradMode(previousGradMode);
        }
    }
}
