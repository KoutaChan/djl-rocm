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

/** A fixed-shape PyTorch accelerator graph that can replay a captured inference workload. */
public final class PtInferenceGraph implements AutoCloseable {

    private long handle;

    PtInferenceGraph(Device device) {
        handle = JniUtils.createInferenceGraph(device);
    }

    /** Starts capture on the graph's private accelerator stream. */
    public void beginCapture() {
        JniUtils.beginInferenceGraphCapture(handle);
    }

    /** Ends capture and instantiates the executable graph. */
    public void endCapture() {
        JniUtils.endInferenceGraphCapture(handle);
    }

    /** Replays the captured workload and orders its output before the caller's current stream. */
    public void replay() {
        JniUtils.replayInferenceGraph(handle);
    }

    /** Releases the native graph and its private memory pool. */
    @Override
    public void close() {
        long nativeHandle = handle;
        if (nativeHandle != 0) {
            handle = 0;
            JniUtils.deleteInferenceGraph(nativeHandle);
        }
    }
}
