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

import java.util.ArrayList;
import java.util.List;

/** Completion ticket for a batch of pinned host-to-device transfers. */
public final class PtTransferTicket implements AutoCloseable {

    private final PtEvent completionEvent;
    private final List<PtNDArray> arrays;

    @SuppressWarnings("PMD.UnusedPrivateField")
    private final List<PtPinnedBuffer> buffers;

    private boolean completionKnown;
    private boolean closed;

    PtTransferTicket(
            PtEvent completionEvent, List<PtNDArray> arrays, List<PtPinnedBuffer> buffers) {
        this.completionEvent = completionEvent;
        this.arrays = new ArrayList<>(arrays);
        this.buffers = new ArrayList<>(buffers);
    }

    /**
     * Returns the device on which the transfer was enqueued.
     *
     * @return the transfer device
     */
    public synchronized Device getDevice() {
        ensureOpen();
        return completionEvent.getDevice();
    }

    /**
     * Makes the transferred arrays available to the stream that is current for the transfer device.
     *
     * <p>This method enqueues a device-side wait for transfer completion and records every
     * destination array as used by the current stream. It does not wait on the calling thread.
     */
    public synchronized void waitOnStream() {
        ensureOpen();
        completionEvent.waitOnStream();
        for (PtNDArray array : arrays) {
            array.recordStream();
        }
    }

    /**
     * Returns whether all copies in this transfer have completed.
     *
     * @return {@code true} if the transfer has completed
     */
    public synchronized boolean isComplete() {
        ensureOpen();
        completionKnown = completionKnown || completionEvent.isComplete();
        return completionKnown;
    }

    /** Waits on the calling thread until all copies in this transfer have completed. */
    public synchronized void synchronize() {
        ensureOpen();
        completionEvent.synchronize();
        completionKnown = true;
    }

    /**
     * Releases this ticket without synchronizing the device.
     *
     * <p>Closing fails if the transfer is still in flight. The ticket remains open so the caller
     * can explicitly call {@link #synchronize()} before retrying. This method never introduces a
     * hidden host synchronization.
     *
     * @throws IllegalStateException if the transfer has not completed
     */
    @Override
    public synchronized void close() {
        if (!closed) {
            if (!completionKnown && !completionEvent.isComplete()) {
                throw new IllegalStateException(
                        "Cannot close a PtTransferTicket while its transfer is in flight.");
            }
            completionKnown = true;
            try {
                completionEvent.close();
            } finally {
                if (completionEvent.isReleased()) {
                    closed = true;
                    arrays.clear();
                    buffers.clear();
                }
            }
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("PtTransferTicket is closed.");
        }
    }
}
