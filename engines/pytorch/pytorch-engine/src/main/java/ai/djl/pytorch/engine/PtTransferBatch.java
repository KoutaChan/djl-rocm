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

import ai.djl.ndarray.types.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** A batch of pinned host-to-device transfers enqueued on one PyTorch stream. */
public final class PtTransferBatch implements AutoCloseable {

    private final PtNDManager manager;
    private final PtEvent completionEvent;
    private final List<PtNDArray> arrays;
    private final List<PtPinnedBuffer> buffers;
    private final Thread ownerThread;
    private PtStreamScope scope;
    private boolean committed;

    PtTransferBatch(PtStream stream, PtNDManager manager) {
        this.manager = Objects.requireNonNull(manager, "manager");
        if (!stream.getDevice().equals(manager.getDevice())) {
            throw new IllegalArgumentException(
                    "The transfer stream device is: "
                            + stream.getDevice()
                            + ", but the manager device is: "
                            + manager.getDevice());
        }
        completionEvent = stream.newEvent();
        arrays = new ArrayList<>();
        buffers = new ArrayList<>();
        ownerThread = Thread.currentThread();
        try {
            scope = stream.openScope();
        } catch (RuntimeException | Error e) {
            completionEvent.close();
            throw e;
        }
    }

    /**
     * Enqueues creation of an array from a pinned host buffer.
     *
     * <p>The returned array is owned by the manager supplied when this batch was created. It must
     * not be consumed before the committed ticket is handed off to the consumer stream.
     *
     * @param buffer the source host transfer buffer
     * @param shape the shape of the destination array
     * @return the destination array
     */
    public PtNDArray copy(PtPinnedBuffer buffer, Shape shape) {
        ensureOpen();
        PtNDArray array = manager.createFromPinnedBuffer(buffer, shape);
        arrays.add(array);
        buffers.add(buffer);
        return array;
    }

    /**
     * Records completion of all transfers currently enqueued in this batch.
     *
     * <p>Committing restores the stream that was current before the batch was opened. The returned
     * ticket retains the source buffers and destination arrays until it is closed.
     *
     * @return a ticket representing completion of this batch
     * @throws IllegalStateException if this batch is empty or has already been closed
     */
    public PtTransferTicket commit() {
        ensureOpen();
        if (arrays.isEmpty()) {
            throw new IllegalStateException("Cannot commit an empty transfer batch.");
        }
        completionEvent.record();
        try {
            scope.close();
        } catch (RuntimeException | Error e) {
            scope = null;
            abortRecordedTransfer(e);
            throw e;
        }
        scope = null;
        PtTransferTicket ticket = new PtTransferTicket(completionEvent, arrays, buffers);
        committed = true;
        arrays.clear();
        buffers.clear();
        return ticket;
    }

    /**
     * Aborts an uncommitted transfer batch.
     *
     * <p>If work has already been enqueued, aborting waits for that work before releasing its
     * arrays. This blocking cleanup is only used for an uncommitted or exceptional path. A
     * committed batch is owned by its {@link PtTransferTicket}.
     */
    @Override
    public void close() {
        if (committed || scope == null) {
            return;
        }
        ensureOwnerThread();
        Throwable failure = null;
        try {
            if (!arrays.isEmpty()) {
                completionEvent.record();
            }
        } catch (RuntimeException | Error e) {
            failure = e;
        }
        try {
            scope.close();
        } catch (RuntimeException | Error e) {
            failure = addSuppressed(failure, e);
        } finally {
            scope = null;
        }
        try {
            if (!arrays.isEmpty()) {
                completionEvent.synchronize();
            }
        } catch (RuntimeException | Error e) {
            failure = addSuppressed(failure, e);
        }
        try {
            completionEvent.close();
        } catch (RuntimeException | Error e) {
            failure = addSuppressed(failure, e);
        }
        for (PtNDArray array : arrays) {
            try {
                array.close();
            } catch (RuntimeException | Error e) {
                failure = addSuppressed(failure, e);
            }
        }
        arrays.clear();
        buffers.clear();
        rethrow(failure);
    }

    private void ensureOpen() {
        ensureOwnerThread();
        if (scope == null || committed) {
            throw new IllegalStateException("PtTransferBatch is closed.");
        }
    }

    private void ensureOwnerThread() {
        if (Thread.currentThread() != ownerThread) {
            throw new IllegalStateException(
                    "PtTransferBatch must be used by the thread that opened it.");
        }
    }

    private void abortRecordedTransfer(Throwable failure) {
        try {
            completionEvent.synchronize();
        } catch (RuntimeException | Error e) {
            failure.addSuppressed(e);
        } finally {
            completionEvent.close();
            for (PtNDArray array : arrays) {
                array.close();
            }
            arrays.clear();
            buffers.clear();
        }
    }

    private static Throwable addSuppressed(Throwable failure, Throwable suppressed) {
        if (failure == null) {
            return suppressed;
        }
        failure.addSuppressed(suppressed);
        return failure;
    }

    private static void rethrow(Throwable failure) {
        if (failure instanceof RuntimeException) {
            throw (RuntimeException) failure;
        }
        if (failure instanceof Error) {
            throw (Error) failure;
        }
    }
}
