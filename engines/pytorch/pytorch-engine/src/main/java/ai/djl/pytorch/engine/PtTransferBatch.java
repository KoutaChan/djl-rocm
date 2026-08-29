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
import java.util.Objects;

/**
 * A batch of pinned host-to-device transfers enqueued on one PyTorch stream.
 *
 * <p>A batch must be used and closed by the thread that created it.
 */
public final class PtTransferBatch implements AutoCloseable {

    private final PtNDManager manager;
    private final PtEvent completionEvent;
    private final ArrayList<PtNDArray> arrays;
    private final ArrayList<PtPinnedBuffer> buffers;
    private final Thread ownerThread;
    private PtStreamScope scope;
    private boolean committed;
    private boolean failed;
    private boolean recorded;
    private boolean released;
    private boolean copyComplete;

    PtTransferBatch(PtStream stream, PtNDManager manager) {
        this.manager = Objects.requireNonNull(manager, "manager");
        if (!stream.getDevice().equals(manager.getDevice())) {
            throw new IllegalArgumentException(
                    "The transfer stream device is: "
                            + stream.getDevice()
                            + ", but the manager device is: "
                            + manager.getDevice());
        }
        arrays = new ArrayList<>(1);
        buffers = new ArrayList<>(1);
        ownerThread = Thread.currentThread();
        completionEvent = stream.newEvent();
        try {
            scope = stream.openScope();
        } catch (RuntimeException | Error e) {
            try {
                completionEvent.close();
            } catch (RuntimeException | Error suppressed) {
                addSuppressed(e, suppressed);
            }
            throw e;
        }
    }

    /**
     * Creates an array and enqueues a copy from a pinned host buffer.
     *
     * <p>The returned array is owned by the manager supplied when this batch was created. It must
     * not be consumed before the consumer stream waits on the committed ticket.
     *
     * @param buffer the source host transfer buffer
     * @param shape the shape of the destination array
     * @return the destination array
     */
    public PtNDArray copy(PtPinnedBuffer buffer, Shape shape) {
        ensureOpen();
        validate(buffer, shape);
        int capacity = Math.addExact(arrays.size(), 1);
        arrays.ensureCapacity(capacity);
        buffers.ensureCapacity(capacity);
        PtNDArray array = manager.create(shape, buffer.getDataType());
        arrays.add(array);
        buffers.add(buffer);
        try {
            array.enqueueCopyFrom(buffer);
            return array;
        } catch (RuntimeException | Error e) {
            failed = true;
            throw e;
        }
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
        recorded = true;
        try {
            try {
                scope.close();
            } finally {
                scope = null;
            }
            PtTransferTicket ticket = new PtTransferTicket(completionEvent, arrays, buffers);
            committed = true;
            arrays.clear();
            buffers.clear();
            return ticket;
        } catch (RuntimeException | Error e) {
            release(e);
            throw e;
        }
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
        if (committed || released) {
            return;
        }
        ensureOwnerThread();
        if (!buffers.isEmpty() && !recorded) {
            completionEvent.record();
            recorded = true;
        }
        Throwable failure = release(null);
        rethrow(failure);
    }

    private void ensureOpen() {
        ensureOwnerThread();
        if (scope == null || committed || failed || recorded || released) {
            throw new IllegalStateException("PtTransferBatch is closed.");
        }
    }

    private void ensureOwnerThread() {
        if (Thread.currentThread() != ownerThread) {
            throw new IllegalStateException(
                    "PtTransferBatch must be used by the thread that opened it.");
        }
    }

    private Throwable release(Throwable failure) {
        if (scope != null) {
            try {
                scope.close();
            } catch (RuntimeException | Error e) {
                failure = addSuppressed(failure, e);
            } finally {
                scope = null;
            }
        }
        if (!buffers.isEmpty() && !copyComplete) {
            try {
                completionEvent.synchronize();
                copyComplete = true;
            } catch (RuntimeException | Error e) {
                return addSuppressed(failure, e);
            }
        }
        if (!completionEvent.isReleased()) {
            try {
                completionEvent.close();
            } catch (RuntimeException | Error e) {
                failure = addSuppressed(failure, e);
            }
        }
        for (int i = arrays.size() - 1; i >= 0; --i) {
            PtNDArray array = arrays.get(i);
            if (!array.isReleased()) {
                try {
                    array.close();
                } catch (RuntimeException | Error e) {
                    failure = addSuppressed(failure, e);
                }
            }
            if (array.isReleased()) {
                arrays.remove(i);
            }
        }
        if (copyComplete) {
            buffers.clear();
        }
        released = completionEvent.isReleased() && arrays.isEmpty() && buffers.isEmpty();
        return failure;
    }

    private static void validate(PtPinnedBuffer buffer, Shape shape) {
        Objects.requireNonNull(buffer, "buffer");
        Objects.requireNonNull(shape, "shape");
        buffer.getHandle();
        long size = shape.size();
        if (size <= 0) {
            throw new IllegalArgumentException("shape must contain at least one element.");
        }
        if (size > buffer.size()) {
            throw new IllegalArgumentException(
                    "The requested array contains "
                            + size
                            + " elements, but the transfer buffer contains "
                            + buffer.size());
        }
    }

    private static Throwable addSuppressed(Throwable failure, Throwable suppressed) {
        if (failure == null) {
            return suppressed;
        }
        if (failure != suppressed) {
            failure.addSuppressed(suppressed);
        }
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
