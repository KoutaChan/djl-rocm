/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.ndarray.types.DataType;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.util.NativeResource;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * {@code PtPinnedBuffer} is a host transfer buffer exposed as a direct {@link ByteBuffer}.
 *
 * <p>The buffer is backed by pinned host memory when the native PyTorch build has an available
 * CUDA/ROCm accelerator. CPU-only builds fall back to regular CPU host memory so the same API can
 * still be used for CPU tensors. When this buffer is used with an asynchronous copy, the caller
 * must not overwrite or close it until the returned {@link PtCopyEvent} has been synchronized or
 * closed. A buffer submitted through a {@link PtTransferBatch} must remain unchanged until its
 * {@link PtTransferTicket} reports completion or has been synchronized.
 */
public final class PtPinnedBuffer extends NativeResource<Long> {

    private final PtNDManager manager;
    private final ByteBuffer buffer;
    private final int size;
    private final DataType dataType;
    private final int capacity;
    private final boolean pinned;

    @SuppressWarnings("this-escape")
    PtPinnedBuffer(
            PtNDManager manager,
            long handle,
            ByteBuffer buffer,
            int size,
            DataType dataType,
            boolean pinned) {
        super(handle);
        this.manager = manager;
        this.buffer = buffer.order(ByteOrder.nativeOrder());
        this.size = size;
        this.dataType = dataType;
        this.capacity = Math.multiplyExact(size, dataType.getNumOfBytes());
        this.pinned = pinned;
        manager.attachInternal(getUid(), this);
    }

    /**
     * Returns the host transfer memory as a direct byte buffer.
     *
     * <p>The returned view shares the native memory with this buffer, but has independent position
     * and limit. Closing this {@code PtPinnedBuffer} invalidates all views returned by this method.
     *
     * @return a direct {@link ByteBuffer} backed by host transfer memory
     */
    public ByteBuffer getByteBuffer() {
        getHandle();
        ByteBuffer view = buffer.duplicate();
        view.order(ByteOrder.nativeOrder());
        return view;
    }

    /**
     * Returns the number of bytes allocated in this buffer.
     *
     * @return the buffer capacity in bytes
     */
    public int capacity() {
        return capacity;
    }

    /**
     * Returns the number of typed elements allocated in this buffer.
     *
     * @return the number of elements
     */
    public int size() {
        return size;
    }

    /**
     * Returns the element type stored in this buffer.
     *
     * @return the buffer data type
     */
    public DataType getDataType() {
        return dataType;
    }

    /**
     * Returns whether this buffer is backed by pinned host memory.
     *
     * @return {@code true} when the native allocation is pinned
     */
    public boolean isPinned() {
        return pinned;
    }

    PtNDManager getManager() {
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        onClose();
        Long pointer = handle.getAndSet(null);
        if (pointer != null) {
            JniUtils.deletePinnedBuffer(pointer);
        }
        manager.detachInternal(getUid());
    }
}
