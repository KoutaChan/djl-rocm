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
import ai.djl.util.NativeResource;

import java.util.Objects;

/** A reusable PyTorch device stream. */
public final class PtStream extends NativeResource<Long> {

    private final Device device;
    private boolean active;

    PtStream(Device device) {
        super(JniUtils.createDeviceStream(Objects.requireNonNull(device, "device")));
        this.device = device;
    }

    /**
     * Returns the device associated with this stream.
     *
     * @return the stream device
     */
    public Device getDevice() {
        return device;
    }

    /**
     * Returns the native stream identifier used by {@link PtAllocatorSnapshot}.
     *
     * <p>This opaque value is the native stream pointer, not this object's JNI handle or a PyTorch
     * stream ID. Compare identifiers only within the same process and device. Zero identifies the
     * native default stream, not an unavailable value.
     *
     * @return the native stream identifier as a 64-bit bit pattern
     * @throws IllegalArgumentException if this is not a GPU stream
     * @throws IllegalStateException if this stream is closed
     */
    public synchronized long getId() {
        if (!device.isGpu()) {
            throw new IllegalArgumentException("Native stream identifiers require a GPU stream.");
        }
        return JniUtils.getStreamId(getHandle());
    }

    /**
     * Makes this stream current for the calling thread.
     *
     * <p>A stream can be reused by different threads, but only one scope may be active at a time.
     * The returned scope must be closed by the thread that opened it.
     *
     * @return a scope that restores the previously current stream when closed
     * @throws IllegalStateException if this stream already has an active scope
     */
    public synchronized PtStreamScope openScope() {
        getHandle();
        if (active) {
            throw new IllegalStateException("PtStream already has an active scope.");
        }
        active = true;
        try {
            return new PtStreamScope(this);
        } catch (RuntimeException | Error e) {
            active = false;
            throw e;
        }
    }

    /**
     * Creates an event associated with this stream's device.
     *
     * <p>The event is not recorded until {@link PtEvent#record()} is called.
     *
     * @return a new device event
     */
    public synchronized PtEvent newEvent() {
        getHandle();
        return new PtEvent(device);
    }

    /**
     * Starts a batch of pinned host-to-device transfers on this stream.
     *
     * <p>The returned batch opens this stream on the calling thread. The batch must be committed or
     * closed before another scope can be opened on this stream.
     *
     * @param manager the manager that owns the destination arrays
     * @return a new transfer batch
     */
    public PtTransferBatch newTransferBatch(PtNDManager manager) {
        return new PtTransferBatch(this, manager);
    }

    synchronized void closeScope() {
        if (!active) {
            throw new IllegalStateException("PtStream does not have an active scope.");
        }
        active = false;
    }

    /** Releases the native stream handle without synchronizing the device. */
    @Override
    public synchronized void close() {
        if (active) {
            throw new IllegalStateException("Cannot close a PtStream with an active scope.");
        }
        onClose();
        Long pointer = handle.getAndSet(null);
        if (pointer != null && pointer != 0) {
            JniUtils.deleteDeviceStream(pointer);
        }
    }
}
