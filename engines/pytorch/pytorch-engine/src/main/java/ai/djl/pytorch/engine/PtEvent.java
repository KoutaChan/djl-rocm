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

/** A reusable PyTorch device event. */
public final class PtEvent extends NativeResource<Long> {

    private final Device device;
    private boolean recorded;

    PtEvent(Device device) {
        super(JniUtils.createDeviceEvent(Objects.requireNonNull(device, "device")));
        this.device = device;
    }

    /**
     * Returns the device associated with this event.
     *
     * @return the event device
     */
    public Device getDevice() {
        return device;
    }

    /** Records this event on the stream that is current for the event device. */
    public synchronized void record() {
        JniUtils.recordDeviceEvent(getHandle());
        recorded = true;
    }

    /**
     * Makes the current stream wait for this event.
     *
     * <p>This method only enqueues a device-side dependency and does not wait for the event on the
     * calling thread.
     *
     * @throws IllegalStateException if this event has not been recorded
     */
    public synchronized void waitOnCurrentStream() {
        if (!recorded) {
            throw new IllegalStateException("Cannot wait for an event before it is recorded.");
        }
        JniUtils.waitDeviceEvent(getHandle());
    }

    /**
     * Returns whether the most recently recorded work has completed.
     *
     * <p>An event that has not been recorded is complete.
     *
     * @return {@code true} if the event has completed
     */
    public synchronized boolean isComplete() {
        return !recorded || JniUtils.queryDeviceEvent(getHandle());
    }

    /** Waits on the calling thread until the most recently recorded work has completed. */
    public synchronized void synchronize() {
        if (recorded) {
            JniUtils.synchronizeDeviceEvent(getHandle());
        }
    }

    /**
     * Releases the native event without synchronizing the device.
     *
     * <p>The caller must ensure that no future operation refers to this event. Call {@link
     * #synchronize()} explicitly when host-side resources protected by the event are about to be
     * reused.
     */
    @Override
    public synchronized void close() {
        onClose();
        Long pointer = handle.getAndSet(null);
        if (pointer != null && pointer != 0) {
            JniUtils.deleteDeviceEvent(pointer);
        }
    }
}
