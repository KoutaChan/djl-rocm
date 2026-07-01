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

import ai.djl.pytorch.jni.JniUtils;
import ai.djl.util.NativeResource;

/**
 * {@code PtCopyEvent} represents completion of an asynchronous pinned host-to-device copy.
 *
 * <p>The event keeps a Java reference to the source pinned buffer so the buffer cannot be garbage
 * collected while the native copy is in flight. The caller must still avoid overwriting or closing
 * the pinned buffer before this event is synchronized or closed.
 */
public final class PtCopyEvent extends NativeResource<Long> {

    private final PtNDManager manager;
    @SuppressWarnings("PMD.UnusedPrivateField")
    private final PtPinnedBuffer source;

    @SuppressWarnings("this-escape")
    PtCopyEvent(PtNDManager manager, long handle, PtPinnedBuffer source) {
        super(handle);
        this.manager = manager;
        this.source = source;
        manager.attachInternal(getUid(), this);
    }

    /**
     * Waits until the asynchronous copy represented by this event has completed.
     *
     * <p>Calling this method more than once is allowed.
     */
    public void synchronize() {
        Long pointer = getHandle();
        if (pointer != 0) {
            JniUtils.synchronizeCopyEvent(pointer);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        onClose();
        Long pointer = handle.getAndSet(null);
        if (pointer != null && pointer != 0) {
            try {
                JniUtils.synchronizeCopyEvent(pointer);
            } finally {
                JniUtils.deleteCopyEvent(pointer);
            }
        }
        manager.detachInternal(getUid());
    }
}
