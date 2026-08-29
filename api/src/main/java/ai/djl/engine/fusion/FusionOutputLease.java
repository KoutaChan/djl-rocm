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
package ai.djl.engine.fusion;

import ai.djl.ndarray.NDArray;

/**
 * A lease that keeps one session-owned persistent output slot from being reused.
 *
 * <p>Lease method executions participate in the owning session's externally serialized,
 * non-thread-safe contract and must not overlap method executions on that session or its other
 * handles. Outstanding handle lifetimes may coexist on distinct slots. A closed lease is invalid;
 * implementations may reuse its underlying slot and storage, but not the public handle. Callers
 * must not retain or use a reference after closing it.
 */
public interface FusionOutputLease extends AutoCloseable {

    /**
     * Returns the maximum-capacity storage for an output.
     *
     * <p>The active leading extent is available through {@link #getDimension}. The returned array
     * is owned by the session. The caller must not close it, attach it to another manager, or use
     * it after this lease is closed.
     *
     * @param output the recipe output handle
     * @return the persistent output tensor
     */
    NDArray get(FusionRecipe.Output output);

    /**
     * Returns the active extent used for a named dimension in this submission.
     *
     * @param dimension the recipe dimension handle
     * @return the active extent
     */
    long getDimension(FusionRecipe.Dimension dimension);

    /**
     * Blocks the calling thread until work producing this output slot has completed.
     *
     * <p>This method neither closes the lease nor releases its ring slot. Callers that only enqueue
     * downstream work on the same device stream do not need this synchronization.
     */
    void synchronize();

    /**
     * Releases the output ring slot without synchronizing the device.
     *
     * <p>The caller must ensure that the fusion producer and every device or host transfer
     * consuming its output have completed before closing the lease. Work ordered after the producer
     * on the same stream satisfies the producer dependency, but a slot that may next be submitted
     * on a different stream requires explicit event completion or {@link #synchronize()} first. In
     * particular, an asynchronous device-to-host transfer on another stream must finish before the
     * slot can be reused. No method may be invoked through this reference after it is closed.
     */
    @Override
    void close();
}
