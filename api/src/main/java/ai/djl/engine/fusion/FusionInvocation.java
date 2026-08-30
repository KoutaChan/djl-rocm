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
 * A single-use handle configuring one submission on a reusable {@link FusionSession} ring slot.
 *
 * <p>Invocation method executions participate in the owning session's externally serialized,
 * non-thread-safe contract and must not overlap method executions on that session or its other
 * handles. Outstanding handle lifetimes may coexist on distinct slots. A closed invocation is
 * invalid; implementations may reuse its underlying slot and storage, but not the public handle.
 * Callers must not retain or use a reference after closing it.
 */
public interface FusionInvocation extends AutoCloseable {

    /**
     * Sets or replaces an input for this invocation.
     *
     * <p>The input is caller-owned. It may be a persistent tensor reused across invocations or an
     * ephemeral tensor produced by the current forward pass. It must remain valid until the
     * returned output lease is safe to release. Repeated calls may update a preallocated native
     * handle table; they do not change recipe topology.
     *
     * @param input the recipe input handle
     * @param array the input tensor
     */
    void setInput(FusionRecipe.Input input, NDArray array);

    /**
     * Sets the active extent of a named leading dimension.
     *
     * @param dimension the recipe dimension handle
     * @param extent the active extent, between zero and the configured maximum
     */
    void setDimension(FusionRecipe.Dimension dimension, long extent);

    /**
     * Enqueues this invocation on the engine-current device stream.
     *
     * <p>This operation does not transfer inputs from host memory, transfer outputs to host memory,
     * or synchronize the device. The backend submits the already-bound plan using the configured
     * input handles and runtime extents. Implementations should cross the hot native boundary once.
     * The returned lease owns the ring slot until it is closed. Work submitted through the
     * engine-current stream is ordered with consumers subsequently enqueued on that same stream. A
     * consumer on another stream requires an explicit event/wait dependency. If submission fails
     * after the backend may have enqueued work, the backend must either establish completion before
     * returning or retain every referenced resource until the owning session is closed
     * successfully.
     *
     * @return a lease for the persistent output slot
     */
    FusionOutputLease submit();

    /**
     * Releases an invocation that has not been submitted.
     *
     * <p>Closing an invocation after a successful submission does not release the output slot; the
     * resulting {@link FusionOutputLease} owns it. This method does not synchronize the device. No
     * method may be invoked through this reference after it is closed.
     */
    @Override
    void close();
}
