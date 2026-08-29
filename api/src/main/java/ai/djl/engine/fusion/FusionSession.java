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

/**
 * An externally serialized fusion execution lane with persistent output and workspace slots.
 *
 * <p>Method executions through a session, its invocations, and its output leases are not
 * thread-safe and must be externally serialized. Outstanding invocation or lease lifetimes may
 * coexist on distinct ring slots; only their method executions must not overlap. Sequential calls
 * may move between threads after the preceding call completes.
 */
public interface FusionSession extends AutoCloseable {

    /**
     * Returns the recipe executed by this session.
     *
     * @return the fusion recipe
     */
    FusionRecipe getRecipe();

    /**
     * Acquires an available ring slot for one invocation.
     *
     * <p>A slot remains unavailable until the invocation is closed without submission or its
     * resulting {@link FusionOutputLease} is closed. Implementations may fail immediately when no
     * slot is available; this method does not imply device synchronization. An invocation is a
     * single-use handle over an underlying reusable slot, so callers must not retain or use the
     * handle after closing it.
     *
     * @return an invocation backed by an available ring slot
     */
    FusionInvocation acquire();

    /**
     * Releases session-owned output storage and workspace.
     *
     * <p>The caller must close or otherwise finish every invocation and output lease before closing
     * the session.
     */
    @Override
    void close();
}
