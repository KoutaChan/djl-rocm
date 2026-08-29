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

import ai.djl.ndarray.NDManager;

/**
 * A prepared fusion plan with model constants bound for repeated inference.
 *
 * <p>Method executions on the same executable are not thread-safe and must be externally
 * serialized. Sessions created from it have independent session lifecycles.
 */
public interface FusionExecutable extends AutoCloseable {

    /**
     * Returns the recipe represented by this executable.
     *
     * @return the fusion recipe
     */
    FusionRecipe getRecipe();

    /**
     * Creates an externally serialized execution session.
     *
     * <p>The session allocates {@code bufferCount} persistent output slots at each output's maximum
     * shape. Method executions using the session, its invocations, and its output leases must be
     * externally serialized, but outstanding handles may coexist on distinct slots and sequential
     * calls may move between threads. The supplied manager and its device resources must outlive
     * the session.
     *
     * @param manager the manager used to allocate session resources
     * @param config the session configuration
     * @return a new execution session
     */
    FusionSession newSession(NDManager manager, FusionSessionConfig config);

    /**
     * Releases bound constants and backend resources owned by this executable.
     *
     * <p>Every session created by this executable must be closed first.
     */
    @Override
    void close();
}
