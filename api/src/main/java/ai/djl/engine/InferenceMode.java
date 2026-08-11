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
package ai.djl.engine;

/**
 * A thread-local scope that disables automatic-gradient recording during inference.
 *
 * <p>The scope does not change parameters, operator results, or mixed-precision settings. It only
 * prevents the engine from constructing backward metadata for operations executed by the current
 * thread. Closing the scope restores the previous gradient-recording state, so scopes may be nested
 * and may safely surround an {@link Autocast} scope.
 *
 * <p>Inference executors should open this scope on the thread that performs the forward pass.
 * Gradient-recording state is thread-local in engines such as PyTorch; disabling it only when the
 * engine is initialized does not affect subsequently created worker threads.
 */
public interface InferenceMode extends AutoCloseable {

    /** Restores the inference and gradient state that was active before this scope was opened. */
    @Override
    void close();
}
