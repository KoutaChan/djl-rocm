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

/** An immutable, device-specific lowering of a {@link FusionRecipe}. */
public interface FusionPlan extends AutoCloseable {

    /**
     * Returns the recipe represented by this plan.
     *
     * @return the fusion recipe
     */
    FusionRecipe getRecipe();

    /**
     * Returns diagnostics describing the prepared native plan.
     *
     * @return the compilation report
     */
    FusionCompilationReport getCompilationReport();

    /**
     * Binds model constants and creates an executable.
     *
     * <p>Constants are caller-owned and must remain valid until the executable is closed. An
     * implementation may prepack constants while binding them. In particular, a backend may project
     * fixed constant inputs of an {@link FusionRecipe.AffineSum} once and retain the projected
     * contribution in the executable. Completion dependencies created during binding are part of
     * the executable and must be honored by sessions submitting on another stream.
     *
     * @param constants the constant tensors for this recipe
     * @return an executable with bound constants
     */
    FusionExecutable bind(FusionConstantBindings constants);

    /**
     * Releases resources owned by this prepared plan.
     *
     * <p>Executables created before this call retain their own backend resources and remain valid.
     */
    @Override
    void close();
}
