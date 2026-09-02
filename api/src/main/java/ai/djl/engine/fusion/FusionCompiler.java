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

import ai.djl.Device;

/** Prepares backend-neutral {@link FusionRecipe}s for one engine device. */
public interface FusionCompiler {

    /**
     * Returns the device targeted by this compiler.
     *
     * @return the target device
     */
    Device getDevice();

    /**
     * Prepares an immutable recipe for repeated binding and execution.
     *
     * <p>All validation, lowering, kernel selection, and workspace planning that does not depend on
     * model constants should happen before this method returns. Implementations must fail here when
     * a recipe cannot be lowered; execution does not fall back after submission.
     *
     * @param recipe the recipe to prepare
     * @return the prepared plan
     */
    default FusionPlan prepare(FusionRecipe recipe) {
        return prepare(recipe, FusionCompileConfig.defaults());
    }

    /**
     * Prepares an immutable recipe using the specified compile configuration.
     *
     * <p>Shape profiles in {@code config} are optimization hints. They do not restrict valid active
     * extents within the bounds declared by the recipe.
     *
     * @param recipe the recipe to prepare
     * @param config the compile configuration
     * @return the prepared plan
     */
    FusionPlan prepare(FusionRecipe recipe, FusionCompileConfig config);
}
