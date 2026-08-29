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

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/** Immutable, index-addressable model constants bound to one {@link FusionRecipe}. */
public final class FusionConstantBindings {

    private final FusionRecipe recipe;
    private final NDArray[] constants;
    private final List<NDArray> constantList;

    private FusionConstantBindings(Builder builder) {
        recipe = builder.recipe;
        constants = builder.constants.clone();
        constantList = Collections.unmodifiableList(Arrays.asList(constants));
    }

    /**
     * Creates a constant binding builder for a recipe.
     *
     * @param recipe the recipe whose constants are bound
     * @return a new builder
     */
    public static Builder builder(FusionRecipe recipe) {
        return new Builder(recipe);
    }

    /**
     * Returns the recipe associated with these bindings.
     *
     * @return the fusion recipe
     */
    public FusionRecipe getRecipe() {
        return recipe;
    }

    /**
     * Returns a bound constant.
     *
     * @param constant the recipe constant handle
     * @return the bound tensor
     */
    public NDArray get(FusionRecipe.Constant constant) {
        int index = checkedIndex(constant);
        return constants[index];
    }

    /**
     * Returns the constants in stable recipe order.
     *
     * @return the bound constants
     */
    public List<NDArray> getConstants() {
        return constantList;
    }

    private int checkedIndex(FusionRecipe.Constant constant) {
        Objects.requireNonNull(constant, "constant");
        int index = constant.getConstantIndex();
        if (index < 0
                || index >= recipe.getConstants().size()
                || recipe.getConstants().get(index) != constant) {
            throw new IllegalArgumentException("The constant belongs to a different recipe.");
        }
        return index;
    }

    /** Builds immutable {@link FusionConstantBindings}. */
    public static final class Builder {

        private final FusionRecipe recipe;
        private final NDArray[] constants;

        private Builder(FusionRecipe recipe) {
            this.recipe = Objects.requireNonNull(recipe, "recipe");
            constants = new NDArray[recipe.getConstants().size()];
        }

        /**
         * Binds a caller-owned tensor to a recipe constant.
         *
         * @param constant the recipe constant handle
         * @param array the constant tensor
         * @return this builder
         */
        public Builder bind(FusionRecipe.Constant constant, NDArray array) {
            int index = checkedIndex(constant);
            constants[index] = Objects.requireNonNull(array, "array");
            return this;
        }

        /**
         * Builds the constant bindings.
         *
         * @return the constant bindings
         */
        public FusionConstantBindings build() {
            for (int i = 0; i < constants.length; ++i) {
                if (constants[i] == null) {
                    throw new IllegalStateException(
                            "Missing fusion constant: " + recipe.getConstants().get(i).getName());
                }
            }
            return new FusionConstantBindings(this);
        }

        private int checkedIndex(FusionRecipe.Constant constant) {
            Objects.requireNonNull(constant, "constant");
            int index = constant.getConstantIndex();
            if (index < 0
                    || index >= recipe.getConstants().size()
                    || recipe.getConstants().get(index) != constant) {
                throw new IllegalArgumentException("The constant belongs to a different recipe.");
            }
            return index;
        }
    }
}
