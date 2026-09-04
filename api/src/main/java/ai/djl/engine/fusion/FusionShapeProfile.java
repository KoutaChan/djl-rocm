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

import java.util.Arrays;
import java.util.Objects;

/** A bounded combination of dimension capacities for backend storage specialization. */
public final class FusionShapeProfile {

    private final FusionRecipe recipe;
    private final long[] capacities;

    private FusionShapeProfile(Builder builder) {
        recipe = builder.recipe;
        capacities = builder.capacities.clone();
    }

    /**
     * Creates a shape profile builder for a recipe.
     *
     * @param recipe the recipe whose dimensions are specialized
     * @return a new builder
     */
    public static Builder builder(FusionRecipe recipe) {
        return new Builder(recipe);
    }

    /**
     * Returns the recipe associated with this profile.
     *
     * @return the fusion recipe
     */
    public FusionRecipe getRecipe() {
        return recipe;
    }

    /**
     * Returns whether this profile specifies a capacity for a dimension.
     *
     * @param dimension the recipe dimension
     * @return {@code true} if the dimension is specialized
     */
    public boolean hasCapacity(FusionRecipe.Dimension dimension) {
        int index = checkedIndex(dimension);
        return capacities[index] >= 0;
    }

    /**
     * Returns the specialized capacity for a dimension.
     *
     * @param dimension the recipe dimension
     * @return the specialized capacity
     * @throws IllegalArgumentException if the profile does not specify the dimension
     */
    public long getCapacity(FusionRecipe.Dimension dimension) {
        int index = checkedIndex(dimension);
        long capacity = capacities[index];
        if (capacity < 0) {
            throw new IllegalArgumentException(
                    "The shape profile does not specify dimension: " + dimension.getName());
        }
        return capacity;
    }

    /**
     * Returns the specialization capacities in recipe dimension order.
     *
     * <p>An unspecified dimension is represented by {@code -1}.
     *
     * @return a copy of the specialization capacities
     */
    public long[] getCapacities() {
        return capacities.clone();
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof FusionShapeProfile)) {
            return false;
        }
        FusionShapeProfile profile = (FusionShapeProfile) other;
        return recipe == profile.recipe && Arrays.equals(capacities, profile.capacities);
    }

    /** {@inheritDoc} */
    @Override
    public int hashCode() {
        return 31 * System.identityHashCode(recipe) + Arrays.hashCode(capacities);
    }

    private int checkedIndex(FusionRecipe.Dimension dimension) {
        Objects.requireNonNull(dimension, "dimension");
        int index = dimension.getIndex();
        if (index < 0
                || index >= recipe.getDimensions().size()
                || recipe.getDimensions().get(index) != dimension) {
            throw new IllegalArgumentException("The dimension belongs to a different recipe.");
        }
        return index;
    }

    /** Builds an immutable {@link FusionShapeProfile}. */
    public static final class Builder {

        private final FusionRecipe recipe;
        private final long[] capacities;

        private Builder(FusionRecipe recipe) {
            this.recipe = Objects.requireNonNull(recipe, "recipe");
            capacities = new long[recipe.getDimensions().size()];
            Arrays.fill(capacities, -1);
        }

        /**
         * Sets a specialized storage capacity.
         *
         * @param dimension the recipe dimension
         * @param capacity the storage capacity
         * @return this builder
         */
        public Builder setCapacity(FusionRecipe.Dimension dimension, long capacity) {
            int index = checkedIndex(dimension);
            if (capacity <= 0 || capacity > dimension.getMaximumExtent()) {
                throw new IllegalArgumentException(
                        "The capacity must be positive and not exceed the dimension maximum.");
            }
            capacities[index] = capacity;
            return this;
        }

        /**
         * Builds the shape profile.
         *
         * @return the shape profile
         */
        public FusionShapeProfile build() {
            return new FusionShapeProfile(this);
        }

        private int checkedIndex(FusionRecipe.Dimension dimension) {
            Objects.requireNonNull(dimension, "dimension");
            int index = dimension.getIndex();
            if (index < 0
                    || index >= recipe.getDimensions().size()
                    || recipe.getDimensions().get(index) != dimension) {
                throw new IllegalArgumentException("The dimension belongs to a different recipe.");
            }
            return index;
        }
    }
}
