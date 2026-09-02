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

/** A preferred combination of active dimension extents for backend specialization. */
public final class FusionShapeProfile {

    private final FusionRecipe recipe;
    private final long[] extents;

    private FusionShapeProfile(Builder builder) {
        recipe = builder.recipe;
        extents = builder.extents.clone();
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
     * Returns whether this profile specifies an extent for a dimension.
     *
     * @param dimension the recipe dimension
     * @return {@code true} if the dimension is specialized
     */
    public boolean hasExtent(FusionRecipe.Dimension dimension) {
        int index = checkedIndex(dimension);
        return extents[index] >= 0;
    }

    /**
     * Returns the specialized extent for a dimension.
     *
     * @param dimension the recipe dimension
     * @return the specialized extent
     * @throws IllegalArgumentException if the profile does not specify the dimension
     */
    public long getExtent(FusionRecipe.Dimension dimension) {
        int index = checkedIndex(dimension);
        long extent = extents[index];
        if (extent < 0) {
            throw new IllegalArgumentException(
                    "The shape profile does not specify dimension: " + dimension.getName());
        }
        return extent;
    }

    /**
     * Returns the specialization extents in recipe dimension order.
     *
     * <p>An unspecified dimension is represented by {@code -1}.
     *
     * @return a copy of the specialization extents
     */
    public long[] getExtents() {
        return extents.clone();
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
        private final long[] extents;

        private Builder(FusionRecipe recipe) {
            this.recipe = Objects.requireNonNull(recipe, "recipe");
            extents = new long[recipe.getDimensions().size()];
            Arrays.fill(extents, -1);
        }

        /**
         * Sets a preferred active extent.
         *
         * @param dimension the recipe dimension
         * @param extent the preferred active extent
         * @return this builder
         */
        public Builder set(FusionRecipe.Dimension dimension, long extent) {
            int index = checkedIndex(dimension);
            if (extent < 0 || extent > dimension.getMaximumExtent()) {
                throw new IllegalArgumentException(
                        "The extent must be between zero and the dimension maximum.");
            }
            extents[index] = extent;
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
