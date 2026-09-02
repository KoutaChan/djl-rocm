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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/** Immutable options used when preparing a {@link FusionRecipe}. */
public final class FusionCompileConfig {

    private final FallbackMode fallbackMode;
    private final List<FusionShapeProfile> shapeProfiles;

    private FusionCompileConfig(Builder builder) {
        fallbackMode = builder.fallbackMode;
        shapeProfiles = Collections.unmodifiableList(new ArrayList<>(builder.shapeProfiles));
    }

    /**
     * Returns the default compile configuration.
     *
     * @return the default compile configuration
     */
    public static FusionCompileConfig defaults() {
        return builder().build();
    }

    /**
     * Creates a builder for a compile configuration.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Returns the fallback policy.
     *
     * @return the fallback policy
     */
    public FallbackMode getFallbackMode() {
        return fallbackMode;
    }

    /**
     * Returns the preferred runtime shape specializations.
     *
     * @return the shape profiles
     */
    public List<FusionShapeProfile> getShapeProfiles() {
        return shapeProfiles;
    }

    /** Defines how an engine handles recipe values that it cannot lower. */
    public enum FallbackMode {
        /** Requires every value to be lowered to the engine's native execution plan. */
        REQUIRED
    }

    /** Builds an immutable {@link FusionCompileConfig}. */
    public static final class Builder {

        private FallbackMode fallbackMode;
        private List<FusionShapeProfile> shapeProfiles;

        private Builder() {
            fallbackMode = FallbackMode.REQUIRED;
            shapeProfiles = new ArrayList<>();
        }

        /**
         * Sets the behavior for recipe values that cannot be lowered.
         *
         * @param fallbackMode the fallback policy
         * @return this builder
         */
        public Builder optFallbackMode(FallbackMode fallbackMode) {
            this.fallbackMode = Objects.requireNonNull(fallbackMode, "fallbackMode");
            return this;
        }

        /**
         * Adds a preferred runtime shape specialization.
         *
         * @param shapeProfile the shape profile
         * @return this builder
         */
        public Builder addShapeProfile(FusionShapeProfile shapeProfile) {
            shapeProfiles.add(Objects.requireNonNull(shapeProfile, "shapeProfile"));
            return this;
        }

        /**
         * Builds the compile configuration.
         *
         * @return the compile configuration
         */
        public FusionCompileConfig build() {
            return new FusionCompileConfig(this);
        }
    }
}
