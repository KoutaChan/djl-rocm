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

import java.util.Objects;

/** Immutable configuration for a {@link FusionSession}. */
public final class FusionSessionConfig {

    private final int outputSlotCount;
    private final FusionShapeProfile requestedShapeProfile;
    private final ProfileFallback profileFallback;

    private FusionSessionConfig(Builder builder) {
        outputSlotCount = builder.outputSlotCount;
        requestedShapeProfile = builder.requestedShapeProfile;
        profileFallback = builder.profileFallback;
    }

    /**
     * Returns the default session configuration.
     *
     * @return the default session configuration
     */
    public static FusionSessionConfig defaults() {
        return builder().build();
    }

    /**
     * Creates a session configuration builder.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Returns the number of persistent output slots.
     *
     * @return the output slot count
     */
    public int getOutputSlotCount() {
        return outputSlotCount;
    }

    /**
     * Returns the minimum storage-capacity profile requested for the session.
     *
     * <p>A {@code null} value selects the recipe's declared maximum capacities.
     *
     * @return the requested profile, or {@code null}
     */
    public FusionShapeProfile getRequestedShapeProfile() {
        return requestedShapeProfile;
    }

    /**
     * Returns the behavior used when the requested profile is not compiled exactly.
     *
     * @return the profile fallback policy
     */
    public ProfileFallback getProfileFallback() {
        return profileFallback;
    }

    /** Controls selection of a compiled storage-capacity profile. */
    public enum ProfileFallback {
        /** Selects the smallest compiled profile that fits, or the recipe maximum. */
        SMALLEST_FITTING_OR_MAXIMUM,

        /**
         * Requires an exact compiled profile, except that the recipe maximum is always available.
         */
        EXACT
    }

    /** Builds an immutable {@link FusionSessionConfig}. */
    public static final class Builder {

        private int outputSlotCount;
        private FusionShapeProfile requestedShapeProfile;
        private ProfileFallback profileFallback;

        private Builder() {
            outputSlotCount = 1;
            profileFallback = ProfileFallback.SMALLEST_FITTING_OR_MAXIMUM;
        }

        /**
         * Sets the number of persistent output slots.
         *
         * @param outputSlotCount the output slot count
         * @return this builder
         */
        public Builder optOutputSlotCount(int outputSlotCount) {
            if (outputSlotCount <= 0) {
                throw new IllegalArgumentException("The output slot count must be positive.");
            }
            this.outputSlotCount = outputSlotCount;
            return this;
        }

        /**
         * Requests a minimum storage-capacity profile for the session.
         *
         * <p>The executable selects a compiled profile according to {@link #optProfileFallback}.
         * Profile capacities are fixed for the session lifetime.
         *
         * @param requestedShapeProfile the requested profile
         * @return this builder
         */
        public Builder optRequestedShapeProfile(FusionShapeProfile requestedShapeProfile) {
            this.requestedShapeProfile =
                    Objects.requireNonNull(requestedShapeProfile, "requestedShapeProfile");
            return this;
        }

        /**
         * Sets how a requested capacity profile is matched to compiled profiles.
         *
         * @param profileFallback the fallback policy
         * @return this builder
         */
        public Builder optProfileFallback(ProfileFallback profileFallback) {
            this.profileFallback = Objects.requireNonNull(profileFallback, "profileFallback");
            return this;
        }

        /**
         * Builds the session configuration.
         *
         * @return the session configuration
         */
        public FusionSessionConfig build() {
            return new FusionSessionConfig(this);
        }
    }
}
