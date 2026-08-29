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

/** Immutable configuration for a {@link FusionSession}. */
public final class FusionSessionConfig {

    private final int bufferCount;

    private FusionSessionConfig(Builder builder) {
        bufferCount = builder.bufferCount;
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
     * Returns the number of persistent output and workspace slots.
     *
     * @return the ring buffer count
     */
    public int getBufferCount() {
        return bufferCount;
    }

    /** Builds an immutable {@link FusionSessionConfig}. */
    public static final class Builder {

        private int bufferCount;

        private Builder() {
            bufferCount = 1;
        }

        /**
         * Sets the number of persistent output and workspace slots.
         *
         * @param bufferCount the ring buffer count
         * @return this builder
         */
        public Builder optBufferCount(int bufferCount) {
            if (bufferCount <= 0) {
                throw new IllegalArgumentException("The buffer count must be positive.");
            }
            this.bufferCount = bufferCount;
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
