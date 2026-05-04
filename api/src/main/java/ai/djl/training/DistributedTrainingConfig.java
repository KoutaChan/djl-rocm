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
package ai.djl.training;

/** Configuration for native distributed data parallel training. */
public final class DistributedTrainingConfig {

    private int rank;
    private int worldSize;
    private int localRank;
    private String masterHost;
    private int masterPort;
    private int bucketCapMb;
    private boolean staticGraph;
    private boolean findUnusedParameters;
    private boolean averageGradients;

    private DistributedTrainingConfig(Builder builder) {
        rank = builder.rank;
        worldSize = builder.worldSize;
        localRank = builder.localRank;
        masterHost = builder.masterHost;
        masterPort = builder.masterPort;
        bucketCapMb = builder.bucketCapMb;
        staticGraph = builder.staticGraph;
        findUnusedParameters = builder.findUnusedParameters;
        averageGradients = builder.averageGradients;
    }

    /**
     * Returns the global rank of this worker.
     *
     * @return the global rank
     */
    public int getRank() {
        return rank;
    }

    /**
     * Returns the number of participating workers.
     *
     * @return the world size
     */
    public int getWorldSize() {
        return worldSize;
    }

    /**
     * Returns the device-local rank for this worker.
     *
     * @return the local rank
     */
    public int getLocalRank() {
        return localRank;
    }

    /**
     * Returns the rendezvous host.
     *
     * @return the rendezvous host
     */
    public String getMasterHost() {
        return masterHost;
    }

    /**
     * Returns the rendezvous port.
     *
     * @return the rendezvous port
     */
    public int getMasterPort() {
        return masterPort;
    }

    /**
     * Returns the all-reduce bucket capacity in MiB.
     *
     * @return the bucket capacity in MiB
     */
    public int getBucketCapMb() {
        return bucketCapMb;
    }

    /**
     * Returns whether the model graph is static across iterations.
     *
     * @return {@code true} if the model graph is static
     */
    public boolean isStaticGraph() {
        return staticGraph;
    }

    /**
     * Returns whether unused parameter detection is enabled.
     *
     * @return {@code true} if unused parameter detection is enabled
     */
    public boolean isFindUnusedParameters() {
        return findUnusedParameters;
    }

    /**
     * Returns whether all-reduced gradients are divided by world size.
     *
     * @return {@code true} if gradients are averaged
     */
    public boolean isAverageGradients() {
        return averageGradients;
    }

    /**
     * Creates a new distributed training config builder.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** Builder for {@link DistributedTrainingConfig}. */
    public static final class Builder {

        private int rank;
        private int worldSize = 1;
        private int localRank;
        private String masterHost = "127.0.0.1";
        private int masterPort = 29500;
        private int bucketCapMb = 25;
        private boolean staticGraph = true;
        private boolean findUnusedParameters;
        private boolean averageGradients = true;

        private Builder() {}

        /**
         * Sets the global rank of this worker.
         *
         * @param rank the global rank
         * @return this builder
         */
        public Builder optRank(int rank) {
            this.rank = rank;
            return this;
        }

        /**
         * Sets the number of participating workers.
         *
         * @param worldSize the world size
         * @return this builder
         */
        public Builder optWorldSize(int worldSize) {
            this.worldSize = worldSize;
            return this;
        }

        /**
         * Sets the device-local rank for this worker.
         *
         * @param localRank the local rank
         * @return this builder
         */
        public Builder optLocalRank(int localRank) {
            this.localRank = localRank;
            return this;
        }

        /**
         * Sets the rendezvous host.
         *
         * @param masterHost the rendezvous host
         * @return this builder
         */
        public Builder optMasterHost(String masterHost) {
            this.masterHost = masterHost;
            return this;
        }

        /**
         * Sets the rendezvous port.
         *
         * @param masterPort the rendezvous port
         * @return this builder
         */
        public Builder optMasterPort(int masterPort) {
            this.masterPort = masterPort;
            return this;
        }

        /**
         * Sets the all-reduce bucket capacity.
         *
         * @param bucketCapMb the bucket capacity in MiB
         * @return this builder
         */
        public Builder optBucketCapMb(int bucketCapMb) {
            this.bucketCapMb = bucketCapMb;
            return this;
        }

        /**
         * Sets whether the model graph is static across iterations.
         *
         * @param staticGraph whether the graph is static
         * @return this builder
         */
        public Builder optStaticGraph(boolean staticGraph) {
            this.staticGraph = staticGraph;
            return this;
        }

        /**
         * Sets whether unused parameter detection is enabled.
         *
         * @param findUnusedParameters whether unused parameter detection is enabled
         * @return this builder
         */
        public Builder optFindUnusedParameters(boolean findUnusedParameters) {
            this.findUnusedParameters = findUnusedParameters;
            return this;
        }

        /**
         * Sets whether all-reduced gradients are divided by world size.
         *
         * @param averageGradients whether gradients are averaged
         * @return this builder
         */
        public Builder optAverageGradients(boolean averageGradients) {
            this.averageGradients = averageGradients;
            return this;
        }

        /**
         * Builds the distributed training config.
         *
         * @return the distributed training config
         */
        public DistributedTrainingConfig build() {
            if (worldSize < 1) {
                throw new IllegalArgumentException("worldSize must be at least 1.");
            }
            if (rank < 0 || rank >= worldSize) {
                throw new IllegalArgumentException("rank must be in [0, worldSize).");
            }
            if (localRank < 0) {
                throw new IllegalArgumentException("localRank must be non-negative.");
            }
            if (masterHost == null || masterHost.isEmpty()) {
                throw new IllegalArgumentException("masterHost must not be empty.");
            }
            if (masterPort <= 0 || masterPort > 65535) {
                throw new IllegalArgumentException("masterPort must be in [1, 65535].");
            }
            if (bucketCapMb <= 0) {
                throw new IllegalArgumentException("bucketCapMb must be positive.");
            }
            if (findUnusedParameters && staticGraph) {
                throw new IllegalArgumentException(
                        "findUnusedParameters requires staticGraph=false.");
            }
            return new DistributedTrainingConfig(this);
        }
    }
}
