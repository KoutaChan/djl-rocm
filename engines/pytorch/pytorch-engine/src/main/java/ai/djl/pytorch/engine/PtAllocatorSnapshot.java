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
package ai.djl.pytorch.engine;

import ai.djl.Device;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Immutable allocator metadata grouped by native stream, private pool, and small/large pool.
 *
 * <p>Values cover PyTorch caching allocator segments only. They exclude tensor contents, allocation
 * traces, driver allocations, and other libraries. The snapshot does not synchronize the device.
 */
public final class PtAllocatorSnapshot {

    private static final int POOL_FIELDS = 9;
    private final Device device;
    private final List<StreamPool> pools;

    PtAllocatorSnapshot(Device device, long[] values) {
        this.device = device;
        List<StreamPool> entries = new ArrayList<>(values.length / POOL_FIELDS);
        for (int offset = 0; offset < values.length; offset += POOL_FIELDS) {
            entries.add(new StreamPool(values, offset));
        }
        pools = Collections.unmodifiableList(entries);
    }

    /**
     * Returns the device represented by this snapshot.
     *
     * @return the GPU device
     */
    public Device getDevice() {
        return device;
    }

    /**
     * Returns immutable per-stream pool summaries.
     *
     * <p>A stream can occur more than once because private pools and small/large pools are
     * distinct. Stream tokens have the same meaning as {@link PtStream#getStreamToken()}.
     *
     * @return the pool summaries
     */
    public List<StreamPool> getPools() {
        return pools;
    }

    /** Aggregated segments belonging to one allocation stream and one allocator pool. */
    public static final class StreamPool {

        private final long streamToken;
        private final long poolIdHigh;
        private final long poolIdLow;
        private final boolean large;
        private final long reservedBytes;
        private final long allocatedBytes;
        private final long activeBytes;
        private final long largestInactiveBlockBytes;
        private final long segmentCount;

        private StreamPool(long[] values, int offset) {
            streamToken = values[offset];
            poolIdHigh = values[offset + 1];
            poolIdLow = values[offset + 2];
            large = values[offset + 3] != 0;
            reservedBytes = values[offset + 4];
            allocatedBytes = values[offset + 5];
            activeBytes = values[offset + 6];
            largestInactiveBlockBytes = values[offset + 7];
            segmentCount = values[offset + 8];
        }

        /**
         * Returns the underlying CUDA/HIP allocation stream pointer as an opaque token.
         *
         * <p>This is not a JNI handle or PyTorch stream ID. Zero is the native default stream.
         *
         * @return the native stream token, comparable within the same process and device
         */
        public long getStreamToken() {
            return streamToken;
        }

        /**
         * Returns the first component of the allocator's private pool ID.
         *
         * <p>Both components are zero for the ordinary pool. They are opaque unsigned bit patterns.
         *
         * @return the first private pool ID component
         */
        public long getPoolIdHigh() {
            return poolIdHigh;
        }

        /**
         * Returns the second component of the allocator's private pool ID.
         *
         * @return the second private pool ID component
         */
        public long getPoolIdLow() {
            return poolIdLow;
        }

        /**
         * Returns whether these segments belong to the allocator's large allocation pool.
         *
         * @return true for the large pool, false for the small pool
         */
        public boolean isLarge() {
            return large;
        }

        /**
         * Returns the total bytes of the segments, including allocated and inactive blocks.
         *
         * @return reserved segment bytes
         */
        public long getReservedBytes() {
            return reservedBytes;
        }

        /**
         * Returns bytes in blocks currently allocated to tensor storage.
         *
         * @return allocated block bytes
         */
        public long getAllocatedBytes() {
            return allocatedBytes;
        }

        /**
         * Returns bytes in blocks that are allocated or still used by a stream.
         *
         * @return active block bytes
         */
        public long getActiveBytes() {
            return activeBytes;
        }

        /**
         * Returns the largest individual inactive block in these segments.
         *
         * <p>This is not a guarantee that an allocation of this size will succeed. Allocator
         * policy, private pool ownership, and segment layout still apply. Inactive blocks in a
         * partly active segment cannot necessarily be returned to the device independently.
         *
         * @return the largest inactive block size, or zero when none exists
         */
        public long getLargestInactiveBlockBytes() {
            return largestInactiveBlockBytes;
        }

        /**
         * Returns the number of segments represented by this summary.
         *
         * @return segment count
         */
        public long getSegmentCount() {
            return segmentCount;
        }
    }
}
