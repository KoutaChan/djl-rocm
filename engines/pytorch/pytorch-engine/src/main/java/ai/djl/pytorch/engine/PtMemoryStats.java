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

/** An immutable snapshot of PyTorch caching allocator memory statistics for a device. */
public final class PtMemoryStats {

    private final long allocatedBytes;
    private final long peakAllocatedBytes;
    private final long reservedBytes;
    private final long peakReservedBytes;
    private final long activeBytes;
    private final long peakActiveBytes;

    PtMemoryStats(long[] values) {
        allocatedBytes = values[0];
        peakAllocatedBytes = values[1];
        reservedBytes = values[2];
        peakReservedBytes = values[3];
        activeBytes = values[4];
        peakActiveBytes = values[5];
    }

    /**
     * Returns the bytes currently occupied by allocated tensors.
     *
     * @return the currently allocated bytes
     */
    public long getAllocatedBytes() {
        return allocatedBytes;
    }

    /**
     * Returns the peak bytes occupied by allocated tensors.
     *
     * @return the peak allocated bytes
     */
    public long getPeakAllocatedBytes() {
        return peakAllocatedBytes;
    }

    /**
     * Returns the bytes currently reserved by the caching allocator.
     *
     * @return the currently reserved bytes
     */
    public long getReservedBytes() {
        return reservedBytes;
    }

    /**
     * Returns the peak bytes reserved by the caching allocator.
     *
     * @return the peak reserved bytes
     */
    public long getPeakReservedBytes() {
        return peakReservedBytes;
    }

    /**
     * Returns the bytes in blocks that are allocated or still used by a stream.
     *
     * @return the currently active bytes
     */
    public long getActiveBytes() {
        return activeBytes;
    }

    /**
     * Returns the peak bytes in blocks that are allocated or still used by a stream.
     *
     * @return the peak active bytes
     */
    public long getPeakActiveBytes() {
        return peakActiveBytes;
    }
}
