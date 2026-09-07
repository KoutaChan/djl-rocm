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

import org.testng.Assert;
import org.testng.annotations.Test;

public class PtAllocatorSnapshotMetadataTest {

    @Test
    public void preservesLargeCountersAndDistinctPoolIdentities() {
        long bytes = 24L << 30;
        long stream = 0xFEDCBA9876543210L;
        long[] packed = {
            0, 0, 0, 0, 1024, 256, 512, 128, 1, stream, 7, 0, 1, bytes, bytes / 4, bytes / 2,
            bytes / 8, 12, stream, 0, 7, 1, 8192, 0, 0, 8192, 1
        };
        PtAllocatorSnapshot snapshot = new PtAllocatorSnapshot(Device.gpu(1), packed);
        Assert.assertEquals(snapshot.getDevice(), Device.gpu(1));
        Assert.assertEquals(snapshot.getPools().size(), 3);
        Assert.assertEquals(snapshot.getPools().get(0).getStreamId(), 0L);
        Assert.assertFalse(snapshot.getPools().get(0).isLarge());
        PtAllocatorSnapshot.StreamPool pool = snapshot.getPools().get(1);
        Assert.assertEquals(pool.getStreamId(), stream);
        Assert.assertEquals(pool.getPoolIdHigh(), 7L);
        Assert.assertEquals(pool.getPoolIdLow(), 0L);
        Assert.assertTrue(pool.isLarge());
        Assert.assertEquals(pool.getReservedBytes(), bytes);
        Assert.assertEquals(pool.getAllocatedBytes(), bytes / 4);
        Assert.assertEquals(pool.getActiveBytes(), bytes / 2);
        Assert.assertEquals(pool.getLargestInactiveBlockBytes(), bytes / 8);
        Assert.assertEquals(pool.getSegmentCount(), 12L);
        Assert.assertEquals(snapshot.getPools().get(2).getPoolIdHigh(), 0L);
        Assert.assertEquals(snapshot.getPools().get(2).getPoolIdLow(), 7L);
        packed[13] = 0;
        Assert.assertEquals(pool.getReservedBytes(), bytes);
        Assert.assertThrows(UnsupportedOperationException.class, () -> snapshot.getPools().clear());
    }

    @Test
    public void emptySnapshotHasNoSyntheticDefaultStream() {
        PtAllocatorSnapshot snapshot = new PtAllocatorSnapshot(Device.gpu(0), new long[0]);
        Assert.assertTrue(snapshot.getPools().isEmpty());
    }

    @Test
    public void memoryStatisticsPreserveFragmentationAndCumulativeCounters() {
        long counter = 1L << 33;
        PtMemoryStats stats =
                new PtMemoryStats(new long[] {10, 20, 40, 50, 30, 35, 8, counter, counter + 1});
        Assert.assertEquals(stats.getAllocatedBytes(), 10L);
        Assert.assertEquals(stats.getPeakAllocatedBytes(), 20L);
        Assert.assertEquals(stats.getReservedBytes(), 40L);
        Assert.assertEquals(stats.getPeakReservedBytes(), 50L);
        Assert.assertEquals(stats.getActiveBytes(), 30L);
        Assert.assertEquals(stats.getPeakActiveBytes(), 35L);
        Assert.assertEquals(stats.getInactiveSplitBytes(), 8L);
        Assert.assertEquals(stats.getAllocationRetries(), counter);
        Assert.assertEquals(stats.getOutOfMemoryCount(), counter + 1);
    }
}
