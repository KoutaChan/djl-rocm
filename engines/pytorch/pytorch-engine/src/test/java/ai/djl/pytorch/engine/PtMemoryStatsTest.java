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
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.testing.TestRequirements;

import org.testng.Assert;
import org.testng.annotations.Test;

public class PtMemoryStatsTest {

    private static final long ELEMENT_COUNT = 16L * 1024 * 1024;

    @Test
    public void memoryStatsRequireGpu() {
        PtEngine engine = (PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME);
        Assert.assertThrows(
                IllegalArgumentException.class, () -> engine.getMemoryStats(Device.cpu()));
        Assert.assertThrows(
                IllegalArgumentException.class, () -> engine.resetPeakMemoryStats(Device.cpu()));
        Assert.assertThrows(
                IllegalArgumentException.class, () -> engine.getAllocatorSnapshot(Device.cpu()));
    }

    @Test
    public void memoryStatsTrackAllocationAndResetPeak() {
        TestRequirements.gpu(PtEngine.ENGINE_NAME);
        PtEngine engine = (PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME);
        Device device = Device.gpu(0);

        engine.resetPeakMemoryStats(device);
        PtMemoryStats before = engine.getMemoryStats(device);
        assertPeakAtLeastCurrent(before);

        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray array = manager.zeros(new Shape(ELEMENT_COUNT), DataType.FLOAT32);
            Assert.assertEquals(array.size(), ELEMENT_COUNT);

            PtMemoryStats allocated = engine.getMemoryStats(device);
            Assert.assertTrue(
                    allocated.getAllocatedBytes()
                            >= before.getAllocatedBytes() + ELEMENT_COUNT * Float.BYTES);
            Assert.assertTrue(allocated.getActiveBytes() >= allocated.getAllocatedBytes());
            Assert.assertTrue(allocated.getReservedBytes() >= allocated.getActiveBytes());
            assertPeakAtLeastCurrent(allocated);
            Assert.assertTrue(allocated.getInactiveSplitBytes() >= 0);
            Assert.assertTrue(allocated.getAllocationRetries() >= before.getAllocationRetries());
            Assert.assertTrue(allocated.getOutOfMemoryCount() >= before.getOutOfMemoryCount());

            engine.resetPeakMemoryStats(device);
            PtMemoryStats reset = engine.getMemoryStats(device);
            assertPeakEqualsCurrent(reset);
            Assert.assertEquals(reset.getAllocationRetries(), allocated.getAllocationRetries());
            Assert.assertEquals(reset.getOutOfMemoryCount(), allocated.getOutOfMemoryCount());
        } finally {
            JniUtils.emptyCudaCache();
        }
    }

    @Test
    @SuppressWarnings("try") // The scopes select the stream for allocations.
    public void allocatorSnapshotMatchesNativeStreamAndReleasedBlocks() {
        TestRequirements.gpu(PtEngine.ENGINE_NAME);
        PtEngine engine = (PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME);
        Device device = Device.gpu(0);
        try (NDManager manager = engine.newBaseManager(device);
                PtStream first = engine.newStream(device);
                PtStream second = engine.newStream(device);
                PtEvent firstDone = first.newEvent();
                PtEvent secondDone = second.newEvent()) {
            long firstId = first.getId();
            long secondId = second.getId();
            Assert.assertNotEquals(firstId, secondId);
            NDArray left;
            NDArray right;
            try (PtStreamScope ignored = first.openScope()) {
                left = manager.zeros(new Shape(1024 * 1024), DataType.FLOAT32);
                firstDone.record();
            }
            try (PtStreamScope ignored = second.openScope()) {
                right = manager.zeros(new Shape(2 * 1024 * 1024), DataType.FLOAT32);
                secondDone.record();
            }
            firstDone.synchronize();
            secondDone.synchronize();
            PtAllocatorSnapshot allocated = engine.getAllocatorSnapshot(device);
            Assert.assertTrue(getAllocatedBytes(allocated, firstId) >= 4L * 1024 * 1024);
            Assert.assertTrue(getAllocatedBytes(allocated, secondId) >= 8L * 1024 * 1024);
            for (PtAllocatorSnapshot.StreamPool pool : allocated.getPools()) {
                Assert.assertTrue(pool.getReservedBytes() >= pool.getActiveBytes());
                Assert.assertTrue(pool.getActiveBytes() >= pool.getAllocatedBytes());
                Assert.assertTrue(pool.getSegmentCount() > 0);
            }
            left.close();
            right.close();
            PtAllocatorSnapshot released = engine.getAllocatorSnapshot(device);
            Assert.assertTrue(getLargestInactiveBlock(released, firstId) >= 4L * 1024 * 1024);
            Assert.assertTrue(getLargestInactiveBlock(released, secondId) >= 8L * 1024 * 1024);
            // Previously returned snapshots retain their values after allocator changes.
            Assert.assertTrue(getAllocatedBytes(allocated, firstId) >= 4L * 1024 * 1024);
        }
    }

    private static long getAllocatedBytes(PtAllocatorSnapshot snapshot, long streamId) {
        return snapshot.getPools().stream()
                .filter(pool -> pool.getStreamId() == streamId)
                .mapToLong(PtAllocatorSnapshot.StreamPool::getAllocatedBytes)
                .sum();
    }

    private static long getLargestInactiveBlock(PtAllocatorSnapshot snapshot, long streamId) {
        return snapshot.getPools().stream()
                .filter(pool -> pool.getStreamId() == streamId)
                .mapToLong(PtAllocatorSnapshot.StreamPool::getLargestInactiveBlockBytes)
                .max()
                .orElse(0);
    }

    private static void assertPeakAtLeastCurrent(PtMemoryStats stats) {
        Assert.assertTrue(stats.getPeakAllocatedBytes() >= stats.getAllocatedBytes());
        Assert.assertTrue(stats.getPeakReservedBytes() >= stats.getReservedBytes());
        Assert.assertTrue(stats.getPeakActiveBytes() >= stats.getActiveBytes());
    }

    private static void assertPeakEqualsCurrent(PtMemoryStats stats) {
        Assert.assertEquals(stats.getPeakAllocatedBytes(), stats.getAllocatedBytes());
        Assert.assertEquals(stats.getPeakReservedBytes(), stats.getReservedBytes());
        Assert.assertEquals(stats.getPeakActiveBytes(), stats.getActiveBytes());
    }
}
