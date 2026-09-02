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

            engine.resetPeakMemoryStats(device);
            assertPeakEqualsCurrent(engine.getMemoryStats(device));
        } finally {
            JniUtils.emptyCudaCache();
        }
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
