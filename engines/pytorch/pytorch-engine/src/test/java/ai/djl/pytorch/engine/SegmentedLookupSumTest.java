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
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import org.testng.Assert;
import org.testng.annotations.Test;

/** Verifies segmented lookup-and-sum semantics and strided stored indices. */
public class SegmentedLookupSumTest {

    @Test
    public void cpuFallbackSelectsAndSumsEverySegment() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            verifySegmentedLookupSum(manager, DataType.FLOAT32, DataType.INT16, false);
        }
    }

    @Test
    public void nativeGpuPathMatchesReferenceAcrossSupportedDtypes() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            for (DataType dataType :
                    new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
                for (DataType indexType :
                        new DataType[] {DataType.INT16, DataType.INT32, DataType.INT64}) {
                    verifySegmentedLookupSum(manager, dataType, indexType, true);
                }
            }
        }
    }

    private static void verifySegmentedLookupSum(
            NDManager manager, DataType dataType, DataType indexType, boolean reportParity) {
        NDArray lookupTable =
                manager.create(
                                new float[] {
                                    0, 0, 1, 10, 2, 20, 3, 30,
                                    0, 0, 4, 40, 5, 50, 6, 60,
                                    0, 0, 7, 70, 8, 80, 9, 90
                                },
                                new Shape(12, 2))
                        .toType(dataType, false);
        NDArray storedFields =
                manager.create(
                                new int[] {
                                    0, 91, 2, 92, 4, 93,
                                    3, 94, 1, 95, 2, 96
                                },
                                new Shape(2, 3, 2))
                        .toType(indexType, false);
        NDArray stridedStoredIndices = storedFields.get("...,0");

        NDArray actual = NDArrays.segmentedLookupSum(lookupTable, stridedStoredIndices);

        Assert.assertEquals(actual.getShape(), new Shape(2, 2));
        float maxAbs =
                assertClose(
                        actual.toType(DataType.FLOAT32, false).toFloatArray(),
                        new float[] {13, 130, 9, 90});
        if (reportParity) {
            System.out.printf(
                    "SEGMENTED_LOOKUP_SUM_PARITY dtype=%s indexDtype=%s maxAbs=%s%n",
                    dataType, indexType, maxAbs);
        }
        Assert.expectThrows(
                IllegalArgumentException.class,
                () ->
                        NDArrays.segmentedLookupSum(
                                lookupTable, stridedStoredIndices.toType(DataType.FLOAT32, false)));
    }

    private static float assertClose(float[] actual, float[] expected) {
        Assert.assertEquals(actual.length, expected.length);
        float maxAbs = 0;
        for (int index = 0; index < actual.length; ++index) {
            maxAbs = Math.max(maxAbs, Math.abs(actual[index] - expected[index]));
            Assert.assertEquals(actual[index], expected[index], 1e-3f, "index=" + index);
        }
        return maxAbs;
    }
}
