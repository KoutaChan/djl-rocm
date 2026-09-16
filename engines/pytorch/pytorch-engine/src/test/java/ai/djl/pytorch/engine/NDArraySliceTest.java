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
import ai.djl.engine.EngineException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;

import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class NDArraySliceTest {

    private static final Device TEST_DEVICE =
            Device.fromName(System.getProperty("ai.djl.pytorch.test.device", "cpu"));

    @DataProvider
    public Object[][] boundedSlices() {
        return new Object[][] {
            {0L, 0L, 1L},
            {1L, 3L, 1L},
            {0L, 4L, 2L},
            {-3L, -1L, 1L},
            {-8L, 8L, 2L},
            {3L, 1L, 1L},
            {8L, 10L, 1L},
            {Long.MIN_VALUE, Long.MAX_VALUE, 1L}
        };
    }

    @Test(dataProvider = "boundedSlices")
    public void testSingleSliceMatchesGeneralIndex(long start, long end, long step) {
        try (NDManager manager = NDManager.newBaseManager(TEST_DEVICE, "PyTorch")) {
            NDArray source = manager.arange(0, 24, 1, DataType.FLOAT32).reshape(4, 2, 3);
            NDIndex index = new NDIndex().addSliceDim(start, end, step);
            NDArray actual = source.get(index);
            NDArray expected = source.get(new NDIndex(start + ":" + end + ":" + step + ",..."));
            Assert.assertEquals(actual.getShape(), expected.getShape());
            Assert.assertEquals(actual.toFloatArray(), expected.toFloatArray());

            NDArray strided = source.transpose(1, 0, 2);
            actual = strided.get(index);
            expected = strided.get(new NDIndex(start + ":" + end + ":" + step + ",..."));
            Assert.assertEquals(actual.getShape(), expected.getShape());
            Assert.assertEquals(actual.toFloatArray(), expected.toFloatArray());
        }
    }

    @Test
    @SuppressWarnings("try") // Closing each owner early verifies the shared storage lifetime.
    public void testSliceSharesStorageAndBelongsToRequestedManager() {
        try (NDManager manager = NDManager.newBaseManager(TEST_DEVICE, "PyTorch");
                NDManager sourceOwner = manager.newSubManager();
                NDManager resultOwner = manager.newSubManager()) {
            NDArray source = sourceOwner.arange(0, 12, 1, DataType.FLOAT32).reshape(4, 3);
            NDArray slice = source.get(resultOwner, new NDIndex("1:3"));
            Assert.assertSame(slice.getManager(), resultOwner);
            slice.addi(10);
            Assert.assertEquals(
                    source.toFloatArray(),
                    new float[] {0, 1, 2, 13, 14, 15, 16, 17, 18, 9, 10, 11});
            sourceOwner.close();
            Assert.assertEquals(slice.toFloatArray(), new float[] {13, 14, 15, 16, 17, 18});
            resultOwner.close();
            Assert.assertTrue(slice.isReleased());
        }
    }

    @Test
    public void testStridedSlicePreservesGradient() {
        try (NDManager manager = NDManager.newBaseManager(TEST_DEVICE, "PyTorch");
                NDManager resultOwner = manager.newSubManager()) {
            NDArray source = manager.arange(0, 12, 1, DataType.FLOAT32).reshape(4, 3);
            source.setRequiresGradient(true);
            try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
                NDArray slice = source.get(resultOwner, new NDIndex("1:4:2"));
                NDArray weights =
                        resultOwner.create(new float[] {1, 2, 3, 4, 5, 6}, new Shape(2, 3));
                collector.backward(slice.mul(weights).sum());
            }
            Assert.assertEquals(
                    source.getGradient().toFloatArray(),
                    new float[] {0, 0, 0, 1, 2, 3, 0, 0, 0, 4, 5, 6});
        }
    }

    @DataProvider
    public Object[][] generalIndices() {
        return new Object[][] {
            {":2", new Shape(2, 3), new float[] {0, 1, 2, 3, 4, 5}},
            {"1:", new Shape(3, 3), new float[] {3, 4, 5, 6, 7, 8, 9, 10, 11}},
            {"::2", new Shape(2, 3), new float[] {0, 1, 2, 6, 7, 8}},
            {"...,1:3", new Shape(4, 2), new float[] {1, 2, 4, 5, 7, 8, 10, 11}},
            {"1:3,1:3", new Shape(2, 2), new float[] {4, 5, 7, 8}},
            {"1", new Shape(3), new float[] {3, 4, 5}},
            {"null,1:3", new Shape(1, 2, 3), new float[] {3, 4, 5, 6, 7, 8}}
        };
    }

    @Test(dataProvider = "generalIndices")
    public void testOtherIndicesKeepTheirSemantics(String index, Shape shape, float[] values) {
        try (NDManager manager = NDManager.newBaseManager(TEST_DEVICE, "PyTorch")) {
            NDArray source = manager.arange(0, 12, 1, DataType.FLOAT32).reshape(4, 3);
            NDArray actual = source.get(new NDIndex(index));
            Assert.assertEquals(actual.getShape(), shape);
            Assert.assertEquals(actual.toFloatArray(), values);
            Assert.assertThrows(EngineException.class, () -> source.get(new NDIndex("3:0:-1")));
            NDArray scalar = manager.create(1f);
            Assert.assertThrows(EngineException.class, () -> scalar.get(new NDIndex("0:1")));
        }
    }
}
