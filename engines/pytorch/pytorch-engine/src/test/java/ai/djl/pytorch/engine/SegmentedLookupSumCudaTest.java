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
import ai.djl.engine.InferenceMode;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.GradientCollector;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** Verifies fused CUDA lookup preparation while retaining the ATen reduction and gradients. */
@SuppressWarnings("try")
public class SegmentedLookupSumCudaTest {

    @DataProvider(name = "types")
    public Object[][] types() {
        Object[][] result = new Object[9][2];
        int row = 0;
        for (DataType type : floatingTypes()) {
            for (DataType index : new DataType[] {DataType.INT16, DataType.INT32, DataType.INT64}) {
                result[row++] = new Object[] {type, index};
            }
        }
        return result;
    }

    @Test(dataProvider = "types")
    public void cudaInferencePreservesReferenceBitsAcrossShapesAndStrides(
            DataType dataType, DataType indexType) {
        Engine engine = requireCuda();
        Shape[] indexShapes = {
            new Shape(7, 1, 1, 13, 2, 3),
            new Shape(11, 5),
            new Shape(3, 2, 5),
            new Shape(2, 3, 17),
            new Shape(17)
        };
        Shape[] tableShapes = {
            new Shape(3 * 16384, 16),
            new Shape(5 * 9, 1),
            new Shape(5 * 9, 1, 7),
            new Shape(17 * 9, 127),
            new Shape(17 * 9, 7)
        };
        for (int shape = 0; shape < indexShapes.length; ++shape) {
            try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                NDArray table = table(manager, tableShapes[shape], dataType);
                long entries = tableShapes[shape].get(0) / indexShapes[shape].getLastDimension();
                for (boolean strided : new boolean[] {false, true}) {
                    NDArray indices =
                            indices(manager, indexShapes[shape], indexType, entries, strided);
                    assertInferenceMatchesReference(engine, table, indices);
                }
            }
        }
    }

    @Test
    public void cudaInferenceHandlesBroadcastIndicesAndEmptyOutputs() {
        Engine engine = requireCuda();
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            NDArray table = table(manager, new Shape(15, 7), DataType.BFLOAT16);
            NDArray indices =
                    manager.create(new long[] {Long.MIN_VALUE, Long.MAX_VALUE, 2}).reshape(1, 3);
            NDArray broadcast = indices.broadcast(new Shape(9, 3));
            assertInferenceMatchesReference(engine, table, broadcast);
            NDArray empty = manager.create(new int[0], new Shape(0, 3));
            assertInferenceMatchesReference(engine, table, empty);
            NDArray emptyWidth = manager.create(new Shape(15, 0), DataType.BFLOAT16);
            assertInferenceMatchesReference(engine, emptyWidth, indices);
        }
    }

    @Test
    public void cudaInferenceFallbackPreservesStridedTablesAndHighRankIndices() {
        Engine engine = requireCuda();
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            for (DataType type : floatingTypes()) {
                NDArray stridedTable = table(manager, new Shape(7, 15), type).transpose();
                NDArray indices = manager.create(new int[] {-1, 8, 3, 2, 1, 0}).reshape(2, 3);
                assertInferenceMatchesReference(engine, stridedTable, indices);
                NDArray contiguousTable = table(manager, new Shape(15, 7), type);
                NDArray highRank = indices.reshape(2, 1, 1, 1, 1, 1, 1, 1, 3);
                assertInferenceMatchesReference(engine, contiguousTable, highRank);
            }
            NDArray doubleTable = table(manager, new Shape(15, 7), DataType.FLOAT64);
            NDArray indices = manager.create(new long[] {-1, Long.MAX_VALUE, 3}).reshape(1, 3);
            assertInferenceMatchesReference(engine, doubleTable, indices);
        }
    }

    @Test
    public void cudaGradientFallbackAccumulatesClampedSelections() {
        Engine engine = requireCuda();
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            for (DataType type : floatingTypes()) {
                NDArray table = table(manager, new Shape(12, 7), type);
                table.setRequiresGradient(true);
                NDArray indices =
                        manager.create(new long[] {-5, 99, 2, 4, 0, 99, -1, 99, 2}).reshape(3, 3);
                try (GradientCollector collector = engine.newGradientCollector()) {
                    collector.backward(NDArrays.segmentedLookupSum(table, indices).sum());
                }
                float[] expected = new float[12 * 7];
                for (int row : new int[] {0, 7, 9, 3, 4, 11, 0, 7, 9}) {
                    for (int feature = 0; feature < 7; ++feature) {
                        expected[row * 7 + feature] += 1;
                    }
                }
                try (NDArray gradient = table.getGradient().toType(DataType.FLOAT32, false)) {
                    Assert.assertEquals(gradient.toFloatArray(), expected);
                }
            }
        }
    }

    private static void assertInferenceMatchesReference(
            Engine engine, NDArray table, NDArray indices) {
        NDArray actual;
        try (InferenceMode ignored = engine.newInferenceMode()) {
            actual = NDArrays.segmentedLookupSum(table, indices);
        }
        NDArray expected;
        // Grad mode forces the unmodified differentiable ATen reference, even without gradients.
        try (GradientCollector ignored = engine.newGradientCollector()) {
            expected = NDArrays.segmentedLookupSum(table, indices);
        }
        try (NDArray ownedActual = actual;
                NDArray ownedExpected = expected) {
            Assert.assertEquals(actual.getShape(), expected.getShape());
            Assert.assertEquals(actual.getDataType(), expected.getDataType());
            Assert.assertEquals(actual.toByteBuffer(), expected.toByteBuffer());
        }
    }

    private static NDArray table(NDManager manager, Shape shape, DataType type) {
        float[] values = new float[Math.toIntExact(shape.size())];
        for (int i = 0; i < values.length; ++i) {
            values[i] = ((i * 17L % 257) - 128) * 0.0078125f + (i % 13 == 0 ? 32f : 0f);
        }
        try (NDArray source = manager.create(values, shape)) {
            return source.toType(type, true);
        }
    }

    private static NDArray indices(
            NDManager manager, Shape shape, DataType type, long entries, boolean strided) {
        int count = Math.toIntExact(shape.size());
        long[] values = new long[count * (strided ? 2 : 1)];
        for (int i = 0; i < count; ++i) {
            long value;
            switch (i % 7) {
                case 0:
                    value = -7;
                    break;
                case 1:
                    value = 0;
                    break;
                case 2:
                    value = entries + 3;
                    break;
                default:
                    value = (i * 19L % entries) + 1;
                    break;
            }
            values[i * (strided ? 2 : 1)] = value;
            if (strided) {
                values[i * 2 + 1] = -99;
            }
        }
        Shape storedShape = strided ? shape.add(2) : shape;
        NDArray source = manager.create(values, storedShape).toType(type, true);
        return strided ? source.get("...,0") : source;
    }

    private static DataType[] floatingTypes() {
        return new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16};
    }

    private static Engine requireCuda() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0 || JniUtils.getFusionBackend() != 1) {
            throw new SkipException("This segmented lookup test requires PyTorch CUDA.");
        }
        return engine;
    }
}
