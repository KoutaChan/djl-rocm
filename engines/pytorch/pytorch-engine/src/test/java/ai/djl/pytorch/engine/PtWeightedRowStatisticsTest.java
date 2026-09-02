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
import org.testng.SkipException;
import org.testng.annotations.Test;

/** Verifies packed weighted row statistics and their portable fallback semantics. */
public class PtWeightedRowStatisticsTest {

    @Test
    public void cpuFallbackComputesPackedStatisticsForBothInputTypes() {
        for (DataType dataType : new DataType[] {DataType.FLOAT32, DataType.BFLOAT16}) {
            try (NDManager manager = Engine.getInstance().newBaseManager(Device.cpu())) {
                NDArray values =
                        manager.create(
                                        new float[] {
                                            1f,
                                            2f,
                                            100f,
                                            4f,
                                            Float.NaN,
                                            -1f,
                                            3f,
                                            0f,
                                            5f,
                                            Float.POSITIVE_INFINITY
                                        },
                                        new Shape(2, 5))
                                .toType(dataType, false);
                NDArray weights =
                        manager.create(new float[] {1f, 2f, 0f, 1f, 0f}).toType(dataType, false);

                NDArray actual = NDArrays.weightedRowStatistics(values, weights);

                Assert.assertEquals(actual.getDataType(), DataType.FLOAT32);
                Assert.assertEquals(actual.getShape(), new Shape(3, 2));
                Assert.assertEquals(
                        actual.toFloatArray(), new float[] {2.25f, 2.5f, 1f, -1f, 4f, 5f}, 0f);
            }
        }
    }

    @Test
    public void cpuFallbackDefinesZeroWeightAndNonFiniteSemantics() {
        try (NDManager manager = Engine.getInstance().newBaseManager(Device.cpu())) {
            NDArray values =
                    manager.create(
                            new float[] {Float.NaN, 3f, 7f, 2f, 4f, Float.NEGATIVE_INFINITY},
                            new Shape(2, 3));

            NDArray zero = NDArrays.weightedRowStatistics(values, manager.zeros(new Shape(3)));
            Assert.assertEquals(zero.toFloatArray(), new float[6], 0f);

            NDArray activeNan =
                    NDArrays.weightedRowStatistics(
                            values, manager.create(new float[] {1f, 0f, 0f}));
            float[] activeNanValues = activeNan.toFloatArray();
            Assert.assertTrue(Float.isNaN(activeNanValues[0]));
            Assert.assertEquals(activeNanValues[1], 2f);
            Assert.assertTrue(Float.isNaN(activeNanValues[2]));
            Assert.assertEquals(activeNanValues[3], 2f);
            Assert.assertTrue(Float.isNaN(activeNanValues[4]));
            Assert.assertEquals(activeNanValues[5], 2f);

            assertAllNaN(
                    NDArrays.weightedRowStatistics(
                                    values, manager.create(new float[] {0f, -1f, 0f}))
                            .toFloatArray());
            assertAllNaN(
                    NDArrays.weightedRowStatistics(
                                    values,
                                    manager.create(new float[] {0f, Float.POSITIVE_INFINITY, 0f}))
                            .toFloatArray());
        }
    }

    @Test
    public void gpuKernelMatchesCpuFallbackAtProductionShapeAndMixedTypes() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("GPU is unavailable");
        }
        final int rows = 11;
        final int columns = 800;
        float[] rawValues = new float[rows * columns];
        float[] rawWeights = new float[columns];
        for (int column = 0; column < columns; ++column) {
            rawWeights[column] = column % 9 == 0 ? 0f : 0.25f + (column % 7) * 0.125f;
            for (int row = 0; row < rows; ++row) {
                rawValues[row * columns + column] =
                        (float) Math.sin(row * 0.37 + column * 0.013) + row * 0.1f;
            }
        }

        for (DataType valueType : new DataType[] {DataType.FLOAT32, DataType.BFLOAT16}) {
            for (DataType weightType : new DataType[] {DataType.FLOAT32, DataType.BFLOAT16}) {
                try (NDManager cpu = engine.newBaseManager(Device.cpu());
                        NDManager gpu = engine.newBaseManager(Device.gpu())) {
                    NDArray cpuValues =
                            cpu.create(rawValues, new Shape(rows, columns))
                                    .toType(valueType, false);
                    NDArray cpuWeights = cpu.create(rawWeights).toType(weightType, false);
                    NDArray expected = NDArrays.weightedRowStatistics(cpuValues, cpuWeights);
                    NDArray gpuValues =
                            gpu.create(rawValues, new Shape(rows, columns))
                                    .toType(valueType, false);
                    NDArray gpuWeights = gpu.create(rawWeights).toType(weightType, false);

                    NDArray actual = NDArrays.weightedRowStatistics(gpuValues, gpuWeights);

                    assertClose(actual.toFloatArray(), expected.toFloatArray(), 2.0e-5f);
                }
            }
        }
    }

    @Test
    public void gpuNonContiguousInputUsesReferenceFallback() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("GPU is unavailable");
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            NDArray values = manager.arange(60f).reshape(10, 6).transpose();
            NDArray weights = manager.arange(10f).add(1f);
            NDArray actual = NDArrays.weightedRowStatistics(values, weights);
            NDArray contiguous = manager.create(values.toFloatArray(), new Shape(6, 10));
            NDArray expected = NDArrays.weightedRowStatistics(contiguous, weights);

            Assert.assertEquals(actual.getShape(), new Shape(3, 6));
            assertClose(actual.toFloatArray(), expected.toFloatArray(), 0f);
        }
    }

    @Test
    public void invalidShapesAndTypesFailFast() {
        try (NDManager manager = Engine.getInstance().newBaseManager(Device.cpu())) {
            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            NDArrays.weightedRowStatistics(
                                    manager.zeros(new Shape(2, 3, 4)),
                                    manager.zeros(new Shape(4))));
            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            NDArrays.weightedRowStatistics(
                                    manager.zeros(new Shape(2, 4)), manager.zeros(new Shape(3))));
            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            NDArrays.weightedRowStatistics(
                                    manager.zeros(new Shape(2, 4), DataType.FLOAT16),
                                    manager.zeros(new Shape(4))));
        }
    }

    private static void assertAllNaN(float[] values) {
        for (float value : values) {
            Assert.assertTrue(Float.isNaN(value));
        }
    }

    private static void assertClose(float[] actual, float[] expected, float tolerance) {
        Assert.assertEquals(actual.length, expected.length);
        for (int index = 0; index < actual.length; ++index) {
            Assert.assertEquals(actual[index], expected[index], tolerance, "index=" + index);
        }
    }
}
