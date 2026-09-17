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
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.GradientCollector;

import org.testng.Assert;
import org.testng.annotations.Test;

/** Verifies padded batch-gather semantics, strided indices, and source gradients. */
public class PaddedBatchGatherTest {

    @Test
    public void oneDimensionalAndExplicitBatchGatherPreservePadding() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            NDArray source = sequentialSource(manager);
            NDArray storedIndices = int16(manager, new int[] {0, 3, 1, 2, 0, 3}, new Shape(2, 3));
            NDArray batchIndices = manager.create(new int[] {1, 0, 1});
            NDArray compactStoredIndices = int16(manager, new int[] {3, 0, 1}, new Shape(3));

            NDArray gathered = NDArrays.paddedBatchGather(source, storedIndices);
            NDArray compact =
                    NDArrays.paddedBatchGatherByBatchIndices(
                            source, batchIndices, compactStoredIndices);

            Assert.assertEquals(gathered.getShape(), new Shape(2, 3, 2));
            assertClose(
                    gathered.toFloatArray(),
                    new float[] {0f, 0f, 5f, 6f, 1f, 2f, 9f, 10f, 0f, 0f, 11f, 12f});
            Assert.assertEquals(compact.getShape(), new Shape(3, 2));
            assertClose(compact.toFloatArray(), new float[] {11f, 12f, 0f, 0f, 7f, 8f});
        }
    }

    @Test
    public void twoDimensionalGatherUsesBothStoredIndices() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            NDArray source =
                    manager.arange(1, 13).toType(DataType.FLOAT32, false).reshape(2, 2, 3, 1);
            NDArray outerStoredIndices = int16(manager, new int[] {1, 2, 0, 1}, new Shape(2, 2));
            NDArray innerStoredIndices = int16(manager, new int[] {3, 1, 2, 2}, new Shape(2, 2));

            NDArray gathered =
                    NDArrays.paddedBatchGather(source, outerStoredIndices, innerStoredIndices);

            Assert.assertEquals(gathered.getShape(), new Shape(2, 2, 1));
            assertClose(gathered.toFloatArray(), new float[] {3f, 4f, 0f, 8f});
        }
    }

    @Test
    public void nativeGpuGatherAcceptsStridedInt16AndMixedBatchIndices() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            NDArray source = sequentialSource(manager).toType(DataType.FLOAT16, false);
            NDArray indexFields =
                    manager.create(
                                    new int[] {0, 91, 3, 92, 1, 93, 2, 94, 0, 95, 3, 96},
                                    new Shape(2, 3, 2))
                            .toType(DataType.INT16, false);
            NDArray storedIndices = indexFields.get("...,0");

            NDArray gathered = NDArrays.paddedBatchGather(source, storedIndices);

            NDArray pairSource =
                    manager.arange(1, 13).toType(DataType.FLOAT16, false).reshape(2, 2, 3, 1);
            NDArray outerStoredIndices = int16(manager, new int[] {1, 2, 0, 1}, new Shape(2, 2));
            NDArray innerStoredIndices = int16(manager, new int[] {3, 1, 2, 2}, new Shape(2, 2));
            NDArray pairGathered =
                    NDArrays.paddedBatchGather(pairSource, outerStoredIndices, innerStoredIndices);

            NDArray compactBatchIndices = manager.create(new int[] {1, 0, 1});
            NDArray compactBatchIndices64 = manager.create(new long[] {1, 0, 1});
            NDArray compactStoredIndices = int16(manager, new int[] {3, 0, 1}, new Shape(3));
            NDArray compactGathered =
                    NDArrays.paddedBatchGatherByBatchIndices(
                            source, compactBatchIndices, compactStoredIndices);
            NDArray compactGatheredWithInt64Batch =
                    NDArrays.paddedBatchGatherByBatchIndices(
                            source, compactBatchIndices64, compactStoredIndices);
            NDArray int64Gathered =
                    NDArrays.paddedBatchGather(
                            source, manager.create(new long[] {0, 3, 1, 2, 0, 3}, new Shape(2, 3)));

            assertClose(
                    gathered.toFloatArray(),
                    new float[] {0f, 0f, 5f, 6f, 1f, 2f, 9f, 10f, 0f, 0f, 11f, 12f});
            assertClose(pairGathered.toFloatArray(), new float[] {3f, 4f, 0f, 8f});
            assertClose(compactGathered.toFloatArray(), new float[] {11f, 12f, 0f, 0f, 7f, 8f});
            assertClose(
                    compactGatheredWithInt64Batch.toFloatArray(),
                    new float[] {11f, 12f, 0f, 0f, 7f, 8f});
            assertClose(
                    int64Gathered.toFloatArray(),
                    new float[] {0f, 0f, 5f, 6f, 1f, 2f, 9f, 10f, 0f, 0f, 11f, 12f});
        }
    }

    @Test
    public void cudaGatherVariantsPreserveClampedNonFiniteValuesAndSignedZero() {
        Engine engine = Engine.getInstance();
        // Fusion reports the native build backend: CUDA = 1, ROCm = 2.
        if (engine.getGpuCount() == 0 || JniUtils.getFusionBackend() != 1) {
            return;
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            for (DataType indexType :
                    new DataType[] {DataType.INT16, DataType.INT32, DataType.INT64}) {
                for (String indexLayout : new String[] {"contiguous", "strided", "broadcast"}) {
                    for (boolean sourceView : new boolean[] {false, true}) {
                        float[][] expected =
                                gatherBoundaryResults(
                                        engine,
                                        Device.cpu(),
                                        dataType,
                                        indexType,
                                        indexLayout,
                                        sourceView);
                        float[][] actual =
                                gatherBoundaryResults(
                                        engine,
                                        Device.gpu(),
                                        dataType,
                                        indexType,
                                        indexLayout,
                                        sourceView);
                        for (int variant = 0; variant < expected.length; ++variant) {
                            Assert.assertEquals(actual[variant].length, expected[variant].length);
                            for (int index = 0; index < expected[variant].length; ++index) {
                                float value = actual[variant][index];
                                float reference = expected[variant][index];
                                String context =
                                        dataType
                                                + "/"
                                                + indexType
                                                + "/"
                                                + indexLayout
                                                + "/sourceView="
                                                + sourceView
                                                + "/variant="
                                                + variant
                                                + "/index="
                                                + index;
                                if (Float.isNaN(reference)) {
                                    Assert.assertTrue(Float.isNaN(value), context);
                                } else {
                                    Assert.assertEquals(
                                            Float.floatToRawIntBits(value),
                                            Float.floatToRawIntBits(reference),
                                            context);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    public void sourceGradientsAccumulateRepeatedRowsAcrossVariantsAndDtypes() {
        Engine engine = Engine.getInstance();
        verifySourceGradients(engine, Device.cpu(), DataType.FLOAT32, false);
        if (engine.getGpuCount() > 0) {
            for (DataType dataType :
                    new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
                verifySourceGradients(engine, Device.gpu(), dataType, true);
            }
        }
    }

    @Test
    public void emptyGatherProducesZeroSourceGradient() {
        Engine engine = Engine.getInstance();
        verifyEmptyGradient(engine, Device.cpu(), DataType.FLOAT32);
        if (engine.getGpuCount() > 0) {
            for (DataType dataType :
                    new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
                verifyEmptyGradient(engine, Device.gpu(), dataType);
            }
        }
    }

    private static float[][] gatherBoundaryResults(
            Engine engine,
            Device device,
            DataType dataType,
            DataType indexType,
            String indexLayout,
            boolean sourceView) {
        float[] values = {
            Float.NaN,
            Float.POSITIVE_INFINITY,
            -0f,
            -2f,
            Float.NEGATIVE_INFINITY,
            0f,
            10f,
            -3f,
            Float.NaN,
            -0f,
            4f,
            Float.POSITIVE_INFINITY
        };
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray source =
                    manager.create(values, new Shape(2, 2, 3))
                            .toType(dataType, false)
                            .transpose(0, 2, 1);
            NDArray pairSource =
                    manager.create(values, new Shape(2, 2, 3))
                            .toType(dataType, false)
                            .expandDims(-1)
                            .repeat(3, 2)
                            .transpose(0, 2, 1, 3);
            if (!sourceView) {
                source = source.reshape(-1).reshape(2, 3, 2);
                pairSource = pairSource.reshape(-1).reshape(2, 3, 2, 2);
            }
            NDArray outer =
                    boundaryIndices(
                            manager,
                            new int[] {0, 1, 3, 4, -1, 2, Short.MIN_VALUE, Short.MAX_VALUE},
                            indexType,
                            indexLayout);
            NDArray inner =
                    boundaryIndices(
                            manager, new int[] {1, 2, 1, 2, 1, 0, 2, 1}, indexType, indexLayout);
            NDArray batches =
                    boundaryIndices(
                            manager,
                            new int[] {0, 1, 1, -1, 2, 0, 1, 0},
                            DataType.INT64,
                            indexLayout);
            NDArray[] results = {
                NDArrays.paddedBatchGather(source, outer),
                NDArrays.paddedBatchGather(pairSource, outer, inner),
                NDArrays.paddedBatchGatherByBatchIndices(source, batches, outer)
            };
            float[][] data = new float[results.length][];
            for (int index = 0; index < results.length; ++index) {
                Assert.assertEquals(results[index].getShape(), new Shape(2, 8, 2));
                Assert.assertEquals(results[index].getDataType(), dataType);
                data[index] = results[index].toType(DataType.FLOAT32, false).toFloatArray();
            }
            return data;
        }
    }

    private static NDArray boundaryIndices(
            NDManager manager, int[] values, DataType dataType, String layout) {
        NDArray indices =
                manager.create(values, new Shape(1, values.length)).toType(dataType, false);
        if ("broadcast".equals(layout)) {
            return indices.broadcast(2, values.length);
        }
        indices = indices.repeat(0, 2);
        return "strided".equals(layout)
                ? indices.expandDims(-1).repeat(2, 2).get("...,0")
                : indices;
    }

    private static void verifySourceGradients(
            Engine engine, Device device, DataType dataType, boolean stridedIndices) {
        verifyImplicitBatchGradient(engine, device, dataType, stridedIndices);
        verifyTwoDimensionalGradient(engine, device, dataType, stridedIndices);
        verifyExplicitBatchGradient(engine, device, dataType, stridedIndices);
    }

    private static void verifyImplicitBatchGradient(
            Engine engine, Device device, DataType dataType, boolean stridedIndices) {
        try (NDManager manager = engine.newBaseManager(device);
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray source = manager.ones(new Shape(2, 3, 2), dataType);
            source.setRequiresGradient(true);
            int[] storedValues = {1, 1, 4, 0, 3, -1, 3, 2};
            Shape indexShape = new Shape(2, 4);
            NDArray storedIndices =
                    stridedIndices
                            ? stridedIndex(manager, storedValues, indexShape, DataType.INT16)
                            : int16(manager, storedValues, indexShape);

            NDArray upstream =
                    manager.create(
                                    new float[] {
                                        1f, 10f, 2f, 20f, 3f, 30f, 4f, 40f,
                                        5f, 50f, 6f, 60f, 7f, 70f, 8f, 80f
                                    },
                                    new Shape(2, 4, 2))
                            .toType(dataType, false);
            collector.backward(
                    NDArrays.paddedBatchGather(source, storedIndices).mul(upstream).sum());

            assertClose(
                    source.getGradient().toType(DataType.FLOAT32, false).toFloatArray(),
                    new float[] {3f, 30f, 0f, 0f, 0f, 0f, 0f, 0f, 8f, 80f, 12f, 120f});
        }
    }

    private static void verifyEmptyGradient(Engine engine, Device device, DataType dataType) {
        try (NDManager manager = engine.newBaseManager(device);
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray source = manager.ones(new Shape(2, 3, 2), dataType);
            source.setRequiresGradient(true);
            NDArray storedIndices =
                    manager.create(new int[0], new Shape(2, 0)).toType(DataType.INT16, false);
            NDArray gathered = NDArrays.paddedBatchGather(source, storedIndices);

            Assert.assertEquals(gathered.getShape(), new Shape(2, 0, 2));
            collector.backward(gathered.sum());

            assertClose(
                    source.getGradient().toType(DataType.FLOAT32, false).toFloatArray(),
                    new float[] {0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f});

            NDArray detached = source.stopGradient();
            NDArray[] forwardResults = {
                NDArrays.paddedBatchGather(detached, storedIndices),
                NDArrays.paddedBatchGather(
                        detached.reshape(2, 3, 1, 2), storedIndices, storedIndices),
                NDArrays.paddedBatchGatherByBatchIndices(detached, storedIndices, storedIndices)
            };
            for (NDArray result : forwardResults) {
                Assert.assertEquals(result.getShape(), new Shape(2, 0, 2));
                Assert.assertEquals(result.getDataType(), dataType);
                Assert.assertEquals(result.size(), 0L);
            }
        }
    }

    private static void verifyTwoDimensionalGradient(
            Engine engine, Device device, DataType dataType, boolean stridedIndices) {
        try (NDManager manager = engine.newBaseManager(device);
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray source = manager.ones(new Shape(2, 2, 3, 2), dataType);
            source.setRequiresGradient(true);
            int[] outerValues = {1, 1, 0, 2, 2, 2, 1, 3};
            int[] innerValues = {2, 2, 1, 4, 3, 3, 1, 1};
            Shape indexShape = new Shape(2, 4);
            NDArray outerIndices =
                    stridedIndices
                            ? stridedIndex(manager, outerValues, indexShape, DataType.INT16)
                            : int16(manager, outerValues, indexShape);
            NDArray innerIndices =
                    stridedIndices
                            ? stridedIndex(manager, innerValues, indexShape, DataType.INT16)
                            : int16(manager, innerValues, indexShape);

            collector.backward(
                    NDArrays.paddedBatchGather(source, outerIndices, innerIndices).sum());

            assertClose(
                    source.getGradient().toType(DataType.FLOAT32, false).toFloatArray(),
                    new float[] {
                        0f, 0f, 2f, 2f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                        1f, 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 2f, 2f
                    });
        }
    }

    private static void verifyExplicitBatchGradient(
            Engine engine, Device device, DataType dataType, boolean stridedIndices) {
        try (NDManager manager = engine.newBaseManager(device);
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray source = manager.ones(new Shape(2, 3, 2), dataType);
            source.setRequiresGradient(true);
            int[] batchValues = {1, 1, 0, 0, -1, 2, 0, 1};
            int[] storedValues = {2, 2, 1, 1, 1, 1, 0, 4};
            Shape indexShape = new Shape(8);
            NDArray batchIndices =
                    stridedIndices
                            ? stridedIndex(manager, batchValues, indexShape, DataType.INT64)
                            : manager.create(batchValues).toType(DataType.INT64, false);
            NDArray storedIndices =
                    stridedIndices
                            ? stridedIndex(manager, storedValues, indexShape, DataType.INT16)
                            : int16(manager, storedValues, indexShape);

            collector.backward(
                    NDArrays.paddedBatchGatherByBatchIndices(source, batchIndices, storedIndices)
                            .sum());

            assertClose(
                    source.getGradient().toType(DataType.FLOAT32, false).toFloatArray(),
                    new float[] {2f, 2f, 0f, 0f, 0f, 0f, 0f, 0f, 2f, 2f, 0f, 0f});
        }
    }

    private static NDArray sequentialSource(NDManager manager) {
        return manager.arange(1, 13).toType(DataType.FLOAT32, false).reshape(2, 3, 2);
    }

    private static NDArray int16(NDManager manager, int[] values, Shape shape) {
        return manager.create(values, shape).toType(DataType.INT16, false);
    }

    private static NDArray stridedIndex(
            NDManager manager, int[] values, Shape shape, DataType dataType) {
        int[] fields = new int[values.length * 2];
        for (int index = 0; index < values.length; index++) {
            fields[index * 2] = values[index];
            fields[index * 2 + 1] = 1000 + index;
        }
        long[] fieldShape = new long[shape.dimension() + 1];
        System.arraycopy(shape.getShape(), 0, fieldShape, 0, shape.dimension());
        fieldShape[fieldShape.length - 1] = 2;
        return manager.create(fields, new Shape(fieldShape)).toType(dataType, false).get("...,0");
    }

    private static void assertClose(float[] actual, float[] expected) {
        Assert.assertEquals(actual.length, expected.length);
        for (int index = 0; index < actual.length; index++) {
            Assert.assertEquals(actual[index], expected[index], 1e-3f, "index=" + index);
        }
    }
}
