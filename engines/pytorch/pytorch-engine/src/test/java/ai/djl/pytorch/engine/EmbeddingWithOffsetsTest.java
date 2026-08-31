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
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.core.Embedding;
import ai.djl.training.GradientCollector;

import org.testng.Assert;
import org.testng.annotations.Test;

public class EmbeddingWithOffsetsTest {

    @Test
    public void testNativeGpuPathAcrossSupportedDtypes() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            DataType[] indexTypes = {DataType.INT16, DataType.INT32, DataType.INT64};
            DataType[] tableTypes = {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16};
            for (DataType rawType : indexTypes) {
                for (DataType offsetType : indexTypes) {
                    if (rawType == DataType.INT16 && offsetType == DataType.INT16) {
                        continue;
                    }
                    for (DataType tableType : tableTypes) {
                        verifyGpuParity(manager, rawType, offsetType, tableType);
                        verifyGpuFeaturePackParity(manager, rawType, offsetType, tableType);
                        verifyGpuStridedFeaturePackParity(
                                manager, rawType, offsetType, tableType);
                    }
                }
            }
        }
    }

    @Test
    public void testBroadcastOffsetsAndTableDtype() {
        try (NDManager manager = Engine.getInstance().newBaseManager()) {
            NDArray rawIds =
                    manager.create(new int[] {0, 1, 0, 1, 0, 1})
                            .toType(DataType.INT16, false)
                            .reshape(2, 3);
            NDArray offsets = manager.create(new int[] {0, 2, 4}).reshape(1, 3);
            NDArray table = manager.arange(12).reshape(6, 2).toType(DataType.FLOAT16, false);

            NDArray actual = NDArrays.embeddingWithOffsets(rawIds, offsets, table);

            Assert.assertEquals(actual.getShape(), new Shape(2, 3, 2));
            Assert.assertEquals(actual.getDataType(), DataType.FLOAT16);
            Assert.assertEquals(
                    actual.toFloatArray(), new float[] {0, 1, 6, 7, 8, 9, 2, 3, 4, 5, 10, 11});
        }
    }

    @Test
    public void testDenseEmbeddingGradient() {
        try (NDManager manager = Engine.getInstance().newBaseManager()) {
            NDArray rawIds = manager.create(new long[] {0, 1, 3, 1}).reshape(2, 2);
            NDArray offsets =
                    manager.create(new int[] {0, 2}).toType(DataType.INT16, false).reshape(1, 2);
            NDArray table = manager.arange(8).toType(DataType.FLOAT32, false).reshape(4, 2);
            table.setRequiresGradient(true);

            try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                collector.backward(NDArrays.embeddingWithOffsets(rawIds, offsets, table).sum());
            }

            Assert.assertEquals(
                    table.getGradient().toFloatArray(), new float[] {1, 1, 0, 0, 0, 0, 3, 3});
        }
    }

    @Test
    public void testEmbeddingFeaturePackMatchesPortableReference() {
        try (NDManager manager = Engine.getInstance().newBaseManager()) {
            NDArray rawIds =
                    manager.create(new int[] {0, 1, 0, 1, 0, 1})
                            .toType(DataType.INT16, false)
                            .reshape(2, 3);
            NDArray offsets = manager.create(new int[] {0, 2, 4}).reshape(1, 3);
            NDArray table = manager.arange(12).reshape(6, 2).toType(DataType.FLOAT32, false);
            NDArray features = manager.create(new float[] {-1, -2, -3, -4}).reshape(2, 2);

            NDArray actual = NDArrays.embeddingFeaturePack(rawIds, offsets, table, features);
            NDArray expected =
                    NDArrays.embeddingWithOffsets(rawIds, offsets, table)
                            .reshape(2, 6)
                            .concat(features, 1);

            Assert.assertEquals(actual.getShape(), new Shape(2, 8));
            Assert.assertEquals(actual.getDataType(), DataType.FLOAT32);
            assertClose(actual.toFloatArray(), expected.toFloatArray());
        }
    }

    @Test
    public void testEmbeddingFeaturePackGradient() {
        try (NDManager manager = Engine.getInstance().newBaseManager()) {
            NDArray rawIds = manager.create(new long[] {0, 1, 3, 1}).reshape(2, 2);
            NDArray offsets =
                    manager.create(new int[] {0, 2}).toType(DataType.INT16, false).reshape(1, 2);
            NDArray table = manager.arange(8).toType(DataType.FLOAT32, false).reshape(4, 2);
            NDArray features = manager.arange(6).toType(DataType.FLOAT32, false).reshape(2, 3);
            table.setRequiresGradient(true);
            features.setRequiresGradient(true);

            try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                collector.backward(
                        NDArrays.embeddingFeaturePack(rawIds, offsets, table, features).sum());
            }

            Assert.assertEquals(
                    table.getGradient().toFloatArray(), new float[] {1, 1, 0, 0, 0, 0, 3, 3});
            Assert.assertEquals(
                    features.getGradient().toFloatArray(), new float[] {1, 1, 1, 1, 1, 1});
        }
    }

    @Test
    public void testEmbeddingFeaturePackRankThreeStridedLeadingSlices() {
        try (NDManager manager = Engine.getInstance().newBaseManager()) {
            verifyStridedFeaturePackParity(
                    manager, DataType.INT16, DataType.INT32, DataType.FLOAT32);
        }
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testRejectsNonBroadcastOffsets() {
        try (NDManager manager = Engine.getInstance().newBaseManager()) {
            NDArray rawIds = manager.zeros(new Shape(2, 3), DataType.INT32);
            NDArray offsets = manager.zeros(new Shape(1, 2), DataType.INT32);
            NDArray table = manager.zeros(new Shape(4, 2));
            NDArrays.embeddingWithOffsets(rawIds, offsets, table);
        }
    }

    private static void verifyGpuParity(
            NDManager manager, DataType rawType, DataType offsetType, DataType tableType) {
        NDArray rawFields =
                manager.create(
                                new int[] {
                                    0, 91, 1, 92, 0, 93,
                                    1, 94, 0, 95, 1, 96
                                },
                                new Shape(2, 3, 2))
                        .toType(rawType, false);
        NDArray offsetFields =
                manager.create(new int[] {0, 81, 2, 82, 4, 83}, new Shape(1, 3, 2))
                        .toType(offsetType, false);
        NDArray rawIds = rawFields.get("...,0");
        NDArray offsets = offsetFields.get("...,0");
        NDArray table = manager.arange(24).reshape(12, 2).toType(tableType, false);

        NDArray actual = NDArrays.embeddingWithOffsets(rawIds, offsets, table);
        NDArray expected =
                Embedding.embedding(rawIds.add(offsets), table, SparseFormat.DENSE)
                        .singletonOrThrow();

        Assert.assertEquals(actual.getShape(), new Shape(2, 3, 2));
        Assert.assertEquals(actual.getDataType(), tableType);
        float maxAbs =
                assertClose(
                        actual.toType(DataType.FLOAT32, false).toFloatArray(),
                        expected.toType(DataType.FLOAT32, false).toFloatArray());
        System.out.printf(
                "EMBEDDING_WITH_OFFSETS_PARITY rawDtype=%s offsetDtype=%s tableDtype=%s"
                    + " maxAbs=%s%n",
                rawType, offsetType, tableType, maxAbs);
    }

    private static void verifyGpuFeaturePackParity(
            NDManager manager, DataType rawType, DataType offsetType, DataType tableType) {
        NDArray rawIds =
                manager.create(new int[] {0, 1, 0, 1, 0, 1}, new Shape(2, 3))
                        .toType(rawType, false);
        NDArray offsets =
                manager.create(new int[] {0, 2, 4}, new Shape(1, 3)).toType(offsetType, false);
        NDArray table = manager.arange(24).reshape(12, 2).toType(tableType, false);
        NDArray features =
                manager.create(new float[] {-1, -2, -3, -4}, new Shape(2, 2))
                        .toType(tableType, false);

        NDArray actual = NDArrays.embeddingFeaturePack(rawIds, offsets, table, features);
        NDArray expected =
                Embedding.embedding(rawIds.add(offsets), table, SparseFormat.DENSE)
                        .singletonOrThrow()
                        .reshape(2, 6)
                        .concat(features, 1);

        Assert.assertEquals(actual.getShape(), new Shape(2, 8));
        Assert.assertEquals(actual.getDataType(), tableType);
        float maxAbs =
                assertClose(
                        actual.toType(DataType.FLOAT32, false).toFloatArray(),
                        expected.toType(DataType.FLOAT32, false).toFloatArray());
        System.out.printf(
                "EMBEDDING_FEATURE_PACK_PARITY rawDtype=%s offsetDtype=%s valueDtype=%s"
                    + " maxAbs=%s%n",
                rawType, offsetType, tableType, maxAbs);
    }

    private static void verifyGpuStridedFeaturePackParity(
            NDManager manager, DataType rawType, DataType offsetType, DataType tableType) {
        float maxAbs =
                verifyStridedFeaturePackParity(manager, rawType, offsetType, tableType);
        System.out.printf(
                "EMBEDDING_FEATURE_PACK_STRIDED_PARITY rawDtype=%s offsetDtype=%s"
                    + " valueDtype=%s maxAbs=%s%n",
                rawType, offsetType, tableType, maxAbs);
    }

    private static float verifyStridedFeaturePackParity(
            NDManager manager, DataType rawType, DataType offsetType, DataType tableType) {
        NDArray rawBacking =
                manager.create(
                                new int[] {
                                    91, 0, 1, 0, 1, 92,
                                    93, 1, 0, 1, 0, 94
                                },
                                new Shape(2, 6))
                        .toType(rawType, false);
        NDArray offsetBacking =
                manager.create(new int[] {81, 0, 2, 4, 6, 82}, new Shape(1, 6))
                        .toType(offsetType, false);
        NDArray featureBacking =
                manager.create(
                                new float[] {
                                    91, -1, -2, -3, -4, -5, -6, 92,
                                    93, -7, -8, -9, -10, -11, -12, 94
                                },
                                new Shape(2, 8))
                        .toType(tableType, false);
        NDArray rawIds = rawBacking.get(":,1:5").reshape(2, 2, 2);
        NDArray offsets = offsetBacking.get(":,1:5").reshape(1, 2, 2);
        NDArray features = featureBacking.get(":,1:7").reshape(2, 2, 3);
        NDArray table = manager.arange(32).reshape(16, 2).toType(tableType, false);

        NDArray actual = NDArrays.embeddingFeaturePack(rawIds, offsets, table, features);
        NDArray expected =
                Embedding.embedding(rawIds.add(offsets), table, SparseFormat.DENSE)
                        .singletonOrThrow()
                        .reshape(2, 2, 4)
                        .concat(features, 2);

        Assert.assertEquals(actual.getShape(), new Shape(2, 2, 7));
        Assert.assertEquals(actual.getDataType(), tableType);
        return assertClose(
                actual.toType(DataType.FLOAT32, false).toFloatArray(),
                expected.toType(DataType.FLOAT32, false).toFloatArray());
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
