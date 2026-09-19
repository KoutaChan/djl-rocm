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
import org.testng.SkipException;
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
                        verifyGpuStridedFeaturePackParity(manager, rawType, offsetType, tableType);
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

    @Test
    public void gpuFeaturePackSupportsArbitraryWidthsRanksAndStrides() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("GPU is unavailable");
        }
        long[][] shapes = {
            {2, 3},
            {2, 3, 3},
            {2, 1, 2, 3},
            {2, 1, 1, 1, 1, 1, 1, 3},
            {2, 1, 1, 1, 1, 1, 1, 1, 3},
            {0, 3}
        };
        for (DataType type :
                new DataType[] {
                    DataType.FLOAT32, DataType.FLOAT64, DataType.FLOAT16, DataType.BFLOAT16
                }) {
            for (int width : new int[] {1, 17, 257}) {
                for (long[] dimensions : shapes) {
                    try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                        Shape shape = new Shape(dimensions);
                        NDArray raw = manager.zeros(shape, DataType.INT32).add(1);
                        NDArray offsets = manager.create(new int[] {0, 2, 4});
                        NDArray table =
                                manager.arange(8 * width).reshape(8, width).toType(type, false);
                        long[] featureShape = dimensions.clone();
                        featureShape[featureShape.length - 1] = 5;
                        NDArray features = manager.ones(new Shape(featureShape), type);
                        verifyFeaturePackComposition(raw, offsets, table, features);
                    }
                }
            }
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            NDArray raw =
                    manager.create(new int[] {0, 91, 1, 92, 2, 93, 3, 94}, new Shape(2, 2, 2))
                            .get("...,0");
            NDArray offsets = manager.create(1L);
            NDArray table = manager.arange(85).toType(DataType.FLOAT32, false).reshape(5, 17);
            NDArray features =
                    manager.arange(12)
                            .toType(DataType.FLOAT32, false)
                            .reshape(2, 3, 2)
                            .get("...,0");
            verifyFeaturePackComposition(raw, offsets, table, features);
            NDArray nonContiguousTable =
                    manager.arange(85).toType(DataType.FLOAT32, false).reshape(17, 5).transpose();
            verifyFeaturePackComposition(raw, offsets, nonContiguousTable, features);
        }
    }

    @Test
    public void gpuEmbeddingOffsetsPreserveIntegerPromotionAndWrapping() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("GPU is unavailable");
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            NDArray table = manager.arange(8).toType(DataType.FLOAT32, false).reshape(4, 2);
            NDArray features = manager.ones(new Shape(1, 3), DataType.FLOAT32);
            for (DataType type : new DataType[] {DataType.INT32, DataType.INT64}) {
                long minimum = type == DataType.INT32 ? Integer.MIN_VALUE : Long.MIN_VALUE;
                NDArray raw =
                        manager.create(new long[] {minimum, minimum + 1}, new Shape(1, 2))
                                .toType(type, false);
                NDArray offsets =
                        manager.create(new long[] {minimum, minimum}, new Shape(1, 2))
                                .toType(type, false);
                verifyFeaturePackComposition(raw, offsets, table, features);
                NDArray expected =
                        Embedding.embedding(raw.add(offsets), table, SparseFormat.DENSE)
                                .singletonOrThrow();
                Assert.assertEquals(
                        NDArrays.embeddingWithOffsets(raw, offsets, table).toFloatArray(),
                        expected.toFloatArray());
            }
            // A scalar tensor follows ATen's dimensioned/scalar promotion rules.
            NDArray raw =
                    manager.create(
                            new int[] {Integer.MIN_VALUE, Integer.MIN_VALUE + 1}, new Shape(1, 2));
            verifyFeaturePackComposition(
                    raw, manager.create((long) Integer.MIN_VALUE), table, features);
            NDArray scalar = manager.create(1);
            Assert.assertEquals(
                    NDArrays.embeddingWithOffsets(scalar, scalar, table).toFloatArray(),
                    new float[] {4, 5});
        }
    }

    @Test
    public void gpuEmbeddingFeaturePackAccumulatesGradients() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("GPU is unavailable");
        }
        for (boolean packed : new boolean[] {false, true}) {
            try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                NDArray raw = manager.create(new long[] {0, 1, 3, 1}, new Shape(2, 2));
                NDArray offsets = manager.create(new int[] {0, 2}, new Shape(1, 2));
                NDArray table = manager.arange(8).toType(DataType.FLOAT32, false).reshape(4, 2);
                NDArray features = manager.ones(new Shape(2, 3), DataType.FLOAT32);
                table.setRequiresGradient(true);
                features.setRequiresGradient(true);
                try (GradientCollector collector = engine.newGradientCollector()) {
                    NDArray result =
                            packed
                                    ? NDArrays.embeddingFeaturePack(raw, offsets, table, features)
                                    : NDArrays.embeddingWithOffsets(raw, offsets, table);
                    collector.backward(result.sum());
                }
                Assert.assertEquals(
                        table.getGradient().toFloatArray(), new float[] {1, 1, 0, 0, 0, 0, 3, 3});
                if (packed) {
                    Assert.assertEquals(
                            features.getGradient().toFloatArray(), new float[] {1, 1, 1, 1, 1, 1});
                }
            }
        }
    }

    @Test
    public void gpuFeaturePackBackwardMatchesCompositionForStridedInputs() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("GPU is unavailable");
        }
        for (DataType type :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            for (int width : new int[] {1, 8, 17}) {
                for (int gradientInputs = 1; gradientInputs <= 3; ++gradientInputs) {
                    try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                        verifyFeaturePackBackward(manager, type, width, gradientInputs);
                    }
                }
            }
        }
    }

    @Test
    public void gpuFeaturePackBackwardAccumulatesManyRepeatedIndices() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("GPU is unavailable");
        }
        for (DataType type :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                NDArray raw = manager.ones(new Shape(8192, 5), DataType.INT32).get(":,1:4");
                NDArray offsets = manager.zeros(new Shape(1, 3), DataType.INT32);
                NDArray table = manager.ones(new Shape(4, 8), type);
                NDArray features = manager.ones(new Shape(8192, 3), type);
                table.setRequiresGradient(true);
                try (GradientCollector collector = engine.newGradientCollector()) {
                    collector.backward(
                            NDArrays.embeddingFeaturePack(raw, offsets, table, features)
                                    .mul(0.25f)
                                    .sum());
                }
                float[] expected = new float[32];
                for (int column = 0; column < 8; ++column) {
                    expected[8 + column] = 6144;
                }
                Assert.assertEquals(
                        table.getGradient().toType(DataType.FLOAT32, false).toFloatArray(),
                        expected);
            }
        }
    }

    @Test
    public void gpuFeaturePackBackwardPreservesSignedRepeatedContributions() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("GPU is unavailable");
        }
        float[] weights = new float[2048 * 27];
        for (int index = 0; index < weights.length; ++index) {
            weights[index] =
                    (float) (1.7 * Math.sin(index * 0.119) + 0.37 * Math.cos(index * 0.031));
        }
        for (DataType type :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                NDArray raw = manager.ones(new Shape(2048, 5), DataType.INT32).get(":,1:4");
                NDArray offsets = manager.zeros(new Shape(1, 3), DataType.INT32);
                NDArray table = manager.ones(new Shape(4, 8), type);
                NDArray referenceTable = manager.ones(new Shape(4, 8), type);
                NDArray features = manager.ones(new Shape(2048, 3), type);
                NDArray upstream = manager.create(weights, new Shape(2048, 27)).toType(type, false);
                table.setRequiresGradient(true);
                referenceTable.setRequiresGradient(true);
                try (GradientCollector collector = engine.newGradientCollector()) {
                    collector.backward(
                            NDArrays.embeddingFeaturePack(raw, offsets, table, features)
                                    .mul(upstream)
                                    .sum());
                }
                try (GradientCollector collector = engine.newGradientCollector()) {
                    collector.backward(
                            Embedding.embedding(
                                            raw.add(offsets), referenceTable, SparseFormat.DENSE)
                                    .singletonOrThrow()
                                    .reshape(2048, 24)
                                    .concat(features, 1)
                                    .mul(upstream)
                                    .sum());
                }
                float[] actual = table.getGradient().toType(DataType.FLOAT32, false).toFloatArray();
                float[] expected =
                        referenceTable.getGradient().toType(DataType.FLOAT32, false).toFloatArray();
                float absoluteTolerance =
                        type == DataType.BFLOAT16
                                ? 0.008f
                                : type == DataType.FLOAT16 ? 0.001f : 3e-4f;
                float relativeTolerance =
                        type == DataType.BFLOAT16
                                ? 0.008f
                                : type == DataType.FLOAT16 ? 0.001f : 1e-5f;
                for (int index = 0; index < actual.length; ++index) {
                    Assert.assertEquals(
                            actual[index],
                            expected[index],
                            absoluteTolerance + relativeTolerance * Math.abs(expected[index]),
                            type + " index=" + index);
                }
            }
        }
    }

    @Test
    public void gpuFeaturePackBackwardKeepsLargeTableFallback() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("GPU is unavailable");
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            NDArray raw = manager.create(new int[] {0, 1, 1, 0}, new Shape(2, 2));
            NDArray offsets = manager.create(new int[] {0, 131071}, new Shape(1, 2));
            NDArray table = manager.ones(new Shape(131073, 8));
            NDArray features = manager.ones(new Shape(2, 3));
            table.setRequiresGradient(true);
            try (GradientCollector collector = engine.newGradientCollector()) {
                collector.backward(
                        NDArrays.embeddingFeaturePack(raw, offsets, table, features).sum());
            }
            float[] gradient = table.getGradient().toFloatArray();
            for (int row = 0; row < 131073; ++row) {
                float expected = row <= 1 || row >= 131071 ? 1 : 0;
                for (int column = 0; column < 8; ++column) {
                    Assert.assertEquals(gradient[row * 8 + column], expected);
                }
            }
        }
    }

    @Test
    public void gpuUnusedFeaturePackDoesNotContributeGradients() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("GPU is unavailable");
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            NDArray raw = manager.create(new int[] {0, 1, 0, 1}, new Shape(2, 2));
            NDArray offsets = manager.create(new int[] {0, 2}, new Shape(1, 2));
            NDArray table = manager.ones(new Shape(4, 8));
            NDArray features = manager.ones(new Shape(2, 3));
            table.setRequiresGradient(true);
            features.setRequiresGradient(true);
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArrays.embeddingFeaturePack(raw, offsets, table, features);
                collector.backward(table.mul(2).sum().add(features.mul(3).sum()));
            }
            Assert.assertEquals(
                    table.getGradient().toFloatArray(),
                    manager.full(new Shape(32), 2.0f).toFloatArray());
            Assert.assertEquals(
                    features.getGradient().toFloatArray(), new float[] {3, 3, 3, 3, 3, 3});
        }
    }

    private static void verifyFeaturePackBackward(
            NDManager manager, DataType type, int width, int gradientInputs) {
        NDArray raw =
                manager.create(
                                new int[] {
                                    91, 0, 1, 0, 0, 1, 0, 92,
                                    93, 0, 1, 0, 1, 0, 1, 94,
                                    95, 1, 0, 1, 0, 1, 0, 96
                                },
                                new Shape(3, 8))
                        .toType(DataType.INT16, false)
                        .get(":,1:7")
                        .reshape(3, 2, 3);
        NDArray offsets =
                manager.create(new int[] {81, 0, 2, 4, 0, 2, 4, 82}, new Shape(1, 8))
                        .get(":,1:7")
                        .reshape(1, 2, 3);
        NDArray table = manager.arange(6 * width).reshape(6, width).toType(type, false);
        NDArray referenceTable = manager.arange(6 * width).reshape(6, width).toType(type, false);
        NDArray features = manager.arange(24).reshape(3, 8).toType(type, false);
        NDArray referenceFeatures = manager.arange(24).reshape(3, 8).toType(type, false);
        boolean tableGradient = (gradientInputs & 1) != 0;
        boolean featureGradient = (gradientInputs & 2) != 0;
        table.setRequiresGradient(tableGradient);
        referenceTable.setRequiresGradient(tableGradient);
        features.setRequiresGradient(featureGradient);
        referenceFeatures.setRequiresGradient(featureGradient);
        NDArray upstream =
                manager.arange(6 * (3 * width + 3))
                        .mod(23)
                        .sub(11)
                        .div(64)
                        .reshape(3, 2, 3 * width + 3)
                        .toType(type, false);
        NDArray actual;
        try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
            actual =
                    NDArrays.embeddingFeaturePack(
                            raw, offsets, table, features.get(":,1:7").reshape(3, 2, 3));
            collector.backward(actual.mul(upstream).sum());
        }
        NDArray expected;
        try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
            expected =
                    Embedding.embedding(raw.add(offsets), referenceTable, SparseFormat.DENSE)
                            .singletonOrThrow()
                            .reshape(3, 2, 3 * width)
                            .concat(referenceFeatures.get(":,1:7").reshape(3, 2, 3), 2);
            collector.backward(expected.mul(upstream).sum());
        }
        assertClose(
                actual.toType(DataType.FLOAT32, false).toFloatArray(),
                expected.toType(DataType.FLOAT32, false).toFloatArray());
        if (tableGradient) {
            assertClose(
                    table.getGradient().toType(DataType.FLOAT32, false).toFloatArray(),
                    referenceTable.getGradient().toType(DataType.FLOAT32, false).toFloatArray());
        }
        if (featureGradient) {
            assertClose(
                    features.getGradient().toType(DataType.FLOAT32, false).toFloatArray(),
                    referenceFeatures.getGradient().toType(DataType.FLOAT32, false).toFloatArray());
        }
    }

    private static void verifyFeaturePackComposition(
            NDArray raw, NDArray offsets, NDArray table, NDArray features) {
        long[] embeddedShape = raw.getShape().getShape().clone();
        embeddedShape[embeddedShape.length - 1] *= table.getShape().get(1);
        NDArray expected =
                Embedding.embedding(raw.add(offsets), table, SparseFormat.DENSE)
                        .singletonOrThrow()
                        .reshape(embeddedShape)
                        .concat(features, embeddedShape.length - 1);
        NDArray actual = NDArrays.embeddingFeaturePack(raw, offsets, table, features);
        Assert.assertEquals(actual.getShape(), expected.getShape());
        Assert.assertEquals(actual.getDataType(), expected.getDataType());
        Assert.assertEquals(
                actual.toType(DataType.FLOAT32, false).toFloatArray(),
                expected.toType(DataType.FLOAT32, false).toFloatArray());
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
        float maxAbs = verifyStridedFeaturePackParity(manager, rawType, offsetType, tableType);
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
