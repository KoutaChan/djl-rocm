/*
 * Copyright 2026 KoutaChan.
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
import ai.djl.training.GradientCollector;

import org.testng.Assert;
import org.testng.annotations.Test;

/** Verifies portable structured-operation semantics and native PyTorch parity. */
public class StructuredAttentionTest {

    private static final int[] MAPPED_ATTENTION_SEEDS = {20260901, 20260903, 20260907, 20260909};
    private static final int MAPPED_ATTENTION_BFLOAT16_ROUNDING_SEED = 20260907;

    @Test
    public void relationAttentionUsesPairwiseBias() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray query = manager.zeros(new Shape(1, 1, 2, 1));
            NDArray key = manager.zeros(new Shape(1, 1, 2, 1));
            NDArray value = manager.create(new float[] {1f, 3f}, new Shape(1, 1, 2, 1));
            NDArray relationKeys = manager.zeros(new Shape(1, 1, 1, 2));
            NDArray relationBias =
                    manager.create(
                            new float[] {0f, 0f, 0f, (float) Math.log(3)}, new Shape(1, 1, 2, 2));
            NDArray relationIds = manager.create(new long[] {0, 1, 1, 0}, new Shape(2, 2));

            NDArray result =
                    NDArrays.relationBiasedScaledDotProductAttention(
                            query, key, value, relationKeys, relationBias, relationIds, 1.0);

            assertClose(result.toFloatArray(), new float[] {2f, 2.5f}, 1e-6f);
        }
    }

    @Test
    public void groupedAttentionCombinesSharedAndIndexedTokens() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray query = manager.zeros(new Shape(2, 1, 1));
            NDArray shared = manager.create(new float[] {0f, 1f, 0f, 3f}, new Shape(1, 2, 2));
            NDArray sharedDeltas = manager.zeros(new Shape(2, 2, 2));
            NDArray indexedDeltas =
                    manager.create(new float[] {0f, 0f, 0f, 4f}, new Shape(2, 1, 2));
            NDArray indexedIds = manager.create(new int[] {0, 1}, new Shape(2, 1));

            NDArray result =
                    NDArrays.groupedIndexedScaledDotProductAttention(
                            query, shared, sharedDeltas, indexedDeltas, indexedIds, 2, 1.0);

            assertClose(result.toFloatArray(), new float[] {2f, 3f}, 1e-6f);
        }
    }

    @Test
    public void mappedGroupedAttentionReadsSharedAndDeltaTables() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray query = manager.zeros(new Shape(2, 1, 1));
            NDArray shared =
                    manager.create(
                            new float[] {0f, 1f, 0f, 3f, 0f, 5f, 0f, 7f}, new Shape(2, 2, 2));
            NDArray groupIndices = manager.create(new int[] {1, 0});
            NDArray deltaTable = manager.create(new float[] {0f, 0f, 0f, 2f}, new Shape(2, 2));
            NDArray deltaIndices = manager.create(new int[] {0, 1, 1, 0}, new Shape(2, 2));
            NDArray indexedDeltas = manager.zeros(new Shape(2, 1, 2));
            NDArray indexedIds = manager.zeros(new Shape(2, 1), DataType.INT32);

            NDArray result =
                    NDArrays.mappedGroupedIndexedScaledDotProductAttention(
                            query,
                            shared,
                            groupIndices,
                            deltaTable,
                            deltaIndices,
                            indexedDeltas,
                            indexedIds,
                            1.0);

            assertClose(result.toFloatArray(), new float[] {7f, 3f}, 1e-6f);
        }
    }

    @Test
    public void groupedPackedAttentionSharesQueriesAndMasksMemory() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray query = manager.zeros(new Shape(1, 1, 1));
            NDArray packedKeyValue =
                    manager.create(
                            new float[] {0f, 1f, 0f, 3f, 0f, 2f, 0f, 10f}, new Shape(1, 2, 2, 2));
            NDArray mask = manager.create(new int[] {1, 1, 1, 0}, new Shape(1, 2, 2));

            NDArray result =
                    NDArrays.groupedPackedScaledDotProductAttention(
                            query, packedKeyValue, mask, 1, 1.0);

            Assert.assertEquals(result.getShape(), new Shape(1, 2, 1, 1));
            assertClose(result.toFloatArray(), new float[] {2f, 2f}, 1e-6f);
        }
    }

    @Test
    public void groupedPackedAttentionPreservesPortableGradients() {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(Device.cpu());
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray query = requiringGradient(manager.randomNormal(new Shape(2, 3, 8)));
            NDArray packedKeyValue =
                    requiringGradient(manager.randomNormal(new Shape(2, 4, 5, 16)));
            NDArray mask = manager.ones(new Shape(2, 4, 5), DataType.INT32);

            collector.backward(
                    NDArrays.groupedPackedScaledDotProductAttention(
                                    query, packedKeyValue, mask, 2, 0.5)
                            .sum());

            assertFiniteNonzeroGradient(query);
            assertFiniteNonzeroGradient(packedKeyValue);
        }
    }

    @Test
    public void groupedPackedAttentionProductionShapeMatchesPortableDtypeBoundaries() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            engine.setRandomSeed(20260913);
            try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                int batch = 2;
                int queryTokens = 34;
                int groups = 4;
                int keyTokens = 29;
                int heads = 4;
                int keyFeatures = 16;
                int valueFeatures = 16;
                Assert.assertTrue(keyTokens > valueFeatures);
                Assert.assertTrue(valueFeatures < 32);
                int queryWidth = heads * keyFeatures;
                int packedWidth = queryWidth + heads * valueFeatures;
                NDArray query =
                        manager.randomNormal(new Shape(batch, queryTokens, queryWidth), dataType);
                NDArray packedKeyValue =
                        manager.randomNormal(
                                new Shape(batch, groups, keyTokens, packedWidth), dataType);
                NDArray mask = manager.ones(new Shape(batch, groups, keyTokens), dataType);
                mask.set(new ai.djl.ndarray.index.NDIndex("..., -1"), 0);

                NDArray expected =
                        groupedPackedAttentionReference(query, packedKeyValue, mask, heads, 0.25);
                NDArray actual =
                        NDArrays.groupedPackedScaledDotProductAttention(
                                query, packedKeyValue, mask, heads, 0.25);

                Assert.assertEquals(actual.getDataType(), dataType);
                Assert.assertEquals(expected.getDataType(), dataType);

                float tolerance =
                        dataType == DataType.FLOAT32
                                ? 2e-4f
                                : dataType == DataType.FLOAT16 ? 3e-3f : 2e-2f;
                assertClose(
                        actual.toType(DataType.FLOAT32, false).toFloatArray(),
                        expected.toType(DataType.FLOAT32, false).toFloatArray(),
                        tolerance);
            }
        }
    }

    @Test
    public void groupedPackedAttentionNativeForwardDoesNotDependOnAutograd() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            engine.setRandomSeed(20260917);
            try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                int batch = 2;
                int queryTokens = 34;
                int groups = 4;
                int keyTokens = 29;
                int heads = 4;
                int queryWidth = 64;
                int packedWidth = 128;
                NDArray queryValues =
                        manager.randomNormal(new Shape(batch, queryTokens, queryWidth), dataType);
                NDArray packedValues =
                        manager.randomNormal(
                                new Shape(batch, groups, keyTokens, packedWidth), dataType);
                NDArray mask = manager.ones(new Shape(batch, groups, keyTokens));
                mask.set(new ai.djl.ndarray.index.NDIndex("..., -1"), 0);

                NDArray inference =
                        NDArrays.groupedPackedScaledDotProductAttention(
                                queryValues, packedValues, mask, heads, 0.25);
                NDArray trainingQuery = requiringGradient(queryValues.duplicate());
                NDArray trainingPacked = requiringGradient(packedValues.duplicate());
                try (GradientCollector collector = engine.newGradientCollector()) {
                    NDArray training =
                            NDArrays.groupedPackedScaledDotProductAttention(
                                    trainingQuery, trainingPacked, mask, heads, 0.25);

                    Assert.assertEquals(training.getShape(), inference.getShape());
                    Assert.assertEquals(training.getDataType(), inference.getDataType());
                    Assert.assertEquals(
                            training.toByteBuffer(),
                            inference.toByteBuffer(),
                            "autograd changed the grouped packed forward for " + dataType);
                    collector.backward(training.sum());
                }
            }
        }
    }

    @Test
    public void groupedPackedAttentionGradientsMatchPortableReference() {
        Engine engine = Engine.getInstance();
        engine.setRandomSeed(20260918);
        verifyGroupedPackedAttentionGradients(engine, Device.cpu(), DataType.FLOAT32, true, true);
        if (engine.getGpuCount() == 0) {
            return;
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            engine.setRandomSeed(20260918);
            verifyGroupedPackedAttentionGradients(engine, Device.gpu(), dataType, true, true);
        }
    }

    @Test
    public void groupedPackedAttentionNativeBackwardHonorsPartialGradients() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        engine.setRandomSeed(20260919);
        verifyGroupedPackedAttentionGradients(engine, Device.gpu(), DataType.BFLOAT16, true, false);
        engine.setRandomSeed(20260919);
        verifyGroupedPackedAttentionGradients(engine, Device.gpu(), DataType.BFLOAT16, false, true);
    }

    @Test
    public void groupedPackedAttentionNativeBackwardMatchesTiledBoundaryShape() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        engine.setRandomSeed(20260921);
        verifyGroupedPackedAttentionGradients(
                engine,
                Device.gpu(),
                DataType.FLOAT32,
                DataType.INT32,
                true,
                true,
                1,
                37,
                4,
                32,
                4,
                16,
                17);
    }

    @Test
    public void groupedPackedAttentionNativeBackwardSupportsEveryMaskDtype() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        for (DataType maskType :
                new DataType[] {
                    DataType.BOOLEAN,
                    DataType.INT32,
                    DataType.FLOAT32,
                    DataType.FLOAT16,
                    DataType.BFLOAT16
                }) {
            engine.setRandomSeed(20260920);
            verifyGroupedPackedAttentionGradients(
                    engine, Device.gpu(), DataType.BFLOAT16, maskType, true, true);
        }
    }

    @Test
    public void groupedPackedAttentionAllInvalidRowsMatchPortableDtypeBehavior() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            engine.setRandomSeed(20260914);
            try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                int heads = 2;
                NDArray query = manager.randomNormal(new Shape(1, 2, 8), dataType);
                NDArray packedKeyValue = manager.randomNormal(new Shape(1, 1, 3, 16), dataType);
                NDArray mask = manager.zeros(new Shape(1, 1, 3), dataType);

                NDArray expected =
                        groupedPackedAttentionReference(query, packedKeyValue, mask, heads, 0.5);
                NDArray actual =
                        NDArrays.groupedPackedScaledDotProductAttention(
                                query, packedKeyValue, mask, heads, 0.5);

                Assert.assertEquals(actual.getDataType(), dataType);
                Assert.assertEquals(expected.getDataType(), dataType);

                float[] expectedValues = expected.toType(DataType.FLOAT32, false).toFloatArray();
                float[] actualValues = actual.toType(DataType.FLOAT32, false).toFloatArray();

                if (dataType == DataType.FLOAT16) {
                    assertAllNaN(expectedValues);
                    assertAllNaN(actualValues);
                } else {
                    assertClose(
                            actualValues,
                            expectedValues,
                            dataType == DataType.FLOAT32 ? 2e-4f : 2e-2f);
                }
            }
        }
    }

    @Test
    public void groupedPackedAttentionNonContiguousGpuInputsUsePortableFallback() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        engine.setRandomSeed(20260915);
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            int batch = 2;
            int queryTokens = 7;
            int groups = 3;
            int keyTokens = 5;
            int heads = 2;
            int keyFeatures = 4;
            int valueFeatures = 3;
            int queryWidth = heads * keyFeatures;
            int packedWidth = queryWidth + heads * valueFeatures;
            NDArray query =
                    manager.randomNormal(new Shape(batch, queryWidth, queryTokens)).swapAxes(1, 2);
            NDArray packedKeyValue =
                    manager.randomNormal(new Shape(batch, groups, packedWidth, keyTokens))
                            .swapAxes(2, 3);
            NDArray mask = manager.ones(new Shape(batch, groups, keyTokens), DataType.INT32);
            mask.set(new ai.djl.ndarray.index.NDIndex("..., -1"), 0);

            NDArray expected =
                    groupedPackedAttentionReference(query, packedKeyValue, mask, heads, 0.5);
            NDArray actual =
                    NDArrays.groupedPackedScaledDotProductAttention(
                            query, packedKeyValue, mask, heads, 0.5);

            assertClose(actual.toFloatArray(), expected.toFloatArray(), 2e-4f);
        }
    }

    @Test
    public void groupedPackedAttentionFloat16FallbackAcceptsFullyMaskedRows() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            NDArray query =
                    manager.randomNormal(new Shape(1, 4, 2), DataType.FLOAT16).swapAxes(1, 2);
            NDArray packedKeyValue =
                    manager.randomNormal(new Shape(1, 1, 8, 3), DataType.FLOAT16).swapAxes(2, 3);
            NDArray mask = manager.zeros(new Shape(1, 1, 3), DataType.INT32);

            NDArray output =
                    NDArrays.groupedPackedScaledDotProductAttention(
                            query, packedKeyValue, mask, 1, 0.5);

            assertAllNaN(output.toType(DataType.FLOAT32, false).toFloatArray());
        }
    }

    private static void assertAllNaN(float[] values) {
        for (float value : values) {
            Assert.assertTrue(Float.isNaN(value), "expected NaN but found " + value);
        }
    }

    @Test
    public void residualAddLayerNormUpdatesOwnedBuffer() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray residual = manager.create(new float[] {1f, 3f}, new Shape(1, 2));
            NDArray update = manager.ones(new Shape(1, 2));
            NDArray weight = manager.ones(new Shape(2));
            NDArray bias = manager.zeros(new Shape(2));

            NDArray normalized =
                    NDArrays.addToOwnedResidualAndLayerNorm(
                            residual, update, weight, bias, 1.0e-5f);

            assertClose(residual.toFloatArray(), new float[] {2f, 4f}, 1e-6f);
            assertClose(normalized.toFloatArray(), new float[] {-0.999995f, 0.999995f}, 1e-5f);
        }
    }

    @Test
    public void rowGatherAndScatterPreserveTrailingDimensions() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray source =
                    manager.create(
                            new float[] {
                                0f, 1f, 2f, 3f,
                                4f, 5f, 6f, 7f,
                                8f, 9f, 10f, 11f,
                                12f, 13f, 14f, 15f
                            },
                            new Shape(4, 2, 2));
            NDArray indices = manager.create(new int[] {3, 1});

            NDArray gathered = NDArrays.gatherRows(source, indices);
            NDArray scattered = NDArrays.scatterRows(gathered, indices, 5);

            Assert.assertEquals(gathered.getShape(), new Shape(2, 2, 2));
            assertClose(
                    gathered.toFloatArray(), new float[] {12f, 13f, 14f, 15f, 4f, 5f, 6f, 7f}, 0f);
            Assert.assertEquals(scattered.getShape(), new Shape(5, 2, 2));
            assertClose(
                    scattered.toFloatArray(),
                    new float[] {
                        0f, 0f, 0f, 0f,
                        4f, 5f, 6f, 7f,
                        0f, 0f, 0f, 0f,
                        12f, 13f, 14f, 15f,
                        0f, 0f, 0f, 0f
                    },
                    0f);
        }
    }

    @Test
    public void rowGatherAndScatterPreserveGradients() {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager();
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray source = manager.ones(new Shape(4, 2));
            source.setRequiresGradient(true);
            NDArray indices = manager.create(new int[] {2, 0, 2});

            collector.backward(NDArrays.gatherRows(source, indices).sum());

            assertClose(
                    source.getGradient().toFloatArray(),
                    new float[] {1f, 1f, 0f, 0f, 2f, 2f, 0f, 0f},
                    0f);
        }

        try (NDManager manager = engine.newBaseManager();
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray rows = manager.ones(new Shape(2, 2));
            rows.setRequiresGradient(true);
            NDArray indices = manager.create(new int[] {3, 1});

            collector.backward(NDArrays.scatterRows(rows, indices, 5).sum());

            assertClose(rows.getGradient().toFloatArray(), new float[] {1f, 1f, 1f, 1f}, 0f);
        }
    }

    @Test
    public void structuredAttentionPreservesAutograd() {
        Engine engine = Engine.getInstance();
        verifyStructuredAttentionGradients(engine, Device.cpu(), DataType.FLOAT32);
        if (engine.getGpuCount() > 0) {
            verifyStructuredAttentionGradients(engine, Device.gpu(), DataType.FLOAT32);
            verifyStructuredAttentionGradients(engine, Device.gpu(), DataType.BFLOAT16);
        }
    }

    @Test
    public void structuredAttentionHonorsPartialGradientRequests() {
        Engine engine = Engine.getInstance();
        verifyPartialGradientRequests(engine, Device.cpu(), DataType.FLOAT32);
        if (engine.getGpuCount() > 0) {
            verifyPartialGradientRequests(engine, Device.gpu(), DataType.BFLOAT16);
            verifyLargeRelationBiasGradient(engine);
        }
    }

    @Test
    public void ownedResidualLayerNormRejectsAutograd() {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager()) {
            GradientCollector collector = engine.newGradientCollector();
            try {
                NDArray residual = manager.ones(new Shape(2, 4));
                residual.setRequiresGradient(true);
                NDArray update = manager.ones(new Shape(2, 4));
                NDArray weight = manager.ones(new Shape(4));
                NDArray bias = manager.zeros(new Shape(4));
                Assert.expectThrows(
                        RuntimeException.class,
                        () ->
                                NDArrays.addToOwnedResidualAndLayerNorm(
                                        residual, update, weight, bias, 1.0e-5f));
            } finally {
                collector.close();
            }
        }
    }

    @Test
    public void structuredOperationsDoNotRetainBatchArrays() {
        Engine engine = Engine.getInstance();
        try (NDManager modelManager = engine.newBaseManager(Device.cpu())) {
            NDArray relationKeys = modelManager.randomNormal(new Shape(1, 1, 2, 4));
            NDArray relationBias = modelManager.randomNormal(new Shape(1, 1, 2, 3));
            NDArray relationIds =
                    modelManager.create(new long[] {0, 1, 2, 3, 0, 1}, new Shape(2, 3));
            NDArray sharedKeyValues = modelManager.randomNormal(new Shape(1, 3, 4));
            NDArray indexedSharedIds = modelManager.create(new int[] {1, 2}, new Shape(2, 1));
            int retainedArrays = modelManager.getManagedArrays().size();

            for (int iteration = 0; iteration < 3; iteration++) {
                runStructuredStep(
                        modelManager,
                        relationKeys,
                        relationBias,
                        relationIds,
                        sharedKeyValues,
                        indexedSharedIds);
            }

            Assert.assertEquals(modelManager.getManagedArrays().size(), retainedArrays);
        }
    }

    @Test
    public void portableOperationsSupportArbitraryDimensions() {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(Device.cpu())) {
            verifyRelationAttention(manager);
            verifyGroupedAttention(manager);
            verifyMappedGroupedAttention(manager);
            verifyRelationAttentionLeadingDimensions(manager);
            verifyGroupedAttentionLeadingDimensions(manager);
            verifyGroupedPackedAttentionLeadingDimensions(manager);
            verifyResidualLayerNorm(manager);
        }
    }

    @Test
    public void nativeKernelsMatchPortableSemantics() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            verifyRelationAttention(manager);
            verifyGroupedAttention(manager);
            verifyMappedGroupedAttention(manager);
            verifyRelationAttentionLeadingDimensions(manager);
            verifyGroupedAttentionLeadingDimensions(manager);
            verifyGroupedPackedAttentionLeadingDimensions(manager);
            verifyResidualLayerNorm(manager);
        }
    }

    @Test
    public void mappedGroupedAttentionNativeInt32MatchesReferenceAcrossDtypes() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            for (int seed : MAPPED_ATTENTION_SEEDS) {
                engine.setRandomSeed(seed);
                try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                    AttentionComparison comparison =
                            verifyMappedGroupedAttentionForward(
                                    manager, dataType, MappedAttentionPath.INT32_NATIVE);
                    printMappedAttentionComparison(dataType, seed, comparison);
                }
            }
        }
    }

    @Test
    public void mappedGroupedAttentionInt64FallbackMatchesReferenceAcrossDtypes() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            for (int seed : MAPPED_ATTENTION_SEEDS) {
                engine.setRandomSeed(seed);
                try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                    AttentionComparison comparison =
                            verifyMappedGroupedAttentionForward(
                                    manager, dataType, MappedAttentionPath.INT64_EAGER_FALLBACK);
                    printMappedAttentionComparison(dataType, seed, comparison);
                }
            }
        }
    }

    @Test
    public void mappedGroupedAttentionNativeBfloat16PreservesEagerRoundingBoundaries() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        engine.setRandomSeed(MAPPED_ATTENTION_BFLOAT16_ROUNDING_SEED);
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            AttentionComparison comparison =
                    verifyMappedGroupedAttentionForward(
                            manager, DataType.BFLOAT16, MappedAttentionPath.INT32_NATIVE);
            System.out.printf(
                    "MAPPED_GROUPED_ATTENTION_BFLOAT16_ROUNDING seed=%d maxAbs=%g%n",
                    MAPPED_ATTENTION_BFLOAT16_ROUNDING_SEED, comparison.maximumAbsoluteError);
        }
    }

    @Test
    public void mappedGroupedAttentionAutogradMatchesReferenceAcrossDtypes() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            engine.setRandomSeed(20260911);
            verifyMappedGroupedAttentionGradients(engine, Device.gpu(), dataType);
            System.out.printf(
                    "MAPPED_GROUPED_ATTENTION_AUTOGRAD dtype=%s path=NATIVE seed=20260911%n",
                    dataType);
        }
    }

    @Test
    public void mappedGroupedAttentionNativeBackwardMatchesProductionGeometry() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        engine.setRandomSeed(20260913);
        verifyMappedGroupedAttentionProductionGradients(engine, DataType.BFLOAT16);
    }

    @Test
    public void bfloat16FallbackComparisonDoesNotRelaxNativeTolerance() {
        float[] actualAtQuantizationBoundary = {1.53125f};
        float[] expected = {1.5625f};
        Assert.expectThrows(
                AssertionError.class,
                () ->
                        assertMappedAttentionClose(
                                actualAtQuantizationBoundary,
                                expected,
                                DataType.BFLOAT16,
                                MappedAttentionPath.INT32_NATIVE));

        AttentionComparison comparison =
                assertMappedAttentionClose(
                        actualAtQuantizationBoundary,
                        expected,
                        DataType.BFLOAT16,
                        MappedAttentionPath.INT64_EAGER_FALLBACK);
        Assert.assertEquals(comparison.maximumBfloat16Ulps, 4);

        Assert.expectThrows(
                AssertionError.class,
                () ->
                        assertMappedAttentionClose(
                                new float[] {1.484375f},
                                expected,
                                DataType.BFLOAT16,
                                MappedAttentionPath.INT64_EAGER_FALLBACK));
    }

    @Test
    public void nativeResidualLayerNormSupportsMixedPrecisionParameters() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            verifyResidualLayerNormDataTypes(
                    manager, DataType.FLOAT32, DataType.FLOAT16, DataType.FLOAT16, 5e-4f);
            verifyResidualLayerNormDataTypes(
                    manager, DataType.FLOAT32, DataType.BFLOAT16, DataType.BFLOAT16, 5e-4f);
            verifyResidualLayerNormDataTypes(
                    manager, DataType.FLOAT16, DataType.FLOAT16, DataType.FLOAT16, 2e-3f);
            verifyResidualLayerNormDataTypes(
                    manager, DataType.BFLOAT16, DataType.BFLOAT16, DataType.BFLOAT16, 2e-2f);
        }
    }

    @Test
    public void ownedResidualLayerNormFallsBackForUnsupportedNativeDtype() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            verifyResidualLayerNormDataTypes(
                    manager, DataType.FLOAT32, DataType.FLOAT64, DataType.FLOAT32, 5e-4f);
        }
    }

    @Test
    public void ownedResidualLayerNormRejectsUnintendedMixedPrecisionParameters() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            NDArray residual = manager.ones(new Shape(2, 17));
            NDArray update = manager.ones(new Shape(2, 17));
            NDArray weight = manager.ones(new Shape(17)).toType(DataType.FLOAT16, false);
            NDArray bias = manager.zeros(new Shape(17)).toType(DataType.FLOAT16, false);

            Assert.expectThrows(
                    RuntimeException.class,
                    () ->
                            NDArrays.addToOwnedResidualAndLayerNorm(
                                    residual, update, weight, bias, 1.0e-5f));
        }
    }

    private static void verifyRelationAttentionLeadingDimensions(NDManager manager) {
        int batch = 2;
        int alternatives = 3;
        int heads = 2;
        int queryTokens = 4;
        int keyTokens = 6;
        int keyFeatures = 5;
        int valueFeatures = 3;
        int relations = 7;
        NDArray query =
                manager.randomNormal(
                        new Shape(batch, alternatives, heads, queryTokens, keyFeatures));
        NDArray key =
                manager.randomNormal(new Shape(batch, alternatives, heads, keyTokens, keyFeatures));
        NDArray value =
                manager.randomNormal(
                        new Shape(batch, alternatives, heads, keyTokens, valueFeatures));
        NDArray relationKeys =
                manager.randomNormal(new Shape(batch, alternatives, heads, keyFeatures, relations));
        NDArray relationBias =
                manager.randomNormal(new Shape(batch, alternatives, 1, queryTokens, keyTokens));
        int[] ids = new int[batch * alternatives * queryTokens * keyTokens];
        for (int index = 0; index < ids.length; index++) {
            ids[index] = index % relations;
        }
        NDArray relationIds =
                manager.create(ids, new Shape(batch, alternatives, queryTokens, keyTokens));
        double scale = 0.31;

        NDArray expected =
                NDArrays.relationBiasedScaledDotProductAttention(
                        query.reshape(batch * alternatives, heads, queryTokens, keyFeatures),
                        key.reshape(batch * alternatives, heads, keyTokens, keyFeatures),
                        value.reshape(batch * alternatives, heads, keyTokens, valueFeatures),
                        relationKeys.reshape(batch * alternatives, heads, keyFeatures, relations),
                        relationBias.reshape(batch * alternatives, 1, queryTokens, keyTokens),
                        relationIds.reshape(batch * alternatives, queryTokens, keyTokens),
                        scale);
        NDArray actual =
                NDArrays.relationBiasedScaledDotProductAttention(
                        query, key, value, relationKeys, relationBias, relationIds, scale);

        Assert.assertEquals(
                actual.getShape(),
                new Shape(batch, alternatives, heads, queryTokens, valueFeatures));
        assertClose(
                actual.toFloatArray(), expected.reshape(actual.getShape()).toFloatArray(), 2e-4f);
    }

    private static void verifyGroupedAttentionLeadingDimensions(NDManager manager) {
        int batches = 2;
        int groupsPerBatch = 3;
        int queriesPerGroup = 4;
        int heads = 2;
        int keyFeatures = 5;
        int valueFeatures = 3;
        int sharedTokens = 7;
        int indexedTokens = 4;
        int packedWidth = heads * (keyFeatures + valueFeatures);
        int queryCount = batches * groupsPerBatch * queriesPerGroup;
        int groupCount = batches * groupsPerBatch;
        NDArray query =
                manager.randomNormal(
                        new Shape(batches, groupsPerBatch, queriesPerGroup, heads, keyFeatures));
        NDArray shared =
                manager.randomNormal(new Shape(batches, groupsPerBatch, sharedTokens, packedWidth));
        NDArray sharedDeltas =
                manager.randomNormal(
                        new Shape(
                                batches,
                                groupsPerBatch,
                                queriesPerGroup,
                                sharedTokens,
                                packedWidth));
        NDArray indexedDeltas =
                manager.randomNormal(
                        new Shape(
                                batches,
                                groupsPerBatch,
                                queriesPerGroup,
                                indexedTokens,
                                packedWidth));
        int[] ids = new int[queryCount * indexedTokens];
        for (int index = 0; index < ids.length; index++) {
            ids[index] = index % (sharedTokens + 1);
        }
        NDArray indexedIds =
                manager.create(
                        ids, new Shape(batches, groupsPerBatch, queriesPerGroup, indexedTokens));
        double scale = 0.29;

        NDArray expected =
                NDArrays.groupedIndexedScaledDotProductAttention(
                        query.reshape(queryCount, heads, keyFeatures),
                        shared.reshape(groupCount, sharedTokens, packedWidth),
                        sharedDeltas.reshape(queryCount, sharedTokens, packedWidth),
                        indexedDeltas.reshape(queryCount, indexedTokens, packedWidth),
                        indexedIds.reshape(queryCount, indexedTokens),
                        queriesPerGroup,
                        scale);
        NDArray actual =
                NDArrays.groupedIndexedScaledDotProductAttention(
                        query,
                        shared,
                        sharedDeltas,
                        indexedDeltas,
                        indexedIds,
                        queriesPerGroup,
                        scale);

        Assert.assertEquals(
                actual.getShape(),
                new Shape(batches, groupsPerBatch, queriesPerGroup, heads, valueFeatures));
        assertClose(
                actual.toFloatArray(), expected.reshape(actual.getShape()).toFloatArray(), 2e-4f);
    }

    private static void verifyGroupedPackedAttentionLeadingDimensions(NDManager manager) {
        int batches = 2;
        int alternatives = 3;
        int queryTokens = 4;
        int groups = 2;
        int keyTokens = 5;
        int heads = 2;
        int keyFeatures = 3;
        int valueFeatures = 4;
        int queryWidth = heads * keyFeatures;
        int packedWidth = heads * (keyFeatures + valueFeatures);
        NDArray query =
                manager.randomNormal(new Shape(batches, alternatives, queryTokens, queryWidth));
        NDArray packedKeyValue =
                manager.randomNormal(
                        new Shape(batches, alternatives, groups, keyTokens, packedWidth));
        NDArray mask = manager.ones(new Shape(batches, alternatives, groups, keyTokens));
        mask.set(new ai.djl.ndarray.index.NDIndex("0,0,0,-1"), 0);
        double scale = 0.37;

        NDArray expected =
                groupedPackedAttentionReference(
                        query.reshape(batches * alternatives, queryTokens, queryWidth),
                        packedKeyValue.reshape(
                                batches * alternatives, groups, keyTokens, packedWidth),
                        mask.reshape(batches * alternatives, groups, keyTokens),
                        heads,
                        scale);
        NDArray actual =
                NDArrays.groupedPackedScaledDotProductAttention(
                        query, packedKeyValue, mask, heads, scale);

        Assert.assertEquals(
                actual.getShape(),
                new Shape(batches, alternatives, groups, queryTokens, heads * valueFeatures));
        assertClose(
                actual.toFloatArray(), expected.reshape(actual.getShape()).toFloatArray(), 2e-4f);
    }

    private static NDArray groupedPackedAttentionReference(
            NDArray query, NDArray packedKeyValue, NDArray mask, int heads, double scale) {
        Shape queryShape = query.getShape();
        Shape packedShape = packedKeyValue.getShape();
        long batch = queryShape.get(0);
        long queryTokens = queryShape.get(1);
        long queryWidth = queryShape.get(2);
        long groups = packedShape.get(1);
        long keyTokens = packedShape.get(2);
        long keyFeatures = queryWidth / heads;
        long valueWidth = packedShape.get(3) - queryWidth;
        long valueFeatures = valueWidth / heads;
        NDArray queries =
                query.reshape(batch, queryTokens, heads, keyFeatures)
                        .swapAxes(1, 2)
                        .expandDims(1)
                        .broadcast(batch, groups, heads, queryTokens, keyFeatures);
        NDArray keys =
                packedKeyValue
                        .get("...,0:{}", queryWidth)
                        .reshape(batch, groups, keyTokens, heads, keyFeatures)
                        .swapAxes(2, 3);
        NDArray values =
                packedKeyValue
                        .get("...,{}:{}", queryWidth, packedShape.get(3))
                        .reshape(batch, groups, keyTokens, heads, valueFeatures)
                        .swapAxes(2, 3);
        NDArray valid =
                mask.neq(0)
                        .reshape(batch, groups, 1, 1, keyTokens)
                        .broadcast(batch, groups, heads, queryTokens, keyTokens);
        NDArray scores = queries.matMul(keys.swapAxes(3, 4)).mul(scale);
        return NDArrays.where(valid, scores, scores.zerosLike().add(-1.0e30f))
                .softmax(4)
                .matMul(values)
                .swapAxes(2, 3)
                .reshape(batch, groups, queryTokens, valueWidth);
    }

    private static void verifyRelationAttention(NDManager manager) {
        NDArray query = manager.randomNormal(new Shape(2, 3, 7, 6));
        NDArray key = manager.randomNormal(new Shape(2, 3, 5, 6));
        NDArray value = manager.randomNormal(new Shape(2, 3, 5, 4));
        NDArray relationKeys = manager.randomNormal(new Shape(1, 3, 6, 11));
        NDArray relationBias = manager.randomNormal(new Shape(1, 3, 7, 5));
        long[] ids = new long[7 * 5];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = i % 11;
        }
        NDArray relationIds = manager.create(ids, new Shape(7, 5));
        double scale = 0.23;

        NDArray expected =
                relationAttentionReference(
                        query, key, value, relationKeys, relationBias, relationIds, scale);
        NDArray actual =
                NDArrays.relationBiasedScaledDotProductAttention(
                        query, key, value, relationKeys, relationBias, relationIds, scale);
        assertClose(actual.toFloatArray(), expected.toFloatArray(), 2e-4f);
    }

    private static void verifyStructuredAttentionGradients(
            Engine engine, Device device, DataType dataType) {
        verifyRelationAttentionGradients(engine, device, dataType);
        verifyGroupedAttentionGradients(engine, device, dataType);
        verifyMappedGroupedAttentionGradients(engine, device, dataType);
    }

    private static void verifyPartialGradientRequests(
            Engine engine, Device device, DataType dataType) {
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray query = manager.randomNormal(new Shape(2, 2, 3, 4), dataType);
            NDArray key = manager.randomNormal(new Shape(2, 2, 5, 4), dataType);
            NDArray value = manager.randomNormal(new Shape(2, 2, 5, 3), dataType);
            NDArray relationKeys = manager.randomNormal(new Shape(1, 2, 4, 7), dataType);
            NDArray relationBias =
                    requiringGradient(manager.randomNormal(new Shape(1, 2, 3, 5), dataType));
            NDArray relationIds =
                    manager.create(
                            new int[] {0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0},
                            new Shape(3, 5));
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        NDArrays.relationBiasedScaledDotProductAttention(
                                query, key, value, relationKeys, relationBias, relationIds, 0.31);
                collector.backward(output.mul(output).sum());
            }

            assertFiniteNonzeroGradient(relationBias);
            Assert.assertFalse(query.hasGradient());
            Assert.assertFalse(key.hasGradient());
            Assert.assertFalse(value.hasGradient());
            Assert.assertFalse(relationKeys.hasGradient());
        }

        try (NDManager manager = engine.newBaseManager(device)) {
            int heads = 2;
            int keyFeatures = 3;
            int packedWidth = heads * (keyFeatures + 4);
            NDArray query =
                    requiringGradient(
                            manager.randomNormal(new Shape(4, heads, keyFeatures), dataType));
            NDArray shared = manager.randomNormal(new Shape(2, 5, packedWidth), dataType);
            NDArray sharedDeltas = manager.randomNormal(new Shape(4, 5, packedWidth), dataType);
            NDArray indexedDeltas = manager.randomNormal(new Shape(4, 3, packedWidth), dataType);
            NDArray indexedIds =
                    manager.create(new int[] {1, 2, 3, 2, 0, 4, 3, 4, 5, 4, 5, 1}, new Shape(4, 3));
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        NDArrays.groupedIndexedScaledDotProductAttention(
                                query, shared, sharedDeltas, indexedDeltas, indexedIds, 2, 0.27);
                collector.backward(output.mul(output).sum());
            }

            assertFiniteNonzeroGradient(query);
            Assert.assertFalse(shared.hasGradient());
            Assert.assertFalse(sharedDeltas.hasGradient());
            Assert.assertFalse(indexedDeltas.hasGradient());
        }

        verifyMappedGroupedAttentionPartialGradients(
                engine, device, dataType, true, false, false, false);
        verifyMappedGroupedAttentionPartialGradients(
                engine, device, dataType, false, true, false, false);
        verifyMappedGroupedAttentionPartialGradients(
                engine, device, dataType, false, false, true, false);
        verifyMappedGroupedAttentionPartialGradients(
                engine, device, dataType, false, false, false, true);
    }

    private static void verifyMappedGroupedAttentionPartialGradients(
            Engine engine,
            Device device,
            DataType dataType,
            boolean queryGradient,
            boolean sharedGradient,
            boolean deltaTableGradient,
            boolean indexedDeltaGradient) {
        try (NDManager manager = engine.newBaseManager(device)) {
            int queries = 4;
            int heads = 2;
            int keyFeatures = 3;
            int valueFeatures = 4;
            int sharedTokens = 5;
            int indexedTokens = 3;
            int packedWidth = heads * (keyFeatures + valueFeatures);
            NDArray queryValues =
                    manager.randomNormal(new Shape(queries, heads, keyFeatures), dataType);
            NDArray sharedValues =
                    manager.randomNormal(new Shape(2, sharedTokens, packedWidth), dataType);
            NDArray deltaTableValues = manager.randomNormal(new Shape(3, packedWidth), dataType);
            NDArray indexedDeltaValues =
                    manager.randomNormal(new Shape(queries, indexedTokens, packedWidth), dataType);
            NDArray query = queryValues.duplicate();
            NDArray shared = sharedValues.duplicate();
            NDArray deltaTable = deltaTableValues.duplicate();
            NDArray indexedDeltas = indexedDeltaValues.duplicate();
            if (queryGradient) {
                query.setRequiresGradient(true);
            }
            if (sharedGradient) {
                shared.setRequiresGradient(true);
            }
            if (deltaTableGradient) {
                deltaTable.setRequiresGradient(true);
            }
            if (indexedDeltaGradient) {
                indexedDeltas.setRequiresGradient(true);
            }
            NDArray groupIndices = manager.create(new int[] {1, 0, 1, 0});
            NDArray deltaIndices =
                    manager.create(
                            new int[] {
                                0, 1, 2, 0, 1,
                                1, 2, 0, 1, 2,
                                2, 0, 1, 2, 0,
                                0, 1, 2, 0, 1
                            },
                            new Shape(queries, sharedTokens));
            NDArray indexedIds =
                    manager.create(
                            new int[] {1, 0, 3, 2, 4, 0, 5, 1, 2, 0, 4, 3},
                            new Shape(queries, indexedTokens));
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        NDArrays.mappedGroupedIndexedScaledDotProductAttention(
                                query,
                                shared,
                                groupIndices,
                                deltaTable,
                                deltaIndices,
                                indexedDeltas,
                                indexedIds,
                                0.27);
                collector.backward(output.mul(output).sum());
            }

            NDArray referenceQuery = queryValues.duplicate();
            NDArray referenceShared = sharedValues.duplicate();
            NDArray referenceDeltaTable = deltaTableValues.duplicate();
            NDArray referenceIndexedDeltas = indexedDeltaValues.duplicate();
            referenceQuery.setRequiresGradient(queryGradient);
            referenceShared.setRequiresGradient(sharedGradient);
            referenceDeltaTable.setRequiresGradient(deltaTableGradient);
            referenceIndexedDeltas.setRequiresGradient(indexedDeltaGradient);
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        mappedGroupedAttentionReference(
                                referenceQuery,
                                referenceShared,
                                groupIndices,
                                referenceDeltaTable,
                                deltaIndices,
                                referenceIndexedDeltas,
                                indexedIds,
                                0.27);
                collector.backward(output.mul(output).sum());
            }

            float tolerance = gradientTolerance(dataType);

            if (queryGradient) {
                assertFiniteNonzeroGradient(query);
                assertGradientClose(query, referenceQuery, tolerance);
            } else {
                Assert.assertFalse(query.hasGradient());
                Assert.assertFalse(referenceQuery.hasGradient());
            }
            if (sharedGradient) {
                assertFiniteNonzeroGradient(shared);
                assertGradientClose(shared, referenceShared, tolerance);
            } else {
                Assert.assertFalse(shared.hasGradient());
                Assert.assertFalse(referenceShared.hasGradient());
            }
            if (deltaTableGradient) {
                assertFiniteNonzeroGradient(deltaTable);
                assertGradientClose(deltaTable, referenceDeltaTable, tolerance);
            } else {
                Assert.assertFalse(deltaTable.hasGradient());
                Assert.assertFalse(referenceDeltaTable.hasGradient());
            }
            if (indexedDeltaGradient) {
                assertFiniteNonzeroGradient(indexedDeltas);
                assertGradientClose(indexedDeltas, referenceIndexedDeltas, tolerance);
            } else {
                Assert.assertFalse(indexedDeltas.hasGradient());
                Assert.assertFalse(referenceIndexedDeltas.hasGradient());
            }
        }
    }

    private static void verifyLargeRelationBiasGradient(Engine engine) {
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            int batch = 64;
            int heads = 4;
            int tokens = 34;
            int features = 16;
            int relations = 15;
            NDArray query =
                    manager.randomNormal(
                            new Shape(batch, heads, tokens, features), DataType.BFLOAT16);
            NDArray key =
                    manager.randomNormal(
                            new Shape(batch, heads, tokens, features), DataType.BFLOAT16);
            NDArray value =
                    manager.randomNormal(
                            new Shape(batch, heads, tokens, features), DataType.BFLOAT16);
            NDArray relationKeys =
                    manager.randomNormal(
                            new Shape(1, heads, features, relations), DataType.BFLOAT16);
            NDArray relationBias =
                    requiringGradient(
                            manager.randomNormal(
                                    new Shape(1, heads, tokens, tokens), DataType.BFLOAT16));
            int[] storedIds = new int[tokens * tokens];
            for (int index = 0; index < storedIds.length; ++index) {
                storedIds[index] = index % relations;
            }
            NDArray relationIds = manager.create(storedIds, new Shape(tokens, tokens));
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        NDArrays.relationBiasedScaledDotProductAttention(
                                query,
                                key,
                                value,
                                relationKeys,
                                relationBias,
                                relationIds,
                                1.0 / Math.sqrt(features));
                collector.backward(output.sum());
            }

            assertFiniteNonzeroGradient(relationBias);
        }
    }

    @Test
    public void groupedPackedAttentionAllInvalidRowsMaskScoreGradients() {
        Engine engine = Engine.getInstance();
        verifyAllInvalidGroupedPackedAttentionGradients(engine, Device.cpu(), DataType.FLOAT32);
        if (engine.getGpuCount() > 0) {
            verifyAllInvalidGroupedPackedAttentionGradients(
                    engine, Device.gpu(), DataType.BFLOAT16);
        }
    }

    @Test
    public void groupedPackedAttentionMaskedProbabilityContributesToSoftmaxJacobian() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            NDArray query = requiringGradient(manager.create(new float[] {1f}, new Shape(1, 1, 1)));
            NDArray packed =
                    requiringGradient(
                            manager.create(
                                    new float[] {-1.0e30f, 1f, 0f, 3f}, new Shape(1, 1, 2, 2)));
            NDArray mask = manager.create(new int[] {1, 0}, new Shape(1, 1, 2));
            try (GradientCollector collector = engine.newGradientCollector()) {
                collector.backward(
                        NDArrays.groupedPackedScaledDotProductAttention(query, packed, mask, 1, 1.0)
                                .sum());
            }

            try (NDArray queryGradient = query.getGradient();
                    NDArray packedGradient = packed.getGradient()) {
                Assert.assertTrue(queryGradient.toFloatArray()[0] > 4.0e29f);
                assertClose(
                        packedGradient.toFloatArray(), new float[] {-0.5f, 0.5f, 0f, 0.5f}, 1e-6f);
            }
        }
    }

    private static void verifyGroupedPackedAttentionGradients(
            Engine engine,
            Device device,
            DataType dataType,
            boolean queryGradient,
            boolean packedGradient) {
        verifyGroupedPackedAttentionGradients(
                engine, device, dataType, DataType.FLOAT32, queryGradient, packedGradient);
    }

    private static void verifyGroupedPackedAttentionGradients(
            Engine engine,
            Device device,
            DataType dataType,
            DataType maskType,
            boolean queryGradient,
            boolean packedGradient) {
        verifyGroupedPackedAttentionGradients(
                engine,
                device,
                dataType,
                maskType,
                queryGradient,
                packedGradient,
                2,
                7,
                3,
                5,
                2,
                4,
                3);
    }

    private static void verifyGroupedPackedAttentionGradients(
            Engine engine,
            Device device,
            DataType dataType,
            DataType maskType,
            boolean queryGradient,
            boolean packedGradient,
            int batch,
            int queryTokens,
            int groups,
            int keyTokens,
            int heads,
            int keyFeatures,
            int valueFeatures) {
        try (NDManager manager = engine.newBaseManager(device)) {
            int queryWidth = heads * keyFeatures;
            int packedWidth = queryWidth + heads * valueFeatures;
            NDArray queryValues =
                    manager.randomNormal(new Shape(batch, queryTokens, queryWidth), dataType);
            NDArray packedValues =
                    manager.randomNormal(
                            new Shape(batch, groups, keyTokens, packedWidth), dataType);
            NDArray mask = manager.ones(new Shape(batch, groups, keyTokens), maskType);
            mask.set(new ai.djl.ndarray.index.NDIndex("..., -1"), 0);
            NDArray outputGradient =
                    manager.randomNormal(
                            new Shape(batch, groups, queryTokens, heads * valueFeatures), dataType);

            NDArray query = queryValues.duplicate();
            NDArray packed = packedValues.duplicate();
            if (queryGradient) {
                query.setRequiresGradient(true);
            }
            if (packedGradient) {
                packed.setRequiresGradient(true);
            }
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        NDArrays.groupedPackedScaledDotProductAttention(
                                query, packed, mask, heads, 0.5);
                collector.backward(output.mul(outputGradient).sum());
            }

            NDArray referenceQuery = queryValues.duplicate();
            NDArray referencePacked = packedValues.duplicate();
            if (queryGradient) {
                referenceQuery.setRequiresGradient(true);
            }
            if (packedGradient) {
                referencePacked.setRequiresGradient(true);
            }
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        groupedPackedAttentionReference(
                                referenceQuery, referencePacked, mask, heads, 0.5);
                collector.backward(output.mul(outputGradient).sum());
            }

            float tolerance = gradientTolerance(dataType);
            if (queryGradient) {
                assertFiniteNonzeroGradient(query);
                float maximumDifference = assertGradientClose(query, referenceQuery, tolerance);
                System.out.printf(
                        "GROUPED_PACKED_ATTENTION_GRADIENT tensor=query dtype=%s mask=%s"
                                + " maxAbs=%s%n",
                        dataType, maskType, maximumDifference);
            } else {
                Assert.assertFalse(query.hasGradient());
            }
            if (packedGradient) {
                assertFiniteNonzeroGradient(packed);
                float maximumDifference = assertGradientClose(packed, referencePacked, tolerance);
                System.out.printf(
                        "GROUPED_PACKED_ATTENTION_GRADIENT tensor=packed dtype=%s mask=%s"
                                + " maxAbs=%s%n",
                        dataType, maskType, maximumDifference);
                try (NDArray packedGradientValues = packed.getGradient();
                        NDArray maskedTokenGradient = packedGradientValues.get("..., -1, :");
                        NDArray maskedTokenGradientValues =
                                maskedTokenGradient.toType(DataType.FLOAT32, false)) {
                    assertAllZero(maskedTokenGradientValues.toFloatArray());
                }
            } else {
                Assert.assertFalse(packed.hasGradient());
            }
        }
    }

    private static void verifyAllInvalidGroupedPackedAttentionGradients(
            Engine engine, Device device, DataType dataType) {
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray query = requiringGradient(manager.ones(new Shape(1, 1, 2), dataType));
            NDArray packed = requiringGradient(manager.ones(new Shape(1, 1, 2, 4), dataType));
            NDArray mask = manager.zeros(new Shape(1, 1, 2), DataType.INT32);
            try (GradientCollector collector = engine.newGradientCollector()) {
                collector.backward(
                        NDArrays.groupedPackedScaledDotProductAttention(query, packed, mask, 1, 1.0)
                                .sum());
            }

            try (NDArray queryGradient = query.getGradient();
                    NDArray packedGradient = packed.getGradient();
                    NDArray queryGradientValues = queryGradient.toType(DataType.FLOAT32, false);
                    NDArray packedGradientValues = packedGradient.toType(DataType.FLOAT32, false)) {
                assertClose(queryGradientValues.toFloatArray(), new float[] {0f, 0f}, 0f);
                assertClose(
                        packedGradientValues.toFloatArray(),
                        new float[] {0f, 0f, 0.5f, 0.5f, 0f, 0f, 0.5f, 0.5f},
                        0f);
            }
        }
    }

    private static void verifyRelationAttentionGradients(
            Engine engine, Device device, DataType dataType) {
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray queryValues = manager.randomNormal(new Shape(2, 2, 3, 4), dataType);
            NDArray keyValues = manager.randomNormal(new Shape(2, 2, 5, 4), dataType);
            NDArray valueValues = manager.randomNormal(new Shape(2, 2, 5, 3), dataType);
            NDArray relationKeyValues = manager.randomNormal(new Shape(1, 2, 4, 7), dataType);
            NDArray relationBiasValues = manager.randomNormal(new Shape(1, 2, 3, 5), dataType);
            NDArray relationIds =
                    manager.create(
                            new int[] {0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0},
                            new Shape(3, 5));

            NDArray query = requiringGradient(queryValues.duplicate());
            NDArray key = requiringGradient(keyValues.duplicate());
            NDArray value = requiringGradient(valueValues.duplicate());
            NDArray relationKeys = requiringGradient(relationKeyValues.duplicate());
            NDArray relationBias = requiringGradient(relationBiasValues.duplicate());
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        NDArrays.relationBiasedScaledDotProductAttention(
                                query, key, value, relationKeys, relationBias, relationIds, 0.31);
                collector.backward(output.mul(output).sum());
            }

            NDArray referenceQuery = requiringGradient(queryValues.duplicate());
            NDArray referenceKey = requiringGradient(keyValues.duplicate());
            NDArray referenceValue = requiringGradient(valueValues.duplicate());
            NDArray referenceRelationKeys = requiringGradient(relationKeyValues.duplicate());
            NDArray referenceRelationBias = requiringGradient(relationBiasValues.duplicate());
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        relationAttentionReference(
                                referenceQuery,
                                referenceKey,
                                referenceValue,
                                referenceRelationKeys,
                                referenceRelationBias,
                                relationIds,
                                0.31);
                collector.backward(output.mul(output).sum());
            }

            assertFiniteNonzeroGradient(query, key, value, relationKeys, relationBias);
            float tolerance = gradientTolerance(dataType);
            assertGradientClose(query, referenceQuery, tolerance);
            assertGradientClose(key, referenceKey, tolerance);
            assertGradientClose(value, referenceValue, tolerance);
            assertGradientClose(relationKeys, referenceRelationKeys, tolerance);
            assertGradientClose(relationBias, referenceRelationBias, tolerance);
        }
    }

    private static void verifyGroupedAttentionGradients(
            Engine engine, Device device, DataType dataType) {
        try (NDManager manager = engine.newBaseManager(device)) {
            int heads = 2;
            int keyFeatures = 3;
            int packedWidth = heads * (keyFeatures + 4);
            NDArray queryValues = manager.randomNormal(new Shape(4, heads, keyFeatures), dataType);
            NDArray sharedValues = manager.randomNormal(new Shape(2, 5, packedWidth), dataType);
            NDArray sharedDeltaValues =
                    manager.randomNormal(new Shape(4, 5, packedWidth), dataType);
            NDArray indexedDeltaValues =
                    manager.randomNormal(new Shape(4, 3, packedWidth), dataType);
            NDArray indexedIds =
                    manager.create(new int[] {1, 2, 3, 2, 0, 4, 3, 4, 5, 4, 5, 1}, new Shape(4, 3));

            NDArray query = requiringGradient(queryValues.duplicate());
            NDArray shared = requiringGradient(sharedValues.duplicate());
            NDArray sharedDeltas = requiringGradient(sharedDeltaValues.duplicate());
            NDArray indexedDeltas = requiringGradient(indexedDeltaValues.duplicate());
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        NDArrays.groupedIndexedScaledDotProductAttention(
                                query, shared, sharedDeltas, indexedDeltas, indexedIds, 2, 0.27);
                collector.backward(output.mul(output).sum());
            }

            NDArray referenceQuery = requiringGradient(queryValues.duplicate());
            NDArray referenceShared = requiringGradient(sharedValues.duplicate());
            NDArray referenceSharedDeltas = requiringGradient(sharedDeltaValues.duplicate());
            NDArray referenceIndexedDeltas = requiringGradient(indexedDeltaValues.duplicate());
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        groupedAttentionReference(
                                referenceQuery,
                                referenceShared,
                                referenceSharedDeltas,
                                referenceIndexedDeltas,
                                indexedIds,
                                2,
                                0.27);
                collector.backward(output.mul(output).sum());
            }

            assertFiniteNonzeroGradient(query, shared, sharedDeltas, indexedDeltas);
            float tolerance = gradientTolerance(dataType);
            assertGradientClose(query, referenceQuery, tolerance);
            assertGradientClose(shared, referenceShared, tolerance);
            assertGradientClose(sharedDeltas, referenceSharedDeltas, tolerance);
            assertGradientClose(indexedDeltas, referenceIndexedDeltas, tolerance);
        }
    }

    private static void verifyMappedGroupedAttentionGradients(
            Engine engine, Device device, DataType dataType) {
        try (NDManager manager = engine.newBaseManager(device)) {
            int queries = 5;
            int heads = 2;
            int keyFeatures = 3;
            int valueFeatures = 4;
            int sharedTokens = 5;
            int indexedTokens = 3;
            int packedWidth = heads * (keyFeatures + valueFeatures);
            NDArray queryValues =
                    manager.randomNormal(new Shape(queries, heads, keyFeatures), dataType);
            NDArray sharedValues =
                    manager.randomNormal(new Shape(3, sharedTokens, packedWidth), dataType);
            NDArray deltaTableValues = manager.randomNormal(new Shape(7, packedWidth), dataType);
            NDArray indexedDeltaValues =
                    manager.randomNormal(new Shape(queries, indexedTokens, packedWidth), dataType);
            NDArray outputGradientValues =
                    manager.randomNormal(new Shape(queries, heads, valueFeatures), dataType);
            NDArray groupIndices = manager.create(new int[] {2, 0, 2, 1, 0});
            NDArray deltaIndices =
                    manager.create(
                            new int[] {
                                0, 1, 2, 3, 4,
                                1, 2, 3, 4, 5,
                                2, 3, 4, 5, 6,
                                3, 4, 5, 6, 0,
                                4, 5, 6, 0, 1
                            },
                            new Shape(queries, sharedTokens));
            NDArray indexedIds =
                    manager.create(
                            new int[] {1, 2, 3, 2, 0, 4, 3, 4, 5, 4, 5, 1, 5, 1, 0},
                            new Shape(queries, indexedTokens));

            NDArray query = requiringGradient(queryValues.duplicate());
            NDArray shared = requiringGradient(sharedValues.duplicate());
            NDArray deltaTable = requiringGradient(deltaTableValues.duplicate());
            NDArray indexedDeltas = requiringGradient(indexedDeltaValues.duplicate());
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        NDArrays.mappedGroupedIndexedScaledDotProductAttention(
                                query,
                                shared,
                                groupIndices,
                                deltaTable,
                                deltaIndices,
                                indexedDeltas,
                                indexedIds,
                                0.27);
                collector.backward(output.mul(outputGradientValues).sum());
            }

            NDArray referenceQuery = requiringGradient(queryValues.duplicate());
            NDArray referenceShared = requiringGradient(sharedValues.duplicate());
            NDArray referenceDeltaTable = requiringGradient(deltaTableValues.duplicate());
            NDArray referenceIndexedDeltas = requiringGradient(indexedDeltaValues.duplicate());
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        mappedGroupedAttentionReference(
                                referenceQuery,
                                referenceShared,
                                groupIndices,
                                referenceDeltaTable,
                                deltaIndices,
                                referenceIndexedDeltas,
                                indexedIds,
                                0.27);
                collector.backward(output.mul(outputGradientValues).sum());
            }

            assertFiniteNonzeroGradient(query, shared, deltaTable, indexedDeltas);
            float tolerance = gradientTolerance(dataType);
            assertGradientClose(query, referenceQuery, tolerance);
            assertGradientClose(shared, referenceShared, tolerance);
            assertGradientClose(deltaTable, referenceDeltaTable, tolerance);
            assertGradientClose(indexedDeltas, referenceIndexedDeltas, tolerance);
        }
    }

    private static void verifyMappedGroupedAttentionProductionGradients(
            Engine engine, DataType dataType) {
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            int queries = 7;
            int groups = 3;
            int heads = 4;
            int keyFeatures = 8;
            int valueFeatures = 16;
            int sharedTokens = 34;
            int indexedTokens = 13;
            int deltaRows = 17;
            int packedWidth = heads * (keyFeatures + valueFeatures);
            NDArray queryValues =
                    manager.randomNormal(new Shape(queries, heads, keyFeatures), dataType);
            NDArray sharedValues =
                    manager.randomNormal(
                            new Shape(groups, sharedTokens, packedWidth), dataType);
            NDArray deltaTableValues =
                    manager.randomNormal(new Shape(deltaRows, packedWidth), dataType);
            NDArray indexedDeltaValues =
                    manager.randomNormal(
                            new Shape(queries, indexedTokens, packedWidth), dataType);
            NDArray outputGradientValues =
                    manager.randomNormal(
                            new Shape(queries, heads, valueFeatures), dataType);
            int[] groupIds = new int[queries];
            int[] deltaIds = new int[queries * sharedTokens];
            int[] indexedIds = new int[queries * indexedTokens];
            for (int query = 0; query < queries; query++) {
                groupIds[query] = (query * 2 + 1) % groups;
            }
            for (int index = 0; index < deltaIds.length; index++) {
                deltaIds[index] = (index * 5 + 3) % deltaRows;
            }
            for (int index = 0; index < indexedIds.length; index++) {
                indexedIds[index] =
                        index % 5 == 0 ? 0 : (index * 7 + 3) % sharedTokens + 1;
            }
            NDArray groupIndices = manager.create(groupIds);
            NDArray deltaIndices =
                    manager.create(deltaIds, new Shape(queries, sharedTokens));
            NDArray storedIndexedIds =
                    manager.create(indexedIds, new Shape(queries, indexedTokens));

            NDArray query = requiringGradient(queryValues.duplicate());
            NDArray shared = requiringGradient(sharedValues.duplicate());
            NDArray deltaTable = requiringGradient(deltaTableValues.duplicate());
            NDArray indexedDeltas = requiringGradient(indexedDeltaValues.duplicate());
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        NDArrays.mappedGroupedIndexedScaledDotProductAttention(
                                query,
                                shared,
                                groupIndices,
                                deltaTable,
                                deltaIndices,
                                indexedDeltas,
                                storedIndexedIds,
                                1.0 / Math.sqrt(keyFeatures));
                collector.backward(output.mul(outputGradientValues).sum());
            }

            NDArray referenceQuery = requiringGradient(queryValues.duplicate());
            NDArray referenceShared = requiringGradient(sharedValues.duplicate());
            NDArray referenceDeltaTable = requiringGradient(deltaTableValues.duplicate());
            NDArray referenceIndexedDeltas = requiringGradient(indexedDeltaValues.duplicate());
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        mappedGroupedAttentionReference(
                                referenceQuery,
                                referenceShared,
                                groupIndices,
                                referenceDeltaTable,
                                deltaIndices,
                                referenceIndexedDeltas,
                                storedIndexedIds,
                                1.0 / Math.sqrt(keyFeatures));
                collector.backward(output.mul(outputGradientValues).sum());
            }

            try (NDArray indexedGradient = indexedDeltas.getGradient();
                    NDArray values = indexedGradient.toType(DataType.FLOAT32, false)) {
                float[] gradients = values.toFloatArray();
                for (int queryIndex = 0; queryIndex < queries; queryIndex++) {
                    for (int token = 0; token < indexedTokens; token++) {
                        if (indexedIds[queryIndex * indexedTokens + token] != 0) {
                            continue;
                        }
                        int offset =
                                (queryIndex * indexedTokens + token) * packedWidth;
                        for (int feature = 0; feature < packedWidth; feature++) {
                            Assert.assertEquals(
                                    gradients[offset + feature],
                                    0f,
                                    "padded indexed gradient at query="
                                            + queryIndex
                                            + ", token="
                                            + token
                                            + ", feature="
                                            + feature);
                        }
                    }
                }
            }

            assertFiniteNonzeroGradient(query, shared, deltaTable, indexedDeltas);
            float tolerance = gradientTolerance(dataType);
            assertGradientClose(query, referenceQuery, tolerance);
            assertGradientClose(shared, referenceShared, tolerance);
            assertGradientClose(deltaTable, referenceDeltaTable, tolerance);
            assertGradientClose(indexedDeltas, referenceIndexedDeltas, tolerance);
        }
    }

    private static float gradientTolerance(DataType dataType) {
        switch (dataType) {
            case FLOAT32:
                return 2e-5f;
            case FLOAT16:
                return 2e-3f;
            case BFLOAT16:
                return 2e-2f;
            default:
                throw new IllegalArgumentException("Unsupported gradient data type: " + dataType);
        }
    }

    private static void runStructuredStep(
            NDManager modelManager,
            NDArray relationKeys,
            NDArray relationBias,
            NDArray relationIds,
            NDArray sharedKeyValues,
            NDArray indexedSharedIds) {
        try (NDManager batchManager = modelManager.newSubManager()) {
            NDArray query = batchManager.randomNormal(new Shape(1, 1, 2, 2));
            NDArray key = batchManager.randomNormal(new Shape(1, 1, 3, 2));
            NDArray value = batchManager.randomNormal(new Shape(1, 1, 3, 2));
            NDArray groupedQuery = batchManager.randomNormal(new Shape(2, 1, 2));
            NDArray sharedDeltas = batchManager.randomNormal(new Shape(2, 3, 4));
            NDArray indexedDeltas = batchManager.randomNormal(new Shape(2, 1, 4));

            NDArray relationOutput =
                    NDArrays.relationBiasedScaledDotProductAttention(
                            query,
                            key,
                            value,
                            relationKeys,
                            relationBias,
                            relationIds,
                            1.0 / Math.sqrt(2));
            Assert.assertSame(relationOutput.getManager(), batchManager);
            NDArray groupedOutput =
                    NDArrays.groupedIndexedScaledDotProductAttention(
                            groupedQuery,
                            sharedKeyValues,
                            sharedDeltas,
                            indexedDeltas,
                            indexedSharedIds,
                            2,
                            1.0 / Math.sqrt(2));
            Assert.assertSame(groupedOutput.getManager(), batchManager);
        }
    }

    private static void verifyGroupedAttention(NDManager manager) {
        verifyGroupedAttentionShape(manager, 2, 3, 3, 5, 9, 7, 4, false);
        verifyGroupedAttentionShape(manager, 3, 5, 4, 8, 16, 34, 13, false);
        verifyGroupedAttentionShape(manager, 1, 7, 5, 33, 37, 65, 17, true);
        verifyGroupedAttentionShape(manager, 1, 3, 2, 7, 11, 9, 5, false, DataType.FLOAT64, 2e-5f);
        verifyStridedGroupedAttention(manager);
    }

    private static void verifyMappedGroupedAttention(NDManager manager) {
        int queries = 7;
        int heads = 3;
        int keyFeatures = 5;
        int valueFeatures = 9;
        int sharedTokens = 11;
        int indexedTokens = 4;
        int packedWidth = heads * (keyFeatures + valueFeatures);
        NDArray query = manager.randomNormal(new Shape(queries, heads, keyFeatures));
        NDArray shared = manager.randomNormal(new Shape(4, sharedTokens, packedWidth));
        NDArray groupIndices = manager.create(new int[] {3, 0, 3, 1, 2, 0, 2});
        NDArray deltaTable = manager.randomNormal(new Shape(13, packedWidth));
        int[] deltaIds = new int[queries * sharedTokens];
        int[] indexedIds = new int[queries * indexedTokens];
        for (int index = 0; index < deltaIds.length; ++index) {
            deltaIds[index] = index % 13;
        }
        for (int index = 0; index < indexedIds.length; ++index) {
            indexedIds[index] = index % (sharedTokens + 1);
        }
        NDArray deltaIndices = manager.create(deltaIds, new Shape(queries, sharedTokens));
        NDArray indexedDeltas =
                manager.randomNormal(new Shape(queries, indexedTokens, packedWidth));
        NDArray storedIndexedIds = manager.create(indexedIds, new Shape(queries, indexedTokens));
        double scale = 0.39;

        NDArray expected =
                mappedGroupedAttentionReference(
                        query,
                        shared,
                        groupIndices,
                        deltaTable,
                        deltaIndices,
                        indexedDeltas,
                        storedIndexedIds,
                        scale);
        NDArray actual =
                NDArrays.mappedGroupedIndexedScaledDotProductAttention(
                        query,
                        shared,
                        groupIndices,
                        deltaTable,
                        deltaIndices,
                        indexedDeltas,
                        storedIndexedIds,
                        scale);

        assertClose(actual.toFloatArray(), expected.toFloatArray(), 2e-4f);
    }

    private static AttentionComparison verifyMappedGroupedAttentionForward(
            NDManager manager, DataType dataType, MappedAttentionPath path) {
        int queries = 7;
        int heads = 4;
        int keyFeatures = 8;
        int valueFeatures = 16;
        int sharedTokens = 34;
        int indexedTokens = 6;
        int packedWidth = heads * (keyFeatures + valueFeatures);
        NDArray query = manager.randomNormal(new Shape(queries, heads, keyFeatures), dataType);
        NDArray shared = manager.randomNormal(new Shape(4, sharedTokens, packedWidth), dataType);
        NDArray deltaTable = manager.randomNormal(new Shape(13, packedWidth), dataType);
        NDArray indexedDeltas =
                manager.randomNormal(new Shape(queries, indexedTokens, packedWidth), dataType);
        int[] groupIds = {3, 0, 3, 1, 2, 0, 2};
        int[] deltaIds = new int[queries * sharedTokens];
        int[] indexedIds = new int[queries * indexedTokens];
        for (int index = 0; index < deltaIds.length; ++index) {
            deltaIds[index] = (index * 5 + 3) % 13;
        }
        for (int index = 0; index < indexedIds.length; ++index) {
            indexedIds[index] = index % 5 == 0 ? 0 : (index * 7) % sharedTokens + 1;
        }
        NDArray groups = manager.create(groupIds);
        NDArray deltas = manager.create(deltaIds, new Shape(queries, sharedTokens));
        NDArray storedIndexedIds = manager.create(indexedIds, new Shape(queries, indexedTokens));
        double scale = 1.0 / Math.sqrt(keyFeatures);

        NDArray expected =
                mappedGroupedAttentionReference(
                        query,
                        shared,
                        groups,
                        deltaTable,
                        deltas,
                        indexedDeltas,
                        storedIndexedIds,
                        scale);
        boolean useLongIndices = path == MappedAttentionPath.INT64_EAGER_FALLBACK;
        NDArray actual =
                NDArrays.mappedGroupedIndexedScaledDotProductAttention(
                        query,
                        shared,
                        useLongIndices ? groups.toType(DataType.INT64, false) : groups,
                        deltaTable,
                        useLongIndices ? deltas.toType(DataType.INT64, false) : deltas,
                        indexedDeltas,
                        useLongIndices
                                ? storedIndexedIds.toType(DataType.INT64, false)
                                : storedIndexedIds,
                        scale);
        return assertMappedAttentionClose(
                actual.toType(DataType.FLOAT32, false).toFloatArray(),
                expected.toType(DataType.FLOAT32, false).toFloatArray(),
                dataType,
                path);
    }

    private static void verifyGroupedAttentionShape(
            NDManager manager,
            int groups,
            int queriesPerGroup,
            int heads,
            int keyFeatures,
            int valueFeatures,
            int sharedTokens,
            int indexedTokens,
            boolean longIndices) {
        verifyGroupedAttentionShape(
                manager,
                groups,
                queriesPerGroup,
                heads,
                keyFeatures,
                valueFeatures,
                sharedTokens,
                indexedTokens,
                longIndices,
                DataType.FLOAT32,
                2e-4f);
    }

    private static void verifyGroupedAttentionShape(
            NDManager manager,
            int groups,
            int queriesPerGroup,
            int heads,
            int keyFeatures,
            int valueFeatures,
            int sharedTokens,
            int indexedTokens,
            boolean longIndices,
            DataType dataType,
            float tolerance) {
        int queries = groups * queriesPerGroup;
        int packedWidth = heads * (keyFeatures + valueFeatures);
        NDArray query =
                manager.randomNormal(new Shape(queries, heads, keyFeatures))
                        .toType(dataType, false);
        NDArray shared =
                manager.randomNormal(new Shape(groups, sharedTokens, packedWidth))
                        .toType(dataType, false);
        NDArray sharedDeltas =
                manager.randomNormal(new Shape(queries, sharedTokens, packedWidth))
                        .toType(dataType, false);
        NDArray indexedDeltas =
                manager.randomNormal(new Shape(queries, indexedTokens, packedWidth))
                        .toType(dataType, false);
        int[] ids = new int[queries * indexedTokens];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = i % (sharedTokens + 1);
        }
        NDArray indexedIds =
                longIndices
                        ? manager.create(
                                java.util.Arrays.stream(ids).asLongStream().toArray(),
                                new Shape(queries, indexedTokens))
                        : manager.create(ids, new Shape(queries, indexedTokens));
        double scale = 0.41;

        NDArray expected =
                groupedAttentionReference(
                        query,
                        shared,
                        sharedDeltas,
                        indexedDeltas,
                        indexedIds,
                        queriesPerGroup,
                        scale);
        NDArray actual =
                NDArrays.groupedIndexedScaledDotProductAttention(
                        query,
                        shared,
                        sharedDeltas,
                        indexedDeltas,
                        indexedIds,
                        queriesPerGroup,
                        scale);
        assertClose(
                actual.toType(DataType.FLOAT32, false).toFloatArray(),
                expected.toType(DataType.FLOAT32, false).toFloatArray(),
                tolerance);
    }

    private static void verifyStridedGroupedAttention(NDManager manager) {
        int queries = 6;
        int heads = 3;
        int keyFeatures = 5;
        int sharedTokens = 7;
        int indexedTokens = 4;
        int packedWidth = 42;
        NDArray query = manager.randomNormal(new Shape(queries, keyFeatures, heads)).swapAxes(1, 2);
        NDArray shared =
                manager.randomNormal(new Shape(2, packedWidth, sharedTokens)).swapAxes(1, 2);
        NDArray sharedDeltas =
                manager.randomNormal(new Shape(queries, packedWidth, sharedTokens)).swapAxes(1, 2);
        NDArray indexedDeltas =
                manager.randomNormal(new Shape(queries, packedWidth, indexedTokens)).swapAxes(1, 2);
        int[] ids = new int[queries * indexedTokens];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = i % (sharedTokens + 1);
        }
        NDArray indexedIds = manager.create(ids, new Shape(indexedTokens, queries)).transpose();
        double scale = 0.37;

        NDArray expected =
                groupedAttentionReference(
                        query, shared, sharedDeltas, indexedDeltas, indexedIds, 3, scale);
        NDArray actual =
                NDArrays.groupedIndexedScaledDotProductAttention(
                        query, shared, sharedDeltas, indexedDeltas, indexedIds, 3, scale);
        assertClose(actual.toFloatArray(), expected.toFloatArray(), 2e-4f);
    }

    private static NDArray relationAttentionReference(
            NDArray query,
            NDArray key,
            NDArray value,
            NDArray relationKeys,
            NDArray relationBias,
            NDArray relationIds,
            double scale) {
        Shape queryShape = query.getShape();
        long batch = queryShape.get(0);
        long heads = queryShape.get(1);
        long queryTokens = queryShape.get(2);
        long keyTokens = key.getShape().get(2);
        NDArray storedIds = relationIds.toType(DataType.INT64, false);
        NDArray relationIndices =
                storedIds.getShape().dimension() == 2
                        ? storedIds
                                .reshape(1, 1, queryTokens, keyTokens)
                                .broadcast(batch, heads, queryTokens, keyTokens)
                        : storedIds
                                .reshape(batch, 1, queryTokens, keyTokens)
                                .broadcast(batch, heads, queryTokens, keyTokens);
        NDArray relationScores = query.matMul(relationKeys).gather(relationIndices, 3);
        return query.matMul(key.swapAxes(2, 3))
                .add(relationScores)
                .mul(scale)
                .add(relationBias)
                .softmax(3)
                .matMul(value);
    }

    private static NDArray groupedAttentionReference(
            NDArray query,
            NDArray sharedKeyValues,
            NDArray sharedDeltas,
            NDArray indexedDeltas,
            NDArray indexedSharedIds,
            long queriesPerGroup,
            double scale) {
        Shape queryShape = query.getShape();
        long queryCount = queryShape.get(0);
        long heads = queryShape.get(1);
        long keyFeatures = queryShape.get(2);
        long groupCount = sharedKeyValues.getShape().get(0);
        long sharedTokens = sharedKeyValues.getShape().get(1);
        long packedWidth = sharedKeyValues.getShape().get(2);
        long indexedTokens = indexedDeltas.getShape().get(1);
        long keyWidth = heads * keyFeatures;
        long valueFeatures = (packedWidth - keyWidth) / heads;

        NDArray querySharedKeyValues =
                sharedKeyValues
                        .reshape(groupCount, 1, sharedTokens, packedWidth)
                        .broadcast(groupCount, queriesPerGroup, sharedTokens, packedWidth)
                        .reshape(queryCount, sharedTokens, packedWidth);
        NDArray storedIds = indexedSharedIds.toType(DataType.INT64, false);
        NDArray gatherIndices =
                storedIds
                        .sub(1)
                        .maximum(0)
                        .expandDims(2)
                        .broadcast(queryCount, indexedTokens, packedWidth);
        NDArray indexedSharedKeyValues = querySharedKeyValues.gather(gatherIndices, 1);

        NDArray sharedKeys =
                querySharedKeyValues
                        .get("...,0:{}", keyWidth)
                        .reshape(queryCount, sharedTokens, heads, keyFeatures)
                        .swapAxes(1, 2);
        NDArray sharedDeltaKeys =
                sharedDeltas
                        .get("...,0:{}", keyWidth)
                        .reshape(queryCount, sharedTokens, heads, keyFeatures)
                        .swapAxes(1, 2);
        NDArray indexedKeys =
                indexedSharedKeyValues
                        .get("...,0:{}", keyWidth)
                        .reshape(queryCount, indexedTokens, heads, keyFeatures)
                        .swapAxes(1, 2);
        NDArray indexedDeltaKeys =
                indexedDeltas
                        .get("...,0:{}", keyWidth)
                        .reshape(queryCount, indexedTokens, heads, keyFeatures)
                        .swapAxes(1, 2);
        NDArray keys = sharedKeys.add(sharedDeltaKeys).concat(indexedKeys.add(indexedDeltaKeys), 2);
        NDArray scores = query.expandDims(2).mul(keys).sum(new int[] {3}).mul(scale);
        NDArray indexedMask =
                storedIds
                        .neq(0)
                        .toType(scores.getDataType(), false)
                        .expandDims(1)
                        .broadcast(queryCount, heads, indexedTokens)
                        .neg()
                        .add(1)
                        .mul(-1.0e9f);
        scores =
                scores.get("...,0:{}", sharedTokens)
                        .concat(scores.get("...,{}:", sharedTokens).add(indexedMask), 2);
        NDArray weights = scores.softmax(2);

        NDArray sharedValues =
                querySharedKeyValues
                        .get("...,{}:", keyWidth)
                        .reshape(queryCount, sharedTokens, heads, valueFeatures)
                        .swapAxes(1, 2);
        NDArray sharedDeltaValues =
                sharedDeltas
                        .get("...,{}:", keyWidth)
                        .reshape(queryCount, sharedTokens, heads, valueFeatures)
                        .swapAxes(1, 2);
        NDArray indexedValues =
                indexedSharedKeyValues
                        .get("...,{}:", keyWidth)
                        .reshape(queryCount, indexedTokens, heads, valueFeatures)
                        .swapAxes(1, 2);
        NDArray indexedDeltaValues =
                indexedDeltas
                        .get("...,{}:", keyWidth)
                        .reshape(queryCount, indexedTokens, heads, valueFeatures)
                        .swapAxes(1, 2);
        NDArray values =
                sharedValues
                        .add(sharedDeltaValues)
                        .concat(indexedValues.add(indexedDeltaValues), 2);
        return weights.expandDims(3).mul(values).sum(new int[] {2});
    }

    private static NDArray mappedGroupedAttentionReference(
            NDArray query,
            NDArray sharedKeyValues,
            NDArray sharedGroupIndices,
            NDArray sharedDeltaTable,
            NDArray sharedDeltaIndices,
            NDArray indexedDeltas,
            NDArray indexedSharedIds,
            double scale) {
        long queryCount = query.getShape().get(0);
        long sharedTokens = sharedKeyValues.getShape().get(1);
        long packedWidth = sharedKeyValues.getShape().get(2);
        NDArray mappedSharedKeyValues =
                NDArrays.gatherRows(
                        sharedKeyValues, sharedGroupIndices.toType(DataType.INT64, false));
        NDArray mappedSharedDeltas =
                NDArrays.gatherRows(
                                sharedDeltaTable,
                                sharedDeltaIndices.toType(DataType.INT64, false).reshape(-1))
                        .reshape(queryCount, sharedTokens, packedWidth);
        return groupedAttentionReference(
                query,
                mappedSharedKeyValues,
                mappedSharedDeltas,
                indexedDeltas,
                indexedSharedIds,
                1,
                scale);
    }

    private static void verifyResidualLayerNorm(NDManager manager) {
        for (int width : new int[] {3, 17, 255, 257, 513}) {
            float[] residualValues = new float[2 * width];
            float[] updateValues = new float[2 * width];
            for (int i = 0; i < residualValues.length; i++) {
                residualValues[i] = (i % 19 - 9) * 0.125f;
                updateValues[i] = (i % 7 - 3) * 0.0625f;
            }
            NDArray residual = manager.create(residualValues, new Shape(2, width));
            NDArray update = manager.create(updateValues, new Shape(2, width));
            NDArray weight = manager.ones(new Shape(width));
            NDArray bias = manager.zeros(new Shape(width));

            NDArray normalized =
                    NDArrays.addToOwnedResidualAndLayerNorm(
                            residual, update, weight, bias, 1.0e-5f);

            float[] summed = new float[residualValues.length];
            float[] expected = new float[residualValues.length];
            for (int row = 0; row < 2; row++) {
                float mean = 0f;
                for (int feature = 0; feature < width; feature++) {
                    int index = row * width + feature;
                    summed[index] = residualValues[index] + updateValues[index];
                    mean += summed[index];
                }
                mean /= width;
                float variance = 0f;
                for (int feature = 0; feature < width; feature++) {
                    float centered = summed[row * width + feature] - mean;
                    variance += centered * centered;
                }
                float inverseDeviation = (float) (1.0 / Math.sqrt(variance / width + 1.0e-5));
                for (int feature = 0; feature < width; feature++) {
                    int index = row * width + feature;
                    expected[index] = (summed[index] - mean) * inverseDeviation;
                }
            }
            assertClose(residual.toFloatArray(), summed, 1e-6f);
            assertClose(normalized.toFloatArray(), expected, 3e-4f);
        }
    }

    private static void verifyResidualLayerNormDataTypes(
            NDManager manager,
            DataType residualType,
            DataType updateType,
            DataType parameterType,
            float tolerance) {
        int rowCount = 3;
        int width = 257;
        float[] residualValues = new float[rowCount * width];
        float[] updateValues = new float[rowCount * width];
        float[] weightValues = new float[width];
        float[] biasValues = new float[width];
        for (int index = 0; index < residualValues.length; index++) {
            residualValues[index] = (index % 29 - 14) * 0.03125f;
            updateValues[index] = (index % 13 - 6) * 0.015625f;
        }
        for (int feature = 0; feature < width; feature++) {
            weightValues[feature] = 0.75f + (feature % 7) * 0.03125f;
            biasValues[feature] = (feature % 5 - 2) * 0.015625f;
        }

        NDArray residual =
                manager.create(residualValues, new Shape(rowCount, width))
                        .toType(residualType, false);
        NDArray update =
                manager.create(updateValues, new Shape(rowCount, width)).toType(updateType, false);
        NDArray weight = manager.create(weightValues).toType(parameterType, false);
        NDArray bias = manager.create(biasValues).toType(parameterType, false);
        float[] roundedResidual = residual.toType(DataType.FLOAT32, false).toFloatArray();
        float[] roundedUpdate = update.toType(DataType.FLOAT32, false).toFloatArray();
        float[] roundedWeight = weight.toType(DataType.FLOAT32, false).toFloatArray();
        float[] roundedBias = bias.toType(DataType.FLOAT32, false).toFloatArray();
        float[] summed = new float[residualValues.length];
        float[] expected =
                residualLayerNormReference(
                        roundedResidual,
                        roundedUpdate,
                        roundedWeight,
                        roundedBias,
                        rowCount,
                        width,
                        1.0e-5f,
                        summed);

        NDArray normalized =
                NDArrays.addToOwnedResidualAndLayerNorm(residual, update, weight, bias, 1.0e-5f);

        assertClose(residual.toType(DataType.FLOAT32, false).toFloatArray(), summed, tolerance);
        assertClose(normalized.toType(DataType.FLOAT32, false).toFloatArray(), expected, tolerance);
    }

    private static float[] residualLayerNormReference(
            float[] residual,
            float[] update,
            float[] weight,
            float[] bias,
            int rowCount,
            int width,
            float epsilon,
            float[] summed) {
        float[] normalized = new float[residual.length];
        for (int row = 0; row < rowCount; row++) {
            int rowOffset = row * width;
            double mean = 0.0;
            for (int feature = 0; feature < width; feature++) {
                int index = rowOffset + feature;
                summed[index] = residual[index] + update[index];
                mean += summed[index];
            }
            mean /= width;
            double variance = 0.0;
            for (int feature = 0; feature < width; feature++) {
                double centered = summed[rowOffset + feature] - mean;
                variance += centered * centered;
            }
            double inverseDeviation = 1.0 / Math.sqrt(variance / width + epsilon);
            for (int feature = 0; feature < width; feature++) {
                int index = rowOffset + feature;
                normalized[index] =
                        (float)
                                ((summed[index] - mean) * inverseDeviation * weight[feature]
                                        + bias[feature]);
            }
        }
        return normalized;
    }

    private static AttentionComparison assertMappedAttentionClose(
            float[] actual, float[] expected, DataType dataType, MappedAttentionPath path) {
        Assert.assertEquals(actual.length, expected.length);
        float absoluteTolerance;
        if (dataType == DataType.FLOAT32) {
            absoluteTolerance = 3e-4f;
        } else if (dataType == DataType.FLOAT16) {
            absoluteTolerance = 6e-3f;
        } else {
            absoluteTolerance = path == MappedAttentionPath.INT32_NATIVE ? 3e-2f : 0.015625f;
        }
        float relativeTolerance =
                dataType == DataType.BFLOAT16 && path == MappedAttentionPath.INT64_EAGER_FALLBACK
                        ? 2e-2f
                        : 0f;
        int maximumBfloat16Ulps = 0;
        float maximumAbsoluteError = 0f;
        float maximumRelativeError = 0f;
        for (int index = 0; index < actual.length; ++index) {
            Assert.assertTrue(Float.isFinite(actual[index]), "non-finite actual element " + index);
            Assert.assertTrue(
                    Float.isFinite(expected[index]), "non-finite expected element " + index);
            float absoluteError = Math.abs(actual[index] - expected[index]);
            float relativeError = absoluteError / Math.max(Math.abs(expected[index]), 1e-12f);
            float allowedError = absoluteTolerance + relativeTolerance * Math.abs(expected[index]);
            Assert.assertTrue(
                    absoluteError <= allowedError,
                    "element "
                            + index
                            + " path="
                            + path
                            + " dtype="
                            + dataType
                            + " actual="
                            + actual[index]
                            + " expected="
                            + expected[index]
                            + " absoluteError="
                            + absoluteError
                            + " allowedError="
                            + allowedError);
            if (dataType == DataType.BFLOAT16
                    && path == MappedAttentionPath.INT64_EAGER_FALLBACK
                    && Math.abs(expected[index]) >= 0.25f) {
                int ulps = bfloat16UlpDistance(actual[index], expected[index]);
                Assert.assertTrue(
                        ulps <= 8,
                        "element "
                                + index
                                + " BF16 fallback differs by "
                                + ulps
                                + " ULPs; actual="
                                + actual[index]
                                + " expected="
                                + expected[index]);
                maximumBfloat16Ulps = Math.max(maximumBfloat16Ulps, ulps);
            }
            maximumAbsoluteError = Math.max(maximumAbsoluteError, absoluteError);
            maximumRelativeError = Math.max(maximumRelativeError, relativeError);
        }
        return new AttentionComparison(
                path, maximumAbsoluteError, maximumRelativeError, maximumBfloat16Ulps);
    }

    private static int bfloat16UlpDistance(float left, float right) {
        int leftBits = Float.floatToRawIntBits(left) >>> 16;
        int rightBits = Float.floatToRawIntBits(right) >>> 16;
        return Math.abs(orderedBfloat16(leftBits) - orderedBfloat16(rightBits));
    }

    private static int orderedBfloat16(int bits) {
        return (bits & 0x8000) == 0 ? 0x8000 + bits : 0x8000 - (bits & 0x7fff);
    }

    private static void printMappedAttentionComparison(
            DataType dataType, int seed, AttentionComparison comparison) {
        System.out.printf(
                "MAPPED_GROUPED_ATTENTION_FORWARD dtype=%s path=%s seed=%d"
                        + " maxAbs=%g maxRel=%g maxBf16Ulps=%d mask=ZERO_PADDED%n",
                dataType,
                comparison.path,
                seed,
                comparison.maximumAbsoluteError,
                comparison.maximumRelativeError,
                comparison.maximumBfloat16Ulps);
    }

    private static void assertClose(float[] actual, float[] expected, float tolerance) {
        Assert.assertEquals(actual.length, expected.length);
        for (int i = 0; i < actual.length; i++) {
            Assert.assertEquals(actual[i], expected[i], tolerance, "element " + i);
        }
    }

    private static NDArray requiringGradient(NDArray array) {
        array.setRequiresGradient(true);
        return array;
    }

    private static void assertFiniteNonzeroGradient(NDArray... arrays) {
        for (NDArray array : arrays) {
            NDArray gradient = array.getGradient();
            Assert.assertNotNull(gradient);
            try (gradient;
                    NDArray values = gradient.toType(DataType.FLOAT32, false)) {
                boolean nonzero = false;
                for (float value : values.toFloatArray()) {
                    Assert.assertTrue(Float.isFinite(value));
                    nonzero |= value != 0f;
                }
                Assert.assertTrue(nonzero, "expected a nonzero gradient for " + array.getShape());
            }
        }
    }

    private static float assertGradientClose(NDArray actual, NDArray expected, float tolerance) {
        NDArray actualGradient = actual.getGradient();
        NDArray expectedGradient = expected.getGradient();
        Assert.assertNotNull(actualGradient);
        Assert.assertNotNull(expectedGradient);
        try (actualGradient;
                expectedGradient;
                NDArray actualValues = actualGradient.toType(DataType.FLOAT32, false);
                NDArray expectedValues = expectedGradient.toType(DataType.FLOAT32, false)) {
            float[] actualData = actualValues.toFloatArray();
            float[] expectedData = expectedValues.toFloatArray();
            float maximumDifference = 0f;
            for (int index = 0; index < actualData.length; index++) {
                maximumDifference =
                        Math.max(
                                maximumDifference,
                                Math.abs(actualData[index] - expectedData[index]));
            }
            assertClose(actualData, expectedData, tolerance);
            return maximumDifference;
        }
    }

    private static void assertAllZero(float[] values) {
        for (float value : values) {
            Assert.assertEquals(value, 0f);
        }
    }

    private enum MappedAttentionPath {
        INT32_NATIVE,
        INT64_EAGER_FALLBACK
    }

    private static final class AttentionComparison {

        private final MappedAttentionPath path;
        private final float maximumAbsoluteError;
        private final float maximumRelativeError;
        private final int maximumBfloat16Ulps;

        private AttentionComparison(
                MappedAttentionPath path,
                float maximumAbsoluteError,
                float maximumRelativeError,
                int maximumBfloat16Ulps) {
            this.path = path;
            this.maximumAbsoluteError = maximumAbsoluteError;
            this.maximumRelativeError = maximumRelativeError;
            this.maximumBfloat16Ulps = maximumBfloat16Ulps;
        }
    }
}
