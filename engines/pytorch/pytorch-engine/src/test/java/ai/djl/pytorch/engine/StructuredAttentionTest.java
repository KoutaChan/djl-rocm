/*
 * Copyright 2026 KoutaChan. Licensed under the Apache License, Version 2.0.
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
        verifyStructuredAttentionGradients(engine, Device.cpu());
        if (engine.getGpuCount() > 0) {
            verifyStructuredAttentionGradients(engine, Device.gpu());
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
            verifyRelationAttentionLeadingDimensions(manager);
            verifyGroupedAttentionLeadingDimensions(manager);
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
            verifyRelationAttentionLeadingDimensions(manager);
            verifyGroupedAttentionLeadingDimensions(manager);
            verifyResidualLayerNorm(manager);
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

    private static void verifyStructuredAttentionGradients(Engine engine, Device device) {
        verifyRelationAttentionGradients(engine, device);
        verifyGroupedAttentionGradients(engine, device);
    }

    private static void verifyRelationAttentionGradients(Engine engine, Device device) {
        try (NDManager manager = engine.newBaseManager(device);
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray query = requiringGradient(manager.randomNormal(new Shape(1, 2, 2, 3, 4)));
            NDArray key = requiringGradient(manager.randomNormal(new Shape(1, 2, 2, 5, 4)));
            NDArray value = requiringGradient(manager.randomNormal(new Shape(1, 2, 2, 5, 3)));
            NDArray relationKeys =
                    requiringGradient(manager.randomNormal(new Shape(1, 2, 2, 4, 7)));
            NDArray relationBias =
                    requiringGradient(manager.randomNormal(new Shape(1, 2, 1, 3, 5)));
            NDArray relationIds =
                    manager.create(
                            new int[] {
                                0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0,
                                1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1
                            },
                            new Shape(1, 2, 3, 5));

            NDArray output =
                    NDArrays.relationBiasedScaledDotProductAttention(
                            query, key, value, relationKeys, relationBias, relationIds, 0.31);
            collector.backward(output.mul(output).sum());

            assertFiniteNonzeroGradient(query, key, value, relationKeys, relationBias);
        }
    }

    private static void verifyGroupedAttentionGradients(Engine engine, Device device) {
        try (NDManager manager = engine.newBaseManager(device);
                GradientCollector collector = engine.newGradientCollector()) {
            int heads = 2;
            int keyFeatures = 3;
            int packedWidth = heads * (keyFeatures + 4);
            NDArray query =
                    requiringGradient(manager.randomNormal(new Shape(2, 2, heads, keyFeatures)));
            NDArray shared = requiringGradient(manager.randomNormal(new Shape(2, 5, packedWidth)));
            NDArray sharedDeltas =
                    requiringGradient(manager.randomNormal(new Shape(2, 2, 5, packedWidth)));
            NDArray indexedDeltas =
                    requiringGradient(manager.randomNormal(new Shape(2, 2, 3, packedWidth)));
            NDArray indexedIds =
                    manager.create(
                            new int[] {1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 1}, new Shape(2, 2, 3));

            NDArray output =
                    NDArrays.groupedIndexedScaledDotProductAttention(
                            query, shared, sharedDeltas, indexedDeltas, indexedIds, 2, 0.27);
            collector.backward(output.mul(output).sum());

            assertFiniteNonzeroGradient(query, shared, sharedDeltas, indexedDeltas);
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
            try (NDArray gradient = array.getGradient()) {
                Assert.assertNotNull(gradient);
                boolean nonzero = false;
                for (float value : gradient.toFloatArray()) {
                    Assert.assertTrue(Float.isFinite(value));
                    nonzero |= value != 0f;
                }
                Assert.assertTrue(nonzero, "expected a nonzero gradient for " + array.getShape());
            }
        }
    }
}
