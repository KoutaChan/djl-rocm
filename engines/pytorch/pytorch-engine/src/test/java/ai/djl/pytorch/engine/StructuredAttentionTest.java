/*
 * Copyright 2026 KoutaChan. Licensed under the Apache License, Version 2.0.
 */
package ai.djl.pytorch.engine;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

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
                            query, key, value, relationKeys, relationBias, relationIds, 1.0, true);

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
                            query, shared, sharedDeltas, indexedDeltas, indexedIds, 2, 1.0, true);

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
                    NDArrays.residualAddLayerNormInPlace(residual, update, weight, bias, 1.0e-5f);

            assertClose(residual.toFloatArray(), new float[] {2f, 4f}, 1e-6f);
            assertClose(normalized.toFloatArray(), new float[] {-0.999995f, 0.999995f}, 1e-5f);
        }
    }

    @Test
    public void portableOperationsDoNotRetainBatchArrays() {
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
                runPortableStep(
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
            verifyResidualLayerNorm(manager);
        }
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
        double scale = 1.0 / Math.sqrt(6);

        NDArray portable =
                NDArrays.relationBiasedScaledDotProductAttention(
                        query, key, value, relationKeys, relationBias, relationIds, scale, true);
        NDArray fused =
                NDArrays.relationBiasedScaledDotProductAttention(
                        query, key, value, relationKeys, relationBias, relationIds, scale, false);
        assertClose(fused.toFloatArray(), portable.toFloatArray(), 2e-4f);
    }

    private static void runPortableStep(
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
                            1.0 / Math.sqrt(2),
                            true);
            Assert.assertSame(relationOutput.getManager(), batchManager);
            NDArray groupedOutput =
                    NDArrays.groupedIndexedScaledDotProductAttention(
                            groupedQuery,
                            sharedKeyValues,
                            sharedDeltas,
                            indexedDeltas,
                            indexedSharedIds,
                            2,
                            1.0 / Math.sqrt(2),
                            true);
            Assert.assertSame(groupedOutput.getManager(), batchManager);
        }
    }

    private static void verifyGroupedAttention(NDManager manager) {
        int queries = 6;
        NDArray query = manager.randomNormal(new Shape(queries, 3, 5));
        NDArray shared = manager.randomNormal(new Shape(2, 7, 42));
        NDArray sharedDeltas = manager.randomNormal(new Shape(queries, 7, 42));
        NDArray indexedDeltas = manager.randomNormal(new Shape(queries, 4, 42));
        int[] ids = new int[queries * 4];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = i % 8;
        }
        NDArray indexedIds = manager.create(ids, new Shape(queries, 4));
        double scale = 1.0 / Math.sqrt(5);

        NDArray portable =
                NDArrays.groupedIndexedScaledDotProductAttention(
                        query, shared, sharedDeltas, indexedDeltas, indexedIds, 3, scale, true);
        NDArray fused =
                NDArrays.groupedIndexedScaledDotProductAttention(
                        query, shared, sharedDeltas, indexedDeltas, indexedIds, 3, scale, false);
        assertClose(fused.toFloatArray(), portable.toFloatArray(), 2e-4f);
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
                    NDArrays.residualAddLayerNormInPlace(residual, update, weight, bias, 1.0e-5f);

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
}
