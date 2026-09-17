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
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.GradientCollector;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class SingleQueryAttentionTest {

    @Test
    public void singleQueryAttentionMatchesEagerTraining() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            verifyTrainingParity(engine, manager, 16, false, null);
            verifyTrainingParity(engine, manager, 32, true, 0.27);
        }
    }

    @Test
    public void singleQueryAttentionSupportsPartialGradients() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            NDArray query = random(manager, new Shape(2, 4, 1, 16));
            NDArray key = random(manager, new Shape(2, 4, 11, 16));
            NDArray value = random(manager, new Shape(2, 4, 11, 16));
            query.setRequiresGradient(true);
            try (GradientCollector collector = engine.newGradientCollector()) {
                collector.backward(
                        query.getNDArrayInternal()
                                .scaledDotProductAttention(key, value, null, 0.0, false)
                                .sum());
            }
            Assert.assertTrue(query.hasGradient());
            Assert.assertFalse(key.hasGradient());
            Assert.assertFalse(value.hasGradient());
            assertFinite(query.getGradient());
        }
    }

    @Test
    public void fusedKeyValueViewsMatchFloat32Training() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            for (boolean masked : new boolean[] {false, true}) {
                NDArray query = requiringGradient(random(manager, new Shape(2, 4, 1, 16)));
                NDArray keyValues = requiringGradient(random(manager, new Shape(2, 151, 128)));
                // Convert before attaching gradients: toType is a tensor-transfer operation.
                NDArray referenceQuery = requiringGradient(query.toType(DataType.FLOAT32, true));
                NDArray referenceKeyValues =
                        requiringGradient(keyValues.toType(DataType.FLOAT32, true));
                NDArray gradient = random(manager, query.getShape());
                NDArray mask = masked ? additiveMask(manager, 2, 151) : null;
                NDArray actual;
                try (GradientCollector collector = engine.newGradientCollector()) {
                    actual =
                            query.getNDArrayInternal()
                                    .scaledDotProductAttention(
                                            fusedKeyValueView(keyValues, 0),
                                            fusedKeyValueView(keyValues, 64),
                                            mask,
                                            0.0,
                                            false);
                    collector.backward(actual.mul(gradient).sum());
                }
                NDArray expected;
                try (GradientCollector collector = engine.newGradientCollector()) {
                    expected =
                            eagerAttention(
                                    referenceQuery,
                                    fusedKeyValueView(referenceKeyValues, 0),
                                    fusedKeyValueView(referenceKeyValues, 64),
                                    mask,
                                    null);
                    collector.backward(expected.mul(gradient).sum());
                }
                assertClose(actual, expected, 0.025f);
                assertClose(query.getGradient(), referenceQuery.getGradient(), 0.035f);
                assertClose(keyValues.getGradient(), referenceKeyValues.getGradient(), 0.035f);
            }
        }
    }

    @Test
    public void largeCommonScoresPreserveValueGradientProbabilityMass() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0 || JniUtils.getFusionBackend() != 2) {
            throw new SkipException(
                    "This single-query attention regression requires PyTorch ROCm.");
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            // At scores 2^28, adding log(151) in FP32 loses the normalization term. Backward
            // must use normalized probabilities, not reconstruct them from rounded logsumexp.
            for (boolean contiguous : new boolean[] {false, true}) {
                for (boolean masked : new boolean[] {false, true}) {
                    NDArray query =
                            requiringGradient(
                                    manager.full(new Shape(2, 4, 1, 16), 8192, DataType.BFLOAT16));
                    NDArray fused = manager.full(new Shape(2, 151, 128), 8192, DataType.BFLOAT16);
                    NDArray key = fusedKeyValueView(fused, 0);
                    NDArray value = fusedKeyValueView(fused, 64);
                    if (contiguous) {
                        key = manager.full(key.getShape(), 8192, DataType.BFLOAT16);
                        value = manager.full(value.getShape(), 8192, DataType.BFLOAT16);
                    }
                    key = requiringGradient(key);
                    value = requiringGradient(value);
                    NDArray mask = masked ? additiveMask(manager, 2, 151) : null;
                    NDArray output;
                    try (GradientCollector collector = engine.newGradientCollector()) {
                        output =
                                query.getNDArrayInternal()
                                        .scaledDotProductAttention(key, value, mask, 0.0, false);
                        collector.backward(output.sum());
                    }
                    assertFinite(output);
                    assertFinite(query.getGradient());
                    assertFinite(key.getGradient());
                    assertValueGradientProbabilities(value.getGradient(), masked);
                }
            }
        }
    }

    @Test
    public void largeBatchPeakedScoresPreserveForwardAndBackwardReductions() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0 || JniUtils.getFusionBackend() != 2) {
            throw new SkipException(
                    "This single-query attention regression requires PyTorch ROCm.");
        }
        int batch = 512;
        float[] queryValues = new float[batch * 4 * 16];
        float[] packedValues = new float[batch * 151 * 128];
        float[] expectedOutput = new float[batch * 4 * 16];
        for (int row = 0; row < batch; ++row) {
            for (int head = 0; head < 4; ++head) {
                int peak = peakKey(row, head);
                for (int channel = 0; channel < 16; ++channel) {
                    int outputIndex = (row * 4 + head) * 16 + channel;
                    queryValues[outputIndex] = 8192 + ((row + head) % 8) * 128;
                    for (int key = 0; key < 151; ++key) {
                        int keyIndex = (row * 151 + key) * 128 + head * 16 + channel;
                        // The peak is BF16-representable and lives beyond the first warp.
                        packedValues[keyIndex] = key == peak ? 8320 : 8192;
                        float value = ((row + head + key + channel) % 17 - 8) * 0.125f;
                        packedValues[keyIndex + 64] = value;
                        if (key == peak) {
                            expectedOutput[outputIndex] = value;
                        }
                    }
                }
            }
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            NDArray sourceQuery =
                    manager.create(queryValues, new Shape(batch, 4, 1, 16))
                            .toType(DataType.BFLOAT16, false);
            NDArray sourceKeyValues =
                    manager.create(packedValues, new Shape(batch, 151, 128))
                            .toType(DataType.BFLOAT16, false);
            NDArray expected = manager.create(expectedOutput, new Shape(batch, 4, 1, 16));
            for (boolean masked : new boolean[] {false, true}) {
                for (int repeat = 0; repeat < 3; ++repeat) {
                    try (NDManager step = manager.newSubManager()) {
                        NDArray query = requiringGradient(sourceQuery.duplicate());
                        NDArray keyValues = requiringGradient(sourceKeyValues.duplicate());
                        query.attach(step);
                        keyValues.attach(step);
                        NDArray mask = masked ? additiveMask(step, batch, 151) : null;
                        NDArray output;
                        try (GradientCollector collector = engine.newGradientCollector()) {
                            output =
                                    query.getNDArrayInternal()
                                            .scaledDotProductAttention(
                                                    fusedKeyValueView(keyValues, 0),
                                                    fusedKeyValueView(keyValues, 64),
                                                    mask,
                                                    0.0,
                                                    false);
                            collector.backward(output.sum());
                        }
                        assertClose(output, expected, 0.0f);
                        assertZero(query.getGradient());
                        assertPeakedPackedGradient(keyValues.getGradient(), batch);
                    }
                }
            }
        }
    }

    @Test
    public void unsupportedMaskPreservesPyTorchFallback() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            NDArray query = requiringGradient(random(manager, new Shape(2, 4, 1, 16)));
            NDArray key = requiringGradient(random(manager, new Shape(2, 4, 9, 16)));
            NDArray value = requiringGradient(random(manager, new Shape(2, 4, 9, 16)));
            NDArray mask = manager.ones(new Shape(2, 1, 1, 9), DataType.BOOLEAN);
            try (GradientCollector collector = engine.newGradientCollector()) {
                collector.backward(
                        query.getNDArrayInternal()
                                .scaledDotProductAttention(key, value, mask, 0.0, false)
                                .sum());
            }
            assertFinite(query.getGradient());
            assertFinite(key.getGradient());
            assertFinite(value.getGradient());
        }
    }

    @Test
    public void differentiableMaskPreservesPyTorchFallback() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            NDArray query = random(manager, new Shape(2, 4, 1, 16));
            NDArray key = random(manager, new Shape(2, 4, 7, 16));
            NDArray value = random(manager, new Shape(2, 4, 7, 16));
            NDArray mask =
                    requiringGradient(manager.zeros(new Shape(2, 1, 1, 7), DataType.BFLOAT16));
            try (GradientCollector collector = engine.newGradientCollector()) {
                collector.backward(
                        query.getNDArrayInternal()
                                .scaledDotProductAttention(key, value, mask, 0.0, false)
                                .sum());
            }
            Assert.assertTrue(mask.hasGradient());
            assertFinite(mask.getGradient());
        }
    }

    @Test
    public void wideHeadLongKeyTrainingPreservesPyTorchFallback() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            NDArray query = requiringGradient(random(manager, new Shape(2, 4, 1, 32)));
            NDArray key = requiringGradient(random(manager, new Shape(2, 4, 64, 32)));
            NDArray value = requiringGradient(random(manager, new Shape(2, 4, 64, 32)));
            NDArray mask = additiveMask(manager, 2, 64);
            try (GradientCollector collector = engine.newGradientCollector()) {
                collector.backward(
                        query.getNDArrayInternal()
                                .scaledDotProductAttention(key, value, mask, 0.0, false)
                                .sum());
            }
            assertFinite(query.getGradient());
            assertFinite(key.getGradient());
            assertFinite(value.getGradient());
        }
    }

    @Test
    public void fullyMaskedRowsProduceZeroOutputAndGradients() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            NDArray query = requiringGradient(random(manager, new Shape(2, 4, 1, 16)));
            NDArray key = requiringGradient(random(manager, new Shape(2, 4, 7, 16)));
            NDArray value = requiringGradient(random(manager, new Shape(2, 4, 7, 16)));
            NDArray mask =
                    manager.full(new Shape(2, 1, 1, 7), Float.NEGATIVE_INFINITY, DataType.BFLOAT16);
            NDArray output;
            try (GradientCollector collector = engine.newGradientCollector()) {
                output =
                        query.getNDArrayInternal()
                                .scaledDotProductAttention(key, value, mask, 0.0, false);
                collector.backward(output.sum());
            }
            assertZero(output);
            assertZero(query.getGradient());
            assertZero(key.getGradient());
            assertZero(value.getGradient());
        }
    }

    private static void verifyTrainingParity(
            Engine engine, NDManager manager, int width, boolean masked, Double scale) {
        Shape queryShape = new Shape(3, 4, 1, width);
        Shape keyValueShape = new Shape(3, 4, 13, width);
        NDArray sourceQuery = random(manager, queryShape);
        NDArray sourceKey = random(manager, keyValueShape);
        NDArray sourceValue = random(manager, keyValueShape);
        NDArray actualQuery = requiringGradient(sourceQuery.duplicate());
        NDArray actualKey = requiringGradient(sourceKey.duplicate());
        NDArray actualValue = requiringGradient(sourceValue.duplicate());
        NDArray expectedQuery = requiringGradient(sourceQuery.duplicate());
        NDArray expectedKey = requiringGradient(sourceKey.duplicate());
        NDArray expectedValue = requiringGradient(sourceValue.duplicate());
        NDArray gradient = random(manager, queryShape);
        NDArray mask = masked ? additiveMask(manager, 3, 13) : null;

        NDArray actual;
        try (GradientCollector collector = engine.newGradientCollector()) {
            actual =
                    scale == null
                            ? actualQuery
                                    .getNDArrayInternal()
                                    .scaledDotProductAttention(
                                            actualKey, actualValue, mask, 0.0, false)
                            : actualQuery
                                    .getNDArrayInternal()
                                    .scaledDotProductAttention(
                                            actualKey, actualValue, mask, 0.0, false, scale);
            collector.backward(actual.mul(gradient).sum());
        }

        NDArray expected;
        try (GradientCollector collector = engine.newGradientCollector()) {
            expected = eagerAttention(expectedQuery, expectedKey, expectedValue, mask, scale);
            collector.backward(expected.mul(gradient).sum());
        }

        assertClose(actual, expected, 0.025f);
        assertClose(actualQuery.getGradient(), expectedQuery.getGradient(), 0.035f);
        assertClose(actualKey.getGradient(), expectedKey.getGradient(), 0.035f);
        assertClose(actualValue.getGradient(), expectedValue.getGradient(), 0.035f);
    }

    private static NDArray eagerAttention(
            NDArray query, NDArray key, NDArray value, NDArray mask, Double explicitScale) {
        double scale =
                explicitScale == null ? 1.0 / Math.sqrt(query.getShape().get(3)) : explicitScale;
        NDArray scores = query.matMul(key.swapAxes(2, 3)).mul(scale);
        if (mask != null) {
            scores = scores.add(mask);
        }
        return scores.softmax(3).matMul(value);
    }

    private static NDArray fusedKeyValueView(NDArray keyValues, int offset) {
        return keyValues
                .get("...,{}:{}", offset, offset + 64)
                .reshape(keyValues.getShape().get(0), keyValues.getShape().get(1), 4, 16)
                .swapAxes(1, 2);
    }

    private static int peakKey(int row, int head) {
        return 130 + 5 * ((row + head) % 5);
    }

    private static void assertPeakedPackedGradient(NDArray gradient, int batch) {
        float[] values = gradient.toType(DataType.FLOAT32, false).toFloatArray();
        for (int row = 0; row < batch; ++row) {
            for (int head = 0; head < 4; ++head) {
                int peak = peakKey(row, head);
                for (int channel = 0; channel < 16; ++channel) {
                    float probabilityMass = 0;
                    for (int key = 0; key < 151; ++key) {
                        int keyIndex = (row * 151 + key) * 128 + head * 16 + channel;
                        Assert.assertEquals(values[keyIndex], 0.0f);
                        float valueGradient = values[keyIndex + 64];
                        Assert.assertEquals(valueGradient, key == peak ? 1.0f : 0.0f);
                        probabilityMass += valueGradient;
                    }
                    Assert.assertEquals(probabilityMass, 1.0f);
                }
            }
        }
    }

    private static void assertValueGradientProbabilities(NDArray gradient, boolean masked) {
        float[] values = gradient.toType(DataType.FLOAT32, false).toFloatArray();
        int activeKeys = masked ? 121 : 151;
        float probability = 1.0f / activeKeys;
        for (int matrix = 0; matrix < 8; ++matrix) {
            for (int channel = 0; channel < 16; ++channel) {
                float sum = 0;
                for (int key = 0; key < 151; ++key) {
                    float value = values[(matrix * 151 + key) * 16 + channel];
                    Assert.assertTrue(Float.isFinite(value) && value >= 0 && value <= 1);
                    Assert.assertEquals(
                            value, masked && key % 5 == 4 ? 0.0f : probability, 0.0001f);
                    sum += value;
                }
                Assert.assertEquals(sum, 1.0f, 0.005f);
            }
        }
    }

    private static NDArray additiveMask(NDManager manager, int batch, int keys) {
        float[] values = new float[batch * keys];
        for (int row = 0; row < batch; ++row) {
            for (int key = 0; key < keys; ++key) {
                values[row * keys + key] = key % 5 == 4 ? -1.0e9f : 0.0f;
            }
        }
        return manager.create(values, new Shape(batch, 1, 1, keys))
                .toType(DataType.BFLOAT16, false);
    }

    private static NDArray random(NDManager manager, Shape shape) {
        return manager.randomNormal(shape).toType(DataType.BFLOAT16, false);
    }

    private static NDArray requiringGradient(NDArray array) {
        array.setRequiresGradient(true);
        return array;
    }

    private static void assertClose(NDArray actual, NDArray expected, float tolerance) {
        float[] actualValues = actual.toType(DataType.FLOAT32, false).toFloatArray();
        float[] expectedValues = expected.toType(DataType.FLOAT32, false).toFloatArray();
        Assert.assertEquals(actualValues.length, expectedValues.length);
        for (int index = 0; index < actualValues.length; ++index) {
            Assert.assertEquals(actualValues[index], expectedValues[index], tolerance);
        }
    }

    private static void assertFinite(NDArray array) {
        for (float value : array.toType(DataType.FLOAT32, false).toFloatArray()) {
            Assert.assertTrue(Float.isFinite(value));
        }
    }

    private static void assertZero(NDArray array) {
        for (float value : array.toType(DataType.FLOAT32, false).toFloatArray()) {
            Assert.assertEquals(value, 0.0f);
        }
    }
}
