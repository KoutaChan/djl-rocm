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
import ai.djl.training.GradientCollector;

import org.testng.Assert;
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
                    manager.full(
                            new Shape(2, 1, 1, 7),
                            Float.NEGATIVE_INFINITY,
                            DataType.BFLOAT16);
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
