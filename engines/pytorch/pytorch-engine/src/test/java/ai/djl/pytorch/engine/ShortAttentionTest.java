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

public class ShortAttentionTest {

    @Test
    public void packedSelfAttentionMatchesFloat32ForwardAndAllGradients() {
        Engine engine = rocmEngine();
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            for (DataType type : new DataType[] {DataType.FLOAT32, DataType.BFLOAT16}) {
                verifyPackedParity(engine, manager, type, 6, 32, false);
                verifyPackedParity(engine, manager, type, 29, 16, false);
                verifyPackedParity(engine, manager, type, 34, 32, true);
                verifyPackedParity(engine, manager, type, 64, 16, true);
                verifyPackedParity(engine, manager, type, 64, 32, true);
            }
        }
    }

    @Test
    public void largeTiedScoresPreserveAnalyticForwardAndGradients() {
        Engine engine = rocmEngine();
        int tokens = 29;
        int width = 16;
        int packedWidth = 4 * width * 3;
        float[] data = new float[2 * tokens * packedWidth];
        for (int row = 0; row < 2 * tokens; ++row) {
            for (int column = 0; column < packedWidth; ++column) {
                data[row * packedWidth + column] =
                        column < 4 * width * 2 ? 8192f : (row % tokens == 0 ? 1f : -1f);
            }
        }
        float[] maskData = new float[tokens];
        for (int key = 2; key < tokens; ++key) {
            maskData[key] = Float.NEGATIVE_INFINITY;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            for (DataType type : new DataType[] {DataType.FLOAT32, DataType.BFLOAT16}) {
                NDArray packed =
                        requiringGradient(
                                manager.create(data, new Shape(2, tokens, packedWidth))
                                        .toType(type, false));
                NDArray mask =
                        requiringGradient(manager.create(maskData, new Shape(1, 1, 1, tokens)));
                NDArray output;
                try (GradientCollector collector = engine.newGradientCollector()) {
                    output =
                            view(packed, 0, tokens, width)
                                    .getNDArrayInternal()
                                    .scaledDotProductAttention(
                                            view(packed, 1, tokens, width),
                                            view(packed, 2, tokens, width),
                                            mask,
                                            0.0,
                                            false);
                    collector.backward(output.sum());
                }
                assertZero(output);
                float[] gradient = floats(packed.getGradient());
                // Two keys tie at score 2^28. P=.5 exactly, V=(+1,-1), O=0.
                // Each query has dS=(+8,-8), dQ=0, and dK=29*8*8192/4.
                for (int row = 0; row < 2 * tokens; ++row) {
                    int token = row % tokens;
                    for (int column = 0; column < packedWidth; ++column) {
                        float expected = 0f;
                        if (token < 2 && column >= 4 * width && column < 8 * width) {
                            expected = token == 0 ? 475136f : -475136f;
                        } else if (token < 2 && column >= 8 * width) {
                            expected = 14.5f;
                        }
                        Assert.assertEquals(gradient[row * packedWidth + column], expected, 0f);
                    }
                }
                float[] maskGradient = floats(mask.getGradient());
                for (int key = 0; key < tokens; ++key) {
                    Assert.assertEquals(
                            maskGradient[key], key == 0 ? 1856f : key == 1 ? -1856f : 0f, 0f);
                }
            }
        }
    }

    @Test
    public void emptyBooleanAndAdditiveMasksHaveZeroOutputAndGradients() {
        Engine engine = rocmEngine();
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            for (DataType type : new DataType[] {DataType.FLOAT32, DataType.BFLOAT16}) {
                for (boolean booleanMask : new boolean[] {false, true}) {
                    NDArray query =
                            requiringGradient(random(manager, new Shape(2, 4, 29, 16), type));
                    NDArray key = requiringGradient(random(manager, new Shape(2, 4, 29, 16), type));
                    NDArray value =
                            requiringGradient(random(manager, new Shape(2, 4, 29, 16), type));
                    NDArray mask =
                            booleanMask
                                    ? manager.zeros(new Shape(2, 1, 1, 29), DataType.BOOLEAN)
                                    : requiringGradient(
                                            manager.full(
                                                    new Shape(2, 1, 1, 29),
                                                    Float.NEGATIVE_INFINITY,
                                                    DataType.FLOAT32));
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
                    if (!booleanMask) assertZero(mask.getGradient());
                }
            }
        }
    }

    @Test
    public void partialGradientsAndFiniteMasksRetainTheirMeaning() {
        Engine engine = rocmEngine();
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            // A finite negative bias common to all keys still defines a uniform distribution.
            NDArray query = manager.zeros(new Shape(2, 4, 3, 16), DataType.BFLOAT16);
            NDArray key = manager.zeros(new Shape(2, 4, 5, 16), DataType.BFLOAT16);
            NDArray value =
                    requiringGradient(manager.ones(new Shape(2, 4, 5, 16), DataType.BFLOAT16));
            NDArray mask = manager.full(new Shape(5), -1.0e9f, DataType.BFLOAT16);
            NDArray output;
            try (GradientCollector collector = engine.newGradientCollector()) {
                output =
                        query.getNDArrayInternal()
                                .scaledDotProductAttention(key, value, mask, 0.0, false);
                collector.backward(output.sum());
            }
            for (float result : floats(output)) Assert.assertEquals(result, 1f, 0f);
            for (float gradient : floats(value.getGradient()))
                Assert.assertEquals(gradient, 0.6f, 0.003f);
            Assert.assertFalse(query.hasGradient());
            Assert.assertFalse(key.hasGradient());

            NDArray maskOnly =
                    requiringGradient(manager.zeros(new Shape(1, 4, 3, 5), DataType.FLOAT32));
            NDArray variedValues = random(manager, new Shape(2, 4, 5, 16), DataType.BFLOAT16);
            try (GradientCollector collector = engine.newGradientCollector()) {
                collector.backward(
                        query.getNDArrayInternal()
                                .scaledDotProductAttention(key, variedValues, maskOnly, 0.0, false)
                                .sum());
            }
            Assert.assertTrue(maskOnly.hasGradient());
            Assert.assertFalse(query.hasGradient());
            for (float gradient : floats(maskOnly.getGradient()))
                Assert.assertTrue(Float.isFinite(gradient));
        }
    }

    @Test
    public void unsupportedHeadUsesStableMathAndBooleanMaskSemantics() {
        Engine engine = rocmEngine();
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            NDArray query =
                    requiringGradient(manager.full(new Shape(1, 1, 3, 8), 8192, DataType.BFLOAT16));
            NDArray key =
                    requiringGradient(manager.full(new Shape(1, 1, 5, 8), 8192, DataType.BFLOAT16));
            NDArray value =
                    requiringGradient(manager.ones(new Shape(1, 1, 5, 8), DataType.BFLOAT16));
            NDArray mask = manager.create(new boolean[] {true, false, false, false, false});
            NDArray output;
            try (GradientCollector collector = engine.newGradientCollector()) {
                output =
                        query.getNDArrayInternal()
                                .scaledDotProductAttention(key, value, mask, 0.0, false);
                collector.backward(output.sum());
            }
            for (float result : floats(output)) Assert.assertEquals(result, 1f, 0f);
            float[] gradient = floats(value.getGradient());
            for (int index = 0; index < gradient.length; ++index) {
                Assert.assertEquals(gradient[index], index < 8 ? 3f : 0f, 0f);
            }
        }
    }

    private static void verifyPackedParity(
            Engine engine,
            NDManager manager,
            DataType type,
            int tokens,
            int width,
            boolean differentiableMask) {
        Shape shape = new Shape(2, tokens, 4 * width * 3);
        NDArray packed = requiringGradient(random(manager, shape, type));
        NDArray referencePacked = requiringGradient(packed.toType(DataType.FLOAT32, true));
        NDArray upstream = random(manager, new Shape(2, 4, tokens, width), type);
        NDArray referenceUpstream = upstream.toType(DataType.FLOAT32, true);
        NDArray mask;
        if (tokens == 6 && !differentiableMask) {
            mask = null;
        } else if (differentiableMask) {
            // Broadcast batch axis; expand creates zero-stride views as in tile relation bias.
            mask =
                    requiringGradient(
                            random(manager, new Shape(1, 4, tokens, tokens), DataType.FLOAT32));
        } else {
            float[] values = new float[2 * tokens];
            for (int index = 0; index < values.length; ++index) {
                values[index] = index % 5 == 4 ? Float.NEGATIVE_INFINITY : 0f;
            }
            mask = manager.create(values, new Shape(2, 1, 1, tokens)).toType(type, false);
        }
        NDArray referenceMask = mask == null ? null : mask.toType(DataType.FLOAT32, true);
        if (differentiableMask) referenceMask.setRequiresGradient(true);
        NDArray actual;
        try (GradientCollector collector = engine.newGradientCollector()) {
            NDArray submittedMask =
                    differentiableMask ? mask.broadcast(2, 4, tokens, tokens) : mask;
            actual =
                    view(packed, 0, tokens, width)
                            .getNDArrayInternal()
                            .scaledDotProductAttention(
                                    view(packed, 1, tokens, width),
                                    view(packed, 2, tokens, width),
                                    submittedMask,
                                    0.0,
                                    false,
                                    0.25);
            collector.backward(actual.mul(upstream).sum());
        }
        NDArray expected;
        try (GradientCollector collector = engine.newGradientCollector()) {
            NDArray scores =
                    view(referencePacked, 0, tokens, width)
                            .matMul(view(referencePacked, 1, tokens, width).swapAxes(2, 3))
                            .mul(0.25);
            if (referenceMask != null) scores = scores.add(referenceMask);
            expected = scores.softmax(3).matMul(view(referencePacked, 2, tokens, width));
            collector.backward(expected.mul(referenceUpstream).sum());
        }
        float tolerance = type == DataType.FLOAT32 ? 0.00003f : 0.015f;
        assertClose(actual, expected, tolerance);
        assertClose(packed.getGradient(), referencePacked.getGradient(), tolerance);
        if (differentiableMask)
            assertClose(mask.getGradient(), referenceMask.getGradient(), tolerance);
    }

    private static NDArray view(NDArray packed, int component, int tokens, int width) {
        int offset = component * 4 * width;
        return packed.get("...,{}:{}", offset, offset + 4 * width)
                .reshape(2, tokens, 4, width)
                .swapAxes(1, 2);
    }

    private static NDArray random(NDManager manager, Shape shape, DataType type) {
        return manager.randomNormal(shape).mul(0.25f).toType(type, false);
    }

    private static NDArray requiringGradient(NDArray array) {
        array.setRequiresGradient(true);
        return array;
    }

    private static float[] floats(NDArray array) {
        return array.toType(DataType.FLOAT32, false).toFloatArray();
    }

    private static void assertZero(NDArray array) {
        for (float value : floats(array)) Assert.assertEquals(value, 0f, 0f);
    }

    private static void assertClose(NDArray actual, NDArray expected, float tolerance) {
        float[] left = floats(actual);
        float[] right = floats(expected);
        Assert.assertEquals(left.length, right.length);
        for (int index = 0; index < left.length; ++index) {
            Assert.assertTrue(Float.isFinite(left[index]), "non-finite result at " + index);
            Assert.assertEquals(left[index], right[index], tolerance, "element " + index);
        }
    }

    private static Engine rocmEngine() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0 || JniUtils.getFusionBackend() != 2) {
            throw new SkipException("Short attention regression requires PyTorch ROCm.");
        }
        return engine;
    }
}
