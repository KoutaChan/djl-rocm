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
import ai.djl.training.GradientCollector;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;

import org.testng.Assert;
import org.testng.annotations.Test;

/** Verifies masked categorical semantics and ROCm optimizer parity. */
public class MaskedCategoricalTest {

    @Test
    public void maskedRowsAndEmptyRowsHaveDefinedValues() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray logits =
                    manager.create(
                            new float[] {0f, 7f, (float) Math.log(3), 4f, 5f, 6f}, new Shape(2, 3));
            NDArray mask = manager.create(new int[] {1, 0, 1, 0, 0, 0}, new Shape(2, 3));

            NDArray probabilities = NDArrays.maskedSoftmax(logits, mask, -1);
            NDArray normalizers = NDArrays.maskedLogSumExp(logits, mask, -1);

            Assert.assertEquals(probabilities.getDataType(), DataType.FLOAT32);
            Assert.assertEquals(normalizers.getShape(), new Shape(2, 1));
            assertClose(
                    probabilities.toFloatArray(),
                    new float[] {0.25f, 0f, 0.75f, 0f, 0f, 0f},
                    1e-6f);
            assertClose(normalizers.toFloatArray(), new float[] {(float) Math.log(4), 0f}, 1e-6f);
        }
    }

    @Test
    public void arbitraryAxisAndBroadcastMaskUsePortableSemantics() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            NDArray logits =
                    manager.create(
                            new float[] {0f, 2f, 1f, 3f, 4f, 6f, 5f, 7f}, new Shape(2, 2, 2));
            NDArray mask = manager.create(new int[] {1, 0}, new Shape(1, 2, 1));

            NDArray probabilities = NDArrays.maskedSoftmax(logits, mask, 1);
            NDArray normalizers = NDArrays.maskedLogSumExp(logits, mask, 1);

            assertClose(
                    probabilities.toFloatArray(),
                    new float[] {1f, 1f, 0f, 0f, 1f, 1f, 0f, 0f},
                    1e-6f);
            assertClose(normalizers.toFloatArray(), new float[] {0f, 2f, 4f, 6f}, 1e-6f);
        }
    }

    @Test
    public void groupedMaskedSoftmaxPoolMatchesPortableSemantics() {
        Engine engine = Engine.getInstance();
        verifyGroupedPool(engine, Device.cpu(), DataType.FLOAT32, 1e-6f);
        if (engine.getGpuCount() > 0) {
            verifyGroupedPool(engine, Device.gpu(), DataType.FLOAT16, 2e-3f);
        }
    }

    @Test
    public void nativeForwardAndBackwardMatchCpuReference() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        verifyMaskedGradientParity(engine, DataType.FLOAT32, 2e-5f);
        verifyMaskedGradientParity(engine, DataType.BFLOAT16, 2e-2f);
    }

    @Test
    public void maskedNonFiniteValuesRemainObservable() {
        Engine engine = Engine.getInstance();
        verifyMaskedNonFiniteValues(engine, Device.cpu());
        if (engine.getGpuCount() > 0) {
            verifyMaskedNonFiniteValues(engine, Device.gpu());
        }
    }

    @Test
    public void fusedAdamWMatchesPortableUpdate() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        float[] weights = {1f, -2f, 0.5f, 3f};
        float[] gradients = {0.2f, -0.4f, 0.8f, -0.1f};
        float[] cpuResult;
        try (NDManager manager = engine.newBaseManager(Device.cpu())) {
            NDArray weight = manager.create(weights);
            NDArray gradient = manager.create(gradients);
            Optimizer optimizer =
                    Optimizer.adamW()
                            .optLearningRateTracker(Tracker.fixed(1.0e-3f))
                            .optWeightDecays(0.01f)
                            .build();
            optimizer.update("weight", weight, gradient);
            optimizer.update("weight", weight, gradient);
            cpuResult = weight.toFloatArray();
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            NDArray weight = manager.create(weights);
            NDArray gradient = manager.create(gradients);
            Optimizer optimizer =
                    Optimizer.adamW()
                            .optLearningRateTracker(Tracker.fixed(1.0e-3f))
                            .optWeightDecays(0.01f)
                            .build();
            optimizer.update("weight", weight, gradient);
            optimizer.update("weight", weight, gradient);
            assertClose(weight.toFloatArray(), cpuResult, 2e-6f);
        }

        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            NDArray weight = manager.ones(new Shape(1));
            NDArray gradient = manager.create(new float[] {Float.NaN});
            Optimizer optimizer =
                    Optimizer.adamW().optLearningRateTracker(Tracker.fixed(1.0e-3f)).build();
            optimizer.update("non-finite", weight, gradient);
            Assert.assertTrue(
                    Float.isNaN(weight.getFloat()), "NaN gradient must remain observable");
        }
    }

    private static void verifyMaskedNonFiniteValues(Engine engine, Device device) {
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray logits = manager.create(new float[] {Float.NaN, 0f, 7f});
            NDArray mask = manager.create(new int[] {1, 1, 0});
            float[] probabilities = NDArrays.maskedSoftmax(logits, mask, -1).toFloatArray();
            float normalizer = NDArrays.maskedLogSumExp(logits, mask, -1).getFloat();
            Assert.assertTrue(Float.isNaN(probabilities[0]));
            Assert.assertTrue(Float.isNaN(probabilities[1]));
            Assert.assertEquals(probabilities[2], 0f);
            Assert.assertTrue(Float.isNaN(normalizer));
        }
    }

    private static void verifyGroupedPool(
            Engine engine, Device device, DataType dataType, float tolerance) {
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray logits =
                    manager.create(
                                    new float[] {
                                        0f, (float) Math.log(2), (float) Math.log(3),
                                        1f, 2f, 3f,
                                        4f, 5f, 6f
                                    },
                                    new Shape(3, 3))
                            .toType(dataType, false);
            NDArray mask =
                    manager.create(
                            new int[] {
                                1, 0, 0, 1, 1, 0,
                                0, 1, 0, 0, 0, 1,
                                0, 0, 0, 0, 0, 0
                            },
                            new Shape(3, 3, 2));
            NDArray values =
                    manager.create(
                                    new float[] {
                                        1f,
                                        2f,
                                        3f,
                                        4f,
                                        5f,
                                        6f,
                                        7f,
                                        8f,
                                        9f,
                                        10f,
                                        11f,
                                        12f,
                                        Float.NaN,
                                        Float.POSITIVE_INFINITY,
                                        13f,
                                        14f,
                                        15f,
                                        16f
                                    },
                                    new Shape(3, 3, 2))
                            .toType(dataType, false);

            NDArray pooled = NDArrays.groupedMaskedSoftmaxPool(logits, mask, values);

            Assert.assertEquals(pooled.getShape(), new Shape(2, 3, 2));
            Assert.assertEquals(pooled.getDataType(), DataType.FLOAT32);
            assertClose(
                    pooled.toFloatArray(),
                    new float[] {
                        4f, 5f, 0f, 0f, 0f, 0f,
                        3f, 4f, 10.523188f, 11.523188f, 0f, 0f
                    },
                    tolerance);
        }
    }

    private static void verifyMaskedGradientParity(
            Engine engine, DataType dataType, float tolerance) {
        float[] values = {
            0.2f, -0.7f, 1.1f, 0.4f,
            -1.0f, 0.3f, 0.8f, -0.2f
        };
        int[] maskValues = {1, 0, 1, 1, 0, 0, 0, 0};
        float[] weights = {0.1f, 0.2f, -0.3f, 0.7f, 0.5f, -0.4f, 0.9f, -0.2f};
        float[] cpuProbabilities;
        float[] cpuNormalizers;
        float[] cpuGradients;
        try (NDManager manager = engine.newBaseManager(Device.cpu());
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray logits = manager.create(values, new Shape(2, 4));
            logits.setRequiresGradient(true);
            NDArray mask = manager.create(maskValues, new Shape(2, 4));
            NDArray lossWeights = manager.create(weights, new Shape(2, 4));
            NDArray probabilities = NDArrays.maskedSoftmax(logits, mask, -1);
            NDArray normalizers = NDArrays.maskedLogSumExp(logits, mask, -1);
            collector.backward(probabilities.mul(lossWeights).sum().add(normalizers.sum()));
            cpuProbabilities = probabilities.toFloatArray();
            cpuNormalizers = normalizers.toFloatArray();
            cpuGradients = logits.getGradient().toFloatArray();
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu());
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray logits = manager.create(values, new Shape(2, 4)).toType(dataType, false);
            logits.setRequiresGradient(true);
            NDArray mask = manager.create(maskValues, new Shape(2, 4));
            NDArray lossWeights = manager.create(weights, new Shape(2, 4));
            NDArray probabilities = NDArrays.maskedSoftmax(logits, mask, -1);
            NDArray normalizers = NDArrays.maskedLogSumExp(logits, mask, -1);
            collector.backward(probabilities.mul(lossWeights).sum().add(normalizers.sum()));
            assertClose(probabilities.toFloatArray(), cpuProbabilities, tolerance);
            assertClose(normalizers.toFloatArray(), cpuNormalizers, tolerance);
            try (NDArray gradient = logits.getGradient().toType(DataType.FLOAT32, false)) {
                assertClose(gradient.toFloatArray(), cpuGradients, tolerance);
            }
        }
    }

    private static void assertClose(float[] actual, float[] expected, float tolerance) {
        Assert.assertEquals(actual.length, expected.length);
        for (int index = 0; index < actual.length; index++) {
            Assert.assertEquals(actual[index], expected[index], tolerance, "index=" + index);
        }
    }
}
