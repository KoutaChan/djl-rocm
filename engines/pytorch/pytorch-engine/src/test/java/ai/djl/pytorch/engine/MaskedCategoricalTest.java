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
    public void indexedMaskedSoftmaxPoolSelectsWithoutMaterializingAllChoices() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            NDArray logits =
                    manager.create(
                            new float[] {7f, 0f, 9f, 2f, (float) Math.log(3), 4f, 5f, 6f, 7f, 8f},
                            new Shape(2, 5));
            NDArray mask =
                    manager.create(
                            new int[] {
                                1, 1, 0, 0, 1,
                                1, 0, 1, 0, 0
                            },
                            new Shape(2, 5));
            NDArray values =
                    manager.create(
                            new float[] {
                                100f,
                                100f,
                                2f,
                                4f,
                                Float.NaN,
                                Float.POSITIVE_INFINITY,
                                6f,
                                8f,
                                10f,
                                12f,
                                1f,
                                2f,
                                Float.NaN,
                                Float.NaN,
                                3f,
                                4f,
                                Float.NaN,
                                Float.NaN,
                                Float.NaN,
                                Float.NaN
                            },
                            new Shape(2, 5, 2));

            NDArray pooled = NDArrays.indexedMaskedSoftmaxPool(logits, mask, values, 4, 1, 3);

            Assert.assertEquals(pooled.getShape(), new Shape(2, 2));
            Assert.assertEquals(pooled.getDataType(), DataType.FLOAT32);
            assertClose(pooled.toFloatArray(), new float[] {8f, 10f, 0f, 0f}, 1e-6f);
            Assert.expectThrows(
                    IllegalArgumentException.class,
                    () -> NDArrays.indexedMaskedSoftmaxPool(logits, mask, values, 1, 1));
        }
    }

    @Test
    public void indexedMaskedSoftmaxPoolNativeMatchesSelectedEagerReference() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        DataType[][] dataTypes = {
            {DataType.FLOAT32, DataType.FLOAT32, DataType.FLOAT32},
            {DataType.FLOAT16, DataType.FLOAT16, DataType.FLOAT16},
            {DataType.BFLOAT16, DataType.BFLOAT16, DataType.BFLOAT16},
            {DataType.FLOAT32, DataType.BOOLEAN, DataType.FLOAT16},
            {DataType.FLOAT32, DataType.BOOLEAN, DataType.BFLOAT16},
            {DataType.FLOAT16, DataType.FLOAT32, DataType.FLOAT32},
            {DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT32}
        };
        for (long batch : new long[] {384, 1408}) {
            for (int[] choices : new int[][] {{1, 4, 7}, {3, 8}}) {
                for (DataType[] types : dataTypes) {
                    verifyIndexedPoolGpuParity(
                            engine, batch, choices, types[0], types[1], types[2]);
                }
            }
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
    public void indexedMaskedSoftmaxPoolValueGradientMatchesCpuReference() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        verifyIndexedPoolValueGradientParity(engine, DataType.FLOAT32, 2e-5f);
        verifyIndexedPoolValueGradientParity(engine, DataType.BFLOAT16, 2e-2f);
        verifyProductionShapeIndexedPoolValueGradientParity(engine);
    }

    @Test
    public void indexedMaskedSoftmaxPoolValueGradientPreservesNonFiniteSemantics() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu());
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray logits =
                    manager.create(new float[] {Float.NaN, 0f, 9f, 1f, 2f, 3f}, new Shape(2, 3));
            NDArray mask =
                    manager.create(new boolean[] {true, true, false, false, false, false})
                            .reshape(2, 3);
            NDArray values =
                    manager.create(
                            new float[] {
                                1f, 2f, 3f, 4f, Float.NaN, Float.NaN, Float.NaN, Float.NaN,
                                Float.NaN, Float.NaN, Float.NaN, Float.NaN
                            },
                            new Shape(2, 3, 2));
            values.setRequiresGradient(true);
            NDArray pooled = NDArrays.indexedMaskedSoftmaxPool(logits, mask, values, 0, 1);
            collector.backward(pooled.sum());

            float[] output = pooled.toFloatArray();
            Assert.assertTrue(Float.isNaN(output[0]));
            Assert.assertTrue(Float.isNaN(output[1]));
            Assert.assertEquals(output[2], 0f);
            Assert.assertEquals(output[3], 0f);
            float[] gradient = values.getGradient().toFloatArray();
            for (int index = 0; index < 4; ++index) {
                Assert.assertTrue(Float.isNaN(gradient[index]), "index=" + index);
            }
            for (int index = 4; index < gradient.length; ++index) {
                Assert.assertEquals(gradient[index], 0f, "index=" + index);
            }
        }
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

    private static void verifyIndexedPoolGpuParity(
            Engine engine,
            long batch,
            int[] choices,
            DataType logitType,
            DataType maskType,
            DataType valueType) {
        final int choiceCount = 10;
        final int featureCount = 256;
        int rows = Math.toIntExact(batch);
        float[] logitData = new float[rows * choiceCount];
        float[] maskData = new float[rows * choiceCount];
        float[] valueData = new float[rows * choiceCount * featureCount];
        for (int row = 0; row < rows; ++row) {
            for (int choice = 0; choice < choiceCount; ++choice) {
                int choiceOffset = row * choiceCount + choice;
                logitData[choiceOffset] = ((row * 17 + choice * 7) % 23 - 11) * 0.125f;
                maskData[choiceOffset] = (row + choice * 3) % 5 == 0 ? 0f : 1f;
                for (int feature = 0; feature < featureCount; ++feature) {
                    int valueOffset = choiceOffset * featureCount + feature;
                    valueData[valueOffset] =
                            ((row * 13 + choice * 19 + feature * 3) % 101 - 50) * 0.03125f;
                }
            }
            if (row % 37 == 0) {
                for (int choice : choices) {
                    maskData[row * choiceCount + choice] = 0f;
                }
            }
        }

        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            NDArray logits =
                    manager.create(logitData, new Shape(batch, choiceCount))
                            .toType(logitType, false);
            NDArray mask =
                    manager.create(maskData, new Shape(batch, choiceCount)).toType(maskType, false);
            NDArray values =
                    manager.create(valueData, new Shape(batch, choiceCount, featureCount))
                            .toType(valueType, false);
            NDArray actual = NDArrays.indexedMaskedSoftmaxPool(logits, mask, values, choices);

            ai.djl.ndarray.NDList selectedLogits = new ai.djl.ndarray.NDList(choices.length);
            ai.djl.ndarray.NDList selectedMasks = new ai.djl.ndarray.NDList(choices.length);
            ai.djl.ndarray.NDList selectedValues = new ai.djl.ndarray.NDList(choices.length);
            for (int choice : choices) {
                selectedLogits.add(logits.get("...,{}", choice).expandDims(-1));
                selectedMasks.add(mask.get("...,{}", choice).expandDims(-1));
                selectedValues.add(values.get("...,{},:", choice).expandDims(-2));
            }
            NDArray selectedLogitArray = NDArrays.concat(selectedLogits, -1);
            NDArray selectedMaskArray = NDArrays.concat(selectedMasks, -1);
            NDArray weights = NDArrays.maskedSoftmax(selectedLogitArray, selectedMaskArray, -1);
            NDArray expected =
                    NDArrays.concat(selectedValues, -2)
                            .toType(DataType.FLOAT32, false)
                            .mul(weights.expandDims(-1))
                            .sum(new int[] {1});
            NDArray present =
                    selectedMaskArray
                            .neq(0)
                            .sum(new int[] {1})
                            .gt(0)
                            .expandDims(-1)
                            .broadcast(expected.getShape());
            expected = NDArrays.where(present, expected, expected.zerosLike());

            float[] actualData = actual.toFloatArray();
            float[] expectedData = expected.toFloatArray();
            Assert.assertEquals(actualData.length, expectedData.length);
            float maxAbs = 0f;
            for (int index = 0; index < actualData.length; ++index) {
                Assert.assertTrue(Float.isFinite(actualData[index]), "actual index=" + index);
                Assert.assertTrue(Float.isFinite(expectedData[index]), "expected index=" + index);
                maxAbs = Math.max(maxAbs, Math.abs(actualData[index] - expectedData[index]));
            }
            float tolerance =
                    logitType == DataType.BFLOAT16 || valueType == DataType.BFLOAT16
                            ? 2e-2f
                            : logitType == DataType.FLOAT16 || valueType == DataType.FLOAT16
                                    ? 2e-3f
                                    : 2e-5f;
            Assert.assertTrue(maxAbs <= tolerance, "maxAbs=" + maxAbs + ", tolerance=" + tolerance);
            System.out.printf(
                    java.util.Locale.ROOT,
                    "INDEXED_MASKED_POOL_PARITY batch=%d choices=%d logits=%s mask=%s values=%s"
                            + " maxAbs=%.9g tolerance=%.9g%n",
                    batch,
                    choices.length,
                    logitType,
                    maskType,
                    valueType,
                    maxAbs,
                    tolerance);
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

    private static void verifyIndexedPoolValueGradientParity(
            Engine engine, DataType dataType, float tolerance) {
        float[] logitsData = {
            5f, -0.25f, 7f, 0.5f, 1.25f,
            -3f, 8f, 2f, -0.75f, 4f,
            1f, 2f, 3f, 4f, 5f
        };
        int[] maskData = {
            1, 1, 0, 0, 1,
            1, 0, 1, 1, 0,
            1, 0, 1, 0, 0
        };
        float[] valueData = new float[3 * 5 * 3];
        for (int index = 0; index < valueData.length; ++index) {
            valueData[index] = (index % 17 - 8) * 0.125f;
        }
        float[] lossWeightData = {
            0.25f, -0.5f, 0.75f,
            -1f, 0.5f, 0.125f,
            0.75f, -0.25f, 0.625f
        };
        int[] choices = {4, 1, 3};
        float[] expectedOutput;
        float[] expectedGradient;
        try (NDManager manager = engine.newBaseManager(Device.cpu());
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray logits = manager.create(logitsData, new Shape(3, 5));
            NDArray mask = manager.create(maskData, new Shape(3, 5));
            NDArray values = manager.create(valueData, new Shape(3, 5, 3));
            values.setRequiresGradient(true);
            NDArray lossWeights = manager.create(lossWeightData, new Shape(3, 3));
            NDArray pooled = NDArrays.indexedMaskedSoftmaxPool(logits, mask, values, choices);
            collector.backward(pooled.mul(lossWeights).sum());
            expectedOutput = pooled.toFloatArray();
            expectedGradient = values.getGradient().toFloatArray();
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu());
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray logits = manager.create(logitsData, new Shape(3, 5)).toType(dataType, false);
            NDArray mask =
                    manager.create(maskData, new Shape(3, 5)).toType(DataType.BOOLEAN, false);
            NDArray values = manager.create(valueData, new Shape(3, 5, 3)).toType(dataType, false);
            values.setRequiresGradient(true);
            NDArray lossWeights = manager.create(lossWeightData, new Shape(3, 3));
            NDArray pooled = NDArrays.indexedMaskedSoftmaxPool(logits, mask, values, choices);
            collector.backward(pooled.mul(lossWeights).sum());
            assertClose(pooled.toFloatArray(), expectedOutput, tolerance);
            try (NDArray gradient = values.getGradient().toType(DataType.FLOAT32, false)) {
                assertClose(gradient.toFloatArray(), expectedGradient, tolerance);
            }
        }
    }

    private static void verifyProductionShapeIndexedPoolValueGradientParity(Engine engine) {
        final int rows = 384;
        final int choiceCount = 10;
        final int featureCount = 256;
        int[] choices = {1, 4, 7};
        float[] logitsData = new float[rows * choiceCount];
        float[] maskData = new float[rows * choiceCount];
        float[] valueData = new float[rows * choiceCount * featureCount];
        float[] lossWeightData = new float[rows * featureCount];
        for (int row = 0; row < rows; ++row) {
            for (int choice = 0; choice < choiceCount; ++choice) {
                int choiceOffset = row * choiceCount + choice;
                logitsData[choiceOffset] = ((row * 11 + choice * 7) % 29 - 14) * 0.125f;
                maskData[choiceOffset] = (row + choice * 3) % 7 == 0 ? 0f : 1f;
                for (int feature = 0; feature < featureCount; ++feature) {
                    valueData[choiceOffset * featureCount + feature] =
                            ((row * 13 + choice * 17 + feature * 5) % 61 - 30) * 0.03125f;
                }
            }
            if (row % 37 == 0) {
                for (int choice : choices) {
                    maskData[row * choiceCount + choice] = 0f;
                }
            }
            for (int feature = 0; feature < featureCount; ++feature) {
                lossWeightData[row * featureCount + feature] =
                        ((row * 19 + feature * 3) % 43 - 21) * 0.0625f;
            }
        }

        float[] expectedOutput;
        float[] expectedGradient;
        try (NDManager manager = engine.newBaseManager(Device.cpu());
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray logits = manager.create(logitsData, new Shape(rows, choiceCount));
            NDArray mask = manager.create(maskData, new Shape(rows, choiceCount));
            NDArray values = manager.create(valueData, new Shape(rows, choiceCount, featureCount));
            values.setRequiresGradient(true);
            NDArray lossWeights = manager.create(lossWeightData, new Shape(rows, featureCount));
            NDArray pooled = NDArrays.indexedMaskedSoftmaxPool(logits, mask, values, choices);
            collector.backward(pooled.transpose(1, 0).mul(lossWeights.transpose(1, 0)).sum());
            expectedOutput = pooled.toFloatArray();
            expectedGradient = values.getGradient().toFloatArray();
        }

        for (DataType dataType : new DataType[] {DataType.FLOAT32, DataType.BFLOAT16}) {
            float tolerance = dataType == DataType.BFLOAT16 ? 2e-2f : 2e-5f;
            try (NDManager manager = engine.newBaseManager(Device.gpu());
                    GradientCollector collector = engine.newGradientCollector()) {
                NDArray logits =
                        manager.create(logitsData, new Shape(rows, choiceCount))
                                .toType(dataType, false);
                NDArray mask =
                        manager.create(maskData, new Shape(rows, choiceCount))
                                .toType(DataType.BOOLEAN, false);
                NDArray values =
                        manager.create(valueData, new Shape(rows, choiceCount, featureCount))
                                .toType(dataType, false);
                values.setRequiresGradient(true);
                NDArray lossWeights = manager.create(lossWeightData, new Shape(rows, featureCount));
                NDArray pooled = NDArrays.indexedMaskedSoftmaxPool(logits, mask, values, choices);
                collector.backward(pooled.transpose(1, 0).mul(lossWeights.transpose(1, 0)).sum());
                assertClose(pooled.toFloatArray(), expectedOutput, tolerance);
                try (NDArray gradient = values.getGradient().toType(DataType.FLOAT32, false)) {
                    assertClose(gradient.toFloatArray(), expectedGradient, tolerance);
                }
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
