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
    public void maskedNonFiniteEntriesDoNotParticipate() {
        Engine engine = Engine.getInstance();
        verifyMaskedNonFiniteIsolation(engine, Device.cpu());
        if (engine.getGpuCount() > 0) {
            verifyMaskedNonFiniteIsolation(engine, Device.gpu());
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
    public void logSumExpGradientIsInvariantToLargeCommonOffsets() {
        Engine engine = Engine.getInstance();
        verifyLogSumExpLargeOffsets(engine, Device.cpu());
        if (engine.getGpuCount() > 0) {
            verifyLogSumExpLargeOffsets(engine, Device.gpu());
        }
    }

    @Test
    public void backwardNonFiniteMaskSemanticsMatchPortableReference() {
        Engine engine = Engine.getInstance();
        verifyBackwardNonFiniteMaskSemantics(engine, Device.cpu());
        if (engine.getGpuCount() > 0) {
            verifyBackwardNonFiniteMaskSemantics(engine, Device.gpu());
        }
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
    public void groupedMaskedSoftmaxPoolValueGradientMatchesCpuReference() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        verifyGroupedPoolValueGradientParity(engine, DataType.FLOAT32, 2e-5f);
        verifyGroupedPoolValueGradientParity(engine, DataType.FLOAT16, 3e-3f);
        verifyGroupedPoolValueGradientParity(engine, DataType.BFLOAT16, 3e-2f);
    }

    @Test
    public void groupedMaskedSoftmaxPoolValueGradientSupportsStridedInputs() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        verifyStridedGroupedPoolValueGradientParity(engine);
    }

    @Test
    public void groupedMaskedSoftmaxPoolValueGradientUsesPortableFallbacks() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        verifyGroupedPoolLogitAndValueGradientParity(engine);
        verifyUnsupportedGroupedPoolValueGradientParity(engine, 65, 3);
        verifyUnsupportedGroupedPoolValueGradientParity(engine, 4, 32768);
    }

    @Test
    public void groupedMaskedSoftmaxPoolValueGradientSupportsEmptyRows() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu());
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray logits = manager.zeros(new Shape(0, 3));
            NDArray mask = manager.zeros(new Shape(0, 3, 2), DataType.BOOLEAN);
            NDArray values = manager.zeros(new Shape(0, 3, 4));
            values.setRequiresGradient(true);
            NDArray pooled = NDArrays.groupedMaskedSoftmaxPool(logits, mask, values);
            Assert.assertEquals(pooled.getShape(), new Shape(2, 0, 4));
            collector.backward(pooled.sum());
            Assert.assertTrue(values.hasGradient());
            try (NDArray gradient = values.getGradient()) {
                Assert.assertEquals(gradient.getShape(), values.getShape());
                Assert.assertEquals(gradient.size(), 0L);
            }
        }
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

    private static void verifyMaskedNonFiniteIsolation(Engine engine, Device device) {
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray logits = manager.create(new float[] {0f, Float.NaN}, new Shape(1, 2));
            NDArray mask = manager.create(new int[] {1, 0}, new Shape(1, 2));
            assertClose(
                    NDArrays.maskedSoftmax(logits, mask, -1).toFloatArray(),
                    new float[] {1f, 0f},
                    0f);

            NDArray normalizerLogits = manager.create(new float[] {0f, Float.NaN}, new Shape(1, 2));
            normalizerLogits.setRequiresGradient(true);
            NDArray normalizer;
            try (GradientCollector collector = engine.newGradientCollector()) {
                normalizer = NDArrays.maskedLogSumExp(normalizerLogits, mask, -1);
                collector.backward(normalizer.sum());
            }
            assertClose(normalizer.toFloatArray(), new float[] {0f}, 0f);
            assertClose(normalizerLogits.getGradient().toFloatArray(), new float[] {1f, 0f}, 0f);

            NDArray extremeLogits =
                    manager.create(new float[] {-Float.MAX_VALUE, 0f}, new Shape(1, 2));
            extremeLogits.setRequiresGradient(true);
            NDArray extremeProbabilities;
            NDArray extremeNormalizer;
            try (GradientCollector collector = engine.newGradientCollector()) {
                extremeProbabilities = NDArrays.maskedSoftmax(extremeLogits, mask, -1);
                extremeNormalizer = NDArrays.maskedLogSumExp(extremeLogits, mask, -1);
                collector.backward(extremeProbabilities.sum().add(extremeNormalizer.sum()));
            }
            assertClose(extremeProbabilities.toFloatArray(), new float[] {1f, 0f}, 0f);
            Assert.assertEquals(extremeNormalizer.getFloat(), -Float.MAX_VALUE);
            assertClose(extremeLogits.getGradient().toFloatArray(), new float[] {1f, 0f}, 0f);

            NDArray poolLogits = manager.zeros(new Shape(1, 2));
            NDArray poolMask = manager.create(new int[] {0, 1}, new Shape(1, 2));
            NDArray values = manager.create(new float[] {Float.NaN, 2f}, new Shape(1, 2, 1));
            NDArray groupedMask = poolMask.expandDims(-1);
            NDArray grouped = NDArrays.groupedMaskedSoftmaxPool(poolLogits, groupedMask, values);
            NDArray indexed = NDArrays.indexedMaskedSoftmaxPool(poolLogits, poolMask, values, 0, 1);
            assertClose(grouped.toFloatArray(), new float[] {2f}, 0f);
            assertClose(indexed.toFloatArray(), new float[] {2f}, 0f);

            NDArray underflowLogits = manager.create(new float[] {0f, -1000f}, new Shape(1, 2));
            NDArray legalMask = manager.ones(new Shape(1, 2));
            NDArray legalValues = manager.create(new float[] {1f, Float.NaN}, new Shape(1, 2, 1));
            NDArray legalGrouped =
                    NDArrays.groupedMaskedSoftmaxPool(
                            underflowLogits, legalMask.expandDims(-1), legalValues);
            NDArray legalIndexed =
                    NDArrays.indexedMaskedSoftmaxPool(
                            underflowLogits, legalMask, legalValues, 0, 1);
            Assert.assertTrue(Float.isNaN(legalGrouped.getFloat()));
            Assert.assertTrue(Float.isNaN(legalIndexed.getFloat()));
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

    private static void verifyLogSumExpLargeOffsets(Engine engine, Device device) {
        for (DataType dataType : new DataType[] {DataType.FLOAT32, DataType.BFLOAT16}) {
            for (int width : new int[] {3, 34}) {
                for (float offset : new float[] {0f, 1e8f, 1e20f}) {
                    try (NDManager manager = engine.newBaseManager(device);
                            GradientCollector collector = engine.newGradientCollector()) {
                        float[] values = new float[3 * width];
                        boolean[] maskValues = new boolean[values.length];
                        float[] expected = new float[values.length];
                        for (int column = 0; column < width; column++) {
                            values[column] = offset;
                            values[width + column] = -offset;
                            values[2 * width + column] = offset;
                            maskValues[column] = column == 0 || column == width - 1;
                            maskValues[width + column] = true;
                            expected[column] = maskValues[column] ? 1f : 0f;
                            expected[width + column] = -3f / width;
                        }
                        NDArray logits =
                                manager.create(values, new Shape(3, width)).toType(dataType, false);
                        logits.setRequiresGradient(true);
                        NDArray mask = manager.create(maskValues, logits.getShape());
                        NDArray normalizers = NDArrays.maskedLogSumExp(logits, mask, -1);
                        NDArray upstream =
                                manager.create(new float[] {2f, -3f, 4f}, new Shape(3, 1));
                        collector.backward(normalizers.mul(upstream).sum());
                        Assert.assertEquals(normalizers.toFloatArray()[2], 0f);
                        try (NDArray gradient =
                                logits.getGradient().toType(DataType.FLOAT32, false)) {
                            assertClose(
                                    gradient.toFloatArray(),
                                    expected,
                                    dataType == DataType.BFLOAT16 ? 1e-3f : 1e-6f);
                        }
                    }
                }
            }
        }
    }

    private static void verifyBackwardNonFiniteMaskSemantics(Engine engine, Device device) {
        try (NDManager manager = engine.newBaseManager(device);
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray logits = manager.create(new float[] {0f, 2f}, new Shape(1, 2));
            logits.setRequiresGradient(true);
            NDArray mask = manager.create(new boolean[] {true, false}, new Shape(1, 2));
            NDArray lossWeights = manager.create(new float[] {1f, Float.NaN}, new Shape(1, 2));
            NDArray probabilities = NDArrays.maskedSoftmax(logits, mask, -1);
            collector.backward(probabilities.mul(lossWeights).sum());
            assertClose(logits.getGradient().toFloatArray(), new float[] {0f, 0f}, 0f);
        }

        try (NDManager manager = engine.newBaseManager(device);
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray logits = manager.create(new float[] {0f, 1f}, new Shape(1, 2));
            NDArray mask =
                    manager.create(new boolean[] {true, false, false, false}, new Shape(1, 2, 2));
            NDArray values = manager.create(new float[] {3f, 5f}, new Shape(1, 2, 1));
            values.setRequiresGradient(true);
            NDArray lossWeights = manager.create(new float[] {1f, Float.NaN}, new Shape(2, 1, 1));
            NDArray pooled = NDArrays.groupedMaskedSoftmaxPool(logits, mask, values);
            collector.backward(pooled.mul(lossWeights).sum());
            assertClose(values.getGradient().toFloatArray(), new float[] {1f, 0f}, 0f);
        }

        try (NDManager manager = engine.newBaseManager(device);
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray logits = manager.create(new float[] {0f, -1000f}, new Shape(1, 2));
            NDArray mask = manager.create(new boolean[] {true, true}, new Shape(1, 2));
            NDArray values = manager.create(new float[] {3f, 5f}, new Shape(1, 2, 1));
            values.setRequiresGradient(true);
            NDArray pooled = NDArrays.indexedMaskedSoftmaxPool(logits, mask, values, 0, 1);
            collector.backward(pooled.mul(Float.NaN));
            float[] gradients = values.getGradient().toFloatArray();
            Assert.assertTrue(Float.isNaN(gradients[0]));
            Assert.assertTrue(Float.isNaN(gradients[1]));
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

    private static void verifyGroupedPoolValueGradientParity(
            Engine engine, DataType dataType, float tolerance) {
        final int rowCount = 4;
        final int choiceCount = 4;
        final int groupCount = 3;
        final int featureCount = 3;
        float[] logitsData = new float[rowCount * choiceCount];
        boolean[] maskData = new boolean[rowCount * choiceCount * groupCount];
        float[] valueData = new float[rowCount * choiceCount * featureCount];
        float[] lossWeightData = new float[groupCount * rowCount * featureCount];
        for (int row = 0; row < rowCount; ++row) {
            for (int choice = 0; choice < choiceCount; ++choice) {
                logitsData[row * choiceCount + choice] = ((row * 7 + choice * 5) % 13 - 6) * 0.25f;
                for (int group = 0; group < groupCount; ++group) {
                    boolean legal = (row + choice * 2 + group) % 3 != 0;
                    if ((row == 1 && group == 2) || row == 3) {
                        legal = false;
                    }
                    maskData[(row * choiceCount + choice) * groupCount + group] = legal;
                }
                for (int feature = 0; feature < featureCount; ++feature) {
                    valueData[(row * choiceCount + choice) * featureCount + feature] =
                            ((row * 11 + choice * 7 + feature * 3) % 23 - 11) * 0.125f;
                }
            }
        }
        for (int group = 0; group < groupCount; ++group) {
            for (int row = 0; row < rowCount; ++row) {
                for (int feature = 0; feature < featureCount; ++feature) {
                    lossWeightData[(group * rowCount + row) * featureCount + feature] =
                            ((group * 17 + row * 5 + feature * 11) % 19 - 9) * 0.1875f;
                }
            }
        }

        float[] expectedOutput;
        float[] expectedGradient;
        try (NDManager manager = engine.newBaseManager(Device.cpu());
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray logits = manager.create(logitsData, new Shape(2, 2, choiceCount));
            NDArray mask = manager.create(maskData, new Shape(2, 2, choiceCount, groupCount));
            NDArray values = manager.create(valueData, new Shape(2, 2, choiceCount, featureCount));
            values.setRequiresGradient(true);
            NDArray lossWeights =
                    manager.create(lossWeightData, new Shape(groupCount, 2, 2, featureCount));
            NDArray pooled = NDArrays.groupedMaskedSoftmaxPool(logits, mask, values);
            collector.backward(pooled.mul(lossWeights).sum());
            expectedOutput = pooled.toFloatArray();
            expectedGradient = values.getGradient().toFloatArray();
        }

        try (NDManager manager = engine.newBaseManager(Device.gpu());
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray logits =
                    manager.create(logitsData, new Shape(2, 2, choiceCount))
                            .toType(dataType, false);
            NDArray mask = manager.create(maskData, new Shape(2, 2, choiceCount, groupCount));
            NDArray values =
                    manager.create(valueData, new Shape(2, 2, choiceCount, featureCount))
                            .toType(dataType, false);
            values.setRequiresGradient(true);
            NDArray lossWeights =
                    manager.create(lossWeightData, new Shape(groupCount, 2, 2, featureCount));
            NDArray pooled = NDArrays.groupedMaskedSoftmaxPool(logits, mask, values);
            collector.backward(pooled.mul(lossWeights).sum());
            assertClose(pooled.toFloatArray(), expectedOutput, tolerance);
            try (NDArray gradient = values.getGradient().toType(DataType.FLOAT32, false)) {
                assertClose(gradient.toFloatArray(), expectedGradient, tolerance);
            }
        }
    }

    private static void verifyStridedGroupedPoolValueGradientParity(Engine engine) {
        final int rowCount = 3;
        final int choiceCount = 4;
        final int groupCount = 2;
        final int featureCount = 3;
        float[] logitBaseData = new float[choiceCount * rowCount];
        boolean[] maskBaseData = new boolean[rowCount * groupCount * choiceCount];
        float[] valueBaseData = new float[rowCount * featureCount * choiceCount];
        float[] lossWeightData = new float[rowCount * groupCount * featureCount];
        for (int index = 0; index < logitBaseData.length; ++index) {
            logitBaseData[index] = (index % 9 - 4) * 0.3125f;
        }
        for (int row = 0; row < rowCount; ++row) {
            for (int group = 0; group < groupCount; ++group) {
                for (int choice = 0; choice < choiceCount; ++choice) {
                    maskBaseData[(row * groupCount + group) * choiceCount + choice] =
                            row != 2 && (row + group + choice) % 3 != 0;
                }
            }
        }
        for (int index = 0; index < valueBaseData.length; ++index) {
            valueBaseData[index] = (index % 17 - 8) * 0.15625f;
        }
        for (int index = 0; index < lossWeightData.length; ++index) {
            lossWeightData[index] = (index % 11 - 5) * 0.21875f;
        }

        float[] expectedOutput;
        float[] expectedBaseGradient;
        try (NDManager manager = engine.newBaseManager(Device.cpu());
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray logits =
                    manager.create(logitBaseData, new Shape(choiceCount, rowCount)).transpose(1, 0);
            NDArray mask =
                    manager.create(maskBaseData, new Shape(rowCount, groupCount, choiceCount))
                            .transpose(0, 2, 1);
            NDArray valueBase =
                    manager.create(valueBaseData, new Shape(rowCount, featureCount, choiceCount));
            valueBase.setRequiresGradient(true);
            NDArray values = valueBase.transpose(0, 2, 1);
            NDArray lossWeights =
                    manager.create(lossWeightData, new Shape(rowCount, groupCount, featureCount));
            NDArray pooled = NDArrays.groupedMaskedSoftmaxPool(logits, mask, values);
            collector.backward(pooled.transpose(1, 0, 2).mul(lossWeights).sum());
            expectedOutput = pooled.toFloatArray();
            expectedBaseGradient = valueBase.getGradient().toFloatArray();
        }

        try (NDManager manager = engine.newBaseManager(Device.gpu());
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray logits =
                    manager.create(logitBaseData, new Shape(choiceCount, rowCount)).transpose(1, 0);
            NDArray mask =
                    manager.create(maskBaseData, new Shape(rowCount, groupCount, choiceCount))
                            .transpose(0, 2, 1);
            NDArray valueBase =
                    manager.create(valueBaseData, new Shape(rowCount, featureCount, choiceCount));
            valueBase.setRequiresGradient(true);
            NDArray values = valueBase.transpose(0, 2, 1);
            NDArray lossWeights =
                    manager.create(lossWeightData, new Shape(rowCount, groupCount, featureCount));
            NDArray pooled = NDArrays.groupedMaskedSoftmaxPool(logits, mask, values);
            collector.backward(pooled.transpose(1, 0, 2).mul(lossWeights).sum());
            assertClose(pooled.toFloatArray(), expectedOutput, 2e-5f);
            assertClose(valueBase.getGradient().toFloatArray(), expectedBaseGradient, 2e-5f);
        }
    }

    private static void verifyGroupedPoolLogitAndValueGradientParity(Engine engine) {
        float[] logitsData = {0.25f, -0.5f, 1.0f, 0.75f, -1.25f, 0.5f};
        boolean[] maskData = {
            true, false, true, true, false, true,
            true, true, false, false, true, false
        };
        float[] valueData = {
            1f, 2f, 3f, 4f, 5f, 6f,
            -1f, -2f, -3f, -4f, -5f, -6f
        };
        float[] lossWeightData = {0.5f, -1f, 0.25f, 0.75f, -0.5f, 1.25f, -0.25f, 0.125f};
        float[] expectedLogitGradient;
        float[] expectedValueGradient;
        try (NDManager manager = engine.newBaseManager(Device.cpu());
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray logits = manager.create(logitsData, new Shape(2, 3));
            logits.setRequiresGradient(true);
            NDArray mask = manager.create(maskData, new Shape(2, 3, 2));
            NDArray values = manager.create(valueData, new Shape(2, 3, 2));
            values.setRequiresGradient(true);
            NDArray lossWeights = manager.create(lossWeightData, new Shape(2, 2, 2));
            NDArray pooled = NDArrays.groupedMaskedSoftmaxPool(logits, mask, values);
            collector.backward(pooled.mul(lossWeights).sum());
            expectedLogitGradient = logits.getGradient().toFloatArray();
            expectedValueGradient = values.getGradient().toFloatArray();
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu());
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray logits = manager.create(logitsData, new Shape(2, 3));
            logits.setRequiresGradient(true);
            NDArray mask = manager.create(maskData, new Shape(2, 3, 2));
            NDArray values = manager.create(valueData, new Shape(2, 3, 2));
            values.setRequiresGradient(true);
            NDArray lossWeights = manager.create(lossWeightData, new Shape(2, 2, 2));
            NDArray pooled = NDArrays.groupedMaskedSoftmaxPool(logits, mask, values);
            collector.backward(pooled.mul(lossWeights).sum());
            assertClose(logits.getGradient().toFloatArray(), expectedLogitGradient, 2e-5f);
            assertClose(values.getGradient().toFloatArray(), expectedValueGradient, 2e-5f);
        }
    }

    private static void verifyUnsupportedGroupedPoolValueGradientParity(
            Engine engine, int choiceCount, int groupCount) {
        final int featureCount = 2;
        float[] logitsData = new float[choiceCount];
        boolean[] maskData = new boolean[choiceCount * groupCount];
        float[] valueData = new float[choiceCount * featureCount];
        float[] lossWeightData = new float[groupCount * featureCount];
        for (int choice = 0; choice < choiceCount; ++choice) {
            logitsData[choice] = (choice % 13 - 6) * 0.125f;
            for (int group = 0; group < groupCount; ++group) {
                maskData[choice * groupCount + group] = (choice + group) % 5 != 0;
            }
            for (int feature = 0; feature < featureCount; ++feature) {
                valueData[choice * featureCount + feature] =
                        ((choice * 3 + feature * 7) % 19 - 9) * 0.0625f;
            }
        }
        for (int index = 0; index < lossWeightData.length; ++index) {
            lossWeightData[index] = (index % 17 - 8) * 0.09375f;
        }
        float[] expectedGradient;
        try (NDManager manager = engine.newBaseManager(Device.cpu());
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray logits = manager.create(logitsData, new Shape(1, choiceCount));
            NDArray mask = manager.create(maskData, new Shape(1, choiceCount, groupCount));
            NDArray values = manager.create(valueData, new Shape(1, choiceCount, featureCount));
            values.setRequiresGradient(true);
            NDArray lossWeights =
                    manager.create(lossWeightData, new Shape(groupCount, 1, featureCount));
            collector.backward(
                    NDArrays.groupedMaskedSoftmaxPool(logits, mask, values).mul(lossWeights).sum());
            expectedGradient = values.getGradient().toFloatArray();
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu());
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray logits = manager.create(logitsData, new Shape(1, choiceCount));
            NDArray mask = manager.create(maskData, new Shape(1, choiceCount, groupCount));
            NDArray values = manager.create(valueData, new Shape(1, choiceCount, featureCount));
            values.setRequiresGradient(true);
            NDArray lossWeights =
                    manager.create(lossWeightData, new Shape(groupCount, 1, featureCount));
            collector.backward(
                    NDArrays.groupedMaskedSoftmaxPool(logits, mask, values).mul(lossWeights).sum());
            assertClose(values.getGradient().toFloatArray(), expectedGradient, 2e-5f);
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
