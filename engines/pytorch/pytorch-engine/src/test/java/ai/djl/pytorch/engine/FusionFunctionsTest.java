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
import ai.djl.engine.fusion.FusionFunctions;
import ai.djl.engine.fusion.FusionRecipe;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.core.Linear;
import ai.djl.training.GradientCollector;

import org.testng.Assert;
import org.testng.annotations.Test;

/** Verifies differentiable functional counterparts of fusion stages. */
public class FusionFunctionsTest {

    @Test
    public void outputPackPreservesSourceGradients() {
        forEachDevice(this::verifyOutputPack);
    }

    @Test
    public void differentiableCastPreservesSourceGradient() {
        forEachDevice(this::verifyDifferentiableCast);
    }

    @Test
    public void segmentedOutputPackScattersOnlySelectedGradients() {
        forEachDevice(this::verifySegmentedOutputPack);
    }

    @Test
    public void binaryBranchBlendMatchesComposedForwardAndBackward() {
        forEachDevice(this::verifyBinaryBranchBlend);
    }

    @Test
    public void binaryBranchBlendDoesNotReadAbsentInputs() {
        forEachDevice(this::verifyBinaryBranchBlendNonFiniteIsolation);
    }

    @Test
    public void affineSumMatchesComposedForwardAndBackward() {
        forEachDevice(this::verifyAffineSum);
    }

    @Test
    public void outputPacksRejectNonDifferentiableDataTypes() {
        try (NDManager manager = Engine.getInstance().newBaseManager(Device.cpu())) {
            NDArray floating = manager.ones(new Shape(1, 1));
            NDArray integer = manager.ones(new Shape(1, 1), DataType.INT32);
            Assert.expectThrows(
                    IllegalArgumentException.class,
                    () -> FusionFunctions.outputPack(new NDList(floating), DataType.INT32));
            Assert.expectThrows(
                    IllegalArgumentException.class,
                    () ->
                            FusionFunctions.segmentedOutputPack(
                                    new NDList(integer),
                                    new long[] {0},
                                    new long[] {1},
                                    DataType.FLOAT32));
        }
    }

    @Test
    public void segmentedOutputPackRejectsInvalidMetadata() {
        try (NDManager manager = Engine.getInstance().newBaseManager(Device.cpu())) {
            NDArray valid = manager.ones(new Shape(2, 3, 4));
            NDArray rankTwo = manager.ones(new Shape(2, 4));
            NDArray wrongBatch = manager.ones(new Shape(1, 3, 4));
            NDArray wrongWidth = manager.ones(new Shape(2, 3, 5));
            Assert.expectThrows(
                    IllegalArgumentException.class,
                    () ->
                            FusionFunctions.segmentedOutputPack(
                                    new NDList(valid),
                                    new long[0],
                                    new long[] {1},
                                    DataType.FLOAT32));
            for (long[] slice : new long[][] {{-1, 1}, {0, 0}, {2, 2}, {Long.MAX_VALUE, 1}}) {
                Assert.expectThrows(
                        IllegalArgumentException.class,
                        () ->
                                FusionFunctions.segmentedOutputPack(
                                        new NDList(valid),
                                        new long[] {slice[0]},
                                        new long[] {slice[1]},
                                        DataType.FLOAT32));
            }
            for (NDArray incompatible : new NDArray[] {rankTwo, wrongBatch, wrongWidth}) {
                Assert.expectThrows(
                        IllegalArgumentException.class,
                        () ->
                                FusionFunctions.segmentedOutputPack(
                                        new NDList(valid, incompatible),
                                        new long[] {0, 0},
                                        new long[] {1, 1},
                                        DataType.FLOAT32));
            }
        }
    }

    private void verifyOutputPack(Device device) {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray first = values(manager, new Shape(2, 2), 0.25f);
            NDArray second = values(manager, new Shape(2, 1), -0.4f);
            NDArray lossWeight = manager.create(new float[] {1, 2, 3, 4, 5, 6}, new Shape(2, 3));
            first.setRequiresGradient(true);
            second.setRequiresGradient(true);

            NDArray output;
            try (GradientCollector collector = engine.newGradientCollector()) {
                output = FusionFunctions.outputPack(new NDList(first, second), DataType.FLOAT32);
                collector.backward(output.mul(lossWeight).sum());
            }

            Assert.assertEquals(output.getShape(), new Shape(2, 3));
            assertClose(first.getGradient(), new float[] {1, 2, 4, 5}, 0.0f);
            assertClose(second.getGradient(), new float[] {3, 6}, 0.0f);
        }
    }

    private void verifyDifferentiableCast(Device device) {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray source = values(manager, new Shape(2, 3), 0.25f);
            source.setRequiresGradient(true);

            NDArray converted;
            try (GradientCollector collector = engine.newGradientCollector()) {
                converted = source.getNDArrayInternal().differentiableCast(DataType.BFLOAT16);
                collector.backward(converted.mul(2f).sum());
            }

            Assert.assertEquals(converted.getDataType(), DataType.BFLOAT16);
            assertClose(source.getGradient(), new float[] {2, 2, 2, 2, 2, 2}, 0.0f);
        }
    }

    private void verifySegmentedOutputPack(Device device) {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray grouped = values(manager, new Shape(2, 2, 4, 2), 0.1f);
            NDArray plain = values(manager, new Shape(2, 3, 2), -0.05f);
            grouped.setRequiresGradient(true);
            plain.setRequiresGradient(true);

            NDArray output;
            try (GradientCollector collector = engine.newGradientCollector()) {
                output =
                        FusionFunctions.segmentedOutputPack(
                                new NDList(grouped, plain),
                                new long[] {1, 0},
                                new long[] {2, 1},
                                DataType.FLOAT32);
                collector.backward(output.sum());
            }

            Assert.assertEquals(output.getShape(), new Shape(2, 5, 2));
            float[] groupedGradient = grouped.getGradient().toFloatArray();
            for (int batch = 0; batch < 2; ++batch) {
                for (int group = 0; group < 2; ++group) {
                    for (int token = 0; token < 4; ++token) {
                        for (int hidden = 0; hidden < 2; ++hidden) {
                            int index = (((batch * 2 + group) * 4 + token) * 2) + hidden;
                            Assert.assertEquals(
                                    groupedGradient[index], token == 1 || token == 2 ? 1.0f : 0.0f);
                        }
                    }
                }
            }
            float[] plainGradient = plain.getGradient().toFloatArray();
            for (int batch = 0; batch < 2; ++batch) {
                for (int token = 0; token < 3; ++token) {
                    for (int hidden = 0; hidden < 2; ++hidden) {
                        int index = ((batch * 3 + token) * 2) + hidden;
                        Assert.assertEquals(plainGradient[index], token == 0 ? 1.0f : 0.0f);
                    }
                }
            }
        }
    }

    private void verifyBinaryBranchBlend(Device device) {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(device)) {
            NDList source =
                    new NDList(
                            values(manager, new Shape(3, 4), 0.11f)
                                    .toType(DataType.BFLOAT16, false),
                            values(manager, new Shape(3, 4), -0.07f)
                                    .toType(DataType.BFLOAT16, false),
                            values(manager, new Shape(3, 1), 0.23f)
                                    .toType(DataType.BFLOAT16, false),
                            manager.create(new float[] {0.2f, 0.7f, 1.0f}, new Shape(3, 1))
                                    .toType(DataType.BFLOAT16, false),
                            manager.create(new float[] {0.8f, 0.4f, 0.6f}, new Shape(3, 1))
                                    .toType(DataType.BFLOAT16, false));
            NDArray lossWeight = values(manager, new Shape(3, 4), 0.17f);
            TrainingResult actual = trainBinary(source, lossWeight, true);
            TrainingResult expected = trainBinary(source, lossWeight, false);
            assertResult(actual, expected, 2e-5f);
        }
    }

    private void verifyBinaryBranchBlendNonFiniteIsolation(Device device) {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(device)) {
            NDList inputs =
                    new NDList(
                            manager.create(
                                    new float[] {
                                        Float.NaN,
                                        Float.POSITIVE_INFINITY,
                                        5f,
                                        6f,
                                        Float.NaN,
                                        Float.NEGATIVE_INFINITY
                                    },
                                    new Shape(3, 2)),
                            manager.create(
                                    new float[] {
                                        3f,
                                        4f,
                                        Float.NaN,
                                        Float.POSITIVE_INFINITY,
                                        Float.NaN,
                                        Float.NEGATIVE_INFINITY
                                    },
                                    new Shape(3, 2)),
                            manager.create(
                                    new float[] {Float.NaN, Float.NaN, Float.NaN}, new Shape(3, 1)),
                            manager.create(new float[] {0f, 1f, 0f}, new Shape(3, 1)),
                            manager.create(new float[] {1f, 0f, 0f}, new Shape(3, 1)));
            inputs.forEach(array -> array.setRequiresGradient(true));
            NDArray output;
            try (GradientCollector collector = engine.newGradientCollector()) {
                output =
                        FusionFunctions.binaryBranchBlend(
                                inputs.get(0),
                                inputs.get(1),
                                inputs.get(2),
                                inputs.get(3),
                                inputs.get(4));
                collector.backward(output.sum());
            }

            assertClose(output, new float[] {3f, 4f, 5f, 6f, 0f, 0f}, 0f);
            for (NDArray input : inputs) {
                Assert.assertTrue(input.hasGradient());
                assertFinite(input.getGradient());
            }

            NDArray presentBaseline = manager.create(new float[] {Float.NaN, 2f}, new Shape(2, 1));
            NDArray presentSelected = manager.create(new float[] {3f, Float.NaN}, new Shape(2, 1));
            NDArray saturatedLogit =
                    manager.create(
                            new float[] {Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY},
                            new Shape(2, 1));
            NDArray presence = manager.ones(new Shape(2, 1));
            NDArray saturatedOutput =
                    FusionFunctions.binaryBranchBlend(
                            presentBaseline, presentSelected, saturatedLogit, presence, presence);
            for (float value : saturatedOutput.toFloatArray()) {
                Assert.assertTrue(Float.isNaN(value));
            }
        }
    }

    private void verifyAffineSum(Device device) {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(device)) {
            NDList inputs =
                    new NDList(
                            values(manager, new Shape(2, 3, 2), 0.09f),
                            values(manager, new Shape(1, 3, 3), -0.04f));
            NDList weights =
                    new NDList(
                            values(manager, new Shape(4, 2), 0.07f),
                            values(manager, new Shape(4, 3), -0.03f));
            NDArray bias = values(manager, new Shape(4), 0.02f);
            NDArray lossWeight = values(manager, new Shape(2, 3, 4), 0.13f);
            TrainingResult actual = trainAffine(inputs, weights, bias, lossWeight, true);
            TrainingResult expected = trainAffine(inputs, weights, bias, lossWeight, false);
            assertResult(actual, expected, 2e-5f);
        }
    }

    private static TrainingResult trainBinary(
            NDList source, NDArray lossWeight, boolean fusionFunction) {
        NDList inputs = duplicateForTraining(source);
        NDArray output;
        try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
            if (fusionFunction) {
                output =
                        FusionFunctions.binaryBranchBlend(
                                inputs.get(0),
                                inputs.get(1),
                                inputs.get(2),
                                inputs.get(3),
                                inputs.get(4));
            } else {
                NDArray baseline = differentiableCast(inputs.get(0), DataType.FLOAT32);
                NDArray selected = differentiableCast(inputs.get(1), DataType.FLOAT32);
                NDArray probability =
                        Activation.sigmoid(differentiableCast(inputs.get(2), DataType.FLOAT32));
                NDArray baselinePresence = differentiableCast(inputs.get(3), DataType.FLOAT32);
                NDArray selectedPresence = differentiableCast(inputs.get(4), DataType.FLOAT32);
                NDArray joint = baselinePresence.mul(selectedPresence);
                output =
                        baseline.mul(baselinePresence.sub(joint.mul(probability)))
                                .add(
                                        selected.mul(
                                                selectedPresence
                                                        .sub(joint)
                                                        .add(joint.mul(probability))));
            }
            collector.backward(output.mul(lossWeight).sum());
        }
        return result(output, inputs);
    }

    private static TrainingResult trainAffine(
            NDList sourceInputs,
            NDList sourceWeights,
            NDArray sourceBias,
            NDArray lossWeight,
            boolean fusionFunction) {
        NDList inputs = duplicateForTraining(sourceInputs);
        NDList weights = duplicateForTraining(sourceWeights);
        NDArray bias = sourceBias.duplicate();
        bias.setRequiresGradient(true);
        NDArray output;
        try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
            if (fusionFunction) {
                output =
                        FusionFunctions.affineSum(
                                inputs, weights, bias, FusionRecipe.Activation.SILU);
            } else {
                output = bias;
                for (int index = 0; index < inputs.size(); ++index) {
                    output =
                            output.add(
                                    Linear.linear(inputs.get(index), weights.get(index), null)
                                            .singletonOrThrow());
                }
                output = Activation.swish(output, 1.0f);
            }
            collector.backward(output.mul(lossWeight).sum());
        }
        NDList differentiable = new NDList(inputs.size() + weights.size() + 1);
        differentiable.addAll(inputs);
        differentiable.addAll(weights);
        differentiable.add(bias);
        return result(output, differentiable);
    }

    private static NDList duplicateForTraining(NDList source) {
        NDList result = new NDList(source.size());
        for (NDArray array : source) {
            NDArray duplicate = array.duplicate();
            duplicate.setRequiresGradient(true);
            result.add(duplicate);
        }
        return result;
    }

    private static NDArray differentiableCast(NDArray array, DataType dataType) {
        return array.getDataType() == dataType
                ? array
                : array.getNDArrayInternal().differentiableCast(dataType);
    }

    private static TrainingResult result(NDArray output, NDList inputs) {
        NDList gradients = new NDList(inputs.size());
        for (NDArray input : inputs) {
            gradients.add(input.getGradient());
        }
        return new TrainingResult(output, gradients);
    }

    private static void assertResult(
            TrainingResult actual, TrainingResult expected, float tolerance) {
        assertClose(actual.output, expected.output, tolerance);
        Assert.assertEquals(actual.gradients.size(), expected.gradients.size());
        for (int index = 0; index < actual.gradients.size(); ++index) {
            assertClose(actual.gradients.get(index), expected.gradients.get(index), tolerance);
        }
    }

    private static NDArray values(NDManager manager, Shape shape, float scale) {
        float[] values = new float[Math.toIntExact(shape.size())];
        for (int index = 0; index < values.length; ++index) {
            values[index] = (index - values.length * 0.4f) * scale;
        }
        return manager.create(values, shape);
    }

    private static void assertClose(NDArray actual, NDArray expected, float tolerance) {
        assertClose(actual, expected.toType(DataType.FLOAT32, false).toFloatArray(), tolerance);
    }

    private static void assertClose(NDArray actual, float[] expected, float tolerance) {
        float[] values = actual.toType(DataType.FLOAT32, false).toFloatArray();
        Assert.assertEquals(values.length, expected.length);
        for (int index = 0; index < values.length; ++index) {
            Assert.assertEquals(values[index], expected[index], tolerance);
        }
    }

    private static void assertFinite(NDArray array) {
        for (float value : array.toType(DataType.FLOAT32, false).toFloatArray()) {
            Assert.assertTrue(Float.isFinite(value));
        }
    }

    private static void forEachDevice(DeviceTest test) {
        Engine engine = Engine.getInstance();
        test.run(Device.cpu());
        if (engine.getGpuCount() > 0) {
            test.run(Device.gpu());
        }
    }

    @FunctionalInterface
    private interface DeviceTest {
        void run(Device device);
    }

    private static final class TrainingResult {
        private final NDArray output;
        private final NDList gradients;

        private TrainingResult(NDArray output, NDList gradients) {
            this.output = output;
            this.gradients = gradients;
        }
    }
}
