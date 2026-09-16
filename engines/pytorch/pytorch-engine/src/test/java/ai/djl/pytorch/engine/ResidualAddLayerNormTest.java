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
import ai.djl.engine.Autocast;
import ai.djl.engine.Engine;
import ai.djl.engine.EngineException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.GradientCollector;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

/** Tests residual addition followed by LayerNorm on the portable and ROCm paths. */
@SuppressWarnings("try") // Autocast resources are used for their scope side effects.
public class ResidualAddLayerNormTest {

    private static final float EPSILON = 1.0e-5f;
    private static final int[] WIDTHS = {1, 17, 64, 256, 257, 384};
    private static final DataType[] AFFINE_TYPES = {DataType.FLOAT32, DataType.BFLOAT16};

    @Test
    public void cpuFallbackMatchesComposedOperationsAndPreservesInputs() {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(Device.cpu())) {
            for (int width : WIDTHS) {
                Shape shape = new Shape(3, 2, width);
                float[] residualValues = sequence(shape.size(), 0.03125f, 0);
                float[] updateValues = sequence(shape.size(), -0.015625f, 7);
                try (NDArray residual = manager.create(residualValues, shape);
                        NDArray update = manager.create(updateValues, shape);
                        NDArray gamma = manager.create(sequence(width, 0.00390625f, 3)).add(1f);
                        NDArray beta = manager.create(sequence(width, -0.001953125f, 11));
                        NDArray expectedSum = residual.add(update);
                        NDArray expectedNormalized =
                                LayerNorm.layerNorm(
                                                expectedSum, new Shape(width), gamma, beta, EPSILON)
                                        .singletonOrThrow();
                        NDList actual =
                                LayerNorm.residualAddLayerNorm(
                                        residual, update, new Shape(width), gamma, beta, EPSILON)) {
                    Assert.assertEquals(actual.size(), 2);
                    assertClose(actual.get(0), expectedNormalized, 2.0e-5f);
                    assertClose(actual.get(1), expectedSum, 0f);
                    Assert.assertEquals(floatValues(residual), residualValues, 0f);
                    Assert.assertEquals(floatValues(update), updateValues, 0f);
                }
            }
        }
    }

    @Test
    public void rocmAutogradMatchesEagerForMixedInputsAndAllWidths() {
        Engine engine = Engine.getInstance();
        requireRocm(engine);
        Device device = Device.gpu();
        try (NDManager manager = engine.newBaseManager(device)) {
            for (int width : WIDTHS) {
                for (DataType parameterType : AFFINE_TYPES) {
                    for (DataType residualType : AFFINE_TYPES) {
                        DataType updateType =
                                residualType == DataType.FLOAT32
                                        ? DataType.BFLOAT16
                                        : DataType.FLOAT32;
                        TrainingResult reference =
                                train(
                                        engine,
                                        manager,
                                        device,
                                        11,
                                        width,
                                        residualType,
                                        updateType,
                                        parameterType,
                                        OutputUse.BOTH,
                                        false);
                        TrainingResult fused =
                                train(
                                        engine,
                                        manager,
                                        device,
                                        11,
                                        width,
                                        residualType,
                                        updateType,
                                        parameterType,
                                        OutputUse.BOTH,
                                        true);
                        assertTrainingResult(fused, reference, 4.0e-3f);
                    }
                }
            }
        }
    }

    @Test
    public void rocmAutogradSupportsBothAndPartialOutputs() {
        Engine engine = Engine.getInstance();
        requireRocm(engine);
        Device device = Device.gpu();
        try (NDManager manager = engine.newBaseManager(device)) {
            for (OutputUse outputUse : OutputUse.values()) {
                for (DataType parameterType : AFFINE_TYPES) {
                    TrainingResult reference =
                            train(
                                    engine,
                                    manager,
                                    device,
                                    outputUse == OutputUse.BOTH ? 257 : 13,
                                    256,
                                    DataType.BFLOAT16,
                                    DataType.BFLOAT16,
                                    parameterType,
                                    outputUse,
                                    false);
                    TrainingResult fused =
                            train(
                                    engine,
                                    manager,
                                    device,
                                    outputUse == OutputUse.BOTH ? 257 : 13,
                                    256,
                                    DataType.BFLOAT16,
                                    DataType.BFLOAT16,
                                    parameterType,
                                    outputUse,
                                    true);
                    assertTrainingResult(fused, reference, 1.5e-2f);
                }
            }
        }
    }

    @Test
    public void rocmAutogradMatchesEagerAcrossChainedResidualBlocks() {
        Engine engine = Engine.getInstance();
        requireRocm(engine);
        Device device = Device.gpu();
        try (NDManager manager = engine.newBaseManager(device)) {
            for (DataType parameterType : AFFINE_TYPES) {
                float[][] reference = trainChain(engine, manager, device, parameterType, false);
                float[][] fused = trainChain(engine, manager, device, parameterType, true);
                Assert.assertEquals(fused.length, reference.length);
                for (int index = 0; index < fused.length; ++index) {
                    assertClose(fused[index], reference[index], 6.0e-3f);
                }
            }
        }
    }

    @Test
    public void disabledAutocastPreservesBfloat16OutputAndDtypeValidation() {
        Engine engine = Engine.getInstance();
        requireRocm(engine);
        Device device = Device.gpu();
        try (NDManager manager = engine.newBaseManager(device);
                Autocast ignored = new PtAutocast(device, DataType.BFLOAT16, false, true)) {
            Shape shape = new Shape(3, 64);
            NDArray residual =
                    manager.create(sequence(shape.size(), 0.03125f, 0), shape)
                            .toType(DataType.BFLOAT16, false);
            NDArray update =
                    manager.create(sequence(shape.size(), -0.015625f, 7), shape)
                            .toType(DataType.BFLOAT16, false);
            NDArray gamma = manager.ones(new Shape(64), DataType.BFLOAT16);
            NDArray beta = manager.zeros(new Shape(64), DataType.BFLOAT16);
            NDList actual = residualAddLayerNorm(residual, update, gamma, beta, 64, true);
            NDList expected = residualAddLayerNorm(residual, update, gamma, beta, 64, false);
            Assert.assertEquals(actual.get(0).getDataType(), DataType.BFLOAT16);
            assertClose(actual.get(0), expected.get(0), 0f);
            assertClose(actual.get(1), expected.get(1), 0f);

            NDArray floatResidual = residual.toType(DataType.FLOAT32, false);
            NDArray floatUpdate = update.toType(DataType.FLOAT32, false);
            Assert.expectThrows(
                    EngineException.class,
                    () -> residualAddLayerNorm(floatResidual, floatUpdate, gamma, beta, 64, false));
            Assert.expectThrows(
                    EngineException.class,
                    () -> residualAddLayerNorm(floatResidual, floatUpdate, gamma, beta, 64, true));
        }
    }

    @Test
    public void rocmAffineReductionMatchesEagerAtLargeRowCount() {
        Engine engine = Engine.getInstance();
        requireRocm(engine);
        Device device = Device.gpu();
        try (NDManager manager = engine.newBaseManager(device)) {
            for (DataType parameterType : AFFINE_TYPES) {
                // The active learner's shape reaches the 1024-partial reduction limit.
                TrainingResult reference =
                        train(
                                engine,
                                manager,
                                device,
                                17408,
                                384,
                                DataType.FLOAT32,
                                DataType.BFLOAT16,
                                parameterType,
                                OutputUse.BOTH,
                                false);
                TrainingResult fused =
                        train(
                                engine,
                                manager,
                                device,
                                17408,
                                384,
                                DataType.FLOAT32,
                                DataType.BFLOAT16,
                                parameterType,
                                OutputUse.BOTH,
                                true);
                assertRelativeClose(fused.normalized, reference.normalized, 3.0e-5f);
                assertClose(fused.summedResidual, reference.summedResidual, 0f);
                assertRelativeClose(fused.residualGradient, reference.residualGradient, 1.0e-4f);
                assertRelativeClose(fused.updateGradient, reference.updateGradient, 1.2e-2f);
                float affineTolerance = parameterType == DataType.FLOAT32 ? 1.0e-4f : 1.2e-2f;
                assertRelativeClose(fused.gammaGradient, reference.gammaGradient, affineTolerance);
                assertRelativeClose(fused.betaGradient, reference.betaGradient, affineTolerance);
            }
        }
    }

    @Test
    public void rocmAffineReductionMatchesEagerAcrossLargeChainedBlocks() {
        Engine engine = Engine.getInstance();
        requireRocm(engine);
        Device device = Device.gpu();
        try (NDManager manager = engine.newBaseManager(device)) {
            for (DataType parameterType : AFFINE_TYPES) {
                float[][] reference =
                        trainChain(engine, manager, device, parameterType, false, 17408, 64);
                float[][] fused =
                        trainChain(engine, manager, device, parameterType, true, 17408, 64);
                for (int index = 0; index < fused.length; ++index) {
                    boolean bfloatGradient =
                            index == 5
                                    || index == 6
                                    || (index >= 7 && parameterType == DataType.BFLOAT16);
                    assertRelativeClose(
                            fused[index], reference[index], bfloatGradient ? 1.2e-2f : 1.0e-4f);
                }
            }
        }
    }

    private static TrainingResult train(
            Engine engine,
            NDManager manager,
            Device device,
            int rows,
            int width,
            DataType residualType,
            DataType updateType,
            DataType parameterType,
            OutputUse outputUse,
            boolean fused) {
        Shape shape = new Shape(rows, width);
        float[] residualValues = sequence(shape.size(), 0.03125f, 0);
        float[] updateValues = sequence(shape.size(), -0.015625f, 7);
        try (NDManager scope = manager.newSubManager();
                NDArray residual = scope.create(residualValues, shape).toType(residualType, false);
                NDArray update = scope.create(updateValues, shape).toType(updateType, false);
                NDArray gamma =
                        scope.create(sequence(width, 0.00390625f, 3))
                                .add(1f)
                                .toType(parameterType, false);
                NDArray beta =
                        scope.create(sequence(width, -0.001953125f, 11))
                                .toType(parameterType, false);
                NDArray normalizedLossWeight =
                        scope.create(sequence(shape.size(), 0.001953125f, 5), shape);
                NDArray summedLossWeight =
                        scope.create(sequence(shape.size(), -0.0009765625f, 13), shape)) {
            residual.setRequiresGradient(true);
            update.setRequiresGradient(true);
            gamma.setRequiresGradient(true);
            beta.setRequiresGradient(true);

            try (GradientCollector collector = engine.newGradientCollector();
                    Autocast ignored = engine.newAutocast(device, DataType.BFLOAT16, true)) {
                NDList outputs;
                if (fused) {
                    outputs =
                            LayerNorm.residualAddLayerNorm(
                                    residual, update, new Shape(width), gamma, beta, EPSILON);
                } else {
                    NDArray summedResidual = residual.add(update);
                    NDArray normalized =
                            LayerNorm.layerNorm(
                                            summedResidual, new Shape(width), gamma, beta, EPSILON)
                                    .singletonOrThrow();
                    outputs = new NDList(normalized, summedResidual);
                }

                NDArray objective;
                switch (outputUse) {
                    case NORMALIZED:
                        objective = outputs.get(0).mul(normalizedLossWeight).sum();
                        break;
                    case SUM:
                        objective = outputs.get(1).mul(summedLossWeight).sum();
                        break;
                    case BOTH:
                        objective =
                                outputs.get(0)
                                        .mul(normalizedLossWeight)
                                        .sum()
                                        .add(outputs.get(1).mul(summedLossWeight).sum());
                        break;
                    default:
                        throw new AssertionError(outputUse);
                }
                collector.backward(objective);

                Assert.assertEquals(outputs.get(0).getDataType(), DataType.FLOAT32);
                Assert.assertEquals(residual.getGradient().getDataType(), residualType);
                Assert.assertEquals(update.getGradient().getDataType(), updateType);
                if (outputUse == OutputUse.SUM) {
                    Assert.assertNull(JniUtils.getGradient((PtNDArray) gamma));
                    Assert.assertNull(JniUtils.getGradient((PtNDArray) beta));
                } else {
                    Assert.assertEquals(gamma.getGradient().getDataType(), parameterType);
                    Assert.assertEquals(beta.getGradient().getDataType(), parameterType);
                }
                Assert.assertEquals(floatValues(residual), cast(residualValues, residualType), 0f);
                Assert.assertEquals(floatValues(update), cast(updateValues, updateType), 0f);
                return new TrainingResult(
                        floatValues(outputs.get(0)),
                        floatValues(outputs.get(1)),
                        floatValues(residual.getGradient()),
                        floatValues(update.getGradient()),
                        gradientValues(gamma),
                        gradientValues(beta));
            }
        }
    }

    private static float[][] trainChain(
            Engine engine,
            NDManager manager,
            Device device,
            DataType parameterType,
            boolean fused) {
        return trainChain(engine, manager, device, parameterType, fused, 37, 256);
    }

    private static float[][] trainChain(
            Engine engine,
            NDManager manager,
            Device device,
            DataType parameterType,
            boolean fused,
            int rows,
            int width) {
        Shape shape = new Shape(rows, width);
        float[] residualValues = sequence(shape.size(), 0.03125f, 0);
        float[] firstUpdateValues = sequence(shape.size(), -0.015625f, 7);
        float[] secondUpdateValues = sequence(shape.size(), 0.0078125f, 19);
        try (NDManager scope = manager.newSubManager();
                NDArray residual = scope.create(residualValues, shape);
                NDArray firstUpdate =
                        scope.create(firstUpdateValues, shape).toType(DataType.BFLOAT16, false);
                NDArray secondUpdate =
                        scope.create(secondUpdateValues, shape).toType(DataType.BFLOAT16, false);
                NDArray firstGamma =
                        scope.create(sequence(width, 0.00390625f, 3))
                                .add(1f)
                                .toType(parameterType, false);
                NDArray firstBeta =
                        scope.create(sequence(width, -0.001953125f, 11))
                                .toType(parameterType, false);
                NDArray secondGamma =
                        scope.create(sequence(width, -0.001953125f, 13))
                                .add(1f)
                                .toType(parameterType, false);
                NDArray secondBeta =
                        scope.create(sequence(width, 0.0009765625f, 17))
                                .toType(parameterType, false);
                NDArray firstNormalizedLossWeight =
                        scope.create(sequence(shape.size(), 0.001953125f, 5), shape);
                NDArray secondNormalizedLossWeight =
                        scope.create(sequence(shape.size(), -0.0009765625f, 9), shape);
                NDArray secondSummedLossWeight =
                        scope.create(sequence(shape.size(), 0.00048828125f, 23), shape)) {
            residual.setRequiresGradient(true);
            firstUpdate.setRequiresGradient(true);
            secondUpdate.setRequiresGradient(true);
            firstGamma.setRequiresGradient(true);
            firstBeta.setRequiresGradient(true);
            secondGamma.setRequiresGradient(true);
            secondBeta.setRequiresGradient(true);

            try (GradientCollector collector = engine.newGradientCollector();
                    Autocast ignored = engine.newAutocast(device, DataType.BFLOAT16, true)) {
                NDList first =
                        residualAddLayerNorm(
                                residual, firstUpdate, firstGamma, firstBeta, width, fused);
                NDList second =
                        residualAddLayerNorm(
                                first.get(1), secondUpdate, secondGamma, secondBeta, width, fused);
                NDArray objective =
                        first.get(0)
                                .mul(firstNormalizedLossWeight)
                                .sum()
                                .add(second.get(0).mul(secondNormalizedLossWeight).sum())
                                .add(second.get(1).mul(secondSummedLossWeight).sum());
                collector.backward(objective);

                Assert.assertEquals(floatValues(residual), residualValues, 0f);
                Assert.assertEquals(
                        floatValues(firstUpdate), cast(firstUpdateValues, DataType.BFLOAT16), 0f);
                Assert.assertEquals(
                        floatValues(secondUpdate), cast(secondUpdateValues, DataType.BFLOAT16), 0f);
                return new float[][] {
                    floatValues(first.get(0)),
                    floatValues(first.get(1)),
                    floatValues(second.get(0)),
                    floatValues(second.get(1)),
                    floatValues(residual.getGradient()),
                    floatValues(firstUpdate.getGradient()),
                    floatValues(secondUpdate.getGradient()),
                    floatValues(firstGamma.getGradient()),
                    floatValues(firstBeta.getGradient()),
                    floatValues(secondGamma.getGradient()),
                    floatValues(secondBeta.getGradient())
                };
            }
        }
    }

    private static NDList residualAddLayerNorm(
            NDArray residual,
            NDArray update,
            NDArray gamma,
            NDArray beta,
            int width,
            boolean fused) {
        if (fused) {
            return LayerNorm.residualAddLayerNorm(
                    residual, update, new Shape(width), gamma, beta, EPSILON);
        }
        NDArray summedResidual = residual.add(update);
        NDArray normalized =
                LayerNorm.layerNorm(summedResidual, new Shape(width), gamma, beta, EPSILON)
                        .singletonOrThrow();
        return new NDList(normalized, summedResidual);
    }

    private static float[] gradientValues(NDArray array) {
        return array.hasGradient() ? floatValues(array.getGradient()) : null;
    }

    private static void assertTrainingResult(
            TrainingResult actual, TrainingResult expected, float tolerance) {
        assertClose(actual.normalized, expected.normalized, tolerance);
        assertClose(actual.summedResidual, expected.summedResidual, tolerance);
        assertClose(actual.residualGradient, expected.residualGradient, tolerance);
        assertClose(actual.updateGradient, expected.updateGradient, tolerance);
        assertNullableClose(actual.gammaGradient, expected.gammaGradient, tolerance);
        assertNullableClose(actual.betaGradient, expected.betaGradient, tolerance);
    }

    private static void assertNullableClose(float[] actual, float[] expected, float tolerance) {
        Assert.assertEquals(actual == null, expected == null);
        if (actual != null) {
            assertClose(actual, expected, tolerance);
        }
    }

    private static float[] cast(float[] values, DataType dataType) {
        if (dataType == DataType.FLOAT32) {
            return values;
        }
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(Device.cpu());
                NDArray array = manager.create(values).toType(dataType, false)) {
            return floatValues(array);
        }
    }

    private static float[] sequence(long size, float scale, int offset) {
        float[] values = new float[Math.toIntExact(size)];
        for (int index = 0; index < values.length; ++index) {
            values[index] = ((index + offset) % 31 - 15) * scale;
        }
        return values;
    }

    private static void assertClose(NDArray actual, NDArray expected, float tolerance) {
        assertClose(floatValues(actual), floatValues(expected), tolerance);
    }

    private static float[] floatValues(NDArray array) {
        if (array.getDataType() == DataType.FLOAT32) {
            return array.toFloatArray();
        }
        try (NDArray converted = array.toType(DataType.FLOAT32, false)) {
            return converted.toFloatArray();
        }
    }

    private static void assertClose(float[] actual, float[] expected, float tolerance) {
        Assert.assertEquals(actual.length, expected.length);
        for (int index = 0; index < actual.length; ++index) {
            Assert.assertEquals(
                    actual[index], expected[index], tolerance, "mismatch at index " + index);
        }
    }

    private static void assertRelativeClose(
            float[] actual, float[] expected, float relativeTolerance) {
        Assert.assertEquals(actual.length, expected.length);
        for (int index = 0; index < actual.length; ++index) {
            float tolerance = 1.0e-5f + relativeTolerance * Math.abs(expected[index]);
            if (!Float.isFinite(actual[index])
                    || Math.abs(actual[index] - expected[index]) > tolerance) {
                Assert.fail(
                        "mismatch at index "
                                + index
                                + ": expected "
                                + expected[index]
                                + " but found "
                                + actual[index]
                                + ", tolerance="
                                + tolerance);
            }
        }
    }

    private static void requireRocm(Engine engine) {
        if (engine.getGpuCount() == 0 || JniUtils.getFusionBackend() != 2) {
            throw new SkipException("This residual add LayerNorm test requires PyTorch ROCm.");
        }
    }

    private enum OutputUse {
        NORMALIZED,
        SUM,
        BOTH
    }

    private static final class TrainingResult {

        private final float[] normalized;
        private final float[] summedResidual;
        private final float[] residualGradient;
        private final float[] updateGradient;
        private final float[] gammaGradient;
        private final float[] betaGradient;

        private TrainingResult(
                float[] normalized,
                float[] summedResidual,
                float[] residualGradient,
                float[] updateGradient,
                float[] gammaGradient,
                float[] betaGradient) {
            this.normalized = normalized;
            this.summedResidual = summedResidual;
            this.residualGradient = residualGradient;
            this.updateGradient = updateGradient;
            this.gammaGradient = gammaGradient;
            this.betaGradient = betaGradient;
        }
    }
}
