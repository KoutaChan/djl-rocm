/*
 * Copyright 2026 KoutaChan.
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
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.training.GradientCollector;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

/** Tests the ROCm inference specialization for ordinary autocast LayerNorm. */
@SuppressWarnings("try") // Autocast resources are used for their scope side effects.
public class AutocastLayerNormTest {

    private static final float EPSILON = 1.0e-5f;
    private static final int[] ACTIVE_EXTENTS = {1, 384, 31, 256, 1};
    private static final DataType[] FLOATING_DATA_TYPES = {
        DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16
    };

    @Test
    public void lowPrecisionInputsAndParametersProduceFloat32() {
        runOnGpuIfAvailable(
                (engine, manager, device) -> {
                    for (DataType inputType :
                            new DataType[] {DataType.FLOAT16, DataType.BFLOAT16}) {
                        for (DataType parameterType :
                                new DataType[] {DataType.FLOAT16, DataType.BFLOAT16}) {
                            verifyLowPrecisionCombination(
                                    engine, manager, device, inputType, parameterType);
                        }
                    }
                });
    }

    @Test
    public void layerNormAndCastMatchesEagerAcrossGpusDtypesAndSlotReuse() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This LayerNorm test requires a PyTorch GPU.");
        }
        int deviceCount = Math.min(2, engine.getGpuCount());
        for (int deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex) {
            Device device = Device.gpu(deviceIndex);
            try (NDManager manager = engine.newBaseManager(device)) {
                for (DataType inputType : FLOATING_DATA_TYPES) {
                    for (DataType convertedType : FLOATING_DATA_TYPES) {
                        verifyLayerNormAndCastSlotReuse(
                                engine, manager, device, inputType, convertedType);
                    }
                }
            }
        }
    }

    private static void verifyLayerNormAndCastSlotReuse(
            Engine engine,
            NDManager manager,
            Device device,
            DataType inputType,
            DataType convertedType) {
        NDManager[] slots = new NDManager[2];
        try {
            for (int boundary = 0; boundary < ACTIVE_EXTENTS.length; ++boundary) {
                int slot = boundary % slots.length;
                if (slots[slot] != null) {
                    slots[slot].close();
                }
                slots[slot] = manager.newSubManager();
                verifyLayerNormAndCastBoundary(
                        engine,
                        slots[slot],
                        device,
                        slot,
                        ACTIVE_EXTENTS[boundary],
                        inputType,
                        convertedType);
            }
        } finally {
            for (NDManager slot : slots) {
                if (slot != null) {
                    slot.close();
                }
            }
        }
    }

    private static void verifyLayerNormAndCastBoundary(
            Engine engine,
            NDManager manager,
            Device device,
            int slot,
            int activeExtent,
            DataType inputType,
            DataType convertedType) {
        Shape inputShape = new Shape(activeExtent, 3, 256);
        Shape normalizedShape = new Shape(256);
        NDArray input =
                manager.create(sequence(inputShape.size(), 0.015625f), inputShape)
                        .toType(inputType, false);
        NDArray weight =
                manager.create(sequence(256, 0.00390625f)).add(1.0f).toType(inputType, false);
        NDArray bias = manager.create(sequence(256, -0.001953125f)).toType(inputType, false);
        NDArray expected = floatReference(input, normalizedShape, weight, bias);

        NDList outputs;
        if (inputType == DataType.FLOAT32) {
            outputs =
                    LayerNorm.layerNormAndCast(
                            input, normalizedShape, weight, bias, EPSILON, convertedType);
        } else {
            try (Autocast ignored = engine.newAutocast(device, inputType, true)) {
                outputs =
                        LayerNorm.layerNormAndCast(
                                input, normalizedShape, weight, bias, EPSILON, convertedType);
            }
        }

        Assert.assertEquals(outputs.size(), 2);
        Assert.assertEquals(outputs.get(0).getDataType(), DataType.FLOAT32);
        Assert.assertEquals(outputs.get(1).getDataType(), convertedType);
        assertClose(outputs.get(0), expected, normalizedTolerance(inputType));
        assertClose(
                outputs.get(1).toType(DataType.FLOAT32, false),
                expected,
                convertedTolerance(inputType, convertedType));
        System.out.printf(
                "LAYER_NORM_AND_CAST_PARITY device=%s slot=%d activeExtent=%d input=%s"
                        + " converted=%s%n",
                device, slot, activeExtent, inputType, convertedType);
    }

    private static float normalizedTolerance(DataType inputType) {
        switch (inputType) {
            case FLOAT32:
                return 2.0e-5f;
            case FLOAT16:
                return 8.0e-4f;
            case BFLOAT16:
                return 2.0e-3f;
            default:
                throw new AssertionError(inputType);
        }
    }

    private static float convertedTolerance(DataType inputType, DataType convertedType) {
        float conversionTolerance;
        switch (convertedType) {
            case FLOAT32:
                conversionTolerance = 0.0f;
                break;
            case FLOAT16:
                conversionTolerance = 1.5e-3f;
                break;
            case BFLOAT16:
                conversionTolerance = 1.0e-2f;
                break;
            default:
                throw new AssertionError(convertedType);
        }
        return Math.max(normalizedTolerance(inputType), conversionTolerance);
    }

    @Test
    public void multidimensionalNormalizedShapeUsesFallback() {
        runOnGpuIfAvailable(
                (engine, manager, device) -> {
                    Shape inputShape = new Shape(2, 3, 4, 8);
                    Shape normalizedShape = new Shape(4, 8);
                    NDArray input =
                            manager.create(sequence(inputShape.size(), 0.03125f), inputShape)
                                    .toType(DataType.BFLOAT16, false);
                    NDArray weight =
                            manager.create(
                                            sequence(normalizedShape.size(), 0.015625f),
                                            normalizedShape)
                                    .add(1.0f)
                                    .toType(DataType.BFLOAT16, false);
                    NDArray bias =
                            manager.create(
                                            sequence(normalizedShape.size(), -0.0078125f),
                                            normalizedShape)
                                    .toType(DataType.BFLOAT16, false);
                    NDArray expected = floatReference(input, normalizedShape, weight, bias);

                    NDArray actual;
                    try (Autocast ignored = engine.newAutocast(device, DataType.BFLOAT16, true)) {
                        actual =
                                LayerNorm.layerNorm(input, normalizedShape, weight, bias, EPSILON)
                                        .singletonOrThrow();
                    }

                    Assert.assertEquals(actual.getDataType(), DataType.FLOAT32);
                    assertClose(actual, expected, 2.0e-3f);
                });
    }

    @Test
    public void float32AffineParametersUseFallback() {
        runOnGpuIfAvailable(
                (engine, manager, device) -> {
                    Shape inputShape = new Shape(2, 9, 128);
                    Shape normalizedShape = new Shape(128);
                    NDArray input =
                            manager.create(sequence(inputShape.size(), 0.03125f), inputShape)
                                    .toType(DataType.FLOAT16, false);
                    NDArray weight = manager.ones(normalizedShape);
                    NDArray bias = manager.zeros(normalizedShape);
                    NDArray expected = floatReference(input, normalizedShape, weight, bias);

                    NDArray actual;
                    try (Autocast ignored = engine.newAutocast(device, DataType.FLOAT16, true)) {
                        actual =
                                LayerNorm.layerNorm(input, normalizedShape, weight, bias, EPSILON)
                                        .singletonOrThrow();
                    }

                    Assert.assertEquals(actual.getDataType(), DataType.FLOAT32);
                    assertClose(actual, expected, 8.0e-4f);
                });
    }

    @Test
    public void disabledAutocastPreservesPyTorchOutputType() {
        runOnGpuIfAvailable(
                (engine, manager, device) -> {
                    Shape inputShape = new Shape(3, 17);
                    NDArray input =
                            manager.create(sequence(inputShape.size(), 0.0625f), inputShape)
                                    .toType(DataType.FLOAT16, false);
                    NDArray weight = manager.ones(new Shape(17)).toType(DataType.FLOAT16, false);
                    NDArray bias = manager.zeros(new Shape(17)).toType(DataType.FLOAT16, false);

                    NDArray output =
                            LayerNorm.layerNorm(input, new Shape(17), weight, bias, EPSILON)
                                    .singletonOrThrow();

                    Assert.assertEquals(output.getDataType(), DataType.FLOAT16);
                    assertAllFinite(output);
                });
    }

    @Test
    public void autocastTrainingRemainsDifferentiable() {
        runOnGpuIfAvailable(
                (engine, manager, device) -> {
                    Shape inputShape = new Shape(2, 5, 64);
                    NDArray input =
                            manager.create(sequence(inputShape.size(), 0.03125f), inputShape)
                                    .toType(DataType.BFLOAT16, false);
                    NDArray weight = manager.ones(new Shape(64)).toType(DataType.BFLOAT16, false);
                    NDArray bias = manager.zeros(new Shape(64)).toType(DataType.BFLOAT16, false);
                    input.setRequiresGradient(true);
                    weight.setRequiresGradient(true);
                    bias.setRequiresGradient(true);

                    try (GradientCollector collector = engine.newGradientCollector();
                            Autocast ignored =
                                    engine.newAutocast(device, DataType.BFLOAT16, true)) {
                        NDArray output =
                                LayerNorm.layerNorm(input, new Shape(64), weight, bias, EPSILON)
                                        .singletonOrThrow();
                        Assert.assertEquals(output.getDataType(), DataType.FLOAT32);
                        collector.backward(output.mul(output).mean());
                    }

                    Assert.assertTrue(input.hasGradient());
                    Assert.assertTrue(weight.hasGradient());
                    Assert.assertTrue(bias.hasGradient());
                    assertAllFinite(input.getGradient());
                    assertAllFinite(weight.getGradient());
                    assertAllFinite(bias.getGradient());
                });
    }

    @Test
    public void layerNormAndCastTrainingRemainsDifferentiable() {
        runOnGpuIfAvailable(
                (engine, manager, device) -> {
                    Shape inputShape = new Shape(2, 5, 64);
                    NDArray input =
                            manager.create(sequence(inputShape.size(), 0.03125f), inputShape)
                                    .toType(DataType.FLOAT16, false);
                    NDArray weight = manager.ones(new Shape(64)).toType(DataType.FLOAT16, false);
                    NDArray bias = manager.zeros(new Shape(64)).toType(DataType.FLOAT16, false);
                    input.setRequiresGradient(true);
                    weight.setRequiresGradient(true);
                    bias.setRequiresGradient(true);

                    try (GradientCollector collector = engine.newGradientCollector();
                            Autocast ignored = engine.newAutocast(device, DataType.FLOAT16, true)) {
                        NDList outputs =
                                LayerNorm.layerNormAndCast(
                                        input,
                                        new Shape(64),
                                        weight,
                                        bias,
                                        EPSILON,
                                        DataType.FLOAT16);
                        Assert.assertEquals(outputs.get(0).getDataType(), DataType.FLOAT32);
                        Assert.assertEquals(outputs.get(1).getDataType(), DataType.FLOAT16);
                        collector.backward(
                                outputs.get(0)
                                        .mean()
                                        .add(
                                                outputs.get(1)
                                                        .toType(DataType.FLOAT32, false)
                                                        .mean()));
                    }

                    Assert.assertTrue(input.hasGradient());
                    Assert.assertTrue(weight.hasGradient());
                    Assert.assertTrue(bias.hasGradient());
                    assertAllFinite(input.getGradient());
                    assertAllFinite(weight.getGradient());
                    assertAllFinite(bias.getGradient());
                });
    }

    private static void verifyLowPrecisionCombination(
            Engine engine,
            NDManager manager,
            Device device,
            DataType inputType,
            DataType parameterType) {
        Shape inputShape = new Shape(3, 5, 257);
        Shape normalizedShape = new Shape(257);
        NDArray input =
                manager.create(sequence(inputShape.size(), 0.03125f), inputShape)
                        .toType(inputType, false);
        NDArray weight =
                manager.create(sequence(257, 0.0078125f)).add(1.0f).toType(parameterType, false);
        NDArray bias = manager.create(sequence(257, -0.00390625f)).toType(parameterType, false);
        NDArray expected = floatReference(input, normalizedShape, weight, bias);

        NDArray actual;
        try (Autocast ignored = engine.newAutocast(device, inputType, true)) {
            actual =
                    LayerNorm.layerNorm(input, normalizedShape, weight, bias, EPSILON)
                            .singletonOrThrow();
        }

        Assert.assertEquals(
                actual.getDataType(),
                DataType.FLOAT32,
                "autocast LayerNorm must retain PyTorch's FLOAT32 output policy");
        assertClose(actual, expected, inputType == DataType.FLOAT16 ? 8.0e-4f : 2.0e-3f);
    }

    private static NDArray floatReference(
            NDArray input, Shape normalizedShape, NDArray weight, NDArray bias) {
        NDArray floatInput = input.toType(DataType.FLOAT32, false);
        NDArray floatWeight = weight.toType(DataType.FLOAT32, false);
        NDArray floatBias = bias.toType(DataType.FLOAT32, false);
        return LayerNorm.layerNorm(floatInput, normalizedShape, floatWeight, floatBias, EPSILON)
                .singletonOrThrow();
    }

    private static float[] sequence(long size, float scale) {
        float[] values = new float[Math.toIntExact(size)];
        for (int index = 0; index < values.length; index++) {
            values[index] = (index % 31 - 15) * scale;
        }
        return values;
    }

    private static void assertClose(NDArray actual, NDArray expected, float tolerance) {
        float[] actualValues = actual.toFloatArray();
        float[] expectedValues = expected.toFloatArray();
        Assert.assertEquals(actualValues.length, expectedValues.length);
        for (int index = 0; index < actualValues.length; index++) {
            Assert.assertEquals(
                    actualValues[index],
                    expectedValues[index],
                    tolerance,
                    "LayerNorm output mismatch at index " + index);
        }
    }

    private static void assertAllFinite(NDArray array) {
        for (float value : array.toType(DataType.FLOAT32, false).toFloatArray()) {
            Assert.assertTrue(Float.isFinite(value), "non-finite value: " + value);
        }
    }

    private static void runOnGpuIfAvailable(GpuTest body) {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This LayerNorm test requires a PyTorch GPU.");
        }
        Device device = Device.gpu();
        try (NDManager manager = engine.newBaseManager(device)) {
            body.run(engine, manager, device);
        }
    }

    @FunctionalInterface
    private interface GpuTest {
        void run(Engine engine, NDManager manager, Device device);
    }
}
