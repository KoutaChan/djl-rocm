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
import ai.djl.engine.InferenceMode;
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

/** Tests autocast LayerNorm specializations and their eager fallbacks. */
@SuppressWarnings("try") // Autocast resources are used for their scope side effects.
public class AutocastLayerNormTest {

    private static final float EPSILON = 1.0e-5f;
    private static final int[] ACTIVE_EXTENTS = {1, 384, 31, 256, 1};
    private static final int[] WIDTHS = {1, 17, 64, 256, 257};
    private static final int[] INFERENCE_WIDTHS = {
        1, 17, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 512, 513, 768, 1024, 1025, 2048, 2049,
        4096, 4097, 8192, 8193, 16384, 32769
    };
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

    @Test
    public void layerNormAndCastInferenceMatchesEagerAcrossWidthsAndAffineDtypes() {
        runOnGpuIfAvailable(
                (engine, manager, device) -> {
                    for (DataType inputType :
                            new DataType[] {DataType.FLOAT16, DataType.BFLOAT16}) {
                        for (DataType parameterType : FLOATING_DATA_TYPES) {
                            for (DataType convertedType :
                                    new DataType[] {DataType.FLOAT16, DataType.BFLOAT16}) {
                                for (int width : INFERENCE_WIDTHS) {
                                    try (NDManager scope = manager.newSubManager()) {
                                        Shape shape = new Shape(3, width);
                                        NDArray input =
                                                scope.create(
                                                                sequence(shape.size(), 0.03125f),
                                                                shape)
                                                        .toType(inputType, false);
                                        NDArray weight =
                                                scope.create(sequence(width, 0.00390625f))
                                                        .add(1f)
                                                        .toType(parameterType, false);
                                        NDArray bias =
                                                scope.create(sequence(width, -0.001953125f))
                                                        .toType(parameterType, false);
                                        // Inference mode must ignore leaf flags as well as
                                        // GradMode.
                                        input.setRequiresGradient(true);
                                        weight.setRequiresGradient(true);
                                        bias.setRequiresGradient(true);
                                        verifyInferenceLayerNormAndCast(
                                                engine,
                                                device,
                                                input,
                                                new Shape(width),
                                                weight,
                                                bias,
                                                convertedType,
                                                true);
                                    }
                                }
                            }
                        }
                    }
                });
    }

    @Test
    public void layerNormAndCastInferenceSupportsTrailingNormalizedDimensions() {
        runOnGpuIfAvailable(
                (engine, manager, device) -> {
                    Shape[] inputShapes = {
                        new Shape(7),
                        new Shape(3, 5, 127),
                        new Shape(2, 3, 5, 127),
                        new Shape(2, 3, 8, 16, 17),
                        new Shape(2, 9, 17, 33),
                        new Shape(2, 3, 17),
                        new Shape(0, 7, 1025)
                    };
                    Shape[] normalizedShapes = {
                        new Shape(7),
                        new Shape(127),
                        new Shape(5, 127),
                        new Shape(8, 16, 17),
                        new Shape(9, 17, 33),
                        new Shape(2, 3, 17),
                        new Shape(7, 1025)
                    };
                    for (DataType inputType :
                            new DataType[] {DataType.FLOAT16, DataType.BFLOAT16}) {
                        for (DataType parameterType : FLOATING_DATA_TYPES) {
                            for (int index = 0; index < inputShapes.length; ++index) {
                                try (NDManager scope = manager.newSubManager()) {
                                    Shape shape = inputShapes[index];
                                    Shape normalizedShape = normalizedShapes[index];
                                    NDArray input =
                                            scope.create(sequence(shape.size(), 0.03125f), shape)
                                                    .toType(inputType, false);
                                    NDArray weight =
                                            scope.create(
                                                            sequence(
                                                                    normalizedShape.size(),
                                                                    0.00390625f),
                                                            normalizedShape)
                                                    .add(1f)
                                                    .toType(parameterType, false);
                                    NDArray bias =
                                            scope.create(
                                                            sequence(
                                                                    normalizedShape.size(),
                                                                    -0.001953125f),
                                                            normalizedShape)
                                                    .toType(parameterType, false);
                                    verifyInferenceLayerNormAndCast(
                                            engine,
                                            device,
                                            input,
                                            normalizedShape,
                                            weight,
                                            bias,
                                            inputType,
                                            true);
                                }
                            }
                        }
                    }
                });
    }

    @Test
    public void layerNormAndCastInferencePreservesSmallVarianceInWideRows() {
        runOnGpuIfAvailable(
                (engine, manager, device) -> {
                    for (DataType inputType :
                            new DataType[] {DataType.FLOAT16, DataType.BFLOAT16}) {
                        for (int width : new int[] {513, 1025, 4097, 32769}) {
                            try (NDManager scope = manager.newSubManager()) {
                                Shape shape = new Shape(3, width);
                                NDArray input =
                                        scope.create(sequence(shape.size(), 0.03125f), shape)
                                                .add(16f)
                                                .toType(inputType, false);
                                NDArray weight = scope.ones(new Shape(width));
                                NDArray bias = scope.zeros(new Shape(width));
                                verifyInferenceLayerNormAndCast(
                                        engine,
                                        device,
                                        input,
                                        new Shape(width),
                                        weight,
                                        bias,
                                        inputType,
                                        true);
                            }
                        }
                    }
                });
    }

    @Test
    public void layerNormAndCastInferenceFallbacksMatchEager() {
        runOnGpuIfAvailable(
                (engine, manager, device) -> {
                    for (int boundary = 0; boundary < 8; ++boundary) {
                        try (NDManager scope = manager.newSubManager()) {
                            int width = boundary == 0 ? 513 : 17;
                            int rows = boundary == 1 ? 0 : 3;
                            DataType inputType =
                                    boundary == 2 ? DataType.FLOAT32 : DataType.BFLOAT16;
                            DataType convertedType =
                                    boundary == 3 ? DataType.FLOAT32 : DataType.FLOAT16;
                            Shape shape = new Shape(rows, width);
                            Shape normalizedShape = new Shape(width);
                            NDArray input =
                                    scope.create(sequence(shape.size(), 0.03125f), shape)
                                            .toType(inputType, false);
                            NDArray weight =
                                    scope.create(sequence(width, 0.00390625f))
                                            .add(1f)
                                            .toType(inputType, false);
                            NDArray bias =
                                    scope.create(sequence(width, -0.001953125f))
                                            .toType(inputType, false);
                            if (boundary == 4) {
                                input = input.transpose();
                                normalizedShape = new Shape(rows);
                                weight = scope.ones(normalizedShape, inputType);
                                bias = scope.zeros(normalizedShape, inputType);
                            } else if (boundary == 5) {
                                normalizedShape = shape;
                                weight = scope.ones(shape, inputType);
                                bias = scope.zeros(shape, inputType);
                            } else if (boundary == 6) {
                                weight =
                                        scope.create(
                                                        sequence(width * 2L, 0.00390625f),
                                                        new Shape(width, 2))
                                                .add(1f)
                                                .toType(inputType, false)
                                                .get(":, 0");
                                bias =
                                        scope.create(
                                                        sequence(width * 2L, -0.001953125f),
                                                        new Shape(width, 2))
                                                .toType(inputType, false)
                                                .get(":, 1");
                            }
                            verifyInferenceLayerNormAndCast(
                                    engine,
                                    device,
                                    input,
                                    normalizedShape,
                                    weight,
                                    bias,
                                    convertedType,
                                    boundary != 7);
                        }
                    }
                });
    }

    @Test
    public void layerNormAndCastInferenceKeepsFloat32OutputsIndependent() {
        runOnGpuIfAvailable(
                (engine, manager, device) -> {
                    NDArray input =
                            manager.create(sequence(51, 0.03125f), new Shape(3, 17))
                                    .toType(DataType.BFLOAT16, false);
                    NDArray weight = manager.ones(new Shape(17));
                    NDArray bias = manager.zeros(new Shape(17));
                    try (InferenceMode inference = engine.newInferenceMode();
                            Autocast autocast =
                                    engine.newAutocast(device, DataType.BFLOAT16, true)) {
                        NDList outputs =
                                LayerNorm.layerNormAndCast(
                                        input,
                                        new Shape(17),
                                        weight,
                                        bias,
                                        EPSILON,
                                        DataType.FLOAT32);
                        float[] normalized = outputs.get(0).toFloatArray();
                        Assert.assertEquals(outputs.get(1).toFloatArray(), normalized);
                        outputs.get(1).addi(3f);
                        Assert.assertEquals(outputs.get(0).toFloatArray(), normalized);
                    }
                });
    }

    @Test
    public void layerNormAndCastCudaTrainingMatchesEagerFallback() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0 || JniUtils.getFusionBackend() != 1) {
            throw new SkipException("This LayerNorm fallback test requires PyTorch CUDA.");
        }
        Device device = Device.gpu();
        try (NDManager manager = engine.newBaseManager(device)) {
            for (DataType inputType : new DataType[] {DataType.FLOAT16, DataType.BFLOAT16}) {
                for (DataType parameterType : FLOATING_DATA_TYPES) {
                    for (OutputUse outputUse : OutputUse.values()) {
                        TrainingResult reference =
                                trainLayerNormAndCast(
                                        engine,
                                        manager,
                                        device,
                                        3,
                                        17,
                                        inputType,
                                        parameterType,
                                        inputType,
                                        outputUse,
                                        false);
                        TrainingResult actual =
                                trainLayerNormAndCast(
                                        engine,
                                        manager,
                                        device,
                                        3,
                                        17,
                                        inputType,
                                        parameterType,
                                        inputType,
                                        outputUse,
                                        true);
                        assertTrainingResult(
                                actual,
                                reference,
                                trainingTolerance(inputType, parameterType, inputType));
                    }
                }
            }
        }
    }

    private static void verifyInferenceLayerNormAndCast(
            Engine engine,
            Device device,
            NDArray input,
            Shape normalizedShape,
            NDArray weight,
            NDArray bias,
            DataType convertedType,
            boolean autocastEnabled) {
        NDArray expected;
        NDArray expectedConverted;
        NDList actual;
        DataType inputType = input.getDataType();
        DataType autocastType =
                inputType == DataType.FLOAT16 ? DataType.FLOAT16 : DataType.BFLOAT16;
        try (InferenceMode inference = engine.newInferenceMode();
                Autocast autocast = engine.newAutocast(device, autocastType, autocastEnabled)) {
            // An FP32 reference bypasses both accelerator low-precision specializations.
            expected =
                    autocastEnabled
                            ? floatReference(input, normalizedShape, weight, bias)
                            : LayerNorm.layerNorm(input, normalizedShape, weight, bias, EPSILON)
                                    .singletonOrThrow();
            expectedConverted = expected.toType(convertedType, false);
            actual =
                    LayerNorm.layerNormAndCast(
                            input, normalizedShape, weight, bias, EPSILON, convertedType);
        }
        Assert.assertEquals(actual.size(), 2);
        Assert.assertEquals(actual.get(0).getDataType(), expected.getDataType());
        Assert.assertEquals(actual.get(1).getDataType(), convertedType);
        Assert.assertEquals(actual.get(0).getShape(), input.getShape());
        Assert.assertEquals(actual.get(1).getShape(), input.getShape());
        assertClose(
                floatValues(actual.get(0)), floatValues(expected), normalizedTolerance(inputType));
        assertClose(
                floatValues(actual.get(1)),
                floatValues(expectedConverted),
                convertedTolerance(inputType, convertedType));
        assertAllFinite(actual.get(0));
        assertAllFinite(actual.get(1));
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
    public void float32AffineParametersRetainAutocastSemantics() {
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
    public void layerNormAndCastAutogradMatchesEagerAcrossWidths() {
        Engine engine = Engine.getInstance();
        requireRocm(engine);
        Device device = Device.gpu();
        try (NDManager manager = engine.newBaseManager(device)) {
            for (int width : WIDTHS) {
                TrainingResult reference =
                        trainLayerNormAndCast(
                                engine,
                                manager,
                                device,
                                11,
                                width,
                                DataType.BFLOAT16,
                                DataType.FLOAT32,
                                DataType.BFLOAT16,
                                OutputUse.BOTH,
                                false);
                TrainingResult fused =
                        trainLayerNormAndCast(
                                engine,
                                manager,
                                device,
                                11,
                                width,
                                DataType.BFLOAT16,
                                DataType.FLOAT32,
                                DataType.BFLOAT16,
                                OutputUse.BOTH,
                                true);
                assertTrainingResult(fused, reference, 2.0e-2f);
            }
        }
    }

    @Test
    public void layerNormAndCastAutogradSupportsInputAndAffineDtypes() {
        Engine engine = Engine.getInstance();
        requireRocm(engine);
        Device device = Device.gpu();
        try (NDManager manager = engine.newBaseManager(device)) {
            for (DataType inputType : new DataType[] {DataType.FLOAT16, DataType.BFLOAT16}) {
                for (DataType parameterType : new DataType[] {DataType.FLOAT32, inputType}) {
                    DataType convertedType =
                            inputType == DataType.FLOAT16 ? DataType.BFLOAT16 : DataType.FLOAT16;
                    TrainingResult reference =
                            trainLayerNormAndCast(
                                    engine,
                                    manager,
                                    device,
                                    13,
                                    64,
                                    inputType,
                                    parameterType,
                                    convertedType,
                                    OutputUse.BOTH,
                                    false);
                    TrainingResult fused =
                            trainLayerNormAndCast(
                                    engine,
                                    manager,
                                    device,
                                    13,
                                    64,
                                    inputType,
                                    parameterType,
                                    convertedType,
                                    OutputUse.BOTH,
                                    true);
                    assertTrainingResult(
                            fused,
                            reference,
                            trainingTolerance(inputType, parameterType, convertedType));
                }
            }
        }
    }

    @Test
    public void layerNormAndCastAutogradSupportsUndefinedAndSharedOutputGradients() {
        Engine engine = Engine.getInstance();
        requireRocm(engine);
        Device device = Device.gpu();
        try (NDManager manager = engine.newBaseManager(device)) {
            for (OutputUse outputUse : OutputUse.values()) {
                TrainingResult reference =
                        trainLayerNormAndCast(
                                engine,
                                manager,
                                device,
                                17,
                                256,
                                DataType.BFLOAT16,
                                DataType.BFLOAT16,
                                DataType.BFLOAT16,
                                outputUse,
                                false);
                TrainingResult fused =
                        trainLayerNormAndCast(
                                engine,
                                manager,
                                device,
                                17,
                                256,
                                DataType.BFLOAT16,
                                DataType.BFLOAT16,
                                DataType.BFLOAT16,
                                outputUse,
                                true);
                assertTrainingResult(fused, reference, 2.0e-2f);
            }
        }
    }

    @Test
    public void layerNormAndCastAutogradSupportsZeroRows() {
        Engine engine = Engine.getInstance();
        requireRocm(engine);
        Device device = Device.gpu();
        try (NDManager manager = engine.newBaseManager(device)) {
            TrainingResult reference =
                    trainLayerNormAndCast(
                            engine,
                            manager,
                            device,
                            0,
                            257,
                            DataType.FLOAT16,
                            DataType.FLOAT32,
                            DataType.BFLOAT16,
                            OutputUse.BOTH,
                            false);
            TrainingResult fused =
                    trainLayerNormAndCast(
                            engine,
                            manager,
                            device,
                            0,
                            257,
                            DataType.FLOAT16,
                            DataType.FLOAT32,
                            DataType.BFLOAT16,
                            OutputUse.BOTH,
                            true);
            assertTrainingResult(fused, reference, 0.0f);
        }
    }

    @Test
    public void layerNormAndCastCpuFallbackMatchesComposedOperations() {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(Device.cpu())) {
            TrainingResult reference = trainLayerNormAndCastCpu(engine, manager, false);
            TrainingResult actual = trainLayerNormAndCastCpu(engine, manager, true);
            assertTrainingResult(actual, reference, 2.0e-5f);
        }
    }

    @Test
    public void layerNormAndCastTrainingSupportsTrailingNormalizedDimensions() {
        runOnGpuIfAvailable(
                (engine, manager, device) -> {
                    for (int width : new int[] {17, 257, 2049}) {
                        for (DataType inputType :
                                new DataType[] {DataType.FLOAT16, DataType.BFLOAT16}) {
                            for (DataType parameterType : FLOATING_DATA_TYPES) {
                                for (OutputUse outputUse : OutputUse.values()) {
                                    Shape shape = new Shape(2, 3, width);
                                    Shape normalizedShape = new Shape(3, width);
                                    TrainingResult reference =
                                            trainLayerNormAndCast(
                                                    engine,
                                                    manager,
                                                    device,
                                                    shape,
                                                    normalizedShape,
                                                    inputType,
                                                    parameterType,
                                                    inputType,
                                                    outputUse,
                                                    false);
                                    TrainingResult actual =
                                            trainLayerNormAndCast(
                                                    engine,
                                                    manager,
                                                    device,
                                                    shape,
                                                    normalizedShape,
                                                    inputType,
                                                    parameterType,
                                                    inputType,
                                                    outputUse,
                                                    true);
                                    assertTrainingResult(
                                            actual,
                                            reference,
                                            trainingTolerance(inputType, parameterType, inputType));
                                }
                            }
                        }
                    }
                });
    }

    private static TrainingResult trainLayerNormAndCast(
            Engine engine,
            NDManager manager,
            Device device,
            int rows,
            int width,
            DataType inputType,
            DataType parameterType,
            DataType convertedType,
            OutputUse outputUse,
            boolean fused) {
        return trainLayerNormAndCast(
                engine,
                manager,
                device,
                new Shape(rows, width),
                new Shape(width),
                inputType,
                parameterType,
                convertedType,
                outputUse,
                fused);
    }

    private static TrainingResult trainLayerNormAndCast(
            Engine engine,
            NDManager manager,
            Device device,
            Shape shape,
            Shape normalizedShape,
            DataType inputType,
            DataType parameterType,
            DataType convertedType,
            OutputUse outputUse,
            boolean fused) {
        try (NDManager scope = manager.newSubManager();
                NDArray input =
                        scope.create(sequence(shape.size(), 0.03125f, 0), shape)
                                .toType(inputType, false);
                NDArray weight =
                        scope.create(
                                        sequence(normalizedShape.size(), 0.00390625f, 3),
                                        normalizedShape)
                                .add(1.0f)
                                .toType(parameterType, false);
                NDArray bias =
                        scope.create(
                                        sequence(normalizedShape.size(), -0.001953125f, 11),
                                        normalizedShape)
                                .toType(parameterType, false);
                NDArray normalizedLossWeight =
                        scope.create(sequence(shape.size(), 0.001953125f, 5), shape);
                NDArray convertedLossWeight =
                        scope.create(sequence(shape.size(), -0.0009765625f, 13), shape)
                                .toType(convertedType, false);
                NDArray secondConvertedLossWeight =
                        scope.create(sequence(shape.size(), 0.00048828125f, 19), shape)
                                .toType(convertedType, false);
                NDArray thirdConvertedLossWeight =
                        scope.create(sequence(shape.size(), -0.000244140625f, 23), shape)
                                .toType(convertedType, false)) {
            input.setRequiresGradient(true);
            weight.setRequiresGradient(true);
            bias.setRequiresGradient(true);

            try (GradientCollector collector = engine.newGradientCollector();
                    Autocast ignored = engine.newAutocast(device, inputType, true)) {
                NDList outputs =
                        layerNormAndCast(
                                input, weight, bias, normalizedShape, convertedType, fused);
                Assert.assertEquals(outputs.get(0).getDataType(), DataType.FLOAT32);
                Assert.assertEquals(outputs.get(1).getDataType(), convertedType);
                NDArray objective =
                        trainingObjective(
                                outputs,
                                normalizedLossWeight,
                                convertedLossWeight,
                                secondConvertedLossWeight,
                                thirdConvertedLossWeight,
                                outputUse,
                                fused);
                collector.backward(objective);
                return new TrainingResult(
                        floatValues(outputs.get(0)),
                        floatValues(outputs.get(1)),
                        gradientValues(input),
                        gradientValues(weight),
                        gradientValues(bias));
            }
        }
    }

    private static TrainingResult trainLayerNormAndCastCpu(
            Engine engine, NDManager manager, boolean fused) {
        int rows = 5;
        int width = 17;
        Shape shape = new Shape(rows, width);
        try (NDManager scope = manager.newSubManager();
                NDArray input = scope.create(sequence(shape.size(), 0.03125f, 0), shape);
                NDArray weight = scope.create(sequence(width, 0.00390625f, 3)).add(1.0f);
                NDArray bias = scope.create(sequence(width, -0.001953125f, 11));
                NDArray normalizedLossWeight =
                        scope.create(sequence(shape.size(), 0.001953125f, 5), shape);
                NDArray convertedLossWeight =
                        scope.create(sequence(shape.size(), -0.0009765625f, 13), shape)
                                .toType(DataType.FLOAT16, false);
                NDArray secondConvertedLossWeight =
                        scope.create(sequence(shape.size(), 0.00048828125f, 19), shape)
                                .toType(DataType.FLOAT16, false);
                NDArray thirdConvertedLossWeight =
                        scope.create(sequence(shape.size(), -0.000244140625f, 23), shape)
                                .toType(DataType.FLOAT16, false);
                GradientCollector collector = engine.newGradientCollector()) {
            input.setRequiresGradient(true);
            weight.setRequiresGradient(true);
            bias.setRequiresGradient(true);
            NDList outputs =
                    layerNormAndCast(
                            input, weight, bias, new Shape(width), DataType.FLOAT16, fused);
            NDArray objective =
                    trainingObjective(
                            outputs,
                            normalizedLossWeight,
                            convertedLossWeight,
                            secondConvertedLossWeight,
                            thirdConvertedLossWeight,
                            OutputUse.BOTH,
                            fused);
            collector.backward(objective);
            return new TrainingResult(
                    floatValues(outputs.get(0)),
                    floatValues(outputs.get(1)),
                    gradientValues(input),
                    gradientValues(weight),
                    gradientValues(bias));
        }
    }

    private static NDList layerNormAndCast(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape normalizedShape,
            DataType convertedType,
            boolean fused) {
        if (fused) {
            return LayerNorm.layerNormAndCast(
                    input, normalizedShape, weight, bias, EPSILON, convertedType);
        }
        NDArray normalized =
                LayerNorm.layerNorm(input, normalizedShape, weight, bias, EPSILON)
                        .singletonOrThrow();
        return new NDList(normalized, normalized.toType(convertedType, false));
    }

    private static NDArray trainingObjective(
            NDList outputs,
            NDArray normalizedLossWeight,
            NDArray convertedLossWeight,
            NDArray secondConvertedLossWeight,
            NDArray thirdConvertedLossWeight,
            OutputUse outputUse,
            boolean useReturnedOutputGraph) {
        if (!useReturnedOutputGraph) {
            return referenceTrainingObjective(
                    outputs.get(0),
                    normalizedLossWeight,
                    convertedLossWeight,
                    secondConvertedLossWeight,
                    thirdConvertedLossWeight,
                    outputUse);
        }
        switch (outputUse) {
            case NORMALIZED:
                return outputs.get(0).mul(normalizedLossWeight).sum();
            case CONVERTED:
                return outputs.get(1).mul(convertedLossWeight).sum();
            case BOTH:
                return outputs.get(0)
                        .mul(normalizedLossWeight)
                        .sum()
                        .add(outputs.get(1).mul(convertedLossWeight).sum());
            case CONVERTED_THREE_CONSUMERS:
                return outputs.get(1)
                        .mul(convertedLossWeight)
                        .sum()
                        .add(outputs.get(1).mul(secondConvertedLossWeight).sum())
                        .add(outputs.get(1).mul(thirdConvertedLossWeight).sum());
            default:
                throw new AssertionError(outputUse);
        }
    }

    private static NDArray referenceTrainingObjective(
            NDArray normalized,
            NDArray normalizedLossWeight,
            NDArray convertedLossWeight,
            NDArray secondConvertedLossWeight,
            NDArray thirdConvertedLossWeight,
            OutputUse outputUse) {
        NDArray convertedGradient = convertedLossWeight.toType(DataType.FLOAT32, false);
        switch (outputUse) {
            case NORMALIZED:
                return normalized.mul(normalizedLossWeight).sum();
            case CONVERTED:
                return normalized.mul(convertedGradient).sum();
            case BOTH:
                return normalized.mul(normalizedLossWeight.add(convertedGradient)).sum();
            case CONVERTED_THREE_CONSUMERS:
                return normalized
                        .mul(
                                convertedGradient
                                        .add(
                                                secondConvertedLossWeight.toType(
                                                        DataType.FLOAT32, false))
                                        .add(
                                                thirdConvertedLossWeight.toType(
                                                        DataType.FLOAT32, false)))
                        .sum();
            default:
                throw new AssertionError(outputUse);
        }
    }

    private static float trainingTolerance(
            DataType inputType, DataType parameterType, DataType convertedType) {
        if (inputType == DataType.BFLOAT16
                || parameterType == DataType.BFLOAT16
                || convertedType == DataType.BFLOAT16) {
            return 2.0e-2f;
        }
        return 6.0e-3f;
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
        return sequence(size, scale, 0);
    }

    private static float[] sequence(long size, float scale, int offset) {
        float[] values = new float[Math.toIntExact(size)];
        for (int index = 0; index < values.length; index++) {
            values[index] = ((index + offset) % 31 - 15) * scale;
        }
        return values;
    }

    private static float[] gradientValues(NDArray array) {
        return array.hasGradient() ? floatValues(array.getGradient()) : null;
    }

    private static float[] floatValues(NDArray array) {
        if (array.getDataType() == DataType.FLOAT32) {
            return array.toFloatArray();
        }
        try (NDArray converted = array.toType(DataType.FLOAT32, false)) {
            return converted.toFloatArray();
        }
    }

    private static void assertTrainingResult(
            TrainingResult actual, TrainingResult expected, float tolerance) {
        assertClose(actual.normalized, expected.normalized, tolerance);
        assertClose(actual.converted, expected.converted, tolerance);
        assertNullableClose(actual.inputGradient, expected.inputGradient, tolerance);
        assertNullableClose(actual.weightGradient, expected.weightGradient, tolerance);
        assertNullableClose(actual.biasGradient, expected.biasGradient, tolerance);
    }

    private static void assertNullableClose(float[] actual, float[] expected, float tolerance) {
        Assert.assertEquals(actual == null, expected == null);
        if (actual != null) {
            assertClose(actual, expected, tolerance);
        }
    }

    private static void assertClose(float[] actual, float[] expected, float tolerance) {
        Assert.assertEquals(actual.length, expected.length);
        for (int index = 0; index < actual.length; index++) {
            Assert.assertEquals(
                    actual[index], expected[index], tolerance, "mismatch at index " + index);
        }
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

    private static void requireRocm(Engine engine) {
        if (engine.getGpuCount() == 0 || JniUtils.getFusionBackend() != 2) {
            throw new SkipException("This autocast LayerNorm test requires PyTorch ROCm.");
        }
    }

    private enum OutputUse {
        NORMALIZED,
        CONVERTED,
        BOTH,
        CONVERTED_THREE_CONSUMERS
    }

    private static final class TrainingResult {

        private final float[] normalized;
        private final float[] converted;
        private final float[] inputGradient;
        private final float[] weightGradient;
        private final float[] biasGradient;

        private TrainingResult(
                float[] normalized,
                float[] converted,
                float[] inputGradient,
                float[] weightGradient,
                float[] biasGradient) {
            this.normalized = normalized;
            this.converted = converted;
            this.inputGradient = inputGradient;
            this.weightGradient = weightGradient;
            this.biasGradient = biasGradient;
        }
    }

    @FunctionalInterface
    private interface GpuTest {
        void run(Engine engine, NDManager manager, Device device);
    }
}
