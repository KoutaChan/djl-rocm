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
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.pytorch.jni.JniUtils;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

/** Verifies the caller-owned broadcast residual SiLU epilogue. */
public class PtOwnedBroadcastResidualSiluTest {

    private static final int BATCH = 2;
    private static final int ITEMS = 3;
    private static final int WIDTH = 4;

    @Test
    public void cpuFallbackMatchesEagerCompositionWithAndWithoutMask() {
        verify(Device.cpu(), DataType.FLOAT32, false, false);
        verify(Device.cpu(), DataType.FLOAT32, true, false);
        verifyBias(Device.cpu(), DataType.FLOAT32, false, false);
        verifyBias(Device.cpu(), DataType.FLOAT32, true, false);
    }

    @Test
    public void nativeKernelPreservesLowPrecisionBoundaries() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("GPU is unavailable");
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            verify(Device.gpu(), dataType, false, false);
            verify(Device.gpu(), dataType, true, false);
            verifyBias(Device.gpu(), dataType, false, false);
            verifyBias(Device.gpu(), dataType, true, false);
            if (JniUtils.getFusionBackend() == 1) {
                verify(Device.gpu(), dataType, false, true);
                verify(Device.gpu(), dataType, true, true);
                verifyBias(Device.gpu(), dataType, false, true);
                verifyBias(Device.gpu(), dataType, true, true);
            }
        }
    }

    @Test
    public void biasEpiloguePreservesNanAndInfinitySemantics() {
        verifyBiasNonFinite(Device.cpu(), DataType.FLOAT32);
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() > 0) {
            for (DataType dataType :
                    new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
                verifyBiasNonFinite(Device.gpu(), dataType);
            }
        }
    }

    @Test
    public void cudaIrregularWidthsPreserveStagedSigmoidRounding() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0 || JniUtils.getFusionBackend() != 1) {
            throw new SkipException("This test requires the CUDA owned residual kernel.");
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            for (int width : new int[] {1, 33, 257, 4097}) {
                try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                    NDArray values =
                            manager.arange(6 * width)
                                    .mod(113)
                                    .sub(56)
                                    .div(17)
                                    .reshape(2, 3, width)
                                    .toType(dataType, false);
                    NDArray bias =
                            manager.arange(width).mod(7).sub(3).div(19).toType(dataType, false);
                    NDArray residual =
                            manager.arange(2 * width)
                                    .mod(11)
                                    .sub(5)
                                    .div(23)
                                    .reshape(2, 1, width)
                                    .toType(dataType, false);
                    NDArray mask =
                            manager.create(new float[] {0, -2, 1, 0.5f, 1, 2}, new Shape(2, 3))
                                    .toType(dataType, false);
                    NDArray updated = values.add(bias).add(residual);
                    float[] expected =
                            updated.mul(Activation.sigmoid(updated))
                                    .mul(mask.expandDims(2))
                                    .toType(DataType.FLOAT32, false)
                                    .toFloatArray();
                    NDArrays.addBiasAndBroadcastResidualToOwnedAndSilu(
                            values, bias, residual, mask);
                    float[] actual = values.toType(DataType.FLOAT32, false).toFloatArray();
                    if (dataType == DataType.FLOAT32) {
                        Assert.assertEquals(actual, expected, 1.0e-6f);
                    } else {
                        Assert.assertEquals(actual, expected);
                    }
                }
            }
        }
    }

    @Test
    public void cudaStridedFallbackAndMaskedNonFiniteValuesMatchStagedOperations() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0 || JniUtils.getFusionBackend() != 1) {
            throw new SkipException("This test requires CUDA.");
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            for (boolean strided : new boolean[] {false, true}) {
                try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                    NDArray values =
                            manager.create(
                                            new float[] {
                                                Float.NaN,
                                                Float.POSITIVE_INFINITY,
                                                Float.NEGATIVE_INFINITY,
                                                -0f,
                                                0f,
                                                0.5f
                                            },
                                            new Shape(1, 2, 3))
                                    .toType(dataType, false);
                    if (strided) {
                        values = values.transpose(0, 2, 1);
                    }
                    long items = values.getShape().get(1);
                    long width = values.getShape().get(2);
                    NDArray residual = manager.zeros(new Shape(1, 1, width), dataType);
                    NDArray mask = manager.zeros(new Shape(1, items), dataType);
                    NDArray updated = values.add(residual);
                    float[] expected =
                            updated.mul(Activation.sigmoid(updated))
                                    .mul(mask.expandDims(2))
                                    .toType(DataType.FLOAT32, false)
                                    .toFloatArray();
                    NDArrays.addBroadcastResidualToOwnedAndSilu(values, residual, mask);
                    float[] actual = values.toType(DataType.FLOAT32, false).toFloatArray();
                    for (int index = 0; index < actual.length; ++index) {
                        if (Float.isNaN(expected[index])) {
                            Assert.assertTrue(Float.isNaN(actual[index]));
                        } else {
                            Assert.assertEquals(
                                    Float.floatToRawIntBits(actual[index]),
                                    Float.floatToRawIntBits(expected[index]));
                        }
                    }
                }
            }
        }
    }

    @Test
    public void gpuEmptyInputsPreserveOwnedValues() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("GPU is unavailable");
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            for (Shape shape : new Shape[] {new Shape(0, 3, 33), new Shape(2, 0, 33)}) {
                try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                    NDArray residual =
                            manager.ones(new Shape(shape.get(0), 1, shape.get(2)), dataType);
                    NDArray bias = manager.ones(new Shape(shape.get(2)), dataType);
                    for (boolean masked : new boolean[] {false, true}) {
                        NDArray mask =
                                masked
                                        ? manager.zeros(
                                                new Shape(shape.get(0), shape.get(1)), dataType)
                                        : null;
                        for (boolean biased : new boolean[] {false, true}) {
                            NDArray values = manager.zeros(shape, dataType);
                            NDArray result =
                                    biased
                                            ? NDArrays.addBiasAndBroadcastResidualToOwnedAndSilu(
                                                    values, bias, residual, mask)
                                            : NDArrays.addBroadcastResidualToOwnedAndSilu(
                                                    values, residual, mask);

                            Assert.assertSame(result, values);
                            Assert.assertEquals(result.getShape(), shape);
                            Assert.assertEquals(result.getDataType(), dataType);
                            Assert.assertEquals(result.getDevice(), residual.getDevice());
                            Assert.assertEquals(
                                    result.toType(DataType.FLOAT32, false).toFloatArray().length,
                                    0);
                        }
                    }
                }
            }
        }
    }

    private static void verify(Device device, DataType dataType, boolean masked, boolean strided) {
        try (NDManager manager = Engine.getInstance().newBaseManager(device)) {
            NDArray values =
                    manager.arange(BATCH * ITEMS * WIDTH)
                            .reshape(BATCH, ITEMS, WIDTH)
                            .sub(9.5f)
                            .div(7.0f)
                            .toType(dataType, false);
            if (strided) {
                values = values.repeat(2, 2).get(":, :, ::2");
            }
            NDArray residual =
                    manager.create(
                                    new float[] {
                                        0.25f, -0.5f, 0.75f, -1.0f,
                                        -0.125f, 0.375f, -0.625f, 0.875f
                                    },
                                    new Shape(BATCH, 1, WIDTH))
                            .toType(dataType, false);
            NDArray mask =
                    masked
                            ? manager.create(
                                            new float[] {1, 0, 1, 0, 1, 1}, new Shape(BATCH, ITEMS))
                                    .toType(dataType, false)
                            : null;
            float[] expected = expected(values, residual, mask);

            NDArray result = NDArrays.addBroadcastResidualToOwnedAndSilu(values, residual, mask);

            Assert.assertSame(result, values);
            Assert.assertEquals(result.getShape(), new Shape(BATCH, ITEMS, WIDTH));
            Assert.assertEquals(
                    result.toType(DataType.FLOAT32, false).toFloatArray(),
                    expected,
                    tolerance(device, dataType));
        }
    }

    private static float[] expected(NDArray values, NDArray residual, NDArray mask) {
        NDArray activated = expectedActivation(values.add(residual));
        NDArray expected = mask == null ? activated : activated.mul(mask.expandDims(2));
        return expected.toType(DataType.FLOAT32, false).toFloatArray();
    }

    private static NDArray expectedActivation(NDArray staged) {
        if (staged.getDevice().isGpu() && JniUtils.getFusionBackend() == 2) {
            return Activation.swish(staged, 1.0f);
        }
        // The portable/CUDA path rounds sigmoid to the input dtype before multiplication.
        return staged.mul(Activation.sigmoid(staged));
    }

    private static float tolerance(Device device, DataType dataType) {
        if (dataType == DataType.FLOAT32) {
            return 1.0e-6f;
        }
        return device.isGpu() && JniUtils.getFusionBackend() == 1 ? 0f : 2.0e-3f;
    }

    private static void verifyBias(
            Device device, DataType dataType, boolean masked, boolean strided) {
        try (NDManager manager = Engine.getInstance().newBaseManager(device)) {
            NDArray values =
                    manager.arange(BATCH * ITEMS * WIDTH)
                            .reshape(BATCH, ITEMS, WIDTH)
                            .sub(9.5f)
                            .div(7.0f)
                            .toType(dataType, false);
            if (strided) {
                values = values.repeat(2, 2).get(":, :, ::2");
            }
            NDArray bias =
                    manager.create(new float[] {0.0625f, -0.1875f, 0.3125f, -0.4375f})
                            .toType(dataType, false);
            NDArray residual =
                    manager.create(
                                    new float[] {
                                        0.25f, -0.5f, 0.75f, -1.0f,
                                        -0.125f, 0.375f, -0.625f, 0.875f
                                    },
                                    new Shape(BATCH, 1, WIDTH))
                            .toType(dataType, false);
            NDArray mask =
                    masked
                            ? manager.create(
                                            new float[] {1, 0, 1, 0, 1, 1}, new Shape(BATCH, ITEMS))
                                    .toType(dataType, false)
                            : null;
            NDArray staged = values.add(bias).add(residual);
            NDArray activated = expectedActivation(staged);
            NDArray expected = mask == null ? activated : activated.mul(mask.expandDims(2));
            float[] expectedValues = expected.toType(DataType.FLOAT32, false).toFloatArray();

            NDArray result =
                    NDArrays.addBiasAndBroadcastResidualToOwnedAndSilu(
                            values, bias, residual, mask);

            Assert.assertSame(result, values);
            Assert.assertEquals(
                    result.toType(DataType.FLOAT32, false).toFloatArray(),
                    expectedValues,
                    tolerance(device, dataType));
        }
    }

    private static void verifyBiasNonFinite(Device device, DataType dataType) {
        try (NDManager manager = Engine.getInstance().newBaseManager(device)) {
            NDArray values =
                    manager.create(new float[] {0, 0, 0, 0}, new Shape(1, 1, WIDTH))
                            .toType(dataType, false);
            NDArray bias =
                    manager.create(
                                    new float[] {
                                        Float.NaN,
                                        Float.POSITIVE_INFINITY,
                                        Float.NEGATIVE_INFINITY,
                                        0.5f
                                    })
                            .toType(dataType, false);
            NDArray residual =
                    manager.create(new float[] {1, -1, 1, -0.25f}, new Shape(1, 1, WIDTH))
                            .toType(dataType, false);

            NDArray result =
                    NDArrays.addBiasAndBroadcastResidualToOwnedAndSilu(values, bias, residual);
            float[] actual = result.toType(DataType.FLOAT32, false).toFloatArray();

            Assert.assertTrue(Float.isNaN(actual[0]));
            Assert.assertEquals(actual[1], Float.POSITIVE_INFINITY);
            Assert.assertTrue(Float.isNaN(actual[2]));
            Assert.assertEquals(actual[3], 0.25f / (1.0f + (float) Math.exp(-0.25f)), 2.0e-3f);
        }
    }
}
