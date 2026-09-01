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
        verify(Device.cpu(), DataType.FLOAT32, false);
        verify(Device.cpu(), DataType.FLOAT32, true);
        verifyBias(Device.cpu(), DataType.FLOAT32, false);
        verifyBias(Device.cpu(), DataType.FLOAT32, true);
    }

    @Test
    public void nativeKernelPreservesLowPrecisionBoundaries() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("GPU is unavailable");
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            verify(Device.gpu(), dataType, false);
            verify(Device.gpu(), dataType, true);
            verifyBias(Device.gpu(), dataType, false);
            verifyBias(Device.gpu(), dataType, true);
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

    private static void verify(Device device, DataType dataType, boolean masked) {
        try (NDManager manager = Engine.getInstance().newBaseManager(device)) {
            NDArray values =
                    manager.arange(BATCH * ITEMS * WIDTH)
                            .reshape(BATCH, ITEMS, WIDTH)
                            .sub(9.5f)
                            .div(7.0f)
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
            float[] expected = expected(values, residual, mask);

            NDArray result = NDArrays.addBroadcastResidualToOwnedAndSilu(values, residual, mask);

            Assert.assertSame(result, values);
            Assert.assertEquals(result.getShape(), new Shape(BATCH, ITEMS, WIDTH));
            Assert.assertEquals(
                    result.toType(DataType.FLOAT32, false).toFloatArray(),
                    expected,
                    dataType == DataType.FLOAT32 ? 1.0e-6f : 2.0e-3f);
        }
    }

    private static float[] expected(NDArray values, NDArray residual, NDArray mask) {
        NDArray activated = Activation.swish(values.add(residual), 1.0f);
        NDArray expected = mask == null ? activated : activated.mul(mask.expandDims(2));
        return expected.toType(DataType.FLOAT32, false).toFloatArray();
    }

    private static void verifyBias(Device device, DataType dataType, boolean masked) {
        try (NDManager manager = Engine.getInstance().newBaseManager(device)) {
            NDArray values =
                    manager.arange(BATCH * ITEMS * WIDTH)
                            .reshape(BATCH, ITEMS, WIDTH)
                            .sub(9.5f)
                            .div(7.0f)
                            .toType(dataType, false);
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
            NDArray activated = Activation.swish(staged, 1.0f);
            NDArray expected = mask == null ? activated : activated.mul(mask.expandDims(2));
            float[] expectedValues = expected.toType(DataType.FLOAT32, false).toFloatArray();

            NDArray result =
                    NDArrays.addBiasAndBroadcastResidualToOwnedAndSilu(
                            values, bias, residual, mask);

            Assert.assertSame(result, values);
            Assert.assertEquals(
                    result.toType(DataType.FLOAT32, false).toFloatArray(),
                    expectedValues,
                    dataType == DataType.FLOAT32 ? 1.0e-6f : 2.0e-3f);
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
