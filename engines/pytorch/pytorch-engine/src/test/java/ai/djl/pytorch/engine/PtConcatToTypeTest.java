/*
 * Copyright 2026 KoutaChan.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
 * the specific language governing permissions and limitations under the License.
 */
package ai.djl.pytorch.engine;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.GradientCollector;
import ai.djl.util.Float16Utils;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

/** Verifies concatenation with an explicit output data type. */
public class PtConcatToTypeTest {

    @Test
    public void cpuFallbackConvertsAndPreservesGradients() {
        try (NDManager manager = Engine.getInstance().newBaseManager(Device.cpu())) {
            NDArray first =
                    manager.create(new float[] {1.25f, -2.5f, 3.75f, -4.5f}, new Shape(2, 2));
            NDArray second = manager.create(new float[] {5.5f, -6.25f}, new Shape(2, 1));
            first.setRequiresGradient(true);
            second.setRequiresGradient(true);

            NDArray actual;
            try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
                actual = NDArrays.concatToType(new NDList(first, second), 1, DataType.FLOAT32);
                collector.backward(actual.sum());
            }

            Assert.assertEquals(actual.getShape(), new Shape(2, 3));
            Assert.assertEquals(
                    actual.toFloatArray(), new float[] {1.25f, -2.5f, 5.5f, 3.75f, -4.5f, -6.25f});
            Assert.assertEquals(first.getGradient().toFloatArray(), new float[] {1, 1, 1, 1});
            Assert.assertEquals(second.getGradient().toFloatArray(), new float[] {1, 1});
        }
    }

    @Test
    public void conversionPreservesNanAndInfinitySemantics() {
        verifyNonFinite(Device.cpu());
        if (Engine.getInstance().getGpuCount() > 0) {
            verifyNonFinite(Device.gpu());
        }
    }

    @Test
    public void gpuPathMatchesExplicitConversionForMixedTypesAndStridedFallback() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("GPU is unavailable");
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            for (DataType outputType :
                    new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
                NDArray first = manager.arange(24).reshape(3, 4, 2).toType(DataType.FLOAT32, false);
                NDArray second =
                        manager.arange(36)
                                .reshape(3, 4, 3)
                                .sub(7.0f)
                                .toType(DataType.FLOAT16, false);
                NDList inputs = new NDList(first, second);
                NDArray expected =
                        NDArrays.concat(
                                new NDList(
                                        first.toType(outputType, false),
                                        second.toType(outputType, false)),
                                2);
                NDArray actual = NDArrays.concatToType(inputs, -1, outputType);
                Assert.assertEquals(actual.getDataType(), outputType);
                Assert.assertEquals(actual.getShape(), new Shape(3, 4, 5));
                Assert.assertEquals(
                        actual.toType(DataType.FLOAT32, false).toFloatArray(),
                        expected.toType(DataType.FLOAT32, false).toFloatArray());

                NDArray stridedFirst = first.swapAxes(0, 1);
                NDArray stridedSecond = second.swapAxes(0, 1);
                NDArray stridedExpected =
                        NDArrays.concat(
                                new NDList(
                                        stridedFirst.toType(outputType, false),
                                        stridedSecond.toType(outputType, false)),
                                2);
                NDArray stridedActual =
                        NDArrays.concatToType(
                                new NDList(stridedFirst, stridedSecond), 2, outputType);
                Assert.assertEquals(
                        stridedActual.toType(DataType.FLOAT32, false).toFloatArray(),
                        stridedExpected.toType(DataType.FLOAT32, false).toFloatArray());
            }
        }
    }

    @Test
    public void gpuProductionRankFourPathMatchesExplicitFloat32ToFloat16Pack() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("GPU is unavailable");
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            int rows = 5;
            int[] widths = {256, 64, 256, 64};
            NDList sources = new NDList(widths.length);
            float[] expected = new float[rows * 640];
            int offset = 0;
            for (int width : widths) {
                float[] values = new float[rows * width];
                for (int row = 0; row < rows; ++row) {
                    for (int column = 0; column < width; ++column) {
                        float value = row * 0.25f + column * 0.003f + offset + 0.0003f;
                        values[row * width + column] = value;
                        expected[row * 640 + offset + column] =
                                Float16Utils.halfToFloat(Float16Utils.floatToHalf(value));
                    }
                }
                sources.add(manager.create(values, new Shape(rows, 1, 1, width)));
                offset += width;
            }

            String profileFile = System.getProperty("ai.djl.pytorch.concatToType.profileFile");
            NDArray actual;
            if (profileFile != null) {
                JniUtils.startProfile(true, true, false);
            }
            try {
                actual = NDArrays.concatToType(sources, -1, DataType.FLOAT16);
            } finally {
                if (profileFile != null) {
                    JniUtils.stopProfile(profileFile);
                }
            }

            Assert.assertEquals(actual.getDataType(), DataType.FLOAT16);
            Assert.assertEquals(actual.getShape(), new Shape(rows, 1, 1, 640));
            Assert.assertEquals(actual.toFloatArray(), expected);
        }
    }

    @Test
    public void gpuPathSelectsTheInputDeviceWhenAnotherDeviceIsCurrent() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() < 2) {
            throw new SkipException("This concat-to-type test requires two PyTorch GPUs.");
        }
        Device inputDevice = Device.gpu(1);
        try (NDManager manager = engine.newBaseManager(inputDevice);
                PtStreamScope currentDevice = engine.newStreamScope(Device.gpu(0))) {
            Assert.assertNotNull(currentDevice);
            NDArray first =
                    manager.create(new float[] {1.25f, -2.5f, 3.75f, -4.5f}, new Shape(2, 2));
            NDArray second =
                    manager.create(new float[] {5.5f, -6.25f}, new Shape(2, 1));

            NDArray actual =
                    NDArrays.concatToType(
                            new NDList(first, second), 1, DataType.FLOAT16);

            Assert.assertEquals(actual.getDevice(), inputDevice);
            Assert.assertEquals(
                    actual.toFloatArray(),
                    new float[] {1.25f, -2.5f, 5.5f, 3.75f, -4.5f, -6.25f});
        }
    }

    private static void verifyNonFinite(Device device) {
        try (NDManager manager = Engine.getInstance().newBaseManager(device)) {
            NDArray first =
                    manager.create(
                            new float[] {Float.NaN, Float.POSITIVE_INFINITY}, new Shape(1, 2));
            NDArray second =
                    manager.create(new float[] {Float.NEGATIVE_INFINITY, 1.0003f}, new Shape(1, 2))
                            .toType(DataType.FLOAT16, false);
            NDArray actual = NDArrays.concatToType(new NDList(first, second), 1, DataType.FLOAT16);
            float[] values = actual.toType(DataType.FLOAT32, false).toFloatArray();

            Assert.assertTrue(Float.isNaN(values[0]));
            Assert.assertEquals(values[1], Float.POSITIVE_INFINITY);
            Assert.assertEquals(values[2], Float.NEGATIVE_INFINITY);
            Assert.assertEquals(values[3], second.toFloatArray()[1]);
        }
    }
}
