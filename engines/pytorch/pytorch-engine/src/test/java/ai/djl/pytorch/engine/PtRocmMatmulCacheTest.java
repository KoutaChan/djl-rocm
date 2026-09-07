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
import ai.djl.engine.InferenceMode;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.jni.JniUtils;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

import java.util.ArrayList;
import java.util.List;

/** Verifies descriptor reuse through actual inference matmul submissions and graph replay. */
@SuppressWarnings("try") // The scopes select inference mode and device streams.
public class PtRocmMatmulCacheTest {

    private static final DataType[] DATA_TYPES = {
        DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16
    };

    @Test
    public void changingRowsBatchStridesAndBroadcastPreservesResults() {
        PtEngine engine = requireRocm();
        for (int deviceIndex = 0; deviceIndex < engine.getGpuCount(); ++deviceIndex) {
            for (DataType dataType : DATA_TYPES) {
                try (NDManager manager = engine.newBaseManager(Device.gpu(deviceIndex));
                        InferenceMode ignored = engine.newInferenceMode()) {
                    for (boolean transposeRight : new boolean[] {false, true}) {
                        List<MatmulCase> cases = createCases(manager, dataType, transposeRight);
                        // Enqueue all shapes before reading any output. The last two cases revisit
                        // cached algorithms after their descriptor family has changed geometry.
                        List<NDArray> outputs = new ArrayList<>();
                        for (MatmulCase value : cases) {
                            outputs.add(value.run());
                        }
                        for (int index = 0; index < cases.size(); ++index) {
                            assertResult(outputs.get(index), cases.get(index), dataType, 1f);
                        }
                    }
                }
            }
        }
    }

    @Test
    public void graphReplaySurvivesDescriptorReconfigurationAndEviction() {
        PtEngine engine = requireRocm();
        for (DataType dataType : DATA_TYPES) {
            try (NDManager manager = engine.newBaseManager(Device.gpu(0));
                    InferenceMode ignored = engine.newInferenceMode();
                    PtAcceleratorGraph graph = engine.newAcceleratorGraph(Device.gpu(0))) {
                List<MatmulCase> cases = new ArrayList<>();
                MatmulCase first = createCase(manager, dataType, 3, 3, 8, 0, false, false, false);
                cases.add(first);
                cases.add(createCase(manager, dataType, 1, 7, 8, 3, false, false, true));
                // Different output widths exercise enough descriptor families to evict early
                // entries while the graph still retains the already-enqueued kernel arguments.
                for (int columns = 16; columns < 280; ++columns) {
                    cases.add(createCase(manager, dataType, 1, 2, columns, 0, false, false, true));
                }
                cases.add(first);
                for (MatmulCase value : cases) {
                    try (NDArray warmup = value.run()) {
                        toFloatArray(warmup);
                    }
                }

                List<NDArray> outputs = new ArrayList<>();
                graph.beginCapture();
                for (MatmulCase value : cases) {
                    outputs.add(value.run());
                }
                graph.endCapture();
                graph.replay();
                assertResult(outputs.get(0), first, dataType, 1f);
                assertResult(outputs.get(1), cases.get(1), dataType, 1f);
                assertResult(outputs.get(outputs.size() - 1), first, dataType, 1f);

                first.left.muli(2f);
                graph.replay();
                assertResult(outputs.get(0), first, dataType, 2f);
                assertResult(outputs.get(1), cases.get(1), dataType, 1f);
                assertResult(outputs.get(outputs.size() - 1), first, dataType, 2f);
            }
        }
    }

    @Test
    public void independentStreamsPreserveUnsynchronizedShapeChanges() {
        PtEngine engine = requireRocm();
        Device device = Device.gpu(0);
        for (DataType dataType : DATA_TYPES) {
            try (NDManager manager = engine.newBaseManager(device);
                    InferenceMode ignored = engine.newInferenceMode();
                    PtStream firstStream = engine.newStream(device);
                    PtStream secondStream = engine.newStream(device);
                    PtEvent firstDone = firstStream.newEvent();
                    PtEvent secondDone = secondStream.newEvent()) {
                List<MatmulCase> firstCases = createCases(manager, dataType, false);
                List<MatmulCase> secondCases = createCases(manager, dataType, true);
                List<NDArray> firstOutputs = new ArrayList<>();
                List<NDArray> secondOutputs = new ArrayList<>();
                for (int index = 0; index < firstCases.size(); ++index) {
                    try (PtStreamScope scope = firstStream.openScope()) {
                        firstOutputs.add(firstCases.get(index).run());
                        firstDone.record();
                    }
                    try (PtStreamScope scope = secondStream.openScope()) {
                        secondOutputs.add(secondCases.get(index).run());
                        secondDone.record();
                    }
                }
                firstDone.synchronize();
                secondDone.synchronize();
                for (int index = 0; index < firstCases.size(); ++index) {
                    assertResult(firstOutputs.get(index), firstCases.get(index), dataType, 1f);
                    assertResult(secondOutputs.get(index), secondCases.get(index), dataType, 1f);
                }
            }
        }
    }

    private static List<MatmulCase> createCases(
            NDManager manager, DataType dataType, boolean transposeRight) {
        List<MatmulCase> cases = new ArrayList<>();
        cases.add(createCase(manager, dataType, 4, 9, 16, 0, transposeRight, false, false));
        cases.add(createCase(manager, dataType, 1, 3, 16, 7, transposeRight, false, false));
        cases.add(createCase(manager, dataType, 3, 6, 16, 5, transposeRight, true, false));
        cases.add(createCase(manager, dataType, 1, 2, 16, 0, transposeRight, false, true));
        cases.add(createCase(manager, dataType, 4, 9, 16, 8, transposeRight, false, false));
        cases.add(cases.get(0));
        cases.add(cases.get(3));
        return cases;
    }

    private static MatmulCase createCase(
            NDManager manager,
            DataType dataType,
            int batch,
            int rows,
            int columns,
            int padding,
            boolean transposeRight,
            boolean broadcastRight,
            boolean matrixOnly) {
        int reduction = 32;
        NDArray left =
                createInput(manager, dataType, new Shape(batch, rows + padding, reduction), 3)
                        .get(":, :" + rows + ", :");
        int rightBatch = broadcastRight ? 1 : batch;
        NDArray right;
        if (transposeRight) {
            right =
                    createInput(
                                    manager,
                                    dataType,
                                    new Shape(rightBatch, columns + padding, reduction),
                                    7)
                            .get(":, :" + columns + ", :")
                            .swapAxes(1, 2);
        } else {
            right =
                    createInput(
                                    manager,
                                    dataType,
                                    new Shape(rightBatch, reduction + padding, columns),
                                    7)
                            .get(":, :" + reduction + ", :");
        }
        if (matrixOnly) {
            left = left.squeeze(0);
            right = right.squeeze(0);
        }
        // The binary-fraction inputs are exactly representable in FP16 and BF16. Compute an
        // independent CPU FP32 reference, including logical views and broadcast batch selection.
        float[] leftValues = toFloatArray(left);
        float[] rightValues = toFloatArray(right);
        float[] expected = new float[batch * rows * columns];
        for (int b = 0; b < batch; ++b) {
            int rightOffset = (broadcastRight ? 0 : b) * reduction * columns;
            for (int row = 0; row < rows; ++row) {
                for (int column = 0; column < columns; ++column) {
                    float sum = 0f;
                    for (int k = 0; k < reduction; ++k) {
                        sum +=
                                leftValues[(b * rows + row) * reduction + k]
                                        * rightValues[rightOffset + k * columns + column];
                    }
                    expected[(b * rows + row) * columns + column] = sum;
                }
            }
        }
        Shape outputShape = matrixOnly ? new Shape(rows, columns) : new Shape(batch, rows, columns);
        return new MatmulCase(left, right, outputShape, expected);
    }

    private static NDArray createInput(
            NDManager manager, DataType dataType, Shape shape, int seed) {
        float[] values = new float[Math.toIntExact(shape.size())];
        for (int index = 0; index < values.length; ++index) {
            values[index] = ((index * seed + 5) % 29 - 14) / 32f;
        }
        return manager.create(values, shape).toType(dataType, false);
    }

    private static float[] toFloatArray(NDArray value) {
        if (value.getDataType() == DataType.FLOAT32) {
            return value.toFloatArray();
        }
        try (NDArray converted = value.toType(DataType.FLOAT32, false)) {
            return converted.toFloatArray();
        }
    }

    private static void assertResult(
            NDArray actual, MatmulCase testCase, DataType dataType, float scale) {
        Assert.assertEquals(actual.getShape(), testCase.outputShape);
        Assert.assertEquals(actual.getDataType(), dataType);
        float[] values = toFloatArray(actual);
        float tolerance = dataType == DataType.FLOAT32 ? 1e-5f : 1e-2f;
        for (int index = 0; index < values.length; ++index) {
            Assert.assertEquals(values[index], testCase.expected[index] * scale, tolerance);
        }
    }

    private static PtEngine requireRocm() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0
                || JniUtils.getFusionBackend() != 2
                || !"1".equals(System.getenv("TORCH_BLAS_PREFER_HIPBLASLT"))) {
            throw new SkipException("This test requires ROCm and TORCH_BLAS_PREFER_HIPBLASLT=1.");
        }
        return engine;
    }

    private static final class MatmulCase {

        private NDArray left;
        private NDArray right;
        private Shape outputShape;
        private float[] expected;

        private MatmulCase(NDArray left, NDArray right, Shape outputShape, float[] expected) {
            this.left = left;
            this.right = right;
            this.outputShape = outputShape;
            this.expected = expected;
        }

        private NDArray run() {
            return left.matMul(right);
        }
    }
}
