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

import org.testng.Assert;
import org.testng.annotations.Test;

/** Verifies padded batch-gather semantics, strided indices, and autograd fallback. */
public class PaddedBatchGatherTest {

    @Test
    public void oneDimensionalAndExplicitBatchGatherPreservePadding() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            NDArray source = sequentialSource(manager);
            NDArray storedIndices = int16(manager, new int[] {0, 3, 1, 2, 0, 3}, new Shape(2, 3));
            NDArray batchIndices = manager.create(new int[] {1, 0, 1});
            NDArray compactStoredIndices = int16(manager, new int[] {3, 0, 1}, new Shape(3));

            NDArray gathered = NDArrays.paddedBatchGather(source, storedIndices);
            NDArray compact =
                    NDArrays.paddedBatchGatherByBatchIndices(
                            source, batchIndices, compactStoredIndices);

            Assert.assertEquals(gathered.getShape(), new Shape(2, 3, 2));
            assertClose(
                    gathered.toFloatArray(),
                    new float[] {0f, 0f, 5f, 6f, 1f, 2f, 9f, 10f, 0f, 0f, 11f, 12f});
            Assert.assertEquals(compact.getShape(), new Shape(3, 2));
            assertClose(compact.toFloatArray(), new float[] {11f, 12f, 0f, 0f, 7f, 8f});
        }
    }

    @Test
    public void twoDimensionalGatherUsesBothStoredIndices() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            NDArray source =
                    manager.arange(1, 13).toType(DataType.FLOAT32, false).reshape(2, 2, 3, 1);
            NDArray outerStoredIndices = int16(manager, new int[] {1, 2, 0, 1}, new Shape(2, 2));
            NDArray innerStoredIndices = int16(manager, new int[] {3, 1, 2, 2}, new Shape(2, 2));

            NDArray gathered =
                    NDArrays.paddedBatchGather(source, outerStoredIndices, innerStoredIndices);

            Assert.assertEquals(gathered.getShape(), new Shape(2, 2, 1));
            assertClose(gathered.toFloatArray(), new float[] {3f, 4f, 0f, 8f});
        }
    }

    @Test
    public void nativeGpuGatherAcceptsStridedInt16AndMixedBatchIndices() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            NDArray source = sequentialSource(manager).toType(DataType.FLOAT16, false);
            NDArray indexFields =
                    manager.create(
                                    new int[] {0, 91, 3, 92, 1, 93, 2, 94, 0, 95, 3, 96},
                                    new Shape(2, 3, 2))
                            .toType(DataType.INT16, false);
            NDArray storedIndices = indexFields.get("...,0");

            NDArray gathered = NDArrays.paddedBatchGather(source, storedIndices);

            NDArray pairSource =
                    manager.arange(1, 13).toType(DataType.FLOAT16, false).reshape(2, 2, 3, 1);
            NDArray outerStoredIndices = int16(manager, new int[] {1, 2, 0, 1}, new Shape(2, 2));
            NDArray innerStoredIndices = int16(manager, new int[] {3, 1, 2, 2}, new Shape(2, 2));
            NDArray pairGathered =
                    NDArrays.paddedBatchGather(pairSource, outerStoredIndices, innerStoredIndices);

            NDArray compactBatchIndices = manager.create(new int[] {1, 0, 1});
            NDArray compactBatchIndices64 = manager.create(new long[] {1, 0, 1});
            NDArray compactStoredIndices = int16(manager, new int[] {3, 0, 1}, new Shape(3));
            NDArray compactGathered =
                    NDArrays.paddedBatchGatherByBatchIndices(
                            source, compactBatchIndices, compactStoredIndices);
            NDArray compactGatheredWithInt64Batch =
                    NDArrays.paddedBatchGatherByBatchIndices(
                            source, compactBatchIndices64, compactStoredIndices);
            NDArray int64Gathered =
                    NDArrays.paddedBatchGather(
                            source, manager.create(new long[] {0, 3, 1, 2, 0, 3}, new Shape(2, 3)));

            assertClose(
                    gathered.toFloatArray(),
                    new float[] {0f, 0f, 5f, 6f, 1f, 2f, 9f, 10f, 0f, 0f, 11f, 12f});
            assertClose(pairGathered.toFloatArray(), new float[] {3f, 4f, 0f, 8f});
            assertClose(compactGathered.toFloatArray(), new float[] {11f, 12f, 0f, 0f, 7f, 8f});
            assertClose(
                    compactGatheredWithInt64Batch.toFloatArray(),
                    new float[] {11f, 12f, 0f, 0f, 7f, 8f});
            assertClose(
                    int64Gathered.toFloatArray(),
                    new float[] {0f, 0f, 5f, 6f, 1f, 2f, 9f, 10f, 0f, 0f, 11f, 12f});
        }
    }

    @Test
    public void trainingFallbackAccumulatesRepeatedSourceGradients() {
        Engine engine = Engine.getInstance();
        verifyTrainingGradient(engine, Device.cpu());
        if (engine.getGpuCount() > 0) {
            verifyTrainingGradient(engine, Device.gpu());
        }
    }

    private static void verifyTrainingGradient(Engine engine, Device device) {
        try (NDManager manager = engine.newBaseManager(device);
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray source = manager.ones(new Shape(2, 3, 1));
            source.setRequiresGradient(true);
            NDArray storedIndices = int16(manager, new int[] {1, 1, 3, 0}, new Shape(2, 2));

            collector.backward(NDArrays.paddedBatchGather(source, storedIndices).sum());

            assertClose(source.getGradient().toFloatArray(), new float[] {2f, 0f, 0f, 0f, 0f, 1f});
        }
    }

    private static NDArray sequentialSource(NDManager manager) {
        return manager.arange(1, 13).toType(DataType.FLOAT32, false).reshape(2, 3, 2);
    }

    private static NDArray int16(NDManager manager, int[] values, Shape shape) {
        return manager.create(values, shape).toType(DataType.INT16, false);
    }

    private static void assertClose(float[] actual, float[] expected) {
        Assert.assertEquals(actual.length, expected.length);
        for (int index = 0; index < actual.length; index++) {
            Assert.assertEquals(actual[index], expected[index], 1e-3f, "index=" + index);
        }
    }
}
