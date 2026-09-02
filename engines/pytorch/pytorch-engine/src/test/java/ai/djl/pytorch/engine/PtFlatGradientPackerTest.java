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
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.core.Embedding;
import ai.djl.pytorch.engine.PtFlatGradientPacker.MissingGradientPolicy;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.GradientCollector;

import org.testng.Assert;
import org.testng.annotations.Test;

/** Tests packing existing leaf gradients into a caller-owned flat PyTorch tensor. */
@SuppressWarnings("try")
public class PtFlatGradientPackerTest {

    @Test
    public void packsInParameterOrderAndClearsLeafGradients() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        try (NDManager manager = engine.newBaseManager()) {
            NDArray vector = manager.create(new float[] {1f, 2f});
            NDArray matrix = manager.create(new float[] {3f, 4f, 5f, 6f}).reshape(2, 2);
            vector.setRequiresGradient(true);
            matrix.setRequiresGradient(true);
            try (GradientCollector collector = engine.newGradientCollector()) {
                collector.backward(microbatchLoss(vector, matrix, 2f, 3f, 0.5f));
            }

            NDArray destination = manager.full(new Shape(6), 9f);
            try (PtFlatGradientPacker packer =
                    engine.newFlatGradientPacker(new NDList(matrix, vector), destination)) {
                Assert.assertSame(packer.getDestination(), destination);
                packer.packAndClear();

                Assert.assertEquals(
                        destination.toFloatArray(),
                        new float[] {0.5f, 0.5f, 0.5f, 0.5f, 2f, 3f},
                        0f);
                assertNoLeafGradient(matrix);
                assertNoLeafGradient(vector);
            }
        }
    }

    @Test
    public void accumulatesMultipleMicrobatchesAndConvertsDataType() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        try (NDManager manager = engine.newBaseManager()) {
            NDArray parameter = manager.create(new float[] {1f, 2f, 3f});
            parameter.setRequiresGradient(true);
            NDArray destination = manager.ones(new Shape(3), DataType.FLOAT64);
            try (PtFlatGradientPacker packer =
                    engine.newFlatGradientPacker(new NDList(parameter), destination)) {
                backwardScaledSum(engine, parameter, 2f);
                packer.accumulateAndClear();
                Assert.assertEquals(destination.toDoubleArray(), new double[] {3d, 3d, 3d}, 0d);
                assertNoLeafGradient(parameter);

                backwardScaledSum(engine, parameter, -0.5f);
                packer.accumulateAndClear(MissingGradientPolicy.ERROR);
                Assert.assertEquals(
                        destination.toDoubleArray(), new double[] {2.5d, 2.5d, 2.5d}, 0d);
                assertNoLeafGradient(parameter);
            }
        }
    }

    @Test
    public void handlesMissingGradientsAccordingToPolicy() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        try (NDManager manager = engine.newBaseManager()) {
            NDArray used = manager.create(new float[] {1f, 2f});
            NDArray unused = manager.create(new float[] {3f, 4f});
            used.setRequiresGradient(true);
            unused.setRequiresGradient(true);
            backwardScaledSum(engine, used, 4f);

            NDArray destination = manager.full(new Shape(4), 7f);
            try (PtFlatGradientPacker packer =
                    engine.newFlatGradientPacker(new NDList(used, unused), destination)) {
                Assert.expectThrows(RuntimeException.class, packer::packAndClear);
                Assert.assertEquals(destination.toFloatArray(), new float[] {7f, 7f, 7f, 7f}, 0f);
                assertLeafGradient(used, new float[] {4f, 4f});
                assertNoLeafGradient(unused);

                packer.packAndClear(MissingGradientPolicy.ZERO);
                Assert.assertEquals(destination.toFloatArray(), new float[] {4f, 4f, 0f, 0f}, 0f);
                assertNoLeafGradient(used);
                assertNoLeafGradient(unused);

                destination.muli(0f).addi(3f);
                backwardScaledSum(engine, used, 0.25f);
                packer.accumulateAndClear(MissingGradientPolicy.ZERO);
                Assert.assertEquals(
                        destination.toFloatArray(), new float[] {3.25f, 3.25f, 3f, 3f}, 0f);
            }
        }
    }

    @Test
    public void zeroAndClearOperationsAreIndependent() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        try (NDManager manager = engine.newBaseManager()) {
            NDArray parameter = manager.create(new float[] {1f, 2f});
            parameter.setRequiresGradient(true);
            backwardScaledSum(engine, parameter, 5f);
            NDArray destination = manager.ones(new Shape(2));
            try (PtFlatGradientPacker packer =
                    engine.newFlatGradientPacker(new NDList(parameter), destination)) {
                packer.zeroDestination();
                Assert.assertEquals(destination.toFloatArray(), new float[2], 0f);
                assertLeafGradient(parameter, new float[] {5f, 5f});

                destination.addi(2f);
                packer.clearParameterGradients();
                Assert.assertEquals(destination.toFloatArray(), new float[] {2f, 2f}, 0f);
                assertNoLeafGradient(parameter);
            }
        }
    }

    @Test
    public void rejectsInvalidPlansAndReleasedResources() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        try (NDManager manager = engine.newBaseManager()) {
            NDArray first = manager.ones(new Shape(2));
            NDArray second = manager.ones(new Shape(2));
            first.setRequiresGradient(true);
            second.setRequiresGradient(true);

            Assert.expectThrows(
                    IllegalArgumentException.class,
                    () -> engine.newFlatGradientPacker(new NDList(), manager.zeros(new Shape(0))));
            Assert.expectThrows(
                    IllegalArgumentException.class,
                    () ->
                            engine.newFlatGradientPacker(
                                    new NDList(first, second), manager.zeros(new Shape(3))));
            Assert.expectThrows(
                    IllegalArgumentException.class,
                    () ->
                            engine.newFlatGradientPacker(
                                    new NDList(first, second), manager.zeros(new Shape(2, 2))));
            Assert.expectThrows(
                    IllegalArgumentException.class,
                    () ->
                            engine.newFlatGradientPacker(
                                    new NDList(first, second),
                                    manager.zeros(new Shape(4), DataType.INT32)));
            NDArray stridedDestination = manager.zeros(new Shape(4, 2)).get(":,0");
            Assert.expectThrows(
                    IllegalArgumentException.class,
                    () ->
                            engine.newFlatGradientPacker(
                                    new NDList(first, second), stridedDestination));
            expectNativePlanFailure(
                    () ->
                            engine.newFlatGradientPacker(
                                    new NDList(first, first), manager.zeros(new Shape(4))));

            PtFlatGradientPacker packer =
                    engine.newFlatGradientPacker(
                            new NDList(first, second), manager.zeros(new Shape(4)));
            packer.close();
            Assert.assertTrue(packer.isReleased());
            Assert.expectThrows(IllegalStateException.class, packer::getDestination);
            Assert.expectThrows(IllegalStateException.class, packer::packAndClear);
            Assert.expectThrows(IllegalStateException.class, packer::accumulateAndClear);
            Assert.expectThrows(IllegalStateException.class, packer::zeroDestination);
            Assert.expectThrows(IllegalStateException.class, packer::clearParameterGradients);
        }
    }

    @Test
    public void rejectsDifferentDevices() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager cpuManager = engine.newBaseManager(Device.cpu());
                NDManager gpuManager = engine.newBaseManager(Device.gpu(0))) {
            NDArray parameter = cpuManager.ones(new Shape(2));
            parameter.setRequiresGradient(true);
            Assert.expectThrows(
                    IllegalArgumentException.class,
                    () ->
                            engine.newFlatGradientPacker(
                                    new NDList(parameter), gpuManager.zeros(new Shape(2))));
        }
    }

    @Test
    public void sparseGradientsFailWithoutChangingState() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        try (NDManager manager = engine.newBaseManager()) {
            NDArray weight = manager.arange(12).toType(DataType.FLOAT32, false).reshape(4, 3);
            weight.setRequiresGradient(true);
            NDArray indices = manager.create(new long[] {0, 2});
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        Embedding.embedding(indices, weight, SparseFormat.COO).singletonOrThrow();
                collector.backward(output.sum());
            }

            NDArray destination = manager.full(new Shape(12), 6f);
            try (PtFlatGradientPacker packer =
                    engine.newFlatGradientPacker(new NDList(weight), destination)) {
                RuntimeException exception =
                        Assert.expectThrows(RuntimeException.class, packer::packAndClear);
                Assert.assertTrue(exception.getMessage().contains("sparse gradients"));
                Assert.assertEquals(destination.toFloatArray(), filled(12, 6f), 0f);
                try (PtNDArray gradient = JniUtils.getGradient((PtNDArray) weight)) {
                    Assert.assertNotNull(gradient);
                }
            }
        }
    }

    private static NDArray microbatchLoss(
            NDArray vector,
            NDArray matrix,
            float firstVectorWeight,
            float secondVectorWeight,
            float matrixWeight) {
        NDArray vectorWeights =
                vector.getManager().create(new float[] {firstVectorWeight, secondVectorWeight});
        return vector.mul(vectorWeights).sum().add(matrix.mul(matrixWeight).sum());
    }

    private static void backwardScaledSum(PtEngine engine, NDArray parameter, float scale) {
        try (GradientCollector collector = engine.newGradientCollector()) {
            collector.backward(parameter.mul(scale).sum());
        }
    }

    private static void expectNativePlanFailure(Runnable action) {
        Assert.expectThrows(RuntimeException.class, action::run);
    }

    private static void assertLeafGradient(NDArray parameter, float[] expected) {
        try (PtNDArray gradient = JniUtils.getGradient((PtNDArray) parameter)) {
            Assert.assertNotNull(gradient);
            Assert.assertEquals(gradient.toFloatArray(), expected, 0f);
        }
    }

    private static void assertNoLeafGradient(NDArray parameter) {
        try (PtNDArray gradient = JniUtils.getGradient((PtNDArray) parameter)) {
            Assert.assertNull(gradient);
        }
    }

    private static float[] filled(int size, float value) {
        float[] result = new float[size];
        java.util.Arrays.fill(result, value);
        return result;
    }
}
