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
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.GradientCollector;

import org.testng.Assert;
import org.testng.annotations.Test;

/** Tests direct accumulation into a caller-owned flat PyTorch gradient tensor. */
@SuppressWarnings("try")
public class PtFlatGradientAccumulatorTest {

    @Test
    public void accumulatesMicrobatchesInParameterOrderWithoutLeafGradients() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        try (NDManager manager = engine.newBaseManager()) {
            NDArray vector = manager.create(new float[] {1f, 2f});
            NDArray matrix = manager.create(new float[] {3f, 4f, 5f, 6f}).reshape(2, 2);
            NDArray unused = manager.create(new float[] {7f});
            vector.setRequiresGradient(true);
            matrix.setRequiresGradient(true);
            unused.setRequiresGradient(true);

            NDArray referenceVector = manager.create(new float[] {1f, 2f});
            NDArray referenceMatrix = manager.create(new float[] {3f, 4f, 5f, 6f}).reshape(2, 2);
            referenceVector.setRequiresGradient(true);
            referenceMatrix.setRequiresGradient(true);
            try (GradientCollector collector = engine.newGradientCollector()) {
                collector.backward(microbatchLoss(referenceVector, referenceMatrix, 2f, 3f, 0.5f));
                collector.backward(microbatchLoss(referenceVector, referenceMatrix, -1f, 4f, 2f));
            }

            NDArray destination = manager.zeros(new Shape(7));
            try (PtFlatGradientAccumulator accumulator =
                    engine.newFlatGradientAccumulator(
                            new NDList(matrix, unused, vector), destination)) {
                Assert.assertSame(accumulator.getGradient(), destination);
                try (GradientCollector collector = accumulator.newGradientCollector()) {
                    collector.backward(microbatchLoss(vector, matrix, 2f, 3f, 0.5f));
                    collector.backward(microbatchLoss(vector, matrix, -1f, 4f, 2f));
                }

                float[] expected = new float[7];
                System.arraycopy(referenceMatrix.getGradient().toFloatArray(), 0, expected, 0, 4);
                System.arraycopy(referenceVector.getGradient().toFloatArray(), 0, expected, 5, 2);
                Assert.assertEquals(destination.toFloatArray(), expected, 0f);
                assertNoLeafGradient(matrix);
                assertNoLeafGradient(unused);
                assertNoLeafGradient(vector);

                accumulator.zeroGradients();
                Assert.assertEquals(destination.toFloatArray(), new float[7], 0f);
                try (GradientCollector collector = accumulator.newGradientCollector()) {
                    collector.backward(microbatchLoss(vector, matrix, 2f, 3f, 0.5f));
                    collector.zeroGradients();
                    Assert.assertEquals(destination.toFloatArray(), new float[7], 0f);
                }
            }
        }
    }

    @Test
    public void rejectsInvalidPlansAndConcurrentCollectors() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        try (NDManager manager = engine.newBaseManager()) {
            NDArray first = manager.ones(new Shape(2));
            NDArray second = manager.ones(new Shape(2));
            first.setRequiresGradient(true);
            second.setRequiresGradient(true);

            expectPlanFailure(
                    () ->
                            engine.newFlatGradientAccumulator(
                                    new NDList(), manager.zeros(new Shape(0))));
            expectPlanFailure(
                    () ->
                            engine.newFlatGradientAccumulator(
                                    new NDList(first, first), manager.zeros(new Shape(4))));
            expectPlanFailure(
                    () ->
                            engine.newFlatGradientAccumulator(
                                    new NDList(first, second), manager.zeros(new Shape(3))));
            expectPlanFailure(
                    () ->
                            engine.newFlatGradientAccumulator(
                                    new NDList(first, second), manager.zeros(new Shape(2, 2))));
            expectPlanFailure(
                    () ->
                            engine.newFlatGradientAccumulator(
                                    new NDList(first, second),
                                    manager.zeros(new Shape(4)).toType(DataType.FLOAT16, false)));
            NDArray stridedDestination = manager.zeros(new Shape(4, 2)).get(":,0");
            expectPlanFailure(
                    () ->
                            engine.newFlatGradientAccumulator(
                                    new NDList(first, second), stridedDestination));

            try (GradientCollector ignored = engine.newGradientCollector()) {
                NDArray nonLeaf = first.mul(2f);
                expectPlanFailure(
                        () ->
                                engine.newFlatGradientAccumulator(
                                        new NDList(nonLeaf), manager.zeros(new Shape(2))));
            }

            PtFlatGradientAccumulator accumulator =
                    engine.newFlatGradientAccumulator(
                            new NDList(first, second), manager.zeros(new Shape(4)));
            GradientCollector collector = accumulator.newGradientCollector();
            try {
                Assert.expectThrows(IllegalStateException.class, accumulator::newGradientCollector);
                Assert.expectThrows(IllegalStateException.class, accumulator::close);
            } finally {
                collector.close();
                accumulator.close();
            }
            Assert.assertTrue(accumulator.isReleased());
        }
    }

    @Test
    public void sparseGradientsFailFastWithoutChangingTheDestination() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        try (NDManager manager = engine.newBaseManager()) {
            NDArray weight = manager.arange(12).toType(DataType.FLOAT32, false).reshape(4, 3);
            weight.setRequiresGradient(true);
            NDArray destination = manager.zeros(new Shape(12));
            try (PtFlatGradientAccumulator accumulator =
                            engine.newFlatGradientAccumulator(new NDList(weight), destination);
                    GradientCollector collector = accumulator.newGradientCollector()) {
                NDArray indices = manager.create(new long[] {0, 2});
                NDArray output =
                        Embedding.embedding(indices, weight, SparseFormat.COO).singletonOrThrow();
                RuntimeException exception =
                        Assert.expectThrows(
                                RuntimeException.class, () -> collector.backward(output.sum()));
                Assert.assertTrue(exception.getMessage().contains("sparse gradients"));
                Assert.assertEquals(destination.toFloatArray(), new float[12], 0f);
                assertNoLeafGradient(weight);
            }
        }
    }

    @Test
    public void rejectsBackwardAndZeroOnDifferentAcceleratorStream() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        Device device = Device.gpu(0);
        try (NDManager manager = engine.newBaseManager(device);
                PtStream firstStream = engine.newStream(device);
                PtStream secondStream = engine.newStream(device);
                PtEvent firstComplete = firstStream.newEvent();
                PtEvent secondComplete = secondStream.newEvent()) {
            NDArray parameter = manager.create(new float[] {1f, 2f, 3f});
            parameter.setRequiresGradient(true);
            NDArray destination = manager.zeros(new Shape(3));
            try (PtFlatGradientAccumulator accumulator =
                    engine.newFlatGradientAccumulator(new NDList(parameter), destination)) {
                try (PtStreamScope ignored = firstStream.openScope();
                        GradientCollector collector = accumulator.newGradientCollector()) {
                    collector.backward(parameter.mul(2f).sum());
                    firstComplete.record();
                }

                try (PtStreamScope ignored = secondStream.openScope()) {
                    RuntimeException zeroFailure =
                            Assert.expectThrows(RuntimeException.class, accumulator::zeroGradients);
                    Assert.assertTrue(zeroFailure.getMessage().contains("accelerator stream"));
                    try (GradientCollector collector = accumulator.newGradientCollector()) {
                        RuntimeException backwardFailure =
                                Assert.expectThrows(
                                        RuntimeException.class,
                                        () -> collector.backward(parameter.mul(3f).sum()));
                        Assert.assertTrue(
                                backwardFailure.getMessage().contains("accelerator stream"));
                    }
                    secondComplete.record();
                }

                firstComplete.synchronize();
                secondComplete.synchronize();
                Assert.assertEquals(destination.toFloatArray(), new float[] {2f, 2f, 2f}, 0f);
            }
        }
    }

    @Test
    public void independentAccumulatorsRunOnTwoAcceleratorStreams() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        Device device = Device.gpu(0);
        try (NDManager manager = engine.newBaseManager(device);
                PtStream firstStream = engine.newStream(device);
                PtStream secondStream = engine.newStream(device);
                PtEvent firstComplete = firstStream.newEvent();
                PtEvent secondComplete = secondStream.newEvent()) {
            NDArray firstParameter = manager.create(new float[] {1f, 2f, 3f});
            NDArray secondParameter = manager.create(new float[] {4f, 5f, 6f});
            firstParameter.setRequiresGradient(true);
            secondParameter.setRequiresGradient(true);
            NDArray firstDestination = manager.zeros(new Shape(3));
            NDArray secondDestination = manager.zeros(new Shape(3));
            try (PtFlatGradientAccumulator firstAccumulator =
                            engine.newFlatGradientAccumulator(
                                    new NDList(firstParameter), firstDestination);
                    PtFlatGradientAccumulator secondAccumulator =
                            engine.newFlatGradientAccumulator(
                                    new NDList(secondParameter), secondDestination)) {
                try (PtStreamScope ignored = firstStream.openScope();
                        GradientCollector collector = firstAccumulator.newGradientCollector()) {
                    collector.backward(firstParameter.mul(2f).sum());
                    firstComplete.record();
                }
                try (PtStreamScope ignored = secondStream.openScope();
                        GradientCollector collector = secondAccumulator.newGradientCollector()) {
                    collector.backward(secondParameter.mul(5f).sum());
                    secondComplete.record();
                }

                firstComplete.synchronize();
                secondComplete.synchronize();
                Assert.assertEquals(firstDestination.toFloatArray(), new float[] {2f, 2f, 2f}, 0f);
                Assert.assertEquals(secondDestination.toFloatArray(), new float[] {5f, 5f, 5f}, 0f);
                assertNoLeafGradient(firstParameter);
                assertNoLeafGradient(secondParameter);
            }
        }
    }

    @Test
    public void supportsAcceleratorFloatingPointTypes() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                NDArray parameter =
                        manager.create(new float[] {1f, 2f, 3f}).toType(dataType, false);
                parameter.setRequiresGradient(true);
                NDArray destination = manager.zeros(new Shape(3), dataType);
                try (PtFlatGradientAccumulator accumulator =
                                engine.newFlatGradientAccumulator(
                                        new NDList(parameter), destination);
                        GradientCollector collector = accumulator.newGradientCollector()) {
                    collector.backward(parameter.mul(3f).sum());
                    Assert.assertEquals(
                            destination.toType(DataType.FLOAT32, false).toFloatArray(),
                            new float[] {3f, 3f, 3f},
                            0f,
                            dataType.toString());
                    assertNoLeafGradient(parameter);
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

    private static void expectPlanFailure(Runnable action) {
        Assert.expectThrows(RuntimeException.class, action::run);
    }

    private static void assertNoLeafGradient(NDArray parameter) {
        try (PtNDArray gradient = JniUtils.getGradient((PtNDArray) parameter)) {
            Assert.assertNull(gradient);
        }
    }
}
