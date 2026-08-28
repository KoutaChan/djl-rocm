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
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.engine.EngineException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradScaler;
import ai.djl.training.GradientCollector;
import ai.djl.training.LocalParameterServer;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.util.PairList;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;

public class GradScalerTest {

    @Test
    public void unscalesFiniteGradientsAndGrows() {
        Engine engine = Engine.getInstance();
        verifyFiniteGradients(engine, Device.cpu());
        if (engine.getGpuCount() > 0) {
            verifyFiniteGradients(engine, Device.gpu());
        }
    }

    @Test
    public void detectsOverflowAndBacksOff() {
        Engine engine = Engine.getInstance();
        verifyOverflow(engine, Device.cpu());
        if (engine.getGpuCount() > 0) {
            verifyOverflow(engine, Device.gpu());
        }
    }

    @Test
    public void supportsRealFloatingGradientDataTypes() {
        Engine engine = Engine.getInstance();
        verifyGradientDataTypes(engine, Device.cpu());
        if (engine.getGpuCount() > 0) {
            verifyGradientDataTypes(engine, Device.gpu());
        }
    }

    @Test
    public void rejectsInvalidGradientDataTypeBeforeMutation() {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(Device.cpu());
                NDArray dense = manager.ones(new Shape(2));
                NDArray integer = manager.ones(new Shape(2), DataType.INT32)) {
            Assert.expectThrows(
                    IllegalArgumentException.class,
                    () -> engine.unscaleGradients(new NDList(dense, integer), 0.5f));
            Assert.assertEquals(dense.toFloatArray(), new float[] {1f, 1f});
        }
    }

    @Test
    public void supportsSparseGradients() {
        Engine engine = Engine.getInstance();
        verifySparseGradients(engine, Device.cpu());
        if (engine.getGpuCount() > 0) {
            verifySparseGradients(engine, Device.gpu());
        }
    }

    @Test
    public void detectsDuplicateSparseFloat16Overflow() {
        Engine engine = Engine.getInstance();
        verifyDuplicateSparseFloat16Overflow(engine, Device.cpu());
        if (engine.getGpuCount() > 0) {
            verifyDuplicateSparseFloat16Overflow(engine, Device.gpu());
        }
    }

    @Test
    public void detectsOverflowIntroducedByLocalGradientReduction() {
        Engine engine = Engine.getInstance();
        GradScaler scaler = GradScaler.builder().optInitialScale(8f).build();
        try (NDManager manager = engine.newBaseManager(Device.cpu());
                NDArray loss = manager.create(1f);
                NDArray scaledLoss = scaler.scale(loss);
                NDArray firstGradient = manager.create(new float[] {Float.MAX_VALUE});
                NDArray secondGradient = manager.create(new float[] {Float.MAX_VALUE});
                LocalParameterServer parameterServer =
                        new LocalParameterServer(
                                Optimizer.sgd()
                                        .setLearningRateTracker(Tracker.fixed(0.1f))
                                        .build())) {
            Assert.assertEquals(scaledLoss.getFloat(), 8f);
            parameterServer.prepareGradients(
                    "weight", new NDArray[] {firstGradient, secondGradient});
            Assert.assertTrue(Float.isInfinite(firstGradient.getFloat()));
            Assert.assertFalse(scaler.unscale(new NDList(firstGradient, secondGradient)));
            scaler.update();
            Assert.assertTrue(scaler.isLastStepSkipped());
        }
    }

    @Test
    public void localGradientPreparationIsNotAppliedTwice() {
        Engine engine = Engine.getInstance();
        Optimizer optimizer = Optimizer.sgd().setLearningRateTracker(Tracker.fixed(1f)).build();
        try (NDManager manager = engine.newBaseManager(Device.cpu());
                NDArray firstWeight = manager.create(new float[] {10f});
                NDArray secondWeight = manager.create(new float[] {10f});
                NDArray firstGradient = manager.create(new float[] {1f});
                NDArray secondGradient = manager.create(new float[] {2f});
                LocalParameterServer parameterServer = new LocalParameterServer(optimizer)) {
            NDArray[] gradients = {firstGradient, secondGradient};
            parameterServer.prepareGradients("weight", gradients);
            parameterServer.update("weight", gradients, new NDArray[] {firstWeight, secondWeight});

            Assert.assertEquals(firstGradient.getFloat(), 3f);
            Assert.assertEquals(firstWeight.getFloat(), 7f);
            Assert.assertEquals(secondWeight.getFloat(), 7f);
        }
    }

    @Test
    public void discardedLocalPreparationDoesNotAffectNextStep() {
        Engine engine = Engine.getInstance();
        Optimizer optimizer = Optimizer.sgd().setLearningRateTracker(Tracker.fixed(1f)).build();
        try (NDManager manager = engine.newBaseManager(Device.cpu());
                NDArray discardedFirstGradient = manager.create(new float[] {4f});
                NDArray discardedSecondGradient = manager.create(new float[] {5f});
                NDArray firstWeight = manager.create(new float[] {10f});
                NDArray secondWeight = manager.create(new float[] {10f});
                NDArray firstGradient = manager.create(new float[] {1f});
                NDArray secondGradient = manager.create(new float[] {2f});
                LocalParameterServer parameterServer = new LocalParameterServer(optimizer)) {
            parameterServer.prepareGradients(
                    "weight", new NDArray[] {discardedFirstGradient, discardedSecondGradient});
            parameterServer.finishGradientStep();

            NDArray[] gradients = {firstGradient, secondGradient};
            parameterServer.update("weight", gradients, new NDArray[] {firstWeight, secondWeight});

            Assert.assertEquals(firstGradient.getFloat(), 3f);
            Assert.assertEquals(firstWeight.getFloat(), 7f);
            Assert.assertEquals(secondWeight.getFloat(), 7f);
        }
    }

    @Test
    public void retainsTemporaryFirstGradientUntilLocalUpdate() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() < 2) {
            throw new SkipException("This integration test requires two PyTorch GPUs.");
        }

        Device firstDevice = Device.gpu(0);
        Device secondDevice = Device.gpu(1);
        if (!supportsBasicTensorOperation(engine, secondDevice)) {
            throw new SkipException("The second GPU cannot execute PyTorch tensor operations.");
        }
        MirrorRecordingBlock block = new MirrorRecordingBlock();
        GradScaler scaler = GradScaler.builder().optInitialScale(8f).build();
        DefaultTrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optDevices(new Device[] {firstDevice, secondDevice})
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)
                        .optOptimizer(
                                Optimizer.sgd().setLearningRateTracker(Tracker.fixed(1f)).build())
                        .optGradScaler(scaler);

        try (Model model =
                Model.newInstance(
                        "temporary-first-gradient", firstDevice, engine.getEngineName())) {
            model.setBlock(block);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(1));
                try (NDManager manager = trainer.getManager().newSubManager();
                        GradientCollector collector = trainer.newGradientCollector();
                        NDArray input = manager.ones(new Shape(1)).toDevice(secondDevice, false);
                        NDList output = trainer.forward(new NDList(input));
                        NDArray loss = output.head().sum()) {
                    collector.backward(loss);
                }

                PtNDArray firstParameter = (PtNDArray) block.getParameter().getArray();
                PtNDArray firstGradient = JniUtils.getGradient(firstParameter);
                if (firstGradient != null) {
                    firstGradient.close();
                }
                Assert.assertNull(firstGradient, "The first-device gradient must be undefined.");

                PtNDArray secondGradient =
                        JniUtils.getGradient((PtNDArray) block.getLastForwardParameter());
                Assert.assertNotNull(secondGradient);
                try {
                    Assert.assertEquals(secondGradient.getFloat(), 8f);
                    Assert.assertTrue(Float.isFinite(secondGradient.getFloat()));
                } finally {
                    secondGradient.close();
                }

                trainer.step();

                Assert.assertEquals(firstParameter.getFloat(), 0f, 1.0e-6f);
                Assert.assertEquals(block.getLastForwardParameter().getFloat(), 0f, 1.0e-6f);
                Assert.assertFalse(scaler.isLastStepSkipped());
            }
        }
    }

    @Test
    public void enforcesStepStateTransitions() {
        Engine engine = Engine.getInstance();
        GradScaler scaler = GradScaler.builder().optInitialScale(8f).build();
        try (NDManager manager = engine.newBaseManager(Device.cpu());
                NDArray loss = manager.create(1f);
                NDArray scaledLoss = scaler.scale(loss);
                NDArray gradient = manager.create(new float[] {8f})) {
            Assert.assertEquals(scaledLoss.getFloat(), 8f);
            Assert.assertTrue(scaler.unscale(new NDList(gradient)));
            Assert.assertThrows(IllegalStateException.class, () -> scaler.scale(loss));
            Assert.assertThrows(IllegalArgumentException.class, () -> scaler.update(false));
            scaler.update();
            Assert.assertEquals(scaler.getScale(), 8f);
        }
        Assert.assertThrows(IllegalStateException.class, scaler::update);
    }

    @Test
    public void rejectsCheckpointChangesDuringActiveStep() throws IOException {
        Engine engine = Engine.getInstance();
        GradScaler scaler = GradScaler.builder().optInitialScale(8f).build();
        Path path = Files.createTempFile("active-grad-scaler", ".bin");
        try (NDManager manager = engine.newBaseManager(Device.cpu());
                NDArray loss = manager.create(1f);
                NDArray scaledLoss = scaler.scale(loss);
                NDArray gradient = manager.create(new float[] {8f})) {
            Assert.assertEquals(scaledLoss.getFloat(), 8f);
            Assert.assertThrows(IllegalStateException.class, scaler::getState);
            Assert.assertThrows(IllegalStateException.class, () -> scaler.saveState(path));
            Assert.assertThrows(
                    IllegalStateException.class,
                    () -> scaler.loadState(new GradScaler.State(4f, 0)));
            Assert.assertTrue(scaler.unscale(new NDList(gradient)));
            scaler.update();
            scaler.saveState(path);
        } finally {
            Files.deleteIfExists(path);
        }
    }

    private static void verifyFiniteGradients(Engine engine, Device device) {
        GradScaler scaler = GradScaler.builder().optInitialScale(8f).optGrowthInterval(2).build();
        try (NDManager manager = engine.newBaseManager(device)) {
            for (int step = 0; step < 2; ++step) {
                try (NDArray loss = manager.create(1f);
                        NDArray scaledLoss = scaler.scale(loss);
                        NDArray gradient = manager.create(new float[] {8f, -16f})) {
                    Assert.assertEquals(scaledLoss.getFloat(), scaler.getScale());
                    Assert.assertTrue(scaler.unscale(new NDList(gradient)));
                    Assert.assertEquals(gradient.toFloatArray(), new float[] {1f, -2f}, 0f);
                    scaler.update(true);
                }
            }
        }
        Assert.assertEquals(scaler.getScale(), 16f);
        Assert.assertEquals(scaler.getGrowthCount(), 0);
        Assert.assertFalse(scaler.isLastStepSkipped());
    }

    private static boolean supportsBasicTensorOperation(Engine engine, Device device) {
        try (NDManager manager = engine.newBaseManager(device);
                NDArray value = manager.ones(new Shape(1));
                NDArray result = value.mul(2f)) {
            return result.getFloat() == 2f;
        } catch (EngineException e) {
            return false;
        }
    }

    private static void verifyOverflow(Engine engine, Device device) {
        GradScaler scaler = GradScaler.builder().optInitialScale(8f).build();
        try (NDManager manager = engine.newBaseManager(device);
                NDArray loss = manager.create(1f);
                NDArray scaledLoss = scaler.scale(loss);
                NDArray gradient = manager.create(new float[] {8f, Float.POSITIVE_INFINITY})) {
            Assert.assertEquals(scaledLoss.getFloat(), 8f);
            Assert.assertFalse(scaler.unscale(new NDList(gradient)));
            Assert.assertEquals(gradient.getFloat(0), 1f);
            Assert.assertTrue(Float.isInfinite(gradient.getFloat(1)));
            scaler.update(false);
        }
        Assert.assertEquals(scaler.getScale(), 4f);
        Assert.assertTrue(scaler.isLastStepSkipped());
    }

    private static void verifyGradientDataTypes(Engine engine, Device device) {
        DataType[] dataTypes = {
            DataType.FLOAT16, DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64
        };
        try (NDManager manager = engine.newBaseManager(device)) {
            for (DataType dataType : dataTypes) {
                GradScaler scaler = GradScaler.builder().optInitialScale(8f).build();
                try (NDArray loss = manager.create(1f);
                        NDArray scaledLoss = scaler.scale(loss);
                        NDArray gradient =
                                manager.create(new float[] {8f, -16f}).toType(dataType, false)) {
                    Assert.assertEquals(scaledLoss.getFloat(), 8f);
                    Assert.assertTrue(scaler.unscale(new NDList(gradient)), dataType.toString());
                    try (NDArray floatGradient = gradient.toType(DataType.FLOAT32, false)) {
                        Assert.assertEquals(
                                floatGradient.toFloatArray(),
                                new float[] {1f, -2f},
                                0f,
                                dataType.toString());
                    }
                    scaler.update(true);
                }
            }
        }
    }

    private static void verifySparseGradients(Engine engine, Device device) {
        GradScaler scaler = GradScaler.builder().optInitialScale(8f).build();
        try (NDManager manager = engine.newBaseManager(device);
                NDArray loss = manager.create(1f);
                NDArray scaledLoss = scaler.scale(loss);
                NDArray dense = manager.create(new float[] {0f, 8f, 0f, -16f});
                NDArray sparse = dense.toSparse(SparseFormat.COO)) {
            Assert.assertTrue(sparse.isSparse());
            Assert.assertEquals(scaledLoss.getFloat(), 8f);
            Assert.assertTrue(scaler.unscale(new NDList(sparse)));
            try (NDArray unscaled = sparse.toDense()) {
                Assert.assertEquals(unscaled.toFloatArray(), new float[] {0f, 1f, 0f, -2f}, 0f);
            }
            scaler.update(true);
        }
    }

    private static void verifyDuplicateSparseFloat16Overflow(Engine engine, Device device) {
        GradScaler scaler = GradScaler.builder().optInitialScale(8f).build();
        try (NDManager manager = engine.newBaseManager(device);
                NDArray loss = manager.create(1f);
                NDArray scaledLoss = scaler.scale(loss);
                NDArray sparseFloat =
                        manager.createCoo(
                                FloatBuffer.wrap(new float[] {40000f, 40000f}),
                                new long[][] {{0L, 0L}},
                                new Shape(1));
                NDArray sparse = sparseFloat.toType(DataType.FLOAT16, false)) {
            Assert.assertEquals(scaledLoss.getFloat(), 8f);
            Assert.assertFalse(scaler.unscale(new NDList(sparse)));
            scaler.update();
            Assert.assertEquals(scaler.getScale(), 4f);
        }
    }

    private static final class MirrorRecordingBlock extends AbstractBlock {

        private static final byte VERSION = 1;

        private Parameter parameter;
        private NDArray lastForwardParameter;

        private MirrorRecordingBlock() {
            super(VERSION);
            parameter =
                    addParameter(
                            Parameter.builder()
                                    .setName("weight")
                                    .setType(Parameter.Type.WEIGHT)
                                    .optShape(new Shape(1))
                                    .build());
        }

        @Override
        public Shape[] getOutputShapes(Shape[] inputShapes) {
            return inputShapes;
        }

        @Override
        protected NDList forwardInternal(
                ParameterStore parameterStore,
                NDList inputs,
                boolean training,
                PairList<String, Object> params) {
            NDArray input = inputs.singletonOrThrow();
            lastForwardParameter = parameterStore.getValue(parameter, input.getDevice(), training);
            return new NDList(input.mul(lastForwardParameter));
        }

        @Override
        public void initializeChildBlocks(
                NDManager manager, DataType dataType, Shape... inputShapes) {}

        private Parameter getParameter() {
            return parameter;
        }

        private NDArray getLastForwardParameter() {
            return lastForwardParameter;
        }
    }
}
