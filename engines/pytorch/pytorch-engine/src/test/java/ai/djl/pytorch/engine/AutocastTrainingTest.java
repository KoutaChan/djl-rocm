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
import ai.djl.engine.Autocast;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Embedding;
import ai.djl.nn.core.Linear;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.GradScaler;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.listener.EvaluatorTrainingListener;
import ai.djl.training.loss.L2Loss;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.Batchifier;
import ai.djl.util.PairList;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@SuppressWarnings("try")
public class AutocastTrainingTest {

    private static final int GPU_DEVICE = 1;

    @Test
    public void easyTrainSupportsBfloat16Autocast() {
        runFiniteTraining(DataType.BFLOAT16);
    }

    @Test
    public void easyTrainSupportsFloat16AutocastAndScaling() {
        runFiniteTraining(DataType.FLOAT16);
    }

    @Test
    public void validationSupportsBfloat16Autocast() {
        runValidation(DataType.BFLOAT16);
    }

    @Test
    public void validationSupportsFloat16Autocast() {
        runValidation(DataType.FLOAT16);
    }

    @Test
    public void legacyCustomLoopSupportsBfloat16Autocast() {
        runLegacyCustomLoop(DataType.BFLOAT16);
    }

    @Test
    public void legacyCustomLoopSupportsFloat16AutocastAndScaling() {
        runLegacyCustomLoop(DataType.FLOAT16);
    }

    @Test
    public void outerAutocastCacheSupportsRepeatedBackward() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This autocast training test requires a PyTorch GPU.");
        }

        Device device = Device.gpu();
        try (NDManager manager = engine.newBaseManager(device);
                NDArray input = manager.ones(new Shape(2, 4), DataType.FLOAT32);
                NDArray weight = manager.ones(new Shape(4, 4), DataType.FLOAT32);
                NDArray bias = manager.zeros(new Shape(4), DataType.FLOAT32)) {
            weight.setRequiresGradient(true);
            bias.setRequiresGradient(true);

            try (Autocast outer = engine.newAutocast(device, DataType.BFLOAT16, true)) {
                for (int iteration = 0; iteration < 2; ++iteration) {
                    try (GradientCollector collector = engine.newGradientCollector();
                            NDArray output =
                                    Linear.linear(input, weight, bias).singletonOrThrow();
                            Autocast disabled =
                                    engine.newAutocast(
                                            device, DataType.BFLOAT16, false, true);
                            NDArray objective =
                                    output.toType(DataType.FLOAT32, false).sum()) {
                        Assert.assertEquals(output.getDataType(), DataType.BFLOAT16);
                        collector.backward(objective);
                    }
                }
            }

            float[] expectedWeightGradient = new float[16];
            Arrays.fill(expectedWeightGradient, 4.0f);
            Assert.assertEquals(weight.getGradient().toFloatArray(), expectedWeightGradient, 0.0f);
            Assert.assertEquals(
                    bias.getGradient().toFloatArray(), new float[] {4f, 4f, 4f, 4f}, 0.0f);
        }
    }

    @Test
    public void trainerPersistsGradScalerState() throws IOException {
        Engine engine = Engine.getInstance();
        Device device = Device.cpu();
        DefaultTrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optDevices(new Device[] {device})
                        .optAutocast(DataType.FLOAT16);
        Path stateFile = Files.createTempFile("trainer-grad-scaler", ".bin");

        try (Model model = Model.newInstance("scaler-state", device, engine.getEngineName())) {
            model.setBlock(Linear.builder().setUnits(1).build());
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(1));
                GradScaler scaler = trainer.getGradScaler().orElseThrow(AssertionError::new);
                scaler.loadState(new GradScaler.State(128f, 7));
                trainer.saveGradScalerState(stateFile);
                scaler.loadState(new GradScaler.State(2f, 0));
                trainer.loadGradScalerState(stateFile);

                Assert.assertEquals(scaler.getScale(), 128f);
                Assert.assertEquals(scaler.getGrowthCount(), 7);
            }
        } finally {
            Files.deleteIfExists(stateFile);
        }
    }

    @Test
    public void optimizerRunsOutsideAutocastWhenStepIsInsideCollector() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This autocast training test requires a PyTorch GPU.");
        }

        Device device = Device.gpu();
        RecordingOptimizer optimizer = new RecordingOptimizer();
        Block block = Linear.builder().setUnits(4).build();
        DefaultTrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optDevices(new Device[] {device})
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)
                        .optOptimizer(optimizer)
                        .optAutocast(DataType.FLOAT16);

        try (Model model =
                Model.newInstance(
                        "step-inside-autocast-collector", device, engine.getEngineName())) {
            model.setBlock(block);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 4));
                boolean autocastEnabledBefore = JniUtils.autocastIsEnabled(GPU_DEVICE);

                try (NDManager manager = trainer.getManager().newSubManager();
                        GradientCollector collector = trainer.newGradientCollector();
                        NDArray data =
                                manager.ones(new Shape(2, 4), DataType.FLOAT32)
                                        .mul(1.0e-2f)
                                        .toDevice(device, false);
                        NDArray label =
                                manager.zeros(new Shape(2, 4), DataType.FLOAT32)
                                        .toDevice(device, false);
                        NDList predictions = trainer.forward(new NDList(data));
                        NDArray loss = trainer.getLoss().evaluate(new NDList(label), predictions)) {
                    collector.backward(loss);
                    Assert.assertTrue(JniUtils.autocastIsEnabled(GPU_DEVICE));
                    trainer.step();
                    Assert.assertTrue(optimizer.wasInvoked());
                    Assert.assertFalse(optimizer.wasAutocastEnabledDuringUpdate());
                    Assert.assertTrue(JniUtils.autocastIsEnabled(GPU_DEVICE));
                    float[] weights = block.getParameters().valueAt(0).getArray().toFloatArray();
                    Assert.assertTrue(Float.isFinite(weights[0]));
                }

                Assert.assertEquals(JniUtils.autocastIsEnabled(GPU_DEVICE), autocastEnabledBefore);
            }
        }
    }

    @Test
    public void nonFiniteGradientSkipsCompleteStep() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This autocast training test requires a PyTorch GPU.");
        }

        Device device = Device.gpu();
        GradScaler scaler = GradScaler.builder().optInitialScale(8f).build();
        Block block = Linear.builder().setUnits(4).build();
        DefaultTrainingConfig config =
                new DefaultTrainingConfig(new OverflowLoss())
                        .optDevices(new Device[] {device})
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)
                        .optOptimizer(optimizer())
                        .optGradScaler(scaler)
                        .optAutocast(DataType.FLOAT16);

        try (Model model = Model.newInstance("fp16-overflow", device, engine.getEngineName())) {
            model.setBlock(block);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 4));
                NDArray weight = block.getParameters().valueAt(0).getArray();
                float[] before = weight.toFloatArray();

                try (Batch batch = batch(trainer.getManager(), device)) {
                    EasyTrain.trainBatch(trainer, batch);
                    trainer.step();
                }

                Assert.assertEquals(weight.toFloatArray(), before, 0f);
                Assert.assertTrue(scaler.isLastStepSkipped());
                Assert.assertEquals(scaler.getScale(), 4f);
                for (Parameter parameter : block.getParameters().values()) {
                    if (parameter.requiresGradient()) {
                        try (NDArray gradient = parameter.getArray().getGradient()) {
                            Assert.assertEquals(
                                    gradient.toFloatArray(),
                                    new float[Math.toIntExact(gradient.size())],
                                    0f);
                        }
                    }
                }
            }
        }
    }

    @Test
    public void nonFiniteSparseGradientIsClearedAfterSkippedStep() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This autocast training test requires a PyTorch GPU.");
        }

        Device device = Device.gpu();
        GradScaler scaler = GradScaler.builder().optInitialScale(8f).build();
        Block block = new SparseEmbeddingBlock();
        DefaultTrainingConfig config =
                new DefaultTrainingConfig(new OverflowLoss())
                        .optDevices(new Device[] {device})
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)
                        .optOptimizer(optimizer())
                        .optGradScaler(scaler)
                        .optAutocast(DataType.FLOAT16);

        try (Model model =
                Model.newInstance("fp16-sparse-overflow", device, engine.getEngineName())) {
            model.setBlock(block);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2));
                NDArray weight = block.getParameters().valueAt(0).getArray();
                float[] before = weight.toFloatArray();

                try (Batch batch = sparseBatch(trainer.getManager(), device)) {
                    EasyTrain.trainBatch(trainer, batch);
                    try (NDArray gradient = weight.getGradient()) {
                        Assert.assertTrue(gradient.isSparse());
                    }
                    trainer.step();
                }

                Assert.assertEquals(weight.toFloatArray(), before, 0f);
                Assert.assertTrue(scaler.isLastStepSkipped());
                try (NDArray gradient = weight.getGradient();
                        NDArray denseGradient = gradient.toDense()) {
                    Assert.assertTrue(gradient.isSparse());
                    Assert.assertEquals(
                            denseGradient.toFloatArray(),
                            new float[Math.toIntExact(denseGradient.size())],
                            0f);
                }
            }
        }
    }

    private static void runFiniteTraining(DataType autocastDataType) {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This autocast training test requires a PyTorch GPU.");
        }

        Device device = Device.gpu();
        RecordingL2Loss loss = new RecordingL2Loss();
        Block block = Linear.builder().setUnits(4).build();
        DefaultTrainingConfig config =
                new DefaultTrainingConfig(loss)
                        .optDevices(new Device[] {device})
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)
                        .optOptimizer(optimizer())
                        .optAutocast(autocastDataType)
                        .addTrainingListeners(new EvaluatorTrainingListener());

        try (Model model =
                Model.newInstance("autocast-" + autocastDataType, device, engine.getEngineName())) {
            model.setBlock(block);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 4));
                NDArray weight = block.getParameters().valueAt(0).getArray();
                float[] before = weight.toFloatArray();

                try (Batch batch = batch(trainer.getManager(), device)) {
                    EasyTrain.trainBatch(trainer, batch);
                    try (NDArray gradient = weight.getGradient()) {
                        Assert.assertEquals(gradient.getDataType(), DataType.FLOAT32);
                    }
                    trainer.step();
                }

                Assert.assertEquals(loss.getPredictionDataType(), autocastDataType);
                Assert.assertEquals(
                        loss.getProbeDataTypes(),
                        Arrays.asList(autocastDataType, autocastDataType));
                Assert.assertEquals(weight.getDataType(), DataType.FLOAT32);
                Assert.assertFalse(Arrays.equals(weight.toFloatArray(), before));
                if (autocastDataType == DataType.FLOAT16) {
                    GradScaler scaler = trainer.getGradScaler().orElseThrow(AssertionError::new);
                    Assert.assertFalse(scaler.isLastStepSkipped());
                    Assert.assertEquals(scaler.getScale(), 65536f);
                } else {
                    Assert.assertFalse(trainer.getGradScaler().isPresent());
                }
            }
        }
    }

    private static void runLegacyCustomLoop(DataType autocastDataType) {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This autocast training test requires a PyTorch GPU.");
        }

        Device device = Device.gpu();
        RecordingL2Loss loss = new RecordingL2Loss();
        Block block = Linear.builder().setUnits(4).build();
        DefaultTrainingConfig config =
                new DefaultTrainingConfig(loss)
                        .optDevices(new Device[] {device})
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)
                        .optOptimizer(optimizer())
                        .optAutocast(autocastDataType);

        try (Model model =
                Model.newInstance(
                        "legacy-autocast-" + autocastDataType, device, engine.getEngineName())) {
            model.setBlock(block);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 4));
                NDArray weight = block.getParameters().valueAt(0).getArray();
                float[] before = weight.toFloatArray();
                boolean autocastEnabledBefore = JniUtils.autocastIsEnabled(GPU_DEVICE);

                try (NDManager manager = trainer.getManager().newSubManager();
                        GradientCollector collector = trainer.newGradientCollector();
                        NDArray data =
                                manager.ones(new Shape(2, 4), DataType.FLOAT32)
                                        .mul(1.0e-2f)
                                        .toDevice(device, false);
                        NDArray label =
                                manager.zeros(new Shape(2, 4), DataType.FLOAT32)
                                        .toDevice(device, false)) {
                    Assert.assertTrue(JniUtils.autocastIsEnabled(GPU_DEVICE));
                    try (NDList predictions = trainer.forward(new NDList(data));
                            NDArray lossValue = loss.evaluate(new NDList(label), predictions)) {
                        collector.backward(lossValue);
                    }
                    Assert.assertTrue(JniUtils.autocastIsEnabled(GPU_DEVICE));
                }
                Assert.assertEquals(JniUtils.autocastIsEnabled(GPU_DEVICE), autocastEnabledBefore);

                trainer.step();
                Assert.assertEquals(loss.getPredictionDataType(), autocastDataType);
                Assert.assertEquals(loss.getProbeDataTypes(), Arrays.asList(autocastDataType));
                Assert.assertFalse(Arrays.equals(weight.toFloatArray(), before));
                if (autocastDataType == DataType.FLOAT16) {
                    Assert.assertTrue(trainer.getGradScaler().isPresent());
                    Assert.assertFalse(
                            trainer.getGradScaler()
                                    .orElseThrow(AssertionError::new)
                                    .isLastStepSkipped());
                } else {
                    Assert.assertFalse(trainer.getGradScaler().isPresent());
                }
            }
        }
    }

    private static void runValidation(DataType autocastDataType) {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This autocast training test requires a PyTorch GPU.");
        }

        Device device = Device.gpu();
        RecordingL2Loss loss = new RecordingL2Loss();
        Block block = Linear.builder().setUnits(4).build();
        DefaultTrainingConfig config =
                new DefaultTrainingConfig(loss)
                        .optDevices(new Device[] {device})
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)
                        .optOptimizer(optimizer())
                        .optAutocast(autocastDataType)
                        .addTrainingListeners(new EvaluatorTrainingListener());

        try (Model model =
                Model.newInstance(
                        "validation-autocast-" + autocastDataType,
                        device,
                        engine.getEngineName())) {
            model.setBlock(block);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(2, 4));
                try (Batch batch = batch(trainer.getManager(), device)) {
                    EasyTrain.validateBatch(trainer, batch);
                }

                Assert.assertEquals(loss.getPredictionDataType(), autocastDataType);
                Assert.assertEquals(loss.getProbeDataTypes(), Arrays.asList(autocastDataType));
                Assert.assertEquals(
                        block.getParameters().valueAt(0).getArray().getDataType(),
                        DataType.FLOAT32);
                if (autocastDataType == DataType.FLOAT16) {
                    GradScaler scaler = trainer.getGradScaler().orElseThrow(AssertionError::new);
                    Assert.assertEquals(scaler.getScale(), 65536f);
                    Assert.assertFalse(scaler.isLastStepSkipped());
                }
            }
        }
    }

    private static Batch batch(NDManager trainerManager, Device device) {
        NDManager batchManager = trainerManager.newSubManager();
        NDArray data =
                batchManager
                        .ones(new Shape(2, 4), DataType.FLOAT32)
                        .mul(1.0e-2f)
                        .toDevice(device, false);
        NDArray label =
                batchManager.zeros(new Shape(2, 4), DataType.FLOAT32).toDevice(device, false);
        return new Batch(
                batchManager,
                new NDList(data),
                new NDList(label),
                2,
                Batchifier.STACK,
                Batchifier.STACK,
                0,
                1);
    }

    private static Batch sparseBatch(NDManager trainerManager, Device device) {
        NDManager batchManager = trainerManager.newSubManager();
        NDArray data = batchManager.create(new long[] {1L, 3L}).toDevice(device, false);
        NDArray label =
                batchManager.zeros(new Shape(2, 4), DataType.FLOAT32).toDevice(device, false);
        return new Batch(
                batchManager,
                new NDList(data),
                new NDList(label),
                2,
                Batchifier.STACK,
                Batchifier.STACK,
                0,
                1);
    }

    private static Optimizer optimizer() {
        return Optimizer.sgd().setLearningRateTracker(Tracker.fixed(1.0e-2f)).build();
    }

    private static final class RecordingL2Loss extends L2Loss {

        private DataType predictionDataType;
        private List<DataType> probeDataTypes = new ArrayList<>();

        private RecordingL2Loss() {
            super("recording-l2");
        }

        @Override
        public NDArray evaluate(NDList label, NDList prediction) {
            predictionDataType = prediction.head().getDataType();
            try (NDArray transposed = label.head().transpose();
                    NDArray probe = label.head().matMul(transposed)) {
                probeDataTypes.add(probe.getDataType());
            }
            return super.evaluate(label, prediction);
        }

        private DataType getPredictionDataType() {
            return predictionDataType;
        }

        private List<DataType> getProbeDataTypes() {
            return probeDataTypes;
        }
    }

    private static final class OverflowLoss extends Loss {

        private OverflowLoss() {
            super("overflow-loss");
        }

        @Override
        public NDArray evaluate(NDList label, NDList prediction) {
            return prediction.head().sum().mul(Float.POSITIVE_INFINITY);
        }
    }

    private static final class SparseEmbeddingBlock extends AbstractBlock {

        private static final byte VERSION = 1;

        private Parameter embedding;

        private SparseEmbeddingBlock() {
            super(VERSION);
            embedding =
                    addParameter(
                            Parameter.builder()
                                    .setName("embedding")
                                    .setType(Parameter.Type.WEIGHT)
                                    .optShape(new Shape(8, 4))
                                    .build());
        }

        @Override
        public Shape[] getOutputShapes(Shape[] inputShapes) {
            return new Shape[] {inputShapes[0].addAll(new Shape(4))};
        }

        @Override
        protected NDList forwardInternal(
                ParameterStore parameterStore,
                NDList inputs,
                boolean training,
                PairList<String, Object> params) {
            NDArray input = inputs.singletonOrThrow();
            NDArray weight = parameterStore.getValue(embedding, input.getDevice(), training);
            return Embedding.embedding(input, weight, SparseFormat.COO);
        }

        @Override
        public void initializeChildBlocks(
                NDManager manager, DataType dataType, Shape... inputShapes) {}
    }

    private static final class RecordingOptimizer extends Optimizer {

        private final Optimizer delegate;
        private boolean invoked;
        private boolean autocastEnabledDuringUpdate;

        private RecordingOptimizer() {
            super(Optimizer.sgd());
            delegate = Optimizer.sgd().setLearningRateTracker(Tracker.fixed(1.0e-2f)).build();
        }

        @Override
        public void update(String parameterId, NDArray weight, NDArray grad) {
            invoked = true;
            autocastEnabledDuringUpdate |= JniUtils.autocastIsEnabled(GPU_DEVICE);
            delegate.update(parameterId, weight, grad);
        }

        private boolean wasInvoked() {
            return invoked;
        }

        private boolean wasAutocastEnabledDuringUpdate() {
            return autocastEnabledDuringUpdate;
        }
    }
}
