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
import ai.djl.engine.Autocast;
import ai.djl.engine.Engine;
import ai.djl.engine.fusion.FusionConstantBindings;
import ai.djl.engine.fusion.FusionExecutable;
import ai.djl.engine.fusion.FusionInvocation;
import ai.djl.engine.fusion.FusionOutputLease;
import ai.djl.engine.fusion.FusionPlan;
import ai.djl.engine.fusion.FusionRecipe;
import ai.djl.engine.fusion.FusionSession;
import ai.djl.engine.fusion.FusionSessionConfig;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.core.Linear;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.GradientCollector;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.nio.ByteBuffer;

/** Tests the projected-residual MLP functional operation and bounded fusion stage. */
public class ProjectedResidualMlpTest {

    @Test
    public void cpuFallbackMatchesPortableFormula() {
        try (NDManager manager = Engine.getInstance().newBaseManager(Device.cpu());
                NDArray input = values(manager, new Shape(2, 3, 4), 0.011f, DataType.FLOAT32);
                NDArray combinedWeight =
                        values(manager, new Shape(7, 4), -0.007f, DataType.FLOAT32);
                NDArray combinedBias = values(manager, new Shape(7), 0.005f, DataType.FLOAT32);
                NDArray outputWeight = values(manager, new Shape(2, 5), 0.009f, DataType.FLOAT32);
                NDArray actual =
                        NDArrays.projectedResidualMlp(
                                input, combinedWeight, combinedBias, outputWeight);
                NDArray combined =
                        Linear.linear(input, combinedWeight, combinedBias).singletonOrThrow();
                NDArray expected =
                        combined.get("...,0:2")
                                .add(
                                        Linear.linear(
                                                        Activation.swish(
                                                                combined.get("...,2:"), 1.0f),
                                                        outputWeight,
                                                        null)
                                                .singletonOrThrow())) {
            assertClose(actual, expected, 1e-6f);
        }
    }

    @Test
    public void descriptorPreservesDynamicPrefixAndStorageAccounting() {
        Fixture fixture = new Fixture(DataType.FLOAT32, 4, 3, 4, 5, 2);
        ByteBuffer descriptor = PtFusionDescriptor.encode(fixture.recipe);
        int commandOffset = Math.toIntExact(descriptor.getLong(14 * Long.BYTES));

        Assert.assertEquals(
                descriptor.getLong((commandOffset + 1) * Long.BYTES),
                PtFusionDescriptor.PROJECTED_RESIDUAL_MLP_V1);
        Assert.assertEquals(descriptor.getLong((commandOffset + 3) * Long.BYTES), 1L);
        Assert.assertEquals(descriptor.getLong((commandOffset + 4) * Long.BYTES), 4L);
        Assert.assertEquals(descriptor.getLong((commandOffset + 5) * Long.BYTES), 0L);
        Assert.assertEquals(PtFusionDescriptor.commandCount(fixture.recipe), 1);
        Assert.assertEquals(PtFusionDescriptor.persistentStorageBytes(fixture.recipe), 672L);
        Assert.assertEquals(PtFusionDescriptor.workspaceBytes(fixture.recipe), 576L);
        Assert.assertEquals(PtFusionDescriptor.executableStorageBytes(fixture.recipe), 0L);
    }

    @DataProvider
    public Object[][] trainingCases() {
        return new Object[][] {
            {DataType.FLOAT32, null, 0, 17, 5, 2},
            {DataType.FLOAT32, null, 1, 1, 1, 1},
            {DataType.FLOAT16, null, 17, 17, 7, 3},
            {DataType.BFLOAT16, null, 64, 64, 5, 6},
            {DataType.FLOAT32, DataType.BFLOAT16, 17, 256, 64, 64},
            {DataType.FLOAT32, DataType.FLOAT16, 257, 257, 5, 2}
        };
    }

    @Test(dataProvider = "trainingCases")
    public void gpuForwardAndBackwardMatchEager(
            DataType parameterType,
            DataType autocastType,
            int rows,
            int inputWidth,
            int hiddenWidth,
            int outputWidth) {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This test requires a PyTorch CUDA or ROCm device.");
        }
        Device device = Device.gpu(0);
        try (NDManager manager = engine.newBaseManager(device);
                NDArray input =
                        values(manager, new Shape(rows, inputWidth), 0.011f, parameterType);
                NDArray combinedWeight =
                        values(
                                manager,
                                new Shape(outputWidth + hiddenWidth, inputWidth),
                                -0.007f,
                                parameterType);
                NDArray combinedBias =
                        values(
                                manager,
                                new Shape(outputWidth + hiddenWidth),
                                0.005f,
                                parameterType);
                NDArray outputWeight =
                        values(
                                manager,
                                new Shape(outputWidth, hiddenWidth),
                                0.009f,
                                parameterType);
                NDArray lossWeight =
                        values(manager, new Shape(rows, outputWidth), 0.003f, parameterType);
                TrainingResult actual =
                        train(
                                engine,
                                device,
                                input,
                                combinedWeight,
                                combinedBias,
                                outputWeight,
                                lossWeight,
                                autocastType,
                                true);
                TrainingResult expected =
                        train(
                                engine,
                                device,
                                input,
                                combinedWeight,
                                combinedBias,
                                outputWeight,
                                lossWeight,
                                autocastType,
                                false)) {
            float tolerance =
                    parameterType == DataType.FLOAT32 && autocastType == null ? 2e-5f : 0.06f;
            assertClose(actual.output, expected.output, tolerance);
            assertClose(actual.inputGradient, expected.inputGradient, tolerance);
            assertClose(actual.combinedWeightGradient, expected.combinedWeightGradient, tolerance);
            assertClose(actual.combinedBiasGradient, expected.combinedBiasGradient, tolerance);
            assertClose(actual.outputWeightGradient, expected.outputWeightGradient, tolerance);
            if (autocastType != null && isRocm()) {
                Assert.assertTrue(
                        JniUtils.getGradientFunctionNames((PtNDArray) actual.output)
                                .contains("ProjectedResidualMlpFunction"));
            }
        }
    }

    @Test
    public void gpuRankThreeForwardAndBackwardMatchEager() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This test requires a PyTorch CUDA or ROCm device.");
        }
        Device device = Device.gpu(0);
        try (NDManager manager = engine.newBaseManager(device);
                NDArray input = values(manager, new Shape(2, 3, 17), 0.011f, DataType.FLOAT32);
                NDArray combinedWeight =
                        values(manager, new Shape(12, 17), -0.007f, DataType.FLOAT32);
                NDArray combinedBias = values(manager, new Shape(12), 0.005f, DataType.FLOAT32);
                NDArray outputWeight = values(manager, new Shape(5, 7), 0.009f, DataType.FLOAT32);
                NDArray lossWeight = values(manager, new Shape(2, 3, 5), 0.003f, DataType.FLOAT32);
                TrainingResult actual =
                        train(
                                engine,
                                device,
                                input,
                                combinedWeight,
                                combinedBias,
                                outputWeight,
                                lossWeight,
                                DataType.FLOAT16,
                                true);
                TrainingResult expected =
                        train(
                                engine,
                                device,
                                input,
                                combinedWeight,
                                combinedBias,
                                outputWeight,
                                lossWeight,
                                DataType.FLOAT16,
                                false)) {
            assertClose(actual.output, expected.output, 0.02f);
            assertClose(actual.inputGradient, expected.inputGradient, 0.02f);
            assertClose(actual.combinedWeightGradient, expected.combinedWeightGradient, 0.02f);
            assertClose(actual.combinedBiasGradient, expected.combinedBiasGradient, 0.02f);
            assertClose(actual.outputWeightGradient, expected.outputWeightGradient, 0.02f);
        }
    }

    @SuppressWarnings("try")
    @Test
    public void gpuOneForwardAndBackwardUseTheInputDeviceWhenGpuZeroIsCurrent() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() < 2) {
            throw new SkipException("This test requires two PyTorch CUDA or ROCm devices.");
        }
        Device targetDevice = Device.gpu(1);
        try (NDManager manager = engine.newBaseManager(targetDevice);
                NDArray input = values(manager, new Shape(64, 256), 0.011f, DataType.BFLOAT16);
                NDArray combinedWeight =
                        values(manager, new Shape(128, 256), -0.007f, DataType.BFLOAT16);
                NDArray combinedBias = values(manager, new Shape(128), 0.005f, DataType.BFLOAT16);
                NDArray outputWeight =
                        values(manager, new Shape(64, 64), 0.009f, DataType.BFLOAT16);
                NDArray lossWeight = values(manager, new Shape(64, 64), 0.003f, DataType.BFLOAT16);
                TrainingResult expected =
                        train(
                                engine,
                                targetDevice,
                                input,
                                combinedWeight,
                                combinedBias,
                                outputWeight,
                                lossWeight,
                                DataType.BFLOAT16,
                                false);
                PtStreamScope ignored = engine.newStreamScope(Device.gpu(0));
                TrainingResult actual =
                        train(
                                engine,
                                targetDevice,
                                input,
                                combinedWeight,
                                combinedBias,
                                outputWeight,
                                lossWeight,
                                DataType.BFLOAT16,
                                true)) {
            assertClose(actual.output, expected.output, 0.06f);
            assertClose(actual.inputGradient, expected.inputGradient, 0.06f);
            assertClose(actual.combinedWeightGradient, expected.combinedWeightGradient, 0.06f);
            assertClose(actual.combinedBiasGradient, expected.combinedBiasGradient, 0.06f);
            assertClose(actual.outputWeightGradient, expected.outputWeightGradient, 0.06f);
        }
    }

    @DataProvider
    public Object[][] fusionDataTypes() {
        return new Object[][] {
            {DataType.FLOAT32, 1},
            {DataType.FLOAT16, 2},
            {DataType.BFLOAT16, 4}
        };
    }

    @Test(dataProvider = "fusionDataTypes")
    public void gpuFusionMatchesFunctionalReference(DataType dataType, int batchSize) {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This test requires a PyTorch CUDA or ROCm device.");
        }
        Device device = Device.gpu(0);
        Fixture fixture = new Fixture(dataType, 4, 3, 4, 5, 2);
        try (NDManager manager = engine.newBaseManager(device);
                NDArray input = values(manager, new Shape(4, 3, 4), 0.013f, dataType);
                NDArray combinedWeight = values(manager, new Shape(7, 4), -0.009f, dataType);
                NDArray combinedBias = values(manager, new Shape(7), 0.006f, dataType);
                NDArray outputWeight = values(manager, new Shape(2, 5), 0.008f, dataType);
                NDArray reference =
                        NDArrays.projectedResidualMlp(
                                input.get("0:{}", batchSize),
                                combinedWeight,
                                combinedBias,
                                outputWeight);
                FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                FusionExecutable executable =
                        plan.bind(fixture.bindings(combinedWeight, combinedBias, outputWeight));
                FusionSession session =
                        executable.newSession(
                                manager,
                                FusionSessionConfig.builder().optOutputSlotCount(1).build());
                FusionInvocation invocation = session.acquire()) {
            invocation.setInput(fixture.input, input);
            invocation.setDimension(fixture.batch, batchSize);
            try (FusionOutputLease lease = invocation.submit()) {
                lease.synchronize();
                try (NDArray actual = lease.get(fixture.output).get("0:{}", batchSize)) {
                    assertClose(actual, reference, dataType == DataType.FLOAT32 ? 2e-4f : 0.04f);
                }
            }
        }
    }

    @SuppressWarnings("try")
    private static TrainingResult train(
            Engine engine,
            Device device,
            NDArray sourceInput,
            NDArray sourceCombinedWeight,
            NDArray sourceCombinedBias,
            NDArray sourceOutputWeight,
            NDArray lossWeight,
            DataType autocastType,
            boolean projectedResidualMlp) {
        NDArray input = sourceInput.duplicate();
        NDArray combinedWeight = sourceCombinedWeight.duplicate();
        NDArray combinedBias = sourceCombinedBias.duplicate();
        NDArray outputWeight = sourceOutputWeight.duplicate();
        input.setRequiresGradient(true);
        combinedWeight.setRequiresGradient(true);
        combinedBias.setRequiresGradient(true);
        outputWeight.setRequiresGradient(true);

        NDArray output;
        try (GradientCollector collector = engine.newGradientCollector();
                Autocast ignored =
                        autocastType == null
                                ? null
                                : engine.newAutocast(device, autocastType, true)) {
            if (projectedResidualMlp) {
                output =
                        NDArrays.projectedResidualMlp(
                                input, combinedWeight, combinedBias, outputWeight);
            } else {
                NDArray combined =
                        Linear.linear(input, combinedWeight, combinedBias).singletonOrThrow();
                long outputWidth = outputWeight.getShape().get(0);
                NDArray skip = combined.get("...,0:{}", outputWidth);
                NDArray hidden = combined.get("...,{}:", outputWidth);
                NDArray activated = Activation.swish(hidden, 1.0f);
                NDArray update = Linear.linear(activated, outputWeight, null).singletonOrThrow();
                output = skip.add(update);
            }
            collector.backward(output.mul(lossWeight).sum());
        }
        TrainingResult result =
                new TrainingResult(
                        output,
                        input.getGradient(),
                        combinedWeight.getGradient(),
                        combinedBias.getGradient(),
                        outputWeight.getGradient());
        input.close();
        combinedWeight.close();
        combinedBias.close();
        outputWeight.close();
        return result;
    }

    private static NDArray values(NDManager manager, Shape shape, float scale, DataType dataType) {
        float[] values = new float[Math.toIntExact(shape.size())];
        for (int index = 0; index < values.length; ++index) {
            values[index] = ((index % 31) - 15) * scale;
        }
        return manager.create(values, shape).toType(dataType, false);
    }

    private static void assertClose(NDArray actual, NDArray expected, float tolerance) {
        float[] actualValues = actual.toType(DataType.FLOAT32, false).toFloatArray();
        float[] expectedValues = expected.toType(DataType.FLOAT32, false).toFloatArray();
        Assert.assertEquals(actualValues.length, expectedValues.length);
        for (int index = 0; index < actualValues.length; ++index) {
            Assert.assertEquals(actualValues[index], expectedValues[index], tolerance);
        }
    }

    private static boolean isRocm() {
        return Engine.getInstance().getVersion().toLowerCase().contains("rocm");
    }

    private static final class TrainingResult implements AutoCloseable {

        private final NDArray output;
        private final NDArray inputGradient;
        private final NDArray combinedWeightGradient;
        private final NDArray combinedBiasGradient;
        private final NDArray outputWeightGradient;

        private TrainingResult(
                NDArray output,
                NDArray inputGradient,
                NDArray combinedWeightGradient,
                NDArray combinedBiasGradient,
                NDArray outputWeightGradient) {
            this.output = output;
            this.inputGradient = inputGradient;
            this.combinedWeightGradient = combinedWeightGradient;
            this.combinedBiasGradient = combinedBiasGradient;
            this.outputWeightGradient = outputWeightGradient;
        }

        @Override
        public void close() {
            output.close();
            inputGradient.close();
            combinedWeightGradient.close();
            combinedBiasGradient.close();
            outputWeightGradient.close();
        }
    }

    private static final class Fixture {

        private final FusionRecipe.Dimension batch;
        private final FusionRecipe.Input input;
        private final FusionRecipe.Constant combinedWeight;
        private final FusionRecipe.Constant combinedBias;
        private final FusionRecipe.Constant outputWeight;
        private final FusionRecipe.Output output;
        private final FusionRecipe recipe;

        private Fixture(
                DataType dataType,
                long maximumBatch,
                long prefix,
                long inputWidth,
                long hiddenWidth,
                long outputWidth) {
            FusionRecipe.Builder builder = FusionRecipe.builder("projected-residual-mlp");
            batch = builder.addDimension("batch", maximumBatch);
            input =
                    builder.addInput(
                            "input",
                            FusionRecipe.TensorSpec.of(dataType, batch, prefix, inputWidth));
            combinedWeight =
                    builder.addConstant(
                            "combinedWeight",
                            FusionRecipe.TensorSpec.fixed(
                                    dataType, outputWidth + hiddenWidth, inputWidth));
            combinedBias =
                    builder.addConstant(
                            "combinedBias",
                            FusionRecipe.TensorSpec.fixed(dataType, outputWidth + hiddenWidth));
            outputWeight =
                    builder.addConstant(
                            "outputWeight",
                            FusionRecipe.TensorSpec.fixed(dataType, outputWidth, hiddenWidth));
            FusionRecipe.ProjectedResidualMlp value =
                    builder.projectedResidualMlp("outputValue", input)
                            .setCombinedWeight(combinedWeight)
                            .setCombinedBias(combinedBias)
                            .setOutputWeight(outputWeight)
                            .build();
            output = builder.addOutput("output", value);
            recipe = builder.build();
        }

        private FusionConstantBindings bindings(
                NDArray combinedWeightArray, NDArray combinedBiasArray, NDArray outputWeightArray) {
            return FusionConstantBindings.builder(recipe)
                    .bind(combinedWeight, combinedWeightArray)
                    .bind(combinedBias, combinedBiasArray)
                    .bind(outputWeight, outputWeightArray)
                    .build();
        }
    }
}
