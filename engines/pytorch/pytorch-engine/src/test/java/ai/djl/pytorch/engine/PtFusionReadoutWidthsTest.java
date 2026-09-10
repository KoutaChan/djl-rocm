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
import ai.djl.engine.EngineException;
import ai.djl.engine.fusion.FusionConstantBindings;
import ai.djl.engine.fusion.FusionExecutable;
import ai.djl.engine.fusion.FusionFunctions;
import ai.djl.engine.fusion.FusionInvocation;
import ai.djl.engine.fusion.FusionOutputLease;
import ai.djl.engine.fusion.FusionPlan;
import ai.djl.engine.fusion.FusionRecipe;
import ai.djl.engine.fusion.FusionSession;
import ai.djl.engine.fusion.FusionSessionConfig;
import ai.djl.engine.fusion.FusionSingleQueryReadoutParameters;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDScope;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.util.ArrayList;
import java.util.List;

public class PtFusionReadoutWidthsTest {

    @DataProvider
    public Object[][] readoutWidths() {
        return new Object[][] {
            {256, 128, 4, 512, 19, DataType.FLOAT32},
            {256, 128, 4, 512, 19, DataType.FLOAT16},
            {256, 128, 4, 512, 19, DataType.BFLOAT16},
            {128, 64, 4, 192, 19, DataType.FLOAT32},
            {384, 192, 4, 256, 37, DataType.FLOAT32},
            {384, 192, 4, 256, 37, DataType.BFLOAT16},
            {416, 128, 2, 513, 17, DataType.FLOAT16},
            {416, 128, 2, 513, 17, DataType.FLOAT32},
            {65, 144, 3, 97, 259, DataType.FLOAT32},
            {17, 261, 1, 31, 11, DataType.FLOAT32},
            {7, 9, 3, 13, 5, DataType.BFLOAT16}
        };
    }

    @Test
    public void descriptorWorkspaceIncludesWideAttentionContext() {
        ReadoutFixture fixture = new ReadoutFixture(17, 261, 1, 31, 11, DataType.FLOAT32);
        Assert.assertEquals(PtFusionDescriptor.workspaceBytes(fixture.recipe), 7_536L);
        Assert.assertEquals(PtFusionDescriptor.persistentStorageBytes(fixture.recipe), 7_944L);
    }

    @Test
    public void gpuReadoutRejectsUnsupportedResourceRequirements() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
        }
        Device device = Device.gpu();
        ReadoutFixture[] fixtures = {
            new ReadoutFixture(128, 64, 4, 192, 6144, DataType.FLOAT32),
            new ReadoutFixture(8, 32768, 32768, 13, 1, DataType.FLOAT32)
        };
        String[] messages = {"shared-memory budget", "grid-y limit"};
        for (int index = 0; index < fixtures.length; ++index) {
            ReadoutFixture fixture = fixtures[index];
            EngineException failure =
                    Assert.expectThrows(
                            EngineException.class,
                            () -> engine.newFusionCompiler(device).prepare(fixture.recipe).close());
            Assert.assertTrue(failure.getMessage().contains(messages[index]), failure.getMessage());
        }
    }

    @Test(dataProvider = "readoutWidths")
    public void gpuReadoutWidthsMatchFunctionalReference(
            int hiddenWidth,
            int attentionWidth,
            int attentionHeads,
            int feedForwardWidth,
            int tokens,
            DataType dataType) {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
        }
        Device device = Device.gpu();
        ReadoutFixture fixture =
                new ReadoutFixture(
                        hiddenWidth,
                        attentionWidth,
                        attentionHeads,
                        feedForwardWidth,
                        tokens,
                        dataType);
        try (NDManager manager = engine.newBaseManager(device)) {
            FusionConstantBindings.Builder bindings =
                    FusionConstantBindings.builder(fixture.recipe);
            List<FusionSingleQueryReadoutParameters> parameters = new ArrayList<>();
            for (FusionRecipe.Constant[] constants : fixture.constants) {
                NDArray[] values = new NDArray[constants.length];
                for (int index = 0; index < constants.length; ++index) {
                    FusionRecipe.Constant constant = constants[index];
                    values[index] =
                            patterned(
                                    manager,
                                    constant.getSpec().getMaximumShape(),
                                    index + 3 + 17 * parameters.size(),
                                    index == 2 || index == 5
                                            ? 0.010f
                                            : index == 4 ? 0.020f : 0.002f);
                    if (index == 7 || index == 9 || index == 15) {
                        values[index] = values[index].add(1f);
                    }
                    values[index] = values[index].toType(constant.getSpec().getDataType(), false);
                    bindings.bind(constant, values[index]);
                }
                parameters.add(
                        new FusionSingleQueryReadoutParameters(
                                values[0],
                                values[1],
                                values[2],
                                values[3],
                                values[4],
                                values[5],
                                values[6],
                                values[7],
                                values[8],
                                values[9],
                                values[10],
                                values[11],
                                values[12],
                                values[13],
                                values[14],
                                values[15],
                                values[16]));
            }
            float[] memoryValues = new float[3 * tokens * hiddenWidth];
            float[] maskValues = new float[3 * tokens];
            for (int batch = 0; batch < 3; ++batch) {
                for (int token = 0; token < tokens; ++token) {
                    boolean valid = batch == 0 || (batch == 1 && token % 3 == 1);
                    maskValues[batch * tokens + token] = valid ? (token % 2 == 0 ? 2f : -1f) : 0f;
                    for (int feature = 0; feature < hiddenWidth; ++feature) {
                        int index = (batch * tokens + token) * hiddenWidth + feature;
                        memoryValues[index] =
                                valid
                                        ? (index % 37 - 18) * 0.015f
                                        : (feature % 3 == 0
                                                ? Float.NaN
                                                : (feature % 3 == 1
                                                        ? Float.POSITIVE_INFINITY
                                                        : Float.NEGATIVE_INFINITY));
                    }
                }
            }
            NDArray memory = manager.create(memoryValues, new Shape(3, tokens, hiddenWidth));
            NDArray mask = manager.create(maskValues, new Shape(3, tokens));
            NDArray query = patterned(manager, new Shape(3, 3, hiddenWidth), 7, 0.012f);
            try (FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                    FusionExecutable executable = plan.bind(bindings.build());
                    FusionSession session =
                            executable.newSession(
                                    manager,
                                    FusionSessionConfig.builder().optOutputSlotCount(1).build())) {
                for (int batch : new int[] {3, 1, 2, 3}) {
                    try (NDScope scope = new NDScope();
                            FusionInvocation invocation = session.acquire()) {
                        scope.suppressNotUsedWarning();
                        invocation.setInput(fixture.memory, memory);
                        invocation.setInput(fixture.query, query);
                        invocation.setInput(fixture.mask, mask);
                        invocation.setDimension(fixture.batch, batch);
                        try (FusionOutputLease lease = invocation.submit();
                                NDList expected =
                                        FusionFunctions.singleQueryCrossAttentionReadoutGroup(
                                                memory.get("0:{}", batch),
                                                query.get("0:{}", batch),
                                                mask.get("0:{}", batch),
                                                2,
                                                attentionHeads,
                                                1e-5f,
                                                parameters)) {
                            lease.synchronize();
                            for (int index = 0; index < fixture.outputs.size(); ++index) {
                                NDArray actual =
                                        lease.get(fixture.outputs.get(index)).get("0:{}", batch);
                                assertClose(actual, expected.get(index), dataType);
                            }
                        }
                    }
                }
            }
        }
    }

    private static void assertClose(NDArray actual, NDArray expected, DataType dataType) {
        Assert.assertEquals(actual.getShape(), expected.getShape());
        float[] actualValues = actual.toType(DataType.FLOAT32, false).toFloatArray();
        float[] expectedValues = expected.toType(DataType.FLOAT32, false).toFloatArray();
        float tolerance =
                dataType == DataType.FLOAT32
                        ? 1e-3f
                        : (dataType == DataType.FLOAT16 ? 6e-2f : 2e-1f);
        for (int index = 0; index < actualValues.length; ++index) {
            Assert.assertTrue(Float.isFinite(actualValues[index]), "Non-finite output at " + index);
            Assert.assertTrue(
                    Float.isFinite(expectedValues[index]), "Non-finite reference at " + index);
            Assert.assertEquals(
                    actualValues[index],
                    expectedValues[index],
                    tolerance,
                    "Readout width parity failed at " + index + " for " + dataType);
        }
    }

    private static NDArray patterned(NDManager manager, Shape shape, int offset, float scale) {
        return manager.arange((float) shape.size())
                .add(offset)
                .mod(31)
                .sub(15)
                .mul(scale)
                .reshape(shape);
    }

    private static final class ReadoutFixture {

        private final FusionRecipe.Dimension batch;
        private final FusionRecipe.Input memory;
        private final FusionRecipe.Input query;
        private final FusionRecipe.Input mask;
        private final List<FusionRecipe.Constant[]> constants = new ArrayList<>();
        private final List<FusionRecipe.Output> outputs = new ArrayList<>();
        private final FusionRecipe recipe;

        private ReadoutFixture(
                int hiddenWidth,
                int attentionWidth,
                int attentionHeads,
                int feedForwardWidth,
                int tokens,
                DataType dataType) {
            FusionRecipe.Builder builder = FusionRecipe.builder("readout-widths");
            batch = builder.addDimension("batch", 3);
            memory =
                    builder.addInput(
                            "memory",
                            FusionRecipe.TensorSpec.of(
                                    DataType.FLOAT32, batch, tokens, hiddenWidth));
            query =
                    builder.addInput(
                            "query",
                            FusionRecipe.TensorSpec.of(DataType.FLOAT32, batch, 3, hiddenWidth));
            mask =
                    builder.addInput(
                            "mask", FusionRecipe.TensorSpec.of(DataType.FLOAT32, batch, tokens));
            FusionRecipe.SingleQueryCrossAttentionReadoutGroupBuilder group =
                    builder.singleQueryCrossAttentionReadoutGroup(
                                    "readouts", memory, query, mask, attentionHeads)
                            .optQueryIndex(2);
            for (int readout = 0; readout < 2; ++readout) {
                int expansionWidth = feedForwardWidth + 5 * readout;
                Shape[] shapes = {
                    new Shape(hiddenWidth, 2L * hiddenWidth),
                    new Shape(hiddenWidth),
                    new Shape(attentionWidth, hiddenWidth),
                    new Shape(attentionWidth),
                    new Shape(2L * attentionWidth, hiddenWidth),
                    new Shape(hiddenWidth, attentionWidth),
                    new Shape(hiddenWidth),
                    new Shape(hiddenWidth),
                    new Shape(hiddenWidth),
                    new Shape(hiddenWidth),
                    new Shape(hiddenWidth),
                    new Shape(expansionWidth, hiddenWidth),
                    new Shape(expansionWidth),
                    new Shape(hiddenWidth, expansionWidth),
                    new Shape(hiddenWidth),
                    new Shape(hiddenWidth),
                    new Shape(hiddenWidth)
                };
                FusionRecipe.Constant[] values = new FusionRecipe.Constant[shapes.length];
                for (int index = 0; index < shapes.length; ++index) {
                    DataType type =
                            index == 7
                                            || index == 8
                                            || index == 9
                                            || index == 10
                                            || index == 15
                                            || index == 16
                                    ? DataType.FLOAT32
                                    : dataType;
                    values[index] =
                            builder.addConstant(
                                    "readout" + readout + "Parameter" + index,
                                    FusionRecipe.TensorSpec.fixed(type, shapes[index].getShape()));
                }
                constants.add(values);
                group.addReadout(
                        values[0],
                        values[1],
                        values[2],
                        values[3],
                        values[4],
                        values[5],
                        values[6],
                        values[7],
                        values[8],
                        values[9],
                        values[10],
                        values[11],
                        values[12],
                        values[13],
                        values[14],
                        values[15],
                        values[16]);
            }
            FusionRecipe.SingleQueryCrossAttentionReadoutGroup readouts = group.build();
            outputs.add(builder.addOutput("first", readouts.getReadoutState(0)));
            outputs.add(builder.addOutput("second", readouts.getReadoutState(1)));
            recipe = builder.build();
        }
    }
}
