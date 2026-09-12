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
import ai.djl.engine.fusion.FusionConstantBindings;
import ai.djl.engine.fusion.FusionExecutable;
import ai.djl.engine.fusion.FusionFunctions;
import ai.djl.engine.fusion.FusionInvocation;
import ai.djl.engine.fusion.FusionOutputLease;
import ai.djl.engine.fusion.FusionPlan;
import ai.djl.engine.fusion.FusionRecipe;
import ai.djl.engine.fusion.FusionSession;
import ai.djl.engine.fusion.FusionSessionConfig;
import ai.djl.engine.fusion.FusionTransformerBlockParameters;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PtFusionIndexedLocalWidthsTest {

    private static final float EPSILON = 1.0e-5f;

    @Test
    public void descriptorWorkspaceIncludesContextAndWideExpansion() {
        // Hidden, heads, attention, expansion, groups, tokens, batch, workspace bytes.
        int[][] configurations = {
            {256, 4, 64, 128, 4, 29, 2, 415_744},
            {384, 4, 64, 256, 3, 29, 2, 579_072},
            {257, 2, 130, 73, 1, 9, 2, 55_944},
            {129, 3, 99, 311, 2, 35, 2, 468_160},
            {8, 4, 64, 193, 1, 32, 2050, 103_071_744}
        };
        for (int[] configuration : configurations) {
            int hidden = configuration[0];
            int heads = configuration[1];
            int attention = configuration[2];
            int expansion = configuration[3];
            int groups = configuration[4];
            int tokens = configuration[5];
            int maximumBatch = configuration[6];
            FusionRecipe.Builder builder = FusionRecipe.builder("indexed-local-workspace");
            FusionRecipe.Dimension batch = builder.addDimension("batch", maximumBatch);
            FusionRecipe.Dimension active =
                    builder.addDimension("active", (long) maximumBatch * groups * tokens);
            FusionRecipe.Input input =
                    builder.addInput(
                            "input",
                            FusionRecipe.TensorSpec.of(
                                    DataType.FLOAT32, batch, groups, tokens, hidden));
            FusionRecipe.Input indices =
                    builder.addInput("indices", FusionRecipe.TensorSpec.of(DataType.INT32, active));
            Shape[] shapes = {
                new Shape(hidden),
                new Shape(hidden),
                new Shape(hidden),
                new Shape(hidden),
                new Shape(3L * attention, hidden),
                new Shape(hidden, attention),
                new Shape(hidden),
                new Shape(hidden),
                new Shape(hidden),
                new Shape(expansion, hidden),
                new Shape(expansion),
                new Shape(hidden, expansion),
                new Shape(hidden),
                new Shape(hidden),
                new Shape(hidden)
            };
            FusionRecipe.Constant[] parameters = new FusionRecipe.Constant[shapes.length];
            for (int index = 0; index < shapes.length; ++index) {
                parameters[index] =
                        builder.addConstant(
                                "parameter" + index,
                                FusionRecipe.TensorSpec.fixed(
                                        DataType.FLOAT32, shapes[index].getShape()));
            }
            FusionRecipe.IndexedLocalTransformerEncoder encoder =
                    builder.indexedLocalTransformerEncoder(
                                    "encoder", input, indices, heads, attention, expansion)
                            .setInputNormalization(parameters[0], parameters[1])
                            .setBlock(
                                    parameters[2],
                                    parameters[3],
                                    parameters[4],
                                    parameters[5],
                                    parameters[6],
                                    parameters[7],
                                    parameters[8],
                                    parameters[9],
                                    parameters[10],
                                    parameters[11],
                                    parameters[12],
                                    parameters[13],
                                    parameters[14])
                            .build();
            builder.addOutput("output", encoder);
            FusionRecipe recipe = builder.build();
            long expectedWorkspace = configuration[7];
            Assert.assertEquals(PtFusionDescriptor.workspaceBytes(recipe), expectedWorkspace);
            Assert.assertEquals(
                    PtFusionDescriptor.persistentStorageBytes(recipe),
                    expectedWorkspace
                            + (long) maximumBatch * groups * tokens * hidden * Float.BYTES);
        }
    }

    @DataProvider
    public Object[][] dimensions() {
        int[][] shapes = {
            {384, 4, 64, 256, 3, 29},
            {416, 8, 128, 513, 5, 7},
            {129, 3, 99, 311, 2, 35},
            {257, 2, 130, 73, 1, 9},
            {1025, 1, 17, 53, 3, 3},
            {1, 1, 1, 1, 3, 3},
            {256, 4, 64, 128, 4, 29},
            {128, 4, 64, 256, 4, 29},
            {128, 4, 64, 256, 3, 17},
            {128, 4, 64, 256, 3, 3},
            {128, 4, 64, 256, 3, 4},
            {256, 4, 64, 128, 3, 5},
            {384, 4, 64, 256, 3, 32},
            {384, 4, 64, 256, 3, 33}
        };
        List<Object[]> cases = new ArrayList<>();
        for (int index = 0; index < shapes.length; ++index) {
            for (DataType type :
                    new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
                int[] shape = shapes[index];
                cases.add(
                        new Object[] {
                            shape[0],
                            shape[1],
                            shape[2],
                            shape[3],
                            shape[4],
                            shape[5],
                            index % 2 != 0,
                            type
                        });
            }
        }
        return cases.toArray(new Object[0][]);
    }

    @Test(dataProvider = "dimensions")
    public void gpuIndexedLocalTransformerSupportsDynamicDimensions(
            int hiddenWidth,
            int heads,
            int attentionWidth,
            int feedForwardWidth,
            int groups,
            int tokens,
            boolean segmented,
            DataType dataType) {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
        }
        Device device = Device.gpu();
        try (NDManager manager = engine.newBaseManager(device)) {
            FusionRecipe.Builder builder = FusionRecipe.builder("indexed-local-widths");
            FusionRecipe.Dimension batch = builder.addDimension("batch", 2);
            FusionRecipe.Dimension active = builder.addDimension("active", 2L * groups * tokens);
            List<FusionRecipe.Input> inputs = new ArrayList<>();
            int[] segmentTokens =
                    segmented ? new int[] {tokens / 3, tokens - tokens / 3} : new int[] {tokens};
            for (int index = 0; index < segmentTokens.length; ++index) {
                inputs.add(
                        builder.addInput(
                                "input" + index,
                                FusionRecipe.TensorSpec.of(
                                        dataType,
                                        batch,
                                        groups,
                                        segmentTokens[index],
                                        hiddenWidth)));
            }
            DataType indexType = segmented ? DataType.INT64 : DataType.INT32;
            FusionRecipe.Input indices =
                    builder.addInput("indices", FusionRecipe.TensorSpec.of(indexType, active));
            Shape[] parameterShapes = {
                new Shape(hiddenWidth),
                new Shape(hiddenWidth),
                new Shape(hiddenWidth),
                new Shape(hiddenWidth),
                new Shape(3L * attentionWidth, hiddenWidth),
                new Shape(hiddenWidth, attentionWidth),
                new Shape(hiddenWidth),
                new Shape(hiddenWidth),
                new Shape(hiddenWidth),
                new Shape(feedForwardWidth, hiddenWidth),
                new Shape(feedForwardWidth),
                new Shape(hiddenWidth, feedForwardWidth),
                new Shape(hiddenWidth),
                new Shape(hiddenWidth),
                new Shape(hiddenWidth)
            };
            FusionRecipe.Constant[] constants = new FusionRecipe.Constant[parameterShapes.length];
            NDArray[] parameters = new NDArray[parameterShapes.length];
            NDArray[] referenceParameters = new NDArray[parameterShapes.length];
            // Eager LayerNorm requires matching parameter types. Exactly representable norm
            // coefficients keep its values identical while native bindings retain FLOAT32 norms.
            for (int index = 0; index < parameterShapes.length; ++index) {
                boolean normWeight = index == 0 || index == 2 || index == 7 || index == 13;
                boolean normBias = index == 1 || index == 3 || index == 8 || index == 14;
                DataType parameterType = normWeight || normBias ? DataType.FLOAT32 : dataType;
                constants[index] =
                        builder.addConstant(
                                "parameter" + index,
                                FusionRecipe.TensorSpec.fixed(
                                        parameterType, parameterShapes[index].getShape()));
                parameters[index] =
                        patterned(
                                manager,
                                parameterShapes[index],
                                index + 1,
                                normWeight ? 1f : 0f,
                                normWeight || normBias ? 1f / 128f : 0.0015f,
                                parameterType);
                referenceParameters[index] = parameters[index].toType(dataType, false);
            }
            FusionRecipe.IndexedLocalTransformerEncoder encoder =
                    builder.indexedLocalTransformerEncoder(
                                    "encoder",
                                    inputs,
                                    indices,
                                    heads,
                                    attentionWidth,
                                    feedForwardWidth)
                            .setInputNormalization(constants[0], constants[1])
                            .setBlock(
                                    constants[2],
                                    constants[3],
                                    constants[4],
                                    constants[5],
                                    constants[6],
                                    constants[7],
                                    constants[8],
                                    constants[9],
                                    constants[10],
                                    constants[11],
                                    constants[12],
                                    constants[13],
                                    constants[14])
                            .build();
            FusionRecipe.Output output = builder.addOutput("output", encoder);
            FusionRecipe recipe = builder.build();
            FusionConstantBindings.Builder bindings = FusionConstantBindings.builder(recipe);
            for (int index = 0; index < constants.length; ++index) {
                bindings.bind(constants[index], parameters[index]);
            }
            FusionTransformerBlockParameters block =
                    new FusionTransformerBlockParameters(
                            referenceParameters[2],
                            referenceParameters[3],
                            referenceParameters[4],
                            referenceParameters[5],
                            referenceParameters[6],
                            referenceParameters[7],
                            referenceParameters[8],
                            referenceParameters[9],
                            referenceParameters[10],
                            referenceParameters[11],
                            referenceParameters[12],
                            referenceParameters[13],
                            referenceParameters[14]);
            try (FusionPlan plan = engine.newFusionCompiler(device).prepare(recipe);
                    FusionExecutable executable = plan.bind(bindings.build());
                    FusionSession session =
                            executable.newSession(manager, FusionSessionConfig.builder().build())) {
                for (int pattern = 0; pattern < 4; ++pattern) {
                    int batchCount = pattern % 2 == 0 ? 2 : 1;
                    int denseRows = batchCount * groups * tokens;
                    long[] selected = new long[denseRows];
                    boolean[] present = new boolean[denseRows];
                    int count = 0;
                    for (int row = 0; row < denseRows; ++row) {
                        if (pattern == 0
                                || (pattern != 2 && row / tokens % 3 != 1 && row % 3 == 0)) {
                            selected[count++] = row;
                            present[row] = true;
                        }
                    }
                    try (NDManager invocationManager = manager.newSubManager();
                            FusionInvocation invocation = session.acquire()) {
                        NDArray routing =
                                invocationManager
                                        .create(Arrays.copyOf(selected, count))
                                        .toType(indexType, false);
                        float[] values = new float[denseRows * hiddenWidth];
                        for (int index = 0; index < values.length; ++index) {
                            values[index] =
                                    present[index / hiddenWidth]
                                            ? ((index * 17L + index / hiddenWidth) % 97 - 48)
                                                    * 0.02f
                                            : Float.NaN;
                        }
                        NDArray input =
                                invocationManager
                                        .create(
                                                values,
                                                new Shape(batchCount, groups, tokens, hiddenWidth))
                                        .toType(dataType, false);
                        NDList segments = new NDList();
                        int offset = 0;
                        for (int index = 0; index < segmentTokens.length; ++index) {
                            NDArray segment =
                                    input.get(":,:,{}:{},:", offset, offset + segmentTokens[index])
                                            .duplicate();
                            offset += segmentTokens[index];
                            segments.add(segment);
                            invocation.setInput(inputs.get(index), segment);
                        }
                        invocation.setInput(indices, routing);
                        invocation.setDimension(batch, batchCount);
                        invocation.setDimension(active, count);
                        try (FusionOutputLease lease = invocation.submit();
                                NDArray expected =
                                        FusionFunctions.indexedLocalTransformerEncoder(
                                                segments,
                                                routing,
                                                referenceParameters[0],
                                                referenceParameters[1],
                                                block,
                                                heads,
                                                EPSILON)) {
                            lease.synchronize();
                            try (NDArray actual = lease.get(output).get("0:{}", batchCount)) {
                                assertClose(actual, expected, present, hiddenWidth, dataType);
                            }
                        }
                    }
                }
            }
        }
    }

    private static NDArray patterned(
            NDManager manager, Shape shape, int seed, float offset, float scale, DataType type) {
        float[] values = new float[Math.toIntExact(shape.size())];
        for (int index = 0; index < values.length; ++index) {
            values[index] =
                    offset + ((index * (seed * 2L + 1) + index / 17 + seed) % 43 - 21) * scale;
        }
        return manager.create(values, shape).toType(type, false);
    }

    private static void assertClose(
            NDArray actual,
            NDArray expected,
            boolean[] present,
            int hiddenWidth,
            DataType dataType) {
        Assert.assertEquals(actual.getShape(), expected.getShape());
        float tolerance =
                dataType == DataType.FLOAT32 ? 0.001f : dataType == DataType.FLOAT16 ? 0.02f : 0.1f;
        float[] actualValues = actual.toType(DataType.FLOAT32, false).toFloatArray();
        float[] expectedValues = expected.toType(DataType.FLOAT32, false).toFloatArray();
        for (int index = 0; index < actualValues.length; ++index) {
            Assert.assertTrue(Float.isFinite(actualValues[index]), "Non-finite output at " + index);
            if (present[index / hiddenWidth]) {
                Assert.assertEquals(
                        actualValues[index],
                        expectedValues[index],
                        tolerance,
                        "Output at " + index);
            } else {
                Assert.assertEquals(actualValues[index], 0f, "Inactive output at " + index);
            }
        }
    }
}
