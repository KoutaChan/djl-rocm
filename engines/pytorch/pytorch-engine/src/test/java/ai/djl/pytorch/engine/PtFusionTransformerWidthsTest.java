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
import ai.djl.engine.fusion.FusionInvocation;
import ai.djl.engine.fusion.FusionOutputLease;
import ai.djl.engine.fusion.FusionPlan;
import ai.djl.engine.fusion.FusionRecipe;
import ai.djl.engine.fusion.FusionSession;
import ai.djl.engine.fusion.FusionSessionConfig;
import ai.djl.engine.fusion.IndexedRelationAttention;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

/** Tests transformer widths, attention heads, and scratch aliases against eager operations. */
public class PtFusionTransformerWidthsTest {

    @Test
    public void descriptorWorkspaceIncludesExpandedAliasCapacity() {
        // Hidden, heads, attention, tokens, expansion, indexed relation, workspace bytes.
        int[][] configurations = {
            {256, 4, 128, 6, 512, 0, 82_944},
            {33, 3, 105, 17, 67, 0, 99_348},
            {513, 3, 96, 3, 19, 0, 37_620},
            {33, 3, 105, 11, 67, 1, 70_620}
        };
        for (int[] configuration : configurations) {
            int hidden = configuration[0];
            int heads = configuration[1];
            int attention = configuration[2];
            int tokens = configuration[3];
            int expansion = configuration[4];
            FusionRecipe.Builder builder = FusionRecipe.builder("transformer-workspace");
            FusionRecipe.Dimension batch = builder.addDimension("batch", 3);
            FusionRecipe.Input input =
                    builder.addInput(
                            "input",
                            FusionRecipe.TensorSpec.of(DataType.FLOAT32, batch, tokens, hidden));
            Shape[] shapes = {
                new Shape(hidden), new Shape(hidden), new Shape(3L * attention, hidden),
                new Shape(hidden, attention), new Shape(hidden), new Shape(hidden),
                new Shape(hidden), new Shape(expansion, hidden), new Shape(expansion),
                new Shape(hidden, expansion), new Shape(hidden), new Shape(hidden),
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
            IndexedRelationAttention relation = null;
            if (configuration[5] != 0) {
                relation =
                        builder.indexedRelationAttention(
                                builder.addConstant(
                                        "relationIds",
                                        FusionRecipe.TensorSpec.fixed(
                                                DataType.INT16, tokens, tokens)),
                                builder.addConstant(
                                        "relationKeys",
                                        FusionRecipe.TensorSpec.fixed(
                                                DataType.FLOAT32, 5, attention)),
                                builder.addConstant(
                                        "relationBias",
                                        FusionRecipe.TensorSpec.fixed(DataType.FLOAT32, 5, heads)));
            }
            FusionRecipe.TransformerEncoderStack stack =
                    builder.transformerEncoderStack("stack", input, heads, attention, expansion)
                            .addBlock(
                                    parameters[0],
                                    parameters[1],
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
                                    relation)
                            .build();
            builder.addOutput("output", stack);
            FusionRecipe recipe = builder.build();
            long expectedWorkspace = configuration[6];
            Assert.assertEquals(PtFusionDescriptor.workspaceBytes(recipe), expectedWorkspace);
            Assert.assertEquals(
                    PtFusionDescriptor.persistentStorageBytes(recipe),
                    expectedWorkspace + 3L * tokens * hidden * Float.BYTES);
        }
    }

    @Test
    public void gpuTransformerWidthsMatchEagerReference() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
        }
        // Hidden width, attention heads, attention width, tokens, feed-forward width.
        int[][] configurations = {
            {256, 4, 128, 6, 512},
            {384, 4, 128, 6, 256},
            {416, 3, 96, 9, 193},
            {33, 3, 105, 17, 67},
            {513, 3, 96, 3, 19},
            {7, 7, 7, 2, 11},
            {32, 1, 35, 33, 65}
        };
        Device device = Device.gpu(0);
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            for (int[] configuration : configurations) {
                try (NDManager manager = engine.newBaseManager(device)) {
                    verifyConfiguration(engine, manager, dataType, configuration, false);
                }
            }
        }
    }

    @Test
    public void gpuIndexedRelationTransformerWidthsMatchEagerReference() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
        }
        int[][] configurations = {
            {384, 4, 64, 17, 256}, {416, 3, 96, 9, 193}, {33, 3, 105, 11, 67}
        };
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            for (int[] configuration : configurations) {
                try (NDManager manager = engine.newBaseManager(Device.gpu(0))) {
                    verifyConfiguration(engine, manager, dataType, configuration, true);
                }
            }
        }
    }

    private static void verifyConfiguration(
            PtEngine engine,
            NDManager manager,
            DataType dataType,
            int[] configuration,
            boolean indexedRelation) {
        int hidden = configuration[0];
        int heads = configuration[1];
        int attention = configuration[2];
        int tokens = configuration[3];
        int expanded = configuration[4];
        FusionRecipe.Builder builder = FusionRecipe.builder("transformer-widths");
        FusionRecipe.Dimension batch = builder.addDimension("batch", 3);
        FusionRecipe.Input input =
                builder.addInput(
                        "input", FusionRecipe.TensorSpec.of(dataType, batch, tokens, hidden));
        Shape[] shapes = {
            new Shape(hidden),
            new Shape(hidden),
            new Shape(3L * attention, hidden),
            new Shape(hidden, attention),
            new Shape(hidden),
            new Shape(hidden),
            new Shape(hidden),
            new Shape(expanded, hidden),
            new Shape(expanded),
            new Shape(hidden, expanded),
            new Shape(hidden),
            new Shape(hidden),
            new Shape(hidden)
        };
        FusionRecipe.Constant[] constants = new FusionRecipe.Constant[shapes.length];
        NDArray[] parameters = new NDArray[shapes.length];
        for (int index = 0; index < shapes.length; ++index) {
            boolean norm = index == 0 || index == 1 || index == 5 || index == 6 || index >= 11;
            DataType parameterType = norm ? DataType.FLOAT32 : dataType;
            constants[index] =
                    builder.addConstant(
                            "parameter" + index,
                            FusionRecipe.TensorSpec.fixed(parameterType, shapes[index].getShape()));
            parameters[index] = patterned(manager, parameterType, shapes[index], index, 0.003f);
            if (index == 0 || index == 5 || index == 11) {
                parameters[index] = parameters[index].add(1f);
            }
        }
        FusionRecipe.TransformerEncoderStackBuilder stack =
                builder.transformerEncoderStack("stack", input, heads, attention, expanded);
        RelationFixture relation =
                indexedRelation
                        ? new RelationFixture(builder, manager, dataType, tokens, heads, attention)
                        : null;
        for (int block = 0; block < 2; ++block) {
            stack.addBlock(
                    constants[0],
                    constants[1],
                    constants[2],
                    constants[3],
                    constants[4],
                    constants[5],
                    constants[6],
                    constants[7],
                    constants[8],
                    constants[9],
                    constants[10],
                    constants[indexedRelation ? 0 : 11],
                    constants[indexedRelation ? 1 : 12],
                    relation == null ? null : relation.relation);
        }
        FusionRecipe.Output output = builder.addOutput("output", stack.build());
        FusionRecipe recipe = builder.build();
        FusionConstantBindings.Builder bindings = FusionConstantBindings.builder(recipe);
        for (int index = 0; index < constants.length; ++index) {
            bindings.bind(constants[index], parameters[index]);
        }
        if (relation != null) {
            relation.bind(bindings);
        }
        try (FusionPlan plan = engine.newFusionCompiler(manager.getDevice()).prepare(recipe);
                FusionExecutable executable = plan.bind(bindings.build());
                FusionSession session =
                        executable.newSession(manager, FusionSessionConfig.defaults())) {
            for (int batchCount : new int[] {3, 1, 2}) {
                try (NDManager working = manager.newSubManager();
                        FusionInvocation invocation = session.acquire()) {
                    NDArray inputArray =
                            patterned(
                                    working,
                                    dataType,
                                    new Shape(batchCount, tokens, hidden),
                                    batchCount,
                                    0.025f);
                    float[] original = inputArray.toType(DataType.FLOAT32, false).toFloatArray();
                    NDArray expected;
                    if (relation != null) {
                        expected =
                                relationReference(
                                        inputArray, parameters, relation, heads, attention);
                    } else {
                        expected = inputArray;
                        for (int block = 0; block < 2; ++block) {
                            expected = reference(expected, parameters, heads, attention);
                        }
                    }
                    invocation.setInput(input, inputArray);
                    invocation.setDimension(batch, batchCount);
                    try (FusionOutputLease lease = invocation.submit()) {
                        lease.synchronize();
                        NDArray actual = lease.get(output).get("0:" + batchCount);
                        float tolerance =
                                dataType == DataType.FLOAT32
                                        ? 4e-4f
                                        : dataType == DataType.FLOAT16 ? 4e-2f : 0.12f;
                        Assert.assertEquals(actual.getShape(), inputArray.getShape());
                        Assert.assertEquals(
                                actual.toType(DataType.FLOAT32, false).toFloatArray(),
                                expected.toType(DataType.FLOAT32, false).toFloatArray(),
                                tolerance,
                                "Transformer parity failed for hidden="
                                        + hidden
                                        + ", attention="
                                        + attention
                                        + ", tokens="
                                        + tokens
                                        + ", dtype="
                                        + dataType
                                        + ", indexedRelation="
                                        + indexedRelation);
                    }
                    Assert.assertEquals(
                            inputArray.toType(DataType.FLOAT32, false).toFloatArray(), original);
                }
            }
        }
    }

    private static NDArray relationReference(
            NDArray input,
            NDArray[] parameters,
            RelationFixture relation,
            int heads,
            int attention) {
        long batch = input.getShape().get(0);
        long tokens = input.getShape().get(1);
        long hidden = input.getShape().get(2);
        long rows = batch * tokens;
        long headWidth = attention / heads;
        double scale = 1.0 / Math.sqrt(headWidth);
        DataType dataType = input.getDataType();
        NDArray residual = input.duplicate();
        NDArray normalized = layerNorm(residual, parameters[0], parameters[1]);
        NDArray relationKeys =
                relation.keys
                        .reshape(RelationFixture.RELATIONS, heads, headWidth)
                        .transpose(1, 2, 0);
        NDArray indices =
                relation.ids
                        .toType(DataType.INT64, false)
                        .reshape(1, 1, tokens, tokens)
                        .broadcast(new Shape(batch, heads, tokens, tokens));
        for (int block = 0; block < 2; ++block) {
            NDArray qkv =
                    normalized
                            .reshape(rows, hidden)
                            .matMul(parameters[2].transpose())
                            .reshape(batch, tokens, 3L * attention);
            NDArray queries =
                    qkv.get("...,0:" + attention)
                            .reshape(batch, tokens, heads, headWidth)
                            .swapAxes(1, 2);
            NDArray keys =
                    qkv.get("...," + attention + ":" + 2 * attention)
                            .reshape(batch, tokens, heads, headWidth)
                            .swapAxes(1, 2);
            NDArray values =
                    qkv.get("...," + 2 * attention + ":" + 3 * attention)
                            .reshape(batch, tokens, heads, headWidth)
                            .swapAxes(1, 2);
            NDArray attentionBias =
                    queries.matMul(relationKeys.expandDims(0))
                            .toType(DataType.FLOAT32, false)
                            .gather(indices, 3)
                            .mul(scale)
                            .add(relation.pairBias.toType(DataType.FLOAT32, false))
                            .toType(dataType, false)
                            .toType(DataType.FLOAT32, false);
            NDArray context =
                    queries.toType(DataType.FLOAT32, false)
                            .matMul(keys.toType(DataType.FLOAT32, false).swapAxes(2, 3))
                            .mul(scale)
                            .add(attentionBias)
                            .softmax(-1)
                            .matMul(values.toType(DataType.FLOAT32, false))
                            .swapAxes(1, 2)
                            .reshape(rows, attention)
                            .toType(dataType, false);
            NDArray update =
                    context.matMul(parameters[3].transpose())
                            .add(parameters[4])
                            .reshape(input.getShape());
            NDArray feedForward =
                    NDArrays.addToOwnedResidualAndLayerNorm(
                            residual, update, parameters[5], parameters[6], 1e-5f);
            NDArray expanded =
                    feedForward
                            .reshape(rows, hidden)
                            .matMul(parameters[7].transpose())
                            .add(parameters[8]);
            NDArray projected =
                    Activation.swish(expanded, 1.0f)
                            .matMul(parameters[9].transpose())
                            .add(parameters[10])
                            .reshape(input.getShape());
            normalized =
                    NDArrays.addToOwnedResidualAndLayerNorm(
                            residual, projected, parameters[0], parameters[1], 1e-5f);
        }
        return normalized;
    }

    private static NDArray reference(
            NDArray input, NDArray[] parameters, int heads, int attention) {
        long batch = input.getShape().get(0);
        long tokens = input.getShape().get(1);
        long hidden = input.getShape().get(2);
        long rows = batch * tokens;
        long headWidth = attention / heads;
        DataType dataType = input.getDataType();
        NDArray normalized = layerNorm(input, parameters[0], parameters[1]);
        NDArray queryKeyValue =
                normalized
                        .reshape(rows, hidden)
                        .matMul(parameters[2].transpose())
                        .reshape(batch, tokens, 3L * attention)
                        .toType(DataType.FLOAT32, false);
        NDArray queries =
                queryKeyValue
                        .get("...,0:" + attention)
                        .reshape(batch, tokens, heads, headWidth)
                        .swapAxes(1, 2);
        NDArray keys =
                queryKeyValue
                        .get("...," + attention + ":" + 2 * attention)
                        .reshape(batch, tokens, heads, headWidth)
                        .swapAxes(1, 2);
        NDArray values =
                queryKeyValue
                        .get("...," + 2 * attention + ":" + 3 * attention)
                        .reshape(batch, tokens, heads, headWidth)
                        .swapAxes(1, 2);
        NDArray context =
                queries.matMul(keys.swapAxes(2, 3))
                        .mul(1.0 / Math.sqrt(headWidth))
                        .softmax(-1)
                        .matMul(values)
                        .swapAxes(1, 2)
                        .reshape(rows, attention)
                        .toType(dataType, false);
        NDArray update =
                context.matMul(parameters[3].transpose())
                        .toType(DataType.FLOAT32, false)
                        .add(parameters[4].toType(DataType.FLOAT32, false))
                        .reshape(input.getShape());
        NDArray residual =
                input.toType(DataType.FLOAT32, false).add(update).toType(dataType, false);
        NDArray feedForward = layerNorm(residual, parameters[5], parameters[6]);
        NDArray expansion =
                feedForward
                        .reshape(rows, hidden)
                        .matMul(parameters[7].transpose())
                        .toType(DataType.FLOAT32, false)
                        .add(parameters[8].toType(DataType.FLOAT32, false));
        NDArray activated = expansion.mul(Activation.sigmoid(expansion)).toType(dataType, false);
        NDArray projected =
                activated
                        .matMul(parameters[9].transpose())
                        .toType(DataType.FLOAT32, false)
                        .add(parameters[10].toType(DataType.FLOAT32, false))
                        .reshape(input.getShape());
        NDArray result =
                residual.toType(DataType.FLOAT32, false).add(projected).toType(dataType, false);
        return layerNorm(result, parameters[11], parameters[12]);
    }

    private static NDArray layerNorm(NDArray input, NDArray weight, NDArray bias) {
        NDArray floatInput = input.toType(DataType.FLOAT32, false);
        return floatInput
                .getNDArrayInternal()
                .layerNorm(floatInput, new Shape(input.getShape().get(2)), weight, bias, 1e-5f)
                .get(0)
                .toType(input.getDataType(), false);
    }

    private static NDArray patterned(
            NDManager manager, DataType dataType, Shape shape, int seed, float scale) {
        float[] values = new float[Math.toIntExact(shape.size())];
        for (int index = 0; index < values.length; ++index) {
            values[index] = ((index * 17 + seed * 13) % 47 - 23) * scale;
        }
        return manager.create(values, shape).toType(dataType, false);
    }

    private static final class RelationFixture {

        private static final int RELATIONS = 5;

        private final FusionRecipe.Constant idsConstant;
        private final FusionRecipe.Constant keysConstant;
        private final FusionRecipe.Constant biasConstant;
        private final IndexedRelationAttention relation;
        private final NDArray ids;
        private final NDArray keys;
        private final NDArray bias;
        private final NDArray pairBias;

        private RelationFixture(
                FusionRecipe.Builder builder,
                NDManager manager,
                DataType dataType,
                int tokens,
                int heads,
                int attention) {
            idsConstant =
                    builder.addConstant(
                            "relationIds",
                            FusionRecipe.TensorSpec.fixed(DataType.INT16, tokens, tokens));
            keysConstant =
                    builder.addConstant(
                            "relationKeys",
                            FusionRecipe.TensorSpec.fixed(dataType, RELATIONS, attention));
            biasConstant =
                    builder.addConstant(
                            "relationBias",
                            FusionRecipe.TensorSpec.fixed(dataType, RELATIONS, heads));
            relation = builder.indexedRelationAttention(idsConstant, keysConstant, biasConstant);
            int[] idsValues = new int[tokens * tokens];
            for (int query = 0; query < tokens; ++query) {
                for (int key = 0; key < tokens; ++key) {
                    idsValues[query * tokens + key] = (query * 3 + key * 7) % RELATIONS;
                }
            }
            ids =
                    manager.create(idsValues, new Shape(tokens, tokens))
                            .toType(DataType.INT16, false);
            keys = patterned(manager, dataType, new Shape(RELATIONS, attention), 21, 0.004f);
            bias = patterned(manager, dataType, new Shape(RELATIONS, heads), 22, 0.006f);
            float[] biasValues = bias.toType(DataType.FLOAT32, false).toFloatArray();
            float[] pairValues = new float[heads * tokens * tokens];
            for (int head = 0; head < heads; ++head) {
                for (int pair = 0; pair < idsValues.length; ++pair) {
                    pairValues[head * idsValues.length + pair] =
                            biasValues[idsValues[pair] * heads + head];
                }
            }
            pairBias =
                    manager.create(pairValues, new Shape(1, heads, tokens, tokens))
                            .toType(dataType, false);
        }

        private void bind(FusionConstantBindings.Builder bindings) {
            bindings.bind(idsConstant, ids).bind(keysConstant, keys).bind(biasConstant, bias);
        }
    }
}
