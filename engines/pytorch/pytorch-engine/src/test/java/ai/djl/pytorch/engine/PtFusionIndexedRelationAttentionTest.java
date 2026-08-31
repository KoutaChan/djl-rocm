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
import ai.djl.ndarray.NDScope;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

import java.nio.ByteBuffer;

/** Tests indexed-relation transformer descriptor encoding and ROCm eager parity. */
public class PtFusionIndexedRelationAttentionTest {

    private static final int TOKENS = 34;
    private static final int HIDDEN_WIDTH = 256;
    private static final int HEADS = 4;
    private static final int ATTENTION_WIDTH = 64;
    private static final int RELATIONS = 23;
    private static final int FEED_FORWARD_WIDTH = 128;

    @Test
    public void indexedRelationTransformerDescriptorReportsPreparedStorage() {
        Fixture fixture = new Fixture(DataType.FLOAT16, 2, 1);
        ByteBuffer descriptor = PtFusionDescriptor.encode(fixture.recipe);
        int commandOffset = Math.toIntExact(descriptor.getLong(14 * Long.BYTES));

        Assert.assertEquals(
                descriptor.getLong((commandOffset + 1) * Long.BYTES),
                PtFusionDescriptor.INDEXED_RELATION_TRANSFORMER_ENCODER_STACK_V1);
        Assert.assertEquals(descriptor.getLong((commandOffset + 4) * Long.BYTES), 17L);
        Assert.assertEquals(descriptor.getLong((commandOffset + 5) * Long.BYTES), 7L);
        int firstAttribute = commandOffset + 7 + 17;
        int relationCountAttribute = firstAttribute + 5 * PtFusionDescriptor.SCALAR_ATTRIBUTE_WORDS;
        Assert.assertEquals(
                descriptor.getLong((relationCountAttribute + 1) * Long.BYTES),
                PtFusionDescriptor.TRANSFORMER_RELATION_COUNT);
        Assert.assertEquals(descriptor.getLong((relationCountAttribute + 4) * Long.BYTES), 23L);
        int reuseAttribute = firstAttribute + 6 * PtFusionDescriptor.SCALAR_ATTRIBUTE_WORDS;
        Assert.assertEquals(
                descriptor.getLong((reuseAttribute + 1) * Long.BYTES),
                PtFusionDescriptor.TRANSFORMER_REUSE_OUTPUT_NORMALIZATION);
        Assert.assertEquals(descriptor.getLong((reuseAttribute + 4) * Long.BYTES), 0L);
        Assert.assertEquals(PtFusionDescriptor.commandCount(fixture.recipe), 1);
        Assert.assertEquals(PtFusionDescriptor.executableStorageBytes(fixture.recipe), 276_648L);
        Assert.assertEquals(PtFusionDescriptor.persistentStorageBytes(fixture.recipe), 134_912L);
        Assert.assertEquals(PtFusionDescriptor.workspaceBytes(fixture.recipe), 100_096L);
    }

    @Test
    public void adjacentBlocksReuseIdenticalOutputNormalizationHandles() {
        Fixture fixture = new Fixture(DataType.FLOAT16, 2, 2);
        ByteBuffer descriptor = PtFusionDescriptor.encode(fixture.recipe);
        int commandOffset = Math.toIntExact(descriptor.getLong(14 * Long.BYTES));
        int operandCount = Math.toIntExact(descriptor.getLong((commandOffset + 4) * Long.BYTES));
        int firstAttribute = commandOffset + 7 + operandCount;
        int reuseAttribute = firstAttribute + 6 * PtFusionDescriptor.SCALAR_ATTRIBUTE_WORDS;

        Assert.assertEquals(
                descriptor.getLong((reuseAttribute + 1) * Long.BYTES),
                PtFusionDescriptor.TRANSFORMER_REUSE_OUTPUT_NORMALIZATION);
        Assert.assertEquals(descriptor.getLong((reuseAttribute + 4) * Long.BYTES), 1L);
    }

    @Test
    public void productionCapacityMatchesPublishedStorageManifest() {
        Fixture fixture = new Fixture(DataType.FLOAT16, 4096, 2);

        Assert.assertEquals(PtFusionDescriptor.executableStorageBytes(fixture.recipe), 550_984L);
        Assert.assertEquals(
                PtFusionDescriptor.persistentStorageBytes(fixture.recipe), 276_299_776L);
        Assert.assertEquals(PtFusionDescriptor.workspaceBytes(fixture.recipe), 204_996_608L);
    }

    @Test
    public void rocmIndexedRelationTransformerMatchesEagerRoundingAndPreservesInputs() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch ROCm device.");
        }
        int[] batchCounts = {1, 31, 256, 384};
        DataType[] dataTypes = {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16};
        for (int deviceIndex = 0; deviceIndex < engine.getGpuCount(); ++deviceIndex) {
            Device device = Device.gpu(deviceIndex);
            for (DataType dataType : dataTypes) {
                Fixture fixture = new Fixture(dataType, 384, 2);
                try (NDManager manager = engine.newBaseManager(device);
                        Parameters parameters = new Parameters(manager, dataType);
                        FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                        FusionExecutable executable = plan.bind(fixture.bindings(parameters));
                        FusionSession session =
                                executable.newSession(
                                        manager,
                                        FusionSessionConfig.builder().optBufferCount(1).build())) {
                    for (int iteration = 0; iteration < 2; ++iteration) {
                        try (NDManager workingManager = manager.newSubManager()) {
                            for (int batchCount : batchCounts) {
                                verifyInvocation(
                                        session,
                                        fixture,
                                        parameters,
                                        workingManager,
                                        dataType,
                                        deviceIndex,
                                        iteration,
                                        batchCount);
                            }
                        }
                        parameters.assertSourceConstantsUnchanged();
                    }
                }
            }
        }
    }

    private static void verifyInvocation(
            FusionSession session,
            Fixture fixture,
            Parameters parameters,
            NDManager workingManager,
            DataType dataType,
            int deviceIndex,
            int iteration,
            int batchCount) {
        try (NDArray input =
                        patternedArray(
                                workingManager,
                                dataType,
                                new Shape(batchCount, TOKENS, HIDDEN_WIDTH),
                                47,
                                23,
                                0.015f);
                NDArray expected = reference(input, parameters, 2);
                FusionInvocation invocation = session.acquire()) {
            invocation.setInput(fixture.input, input);
            invocation.setDimension(fixture.batch, batchCount);
            try (FusionOutputLease lease = invocation.submit()) {
                lease.synchronize();
                assertClose(
                        lease.get(fixture.output).get("0:" + batchCount),
                        expected,
                        dataType,
                        deviceIndex,
                        iteration,
                        batchCount);
            }
        }
    }

    private static NDArray reference(NDArray input, Parameters parameters, int blockCount) {
        try (NDScope scope = new NDScope()) {
            scope.suppressNotUsedWarning();
            int batchCount = Math.toIntExact(input.getShape().get(0));
            NDArray residual = input.duplicate();
            NDArray normalized =
                    mixedPrecisionLayerNormReference(
                            residual,
                            parameters.attentionInputWeight,
                            parameters.attentionInputBias);
            NDArray relationKeys =
                    parameters
                            .relationKeys
                            .reshape(RELATIONS, HEADS, ATTENTION_WIDTH / HEADS)
                            .transpose(1, 2, 0)
                            .expandDims(0);
            for (int block = 0; block < blockCount; ++block) {
                NDArray qkv =
                        normalized
                                .reshape(batchCount * TOKENS, HIDDEN_WIDTH)
                                .matMul(parameters.queryKeyValue.transpose())
                                .reshape(batchCount, TOKENS, 3L * ATTENTION_WIDTH);
                NDArray queries =
                        qkv.get("...,0:64")
                                .reshape(batchCount, TOKENS, HEADS, ATTENTION_WIDTH / HEADS)
                                .swapAxes(1, 2);
                NDArray keys =
                        qkv.get("...,64:128")
                                .reshape(batchCount, TOKENS, HEADS, ATTENTION_WIDTH / HEADS)
                                .swapAxes(1, 2);
                NDArray values =
                        qkv.get("...,128:192")
                                .reshape(batchCount, TOKENS, HEADS, ATTENTION_WIDTH / HEADS)
                                .swapAxes(1, 2);
                NDArray context =
                        NDArrays.relationBiasedScaledDotProductAttention(
                                        queries,
                                        keys,
                                        values,
                                        relationKeys,
                                        parameters.pairBias,
                                        parameters.eagerRelationIds,
                                        0.25)
                                .swapAxes(1, 2)
                                .reshape(batchCount * TOKENS, ATTENTION_WIDTH);
                NDArray attentionUpdate =
                        context.matMul(parameters.attentionOutput.transpose())
                                .add(parameters.attentionOutputBias)
                                .reshape(batchCount, TOKENS, HIDDEN_WIDTH);
                NDArray feedForwardInput =
                        NDArrays.addToOwnedResidualAndLayerNorm(
                                residual,
                                attentionUpdate,
                                parameters.feedForwardInputWeight,
                                parameters.feedForwardInputBias,
                                1.0e-5f);
                NDArray expanded =
                        feedForwardInput
                                .reshape(batchCount * TOKENS, HIDDEN_WIDTH)
                                .matMul(parameters.expansion.transpose())
                                .add(parameters.expansionBias);
                NDArray feedForwardUpdate =
                        Activation.swish(expanded, 1.0f)
                                .matMul(parameters.projection.transpose())
                                .add(parameters.projectionBias)
                                .reshape(batchCount, TOKENS, HIDDEN_WIDTH);
                normalized =
                        NDArrays.addToOwnedResidualAndLayerNorm(
                                residual,
                                feedForwardUpdate,
                                parameters.attentionInputWeight,
                                parameters.attentionInputBias,
                                1.0e-5f);
            }
            NDScope.unregister(normalized);
            return normalized;
        }
    }

    private static NDArray mixedPrecisionLayerNormReference(
            NDArray input, NDArray weight, NDArray bias) {
        DataType outputType = input.getDataType();
        NDArray floatInput = input.toType(DataType.FLOAT32, false);
        NDArray floatWeight = weight.toType(DataType.FLOAT32, false);
        NDArray floatBias = bias.toType(DataType.FLOAT32, false);
        NDArray normalized =
                floatInput
                        .getNDArrayInternal()
                        .layerNorm(
                                floatInput, new Shape(HIDDEN_WIDTH), floatWeight, floatBias, 1e-5f)
                        .get(0);
        return outputType == DataType.FLOAT32 ? normalized : normalized.toType(outputType, false);
    }

    private static void assertClose(
            NDArray actual,
            NDArray expected,
            DataType dataType,
            int deviceIndex,
            int iteration,
            int batchCount) {
        float tolerance =
                dataType == DataType.FLOAT32
                        ? 4.0e-4f
                        : dataType == DataType.FLOAT16 ? 4.0e-2f : 8.0e-2f;
        float[] actualValues = actual.toType(DataType.FLOAT32, false).toFloatArray();
        float[] expectedValues = expected.toType(DataType.FLOAT32, false).toFloatArray();
        Assert.assertEquals(actualValues.length, expectedValues.length);
        float maximum = 0.0f;
        double total = 0.0;
        for (int index = 0; index < actualValues.length; ++index) {
            Assert.assertTrue(
                    Float.isFinite(actualValues[index]) && Float.isFinite(expectedValues[index]),
                    "Non-finite indexed-relation value at index " + index);
            float difference = Math.abs(actualValues[index] - expectedValues[index]);
            maximum = Math.max(maximum, difference);
            total += difference;
        }
        double mean = total / actualValues.length;
        System.out.printf(
                "indexed-relation-parity device=%d dtype=%s iteration=%d batch=%d"
                        + " maxAbs=%.9g meanAbs=%.9g tolerance=%.9g%n",
                deviceIndex, dataType, iteration, batchCount, maximum, mean, tolerance);
        Assert.assertTrue(
                maximum <= tolerance,
                "Indexed-relation parity exceeded its fixed tolerance: maxAbs="
                        + maximum
                        + ", meanAbs="
                        + mean
                        + ", tolerance="
                        + tolerance);
    }

    private static NDArray patternedArray(
            NDManager manager,
            DataType dataType,
            Shape shape,
            int period,
            int center,
            float scale) {
        try (NDScope scope = new NDScope()) {
            scope.suppressNotUsedWarning();
            NDArray result =
                    manager.arange((float) shape.size())
                            .mod(period)
                            .sub(center)
                            .mul(scale)
                            .reshape(shape)
                            .toType(dataType, false);
            NDScope.unregister(result);
            return result;
        }
    }

    private static float[] patternedValues(int count, int period, int center, float scale) {
        float[] values = new float[count];
        for (int index = 0; index < count; ++index) {
            values[index] = (index % period - center) * scale;
        }
        return values;
    }

    private static FusionRecipe.Constant vector(
            FusionRecipe.Builder builder, String name, int width, DataType dataType) {
        return builder.addConstant(name, FusionRecipe.TensorSpec.fixed(dataType, width));
    }

    private static FusionRecipe.Constant matrix(
            FusionRecipe.Builder builder, String name, int rows, int columns, DataType dataType) {
        return builder.addConstant(name, FusionRecipe.TensorSpec.fixed(dataType, rows, columns));
    }

    private static final class Fixture {

        private final FusionRecipe.Dimension batch;
        private final FusionRecipe.Input input;
        private final FusionRecipe.Constant relationIds;
        private final FusionRecipe.Constant relationKeys;
        private final FusionRecipe.Constant relationBias;
        private final FusionRecipe.Constant attentionInputWeight;
        private final FusionRecipe.Constant attentionInputBias;
        private final FusionRecipe.Constant queryKeyValue;
        private final FusionRecipe.Constant attentionOutput;
        private final FusionRecipe.Constant attentionOutputBias;
        private final FusionRecipe.Constant feedForwardInputWeight;
        private final FusionRecipe.Constant feedForwardInputBias;
        private final FusionRecipe.Constant expansion;
        private final FusionRecipe.Constant expansionBias;
        private final FusionRecipe.Constant projection;
        private final FusionRecipe.Constant projectionBias;
        private final FusionRecipe.Output output;
        private final FusionRecipe recipe;

        private Fixture(DataType dataType, int maxBatch, int blockCount) {
            FusionRecipe.Builder builder = FusionRecipe.builder("indexed-relation-transformer");
            batch = builder.addDimension("batch", maxBatch);
            input =
                    builder.addInput(
                            "input",
                            FusionRecipe.TensorSpec.of(dataType, batch, TOKENS, HIDDEN_WIDTH));
            relationIds =
                    builder.addConstant(
                            "relationIds",
                            FusionRecipe.TensorSpec.fixed(DataType.INT16, TOKENS, TOKENS));
            relationKeys = matrix(builder, "relationKeys", RELATIONS, ATTENTION_WIDTH, dataType);
            relationBias = matrix(builder, "relationBias", RELATIONS, HEADS, dataType);
            attentionInputWeight =
                    vector(builder, "attentionInputWeight", HIDDEN_WIDTH, DataType.FLOAT32);
            attentionInputBias =
                    vector(builder, "attentionInputBias", HIDDEN_WIDTH, DataType.FLOAT32);
            queryKeyValue = matrix(builder, "qkv", 3 * ATTENTION_WIDTH, HIDDEN_WIDTH, dataType);
            attentionOutput =
                    matrix(builder, "attentionOutput", HIDDEN_WIDTH, ATTENTION_WIDTH, dataType);
            attentionOutputBias = vector(builder, "attentionOutputBias", HIDDEN_WIDTH, dataType);
            feedForwardInputWeight =
                    vector(builder, "feedForwardInputWeight", HIDDEN_WIDTH, DataType.FLOAT32);
            feedForwardInputBias =
                    vector(builder, "feedForwardInputBias", HIDDEN_WIDTH, DataType.FLOAT32);
            expansion = matrix(builder, "expansion", FEED_FORWARD_WIDTH, HIDDEN_WIDTH, dataType);
            expansionBias = vector(builder, "expansionBias", FEED_FORWARD_WIDTH, dataType);
            projection = matrix(builder, "projection", HIDDEN_WIDTH, FEED_FORWARD_WIDTH, dataType);
            projectionBias = vector(builder, "projectionBias", HIDDEN_WIDTH, dataType);
            IndexedRelationAttention relation =
                    builder.indexedRelationAttention(relationIds, relationKeys, relationBias);
            FusionRecipe.TransformerEncoderStackBuilder stack =
                    builder.transformerEncoderStack(
                            "stack", input, HEADS, ATTENTION_WIDTH, FEED_FORWARD_WIDTH);
            for (int block = 0; block < blockCount; ++block) {
                stack.addBlock(
                        attentionInputWeight,
                        attentionInputBias,
                        queryKeyValue,
                        attentionOutput,
                        attentionOutputBias,
                        feedForwardInputWeight,
                        feedForwardInputBias,
                        expansion,
                        expansionBias,
                        projection,
                        projectionBias,
                        attentionInputWeight,
                        attentionInputBias,
                        relation);
            }
            output = builder.addOutput("output", stack.build());
            recipe = builder.build();
        }

        private FusionConstantBindings bindings(Parameters parameters) {
            return FusionConstantBindings.builder(recipe)
                    .bind(relationIds, parameters.relationIds)
                    .bind(relationKeys, parameters.relationKeys)
                    .bind(relationBias, parameters.relationBias)
                    .bind(attentionInputWeight, parameters.attentionInputWeight)
                    .bind(attentionInputBias, parameters.attentionInputBias)
                    .bind(queryKeyValue, parameters.queryKeyValue)
                    .bind(attentionOutput, parameters.attentionOutput)
                    .bind(attentionOutputBias, parameters.attentionOutputBias)
                    .bind(feedForwardInputWeight, parameters.feedForwardInputWeight)
                    .bind(feedForwardInputBias, parameters.feedForwardInputBias)
                    .bind(expansion, parameters.expansion)
                    .bind(expansionBias, parameters.expansionBias)
                    .bind(projection, parameters.projection)
                    .bind(projectionBias, parameters.projectionBias)
                    .build();
        }
    }

    private static final class Parameters implements AutoCloseable {

        private final NDManager manager;
        private final NDArray relationIds;
        private final NDArray eagerRelationIds;
        private final NDArray relationKeys;
        private final NDArray relationBias;
        private final NDArray pairBias;
        private final NDArray attentionInputWeight;
        private final NDArray attentionInputBias;
        private final NDArray queryKeyValue;
        private final NDArray attentionOutput;
        private final NDArray attentionOutputBias;
        private final NDArray feedForwardInputWeight;
        private final NDArray feedForwardInputBias;
        private final NDArray expansion;
        private final NDArray expansionBias;
        private final NDArray projection;
        private final NDArray projectionBias;

        private Parameters(NDManager parent, DataType dataType) {
            manager = parent.newSubManager();
            int[] ids = new int[TOKENS * TOKENS];
            long[] eagerIds = new long[ids.length];
            for (int query = 0; query < TOKENS; ++query) {
                for (int key = 0; key < TOKENS; ++key) {
                    int index = query * TOKENS + key;
                    ids[index] = (query * 7 + key * 3) % RELATIONS;
                    eagerIds[index] = ids[index];
                }
            }
            float[] relationKeyValues =
                    patternedValues(RELATIONS * ATTENTION_WIDTH, 29, 14, 0.003f);
            float[] relationBiasValues = patternedValues(RELATIONS * HEADS, 13, 6, 0.004f);
            float[] pairBiasValues = new float[HEADS * TOKENS * TOKENS];
            for (int head = 0; head < HEADS; ++head) {
                for (int query = 0; query < TOKENS; ++query) {
                    for (int key = 0; key < TOKENS; ++key) {
                        int relation = ids[query * TOKENS + key];
                        pairBiasValues[(head * TOKENS + query) * TOKENS + key] =
                                relationBiasValues[relation * HEADS + head];
                    }
                }
            }
            relationIds =
                    manager.create(ids, new Shape(TOKENS, TOKENS)).toType(DataType.INT16, false);
            eagerRelationIds = manager.create(eagerIds, new Shape(TOKENS, TOKENS));
            relationKeys =
                    manager.create(relationKeyValues, new Shape(RELATIONS, ATTENTION_WIDTH))
                            .toType(dataType, false);
            relationBias =
                    manager.create(relationBiasValues, new Shape(RELATIONS, HEADS))
                            .toType(dataType, false);
            pairBias =
                    manager.create(pairBiasValues, new Shape(1, HEADS, TOKENS, TOKENS))
                            .toType(dataType, false);
            attentionInputWeight =
                    manager.create(patternedValues(HIDDEN_WIDTH, 19, 9, 0.01f)).add(1f);
            attentionInputBias = manager.create(patternedValues(HIDDEN_WIDTH, 17, 8, 0.002f));
            queryKeyValue =
                    manager.create(
                                    patternedValues(
                                            3 * ATTENTION_WIDTH * HIDDEN_WIDTH, 31, 15, 0.002f),
                                    new Shape(3 * ATTENTION_WIDTH, HIDDEN_WIDTH))
                            .toType(dataType, false);
            attentionOutput =
                    manager.create(
                                    patternedValues(HIDDEN_WIDTH * ATTENTION_WIDTH, 37, 18, 0.002f),
                                    new Shape(HIDDEN_WIDTH, ATTENTION_WIDTH))
                            .toType(dataType, false);
            attentionOutputBias =
                    manager.create(patternedValues(HIDDEN_WIDTH, 13, 6, 0.003f))
                            .toType(dataType, false);
            feedForwardInputWeight =
                    manager.create(patternedValues(HIDDEN_WIDTH, 23, 11, 0.008f)).add(1f);
            feedForwardInputBias = manager.create(patternedValues(HIDDEN_WIDTH, 11, 5, 0.002f));
            expansion =
                    manager.create(
                                    patternedValues(
                                            FEED_FORWARD_WIDTH * HIDDEN_WIDTH, 41, 20, 0.0015f),
                                    new Shape(FEED_FORWARD_WIDTH, HIDDEN_WIDTH))
                            .toType(dataType, false);
            expansionBias =
                    manager.create(patternedValues(FEED_FORWARD_WIDTH, 17, 8, 0.002f))
                            .toType(dataType, false);
            projection =
                    manager.create(
                                    patternedValues(
                                            HIDDEN_WIDTH * FEED_FORWARD_WIDTH, 43, 21, 0.0015f),
                                    new Shape(HIDDEN_WIDTH, FEED_FORWARD_WIDTH))
                            .toType(dataType, false);
            projectionBias =
                    manager.create(patternedValues(HIDDEN_WIDTH, 19, 9, 0.002f))
                            .toType(dataType, false);
        }

        private void assertSourceConstantsUnchanged() {
            Assert.assertEquals(relationIds.getShape(), new Shape(TOKENS, TOKENS));
            Assert.assertEquals(relationKeys.getShape(), new Shape(RELATIONS, ATTENTION_WIDTH));
            Assert.assertEquals(relationBias.getShape(), new Shape(RELATIONS, HEADS));
        }

        @Override
        public void close() {
            manager.close();
        }
    }
}
