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
import ai.djl.engine.fusion.FusionFunctions;
import ai.djl.engine.fusion.FusionIndexedRelationParameters;
import ai.djl.engine.fusion.FusionSingleQueryReadoutParameters;
import ai.djl.engine.fusion.FusionTransformerBlockParameters;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDScope;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.training.GradientCollector;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Supplier;

public class FusionFunctionsCompositeTest {

    private static final float EPSILON = 1.0e-5f;

    @Test
    public void cpuTransformerStackMatchesForwardAndBackwardReference() {
        verifyTransformerStack(false);
        verifyTransformerStack(true);
    }

    @Test
    public void cpuIndexedLocalTransformerMatchesForwardAndBackwardReference() {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(Device.cpu())) {
            PairFactory factory = new PairFactory(manager);
            ArrayPair firstInput = factory.create(new Shape(2, 2, 2, 4));
            ArrayPair secondInput = factory.create(new Shape(2, 2, 1, 4));
            ArrayPair inputNormWeight = factory.createVector(4, 1f);
            ArrayPair inputNormBias = factory.createVector(4, 0f);
            BlockPair block = factory.createBlock(4, 4, 6, null);
            NDArray indices = manager.create(new long[] {0, 2, 3, 5, 6, 8, 9, 11}, new Shape(8));
            NDList actualInputs = new NDList(firstInput.actual, secondInput.actual);
            NDList expectedInputs = new NDList(firstInput.expected, secondInput.expected);

            compareForwardAndBackward(
                    engine,
                    factory.pairs,
                    manager,
                    new Shape(2, 2, 3, 4),
                    3.0e-4f,
                    () ->
                            FusionFunctions.indexedLocalTransformerEncoder(
                                    actualInputs,
                                    indices,
                                    inputNormWeight.actual,
                                    inputNormBias.actual,
                                    block.actual,
                                    2,
                                    EPSILON),
                    () ->
                            indexedLocalReference(
                                    expectedInputs,
                                    indices,
                                    inputNormWeight.expected,
                                    inputNormBias.expected,
                                    block.expected,
                                    2,
                                    EPSILON));
        }
    }

    @Test
    public void cpuSingleQueryReadoutMatchesForwardAndBackwardReference() {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(Device.cpu())) {
            PairFactory factory = new PairFactory(manager);
            ArrayPair memory = factory.create(new Shape(2, 4, 4));
            ArrayPair querySource = factory.create(new Shape(2, 2, 4));
            ReadoutPair readout = factory.createReadout(4, 4, 6);
            NDArray mask =
                    manager.create(new float[] {1f, 1f, 0f, 0f, 1f, 0f, 1f, 0f}, new Shape(2, 4));

            compareForwardAndBackward(
                    engine,
                    factory.pairs,
                    manager,
                    new Shape(2, 4),
                    3.0e-4f,
                    () ->
                            FusionFunctions.singleQueryCrossAttentionReadoutGroup(
                                            memory.actual,
                                            querySource.actual,
                                            mask,
                                            1,
                                            2,
                                            EPSILON,
                                            Collections.singletonList(readout.actual))
                                    .singletonOrThrow(),
                    () ->
                            singleQueryReference(
                                    memory.expected,
                                    querySource.expected,
                                    mask,
                                    1,
                                    2,
                                    EPSILON,
                                    readout.expected));
        }
    }

    @Test
    public void singleQueryReadoutDoesNotReadMaskedNonFiniteMemory() {
        Engine engine = Engine.getInstance();
        verifySingleQueryNonFiniteIsolation(Device.cpu(), DataType.FLOAT32);
        if (engine.getGpuCount() > 0) {
            verifySingleQueryNonFiniteIsolation(Device.gpu(), DataType.FLOAT16);
            verifySingleQueryNonFiniteIsolation(Device.gpu(), DataType.BFLOAT16);
        }
    }

    @Test
    public void singleQueryReadoutResultFollowsCallingScope() {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(Device.cpu())) {
            NDArray memory = patterned(manager, new Shape(1, 2, 4), 3, 0.037f);
            NDArray querySource = patterned(manager, new Shape(1, 4), 5, 0.031f);
            NDArray mask = manager.ones(new Shape(1, 2));
            FusionSingleQueryReadoutParameters readout =
                    typedReadout(manager, DataType.FLOAT32, 4, 4, 6);
            NDArray output;
            try (NDScope scope = new NDScope()) {
                scope.suppressNotUsedWarning();
                output =
                        FusionFunctions.singleQueryCrossAttentionReadoutGroup(
                                        memory,
                                        querySource,
                                        mask,
                                        0,
                                        2,
                                        EPSILON,
                                        Collections.singletonList(readout))
                                .singletonOrThrow();
                Assert.assertFalse(output.isReleased());
            }
            Assert.assertTrue(output.isReleased());
        }
    }

    @Test
    public void gpuSingleQueryReadoutConvertsMixedInputToProjectionType() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        for (DataType projectionType : new DataType[] {DataType.FLOAT16, DataType.BFLOAT16}) {
            try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                NDArray memory = patterned(manager, new Shape(2, 4, 4), 3, 0.037f);
                NDArray querySource = patterned(manager, new Shape(2, 2, 4), 5, 0.031f);
                memory.setRequiresGradient(true);
                querySource.setRequiresGradient(true);
                NDArray mask = manager.ones(new Shape(2, 4), DataType.FLOAT32);
                FusionSingleQueryReadoutParameters readout =
                        typedReadout(manager, projectionType, 4, 4, 6);
                readout.getQuerySeedWeight().setRequiresGradient(true);
                readout.getQueryNormWeight().setRequiresGradient(true);
                readout.getOutputNormBias().setRequiresGradient(true);
                NDArray output;
                try (GradientCollector collector = engine.newGradientCollector()) {
                    output =
                            FusionFunctions.singleQueryCrossAttentionReadoutGroup(
                                            memory,
                                            querySource,
                                            mask,
                                            1,
                                            2,
                                            EPSILON,
                                            Collections.singletonList(readout))
                                    .singletonOrThrow();
                    collector.backward(
                            output.mul(patterned(manager, output.getShape(), 11, 0.043f)).sum());
                }
                Assert.assertEquals(output.getDataType(), projectionType);
                for (NDArray differentiable :
                        new NDArray[] {
                            memory,
                            querySource,
                            readout.getQuerySeedWeight(),
                            readout.getQueryNormWeight(),
                            readout.getOutputNormBias()
                        }) {
                    Assert.assertTrue(differentiable.hasGradient());
                    assertFinite(differentiable.getGradient());
                }
            }
        }
    }

    private static void verifySingleQueryNonFiniteIsolation(
            Device device, DataType projectionType) {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray memory = patterned(manager, new Shape(2, 4, 4), 3, 0.037f);
            memory.set(new NDIndex("0,1,:"), Float.NaN);
            memory.set(new NDIndex("0,3,:"), Float.POSITIVE_INFINITY);
            memory.set(new NDIndex("1,:,:"), Float.NaN);
            NDArray querySource = patterned(manager, new Shape(2, 2, 4), 5, 0.031f);
            NDArray mask = manager.create(new int[] {1, 0, 1, 0, 0, 0, 0, 0}, new Shape(2, 4));
            memory.setRequiresGradient(true);
            querySource.setRequiresGradient(true);
            FusionSingleQueryReadoutParameters readout =
                    typedReadout(manager, projectionType, 4, 4, 6);
            readout.getQuerySeedWeight().setRequiresGradient(true);
            readout.getKeyValueWeight().setRequiresGradient(true);
            NDArray output;
            try (GradientCollector collector = engine.newGradientCollector()) {
                output =
                        FusionFunctions.singleQueryCrossAttentionReadoutGroup(
                                        memory,
                                        querySource,
                                        mask,
                                        1,
                                        2,
                                        EPSILON,
                                        Collections.singletonList(readout))
                                .singletonOrThrow();
                collector.backward(output.sum());
            }

            assertFinite(output);
            assertFinite(memory.getGradient());
            assertFinite(querySource.getGradient());
            assertFinite(readout.getQuerySeedWeight().getGradient());
            assertFinite(readout.getKeyValueWeight().getGradient());
            float[] memoryGradient =
                    memory.getGradient().toType(DataType.FLOAT32, false).toFloatArray();
            for (int batch = 0; batch < 2; ++batch) {
                for (int token = 0; token < 4; ++token) {
                    if (mask.getInt(batch, token) == 0) {
                        for (int hidden = 0; hidden < 4; ++hidden) {
                            int index = (batch * 4 + token) * 4 + hidden;
                            Assert.assertEquals(memoryGradient[index], 0f);
                        }
                    }
                }
            }
        }
    }

    private static void verifyTransformerStack(boolean indexedRelation) {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(Device.cpu())) {
            PairFactory factory = new PairFactory(manager);
            ArrayPair input = factory.create(new Shape(2, 3, 4));
            RelationPair relation = indexedRelation ? factory.createRelation(3, 3, 4, 2) : null;
            List<BlockPair> blockPairs = new ArrayList<>();
            blockPairs.add(factory.createBlock(4, 4, 6, relation));
            if (indexedRelation) {
                blockPairs.add(factory.createBlock(4, 4, 6, relation));
            }
            List<FusionTransformerBlockParameters> actualBlocks = new ArrayList<>();
            List<FusionTransformerBlockParameters> expectedBlocks = new ArrayList<>();
            for (BlockPair block : blockPairs) {
                actualBlocks.add(block.actual);
                expectedBlocks.add(block.expected);
            }
            if (indexedRelation) {
                Assert.assertNotSame(
                        actualBlocks.get(0).getOutputWeight(),
                        actualBlocks.get(1).getAttentionInputWeight());
            }

            compareForwardAndBackward(
                    engine,
                    factory.pairs,
                    manager,
                    new Shape(2, 3, 4),
                    3.0e-4f,
                    () ->
                            FusionFunctions.transformerEncoderStack(
                                    input.actual, actualBlocks, 2, EPSILON),
                    () -> transformerStackReference(input.expected, expectedBlocks, 2, EPSILON));
        }
    }

    private static void compareForwardAndBackward(
            Engine engine,
            List<ArrayPair> pairs,
            NDManager manager,
            Shape outputShape,
            float tolerance,
            Supplier<NDArray> actualForward,
            Supplier<NDArray> expectedForward) {
        NDArray lossWeight = patterned(manager, outputShape, 97, 0.031f);
        NDArray actual;
        try (GradientCollector collector = engine.newGradientCollector()) {
            actual = actualForward.get();
            collector.backward(actual.mul(lossWeight).sum());
        }
        NDArray expected;
        try (GradientCollector collector = engine.newGradientCollector()) {
            expected = expectedForward.get();
            collector.backward(expected.mul(lossWeight).sum());
        }

        assertClose(actual, expected, tolerance);
        for (ArrayPair pair : pairs) {
            Assert.assertTrue(pair.actual.hasGradient());
            Assert.assertTrue(pair.expected.hasGradient());
            assertClose(pair.actual.getGradient(), pair.expected.getGradient(), tolerance);
        }
    }

    private static NDArray transformerStackReference(
            NDArray input,
            List<FusionTransformerBlockParameters> blocks,
            int attentionHeads,
            float epsilon) {
        try (NDScope scope = new NDScope()) {
            scope.suppressNotUsedWarning();
            NDArray state = input;
            NDArray reusedNormalization = null;
            for (int index = 0; index < blocks.size(); ++index) {
                FusionTransformerBlockParameters block = blocks.get(index);
                NDArray normalized =
                        reusedNormalization == null
                                ? referenceLayerNorm(
                                        state,
                                        block.getAttentionInputWeight(),
                                        block.getAttentionInputBias(),
                                        epsilon)
                                : reusedNormalization;
                NDArray context = referenceAttention(normalized, block, attentionHeads, null);
                NDArray attentionUpdate =
                        referenceLinear(
                                context,
                                block.getAttentionOutputWeight(),
                                block.getAttentionOutputBias());
                NDArray residual = state.add(attentionUpdate);
                NDArray feedForwardInput =
                        referenceLayerNorm(
                                residual,
                                block.getFeedForwardInputWeight(),
                                block.getFeedForwardInputBias(),
                                epsilon);
                NDArray expanded =
                        referenceLinear(
                                feedForwardInput,
                                block.getFeedForwardExpansionWeight(),
                                block.getFeedForwardExpansionBias());
                NDArray feedForwardUpdate =
                        referenceLinear(
                                Activation.swish(expanded, 1f),
                                block.getFeedForwardProjectionWeight(),
                                block.getFeedForwardProjectionBias());
                NDArray summedOutput = residual.add(feedForwardUpdate);
                NDArray normalizedOutput =
                        referenceLayerNorm(
                                summedOutput,
                                block.getOutputWeight(),
                                block.getOutputBias(),
                                epsilon);
                if (block.getIndexedRelation() != null && index + 1 < blocks.size()) {
                    FusionTransformerBlockParameters next = blocks.get(index + 1);
                    state = summedOutput;
                    reusedNormalization =
                            block.getOutputWeight() == next.getAttentionInputWeight()
                                            && block.getOutputBias() == next.getAttentionInputBias()
                                    ? normalizedOutput
                                    : null;
                } else {
                    state = normalizedOutput;
                    reusedNormalization = null;
                }
            }
            NDScope.unregister(state);
            return state;
        }
    }

    private static NDArray indexedLocalReference(
            NDList inputSegments,
            NDArray indices,
            NDArray inputNormWeight,
            NDArray inputNormBias,
            FusionTransformerBlockParameters block,
            int attentionHeads,
            float epsilon) {
        try (NDScope scope = new NDScope()) {
            scope.suppressNotUsedWarning();
            NDArray input = NDArrays.concat(inputSegments, 2);
            Shape shape = input.getShape();
            long batch = shape.get(0);
            long groups = shape.get(1);
            long tokens = shape.get(2);
            long hidden = shape.get(3);
            long denseRows = batch * groups * tokens;
            NDArray selected = NDArrays.gatherRows(input.reshape(denseRows, hidden), indices);
            NDArray state = referenceLayerNorm(selected, inputNormWeight, inputNormBias, epsilon);
            NDArray normalized =
                    referenceLayerNorm(
                            state,
                            block.getAttentionInputWeight(),
                            block.getAttentionInputBias(),
                            epsilon);
            NDArray context =
                    referenceAttention(
                            normalized,
                            block,
                            attentionHeads,
                            new LocalAttentionShape(batch, groups, tokens, indices));
            NDArray update =
                    referenceLinear(
                            context,
                            block.getAttentionOutputWeight(),
                            block.getAttentionOutputBias());
            NDArray residual = state.add(update);
            NDArray feedForwardInput =
                    referenceLayerNorm(
                            residual,
                            block.getFeedForwardInputWeight(),
                            block.getFeedForwardInputBias(),
                            epsilon);
            NDArray expanded =
                    referenceLinear(
                            feedForwardInput,
                            block.getFeedForwardExpansionWeight(),
                            block.getFeedForwardExpansionBias());
            NDArray projected =
                    referenceLinear(
                            Activation.swish(expanded, 1f),
                            block.getFeedForwardProjectionWeight(),
                            block.getFeedForwardProjectionBias());
            NDArray encoded =
                    referenceLayerNorm(
                            residual.add(projected),
                            block.getOutputWeight(),
                            block.getOutputBias(),
                            epsilon);
            NDArray output = NDArrays.scatterRows(encoded, indices, denseRows).reshape(shape);
            NDScope.unregister(output);
            return output;
        }
    }

    private static NDArray referenceAttention(
            NDArray input,
            FusionTransformerBlockParameters block,
            int attentionHeads,
            LocalAttentionShape local) {
        long attentionWidth = block.getQueryKeyValueWeight().getShape().get(0) / 3;
        long headWidth = attentionWidth / attentionHeads;
        NDArray selectedQueryKeyValue =
                referenceLinear(input, block.getQueryKeyValueWeight(), null);
        long batch;
        long tokens;
        NDArray queryKeyValue;
        NDArray attentionBias = null;
        if (local == null) {
            batch = input.getShape().get(0);
            tokens = input.getShape().get(1);
            queryKeyValue = selectedQueryKeyValue;
        } else {
            batch = local.batch * local.groups;
            tokens = local.tokens;
            long denseRows = batch * tokens;
            queryKeyValue =
                    NDArrays.scatterRows(selectedQueryKeyValue, local.indices, denseRows)
                            .reshape(batch, tokens, 3 * attentionWidth);
            NDArray validRows =
                    input.getManager()
                            .ones(
                                    new Shape(local.indices.getShape().size(), 1),
                                    input.getDataType());
            NDArray valid =
                    NDArrays.scatterRows(validRows, local.indices, denseRows)
                            .reshape(batch, 1, 1, tokens);
            NDArray hasValid = valid.sum(new int[] {3}, true).gt(0).broadcast(valid.getShape());
            NDArray attentionValid =
                    NDArrays.where(hasValid, valid.neq(0), valid.onesLike().neq(0));
            attentionBias =
                    NDArrays.where(
                            attentionValid,
                            valid.zerosLike(),
                            valid.zerosLike().add(Float.NEGATIVE_INFINITY));
        }
        NDArray queries =
                queryKeyValue
                        .get("...,0:{}", attentionWidth)
                        .reshape(batch, tokens, attentionHeads, headWidth)
                        .swapAxes(1, 2);
        NDArray keys =
                queryKeyValue
                        .get("...,{}:{}", attentionWidth, 2 * attentionWidth)
                        .reshape(batch, tokens, attentionHeads, headWidth)
                        .swapAxes(1, 2);
        NDArray values =
                queryKeyValue
                        .get("...,{}:{}", 2 * attentionWidth, 3 * attentionWidth)
                        .reshape(batch, tokens, attentionHeads, headWidth)
                        .swapAxes(1, 2);
        NDArray context;
        if (block.getIndexedRelation() == null) {
            NDArray scores = queries.matMul(keys.swapAxes(2, 3)).mul(1.0 / Math.sqrt(headWidth));
            if (attentionBias != null) {
                scores = scores.add(attentionBias);
            }
            context = scores.softmax(3).matMul(values);
        } else {
            FusionIndexedRelationParameters relation = block.getIndexedRelation();
            NDArray relationIds = relation.getRelationIds().toType(DataType.INT64, false);
            long relationCount = relation.getRelationKeys().getShape().get(0);
            NDArray relationKeys =
                    relation.getRelationKeys()
                            .reshape(relationCount, attentionHeads, headWidth)
                            .transpose(1, 2, 0)
                            .expandDims(0);
            NDArray pairBias =
                    NDArrays.gatherRows(
                                    relation.getRelationBias(),
                                    relationIds.reshape(tokens * tokens))
                            .reshape(tokens, tokens, attentionHeads)
                            .transpose(2, 0, 1)
                            .expandDims(0);
            context =
                    NDArrays.relationBiasedScaledDotProductAttention(
                            queries,
                            keys,
                            values,
                            relationKeys,
                            pairBias,
                            relationIds,
                            1.0 / Math.sqrt(headWidth));
        }
        NDArray dense = context.swapAxes(1, 2).reshape(batch * tokens, attentionWidth);
        if (local != null) {
            return NDArrays.gatherRows(dense, local.indices);
        }
        return dense.reshape(input.getShape().get(0), input.getShape().get(1), attentionWidth);
    }

    private static NDArray singleQueryReference(
            NDArray memory,
            NDArray querySource,
            NDArray validMask,
            int queryIndex,
            int attentionHeads,
            float epsilon,
            FusionSingleQueryReadoutParameters readout) {
        try (NDScope scope = new NDScope()) {
            scope.suppressNotUsedWarning();
            long batch = memory.getShape().get(0);
            long tokens = memory.getShape().get(1);
            NDArray validTokens = validMask.neq(0);
            NDArray floatMask = validTokens.toType(DataType.FLOAT32, false);
            NDArray counts = floatMask.sum(new int[] {1}, true);
            NDArray selectedMemory =
                    NDArrays.where(
                            validTokens.expandDims(2).broadcast(memory.getShape()),
                            memory,
                            memory.zerosLike());
            NDArray mean =
                    selectedMemory
                            .toType(DataType.FLOAT32, false)
                            .sum(new int[] {1})
                            .div(counts.maximum(1f))
                            .toType(memory.getDataType(), false);
            NDArray seedInput =
                    NDArrays.concat(new NDList(querySource.get(":,{}", queryIndex), mean), 1);
            NDArray state =
                    referenceLinear(
                            seedInput, readout.getQuerySeedWeight(), readout.getQuerySeedBias());
            NDArray query =
                    referenceLinear(state, readout.getQueryWeight(), readout.getQueryBias());
            long attentionWidth = query.getShape().get(1);
            long headWidth = attentionWidth / attentionHeads;
            NDArray keyValue = referenceLinear(selectedMemory, readout.getKeyValueWeight(), null);
            NDArray queries = query.reshape(batch, attentionHeads, 1, headWidth);
            NDArray keys =
                    keyValue.get("...,0:{}", attentionWidth)
                            .reshape(batch, tokens, attentionHeads, headWidth)
                            .swapAxes(1, 2);
            NDArray values =
                    keyValue.get("...,{}:{}", attentionWidth, 2 * attentionWidth)
                            .reshape(batch, tokens, attentionHeads, headWidth)
                            .swapAxes(1, 2);
            NDArray scores = queries.matMul(keys.swapAxes(2, 3)).mul(1.0 / Math.sqrt(headWidth));
            NDArray presence =
                    counts.gt(0).toType(query.getDataType(), false).reshape(batch, 1, 1, 1);
            NDArray attentionValidTokens =
                    validTokens.logicalOr(counts.eq(0).broadcast(validTokens.getShape()));
            NDArray attentionBiasValues = attentionValidTokens.toType(query.getDataType(), false);
            NDArray attentionBias =
                    NDArrays.where(
                                    attentionValidTokens,
                                    attentionBiasValues.zerosLike(),
                                    attentionBiasValues.zerosLike().add(Float.NEGATIVE_INFINITY))
                            .reshape(batch, 1, 1, tokens);
            NDArray context =
                    scores.add(attentionBias)
                            .softmax(3)
                            .matMul(values)
                            .mul(presence)
                            .swapAxes(1, 2)
                            .reshape(batch, attentionWidth);
            NDArray attentionUpdate =
                    referenceLinear(context, readout.getContextWeight(), readout.getContextBias());
            state =
                    referenceLayerNorm(
                            state.add(attentionUpdate),
                            readout.getQueryNormWeight(),
                            readout.getQueryNormBias(),
                            epsilon);
            NDArray feedForwardInput =
                    referenceLayerNorm(
                            state,
                            readout.getFeedForwardNormWeight(),
                            readout.getFeedForwardNormBias(),
                            epsilon);
            NDArray expanded =
                    referenceLinear(
                            feedForwardInput,
                            readout.getFeedForwardExpansionWeight(),
                            readout.getFeedForwardExpansionBias());
            NDArray update =
                    referenceLinear(
                            Activation.swish(expanded, 1f),
                            readout.getFeedForwardProjectionWeight(),
                            readout.getFeedForwardProjectionBias());
            NDArray output =
                    referenceLayerNorm(
                            state.add(update),
                            readout.getOutputNormWeight(),
                            readout.getOutputNormBias(),
                            epsilon);
            NDScope.unregister(output);
            return output;
        }
    }

    private static NDArray referenceLinear(NDArray input, NDArray weight, NDArray bias) {
        NDArray result = input.matMul(weight.transpose());
        return bias == null ? result : result.add(bias);
    }

    private static NDArray referenceLayerNorm(
            NDArray input, NDArray weight, NDArray bias, float epsilon) {
        Shape shape = input.getShape();
        return input.getNDArrayInternal()
                .layerNorm(
                        input, new Shape(shape.get(shape.dimension() - 1)), weight, bias, epsilon)
                .singletonOrThrow();
    }

    private static NDArray patterned(NDManager manager, Shape shape, int offset, float scale) {
        return manager.arange((float) shape.size())
                .add(offset)
                .mod(23)
                .sub(11)
                .mul(scale)
                .reshape(shape);
    }

    private static FusionSingleQueryReadoutParameters typedReadout(
            NDManager manager,
            DataType projectionType,
            int hiddenWidth,
            int attentionWidth,
            int feedForwardWidth) {
        int offset = 31;
        NDArray seedWeight =
                patterned(manager, new Shape(hiddenWidth, 2L * hiddenWidth), offset++, 0.017f)
                        .toType(projectionType, false);
        NDArray seedBias =
                patterned(manager, new Shape(hiddenWidth), offset++, 0.013f)
                        .toType(projectionType, false);
        NDArray queryWeight =
                patterned(manager, new Shape(attentionWidth, hiddenWidth), offset++, 0.019f)
                        .toType(projectionType, false);
        NDArray queryBias =
                patterned(manager, new Shape(attentionWidth), offset++, 0.011f)
                        .toType(projectionType, false);
        NDArray keyValueWeight =
                patterned(manager, new Shape(2L * attentionWidth, hiddenWidth), offset++, 0.021f)
                        .toType(projectionType, false);
        NDArray contextWeight =
                patterned(manager, new Shape(hiddenWidth, attentionWidth), offset++, 0.015f)
                        .toType(projectionType, false);
        NDArray contextBias =
                patterned(manager, new Shape(hiddenWidth), offset++, 0.012f)
                        .toType(projectionType, false);
        NDArray queryNormWeight =
                patterned(manager, new Shape(hiddenWidth), offset++, 0.009f).add(1f);
        NDArray queryNormBias = patterned(manager, new Shape(hiddenWidth), offset++, 0.007f);
        NDArray feedForwardNormWeight =
                patterned(manager, new Shape(hiddenWidth), offset++, 0.009f).add(1f);
        NDArray feedForwardNormBias = patterned(manager, new Shape(hiddenWidth), offset++, 0.007f);
        NDArray expansionWeight =
                patterned(manager, new Shape(feedForwardWidth, hiddenWidth), offset++, 0.018f)
                        .toType(projectionType, false);
        NDArray expansionBias =
                patterned(manager, new Shape(feedForwardWidth), offset++, 0.010f)
                        .toType(projectionType, false);
        NDArray projectionWeight =
                patterned(manager, new Shape(hiddenWidth, feedForwardWidth), offset++, 0.016f)
                        .toType(projectionType, false);
        NDArray projectionBias =
                patterned(manager, new Shape(hiddenWidth), offset++, 0.008f)
                        .toType(projectionType, false);
        NDArray outputNormWeight =
                patterned(manager, new Shape(hiddenWidth), offset++, 0.009f).add(1f);
        NDArray outputNormBias = patterned(manager, new Shape(hiddenWidth), offset, 0.007f);
        return new FusionSingleQueryReadoutParameters(
                seedWeight,
                seedBias,
                queryWeight,
                queryBias,
                keyValueWeight,
                contextWeight,
                contextBias,
                queryNormWeight,
                queryNormBias,
                feedForwardNormWeight,
                feedForwardNormBias,
                expansionWeight,
                expansionBias,
                projectionWeight,
                projectionBias,
                outputNormWeight,
                outputNormBias);
    }

    private static void assertFinite(NDArray array) {
        for (float value : array.toType(DataType.FLOAT32, false).toFloatArray()) {
            Assert.assertTrue(Float.isFinite(value));
        }
    }

    private static void assertClose(NDArray actual, NDArray expected, float tolerance) {
        float[] actualValues = actual.toFloatArray();
        float[] expectedValues = expected.toFloatArray();
        Assert.assertEquals(actualValues.length, expectedValues.length);
        for (int index = 0; index < actualValues.length; ++index) {
            Assert.assertEquals(actualValues[index], expectedValues[index], tolerance);
        }
    }

    private static final class PairFactory {

        private final NDManager manager;
        private final List<ArrayPair> pairs;
        private int offset;

        private PairFactory(NDManager manager) {
            this.manager = manager;
            pairs = new ArrayList<>();
            offset = 1;
        }

        private ArrayPair create(Shape shape) {
            NDArray source = patterned(manager, shape, offset++, 0.037f);
            NDArray actual = source.duplicate();
            NDArray expected = source.duplicate();
            actual.setRequiresGradient(true);
            expected.setRequiresGradient(true);
            ArrayPair pair = new ArrayPair(actual, expected);
            pairs.add(pair);
            return pair;
        }

        private ArrayPair createVector(long width, float base) {
            NDArray source = patterned(manager, new Shape(width), offset++, 0.013f).add(base);
            NDArray actual = source.duplicate();
            NDArray expected = source.duplicate();
            actual.setRequiresGradient(true);
            expected.setRequiresGradient(true);
            ArrayPair pair = new ArrayPair(actual, expected);
            pairs.add(pair);
            return pair;
        }

        private BlockPair createBlock(
                int hiddenWidth, int attentionWidth, int feedForwardWidth, RelationPair relation) {
            ArrayPair attentionInputWeight = createVector(hiddenWidth, 1f);
            ArrayPair attentionInputBias = createVector(hiddenWidth, 0f);
            ArrayPair queryKeyValueWeight = create(new Shape(3L * attentionWidth, hiddenWidth));
            ArrayPair attentionOutputWeight = create(new Shape(hiddenWidth, attentionWidth));
            ArrayPair attentionOutputBias = createVector(hiddenWidth, 0f);
            ArrayPair feedForwardInputWeight = createVector(hiddenWidth, 1f);
            ArrayPair feedForwardInputBias = createVector(hiddenWidth, 0f);
            ArrayPair expansionWeight = create(new Shape(feedForwardWidth, hiddenWidth));
            ArrayPair expansionBias = createVector(feedForwardWidth, 0f);
            ArrayPair projectionWeight = create(new Shape(hiddenWidth, feedForwardWidth));
            ArrayPair projectionBias = createVector(hiddenWidth, 0f);
            ArrayPair outputWeight = createVector(hiddenWidth, 1f);
            ArrayPair outputBias = createVector(hiddenWidth, 0f);
            return new BlockPair(
                    new FusionTransformerBlockParameters(
                            attentionInputWeight.actual,
                            attentionInputBias.actual,
                            queryKeyValueWeight.actual,
                            attentionOutputWeight.actual,
                            attentionOutputBias.actual,
                            feedForwardInputWeight.actual,
                            feedForwardInputBias.actual,
                            expansionWeight.actual,
                            expansionBias.actual,
                            projectionWeight.actual,
                            projectionBias.actual,
                            outputWeight.actual,
                            outputBias.actual,
                            relation == null ? null : relation.actual),
                    new FusionTransformerBlockParameters(
                            attentionInputWeight.expected,
                            attentionInputBias.expected,
                            queryKeyValueWeight.expected,
                            attentionOutputWeight.expected,
                            attentionOutputBias.expected,
                            feedForwardInputWeight.expected,
                            feedForwardInputBias.expected,
                            expansionWeight.expected,
                            expansionBias.expected,
                            projectionWeight.expected,
                            projectionBias.expected,
                            outputWeight.expected,
                            outputBias.expected,
                            relation == null ? null : relation.expected));
        }

        private RelationPair createRelation(
                int tokens, int relationCount, int attentionWidth, int attentionHeads) {
            NDArray relationIds =
                    manager.create(new int[] {0, 1, 2, 2, 0, 1, 1, 2, 0}, new Shape(tokens, tokens))
                            .toType(DataType.INT16, false);
            ArrayPair relationKeys = create(new Shape(relationCount, attentionWidth));
            ArrayPair relationBias = create(new Shape(relationCount, attentionHeads));
            return new RelationPair(
                    new FusionIndexedRelationParameters(
                            relationIds, relationKeys.actual, relationBias.actual),
                    new FusionIndexedRelationParameters(
                            relationIds, relationKeys.expected, relationBias.expected));
        }

        private ReadoutPair createReadout(
                int hiddenWidth, int attentionWidth, int feedForwardWidth) {
            ArrayPair seedWeight = create(new Shape(hiddenWidth, 2L * hiddenWidth));
            ArrayPair seedBias = createVector(hiddenWidth, 0f);
            ArrayPair queryWeight = create(new Shape(attentionWidth, hiddenWidth));
            ArrayPair queryBias = createVector(attentionWidth, 0f);
            ArrayPair keyValueWeight = create(new Shape(2L * attentionWidth, hiddenWidth));
            ArrayPair contextWeight = create(new Shape(hiddenWidth, attentionWidth));
            ArrayPair contextBias = createVector(hiddenWidth, 0f);
            ArrayPair queryNormWeight = createVector(hiddenWidth, 1f);
            ArrayPair queryNormBias = createVector(hiddenWidth, 0f);
            ArrayPair feedForwardNormWeight = createVector(hiddenWidth, 1f);
            ArrayPair feedForwardNormBias = createVector(hiddenWidth, 0f);
            ArrayPair expansionWeight = create(new Shape(feedForwardWidth, hiddenWidth));
            ArrayPair expansionBias = createVector(feedForwardWidth, 0f);
            ArrayPair projectionWeight = create(new Shape(hiddenWidth, feedForwardWidth));
            ArrayPair projectionBias = createVector(hiddenWidth, 0f);
            ArrayPair outputNormWeight = createVector(hiddenWidth, 1f);
            ArrayPair outputNormBias = createVector(hiddenWidth, 0f);
            return new ReadoutPair(
                    new FusionSingleQueryReadoutParameters(
                            seedWeight.actual,
                            seedBias.actual,
                            queryWeight.actual,
                            queryBias.actual,
                            keyValueWeight.actual,
                            contextWeight.actual,
                            contextBias.actual,
                            queryNormWeight.actual,
                            queryNormBias.actual,
                            feedForwardNormWeight.actual,
                            feedForwardNormBias.actual,
                            expansionWeight.actual,
                            expansionBias.actual,
                            projectionWeight.actual,
                            projectionBias.actual,
                            outputNormWeight.actual,
                            outputNormBias.actual),
                    new FusionSingleQueryReadoutParameters(
                            seedWeight.expected,
                            seedBias.expected,
                            queryWeight.expected,
                            queryBias.expected,
                            keyValueWeight.expected,
                            contextWeight.expected,
                            contextBias.expected,
                            queryNormWeight.expected,
                            queryNormBias.expected,
                            feedForwardNormWeight.expected,
                            feedForwardNormBias.expected,
                            expansionWeight.expected,
                            expansionBias.expected,
                            projectionWeight.expected,
                            projectionBias.expected,
                            outputNormWeight.expected,
                            outputNormBias.expected));
        }
    }

    private static final class ArrayPair {

        private final NDArray actual;
        private final NDArray expected;

        private ArrayPair(NDArray actual, NDArray expected) {
            this.actual = actual;
            this.expected = expected;
        }
    }

    private static final class BlockPair {

        private final FusionTransformerBlockParameters actual;
        private final FusionTransformerBlockParameters expected;

        private BlockPair(
                FusionTransformerBlockParameters actual,
                FusionTransformerBlockParameters expected) {
            this.actual = actual;
            this.expected = expected;
        }
    }

    private static final class RelationPair {

        private final FusionIndexedRelationParameters actual;
        private final FusionIndexedRelationParameters expected;

        private RelationPair(
                FusionIndexedRelationParameters actual, FusionIndexedRelationParameters expected) {
            this.actual = actual;
            this.expected = expected;
        }
    }

    private static final class ReadoutPair {

        private final FusionSingleQueryReadoutParameters actual;
        private final FusionSingleQueryReadoutParameters expected;

        private ReadoutPair(
                FusionSingleQueryReadoutParameters actual,
                FusionSingleQueryReadoutParameters expected) {
            this.actual = actual;
            this.expected = expected;
        }
    }

    private static final class LocalAttentionShape {

        private final long batch;
        private final long groups;
        private final long tokens;
        private final NDArray indices;

        private LocalAttentionShape(long batch, long groups, long tokens, NDArray indices) {
            this.batch = batch;
            this.groups = groups;
            this.tokens = tokens;
            this.indices = indices;
        }
    }
}
