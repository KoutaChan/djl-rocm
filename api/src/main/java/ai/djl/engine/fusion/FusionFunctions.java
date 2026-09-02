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
package ai.djl.engine.fusion;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDScope;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.LayerNorm;

import java.util.List;
import java.util.Objects;

/**
 * Provides differentiable functional counterparts to bounded fusion stages.
 *
 * <p>Each method returns ordinary caller-owned NDArrays that follow normal {@link
 * ai.djl.ndarray.NDManager} lifetime rules and participate in the active engine's automatic
 * differentiation graph. An engine may use a fused forward and backward implementation when one is
 * available without changing those functional semantics.
 *
 * <p>This API complements the persistent, inference-only {@link FusionRecipe} and {@link
 * FusionSession} API. Functional results never refer to reusable session ring slots and therefore
 * do not require a {@link FusionOutputLease}. Caller ownership describes lifetime rather than
 * storage uniqueness; a no-op single-source stage may return its input or a view.
 */
public final class FusionFunctions {

    private FusionFunctions() {}

    /**
     * Concatenates floating-point sources along their last axis and converts the result type.
     *
     * @param sources sources with equal leading dimensions
     * @param outputDataType output floating-point data type
     * @return the packed value
     */
    public static NDArray outputPack(NDList sources, DataType outputDataType) {
        validatePackDataTypes(sources, outputDataType);
        NDArray first = first(sources);
        return NDArrays.concatToType(sources, first.getShape().dimension() - 1, outputDataType);
    }

    /**
     * Packs selected token ranges from batch-major sources into one contiguous memory.
     *
     * <p>Each source is shaped {@code [batch, groups..., tokens, hidden]}. Group axes are flattened
     * in row-major order, and the selected token range is appended once for every group.
     *
     * @param sources sources in packing order
     * @param tokenOffsets first selected token for each source
     * @param tokenCounts selected tokens per source group
     * @param outputDataType output floating-point data type
     * @return the packed {@code [batch, packedTokens, hidden]} value
     */
    public static NDArray segmentedOutputPack(
            NDList sources, long[] tokenOffsets, long[] tokenCounts, DataType outputDataType) {
        validatePackDataTypes(sources, outputDataType);
        return first(sources)
                .getNDArrayInternal()
                .segmentedOutputPack(
                        sources.subNDList(1), tokenOffsets, tokenCounts, outputDataType);
    }

    /**
     * Blends two contexts using their presence values and a selected-branch logit.
     *
     * <p>The operation implements the same differentiable equation as {@link
     * FusionRecipe.BinaryBranchBlend}. Contexts are {@code [batch, width]}; the logit and presence
     * values are {@code [batch, 1]}. The result is FLOAT32.
     *
     * @param baselineContext baseline context
     * @param selectedContext selected context
     * @param selectedLogit selected-branch logit
     * @param baselinePresence baseline presence value
     * @param selectedPresence selected presence value
     * @return the blended context
     */
    public static NDArray binaryBranchBlend(
            NDArray baselineContext,
            NDArray selectedContext,
            NDArray selectedLogit,
            NDArray baselinePresence,
            NDArray selectedPresence) {
        return baselineContext
                .getNDArrayInternal()
                .binaryBranchBlend(
                        selectedContext, selectedLogit, baselinePresence, selectedPresence);
    }

    /**
     * Projects, broadcasts, and sums several values before applying an optional activation.
     *
     * <p>For matching pairs {@code (X_i, W_i)}, this computes {@code activation(bias + sum_i X_i
     * W_i^T)}. Every weight is shaped {@code [outputWidth, inputWidth]}. Input prefix dimensions
     * follow ordinary broadcasting rules. The optional bias is shaped {@code [outputWidth]}.
     *
     * @param inputs values to project
     * @param weights projection weights paired with {@code inputs}
     * @param bias optional output bias
     * @param activation output activation
     * @return the affine sum
     */
    public static NDArray affineSum(
            NDList inputs, NDList weights, NDArray bias, FusionRecipe.Activation activation) {
        return first(inputs)
                .getNDArrayInternal()
                .affineSum(inputs.subNDList(1), weights, bias, Objects.requireNonNull(activation));
    }

    /**
     * Gathers, projects, and scatters rows with one shared destination index vector.
     *
     * <p>For active row {@code p}, {@code destinationIndices[p]} selects destination {@code d}.
     * Source {@code i} contributes row {@code floor(d / indexDivisors[i])}. The gathered rows are
     * concatenated and transformed as {@code outputWeight * activation(hiddenWeight * x +
     * hiddenBias) + outputBias}. The compact results are scattered to {@code d}; unselected rows
     * are zero. Destination indices must be unique and in {@code [0, destinationRows)}.
     *
     * <p>Destination indices and divisors are routing metadata and are not differentiable. Source,
     * weight, and bias gradients follow ordinary gather, affine, activation, and scatter semantics.
     *
     * @param destinationIndices one-dimensional INT32 or INT64 destination indices
     * @param sources two-dimensional floating-point sources in concatenation order
     * @param indexDivisors positive source index divisors, one per source
     * @param destinationRows number of rows in the dense result
     * @param hiddenWeight hidden projection weight shaped {@code [hiddenWidth, concatenatedWidth]}
     * @param hiddenBias optional hidden projection bias shaped {@code [hiddenWidth]}
     * @param outputWeight output projection weight shaped {@code [outputWidth, hiddenWidth]}
     * @param outputBias optional output projection bias shaped {@code [outputWidth]}
     * @param activation activation after the hidden projection
     * @return dense result shaped {@code [destinationRows, outputWidth]}
     */
    public static NDArray indexedAffine(
            NDArray destinationIndices,
            NDList sources,
            long[] indexDivisors,
            long destinationRows,
            NDArray hiddenWeight,
            NDArray hiddenBias,
            NDArray outputWeight,
            NDArray outputBias,
            FusionRecipe.Activation activation) {
        Objects.requireNonNull(destinationIndices, "destinationIndices");
        validateRoutingVector(destinationIndices, "Destination indices");
        Objects.requireNonNull(indexDivisors, "indexDivisors");
        Objects.requireNonNull(hiddenWeight, "hiddenWeight");
        Objects.requireNonNull(outputWeight, "outputWeight");
        return destinationIndices
                .getNDArrayInternal()
                .indexedAffine(
                        Objects.requireNonNull(sources, "sources"),
                        indexDivisors,
                        destinationRows,
                        hiddenWeight,
                        hiddenBias,
                        outputWeight,
                        outputBias,
                        Objects.requireNonNull(activation, "activation"));
    }

    /**
     * Pools one candidate memory into several independently mapped output sets.
     *
     * <p>{@code scores}, {@code masks}, and {@code values} have shapes {@code [batch, candidates]},
     * {@code [batch, candidates, groups]}, and {@code [batch, candidates, width]}. Each
     * one-dimensional integer mapping selects a source group for every destination in one output
     * set. Mapping value {@code -1} produces a zero context and zero presence. Source groups may be
     * repeated.
     *
     * <p>Each context list element has shape {@code [batch, destinations, width]} and FLOAT32 data
     * type. The corresponding presence element has shape {@code [batch, destinations]} and the mask
     * data type. Scores and values are differentiable; masks and mappings are not.
     *
     * @param scores shared floating-point scores shaped {@code [batch, candidates]}
     * @param masks floating-point masks shaped {@code [batch, candidates, groups]}
     * @param values floating-point values shaped {@code [batch, candidates, width]}
     * @param destinationGroupIndices one INT32 or INT64 mapping per output set
     * @return contexts and presence values in mapping order
     */
    public static MappedGroupedMaskedSoftmaxPoolResult mappedGroupedMaskedSoftmaxPool(
            NDArray scores, NDArray masks, NDArray values, NDList destinationGroupIndices) {
        Objects.requireNonNull(scores, "scores");
        Objects.requireNonNull(masks, "masks");
        Objects.requireNonNull(values, "values");
        Objects.requireNonNull(destinationGroupIndices, "destinationGroupIndices");
        for (NDArray mapping : destinationGroupIndices) {
            validateRoutingVector(mapping, "Destination group mappings");
        }
        NDList results =
                scores.getNDArrayInternal()
                        .mappedGroupedMaskedSoftmaxPool(masks, values, destinationGroupIndices);
        int outputSetCount = destinationGroupIndices.size();
        return new MappedGroupedMaskedSoftmaxPoolResult(
                results.subNDList(0, outputSetCount), results.subNDList(outputSetCount));
    }

    /**
     * Applies a stack of functional transformer encoder blocks.
     *
     * <p>This method is the caller-owned, differentiable counterpart of {@link
     * FusionRecipe.TransformerEncoderStack}. Ordinary and indexed-relation attention use the same
     * parameter carrier. Floating-point inputs and parameters retain their autograd connections;
     * relation IDs, head count, and epsilon are non-differentiable metadata.
     *
     * @param input input shaped {@code [batch, tokens, hiddenWidth]}
     * @param blocks encoder blocks in execution order
     * @param attentionHeads attention head count
     * @param epsilon LayerNorm epsilon
     * @return encoded value with the same shape as {@code input}
     */
    public static NDArray transformerEncoderStack(
            NDArray input,
            List<FusionTransformerBlockParameters> blocks,
            int attentionHeads,
            float epsilon) {
        if (blocks.isEmpty()) {
            throw new IllegalArgumentException("A transformer encoder stack requires a block.");
        }
        for (FusionTransformerBlockParameters block : blocks) {
            transformerAttentionWidth(block, attentionHeads);
        }
        try (NDScope scope = new NDScope()) {
            scope.suppressNotUsedWarning();
            NDArray state = input;
            NDArray reusedNormalization = null;
            for (int index = 0; index < blocks.size(); ++index) {
                FusionTransformerBlockParameters block = blocks.get(index);
                NDArray attentionInput =
                        reusedNormalization == null
                                ? layerNorm(
                                        state,
                                        block.getAttentionInputWeight(),
                                        block.getAttentionInputBias(),
                                        epsilon)
                                : reusedNormalization;
                NDArray attentionContext =
                        transformerAttention(attentionInput, block, attentionHeads);
                NDArray attentionUpdate =
                        linear(
                                attentionContext,
                                block.getAttentionOutputWeight(),
                                block.getAttentionOutputBias());
                NDList feedForwardState =
                        residualLayerNorm(
                                state,
                                attentionUpdate,
                                block.getFeedForwardInputWeight(),
                                block.getFeedForwardInputBias(),
                                epsilon);
                NDArray residual = feedForwardState.get(1);
                NDArray expanded =
                        linear(
                                feedForwardState.get(0),
                                block.getFeedForwardExpansionWeight(),
                                block.getFeedForwardExpansionBias());
                NDArray feedForwardUpdate =
                        linear(
                                Activation.swish(expanded, 1f),
                                block.getFeedForwardProjectionWeight(),
                                block.getFeedForwardProjectionBias());
                NDList outputState =
                        residualLayerNorm(
                                residual,
                                feedForwardUpdate,
                                block.getOutputWeight(),
                                block.getOutputBias(),
                                epsilon);

                boolean relationStack = block.getIndexedRelation() != null;
                if (relationStack && index + 1 < blocks.size()) {
                    state = outputState.get(1);
                    reusedNormalization =
                            canReuseOutputNormalization(blocks, index) ? outputState.get(0) : null;
                } else {
                    state = outputState.get(0);
                    reusedNormalization = null;
                }
            }
            NDScope.unregister(state);
            return state;
        }
    }

    /**
     * Applies one transformer encoder block to indexed rows of logical local-group inputs.
     *
     * <p>Input segments are shaped {@code [batch, groups, segmentTokens, hiddenWidth]} and are
     * logically concatenated on the token axis. The result has the corresponding dense shape;
     * unselected rows are zero. Gradients flow to selected source rows and all floating-point
     * parameters. The flattened row indices are non-differentiable metadata.
     *
     * @param inputSegments local-group input segments in token order
     * @param indices unique flattened token-row indices
     * @param inputNormWeight input LayerNorm scale
     * @param inputNormBias input LayerNorm bias
     * @param block transformer block parameters
     * @param attentionHeads attention head count
     * @param epsilon LayerNorm epsilon
     * @return dense encoded local-group value
     */
    public static NDArray indexedLocalTransformerEncoder(
            NDList inputSegments,
            NDArray indices,
            NDArray inputNormWeight,
            NDArray inputNormBias,
            FusionTransformerBlockParameters block,
            int attentionHeads,
            float epsilon) {
        validateRoutingVector(indices, "Indexed local transformer indices");
        long attentionWidth = transformerAttentionWidth(block, attentionHeads);
        if (block.getIndexedRelation() != null) {
            throw new IllegalArgumentException(
                    "Indexed local transformer attention does not use relation parameters.");
        }
        try (NDScope scope = new NDScope()) {
            scope.suppressNotUsedWarning();
            NDArray input =
                    inputSegments.size() == 1
                            ? inputSegments.get(0)
                            : NDArrays.concat(inputSegments, 2);
            Shape inputShape = input.getShape();
            long batch = inputShape.get(0);
            long groups = inputShape.get(1);
            long tokens = inputShape.get(2);
            long hiddenWidth = inputShape.get(3);
            long denseRows = Math.multiplyExact(Math.multiplyExact(batch, groups), tokens);
            NDArray routingIndices = indices.toType(DataType.INT64, false).stopGradient();
            long activeRows = routingIndices.getShape().size();

            NDArray selectedInput =
                    NDArrays.gatherRows(input.reshape(denseRows, hiddenWidth), routingIndices);
            NDArray state = layerNorm(selectedInput, inputNormWeight, inputNormBias, epsilon);
            NDArray attentionInput =
                    layerNorm(
                            state,
                            block.getAttentionInputWeight(),
                            block.getAttentionInputBias(),
                            epsilon);
            long headWidth = attentionWidth / attentionHeads;
            NDArray selectedQueryKeyValue =
                    linear(attentionInput, block.getQueryKeyValueWeight(), null);
            NDArray queryKeyValue =
                    NDArrays.scatterRows(selectedQueryKeyValue, routingIndices, denseRows)
                            .reshape(batch * groups, tokens, 3 * attentionWidth);
            NDArray queries =
                    queryKeyValue
                            .get("...,0:{}", attentionWidth)
                            .reshape(batch * groups, tokens, attentionHeads, headWidth)
                            .swapAxes(1, 2);
            NDArray keys =
                    queryKeyValue
                            .get("...,{}:{}", attentionWidth, 2 * attentionWidth)
                            .reshape(batch * groups, tokens, attentionHeads, headWidth)
                            .swapAxes(1, 2);
            NDArray values =
                    queryKeyValue
                            .get("...,{}:{}", 2 * attentionWidth, 3 * attentionWidth)
                            .reshape(batch * groups, tokens, attentionHeads, headWidth)
                            .swapAxes(1, 2);
            NDArray validRows =
                    indices.getManager().ones(new Shape(activeRows, 1), input.getDataType());
            NDArray validMask =
                    NDArrays.scatterRows(validRows, routingIndices, denseRows)
                            .reshape(batch * groups, 1, 1, tokens);
            NDArray attentionBias = validMask.neg().add(1f).mul(-1.0e9f);
            NDArray context =
                    queries.getNDArrayInternal()
                            .scaledDotProductAttention(keys, values, attentionBias, 0.0, false)
                            .swapAxes(1, 2)
                            .reshape(denseRows, attentionWidth);
            NDArray selectedContext = NDArrays.gatherRows(context, routingIndices);
            NDArray attentionUpdate =
                    linear(
                            selectedContext,
                            block.getAttentionOutputWeight(),
                            block.getAttentionOutputBias());
            NDList feedForwardState =
                    residualLayerNorm(
                            state,
                            attentionUpdate,
                            block.getFeedForwardInputWeight(),
                            block.getFeedForwardInputBias(),
                            epsilon);
            NDArray expanded =
                    linear(
                            feedForwardState.get(0),
                            block.getFeedForwardExpansionWeight(),
                            block.getFeedForwardExpansionBias());
            NDArray feedForwardUpdate =
                    linear(
                            Activation.swish(expanded, 1f),
                            block.getFeedForwardProjectionWeight(),
                            block.getFeedForwardProjectionBias());
            NDArray encoded =
                    residualLayerNorm(
                                    feedForwardState.get(1),
                                    feedForwardUpdate,
                                    block.getOutputWeight(),
                                    block.getOutputBias(),
                                    epsilon)
                            .get(0);
            NDArray output =
                    NDArrays.scatterRows(
                                    encoded.reshape(activeRows, hiddenWidth),
                                    routingIndices,
                                    denseRows)
                            .reshape(inputShape);
            NDScope.unregister(output);
            return output;
        }
    }

    /**
     * Evaluates several differentiable single-query cross-attention readouts over shared memory.
     *
     * <p>This is the caller-owned counterpart of {@link
     * FusionRecipe.SingleQueryCrossAttentionReadoutGroup}. The nonzero validity mask and query
     * index are discrete metadata. Gradients flow to memory, the selected query-source row, and all
     * floating-point parameters. One independently owned state is returned per readout.
     *
     * @param memory shared memory shaped {@code [batch, tokens, hiddenWidth]}
     * @param querySource source shaped {@code [batch, hiddenWidth]} or {@code [batch, queryTokens,
     *     hiddenWidth]}
     * @param validMask nonzero-is-valid mask shaped {@code [batch, tokens]}
     * @param queryIndex selected query token, or zero for a two-dimensional source
     * @param attentionHeads attention head count
     * @param epsilon LayerNorm epsilon
     * @param readouts readout parameter sets in output order
     * @return readout states in declaration order
     */
    public static NDList singleQueryCrossAttentionReadoutGroup(
            NDArray memory,
            NDArray querySource,
            NDArray validMask,
            int queryIndex,
            int attentionHeads,
            float epsilon,
            List<FusionSingleQueryReadoutParameters> readouts) {
        if (readouts.isEmpty()) {
            throw new IllegalArgumentException("A readout group requires a readout.");
        }
        int queryRank = querySource.getShape().dimension();
        if (queryRank != 2 && queryRank != 3) {
            throw new IllegalArgumentException("Query source must have rank two or three.");
        }
        if ((queryRank == 2 && queryIndex != 0)
                || (queryRank == 3
                        && (queryIndex < 0 || queryIndex >= querySource.getShape().get(1)))) {
            throw new IllegalArgumentException("Query index is outside the query source.");
        }
        try (NDScope scope = new NDScope()) {
            scope.suppressNotUsedWarning();
            Shape memoryShape = memory.getShape();
            long batch = memoryShape.get(0);
            long tokens = memoryShape.get(1);
            NDArray floatMask = validMask.neq(0).toType(DataType.FLOAT32, false);
            NDArray validCount = floatMask.sum(new int[] {1}, true);
            NDArray mean =
                    differentiableCast(memory, DataType.FLOAT32)
                            .mul(floatMask.expandDims(2))
                            .sum(new int[] {1})
                            .div(validCount.maximum(1f));
            NDArray querySeed =
                    querySource.getShape().dimension() == 2
                            ? querySource
                            : querySource.get(":,{}", queryIndex);
            DataType projectionDataType = readouts.get(0).getQuerySeedWeight().getDataType();
            NDArray seedInput =
                    NDArrays.concat(
                            new NDList(
                                    differentiableCast(querySeed, projectionDataType),
                                    differentiableCast(mean, projectionDataType)),
                            1);
            NDArray projectionMemory = differentiableCast(memory, projectionDataType);
            NDArray presence = validCount.gt(0).reshape(batch, 1, 1, 1);
            NDList outputs = new NDList(readouts.size());
            for (FusionSingleQueryReadoutParameters readout : readouts) {
                NDArray seedState =
                        linear(seedInput, readout.getQuerySeedWeight(), readout.getQuerySeedBias());
                NDArray query = linear(seedState, readout.getQueryWeight(), readout.getQueryBias());
                long attentionWidth = query.getShape().get(1);
                long headWidth = attentionHeadWidth(attentionWidth, attentionHeads);
                if (readout.getKeyValueWeight().getShape().get(0) != 2 * attentionWidth) {
                    throw new IllegalArgumentException(
                            "Key-value projection width must be twice the attention width.");
                }
                NDArray keyValue = linear(projectionMemory, readout.getKeyValueWeight(), null);
                NDArray queries = query.reshape(batch, attentionHeads, 1, headWidth);
                NDArray keys =
                        keyValue.get("...,0:{}", attentionWidth)
                                .reshape(batch, tokens, attentionHeads, headWidth)
                                .swapAxes(1, 2);
                NDArray values =
                        keyValue.get("...,{}:{}", attentionWidth, 2 * attentionWidth)
                                .reshape(batch, tokens, attentionHeads, headWidth)
                                .swapAxes(1, 2);
                NDArray attentionBias =
                        validMask
                                .eq(0)
                                .toType(query.getDataType(), false)
                                .mul(-1.0e9f)
                                .reshape(batch, 1, 1, tokens);
                NDArray context =
                        queries.getNDArrayInternal()
                                .scaledDotProductAttention(keys, values, attentionBias, 0.0, false)
                                .mul(presence.toType(query.getDataType(), false))
                                .swapAxes(1, 2)
                                .reshape(batch, attentionWidth);
                NDArray attentionUpdate =
                        linear(context, readout.getContextWeight(), readout.getContextBias());
                NDArray state =
                        readoutResidualLayerNorm(
                                seedState,
                                attentionUpdate,
                                readout.getQueryNormWeight(),
                                readout.getQueryNormBias(),
                                epsilon,
                                projectionDataType);
                NDArray feedForwardInput =
                        readoutLayerNorm(
                                state,
                                readout.getFeedForwardNormWeight(),
                                readout.getFeedForwardNormBias(),
                                epsilon,
                                projectionDataType);
                NDArray expanded =
                        linear(
                                feedForwardInput,
                                readout.getFeedForwardExpansionWeight(),
                                readout.getFeedForwardExpansionBias());
                NDArray feedForwardUpdate =
                        linear(
                                Activation.swish(expanded, 1f),
                                readout.getFeedForwardProjectionWeight(),
                                readout.getFeedForwardProjectionBias());
                NDArray output =
                        readoutResidualLayerNorm(
                                state,
                                feedForwardUpdate,
                                readout.getOutputNormWeight(),
                                readout.getOutputNormBias(),
                                epsilon,
                                projectionDataType);
                outputs.add(output);
            }
            NDScope.unregister(outputs);
            return outputs;
        }
    }

    /**
     * Applies a projected-residual multilayer perceptron.
     *
     * <p>The result is an ordinary caller-owned {@link NDArray} that participates in the engine's
     * automatic differentiation graph. A backend may select an equivalent fused implementation; the
     * result is never backed by a reusable {@link FusionSession} output slot and does not require a
     * {@link FusionOutputLease}.
     *
     * @param input input shaped {@code [..., I]}
     * @param combinedWeight combined projection weight shaped {@code [O + H, I]}
     * @param combinedBias combined projection bias shaped {@code [O + H]}
     * @param outputWeight bias-free output projection weight shaped {@code [O, H]}
     * @return output shaped {@code [..., O]}
     */
    public static NDArray projectedResidualMlp(
            NDArray input, NDArray combinedWeight, NDArray combinedBias, NDArray outputWeight) {
        return NDArrays.projectedResidualMlp(input, combinedWeight, combinedBias, outputWeight);
    }

    private static NDArray transformerAttention(
            NDArray input, FusionTransformerBlockParameters block, int attentionHeads) {
        Shape inputShape = input.getShape();
        long batch = inputShape.get(0);
        long tokens = inputShape.get(1);
        long attentionWidth = transformerAttentionWidth(block, attentionHeads);
        long headWidth = attentionWidth / attentionHeads;
        NDArray queryKeyValue = linear(input, block.getQueryKeyValueWeight(), null);
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
        FusionIndexedRelationParameters relation = block.getIndexedRelation();
        NDArray context;
        if (relation == null) {
            context =
                    queries.getNDArrayInternal()
                            .scaledDotProductAttention(keys, values, null, 0.0, false);
        } else {
            validateRelationIds(relation.getRelationIds(), tokens);
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
        return context.swapAxes(1, 2).reshape(batch, tokens, attentionWidth);
    }

    private static boolean canReuseOutputNormalization(
            List<FusionTransformerBlockParameters> blocks, int index) {
        if (index + 1 >= blocks.size() || blocks.get(index).getIndexedRelation() == null) {
            return false;
        }
        FusionTransformerBlockParameters current = blocks.get(index);
        FusionTransformerBlockParameters next = blocks.get(index + 1);
        return next.getIndexedRelation() != null
                && current.getOutputWeight() == next.getAttentionInputWeight()
                && current.getOutputBias() == next.getAttentionInputBias();
    }

    private static NDArray linear(NDArray input, NDArray weight, NDArray bias) {
        return Linear.linear(input, weight, bias).singletonOrThrow();
    }

    private static NDArray layerNorm(NDArray input, NDArray weight, NDArray bias, float epsilon) {
        Shape shape = input.getShape();
        return LayerNorm.layerNorm(
                        input, new Shape(shape.get(shape.dimension() - 1)), weight, bias, epsilon)
                .singletonOrThrow();
    }

    private static NDList residualLayerNorm(
            NDArray residual, NDArray update, NDArray weight, NDArray bias, float epsilon) {
        Shape shape = residual.getShape();
        return LayerNorm.residualAddLayerNorm(
                residual,
                update,
                new Shape(shape.get(shape.dimension() - 1)),
                weight,
                bias,
                epsilon);
    }

    private static NDArray readoutLayerNorm(
            NDArray input, NDArray weight, NDArray bias, float epsilon, DataType outputDataType) {
        NDArray normalized =
                layerNorm(differentiableCast(input, weight.getDataType()), weight, bias, epsilon);
        return differentiableCast(normalized, outputDataType);
    }

    private static NDArray readoutResidualLayerNorm(
            NDArray residual,
            NDArray update,
            NDArray weight,
            NDArray bias,
            float epsilon,
            DataType outputDataType) {
        if (weight.getDataType() == outputDataType) {
            NDArray normalized = residualLayerNorm(residual, update, weight, bias, epsilon).get(0);
            return differentiableCast(normalized, outputDataType);
        }
        NDArray summed = differentiableCast(residual.add(update), outputDataType);
        NDArray normalized =
                layerNorm(differentiableCast(summed, weight.getDataType()), weight, bias, epsilon);
        return differentiableCast(normalized, outputDataType);
    }

    private static NDArray differentiableCast(NDArray array, DataType dataType) {
        if (array.getDataType() == dataType) {
            return array;
        }
        return array.getNDArrayInternal().differentiableCast(dataType);
    }

    private static NDArray first(NDList arrays) {
        if (arrays.isEmpty()) {
            throw new IllegalArgumentException("At least one source is required.");
        }
        return arrays.get(0);
    }

    private static long transformerAttentionWidth(
            FusionTransformerBlockParameters block, int attentionHeads) {
        Shape shape = block.getQueryKeyValueWeight().getShape();
        if (shape.dimension() != 2 || shape.get(0) % 3 != 0) {
            throw new IllegalArgumentException(
                    "Query-key-value projection must have three equally sized outputs.");
        }
        long attentionWidth = shape.get(0) / 3;
        attentionHeadWidth(attentionWidth, attentionHeads);
        return attentionWidth;
    }

    private static long attentionHeadWidth(long attentionWidth, int attentionHeads) {
        if (attentionHeads <= 0 || attentionWidth <= 0 || attentionWidth % attentionHeads != 0) {
            throw new IllegalArgumentException(
                    "Attention width must be positive and divisible by the head count.");
        }
        return attentionWidth / attentionHeads;
    }

    private static void validateRoutingVector(NDArray routing, String name) {
        DataType dataType = routing.getDataType();
        if (routing.getShape().dimension() != 1
                || (dataType != DataType.INT32 && dataType != DataType.INT64)) {
            throw new IllegalArgumentException(name + " must be a rank-one INT32 or INT64 tensor.");
        }
    }

    private static void validateRelationIds(NDArray relationIds, long tokens) {
        DataType dataType = relationIds.getDataType();
        if ((dataType != DataType.INT32 && dataType != DataType.INT64)
                || !relationIds.getShape().equals(new Shape(tokens, tokens))) {
            throw new IllegalArgumentException(
                    "Relation IDs must be a square INT32 or INT64 token matrix.");
        }
    }

    private static void validatePackDataTypes(NDList sources, DataType outputDataType) {
        first(sources);
        if (!isFusionDataType(outputDataType)) {
            throw new IllegalArgumentException(
                    "Fusion output packs require FLOAT16, BFLOAT16, or FLOAT32 output.");
        }
        for (NDArray source : sources) {
            if (!isFusionDataType(source.getDataType())) {
                throw new IllegalArgumentException(
                        "Fusion output packs require FLOAT16, BFLOAT16, or FLOAT32 sources.");
            }
        }
    }

    private static boolean isFusionDataType(DataType dataType) {
        return dataType == DataType.FLOAT16
                || dataType == DataType.BFLOAT16
                || dataType == DataType.FLOAT32;
    }
}
