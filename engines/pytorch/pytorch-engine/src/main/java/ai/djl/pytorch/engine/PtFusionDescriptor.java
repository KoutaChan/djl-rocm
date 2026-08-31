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

import ai.djl.engine.fusion.FusionRecipe;
import ai.djl.ndarray.types.DataType;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Encodes a fusion recipe into the closed, table-based PyTorch native command format.
 *
 * <p>Version 2 is a native-order {@code int64} stream. Its 16-word header contains magic, version,
 * total words, flags, six table counts, and the offsets of the dimension, value, input, constant,
 * command, and output tables. A dimension record is {@code [recordWords, kind, maximumExtent]}. A
 * value record is {@code [recordWords, valueId, dtype, layout, leadingDimensionId|-1, innerRank,
 * flags, innerDimensions...]}. Input and constant records map a binding ordinal to a value ID. A
 * command uses the generic envelope {@code [recordWords, opcode, flags, resultCount, operandCount,
 * attributeCount, results..., operands..., attributes...]}; each typed attribute is {@code
 * [recordWords, key, type, elementCount, payload...]}. An output record maps an output ordinal to a
 * value ID.
 *
 * <p>Wire dtype, layout, dimension-kind, attribute-type, and opcode numbers are stable protocol
 * constants and must never depend on Java enum ordinals. New compatible constants may be appended;
 * incompatible record changes require a descriptor version bump. The mirrored native parser in
 * {@code djl_pytorch_fusion.cc} rejects unknown constants and unsupported nonzero flags.
 */
final class PtFusionDescriptor {

    static final long MAGIC = 0x444a4c5f46555332L;
    static final long VERSION = 2;
    static final long DTYPE_FLOAT16 = 1;
    static final long DTYPE_BFLOAT16 = 2;
    static final long DTYPE_FLOAT32 = 3;
    static final long DTYPE_BOOLEAN = 4;
    static final long DTYPE_UINT8 = 5;
    static final long DTYPE_INT8 = 6;
    static final long DTYPE_INT16 = 7;
    static final long DTYPE_INT32 = 8;
    static final long DTYPE_INT64 = 9;
    static final long DTYPE_FLOAT64 = 10;
    static final long OUTPUT_PACK_V1 = 1;
    static final long AFFINE_SUM_V1 = 2;
    static final long INDEXED_AFFINE_V1 = 3;
    static final long TRANSFORMER_ENCODER_STACK_V1 = 4;
    static final long BINARY_BRANCH_BLEND_V1 = 5;
    static final long SINGLE_QUERY_CROSS_ATTENTION_READOUT_GROUP_V1 = 6;
    // Opcode 7 is reserved for INDEXED_BINARY_SOFTMAX_POOL_V1.
    static final long INDEXED_LOCAL_TRANSFORMER_ENCODER_V1 = 8;
    static final long DIMENSION_PREFIX_EXTENT = 1;
    static final long LAYOUT_CONTIGUOUS = 1;
    static final long ATTRIBUTE_INT64 = 1;
    static final long ATTRIBUTE_FLOAT64_BITS = 2;
    static final long AFFINE_TERM_COUNT = 1;
    static final long AFFINE_ACTIVATION = 2;
    static final long AFFINE_HAS_BIAS = 3;
    static final long INDEXED_SOURCE_COUNT = 1;
    static final long INDEXED_ACTIVATION = 2;
    static final long INDEXED_HAS_HIDDEN_BIAS = 3;
    static final long INDEXED_HAS_OUTPUT_BIAS = 4;
    static final long INDEXED_SOURCE_DIVISORS = 5;
    static final long TRANSFORMER_BLOCK_COUNT = 1;
    static final long TRANSFORMER_ATTENTION_HEADS = 2;
    static final long TRANSFORMER_ATTENTION_WIDTH = 3;
    static final long TRANSFORMER_FEED_FORWARD_WIDTH = 4;
    static final long TRANSFORMER_EPSILON = 5;
    static final long READOUT_COUNT = 1;
    static final long READOUT_ATTENTION_HEADS = 2;
    static final long READOUT_ATTENTION_WIDTH = 3;
    static final long READOUT_MAXIMUM_FEED_FORWARD_WIDTH = 4;
    static final long READOUT_EPSILON = 5;
    static final long READOUT_FEED_FORWARD_WIDTHS = 6;
    static final long READOUT_QUERY_INDEX = 7;
    static final long LOCAL_TRANSFORMER_ATTENTION_HEADS = 1;
    static final long LOCAL_TRANSFORMER_ATTENTION_WIDTH = 2;
    static final long LOCAL_TRANSFORMER_FEED_FORWARD_WIDTH = 3;
    static final long LOCAL_TRANSFORMER_EPSILON = 4;
    static final long INDEXED_LOCAL_TRANSFORMER_TILE_ROWS = 65_536;
    static final long ACTIVATION_NONE = 0;
    static final long ACTIVATION_SILU = 1;

    static final int HEADER_WORDS = 16;
    static final int DIMENSION_RECORD_WORDS = 3;
    static final int VALUE_RECORD_HEADER_WORDS = 7;
    static final int BINDING_RECORD_WORDS = 2;
    static final int COMMAND_RECORD_HEADER_WORDS = 6;
    static final int SCALAR_ATTRIBUTE_WORDS = 5;
    static final int OUTPUT_RECORD_WORDS = 2;

    private PtFusionDescriptor() {}

    static ByteBuffer encode(FusionRecipe recipe) {
        int dimensionWords =
                Math.multiplyExact(recipe.getDimensions().size(), DIMENSION_RECORD_WORDS);
        int valueWords = 0;
        for (FusionRecipe.Value value : recipe.getValues()) {
            valueWords =
                    Math.addExact(
                            valueWords,
                            Math.addExact(
                                    VALUE_RECORD_HEADER_WORDS,
                                    value.getSpec().getInnerShape().length));
        }
        int inputWords = Math.multiplyExact(recipe.getInputs().size(), BINDING_RECORD_WORDS);
        int constantWords = Math.multiplyExact(recipe.getConstants().size(), BINDING_RECORD_WORDS);
        int commandWords = 0;
        for (FusionRecipe.Value value : recipe.getValues()) {
            if (value instanceof FusionRecipe.OutputPack) {
                commandWords =
                        Math.addExact(
                                commandWords,
                                outputPackCommandWords((FusionRecipe.OutputPack) value));
            } else if (value instanceof FusionRecipe.AffineSum) {
                commandWords =
                        Math.addExact(
                                commandWords,
                                affineSumCommandWords((FusionRecipe.AffineSum) value));
            } else if (value instanceof FusionRecipe.IndexedAffine) {
                commandWords =
                        Math.addExact(
                                commandWords,
                                indexedAffineCommandWords((FusionRecipe.IndexedAffine) value));
            } else if (value instanceof FusionRecipe.TransformerEncoderStack) {
                commandWords =
                        Math.addExact(
                                commandWords,
                                transformerEncoderStackCommandWords(
                                        (FusionRecipe.TransformerEncoderStack) value));
            } else if (value instanceof FusionRecipe.IndexedLocalTransformerEncoder) {
                commandWords = Math.addExact(commandWords, indexedLocalTransformerCommandWords());
            } else if (value instanceof FusionRecipe.BinaryBranchBlend) {
                commandWords = Math.addExact(commandWords, binaryBranchBlendCommandWords());
            } else if (isFirstSingleQueryReadoutState(value)) {
                commandWords =
                        Math.addExact(
                                commandWords,
                                singleQueryReadoutGroupCommandWords(
                                        singleQueryReadoutGroup(value)));
            } else if (value instanceof FusionRecipe.SingleQueryCrossAttentionReadoutState) {
                // The first state encodes the shared multi-result command.
            } else if (!(value instanceof FusionRecipe.Input)
                    && !(value instanceof FusionRecipe.Constant)) {
                throw new UnsupportedOperationException(
                        "The PyTorch fusion backend does not support value type: "
                                + value.getClass().getSimpleName());
            }
        }
        int outputWords = Math.multiplyExact(recipe.getOutputs().size(), OUTPUT_RECORD_WORDS);

        int dimensionOffset = HEADER_WORDS;
        int valueOffset = Math.addExact(dimensionOffset, dimensionWords);
        int inputOffset = Math.addExact(valueOffset, valueWords);
        int constantOffset = Math.addExact(inputOffset, inputWords);
        int commandOffset = Math.addExact(constantOffset, constantWords);
        int outputOffset = Math.addExact(commandOffset, commandWords);
        int totalWords = Math.addExact(outputOffset, outputWords);

        ByteBuffer descriptor =
                ByteBuffer.allocateDirect(Math.multiplyExact(totalWords, Long.BYTES))
                        .order(ByteOrder.nativeOrder());
        descriptor.putLong(MAGIC);
        descriptor.putLong(VERSION);
        descriptor.putLong(totalWords);
        descriptor.putLong(0);
        descriptor.putLong(recipe.getDimensions().size());
        descriptor.putLong(recipe.getValues().size());
        descriptor.putLong(recipe.getInputs().size());
        descriptor.putLong(recipe.getConstants().size());
        descriptor.putLong(commandCount(recipe));
        descriptor.putLong(recipe.getOutputs().size());
        descriptor.putLong(dimensionOffset);
        descriptor.putLong(valueOffset);
        descriptor.putLong(inputOffset);
        descriptor.putLong(constantOffset);
        descriptor.putLong(commandOffset);
        descriptor.putLong(outputOffset);

        for (FusionRecipe.Dimension dimension : recipe.getDimensions()) {
            descriptor.putLong(DIMENSION_RECORD_WORDS);
            descriptor.putLong(DIMENSION_PREFIX_EXTENT);
            descriptor.putLong(dimension.getMaximumExtent());
        }
        for (FusionRecipe.Value value : recipe.getValues()) {
            putValue(descriptor, value);
        }
        for (FusionRecipe.Input input : recipe.getInputs()) {
            descriptor.putLong(input.getInputIndex());
            descriptor.putLong(input.getIndex());
        }
        for (FusionRecipe.Constant constant : recipe.getConstants()) {
            descriptor.putLong(constant.getConstantIndex());
            descriptor.putLong(constant.getIndex());
        }
        for (FusionRecipe.Value value : recipe.getValues()) {
            if (value instanceof FusionRecipe.OutputPack) {
                putOutputPackCommand(descriptor, (FusionRecipe.OutputPack) value);
            } else if (value instanceof FusionRecipe.AffineSum) {
                putAffineSumCommand(descriptor, (FusionRecipe.AffineSum) value);
            } else if (value instanceof FusionRecipe.IndexedAffine) {
                putIndexedAffineCommand(descriptor, (FusionRecipe.IndexedAffine) value);
            } else if (value instanceof FusionRecipe.TransformerEncoderStack) {
                putTransformerEncoderStackCommand(
                        descriptor, (FusionRecipe.TransformerEncoderStack) value);
            } else if (value instanceof FusionRecipe.IndexedLocalTransformerEncoder) {
                putIndexedLocalTransformerCommand(
                        descriptor, (FusionRecipe.IndexedLocalTransformerEncoder) value);
            } else if (value instanceof FusionRecipe.BinaryBranchBlend) {
                putBinaryBranchBlendCommand(descriptor, (FusionRecipe.BinaryBranchBlend) value);
            } else if (isFirstSingleQueryReadoutState(value)) {
                putSingleQueryReadoutGroupCommand(descriptor, singleQueryReadoutGroup(value));
            }
        }
        for (FusionRecipe.Output output : recipe.getOutputs()) {
            descriptor.putLong(output.getIndex());
            descriptor.putLong(output.getValue().getIndex());
        }
        if (descriptor.position() != descriptor.capacity()) {
            throw new AssertionError("Fusion descriptor size calculation is inconsistent.");
        }
        descriptor.flip();
        return descriptor;
    }

    static int commandCount(FusionRecipe recipe) {
        int count = 0;
        for (FusionRecipe.Value value : recipe.getValues()) {
            if (value instanceof FusionRecipe.OutputPack) {
                ++count;
            } else if (value instanceof FusionRecipe.AffineSum) {
                ++count;
            } else if (value instanceof FusionRecipe.IndexedAffine) {
                ++count;
            } else if (value instanceof FusionRecipe.TransformerEncoderStack) {
                ++count;
            } else if (value instanceof FusionRecipe.IndexedLocalTransformerEncoder) {
                ++count;
            } else if (value instanceof FusionRecipe.BinaryBranchBlend) {
                ++count;
            } else if (isFirstSingleQueryReadoutState(value)) {
                ++count;
            }
        }
        return count;
    }

    static long persistentStorageBytes(FusionRecipe recipe) {
        long bytes = 0;
        for (FusionRecipe.Value value : recipe.getValues()) {
            if (isComputed(value)) {
                bytes = Math.addExact(bytes, storageBytes(value));
            }
        }
        bytes = Math.addExact(bytes, affineWorkspaceBytes(recipe));
        bytes = Math.addExact(bytes, indexedAffineWorkspaceBytes(recipe));
        bytes = Math.addExact(bytes, transformerWorkspaceBytes(recipe));
        bytes = Math.addExact(bytes, indexedLocalTransformerWorkspaceBytes(recipe));
        bytes = Math.addExact(bytes, singleQueryReadoutWorkspaceBytes(recipe));
        return bytes;
    }

    static long workspaceBytes(FusionRecipe recipe) {
        Set<FusionRecipe.Value> outputs = new HashSet<>();
        for (FusionRecipe.Output output : recipe.getOutputs()) {
            outputs.add(output.getValue());
        }
        long bytes = 0;
        for (FusionRecipe.Value value : recipe.getValues()) {
            if (isComputed(value) && !outputs.contains(value)) {
                bytes = Math.addExact(bytes, storageBytes(value));
            }
        }
        bytes = Math.addExact(bytes, affineWorkspaceBytes(recipe));
        bytes = Math.addExact(bytes, indexedAffineWorkspaceBytes(recipe));
        bytes = Math.addExact(bytes, transformerWorkspaceBytes(recipe));
        bytes = Math.addExact(bytes, indexedLocalTransformerWorkspaceBytes(recipe));
        bytes = Math.addExact(bytes, singleQueryReadoutWorkspaceBytes(recipe));
        return bytes;
    }

    static long executableStorageBytes(FusionRecipe recipe) {
        long bytes = 0;
        for (FusionRecipe.Value value : recipe.getValues()) {
            if (!(value instanceof FusionRecipe.AffineSum)) {
                continue;
            }
            FusionRecipe.AffineSum affineSum = (FusionRecipe.AffineSum) value;
            long outputWidth =
                    affineSum.getSpec()
                            .getInnerShape()[affineSum.getSpec().getInnerShape().length - 1];
            for (AffineGroup group : affineGroups(affineSum)) {
                if (group.precomputeAtBind) {
                    bytes =
                            Math.addExact(
                                    bytes,
                                    affineStorageBytes(
                                            affineSum, affineRows(affineSum, group), outputWidth));
                } else {
                    bytes =
                            Math.addExact(
                                    bytes,
                                    affineStorageBytes(affineSum, group.inputWidth, outputWidth));
                }
            }
        }
        for (FusionRecipe.Value value : recipe.getValues()) {
            if (!(value instanceof FusionRecipe.IndexedAffine)) {
                continue;
            }
            FusionRecipe.IndexedAffine indexed = (FusionRecipe.IndexedAffine) value;
            bytes = Math.addExact(bytes, storageBytes(indexed.getHiddenWeight()));
        }
        for (FusionRecipe.Value value : recipe.getValues()) {
            if (!(value instanceof FusionRecipe.TransformerEncoderStack)) {
                continue;
            }
            FusionRecipe.TransformerEncoderStack stack =
                    (FusionRecipe.TransformerEncoderStack) value;
            int elementBytes = stack.getSpec().getDataType().getNumOfBytes();
            long blockElements = 0;
            for (FusionRecipe.TransformerEncoderBlock block : stack.getBlocks()) {
                blockElements =
                        Math.addExact(
                                blockElements, storageElements(block.getQueryKeyValueWeight()));
                blockElements =
                        Math.addExact(
                                blockElements, storageElements(block.getAttentionOutputWeight()));
                blockElements =
                        Math.addExact(
                                blockElements,
                                storageElements(block.getFeedForwardExpansionWeight()));
                blockElements =
                        Math.addExact(
                                blockElements,
                                storageElements(block.getFeedForwardProjectionWeight()));
            }
            bytes = Math.addExact(bytes, Math.multiplyExact(blockElements, elementBytes));
        }
        for (FusionRecipe.Value value : recipe.getValues()) {
            if (!(value instanceof FusionRecipe.IndexedLocalTransformerEncoder)) {
                continue;
            }
            FusionRecipe.IndexedLocalTransformerEncoder encoder =
                    (FusionRecipe.IndexedLocalTransformerEncoder) value;
            FusionRecipe.TransformerEncoderBlock block = encoder.getBlock();
            long elements = storageElements(block.getQueryKeyValueWeight());
            elements = Math.addExact(elements, storageElements(block.getAttentionOutputWeight()));
            elements =
                    Math.addExact(elements, storageElements(block.getFeedForwardExpansionWeight()));
            elements =
                    Math.addExact(
                            elements, storageElements(block.getFeedForwardProjectionWeight()));
            bytes =
                    Math.addExact(
                            bytes,
                            Math.multiplyExact(
                                    elements, encoder.getSpec().getDataType().getNumOfBytes()));
        }
        for (FusionRecipe.Value value : recipe.getValues()) {
            if (!isFirstSingleQueryReadoutState(value)) {
                continue;
            }
            FusionRecipe.SingleQueryCrossAttentionReadoutGroup group =
                    singleQueryReadoutGroup(value);
            int elementBytes = value.getSpec().getDataType().getNumOfBytes();
            long hiddenWidth = group.getMemory().getSpec().getInnerShape()[1];
            long readoutCount = group.getReadouts().size();
            long projectionElements = 0;
            projectionElements =
                    Math.addExact(
                            projectionElements,
                            Math.multiplyExact(readoutCount, 2L * hiddenWidth * hiddenWidth));
            projectionElements =
                    Math.addExact(
                            projectionElements,
                            Math.multiplyExact(
                                    readoutCount, (long) group.getAttentionWidth() * hiddenWidth));
            projectionElements =
                    Math.addExact(
                            projectionElements,
                            Math.multiplyExact(
                                    readoutCount, 2L * group.getAttentionWidth() * hiddenWidth));
            projectionElements =
                    Math.addExact(
                            projectionElements,
                            Math.multiplyExact(
                                    readoutCount, (long) group.getAttentionWidth() * hiddenWidth));
            projectionElements =
                    Math.addExact(
                            projectionElements,
                            Math.multiplyExact(
                                    readoutCount,
                                    2L * hiddenWidth * group.getMaximumFeedForwardWidth()));
            long vectorElements =
                    Math.multiplyExact(
                            readoutCount,
                            3L * hiddenWidth
                                    + group.getAttentionWidth()
                                    + group.getMaximumFeedForwardWidth());
            bytes =
                    Math.addExact(
                            bytes,
                            Math.multiplyExact(
                                    Math.addExact(projectionElements, vectorElements),
                                    elementBytes));
            DataType normDataType =
                    group.getReadouts().get(0).getQueryNormWeight().getSpec().getDataType();
            bytes =
                    Math.addExact(
                            bytes,
                            Math.multiplyExact(
                                    Math.multiplyExact(readoutCount, 6L * hiddenWidth),
                                    normDataType.getNumOfBytes()));
        }
        return bytes;
    }

    private static boolean isComputed(FusionRecipe.Value value) {
        return value instanceof FusionRecipe.OutputPack
                || value instanceof FusionRecipe.AffineSum
                || value instanceof FusionRecipe.IndexedAffine
                || value instanceof FusionRecipe.TransformerEncoderStack
                || value instanceof FusionRecipe.IndexedLocalTransformerEncoder
                || value instanceof FusionRecipe.BinaryBranchBlend
                || value instanceof FusionRecipe.SingleQueryCrossAttentionReadoutState;
    }

    private static long singleQueryReadoutWorkspaceBytes(FusionRecipe recipe) {
        long bytes = 0;
        for (FusionRecipe.Value value : recipe.getValues()) {
            if (!isFirstSingleQueryReadoutState(value)) {
                continue;
            }
            FusionRecipe.SingleQueryCrossAttentionReadoutGroup group =
                    singleQueryReadoutGroup(value);
            long maximumBatch = value.getSpec().getLeadingDimension().getMaximumExtent();
            long hiddenWidth = group.getMemory().getSpec().getInnerShape()[1];
            long readoutCount = group.getReadouts().size();
            long regionElements =
                    Math.max(
                            Math.multiplyExact(maximumBatch, 2L * hiddenWidth),
                            Math.multiplyExact(
                                    Math.multiplyExact(readoutCount, maximumBatch), hiddenWidth));
            long stateElements =
                    Math.multiplyExact(Math.multiplyExact(readoutCount, maximumBatch), hiddenWidth);
            long tailElements =
                    Math.multiplyExact(
                            Math.multiplyExact(readoutCount, maximumBatch),
                            Math.max(hiddenWidth, group.getMaximumFeedForwardWidth()));
            long elements =
                    Math.addExact(regionElements, Math.addExact(stateElements, tailElements));
            bytes =
                    Math.addExact(
                            bytes,
                            Math.multiplyExact(
                                    elements, value.getSpec().getDataType().getNumOfBytes()));
        }
        return bytes;
    }

    private static boolean isFirstSingleQueryReadoutState(FusionRecipe.Value value) {
        return value instanceof FusionRecipe.SingleQueryCrossAttentionReadoutState
                && ((FusionRecipe.SingleQueryCrossAttentionReadoutState) value).getReadoutIndex()
                        == 0;
    }

    private static FusionRecipe.SingleQueryCrossAttentionReadoutGroup singleQueryReadoutGroup(
            FusionRecipe.Value value) {
        return ((FusionRecipe.SingleQueryCrossAttentionReadoutState) value).getGroup();
    }

    private static long transformerWorkspaceBytes(FusionRecipe recipe) {
        long bytes = 0;
        for (FusionRecipe.Value value : recipe.getValues()) {
            if (!(value instanceof FusionRecipe.TransformerEncoderStack)) {
                continue;
            }
            FusionRecipe.TransformerEncoderStack stack =
                    (FusionRecipe.TransformerEncoderStack) value;
            long[] shape = stack.getSpec().getMaximumShape().getShape();
            long batch = shape[0];
            long tokens = shape[1];
            long hidden = shape[2];
            long rows = Math.multiplyExact(batch, tokens);
            long elements = Math.multiplyExact(rows, hidden); // normalized
            elements =
                    Math.addExact(
                            elements,
                            Math.multiplyExact(
                                    rows, Math.multiplyExact(3L, stack.getAttentionWidth())));
            elements =
                    Math.addExact(elements, Math.multiplyExact(rows, stack.getFeedForwardWidth()));
            bytes =
                    Math.addExact(
                            bytes,
                            Math.multiplyExact(
                                    elements, stack.getSpec().getDataType().getNumOfBytes()));
        }
        return bytes;
    }

    private static long indexedLocalTransformerWorkspaceBytes(FusionRecipe recipe) {
        long bytes = 0;
        for (FusionRecipe.Value value : recipe.getValues()) {
            if (!(value instanceof FusionRecipe.IndexedLocalTransformerEncoder)) {
                continue;
            }
            FusionRecipe.IndexedLocalTransformerEncoder encoder =
                    (FusionRecipe.IndexedLocalTransformerEncoder) value;
            long activeRows =
                    encoder.getIndices().getSpec().getLeadingDimension().getMaximumExtent();
            long hiddenWidth = encoder.getSpec().getInnerShape()[2];
            long tileRows = Math.min(activeRows, INDEXED_LOCAL_TRANSFORMER_TILE_ROWS);
            long elements = Math.multiplyExact(activeRows, 3L * encoder.getAttentionWidth());
            elements = Math.addExact(elements, Math.multiplyExact(tileRows, hiddenWidth));
            bytes =
                    Math.addExact(
                            bytes,
                            Math.multiplyExact(
                                    elements, encoder.getSpec().getDataType().getNumOfBytes()));
        }
        return bytes;
    }

    private static long indexedAffineWorkspaceBytes(FusionRecipe recipe) {
        long bytes = 0;
        for (FusionRecipe.Value value : recipe.getValues()) {
            if (!(value instanceof FusionRecipe.IndexedAffine)) {
                continue;
            }
            FusionRecipe.IndexedAffine indexed = (FusionRecipe.IndexedAffine) value;
            long activeRows =
                    indexed.getIndices().getSpec().getLeadingDimension().getMaximumExtent();
            long inputWidth = 0;
            for (FusionRecipe.IndexedAffineSource source : indexed.getSources()) {
                inputWidth =
                        Math.addExact(inputWidth, source.getInput().getSpec().getInnerShape()[0]);
            }
            long hiddenWidth = indexed.getHiddenWeight().getSpec().getInnerShape()[0];
            int elementBytes = indexed.getSpec().getDataType().getNumOfBytes();
            bytes =
                    Math.addExact(
                            bytes,
                            Math.multiplyExact(
                                    Math.multiplyExact(activeRows, inputWidth), elementBytes));
            bytes =
                    Math.addExact(
                            bytes,
                            Math.multiplyExact(
                                    Math.multiplyExact(activeRows, hiddenWidth), elementBytes));
        }
        return bytes;
    }

    private static long affineWorkspaceBytes(FusionRecipe recipe) {
        long bytes = 0;
        for (FusionRecipe.Value value : recipe.getValues()) {
            if (!(value instanceof FusionRecipe.AffineSum)) {
                continue;
            }
            FusionRecipe.AffineSum affineSum = (FusionRecipe.AffineSum) value;
            long[] outputInner = affineSum.getSpec().getInnerShape();
            long[] outputPrefix = Arrays.copyOf(outputInner, outputInner.length - 1);
            boolean directGroupAvailable = false;
            for (AffineGroup group : affineGroups(affineSum)) {
                if (group.precomputeAtBind) {
                    continue;
                }
                long rowCount = affineRows(affineSum, group);
                if (group.requiresInputPack) {
                    bytes =
                            Math.addExact(
                                    bytes,
                                    affineStorageBytes(affineSum, rowCount, group.inputWidth));
                }
                if (!directGroupAvailable
                        && group.dynamicLeading
                        && Arrays.equals(group.prefix, outputPrefix)) {
                    directGroupAvailable = true;
                } else {
                    bytes =
                            Math.addExact(
                                    bytes,
                                    affineStorageBytes(
                                            affineSum,
                                            rowCount,
                                            outputInner[outputInner.length - 1]));
                }
            }
        }
        return bytes;
    }

    private static ArrayList<AffineGroup> affineGroups(FusionRecipe.AffineSum affineSum) {
        ArrayList<AffineGroup> groups = new ArrayList<>();
        for (FusionRecipe.AffineTerm term : affineSum.getTerms()) {
            FusionRecipe.TensorSpec inputSpec = term.getInput().getSpec();
            long[] inputInner = inputSpec.getInnerShape();
            boolean dynamicLeading = inputSpec.getLeadingDimension() != null;
            boolean constantInput = term.getInput() instanceof FusionRecipe.Constant;
            boolean precomputeAtBind = !dynamicLeading && constantInput;
            long[] prefix =
                    Arrays.copyOfRange(inputInner, dynamicLeading ? 0 : 1, inputInner.length - 1);
            AffineGroup group = findGroup(groups, prefix, dynamicLeading, precomputeAtBind);
            if (group == null) {
                group = new AffineGroup(prefix, dynamicLeading, precomputeAtBind);
                groups.add(group);
            }
            if (group.termCount != 0
                    || inputSpec.getDataType() != affineSum.getSpec().getDataType()) {
                group.requiresInputPack = true;
            }
            group.inputWidth = Math.addExact(group.inputWidth, inputInner[inputInner.length - 1]);
            ++group.termCount;
        }
        return groups;
    }

    private static AffineGroup findGroup(
            ArrayList<AffineGroup> groups,
            long[] prefix,
            boolean dynamicLeading,
            boolean precomputeAtBind) {
        for (AffineGroup group : groups) {
            if (group.dynamicLeading == dynamicLeading
                    && group.precomputeAtBind == precomputeAtBind
                    && Arrays.equals(group.prefix, prefix)) {
                return group;
            }
        }
        return null;
    }

    private static long affineRows(FusionRecipe.AffineSum affineSum, AffineGroup group) {
        long rows =
                group.dynamicLeading
                        ? affineSum.getSpec().getLeadingDimension().getMaximumExtent()
                        : 1;
        for (long extent : group.prefix) {
            rows = Math.multiplyExact(rows, extent);
        }
        return rows;
    }

    private static long affineStorageBytes(
            FusionRecipe.AffineSum affineSum, long rows, long width) {
        long elements = Math.multiplyExact(rows, width);
        return Math.multiplyExact(elements, affineSum.getSpec().getDataType().getNumOfBytes());
    }

    private static long storageBytes(FusionRecipe.Value value) {
        long elements = storageElements(value);
        return Math.multiplyExact(elements, value.getSpec().getDataType().getNumOfBytes());
    }

    private static long storageElements(FusionRecipe.Value value) {
        long elements = 1;
        for (long extent : value.getSpec().getMaximumShape().getShape()) {
            elements = Math.multiplyExact(elements, extent);
        }
        return elements;
    }

    static long dtypeCode(DataType dataType) {
        switch (dataType) {
            case FLOAT16:
                return DTYPE_FLOAT16;
            case BFLOAT16:
                return DTYPE_BFLOAT16;
            case FLOAT32:
                return DTYPE_FLOAT32;
            case BOOLEAN:
                return DTYPE_BOOLEAN;
            case UINT8:
                return DTYPE_UINT8;
            case INT8:
                return DTYPE_INT8;
            case INT16:
                return DTYPE_INT16;
            case INT32:
                return DTYPE_INT32;
            case INT64:
                return DTYPE_INT64;
            case FLOAT64:
                return DTYPE_FLOAT64;
            default:
                throw new UnsupportedOperationException(
                        "PyTorch fusion does not support data type: " + dataType);
        }
    }

    private static void putValue(ByteBuffer descriptor, FusionRecipe.Value value) {
        FusionRecipe.TensorSpec spec = value.getSpec();
        long[] innerShape = spec.getInnerShape();
        descriptor.putLong(Math.addExact(VALUE_RECORD_HEADER_WORDS, innerShape.length));
        descriptor.putLong(value.getIndex());
        descriptor.putLong(dtypeCode(spec.getDataType()));
        descriptor.putLong(LAYOUT_CONTIGUOUS);
        FusionRecipe.Dimension dimension = spec.getLeadingDimension();
        descriptor.putLong(dimension == null ? -1 : dimension.getIndex());
        descriptor.putLong(innerShape.length);
        descriptor.putLong(0);
        for (long extent : innerShape) {
            descriptor.putLong(extent);
        }
    }

    private static int outputPackCommandWords(FusionRecipe.OutputPack outputPack) {
        return Math.addExact(COMMAND_RECORD_HEADER_WORDS + 1, outputPack.getSources().size());
    }

    private static void putOutputPackCommand(
            ByteBuffer descriptor, FusionRecipe.OutputPack outputPack) {
        descriptor.putLong(outputPackCommandWords(outputPack));
        descriptor.putLong(OUTPUT_PACK_V1);
        descriptor.putLong(0);
        descriptor.putLong(1);
        descriptor.putLong(outputPack.getSources().size());
        descriptor.putLong(0);
        descriptor.putLong(outputPack.getIndex());
        for (FusionRecipe.Value source : outputPack.getSources()) {
            descriptor.putLong(source.getIndex());
        }
    }

    private static int affineSumCommandWords(FusionRecipe.AffineSum affineSum) {
        int operandCount = Math.multiplyExact(affineSum.getTerms().size(), 2);
        if (affineSum.getBias() != null) {
            operandCount = Math.addExact(operandCount, 1);
        }
        return Math.addExact(
                Math.addExact(COMMAND_RECORD_HEADER_WORDS + 1, operandCount),
                Math.multiplyExact(3, SCALAR_ATTRIBUTE_WORDS));
    }

    private static void putAffineSumCommand(
            ByteBuffer descriptor, FusionRecipe.AffineSum affineSum) {
        int operandCount = Math.multiplyExact(affineSum.getTerms().size(), 2);
        if (affineSum.getBias() != null) {
            ++operandCount;
        }
        descriptor.putLong(affineSumCommandWords(affineSum));
        descriptor.putLong(AFFINE_SUM_V1);
        descriptor.putLong(0);
        descriptor.putLong(1);
        descriptor.putLong(operandCount);
        descriptor.putLong(3);
        descriptor.putLong(affineSum.getIndex());
        for (FusionRecipe.AffineTerm term : affineSum.getTerms()) {
            descriptor.putLong(term.getInput().getIndex());
            descriptor.putLong(term.getWeight().getIndex());
        }
        if (affineSum.getBias() != null) {
            descriptor.putLong(affineSum.getBias().getIndex());
        }
        putScalarAttribute(descriptor, AFFINE_TERM_COUNT, affineSum.getTerms().size());
        putScalarAttribute(
                descriptor, AFFINE_ACTIVATION, activationCode(affineSum.getActivation()));
        putScalarAttribute(descriptor, AFFINE_HAS_BIAS, affineSum.getBias() == null ? 0 : 1);
    }

    private static int indexedAffineCommandWords(FusionRecipe.IndexedAffine indexed) {
        int operandCount = Math.addExact(indexed.getSources().size(), 3);
        if (indexed.getHiddenBias() != null) {
            ++operandCount;
        }
        if (indexed.getOutputBias() != null) {
            ++operandCount;
        }
        int divisorWords = Math.addExact(4, indexed.getSources().size());
        return Math.addExact(
                Math.addExact(COMMAND_RECORD_HEADER_WORDS + 1, operandCount),
                Math.addExact(Math.multiplyExact(4, SCALAR_ATTRIBUTE_WORDS), divisorWords));
    }

    private static void putIndexedAffineCommand(
            ByteBuffer descriptor, FusionRecipe.IndexedAffine indexed) {
        int operandCount = Math.addExact(indexed.getSources().size(), 3);
        if (indexed.getHiddenBias() != null) {
            ++operandCount;
        }
        if (indexed.getOutputBias() != null) {
            ++operandCount;
        }
        descriptor.putLong(indexedAffineCommandWords(indexed));
        descriptor.putLong(INDEXED_AFFINE_V1);
        descriptor.putLong(0);
        descriptor.putLong(1);
        descriptor.putLong(operandCount);
        descriptor.putLong(5);
        descriptor.putLong(indexed.getIndex());
        descriptor.putLong(indexed.getIndices().getIndex());
        for (FusionRecipe.IndexedAffineSource source : indexed.getSources()) {
            descriptor.putLong(source.getInput().getIndex());
        }
        descriptor.putLong(indexed.getHiddenWeight().getIndex());
        if (indexed.getHiddenBias() != null) {
            descriptor.putLong(indexed.getHiddenBias().getIndex());
        }
        descriptor.putLong(indexed.getOutputWeight().getIndex());
        if (indexed.getOutputBias() != null) {
            descriptor.putLong(indexed.getOutputBias().getIndex());
        }
        putScalarAttribute(descriptor, INDEXED_SOURCE_COUNT, indexed.getSources().size());
        putScalarAttribute(descriptor, INDEXED_ACTIVATION, activationCode(indexed.getActivation()));
        putScalarAttribute(
                descriptor, INDEXED_HAS_HIDDEN_BIAS, indexed.getHiddenBias() == null ? 0 : 1);
        putScalarAttribute(
                descriptor, INDEXED_HAS_OUTPUT_BIAS, indexed.getOutputBias() == null ? 0 : 1);
        descriptor.putLong(Math.addExact(4, indexed.getSources().size()));
        descriptor.putLong(INDEXED_SOURCE_DIVISORS);
        descriptor.putLong(ATTRIBUTE_INT64);
        descriptor.putLong(indexed.getSources().size());
        for (FusionRecipe.IndexedAffineSource source : indexed.getSources()) {
            descriptor.putLong(source.getIndexDivisor());
        }
    }

    private static int transformerEncoderStackCommandWords(
            FusionRecipe.TransformerEncoderStack stack) {
        int operandCount = Math.addExact(1, Math.multiplyExact(13, stack.getBlocks().size()));
        return Math.addExact(
                Math.addExact(COMMAND_RECORD_HEADER_WORDS + 1, operandCount),
                Math.multiplyExact(5, SCALAR_ATTRIBUTE_WORDS));
    }

    private static void putTransformerEncoderStackCommand(
            ByteBuffer descriptor, FusionRecipe.TransformerEncoderStack stack) {
        int operandCount = Math.addExact(1, Math.multiplyExact(13, stack.getBlocks().size()));
        descriptor.putLong(transformerEncoderStackCommandWords(stack));
        descriptor.putLong(TRANSFORMER_ENCODER_STACK_V1);
        descriptor.putLong(0);
        descriptor.putLong(1);
        descriptor.putLong(operandCount);
        descriptor.putLong(5);
        descriptor.putLong(stack.getIndex());
        descriptor.putLong(stack.getInput().getIndex());
        for (FusionRecipe.TransformerEncoderBlock block : stack.getBlocks()) {
            descriptor.putLong(block.getAttentionInputWeight().getIndex());
            descriptor.putLong(block.getAttentionInputBias().getIndex());
            descriptor.putLong(block.getQueryKeyValueWeight().getIndex());
            descriptor.putLong(block.getAttentionOutputWeight().getIndex());
            descriptor.putLong(block.getAttentionOutputBias().getIndex());
            descriptor.putLong(block.getFeedForwardInputWeight().getIndex());
            descriptor.putLong(block.getFeedForwardInputBias().getIndex());
            descriptor.putLong(block.getFeedForwardExpansionWeight().getIndex());
            descriptor.putLong(block.getFeedForwardExpansionBias().getIndex());
            descriptor.putLong(block.getFeedForwardProjectionWeight().getIndex());
            descriptor.putLong(block.getFeedForwardProjectionBias().getIndex());
            descriptor.putLong(block.getOutputWeight().getIndex());
            descriptor.putLong(block.getOutputBias().getIndex());
        }
        putScalarAttribute(descriptor, TRANSFORMER_BLOCK_COUNT, stack.getBlocks().size());
        putScalarAttribute(descriptor, TRANSFORMER_ATTENTION_HEADS, stack.getAttentionHeads());
        putScalarAttribute(descriptor, TRANSFORMER_ATTENTION_WIDTH, stack.getAttentionWidth());
        putScalarAttribute(descriptor, TRANSFORMER_FEED_FORWARD_WIDTH, stack.getFeedForwardWidth());
        putFloatAttribute(descriptor, TRANSFORMER_EPSILON, stack.getEpsilon());
    }

    private static int indexedLocalTransformerCommandWords() {
        int operandCount = 17;
        return Math.addExact(
                COMMAND_RECORD_HEADER_WORDS + 1 + operandCount,
                Math.multiplyExact(4, SCALAR_ATTRIBUTE_WORDS));
    }

    private static void putIndexedLocalTransformerCommand(
            ByteBuffer descriptor, FusionRecipe.IndexedLocalTransformerEncoder encoder) {
        FusionRecipe.TransformerEncoderBlock block = encoder.getBlock();
        descriptor.putLong(indexedLocalTransformerCommandWords());
        descriptor.putLong(INDEXED_LOCAL_TRANSFORMER_ENCODER_V1);
        descriptor.putLong(0);
        descriptor.putLong(1);
        descriptor.putLong(17);
        descriptor.putLong(4);
        descriptor.putLong(encoder.getIndex());
        descriptor.putLong(encoder.getInput().getIndex());
        descriptor.putLong(encoder.getIndices().getIndex());
        descriptor.putLong(encoder.getInputNormWeight().getIndex());
        descriptor.putLong(encoder.getInputNormBias().getIndex());
        descriptor.putLong(block.getAttentionInputWeight().getIndex());
        descriptor.putLong(block.getAttentionInputBias().getIndex());
        descriptor.putLong(block.getQueryKeyValueWeight().getIndex());
        descriptor.putLong(block.getAttentionOutputWeight().getIndex());
        descriptor.putLong(block.getAttentionOutputBias().getIndex());
        descriptor.putLong(block.getFeedForwardInputWeight().getIndex());
        descriptor.putLong(block.getFeedForwardInputBias().getIndex());
        descriptor.putLong(block.getFeedForwardExpansionWeight().getIndex());
        descriptor.putLong(block.getFeedForwardExpansionBias().getIndex());
        descriptor.putLong(block.getFeedForwardProjectionWeight().getIndex());
        descriptor.putLong(block.getFeedForwardProjectionBias().getIndex());
        descriptor.putLong(block.getOutputWeight().getIndex());
        descriptor.putLong(block.getOutputBias().getIndex());
        putScalarAttribute(
                descriptor, LOCAL_TRANSFORMER_ATTENTION_HEADS, encoder.getAttentionHeads());
        putScalarAttribute(
                descriptor, LOCAL_TRANSFORMER_ATTENTION_WIDTH, encoder.getAttentionWidth());
        putScalarAttribute(
                descriptor, LOCAL_TRANSFORMER_FEED_FORWARD_WIDTH, encoder.getFeedForwardWidth());
        putFloatAttribute(descriptor, LOCAL_TRANSFORMER_EPSILON, encoder.getEpsilon());
    }

    private static int binaryBranchBlendCommandWords() {
        return COMMAND_RECORD_HEADER_WORDS + 1 + 5;
    }

    private static void putBinaryBranchBlendCommand(
            ByteBuffer descriptor, FusionRecipe.BinaryBranchBlend blend) {
        descriptor.putLong(binaryBranchBlendCommandWords());
        descriptor.putLong(BINARY_BRANCH_BLEND_V1);
        descriptor.putLong(0);
        descriptor.putLong(1);
        descriptor.putLong(5);
        descriptor.putLong(0);
        descriptor.putLong(blend.getIndex());
        descriptor.putLong(blend.getBaselineContext().getIndex());
        descriptor.putLong(blend.getSelectedContext().getIndex());
        descriptor.putLong(blend.getSelectedLogit().getIndex());
        descriptor.putLong(blend.getBaselinePresence().getIndex());
        descriptor.putLong(blend.getSelectedPresence().getIndex());
    }

    private static int singleQueryReadoutGroupCommandWords(
            FusionRecipe.SingleQueryCrossAttentionReadoutGroup group) {
        int operandCount = Math.addExact(3, Math.multiplyExact(17, group.getReadouts().size()));
        int feedForwardWidthAttributeWords = Math.addExact(4, group.getReadouts().size());
        return Math.addExact(
                Math.addExact(
                        COMMAND_RECORD_HEADER_WORDS + group.getReadoutStates().size(),
                        operandCount),
                Math.addExact(
                        Math.multiplyExact(6, SCALAR_ATTRIBUTE_WORDS),
                        feedForwardWidthAttributeWords));
    }

    private static void putSingleQueryReadoutGroupCommand(
            ByteBuffer descriptor, FusionRecipe.SingleQueryCrossAttentionReadoutGroup group) {
        int operandCount = Math.addExact(3, Math.multiplyExact(17, group.getReadouts().size()));
        descriptor.putLong(singleQueryReadoutGroupCommandWords(group));
        descriptor.putLong(SINGLE_QUERY_CROSS_ATTENTION_READOUT_GROUP_V1);
        descriptor.putLong(0);
        descriptor.putLong(group.getReadoutStates().size());
        descriptor.putLong(operandCount);
        descriptor.putLong(7);
        for (FusionRecipe.SingleQueryCrossAttentionReadoutState state : group.getReadoutStates()) {
            descriptor.putLong(state.getIndex());
        }
        descriptor.putLong(group.getMemory().getIndex());
        descriptor.putLong(group.getQuerySource().getIndex());
        descriptor.putLong(group.getValidMask().getIndex());
        for (FusionRecipe.SingleQueryCrossAttentionReadout readout : group.getReadouts()) {
            descriptor.putLong(readout.getQuerySeedWeight().getIndex());
            descriptor.putLong(readout.getQuerySeedBias().getIndex());
            descriptor.putLong(readout.getQueryWeight().getIndex());
            descriptor.putLong(readout.getQueryBias().getIndex());
            descriptor.putLong(readout.getKeyValueWeight().getIndex());
            descriptor.putLong(readout.getContextWeight().getIndex());
            descriptor.putLong(readout.getContextBias().getIndex());
            descriptor.putLong(readout.getQueryNormWeight().getIndex());
            descriptor.putLong(readout.getQueryNormBias().getIndex());
            descriptor.putLong(readout.getFeedForwardNormWeight().getIndex());
            descriptor.putLong(readout.getFeedForwardNormBias().getIndex());
            descriptor.putLong(readout.getFeedForwardExpansionWeight().getIndex());
            descriptor.putLong(readout.getFeedForwardExpansionBias().getIndex());
            descriptor.putLong(readout.getFeedForwardProjectionWeight().getIndex());
            descriptor.putLong(readout.getFeedForwardProjectionBias().getIndex());
            descriptor.putLong(readout.getOutputNormWeight().getIndex());
            descriptor.putLong(readout.getOutputNormBias().getIndex());
        }
        putScalarAttribute(descriptor, READOUT_COUNT, group.getReadouts().size());
        putScalarAttribute(descriptor, READOUT_ATTENTION_HEADS, group.getAttentionHeads());
        putScalarAttribute(descriptor, READOUT_ATTENTION_WIDTH, group.getAttentionWidth());
        putScalarAttribute(
                descriptor, READOUT_MAXIMUM_FEED_FORWARD_WIDTH, group.getMaximumFeedForwardWidth());
        putFloatAttribute(descriptor, READOUT_EPSILON, group.getEpsilon());
        putScalarAttribute(descriptor, READOUT_QUERY_INDEX, group.getQueryIndex());
        descriptor.putLong(Math.addExact(4, group.getReadouts().size()));
        descriptor.putLong(READOUT_FEED_FORWARD_WIDTHS);
        descriptor.putLong(ATTRIBUTE_INT64);
        descriptor.putLong(group.getReadouts().size());
        for (FusionRecipe.SingleQueryCrossAttentionReadout readout : group.getReadouts()) {
            descriptor.putLong(readout.getFeedForwardWidth());
        }
    }

    private static void putScalarAttribute(ByteBuffer descriptor, long key, long value) {
        descriptor.putLong(SCALAR_ATTRIBUTE_WORDS);
        descriptor.putLong(key);
        descriptor.putLong(ATTRIBUTE_INT64);
        descriptor.putLong(1);
        descriptor.putLong(value);
    }

    private static void putFloatAttribute(ByteBuffer descriptor, long key, double value) {
        descriptor.putLong(SCALAR_ATTRIBUTE_WORDS);
        descriptor.putLong(key);
        descriptor.putLong(ATTRIBUTE_FLOAT64_BITS);
        descriptor.putLong(1);
        descriptor.putLong(Double.doubleToRawLongBits(value));
    }

    private static long activationCode(FusionRecipe.Activation activation) {
        switch (activation) {
            case NONE:
                return ACTIVATION_NONE;
            case SILU:
                return ACTIVATION_SILU;
            default:
                throw new UnsupportedOperationException(
                        "PyTorch fusion does not support activation: " + activation);
        }
    }

    private static final class AffineGroup {

        private final long[] prefix;
        private final boolean dynamicLeading;
        private final boolean precomputeAtBind;
        private long inputWidth;
        private int termCount;
        private boolean requiresInputPack;

        private AffineGroup(long[] prefix, boolean dynamicLeading, boolean precomputeAtBind) {
            this.prefix = prefix;
            this.dynamicLeading = dynamicLeading;
            this.precomputeAtBind = precomputeAtBind;
        }
    }
}
