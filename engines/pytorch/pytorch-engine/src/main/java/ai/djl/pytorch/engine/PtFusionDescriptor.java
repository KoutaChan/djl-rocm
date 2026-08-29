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
    static final long DIMENSION_PREFIX_EXTENT = 1;
    static final long LAYOUT_CONTIGUOUS = 1;

    static final int HEADER_WORDS = 16;
    static final int DIMENSION_RECORD_WORDS = 3;
    static final int VALUE_RECORD_HEADER_WORDS = 7;
    static final int BINDING_RECORD_WORDS = 2;
    static final int COMMAND_RECORD_HEADER_WORDS = 6;
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
            }
        }
        return count;
    }

    static long persistentStorageBytes(FusionRecipe recipe) {
        long bytes = 0;
        for (FusionRecipe.Value value : recipe.getValues()) {
            if (value instanceof FusionRecipe.OutputPack) {
                bytes = Math.addExact(bytes, storageBytes(value));
            }
        }
        return bytes;
    }

    static long workspaceBytes(FusionRecipe recipe) {
        Set<FusionRecipe.Value> outputs = new HashSet<>();
        for (FusionRecipe.Output output : recipe.getOutputs()) {
            outputs.add(output.getValue());
        }
        long bytes = 0;
        for (FusionRecipe.Value value : recipe.getValues()) {
            if (value instanceof FusionRecipe.OutputPack && !outputs.contains(value)) {
                bytes = Math.addExact(bytes, storageBytes(value));
            }
        }
        return bytes;
    }

    private static long storageBytes(FusionRecipe.Value value) {
        long elements = 1;
        for (long extent : value.getSpec().getMaximumShape().getShape()) {
            elements = Math.multiplyExact(elements, extent);
        }
        return Math.multiplyExact(elements, value.getSpec().getDataType().getNumOfBytes());
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
}
