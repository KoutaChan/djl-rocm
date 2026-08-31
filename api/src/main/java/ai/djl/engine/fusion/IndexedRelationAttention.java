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

import ai.djl.engine.fusion.FusionRecipe.Constant;
import ai.djl.engine.fusion.FusionRecipe.TensorSpec;
import ai.djl.ndarray.types.DataType;

/**
 * Describes fixed indexed relation parameters for one transformer attention block.
 *
 * <p>Relation IDs are zero-based {@code INT16 [tokens, tokens]}; zero is a normal relation ID, not
 * padding. Keys use {@code [relations, attentionWidth]} and scalar biases use {@code [relations,
 * attentionHeads]}.
 *
 * <p>A backend may snapshot and repack these constants while binding an executable. Callers must
 * therefore treat bound constants as immutable, although ownership remains with the caller.
 */
public final class IndexedRelationAttention {

    private final Constant relationIds;
    private final Constant relationKeys;
    private final Constant relationBias;

    IndexedRelationAttention(Constant relationIds, Constant relationKeys, Constant relationBias) {
        this.relationIds = relationIds;
        this.relationKeys = relationKeys;
        this.relationBias = relationBias;
    }

    /** Returns the zero-based {@code INT16 [tokens, tokens]} relation IDs. */
    public Constant getRelationIds() {
        return relationIds;
    }

    /** Returns the {@code [relations, attentionWidth]} relation keys. */
    public Constant getRelationKeys() {
        return relationKeys;
    }

    /** Returns the {@code [relations, attentionHeads]} relation bias. */
    public Constant getRelationBias() {
        return relationBias;
    }

    void validate(
            Constant sharedRelationIds,
            long tokenCount,
            long attentionHeads,
            long attentionWidth,
            DataType dataType) {
        if (relationIds != sharedRelationIds) {
            throw new IllegalArgumentException(
                    "Every transformer block must share one relation-ID constant.");
        }
        TensorSpec ids = relationIds.getSpec();
        if (ids.getLeadingDimension() != null
                || ids.getDataType() != DataType.INT16
                || ids.getInnerShape().length != 2
                || ids.getInnerShape()[0] != tokenCount
                || ids.getInnerShape()[1] != tokenCount) {
            throw new IllegalArgumentException(
                    "Relation IDs must have fixed INT16 [tokens, tokens] shape.");
        }
        TensorSpec keys = relationKeys.getSpec();
        if (keys.getLeadingDimension() != null
                || keys.getDataType() != dataType
                || keys.getInnerShape().length != 2
                || keys.getInnerShape()[0] <= 0
                || keys.getInnerShape()[1] != attentionWidth) {
            throw new IllegalArgumentException(
                    "Relation keys must have [relations, attentionWidth] shape and stack type.");
        }
        TensorSpec bias = relationBias.getSpec();
        if (bias.getLeadingDimension() != null
                || bias.getDataType() != dataType
                || bias.getInnerShape().length != 2
                || bias.getInnerShape()[0] != keys.getInnerShape()[0]
                || bias.getInnerShape()[1] != attentionHeads) {
            throw new IllegalArgumentException(
                    "Relation bias must have [relations, attentionHeads] shape and stack type.");
        }
    }
}
