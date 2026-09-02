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

/** Tensor parameters for indexed-relation transformer attention. */
public final class FusionIndexedRelationParameters {

    private final NDArray relationIds;
    private final NDArray relationKeys;
    private final NDArray relationBias;

    /**
     * Constructs indexed-relation parameters.
     *
     * <p>The relation IDs are shaped {@code [tokens, tokens]}. Relation keys use {@code [relations,
     * attentionWidth]} and relation bias uses {@code [relations, attentionHeads]}. This object
     * borrows the arrays and does not close, attach, or copy them. Relation IDs are discrete
     * metadata. Automatic differentiation applies only to the floating-point key and bias arrays.
     *
     * @param relationIds zero-based relation IDs
     * @param relationKeys relation-key table
     * @param relationBias per-relation, per-head bias table
     */
    public FusionIndexedRelationParameters(
            NDArray relationIds, NDArray relationKeys, NDArray relationBias) {
        this.relationIds = relationIds;
        this.relationKeys = relationKeys;
        this.relationBias = relationBias;
    }

    /** Returns the zero-based relation IDs. */
    public NDArray getRelationIds() {
        return relationIds;
    }

    /** Returns the relation-key table. */
    public NDArray getRelationKeys() {
        return relationKeys;
    }

    /** Returns the per-relation, per-head bias table. */
    public NDArray getRelationBias() {
        return relationBias;
    }
}
