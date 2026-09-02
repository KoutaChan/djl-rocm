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

/** Borrowed tensor parameters for one functional transformer encoder block. */
public final class FusionTransformerBlockParameters {

    private final NDArray attentionInputWeight;
    private final NDArray attentionInputBias;
    private final NDArray queryKeyValueWeight;
    private final NDArray attentionOutputWeight;
    private final NDArray attentionOutputBias;
    private final NDArray feedForwardInputWeight;
    private final NDArray feedForwardInputBias;
    private final NDArray feedForwardExpansionWeight;
    private final NDArray feedForwardExpansionBias;
    private final NDArray feedForwardProjectionWeight;
    private final NDArray feedForwardProjectionBias;
    private final NDArray outputWeight;
    private final NDArray outputBias;
    private final FusionIndexedRelationParameters indexedRelation;

    /** Constructs a transformer block without indexed-relation attention. */
    public FusionTransformerBlockParameters(
            NDArray attentionInputWeight,
            NDArray attentionInputBias,
            NDArray queryKeyValueWeight,
            NDArray attentionOutputWeight,
            NDArray attentionOutputBias,
            NDArray feedForwardInputWeight,
            NDArray feedForwardInputBias,
            NDArray feedForwardExpansionWeight,
            NDArray feedForwardExpansionBias,
            NDArray feedForwardProjectionWeight,
            NDArray feedForwardProjectionBias,
            NDArray outputWeight,
            NDArray outputBias) {
        this(
                attentionInputWeight,
                attentionInputBias,
                queryKeyValueWeight,
                attentionOutputWeight,
                attentionOutputBias,
                feedForwardInputWeight,
                feedForwardInputBias,
                feedForwardExpansionWeight,
                feedForwardExpansionBias,
                feedForwardProjectionWeight,
                feedForwardProjectionBias,
                outputWeight,
                outputBias,
                null);
    }

    /**
     * Constructs a transformer block.
     *
     * <p>This object borrows every array. It does not close, attach, or copy parameters. The
     * optional indexed-relation object selects relation-biased attention. All floating-point arrays
     * remain ordinary autograd inputs.
     */
    public FusionTransformerBlockParameters(
            NDArray attentionInputWeight,
            NDArray attentionInputBias,
            NDArray queryKeyValueWeight,
            NDArray attentionOutputWeight,
            NDArray attentionOutputBias,
            NDArray feedForwardInputWeight,
            NDArray feedForwardInputBias,
            NDArray feedForwardExpansionWeight,
            NDArray feedForwardExpansionBias,
            NDArray feedForwardProjectionWeight,
            NDArray feedForwardProjectionBias,
            NDArray outputWeight,
            NDArray outputBias,
            FusionIndexedRelationParameters indexedRelation) {
        this.attentionInputWeight = attentionInputWeight;
        this.attentionInputBias = attentionInputBias;
        this.queryKeyValueWeight = queryKeyValueWeight;
        this.attentionOutputWeight = attentionOutputWeight;
        this.attentionOutputBias = attentionOutputBias;
        this.feedForwardInputWeight = feedForwardInputWeight;
        this.feedForwardInputBias = feedForwardInputBias;
        this.feedForwardExpansionWeight = feedForwardExpansionWeight;
        this.feedForwardExpansionBias = feedForwardExpansionBias;
        this.feedForwardProjectionWeight = feedForwardProjectionWeight;
        this.feedForwardProjectionBias = feedForwardProjectionBias;
        this.outputWeight = outputWeight;
        this.outputBias = outputBias;
        this.indexedRelation = indexedRelation;
    }

    /** Returns the attention-input LayerNorm scale. */
    public NDArray getAttentionInputWeight() {
        return attentionInputWeight;
    }

    /** Returns the attention-input LayerNorm bias. */
    public NDArray getAttentionInputBias() {
        return attentionInputBias;
    }

    /** Returns the combined query-key-value projection weight. */
    public NDArray getQueryKeyValueWeight() {
        return queryKeyValueWeight;
    }

    /** Returns the attention output projection weight. */
    public NDArray getAttentionOutputWeight() {
        return attentionOutputWeight;
    }

    /** Returns the attention output projection bias. */
    public NDArray getAttentionOutputBias() {
        return attentionOutputBias;
    }

    /** Returns the feed-forward-input LayerNorm scale. */
    public NDArray getFeedForwardInputWeight() {
        return feedForwardInputWeight;
    }

    /** Returns the feed-forward-input LayerNorm bias. */
    public NDArray getFeedForwardInputBias() {
        return feedForwardInputBias;
    }

    /** Returns the feed-forward expansion weight. */
    public NDArray getFeedForwardExpansionWeight() {
        return feedForwardExpansionWeight;
    }

    /** Returns the feed-forward expansion bias. */
    public NDArray getFeedForwardExpansionBias() {
        return feedForwardExpansionBias;
    }

    /** Returns the feed-forward output projection weight. */
    public NDArray getFeedForwardProjectionWeight() {
        return feedForwardProjectionWeight;
    }

    /** Returns the feed-forward output projection bias. */
    public NDArray getFeedForwardProjectionBias() {
        return feedForwardProjectionBias;
    }

    /** Returns the output LayerNorm scale. */
    public NDArray getOutputWeight() {
        return outputWeight;
    }

    /** Returns the output LayerNorm bias. */
    public NDArray getOutputBias() {
        return outputBias;
    }

    /** Returns indexed-relation parameters, or {@code null} for ordinary attention. */
    public FusionIndexedRelationParameters getIndexedRelation() {
        return indexedRelation;
    }
}
