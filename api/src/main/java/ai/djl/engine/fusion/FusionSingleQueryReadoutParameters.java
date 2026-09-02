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

/** Borrowed tensor parameters for one functional single-query attention readout. */
public final class FusionSingleQueryReadoutParameters {

    private final NDArray querySeedWeight;
    private final NDArray querySeedBias;
    private final NDArray queryWeight;
    private final NDArray queryBias;
    private final NDArray keyValueWeight;
    private final NDArray contextWeight;
    private final NDArray contextBias;
    private final NDArray queryNormWeight;
    private final NDArray queryNormBias;
    private final NDArray feedForwardNormWeight;
    private final NDArray feedForwardNormBias;
    private final NDArray feedForwardExpansionWeight;
    private final NDArray feedForwardExpansionBias;
    private final NDArray feedForwardProjectionWeight;
    private final NDArray feedForwardProjectionBias;
    private final NDArray outputNormWeight;
    private final NDArray outputNormBias;

    /**
     * Constructs one readout parameter set.
     *
     * <p>The object borrows every array and does not close, attach, or copy it. All arrays remain
     * ordinary autograd inputs.
     */
    public FusionSingleQueryReadoutParameters(
            NDArray querySeedWeight,
            NDArray querySeedBias,
            NDArray queryWeight,
            NDArray queryBias,
            NDArray keyValueWeight,
            NDArray contextWeight,
            NDArray contextBias,
            NDArray queryNormWeight,
            NDArray queryNormBias,
            NDArray feedForwardNormWeight,
            NDArray feedForwardNormBias,
            NDArray feedForwardExpansionWeight,
            NDArray feedForwardExpansionBias,
            NDArray feedForwardProjectionWeight,
            NDArray feedForwardProjectionBias,
            NDArray outputNormWeight,
            NDArray outputNormBias) {
        this.querySeedWeight = querySeedWeight;
        this.querySeedBias = querySeedBias;
        this.queryWeight = queryWeight;
        this.queryBias = queryBias;
        this.keyValueWeight = keyValueWeight;
        this.contextWeight = contextWeight;
        this.contextBias = contextBias;
        this.queryNormWeight = queryNormWeight;
        this.queryNormBias = queryNormBias;
        this.feedForwardNormWeight = feedForwardNormWeight;
        this.feedForwardNormBias = feedForwardNormBias;
        this.feedForwardExpansionWeight = feedForwardExpansionWeight;
        this.feedForwardExpansionBias = feedForwardExpansionBias;
        this.feedForwardProjectionWeight = feedForwardProjectionWeight;
        this.feedForwardProjectionBias = feedForwardProjectionBias;
        this.outputNormWeight = outputNormWeight;
        this.outputNormBias = outputNormBias;
    }

    /** Returns the query-seed projection weight. */
    public NDArray getQuerySeedWeight() {
        return querySeedWeight;
    }

    /** Returns the query-seed projection bias. */
    public NDArray getQuerySeedBias() {
        return querySeedBias;
    }

    /** Returns the attention-query projection weight. */
    public NDArray getQueryWeight() {
        return queryWeight;
    }

    /** Returns the attention-query projection bias. */
    public NDArray getQueryBias() {
        return queryBias;
    }

    /** Returns the packed key-value projection weight. */
    public NDArray getKeyValueWeight() {
        return keyValueWeight;
    }

    /** Returns the attention-context projection weight. */
    public NDArray getContextWeight() {
        return contextWeight;
    }

    /** Returns the attention-context projection bias. */
    public NDArray getContextBias() {
        return contextBias;
    }

    /** Returns the post-attention LayerNorm scale. */
    public NDArray getQueryNormWeight() {
        return queryNormWeight;
    }

    /** Returns the post-attention LayerNorm bias. */
    public NDArray getQueryNormBias() {
        return queryNormBias;
    }

    /** Returns the feed-forward-input LayerNorm scale. */
    public NDArray getFeedForwardNormWeight() {
        return feedForwardNormWeight;
    }

    /** Returns the feed-forward-input LayerNorm bias. */
    public NDArray getFeedForwardNormBias() {
        return feedForwardNormBias;
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
    public NDArray getOutputNormWeight() {
        return outputNormWeight;
    }

    /** Returns the output LayerNorm bias. */
    public NDArray getOutputNormBias() {
        return outputNormBias;
    }
}
