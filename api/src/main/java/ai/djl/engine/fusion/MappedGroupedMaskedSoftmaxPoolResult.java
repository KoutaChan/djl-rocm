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

import ai.djl.ndarray.NDList;

/** Results of a functional mapped grouped masked-softmax pool. */
public final class MappedGroupedMaskedSoftmaxPoolResult {

    private final NDList contexts;
    private final NDList presence;

    MappedGroupedMaskedSoftmaxPoolResult(NDList contexts, NDList presence) {
        this.contexts = contexts;
        this.presence = presence;
    }

    /**
     * Returns the FLOAT32 pooled contexts in mapping order.
     *
     * @return one context tensor per output set
     */
    public NDList getContexts() {
        return contexts;
    }

    /**
     * Returns the non-differentiable presence values in mapping order.
     *
     * @return one presence tensor per output set
     */
    public NDList getPresence() {
        return presence;
    }
}
