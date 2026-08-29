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

import ai.djl.engine.fusion.FusionOutputLease;
import ai.djl.engine.fusion.FusionRecipe;
import ai.djl.ndarray.NDArray;

/** A generation-checked single-use lease for one persistent PyTorch fusion output slot. */
final class PtFusionOutputLease implements FusionOutputLease {

    private final PtFusionSlot slot;
    private final long generation;
    private boolean closed;

    PtFusionOutputLease(PtFusionSlot slot, long generation) {
        this.slot = slot;
        this.generation = generation;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray get(FusionRecipe.Output output) {
        checkOpen();
        return slot.getOutput(generation, output);
    }

    /** {@inheritDoc} */
    @Override
    public long getDimension(FusionRecipe.Dimension dimension) {
        checkOpen();
        return slot.getDimension(generation, dimension);
    }

    /** {@inheritDoc} */
    @Override
    public void synchronize() {
        checkOpen();
        slot.synchronize(generation);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (!closed) {
            slot.releaseLease(generation);
            closed = true;
        }
    }

    private void checkOpen() {
        if (closed) {
            throw new IllegalStateException("Fusion output lease is closed.");
        }
    }
}
