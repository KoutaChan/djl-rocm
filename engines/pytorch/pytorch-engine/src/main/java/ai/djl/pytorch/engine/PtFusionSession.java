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

import ai.djl.engine.fusion.FusionInvocation;
import ai.djl.engine.fusion.FusionRecipe;
import ai.djl.engine.fusion.FusionSession;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.util.NativeResource;

import java.nio.ByteBuffer;
import java.util.Objects;

/** An externally serialized PyTorch fusion execution lane. */
final class PtFusionSession extends NativeResource<Long> implements FusionSession {

    private final FusionRecipe recipe;
    private final long[] capacities;
    private final PtFusionSlot[] slots;
    private final PtNDArray[][] outputs;
    private int nextOutputSlot;

    PtFusionSession(
            FusionRecipe recipe,
            PtNDManager manager,
            long handle,
            int outputSlotCount,
            long[] capacities) {
        super(handle);
        this.recipe = recipe;
        this.capacities = capacities;
        slots = new PtFusionSlot[outputSlotCount];
        outputs = new PtNDArray[outputSlotCount][recipe.getOutputs().size()];
        try {
            for (int outputSlotIndex = 0; outputSlotIndex < outputSlotCount; ++outputSlotIndex) {
                for (int outputIndex = 0; outputIndex < recipe.getOutputs().size(); ++outputIndex) {
                    outputs[outputSlotIndex][outputIndex] =
                            JniUtils.getFusionSessionOutput(
                                    manager, handle, outputSlotIndex, outputIndex);
                }
                slots[outputSlotIndex] =
                        new PtFusionSlot(this, outputSlotIndex, recipe, this.capacities);
            }
        } catch (RuntimeException | Error e) {
            this.handle.set(null);
            closeOutputs(e);
            throw e;
        }
    }

    /** {@inheritDoc} */
    @Override
    public FusionRecipe getRecipe() {
        return recipe;
    }

    /** {@inheritDoc} */
    @Override
    public long getCapacity(FusionRecipe.Dimension dimension) {
        Objects.requireNonNull(dimension, "dimension");
        int index = dimension.getIndex();
        if (index < 0
                || index >= recipe.getDimensions().size()
                || recipe.getDimensions().get(index) != dimension) {
            throw new IllegalArgumentException(
                    "The dimension belongs to a different fusion recipe.");
        }
        return capacities[index];
    }

    /** {@inheritDoc} */
    @Override
    public FusionInvocation acquire() {
        getHandle();
        for (int offset = 0; offset < slots.length; ++offset) {
            int index = (nextOutputSlot + offset) % slots.length;
            PtFusionSlot slot = slots[index];
            long generation = slot.tryAcquire();
            if (generation != 0) {
                nextOutputSlot = (index + 1) % slots.length;
                try {
                    return new PtFusionInvocation(slot, generation);
                } catch (RuntimeException | Error e) {
                    slot.closeInvocation(generation);
                    throw e;
                }
            }
        }
        throw new IllegalStateException("No fusion output slot is available.");
    }

    PtNDArray getOutput(int outputSlotIndex, FusionRecipe.Output output) {
        int index = output.getIndex();
        if (index < 0
                || index >= recipe.getOutputs().size()
                || recipe.getOutputs().get(index) != output) {
            throw new IllegalArgumentException("The output belongs to a different fusion recipe.");
        }
        return outputs[outputSlotIndex][index];
    }

    void submit(int outputSlotIndex, ByteBuffer inputHandles, ByteBuffer dimensions) {
        JniUtils.submitFusion(getHandle(), outputSlotIndex, inputHandles, dimensions);
    }

    void synchronize(int outputSlotIndex) {
        JniUtils.synchronizeFusionOutput(getHandle(), outputSlotIndex);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        for (PtFusionSlot slot : slots) {
            if (slot != null && !slot.isFree()) {
                throw new IllegalStateException(
                        "Cannot close a fusion session with active output slots.");
            }
        }
        Throwable failure = null;
        try {
            onClose();
        } catch (RuntimeException | Error e) {
            failure = e;
        }
        Long pointer = handle.get();
        if (pointer != null) {
            try {
                JniUtils.deleteFusionSession(pointer);
            } catch (RuntimeException | Error e) {
                failure = addFailure(failure, e);
                throwFailure(failure);
                return;
            }
            handle.set(null);
        }
        failure = closeOutputs(failure);
        throwFailure(failure);
    }

    private Throwable closeOutputs(Throwable failure) {
        for (int outputSlotIndex = 0; outputSlotIndex < outputs.length; ++outputSlotIndex) {
            PtNDArray[] slotOutputs = outputs[outputSlotIndex];
            for (int outputIndex = 0; outputIndex < slotOutputs.length; ++outputIndex) {
                PtNDArray output = slotOutputs[outputIndex];
                if (output != null) {
                    try {
                        output.close();
                        slotOutputs[outputIndex] = null;
                    } catch (RuntimeException | Error e) {
                        failure = addFailure(failure, e);
                    }
                }
            }
        }
        return failure;
    }

    private static Throwable addFailure(Throwable failure, Throwable additional) {
        if (failure == null) {
            return additional;
        }
        failure.addSuppressed(additional);
        return failure;
    }

    private static void throwFailure(Throwable failure) {
        if (failure instanceof RuntimeException) {
            throw (RuntimeException) failure;
        }
        if (failure instanceof Error) {
            throw (Error) failure;
        }
        if (failure != null) {
            throw new IllegalStateException("Failed to close the fusion session.", failure);
        }
    }
}
