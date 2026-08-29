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

/** An externally serialized PyTorch fusion execution lane. */
final class PtFusionSession extends NativeResource<Long> implements FusionSession {

    private final FusionRecipe recipe;
    private final PtFusionSlot[] slots;
    private final PtNDArray[][] outputs;
    private int nextBuffer;

    PtFusionSession(FusionRecipe recipe, PtNDManager manager, long handle, int bufferCount) {
        super(handle);
        this.recipe = recipe;
        slots = new PtFusionSlot[bufferCount];
        outputs = new PtNDArray[bufferCount][recipe.getOutputs().size()];
        try {
            for (int bufferIndex = 0; bufferIndex < bufferCount; ++bufferIndex) {
                for (int outputIndex = 0; outputIndex < recipe.getOutputs().size(); ++outputIndex) {
                    outputs[bufferIndex][outputIndex] =
                            JniUtils.getFusionSessionOutput(
                                    manager, handle, bufferIndex, outputIndex);
                }
                slots[bufferIndex] = new PtFusionSlot(this, bufferIndex, recipe);
            }
        } catch (RuntimeException | Error e) {
            Long pointer = this.handle.getAndSet(null);
            closeOutputs(e);
            if (pointer != null) {
                try {
                    JniUtils.deleteFusionSession(pointer);
                } catch (RuntimeException | Error cleanupFailure) {
                    e.addSuppressed(cleanupFailure);
                }
            }
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
    public FusionInvocation acquire() {
        getHandle();
        for (int offset = 0; offset < slots.length; ++offset) {
            int index = (nextBuffer + offset) % slots.length;
            PtFusionSlot slot = slots[index];
            long generation = slot.tryAcquire();
            if (generation != 0) {
                nextBuffer = (index + 1) % slots.length;
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

    PtNDArray getOutput(int bufferIndex, FusionRecipe.Output output) {
        int index = output.getIndex();
        if (index < 0
                || index >= recipe.getOutputs().size()
                || recipe.getOutputs().get(index) != output) {
            throw new IllegalArgumentException("The output belongs to a different fusion recipe.");
        }
        return outputs[bufferIndex][index];
    }

    void submit(int bufferIndex, ByteBuffer inputHandles, ByteBuffer dimensions) {
        JniUtils.submitFusion(getHandle(), bufferIndex, inputHandles, dimensions);
    }

    void synchronize(int bufferIndex) {
        JniUtils.synchronizeFusionOutput(getHandle(), bufferIndex);
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
        Long pointer = handle.getAndSet(null);
        if (pointer != null) {
            failure = closeOutputs(failure);
            try {
                JniUtils.deleteFusionSession(pointer);
            } catch (RuntimeException | Error e) {
                failure = addFailure(failure, e);
            }
        }
        throwFailure(failure);
    }

    private Throwable closeOutputs(Throwable failure) {
        for (PtNDArray[] buffer : outputs) {
            for (PtNDArray output : buffer) {
                if (output != null) {
                    try {
                        output.close();
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
