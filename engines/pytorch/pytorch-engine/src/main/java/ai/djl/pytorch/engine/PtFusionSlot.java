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
import ai.djl.ndarray.NDArray;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

/** Persistent storage and lifecycle state for one PyTorch fusion ring slot. */
final class PtFusionSlot {

    private enum State {
        FREE,
        ACQUIRED,
        SUBMITTED
    }

    private final PtFusionSession session;
    private final int outputSlotIndex;
    private final FusionRecipe recipe;
    private final long[] capacities;
    private final NDArray[] inputs;
    private final boolean[] inputSet;
    private final boolean[] dimensionSet;
    private final ByteBuffer inputHandles;
    private final ByteBuffer dimensions;
    private State state;
    private long generation;
    private int unsetInputs;
    private int unsetDimensions;

    PtFusionSlot(
            PtFusionSession session, int outputSlotIndex, FusionRecipe recipe, long[] capacities) {
        this.session = session;
        this.outputSlotIndex = outputSlotIndex;
        this.recipe = recipe;
        this.capacities = capacities;
        inputs = new NDArray[recipe.getInputs().size()];
        inputSet = new boolean[inputs.length];
        dimensionSet = new boolean[recipe.getDimensions().size()];
        inputHandles = directLongBuffer(inputs.length);
        dimensions = directLongBuffer(dimensionSet.length);
        state = State.FREE;
    }

    long tryAcquire() {
        if (state != State.FREE) {
            return 0;
        }
        ++generation;
        state = State.ACQUIRED;
        unsetInputs = inputs.length;
        unsetDimensions = dimensionSet.length;
        return generation;
    }

    boolean isFree() {
        return state == State.FREE;
    }

    void setInput(long expectedGeneration, FusionRecipe.Input input, NDArray array) {
        checkState(expectedGeneration, State.ACQUIRED);
        int index = checkedInputIndex(input);
        Objects.requireNonNull(array, "array");
        if (!(array instanceof PtNDArray)) {
            throw new IllegalArgumentException("Fusion inputs must be PyTorch NDArrays.");
        }
        long handle = ((PtNDArray) array).getHandle();
        inputs[index] = array;
        inputHandles.putLong(index * Long.BYTES, handle);
        if (!inputSet[index]) {
            inputSet[index] = true;
            --unsetInputs;
        }
    }

    void setDimension(long expectedGeneration, FusionRecipe.Dimension dimension, long extent) {
        checkState(expectedGeneration, State.ACQUIRED);
        int index = checkedDimensionIndex(dimension);
        if (extent < 0 || extent > capacities[index]) {
            throw new IllegalArgumentException(
                    "The active extent must be between zero and the session capacity.");
        }
        dimensions.putLong(index * Long.BYTES, extent);
        if (!dimensionSet[index]) {
            dimensionSet[index] = true;
            --unsetDimensions;
        }
    }

    void submit(long expectedGeneration) {
        checkState(expectedGeneration, State.ACQUIRED);
        if (unsetInputs != 0 || unsetDimensions != 0) {
            throw new IllegalStateException(
                    "Every fusion input and dimension must be set before submission.");
        }
        session.submit(outputSlotIndex, inputHandles, dimensions);
        state = State.SUBMITTED;
    }

    NDArray getOutput(long expectedGeneration, FusionRecipe.Output output) {
        checkState(expectedGeneration, State.SUBMITTED);
        return session.getOutput(outputSlotIndex, Objects.requireNonNull(output, "output"));
    }

    long getDimension(long expectedGeneration, FusionRecipe.Dimension dimension) {
        checkState(expectedGeneration, State.SUBMITTED);
        int index = checkedDimensionIndex(dimension);
        return dimensions.getLong(index * Long.BYTES);
    }

    void synchronize(long expectedGeneration) {
        checkState(expectedGeneration, State.SUBMITTED);
        session.synchronize(outputSlotIndex);
    }

    void closeInvocation(long expectedGeneration) {
        checkGeneration(expectedGeneration);
        if (state == State.ACQUIRED) {
            release();
        } else if (state != State.SUBMITTED) {
            throw new IllegalStateException("Fusion invocation is no longer active.");
        }
    }

    void releaseLease(long expectedGeneration) {
        checkState(expectedGeneration, State.SUBMITTED);
        release();
    }

    private void release() {
        Arrays.fill(inputs, null);
        Arrays.fill(inputSet, false);
        Arrays.fill(dimensionSet, false);
        state = State.FREE;
    }

    private int checkedInputIndex(FusionRecipe.Input input) {
        Objects.requireNonNull(input, "input");
        int index = input.getInputIndex();
        if (index < 0
                || index >= recipe.getInputs().size()
                || recipe.getInputs().get(index) != input) {
            throw new IllegalArgumentException("The input belongs to a different fusion recipe.");
        }
        return index;
    }

    private int checkedDimensionIndex(FusionRecipe.Dimension dimension) {
        Objects.requireNonNull(dimension, "dimension");
        int index = dimension.getIndex();
        if (index < 0
                || index >= recipe.getDimensions().size()
                || recipe.getDimensions().get(index) != dimension) {
            throw new IllegalArgumentException(
                    "The dimension belongs to a different fusion recipe.");
        }
        return index;
    }

    private void checkState(long expectedGeneration, State expectedState) {
        checkGeneration(expectedGeneration);
        if (state != expectedState) {
            throw new IllegalStateException(
                    "Fusion slot is " + state + ", expected " + expectedState + '.');
        }
    }

    private void checkGeneration(long expectedGeneration) {
        if (generation != expectedGeneration) {
            throw new IllegalStateException("Fusion handle belongs to an expired slot generation.");
        }
    }

    private static ByteBuffer directLongBuffer(int size) {
        return ByteBuffer.allocateDirect(Math.multiplyExact(size, Long.BYTES))
                .order(ByteOrder.nativeOrder());
    }
}
