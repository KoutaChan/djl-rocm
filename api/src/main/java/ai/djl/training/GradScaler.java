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
package ai.djl.training;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Dynamically scales a loss to protect low-precision gradients from underflow.
 *
 * <p>{@code GradScaler} is primarily intended for {@link DataType#FLOAT16} autocast training. The
 * loss is multiplied by the current scale before backward. Immediately before the optimizer step,
 * gradients are divided by the same scale and checked for NaN or infinity. A non-finite gradient
 * skips the complete optimizer step and reduces the scale; consecutive finite steps eventually
 * increase it.
 *
 * <p>A scaler is stateful and should belong to one {@link Trainer}. After a completed optimizer
 * step, save and restore {@link State} together with the model and optimizer when exact training
 * continuation is required.
 */
public final class GradScaler {

    private static final float MIN_SCALE = Float.MIN_NORMAL;
    private static final String STATE_MAGIC = "DJL_GRAD_SCALER";
    private static final int STATE_VERSION = 1;

    private float scale;
    private final float growthFactor;
    private final float backoffFactor;
    private final int growthInterval;

    private int growthTracker;
    private boolean scaledSinceUpdate;
    private boolean unscaledSinceUpdate;
    private boolean gradientsFiniteSinceUnscale;
    private boolean lastStepSkipped;

    private GradScaler(Builder builder) {
        scale = builder.initialScale;
        growthFactor = builder.growthFactor;
        backoffFactor = builder.backoffFactor;
        growthInterval = builder.growthInterval;
    }

    /**
     * Multiplies a backward target by the current scale.
     *
     * <p>The returned array is newly allocated and must be closed by the caller.
     *
     * @param target the loss or other backward target
     * @return the scaled target
     */
    public synchronized NDArray scale(NDArray target) {
        requireRealFloating(target.getDataType(), "GradScaler target");
        if (unscaledSinceUpdate) {
            throw new IllegalStateException(
                    "GradScaler.update() is required before scaling the next step.");
        }
        NDArray scaled = target.mul(scale);
        scaledSinceUpdate = true;
        return scaled;
    }

    /**
     * Unscales gradients in place and checks that every value is finite.
     *
     * <p>This method may be called only once between {@link #scale(NDArray)} and {@link
     * #update(boolean)}.
     *
     * @param gradients the gradients to unscale
     * @return {@code true} if all gradients are finite
     */
    public synchronized boolean unscaleAndCheckFinite(NDList gradients) {
        if (!scaledSinceUpdate) {
            throw new IllegalStateException(
                    "GradScaler.unscaleAndCheckFinite() requires scale() in the current step.");
        }
        if (unscaledSinceUpdate) {
            throw new IllegalStateException(
                    "GradScaler gradients have already been unscaled for the current step.");
        }
        if (gradients.isEmpty()) {
            throw new IllegalArgumentException("GradScaler requires at least one gradient.");
        }

        Engine engine = gradients.head().getManager().getEngine();
        for (NDArray gradient : gradients) {
            if (gradient.getManager().getEngine() != engine) {
                throw new IllegalArgumentException(
                        "All gradients passed to GradScaler must use the same engine.");
            }
            requireRealFloating(gradient.getDataType(), "GradScaler gradient");
        }

        float inverseScale = (float) (1.0d / scale);
        boolean gradientsFinite = engine.unscaleGradientsAndCheckFinite(gradients, inverseScale);
        gradientsFiniteSinceUnscale = gradientsFinite;
        unscaledSinceUpdate = true;
        return gradientsFinite;
    }

    /** Updates the dynamic scale using the result of the latest finite-gradient check. */
    public synchronized void update() {
        requireUnscaled();
        updateScale(gradientsFiniteSinceUnscale);
    }

    /**
     * Updates the dynamic scale after an optimizer step or skipped step.
     *
     * <p>The supplied result must match the value returned by the current step's {@link
     * #unscaleAndCheckFinite(NDList)} call. Prefer {@link #update()} when no additional assertion
     * is needed.
     *
     * @param gradientsFinite the result returned by {@link #unscaleAndCheckFinite(NDList)}
     */
    public synchronized void update(boolean gradientsFinite) {
        requireUnscaled();
        if (gradientsFinite != gradientsFiniteSinceUnscale) {
            throw new IllegalArgumentException(
                    "GradScaler update result does not match the latest finite-gradient check.");
        }
        updateScale(gradientsFinite);
    }

    private void updateScale(boolean gradientsFinite) {
        lastStepSkipped = !gradientsFinite;
        if (gradientsFinite) {
            ++growthTracker;
            if (growthTracker >= growthInterval) {
                float grownScale = scale * growthFactor;
                if (Float.isFinite(grownScale)) {
                    scale = grownScale;
                }
                growthTracker = 0;
            }
        } else {
            scale = Math.max(scale * backoffFactor, MIN_SCALE);
            growthTracker = 0;
        }

        scaledSinceUpdate = false;
        unscaledSinceUpdate = false;
        gradientsFiniteSinceUnscale = false;
    }

    /** Returns the current scale. */
    public synchronized float getScale() {
        return scale;
    }

    /** Returns the number of consecutive finite steps since the last scale change. */
    public synchronized int getGrowthTracker() {
        return growthTracker;
    }

    /** Returns whether the most recent optimizer step was skipped. */
    public synchronized boolean wasLastStepSkipped() {
        return lastStepSkipped;
    }

    /** Returns a checkpointable snapshot of the dynamic state. */
    public synchronized State getState() {
        if (scaledSinceUpdate) {
            throw new IllegalStateException(
                    "GradScaler state can only be saved between completed steps.");
        }
        return new State(scale, growthTracker);
    }

    /**
     * Saves the scaler configuration and dynamic state.
     *
     * @param path the file to save
     * @throws IOException if the state cannot be written
     */
    public void saveState(Path path) throws IOException {
        State state = getState();
        Path parent = path.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (DataOutputStream output = new DataOutputStream(Files.newOutputStream(path))) {
            output.writeUTF(STATE_MAGIC);
            output.writeInt(STATE_VERSION);
            output.writeFloat(growthFactor);
            output.writeFloat(backoffFactor);
            output.writeInt(growthInterval);
            output.writeFloat(state.scale);
            output.writeInt(state.growthTracker);
        }
    }

    /**
     * Loads dynamic state saved by {@link #saveState(Path)}.
     *
     * <p>The saved growth configuration must match this scaler so resumed training cannot silently
     * use different scale dynamics.
     *
     * @param path the state file to load
     * @throws IOException if the state is invalid, incompatible, or cannot be read
     */
    public void loadState(Path path) throws IOException {
        float savedGrowthFactor;
        float savedBackoffFactor;
        int savedGrowthInterval;
        State state;
        try (DataInputStream input = new DataInputStream(Files.newInputStream(path))) {
            String magic = input.readUTF();
            if (!STATE_MAGIC.equals(magic)) {
                throw new IOException("Invalid GradScaler state file: " + path);
            }
            int version = input.readInt();
            if (version != STATE_VERSION) {
                throw new IOException("Unsupported GradScaler state version: " + version);
            }
            savedGrowthFactor = input.readFloat();
            savedBackoffFactor = input.readFloat();
            savedGrowthInterval = input.readInt();
            state = new State(input.readFloat(), input.readInt());
        }

        if (Float.compare(savedGrowthFactor, growthFactor) != 0
                || Float.compare(savedBackoffFactor, backoffFactor) != 0
                || savedGrowthInterval != growthInterval) {
            throw new IOException("GradScaler state configuration does not match this scaler.");
        }
        try {
            loadState(state);
        } catch (IllegalArgumentException e) {
            throw new IOException("Invalid GradScaler state file: " + path, e);
        }
    }

    /**
     * Restores dynamic state from a checkpoint.
     *
     * @param state the state to restore
     */
    public synchronized void loadState(State state) {
        if (scaledSinceUpdate) {
            throw new IllegalStateException(
                    "GradScaler state can only be restored between completed steps.");
        }
        if (state == null) {
            throw new IllegalArgumentException("GradScaler state must not be null.");
        }
        requirePositiveFinite(state.scale, "scale");
        if (state.growthTracker < 0 || state.growthTracker >= growthInterval) {
            throw new IllegalArgumentException(
                    "GradScaler growthTracker must be in [0, growthInterval).");
        }
        scale = state.scale;
        growthTracker = state.growthTracker;
        scaledSinceUpdate = false;
        unscaledSinceUpdate = false;
        gradientsFiniteSinceUnscale = false;
        lastStepSkipped = false;
    }

    /** Creates a builder for a {@code GradScaler}. */
    public static Builder builder() {
        return new Builder();
    }

    private static void requireRealFloating(DataType dataType, String name) {
        switch (dataType) {
            case FLOAT16:
            case BFLOAT16:
            case FLOAT32:
            case FLOAT64:
                return;
            default:
                throw new IllegalArgumentException(name + " must use a real floating data type.");
        }
    }

    private static void requirePositiveFinite(float value, String name) {
        if (value < MIN_SCALE || !Float.isFinite(value)) {
            throw new IllegalArgumentException(
                    name + " must be finite and at least Float.MIN_NORMAL.");
        }
    }

    private void requireUnscaled() {
        if (!unscaledSinceUpdate) {
            throw new IllegalStateException(
                    "GradScaler.update() requires gradients to be unscaled first.");
        }
    }

    /** Checkpointable dynamic {@code GradScaler} state. */
    public static final class State {

        private final float scale;
        private final int growthTracker;

        /**
         * Creates a state value.
         *
         * @param scale the current loss scale
         * @param growthTracker the current finite-step counter
         */
        public State(float scale, int growthTracker) {
            this.scale = scale;
            this.growthTracker = growthTracker;
        }

        /** Returns the saved loss scale. */
        public float getScale() {
            return scale;
        }

        /** Returns the saved finite-step counter. */
        public int getGrowthTracker() {
            return growthTracker;
        }
    }

    /** Builder for {@link GradScaler}. */
    public static final class Builder {

        private float initialScale = 65536f;
        private float growthFactor = 2f;
        private float backoffFactor = 0.5f;
        private int growthInterval = 2000;

        private Builder() {}

        /** Sets the initial loss scale. */
        public Builder optInitialScale(float initialScale) {
            this.initialScale = initialScale;
            return this;
        }

        /** Sets the multiplier applied after enough consecutive finite steps. */
        public Builder optGrowthFactor(float growthFactor) {
            this.growthFactor = growthFactor;
            return this;
        }

        /** Sets the multiplier applied after a non-finite step. */
        public Builder optBackoffFactor(float backoffFactor) {
            this.backoffFactor = backoffFactor;
            return this;
        }

        /** Sets the number of consecutive finite steps required to grow the scale. */
        public Builder optGrowthInterval(int growthInterval) {
            this.growthInterval = growthInterval;
            return this;
        }

        /** Builds a {@link GradScaler}. */
        public GradScaler build() {
            requirePositiveFinite(initialScale, "initialScale");
            if (growthFactor <= 1f || !Float.isFinite(growthFactor)) {
                throw new IllegalArgumentException(
                        "growthFactor must be finite and greater than 1.");
            }
            if (backoffFactor <= 0f || backoffFactor >= 1f || !Float.isFinite(backoffFactor)) {
                throw new IllegalArgumentException(
                        "backoffFactor must be finite and in the range (0, 1).");
            }
            if (growthInterval <= 0) {
                throw new IllegalArgumentException("growthInterval must be positive.");
            }
            return new GradScaler(this);
        }
    }
}
