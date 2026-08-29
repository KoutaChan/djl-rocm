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

import ai.djl.Device;
import ai.djl.engine.fusion.FusionConstantBindings;
import ai.djl.engine.fusion.FusionExecutable;
import ai.djl.engine.fusion.FusionRecipe;
import ai.djl.engine.fusion.FusionSession;
import ai.djl.engine.fusion.FusionSessionConfig;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.util.NativeResource;

import java.util.Objects;

/** A native PyTorch fusion plan with constants bound. */
final class PtFusionExecutable extends NativeResource<Long> implements FusionExecutable {

    private final Device device;
    private final FusionRecipe recipe;

    @SuppressWarnings("unused")
    private final FusionConstantBindings constants;

    PtFusionExecutable(
            Device device, FusionRecipe recipe, long handle, FusionConstantBindings constants) {
        super(handle);
        this.device = device;
        this.recipe = recipe;
        this.constants = constants;
    }

    /** {@inheritDoc} */
    @Override
    public FusionRecipe getRecipe() {
        return recipe;
    }

    /** {@inheritDoc} */
    @Override
    public FusionSession newSession(NDManager manager, FusionSessionConfig config) {
        Objects.requireNonNull(manager, "manager");
        Objects.requireNonNull(config, "config");
        if (!(manager instanceof PtNDManager)) {
            throw new IllegalArgumentException("The fusion session requires a PyTorch NDManager.");
        }
        PtNDManager ptManager = (PtNDManager) manager;
        if (!device.equals(ptManager.getDevice())) {
            throw new IllegalArgumentException(
                    "The fusion session manager must use the compiler device.");
        }
        long sessionHandle = JniUtils.createFusionSession(getHandle(), config.getBufferCount());
        return new PtFusionSession(recipe, ptManager, sessionHandle, config.getBufferCount());
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        onClose();
        Long pointer = handle.getAndSet(null);
        if (pointer != null) {
            JniUtils.deleteFusionExecutable(pointer);
        }
    }
}
