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
import ai.djl.engine.fusion.FusionCompilationReport;
import ai.djl.engine.fusion.FusionCompileConfig;
import ai.djl.engine.fusion.FusionCompiler;
import ai.djl.engine.fusion.FusionPlan;
import ai.djl.engine.fusion.FusionRecipe;
import ai.djl.engine.fusion.FusionShapeProfile;
import ai.djl.pytorch.jni.JniUtils;

import java.nio.ByteBuffer;
import java.util.Objects;

/** PyTorch ROCm implementation of the bounded fusion compiler. */
final class PtFusionCompiler implements FusionCompiler {

    private final Device device;

    PtFusionCompiler(Device device) {
        this.device = Objects.requireNonNull(device, "device");
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        return device;
    }

    /** {@inheritDoc} */
    @Override
    public FusionPlan prepare(FusionRecipe recipe, FusionCompileConfig config) {
        Objects.requireNonNull(recipe, "recipe");
        Objects.requireNonNull(config, "config");
        if (!device.isGpu()) {
            throw new UnsupportedOperationException("PyTorch fusion requires a ROCm device.");
        }
        for (FusionShapeProfile profile : config.getShapeProfiles()) {
            if (profile.getRecipe() != recipe) {
                throw new IllegalArgumentException(
                        "A shape profile belongs to a different fusion recipe.");
            }
        }

        ByteBuffer descriptor = PtFusionDescriptor.encode(recipe);
        FusionCompilationReport report =
                FusionCompilationReport.builder("PyTorch ROCm AOT")
                        .optCommandCount(PtFusionDescriptor.commandCount(recipe))
                        .optPersistentStorageBytes(
                                PtFusionDescriptor.persistentStorageBytes(recipe))
                        .optWorkspaceBytes(PtFusionDescriptor.workspaceBytes(recipe))
                        .optNativeOnly(true)
                        .build();
        long handle = JniUtils.prepareFusionPlan(device, descriptor);
        try {
            return new PtFusionPlan(device, recipe, report, handle);
        } catch (RuntimeException | Error failure) {
            try {
                JniUtils.deleteFusionPlan(handle);
            } catch (RuntimeException | Error cleanupFailure) {
                failure.addSuppressed(cleanupFailure);
            }
            throw failure;
        }
    }
}
