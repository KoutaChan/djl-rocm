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

/** PyTorch accelerator implementation of the bounded fusion compiler. */
final class PtFusionCompiler implements FusionCompiler {

    private static final int FUSION_BACKEND_CUDA = 1;
    private static final int FUSION_BACKEND_ROCM = 2;

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
            throw new UnsupportedOperationException(
                    "PyTorch fusion requires a CUDA or ROCm device.");
        }
        for (FusionShapeProfile profile : config.getShapeProfiles()) {
            if (profile.getRecipe() != recipe) {
                throw new IllegalArgumentException(
                        "A shape profile belongs to a different fusion recipe.");
            }
        }

        String backendName = fusionBackendName(JniUtils.getFusionBackend());
        ByteBuffer descriptor = PtFusionDescriptor.encode(recipe);
        long handle = JniUtils.prepareFusionPlan(device, descriptor);
        try {
            long[] stats = JniUtils.getFusionPlanStats(handle);
            if (stats == null || stats.length != 10) {
                throw new IllegalStateException("Invalid native Fusion plan statistics.");
            }
            for (long value : stats) {
                if (value < 0) {
                    throw new IllegalStateException("Invalid native Fusion plan statistics.");
                }
            }
            FusionCompilationReport report =
                    FusionCompilationReport.builder("PyTorch " + backendName + " AOT")
                            .optCommandCount(PtFusionDescriptor.commandCount(recipe))
                            .optExecutableStorageBytes(stats[0])
                            .optPersistentStorageBytes(stats[1])
                            .optWorkspaceBytes(stats[2])
                            .optExportedOutputBytes(stats[3])
                            .optArenaBytes(stats[4])
                            .optStoragePlannerVersion(Math.toIntExact(stats[5]))
                            .optLogicalAllocationCount(stats[6])
                            .optBackingAllocationCount(stats[7])
                            .optAliasViewCount(stats[8])
                            .optInPlaceReuseCount(stats[9])
                            .optNativeOnly(true)
                            .build();
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

    private static String fusionBackendName(int backend) {
        if (backend == FUSION_BACKEND_CUDA) {
            return "CUDA";
        }
        if (backend == FUSION_BACKEND_ROCM) {
            return "ROCm";
        }
        throw new UnsupportedOperationException(
                "The loaded PyTorch native library has no CUDA or ROCm fusion backend.");
    }
}
