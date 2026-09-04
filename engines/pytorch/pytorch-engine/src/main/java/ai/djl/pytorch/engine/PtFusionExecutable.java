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
import ai.djl.engine.fusion.FusionConstantBindings;
import ai.djl.engine.fusion.FusionExecutable;
import ai.djl.engine.fusion.FusionRecipe;
import ai.djl.engine.fusion.FusionSession;
import ai.djl.engine.fusion.FusionSessionConfig;
import ai.djl.engine.fusion.FusionShapeProfile;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.util.NativeResource;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** A native PyTorch fusion plan with constants bound. */
final class PtFusionExecutable extends NativeResource<Long> implements FusionExecutable {

    private final Device device;
    private final FusionRecipe recipe;
    private final List<FusionShapeProfile> shapeProfiles;
    private final PtFusionProfileDescriptor.Catalog shapeProfileCatalog;
    private final Map<FusionShapeProfile, FusionCompilationReport> shapeProfileReports;

    @SuppressWarnings("unused")
    private final FusionConstantBindings constants;

    PtFusionExecutable(
            Device device,
            FusionRecipe recipe,
            long handle,
            FusionConstantBindings constants,
            List<FusionShapeProfile> shapeProfiles,
            PtFusionProfileDescriptor.Catalog shapeProfileCatalog,
            Map<FusionShapeProfile, FusionCompilationReport> shapeProfileReports) {
        super(handle);
        this.device = device;
        this.recipe = recipe;
        this.constants = constants;
        this.shapeProfiles = shapeProfiles;
        this.shapeProfileCatalog = shapeProfileCatalog;
        this.shapeProfileReports = shapeProfileReports;
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
        int outputSlotCount = config.getOutputSlotCount();
        Selection selection = selectProfile(config, outputSlotCount);
        long sessionHandle =
                JniUtils.createFusionSession(getHandle(), selection.variantIndex, outputSlotCount);
        try {
            return new PtFusionSession(
                    recipe, ptManager, sessionHandle, outputSlotCount, selection.capacities);
        } catch (RuntimeException | Error failure) {
            try {
                JniUtils.deleteFusionSession(sessionHandle);
            } catch (RuntimeException | Error cleanupFailure) {
                failure.addSuppressed(cleanupFailure);
            }
            throw failure;
        }
    }

    private Selection selectProfile(FusionSessionConfig config, int outputSlotCount) {
        long[] maximumCapacities = shapeProfileCatalog.getMaximumCapacities();
        FusionShapeProfile requested = config.getRequestedShapeProfile();
        if (requested == null) {
            return new Selection(0, maximumCapacities);
        }
        long[] requestedCapacities = PtFusionProfileDescriptor.resolve(recipe, requested);
        if (config.getProfileFallback() == FusionSessionConfig.ProfileFallback.EXACT) {
            if (Arrays.equals(requestedCapacities, maximumCapacities)) {
                return new Selection(0, maximumCapacities);
            }
            for (int i = 0; i < shapeProfileCatalog.size(); ++i) {
                long[] capacities = shapeProfileCatalog.getCapacities(i);
                if (Arrays.equals(requestedCapacities, capacities)) {
                    return new Selection(i + 1, capacities);
                }
            }
            throw new IllegalArgumentException(
                    "No exactly matching Fusion storage-capacity profile was compiled.");
        }

        int bestIndex = -1;
        long bestStorageBytes = Long.MAX_VALUE;
        for (int i = 0; i < shapeProfileCatalog.size(); ++i) {
            long[] candidate = shapeProfileCatalog.getCapacities(i);
            if (!fits(candidate, requestedCapacities)) {
                continue;
            }
            long storageBytes = sessionStorageCost(i, outputSlotCount);
            if (bestIndex < 0 || storageBytes < bestStorageBytes) {
                bestIndex = i;
                bestStorageBytes = storageBytes;
            }
        }
        if (bestIndex < 0) {
            return new Selection(0, maximumCapacities);
        }
        return new Selection(bestIndex + 1, shapeProfileCatalog.getCapacities(bestIndex));
    }

    private long sessionStorageCost(int profileIndex, int outputSlotCount) {
        try {
            return shapeProfileReports
                    .get(shapeProfiles.get(profileIndex))
                    .getSessionStorageBytes(outputSlotCount);
        } catch (ArithmeticException overflow) {
            return Long.MAX_VALUE;
        }
    }

    private static boolean fits(long[] candidate, long[] requested) {
        for (int i = 0; i < candidate.length; ++i) {
            if (candidate[i] < requested[i]) {
                return false;
            }
        }
        return true;
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

    private static final class Selection {

        private final int variantIndex;
        private final long[] capacities;

        private Selection(int variantIndex, long[] capacities) {
            this.variantIndex = variantIndex;
            this.capacities = capacities;
        }
    }
}
