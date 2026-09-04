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
import ai.djl.engine.fusion.FusionPlan;
import ai.djl.engine.fusion.FusionRecipe;
import ai.djl.engine.fusion.FusionShapeProfile;
import ai.djl.ndarray.NDArray;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.util.NativeResource;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** An immutable native PyTorch fusion command plan. */
final class PtFusionPlan extends NativeResource<Long> implements FusionPlan {

    private final Device device;
    private final FusionRecipe recipe;
    private final FusionCompilationReport compilationReport;
    private final List<FusionShapeProfile> shapeProfiles;
    private final PtFusionProfileDescriptor.Catalog shapeProfileCatalog;
    private final Map<FusionShapeProfile, FusionCompilationReport> shapeProfileReports;

    PtFusionPlan(
            Device device,
            FusionRecipe recipe,
            FusionCompilationReport compilationReport,
            List<FusionShapeProfile> shapeProfiles,
            PtFusionProfileDescriptor.Catalog shapeProfileCatalog,
            List<FusionCompilationReport> shapeProfileReports,
            long handle) {
        super(handle);
        this.device = device;
        this.recipe = recipe;
        this.compilationReport = compilationReport;
        this.shapeProfiles = Collections.unmodifiableList(new ArrayList<>(shapeProfiles));
        this.shapeProfileCatalog = shapeProfileCatalog;
        if (shapeProfiles.size() != shapeProfileReports.size()
                || shapeProfiles.size() != shapeProfileCatalog.size()) {
            throw new IllegalArgumentException("Inconsistent Fusion shape profile metadata.");
        }
        Map<FusionShapeProfile, FusionCompilationReport> reports = new LinkedHashMap<>();
        for (int i = 0; i < shapeProfiles.size(); ++i) {
            reports.put(shapeProfiles.get(i), shapeProfileReports.get(i));
        }
        this.shapeProfileReports = Collections.unmodifiableMap(reports);
    }

    /** {@inheritDoc} */
    @Override
    public FusionRecipe getRecipe() {
        return recipe;
    }

    /** {@inheritDoc} */
    @Override
    public FusionCompilationReport getCompilationReport() {
        return compilationReport;
    }

    /** {@inheritDoc} */
    @Override
    public FusionCompilationReport getCompilationReport(FusionShapeProfile profile) {
        Objects.requireNonNull(profile, "profile");
        if (profile.getRecipe() != recipe) {
            throw new IllegalArgumentException(
                    "The shape profile belongs to a different fusion recipe.");
        }
        FusionCompilationReport report = shapeProfileReports.get(profile);
        if (report == null) {
            throw new IllegalArgumentException("The shape profile was not compiled by this plan.");
        }
        return report;
    }

    /** {@inheritDoc} */
    @Override
    public Map<FusionShapeProfile, FusionCompilationReport> getShapeProfileReports() {
        return shapeProfileReports;
    }

    /** {@inheritDoc} */
    @Override
    public FusionExecutable bind(FusionConstantBindings constants) {
        Objects.requireNonNull(constants, "constants");
        if (constants.getRecipe() != recipe) {
            throw new IllegalArgumentException(
                    "The constant bindings belong to a different fusion recipe.");
        }
        List<NDArray> arrays = constants.getConstants();
        ByteBuffer constantHandles =
                ByteBuffer.allocateDirect(Math.multiplyExact(arrays.size(), Long.BYTES))
                        .order(ByteOrder.nativeOrder());
        for (NDArray array : arrays) {
            if (!(array instanceof PtNDArray)) {
                throw new IllegalArgumentException("Fusion constants must be PyTorch NDArrays.");
            }
            constantHandles.putLong(((PtNDArray) array).getHandle());
        }
        constantHandles.flip();
        long executableHandle = JniUtils.bindFusionPlan(getHandle(), constantHandles);
        try {
            return new PtFusionExecutable(
                    device,
                    recipe,
                    executableHandle,
                    constants,
                    shapeProfiles,
                    shapeProfileCatalog,
                    shapeProfileReports);
        } catch (RuntimeException | Error failure) {
            try {
                JniUtils.deleteFusionExecutable(executableHandle);
            } catch (RuntimeException | Error cleanupFailure) {
                failure.addSuppressed(cleanupFailure);
            }
            throw failure;
        }
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        onClose();
        Long pointer = handle.getAndSet(null);
        if (pointer != null) {
            JniUtils.deleteFusionPlan(pointer);
        }
    }
}
