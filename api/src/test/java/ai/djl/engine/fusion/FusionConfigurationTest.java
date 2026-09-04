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
package ai.djl.engine.fusion;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.DataType;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Proxy;

public class FusionConfigurationTest {

    @Test
    public void shapeProfilesAreRecipeScopedAndImmutable() {
        RecipeHandles handles = newRecipe("profile", true);
        FusionShapeProfile profile =
                FusionShapeProfile.builder(handles.recipe).setCapacity(handles.rows, 16).build();
        FusionShapeProfile equalProfile =
                FusionShapeProfile.builder(handles.recipe).setCapacity(handles.rows, 16).build();
        FusionShapeProfile differentProfile =
                FusionShapeProfile.builder(handles.recipe).setCapacity(handles.rows, 8).build();
        FusionCompileConfig config = FusionCompileConfig.builder().addShapeProfile(profile).build();

        Assert.assertSame(profile.getRecipe(), handles.recipe);
        Assert.assertTrue(profile.hasCapacity(handles.rows));
        Assert.assertEquals(profile.getCapacity(handles.rows), 16);
        Assert.assertEquals(profile.getCapacities(), new long[] {16});
        long[] capacities = profile.getCapacities();
        capacities[0] = 1;
        Assert.assertEquals(profile.getCapacity(handles.rows), 16);
        Assert.assertEquals(profile, equalProfile);
        Assert.assertEquals(profile.hashCode(), equalProfile.hashCode());
        Assert.assertNotEquals(profile, differentProfile);
        Assert.assertEquals(config.getFallbackMode(), FusionCompileConfig.FallbackMode.REQUIRED);
        Assert.assertSame(config.getShapeProfiles().get(0), profile);
        Assert.assertThrows(
                UnsupportedOperationException.class, () -> config.getShapeProfiles().clear());

        RecipeHandles foreign = newRecipe("foreign-profile", false);
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> FusionShapeProfile.builder(handles.recipe).setCapacity(foreign.rows, 1));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> FusionShapeProfile.builder(handles.recipe).setCapacity(handles.rows, 33));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> FusionShapeProfile.builder(handles.recipe).setCapacity(handles.rows, 0));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> FusionShapeProfile.builder(handles.recipe).setCapacity(handles.rows, -1));
    }

    @Test
    public void constantBindingsAreCompleteAndIndexed() {
        RecipeHandles handles = newRecipe("bindings", true);
        NDArray array = dummyArray();
        FusionConstantBindings bindings =
                FusionConstantBindings.builder(handles.recipe)
                        .bind(handles.constant, array)
                        .build();

        Assert.assertSame(bindings.getRecipe(), handles.recipe);
        Assert.assertSame(bindings.get(handles.constant), array);
        Assert.assertSame(bindings.getConstants().get(0), array);
        Assert.assertThrows(
                UnsupportedOperationException.class, () -> bindings.getConstants().clear());

        RecipeHandles missing = newRecipe("missing-bindings", true);
        Assert.assertThrows(
                IllegalStateException.class,
                () -> FusionConstantBindings.builder(missing.recipe).build());
        RecipeHandles foreign = newRecipe("foreign-bindings", true);
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> FusionConstantBindings.builder(handles.recipe).bind(foreign.constant, array));
    }

    @Test
    public void reportAndSessionConfigurationValidateBounds() {
        FusionCompilationReport report =
                FusionCompilationReport.builder("test-backend")
                        .optCommandCount(3)
                        .optExecutableStorageBytes(64)
                        .optExecutionStorageBytes(256)
                        .optWorkspaceBytes(128)
                        .optExportedOutputBytes(128)
                        .optArenaBytes(96)
                        .optBackendWorkspaceUpperBoundBytes(32)
                        .optStoragePlannerVersion(2)
                        .optLogicalAllocationCount(7)
                        .optBackingAllocationCount(4)
                        .optAliasViewCount(5)
                        .optInPlaceReuseCount(2)
                        .optNativeOnly(true)
                        .build();
        Assert.assertEquals(report.getBackend(), "test-backend");
        Assert.assertEquals(report.getCommandCount(), 3);
        Assert.assertEquals(report.getExecutableStorageBytes(), 64);
        Assert.assertEquals(report.getExecutionStorageBytes(), 256);
        Assert.assertEquals(report.getWorkspaceBytes(), 128);
        Assert.assertEquals(report.getExportedOutputBytes(), 128);
        Assert.assertEquals(report.getArenaBytes(), 96);
        Assert.assertEquals(report.getBackendWorkspaceUpperBoundBytes(), 32);
        Assert.assertEquals(report.getRequiredExecutionLaneStorageBytes(), 128);
        Assert.assertEquals(report.getRetainedSessionStorageBytes(1), 160);
        Assert.assertEquals(report.getRetainedSessionStorageBytes(3), 480);
        Assert.assertEquals(report.getSessionStorageBytes(1), 288);
        Assert.assertEquals(report.getSessionStorageBytes(3), 608);
        Assert.assertEquals(report.getStoragePlannerVersion(), 2);
        Assert.assertTrue(report.isStoragePlannerEnabled());
        Assert.assertEquals(report.getLogicalAllocationCount(), 7);
        Assert.assertEquals(report.getBackingAllocationCount(), 4);
        Assert.assertEquals(report.getAliasViewCount(), 5);
        Assert.assertEquals(report.getInPlaceReuseCount(), 2);
        Assert.assertTrue(report.isNativeOnly());

        FusionSessionConfig defaults = FusionSessionConfig.defaults();
        RecipeHandles sessionHandles = newRecipe("session-profile", false);
        FusionShapeProfile requestedProfile =
                FusionShapeProfile.builder(sessionHandles.recipe)
                        .setCapacity(sessionHandles.rows, 16)
                        .build();
        FusionSessionConfig triple =
                FusionSessionConfig.builder()
                        .optOutputSlotCount(3)
                        .optRequestedShapeProfile(requestedProfile)
                        .optProfileFallback(FusionSessionConfig.ProfileFallback.EXACT)
                        .build();
        Assert.assertEquals(defaults.getOutputSlotCount(), 1);
        Assert.assertNull(defaults.getRequestedShapeProfile());
        Assert.assertEquals(
                defaults.getProfileFallback(),
                FusionSessionConfig.ProfileFallback.SMALLEST_FITTING_OR_MAXIMUM);
        Assert.assertEquals(triple.getOutputSlotCount(), 3);
        Assert.assertSame(triple.getRequestedShapeProfile(), requestedProfile);
        Assert.assertEquals(triple.getProfileFallback(), FusionSessionConfig.ProfileFallback.EXACT);
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> FusionSessionConfig.builder().optOutputSlotCount(0));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> FusionCompilationReport.builder("test").optCommandCount(-1));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> FusionCompilationReport.builder("test").optExecutableStorageBytes(-1));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> FusionCompilationReport.builder("test").optExecutionStorageBytes(-1));
        Assert.assertThrows(IllegalArgumentException.class, () -> report.getSessionStorageBytes(0));
        Assert.assertThrows(
                IllegalArgumentException.class, () -> report.getRetainedSessionStorageBytes(0));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () ->
                        FusionCompilationReport.builder("test")
                                .optBackendWorkspaceUpperBoundBytes(-1));
        Assert.assertThrows(
                IllegalStateException.class,
                () ->
                        FusionCompilationReport.builder("test")
                                .optExecutionStorageBytes(256)
                                .optWorkspaceBytes(128)
                                .optExportedOutputBytes(128)
                                .optArenaBytes(129)
                                .optBackendWorkspaceUpperBoundBytes(0)
                                .optStoragePlannerVersion(1)
                                .optLogicalAllocationCount(1)
                                .optBackingAllocationCount(1)
                                .optAliasViewCount(0)
                                .optInPlaceReuseCount(0)
                                .build());
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> FusionCompilationReport.builder("test").optWorkspaceBytes(-1));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> FusionCompilationReport.builder("test").optInPlaceReuseCount(-1));
        Assert.assertThrows(
                IllegalStateException.class, () -> FusionCompilationReport.builder("test").build());
        Assert.assertThrows(
                IllegalStateException.class,
                () ->
                        FusionCompilationReport.builder("test")
                                .optExecutionStorageBytes(64)
                                .optWorkspaceBytes(128)
                                .build());
        Assert.assertThrows(
                IllegalStateException.class,
                () ->
                        FusionCompilationReport.builder("test")
                                .optLogicalAllocationCount(1)
                                .optInPlaceReuseCount(2)
                                .build());
    }

    private static RecipeHandles newRecipe(String name, boolean withConstant) {
        FusionRecipe.Builder builder = FusionRecipe.builder(name);
        FusionRecipe.Dimension rows = builder.addDimension("rows", 32);
        FusionRecipe.Input input =
                builder.addInput("input", FusionRecipe.TensorSpec.of(DataType.FLOAT32, rows, 2));
        FusionRecipe.Constant constant =
                withConstant
                        ? builder.addConstant(
                                "constant", FusionRecipe.TensorSpec.fixed(DataType.FLOAT32, 2))
                        : null;
        FusionRecipe.OutputPack packed = builder.outputPack("packed", input);
        builder.addOutput("output", packed);
        return new RecipeHandles(builder.build(), rows, constant);
    }

    private static NDArray dummyArray() {
        return (NDArray)
                Proxy.newProxyInstance(
                        FusionConfigurationTest.class.getClassLoader(),
                        new Class<?>[] {NDArray.class},
                        (proxy, method, args) -> {
                            throw new UnsupportedOperationException(method.getName());
                        });
    }

    private static final class RecipeHandles {

        private FusionRecipe recipe;
        private FusionRecipe.Dimension rows;
        private FusionRecipe.Constant constant;

        private RecipeHandles(
                FusionRecipe recipe, FusionRecipe.Dimension rows, FusionRecipe.Constant constant) {
            this.recipe = recipe;
            this.rows = rows;
            this.constant = constant;
        }
    }
}
