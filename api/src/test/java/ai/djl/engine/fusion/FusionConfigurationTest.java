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
                FusionShapeProfile.builder(handles.recipe).set(handles.rows, 16).build();
        FusionCompileConfig config = FusionCompileConfig.builder().addShapeProfile(profile).build();

        Assert.assertSame(profile.getRecipe(), handles.recipe);
        Assert.assertTrue(profile.hasExtent(handles.rows));
        Assert.assertEquals(profile.getExtent(handles.rows), 16);
        Assert.assertEquals(profile.getExtents(), new long[] {16});
        Assert.assertEquals(config.getFallbackMode(), FusionCompileConfig.FallbackMode.REQUIRED);
        Assert.assertSame(config.getShapeProfiles().get(0), profile);
        Assert.assertThrows(
                UnsupportedOperationException.class, () -> config.getShapeProfiles().clear());

        RecipeHandles foreign = newRecipe("foreign-profile", false);
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> FusionShapeProfile.builder(handles.recipe).set(foreign.rows, 1));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> FusionShapeProfile.builder(handles.recipe).set(handles.rows, 33));
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
                        .optPersistentStorageBytes(256)
                        .optWorkspaceBytes(128)
                        .optNativeOnly(true)
                        .build();
        Assert.assertEquals(report.getBackend(), "test-backend");
        Assert.assertEquals(report.getCommandCount(), 3);
        Assert.assertEquals(report.getPersistentStorageBytes(), 256);
        Assert.assertEquals(report.getWorkspaceBytes(), 128);
        Assert.assertTrue(report.isNativeOnly());

        FusionSessionConfig defaults = FusionSessionConfig.defaults();
        FusionSessionConfig triple = FusionSessionConfig.builder().optBufferCount(3).build();
        Assert.assertEquals(defaults.getBufferCount(), 1);
        Assert.assertEquals(triple.getBufferCount(), 3);
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> FusionSessionConfig.builder().optBufferCount(0));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> FusionCompilationReport.builder("test").optCommandCount(-1));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> FusionCompilationReport.builder("test").optPersistentStorageBytes(-1));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> FusionCompilationReport.builder("test").optWorkspaceBytes(-1));
        Assert.assertThrows(
                IllegalStateException.class,
                () ->
                        FusionCompilationReport.builder("test")
                                .optPersistentStorageBytes(64)
                                .optWorkspaceBytes(128)
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
