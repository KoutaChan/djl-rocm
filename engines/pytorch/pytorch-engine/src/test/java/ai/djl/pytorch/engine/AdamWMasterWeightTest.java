/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
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
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;

import org.testng.Assert;
import org.testng.annotations.Test;

public class AdamWMasterWeightTest {

    @Test
    public void masterWeightMatchesFullPrecisionAdamW() {
        Engine engine = Engine.getInstance();
        verifyMasterWeightUpdate(engine, Device.cpu());
        if (engine.getGpuCount() > 0) {
            verifyMasterWeightUpdate(engine, Device.gpu());
        }
    }

    @Test
    public void rejectsIncompatibleShapesAndDataTypes() {
        try (NDManager manager = Engine.getInstance().newBaseManager(Device.cpu());
                NDArray modelWeight = manager.ones(new Shape(4), DataType.BFLOAT16);
                NDArray masterWeight = manager.ones(new Shape(4), DataType.FLOAT32);
                NDArray gradient = manager.ones(new Shape(4), DataType.FLOAT32);
                NDArray wrongShape = manager.ones(new Shape(3), DataType.FLOAT32);
                NDArray integerWeight = manager.ones(new Shape(4), DataType.INT32);
                NDArray lowPrecisionGradient = manager.ones(new Shape(4), DataType.BFLOAT16)) {
            Optimizer optimizer = optimizer();

            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            optimizer.updateWithMasterWeight(
                                    "model-shape", wrongShape, masterWeight, gradient));
            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            optimizer.updateWithMasterWeight(
                                    "gradient-shape", modelWeight, masterWeight, wrongShape));
            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            optimizer.updateWithMasterWeight(
                                    "model-dtype", integerWeight, masterWeight, gradient));
            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            optimizer.updateWithMasterWeight(
                                    "master-dtype", modelWeight, integerWeight, integerWeight));
            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            optimizer.updateWithMasterWeight(
                                    "gradient-dtype",
                                    modelWeight,
                                    masterWeight,
                                    lowPrecisionGradient));
        }
    }

    @Test
    public void rejectsIncompatibleDevices() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager cpu = engine.newBaseManager(Device.cpu());
                NDManager gpu = engine.newBaseManager(Device.gpu());
                NDArray cpuModelWeight = cpu.ones(new Shape(4), DataType.BFLOAT16);
                NDArray cpuGradient = cpu.ones(new Shape(4), DataType.FLOAT32);
                NDArray gpuModelWeight = gpu.ones(new Shape(4), DataType.BFLOAT16);
                NDArray gpuMasterWeight = gpu.ones(new Shape(4), DataType.FLOAT32);
                NDArray gpuGradient = gpu.ones(new Shape(4), DataType.FLOAT32)) {
            Optimizer optimizer = optimizer();

            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            optimizer.updateWithMasterWeight(
                                    "weight-device", cpuModelWeight, gpuMasterWeight, gpuGradient));
            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            optimizer.updateWithMasterWeight(
                                    "gradient-device",
                                    gpuModelWeight,
                                    gpuMasterWeight,
                                    cpuGradient));
        }
    }

    private static void verifyMasterWeightUpdate(Engine engine, Device device) {
        try (NDManager manager = engine.newBaseManager(device);
                NDArray referenceWeight = manager.create(new float[] {1.0f, -2.0f, 0.5f, 3.0f});
                NDArray masterWeight = referenceWeight.duplicate();
                NDArray modelWeight = masterWeight.toType(DataType.BFLOAT16, false);
                NDArray gradient = manager.create(new float[] {0.2f, -0.4f, 0.8f, -0.1f})) {
            Optimizer reference = optimizer();
            Optimizer mixedPrecision = optimizer();
            for (int step = 0; step < 3; step++) {
                reference.update("weight", referenceWeight, gradient);
                mixedPrecision.updateWithMasterWeight(
                        "weight", modelWeight, masterWeight, gradient);
            }

            assertClose(masterWeight.toFloatArray(), referenceWeight.toFloatArray(), 2e-6f);
            try (NDArray restored = modelWeight.toType(DataType.FLOAT32, false)) {
                assertClose(restored.toFloatArray(), masterWeight.toFloatArray(), 2e-2f);
            }
        }
    }

    private static Optimizer optimizer() {
        return Optimizer.adamW()
                .optLearningRateTracker(Tracker.fixed(1.0e-3f))
                .optWeightDecays(0.01f)
                .build();
    }

    private static void assertClose(float[] actual, float[] expected, float tolerance) {
        Assert.assertEquals(actual.length, expected.length);
        for (int i = 0; i < actual.length; i++) {
            Assert.assertEquals(actual[i], expected[i], tolerance, "index=" + i);
        }
    }
}
