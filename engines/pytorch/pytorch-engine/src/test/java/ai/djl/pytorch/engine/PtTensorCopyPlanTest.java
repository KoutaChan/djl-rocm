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
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.GradientCollector;

import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.util.ArrayList;
import java.util.List;

/** Tests fixed tensor copy plans, including view identity and autograd behavior. */
@SuppressWarnings("try")
public class PtTensorCopyPlanTest {

    @DataProvider(name = "devicesAndTypes")
    public Object[][] devicesAndTypes() {
        List<Object[]> cases = new ArrayList<>();
        for (int index = -1; index < Math.min(1, Engine.getInstance().getGpuCount()); index++) {
            Device device = index < 0 ? Device.cpu() : Device.gpu(index);
            for (DataType type :
                    new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
                cases.add(new Object[] {device, type});
            }
        }
        return cases.toArray(new Object[0][]);
    }

    @Test(dataProvider = "devicesAndTypes")
    public void copiesIntoExistingViewsAcrossRepeatedUpdates(Device device, DataType type) {
        PtEngine engine = (PtEngine) Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray source = manager.arange(8).toType(type, false);
            NDArray destination = manager.zeros(new Shape(8), type);
            NDList sources = new NDList(source.get("2:6").reshape(2, 2), source.get("0:2"));
            NDList destinations =
                    new NDList(destination.get("0:4").reshape(2, 2), destination.get("6:8"));
            try (PtTensorCopyPlan plan = engine.newTensorCopyPlan(sources, destinations)) {
                for (int update = 0; update < 8; update++) {
                    source.addi(1f);
                    plan.copy();
                    float base = update + 1f;
                    assertValues(
                            destination,
                            new float[] {
                                base + 2, base + 3, base + 4, base + 5, 0, 0, base, base + 1
                            });
                }
            }
        }
    }

    @Test(dataProvider = "devicesAndTypes")
    public void copiesStridedViews(Device device, DataType type) {
        PtEngine engine = (PtEngine) Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray source = manager.arange(8).toType(type, false).reshape(4, 2);
            NDArray destination = manager.zeros(new Shape(4, 2), type);
            try (PtTensorCopyPlan plan =
                    engine.newTensorCopyPlan(
                            new NDList(source.get(":,0")), new NDList(destination.get(":,1")))) {
                plan.copy();
                assertValues(destination, new float[] {0, 0, 0, 2, 0, 4, 0, 6});
            }
        }
    }

    @Test(dataProvider = "devicesAndTypes")
    public void preservesLeafIdentityAndDoesNotRecordCopyInAutograd(Device device, DataType type) {
        PtEngine engine = (PtEngine) Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray source = manager.full(new Shape(3), 2f, type);
            NDArray destination = manager.zeros(new Shape(3), type);
            source.setRequiresGradient(true);
            destination.setRequiresGradient(true);
            try (PtTensorCopyPlan plan =
                    engine.newTensorCopyPlan(new NDList(source), new NDList(destination))) {
                try (GradientCollector collector = engine.newGradientCollector()) {
                    plan.copy();
                    collector.backward(destination.mul(3f).sum());
                }
                assertValues(destination, new float[] {2, 2, 2});
                try (PtNDArray sourceGradient = JniUtils.getGradient((PtNDArray) source);
                        PtNDArray destinationGradient =
                                JniUtils.getGradient((PtNDArray) destination)) {
                    Assert.assertNull(sourceGradient);
                    Assert.assertNotNull(destinationGradient);
                    assertValues(destinationGradient, new float[] {3, 3, 3});
                }
            }
        }
    }

    @Test
    public void rejectsMismatchedPlansAndClosedPlanExecution() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(Device.cpu())) {
            NDArray source = manager.ones(new Shape(2));
            NDArray destination = manager.zeros(new Shape(2));
            Assert.expectThrows(
                    RuntimeException.class,
                    () -> engine.newTensorCopyPlan(new NDList(), new NDList()));
            Assert.expectThrows(
                    RuntimeException.class,
                    () -> engine.newTensorCopyPlan(new NDList(source), new NDList()));
            Assert.expectThrows(
                    RuntimeException.class,
                    () ->
                            engine.newTensorCopyPlan(
                                    new NDList(source),
                                    new NDList(manager.zeros(new Shape(1, 2)))));
            Assert.expectThrows(
                    RuntimeException.class,
                    () ->
                            engine.newTensorCopyPlan(
                                    new NDList(source),
                                    new NDList(manager.zeros(new Shape(2), DataType.FLOAT64))));
            PtTensorCopyPlan plan =
                    engine.newTensorCopyPlan(new NDList(source), new NDList(destination));
            plan.copy();
            plan.close();
            plan.close();
            Assert.assertTrue(plan.isReleased());
            Assert.expectThrows(IllegalStateException.class, plan::copy);
            assertValues(destination, new float[] {1, 1});
        }
    }

    @Test
    public void rejectsCrossDevicePairs() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager cpu = engine.newBaseManager(Device.cpu());
                NDManager gpu = engine.newBaseManager(Device.gpu(0))) {
            Assert.expectThrows(
                    RuntimeException.class,
                    () ->
                            engine.newTensorCopyPlan(
                                    new NDList(cpu.ones(new Shape(2))),
                                    new NDList(gpu.zeros(new Shape(2)))));
        }
    }

    @Test
    public void rejectsChangingTheExecutionStream() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        Device device = Device.gpu(0);
        try (NDManager manager = engine.newBaseManager(device);
                PtStream first = engine.newStream(device);
                PtStream second = engine.newStream(device)) {
            PtTensorCopyPlan plan;
            try (PtStreamScope ignored = first.openScope()) {
                NDArray source = manager.ones(new Shape(2));
                NDArray destination = manager.zeros(new Shape(2));
                plan = engine.newTensorCopyPlan(new NDList(source), new NDList(destination));
                plan.copy();
                assertValues(destination, new float[] {1, 1});
            }
            try (plan) {
                try (PtStreamScope ignored = second.openScope()) {
                    Assert.expectThrows(RuntimeException.class, plan::copy);
                }
                try (PtStreamScope ignored = first.openScope()) {
                    plan.copy();
                }
            }
        }
    }

    private static void assertValues(NDArray array, float[] expected) {
        if (array.getDataType() == DataType.FLOAT32) {
            Assert.assertEquals(array.toFloatArray(), expected, 0f);
        } else {
            try (NDArray floatArray = array.toType(DataType.FLOAT32, false)) {
                Assert.assertEquals(floatArray.toFloatArray(), expected, 0f);
            }
        }
    }
}
