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
import ai.djl.engine.EngineException;
import ai.djl.engine.fusion.FusionCompileConfig;
import ai.djl.engine.fusion.FusionConstantBindings;
import ai.djl.engine.fusion.FusionExecutable;
import ai.djl.engine.fusion.FusionInvocation;
import ai.djl.engine.fusion.FusionOutputLease;
import ai.djl.engine.fusion.FusionPlan;
import ai.djl.engine.fusion.FusionRecipe;
import ai.djl.engine.fusion.FusionSession;
import ai.djl.engine.fusion.FusionSessionConfig;
import ai.djl.engine.fusion.FusionShapeProfile;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDScope;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

import java.nio.ByteBuffer;

public class PtMappedGroupedMaskedSoftmaxPoolTest {

    private static final int MAXIMUM_BATCH = 384;
    private static final int CANDIDATES = 16;
    private static final int GROUPS = 11;
    private static final int WIDTH = 256;
    private static final int[] ALTERNATIVES = {8, 7, 10, -1, 1, 2, 3, -1, 4, 5};
    private static final int[] PASS = {9};

    @Test
    public void descriptorEncodesOneMultiResultCommandAndStorage() {
        Fixture fixture = new Fixture(DataType.FLOAT16, ALTERNATIVES, PASS);
        ByteBuffer descriptor = PtFusionDescriptor.encode(fixture.recipe);
        int commandOffset = Math.toIntExact(descriptor.getLong(14 * Long.BYTES));

        Assert.assertEquals(
                descriptor.getLong((commandOffset + 1) * Long.BYTES),
                PtFusionDescriptor.MAPPED_GROUPED_MASKED_SOFTMAX_POOL_GROUP_V1);
        Assert.assertEquals(descriptor.getLong((commandOffset + 2) * Long.BYTES), 0L);
        Assert.assertEquals(descriptor.getLong((commandOffset + 3) * Long.BYTES), 4L);
        Assert.assertEquals(descriptor.getLong((commandOffset + 4) * Long.BYTES), 5L);
        Assert.assertEquals(descriptor.getLong((commandOffset + 5) * Long.BYTES), 0L);
        Assert.assertEquals(
                descriptor.getLong((commandOffset + 6) * Long.BYTES),
                fixture.alternativeContexts.getValue().getIndex());
        Assert.assertEquals(
                descriptor.getLong((commandOffset + 7) * Long.BYTES),
                fixture.alternativePresence.getValue().getIndex());
        Assert.assertEquals(
                descriptor.getLong((commandOffset + 8) * Long.BYTES),
                fixture.passContext.getValue().getIndex());
        Assert.assertEquals(
                descriptor.getLong((commandOffset + 9) * Long.BYTES),
                fixture.passPresence.getValue().getIndex());
        Assert.assertEquals(PtFusionDescriptor.commandCount(fixture.recipe), 1);
        Assert.assertEquals(PtFusionDescriptor.executableStorageBytes(fixture.recipe), 176L);
        Assert.assertEquals(PtFusionDescriptor.persistentStorageBytes(fixture.recipe), 4_333_824L);
        Assert.assertEquals(PtFusionDescriptor.workspaceBytes(fixture.recipe), 0L);
    }

    @Test
    public void rocmParityCoversBothDevicesTypesAndProductionBatches() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch ROCm device.");
        }
        int deviceCount = Math.min(2, engine.getGpuCount());
        int[] batches = {1, 31, 256, 384};
        for (int deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex) {
            Device device = Device.gpu(deviceIndex);
            for (DataType dataType :
                    new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
                Fixture fixture = new Fixture(dataType, ALTERNATIVES, PASS);
                try (NDManager manager = engine.newBaseManager(device);
                        NDArray alternativeMapping = manager.create(ALTERNATIVES);
                        NDArray passMapping = manager.create(PASS);
                        FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                        FusionExecutable executable =
                                plan.bind(fixture.bindings(alternativeMapping, passMapping));
                        FusionSession session =
                                executable.newSession(
                                        manager,
                                        FusionSessionConfig.builder()
                                                .optOutputSlotCount(2)
                                                .build());
                        Inputs inputs = new Inputs(manager, dataType)) {
                    for (int batch : batches) {
                        try (FusionOutputLease lease = fixture.submit(session, inputs, batch)) {
                            lease.synchronize();
                            assertOutputShapes(fixture, lease, batch, dataType);
                            assertParity(fixture, lease, inputs, batch, dataType, deviceIndex);
                        }
                    }
                }
            }
        }
    }

    @Test
    public void rocmTwoSlotsRetainBorrowedInputsAcrossChangingExtents() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch ROCm device.");
        }
        Device device = Device.gpu(0);
        Fixture fixture = new Fixture(DataType.FLOAT32, ALTERNATIVES, PASS);
        try (NDManager sessionManager = engine.newBaseManager(device);
                NDManager firstInputManager = engine.newBaseManager(device);
                NDManager secondInputManager = engine.newBaseManager(device);
                NDArray alternativeMapping = sessionManager.create(ALTERNATIVES);
                NDArray passMapping = sessionManager.create(PASS);
                FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                FusionExecutable executable =
                        plan.bind(fixture.bindings(alternativeMapping, passMapping));
                FusionSession session =
                        executable.newSession(
                                sessionManager,
                                FusionSessionConfig.builder().optOutputSlotCount(2).build());
                Inputs first = new Inputs(firstInputManager, DataType.FLOAT32);
                Inputs second = new Inputs(secondInputManager, DataType.FLOAT32)) {
            exercisePair(fixture, session, first, 1, second, 384);
            exercisePair(fixture, session, first, 31, second, 256);
            try (FusionOutputLease finalLease = fixture.submit(session, first, 1)) {
                finalLease.synchronize();
                assertOutputShapes(fixture, finalLease, 1, DataType.FLOAT32);
            }
            Assert.assertTrue(Float.isFinite(first.scores.getFloat(0, 0)));
            Assert.assertTrue(Float.isFinite(second.values.getFloat(0, 0, 0)));
        }
    }

    @Test
    public void rocmCapacityProfileKeepsMappedOutputOffsetsCapacityRelative() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch ROCm device.");
        }
        Device device = Device.gpu(0);
        int capacity = 31;
        Fixture fixture = new Fixture(DataType.FLOAT32, ALTERNATIVES, PASS);
        FusionShapeProfile profile =
                FusionShapeProfile.builder(fixture.recipe)
                        .setCapacity(fixture.batch, capacity)
                        .build();
        FusionCompileConfig compileConfig =
                FusionCompileConfig.builder().addShapeProfile(profile).build();
        try (NDManager manager = engine.newBaseManager(device);
                NDArray alternativeMapping = manager.create(ALTERNATIVES);
                NDArray passMapping = manager.create(PASS);
                FusionPlan plan =
                        engine.newFusionCompiler(device).prepare(fixture.recipe, compileConfig);
                FusionExecutable executable =
                        plan.bind(fixture.bindings(alternativeMapping, passMapping));
                FusionSession session =
                        executable.newSession(
                                manager,
                                FusionSessionConfig.builder()
                                        .optRequestedShapeProfile(profile)
                                        .optProfileFallback(
                                                FusionSessionConfig.ProfileFallback.EXACT)
                                        .build());
                Inputs inputs = new Inputs(manager, DataType.FLOAT32);
                FusionOutputLease lease = fixture.submit(session, inputs, capacity)) {
            lease.synchronize();
            Assert.assertEquals(session.getCapacity(fixture.batch), capacity);
            Assert.assertEquals(
                    lease.get(fixture.alternativeContexts).getShape(),
                    new Shape(capacity, fixture.firstMapping.length, WIDTH));
            Assert.assertEquals(
                    lease.get(fixture.alternativePresence).getShape(),
                    new Shape(capacity, fixture.firstMapping.length));
            Assert.assertEquals(
                    lease.get(fixture.passContext).getShape(),
                    new Shape(capacity, fixture.secondMapping.length, WIDTH));
            Assert.assertEquals(
                    lease.get(fixture.passPresence).getShape(),
                    new Shape(capacity, fixture.secondMapping.length));
            assertParity(fixture, lease, inputs, capacity, DataType.FLOAT32, 0);
        }
    }

    @Test
    public void rocmBindValidatesMappingsAndAcceptsDuplicatesAndAllZeroSets() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch ROCm device.");
        }
        Device device = Device.gpu(0);
        int[] duplicates = {2, 2};
        int[] allZero = {-1, -1, -1};
        Fixture accepted = new Fixture(DataType.FLOAT32, duplicates, allZero);
        try (NDManager manager = engine.newBaseManager(device);
                NDArray duplicateMapping = manager.create(duplicates);
                NDArray zeroMapping = manager.create(allZero);
                FusionPlan plan = engine.newFusionCompiler(device).prepare(accepted.recipe);
                FusionExecutable executable =
                        plan.bind(accepted.bindings(duplicateMapping, zeroMapping));
                FusionSession session =
                        executable.newSession(manager, FusionSessionConfig.defaults());
                Inputs inputs = new Inputs(manager, DataType.FLOAT32);
                FusionOutputLease lease = accepted.submit(session, inputs, 31)) {
            lease.synchronize();
            assertParity(accepted, lease, inputs, 31, DataType.FLOAT32, 0);
        }

        Fixture invalid = new Fixture(DataType.FLOAT32, new int[] {GROUPS}, PASS);
        try (NDManager manager = engine.newBaseManager(device);
                NDArray invalidMapping = manager.create(new int[] {GROUPS});
                NDArray passMapping = manager.create(PASS);
                FusionPlan plan = engine.newFusionCompiler(device).prepare(invalid.recipe)) {
            Assert.assertThrows(
                    EngineException.class,
                    () -> plan.bind(invalid.bindings(invalidMapping, passMapping)));
        }
    }

    private static void exercisePair(
            Fixture fixture,
            FusionSession session,
            Inputs firstInputs,
            int firstBatch,
            Inputs secondInputs,
            int secondBatch) {
        FusionOutputLease first = fixture.submit(session, firstInputs, firstBatch);
        FusionOutputLease second = fixture.submit(session, secondInputs, secondBatch);
        try {
            second.synchronize();
            first.synchronize();
            assertOutputShapes(fixture, second, secondBatch, DataType.FLOAT32);
            assertOutputShapes(fixture, first, firstBatch, DataType.FLOAT32);
        } finally {
            second.close();
            first.close();
        }
    }

    private static void assertOutputShapes(
            Fixture fixture, FusionOutputLease lease, int activeBatch, DataType presenceDataType) {
        Assert.assertEquals(lease.getDimension(fixture.batch), activeBatch);
        Assert.assertEquals(
                lease.get(fixture.alternativeContexts).getShape(),
                new Shape(MAXIMUM_BATCH, fixture.firstMapping.length, WIDTH));
        Assert.assertEquals(
                lease.get(fixture.alternativePresence).getShape(),
                new Shape(MAXIMUM_BATCH, fixture.firstMapping.length));
        Assert.assertEquals(
                lease.get(fixture.passContext).getShape(),
                new Shape(MAXIMUM_BATCH, fixture.secondMapping.length, WIDTH));
        Assert.assertEquals(
                lease.get(fixture.passPresence).getShape(),
                new Shape(MAXIMUM_BATCH, fixture.secondMapping.length));
        Assert.assertEquals(lease.get(fixture.alternativeContexts).getDataType(), DataType.FLOAT32);
        Assert.assertEquals(lease.get(fixture.alternativePresence).getDataType(), presenceDataType);
    }

    private static void assertParity(
            Fixture fixture,
            FusionOutputLease lease,
            Inputs inputs,
            int batch,
            DataType dataType,
            int deviceIndex) {
        try (NDScope scope = new NDScope()) {
            scope.suppressNotUsedWarning();
            NDArray scores = inputs.scores.get("0:" + batch);
            NDArray masks = inputs.masks.get("0:" + batch);
            NDArray values = inputs.values.get("0:" + batch);
            NDArray grouped = NDArrays.groupedMaskedSoftmaxPool(scores, masks, values);
            NDArray presenceByGroup = masks.neq(0).sum(new int[] {1}).gt(0).toType(dataType, false);
            NDArray expectedFirstContexts =
                    mappedContexts(grouped, values, fixture.firstMapping, batch);
            NDArray expectedFirstPresence =
                    mappedPresence(presenceByGroup, scores, fixture.firstMapping, batch, dataType);
            NDArray expectedSecondContexts =
                    mappedContexts(grouped, values, fixture.secondMapping, batch);
            NDArray expectedSecondPresence =
                    mappedPresence(presenceByGroup, scores, fixture.secondMapping, batch, dataType);
            float tolerance =
                    dataType == DataType.FLOAT32
                            ? 2.0e-5f
                            : dataType == DataType.FLOAT16 ? 2.0e-3f : 1.0e-2f;
            assertClose(
                    lease.get(fixture.alternativeContexts).get("0:" + batch),
                    expectedFirstContexts,
                    tolerance,
                    deviceIndex,
                    dataType,
                    batch,
                    "first-contexts");
            assertClose(
                    lease.get(fixture.passContext).get("0:" + batch),
                    expectedSecondContexts,
                    tolerance,
                    deviceIndex,
                    dataType,
                    batch,
                    "second-contexts");
            Assert.assertEquals(
                    lease.get(fixture.alternativePresence)
                            .get("0:" + batch)
                            .toType(DataType.FLOAT32, false)
                            .toFloatArray(),
                    expectedFirstPresence.toType(DataType.FLOAT32, false).toFloatArray());
            Assert.assertEquals(
                    lease.get(fixture.passPresence)
                            .get("0:" + batch)
                            .toType(DataType.FLOAT32, false)
                            .toFloatArray(),
                    expectedSecondPresence.toType(DataType.FLOAT32, false).toFloatArray());
        }
    }

    private static NDArray mappedContexts(
            NDArray grouped, NDArray values, int[] mapping, int batch) {
        NDList destinations = new NDList(mapping.length);
        for (int sourceGroup : mapping) {
            destinations.add(
                    sourceGroup < 0
                            ? values.get("0:" + batch + ",0,:").zerosLike()
                            : grouped.get(sourceGroup));
        }
        return NDArrays.stack(destinations, 1);
    }

    private static NDArray mappedPresence(
            NDArray presenceByGroup, NDArray scores, int[] mapping, int batch, DataType dataType) {
        NDList destinations = new NDList(mapping.length);
        for (int sourceGroup : mapping) {
            destinations.add(
                    sourceGroup < 0
                            ? scores.get("0:" + batch + ",0").zerosLike().toType(dataType, false)
                            : presenceByGroup.get(":," + sourceGroup));
        }
        return NDArrays.stack(destinations, 1);
    }

    private static void assertClose(
            NDArray actual,
            NDArray expected,
            float tolerance,
            int deviceIndex,
            DataType dataType,
            int batch,
            String output) {
        float[] actualValues = actual.toFloatArray();
        float[] expectedValues = expected.toFloatArray();
        Assert.assertEquals(actualValues.length, expectedValues.length);
        float maximum = 0.0f;
        double total = 0.0;
        for (int index = 0; index < actualValues.length; ++index) {
            float difference = Math.abs(actualValues[index] - expectedValues[index]);
            maximum = Math.max(maximum, difference);
            total += difference;
        }
        double mean = total / actualValues.length;
        System.out.printf(
                "mapped-pool-parity device=%d dtype=%s batch=%d output=%s maxAbs=%.9g"
                        + " meanAbs=%.9g tolerance=%.9g%n",
                deviceIndex, dataType, batch, output, maximum, mean, tolerance);
        Assert.assertTrue(
                maximum <= tolerance,
                "Mapped pool parity exceeded its fixed tolerance: device="
                        + deviceIndex
                        + ", dtype="
                        + dataType
                        + ", batch="
                        + batch
                        + ", output="
                        + output
                        + ", maxAbs="
                        + maximum
                        + ", meanAbs="
                        + mean
                        + ", tolerance="
                        + tolerance);
    }

    private static final class Inputs implements AutoCloseable {

        private final NDArray scores;
        private final NDArray masks;
        private final NDArray values;

        private Inputs(NDManager manager, DataType dataType) {
            scores =
                    manager.arange((float) (MAXIMUM_BATCH * CANDIDATES))
                            .mod(17)
                            .sub(8)
                            .mul(0.125f)
                            .reshape(MAXIMUM_BATCH, CANDIDATES)
                            .toType(dataType, false);
            masks =
                    manager.arange((float) (MAXIMUM_BATCH * CANDIDATES * GROUPS))
                            .mod(5)
                            .eq(0)
                            .reshape(MAXIMUM_BATCH, CANDIDATES, GROUPS)
                            .toType(dataType, false);
            values =
                    manager.arange((float) (MAXIMUM_BATCH * CANDIDATES * WIDTH))
                            .mod(31)
                            .sub(15)
                            .mul(0.03125f)
                            .reshape(MAXIMUM_BATCH, CANDIDATES, WIDTH)
                            .toType(dataType, false);
        }

        @Override
        public void close() {
            values.close();
            masks.close();
            scores.close();
        }
    }

    private static final class Fixture {

        private final int[] firstMapping;
        private final int[] secondMapping;
        private final FusionRecipe.Dimension batch;
        private final FusionRecipe.Input scores;
        private final FusionRecipe.Input masks;
        private final FusionRecipe.Input values;
        private final FusionRecipe.Constant firstMappingConstant;
        private final FusionRecipe.Constant secondMappingConstant;
        private final FusionRecipe.Output alternativeContexts;
        private final FusionRecipe.Output alternativePresence;
        private final FusionRecipe.Output passContext;
        private final FusionRecipe.Output passPresence;
        private final FusionRecipe recipe;

        private Fixture(DataType dataType, int[] firstMapping, int[] secondMapping) {
            this.firstMapping = firstMapping.clone();
            this.secondMapping = secondMapping.clone();
            FusionRecipe.Builder builder = FusionRecipe.builder("mapped-grouped-pool-test");
            batch = builder.addDimension("batch", MAXIMUM_BATCH);
            scores =
                    builder.addInput(
                            "scores", FusionRecipe.TensorSpec.of(dataType, batch, CANDIDATES));
            masks =
                    builder.addInput(
                            "masks",
                            FusionRecipe.TensorSpec.of(dataType, batch, CANDIDATES, GROUPS));
            values =
                    builder.addInput(
                            "values",
                            FusionRecipe.TensorSpec.of(dataType, batch, CANDIDATES, WIDTH));
            firstMappingConstant =
                    builder.addConstant(
                            "firstMapping",
                            FusionRecipe.TensorSpec.fixed(DataType.INT32, firstMapping.length));
            secondMappingConstant =
                    builder.addConstant(
                            "secondMapping",
                            FusionRecipe.TensorSpec.fixed(DataType.INT32, secondMapping.length));
            FusionRecipe.MappedGroupedMaskedSoftmaxPoolGroup group =
                    builder.mappedGroupedMaskedSoftmaxPoolGroup(
                                    "candidateTypes", scores, masks, values)
                            .addOutputSet("first", firstMappingConstant)
                            .addOutputSet("second", secondMappingConstant)
                            .build();
            alternativeContexts =
                    builder.addOutput("firstContexts", group.getOutputSet(0).getContexts());
            alternativePresence =
                    builder.addOutput("firstPresence", group.getOutputSet(0).getPresence());
            passContext = builder.addOutput("secondContexts", group.getOutputSet(1).getContexts());
            passPresence = builder.addOutput("secondPresence", group.getOutputSet(1).getPresence());
            recipe = builder.build();
        }

        private FusionConstantBindings bindings(NDArray first, NDArray second) {
            return FusionConstantBindings.builder(recipe)
                    .bind(firstMappingConstant, first)
                    .bind(secondMappingConstant, second)
                    .build();
        }

        private FusionOutputLease submit(FusionSession session, Inputs inputs, int activeBatch) {
            try (FusionInvocation invocation = session.acquire()) {
                invocation.setInput(scores, inputs.scores);
                invocation.setInput(masks, inputs.masks);
                invocation.setInput(values, inputs.values);
                invocation.setDimension(batch, activeBatch);
                return invocation.submit();
            }
        }
    }
}
