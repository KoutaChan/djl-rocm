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
import ai.djl.engine.fusion.FusionConstantBindings;
import ai.djl.engine.fusion.FusionExecutable;
import ai.djl.engine.fusion.FusionInvocation;
import ai.djl.engine.fusion.FusionOutputLease;
import ai.djl.engine.fusion.FusionPlan;
import ai.djl.engine.fusion.FusionRecipe;
import ai.djl.engine.fusion.FusionSession;
import ai.djl.engine.fusion.FusionSessionConfig;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDScope;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.jni.JniUtils;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicReference;

public class PtFusionTest {

    @Test
    public void outputPackDescriptorPreservesMixedInputLayout() {
        FusionFixture fixture = new FusionFixture();
        ByteBuffer descriptor = PtFusionDescriptor.encode(fixture.recipe);

        Assert.assertTrue(descriptor.isDirect());
        Assert.assertEquals(descriptor.order(), ByteOrder.nativeOrder());
        long[] expected = {
            PtFusionDescriptor.MAGIC,
            PtFusionDescriptor.VERSION,
            69,
            0,
            1,
            4,
            3,
            0,
            1,
            1,
            16,
            19,
            51,
            57,
            57,
            67,
            3,
            PtFusionDescriptor.DIMENSION_PREFIX_EXTENT,
            4,
            8,
            0,
            PtFusionDescriptor.DTYPE_FLOAT16,
            PtFusionDescriptor.LAYOUT_CONTIGUOUS,
            0,
            1,
            0,
            2,
            8,
            1,
            PtFusionDescriptor.DTYPE_BFLOAT16,
            PtFusionDescriptor.LAYOUT_CONTIGUOUS,
            0,
            1,
            0,
            1,
            8,
            2,
            PtFusionDescriptor.DTYPE_FLOAT32,
            PtFusionDescriptor.LAYOUT_CONTIGUOUS,
            0,
            1,
            0,
            3,
            8,
            3,
            PtFusionDescriptor.DTYPE_FLOAT32,
            PtFusionDescriptor.LAYOUT_CONTIGUOUS,
            0,
            1,
            0,
            6,
            0,
            0,
            1,
            1,
            2,
            2,
            10,
            PtFusionDescriptor.OUTPUT_PACK_V1,
            0,
            1,
            3,
            0,
            3,
            0,
            1,
            2,
            0,
            3
        };
        Assert.assertEquals(descriptor.remaining(), expected.length * Long.BYTES);
        for (long word : expected) {
            Assert.assertEquals(descriptor.getLong(), word);
        }
        Assert.assertFalse(descriptor.hasRemaining());
    }

    @Test
    public void fusionDescriptorUsesStableDataTypeCodes() {
        Assert.assertEquals(
                PtFusionDescriptor.dtypeCode(DataType.FLOAT16), PtFusionDescriptor.DTYPE_FLOAT16);
        Assert.assertEquals(
                PtFusionDescriptor.dtypeCode(DataType.BFLOAT16), PtFusionDescriptor.DTYPE_BFLOAT16);
        Assert.assertEquals(
                PtFusionDescriptor.dtypeCode(DataType.FLOAT32), PtFusionDescriptor.DTYPE_FLOAT32);
        Assert.assertEquals(
                PtFusionDescriptor.dtypeCode(DataType.BOOLEAN), PtFusionDescriptor.DTYPE_BOOLEAN);
        Assert.assertEquals(
                PtFusionDescriptor.dtypeCode(DataType.UINT8), PtFusionDescriptor.DTYPE_UINT8);
        Assert.assertEquals(
                PtFusionDescriptor.dtypeCode(DataType.INT8), PtFusionDescriptor.DTYPE_INT8);
        Assert.assertEquals(
                PtFusionDescriptor.dtypeCode(DataType.INT16), PtFusionDescriptor.DTYPE_INT16);
        Assert.assertEquals(
                PtFusionDescriptor.dtypeCode(DataType.INT32), PtFusionDescriptor.DTYPE_INT32);
        Assert.assertEquals(
                PtFusionDescriptor.dtypeCode(DataType.INT64), PtFusionDescriptor.DTYPE_INT64);
        Assert.assertEquals(
                PtFusionDescriptor.dtypeCode(DataType.FLOAT64), PtFusionDescriptor.DTYPE_FLOAT64);

        Assert.assertNotEquals(PtFusionDescriptor.DTYPE_FLOAT32, (long) DataType.FLOAT32.ordinal());
        Assert.assertNotEquals(
                PtFusionDescriptor.DTYPE_BFLOAT16, (long) DataType.BFLOAT16.ordinal());
    }

    @Test
    public void fusionDescriptorReportsPersistentStorageBytes() {
        FusionFixture fixture = new FusionFixture();
        Assert.assertEquals(PtFusionDescriptor.persistentStorageBytes(fixture.recipe), 96L);
        Assert.assertEquals(PtFusionDescriptor.workspaceBytes(fixture.recipe), 0L);

        MultiOutputFixture multiOutput = new MultiOutputFixture();
        Assert.assertEquals(PtFusionDescriptor.persistentStorageBytes(multiOutput.recipe), 32L);
        Assert.assertEquals(PtFusionDescriptor.workspaceBytes(multiOutput.recipe), 0L);

        NestedFixture nested = new NestedFixture();
        Assert.assertEquals(PtFusionDescriptor.persistentStorageBytes(nested.recipe), 80L);
        Assert.assertEquals(PtFusionDescriptor.workspaceBytes(nested.recipe), 32L);
    }

    @Test
    public void fusionDescriptorRejectsPersistentStorageOverflow() {
        FusionRecipe.Builder builder = FusionRecipe.builder("storage-overflow-test");
        FusionRecipe.Dimension rows = builder.addDimension("rows", Long.MAX_VALUE);
        FusionRecipe.Input input =
                builder.addInput("input", FusionRecipe.TensorSpec.of(DataType.FLOAT32, rows, 2));
        builder.addOutput("output", builder.outputPack("pack", input));
        FusionRecipe recipe = builder.build();

        Assert.assertThrows(
                ArithmeticException.class, () -> PtFusionDescriptor.persistentStorageBytes(recipe));
    }

    @Test
    public void cpuPreparationFailsBeforeNativeSubmission() {
        FusionFixture fixture = new FusionFixture();
        Assert.assertThrows(
                UnsupportedOperationException.class,
                () -> Engine.getInstance().newFusionCompiler(Device.cpu()).prepare(fixture.recipe));
    }

    @Test
    public void rocmPreparationRejectsMalformedClosedCommandIr() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch ROCm device.");
        }
        FusionFixture fixture = new FusionFixture();
        ByteBuffer valid = PtFusionDescriptor.encode(fixture.recipe);
        int totalWords = valid.capacity() / Long.BYTES;

        assertPreparationFails(Device.gpu(0), copyDescriptor(valid, totalWords - 1));
        assertPreparationFails(Device.gpu(0), copyDescriptor(valid, totalWords + 1));

        ByteBuffer excessiveValueCount = copyDescriptor(valid, totalWords);
        excessiveValueCount.putLong(5 * Long.BYTES, Integer.MAX_VALUE);
        assertPreparationFails(Device.gpu(0), excessiveValueCount);

        ByteBuffer excessiveCommandCount = copyDescriptor(valid, totalWords);
        excessiveCommandCount.putLong(8 * Long.BYTES, Integer.MAX_VALUE);
        assertPreparationFails(Device.gpu(0), excessiveCommandCount);

        ByteBuffer invalidValueLayout = copyDescriptor(valid, totalWords);
        invalidValueLayout.putLong(22 * Long.BYTES, 99);
        assertPreparationFails(Device.gpu(0), invalidValueLayout);

        ByteBuffer invalidTopology = copyDescriptor(valid, totalWords);
        invalidTopology.putLong(64 * Long.BYTES, 3);
        assertPreparationFails(Device.gpu(0), invalidTopology);

        ByteBuffer invalidResultSpec = copyDescriptor(valid, totalWords);
        invalidResultSpec.putLong(50 * Long.BYTES, 5);
        assertPreparationFails(Device.gpu(0), invalidResultSpec);

        ByteBuffer multipleCommands = PtFusionDescriptor.encode(new MultiOutputFixture().recipe);
        int commandOffset = Math.toIntExact(multipleCommands.getLong(14 * Long.BYTES));
        int firstCommandWords =
                Math.toIntExact(multipleCommands.getLong(commandOffset * Long.BYTES));
        int secondCommandOffset = commandOffset + firstCommandWords;
        ByteBuffer duplicateProducer =
                copyDescriptor(multipleCommands, multipleCommands.capacity() / Long.BYTES);
        duplicateProducer.putLong((secondCommandOffset + 6) * Long.BYTES, 2);
        assertPreparationFails(Device.gpu(0), duplicateProducer);

        int outputOffset = Math.toIntExact(multipleCommands.getLong(15 * Long.BYTES));
        ByteBuffer unreachableCommand =
                copyDescriptor(multipleCommands, multipleCommands.capacity() / Long.BYTES);
        unreachableCommand.putLong((outputOffset + 3) * Long.BYTES, 2);
        assertPreparationFails(Device.gpu(0), unreachableCommand);
    }

    @Test
    public void rocmOutputPackUsesPersistentRingStorage() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch ROCm device.");
        }
        FusionFixture fixture = new FusionFixture();
        Device device = Device.gpu(0);
        try (NDManager manager = engine.newBaseManager(device);
                FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                FusionExecutable executable =
                        plan.bind(FusionConstantBindings.builder(fixture.recipe).build());
                FusionSession session =
                        executable.newSession(
                                manager, FusionSessionConfig.builder().optBufferCount(2).build())) {
            NDArray half =
                    manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2))
                            .toType(DataType.FLOAT16, false);
            NDArray bfloat =
                    manager.create(new float[] {5f, 6f}, new Shape(2, 1))
                            .toType(DataType.BFLOAT16, false);
            NDArray single =
                    manager.create(new float[] {7f, 8f, 9f, 10f, 11f, 12f}, new Shape(2, 3));

            FusionOutputLease first;
            try (FusionInvocation invocation = session.acquire()) {
                setInputs(invocation, fixture, half, bfloat, single, 2);
                first = invocation.submit();
            }
            FusionOutputLease second = submit(session, fixture, half, bfloat, single, 2);
            Assert.assertThrows(IllegalStateException.class, session::acquire);
            NDArray firstOutput = first.get(fixture.output);
            Assert.assertEquals(firstOutput.getShape(), new Shape(4, 6));
            assertRows(firstOutput.toFloatArray());
            first.close();

            try (FusionOutputLease reused = submit(session, fixture, half, bfloat, single, 2)) {
                Assert.assertSame(reused.get(fixture.output), firstOutput);
                assertRows(reused.get(fixture.output).toFloatArray());
            }
            second.close();
        }
    }

    @Test
    public void rocmSessionCreationRollsBackForUnavailableManager() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch ROCm device.");
        }
        FusionFixture fixture = new FusionFixture();
        Device device = Device.gpu(0);
        try (FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                FusionExecutable executable =
                        plan.bind(FusionConstantBindings.builder(fixture.recipe).build())) {
            PtNDManager capped = (PtNDManager) engine.newBaseManager(device);
            try {
                capped.cap();
                Assert.assertThrows(
                        IllegalStateException.class,
                        () -> executable.newSession(capped, FusionSessionConfig.defaults()));
            } finally {
                capped.close();
            }

            PtNDManager closed = (PtNDManager) engine.newBaseManager(device);
            closed.close();
            Assert.assertThrows(
                    IllegalStateException.class,
                    () -> executable.newSession(closed, FusionSessionConfig.defaults()));
        }
    }

    @Test
    @SuppressWarnings("try")
    public void rocmSessionWaitsForAllocationAndReusesSlotAcrossStreams() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch ROCm device.");
        }
        FusionFixture fixture = new FusionFixture();
        Device device = Device.gpu(0);
        try (PtNDManager manager = (PtNDManager) engine.newBaseManager(device);
                FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                FusionExecutable executable =
                        plan.bind(FusionConstantBindings.builder(fixture.recipe).build());
                PtStream allocationStream = engine.newStream(device);
                PtStream submissionStream = engine.newStream(device)) {
            NDArray half =
                    manager.create(new float[] {1f, 2f}, new Shape(1, 2))
                            .toType(DataType.FLOAT16, false);
            NDArray bfloat =
                    manager.create(new float[] {3f}, new Shape(1, 1))
                            .toType(DataType.BFLOAT16, false);
            NDArray single = manager.create(new float[] {4f, 5f, 6f}, new Shape(1, 3));
            half.toByteBuffer();
            bfloat.toByteBuffer();
            single.toByteBuffer();

            FusionSession session;
            try (PtStreamScope ignored = allocationStream.openScope()) {
                session = executable.newSession(manager, FusionSessionConfig.defaults());
            }
            try (session) {
                try (PtStreamScope ignored = submissionStream.openScope();
                        FusionInvocation invocation = session.acquire()) {
                    setInputs(invocation, fixture, half, bfloat, single, 1);
                    try (FusionOutputLease lease = invocation.submit()) {
                        lease.synchronize();
                        Assert.assertEquals(
                                lease.get(fixture.output).get(0).toFloatArray(),
                                new float[] {1f, 2f, 3f, 4f, 5f, 6f});
                    }
                }

                try (PtStreamScope ignored = allocationStream.openScope();
                        FusionInvocation invocation = session.acquire()) {
                    setInputs(invocation, fixture, half, bfloat, single, 1);
                    try (FusionOutputLease lease = invocation.submit()) {
                        lease.synchronize();
                        Assert.assertEquals(
                                lease.get(fixture.output).get(0).toFloatArray(),
                                new float[] {1f, 2f, 3f, 4f, 5f, 6f});
                    }
                }
            }
        }
    }

    @Test
    public void rocmDuplicateExportsAliasOnePersistentValue() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch ROCm device.");
        }
        DuplicateOutputFixture fixture = new DuplicateOutputFixture();
        Device device = Device.gpu(0);
        try (NDManager manager = engine.newBaseManager(device);
                FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                FusionExecutable executable =
                        plan.bind(FusionConstantBindings.builder(fixture.recipe).build());
                FusionSession session =
                        executable.newSession(
                                manager, FusionSessionConfig.builder().optBufferCount(1).build());
                NDArray input = manager.create(new float[] {1f, 2f}, new Shape(2, 1));
                FusionInvocation invocation = session.acquire()) {
            invocation.setInput(fixture.input, input);
            invocation.setDimension(fixture.rows, 2);
            try (FusionOutputLease lease = invocation.submit()) {
                lease.synchronize();
                NDArray first = lease.get(fixture.firstOutput);
                NDArray second = lease.get(fixture.secondOutput);
                first.addi(5f);
                Assert.assertEquals(second.toFloatArray()[0], 6f);
                Assert.assertEquals(second.toFloatArray()[1], 7f);
            }
        }
    }

    @Test
    public void rocmNestedOutputPackUsesConstantAndIntermediateStorage() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch ROCm device.");
        }
        NestedFixture fixture = new NestedFixture();
        Device device = Device.gpu(0);
        try (NDManager manager = engine.newBaseManager(device);
                NDArray constant = manager.create(new float[] {9f, 8f, 7f, 6f}, new Shape(4, 1));
                FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                FusionExecutable executable =
                        plan.bind(
                                FusionConstantBindings.builder(fixture.recipe)
                                        .bind(fixture.constant, constant)
                                        .build());
                FusionSession session =
                        executable.newSession(
                                manager, FusionSessionConfig.builder().optBufferCount(1).build());
                NDArray input = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
                FusionInvocation invocation = session.acquire()) {
            invocation.setInput(fixture.input, input);
            invocation.setDimension(fixture.rows, 2);
            try (FusionOutputLease lease = invocation.submit()) {
                lease.synchronize();
                float[] actual = lease.get(fixture.output).toFloatArray();
                float[] expected = {1f, 2f, 9f, 3f, 4f, 8f};
                for (int index = 0; index < expected.length; ++index) {
                    Assert.assertEquals(actual[index], expected[index], 1e-5f);
                }
            }
        }
    }

    @Test
    public void rocmConstantBindingRequiresMaximumShape() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch ROCm device.");
        }
        NestedFixture fixture = new NestedFixture();
        Device device = Device.gpu(0);
        try (NDManager manager = engine.newBaseManager(device);
                NDArray tooShort = manager.create(new float[] {9f, 8f}, new Shape(2, 1));
                FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe)) {
            FusionConstantBindings bindings =
                    FusionConstantBindings.builder(fixture.recipe)
                            .bind(fixture.constant, tooShort)
                            .build();
            Assert.assertThrows(EngineException.class, () -> plan.bind(bindings));
        }
    }

    @Test
    public void rocmOutputPackRejectsInputRowsBelowActiveExtent() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch ROCm device.");
        }
        FusionFixture fixture = new FusionFixture();
        Device device = Device.gpu(0);
        try (NDManager manager = engine.newBaseManager(device);
                FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                FusionExecutable executable =
                        plan.bind(FusionConstantBindings.builder(fixture.recipe).build());
                FusionSession session =
                        executable.newSession(
                                manager, FusionSessionConfig.builder().optBufferCount(1).build());
                NDArray half =
                        manager.create(new float[] {1f, 2f}, new Shape(1, 2))
                                .toType(DataType.FLOAT16, false);
                NDArray bfloat =
                        manager.create(new float[] {5f, 6f}, new Shape(2, 1))
                                .toType(DataType.BFLOAT16, false);
                NDArray single =
                        manager.create(new float[] {7f, 8f, 9f, 10f, 11f, 12f}, new Shape(2, 3));
                FusionInvocation invocation = session.acquire()) {
            setInputs(invocation, fixture, half, bfloat, single, 2);
            Assert.assertThrows(EngineException.class, invocation::submit);
        }
    }

    @Test
    public void rocmOutputPackRejectsInputRowsAbovePreparedMaximum() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch ROCm device.");
        }
        FusionFixture fixture = new FusionFixture();
        Device device = Device.gpu(0);
        try (NDManager manager = engine.newBaseManager(device);
                FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                FusionExecutable executable =
                        plan.bind(FusionConstantBindings.builder(fixture.recipe).build());
                FusionSession session =
                        executable.newSession(
                                manager, FusionSessionConfig.builder().optBufferCount(1).build());
                NDArray half = manager.zeros(new Shape(5, 2), DataType.FLOAT16);
                NDArray bfloat = manager.zeros(new Shape(2, 1), DataType.BFLOAT16);
                NDArray single = manager.zeros(new Shape(2, 3), DataType.FLOAT32);
                FusionInvocation invocation = session.acquire()) {
            setInputs(invocation, fixture, half, bfloat, single, 2);
            Assert.assertThrows(EngineException.class, invocation::submit);
        }
    }

    @Test
    public void rocmOutputPackValidatesEveryCommandBeforeLaunching() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch ROCm device.");
        }
        MultiOutputFixture fixture = new MultiOutputFixture();
        Device device = Device.gpu(0);
        try (NDManager manager = engine.newBaseManager(device);
                FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                FusionExecutable executable =
                        plan.bind(FusionConstantBindings.builder(fixture.recipe).build());
                FusionSession session =
                        executable.newSession(
                                manager, FusionSessionConfig.builder().optBufferCount(1).build())) {
            NDArray zeros = manager.create(new float[] {0f, 0f}, new Shape(2, 1));
            try (FusionInvocation invocation = session.acquire()) {
                fixture.setInputs(invocation, zeros, zeros, 2);
                try (FusionOutputLease lease = invocation.submit()) {
                    Assert.assertEquals(lease.get(fixture.firstOutput).toFloatArray()[0], 0f);
                }
            }

            NDArray nines = manager.create(new float[] {9f, 9f}, new Shape(2, 1));
            NDArray tooShort = manager.create(new float[] {1f}, new Shape(1, 1));
            try (FusionInvocation invocation = session.acquire()) {
                fixture.setInputs(invocation, nines, tooShort, 2);
                Assert.assertThrows(EngineException.class, invocation::submit);
            }

            try (FusionInvocation invocation = session.acquire()) {
                fixture.setInputs(invocation, nines, tooShort, 0);
                try (FusionOutputLease lease = invocation.submit()) {
                    Assert.assertEquals(lease.get(fixture.firstOutput).toFloatArray()[0], 0f);
                }
            }
        }
    }

    @Test
    public void rocmSessionSurvivesNdScopeAndSequentialThreadMigration() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch ROCm device.");
        }
        FusionFixture fixture = new FusionFixture();
        Device device = Device.gpu(0);
        try (NDManager manager = engine.newBaseManager(device);
                FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                FusionExecutable executable =
                        plan.bind(FusionConstantBindings.builder(fixture.recipe).build())) {
            NDArray half =
                    manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2))
                            .toType(DataType.FLOAT16, false);
            NDArray bfloat =
                    manager.create(new float[] {5f, 6f}, new Shape(2, 1))
                            .toType(DataType.BFLOAT16, false);
            NDArray single =
                    manager.create(new float[] {7f, 8f, 9f, 10f, 11f, 12f}, new Shape(2, 3));

            FusionSession session;
            try (NDScope scope = new NDScope()) {
                scope.suppressNotUsedWarning();
                session =
                        executable.newSession(
                                manager, FusionSessionConfig.builder().optBufferCount(1).build());
            }
            try {
                runOnNewThread(
                        () -> {
                            try (FusionInvocation invocation = session.acquire()) {
                                setInputs(invocation, fixture, half, bfloat, single, 2);
                                try (FusionOutputLease lease = invocation.submit()) {
                                    lease.synchronize();
                                    assertRows(lease.get(fixture.output).toFloatArray());
                                }
                            }
                        });
                runOnNewThread(
                        () -> {
                            try (FusionInvocation invocation = session.acquire()) {
                                setInputs(invocation, fixture, half, bfloat, single, 2);
                                try (FusionOutputLease lease = invocation.submit()) {
                                    lease.synchronize();
                                    assertRows(lease.get(fixture.output).toFloatArray());
                                }
                            }
                            session.close();
                        });
            } catch (RuntimeException | Error e) {
                session.close();
                throw e;
            }
        }
    }

    @Test
    public void rocmSingleUseFacadesRejectExpiredGenerations() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch ROCm device.");
        }
        FusionFixture fixture = new FusionFixture();
        Device device = Device.gpu(0);
        try (NDManager manager = engine.newBaseManager(device);
                FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                FusionExecutable executable =
                        plan.bind(FusionConstantBindings.builder(fixture.recipe).build());
                FusionSession session =
                        executable.newSession(
                                manager, FusionSessionConfig.builder().optBufferCount(1).build())) {
            NDArray half =
                    manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2))
                            .toType(DataType.FLOAT16, false);
            NDArray bfloat =
                    manager.create(new float[] {5f, 6f}, new Shape(2, 1))
                            .toType(DataType.BFLOAT16, false);
            NDArray single =
                    manager.create(new float[] {7f, 8f, 9f, 10f, 11f, 12f}, new Shape(2, 3));

            FusionInvocation expiredInvocation = session.acquire();
            setInputs(expiredInvocation, fixture, half, bfloat, single, 2);
            FusionOutputLease expiredLease = expiredInvocation.submit();
            expiredLease.synchronize();
            expiredLease.close();

            FusionInvocation currentInvocation = session.acquire();
            setInputs(currentInvocation, fixture, half, bfloat, single, 2);
            FusionOutputLease currentLease = currentInvocation.submit();
            currentInvocation.close();

            Assert.assertThrows(
                    IllegalStateException.class,
                    () -> expiredInvocation.setDimension(fixture.rows, 2));
            Assert.assertThrows(
                    IllegalStateException.class, () -> expiredLease.get(fixture.output));
            expiredLease.close();
            Assert.assertThrows(IllegalStateException.class, session::acquire);

            currentLease.synchronize();
            currentLease.close();
            expiredInvocation.close();
        }
    }

    private static FusionOutputLease submit(
            FusionSession session,
            FusionFixture fixture,
            NDArray half,
            NDArray bfloat,
            NDArray single,
            long rows) {
        FusionInvocation invocation = session.acquire();
        setInputs(invocation, fixture, half, bfloat, single, rows);
        return invocation.submit();
    }

    private static void setInputs(
            FusionInvocation invocation,
            FusionFixture fixture,
            NDArray half,
            NDArray bfloat,
            NDArray single,
            long rows) {
        invocation.setInput(fixture.half, half);
        invocation.setInput(fixture.bfloat, bfloat);
        invocation.setInput(fixture.single, single);
        invocation.setDimension(fixture.rows, rows);
    }

    private static void assertRows(float[] actual) {
        float[] expected = {
            1f, 2f, 5f, 7f, 8f, 9f,
            3f, 4f, 6f, 10f, 11f, 12f
        };
        for (int i = 0; i < expected.length; ++i) {
            Assert.assertEquals(actual[i], expected[i], 1e-3f);
        }
    }

    private static ByteBuffer copyDescriptor(ByteBuffer source, int words) {
        ByteBuffer copy =
                ByteBuffer.allocateDirect(Math.multiplyExact(words, Long.BYTES))
                        .order(ByteOrder.nativeOrder());
        int copiedWords = Math.min(words, source.capacity() / Long.BYTES);
        for (int index = 0; index < copiedWords; ++index) {
            copy.putLong(index * Long.BYTES, source.getLong(index * Long.BYTES));
        }
        return copy;
    }

    private static void assertPreparationFails(Device device, ByteBuffer descriptor) {
        try {
            long handle = JniUtils.prepareFusionPlan(device, descriptor);
            JniUtils.deleteFusionPlan(handle);
            Assert.fail("Malformed fusion descriptor unexpectedly prepared successfully.");
        } catch (EngineException expected) {
            // expected
        }
    }

    private static void runOnNewThread(Runnable task) {
        AtomicReference<Throwable> failure = new AtomicReference<>();
        Thread thread =
                new Thread(
                        () -> {
                            try {
                                task.run();
                            } catch (Throwable t) {
                                failure.set(t);
                            }
                        },
                        "pt-fusion-test");
        thread.start();
        try {
            thread.join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AssertionError("Interrupted while waiting for the fusion test thread.", e);
        }
        Throwable cause = failure.get();
        if (cause instanceof RuntimeException) {
            throw (RuntimeException) cause;
        }
        if (cause instanceof Error) {
            throw (Error) cause;
        }
        if (cause != null) {
            throw new AssertionError("Fusion test thread failed.", cause);
        }
    }

    private static final class FusionFixture {

        private final FusionRecipe.Dimension rows;
        private final FusionRecipe.Input half;
        private final FusionRecipe.Input bfloat;
        private final FusionRecipe.Input single;
        private final FusionRecipe.Output output;
        private final FusionRecipe recipe;

        private FusionFixture() {
            FusionRecipe.Builder builder = FusionRecipe.builder("output-pack-test");
            rows = builder.addDimension("rows", 4);
            half = builder.addInput("half", FusionRecipe.TensorSpec.of(DataType.FLOAT16, rows, 2));
            bfloat =
                    builder.addInput(
                            "bfloat", FusionRecipe.TensorSpec.of(DataType.BFLOAT16, rows, 1));
            single =
                    builder.addInput(
                            "single", FusionRecipe.TensorSpec.of(DataType.FLOAT32, rows, 3));
            FusionRecipe.OutputPack pack = builder.outputPack("packed", half, bfloat, single);
            output = builder.addOutput("scores", pack);
            recipe = builder.build();
        }
    }

    private static final class MultiOutputFixture {

        private final FusionRecipe.Dimension rows;
        private final FusionRecipe.Input first;
        private final FusionRecipe.Input second;
        private final FusionRecipe.Output firstOutput;
        private final FusionRecipe recipe;

        private MultiOutputFixture() {
            FusionRecipe.Builder builder = FusionRecipe.builder("multi-output-pack-test");
            rows = builder.addDimension("rows", 4);
            first =
                    builder.addInput(
                            "first", FusionRecipe.TensorSpec.of(DataType.FLOAT32, rows, 1));
            second =
                    builder.addInput(
                            "second", FusionRecipe.TensorSpec.of(DataType.FLOAT32, rows, 1));
            firstOutput =
                    builder.addOutput("first-output", builder.outputPack("first-pack", first));
            builder.addOutput("second-output", builder.outputPack("second-pack", second));
            recipe = builder.build();
        }

        private void setInputs(
                FusionInvocation invocation, NDArray firstArray, NDArray secondArray, long extent) {
            invocation.setInput(first, firstArray);
            invocation.setInput(second, secondArray);
            invocation.setDimension(rows, extent);
        }
    }

    private static final class DuplicateOutputFixture {

        private final FusionRecipe.Dimension rows;
        private final FusionRecipe.Input input;
        private final FusionRecipe.Output firstOutput;
        private final FusionRecipe.Output secondOutput;
        private final FusionRecipe recipe;

        private DuplicateOutputFixture() {
            FusionRecipe.Builder builder = FusionRecipe.builder("duplicate-output-test");
            rows = builder.addDimension("rows", 4);
            input =
                    builder.addInput(
                            "input", FusionRecipe.TensorSpec.of(DataType.FLOAT32, rows, 1));
            FusionRecipe.OutputPack pack = builder.outputPack("pack", input);
            firstOutput = builder.addOutput("first", pack);
            secondOutput = builder.addOutput("second", pack);
            recipe = builder.build();
        }
    }

    private static final class NestedFixture {

        private final FusionRecipe.Dimension rows;
        private final FusionRecipe.Input input;
        private final FusionRecipe.Constant constant;
        private final FusionRecipe.Output output;
        private final FusionRecipe recipe;

        private NestedFixture() {
            FusionRecipe.Builder builder = FusionRecipe.builder("nested-output-pack-test");
            rows = builder.addDimension("rows", 4);
            input =
                    builder.addInput(
                            "input", FusionRecipe.TensorSpec.of(DataType.FLOAT32, rows, 2));
            constant =
                    builder.addConstant(
                            "constant", FusionRecipe.TensorSpec.of(DataType.FLOAT32, rows, 1));
            FusionRecipe.OutputPack intermediate = builder.outputPack("intermediate", input);
            output =
                    builder.addOutput(
                            "output", builder.outputPack("result", intermediate, constant));
            recipe = builder.build();
        }
    }
}
