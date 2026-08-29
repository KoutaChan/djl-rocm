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
        Assert.assertEquals(PtFusionDescriptor.ACTIVATION_NONE, 0L);
        Assert.assertEquals(PtFusionDescriptor.ACTIVATION_SILU, 1L);
    }

    @Test
    public void affineSumDescriptorUsesClosedSemanticAttributes() {
        AffineFixture fixture = new AffineFixture(DataType.FLOAT32);
        ByteBuffer descriptor = PtFusionDescriptor.encode(fixture.recipe);
        int commandOffset = Math.toIntExact(descriptor.getLong(14 * Long.BYTES));

        Assert.assertEquals(
                descriptor.getLong((commandOffset + 1) * Long.BYTES),
                PtFusionDescriptor.AFFINE_SUM_V1);
        Assert.assertEquals(descriptor.getLong((commandOffset + 3) * Long.BYTES), 1L);
        Assert.assertEquals(descriptor.getLong((commandOffset + 4) * Long.BYTES), 9L);
        Assert.assertEquals(descriptor.getLong((commandOffset + 5) * Long.BYTES), 3L);
        int firstAttribute = commandOffset + 16;
        Assert.assertEquals(
                descriptor.getLong((firstAttribute + 1) * Long.BYTES),
                PtFusionDescriptor.AFFINE_TERM_COUNT);
        Assert.assertEquals(descriptor.getLong((firstAttribute + 4) * Long.BYTES), 4L);
        int activationAttribute = firstAttribute + PtFusionDescriptor.SCALAR_ATTRIBUTE_WORDS;
        Assert.assertEquals(
                descriptor.getLong((activationAttribute + 1) * Long.BYTES),
                PtFusionDescriptor.AFFINE_ACTIVATION);
        Assert.assertEquals(
                descriptor.getLong((activationAttribute + 4) * Long.BYTES),
                PtFusionDescriptor.ACTIVATION_SILU);
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

        AffineFixture affine = new AffineFixture(DataType.FLOAT32);
        Assert.assertEquals(PtFusionDescriptor.executableStorageBytes(affine.recipe), 84L);
        Assert.assertEquals(PtFusionDescriptor.persistentStorageBytes(affine.recipe), 240L);
        Assert.assertEquals(PtFusionDescriptor.workspaceBytes(affine.recipe), 144L);

        DirectAffineFixture direct = new DirectAffineFixture();
        Assert.assertEquals(PtFusionDescriptor.executableStorageBytes(direct.recipe), 24L);
        Assert.assertEquals(PtFusionDescriptor.persistentStorageBytes(direct.recipe), 96L);
        Assert.assertEquals(PtFusionDescriptor.workspaceBytes(direct.recipe), 0L);

        FixedRuntimeAffineFixture fixedRuntime = new FixedRuntimeAffineFixture();
        Assert.assertEquals(PtFusionDescriptor.executableStorageBytes(fixedRuntime.recipe), 16L);
        Assert.assertEquals(PtFusionDescriptor.persistentStorageBytes(fixedRuntime.recipe), 40L);
        Assert.assertEquals(PtFusionDescriptor.workspaceBytes(fixedRuntime.recipe), 8L);
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

        ByteBuffer affine = PtFusionDescriptor.encode(new AffineFixture(DataType.FLOAT32).recipe);
        int affineWords = affine.capacity() / Long.BYTES;
        int affineCommandOffset = Math.toIntExact(affine.getLong(14 * Long.BYTES));
        int affineOperandCount =
                Math.toIntExact(affine.getLong((affineCommandOffset + 4) * Long.BYTES));
        int affineFirstAttribute = affineCommandOffset + 7 + affineOperandCount;
        ByteBuffer excessiveAttributeCount = copyDescriptor(affine, affineWords);
        excessiveAttributeCount.putLong((affineCommandOffset + 5) * Long.BYTES, Integer.MAX_VALUE);
        assertPreparationFails(Device.gpu(0), excessiveAttributeCount);

        ByteBuffer invalidTermCount = copyDescriptor(affine, affineWords);
        invalidTermCount.putLong((affineFirstAttribute + 4) * Long.BYTES, 0);
        assertPreparationFails(Device.gpu(0), invalidTermCount);

        int affineActivationAttribute =
                affineFirstAttribute + PtFusionDescriptor.SCALAR_ATTRIBUTE_WORDS;
        ByteBuffer invalidActivation = copyDescriptor(affine, affineWords);
        invalidActivation.putLong((affineActivationAttribute + 4) * Long.BYTES, 99);
        assertPreparationFails(Device.gpu(0), invalidActivation);

        ByteBuffer excessiveElementCount = copyDescriptor(valid, totalWords);
        excessiveElementCount.putLong(18 * Long.BYTES, Long.MAX_VALUE);
        assertPreparationFails(Device.gpu(0), excessiveElementCount);

        ByteBuffer excessivePayloadBytes =
                copyDescriptor(multipleCommands, multipleCommands.capacity() / Long.BYTES);
        excessivePayloadBytes.putLong(18 * Long.BYTES, Long.MAX_VALUE / 2);
        assertPreparationFails(Device.gpu(0), excessivePayloadBytes);

        FusionRecipe.Builder excessiveRankBuilder = FusionRecipe.builder("affine-rank-test");
        FusionRecipe.Dimension rows = excessiveRankBuilder.addDimension("rows", 1);
        FusionRecipe.Input excessiveRankInput =
                excessiveRankBuilder.addInput(
                        "input",
                        FusionRecipe.TensorSpec.of(
                                DataType.FLOAT32, rows, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1));
        FusionRecipe.Constant excessiveRankWeight =
                excessiveRankBuilder.addConstant(
                        "weight", FusionRecipe.TensorSpec.fixed(DataType.FLOAT32, 1, 1));
        FusionRecipe.AffineSum excessiveRankValue =
                excessiveRankBuilder
                        .affineSum("affine", 1)
                        .addTerm(excessiveRankInput, excessiveRankWeight)
                        .build();
        excessiveRankBuilder.addOutput("output", excessiveRankValue);
        assertPreparationFails(
                Device.gpu(0), PtFusionDescriptor.encode(excessiveRankBuilder.build()));
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
    public void rocmAffineSumMatchesBroadcastReferenceAcrossDataTypes() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch ROCm device.");
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            AffineFixture fixture = new AffineFixture(dataType);
            Device device = Device.gpu(0);
            try (NDManager manager = engine.newBaseManager(device);
                    NDArray candidate =
                            typed(
                                    manager,
                                    fixture.dataType,
                                    new float[] {1f, 2f, 3f, 4f, -1f, 2f, 0.5f, -2f},
                                    new Shape(2, 2, 2));
                    NDArray tile =
                            typed(
                                    manager,
                                    fixture.dataType,
                                    new float[] {0.5f, -1f, 2f, 1.5f},
                                    new Shape(2, 2, 1));
                    NDArray context =
                            typed(
                                    manager,
                                    fixture.dataType,
                                    new float[] {2f, -1f, 0.25f, 3f},
                                    new Shape(2, 1, 2));
                    NDArray fixed =
                            typed(
                                    manager,
                                    fixture.dataType,
                                    new float[] {1.5f, -0.5f},
                                    new Shape(1, 2, 1));
                    NDArray candidateWeight =
                            typed(
                                    manager,
                                    fixture.dataType,
                                    new float[] {1f, 0.5f, -1f, 2f, 0.25f, -0.75f},
                                    new Shape(3, 2));
                    NDArray tileWeight =
                            typed(
                                    manager,
                                    fixture.dataType,
                                    new float[] {0.5f, 1.5f, -2f},
                                    new Shape(3, 1));
                    NDArray contextWeight =
                            typed(
                                    manager,
                                    fixture.dataType,
                                    new float[] {1f, -0.5f, 0.25f, 1f, -1f, 0.75f},
                                    new Shape(3, 2));
                    NDArray fixedWeight =
                            typed(
                                    manager,
                                    fixture.dataType,
                                    new float[] {0.75f, -1.25f, 0.5f},
                                    new Shape(3, 1));
                    NDArray bias =
                            typed(
                                    manager,
                                    fixture.dataType,
                                    new float[] {0.1f, -0.2f, 0.3f},
                                    new Shape(3));
                    FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                    FusionExecutable executable =
                            plan.bind(
                                    fixture.bindings(
                                            candidateWeight,
                                            tileWeight,
                                            contextWeight,
                                            fixed,
                                            fixedWeight,
                                            bias));
                    FusionSession session =
                            executable.newSession(
                                    manager,
                                    FusionSessionConfig.builder().optBufferCount(2).build());
                    FusionInvocation invocation = session.acquire()) {
                fixture.setInputs(invocation, candidate, tile, context, 2);
                try (FusionOutputLease lease = invocation.submit()) {
                    lease.synchronize();
                    float tolerance = dataType == DataType.FLOAT32 ? 1e-5f : 3e-2f;
                    NDArray output = lease.get(fixture.output);
                    if (dataType == DataType.BFLOAT16) {
                        try (NDArray floatOutput = output.toType(DataType.FLOAT32, false)) {
                            assertAffineReference(floatOutput.toFloatArray(), tolerance);
                        }
                    } else {
                        assertAffineReference(output.toFloatArray(), tolerance);
                    }
                }
            }
        }
    }

    @Test
    public void rocmAffineSumUsesDirectOutputForOneUnbiasedTerm() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch ROCm device.");
        }
        DirectAffineFixture fixture = new DirectAffineFixture();
        Device device = Device.gpu(0);
        try (NDManager manager = engine.newBaseManager(device);
                NDArray input = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(1, 2, 2));
                NDArray weight =
                        manager.create(new float[] {1f, 0f, 0f, 1f, 1f, 1f}, new Shape(3, 2));
                FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                FusionExecutable executable =
                        plan.bind(
                                FusionConstantBindings.builder(fixture.recipe)
                                        .bind(fixture.weight, weight)
                                        .build());
                FusionSession session =
                        executable.newSession(manager, FusionSessionConfig.defaults());
                FusionInvocation invocation = session.acquire()) {
            invocation.setInput(fixture.input, input);
            invocation.setDimension(fixture.rows, 1);
            try (FusionOutputLease lease = invocation.submit()) {
                lease.synchronize();
                float[] actual = lease.get(fixture.output).toFloatArray();
                float[] expected = {1f, 2f, 3f, 3f, 4f, 7f};
                for (int index = 0; index < expected.length; ++index) {
                    Assert.assertEquals(actual[index], expected[index], 1e-5f);
                }
            }
        }
    }

    @Test
    public void rocmAffineSumSeparatesFixedRuntimeAndPrecomputedConstantGroups() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch ROCm device.");
        }
        FixedRuntimeAffineFixture fixture = new FixedRuntimeAffineFixture();
        Device device = Device.gpu(0);
        try (NDManager manager = engine.newBaseManager(device);
                NDArray dynamic = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2, 1));
                NDArray fixedRuntime = manager.create(new float[] {10f, 20f}, new Shape(1, 2, 1));
                NDArray fixedConstant =
                        manager.create(new float[] {100f, 200f}, new Shape(1, 2, 1));
                NDArray secondFixedConstant =
                        manager.create(new float[] {1000f, 2000f}, new Shape(1, 2, 1));
                NDArray dynamicWeight = manager.ones(new Shape(1, 1), DataType.FLOAT32);
                NDArray fixedRuntimeWeight = manager.ones(new Shape(1, 1), DataType.FLOAT32);
                NDArray fixedConstantWeight = manager.ones(new Shape(1, 1), DataType.FLOAT32);
                NDArray secondFixedConstantWeight =
                        manager.ones(new Shape(1, 1), DataType.FLOAT32);
                FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                FusionExecutable executable =
                        plan.bind(
                                FusionConstantBindings.builder(fixture.recipe)
                                        .bind(fixture.dynamicWeight, dynamicWeight)
                                        .bind(fixture.fixedRuntimeWeight, fixedRuntimeWeight)
                                        .bind(fixture.fixedConstant, fixedConstant)
                                        .bind(fixture.fixedConstantWeight, fixedConstantWeight)
                                        .bind(fixture.secondFixedConstant, secondFixedConstant)
                                        .bind(
                                                fixture.secondFixedConstantWeight,
                                                secondFixedConstantWeight)
                                        .build());
                FusionSession session =
                        executable.newSession(
                                manager, FusionSessionConfig.builder().optBufferCount(1).build())) {
            try (FusionInvocation invocation = session.acquire()) {
                invocation.setInput(fixture.dynamic, dynamic);
                invocation.setInput(fixture.fixedRuntime, fixedRuntime);
                invocation.setDimension(fixture.rows, 0);
                try (FusionOutputLease lease = invocation.submit()) {
                    lease.synchronize();
                }
            }
            try (FusionInvocation invocation = session.acquire()) {
                invocation.setInput(fixture.dynamic, dynamic);
                invocation.setInput(fixture.fixedRuntime, fixedRuntime);
                invocation.setDimension(fixture.rows, 2);
                try (FusionOutputLease lease = invocation.submit()) {
                    lease.synchronize();
                    float[] actual = lease.get(fixture.output).toFloatArray();
                    float[] expected = {1111f, 2222f, 1113f, 2224f};
                    for (int index = 0; index < expected.length; ++index) {
                        Assert.assertEquals(actual[index], expected[index], 1e-5f);
                    }
                }
            }
        }
    }

    @Test
    @SuppressWarnings("try")
    public void rocmAffineSumWaitsForBindingAcrossStreams() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch ROCm device.");
        }
        AffineFixture fixture = new AffineFixture(DataType.FLOAT32);
        Device device = Device.gpu(0);
        try (PtNDManager manager = (PtNDManager) engine.newBaseManager(device);
                NDArray candidate = manager.ones(new Shape(1, 2, 2), DataType.FLOAT32);
                NDArray tile = manager.ones(new Shape(1, 2, 1), DataType.FLOAT32);
                NDArray context = manager.ones(new Shape(1, 1, 2), DataType.FLOAT32);
                NDArray fixed = manager.ones(new Shape(1, 2, 1), DataType.FLOAT32);
                NDArray candidateWeight = manager.ones(new Shape(3, 2), DataType.FLOAT32);
                NDArray tileWeight = manager.ones(new Shape(3, 1), DataType.FLOAT32);
                NDArray contextWeight = manager.ones(new Shape(3, 2), DataType.FLOAT32);
                NDArray fixedWeight = manager.ones(new Shape(3, 1), DataType.FLOAT32);
                NDArray bias = manager.zeros(new Shape(3), DataType.FLOAT32);
                FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                PtStream bindingStream = engine.newStream(device);
                PtStream submissionStream = engine.newStream(device)) {
            candidate.toByteBuffer();
            tile.toByteBuffer();
            context.toByteBuffer();
            fixed.toByteBuffer();
            candidateWeight.toByteBuffer();
            tileWeight.toByteBuffer();
            contextWeight.toByteBuffer();
            fixedWeight.toByteBuffer();
            bias.toByteBuffer();

            FusionExecutable executable;
            try (PtStreamScope ignored = bindingStream.openScope()) {
                executable =
                        plan.bind(
                                fixture.bindings(
                                        candidateWeight,
                                        tileWeight,
                                        contextWeight,
                                        fixed,
                                        fixedWeight,
                                        bias));
            }
            try (executable;
                    FusionSession session =
                            executable.newSession(manager, FusionSessionConfig.defaults());
                    PtStreamScope ignored = submissionStream.openScope();
                    FusionInvocation invocation = session.acquire()) {
                fixture.setInputs(invocation, candidate, tile, context, 1);
                try (FusionOutputLease lease = invocation.submit()) {
                    lease.synchronize();
                    float expected = (float) (6.0 / (1.0 + Math.exp(-6.0)));
                    float[] actual = lease.get(fixture.output).toFloatArray();
                    for (int index = 0; index < 6; ++index) {
                        Assert.assertEquals(actual[index], expected, 1e-5f);
                    }
                }
            }
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

    private static NDArray typed(NDManager manager, DataType dataType, float[] data, Shape shape) {
        return manager.create(data, shape).toType(dataType, false);
    }

    private static void assertAffineReference(float[] actual, float tolerance) {
        float[] candidate = {1f, 2f, 3f, 4f, -1f, 2f, 0.5f, -2f};
        float[] tile = {0.5f, -1f, 2f, 1.5f};
        float[] context = {2f, -1f, 0.25f, 3f};
        float[] fixed = {1.5f, -0.5f};
        float[] candidateWeight = {1f, 0.5f, -1f, 2f, 0.25f, -0.75f};
        float[] tileWeight = {0.5f, 1.5f, -2f};
        float[] contextWeight = {1f, -0.5f, 0.25f, 1f, -1f, 0.75f};
        float[] fixedWeight = {0.75f, -1.25f, 0.5f};
        float[] bias = {0.1f, -0.2f, 0.3f};
        int actualIndex = 0;
        for (int batch = 0; batch < 2; ++batch) {
            for (int candidateIndex = 0; candidateIndex < 2; ++candidateIndex) {
                for (int output = 0; output < 3; ++output) {
                    float sum = bias[output];
                    for (int feature = 0; feature < 2; ++feature) {
                        sum +=
                                candidate[(batch * 2 + candidateIndex) * 2 + feature]
                                        * candidateWeight[output * 2 + feature];
                        sum += context[batch * 2 + feature] * contextWeight[output * 2 + feature];
                    }
                    sum += tile[batch * 2 + candidateIndex] * tileWeight[output];
                    sum += fixed[candidateIndex] * fixedWeight[output];
                    float expected = (float) (sum / (1.0 + Math.exp(-sum)));
                    Assert.assertEquals(actual[actualIndex++], expected, tolerance);
                }
            }
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

    private static final class AffineFixture {

        private final DataType dataType;
        private final FusionRecipe.Dimension rows;
        private final FusionRecipe.Input candidate;
        private final FusionRecipe.Input tile;
        private final FusionRecipe.Input context;
        private final FusionRecipe.Constant fixed;
        private final FusionRecipe.Constant candidateWeight;
        private final FusionRecipe.Constant tileWeight;
        private final FusionRecipe.Constant contextWeight;
        private final FusionRecipe.Constant fixedWeight;
        private final FusionRecipe.Constant bias;
        private final FusionRecipe.Output output;
        private final FusionRecipe recipe;

        private AffineFixture(DataType dataType) {
            this.dataType = dataType;
            FusionRecipe.Builder builder = FusionRecipe.builder("affine-sum-test");
            rows = builder.addDimension("rows", 4);
            candidate =
                    builder.addInput("candidate", FusionRecipe.TensorSpec.of(dataType, rows, 2, 2));
            tile = builder.addInput("tile", FusionRecipe.TensorSpec.of(dataType, rows, 2, 1));
            context = builder.addInput("context", FusionRecipe.TensorSpec.of(dataType, rows, 1, 2));
            fixed = builder.addConstant("fixed", FusionRecipe.TensorSpec.fixed(dataType, 1, 2, 1));
            candidateWeight =
                    builder.addConstant(
                            "candidateWeight", FusionRecipe.TensorSpec.fixed(dataType, 3, 2));
            tileWeight =
                    builder.addConstant(
                            "tileWeight", FusionRecipe.TensorSpec.fixed(dataType, 3, 1));
            contextWeight =
                    builder.addConstant(
                            "contextWeight", FusionRecipe.TensorSpec.fixed(dataType, 3, 2));
            fixedWeight =
                    builder.addConstant(
                            "fixedWeight", FusionRecipe.TensorSpec.fixed(dataType, 3, 1));
            bias = builder.addConstant("bias", FusionRecipe.TensorSpec.fixed(dataType, 3));
            FusionRecipe.AffineSum value =
                    builder.affineSum("hidden", 3)
                            .addTerm(candidate, candidateWeight)
                            .addTerm(tile, tileWeight)
                            .addTerm(context, contextWeight)
                            .addTerm(fixed, fixedWeight)
                            .optBias(bias)
                            .optActivation(FusionRecipe.Activation.SILU)
                            .build();
            output = builder.addOutput("output", value);
            recipe = builder.build();
        }

        private FusionConstantBindings bindings(
                NDArray candidateWeightArray,
                NDArray tileWeightArray,
                NDArray contextWeightArray,
                NDArray fixedArray,
                NDArray fixedWeightArray,
                NDArray biasArray) {
            return FusionConstantBindings.builder(recipe)
                    .bind(candidateWeight, candidateWeightArray)
                    .bind(tileWeight, tileWeightArray)
                    .bind(contextWeight, contextWeightArray)
                    .bind(fixed, fixedArray)
                    .bind(fixedWeight, fixedWeightArray)
                    .bind(bias, biasArray)
                    .build();
        }

        private void setInputs(
                FusionInvocation invocation,
                NDArray candidateArray,
                NDArray tileArray,
                NDArray contextArray,
                long extent) {
            invocation.setInput(candidate, candidateArray);
            invocation.setInput(tile, tileArray);
            invocation.setInput(context, contextArray);
            invocation.setDimension(rows, extent);
        }
    }

    private static final class DirectAffineFixture {

        private final FusionRecipe.Dimension rows;
        private final FusionRecipe.Input input;
        private final FusionRecipe.Constant weight;
        private final FusionRecipe.Output output;
        private final FusionRecipe recipe;

        private DirectAffineFixture() {
            FusionRecipe.Builder builder = FusionRecipe.builder("direct-affine-test");
            rows = builder.addDimension("rows", 4);
            input =
                    builder.addInput(
                            "input", FusionRecipe.TensorSpec.of(DataType.FLOAT32, rows, 2, 2));
            weight =
                    builder.addConstant(
                            "weight", FusionRecipe.TensorSpec.fixed(DataType.FLOAT32, 3, 2));
            FusionRecipe.AffineSum value =
                    builder.affineSum("affine", 3).addTerm(input, weight).build();
            output = builder.addOutput("output", value);
            recipe = builder.build();
        }
    }

    private static final class FixedRuntimeAffineFixture {

        private final FusionRecipe.Dimension rows;
        private final FusionRecipe.Input dynamic;
        private final FusionRecipe.Input fixedRuntime;
        private final FusionRecipe.Constant dynamicWeight;
        private final FusionRecipe.Constant fixedRuntimeWeight;
        private final FusionRecipe.Constant fixedConstant;
        private final FusionRecipe.Constant fixedConstantWeight;
        private final FusionRecipe.Constant secondFixedConstant;
        private final FusionRecipe.Constant secondFixedConstantWeight;
        private final FusionRecipe.Output output;
        private final FusionRecipe recipe;

        private FixedRuntimeAffineFixture() {
            FusionRecipe.Builder builder = FusionRecipe.builder("fixed-runtime-affine-test");
            rows = builder.addDimension("rows", 4);
            dynamic =
                    builder.addInput(
                            "dynamic", FusionRecipe.TensorSpec.of(DataType.FLOAT32, rows, 2, 1));
            fixedRuntime =
                    builder.addInput(
                            "fixedRuntime",
                            FusionRecipe.TensorSpec.fixed(DataType.FLOAT32, 1, 2, 1));
            dynamicWeight =
                    builder.addConstant(
                            "dynamicWeight", FusionRecipe.TensorSpec.fixed(DataType.FLOAT32, 1, 1));
            fixedRuntimeWeight =
                    builder.addConstant(
                            "fixedRuntimeWeight",
                            FusionRecipe.TensorSpec.fixed(DataType.FLOAT32, 1, 1));
            fixedConstant =
                    builder.addConstant(
                            "fixedConstant",
                            FusionRecipe.TensorSpec.fixed(DataType.FLOAT32, 1, 2, 1));
            fixedConstantWeight =
                    builder.addConstant(
                            "fixedConstantWeight",
                            FusionRecipe.TensorSpec.fixed(DataType.FLOAT32, 1, 1));
            secondFixedConstant =
                    builder.addConstant(
                            "secondFixedConstant",
                            FusionRecipe.TensorSpec.fixed(DataType.FLOAT32, 1, 2, 1));
            secondFixedConstantWeight =
                    builder.addConstant(
                            "secondFixedConstantWeight",
                            FusionRecipe.TensorSpec.fixed(DataType.FLOAT32, 1, 1));
            FusionRecipe.AffineSum value =
                    builder.affineSum("affine", 1)
                            .addTerm(dynamic, dynamicWeight)
                            .addTerm(fixedRuntime, fixedRuntimeWeight)
                            .addTerm(fixedConstant, fixedConstantWeight)
                            .addTerm(secondFixedConstant, secondFixedConstantWeight)
                            .build();
            output = builder.addOutput("output", value);
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
