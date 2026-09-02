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
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDScope;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.pytorch.jni.JniUtils;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;
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
    public void indexedAffineDescriptorUsesDivisorVector() {
        IndexedAffineFixture fixture = new IndexedAffineFixture(DataType.FLOAT32);
        ByteBuffer descriptor = PtFusionDescriptor.encode(fixture.recipe);
        int commandOffset = Math.toIntExact(descriptor.getLong(14 * Long.BYTES));

        Assert.assertEquals(
                descriptor.getLong((commandOffset + 1) * Long.BYTES),
                PtFusionDescriptor.INDEXED_AFFINE_V1);
        Assert.assertEquals(descriptor.getLong((commandOffset + 3) * Long.BYTES), 1L);
        Assert.assertEquals(descriptor.getLong((commandOffset + 4) * Long.BYTES), 7L);
        Assert.assertEquals(descriptor.getLong((commandOffset + 5) * Long.BYTES), 5L);
        int firstAttribute = commandOffset + 14;
        Assert.assertEquals(
                descriptor.getLong((firstAttribute + 1) * Long.BYTES),
                PtFusionDescriptor.INDEXED_SOURCE_COUNT);
        Assert.assertEquals(descriptor.getLong((firstAttribute + 4) * Long.BYTES), 2L);
        int divisorsAttribute = firstAttribute + 4 * PtFusionDescriptor.SCALAR_ATTRIBUTE_WORDS;
        Assert.assertEquals(
                descriptor.getLong((divisorsAttribute + 1) * Long.BYTES),
                PtFusionDescriptor.INDEXED_SOURCE_DIVISORS);
        Assert.assertEquals(descriptor.getLong((divisorsAttribute + 3) * Long.BYTES), 2L);
        Assert.assertEquals(descriptor.getLong((divisorsAttribute + 4) * Long.BYTES), 3L);
        Assert.assertEquals(descriptor.getLong((divisorsAttribute + 5) * Long.BYTES), 1L);
    }

    @Test
    public void transformerEncoderDescriptorReportsPhysicalWorkspace() {
        TransformerFixture fixture = new TransformerFixture();
        ByteBuffer descriptor = PtFusionDescriptor.encode(fixture.recipe);
        int commandOffset = Math.toIntExact(descriptor.getLong(14 * Long.BYTES));

        Assert.assertEquals(
                descriptor.getLong((commandOffset + 1) * Long.BYTES),
                PtFusionDescriptor.TRANSFORMER_ENCODER_STACK_V1);
        Assert.assertEquals(descriptor.getLong((commandOffset + 4) * Long.BYTES), 14L);
        Assert.assertEquals(descriptor.getLong((commandOffset + 5) * Long.BYTES), 5L);
        Assert.assertEquals(PtFusionDescriptor.commandCount(fixture.recipe), 1);
        Assert.assertEquals(PtFusionDescriptor.executableStorageBytes(fixture.recipe), 786_432L);
        Assert.assertEquals(PtFusionDescriptor.persistentStorageBytes(fixture.recipe), 33_792L);
        Assert.assertEquals(PtFusionDescriptor.workspaceBytes(fixture.recipe), 27_648L);
    }

    @Test
    public void indexedLocalTransformerDescriptorReportsAliasedTileWorkspace() {
        IndexedLocalTransformerFixture fixture =
                new IndexedLocalTransformerFixture(DataType.FLOAT16, 4096);
        ByteBuffer descriptor = PtFusionDescriptor.encode(fixture.recipe);
        int commandOffset = Math.toIntExact(descriptor.getLong(14 * Long.BYTES));

        Assert.assertEquals(
                descriptor.getLong((commandOffset + 1) * Long.BYTES),
                PtFusionDescriptor.INDEXED_LOCAL_TRANSFORMER_ENCODER_V1);
        Assert.assertEquals(descriptor.getLong((commandOffset + 3) * Long.BYTES), 1L);
        Assert.assertEquals(descriptor.getLong((commandOffset + 4) * Long.BYTES), 17L);
        Assert.assertEquals(descriptor.getLong((commandOffset + 5) * Long.BYTES), 4L);
        Assert.assertEquals(PtFusionDescriptor.commandCount(fixture.recipe), 1);
        Assert.assertEquals(PtFusionDescriptor.executableStorageBytes(fixture.recipe), 262_144L);
        Assert.assertEquals(
                PtFusionDescriptor.persistentStorageBytes(fixture.recipe), 459_276_288L);
        Assert.assertEquals(PtFusionDescriptor.workspaceBytes(fixture.recipe), 216_006_656L);
    }

    @Test
    public void segmentedIndexedLocalTransformerUsesV2WithoutExtraStorage() {
        IndexedLocalTransformerFixture dense =
                new IndexedLocalTransformerFixture(DataType.FLOAT16, 4096);
        IndexedLocalTransformerFixture segmented =
                new IndexedLocalTransformerFixture(DataType.FLOAT16, 4096, true);
        ByteBuffer descriptor = PtFusionDescriptor.encode(segmented.recipe);
        int commandOffset = Math.toIntExact(descriptor.getLong(14 * Long.BYTES));

        Assert.assertEquals(
                descriptor.getLong((commandOffset + 1) * Long.BYTES),
                PtFusionDescriptor.INDEXED_LOCAL_TRANSFORMER_ENCODER_SEGMENTED_V2);
        Assert.assertEquals(descriptor.getLong((commandOffset + 4) * Long.BYTES), 19L);
        Assert.assertEquals(descriptor.getLong((commandOffset + 5) * Long.BYTES), 4L);
        Assert.assertEquals(
                PtFusionDescriptor.executableStorageBytes(segmented.recipe),
                PtFusionDescriptor.executableStorageBytes(dense.recipe));
        Assert.assertEquals(
                PtFusionDescriptor.persistentStorageBytes(segmented.recipe),
                PtFusionDescriptor.persistentStorageBytes(dense.recipe));
        Assert.assertEquals(
                PtFusionDescriptor.workspaceBytes(segmented.recipe),
                PtFusionDescriptor.workspaceBytes(dense.recipe));
    }

    @Test
    public void segmentedOutputPackDescriptorPreservesSourceTypeAndStorage() {
        FusionRecipe.Builder builder = FusionRecipe.builder("segmented-output-pack-descriptor");
        FusionRecipe.Dimension batch = builder.addDimension("batch", 4);
        FusionRecipe.Input first =
                builder.addInput(
                        "first", FusionRecipe.TensorSpec.of(DataType.FLOAT16, batch, 1, 3));
        FusionRecipe.Input second =
                builder.addInput(
                        "second", FusionRecipe.TensorSpec.of(DataType.FLOAT16, batch, 2, 3));
        FusionRecipe.SegmentedOutputPack pack = builder.segmentedOutputPack("pack", first, second);
        builder.addOutput("output", pack);
        FusionRecipe recipe = builder.build();
        ByteBuffer descriptor = PtFusionDescriptor.encode(recipe);
        int commandOffset = Math.toIntExact(descriptor.getLong(14 * Long.BYTES));

        Assert.assertEquals(
                descriptor.getLong((commandOffset + 1) * Long.BYTES),
                PtFusionDescriptor.SEGMENTED_OUTPUT_PACK_V1);
        Assert.assertEquals(descriptor.getLong((commandOffset + 3) * Long.BYTES), 1L);
        Assert.assertEquals(descriptor.getLong((commandOffset + 4) * Long.BYTES), 2L);
        Assert.assertEquals(PtFusionDescriptor.commandCount(recipe), 1);
        Assert.assertEquals(PtFusionDescriptor.persistentStorageBytes(recipe), 72L);
        Assert.assertEquals(PtFusionDescriptor.workspaceBytes(recipe), 0L);
    }

    @Test
    public void binaryBranchBlendDescriptorUsesClosedOperandOrder() {
        BinaryBranchBlendFixture fixture = new BinaryBranchBlendFixture();
        ByteBuffer descriptor = PtFusionDescriptor.encode(fixture.recipe);
        int commandOffset = Math.toIntExact(descriptor.getLong(14 * Long.BYTES));

        Assert.assertEquals(
                descriptor.getLong((commandOffset + 1) * Long.BYTES),
                PtFusionDescriptor.BINARY_BRANCH_BLEND_V1);
        Assert.assertEquals(descriptor.getLong((commandOffset + 2) * Long.BYTES), 0L);
        Assert.assertEquals(descriptor.getLong((commandOffset + 3) * Long.BYTES), 1L);
        Assert.assertEquals(descriptor.getLong((commandOffset + 4) * Long.BYTES), 5L);
        Assert.assertEquals(descriptor.getLong((commandOffset + 5) * Long.BYTES), 0L);
        Assert.assertEquals(
                descriptor.getLong((commandOffset + 6) * Long.BYTES), fixture.blend.getIndex());
        Assert.assertEquals(
                descriptor.getLong((commandOffset + 7) * Long.BYTES),
                fixture.baselineContext.getIndex());
        Assert.assertEquals(
                descriptor.getLong((commandOffset + 8) * Long.BYTES),
                fixture.selectedContext.getIndex());
        Assert.assertEquals(
                descriptor.getLong((commandOffset + 9) * Long.BYTES),
                fixture.selectedLogit.getIndex());
        Assert.assertEquals(
                descriptor.getLong((commandOffset + 10) * Long.BYTES),
                fixture.baselinePresence.getIndex());
        Assert.assertEquals(
                descriptor.getLong((commandOffset + 11) * Long.BYTES),
                fixture.selectedPresence.getIndex());
        Assert.assertEquals(PtFusionDescriptor.commandCount(fixture.recipe), 1);
        Assert.assertEquals(PtFusionDescriptor.persistentStorageBytes(fixture.recipe), 48L);
        Assert.assertEquals(PtFusionDescriptor.workspaceBytes(fixture.recipe), 0L);
    }

    @Test
    public void gpuBinaryBranchBlendMatchesPresenceSemantics() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
        }
        BinaryBranchBlendFixture fixture = new BinaryBranchBlendFixture();
        Device device = Device.gpu(0);
        try (NDManager manager = engine.newBaseManager(device);
                NDArray baseline =
                        typed(
                                manager,
                                DataType.FLOAT16,
                                new float[] {2f, 4f, 6f, 10f, 20f, 30f, 1f, 2f, 3f, 7f, 8f, 9f},
                                new Shape(4, 3));
                NDArray selected =
                        typed(
                                manager,
                                DataType.BFLOAT16,
                                new float[] {6f, 8f, 10f, 40f, 50f, 60f, 4f, 5f, 6f, 1f, 2f, 3f},
                                new Shape(4, 3));
                NDArray selectedLogit =
                        typed(
                                manager,
                                DataType.FLOAT16,
                                new float[] {0f, 20f, -20f, 0f},
                                new Shape(4, 1));
                NDArray baselinePresence =
                        typed(
                                manager,
                                DataType.BFLOAT16,
                                new float[] {1f, 1f, 0f, 0f},
                                new Shape(4, 1));
                NDArray selectedPresence =
                        manager.create(new float[] {1f, 0f, 1f, 0f}, new Shape(4, 1));
                FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                FusionExecutable executable =
                        plan.bind(FusionConstantBindings.builder(fixture.recipe).build());
                FusionSession session =
                        executable.newSession(manager, FusionSessionConfig.defaults());
                FusionInvocation invocation = session.acquire()) {
            invocation.setInput(fixture.baselineContext, baseline);
            invocation.setInput(fixture.selectedContext, selected);
            invocation.setInput(fixture.selectedLogit, selectedLogit);
            invocation.setInput(fixture.baselinePresence, baselinePresence);
            invocation.setInput(fixture.selectedPresence, selectedPresence);
            invocation.setDimension(fixture.rows, 4);
            try (FusionOutputLease lease = invocation.submit()) {
                lease.synchronize();
                Assert.assertEquals(
                        lease.get(fixture.output).toFloatArray(),
                        new float[] {4f, 6f, 8f, 10f, 20f, 30f, 4f, 5f, 6f, 0f, 0f, 0f},
                        1e-3f);
            }
        }
    }

    @Test
    public void singleQueryReadoutDescriptorReportsGroupedWorkspace() {
        PtSingleQueryReadoutTestSupport.SingleQueryReadoutFixture fixture =
                new PtSingleQueryReadoutTestSupport.SingleQueryReadoutFixture(DataType.FLOAT16);
        ByteBuffer descriptor = PtFusionDescriptor.encode(fixture.recipe);
        int commandOffset = Math.toIntExact(descriptor.getLong(14 * Long.BYTES));

        Assert.assertEquals(
                descriptor.getLong((commandOffset + 1) * Long.BYTES),
                PtFusionDescriptor.SINGLE_QUERY_CROSS_ATTENTION_READOUT_GROUP_V1);
        Assert.assertEquals(descriptor.getLong((commandOffset + 3) * Long.BYTES), 2L);
        Assert.assertEquals(descriptor.getLong((commandOffset + 4) * Long.BYTES), 37L);
        Assert.assertEquals(descriptor.getLong((commandOffset + 5) * Long.BYTES), 7L);
        Assert.assertEquals(PtFusionDescriptor.commandCount(fixture.recipe), 1);
        Assert.assertEquals(PtFusionDescriptor.executableStorageBytes(fixture.recipe), 1_590_016L);
        Assert.assertEquals(PtFusionDescriptor.persistentStorageBytes(fixture.recipe), 9_216L);
        Assert.assertEquals(PtFusionDescriptor.workspaceBytes(fixture.recipe), 7_168L);
    }

    @Test
    public void gpuSingleQueryReadoutMatchesMaterializedReferenceAcrossDevicesAndTypes() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
        }
        int deviceCount = Math.min(2, engine.getGpuCount());
        int[] batches = {1, 31, 256, 384};
        DataType[][] dataTypes = {
            {DataType.FLOAT32, DataType.FLOAT32},
            {DataType.FLOAT16, DataType.FLOAT16},
            {DataType.FLOAT16, DataType.FLOAT32},
            {DataType.BFLOAT16, DataType.BFLOAT16},
            {DataType.BFLOAT16, DataType.FLOAT32}
        };
        for (int deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex) {
            Device device = Device.gpu(deviceIndex);
            for (DataType[] types : dataTypes) {
                DataType dataType = types[0];
                DataType memoryDataType = types[1];
                PtSingleQueryReadoutTestSupport.SingleQueryReadoutFixture fixture =
                        new PtSingleQueryReadoutTestSupport.SingleQueryReadoutFixture(
                                dataType, memoryDataType, DataType.FLOAT32, 384, true, 3);
                try (NDManager manager = engine.newBaseManager(device);
                        PtSingleQueryReadoutTestSupport.BoundSingleQueryReadouts bound =
                                fixture.bind(manager);
                        NDArray memory =
                                patternedArray(
                                        manager,
                                        memoryDataType,
                                        new Shape(384, 151, 256),
                                        37,
                                        18,
                                        0.015f);
                        NDArray mask = legalReadoutMask(manager, DataType.FLOAT32, 384, 151);
                        FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                        FusionExecutable executable = plan.bind(bound.bindings);
                        FusionSession session =
                                executable.newSession(
                                        manager,
                                        FusionSessionConfig.builder().optBufferCount(1).build())) {
                    for (int batch : batches) {
                        try (FusionInvocation invocation = session.acquire()) {
                            invocation.setInput(fixture.memory, memory);
                            invocation.setInput(fixture.mask, mask);
                            invocation.setDimension(fixture.batch, batch);
                            try (FusionOutputLease lease = invocation.submit()) {
                                lease.synchronize();
                                assertSingleQueryReadoutReference(
                                        lease.get(fixture.policyOutput),
                                        memory,
                                        mask,
                                        bound.readouts.get(0),
                                        fixture.queryIndex,
                                        batch,
                                        dataType,
                                        deviceIndex,
                                        "policy");
                                assertSingleQueryReadoutReference(
                                        lease.get(fixture.valueOutput),
                                        memory,
                                        mask,
                                        bound.readouts.get(1),
                                        fixture.queryIndex,
                                        batch,
                                        dataType,
                                        deviceIndex,
                                        "value");
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    public void gpuIndexedLocalTransformerMatchesSparseReferenceAcrossDevicesAndTypes() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
        }
        int maximumBatch = 384;
        int[] batches = {1, 31, 256, 384};
        IndexedLocalPattern pattern = indexedLocalPattern(maximumBatch);
        int deviceCount = Math.min(2, engine.getGpuCount());
        for (int deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex) {
            Device device = Device.gpu(deviceIndex);
            for (DataType dataType :
                    new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
                IndexedLocalTransformerFixture fixture =
                        new IndexedLocalTransformerFixture(dataType, maximumBatch);
                try (NDManager manager = engine.newBaseManager(device);
                        BoundIndexedLocalTransformer bound = fixture.bind(manager);
                        NDArray maximumInput =
                                patternedArray(
                                        manager,
                                        dataType,
                                        new Shape(maximumBatch, 4, 29, 256),
                                        61,
                                        30,
                                        0.015f);
                        NDArray maximumMask =
                                manager.create(pattern.mask, new Shape(maximumBatch, 4, 29));
                        NDArray maximumIndices =
                                manager.create(pattern.indices, new Shape(pattern.indices.length));
                        FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                        FusionExecutable executable = plan.bind(bound.bindings);
                        FusionSession session =
                                executable.newSession(
                                        manager,
                                        FusionSessionConfig.builder().optBufferCount(2).build())) {
                    for (int batchCount : batches) {
                        int activeRows = pattern.presentCount(batchCount);
                        try (NDArray input = maximumInput.get("0:" + batchCount);
                                NDArray mask = maximumMask.get("0:" + batchCount);
                                NDArray indices = maximumIndices.get("0:" + activeRows);
                                FusionInvocation invocation = session.acquire()) {
                            invocation.setInput(fixture.input, input);
                            invocation.setInput(fixture.indices, indices);
                            invocation.setDimension(fixture.batch, batchCount);
                            invocation.setDimension(fixture.active, activeRows);
                            try (FusionOutputLease lease = invocation.submit();
                                    NDArray expected =
                                            indexedLocalTransformerReference(
                                                    input, indices, mask, bound)) {
                                lease.synchronize();
                                try (NDArray actual =
                                        lease.get(fixture.output).get("0:" + batchCount)) {
                                    assertIndexedLocalTransformerClose(
                                            actual,
                                            expected,
                                            outputTolerance(dataType),
                                            deviceIndex,
                                            dataType,
                                            batchCount,
                                            activeRows);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    public void gpuSegmentedIndexedLocalTransformerMatchesSparseReferenceAcrossDevicesAndTypes() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
        }
        int maximumBatch = 384;
        int[] batches = {1, 31, 256, 384};
        IndexedLocalPattern pattern = indexedLocalPattern(maximumBatch);
        int deviceCount = Math.min(2, engine.getGpuCount());
        for (int deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex) {
            Device device = Device.gpu(deviceIndex);
            for (DataType dataType :
                    new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
                IndexedLocalTransformerFixture fixture =
                        new IndexedLocalTransformerFixture(dataType, maximumBatch, true);
                try (NDManager manager = engine.newBaseManager(device);
                        BoundIndexedLocalTransformer bound = fixture.bind(manager);
                        NDArray maximumInput =
                                patternedArray(
                                        manager,
                                        dataType,
                                        new Shape(maximumBatch, 4, 29, 256),
                                        61,
                                        30,
                                        0.015f);
                        NDArray maximumPlayer = maximumInput.get(":,:,0:1,:").duplicate();
                        NDArray maximumRiver = maximumInput.get(":,:,1:25,:").duplicate();
                        NDArray maximumMeld = maximumInput.get(":,:,25:29,:").duplicate();
                        NDArray maximumMask =
                                manager.create(pattern.mask, new Shape(maximumBatch, 4, 29));
                        NDArray maximumIndices =
                                manager.create(pattern.indices, new Shape(pattern.indices.length));
                        FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                        FusionExecutable executable = plan.bind(bound.bindings);
                        FusionSession session =
                                executable.newSession(
                                        manager,
                                        FusionSessionConfig.builder().optBufferCount(2).build())) {
                    for (int batchCount : batches) {
                        int activeRows = pattern.presentCount(batchCount);
                        try (NDArray input = maximumInput.get("0:" + batchCount);
                                NDArray player = maximumPlayer.get("0:" + batchCount);
                                NDArray river = maximumRiver.get("0:" + batchCount);
                                NDArray meld = maximumMeld.get("0:" + batchCount);
                                NDArray mask = maximumMask.get("0:" + batchCount);
                                NDArray indices = maximumIndices.get("0:" + activeRows);
                                FusionInvocation invocation = session.acquire()) {
                            NDArray[] inputSegments = {player, river, meld};
                            for (int segment = 0; segment < inputSegments.length; ++segment) {
                                invocation.setInput(
                                        fixture.inputs.get(segment), inputSegments[segment]);
                            }
                            invocation.setInput(fixture.indices, indices);
                            invocation.setDimension(fixture.batch, batchCount);
                            invocation.setDimension(fixture.active, activeRows);
                            try (FusionOutputLease lease = invocation.submit();
                                    NDArray expected =
                                            indexedLocalTransformerReference(
                                                    input, indices, mask, bound)) {
                                lease.synchronize();
                                try (NDArray actual =
                                        lease.get(fixture.output).get("0:" + batchCount)) {
                                    assertIndexedLocalTransformerClose(
                                            actual,
                                            expected,
                                            outputTolerance(dataType),
                                            deviceIndex,
                                            dataType,
                                            batchCount,
                                            activeRows);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    public void gpuTransformerEncoderStackMatchesEagerReference() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
        }
        Device device = Device.gpu(0);
        for (DataType dataType : new DataType[] {DataType.FLOAT32, DataType.FLOAT16}) {
            TransformerFixture fixture = new TransformerFixture(dataType, 2);
            try (NDManager manager = engine.newBaseManager(device);
                    NDArray input =
                            typed(
                                    manager,
                                    dataType,
                                    patternedValues(6 * 256, 29, 14, 0.025f),
                                    new Shape(1, 6, 256));
                    NDArray attentionInputWeight =
                            manager.create(patternedValues(256, 19, 9, 0.01f)).add(1f);
                    NDArray attentionInputBias =
                            manager.create(patternedValues(256, 17, 8, 0.002f));
                    NDArray queryKeyValue =
                            typed(
                                    manager,
                                    dataType,
                                    patternedValues(384 * 256, 31, 15, 0.002f),
                                    new Shape(384, 256));
                    NDArray attentionOutput =
                            typed(
                                    manager,
                                    dataType,
                                    patternedValues(256 * 128, 37, 18, 0.002f),
                                    new Shape(256, 128));
                    NDArray attentionOutputBias =
                            typed(
                                    manager,
                                    dataType,
                                    patternedValues(256, 13, 6, 0.003f),
                                    new Shape(256));
                    NDArray feedForwardInputWeight =
                            manager.create(patternedValues(256, 23, 11, 0.008f)).add(1f);
                    NDArray feedForwardInputBias =
                            manager.create(patternedValues(256, 11, 5, 0.002f));
                    NDArray expansion =
                            typed(
                                    manager,
                                    dataType,
                                    patternedValues(512 * 256, 41, 20, 0.0015f),
                                    new Shape(512, 256));
                    NDArray expansionBias =
                            typed(
                                    manager,
                                    dataType,
                                    patternedValues(512, 17, 8, 0.002f),
                                    new Shape(512));
                    NDArray projection =
                            typed(
                                    manager,
                                    dataType,
                                    patternedValues(256 * 512, 43, 21, 0.0015f),
                                    new Shape(256, 512));
                    NDArray projectionBias =
                            typed(
                                    manager,
                                    dataType,
                                    patternedValues(256, 19, 9, 0.002f),
                                    new Shape(256));
                    NDArray outputWeight =
                            manager.create(patternedValues(256, 29, 14, 0.006f)).add(1f);
                    NDArray outputBias = manager.create(patternedValues(256, 31, 15, 0.002f));
                    NDArray firstBlock =
                            transformerEncoderReference(
                                    input,
                                    attentionInputWeight,
                                    attentionInputBias,
                                    queryKeyValue,
                                    attentionOutput,
                                    attentionOutputBias,
                                    feedForwardInputWeight,
                                    feedForwardInputBias,
                                    expansion,
                                    expansionBias,
                                    projection,
                                    projectionBias,
                                    outputWeight,
                                    outputBias);
                    NDArray expected =
                            transformerEncoderReference(
                                    firstBlock,
                                    attentionInputWeight,
                                    attentionInputBias,
                                    queryKeyValue,
                                    attentionOutput,
                                    attentionOutputBias,
                                    feedForwardInputWeight,
                                    feedForwardInputBias,
                                    expansion,
                                    expansionBias,
                                    projection,
                                    projectionBias,
                                    outputWeight,
                                    outputBias);
                    FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                    FusionExecutable executable =
                            plan.bind(
                                    fixture.bindings(
                                            attentionInputWeight,
                                            attentionInputBias,
                                            queryKeyValue,
                                            attentionOutput,
                                            attentionOutputBias,
                                            feedForwardInputWeight,
                                            feedForwardInputBias,
                                            expansion,
                                            expansionBias,
                                            projection,
                                            projectionBias,
                                            outputWeight,
                                            outputBias));
                    FusionSession session =
                            executable.newSession(manager, FusionSessionConfig.defaults());
                    FusionInvocation invocation = session.acquire()) {
                invocation.setInput(fixture.input, input);
                invocation.setDimension(fixture.batch, 1);
                try (FusionOutputLease lease = invocation.submit();
                        NDArray actual =
                                lease.get(fixture.output)
                                        .get("0:1")
                                        .toType(DataType.FLOAT32, false);
                        NDArray expectedFloat = expected.toType(DataType.FLOAT32, false)) {
                    lease.synchronize();
                    float tolerance = dataType == DataType.FLOAT32 ? 4e-4f : 4e-2f;
                    Assert.assertEquals(actual.getShape(), new Shape(1, 6, 256));
                    Assert.assertEquals(
                            actual.toFloatArray(), expectedFloat.toFloatArray(), tolerance);
                }
            }
        }
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

        MixedDirectAffineFixture mixedDirect = new MixedDirectAffineFixture(DataType.FLOAT16);
        Assert.assertEquals(PtFusionDescriptor.executableStorageBytes(mixedDirect.recipe), 12L);
        Assert.assertEquals(PtFusionDescriptor.persistentStorageBytes(mixedDirect.recipe), 80L);
        Assert.assertEquals(PtFusionDescriptor.workspaceBytes(mixedDirect.recipe), 32L);

        FixedRuntimeAffineFixture fixedRuntime = new FixedRuntimeAffineFixture();
        Assert.assertEquals(PtFusionDescriptor.executableStorageBytes(fixedRuntime.recipe), 16L);
        Assert.assertEquals(PtFusionDescriptor.persistentStorageBytes(fixedRuntime.recipe), 40L);
        Assert.assertEquals(PtFusionDescriptor.workspaceBytes(fixedRuntime.recipe), 8L);

        IndexedAffineFixture indexed = new IndexedAffineFixture(DataType.FLOAT32);
        Assert.assertEquals(PtFusionDescriptor.executableStorageBytes(indexed.recipe), 32L);
        Assert.assertEquals(PtFusionDescriptor.persistentStorageBytes(indexed.recipe), 120L);
        Assert.assertEquals(PtFusionDescriptor.workspaceBytes(indexed.recipe), 96L);
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
        UnsupportedOperationException failure =
                Assert.expectThrows(
                        UnsupportedOperationException.class,
                        () ->
                                Engine.getInstance()
                                        .newFusionCompiler(Device.cpu())
                                        .prepare(fixture.recipe));
        Assert.assertTrue(failure.getMessage().contains("CUDA or ROCm device"));
    }

    @Test
    public void gpuPreparationRejectsMalformedClosedCommandIr() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
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

        ByteBuffer indexed =
                PtFusionDescriptor.encode(new IndexedAffineFixture(DataType.FLOAT32).recipe);
        int indexedWords = indexed.capacity() / Long.BYTES;
        int indexedCommandOffset = Math.toIntExact(indexed.getLong(14 * Long.BYTES));
        int indexedOperandCount =
                Math.toIntExact(indexed.getLong((indexedCommandOffset + 4) * Long.BYTES));
        int indexedDivisorsAttribute =
                indexedCommandOffset
                        + 7
                        + indexedOperandCount
                        + 4 * PtFusionDescriptor.SCALAR_ATTRIBUTE_WORDS;
        ByteBuffer invalidDivisor = copyDescriptor(indexed, indexedWords);
        invalidDivisor.putLong((indexedDivisorsAttribute + 4) * Long.BYTES, 0);
        assertPreparationFails(Device.gpu(0), invalidDivisor);

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
    public void gpuOutputPackUsesPersistentRingStorage() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
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
    public void gpuOutputPackConvertsIntoConfiguredType() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
        }
        Device device = Device.gpu(0);
        for (DataType outputType :
                new DataType[] {DataType.FLOAT16, DataType.BFLOAT16, DataType.FLOAT32}) {
            FusionRecipe.Builder builder = FusionRecipe.builder("typed-output-pack-test");
            FusionRecipe.Dimension rows = builder.addDimension("rows", 4);
            FusionRecipe.Input halfInput =
                    builder.addInput("half", FusionRecipe.TensorSpec.of(DataType.FLOAT16, rows, 2));
            FusionRecipe.Input singleInput =
                    builder.addInput(
                            "single", FusionRecipe.TensorSpec.of(DataType.FLOAT32, rows, 3));
            FusionRecipe.OutputPack pack =
                    builder.outputPack("pack")
                            .addSource(halfInput)
                            .addSource(singleInput)
                            .optOutputDataType(outputType)
                            .build();
            FusionRecipe.Output output = builder.addOutput("output", pack);
            FusionRecipe recipe = builder.build();

            try (NDManager manager = engine.newBaseManager(device);
                    FusionPlan plan = engine.newFusionCompiler(device).prepare(recipe);
                    FusionExecutable executable =
                            plan.bind(FusionConstantBindings.builder(recipe).build());
                    FusionSession session =
                            executable.newSession(manager, FusionSessionConfig.defaults())) {
                NDArray half =
                        manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2))
                                .toType(DataType.FLOAT16, false);
                NDArray single =
                        manager.create(new float[] {5f, 6f, 7f, 8f, 9f, 10f}, new Shape(2, 3));
                try (FusionInvocation invocation = session.acquire()) {
                    invocation.setInput(halfInput, half);
                    invocation.setInput(singleInput, single);
                    invocation.setDimension(rows, 2);
                    try (FusionOutputLease lease = invocation.submit()) {
                        lease.synchronize();
                        NDArray actual = lease.get(output);
                        Assert.assertEquals(actual.getDataType(), outputType);
                        try (NDArray slice = actual.get("0:2");
                                NDArray comparable = slice.toType(DataType.FLOAT32, true)) {
                            Assert.assertEquals(
                                    comparable.toFloatArray(),
                                    new float[] {1f, 2f, 5f, 6f, 7f, 3f, 4f, 8f, 9f, 10f},
                                    outputType == DataType.FLOAT32 ? 0f : 1e-2f);
                        }
                    }
                }
            }
        }
    }

    @Test
    public void gpuSegmentedOutputPackIsBitExactAcrossTypesAndSlots() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
        }
        Device device = Device.gpu(0);
        for (DataType dataType :
                new DataType[] {DataType.FLOAT16, DataType.BFLOAT16, DataType.FLOAT32}) {
            FusionRecipe.Builder builder = FusionRecipe.builder("segmented-output-pack-test");
            FusionRecipe.Dimension batch = builder.addDimension("batch", 4);
            FusionRecipe.Input roundInput =
                    builder.addInput("round", FusionRecipe.TensorSpec.of(dataType, batch, 1, 3));
            FusionRecipe.Input playerMemoryInput =
                    builder.addInput(
                            "playerMemory", FusionRecipe.TensorSpec.of(dataType, batch, 2, 4, 3));
            FusionRecipe.Input tileInput =
                    builder.addInput("tiles", FusionRecipe.TensorSpec.of(dataType, batch, 3, 3));
            FusionRecipe.SegmentedOutputPack pack =
                    builder.segmentedOutputPack("memory")
                            .addSource(roundInput)
                            .addSourceSlice(playerMemoryInput, 0, 1)
                            .addSource(tileInput)
                            .addSourceSlice(playerMemoryInput, 1, 2)
                            .addSourceSlice(playerMemoryInput, 3, 1)
                            .build();
            FusionRecipe.Output output = builder.addOutput("memory", pack);
            FusionRecipe recipe = builder.build();

            try (NDManager manager = engine.newBaseManager(device);
                    NDArray round =
                            patternedArray(manager, dataType, new Shape(4, 1, 3), 11, 5, 0.125f);
                    NDArray playerMemory =
                            patternedArray(
                                    manager, dataType, new Shape(4, 2, 4, 3), 13, 6, 0.125f);
                    NDArray tiles =
                            patternedArray(manager, dataType, new Shape(4, 3, 3), 17, 8, 0.125f);
                    NDArray expected =
                            NDArrays.concat(
                                    new NDList(
                                            round,
                                            playerMemory.get(":,:,0,:").reshape(4, 2, 3),
                                            tiles,
                                            playerMemory.get(":,:,1:3,:").reshape(4, 4, 3),
                                            playerMemory.get(":,:,3,:").reshape(4, 2, 3)),
                                    1);
                    FusionPlan plan = engine.newFusionCompiler(device).prepare(recipe);
                    FusionExecutable executable =
                            plan.bind(FusionConstantBindings.builder(recipe).build());
                    FusionSession session =
                            executable.newSession(
                                    manager,
                                    FusionSessionConfig.builder().optBufferCount(2).build())) {
                FusionOutputLease firstLease;
                try (FusionInvocation invocation = session.acquire()) {
                    invocation.setInput(roundInput, round);
                    invocation.setInput(playerMemoryInput, playerMemory);
                    invocation.setInput(tileInput, tiles);
                    invocation.setDimension(batch, 4);
                    firstLease = invocation.submit();
                }
                try (firstLease;
                        FusionInvocation invocation = session.acquire()) {
                    invocation.setInput(roundInput, round);
                    invocation.setInput(playerMemoryInput, playerMemory);
                    invocation.setInput(tileInput, tiles);
                    invocation.setDimension(batch, 2);
                    try (FusionOutputLease secondLease = invocation.submit()) {
                        secondLease.synchronize();
                        Assert.assertEquals(
                                secondLease.get(output).get("0:2").toByteBuffer(),
                                expected.get("0:2").toByteBuffer());
                    }
                    firstLease.synchronize();
                    Assert.assertEquals(
                            firstLease.get(output).toByteBuffer(), expected.toByteBuffer());
                }
            }
        }
    }

    @Test
    public void gpuSegmentedOutputPackConvertsMixedSourcesInOneLaunch() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
        }
        Device device = Device.gpu(0);
        FusionRecipe.Builder builder = FusionRecipe.builder("mixed-segmented-output-pack-test");
        FusionRecipe.Dimension batch = builder.addDimension("batch", 4);
        FusionRecipe.Input strategicInput =
                builder.addInput(
                        "strategic", FusionRecipe.TensorSpec.of(DataType.FLOAT16, batch, 6, 3));
        FusionRecipe.Input roundTileInput =
                builder.addInput(
                        "roundTiles", FusionRecipe.TensorSpec.of(DataType.FLOAT32, batch, 5, 3));
        FusionRecipe.SegmentedOutputPack pack =
                builder.segmentedOutputPack("memory")
                        .addSourceSlice(strategicInput, 0, 1)
                        .addSourceSlice(strategicInput, 2, 4)
                        .addSourceSlice(roundTileInput, 1, 4)
                        .optOutputDataType(DataType.FLOAT32)
                        .build();
        FusionRecipe.Output output = builder.addOutput("memory", pack);
        FusionRecipe recipe = builder.build();

        try (NDManager manager = engine.newBaseManager(device);
                NDArray strategic =
                        patternedArray(
                                manager, DataType.FLOAT16, new Shape(4, 6, 3), 11, 5, 0.125f);
                NDArray roundTiles =
                        patternedArray(
                                manager, DataType.FLOAT32, new Shape(4, 5, 3), 17, 8, 0.125f);
                NDArray expected =
                        NDArrays.concat(
                                new NDList(
                                        strategic.get(":,0:1,:").toType(DataType.FLOAT32, false),
                                        strategic.get(":,2:6,:").toType(DataType.FLOAT32, false),
                                        roundTiles.get(":,1:5,:")),
                                1);
                FusionPlan plan = engine.newFusionCompiler(device).prepare(recipe);
                FusionExecutable executable =
                        plan.bind(FusionConstantBindings.builder(recipe).build());
                FusionSession session =
                        executable.newSession(manager, FusionSessionConfig.builder().build())) {
            try (FusionInvocation invocation = session.acquire()) {
                invocation.setInput(strategicInput, strategic);
                invocation.setInput(roundTileInput, roundTiles);
                invocation.setDimension(batch, 4);
                try (FusionOutputLease lease = invocation.submit()) {
                    lease.synchronize();
                    Assert.assertEquals(lease.get(output).toByteBuffer(), expected.toByteBuffer());
                }
            }
        }
    }

    @Test
    public void gpuAffineSumMatchesBroadcastReferenceAcrossDataTypes() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
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
    public void gpuAffineSumConvertsSingleDynamicSourceToProjectionDataType() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
        }
        for (DataType projectionDataType : new DataType[] {DataType.FLOAT16, DataType.BFLOAT16}) {
            MixedDirectAffineFixture fixture = new MixedDirectAffineFixture(projectionDataType);
            Device device = Device.gpu(0);
            try (NDManager manager = engine.newBaseManager(device);
                    NDArray input =
                            manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(1, 2, 2));
                    NDArray weight =
                            typed(
                                    manager,
                                    projectionDataType,
                                    new float[] {1f, 0f, 0f, 1f, 1f, 1f},
                                    new Shape(3, 2));
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
                    try (NDArray floatOutput =
                            lease.get(fixture.output).toType(DataType.FLOAT32, false)) {
                        float[] actual = floatOutput.toFloatArray();
                        float[] expected = {1f, 2f, 3f, 3f, 4f, 7f};
                        for (int index = 0; index < expected.length; ++index) {
                            Assert.assertEquals(actual[index], expected[index], 2e-2f);
                        }
                    }
                }
            }
        }
    }

    @Test
    public void gpuAffineSumConvertsMixedSourcesAndFixedPrecomputeInputs() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
        }
        for (DataType projectionDataType : new DataType[] {DataType.FLOAT16, DataType.BFLOAT16}) {
            MixedSourceAffineFixture fixture = new MixedSourceAffineFixture(projectionDataType);
            Device device = Device.gpu(0);
            try (NDManager manager = engine.newBaseManager(device);
                    NDArray first =
                            manager.create(
                                    new float[] {1f, 2f, 3f, 4f, -1f, 2f, 0.5f, -2f},
                                    new Shape(2, 2, 2));
                    NDArray second =
                            typed(
                                    manager,
                                    DataType.FLOAT16,
                                    new float[] {0.5f, -1f, 2f, 1.5f},
                                    new Shape(2, 2, 1));
                    NDArray fixed = manager.create(new float[] {10f, -4f}, new Shape(1, 2, 1));
                    NDArray firstWeight =
                            typed(
                                    manager,
                                    projectionDataType,
                                    new float[] {1f, 0f, 0f, 1f},
                                    new Shape(2, 2));
                    NDArray secondWeight =
                            typed(
                                    manager,
                                    projectionDataType,
                                    new float[] {2f, -1f},
                                    new Shape(2, 1));
                    NDArray fixedWeight =
                            typed(
                                    manager,
                                    projectionDataType,
                                    new float[] {0.5f, 1.5f},
                                    new Shape(2, 1));
                    NDArray bias =
                            typed(
                                    manager,
                                    projectionDataType,
                                    new float[] {0.25f, -0.5f},
                                    new Shape(2));
                    FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                    FusionExecutable executable =
                            plan.bind(
                                    fixture.bindings(
                                            firstWeight, secondWeight, fixed, fixedWeight, bias));
                    FusionSession session =
                            executable.newSession(manager, FusionSessionConfig.defaults());
                    FusionInvocation invocation = session.acquire()) {
                fixture.setInputs(invocation, first, second, 2);
                try (FusionOutputLease lease = invocation.submit()) {
                    lease.synchronize();
                    try (NDArray floatOutput =
                            lease.get(fixture.output).toType(DataType.FLOAT32, false)) {
                        float[] actual = floatOutput.toFloatArray();
                        float[] expected = {7.25f, 16f, -0.75f, -1.5f, 8.25f, 14.5f, 1.75f, -10f};
                        float tolerance = projectionDataType == DataType.FLOAT16 ? 2e-2f : 8e-2f;
                        for (int index = 0; index < expected.length; ++index) {
                            Assert.assertEquals(actual[index], expected[index], tolerance);
                        }
                    }
                }
            }
        }
    }

    @Test
    public void gpuAffineSumUsesDirectOutputForOneUnbiasedTerm() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
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
    public void gpuAffineSumSeparatesFixedRuntimeAndPrecomputedConstantGroups() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
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
    public void gpuAffineSumWaitsForBindingAcrossStreams() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
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
    public void gpuIndexedAffineGathersProjectsScattersAndClearsAcrossDataTypes() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
        }
        Device device = Device.gpu(0);
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            IndexedAffineFixture fixture = new IndexedAffineFixture(dataType);
            float tolerance = dataType == DataType.FLOAT32 ? 1e-4f : 8e-2f;
            try (NDManager manager = engine.newBaseManager(device);
                    NDArray indices = manager.create(new int[] {0, 2, 4}, new Shape(3));
                    NDArray state = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
                    NDArray branch =
                            typed(
                                    manager,
                                    DataType.FLOAT16,
                                    new float[] {
                                        10f, 20f, 21f, 22f, 30f, 40f,
                                        41f, 42f, 50f, 60f, 61f, 62f
                                    },
                                    new Shape(6, 2));
                    NDArray hiddenWeight =
                            typed(
                                    manager,
                                    dataType,
                                    new float[] {1f, 0f, 1f, 0f, 0f, 1f, 0f, 1f},
                                    new Shape(2, 4));
                    NDArray hiddenBias =
                            typed(manager, dataType, new float[] {0.1f, -0.2f}, new Shape(2));
                    NDArray outputWeight =
                            typed(manager, dataType, new float[] {0.5f, -1f}, new Shape(1, 2));
                    NDArray outputBias =
                            typed(manager, dataType, new float[] {0.25f}, new Shape(1));
                    FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                    FusionExecutable executable =
                            plan.bind(
                                    fixture.bindings(
                                            hiddenWeight, hiddenBias, outputWeight, outputBias));
                    FusionSession session =
                            executable.newSession(
                                    manager,
                                    FusionSessionConfig.builder().optBufferCount(1).build())) {
                try (FusionInvocation invocation = session.acquire()) {
                    fixture.setInputs(invocation, indices, state, branch, 2, 6, 3);
                    try (FusionOutputLease lease = invocation.submit()) {
                        try (NDArray floatOutput =
                                lease.get(fixture.output).toType(DataType.FLOAT32, true)) {
                            float[] actual = floatOutput.toFloatArray();
                            float[] expected = indexedAffineReference();
                            for (int index = 0; index < expected.length; ++index) {
                                Assert.assertEquals(actual[index], expected[index], tolerance);
                            }
                        }
                    }
                }
                try (FusionInvocation invocation = session.acquire()) {
                    fixture.setInputs(invocation, indices, state, branch, 2, 6, 0);
                    try (FusionOutputLease lease = invocation.submit()) {
                        try (NDArray floatOutput =
                                lease.get(fixture.output).toType(DataType.FLOAT32, true)) {
                            float[] actual = floatOutput.toFloatArray();
                            for (int index = 0; index < 6; ++index) {
                                Assert.assertEquals(actual[index], 0f);
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    public void gpuIndexedAffineRoundsSiluBeforeOutputProjection() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
        }
        Device device = Device.gpu(0);
        IndexedAffineFixture fixture = new IndexedAffineFixture(DataType.FLOAT16);
        try (NDManager manager = engine.newBaseManager(device);
                NDArray indices = manager.create(new int[] {0}, new Shape(1));
                NDArray state = manager.zeros(new Shape(1, 2), DataType.FLOAT32);
                NDArray branch = manager.zeros(new Shape(1, 2), DataType.FLOAT16);
                NDArray hiddenWeight = manager.zeros(new Shape(2, 4), DataType.FLOAT16);
                NDArray hiddenBias =
                        typed(
                                manager,
                                DataType.FLOAT16,
                                new float[] {-4.44140625f, -3.25f},
                                new Shape(2));
                NDArray outputWeight =
                        typed(
                                manager,
                                DataType.FLOAT16,
                                new float[] {1.580078125f, -2.630859375f},
                                new Shape(1, 2));
                NDArray outputBias = manager.zeros(new Shape(1), DataType.FLOAT16);
                NDArray eagerActivation = Activation.swish(hiddenBias.reshape(1, 2), 1.0f);
                NDArray eagerOutput =
                        eagerActivation.matMul(outputWeight.transpose()).add(outputBias);
                FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                FusionExecutable executable =
                        plan.bind(
                                fixture.bindings(
                                        hiddenWeight, hiddenBias, outputWeight, outputBias));
                FusionSession session =
                        executable.newSession(manager, FusionSessionConfig.defaults());
                FusionInvocation invocation = session.acquire()) {
            fixture.setInputs(invocation, indices, state, branch, 1, 1, 1);
            try (FusionOutputLease lease = invocation.submit()) {
                lease.synchronize();
                Assert.assertEquals(
                        eagerOutput.toFloatArray(), new float[] {0.237548828125f}, 0.0f);
                try (NDArray actual = lease.get(fixture.output).get("0:1")) {
                    Assert.assertEquals(actual.toFloatArray(), eagerOutput.toFloatArray(), 0.0f);
                }
            }
        }
    }

    @Test
    public void gpuIndexedAffineSupportsWideUnbiasedInt64InvocationReuse() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
        }
        WideIndexedAffineFixture fixture = new WideIndexedAffineFixture();
        Device device = Device.gpu(0);
        try (NDManager manager = engine.newBaseManager(device);
                NDArray firstIndices = manager.create(new long[] {1, 5}, new Shape(2));
                NDArray secondIndices = manager.create(new long[] {2}, new Shape(1));
                NDArray state =
                        typed(manager, DataType.BFLOAT16, new float[] {1f, 2f}, new Shape(2, 1));
                NDArray branch =
                        manager.create(
                                new float[] {10f, 20f, 30f, 40f, 50f, 60f}, new Shape(6, 1));
                NDArray hiddenWeight =
                        manager.create(new float[] {1f, 0f, 0f, 1f}, new Shape(2, 2));
                NDArray outputWeight =
                        manager.create(new float[] {1f, 0f, 0f, 1f}, new Shape(2, 2));
                FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                FusionExecutable executable =
                        plan.bind(fixture.bindings(hiddenWeight, outputWeight));
                FusionSession session =
                        executable.newSession(
                                manager, FusionSessionConfig.builder().optBufferCount(1).build())) {
            try (FusionInvocation invocation = session.acquire()) {
                fixture.setInputs(invocation, firstIndices, state, branch, 2);
                try (FusionOutputLease lease = invocation.submit()) {
                    float[] actual = lease.get(fixture.output).toFloatArray();
                    float[] expected = {
                        0f, 0f, 1f, 20f, 0f, 0f,
                        0f, 0f, 0f, 0f, 2f, 60f
                    };
                    Assert.assertEquals(actual, expected);
                }
            }
            try (FusionInvocation invocation = session.acquire()) {
                fixture.setInputs(invocation, secondIndices, state, branch, 1);
                try (FusionOutputLease lease = invocation.submit()) {
                    float[] actual = lease.get(fixture.output).toFloatArray();
                    float[] expected = {
                        0f, 0f, 0f, 0f, 1f, 30f,
                        0f, 0f, 0f, 0f, 0f, 0f
                    };
                    Assert.assertEquals(actual, expected);
                }
            }
        }
    }

    @Test
    @SuppressWarnings("try")
    public void gpuMixedAffineBindKeepsFixedSourceAliveAcrossStreams() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
        }
        MixedSourceAffineFixture fixture = new MixedSourceAffineFixture(DataType.FLOAT16);
        Device device = Device.gpu(0);
        try (PtNDManager manager = (PtNDManager) engine.newBaseManager(device);
                NDArray blocker = manager.ones(new Shape(2048, 2048), DataType.FLOAT32);
                FusionPlan plan = engine.newFusionCompiler(device).prepare(fixture.recipe);
                PtStream bindingStream = engine.newStream(device);
                PtEvent inputsReady = bindingStream.newEvent();
                PtEvent bindingComplete = bindingStream.newEvent()) {
            FusionExecutable executable;
            try (PtNDManager constants = (PtNDManager) engine.newBaseManager(device)) {
                NDArray firstWeight =
                        typed(
                                constants,
                                DataType.FLOAT16,
                                new float[] {1f, 0f, 0f, 1f},
                                new Shape(2, 2));
                NDArray secondWeight =
                        typed(constants, DataType.FLOAT16, new float[] {2f, -1f}, new Shape(2, 1));
                NDArray fixed = constants.create(new float[] {10f, -4f}, new Shape(1, 2, 1));
                NDArray fixedWeight =
                        typed(
                                constants,
                                DataType.FLOAT16,
                                new float[] {0.5f, 1.5f},
                                new Shape(2, 1));
                NDArray bias =
                        typed(
                                constants,
                                DataType.FLOAT16,
                                new float[] {0.25f, -0.5f},
                                new Shape(2));
                inputsReady.record();
                try (PtStreamScope ignored = bindingStream.openScope()) {
                    inputsReady.waitOnStream();
                    for (int iteration = 0; iteration < 32; ++iteration) {
                        try (NDArray ignoredProduct = blocker.matMul(blocker)) {
                            // Keep the bind work queued while its caller-owned constants close.
                        }
                    }
                    executable =
                            plan.bind(
                                    fixture.bindings(
                                            firstWeight, secondWeight, fixed, fixedWeight, bias));
                    bindingComplete.record();
                }
            }
            executable.close();

            NDArray[] churn = new NDArray[64];
            try {
                for (int index = 0; index < churn.length; ++index) {
                    churn[index] = manager.zeros(new Shape(1, 2, 1), DataType.FLOAT32);
                }
                bindingComplete.synchronize();
                Assert.assertTrue(bindingComplete.isComplete());
            } finally {
                for (NDArray array : churn) {
                    if (array != null) {
                        array.close();
                    }
                }
            }
        }
    }

    @Test
    public void gpuSessionCreationRollsBackForUnavailableManager() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
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
    public void gpuSessionWaitsForAllocationAndReusesSlotAcrossStreams() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
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
    public void gpuDuplicateExportsAliasOnePersistentValue() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
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
    public void gpuNestedOutputPackUsesConstantAndIntermediateStorage() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
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
    public void gpuConstantBindingRequiresMaximumShape() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
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
    public void gpuOutputPackRejectsInputRowsBelowActiveExtent() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
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
    public void gpuOutputPackRejectsInputRowsAbovePreparedMaximum() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
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
    public void gpuOutputPackValidatesEveryCommandBeforeLaunching() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
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
    public void gpuSessionSurvivesNdScopeAndSequentialThreadMigration() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
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
    public void gpuSingleUseFacadesRejectExpiredGenerations() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This fusion test requires a PyTorch CUDA or ROCm device.");
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

    private static float[] patternedValues(int count, int period, int center, float scale) {
        float[] values = new float[count];
        for (int index = 0; index < count; ++index) {
            values[index] = (index % period - center) * scale;
        }
        return values;
    }

    private static NDArray patternedArray(
            NDManager manager,
            DataType dataType,
            Shape shape,
            int period,
            int center,
            float scale) {
        try (NDScope scope = new NDScope()) {
            scope.suppressNotUsedWarning();
            NDArray result =
                    manager.arange((float) shape.size())
                            .mod(period)
                            .sub(center)
                            .mul(scale)
                            .reshape(shape)
                            .toType(dataType, false);
            NDScope.unregister(result);
            return result;
        }
    }

    private static NDArray legalReadoutMask(
            NDManager manager, DataType dataType, int batchCount, int tokenCount) {
        try (NDScope scope = new NDScope()) {
            scope.suppressNotUsedWarning();
            NDArray tokens = manager.arange(tokenCount).reshape(1, tokenCount);
            NDArray validCounts =
                    manager.arange(batchCount).mod(52).add(100).reshape(batchCount, 1);
            NDArray result =
                    tokens.broadcast(batchCount, tokenCount)
                            .lt(validCounts)
                            .toType(dataType, false);
            NDScope.unregister(result);
            return result;
        }
    }

    private static void assertSingleQueryReadoutReference(
            NDArray actualStorage,
            NDArray memoryStorage,
            NDArray maskStorage,
            PtSingleQueryReadoutTestSupport.SingleQueryReadoutArrays weights,
            int queryIndex,
            int batchCount,
            DataType dataType,
            int deviceIndex,
            String readoutName) {
        try (NDScope scope = new NDScope()) {
            scope.suppressNotUsedWarning();
            NDArray memory = memoryStorage.get("0:" + batchCount);
            NDArray mask = maskStorage.get("0:" + batchCount);
            NDArray querySeed = memory.get(":," + queryIndex).toType(dataType, false);
            NDArray validTokens = mask.neq(0);
            NDArray floatMask = validTokens.toType(DataType.FLOAT32, false);
            NDArray selectedMemory =
                    NDArrays.where(
                            validTokens.expandDims(2).broadcast(memory.getShape()),
                            memory,
                            memory.zerosLike());
            NDArray mean =
                    selectedMemory
                            .toType(DataType.FLOAT32, false)
                            .sum(new int[] {1})
                            .div(floatMask.sum(new int[] {1}, true).maximum(1f))
                            .toType(dataType, false);
            NDArray seedInput = NDArrays.concat(new NDList(querySeed, mean), 1);
            NDArray seedProduct = seedInput.matMul(weights.seedWeight.transpose());
            NDArray seedState = seedProduct.add(weights.seedBias);
            NDArray query =
                    seedState.matMul(weights.queryWeight.transpose()).add(weights.queryBias);

            int attentionWidth = Math.toIntExact(weights.queryBias.getShape().get(0));
            int attentionHeads = 4;
            int headWidth = attentionWidth / attentionHeads;
            int tokenCount = Math.toIntExact(memory.getShape().get(1));
            int hiddenWidth = Math.toIntExact(memory.getShape().get(2));
            NDArray projectionMemory = selectedMemory.toType(dataType, false);
            NDArray queryHeads = query.reshape(batchCount, attentionHeads, 1, headWidth);
            NDArray keyValues =
                    projectionMemory
                            .reshape(batchCount * tokenCount, hiddenWidth)
                            .matMul(weights.keyValueWeight.transpose())
                            .reshape(batchCount, tokenCount, 2 * attentionWidth);
            NDArray keys =
                    keyValues
                            .get("...,0:" + attentionWidth)
                            .reshape(batchCount, tokenCount, attentionHeads, headWidth)
                            .swapAxes(1, 2);
            NDArray values =
                    keyValues
                            .get("...," + attentionWidth + ':' + (2 * attentionWidth))
                            .reshape(batchCount, tokenCount, attentionHeads, headWidth)
                            .swapAxes(1, 2);
            NDArray attentionValidTokens =
                    validTokens.logicalOr(
                            floatMask
                                    .sum(new int[] {1}, true)
                                    .eq(0)
                                    .broadcast(validTokens.getShape()));
            NDArray biasValues = attentionValidTokens.toType(dataType, false);
            NDArray invalidBias =
                    NDArrays.where(
                                    attentionValidTokens,
                                    biasValues.zerosLike(),
                                    biasValues.zerosLike().add(Float.NEGATIVE_INFINITY))
                            .reshape(batchCount, 1, 1, tokenCount);
            NDArray probabilities =
                    queryHeads
                            .matMul(keys.swapAxes(2, 3))
                            .div((float) Math.sqrt(headWidth))
                            .add(invalidBias)
                            .softmax(3);
            NDArray materializedContext =
                    probabilities.matMul(values).swapAxes(1, 2).reshape(batchCount, attentionWidth);

            NDArray keyWeights =
                    weights.keyValueWeight
                            .get("0:" + attentionWidth)
                            .reshape(attentionHeads, headWidth, hiddenWidth);
            NDArray direction = queryHeads.matMul(keyWeights).div((float) Math.sqrt(headWidth));
            NDArray reorderedProbabilities =
                    direction
                            .matMul(projectionMemory.expandDims(1).swapAxes(2, 3))
                            .add(invalidBias)
                            .softmax(3);
            NDArray pooled = reorderedProbabilities.matMul(projectionMemory.expandDims(1));
            NDArray valueWeights =
                    weights.keyValueWeight
                            .get(attentionWidth + ":" + (2 * attentionWidth))
                            .reshape(attentionHeads, headWidth, hiddenWidth)
                            .swapAxes(1, 2);
            NDArray reorderedContext =
                    pooled.matMul(valueWeights).swapAxes(1, 2).reshape(batchCount, attentionWidth);
            assertReadoutClose(
                    reorderedContext,
                    materializedContext,
                    attentionTolerance(dataType),
                    deviceIndex,
                    dataType,
                    batchCount,
                    readoutName + "-attention");

            NDArray attentionUpdate =
                    materializedContext
                            .matMul(weights.contextWeight.transpose())
                            .add(weights.contextBias);
            NDArray state =
                    mixedPrecisionLayerNormReference(
                            seedState.add(attentionUpdate),
                            weights.queryNormWeight,
                            weights.queryNormBias);
            NDArray feedForwardInput =
                    mixedPrecisionLayerNormReference(
                            state, weights.feedForwardNormWeight, weights.feedForwardNormBias);
            NDArray expanded =
                    feedForwardInput
                            .matMul(weights.expansionWeight.transpose())
                            .add(weights.expansionBias);
            NDArray projected =
                    expanded.mul(Activation.sigmoid(expanded))
                            .matMul(weights.projectionWeight.transpose())
                            .add(weights.projectionBias);
            NDArray expected =
                    mixedPrecisionLayerNormReference(
                            state.add(projected), weights.outputNormWeight, weights.outputNormBias);
            NDArray actual = actualStorage.get("0:" + batchCount);
            Assert.assertEquals(actual.getShape(), new Shape(batchCount, hiddenWidth));
            assertReadoutClose(
                    actual,
                    expected,
                    outputTolerance(dataType),
                    deviceIndex,
                    dataType,
                    batchCount,
                    readoutName + "-output");
        }
    }

    private static float attentionTolerance(DataType dataType) {
        if (dataType == DataType.FLOAT32) {
            return 5.0e-4f;
        }
        if (dataType == DataType.FLOAT16) {
            return 4.0e-2f;
        }
        return 1.5e-1f;
    }

    private static float outputTolerance(DataType dataType) {
        if (dataType == DataType.FLOAT32) {
            return 1.0e-3f;
        }
        if (dataType == DataType.FLOAT16) {
            return 6.0e-2f;
        }
        return 2.0e-1f;
    }

    private static void assertReadoutClose(
            NDArray actual,
            NDArray expected,
            float tolerance,
            int deviceIndex,
            DataType dataType,
            int batchCount,
            String stage) {
        float[] actualValues = actual.toType(DataType.FLOAT32, false).toFloatArray();
        float[] expectedValues = expected.toType(DataType.FLOAT32, false).toFloatArray();
        Assert.assertEquals(actualValues.length, expectedValues.length);
        float maximum = 0f;
        double total = 0.0;
        for (int index = 0; index < actualValues.length; ++index) {
            Assert.assertTrue(
                    Float.isFinite(actualValues[index]) && Float.isFinite(expectedValues[index]),
                    "Non-finite single-query readout value at " + stage + " index " + index);
            float difference = Math.abs(actualValues[index] - expectedValues[index]);
            maximum = Math.max(maximum, difference);
            total += difference;
        }
        double mean = total / actualValues.length;
        System.out.printf(
                "single-query-parity device=%d dtype=%s batch=%d stage=%s maxAbs=%.9g"
                        + " meanAbs=%.9g tolerance=%.9g%n",
                deviceIndex, dataType, batchCount, stage, maximum, mean, tolerance);
        Assert.assertTrue(
                maximum <= tolerance,
                "Single-query readout parity exceeded its fixed tolerance: device="
                        + deviceIndex
                        + ", dtype="
                        + dataType
                        + ", batch="
                        + batchCount
                        + ", stage="
                        + stage
                        + ", maxAbs="
                        + maximum
                        + ", meanAbs="
                        + mean
                        + ", tolerance="
                        + tolerance);
    }

    private static IndexedLocalPattern indexedLocalPattern(int maximumBatch) {
        int denseRows = maximumBatch * 4 * 29;
        float[] mask = new float[denseRows];
        int[] temporaryIndices = new int[denseRows];
        int activeRows = 0;
        for (int denseRow = 0; denseRow < denseRows; ++denseRow) {
            int token = denseRow % 29;
            int group = denseRow / 29;
            boolean present = token == 0 || Math.floorMod(group * 131 + token * 47, 1000) < 234;
            if (present) {
                mask[denseRow] = 1f;
                temporaryIndices[activeRows++] = denseRow;
            }
        }
        int[] indices = new int[activeRows];
        System.arraycopy(temporaryIndices, 0, indices, 0, activeRows);
        return new IndexedLocalPattern(mask, indices);
    }

    private static NDArray indexedLocalTransformerReference(
            NDArray input, NDArray indices, NDArray mask, BoundIndexedLocalTransformer weights) {
        try (NDScope scope = new NDScope()) {
            scope.suppressNotUsedWarning();
            long batchCount = input.getShape().get(0);
            long denseRows = batchCount * 4L * 29L;
            long activeRows = indices.getShape().get(0);
            NDArray selectedInput = NDArrays.gatherRows(input.reshape(denseRows, 256), indices);
            NDArray state =
                    mixedPrecisionLayerNormReference(
                            selectedInput, weights.inputNormWeight, weights.inputNormBias);
            NDArray normalized =
                    mixedPrecisionLayerNormReference(
                            state, weights.attentionInputWeight, weights.attentionInputBias);
            NDArray selectedQueryKeyValue = normalized.matMul(weights.queryKeyValue.transpose());
            NDArray queryKeyValue =
                    NDArrays.scatterRows(selectedQueryKeyValue, indices, denseRows)
                            .reshape(batchCount * 4L, 29, 192);
            NDArray queries =
                    queryKeyValue
                            .get("...,0:64")
                            .reshape(batchCount * 4L, 29, 4, 16)
                            .swapAxes(1, 2);
            NDArray keys =
                    queryKeyValue
                            .get("...,64:128")
                            .reshape(batchCount * 4L, 29, 4, 16)
                            .swapAxes(1, 2);
            NDArray values =
                    queryKeyValue
                            .get("...,128:192")
                            .reshape(batchCount * 4L, 29, 4, 16)
                            .swapAxes(1, 2);
            NDArray validMask =
                    mask.reshape(batchCount * 4L, 1, 1, 29).toType(input.getDataType(), false);
            NDArray hasValid =
                    validMask.sum(new int[] {3}, true).gt(0).broadcast(validMask.getShape());
            NDArray attentionValid =
                    NDArrays.where(hasValid, validMask.neq(0), validMask.onesLike().neq(0));
            NDArray attentionMask =
                    NDArrays.where(
                            attentionValid,
                            validMask.zerosLike(),
                            validMask.zerosLike().add(Float.NEGATIVE_INFINITY));
            NDArray context =
                    queries.getNDArrayInternal()
                            .scaledDotProductAttention(keys, values, attentionMask, 0.0, false)
                            .swapAxes(1, 2)
                            .reshape(denseRows, 64);
            NDArray selectedContext = NDArrays.gatherRows(context, indices);
            NDArray attentionUpdate =
                    selectedContext
                            .matMul(weights.attentionOutput.transpose())
                            .add(weights.attentionOutputBias);
            state = state.add(attentionUpdate);
            normalized =
                    mixedPrecisionLayerNormReference(
                            state, weights.feedForwardInputWeight, weights.feedForwardInputBias);
            NDArray expanded =
                    normalized.matMul(weights.expansion.transpose()).add(weights.expansionBias);
            NDArray projected =
                    expanded.mul(Activation.sigmoid(expanded))
                            .matMul(weights.projection.transpose())
                            .add(weights.projectionBias);
            NDArray encoded =
                    mixedPrecisionLayerNormReference(
                            state.add(projected), weights.outputWeight, weights.outputBias);
            NDArray result =
                    NDArrays.scatterRows(encoded.reshape(activeRows, 256), indices, denseRows)
                            .reshape(input.getShape());
            NDScope.unregister(result);
            return result;
        }
    }

    private static void assertIndexedLocalTransformerClose(
            NDArray actual,
            NDArray expected,
            float tolerance,
            int deviceIndex,
            DataType dataType,
            int batchCount,
            int activeRows) {
        float[] actualValues = actual.toType(DataType.FLOAT32, false).toFloatArray();
        float[] expectedValues = expected.toType(DataType.FLOAT32, false).toFloatArray();
        Assert.assertEquals(actualValues.length, expectedValues.length);
        float maximum = 0f;
        double total = 0.0;
        for (int index = 0; index < actualValues.length; ++index) {
            Assert.assertTrue(
                    Float.isFinite(actualValues[index]) && Float.isFinite(expectedValues[index]),
                    "Non-finite indexed local transformer value at index " + index);
            float difference = Math.abs(actualValues[index] - expectedValues[index]);
            maximum = Math.max(maximum, difference);
            total += difference;
        }
        double mean = total / actualValues.length;
        System.out.printf(
                "indexed-local-transformer-parity device=%d dtype=%s batch=%d active=%d "
                        + "maxAbs=%.9g meanAbs=%.9g tolerance=%.9g%n",
                deviceIndex, dataType, batchCount, activeRows, maximum, mean, tolerance);
        Assert.assertTrue(
                maximum <= tolerance,
                "Indexed local transformer parity exceeded its fixed tolerance: device="
                        + deviceIndex
                        + ", dtype="
                        + dataType
                        + ", batch="
                        + batchCount
                        + ", active="
                        + activeRows
                        + ", maxAbs="
                        + maximum
                        + ", meanAbs="
                        + mean
                        + ", tolerance="
                        + tolerance);
    }

    private static NDArray transformerEncoderReference(
            NDArray input,
            NDArray attentionInputWeight,
            NDArray attentionInputBias,
            NDArray queryKeyValueWeight,
            NDArray attentionOutputWeight,
            NDArray attentionOutputBias,
            NDArray feedForwardInputWeight,
            NDArray feedForwardInputBias,
            NDArray expansionWeight,
            NDArray expansionBias,
            NDArray projectionWeight,
            NDArray projectionBias,
            NDArray outputWeight,
            NDArray outputBias) {
        NDArray normalized =
                mixedPrecisionLayerNormReference(input, attentionInputWeight, attentionInputBias);
        NDArray queryKeyValue =
                normalized
                        .reshape(6, 256)
                        .matMul(queryKeyValueWeight.transpose())
                        .reshape(1, 6, 384);
        NDArray queries = queryKeyValue.get("...,0:128").reshape(1, 6, 4, 32).swapAxes(1, 2);
        NDArray keys = queryKeyValue.get("...,128:256").reshape(1, 6, 4, 32).swapAxes(1, 2);
        NDArray values = queryKeyValue.get("...,256:384").reshape(1, 6, 4, 32).swapAxes(1, 2);
        NDArray context =
                queries.getNDArrayInternal()
                        .scaledDotProductAttention(keys, values, null, 0.0, false)
                        .swapAxes(1, 2)
                        .reshape(6, 128);
        NDArray residual =
                input.add(
                        context.matMul(attentionOutputWeight.transpose())
                                .add(attentionOutputBias)
                                .reshape(1, 6, 256));
        NDArray feedForwardInput =
                mixedPrecisionLayerNormReference(
                        residual, feedForwardInputWeight, feedForwardInputBias);
        NDArray expanded =
                feedForwardInput
                        .reshape(6, 256)
                        .matMul(expansionWeight.transpose())
                        .add(expansionBias);
        NDArray projected =
                expanded.mul(Activation.sigmoid(expanded))
                        .matMul(projectionWeight.transpose())
                        .add(projectionBias)
                        .reshape(1, 6, 256);
        NDArray output = residual.add(projected);
        return mixedPrecisionLayerNormReference(output, outputWeight, outputBias);
    }

    private static NDArray mixedPrecisionLayerNormReference(
            NDArray input, NDArray weight, NDArray bias) {
        DataType outputType = input.getDataType();
        NDArray floatInput = input.toType(DataType.FLOAT32, false);
        NDArray floatWeight = weight.toType(DataType.FLOAT32, false);
        NDArray floatBias = bias.toType(DataType.FLOAT32, false);
        NDArray normalized =
                floatInput
                        .getNDArrayInternal()
                        .layerNorm(floatInput, new Shape(256), floatWeight, floatBias, 1e-5f)
                        .get(0);
        if (outputType == DataType.FLOAT32) {
            return normalized;
        }
        return normalized.toType(outputType, false);
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

    private static float[] indexedAffineReference() {
        float[] output = new float[6];
        float[][] state = {{1f, 2f}, {3f, 4f}};
        float[][] branch = {
            {10f, 20f}, {21f, 22f}, {30f, 40f},
            {41f, 42f}, {50f, 60f}, {61f, 62f}
        };
        int[] destinations = {0, 2, 4};
        for (int destination : destinations) {
            int stateRow = destination / 3;
            float first = silu(state[stateRow][0] + branch[destination][0] + 0.1f);
            float second = silu(state[stateRow][1] + branch[destination][1] - 0.2f);
            output[destination] = 0.25f + 0.5f * first - second;
        }
        return output;
    }

    private static float silu(float value) {
        return (float) (value / (1.0 + Math.exp(-value)));
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

    private static final class BinaryBranchBlendFixture {

        private final FusionRecipe.Dimension rows;
        private final FusionRecipe.Input baselineContext;
        private final FusionRecipe.Input selectedContext;
        private final FusionRecipe.Input selectedLogit;
        private final FusionRecipe.Input baselinePresence;
        private final FusionRecipe.Input selectedPresence;
        private final FusionRecipe.BinaryBranchBlend blend;
        private final FusionRecipe.Output output;
        private final FusionRecipe recipe;

        private BinaryBranchBlendFixture() {
            FusionRecipe.Builder builder = FusionRecipe.builder("binary-branch-blend");
            rows = builder.addDimension("rows", 4);
            baselineContext =
                    builder.addInput(
                            "baselineContext",
                            FusionRecipe.TensorSpec.of(DataType.FLOAT16, rows, 3));
            selectedContext =
                    builder.addInput(
                            "selectedContext",
                            FusionRecipe.TensorSpec.of(DataType.BFLOAT16, rows, 3));
            selectedLogit =
                    builder.addInput(
                            "selectedLogit", FusionRecipe.TensorSpec.of(DataType.FLOAT16, rows, 1));
            baselinePresence =
                    builder.addInput(
                            "baselinePresence",
                            FusionRecipe.TensorSpec.of(DataType.BFLOAT16, rows, 1));
            selectedPresence =
                    builder.addInput(
                            "selectedPresence",
                            FusionRecipe.TensorSpec.of(DataType.FLOAT32, rows, 1));
            blend =
                    builder.binaryBranchBlend(
                            "blend",
                            baselineContext,
                            selectedContext,
                            selectedLogit,
                            baselinePresence,
                            selectedPresence);
            output = builder.addOutput("output", blend);
            recipe = builder.build();
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

    private static final class IndexedAffineFixture {

        private final FusionRecipe.Dimension sourceRows;
        private final FusionRecipe.Dimension destinationRows;
        private final FusionRecipe.Dimension activeRows;
        private final FusionRecipe.Input indices;
        private final FusionRecipe.Input state;
        private final FusionRecipe.Input branch;
        private final FusionRecipe.Constant hiddenWeight;
        private final FusionRecipe.Constant hiddenBias;
        private final FusionRecipe.Constant outputWeight;
        private final FusionRecipe.Constant outputBias;
        private final FusionRecipe.Output output;
        private final FusionRecipe recipe;

        private IndexedAffineFixture(DataType dataType) {
            FusionRecipe.Builder builder = FusionRecipe.builder("indexed-affine-test");
            sourceRows = builder.addDimension("sourceRows", 2);
            destinationRows = builder.addDimension("destinationRows", 6);
            activeRows = builder.addDimension("activeRows", 4);
            indices =
                    builder.addInput(
                            "indices", FusionRecipe.TensorSpec.of(DataType.INT32, activeRows));
            state =
                    builder.addInput(
                            "state", FusionRecipe.TensorSpec.of(DataType.FLOAT32, sourceRows, 2));
            branch =
                    builder.addInput(
                            "branch",
                            FusionRecipe.TensorSpec.of(DataType.FLOAT16, destinationRows, 2));
            hiddenWeight =
                    builder.addConstant(
                            "hiddenWeight", FusionRecipe.TensorSpec.fixed(dataType, 2, 4));
            hiddenBias =
                    builder.addConstant("hiddenBias", FusionRecipe.TensorSpec.fixed(dataType, 2));
            outputWeight =
                    builder.addConstant(
                            "outputWeight", FusionRecipe.TensorSpec.fixed(dataType, 1, 2));
            outputBias =
                    builder.addConstant("outputBias", FusionRecipe.TensorSpec.fixed(dataType, 1));
            FusionRecipe.IndexedAffine indexed =
                    builder.indexedAffine("scores", indices, destinationRows)
                            .addSource(state, 3)
                            .addSource(branch, 1)
                            .setHiddenWeight(hiddenWeight)
                            .optHiddenBias(hiddenBias)
                            .optActivation(FusionRecipe.Activation.SILU)
                            .setOutputWeight(outputWeight)
                            .optOutputBias(outputBias)
                            .build();
            output = builder.addOutput("output", indexed);
            recipe = builder.build();
        }

        private FusionConstantBindings bindings(
                NDArray hiddenWeightArray,
                NDArray hiddenBiasArray,
                NDArray outputWeightArray,
                NDArray outputBiasArray) {
            return FusionConstantBindings.builder(recipe)
                    .bind(hiddenWeight, hiddenWeightArray)
                    .bind(hiddenBias, hiddenBiasArray)
                    .bind(outputWeight, outputWeightArray)
                    .bind(outputBias, outputBiasArray)
                    .build();
        }

        private void setInputs(
                FusionInvocation invocation,
                NDArray indicesArray,
                NDArray stateArray,
                NDArray branchArray,
                long sourceRowCount,
                long destinationRowCount,
                long activeRowCount) {
            invocation.setInput(indices, indicesArray);
            invocation.setInput(state, stateArray);
            invocation.setInput(branch, branchArray);
            invocation.setDimension(sourceRows, sourceRowCount);
            invocation.setDimension(destinationRows, destinationRowCount);
            invocation.setDimension(activeRows, activeRowCount);
        }
    }

    private static final class WideIndexedAffineFixture {

        private final FusionRecipe.Dimension sourceRows;
        private final FusionRecipe.Dimension destinationRows;
        private final FusionRecipe.Dimension activeRows;
        private final FusionRecipe.Input indices;
        private final FusionRecipe.Input state;
        private final FusionRecipe.Input branch;
        private final FusionRecipe.Constant hiddenWeight;
        private final FusionRecipe.Constant outputWeight;
        private final FusionRecipe.Output output;
        private final FusionRecipe recipe;

        private WideIndexedAffineFixture() {
            FusionRecipe.Builder builder = FusionRecipe.builder("wide-indexed-affine-test");
            sourceRows = builder.addDimension("sourceRows", 2);
            destinationRows = builder.addDimension("destinationRows", 6);
            activeRows = builder.addDimension("activeRows", 4);
            indices =
                    builder.addInput(
                            "indices", FusionRecipe.TensorSpec.of(DataType.INT64, activeRows));
            state =
                    builder.addInput(
                            "state", FusionRecipe.TensorSpec.of(DataType.BFLOAT16, sourceRows, 1));
            branch =
                    builder.addInput(
                            "branch",
                            FusionRecipe.TensorSpec.of(DataType.FLOAT32, destinationRows, 1));
            hiddenWeight =
                    builder.addConstant(
                            "hiddenWeight", FusionRecipe.TensorSpec.fixed(DataType.FLOAT32, 2, 2));
            outputWeight =
                    builder.addConstant(
                            "outputWeight", FusionRecipe.TensorSpec.fixed(DataType.FLOAT32, 2, 2));
            FusionRecipe.IndexedAffine indexed =
                    builder.indexedAffine("scores", indices, destinationRows)
                            .addSource(state, 3)
                            .addSource(branch, 1)
                            .setHiddenWeight(hiddenWeight)
                            .setOutputWeight(outputWeight)
                            .build();
            output = builder.addOutput("output", indexed);
            recipe = builder.build();
        }

        private FusionConstantBindings bindings(
                NDArray hiddenWeightArray, NDArray outputWeightArray) {
            return FusionConstantBindings.builder(recipe)
                    .bind(hiddenWeight, hiddenWeightArray)
                    .bind(outputWeight, outputWeightArray)
                    .build();
        }

        private void setInputs(
                FusionInvocation invocation,
                NDArray indicesArray,
                NDArray stateArray,
                NDArray branchArray,
                long activeRowCount) {
            invocation.setInput(indices, indicesArray);
            invocation.setInput(state, stateArray);
            invocation.setInput(branch, branchArray);
            invocation.setDimension(sourceRows, 2);
            invocation.setDimension(destinationRows, 6);
            invocation.setDimension(activeRows, activeRowCount);
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

    private static final class MixedDirectAffineFixture {

        private final FusionRecipe.Dimension rows;
        private final FusionRecipe.Input input;
        private final FusionRecipe.Constant weight;
        private final FusionRecipe.Output output;
        private final FusionRecipe recipe;

        private MixedDirectAffineFixture(DataType projectionDataType) {
            FusionRecipe.Builder builder = FusionRecipe.builder("mixed-direct-affine-test");
            rows = builder.addDimension("rows", 4);
            input =
                    builder.addInput(
                            "input", FusionRecipe.TensorSpec.of(DataType.FLOAT32, rows, 2, 2));
            weight =
                    builder.addConstant(
                            "weight", FusionRecipe.TensorSpec.fixed(projectionDataType, 3, 2));
            FusionRecipe.AffineSum value =
                    builder.affineSum("affine", 3).addTerm(input, weight).build();
            output = builder.addOutput("output", value);
            recipe = builder.build();
        }
    }

    private static final class MixedSourceAffineFixture {

        private final FusionRecipe.Dimension rows;
        private final FusionRecipe.Input first;
        private final FusionRecipe.Input second;
        private final FusionRecipe.Constant firstWeight;
        private final FusionRecipe.Constant secondWeight;
        private final FusionRecipe.Constant fixed;
        private final FusionRecipe.Constant fixedWeight;
        private final FusionRecipe.Constant bias;
        private final FusionRecipe.Output output;
        private final FusionRecipe recipe;

        private MixedSourceAffineFixture(DataType projectionDataType) {
            FusionRecipe.Builder builder = FusionRecipe.builder("mixed-source-affine-test");
            rows = builder.addDimension("rows", 4);
            first =
                    builder.addInput(
                            "first", FusionRecipe.TensorSpec.of(DataType.FLOAT32, rows, 2, 2));
            second =
                    builder.addInput(
                            "second", FusionRecipe.TensorSpec.of(DataType.FLOAT16, rows, 2, 1));
            firstWeight =
                    builder.addConstant(
                            "firstWeight", FusionRecipe.TensorSpec.fixed(projectionDataType, 2, 2));
            secondWeight =
                    builder.addConstant(
                            "secondWeight",
                            FusionRecipe.TensorSpec.fixed(projectionDataType, 2, 1));
            fixed =
                    builder.addConstant(
                            "fixed", FusionRecipe.TensorSpec.fixed(DataType.FLOAT32, 1, 2, 1));
            fixedWeight =
                    builder.addConstant(
                            "fixedWeight", FusionRecipe.TensorSpec.fixed(projectionDataType, 2, 1));
            bias =
                    builder.addConstant(
                            "bias", FusionRecipe.TensorSpec.fixed(projectionDataType, 2));
            FusionRecipe.AffineSum value =
                    builder.affineSum("affine", 2)
                            .addTerm(first, firstWeight)
                            .addTerm(second, secondWeight)
                            .addTerm(fixed, fixedWeight)
                            .optBias(bias)
                            .build();
            output = builder.addOutput("output", value);
            recipe = builder.build();
        }

        private FusionConstantBindings bindings(
                NDArray firstWeightArray,
                NDArray secondWeightArray,
                NDArray fixedArray,
                NDArray fixedWeightArray,
                NDArray biasArray) {
            return FusionConstantBindings.builder(recipe)
                    .bind(firstWeight, firstWeightArray)
                    .bind(secondWeight, secondWeightArray)
                    .bind(fixed, fixedArray)
                    .bind(fixedWeight, fixedWeightArray)
                    .bind(bias, biasArray)
                    .build();
        }

        private void setInputs(
                FusionInvocation invocation, NDArray firstArray, NDArray secondArray, long extent) {
            invocation.setInput(first, firstArray);
            invocation.setInput(second, secondArray);
            invocation.setDimension(rows, extent);
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

    private static final class IndexedLocalTransformerFixture {

        private final FusionRecipe.Dimension batch;
        private final FusionRecipe.Dimension active;
        private final FusionRecipe.Input input;
        private final List<FusionRecipe.Input> inputs;
        private final FusionRecipe.Input indices;
        private final FusionRecipe.Constant inputNormWeight;
        private final FusionRecipe.Constant inputNormBias;
        private final FusionRecipe.Constant attentionInputWeight;
        private final FusionRecipe.Constant attentionInputBias;
        private final FusionRecipe.Constant queryKeyValue;
        private final FusionRecipe.Constant attentionOutput;
        private final FusionRecipe.Constant attentionOutputBias;
        private final FusionRecipe.Constant feedForwardInputWeight;
        private final FusionRecipe.Constant feedForwardInputBias;
        private final FusionRecipe.Constant expansion;
        private final FusionRecipe.Constant expansionBias;
        private final FusionRecipe.Constant projection;
        private final FusionRecipe.Constant projectionBias;
        private final FusionRecipe.Constant outputWeight;
        private final FusionRecipe.Constant outputBias;
        private final FusionRecipe.Output output;
        private final FusionRecipe recipe;

        private IndexedLocalTransformerFixture(DataType dataType, int maximumBatch) {
            this(dataType, maximumBatch, false);
        }

        private IndexedLocalTransformerFixture(
                DataType dataType, int maximumBatch, boolean segmented) {
            long maximumActiveRows = Math.multiplyExact((long) maximumBatch, 4L * 29L);
            FusionRecipe.Builder builder = FusionRecipe.builder("indexed-local-transformer");
            batch = builder.addDimension("batch", maximumBatch);
            active = builder.addDimension("active", maximumActiveRows);
            FusionRecipe.Input river = null;
            FusionRecipe.Input meld = null;
            if (segmented) {
                input =
                        builder.addInput(
                                "player", FusionRecipe.TensorSpec.of(dataType, batch, 4, 1, 256));
                river =
                        builder.addInput(
                                "river", FusionRecipe.TensorSpec.of(dataType, batch, 4, 24, 256));
                meld =
                        builder.addInput(
                                "meld", FusionRecipe.TensorSpec.of(dataType, batch, 4, 4, 256));
            } else {
                input =
                        builder.addInput(
                                "input", FusionRecipe.TensorSpec.of(dataType, batch, 4, 29, 256));
            }
            inputs = segmented ? Arrays.asList(input, river, meld) : Arrays.asList(input);
            indices =
                    builder.addInput("indices", FusionRecipe.TensorSpec.of(DataType.INT32, active));
            inputNormWeight = vector(builder, "inputNormWeight", 256, DataType.FLOAT32);
            inputNormBias = vector(builder, "inputNormBias", 256, DataType.FLOAT32);
            attentionInputWeight = vector(builder, "attentionInputWeight", 256, DataType.FLOAT32);
            attentionInputBias = vector(builder, "attentionInputBias", 256, DataType.FLOAT32);
            queryKeyValue = matrix(builder, "qkv", 192, 256, dataType);
            attentionOutput = matrix(builder, "attentionOutput", 256, 64, dataType);
            attentionOutputBias = vector(builder, "attentionOutputBias", 256, dataType);
            feedForwardInputWeight =
                    vector(builder, "feedForwardInputWeight", 256, DataType.FLOAT32);
            feedForwardInputBias = vector(builder, "feedForwardInputBias", 256, DataType.FLOAT32);
            expansion = matrix(builder, "expansion", 128, 256, dataType);
            expansionBias = vector(builder, "expansionBias", 128, dataType);
            projection = matrix(builder, "projection", 256, 128, dataType);
            projectionBias = vector(builder, "projectionBias", 256, dataType);
            outputWeight = vector(builder, "outputWeight", 256, DataType.FLOAT32);
            outputBias = vector(builder, "outputBias", 256, DataType.FLOAT32);
            FusionRecipe.IndexedLocalTransformerEncoderBuilder encoderBuilder =
                    segmented
                            ? builder.indexedLocalTransformerEncoder(
                                    "encoder", inputs, indices, 4, 64, 128)
                            : builder.indexedLocalTransformerEncoder(
                                    "encoder", input, indices, 4, 64, 128);
            FusionRecipe.IndexedLocalTransformerEncoder encoder =
                    encoderBuilder
                            .setInputNormalization(inputNormWeight, inputNormBias)
                            .setBlock(
                                    attentionInputWeight,
                                    attentionInputBias,
                                    queryKeyValue,
                                    attentionOutput,
                                    attentionOutputBias,
                                    feedForwardInputWeight,
                                    feedForwardInputBias,
                                    expansion,
                                    expansionBias,
                                    projection,
                                    projectionBias,
                                    outputWeight,
                                    outputBias)
                            .build();
            output = builder.addOutput("output", encoder);
            recipe = builder.build();
        }

        private BoundIndexedLocalTransformer bind(NDManager manager) {
            NDList resources = new NDList();
            NDArray inputNormWeightArray = parameter(manager, inputNormWeight, 29, 14, 0.004f, 1f);
            NDArray inputNormBiasArray = parameter(manager, inputNormBias, 31, 15, 0.001f, 0f);
            NDArray attentionInputWeightArray =
                    parameter(manager, attentionInputWeight, 23, 11, 0.004f, 1f);
            NDArray attentionInputBiasArray =
                    parameter(manager, attentionInputBias, 19, 9, 0.001f, 0f);
            NDArray queryKeyValueArray = parameter(manager, queryKeyValue, 37, 18, 0.0015f, 0f);
            NDArray attentionOutputArray = parameter(manager, attentionOutput, 41, 20, 0.0015f, 0f);
            NDArray attentionOutputBiasArray =
                    parameter(manager, attentionOutputBias, 17, 8, 0.001f, 0f);
            NDArray feedForwardInputWeightArray =
                    parameter(manager, feedForwardInputWeight, 27, 13, 0.004f, 1f);
            NDArray feedForwardInputBiasArray =
                    parameter(manager, feedForwardInputBias, 21, 10, 0.001f, 0f);
            NDArray expansionArray = parameter(manager, expansion, 43, 21, 0.0015f, 0f);
            NDArray expansionBiasArray = parameter(manager, expansionBias, 13, 6, 0.001f, 0f);
            NDArray projectionArray = parameter(manager, projection, 47, 23, 0.0015f, 0f);
            NDArray projectionBiasArray = parameter(manager, projectionBias, 25, 12, 0.001f, 0f);
            NDArray outputWeightArray = parameter(manager, outputWeight, 33, 16, 0.004f, 1f);
            NDArray outputBiasArray = parameter(manager, outputBias, 35, 17, 0.001f, 0f);
            resources.addAll(
                    new NDList(
                            inputNormWeightArray,
                            inputNormBiasArray,
                            attentionInputWeightArray,
                            attentionInputBiasArray,
                            queryKeyValueArray,
                            attentionOutputArray,
                            attentionOutputBiasArray,
                            feedForwardInputWeightArray,
                            feedForwardInputBiasArray,
                            expansionArray,
                            expansionBiasArray,
                            projectionArray,
                            projectionBiasArray,
                            outputWeightArray,
                            outputBiasArray));
            FusionConstantBindings bindings =
                    FusionConstantBindings.builder(recipe)
                            .bind(inputNormWeight, inputNormWeightArray)
                            .bind(inputNormBias, inputNormBiasArray)
                            .bind(attentionInputWeight, attentionInputWeightArray)
                            .bind(attentionInputBias, attentionInputBiasArray)
                            .bind(queryKeyValue, queryKeyValueArray)
                            .bind(attentionOutput, attentionOutputArray)
                            .bind(attentionOutputBias, attentionOutputBiasArray)
                            .bind(feedForwardInputWeight, feedForwardInputWeightArray)
                            .bind(feedForwardInputBias, feedForwardInputBiasArray)
                            .bind(expansion, expansionArray)
                            .bind(expansionBias, expansionBiasArray)
                            .bind(projection, projectionArray)
                            .bind(projectionBias, projectionBiasArray)
                            .bind(outputWeight, outputWeightArray)
                            .bind(outputBias, outputBiasArray)
                            .build();
            return new BoundIndexedLocalTransformer(
                    bindings,
                    resources,
                    inputNormWeightArray,
                    inputNormBiasArray,
                    attentionInputWeightArray,
                    attentionInputBiasArray,
                    queryKeyValueArray,
                    attentionOutputArray,
                    attentionOutputBiasArray,
                    feedForwardInputWeightArray,
                    feedForwardInputBiasArray,
                    expansionArray,
                    expansionBiasArray,
                    projectionArray,
                    projectionBiasArray,
                    outputWeightArray,
                    outputBiasArray);
        }

        private static NDArray parameter(
                NDManager manager,
                FusionRecipe.Constant constant,
                int period,
                int center,
                float scale,
                float offset) {
            NDArray value =
                    patternedArray(
                            manager,
                            constant.getSpec().getDataType(),
                            constant.getSpec().getMaximumShape(),
                            period,
                            center,
                            scale);
            return offset == 0f ? value : value.add(offset);
        }

        private static FusionRecipe.Constant vector(
                FusionRecipe.Builder builder, String name, int width, DataType dataType) {
            return builder.addConstant(name, FusionRecipe.TensorSpec.fixed(dataType, width));
        }

        private static FusionRecipe.Constant matrix(
                FusionRecipe.Builder builder,
                String name,
                int rows,
                int columns,
                DataType dataType) {
            return builder.addConstant(
                    name, FusionRecipe.TensorSpec.fixed(dataType, rows, columns));
        }
    }

    private static final class BoundIndexedLocalTransformer implements AutoCloseable {

        private final FusionConstantBindings bindings;
        private final NDList resources;
        private final NDArray inputNormWeight;
        private final NDArray inputNormBias;
        private final NDArray attentionInputWeight;
        private final NDArray attentionInputBias;
        private final NDArray queryKeyValue;
        private final NDArray attentionOutput;
        private final NDArray attentionOutputBias;
        private final NDArray feedForwardInputWeight;
        private final NDArray feedForwardInputBias;
        private final NDArray expansion;
        private final NDArray expansionBias;
        private final NDArray projection;
        private final NDArray projectionBias;
        private final NDArray outputWeight;
        private final NDArray outputBias;

        private BoundIndexedLocalTransformer(
                FusionConstantBindings bindings,
                NDList resources,
                NDArray inputNormWeight,
                NDArray inputNormBias,
                NDArray attentionInputWeight,
                NDArray attentionInputBias,
                NDArray queryKeyValue,
                NDArray attentionOutput,
                NDArray attentionOutputBias,
                NDArray feedForwardInputWeight,
                NDArray feedForwardInputBias,
                NDArray expansion,
                NDArray expansionBias,
                NDArray projection,
                NDArray projectionBias,
                NDArray outputWeight,
                NDArray outputBias) {
            this.bindings = bindings;
            this.resources = resources;
            this.inputNormWeight = inputNormWeight;
            this.inputNormBias = inputNormBias;
            this.attentionInputWeight = attentionInputWeight;
            this.attentionInputBias = attentionInputBias;
            this.queryKeyValue = queryKeyValue;
            this.attentionOutput = attentionOutput;
            this.attentionOutputBias = attentionOutputBias;
            this.feedForwardInputWeight = feedForwardInputWeight;
            this.feedForwardInputBias = feedForwardInputBias;
            this.expansion = expansion;
            this.expansionBias = expansionBias;
            this.projection = projection;
            this.projectionBias = projectionBias;
            this.outputWeight = outputWeight;
            this.outputBias = outputBias;
        }

        @Override
        public void close() {
            resources.close();
        }
    }

    private static final class IndexedLocalPattern {

        private final float[] mask;
        private final int[] indices;

        private IndexedLocalPattern(float[] mask, int[] indices) {
            this.mask = mask;
            this.indices = indices;
        }

        private int presentCount(int batchCount) {
            int denseRows = batchCount * 4 * 29;
            int low = 0;
            int high = indices.length;
            while (low < high) {
                int middle = low + (high - low) / 2;
                if (indices[middle] < denseRows) {
                    low = middle + 1;
                } else {
                    high = middle;
                }
            }
            return low;
        }
    }

    private static final class TransformerFixture {

        private final FusionRecipe.Dimension batch;
        private final FusionRecipe.Input input;
        private final FusionRecipe.Constant attentionInputWeight;
        private final FusionRecipe.Constant attentionInputBias;
        private final FusionRecipe.Constant queryKeyValue;
        private final FusionRecipe.Constant attentionOutput;
        private final FusionRecipe.Constant attentionOutputBias;
        private final FusionRecipe.Constant feedForwardInputWeight;
        private final FusionRecipe.Constant feedForwardInputBias;
        private final FusionRecipe.Constant expansion;
        private final FusionRecipe.Constant expansionBias;
        private final FusionRecipe.Constant projection;
        private final FusionRecipe.Constant projectionBias;
        private final FusionRecipe.Constant outputWeight;
        private final FusionRecipe.Constant outputBias;
        private final FusionRecipe.Output output;
        private final FusionRecipe recipe;

        private TransformerFixture() {
            this(DataType.FLOAT16, 1);
        }

        private TransformerFixture(DataType dataType, int blockCount) {
            FusionRecipe.Builder builder = FusionRecipe.builder("transformer");
            batch = builder.addDimension("batch", 2);
            input = builder.addInput("input", FusionRecipe.TensorSpec.of(dataType, batch, 6, 256));
            attentionInputWeight = vector(builder, "attentionInputWeight", 256, DataType.FLOAT32);
            attentionInputBias = vector(builder, "attentionInputBias", 256, DataType.FLOAT32);
            queryKeyValue = matrix(builder, "qkv", 384, 256, dataType);
            attentionOutput = matrix(builder, "attentionOutput", 256, 128, dataType);
            attentionOutputBias = vector(builder, "attentionOutputBias", 256, dataType);
            feedForwardInputWeight =
                    vector(builder, "feedForwardInputWeight", 256, DataType.FLOAT32);
            feedForwardInputBias = vector(builder, "feedForwardInputBias", 256, DataType.FLOAT32);
            expansion = matrix(builder, "expansion", 512, 256, dataType);
            expansionBias = vector(builder, "expansionBias", 512, dataType);
            projection = matrix(builder, "projection", 256, 512, dataType);
            projectionBias = vector(builder, "projectionBias", 256, dataType);
            outputWeight = vector(builder, "outputWeight", 256, DataType.FLOAT32);
            outputBias = vector(builder, "outputBias", 256, DataType.FLOAT32);
            FusionRecipe.TransformerEncoderStackBuilder stackBuilder =
                    builder.transformerEncoderStack("stack", input, 4, 128, 512);
            for (int blockIndex = 0; blockIndex < blockCount; ++blockIndex) {
                stackBuilder.addBlock(
                        attentionInputWeight,
                        attentionInputBias,
                        queryKeyValue,
                        attentionOutput,
                        attentionOutputBias,
                        feedForwardInputWeight,
                        feedForwardInputBias,
                        expansion,
                        expansionBias,
                        projection,
                        projectionBias,
                        outputWeight,
                        outputBias);
            }
            FusionRecipe.TransformerEncoderStack stack = stackBuilder.build();
            output = builder.addOutput("output", stack);
            recipe = builder.build();
        }

        private FusionConstantBindings bindings(
                NDArray attentionInputWeightArray,
                NDArray attentionInputBiasArray,
                NDArray queryKeyValueArray,
                NDArray attentionOutputArray,
                NDArray attentionOutputBiasArray,
                NDArray feedForwardInputWeightArray,
                NDArray feedForwardInputBiasArray,
                NDArray expansionArray,
                NDArray expansionBiasArray,
                NDArray projectionArray,
                NDArray projectionBiasArray,
                NDArray outputWeightArray,
                NDArray outputBiasArray) {
            return FusionConstantBindings.builder(recipe)
                    .bind(attentionInputWeight, attentionInputWeightArray)
                    .bind(attentionInputBias, attentionInputBiasArray)
                    .bind(queryKeyValue, queryKeyValueArray)
                    .bind(attentionOutput, attentionOutputArray)
                    .bind(attentionOutputBias, attentionOutputBiasArray)
                    .bind(feedForwardInputWeight, feedForwardInputWeightArray)
                    .bind(feedForwardInputBias, feedForwardInputBiasArray)
                    .bind(expansion, expansionArray)
                    .bind(expansionBias, expansionBiasArray)
                    .bind(projection, projectionArray)
                    .bind(projectionBias, projectionBiasArray)
                    .bind(outputWeight, outputWeightArray)
                    .bind(outputBias, outputBiasArray)
                    .build();
        }

        private static FusionRecipe.Constant vector(
                FusionRecipe.Builder builder, String name, int width, DataType dataType) {
            return builder.addConstant(name, FusionRecipe.TensorSpec.fixed(dataType, width));
        }

        private static FusionRecipe.Constant matrix(
                FusionRecipe.Builder builder,
                String name,
                int rows,
                int columns,
                DataType dataType) {
            return builder.addConstant(
                    name, FusionRecipe.TensorSpec.fixed(dataType, rows, columns));
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
