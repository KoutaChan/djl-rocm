/*
 * Copyright 2026 KoutaChan.
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
import ai.djl.engine.Autocast;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.GradientCollector;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.Arrays;

/** Verifies portable structured-operation semantics and native PyTorch parity. */
public class StructuredAttentionTest {

    private static final int[] MAPPED_ATTENTION_SEEDS = {20260901, 20260903, 20260907, 20260909};
    private static final int MAPPED_ATTENTION_BFLOAT16_ROUNDING_SEED = 20260907;

    @Test
    public void relationAttentionUsesPairwiseBias() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray query = manager.zeros(new Shape(1, 1, 2, 1));
            NDArray key = manager.zeros(new Shape(1, 1, 2, 1));
            NDArray value = manager.create(new float[] {1f, 3f}, new Shape(1, 1, 2, 1));
            NDArray relationKeys = manager.zeros(new Shape(1, 1, 1, 2));
            NDArray relationBias =
                    manager.create(
                            new float[] {0f, 0f, 0f, (float) Math.log(3)}, new Shape(1, 1, 2, 2));
            NDArray relationIds = manager.create(new long[] {0, 1, 1, 0}, new Shape(2, 2));

            NDArray result =
                    NDArrays.relationBiasedScaledDotProductAttention(
                            query, key, value, relationKeys, relationBias, relationIds, 1.0);

            assertClose(result.toFloatArray(), new float[] {2f, 2.5f}, 1e-6f);
        }
    }

    @Test
    public void groupedAttentionCombinesSharedAndIndexedTokens() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray query = manager.zeros(new Shape(2, 1, 1));
            NDArray shared = manager.create(new float[] {0f, 1f, 0f, 3f}, new Shape(1, 2, 2));
            NDArray sharedDeltas = manager.zeros(new Shape(2, 2, 2));
            NDArray indexedDeltas =
                    manager.create(new float[] {0f, 0f, 0f, 4f}, new Shape(2, 1, 2));
            NDArray indexedIds = manager.create(new int[] {0, 1}, new Shape(2, 1));

            NDArray result =
                    NDArrays.groupedIndexedScaledDotProductAttention(
                            query, shared, sharedDeltas, indexedDeltas, indexedIds, 2, 1.0);

            assertClose(result.toFloatArray(), new float[] {2f, 3f}, 1e-6f);
        }
    }

    @Test
    public void groupedAttentionStrictlyExcludesPaddedIndexedTokens() {
        Engine engine = Engine.getInstance();
        verifyGroupedAttentionStrictPadding(engine, Device.cpu());
        if (engine.getGpuCount() > 0) {
            verifyGroupedAttentionStrictPadding(engine, Device.gpu());
        }
    }

    @Test
    public void groupedAttentionKeepsLegalUnderflowValueObservable() {
        Engine engine = Engine.getInstance();
        verifyGroupedAttentionLegalUnderflowValue(engine, Device.cpu());
        if (engine.getGpuCount() > 0) {
            verifyGroupedAttentionLegalUnderflowValue(engine, Device.gpu());
        }
    }

    @Test
    public void groupedAttentionLargeCommonScorePreservesGradientMass() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        for (DataType dataType : new DataType[] {DataType.FLOAT32, DataType.BFLOAT16}) {
            for (float magnitude : new float[] {1f, 8192f}) {
                try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                    NDArray query =
                            requiringGradient(
                                    manager.create(new float[] {magnitude}, new Shape(1, 1, 1))
                                            .toType(dataType, false));
                    NDArray shared =
                            requiringGradient(
                                    manager.create(
                                                    new float[] {magnitude, 2f, magnitude, 6f},
                                                    new Shape(1, 2, 2))
                                            .toType(dataType, false));
                    NDArray sharedDeltas =
                            requiringGradient(manager.zeros(new Shape(1, 2, 2), dataType));
                    NDArray indexedDeltas =
                            requiringGradient(manager.zeros(new Shape(1, 1, 2), dataType));
                    NDArray ids = manager.zeros(new Shape(1, 1), DataType.INT32);
                    NDArray output;
                    try (GradientCollector collector = engine.newGradientCollector()) {
                        output =
                                NDArrays.groupedIndexedScaledDotProductAttention(
                                        query, shared, sharedDeltas, indexedDeltas, ids, 1, 1.0);
                        collector.backward(output.sum());
                    }

                    // Equal scores must give each valid value gradient one half even when
                    // the common score is 2^26 and an FP32 absolute LSE loses log(2).
                    try (NDArray queryGradient = query.getGradient();
                            NDArray sharedGradient = shared.getGradient();
                            NDArray deltaGradient = sharedDeltas.getGradient();
                            NDArray indexedGradient = indexedDeltas.getGradient()) {
                        float[] actual =
                                sharedGradient.toType(DataType.FLOAT32, false).toFloatArray();
                        System.out.printf(
                                "GROUPED_ATTENTION_GRADIENT_MASS dtype=%s magnitude=%g values=%g,%g"
                                        + " mass=%g%n",
                                dataType, magnitude, actual[1], actual[3], actual[1] + actual[3]);
                        assertClose(
                                output.toType(DataType.FLOAT32, false).toFloatArray(),
                                new float[] {4f},
                                1e-6f);
                        float[] expected = {-magnitude, 0.5f, magnitude, 0.5f};
                        assertClose(actual, expected, 1e-6f);
                        assertClose(
                                deltaGradient.toType(DataType.FLOAT32, false).toFloatArray(),
                                expected,
                                1e-6f);
                        assertClose(
                                queryGradient.toType(DataType.FLOAT32, false).toFloatArray(),
                                new float[] {0f},
                                1e-6f);
                        assertAllZero(
                                indexedGradient.toType(DataType.FLOAT32, false).toFloatArray());
                    }
                }
            }
        }
    }

    private static void verifyGroupedAttentionStrictPadding(Engine engine, Device device) {
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray query = requiringGradient(manager.create(new float[] {1f}, new Shape(1, 1, 1)));
            NDArray shared =
                    requiringGradient(
                            manager.create(new float[] {-Float.MAX_VALUE, 2f}, new Shape(1, 1, 2)));
            NDArray sharedDeltas = requiringGradient(manager.zeros(new Shape(1, 1, 2)));
            NDArray indexedDeltas =
                    requiringGradient(
                            manager.create(
                                    new float[] {Float.MAX_VALUE, Float.NaN}, new Shape(1, 1, 2)));
            NDArray indexedIds = manager.zeros(new Shape(1, 1), DataType.INT32);
            NDArray output;
            try (GradientCollector collector = engine.newGradientCollector()) {
                output =
                        NDArrays.groupedIndexedScaledDotProductAttention(
                                query, shared, sharedDeltas, indexedDeltas, indexedIds, 1, 1.0);
                collector.backward(output.sum());
            }

            assertClose(output.toFloatArray(), new float[] {2f}, 0f);
            assertClose(query.getGradient().toFloatArray(), new float[] {0f}, 0f);
            assertClose(indexedDeltas.getGradient().toFloatArray(), new float[] {0f, 0f}, 0f);
        }
    }

    private static void verifyGroupedAttentionLegalUnderflowValue(Engine engine, Device device) {
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray query = manager.create(new float[] {1f}, new Shape(1, 1, 1));
            NDArray shared = manager.create(new float[] {0f, 1f}, new Shape(1, 1, 2));
            NDArray sharedDeltas = manager.zeros(new Shape(1, 1, 2));
            NDArray indexedDeltas =
                    manager.create(new float[] {-1000f, Float.NaN}, new Shape(1, 1, 2));
            NDArray indexedIds = manager.ones(new Shape(1, 1), DataType.INT32);
            NDArray output =
                    NDArrays.groupedIndexedScaledDotProductAttention(
                            query, shared, sharedDeltas, indexedDeltas, indexedIds, 1, 1.0);
            Assert.assertTrue(Float.isNaN(output.getFloat()));
        }
    }

    @Test
    public void mappedGroupedAttentionReadsSharedAndDeltaTables() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray query = manager.zeros(new Shape(2, 1, 1));
            NDArray shared =
                    manager.create(
                            new float[] {0f, 1f, 0f, 3f, 0f, 5f, 0f, 7f}, new Shape(2, 2, 2));
            NDArray groupIndices = manager.create(new int[] {1, 0});
            NDArray deltaTable = manager.create(new float[] {0f, 0f, 0f, 2f}, new Shape(2, 2));
            NDArray deltaIndices = manager.create(new int[] {0, 1, 1, 0}, new Shape(2, 2));
            NDArray indexedDeltas = manager.zeros(new Shape(2, 1, 2));
            NDArray indexedIds = manager.zeros(new Shape(2, 1), DataType.INT32);

            NDArray result =
                    NDArrays.mappedGroupedIndexedScaledDotProductAttention(
                            query,
                            shared,
                            groupIndices,
                            deltaTable,
                            deltaIndices,
                            indexedDeltas,
                            indexedIds,
                            1.0);

            assertClose(result.toFloatArray(), new float[] {7f, 3f}, 1e-6f);
        }
    }

    @Test
    public void groupedPackedAttentionSharesQueriesAndMasksMemory() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray query = manager.zeros(new Shape(1, 1, 1));
            NDArray packedKeyValue =
                    manager.create(
                            new float[] {0f, 1f, 0f, 3f, 0f, 2f, 0f, 10f}, new Shape(1, 2, 2, 2));
            NDArray mask = manager.create(new int[] {1, 1, 1, 0}, new Shape(1, 2, 2));

            NDArray result =
                    NDArrays.groupedPackedScaledDotProductAttention(
                            query, packedKeyValue, mask, 1, 1.0);

            Assert.assertEquals(result.getShape(), new Shape(1, 2, 1, 1));
            assertClose(result.toFloatArray(), new float[] {2f, 2f}, 1e-6f);
        }
    }

    @Test
    public void groupedPackedAttentionPreservesPortableGradients() {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(Device.cpu());
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray query = requiringGradient(manager.randomNormal(new Shape(2, 3, 8)));
            NDArray packedKeyValue =
                    requiringGradient(manager.randomNormal(new Shape(2, 4, 5, 16)));
            NDArray mask = manager.ones(new Shape(2, 4, 5), DataType.INT32);

            collector.backward(
                    NDArrays.groupedPackedScaledDotProductAttention(
                                    query, packedKeyValue, mask, 2, 0.5)
                            .sum());

            assertFiniteNonzeroGradient(query);
            assertFiniteNonzeroGradient(packedKeyValue);
        }
    }

    @Test
    public void groupedPackedAttentionProductionShapeMatchesPortableDtypeBoundaries() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            engine.setRandomSeed(20260913);
            try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                int batch = 2;
                int queryTokens = 34;
                int groups = 4;
                int keyTokens = 29;
                int heads = 4;
                int keyFeatures = 16;
                int valueFeatures = 16;
                Assert.assertTrue(keyTokens > valueFeatures);
                Assert.assertTrue(valueFeatures < 32);
                int queryWidth = heads * keyFeatures;
                int packedWidth = queryWidth + heads * valueFeatures;
                NDArray query =
                        manager.randomNormal(new Shape(batch, queryTokens, queryWidth), dataType);
                NDArray packedKeyValue =
                        manager.randomNormal(
                                new Shape(batch, groups, keyTokens, packedWidth), dataType);
                NDArray mask = manager.ones(new Shape(batch, groups, keyTokens), dataType);
                mask.set(new ai.djl.ndarray.index.NDIndex("..., -1"), 0);

                NDArray expected =
                        groupedPackedAttentionReference(query, packedKeyValue, mask, heads, 0.25);
                NDArray actual =
                        NDArrays.groupedPackedScaledDotProductAttention(
                                query, packedKeyValue, mask, heads, 0.25);

                Assert.assertEquals(actual.getDataType(), dataType);
                Assert.assertEquals(expected.getDataType(), dataType);

                float tolerance =
                        dataType == DataType.FLOAT32
                                ? 2e-4f
                                : dataType == DataType.FLOAT16 ? 3e-3f : 2e-2f;
                assertClose(
                        actual.toType(DataType.FLOAT32, false).toFloatArray(),
                        expected.toType(DataType.FLOAT32, false).toFloatArray(),
                        tolerance);
            }
        }
    }

    @Test
    public void groupedPackedAttentionNativeForwardDoesNotDependOnAutograd() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            engine.setRandomSeed(20260917);
            try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                int batch = 2;
                int queryTokens = 34;
                int groups = 4;
                int keyTokens = 29;
                int heads = 4;
                int queryWidth = 64;
                int packedWidth = 128;
                NDArray queryValues =
                        manager.randomNormal(new Shape(batch, queryTokens, queryWidth), dataType);
                NDArray packedValues =
                        manager.randomNormal(
                                new Shape(batch, groups, keyTokens, packedWidth), dataType);
                NDArray mask = manager.ones(new Shape(batch, groups, keyTokens));
                mask.set(new ai.djl.ndarray.index.NDIndex("..., -1"), 0);

                NDArray inference =
                        NDArrays.groupedPackedScaledDotProductAttention(
                                queryValues, packedValues, mask, heads, 0.25);
                NDArray trainingQuery = requiringGradient(queryValues.duplicate());
                NDArray trainingPacked = requiringGradient(packedValues.duplicate());
                try (GradientCollector collector = engine.newGradientCollector()) {
                    NDArray training =
                            NDArrays.groupedPackedScaledDotProductAttention(
                                    trainingQuery, trainingPacked, mask, heads, 0.25);

                    Assert.assertEquals(training.getShape(), inference.getShape());
                    Assert.assertEquals(training.getDataType(), inference.getDataType());
                    Assert.assertEquals(
                            training.toByteBuffer(),
                            inference.toByteBuffer(),
                            "autograd changed the grouped packed forward for " + dataType);
                    collector.backward(training.sum());
                }
            }
        }
    }

    @Test
    public void groupedPackedAttentionGradientsMatchPortableReference() {
        Engine engine = Engine.getInstance();
        engine.setRandomSeed(20260918);
        verifyGroupedPackedAttentionGradients(engine, Device.cpu(), DataType.FLOAT32, true, true);
        if (engine.getGpuCount() == 0) {
            return;
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            engine.setRandomSeed(20260918);
            verifyGroupedPackedAttentionGradients(engine, Device.gpu(), dataType, true, true);
        }
    }

    @Test
    public void groupedPackedAttentionProductionPrefixesMatchPortableReference() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        for (DataType dataType : new DataType[] {DataType.FLOAT32, DataType.BFLOAT16}) {
            for (int presentTokens : new int[] {29, 2, 4, 8, 16}) {
                engine.setRandomSeed(20260923);
                try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                    NDArray queryValues = manager.randomNormal(new Shape(2, 34, 64), dataType);
                    NDArray packedValues = manager.randomNormal(new Shape(2, 4, 29, 128), dataType);
                    NDArray mask = manager.zeros(new Shape(2, 4, 29), DataType.INT32);
                    mask.set(new ai.djl.ndarray.index.NDIndex("..., :{}", presentTokens), 1);
                    NDArray outputGradient =
                            manager.randomNormal(new Shape(2, 4, 34, 64), dataType).mul(0.125f);
                    NDArray query = requiringGradient(queryValues.duplicate());
                    NDArray packed = requiringGradient(packedValues.duplicate());
                    NDArray output;
                    try (GradientCollector collector = engine.newGradientCollector()) {
                        output =
                                NDArrays.groupedPackedScaledDotProductAttention(
                                        query, packed, mask, 4, 0.25);
                        collector.backward(output.mul(outputGradient).sum());
                    }
                    NDArray referenceQuery = requiringGradient(queryValues.duplicate());
                    NDArray referencePacked = requiringGradient(packedValues.duplicate());
                    NDArray referenceOutput;
                    try (GradientCollector collector = engine.newGradientCollector()) {
                        referenceOutput =
                                groupedPackedAttentionReference(
                                        referenceQuery, referencePacked, mask, 4, 0.25);
                        collector.backward(referenceOutput.mul(outputGradient).sum());
                    }

                    float tolerance = gradientTolerance(dataType);
                    assertClose(
                            output.toType(DataType.FLOAT32, false).toFloatArray(),
                            referenceOutput.toType(DataType.FLOAT32, false).toFloatArray(),
                            tolerance);
                    assertGradientClose(query, referenceQuery, tolerance);
                    assertGradientClose(packed, referencePacked, tolerance);
                    if (presentTokens < 29) {
                        try (NDArray gradient = packed.getGradient();
                                NDArray masked = gradient.get("..., {}:, :", presentTokens)) {
                            assertAllZero(masked.toType(DataType.FLOAT32, false).toFloatArray());
                        }
                    }
                }
            }
        }
    }

    @Test
    public void groupedPackedAttentionMaskedKeyLaneStillSuppliesQueryFeature() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        for (DataType dataType : new DataType[] {DataType.FLOAT32, DataType.BFLOAT16}) {
            try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                float[] queryValues = new float[16];
                queryValues[5] = 4f;
                float[] packedValues = new float[29 * 32];
                packedValues[27 * 32 + 5] = 1f;
                packedValues[28 * 32 + 5] = -1f;
                packedValues[27 * 32 + 16] = 1f;
                packedValues[28 * 32 + 16] = -1f;
                int[] maskValues = new int[29];
                maskValues[27] = 1;
                maskValues[28] = 1;
                NDArray query =
                        requiringGradient(
                                manager.create(queryValues, new Shape(1, 1, 16))
                                        .toType(dataType, false));
                NDArray packed =
                        requiringGradient(
                                manager.create(packedValues, new Shape(1, 1, 29, 32))
                                        .toType(dataType, false));
                NDArray mask = manager.create(maskValues, new Shape(1, 1, 29));
                NDArray output;
                try (GradientCollector collector = engine.newGradientCollector()) {
                    output =
                            NDArrays.groupedPackedScaledDotProductAttention(
                                    query, packed, mask, 1, 0.25);
                    collector.backward(output.get("..., 0").sum());
                }

                // Only key lanes 27/28 are present, but their dot products require lane 5.
                // Scores are +1/-1, so the first output is tanh(1).
                float probability = (float) (1.0 / (1.0 + Math.exp(-2.0)));
                float variance = probability * (1f - probability);
                float[] expectedOutput = new float[16];
                expectedOutput[0] = (float) Math.tanh(1.0);
                float[] expectedQueryGradient = new float[16];
                expectedQueryGradient[5] = variance;
                float[] expectedPackedGradient = new float[29 * 32];
                expectedPackedGradient[27 * 32 + 5] = 2f * variance;
                expectedPackedGradient[28 * 32 + 5] = -2f * variance;
                expectedPackedGradient[27 * 32 + 16] = probability;
                expectedPackedGradient[28 * 32 + 16] = 1f - probability;
                float tolerance = dataType == DataType.BFLOAT16 ? 1e-2f : 1e-6f;
                assertClose(
                        output.toType(DataType.FLOAT32, false).toFloatArray(),
                        expectedOutput,
                        tolerance);
                try (NDArray queryGradient = query.getGradient();
                        NDArray packedGradient = packed.getGradient()) {
                    assertClose(
                            queryGradient.toType(DataType.FLOAT32, false).toFloatArray(),
                            expectedQueryGradient,
                            tolerance);
                    assertClose(
                            packedGradient.toType(DataType.FLOAT32, false).toFloatArray(),
                            expectedPackedGradient,
                            tolerance);
                }
            }
        }
    }

    @Test
    public void groupedPackedAttentionNativeBackwardHonorsPartialGradients() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        engine.setRandomSeed(20260919);
        verifyGroupedPackedAttentionGradients(engine, Device.gpu(), DataType.BFLOAT16, true, false);
        engine.setRandomSeed(20260919);
        verifyGroupedPackedAttentionGradients(engine, Device.gpu(), DataType.BFLOAT16, false, true);
    }

    @Test
    public void groupedPackedAttentionNativeBackwardMatchesTiledBoundaryShape() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        engine.setRandomSeed(20260921);
        verifyGroupedPackedAttentionGradients(
                engine,
                Device.gpu(),
                DataType.FLOAT32,
                DataType.INT32,
                true,
                true,
                1,
                37,
                4,
                32,
                4,
                16,
                17);
    }

    @Test
    public void groupedPackedAttentionElementwiseBackwardMasksNonFiniteToken() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        engine.setRandomSeed(20260922);
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            int keyTokens = 32;
            int keyFeatures = 32;
            int valueFeatures = 64;
            NDArray query = requiringGradient(manager.randomNormal(new Shape(1, 1, keyFeatures)));
            NDArray packedValues =
                    manager.randomNormal(new Shape(1, 1, keyTokens, keyFeatures + valueFeatures));
            packedValues.set(new ai.djl.ndarray.index.NDIndex("..., -1, :"), Float.NaN);
            NDArray packed = requiringGradient(packedValues);
            NDArray mask = manager.ones(new Shape(1, 1, keyTokens), DataType.INT32);
            mask.set(new ai.djl.ndarray.index.NDIndex("..., -1"), 0);
            NDArray output;
            try (GradientCollector collector = engine.newGradientCollector()) {
                output =
                        NDArrays.groupedPackedScaledDotProductAttention(
                                query, packed, mask, 1, 1.0 / Math.sqrt(keyFeatures));
                collector.backward(output.sum());
            }

            for (float value : output.toFloatArray()) {
                Assert.assertTrue(Float.isFinite(value));
            }
            assertFiniteNonzeroGradient(query);
            try (NDArray maskedGradient = packed.getGradient().get("..., -1, :")) {
                assertAllZero(maskedGradient.toFloatArray());
            }
        }
    }

    @Test
    public void groupedPackedAttentionNativeBackwardSupportsEveryMaskDtype() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        for (DataType maskType :
                new DataType[] {
                    DataType.BOOLEAN,
                    DataType.INT32,
                    DataType.FLOAT32,
                    DataType.FLOAT16,
                    DataType.BFLOAT16
                }) {
            engine.setRandomSeed(20260920);
            verifyGroupedPackedAttentionGradients(
                    engine, Device.gpu(), DataType.BFLOAT16, maskType, true, true);
        }
    }

    @Test
    public void groupedPackedAttentionAllInvalidRowsReturnZeroAcrossDtypes() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            engine.setRandomSeed(20260914);
            try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                int heads = 2;
                NDArray query = manager.randomNormal(new Shape(1, 2, 8), dataType);
                NDArray packedKeyValue = manager.randomNormal(new Shape(1, 1, 3, 16), dataType);
                NDArray mask = manager.zeros(new Shape(1, 1, 3), dataType);

                NDArray expected =
                        groupedPackedAttentionReference(query, packedKeyValue, mask, heads, 0.5);
                NDArray actual =
                        NDArrays.groupedPackedScaledDotProductAttention(
                                query, packedKeyValue, mask, heads, 0.5);

                Assert.assertEquals(actual.getDataType(), dataType);
                Assert.assertEquals(expected.getDataType(), dataType);

                float[] expectedValues = expected.toType(DataType.FLOAT32, false).toFloatArray();
                float[] actualValues = actual.toType(DataType.FLOAT32, false).toFloatArray();

                assertAllZero(expectedValues);
                assertAllZero(actualValues);
            }
        }
    }

    @Test
    public void groupedPackedAttentionSupportsSharedMemoryBoundary() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        // With one key, the second shape exceeds 64 KiB only after the static ballot word.
        for (int valueFeatures : new int[] {16381, 16382}) {
            try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                NDArray query = manager.zeros(new Shape(1, 1, 1));
                NDArray packed = manager.ones(new Shape(1, 1, 1, 1 + valueFeatures));
                NDArray mask = manager.ones(new Shape(1, 1, 1), DataType.BOOLEAN);

                NDArray output =
                        NDArrays.groupedPackedScaledDotProductAttention(
                                query, packed, mask, 1, 1.0);

                Assert.assertEquals(output.getShape(), new Shape(1, 1, 1, valueFeatures));
                for (float value : output.toFloatArray()) {
                    Assert.assertEquals(value, 1.0f);
                }
            }
        }
    }

    @Test
    public void groupedPackedAttentionNonContiguousGpuInputsUsePortableFallback() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        engine.setRandomSeed(20260915);
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            int batch = 2;
            int queryTokens = 7;
            int groups = 3;
            int keyTokens = 5;
            int heads = 2;
            int keyFeatures = 4;
            int valueFeatures = 3;
            int queryWidth = heads * keyFeatures;
            int packedWidth = queryWidth + heads * valueFeatures;
            NDArray query =
                    manager.randomNormal(new Shape(batch, queryWidth, queryTokens)).swapAxes(1, 2);
            NDArray packedKeyValue =
                    manager.randomNormal(new Shape(batch, groups, packedWidth, keyTokens))
                            .swapAxes(2, 3);
            NDArray mask = manager.ones(new Shape(batch, groups, keyTokens), DataType.INT32);
            mask.set(new ai.djl.ndarray.index.NDIndex("..., -1"), 0);
            packedKeyValue.set(new ai.djl.ndarray.index.NDIndex("..., -1, :"), Float.NaN);

            NDArray expected =
                    groupedPackedAttentionReference(query, packedKeyValue, mask, heads, 0.5);
            NDArray actual =
                    NDArrays.groupedPackedScaledDotProductAttention(
                            query, packedKeyValue, mask, heads, 0.5);

            assertClose(actual.toFloatArray(), expected.toFloatArray(), 2e-4f);
        }
    }

    @Test
    public void groupedPackedAttentionFloat16FallbackReturnsZeroForFullyMaskedRows() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            NDArray query =
                    manager.randomNormal(new Shape(1, 4, 2), DataType.FLOAT16).swapAxes(1, 2);
            NDArray packedKeyValue =
                    manager.randomNormal(new Shape(1, 1, 8, 3), DataType.FLOAT16).swapAxes(2, 3);
            NDArray mask = manager.zeros(new Shape(1, 1, 3), DataType.INT32);

            NDArray output =
                    NDArrays.groupedPackedScaledDotProductAttention(
                            query, packedKeyValue, mask, 1, 0.5);

            assertAllZero(output.toType(DataType.FLOAT32, false).toFloatArray());
        }
    }

    @Test
    public void residualAddLayerNormUpdatesOwnedBuffer() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray residual = manager.create(new float[] {1f, 3f}, new Shape(1, 2));
            NDArray update = manager.ones(new Shape(1, 2));
            NDArray weight = manager.ones(new Shape(2));
            NDArray bias = manager.zeros(new Shape(2));

            NDArray normalized =
                    NDArrays.addToOwnedResidualAndLayerNorm(
                            residual, update, weight, bias, 1.0e-5f);

            assertClose(residual.toFloatArray(), new float[] {2f, 4f}, 1e-6f);
            assertClose(normalized.toFloatArray(), new float[] {-0.999995f, 0.999995f}, 1e-5f);
        }
    }

    @Test
    public void rowGatherAndScatterPreserveTrailingDimensions() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray source =
                    manager.create(
                            new float[] {
                                0f, 1f, 2f, 3f,
                                4f, 5f, 6f, 7f,
                                8f, 9f, 10f, 11f,
                                12f, 13f, 14f, 15f
                            },
                            new Shape(4, 2, 2));
            NDArray indices = manager.create(new int[] {3, 1});

            NDArray gathered = NDArrays.gatherRows(source, indices);
            NDArray scattered = NDArrays.scatterRows(gathered, indices, 5);

            Assert.assertEquals(gathered.getShape(), new Shape(2, 2, 2));
            assertClose(
                    gathered.toFloatArray(), new float[] {12f, 13f, 14f, 15f, 4f, 5f, 6f, 7f}, 0f);
            Assert.assertEquals(scattered.getShape(), new Shape(5, 2, 2));
            assertClose(
                    scattered.toFloatArray(),
                    new float[] {
                        0f, 0f, 0f, 0f,
                        4f, 5f, 6f, 7f,
                        0f, 0f, 0f, 0f,
                        12f, 13f, 14f, 15f,
                        0f, 0f, 0f, 0f
                    },
                    0f);
        }
    }

    @Test
    public void rowGatherAndScatterPreserveGradients() {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager();
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray source = manager.ones(new Shape(4, 2));
            source.setRequiresGradient(true);
            NDArray indices = manager.create(new int[] {2, 0, 2});

            collector.backward(NDArrays.gatherRows(source, indices).sum());

            assertClose(
                    source.getGradient().toFloatArray(),
                    new float[] {1f, 1f, 0f, 0f, 2f, 2f, 0f, 0f},
                    0f);
        }

        try (NDManager manager = engine.newBaseManager();
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray rows = manager.ones(new Shape(2, 2));
            rows.setRequiresGradient(true);
            NDArray indices = manager.create(new int[] {3, 1});

            collector.backward(NDArrays.scatterRows(rows, indices, 5).sum());

            assertClose(rows.getGradient().toFloatArray(), new float[] {1f, 1f, 1f, 1f}, 0f);
        }
    }

    @Test
    public void structuredAttentionPreservesAutograd() {
        Engine engine = Engine.getInstance();
        verifyStructuredAttentionGradients(engine, Device.cpu(), DataType.FLOAT32);
        if (engine.getGpuCount() > 0) {
            verifyStructuredAttentionGradients(engine, Device.gpu(), DataType.FLOAT32);
            verifyStructuredAttentionGradients(engine, Device.gpu(), DataType.BFLOAT16);
        }
    }

    @Test
    public void structuredAttentionHonorsPartialGradientRequests() {
        Engine engine = Engine.getInstance();
        verifyPartialGradientRequests(engine, Device.cpu(), DataType.FLOAT32);
        if (engine.getGpuCount() > 0) {
            verifyPartialGradientRequests(engine, Device.gpu(), DataType.BFLOAT16);
            verifyLargeRelationBiasGradient(engine);
        }
    }

    @Test
    public void attentionMaskOnlyGradientsMatchReferenceAcrossShapes() {
        Engine engine = Engine.getInstance();
        verifyAttentionMaskOnlyGradients(engine, Device.cpu(), DataType.FLOAT32);
        if (engine.getGpuCount() > 0) {
            for (DataType dataType :
                    new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
                verifyAttentionMaskOnlyGradients(engine, Device.gpu(), dataType);
            }
        }
    }

    private static void verifyAttentionMaskOnlyGradients(
            Engine engine, Device device, DataType dataType) {
        for (int[] geometry : new int[][] {{1, 5, 3, 7}, {3, 17, 33, 9}, {65, 67, 64, 31}}) {
            try (NDManager manager = engine.newBaseManager(device)) {
                int queries = geometry[0];
                int keys = geometry[1];
                int keyWidth = geometry[2];
                int valueWidth = geometry[3];
                NDArray query =
                        manager.randomNormal(new Shape(2, 3, queries, keyWidth))
                                .mul(0.125f)
                                .toType(dataType, false);
                NDArray key =
                        manager.randomNormal(new Shape(2, 3, keys, keyWidth))
                                .mul(0.125f)
                                .toType(dataType, false);
                NDArray value =
                        manager.randomNormal(new Shape(2, 3, keys, valueWidth))
                                .mul(0.125f)
                                .toType(dataType, false);
                NDArray mask =
                        requiringGradient(
                                manager.randomNormal(new Shape(1, 3, queries, keys))
                                        .mul(0.125f)
                                        .toType(dataType, false));
                NDArray referenceMask = requiringGradient(mask.toType(DataType.FLOAT32, true));
                double scale = 1.0 / Math.sqrt(keyWidth);
                NDArray actual;
                try (GradientCollector collector = engine.newGradientCollector()) {
                    actual =
                            query.getNDArrayInternal()
                                    .scaledDotProductAttention(
                                            key,
                                            value,
                                            mask.broadcast(2, 3, queries, keys),
                                            0.0,
                                            false,
                                            scale);
                    collector.backward(actual.sum());
                }
                NDArray expected;
                try (GradientCollector collector = engine.newGradientCollector()) {
                    expected =
                            query.toType(DataType.FLOAT32, false)
                                    .matMul(key.toType(DataType.FLOAT32, false).swapAxes(2, 3))
                                    .mul(scale)
                                    .add(referenceMask)
                                    .softmax(3)
                                    .matMul(value.toType(DataType.FLOAT32, false));
                    collector.backward(expected.sum());
                }
                float tolerance = gradientTolerance(dataType);
                assertClose(
                        actual.toType(DataType.FLOAT32, false).toFloatArray(),
                        expected.toFloatArray(),
                        tolerance);
                assertClose(
                        mask.getGradient().toType(DataType.FLOAT32, false).toFloatArray(),
                        referenceMask.getGradient().toFloatArray(),
                        tolerance);
                Assert.assertFalse(query.hasGradient());
                Assert.assertFalse(key.hasGradient());
                Assert.assertFalse(value.hasGradient());
            }
        }
    }

    @Test
    public void ownedResidualLayerNormRejectsAutograd() {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager()) {
            GradientCollector collector = engine.newGradientCollector();
            try {
                NDArray residual = manager.ones(new Shape(2, 4));
                residual.setRequiresGradient(true);
                NDArray update = manager.ones(new Shape(2, 4));
                NDArray weight = manager.ones(new Shape(4));
                NDArray bias = manager.zeros(new Shape(4));
                Assert.expectThrows(
                        RuntimeException.class,
                        () ->
                                NDArrays.addToOwnedResidualAndLayerNorm(
                                        residual, update, weight, bias, 1.0e-5f));
            } finally {
                collector.close();
            }
        }
    }

    @Test
    public void structuredOperationsDoNotRetainBatchArrays() {
        Engine engine = Engine.getInstance();
        try (NDManager modelManager = engine.newBaseManager(Device.cpu())) {
            NDArray relationKeys = modelManager.randomNormal(new Shape(1, 1, 2, 4));
            NDArray relationBias = modelManager.randomNormal(new Shape(1, 1, 2, 3));
            NDArray relationIds =
                    modelManager.create(new long[] {0, 1, 2, 3, 0, 1}, new Shape(2, 3));
            NDArray sharedKeyValues = modelManager.randomNormal(new Shape(1, 3, 4));
            NDArray indexedSharedIds = modelManager.create(new int[] {1, 2}, new Shape(2, 1));
            int retainedArrays = modelManager.getManagedArrays().size();

            for (int iteration = 0; iteration < 3; iteration++) {
                runStructuredStep(
                        modelManager,
                        relationKeys,
                        relationBias,
                        relationIds,
                        sharedKeyValues,
                        indexedSharedIds);
            }

            Assert.assertEquals(modelManager.getManagedArrays().size(), retainedArrays);
        }
    }

    @Test
    public void portableOperationsSupportArbitraryDimensions() {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(Device.cpu())) {
            verifyRelationAttention(manager);
            verifyGroupedAttention(manager);
            verifyMappedGroupedAttention(manager);
            verifyRelationAttentionLeadingDimensions(manager);
            verifyGroupedAttentionLeadingDimensions(manager);
            verifyGroupedPackedAttentionLeadingDimensions(manager);
            verifyResidualLayerNorm(manager);
        }
    }

    @Test
    public void nativeKernelsMatchPortableSemantics() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            verifyRelationAttention(manager);
            verifyGroupedAttention(manager);
            verifyMappedGroupedAttention(manager);
            verifyRelationAttentionLeadingDimensions(manager);
            verifyGroupedAttentionLeadingDimensions(manager);
            verifyGroupedPackedAttentionLeadingDimensions(manager);
            verifyResidualLayerNorm(manager);
        }
    }

    @Test
    public void mappedGroupedAttentionNativeInt32MatchesReferenceAcrossDtypes() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            for (int seed : MAPPED_ATTENTION_SEEDS) {
                engine.setRandomSeed(seed);
                try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                    AttentionComparison comparison =
                            verifyMappedGroupedAttentionForward(
                                    manager, dataType, MappedAttentionPath.INT32_NATIVE);
                    printMappedAttentionComparison(dataType, seed, comparison);
                }
            }
        }
    }

    @Test
    public void mappedGroupedAttentionInt64FallbackMatchesReferenceAcrossDtypes() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            for (int seed : MAPPED_ATTENTION_SEEDS) {
                engine.setRandomSeed(seed);
                try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                    AttentionComparison comparison =
                            verifyMappedGroupedAttentionForward(
                                    manager, dataType, MappedAttentionPath.INT64_EAGER_FALLBACK);
                    printMappedAttentionComparison(dataType, seed, comparison);
                }
            }
        }
    }

    @Test
    public void mappedGroupedAttentionNativeBfloat16PreservesEagerRoundingBoundaries() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        engine.setRandomSeed(MAPPED_ATTENTION_BFLOAT16_ROUNDING_SEED);
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            AttentionComparison comparison =
                    verifyMappedGroupedAttentionForward(
                            manager, DataType.BFLOAT16, MappedAttentionPath.INT32_NATIVE);
            System.out.printf(
                    "MAPPED_GROUPED_ATTENTION_BFLOAT16_ROUNDING seed=%d maxAbs=%g%n",
                    MAPPED_ATTENTION_BFLOAT16_ROUNDING_SEED, comparison.maximumAbsoluteError);
        }
    }

    @Test
    public void mappedGroupedAttentionAutogradMatchesReferenceAcrossDtypes() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            engine.setRandomSeed(20260911);
            verifyMappedGroupedAttentionGradients(engine, Device.gpu(), dataType);
            System.out.printf(
                    "MAPPED_GROUPED_ATTENTION_AUTOGRAD dtype=%s path=NATIVE seed=20260911%n",
                    dataType);
        }
    }

    @Test
    public void mappedGroupedAttentionNativeBackwardMatchesProductionGeometry() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        engine.setRandomSeed(20260913);
        verifyMappedGroupedAttentionProductionGradients(engine, DataType.BFLOAT16);
    }

    @Test
    public void mappedGroupedAttentionLargeQueryBackwardMatchesReferenceAcrossDtypes() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            for (int[] geometry : new int[][] {{1025, 1}, {2049, 3}}) {
                engine.setRandomSeed(20260924);
                verifyMappedGroupedAttentionProductionGradients(
                        engine, dataType, geometry[0], geometry[1], true, true, true, true);
            }
        }
    }

    @Test
    public void mappedGroupedAttentionLargeQueryBackwardHonorsOptionalTableGradient() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            engine.setRandomSeed(20260925);
            verifyMappedGroupedAttentionProductionGradients(
                    engine, dataType, 1025, 1, false, false, true, false);
            engine.setRandomSeed(20260925);
            verifyMappedGroupedAttentionProductionGradients(
                    engine, dataType, 1025, 3, true, true, false, true);
        }
    }

    @Test
    public void mappedGroupedAttentionCudaMatchesCpuAndStridedPortableGradients() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0 || JniUtils.getFusionBackend() != 1) {
            return;
        }
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            for (boolean strided : new boolean[] {false, true}) {
                Device referenceDevice = strided ? Device.gpu() : Device.cpu();
                try (NDManager referenceManager = engine.newBaseManager(referenceDevice);
                        NDManager actualManager = engine.newBaseManager(Device.gpu())) {
                    NDArray[] expected =
                            mappedCudaFixture(
                                    engine, referenceManager, dataType, true, strided, null);
                    NDArray[] actual =
                            mappedCudaFixture(
                                    engine, actualManager, dataType, false, strided, null);
                    assertMappedCudaFixtureClose(actual, expected, dataType);
                }
            }
        }
    }

    @Test
    public void mappedGroupedAttentionCudaAutocastPreservesPortableDtypeAndGradients() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0 || JniUtils.getFusionBackend() != 1) {
            return;
        }
        for (DataType dataType : new DataType[] {DataType.FLOAT16, DataType.BFLOAT16}) {
            try (NDManager manager = engine.newBaseManager(Device.gpu())) {
                NDArray[] expected =
                        mappedCudaFixture(engine, manager, dataType, true, false, dataType);
                NDArray[] actual =
                        mappedCudaFixture(engine, manager, dataType, false, false, dataType);
                Assert.assertEquals(actual[0].getDataType(), expected[0].getDataType());
                assertMappedCudaFixtureClose(actual, expected, dataType);
                System.out.printf(
                        "MAPPED_CUDA_AUTOCAST input=%s output=%s reference=%s%n",
                        dataType, actual[0].getDataType(), expected[0].getDataType());
            }
        }
    }

    @Test
    public void mappedGroupedAttentionSupportsResourceBoundedGeometries() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        // queries, heads, key features, value features, shared tokens, indexed tokens.
        // Cover warp tails, unequal/odd features, no indexed tokens, and scratch
        // requirements that select four, two, one, or no native warps per block.
        int[][] geometries = {
            {1, 1, 1, 3, 1, 0},
            {3, 3, 17, 33, 31, 7},
            {5, 2, 65, 129, 65, 19},
            {3, 1, 7, 11, 1021, 17},
            {3, 1, 7, 11, 2045, 17},
            {3, 1, 7, 11, 4093, 17},
            {3, 1, 7, 11, 6141, 17}
        };
        for (DataType dataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            for (int[] geometry : geometries) {
                for (boolean backward : new boolean[] {false, true}) {
                    // Repeated table rows require an FP32 accumulator in the reference;
                    // low-precision gather backward loses small contributions as rows grow.
                    try (NDManager referenceManager = engine.newBaseManager(Device.cpu());
                            NDManager manager = engine.newBaseManager(Device.gpu())) {
                        NDArray[] expected =
                                mappedCudaFixture(
                                        engine,
                                        referenceManager,
                                        dataType,
                                        true,
                                        false,
                                        null,
                                        geometry,
                                        backward);
                        NDArray[] actual =
                                mappedCudaFixture(
                                        engine, manager, dataType, false, false, null, geometry,
                                        backward);
                        assertMappedCudaFixtureClose(actual, expected, dataType);
                    } catch (AssertionError error) {
                        throw new AssertionError(
                                "mapped attention dtype="
                                        + dataType
                                        + " geometry="
                                        + Arrays.toString(geometry)
                                        + " backward="
                                        + backward,
                                error);
                    }
                }
            }
        }
    }

    private static NDArray[] mappedCudaFixture(
            Engine engine,
            NDManager manager,
            DataType dataType,
            boolean portableReference,
            boolean strided,
            DataType autocastType) {
        return mappedCudaFixture(
                engine,
                manager,
                dataType,
                portableReference,
                strided,
                autocastType,
                new int[] {5, 4, 8, 16, 34, 13},
                true);
    }

    @SuppressWarnings("try")
    private static NDArray[] mappedCudaFixture(
            Engine engine,
            NDManager manager,
            DataType dataType,
            boolean portableReference,
            boolean strided,
            DataType autocastType,
            int[] geometry,
            boolean backward) {
        final int queries = geometry[0];
        final int heads = geometry[1];
        final int keyFeatures = geometry[2];
        final int valueFeatures = geometry[3];
        final int sharedTokens = geometry[4];
        final int indexedTokens = geometry[5];
        final int packedWidth = heads * (keyFeatures + valueFeatures);
        Shape[] shapes = {
            new Shape(queries, heads, keyFeatures),
            new Shape(2, sharedTokens, packedWidth),
            new Shape(3, packedWidth),
            new Shape(queries, indexedTokens, packedWidth)
        };
        boolean floatReference = portableReference && !manager.getDevice().isGpu();
        NDArray[] inputs = new NDArray[shapes.length];
        for (int input = 0; input < inputs.length; ++input) {
            float[] data = new float[(int) shapes[input].size()];
            for (int index = 0; index < data.length; ++index) {
                boolean padded = input == 3 && (index / packedWidth % indexedTokens) % 3 == 0;
                data[index] = padded ? Float.NaN : ((index * 5 + input * 3) % 17 - 8) * 0.03125f;
            }
            NDArray values = manager.create(data, shapes[input]).toType(dataType, false);
            if (floatReference) {
                values = values.toType(DataType.FLOAT32, false);
            }
            if (strided && (input == 0 || input == 3)) {
                values = values.expandDims(-1).repeat(shapes[input].dimension(), 2).get("...,0");
            }
            inputs[input] = backward ? requiringGradient(values) : values;
        }
        int[] deltaIds = new int[queries * sharedTokens];
        int[] indexedIds = new int[queries * indexedTokens];
        for (int index = 0; index < deltaIds.length; ++index) {
            deltaIds[index] = index % 2;
        }
        for (int query = 0; query < queries; ++query) {
            for (int token = 0; token < indexedTokens; ++token) {
                indexedIds[query * indexedTokens + token] =
                        token % 3 == 0 ? 0 : token % sharedTokens + 1;
            }
        }
        int[] groupIds = new int[queries];
        for (int query = 0; query < queries; ++query) {
            groupIds[query] = (query + 1) % 2;
        }
        NDArray groups = manager.create(groupIds);
        NDArray relations = manager.create(deltaIds, new Shape(queries, sharedTokens));
        NDArray stored = manager.create(indexedIds, new Shape(queries, indexedTokens));
        float[] weightData = new float[queries * heads * valueFeatures];
        for (int index = 0; index < weightData.length; ++index) {
            weightData[index] = (index % 7 - 3) * 0.125f;
        }
        NDArray weights = manager.create(weightData, new Shape(queries, heads, valueFeatures));
        NDArray output;
        try (GradientCollector collector = engine.newGradientCollector()) {
            try (Autocast ignored =
                    autocastType == null
                            ? null
                            : engine.newAutocast(manager.getDevice(), autocastType, true)) {
                output =
                        portableReference
                                ? mappedGroupedAttentionReference(
                                        inputs[0],
                                        inputs[1],
                                        groups,
                                        inputs[2],
                                        relations,
                                        inputs[3],
                                        stored,
                                        1.0 / Math.sqrt(keyFeatures),
                                        floatReference && dataType == DataType.BFLOAT16)
                                : NDArrays.mappedGroupedIndexedScaledDotProductAttention(
                                        inputs[0],
                                        inputs[1],
                                        groups,
                                        inputs[2],
                                        relations,
                                        inputs[3],
                                        stored,
                                        1.0 / Math.sqrt(keyFeatures));
            }
            if (backward) {
                collector.backward(output.mul(weights).sum());
            }
        }
        Assert.assertEquals(output.getShape(), new Shape(queries, heads, valueFeatures));
        if (!backward) {
            return new NDArray[] {output};
        }
        NDArray[] results = {
            output,
            inputs[0].getGradient(),
            inputs[1].getGradient(),
            inputs[2].getGradient(),
            inputs[3].getGradient()
        };
        float[] indexedGradient = results[4].toType(DataType.FLOAT32, false).toFloatArray();
        for (int row = 0; row < indexedIds.length; ++row) {
            if (indexedIds[row] == 0) {
                for (int feature = 0; feature < packedWidth; ++feature) {
                    Assert.assertEquals(indexedGradient[row * packedWidth + feature], 0f);
                }
            }
        }
        assertAllZero(results[3].get("2,:").toType(DataType.FLOAT32, false).toFloatArray());
        return results;
    }

    private static void assertMappedCudaFixtureClose(
            NDArray[] actual, NDArray[] expected, DataType dataType) {
        for (int index = 0; index < actual.length; ++index) {
            Assert.assertEquals(actual[index].getShape(), expected[index].getShape());
            float[] values = actual[index].toType(DataType.FLOAT32, false).toFloatArray();
            float[] reference = expected[index].toType(DataType.FLOAT32, false).toFloatArray();
            for (float value : values) {
                Assert.assertTrue(Float.isFinite(value));
            }
            try {
                assertClose(values, reference, gradientTolerance(dataType));
            } catch (AssertionError error) {
                String[] resultNames = {
                    "output",
                    "query gradient",
                    "shared gradient",
                    "table gradient",
                    "indexed gradient"
                };
                throw new AssertionError("mapped attention " + resultNames[index], error);
            }
        }
    }

    @Test
    public void bfloat16FallbackComparisonDoesNotRelaxNativeTolerance() {
        float[] actualAtQuantizationBoundary = {1.53125f};
        float[] expected = {1.5625f};
        Assert.expectThrows(
                AssertionError.class,
                () ->
                        assertMappedAttentionClose(
                                actualAtQuantizationBoundary,
                                expected,
                                DataType.BFLOAT16,
                                MappedAttentionPath.INT32_NATIVE));

        AttentionComparison comparison =
                assertMappedAttentionClose(
                        actualAtQuantizationBoundary,
                        expected,
                        DataType.BFLOAT16,
                        MappedAttentionPath.INT64_EAGER_FALLBACK);
        Assert.assertEquals(comparison.maximumBfloat16Ulps, 4);

        Assert.expectThrows(
                AssertionError.class,
                () ->
                        assertMappedAttentionClose(
                                new float[] {1.484375f},
                                expected,
                                DataType.BFLOAT16,
                                MappedAttentionPath.INT64_EAGER_FALLBACK));
    }

    @Test
    public void nativeResidualLayerNormSupportsMixedPrecisionParameters() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            verifyResidualLayerNormDataTypes(
                    manager, DataType.FLOAT32, DataType.FLOAT16, DataType.FLOAT16, 5e-4f);
            verifyResidualLayerNormDataTypes(
                    manager, DataType.FLOAT32, DataType.BFLOAT16, DataType.BFLOAT16, 5e-4f);
            verifyResidualLayerNormDataTypes(
                    manager, DataType.FLOAT16, DataType.FLOAT16, DataType.FLOAT16, 2e-3f);
            verifyResidualLayerNormDataTypes(
                    manager, DataType.BFLOAT16, DataType.BFLOAT16, DataType.BFLOAT16, 2e-2f);
        }
    }

    @Test
    public void ownedResidualLayerNormFallsBackForUnsupportedNativeDtype() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            verifyResidualLayerNormDataTypes(
                    manager, DataType.FLOAT32, DataType.FLOAT64, DataType.FLOAT32, 5e-4f);
        }
    }

    @Test
    public void ownedResidualLayerNormRejectsUnintendedMixedPrecisionParameters() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            NDArray residual = manager.ones(new Shape(2, 17));
            NDArray update = manager.ones(new Shape(2, 17));
            NDArray weight = manager.ones(new Shape(17)).toType(DataType.FLOAT16, false);
            NDArray bias = manager.zeros(new Shape(17)).toType(DataType.FLOAT16, false);

            Assert.expectThrows(
                    RuntimeException.class,
                    () ->
                            NDArrays.addToOwnedResidualAndLayerNorm(
                                    residual, update, weight, bias, 1.0e-5f));
        }
    }

    private static void verifyRelationAttentionLeadingDimensions(NDManager manager) {
        int batch = 2;
        int alternatives = 3;
        int heads = 2;
        int queryTokens = 4;
        int keyTokens = 6;
        int keyFeatures = 5;
        int valueFeatures = 3;
        int relations = 7;
        NDArray query =
                manager.randomNormal(
                        new Shape(batch, alternatives, heads, queryTokens, keyFeatures));
        NDArray key =
                manager.randomNormal(new Shape(batch, alternatives, heads, keyTokens, keyFeatures));
        NDArray value =
                manager.randomNormal(
                        new Shape(batch, alternatives, heads, keyTokens, valueFeatures));
        NDArray relationKeys =
                manager.randomNormal(new Shape(batch, alternatives, heads, keyFeatures, relations));
        NDArray relationBias =
                manager.randomNormal(new Shape(batch, alternatives, 1, queryTokens, keyTokens));
        int[] ids = new int[batch * alternatives * queryTokens * keyTokens];
        for (int index = 0; index < ids.length; index++) {
            ids[index] = index % relations;
        }
        NDArray relationIds =
                manager.create(ids, new Shape(batch, alternatives, queryTokens, keyTokens));
        double scale = 0.31;

        NDArray expected =
                NDArrays.relationBiasedScaledDotProductAttention(
                        query.reshape(batch * alternatives, heads, queryTokens, keyFeatures),
                        key.reshape(batch * alternatives, heads, keyTokens, keyFeatures),
                        value.reshape(batch * alternatives, heads, keyTokens, valueFeatures),
                        relationKeys.reshape(batch * alternatives, heads, keyFeatures, relations),
                        relationBias.reshape(batch * alternatives, 1, queryTokens, keyTokens),
                        relationIds.reshape(batch * alternatives, queryTokens, keyTokens),
                        scale);
        NDArray actual =
                NDArrays.relationBiasedScaledDotProductAttention(
                        query, key, value, relationKeys, relationBias, relationIds, scale);

        Assert.assertEquals(
                actual.getShape(),
                new Shape(batch, alternatives, heads, queryTokens, valueFeatures));
        assertClose(
                actual.toFloatArray(), expected.reshape(actual.getShape()).toFloatArray(), 2e-4f);
    }

    private static void verifyGroupedAttentionLeadingDimensions(NDManager manager) {
        int batches = 2;
        int groupsPerBatch = 3;
        int queriesPerGroup = 4;
        int heads = 2;
        int keyFeatures = 5;
        int valueFeatures = 3;
        int sharedTokens = 7;
        int indexedTokens = 4;
        int packedWidth = heads * (keyFeatures + valueFeatures);
        int queryCount = batches * groupsPerBatch * queriesPerGroup;
        int groupCount = batches * groupsPerBatch;
        NDArray query =
                manager.randomNormal(
                        new Shape(batches, groupsPerBatch, queriesPerGroup, heads, keyFeatures));
        NDArray shared =
                manager.randomNormal(new Shape(batches, groupsPerBatch, sharedTokens, packedWidth));
        NDArray sharedDeltas =
                manager.randomNormal(
                        new Shape(
                                batches,
                                groupsPerBatch,
                                queriesPerGroup,
                                sharedTokens,
                                packedWidth));
        NDArray indexedDeltas =
                manager.randomNormal(
                        new Shape(
                                batches,
                                groupsPerBatch,
                                queriesPerGroup,
                                indexedTokens,
                                packedWidth));
        int[] ids = new int[queryCount * indexedTokens];
        for (int index = 0; index < ids.length; index++) {
            ids[index] = index % (sharedTokens + 1);
        }
        NDArray indexedIds =
                manager.create(
                        ids, new Shape(batches, groupsPerBatch, queriesPerGroup, indexedTokens));
        double scale = 0.29;

        NDArray expected =
                NDArrays.groupedIndexedScaledDotProductAttention(
                        query.reshape(queryCount, heads, keyFeatures),
                        shared.reshape(groupCount, sharedTokens, packedWidth),
                        sharedDeltas.reshape(queryCount, sharedTokens, packedWidth),
                        indexedDeltas.reshape(queryCount, indexedTokens, packedWidth),
                        indexedIds.reshape(queryCount, indexedTokens),
                        queriesPerGroup,
                        scale);
        NDArray actual =
                NDArrays.groupedIndexedScaledDotProductAttention(
                        query,
                        shared,
                        sharedDeltas,
                        indexedDeltas,
                        indexedIds,
                        queriesPerGroup,
                        scale);

        Assert.assertEquals(
                actual.getShape(),
                new Shape(batches, groupsPerBatch, queriesPerGroup, heads, valueFeatures));
        assertClose(
                actual.toFloatArray(), expected.reshape(actual.getShape()).toFloatArray(), 2e-4f);
    }

    private static void verifyGroupedPackedAttentionLeadingDimensions(NDManager manager) {
        int batches = 2;
        int alternatives = 3;
        int queryTokens = 4;
        int groups = 2;
        int keyTokens = 5;
        int heads = 2;
        int keyFeatures = 3;
        int valueFeatures = 4;
        int queryWidth = heads * keyFeatures;
        int packedWidth = heads * (keyFeatures + valueFeatures);
        NDArray query =
                manager.randomNormal(new Shape(batches, alternatives, queryTokens, queryWidth));
        NDArray packedKeyValue =
                manager.randomNormal(
                        new Shape(batches, alternatives, groups, keyTokens, packedWidth));
        NDArray mask = manager.ones(new Shape(batches, alternatives, groups, keyTokens));
        mask.set(new ai.djl.ndarray.index.NDIndex("0,0,0,-1"), 0);
        double scale = 0.37;

        NDArray expected =
                groupedPackedAttentionReference(
                        query.reshape(batches * alternatives, queryTokens, queryWidth),
                        packedKeyValue.reshape(
                                batches * alternatives, groups, keyTokens, packedWidth),
                        mask.reshape(batches * alternatives, groups, keyTokens),
                        heads,
                        scale);
        NDArray actual =
                NDArrays.groupedPackedScaledDotProductAttention(
                        query, packedKeyValue, mask, heads, scale);

        Assert.assertEquals(
                actual.getShape(),
                new Shape(batches, alternatives, groups, queryTokens, heads * valueFeatures));
        assertClose(
                actual.toFloatArray(), expected.reshape(actual.getShape()).toFloatArray(), 2e-4f);
    }

    private static NDArray groupedPackedAttentionReference(
            NDArray query, NDArray packedKeyValue, NDArray mask, int heads, double scale) {
        Shape queryShape = query.getShape();
        Shape packedShape = packedKeyValue.getShape();
        long batch = queryShape.get(0);
        long queryTokens = queryShape.get(1);
        long queryWidth = queryShape.get(2);
        long groups = packedShape.get(1);
        long keyTokens = packedShape.get(2);
        long keyFeatures = queryWidth / heads;
        long valueWidth = packedShape.get(3) - queryWidth;
        long valueFeatures = valueWidth / heads;
        NDArray queries =
                query.reshape(batch, queryTokens, heads, keyFeatures)
                        .swapAxes(1, 2)
                        .expandDims(1)
                        .broadcast(batch, groups, heads, queryTokens, keyFeatures);
        NDArray keys =
                packedKeyValue
                        .get("...,0:{}", queryWidth)
                        .reshape(batch, groups, keyTokens, heads, keyFeatures)
                        .swapAxes(2, 3);
        NDArray values =
                packedKeyValue
                        .get("...,{}:{}", queryWidth, packedShape.get(3))
                        .reshape(batch, groups, keyTokens, heads, valueFeatures)
                        .swapAxes(2, 3);
        NDArray tokenValid = mask.neq(0).reshape(batch, groups, 1, keyTokens, 1);
        keys = NDArrays.where(tokenValid.broadcast(keys.getShape()), keys, keys.zerosLike());
        values =
                NDArrays.where(tokenValid.broadcast(values.getShape()), values, values.zerosLike());
        NDArray valid =
                tokenValid
                        .reshape(batch, groups, 1, 1, keyTokens)
                        .broadcast(batch, groups, heads, queryTokens, keyTokens);
        NDArray scores = queries.matMul(keys.swapAxes(3, 4)).mul(scale);
        NDArray safeScores = NDArrays.where(valid, scores, scores.zerosLike());
        NDArray present = valid.sum(new int[] {4}, true).gt(0).broadcast(valid.getShape());
        NDArray attentionValid = NDArrays.where(present, valid, valid.onesLike());
        NDArray probabilities =
                NDArrays.where(
                        valid,
                        NDArrays.where(
                                        attentionValid,
                                        safeScores,
                                        scores.zerosLike().add(Float.NEGATIVE_INFINITY))
                                .softmax(4),
                        scores.zerosLike());
        return probabilities
                .matMul(values)
                .swapAxes(2, 3)
                .reshape(batch, groups, queryTokens, valueWidth);
    }

    private static void verifyRelationAttention(NDManager manager) {
        NDArray query = manager.randomNormal(new Shape(2, 3, 7, 6));
        NDArray key = manager.randomNormal(new Shape(2, 3, 5, 6));
        NDArray value = manager.randomNormal(new Shape(2, 3, 5, 4));
        NDArray relationKeys = manager.randomNormal(new Shape(1, 3, 6, 11));
        NDArray relationBias = manager.randomNormal(new Shape(1, 3, 7, 5));
        long[] ids = new long[7 * 5];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = i % 11;
        }
        NDArray relationIds = manager.create(ids, new Shape(7, 5));
        double scale = 0.23;

        NDArray expected =
                relationAttentionReference(
                        query, key, value, relationKeys, relationBias, relationIds, scale);
        NDArray actual =
                NDArrays.relationBiasedScaledDotProductAttention(
                        query, key, value, relationKeys, relationBias, relationIds, scale);
        assertClose(actual.toFloatArray(), expected.toFloatArray(), 2e-4f);
    }

    private static void verifyStructuredAttentionGradients(
            Engine engine, Device device, DataType dataType) {
        verifyRelationAttentionGradients(engine, device, dataType);
        verifyGroupedAttentionGradients(engine, device, dataType);
        verifyMappedGroupedAttentionGradients(engine, device, dataType);
    }

    private static void verifyPartialGradientRequests(
            Engine engine, Device device, DataType dataType) {
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray query = manager.randomNormal(new Shape(2, 2, 3, 4), dataType);
            NDArray key = manager.randomNormal(new Shape(2, 2, 5, 4), dataType);
            NDArray value = manager.randomNormal(new Shape(2, 2, 5, 3), dataType);
            NDArray relationKeys = manager.randomNormal(new Shape(1, 2, 4, 7), dataType);
            NDArray relationBias =
                    requiringGradient(manager.randomNormal(new Shape(1, 2, 3, 5), dataType));
            NDArray relationIds =
                    manager.create(
                            new int[] {0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0},
                            new Shape(3, 5));
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        NDArrays.relationBiasedScaledDotProductAttention(
                                query, key, value, relationKeys, relationBias, relationIds, 0.31);
                collector.backward(output.mul(output).sum());
            }

            assertFiniteNonzeroGradient(relationBias);
            Assert.assertFalse(query.hasGradient());
            Assert.assertFalse(key.hasGradient());
            Assert.assertFalse(value.hasGradient());
            Assert.assertFalse(relationKeys.hasGradient());
        }

        try (NDManager manager = engine.newBaseManager(device)) {
            int heads = 2;
            int keyFeatures = 3;
            int packedWidth = heads * (keyFeatures + 4);
            NDArray query =
                    requiringGradient(
                            manager.randomNormal(new Shape(4, heads, keyFeatures), dataType));
            NDArray shared = manager.randomNormal(new Shape(2, 5, packedWidth), dataType);
            NDArray sharedDeltas = manager.randomNormal(new Shape(4, 5, packedWidth), dataType);
            NDArray indexedDeltas = manager.randomNormal(new Shape(4, 3, packedWidth), dataType);
            NDArray indexedIds =
                    manager.create(new int[] {1, 2, 3, 2, 0, 4, 3, 4, 5, 4, 5, 1}, new Shape(4, 3));
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        NDArrays.groupedIndexedScaledDotProductAttention(
                                query, shared, sharedDeltas, indexedDeltas, indexedIds, 2, 0.27);
                collector.backward(output.mul(output).sum());
            }

            assertFiniteNonzeroGradient(query);
            Assert.assertFalse(shared.hasGradient());
            Assert.assertFalse(sharedDeltas.hasGradient());
            Assert.assertFalse(indexedDeltas.hasGradient());
        }

        verifyMappedGroupedAttentionPartialGradients(
                engine, device, dataType, true, false, false, false);
        verifyMappedGroupedAttentionPartialGradients(
                engine, device, dataType, false, true, false, false);
        verifyMappedGroupedAttentionPartialGradients(
                engine, device, dataType, false, false, true, false);
        verifyMappedGroupedAttentionPartialGradients(
                engine, device, dataType, false, false, false, true);
    }

    private static void verifyMappedGroupedAttentionPartialGradients(
            Engine engine,
            Device device,
            DataType dataType,
            boolean queryGradient,
            boolean sharedGradient,
            boolean deltaTableGradient,
            boolean indexedDeltaGradient) {
        try (NDManager manager = engine.newBaseManager(device)) {
            int queries = 4;
            int heads = 2;
            int keyFeatures = 3;
            int valueFeatures = 4;
            int sharedTokens = 5;
            int indexedTokens = 3;
            int packedWidth = heads * (keyFeatures + valueFeatures);
            NDArray queryValues =
                    manager.randomNormal(new Shape(queries, heads, keyFeatures), dataType);
            NDArray sharedValues =
                    manager.randomNormal(new Shape(2, sharedTokens, packedWidth), dataType);
            NDArray deltaTableValues = manager.randomNormal(new Shape(3, packedWidth), dataType);
            NDArray indexedDeltaValues =
                    manager.randomNormal(new Shape(queries, indexedTokens, packedWidth), dataType);
            NDArray query = queryValues.duplicate();
            NDArray shared = sharedValues.duplicate();
            NDArray deltaTable = deltaTableValues.duplicate();
            NDArray indexedDeltas = indexedDeltaValues.duplicate();
            if (queryGradient) {
                query.setRequiresGradient(true);
            }
            if (sharedGradient) {
                shared.setRequiresGradient(true);
            }
            if (deltaTableGradient) {
                deltaTable.setRequiresGradient(true);
            }
            if (indexedDeltaGradient) {
                indexedDeltas.setRequiresGradient(true);
            }
            NDArray groupIndices = manager.create(new int[] {1, 0, 1, 0});
            NDArray deltaIndices =
                    manager.create(
                            new int[] {
                                0, 1, 2, 0, 1,
                                1, 2, 0, 1, 2,
                                2, 0, 1, 2, 0,
                                0, 1, 2, 0, 1
                            },
                            new Shape(queries, sharedTokens));
            NDArray indexedIds =
                    manager.create(
                            new int[] {1, 0, 3, 2, 4, 0, 5, 1, 2, 0, 4, 3},
                            new Shape(queries, indexedTokens));
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        NDArrays.mappedGroupedIndexedScaledDotProductAttention(
                                query,
                                shared,
                                groupIndices,
                                deltaTable,
                                deltaIndices,
                                indexedDeltas,
                                indexedIds,
                                0.27);
                collector.backward(output.mul(output).sum());
            }

            // Compare once-rounded FP32 accumulations, including when only one
            // shared input requests a gradient. BF16 gather atomics are not an oracle.
            NDArray referenceQuery = queryValues.toType(DataType.FLOAT32, true);
            NDArray referenceShared = sharedValues.toType(DataType.FLOAT32, true);
            NDArray referenceDeltaTable = deltaTableValues.toType(DataType.FLOAT32, true);
            NDArray referenceIndexedDeltas = indexedDeltaValues.toType(DataType.FLOAT32, true);
            referenceQuery.setRequiresGradient(queryGradient);
            referenceShared.setRequiresGradient(sharedGradient);
            referenceDeltaTable.setRequiresGradient(deltaTableGradient);
            referenceIndexedDeltas.setRequiresGradient(indexedDeltaGradient);
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        mappedGroupedAttentionReference(
                                referenceQuery,
                                referenceShared,
                                groupIndices,
                                referenceDeltaTable,
                                deltaIndices,
                                referenceIndexedDeltas,
                                indexedIds,
                                0.27,
                                dataType == DataType.BFLOAT16);
                collector.backward(output.mul(output).sum());
            }

            float tolerance = gradientTolerance(dataType);

            if (queryGradient) {
                assertFiniteNonzeroGradient(query);
                assertGradientClose(query, referenceQuery, tolerance, true);
            } else {
                Assert.assertFalse(query.hasGradient());
                Assert.assertFalse(referenceQuery.hasGradient());
            }
            if (sharedGradient) {
                assertFiniteNonzeroGradient(shared);
                assertGradientClose(shared, referenceShared, tolerance, true);
            } else {
                Assert.assertFalse(shared.hasGradient());
                Assert.assertFalse(referenceShared.hasGradient());
            }
            if (deltaTableGradient) {
                assertFiniteNonzeroGradient(deltaTable);
                assertGradientClose(deltaTable, referenceDeltaTable, tolerance, true);
            } else {
                Assert.assertFalse(deltaTable.hasGradient());
                Assert.assertFalse(referenceDeltaTable.hasGradient());
            }
            if (indexedDeltaGradient) {
                assertFiniteNonzeroGradient(indexedDeltas);
                assertGradientClose(indexedDeltas, referenceIndexedDeltas, tolerance, true);
            } else {
                Assert.assertFalse(indexedDeltas.hasGradient());
                Assert.assertFalse(referenceIndexedDeltas.hasGradient());
            }
        }
    }

    private static void verifyLargeRelationBiasGradient(Engine engine) {
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            int batch = 64;
            int heads = 4;
            int tokens = 34;
            int features = 16;
            int relations = 15;
            NDArray query =
                    manager.randomNormal(
                            new Shape(batch, heads, tokens, features), DataType.BFLOAT16);
            NDArray key =
                    manager.randomNormal(
                            new Shape(batch, heads, tokens, features), DataType.BFLOAT16);
            NDArray value =
                    manager.randomNormal(
                            new Shape(batch, heads, tokens, features), DataType.BFLOAT16);
            NDArray relationKeys =
                    manager.randomNormal(
                            new Shape(1, heads, features, relations), DataType.BFLOAT16);
            NDArray relationBias =
                    requiringGradient(
                            manager.randomNormal(
                                    new Shape(1, heads, tokens, tokens), DataType.BFLOAT16));
            int[] storedIds = new int[tokens * tokens];
            for (int index = 0; index < storedIds.length; ++index) {
                storedIds[index] = index % relations;
            }
            NDArray relationIds = manager.create(storedIds, new Shape(tokens, tokens));
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        NDArrays.relationBiasedScaledDotProductAttention(
                                query,
                                key,
                                value,
                                relationKeys,
                                relationBias,
                                relationIds,
                                1.0 / Math.sqrt(features));
                collector.backward(output.sum());
            }

            assertFiniteNonzeroGradient(relationBias);
        }
    }

    @Test
    public void groupedPackedAttentionAllInvalidRowsMaskScoreGradients() {
        Engine engine = Engine.getInstance();
        verifyAllInvalidGroupedPackedAttentionGradients(engine, Device.cpu(), DataType.FLOAT32);
        if (engine.getGpuCount() > 0) {
            verifyAllInvalidGroupedPackedAttentionGradients(
                    engine, Device.gpu(), DataType.BFLOAT16);
        }
    }

    @Test
    public void groupedPackedAttentionStrictlyExcludesMaskedTokens() {
        Engine engine = Engine.getInstance();
        verifyGroupedPackedAttentionStrictMask(engine, Device.cpu());
        if (engine.getGpuCount() > 0) {
            verifyGroupedPackedAttentionStrictMask(engine, Device.gpu());
        }
    }

    @Test
    public void groupedPackedAttentionKeepsLegalUnderflowValueObservable() {
        Engine engine = Engine.getInstance();
        verifyGroupedPackedAttentionLegalUnderflowValue(engine, Device.cpu());
        if (engine.getGpuCount() > 0) {
            verifyGroupedPackedAttentionLegalUnderflowValue(engine, Device.gpu());
        }
    }

    private static void verifyGroupedPackedAttentionStrictMask(Engine engine, Device device) {
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray query = requiringGradient(manager.create(new float[] {1f}, new Shape(1, 1, 1)));
            NDArray packed =
                    requiringGradient(
                            manager.create(
                                    new float[] {-Float.MAX_VALUE, 1f, 0f, Float.NaN},
                                    new Shape(1, 1, 2, 2)));
            NDArray mask = manager.create(new int[] {1, 0}, new Shape(1, 1, 2));
            NDArray output;
            try (GradientCollector collector = engine.newGradientCollector()) {
                output =
                        NDArrays.groupedPackedScaledDotProductAttention(
                                query, packed, mask, 1, 1.0);
                collector.backward(output.sum());
            }

            try (NDArray queryGradient = query.getGradient();
                    NDArray packedGradient = packed.getGradient()) {
                assertClose(output.toFloatArray(), new float[] {1f}, 0f);
                assertClose(queryGradient.toFloatArray(), new float[] {0f}, 0f);
                assertClose(packedGradient.toFloatArray(), new float[] {0f, 1f, 0f, 0f}, 0f);
            }
        }
    }

    private static void verifyGroupedPackedAttentionLegalUnderflowValue(
            Engine engine, Device device) {
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray query = manager.create(new float[] {1f}, new Shape(1, 1, 1));
            NDArray packed =
                    manager.create(new float[] {0f, 1f, -1000f, Float.NaN}, new Shape(1, 1, 2, 2));
            NDArray mask = manager.ones(new Shape(1, 1, 2), DataType.INT32);
            NDArray output =
                    NDArrays.groupedPackedScaledDotProductAttention(query, packed, mask, 1, 1.0);
            Assert.assertTrue(Float.isNaN(output.getFloat()));
        }
    }

    private static void verifyGroupedPackedAttentionGradients(
            Engine engine,
            Device device,
            DataType dataType,
            boolean queryGradient,
            boolean packedGradient) {
        verifyGroupedPackedAttentionGradients(
                engine, device, dataType, DataType.FLOAT32, queryGradient, packedGradient);
    }

    private static void verifyGroupedPackedAttentionGradients(
            Engine engine,
            Device device,
            DataType dataType,
            DataType maskType,
            boolean queryGradient,
            boolean packedGradient) {
        verifyGroupedPackedAttentionGradients(
                engine,
                device,
                dataType,
                maskType,
                queryGradient,
                packedGradient,
                2,
                7,
                3,
                5,
                2,
                4,
                3);
    }

    private static void verifyGroupedPackedAttentionGradients(
            Engine engine,
            Device device,
            DataType dataType,
            DataType maskType,
            boolean queryGradient,
            boolean packedGradient,
            int batch,
            int queryTokens,
            int groups,
            int keyTokens,
            int heads,
            int keyFeatures,
            int valueFeatures) {
        try (NDManager manager = engine.newBaseManager(device)) {
            int queryWidth = heads * keyFeatures;
            int packedWidth = queryWidth + heads * valueFeatures;
            NDArray queryValues =
                    manager.randomNormal(new Shape(batch, queryTokens, queryWidth), dataType);
            NDArray packedValues =
                    manager.randomNormal(
                            new Shape(batch, groups, keyTokens, packedWidth), dataType);
            NDArray mask = manager.ones(new Shape(batch, groups, keyTokens), maskType);
            mask.set(new ai.djl.ndarray.index.NDIndex("..., -1"), 0);
            NDArray outputGradient =
                    manager.randomNormal(
                            new Shape(batch, groups, queryTokens, heads * valueFeatures), dataType);

            NDArray query = queryValues.duplicate();
            NDArray packed = packedValues.duplicate();
            if (queryGradient) {
                query.setRequiresGradient(true);
            }
            if (packedGradient) {
                packed.setRequiresGradient(true);
            }
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        NDArrays.groupedPackedScaledDotProductAttention(
                                query, packed, mask, heads, 0.5);
                collector.backward(output.mul(outputGradient).sum());
            }

            NDArray referenceQuery = queryValues.duplicate();
            NDArray referencePacked = packedValues.duplicate();
            if (queryGradient) {
                referenceQuery.setRequiresGradient(true);
            }
            if (packedGradient) {
                referencePacked.setRequiresGradient(true);
            }
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        groupedPackedAttentionReference(
                                referenceQuery, referencePacked, mask, heads, 0.5);
                collector.backward(output.mul(outputGradient).sum());
            }

            float tolerance = gradientTolerance(dataType);
            if (queryGradient) {
                assertFiniteNonzeroGradient(query);
                float maximumDifference = assertGradientClose(query, referenceQuery, tolerance);
                System.out.printf(
                        "GROUPED_PACKED_ATTENTION_GRADIENT tensor=query dtype=%s mask=%s"
                                + " maxAbs=%s%n",
                        dataType, maskType, maximumDifference);
            } else {
                Assert.assertFalse(query.hasGradient());
            }
            if (packedGradient) {
                assertFiniteNonzeroGradient(packed);
                float maximumDifference = assertGradientClose(packed, referencePacked, tolerance);
                System.out.printf(
                        "GROUPED_PACKED_ATTENTION_GRADIENT tensor=packed dtype=%s mask=%s"
                                + " maxAbs=%s%n",
                        dataType, maskType, maximumDifference);
                try (NDArray packedGradientValues = packed.getGradient();
                        NDArray maskedTokenGradient = packedGradientValues.get("..., -1, :");
                        NDArray maskedTokenGradientValues =
                                maskedTokenGradient.toType(DataType.FLOAT32, false)) {
                    assertAllZero(maskedTokenGradientValues.toFloatArray());
                }
            } else {
                Assert.assertFalse(packed.hasGradient());
            }
        }
    }

    private static void verifyAllInvalidGroupedPackedAttentionGradients(
            Engine engine, Device device, DataType dataType) {
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray query = requiringGradient(manager.ones(new Shape(1, 1, 2), dataType));
            NDArray packed =
                    requiringGradient(
                            manager.create(
                                            new float[] {
                                                Float.NaN, Float.NaN, Float.NaN, Float.NaN,
                                                Float.NaN, Float.NaN, Float.NaN, Float.NaN
                                            },
                                            new Shape(1, 1, 2, 4))
                                    .toType(dataType, false));
            NDArray mask = manager.zeros(new Shape(1, 1, 2), DataType.INT32);
            NDArray output;
            try (GradientCollector collector = engine.newGradientCollector()) {
                output =
                        NDArrays.groupedPackedScaledDotProductAttention(
                                query, packed, mask, 1, 1.0);
                collector.backward(output.sum());
            }

            try (NDArray queryGradient = query.getGradient();
                    NDArray packedGradient = packed.getGradient();
                    NDArray queryGradientValues = queryGradient.toType(DataType.FLOAT32, false);
                    NDArray packedGradientValues = packedGradient.toType(DataType.FLOAT32, false)) {
                assertAllZero(output.toType(DataType.FLOAT32, false).toFloatArray());
                assertClose(queryGradientValues.toFloatArray(), new float[] {0f, 0f}, 0f);
                assertClose(
                        packedGradientValues.toFloatArray(),
                        new float[] {0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f},
                        0f);
            }
        }
    }

    private static void verifyRelationAttentionGradients(
            Engine engine, Device device, DataType dataType) {
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray queryValues = manager.randomNormal(new Shape(2, 2, 3, 4), dataType);
            NDArray keyValues = manager.randomNormal(new Shape(2, 2, 5, 4), dataType);
            NDArray valueValues = manager.randomNormal(new Shape(2, 2, 5, 3), dataType);
            NDArray relationKeyValues = manager.randomNormal(new Shape(1, 2, 4, 7), dataType);
            NDArray relationBiasValues = manager.randomNormal(new Shape(1, 2, 3, 5), dataType);
            NDArray relationIds =
                    manager.create(
                            new int[] {0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0},
                            new Shape(3, 5));

            NDArray query = requiringGradient(queryValues.duplicate());
            NDArray key = requiringGradient(keyValues.duplicate());
            NDArray value = requiringGradient(valueValues.duplicate());
            NDArray relationKeys = requiringGradient(relationKeyValues.duplicate());
            NDArray relationBias = requiringGradient(relationBiasValues.duplicate());
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        NDArrays.relationBiasedScaledDotProductAttention(
                                query, key, value, relationKeys, relationBias, relationIds, 0.31);
                collector.backward(output.mul(output).sum());
            }

            NDArray referenceQuery = requiringGradient(queryValues.duplicate());
            NDArray referenceKey = requiringGradient(keyValues.duplicate());
            NDArray referenceValue = requiringGradient(valueValues.duplicate());
            NDArray referenceRelationKeys = requiringGradient(relationKeyValues.duplicate());
            NDArray referenceRelationBias = requiringGradient(relationBiasValues.duplicate());
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        relationAttentionReference(
                                referenceQuery,
                                referenceKey,
                                referenceValue,
                                referenceRelationKeys,
                                referenceRelationBias,
                                relationIds,
                                0.31);
                collector.backward(output.mul(output).sum());
            }

            assertFiniteNonzeroGradient(query, key, value, relationKeys, relationBias);
            float tolerance = structuredAttentionGradientTolerance(dataType);
            assertGradientClose(query, referenceQuery, tolerance);
            assertGradientClose(key, referenceKey, tolerance);
            assertGradientClose(value, referenceValue, tolerance);
            assertGradientClose(relationKeys, referenceRelationKeys, tolerance);
            assertGradientClose(relationBias, referenceRelationBias, tolerance);
        }
    }

    private static void verifyGroupedAttentionGradients(
            Engine engine, Device device, DataType dataType) {
        try (NDManager manager = engine.newBaseManager(device)) {
            int heads = 2;
            int keyFeatures = 3;
            int packedWidth = heads * (keyFeatures + 4);
            NDArray queryValues = manager.randomNormal(new Shape(4, heads, keyFeatures), dataType);
            NDArray sharedValues = manager.randomNormal(new Shape(2, 5, packedWidth), dataType);
            NDArray sharedDeltaValues =
                    manager.randomNormal(new Shape(4, 5, packedWidth), dataType);
            NDArray indexedDeltaValues =
                    manager.randomNormal(new Shape(4, 3, packedWidth), dataType);
            NDArray indexedIds =
                    manager.create(new int[] {1, 2, 3, 2, 0, 4, 3, 4, 5, 4, 5, 1}, new Shape(4, 3));

            NDArray query = requiringGradient(queryValues.duplicate());
            NDArray shared = requiringGradient(sharedValues.duplicate());
            NDArray sharedDeltas = requiringGradient(sharedDeltaValues.duplicate());
            NDArray indexedDeltas = requiringGradient(indexedDeltaValues.duplicate());
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        NDArrays.groupedIndexedScaledDotProductAttention(
                                query, shared, sharedDeltas, indexedDeltas, indexedIds, 2, 0.27);
                collector.backward(output.mul(output).sum());
            }

            NDArray referenceQuery = requiringGradient(queryValues.duplicate());
            NDArray referenceShared = requiringGradient(sharedValues.duplicate());
            NDArray referenceSharedDeltas = requiringGradient(sharedDeltaValues.duplicate());
            NDArray referenceIndexedDeltas = requiringGradient(indexedDeltaValues.duplicate());
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        groupedAttentionReference(
                                referenceQuery,
                                referenceShared,
                                referenceSharedDeltas,
                                referenceIndexedDeltas,
                                indexedIds,
                                2,
                                0.27);
                collector.backward(output.mul(output).sum());
            }

            assertFiniteNonzeroGradient(query, shared, sharedDeltas, indexedDeltas);
            float tolerance = structuredAttentionGradientTolerance(dataType);
            assertGradientClose(query, referenceQuery, tolerance);
            assertGradientClose(shared, referenceShared, tolerance);
            assertGradientClose(sharedDeltas, referenceSharedDeltas, tolerance);
            assertGradientClose(indexedDeltas, referenceIndexedDeltas, tolerance);
        }
    }

    private static void verifyMappedGroupedAttentionGradients(
            Engine engine, Device device, DataType dataType) {
        try (NDManager manager = engine.newBaseManager(device)) {
            int queries = 5;
            int heads = 2;
            int keyFeatures = 3;
            int valueFeatures = 4;
            int sharedTokens = 5;
            int indexedTokens = 3;
            int packedWidth = heads * (keyFeatures + valueFeatures);
            NDArray queryValues =
                    manager.randomNormal(new Shape(queries, heads, keyFeatures), dataType);
            NDArray sharedValues =
                    manager.randomNormal(new Shape(3, sharedTokens, packedWidth), dataType);
            NDArray deltaTableValues = manager.randomNormal(new Shape(7, packedWidth), dataType);
            NDArray indexedDeltaValues =
                    manager.randomNormal(new Shape(queries, indexedTokens, packedWidth), dataType);
            NDArray outputGradientValues =
                    manager.randomNormal(new Shape(queries, heads, valueFeatures), dataType);
            NDArray groupIndices = manager.create(new int[] {2, 0, 2, 1, 0});
            NDArray deltaIndices =
                    manager.create(
                            new int[] {
                                0, 1, 2, 3, 4,
                                1, 2, 3, 4, 5,
                                2, 3, 4, 5, 6,
                                3, 4, 5, 6, 0,
                                4, 5, 6, 0, 1
                            },
                            new Shape(queries, sharedTokens));
            NDArray indexedIds =
                    manager.create(
                            new int[] {1, 2, 3, 2, 0, 4, 3, 4, 5, 4, 5, 1, 5, 1, 0},
                            new Shape(queries, indexedTokens));

            NDArray query = requiringGradient(queryValues.duplicate());
            NDArray shared = requiringGradient(sharedValues.duplicate());
            NDArray deltaTable = requiringGradient(deltaTableValues.duplicate());
            NDArray indexedDeltas = requiringGradient(indexedDeltaValues.duplicate());
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        NDArrays.mappedGroupedIndexedScaledDotProductAttention(
                                query,
                                shared,
                                groupIndices,
                                deltaTable,
                                deltaIndices,
                                indexedDeltas,
                                indexedIds,
                                0.27);
                collector.backward(output.mul(outputGradientValues).sum());
            }

            NDArray referenceQuery = requiringGradient(queryValues.duplicate());
            NDArray referenceShared = requiringGradient(sharedValues.duplicate());
            NDArray referenceDeltaTable = requiringGradient(deltaTableValues.duplicate());
            NDArray referenceIndexedDeltas = requiringGradient(indexedDeltaValues.duplicate());
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        mappedGroupedAttentionReference(
                                referenceQuery,
                                referenceShared,
                                groupIndices,
                                referenceDeltaTable,
                                deltaIndices,
                                referenceIndexedDeltas,
                                indexedIds,
                                0.27);
                collector.backward(output.mul(outputGradientValues).sum());
            }

            assertFiniteNonzeroGradient(query, shared, deltaTable, indexedDeltas);
            float tolerance = structuredAttentionGradientTolerance(dataType);
            assertGradientClose(query, referenceQuery, tolerance);
            assertGradientClose(shared, referenceShared, tolerance);
            assertGradientClose(deltaTable, referenceDeltaTable, tolerance);
            assertGradientClose(indexedDeltas, referenceIndexedDeltas, tolerance);
        }
    }

    private static void verifyMappedGroupedAttentionProductionGradients(
            Engine engine, DataType dataType) {
        verifyMappedGroupedAttentionProductionGradients(
                engine, dataType, 7, 17, true, true, true, true);
    }

    private static void verifyMappedGroupedAttentionProductionGradients(
            Engine engine,
            DataType dataType,
            int queries,
            int usedDeltaRows,
            boolean queryGradient,
            boolean sharedGradient,
            boolean deltaTableGradient,
            boolean indexedDeltaGradient) {
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            int groups = 3;
            int heads = 4;
            int keyFeatures = 8;
            int valueFeatures = 16;
            int sharedTokens = 34;
            int indexedTokens = 13;
            int deltaRows = 17;
            int packedWidth = heads * (keyFeatures + valueFeatures);
            NDArray queryValues =
                    manager.randomNormal(new Shape(queries, heads, keyFeatures), dataType);
            NDArray sharedValues =
                    manager.randomNormal(new Shape(groups, sharedTokens, packedWidth), dataType);
            NDArray deltaTableValues =
                    manager.randomNormal(new Shape(deltaRows, packedWidth), dataType);
            NDArray indexedDeltaValues =
                    manager.randomNormal(new Shape(queries, indexedTokens, packedWidth), dataType);
            NDArray outputGradientValues =
                    manager.randomNormal(new Shape(queries, heads, valueFeatures), dataType);
            // Keep shared-gradient magnitudes comparable as query count grows; the reference
            // still sums every contribution independently through gather's backward.
            if (queries > 7) {
                outputGradientValues = outputGradientValues.mul((float) (1.0 / Math.sqrt(queries)));
            }
            int[] groupIds = new int[queries];
            int[] deltaIds = new int[queries * sharedTokens];
            int[] indexedIds = new int[queries * indexedTokens];
            for (int query = 0; query < queries; query++) {
                groupIds[query] = (query * 2 + 1) % groups;
            }
            for (int index = 0; index < deltaIds.length; index++) {
                deltaIds[index] = (index * 5 + 3) % usedDeltaRows;
            }
            for (int index = 0; index < indexedIds.length; index++) {
                indexedIds[index] = index % 5 == 0 ? 0 : (index * 7 + 3) % sharedTokens + 1;
            }
            NDArray groupIndices = manager.create(groupIds);
            NDArray deltaIndices = manager.create(deltaIds, new Shape(queries, sharedTokens));
            NDArray storedIndexedIds =
                    manager.create(indexedIds, new Shape(queries, indexedTokens));

            NDArray query = queryValues.duplicate();
            NDArray shared = sharedValues.duplicate();
            NDArray deltaTable = deltaTableValues.duplicate();
            NDArray indexedDeltas = indexedDeltaValues.duplicate();
            query.setRequiresGradient(queryGradient);
            shared.setRequiresGradient(sharedGradient);
            deltaTable.setRequiresGradient(deltaTableGradient);
            indexedDeltas.setRequiresGradient(indexedDeltaGradient);
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        NDArrays.mappedGroupedIndexedScaledDotProductAttention(
                                query,
                                shared,
                                groupIndices,
                                deltaTable,
                                deltaIndices,
                                indexedDeltas,
                                storedIndexedIds,
                                1.0 / Math.sqrt(keyFeatures));
                collector.backward(output.mul(outputGradientValues).sum());
            }

            NDArray referenceQuery = queryValues.duplicate();
            NDArray referenceShared = sharedValues.duplicate();
            NDArray referenceDeltaTable = deltaTableValues.duplicate();
            NDArray referenceIndexedDeltas = indexedDeltaValues.duplicate();
            // Repeated mapped rows must accumulate in FP32, matching the operation's contract.
            // Low-precision index_select backward rounds thousands of atomic contributions and
            // is not a reference for a final, once-rounded shared-table gradient.
            DataType referenceType = queries > 7 ? DataType.FLOAT32 : dataType;
            referenceQuery = referenceQuery.toType(referenceType, false);
            referenceShared = referenceShared.toType(referenceType, false);
            referenceDeltaTable = referenceDeltaTable.toType(referenceType, false);
            referenceIndexedDeltas = referenceIndexedDeltas.toType(referenceType, false);
            referenceQuery.setRequiresGradient(queryGradient);
            referenceShared.setRequiresGradient(sharedGradient);
            referenceDeltaTable.setRequiresGradient(deltaTableGradient);
            referenceIndexedDeltas.setRequiresGradient(indexedDeltaGradient);
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray referenceOutput =
                        mappedGroupedAttentionReference(
                                referenceQuery,
                                referenceShared,
                                groupIndices,
                                referenceDeltaTable,
                                deltaIndices,
                                referenceIndexedDeltas,
                                storedIndexedIds,
                                1.0 / Math.sqrt(keyFeatures),
                                queries > 7 && dataType == DataType.BFLOAT16);
                collector.backward(
                        referenceOutput
                                .mul(outputGradientValues.toType(referenceType, false))
                                .sum());
            }

            if (indexedDeltaGradient) {
                try (NDArray indexedGradient = indexedDeltas.getGradient();
                        NDArray values = indexedGradient.toType(DataType.FLOAT32, false)) {
                    float[] gradients = values.toFloatArray();
                    for (int queryIndex = 0; queryIndex < queries; queryIndex++) {
                        for (int token = 0; token < indexedTokens; token++) {
                            if (indexedIds[queryIndex * indexedTokens + token] != 0) {
                                continue;
                            }
                            int offset = (queryIndex * indexedTokens + token) * packedWidth;
                            for (int feature = 0; feature < packedWidth; feature++) {
                                Assert.assertEquals(
                                        gradients[offset + feature],
                                        0f,
                                        "padded indexed gradient at query="
                                                + queryIndex
                                                + ", token="
                                                + token
                                                + ", feature="
                                                + feature);
                            }
                        }
                    }
                }
            }

            NDArray[] actualGradients = {query, shared, deltaTable, indexedDeltas};
            NDArray[] referenceGradients = {
                referenceQuery, referenceShared, referenceDeltaTable, referenceIndexedDeltas
            };
            boolean[] requestedGradients = {
                queryGradient, sharedGradient, deltaTableGradient, indexedDeltaGradient
            };
            float tolerance = gradientTolerance(dataType);
            float maximumGradientDifference = 0f;
            for (int index = 0; index < actualGradients.length; index++) {
                if (requestedGradients[index]) {
                    assertFiniteNonzeroGradient(actualGradients[index]);
                    maximumGradientDifference =
                            Math.max(
                                    maximumGradientDifference,
                                    assertGradientClose(
                                            actualGradients[index],
                                            referenceGradients[index],
                                            tolerance,
                                            queries > 7));
                } else {
                    Assert.assertFalse(actualGradients[index].hasGradient());
                    Assert.assertFalse(referenceGradients[index].hasGradient());
                }
            }
            if (deltaTableGradient && usedDeltaRows < deltaRows) {
                try (NDArray gradient = deltaTable.getGradient();
                        NDArray unused = gradient.get("{}:,:", usedDeltaRows)) {
                    assertAllZero(unused.toType(DataType.FLOAT32, false).toFloatArray());
                }
            }
            System.out.printf(
                    "MAPPED_GROUPED_ATTENTION_BACKWARD dtype=%s queries=%d usedDeltaRows=%d"
                            + " tableGradient=%s queryGradient=%s sharedGradient=%s"
                            + " indexedGradient=%s maxAbs=%g%n",
                    dataType,
                    queries,
                    usedDeltaRows,
                    deltaTableGradient,
                    queryGradient,
                    sharedGradient,
                    indexedDeltaGradient,
                    maximumGradientDifference);
        }
    }

    private static float gradientTolerance(DataType dataType) {
        switch (dataType) {
            case FLOAT32:
                return 2e-5f;
            case FLOAT16:
                return 2e-3f;
            case BFLOAT16:
                return 2e-2f;
            default:
                throw new IllegalArgumentException("Unsupported gradient data type: " + dataType);
        }
    }

    private static float structuredAttentionGradientTolerance(DataType dataType) {
        return dataType == DataType.BFLOAT16 ? 8e-2f : gradientTolerance(dataType);
    }

    private static void runStructuredStep(
            NDManager modelManager,
            NDArray relationKeys,
            NDArray relationBias,
            NDArray relationIds,
            NDArray sharedKeyValues,
            NDArray indexedSharedIds) {
        try (NDManager batchManager = modelManager.newSubManager()) {
            NDArray query = batchManager.randomNormal(new Shape(1, 1, 2, 2));
            NDArray key = batchManager.randomNormal(new Shape(1, 1, 3, 2));
            NDArray value = batchManager.randomNormal(new Shape(1, 1, 3, 2));
            NDArray groupedQuery = batchManager.randomNormal(new Shape(2, 1, 2));
            NDArray sharedDeltas = batchManager.randomNormal(new Shape(2, 3, 4));
            NDArray indexedDeltas = batchManager.randomNormal(new Shape(2, 1, 4));

            NDArray relationOutput =
                    NDArrays.relationBiasedScaledDotProductAttention(
                            query,
                            key,
                            value,
                            relationKeys,
                            relationBias,
                            relationIds,
                            1.0 / Math.sqrt(2));
            Assert.assertSame(relationOutput.getManager(), batchManager);
            NDArray groupedOutput =
                    NDArrays.groupedIndexedScaledDotProductAttention(
                            groupedQuery,
                            sharedKeyValues,
                            sharedDeltas,
                            indexedDeltas,
                            indexedSharedIds,
                            2,
                            1.0 / Math.sqrt(2));
            Assert.assertSame(groupedOutput.getManager(), batchManager);
        }
    }

    private static void verifyGroupedAttention(NDManager manager) {
        verifyGroupedAttentionShape(manager, 2, 3, 3, 5, 9, 7, 4, false);
        verifyGroupedAttentionShape(manager, 3, 5, 4, 8, 16, 34, 13, false);
        verifyGroupedAttentionShape(manager, 1, 7, 5, 33, 37, 65, 17, true);
        verifyGroupedAttentionShape(manager, 1, 3, 2, 7, 11, 9, 5, false, DataType.FLOAT64, 2e-5f);
        verifyStridedGroupedAttention(manager);
    }

    private static void verifyMappedGroupedAttention(NDManager manager) {
        int queries = 7;
        int heads = 3;
        int keyFeatures = 5;
        int valueFeatures = 9;
        int sharedTokens = 11;
        int indexedTokens = 4;
        int packedWidth = heads * (keyFeatures + valueFeatures);
        NDArray query = manager.randomNormal(new Shape(queries, heads, keyFeatures));
        NDArray shared = manager.randomNormal(new Shape(4, sharedTokens, packedWidth));
        NDArray groupIndices = manager.create(new int[] {3, 0, 3, 1, 2, 0, 2});
        NDArray deltaTable = manager.randomNormal(new Shape(13, packedWidth));
        int[] deltaIds = new int[queries * sharedTokens];
        int[] indexedIds = new int[queries * indexedTokens];
        for (int index = 0; index < deltaIds.length; ++index) {
            deltaIds[index] = index % 13;
        }
        for (int index = 0; index < indexedIds.length; ++index) {
            indexedIds[index] = index % (sharedTokens + 1);
        }
        NDArray deltaIndices = manager.create(deltaIds, new Shape(queries, sharedTokens));
        NDArray indexedDeltas =
                manager.randomNormal(new Shape(queries, indexedTokens, packedWidth));
        NDArray storedIndexedIds = manager.create(indexedIds, new Shape(queries, indexedTokens));
        double scale = 0.39;

        NDArray expected =
                mappedGroupedAttentionReference(
                        query,
                        shared,
                        groupIndices,
                        deltaTable,
                        deltaIndices,
                        indexedDeltas,
                        storedIndexedIds,
                        scale);
        NDArray actual =
                NDArrays.mappedGroupedIndexedScaledDotProductAttention(
                        query,
                        shared,
                        groupIndices,
                        deltaTable,
                        deltaIndices,
                        indexedDeltas,
                        storedIndexedIds,
                        scale);

        assertClose(actual.toFloatArray(), expected.toFloatArray(), 2e-4f);
    }

    private static AttentionComparison verifyMappedGroupedAttentionForward(
            NDManager manager, DataType dataType, MappedAttentionPath path) {
        int queries = 7;
        int heads = 4;
        int keyFeatures = 8;
        int valueFeatures = 16;
        int sharedTokens = 34;
        int indexedTokens = 6;
        int packedWidth = heads * (keyFeatures + valueFeatures);
        NDArray query = manager.randomNormal(new Shape(queries, heads, keyFeatures), dataType);
        NDArray shared = manager.randomNormal(new Shape(4, sharedTokens, packedWidth), dataType);
        NDArray deltaTable = manager.randomNormal(new Shape(13, packedWidth), dataType);
        NDArray indexedDeltas =
                manager.randomNormal(new Shape(queries, indexedTokens, packedWidth), dataType);
        int[] groupIds = {3, 0, 3, 1, 2, 0, 2};
        int[] deltaIds = new int[queries * sharedTokens];
        int[] indexedIds = new int[queries * indexedTokens];
        for (int index = 0; index < deltaIds.length; ++index) {
            deltaIds[index] = (index * 5 + 3) % 13;
        }
        for (int index = 0; index < indexedIds.length; ++index) {
            indexedIds[index] = index % 5 == 0 ? 0 : (index * 7) % sharedTokens + 1;
        }
        NDArray groups = manager.create(groupIds);
        NDArray deltas = manager.create(deltaIds, new Shape(queries, sharedTokens));
        NDArray storedIndexedIds = manager.create(indexedIds, new Shape(queries, indexedTokens));
        double scale = 1.0 / Math.sqrt(keyFeatures);

        NDArray expected =
                mappedGroupedAttentionReference(
                        query,
                        shared,
                        groups,
                        deltaTable,
                        deltas,
                        indexedDeltas,
                        storedIndexedIds,
                        scale);
        boolean useLongIndices = path == MappedAttentionPath.INT64_EAGER_FALLBACK;
        NDArray actual =
                NDArrays.mappedGroupedIndexedScaledDotProductAttention(
                        query,
                        shared,
                        useLongIndices ? groups.toType(DataType.INT64, false) : groups,
                        deltaTable,
                        useLongIndices ? deltas.toType(DataType.INT64, false) : deltas,
                        indexedDeltas,
                        useLongIndices
                                ? storedIndexedIds.toType(DataType.INT64, false)
                                : storedIndexedIds,
                        scale);
        return assertMappedAttentionClose(
                actual.toType(DataType.FLOAT32, false).toFloatArray(),
                expected.toType(DataType.FLOAT32, false).toFloatArray(),
                dataType,
                path);
    }

    private static void verifyGroupedAttentionShape(
            NDManager manager,
            int groups,
            int queriesPerGroup,
            int heads,
            int keyFeatures,
            int valueFeatures,
            int sharedTokens,
            int indexedTokens,
            boolean longIndices) {
        verifyGroupedAttentionShape(
                manager,
                groups,
                queriesPerGroup,
                heads,
                keyFeatures,
                valueFeatures,
                sharedTokens,
                indexedTokens,
                longIndices,
                DataType.FLOAT32,
                2e-4f);
    }

    private static void verifyGroupedAttentionShape(
            NDManager manager,
            int groups,
            int queriesPerGroup,
            int heads,
            int keyFeatures,
            int valueFeatures,
            int sharedTokens,
            int indexedTokens,
            boolean longIndices,
            DataType dataType,
            float tolerance) {
        int queries = groups * queriesPerGroup;
        int packedWidth = heads * (keyFeatures + valueFeatures);
        NDArray query =
                manager.randomNormal(new Shape(queries, heads, keyFeatures))
                        .toType(dataType, false);
        NDArray shared =
                manager.randomNormal(new Shape(groups, sharedTokens, packedWidth))
                        .toType(dataType, false);
        NDArray sharedDeltas =
                manager.randomNormal(new Shape(queries, sharedTokens, packedWidth))
                        .toType(dataType, false);
        NDArray indexedDeltas =
                manager.randomNormal(new Shape(queries, indexedTokens, packedWidth))
                        .toType(dataType, false);
        int[] ids = new int[queries * indexedTokens];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = i % (sharedTokens + 1);
        }
        NDArray indexedIds =
                longIndices
                        ? manager.create(
                                java.util.Arrays.stream(ids).asLongStream().toArray(),
                                new Shape(queries, indexedTokens))
                        : manager.create(ids, new Shape(queries, indexedTokens));
        double scale = 0.41;

        NDArray expected =
                groupedAttentionReference(
                        query,
                        shared,
                        sharedDeltas,
                        indexedDeltas,
                        indexedIds,
                        queriesPerGroup,
                        scale);
        NDArray actual =
                NDArrays.groupedIndexedScaledDotProductAttention(
                        query,
                        shared,
                        sharedDeltas,
                        indexedDeltas,
                        indexedIds,
                        queriesPerGroup,
                        scale);
        assertClose(
                actual.toType(DataType.FLOAT32, false).toFloatArray(),
                expected.toType(DataType.FLOAT32, false).toFloatArray(),
                tolerance);
    }

    private static void verifyStridedGroupedAttention(NDManager manager) {
        int queries = 6;
        int heads = 3;
        int keyFeatures = 5;
        int sharedTokens = 7;
        int indexedTokens = 4;
        int packedWidth = 42;
        NDArray query = manager.randomNormal(new Shape(queries, keyFeatures, heads)).swapAxes(1, 2);
        NDArray shared =
                manager.randomNormal(new Shape(2, packedWidth, sharedTokens)).swapAxes(1, 2);
        NDArray sharedDeltas =
                manager.randomNormal(new Shape(queries, packedWidth, sharedTokens)).swapAxes(1, 2);
        NDArray indexedDeltas =
                manager.randomNormal(new Shape(queries, packedWidth, indexedTokens)).swapAxes(1, 2);
        int[] ids = new int[queries * indexedTokens];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = i % (sharedTokens + 1);
        }
        NDArray indexedIds = manager.create(ids, new Shape(indexedTokens, queries)).transpose();
        double scale = 0.37;

        NDArray expected =
                groupedAttentionReference(
                        query, shared, sharedDeltas, indexedDeltas, indexedIds, 3, scale);
        NDArray actual =
                NDArrays.groupedIndexedScaledDotProductAttention(
                        query, shared, sharedDeltas, indexedDeltas, indexedIds, 3, scale);
        assertClose(actual.toFloatArray(), expected.toFloatArray(), 2e-4f);
    }

    private static NDArray relationAttentionReference(
            NDArray query,
            NDArray key,
            NDArray value,
            NDArray relationKeys,
            NDArray relationBias,
            NDArray relationIds,
            double scale) {
        Shape queryShape = query.getShape();
        long batch = queryShape.get(0);
        long heads = queryShape.get(1);
        long queryTokens = queryShape.get(2);
        long keyTokens = key.getShape().get(2);
        NDArray storedIds = relationIds.toType(DataType.INT64, false);
        NDArray relationIndices =
                storedIds.getShape().dimension() == 2
                        ? storedIds
                                .reshape(1, 1, queryTokens, keyTokens)
                                .broadcast(batch, heads, queryTokens, keyTokens)
                        : storedIds
                                .reshape(batch, 1, queryTokens, keyTokens)
                                .broadcast(batch, heads, queryTokens, keyTokens);
        NDArray relationScores = query.matMul(relationKeys).gather(relationIndices, 3);
        return query.matMul(key.swapAxes(2, 3))
                .add(relationScores)
                .mul(scale)
                .add(relationBias)
                .softmax(3)
                .matMul(value);
    }

    private static NDArray groupedAttentionReference(
            NDArray query,
            NDArray sharedKeyValues,
            NDArray sharedDeltas,
            NDArray indexedDeltas,
            NDArray indexedSharedIds,
            long queriesPerGroup,
            double scale) {
        return groupedAttentionReference(
                query,
                sharedKeyValues,
                sharedDeltas,
                indexedDeltas,
                indexedSharedIds,
                queriesPerGroup,
                scale,
                false);
    }

    private static NDArray groupedAttentionReference(
            NDArray query,
            NDArray sharedKeyValues,
            NDArray sharedDeltas,
            NDArray indexedDeltas,
            NDArray indexedSharedIds,
            long queriesPerGroup,
            double scale,
            boolean bfloat16ForwardWithFloat32Gradients) {
        Shape queryShape = query.getShape();
        long queryCount = queryShape.get(0);
        long heads = queryShape.get(1);
        long keyFeatures = queryShape.get(2);
        long groupCount = sharedKeyValues.getShape().get(0);
        long sharedTokens = sharedKeyValues.getShape().get(1);
        long packedWidth = sharedKeyValues.getShape().get(2);
        long indexedTokens = indexedDeltas.getShape().get(1);
        long keyWidth = heads * keyFeatures;
        long valueFeatures = (packedWidth - keyWidth) / heads;

        NDArray querySharedKeyValues =
                sharedKeyValues
                        .reshape(groupCount, 1, sharedTokens, packedWidth)
                        .broadcast(groupCount, queriesPerGroup, sharedTokens, packedWidth)
                        .reshape(queryCount, sharedTokens, packedWidth);
        NDArray storedIds = indexedSharedIds.toType(DataType.INT64, false);
        NDArray gatherIndices =
                storedIds
                        .sub(1)
                        .maximum(0)
                        .expandDims(2)
                        .broadcast(queryCount, indexedTokens, packedWidth);
        NDArray indexedSharedKeyValues = querySharedKeyValues.gather(gatherIndices, 1);

        NDArray sharedKeys =
                querySharedKeyValues
                        .get("...,0:{}", keyWidth)
                        .reshape(queryCount, sharedTokens, heads, keyFeatures)
                        .swapAxes(1, 2);
        NDArray sharedDeltaKeys =
                sharedDeltas
                        .get("...,0:{}", keyWidth)
                        .reshape(queryCount, sharedTokens, heads, keyFeatures)
                        .swapAxes(1, 2);
        NDArray indexedKeys =
                indexedSharedKeyValues
                        .get("...,0:{}", keyWidth)
                        .reshape(queryCount, indexedTokens, heads, keyFeatures)
                        .swapAxes(1, 2);
        NDArray indexedDeltaKeys =
                indexedDeltas
                        .get("...,0:{}", keyWidth)
                        .reshape(queryCount, indexedTokens, heads, keyFeatures)
                        .swapAxes(1, 2);
        NDArray indexedPresent =
                storedIds.neq(0).expandDims(1).broadcast(queryCount, heads, indexedTokens);
        NDArray combinedIndexedKeys =
                roundReferenceForward(
                        indexedKeys.add(indexedDeltaKeys), bfloat16ForwardWithFloat32Gradients);
        NDArray participatingIndexedKeys =
                NDArrays.where(
                        indexedPresent.expandDims(3).broadcast(combinedIndexedKeys.getShape()),
                        combinedIndexedKeys,
                        combinedIndexedKeys.zerosLike());
        NDArray keys =
                roundReferenceForward(
                                sharedKeys.add(sharedDeltaKeys),
                                bfloat16ForwardWithFloat32Gradients)
                        .concat(participatingIndexedKeys, 2);
        NDArray products =
                roundReferenceForward(
                        query.expandDims(2).mul(keys), bfloat16ForwardWithFloat32Gradients);
        NDArray dots =
                roundReferenceForward(
                        products.sum(new int[] {3}), bfloat16ForwardWithFloat32Gradients);
        NDArray scores =
                roundReferenceForward(dots.mul(scale), bfloat16ForwardWithFloat32Gradients);
        NDArray finiteScores = scores;
        NDArray indexedScores = scores.get("...,{}:", sharedTokens);
        scores =
                scores.get("...,0:{}", sharedTokens)
                        .concat(
                                NDArrays.where(
                                        indexedPresent,
                                        indexedScores,
                                        indexedScores.zerosLike().add(Float.NEGATIVE_INFINITY)),
                                2);
        NDArray weights;
        if (bfloat16ForwardWithFloat32Gradients) {
            // BF16 softmax saves rounded probabilities. Its FP32 adjoint is diag(p)-p*p^T
            // with that saved p, not with the unrounded FP32 softmax result. This zero-valued
            // linearization supplies that Jacobian without invoking native mapped attention.
            NDArray probabilities =
                    scores.softmax(2)
                            .toType(DataType.BFLOAT16, false)
                            .toType(DataType.FLOAT32, false)
                            .stopGradient();
            NDArray scoreDelta = finiteScores.sub(finiteScores.stopGradient());
            NDArray centeredDelta =
                    scoreDelta.sub(probabilities.mul(scoreDelta).sum(new int[] {2}, true));
            weights = probabilities.add(probabilities.mul(centeredDelta));
        } else {
            weights = scores.softmax(2);
        }

        NDArray sharedValues =
                querySharedKeyValues
                        .get("...,{}:", keyWidth)
                        .reshape(queryCount, sharedTokens, heads, valueFeatures)
                        .swapAxes(1, 2);
        NDArray sharedDeltaValues =
                sharedDeltas
                        .get("...,{}:", keyWidth)
                        .reshape(queryCount, sharedTokens, heads, valueFeatures)
                        .swapAxes(1, 2);
        NDArray indexedValues =
                indexedSharedKeyValues
                        .get("...,{}:", keyWidth)
                        .reshape(queryCount, indexedTokens, heads, valueFeatures)
                        .swapAxes(1, 2);
        NDArray indexedDeltaValues =
                indexedDeltas
                        .get("...,{}:", keyWidth)
                        .reshape(queryCount, indexedTokens, heads, valueFeatures)
                        .swapAxes(1, 2);
        NDArray combinedIndexedValues =
                roundReferenceForward(
                        indexedValues.add(indexedDeltaValues), bfloat16ForwardWithFloat32Gradients);
        NDArray participatingIndexedValues =
                NDArrays.where(
                        indexedPresent.expandDims(3).broadcast(combinedIndexedValues.getShape()),
                        combinedIndexedValues,
                        combinedIndexedValues.zerosLike());
        NDArray values =
                roundReferenceForward(
                                sharedValues.add(sharedDeltaValues),
                                bfloat16ForwardWithFloat32Gradients)
                        .concat(participatingIndexedValues, 2);
        NDArray weightedValues =
                roundReferenceForward(
                        weights.expandDims(3).mul(values), bfloat16ForwardWithFloat32Gradients);
        return roundReferenceForward(
                weightedValues.sum(new int[] {2}), bfloat16ForwardWithFloat32Gradients);
    }

    private static NDArray roundReferenceForward(NDArray values, boolean bfloat16Forward) {
        if (!bfloat16Forward) {
            return values;
        }
        NDArray rounded =
                values.toType(DataType.BFLOAT16, false)
                        .toType(DataType.FLOAT32, false)
                        .stopGradient();
        // Round only the forward value. Casting a differentiable tensor back to BF16 would
        // also round every adjoint before the shared-table reduction being tested.
        return rounded.add(values.sub(values.stopGradient()));
    }

    private static NDArray mappedGroupedAttentionReference(
            NDArray query,
            NDArray sharedKeyValues,
            NDArray sharedGroupIndices,
            NDArray sharedDeltaTable,
            NDArray sharedDeltaIndices,
            NDArray indexedDeltas,
            NDArray indexedSharedIds,
            double scale) {
        return mappedGroupedAttentionReference(
                query,
                sharedKeyValues,
                sharedGroupIndices,
                sharedDeltaTable,
                sharedDeltaIndices,
                indexedDeltas,
                indexedSharedIds,
                scale,
                false);
    }

    private static NDArray mappedGroupedAttentionReference(
            NDArray query,
            NDArray sharedKeyValues,
            NDArray sharedGroupIndices,
            NDArray sharedDeltaTable,
            NDArray sharedDeltaIndices,
            NDArray indexedDeltas,
            NDArray indexedSharedIds,
            double scale,
            boolean bfloat16ForwardWithFloat32Gradients) {
        long queryCount = query.getShape().get(0);
        long sharedTokens = sharedKeyValues.getShape().get(1);
        long packedWidth = sharedKeyValues.getShape().get(2);
        NDArray mappedSharedKeyValues =
                NDArrays.gatherRows(
                        sharedKeyValues, sharedGroupIndices.toType(DataType.INT64, false));
        NDArray mappedSharedDeltas =
                NDArrays.gatherRows(
                                sharedDeltaTable,
                                sharedDeltaIndices.toType(DataType.INT64, false).reshape(-1))
                        .reshape(queryCount, sharedTokens, packedWidth);
        return groupedAttentionReference(
                query,
                mappedSharedKeyValues,
                mappedSharedDeltas,
                indexedDeltas,
                indexedSharedIds,
                1,
                scale,
                bfloat16ForwardWithFloat32Gradients);
    }

    private static void verifyResidualLayerNorm(NDManager manager) {
        for (int width : new int[] {3, 17, 255, 257, 513}) {
            float[] residualValues = new float[2 * width];
            float[] updateValues = new float[2 * width];
            for (int i = 0; i < residualValues.length; i++) {
                residualValues[i] = (i % 19 - 9) * 0.125f;
                updateValues[i] = (i % 7 - 3) * 0.0625f;
            }
            NDArray residual = manager.create(residualValues, new Shape(2, width));
            NDArray update = manager.create(updateValues, new Shape(2, width));
            NDArray weight = manager.ones(new Shape(width));
            NDArray bias = manager.zeros(new Shape(width));

            NDArray normalized =
                    NDArrays.addToOwnedResidualAndLayerNorm(
                            residual, update, weight, bias, 1.0e-5f);

            float[] summed = new float[residualValues.length];
            float[] expected = new float[residualValues.length];
            for (int row = 0; row < 2; row++) {
                float mean = 0f;
                for (int feature = 0; feature < width; feature++) {
                    int index = row * width + feature;
                    summed[index] = residualValues[index] + updateValues[index];
                    mean += summed[index];
                }
                mean /= width;
                float variance = 0f;
                for (int feature = 0; feature < width; feature++) {
                    float centered = summed[row * width + feature] - mean;
                    variance += centered * centered;
                }
                float inverseDeviation = (float) (1.0 / Math.sqrt(variance / width + 1.0e-5));
                for (int feature = 0; feature < width; feature++) {
                    int index = row * width + feature;
                    expected[index] = (summed[index] - mean) * inverseDeviation;
                }
            }
            assertClose(residual.toFloatArray(), summed, 1e-6f);
            assertClose(normalized.toFloatArray(), expected, 3e-4f);
        }
    }

    private static void verifyResidualLayerNormDataTypes(
            NDManager manager,
            DataType residualType,
            DataType updateType,
            DataType parameterType,
            float tolerance) {
        int rowCount = 3;
        int width = 257;
        float[] residualValues = new float[rowCount * width];
        float[] updateValues = new float[rowCount * width];
        float[] weightValues = new float[width];
        float[] biasValues = new float[width];
        for (int index = 0; index < residualValues.length; index++) {
            residualValues[index] = (index % 29 - 14) * 0.03125f;
            updateValues[index] = (index % 13 - 6) * 0.015625f;
        }
        for (int feature = 0; feature < width; feature++) {
            weightValues[feature] = 0.75f + (feature % 7) * 0.03125f;
            biasValues[feature] = (feature % 5 - 2) * 0.015625f;
        }

        NDArray residual =
                manager.create(residualValues, new Shape(rowCount, width))
                        .toType(residualType, false);
        NDArray update =
                manager.create(updateValues, new Shape(rowCount, width)).toType(updateType, false);
        NDArray weight = manager.create(weightValues).toType(parameterType, false);
        NDArray bias = manager.create(biasValues).toType(parameterType, false);
        float[] roundedResidual = residual.toType(DataType.FLOAT32, false).toFloatArray();
        float[] roundedUpdate = update.toType(DataType.FLOAT32, false).toFloatArray();
        float[] roundedWeight = weight.toType(DataType.FLOAT32, false).toFloatArray();
        float[] roundedBias = bias.toType(DataType.FLOAT32, false).toFloatArray();
        float[] summed = new float[residualValues.length];
        float[] expected =
                residualLayerNormReference(
                        roundedResidual,
                        roundedUpdate,
                        roundedWeight,
                        roundedBias,
                        rowCount,
                        width,
                        1.0e-5f,
                        summed);

        NDArray normalized =
                NDArrays.addToOwnedResidualAndLayerNorm(residual, update, weight, bias, 1.0e-5f);

        assertClose(residual.toType(DataType.FLOAT32, false).toFloatArray(), summed, tolerance);
        assertClose(normalized.toType(DataType.FLOAT32, false).toFloatArray(), expected, tolerance);
    }

    private static float[] residualLayerNormReference(
            float[] residual,
            float[] update,
            float[] weight,
            float[] bias,
            int rowCount,
            int width,
            float epsilon,
            float[] summed) {
        float[] normalized = new float[residual.length];
        for (int row = 0; row < rowCount; row++) {
            int rowOffset = row * width;
            double mean = 0.0;
            for (int feature = 0; feature < width; feature++) {
                int index = rowOffset + feature;
                summed[index] = residual[index] + update[index];
                mean += summed[index];
            }
            mean /= width;
            double variance = 0.0;
            for (int feature = 0; feature < width; feature++) {
                double centered = summed[rowOffset + feature] - mean;
                variance += centered * centered;
            }
            double inverseDeviation = 1.0 / Math.sqrt(variance / width + epsilon);
            for (int feature = 0; feature < width; feature++) {
                int index = rowOffset + feature;
                normalized[index] =
                        (float)
                                ((summed[index] - mean) * inverseDeviation * weight[feature]
                                        + bias[feature]);
            }
        }
        return normalized;
    }

    private static AttentionComparison assertMappedAttentionClose(
            float[] actual, float[] expected, DataType dataType, MappedAttentionPath path) {
        Assert.assertEquals(actual.length, expected.length);
        float absoluteTolerance;
        if (dataType == DataType.FLOAT32) {
            absoluteTolerance = 3e-4f;
        } else if (dataType == DataType.FLOAT16) {
            absoluteTolerance = 6e-3f;
        } else {
            absoluteTolerance = path == MappedAttentionPath.INT32_NATIVE ? 3e-2f : 0.015625f;
        }
        float relativeTolerance =
                dataType == DataType.BFLOAT16 && path == MappedAttentionPath.INT64_EAGER_FALLBACK
                        ? 2e-2f
                        : 0f;
        int maximumBfloat16Ulps = 0;
        float maximumAbsoluteError = 0f;
        float maximumRelativeError = 0f;
        for (int index = 0; index < actual.length; ++index) {
            Assert.assertTrue(Float.isFinite(actual[index]), "non-finite actual element " + index);
            Assert.assertTrue(
                    Float.isFinite(expected[index]), "non-finite expected element " + index);
            float absoluteError = Math.abs(actual[index] - expected[index]);
            float relativeError = absoluteError / Math.max(Math.abs(expected[index]), 1e-12f);
            float allowedError = absoluteTolerance + relativeTolerance * Math.abs(expected[index]);
            Assert.assertTrue(
                    absoluteError <= allowedError,
                    "element "
                            + index
                            + " path="
                            + path
                            + " dtype="
                            + dataType
                            + " actual="
                            + actual[index]
                            + " expected="
                            + expected[index]
                            + " absoluteError="
                            + absoluteError
                            + " allowedError="
                            + allowedError);
            if (dataType == DataType.BFLOAT16
                    && path == MappedAttentionPath.INT64_EAGER_FALLBACK
                    && Math.abs(expected[index]) >= 0.25f) {
                int ulps = bfloat16UlpDistance(actual[index], expected[index]);
                Assert.assertTrue(
                        ulps <= 8,
                        "element "
                                + index
                                + " BF16 fallback differs by "
                                + ulps
                                + " ULPs; actual="
                                + actual[index]
                                + " expected="
                                + expected[index]);
                maximumBfloat16Ulps = Math.max(maximumBfloat16Ulps, ulps);
            }
            maximumAbsoluteError = Math.max(maximumAbsoluteError, absoluteError);
            maximumRelativeError = Math.max(maximumRelativeError, relativeError);
        }
        return new AttentionComparison(
                path, maximumAbsoluteError, maximumRelativeError, maximumBfloat16Ulps);
    }

    private static int bfloat16UlpDistance(float left, float right) {
        int leftBits = Float.floatToRawIntBits(left) >>> 16;
        int rightBits = Float.floatToRawIntBits(right) >>> 16;
        return Math.abs(orderedBfloat16(leftBits) - orderedBfloat16(rightBits));
    }

    private static int orderedBfloat16(int bits) {
        return (bits & 0x8000) == 0 ? 0x8000 + bits : 0x8000 - (bits & 0x7fff);
    }

    private static void printMappedAttentionComparison(
            DataType dataType, int seed, AttentionComparison comparison) {
        System.out.printf(
                "MAPPED_GROUPED_ATTENTION_FORWARD dtype=%s path=%s seed=%d"
                        + " maxAbs=%g maxRel=%g maxBf16Ulps=%d mask=ZERO_PADDED%n",
                dataType,
                comparison.path,
                seed,
                comparison.maximumAbsoluteError,
                comparison.maximumRelativeError,
                comparison.maximumBfloat16Ulps);
    }

    private static void assertClose(float[] actual, float[] expected, float tolerance) {
        Assert.assertEquals(actual.length, expected.length);
        for (int i = 0; i < actual.length; i++) {
            Assert.assertEquals(actual[i], expected[i], tolerance, "element " + i);
        }
    }

    private static NDArray requiringGradient(NDArray array) {
        array.setRequiresGradient(true);
        return array;
    }

    private static void assertFiniteNonzeroGradient(NDArray... arrays) {
        for (NDArray array : arrays) {
            NDArray gradient = array.getGradient();
            Assert.assertNotNull(gradient);
            try (gradient;
                    NDArray values = gradient.toType(DataType.FLOAT32, false)) {
                boolean nonzero = false;
                for (float value : values.toFloatArray()) {
                    Assert.assertTrue(Float.isFinite(value));
                    nonzero |= value != 0f;
                }
                Assert.assertTrue(nonzero, "expected a nonzero gradient for " + array.getShape());
            }
        }
    }

    private static float assertGradientClose(NDArray actual, NDArray expected, float tolerance) {
        return assertGradientClose(actual, expected, tolerance, false);
    }

    private static float assertGradientClose(
            NDArray actual, NDArray expected, float tolerance, boolean roundFinalGradient) {
        NDArray actualGradient = actual.getGradient();
        NDArray expectedGradient = expected.getGradient();
        Assert.assertNotNull(actualGradient);
        Assert.assertNotNull(expectedGradient);
        try (actualGradient;
                expectedGradient;
                NDArray actualValues = actualGradient.toType(DataType.FLOAT32, false);
                NDArray roundedExpected =
                        roundFinalGradient
                                ? expectedGradient.toType(actual.getDataType(), false)
                                : expectedGradient;
                NDArray expectedValues = roundedExpected.toType(DataType.FLOAT32, false)) {
            float[] actualData = actualValues.toFloatArray();
            float[] expectedData = expectedValues.toFloatArray();
            float maximumDifference = 0f;
            for (int index = 0; index < actualData.length; index++) {
                maximumDifference =
                        Math.max(
                                maximumDifference,
                                Math.abs(actualData[index] - expectedData[index]));
            }
            assertClose(actualData, expectedData, tolerance);
            return maximumDifference;
        }
    }

    private static void assertAllZero(float[] values) {
        for (float value : values) {
            Assert.assertEquals(value, 0f);
        }
    }

    private enum MappedAttentionPath {
        INT32_NATIVE,
        INT64_EAGER_FALLBACK
    }

    private static final class AttentionComparison {

        private final MappedAttentionPath path;
        private final float maximumAbsoluteError;
        private final float maximumRelativeError;
        private final int maximumBfloat16Ulps;

        private AttentionComparison(
                MappedAttentionPath path,
                float maximumAbsoluteError,
                float maximumRelativeError,
                int maximumBfloat16Ulps) {
            this.path = path;
            this.maximumAbsoluteError = maximumAbsoluteError;
            this.maximumRelativeError = maximumRelativeError;
            this.maximumBfloat16Ulps = maximumBfloat16Ulps;
        }
    }
}
