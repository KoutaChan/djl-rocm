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
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.EmbeddingReduction;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.core.Embedding;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** Verifies owned masked embedding residual semantics through the PyTorch JNI boundary. */
public class PtOwnedMaskedEmbeddingResidualTest {

    private static final int BATCH = 2;
    private static final int TOKENS = 3;
    private static final int WIDTH = 3;

    @DataProvider
    public Object[][] indexDataTypes() {
        return new Object[][] {
            {DataType.INT16}, {DataType.INT32}, {DataType.INT64},
        };
    }

    @Test(dataProvider = "indexDataTypes")
    public void stridedIndicesPreserveSumAndOwnedMutation(DataType indexDataType) {
        verify(DataType.FLOAT32, indexDataType, EmbeddingReduction.SUM, 1);
    }

    @Test(dataProvider = "indexDataTypes")
    public void stridedIndicesPreserveMeanValidAndPadding(DataType indexDataType) {
        verify(DataType.FLOAT32, indexDataType, EmbeddingReduction.MEAN_VALID, 2);
    }

    @Test
    public void nativeKernelMatchesPortableLowPrecisionBoundaries() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("GPU is unavailable");
        }
        for (DataType tokenDataType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            for (DataType indexDataType :
                    new DataType[] {DataType.INT16, DataType.INT32, DataType.INT64}) {
                verify(tokenDataType, indexDataType, EmbeddingReduction.SUM, 1);
                verify(tokenDataType, indexDataType, EmbeddingReduction.MEAN_VALID, 2);
            }
        }
    }

    @DataProvider
    public Object[][] lowPrecisionReductions() {
        return new Object[][] {
            {DataType.FLOAT16, EmbeddingReduction.SUM},
            {DataType.FLOAT16, EmbeddingReduction.MEAN_VALID},
            {DataType.BFLOAT16, EmbeddingReduction.SUM},
            {DataType.BFLOAT16, EmbeddingReduction.MEAN_VALID},
        };
    }

    @Test(dataProvider = "lowPrecisionReductions")
    public void lowPrecisionResidualPreservesIntermediateRounding(
            DataType dataType, EmbeddingReduction reduction) {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("GPU is unavailable");
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu());
                Autocast ignored = engine.newAutocast(Device.gpu(), dataType, true)) {
            float unit = dataType == DataType.FLOAT16 ? 0x1p-10f : 0x1p-7f;
            NDArray table =
                    manager.create(
                                    new float[] {
                                        9f, -9f, 9f, 1f, -1f, 1f + unit, unit / 2, -unit / 2, unit,
                                        0.75f, 0.125f, -0.125f, -0.75f, 1f + unit, 0.5f
                                    },
                                    new Shape(5, WIDTH))
                            .toType(dataType, false);
            NDArray tokens =
                    manager.create(
                                    new float[] {
                                        unit / 4, -unit / 4, unit / 4, 0.125f, 0.25f, 0.375f, 2f,
                                        2f, 2f, -0.5f, -0.125f, 0.25f, 0.25f, -0.25f, -0.125f, 0.5f,
                                        -0.5f, unit
                                    },
                                    new Shape(BATCH, TOKENS, WIDTH))
                            .toType(dataType, false);
            NDArray raw =
                    manager.create(
                            new int[] {
                                1, 2, 1,
                                0, 2, 1,
                                2, 3, 0,
                                3, 0, 1,
                                4, 1, 1,
                                0, 0, 1
                            },
                            new Shape(BATCH, TOKENS, 3));
            NDArray first = raw.get("...,0");
            NDArray second = raw.get("...,1");
            NDArray validMask = raw.get("...,2");
            NDArray firstPresent = first.neq(0).toType(dataType, false);
            NDArray secondPresent = second.neq(0).toType(dataType, false);
            NDArray identity =
                    Embedding.embedding(first, table, SparseFormat.DENSE)
                            .singletonOrThrow()
                            .mul(firstPresent.expandDims(2))
                            .add(
                                    Embedding.embedding(second, table, SparseFormat.DENSE)
                                            .singletonOrThrow()
                                            .mul(secondPresent.expandDims(2)));
            if (reduction == EmbeddingReduction.MEAN_VALID) {
                identity = identity.div(firstPresent.add(secondPresent).maximum(1).expandDims(2));
            }
            float[] expected =
                    tokens.add(identity)
                            .mul(validMask.toType(dataType, false).expandDims(2))
                            .toType(DataType.FLOAT32, false)
                            .toFloatArray();
            // Rounding only once after the residual changes these halfway cases by one ULP.
            Assert.assertEquals(expected[0], reduction == EmbeddingReduction.SUM ? 1f : 0.5f);
            NDArray mask =
                    NDArrays.addMaskedEmbeddingResidualToOwnedTokens(
                            tokens, new NDList(first, second), table, validMask, 0, reduction);
            Assert.assertEquals(tokens.toType(DataType.FLOAT32, false).toFloatArray(), expected);
            Assert.assertEquals(
                    mask.toType(DataType.FLOAT32, false).toFloatArray(),
                    new float[] {1, 1, 0, 1, 1, 1});
        }
    }

    private static void verify(
            DataType tokenDataType,
            DataType indexDataType,
            EmbeddingReduction reduction,
            int indexCount) {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("GPU is unavailable");
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            NDArray table =
                    manager.create(
                                    new float[] {
                                        9f, 9f, 9f,
                                        1f, 2f, 3f,
                                        4f, 5f, 6f,
                                        7f, 8f, 9f,
                                        10f, 11f, 12f
                                    },
                                    new Shape(5, WIDTH))
                            .toType(tokenDataType, false);
            NDArray tokens =
                    manager.create(
                                    new float[] {
                                        0f, 1f, 2f,
                                        3f, 4f, 5f,
                                        6f, 7f, 8f,
                                        9f, 10f, 11f,
                                        12f, 13f, 14f,
                                        15f, 16f, 17f
                                    },
                                    new Shape(BATCH, TOKENS, WIDTH))
                            .toType(tokenDataType, false);
            NDArray raw =
                    manager.create(
                                    new int[] {
                                        1, 0, 1,
                                        0, 2, 1,
                                        2, 3, 0,
                                        3, 0, 1,
                                        4, 1, 1,
                                        0, 0, 0
                                    },
                                    new Shape(BATCH, TOKENS, 3))
                            .toType(indexDataType, false);
            NDArray firstIndices = raw.get("...,0");
            NDArray secondIndices = raw.get("...,1");
            NDArray validMask = raw.get("...,2");
            NDList indices =
                    indexCount == 1
                            ? new NDList(firstIndices)
                            : new NDList(firstIndices, secondIndices);
            float[] expected =
                    expected(
                            tokens.toType(DataType.FLOAT32, false).toFloatArray(),
                            table.toType(DataType.FLOAT32, false).toFloatArray(),
                            raw.toType(DataType.INT64, false).toLongArray(),
                            indexCount);

            NDArray convertedMask =
                    NDArrays.addMaskedEmbeddingResidualToOwnedTokens(
                            tokens, indices, table, validMask, 0, reduction);

            Assert.assertSame(convertedMask.getManager(), tokens.getManager());
            Assert.assertEquals(convertedMask.getDataType(), tokens.getDataType());
            Assert.assertEquals(convertedMask.getShape(), new Shape(BATCH, TOKENS));
            Assert.assertEquals(tokens.toType(DataType.FLOAT32, false).toFloatArray(), expected);
            Assert.assertEquals(
                    convertedMask.toType(DataType.FLOAT32, false).toFloatArray(),
                    new float[] {1, 1, 0, 1, 1, 0});
        }
    }

    private static float[] expected(float[] base, float[] table, long[] raw, int indexCount) {
        float[] expected = base.clone();
        for (int row = 0; row < BATCH * TOKENS; ++row) {
            long valid = raw[row * 3 + 2];
            int validIndices = 0;
            for (int source = 0; source < indexCount; ++source) {
                if (raw[row * 3 + source] != 0) {
                    ++validIndices;
                }
            }
            for (int column = 0; column < WIDTH; ++column) {
                float identity = 0f;
                for (int source = 0; source < indexCount; ++source) {
                    int tableRow = Math.toIntExact(raw[row * 3 + source]);
                    if (tableRow != 0) {
                        identity += table[tableRow * WIDTH + column];
                    }
                }
                if (indexCount == 2 && validIndices > 0) {
                    identity /= validIndices;
                }
                expected[row * WIDTH + column] =
                        (expected[row * WIDTH + column] + identity) * valid;
            }
        }
        return expected;
    }
}
