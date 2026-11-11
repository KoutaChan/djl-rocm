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
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.core.Embedding;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.GradientCollector;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.ShortBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;

/**
 * Checks that the CPU computes packed relation attention one block of rows at a time with the
 * bits of the portable decomposition, and that its memory does not grow with the number of rows.
 */
public class PackedRelationCpuTest {

    private static final Path CLEAR_REFS = Paths.get("/proc/self/clear_refs");
    private static final Path STATUS = Paths.get("/proc/self/status");

    @DataProvider
    public Object[][] shapes() {
        // rows, queries, keys, heads, width, table rows, entries per segment, rows per block (0 for
        // the default block), data type
        return new Object[][] {
            {5, 7, 9, 2, 8, 76, 12, 1, DataType.FLOAT32},
            {5, 7, 9, 2, 8, 76, 12, 2, DataType.FLOAT32},
            {4, 23, 23, 8, 128, 112, 16, 3, DataType.FLOAT32},
            {6, 4, 31, 4, 32, 90, 15, 0, DataType.FLOAT32},
            {5, 7, 9, 2, 8, 76, 12, 2, DataType.BFLOAT16},
            {4, 23, 23, 8, 128, 112, 16, 3, DataType.BFLOAT16},
            {5, 7, 9, 2, 8, 76, 12, 3, DataType.FLOAT16}
        };
    }

    @Test(dataProvider = "shapes")
    public void blocksMatchDecompositionBitwise(
            int rows,
            int queries,
            int keys,
            int heads,
            int width,
            int tableRows,
            int entries,
            int blockRows,
            DataType dataType) {
        Inputs inputs =
                new Inputs(rows, queries, keys, heads, width, tableRows, entries, dataType, 31L);
        assertMatchesDecomposition(inputs, blockRows, dataType + " block rows " + blockRows);
    }

    @DataProvider
    public Object[][] zeroRelativeValues() {
        // rows, queries, keys, heads, width, table rows, entries per segment, rows per block (0 for
        // the default block), data type, the relative value of every table row
        return new Object[][] {
            {4, 23, 23, 8, 128, 112, 16, 3, DataType.FLOAT32, 0f},
            {4, 23, 23, 8, 128, 112, 16, 0, DataType.FLOAT32, 0f},
            {5, 7, 9, 2, 8, 76, 12, 2, DataType.BFLOAT16, 0f},
            {5, 7, 9, 2, 8, 76, 12, 1, DataType.FLOAT16, 0f},
            {4, 23, 23, 8, 128, 112, 16, 3, DataType.FLOAT32, -0f}
        };
    }

    @Test(dataProvider = "zeroRelativeValues")
    public void zeroRelativeValuesMatchDecompositionBitwise(
            int rows,
            int queries,
            int keys,
            int heads,
            int width,
            int tableRows,
            int entries,
            int blockRows,
            DataType dataType,
            float relative) {
        Inputs inputs =
                new Inputs(rows, queries, keys, heads, width, tableRows, entries, dataType, 47L);
        for (int row = 0; row < tableRows; row++) {
            for (int column = heads; column < heads + width; column++) {
                inputs.table[row * (heads + width) + column] = relative;
            }
        }
        // Values of -0 tell whether +0 relative values are added to them.
        for (int i = 2 * width - 1; i < inputs.keyValue.length; i += 2 * width) {
            if (!Float.isNaN(inputs.keyValue[i])) {
                inputs.keyValue[i] = -0f;
            }
        }
        String what = dataType + " relative " + relative + " block rows " + blockRows;
        assertMatchesDecomposition(inputs, blockRows, what);
    }

    private static void assertMatchesDecomposition(Inputs inputs, int blockRows, String what) {
        long blockTriples = blockRows == 0 ? 0 : (long) blockRows * inputs.queries * inputs.keys;
        for (Mode mode : Mode.values()) {
            Result expected = run(inputs, 0, mode, true);
            Result actual = run(inputs, blockTriples, mode, false);
            String label = what + " " + mode;
            assertBits(actual.output, expected.output, label + " output");
            if (mode != Mode.INFERENCE) {
                assertBits(actual.query, expected.query, label + " query gradient");
                assertBits(actual.keyValue, expected.keyValue, label + " key and value gradient");
                assertBits(actual.table, expected.table, label + " table gradient");
            }
        }
    }

    @Test
    public void memoryDoesNotGrowWithRows() {
        if (!Files.isWritable(CLEAR_REFS) || !Files.isReadable(STATUS)) {
            throw new SkipException("peak RSS needs " + CLEAR_REFS);
        }
        // 64 x 64 triples per row: 4 rows fill one default block, so 16 rows take 4 blocks and 128
        // rows 32 blocks.
        int small = 16;
        int large = 128;
        for (boolean training : new boolean[] {false, true}) {
            attendRandom(small, training);
            long smallPeak = peakGrowth(small, training);
            long largePeak = peakGrowth(large, training);
            // Inputs, output and gradients grow by 112 rows of 0.13 MiB. Without blocks, the
            // attention of the 112 rows alone held more than 2 GB.
            long allowed = 128L * 1024 * 1024;
            Assert.assertTrue(
                    largePeak - smallPeak < allowed,
                    "peak RSS grows from "
                            + smallPeak
                            + " B for "
                            + small
                            + " rows to "
                            + largePeak
                            + " B for "
                            + large
                            + " rows (training "
                            + training
                            + ")");
        }
    }

    @Test
    public void rejectsCodesOutsideTheirSegment() {
        Inputs inputs = new Inputs(2, 3, 4, 2, 8, 76, 12, DataType.FLOAT32, 5L);
        // The last segment holds 76 - 6 * 12 = 4 rows.
        inputs.codes[1] = (short) (inputs.codes[1] & ~(15 << 8) | (4 << 8));
        Assert.assertThrows(RuntimeException.class, () -> run(inputs, 0, Mode.INFERENCE, false));
    }

    private static long peakGrowth(int rows, boolean training) {
        try {
            Files.write(CLEAR_REFS, "5".getBytes(StandardCharsets.US_ASCII));
            long before = status("VmRSS");
            attendRandom(rows, training);
            return status("VmHWM") - before;
        } catch (IOException e) {
            throw new AssertionError("Failed to read the peak RSS", e);
        }
    }

    private static void attendRandom(int rows, boolean training) {
        int tokens = 64;
        int heads = 4;
        int width = 64;
        Engine engine = Engine.getEngine(PtEngine.ENGINE_NAME);
        try (NDManager manager = engine.newBaseManager(Device.cpu())) {
            NDArray query = manager.randomNormal(new Shape(rows, tokens, width));
            NDArray keyValue = manager.randomNormal(new Shape(rows, tokens, 2L * width));
            NDArray mask = manager.ones(new Shape(rows, tokens));
            NDArray codes =
                    manager.randomInteger(
                            Short.MIN_VALUE,
                            Short.MAX_VALUE + 1L,
                            new Shape(rows, tokens, tokens, 2),
                            DataType.INT16);
            NDArray table = manager.randomNormal(new Shape(112, heads + width));
            if (!training) {
                NDArrays.packedRelationScaledDotProductAttention(
                                query, keyValue, mask, codes, table, heads, 16, 0.25)
                        .sum()
                        .getFloat();
                return;
            }
            query.setRequiresGradient(true);
            keyValue.setRequiresGradient(true);
            table.setRequiresGradient(true);
            try (GradientCollector collector = engine.newGradientCollector()) {
                collector.backward(
                        NDArrays.packedRelationScaledDotProductAttention(
                                        query, keyValue, mask, codes, table, heads, 16, 0.25)
                                .sum());
            }
            table.getGradient().sum().getFloat();
        }
    }

    private static long status(String field) throws IOException {
        List<String> lines = Files.readAllLines(STATUS);
        for (String line : lines) {
            if (line.startsWith(field + ':')) {
                return Long.parseLong(line.replaceAll("[^0-9]", "")) * 1024;
            }
        }
        throw new IllegalStateException("No " + field + " in " + STATUS);
    }

    private static Result run(Inputs in, long blockTriples, Mode mode, boolean decomposition) {
        Engine engine = Engine.getEngine(PtEngine.ENGINE_NAME);
        try (NDManager manager = engine.newBaseManager(Device.cpu())) {
            NDArray query =
                    manager.create(in.query, new Shape(in.rows, in.queries, in.width))
                            .toType(in.dataType, false);
            NDArray keyValue =
                    manager.create(in.keyValue, new Shape(in.rows, in.keys, 2L * in.width))
                            .toType(in.dataType, false);
            NDArray mask = manager.create(in.mask, new Shape(in.rows, in.keys));
            NDArray codes =
                    manager.create(
                            ShortBuffer.wrap(in.codes),
                            new Shape(in.rows, in.queries, in.keys, in.words()),
                            DataType.INT16);
            NDArray table =
                    manager.create(in.table, new Shape(in.tableRows, in.heads + in.width))
                            .toType(in.dataType, false);
            if (mode == Mode.INFERENCE) {
                NDArray output =
                        attend(in, query, keyValue, mask, codes, table, blockTriples, decomposition);
                return new Result(floats(output), null, null, null);
            }
            query.setRequiresGradient(true);
            keyValue.setRequiresGradient(true);
            table.setRequiresGradient(true);
            NDArray output;
            try (GradientCollector collector = engine.newGradientCollector()) {
                output = attend(in, query, keyValue, mask, codes, table, blockTriples, decomposition);
                NDArray target =
                        mode == Mode.SUM
                                ? output.sum()
                                : output.mul(
                                                manager.create(
                                                        in.weights,
                                                        new Shape(in.rows, in.queries, in.width)))
                                        .sum();
                if (mode == Mode.HIGHER_ORDER) {
                    JniUtils.backward(
                            (PtNDArray) target, (PtNDArray) target.onesLike(), false, true);
                } else {
                    collector.backward(target);
                }
            }
            return new Result(
                    floats(output),
                    floats(query.getGradient()),
                    floats(keyValue.getGradient()),
                    floats(table.getGradient()));
        }
    }

    private static NDArray attend(
            Inputs in,
            NDArray query,
            NDArray keyValue,
            NDArray mask,
            NDArray codes,
            NDArray table,
            long blockTriples,
            boolean decomposition) {
        if (decomposition) {
            return decomposition(
                    query, keyValue, mask, codes, table, in.heads, in.entries, in.scale);
        }
        if (blockTriples == 0) {
            return NDArrays.packedRelationScaledDotProductAttention(
                    query, keyValue, mask, codes, table, in.heads, in.entries, in.scale);
        }
        return JniUtils.packedRelationScaledDotProductAttention(
                (PtNDArray) query,
                (PtNDArray) keyValue,
                (PtNDArray) mask,
                (PtNDArray) codes,
                (PtNDArray) table,
                in.heads,
                in.entries,
                (float) in.scale,
                blockTriples);
    }

    /** The portable decomposition that the CPU used before it computed blocks of rows. */
    private static NDArray decomposition(
            NDArray query,
            NDArray packedKeyValue,
            NDArray mask,
            NDArray packedCodes,
            NDArray relationTable,
            long heads,
            long entriesPerSegment,
            double scale) {
        NDManager outputManager = query.getManager();
        try (NDManager scope = outputManager.newSubManager()) {
            scope.tempAttachAll(query, packedKeyValue, mask, packedCodes, relationTable);
            NDArray floatQuery = query.getNDArrayInternal().differentiableCast(DataType.FLOAT32);
            NDArray floatMemory =
                    packedKeyValue.getNDArrayInternal().differentiableCast(DataType.FLOAT32);
            NDArray floatTable =
                    relationTable.getNDArrayInternal().differentiableCast(DataType.FLOAT32);
            long batch = query.getShape().get(0);
            long queries = query.getShape().get(1);
            long width = query.getShape().get(2);
            long keys = packedKeyValue.getShape().get(1);
            long features = width / heads;
            int entries = Math.toIntExact(entriesPerSegment);
            int rows = Math.toIntExact(relationTable.getShape().get(0));
            int segments = (rows + entries - 1) / entries;
            int words = (segments + 3) / 4;
            short[] packed = packedCodes.toShortArray();
            int pairs = Math.toIntExact(batch * queries * keys);
            NDArray relation = null;
            for (int segment = 0; segment < segments; segment++) {
                int[] indices = new int[pairs];
                for (int pair = 0; pair < pairs; pair++) {
                    int code =
                            (Short.toUnsignedInt(packed[pair * words + segment / 4])
                                            >>> ((segment % 4) * 4))
                                    & 15;
                    int index = segment * entries + code;
                    if (code >= entries || index >= rows) {
                        throw new IllegalArgumentException(
                                "packed relation code exceeds its segment");
                    }
                    indices[pair] = index;
                }
                NDArray gathered =
                        Embedding.embedding(
                                        scope.create(indices, new Shape(batch, queries, keys)),
                                        floatTable,
                                        SparseFormat.DENSE)
                                .singletonOrThrow();
                relation = relation == null ? gathered : relation.add(gathered);
            }
            NDArray valid = mask.neq(0).reshape(batch, 1, 1, keys).stopGradient();
            NDArray validMemory = mask.neq(0).expandDims(2).broadcast(packedKeyValue.getShape());
            NDArray safeMemory = NDArrays.where(validMemory, floatMemory, floatMemory.zerosLike());
            NDArray q = floatQuery.reshape(batch, queries, heads, features).swapAxes(1, 2);
            NDArray k =
                    safeMemory
                            .get("...,0:{}", width)
                            .reshape(batch, keys, heads, features)
                            .swapAxes(1, 2);
            NDArray v =
                    safeMemory
                            .get("...,{}:", width)
                            .reshape(batch, keys, heads, features)
                            .swapAxes(1, 2);
            NDArray bias = relation.get("...,0:{}", heads).transpose(0, 3, 1, 2);
            NDArray scores = q.matMul(k.swapAxes(2, 3)).mul(scale).add(bias);
            NDArray fullValid = valid.broadcast(scores.getShape());
            NDArray present = valid.sum(new int[] {3}, true).gt(0).broadcast(scores.getShape());
            NDArray safeScores = NDArrays.where(fullValid, scores, scores.zerosLike());
            NDArray attentionValid = NDArrays.where(present, fullValid, fullValid.onesLike());
            NDArray probabilities =
                    NDArrays.where(
                            fullValid,
                            NDArrays.where(
                                            attentionValid,
                                            safeScores,
                                            scores.zerosLike().add(Float.NEGATIVE_INFINITY))
                                    .softmax(3),
                            scores.zerosLike());
            NDArray relativeValues =
                    relation.get("...,{}:", heads)
                            .reshape(batch, queries, keys, heads, features)
                            .transpose(0, 3, 1, 2, 4);
            NDArray values = v.expandDims(2).add(relativeValues);
            values =
                    NDArrays.where(
                            fullValid.expandDims(4).broadcast(values.getShape()),
                            values,
                            values.zerosLike());
            NDArray result =
                    probabilities
                            .expandDims(4)
                            .mul(values)
                            .sum(new int[] {3})
                            .swapAxes(1, 2)
                            .reshape(batch, queries, width)
                            .getNDArrayInternal()
                            .differentiableCast(query.getDataType());
            outputManager.attachAll(result);
            return result;
        }
    }

    private static float[] floats(NDArray array) {
        return array.toType(DataType.FLOAT32, false).toFloatArray();
    }

    private static void assertBits(float[] actual, float[] expected, String what) {
        Assert.assertEquals(actual.length, expected.length, what + " length");
        for (int i = 0; i < actual.length; i++) {
            if (Float.floatToRawIntBits(actual[i]) != Float.floatToRawIntBits(expected[i])) {
                Assert.fail(what + " differs at " + i + ": " + actual[i] + " != " + expected[i]);
            }
        }
    }

    /** How the attention is run and differentiated. */
    private enum Mode {
        INFERENCE,
        WEIGHTED,
        SUM,
        HIGHER_ORDER
    }

    private static final class Inputs {
        final int rows;
        final int queries;
        final int keys;
        final int heads;
        final int width;
        final int tableRows;
        final int entries;
        final DataType dataType;
        final double scale;
        final float[] query;
        final float[] keyValue;
        final float[] mask;
        final short[] codes;
        final float[] table;
        final float[] weights;

        /**
         * Random inputs. Row r keeps all but its last r keys and the last row keeps none; padded
         * memory is NaN, so the masks must apply before any arithmetic.
         */
        Inputs(
                int rows,
                int queries,
                int keys,
                int heads,
                int width,
                int tableRows,
                int entries,
                DataType dataType,
                long seed) {
            this.rows = rows;
            this.queries = queries;
            this.keys = keys;
            this.heads = heads;
            this.width = width;
            this.tableRows = tableRows;
            this.entries = entries;
            this.dataType = dataType;
            scale = 0.7 / Math.sqrt((double) width / heads);
            Random random = new Random(seed);
            query = gaussian(random, rows * queries * width);
            keyValue = gaussian(random, rows * keys * 2 * width);
            mask = new float[rows * keys];
            for (int row = 0; row < rows; row++) {
                int present = row == rows - 1 ? 0 : keys - row;
                for (int key = 0; key < keys; key++) {
                    if (key < present) {
                        mask[row * keys + key] = 1f;
                    } else {
                        for (int i = 0; i < 2 * width; i++) {
                            keyValue[(row * keys + key) * 2 * width + i] = Float.NaN;
                        }
                    }
                }
            }
            codes = codes(random, rows * queries * keys, tableRows, entries);
            table = gaussian(random, tableRows * (heads + width));
            weights = gaussian(random, rows * queries * width);
        }

        int words() {
            return words(tableRows, entries);
        }

        private static int words(int tableRows, int entries) {
            return ((tableRows + entries - 1) / entries + 3) / 4;
        }

        private static short[] codes(Random random, int pairs, int tableRows, int entries) {
            int segments = (tableRows + entries - 1) / entries;
            int words = words(tableRows, entries);
            short[] codes = new short[pairs * words];
            for (int pair = 0; pair < pairs; pair++) {
                for (int segment = 0; segment < segments; segment++) {
                    int size = Math.min(entries, tableRows - segment * entries);
                    int code = random.nextInt(size);
                    codes[pair * words + segment / 4] |= (short) (code << ((segment % 4) * 4));
                }
            }
            return codes;
        }

        private static float[] gaussian(Random random, int count) {
            float[] values = new float[count];
            for (int i = 0; i < count; i++) {
                values[i] = (float) random.nextGaussian();
            }
            return values;
        }
    }

    private static final class Result {
        final float[] output;
        final float[] query;
        final float[] keyValue;
        final float[] table;

        Result(float[] output, float[] query, float[] keyValue, float[] table) {
            this.output = output;
            this.query = query;
            this.keyValue = keyValue;
            this.table = table;
        }
    }
}
