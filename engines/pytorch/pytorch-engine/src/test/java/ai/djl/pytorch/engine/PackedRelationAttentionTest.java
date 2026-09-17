/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.djl.pytorch.engine;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.nio.ShortBuffer;

/** Checks packed relation decoding, relative values, masking, and all trainable gradients. */
public class PackedRelationAttentionTest {
    private static final int B = 2;
    private static final int Q = 2;
    private static final int K = 3;
    private static final int A = 4;
    private static final int H = 2;
    private static final float SCALE = 0.7f;

    @Test
    public void cpuReferenceAndGradients() {
        float[][] data = data();
        Result result = run(Device.cpu(), data, true);
        assertClose(result.output, reference(data), 2e-6f);
        int[][] probes = {{0, 3, 5}, {0, 5, 11, 17}, {0, 3, 66, 67, 72 * 6 + 5}};
        for (int tensor = 0; tensor < probes.length; tensor++) {
            for (int index : probes[tensor]) {
                float original = data[tensor][index];
                data[tensor][index] = original + 0.001f;
                float plus = sum(run(Device.cpu(), data, false).output);
                data[tensor][index] = original - 0.001f;
                float minus = sum(run(Device.cpu(), data, false).output);
                data[tensor][index] = original;
                Assert.assertEquals(
                        result.gradients[tensor][index],
                        (plus - minus) / 0.002f,
                        8e-4f,
                        "gradient tensor=" + tensor + " index=" + index);
            }
        }
        for (int i = Q * A; i < B * Q * A; i++) {
            Assert.assertEquals(result.output[i], 0f);
            Assert.assertEquals(result.gradients[0][i], 0f);
        }
        for (int i = K * 2 * A; i < B * K * 2 * A; i++) {
            Assert.assertEquals(result.gradients[1][i], 0f);
        }
    }

    @Test
    public void mixedPrecisionTablePreservesGradients() {
        for (DataType dtype : new DataType[] {DataType.FLOAT16, DataType.BFLOAT16}) {
            Result result = run(Device.cpu(), data(), true, dtype);
            assertClose(result.output, reference(data()), 0.008f);
            for (float[] gradient : result.gradients) {
                float magnitude = 0;
                for (float value : gradient) {
                    Assert.assertTrue(Float.isFinite(value));
                    magnitude += Math.abs(value);
                }
                Assert.assertTrue(magnitude > 0, "mixed-precision gradient was disconnected");
            }
        }
    }

    @Test
    public void unmaskedNonFiniteScoresRemainObservable() {
        float[][] data = data();
        data[0][0] = Float.NaN;
        Result cpu = run(Device.cpu(), data, false);
        Assert.assertTrue(Float.isNaN(cpu.output[0]));
        Assert.assertEquals(cpu.output[Q * A], 0f);
        if (Engine.getInstance().getGpuCount() > 0) {
            Result gpu = run(Device.gpu(), data, false);
            Assert.assertTrue(Float.isNaN(gpu.output[0]));
            Assert.assertEquals(gpu.output[Q * A], 0f);
        }
    }

    @Test
    public void rocmMatchesCpuWhenAvailable() {
        if (Engine.getInstance().getGpuCount() == 0) {
            return;
        }
        float[][] data = data();
        Result cpu = run(Device.cpu(), data, true);
        Result gpu = run(Device.gpu(), data, true);
        assertClose(gpu.output, cpu.output, 2e-5f);
        for (int i = 0; i < 3; i++) {
            assertClose(gpu.gradients[i], cpu.gradients[i], 5e-5f);
        }
    }

    private static float[][] data() {
        float[][] data = {new float[B * Q * A], new float[B * K * 2 * A], new float[76 * (H + A)]};
        for (int t = 0; t < data.length; t++) {
            for (int i = 0; i < data[t].length; i++) {
                data[t][i] = (float) Math.sin(0.37 * i + t) * 0.2f;
            }
        }
        // Padding may carry non-finite memory; it must be excluded before arithmetic.
        for (int i = K * 2 * A; i < B * K * 2 * A; i++) {
            data[1][i] = Float.NaN;
        }
        return data;
    }

    private static short[] codes() {
        short[] codes = new short[B * Q * K * 2];
        for (int pair = 0; pair < B * Q * K; pair++) {
            for (int segment = 0; segment < 7; segment++) {
                int code = segment == 6 ? pair % 4 : (pair * 3 + segment + 8) % 12;
                codes[pair * 2 + segment / 4] |= (short) (code << ((segment % 4) * 4));
            }
        }
        return codes;
    }

    private static Result run(Device device, float[][] data, boolean gradients) {
        return run(device, data, gradients, DataType.FLOAT32);
    }

    private static Result run(Device device, float[][] data, boolean gradients, DataType dtype) {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray q = manager.create(data[0], new Shape(B, Q, A)).toType(dtype, false);
            NDArray kv = manager.create(data[1], new Shape(B, K, 2 * A)).toType(dtype, false);
            NDArray table = manager.create(data[2], new Shape(76, H + A)).toType(dtype, false);
            NDArray mask = manager.create(new int[] {1, 0, 1, 0, 0, 0}, new Shape(B, K));
            NDArray packed =
                    manager.create(
                            ShortBuffer.wrap(codes()), new Shape(B, Q, K, 2), DataType.INT16);
            if (!gradients) {
                return new Result(
                        NDArrays.packedRelationScaledDotProductAttention(
                                        q, kv, mask, packed, table, H, 12, SCALE)
                                .toType(DataType.FLOAT32, false)
                                .toFloatArray(),
                        null);
            }
            q.setRequiresGradient(true);
            kv.setRequiresGradient(true);
            table.setRequiresGradient(true);
            NDArray output;
            try (GradientCollector collector = engine.newGradientCollector()) {
                output =
                        NDArrays.packedRelationScaledDotProductAttention(
                                q, kv, mask, packed, table, H, 12, SCALE);
                collector.backward(output.sum());
            }
            return new Result(
                    output.toType(DataType.FLOAT32, false).toFloatArray(),
                    new float[][] {
                        q.getGradient().toType(DataType.FLOAT32, false).toFloatArray(),
                        kv.getGradient().toType(DataType.FLOAT32, false).toFloatArray(),
                        table.getGradient().toType(DataType.FLOAT32, false).toFloatArray()
                    });
        }
    }

    private static float[] reference(float[][] data) {
        short[] codes = codes();
        float[] out = new float[B * Q * A];
        for (int query = 0; query < Q; query++) {
            for (int head = 0; head < H; head++) {
                double[] scores = new double[K];
                double[][] values = new double[K][A / H];
                double denominator = 0;
                for (int key = 0; key < K; key += 2) {
                    for (int d = 0; d < A / H; d++) {
                        scores[key] +=
                                data[0][query * A + head * (A / H) + d]
                                        * data[1][key * 2 * A + head * (A / H) + d]
                                        * SCALE;
                        values[key][d] = data[1][key * 2 * A + A + head * (A / H) + d];
                    }
                    for (int segment = 0; segment < 7; segment++) {
                        int pair = query * K + key;
                        int code =
                                (Short.toUnsignedInt(codes[pair * 2 + segment / 4])
                                                >>> ((segment % 4) * 4))
                                        & 15;
                        int row = segment * 12 + code;
                        scores[key] += data[2][row * (H + A) + head];
                        for (int d = 0; d < A / H; d++) {
                            values[key][d] += data[2][row * (H + A) + H + head * (A / H) + d];
                        }
                    }
                    scores[key] = Math.exp(scores[key]);
                    denominator += scores[key];
                }
                for (int d = 0; d < A / H; d++) {
                    out[query * A + head * (A / H) + d] =
                            (float)
                                    ((scores[0] * values[0][d] + scores[2] * values[2][d])
                                            / denominator);
                }
            }
        }
        return out;
    }

    private static float sum(float[] values) {
        float sum = 0;
        for (float value : values) {
            sum += value;
        }
        return sum;
    }

    private static void assertClose(float[] actual, float[] expected, float tolerance) {
        Assert.assertEquals(actual.length, expected.length);
        for (int i = 0; i < actual.length; i++) {
            Assert.assertTrue(Float.isFinite(actual[i]), "non-finite value at " + i);
            Assert.assertEquals(actual[i], expected[i], tolerance, "element " + i);
        }
    }

    private static final class Result {
        final float[] output;
        final float[][] gradients;

        Result(float[] output, float[][] gradients) {
            this.output = output;
            this.gradients = gradients;
        }
    }
}
