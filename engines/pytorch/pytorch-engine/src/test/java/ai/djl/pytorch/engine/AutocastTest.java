/*
 * Copyright 2025 KoutaChan. Licensed under the Apache License, Version 2.0.
 */
package ai.djl.pytorch.engine;

import ai.djl.Device;
import ai.djl.engine.Autocast;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.jni.JniUtils;

import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.Test;

/**
 * Exercises the {@link Autocast} scope guard (libtorch's {@code at::autocast}
 * backend) end-to-end: matmul / SDPA / rmsNorm results should come out in the
 * autocast dtype, nested scopes should compose, and the previous state must be
 * restored on {@code close()}.
 */
@SuppressWarnings("try") // Autocast resource var is used for side effects via close()
public class AutocastTest {

    private static final int CPU_DEVICE = 0;
    private static final int GPU_DEVICE = 1;

    @AfterMethod
    public void resetAutocastState() {
        // Ensure any test that throws mid-scope does not bleed state into
        // subsequent tests.
        if (JniUtils.autocastIsEnabled(CPU_DEVICE)) {
            JniUtils.autocastSetEnabled(CPU_DEVICE, false);
        }
        if (JniUtils.autocastIsEnabled(GPU_DEVICE)) {
            JniUtils.autocastSetEnabled(GPU_DEVICE, false);
        }
        JniUtils.autocastClearCache();
    }

    @Test
    public void saveAndRestoreState() {
        Engine engine = Engine.getInstance();
        Assert.assertTrue(engine.supportsAutocast(), "PyTorch engine must advertise autocast");

        int before = JniUtils.autocastGetDtype(CPU_DEVICE);
        boolean beforeEnabled = JniUtils.autocastIsEnabled(CPU_DEVICE);
        try (Autocast ac = engine.newAutocast(Device.cpu(), DataType.BFLOAT16)) {
            Assert.assertTrue(JniUtils.autocastIsEnabled(CPU_DEVICE));
            Assert.assertEquals(JniUtils.autocastGetDtype(CPU_DEVICE), DataType.BFLOAT16.ordinal());
        }
        Assert.assertEquals(JniUtils.autocastIsEnabled(CPU_DEVICE), beforeEnabled);
        Assert.assertEquals(JniUtils.autocastGetDtype(CPU_DEVICE), before);
    }

    @Test
    public void nestedScopesRestoreOuterState() {
        Engine engine = Engine.getInstance();
        try (Autocast outer = engine.newAutocast(Device.cpu(), DataType.BFLOAT16)) {
            Assert.assertTrue(JniUtils.autocastIsEnabled(CPU_DEVICE));
            Assert.assertEquals(JniUtils.autocastGetDtype(CPU_DEVICE), DataType.BFLOAT16.ordinal());

            try (Autocast inner = engine.newAutocast(Device.cpu(), DataType.FLOAT16)) {
                Assert.assertEquals(
                        JniUtils.autocastGetDtype(CPU_DEVICE), DataType.FLOAT16.ordinal());
            }
            // After the inner scope exits, the outer scope's BF16 state must
            // be visible again.
            Assert.assertTrue(JniUtils.autocastIsEnabled(CPU_DEVICE));
            Assert.assertEquals(JniUtils.autocastGetDtype(CPU_DEVICE), DataType.BFLOAT16.ordinal());
        }
        Assert.assertFalse(JniUtils.autocastIsEnabled(CPU_DEVICE));
    }

    @Test
    public void cacheEnabledFlagPropagates() {
        Engine engine = Engine.getInstance();
        boolean before = JniUtils.autocastIsCacheEnabled();
        try (Autocast ac = engine.newAutocast(Device.cpu(), DataType.BFLOAT16, false)) {
            Assert.assertFalse(JniUtils.autocastIsCacheEnabled());
        }
        Assert.assertEquals(JniUtils.autocastIsCacheEnabled(), before);
    }

    @Test
    public void sdpaOutputsAutocastDtype() {
        // SDPA is in PyTorch's autocast allowlist, so FP32 inputs inside a
        // BF16 autocast scope must produce a BF16 output on CUDA/ROCm.
        runOnGpuIfAvailable(
                (engine, manager, device) -> {
                    NDArray q = manager.randomNormal(new Shape(1, 4, 8, 16)).toDevice(device, false);
                    NDArray k = manager.randomNormal(new Shape(1, 4, 8, 16)).toDevice(device, false);
                    NDArray v = manager.randomNormal(new Shape(1, 4, 8, 16)).toDevice(device, false);
                    Assert.assertEquals(q.getDataType(), DataType.FLOAT32);

                    try (Autocast ac = engine.newAutocast(device, DataType.BFLOAT16)) {
                        NDArray out =
                                q.getNDArrayInternal()
                                        .scaledDotProductAttention(k, v, null, 0.0, false);
                        Assert.assertEquals(
                                out.getDataType(),
                                DataType.BFLOAT16,
                                "SDPA must emit the autocast dtype");
                        Assert.assertTrue(isAllFinite(out), "SDPA output must be finite");
                    }
                });
    }

    @Test
    public void rmsNormOutputsAutocastDtype() {
        // Confirms our custom at::rms_norm JNI routes through the dispatcher,
        // so autocast can see and cast it. Regression test for fork-specific
        // bindings accidentally bypassing dispatch.
        runOnGpuIfAvailable(
                (engine, manager, device) -> {
                    NDArray x = manager.randomNormal(new Shape(2, 4, 32)).toDevice(device, false);
                    NDArray weight = manager.ones(new Shape(32)).toDevice(device, false);
                    Assert.assertEquals(x.getDataType(), DataType.FLOAT32);

                    try (Autocast ac = engine.newAutocast(device, DataType.BFLOAT16)) {
                        NDArray out =
                                x.getNDArrayInternal().rmsNorm(new long[] {32}, weight, 1e-5f);
                        Assert.assertEquals(
                                out.getDataType(),
                                DataType.BFLOAT16,
                                "rms_norm must emit the autocast dtype");
                        Assert.assertTrue(isAllFinite(out), "rms_norm output must be finite");
                    }
                });
    }

    @Test
    public void matmulOutputsAutocastDtype() {
        // Baseline sanity check: torch.matmul is the canonical autocast op.
        // If this fails the backend is fundamentally not wired up.
        runOnGpuIfAvailable(
                (engine, manager, device) -> {
                    NDArray a = manager.randomNormal(new Shape(64, 64)).toDevice(device, false);
                    NDArray b = manager.randomNormal(new Shape(64, 64)).toDevice(device, false);

                    try (Autocast ac = engine.newAutocast(device, DataType.BFLOAT16)) {
                        NDArray out = a.matMul(b);
                        Assert.assertEquals(out.getDataType(), DataType.BFLOAT16);
                    }
                });
    }

    @Test
    public void matmulWithinTolerance() {
        // FP32 matmul vs autocast BF16 matmul on random inputs should match
        // within BF16's ~1e-2 relative error. Guards against the autocast
        // cast silently aliasing to something else.
        runOnGpuIfAvailable(
                (engine, manager, device) -> {
                    NDArray a = manager.randomNormal(new Shape(64, 64)).toDevice(device, false);
                    NDArray b = manager.randomNormal(new Shape(64, 64)).toDevice(device, false);

                    NDArray fpOut = a.matMul(b);
                    NDArray bfOut;
                    try (Autocast ac = engine.newAutocast(device, DataType.BFLOAT16)) {
                        bfOut = a.matMul(b).toType(DataType.FLOAT32, false);
                    }
                    NDArray diff = fpOut.sub(bfOut).abs();
                    NDArray ref = fpOut.abs().add(1e-6f);
                    float maxRelError = diff.div(ref).max().getFloat();
                    Assert.assertTrue(
                            maxRelError < 5e-2f,
                            "BF16 matmul relative error " + maxRelError + " exceeds tolerance");
                });
    }

    private static boolean isAllFinite(NDArray a) {
        // Fall back to Java-side finite check: isNaN / isInfinite exist across
        // DJL versions. A single aggregated boolean keeps the JNI chatter low.
        float[] flat = a.toType(DataType.FLOAT32, false).toFloatArray();
        for (float value : flat) {
            if (Float.isNaN(value) || Float.isInfinite(value)) {
                return false;
            }
        }
        return true;
    }

    @FunctionalInterface
    private interface GpuTest {
        void run(Engine engine, NDManager manager, Device device);
    }

    private static void runOnGpuIfAvailable(GpuTest body) {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            // Skip silently: autocast on CPU has different op coverage, so
            // these dtype-propagation tests only make sense on CUDA/ROCm.
            return;
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            body.run(engine, manager, Device.gpu());
        }
    }
}
