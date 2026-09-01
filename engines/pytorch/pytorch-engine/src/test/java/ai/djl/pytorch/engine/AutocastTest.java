/*
 * Copyright 2025 KoutaChan.
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
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.jni.JniUtils;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.Test;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * Tests the PyTorch {@link Autocast} implementation. Matrix multiplication and scaled dot-product
 * attention should use the autocast data type, RMSNorm should remain in FLOAT32, nested scopes
 * should compose, and closing a scope should restore the previous state.
 */
@SuppressWarnings("try") // Autocast resources are used for their scope side effects.
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

        int previousDataType = JniUtils.autocastGetDataType(CPU_DEVICE);
        boolean previousEnabled = JniUtils.autocastIsEnabled(CPU_DEVICE);
        try (Autocast ac = engine.newAutocast(Device.cpu(), DataType.BFLOAT16)) {
            Assert.assertTrue(JniUtils.autocastIsEnabled(CPU_DEVICE));
            Assert.assertEquals(
                    JniUtils.autocastGetDataType(CPU_DEVICE), DataType.BFLOAT16.ordinal());
        }
        Assert.assertEquals(JniUtils.autocastIsEnabled(CPU_DEVICE), previousEnabled);
        Assert.assertEquals(JniUtils.autocastGetDataType(CPU_DEVICE), previousDataType);
    }

    @Test
    public void nestedScopesRestoreOuterState() {
        Engine engine = Engine.getInstance();
        try (Autocast outer = engine.newAutocast(Device.cpu(), DataType.BFLOAT16)) {
            Assert.assertTrue(JniUtils.autocastIsEnabled(CPU_DEVICE));
            Assert.assertEquals(
                    JniUtils.autocastGetDataType(CPU_DEVICE), DataType.BFLOAT16.ordinal());

            try (Autocast inner = engine.newAutocast(Device.cpu(), DataType.FLOAT16)) {
                Assert.assertEquals(
                        JniUtils.autocastGetDataType(CPU_DEVICE), DataType.FLOAT16.ordinal());
            }
            // After the inner scope exits, the outer scope's BFLOAT16 state must
            // be visible again.
            Assert.assertTrue(JniUtils.autocastIsEnabled(CPU_DEVICE));
            Assert.assertEquals(
                    JniUtils.autocastGetDataType(CPU_DEVICE), DataType.BFLOAT16.ordinal());
        }
        Assert.assertFalse(JniUtils.autocastIsEnabled(CPU_DEVICE));
    }

    @Test
    public void disabledNestedScopeRestoresOuterState() {
        Engine engine = Engine.getInstance();
        try (Autocast outer = engine.newAutocast(Device.cpu(), DataType.BFLOAT16, true)) {
            Assert.assertTrue(JniUtils.autocastIsEnabled(CPU_DEVICE));
            Assert.assertTrue(JniUtils.autocastIsCacheEnabled());

            try (Autocast disabled =
                    engine.newAutocast(Device.cpu(), DataType.BFLOAT16, false, true)) {
                Assert.assertFalse(JniUtils.autocastIsEnabled(CPU_DEVICE));
                Assert.assertTrue(JniUtils.autocastIsCacheEnabled());
            }

            Assert.assertTrue(JniUtils.autocastIsEnabled(CPU_DEVICE));
            Assert.assertEquals(
                    JniUtils.autocastGetDataType(CPU_DEVICE), DataType.BFLOAT16.ordinal());
            Assert.assertTrue(JniUtils.autocastIsCacheEnabled());
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
    public void scopeRejectsCloseFromAnotherThread() throws Exception {
        Engine engine = Engine.getInstance();
        boolean previousEnabled = JniUtils.autocastIsEnabled(CPU_DEVICE);
        Autocast autocast = engine.newAutocast(Device.cpu(), DataType.BFLOAT16);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<Throwable> future =
                    executor.submit(
                            () -> {
                                try {
                                    autocast.close();
                                    return null;
                                } catch (Throwable t) {
                                    return t;
                                }
                            });

            Throwable thrown = future.get(30, TimeUnit.SECONDS);
            Assert.assertTrue(thrown instanceof IllegalStateException);
            Assert.assertTrue(JniUtils.autocastIsEnabled(CPU_DEVICE));
        } finally {
            executor.shutdownNow();
            autocast.close();
        }
        Assert.assertEquals(JniUtils.autocastIsEnabled(CPU_DEVICE), previousEnabled);
    }

    @Test
    public void sdpaUsesAutocastDataType() {
        // SDPA is in PyTorch's autocast allowlist, so FLOAT32 inputs inside a
        // BFLOAT16 autocast scope must produce a BFLOAT16 output on CUDA/ROCm.
        runOnGpuIfAvailable(
                (engine, manager, device) -> {
                    NDArray q =
                            manager.randomNormal(new Shape(1, 4, 8, 16)).toDevice(device, false);
                    NDArray k =
                            manager.randomNormal(new Shape(1, 4, 8, 16)).toDevice(device, false);
                    NDArray v =
                            manager.randomNormal(new Shape(1, 4, 8, 16)).toDevice(device, false);
                    Assert.assertEquals(q.getDataType(), DataType.FLOAT32);

                    try (Autocast ac = engine.newAutocast(device, DataType.BFLOAT16)) {
                        NDArray out =
                                q.getNDArrayInternal()
                                        .scaledDotProductAttention(k, v, null, 0.0, false);
                        Assert.assertEquals(
                                out.getDataType(),
                                DataType.BFLOAT16,
                                "SDPA must emit the autocast data type");
                        Assert.assertTrue(isAllFinite(out), "SDPA output must be finite");
                    }
                });
    }

    @Test
    public void rmsNormStaysFloat32() {
        // RMS normalization is numerically sensitive, so PyTorch's autocast
        // policy keeps a FLOAT32 input in FLOAT32.
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
                                DataType.FLOAT32,
                                "Numerically sensitive RMSNorm must stay in FLOAT32");
                        Assert.assertTrue(isAllFinite(out), "rms_norm output must be finite");
                    }
                });
    }

    @Test
    public void matmulUsesAutocastDataType() {
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
        // FLOAT32 matmul and autocast BFLOAT16 matmul on random inputs should match
        // within BFLOAT16's ~1e-2 relative error. Guards against the autocast
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
                    float maxAbsError = diff.max().getFloat();
                    float maxReference = fpOut.abs().max().getFloat();
                    float relativeMaxError = maxAbsError / (maxReference + 1e-6f);
                    Assert.assertTrue(
                            relativeMaxError < 5e-2f,
                            "BFLOAT16 matmul relative max error "
                                    + relativeMaxError
                                    + " exceeds tolerance");
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

    private static void runOnGpuIfAvailable(GpuTest body) {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This autocast test requires a PyTorch GPU.");
        }
        try (NDManager manager = engine.newBaseManager(Device.gpu())) {
            body.run(engine, manager, Device.gpu());
        }
    }

    @FunctionalInterface
    private interface GpuTest {
        void run(Engine engine, NDManager manager, Device device);
    }
}
