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
import ai.djl.pytorch.jni.JniUtils;

import org.testng.Assert;
import org.testng.annotations.Test;

/** Verifies portable and native routing-mask semantics. */
public class RoutingMasksTest {

    /** Verifies field selection, category sets, and mask propagation. */
    @Test
    public void categoricalRulesSelectFieldsAndCategorySets() {
        Engine engine = Engine.getInstance();
        verifyCategoricalMasks(engine, Device.cpu(), DataType.INT32, DataType.FLOAT32);
        if (engine.getGpuCount() > 0) {
            for (DataType indexType :
                    new DataType[] {DataType.INT16, DataType.INT32, DataType.INT64}) {
                for (DataType maskType :
                        new DataType[] {DataType.FLOAT16, DataType.BFLOAT16, DataType.FLOAT32}) {
                    verifyCategoricalMasks(engine, Device.gpu(), indexType, maskType);
                }
            }
        }
    }

    /** Verifies the combined, presence, and representative channels. */
    @Test
    public void binaryChoiceMasksPreserveEncodedRouteSemantics() {
        Engine engine = Engine.getInstance();
        verifyBinaryChoiceMasks(engine, Device.cpu(), DataType.INT32, DataType.FLOAT32);
        if (engine.getGpuCount() > 0) {
            for (DataType indexType :
                    new DataType[] {DataType.INT16, DataType.INT32, DataType.INT64}) {
                for (DataType maskType :
                        new DataType[] {DataType.FLOAT16, DataType.BFLOAT16, DataType.FLOAT32}) {
                    verifyBinaryChoiceMasks(engine, Device.gpu(), indexType, maskType);
                }
            }
        }
    }

    @Test
    public void categoricalRulesPreserveIntegerBoundsAndNonFiniteMasks() {
        Engine engine = Engine.getInstance();
        for (DataType maskType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            for (boolean strided : new boolean[] {false, true}) {
                verifyCategoricalBounds(engine, Device.cpu(), maskType, strided);
                if (hasCudaBackend(engine)) {
                    verifyCategoricalBounds(engine, Device.gpu(), maskType, strided);
                }
            }
        }
    }

    @Test
    public void categoricalRuleCountsBeyondNativeLimitUsePortableSemantics() {
        Engine engine = Engine.getInstance();
        verifyManyCategoricalRules(engine, Device.cpu());
        if (engine.getGpuCount() > 0) {
            verifyManyCategoricalRules(engine, Device.gpu());
        }
    }

    @Test
    public void binaryChoiceMasksPreserveLongPaddingAndNonFiniteArithmetic() {
        Engine engine = Engine.getInstance();
        for (DataType maskType :
                new DataType[] {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}) {
            for (boolean strided : new boolean[] {false, true}) {
                float[] expected = binaryBoundaryResult(engine, Device.cpu(), maskType, strided);
                if (hasCudaBackend(engine)) {
                    float[] actual = binaryBoundaryResult(engine, Device.gpu(), maskType, strided);
                    assertClose(actual, expected, 0f);
                }
            }
        }
    }

    private static void verifyCategoricalBounds(
            Engine engine, Device device, DataType maskType, boolean strided) {
        long[] categories = {
            -1L,
            0L,
            63L,
            64L,
            Long.MIN_VALUE,
            Long.MAX_VALUE,
            1L,
            62L,
            63L,
            64L,
            -1L,
            0L,
            1L,
            62L,
            Long.MAX_VALUE,
            Long.MIN_VALUE
        };
        float[] masks = {
            1f, 0.5f, -0.25f, 2f, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, -0f
        };
        int[] fields = {0, 1, 0, 1};
        long[] sets = {1L | Long.MIN_VALUE, -1L, 0L, Long.MIN_VALUE};
        float[] expected = new float[masks.length * fields.length];
        for (int row = 0; row < masks.length; ++row) {
            for (int rule = 0; rule < fields.length; ++rule) {
                long category = categories[fields[rule] * masks.length + row];
                boolean selected =
                        category >= 0 && category < 64 && (sets[rule] & (1L << category)) != 0;
                expected[row * fields.length + rule] = (selected ? 1f : 0f) * masks[row];
            }
        }
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray metadata =
                    manager.create(categories, new Shape(2, masks.length)).transpose(1, 0);
            if (!strided) {
                metadata = metadata.reshape(-1).reshape(masks.length, 2);
            }
            NDArray mask = manager.create(masks).toType(maskType, false);
            if (strided) {
                mask = mask.expandDims(1).repeat(1, 2).get(":,0");
            }
            NDArray actual = NDArrays.categoricalMasks(metadata, mask, fields, sets);
            Assert.assertEquals(actual.getDataType(), maskType);
            Assert.assertEquals(actual.getShape(), new Shape(masks.length, fields.length));
            assertClose(actual.toType(DataType.FLOAT32, false).toFloatArray(), expected, 0f);
        }
    }

    private static void verifyManyCategoricalRules(Engine engine, Device device) {
        int[] fields = new int[65];
        long[] sets = new long[fields.length];
        float[] expected = new float[3 * fields.length];
        for (int rule = 0; rule < fields.length; ++rule) {
            sets[rule] = rule % 2 == 0 ? 1L : Long.MIN_VALUE;
            expected[rule] = rule % 2 == 0 ? 1f : 0f;
            expected[fields.length + rule] = rule % 2 == 0 ? 0f : 1f;
        }
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray categories = manager.create(new long[] {0L, 63L, 64L}, new Shape(3, 1));
            NDArray result =
                    NDArrays.categoricalMasks(categories, manager.ones(new Shape(3)), fields, sets);
            assertClose(result.toFloatArray(), expected, 0f);
        }
    }

    private static float[] binaryBoundaryResult(
            Engine engine, Device device, DataType maskType, boolean strided) {
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray routes =
                    manager.create(
                                    new long[] {
                                        Long.MIN_VALUE, Long.MAX_VALUE, 0L, Long.MIN_VALUE,
                                        Long.MIN_VALUE, Long.MIN_VALUE, Long.MAX_VALUE,
                                                Long.MIN_VALUE,
                                        Long.MAX_VALUE, Long.MIN_VALUE, Long.MIN_VALUE,
                                                Long.MIN_VALUE
                                    },
                                    new Shape(3, 4))
                            .transpose(1, 0);
            if (!strided) {
                routes = routes.reshape(-1).reshape(4, 3);
            }
            NDArray masks =
                    manager.create(
                                    new float[] {
                                        0.1f,
                                        0.2f,
                                        -1f,
                                        Float.NaN,
                                        1f,
                                        -1f,
                                        Float.POSITIVE_INFINITY,
                                        Float.NEGATIVE_INFINITY,
                                        -1f,
                                        65504f,
                                        65504f,
                                        -1f
                                    },
                                    new Shape(4, 3))
                            .toType(maskType, false);
            NDArray result =
                    NDArrays.binaryChoiceMasks(
                            routes, masks.get(":,0"), masks.get(":,1"), 0, 1, 2, Long.MIN_VALUE);
            Assert.assertEquals(result.getDataType(), maskType);
            Assert.assertEquals(result.getShape(), new Shape(4, 4));
            return result.toType(DataType.FLOAT32, false).toFloatArray();
        }
    }

    private static void verifyCategoricalMasks(
            Engine engine, Device device, DataType indexType, DataType maskType) {
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray categories =
                    manager.create(
                                    new int[] {1, 1, 2, 2, 3, 3, 2, 3, 1, 2, 0, 0},
                                    new Shape(2, 3, 2))
                            .toType(indexType, false);
            NDArray mask =
                    manager.create(new float[] {1f, 0.5f, 0f, 0.25f, 1f, 0f}, new Shape(2, 3))
                            .toType(maskType, false);
            NDArray result =
                    NDArrays.categoricalMasks(
                            categories,
                            mask,
                            new int[] {0, 0, 1, 1, 1},
                            new long[] {1L << 1, 1L << 2, 1L << 1, 1L << 2, (1L << 2) | (1L << 3)});

            Assert.assertEquals(result.getShape(), new Shape(2, 3, 5));
            Assert.assertEquals(result.getDataType(), maskType);
            assertClose(
                    result.toType(DataType.FLOAT32, false).toFloatArray(),
                    new float[] {
                        1f, 0f, 1f, 0f, 0f,
                        0f, 0.5f, 0f, 0.5f, 0.5f,
                        0f, 0f, 0f, 0f, 0f,
                        0f, 0.25f, 0f, 0f, 0.25f,
                        1f, 0f, 0f, 1f, 1f,
                        0f, 0f, 0f, 0f, 0f
                    },
                    1e-3f);
        }
    }

    private static void verifyBinaryChoiceMasks(
            Engine engine, Device device, DataType indexType, DataType maskType) {
        try (NDManager manager = engine.newBaseManager(device)) {
            NDArray routes =
                    manager.create(
                                    new int[] {7, 1, 0, 2, 0, 1, 1, 0, 9, 2, 1, 2, 0, 0, 0, 0},
                                    new Shape(2, 2, 4))
                            .toType(indexType, false);
            NDArray maskChannels =
                    manager.create(
                                    new float[] {
                                        1f, 0f, -1f,
                                        0f, 1f, -1f,
                                        0.5f, 0f, -1f,
                                        0f, 0f, -1f
                                    },
                                    new Shape(2, 2, 3))
                            .toType(maskType, false);
            NDArray firstMask = maskChannels.get("...,0");
            NDArray secondMask = maskChannels.get("...,1");
            NDArray result = NDArrays.binaryChoiceMasks(routes, firstMask, secondMask, 0, 2, 3, 0);

            Assert.assertEquals(result.getShape(), new Shape(2, 2, 4));
            Assert.assertEquals(result.getDataType(), maskType);
            assertClose(
                    result.toType(DataType.FLOAT32, false).toFloatArray(),
                    new float[] {
                        1f, 0f, 1f, 1f,
                        1f, 1f, 0f, 0f,
                        0.5f, 0.5f, 0.5f, 0.5f,
                        0f, 0f, 0f, 0f
                    },
                    1e-3f);
        }
    }

    private static boolean hasCudaBackend(Engine engine) {
        // Fusion reports the native build backend: CUDA = 1, ROCm = 2.
        return engine.getGpuCount() > 0 && JniUtils.getFusionBackend() == 1;
    }

    private static void assertClose(float[] actual, float[] expected, float tolerance) {
        Assert.assertEquals(actual.length, expected.length);
        for (int index = 0; index < actual.length; ++index) {
            if (Float.isNaN(expected[index])) {
                Assert.assertTrue(Float.isNaN(actual[index]), "index=" + index);
            } else if (Float.isInfinite(expected[index])) {
                Assert.assertEquals(actual[index], expected[index], "index=" + index);
            } else {
                Assert.assertEquals(actual[index], expected[index], tolerance, "index=" + index);
            }
        }
    }
}
