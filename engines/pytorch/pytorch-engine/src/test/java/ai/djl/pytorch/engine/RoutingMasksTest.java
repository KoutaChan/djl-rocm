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

import org.testng.Assert;
import org.testng.annotations.Test;

/** Verifies portable and ROCm routing-mask semantics. */
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
            NDArray firstMask =
                    manager.create(new float[] {1f, 0f, 0.5f, 0f}, new Shape(2, 2))
                            .toType(maskType, false);
            NDArray secondMask =
                    manager.create(new float[] {0f, 1f, 0f, 0f}, new Shape(2, 2))
                            .toType(maskType, false);
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

    private static void assertClose(float[] actual, float[] expected, float tolerance) {
        Assert.assertEquals(actual.length, expected.length);
        for (int index = 0; index < actual.length; ++index) {
            Assert.assertEquals(actual[index], expected[index], tolerance, "index=" + index);
        }
    }
}
