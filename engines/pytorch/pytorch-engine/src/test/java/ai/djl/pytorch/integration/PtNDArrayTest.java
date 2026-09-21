/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.pytorch.integration;

import ai.djl.engine.EngineException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.testing.TestRequirements;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.Arrays;

public class PtNDArrayTest {

    @Test
    public void testStringTensor() {
        TestRequirements.notMacX86();

        try (NDManager manager = NDManager.newBaseManager()) {
            String[] str = {"a", "b", "c"};
            NDArray arr = manager.create(str);
            Assert.assertEquals(arr.toString(), Arrays.toString(str));
            Assert.assertEquals(arr.toDebugString(), Arrays.toString(str));
            Assert.assertEquals(arr.toDebugString(true), Arrays.toString(str));

            Assert.assertThrows(UnsupportedOperationException.class, () -> arr.get(0));
        }
    }

    @Test
    public void testLargeTensor() {
        TestRequirements.notMacX86();

        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.zeros(new Shape(10 * 2850, 18944), DataType.FLOAT32);
            Assert.assertThrows(EngineException.class, array::toByteArray);
        }
    }

    @Test
    public void testScalarOperations() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray values = manager.create(new float[] {-2.0f, 0.0f, 3.0f});

            Assert.assertEquals(values.add(2.0f).toFloatArray(), new float[] {0.0f, 2.0f, 5.0f});
            Assert.assertEquals(values.sub(1).toFloatArray(), new float[] {-3.0f, -1.0f, 2.0f});
            Assert.assertEquals(values.mul(2).toFloatArray(), new float[] {-4.0f, 0.0f, 6.0f});
            Assert.assertEquals(values.div(2.0f).toFloatArray(), new float[] {-1.0f, 0.0f, 1.5f});
            Assert.assertEquals(
                    values.maximum(0.0f).toFloatArray(), new float[] {0.0f, 0.0f, 3.0f});
            Assert.assertEquals(
                    values.minimum(0.0f).toFloatArray(), new float[] {-2.0f, 0.0f, 0.0f});
            Assert.assertEquals(values.eq(0).toBooleanArray(), new boolean[] {false, true, false});
            Assert.assertEquals(
                    values.gte(0.0f).toBooleanArray(), new boolean[] {false, true, true});

            NDArray mutable = values.duplicate();
            mutable.addi(1.0f).muli(2).subi(2.0f).divi(2);
            Assert.assertEquals(mutable.toFloatArray(), values.toFloatArray());
        }
    }

    @Test
    public void testFloorDivideIntegerScalars() {
        int[] ints = {
            Integer.MIN_VALUE, Integer.MIN_VALUE + 1, -16_777_217, -117,
            -116, -115, -1, 0,
            1, 115, 116, 117,
            16_777_217, Integer.MAX_VALUE - 1, Integer.MAX_VALUE, 40
        };
        long[] longs = {
            Long.MIN_VALUE, Long.MIN_VALUE + 1, -9_007_199_254_740_993L, -117,
            -116, -115, -1, 0,
            1, 115, 116, 117,
            9_007_199_254_740_993L, Long.MAX_VALUE - 1, Long.MAX_VALUE, 40
        };
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray intValues = manager.create(ints, new Shape(4, 4));
            NDArray longValues = manager.create(longs, new Shape(4, 4));
            for (int divisor : new int[] {-116, -40, -3, 1, 2, 3, 13, 34, 40, 116}) {
                int[] expectedInts = new int[ints.length];
                long[] expectedLongs = new long[longs.length];
                for (int i = 0; i < ints.length; ++i) {
                    expectedInts[i] = Math.floorDiv(ints[i], divisor);
                    expectedLongs[i] = Math.floorDiv(longs[i], divisor);
                }
                try (NDArray intResult = intValues.floorDivide(divisor);
                        NDArray longResult = longValues.floorDivide((long) divisor)) {
                    Assert.assertEquals(intResult.getDataType(), DataType.INT32);
                    Assert.assertEquals(longResult.getDataType(), DataType.INT64);
                    Assert.assertEquals(intResult.getShape(), intValues.getShape());
                    Assert.assertEquals(longResult.getShape(), longValues.getShape());
                    Assert.assertEquals(intResult.toIntArray(), expectedInts);
                    Assert.assertEquals(longResult.toLongArray(), expectedLongs);
                }
            }
            Assert.assertEquals(intValues.toIntArray(), ints);
            Assert.assertEquals(longValues.toLongArray(), longs);
            NDArray empty = manager.zeros(new Shape(0, 3), DataType.INT32).floorDivide(40);
            Assert.assertEquals(empty.getShape(), new Shape(0, 3));
            Assert.assertEquals(empty.getDataType(), DataType.INT32);
            NDArray scalar = manager.create(-117L).floorDivide(116);
            Assert.assertEquals(scalar.getShape(), new Shape());
            Assert.assertEquals(scalar.getDataType(), DataType.INT64);
            Assert.assertEquals(scalar.toLongArray(), new long[] {-2});
        }
    }

    @Test
    public void testFloorDivideFloatingScalarsAndZero() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray values = manager.create(new float[] {-3.5f, -1.0f, 0.0f, 1.0f, 3.5f});
            NDArray result = values.floorDivide(2.0f);
            Assert.assertEquals(result.getDataType(), DataType.FLOAT32);
            Assert.assertEquals(result.getShape(), values.getShape());
            Assert.assertEquals(result.toFloatArray(), new float[] {-2, -1, 0, 0, 1});
            Assert.assertEquals(
                    values.floorDivide(-2.0).toFloatArray(), new float[] {1, 0, -0.0f, -1, -2});
            NDArray integers = manager.create(new int[] {-3, -1, 0, 1, 3});
            Assert.assertEquals(
                    integers.floorDivide(-1).toIntArray(), new int[] {3, 1, 0, -1, -3});
            NDArray promoted = integers.floorDivide(2.0f);
            Assert.assertEquals(promoted.getDataType(), DataType.FLOAT32);
            Assert.assertEquals(promoted.toFloatArray(), new float[] {-2, -1, 0, 0, 1});
            for (Number zero : new Number[] {0, 0L, 0.0f, -0.0f, 0.0, -0.0}) {
                Assert.assertThrows(IllegalArgumentException.class, () -> values.floorDivide(zero));
                Assert.assertThrows(IllegalArgumentException.class, () -> integers.floorDivide(zero));
            }
        }
    }

    @Test
    public void testCopyToConvertsFloatingDataType() {
        try (NDManager manager = NDManager.newBaseManager();
                NDArray source = manager.create(new float[] {1.25f, -2.5f, 3.75f});
                NDArray target = manager.zeros(new Shape(3), DataType.BFLOAT16)) {
            source.copyTo(target);
            try (NDArray restored = target.toType(DataType.FLOAT32, false)) {
                Assert.assertEquals(
                        restored.toFloatArray(), new float[] {1.25f, -2.5f, 3.75f}, 1e-2f);
            }
        }
    }

    @Test
    public void testCopyToPortableFallbackConvertsFloatingDataType() {
        try (NDManager manager = NDManager.newBaseManager();
                NDArray source =
                        manager.create(new float[] {1.25f, -2.5f, 3.75f}, new Shape(1, 3));
                NDArray target = manager.zeros(new Shape(3), DataType.BFLOAT16)) {
            source.copyTo(target);
            try (NDArray restored = target.toType(DataType.FLOAT32, false)) {
                Assert.assertEquals(
                        restored.toFloatArray(), new float[] {1.25f, -2.5f, 3.75f}, 1e-2f);
            }
        }
    }

    @Test
    public void testSwish() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray values = manager.create(new float[] {-2.0f, 0.0f, 3.0f});
            float[] expected = values.mul(Activation.sigmoid(values)).toFloatArray();

            Assert.assertEquals(Activation.swish(values, 1.0f).toFloatArray(), expected, 1e-6f);
        }
    }
}
