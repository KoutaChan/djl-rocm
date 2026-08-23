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
