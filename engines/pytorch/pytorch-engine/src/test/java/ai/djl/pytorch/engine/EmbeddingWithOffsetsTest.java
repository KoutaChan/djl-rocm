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

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;

import org.testng.Assert;
import org.testng.annotations.Test;

public class EmbeddingWithOffsetsTest {

    @Test
    public void testBroadcastOffsetsAndTableDtype() {
        try (NDManager manager = Engine.getInstance().newBaseManager()) {
            NDArray rawIds =
                    manager.create(new int[] {0, 1, 0, 1, 0, 1})
                            .toType(DataType.INT16, false)
                            .reshape(2, 3);
            NDArray offsets = manager.create(new int[] {0, 2, 4}).reshape(1, 3);
            NDArray table = manager.arange(12).reshape(6, 2).toType(DataType.FLOAT16, false);

            NDArray actual = NDArrays.embeddingWithOffsets(rawIds, offsets, table);

            Assert.assertEquals(actual.getShape(), new Shape(2, 3, 2));
            Assert.assertEquals(actual.getDataType(), DataType.FLOAT16);
            Assert.assertEquals(
                    actual.toFloatArray(), new float[] {0, 1, 6, 7, 8, 9, 2, 3, 4, 5, 10, 11});
        }
    }

    @Test
    public void testDenseEmbeddingGradient() {
        try (NDManager manager = Engine.getInstance().newBaseManager()) {
            NDArray rawIds = manager.create(new long[] {0, 1, 3, 1}).reshape(2, 2);
            NDArray offsets =
                    manager.create(new int[] {0, 2}).toType(DataType.INT16, false).reshape(1, 2);
            NDArray table = manager.arange(8).toType(DataType.FLOAT32, false).reshape(4, 2);
            table.setRequiresGradient(true);

            try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                collector.backward(NDArrays.embeddingWithOffsets(rawIds, offsets, table).sum());
            }

            Assert.assertEquals(
                    table.getGradient().toFloatArray(), new float[] {1, 1, 0, 0, 0, 0, 3, 3});
        }
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testRejectsNonBroadcastOffsets() {
        try (NDManager manager = Engine.getInstance().newBaseManager()) {
            NDArray rawIds = manager.zeros(new Shape(2, 3), DataType.INT32);
            NDArray offsets = manager.zeros(new Shape(1, 2), DataType.INT32);
            NDArray table = manager.zeros(new Shape(4, 2));
            NDArrays.embeddingWithOffsets(rawIds, offsets, table);
        }
    }
}
