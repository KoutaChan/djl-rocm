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
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.jni.JniUtils;

import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.nio.ByteBuffer;

public class PinnedBufferAllocationTest {

    private static final Device TEST_DEVICE =
            Device.fromName(System.getProperty("ai.djl.pytorch.test.device", "cpu"));

    @DataProvider
    public Object[][] elementTypes() {
        return new Object[][] {{DataType.INT16}, {DataType.INT32}, {DataType.FLOAT32}};
    }

    @Test(dataProvider = "elementTypes")
    public void testTransferKeepsAllocatedShapeWhenCallerChangesDimensions(DataType dataType) {
        try (PtNDManager manager = (PtNDManager) NDManager.newBaseManager(TEST_DEVICE, "PyTorch");
                PtPinnedBuffer buffer = manager.allocatePinned(8, dataType);
                PtPinnedBuffer tooSmall = manager.allocatePinned(5, dataType);
                PtPinnedBuffer wrongType = manager.allocatePinned(8, DataType.FLOAT64)) {
            Shape requested = new Shape(2, 3);
            PtNDArray array = manager.create(requested, dataType);
            requested.getShape()[0] = 99;
            ByteBuffer data = buffer.getByteBuffer();
            for (int index = 0; index < 8; ++index) {
                if (dataType == DataType.INT16) {
                    data.putShort((short) (index + 1));
                } else if (dataType == DataType.INT32) {
                    data.putInt(index + 1);
                } else {
                    data.putFloat(index + 1);
                }
            }
            array.enqueueCopyFrom(buffer);
            Assert.assertEquals(array.getShape(), new Shape(2, 3));
            Assert.assertEquals(array.getShape(), JniUtils.getShape(array));
            Assert.assertEquals(array.getDataType(), JniUtils.getDataType(array));
            Assert.assertEquals(
                    array.toType(DataType.FLOAT32, false).toFloatArray(),
                    new float[] {1, 2, 3, 4, 5, 6});
            Assert.assertThrows(
                    IllegalArgumentException.class, () -> array.enqueueCopyFrom(tooSmall));
            Assert.assertThrows(
                    IllegalArgumentException.class, () -> array.enqueueCopyFrom(wrongType));
            Assert.assertSame(array.getManager(), manager);
        }
    }

    @Test
    public void testLegacyComplexAllocationUsesNativeDataType() {
        try (PtNDManager manager = (PtNDManager) NDManager.newBaseManager(TEST_DEVICE, "PyTorch")) {
            PtNDArray array = manager.create(new Shape(2, 3), DataType.COMPLEX64);
            Assert.assertEquals(array.getShape(), JniUtils.getShape(array));
            Assert.assertEquals(array.getDataType(), JniUtils.getDataType(array));
        }
    }
}
