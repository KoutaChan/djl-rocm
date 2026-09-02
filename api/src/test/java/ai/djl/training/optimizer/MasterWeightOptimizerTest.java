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
package ai.djl.training.optimizer;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Proxy;

public class MasterWeightOptimizerTest {

    private static final Shape SHAPE = new Shape(4);

    @Test
    public void rejectsIncompatibleShapes() {
        Optimizer optimizer = rejectingOptimizer();
        NDArray modelWeight = array(SHAPE, Device.cpu(), DataType.BFLOAT16);
        NDArray masterWeight = array(SHAPE, Device.cpu(), DataType.FLOAT32);
        NDArray gradient = array(SHAPE, Device.cpu(), DataType.FLOAT32);
        NDArray wrongShape = array(new Shape(3), Device.cpu(), DataType.FLOAT32);

        Assert.assertThrows(
                IllegalArgumentException.class,
                () ->
                        optimizer.updateWithMasterWeight(
                                "model-shape", wrongShape, masterWeight, gradient));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () ->
                        optimizer.updateWithMasterWeight(
                                "gradient-shape", modelWeight, masterWeight, wrongShape));
    }

    @Test
    public void rejectsIncompatibleDevices() {
        Optimizer optimizer = rejectingOptimizer();
        NDArray cpuModelWeight = array(SHAPE, Device.cpu(), DataType.BFLOAT16);
        NDArray cpuGradient = array(SHAPE, Device.cpu(), DataType.FLOAT32);
        NDArray gpuModelWeight = array(SHAPE, Device.gpu(), DataType.BFLOAT16);
        NDArray gpuMasterWeight = array(SHAPE, Device.gpu(), DataType.FLOAT32);
        NDArray gpuGradient = array(SHAPE, Device.gpu(), DataType.FLOAT32);

        Assert.assertThrows(
                IllegalArgumentException.class,
                () ->
                        optimizer.updateWithMasterWeight(
                                "weight-device", cpuModelWeight, gpuMasterWeight, gpuGradient));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () ->
                        optimizer.updateWithMasterWeight(
                                "gradient-device", gpuModelWeight, gpuMasterWeight, cpuGradient));
    }

    @Test
    public void rejectsIncompatibleDataTypes() {
        Optimizer optimizer = rejectingOptimizer();
        NDArray float16 = array(SHAPE, Device.cpu(), DataType.FLOAT16);
        NDArray bfloat16 = array(SHAPE, Device.cpu(), DataType.BFLOAT16);
        NDArray float32 = array(SHAPE, Device.cpu(), DataType.FLOAT32);
        NDArray int32 = array(SHAPE, Device.cpu(), DataType.INT32);
        NDArray complex64 = array(SHAPE, Device.cpu(), DataType.COMPLEX64);

        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> optimizer.updateWithMasterWeight("model-dtype", int32, float32, float32));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> optimizer.updateWithMasterWeight("master-dtype", bfloat16, int32, int32));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> optimizer.updateWithMasterWeight("gradient-dtype", bfloat16, float32, int32));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () ->
                        optimizer.updateWithMasterWeight(
                                "complex-model", complex64, float32, float32));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () ->
                        optimizer.updateWithMasterWeight(
                                "complex-master", float16, complex64, complex64));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () ->
                        optimizer.updateWithMasterWeight(
                                "lower-precision-master", float32, bfloat16, bfloat16));
    }

    private static Optimizer rejectingOptimizer() {
        return new Optimizer(Optimizer.sgd()) {
            @Override
            public void update(String parameterId, NDArray weight, NDArray grad) {
                Assert.fail("Invalid master-weight arguments reached the optimizer update");
            }
        };
    }

    private static NDArray array(Shape shape, Device device, DataType dataType) {
        return (NDArray)
                Proxy.newProxyInstance(
                        NDArray.class.getClassLoader(),
                        new Class<?>[] {NDArray.class},
                        (proxy, method, args) -> {
                            switch (method.getName()) {
                                case "getShape":
                                    return shape;
                                case "getDevice":
                                    return device;
                                case "getDataType":
                                    return dataType;
                                case "toString":
                                    return dataType + " " + shape + " " + device;
                                default:
                                    throw new UnsupportedOperationException(method.getName());
                            }
                        });
    }
}
