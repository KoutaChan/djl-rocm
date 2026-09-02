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
package ai.djl.ndarray;

import ai.djl.Device;
import ai.djl.ndarray.types.Shape;
import ai.djl.testing.Assertions;

import org.testng.Assert;
import org.testng.annotations.Test;

public class ProjectedResidualMlpApiTest {

    @Test
    public void projectedResidualMlpSupportsLeadingDimensions() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            NDArray input =
                    manager.create(new float[] {1, 2, 3, 4, -1, 2, 0, -3}, new Shape(2, 2, 2));
            NDArray combinedWeight =
                    manager.create(new float[] {1, 0, 0, 1, 1, 1}, new Shape(3, 2));
            NDArray combinedBias = manager.create(new float[] {0.5f, 0, 0});
            NDArray outputWeight = manager.create(new float[] {2, -1}, new Shape(1, 2));

            NDArray actual =
                    NDArrays.projectedResidualMlp(
                            input, combinedWeight, combinedBias, outputWeight);
            NDArray combined =
                    input.getNDArrayInternal()
                            .linear(input, combinedWeight, combinedBias)
                            .singletonOrThrow();
            NDArray expected =
                    combined.get("...,0:1")
                            .add(
                                    input.getNDArrayInternal()
                                            .linear(
                                                    ai.djl.nn.Activation.swish(
                                                            combined.get("...,1:"), 1.0f),
                                                    outputWeight,
                                                    null)
                                            .singletonOrThrow());

            Assert.assertEquals(actual.getShape(), new Shape(2, 2, 1));
            Assertions.assertAlmostEquals(actual, expected, 1e-6, 1e-6);
        }
    }

    @Test
    public void projectedResidualMlpRejectsIncompatibleShapes() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            NDArray input = manager.zeros(new Shape(2, 6));
            NDArray combinedWeight = manager.zeros(new Shape(8, 6));
            NDArray combinedBias = manager.zeros(new Shape(7));
            NDArray outputWeight = manager.zeros(new Shape(3, 4));

            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            NDArrays.projectedResidualMlp(
                                    input, combinedWeight, combinedBias, outputWeight));
        }
    }
}
