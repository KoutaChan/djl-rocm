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
package ai.djl.engine.fusion;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;

import org.testng.Assert;
import org.testng.annotations.Test;

/** Tests functional fusion operations. */
public class FusionFunctionsTest {

    @Test
    public void projectedResidualMlpPreservesAutograd() {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(Device.cpu())) {
            NDArray input = manager.create(new float[] {1, 2, -1, 3}, new Shape(2, 2));
            NDArray combinedWeight =
                    manager.create(new float[] {1, 0, 0, 1, 1, -1}, new Shape(3, 2));
            NDArray combinedBias = manager.create(new float[] {0.5f, -0.25f, 0.75f});
            NDArray outputWeight = manager.create(new float[] {2, -1}, new Shape(1, 2));
            input.setRequiresGradient(true);
            combinedWeight.setRequiresGradient(true);
            combinedBias.setRequiresGradient(true);
            outputWeight.setRequiresGradient(true);

            NDArray output;
            try (GradientCollector collector = engine.newGradientCollector()) {
                output =
                        FusionFunctions.projectedResidualMlp(
                                input, combinedWeight, combinedBias, outputWeight);
                collector.backward(output.sum());
            }

            Assert.assertEquals(output.getShape(), new Shape(2, 1));
            assertFiniteNonZeroGradient(input);
            assertFiniteNonZeroGradient(combinedWeight);
            assertFiniteNonZeroGradient(combinedBias);
            assertFiniteNonZeroGradient(outputWeight);
        }
    }

    private static void assertFiniteNonZeroGradient(NDArray array) {
        Assert.assertTrue(array.hasGradient());
        boolean hasNonZero = false;
        for (float value : array.getGradient().toFloatArray()) {
            Assert.assertTrue(Float.isFinite(value));
            hasNonZero |= value != 0f;
        }
        Assert.assertTrue(hasNonZero, "Expected a nonzero gradient.");
    }
}
