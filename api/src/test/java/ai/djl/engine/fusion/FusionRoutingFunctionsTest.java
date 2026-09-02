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
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.testing.Assertions;
import ai.djl.training.GradientCollector;

import org.testng.Assert;
import org.testng.annotations.Test;

public class FusionRoutingFunctionsTest {

    @Test
    public void indexedAffineGathersProjectsAndScatters() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            NDArray indices = manager.create(new int[] {2, 0});
            NDArray first = manager.create(new float[] {1, 2, 3, 4, 5, 6}, new Shape(3, 2));
            NDArray second = manager.create(new float[] {10, 20}, new Shape(2, 1));
            NDArray hiddenWeight = manager.create(new float[] {1, 0, 0, 0, 1, 0}, new Shape(2, 3));
            NDArray hiddenBias = manager.zeros(new Shape(2));
            NDArray outputWeight = manager.create(new float[] {1, 1}, new Shape(1, 2));
            NDArray outputBias = manager.zeros(new Shape(1));

            NDArray actual =
                    FusionFunctions.indexedAffine(
                            indices,
                            new NDList(first, second),
                            new long[] {1, 2},
                            3,
                            hiddenWeight,
                            hiddenBias,
                            outputWeight,
                            outputBias,
                            FusionRecipe.Activation.NONE);

            Assert.assertEquals(actual.getShape(), new Shape(3, 1));
            Assertions.assertAlmostEquals(
                    actual, manager.create(new float[] {3, 0, 11}, new Shape(3, 1)), 1e-6, 1e-6);
        }
    }

    @Test
    public void mappedGroupedMaskedSoftmaxPoolReturnsMappedOutputSets() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            NDArray scores = manager.zeros(new Shape(1, 3));
            NDArray masks = manager.create(new float[] {1, 0, 1, 1, 0, 1}, new Shape(1, 3, 2));
            NDArray values = manager.create(new float[] {1, 10, 3, 30, 5, 50}, new Shape(1, 3, 2));
            NDArray firstMapping = manager.create(new int[] {1, -1});
            NDArray secondMapping = manager.create(new int[] {0});

            MappedGroupedMaskedSoftmaxPoolResult actual =
                    FusionFunctions.mappedGroupedMaskedSoftmaxPool(
                            scores, masks, values, new NDList(firstMapping, secondMapping));

            Assert.assertEquals(actual.getContexts().size(), 2);
            Assert.assertEquals(actual.getPresence().size(), 2);
            Assertions.assertAlmostEquals(
                    actual.getContexts().get(0),
                    manager.create(new float[] {4, 40, 0, 0}, new Shape(1, 2, 2)),
                    1e-6,
                    1e-6);
            Assertions.assertAlmostEquals(
                    actual.getContexts().get(1),
                    manager.create(new float[] {2, 20}, new Shape(1, 1, 2)),
                    1e-6,
                    1e-6);
            Assertions.assertAlmostEquals(
                    actual.getPresence().get(0),
                    manager.create(new float[] {1, 0}, new Shape(1, 2)),
                    0,
                    0);
            Assertions.assertAlmostEquals(
                    actual.getPresence().get(1),
                    manager.create(new float[] {1}, new Shape(1, 1)),
                    0,
                    0);
        }
    }

    @Test
    public void indexedAffineAccumulatesRepeatedGatherGradients() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            NDArray indices = manager.create(new int[] {2, 3});
            NDArray source = manager.create(new float[] {1, 2}, new Shape(2, 1));
            NDArray hiddenWeight = manager.ones(new Shape(1, 1));
            NDArray outputWeight = manager.ones(new Shape(1, 1));
            source.setRequiresGradient(true);

            try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                NDArray output =
                        FusionFunctions.indexedAffine(
                                indices,
                                new NDList(source),
                                new long[] {2},
                                4,
                                hiddenWeight,
                                null,
                                outputWeight,
                                null,
                                FusionRecipe.Activation.NONE);
                collector.backward(output.sum());
            }

            Assertions.assertAlmostEquals(
                    source.getGradient(),
                    manager.create(new float[] {0, 2}, new Shape(2, 1)),
                    1e-6,
                    1e-6);
        }
    }

    @Test
    public void mappedGroupedMaskedSoftmaxPoolAccumulatesDuplicateMappingGradients() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            NDArray scores = manager.zeros(new Shape(1, 2));
            NDArray masks = manager.ones(new Shape(1, 2, 1));
            NDArray values = manager.create(new float[] {1, 3}, new Shape(1, 2, 1));
            scores.setRequiresGradient(true);
            values.setRequiresGradient(true);

            try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                MappedGroupedMaskedSoftmaxPoolResult output =
                        FusionFunctions.mappedGroupedMaskedSoftmaxPool(
                                scores,
                                masks,
                                values,
                                new NDList(
                                        manager.create(new int[] {0, 0}),
                                        manager.create(new int[] {0})));
                collector.backward(
                        output.getContexts().get(0).sum().add(output.getContexts().get(1).sum()));
            }

            Assertions.assertAlmostEquals(
                    scores.getGradient(),
                    manager.create(new float[] {-1.5f, 1.5f}, new Shape(1, 2)),
                    1e-6,
                    1e-6);
            Assertions.assertAlmostEquals(
                    values.getGradient(),
                    manager.create(new float[] {1.5f, 1.5f}, new Shape(1, 2, 1)),
                    1e-6,
                    1e-6);
        }
    }

    @Test
    public void mappedGroupedMaskedSoftmaxPoolDoesNotReadMaskedOrMissingValues() {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(Device.cpu())) {
            NDArray scores = manager.zeros(new Shape(1, 2));
            NDArray masks = manager.create(new float[] {1, 0, 0, 1}, new Shape(1, 2, 2));
            NDArray values = manager.create(new float[] {Float.NaN, 2}, new Shape(1, 2, 1));
            scores.setRequiresGradient(true);
            values.setRequiresGradient(true);

            MappedGroupedMaskedSoftmaxPoolResult output;
            try (GradientCollector collector = engine.newGradientCollector()) {
                output =
                        FusionFunctions.mappedGroupedMaskedSoftmaxPool(
                                scores,
                                masks,
                                values,
                                new NDList(manager.create(new int[] {1, -1})));
                collector.backward(output.getContexts().singletonOrThrow().sum());
            }

            Assertions.assertAlmostEquals(
                    output.getContexts().singletonOrThrow(),
                    manager.create(new float[] {2, 0}, new Shape(1, 2, 1)),
                    0,
                    0);
            Assertions.assertAlmostEquals(
                    output.getPresence().singletonOrThrow(),
                    manager.create(new float[] {1, 0}, new Shape(1, 2)),
                    0,
                    0);
            Assertions.assertAlmostEquals(
                    scores.getGradient(), manager.zeros(new Shape(1, 2)), 0, 0);
            Assertions.assertAlmostEquals(
                    values.getGradient(),
                    manager.create(new float[] {0, 1}, new Shape(1, 2, 1)),
                    0,
                    0);
        }
    }
}
