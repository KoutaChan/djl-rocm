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
package ai.djl.training;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;

import org.testng.Assert;
import org.testng.annotations.Test;

public class TrainerGradientCheckTest {

    @Test
    public void acceptsNonZeroGradientsWhoseSignedSumCancels() {
        Engine engine = Engine.getInstance();
        Device device = Device.cpu();
        Linear block = Linear.builder().setUnits(1).optBias(false).build();
        Optimizer optimizer = Optimizer.sgd().setLearningRateTracker(Tracker.fixed(0.1f)).build();
        DefaultTrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optDevices(new Device[] {device})
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)
                        .optOptimizer(optimizer);

        try (Model model =
                Model.newInstance("cancelling-gradient", device, engine.getEngineName())) {
            model.setBlock(block);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(1, 2));
                NDArray weight = block.getParameters().valueAt(0).getArray();
                float[] before = weight.toFloatArray();

                try (NDManager scoped = trainer.getManager().newSubManager();
                        GradientCollector collector = trainer.newGradientCollector();
                        NDArray input = scoped.create(new float[] {1f, -1f}, new Shape(1, 2));
                        NDList output = trainer.forward(new NDList(input));
                        NDArray backwardTarget = output.singletonOrThrow().sum()) {
                    collector.backward(backwardTarget);
                }

                try (NDArray gradient = weight.getGradient();
                        NDArray sum = gradient.sum();
                        NDArray nonZero = gradient.neq(0);
                        NDArray anyNonZero = nonZero.any()) {
                    Assert.assertEquals(sum.getFloat(), 0f);
                    Assert.assertTrue(anyNonZero.getBoolean());
                }

                trainer.step();
                float[] after = weight.toFloatArray();
                Assert.assertTrue(after[0] < before[0]);
                Assert.assertTrue(after[1] > before[1]);
            }
        }
    }
}
