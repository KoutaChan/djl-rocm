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
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.core.Linear;
import ai.djl.testing.TestRequirements;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;

import org.testng.Assert;
import org.testng.annotations.Test;

/** Checks that optimizer state does not outlive the training that created it. */
public class OptimizerMemoryTest {

    private static final int WIDTH = 512;
    private static final int TRAININGS = 4;

    @Test
    public void deviceMemoryStaysFlatAcrossTrainings() {
        TestRequirements.gpu(PtEngine.ENGINE_NAME);
        PtEngine engine = (PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME);
        Device device = Device.gpu(0);
        long[] allocated = new long[TRAININGS];
        for (int training = 0; training < TRAININGS; training++) {
            train(device, Optimizer.adamW().build());
            allocated[training] = engine.getMemoryStats(device).getAllocatedBytes();
        }
        // The first training also allocates the BLAS workspace of this thread.
        for (int training = 1; training < TRAININGS; training++) {
            Assert.assertEquals(
                    allocated[training],
                    allocated[1],
                    "bytes allocated on " + device + " after training " + training);
        }
    }

    private static void train(Device device, Optimizer optimizer) {
        DefaultTrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optDevices(new Device[] {device})
                        .optOptimizer(optimizer);
        try (Model model = Model.newInstance("optimizer-memory", device, PtEngine.ENGINE_NAME)) {
            model.setBlock(Linear.builder().setUnits(WIDTH).build());
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(1, WIDTH));
                for (int step = 0; step < 2; step++) {
                    try (NDManager scoped = trainer.getManager().newSubManager();
                            GradientCollector collector = trainer.newGradientCollector()) {
                        NDArray input = scoped.ones(new Shape(1, WIDTH));
                        NDList output = trainer.forward(new NDList(input));
                        collector.backward(output.singletonOrThrow().sum());
                    }
                    trainer.step();
                }
            }
        }
    }
}
