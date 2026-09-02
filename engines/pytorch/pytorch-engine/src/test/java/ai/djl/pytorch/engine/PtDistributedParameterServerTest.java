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

import ai.djl.ndarray.NDList;
import ai.djl.training.DistributedTrainingConfig;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;

import org.testng.Assert;
import org.testng.annotations.Test;

public class PtDistributedParameterServerTest {

    @Test
    public void rejectsBackwardPreparationBeforeForwardPreparation() {
        DistributedTrainingConfig config =
                DistributedTrainingConfig.builder().optWorldSize(2).build();
        Optimizer optimizer = Optimizer.sgd().setLearningRateTracker(Tracker.fixed(0.1f)).build();

        try (PtDistributedParameterServer parameterServer =
                new PtDistributedParameterServer(optimizer, config)) {
            IllegalStateException exception =
                    Assert.expectThrows(
                            IllegalStateException.class,
                            () -> parameterServer.prepareForBackward(new NDList()));
            Assert.assertTrue(exception.getMessage().contains("prepared before the forward pass"));
        }
    }
}
