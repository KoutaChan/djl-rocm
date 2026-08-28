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
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;

import org.testng.Assert;
import org.testng.annotations.Test;

public class ParameterStoreGradientFinalizationTest {

    @Test
    public void onlyPreparesGradientsWhenInspectionIsRequested() {
        Engine engine = Engine.getInstance();
        Device device = Device.cpu();
        RecordingParameterServer parameterServer = new RecordingParameterServer();

        try (NDManager manager = engine.newBaseManager(device);
                NDArray array = manager.ones(new Shape(1))) {
            array.setRequiresGradient(true);
            Parameter parameter =
                    Parameter.builder()
                            .setName("weight")
                            .setType(Parameter.Type.WEIGHT)
                            .optArray(array)
                            .build();
            ParameterStore parameterStore = new ParameterStore(manager, false);
            parameterStore.setParameterServer(parameterServer, new Device[] {device});
            parameterStore.getValue(parameter, device, true);

            parameterStore.finalizeGradients(false);
            Assert.assertEquals(parameterServer.getFinalizeCalls(), 1);
            Assert.assertEquals(parameterServer.getPrepareCalls(), 0);

            parameterStore.finalizeGradients();
            parameterStore.finalizeGradients();
            Assert.assertEquals(parameterServer.getFinalizeCalls(), 1);
            Assert.assertEquals(parameterServer.getPrepareCalls(), 1);

            NDList gradients = parameterStore.getGradients();
            try {
                Assert.assertSame(gradients.head(), parameterServer.getPreparedGradients()[0]);
                Assert.assertEquals(gradients.head().getFloat(), 3f);
                parameterStore.updateAllParameters();
                Assert.assertSame(
                        parameterServer.getUpdatedGradients(),
                        parameterServer.getPreparedGradients());
                Assert.assertEquals(parameterServer.getUpdatedGradients()[0].getFloat(), 3f);
            } finally {
                parameterStore.releaseGradientStep();
                gradients.close();
            }
            Assert.assertEquals(parameterServer.getFinishCalls(), 1);
            parameterStore.close();
            Assert.assertTrue(parameterServer.isClosed());
        }
    }

    private static final class RecordingParameterServer implements ParameterServer {

        private int finalizeCalls;
        private int prepareCalls;
        private int finishCalls;
        private boolean finalized;
        private boolean closed;
        private NDArray[] preparedGradients;
        private NDArray[] updatedGradients;

        @Override
        public void init(String parameterId, NDArray[] value) {}

        @Override
        public void update(String parameterId, NDArray[] grads, NDArray[] params) {
            updatedGradients = grads;
        }

        @Override
        public void finalizeGradients() {
            ++finalizeCalls;
            finalized = true;
        }

        @Override
        public boolean requiresGradientPreparation() {
            return true;
        }

        @Override
        public void prepareGradients(String parameterId, NDArray[] gradients) {
            Assert.assertTrue(finalized);
            Assert.assertEquals(gradients.length, 1);
            ++prepareCalls;
            preparedGradients = gradients;
            gradients[0].fillI(3f);
            finalized = false;
        }

        @Override
        public void finishGradientStep() {
            ++finishCalls;
        }

        @Override
        public void close() {
            closed = true;
        }

        private int getFinalizeCalls() {
            return finalizeCalls;
        }

        private int getPrepareCalls() {
            return prepareCalls;
        }

        private int getFinishCalls() {
            return finishCalls;
        }

        private NDArray[] getPreparedGradients() {
            return preparedGradients;
        }

        private NDArray[] getUpdatedGradients() {
            return updatedGradients;
        }

        private boolean isClosed() {
            return closed;
        }
    }
}
