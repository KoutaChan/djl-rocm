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
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public class OptimizerStateTest {

    @Test
    public void testSaveStatePreservesStateArrayName() throws IOException {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            Path path = Files.createTempFile("optimizer-state", ".bin");
            TestOptimizer optimizer = new TestOptimizer();
            TestOptimizer loaded = new TestOptimizer();
            try {
                NDArray state = manager.ones(new Shape(2, 2));
                state.setName("original");
                optimizer.addState("parameter", Device.cpu(), state);

                optimizer.saveState(path);

                Assert.assertEquals(state.getName(), "original");

                loaded.loadState(manager, path);
                NDArray loadedState = loaded.getStateArray("parameter", Device.cpu());
                Assert.assertNotNull(loadedState);
                Assert.assertEquals(loadedState.toFloatArray(), new float[] {1f, 1f, 1f, 1f});
            } finally {
                loaded.closeStates();
                Files.deleteIfExists(path);
            }
        }
    }

    private static final class TestOptimizer extends Optimizer {

        private Map<String, Map<Device, NDArray>> states;

        TestOptimizer() {
            super(Optimizer.sgd());
            states = new ConcurrentHashMap<>();
        }

        /** {@inheritDoc} */
        @Override
        public void update(String parameterId, NDArray weight, NDArray grad) {}

        void addState(String parameterId, Device device, NDArray array) {
            states.computeIfAbsent(parameterId, k -> new ConcurrentHashMap<>()).put(device, array);
        }

        NDArray getStateArray(String parameterId, Device device) {
            return states.get(parameterId).get(device);
        }

        void closeStates() {
            for (Map<Device, NDArray> arrays : states.values()) {
                for (NDArray array : arrays.values()) {
                    array.close();
                }
            }
        }

        /** {@inheritDoc} */
        @Override
        protected Set<String> getStateNames() {
            return stateNames("states");
        }

        /** {@inheritDoc} */
        @Override
        protected Map<String, Map<Device, NDArray>> getState(String stateName) {
            if ("states".equals(stateName)) {
                return states;
            }
            throw new IllegalArgumentException("Unknown state: " + stateName);
        }

        /** {@inheritDoc} */
        @Override
        protected void setState(String stateName, Map<String, Map<Device, NDArray>> state) {
            if ("states".equals(stateName)) {
                states = new ConcurrentHashMap<>(state);
                return;
            }
            throw new IllegalArgumentException("Unknown state: " + stateName);
        }
    }
}
