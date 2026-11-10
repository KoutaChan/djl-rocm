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
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.tracker.Tracker;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

public class OptimizerStateTest {

    private static final float[] WEIGHT = {1f, -2f, 3f, 0.5f};
    private static final float[] GRADIENT = {0.5f, 0.25f, -1f, 2f};

    @Test
    public void testCloseReleasesStateAndStartsOver() {
        List<Supplier<Optimizer>> optimizers =
                Arrays.asList(
                        () ->
                                Optimizer.sgd()
                                        .setLearningRateTracker(Tracker.fixed(0.1f))
                                        .optMomentum(0.9f)
                                        .build(),
                        () -> Optimizer.adam().build(),
                        () -> Optimizer.adamW().build());
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            for (Supplier<Optimizer> supplier : optimizers) {
                Optimizer optimizer = supplier.get();
                NDArray weight = manager.create(WEIGHT);
                update(manager, optimizer, weight);
                update(manager, optimizer, weight);
                List<NDArray> states = stateArrays(optimizer);
                Assert.assertFalse(states.isEmpty());

                optimizer.close();
                for (NDArray state : states) {
                    Assert.assertTrue(state.isReleased());
                }
                Assert.assertTrue(stateArrays(optimizer).isEmpty());

                NDArray reused = manager.create(WEIGHT);
                NDArray fresh = manager.create(WEIGHT);
                try (Optimizer newOptimizer = supplier.get()) {
                    update(manager, optimizer, reused);
                    update(manager, newOptimizer, fresh);
                    assertBits(reused.toFloatArray(), fresh.toFloatArray());
                } finally {
                    optimizer.close();
                }
            }
        }
    }

    @Test
    public void testTrainerCloseReleasesOptimizerState() {
        Device device = Device.cpu();
        Linear block = Linear.builder().setUnits(2).build();
        Optimizer optimizer = Optimizer.adamW().build();
        DefaultTrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optDevices(new Device[] {device})
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)
                        .optOptimizer(optimizer);
        List<NDArray> states;
        try (Model model = Model.newInstance("optimizer-state", device)) {
            model.setBlock(block);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(1, 3));
                for (int step = 0; step < 2; step++) {
                    try (NDManager scoped = trainer.getManager().newSubManager();
                            GradientCollector collector = trainer.newGradientCollector()) {
                        NDArray input = scoped.create(new float[] {1f, -1f, 2f}, new Shape(1, 3));
                        NDList output = trainer.forward(new NDList(input));
                        collector.backward(output.singletonOrThrow().sum());
                    }
                    trainer.step();
                }
                states = stateArrays(optimizer);
                Assert.assertFalse(states.isEmpty());
            }
        }
        for (NDArray state : states) {
            Assert.assertTrue(state.isReleased());
        }
        Assert.assertTrue(stateArrays(optimizer).isEmpty());
    }

    @Test
    public void testLoadStateReleasesReplacedState() throws IOException {
        Path path = Files.createTempFile("optimizer-state", ".bin");
        try (NDManager manager = NDManager.newBaseManager(Device.cpu());
                Optimizer optimizer = Optimizer.adam().build()) {
            NDArray weight = manager.create(WEIGHT);
            update(manager, optimizer, weight);
            optimizer.saveState(path);
            List<NDArray> replaced = stateArrays(optimizer);

            optimizer.loadState(manager, path);
            for (NDArray state : replaced) {
                Assert.assertTrue(state.isReleased());
            }
            for (NDArray state : stateArrays(optimizer)) {
                Assert.assertFalse(state.isReleased());
            }
        } finally {
            Files.deleteIfExists(path);
        }
    }

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

    private static void update(NDManager manager, Optimizer optimizer, NDArray weight) {
        try (NDArray gradient = manager.create(GRADIENT)) {
            optimizer.update("weight", weight, gradient);
        }
    }

    private static List<NDArray> stateArrays(Optimizer optimizer) {
        List<NDArray> arrays = new ArrayList<>();
        for (String stateName : optimizer.getStateNames()) {
            for (Map<Device, NDArray> state : optimizer.getState(stateName).values()) {
                arrays.addAll(state.values());
            }
        }
        return arrays;
    }

    private static void assertBits(float[] actual, float[] expected) {
        Assert.assertEquals(actual.length, expected.length);
        for (int i = 0; i < actual.length; i++) {
            Assert.assertEquals(
                    Float.floatToRawIntBits(actual[i]),
                    Float.floatToRawIntBits(expected[i]),
                    "element " + i);
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
