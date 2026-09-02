/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.mxnet.integration;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.mxnet.engine.MxParameterServer;
import ai.djl.mxnet.jna.JnaUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.testing.Assertions;
import ai.djl.testing.TestRequirements;
import ai.djl.training.GradScaler;
import ai.djl.training.ParameterServer;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.ParameterTracker;
import ai.djl.training.tracker.Tracker;

import org.testng.Assert;
import org.testng.annotations.Test;

public class MxParameterStoreTest {

    @Test
    public void testInfiniteDetection() {
        TestRequirements.notArm();

        try (NDManager manager = NDManager.newBaseManager(Device.cpu());
                NDArray values =
                        manager.create(
                                new float[] {
                                    Float.NaN,
                                    Float.POSITIVE_INFINITY,
                                    Float.NEGATIVE_INFINITY,
                                    Float.MAX_VALUE,
                                    0f
                                });
                NDArray infinite = values.isInfinite()) {
            Assert.assertEquals(
                    infinite.toBooleanArray(), new boolean[] {false, true, true, false, false});
        }
    }

    @Test
    public void testParameterStore() {
        TestRequirements.notArm();

        try (Model model = Model.newInstance("model")) {
            NDManager manager = model.getNDManager();
            int numGpus = manager.getEngine().getGpuCount();
            int numDevices;
            if (numGpus > 0) {
                numDevices = numGpus;
            } else {
                numDevices = 4;
            }
            int numWeights = Boolean.getBoolean("nightly") ? 100 : 2;
            int numUpdates = Boolean.getBoolean("nightly") ? 1000 : 10;
            NDArray[][] weights = new NDArray[numWeights][numDevices];
            NDArray[][] grads = new NDArray[numWeights][numDevices];
            NDArray[] expected = new NDArray[numWeights];
            float lr = .1f;
            for (int i = 0; i < numWeights; i++) {
                NDArray w = manager.randomNormal(new Shape(1, 1));
                NDArray g = manager.randomNormal(new Shape(1, 1));
                // simulate aggregate gradient from all device and apply on weight
                expected[i] = w;
                for (int n = 0; n < numUpdates; n++) {
                    expected[i] = updateHelper(expected[i], g, numDevices, lr);
                }
                // copy weight and gradient to all devices
                for (int j = 0; j < numDevices; j++) {
                    Device device;
                    if (numGpus > 0) {
                        device = Device.gpu(j);
                    } else {
                        device = Device.cpu();
                    }
                    weights[i][j] = w.toDevice(device, true);
                    grads[i][j] = g.toDevice(device, true);
                }
            }

            TestOptimizer optimizer =
                    TestOptimizer.builder().setLearningRateTracker(Tracker.fixed(lr)).build();

            try (ParameterServer ps = new MxParameterServer(optimizer)) {

                // init
                for (int i = 0; i < numWeights; i++) {
                    ps.init(String.valueOf(i), new NDArray[] {weights[i][0]});
                }
                for (int n = 0; n < numUpdates; n++) {
                    for (int i = 0; i < numWeights; i++) {
                        ps.update(String.valueOf(i), grads[i], weights[i]);
                    }
                }
                for (int i = 0; i < numWeights; i++) {
                    Assertions.assertAlmostEquals(weights[i][0], expected[i]);
                    // check the number of updates has been invoked
                    Assert.assertEquals(optimizer.updateCount, numWeights * numUpdates);
                }
            }
        }
    }

    @Test
    public void testPreparedGradientsAreNotAggregatedTwice() {
        TestRequirements.notArm();

        TestOptimizer optimizer =
                TestOptimizer.builder().setLearningRateTracker(Tracker.fixed(1f)).build();
        try (NDManager manager = NDManager.newBaseManager(Device.cpu());
                NDArray firstWeight = manager.create(new float[] {10f});
                NDArray secondWeight = manager.create(new float[] {10f});
                NDArray firstGradient = manager.create(new float[] {1f});
                NDArray secondGradient = manager.create(new float[] {2f});
                ParameterServer parameterServer = new MxParameterServer(optimizer)) {
            parameterServer.init("weight", new NDArray[] {firstWeight});
            NDArray[] gradients = {firstGradient, secondGradient};

            Assert.assertTrue(parameterServer.requiresGradientPreparation());
            parameterServer.prepareGradients("weight", gradients);
            Assert.assertEquals(firstGradient.getFloat(), 3f);
            Assert.assertEquals(optimizer.updateCount, 0);

            parameterServer.update("weight", gradients, new NDArray[] {firstWeight, secondWeight});
            parameterServer.finishGradientStep();
            Assert.assertEquals(firstWeight.getFloat(), 13f);
            Assert.assertEquals(secondWeight.getFloat(), 13f);
            Assert.assertEquals(optimizer.updateCount, 1);
        }
    }

    @Test
    public void testFinishGradientStepDiscardsPreparedState() {
        TestRequirements.notArm();

        TestOptimizer optimizer =
                TestOptimizer.builder().setLearningRateTracker(Tracker.fixed(1f)).build();
        try (NDManager manager = NDManager.newBaseManager(Device.cpu());
                NDArray firstWeight = manager.create(new float[] {10f});
                NDArray secondWeight = manager.create(new float[] {10f});
                NDArray firstGradient = manager.create(new float[] {1f});
                NDArray secondGradient = manager.create(new float[] {2f});
                ParameterServer parameterServer = new MxParameterServer(optimizer)) {
            parameterServer.init("weight", new NDArray[] {firstWeight});
            NDArray[] gradients = {firstGradient, secondGradient};

            parameterServer.prepareGradients("weight", gradients);
            parameterServer.finishGradientStep();
            firstGradient.fillI(1f);
            parameterServer.update("weight", gradients, new NDArray[] {firstWeight, secondWeight});
            parameterServer.finishGradientStep();

            Assert.assertEquals(firstWeight.getFloat(), 13f);
            Assert.assertEquals(secondWeight.getFloat(), 13f);
            Assert.assertEquals(optimizer.updateCount, 1);
        }
    }

    @Test
    public void testReductionOverflowSkipsUpdateAndRecovers() {
        TestRequirements.notArm();
        int previousNumpyMode = JnaUtils.isNumpyMode();
        JnaUtils.setNumpyMode(JnaUtils.NumpyMode.THREAD_LOCAL_ON);
        try {
            TestOptimizer optimizer =
                    TestOptimizer.builder().setLearningRateTracker(Tracker.fixed(1f)).build();
            GradScaler scaler = GradScaler.builder().optInitialScale(8f).build();
            try (NDManager manager = NDManager.newBaseManager(Device.cpu());
                    NDArray loss = manager.create(new float[] {1f});
                    NDArray firstWeight = manager.create(new float[] {10f});
                    NDArray secondWeight = manager.create(new float[] {10f});
                    NDArray firstGradient = manager.create(new float[] {Float.MAX_VALUE});
                    NDArray secondGradient = manager.create(new float[] {Float.MAX_VALUE});
                    ParameterServer parameterServer = new MxParameterServer(optimizer)) {
                parameterServer.init("weight", new NDArray[] {firstWeight});
                NDArray[] gradients = {firstGradient, secondGradient};
                NDArray[] weights = {firstWeight, secondWeight};

                boolean finite;
                try {
                    try (NDArray scaledLoss = scaler.scale(loss)) {
                        Assert.assertEquals(scaledLoss.getFloat(), 8f);
                    }
                    parameterServer.prepareGradients("weight", gradients);
                    Assert.assertEquals(firstGradient.getFloat(), Float.POSITIVE_INFINITY);
                    finite = scaler.unscale(new NDList(gradients));
                    if (finite) {
                        parameterServer.update("weight", gradients, weights);
                    }
                    scaler.update();
                } finally {
                    parameterServer.finishGradientStep();
                }

                Assert.assertFalse(finite);
                Assert.assertEquals(firstWeight.getFloat(), 10f);
                Assert.assertEquals(secondWeight.getFloat(), 10f);
                Assert.assertEquals(optimizer.updateCount, 0);
                Assert.assertEquals(scaler.getScale(), 4f);

                firstGradient.fillI(4f);
                secondGradient.fillI(8f);
                try {
                    try (NDArray scaledLoss = scaler.scale(loss)) {
                        Assert.assertEquals(scaledLoss.getFloat(), 4f);
                    }
                    parameterServer.prepareGradients("weight", gradients);
                    finite = scaler.unscale(new NDList(gradients));
                    if (finite) {
                        parameterServer.update("weight", gradients, weights);
                    }
                    scaler.update();
                } finally {
                    parameterServer.finishGradientStep();
                }

                Assert.assertTrue(finite);
                Assert.assertEquals(firstWeight.getFloat(), 13f);
                Assert.assertEquals(secondWeight.getFloat(), 13f);
                Assert.assertEquals(optimizer.updateCount, 1);
            }
        } finally {
            JnaUtils.setNumpyMode(JnaUtils.NumpyMode.values()[previousNumpyMode]);
        }
    }

    private static NDArray updateHelper(NDArray weight, NDArray grad, int numDevices, float lr) {
        return weight.add(grad.mul(numDevices).mul(lr));
    }

    private static class TestOptimizer extends Optimizer {

        private ParameterTracker learningRateTracker;
        int updateCount;

        protected TestOptimizer(TestOptimizer.Builder builder) {
            super(builder);
            learningRateTracker = builder.getLearningRateTracker();
        }

        /** {@inheritDoc} */
        @Override
        public void update(String parameterId, NDArray weight, NDArray grad) {
            weight.addi(
                    grad.mul(learningRateTracker.getNewValue(parameterId, 0))
                            .toDevice(weight.getDevice(), false));
            updateCount++;
        }

        public static Builder builder() {
            return new Builder();
        }

        public static final class Builder extends OptimizerBuilder<Builder> {

            private Tracker learningRateTracker;

            Builder() {}

            public MxParameterStoreTest.TestOptimizer.Builder setLearningRateTracker(
                    Tracker learningRateTracker) {
                this.learningRateTracker = learningRateTracker;
                return this;
            }

            public Tracker getLearningRateTracker() {
                return learningRateTracker;
            }

            /** {@inheritDoc} */
            @Override
            protected MxParameterStoreTest.TestOptimizer.Builder self() {
                return this;
            }

            public MxParameterStoreTest.TestOptimizer build() {
                if (learningRateTracker == null) {
                    throw new IllegalArgumentException("No lrTracker set");
                }
                return new MxParameterStoreTest.TestOptimizer(this);
            }
        }
    }
}
