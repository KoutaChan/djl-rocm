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
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import ai.djl.util.PairList;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TrainerForwardPreparationTest {

    @Test
    public void preparesParameterServerBeforeTrainingForward() throws ReflectiveOperationException {
        Engine engine = Engine.getInstance();
        Device device = Device.cpu();
        List<String> events = new ArrayList<>();
        RecordingBlock block = new RecordingBlock(events);
        RecordingParameterServer parameterServer = new RecordingParameterServer(events);
        DefaultTrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optDevices(new Device[] {device})
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);

        try (Model model =
                Model.newInstance(
                        "forward-parameter-preparation", device, engine.getEngineName())) {
            model.setBlock(block);
            try (Trainer trainer = model.newTrainer(config)) {
                parameterStore(trainer).setParameterServer(parameterServer, new Device[] {device});
                trainer.initialize(new Shape(1));
                events.clear();

                try (NDManager scoped = trainer.getManager().newSubManager();
                        NDArray input = scoped.ones(new Shape(1))) {
                    try (NDList output = trainer.evaluate(new NDList(input))) {
                        Assert.assertEquals(output.singletonOrThrow().getFloat(), 1f);
                    }
                    Assert.assertEquals(events, Arrays.asList("prepare", "forward"));

                    events.clear();
                    try (NDList output = trainer.forward(new NDList(input))) {
                        Assert.assertEquals(output.singletonOrThrow().getFloat(), 1f);
                    }
                    Assert.assertEquals(events, Arrays.asList("prepare", "forward", "backward"));

                    events.clear();
                    try (NDList output = trainer.forward(new NDList(input), new NDList(input))) {
                        Assert.assertEquals(output.singletonOrThrow().getFloat(), 1f);
                    }
                }

                Assert.assertEquals(events, Arrays.asList("prepare", "forward", "backward"));
                Assert.assertEquals(parameterServer.getPrepareCalls(), 3);
            }
        }
        Assert.assertTrue(parameterServer.isClosed());
    }

    private static ParameterStore parameterStore(Trainer trainer)
            throws ReflectiveOperationException {
        Field field = Trainer.class.getDeclaredField("parameterStore");
        field.setAccessible(true);
        return (ParameterStore) field.get(trainer);
    }

    private static final class RecordingParameterServer implements ParameterServer {

        private List<String> events;
        private int initializedParameters;
        private int prepareCalls;
        private boolean closed;

        private RecordingParameterServer(List<String> events) {
            this.events = events;
        }

        @Override
        public void init(String parameterId, NDArray[] value) {
            ++initializedParameters;
        }

        @Override
        public void prepareForForward() {
            Assert.assertEquals(initializedParameters, 1);
            ++prepareCalls;
            events.add("prepare");
        }

        @Override
        public void prepareForBackward(NDList outputs) {
            events.add("backward");
        }

        @Override
        public void update(String parameterId, NDArray[] grads, NDArray[] params) {}

        @Override
        public void close() {
            closed = true;
        }

        private int getPrepareCalls() {
            return prepareCalls;
        }

        private boolean isClosed() {
            return closed;
        }
    }

    private static final class RecordingBlock extends AbstractBlock {

        private static final byte VERSION = 1;

        private List<String> events;
        private Parameter weight;

        private RecordingBlock(List<String> events) {
            super(VERSION);
            this.events = events;
            weight =
                    addParameter(
                            Parameter.builder()
                                    .setName("weight")
                                    .setType(Parameter.Type.WEIGHT)
                                    .optShape(new Shape(1))
                                    .build());
        }

        @Override
        protected NDList forwardInternal(
                ParameterStore parameterStore,
                NDList inputs,
                boolean training,
                PairList<String, Object> params) {
            events.add("forward");
            NDArray input = inputs.singletonOrThrow();
            return new NDList(
                    input.mul(parameterStore.getValue(weight, input.getDevice(), training)));
        }

        @Override
        public Shape[] getOutputShapes(Shape[] inputShapes) {
            return inputShapes;
        }

        @Override
        public void initializeChildBlocks(
                NDManager manager, DataType dataType, Shape... inputShapes) {}
    }
}
