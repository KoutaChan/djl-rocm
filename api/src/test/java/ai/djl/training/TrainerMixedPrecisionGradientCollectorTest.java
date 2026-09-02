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
import ai.djl.ndarray.types.DataType;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.loss.Loss;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Proxy;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class TrainerMixedPrecisionGradientCollectorTest {

    @Test
    public void backwardCleanupPreservesFailureAndRestoresAutocast()
            throws ReflectiveOperationException {
        Engine engine = Engine.getInstance();
        Device device = Device.cpu();
        DefaultTrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optDevices(new Device[] {device})
                        .optAutocast(DataType.BFLOAT16);

        try (Model model =
                Model.newInstance("mixed-precision-cleanup", device, engine.getEngineName())) {
            model.setBlock(new SequentialBlock());
            try (Trainer trainer = model.newTrainer(config)) {
                RuntimeException backwardFailure = new RuntimeException("backward failed");
                RuntimeException cleanupFailure = new RuntimeException("cleanup failed");
                RecordingGradientCollector delegate =
                        new RecordingGradientCollector(backwardFailure);
                GradientCollector collector = mixedPrecisionCollector(trainer, delegate);
                AtomicInteger closeCount = new AtomicInteger();
                NDArray converted = convertedTarget(cleanupFailure, closeCount);
                NDArray target = lowPrecisionTarget(converted);

                RuntimeException thrown = null;
                try {
                    collector.backward(target);
                } catch (RuntimeException e) {
                    thrown = e;
                }

                try {
                    Assert.assertSame(thrown, backwardFailure);
                    Assert.assertEquals(thrown.getSuppressed(), new Throwable[] {cleanupFailure});
                    Assert.assertEquals(closeCount.get(), 1);
                    Assert.assertFalse(autocastScopes(collector).isEmpty());
                } finally {
                    collector.close();
                }
                Assert.assertTrue(delegate.isClosed());
            }
        }
    }

    private static GradientCollector mixedPrecisionCollector(
            Trainer trainer, GradientCollector delegate) throws ReflectiveOperationException {
        Class<?> collectorClass =
                Arrays.stream(Trainer.class.getDeclaredClasses())
                        .filter(
                                type ->
                                        "MixedPrecisionGradientCollector"
                                                .equals(type.getSimpleName()))
                        .findFirst()
                        .orElseThrow(AssertionError::new);
        Constructor<?> constructor =
                collectorClass.getDeclaredConstructor(Trainer.class, GradientCollector.class);
        constructor.setAccessible(true);
        return (GradientCollector) constructor.newInstance(trainer, delegate);
    }

    private static List<?> autocastScopes(GradientCollector collector)
            throws ReflectiveOperationException {
        Field field = collector.getClass().getDeclaredField("autocastScopes");
        field.setAccessible(true);
        return (List<?>) field.get(collector);
    }

    private static NDArray lowPrecisionTarget(NDArray converted) {
        return (NDArray)
                Proxy.newProxyInstance(
                        NDArray.class.getClassLoader(),
                        new Class<?>[] {NDArray.class},
                        (proxy, method, arguments) -> {
                            switch (method.getName()) {
                                case "getDataType":
                                    return DataType.FLOAT16;
                                case "toType":
                                    return converted;
                                default:
                                    return objectMethod(proxy, method.getName(), arguments);
                            }
                        });
    }

    private static NDArray convertedTarget(
            RuntimeException cleanupFailure, AtomicInteger closeCount) {
        return (NDArray)
                Proxy.newProxyInstance(
                        NDArray.class.getClassLoader(),
                        new Class<?>[] {NDArray.class},
                        (proxy, method, arguments) -> {
                            if ("close".equals(method.getName())) {
                                closeCount.incrementAndGet();
                                throw cleanupFailure;
                            }
                            return objectMethod(proxy, method.getName(), arguments);
                        });
    }

    private static Object objectMethod(Object proxy, String name, Object[] arguments) {
        switch (name) {
            case "equals":
                return proxy == arguments[0];
            case "hashCode":
                return System.identityHashCode(proxy);
            case "toString":
                return proxy.getClass().getName();
            default:
                throw new AssertionError("Unexpected method: " + name);
        }
    }

    private static final class RecordingGradientCollector implements GradientCollector {

        private final RuntimeException backwardFailure;
        private boolean closed;

        private RecordingGradientCollector(RuntimeException backwardFailure) {
            this.backwardFailure = backwardFailure;
        }

        @Override
        public void backward(NDArray target) {
            throw backwardFailure;
        }

        @Override
        public void zeroGradients() {}

        @Override
        public void close() {
            closed = true;
        }

        private boolean isClosed() {
            return closed;
        }
    }
}
