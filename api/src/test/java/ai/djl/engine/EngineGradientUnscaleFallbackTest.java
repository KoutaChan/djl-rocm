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
package ai.djl.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Proxy;
import java.util.concurrent.atomic.AtomicInteger;

public class EngineGradientUnscaleFallbackTest {

    @Test
    public void sparseGradientIsRejectedBeforeDenseMutation() {
        Engine engine = new FallbackEngine("fallback");
        AtomicInteger mutations = new AtomicInteger();
        NDArray dense = gradient(engine, DataType.FLOAT32, false, mutations);
        NDArray sparse = gradient(engine, DataType.FLOAT32, true, mutations);

        Assert.expectThrows(
                UnsupportedOperationException.class,
                () -> engine.unscaleGradientsAndCheckFinite(new NDList(dense, sparse), 0.5f));
        Assert.assertEquals(mutations.get(), 0);
    }

    @Test
    public void foreignEngineGradientIsRejectedBeforeDenseMutation() {
        Engine engine = new FallbackEngine("fallback");
        Engine foreignEngine = new FallbackEngine("foreign");
        AtomicInteger mutations = new AtomicInteger();
        NDArray dense = gradient(engine, DataType.FLOAT32, false, mutations);
        NDArray foreign = gradient(foreignEngine, DataType.FLOAT32, false, mutations);

        Assert.expectThrows(
                IllegalArgumentException.class,
                () -> engine.unscaleGradientsAndCheckFinite(new NDList(dense, foreign), 0.5f));
        Assert.assertEquals(mutations.get(), 0);
    }

    @Test
    public void nonFloatingGradientIsRejectedBeforeDenseMutation() {
        Engine engine = new FallbackEngine("fallback");
        AtomicInteger mutations = new AtomicInteger();
        NDArray dense = gradient(engine, DataType.FLOAT32, false, mutations);
        NDArray integer = gradient(engine, DataType.INT32, false, mutations);

        Assert.expectThrows(
                IllegalArgumentException.class,
                () -> engine.unscaleGradientsAndCheckFinite(new NDList(dense, integer), 0.5f));
        Assert.assertEquals(mutations.get(), 0);
    }

    private static NDArray gradient(
            Engine engine, DataType dataType, boolean sparse, AtomicInteger mutations) {
        NDManager manager =
                (NDManager)
                        Proxy.newProxyInstance(
                                NDManager.class.getClassLoader(),
                                new Class<?>[] {NDManager.class},
                                (proxy, method, arguments) -> {
                                    if ("getEngine".equals(method.getName())) {
                                        return engine;
                                    }
                                    return objectMethod(proxy, method.getName(), arguments);
                                });
        return (NDArray)
                Proxy.newProxyInstance(
                        NDArray.class.getClassLoader(),
                        new Class<?>[] {NDArray.class},
                        (proxy, method, arguments) -> {
                            switch (method.getName()) {
                                case "getManager":
                                    return manager;
                                case "getDataType":
                                    return dataType;
                                case "isSparse":
                                    return sparse;
                                case "muli":
                                    mutations.incrementAndGet();
                                    return proxy;
                                default:
                                    return objectMethod(proxy, method.getName(), arguments);
                            }
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

    private static final class FallbackEngine extends Engine {

        private final String name;

        private FallbackEngine(String name) {
            this.name = name;
        }

        @Override
        public Engine getAlternativeEngine() {
            return this;
        }

        @Override
        public String getEngineName() {
            return name;
        }

        @Override
        public int getRank() {
            return Integer.MAX_VALUE;
        }

        @Override
        public String getVersion() {
            return "test";
        }

        @Override
        public boolean hasCapability(String capability) {
            return false;
        }

        @Override
        public Model newModel(String modelName, Device device) {
            throw new UnsupportedOperationException();
        }

        @Override
        public NDManager newBaseManager() {
            throw new UnsupportedOperationException();
        }

        @Override
        public NDManager newBaseManager(Device device) {
            throw new UnsupportedOperationException();
        }
    }
}
