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

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.GradientCollector;
import ai.djl.training.GradientCollectorMode;

import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/** Tests the PyTorch {@link GradientCollector} threading contract. */
@SuppressWarnings("try")
public class PtGradientCollectorTest {

    @AfterMethod
    public void resetGradMode() {
        JniUtils.setGradMode(false);
    }

    @Test
    public void gradientCollectorModeIsThreadConfined() {
        Engine engine = Engine.getInstance();
        Assert.assertEquals(
                engine.getGradientCollectorMode(),
                GradientCollectorMode.THREAD_CONFINED,
                "PyTorch gradient collectors must be thread-confined");
    }

    @Test
    public void nestedCollectorsAreRejected() {
        Engine engine = Engine.getInstance();
        try (GradientCollector collector = engine.newGradientCollector()) {
            IllegalStateException exception =
                    Assert.expectThrows(IllegalStateException.class, engine::newGradientCollector);
            Assert.assertTrue(
                    exception.getMessage().contains("Nested PtGradientCollectors"),
                    "Nested collector error must describe the unsupported pattern");
        }
    }

    @Test
    public void collectorRejectsCallsFromAnotherThread() throws Exception {
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager()) {
            GradientCollector collector = engine.newGradientCollector();
            ExecutorService executor = Executors.newSingleThreadExecutor();
            try {
                NDArray input = manager.ones(new Shape(1));
                input.setRequiresGradient(true);
                NDArray loss = input.add(2f).square().sum();

                Future<Throwable> future =
                        executor.submit(
                                () -> {
                                    try {
                                        collector.backward(loss);
                                        return null;
                                    } catch (Throwable t) {
                                        return t;
                                    }
                                });

                Throwable thrown = future.get(30, TimeUnit.SECONDS);
                Assert.assertTrue(
                        thrown instanceof IllegalStateException,
                        "Cross-thread backward must fail fast");
                Assert.assertTrue(
                        thrown.getMessage().contains("thread that created it"),
                        "Cross-thread error must mention owner thread");
            } finally {
                executor.shutdownNow();
                collector.close();
            }
        }
    }

    @Test
    public void collectorsCanRunInParallelOnDifferentThreads() throws Exception {
        Engine engine = Engine.getInstance();
        ExecutorService executor = Executors.newFixedThreadPool(2);
        CountDownLatch ready = new CountDownLatch(2);
        CountDownLatch start = new CountDownLatch(1);
        try {
            Future<Float> first = executor.submit(() -> runBackward(engine, ready, start));
            Future<Float> second = executor.submit(() -> runBackward(engine, ready, start));

            Assert.assertTrue(ready.await(30, TimeUnit.SECONDS), "workers must be ready");
            start.countDown();

            Assert.assertEquals(first.get(30, TimeUnit.SECONDS), 6f, 1e-5f);
            Assert.assertEquals(second.get(30, TimeUnit.SECONDS), 6f, 1e-5f);
        } finally {
            executor.shutdownNow();
        }
    }

    private static float runBackward(Engine engine, CountDownLatch ready, CountDownLatch start)
            throws InterruptedException {
        JniUtils.setGradMode(false);
        ready.countDown();
        Assert.assertTrue(start.await(30, TimeUnit.SECONDS), "workers must start together");
        try (NDManager manager = engine.newBaseManager();
                GradientCollector collector = engine.newGradientCollector()) {
            NDArray input = manager.ones(new Shape(1));
            input.setRequiresGradient(true);
            NDArray loss = input.add(2f).square().sum();
            collector.backward(loss);
            return input.getGradient().getFloat();
        } finally {
            Assert.assertFalse(
                    JniUtils.isGradMode(),
                    "collector must restore grad mode after close on the worker thread");
        }
    }
}
