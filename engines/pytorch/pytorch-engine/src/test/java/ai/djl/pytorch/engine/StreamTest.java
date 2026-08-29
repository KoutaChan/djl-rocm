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
import ai.djl.engine.Engine;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

import java.nio.FloatBuffer;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@SuppressWarnings("try") // Stream scopes affect thread-local state until close().
public class StreamTest {

    @Test
    public void testPinnedTransferBatch() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        Device device = engine.getGpuCount() == 0 ? Device.cpu() : Device.gpu(0);
        try (PtNDManager host = (PtNDManager) engine.newBaseManager();
                PtNDManager manager = (PtNDManager) engine.newBaseManager(device);
                PtPinnedBuffer buffer = host.allocatePinned(8, DataType.FLOAT32);
                PtStream stream = engine.newStream(device)) {
            FloatBuffer data = buffer.getByteBuffer().asFloatBuffer();
            for (int i = 0; i < data.capacity(); ++i) {
                data.put(i, i + 0.5f);
            }

            PtNDArray array;
            PtTransferTicket ticket;
            try (PtTransferBatch batch = stream.newTransferBatch(manager)) {
                array = batch.copy(buffer, new Shape(2, 4));
                ticket = batch.commit();
            }
            try (PtTransferTicket ignored = ticket) {
                ticket.handoffToCurrentStream();
                Assert.assertEquals(
                        array.toFloatArray(),
                        new float[] {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f});
                ticket.synchronize();
                Assert.assertTrue(ticket.isComplete());
            }
        }
    }

    @Test
    public void testStreamCanMoveBetweenThreads() throws InterruptedException, ExecutionException {
        PtEngine engine = (PtEngine) Engine.getInstance();
        try (PtStream stream = engine.newStream(Device.cpu());
                PtStreamScope scope = stream.openScope()) {
            Assert.assertThrows(IllegalStateException.class, stream::openScope);

            ExecutorService executor = Executors.newSingleThreadExecutor();
            try {
                Throwable error =
                        executor.submit(
                                        () -> {
                                            try {
                                                scope.close();
                                                return null;
                                            } catch (Throwable t) {
                                                return t;
                                            }
                                        })
                                .get();
                Assert.assertTrue(error instanceof IllegalStateException);
            } finally {
                executor.shutdownNow();
            }
        }

        PtStream stream = engine.newStream(Device.cpu());
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            executor.submit(
                            () -> {
                                try (PtStreamScope ignored = stream.openScope()) {
                                    // The persistent stream is not bound to its creator thread.
                                }
                            })
                    .get();
        } finally {
            executor.shutdownNow();
            stream.close();
        }
    }

    @Test
    public void testUncommittedBatchReleasesDestination() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        try (PtNDManager manager = (PtNDManager) engine.newBaseManager(Device.cpu());
                PtPinnedBuffer buffer = manager.allocatePinned(4, DataType.FLOAT32);
                PtStream stream = engine.newStream(Device.cpu())) {
            PtTransferBatch batch = stream.newTransferBatch(manager);
            PtNDArray array = batch.copy(buffer, new Shape(4));
            batch.close();

            Assert.assertTrue(array.isReleased());
            Assert.assertThrows(IllegalStateException.class, batch::commit);
        }
    }

    @Test
    public void testAcceleratorEventRequiresRecord() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This event test requires a PyTorch GPU.");
        }
        Device device = Device.gpu(0);
        try (PtEvent event = engine.newEvent(device);
                PtStream stream = engine.newStream(device)) {
            Assert.assertThrows(IllegalStateException.class, event::waitOnCurrentStream);
            try (PtStreamScope ignored = stream.openScope()) {
                event.record();
            }
            event.waitOnCurrentStream();
            event.synchronize();
            Assert.assertTrue(event.isComplete());
        }
    }

    @Test
    public void testHandoffUsesTheTransferDeviceCurrentStream() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() < 2) {
            throw new SkipException("This handoff test requires two PyTorch GPUs.");
        }
        Device transferDevice = Device.gpu(1);
        try (PtNDManager host = (PtNDManager) engine.newBaseManager(Device.cpu());
                PtNDManager manager = (PtNDManager) engine.newBaseManager(transferDevice);
                PtPinnedBuffer buffer = host.allocatePinned(4, DataType.FLOAT32);
                PtStream stream = engine.newStream(transferDevice)) {
            FloatBuffer data = buffer.getByteBuffer().asFloatBuffer();
            data.put(0, 1.0f);
            data.put(1, 2.0f);
            data.put(2, 3.0f);
            data.put(3, 4.0f);

            PtNDArray array;
            PtTransferTicket ticket;
            try (PtTransferBatch batch = stream.newTransferBatch(manager)) {
                array = batch.copy(buffer, new Shape(4));
                ticket = batch.commit();
            }
            try (PtTransferTicket ignored = ticket;
                    PtStreamScope selectedDevice = engine.newStreamScope(Device.gpu(0))) {
                ticket.handoffToCurrentStream();
                Assert.assertEquals(array.toFloatArray(), new float[] {1.0f, 2.0f, 3.0f, 4.0f});
                ticket.synchronize();
            }
        }
    }
}
