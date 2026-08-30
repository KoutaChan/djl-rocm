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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

@SuppressWarnings("try") // Stream scopes affect thread-local state until close().
public class StreamTest {

    @Test
    public void pinnedTransferBatch() {
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
                ticket.waitOnStream();
                Assert.assertEquals(
                        array.toFloatArray(),
                        new float[] {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f});
                ticket.synchronize();
                Assert.assertTrue(ticket.isComplete());
            }
        }
    }

    @Test
    public void streamCanMoveBetweenThreads() throws Exception {
        PtEngine engine = (PtEngine) Engine.getInstance();
        try (PtStream stream = engine.newStream(Device.cpu());
                PtStreamScope scope = stream.openScope()) {
            Assert.assertThrows(IllegalStateException.class, stream::openScope);

            ExecutorService executor = Executors.newSingleThreadExecutor();
            try {
                Future<Throwable> future =
                        executor.submit(
                                () -> {
                                    try {
                                        scope.close();
                                        return null;
                                    } catch (Throwable t) {
                                        return t;
                                    }
                                });
                Throwable error = future.get(30, TimeUnit.SECONDS);
                Assert.assertTrue(error instanceof IllegalStateException);
            } finally {
                executor.shutdownNow();
            }
        }

        PtStream stream = engine.newStream(Device.cpu());
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<?> future =
                    executor.submit(
                            () -> {
                                try (PtStreamScope ignored = stream.openScope()) {
                                    // The persistent stream is not bound to its creator thread.
                                }
                            });
            future.get(30, TimeUnit.SECONDS);
        } finally {
            executor.shutdownNow();
            stream.close();
        }
    }

    @Test
    public void uncommittedBatchReleasesDestination() {
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
    public void acceleratorEventRequiresRecord() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This event test requires a PyTorch GPU.");
        }
        Device device = Device.gpu(0);
        try (PtEvent event = engine.newEvent(device);
                PtStream stream = engine.newStream(device)) {
            Assert.assertThrows(IllegalStateException.class, event::waitOnStream);
            try (PtStreamScope ignored = stream.openScope()) {
                event.record();
            }
            event.waitOnStream();
            event.synchronize();
            Assert.assertTrue(event.isComplete());
        }
    }

    @Test
    public void closedEventRejectsOperations() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        PtEvent event = engine.newEvent(Device.cpu());
        event.close();

        Assert.assertThrows(IllegalStateException.class, event::record);
        Assert.assertThrows(IllegalStateException.class, event::waitOnStream);
        Assert.assertThrows(IllegalStateException.class, event::isComplete);
        Assert.assertThrows(IllegalStateException.class, event::synchronize);
    }

    @Test
    public void ticketUsesTheTransferDeviceCurrentStream() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() < 2) {
            throw new SkipException("This transfer stream test requires two PyTorch GPUs.");
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
                ticket.waitOnStream();
                Assert.assertEquals(array.toFloatArray(), new float[] {1.0f, 2.0f, 3.0f, 4.0f});
                ticket.synchronize();
            }
        }
    }

    @Test
    public void currentStreamCopiesOnCpu() {
        verifyCurrentStreamCopies(Device.cpu());
    }

    @Test
    public void currentStreamCopiesOnGpu() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This transfer test requires a PyTorch GPU.");
        }
        verifyCurrentStreamCopies(Device.gpu(0));
    }

    @Test
    public void asynchronousPinnedCopiesOnCpu() {
        verifyAsynchronousPinnedCopies(Device.cpu());
    }

    @Test
    public void asynchronousPinnedCopiesOnGpu() {
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This asynchronous transfer test requires a PyTorch GPU.");
        }
        verifyAsynchronousPinnedCopies(Device.gpu(0));
    }

    private void verifyCurrentStreamCopies(Device device) {
        PtEngine engine = (PtEngine) Engine.getInstance();
        try (PtNDManager host = (PtNDManager) engine.newBaseManager(Device.cpu());
                PtNDManager manager = (PtNDManager) engine.newBaseManager(device);
                PtPinnedBuffer source = host.allocatePinned(4, DataType.FLOAT32);
                PtPinnedBuffer target = host.allocatePinned(4, DataType.FLOAT32);
                PtNDArray array = (PtNDArray) manager.zeros(new Shape(4));
                PtStream stream = engine.newStream(device);
                PtEvent ready = stream.newEvent();
                PtEvent completion = stream.newEvent()) {
            float[] expected = {1.0f, 2.0f, 3.0f, 4.0f};
            source.getByteBuffer().asFloatBuffer().put(expected);

            ready.record();
            try (PtStreamScope ignored = stream.openScope()) {
                ready.waitOnStream();
                array.enqueueCopyFrom(source);
                completion.record();
            }
            completion.synchronize();
            Assert.assertEquals(array.toFloatArray(), expected);

            try (PtStreamScope ignored = stream.openScope()) {
                array.enqueueCopyTo(target);
                completion.record();
            }
            completion.synchronize();
            float[] actual = new float[expected.length];
            target.getByteBuffer().asFloatBuffer().get(actual);
            Assert.assertEquals(actual, expected);
        }
    }

    private void verifyAsynchronousPinnedCopies(Device device) {
        PtEngine engine = (PtEngine) Engine.getInstance();
        try (PtNDManager host = (PtNDManager) engine.newBaseManager(Device.cpu());
                PtNDManager manager = (PtNDManager) engine.newBaseManager(device);
                PtPinnedBuffer source = host.allocatePinned(4, DataType.FLOAT32);
                PtPinnedBuffer target = host.allocatePinned(4, DataType.FLOAT32);
                PtNDArray array = (PtNDArray) manager.zeros(new Shape(4))) {
            float[] expected = {1.0f, 2.0f, 3.0f, 4.0f};
            source.getByteBuffer().asFloatBuffer().put(expected);

            try (PtCopyEvent copy = array.copyFromPinnedBufferAsync(source)) {
                copy.synchronize();
            }
            try (PtCopyEvent copy = array.copyToPinnedBufferAsync(target)) {
                copy.synchronize();
            }

            float[] actual = new float[expected.length];
            target.getByteBuffer().asFloatBuffer().get(actual);
            Assert.assertEquals(actual, expected);
        }
    }
}
