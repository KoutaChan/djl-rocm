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
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.core.Embedding;
import ai.djl.training.GradientCollector;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.lang.ref.PhantomReference;
import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

/** Tests that tensors and the Java memory they share stay valid while either side uses them. */
public class PtNDManagerTest {

    private static final int ROWS = 16;
    private static final int WIDTH = 4;
    private static final int SIZE = 4096;
    private static final int BUFFERS = 1024;
    private static final int THREADS = 4;
    private static final int VIEWS = 8;

    @Test
    public void closedIndexArrayKeepsItsDataForBackward() throws InterruptedException {
        int[] indices = indices();
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(Device.cpu())) {
            NDArray weight = manager.zeros(new Shape(ROWS, WIDTH));
            weight.setRequiresGradient(true);
            try (GradientCollector collector = engine.newGradientCollector()) {
                // The embedding backward still reads the index array that the scope closed.
                NDArray output = embed(manager, weight, IntBuffer.wrap(indices));
                List<ByteBuffer> reused = reuseReleasedMemory(SIZE * Integer.BYTES);
                collector.backward(output.sum());
                Reference.reachabilityFence(reused);
            }
            Assert.assertEquals(weight.getGradient().toFloatArray(), gradient(indices));
        }
    }

    @Test
    public void directIndexArrayKeepsItsBufferForBackward() throws InterruptedException {
        int[] indices = indices();
        ReferenceQueue<ByteBuffer> queue = new ReferenceQueue<>();
        List<Reference<ByteBuffer>> markers = new ArrayList<>();
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(Device.cpu())) {
            NDArray weight = manager.zeros(new Shape(ROWS, WIDTH));
            weight.setRequiresGradient(true);
            try (GradientCollector collector = engine.newGradientCollector()) {
                // Only the tensor saved for backward keeps the direct buffer reachable.
                NDArray output = embed(manager, weight, tracked(indices, queue, markers));
                List<ByteBuffer> reused = reuseReleasedMemory(SIZE * Integer.BYTES);
                Assert.assertNull(queue.poll(), "backward still reads the buffer");
                collector.backward(output.sum());
                Reference.reachabilityFence(reused);
            }
            Assert.assertEquals(weight.getGradient().toFloatArray(), gradient(indices));
        }
    }

    @Test
    public void viewKeepsTheDataOfAClosedArray() throws InterruptedException {
        float[] data = values(0);
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            for (Buffer buffer : copiedBuffers(data)) {
                NDArray view = closedView(manager, buffer);
                List<ByteBuffer> reused = reuseReleasedMemory(SIZE * Float.BYTES);
                Assert.assertEquals(view.toFloatArray(), data);
                Reference.reachabilityFence(reused);
            }
        }
    }

    @Test
    public void viewKeepsTheDirectBufferOfAClosedArray() throws InterruptedException {
        float[] data = values(0);
        ReferenceQueue<ByteBuffer> queue = new ReferenceQueue<>();
        List<Reference<ByteBuffer>> markers = new ArrayList<>();
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            NDArray view = closedView(manager, tracked(data, queue, markers));
            List<ByteBuffer> reused = reuseReleasedMemory(SIZE * Float.BYTES);
            Assert.assertNull(queue.poll(), "the view still reads the buffer");
            Assert.assertEquals(view.toFloatArray(), data);
            Reference.reachabilityFence(reused);
        }
    }

    @Test
    public void setKeepsTheBufferOfEveryStorage() throws InterruptedException {
        float[] first = values(0);
        float[] second = values(SIZE);
        ReferenceQueue<ByteBuffer> queue = new ReferenceQueue<>();
        List<Reference<ByteBuffer>> markers = new ArrayList<>();
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            NDArray firstView;
            NDArray secondView;
            try (NDManager scope = manager.newSubManager()) {
                NDArray array = scope.zeros(new Shape(SIZE));
                array.set(tracked(first, queue, markers));
                firstView = array.reshape(64, 64);
                // The second set replaces the storage that the first view still reads.
                array.set(tracked(second, queue, markers));
                secondView = array.reshape(64, 64);
                firstView.attach(manager);
                secondView.attach(manager);
            }
            List<ByteBuffer> reused = reuseReleasedMemory(SIZE * Float.BYTES);
            Assert.assertNull(queue.poll(), "a view still reads the buffer");
            Assert.assertEquals(firstView.toFloatArray(), first);
            Assert.assertEquals(secondView.toFloatArray(), second);
            Reference.reachabilityFence(reused);
        }
    }

    @Test
    public void setCopiesHeapDataIntoLibtorchMemory() throws InterruptedException {
        float[] data = values(0);
        int[] indices = indices();
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(Device.cpu())) {
            for (Buffer buffer : copiedBuffers(data)) {
                NDArray view;
                try (NDManager scope = manager.newSubManager()) {
                    NDArray array = scope.zeros(new Shape(SIZE));
                    array.set(buffer);
                    view = array.reshape(64, 64);
                    view.attach(manager);
                }
                List<ByteBuffer> reused = reuseReleasedMemory(SIZE * Float.BYTES);
                Assert.assertEquals(view.toFloatArray(), data);
                Reference.reachabilityFence(reused);
            }

            NDArray weight = manager.zeros(new Shape(ROWS, WIDTH));
            weight.setRequiresGradient(true);
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output =
                        embed(
                                manager,
                                weight,
                                scope -> {
                                    NDArray index = scope.zeros(new Shape(SIZE), DataType.INT32);
                                    index.set(IntBuffer.wrap(indices));
                                    return index;
                                });
                List<ByteBuffer> reused = reuseReleasedMemory(SIZE * Integer.BYTES);
                collector.backward(output.sum());
                Reference.reachabilityFence(reused);
            }
            Assert.assertEquals(weight.getGradient().toFloatArray(), gradient(indices));
        }
    }

    @Test
    public void directBufferStaysShared() {
        float[] data = values(0);
        ByteBuffer buffer = ByteBuffer.allocateDirect(SIZE * Float.BYTES);
        buffer.order(ByteOrder.nativeOrder());
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            NDArray created = manager.create(buffer, new Shape(SIZE), DataType.FLOAT32);
            NDArray replaced = manager.zeros(new Shape(SIZE));
            replaced.set(buffer);
            // Writes after the arrays took the buffer are visible without a copy.
            buffer.asFloatBuffer().put(data);
            Assert.assertEquals(created.toFloatArray(), data);
            Assert.assertEquals(replaced.toFloatArray(), data);
        }
    }

    @Test
    public void byteBufferStaysValidAfterClose() {
        float[] data = values(0);
        ByteBuffer bytes;
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            bytes = manager.create(data).toByteBuffer(true);
        }
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            // Arrays of the same size reuse the memory that libtorch released.
            List<NDArray> reused = new ArrayList<>(BUFFERS);
            for (int i = 0; i < BUFFERS; i++) {
                reused.add(manager.full(new Shape(SIZE), -1f));
            }
            float[] actual = new float[SIZE];
            bytes.asFloatBuffer().get(actual);
            Assert.assertEquals(actual, data);
            Reference.reachabilityFence(reused);
        }
    }

    @Test
    public void wrappedBufferIsReleasedWithItsLastTensor() throws InterruptedException {
        Engine engine = Engine.getInstance();
        ReferenceQueue<ByteBuffer> queue = new ReferenceQueue<>();
        List<Reference<ByteBuffer>> markers = new ArrayList<>();
        try (NDManager manager = engine.newBaseManager(Device.cpu())) {
            NDArray view = closedView(manager, tracked(values(0), queue, markers));
            awaitGarbageCollection();
            Assert.assertNull(queue.poll(), "a view still reads the buffer");
            view.close();
            Assert.assertSame(awaitRelease(queue), markers.get(0));

            NDArray weight = manager.zeros(new Shape(ROWS, WIDTH));
            weight.setRequiresGradient(true);
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output = embed(manager, weight, tracked(indices(), queue, markers));
                awaitGarbageCollection();
                Assert.assertNull(queue.poll(), "backward still reads the buffer");
                collector.backward(output.sum());
                output.close();
            }
            Assert.assertSame(awaitRelease(queue), markers.get(1));

            NDArray array = manager.zeros(new Shape(SIZE));
            array.set(tracked(values(0), queue, markers));
            awaitGarbageCollection();
            Assert.assertNull(queue.poll(), "the array still reads the buffer");
            array.close();
            Assert.assertSame(awaitRelease(queue), markers.get(2));
        }
    }

    @Test
    public void sparseArrayKeepsItsDirectValues() throws InterruptedException {
        byte[] values = new byte[SIZE];
        long[][] indices = new long[2][SIZE];
        byte[] expected = new byte[SIZE * WIDTH];
        for (int i = 0; i < SIZE; i++) {
            values[i] = (byte) (i % 127 + 1);
            indices[0][i] = i;
            indices[1][i] = i % WIDTH;
            expected[i * WIDTH + i % WIDTH] = values[i];
        }
        ReferenceQueue<ByteBuffer> queue = new ReferenceQueue<>();
        List<Reference<ByteBuffer>> markers = new ArrayList<>();
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            // createCoo closes the array that shares the values before it returns.
            NDArray sparse =
                    manager.createCoo(
                            tracked(values, queue, markers), indices, new Shape(SIZE, WIDTH));
            List<ByteBuffer> reused = reuseReleasedMemory(SIZE);
            Assert.assertNull(queue.poll(), "the sparse array still reads the buffer");
            Assert.assertEquals(sparse.toDense().toByteArray(), expected);
            Reference.reachabilityFence(reused);
            sparse.close();
            Assert.assertSame(awaitRelease(queue), markers.get(0));
        }
    }

    @Test
    public void concurrentReleasesKeepEveryBuffer()
            throws InterruptedException, ExecutionException {
        ExecutorService executor = Executors.newFixedThreadPool(THREADS);
        try {
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
            List<Future<Integer>> workers = new ArrayList<>(THREADS);
            for (int i = 0; i < THREADS; i++) {
                workers.add(executor.submit(() -> countDifferingViews(deadline)));
            }
            for (Future<Integer> worker : workers) {
                Assert.assertEquals(worker.get().intValue(), 0);
            }
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    public void savedScalarIsReleasedOnTheAutogradDeviceThread() throws InterruptedException {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new SkipException("This release test requires a GPU.");
        }
        float[] expected = new float[SIZE];
        Arrays.fill(expected, 3f);
        ReferenceQueue<ByteBuffer> queue = new ReferenceQueue<>();
        List<Reference<ByteBuffer>> markers = new ArrayList<>();
        try (NDManager cpu = engine.newBaseManager(Device.cpu());
                NDManager gpu = engine.newBaseManager(Device.gpu())) {
            NDArray weight = gpu.ones(new Shape(SIZE));
            weight.setRequiresGradient(true);
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output;
                try (NDManager scope = cpu.newSubManager()) {
                    // Only the GPU graph keeps the scalar; backward frees it on a device thread.
                    NDArray scalar =
                            scope.create(
                                    tracked(new float[] {3f}, queue, markers),
                                    new Shape(),
                                    DataType.FLOAT32);
                    output = weight.mul(scalar).sum();
                    output.attach(gpu);
                }
                awaitGarbageCollection();
                Assert.assertNull(queue.poll(), "backward still reads the buffer");
                collector.backward(output);
                output.close();
            }
            Assert.assertEquals(weight.getGradient().toFloatArray(), expected);

            // A device thread cannot delete the reference, so the buffer waits for a Java thread.
            awaitGarbageCollection();
            Assert.assertNull(queue.poll(), "only a Java thread deletes the reference");
            cpu.create(directBuffer(new float[WIDTH]), new Shape(WIDTH), DataType.FLOAT32).close();
            Assert.assertSame(awaitRelease(queue), markers.get(0));
        }
    }

    @Test(dataProvider = "partialBuffers")
    public void onlyTheRemainingDataIsCopied(Buffer buffer, DataType dataType) {
        double[] expected = {1, 2, 3, 4};
        Shape shape = new Shape(expected.length);
        buffer.mark();
        for (Device device : devices()) {
            try (NDManager manager = NDManager.newBaseManager(device)) {
                NDArray created = manager.create(buffer, shape, dataType);
                Assert.assertEquals(buffer.remaining(), 0, "the data was not consumed");
                buffer.reset();
                NDArray replaced = manager.zeros(shape, dataType);
                replaced.set(buffer);
                Assert.assertEquals(buffer.remaining(), 0, "the data was not consumed");
                buffer.reset();
                for (NDArray array : new NDArray[] {created, replaced}) {
                    Assert.assertEquals(array.getDataType(), dataType);
                    Assert.assertEquals(
                            array.toType(DataType.FLOAT64, false).toDoubleArray(), expected);
                }
            }
        }
    }

    @Test
    public void emptyDataCreatesEmptyArrays() {
        Shape shape = new Shape(0);
        for (Device device : devices()) {
            try (NDManager manager = NDManager.newBaseManager(device)) {
                Buffer[] buffers = {
                    FloatBuffer.wrap(new float[0]),
                    directBuffer(new float[0]).asFloatBuffer(),
                    directBuffer(new float[0])
                };
                for (Buffer buffer : buffers) {
                    NDArray created = manager.create(buffer, shape, DataType.FLOAT32);
                    Assert.assertEquals(created.getShape(), shape);
                    Assert.assertEquals(created.toFloatArray().length, 0);
                    NDArray replaced = manager.zeros(shape);
                    replaced.set(buffer);
                    Assert.assertEquals(replaced.toFloatArray().length, 0);
                }
            }
        }
    }

    @Test
    public void readersReturnTheDataOfEveryLayout() {
        for (Device device : devices()) {
            try (NDManager manager = NDManager.newBaseManager(device)) {
                double[] doubles = {1.5, -2.5};
                float[] floats = {1.5f, -2.5f, 3.5f, -4.5f};
                short[] shorts = {1, -2};
                int[] ints = {1, -2};
                long[] longs = {1L << 40, -2L};
                boolean[] booleans = {true, false};
                Assert.assertEquals(manager.create(doubles).toDoubleArray(), doubles);
                Assert.assertEquals(manager.create(floats).toFloatArray(), floats);
                Assert.assertEquals(
                        manager.create(ShortBuffer.wrap(shorts), new Shape(2), DataType.INT16)
                                .toShortArray(),
                        shorts);
                Assert.assertEquals(manager.create(ints).toIntArray(), ints);
                Assert.assertEquals(manager.create(longs).toLongArray(), longs);
                Assert.assertEquals(manager.create(booleans).toBooleanArray(), booleans);
                Assert.assertEquals(
                        manager.create(
                                        ByteBuffer.wrap(new byte[] {-1, 1}),
                                        new Shape(2),
                                        DataType.UINT8)
                                .toUint8Array(),
                        new int[] {255, 1});

                NDArray matrix = manager.create(floats, new Shape(2, 2));
                // A contiguous view that starts inside the storage, and one that is not contiguous.
                Assert.assertEquals(matrix.get(1).toFloatArray(), new float[] {3.5f, -4.5f});
                Assert.assertEquals(
                        matrix.transpose().toFloatArray(), new float[] {1.5f, 3.5f, -2.5f, -4.5f});
            }
        }
    }

    /** Returns buffers of 1, 2, 3, 4 after other data, with an offset into any backing array. */
    @DataProvider
    public Object[][] partialBuffers() {
        float[] floats = {9, 9, 9, 9, 1, 2, 3, 4, 9};
        ByteBuffer floatBytes = directBuffer(floats);
        byte[] rawFloats = new byte[floatBytes.remaining()];
        floatBytes.get(rawFloats);
        Buffer rawHeap =
                ByteBuffer.wrap(rawFloats, Float.BYTES, floats.length * Float.BYTES - Float.BYTES)
                        .slice()
                        .order(ByteOrder.nativeOrder())
                        .position(3 * Float.BYTES)
                        .limit(7 * Float.BYTES);
        return new Object[][] {
            {
                ByteBuffer.wrap(new byte[] {9, 9, 9, 9, 1, 2, 3, 4, 9}, 1, 7)
                        .slice()
                        .position(3)
                        .limit(7),
                DataType.INT8
            },
            {rawHeap, DataType.FLOAT32},
            {
                ShortBuffer.wrap(new short[] {9, 9, 9, 9, 1, 2, 3, 4, 9}, 1, 7)
                        .slice()
                        .position(3)
                        .limit(7),
                DataType.INT16
            },
            {
                IntBuffer.wrap(new int[] {9, 9, 9, 9, 1, 2, 3, 4, 9}, 1, 7)
                        .slice()
                        .position(3)
                        .limit(7),
                DataType.INT32
            },
            {
                LongBuffer.wrap(new long[] {9, 9, 9, 9, 1, 2, 3, 4, 9}, 1, 7)
                        .slice()
                        .position(3)
                        .limit(7),
                DataType.INT64
            },
            {FloatBuffer.wrap(floats, 1, 7).slice().position(3).limit(7), DataType.FLOAT32},
            {
                DoubleBuffer.wrap(new double[] {9, 9, 9, 9, 1, 2, 3, 4, 9}, 1, 7)
                        .slice()
                        .position(3)
                        .limit(7),
                DataType.FLOAT64
            },
            {
                FloatBuffer.wrap(floats, 1, 7).asReadOnlyBuffer().position(4).limit(8),
                DataType.FLOAT32
            },
            {
                ByteBuffer.wrap(rawFloats)
                        .order(ByteOrder.nativeOrder())
                        .asFloatBuffer()
                        .position(4)
                        .limit(8),
                DataType.FLOAT32
            },
        };
    }

    /** Returns the devices to test: the CPU and the first GPU if there is one. */
    private static Device[] devices() {
        return Engine.getInstance().getGpuCount() > 0
                ? new Device[] {Device.cpu(), Device.gpu()}
                : new Device[] {Device.cpu()};
    }

    /**
     * Creates views whose buffers only libtorch keeps reachable, and returns how many of them read
     * other data after the released memory has been reused.
     */
    private static int countDifferingViews(long deadline) throws InterruptedException {
        int differing = 0;
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            for (int round = 0; System.nanoTime() < deadline; round++) {
                float[] data = values(round);
                List<NDArray> views = new ArrayList<>(VIEWS);
                for (int i = 0; i < VIEWS; i++) {
                    views.add(closedView(manager, directBuffer(data)));
                }
                List<ByteBuffer> reused = reuseReleasedMemory(SIZE * Float.BYTES);
                for (NDArray view : views) {
                    if (!Arrays.equals(view.toFloatArray(), data)) {
                        differing++;
                    }
                    view.close();
                }
                Reference.reachabilityFence(reused);
            }
        }
        return differing;
    }

    /** Looks up rows with an index array that a closed scope created from the data. */
    private static NDArray embed(NDManager manager, NDArray weight, Buffer indices) {
        return embed(
                manager, weight, scope -> scope.create(indices, new Shape(SIZE), DataType.INT32));
    }

    /** Looks up rows with an index array that a closed scope created. */
    private static NDArray embed(
            NDManager manager, NDArray weight, Function<NDManager, NDArray> indices) {
        try (NDManager scope = manager.newSubManager()) {
            NDArray output =
                    Embedding.embedding(indices.apply(scope), weight, SparseFormat.DENSE)
                            .singletonOrThrow();
            output.attach(manager);
            return output;
        }
    }

    /** Returns a view of an array that a closed scope created from the data. */
    private static NDArray closedView(NDManager manager, Buffer data) {
        try (NDManager scope = manager.newSubManager()) {
            NDArray view = scope.create(data, new Shape(SIZE), DataType.FLOAT32).reshape(64, 64);
            view.attach(manager);
            return view;
        }
    }

    private static int[] indices() {
        int[] indices = new int[SIZE];
        for (int i = 0; i < SIZE; i++) {
            indices[i] = (i * 7 + 3) % ROWS;
        }
        return indices;
    }

    /** Returns the gradient of the summed embedding with respect to its weight. */
    private static float[] gradient(int[] indices) {
        float[] gradient = new float[ROWS * WIDTH];
        for (int index : indices) {
            for (int j = 0; j < WIDTH; j++) {
                gradient[index * WIDTH + j] += 1;
            }
        }
        return gradient;
    }

    private static float[] values(int start) {
        float[] values = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            values[i] = start + i;
        }
        return values;
    }

    /** Returns buffers that arrays copy: one with an array and one that only a view can read. */
    private static Buffer[] copiedBuffers(float[] data) {
        return new Buffer[] {FloatBuffer.wrap(data), directBuffer(data).asFloatBuffer()};
    }

    private static ByteBuffer directBuffer(float[] data) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(data.length * Float.BYTES);
        buffer.order(ByteOrder.nativeOrder()).asFloatBuffer().put(data);
        return buffer;
    }

    private static ByteBuffer directBuffer(int[] data) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(data.length * Integer.BYTES);
        buffer.order(ByteOrder.nativeOrder()).asIntBuffer().put(data);
        return buffer;
    }

    private static ByteBuffer directBuffer(byte[] data) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(data.length);
        buffer.put(data).flip();
        return buffer;
    }

    /** Returns a direct buffer with the data whose release the queue reports. */
    private static ByteBuffer tracked(
            byte[] data, ReferenceQueue<ByteBuffer> queue, List<Reference<ByteBuffer>> markers) {
        ByteBuffer buffer = directBuffer(data);
        markers.add(new PhantomReference<>(buffer, queue));
        return buffer;
    }

    /** Returns a direct buffer with the data whose release the queue reports. */
    private static ByteBuffer tracked(
            float[] data, ReferenceQueue<ByteBuffer> queue, List<Reference<ByteBuffer>> markers) {
        ByteBuffer buffer = directBuffer(data);
        markers.add(new PhantomReference<>(buffer, queue));
        return buffer;
    }

    /** Returns a direct buffer with the indices whose release the queue reports. */
    private static ByteBuffer tracked(
            int[] data, ReferenceQueue<ByteBuffer> queue, List<Reference<ByteBuffer>> markers) {
        ByteBuffer buffer = directBuffer(data);
        markers.add(new PhantomReference<>(buffer, queue));
        return buffer;
    }

    /** Waits until the garbage collector has processed every unreachable object. */
    private static void awaitGarbageCollection() throws InterruptedException {
        ReferenceQueue<Object> queue = new ReferenceQueue<>();
        // The second marker is enqueued only after all references from the first collection.
        for (int i = 0; i < 2; i++) {
            PhantomReference<Object> marker = new PhantomReference<>(new Object(), queue);
            System.gc();
            Assert.assertSame(queue.remove(30_000), marker, "garbage collection did not finish");
        }
    }

    /** Collects garbage until the queue reports a released buffer. */
    private static Reference<? extends ByteBuffer> awaitRelease(ReferenceQueue<ByteBuffer> queue)
            throws InterruptedException {
        for (int i = 0; i < 300; i++) {
            System.gc();
            Reference<? extends ByteBuffer> released = queue.remove(100);
            if (released != null) {
                return released;
            }
        }
        throw new AssertionError("the buffer was not released");
    }

    /**
     * Waits until the garbage collector has released unreachable direct buffers, then allocates
     * cleared buffers of the same size so that the released memory is reused.
     */
    private static List<ByteBuffer> reuseReleasedMemory(int bytes) throws InterruptedException {
        awaitGarbageCollection();
        List<ByteBuffer> buffers = new ArrayList<>(BUFFERS);
        for (int i = 0; i < BUFFERS; i++) {
            buffers.add(ByteBuffer.allocateDirect(bytes));
        }
        return buffers;
    }
}
