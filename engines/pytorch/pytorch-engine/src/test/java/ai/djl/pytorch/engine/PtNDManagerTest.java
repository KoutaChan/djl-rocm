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
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.core.Embedding;
import ai.djl.training.GradientCollector;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.ref.PhantomReference;
import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

/** Tests that arrays created from Java data stay valid while libtorch still uses their tensors. */
public class PtNDManagerTest {

    private static final int ROWS = 16;
    private static final int WIDTH = 4;
    private static final int SIZE = 4096;
    private static final int BUFFERS = 64;

    @Test
    public void closedIndexArrayKeepsItsDataForBackward() throws InterruptedException {
        int[] indices = new int[SIZE];
        float[] expected = new float[ROWS * WIDTH];
        for (int i = 0; i < SIZE; i++) {
            indices[i] = (i * 7 + 3) % ROWS;
            for (int j = 0; j < WIDTH; j++) {
                expected[indices[i] * WIDTH + j] += 1;
            }
        }
        Engine engine = Engine.getInstance();
        try (NDManager manager = engine.newBaseManager(Device.cpu())) {
            NDArray weight = manager.zeros(new Shape(ROWS, WIDTH));
            weight.setRequiresGradient(true);
            try (GradientCollector collector = engine.newGradientCollector()) {
                NDArray output;
                try (NDManager scope = manager.newSubManager()) {
                    NDArray index = scope.create(indices);
                    output =
                            Embedding.embedding(index, weight, SparseFormat.DENSE)
                                    .singletonOrThrow();
                    output.attach(manager);
                }
                // The embedding backward still reads the index array that the scope closed.
                List<ByteBuffer> reused = reuseReleasedMemory(SIZE * Integer.BYTES);
                collector.backward(output.sum());
                Reference.reachabilityFence(reused);
            }
            Assert.assertEquals(weight.getGradient().toFloatArray(), expected);
        }
    }

    @Test
    public void viewKeepsTheDataOfAClosedArray() throws InterruptedException {
        float[] data = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            data[i] = i;
        }
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            NDArray view;
            try (NDManager scope = manager.newSubManager()) {
                view = scope.create(data).reshape(64, 64);
                view.attach(manager);
            }
            List<ByteBuffer> reused = reuseReleasedMemory(SIZE * Float.BYTES);
            Assert.assertEquals(view.toFloatArray(), data);
            Reference.reachabilityFence(reused);
        }
    }

    /**
     * Waits until the garbage collector has released unreachable direct buffers, then allocates
     * cleared buffers of the same size so that the released memory is reused.
     */
    private static List<ByteBuffer> reuseReleasedMemory(int bytes) throws InterruptedException {
        ReferenceQueue<Object> queue = new ReferenceQueue<>();
        // The second marker is enqueued only after all references from the first collection.
        for (int i = 0; i < 2; i++) {
            PhantomReference<Object> marker = new PhantomReference<>(new Object(), queue);
            System.gc();
            Assert.assertSame(queue.remove(30_000), marker, "garbage collection did not finish");
        }
        List<ByteBuffer> buffers = new ArrayList<>(BUFFERS);
        for (int i = 0; i < BUFFERS; i++) {
            buffers.add(ByteBuffer.allocateDirect(bytes));
        }
        return buffers;
    }
}
