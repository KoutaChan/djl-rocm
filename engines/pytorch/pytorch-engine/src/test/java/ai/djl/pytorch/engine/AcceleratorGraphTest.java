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

import org.testng.Assert;
import org.testng.annotations.Test;

/** Verifies fixed-shape accelerator graph capture through the engine API. */
public class AcceleratorGraphTest {

    @Test
    public void capturedWorkloadCanBeReplayed() {
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            return;
        }
        Device device = Device.gpu(0);
        try (NDManager manager = engine.newBaseManager(device);
                NDArray input = manager.ones(new Shape(64, 32));
                NDArray bias = manager.ones(new Shape(1, 32)).mul(2f);
                NDArray warmup = input.add(bias).square();
                PtAcceleratorGraph graph = ((PtEngine) engine).newAcceleratorGraph(device)) {
            Assert.assertEquals(warmup.sum().getFloat(), 64f * 32f * 9f, 1e-3f);
            graph.beginCapture();
            NDArray output = input.add(bias).square();
            graph.endCapture();
            graph.replay();
            Assert.assertEquals(output.sum().getFloat(), 64f * 32f * 9f, 1e-3f);
            output.close();
        }
    }
}
