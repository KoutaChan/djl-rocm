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
import ai.djl.ndarray.NDManager;

import org.testng.Assert;
import org.testng.annotations.Test;

public class EngineFusionCompatibilityTest {

    @Test
    public void unsupportedEngineRejectsFusionCompiler() {
        Engine engine = new UnsupportedFusionEngine();
        Assert.assertThrows(
                UnsupportedOperationException.class, () -> engine.newFusionCompiler(Device.cpu()));
    }

    private static final class UnsupportedFusionEngine extends Engine {

        @Override
        public Engine getAlternativeEngine() {
            return this;
        }

        @Override
        public String getEngineName() {
            return "unsupported-fusion-test";
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
        public Model newModel(String name, Device device) {
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
