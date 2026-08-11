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
import ai.djl.engine.InferenceMode;
import ai.djl.pytorch.jni.JniUtils;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.testng.Assert;
import org.testng.annotations.Test;

@SuppressWarnings("try") // InferenceMode resources affect thread-local state until close().
public class PtInferenceModeTest {

    @Test
    public void testFreshWorkerScopeDisablesAndRestoresGradMode() throws Exception {
        Engine engine = Engine.getInstance();
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            boolean restored =
                    executor
                            .submit(
                                    () -> {
                                        Assert.assertTrue(JniUtils.isGradMode());
                                        try (InferenceMode ignored = engine.newInferenceMode()) {
                                            Assert.assertFalse(JniUtils.isGradMode());
                                            try (InferenceMode nested = engine.newInferenceMode()) {
                                                Assert.assertFalse(JniUtils.isGradMode());
                                            }
                                            Assert.assertFalse(JniUtils.isGradMode());
                                        }
                                        return JniUtils.isGradMode();
                                    })
                            .get();
            Assert.assertTrue(restored);
        } finally {
            executor.shutdownNow();
        }
    }
}
