/*
 * Copyright 2025 KoutaChan.
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

/**
 * A thread-local automatic mixed-precision scope.
 *
 * <p>While the scope is open, matrix multiplication, convolution, and attention operations can use
 * a lower-precision data type on the configured {@link ai.djl.Device}, while numerically sensitive
 * operations remain in {@link ai.djl.ndarray.types.DataType#FLOAT32}. Closing the scope restores
 * the previous autocast state, so scopes can be nested.
 *
 * <p>Engines that do not implement autocast return a no-op guard from {@link Engine#newAutocast},
 * so callers can always wrap their forward pass in {@code try-with-resources} without
 * feature-detection.
 *
 * <p>For FLOAT16 training, use {@link ai.djl.training.DefaultTrainingConfig#optAutocast} and {@link
 * ai.djl.training.Trainer#newGradientCollector()}. The training API pairs FLOAT16 autocast with a
 * dynamic {@link ai.djl.training.GradScaler}; a raw engine autocast scope does not perform loss
 * scaling by itself.
 *
 * <p>Typical usage:
 *
 * <pre>{@code
 * try (Autocast autocast = engine.newAutocast(device, DataType.BFLOAT16)) {
 *     NDArray output = input.matMul(weight);
 * }
 * }</pre>
 */
public interface Autocast extends AutoCloseable {

    /** {@inheritDoc} */
    @Override
    void close();
}
