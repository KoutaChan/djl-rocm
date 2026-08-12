/*
 * Copyright 2025 KoutaChan. Licensed under the Apache License, Version 2.0.
 */
package ai.djl.engine;

/**
 * Thread-local guard around an automatic mixed-precision ("autocast") scope.
 *
 * <p>Inside the lifetime of an {@code Autocast} instance, heavy linear / conv / attention style ops
 * are transparently cast to a lower-precision dtype (typically {@link
 * ai.djl.ndarray.types.DataType#BFLOAT16}) on the configured {@link ai.djl.Device}, while
 * numerically sensitive ops (softmax, reductions, loss functions) stay in {@code FP32}. On {@link
 * #close()} the previous autocast state is restored, so scopes nest safely.
 *
 * <p>Engines that do not implement autocast return a no-op guard from {@link Engine#newAutocast},
 * so callers can always wrap their forward pass in {@code try-with-resources} without
 * feature-detection.
 *
 * <p>Typical usage (training loop):
 *
 * <pre>{@code
 * try (GradientCollector gc = engine.newGradientCollector()) {
 *     NDArray loss;
 *     try (Autocast ac = engine.newAutocast(device, DataType.BFLOAT16)) {
 *         NDList out = model.forward(input);
 *         loss = computeLoss(out).toType(DataType.FLOAT32, false);
 *     }
 *     // backward + optimizer step run with the FP32 master weights outside the scope
 *     gc.backward(loss);
 *     trainer.step();
 * }
 * }</pre>
 */
public interface Autocast extends AutoCloseable {

    /** {@inheritDoc} */
    @Override
    void close();
}
