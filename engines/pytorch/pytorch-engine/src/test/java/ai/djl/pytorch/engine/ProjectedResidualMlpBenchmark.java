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
import ai.djl.engine.fusion.FusionConstantBindings;
import ai.djl.engine.fusion.FusionExecutable;
import ai.djl.engine.fusion.FusionInvocation;
import ai.djl.engine.fusion.FusionOutputLease;
import ai.djl.engine.fusion.FusionPlan;
import ai.djl.engine.fusion.FusionRecipe;
import ai.djl.engine.fusion.FusionSession;
import ai.djl.engine.fusion.FusionSessionConfig;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

/** Runs the sustained projected-residual MLP functional and fusion benchmark. */
public final class ProjectedResidualMlpBenchmark {

    private static final int PREFIX_ROWS = 34;
    private static final int INPUT_WIDTH = 256;
    private static final int HIDDEN_WIDTH = 64;
    private static final int OUTPUT_WIDTH = 64;

    private ProjectedResidualMlpBenchmark() {}

    /**
     * Runs the benchmark.
     *
     * @param args optional warmup count followed by measured iteration count
     */
    public static void main(String[] args) {
        int warmup = args.length > 0 ? Integer.parseInt(args[0]) : 20;
        int iterations = args.length > 1 ? Integer.parseInt(args[1]) : 100;
        DataType dataType = args.length > 2 ? DataType.valueOf(args[2]) : DataType.FLOAT16;
        PtEngine engine = (PtEngine) Engine.getInstance();
        if (engine.getGpuCount() == 0) {
            throw new IllegalStateException("A PyTorch CUDA or ROCm device is required.");
        }
        for (int batch : new int[] {256, 512, 1024, 2048, 4096}) {
            benchmark(engine, Device.gpu(0), batch, dataType, warmup, iterations);
        }
    }

    private static void benchmark(
            PtEngine engine,
            Device device,
            int batch,
            DataType dataType,
            int warmup,
            int iterations) {
        Recipe recipe = new Recipe(batch, dataType);
        try (NDManager manager = engine.newBaseManager(device);
                NDArray input =
                        manager.ones(new Shape(batch, PREFIX_ROWS, INPUT_WIDTH), dataType)
                                .mul(0.01f);
                NDArray combinedWeight =
                        manager.ones(new Shape(OUTPUT_WIDTH + HIDDEN_WIDTH, INPUT_WIDTH), dataType)
                                .mul(0.002f);
                NDArray combinedBias =
                        manager.ones(new Shape(OUTPUT_WIDTH + HIDDEN_WIDTH), dataType)
                                .mul(-0.003f);
                NDArray outputWeight =
                        manager.ones(new Shape(OUTPUT_WIDTH, HIDDEN_WIDTH), dataType).mul(0.004f);
                FusionPlan plan = engine.newFusionCompiler(device).prepare(recipe.recipe);
                FusionExecutable executable =
                        plan.bind(
                                FusionConstantBindings.builder(recipe.recipe)
                                        .bind(recipe.combinedWeight, combinedWeight)
                                        .bind(recipe.combinedBias, combinedBias)
                                        .bind(recipe.outputWeight, outputWeight)
                                        .build());
                FusionSession session =
                        executable.newSession(
                                manager,
                                FusionSessionConfig.builder().optOutputSlotCount(1).build())) {
            runFunctional(input, combinedWeight, combinedBias, outputWeight, warmup);
            runFusion(session, recipe, input, batch, warmup);
            synchronize(device);
            double functionalNanos =
                    timeFunctional(
                            device, input, combinedWeight, combinedBias, outputWeight, iterations);
            double fusionNanos = timeFusion(device, session, recipe, input, batch, iterations);
            float maximumError =
                    maximumSampleError(
                            session,
                            recipe,
                            input,
                            combinedWeight,
                            combinedBias,
                            outputWeight,
                            batch);
            long rows = (long) batch * PREFIX_ROWS;
            System.out.printf(
                    "dataType=%s batch=%d rows=%d functional_us=%.3f functional_rows_s=%.3f "
                            + "fusion_us=%.3f fusion_rows_s=%.3f max_error=%.7f%n",
                    dataType,
                    batch,
                    rows,
                    functionalNanos / 1_000.0,
                    rows * 1_000_000_000.0 / functionalNanos,
                    fusionNanos / 1_000.0,
                    rows * 1_000_000_000.0 / fusionNanos,
                    maximumError);
        }
    }

    @SuppressWarnings("try")
    private static void runFunctional(
            NDArray input,
            NDArray combinedWeight,
            NDArray combinedBias,
            NDArray outputWeight,
            int iterations) {
        for (int iteration = 0; iteration < iterations; ++iteration) {
            try (NDArray ignored =
                    NDArrays.projectedResidualMlp(
                            input, combinedWeight, combinedBias, outputWeight)) {
                // The event recorded by the caller establishes completion.
            }
        }
    }

    @SuppressWarnings("try")
    private static void runFusion(
            FusionSession session, Recipe recipe, NDArray input, int batch, int iterations) {
        for (int iteration = 0; iteration < iterations; ++iteration) {
            try (FusionInvocation invocation = session.acquire()) {
                invocation.setInput(recipe.input, input);
                invocation.setDimension(recipe.batch, batch);
                try (FusionOutputLease ignored = invocation.submit()) {
                    // Reuse is stream ordered; the event recorded by the caller establishes
                    // completion.
                }
            }
        }
    }

    private static double timeFunctional(
            Device device,
            NDArray input,
            NDArray combinedWeight,
            NDArray combinedBias,
            NDArray outputWeight,
            int iterations) {
        long started = System.nanoTime();
        runFunctional(input, combinedWeight, combinedBias, outputWeight, iterations);
        try (PtEvent completion = new PtEvent(device)) {
            completion.record();
            completion.synchronize();
        }
        return (System.nanoTime() - started) / (double) iterations;
    }

    private static void synchronize(Device device) {
        try (PtEvent completion = new PtEvent(device)) {
            completion.record();
            completion.synchronize();
        }
    }

    private static double timeFusion(
            Device device,
            FusionSession session,
            Recipe recipe,
            NDArray input,
            int batch,
            int iterations) {
        long started = System.nanoTime();
        runFusion(session, recipe, input, batch, iterations);
        try (PtEvent completion = new PtEvent(device)) {
            completion.record();
            completion.synchronize();
        }
        return (System.nanoTime() - started) / (double) iterations;
    }

    private static float maximumSampleError(
            FusionSession session,
            Recipe recipe,
            NDArray input,
            NDArray combinedWeight,
            NDArray combinedBias,
            NDArray outputWeight,
            int batch) {
        try (NDArray reference =
                        NDArrays.projectedResidualMlp(
                                input, combinedWeight, combinedBias, outputWeight);
                FusionInvocation invocation = session.acquire()) {
            invocation.setInput(recipe.input, input);
            invocation.setDimension(recipe.batch, batch);
            try (FusionOutputLease lease = invocation.submit()) {
                lease.synchronize();
                try (NDArray actual = lease.get(recipe.output).get("0:{}", batch);
                        NDArray difference = actual.sub(reference).abs();
                        NDArray maximum = difference.max()) {
                    return maximum.toType(DataType.FLOAT32, false).getFloat();
                }
            }
        }
    }

    private static final class Recipe {

        private final FusionRecipe.Dimension batch;
        private final FusionRecipe.Input input;
        private final FusionRecipe.Constant combinedWeight;
        private final FusionRecipe.Constant combinedBias;
        private final FusionRecipe.Constant outputWeight;
        private final FusionRecipe.Output output;
        private final FusionRecipe recipe;

        private Recipe(int maximumBatch, DataType dataType) {
            FusionRecipe.Builder builder = FusionRecipe.builder("projected-residual-mlp-benchmark");
            batch = builder.addDimension("batch", maximumBatch);
            input =
                    builder.addInput(
                            "input",
                            FusionRecipe.TensorSpec.of(dataType, batch, PREFIX_ROWS, INPUT_WIDTH));
            combinedWeight =
                    builder.addConstant(
                            "combinedWeight",
                            FusionRecipe.TensorSpec.fixed(
                                    dataType, OUTPUT_WIDTH + HIDDEN_WIDTH, INPUT_WIDTH));
            combinedBias =
                    builder.addConstant(
                            "combinedBias",
                            FusionRecipe.TensorSpec.fixed(dataType, OUTPUT_WIDTH + HIDDEN_WIDTH));
            outputWeight =
                    builder.addConstant(
                            "outputWeight",
                            FusionRecipe.TensorSpec.fixed(dataType, OUTPUT_WIDTH, HIDDEN_WIDTH));
            FusionRecipe.ProjectedResidualMlp value =
                    builder.projectedResidualMlp("value", input)
                            .setCombinedWeight(combinedWeight)
                            .setCombinedBias(combinedBias)
                            .setOutputWeight(outputWeight)
                            .build();
            output = builder.addOutput("output", value);
            recipe = builder.build();
        }
    }
}
