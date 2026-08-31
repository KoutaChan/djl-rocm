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

import ai.djl.engine.fusion.FusionConstantBindings;
import ai.djl.engine.fusion.FusionRecipe;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDScope;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.util.ArrayList;
import java.util.List;

final class PtSingleQueryReadoutTestSupport {

    private static NDArray patternedArray(
            NDManager manager,
            DataType dataType,
            Shape shape,
            int period,
            int center,
            float scale) {
        try (NDScope scope = new NDScope()) {
            scope.suppressNotUsedWarning();
            NDArray result =
                    manager.arange((float) shape.size())
                            .mod(period)
                            .sub(center)
                            .mul(scale)
                            .reshape(shape)
                            .toType(dataType, false);
            NDScope.unregister(result);
            return result;
        }
    }

    static final class SingleQueryReadoutFixture {

        final FusionRecipe.Dimension batch;
        final FusionRecipe.Input memory;
        final FusionRecipe.Input querySource;
        final FusionRecipe.Input mask;
        final FusionRecipe.Output policyOutput;
        final FusionRecipe.Output valueOutput;
        final int queryIndex;
        final List<SingleQueryReadoutConstants> readoutConstants;
        final FusionRecipe recipe;

        SingleQueryReadoutFixture(DataType dataType) {
            this(dataType, dataType, 2, false, 0);
        }

        SingleQueryReadoutFixture(
                DataType dataType, int maximumBatch, boolean shareMemoryAsQuery, int queryIndex) {
            this(dataType, dataType, maximumBatch, shareMemoryAsQuery, queryIndex);
        }

        SingleQueryReadoutFixture(
                DataType dataType,
                DataType maskDataType,
                int maximumBatch,
                boolean shareMemoryAsQuery,
                int queryIndex) {
            this(dataType, dataType, maskDataType, maximumBatch, shareMemoryAsQuery, queryIndex);
        }

        SingleQueryReadoutFixture(
                DataType computeDataType,
                DataType memoryDataType,
                DataType maskDataType,
                int maximumBatch,
                boolean shareMemoryAsQuery,
                int queryIndex) {
            FusionRecipe.Builder builder = FusionRecipe.builder("single-query-readout");
            batch = builder.addDimension("batch", maximumBatch);
            memory =
                    builder.addInput(
                            "memory", FusionRecipe.TensorSpec.of(memoryDataType, batch, 151, 256));
            querySource =
                    shareMemoryAsQuery
                            ? memory
                            : builder.addInput(
                                    "querySource",
                                    FusionRecipe.TensorSpec.of(memoryDataType, batch, 6, 256));
            mask = builder.addInput("mask", FusionRecipe.TensorSpec.of(maskDataType, batch, 151));
            FusionRecipe.SingleQueryCrossAttentionReadoutGroupBuilder group =
                    builder.singleQueryCrossAttentionReadoutGroup(
                                    "readouts", memory, querySource, mask, 4)
                            .optQueryIndex(queryIndex);
            readoutConstants = new ArrayList<>(2);
            readoutConstants.add(addReadout(builder, group, "policy", 384, computeDataType));
            readoutConstants.add(addReadout(builder, group, "value", 256, computeDataType));
            FusionRecipe.SingleQueryCrossAttentionReadoutGroup readouts = group.build();
            policyOutput = builder.addOutput("policy", readouts.getReadoutState(0));
            valueOutput = builder.addOutput("value", readouts.getReadoutState(1));
            this.queryIndex = queryIndex;
            recipe = builder.build();
        }

        BoundSingleQueryReadouts bind(NDManager manager) {
            FusionConstantBindings.Builder bindings = FusionConstantBindings.builder(recipe);
            NDList resources = new NDList();
            List<SingleQueryReadoutArrays> arrays = new ArrayList<>(readoutConstants.size());
            for (SingleQueryReadoutConstants constants : readoutConstants) {
                NDArray[] values = new NDArray[constants.values.length];
                for (int index = 0; index < values.length; ++index) {
                    FusionRecipe.Constant constant = constants.values[index];
                    values[index] = readoutParameterArray(manager, constant, index);
                    resources.add(values[index]);
                    bindings.bind(constant, values[index]);
                }
                arrays.add(new SingleQueryReadoutArrays(values));
            }
            return new BoundSingleQueryReadouts(bindings.build(), arrays, resources);
        }

        private static SingleQueryReadoutConstants addReadout(
                FusionRecipe.Builder builder,
                FusionRecipe.SingleQueryCrossAttentionReadoutGroupBuilder group,
                String prefix,
                int feedForwardWidth,
                DataType dataType) {
            FusionRecipe.Constant seedWeight =
                    matrix(builder, prefix + "SeedWeight", 256, 512, dataType);
            FusionRecipe.Constant seedBias = vector(builder, prefix + "SeedBias", 256, dataType);
            FusionRecipe.Constant queryWeight =
                    matrix(builder, prefix + "QueryWeight", 64, 256, dataType);
            FusionRecipe.Constant queryBias = vector(builder, prefix + "QueryBias", 64, dataType);
            FusionRecipe.Constant keyValue =
                    matrix(builder, prefix + "KeyValue", 128, 256, dataType);
            FusionRecipe.Constant contextWeight =
                    matrix(builder, prefix + "ContextWeight", 256, 64, dataType);
            FusionRecipe.Constant contextBias =
                    vector(builder, prefix + "ContextBias", 256, dataType);
            FusionRecipe.Constant queryNormWeight =
                    vector(builder, prefix + "QueryNormWeight", 256, DataType.FLOAT32);
            FusionRecipe.Constant queryNormBias =
                    vector(builder, prefix + "QueryNormBias", 256, DataType.FLOAT32);
            FusionRecipe.Constant feedForwardNormWeight =
                    vector(builder, prefix + "FeedForwardNormWeight", 256, DataType.FLOAT32);
            FusionRecipe.Constant feedForwardNormBias =
                    vector(builder, prefix + "FeedForwardNormBias", 256, DataType.FLOAT32);
            FusionRecipe.Constant expansion =
                    matrix(builder, prefix + "Expansion", feedForwardWidth, 256, dataType);
            FusionRecipe.Constant expansionBias =
                    vector(builder, prefix + "ExpansionBias", feedForwardWidth, dataType);
            FusionRecipe.Constant projection =
                    matrix(builder, prefix + "Projection", 256, feedForwardWidth, dataType);
            FusionRecipe.Constant projectionBias =
                    vector(builder, prefix + "ProjectionBias", 256, dataType);
            FusionRecipe.Constant outputNormWeight =
                    vector(builder, prefix + "OutputNormWeight", 256, DataType.FLOAT32);
            FusionRecipe.Constant outputNormBias =
                    vector(builder, prefix + "OutputNormBias", 256, DataType.FLOAT32);
            group.addReadout(
                    seedWeight,
                    seedBias,
                    queryWeight,
                    queryBias,
                    keyValue,
                    contextWeight,
                    contextBias,
                    queryNormWeight,
                    queryNormBias,
                    feedForwardNormWeight,
                    feedForwardNormBias,
                    expansion,
                    expansionBias,
                    projection,
                    projectionBias,
                    outputNormWeight,
                    outputNormBias);
            return new SingleQueryReadoutConstants(
                    new FusionRecipe.Constant[] {
                        seedWeight,
                        seedBias,
                        queryWeight,
                        queryBias,
                        keyValue,
                        contextWeight,
                        contextBias,
                        queryNormWeight,
                        queryNormBias,
                        feedForwardNormWeight,
                        feedForwardNormBias,
                        expansion,
                        expansionBias,
                        projection,
                        projectionBias,
                        outputNormWeight,
                        outputNormBias
                    });
        }

        private static FusionRecipe.Constant vector(
                FusionRecipe.Builder builder, String name, int width, DataType dataType) {
            return builder.addConstant(name, FusionRecipe.TensorSpec.fixed(dataType, width));
        }

        private static NDArray readoutParameterArray(
                NDManager manager, FusionRecipe.Constant constant, int parameterIndex) {
            int[] periods = {31, 17, 29, 13, 37, 41, 19, 23, 11, 29, 17, 43, 19, 47, 23, 31, 13};
            float[] scales = {
                0.0015f, 0.002f, 0.002f, 0.001f, 0.0015f, 0.0015f, 0.001f, 0.003f, 0.001f, 0.003f,
                0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.003f, 0.001f
            };
            Shape shape = constant.getSpec().getMaximumShape();
            NDArray values =
                    patternedArray(
                            manager,
                            constant.getSpec().getDataType(),
                            shape,
                            periods[parameterIndex],
                            periods[parameterIndex] / 2,
                            scales[parameterIndex]);
            if (parameterIndex != 7 && parameterIndex != 9 && parameterIndex != 15) {
                return values;
            }
            try (NDArray closeable = values) {
                return closeable.add(1f);
            }
        }

        private static FusionRecipe.Constant matrix(
                FusionRecipe.Builder builder,
                String name,
                int rows,
                int columns,
                DataType dataType) {
            return builder.addConstant(
                    name, FusionRecipe.TensorSpec.fixed(dataType, rows, columns));
        }
    }

    private static final class SingleQueryReadoutConstants {

        final FusionRecipe.Constant[] values;

        private SingleQueryReadoutConstants(FusionRecipe.Constant[] values) {
            this.values = values;
        }
    }

    static final class SingleQueryReadoutArrays {

        final NDArray seedWeight;
        final NDArray seedBias;
        final NDArray queryWeight;
        final NDArray queryBias;
        final NDArray keyValueWeight;
        final NDArray contextWeight;
        final NDArray contextBias;
        final NDArray queryNormWeight;
        final NDArray queryNormBias;
        final NDArray feedForwardNormWeight;
        final NDArray feedForwardNormBias;
        final NDArray expansionWeight;
        final NDArray expansionBias;
        final NDArray projectionWeight;
        final NDArray projectionBias;
        final NDArray outputNormWeight;
        final NDArray outputNormBias;

        private SingleQueryReadoutArrays(NDArray[] values) {
            seedWeight = values[0];
            seedBias = values[1];
            queryWeight = values[2];
            queryBias = values[3];
            keyValueWeight = values[4];
            contextWeight = values[5];
            contextBias = values[6];
            queryNormWeight = values[7];
            queryNormBias = values[8];
            feedForwardNormWeight = values[9];
            feedForwardNormBias = values[10];
            expansionWeight = values[11];
            expansionBias = values[12];
            projectionWeight = values[13];
            projectionBias = values[14];
            outputNormWeight = values[15];
            outputNormBias = values[16];
        }
    }

    static final class BoundSingleQueryReadouts implements AutoCloseable {

        final FusionConstantBindings bindings;
        final List<SingleQueryReadoutArrays> readouts;
        final NDList resources;

        private BoundSingleQueryReadouts(
                FusionConstantBindings bindings,
                List<SingleQueryReadoutArrays> readouts,
                NDList resources) {
            this.bindings = bindings;
            this.readouts = readouts;
            this.resources = resources;
        }

        @Override
        public void close() {
            resources.close();
        }
    }
}
