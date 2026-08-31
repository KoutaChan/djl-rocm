/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.ndarray.internal;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDArrayIndexer;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.EmbeddingReduction;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.Activation;
import ai.djl.nn.core.Embedding;
import ai.djl.nn.recurrent.RNN;

import java.util.List;

/** An internal interface that encapsulates engine specific operations. */
@SuppressWarnings("MissingJavadocMethod")
public interface NDArrayEx {

    /*
    // NDArrays
    */

    /** Adds integer namespace offsets and performs a dense embedding lookup. */
    default NDArray embeddingWithOffsets(NDArray offsets, NDArray table) {
        NDArray indices = getArray().add(offsets);
        return Embedding.embedding(indices, table, SparseFormat.DENSE).singletonOrThrow();
    }

    /** Selects leading-axis rows while preserving all trailing dimensions. */
    default NDArray gatherRows(NDArray rowIndices) {
        NDArray rows = getArray();
        Shape inputShape = rows.getShape();
        long inputRows = inputShape.get(0);
        long outputRows = rowIndices.getShape().size();
        long rowWidth = inputShape.slice(1).size();
        NDArray gatherIndices =
                rowIndices
                        .reshape(outputRows, 1)
                        .toType(DataType.INT64, false)
                        .broadcast(outputRows, rowWidth)
                        .stopGradient();
        long[] outputShape = inputShape.getShape().clone();
        outputShape[0] = outputRows;
        return rows.reshape(inputRows, rowWidth)
                .gather(gatherIndices, 0)
                .reshape(new Shape(outputShape));
    }

    /** Places leading-axis rows into a zero-initialized dense tensor. */
    default NDArray scatterRows(NDArray rowIndices, long rowCount) {
        NDArray rows = getArray();
        Shape inputShape = rows.getShape();
        long inputRows = inputShape.get(0);
        long rowWidth = inputShape.slice(1).size();
        NDArray scatterIndices =
                rowIndices
                        .reshape(inputRows, 1)
                        .toType(DataType.INT64, false)
                        .broadcast(inputRows, rowWidth)
                        .stopGradient();
        long[] outputShape = inputShape.getShape().clone();
        outputShape[0] = rowCount;
        return rows.getManager()
                .zeros(new Shape(rowCount, rowWidth), rows.getDataType())
                .scatter(scatterIndices, rows.reshape(inputRows, rowWidth), 0)
                .reshape(new Shape(outputShape));
    }

    /** Sums one lookup row from each contiguous table segment. */
    default NDArray segmentedLookupSum(NDArray storedIndices) {
        NDArray lookupTable = getArray();
        Shape tableShape = lookupTable.getShape();
        Shape indexShape = storedIndices.getShape();
        DataType indexType = storedIndices.getDataType();
        int indexRank = indexShape.dimension();
        if (indexType != DataType.INT16
                && indexType != DataType.INT32
                && indexType != DataType.INT64) {
            throw new IllegalArgumentException(
                    "segmented lookup indices must be INT16, INT32, or INT64: " + indexType);
        }
        if (tableShape.dimension() < 2
                || indexRank < 1
                || tableShape.get(0) <= 0
                || indexShape.get(indexRank - 1) <= 0
                || tableShape.get(0) % indexShape.get(indexRank - 1) != 0) {
            throw new IllegalArgumentException(
                    "segmented lookup sum requires table [segments * entries,...] and indices "
                            + "[...,segments]: "
                            + tableShape
                            + " / "
                            + indexShape);
        }
        long segmentCount = indexShape.get(indexRank - 1);
        long entriesPerSegment = tableShape.get(0) / segmentCount;
        NDArray segmentOffsets =
                lookupTable
                        .getManager()
                        .arange(segmentCount)
                        .toType(DataType.INT64, false)
                        .mul(entriesPerSegment);
        NDArray rowIndices =
                storedIndices
                        .toType(DataType.INT64, false)
                        .maximum(1)
                        .minimum(entriesPerSegment)
                        .sub(1)
                        .add(segmentOffsets)
                        .reshape(-1)
                        .stopGradient();

        long[] gatheredShape = new long[indexRank + tableShape.dimension() - 1];
        System.arraycopy(indexShape.getShape(), 0, gatheredShape, 0, indexRank);
        System.arraycopy(
                tableShape.getShape(), 1, gatheredShape, indexRank, tableShape.dimension() - 1);
        long[] outputShape = new long[gatheredShape.length - 1];
        System.arraycopy(gatheredShape, 0, outputShape, 0, indexRank - 1);
        System.arraycopy(
                gatheredShape, indexRank, outputShape, indexRank - 1, tableShape.dimension() - 1);
        return NDArrays.gatherRows(lookupTable, rowIndices)
                .reshape(new Shape(gatheredShape))
                .sum(new int[] {indexRank - 1})
                .reshape(new Shape(outputShape));
    }

    /** Selects one-based per-batch table entries while preserving zero padding. */
    default NDArray paddedBatchGather(NDArray storedIndices) {
        NDArray source = getArray();
        Shape sourceShape = source.getShape();
        Shape indexShape = storedIndices.getShape();
        if (sourceShape.dimension() < 2
                || indexShape.dimension() < 1
                || sourceShape.get(0) <= 0
                || sourceShape.get(1) <= 0
                || indexShape.get(0) != sourceShape.get(0)) {
            throw new IllegalArgumentException(
                    "padded batch gather requires source [B,N,...] and indices [B,...]: "
                            + sourceShape
                            + " / "
                            + indexShape);
        }
        long batchCount = sourceShape.get(0);
        long entries = sourceShape.get(1);
        long indicesPerBatch = indexShape.size() / batchCount;
        NDArray present = storedIndices.gt(0).logicalAnd(storedIndices.lte(entries)).stopGradient();
        NDArray batchOffsets =
                source.getManager()
                        .arange(batchCount)
                        .toType(DataType.INT64, false)
                        .mul(entries)
                        .reshape(batchCount, 1);
        NDArray rowIndices =
                storedIndices
                        .reshape(batchCount, indicesPerBatch)
                        .toType(DataType.INT64, false)
                        .maximum(1)
                        .minimum(entries)
                        .sub(1)
                        .add(batchOffsets)
                        .reshape(indexShape.size())
                        .stopGradient();
        return zeroPaddedGatherResult(source, storedIndices, present, rowIndices, 2);
    }

    /** Selects one-based entries from two per-batch table dimensions. */
    default NDArray paddedBatchGather(NDArray outerStoredIndices, NDArray innerStoredIndices) {
        NDArray source = getArray();
        Shape sourceShape = source.getShape();
        Shape indexShape = outerStoredIndices.getShape();
        if (sourceShape.dimension() < 3
                || indexShape.dimension() < 1
                || !indexShape.equals(innerStoredIndices.getShape())
                || sourceShape.get(0) <= 0
                || sourceShape.get(1) <= 0
                || sourceShape.get(2) <= 0
                || indexShape.get(0) != sourceShape.get(0)) {
            throw new IllegalArgumentException(
                    "two-dimensional padded batch gather requires source [B,N,M,...] and "
                            + "matching indices [B,...]: "
                            + sourceShape
                            + " / "
                            + indexShape
                            + " / "
                            + innerStoredIndices.getShape());
        }
        long batchCount = sourceShape.get(0);
        long outerEntries = sourceShape.get(1);
        long innerEntries = sourceShape.get(2);
        long indicesPerBatch = indexShape.size() / batchCount;
        NDArray present =
                outerStoredIndices
                        .gt(0)
                        .logicalAnd(outerStoredIndices.lte(outerEntries))
                        .logicalAnd(innerStoredIndices.gt(0))
                        .logicalAnd(innerStoredIndices.lte(innerEntries))
                        .stopGradient();
        NDArray batchOffsets =
                source.getManager()
                        .arange(batchCount)
                        .toType(DataType.INT64, false)
                        .mul(Math.multiplyExact(outerEntries, innerEntries))
                        .reshape(batchCount, 1);
        NDArray rowIndices =
                outerStoredIndices
                        .reshape(batchCount, indicesPerBatch)
                        .toType(DataType.INT64, false)
                        .maximum(1)
                        .minimum(outerEntries)
                        .sub(1)
                        .mul(innerEntries)
                        .add(
                                innerStoredIndices
                                        .reshape(batchCount, indicesPerBatch)
                                        .toType(DataType.INT64, false)
                                        .maximum(1)
                                        .minimum(innerEntries)
                                        .sub(1))
                        .add(batchOffsets)
                        .reshape(indexShape.size())
                        .stopGradient();
        return zeroPaddedGatherResult(source, outerStoredIndices, present, rowIndices, 3);
    }

    /** Selects one-based table entries using explicit zero-based batch indices. */
    default NDArray paddedBatchGatherByBatchIndices(NDArray batchIndices, NDArray storedIndices) {
        NDArray source = getArray();
        Shape sourceShape = source.getShape();
        if (sourceShape.dimension() < 2
                || sourceShape.get(0) <= 0
                || sourceShape.get(1) <= 0
                || !batchIndices.getShape().equals(storedIndices.getShape())) {
            throw new IllegalArgumentException(
                    "explicit-batch padded gather requires source [B,N,...] and matching index "
                            + "shapes: "
                            + sourceShape
                            + " / "
                            + batchIndices.getShape()
                            + " / "
                            + storedIndices.getShape());
        }
        long batchCount = sourceShape.get(0);
        long entries = sourceShape.get(1);
        NDArray present =
                storedIndices
                        .gt(0)
                        .logicalAnd(storedIndices.lte(entries))
                        .logicalAnd(batchIndices.gte(0))
                        .logicalAnd(batchIndices.lt(batchCount))
                        .stopGradient();
        NDArray rowIndices =
                batchIndices
                        .toType(DataType.INT64, false)
                        .maximum(0)
                        .minimum(batchCount - 1)
                        .mul(entries)
                        .add(
                                storedIndices
                                        .toType(DataType.INT64, false)
                                        .maximum(1)
                                        .minimum(entries)
                                        .sub(1))
                        .reshape(storedIndices.getShape().size())
                        .stopGradient();
        return zeroPaddedGatherResult(source, storedIndices, present, rowIndices, 2);
    }

    /** Builds the portable padded-gather result after its flattened row indices are known. */
    static NDArray zeroPaddedGatherResult(
            NDArray source,
            NDArray storedIndices,
            NDArray present,
            NDArray rowIndices,
            int indexedDimensions) {
        Shape sourceShape = source.getShape();
        Shape indexShape = storedIndices.getShape();
        long tableRows = sourceShape.slice(0, indexedDimensions).size();
        Shape trailingShape = sourceShape.slice(indexedDimensions);
        long rowWidth = trailingShape.size();
        NDArray gathered = NDArrays.gatherRows(source.reshape(tableRows, rowWidth), rowIndices);
        long[] outputDimensions = new long[indexShape.dimension() + trailingShape.dimension()];
        System.arraycopy(indexShape.getShape(), 0, outputDimensions, 0, indexShape.dimension());
        System.arraycopy(
                trailingShape.getShape(),
                0,
                outputDimensions,
                indexShape.dimension(),
                trailingShape.dimension());
        NDArray paddingMask = present.toType(source.getDataType(), false);
        for (int axis = 0; axis < trailingShape.dimension(); axis++) {
            paddingMask = paddingMask.expandDims(indexShape.dimension());
        }
        return gathered.reshape(new Shape(outputDimensions)).mul(paddingMask);
    }

    /** Returns float32 probabilities normalized over nonzero mask entries. */
    default NDArray maskedSoftmax(NDArray mask, int axis) {
        NDArray logits = getArray().toType(DataType.FLOAT32, false);
        NDArray floatMask =
                mask.neq(0)
                        .toType(DataType.FLOAT32, false)
                        .broadcast(logits.getShape())
                        .stopGradient();
        NDArray masked = logits.add(floatMask.neg().add(1.0f).mul(-1.0e30f));
        return NDArrays.where(floatMask, masked.softmax(axis), logits.zerosLike());
    }

    /** Pools values with independently masked softmax weights for several groups. */
    default NDArray groupedMaskedSoftmaxPool(NDArray mask, NDArray values) {
        NDArray logits = getArray();
        int choiceAxis = logits.getShape().dimension() - 1;
        NDArray expandedLogits = logits.expandDims(-1).broadcast(mask.getShape());
        NDArray weights = NDArrays.maskedSoftmax(expandedLogits, mask, choiceAxis);
        NDArray pooled =
                weights.expandDims(-1)
                        .mul(values.toType(DataType.FLOAT32, false).expandDims(-2))
                        .sum(new int[] {choiceAxis});
        NDArray present = mask.neq(0).sum(new int[] {choiceAxis}).gt(0).expandDims(-1);
        NDArray masked = NDArrays.where(present, pooled, pooled.zerosLike());
        long groupCount = mask.getShape().get(mask.getShape().dimension() - 1);
        NDList groups = new NDList(Math.toIntExact(groupCount));
        for (int group = 0; group < groupCount; ++group) {
            groups.add(masked.get("...,{},:", group));
        }
        return NDArrays.stack(groups, 0);
    }

    /** Returns the float32 log normalizer over nonzero mask entries, retaining the reduced axis. */
    default NDArray maskedLogSumExp(NDArray mask, int axis) {
        NDArray logits = getArray().toType(DataType.FLOAT32, false);
        int normalizedAxis = axis < 0 ? axis + logits.getShape().dimension() : axis;
        long[] reducedShape = logits.getShape().getShape().clone();
        reducedShape[normalizedAxis] = 1;
        NDArray floatMask =
                mask.neq(0)
                        .toType(DataType.FLOAT32, false)
                        .broadcast(logits.getShape())
                        .stopGradient();
        NDArray masked = logits.add(floatMask.neg().add(1.0f).mul(-1.0e30f));
        NDArray maximum = masked.max(new int[] {normalizedAxis}).reshape(new Shape(reducedShape));
        NDArray sum =
                masked.sub(maximum)
                        .exp()
                        .mul(floatMask)
                        .sum(new int[] {normalizedAxis})
                        .reshape(new Shape(reducedShape))
                        .maximum(1.0e-30f);
        NDArray present =
                floatMask
                        .sum(new int[] {normalizedAxis})
                        .reshape(new Shape(reducedShape))
                        .gt(0.5f)
                        .toType(DataType.FLOAT32, false)
                        .stopGradient();
        return sum.log().add(maximum).mul(present);
    }

    /**
     * Applies reverse division with a scalar - i.e., (n / thisArrayValues).
     *
     * @param n the Value to use for reverse division
     * @return a copy of the array after applying reverse division
     */
    default NDArray rdiv(Number n) {
        NDArray array = getArray();
        NDArray b = array.getManager().create(n).toType(array.getDataType(), false);
        return rdiv(b);
    }

    /**
     * Applies reverse division with a scalar - i.e., (n / thisArrayValues).
     *
     * @param b the ndarray to use for reverse division
     * @return a copy of the array after applying reverse division
     */
    default NDArray rdiv(NDArray b) {
        return b.div(getArray());
    }

    /**
     * Applies in place reverse division - i.e., (n / thisArrayValues).
     *
     * @param n the value to use for reverse division
     * @return this array after applying reverse division
     */
    default NDArray rdivi(Number n) {
        NDArray array = getArray();
        NDArray b = array.getManager().create(n).toType(array.getDataType(), false);
        return rdivi(b);
    }

    /**
     * Applies in place reverse division - i.e., (n / thisArrayValues).
     *
     * @param b the ndarray to use for reverse division
     * @return this array after applying reverse division
     */
    NDArray rdivi(NDArray b);

    /**
     * Applies reverse subtraction with duplicates - i.e., (n - thisArrayValues).
     *
     * @param n the value to use for reverse subtraction
     * @return a copy of array after reverse subtraction
     */
    default NDArray rsub(Number n) {
        return getArray().sub(n).neg();
    }

    /**
     * Applies reverse subtraction with duplicates - i.e., (n - thisArrayValues).
     *
     * @param b the ndarray to use for reverse subtraction
     * @return a copy of the array after reverse subtraction
     */
    default NDArray rsub(NDArray b) {
        return b.sub(getArray());
    }

    /**
     * Applies reverse subtraction in place - i.e., (n - thisArrayValues).
     *
     * @param n the value to use for reverse subtraction
     * @return this array after reverse subtraction
     */
    default NDArray rsubi(Number n) {
        return getArray().subi(n).negi();
    }

    /**
     * Applies reverse subtraction in place - i.e., (n - thisArrayValues).
     *
     * @param b the ndarray to use for reverse subtraction
     * @return this array after reverse subtraction
     */
    default NDArray rsubi(NDArray b) {
        return getArray().subi(b).negi();
    }

    /**
     * Applies reverse remainder of division with a scalar.
     *
     * @param n the value to use for reverse division
     * @return a copy of array after applying reverse division
     */
    default NDArray rmod(Number n) {
        NDArray array = getArray();
        NDArray b = array.getManager().create(n).toType(array.getDataType(), false);
        return rmod(b);
    }

    /**
     * Applies reverse remainder of division.
     *
     * @param b the ndarray to use for reverse division
     * @return a copy of array after applying reverse division
     */
    default NDArray rmod(NDArray b) {
        return b.mod(getArray());
    }

    /**
     * Applies in place reverse remainder of division with a scalar.
     *
     * @param n the value to use for reverse division
     * @return this array after applying reverse division
     */
    default NDArray rmodi(Number n) {
        NDArray array = getArray();
        NDArray b = array.getManager().create(n).toType(array.getDataType(), false);
        return rmodi(b);
    }

    /**
     * Applies in place reverse remainder of division.
     *
     * @param b the ndarray to use for reverse division
     * @return this array after applying reverse division
     */
    NDArray rmodi(NDArray b);

    /**
     * Reverses the power of each element being raised in the {@code NDArray}.
     *
     * @param n the value to use for reverse power
     * @return a copy of array after applying reverse power
     */
    default NDArray rpow(Number n) {
        NDArray array = getArray();
        NDArray b = array.getManager().create(n).toType(array.getDataType(), false);
        return b.pow(array);
    }

    /**
     * Reverses the power of each element being raised in the {@code NDArray} in place.
     *
     * @param n the value to use for reverse power
     * @return a copy of array after applying reverse power
     */
    NDArray rpowi(Number n);

    /*
    // Activations
    */

    /**
     * Computes rectified linear activation.
     *
     * @return a copy of array after applying relu
     */
    NDArray relu();

    NDArray sigmoid();

    NDArray tanh();

    NDArray softPlus();

    NDArray softSign();

    NDArray leakyRelu(float alpha);

    NDArray elu(float alpha);

    NDArray selu();

    NDArray gelu();

    default NDArray swish(float beta) {
        return Activation.sigmoid(getArray().mul(beta)).mul(getArray());
    }

    default NDArray mish() {
        return getArray().exp().add(1).log2().tanh().mul(getArray());
    }

    /*
    // Pooling Operations
    */

    NDArray maxPool(Shape kernelShape, Shape stride, Shape padding, boolean ceilMode);

    NDArray globalMaxPool();

    NDArray avgPool(
            Shape kernelShape,
            Shape stride,
            Shape padding,
            boolean ceilMode,
            boolean countIncludePad);

    NDArray globalAvgPool();

    NDArray lpPool(
            float normType, Shape kernelShape, Shape stride, Shape padding, boolean ceilMode);

    NDArray globalLpPool(float normType);

    /*
    // Optimizer
    */

    void adadeltaUpdate(
            NDList inputs,
            NDList weights,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float rho,
            float epsilon);

    void adagradUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float epsilon);

    void adamUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float learningRateBiasCorrection,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float beta1,
            float beta2,
            float epsilon,
            boolean lazyUpdate,
            boolean adamw);

    void nagUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float momentum);

    void rmspropUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float rho,
            float momentum,
            float epsilon,
            boolean centered);

    void sgdUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float momentum,
            boolean lazyUpdate);

    /*
    // Neural network
    */

    NDList convolution(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape stride,
            Shape padding,
            Shape dilation,
            int groups);

    NDList deconvolution(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape stride,
            Shape padding,
            Shape outPadding,
            Shape dilation,
            int groups);

    NDList linear(NDArray input, NDArray weight, NDArray bias);

    NDList embedding(NDArray input, NDArray weight, SparseFormat sparse);

    NDList prelu(NDArray input, NDArray alpha);

    NDList dropout(NDArray input, float rate, boolean training);

    NDList layerNorm(NDArray input, Shape normalizedShape, NDArray gamma, NDArray beta, float eps);

    NDList batchNorm(
            NDArray input,
            NDArray runningMean,
            NDArray runningVar,
            NDArray gamma,
            NDArray beta,
            int axis,
            float momentum,
            float eps,
            boolean training);

    /**
     * Applies RNN operation to input data.
     *
     * @param input the inputs to the recurrent operation.
     * @param state the hidden state to the recurrent operation.
     * @param params all params (weights and biases) for the recurrent operation
     * @param hasBiases If false, then the recurrent operation does not use bias weights b_ih and
     *     b_hh
     * @param numLayers the number of recurrent layers.
     * @param activation the activation function to use
     * @param dropRate If non-zero, introduces a Dropout layer on the outputs of each RNN layer
     *     except the last layer, with dropout probability equal to dropout
     * @param training apply dropout if is true
     * @param bidirectional If true, becomes a bidirectional RNN
     * @param batchFirst If true, then the input and output NDArray are provided as (batch, seq,
     *     feature)
     * @return the output of the operation
     */
    NDList rnn(
            NDArray input,
            NDArray state,
            NDList params,
            boolean hasBiases,
            int numLayers,
            RNN.Activation activation,
            double dropRate,
            boolean training,
            boolean bidirectional,
            boolean batchFirst);

    /**
     * Applies GRU operation to input data.
     *
     * @param input the inputs to the GRU operation.
     * @param state the hidden state to the GRU operation.
     * @param params all params (weights and biases) for the GRU operation
     * @param hasBiases If false, then the recurrent operation does not use bias weights b_ih and
     *     b_hh
     * @param numLayers the number of recurrent layers.
     * @param dropRate If non-zero, introduces a Dropout layer on the outputs of each GRU layer
     *     except the last layer, with dropout probability equal to dropout
     * @param training apply dropout if is true
     * @param bidirectional If true, becomes a bidirectional GRU
     * @param batchFirst If true, then the input and output NDArray are provided as (batch, seq,
     *     feature)
     * @return the output of the operation
     */
    NDList gru(
            NDArray input,
            NDArray state,
            NDList params,
            boolean hasBiases,
            int numLayers,
            double dropRate,
            boolean training,
            boolean bidirectional,
            boolean batchFirst);

    /**
     * Applies LSTM operation to input data.
     *
     * @param input the inputs to the LSTM operation.
     * @param states the hidden state and cell state to the LSTM operation.
     * @param params all params (weights and biases) for the LSTM operation
     * @param hasBiases If false, then the recurrent operation does not use bias weights b_ih and
     *     b_hh
     * @param numLayers the number of recurrent layers.
     * @param dropRate If non-zero, introduces a Dropout layer on the outputs of each LSTM layer
     *     except the last layer, with dropout probability equal to dropout
     * @param training apply dropout if is true
     * @param bidirectional If true, becomes a bidirectional LSTM
     * @param batchFirst If true, then the input and output NDArray are provided as (batch, seq,
     *     feature)
     * @return the output of the operation
     */
    NDList lstm(
            NDArray input,
            NDList states,
            NDList params,
            boolean hasBiases,
            int numLayers,
            double dropRate,
            boolean training,
            boolean bidirectional,
            boolean batchFirst);

    /*
    // Image and CV
    */

    /**
     * Normalizes a NDArray of shape CHW or NCHW with mean and standard deviation.
     *
     * <p>Given mean `(m1, ..., mn)` and std `(s\ :sub:`1`\ , ..., s\ :sub:`n`)` for `n` channels,
     * this transform normalizes each channel of the input tensor with: output[i] = (input[i] - m\
     * :sub:`i`\ ) / s\ :sub:`i`
     *
     * @param mean the mean value for each channel
     * @param std the standard deviation for each channel
     * @return the result of normalization
     */
    default NDArray normalize(float[] mean, float[] std) {
        NDManager manager = getArray().getManager();
        int dim = getArray().getShape().dimension();
        Shape shape = (dim == 3) ? new Shape(3, 1, 1) : new Shape(1, 3, 1, 1);
        try (NDArray meanArr = manager.create(mean, shape);
                NDArray stdArr = manager.create(std, shape)) {
            return getArray().sub(meanArr).divi(stdArr);
        }
    }

    default NDArray toTensor() {
        NDManager manager = getArray().getManager();
        try (NDManager subManager = manager.newSubManager()) {
            NDArray array = getArray();
            array.attach(subManager);

            NDArray result = array;
            int dim = result.getShape().dimension();
            if (dim == 3) {
                result = result.expandDims(0);
            }
            // For Apple Silicon MPS it is important not to switch to 64-bit float here
            if (result.getDataType() == DataType.FLOAT32) {
                result = result.div(255.0f).transpose(0, 3, 1, 2);
            } else {
                result = result.div(255.0).transpose(0, 3, 1, 2);
            }
            if (dim == 3) {
                result = result.squeeze(0);
            }
            // The network by default takes float32
            if (!result.getDataType().equals(DataType.FLOAT32)) {
                result = result.toType(DataType.FLOAT32, false);
            }
            array.attach(manager);
            result.attach(manager);
            return result;
        }
    }

    NDArray interpolation(long[] size, int mode, boolean alignCorners);

    NDArray resize(int width, int height, int interpolation);

    default NDArray crop(int x, int y, int width, int height) {
        NDArray array = getArray();
        StringBuilder sb = new StringBuilder(30);
        if (array.getShape().dimension() == 4) {
            sb.append(":,");
        }
        sb.append(y)
                .append(':')
                .append(y + height)
                .append(',')
                .append(x)
                .append(':')
                .append(x + width)
                .append(",:");
        return array.get(sb.toString());
    }

    // TODO: default can be implemented by using np.flip
    NDArray randomFlipLeftRight();

    // TODO: default can be implemented by using np.flip
    NDArray randomFlipTopBottom();

    // TODO: add TorchVision support
    NDArray randomBrightness(float brightness);

    // TODO: add TorchVision support
    NDArray randomHue(float hue);

    // TODO: add TorchVision support
    NDArray randomColorJitter(float brightness, float contrast, float saturation, float hue);

    /*
    // Miscellaneous
    */

    /**
     * Returns an {@link NDArrayIndexer}.
     *
     * @param manager the manager used to create the arrays
     * @return an {@link NDArrayIndexer}
     */
    NDArrayIndexer getIndexer(NDManager manager);

    /**
     * Returns elements chosen from the {@code NDArray} or the other {@code NDArray} depending on
     * condition.
     *
     * <p>Given three {@code NDArray}s, condition, this, and other, returns an {@code NDArray} with
     * the elements from this or other, depending on whether the elements from condition {@code
     * NDArray} are {@code true} or {@code false}. If condition has the same shape as this, each
     * element in the output {@code NDArray} is from this if the corresponding element in the
     * condition is {@code true}, and from other if {@code false}.
     *
     * <p>Note that all non-zero values are interpreted as {@code true} in condition {@link
     * NDArray}.
     *
     * @param condition the condition {@code NDArray}
     * @param other the other {@code NDArray}
     * @return the result {@code NDArray}
     */
    NDArray where(NDArray condition, NDArray other);

    /**
     * Joins a sequence of {@code NDArray}s in {@link NDList} along a new axis.
     *
     * <p>The axis parameter specifies the index of the new axis in the dimensions of the result.
     * For example, if axis=0 it will be the first dimension and if axis=-1 it will be the last
     * dimension.
     *
     * @param arrays the input {@link NDList}. Each {@code NDArray} in the {@link NDList} must have
     *     the same shape as the {@code NDArray}
     * @param axis the axis in the result {@code NDArray} along which the input {@link NDList} are
     *     stacked
     * @return the result {@code NDArray}. The stacked {@code NDArray} has one more dimension than
     *     the the {@code NDArray}
     */
    NDArray stack(NDList arrays, int axis);

    /**
     * Joins a sequence of {@code NDArray}s in {@link NDList} along first axis.
     *
     * @param arrays the input {@link NDList}. Each {@code NDArray} in the {@link NDList} must have
     *     the same shape as the {@code NDArray}
     * @return the result {@code NDArray}. The stacked {@code NDArray} has one more dimension than
     *     the {@code NDArray}s in {@link NDList}
     */
    default NDArray stack(NDList arrays) {
        return stack(arrays, 0);
    }

    /**
     * Joins a {@link NDList} along an existing axis.
     *
     * @param arrays a {@link NDList} which have the same shape as the {@code NDArray}, except in
     *     the dimension corresponding to axis
     * @param axis the axis along which the {@link NDList} will be joined
     * @return the concatenated {@code NDArray}
     */
    NDArray concat(NDList arrays, int axis);

    /**
     * Joins a {@link NDList} along first axis.
     *
     * @param arrays a {@link NDList} which have the same shape as the {@code NDArray}, except in
     *     the dimension corresponding to axis
     * @return the concatenated {@code NDArray}
     */
    default NDArray concat(NDList arrays) {
        return concat(arrays, 0);
    }

    /**
     * Computes Multibox training targets.
     *
     * @param inputs a NDList of (anchors, labels, and class prediction)
     * @param iouThreshold the anchor-GroundTruth overlap threshold to be regarded as a positive
     *     match
     * @param ignoreLabel the label for ignored anchors
     * @param negativeMiningRatio the max negative to positive samples ratio, use -1 to disable
     *     mining
     * @param negativeMiningThreshold the threshold used for negative mining
     * @param minNegativeSamples the minimum number of negative samples
     * @return an NDList of (bounding box labels, bounding box masks, class labels)
     */
    NDList multiBoxTarget(
            NDList inputs,
            float iouThreshold,
            float ignoreLabel,
            float negativeMiningRatio,
            float negativeMiningThreshold,
            int minNegativeSamples);

    /**
     * Generate prior(anchor) boxes from data, sizes and ratios.
     *
     * @param sizes List of sizes of generated MultiBoxPriores
     * @param ratios List of aspect ratios of generated MultiBoxPriores
     * @param steps Priorbox step across y and x, -1 for auto calculation
     * @param offsets Priorbox center offsets, y and x respectively
     * @param clip Whether to clip out-of-boundary boxes
     * @return an NDList of anchor boxes
     */
    NDList multiBoxPrior(
            List<Float> sizes,
            List<Float> ratios,
            List<Float> steps,
            List<Float> offsets,
            boolean clip);

    /**
     * Converts multi-box detection predictions.
     *
     * @param inputs a NDList of (anchors, labels, and class prediction) in that order
     * @param clip whether to clip out-of-boundary boxes
     * @param threshold the threshold to be a positive prediction
     * @param backgroundId the background id
     * @param nmsThreshold the non-maximum suppression threshold
     * @param forceSuppress whether to suppress all detections regardless of class_id
     * @param nmsTopK the number of detections to keep before NMS, -1 for no limit
     * @return an NDList
     */
    NDList multiBoxDetection(
            NDList inputs,
            boolean clip,
            float threshold,
            int backgroundId,
            float nmsThreshold,
            boolean forceSuppress,
            int nmsTopK);

    /**
     * Fused root-mean-square layer normalization: {@code x / rms(x) * weight} computed as a single
     * kernel. The default implementation throws; implement on each engine that can fuse this.
     *
     * @param normalizedShape sizes of the trailing dims to normalise over (usually the feature dim)
     * @param weight affine scale broadcastable over {@code normalizedShape}, or {@code null} for no
     *     affine
     * @param eps variance epsilon added before the square root; callers are expected to pass a
     *     concrete value (typically {@code 1e-6f})
     * @return normalised tensor with the same shape as {@code this}
     */
    default NDArray rmsNorm(long[] normalizedShape, NDArray weight, double eps) {
        throw new UnsupportedOperationException("rmsNorm is not supported by this engine");
    }

    /**
     * Applies relation-biased scaled-dot-product attention in canonical engine layout.
     *
     * <p>The operation evaluates {@code softmax((Q K^T + gather(Q R, relationIds)) * scale +
     * relationBias) V}. Query, key, and value use {@code [batch, heads, tokens, features]}.
     * Relation keys use {@code [1|batch, heads, queryFeatures, relations]}, relation IDs use {@code
     * [queryTokens, keyTokens]} or {@code [batch, queryTokens, keyTokens]}, and relation bias is
     * broadcastable to {@code [batch, heads, queryTokens, keyTokens]}.
     *
     * <p>Engines may use fused forward and backward implementations. The default implementation is
     * a differentiable decomposition and therefore also defines the portable fallback semantics.
     *
     * @param key key tensor
     * @param value value tensor
     * @param relationKeys relation-key projections
     * @param relationBias additive pairwise relation bias
     * @param relationIds relation ID for each query-key pair
     * @param scale scale applied to content and relation-key dot products
     * @return attended values shaped {@code [batch, heads, queryTokens, valueFeatures]}
     */
    default NDArray canonicalRelationBiasedScaledDotProductAttention(
            NDArray key,
            NDArray value,
            NDArray relationKeys,
            NDArray relationBias,
            NDArray relationIds,
            double scale) {
        NDArray query = getArray();
        NDManager outputManager = query.getManager();
        try (NDManager scope = outputManager.newSubManager()) {
            scope.tempAttachAll(query, key, value, relationKeys, relationBias, relationIds);
            Shape queryShape = query.getShape();
            Shape keyShape = key.getShape();
            Shape valueShape = value.getShape();
            if (queryShape.dimension() != 4
                    || keyShape.dimension() != 4
                    || valueShape.dimension() != 4) {
                throw new IllegalArgumentException("query, key, and value must be rank 4");
            }
            long batch = queryShape.get(0);
            long heads = queryShape.get(1);
            long queryTokens = queryShape.get(2);
            long keyTokens = keyShape.get(2);
            if (keyShape.get(0) != batch
                    || valueShape.get(0) != batch
                    || keyShape.get(1) != heads
                    || valueShape.get(1) != heads
                    || keyShape.get(3) != queryShape.get(3)
                    || valueShape.get(2) != keyTokens) {
                throw new IllegalArgumentException("query, key, and value shapes are incompatible");
            }

            NDArray storedRelationIds = relationIds.toType(DataType.INT64, false);
            NDArray relationIndices;
            if (storedRelationIds.getShape().dimension() == 2) {
                relationIndices =
                        storedRelationIds
                                .reshape(1, 1, queryTokens, keyTokens)
                                .broadcast(batch, heads, queryTokens, keyTokens);
            } else if (storedRelationIds.getShape().dimension() == 3) {
                relationIndices =
                        storedRelationIds
                                .reshape(batch, 1, queryTokens, keyTokens)
                                .broadcast(batch, heads, queryTokens, keyTokens);
            } else {
                throw new IllegalArgumentException("relation IDs must be rank 2 or 3");
            }

            NDArray relationScores = query.matMul(relationKeys).gather(relationIndices, 3);
            NDArray contentScores = query.matMul(key.swapAxes(2, 3));
            NDArray result =
                    contentScores
                            .add(relationScores)
                            .mul(scale)
                            .add(relationBias)
                            .softmax(3)
                            .matMul(value);
            outputManager.attachAll(result);
            return result;
        }
    }

    /**
     * Applies grouped attention in canonical packed engine layout.
     *
     * <p>Consecutive queries share one packed key/value table. Query-specific packed deltas are
     * added to every shared token. Each auxiliary token adds a query-specific delta to one indexed
     * shared token. Index zero means that the auxiliary token is absent; positive indices are
     * one-based shared-token indices. Packed tensors store all head keys followed by all head
     * values in their last dimension.
     *
     * <p>This representation avoids materializing a complete key/value table for every query. It is
     * useful for candidate sets, graph neighborhoods, beam states, and other workloads in which
     * many queries read mostly shared memory.
     *
     * @param sharedKeyValues packed shared data shaped {@code [group, sharedTokens, packedWidth]}
     * @param sharedDeltas query-specific shared-token deltas shaped {@code [query, sharedTokens,
     *     packedWidth]}
     * @param indexedDeltas auxiliary-token deltas shaped {@code [query, indexedTokens,
     *     packedWidth]}
     * @param indexedSharedIds one-based shared-token IDs shaped {@code [query, indexedTokens]};
     *     zero denotes padding
     * @param queriesPerGroup consecutive query count sharing one group
     * @param scale attention score scale
     * @return attended values shaped {@code [query, heads, valueFeatures]}
     */
    default NDArray canonicalGroupedIndexedScaledDotProductAttention(
            NDArray sharedKeyValues,
            NDArray sharedDeltas,
            NDArray indexedDeltas,
            NDArray indexedSharedIds,
            long queriesPerGroup,
            double scale) {
        NDArray query = getArray();
        NDManager outputManager = query.getManager();
        try (NDManager scope = outputManager.newSubManager()) {
            scope.tempAttachAll(
                    query, sharedKeyValues, sharedDeltas, indexedDeltas, indexedSharedIds);
            Shape queryShape = query.getShape();
            Shape sharedShape = sharedKeyValues.getShape();
            Shape sharedDeltaShape = sharedDeltas.getShape();
            Shape indexedDeltaShape = indexedDeltas.getShape();
            Shape indexedIdShape = indexedSharedIds.getShape();
            if (queryShape.dimension() != 3
                    || sharedShape.dimension() != 3
                    || sharedDeltaShape.dimension() != 3
                    || indexedDeltaShape.dimension() != 3
                    || indexedIdShape.dimension() != 2) {
                throw new IllegalArgumentException(
                        "grouped indexed attention inputs have invalid rank");
            }

            long queryCount = queryShape.get(0);
            long heads = queryShape.get(1);
            long keyFeatures = queryShape.get(2);
            long groupCount = sharedShape.get(0);
            long sharedTokens = sharedShape.get(1);
            long packedWidth = sharedShape.get(2);
            long indexedTokens = indexedDeltaShape.get(1);
            long keyWidth = heads * keyFeatures;
            if (queriesPerGroup <= 0
                    || queryCount != groupCount * queriesPerGroup
                    || sharedDeltaShape.get(0) != queryCount
                    || sharedDeltaShape.get(1) != sharedTokens
                    || sharedDeltaShape.get(2) != packedWidth
                    || indexedDeltaShape.get(0) != queryCount
                    || indexedDeltaShape.get(2) != packedWidth
                    || indexedIdShape.get(0) != queryCount
                    || indexedIdShape.get(1) != indexedTokens
                    || packedWidth <= keyWidth
                    || (packedWidth - keyWidth) % heads != 0) {
                throw new IllegalArgumentException(
                        "grouped indexed attention shapes are incompatible");
            }
            long valueFeatures = (packedWidth - keyWidth) / heads;

            NDArray querySharedKeyValues =
                    sharedKeyValues
                            .reshape(groupCount, 1, sharedTokens, packedWidth)
                            .broadcast(groupCount, queriesPerGroup, sharedTokens, packedWidth)
                            .reshape(queryCount, sharedTokens, packedWidth);
            NDArray storedIds = indexedSharedIds.toType(DataType.INT64, false);
            NDArray gatherIndices =
                    storedIds
                            .sub(1)
                            .maximum(0)
                            .expandDims(2)
                            .broadcast(queryCount, indexedTokens, packedWidth);
            NDArray indexedSharedKeyValues = querySharedKeyValues.gather(gatherIndices, 1);

            NDArray sharedKeys =
                    querySharedKeyValues
                            .get("...,0:" + keyWidth)
                            .reshape(queryCount, sharedTokens, heads, keyFeatures)
                            .swapAxes(1, 2);
            NDArray sharedDeltaKeys =
                    sharedDeltas
                            .get("...,0:" + keyWidth)
                            .reshape(queryCount, sharedTokens, heads, keyFeatures)
                            .swapAxes(1, 2);
            NDArray indexedKeys =
                    indexedSharedKeyValues
                            .get("...,0:" + keyWidth)
                            .reshape(queryCount, indexedTokens, heads, keyFeatures)
                            .swapAxes(1, 2);
            NDArray indexedDeltaKeys =
                    indexedDeltas
                            .get("...,0:" + keyWidth)
                            .reshape(queryCount, indexedTokens, heads, keyFeatures)
                            .swapAxes(1, 2);
            NDArray keys =
                    sharedKeys.add(sharedDeltaKeys).concat(indexedKeys.add(indexedDeltaKeys), 2);
            NDArray scores = query.expandDims(2).mul(keys).sum(new int[] {3}).mul(scale);
            NDArray indexedMask =
                    storedIds
                            .neq(0)
                            .toType(scores.getDataType(), false)
                            .expandDims(1)
                            .broadcast(queryCount, heads, indexedTokens)
                            .neg()
                            .add(1)
                            .mul(-1.0e9f);
            scores =
                    scores.get("...,0:" + sharedTokens)
                            .concat(scores.get("...," + sharedTokens + ":").add(indexedMask), 2);
            NDArray weights = scores.softmax(2);

            NDArray sharedValues =
                    querySharedKeyValues
                            .get("...," + keyWidth + ":")
                            .reshape(queryCount, sharedTokens, heads, valueFeatures)
                            .swapAxes(1, 2);
            NDArray sharedDeltaValues =
                    sharedDeltas
                            .get("...," + keyWidth + ":")
                            .reshape(queryCount, sharedTokens, heads, valueFeatures)
                            .swapAxes(1, 2);
            NDArray indexedValues =
                    indexedSharedKeyValues
                            .get("...," + keyWidth + ":")
                            .reshape(queryCount, indexedTokens, heads, valueFeatures)
                            .swapAxes(1, 2);
            NDArray indexedDeltaValues =
                    indexedDeltas
                            .get("...," + keyWidth + ":")
                            .reshape(queryCount, indexedTokens, heads, valueFeatures)
                            .swapAxes(1, 2);
            NDArray values =
                    sharedValues
                            .add(sharedDeltaValues)
                            .concat(indexedValues.add(indexedDeltaValues), 2);
            NDArray result = weights.expandDims(3).mul(values).sum(new int[] {2});
            outputManager.attachAll(result);
            return result;
        }
    }

    /**
     * Adds an inference residual in place and applies affine LayerNorm to the updated value.
     *
     * <p>The returned array is normalized while {@code this} retains the unnormalized sum. Engines
     * may fuse the addition and normalization for supported layouts. This operation mutates its
     * receiver and is intended for inference graphs with explicit buffer ownership.
     *
     * @param update value added to the residual
     * @param weight LayerNorm affine scale
     * @param bias LayerNorm affine bias
     * @param eps normalization epsilon
     * @return normalized updated residual
     */
    default NDArray addToOwnedResidualAndLayerNorm(
            NDArray update, NDArray weight, NDArray bias, float eps) {
        NDArray residual = getArray();
        NDManager outputManager = residual.getManager();
        try (NDManager scope = outputManager.newSubManager()) {
            scope.tempAttachAll(residual, update, weight, bias);
            residual.addi(update);
            Shape shape = residual.getShape();
            NDArray result =
                    layerNorm(
                                    residual,
                                    new Shape(shape.get(shape.dimension() - 1)),
                                    weight,
                                    bias,
                                    eps)
                            .get(0);
            outputManager.attachAll(result);
            return result;
        }
    }

    /**
     * Adds masked embedding rows to an owned token buffer and applies its token mask in place.
     *
     * @param storedIndices one or two stored-index arrays
     * @param embeddingTable embedding lookup table
     * @param validMask token-validity mask containing zero or one
     * @param paddingIndex index excluded from the embedding reduction
     * @param reduction reduction applied to valid embedding rows
     * @return the valid mask cast to the token data type
     */
    default NDArray addMaskedEmbeddingResidualToOwnedTokens(
            NDList storedIndices,
            NDArray embeddingTable,
            NDArray validMask,
            long paddingIndex,
            EmbeddingReduction reduction) {
        NDArray tokens = getArray();
        NDManager outputManager = tokens.getManager();
        try (NDManager scope = outputManager.newSubManager()) {
            scope.tempAttachAll(tokens, embeddingTable, storedIndices, validMask);
            NDArray convertedValidMask =
                    validMask.toType(tokens.getDataType(), false).stopGradient();
            NDArray identity = null;
            NDArray validCount = null;
            for (NDArray storedIndex : storedIndices) {
                NDArray present =
                        storedIndex
                                .neq(paddingIndex)
                                .toType(tokens.getDataType(), false)
                                .stopGradient();
                NDArray embedded =
                        Embedding.embedding(
                                        storedIndex.toType(DataType.INT32, false),
                                        embeddingTable,
                                        SparseFormat.DENSE)
                                .singletonOrThrow()
                                .mul(present.expandDims(2));
                identity = identity == null ? embedded : identity.add(embedded);
                if (reduction == EmbeddingReduction.MEAN_VALID) {
                    validCount = validCount == null ? present : validCount.add(present);
                }
            }
            if (reduction == EmbeddingReduction.MEAN_VALID) {
                identity = identity.div(validCount.maximum(1.0f).expandDims(2).stopGradient());
            }
            tokens.addi(identity).muli(convertedValidMask.expandDims(2));
            outputManager.attachAll(convertedValidMask);
            return convertedValidMask;
        }
    }

    /**
     * Fused scaled-dot-product attention: roughly {@code softmax(Q Kᵀ / sqrt(d_k) + attnMask) V}
     * computed in a single fused kernel.
     *
     * <p>Engine backends may dispatch to FlashAttention, memory-efficient attention or the standard
     * math implementation depending on input shapes and hardware. The default implementation
     * throws; implement on each engine that can fuse this.
     *
     * @param key key tensor, shape {@code [..., K, D]}. same leading dims as the query
     * @param value value tensor, shape {@code [..., K, D_v]}. same leading dims as the query
     * @param attnMask additive float bias broadcastable over {@code [..., Q, K]}, or {@code null}
     * @param dropoutP dropout probability (use {@code 0.0} at inference time)
     * @param isCausal if true, apply a causal upper-triangular mask; mutually exclusive with {@code
     *     attnMask}
     * @return attention output with shape {@code [..., Q, D_v]}
     */
    default NDArray scaledDotProductAttention(
            NDArray key, NDArray value, NDArray attnMask, double dropoutP, boolean isCausal) {
        throw new UnsupportedOperationException(
                "scaledDotProductAttention is not supported by this engine");
    }

    /**
     * Applies scaled-dot-product attention with an explicit score scale.
     *
     * <p>This overload has the same broadcasting and masking semantics as {@link
     * #scaledDotProductAttention(NDArray, NDArray, NDArray, double, boolean)}, but uses {@code
     * scale} instead of {@code 1 / sqrt(keyFeatures)}.
     *
     * @param key key tensor, shape {@code [..., K, D]}
     * @param value value tensor, shape {@code [..., K, D_v]}
     * @param attnMask additive float bias broadcastable over {@code [..., Q, K]}, or {@code null}
     * @param dropoutP dropout probability
     * @param isCausal whether to apply a causal mask
     * @param scale multiplier applied to query-key scores
     * @return attention output with shape {@code [..., Q, D_v]}
     */
    default NDArray scaledDotProductAttention(
            NDArray key,
            NDArray value,
            NDArray attnMask,
            double dropoutP,
            boolean isCausal,
            double scale) {
        throw new UnsupportedOperationException(
                "scaledDotProductAttention with explicit scale is not supported by this engine");
    }

    /**
     * Get internal {@link NDArray}.
     *
     * @return a NDArray
     */
    NDArray getArray();
}
