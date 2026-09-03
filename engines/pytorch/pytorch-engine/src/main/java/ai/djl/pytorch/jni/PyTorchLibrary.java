/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.pytorch.jni;

import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.Set;

/** A class containing utilities to interact with the PyTorch Engine's JNI layer. */
final class PyTorchLibrary {

    static final PyTorchLibrary LIB = new PyTorchLibrary();

    private PyTorchLibrary() {}

    native boolean torchIsGradMode();

    native void torchSetGradMode(boolean enable);

    native long torchOpenInferenceMode();

    native void torchCloseInferenceMode(long handle);

    native long torchOpenStreamScope(int[] device);

    native void torchCloseStreamScope(long handle);

    native long torchCreateDeviceStream(int[] device);

    native long torchOpenDeviceStream(long handle);

    native void torchDeleteDeviceStream(long handle);

    native long torchCreateDeviceEvent(int[] device);

    native void torchRecordDeviceEvent(long handle);

    native void torchWaitDeviceEvent(long handle);

    native boolean torchQueryDeviceEvent(long handle);

    native void torchSynchronizeDeviceEvent(long handle);

    native void torchDeleteDeviceEvent(long handle);

    native long torchCreateAcceleratorGraph(int[] device);

    native void torchBeginAcceleratorGraphCapture(long handle);

    native void torchEndAcceleratorGraphCapture(long handle);

    native void torchReplayAcceleratorGraph(long handle);

    native void torchDeleteAcceleratorGraph(long handle);

    native int torchGetFusionBackend();

    native long torchPrepareFusionPlan(int[] device, ByteBuffer descriptor);

    native long[] torchGetFusionPlanStats(long planHandle);

    native long torchBindFusionPlan(long planHandle, ByteBuffer constantHandles);

    native long torchCreateFusionSession(long executableHandle, int bufferCount);

    native long torchGetFusionSessionOutput(long sessionHandle, int bufferIndex, int outputIndex);

    native void torchSubmitFusion(
            long sessionHandle, int bufferIndex, ByteBuffer inputHandles, ByteBuffer dimensions);

    native void torchSynchronizeFusionOutput(long sessionHandle, int bufferIndex);

    native void torchDeleteFusionPlan(long handle);

    native void torchDeleteFusionExecutable(long handle);

    native void torchDeleteFusionSession(long handle);

    native boolean torchAutocastIsEnabled(int deviceType);

    native void torchAutocastSetEnabled(int deviceType, boolean enabled);

    native int torchAutocastGetDtype(int deviceType);

    native void torchAutocastSetDtype(int deviceType, int dtype);

    native boolean torchAutocastIsCacheEnabled();

    native void torchAutocastSetCacheEnabled(boolean enabled);

    native void torchAutocastClearCache();

    native int torchAutocastIncrementNesting();

    native int torchAutocastDecrementNesting();

    native int torchGetNumInteropThreads();

    native int torchGetNumThreads();

    native void torchSetNumInteropThreads(int threads);

    native void torchSetNumThreads(int threads);

    native void torchSetBenchmarkCuDNN(boolean enable);

    native void torchManualSeed(long seed);

    native void torchShowConfig(Set<String> set);

    native int torchGetGpuCount();

    native long[] torchGetMemoryStats(int deviceId);

    native void torchResetPeakMemoryStats(int deviceId);

    native void torchStartProfile(boolean useCuda, boolean recordShape, boolean profileMemory);

    native void torchStopProfile(String outputFile);

    native long[] torchSizes(long handle);

    native byte[] torchDataPtr(long handle);

    native ByteBuffer torchDirectByteBuffer(long handle);

    native boolean torchIsContiguous(long handle);

    native long torchToContiguous(long handle);

    native int torchDType(long handle);

    native int[] torchDevice(long handle);

    native int torchLayout(long handle);

    native long torchTo(long handle, int dType, int[] device);

    native long torchDifferentiableCast(long handle, int dataType);

    native long torchGetItem(long handle, long index);

    native long torchGetItem(long handle, long[] indices);

    native long torchToSparse(long handle);

    native long torchToDense(long handle);

    native long tensorClone(long handle);

    native void torchCudaEmptyCache();

    native long torchEmpty(long[] shape, int dType, int layout, int[] device, boolean requiredGrad);

    native long torchZeros(long[] shape, int dType, int layout, int[] device, boolean requiredGrad);

    native long torchOnes(long[] shape, int dType, int layout, int[] device, boolean requiredGrad);

    native long torchFull(
            long[] shape,
            double fillValue,
            int dType,
            int layout,
            int[] device,
            boolean requiredGrad);

    native long torchZerosLike(
            long handle, int dType, int layout, int[] device, boolean requiredGrad);

    native long torchOnesLike(
            long handle, int dType, int layout, int[] device, boolean requiredGrad);

    native long torchSparseCoo(
            long[] shape, long indicesHandle, long valueHandle, boolean requiredGrad);

    native long torchArange(
            float start,
            float end,
            float step,
            int dType,
            int layout,
            int[] device,
            boolean requiredGrad);

    native long torchLinspace(
            float start,
            float end,
            int step,
            int dType,
            int layout,
            int[] device,
            boolean requiredGrad);

    native long torchAdd(long self, long other);

    native long torchAddScalar(
            long self, long integerValue, double floatingValue, boolean floating);

    native void torchAddi(long self, long other);

    native void torchAddiScalar(
            long self, long integerValue, double floatingValue, boolean floating);

    native long torchExpand(long self, long[] shape);

    native long torchSub(long self, long other);

    native long torchSubScalar(
            long self, long integerValue, double floatingValue, boolean floating);

    native void torchSubi(long self, long other);

    native void torchSubiScalar(
            long self, long integerValue, double floatingValue, boolean floating);

    native void torchFill(long self, double value);

    native long torchMul(long self, long other);

    native long torchMulScalar(
            long self, long integerValue, double floatingValue, boolean floating);

    native void torchMuli(long self, long other);

    native void torchMuliScalar(
            long self, long integerValue, double floatingValue, boolean floating);

    native long torchTrueDivide(long self, long other);

    native long torchTrueDivideScalar(
            long self, long integerValue, double floatingValue, boolean floating);

    native void torchTrueDividei(long self, long other);

    native void torchTrueDivideiScalar(
            long self, long integerValue, double floatingValue, boolean floating);

    native long torchRemainder(long self, long other);

    native long torchRemainderScalar(
            long self, long integerValue, double floatingValue, boolean floating);

    native void torchRemainderi(long self, long other);

    native void torchRemainderiScalar(
            long self, long integerValue, double floatingValue, boolean floating);

    native long torchRot90(long self, long k, long[] axes);

    native long torchPow(long self, long exponent);

    native long torchPowScalar(
            long self, long integerValue, double floatingValue, boolean floating);

    native void torchPowi(long self, long exponent);

    native void torchPowiScalar(
            long self, long integerValue, double floatingValue, boolean floating);

    native long torchSign(long self);

    native void torchSigni(long self);

    native long torchMatmul(long self, long other);

    native long torchBmm(long self, long other);

    native long torchXLogY(long self, long other);

    native long torchDot(long self, long other);

    native long torchLogicalAnd(long self, long other);

    native long torchLogicalOr(long self, long other);

    native long torchLogicalXor(long self, long other);

    native long torchLogicalNot(long handle);

    native long torchPad(long handle, long[] shape, double value);

    native long torchReshape(long handle, long[] shape);

    native long torchSoftmax(long handle, long dim, int dType);

    native long torchLogSoftmax(long handle, long dim, int dType);

    native long torchMaskedSoftmax(long logits, long mask, long axis);

    native long torchWeightedRowStatistics(long values, long weights);

    native long torchGroupedMaskedSoftmaxPool(long logits, long mask, long values);

    native long torchIndexedMaskedSoftmaxPool(
            long logits, long mask, long values, int[] choiceIndices);

    native long torchMaskedLogSumExp(long logits, long mask, long axis);

    native long torchCategoricalMasks(
            long categories, long mask, int[] fieldIndices, long[] categorySets);

    native long torchBinaryChoiceMasks(
            long routes,
            long firstMask,
            long secondMask,
            int representativeField,
            int firstRouteField,
            int secondRouteField,
            long paddingValue);

    native long torchScaledDotProductAttention(
            long query,
            long key,
            long value,
            long mask,
            double dropoutP,
            boolean isCausal,
            double scale);

    native long torchIndexedRelationBias(
            long relationLogits, long relationBias, long relationIds, float scale);

    native long torchGroupedPackedScaledDotProductAttention(
            long query, long packedKeyValue, long mask, long heads, float scale);

    native long torchGroupedIndexedScaledDotProductAttention(
            long query,
            long sharedKeyValues,
            long sharedDeltas,
            long indexedDeltas,
            long indexedSharedIds,
            long queriesPerGroup,
            float scale);

    native long torchMappedGroupedIndexedScaledDotProductAttention(
            long query,
            long sharedKeyValues,
            long sharedGroupIndices,
            long sharedDeltaTable,
            long sharedDeltaIndices,
            long indexedDeltas,
            long indexedSharedIds,
            float scale);

    native long torchAddToOwnedResidualAndLayerNorm(
            long residual, long update, long weight, long bias, float epsilon);

    native long torchAddMaskedEmbeddingResidualToOwnedTokens(
            long tokens,
            long[] storedIndices,
            long embeddingTable,
            long validMask,
            long paddingIndex,
            int reduction);

    native void torchAddBroadcastResidualToOwnedAndSilu(long values, long residual, long mask);

    native void torchAddBiasAndBroadcastResidualToOwnedAndSilu(
            long values, long bias, long residual, long mask);

    native long torchRmsNorm(long input, long[] normalizedShape, long weight, double eps);

    native long torchArgMax(long handle);

    native long torchArgMax(long handle, long dim, boolean keepDim);

    native long[] torchTopK(long handle, long k, long axis, boolean largest, boolean sorted);

    native long torchArgMin(long handle);

    native long torchArgMin(long handle, long dim, boolean keepDim);

    native long torchArgSort(long handle, long dim, boolean keepDim);

    native long torchSort(long handle, long dim, boolean descending);

    native long torchPermute(long handle, long[] dims);

    native long torchFlip(long handle, long[] dims);

    native long torchTranspose(long handle, long axis1, long axis2);

    native boolean contentEqual(long handle1, long handle2);

    native long torchFromBlob(
            ByteBuffer data,
            long[] shape,
            int dType,
            int layout,
            int[] device,
            boolean requiredGrad);

    native long torchIndex(long handle, long[] minIndices, long[] maxIndices, long[] stepIndices);

    native void torchIndexPut(
            long handle,
            long valueHandle,
            long[] minIndices,
            long[] maxIndices,
            long[] stepIndices);

    native void torchIndexAdvPut(long handle, long torchIndexHandle, long data);

    native void torchSet(long handle, ByteBuffer data);

    native long torchAllocatePinnedBuffer(long size, int dtype);

    native ByteBuffer torchGetPinnedBuffer(long handle);

    native boolean torchIsPinnedBuffer(long handle);

    native void torchDeletePinnedBuffer(long handle);

    native void torchCopyFromDirectBuffer(long handle, ByteBuffer data);

    native void torchCopyFromPinnedBuffer(long handle, long pinnedBufferHandle);

    native long torchCopyFromPinnedBufferAsync(long handle, long pinnedBufferHandle);

    native void torchEnqueueCopyFrom(long handle, long pinnedBufferHandle);

    native long torchCopyToPinnedBufferAsync(long handle, long pinnedBufferHandle);

    native void torchEnqueueCopyTo(long handle, long pinnedBufferHandle);

    native void torchSynchronizeCopyEvent(long handle);

    native void torchDeleteCopyEvent(long handle);

    native void torchRecordStream(long handle);

    native void torchCopyTo(long sourceHandle, long targetHandle);

    native long torchSlice(long handle, long dim, long start, long end, long step);

    native long torchGather(long handle, long index, long dim, boolean sparseGrad);

    native long torchTake(long handle, long index);

    native long torchPut(long handle, long index, long value);

    native long torchScatter(long handle, long index, long value, int axis);

    native long torchEmbeddingWithOffsets(long rawIds, long offsets, long table);

    native long torchEmbeddingFeaturePack(long rawIds, long offsets, long table, long features);

    native long torchGatherRows(long handle, long index);

    native long torchScatterRows(long handle, long index, long rowCount);

    native long torchSegmentedLookupSum(long handle, long storedIndices);

    native long torchPaddedBatchGather(long handle, long storedIndices);

    native long torchPaddedBatchGather2d(
            long handle, long outerStoredIndices, long innerStoredIndices);

    native long torchPaddedBatchGatherByBatchIndices(
            long handle, long batchIndices, long storedIndices);

    native long torchIndexAdd(long handle, long index, long value, int axis);

    native long torchMaskedSelect(long handle, long maskHandle);

    native void torchMaskedPut(long handle, long valueHandle, long maskHandle);

    native void torchDeleteTensor(long handle);

    native void torchDeleteIndex(long handle);

    native void torchDeleteModule(long handle);

    native void torchDeleteIValue(long handle);

    native long torchMaximum(long self, long other);

    native long torchMaximumScalar(
            long self, long integerValue, double floatingValue, boolean floating);

    native long torchMax(long handle);

    native long torchMax(long handle, long dim, boolean keepDim);

    native long torchMinimum(long self, long other);

    native long torchMinimumScalar(
            long self, long integerValue, double floatingValue, boolean floating);

    native long[] torchMedian(long self, long dim, boolean keepDim);

    native long torchQuantile(long self, float quantile, long dim, boolean keepDim);

    native long torchMin(long handle);

    native long torchMin(long handle, long dim, boolean keepDim);

    native long torchMean(long handle);

    native long torchMean(long handle, long dim, boolean keepDim);

    native long torchSum(long handle);

    native long torchSum(long handle, long[] dim, boolean keepDim);

    native long torchCumProd(long handle, long dim, int dtype);

    native long torchProd(long handle);

    native long torchProd(long handle, long dim, boolean keepDim);

    native long torchCumSum(long handle, long dim);

    native long torchDiagonal(long handle, long offset, long axis1, long axis2);

    native long torchFlatten(long handle, long startDim, long endDim);

    native long torchFft(long handle, long length, long axis);

    native long torchIfft(long handle, long length, long axis);

    native long torchRfft(long handle, long length, long axis);

    native long torchIrfft(long handle, long length, long axis);

    native long torchStft(
            long handle,
            long nFft,
            long hopLength,
            long windowHandle,
            boolean center,
            boolean normalize,
            boolean returnComplex);

    native long torchFft2(long handle, long[] sizes, long[] axes);

    native long torchIfft2(long handle, long[] sizes, long[] axes);

    native long torchViewAsReal(long handle);

    native long torchViewAsComplex(long handle);

    native long conj(long handle);

    native long[] torchSplit(long handle, long size, long dim);

    native long[] torchSplit(long handle, long[] indices, long dim);

    native long torchUnsqueeze(long handle, long dim);

    native long torchSqueeze(long handle);

    native long torchSqueeze(long handle, long axis);

    native long[] torchUnique(
            long handle, long dim, boolean sorted, boolean returnInverse, boolean returnCounts);

    native long torchStack(long[] handles, long dim);

    native long torchCat(long[] handles, long dim);

    native long torchConcatToType(long[] handles, long dim, int dataType);

    native long torchRepeat(long handle, long[] repeats);

    native long torchRepeatInterleave(long handle, long repeat, long axis);

    native long torchAbs(long handle);

    native long torchSquare(long self);

    native long torchFloor(long handle);

    native long torchCeil(long handle);

    native long torchClamp(long handle, long min, long max);

    native long torchRound(long handle);

    native long torchTrunc(long handle);

    native long torchExp(long handle);

    native long torchLgamma(long handle);

    native long torchLog(long handle);

    native long torchLog10(long handle);

    native long torchLog2(long handle);

    native long torchSin(long handle);

    native long torchCos(long handle);

    native long torchTan(long handle);

    native long torchASin(long handle);

    native long torchAcos(long handle);

    native long torchAtan(long handle);

    native long torchAtan2(long self, long other);

    native long torchSqrt(long handle);

    native long torchSinh(long handle);

    native long torchCosh(long handle);

    native long torchTanh(long handle);

    native long torchSigmoid(long handle);

    native long torchSilu(long handle);

    native long torchWhere(long handle, long x, long y);

    native long torchAll(long self);

    native long torchAny(long self);

    native long torchNone(long self);

    native long torchEq(long self, long other);

    native long torchEqScalar(long self, long integerValue, double floatingValue, boolean floating);

    native long torchNeq(long self, long other);

    native long torchNeqScalar(
            long self, long integerValue, double floatingValue, boolean floating);

    native long torchGt(long self, long other);

    native long torchGtScalar(long self, long integerValue, double floatingValue, boolean floating);

    native long torchGte(long self, long other);

    native long torchGteScalar(
            long self, long integerValue, double floatingValue, boolean floating);

    native long torchLt(long self, long other);

    native long torchLtScalar(long self, long integerValue, double floatingValue, boolean floating);

    native long torchLte(long self, long other);

    native long torchLteScalar(
            long self, long integerValue, double floatingValue, boolean floating);

    native long torchNeg(long self);

    native void torchNegi(long self);

    native long torchIsNaN(long self);

    native long torchIsInf(long self);

    native long torchRandint(
            long low,
            long high,
            long[] sizes,
            int dType,
            int layout,
            int[] device,
            boolean requiredGrad);

    native long torchRandPerm(long n, int dType, int layout, int[] device, boolean requireGrad);

    native long torchNormal(
            double mean,
            double std,
            long[] sizes,
            int dType,
            int layout,
            int[] device,
            boolean requiredGrad);

    native long tensorUniform(
            double from,
            double to,
            long[] sizes,
            int dType,
            int layout,
            int[] device,
            boolean requiredGrad);

    native long torchEye(int n, int m, int dType, int layout, int[] device, boolean requiredGrad);

    native long torchHannWindow(long nfft, boolean periodic, int[] device);

    native long torchErfinv(long handle);

    native long torchErf(long handle);

    native long torchInverse(long self);

    native long torchNNInterpolate(long handle, long[] size, int mode, boolean alignCorners);

    native long torchNNLinear(long handle, long weightHandle, long biasHandle);

    native long torchNNProjectedResidualMlp(
            long inputHandle,
            long combinedWeightHandle,
            long combinedBiasHandle,
            long outputWeightHandle);

    native long torchNNEmbedding(long handle, long weightHandle, boolean sparse);

    native long torchNNRelu(long handle);

    native long torchNNSoftPlus(long handle);

    native long torchNNSoftSign(long handle);

    native long torchNNLeakyRelu(long handle, double negativeSlope);

    native long torchNNElu(long handle, double alpha);

    native long torchNNSelu(long handle);

    native long torchNNGelu(long handle);

    native long torchNNConvNd(
            long inputHandle,
            long weightHandle,
            long biasHandle,
            long[] stride,
            long[] padding,
            long[] dilation,
            int groups);

    native long torchNNDropout(long inputHandle, double probability, boolean isTrain);

    native long torchNNNormalize(long inputHandle, double p, long dim, double eps);

    native long torchNNLayerNorm(
            long inputHandle,
            long[] normalizedShape,
            long weigthHandle,
            long biasHandle,
            double eps);

    native long[] torchNNResidualAddLayerNorm(
            long residualHandle,
            long updateHandle,
            long[] normalizedShape,
            long weightHandle,
            long biasHandle,
            double eps);

    native long[] torchNNLayerNormAndCast(
            long inputHandle,
            long[] normalizedShape,
            long weightHandle,
            long biasHandle,
            double eps,
            int convertedDataType);

    native long torchNNBatchNorm(
            long inputHandle,
            long runningMeanHandle,
            long runningVarHandle,
            long weigthHandle,
            long biasHandle,
            boolean training,
            double momentum,
            double eps);

    native long[] torchNNRnn(
            long inputHandle,
            long hxHandle,
            long[] paramHandles,
            boolean hasBiases,
            int numLayers,
            int activation,
            double dropRate,
            boolean training,
            boolean bidirectional,
            boolean batchFirst);

    native long[] torchNNGru(
            long inputHandle,
            long hxHandle,
            long[] paramHandles,
            boolean hasBiases,
            int numLayers,
            double dropRate,
            boolean training,
            boolean bidirectional,
            boolean batchFirst);

    native long[] torchNNLstm(
            long inputHandle,
            long[] hxHandles,
            long[] paramHandles,
            boolean hasBiases,
            int numLayers,
            double dropRate,
            boolean training,
            boolean bidirectional,
            boolean batchFirst);

    native long torchNNAvgPool(
            long inputHandle,
            long[] kernel,
            long[] stride,
            long[] pad,
            boolean useCeil,
            boolean countIncludePad);

    native long torchNNMaxPool(
            long inputHandle, long[] kernelSize, long[] stride, long[] padding, boolean ceilMode);

    native long torchNNAdaptiveAvgPool(long inputHandle, long[] outputSize);

    native long torchNNAdaptiveMaxPool(long inputHandle, long[] outputSize);

    native long torchNNLpPool(
            long inputHandle, double normType, long[] kernelSize, long[] stride, boolean ceilMode);

    native long torchNNOneHot(long inputHandle, int depth);

    native boolean torchRequiresGrad(long inputHandle);

    native String torchGradFnName(long inputHandle);

    native void torchAttachGrad(long inputHandle, boolean requiresGrad);

    native long torchGrad(long inputHandle);

    native long torchDetachGrad(long inputHandle);

    native void torchBackward(
            long inputHandle, long gradHandle, boolean keepGraph, boolean createGraph);

    native long torchCreateFlatGradientAccumulator(long[] parameterHandles, long gradientHandle);

    native void torchFlatGradientAccumulatorBackward(
            long accumulatorHandle, long targetHandle, long targetGradientHandle);

    native void torchZeroFlatGradientAccumulator(long accumulatorHandle);

    native void torchDeleteFlatGradientAccumulator(long accumulatorHandle);

    native long torchCreateFlatGradientPacker(long[] parameterHandles, long destinationHandle);

    native void torchFlatGradientPackerPackAndClear(
            long packerHandle, boolean zeroMissingGradients);

    native void torchFlatGradientPackerAccumulateAndClear(
            long packerHandle, boolean zeroMissingGradients);

    native void torchZeroFlatGradientPackerDestination(long packerHandle);

    native void torchClearFlatGradientPackerParameterGradients(long packerHandle);

    native void torchDeleteFlatGradientPacker(long packerHandle);

    native long distributedCreateReducer(
            long[] parameterHandles,
            String masterHost,
            int masterPort,
            int rank,
            int worldSize,
            int localRank,
            int bucketCapMb,
            boolean staticGraph,
            boolean findUnusedParameters,
            boolean averageGradients);

    native void distributedPrepareForBackward(long reducerHandle);

    native void distributedFinalizeBackward(long reducerHandle);

    native void distributedDeleteReducer(long reducerHandle);

    native long moduleLoad(
            String path,
            int[] device,
            boolean mapLocation,
            String[] extraFileNames,
            String[] extraFileValues,
            boolean trainParam);

    native long moduleLoad(
            InputStream is, int[] device, boolean mapLocation, byte[] buffer, long size);

    native void moduleEval(long handle);

    native void moduleTrain(long handle);

    native long moduleRunMethod(
            long moduleHandle,
            String methodName,
            long[] iValueHandles,
            boolean isTrain,
            boolean separateCudaStream);

    native void setGraphExecutorOptimize(boolean enabled);

    native void moduleWrite(long moduleHandle, OutputStream os, byte[] buffer, boolean writeSize);

    native long[] moduleGetParams(long moduleHandle);

    native String[] moduleGetParamNames(long moduleHandle);

    native String[] moduleGetMethodNames(long moduleHandle);

    native long iValueFromTensor(long tensorHandle);

    native long iValueFromBool(boolean value);

    native long iValueFromLong(long value);

    native long iValueFromDouble(double value);

    native long iValueFromString(String value);

    native long iValueFromBoolList(boolean... value);

    native long iValueFromLongList(long... value);

    native long iValueFromDoubleList(double... value);

    native long iValueFromTensorList(long[] tensorHandles);

    native long iValueFromList(long[] ivalueHandles);

    native long iValueFromTuple(long[] ivalueHandles);

    native long iValueFromStringMap(String[] keys, long[] tensorHandles);

    native long iValueFromStringIValueMap(String[] keys, long[] tensorHandles);

    native long iValueToTensor(long iValueHandle);

    native boolean iValueToBool(long iValueHandle);

    native long iValueToLong(long iValueHandle);

    native double iValueToDouble(long iValueHandle);

    native String iValueToString(long iValueHandle);

    native boolean[] iValueToBoolList(long iValueHandle);

    native long[] iValueToLongList(long iValueHandle);

    native double[] iValueToDoubleList(long iValueHandle);

    native long[] iValueToTensorList(long iValueHandle);

    native long[] iValueToIValueList(long iValueHandle);

    native long[] iValueToIValueTuple(long iValueHandle);

    native long[] iValueToMap(long iValueHandle);

    native String iValueGetType(long iValueHandle);

    native boolean iValueIsTensor(long iValueHandle);

    native boolean iValueIsBool(long iValueHandle);

    native boolean iValueIsLong(long iValueHandle);

    native boolean iValueIsDouble(long iValueHandle);

    native boolean iValueIsString(long iValueHandle);

    native boolean iValueIsBoolList(long iValueHandle);

    native boolean iValueIsLongList(long iValueHandle);

    native boolean iValueIsDoubleList(long iValueHandle);

    native boolean iValueIsTensorList(long iValueHandle);

    native boolean iValueIsList(long iValueHandle);

    native boolean iValueIsTuple(long iValueHandle);

    native boolean iValueIsMap(long iValueHandle);

    native void zeroGrad(long handle);

    native boolean torchUnscaleGradientsAndCheckFinite(long[] gradientHandles, float inverseScale);

    native void adamUpdate(
            long weight,
            long grad,
            long mean,
            long variance,
            float lr,
            float learningRateBiasCorrection,
            float wd,
            float rescaleGrad,
            float clipGrad,
            float beta1,
            float beta2,
            float eps,
            boolean adamw);

    native void sgdUpdate(
            long weight,
            long grad,
            long state,
            float lr,
            float wd,
            float rescaleGrad,
            float clipGrad,
            float momentum);

    native long torchNorm(long handle, int ord, long[] axis, boolean keepDims);

    native long torchNonZeros(long handle);

    native long torchIndexInit(int size);

    native long torchIndexAdvGet(long handle, long torchIndexHandle);

    native void torchIndexAppendNoneEllipsis(long torchIndexHandle, boolean isEllipsis);

    native void torchIndexAppendSlice(
            long torchIndexHandle, long min, long max, long step, int nullSliceBinary);

    native void torchIndexAppendFixed(long torchIndexHandle, long idx);

    native void torchIndexAppendArray(long torchIndexHandle, long arrayHandle);

    native long torchDiff(long self, int n, int dim);
}
