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

import ai.djl.Device;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDScope;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.index.dim.NDIndexAll;
import ai.djl.ndarray.index.dim.NDIndexBooleans;
import ai.djl.ndarray.index.dim.NDIndexElement;
import ai.djl.ndarray.index.dim.NDIndexFixed;
import ai.djl.ndarray.index.dim.NDIndexNull;
import ai.djl.ndarray.index.dim.NDIndexPick;
import ai.djl.ndarray.index.dim.NDIndexSlice;
import ai.djl.ndarray.index.dim.NDIndexTake;
import ai.djl.ndarray.index.full.NDIndexFullPick;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.EmbeddingReduction;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.recurrent.RNN;
import ai.djl.pytorch.engine.PtDeviceType;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtSymbolBlock;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.ListIterator;
import java.util.Set;

/**
 * A class containing utilities to interact with the PyTorch Engine's Java Native Interface (JNI)
 * layer.
 */
@SuppressWarnings("MissingJavadocMethod")
public final class JniUtils {

    private static final Logger logger = LoggerFactory.getLogger(JniUtils.class);

    private static Set<String> configs;

    private static final int NULL_PTR = 0;

    private static final int BYTE_LENGTH = 4194304;

    private JniUtils() {}

    private static boolean isFloatingScalar(Number value) {
        if (value instanceof Float || value instanceof Double) {
            return true;
        }
        if (value instanceof Integer || value instanceof Long || value instanceof Byte) {
            return false;
        }
        throw new IllegalArgumentException(
                "Conversion of " + value.getClass().getName() + " not supported!");
    }

    private static int layoutMapper(SparseFormat fmt, Device device) {
        if (fmt == SparseFormat.DENSE) {
            // Enable MKLDNN with environment variable
            // Using MKLDNN with GPU would throw exception on libtorch
            if (Boolean.getBoolean("ai.djl.pytorch.use_mkldnn") && !device.equals(Device.gpu())) {
                return 2;
            }
            return 0;
        } else if (fmt == SparseFormat.COO) {
            return 1;
        } else {
            throw new IllegalArgumentException(
                    "Current PyTorch only support SparseFormat.DENSE and SparseFormat.COO");
        }
    }

    public static boolean isGradMode() {
        return PyTorchLibrary.LIB.torchIsGradMode();
    }

    public static void setGradMode(boolean enable) {
        PyTorchLibrary.LIB.torchSetGradMode(enable);
    }

    public static long openInferenceMode() {
        return PyTorchLibrary.LIB.torchOpenInferenceMode();
    }

    public static void closeInferenceMode(long handle) {
        PyTorchLibrary.LIB.torchCloseInferenceMode(handle);
    }

    public static long openStreamScope(Device device) {
        return PyTorchLibrary.LIB.torchOpenStreamScope(
                new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()});
    }

    public static void closeStreamScope(long handle) {
        PyTorchLibrary.LIB.torchCloseStreamScope(handle);
    }

    public static long createDeviceStream(Device device) {
        return PyTorchLibrary.LIB.torchCreateDeviceStream(
                new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()});
    }

    public static long openDeviceStream(long handle) {
        return PyTorchLibrary.LIB.torchOpenDeviceStream(handle);
    }

    public static long getDeviceStreamToken(long handle) {
        return PyTorchLibrary.LIB.torchGetDeviceStreamToken(handle);
    }

    public static void deleteDeviceStream(long handle) {
        PyTorchLibrary.LIB.torchDeleteDeviceStream(handle);
    }

    public static long createDeviceEvent(Device device) {
        return PyTorchLibrary.LIB.torchCreateDeviceEvent(
                new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()});
    }

    public static void recordDeviceEvent(long handle) {
        PyTorchLibrary.LIB.torchRecordDeviceEvent(handle);
    }

    public static void waitDeviceEvent(long handle) {
        PyTorchLibrary.LIB.torchWaitDeviceEvent(handle);
    }

    public static boolean queryDeviceEvent(long handle) {
        return PyTorchLibrary.LIB.torchQueryDeviceEvent(handle);
    }

    public static void synchronizeDeviceEvent(long handle) {
        PyTorchLibrary.LIB.torchSynchronizeDeviceEvent(handle);
    }

    public static void deleteDeviceEvent(long handle) {
        PyTorchLibrary.LIB.torchDeleteDeviceEvent(handle);
    }

    public static long createAcceleratorGraph(Device device) {
        return PyTorchLibrary.LIB.torchCreateAcceleratorGraph(
                new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()});
    }

    public static void beginAcceleratorGraphCapture(long handle) {
        PyTorchLibrary.LIB.torchBeginAcceleratorGraphCapture(handle);
    }

    public static void endAcceleratorGraphCapture(long handle) {
        PyTorchLibrary.LIB.torchEndAcceleratorGraphCapture(handle);
    }

    public static void replayAcceleratorGraph(long handle) {
        PyTorchLibrary.LIB.torchReplayAcceleratorGraph(handle);
    }

    public static void deleteAcceleratorGraph(long handle) {
        PyTorchLibrary.LIB.torchDeleteAcceleratorGraph(handle);
    }

    public static long prepareFusionPlan(
            Device device, ByteBuffer descriptor, ByteBuffer profileDescriptor) {
        return PyTorchLibrary.LIB.torchPrepareFusionPlanWithProfiles(
                new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                descriptor,
                profileDescriptor);
    }

    public static long[] getFusionPlanStats(long planHandle, int variantIndex) {
        return PyTorchLibrary.LIB.torchGetFusionPlanVariantStats(planHandle, variantIndex);
    }

    /**
     * Returns the fusion backend compiled into the loaded native library.
     *
     * @return {@code 0} when unavailable, {@code 1} for CUDA, or {@code 2} for ROCm
     */
    public static int getFusionBackend() {
        return PyTorchLibrary.LIB.torchGetFusionBackend();
    }

    public static long bindFusionPlan(long planHandle, ByteBuffer constantHandles) {
        return PyTorchLibrary.LIB.torchBindFusionPlan(planHandle, constantHandles);
    }

    public static long createFusionSession(
            long executableHandle, int variantIndex, int outputSlotCount) {
        return PyTorchLibrary.LIB.torchCreateFusionProfileSession(
                executableHandle, variantIndex, outputSlotCount);
    }

    public static PtNDArray getFusionSessionOutput(
            PtNDManager manager, long sessionHandle, int outputSlotIndex, int outputIndex) {
        long outputHandle =
                PyTorchLibrary.LIB.torchGetFusionSessionOutput(
                        sessionHandle, outputSlotIndex, outputIndex);
        PtNDArray output = new PtNDArray(manager, outputHandle);
        NDScope.unregister(output);
        return output;
    }

    public static void submitFusion(
            long sessionHandle,
            int outputSlotIndex,
            ByteBuffer inputHandles,
            ByteBuffer dimensions) {
        PyTorchLibrary.LIB.torchSubmitFusion(
                sessionHandle, outputSlotIndex, inputHandles, dimensions);
    }

    public static void synchronizeFusionOutput(long sessionHandle, int outputSlotIndex) {
        PyTorchLibrary.LIB.torchSynchronizeFusionOutput(sessionHandle, outputSlotIndex);
    }

    public static void deleteFusionPlan(long handle) {
        PyTorchLibrary.LIB.torchDeleteFusionPlan(handle);
    }

    public static void deleteFusionExecutable(long handle) {
        PyTorchLibrary.LIB.torchDeleteFusionExecutable(handle);
    }

    public static void deleteFusionSession(long handle) {
        PyTorchLibrary.LIB.torchDeleteFusionSession(handle);
    }

    // Autocast state is thread-local. deviceType follows PtDeviceType and dataType uses
    // DataType.ordinal(). ROCm registers autocast under the CUDA device type.

    public static boolean autocastIsEnabled(int deviceType) {
        return PyTorchLibrary.LIB.torchAutocastIsEnabled(deviceType);
    }

    public static void autocastSetEnabled(int deviceType, boolean enabled) {
        PyTorchLibrary.LIB.torchAutocastSetEnabled(deviceType, enabled);
    }

    public static int autocastGetDataType(int deviceType) {
        return PyTorchLibrary.LIB.torchAutocastGetDtype(deviceType);
    }

    public static void autocastSetDataType(int deviceType, int dataType) {
        PyTorchLibrary.LIB.torchAutocastSetDtype(deviceType, dataType);
    }

    /**
     * Returns the autocast data type.
     *
     * @param deviceType the PyTorch device type
     * @return the {@link DataType} ordinal
     * @deprecated Use {@link #autocastGetDataType(int)}.
     */
    @Deprecated
    public static int autocastGetDtype(int deviceType) {
        return autocastGetDataType(deviceType);
    }

    /**
     * Sets the autocast data type.
     *
     * @param deviceType the PyTorch device type
     * @param dataType the {@link DataType} ordinal
     * @deprecated Use {@link #autocastSetDataType(int, int)}.
     */
    @Deprecated
    public static void autocastSetDtype(int deviceType, int dataType) {
        autocastSetDataType(deviceType, dataType);
    }

    public static boolean autocastIsCacheEnabled() {
        return PyTorchLibrary.LIB.torchAutocastIsCacheEnabled();
    }

    public static void autocastSetCacheEnabled(boolean enabled) {
        PyTorchLibrary.LIB.torchAutocastSetCacheEnabled(enabled);
    }

    public static void autocastClearCache() {
        PyTorchLibrary.LIB.torchAutocastClearCache();
    }

    public static int autocastIncrementNesting() {
        return PyTorchLibrary.LIB.torchAutocastIncrementNesting();
    }

    public static int autocastDecrementNesting() {
        return PyTorchLibrary.LIB.torchAutocastDecrementNesting();
    }

    public static int getNumInteropThreads() {
        return PyTorchLibrary.LIB.torchGetNumInteropThreads();
    }

    public static int getNumThreads() {
        return PyTorchLibrary.LIB.torchGetNumThreads();
    }

    public static void setNumInteropThreads(int threads) {
        PyTorchLibrary.LIB.torchSetNumInteropThreads(threads);
    }

    public static void setNumThreads(int threads) {
        PyTorchLibrary.LIB.torchSetNumThreads(threads);
    }

    public static void setBenchmarkCuDNN(boolean enable) {
        PyTorchLibrary.LIB.torchSetBenchmarkCuDNN(enable);
    }

    public static synchronized Set<String> getFeatures() {
        if (configs != null) {
            return configs;
        }
        Set<String> features = new HashSet<>();
        PyTorchLibrary.LIB.torchShowConfig(features);
        configs = features;
        return configs;
    }

    public static int getGpuCount() {
        return PyTorchLibrary.LIB.torchGetGpuCount();
    }

    public static long[] getMemoryStats(int deviceId) {
        return PyTorchLibrary.LIB.torchGetMemoryStats(deviceId);
    }

    public static long[] getAllocatorSnapshot(int deviceId) {
        return PyTorchLibrary.LIB.torchGetAllocatorSnapshot(deviceId);
    }

    public static void resetPeakMemoryStats(int deviceId) {
        PyTorchLibrary.LIB.torchResetPeakMemoryStats(deviceId);
    }

    public static void setSeed(long seed) {
        PyTorchLibrary.LIB.torchManualSeed(seed);
    }

    /**
     * Calls this method to start profile the area you are interested in.
     *
     * <p>Example usage
     *
     * <pre>
     *      JniUtils.startProfile(false, true, true);
     *      Predictor.predict(img);
     *      JniUtils.stopProfile(outputFile)
     * </pre>
     *
     * @param useCuda Enables timing of CUDA events as well using the cudaEvent API.
     * @param recordShape If shapes recording is set, information about input dimensions will be
     *     collected
     * @param profileMemory Whether to report memory usage
     */
    public static synchronized void startProfile(
            boolean useCuda, boolean recordShape, boolean profileMemory) {
        PyTorchLibrary.LIB.torchStartProfile(useCuda, recordShape, profileMemory);
    }

    public static synchronized void stopProfile(String outputFile) {
        PyTorchLibrary.LIB.torchStopProfile(outputFile);
    }

    // TODO: Unchecked Datatype and device mapping
    public static PtNDArray createNdFromByteBuffer(
            PtNDManager manager,
            ByteBuffer data,
            Shape shape,
            DataType dType,
            SparseFormat fmt,
            Device device) {
        int layout = layoutMapper(fmt, device);
        long handle =
                PyTorchLibrary.LIB.torchFromBlob(
                        data,
                        shape.getShape(),
                        dType.ordinal(),
                        layout,
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false);

        if (layout == 1 || layout == 2 || device.isGpu()) {
            // MKLDNN & COO & GPU device will explicitly make a copy in native code
            // so we don't want to hold a reference on Java side
            return new PtNDArray(manager, handle);
        }
        return new PtNDArray(manager, handle, data);
    }

    public static void emptyCudaCache() {
        PyTorchLibrary.LIB.torchCudaEmptyCache();
    }

    public static PtNDArray createEmptyNdArray(
            PtNDManager manager, Shape shape, DataType dType, Device device, SparseFormat fmt) {
        int layoutVal = layoutMapper(fmt, device);
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchEmpty(
                        shape.getShape(),
                        dType.ordinal(),
                        layoutVal,
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false));
    }

    public static PtNDArray createZerosNdArray(
            PtNDManager manager, Shape shape, DataType dType, Device device, SparseFormat fmt) {
        int layoutVal = layoutMapper(fmt, device);
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchZeros(
                        shape.getShape(),
                        dType.ordinal(),
                        layoutVal,
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false));
    }

    public static PtNDArray createOnesNdArray(
            PtNDManager manager, Shape shape, DataType dType, Device device, SparseFormat fmt) {
        int layoutVal = layoutMapper(fmt, device);
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchOnes(
                        shape.getShape(),
                        dType.ordinal(),
                        layoutVal,
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false));
    }

    public static PtNDArray full(
            PtNDManager manager,
            Shape shape,
            double fillValue,
            DataType dType,
            Device device,
            SparseFormat fmt) {
        int layoutVal = layoutMapper(fmt, device);
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchFull(
                        shape.getShape(),
                        fillValue,
                        dType.ordinal(),
                        layoutVal,
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false));
    }

    public static PtNDArray zerosLike(
            PtNDArray array, DataType dType, Device device, SparseFormat fmt) {
        int layoutVal = layoutMapper(fmt, device);
        return new PtNDArray(
                array.getManager(),
                PyTorchLibrary.LIB.torchZerosLike(
                        array.getHandle(),
                        dType.ordinal(),
                        layoutVal,
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false));
    }

    public static PtNDArray onesLike(
            PtNDArray array, DataType dType, Device device, SparseFormat fmt) {
        int layoutVal = layoutMapper(fmt, device);
        return new PtNDArray(
                array.getManager(),
                PyTorchLibrary.LIB.torchOnesLike(
                        array.getHandle(),
                        dType.ordinal(),
                        layoutVal,
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false));
    }

    public static PtNDArray arange(
            PtNDManager manager,
            float start,
            float stop,
            float step,
            DataType dType,
            Device device,
            SparseFormat fmt) {
        int layoutVal = layoutMapper(fmt, device);
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchArange(
                        start,
                        stop,
                        step,
                        dType.ordinal(),
                        layoutVal,
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false));
    }

    public static PtNDArray linspace(
            PtNDManager manager,
            float start,
            float stop,
            int step,
            DataType dType,
            Device device,
            SparseFormat fmt) {
        int layoutVal = layoutMapper(fmt, device);
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchLinspace(
                        start,
                        stop,
                        step,
                        dType.ordinal(),
                        layoutVal,
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false));
    }

    public static PtNDArray createSparseCoo(PtNDArray indices, PtNDArray values, Shape shape) {
        return new PtNDArray(
                values.getManager(),
                PyTorchLibrary.LIB.torchSparseCoo(
                        shape.getShape(), indices.getHandle(), values.getHandle(), false));
    }

    public static PtNDArray to(PtNDArray ndArray, DataType dataType, Device device) {
        PtNDManager manager = ndArray.getManager();
        // the device of the manager should always match the one in NDArray which the manager attach
        // to
        if (!device.equals(manager.getDevice())) {
            manager = manager.newSubManager(device);
        }
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchTo(
                        ndArray.getHandle(),
                        dataType.ordinal(),
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()}));
    }

    /** Converts a floating-point array without detaching it from the autograd graph. */
    public static PtNDArray differentiableCast(PtNDArray ndArray, DataType dataType) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchDifferentiableCast(
                        ndArray.getHandle(), dataType.ordinal()));
    }

    public static PtNDArray toSparse(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchToSparse(ndArray.getHandle()));
    }

    public static PtNDArray toDense(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchToDense(ndArray.getHandle()));
    }

    public static PtNDArray broadcast(PtNDArray ndArray, Shape shape) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchExpand(ndArray.getHandle(), shape.getShape()));
    }

    public static PtNDArray slice(PtNDArray ndArray, long dim, long start, long stop, long step) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchSlice(ndArray.getHandle(), dim, start, stop, step));
    }

    public static PtNDArray index(
            PtNDArray ndArray,
            long[] minIndices,
            long[] maxIndices,
            long[] stepIndices,
            PtNDManager manager) {
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchIndex(
                        ndArray.getHandle(), minIndices, maxIndices, stepIndices));
    }

    @SuppressWarnings("OptionalGetWithoutIsPresent")
    public static PtNDArray indexAdv(PtNDArray ndArray, NDIndex index, PtNDManager manager) {
        if (ndArray == null) {
            return ndArray;
        }
        List<NDIndexElement> indices = index.getIndices();
        long torchIndexHandle = PyTorchLibrary.LIB.torchIndexInit(indices.size());
        try {
            // Index aggregation
            ListIterator<NDIndexElement> it = indices.listIterator();
            while (it.hasNext()) {
                if (it.nextIndex() == index.getEllipsisIndex()) {
                    PyTorchLibrary.LIB.torchIndexAppendNoneEllipsis(torchIndexHandle, true);
                }

                NDIndexElement elem = it.next();
                if (elem instanceof NDIndexNull) {
                    PyTorchLibrary.LIB.torchIndexAppendNoneEllipsis(torchIndexHandle, false);
                } else if (elem instanceof NDIndexSlice) {
                    Long min = ((NDIndexSlice) elem).getMin();
                    Long max = ((NDIndexSlice) elem).getMax();
                    Long step = ((NDIndexSlice) elem).getStep();
                    int nullSliceBinary = (min == null ? 1 : 0) * 2 + (max == null ? 1 : 0);
                    // nullSliceBinary encodes whether the slice end {min, max} is null:
                    // is_null == 1, ! is_null == 0;
                    // 0b11 == 3, 0b10 = 2, ...
                    // If {min, max} is null, then its value is ineffective, thus set to -1.
                    PyTorchLibrary.LIB.torchIndexAppendSlice(
                            torchIndexHandle,
                            min == null ? -1 : min,
                            max == null ? -1 : max,
                            step == null ? 1 : step,
                            nullSliceBinary);
                } else if (elem instanceof NDIndexAll) {
                    PyTorchLibrary.LIB.torchIndexAppendSlice(torchIndexHandle, -1, -1, 1, 3);
                } else if (elem instanceof NDIndexFixed) {
                    PyTorchLibrary.LIB.torchIndexAppendFixed(
                            torchIndexHandle, ((NDIndexFixed) elem).getIndex());
                } else if (elem instanceof NDIndexBooleans) {
                    PtNDArray indexArr = (PtNDArray) ((NDIndexBooleans) elem).getIndex();
                    PyTorchLibrary.LIB.torchIndexAppendArray(
                            torchIndexHandle, indexArr.getHandle());
                } else if (elem instanceof NDIndexTake) {
                    PtNDArray indexArr = manager.from(((NDIndexTake) elem).getIndex());
                    if (indexArr.getDataType() != DataType.INT64) {
                        indexArr = indexArr.toType(DataType.INT64, true);
                    }
                    PyTorchLibrary.LIB.torchIndexAppendArray(
                            torchIndexHandle, indexArr.getHandle());
                } else if (elem instanceof NDIndexPick) {
                    // Backward compatible
                    NDIndexFullPick fullPick =
                            NDIndexFullPick.fromIndex(index, ndArray.getShape()).get();
                    return pick(ndArray, manager.from(fullPick.getIndices()), fullPick.getAxis());
                }
            }
            if (indices.size() == index.getEllipsisIndex()) {
                PyTorchLibrary.LIB.torchIndexAppendNoneEllipsis(torchIndexHandle, true);
            }
            long ret = PyTorchLibrary.LIB.torchIndexAdvGet(ndArray.getHandle(), torchIndexHandle);
            return new PtNDArray(manager, ret);
        } finally {
            PyTorchLibrary.LIB.torchDeleteIndex(torchIndexHandle);
        }
    }

    @SuppressWarnings("OptionalGetWithoutIsPresent")
    public static void indexAdvPut(PtNDArray ndArray, NDIndex index, PtNDArray data) {
        if (ndArray == null) {
            return;
        }
        List<NDIndexElement> indices = index.getIndices();
        long torchIndexHandle = PyTorchLibrary.LIB.torchIndexInit(indices.size());
        try {
            // Index aggregation
            ListIterator<NDIndexElement> it = indices.listIterator();
            while (it.hasNext()) {
                if (it.nextIndex() == index.getEllipsisIndex()) {
                    PyTorchLibrary.LIB.torchIndexAppendNoneEllipsis(torchIndexHandle, true);
                }

                NDIndexElement elem = it.next();
                if (elem instanceof NDIndexNull) {
                    PyTorchLibrary.LIB.torchIndexAppendNoneEllipsis(torchIndexHandle, false);
                } else if (elem instanceof NDIndexSlice) {
                    Long min = ((NDIndexSlice) elem).getMin();
                    Long max = ((NDIndexSlice) elem).getMax();
                    Long step = ((NDIndexSlice) elem).getStep();
                    int nullSliceBinary = (min == null ? 1 : 0) * 2 + (max == null ? 1 : 0);
                    // nullSliceBinary encodes whether the slice end {min, max} is null:
                    // is_null == 1, ! is_null == 0;
                    // 0b11 == 3, 0b10 = 2, ...
                    // If {min, max} is null, then its value is ineffective, thus set to -1.
                    PyTorchLibrary.LIB.torchIndexAppendSlice(
                            torchIndexHandle,
                            min == null ? -1 : min,
                            max == null ? -1 : max,
                            step == null ? 1 : step,
                            nullSliceBinary);
                } else if (elem instanceof NDIndexAll) {
                    PyTorchLibrary.LIB.torchIndexAppendSlice(torchIndexHandle, -1, -1, 1, 3);
                } else if (elem instanceof NDIndexFixed) {
                    PyTorchLibrary.LIB.torchIndexAppendFixed(
                            torchIndexHandle, ((NDIndexFixed) elem).getIndex());
                } else if (elem instanceof NDIndexBooleans) {
                    PtNDArray indexArr = (PtNDArray) ((NDIndexBooleans) elem).getIndex();
                    PyTorchLibrary.LIB.torchIndexAppendArray(
                            torchIndexHandle, indexArr.getHandle());
                } else if (elem instanceof NDIndexTake) {
                    PtNDArray indexArr = (PtNDArray) ((NDIndexTake) elem).getIndex();
                    if (indexArr.getDataType() != DataType.INT64) {
                        indexArr = indexArr.toType(DataType.INT64, true);
                    }
                    PyTorchLibrary.LIB.torchIndexAppendArray(
                            torchIndexHandle, indexArr.getHandle());
                } else if (elem instanceof NDIndexPick) {
                    // Backward compatible
                    NDIndexFullPick fullPick =
                            NDIndexFullPick.fromIndex(index, ndArray.getShape()).get();
                    pick(
                            ndArray,
                            ndArray.getManager().from(fullPick.getIndices()),
                            fullPick.getAxis());
                    return;
                }
            }
            if (indices.size() == index.getEllipsisIndex()) {
                PyTorchLibrary.LIB.torchIndexAppendNoneEllipsis(torchIndexHandle, true);
            }

            PyTorchLibrary.LIB.torchIndexAdvPut(
                    ndArray.getHandle(), torchIndexHandle, data.getHandle());
        } finally {
            PyTorchLibrary.LIB.torchDeleteIndex(torchIndexHandle);
        }
    }

    public static void indexSet(
            PtNDArray ndArray,
            PtNDArray value,
            long[] minIndices,
            long[] maxIndices,
            long[] stepIndices) {
        PyTorchLibrary.LIB.torchIndexPut(
                ndArray.getHandle(), value.getHandle(), minIndices, maxIndices, stepIndices);
    }

    public static void set(PtNDArray self, ByteBuffer data) {
        // Note the ByteBuffer here is directByteBuffer
        PyTorchLibrary.LIB.torchSet(self.getHandle(), data);
    }

    public static long allocatePinnedBuffer(int size, DataType dataType) {
        return PyTorchLibrary.LIB.torchAllocatePinnedBuffer(size, dataType.ordinal());
    }

    public static ByteBuffer getPinnedBuffer(long handle) {
        return PyTorchLibrary.LIB.torchGetPinnedBuffer(handle);
    }

    public static boolean isPinnedBuffer(long handle) {
        return PyTorchLibrary.LIB.torchIsPinnedBuffer(handle);
    }

    public static void deletePinnedBuffer(long handle) {
        PyTorchLibrary.LIB.torchDeletePinnedBuffer(handle);
    }

    public static void copyFromDirectBuffer(PtNDArray self, ByteBuffer data) {
        PyTorchLibrary.LIB.torchCopyFromDirectBuffer(self.getHandle(), data);
    }

    public static void copyFromPinnedBuffer(PtNDArray self, long pinnedBufferHandle) {
        PyTorchLibrary.LIB.torchCopyFromPinnedBuffer(self.getHandle(), pinnedBufferHandle);
    }

    public static long copyFromPinnedBufferAsync(PtNDArray self, long pinnedBufferHandle) {
        return PyTorchLibrary.LIB.torchCopyFromPinnedBufferAsync(
                self.getHandle(), pinnedBufferHandle);
    }

    public static void enqueueCopyFrom(PtNDArray self, long pinnedBufferHandle) {
        PyTorchLibrary.LIB.torchEnqueueCopyFrom(self.getHandle(), pinnedBufferHandle);
    }

    public static long copyToPinnedBufferAsync(PtNDArray self, long pinnedBufferHandle) {
        return PyTorchLibrary.LIB.torchCopyToPinnedBufferAsync(
                self.getHandle(), pinnedBufferHandle);
    }

    public static void enqueueCopyTo(PtNDArray self, long pinnedBufferHandle) {
        PyTorchLibrary.LIB.torchEnqueueCopyTo(self.getHandle(), pinnedBufferHandle);
    }

    public static void copyTo(PtNDArray source, PtNDArray target) {
        PyTorchLibrary.LIB.torchCopyTo(source.getHandle(), target.getHandle());
    }

    public static void synchronizeCopyEvent(long handle) {
        PyTorchLibrary.LIB.torchSynchronizeCopyEvent(handle);
    }

    public static void deleteCopyEvent(long handle) {
        PyTorchLibrary.LIB.torchDeleteCopyEvent(handle);
    }

    public static void recordStream(PtNDArray self) {
        PyTorchLibrary.LIB.torchRecordStream(self.getHandle());
    }

    public static PtNDArray gather(PtNDArray ndArray, PtNDArray index, long dim) {
        if (index.getDataType() != DataType.INT64) {
            index = index.toType(DataType.INT64, true);
        }
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchGather(ndArray.getHandle(), index.getHandle(), dim, false));
    }

    public static PtNDArray take(PtNDArray ndArray, PtNDArray index, PtNDManager manager) {
        if (index.getDataType() != DataType.INT64) {
            index = index.toType(DataType.INT64, true);
        }
        return new PtNDArray(
                manager, PyTorchLibrary.LIB.torchTake(ndArray.getHandle(), index.getHandle()));
    }

    public static PtNDArray put(PtNDArray ndArray, PtNDArray index, PtNDArray value) {
        if (index.getDataType() != DataType.INT64) {
            index = index.toType(DataType.INT64, true);
        }
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchPut(
                        ndArray.getHandle(), index.getHandle(), value.getHandle()));
    }

    public static PtNDArray scatter(PtNDArray ndArray, PtNDArray index, PtNDArray value, int axis) {
        if (index.getDataType() != DataType.INT64) {
            index = index.toType(DataType.INT64, true);
        }
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchScatter(
                        ndArray.getHandle(), index.getHandle(), value.getHandle(), axis));
    }

    /** Adds namespace offsets to integer IDs and performs a dense embedding lookup. */
    public static PtNDArray embeddingWithOffsets(
            PtNDArray rawIds, PtNDArray offsets, PtNDArray table) {
        return new PtNDArray(
                rawIds.getManager(),
                PyTorchLibrary.LIB.torchEmbeddingWithOffsets(
                        rawIds.getHandle(), offsets.getHandle(), table.getHandle()));
    }

    /** Packs flattened offset embedding fields followed by dense features. */
    public static PtNDArray embeddingFeaturePack(
            PtNDArray rawIds, PtNDArray offsets, PtNDArray table, PtNDArray features) {
        return new PtNDArray(
                rawIds.getManager(),
                PyTorchLibrary.LIB.torchEmbeddingFeaturePack(
                        rawIds.getHandle(),
                        offsets.getHandle(),
                        table.getHandle(),
                        features.getHandle()));
    }

    /** Selects leading-axis rows with a one-dimensional index tensor. */
    public static PtNDArray gatherRows(PtNDArray rows, PtNDArray rowIndices) {
        if (rowIndices.getDataType() != DataType.INT64) {
            rowIndices = rowIndices.toType(DataType.INT64, true);
        }
        return new PtNDArray(
                rows.getManager(),
                PyTorchLibrary.LIB.torchGatherRows(rows.getHandle(), rowIndices.getHandle()));
    }

    /** Places leading-axis rows into a zero-initialized dense tensor. */
    public static PtNDArray scatterRows(PtNDArray rows, PtNDArray rowIndices, long rowCount) {
        if (rowIndices.getDataType() != DataType.INT64) {
            rowIndices = rowIndices.toType(DataType.INT64, true);
        }
        return new PtNDArray(
                rows.getManager(),
                PyTorchLibrary.LIB.torchScatterRows(
                        rows.getHandle(), rowIndices.getHandle(), rowCount));
    }

    /** Builds masked categorical membership indicators. */
    public static PtNDArray categoricalMasks(
            PtNDArray categories, PtNDArray mask, int[] fieldIndices, long[] categorySets) {
        return new PtNDArray(
                categories.getManager(),
                PyTorchLibrary.LIB.torchCategoricalMasks(
                        categories.getHandle(), mask.getHandle(), fieldIndices, categorySets));
    }

    /** Builds the four masks associated with a routed binary choice. */
    public static PtNDArray binaryChoiceMasks(
            PtNDArray routes,
            PtNDArray firstMask,
            PtNDArray secondMask,
            int representativeField,
            int firstRouteField,
            int secondRouteField,
            long paddingValue) {
        return new PtNDArray(
                routes.getManager(),
                PyTorchLibrary.LIB.torchBinaryChoiceMasks(
                        routes.getHandle(),
                        firstMask.getHandle(),
                        secondMask.getHandle(),
                        representativeField,
                        firstRouteField,
                        secondRouteField,
                        paddingValue));
    }

    /** Sums one lookup row from each contiguous table segment. */
    public static PtNDArray segmentedLookupSum(PtNDArray lookupTable, PtNDArray storedIndices) {
        return new PtNDArray(
                lookupTable.getManager(),
                PyTorchLibrary.LIB.torchSegmentedLookupSum(
                        lookupTable.getHandle(), storedIndices.getHandle()));
    }

    /** Selects one-based entries from an independent table in each batch. */
    public static PtNDArray paddedBatchGather(PtNDArray source, PtNDArray storedIndices) {
        return new PtNDArray(
                source.getManager(),
                PyTorchLibrary.LIB.torchPaddedBatchGather(
                        source.getHandle(), storedIndices.getHandle()));
    }

    /** Selects one-based entries from two independent table dimensions in each batch. */
    public static PtNDArray paddedBatchGather(
            PtNDArray source, PtNDArray outerStoredIndices, PtNDArray innerStoredIndices) {
        return new PtNDArray(
                source.getManager(),
                PyTorchLibrary.LIB.torchPaddedBatchGather2d(
                        source.getHandle(),
                        outerStoredIndices.getHandle(),
                        innerStoredIndices.getHandle()));
    }

    /** Selects one-based table entries from explicit zero-based batch rows. */
    public static PtNDArray paddedBatchGatherByBatchIndices(
            PtNDArray source, PtNDArray batchIndices, PtNDArray storedIndices) {
        return new PtNDArray(
                source.getManager(),
                PyTorchLibrary.LIB.torchPaddedBatchGatherByBatchIndices(
                        source.getHandle(), batchIndices.getHandle(), storedIndices.getHandle()));
    }

    /** Returns {@code ndArray.index_add(axis, index, value)} without mutating the input tensor. */
    public static PtNDArray indexAdd(
            PtNDArray ndArray, PtNDArray index, PtNDArray value, int axis) {
        if (index.getDataType() != DataType.INT64) {
            index = index.toType(DataType.INT64, true);
        }
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchIndexAdd(
                        ndArray.getHandle(), index.getHandle(), value.getHandle(), axis));
    }

    public static PtNDArray pick(PtNDArray ndArray, PtNDArray index, long dim) {
        Shape indexShape = index.getShape();
        Shape ndShape = ndArray.getShape();
        int shapeDims = indexShape.dimension();
        int ndDims = ndShape.dimension();
        if (shapeDims != ndDims) {
            for (int i = 0; i < ndDims - shapeDims; ++i) {
                if (indexShape.equals(ndShape.slice(i, shapeDims))) {
                    long[] shapes = indexShape.getShape();
                    long[] newShape = new long[ndDims];
                    Arrays.fill(newShape, 0, i, 1L);
                    Arrays.fill(newShape, i, i + shapes.length, shapes[i]);
                    Arrays.fill(newShape, i + shapes.length, ndDims, 1L);
                    indexShape = new Shape(newShape);
                    break;
                }
            }
            if (indexShape.equals(index.getShape())) {
                throw new IllegalArgumentException(
                        "expand shape failed! Cannot expand from " + indexShape + "to " + ndShape);
            }
            index = index.reshape(indexShape);
        }
        if (index.getDataType() != DataType.INT64) {
            index = index.toType(DataType.INT64, true);
        }
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchGather(ndArray.getHandle(), index.getHandle(), dim, false));
    }

    public static PtNDArray where(PtNDArray condition, PtNDArray self, PtNDArray other) {
        return new PtNDArray(
                self.getManager(),
                PyTorchLibrary.LIB.torchWhere(
                        condition.getHandle(), self.getHandle(), other.getHandle()));
    }

    public static PtNDArray booleanMask(PtNDArray ndArray, PtNDArray indicesNd) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchMaskedSelect(ndArray.getHandle(), indicesNd.getHandle()));
    }

    public static void booleanMaskSet(PtNDArray ndArray, PtNDArray value, PtNDArray indicesNd) {
        PyTorchLibrary.LIB.torchMaskedPut(
                ndArray.getHandle(), value.getHandle(), indicesNd.getHandle());
    }

    public static PtNDArray getItem(PtNDArray ndArray, long[] indices, PtNDManager manager) {
        // use a specialized API here
        // due to significant performance gain
        // for commonly used data loading call
        if (indices.length == 1) {
            return new PtNDArray(
                    manager, PyTorchLibrary.LIB.torchGetItem(ndArray.getHandle(), indices[0]));
        }
        return new PtNDArray(
                manager, PyTorchLibrary.LIB.torchGetItem(ndArray.getHandle(), indices));
    }

    public static PtNDArray clone(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.tensorClone(ndArray.getHandle()));
    }

    public static PtNDArray pad(PtNDArray ndArray, long[] shape, double value) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchPad(ndArray.getHandle(), shape, value));
    }

    public static PtNDArray reshape(PtNDArray ndArray, long[] shape) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchReshape(ndArray.getHandle(), shape));
    }

    public static PtNDArray stack(PtNDArray[] arrays, int dim) {
        long[] pointers = Arrays.stream(arrays).mapToLong(PtNDArray::getHandle).toArray();
        return new PtNDArray(arrays[0].getManager(), PyTorchLibrary.LIB.torchStack(pointers, dim));
    }

    public static PtNDArray cat(PtNDArray[] arrays, long dim) {
        long[] pointers = Arrays.stream(arrays).mapToLong(PtNDArray::getHandle).toArray();
        return new PtNDArray(arrays[0].getManager(), PyTorchLibrary.LIB.torchCat(pointers, dim));
    }

    /** Converts floating-point arrays while concatenating them along an existing axis. */
    public static PtNDArray concatToType(PtNDArray[] arrays, long dim, DataType dataType) {
        long[] pointers = Arrays.stream(arrays).mapToLong(PtNDArray::getHandle).toArray();
        return new PtNDArray(
                arrays[0].getManager(),
                PyTorchLibrary.LIB.torchConcatToType(pointers, dim, dataType.ordinal()));
    }

    public static PtNDArray tile(PtNDArray ndArray, long[] repeats) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchRepeat(ndArray.getHandle(), repeats));
    }

    public static PtNDArray repeat(PtNDArray ndArray, long repeat, long dim) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchRepeatInterleave(ndArray.getHandle(), repeat, dim));
    }

    public static PtNDArray softmax(PtNDArray ndArray, long dim, DataType dTpe) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchSoftmax(ndArray.getHandle(), dim, dTpe.ordinal()));
    }

    /** Returns float32 probabilities normalized over legal mask entries. */
    public static PtNDArray maskedSoftmax(PtNDArray logits, PtNDArray mask, long axis) {
        return new PtNDArray(
                logits.getManager(),
                PyTorchLibrary.LIB.torchMaskedSoftmax(logits.getHandle(), mask.getHandle(), axis));
    }

    /** Returns packed float32 weighted mean, minimum, and maximum values for each row. */
    public static PtNDArray weightedRowStatistics(PtNDArray values, PtNDArray weights) {
        return new PtNDArray(
                values.getManager(),
                PyTorchLibrary.LIB.torchWeightedRowStatistics(
                        values.getHandle(), weights.getHandle()));
    }

    /** Pools values with independently masked softmax weights for several groups. */
    public static PtNDArray groupedMaskedSoftmaxPool(
            PtNDArray logits, PtNDArray mask, PtNDArray values) {
        return new PtNDArray(
                logits.getManager(),
                PyTorchLibrary.LIB.torchGroupedMaskedSoftmaxPool(
                        logits.getHandle(), mask.getHandle(), values.getHandle()));
    }

    /** Pools selected values with masked softmax weights without materializing the subset. */
    public static PtNDArray indexedMaskedSoftmaxPool(
            PtNDArray logits, PtNDArray mask, PtNDArray values, int[] choiceIndices) {
        return new PtNDArray(
                logits.getManager(),
                PyTorchLibrary.LIB.torchIndexedMaskedSoftmaxPool(
                        logits.getHandle(), mask.getHandle(), values.getHandle(), choiceIndices));
    }

    /** Returns the float32 log normalizer over legal mask entries. */
    public static PtNDArray maskedLogSumExp(PtNDArray logits, PtNDArray mask, long axis) {
        return new PtNDArray(
                logits.getManager(),
                PyTorchLibrary.LIB.torchMaskedLogSumExp(
                        logits.getHandle(), mask.getHandle(), axis));
    }

    public static PtNDArray scaledDotProductAttention(
            PtNDArray query,
            PtNDArray key,
            PtNDArray value,
            PtNDArray attnMask,
            double dropoutP,
            boolean isCausal,
            double scale) {
        long maskHandle = attnMask == null ? 0L : attnMask.getHandle();
        return new PtNDArray(
                query.getManager(),
                PyTorchLibrary.LIB.torchScaledDotProductAttention(
                        query.getHandle(),
                        key.getHandle(),
                        value.getHandle(),
                        maskHandle,
                        dropoutP,
                        isCausal,
                        scale));
    }

    /** Gathers relation logits into a pairwise additive attention bias. */
    public static PtNDArray indexedRelationBias(
            PtNDArray relationLogits, PtNDArray relationBias, PtNDArray relationIds, float scale) {
        return new PtNDArray(
                relationLogits.getManager(),
                PyTorchLibrary.LIB.torchIndexedRelationBias(
                        relationLogits.getHandle(),
                        relationBias.getHandle(),
                        relationIds.getHandle(),
                        scale));
    }

    /**
     * Applies grouped packed attention with the native backend's differentiable execution plan.
     *
     * @param query token-major shared query projection
     * @param packedKeyValue token-major grouped key/value projection
     * @param mask nonzero valid-token mask
     * @param heads number of attention heads
     * @param scale query-key score scale
     * @return grouped attended values in token-major packed-head layout
     */
    public static PtNDArray groupedPackedScaledDotProductAttention(
            PtNDArray query, PtNDArray packedKeyValue, PtNDArray mask, long heads, float scale) {
        return new PtNDArray(
                query.getManager(),
                PyTorchLibrary.LIB.torchGroupedPackedScaledDotProductAttention(
                        query.getHandle(),
                        packedKeyValue.getHandle(),
                        mask.getHandle(),
                        heads,
                        scale));
    }

    /**
     * Applies grouped indexed attention without materializing a complete key/value table per query.
     *
     * <p>Packed key/value tensors store all head keys followed by all head values. Positive indexed
     * IDs are one-based shared-token indices and zero denotes padding.
     *
     * @param query queries shaped {@code [query,heads,keyFeatures]}
     * @param sharedKeyValues shared packed data shaped {@code [group,sharedTokens,packedWidth]}
     * @param sharedDeltas query-specific shared-token deltas
     * @param indexedDeltas query-specific indexed-token deltas
     * @param indexedSharedIds one-based shared-token IDs; zero denotes padding
     * @param queriesPerGroup number of consecutive queries sharing one packed table
     * @param scale attention score scale
     * @return attended values shaped {@code [query,heads,valueFeatures]}
     */
    public static PtNDArray groupedIndexedScaledDotProductAttention(
            PtNDArray query,
            PtNDArray sharedKeyValues,
            PtNDArray sharedDeltas,
            PtNDArray indexedDeltas,
            PtNDArray indexedSharedIds,
            long queriesPerGroup,
            float scale) {
        return new PtNDArray(
                query.getManager(),
                PyTorchLibrary.LIB.torchGroupedIndexedScaledDotProductAttention(
                        query.getHandle(),
                        sharedKeyValues.getHandle(),
                        sharedDeltas.getHandle(),
                        indexedDeltas.getHandle(),
                        indexedSharedIds.getHandle(),
                        queriesPerGroup,
                        scale));
    }

    /**
     * Applies differentiable grouped indexed attention through explicit group and shared-delta
     * lookup mappings.
     *
     * @param query queries shaped {@code [query,heads,keyFeatures]}
     * @param sharedKeyValues packed shared data shaped {@code [groups,sharedTokens,packedWidth]}
     * @param sharedGroupIndices zero-based group indices
     * @param sharedDeltaTable packed shared-token delta lookup table
     * @param sharedDeltaIndices zero-based delta indices
     * @param indexedDeltas query-specific auxiliary deltas
     * @param indexedSharedIds one-based shared-token IDs; zero denotes padding
     * @param scale attention score scale
     * @return attended values shaped {@code [query,heads,valueFeatures]}
     */
    public static PtNDArray mappedGroupedIndexedScaledDotProductAttention(
            PtNDArray query,
            PtNDArray sharedKeyValues,
            PtNDArray sharedGroupIndices,
            PtNDArray sharedDeltaTable,
            PtNDArray sharedDeltaIndices,
            PtNDArray indexedDeltas,
            PtNDArray indexedSharedIds,
            float scale) {
        return new PtNDArray(
                query.getManager(),
                PyTorchLibrary.LIB.torchMappedGroupedIndexedScaledDotProductAttention(
                        query.getHandle(),
                        sharedKeyValues.getHandle(),
                        sharedGroupIndices.getHandle(),
                        sharedDeltaTable.getHandle(),
                        sharedDeltaIndices.getHandle(),
                        indexedDeltas.getHandle(),
                        indexedSharedIds.getHandle(),
                        scale));
    }

    /**
     * Adds an inference residual in place and returns its affine LayerNorm.
     *
     * <p>The residual remains the unnormalized sum so the following residual branch observes the
     * same value as the ordinary {@code addi} followed by LayerNorm path.
     */
    public static PtNDArray addToOwnedResidualAndLayerNorm(
            PtNDArray residual, PtNDArray update, PtNDArray weight, PtNDArray bias, float epsilon) {
        return new PtNDArray(
                residual.getManager(),
                PyTorchLibrary.LIB.torchAddToOwnedResidualAndLayerNorm(
                        residual.getHandle(),
                        update.getHandle(),
                        weight.getHandle(),
                        bias.getHandle(),
                        epsilon));
    }

    /** Adds masked embedding rows to an owned token buffer and returns its converted mask. */
    public static PtNDArray addMaskedEmbeddingResidualToOwnedTokens(
            PtNDArray tokens,
            NDList storedIndices,
            PtNDArray embeddingTable,
            PtNDArray validMask,
            long paddingIndex,
            EmbeddingReduction reduction) {
        long[] indexHandles = new long[storedIndices.size()];
        PtNDManager manager = tokens.getManager();
        for (int index = 0; index < indexHandles.length; ++index) {
            indexHandles[index] = manager.from(storedIndices.get(index)).getHandle();
        }
        int reductionValue;
        switch (reduction) {
            case SUM:
                reductionValue = 0;
                break;
            case MEAN_VALID:
                reductionValue = 1;
                break;
            default:
                throw new AssertionError("Unsupported embedding reduction: " + reduction);
        }
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchAddMaskedEmbeddingResidualToOwnedTokens(
                        tokens.getHandle(),
                        indexHandles,
                        embeddingTable.getHandle(),
                        validMask.getHandle(),
                        paddingIndex,
                        reductionValue));
    }

    /** Adds a broadcast residual to caller-owned values and applies SiLU and an optional mask. */
    public static PtNDArray addBroadcastResidualToOwnedAndSilu(
            PtNDArray values, PtNDArray residual, PtNDArray mask) {
        PyTorchLibrary.LIB.torchAddBroadcastResidualToOwnedAndSilu(
                values.getHandle(), residual.getHandle(), mask == null ? 0L : mask.getHandle());
        return values;
    }

    /** Adds a bias and broadcast residual to caller-owned values before SiLU. */
    public static PtNDArray addBiasAndBroadcastResidualToOwnedAndSilu(
            PtNDArray values, PtNDArray bias, PtNDArray residual, PtNDArray mask) {
        PyTorchLibrary.LIB.torchAddBiasAndBroadcastResidualToOwnedAndSilu(
                values.getHandle(),
                bias.getHandle(),
                residual.getHandle(),
                mask == null ? 0L : mask.getHandle());
        return values;
    }

    public static PtNDArray rmsNorm(
            PtNDArray input, long[] normalizedShape, PtNDArray weight, double eps) {
        long weightHandle = weight == null ? 0L : weight.getHandle();
        return new PtNDArray(
                input.getManager(),
                PyTorchLibrary.LIB.torchRmsNorm(
                        input.getHandle(), normalizedShape, weightHandle, eps));
    }

    public static PtNDArray logSoftmax(PtNDArray ndArray, long dim, DataType dTpe) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchLogSoftmax(ndArray.getHandle(), dim, dTpe.ordinal()));
    }

    public static PtNDArray argMax(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchArgMax(ndArray.getHandle()));
    }

    public static PtNDArray argMax(PtNDArray ndArray, long dim, boolean keepDim) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchArgMax(ndArray.getHandle(), dim, keepDim));
    }

    public static NDList topK(
            PtNDArray ndArray, long k, long axis, boolean largest, boolean sorted) {
        long[] handles =
                PyTorchLibrary.LIB.torchTopK(ndArray.getHandle(), k, axis, largest, sorted);
        NDList list = new NDList(handles.length);
        for (long handle : handles) {
            PtNDArray array = new PtNDArray(ndArray.getManager(), handle);
            list.add(array);
        }
        return list;
    }

    public static PtNDArray argMin(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchArgMin(ndArray.getHandle()));
    }

    public static PtNDArray argMin(PtNDArray ndArray, long dim, boolean keepDim) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchArgMin(ndArray.getHandle(), dim, keepDim));
    }

    public static PtNDArray argSort(PtNDArray ndArray, long dim, boolean keepDim) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchArgSort(ndArray.getHandle(), dim, keepDim));
    }

    public static PtNDArray sort(PtNDArray ndArray, long dim, boolean descending) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchSort(ndArray.getHandle(), dim, descending));
    }

    public static PtNDArray permute(PtNDArray ndArray, long[] dims) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchPermute(ndArray.getHandle(), dims));
    }

    public static PtNDArray flip(PtNDArray ndArray, long[] dims) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchFlip(ndArray.getHandle(), dims));
    }

    public static PtNDArray transpose(PtNDArray ndArray, long dim1, long dim2) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchTranspose(ndArray.getHandle(), dim1, dim2));
    }

    public static boolean contentEqual(PtNDArray ndArray1, PtNDArray ndArray2) {
        return PyTorchLibrary.LIB.contentEqual(ndArray1.getHandle(), ndArray2.getHandle());
    }

    public static PtNDArray add(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchAdd(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray add(PtNDArray ndArray, Number value) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchAddScalar(
                        ndArray.getHandle(),
                        value.longValue(),
                        value.doubleValue(),
                        isFloatingScalar(value)));
    }

    public static void addi(PtNDArray ndArray1, PtNDArray ndArray2) {
        PyTorchLibrary.LIB.torchAddi(ndArray1.getHandle(), ndArray2.getHandle());
    }

    public static void addi(PtNDArray ndArray, Number value) {
        PyTorchLibrary.LIB.torchAddiScalar(
                ndArray.getHandle(),
                value.longValue(),
                value.doubleValue(),
                isFloatingScalar(value));
    }

    public static PtNDArray sub(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchSub(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray sub(PtNDArray ndArray, Number value) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchSubScalar(
                        ndArray.getHandle(),
                        value.longValue(),
                        value.doubleValue(),
                        isFloatingScalar(value)));
    }

    public static void subi(PtNDArray ndArray1, PtNDArray ndArray2) {
        PyTorchLibrary.LIB.torchSubi(ndArray1.getHandle(), ndArray2.getHandle());
    }

    public static void subi(PtNDArray ndArray, Number value) {
        PyTorchLibrary.LIB.torchSubiScalar(
                ndArray.getHandle(),
                value.longValue(),
                value.doubleValue(),
                isFloatingScalar(value));
    }

    public static void fill(PtNDArray ndArray, double value) {
        PyTorchLibrary.LIB.torchFill(ndArray.getHandle(), value);
    }

    public static PtNDArray mul(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchMul(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray mul(PtNDArray ndArray, Number value) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchMulScalar(
                        ndArray.getHandle(),
                        value.longValue(),
                        value.doubleValue(),
                        isFloatingScalar(value)));
    }

    public static void muli(PtNDArray ndArray1, PtNDArray ndArray2) {
        PyTorchLibrary.LIB.torchMuli(ndArray1.getHandle(), ndArray2.getHandle());
    }

    public static void muli(PtNDArray ndArray, Number value) {
        PyTorchLibrary.LIB.torchMuliScalar(
                ndArray.getHandle(),
                value.longValue(),
                value.doubleValue(),
                isFloatingScalar(value));
    }

    public static PtNDArray div(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchTrueDivide(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray div(PtNDArray ndArray, Number value) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchTrueDivideScalar(
                        ndArray.getHandle(),
                        value.longValue(),
                        value.doubleValue(),
                        isFloatingScalar(value)));
    }

    public static void divi(PtNDArray ndArray1, PtNDArray ndArray2) {
        PyTorchLibrary.LIB.torchTrueDividei(ndArray1.getHandle(), ndArray2.getHandle());
    }

    public static void divi(PtNDArray ndArray, Number value) {
        PyTorchLibrary.LIB.torchTrueDivideiScalar(
                ndArray.getHandle(),
                value.longValue(),
                value.doubleValue(),
                isFloatingScalar(value));
    }

    public static PtNDArray remainder(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchRemainder(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray remainder(PtNDArray ndArray, Number value) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchRemainderScalar(
                        ndArray.getHandle(),
                        value.longValue(),
                        value.doubleValue(),
                        isFloatingScalar(value)));
    }

    public static void remainderi(PtNDArray ndArray1, PtNDArray ndArray2) {
        PyTorchLibrary.LIB.torchRemainderi(ndArray1.getHandle(), ndArray2.getHandle());
    }

    public static void remainderi(PtNDArray ndArray, Number value) {
        PyTorchLibrary.LIB.torchRemainderiScalar(
                ndArray.getHandle(),
                value.longValue(),
                value.doubleValue(),
                isFloatingScalar(value));
    }

    public static PtNDArray pow(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchPow(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray pow(PtNDArray ndArray, Number value) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchPowScalar(
                        ndArray.getHandle(),
                        value.longValue(),
                        value.doubleValue(),
                        isFloatingScalar(value)));
    }

    public static void powi(PtNDArray ndArray1, PtNDArray ndArray2) {
        PyTorchLibrary.LIB.torchPowi(ndArray1.getHandle(), ndArray2.getHandle());
    }

    public static void powi(PtNDArray ndArray, Number value) {
        PyTorchLibrary.LIB.torchPowiScalar(
                ndArray.getHandle(),
                value.longValue(),
                value.doubleValue(),
                isFloatingScalar(value));
    }

    public static PtNDArray sign(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchSign(ndArray.getHandle()));
    }

    public static void signi(PtNDArray ndArray) {
        PyTorchLibrary.LIB.torchSigni(ndArray.getHandle());
    }

    public static PtNDArray logicalAnd(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchLogicalAnd(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray logicalOr(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchLogicalOr(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray logicalXor(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchLogicalXor(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray logicalNot(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchLogicalNot(ndArray.getHandle()));
    }

    public static PtNDArray matmul(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchMatmul(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray bmm(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchBmm(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray xlogy(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchXLogY(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray dot(PtNDArray ndArray1, PtNDArray ndArray2) {
        if (ndArray1.getShape().dimension() == 1) {
            return new PtNDArray(
                    ndArray1.getManager(),
                    PyTorchLibrary.LIB.torchDot(ndArray1.getHandle(), ndArray2.getHandle()));
        }
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchMatmul(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray max(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchMaximum(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray max(PtNDArray ndArray, Number value) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchMaximumScalar(
                        ndArray.getHandle(),
                        value.longValue(),
                        value.doubleValue(),
                        isFloatingScalar(value)));
    }

    public static PtNDArray max(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchMax(ndArray.getHandle()));
    }

    public static PtNDArray max(PtNDArray ndArray, long dim, boolean keepDim) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchMax(ndArray.getHandle(), dim, keepDim));
    }

    public static PtNDArray min(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchMinimum(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray min(PtNDArray ndArray, Number value) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchMinimumScalar(
                        ndArray.getHandle(),
                        value.longValue(),
                        value.doubleValue(),
                        isFloatingScalar(value)));
    }

    public static PtNDArray min(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchMin(ndArray.getHandle()));
    }

    public static PtNDArray min(PtNDArray ndArray, long dim, boolean keepDim) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchMin(ndArray.getHandle(), dim, keepDim));
    }

    public static NDList median(PtNDArray ndArray, long dim, boolean keepDim) {
        long[] handles = PyTorchLibrary.LIB.torchMedian(ndArray.getHandle(), dim, keepDim);
        return new NDList(
                new PtNDArray(ndArray.getManager(), handles[0]),
                new PtNDArray(ndArray.getManager(), handles[1]));
    }

    public static PtNDArray percentile(
            PtNDArray ndArray, Number percentile, long dim, boolean keepDim) {
        float quantile = percentile.floatValue() / 100;
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchQuantile(ndArray.getHandle(), quantile, dim, keepDim));
    }

    public static PtNDArray mean(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchMean(ndArray.getHandle()));
    }

    public static PtNDArray mean(PtNDArray ndArray, long dim, boolean keepDim) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchMean(ndArray.getHandle(), dim, keepDim));
    }

    public static PtNDArray rot90(PtNDArray ndArray, int times, int[] axes) {
        long[] longaxes = Arrays.stream(axes).mapToLong(i -> i).toArray();
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchRot90(ndArray.getHandle(), times, longaxes));
    }

    public static PtNDArray sum(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchSum(ndArray.getHandle()));
    }

    public static PtNDArray sum(PtNDArray ndArray, long[] dims, boolean keepDim) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchSum(ndArray.getHandle(), dims, keepDim));
    }

    public static PtNDArray cumProd(PtNDArray ndArray, long dim, DataType dataType) {
        int dtPosition = -1;
        if (dataType != null) {
            dtPosition = dataType.ordinal();
        }
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchCumProd(ndArray.getHandle(), dim, dtPosition));
    }

    public static PtNDArray prod(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchProd(ndArray.getHandle()));
    }

    public static PtNDArray prod(PtNDArray ndArray, long dim, boolean keepDim) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchProd(ndArray.getHandle(), dim, keepDim));
    }

    public static PtNDArray cumSum(PtNDArray ndArray, long dim) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchCumSum(ndArray.getHandle(), dim));
    }

    public static PtNDArray diagonal(PtNDArray ndArray, long offset, long axis1, long axis2) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchDiagonal(ndArray.getHandle(), offset, axis1, axis2));
    }

    public static PtNDArray oneHot(PtNDArray ndArray, int depth, DataType dataType) {
        return new PtNDArray(
                        ndArray.getManager(),
                        PyTorchLibrary.LIB.torchNNOneHot(
                                ndArray.toType(DataType.INT64, false).getHandle(), depth))
                .toType(dataType, false);
    }

    public static NDList split(PtNDArray ndArray, long size, long axis) {
        long[] ndPtrs = PyTorchLibrary.LIB.torchSplit(ndArray.getHandle(), size, axis);
        NDList list = new NDList();
        for (long ptr : ndPtrs) {
            list.add(new PtNDArray(ndArray.getManager(), ptr));
        }
        return list;
    }

    public static NDList split(PtNDArray ndArray, long[] indices, long axis) {
        long[] ndPtrs = PyTorchLibrary.LIB.torchSplit(ndArray.getHandle(), indices, axis);
        NDList list = new NDList();
        for (long ptr : ndPtrs) {
            list.add(new PtNDArray(ndArray.getManager(), ptr));
        }
        return list;
    }

    public static PtNDArray squeeze(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchSqueeze(ndArray.getHandle()));
    }

    public static PtNDArray squeeze(PtNDArray ndArray, long dim) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchSqueeze(ndArray.getHandle(), dim));
    }

    public static PtNDArray unsqueeze(PtNDArray ndArray, long dim) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchUnsqueeze(ndArray.getHandle(), dim));
    }

    public static NDList unique(
            PtNDArray ndArray,
            Integer dim,
            boolean sorted,
            boolean returnInverse,
            boolean returnCounts) {
        long[] handles;
        if (dim == null) {
            // In this case the output will be flattened.
            handles =
                    PyTorchLibrary.LIB.torchUnique(
                            ndArray.getHandle(), -1, sorted, returnInverse, returnCounts);
        } else {
            // Dimension wrap
            dim = Math.floorMod(dim, ndArray.getShape().dimension());
            handles =
                    PyTorchLibrary.LIB.torchUnique(
                            ndArray.getHandle(), dim, sorted, returnInverse, returnCounts);
        }
        NDList list = new NDList(handles.length);
        for (long handle : handles) {
            PtNDArray array = new PtNDArray(ndArray.getManager(), handle);
            list.add(array);
        }
        return list;
    }

    public static PtNDArray flatten(PtNDArray ndArray, long startDim, long endDim) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchFlatten(ndArray.getHandle(), startDim, endDim));
    }

    public static PtNDArray fft(PtNDArray ndArray, long length, long axis) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchFft(ndArray.getHandle(), length, axis));
    }

    public static PtNDArray ifft(PtNDArray ndArray, long length, long axis) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchIfft(ndArray.getHandle(), length, axis));
    }

    public static PtNDArray rfft(PtNDArray ndArray, long length, long axis) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchRfft(ndArray.getHandle(), length, axis));
    }

    public static PtNDArray irfft(PtNDArray ndArray, long length, long axis) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchIrfft(ndArray.getHandle(), length, axis));
    }

    public static PtNDArray stft(
            PtNDArray ndArray,
            long nFft,
            long hopLength,
            PtNDArray window,
            boolean center,
            boolean normalize,
            boolean returnComplex) {
        long handle =
                PyTorchLibrary.LIB.torchStft(
                        ndArray.getHandle(),
                        nFft,
                        hopLength,
                        window.getHandle(),
                        center,
                        normalize,
                        returnComplex);
        if (handle == -1) {
            throw new UnsupportedOperationException("real() is not supported.");
        }
        return new PtNDArray(ndArray.getManager(), handle);
    }

    public static PtNDArray fft2(PtNDArray ndArray, long[] sizes, long[] axes) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchFft2(ndArray.getHandle(), sizes, axes));
    }

    public static PtNDArray ifft2(PtNDArray ndArray, long[] sizes, long[] axes) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchIfft2(ndArray.getHandle(), sizes, axes));
    }

    public static PtNDArray real(PtNDArray ndArray) {
        long handle = PyTorchLibrary.LIB.torchViewAsReal(ndArray.getHandle());
        if (handle == -1) {
            throw new UnsupportedOperationException("real() is not supported.");
        }
        return new PtNDArray(ndArray.getManager(), handle);
    }

    public static PtNDArray complex(PtNDArray ndArray) {
        long handle = PyTorchLibrary.LIB.torchViewAsComplex(ndArray.getHandle());
        if (handle == -1) {
            throw new UnsupportedOperationException("complex() is not supported.");
        }
        return new PtNDArray(ndArray.getManager(), handle);
    }

    public static PtNDArray conj(PtNDArray ndArray) {
        return new PtNDArray(ndArray.getManager(), PyTorchLibrary.LIB.conj(ndArray.getHandle()));
    }

    public static PtNDArray abs(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchAbs(ndArray.getHandle()));
    }

    public static PtNDArray square(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchSquare(ndArray.getHandle()));
    }

    public static PtNDArray floor(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchFloor(ndArray.getHandle()));
    }

    public static PtNDArray ceil(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchCeil(ndArray.getHandle()));
    }

    public static PtNDArray round(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchRound(ndArray.getHandle()));
    }

    public static PtNDArray trunc(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchTrunc(ndArray.getHandle()));
    }

    public static PtNDArray clip(PtNDArray ndArray, Number min, Number max) {
        PtNDArray minNd = (PtNDArray) ndArray.getManager().create(min);
        PtNDArray maxNd = (PtNDArray) ndArray.getManager().create(max);
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchClamp(
                        ndArray.getHandle(), minNd.getHandle(), maxNd.getHandle()));
    }

    public static PtNDArray exp(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchExp(ndArray.getHandle()));
    }

    public static PtNDArray gammaln(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchLgamma(ndArray.getHandle()));
    }

    public static PtNDArray log(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchLog(ndArray.getHandle()));
    }

    public static PtNDArray log10(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchLog10(ndArray.getHandle()));
    }

    public static PtNDArray log2(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchLog2(ndArray.getHandle()));
    }

    public static PtNDArray sin(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchSin(ndArray.getHandle()));
    }

    public static PtNDArray cos(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchCos(ndArray.getHandle()));
    }

    public static PtNDArray tan(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchTan(ndArray.getHandle()));
    }

    public static PtNDArray asin(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchASin(ndArray.getHandle()));
    }

    public static PtNDArray acos(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchAcos(ndArray.getHandle()));
    }

    public static PtNDArray atan(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchAtan(ndArray.getHandle()));
    }

    public static PtNDArray atan2(PtNDArray self, PtNDArray other) {
        return new PtNDArray(
                self.getManager(),
                PyTorchLibrary.LIB.torchAtan2(self.getHandle(), other.getHandle()));
    }

    public static PtNDArray sqrt(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchSqrt(ndArray.getHandle()));
    }

    public static PtNDArray sinh(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchSinh(ndArray.getHandle()));
    }

    public static PtNDArray cosh(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchCosh(ndArray.getHandle()));
    }

    public static PtNDArray tanh(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchTanh(ndArray.getHandle()));
    }

    public static PtNDArray sigmoid(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchSigmoid(ndArray.getHandle()));
    }

    public static PtNDArray silu(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchSilu(ndArray.getHandle()));
    }

    public static PtNDArray all(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchAll(ndArray.getHandle()));
    }

    public static PtNDArray any(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchAny(ndArray.getHandle()));
    }

    public static PtNDArray none(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchNone(ndArray.getHandle()));
    }

    public static PtNDArray eq(PtNDArray self, PtNDArray other) {
        return new PtNDArray(
                self.getManager(), PyTorchLibrary.LIB.torchEq(self.getHandle(), other.getHandle()));
    }

    public static PtNDArray eq(PtNDArray self, Number value) {
        return new PtNDArray(
                self.getManager(),
                PyTorchLibrary.LIB.torchEqScalar(
                        self.getHandle(),
                        value.longValue(),
                        value.doubleValue(),
                        isFloatingScalar(value)));
    }

    public static PtNDArray neq(PtNDArray self, PtNDArray other) {
        return new PtNDArray(
                self.getManager(),
                PyTorchLibrary.LIB.torchNeq(self.getHandle(), other.getHandle()));
    }

    public static PtNDArray neq(PtNDArray self, Number value) {
        return new PtNDArray(
                self.getManager(),
                PyTorchLibrary.LIB.torchNeqScalar(
                        self.getHandle(),
                        value.longValue(),
                        value.doubleValue(),
                        isFloatingScalar(value)));
    }

    public static PtNDArray gt(PtNDArray self, PtNDArray other) {
        return new PtNDArray(
                self.getManager(), PyTorchLibrary.LIB.torchGt(self.getHandle(), other.getHandle()));
    }

    public static PtNDArray gt(PtNDArray self, Number value) {
        return new PtNDArray(
                self.getManager(),
                PyTorchLibrary.LIB.torchGtScalar(
                        self.getHandle(),
                        value.longValue(),
                        value.doubleValue(),
                        isFloatingScalar(value)));
    }

    public static PtNDArray gte(PtNDArray self, PtNDArray other) {
        return new PtNDArray(
                self.getManager(),
                PyTorchLibrary.LIB.torchGte(self.getHandle(), other.getHandle()));
    }

    public static PtNDArray gte(PtNDArray self, Number value) {
        return new PtNDArray(
                self.getManager(),
                PyTorchLibrary.LIB.torchGteScalar(
                        self.getHandle(),
                        value.longValue(),
                        value.doubleValue(),
                        isFloatingScalar(value)));
    }

    public static PtNDArray lt(PtNDArray self, PtNDArray other) {
        return new PtNDArray(
                self.getManager(), PyTorchLibrary.LIB.torchLt(self.getHandle(), other.getHandle()));
    }

    public static PtNDArray lt(PtNDArray self, Number value) {
        return new PtNDArray(
                self.getManager(),
                PyTorchLibrary.LIB.torchLtScalar(
                        self.getHandle(),
                        value.longValue(),
                        value.doubleValue(),
                        isFloatingScalar(value)));
    }

    public static PtNDArray lte(PtNDArray self, PtNDArray other) {
        return new PtNDArray(
                self.getManager(),
                PyTorchLibrary.LIB.torchLte(self.getHandle(), other.getHandle()));
    }

    public static PtNDArray lte(PtNDArray self, Number value) {
        return new PtNDArray(
                self.getManager(),
                PyTorchLibrary.LIB.torchLteScalar(
                        self.getHandle(),
                        value.longValue(),
                        value.doubleValue(),
                        isFloatingScalar(value)));
    }

    public static PtNDArray neg(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchNeg(ndArray.getHandle()));
    }

    public static void negi(PtNDArray ndArray) {
        PyTorchLibrary.LIB.torchNegi(ndArray.getHandle());
    }

    public static PtNDArray isNaN(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchIsNaN(ndArray.getHandle()));
    }

    public static PtNDArray isInf(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchIsInf(ndArray.getHandle()));
    }

    public static PtNDArray randint(
            PtNDManager manager,
            long low,
            long high,
            Shape size,
            DataType dataType,
            Device device) {
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchRandint(
                        low,
                        high,
                        size.getShape(),
                        dataType.ordinal(),
                        layoutMapper(SparseFormat.DENSE, device),
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false));
    }

    public static PtNDArray randperm(
            PtNDManager manager, long n, DataType dataType, Device device) {
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchRandPerm(
                        n,
                        dataType.ordinal(),
                        layoutMapper(SparseFormat.DENSE, device),
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false));
    }

    public static PtNDArray normal(
            PtNDManager manager,
            double mean,
            double std,
            Shape size,
            DataType dataType,
            Device device) {
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchNormal(
                        mean,
                        std,
                        size.getShape(),
                        dataType.ordinal(),
                        layoutMapper(SparseFormat.DENSE, device),
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false));
    }

    public static PtNDArray uniform(
            PtNDManager manager,
            double low,
            double high,
            Shape size,
            DataType dataType,
            Device device) {
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.tensorUniform(
                        low,
                        high,
                        size.getShape(),
                        dataType.ordinal(),
                        layoutMapper(SparseFormat.DENSE, device),
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false));
    }

    public static PtNDArray eye(
            PtNDManager manager, int n, int m, DataType dataType, Device device, SparseFormat fmt) {
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchEye(
                        n,
                        m,
                        dataType.ordinal(),
                        layoutMapper(fmt, device),
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false));
    }

    public static PtNDArray hannWindow(
            PtNDManager manager, long numPoints, boolean periodic, Device device) {
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchHannWindow(
                        numPoints,
                        periodic,
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()}));
    }

    public static PtNDArray erfinv(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchErfinv(ndArray.getHandle()));
    }

    public static PtNDArray erf(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchErf(ndArray.getHandle()));
    }

    public static PtNDArray inverse(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchInverse(ndArray.getHandle()));
    }

    public static PtNDArray interpolate(
            PtNDArray ndArray, long[] size, int mode, boolean alignCorners) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNNInterpolate(
                        ndArray.getHandle(), size, mode, alignCorners));
    }

    public static PtNDArray linear(PtNDArray input, PtNDArray weight, PtNDArray bias) {
        return new PtNDArray(
                input.getManager(),
                PyTorchLibrary.LIB.torchNNLinear(
                        input.getHandle(),
                        weight.getHandle(),
                        bias == null ? NULL_PTR : bias.getHandle()));
    }

    public static PtNDArray projectedResidualMlp(
            PtNDArray input,
            PtNDArray combinedWeight,
            PtNDArray combinedBias,
            PtNDArray outputWeight) {
        return new PtNDArray(
                input.getManager(),
                PyTorchLibrary.LIB.torchNNProjectedResidualMlp(
                        input.getHandle(),
                        combinedWeight.getHandle(),
                        combinedBias.getHandle(),
                        outputWeight.getHandle()));
    }

    public static PtNDArray embedding(PtNDArray input, PtNDArray weight, boolean sparse) {
        return new PtNDArray(
                input.getManager(),
                PyTorchLibrary.LIB.torchNNEmbedding(input.getHandle(), weight.getHandle(), sparse));
    }

    public static PtNDArray relu(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchNNRelu(ndArray.getHandle()));
    }

    public static PtNDArray softPlus(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchNNSoftPlus(ndArray.getHandle()));
    }

    public static PtNDArray softSign(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchNNSoftSign(ndArray.getHandle()));
    }

    public static PtNDArray leakyRelu(PtNDArray ndArray, double negativeSlope) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNNLeakyRelu(ndArray.getHandle(), negativeSlope));
    }

    public static PtNDArray elu(PtNDArray ndArray, double alpha) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchNNElu(ndArray.getHandle(), alpha));
    }

    public static PtNDArray selu(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchNNSelu(ndArray.getHandle()));
    }

    public static PtNDArray gelu(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchNNGelu(ndArray.getHandle()));
    }

    public static PtNDArray convolution(
            PtNDArray ndArray,
            PtNDArray weight,
            PtNDArray bias,
            Shape stride,
            Shape padding,
            Shape dilation,
            int groups) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNNConvNd(
                        ndArray.getHandle(),
                        weight.getHandle(),
                        (bias != null) ? bias.getHandle() : NULL_PTR,
                        stride.getShape(),
                        padding.getShape(),
                        dilation.getShape(),
                        groups));
    }

    public static PtNDArray batchNorm(
            PtNDArray ndArray,
            PtNDArray gamma,
            PtNDArray beta,
            PtNDArray runningMean,
            PtNDArray runningVar,
            boolean isTraining,
            double momentum,
            double eps) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNNBatchNorm(
                        ndArray.getHandle(),
                        gamma.getHandle(),
                        beta.getHandle(),
                        runningMean.getHandle(),
                        runningVar.getHandle(),
                        isTraining,
                        momentum,
                        eps));
    }

    public static PtNDArray layerNorm(
            PtNDArray ndArray, Shape normalizedShape, PtNDArray gamma, PtNDArray beta, double eps) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNNLayerNorm(
                        ndArray.getHandle(),
                        normalizedShape.getShape(),
                        gamma.getHandle(),
                        beta.getHandle(),
                        eps));
    }

    /** Adds a residual update and returns the normalized output followed by the sum. */
    public static NDList residualAddLayerNorm(
            PtNDArray residual,
            PtNDArray update,
            Shape normalizedShape,
            PtNDArray gamma,
            PtNDArray beta,
            double eps) {
        long[] handles =
                PyTorchLibrary.LIB.torchNNResidualAddLayerNorm(
                        residual.getHandle(),
                        update.getHandle(),
                        normalizedShape.getShape(),
                        gamma.getHandle(),
                        beta.getHandle(),
                        eps);
        PtNDManager manager = residual.getManager();
        return new NDList(new PtNDArray(manager, handles[0]), new PtNDArray(manager, handles[1]));
    }

    /** Applies LayerNorm and returns its ordinary output together with a converted copy. */
    public static NDList layerNormAndCast(
            PtNDArray ndArray,
            Shape normalizedShape,
            PtNDArray gamma,
            PtNDArray beta,
            double eps,
            DataType convertedDataType) {
        long[] handles =
                PyTorchLibrary.LIB.torchNNLayerNormAndCast(
                        ndArray.getHandle(),
                        normalizedShape.getShape(),
                        gamma.getHandle(),
                        beta.getHandle(),
                        eps,
                        convertedDataType.ordinal());
        PtNDManager manager = ndArray.getManager();
        return new NDList(new PtNDArray(manager, handles[0]), new PtNDArray(manager, handles[1]));
    }

    public static PtNDArray normalize(PtNDArray ndArray, double p, long dim, double eps) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNNNormalize(ndArray.getHandle(), p, dim, eps));
    }

    public static PtNDArray dropout(PtNDArray ndArray, double prob, boolean training) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNNDropout(ndArray.getHandle(), prob, training));
    }

    public static NDList rnn(
            PtNDArray input,
            PtNDArray hx,
            NDList params,
            boolean hasBiases,
            int numLayers,
            RNN.Activation activation,
            double dropRate,
            boolean training,
            boolean bidirectional,
            boolean batchFirst) {
        PtNDManager manager = input.getManager();
        long[] paramHandles =
                params.stream().mapToLong(array -> ((PtNDArray) array).getHandle()).toArray();
        long[] outputs =
                PyTorchLibrary.LIB.torchNNRnn(
                        input.getHandle(),
                        hx.getHandle(),
                        paramHandles,
                        hasBiases,
                        numLayers,
                        activation.ordinal(),
                        dropRate,
                        training,
                        bidirectional,
                        batchFirst);
        NDList res = new NDList();
        for (long output : outputs) {
            res.add(new PtNDArray(manager, output));
        }
        return res;
    }

    public static NDList gru(
            PtNDArray input,
            PtNDArray hx,
            NDList params,
            boolean hasBiases,
            int numLayers,
            double dropRate,
            boolean training,
            boolean bidirectional,
            boolean batchFirst) {
        PtNDManager manager = input.getManager();
        long[] paramHandles =
                params.stream().mapToLong(array -> ((PtNDArray) array).getHandle()).toArray();
        long[] outputs =
                PyTorchLibrary.LIB.torchNNGru(
                        input.getHandle(),
                        hx.getHandle(),
                        paramHandles,
                        hasBiases,
                        numLayers,
                        dropRate,
                        training,
                        bidirectional,
                        batchFirst);
        NDList res = new NDList();
        for (long output : outputs) {
            res.add(new PtNDArray(manager, output));
        }
        return res;
    }

    public static NDList lstm(
            PtNDArray input,
            NDList hx,
            NDList params,
            boolean hasBiases,
            int numLayers,
            double dropRate,
            boolean training,
            boolean bidirectional,
            boolean batchFirst) {
        PtNDManager manager = input.getManager();
        long[] hxHandles =
                hx.stream().mapToLong(array -> ((PtNDArray) array).getHandle()).toArray();
        long[] paramHandles =
                params.stream().mapToLong(array -> ((PtNDArray) array).getHandle()).toArray();
        long[] outputs =
                PyTorchLibrary.LIB.torchNNLstm(
                        input.getHandle(),
                        hxHandles,
                        paramHandles,
                        hasBiases,
                        numLayers,
                        dropRate,
                        training,
                        bidirectional,
                        batchFirst);
        NDList res = new NDList();
        for (long output : outputs) {
            res.add(new PtNDArray(manager, output));
        }
        return res;
    }

    public static PtNDArray avgPool(
            PtNDArray ndArray,
            Shape kernelSize,
            Shape stride,
            Shape padding,
            boolean ceilMode,
            boolean countIncludePad) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNNAvgPool(
                        ndArray.getHandle(),
                        kernelSize.getShape(),
                        stride.getShape(),
                        padding.getShape(),
                        ceilMode,
                        countIncludePad));
    }

    public static PtNDArray maxPool(
            PtNDArray ndArray, Shape kernelSize, Shape stride, Shape padding, boolean ceilMode) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNNMaxPool(
                        ndArray.getHandle(),
                        kernelSize.getShape(),
                        stride.getShape(),
                        padding.getShape(),
                        ceilMode));
    }

    public static PtNDArray adaptiveMaxPool(PtNDArray ndArray, Shape outputSize) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNNAdaptiveMaxPool(
                        ndArray.getHandle(), outputSize.getShape()));
    }

    public static PtNDArray adaptiveAvgPool(PtNDArray ndArray, Shape outputSize) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNNAdaptiveAvgPool(
                        ndArray.getHandle(), outputSize.getShape()));
    }

    public static PtNDArray lpPool(
            PtNDArray ndArray, double normType, Shape kernelSize, Shape stride, boolean ceilMode) {
        if (ndArray.getShape().dimension() - 2 == 3) {
            throw new UnsupportedOperationException("3D lpPool is not supported in PyTorch engine");
        }
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNNLpPool(
                        ndArray.getHandle(),
                        normType,
                        kernelSize.getShape(),
                        stride.getShape(),
                        ceilMode));
    }

    public static DataType getDataType(PtNDArray ndArray) {
        int dataType = PyTorchLibrary.LIB.torchDType(ndArray.getHandle());
        return DataType.values()[dataType];
    }

    public static Device getDevice(PtNDArray ndArray) {
        int[] device = PyTorchLibrary.LIB.torchDevice(ndArray.getHandle());
        String deviceType = PtDeviceType.fromDeviceType(device[0]);
        return Device.of(deviceType, device[1]);
    }

    public static SparseFormat getSparseFormat(PtNDArray ndArray) {
        int layout = PyTorchLibrary.LIB.torchLayout(ndArray.getHandle());
        if (layout == 0) {
            return SparseFormat.DENSE;
        } else if (layout == 1) {
            return SparseFormat.COO;
        } else if (layout == 2) {
            logger.debug("MKLDNN layout is used!");
            return SparseFormat.DENSE;
        }
        throw new UnsupportedOperationException("Unsupported data format");
    }

    public static Shape getShape(PtNDArray ndArray) {
        return new Shape(PyTorchLibrary.LIB.torchSizes(ndArray.getHandle()));
    }

    public static boolean isContiguous(PtNDArray ndArray) {
        return PyTorchLibrary.LIB.torchIsContiguous(ndArray.getHandle());
    }

    public static ByteBuffer getByteBuffer(PtNDArray ndArray, boolean tryDirect) {
        if (ndArray.getDevice().equals(Device.cpu())) {
            if (tryDirect
                    && !ndArray.isSparse()
                    && getLayout(ndArray) != 2
                    && PyTorchLibrary.LIB.torchIsContiguous(ndArray.getHandle())) {
                return PyTorchLibrary.LIB
                        .torchDirectByteBuffer(ndArray.getHandle())
                        .order(ByteOrder.nativeOrder());
            }
            return ByteBuffer.wrap(PyTorchLibrary.LIB.torchDataPtr(ndArray.getHandle()))
                    .order(ByteOrder.nativeOrder());
        }

        PtNDArray cpuArray = ndArray.toDevice(Device.cpu(), false);
        try {
            return ByteBuffer.wrap(PyTorchLibrary.LIB.torchDataPtr(cpuArray.getHandle()))
                    .order(ByteOrder.nativeOrder());
        } finally {
            PtNDManager manager = cpuArray.getManager();
            if (manager == ndArray.getManager()) {
                cpuArray.close();
            } else {
                manager.close();
            }
        }
    }

    public static void deleteNDArray(long handle) {
        PyTorchLibrary.LIB.torchDeleteTensor(handle);
    }

    public static boolean requiresGrad(PtNDArray ndArray) {
        return PyTorchLibrary.LIB.torchRequiresGrad(ndArray.getHandle());
    }

    public static String getGradientFunctionNames(PtNDArray ndArray) {
        return PyTorchLibrary.LIB.torchGradFnName(ndArray.getHandle());
    }

    public static void attachGradient(PtNDArray ndArray, boolean requiresGrad) {
        PyTorchLibrary.LIB.torchAttachGrad(ndArray.getHandle(), requiresGrad);
    }

    public static PtNDArray detachGradient(PtNDArray ndArray) {
        // TODO: detached ndarray may not use the same manager for the attached one
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchDetachGrad(ndArray.getHandle()));
    }

    public static PtNDArray getGradient(PtNDArray ndArray) {
        long pointer = PyTorchLibrary.LIB.torchGrad(ndArray.getHandle());
        if (pointer == NULL_PTR) {
            return null;
        }
        return new PtNDArray(ndArray.getManager(), pointer);
    }

    public static void backward(
            PtNDArray ndArray, PtNDArray gradNd, boolean keepGraph, boolean createGraph) {
        PyTorchLibrary.LIB.torchBackward(
                ndArray.getHandle(), gradNd.getHandle(), keepGraph, createGraph);
    }

    public static long createFlatGradientAccumulator(PtNDArray[] parameters, PtNDArray gradient) {
        long[] parameterHandles =
                Arrays.stream(parameters).mapToLong(PtNDArray::getHandle).toArray();
        return PyTorchLibrary.LIB.torchCreateFlatGradientAccumulator(
                parameterHandles, gradient.getHandle());
    }

    public static void backwardFlatGradientAccumulator(
            long accumulatorHandle, PtNDArray target, PtNDArray targetGradient) {
        PyTorchLibrary.LIB.torchFlatGradientAccumulatorBackward(
                accumulatorHandle, target.getHandle(), targetGradient.getHandle());
    }

    public static void zeroFlatGradientAccumulator(long accumulatorHandle) {
        PyTorchLibrary.LIB.torchZeroFlatGradientAccumulator(accumulatorHandle);
    }

    public static void deleteFlatGradientAccumulator(long accumulatorHandle) {
        PyTorchLibrary.LIB.torchDeleteFlatGradientAccumulator(accumulatorHandle);
    }

    public static long createFlatGradientPacker(PtNDArray[] parameters, PtNDArray destination) {
        long[] parameterHandles =
                Arrays.stream(parameters).mapToLong(PtNDArray::getHandle).toArray();
        return PyTorchLibrary.LIB.torchCreateFlatGradientPacker(
                parameterHandles, destination.getHandle());
    }

    public static void packAndClearFlatGradientPacker(
            long packerHandle, boolean zeroMissingGradients) {
        PyTorchLibrary.LIB.torchFlatGradientPackerPackAndClear(packerHandle, zeroMissingGradients);
    }

    public static void accumulateAndClearFlatGradientPacker(
            long packerHandle, boolean zeroMissingGradients) {
        PyTorchLibrary.LIB.torchFlatGradientPackerAccumulateAndClear(
                packerHandle, zeroMissingGradients);
    }

    public static void zeroFlatGradientPackerDestination(long packerHandle) {
        PyTorchLibrary.LIB.torchZeroFlatGradientPackerDestination(packerHandle);
    }

    public static void clearFlatGradientPackerParameterGradients(long packerHandle) {
        PyTorchLibrary.LIB.torchClearFlatGradientPackerParameterGradients(packerHandle);
    }

    public static void deleteFlatGradientPacker(long packerHandle) {
        PyTorchLibrary.LIB.torchDeleteFlatGradientPacker(packerHandle);
    }

    public static long distributedCreateReducer(
            long[] parameterHandles,
            String masterHost,
            int masterPort,
            int rank,
            int worldSize,
            int localRank,
            int bucketCapMb,
            boolean staticGraph,
            boolean findUnusedParameters,
            boolean averageGradients) {
        return PyTorchLibrary.LIB.distributedCreateReducer(
                parameterHandles,
                masterHost,
                masterPort,
                rank,
                worldSize,
                localRank,
                bucketCapMb,
                staticGraph,
                findUnusedParameters,
                averageGradients);
    }

    public static void distributedPrepareForBackward(long reducerHandle) {
        PyTorchLibrary.LIB.distributedPrepareForBackward(reducerHandle);
    }

    public static void distributedFinalizeBackward(long reducerHandle) {
        PyTorchLibrary.LIB.distributedFinalizeBackward(reducerHandle);
    }

    public static void distributedDeleteReducer(long reducerHandle) {
        PyTorchLibrary.LIB.distributedDeleteReducer(reducerHandle);
    }

    public static void deleteModule(long pointer) {
        PyTorchLibrary.LIB.torchDeleteModule(pointer);
    }

    public static void setGraphExecutorOptimize(boolean enabled) {
        PyTorchLibrary.LIB.setGraphExecutorOptimize(enabled);
    }

    public static PtSymbolBlock loadModule(
            PtNDManager manager,
            Path path,
            boolean mapLocation,
            String[] extraFileKeys,
            String[] extraFileValues,
            boolean trainParam) {
        Device device = manager.getDevice();
        // MPS doesn't support mapLocation
        if ("mps".equals(device.getDeviceType())) {
            mapLocation = false;
        }
        logger.debug("mapLocation: {}", mapLocation);
        logger.debug("extraFileKeys: {}", Arrays.toString(extraFileKeys));
        long handle =
                PyTorchLibrary.LIB.moduleLoad(
                        path.toString(),
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        mapLocation,
                        extraFileKeys,
                        extraFileValues,
                        trainParam);
        return new PtSymbolBlock(manager, handle);
    }

    public static PtSymbolBlock loadModule(
            PtNDManager manager, InputStream is, boolean mapLocation, boolean hasSize)
            throws IOException {
        long handle = loadModuleHandle(is, manager.getDevice(), mapLocation, hasSize);
        return new PtSymbolBlock(manager, handle);
    }

    public static long loadModuleHandle(
            InputStream is, Device device, boolean mapLocation, boolean hasSize)
            throws IOException {
        byte[] buf = new byte[BYTE_LENGTH];
        long size = -1;
        if (hasSize) {
            size = new DataInputStream(is).readLong();
        }
        // MPS doesn't support mapLocation
        if ("mps".equals(device.getDeviceType())) {
            mapLocation = false;
        }
        logger.debug("mapLocation: {}", mapLocation);
        return PyTorchLibrary.LIB.moduleLoad(
                is,
                new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                mapLocation,
                buf,
                size);
    }

    public static void writeModule(PtSymbolBlock block, OutputStream os, boolean writeSize) {
        byte[] buf = new byte[BYTE_LENGTH];
        PyTorchLibrary.LIB.moduleWrite(block.getHandle(), os, buf, writeSize);
    }

    public static NDList moduleGetParams(PtSymbolBlock block, PtNDManager manager) {
        long[] handles = PyTorchLibrary.LIB.moduleGetParams(block.getHandle());
        String[] names = PyTorchLibrary.LIB.moduleGetParamNames(block.getHandle());
        NDList list = new NDList(handles.length);
        for (int i = 0; i < handles.length; i++) {
            PtNDArray array = new PtNDArray(manager, handles[i]);
            array.setName(names[i]);
            list.add(array);
        }
        return list;
    }

    public static String[] getMethodNames(PtSymbolBlock block) {
        return PyTorchLibrary.LIB.moduleGetMethodNames(block.getHandle());
    }

    public static void enableInferenceMode(PtSymbolBlock block) {
        PyTorchLibrary.LIB.moduleEval(block.getHandle());
    }

    public static void enableTrainingMode(PtSymbolBlock block) {
        PyTorchLibrary.LIB.moduleTrain(block.getHandle());
    }

    public static void zeroGrad(PtNDArray weight) {
        PyTorchLibrary.LIB.zeroGrad(weight.getHandle());
    }

    public static boolean unscaleGradients(List<PtNDArray> gradients, float inverseScale) {
        long[] handles = gradients.stream().mapToLong(PtNDArray::getHandle).toArray();
        // The native name is retained for JNI binary compatibility.
        return PyTorchLibrary.LIB.torchUnscaleGradientsAndCheckFinite(handles, inverseScale);
    }

    public static void adamUpdate(
            PtNDArray weight,
            PtNDArray grad,
            PtNDArray mean,
            PtNDArray variance,
            float lr,
            float learningRateBiasCorrection,
            float wd,
            float rescaleGrad,
            float clipGrad,
            float beta1,
            float beta2,
            float eps,
            boolean adamw) {
        PyTorchLibrary.LIB.adamUpdate(
                weight.getHandle(),
                grad.getHandle(),
                mean.getHandle(),
                variance.getHandle(),
                lr,
                learningRateBiasCorrection,
                wd,
                rescaleGrad,
                clipGrad,
                beta1,
                beta2,
                eps,
                adamw);
    }

    public static void sgdUpdate(
            PtNDArray weight,
            PtNDArray grad,
            PtNDArray state,
            float lr,
            float wd,
            float rescaleGrad,
            float clipGrad,
            float momentum) {
        PyTorchLibrary.LIB.sgdUpdate(
                weight.getHandle(),
                grad.getHandle(),
                (state == null) ? NULL_PTR : state.getHandle(),
                lr,
                wd,
                rescaleGrad,
                clipGrad,
                momentum);
    }

    // Internal use only
    public static int getLayout(PtNDArray array) {
        return PyTorchLibrary.LIB.torchLayout(array.getHandle());
    }

    public static PtNDArray norm(PtNDArray ndArray, int ord, int[] axes, boolean keepDims) {
        long[] longAxes = Arrays.stream(axes).mapToLong(i -> i).toArray();
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNorm(ndArray.getHandle(), ord, longAxes, keepDims));
    }

    public static PtNDArray nonZeros(PtNDArray ndArray) {
        if (ndArray.isScalar()) {
            ndArray = (PtNDArray) ndArray.reshape(-1);
        }
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchNonZeros(ndArray.getHandle()));
    }

    public static PtNDArray diff(PtNDArray ndArray, int n, int dim) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchDiff(ndArray.getHandle(), n, dim));
    }
}
