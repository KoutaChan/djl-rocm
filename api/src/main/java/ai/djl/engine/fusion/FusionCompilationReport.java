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
package ai.djl.engine.fusion;

import java.util.Objects;

/**
 * Immutable diagnostics produced when a {@link FusionRecipe} is prepared.
 *
 * <p>Storage sizes are logical tensor payload bytes. They exclude allocator alignment and rounding,
 * tensor metadata, stream and event objects, and backend-library scratch storage from the logical
 * execution image. The separately reported backend-workspace upper bound must not be added to that
 * image; it is owned by an execution lane and shared by sessions on the same device stream.
 * Workspace is a subset of the execution image and must not be added to it. Exported output bytes
 * are retained for each output slot. Planner arenas are also owned by the execution lane.
 * Caller-owned inputs and constants are excluded.
 */
public final class FusionCompilationReport {

    private final String backend;
    private final int commandCount;
    private final long executableStorageBytes;
    private final long executionStorageBytes;
    private final long workspaceBytes;
    private final long exportedOutputBytes;
    private final long arenaBytes;
    private final long backendWorkspaceUpperBoundBytes;
    private final int storagePlannerVersion;
    private final long logicalAllocationCount;
    private final long backingAllocationCount;
    private final long aliasViewCount;
    private final long inPlaceReuseCount;
    private final boolean nativeOnly;

    private FusionCompilationReport(Builder builder) {
        backend = builder.backend;
        commandCount = builder.commandCount;
        executableStorageBytes = builder.executableStorageBytes;
        executionStorageBytes = builder.executionStorageBytes;
        workspaceBytes = builder.workspaceBytes;
        exportedOutputBytes = builder.exportedOutputBytes;
        arenaBytes = builder.arenaBytes;
        backendWorkspaceUpperBoundBytes = builder.backendWorkspaceUpperBoundBytes;
        storagePlannerVersion = builder.storagePlannerVersion;
        logicalAllocationCount = builder.logicalAllocationCount;
        backingAllocationCount = builder.backingAllocationCount;
        aliasViewCount = builder.aliasViewCount;
        inPlaceReuseCount = builder.inPlaceReuseCount;
        nativeOnly = builder.nativeOnly;
    }

    /**
     * Creates a compilation report builder.
     *
     * @param backend the backend implementation name
     * @return a new builder
     */
    public static Builder builder(String backend) {
        return new Builder(backend);
    }

    /**
     * Returns the backend implementation name.
     *
     * @return the backend name
     */
    public String getBackend() {
        return backend;
    }

    /**
     * Returns the number of backend commands in the prepared plan.
     *
     * @return the backend command count
     */
    public int getCommandCount() {
        return commandCount;
    }

    /**
     * Returns backend-owned constants and precomputed values retained by each executable.
     *
     * <p>Caller-owned constant tensors are excluded. This storage is shared by all sessions created
     * from the executable and is not multiplied by the session output slot count.
     *
     * @return the persistent per-executable storage size in bytes
     */
    public long getExecutableStorageBytes() {
        return executableStorageBytes;
    }

    /**
     * Returns the logical storage image required to execute one submission.
     *
     * <p>This combines slot-owned storage and the execution-lane arena required by the plan. Use
     * {@link #getRetainedSessionStorageBytes(int)} to account for multiple output slots without
     * double-counting shared lane storage.
     *
     * @return the one-submission execution storage size in bytes
     */
    public long getExecutionStorageBytes() {
        return executionStorageBytes;
    }

    /**
     * Returns the intermediate workspace bytes required by each execution slot.
     *
     * <p>Named output storage is excluded. The value is therefore a subset of {@link
     * #getExecutionStorageBytes()}.
     *
     * @return the intermediate workspace size in bytes
     */
    public long getWorkspaceBytes() {
        return workspaceBytes;
    }

    /** Returns named output storage retained for the lifetime of each execution slot. */
    public long getExportedOutputBytes() {
        return exportedOutputBytes;
    }

    /** Returns the minimum planner arena capacity required on an execution lane. */
    public long getArenaBytes() {
        return arenaBytes;
    }

    /**
     * Returns the backend-library scratch upper bound for one execution lane.
     *
     * <p>The backend allocates this storage lazily up to the largest algorithm requirement observed
     * on the lane. Sessions on the same device stream share it, and output slot count does not
     * multiply it.
     *
     * @return the per-lane backend workspace upper bound in bytes
     */
    public long getBackendWorkspaceUpperBoundBytes() {
        return backendWorkspaceUpperBoundBytes;
    }

    /**
     * Returns the logical tensor payload retained exclusively by a session.
     *
     * <p>Planner arena storage is excluded because it is owned by the device-stream execution lane.
     * All remaining execution storage is retained once per output slot. Allocator rounding and
     * backend-library workspace remain excluded.
     *
     * @param outputSlotCount the configured positive output slot count
     * @return the session-retained storage size in bytes
     */
    public long getRetainedSessionStorageBytes(int outputSlotCount) {
        if (outputSlotCount <= 0) {
            throw new IllegalArgumentException("The output slot count must be positive.");
        }
        long slotStorage = Math.subtractExact(executionStorageBytes, arenaBytes);
        return Math.multiplyExact(slotStorage, outputSlotCount);
    }

    /**
     * Returns a conservative storage bound required by the shared execution lane.
     *
     * <p>A lane grows to the maximum capacity required by any session using that device stream, so
     * lane requirements from multiple plans must be combined with {@code max}, not addition, when
     * their arena data types match. The lane does not shrink while retained by a session;
     * therefore, selecting a smaller profile does not release capacity already reached by that
     * lane. Its lane-owned storage is released after the final retaining session closes.
     *
     * @return the planner arena plus backend-workspace upper bound in bytes
     */
    public long getRequiredExecutionLaneStorageBytes() {
        return Math.addExact(arenaBytes, backendWorkspaceUpperBoundBytes);
    }

    /**
     * Returns a conservative isolated-session storage upper bound.
     *
     * <p>This adds the plan's conservative execution-lane storage bound to storage retained
     * exclusively by the session. Multiple sessions on the same device stream share that lane
     * capacity, so this value must not be added across such sessions.
     *
     * @param outputSlotCount the configured positive output slot count
     * @return the isolated-session storage estimate in bytes
     */
    public long getSessionStorageBytes(int outputSlotCount) {
        return Math.addExact(
                getRequiredExecutionLaneStorageBytes(),
                getRetainedSessionStorageBytes(outputSlotCount));
    }

    /** Returns the native storage planner version, or zero when disabled. */
    public int getStoragePlannerVersion() {
        return storagePlannerVersion;
    }

    /** Returns whether a native storage planner phase is enabled. */
    public boolean isStoragePlannerEnabled() {
        return storagePlannerVersion > 0;
    }

    /** Returns the logical storage request count before arena lowering. */
    public long getLogicalAllocationCount() {
        return logicalAllocationCount;
    }

    /** Returns physical backing allocations in the one-submission plan image. */
    public long getBackingAllocationCount() {
        return backingAllocationCount;
    }

    /** Returns the tensor view count per execution slot. */
    public long getAliasViewCount() {
        return aliasViewCount;
    }

    /** Returns the number of intermediate blocks handed directly to command results. */
    public long getInPlaceReuseCount() {
        return inPlaceReuseCount;
    }

    /**
     * Returns whether every recipe value was lowered to native backend commands.
     *
     * @return {@code true} if the plan contains no reference fallback
     */
    public boolean isNativeOnly() {
        return nativeOnly;
    }

    /** Builds an immutable {@link FusionCompilationReport}. */
    public static final class Builder {

        private String backend;
        private int commandCount;
        private long executableStorageBytes;
        private long executionStorageBytes;
        private long workspaceBytes;
        private long exportedOutputBytes = -1;
        private long arenaBytes = -1;
        private long backendWorkspaceUpperBoundBytes = -1;
        private int storagePlannerVersion = -1;
        private long logicalAllocationCount = -1;
        private long backingAllocationCount = -1;
        private long aliasViewCount = -1;
        private long inPlaceReuseCount = -1;
        private boolean nativeOnly;

        private Builder(String backend) {
            this.backend = requireBackend(backend);
            nativeOnly = true;
        }

        /**
         * Sets the number of backend commands.
         *
         * @param commandCount the command count
         * @return this builder
         */
        public Builder optCommandCount(int commandCount) {
            if (commandCount < 0) {
                throw new IllegalArgumentException("The command count must not be negative.");
            }
            this.commandCount = commandCount;
            return this;
        }

        /**
         * Sets backend-owned constants and precomputed values retained by each executable.
         *
         * @param executableStorageBytes the persistent per-executable storage size in bytes
         * @return this builder
         */
        public Builder optExecutableStorageBytes(long executableStorageBytes) {
            if (executableStorageBytes < 0) {
                throw new IllegalArgumentException(
                        "The executable storage size must not be negative.");
            }
            this.executableStorageBytes = executableStorageBytes;
            return this;
        }

        /**
         * Sets the logical storage image required to execute one submission.
         *
         * @param executionStorageBytes the one-submission execution storage size in bytes
         * @return this builder
         */
        public Builder optExecutionStorageBytes(long executionStorageBytes) {
            if (executionStorageBytes < 0) {
                throw new IllegalArgumentException(
                        "The execution storage size must not be negative.");
            }
            this.executionStorageBytes = executionStorageBytes;
            return this;
        }

        /**
         * Sets intermediate workspace size for one submission, excluding named output storage.
         *
         * @param workspaceBytes the intermediate workspace size in bytes
         * @return this builder
         */
        public Builder optWorkspaceBytes(long workspaceBytes) {
            if (workspaceBytes < 0) {
                throw new IllegalArgumentException("The workspace size must not be negative.");
            }
            this.workspaceBytes = workspaceBytes;
            return this;
        }

        /** Sets named output storage retained by each execution slot. */
        public Builder optExportedOutputBytes(long exportedOutputBytes) {
            this.exportedOutputBytes = requireNonNegative(exportedOutputBytes, "output storage");
            return this;
        }

        /** Sets planner arena storage required on a shared execution lane. */
        public Builder optArenaBytes(long arenaBytes) {
            this.arenaBytes = requireNonNegative(arenaBytes, "arena storage");
            return this;
        }

        /** Sets backend-library scratch upper bound required on a shared execution lane. */
        public Builder optBackendWorkspaceUpperBoundBytes(long backendWorkspaceUpperBoundBytes) {
            this.backendWorkspaceUpperBoundBytes =
                    requireNonNegative(
                            backendWorkspaceUpperBoundBytes, "backend workspace upper bound");
            return this;
        }

        /** Sets the native storage planner version, or zero when disabled. */
        public Builder optStoragePlannerVersion(int storagePlannerVersion) {
            if (storagePlannerVersion < 0) {
                throw new IllegalArgumentException(
                        "The storage planner version must not be negative.");
            }
            this.storagePlannerVersion = storagePlannerVersion;
            return this;
        }

        /** Sets the logical storage request count. */
        public Builder optLogicalAllocationCount(long logicalAllocationCount) {
            this.logicalAllocationCount =
                    requireNonNegative(logicalAllocationCount, "logical allocation count");
            return this;
        }

        /** Sets the physical backing allocation count. */
        public Builder optBackingAllocationCount(long backingAllocationCount) {
            this.backingAllocationCount =
                    requireNonNegative(backingAllocationCount, "backing allocation count");
            return this;
        }

        /** Sets the tensor view count. */
        public Builder optAliasViewCount(long aliasViewCount) {
            this.aliasViewCount = requireNonNegative(aliasViewCount, "alias view count");
            return this;
        }

        /** Sets the number of intermediate blocks handed directly to command results. */
        public Builder optInPlaceReuseCount(long inPlaceReuseCount) {
            this.inPlaceReuseCount = requireNonNegative(inPlaceReuseCount, "in-place reuse count");
            return this;
        }

        /**
         * Sets whether every recipe value uses native backend commands.
         *
         * @param nativeOnly whether the plan is fully native
         * @return this builder
         */
        public Builder optNativeOnly(boolean nativeOnly) {
            this.nativeOnly = nativeOnly;
            return this;
        }

        /**
         * Builds the compilation report.
         *
         * @return the compilation report
         */
        public FusionCompilationReport build() {
            if (exportedOutputBytes < 0
                    || arenaBytes < 0
                    || backendWorkspaceUpperBoundBytes < 0
                    || storagePlannerVersion < 0
                    || logicalAllocationCount < 0
                    || backingAllocationCount < 0
                    || aliasViewCount < 0
                    || inPlaceReuseCount < 0) {
                throw new IllegalStateException(
                        "Fusion storage planner details must be specified explicitly.");
            }
            if (workspaceBytes > executionStorageBytes) {
                throw new IllegalStateException(
                        "The workspace size must not exceed execution storage.");
            }
            if (exportedOutputBytes > executionStorageBytes
                    || workspaceBytes != executionStorageBytes - exportedOutputBytes) {
                throw new IllegalStateException(
                        "Execution storage must equal output storage plus workspace.");
            }
            if (arenaBytes > workspaceBytes) {
                throw new IllegalStateException("Arena storage must not exceed workspace storage.");
            }
            if (backingAllocationCount > logicalAllocationCount
                    && backingAllocationCount - logicalAllocationCount > aliasViewCount) {
                throw new IllegalStateException("Invalid Fusion allocation counts.");
            }
            if (inPlaceReuseCount > logicalAllocationCount) {
                throw new IllegalStateException(
                        "In-place reuse count must not exceed logical allocations.");
            }
            return new FusionCompilationReport(this);
        }

        private static long requireNonNegative(long value, String name) {
            if (value < 0) {
                throw new IllegalArgumentException("The " + name + " must not be negative.");
            }
            return value;
        }

        private static String requireBackend(String backend) {
            Objects.requireNonNull(backend, "backend");
            if (backend.trim().isEmpty()) {
                throw new IllegalArgumentException("The backend name must not be empty.");
            }
            return backend;
        }
    }
}
