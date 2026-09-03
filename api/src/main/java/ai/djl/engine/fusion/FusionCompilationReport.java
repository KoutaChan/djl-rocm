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
 * tensor metadata, stream and event objects, and backend-library scratch storage. Workspace is a
 * subset of per-slot peak storage and must not be added to it. Exported output bytes remain pinned
 * for each slot; planner arena bytes may be acquired only while a submission is active and reused
 * across plans by the backend allocator. Caller-owned inputs and constants are excluded.
 */
public final class FusionCompilationReport {

    private final String backend;
    private final int commandCount;
    private final long executableStorageBytes;
    private final long persistentStorageBytes;
    private final long workspaceBytes;
    private final long exportedOutputBytes;
    private final long arenaBytes;
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
        persistentStorageBytes = builder.persistentStorageBytes;
        workspaceBytes = builder.workspaceBytes;
        exportedOutputBytes = builder.exportedOutputBytes;
        arenaBytes = builder.arenaBytes;
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
     * from the executable and is not multiplied by the session buffer count.
     *
     * @return the persistent per-executable storage size in bytes
     */
    public long getExecutableStorageBytes() {
        return executableStorageBytes;
    }

    /**
     * Returns peak output and intermediate storage required by each active execution slot.
     *
     * <p>This is a capacity-planning peak, not necessarily the amount retained while the session is
     * idle. {@link #getExportedOutputBytes()} reports the portion pinned for the slot lifetime.
     *
     * @return the peak per-slot storage size in bytes
     */
    public long getPersistentStorageBytes() {
        return persistentStorageBytes;
    }

    /**
     * Returns the intermediate workspace bytes required by each execution slot.
     *
     * <p>Named output storage is excluded. The value is therefore a subset of {@link
     * #getPersistentStorageBytes()}.
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

    /** Returns transient planner arena storage required by each active submission. */
    public long getArenaBytes() {
        return arenaBytes;
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

    /** Returns the physical backing allocation count per execution slot. */
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
        private long persistentStorageBytes;
        private long workspaceBytes;
        private long exportedOutputBytes = -1;
        private long arenaBytes = -1;
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
         * Sets peak output and intermediate storage used by each active execution slot.
         *
         * @param persistentStorageBytes the peak per-slot storage size in bytes
         * @return this builder
         */
        public Builder optPersistentStorageBytes(long persistentStorageBytes) {
            if (persistentStorageBytes < 0) {
                throw new IllegalArgumentException(
                        "The persistent storage size must not be negative.");
            }
            this.persistentStorageBytes = persistentStorageBytes;
            return this;
        }

        /**
         * Sets the intermediate per-slot workspace size, excluding named output storage.
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

        /** Sets storage backing planner-managed arenas in each execution slot. */
        public Builder optArenaBytes(long arenaBytes) {
            this.arenaBytes = requireNonNegative(arenaBytes, "arena storage");
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
                    || storagePlannerVersion < 0
                    || logicalAllocationCount < 0
                    || backingAllocationCount < 0
                    || aliasViewCount < 0
                    || inPlaceReuseCount < 0) {
                throw new IllegalStateException(
                        "Fusion storage planner details must be specified explicitly.");
            }
            if (workspaceBytes > persistentStorageBytes) {
                throw new IllegalStateException(
                        "The workspace size must not exceed persistent storage.");
            }
            if (exportedOutputBytes > persistentStorageBytes
                    || workspaceBytes != persistentStorageBytes - exportedOutputBytes) {
                throw new IllegalStateException(
                        "Persistent storage must equal output storage plus workspace.");
            }
            if (arenaBytes > persistentStorageBytes) {
                throw new IllegalStateException(
                        "Arena storage must not exceed persistent storage.");
            }
            if (backingAllocationCount > logicalAllocationCount + aliasViewCount) {
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
