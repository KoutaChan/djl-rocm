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
 * subset of per-slot persistent storage and must not be added to it. For a session with {@code n}
 * execution slots, the reported logical backend-owned payload is {@code executableStorageBytes + n
 * * persistentStorageBytes}; caller-owned inputs and constants are excluded.
 */
public final class FusionCompilationReport {

    private final String backend;
    private final int commandCount;
    private final long executableStorageBytes;
    private final long persistentStorageBytes;
    private final long workspaceBytes;
    private final boolean nativeOnly;

    private FusionCompilationReport(Builder builder) {
        backend = builder.backend;
        commandCount = builder.commandCount;
        executableStorageBytes = builder.executableStorageBytes;
        persistentStorageBytes = builder.persistentStorageBytes;
        workspaceBytes = builder.workspaceBytes;
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
     * Returns all persistent output and intermediate storage required by each execution slot.
     *
     * @return the persistent per-slot storage size in bytes
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
         * Sets all persistent output and intermediate storage used by each execution slot.
         *
         * @param persistentStorageBytes the persistent per-slot storage size in bytes
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
            if (workspaceBytes > persistentStorageBytes) {
                throw new IllegalStateException(
                        "The workspace size must not exceed persistent storage.");
            }
            return new FusionCompilationReport(this);
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
