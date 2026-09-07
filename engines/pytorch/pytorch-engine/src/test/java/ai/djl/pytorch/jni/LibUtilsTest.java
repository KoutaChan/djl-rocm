/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0
 */
package ai.djl.pytorch.jni;

import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;

public class LibUtilsTest {

    @AfterMethod
    public void clearOverrides() {
        System.clearProperty("DJL_ROCM_VERSION");
    }

    @Test
    public void testRocmVersionOverride() {
        System.setProperty("DJL_ROCM_VERSION", "10.0");

        Assert.assertEquals(LibUtils.detectRocmFlavor(), "rocm10.0");
    }

    @Test
    public void testRocmTenLayouts() throws IOException {
        Path root = Files.createTempDirectory("rocm-core-10.0");
        try {
            Path info = Files.createDirectories(root.resolve(".info"));
            Files.writeString(info.resolve("version"), "10.0.0\n");
            Assert.assertEquals(LibUtils.readRocmVersion(root.toFile()), "10.0");

            Assert.assertEquals(LibUtils.readRocmVersion(Path.of("core-10.0").toFile()), "10.0");
        } finally {
            try (Stream<Path> paths = Files.walk(root)) {
                paths.sorted((left, right) -> right.compareTo(left))
                        .forEach(
                                path -> {
                                    try {
                                        Files.delete(path);
                                    } catch (IOException ignored) {
                                        // Best effort cleanup of the temporary test tree.
                                    }
                                });
            }
        }
    }

    @Test
    public void testExactSdkSelectionWithinFlavor() throws IOException {
        Path root = Files.createTempDirectory("rocm-sdk-selection");
        try {
            File newer = root.resolve("rocm-7.2.2").toFile();
            File matching = root.resolve("rocm-7.2.0").toFile();
            Files.createDirectories(newer.toPath().resolve(".info"));
            Files.createDirectories(matching.toPath().resolve(".info"));
            Files.writeString(newer.toPath().resolve(".info/version"), "7.2.2-build\n");
            Files.writeString(matching.toPath().resolve(".info/version"), "7.2.0-build\n");
            List<File> candidates = List.of(newer, matching);
            Assert.assertEquals(LibUtils.readRocmSdkVersion(newer), "7.2.2");
            Assert.assertEquals(LibUtils.readRocmVersion(newer), "7.2");
            Assert.assertEquals(LibUtils.selectRocmRoot(candidates, "rocm7.2", "7.2.0"), matching);
            Assert.assertEquals(LibUtils.selectRocmRoot(candidates, "rocm7.2", "7.2.2"), newer);
            Assert.assertNull(LibUtils.selectRocmRoot(candidates, "rocm7.2", "7.2.1"));
            Assert.assertNull(LibUtils.selectRocmRoot(candidates, "rocm7.1", "7.2.0"));
        } finally {
            Utils.deleteQuietly(root);
        }
    }
}
