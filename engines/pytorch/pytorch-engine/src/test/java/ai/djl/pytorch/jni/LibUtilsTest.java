/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0
 */
package ai.djl.pytorch.jni;

import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
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
}
