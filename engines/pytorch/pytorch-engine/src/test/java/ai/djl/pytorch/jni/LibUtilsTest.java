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
package ai.djl.pytorch.jni;

import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public class LibUtilsTest {

    private String rocmVersion;

    @BeforeMethod
    public void setUp() {
        rocmVersion = System.getProperty("DJL_ROCM_VERSION");
    }

    @AfterMethod
    public void tearDown() {
        if (rocmVersion == null) {
            System.clearProperty("DJL_ROCM_VERSION");
        } else {
            System.setProperty("DJL_ROCM_VERSION", rocmVersion);
        }
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
            Utils.deleteQuietly(root);
        }
    }

    @Test
    public void testSdkSelection() throws IOException {
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

    @Test
    public void testRocmLibraryNames() throws IOException {
        Path root = Files.createTempDirectory("rocm-libraries");
        try {
            Files.createFile(root.resolve("libamdhip64.so.backup"));
            Files.createFile(root.resolve("libamdhip64.something"));
            Path library = Files.createFile(root.resolve("libhipblaslt.so.1"));
            Assert.assertEquals(LibUtils.getRocmLoadOrder(root, null, null), List.of(library));
        } finally {
            Utils.deleteQuietly(root);
        }
    }
}
