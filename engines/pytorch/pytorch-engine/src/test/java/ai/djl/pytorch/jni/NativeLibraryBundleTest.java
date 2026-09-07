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
import org.testng.SkipException;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.io.IOException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;

public class NativeLibraryBundleTest {

    private static final String JNI = "libdjl_torch.so";
    private static final String HIPBLASLT = "libhipblaslt.so.1";
    private static final String METADATA =
            "lib/hipblaslt/library/gfx1100/TensileLibrary_lazy_gfx1100.dat.zlib";

    private Path root;
    private Path sdk;
    private Path torch;
    private Path cache;
    private Map<String, String> files;
    private Map<String, String> properties;
    private int jarId;

    @BeforeMethod
    public void setUp() throws IOException {
        root = Files.createTempDirectory("native-bundle-test");
        sdk = root.resolve("sdk");
        torch = Files.createDirectories(root.resolve("torch"));
        cache = root.resolve("cache");
        Files.createDirectories(sdk.resolve(".info"));
        Files.writeString(sdk.resolve(".info/version"), "10.0.0\n");
        Files.createDirectories(sdk.resolve(METADATA).getParent());
        Files.writeString(sdk.resolve(METADATA), "mapping");
        files = new LinkedHashMap<>();
        files.put(JNI, "jni binary");
        files.put(HIPBLASLT, "patched hipblaslt binary");
        files.put(RocmLibraryLoader.LIBRARY_NAME, "runtime loader binary");
        files.put("licenses/hipblaslt-LICENSE.md", "test license");
        properties = new LinkedHashMap<>();
        properties.put("schemaVersion", "1");
        properties.put("pytorchVersion", "2.11.0");
        properties.put("djlVersion", "0.40.1-rocm");
        properties.put("flavor", "rocm10.0");
        properties.put("classifier", "linux-x86_64");
        properties.put("libraries", JNI + ',' + HIPBLASLT + ',' + RocmLibraryLoader.LIBRARY_NAME);
        properties.put("rocmLoaderLibrary", RocmLibraryLoader.LIBRARY_NAME);
        properties.put("hipblasltLibrary", HIPBLASLT);
        properties.put("hipblasltSoname", "libhipblaslt.so.1");
        properties.put("rocmVersion", "10.0.0");
        properties.put("hipblasltKernelMetadata." + METADATA, hash("mapping"));
    }

    @AfterMethod
    public void tearDown() {
        Utils.deleteQuietly(root);
    }

    @Test
    public void testBundleCompatibility() throws IOException {
        NativeLibraryBundle bundle = bundle();
        bundle.checkCompatibility("2.11.0", "0.40.1-rocm", "rocm10.0", "linux-x86_64");
        Assert.expectThrows(
                IOException.class,
                () ->
                        bundle.checkCompatibility(
                                "2.10.0", "0.40.1-rocm", "rocm10.0", "linux-x86_64"));
        Assert.expectThrows(
                IOException.class,
                () ->
                        bundle.checkCompatibility(
                                "2.11.0", "0.40.2-rocm", "rocm10.0", "linux-x86_64"));
    }

    @Test
    public void testWholeBundleAndKernelLink() throws IOException {
        requireSymbolicLinks();
        NativeLibraryBundle bundle = bundle();
        Path extracted = bundle.extract(cache, sdk, torch);
        Assert.assertEquals(Files.readString(extracted.resolve(JNI)), files.get(JNI));
        Assert.assertEquals(Files.readString(extracted.resolve(HIPBLASLT)), files.get(HIPBLASLT));
        Assert.assertTrue(Files.isRegularFile(extracted.resolve("licenses/hipblaslt-LICENSE.md")));
        Assert.assertEquals(
                extracted.resolve("hipblaslt/library").toRealPath(),
                sdk.resolve("lib/hipblaslt/library").toRealPath());
        Assert.assertEquals(bundle.extract(cache, sdk, torch), extracted);
        // Cleanup must delete the cache link, never the SDK that it points to.
        Utils.deleteQuietly(extracted);
        Assert.assertEquals(Files.readString(sdk.resolve(METADATA)), "mapping");
    }

    @DataProvider
    public Object[][] supportedSdks() {
        return new Object[][] {
            {"rocm6.4", "6.4.0", "2.9.1", "libhipblaslt.so.0"},
            {"rocm7.0", "7.0.0", "2.10.0", "libhipblaslt.so.1"},
            {"rocm7.1", "7.1.0", "2.11.0", "libhipblaslt.so.1"},
            {"rocm7.2", "7.2.0", "2.11.0", "libhipblaslt.so.1"},
            {"rocm7.2", "7.2.1", "2.11.0", "libhipblaslt.so.1"},
            {"rocm7.2", "7.2.2", "2.11.0", "libhipblaslt.so.1"},
            {"rocm10.0", "10.0.0", "2.11.0", "libhipblaslt.so.1"}
        };
    }

    @Test(dataProvider = "supportedSdks")
    public void testSdkSpecificRuntime(
            String flavor, String sdkVersion, String pytorchVersion, String soname)
            throws IOException {
        requireSymbolicLinks();
        Files.writeString(sdk.resolve(".info/version"), sdkVersion + "-test-build\n");
        String metadata =
                flavor.equals("rocm10.0")
                        ? METADATA
                        : "lib/hipblaslt/library/TensileLibrary_lazy_gfx1100.dat";
        Files.createDirectories(sdk.resolve(metadata).getParent());
        Files.writeString(sdk.resolve(metadata), "mapping");
        files.remove(HIPBLASLT);
        files.put(soname, "patched hipblaslt binary");
        properties.put("pytorchVersion", pytorchVersion);
        properties.put("flavor", flavor);
        properties.put("rocmVersion", sdkVersion);
        properties.put("libraries", JNI + ',' + soname + ',' + RocmLibraryLoader.LIBRARY_NAME);
        properties.put("hipblasltLibrary", soname);
        properties.put("hipblasltSoname", soname);
        properties.remove("hipblasltKernelMetadata." + METADATA);
        properties.put("hipblasltKernelMetadata." + metadata, hash("mapping"));
        Files.writeString(torch.resolve("libhipblaslt.so"), "original Torch hipblaslt");
        Files.writeString(torch.resolve("libtorch_hip.so"), "Torch library");
        Files.writeString(torch.resolve("libamdhip64.so"), "Torch HIP runtime");
        Files.writeString(sdk.resolve("lib/libamdhip64.so"), "matching SDK HIP runtime");

        NativeLibraryBundle bundle = bundle();
        bundle.checkCompatibility(pytorchVersion, "0.40.1-rocm", flavor, "linux-x86_64");
        Path runtime = bundle.extract(cache, sdk, torch);
        Assert.assertEquals(
                Files.readString(runtime.resolve("libhipblaslt.so")), "patched hipblaslt binary");
        Assert.assertEquals(
                runtime.resolve("libhipblaslt.so").toRealPath(), runtime.resolve(soname));
        Assert.assertEquals(
                runtime.resolve("libtorch_hip.so").toRealPath(), torch.resolve("libtorch_hip.so"));
        Assert.assertEquals(
                Files.readString(runtime.resolve("libamdhip64.so")), "matching SDK HIP runtime");
        Assert.assertEquals(
                Files.readString(torch.resolve("libhipblaslt.so")), "original Torch hipblaslt");
    }

    @Test
    public void testCacheSeparatesLibTorchInstallations() throws IOException {
        requireSymbolicLinks();
        NativeLibraryBundle bundle = bundle();
        Path first = bundle.extract(cache, sdk, torch);
        Path otherTorch = Files.createDirectories(root.resolve("other-torch"));
        Assert.assertNotEquals(bundle.extract(cache, sdk, otherTorch), first);
    }

    @Test
    public void testCacheSeparatesLibraryContentsAndSdkLocations() throws IOException {
        requireSymbolicLinks();
        Path first = bundle().extract(cache, sdk, torch);
        files.put(HIPBLASLT, "next patched hipblaslt binary");
        properties.remove("sha256." + HIPBLASLT);
        NativeLibraryBundle next = bundle();
        Path changed = next.extract(cache, sdk, torch);
        Assert.assertNotEquals(changed, first);
        Path otherSdk = root.resolve("other-sdk");
        Files.createDirectories(otherSdk.resolve(".info"));
        Files.copy(sdk.resolve(".info/version"), otherSdk.resolve(".info/version"));
        Files.createDirectories(otherSdk.resolve(METADATA).getParent());
        Files.copy(sdk.resolve(METADATA), otherSdk.resolve(METADATA));
        Assert.assertNotEquals(next.extract(cache, otherSdk, torch), changed);
        Assert.assertEquals(Files.readString(first.resolve(HIPBLASLT)), "patched hipblaslt binary");
    }

    @Test
    public void testConcurrentExtraction() throws Exception {
        requireSymbolicLinks();
        NativeLibraryBundle first = bundle();
        NativeLibraryBundle second = bundle();
        ExecutorService executor = Executors.newFixedThreadPool(2);
        try {
            Future<Path> left = executor.submit(() -> first.extract(cache, sdk, torch));
            Future<Path> right = executor.submit(() -> second.extract(cache, sdk, torch));
            Assert.assertEquals(left.get(), right.get());
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    public void testRejectsCorruptPayloadBeforePublishingCache() throws IOException {
        properties.put("sha256." + HIPBLASLT, hash("wrong binary"));
        NativeLibraryBundle bundle = bundle();
        IOException failure =
                Assert.expectThrows(IOException.class, () -> bundle.extract(cache, sdk, torch));
        Assert.assertTrue(failure.getMessage().contains("checksum mismatch"));
        try (java.util.stream.Stream<Path> paths = Files.walk(cache)) {
            Assert.assertFalse(paths.anyMatch(path -> path.getFileName().toString().equals(JNI)));
        }
    }

    @Test
    public void testRejectsIncompatibleKernelMetadata() throws IOException {
        NativeLibraryBundle bundle = bundle();
        Files.writeString(sdk.resolve(METADATA), "different SDK mapping");
        IOException failure =
                Assert.expectThrows(IOException.class, () -> bundle.extract(cache, sdk, torch));
        Assert.assertTrue(failure.getMessage().contains("matching ROCm"));
        Assert.assertFalse(Files.exists(cache));
    }

    @Test
    public void testRejectsDifferentSdkVersionWithIdenticalKernels() throws IOException {
        NativeLibraryBundle bundle = bundle();
        Files.writeString(sdk.resolve(".info/version"), "10.1.0\n");
        IOException failure =
                Assert.expectThrows(IOException.class, () -> bundle.extract(cache, sdk, torch));
        Assert.assertTrue(failure.getMessage().contains("rocmVersion"));
        Assert.assertFalse(Files.exists(cache));
    }

    @Test
    public void testRejectsTraversal() throws IOException {
        files.put("../outside", "escape");
        NativeLibraryBundle bundle = bundle();
        IOException failure =
                Assert.expectThrows(IOException.class, () -> bundle.extract(cache, sdk, torch));
        Assert.assertTrue(failure.getMessage().contains("Invalid native bundle path"));
        Assert.assertFalse(Files.exists(cache.resolve("native-bundles/outside")));
    }

    @Test
    public void testPreloadUsesBundleBeforeBlasAndTorchDependencies() throws IOException {
        Path libtorch = torch;
        Path bundle = Files.createDirectories(root.resolve("bundle"));
        Path hip = Files.createFile(sdk.resolve("lib/libamdhip64.so.7"));
        Files.createFile(libtorch.resolve("libamdhip64.so.7"));
        Path zlib = sdk.resolve("lib/rocm_sysdeps/lib/librocm_sysdeps_z.so.1");
        Files.createDirectories(zlib.getParent());
        Files.createFile(zlib);
        Path origami = Files.createFile(sdk.resolve("lib/liborigami.so.1"));
        Path patched = Files.createFile(bundle.resolve(HIPBLASLT));
        Files.createFile(sdk.resolve("lib/" + HIPBLASLT));
        Files.createFile(libtorch.resolve(HIPBLASLT));
        Path blas = Files.createFile(sdk.resolve("lib/libhipblas.so.3"));
        Path rocblas = Files.createFile(sdk.resolve("lib/librocblas.so.4"));
        List<Path> order = LibUtils.rocmPreloadLibraries(libtorch, sdk.toFile(), bundle);
        Assert.assertEquals(
                order, java.util.Arrays.asList(hip, zlib, origami, patched, rocblas, blas));
    }

    private NativeLibraryBundle bundle() throws IOException {
        properties.put("files", String.join(",", files.keySet()));
        for (Map.Entry<String, String> file : files.entrySet()) {
            properties.putIfAbsent("sha256." + file.getKey(), hash(file.getValue()));
        }
        // Each revision gets a separate real JAR. Relative URL resolution must stay in that JAR.
        Path jar = root.resolve("bundle-" + jarId++ + ".jar");
        String prefix = "jnilib/linux-x86_64/" + properties.get("flavor") + '/';
        try (JarOutputStream output = new JarOutputStream(Files.newOutputStream(jar))) {
            for (Map.Entry<String, String> file : files.entrySet()) {
                writeEntry(output, prefix + file.getKey(), file.getValue());
            }
            StringBuilder manifest = new StringBuilder();
            properties.forEach(
                    (key, value) -> manifest.append(key).append('=').append(value).append('\n'));
            writeEntry(output, prefix + "native-bundle.properties", manifest.toString());
        }
        return new NativeLibraryBundle(
                new URL("jar:" + jar.toUri() + "!/" + prefix + "native-bundle.properties"));
    }

    private void requireSymbolicLinks() throws IOException {
        Path link = root.resolve("test-link");
        try {
            Files.createSymbolicLink(link, sdk);
            Files.delete(link);
        } catch (IOException | UnsupportedOperationException e) {
            throw new SkipException("ROCm bundle extraction requires Linux symbolic links", e);
        }
    }

    private static void writeEntry(JarOutputStream output, String name, String text)
            throws IOException {
        output.putNextEntry(new JarEntry(name));
        output.write(text.getBytes(StandardCharsets.UTF_8));
        output.closeEntry();
    }

    private static String hash(String value) {
        try {
            byte[] digest =
                    MessageDigest.getInstance("SHA-256")
                            .digest(value.getBytes(StandardCharsets.UTF_8));
            StringBuilder text = new StringBuilder();
            for (byte item : digest) {
                text.append(String.format("%02x", item));
            }
            return text.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }
}
