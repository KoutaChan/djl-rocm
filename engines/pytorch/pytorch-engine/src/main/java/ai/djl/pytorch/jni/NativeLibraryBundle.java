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

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Properties;
import java.util.stream.Stream;

/** Extracts a versioned JNI and ROCm host-library bundle as one immutable cache entry. */
final class NativeLibraryBundle {

    private static final String MANIFEST = "native-bundle.properties";
    private static final String KERNEL_METADATA = "hipblasltKernelMetadata.";

    private final URL source;
    private final byte[] manifest;
    private final Properties properties;
    private final String hipblasltLibrary;

    NativeLibraryBundle(URL source) throws IOException {
        this.source = source;
        try (InputStream input = Utils.openUrl(source)) {
            manifest = Utils.toByteArray(input);
        }
        properties = new Properties();
        properties.load(new ByteArrayInputStream(manifest));
        hipblasltLibrary = required("hipblasltLibrary");
        require("rocmLoaderLibrary", RocmLibraryLoader.LIBRARY_NAME);
        if (relativePath(hipblasltLibrary).getNameCount() != 1) {
            throw new IOException("hipBLASLt must be a library file at the bundle root");
        }
    }

    void checkCompatibility(String pytorch, String djl, String flavor, String classifier)
            throws IOException {
        require("schemaVersion", "1");
        require("pytorchVersion", pytorch);
        require("djlVersion", djl);
        require("flavor", flavor);
        require("classifier", classifier);
        require(
                "libraries",
                "libdjl_torch.so," + hipblasltLibrary + ',' + RocmLibraryLoader.LIBRARY_NAME);
        require("hipblasltSoname", hipblasltLibrary);
    }

    String rocmVersion() throws IOException {
        return required("rocmVersion");
    }

    Path extract(Path cacheRoot, Path sdkRoot, Path libTorchRoot) throws IOException {
        Path sdk = sdkRoot.toRealPath();
        Path torch = libTorchRoot.toRealPath();
        verifyKernelMetadata(sdk);
        String contentKey = hash(manifest);
        // The host library locates kernels relative to itself. A distinct SDK link must never
        // be substituted underneath a bundle already in use by another JVM.
        String sdkKey = hash((sdk + "\n" + torch).getBytes(StandardCharsets.UTF_8));
        Path cache = cacheRoot.resolve("native-bundles");
        Files.createDirectories(cache);
        synchronized (NativeLibraryBundle.class) {
            return extractLocked(cache, contentKey + '-' + sdkKey, sdk, torch);
        }
    }

    private Path extractLocked(Path cache, String key, Path sdk, Path torch) throws IOException {
        // The Java monitor serializes threads; the file lock serializes JVMs.
        Path directory = cache.resolve(key);
        try (FileChannel channel =
                        FileChannel.open(
                                cache.resolve(key + ".lock"),
                                StandardOpenOption.CREATE,
                                StandardOpenOption.WRITE);
                FileLock lock = channel.lock()) {
            if (!lock.isValid()) {
                throw new IOException("Cannot lock native bundle cache: " + directory);
            }
            if (Files.isRegularFile(directory.resolve(MANIFEST))) {
                return directory;
            }
            Path temporary = Files.createTempDirectory(cache, "extract-");
            try {
                String files = required("files");
                for (String file : files.split(",")) {
                    Path target = temporary.resolve(relativePath(file));
                    Files.createDirectories(target.getParent());
                    MessageDigest digest = newDigest();
                    // Resolve siblings from this manifest's JAR, never from another JAR's
                    // resource of the same name on the classpath.
                    try (InputStream input =
                            new DigestInputStream(Utils.openUrl(new URL(source, file)), digest)) {
                        Files.copy(input, target);
                    }
                    if (!hex(digest.digest()).equals(required("sha256." + file))) {
                        throw new IOException("Native bundle checksum mismatch: " + file);
                    }
                }
                if (!Files.isRegularFile(temporary.resolve("libdjl_torch.so"))
                        || !Files.isRegularFile(temporary.resolve(hipblasltLibrary))
                        || !Files.isRegularFile(
                                temporary.resolve(RocmLibraryLoader.LIBRARY_NAME))) {
                    throw new IOException("Native bundle is missing a required native library");
                }
                Path kernels = temporary.resolve("hipblaslt/library");
                Files.createDirectories(kernels.getParent());
                Files.createSymbolicLink(kernels, sdk.resolve("lib/hipblaslt/library"));
                // Older LibTorch archives depend on unversioned library names. Their $ORIGIN
                // must resolve to this private runtime, not to their original hipBLASLt copy.
                linkRuntimeDirectory(temporary, sdk.resolve("lib"));
                linkRuntimeDirectory(temporary, sdk.resolve("lib64"));
                linkRuntimeDirectory(temporary, sdk.resolve("lib/host-math/lib"));
                linkRuntimeDirectory(temporary, sdk.resolve("lib/rocm_sysdeps/lib"));
                linkRuntimeDirectory(temporary, torch);
                Files.write(temporary.resolve(MANIFEST), manifest);
                // A crash cannot expose a partly extracted library as a usable cache entry.
                Files.move(temporary, directory, StandardCopyOption.ATOMIC_MOVE);
                return directory;
            } finally {
                Utils.deleteQuietly(temporary);
            }
        }
    }

    private void linkRuntimeDirectory(Path destination, Path sourceDirectory) throws IOException {
        if (!Files.isDirectory(sourceDirectory)) {
            return;
        }
        try (Stream<Path> paths = Files.list(sourceDirectory)) {
            for (Path entry : (Iterable<Path>) paths::iterator) {
                String name = entry.getFileName().toString();
                Path link = destination.resolve(name);
                if (!Files.exists(link, LinkOption.NOFOLLOW_LINKS)) {
                    Path target =
                            "libhipblaslt.so".equals(LibUtils.sharedLibraryName(name))
                                    ? Paths.get(hipblasltLibrary)
                                    : entry;
                    Files.createSymbolicLink(link, target);
                }
            }
        }
    }

    private void verifyKernelMetadata(Path sdk) throws IOException {
        String version = LibUtils.readRocmSdkVersion(sdk.toFile());
        if (version == null) {
            throw new IOException(
                    "Unrecognized ROCm SDK version in " + sdk.resolve(".info/version"));
        }
        require("rocmVersion", version);
        boolean found = false;
        for (String key : properties.stringPropertyNames()) {
            if (key.startsWith(KERNEL_METADATA)) {
                found = true;
                String name = key.substring(KERNEL_METADATA.length());
                Path path = sdk.resolve(relativePath(name));
                MessageDigest digest = newDigest();
                try (InputStream input = Files.newInputStream(path)) {
                    byte[] buffer = new byte[65536];
                    int read;
                    while ((read = input.read(buffer)) != -1) {
                        digest.update(buffer, 0, read);
                    }
                }
                if (!hex(digest.digest()).equals(properties.getProperty(key))) {
                    throw new IOException(
                            "Bundled hipBLASLt requires its matching ROCm "
                                    + required("rocmVersion")
                                    + " SDK kernel metadata: "
                                    + path);
                }
            }
        }
        if (!found) {
            throw new IOException("Native bundle has no hipBLASLt kernel compatibility metadata");
        }
    }

    private void require(String key, String expected) throws IOException {
        String actual = required(key);
        if (!expected.equals(actual)) {
            throw new IOException(
                    "Incompatible native bundle "
                            + key
                            + ": expected "
                            + expected
                            + ", got "
                            + actual);
        }
    }

    private String required(String key) throws IOException {
        String value = properties.getProperty(key);
        if (value == null || value.isEmpty()) {
            throw new IOException("Missing native bundle property: " + key);
        }
        return value;
    }

    private static Path relativePath(String name) throws IOException {
        Path path = Paths.get(name);
        if (path.isAbsolute()
                || name.isEmpty()
                || name.indexOf('\\') >= 0
                || !path.normalize().equals(path)
                || path.startsWith("..")) {
            throw new IOException("Invalid native bundle path: " + name);
        }
        return path;
    }

    private static String hash(byte[] bytes) {
        return hex(newDigest().digest(bytes));
    }

    private static MessageDigest newDigest() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }

    private static String hex(byte[] bytes) {
        StringBuilder result = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) {
            result.append(Character.forDigit((value & 0xff) >>> 4, 16));
            result.append(Character.forDigit(value & 0xf, 16));
        }
        return result.toString();
    }
}
