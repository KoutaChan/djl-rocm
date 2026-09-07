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

import ai.djl.engine.EngineException;
import ai.djl.repository.Version;
import ai.djl.util.ClassLoaderUtils;
import ai.djl.util.Platform;
import ai.djl.util.Utils;
import ai.djl.util.cuda.CudaUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Utilities for finding the PyTorch Engine binary on the System.
 *
 * <p>The Engine will be searched for in a variety of locations in the following order:
 *
 * <ol>
 *   <li>In the path specified by the PYTORCH_LIBRARY_PATH environment variable
 *   <li>In a jar file location in the classpath. These jars can be created with the pytorch-native
 *       module.
 * </ol>
 */
@SuppressWarnings("MissingJavadocMethod")
public final class LibUtils {

    private static final Logger logger = LoggerFactory.getLogger(LibUtils.class);

    private static final String NATIVE_LIB_NAME = System.mapLibraryName("torch");
    private static final String JNI_LIB_NAME = System.mapLibraryName("djl_torch");

    private static final Pattern VERSION_PATTERN =
            Pattern.compile("(\\d+\\.\\d+\\.\\d+(-[a-z]+)?)(-SNAPSHOT)?(-\\d+)?");
    private static final Pattern LIB_PATTERN = Pattern.compile("(.*\\.(so(\\.\\d+)*|dll|dylib))");

    private static LibTorch libTorch;

    private LibUtils() {}

    public static synchronized void loadLibrary() {
        // TODO workaround to make it work on Android Studio
        // It should search for several places to find the native library
        if ("http://www.android.com/".equals(System.getProperty("java.vendor.url"))) {
            System.loadLibrary("djl_torch"); // NOPMD
            return;
        }
        libTorch = getLibTorch();
        loadLibTorch(libTorch);

        Path path = findJniLibrary(libTorch).toAbsolutePath();
        loadNativeLibrary(path.toString());
    }

    private static LibTorch getLibTorch() {
        LibTorch lib = findOverrideLibrary();
        if (lib != null) {
            return lib;
        }
        return findNativeLibrary();
    }

    public static String getVersion() {
        Matcher m = VERSION_PATTERN.matcher(libTorch.version);
        if (m.matches()) {
            return m.group(1);
        }
        return libTorch.version;
    }

    public static String getLibtorchPath() {
        return libTorch.dir.toString();
    }

    private static void loadLibTorch(LibTorch libTorch) {
        Path libDir = libTorch.dir.toAbsolutePath();
        if (Files.exists(libDir.resolve("libstdc++.so.6"))) {
            String libstd = Utils.getEnvOrSystemProperty("LIBSTDCXX_LIBRARY_PATH");
            if (libstd != null) {
                try {
                    logger.info("Loading libstdc++.so.6 from: {}", libstd);
                    System.load(libstd);
                } catch (UnsatisfiedLinkError e) {
                    logger.warn("Failed Loading libstdc++.so.6 from: {}", libstd);
                }
            }
        }
        String libExclusion = Utils.getEnvOrSystemProperty("PYTORCH_LIBRARY_EXCLUSION", "");
        Set<String> exclusion = new HashSet<>(Arrays.asList(libExclusion.split(",")));
        // flavor "cpu" contains "cu", so contains("cu") gives false positives
        boolean isCuda = libTorch.flavor.startsWith("cu");
        boolean isRocm = libTorch.flavor.startsWith("rocm");
        boolean isWindowsRocm = isRocm && libTorch.classifier.startsWith("win");
        if (libTorch.flavor.startsWith("rocm10.") && !isWindowsRocm) {
            preloadRocmLibraries(libTorch.flavor, libDir);
        }
        List<String> deferred =
                Arrays.asList(
                        System.mapLibraryName("fbgemm"),
                        System.mapLibraryName("caffe2_nvrtc"),
                        System.mapLibraryName("torch_cpu"),
                        System.mapLibraryName("c10_cuda"),
                        System.mapLibraryName("torch_cuda_cpp"),
                        System.mapLibraryName("torch_cuda_cu"),
                        System.mapLibraryName("torch_cuda"),
                        System.mapLibraryName("nvfuser_codegen"),
                        System.mapLibraryName("torch"));

        Set<String> loadLater = new HashSet<>(deferred);
        List<Path> windowsRocmLibraries = Collections.emptyList();
        if (isWindowsRocm) {
            windowsRocmLibraries = getWindowsRocmLoadOrder(libDir);
            for (Path path : windowsRocmLibraries) {
                loadLater.add(path.toFile().getName());
            }
        }
        try (Stream<Path> paths = Files.walk(libDir)) {
            Map<Path, Integer> rank = new ConcurrentHashMap<>();
            paths.filter(
                            path -> {
                                String name = path.getFileName().toString();
                                if (!LIB_PATTERN.matcher(name).matches()
                                        || exclusion.contains(name)) {
                                    return false;
                                } else if (!isCuda
                                        && name.contains("nvrtc")
                                        && name.contains("cudart")
                                        && name.contains("nvTools")) {
                                    return false;
                                } else if (name.startsWith("libarm_compute-")
                                        || name.startsWith("libopenblasp")) {
                                    rank.put(path, 2);
                                    return true;
                                } else if (name.startsWith("libarm_compute_")) {
                                    rank.put(path, 3);
                                    return true;
                                } else if (!isWindowsRocm
                                        && !loadLater.contains(name)
                                        && Files.isRegularFile(path)
                                        && !name.endsWith(JNI_LIB_NAME)
                                        && !name.contains("torch_")
                                        && !name.contains("caffe2_")
                                        && !name.startsWith("cudnn")) {
                                    rank.put(path, 1);
                                    return true;
                                }
                                return false;
                            })
                    .sorted(Comparator.comparingInt(rank::get))
                    .map(Path::toString)
                    .forEach(LibUtils::loadNativeLibrary);

            if (Files.exists((libDir.resolve("cudnn64_8.dll")))) {
                loadNativeLibrary(libDir.resolve("cudnn64_8.dll").toString());
                loadNativeLibrary(libDir.resolve("cudnn_ops_infer64_8.dll").toString());
                loadNativeLibrary(libDir.resolve("cudnn_ops_train64_8.dll").toString());
                loadNativeLibrary(libDir.resolve("cudnn_cnn_infer64_8.dll").toString());
                loadNativeLibrary(libDir.resolve("cudnn_cnn_train64_8.dll").toString());
                loadNativeLibrary(libDir.resolve("cudnn_adv_infer64_8.dll").toString());
                loadNativeLibrary(libDir.resolve("cudnn_adv_train64_8.dll").toString());
            } else if (Files.exists((libDir.resolve("cudnn64_7.dll")))) {
                loadNativeLibrary(libDir.resolve("cudnn64_7.dll").toString());
            }

            if (!windowsRocmLibraries.isEmpty()) {
                // Windows does not reliably resolve sibling ROCm DLL dependencies for an
                // absolute System.load(). Load the dependency graph explicitly.
                for (Path path : windowsRocmLibraries) {
                    loadNativeLibrary(path.toString());
                }
                deferred = Collections.emptyList();
            } else if (isRocm) {
                // ROCm libtorch ships the hipified libs as libtorch_hip.so / libc10_hip.so;
                // some releases keep the libtorch_cuda.so naming, so try both.
                deferred =
                        Arrays.asList(
                                System.mapLibraryName("fbgemm"),
                                System.mapLibraryName("torch_cpu"),
                                System.mapLibraryName("c10_hip"),
                                System.mapLibraryName("torch_hip"),
                                System.mapLibraryName("c10_cuda"),
                                System.mapLibraryName("torch_cuda"),
                                System.mapLibraryName("torch"));
            } else if (!isCuda) {
                deferred =
                        Arrays.asList(
                                System.mapLibraryName("fbgemm"),
                                System.mapLibraryName("torch_cpu"),
                                System.mapLibraryName("torch"));
            }

            for (String dep : deferred) {
                Path path = libDir.resolve(dep);
                if (Files.exists(path)) {
                    loadNativeLibrary(path.toString());
                }
            }
        } catch (IOException e) {
            throw new EngineException("Folder not exist! " + libDir, e);
        }
    }

    private static List<Path> getWindowsRocmLoadOrder(Path libDir) {
        List<Path> libraries = new ArrayList<>();
        addFirstMatching(libraries, libDir, "amd_comgr");
        addFirstMatching(libraries, libDir, "amdhip64");
        addFirstMatching(libraries, libDir, "hiprtc-builtins");
        addFirstMatching(libraries, libDir, "hiprtc", "hiprtc-builtins");
        addIfExists(libraries, libDir, System.mapLibraryName("caffe2_nvrtc"));
        addIfExists(libraries, libDir, System.mapLibraryName("c10"));
        addIfExists(libraries, libDir, System.mapLibraryName("hipblaslt"));
        addIfExists(libraries, libDir, System.mapLibraryName("rocblas"));
        addIfExists(libraries, libDir, System.mapLibraryName("hipblas"));
        addIfExists(libraries, libDir, System.mapLibraryName("rocfft"));
        addIfExists(libraries, libDir, System.mapLibraryName("hipfft"));
        addIfExists(libraries, libDir, System.mapLibraryName("rocrand"));
        addIfExists(libraries, libDir, System.mapLibraryName("hiprand"));
        addIfExists(libraries, libDir, System.mapLibraryName("rocsparse"));
        addIfExists(libraries, libDir, System.mapLibraryName("hipsparse"));
        addIfExists(libraries, libDir, System.mapLibraryName("rocsolver"));
        addIfExists(libraries, libDir, System.mapLibraryName("hipsolver"));
        addIfExists(libraries, libDir, System.mapLibraryName("MIOpen"));
        addIfExists(libraries, libDir, System.mapLibraryName("c10_hip"));
        addIfExists(libraries, libDir, System.mapLibraryName("fbgemm"));
        addIfExists(libraries, libDir, System.mapLibraryName("torch_cpu"));
        addIfExists(libraries, libDir, System.mapLibraryName("shm"));
        addIfExists(libraries, libDir, System.mapLibraryName("torch_hip"));
        addIfExists(libraries, libDir, System.mapLibraryName("torch"));
        return libraries;
    }

    private static void preloadRocmLibraries(String flavor, Path libDir) {
        File root = findRocmRoot(flavor);
        List<Path> directories = new ArrayList<>(4);
        directories.add(libDir);
        if (root != null) {
            directories.add(root.toPath().resolve("lib"));
            directories.add(root.toPath().resolve("lib/host-math/lib"));
            directories.add(root.toPath().resolve("lib/rocm_sysdeps/lib"));
        }
        List<String> libraries =
                Arrays.asList(
                        "amd_comgr",
                        "amdhip64",
                        "rocprofiler-sdk",
                        "rocprofiler-sdk-roctx",
                        "roctracer64",
                        "roctx64",
                        "hiprtc",
                        // rocBLAS reaches hipBLASLt through its SDK RPATH. Load the
                        // selected runtime's override before loading hipBLAS/rocBLAS.
                        "hipblaslt",
                        "hipblas",
                        "hipfft",
                        "hiprand",
                        "hipsparse",
                        "hipsparselt",
                        "hipsolver",
                        "rccl",
                        "MIOpen",
                        "hipdnn",
                        "rocm-openblas",
                        "rocm_smi64");
        for (String library : libraries) {
            Path path = findVersionedLibrary(directories, library);
            if (path != null) {
                loadNativeLibrary(path.toString());
            }
        }
    }

    private static Path findVersionedLibrary(List<Path> directories, String shortName) {
        String prefix = "lib" + shortName + ".so";
        for (Path directory : directories) {
            if (!Files.isDirectory(directory)) {
                continue;
            }
            try (Stream<Path> paths = Files.list(directory)) {
                Path match =
                        paths.filter(Files::isRegularFile)
                                .filter(path -> path.getFileName().toString().startsWith(prefix))
                                .sorted()
                                .findFirst()
                                .orElse(null);
                if (match != null) {
                    return match.toAbsolutePath();
                }
            } catch (IOException e) {
                logger.debug("Failed to inspect ROCm library directory: {}", directory, e);
            }
        }
        return null;
    }

    static String detectRocmFlavor() {
        String override = Utils.getEnvOrSystemProperty("DJL_ROCM_VERSION");
        if (override != null && !override.isEmpty()) {
            return "rocm" + override;
        }
        if (!System.getProperty("os.name", "").toLowerCase().startsWith("linux")) {
            return null;
        }
        File root = findRocmRoot(null);
        return root == null ? null : "rocm" + readRocmVersion(root);
    }

    private static void addIfExists(List<Path> libraries, Path libDir, String name) {
        Path path = libDir.resolve(name);
        if (Files.isRegularFile(path)) {
            libraries.add(path);
        }
    }

    private static void addFirstMatching(
            List<Path> libraries, Path libDir, String prefix, String... excludedPrefixes) {
        try (Stream<Path> paths = Files.list(libDir)) {
            paths.filter(Files::isRegularFile)
                    .filter(
                            path -> {
                                String name =
                                        path.getFileName().toString().toLowerCase(Locale.ROOT);
                                if (!name.endsWith(".dll")
                                        || !name.startsWith(prefix.toLowerCase(Locale.ROOT))) {
                                    return false;
                                }
                                for (String excluded : excludedPrefixes) {
                                    if (name.startsWith(excluded.toLowerCase(Locale.ROOT))) {
                                        return false;
                                    }
                                }
                                return true;
                            })
                    .sorted()
                    .findFirst()
                    .ifPresent(libraries::add);
        } catch (IOException e) {
            throw new EngineException("Failed to inspect ROCm libraries in: " + libDir, e);
        }
    }

    private static LibTorch findOverrideLibrary() {
        String libPath = Utils.getEnvOrSystemProperty("PYTORCH_LIBRARY_PATH");
        if (libPath != null) {
            return findLibraryInPath(libPath);
        }
        return null;
    }

    private static LibTorch findLibraryInPath(String libPath) {
        String[] paths = libPath.split(File.pathSeparator);
        for (String path : paths) {
            File p = new File(path);
            if (!p.exists()) {
                continue;
            }

            if (p.isFile() && NATIVE_LIB_NAME.equals(p.getName())) {
                return new LibTorch(p.getParentFile().toPath().toAbsolutePath());
            }

            File file = new File(path, NATIVE_LIB_NAME);
            if (file.exists() && file.isFile()) {
                return new LibTorch(p.toPath().toAbsolutePath());
            }
        }
        return null;
    }

    private static Path findJniLibrary(LibTorch libTorch) {
        String classifier = libTorch.classifier;
        String version = libTorch.version;
        String djlVersion = libTorch.apiVersion;
        String flavor = libTorch.flavor;

        String jniVersion = null;
        String jniCacheKey = djlVersion;
        try {
            URL url = ClassLoaderUtils.getResource("jnilib/pytorch.properties");
            if (url != null) {
                Properties prop = new Properties();
                try (InputStream is = Utils.openUrl(url)) {
                    prop.load(is);
                }
                jniVersion = prop.getProperty("jni_version");
                if (jniVersion == null) {
                    throw new AssertionError("No PyTorch jni version found.");
                }
                jniCacheKey = prop.getProperty("jni_cache_key", jniCacheKey);
            }
        } catch (IOException e) {
            throw new AssertionError("Failed to read PyTorch jni properties file.", e);
        }

        // Looking for JNI in libTorch.dir first
        Path libDir = libTorch.dir.toAbsolutePath();
        Path path = libDir.resolve(jniCacheKey + '-' + JNI_LIB_NAME);
        if (Files.exists(path)) {
            return path;
        }
        path = libDir.resolve(JNI_LIB_NAME);
        if (Files.exists(path)) {
            return path;
        }

        // always use cache dir, cache dir might be different from libTorch.dir
        Path cacheDir = Utils.getEngineCacheDir("pytorch");
        Path dir = cacheDir.resolve(version + '-' + flavor + '-' + classifier);
        path = dir.resolve(jniCacheKey + '-' + JNI_LIB_NAME);
        if (Files.exists(path)) {
            return path;
        }

        Matcher matcher = VERSION_PATTERN.matcher(version);
        if (!matcher.matches()) {
            throw new EngineException("Unexpected version: " + version);
        }
        version = matcher.group(1);

        if (jniVersion == null) {
            downloadJniLib(dir, path, djlVersion, version, classifier, flavor);
            return path;
        } else if (!jniVersion.startsWith(version + '-' + djlVersion)) {
            logger.warn("Found mismatch PyTorch jni: {}", jniVersion);
            downloadJniLib(dir, path, djlVersion, version, classifier, flavor);
            return path;
        }

        Path tmp = null;
        String libPath = "jnilib/" + classifier + '/' + flavor + '/' + JNI_LIB_NAME;
        logger.info("Extracting {} to cache ...", libPath);
        try (InputStream is = ClassLoaderUtils.getResourceAsStream(libPath)) {
            Files.createDirectories(dir);
            tmp = Files.createTempFile(dir, "jni", "tmp");
            Files.copy(is, tmp, StandardCopyOption.REPLACE_EXISTING);
            Utils.moveQuietly(tmp, path);
            return path;
        } catch (IOException e) {
            throw new EngineException("Cannot copy jni files", e);
        } finally {
            if (tmp != null) {
                Utils.deleteQuietly(tmp);
            }
        }
    }

    private static LibTorch findNativeLibrary() {
        Platform platform = Platform.detectPlatform("pytorch");
        String overrideVersion = Utils.getEnvOrSystemProperty("PYTORCH_VERSION");
        if (overrideVersion != null
                && !overrideVersion.isEmpty()
                && !platform.getVersion().startsWith(overrideVersion)) {
            // platform.version can be 1.8.1-20210421
            logger.warn("Override PyTorch version: {}.", overrideVersion);
            platform = Platform.detectPlatform("pytorch", overrideVersion);
            return downloadPyTorch(platform);
        }

        if (platform.isPlaceholder()) {
            return downloadPyTorch(platform);
        }

        return copyNativeLibraryFromClasspath(platform);
    }

    private static LibTorch copyNativeLibraryFromClasspath(Platform platform) {
        logger.debug("Found bundled PyTorch package: {}.", platform);
        String version = platform.getVersion();
        String flavor = platform.getFlavor();
        if (!flavor.endsWith("-precxx11")
                && Arrays.asList(platform.getLibraries()).contains("libstdc++.so.6")) {
            // for PyTorch 1.9.1 and older
            flavor += "-precxx11"; // NOPMD
        }
        String classifier = platform.getClassifier();

        Path tmp = null;
        try {
            Path cacheDir = Utils.getEngineCacheDir("pytorch");
            logger.debug("Using cache dir: {}", cacheDir);
            Path dir = cacheDir.resolve(version + '-' + flavor + '-' + classifier);
            Path path = dir.resolve(NATIVE_LIB_NAME);
            if (Files.exists(path)) {
                return new LibTorch(dir.toAbsolutePath(), platform, flavor);
            }
            Utils.deleteQuietly(dir);

            Matcher m = VERSION_PATTERN.matcher(version);
            if (!m.matches()) {
                throw new AssertionError("Unexpected version: " + version);
            }
            String pathPrefix = "pytorch/" + flavor + '/' + classifier;

            Files.createDirectories(cacheDir);
            tmp = Files.createTempDirectory(cacheDir, "tmp");
            for (String file : platform.getLibraries()) {
                String libPath = pathPrefix + '/' + file;
                logger.info("Extracting {} to cache ...", libPath);
                try (InputStream is = ClassLoaderUtils.getResourceAsStream(libPath)) {
                    Files.copy(is, tmp.resolve(file), StandardCopyOption.REPLACE_EXISTING);
                }
            }

            Utils.moveQuietly(tmp, dir);
            return new LibTorch(dir.toAbsolutePath(), platform, flavor);
        } catch (IOException e) {
            throw new EngineException("Failed to extract PyTorch native library", e);
        } finally {
            if (tmp != null) {
                Utils.deleteQuietly(tmp);
            }
        }
    }

    private static void loadNativeLibrary(String path) {
        logger.debug("Loading native library: {}", path);
        String nativeHelper = System.getProperty("ai.djl.pytorch.native_helper");
        if (nativeHelper != null && !nativeHelper.isEmpty()) {
            ClassLoaderUtils.nativeLoad(nativeHelper, path);
        } else {
            System.load(path); // NOPMD
        }
    }

    private static LibTorch downloadPyTorch(Platform platform) {
        String version = platform.getVersion();
        String classifier = platform.getClassifier();
        String flavor = Utils.getEnvOrSystemProperty("PYTORCH_FLAVOR");
        String precxx11;
        if (flavor == null || flavor.isEmpty()) {
            flavor = platform.getFlavor();
            if (System.getProperty("os.name").startsWith("Linux")
                    && (Boolean.parseBoolean(Utils.getEnvOrSystemProperty("PYTORCH_PRECXX11"))
                            || ("aarch64".equals(platform.getOsArch())
                                    && new Version(version).compareTo(new Version("2.7.1")) < 0))) {
                precxx11 = "-precxx11";
            } else {
                precxx11 = "";
            }
            flavor += precxx11;
        } else {
            logger.info("Uses override PYTORCH_FLAVOR: {}", flavor);
        }

        Path cacheDir = Utils.getEngineCacheDir("pytorch");
        Path dir = cacheDir.resolve(version + '-' + flavor + '-' + classifier);
        Path path = dir.resolve(NATIVE_LIB_NAME);
        if (Files.exists(path)) {
            logger.debug("Using cache dir: {}", dir);
            return new LibTorch(dir.toAbsolutePath(), platform, flavor);
        }

        // Fetch libtorch directly from download.pytorch.org. publish.djl.ai
        // only mirrors the handful of PyTorch versions that upstream DJL
        // releases have pinned (currently up to 2.7.1), and this fork
        // targets newer versions (2.11.0+), so the legacy files.txt path
        // would 404. The pytorch.org URL is well-defined per flavor and
        // classifier, so we build it inline and stream the zip.
        String bareFlavor =
                flavor.endsWith("-precxx11")
                        ? flavor.substring(0, flavor.length() - "-precxx11".length())
                        : flavor;
        String url = buildPytorchOrgUrl(version, bareFlavor, classifier);
        logger.info("Downloading {} ...", url);
        Path tmp = null;
        try {
            Files.createDirectories(cacheDir);
            tmp = Files.createTempDirectory(cacheDir, "libtorch");
            boolean found = false;
            try (ZipInputStream zis = new ZipInputStream(Utils.openUrl(url))) {
                ZipEntry entry;
                while ((entry = zis.getNextEntry()) != null) {
                    String name = entry.getName();
                    if (entry.isDirectory() || !name.startsWith("libtorch/lib/")) {
                        continue;
                    }
                    found = true;
                    String fileName = name.substring("libtorch/lib/".length());
                    Path target = tmp.resolve(fileName);
                    Path parent = target.getParent();
                    if (parent != null) {
                        Files.createDirectories(parent);
                    }
                    Files.copy(zis, target, StandardCopyOption.REPLACE_EXISTING);
                }
            }
            if (!found) {
                throw new EngineException("No libtorch/lib/ entries found in archive: " + url);
            }
            Utils.moveQuietly(tmp, dir);
            return new LibTorch(dir.toAbsolutePath(), platform, flavor);
        } catch (IOException e) {
            throw new EngineException("Failed to download libtorch: " + url, e);
        } finally {
            if (tmp != null) {
                Utils.deleteQuietly(tmp);
            }
        }
    }

    /**
     * Build the download.pytorch.org zip URL for a given PyTorch version + flavor + OS/arch
     * classifier. The fork targets modern PyTorch (2.11.0+) where the Linux filename dropped the
     * {@code -cxx11-abi-} infix and upstream macOS x86_64 has been EOL since 2.2.x, so this helper
     * only has to cover the live URL patterns.
     *
     * @param version PyTorch version (e.g. {@code 2.11.0})
     * @param bareFlavor flavor without the optional {@code -precxx11} suffix, e.g. {@code cpu} /
     *     {@code cu128} / {@code rocm7.1}
     * @param classifier OS+arch classifier, e.g. {@code linux-x86_64} / {@code win-x86_64} / {@code
     *     osx-aarch64}
     */
    private static String buildPytorchOrgUrl(String version, String bareFlavor, String classifier) {
        if (classifier.startsWith("osx")) {
            // macOS: only cpu, arm64 from 2.2+
            return "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-"
                    + version
                    + ".zip";
        }
        String zipPrefix =
                classifier.startsWith("win")
                        ? "libtorch-win-shared-with-deps-"
                        : "libtorch-shared-with-deps-";
        return "https://download.pytorch.org/libtorch/"
                + bareFlavor
                + '/'
                + zipPrefix
                + version
                + "%2B"
                + bareFlavor
                + ".zip";
    }

    private static void downloadJniLib(
            Path cacheDir,
            Path path,
            String djlVersion,
            String version,
            String classifier,
            String flavor) {
        String url =
                "https://publish.djl.ai/pytorch/"
                        + version
                        + "/jnilib/"
                        + djlVersion
                        + '/'
                        + classifier
                        + '/'
                        + flavor
                        + '/'
                        + JNI_LIB_NAME;
        logger.info("Downloading jni {} to cache ...", url);
        Path tmp = null;
        try (InputStream is = Utils.openUrl(url)) {
            Files.createDirectories(cacheDir);
            tmp = Files.createTempFile(cacheDir, "jni", "tmp");
            Files.copy(is, tmp, StandardCopyOption.REPLACE_EXISTING);
            Utils.moveQuietly(tmp, path);
        } catch (IOException e) {
            throw new EngineException("Cannot download jni files: " + url, e);
        } finally {
            if (tmp != null) {
                Utils.deleteQuietly(tmp);
            }
        }
    }

    private static final class LibTorch {

        Path dir;
        String version;
        String apiVersion;
        String flavor;
        String classifier;

        LibTorch(Path dir) {
            Platform platform = Platform.detectPlatform("pytorch");
            this.dir = dir;
            this.apiVersion = platform.getApiVersion();
            this.classifier = platform.getClassifier();
            version = Utils.getEnvOrSystemProperty("PYTORCH_VERSION");
            if (version == null || version.isEmpty()) {
                version = platform.getVersion();
            }
            flavor = Utils.getEnvOrSystemProperty("PYTORCH_FLAVOR");
            if (flavor == null || flavor.isEmpty()) {
                String rocmFlavor = LibUtils.detectRocmFlavor();
                if (CudaUtils.getGpuCount() > 0) {
                    flavor = "cu" + CudaUtils.getCudaVersionString() + "-precxx11";
                } else if (rocmFlavor != null) {
                    flavor = rocmFlavor;
                } else if ("linux".equals(platform.getOsPrefix())) {
                    flavor = "cpu-precxx11";
                } else {
                    flavor = "cpu";
                }
            }
        }

        LibTorch(Path dir, Platform platform, String flavor) {
            this.dir = dir;
            this.version = platform.getVersion();
            this.apiVersion = platform.getApiVersion();
            this.classifier = platform.getClassifier();
            this.flavor = flavor;
        }
    }

    private static File findRocmRoot(String flavor) {
        String configured = Utils.getEnvOrSystemProperty("ROCM_PATH");
        if (configured == null || configured.isEmpty()) {
            configured = Utils.getEnvOrSystemProperty("ROCM_HOME");
        }
        if (configured != null && !configured.isEmpty()) {
            File root = new File(configured);
            if (readRocmVersion(root) != null) {
                return root;
            }
        }

        String requested = flavor == null ? null : flavor.substring("rocm".length());
        File opt = new File("/opt");
        File rocm = new File(opt, "rocm");
        List<File> candidates = new ArrayList<>();
        candidates.add(rocm);
        File[] cores =
                rocm.listFiles(file -> file.isDirectory() && file.getName().startsWith("core-"));
        if (cores != null) {
            candidates.addAll(Arrays.asList(cores));
        }
        File[] siblings =
                opt.listFiles(file -> file.isDirectory() && file.getName().startsWith("rocm-"));
        if (siblings != null) {
            candidates.addAll(Arrays.asList(siblings));
        }
        return candidates.stream()
                .filter(file -> readRocmVersion(file) != null)
                .filter(file -> requested == null || readRocmVersion(file).startsWith(requested))
                .max(Comparator.comparing(file -> new Version(readRocmVersion(file))))
                .orElse(null);
    }

    static String readRocmVersion(File root) {
        File versionFile = new File(root, ".info/version");
        if (versionFile.isFile()) {
            try {
                String text =
                        new String(Files.readAllBytes(versionFile.toPath()), StandardCharsets.UTF_8)
                                .trim();
                Matcher matcher = Pattern.compile("(\\d+\\.\\d+)(?:\\.\\d+)?").matcher(text);
                if (matcher.find()) {
                    return matcher.group(1);
                }
            } catch (IOException ignored) {
                // Try the directory name below.
            }
        }
        Matcher matcher = Pattern.compile("(?:rocm-|core-)(\\d+\\.\\d+)").matcher(root.getName());
        return matcher.matches() ? matcher.group(1) : null;
    }
}
