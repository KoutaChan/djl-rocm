import ai.djl.pytorch.PrepareHipblaslt
import groovy.json.JsonSlurper
import java.security.MessageDigest
import java.util.Properties

plugins {
    ai.djl.javaProject
    ai.djl.publish
}

val ptVersion: String = when {
    project.hasProperty("pt_version") && project.property("pt_version") != "" ->
        project.property("pt_version").toString()

    else -> libs.versions.pytorch.get()
}

// Accept CPU, CUDA and ROCm flavors, retaining rocm_flavor as a legacy fallback.
val flavor: String = when {
    project.hasProperty("flavor") && project.property("flavor") != "" ->
        project.property("flavor").toString()

    project.hasProperty("rocm_flavor") && project.property("rocm_flavor") != "" ->
        project.property("rocm_flavor").toString()

    else -> "rocm6.4"
}

// OS+arch classifier matching DJL's native-jar conventions.
val classifier: String = when {
    project.hasProperty("classifier") && project.property("classifier") != "" ->
        project.property("classifier").toString()

    else -> "linux-x86_64"
}

// pytorch-native's build output file name depends on the target OS.
val jniLibFileName: String = when {
    classifier.startsWith("win") -> "djl_torch.dll"
    classifier.startsWith("osx") -> "libdjl_torch.dylib"
    else -> "libdjl_torch.so"
}

// Include the platform in the artifact ID to avoid GitHub Packages corrupting
// classifiers with dashes or digits (github.com/orgs/community/discussions/49682).
// The JAR retains DJL's jnilib/<classifier>/<flavor>/<library> layout.
val artifactSlug = "pytorch-jni-$flavor-$classifier"

group = "ai.djl.pytorch"
val isRelease = project.hasProperty("release") || project.hasProperty("staging")
version = ptVersion + '-' + libs.versions.djl.get() + if (isRelease) "" else "-SNAPSHOT"

val nativeDir = project.parent!!.projectDir / "pytorch-native"
val hipblasltProfiles = JsonSlurper().parse(nativeDir / "rocm/hipblaslt/profiles.json") as Map<*, *>
val bundleHipblaslt = flavor in hipblasltProfiles && classifier == "linux-x86_64"
val hipblasltBundleDir = providers.gradleProperty("hipblaslt_bundle_dir").orNull
    ?.let { file(it) } ?: nativeDir / "build/rocm-runtime"
val explicitJniLibrary = providers.gradleProperty("jni_library").orNull?.let { file(it) }
val rocmLoaderName = "libdjl_rocm_loader.so"
val rocmLoaderLibrary = providers.gradleProperty("rocm_loader_library").orNull
    ?.let { file(it) } ?: nativeDir / "build/rocm-loader/$rocmLoaderName"

val prepareRocmLoader = tasks.register("prepareRocmLoader") {
    val enabled = bundleHipblaslt
    val suppliedLibrary = providers.gradleProperty("rocm_loader_library").isPresent
    val sourceDir = nativeDir / "rocm/runtime-loader"
    val outputFile = rocmLoaderLibrary
    val injected = project.objects.newInstance<InjectedOps>()

    onlyIf { enabled && !suppliedLibrary }
    inputs.dir(sourceDir)
    outputs.file(outputFile)
    // The bootstrap is tiny. Rebuild against the active compiler/JDK instead
    // of sharing a previous SDK image's libc baseline through a stale output.
    outputs.upToDateWhen { false }
    doLast {
        injected.exec.exec {
            commandLine("bash", (sourceDir / "build.sh").absolutePath, outputFile.absolutePath)
        }
    }
}

val prepareHipblaslt = tasks.register<PrepareHipblaslt>("prepareHipblaslt") {
    val enabled = bundleHipblaslt
    val suppliedBundle = providers.gradleProperty("hipblaslt_bundle_dir").isPresent

    sourceDir.set(nativeDir / "rocm/hipblaslt")
    outputDir.set(hipblasltBundleDir)
    flavorName.set(flavor)
    onlyIf { enabled && !suppliedBundle }
}

val stageJniLib = tasks.register("stageJniLib") {
    val djlVersion = libs.versions.djl.get()
    val logger = project.logger
    val nativeRoot = nativeDir
    val flavorName = flavor
    val classifierName = classifier
    val libName = jniLibFileName
    val publishedVersion = project.version
    val pytorchVersion = ptVersion
    val withHipblaslt = bundleHipblaslt
    val vendorProfile = hipblasltProfiles[flavorName] as Map<*, *>?
    val vendorDir = hipblasltBundleDir
    val suppliedJni = explicitJniLibrary
    val loaderName = rocmLoaderName
    val loaderLibrary = rocmLoaderLibrary
    val stageDir = layout.buildDirectory.dir("jnilib-stage")
    val injected = project.objects.newInstance<InjectedOps>()

    dependsOn(prepareHipblaslt, prepareRocmLoader)
    outputs.dir(stageDir)
    outputs.upToDateWhen { false }

    doLast {
        fun sha256(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().buffered().use { input ->
                val buffer = ByteArray(64 * 1024)
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    digest.update(buffer, 0, read)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }

        fun loadProperties(file: File): Properties = Properties().apply {
            file.inputStream().use { load(it) }
        }

        // This task is also used to package multiple flavors in one checkout.
        // Clear the staging root, not just the current classifier's directory.
        injected.fs.delete { delete(stageDir) }
        val entry = "$classifierName/$flavorName/$libName"
        val targetFile = stageDir.get().asFile / entry
        targetFile.parentFile.mkdirs()

        // Prefer a freshly built library, fall back to the jnilib tree
        // populated by pytorch-native:compileJNI (CI mode).
        val freshFile = nativeRoot / "build/$libName"
        val cachedFile = nativeRoot / "jnilib/${djlVersion}/$entry"
        val source = when {
            suppliedJni != null -> suppliedJni
            freshFile.exists() -> freshFile
            cachedFile.exists() -> cachedFile
            else -> throw GradleException(
                "$libName not found. Build it first:\n" +
                        "  cd $nativeRoot && ./build.sh <pt_version> $flavorName <cxx11|precxx11> <arch>"
            )
        }
        if (suppliedJni == null && source == freshFile) {
            val identityFile = nativeRoot / "build/native-build.properties"
            if (withHipblaslt || identityFile.exists()) {
                val identity = loadProperties(identityFile)
                val expected = mapOf(
                    "pt_version" to pytorchVersion,
                    "flavor" to flavorName,
                    "classifier" to classifierName,
                    "sha256" to sha256(source)
                )
                for ((key, value) in expected) {
                    check(identity.getProperty(key) == value) {
                        "JNI build identity mismatch for $key. Rebuild $flavorName/$classifierName before packaging."
                    }
                }
            }
        }
        logger.lifecycle("Bundling $source into jnilib/$entry")
        injected.fs.copy {
            from(source)
            into(targetFile.parent)
            rename { libName }
        }

        val contentHash = if (withHipblaslt) {
            val vendor = loadProperties(vendorDir / "hipblaslt.properties")
            for ((key, value) in mapOf("flavor" to flavorName, "classifier" to classifierName)) {
                check(vendor.getProperty(key) == value) {
                    "hipBLASLt bundle $key mismatch: expected $value in $vendorDir/hipblaslt.properties"
                }
            }
            val rocmVersion = vendor.getProperty("rocm_version")
            val sdkProfiles = vendorProfile!!["sdk_versions"] as Map<*, *>
            val sdkProfile = sdkProfiles[rocmVersion] as Map<*, *>?
            check(sdkProfile != null) {
                "hipBLASLt SDK version $rocmVersion is not supported for $flavorName"
            }
            for (key in listOf("source_repository", "source_revision", "hipblaslt_version", "soname")) {
                check(vendor.getProperty(key) == sdkProfile[key]) {
                    "hipBLASLt bundle $key does not match the $flavorName / $rocmVersion SDK profile"
                }
            }
            val vendorName = requireNotNull(vendor.getProperty("library")) {
                "Missing library in $vendorDir/hipblaslt.properties"
            }
            val vendorLibrary = vendorDir / vendorName
            check(
                vendorName.matches(Regex("libhipblaslt\\.so\\.[0-9]+(?:\\.[0-9]+)*")) &&
                        vendor.getProperty("soname") == vendorName
            ) {
                "Unexpected hipBLASLt library name in $vendorDir"
            }
            check(vendor.getProperty("sha256") == sha256(vendorLibrary)) {
                "hipBLASLt library does not match $vendorDir/hipblaslt.properties"
            }
            check(loaderLibrary.isFile) { "ROCm runtime loader is missing: $loaderLibrary" }
            injected.fs.copy {
                from(vendorLibrary)
                from(loaderLibrary) { rename { loaderName } }
                from(vendorDir / "licenses") { into("licenses") }
                into(targetFile.parent)
            }
            val bundleRoot = targetFile.parentFile
            val files = bundleRoot.walkTopDown()
                .filter { it.isFile }
                .map { it.relativeTo(bundleRoot).invariantSeparatorsPath }
                .sorted()
                .toList()
            check(files.any { it.startsWith("licenses/") }) {
                "hipBLASLt redistribution notices are missing in $vendorDir/licenses"
            }
            val metadata = sortedMapOf(
                "schemaVersion" to "1",
                "djlVersion" to djlVersion,
                "pytorchVersion" to pytorchVersion,
                "flavor" to flavorName,
                "classifier" to classifierName,
                "libraries" to "$libName,$vendorName,$loaderName",
                "rocmLoaderLibrary" to loaderName,
                "files" to files.joinToString(",")
            )
            val vendorFields = mapOf(
                "source_repository" to "hipblasltSourceRepository",
                "source_revision" to "hipblasltRevision",
                "msgpack_revision" to "msgpackRevision",
                "patch_sha256" to "hipblasltPatchSha256",
                "hipblaslt_version" to "hipblasltVersion",
                "library" to "hipblasltLibrary",
                "soname" to "hipblasltSoname",
                "rocm_version" to "rocmVersion",
                "sdk_library_sha256" to "hipblasltSdkLibrarySha256",
                "kernel_metadata_sha256" to "hipblasltKernelMetadataSha256",
                "gpu_targets" to "hipblasltGpuTargets"
            )
            for ((key, manifestKey) in vendorFields) {
                metadata[manifestKey] = requireNotNull(vendor.getProperty(key)) {
                    "Missing $key in $vendorDir/hipblaslt.properties"
                }
            }
            val kernelMetadata = vendor.stringPropertyNames().filter { it.startsWith("kernel_metadata.") }
            check(kernelMetadata.isNotEmpty()) {
                "SDK kernel metadata is missing in $vendorDir/hipblaslt.properties"
            }
            for (key in kernelMetadata) {
                metadata["hipblasltKernelMetadata." + key.removePrefix("kernel_metadata.")] = vendor.getProperty(key)
            }
            for (file in files) {
                metadata["sha256.$file"] = sha256(bundleRoot / file)
            }
            val manifest = bundleRoot / "native-bundle.properties"
            manifest.writeText(metadata.entries.joinToString("") { "${it.key}=${it.value}\n" })
            sha256(manifest)
        } else {
            sha256(targetFile)
        }

        (stageDir.get().asFile / "pytorch.properties").text =
                "jni_version=$publishedVersion\n" +
                        "jni_cache_key=$djlVersion-$contentHash\n"
    }
}

tasks {
    jar {
        dependsOn(stageJniLib)
        from(stageJniLib.map { it.outputs.files }) {
            into("jnilib")
        }
        isPreserveFileTimestamps = false
        isReproducibleFileOrder = true
        archiveBaseName = artifactSlug
        archiveClassifier = ""
    }

    clean {
        val injected = project.objects.newInstance<InjectedOps>()
        doFirst {
            injected.fs.delete { delete("jnilib") }
        }
    }
}

publishing {
    publications {
        named<MavenPublication>("maven") {
            artifactId = artifactSlug
            pom {
                name = "DJL PyTorch JNI ($flavor / $classifier)"
                description =
                        "DJL PyTorch JNI ($jniLibFileName) for the $flavor libtorch flavor on" +
                                " $classifier. Supported Linux ROCm flavors include the patched hipBLASLt host" +
                                " library and require the matching ROCm SDK. LibTorch is resolved" +
                                " separately at runtime."
                url = "http://www.djl.ai/engines/pytorch/${project.name}"
            }
        }
    }
}

interface InjectedOps {
    @get:Inject
    val fs: FileSystemOperations

    @get:Inject
    val exec: ExecOperations
}
